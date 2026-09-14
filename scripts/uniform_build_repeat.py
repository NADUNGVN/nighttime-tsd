#!/usr/bin/env python3
"""Step A only: three independent Uniform builds with one frozen ONNX/calibration table."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_cctsdb_measurement import sha256
from capture_cctsdb_validator import FROZEN_WEIGHTS_SHA256, write_json
from run_architecture_matrix import GpuPhaseLock

STUDY = 'uniform_build_repeat_v1'
SETTINGS = {'imgsz':640,'batch':1,'workspace_bytes':4 << 30,'builder_optimization_level':3,
            'avg_timing_iterations':1,'timing_cache':'fresh_empty_for_each_repeat','int8':True,'fp16':False,
            'tf32':False,'sigmoid':'FP32 precision+output with OBEY; no bbox/classification override'}
GPU_PROCESS_GUARD_VERSION = 'operator-confirmed-nvidia-smi-path-v1'
DESKTOP_ALLOWLIST = {
    'snapd-desktop-integration': '/snap/snapd-desktop-integration/<numeric-revision>/usr/bin/snapd-desktop-integration',
    'xorg': '/usr/lib/xorg/Xorg or /usr/bin/Xorg',
    'gnome-shell': '/usr/bin/gnome-shell or /usr/libexec/gnome-shell',
    'gnome-session': '/usr/bin/gnome-session-binary or /usr/libexec/gnome-session-binary',
}
_DESKTOP_EXECUTABLES = (
    ('snapd-desktop-integration', re.compile(r'^/snap/snapd-desktop-integration/[1-9][0-9]*/usr/bin/snapd-desktop-integration$')),
    ('xorg', re.compile(r'^/(?:usr/lib/xorg/Xorg|usr/bin/Xorg)$')),
    ('gnome-shell', re.compile(r'^/(?:usr/bin|usr/libexec)/gnome-shell$')),
    ('gnome-session', re.compile(r'^/(?:usr/bin|usr/libexec)/gnome-session-binary$')),
)


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_bytes(path,data):
    with path.open('xb') as f:
        f.write(data)


def onnx_dependencies():
    """Distribution names differ from the shared onnxruntime import module."""
    try:
        packages={p:importlib.metadata.version(p) for p in ('onnx','onnxslim')}
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f'Missing ONNX export dependency: {error}. Inspect the environment first; no automatic installation.') from error
    variants={}
    for name in ('onnxruntime','onnxruntime-gpu','onnxruntime-qnn'):
        try:
            variants[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    if not variants:
        raise RuntimeError('No ONNX Runtime distribution found. Stop before export; inspect g0_size_env with python -m pip show onnxruntime onnxruntime-gpu onnxruntime-qnn. Do not auto-install or upgrade.')
    try:
        ort=importlib.import_module('onnxruntime')
        providers=list(ort.get_available_providers())
    except Exception as error:
        raise RuntimeError(f'ONNX Runtime is installed ({variants}) but cannot import/query providers; do not rebuild or auto-upgrade: {error}') from error
    packages.update(variants)
    return {'distributions':packages,'runtime_import_version':ort.__version__,
            'runtime_import_path':ort.__file__,'available_providers':providers,
            'multiple_runtime_distributions':len(variants)>1}


def environment():
    import torch
    import ultralytics
    import tensorrt as trt
    versions={'torch':torch.__version__,'ultralytics':ultralytics.__version__,'tensorrt':trt.__version__,'numpy':np.__version__}
    expected={'torch':'2.5.1+cu121','ultralytics':'8.4.102','tensorrt':'10.16.1.11','numpy':'2.4.4'}
    if versions != expected or not torch.cuda.is_available():
        raise ValueError(f'Use existing nighttime-tsd/g0_size_env without upgrades; expected {expected}, got {versions}')
    versions.update(python=sys.version,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                    pycocotools=importlib.metadata.version('pycocotools'))
    if versions['pycocotools']!='2.0.10':
        raise ValueError('Use g0_size_env with pycocotools 2.0.10')
    return versions


def desktop_allowlist_id(path):
    """Return the narrow desktop allowlist entry matching a reported path."""
    if not isinstance(path, str):
        return None
    for allowlist_id, pattern in _DESKTOP_EXECUTABLES:
        if pattern.fullmatch(path) is not None:
            return allowlist_id
    return None


def is_allowed_desktop_executable(path):
    return desktop_allowlist_id(path) is not None


def is_allowed_snap_desktop_executable(path):
    """Backward-compatible check for the original Snap exception."""
    return desktop_allowlist_id(path) == 'snapd-desktop-integration'


def parse_desktop_confirmations(values):
    """Parse explicit current-server operator confirmations as PID=reported-path."""
    confirmations = {}
    for value in values or ():
        if not isinstance(value, str):
            raise ValueError('Desktop confirmation must be PID=PATH')
        pid_text, separator, path = value.partition('=')
        if not separator or not pid_text.isdigit() or not path:
            raise ValueError(f'Invalid desktop confirmation {value!r}; expected PID=PATH')
        pid = int(pid_text)
        if pid <= 0 or pid in confirmations:
            raise ValueError(f'Duplicate or invalid desktop confirmation PID: {pid_text!r}')
        if not is_allowed_desktop_executable(path):
            raise ValueError(f'Desktop confirmation path is outside the allowlist: {path!r}')
        confirmations[pid] = path
    return confirmations


def parse_background_confirmations(values):
    """Parse explicit PID=COMMAND confirmations for the bounded concurrent variant."""
    confirmations = {}
    for value in values or ():
        if not isinstance(value, str):
            raise ValueError('Background workload confirmation must be PID=COMMAND')
        pid_text, separator, command = value.partition('=')
        command = ' '.join(command.split())
        if not separator or not pid_text.isdigit() or not command:
            raise ValueError(f'Invalid background confirmation {value!r}; expected PID=COMMAND')
        pid = int(pid_text)
        if pid <= 0 or pid in confirmations:
            raise ValueError(f'Duplicate or invalid background confirmation PID: {pid_text!r}')
        if any(token in command for token in ('*', '?', '[', ']')):
            raise ValueError('Background workload command must be an exact command, not a pattern')
        confirmations[pid] = command
    return confirmations


def inspect_background_processes(confirmations):
    """Check confirmed commands with ps; do not read /proc or wait for a PID."""
    checks = {}
    for pid, expected_command in sorted((confirmations or {}).items()):
        try:
            done = subprocess.run(['ps', '-o', 'args=', '-p', str(pid)], capture_output=True,
                                  text=True, timeout=15, check=False)
        except OSError as error:
            checks[pid] = {'status': 'unverifiable', 'expected_command': expected_command,
                           'observed_command': None, 'error': str(error)}
            continue
        observed_command = ' '.join((done.stdout or '').split())
        if not observed_command and done.returncode in (0, 1):
            checks[pid] = {'status': 'exited', 'expected_command': expected_command,
                           'observed_command': None}
        elif observed_command == expected_command:
            checks[pid] = {'status': 'running', 'expected_command': expected_command,
                           'observed_command': observed_command}
        else:
            checks[pid] = {'status': 'command_mismatch', 'expected_command': expected_command,
                           'observed_command': observed_command or None}
    return checks


def _unverifiable_process(raw, reason):
    return {'pid': None, 'process_name': None, 'used_gpu_memory': None, 'raw': raw,
            'resolved_executable': None, 'classification': 'blocked_unverifiable',
            'allowed': False, 'allowlist_exception': None, 'reason': reason}


def classify_cuda_processes(raw, current_pid=None, confirmed_desktop=None,
                            confirmed_background=None, background_checks=None):
    """Classify nvidia-smi rows using exact current rows and operator confirmation."""
    if current_pid is None:
        current_pid = os.getpid()
    confirmed_desktop = confirmed_desktop or {}
    confirmed_background = confirmed_background or {}
    background_checks = background_checks or {}
    if raw is None:
        return [_unverifiable_process('', 'nvidia-smi returned no process text')]
    if not isinstance(raw, str):
        return [_unverifiable_process(repr(raw), 'nvidia-smi process output is not text')]
    if raw.strip() in ('', 'No running processes found'):
        return []
    details = []
    for row in raw.splitlines():
        if not row.strip():
            continue
        fields = [field.strip() for field in row.split(',', 2)]
        if len(fields) != 3 or not fields[0].isdigit():
            details.append(_unverifiable_process(row, f'cannot parse nvidia-smi process row: {row!r}'))
            continue
        pid = int(fields[0])
        process_name, used_gpu_memory = fields[1:]
        detail = {'pid': pid, 'process_name': process_name, 'used_gpu_memory': used_gpu_memory,
                  'raw': row, 'resolved_executable': None, 'classification': None,
                  'allowed': False, 'allowlist_exception': None, 'reason': None}
        if pid == current_pid:
            detail.update(classification='current_runner_process', allowed=True,
                          reason='current process owned by this runner')
            details.append(detail)
            continue
        detail['resolved_executable'] = process_name if process_name.startswith('/') else None
        allowlist_id = desktop_allowlist_id(process_name)
        if pid in confirmed_background:
            check = background_checks.get(pid, {})
            detail.update(expected_command=confirmed_background[pid],
                          observed_command=check.get('observed_command'))
            if check.get('status') == 'running':
                detail.update(classification='allowed_background_operator_confirmed', allowed=True,
                              operator_confirmed=True,
                              reason='exact current ps PID/command and nvidia-smi PID row were explicitly confirmed by operator')
            else:
                detail.update(classification='blocked_background_command_unverified',
                              reason='confirmed background PID/command is no longer an exact current process match')
        elif allowlist_id is not None and confirmed_desktop.get(pid) == process_name:
            detail.update(classification='allowed_desktop_operator_confirmed', allowed=True,
                          allowlist_exception=allowlist_id, desktop_allowlist_id=allowlist_id,
                          operator_confirmed=True,
                          reason='exact current nvidia-smi PID/path row was explicitly confirmed by operator; /proc was not read')
        elif allowlist_id is not None:
            detail.update(classification='blocked_desktop_unconfirmed',
                          desktop_allowlist_id=allowlist_id,
                          reason='desktop path matches allowlist but current PID/path lacks explicit operator confirmation')
        elif process_name in ('', '[Not Found]'):
            detail.update(classification='blocked_unverifiable_process',
                          reason='nvidia-smi did not provide a usable process path')
        else:
            detail.update(classification='blocked_non_allowlisted_process',
                          reason='reported nvidia-smi process path is not in the Snap desktop allowlist')
        details.append(detail)
    return details


def snapshot(confirmed_desktop=None, confirmed_background=None):
    commands={
        'device':['nvidia-smi','--query-gpu=uuid,name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used','--format=csv,noheader'],
        'processes':['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']}
    out={}
    for key,command in commands.items():
        done=subprocess.run(command,capture_output=True,text=True,timeout=15,check=True)
        out[key]=done.stdout.strip() if isinstance(done.stdout,str) else done.stdout
    confirmed_desktop = confirmed_desktop or {}
    confirmed_background = confirmed_background or {}
    background_checks = inspect_background_processes(confirmed_background)
    out['process_details']=classify_cuda_processes(
        out['processes'], confirmed_desktop=confirmed_desktop,
        confirmed_background=confirmed_background, background_checks=background_checks)
    observed_pids={detail['pid'] for detail in out['process_details'] if isinstance(detail,dict) and isinstance(detail.get('pid'),int)}
    unmatched=[{'pid':pid,'reported_path':path} for pid,path in sorted(confirmed_desktop.items()) if pid not in observed_pids]
    blocked=[detail for detail in out['process_details'] if not detail.get('allowed',False)]
    background_workload=[]
    for pid, check in sorted(background_checks.items()):
        observed_on_gpu=pid in observed_pids
        status=check['status']
        if status == 'running' and not observed_on_gpu:
            status='present_not_observed_on_gpu'
        blocking=status in ('command_mismatch', 'unverifiable')
        if blocking and not observed_on_gpu:
            blocked.append({'pid':pid, 'process_name':None, 'used_gpu_memory':None,
                            'raw':None, 'resolved_executable':None,
                            'classification':'blocked_background_command_unverified',
                            'allowed':False, 'expected_command':check['expected_command'],
                            'observed_command':check.get('observed_command'),
                            'reason':'confirmed background PID/command is not an exact current process match'})
        background_workload.append({
            'pid':pid, 'expected_command':check['expected_command'],
            'observed_command':check.get('observed_command'), 'status':status,
            'observed_on_gpu':observed_on_gpu,
            'authorized':status == 'running' and observed_on_gpu,
            'verification_method':'ps -o args= PID plus exact PID presence in this nvidia-smi snapshot; /proc/<pid>/exe was not read',
            'operator_confirmed':status == 'running' and observed_on_gpu,
            'blocking':blocking,
        })
    authorized_background=any(item['authorized'] for item in background_workload)
    out['process_guard']={'version':GPU_PROCESS_GUARD_VERSION,
                          'allowlist':DESKTOP_ALLOWLIST,
                          'verification_method':'exact PID/path comparison against this snapshot nvidia-smi output plus explicit operator CLI confirmation; /proc/<pid>/exe was not read',
                          'operator_confirmation':{'method':'--confirm-desktop-process PID=PATH','confirmed_processes':[{'pid':pid,'reported_path':path} for pid,path in sorted(confirmed_desktop.items())]},
                          'background_operator_confirmation':{'method':'--confirm-background-process PID=COMMAND',
                              'confirmed_processes':[{'pid':pid,'expected_command':command} for pid,command in sorted(confirmed_background.items())]},
                          'background_workload':background_workload,
                          'unmatched_confirmations':unmatched,
                          'blocked_processes':blocked,
                          'telemetry_status':'complete' if all(isinstance(out.get(key),str) for key in ('device','processes')) else 'limited',
                          'telemetry_limitation':'nvidia-smi output is the available workload telemetry; it cannot prove zero interference between snapshots',
                          'external_workload_detected':bool(blocked or unmatched or authorized_background),
                          'external_workload_authorized':authorized_background and not bool(blocked or unmatched),
                          'disclaimer':'Allowed desktop processes remain visible; this is not proof of zero GPU interference.'}
    return out


def ensure_idle(state, confirmed_background=None):
    details=state.get('process_details')
    if details is None:
        details=classify_cuda_processes(state.get('processes'), current_pid=os.getpid())
    if not isinstance(details,list):
        raise RuntimeError('Other CUDA processes active or unverifiable; process classification is invalid')
    def allowed(detail):
        if not isinstance(detail,dict) or detail.get('allowed') is not True:
            return False
        if detail.get('classification')=='current_runner_process':
            return detail.get('pid')==os.getpid()
        if detail.get('classification')=='allowed_desktop_operator_confirmed':
            return (detail.get('desktop_allowlist_id')==detail.get('allowlist_exception') and
                    detail.get('operator_confirmed') is True and
                    is_allowed_desktop_executable(detail.get('resolved_executable')))
        if detail.get('classification')=='allowed_background_operator_confirmed':
            return detail.get('operator_confirmed') is True
        return False
    blocked=[detail for detail in details if not allowed(detail)]
    guard=state.get('process_guard',{})
    unmatched=guard.get('unmatched_confirmations',[])
    if unmatched:
        blocked.extend(unmatched)
    blocked.extend(item for item in guard.get('background_workload', []) if item.get('blocking'))
    if blocked:
        raise RuntimeError(f'Other CUDA processes active or unverifiable; do not kill automatically: {blocked}')


def confirmation_cli_args(confirmations):
    return [item for pid,path in sorted(confirmations.items()) for item in ('--confirm-desktop-process',f'{pid}={path}')]


def validate_selection(repo,manifest):
    if (manifest.get('strategy'),manifest.get('seed'),manifest.get('selected_size'),manifest.get('candidate_size'))!=('uniform',42,1024,14720):
        raise ValueError('Requires existing Uniform-v2 seed42 n1024 train-only manifest')
    names=[]
    base=repo/'data/processed/cctsdb2021_clean'
    for item in manifest['files']:
        rel=Path(item['source_image'])
        if len(rel.parts)!=3 or rel.parts[:2]!=('train','images') or '..' in rel.parts:
            raise ValueError('Calibration path is not strictly train/images/basename')
        if not (base/rel).is_file():
            raise FileNotFoundError(base/rel)
        names.append(rel.name)
    if len(names)!=1024 or len(set(names))!=1024:
        raise ValueError('Calibration IDs missing/duplicated')
    return sorted(names)


def prepare(repo,root,confirmed_desktop):
    import torch
    from ultralytics import YOLO
    from ultralytics.engine.exporter import Exporter
    env=environment()
    # Check evaluation inputs/dependencies before spending time on builds.
    from run_g0_int8_capture import check_reference
    check_reference(repo/'results/measurement_audit_v1/server_fp16_capture_v1',
                    repo/'results/measurement_audit_v1/server_native_size_v1',
                    (repo/'../nighttime-tsd/data/raw/CCTSDB2021/xml.zip').resolve())
    onnx_packages=onnx_dependencies()
    if root.exists():
        raise FileExistsError('Study directory exists: preserve it; do not auto-skip/rebuild')
    source=repo/'results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt'
    if sha256(source)!=FROZEN_WEIGHTS_SHA256:
        raise ValueError('Frozen source weight hash mismatch')
    cal=repo/'data/processed/cctsdb2021_clean/calibration/uniform_v2_s42_n1024'
    manifest=read(cal/'calibration_manifest.json');names=validate_selection(repo,manifest)
    historical=read(repo/'results/calibration_method_v2/rtx8000/yolo11n/engines/yolo11n_int8_uniform_s42.engine.provenance.json')
    if sha256(cal/'calibration_manifest.json')!=historical['calibration_manifest_sha256']:
        raise ValueError('Uniform calibration manifest differs from historical seed42')
    for split in ('dev','test'):
        excluded={p.name for p in (repo/f'data/processed/cctsdb2021_clean/{split}/images').iterdir() if p.is_file()}
        if set(names)&excluded:
            raise ValueError('Calibration IDs overlap excluded split')
    for name in names:
        if sha256(cal/'images'/name)!=sha256(repo/'data/processed/cctsdb2021_clean/train/images'/name):
            raise ValueError(f'Calibration image differs from train: {name}')
    with GpuPhaseLock(repo/'results/architecture_matrix_v1/.gpu_phase.lock',STUDY+'_prepare'):
        before=snapshot(confirmed_desktop);ensure_idle(before)
        root.mkdir(parents=True)
        shutil.copy2(source,root/'source.pt')
        # Exactly one ONNX export, no INT8/engine exporter. Keep all source artifacts.
        model=YOLO(str(root/'source.pt'))
        if list(model.names.values())!=['prohibitory','mandatory','warning']:
            raise ValueError('Source classes differ from locked CCTSDB model')
        metadata={'task':'detect','batch':1,'imgsz':[640,640],'stride':int(model.model.stride.max()),
                  'names':model.names,'channels':3,'end2end':False}
        exported=model.export(format='onnx',imgsz=640,batch=1,dynamic=False,half=False,opset=17,simplify=True,device=0)
        onnx=Path(exported)
        if onnx.resolve()!=(root/'source.onnx').resolve():
            raise ValueError('Unexpected ONNX output path')
        import onnx as ox
        graph=ox.load(str(onnx))
        if any(n.op_type in ('QuantizeLinear','DequantizeLinear') for n in graph.graph.node):
            raise ValueError('Step A requires same unquantized ONNX, not Q/DQ')
        exp=Exporter(overrides={'format':'engine','data':str(cal/'calibration.yaml'),'imgsz':640,'batch':1,'fraction':1.,'split':'val','rect':False,'device':'0'})
        exp.imgsz=(640,640);exp.model=SimpleNamespace(task='detect')
        torch.manual_seed(42)
        loader=exp.get_int8_calibration_dataloader()
        pool=np.lib.format.open_memmap(root/'calibration_uint8.npy',mode='w+',dtype=np.uint8,shape=(1024,3,640,640))
        order=[];stream=hashlib.sha256()
        for i,batch in enumerate(loader):
            if i>=1024 or tuple(batch['img'].shape)!=(1,3,640,640) or batch['img'].dtype!=torch.uint8:
                raise ValueError('Calibration dataloader shape/dtype/count mismatch')
            name=Path(batch['im_file'][0]).name
            image=batch['img'].cpu().numpy()[0]
            pool[i]=image;stream.update((image.astype(np.float32)/255.).tobytes())
            order.append(name)
            if (i+1)%256==0:
                print(f'FROZEN CALIBRATION {i+1}/1024',flush=True)
        pool.flush();del pool
        if sorted(order)!=names:
            raise ValueError('Actual dataloader image IDs differ from manifest')
        after=snapshot(confirmed_desktop)
        contract={'study':STUDY,'environment':env,'settings':SETTINGS,'source_weights_sha256':sha256(source),'engine_metadata':metadata,
            'historical_uniform_calibration_cache_sha256':historical['calibration_cache']['sha256'],
            'onnx_export':{'opset':17,'simplify':True,'dynamic':False,'half':False,'imgsz':640,'batch':1},
            'onnx_packages':onnx_packages,
            'onnx_sha256':sha256(onnx),'calibration_tensor_file_sha256':sha256(root/'calibration_uint8.npy'),
            'calibration_float32_stream_sha256':stream.hexdigest(),'calibration_manifest_sha256':sha256(cal/'calibration_manifest.json'),
            'calibration_order':order,'calibration_source_image_sha256':{name:sha256(cal/'images'/name) for name in names},
            'calibration_loader_source_sha256':digest_bytes(inspect.getsource(Exporter.get_int8_calibration_dataloader).encode()),
            'gpu_before':before,'gpu_after':after,'git_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True).stdout.strip(),
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Step A new controlled baseline, not a byte-identical recreation of historical auto-workspace build. No changed calibration IDs, no bbox/classification protection, no test, no retraining. Repeat 1 generates one private cache; repeats 2/3 use that exact table. Timing cache starts empty in every build to measure tactic-selection variability.'}
        write_json(root/'study_manifest.json',contract)


def load_contract(root):
    c=read(root/'study_manifest.json')
    if c['study']!=STUDY or c['settings']!=SETTINGS or c['source_weights_sha256']!=FROZEN_WEIGHTS_SHA256:
        raise ValueError('Invalid repeat study contract')
    if sha256(root/'source.onnx')!=c['onnx_sha256'] or sha256(root/'calibration_uint8.npy')!=c['calibration_tensor_file_sha256']:
        raise ValueError('Frozen ONNX/calibration tensors changed')
    return c


def inspector_signature(info):
    """Inspectable plan fingerprint, not proof of arithmetic precision."""
    layers=info.get('Layers',[])
    if not layers or not all(isinstance(l,dict) for l in layers):
        raise ValueError('Detailed layer inspector data required')
    signature=[{k:l.get(k) for k in ('Name','LayerType','Inputs','Outputs','Weights','TacticName')} for l in layers]
    counts={}
    for layer in layers:
        if layer.get('ParameterType')=='Convolution':
            kind=layer.get('Weights',{}).get('Type','unknown')
            counts[kind]=counts.get(kind,0)+1
    return {'layers':len(layers),'conv_weight_types':counts,
            'plan_signature_sha256':digest_bytes(json.dumps(signature,sort_keys=True).encode())}


def build(repo,root,index,confirmed_desktop):
    import torch
    import tensorrt as trt
    env=environment();contract=load_contract(root)
    if env!=contract['environment']:
        raise ValueError('Build environment changed')
    dest=root/f'repeat_{index}'
    if dest.exists():
        raise FileExistsError(dest)
    cache=root/'repeat_1/calibration.cache'
    if index>1:
        first=read(root/'repeat_1/build_manifest.json')
        if sha256(cache)!=first['calibration_cache_sha256']:
            raise ValueError('Calibration cache changed since first build')
    class Calibrator(trt.IInt8Calibrator):
        def __init__(self):
            super().__init__();self.position=0;self.stream=hashlib.sha256();self.tensor=None
            self.pool=np.load(root/'calibration_uint8.npy',mmap_mode='r') if index==1 else None
            self.cache_read=False
        def get_algorithm(self):
            return trt.CalibrationAlgoType.MINMAX_CALIBRATION
        def get_batch_size(self):
            return 1
        def read_calibration_cache(self):
            if index==1:
                return None
            self.cache_read=True
            return cache.read_bytes()
        def write_calibration_cache(self,data):
            blob=bytes(data)
            target=dest/'calibration.cache'
            if target.exists():
                if target.read_bytes()!=blob:
                    raise ValueError('Calibration table unexpectedly changed')
            else:
                write_bytes(target,blob)
        def get_batch(self,names):
            if index>1:
                raise RuntimeError('Cache-only repeat attempted recalibration')
            if self.position==1024:
                return None
            a=np.ascontiguousarray(self.pool[self.position:self.position+1],dtype=np.float32)/255.
            self.stream.update(a.tobytes());self.tensor=torch.from_numpy(a).to('cuda:0');self.position+=1
            return [self.tensor.data_ptr()]
    with GpuPhaseLock(repo/'results/architecture_matrix_v1/.gpu_phase.lock',STUDY+f'_build_{index}'):
        before=snapshot(confirmed_desktop);ensure_idle(before);dest.mkdir()
        logger=trt.Logger(trt.Logger.VERBOSE);builder=trt.Builder(logger);network=builder.create_network(0)
        parser=trt.OnnxParser(network,logger)
        if not parser.parse_from_file(str(root/'source.onnx')):
            raise ValueError([str(parser.get_error(i)) for i in range(parser.num_errors)])
        config=builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,SETTINGS['workspace_bytes'])
        config.builder_optimization_level=SETTINGS['builder_optimization_level'];config.avg_timing_iterations=SETTINGS['avg_timing_iterations']
        config.set_flag(trt.BuilderFlag.INT8);config.clear_flag(trt.BuilderFlag.FP16);config.clear_flag(trt.BuilderFlag.TF32)
        config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        timing=config.create_timing_cache(b'')
        if not config.set_timing_cache(timing,False):
            raise RuntimeError('Cannot initialize independent timing cache')
        cal=Calibrator();config.int8_calibrator=cal
        protected=[]
        for i in range(network.num_layers):
            layer=network.get_layer(i)
            if layer.type==trt.LayerType.ACTIVATION and 'sigmoid' in layer.name.lower():
                layer.precision=trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j,trt.float32)
                protected.append(layer.name)
        if not protected or network.num_inputs!=1 or tuple(network.get_input(0).shape)!=(1,3,640,640):
            raise ValueError('Unexpected network/Sigmoid constraints')
        plan=builder.build_serialized_network(network,config)
        if plan is None:
            raise RuntimeError('TensorRT build failed; see build.log')
        if index==1 and (cal.position!=1024 or cal.stream.hexdigest()!=contract['calibration_float32_stream_sha256']):
            raise ValueError('First build did not consume exactly frozen calibration stream')
        if index>1:
            if not cal.cache_read or cal.position!=0:
                raise ValueError('Repeat must use identical cache without calibration')
            if not (dest/'calibration.cache').exists():
                shutil.copy2(cache,dest/'calibration.cache')
            if sha256(dest/'calibration.cache')!=sha256(cache):
                raise ValueError('Cache differs between repeats')
        write_bytes(dest/'timing.cache',bytes(config.get_timing_cache().serialize()))
        metadata=json.dumps(contract['engine_metadata']).encode('utf-8')
        write_bytes(dest/'model.engine',len(metadata).to_bytes(4,'little')+metadata+bytes(plan))
        with trt.Runtime(logger) as runtime:
            engine=runtime.deserialize_cuda_engine(plan)
            if engine is None:
                raise RuntimeError('New engine failed deserialization')
            inspector=engine.create_engine_inspector()
            info=json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
            write_json(dest/'inspector.json',info)
            del inspector,engine
        after=snapshot(confirmed_desktop)
        manifest={'study':STUDY,'repeat':index,'engine_sha256':sha256(dest/'model.engine'),'source_weights_sha256':FROZEN_WEIGHTS_SHA256,
            'onnx_sha256':contract['onnx_sha256'],'study_manifest_sha256':sha256(root/'study_manifest.json'),
            'calibration_cache_sha256':sha256(dest/'calibration.cache'),'calibration_mode':'generated_from_frozen_tensor_stream' if index==1 else 'same_study_cache_only',
            'cache_matches_historical_uniform':sha256(dest/'calibration.cache')==contract['historical_uniform_calibration_cache_sha256'],
            'calibration_batches_consumed':cal.position,'calibration_cache_read':cal.cache_read,'timing_cache_sha256':sha256(dest/'timing.cache'),
            'settings':SETTINGS,'builder_flags':int(config.flags),'sigmoid_fp32_constraints':protected,'environment':env,
            'gpu_before':before,'gpu_after':after,'created_utc':datetime.now(timezone.utc).isoformat(),
            'inspector_sha256':sha256(dest/'inspector.json'),'script_sha256':sha256(Path(__file__))}
        write_json(dest/'build_manifest.json',manifest)
    print(f'DONE BUILD {index}: {dest}',flush=True)


def repeat_capture_inputs(repo,root,index):
    # No arbitrary engine path or test dataset accepted by capture CLI.
    root=root.resolve()
    if not root.is_relative_to((repo/'results/measurement_audit_v1').resolve()) or index not in (1,2,3):
        raise ValueError('Repeat study must be inside scoped audit directory')
    contract=load_contract(root);p=root/f'repeat_{index}';m=read(p/'build_manifest.json');engine=p/'model.engine'
    if m['study']!=STUDY or m['repeat']!=index or m['study_manifest_sha256']!=sha256(root/'study_manifest.json') or m['settings']!=SETTINGS:
        raise ValueError('Repeat build provenance mismatch')
    if m['onnx_sha256']!=contract['onnx_sha256'] or sha256(engine)!=m['engine_sha256']:
        raise ValueError('Repeat engine/ONNX identity mismatch')
    previous=repo/'results/calibration_method_v2/rtx8000/yolo11n/dev_eval/yolo11n_int8_uniform_s42.json'
    return engine,p/'build_manifest.json',previous,read(previous),{'source_weights_sha256':FROZEN_WEIGHTS_SHA256,'environment':{'tensorrt_python':m['environment']['tensorrt']}},m['engine_sha256']


def summarize(repo,root):
    from run_g0_int8_capture import check_same_targets
    records=[];review_flags=[];reference=read(repo/'results/measurement_audit_v1/server_fp16_capture_v1/validator_predictions.json')
    for i in (1,2,3):
        p=root/f'repeat_{i}';b=read(p/'build_manifest.json');c=read(p/'capture/capture_report.json');v=read(p/'verification/verification_summary.json')
        if c['status']!='pass' or v['native_matching_status']!='pass' or c['model_sha256']!=b['engine_sha256']:
            raise ValueError('Unverified repeat')
        check_same_targets(reference,read(p/'capture/validator_predictions.json'))
        sizes=read(p/'verification/size_coco_xml.json')
        if sha256(p/'capture/validator_predictions.json')!=c['predictions_sha256'] or c['predictions_sha256']!=v['capture_prediction_sha256']:
            raise ValueError('Capture/verification prediction identity mismatch')
        if sha256(p/'inspector.json')!=b['inspector_sha256']:
            raise ValueError('Inspector changed')
        for label,state in (('build_before',b.get('gpu_before')),('build_after',b.get('gpu_after')),('capture_before',c.get('gpu_before')),('capture_after',c.get('gpu_after'))):
            if state is None:
                review_flags.append(f'repeat_{i}_{label}_telemetry_missing')
            elif state.get('process_guard',{}).get('external_workload_detected'):
                review_flags.append(f'repeat_{i}_{label}_external_gpu_workload_or_unverified_process')
        records.append({'repeat':i,'engine_sha256':b['engine_sha256'],'calibration_cache_sha256':b['calibration_cache_sha256'],
            'inspector':inspector_signature(read(p/'inspector.json')),
            'ultralytics':c['metrics'],'coco_xml':sizes['metrics'],'gpu_before':b['gpu_before'],'gpu_after':b['gpu_after']})
    if len({r['calibration_cache_sha256'] for r in records})!=1:
        raise ValueError('Calibration scales differ: cannot interpret as build-only variability')
    stats={}
    for size in ('all','xs','s','m','l','xl'):
        stats[size]={}
        for metric in ('map50','map50_95'):
            a=np.array([r['coco_xml'][size][metric] for r in records])
            stats[size][metric]={'mean':float(a.mean()),'sample_std':float(a.std(ddof=1)),'min':float(a.min()),'max':float(a.max()),'range_pp':float(100*np.ptp(a))}
    write_json(root/'repeat_summary.json',{'study':STUDY,'repeats':records,'build_variability_coco_xml':stats,
        'gpu_protocol_status':'review_required_external_workload_or_limited_telemetry' if review_flags else 'operator_confirmed_shared_lab_telemetry_recorded',
        'review_flags':sorted(set(review_flags)),
        'status':'step_A_completed_review_required','next_action':'STOP for review; no bbox/classification override or scaling authorized by this runner.',
        'limitations':'Shared lab server may retain operator-confirmed desktop GPU processes. Process confirmation matches exact current nvidia-smi PID/path rows and does not read /proc; it is not proof of GPU isolation. Telemetry is sampled before/after phases and cannot rule out workload between snapshots. Three observed builds do not guarantee determinism. Same calibration table isolates build variation, not calibration-sampling variance. Fresh timing caches intentionally permit tactic retiming. New explicit workspace baseline is not a perfect replica of historical defaults.'})
    print(json.dumps(stats,indent=2));print(f'DONE: {root / "repeat_summary.json"}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--phase',choices=('all','prepare','build','evaluate'),default='all')
    parser.add_argument('--repeat',type=int,choices=(1,2,3))
    parser.add_argument('--confirm-desktop-process',action='append',default=[],metavar='PID=PATH',
                        help='Explicitly confirm a current nvidia-smi desktop row; no /proc access is used')
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1];root=args.out_dir.resolve()
    confirmed_desktop=parse_desktop_confirmations(args.confirm_desktop_process)
    if not root.is_relative_to((repo/'results/measurement_audit_v1').resolve()):
        parser.error('Output must be inside results/measurement_audit_v1')
    if args.phase=='build':
        if args.repeat is None:
            parser.error('--repeat required for build phase')
        build(repo,root,args.repeat,confirmed_desktop);return
    if args.phase=='prepare':
        prepare(repo,root,confirmed_desktop);return
    if args.phase=='all':
        if root.exists():
            parser.error('Output exists; do not auto-resume; inspect partial results first')
        subprocess.run([sys.executable,__file__,'--phase','prepare','--out-dir',str(root),*confirmation_cli_args(confirmed_desktop)],cwd=repo,check=True)
        for i in (1,2,3):
            # Each repeat gets a fresh OS process and an exclusive log, outside repeat dir.
            log=root/f'build_{i}.log'
            print(f'START BUILD {i}/3; verbose log: {log}',flush=True)
            with log.open('x',encoding='utf-8') as f:
                proc=subprocess.run([sys.executable,__file__,'--phase','build','--repeat',str(i),'--out-dir',str(root),*confirmation_cli_args(confirmed_desktop)],cwd=repo,stdout=f,stderr=subprocess.STDOUT)
            with log.open('rb') as source,gzip.open(root/f'build_{i}.log.gz','xb') as compressed:
                shutil.copyfileobj(source,compressed)
            if proc.returncode:
                raise RuntimeError(f'Build {i} failed; inspect {log}; outputs preserved')
            print(f'FINISHED BUILD {i}/3',flush=True)
    for i in (1,2,3):
        commands=[['capture_cctsdb_validator.py','--repeat-study',str(root),'--repeat-index',str(i),'--out-dir',str(root/f'repeat_{i}/capture'),*confirmation_cli_args(confirmed_desktop)],
            ['verify_cctsdb_capture.py','--capture-dir',str(root/f'repeat_{i}/capture'),'--xml',str((repo/'../nighttime-tsd/data/raw/CCTSDB2021/xml.zip').resolve()),'--out-dir',str(root/f'repeat_{i}/verification')]]
        for script,*options in commands:
            subprocess.run([sys.executable,str(repo/'scripts'/script),*options],cwd=repo,check=True)
    summarize(repo,root)


if __name__=='__main__':
    main()
