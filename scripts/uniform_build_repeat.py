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


def snapshot():
    commands={
        'device':['nvidia-smi','--query-gpu=uuid,name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used','--format=csv,noheader'],
        'processes':['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']}
    out={}
    for key,command in commands.items():
        done=subprocess.run(command,capture_output=True,text=True,timeout=15,check=True)
        out[key]=done.stdout.strip()
    return out


def ensure_idle(state):
    foreign=[]
    for row in state['processes'].splitlines():
        pid=row.split(',')[0].strip()
        if pid.isdigit() and int(pid)!=os.getpid():
            foreign.append(row)
    if foreign:
        raise RuntimeError(f'Other CUDA processes active; do not kill automatically: {foreign}')


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


def prepare(repo,root):
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
        before=snapshot();ensure_idle(before)
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
        contract={'study':STUDY,'environment':env,'settings':SETTINGS,'source_weights_sha256':sha256(source),'engine_metadata':metadata,
            'historical_uniform_calibration_cache_sha256':historical['calibration_cache']['sha256'],
            'onnx_export':{'opset':17,'simplify':True,'dynamic':False,'half':False,'imgsz':640,'batch':1},
            'onnx_packages':onnx_packages,
            'onnx_sha256':sha256(onnx),'calibration_tensor_file_sha256':sha256(root/'calibration_uint8.npy'),
            'calibration_float32_stream_sha256':stream.hexdigest(),'calibration_manifest_sha256':sha256(cal/'calibration_manifest.json'),
            'calibration_order':order,'calibration_source_image_sha256':{name:sha256(cal/'images'/name) for name in names},
            'calibration_loader_source_sha256':digest_bytes(inspect.getsource(Exporter.get_int8_calibration_dataloader).encode()),
            'gpu_before':before,'gpu_after':snapshot(),'git_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True).stdout.strip(),
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


def build(repo,root,index):
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
        before=snapshot();ensure_idle(before);dest.mkdir()
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
        manifest={'study':STUDY,'repeat':index,'engine_sha256':sha256(dest/'model.engine'),'source_weights_sha256':FROZEN_WEIGHTS_SHA256,
            'onnx_sha256':contract['onnx_sha256'],'study_manifest_sha256':sha256(root/'study_manifest.json'),
            'calibration_cache_sha256':sha256(dest/'calibration.cache'),'calibration_mode':'generated_from_frozen_tensor_stream' if index==1 else 'same_study_cache_only',
            'cache_matches_historical_uniform':sha256(dest/'calibration.cache')==contract['historical_uniform_calibration_cache_sha256'],
            'calibration_batches_consumed':cal.position,'calibration_cache_read':cal.cache_read,'timing_cache_sha256':sha256(dest/'timing.cache'),
            'settings':SETTINGS,'builder_flags':int(config.flags),'sigmoid_fp32_constraints':protected,'environment':env,
            'gpu_before':before,'gpu_after':snapshot(),'created_utc':datetime.now(timezone.utc).isoformat(),
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
    records=[];reference=read(repo/'results/measurement_audit_v1/server_fp16_capture_v1/validator_predictions.json')
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
        'status':'step_A_completed_review_required','next_action':'STOP for review; no bbox/classification override or scaling authorized by this runner.',
        'limitations':'Three observed builds do not guarantee determinism. Same calibration table isolates build variation, not calibration-sampling variance. Fresh timing caches intentionally permit tactic retiming. GPU snapshots do not eliminate thermal/clock noise. New explicit workspace baseline is not a perfect replica of historical defaults.'})
    print(json.dumps(stats,indent=2));print(f'DONE: {root / "repeat_summary.json"}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--phase',choices=('all','prepare','build','evaluate'),default='all')
    parser.add_argument('--repeat',type=int,choices=(1,2,3))
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1];root=args.out_dir.resolve()
    if not root.is_relative_to((repo/'results/measurement_audit_v1').resolve()):
        parser.error('Output must be inside results/measurement_audit_v1')
    if args.phase=='build':
        if args.repeat is None:
            parser.error('--repeat required for build phase')
        build(repo,root,args.repeat);return
    if args.phase=='prepare':
        prepare(repo,root);return
    if args.phase=='all':
        if root.exists():
            parser.error('Output exists; do not auto-resume; inspect partial results first')
        subprocess.run([sys.executable,__file__,'--phase','prepare','--out-dir',str(root)],cwd=repo,check=True)
        for i in (1,2,3):
            # Each repeat gets a fresh OS process and an exclusive log, outside repeat dir.
            log=root/f'build_{i}.log'
            print(f'START BUILD {i}/3; verbose log: {log}',flush=True)
            with log.open('x',encoding='utf-8') as f:
                proc=subprocess.run([sys.executable,__file__,'--phase','build','--repeat',str(i),'--out-dir',str(root)],cwd=repo,stdout=f,stderr=subprocess.STDOUT)
            with log.open('rb') as source,gzip.open(root/f'build_{i}.log.gz','xb') as compressed:
                shutil.copyfileobj(source,compressed)
            if proc.returncode:
                raise RuntimeError(f'Build {i} failed; inspect {log}; outputs preserved')
            print(f'FINISHED BUILD {i}/3',flush=True)
    for i in (1,2,3):
        commands=[['capture_cctsdb_validator.py','--repeat-study',str(root),'--repeat-index',str(i),'--out-dir',str(root/f'repeat_{i}/capture')],
            ['verify_cctsdb_capture.py','--capture-dir',str(root/f'repeat_{i}/capture'),'--xml',str((repo/'../nighttime-tsd/data/raw/CCTSDB2021/xml.zip').resolve()),'--out-dir',str(root/f'repeat_{i}/verification')]]
        for script,*options in commands:
            subprocess.run([sys.executable,str(repo/'scripts'/script),*options],cwd=repo,check=True)
    summarize(repo,root)


if __name__=='__main__':
    main()
