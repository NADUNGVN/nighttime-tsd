#!/usr/bin/env python3
"""Read existing engine inspector metadata; never build, calibrate or enqueue inference."""
import argparse
import json
import subprocess
from pathlib import Path

from capture_cctsdb_validator import capture_inputs, write_json
from run_architecture_matrix import GpuPhaseLock


def unwrap_engine(data):
    """Ultralytics prefixes a plan with little-endian JSON metadata length."""
    if len(data)>=4:
        length=int.from_bytes(data[:4],'little')
        if 0<length<=min(16*1024*1024,len(data)-5):
            try:
                metadata=json.loads(data[4:4+length].decode('utf-8'))
                if isinstance(metadata,dict):
                    return data[4+length:],metadata
            except (UnicodeDecodeError,json.JSONDecodeError):
                pass
    return data,None


def format_evidence(value,path=''):
    """Retain explicitly named fields, not precision inferred from layer names."""
    out=[]
    if isinstance(value,dict):
        for key,item in value.items():
            p=f'{path}/{key}'
            if key.lower().replace(' ','') in ('precision','datatype','format/datatype','format','type') and not isinstance(item,(dict,list)):
                out.append({'field':p,'value':item})
            elif isinstance(item,(dict,list)):
                out.extend(format_evidence(item,p))
    elif isinstance(value,list):
        for i,item in enumerate(value):
            out.extend(format_evidence(item,f'{path}/{i}'))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--device',type=int,default=0)
    args=parser.parse_args()
    if args.out_dir.exists():
        parser.error('Output exists; use a new version')
    import torch
    import tensorrt as trt
    repo=Path(__file__).resolve().parents[1]
    inputs={p:capture_inputs(repo,p) for p in ('fp16','uniform','low_luminance','vcsc_proportional')}
    for _,_,_,_,prov,_ in inputs.values():
        if prov['environment']['tensorrt_python']!=trt.__version__:
            raise ValueError('TensorRT version differs from export; no automatic rebuild')
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    torch.cuda.set_device(args.device)
    summaries={}
    with GpuPhaseLock(repo/'results/architecture_matrix_v1/.gpu_phase.lock','g0_engine_inspection'):
        args.out_dir.mkdir(parents=True)
        logger=trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger,'')
        for policy,(path,_,_,_,prov,digest) in inputs.items():
            plan,metadata=unwrap_engine(path.read_bytes())
            with trt.Runtime(logger) as runtime:
                engine=runtime.deserialize_cuda_engine(plan)
                if engine is None:
                    raise RuntimeError(f'Cannot deserialize existing engine: {policy}')
                inspector=engine.create_engine_inspector()
                raw=inspector.get_engine_information(trt.LayerInformationFormat.JSON)
                try:
                    parsed=json.loads(raw)
                except json.JSONDecodeError:
                    parsed={'unparsed_text':raw}
                io_tensors=[{'name':engine.get_tensor_name(i),'mode':str(engine.get_tensor_mode(engine.get_tensor_name(i))),
                    'dtype':str(engine.get_tensor_dtype(engine.get_tensor_name(i))),'shape':list(engine.get_tensor_shape(engine.get_tensor_name(i)))} for i in range(engine.num_io_tensors)]
                result={'policy':policy,'engine_sha256':digest,'source_weights_sha256':prov['source_weights_sha256'],
                    'tensorrt':trt.__version__,'gpu':torch.cuda.get_device_name(args.device),'profiling_verbosity':str(engine.profiling_verbosity),
                    'num_layers':engine.num_layers,'num_optimization_profiles':engine.num_optimization_profiles,'io_tensors':io_tensors,
                    'ultralytics_metadata':metadata,'inspector':parsed,'explicit_fields':format_evidence(parsed),
                    'export_args':prov['export_args'],'calibration_cache':prov.get('calibration_cache'),
                    'limitation':'Inspector disclosure depends on verbosity fixed at build. Tensor I/O or intermediate format is not proof of arithmetic/accumulator precision. Layer names or requested INT8 flag cannot prove all layers execute INT8; fused layers need care. Missing details remain unknown; no rebuild authorized.'}
                write_json(args.out_dir/f'{policy}.json',result)
                summaries[policy]={k:result[k] for k in ('engine_sha256','profiling_verbosity','num_layers','io_tensors')}
                summaries[policy]['explicit_field_count']=len(result['explicit_fields'])
                print(f"INSPECTED: {policy}: {engine.num_layers} layers; verbosity={engine.profiling_verbosity}",flush=True)
                del inspector,engine
    write_json(args.out_dir/'inspection_summary.json',{'status':'inspection_completed_not_precision_certified','engines':summaries,
        'git_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True).stdout.strip(),
        'source':'https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/engine-tools.html',
        'scope':'Deserialization and inspector only; no builder, calibration, inference, layer modification or precision attribution from names.'})
    print(f'DONE: {args.out_dir}')


if __name__=='__main__':
    main()
