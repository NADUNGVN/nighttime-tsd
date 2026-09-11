#!/usr/bin/env python3
"""Native-coordinate, GT-centric dev candidate diagnostics; no inference or AP changes."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import scale_boxes

from audit_cctsdb_measurement import sha256, lf_sha256, load_records
from capture_cctsdb_validator import write_json
from run_g0_int8_capture import check_same_targets, POLICIES, read


def candidates(record):
    native=record['validator_input']
    gt=torch.tensor(native['target_xyxy'],dtype=torch.float32).reshape(-1,4)
    pred=torch.tensor(native['prediction_xyxy'],dtype=torch.float32).reshape(-1,4)
    ious=box_iou(gt,pred).numpy()
    rows=[]
    for i,cls in enumerate(native['target_class_id']):
        ids=[j for j,c in enumerate(record['class_id']) if c==cls]
        best=max(ids,key=lambda j:(float(ious[i,j]),record['confidence'][j],-j)) if ids else None
        above=[j for j in ids if ious[i,j]>=.5]
        rows.append({'gt_index':i,'class_id':int(cls),'best_index':best,'best_iou':float(ious[i,best]) if best is not None else 0.,
            'best_score':record['confidence'][best] if best is not None else None,
            'max_score_iou50':max((record['confidence'][j] for j in above),default=None),
            'candidate_count_iou50':len(above),
            'best_iou_conf025':max((float(ious[i,j]) for j in ids if record['confidence'][j]>=.25),default=0.)})
    return rows


def compare(reference,current):
    a,b=candidates(reference),candidates(current)
    rows=[]
    for x,y in zip(a,b):
        rows.append({'image':reference['image'],'gt_index':x['gt_index'],'class_id':x['class_id'],'fp16':x,'int8':y,
            'delta_best_iou':y['best_iou']-x['best_iou'],
            'delta_max_score_iou50':y['max_score_iou50']-x['max_score_iou50'] if x['max_score_iou50'] is not None and y['max_score_iou50'] is not None else None})
    return rows


def select_examples(rows,limit=2):
    definitions={
        'iou75_loss':(lambda r:r['fp16']['best_iou']>=.75 and r['int8']['best_iou']<.75,lambda r:r['delta_best_iou']),
        'score_drop_with_iou75_retained':(lambda r:min(r['fp16']['best_iou'],r['int8']['best_iou'])>=.75 and r['delta_max_score_iou50'] is not None and r['delta_max_score_iou50']<0,lambda r:r['delta_max_score_iou50']),
        'iou75_gain_control':(lambda r:r['fp16']['best_iou']<.75 and r['int8']['best_iou']>=.75,lambda r:-r['delta_best_iou'])}
    result=[]
    for category,(eligible,key) in definitions.items():
        seen=set()
        for r in sorted(filter(eligible,rows),key=lambda r:(key(r),r['image'],r['gt_index'])):
            if r['image'] in seen:
                continue
            result.append({'category':category,**r});seen.add(r['image'])
            if len(seen)>=limit:
                break
    return result


def render_case(case,reference,current,image_path,destination,policy):
    with Image.open(image_path) as im:
        source=im.convert('RGB')
    if source.size != tuple(reversed(reference['orig_shape'])):
        raise ValueError('Image dimensions do not match capture')
    n=reference['validator_input'];i=case['gt_index']
    gt=scale_boxes(n['imgsz'],torch.tensor([n['target_xyxy'][i]],dtype=torch.float32),reference['orig_shape'],ratio_pad=n['ratio_pad'])[0].tolist()
    cx,cy=(gt[0]+gt[2])/2,(gt[1]+gt[3])/2
    span=max(96,3*max(gt[2]-gt[0],gt[3]-gt[1]))
    crop=[max(0,int(cx-span/2)),max(0,int(cy-span/2)),min(source.width,int(cx+span/2)+1),min(source.height,int(cy+span/2)+1)]
    panel=Image.new('RGB',(1000,580),'white');draw=ImageDraw.Draw(panel)
    draw.text((10,8),f"{policy} | {case['category']} | {case['image']} GT#{i} class={case['class_id']}",fill='black')
    draw.text((10,27),'GREEN = captured target (scaled for display); ORANGE = best same-class IoU candidate; not AP matching',fill='black')
    for column,(name,r,detail) in enumerate((('FP16',reference,case['fp16']),(policy,current,case['int8']))):
        patch=source.crop(crop).resize((480,480))
        panel.paste(patch,(10+column*500,85))
        score=detail['best_score']
        draw.text((10+column*500,50),f"{name}: native IoU={detail['best_iou']:.3f} score={score if score is None else round(score,3)}",fill='black')
        draw.text((10+column*500,66),f"IoU>=0.5 candidates: {detail['candidate_count_iou50']}",fill='black')
        boxes=[(gt,'lime')]
        if detail['best_index'] is not None:
            boxes.append((r['xyxy'][detail['best_index']],'orange'))
        for box,color in boxes:
            coords=[10+column*500+(box[0]-crop[0])*480/(crop[2]-crop[0]),85+(box[1]-crop[1])*480/(crop[3]-crop[1]),10+column*500+(box[2]-crop[0])*480/(crop[2]-crop[0]),85+(box[3]-crop[1])*480/(crop[3]-crop[1])]
            # Clamp for visualization only; no geometry metric is changed.
            coords=[min(489+column*500,max(10+column*500,v)) if j%2==0 else min(564,max(85,v)) for j,v in enumerate(coords)]
            draw.rectangle(coords,outline=color,width=3)
    panel.save(destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--allow-lf-normalization',action='store_true')
    args=parser.parse_args()
    if args.out_dir.exists():
        parser.error('Use a new output directory; never overwrite')
    repo=Path(__file__).resolve().parents[1];root=repo/'results/measurement_audit_v1'
    paths={'fp16':root/'server_fp16_capture_v1',**{p:root/'server_int8_capture_v1'/p/'capture' for p in POLICIES}}
    captures={};hashes={}
    for model,path in paths.items():
        report=read(path/'capture_report.json');pred=path/'validator_predictions.json'
        exact=sha256(pred)==report['predictions_sha256']
        if not exact and not (args.allow_lf_normalization and lf_sha256(pred)==report['predictions_sha256']):
            raise ValueError(f'Prediction hash mismatch: {model}')
        if report['status']!='pass' or report['dataset_split']!='CCTSDB2021/dev':
            raise ValueError('Only verified dev captures allowed')
        captures[model]=read(pred)
        hashes[model]={'prediction_sha256':sha256(pred),'hash_match':'exact_bytes' if exact else 'LF_normalized_checkout','engine_sha256':report['model_sha256']}
    ref=load_records(captures['fp16']);all_rows={};examples=[];summary={}
    for policy in POLICIES:
        check_same_targets(captures['fp16'],captures[policy]);cur=load_records(captures[policy])
        rows=[r for name in sorted(ref) for r in compare(ref[name],cur[name])]
        all_rows[policy]=rows
        summary[policy]={'gt_count':len(rows),'iou75_loss':sum(r['fp16']['best_iou']>=.75 and r['int8']['best_iou']<.75 for r in rows),
            'iou75_gain':sum(r['fp16']['best_iou']<.75 and r['int8']['best_iou']>=.75 for r in rows),
            'median_delta_best_iou':float(np.median([r['delta_best_iou'] for r in rows])),
            'multiple_iou50_candidates_fp16':sum(r['fp16']['candidate_count_iou50']>1 for r in rows),
            'multiple_iou50_candidates_int8':sum(r['int8']['candidate_count_iou50']>1 for r in rows)}
        examples.extend({'policy':policy,**r} for r in select_examples(rows))
    args.out_dir.mkdir(parents=True)
    write_json(args.out_dir/'candidate_per_gt.json',all_rows)
    for index,case in enumerate(examples):
        image_path=repo/'data/processed/cctsdb2021_clean/dev/images'/case['image']
        if image_path.is_file():
            file=f"case_{index+1:02d}_{case['policy']}_{case['category']}.png"
            render_case(case,ref[case['image']],load_records(captures[case['policy']])[case['image']],image_path,args.out_dir/file,case['policy'])
            case['overlay']=file;case['image_sha256']=sha256(image_path)
        else:
            case['overlay']=None
    write_json(args.out_dir/'selected_cases.json',examples)
    write_json(args.out_dir/'diagnostic_summary.json',{'dataset_split':'CCTSDB2021/dev','inputs':hashes,'metrics':summary,
        'git_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True).stdout.strip(),'script_sha256':sha256(Path(__file__)),
        'selection':'Two distinct images per policy/category, extreme loss / score drop / gain control, ties filename then GT index. Illustrative purposive examples, not representative random samples.',
        'scope':'Native-coordinate best same-class candidate per GT, many-to-one permitted; NOT COCO matching, AP, duplicate FP count, or causal attribution. Scores post-NMS captured at conf>=0.001; candidates below threshold unavailable. GT display transformed to original image for visualization only. Frozen engines, no inference/export/test.'})
    print(json.dumps(summary,indent=2));print(f'DONE: {args.out_dir}')


if __name__=='__main__':
    main()
