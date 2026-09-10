#!/usr/bin/env python3
"""Dev-only paired image bootstrap and localization diagnostics from frozen captures."""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_cctsdb_measurement import load_records, load_xml, sha256, size_bin
from capture_cctsdb_validator import write_json
from run_g0_int8_capture import check_reference, check_same_targets, read, POLICIES
from verify_cctsdb_capture import coco_size

LABELS = ("all", "xs", "s", "m", "l", "xl")
MODELS = ("fp16", *POLICIES)


def ap_values(evaluator):
    p = evaluator.eval["precision"]
    out = []
    for a in range(len(LABELS)):
        both = (p[0,:,:,a,0], p[:,:,:,a,0])
        out.append([float(v[v >= 0].mean()) if np.any(v >= 0) else float("nan") for v in both])
    return np.array(out)


def resample_ap(evaluator, indices):
    """Reuse within-image matching, duplicate occurrences, rerun official accumulation.

    Repeated source images are independent occurrences in evalImgs, not unique IDs
    passed to evaluate() (which would deduplicate them). Original evaluator is untouched.
    """
    n = len(evaluator.params.imgIds)
    if not len(indices) or any(i < 0 or i >= n for i in indices):
        raise ValueError("Invalid bootstrap image indices")
    boot = copy.copy(evaluator)
    boot.params = copy.deepcopy(evaluator.params)
    boot.params.imgIds = list(range(1, len(indices)+1))
    boot._paramsEval = copy.deepcopy(boot.params)
    blocks = len(evaluator.params.catIds)*len(evaluator.params.areaRng)
    boot.evalImgs = [evaluator.evalImgs[block*n+int(i)] for block in range(blocks) for i in indices]
    with contextlib.redirect_stdout(io.StringIO()):
        boot.accumulate()
    values = ap_values(boot)
    # Hold the original class support fixed within each size endpoint. A rare
    # class disappearing in a draw must not silently change the macro estimand.
    original_support = np.any(evaluator.eval["precision"] >= 0, axis=(0,1,4))
    sampled_support = np.any(boot.eval["precision"] >= 0, axis=(0,1,4))
    values[np.any(original_support & ~sampled_support, axis=0)] = np.nan
    return values


def ci(values, point):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    limits = np.quantile(valid, [.025,.975], method="linear").tolist() if len(valid) else [None,None]
    return {"point_delta_pp": float(point) if np.isfinite(point) else None, "ci95_percentile_pp": limits,
            "valid_resamples": len(valid), "undefined_resamples": len(values)-len(valid)}


def summarize_draws(draws, points):
    contrasts = [(policy,"fp16") for policy in POLICIES] + [("low_luminance","uniform"),("vcsc_proportional","uniform")]
    rows = {}
    for left,right in contrasts:
        i,j = MODELS.index(left), MODELS.index(right)
        rows[f"{left}_minus_{right}"] = {label: {metric: ci(100*(draws[:,i,a,t]-draws[:,j,a,t]),100*(points[i,a,t]-points[j,a,t]))
            for t,metric in enumerate(("map50","map50_95"))} for a,label in enumerate(LABELS)}
    return rows


def localization(evaluator):
    """GT-centric diagnostics using ALL-area, same-class score-greedy matching."""
    output = {}
    # All-area only, so area filtering does not change the assignment.
    for e in evaluator.evalImgs:
        if e is None or e["aRng"] != evaluator.params.areaRng[0]:
            continue
        for j,gid in enumerate(e["gtIds"]):
            gt = evaluator.cocoGt.anns[gid]
            image = evaluator.cocoGt.imgs[gt["image_id"]]["file_name"]
            matches = e["gtMatches"][:,j]
            key = str(gid)
            row = {"image":image,"class_id":gt["category_id"],"size":size_bin(gt["area"]),
                   "matched_iou50":bool(matches[0]),"matched_iou75":bool(matches[5]),"matched_iou90":bool(matches[8]),
                   "matched50_pair_iou":None,"matched50_center_error_over_sqrt_gt_area":None}
            if matches[0]:
                dt = evaluator.cocoDt.anns[int(matches[0])]
                x,y,w,h = gt["bbox"];u,v,s,t = dt["bbox"]
                inter = max(0,min(x+w,u+s)-max(x,u))*max(0,min(y+h,v+t)-max(y,v))
                row["matched50_pair_iou"] = inter/(w*h+s*t-inter)
                row["matched50_center_error_over_sqrt_gt_area"] = float(np.hypot(x+w/2-u-s/2,y+h/2-v-t/2)/np.sqrt(w*h))
            output[key] = row
    return output


def localization_summary(per_model):
    ref = per_model["fp16"]
    out = {}
    for model, rows in per_model.items():
        if set(rows) != set(ref):
            raise ValueError("Localization GT identity mismatch")
        out[model] = {}
        for label in LABELS:
            ids = [k for k,v in rows.items() if label == "all" or v["size"] == label]
            metrics = {"instances":len(ids)}
            for threshold in (50,75,90):
                key = f"matched_iou{threshold}"
                count = sum(rows[k][key] for k in ids)
                metrics[f"recall_iou{threshold}"] = count/len(ids) if ids else None
                metrics[f"lost_vs_fp16_iou{threshold}"] = sum(ref[k][key] and not rows[k][key] for k in ids)
                metrics[f"gained_vs_fp16_iou{threshold}"] = sum(rows[k][key] and not ref[k][key] for k in ids)
            for field in ("matched50_pair_iou","matched50_center_error_over_sqrt_gt_area"):
                values = [rows[k][field] for k in ids if rows[k][field] is not None]
                metrics[field+"_median"] = float(np.median(values)) if values else None
            out[model][label] = metrics
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--resamples",type=int,default=1000)
    parser.add_argument("--seed",type=int,default=20260910)
    args = parser.parse_args()
    if args.out_dir.exists() or args.resamples < 2:
        parser.error("Use a new output directory and at least two resamples")
    repo = Path(__file__).resolve().parents[1]
    root = repo / "results/measurement_audit_v1"
    ref_capture = root / "server_fp16_capture_v1"
    ref_verify = root / "server_native_size_v1"
    xml_path = (repo / "../nighttime-tsd/data/raw/CCTSDB2021/xml.zip").resolve()
    ref_report,_ = check_reference(ref_capture,ref_verify,xml_path)
    if ref_report["environment"]["numpy"] != np.__version__:
        raise ValueError("NumPy differs from capture; use existing server audit environment")
    captures = {"fp16":ref_capture, **{p:root/"server_int8_capture_v1"/p/"capture" for p in POLICIES}}
    verifications = {"fp16":ref_verify, **{p:root/"server_int8_capture_v1"/p/"verification" for p in POLICIES}}
    reference = read(ref_capture/"validator_predictions.json")
    xml = load_xml(xml_path,load_records(reference))
    evaluators,points,gt_records,inputs = [],[],{},{}
    for model in MODELS:
        cap,ver = captures[model],verifications[model]
        report = read(cap/"capture_report.json");summ = read(ver/"verification_summary.json")
        if report["dataset_split"] != "CCTSDB2021/dev" or report["status"] != "pass" or summ["native_matching_status"] != "pass":
            raise ValueError(f"Unverified dev input: {model}")
        if sha256(cap/"validator_predictions.json") != report["predictions_sha256"] or report["predictions_sha256"] != summ["capture_prediction_sha256"] or sha256(cap/"capture_report.json") != summ["capture_report_sha256"]:
            raise ValueError(f"Capture hash mismatch: {model}")
        payload = read(cap/"validator_predictions.json")
        if payload["model_sha256"] != report["model_sha256"] or report["model_sha256"] != summ["model_sha256"]:
            raise ValueError("Engine identity differs between capture and verification")
        check_same_targets(reference,payload)
        previous = read(ver/"size_coco_xml.json")
        if previous["xml_sha256"] != sha256(xml_path):
            raise ValueError("XML differs from validated size inputs")
        current,evaluator = coco_size(list(load_records(payload).values()),xml,return_evaluator=True)
        if current["metric_id"] != previous["metric_id"] or current["rules"] != previous["rules"] or current["evaluator_source_sha256"] != previous["evaluator_source_sha256"]:
            raise ValueError("COCO evaluator conventions changed")
        point = ap_values(evaluator)
        expected = np.array([[previous["metrics"][label][k] for k in ("map50","map50_95")] for label in LABELS])
        if not np.allclose(point,expected,rtol=0,atol=1e-12,equal_nan=True):
            raise ValueError(f"COCO point replay mismatch: {model}")
        points.append(point);evaluators.append(evaluator);gt_records[model]=localization(evaluator)
        inputs[model] = {"predictions_sha256":report["predictions_sha256"],"engine_sha256":report["model_sha256"],
                         "capture_report_sha256":sha256(cap/"capture_report.json"),"size_report_sha256":sha256(ver/"size_coco_xml.json")}
        print(f"VERIFIED: {model}: COCO point replay exact within 1e-12",flush=True)
    args.out_dir.mkdir(parents=True)
    n = len(evaluators[0].params.imgIds)
    rng = np.random.Generator(np.random.PCG64(args.seed))
    samples = rng.integers(0,n,size=(args.resamples,n))
    # Keep canonical source-image order for ties, duplicating each occurrence.
    samples.sort(axis=1)
    names = [evaluators[0].cocoGt.imgs[i]["file_name"] for i in evaluators[0].params.imgIds]
    write_json(args.out_dir/"bootstrap_samples.json",{"seed":args.seed,"generator":"PCG64","image_names":names,"indices":samples.tolist()})
    write_json(args.out_dir/"localization_per_gt.json",gt_records)
    write_json(args.out_dir/"localization_summary.json",{"metrics":localization_summary(gt_records),
        "scope":"GT recall at IoU thresholds, class-correct ALL-area COCO matching at captured conf>=0.001 and max_det=300. Pair-error medians condition on IoU50-matched GT (selection bias); not a causal decomposition or deployment confidence threshold."})
    draws = np.empty((args.resamples,len(MODELS),len(LABELS),2))
    start = time.monotonic()
    for b,sample in enumerate(samples):
        for m,evaluator in enumerate(evaluators):
            draws[b,m] = resample_ap(evaluator,sample)
        if (b+1)%10==0 or b==0 or b+1==args.resamples:
            elapsed=time.monotonic()-start
            print(f"BOOTSTRAP {b+1}/{args.resamples}: {elapsed:.1f}s elapsed, estimated remaining {elapsed/(b+1)*(args.resamples-b-1):.1f}s",flush=True)
    # JSON has no NaN: absent-GT bootstrap endpoints are explicit null.
    values = draws.astype(object);values[~np.isfinite(draws)] = None
    write_json(args.out_dir/"bootstrap_draws.json",{"axes":["resample","model","size","metric"],"models":MODELS,"sizes":LABELS,"metrics":["map50","map50_95"],"values":values.tolist()})
    result = {"schema_version":1,"created_utc":datetime.now(timezone.utc).isoformat(),"dataset_split":"CCTSDB2021/dev",
        "git_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=repo,capture_output=True,text=True).stdout.strip(),
        "script_sha256":sha256(Path(__file__)),"xml_sha256":sha256(xml_path),"inputs":inputs,
        "python":platform.python_version(),"numpy":np.__version__,"pycocotools":current["pycocotools"],"evaluator_source_sha256":current["evaluator_source_sha256"],
        "seed":args.seed,"resamples":args.resamples,"sample_plan_sha256":sha256(args.out_dir/"bootstrap_samples.json"),
        "metric_id":current["metric_id"],"contrasts":summarize_draws(draws,np.array(points)),
        "procedure":"IID image pairs bootstrap, N images with replacement; same sample in every model. Duplicate occurrences retained. Reuse within-image COCO matches, rerun official accumulation/AP. Canonical filename order for score ties. 95% percentile CI, NumPy linear quantiles; an endpoint becomes null if any originally supported class has no GT in that size bin in a draw; finite resample count reported.",
        "limitations":["Conditional on frozen seed42 engines, not calibration-seed variability.","Image IID assumes independent scenes; sequence/camera dependence may narrow CIs.","Exploratory dev contrasts, unadjusted for multiple comparisons; not confirmatory significance or policy selection.","COCO/XML only; do not mix with Ultralytics or historical custom-size AP.","No environmental test-domain metrics or official test data used."],
        "status":"completed" if args.resamples>=1000 else "smoke_only","global_g0":"review_required"}
    write_json(args.out_dir/"analysis_summary.json",result)
    print(f"DONE: {args.out_dir/'analysis_summary.json'}",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
