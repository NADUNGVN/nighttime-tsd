#!/usr/bin/env python3
"""CPU rematching of native capture; separately versioned COCO/XML size diagnostic."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import io
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import ultralytics
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import box_iou

from audit_cctsdb_measurement import NAMES, area, lf_sha256, load_records, load_xml, sha256, size_bin
from capture_cctsdb_validator import write_json

VERSION = "native_rematch_coco_size_v1"


def rematch_native(records, thresholds):
    """Recompute IoU and assignments from bbox; do not use saved TP as input."""
    proxy = SimpleNamespace(iouv=torch.tensor(thresholds, dtype=torch.float32))
    differences, total = [], 0
    for record in records:
        native = record["validator_input"]
        pred = torch.tensor(native["prediction_xyxy"], dtype=torch.float32).reshape(-1, 4)
        truth = torch.tensor(native["target_xyxy"], dtype=torch.float32).reshape(-1, 4)
        pc = torch.tensor(record["class_id"], dtype=torch.float32)
        gc = torch.tensor(native["target_class_id"], dtype=torch.float32)
        saved = np.asarray(record["validator_statistics"]["tp"], dtype=bool).reshape(len(pred), len(thresholds))
        actual = BaseValidator.match_predictions(proxy, pc, gc, box_iou(truth, pred)).numpy()
        changed = np.argwhere(actual != saved)
        total += saved.size
        if changed.size:
            differences.append({"image": record["image"], "changed_bits": len(changed),
                "samples": [{"prediction_index": int(i), "iou_threshold": thresholds[j], "saved_tp": bool(saved[i,j]), "cpu_tp": bool(actual[i,j])} for i,j in changed[:20]]})
    return {"images": len(records), "tp_decisions": total, "changed_tp_decisions": sum(r["changed_bits"] for r in differences),
            "changed_images": len(differences), "differences": differences,
            "status": "pass" if not differences else "review_required",
            "method": "Recompute float32 IoU and installed BaseValidator matching on CPU from native bbox, then compare captured GPU decisions. Not an independent validation of Ultralytics assignment design.",
            "matching_source_sha256": hashlib.sha256(inspect.getsource(BaseValidator.match_predictions).encode()).hexdigest()}


def size_ranges():
    # COCO ranges include both endpoints. nextafter implements our exclusive
    # lower bounds without moving thresholds or rounding annotations.
    bounds = [(0., 1e20), (0.,210.), (np.nextafter(210.,np.inf),400.),
              (np.nextafter(400.,np.inf),1000.), (np.nextafter(1000.,np.inf),2000.),
              (np.nextafter(2000.,np.inf),1e20)]
    return [list(b) for b in bounds]


def coco_size(records, xml, *, return_evaluator=False):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if importlib.metadata.version("pycocotools") != "2.0.10":
        raise ValueError("Size diagnostic requires pycocotools==2.0.10; use isolated environment if needed")
    if set(xml) != {r["image"] for r in records}:
        raise ValueError("XML/capture membership mismatch")
    annotations, detections, images, outside = [], [], [], []
    for image_id, r in enumerate(sorted(records, key=lambda r: r["image"]), start=1):
        name = r["image"]
        if list(xml[name]["shape"]) != list(r["orig_shape"]):
            raise ValueError(f"XML/capture dimension mismatch: {name}")
        height, width = r["orig_shape"]
        images.append({"id": image_id, "file_name": name, "height": height, "width": width})
        for cls, b in xml[name]["rows"]:
            annotations.append({"id": len(annotations)+1, "image_id": image_id, "category_id": cls,
                "bbox": [b[0],b[1],b[2]-b[0],b[3]-b[1]], "area": area(b), "iscrowd": 0})
            if b[0] < 0 or b[1] < 0 or b[2] > width or b[3] > height:
                outside.append({"image": name, "xml_box": b})
        # Stable score sort; ties preserve within-image capture index.
        order = sorted(range(len(r["confidence"])), key=lambda i: -r["confidence"][i])[:300]
        for i in order:
            b = r["xyxy"][i]
            detections.append({"image_id": image_id, "category_id": r["class_id"][i],
                "bbox": [b[0],b[1],b[2]-b[0],b[3]-b[1]], "score": r["confidence"][i]})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": annotations, "categories": [{"id":i,"name":n} for i,n in enumerate(NAMES)], "info": {}}
        gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = {**gt.dataset, "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.areaRng = size_ranges()
        evaluator.params.areaRngLbl = ["all","xs","s","m","l","xl"]
        evaluator.params.maxDets = [300]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"]
    metrics = {}
    for a, label in enumerate(evaluator.params.areaRngLbl):
        per_class = {}
        for c, name in enumerate(NAMES):
            p = precision[:,:,c,a,0]
            valid = p[p >= 0]
            p50 = p[0][p[0] >= 0]
            per_class[name] = {"ap50": float(p50.mean()) if len(p50) else None,
                               "ap50_95": float(valid.mean()) if len(valid) else None}
        values = precision[:,:,:,a,0]
        p50 = values[0][values[0] >= 0]
        valid = values[values >= 0]
        metrics[label] = {"instances": sum(label == "all" or size_bin(ann["area"]) == label for ann in annotations),
            "map50": float(p50.mean()) if len(p50) else None, "map50_95": float(valid.mean()) if len(valid) else None, "per_class": per_class}
    report = {"metrics": metrics, "images": len(images), "predictions": len(detections), "out_of_image_xml_boxes_preserved": outside,
        "metric_id": "COCO_bbox_AP_custom_CCTSDB_area_XML_original_coordinates_v1",
        "rules": {"ground_truth": "XML bbox and continuous area (xmax-xmin)*(ymax-ymin), unrounded and unclipped; no +1. XML mismatches are not silently replaced by YOLO labels.",
            "predictions": "Same-val captured original-image scaled/clipped bbox; no separate predict call.",
            "matching": "Official pycocotools score-greedy one-to-one non-crowd matching; in-range GT preferred, out-of-range GT ignored; unmatched out-of-range detections ignored per COCO area convention.",
            "ties": "Lexicographic image basename, then stable captured within-image order; COCO stable confidence sort.",
            "max_detections": 300, "iou_thresholds": evaluator.params.iouThrs.tolist(), "area_ranges": size_ranges(),
            "no_gt_class": "null, excluded from macro AP; never report zero for an absent class"},
        "pycocotools": importlib.metadata.version("pycocotools"),
        "evaluator_source_sha256": hashlib.sha256(inspect.getsource(COCOeval).encode()).hexdigest(),
        "interpretation": "Versioned diagnostic, not official CCTSDB evaluator or Ultralytics mAP. All and size AP share this COCO/XML convention. Never compare its size AP to historical custom-size AP as if only model changed."}
    return (report, evaluator) if return_evaluator else report


def validate_capture(payload):
    if payload.get("schema_version") != 2 or payload.get("capture_mode") != "same_val_process_batch":
        raise ValueError("Requires same-pass capture schema v2")
    records = list(load_records(payload).values())
    thresholds = payload["iou_thresholds"]
    if len(thresholds) != 10 or not np.allclose(thresholds, np.linspace(.5,.95,10), atol=1e-7,rtol=0):
        raise ValueError("Unexpected IoU thresholds")
    for r in records:
        n = r["validator_input"]
        g = np.asarray(n["target_xyxy"],dtype=float).reshape(-1,4)
        p = np.asarray(n["prediction_xyxy"],dtype=float).reshape(-1,4)
        cls = n["target_class_id"]
        if len(g)!=len(cls) or len(p)!=len(r["confidence"]) or any(c not in (0,1,2) for c in cls):
            raise ValueError(f"Native arrays mismatch: {r['image']}")
        if not np.isfinite(g).all() or not np.isfinite(p).all() or np.any(g[:,2:]<=g[:,:2]) or np.any(p[:,2:]<p[:,:2]):
            raise ValueError(f"Invalid native geometry: {r['image']}")
        tp = np.asarray(r["validator_statistics"]["tp"])
        if tp.size and (tp.shape != (len(p),10) or tp.dtype != bool):
            raise ValueError(f"Invalid TP matrix: {r['image']}")
        if not tp.size and len(p):
            raise ValueError("Missing TP decisions")
    return records, thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--xml", type=Path, help="Optional: without XML only rematching is performed")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-lf-normalization", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Output already exists; use a new audit directory")
    report_path = args.capture_dir / "capture_report.json"
    pred_path = args.capture_dir / "validator_predictions.json"
    captured = json.loads(report_path.read_text(encoding="utf-8"))
    if captured["environment"]["ultralytics"] != ultralytics.__version__:
        parser.error("Ultralytics version differs from capture; activate original environment")
    actual_hash = sha256(pred_path)
    hash_mode = "exact_bytes"
    if actual_hash != captured["predictions_sha256"]:
        if not args.allow_lf_normalization or lf_sha256(pred_path) != captured["predictions_sha256"]:
            parser.error("Capture prediction hash mismatch")
        hash_mode = "LF_normalized_checkout"
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    if payload["model_sha256"] != captured["model_sha256"]:
        parser.error("Capture model identity mismatch")
    if captured.get("dataset_split") != "CCTSDB2021/dev" or captured.get("status") != "pass":
        parser.error("This G0 step requires a passed dev capture; test/policy selection not supported")
    records, thresholds = validate_capture(payload)
    if len(records)!=captured["images"] or sum(len(r["validator_input"]["target_class_id"]) for r in records)!=captured["instances"]:
        parser.error("Capture report counts do not match records")
    xml = load_xml(args.xml, {r["image"]:r for r in records}) if args.xml else None
    native = rematch_native(records, thresholds)
    sizes = coco_size(records, xml) if xml is not None else None
    args.out_dir.mkdir(parents=True)
    write_json(args.out_dir / "native_matching.json", native)
    if sizes is not None:
        sizes["xml_sha256"] = sha256(args.xml)
        write_json(args.out_dir / "size_coco_xml.json", sizes)
    result = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "dataset_split": captured["dataset_split"], "model_sha256": captured["model_sha256"],
        "git_commit": subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
        "script_sha256": sha256(Path(__file__)), "capture_report_sha256": sha256(report_path), "capture_prediction_sha256": actual_hash,
        "capture_hash_match": hash_mode, "native_matching_status": native["status"],
        "size_diagnostic": "completed" if sizes else "not_run_no_xml", "global_g0": "review_required",
        "environment": {"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"ultralytics":ultralytics.__version__},
        "scope": "No model loaded or inference. Size diagnostic is a separately versioned metric; resolving annotation conventions and confirming INT8 captures remains required."}
    write_json(args.out_dir / "verification_summary.json", result)
    print(json.dumps({**result,"changed_tp_decisions":native["changed_tp_decisions"]},indent=2))
    print(f"DONE: {args.out_dir}")
    return 0 if native["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
