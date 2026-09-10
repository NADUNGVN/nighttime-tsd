#!/usr/bin/env python3
"""Capture and verify three frozen initial INT8 engines on dev, sequentially."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from audit_cctsdb_measurement import sha256, load_records, load_xml
from capture_cctsdb_validator import capture_inputs, write_json, PINNED_VERSION

POLICIES = ("uniform", "low_luminance", "vcsc_proportional")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_reference(capture, verification, xml_path):
    report = read(capture / "capture_report.json")
    summary = read(verification / "verification_summary.json")
    sizes = read(verification / "size_coco_xml.json")
    native = read(verification / "native_matching.json")
    if report["status"] != "pass" or report["dataset_split"] != "CCTSDB2021/dev":
        raise ValueError("FP16 reference is not a passed dev capture")
    if report["images"] != 1636 or report["instances"] != 2706:
        raise ValueError("FP16 reference counts mismatch")
    if sha256(capture / "validator_predictions.json") != report["predictions_sha256"]:
        raise ValueError("FP16 prediction hash mismatch")
    if summary["capture_report_sha256"] != sha256(capture / "capture_report.json") or summary["capture_prediction_sha256"] != report["predictions_sha256"]:
        raise ValueError("FP16 verification belongs to another capture")
    if summary["native_matching_status"] != "pass" or native["changed_tp_decisions"] != 0:
        raise ValueError("FP16 native matching has not passed")
    if sizes["xml_sha256"] != sha256(xml_path):
        raise ValueError("XML differs from FP16 size diagnostic")
    return report, sizes


def check_same_targets(reference, current):
    a, b = load_records(reference), load_records(current)
    if set(a) != set(b):
        raise ValueError("INT8/FP16 image membership mismatch")
    for name in a:
        for key in ("orig_shape",):
            if a[name][key] != b[name][key]:
                raise ValueError(f"Image dimensions changed: {name}")
        for key in ("imgsz", "ratio_pad", "target_xyxy", "target_class_id"):
            if a[name]["validator_input"][key] != b[name]["validator_input"][key]:
                raise ValueError(f"INT8/FP16 target/preprocessing mismatch: {name}: {key}")


def comparison_row(policy, capture, size, reference_capture, reference_size):
    if size["metric_id"] != reference_size["metric_id"] or size["rules"] != reference_size["rules"] or size["evaluator_source_sha256"] != reference_size["evaluator_source_sha256"]:
        raise ValueError("Size evaluator conventions differ from FP16")
    if size["xml_sha256"] != reference_size["xml_sha256"]:
        raise ValueError("XML changed between representations")
    bins = {}
    for name, metric in size["metrics"].items():
        ref = reference_size["metrics"][name]
        if metric["instances"] != ref["instances"]:
            raise ValueError(f"Size counts differ: {name}")
        bins[name] = {"instances": metric["instances"]}
        for k in ("map50", "map50_95"):
            bins[name][k] = metric[k]
            bins[name][f"delta_{k}_pp"] = 100*(metric[k]-ref[k]) if metric[k] is not None and ref[k] is not None else None
    return {"policy": policy, "calibration_seed": 42, "engine_sha256": capture["model_sha256"],
        "ultralytics_full": {"metrics": capture["metrics"], "delta_pp": {k:100*(capture["metrics"][k]-reference_capture["metrics"][k]) for k in ("map50","map50_95","precision","recall")}},
        "coco_xml": bins}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.out_dir.resolve()
    if output.exists():
        parser.error("Output exists; preserve it and choose a new version")
    import torch
    import numpy as np
    import tensorrt as trt
    import ultralytics
    if not torch.cuda.is_available() or ultralytics.__version__ != PINNED_VERSION or importlib.metadata.version("pycocotools") != "2.0.10":
        parser.error("Requires CUDA, Ultralytics 8.4.102 and pycocotools 2.0.10; use g0_size_env based on nighttime-tsd")
    capture_ref = repo / "results/measurement_audit_v1/server_fp16_capture_v1"
    verify_ref = repo / "results/measurement_audit_v1/server_native_size_v1"
    xml = (repo / "../nighttime-tsd/data/raw/CCTSDB2021/xml.zip").resolve()
    ref_report, ref_sizes = check_reference(capture_ref, verify_ref, xml)
    fp16 = capture_inputs(repo, "fp16")
    if fp16[-1] != ref_report["model_sha256"]:
        raise ValueError("FP16 capture belongs to another engine")
    for key, actual in (("torch", torch.__version__), ("numpy", np.__version__), ("tensorrt", trt.__version__), ("ultralytics", ultralytics.__version__)):
        if ref_report["environment"][key] != actual:
            raise ValueError(f"Environment differs from FP16 capture: {key}")
    # Check every engine before launching the first inference. No automatic export.
    identities = {}
    for policy in POLICIES:
        *_, prov, digest = capture_inputs(repo, policy)
        if prov["environment"]["tensorrt_python"] != trt.__version__:
            raise ValueError(f"TensorRT version differs: {policy}")
        identities[policy] = digest
    reference_predictions = read(capture_ref / "validator_predictions.json")
    load_xml(xml, load_records(reference_predictions))
    output.mkdir(parents=True)
    rows = []
    for policy in POLICIES:
        dest = output / policy
        commands = [
            [sys.executable, str(repo / "scripts/capture_cctsdb_validator.py"), "--representation", policy, "--out-dir", str(dest / "capture"), "--device", args.device],
            [sys.executable, str(repo / "scripts/verify_cctsdb_capture.py"), "--capture-dir", str(dest / "capture"), "--xml", str(xml), "--out-dir", str(dest / "verification")]]
        for command in commands:
            print("START: " + " ".join(command), flush=True)
            subprocess.run(command, cwd=repo, check=True)
        captured = read(dest / "capture/capture_report.json")
        if captured["model_sha256"] != identities[policy]:
            raise ValueError("Engine changed after preflight")
        check_same_targets(reference_predictions, read(dest / "capture/validator_predictions.json"))
        row = comparison_row(policy, captured, read(dest / "verification/size_coco_xml.json"), ref_report, ref_sizes)
        rows.append(row)
        write_json(dest / "comparison.json", row)
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(["git","rev-parse","HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip(),
        "reference_capture_sha256": sha256(capture_ref / "capture_report.json"), "reference_size_sha256": sha256(verify_ref / "size_coco_xml.json"),
        "dataset_split": "CCTSDB2021/dev", "reference_metrics": ref_report["metrics"], "reference_coco_xml": ref_sizes["metrics"], "rows": rows,
        "status": "capture_and_verification_completed", "global_g0": "review_required",
        "scope": "Single-seed measurement audit only; not a policy selection or superiority claim. No training/export/test evaluation. Review original XML conventions and separate COCO from Ultralytics metrics."}
    write_json(output / "comparison_summary.json", report)
    print("policy                 UL_AP50  COCO_AP50  XS_delta_pp  S_delta_pp")
    for r in rows:
        print(f"{r['policy']:22s} {100*r['ultralytics_full']['metrics']['map50']:7.2f} {100*r['coco_xml']['all']['map50']:10.2f} {r['coco_xml']['xs']['delta_map50_pp']:12.2f} {r['coco_xml']['s']['delta_map50_pp']:11.2f}")
    print(f"DONE: {output / 'comparison_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
