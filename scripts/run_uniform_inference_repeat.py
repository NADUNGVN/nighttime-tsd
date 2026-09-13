#!/usr/bin/env python3
"""Run the approved Uniform inference-repeatability diagnostic; never rebuild."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from audit_cctsdb_measurement import sha256


SOURCE_STUDY = "uniform_build_repeat_v1"
STUDY = "uniform_inference_repeat_v1"
PREDICTION_PAYLOAD_VERSION = "uniform_inference_repeat_prediction_payload_v1"
DATASET_SPLIT = "CCTSDB2021/dev"
ENGINE_REPEATS = (1, 2, 3)
ROUND_ORDER = ((1, 2, 3), (2, 3, 1), (3, 1, 2))
SIZE_LABELS = ("all", "xs", "s", "m", "l", "xl")
SIZE_METRICS = ("map50", "map50_95")
FULL_METRICS = ("map50", "map50_95", "precision", "recall")
EXPECTED_RUNTIME = {
    "split": "val",
    "imgsz": 640,
    "batch": 1,
    "workers": 0,
    "task": "detect",
    "mode": "val",
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": False,
    "plots": False,
    "verbose": False,
    "save_json": False,
    "save_txt": False,
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value):
    """Serialize comparison payloads without whitespace or unstable metadata."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def prediction_payload(capture):
    """Return the versioned output payload; timestamps, paths and engine identity are excluded."""
    required = ("schema_version", "capture_mode", "iou_thresholds", "records", "coordinate_contract")
    missing = [key for key in required if key not in capture]
    if missing:
        raise ValueError(f"Prediction payload missing fields: {missing}")
    if not isinstance(capture["records"], list) or not capture["records"]:
        raise ValueError("Prediction payload has no records")
    return {
        "payload_version": PREDICTION_PAYLOAD_VERSION,
        "source_schema_version": capture["schema_version"],
        "capture_mode": capture["capture_mode"],
        "iou_thresholds": capture["iou_thresholds"],
        "coordinate_contract": capture["coordinate_contract"],
        # Preserve image order and within-image detection order exactly.
        "records": capture["records"],
    }


def prediction_payload_hash(capture):
    return hashlib.sha256(canonical_json(prediction_payload(capture))).hexdigest()


def _detection_multiset(record):
    """Compare detections as an unordered multiset only for order diagnostics."""
    rows = []
    for xyxy, confidence, class_id in zip(record.get("xyxy", ()),
                                          record.get("confidence", ()),
                                          record.get("class_id", ())):
        rows.append(canonical_json({"xyxy": xyxy, "confidence": confidence, "class_id": class_id}))
    return sorted(rows)


def prediction_difference(baseline, current):
    """Describe payload changes without pairing detections by an unsafe zip."""
    a = prediction_payload(baseline)
    b = prediction_payload(current)
    differences = {}
    for key in ("source_schema_version", "capture_mode", "iou_thresholds", "coordinate_contract"):
        if a[key] != b[key]:
            differences[f"{key}_changed"] = True

    arecords, brecords = a["records"], b["records"]
    anames, bnames = [r.get("image") for r in arecords], [r.get("image") for r in brecords]
    acounts, bcounts = Counter(anames), Counter(bnames)
    if len(anames) != len(set(anames)):
        differences["baseline_duplicate_images"] = sorted(name for name, count in acounts.items() if count > 1)
    if len(bnames) != len(set(bnames)):
        differences["current_duplicate_images"] = sorted(name for name, count in bcounts.items() if count > 1)
    if set(anames) != set(bnames):
        differences["image_membership_changed"] = {
            "missing_from_current": sorted(set(anames) - set(bnames)),
            "unexpected_in_current": sorted(set(bnames) - set(anames)),
        }
    if anames != bnames:
        first = next((i for i, (left, right) in enumerate(zip(anames, bnames)) if left != right),
                     min(len(anames), len(bnames)))
        differences["image_order_changed"] = {
            "first_difference_index": first,
            "baseline_image": anames[first] if first < len(anames) else None,
            "current_image": bnames[first] if first < len(bnames) else None,
            "baseline_count": len(anames),
            "current_count": len(bnames),
        }

    if len(anames) == len(set(anames)) and len(bnames) == len(set(bnames)) and set(anames) == set(bnames):
        by_name_a, by_name_b = {r["image"]: r for r in arecords}, {r["image"]: r for r in brecords}
        categories = {"bbox_changed": [], "confidence_changed": [], "class_changed": [],
                      "detection_order_changed": [], "orig_shape_changed": [],
                      "validator_input_changed": [], "validator_statistics_changed": [],
                      "record_other_changed": []}
        for name in sorted(by_name_a):
            left, right = by_name_a[name], by_name_b[name]
            if left == right:
                continue
            detections_changed = any(left.get(key) != right.get(key)
                                     for key in ("xyxy", "confidence", "class_id"))
            if detections_changed:
                # Only call it order drift when complete detection triples have
                # the same multiset. No positional zip is used to infer a bbox
                # correspondence.
                if _detection_multiset(left) == _detection_multiset(right):
                    categories["detection_order_changed"].append(name)
                else:
                    if left.get("xyxy") != right.get("xyxy"):
                        categories["bbox_changed"].append(name)
                    if left.get("confidence") != right.get("confidence"):
                        categories["confidence_changed"].append(name)
                    if left.get("class_id") != right.get("class_id"):
                        categories["class_changed"].append(name)
            if left.get("orig_shape") != right.get("orig_shape"):
                categories["orig_shape_changed"].append(name)
            if left.get("validator_input") != right.get("validator_input"):
                categories["validator_input_changed"].append(name)
            if left.get("validator_statistics") != right.get("validator_statistics"):
                categories["validator_statistics_changed"].append(name)
            known = {"image", "xyxy", "confidence", "class_id", "orig_shape",
                     "validator_input", "validator_statistics"}
            if ({k: v for k, v in left.items() if k not in known} !=
                    {k: v for k, v in right.items() if k not in known}):
                categories["record_other_changed"].append(name)
        differences.update({key: value for key, value in categories.items() if value})
    return {"exact": not differences, "differences": differences}


def numeric_deltas(baseline, current):
    """Return current-baseline numeric leaves while preserving metric structure."""
    if isinstance(baseline, dict) and isinstance(current, dict):
        return {key: numeric_deltas(baseline.get(key), current.get(key))
                for key in sorted(set(baseline) | set(current))}
    if isinstance(baseline, (int, float)) and not isinstance(baseline, bool) and isinstance(current, (int, float)) and not isinstance(current, bool):
        return float(current) - float(baseline)
    if baseline == current:
        return 0 if isinstance(baseline, (int, float)) else None
    return {"baseline": baseline, "current": current}


def metric_aggregate(values):
    if len(values) < 2:
        raise ValueError("At least two values are required for sample SD")
    mean = sum(values) / len(values)
    sample_std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return {"mean": mean, "sample_std": sample_std, "sample_std_pp": 100 * sample_std,
            "min": min(values), "max": max(values), "range_pp": 100 * (max(values) - min(values))}


def aggregate_within_engine(runs):
    """Aggregate three captures for one engine; the round-1 capture is a neutral baseline."""
    ordered = sorted(runs, key=lambda row: row["round"])
    if [row["round"] for row in ordered] != [1, 2, 3]:
        raise ValueError("Each engine must have exactly rounds 1, 2 and 3")
    if len({row["engine_sha256"] for row in ordered}) != 1:
        raise ValueError("Engine identity changed within an engine group")
    baseline = ordered[0]
    aggregate = {
        "engine_sha256": baseline["engine_sha256"],
        "baseline": "round_1_first_capture; not selected by metric",
        "prediction_payload_exact_all": all(row["prediction_payload_sha256"] == baseline["prediction_payload_sha256"] for row in ordered),
        "metrics_exact_all": all(row["metrics_exact"] for row in ordered),
        "captures": [{"round": row["round"], "prediction_payload_sha256": row["prediction_payload_sha256"],
                      "prediction_payload_exact_vs_round_1": row["prediction_payload_sha256"] == baseline["prediction_payload_sha256"],
                      "metrics_exact_vs_round_1": row["metrics_exact"]} for row in ordered],
        "ultralytics": {},
        "coco_xml": {},
    }
    for metric in FULL_METRICS:
        aggregate["ultralytics"][metric] = metric_aggregate([row["capture_report"]["metrics"][metric] for row in ordered])
    for size in SIZE_LABELS:
        aggregate["coco_xml"][size] = {}
        for metric in SIZE_METRICS:
            aggregate["coco_xml"][size][metric] = metric_aggregate([
                row["size_report"]["metrics"][size][metric] for row in ordered])
    return aggregate


def validate_engine_manifest(manifest, repeat_id, study_manifest, engine_sha256=None):
    if manifest.get("study") != SOURCE_STUDY or manifest.get("repeat") != repeat_id:
        raise ValueError(f"Repeat {repeat_id} build manifest identity mismatch")
    for key in ("source_weights_sha256", "onnx_sha256", "settings"):
        expected = study_manifest.get(key)
        if key not in manifest or manifest[key] != expected:
            raise ValueError(f"Repeat {repeat_id} {key} differs from study manifest")
    if engine_sha256 is not None and manifest.get("engine_sha256") != engine_sha256:
        raise ValueError(f"Repeat {repeat_id} engine hash mismatch")
    return manifest["engine_sha256"]


def validate_capture_contract(capture, verification):
    if capture.get("dataset_split") != DATASET_SPLIT or verification.get("dataset_split") != DATASET_SPLIT:
        raise ValueError("Capture is not CCTSDB2021/dev")
    if capture.get("images") != 1636 or capture.get("instances") != 2706:
        raise ValueError("Capture counts differ from the fixed dev contract")
    if capture.get("status") != "pass" or verification.get("native_matching_status") != "pass":
        raise ValueError("Capture/native matching did not pass")
    runtime = capture.get("runtime_arguments", {})
    for key, expected in EXPECTED_RUNTIME.items():
        if runtime.get(key) != expected:
            raise ValueError(f"Runtime contract differs for {key}: {runtime.get(key)!r}")
    if verification.get("size_diagnostic") != "completed":
        raise ValueError("COCO/XML size diagnostic was not completed")


def ensure_output_absent(output):
    if output.exists():
        raise FileExistsError(f"Output exists; inspect it and choose a new version: {output}")


def round_plan(output):
    return [{"round": round_id, "engine_repeat": engine_id,
             "run_dir": output / f"round_{round_id}" / f"engine_{engine_id}"}
            for round_id, engines in enumerate(ROUND_ORDER, start=1)
            for engine_id in engines]


def confirmation_args(confirmations):
    return [item for pid, path in sorted(confirmations.items())
            for item in ("--confirm-desktop-process", f"{pid}={path}")]


def capture_command(repo, source_root, engine_id, run_dir, device, confirmations):
    return [sys.executable, str(repo / "scripts/capture_cctsdb_validator.py"),
            "--repeat-study", str(source_root), "--repeat-index", str(engine_id),
            "--out-dir", str(run_dir / "capture"), "--device", device,
            *confirmation_args(confirmations)]


def verification_command(repo, capture_dir, out_dir, xml_path):
    return [sys.executable, str(repo / "scripts/verify_cctsdb_capture.py"),
            "--capture-dir", str(capture_dir), "--xml", str(xml_path),
            "--out-dir", str(out_dir)]


def size_convention(size, reference):
    for key in ("metric_id", "rules", "evaluator_source_sha256", "xml_sha256"):
        if size.get(key) != reference.get(key):
            raise ValueError(f"Size evaluator convention differs for {key}")
    for label in SIZE_LABELS:
        if size["metrics"][label]["instances"] != reference["metrics"][label]["instances"]:
            raise ValueError(f"Size instance count differs for {label}")


def _run_child(command, repo):
    print("START: " + " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=repo)
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Child command failed with exit {return_code}: {' '.join(command)}")
    return process.pid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confirm-desktop-process", action="append", default=[], metavar="PID=PATH",
                        help="Explicitly confirm a current nvidia-smi desktop row; no /proc access is used")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    output = args.out_dir.resolve()
    source_root = (repo / "results/measurement_audit_v1" / SOURCE_STUDY).resolve()
    expected_output = (repo / "results/measurement_audit_v1" / STUDY).resolve()
    if output != expected_output:
        parser.error(f"Output must be exactly {expected_output}")
    ensure_output_absent(output)

    # Heavy/runtime imports stay in main so all local unit tests remain TensorRT-free.
    from capture_cctsdb_validator import write_json
    from run_g0_int8_capture import check_reference, check_same_targets
    from uniform_build_repeat import (environment, parse_desktop_confirmations,
                                      repeat_capture_inputs)
    from audit_cctsdb_measurement import load_records, load_xml

    confirmations = parse_desktop_confirmations(args.confirm_desktop_process)
    source_manifest = read(source_root / "study_manifest.json")
    if source_manifest.get("study") != SOURCE_STUDY:
        parser.error("Source is not the approved Uniform build-repeat study")
    current_environment = environment()
    if current_environment != source_manifest.get("environment"):
        parser.error("Runtime environment/GPU differs from the Step A engine study; stop for review")

    engines = {}
    common_cache = None
    for engine_id in ENGINE_REPEATS:
        manifest = read(source_root / f"repeat_{engine_id}/build_manifest.json")
        engine_path, _provenance, _previous, _old, _prov, digest = repeat_capture_inputs(repo, source_root, engine_id)
        validate_engine_manifest(manifest, engine_id, source_manifest, digest)
        if not engine_path.is_file() or sha256(engine_path) != manifest["engine_sha256"]:
            parser.error(f"Repeat {engine_id} engine bytes/hash are not available or differ from manifest")
        cache = manifest.get("calibration_cache_sha256")
        if common_cache is None:
            common_cache = cache
        elif cache != common_cache:
            parser.error("Step A engines do not share one calibration cache hash")
        if not manifest.get("cache_matches_historical_uniform"):
            parser.error(f"Repeat {engine_id} does not carry the approved historical Uniform cache identity")
        engines[engine_id] = {"engine_sha256": digest, "manifest": manifest}

    xml = (repo / "../nighttime-tsd/data/raw/CCTSDB2021/xml.zip").resolve()
    reference_capture = repo / "results/measurement_audit_v1/server_fp16_capture_v1"
    reference_verification = repo / "results/measurement_audit_v1/server_native_size_v1"
    reference_report, reference_size = check_reference(reference_capture, reference_verification, xml)
    reference_predictions = read(reference_capture / "validator_predictions.json")
    load_xml(xml, load_records(reference_predictions))

    plan = round_plan(output)
    output.mkdir(parents=True)
    write_json(output / "study_manifest.json", {
        "schema_version": 1,
        "study": STUDY,
        "source_study": SOURCE_STUDY,
        "source_study_root": str(source_root),
        "source_study_commit": source_manifest.get("git_commit"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                                      capture_output=True, text=True, check=True).stdout.strip(),
        "dataset_split": DATASET_SPLIT,
        "engine_repeats": {str(i): engines[i]["engine_sha256"] for i in ENGINE_REPEATS},
        "round_order": [list(order) for order in ROUND_ORDER],
        "capture_contract": EXPECTED_RUNTIME,
        "prediction_payload_version": PREDICTION_PAYLOAD_VERSION,
        "reference_capture_sha256": sha256(reference_capture / "capture_report.json"),
        "reference_size_sha256": sha256(reference_verification / "size_coco_xml.json"),
        "environment": current_environment,
        "operator_confirmations": [{"pid": pid, "reported_path": path}
                                    for pid, path in sorted(confirmations.items())],
        "process_policy": "one fresh capture subprocess per round/engine; verification is a separate subprocess",
        "scope": "Inference repeatability only. No build, export, calibration, training, benchmark or policy selection. Three captures per existing engine; no auto-resume.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })

    records = []
    baseline_by_engine = {}
    for item in plan:
        round_id, engine_id, run_dir = item["round"], item["engine_repeat"], item["run_dir"]
        run_dir.mkdir(parents=True)
        capture_dir, verification_dir = run_dir / "capture", run_dir / "verification"
        capture_cmd = capture_command(repo, source_root, engine_id, run_dir, args.device, confirmations)
        capture_pid = _run_child(capture_cmd, repo)
        verify_cmd = verification_command(repo, capture_dir, verification_dir, xml)
        verify_pid = _run_child(verify_cmd, repo)

        capture = read(capture_dir / "capture_report.json")
        predictions = read(capture_dir / "validator_predictions.json")
        verification = read(verification_dir / "verification_summary.json")
        size = read(verification_dir / "size_coco_xml.json")
        validate_capture_contract(capture, verification)
        expected_engine = engines[engine_id]["engine_sha256"]
        if capture.get("model_sha256") != expected_engine or verification.get("model_sha256") != expected_engine:
            raise ValueError(f"Round {round_id} engine {engine_id} model identity mismatch")
        if sha256(capture_dir / "validator_predictions.json") != capture.get("predictions_sha256"):
            raise ValueError(f"Round {round_id} engine {engine_id} prediction file hash mismatch")
        if verification.get("capture_hash_match") != "exact_bytes":
            raise ValueError(f"Round {round_id} engine {engine_id} prediction hash was not exact bytes")
        if verification.get("capture_prediction_sha256") != capture.get("predictions_sha256"):
            raise ValueError(f"Round {round_id} engine {engine_id} verification prediction hash mismatch")
        if verification.get("capture_report_sha256") != sha256(capture_dir / "capture_report.json"):
            raise ValueError(f"Round {round_id} engine {engine_id} capture report hash mismatch")
        check_same_targets(reference_predictions, predictions)
        size_convention(size, reference_size)
        payload_hash = prediction_payload_hash(predictions)
        previous = baseline_by_engine.get(engine_id)
        if previous is None:
            if round_id != 1:
                raise ValueError(f"Engine {engine_id} did not appear in required round 1")
            payload_comparison = {"exact": True, "differences": {}}
            metrics_exact = True
            metrics_delta = {"ultralytics": {}, "coco_xml": {}}
            baseline_by_engine[engine_id] = {"predictions": predictions, "capture_report": capture, "size_report": size,
                                             "prediction_payload_sha256": payload_hash}
        else:
            payload_comparison = prediction_difference(previous["predictions"], predictions)
            metrics_exact = (previous["capture_report"]["metrics"] == capture["metrics"] and
                             previous["size_report"]["metrics"] == size["metrics"])
            metrics_delta = {"ultralytics": numeric_deltas(previous["capture_report"]["metrics"], capture["metrics"]),
                             "coco_xml": numeric_deltas(previous["size_report"]["metrics"], size["metrics"])}
        run = {"round": round_id, "engine_repeat": engine_id, "engine_sha256": expected_engine,
               "capture_process_pid": capture_pid, "verification_process_pid": verify_pid,
               "capture_command": capture_cmd, "verification_command": verify_cmd,
               "prediction_payload_version": PREDICTION_PAYLOAD_VERSION,
               "prediction_payload_sha256": payload_hash,
               "prediction_file_sha256": capture["predictions_sha256"],
               "prediction_payload_exact_vs_round_1": payload_comparison["exact"],
               "prediction_difference_vs_round_1": payload_comparison["differences"],
               "metrics_exact_vs_round_1": metrics_exact, "metrics_delta_vs_round_1": metrics_delta,
               "capture_report": capture, "size_report": size,
               "verification": {"native_matching_status": verification["native_matching_status"],
                                "size_diagnostic": verification["size_diagnostic"],
                                "global_g0": verification["global_g0"],
                                "capture_hash_match": verification["capture_hash_match"]}}
        records.append(run)
        write_json(run_dir / "execution_manifest.json", {
            "schema_version": 1, "study": STUDY, "round": round_id, "engine_repeat": engine_id,
            "engine_sha256": expected_engine, "capture_process_pid": capture_pid,
            "verification_process_pid": verify_pid, "capture_command": capture_cmd,
            "verification_command": verify_cmd, "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_json(run_dir / "comparison.json", run)
        print(f"FINISHED round {round_id}/3 engine {engine_id}", flush=True)

    aggregates = {str(engine_id): aggregate_within_engine([
        row for row in records if row["engine_repeat"] == engine_id]) for engine_id in ENGINE_REPEATS}
    review_flags = []
    for row in records:
        if not row["prediction_payload_exact_vs_round_1"]:
            review_flags.append(f"engine_{row['engine_repeat']}_round_{row['round']}_prediction_payload_changed")
        if not row["metrics_exact_vs_round_1"]:
            review_flags.append(f"engine_{row['engine_repeat']}_round_{row['round']}_metrics_changed")
        for phase in ("gpu_before", "gpu_after"):
            snapshot = row["capture_report"].get(phase)
            guard = snapshot.get("process_guard", {}) if isinstance(snapshot, dict) else {}
            if not isinstance(snapshot, dict) or guard.get("telemetry_status") != "complete":
                review_flags.append(f"engine_{row['engine_repeat']}_round_{row['round']}_{phase}_telemetry_limited")
            if guard.get("external_workload_detected") or row["capture_report"].get("external_gpu_workload_detected"):
                review_flags.append(f"engine_{row['engine_repeat']}_round_{row['round']}_external_workload_or_unverified_process")
    write_json(output / "repeat_summary.json", {
        "schema_version": 1, "study": STUDY, "source_study": SOURCE_STUDY,
        "status": "inference_repeatability_completed_review_required", "global_g0": "review_required",
        "round_order": [list(order) for order in ROUND_ORDER], "records": records,
        "within_engine": aggregates, "review_flags": sorted(set(review_flags)),
        "next_action": "STOP for review; do not select a capture or continue to B/C, timing, calibration or scale-up.",
        "limitations": "Three captures per existing engine are a finite diagnostic, not a population estimate or latency benchmark. Exact payload hash excludes timestamps and paths but preserves image order, detection order, classes, boxes, confidence and coordinate contract. Telemetry is sampled before/after each capture and does not prove GPU isolation or exclude workload between snapshots. No inference repeat can attribute Step A build differences to one tactic or precision mechanism.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"DONE: {output / 'repeat_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
