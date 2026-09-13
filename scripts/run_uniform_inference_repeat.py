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
SOURCE_STUDY_DIR = "server_uniform_build_repeat_v1"
STUDY_DIR = "server_uniform_inference_repeat_v1"
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


def _validate_finite_json(value, path="payload"):
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite numeric value in {path}")


def _validate_prediction_records(records):
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Prediction record {index} is not an object")
        image = record.get("image")
        if not isinstance(image, str) or "/" in image or "\\" in image:
            raise ValueError(f"Prediction record {index} has invalid image ID")
        shape = record.get("orig_shape")
        if (not isinstance(shape, list) or len(shape) != 2 or
                any(type(value) is not int or value <= 0 for value in shape)):
            raise ValueError(f"Prediction record {index} has invalid orig_shape")
        arrays = [record.get(key) for key in ("xyxy", "confidence", "class_id")]
        if any(not isinstance(value, list) for value in arrays):
            raise ValueError(f"Prediction record {index} has malformed detection arrays")
        if len({len(value) for value in arrays}) != 1:
            raise ValueError(f"Prediction record {index} detection arrays have different lengths")
        for box in record["xyxy"]:
            if (not isinstance(box, list) or len(box) != 4 or
                    any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in box) or
                    box[2] < box[0] or box[3] < box[1]):
                raise ValueError(f"Prediction record {index} has malformed bbox")
        for score in record["confidence"]:
            if (not isinstance(score, (int, float)) or isinstance(score, bool) or
                    not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError(f"Prediction record {index} has malformed confidence")
        for class_id in record["class_id"]:
            if type(class_id) is not int or class_id not in (0, 1, 2):
                raise ValueError(f"Prediction record {index} has malformed class")
        for key in ("validator_input", "validator_statistics"):
            if not isinstance(record.get(key), dict):
                raise ValueError(f"Prediction record {index} is missing {key}")
    _validate_finite_json(records, "records")


def prediction_payload(capture):
    """Return the versioned output payload; timestamps, paths and engine identity are excluded."""
    required = ("schema_version", "capture_mode", "iou_thresholds", "records", "coordinate_contract")
    missing = [key for key in required if key not in capture]
    if missing:
        raise ValueError(f"Prediction payload missing fields: {missing}")
    if not isinstance(capture["records"], list) or not capture["records"]:
        raise ValueError("Prediction payload has no records")
    _validate_prediction_records(capture["records"])
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


def resolve_protocol_paths(repo):
    """Resolve the logical study IDs to the exact approved server directories."""
    audit_root = Path(repo).resolve() / "results/measurement_audit_v1"
    return audit_root / SOURCE_STUDY_DIR, audit_root / STUDY_DIR


def validate_output_target(repo, output):
    source_root, expected_output = resolve_protocol_paths(repo)
    if Path(output).resolve() != expected_output:
        raise ValueError(f"Output must be exactly {expected_output}")
    return source_root, expected_output


def parse_gpu_identity(snapshot_or_device):
    """Extract the stable UUID/name/driver fields from an nvidia-smi device snapshot."""
    if isinstance(snapshot_or_device, dict):
        raw = snapshot_or_device.get("device")
    else:
        raw = snapshot_or_device
    if not isinstance(raw, str):
        raise ValueError("GPU device telemetry is missing")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("GPU device telemetry must contain exactly one device row")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) < 3 or not fields[0] or not fields[1] or not fields[2]:
        raise ValueError(f"GPU device telemetry is malformed: {raw!r}")
    return {"uuid": fields[0], "name": fields[1], "driver_version": fields[2]}


def validate_gpu_identity(current_snapshot, study_manifest):
    expected = parse_gpu_identity(study_manifest.get("gpu_before"))
    actual = parse_gpu_identity(current_snapshot)
    mismatches = {key: {"expected": expected[key], "actual": actual[key]}
                  for key in ("uuid", "name", "driver_version") if expected[key] != actual[key]}
    if mismatches:
        raise ValueError(f"GPU identity differs from Step A snapshot: {mismatches}")
    return {"expected_step_a": expected, "current": actual, "matched": True}


def validate_device_argument(device):
    if str(device) != "0":
        raise ValueError("Inference repeatability protocol is locked to --device 0")


def validate_run_gpu_snapshots(capture, study_manifest):
    binding = {}
    for phase in ("gpu_before", "gpu_after"):
        snapshot = capture.get(phase)
        binding[phase] = validate_gpu_identity(snapshot, study_manifest)
    return binding


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


def environment_probe_command(repo):
    return [sys.executable, str(Path(repo) / "scripts/probe_inference_environment.py")]


def parse_environment_probe(stdout, stderr, return_code):
    """Parse one JSON child response; reject warnings or text mixed into stdout."""
    if return_code:
        try:
            failure = json.loads(stdout)
        except (TypeError, json.JSONDecodeError):
            failure = None
        detail = failure.get("error") if isinstance(failure, dict) else None
        detail = detail or stderr.strip() or "no structured error"
        raise RuntimeError(f"Environment preflight child failed with exit {return_code}: {detail}")
    try:
        report = json.loads(stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("Environment preflight child did not return pure JSON") from error
    if (not isinstance(report, dict) or report.get("schema_version") != 1 or
            report.get("status") != "ok" or not isinstance(report.get("environment"), dict)):
        raise RuntimeError(f"Environment preflight child returned an invalid report: {report!r}")
    return {"status": "ok", "environment": report["environment"],
            "child": {"return_code": return_code,
                      "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
                      "stderr": stderr}}


def run_environment_preflight(repo, runner=None):
    """Run CUDA-touching environment() in a short-lived child and wait for it."""
    runner = subprocess.run if runner is None else runner
    completed = runner(environment_probe_command(repo), cwd=repo, capture_output=True,
                       text=True, check=False)
    return parse_environment_probe(completed.stdout, completed.stderr, completed.returncode)


def run_capture_pair(repo, source_root, engine_id, run_dir, device, confirmations,
                     xml_path, run_child=None):
    """Run capture, wait, then run verification in a separate child process."""
    run_child = _run_child if run_child is None else run_child
    capture_dir, verification_dir = run_dir / "capture", run_dir / "verification"
    capture_cmd = capture_command(repo, source_root, engine_id, run_dir, device, confirmations)
    capture_pid = run_child(capture_cmd, repo)
    verify_cmd = verification_command(repo, capture_dir, verification_dir, xml_path)
    verify_pid = run_child(verify_cmd, repo)
    return capture_pid, verify_pid, capture_cmd, verify_cmd


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


def main(argv=None, *, repo_override=None, helpers=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confirm-desktop-process", action="append", default=[], metavar="PID=PATH",
                        help="Explicitly confirm a current nvidia-smi desktop row; no /proc access is used")
    args = parser.parse_args(argv)
    repo = Path(repo_override if repo_override is not None else Path(__file__).resolve().parents[1]).resolve()
    output = args.out_dir.resolve()
    try:
        source_root, expected_output = validate_output_target(repo, output)
        validate_device_argument(args.device)
    except ValueError as error:
        parser.error(str(error))
    ensure_output_absent(output)

    # CUDA-touching environment() runs only in the short-lived child. The parent
    # receives a parsed report before importing orchestration helpers.
    helpers = dict(helpers or {})
    preflight = helpers.get("environment_preflight", run_environment_preflight)(repo)
    if (not isinstance(preflight, dict) or preflight.get("status") != "ok" or
            not isinstance(preflight.get("environment"), dict)):
        raise RuntimeError("Environment preflight did not return a usable report")

    # Heavy/runtime imports stay after the child preflight so the parent never
    # calls the CUDA-touching environment helper itself.
    if "write_json" not in helpers:
        from capture_cctsdb_validator import write_json
        helpers["write_json"] = write_json
    if "check_reference" not in helpers or "check_same_targets" not in helpers:
        from run_g0_int8_capture import check_reference, check_same_targets
        helpers.setdefault("check_reference", check_reference)
        helpers.setdefault("check_same_targets", check_same_targets)
    if any(key not in helpers for key in ("ensure_idle", "parse_desktop_confirmations",
                                          "repeat_capture_inputs", "snapshot")):
        from uniform_build_repeat import (ensure_idle, parse_desktop_confirmations,
                                          repeat_capture_inputs, snapshot)
        helpers.setdefault("ensure_idle", ensure_idle)
        helpers.setdefault("parse_desktop_confirmations", parse_desktop_confirmations)
        helpers.setdefault("repeat_capture_inputs", repeat_capture_inputs)
        helpers.setdefault("snapshot", snapshot)
    if "load_records" not in helpers or "load_xml" not in helpers:
        from audit_cctsdb_measurement import load_records, load_xml
        helpers.setdefault("load_records", load_records)
        helpers.setdefault("load_xml", load_xml)

    write_json = helpers["write_json"]
    check_reference = helpers["check_reference"]
    check_same_targets = helpers["check_same_targets"]
    ensure_idle = helpers["ensure_idle"]
    parse_desktop_confirmations = helpers["parse_desktop_confirmations"]
    repeat_capture_inputs = helpers["repeat_capture_inputs"]
    snapshot = helpers["snapshot"]
    load_records = helpers["load_records"]
    load_xml = helpers["load_xml"]
    run_child = helpers.get("run_child", _run_child)

    confirmations = parse_desktop_confirmations(args.confirm_desktop_process)
    source_manifest = read(source_root / "study_manifest.json")
    if source_manifest.get("study") != SOURCE_STUDY:
        parser.error("Source is not the approved Uniform build-repeat study")
    current_environment = preflight["environment"]
    if current_environment != source_manifest.get("environment"):
        parser.error("Runtime environment/GPU differs from the Step A engine study; stop for review")
    current_gpu_before = snapshot(confirmations)
    ensure_idle(current_gpu_before)
    try:
        gpu_binding = validate_gpu_identity(current_gpu_before, source_manifest)
    except ValueError as error:
        parser.error(str(error))

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
        "environment_preflight": preflight,
        "source_step_a_gpu_before": source_manifest.get("gpu_before"),
        "source_step_a_gpu_after": source_manifest.get("gpu_after"),
        "gpu_before": current_gpu_before,
        "gpu_identity_binding": {"device_argument": args.device, **gpu_binding},
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
        capture_pid, verify_pid, capture_cmd, verify_cmd = run_capture_pair(
            repo, source_root, engine_id, run_dir, args.device, confirmations, xml, run_child)

        capture = read(capture_dir / "capture_report.json")
        predictions = read(capture_dir / "validator_predictions.json")
        verification = read(verification_dir / "verification_summary.json")
        size = read(verification_dir / "size_coco_xml.json")
        validate_capture_contract(capture, verification)
        try:
            capture_gpu_binding = validate_run_gpu_snapshots(capture, source_manifest)
        except ValueError as error:
            raise RuntimeError(f"Round {round_id} engine {engine_id} GPU identity check failed: {error}") from error
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
               "gpu_identity_binding": capture_gpu_binding,
               "prediction_file_sha256": capture["predictions_sha256"],
               "prediction_payload_exact_vs_round_1": payload_comparison["exact"],
               "prediction_difference_vs_round_1": payload_comparison["differences"],
               "metrics_exact": metrics_exact, "metrics_exact_vs_round_1": metrics_exact,
               "metrics_delta_vs_round_1": metrics_delta,
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
