#!/usr/bin/env python3
"""Build and evaluate the locked YOLO11n precision-head ablation study."""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from audit_cctsdb_measurement import load_records, sha256
from run_architecture_matrix import GpuPhaseLock
from run_uniform_inference_repeat import (
    DATASET_SPLIT,
    EXPECTED_RUNTIME,
    PREDICTION_PAYLOAD_VERSION,
    metric_aggregate,
    numeric_deltas,
    parse_gpu_identity,
    prediction_difference,
    prediction_payload_hash,
    size_convention,
    validate_capture_contract,
    validate_gpu_identity,
)
from run_uniform_timing_cache_replay import (
    CACHE_COVERAGE_UNKNOWN,
    FROZEN_WEIGHTS_PATH,
    FROZEN_WEIGHTS_SHA256,
    REPLAY_SETTINGS,
    SOURCE_CALIBRATION_CACHE_SHA256,
    SOURCE_CODE_COMMIT,
    SOURCE_ONNX_SHA256,
    SOURCE_RESULT_COMMIT,
    SOURCE_STUDY,
    SOURCE_TIMING_CACHE_SHA256,
    CalibrationCacheAudit,
    attach_timing_cache,
    confirmation_args,
    environment_preflight,
    validate_source_contract,
)


STUDY = "yolo11n_precision_head_ablation_v1"
STUDY_ATTEMPT = 2
PREVIOUS_STUDY_DIR = "server_yolo11n_precision_head_ablation_v1"
STUDY_DIR = "server_yolo11n_precision_head_ablation_v1_attempt2"
PREVIOUS_ATTEMPT_RELATIVE_PATH = (
    f"results/measurement_audit_v1/{PREVIOUS_STUDY_DIR}")
EXECUTION_REASON = "implementation_fix_non_convolution_namespace_selection"
BASELINE_STUDY = "uniform_timing_cache_replay_v1"
BASELINE_STUDY_DIR = "server_uniform_timing_cache_replay_v1"
BASELINE_RESULT_COMMIT = "3797075c934ca5f88c0d64b998c38def0deef49f"
REPLAY_VARIANT = "precision_head_ablation_common_timing_cache_v1"
ARMS = ("baseline_int8", "bbox_fp32", "classification_fp32", "both_fp32")
REPEATS = (1, 2, 3)
BRANCH_PREFIXES = {
    "bbox_fp32": ("/model.23/cv2.",),
    "classification_fp32": ("/model.23/cv3.",),
    "both_fp32": ("/model.23/cv2.", "/model.23/cv3."),
}
ENDPOINTS = (("full", "ultralytics"), ("xs", "coco_xml"), ("s", "coco_xml"))
SIZE_LABELS = ("all", "xs", "s", "m", "l", "xl")
SIZE_METRICS = ("map50", "map50_95")
FULL_METRICS = ("map50", "map50_95", "precision", "recall")
EXPECTED_BUILDER_FLAGS = 514


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, allow_nan=False)
        handle.write("\n")


def write_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)


def git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                            text=True, check=True)
    return result.stdout.strip()


def resolve_protocol_paths(repo: Path):
    audit_root = Path(repo).resolve() / "results/measurement_audit_v1"
    return audit_root / "server_uniform_build_repeat_v1", audit_root / STUDY_DIR


def validate_output_target(repo: Path, output: Path):
    source, expected = resolve_protocol_paths(repo)
    if Path(output).resolve() != expected:
        raise ValueError(f"Output must be exactly {expected}")
    return source, expected


def build_plan():
    return [{"sequence": sequence, "arm": arm, "repeat": repeat}
            for sequence, (arm, repeat) in enumerate(
                ((arm, repeat) for arm in ARMS for repeat in REPEATS), start=1)]


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def validate_ablation_sources(repo: Path):
    """Validate Step-A inputs and the accepted timing-replay reference."""
    source = validate_source_contract(repo)
    baseline_root = Path(repo).resolve() / "results/measurement_audit_v1" / BASELINE_STUDY_DIR
    baseline_study_path = baseline_root / "study_manifest.json"
    baseline_build_path = baseline_root / "repeat_1/build_manifest.json"
    baseline_capture_path = baseline_root / "repeat_1/capture/capture_report.json"
    baseline_predictions_path = baseline_root / "repeat_1/capture/validator_predictions.json"
    baseline_size_path = baseline_root / "repeat_1/verification/size_coco_xml.json"
    baseline_summary_path = baseline_root / "repeat_summary.json"
    for path, label in (
        (baseline_study_path, "accepted timing-replay study manifest"),
        (baseline_build_path, "accepted timing-replay repeat_1 build manifest"),
        (baseline_capture_path, "accepted timing-replay repeat_1 capture report"),
        (baseline_predictions_path, "accepted timing-replay repeat_1 predictions"),
        (baseline_size_path, "accepted timing-replay repeat_1 size report"),
        (baseline_summary_path, "accepted timing-replay summary"),
    ):
        _require_file(path, label)
    baseline_study = read(baseline_study_path)
    baseline_build = read(baseline_build_path)
    baseline_capture = read(baseline_capture_path)
    baseline_predictions = read(baseline_predictions_path)
    baseline_size = read(baseline_size_path)
    baseline_summary = read(baseline_summary_path)
    if baseline_study.get("study") != BASELINE_STUDY:
        raise ValueError("Precision ablation baseline reference is not timing-cache replay")
    if baseline_summary.get("classification") != "replay_exact_observed":
        raise ValueError("Precision ablation requires the accepted exact timing-replay reference")
    if baseline_build.get("study") != BASELINE_STUDY or baseline_build.get("repeat") != 1:
        raise ValueError("Timing-replay reference build identity mismatch")
    source_build = source["build_manifests"][1]
    if source_build.get("builder_flags") != EXPECTED_BUILDER_FLAGS or not isinstance(
            source_build.get("sigmoid_fp32_constraints"), list):
        raise ValueError("Step-A builder flags/Sigmoid constraint baseline is incomplete")
    if baseline_build.get("settings") != REPLAY_SETTINGS:
        raise ValueError("Timing-replay reference settings differ from the accepted replay contract")
    if baseline_build.get("builder_flags") != source_build.get("builder_flags"):
        raise ValueError("Timing-replay reference builder flags differ from Step A")
    for key, expected in (("source_result_commit", SOURCE_RESULT_COMMIT),
                          ("source_study_code_commit", SOURCE_CODE_COMMIT),
                          ("source_weights_sha256", FROZEN_WEIGHTS_SHA256),
                          ("onnx_sha256", SOURCE_ONNX_SHA256),
                          ("calibration_cache_input_sha256", SOURCE_CALIBRATION_CACHE_SHA256),
                          ("timing_cache_input_sha256", SOURCE_TIMING_CACHE_SHA256)):
        if baseline_build.get(key) != expected:
            raise ValueError(f"Timing-replay reference differs for {key}")
    if baseline_build.get("timing_cache_attach") != {
            "called": True, "ignore_mismatch": False, "return_value": True}:
        raise ValueError("Timing-replay reference did not attach timing cache without fallback")
    if baseline_capture.get("status") != "pass":
        raise ValueError("Timing-replay reference capture did not pass")
    if baseline_capture.get("predictions_sha256") != sha256(baseline_predictions_path):
        raise ValueError("Timing-replay reference prediction hash mismatch")
    prediction_payload_hash(baseline_predictions)
    return {
        **source,
        "baseline_root": baseline_root,
        "baseline_study": baseline_study,
        "baseline_study_path": baseline_study_path,
        "baseline_build": baseline_build,
        "baseline_build_path": baseline_build_path,
        "baseline_capture": baseline_capture,
        "baseline_capture_path": baseline_capture_path,
        "baseline_predictions": baseline_predictions,
        "baseline_predictions_path": baseline_predictions_path,
        "baseline_size": baseline_size,
        "baseline_size_path": baseline_size_path,
        "baseline_summary_path": baseline_summary_path,
        "baseline_result_commit": BASELINE_RESULT_COMMIT,
    }


def _expected_calibration_input():
    return {"source_repeat": 1, "sha256": SOURCE_CALIBRATION_CACHE_SHA256,
            "copy_per_build": True}


def _expected_timing_input():
    return {"source_repeat": 1, "sha256": SOURCE_TIMING_CACHE_SHA256,
            "copy_per_build": True, "ignore_mismatch": False}


def expected_study_manifest(source: dict, *, environment=None, gpu_before=None,
                            gpu_binding=None, confirmations=None, repo=None,
                            execution_git_commit=None, runner_script_sha256=None):
    if execution_git_commit is None:
        execution_git_commit = source.get("execution_git_commit", "test")
    if runner_script_sha256 is None:
        runner_script_sha256 = source.get("runner_script_sha256", sha256(Path(__file__)))
    return {
        "schema_version": 1,
        "study": STUDY,
        "execution_attempt": STUDY_ATTEMPT,
        "previous_attempt_path": PREVIOUS_ATTEMPT_RELATIVE_PATH,
        "execution_reason": EXECUTION_REASON,
        "execution_git_commit": execution_git_commit,
        "runner_script_sha256": runner_script_sha256,
        "source_study": SOURCE_STUDY,
        "source_result_commit": SOURCE_RESULT_COMMIT,
        "source_study_code_commit": source["manifest"].get("git_commit"),
        "source_study_manifest_sha256": sha256(source["manifest_path"]),
        "baseline_reference_study": BASELINE_STUDY,
        "baseline_reference_result_commit": BASELINE_RESULT_COMMIT,
        "baseline_reference_study_manifest_sha256": sha256(source["baseline_study_path"]),
        "source_weights_sha256": FROZEN_WEIGHTS_SHA256,
        "frozen_weights_path": FROZEN_WEIGHTS_PATH.as_posix(),
        "frozen_weights_measured_sha256": source["input_hashes"]["weights_sha256"],
        "onnx_sha256": SOURCE_ONNX_SHA256,
        "calibration_cache_input": _expected_calibration_input(),
        "timing_cache_input": _expected_timing_input(),
        "settings": REPLAY_SETTINGS,
        "builder_flags": source["build_manifests"][1].get("builder_flags"),
        "sigmoid_fp32_constraints": source["build_manifests"][1].get("sigmoid_fp32_constraints"),
        "arms": list(ARMS),
        "repeats_per_arm": list(REPEATS),
        "build_order": build_plan(),
        "capture_order": build_plan(),
        "dataset_split": DATASET_SPLIT,
        "dev_images": 1636,
        "dev_instances": 2706,
        "protocol_variant": REPLAY_VARIANT,
        "environment": environment,
        "gpu_before": gpu_before,
        "gpu_identity_binding": gpu_binding,
        "operator_confirmations": [
            {"pid": pid, "reported_path": path}
            for pid, path in sorted((confirmations or {}).items())
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "YOLO11n frozen Uniform-calibrated dev-only precision-head diagnostic; no retraining, official test, or device benchmark.",
    }


def validate_ablation_study_manifest(manifest, source: dict) -> None:
    expected = expected_study_manifest(source)
    exact_keys = (
        "study", "execution_attempt", "previous_attempt_path", "execution_reason",
        "source_study", "source_result_commit", "source_study_code_commit",
        "source_study_manifest_sha256", "baseline_reference_study",
        "baseline_reference_result_commit", "baseline_reference_study_manifest_sha256",
        "source_weights_sha256", "frozen_weights_path", "frozen_weights_measured_sha256",
        "onnx_sha256", "calibration_cache_input", "timing_cache_input", "settings",
        "builder_flags", "sigmoid_fp32_constraints", "arms", "repeats_per_arm",
        "build_order", "capture_order", "dataset_split", "dev_images", "dev_instances",
        "protocol_variant",
    )
    for key in exact_keys:
        if manifest.get(key) != expected[key]:
            raise ValueError(f"Precision ablation study contract differs for {key}")
    execution_commit = manifest.get("execution_git_commit")
    if not isinstance(execution_commit, str) or not execution_commit.strip():
        raise ValueError("Precision ablation study execution git commit is missing")
    runner_hash = manifest.get("runner_script_sha256")
    if (not isinstance(runner_hash, str) or len(runner_hash) != 64 or
            any(char not in "0123456789abcdef" for char in runner_hash)):
        raise ValueError("Precision ablation study runner script hash is invalid")


def _normalize_layer_specs(layer_specs):
    normalized = []
    seen = set()
    for item in layer_specs:
        if isinstance(item, dict):
            name, is_conv = item.get("name"), item.get("is_convolution")
        else:
            name, is_conv = item
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Network layer names must be non-empty and unique")
        if not isinstance(is_conv, bool):
            raise ValueError(f"Layer type is missing for {name}")
        normalized.append((name, is_conv))
        seen.add(name)
    return normalized


def select_precision_layers(layer_specs, arm: str):
    """Select convolution layers under the exact branch prefix.

    A branch namespace also contains activation/other helper layers. Those
    nodes are candidates for audit but are not precision-head targets.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown precision-ablation arm: {arm}")
    specs = _normalize_layer_specs(layer_specs)
    prefixes = BRANCH_PREFIXES.get(arm, ())
    selected = []
    for name, is_convolution in specs:
        if any(name.startswith(prefix) for prefix in prefixes):
            if is_convolution:
                selected.append(name)
    if arm != "baseline_int8" and not selected:
        raise ValueError(f"No parsed convolution layers matched {arm} prefixes {prefixes}")
    return selected


def validate_selected_precision_layers(layer_specs, arm: str, selected_names):
    expected = select_precision_layers(layer_specs, arm)
    if list(selected_names) != expected:
        raise ValueError(f"Selected layer list differs for {arm}: expected {expected}, got {selected_names}")
    prefixes = BRANCH_PREFIXES.get(arm, ())
    if any(not any(name.startswith(prefix) for prefix in prefixes) for name in selected_names):
        raise ValueError(f"Selected layer outside locked prefix for {arm}")
    return expected


def validate_manifest_layer_names(arm: str, selected_names, selected_types):
    if arm not in ARMS:
        raise ValueError(f"Unknown precision-ablation arm: {arm}")
    if not isinstance(selected_names, list) or len(selected_names) != len(set(selected_names)):
        raise ValueError(f"{arm} matched layer names are missing or duplicated")
    if not isinstance(selected_types, dict) or set(selected_types) != set(selected_names):
        raise ValueError(f"{arm} matched layer types are incomplete")
    prefixes = BRANCH_PREFIXES.get(arm, ())
    if arm == "baseline_int8" and selected_names:
        raise ValueError("Baseline arm has unexpected head precision layers")
    for name in selected_names:
        if not any(name.startswith(prefix) for prefix in prefixes):
            raise ValueError(f"{arm} selected layer is outside its prefix: {name}")
        if selected_types[name] != "CONVOLUTION":
            raise ValueError(f"{arm} selected layer is not a convolution: {name}")


def validate_constraint_audit(constraint, arm: str, source_sigmoid_names) -> None:
    """Validate persisted requested/effective layer evidence without TensorRT imports."""
    validate_manifest_layer_names(arm, constraint.get("matched_layer_names", []),
                                 constraint.get("matched_layer_types", {}))
    if constraint.get("branch_prefixes") != list(BRANCH_PREFIXES.get(arm, ())):
        raise ValueError(f"{arm} branch prefix contract differs")
    if constraint.get("obey_precision_constraints") is not True:
        raise ValueError(f"{arm} OBEY constraint evidence is missing")
    requested = constraint.get("requested", {})
    effective = constraint.get("effective", {})
    selected = constraint.get("matched_layer_names", [])
    candidates = constraint.get("prefix_candidate_names", [])
    excluded = constraint.get("excluded_prefix_non_convolution_names", [])
    if (not isinstance(candidates, list) or len(candidates) != len(set(candidates)) or
            not isinstance(excluded, list) or len(excluded) != len(set(excluded)) or
            set(selected) | set(excluded) != set(candidates) or
            set(selected) & set(excluded)):
        raise ValueError(f"{arm} prefix candidate/effective layer evidence is incomplete")
    if arm == "baseline_int8" and candidates:
        raise ValueError("Baseline arm has unexpected prefix candidates")
    if set(requested) != set(selected) or set(effective) != set(selected):
        raise ValueError(f"{arm} precision constraint evidence is incomplete")
    for name in selected:
        if requested[name].get("precision") != "FP32" or any(
                value != "FP32" for value in requested[name].get("output_types", [])):
            raise ValueError(f"{arm} requested constraint is not FP32: {name}")
        if effective[name] != {"precision_fp32": True, "all_outputs_fp32": True}:
            raise ValueError(f"{arm} effective constraint is not FP32: {name}")
    if constraint.get("baseline_sigmoid_fp32_constraints") != source_sigmoid_names:
        raise ValueError(f"{arm} baseline Sigmoid constraints changed")


def validate_arm_layer_contract(manifests) -> None:
    """Check repeat stability and the locked bbox/classification set algebra."""
    names = {}
    for arm in ARMS:
        repeats = manifests.get(arm, {})
        if set(repeats) != set(REPEATS):
            raise ValueError(f"{arm} precision layer evidence is incomplete across repeats")
        values = [tuple(repeats[repeat]["constraint_audit"].get("matched_layer_names", []))
                  for repeat in REPEATS]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"{arm} precision layer list differs across repeats")
        candidates = [tuple(repeats[repeat]["constraint_audit"].get("prefix_candidate_names", []))
                     for repeat in REPEATS]
        excluded = [tuple(repeats[repeat]["constraint_audit"].get(
            "excluded_prefix_non_convolution_names", [])) for repeat in REPEATS]
        if any(value != candidates[0] for value in candidates[1:]) or any(
                value != excluded[0] for value in excluded[1:]):
            raise ValueError(f"{arm} prefix candidate layer list differs across repeats")
        names[arm] = set(values[0])
    if names["baseline_int8"]:
        raise ValueError("Baseline arm unexpectedly selected head layers")
    if names["both_fp32"] != names["bbox_fp32"] | names["classification_fp32"]:
        raise ValueError("both_fp32 layer set is not the bbox/classification union")
    if names["bbox_fp32"] & names["classification_fp32"]:
        raise ValueError("bbox and classification precision layer sets overlap")


def _dtype_text(value):
    return str(value)


def _is_fp32(value, trt):
    return _dtype_text(value) == _dtype_text(trt.float32)


def layer_constraint_state(layer, trt):
    output_types = []
    for output_index in range(layer.num_outputs):
        output_types.append(_dtype_text(layer.get_output_type(output_index)))
    return {"precision": _dtype_text(layer.precision), "output_types": output_types,
            "num_outputs": layer.num_outputs}


def apply_precision_constraints(network, trt, arm: str, source_sigmoid_names):
    layer_specs = []
    layers = []
    for layer_index in range(network.num_layers):
        layer = network.get_layer(layer_index)
        layers.append(layer)
        layer_specs.append({"name": layer.name,
                            "is_convolution": layer.type == trt.LayerType.CONVOLUTION})
    normalized_specs = _normalize_layer_specs(layer_specs)
    selected = select_precision_layers(normalized_specs, arm)
    prefixes = BRANCH_PREFIXES.get(arm, ())
    prefix_candidates = [name for name, _is_conv in normalized_specs
                         if any(name.startswith(prefix) for prefix in prefixes)]
    excluded_non_convolution = [name for name, is_conv in normalized_specs
                                if any(name.startswith(prefix) for prefix in prefixes)
                                and not is_conv]
    by_name = {layer.name: layer for layer in layers}
    before = {}
    after = {}
    requested = {}
    for name in selected:
        layer = by_name[name]
        before[name] = layer_constraint_state(layer, trt)
        requested[name] = {"precision": "FP32",
                           "output_types": ["FP32"] * layer.num_outputs}
        layer.precision = trt.float32
        for output_index in range(layer.num_outputs):
            layer.set_output_type(output_index, trt.float32)
        after[name] = layer_constraint_state(layer, trt)
    protected = []
    for layer in layers:
        if layer.type == trt.LayerType.ACTIVATION and "sigmoid" in layer.name.lower():
            layer.precision = trt.float32
            for output_index in range(layer.num_outputs):
                layer.set_output_type(output_index, trt.float32)
            protected.append(layer.name)
    if protected != source_sigmoid_names:
        raise ValueError("Baseline Sigmoid FP32 constraint list differs from Step A")
    effective = {
        name: {
            "precision_fp32": _is_fp32(after[name]["precision"], trt),
            "all_outputs_fp32": all(_is_fp32(value, trt) for value in after[name]["output_types"]),
        }
        for name in selected
    }
    if any(not state["precision_fp32"] or not state["all_outputs_fp32"]
           for state in effective.values()):
        raise RuntimeError("TensorRT did not retain an FP32 precision/output constraint")
    return {
        "arm": arm,
        "branch_prefixes": list(BRANCH_PREFIXES.get(arm, ())),
        "prefix_candidate_names": prefix_candidates,
        "excluded_prefix_non_convolution_names": excluded_non_convolution,
        "matched_layer_names": selected,
        "matched_layer_count": len(selected),
        "matched_layer_types": {name: "CONVOLUTION" for name in selected},
        "requested": requested,
        "before": before,
        "after": after,
        "effective": effective,
        "obey_precision_constraints": True,
        "baseline_sigmoid_fp32_constraints": protected,
    }


def preflight(repo: Path, source: dict, args, helpers):
    probe = helpers.get("environment_preflight", environment_preflight)(repo)
    if probe.get("status") != "ok" or not isinstance(probe.get("environment"), dict):
        raise RuntimeError("Environment preflight did not return a usable report")
    if probe["environment"] != source["manifest"].get("environment"):
        raise ValueError("Runtime environment/GPU differs from the Step-A engine study")
    if any(key not in helpers for key in ("ensure_idle", "snapshot")):
        from uniform_build_repeat import ensure_idle, snapshot
        helpers.setdefault("ensure_idle", ensure_idle)
        helpers.setdefault("snapshot", snapshot)
    current = helpers["snapshot"](args.confirmed_desktop)
    helpers["ensure_idle"](current)
    binding = validate_gpu_identity(current, {"gpu_before": source["manifest"].get("gpu_before")})
    return probe, current, binding


def _build_child_command(repo: Path, output: Path, arm: str, repeat: int, confirmations):
    return [sys.executable, str(repo / "scripts/run_yolo11n_precision_head_ablation.py"),
            "--phase", "build", "--out-dir", str(output), "--arm", arm,
            "--repeat", str(repeat), *confirmation_args(confirmations)]


def ablation_capture_command(repo: Path, study_root: Path, arm: str, repeat: int,
                             run_dir: Path, device, confirmations):
    return [sys.executable, str(repo / "scripts/capture_cctsdb_validator.py"),
            "--precision-head-ablation-study", str(study_root),
            "--ablation-arm", arm, "--repeat-index", str(repeat),
            "--out-dir", str(run_dir / "capture"), "--device", str(device),
            *confirmation_args(confirmations)]


def verification_command(repo: Path, capture_dir: Path, out_dir: Path, xml_path: Path):
    return [sys.executable, str(repo / "scripts/verify_cctsdb_capture.py"),
            "--capture-dir", str(capture_dir), "--xml", str(xml_path),
            "--out-dir", str(out_dir)]


def build(repo: Path, output: Path, arm: str, repeat: int, confirmations):
    """Build one child; imports CUDA/TensorRT only in this short-lived process."""
    source = validate_ablation_sources(repo)
    _, expected_output = resolve_protocol_paths(repo)
    if Path(output).resolve() != expected_output:
        raise ValueError(f"Precision ablation build must use exactly {expected_output}")
    study_manifest_path = Path(output) / "study_manifest.json"
    _require_file(study_manifest_path, "precision ablation study manifest")
    validate_ablation_study_manifest(read(study_manifest_path), source)
    if arm not in ARMS or repeat not in REPEATS:
        raise ValueError("Invalid precision ablation build identity")
    dest = Path(output) / arm / f"repeat_{repeat}"
    if dest.exists():
        raise FileExistsError(f"Precision ablation output exists; preserve it: {dest}")
    dest.mkdir(parents=True)
    calibration_bytes = source["calibration_cache"].read_bytes()
    timing_bytes = source["timing_cache"].read_bytes()
    write_bytes(dest / "calibration_cache_input.cache", calibration_bytes)
    write_bytes(dest / "timing_cache_input.cache", timing_bytes)

    from uniform_build_repeat import environment, ensure_idle, inspector_signature, snapshot
    import tensorrt as trt

    env = environment()
    if env != source["manifest"].get("environment"):
        raise ValueError("Build environment changed from Step A")
    source_build = source["build_manifests"][1]
    with GpuPhaseLock(repo / "results/architecture_matrix_v1/.gpu_phase.lock",
                      f"{STUDY}_{arm}_{repeat}"):
        before = snapshot(confirmations)
        ensure_idle(before)
        validate_gpu_identity(before, {"gpu_before": source["manifest"].get("gpu_before")})
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(source["source_onnx"])):
            raise ValueError([str(parser.get_error(i)) for i in range(parser.num_errors)])
        if network.num_inputs != 1 or tuple(network.get_input(0).shape) != (1, 3, 640, 640):
            raise ValueError("Unexpected frozen YOLO11n input contract")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, REPLAY_SETTINGS["workspace_bytes"])
        config.builder_optimization_level = REPLAY_SETTINGS["builder_optimization_level"]
        config.avg_timing_iterations = REPLAY_SETTINGS["avg_timing_iterations"]
        config.set_flag(trt.BuilderFlag.INT8)
        config.clear_flag(trt.BuilderFlag.FP16)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        timing_attach = attach_timing_cache(config, timing_bytes)

        calibrator_audit = CalibrationCacheAudit(calibration_bytes)

        class CacheOnlyCalibrator(trt.IInt8Calibrator):
            def __init__(self):
                super().__init__()

            def get_algorithm(self):
                return trt.CalibrationAlgoType.MINMAX_CALIBRATION

            def get_batch_size(self):
                return 1

            def read_calibration_cache(self):
                return calibrator_audit.read()

            def write_calibration_cache(self, data):
                output = calibrator_audit.write(data)
                index = calibrator_audit.write_calls
                filename = "calibration_cache_output.cache" if index == 1 else f"calibration_cache_output_{index}.cache"
                write_bytes(dest / filename, output)

            def get_batch(self, names):
                return calibrator_audit.record_batch()

        config.int8_calibrator = CacheOnlyCalibrator()
        constraint_audit = apply_precision_constraints(
            network, trt, arm, source_build.get("sigmoid_fp32_constraints"))
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("TensorRT build failed; inspect build log")
        calibrator_audit.validate()
        if int(config.flags) != source_build.get("builder_flags"):
            raise ValueError("Effective builder flags differ from Step A")
        timing_output = bytes(config.get_timing_cache().serialize())
        write_bytes(dest / "timing_cache_output.cache", timing_output)
        metadata = json.dumps(source["manifest"]["engine_metadata"]).encode("utf-8")
        write_bytes(dest / "model.engine", len(metadata).to_bytes(4, "little") + metadata + bytes(plan))
        with trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(plan)
            if engine is None:
                raise RuntimeError("Precision ablation engine failed deserialization")
            inspector = engine.create_engine_inspector()
            inspector_info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
            write_json(dest / "inspector.json", inspector_info)
            del inspector, engine
        after = snapshot(confirmations)
        ensure_idle(after)

    outputs = calibrator_audit.outputs
    if not outputs:
        calibration_output_status, calibration_output_hash, calibration_output_changed = "not_returned", None, None
    else:
        calibration_output_status = "same" if all(item == calibration_bytes for item in outputs) else "changed"
        calibration_output_hash = sha256(dest / "calibration_cache_output.cache")
        calibration_output_changed = any(item != calibration_bytes for item in outputs)
    inspector_info = read(dest / "inspector.json")
    manifest = {
        "schema_version": 1,
        "study": STUDY,
        "arm": arm,
        "repeat": repeat,
        "source_study": SOURCE_STUDY,
        "source_result_commit": SOURCE_RESULT_COMMIT,
        "source_result_commit_available": source.get("source_result_commit_available"),
        "source_study_code_commit": source["manifest"].get("git_commit"),
        "source_study_manifest_sha256": sha256(source["manifest_path"]),
        "source_summary_sha256": sha256(source["source_summary"]),
        "source_build_manifest_sha256": sha256(source["root"] / "repeat_1/build_manifest.json"),
        "study_manifest_sha256": sha256(study_manifest_path),
        "baseline_reference_study": BASELINE_STUDY,
        "baseline_reference_result_commit": BASELINE_RESULT_COMMIT,
        "baseline_reference_study_manifest_sha256": sha256(source["baseline_study_path"]),
        "source_weights_sha256": FROZEN_WEIGHTS_SHA256,
        "frozen_weights_path": FROZEN_WEIGHTS_PATH.as_posix(),
        "frozen_weights_measured_sha256": source["input_hashes"]["weights_sha256"],
        "onnx_sha256": SOURCE_ONNX_SHA256,
        "calibration_cache_source_repeat": 1,
        "calibration_cache_input_sha256": sha256(dest / "calibration_cache_input.cache"),
        "calibration_cache_read": calibrator_audit.read_calls >= 1 and not calibrator_audit.read_violations,
        "calibration_cache_read_calls": calibrator_audit.read_calls,
        "calibration_cache_read_violations": calibrator_audit.read_violations,
        "calibration_batches_consumed": calibrator_audit.batch_calls,
        "calibration_cache_write_calls": calibrator_audit.write_calls,
        "calibration_cache_write_violations": calibrator_audit.write_violations,
        "calibration_cache_output_status": calibration_output_status,
        "calibration_cache_output_sha256": calibration_output_hash,
        "calibration_cache_output_files": [
            "calibration_cache_output.cache" if index == 1
            else f"calibration_cache_output_{index}.cache"
            for index in range(1, calibrator_audit.write_calls + 1)
        ],
        "calibration_cache_output_changed": calibration_output_changed,
        "timing_cache_input_source": "source.repeat_1/timing.cache",
        "timing_cache_input_sha256": sha256(dest / "timing_cache_input.cache"),
        "timing_cache_source_sha256": SOURCE_TIMING_CACHE_SHA256,
        "timing_cache_attach": timing_attach,
        "timing_cache_output_sha256": sha256(dest / "timing_cache_output.cache"),
        "timing_cache_output_changed": timing_output != timing_bytes,
        "timing_cache_coverage": CACHE_COVERAGE_UNKNOWN,
        "cache_chain": False,
        "settings": REPLAY_SETTINGS,
        "builder_flags": int(config.flags),
        "constraint_audit": constraint_audit,
        "inspector_sha256": sha256(dest / "inspector.json"),
        "inspector_signature": inspector_signature(inspector_info),
        "source_inspector_signature": inspector_signature(read(source["root"] / "repeat_1/inspector.json")),
        "engine_sha256": sha256(dest / "model.engine"),
        "environment": env,
        "gpu_before": before,
        "gpu_after": after,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "termination_status": "completed",
        "scope": "Precision-head diagnostic intervention only; no retraining, official test, or device benchmark.",
    }
    write_json(dest / "build_manifest.json", manifest)
    print(f"DONE BUILD {arm} repeat {repeat}: {dest}", flush=True)


def _validate_ablation_build(output: Path, source: dict, arm: str, repeat: int):
    study_manifest_path = Path(output) / "study_manifest.json"
    _require_file(study_manifest_path, "precision ablation study manifest")
    validate_ablation_study_manifest(read(study_manifest_path), source)
    dest = Path(output) / arm / f"repeat_{repeat}"
    manifest_path = dest / "build_manifest.json"
    engine = dest / "model.engine"
    for path, label in ((manifest_path, f"{arm} repeat_{repeat} build manifest"),
                        (engine, f"{arm} repeat_{repeat} engine"),
                        (dest / "inspector.json", f"{arm} repeat_{repeat} inspector"),
                        (dest / "calibration_cache_input.cache", f"{arm} calibration input"),
                        (dest / "timing_cache_input.cache", f"{arm} timing input"),
                        (dest / "timing_cache_output.cache", f"{arm} timing output")):
        _require_file(path, label)
    manifest = read(manifest_path)
    expected = {
        "study": STUDY, "arm": arm, "repeat": repeat, "source_study": SOURCE_STUDY,
        "source_result_commit": SOURCE_RESULT_COMMIT,
        "source_result_commit_available": source.get("source_result_commit_available"),
        "source_study_code_commit": source["manifest"].get("git_commit"),
        "source_study_manifest_sha256": sha256(source["manifest_path"]),
        "source_summary_sha256": sha256(source["source_summary"]),
        "source_build_manifest_sha256": sha256(source["root"] / "repeat_1/build_manifest.json"),
        "study_manifest_sha256": sha256(study_manifest_path),
        "baseline_reference_study": BASELINE_STUDY,
        "baseline_reference_result_commit": BASELINE_RESULT_COMMIT,
        "source_weights_sha256": FROZEN_WEIGHTS_SHA256,
        "frozen_weights_path": FROZEN_WEIGHTS_PATH.as_posix(),
        "frozen_weights_measured_sha256": source["input_hashes"]["weights_sha256"],
        "onnx_sha256": SOURCE_ONNX_SHA256,
        "calibration_cache_input_sha256": SOURCE_CALIBRATION_CACHE_SHA256,
        "timing_cache_input_sha256": SOURCE_TIMING_CACHE_SHA256,
        "calibration_cache_read": True,
        "calibration_cache_source_repeat": 1,
        "calibration_batches_consumed": 0,
        "timing_cache_attach": {"called": True, "ignore_mismatch": False, "return_value": True},
        "timing_cache_coverage": CACHE_COVERAGE_UNKNOWN,
        "cache_chain": False,
        "settings": REPLAY_SETTINGS,
        "builder_flags": source["build_manifests"][1].get("builder_flags"),
        "termination_status": "completed",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{arm} repeat_{repeat} build contract differs for {key}")
    study_manifest = read(study_manifest_path)
    if manifest.get("environment") != study_manifest.get("environment"):
        raise ValueError(f"{arm} repeat_{repeat} environment differs from preflight")
    for phase in ("gpu_before", "gpu_after"):
        validate_gpu_identity(manifest.get(phase), study_manifest)
        guard = manifest.get(phase, {}).get("process_guard", {})
        if (guard.get("telemetry_status") != "complete" or
                guard.get("external_workload_detected") or guard.get("blocked_processes") or
                guard.get("unmatched_confirmations")):
            raise ValueError(f"{arm} repeat_{repeat} build {phase} workload/telemetry guard failed")
    if sha256(engine) != manifest.get("engine_sha256"):
        raise ValueError(f"{arm} repeat_{repeat} engine hash mismatch")
    if sha256(dest / "inspector.json") != manifest.get("inspector_sha256"):
        raise ValueError(f"{arm} repeat_{repeat} inspector hash mismatch")
    for filename, expected_hash in (
        ("calibration_cache_input.cache", SOURCE_CALIBRATION_CACHE_SHA256),
        ("timing_cache_input.cache", SOURCE_TIMING_CACHE_SHA256),
        ("timing_cache_output.cache", manifest.get("timing_cache_output_sha256")),
    ):
        if expected_hash is None or sha256(dest / filename) != expected_hash:
            raise ValueError(f"{arm} repeat_{repeat} {filename} hash mismatch")
    if manifest.get("calibration_cache_read_calls", 0) < 1:
        raise ValueError(f"{arm} repeat_{repeat} calibration cache was not read")
    if manifest.get("calibration_cache_read_violations") or manifest.get("calibration_cache_write_violations"):
        raise ValueError(f"{arm} repeat_{repeat} calibration callback violation")
    if manifest.get("calibration_cache_output_changed") is True:
        raise ValueError(f"{arm} repeat_{repeat} calibration output changed")
    if manifest.get("cache_chain") is not False or manifest.get("timing_cache_input_source") != "source.repeat_1/timing.cache":
        raise ValueError(f"{arm} repeat_{repeat} timing cache was chained or has an unexpected source")
    constraint = manifest.get("constraint_audit", {})
    if constraint.get("arm") != arm:
        raise ValueError(f"{arm} repeat_{repeat} constraint arm mismatch")
    validate_constraint_audit(
        constraint, arm, source["build_manifests"][1].get("sigmoid_fp32_constraints"))
    return manifest


def _aggregate_metrics(records):
    output = {"ultralytics": {}, "coco_xml": {}}
    if len(records) < 2:
        return output
    for metric in FULL_METRICS:
        output["ultralytics"][metric] = metric_aggregate(
            [row["capture"]["metrics"][metric] for row in records])
    for label in SIZE_LABELS:
        output["coco_xml"][label] = {
            metric: metric_aggregate([row["size"]["metrics"][label][metric] for row in records])
            for metric in SIZE_METRICS
        }
    return output


def _aggregate_delta(aggregate, baseline):
    output = {"ultralytics": {}, "coco_xml": {}}
    for metric in FULL_METRICS:
        output["ultralytics"][metric] = aggregate["ultralytics"][metric]["mean"] - baseline["ultralytics"][metric]["mean"]
    for label in SIZE_LABELS:
        output["coco_xml"][label] = {
            metric: aggregate["coco_xml"][label][metric]["mean"] - baseline["coco_xml"][label][metric]["mean"]
            for metric in SIZE_METRICS
        }
    return output


def classify_arm(records, invalid_reasons=()):
    if invalid_reasons:
        return "incomplete_or_invalid"
    ids = [row.get("repeat") for row in records]
    if len(records) != 3 or set(ids) != set(REPEATS):
        return "incomplete_or_invalid"
    payloads = {row["prediction_payload_sha256"] for row in records}
    metrics = {json.dumps({"ultralytics": row["capture"]["metrics"],
                           "coco_xml": row["size"]["metrics"]},
                          sort_keys=True, separators=(",", ":")) for row in records}
    return "replay_exact_observed" if len(payloads) == 1 and len(metrics) == 1 else "replay_variation_observed"


def _comparison(row, baseline_row, baseline_capture, baseline_predictions, baseline_size):
    reference_difference = prediction_difference(baseline_predictions, row["predictions"])
    reference_metrics = {
        "ultralytics": numeric_deltas(baseline_capture["metrics"], row["capture"]["metrics"]),
        "coco_xml": numeric_deltas(baseline_size["metrics"], row["size"]["metrics"]),
    }
    reference_exact = (baseline_capture["metrics"] == row["capture"]["metrics"] and
                       baseline_size["metrics"] == row["size"]["metrics"])
    if baseline_row is None:
        arm_difference = {"exact": True, "differences": {}}
        arm_metrics = {"ultralytics": {}, "coco_xml": {}}
        arm_exact = True
    else:
        arm_difference = prediction_difference(baseline_row["predictions"], row["predictions"])
        arm_metrics = {
            "ultralytics": numeric_deltas(baseline_row["capture"]["metrics"], row["capture"]["metrics"]),
            "coco_xml": numeric_deltas(baseline_row["size"]["metrics"], row["size"]["metrics"]),
        }
        arm_exact = (baseline_row["capture"]["metrics"] == row["capture"]["metrics"] and
                     baseline_row["size"]["metrics"] == row["size"]["metrics"])
    return {
        "prediction_payload_exact_vs_timing_replay_reference": reference_difference["exact"],
        "prediction_difference_vs_timing_replay_reference": reference_difference["differences"],
        "metrics_exact_vs_timing_replay_reference": reference_exact,
        "metrics_delta_vs_timing_replay_reference": reference_metrics,
        "prediction_payload_exact_vs_arm_repeat_1": arm_difference["exact"],
        "prediction_difference_vs_arm_repeat_1": arm_difference["differences"],
        "metrics_exact_vs_arm_repeat_1": arm_exact,
        "metrics_delta_vs_arm_repeat_1": arm_metrics,
    }


def _guard_invalid_reasons(row):
    reasons = []
    for source_name in ("build", "capture"):
        for phase in ("gpu_before", "gpu_after"):
            guard = row[source_name].get(phase, {}).get("process_guard", {})
            if guard.get("telemetry_status") != "complete":
                reasons.append(f"{row['arm']}_repeat_{row['repeat']}_{source_name}_{phase}_telemetry")
            if guard.get("external_workload_detected") or guard.get("blocked_processes") or guard.get("unmatched_confirmations"):
                reasons.append(f"{row['arm']}_repeat_{row['repeat']}_{source_name}_{phase}_external_workload")
    build = row["build_manifest"]
    if (not row["capture"].get("status") == "pass" or
            row["verification"].get("native_matching_status") != "pass" or
            row["verification"].get("size_diagnostic") != "completed" or
            build.get("calibration_batches_consumed") != 0 or
            not build.get("calibration_cache_read") or
            build.get("calibration_cache_read_violations") or
            build.get("calibration_cache_write_violations")):
        reasons.append(f"{row['arm']}_repeat_{row['repeat']}_contract")
    return reasons


def _diagnostic_branch_classification(arms):
    if any(item["classification"] == "incomplete_or_invalid" for item in arms.values()):
        return "incomplete_or_invalid"
    for arm in ("bbox_fp32", "classification_fp32", "both_fp32"):
        deltas = arms[arm]["delta_vs_baseline_arm_mean"]
        variation = arms[arm]["build_variability"]
        baseline_variation = arms["baseline_int8"]["build_variability"]
        improved = 0
        for label, group in ENDPOINTS:
            metric = "map50"
            if group == "ultralytics":
                current = deltas[group][metric]
                current_range = variation[group][metric]["range_pp"] / 100
                base_range = baseline_variation[group][metric]["range_pp"] / 100
            else:
                size_label = "all" if label == "full" else label
                current = deltas[group][size_label][metric]
                current_range = variation[group][size_label][metric]["range_pp"] / 100
                base_range = baseline_variation[group][size_label][metric]["range_pp"] / 100
            if current > max(current_range, base_range) + 1e-12:
                improved += 1
        if improved >= 2:
            return "diagnostic_branch_sensitive"
    return "no_branch_signal"


def evaluate(repo: Path, output: Path, args, source: dict, helpers):
    manifests = {}
    for item in build_plan():
        manifests[(item["arm"], item["repeat"])] = _validate_ablation_build(
            output, source, item["arm"], item["repeat"])
    validate_arm_layer_contract({
        arm: {repeat: manifests[(arm, repeat)] for repeat in REPEATS}
        for arm in ARMS
    })
    baseline_capture = source["baseline_capture"]
    baseline_predictions = source["baseline_predictions"]
    baseline_size = source["baseline_size"]
    xml = (repo / "../nighttime-tsd/data/raw/CCTSDB2021/xml.zip").resolve()
    run_child = helpers.get("run_child", _run_child)
    write_json_fn = helpers.get("write_json", write_json)
    records_by_arm = {arm: [] for arm in ARMS}
    baseline_rows = {}
    for item in build_plan():
        arm, repeat = item["arm"], item["repeat"]
        run_dir = output / arm / f"repeat_{repeat}"
        capture_cmd = ablation_capture_command(repo, output, arm, repeat, run_dir,
                                               args.device, args.confirmed_desktop)
        capture_pid = run_child(capture_cmd, repo)
        verify_cmd = verification_command(repo, run_dir / "capture", run_dir / "verification", xml)
        verify_pid = run_child(verify_cmd, repo)
        capture = read(run_dir / "capture/capture_report.json")
        predictions = read(run_dir / "capture/validator_predictions.json")
        verification = read(run_dir / "verification/verification_summary.json")
        native = read(run_dir / "verification/native_matching.json")
        size = read(run_dir / "verification/size_coco_xml.json")
        validate_capture_contract(capture, verification)
        for phase in ("gpu_before", "gpu_after"):
            validate_gpu_identity(capture[phase], {"gpu_before": source["manifest"]["gpu_before"]})
        build_manifest = manifests[(arm, repeat)]
        engine_hash = build_manifest["engine_sha256"]
        if capture.get("model_sha256") != engine_hash or verification.get("model_sha256") != engine_hash:
            raise ValueError(f"{arm} repeat_{repeat} capture engine identity mismatch")
        if capture.get("engine_provenance_sha256") != sha256(run_dir / "build_manifest.json"):
            raise ValueError(f"{arm} repeat_{repeat} capture provenance mismatch")
        if sha256(run_dir / "capture/validator_predictions.json") != capture.get("predictions_sha256"):
            raise ValueError(f"{arm} repeat_{repeat} prediction hash mismatch")
        if verification.get("capture_hash_match") != "exact_bytes" or verification.get("capture_prediction_sha256") != capture.get("predictions_sha256"):
            raise ValueError(f"{arm} repeat_{repeat} verification prediction identity mismatch")
        if verification.get("capture_report_sha256") != sha256(run_dir / "capture/capture_report.json"):
            raise ValueError(f"{arm} repeat_{repeat} capture report hash mismatch")
        check_same_targets(baseline_predictions, predictions)
        size_convention(size, baseline_size)
        row = {
            "arm": arm, "repeat": repeat, "sequence": item["sequence"],
            "engine_sha256": engine_hash, "capture_process_pid": capture_pid,
            "verification_process_pid": verify_pid, "capture_command": capture_cmd,
            "verification_command": verify_cmd, "build_manifest": build_manifest,
            "build": {"gpu_before": build_manifest["gpu_before"], "gpu_after": build_manifest["gpu_after"]},
            "capture": capture, "predictions": predictions, "verification": verification,
            "native": native, "size": size,
            "prediction_payload_version": PREDICTION_PAYLOAD_VERSION,
            "prediction_payload_sha256": prediction_payload_hash(predictions),
        }
        row.update(_comparison(row, baseline_rows.get(arm), baseline_capture, baseline_predictions, baseline_size))
        baseline_rows.setdefault(arm, row)
        records_by_arm[arm].append(row)
        public_row = dict(row)
        public_row.pop("predictions", None)
        public_row.pop("build_manifest", None)
        write_json_fn(run_dir / "execution_manifest.json", {
            "schema_version": 1, "study": STUDY, "arm": arm, "repeat": repeat,
            "engine_sha256": engine_hash, "capture_process_pid": capture_pid,
            "verification_process_pid": verify_pid, "capture_command": capture_cmd,
            "verification_command": verify_cmd, "capture_return_code": 0,
            "verification_return_code": 0, "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_json_fn(run_dir / "comparison.json", public_row)
        print(f"FINISHED {arm} build/capture/verify {repeat}/3", flush=True)

    arms = {}
    all_reasons = []
    for arm in ARMS:
        reasons = []
        for row in records_by_arm[arm]:
            reasons.extend(_guard_invalid_reasons(row))
        classification = classify_arm(records_by_arm[arm], reasons)
        all_reasons.extend(reasons)
        aggregate = _aggregate_metrics(records_by_arm[arm])
        arms[arm] = {
            "arm": arm,
            "classification": classification,
            "review_flags": sorted(set(reasons)),
            "records": [{key: value for key, value in row.items() if key != "predictions" and key != "build_manifest"}
                        for row in records_by_arm[arm]],
            "build_variability": aggregate,
            "matched_layer_names": manifests[(arm, 1)]["constraint_audit"]["matched_layer_names"],
            "matched_layer_count": manifests[(arm, 1)]["constraint_audit"]["matched_layer_count"],
            "matched_layer_types": manifests[(arm, 1)]["constraint_audit"]["matched_layer_types"],
        }
    baseline_aggregate = arms["baseline_int8"]["build_variability"]
    for arm in ARMS:
        arms[arm]["delta_vs_baseline_arm_mean"] = _aggregate_delta(
            arms[arm]["build_variability"], baseline_aggregate)
    branch_deltas = {
        "bbox_delta": arms["bbox_fp32"]["delta_vs_baseline_arm_mean"],
        "classification_delta": arms["classification_fp32"]["delta_vs_baseline_arm_mean"],
        "both_delta": arms["both_fp32"]["delta_vs_baseline_arm_mean"],
    }
    for arm in ARMS:
        arms[arm]["branch_delta_full_xs_s"] = {
            label: ({metric: arms[arm]["delta_vs_baseline_arm_mean"][group][metric]
                     for metric in FULL_METRICS}
                    if group == "ultralytics" else
                    {metric: arms[arm]["delta_vs_baseline_arm_mean"][group][
                        "all" if label == "full" else label][metric]
                     for metric in SIZE_METRICS})
            for label, group in ENDPOINTS
        }
    overall = ("incomplete_or_invalid" if any(item["classification"] == "incomplete_or_invalid" for item in arms.values())
               else "replay_variation_observed" if any(item["classification"] == "replay_variation_observed" for item in arms.values())
               else "replay_exact_observed")
    diagnostic = _diagnostic_branch_classification(arms)
    summary = {
        "schema_version": 1, "study": STUDY, "source_study": SOURCE_STUDY,
        "source_result_commit": SOURCE_RESULT_COMMIT, "baseline_reference_result_commit": BASELINE_RESULT_COMMIT,
        "dataset_split": DATASET_SPLIT, "dev_images": 1636, "dev_instances": 2706,
        "arms": arms, "branch_deltas": branch_deltas, "status": overall,
        "classification": overall, "diagnostic_branch_classification": diagnostic,
        "global_g0": "review_required", "common_timing_cache_coverage": CACHE_COVERAGE_UNKNOWN,
        "common_timing_cache_input_sha256": SOURCE_TIMING_CACHE_SHA256,
        "common_calibration_cache_input_sha256": SOURCE_CALIBRATION_CACHE_SHA256,
        "build_order": build_plan(), "capture_order": build_plan(),
        "review_flags": sorted(set(all_reasons + [
            f"{arm}_timing_cache_output_changed" for arm in ARMS
            if any(manifests[(arm, repeat)].get("timing_cache_output_changed") for repeat in REPEATS)
        ])),
        "next_action": "STOP for reviewer decision; do not open calibration policy, retraining, 15-model matrix or benchmark.",
        "limitations": "Common timing-cache input was selected after accepted Step-A/timing-replay results and is not a tactic lock. TensorRT may reject the cache after precision constraints; coverage remains unknown. Layer names/constraints are diagnostic evidence, not proof that all branch arithmetic is FP32. Shared-server telemetry is sampled and does not prove GPU isolation. No official test is used.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_fn(output / "comparison_summary.json", summary)
    write_json_fn(output / "repeat_summary.json", summary)
    print(f"DONE: {output / 'comparison_summary.json'}")
    return 0


def check_same_targets(reference_capture, current_capture) -> None:
    reference = load_records(reference_capture)
    current = load_records(current_capture)
    if set(reference) != set(current):
        raise ValueError("Precision ablation capture image membership differs from baseline reference")
    for name in reference:
        for key in ("orig_shape",):
            if reference[name][key] != current[name][key]:
                raise ValueError(f"Precision ablation image dimensions changed: {name}")
        for key in ("imgsz", "ratio_pad", "target_xyxy", "target_class_id"):
            if reference[name]["validator_input"][key] != current[name]["validator_input"][key]:
                raise ValueError(f"Precision ablation target/preprocessing mismatch: {name}: {key}")


def ablation_capture_inputs(repo: Path, root: Path, arm: str, repeat: int):
    source, expected = resolve_protocol_paths(repo)
    root = Path(root).resolve()
    if root != expected:
        raise ValueError(f"Precision ablation study must use exactly {expected}")
    if arm not in ARMS or repeat not in REPEATS:
        raise ValueError("Precision ablation capture requires a valid arm and repeat")
    contract = validate_ablation_sources(repo)
    build_manifest = _validate_ablation_build(root, contract, arm, repeat)
    engine = root / arm / f"repeat_{repeat}/model.engine"
    previous = contract["baseline_capture_path"]
    previous_report = contract["baseline_capture"]
    provenance = {
        "source_weights_sha256": FROZEN_WEIGHTS_SHA256,
        "environment": {"tensorrt_python": build_manifest["environment"]["tensorrt"]},
    }
    return engine, root / arm / f"repeat_{repeat}/build_manifest.json", previous, previous_report, provenance, build_manifest["engine_sha256"]


def _run_child(command, repo: Path):
    print("START: " + " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=repo)
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Child command failed with exit {return_code}: {' '.join(command)}")
    return process.pid


def _write_study_manifest(output, source, probe, gpu_before, gpu_binding, confirmations, repo):
    write_json(output / "study_manifest.json", expected_study_manifest(
        source, environment=probe["environment"], gpu_before=gpu_before,
        gpu_binding=gpu_binding, confirmations=confirmations, repo=repo,
        execution_git_commit=git_value(repo, "rev-parse", "HEAD"),
        runner_script_sha256=sha256(Path(__file__))))


def _compress_log(log_path: Path, compressed_path: Path):
    with log_path.open("rb") as source, gzip.open(compressed_path, "xb") as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)


def main(argv=None, *, repo_override=None, helpers=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "prepare", "build", "evaluate"), default="all")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--repeat", type=int, choices=REPEATS)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confirm-desktop-process", action="append", default=[], metavar="PID=PATH")
    args = parser.parse_args(argv)
    repo = Path(repo_override if repo_override is not None else Path(__file__).resolve().parents[1]).resolve()
    helpers = dict(helpers or {})
    try:
        source_root, output = validate_output_target(repo, args.out_dir)
        if str(args.device) != "0":
            raise ValueError("Precision ablation protocol is locked to --device 0")
    except ValueError as error:
        parser.error(str(error))
    if args.phase == "build" and (args.arm is None or args.repeat is None):
        parser.error("--arm and --repeat are required for build phase")
    if args.phase != "build" and (args.arm is not None or args.repeat is not None):
        parser.error("--arm/--repeat are only valid for build phase")
    if args.phase == "build":
        if not output.exists() or not (output / "study_manifest.json").is_file():
            parser.error("Build phase requires an existing prepared study manifest; do not run standalone")
    parse_desktop = helpers.get("parse_desktop_confirmations")
    if parse_desktop is None:
        from uniform_build_repeat import parse_desktop_confirmations
        parse_desktop = parse_desktop_confirmations
    try:
        args.confirmed_desktop = parse_desktop(args.confirm_desktop_process)
    except ValueError as error:
        parser.error(str(error))
    source = validate_ablation_sources(repo)
    if args.phase == "build":
        build(repo, output, args.arm, args.repeat, args.confirmed_desktop)
        return 0
    if args.phase == "all":
        if output.exists():
            parser.error("Output exists; preserve it and do not resume or overwrite")
        probe, gpu_before, gpu_binding = preflight(repo, source, args, helpers)
        output.mkdir(parents=True)
        _write_study_manifest(output, source, probe, gpu_before, gpu_binding,
                              args.confirmed_desktop, repo)
        for item in build_plan():
            log = output / f"build_{item['arm']}_{item['repeat']}.log"
            command = _build_child_command(repo, output, item["arm"], item["repeat"], args.confirmed_desktop)
            print(f"START BUILD {item['sequence']}/12 {item['arm']} repeat {item['repeat']}; verbose log: {log}", flush=True)
            with log.open("x", encoding="utf-8") as handle:
                process = subprocess.run(command, cwd=repo, stdout=handle, stderr=subprocess.STDOUT, check=False)
            _compress_log(log, output / f"build_{item['arm']}_{item['repeat']}.log.gz")
            if process.returncode:
                raise RuntimeError(f"Build {item['arm']} repeat {item['repeat']} failed; preserve partial output and inspect {log}")
            print(f"FINISHED BUILD {item['sequence']}/12 {item['arm']} repeat {item['repeat']}", flush=True)
        source_after = validate_ablation_sources(repo)
        if source_after["input_hashes"] != source["input_hashes"]:
            raise RuntimeError("Locked Step-A source inputs changed during ablation builds")
        return evaluate(repo, output, args, source_after, helpers)
    if args.phase == "prepare":
        if output.exists():
            parser.error("Output exists; preserve it and do not overwrite")
        probe, gpu_before, gpu_binding = preflight(repo, source, args, helpers)
        output.mkdir(parents=True)
        _write_study_manifest(output, source, probe, gpu_before, gpu_binding,
                              args.confirmed_desktop, repo)
        print(f"PREPARED: {output / 'study_manifest.json'}")
        return 0
    if args.phase == "evaluate":
        if not output.exists():
            parser.error("Evaluate phase requires prepared output")
        preflight(repo, source, args, helpers)
        return evaluate(repo, output, args, source, helpers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
