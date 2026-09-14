#!/usr/bin/env python3
"""Build three Uniform engines with one frozen timing-cache input, then evaluate once."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
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


SOURCE_STUDY = "uniform_build_repeat_v1"
STUDY = "uniform_timing_cache_replay_v1"
SOURCE_STUDY_DIR = "server_uniform_build_repeat_v1"
STUDY_DIR = "server_uniform_timing_cache_replay_v1"
SOURCE_RESULT_COMMIT = "a5e79e7c15259facc24114279a050ded439cc9ed"
SOURCE_CODE_COMMIT = "d9378cb8416be714dff2f823405f727bc9258f0b"
FROZEN_WEIGHTS_SHA256 = "3e5fc7a2148c16539cd9fb7cc7cacd81a4eec1dfc28143cdf9b6dcd872ba4ab8"
SOURCE_ONNX_SHA256 = "d187dc23430cbe227594f6ac28b2a5d88793adc1bcde72f4b7b4b7ddc94557e4"
SOURCE_CALIBRATION_CACHE_SHA256 = "31e9d0b3f69470ac20f7380d8887c6dc47954afbe85d84be39870ba44e01a502"
SOURCE_TIMING_CACHE_SHA256 = "4c765a0224845e9ddc537253878c696e56c11045369aef224cff6cc0c4178f38"
TIMING_CACHE_SOURCE_REPEAT = 1
ENGINE_REPEATS = (1, 2, 3)

# This is intentionally the Step-A builder contract with only the timing-cache
# source changed. No editable cache, algorithm selector, Q/DQ, or precision
# policy is introduced by this feasibility study.
SOURCE_SETTINGS = {
    "imgsz": 640,
    "batch": 1,
    "workspace_bytes": 4 << 30,
    "builder_optimization_level": 3,
    "avg_timing_iterations": 1,
    "timing_cache": "fresh_empty_for_each_repeat",
    "int8": True,
    "fp16": False,
    "tf32": False,
    "sigmoid": "FP32 precision+output with OBEY; no bbox/classification override",
}
REPLAY_SETTINGS = {**SOURCE_SETTINGS, "timing_cache": "source_repeat_1_replay_per_process"}
REPLAY_VARIANT = "ordinary_timing_cache_replay_v1"
SIZE_LABELS = ("all", "xs", "s", "m", "l", "xl")
SIZE_METRICS = ("map50", "map50_95")
FULL_METRICS = ("map50", "map50_95", "precision", "recall")
_DESKTOP_EXECUTABLES = (
    ("snapd-desktop-integration", re.compile(r"^/snap/snapd-desktop-integration/[1-9][0-9]*/usr/bin/snapd-desktop-integration$")),
    ("xorg", re.compile(r"^/(?:usr/lib/xorg/Xorg|usr/bin/Xorg)$")),
    ("gnome-shell", re.compile(r"^/(?:usr/bin|usr/libexec)/gnome-shell$")),
    ("gnome-session", re.compile(r"^/(?:usr/bin|usr/libexec)/gnome-session-binary$")),
)


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
    return audit_root / SOURCE_STUDY_DIR, audit_root / STUDY_DIR


def validate_output_target(repo: Path, output: Path):
    source, expected = resolve_protocol_paths(repo)
    if Path(output).resolve() != expected:
        raise ValueError(f"Output must be exactly {expected}")
    return source, expected


def parse_desktop_confirmations(values):
    """Pure parser for the parent boundary; CUDA helpers load only after probe."""
    confirmations = {}
    for value in values or ():
        if not isinstance(value, str):
            raise ValueError("Desktop confirmation must be PID=PATH")
        pid_text, separator, path = value.partition("=")
        if not separator or not pid_text.isdigit() or not path:
            raise ValueError(f"Invalid desktop confirmation {value!r}; expected PID=PATH")
        pid = int(pid_text)
        if pid <= 0 or pid in confirmations:
            raise ValueError(f"Duplicate or invalid desktop confirmation PID: {pid_text!r}")
        if not any(pattern.fullmatch(path) for _name, pattern in _DESKTOP_EXECUTABLES):
            raise ValueError(f"Desktop confirmation path is outside the allowlist: {path!r}")
        confirmations[pid] = path
    return confirmations


def check_same_targets(reference_capture, current_capture) -> None:
    """Keep the dev membership/preprocessing target contract identical to Step A."""
    reference = load_records(reference_capture)
    current = load_records(current_capture)
    if set(reference) != set(current):
        raise ValueError("Timing-cache replay capture image membership differs from Step A")
    for name in reference:
        for key in ("orig_shape",):
            if reference[name][key] != current[name][key]:
                raise ValueError(f"Timing-cache replay image dimensions changed: {name}")
        for key in ("imgsz", "ratio_pad", "target_xyxy", "target_class_id"):
            if reference[name]["validator_input"][key] != current[name]["validator_input"][key]:
                raise ValueError(f"Timing-cache replay target/preprocessing mismatch: {name}: {key}")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def validate_source_contract(repo: Path):
    """Validate immutable Step-A inputs before creating the replay output."""
    source_root, _ = resolve_protocol_paths(repo)
    manifest_path = source_root / "study_manifest.json"
    _require_file(manifest_path, "Step-A study manifest")
    manifest = read(manifest_path)
    if manifest.get("study") != SOURCE_STUDY:
        raise ValueError("Source is not the approved Uniform build-repeat study")
    if manifest.get("git_commit") != SOURCE_CODE_COMMIT:
        raise ValueError("Step-A source code commit differs from the locked contract")
    if manifest.get("source_weights_sha256") != FROZEN_WEIGHTS_SHA256:
        raise ValueError("Frozen source weights differ from the locked contract")
    if manifest.get("onnx_sha256") != SOURCE_ONNX_SHA256:
        raise ValueError("Step-A ONNX manifest hash differs from the locked contract")
    if manifest.get("settings") != SOURCE_SETTINGS:
        raise ValueError("Step-A builder settings differ from the locked contract")

    source_onnx = source_root / "source.onnx"
    calibration = source_root / f"repeat_{TIMING_CACHE_SOURCE_REPEAT}" / "calibration.cache"
    timing = source_root / f"repeat_{TIMING_CACHE_SOURCE_REPEAT}" / "timing.cache"
    source_summary = source_root / "repeat_summary.json"
    for path, label in ((source_onnx, "frozen source.onnx"),
                        (calibration, "repeat_1 calibration cache"),
                        (timing, "repeat_1 timing cache input"),
                        (source_summary, "Step-A repeat summary")):
        _require_file(path, label)
    actual_hashes = {
        "onnx_sha256": sha256(source_onnx),
        "calibration_cache_sha256": sha256(calibration),
        "timing_cache_sha256": sha256(timing),
    }
    expected_hashes = {
        "onnx_sha256": SOURCE_ONNX_SHA256,
        "calibration_cache_sha256": SOURCE_CALIBRATION_CACHE_SHA256,
        "timing_cache_sha256": SOURCE_TIMING_CACHE_SHA256,
    }
    if actual_hashes != expected_hashes:
        raise ValueError(f"Locked Step-A input hash mismatch: {actual_hashes}")

    build_manifests = {}
    for repeat in ENGINE_REPEATS:
        path = source_root / f"repeat_{repeat}" / "build_manifest.json"
        _require_file(path, f"Step-A repeat_{repeat} build manifest")
        item = read(path)
        if item.get("study") != SOURCE_STUDY or item.get("repeat") != repeat:
            raise ValueError(f"Step-A repeat_{repeat} manifest identity mismatch")
        if item.get("source_weights_sha256") != FROZEN_WEIGHTS_SHA256 or item.get("onnx_sha256") != SOURCE_ONNX_SHA256:
            raise ValueError(f"Step-A repeat_{repeat} source identity mismatch")
        if item.get("settings") != SOURCE_SETTINGS:
            raise ValueError(f"Step-A repeat_{repeat} settings mismatch")
        if item.get("calibration_cache_sha256") != SOURCE_CALIBRATION_CACHE_SHA256:
            raise ValueError(f"Step-A repeat_{repeat} calibration cache mismatch")
        build_manifests[repeat] = item
    if not all(build_manifests[repeat].get("sigmoid_fp32_constraints") ==
               build_manifests[1].get("sigmoid_fp32_constraints") for repeat in ENGINE_REPEATS):
        raise ValueError("Step-A Sigmoid FP32 constraint lists differ")

    result_commit_available = subprocess.run(
        ["git", "cat-file", "-e", f"{SOURCE_RESULT_COMMIT}^{{commit}}"],
        cwd=repo, capture_output=True, text=True, check=False).returncode == 0
    return {
        "root": source_root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "source_onnx": source_onnx,
        "calibration_cache": calibration,
        "timing_cache": timing,
        "source_summary": source_summary,
        "source_summary_data": read(source_summary),
        "build_manifests": build_manifests,
        "source_result_commit": SOURCE_RESULT_COMMIT,
        "source_result_commit_available": result_commit_available,
        "input_hashes": expected_hashes,
    }


def confirmation_args(confirmations):
    return [item for pid, path in sorted(confirmations.items())
            for item in ("--confirm-desktop-process", f"{pid}={path}")]


def timing_capture_command(repo, study_root, engine_id, run_dir, device, confirmations):
    """Dispatch capture through the bounded timing-replay provenance path."""
    return [sys.executable, str(repo / "scripts/capture_cctsdb_validator.py"),
            "--timing-cache-study", str(study_root), "--repeat-index", str(engine_id),
            "--out-dir", str(run_dir / "capture"), "--device", str(device),
            *confirmation_args(confirmations)]


def verification_command(repo, capture_dir, out_dir, xml_path):
    return [sys.executable, str(repo / "scripts/verify_cctsdb_capture.py"),
            "--capture-dir", str(capture_dir), "--xml", str(xml_path),
            "--out-dir", str(out_dir)]


def environment_preflight(repo: Path, runner=None):
    from run_uniform_inference_repeat import run_environment_preflight
    return run_environment_preflight(repo, runner)


def _load_guard_helpers(helpers):
    if any(key not in helpers for key in ("ensure_idle", "parse_desktop_confirmations", "snapshot")):
        from uniform_build_repeat import ensure_idle, parse_desktop_confirmations, snapshot
        helpers.setdefault("ensure_idle", ensure_idle)
        helpers.setdefault("parse_desktop_confirmations", parse_desktop_confirmations)
        helpers.setdefault("snapshot", snapshot)
    return helpers


def preflight(repo: Path, source: dict, args, helpers):
    probe = helpers.get("environment_preflight", environment_preflight)(repo)
    if probe.get("status") != "ok" or not isinstance(probe.get("environment"), dict):
        raise RuntimeError("Environment preflight did not return a usable report")
    if probe["environment"] != source["manifest"].get("environment"):
        raise ValueError("Runtime environment/GPU differs from the Step-A engine study")
    helpers = _load_guard_helpers(helpers)
    current = helpers["snapshot"](args.confirmed_desktop)
    helpers["ensure_idle"](current)
    binding = validate_gpu_identity(current, {"gpu_before": source["manifest"].get("gpu_before")})
    return probe, current, binding


def _build_child_command(repo: Path, output: Path, repeat: int, confirmations):
    return [sys.executable, str(repo / "scripts/run_uniform_timing_cache_replay.py"),
            "--phase", "build", "--out-dir", str(output), "--repeat", str(repeat),
            *confirmation_args(confirmations)]


def _run_child(command, repo: Path):
    print("START: " + " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=repo)
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Child command failed with exit {return_code}: {' '.join(command)}")
    return process.pid


def build(repo: Path, output: Path, repeat: int, confirmations):
    """Build one replay engine; this function is only called in a child process."""
    import torch
    import tensorrt as trt
    from uniform_build_repeat import environment, ensure_idle, inspector_signature, snapshot

    source = validate_source_contract(repo)
    env = environment()
    if env != source["manifest"].get("environment"):
        raise ValueError("Build environment changed from Step A")
    if repeat not in ENGINE_REPEATS:
        raise ValueError(f"Unsupported replay repeat: {repeat}")
    dest = output / f"repeat_{repeat}"
    if dest.exists():
        raise FileExistsError(f"Replay repeat output exists; preserve it: {dest}")
    dest.mkdir(parents=True)
    calibration_bytes = source["calibration_cache"].read_bytes()
    timing_bytes = source["timing_cache"].read_bytes()
    write_bytes(dest / "calibration_cache_input.cache", calibration_bytes)
    write_bytes(dest / "timing_cache_input.cache", timing_bytes)

    source_build = source["build_manifests"][1]
    with GpuPhaseLock(repo / "results/architecture_matrix_v1/.gpu_phase.lock",
                      f"{STUDY}_build_{repeat}"):
        before = snapshot(confirmations)
        ensure_idle(before)
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(source["source_onnx"])):
            raise ValueError([str(parser.get_error(i)) for i in range(parser.num_errors)])
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, REPLAY_SETTINGS["workspace_bytes"])
        config.builder_optimization_level = REPLAY_SETTINGS["builder_optimization_level"]
        config.avg_timing_iterations = REPLAY_SETTINGS["avg_timing_iterations"]
        config.set_flag(trt.BuilderFlag.INT8)
        config.clear_flag(trt.BuilderFlag.FP16)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        timing_cache = config.create_timing_cache(timing_bytes)
        if not config.set_timing_cache(timing_cache, False):
            raise RuntimeError("Cannot attach locked timing-cache input with ignore_mismatch=False")

        class CacheOnlyCalibrator(trt.IInt8Calibrator):
            def __init__(self):
                super().__init__()
                self.read_calls = 0
                self.batch_calls = 0
                self.write_calls = 0
                self.output = None

            def get_algorithm(self):
                return trt.CalibrationAlgoType.MINMAX_CALIBRATION

            def get_batch_size(self):
                return 1

            def read_calibration_cache(self):
                self.read_calls += 1
                return calibration_bytes

            def write_calibration_cache(self, data):
                self.write_calls += 1
                self.output = bytes(data)
                output_path = dest / "calibration_cache_output.cache"
                if not output_path.exists():
                    write_bytes(output_path, self.output)

            def get_batch(self, names):
                self.batch_calls += 1
                raise RuntimeError("Cache-only replay attempted recalibration")

        calibrator = CacheOnlyCalibrator()
        config.int8_calibrator = calibrator
        protected = []
        for layer_index in range(network.num_layers):
            layer = network.get_layer(layer_index)
            if layer.type == trt.LayerType.ACTIVATION and "sigmoid" in layer.name.lower():
                layer.precision = trt.float32
                for output_index in range(layer.num_outputs):
                    layer.set_output_type(output_index, trt.float32)
                protected.append(layer.name)
        if not protected or network.num_inputs != 1 or tuple(network.get_input(0).shape) != (1, 3, 640, 640):
            raise ValueError("Unexpected network/Sigmoid constraints")
        if protected != source_build.get("sigmoid_fp32_constraints"):
            raise ValueError("Sigmoid FP32 constraint list differs from Step A")

        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("TensorRT build failed; inspect build log")
        if calibrator.read_calls != 1 or calibrator.batch_calls != 0:
            raise RuntimeError("Calibration cache read/batch contract was not satisfied")
        if calibrator.output is not None and calibrator.output != calibration_bytes:
            raise RuntimeError("TensorRT returned a calibration table different from locked input")

        timing_output = bytes(config.get_timing_cache().serialize())
        write_bytes(dest / "timing_cache_output.cache", timing_output)
        metadata = json.dumps(source["manifest"]["engine_metadata"]).encode("utf-8")
        write_bytes(dest / "model.engine", len(metadata).to_bytes(4, "little") + metadata + bytes(plan))
        with trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(plan)
            if engine is None:
                raise RuntimeError("Replay engine failed deserialization")
            inspector = engine.create_engine_inspector()
            inspector_info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
            write_json(dest / "inspector.json", inspector_info)
            del inspector, engine
        after = snapshot(confirmations)
        ensure_idle(after)

    calibration_output = dest / "calibration_cache_output.cache"
    if calibrator.output is None:
        calibration_output_status = "not_returned"
        calibration_output_hash = None
        calibration_output_changed = None
    else:
        calibration_output_status = "same" if calibrator.output == calibration_bytes else "changed"
        calibration_output_hash = sha256(calibration_output)
        calibration_output_changed = calibrator.output != calibration_bytes
    inspector_info = read(dest / "inspector.json")
    manifest = {
        "schema_version": 1,
        "study": STUDY,
        "repeat": repeat,
        "source_study": SOURCE_STUDY,
        "source_result_commit": source["source_result_commit"],
        "source_result_commit_available": source["source_result_commit_available"],
        "source_study_code_commit": source["manifest"].get("git_commit"),
        "source_study_manifest_sha256": sha256(source["manifest_path"]),
        "source_summary_sha256": sha256(source["source_summary"]),
        "source_build_manifest_sha256": sha256(source["root"] / "repeat_1/build_manifest.json"),
        "source_weights_sha256": FROZEN_WEIGHTS_SHA256,
        "onnx_sha256": SOURCE_ONNX_SHA256,
        "calibration_cache_source_repeat": TIMING_CACHE_SOURCE_REPEAT,
        "calibration_cache_input_sha256": sha256(dest / "calibration_cache_input.cache"),
        "calibration_cache_read": calibrator.read_calls == 1,
        "calibration_cache_read_calls": calibrator.read_calls,
        "calibration_batches_consumed": calibrator.batch_calls,
        "calibration_cache_write_calls": calibrator.write_calls,
        "calibration_cache_output_sha256": calibration_output_hash,
        "calibration_cache_output_status": calibration_output_status,
        "calibration_cache_output_changed": calibration_output_changed,
        "timing_cache_input_source": f"source.repeat_{TIMING_CACHE_SOURCE_REPEAT}/timing.cache",
        "timing_cache_input_sha256": sha256(dest / "timing_cache_input.cache"),
        "timing_cache_source_sha256": SOURCE_TIMING_CACHE_SHA256,
        "timing_cache_attach": {"called": True, "ignore_mismatch": False, "return_value": True},
        "timing_cache_output_sha256": sha256(dest / "timing_cache_output.cache"),
        "timing_cache_output_changed": timing_output != timing_bytes,
        "settings": REPLAY_SETTINGS,
        "builder_flags": int(config.flags),
        "sigmoid_fp32_constraints": protected,
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
        "scope": "Ordinary timing-cache replay feasibility check before precision/calibration interventions; no editable cache, algorithm selector, Q/DQ or policy change.",
    }
    write_json(dest / "build_manifest.json", manifest)
    print(f"DONE BUILD {repeat}: {dest}", flush=True)


def replay_capture_inputs(repo: Path, root: Path, index: int):
    """Bound capture dispatch for the timing-replay study; rejects arbitrary engines."""
    source, expected = resolve_protocol_paths(repo)
    root = Path(root).resolve()
    if root != expected:
        raise ValueError(f"Timing-cache replay study must use exactly {expected}")
    if index not in ENGINE_REPEATS:
        raise ValueError("Timing-cache replay repeat must be 1, 2 or 3")
    contract = validate_source_contract(repo)
    if not (root / "study_manifest.json").is_file():
        raise ValueError("Timing-cache replay study manifest is missing")
    study_manifest = read(root / "study_manifest.json")
    if study_manifest.get("study") != STUDY:
        raise ValueError("Capture study manifest is not timing-cache replay")
    build_manifest_path = root / f"repeat_{index}/build_manifest.json"
    engine = root / f"repeat_{index}/model.engine"
    for path, label in ((build_manifest_path, "replay build manifest"), (engine, "replay engine")):
        _require_file(path, label)
    build_manifest = read(build_manifest_path)
    if build_manifest.get("study") != STUDY or build_manifest.get("repeat") != index:
        raise ValueError("Replay build provenance identity mismatch")
    if build_manifest.get("onnx_sha256") != contract["input_hashes"]["onnx_sha256"]:
        raise ValueError("Replay build ONNX identity mismatch")
    if build_manifest.get("calibration_cache_input_sha256") != SOURCE_CALIBRATION_CACHE_SHA256:
        raise ValueError("Replay build calibration cache identity mismatch")
    if sha256(engine) != build_manifest.get("engine_sha256"):
        raise ValueError("Replay engine bytes/hash mismatch")
    previous = contract["root"] / "repeat_1/capture/capture_report.json"
    _require_file(previous, "Step-A repeat_1 capture report")
    previous_report = read(previous)
    provenance = {
        "source_weights_sha256": FROZEN_WEIGHTS_SHA256,
        "environment": {"tensorrt_python": build_manifest["environment"]["tensorrt"]},
    }
    return engine, build_manifest_path, previous, previous_report, provenance, build_manifest["engine_sha256"]


def _validate_replay_build(output: Path, source: dict, repeat: int):
    dest = output / f"repeat_{repeat}"
    manifest_path = dest / "build_manifest.json"
    engine = dest / "model.engine"
    for path, label in ((manifest_path, f"replay repeat_{repeat} manifest"),
                        (engine, f"replay repeat_{repeat} engine")):
        _require_file(path, label)
    manifest = read(manifest_path)
    required = {
        "study": STUDY,
        "repeat": repeat,
        "source_study": SOURCE_STUDY,
        "onnx_sha256": SOURCE_ONNX_SHA256,
        "calibration_cache_input_sha256": SOURCE_CALIBRATION_CACHE_SHA256,
        "timing_cache_input_sha256": SOURCE_TIMING_CACHE_SHA256,
        "calibration_cache_read": True,
        "calibration_batches_consumed": 0,
        "timing_cache_attach": {"called": True, "ignore_mismatch": False, "return_value": True},
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Replay repeat_{repeat} build contract differs for {key}")
    if sha256(engine) != manifest.get("engine_sha256"):
        raise ValueError(f"Replay repeat_{repeat} engine hash mismatch")
    for name, expected in (("calibration_cache_input.cache", SOURCE_CALIBRATION_CACHE_SHA256),
                           ("timing_cache_input.cache", SOURCE_TIMING_CACHE_SHA256),
                           ("timing_cache_output.cache", manifest.get("timing_cache_output_sha256"))):
        path = dest / name
        _require_file(path, f"replay repeat_{repeat} {name}")
        if expected is None or sha256(path) != expected:
            raise ValueError(f"Replay repeat_{repeat} {name} hash mismatch")
    _require_file(dest / "inspector.json", f"replay repeat_{repeat} inspector")
    if sha256(dest / "inspector.json") != manifest.get("inspector_sha256"):
        raise ValueError(f"Replay repeat_{repeat} inspector hash mismatch")
    return manifest


def _aggregate_build_metrics(records):
    ultralytics = {}
    coco_xml = {}
    for metric in FULL_METRICS:
        ultralytics[metric] = metric_aggregate([row["capture"]["metrics"][metric] for row in records])
    for label in SIZE_LABELS:
        coco_xml[label] = {}
        for metric in SIZE_METRICS:
            coco_xml[label][metric] = metric_aggregate([
                row["size"]["metrics"][label][metric] for row in records])
    return {"ultralytics": ultralytics, "coco_xml": coco_xml}


def classify_replay(records, invalid_reasons=()):
    if invalid_reasons:
        return "incomplete_or_invalid"
    payloads = {row["prediction_payload_sha256"] for row in records}
    metric_payloads = {json.dumps({"ultralytics": row["capture"]["metrics"],
                                   "coco_xml": row["size"]["metrics"]},
                                  sort_keys=True, separators=(",", ":")) for row in records}
    return "replay_exact_observed" if len(payloads) == 1 and len(metric_payloads) == 1 else "replay_variation_observed"


def _build_comparison_row(row, baseline, source_capture, source_size):
    prediction_vs_source = prediction_difference(source_capture, row["predictions"])
    metrics_vs_source = {
        "ultralytics": numeric_deltas(source_capture["metrics"], row["capture"]["metrics"]),
        "coco_xml": numeric_deltas(source_size["metrics"], row["size"]["metrics"]),
    }
    metrics_exact_vs_source = (source_capture["metrics"] == row["capture"]["metrics"] and
                               source_size["metrics"] == row["size"]["metrics"])
    if baseline is None:
        prediction_vs_replay_baseline = {"exact": True, "differences": {}}
        metrics_exact_vs_replay_baseline = True
        metrics_vs_replay_baseline = {"ultralytics": {}, "coco_xml": {}}
    else:
        prediction_vs_replay_baseline = prediction_difference(baseline["predictions"], row["predictions"])
        metrics_exact_vs_replay_baseline = (baseline["capture"]["metrics"] == row["capture"]["metrics"] and
                                            baseline["size"]["metrics"] == row["size"]["metrics"])
        metrics_vs_replay_baseline = {
            "ultralytics": numeric_deltas(baseline["capture"]["metrics"], row["capture"]["metrics"]),
            "coco_xml": numeric_deltas(baseline["size"]["metrics"], row["size"]["metrics"]),
        }
    return {
        "prediction_payload_exact_vs_replay_build_1": prediction_vs_replay_baseline["exact"],
        "prediction_difference_vs_replay_build_1": prediction_vs_replay_baseline["differences"],
        "metrics_exact_vs_replay_build_1": metrics_exact_vs_replay_baseline,
        "metrics_delta_vs_replay_build_1": metrics_vs_replay_baseline,
        "prediction_payload_exact_vs_step_a_repeat_1": prediction_vs_source["exact"],
        "prediction_difference_vs_step_a_repeat_1": prediction_vs_source["differences"],
        "metrics_exact_vs_step_a_repeat_1": metrics_exact_vs_source,
        "metrics_delta_vs_step_a_repeat_1": metrics_vs_source,
    }


def evaluate(repo: Path, output: Path, args, source: dict, helpers):
    for repeat in ENGINE_REPEATS:
        _validate_replay_build(output, source, repeat)
    if not (output / "study_manifest.json").is_file():
        raise FileNotFoundError("Timing-cache replay study manifest is missing")
    study_manifest = read(output / "study_manifest.json")
    if study_manifest.get("study") != STUDY:
        raise ValueError("Timing-cache replay output manifest identity mismatch")
    xml = (repo / "../nighttime-tsd/data/raw/CCTSDB2021/xml.zip").resolve()
    source_capture = read(source["root"] / "repeat_1/capture/capture_report.json")
    source_predictions = read(source["root"] / "repeat_1/capture/validator_predictions.json")
    source_size = read(source["root"] / "repeat_1/verification/size_coco_xml.json")
    records = []
    baseline = None
    run_child = helpers.get("run_child", _run_child)
    write_json_fn = helpers.get("write_json", write_json)
    for repeat in ENGINE_REPEATS:
        run_dir = output / f"repeat_{repeat}"
        capture_dir, verification_dir = run_dir / "capture", run_dir / "verification"
        capture_cmd = timing_capture_command(repo, output, repeat, run_dir, args.device, args.confirmed_desktop)
        capture_pid = run_child(capture_cmd, repo)
        verify_cmd = verification_command(repo, capture_dir, verification_dir, xml)
        verify_pid = run_child(verify_cmd, repo)
        capture = read(capture_dir / "capture_report.json")
        predictions = read(capture_dir / "validator_predictions.json")
        verification = read(verification_dir / "verification_summary.json")
        size = read(verification_dir / "size_coco_xml.json")
        validate_capture_contract(capture, verification)
        gpu_binding = {}
        for phase in ("gpu_before", "gpu_after"):
            gpu_binding[phase] = validate_gpu_identity(capture[phase],
                                                        {"gpu_before": source["manifest"]["gpu_before"]})
        build_manifest = read(run_dir / "build_manifest.json")
        engine_hash = build_manifest["engine_sha256"]
        if capture.get("model_sha256") != engine_hash or verification.get("model_sha256") != engine_hash:
            raise ValueError(f"Replay repeat_{repeat} capture engine identity mismatch")
        if capture.get("engine_provenance_sha256") != sha256(run_dir / "build_manifest.json"):
            raise ValueError(f"Replay repeat_{repeat} capture provenance mismatch")
        if sha256(capture_dir / "validator_predictions.json") != capture.get("predictions_sha256"):
            raise ValueError(f"Replay repeat_{repeat} prediction hash mismatch")
        if verification.get("capture_hash_match") != "exact_bytes" or verification.get("capture_prediction_sha256") != capture.get("predictions_sha256"):
            raise ValueError(f"Replay repeat_{repeat} verification prediction identity mismatch")
        if verification.get("capture_report_sha256") != sha256(capture_dir / "capture_report.json"):
            raise ValueError(f"Replay repeat_{repeat} capture report hash mismatch")
        check_same_targets(source_predictions, predictions)
        size_convention(size, source_size)
        payload_hash = prediction_payload_hash(predictions)
        row = {
            "repeat": repeat,
            "engine_sha256": engine_hash,
            "capture_process_pid": capture_pid,
            "verification_process_pid": verify_pid,
            "capture_command": capture_cmd,
            "verification_command": verify_cmd,
            "calibration_cache_input_sha256": build_manifest["calibration_cache_input_sha256"],
            "calibration_cache_read": build_manifest["calibration_cache_read"],
            "calibration_batches_consumed": build_manifest["calibration_batches_consumed"],
            "timing_cache_input_sha256": build_manifest["timing_cache_input_sha256"],
            "timing_cache_output_sha256": build_manifest["timing_cache_output_sha256"],
            "timing_cache_output_changed": build_manifest["timing_cache_output_changed"],
            "builder_flags": build_manifest["builder_flags"],
            "settings": build_manifest["settings"],
            "inspector_signature": build_manifest["inspector_signature"],
            "source_inspector_signature": build_manifest["source_inspector_signature"],
            "gpu_before": build_manifest["gpu_before"],
            "gpu_after": build_manifest["gpu_after"],
            "capture": capture,
            "predictions": predictions,
            "size": size,
            "verification": verification,
            "prediction_payload_version": PREDICTION_PAYLOAD_VERSION,
            "prediction_payload_sha256": payload_hash,
        }
        row.update(_build_comparison_row(row, baseline, source_capture, source_size))
        if baseline is None:
            baseline = row
        public_row = dict(row)
        public_row.pop("predictions", None)
        records.append(public_row)
        write_json_fn(run_dir / "execution_manifest.json", {
            "schema_version": 1, "study": STUDY, "repeat": repeat,
            "engine_sha256": engine_hash, "capture_process_pid": capture_pid,
            "verification_process_pid": verify_pid, "capture_command": capture_cmd,
            "verification_command": verify_cmd, "capture_return_code": 0,
            "verification_return_code": 0, "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_json_fn(run_dir / "comparison.json", public_row)
        print(f"FINISHED BUILD/CAPTURE/VERIFY {repeat}/3", flush=True)

    invalid_reasons = []
    for row in records:
        if row["capture"].get("status") != "pass": invalid_reasons.append(f"repeat_{row['repeat']}_capture")
        if row["verification"].get("native_matching_status") != "pass": invalid_reasons.append(f"repeat_{row['repeat']}_native")
        if row["verification"].get("size_diagnostic") != "completed": invalid_reasons.append(f"repeat_{row['repeat']}_size")
        for phase in ("gpu_before", "gpu_after"):
            guard = row["capture"].get(phase, {}).get("process_guard", {})
            if guard.get("telemetry_status") != "complete":
                invalid_reasons.append(f"repeat_{row['repeat']}_{phase}_telemetry")
            if guard.get("external_workload_detected") or guard.get("blocked_processes") or guard.get("unmatched_confirmations"):
                invalid_reasons.append(f"repeat_{row['repeat']}_{phase}_external_workload")
        if row["calibration_batches_consumed"] != 0 or not row["calibration_cache_read"]:
            invalid_reasons.append(f"repeat_{row['repeat']}_calibration_contract")
    status = classify_replay(records, invalid_reasons)
    review_flags = sorted(set(invalid_reasons))
    for row in records:
        if row["timing_cache_output_changed"]:
            review_flags.append(f"repeat_{row['repeat']}_timing_cache_output_changed")
        if not row["prediction_payload_exact_vs_replay_build_1"]:
            review_flags.append(f"repeat_{row['repeat']}_prediction_payload_changed")
        if not row["metrics_exact_vs_replay_build_1"]:
            review_flags.append(f"repeat_{row['repeat']}_metrics_changed")
    review_flags = sorted(set(review_flags))
    summary = {
        "schema_version": 1,
        "study": STUDY,
        "source_study": SOURCE_STUDY,
        "source_result_commit": source["source_result_commit"],
        "source_code_commit": source["manifest"].get("git_commit"),
        "timing_cache_source": {"repeat": TIMING_CACHE_SOURCE_REPEAT,
                                 "path": "source.repeat_1/timing.cache",
                                 "sha256": SOURCE_TIMING_CACHE_SHA256,
                                 "copied_per_build": True},
        "calibration_cache_source": {"repeat": TIMING_CACHE_SOURCE_REPEAT,
                                      "path": "source.repeat_1/calibration.cache",
                                      "sha256": SOURCE_CALIBRATION_CACHE_SHA256},
        "prediction_payload_version": PREDICTION_PAYLOAD_VERSION,
        "dataset_split": DATASET_SPLIT,
        "round_order": [1, 2, 3],
        "status": status,
        "global_g0": "review_required",
        "records": records,
        "build_variability": _aggregate_build_metrics(records),
        "step_a_fresh_cache_build_variability": source["source_summary_data"].get("build_variability_coco_xml"),
        "source_step_a_repeat_1": {
            "capture_report_sha256": sha256(source["root"] / "repeat_1/capture/capture_report.json"),
            "prediction_payload_sha256": prediction_payload_hash(source_predictions),
            "metrics": source_capture.get("metrics"),
            "coco_xml": source_size.get("metrics"),
        },
        "classification": status,
        "review_flags": review_flags,
        "next_action": "STOP for review; do not add editable tactic control, change calibration, retrain or scale to 15 models from this runner.",
        "limitations": "Prospective feasibility check selected source repeat_1 timing.cache after Step-A results were known. Ordinary timing-cache reuse is not a tactic lock. Output-cache change is reported, not interpreted as tactic evidence; output-cache equality would not prove full cache hits. Same calibration cache isolates neither calibration-sampling variance nor all builder/runtime effects. Shared lab telemetry is sampled before/after phases and does not prove GPU isolation.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_fn(output / "repeat_summary.json", summary)
    print(f"DONE: {output / 'repeat_summary.json'}")
    return 0


def main(argv=None, *, repo_override=None, helpers=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "prepare", "build", "evaluate"), default="all")
    parser.add_argument("--repeat", type=int, choices=ENGINE_REPEATS)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confirm-desktop-process", action="append", default=[], metavar="PID=PATH",
                        help="Explicitly confirm a current nvidia-smi desktop row; no /proc access is used")
    args = parser.parse_args(argv)
    repo = Path(repo_override if repo_override is not None else Path(__file__).resolve().parents[1]).resolve()
    helpers = dict(helpers or {})
    try:
        source_root, output = validate_output_target(repo, args.out_dir)
        if str(args.device) != "0":
            raise ValueError("Timing-cache replay protocol is locked to --device 0")
    except ValueError as error:
        parser.error(str(error))
    if args.phase == "build" and args.repeat is None:
        parser.error("--repeat is required for build phase")
    if args.phase != "build" and args.repeat is not None:
        parser.error("--repeat is only valid for build phase")
    if args.phase == "build" and output.exists() and not (output / "study_manifest.json").exists():
        parser.error("Replay output exists without a study manifest; preserve and inspect it")
    parse_desktop = helpers.get("parse_desktop_confirmations", parse_desktop_confirmations)
    try:
        args.confirmed_desktop = parse_desktop(args.confirm_desktop_process)
    except ValueError as error:
        parser.error(str(error))

    source = validate_source_contract(repo)
    if args.phase == "build":
        build(repo, output, args.repeat, args.confirmed_desktop)
        return 0
    if args.phase == "all":
        if output.exists():
            parser.error("Output exists; preserve it and do not resume or overwrite")
        probe, gpu_before, gpu_binding = preflight(repo, source, args, helpers)
        output.mkdir(parents=True)
        write_json(output / "study_manifest.json", {
            "schema_version": 1, "study": STUDY, "source_study": SOURCE_STUDY,
            "source_study_root": str(source_root), "source_result_commit": SOURCE_RESULT_COMMIT,
            "source_study_code_commit": source["manifest"].get("git_commit"),
            "git_commit": git_value(repo, "rev-parse", "HEAD"),
            "git_status": git_value(repo, "status", "--short"),
            "dataset_split": DATASET_SPLIT, "engine_repeats": list(ENGINE_REPEATS),
            "build_order": [1, 2, 3], "capture_order": [1, 2, 3],
            "settings": REPLAY_SETTINGS, "source_step_a_settings": SOURCE_SETTINGS,
            "source_weights_sha256": FROZEN_WEIGHTS_SHA256,
            "onnx_sha256": SOURCE_ONNX_SHA256,
            "calibration_cache_input": {"source_repeat": 1, "sha256": SOURCE_CALIBRATION_CACHE_SHA256},
            "timing_cache_input": {"source_repeat": 1, "sha256": SOURCE_TIMING_CACHE_SHA256,
                                    "copy_per_build": True, "ignore_mismatch": False},
            "environment": probe["environment"], "environment_preflight": probe,
            "gpu_before": gpu_before, "gpu_identity_binding": {"device_argument": args.device, **gpu_binding},
            "operator_confirmations": [{"pid": pid, "reported_path": path}
                                        for pid, path in sorted(args.confirmed_desktop.items())],
            "protocol_variant": REPLAY_VARIANT,
            "process_policy": "parent CPU-only orchestration; one fresh build child per repeat; one capture and one CPU verification child per engine",
            "scope": "Three independent timing-cache replay builds and one CCTSDB2021/dev capture per build. No export, retraining, test split, benchmark or calibration intervention.",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        for repeat in ENGINE_REPEATS:
            log = output / f"build_{repeat}.log"
            command = _build_child_command(repo, output, repeat, args.confirmed_desktop)
            print(f"START BUILD {repeat}/3; verbose log: {log}", flush=True)
            with log.open("x", encoding="utf-8") as handle:
                process = subprocess.run(command, cwd=repo, stdout=handle, stderr=subprocess.STDOUT, check=False)
            with log.open("rb") as source_log, gzip.open(output / f"build_{repeat}.log.gz", "xb") as compressed:
                while True:
                    chunk = source_log.read(1024 * 1024)
                    if not chunk:
                        break
                    compressed.write(chunk)
            if process.returncode:
                raise RuntimeError(f"Build {repeat} failed; inspect {log}; partial outputs preserved")
            print(f"FINISHED BUILD {repeat}/3", flush=True)
        source_after = validate_source_contract(repo)
        if source_after["input_hashes"] != source["input_hashes"]:
            raise RuntimeError("Step-A source inputs changed during replay build phase")
        return evaluate(repo, output, args, source_after, helpers)
    if args.phase == "prepare":
        if output.exists():
            parser.error("Output exists; preserve it and do not overwrite")
        probe, gpu_before, gpu_binding = preflight(repo, source, args, helpers)
        output.mkdir(parents=True)
        write_json(output / "study_manifest.json", {
            "schema_version": 1, "study": STUDY, "source_study": SOURCE_STUDY,
            "source_result_commit": SOURCE_RESULT_COMMIT, "source_study_code_commit": source["manifest"].get("git_commit"),
            "git_commit": git_value(repo, "rev-parse", "HEAD"), "settings": REPLAY_SETTINGS,
            "source_weights_sha256": FROZEN_WEIGHTS_SHA256, "onnx_sha256": SOURCE_ONNX_SHA256,
            "calibration_cache_input": {"source_repeat": 1, "sha256": SOURCE_CALIBRATION_CACHE_SHA256},
            "timing_cache_input": {"source_repeat": 1, "sha256": SOURCE_TIMING_CACHE_SHA256, "ignore_mismatch": False},
            "environment": probe["environment"], "environment_preflight": probe, "gpu_before": gpu_before,
            "gpu_identity_binding": {"device_argument": args.device, **gpu_binding}, "protocol_variant": REPLAY_VARIANT,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"PREPARED: {output / 'study_manifest.json'}")
        return 0
    if args.phase == "evaluate":
        probe, _gpu_before, _gpu_binding = preflight(repo, source, args, helpers)
        return evaluate(repo, output, args, source, helpers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
