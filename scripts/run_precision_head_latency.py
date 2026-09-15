#!/usr/bin/env python3
"""Measure locked YOLO11n precision-head engine latency on one RTX8000.

The parent owns the GPU phase lock and runs one fresh child per engine/round.
The child loads an existing engine through Ultralytics, preloads the image pool,
and measures synchronous ``model.predict`` wall time.  No engine is built or
exported by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_architecture_matrix import GpuPhaseLock

STUDY = "yolo11n_precision_head_latency_v1"
OUTPUT_DIR_NAME = "server_yolo11n_precision_head_latency_v1"
ATTEMPT2_DIR_NAME = "server_yolo11n_precision_head_ablation_v1_attempt2"
ACCURACY_DIR_NAME = "precision_head_paired_analysis_v1"
DATASET_SPLIT = "CCTSDB2021/dev"
ENGINE_KEYS = (
    "fp16",
    "baseline_int8_1", "bbox_fp32_1", "classification_fp32_1", "both_fp32_1",
    "baseline_int8_2", "bbox_fp32_2", "classification_fp32_2", "both_fp32_2",
    "baseline_int8_3", "bbox_fp32_3", "classification_fp32_3", "both_fp32_3",
)
ENGINE_MODELS = {
    "fp16": "fp16",
    **{f"{arm}_{repeat}": arm for arm in ("baseline_int8", "bbox_fp32", "classification_fp32", "both_fp32") for repeat in (1, 2, 3)},
}
ENGINE_REPEATS = {key: (None if key == "fp16" else int(key.rsplit("_", 1)[1])) for key in ENGINE_KEYS}
ARM_NAMES = ("baseline_int8", "bbox_fp32", "classification_fp32", "both_fp32")
ROUNDS = (1, 2, 3)
POOL_SIZE = 256
WARMUP_CALLS = 200
MEASURED_CALLS = 1000
SEED = 20260916
EXPECTED_IMAGES = 1636
EXPECTED_ENV_KEYS = ("torch", "ultralytics", "tensorrt", "numpy", "cuda", "gpu", "pycocotools")
EXPECTED_ENV = {
    "torch": "2.5.1+cu121",
    "ultralytics": "8.4.102",
    "tensorrt": "10.16.1.11",
    "numpy": "2.4.4",
    "cuda": "12.1",
    "gpu": "Quadro RTX 8000",
    "pycocotools": "2.0.10",
}
RUNTIME_OPTIONS = {
    "imgsz": 640,
    "batch": 1,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": False,
    "task": "detect",
    "verbose": False,
}
PERCENTILE_METHOD = "numpy.quantile(method='linear')"


def canonical_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Output exists; preserve it and do not overwrite: {path}")


def validate_output_target(repo: Path, output: Path) -> Path:
    expected = (repo / "results/measurement_audit_v1" / OUTPUT_DIR_NAME).resolve()
    if output.resolve() != expected:
        raise ValueError(f"Output must be exactly {expected}")
    return expected


def percentile_linear(values, fraction: float) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Latency samples must be a non-empty finite vector")
    return float(np.quantile(array, fraction, method="linear"))


def latency_statistics(values) -> dict:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("Latency samples must be positive finite values")
    mean = float(array.mean())
    return {
        "n_calls": int(len(array)),
        "mean_ms": mean,
        "median_ms": percentile_linear(array, 0.50),
        "p95_ms": percentile_linear(array, 0.95),
        "p99_ms": percentile_linear(array, 0.99),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "serial_fps": 1000.0 / mean,
        "percentile_method": PERCENTILE_METHOD,
    }


def measure_predict(model, images, predict, synchronize, timer_ns=time.perf_counter_ns,
                    warmup: int = WARMUP_CALLS, samples: int = MEASURED_CALLS) -> list[float]:
    """Run warmup and measured calls; return only measured wall-clock samples.

    The injectable clock/synchronizer makes the timing boundary auditable in
    CPU/mock tests without importing CUDA or TensorRT.
    """
    if not images or warmup < 0 or samples <= 0:
        raise ValueError("images must be non-empty, warmup nonnegative and samples positive")
    for index in range(warmup):
        predict(images[index % len(images)])
    synchronize()
    values = []
    for index in range(samples):
        synchronize()
        start = timer_ns()
        predict(images[index % len(images)])
        synchronize()
        values.append((timer_ns() - start) / 1_000_000.0)
    return values


def engine_schedule() -> list[dict]:
    """Return the locked 39-session schedule, three rotations of four positions."""
    first = list(ENGINE_KEYS)
    orders = [first, first[4:] + first[:4], first[8:] + first[:8]]
    return [
        {"sequence": index, "round": round_id, "engine_key": engine_key,
         "model": ENGINE_MODELS[engine_key], "build_repeat": ENGINE_REPEATS[engine_key]}
        for round_id, order in enumerate(orders, start=1)
        for index, engine_key in enumerate(order, start=1 + (round_id - 1) * len(ENGINE_KEYS))
    ]


def validate_schedule(schedule: list[dict]) -> None:
    expected = engine_schedule()
    if schedule != expected:
        raise ValueError("Latency schedule differs from locked 39-session round rotation")
    if len(schedule) != 39 or {item["engine_key"] for item in schedule} != set(ENGINE_KEYS):
        raise ValueError("Latency schedule must contain every one of 13 engines in every round")


def select_image_pool(image_names: list[str], seed: int = SEED, pool_size: int = POOL_SIZE) -> list[str]:
    names = sorted(image_names)
    if len(names) != EXPECTED_IMAGES or len(set(names)) != EXPECTED_IMAGES:
        raise ValueError(f"Expected {EXPECTED_IMAGES} unique dev image IDs")
    if pool_size <= 0 or pool_size > len(names):
        raise ValueError("Invalid image pool size")
    rng = np.random.Generator(np.random.PCG64(seed))
    selected = np.sort(rng.choice(len(names), size=pool_size, replace=False))
    return [names[int(index)] for index in selected]


def cyclic_sequence(pool_names: list[str], count: int) -> list[str]:
    if not pool_names or count <= 0:
        raise ValueError("Pool and sequence count must be positive")
    return [pool_names[index % len(pool_names)] for index in range(count)]


def make_image_pool_manifest(image_names: list[str], images_dir: Path, seed: int = SEED) -> dict:
    selected = select_image_pool(image_names, seed)
    files = []
    for name in selected:
        path = images_dir / name
        if path.parent != images_dir or not path.is_file():
            raise FileNotFoundError(f"Missing selected dev image: {path}")
        files.append({"image": name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    measured = cyclic_sequence(selected, MEASURED_CALLS)
    warmup = cyclic_sequence(selected, WARMUP_CALLS)
    return {
        "schema_version": 1,
        "dataset_split": DATASET_SPLIT,
        "source_image_count": len(image_names),
        "pool_size": len(selected),
        "seed": seed,
        "generator": "numpy.random.Generator(PCG64)",
        "selection": "sample without replacement from sorted dev image IDs, then sort selected IDs",
        "selected_files": files,
        "warmup_sequence": warmup,
        "measured_sequence": measured,
        "warmup_sequence_sha256": sha256_bytes(canonical_json(warmup)),
        "measured_sequence_sha256": sha256_bytes(canonical_json(measured)),
        "same_sequence_for_every_engine": True,
        "disk_decode_in_timing": False,
    }


def parse_gpu_identity(snapshot_or_device):
    raw = snapshot_or_device.get("device") if isinstance(snapshot_or_device, dict) else snapshot_or_device
    if not isinstance(raw, str):
        raise ValueError("GPU device telemetry is missing")
    rows = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("GPU device telemetry must contain exactly one device row")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) < 3 or not all(fields[:3]):
        raise ValueError(f"GPU device telemetry is malformed: {raw!r}")
    return {"uuid": fields[0], "name": fields[1], "driver_version": fields[2]}


def validate_gpu_identity(snapshot, expected: dict) -> dict:
    actual = parse_gpu_identity(snapshot)
    mismatches = {key: {"expected": expected[key], "actual": actual[key]}
                  for key in ("uuid", "name", "driver_version") if expected.get(key) != actual.get(key)}
    if mismatches:
        raise ValueError(f"GPU identity differs from locked RTX8000 identity: {mismatches}")
    return {"expected": expected, "actual": actual, "matched": True}


def confirmation_cli_args(confirmations: dict[int, str]) -> list[str]:
    return [part for pid, path in sorted(confirmations.items()) for part in ("--confirm-desktop-process", f"{pid}={path}")]


def expected_capture_runtime(report: dict) -> None:
    runtime = report.get("runtime_arguments", {})
    for key, expected in {"split": "val", "imgsz": 640, "batch": 1, "workers": 0,
                          "task": "detect", "mode": "val", "conf": 0.001,
                          "iou": 0.7, "max_det": 300, "rect": False,
                          "plots": False, "verbose": False, "save_json": False,
                          "save_txt": False}.items():
        if runtime.get(key) != expected:
            raise ValueError(f"Existing accuracy capture runtime differs for {key}: {runtime.get(key)!r}")


def validate_engine_file(path: Path, expected_hash: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing existing engine binary: {path}")
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError(f"Engine bytes differ from manifest before deserialization: {path}")
    return actual


def _check_environment_subset(actual: dict, expected: dict, label: str, require_all: bool = False) -> None:
    for key in ("torch", "ultralytics", "tensorrt", "numpy", "cuda", "gpu", "pycocotools"):
        if key not in actual:
            if require_all:
                raise ValueError(f"{label} environment is missing {key}")
            continue
        if actual.get(key) != expected.get(key):
            raise ValueError(f"{label} environment.{key} differs: {actual.get(key)!r} != {expected.get(key)!r}")


def _load_accuracy_links(accuracy_root: Path, models: list[str]) -> dict:
    summary_path = accuracy_root / "analysis_summary.json"
    points_path = accuracy_root / "point_estimates.json"
    ci_path = accuracy_root / "contrast_ci.json"
    for path in (summary_path, points_path, ci_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing paired accuracy artifact: {path}")
    summary = read_json(summary_path)
    points = read_json(points_path)
    contrasts = read_json(ci_path)
    if summary.get("status") != "completed" or summary.get("estimator_id") != "coco_xml_paired_image_bootstrap_v1":
        raise ValueError("Paired accuracy analysis is not a completed locked analysis")
    if set(points.get("models", {})) != set(models):
        raise ValueError("Paired accuracy point table does not cover latency models")
    if len(contrasts.get("contrasts", {})) != 10:
        raise ValueError("Paired accuracy CI table does not contain all ten fixed contrasts")
    refs = {
        "analysis_summary": {"path": str(summary_path.resolve()), "sha256": sha256_file(summary_path)},
        "point_estimates": {"path": str(points_path.resolve()), "sha256": sha256_file(points_path)},
        "contrast_ci": {"path": str(ci_path.resolve()), "sha256": sha256_file(ci_path)},
        "estimator_id": summary["estimator_id"],
        "models": {},
    }
    for model in models:
        contrast_refs = [name for name, left, right in (
            ("bbox_minus_baseline", "bbox_fp32", "baseline_int8"),
            ("classification_minus_baseline", "classification_fp32", "baseline_int8"),
            ("both_minus_baseline", "both_fp32", "baseline_int8"),
            ("bbox_minus_classification", "bbox_fp32", "classification_fp32"),
            ("both_minus_bbox", "both_fp32", "bbox_fp32"),
            ("both_minus_classification", "both_fp32", "classification_fp32"),
            ("baseline_minus_fp16", "baseline_int8", "fp16"),
            ("bbox_minus_fp16", "bbox_fp32", "fp16"),
            ("classification_minus_fp16", "classification_fp32", "fp16"),
            ("both_minus_fp16", "both_fp32", "fp16"),
        ) if model in (left, right)]
        refs["models"][model] = {
            "point_estimate_ref": {"file": "point_estimates.json", "model": model},
            "contrast_ci_refs": contrast_refs,
            "evaluator": "COCO/XML diagnostic points; CIs are fixed contrast CIs, not per-model latency CIs",
        }
    return refs


def validate_inputs(repo: Path, attempt2_root: Path, fp16_capture: Path,
                    fp16_verification: Path, accuracy_root: Path, images_dir: Path) -> dict:
    expected_attempt2 = (repo / "results/measurement_audit_v1" / ATTEMPT2_DIR_NAME).resolve()
    expected_fp16_capture = (repo / "results/measurement_audit_v1/server_fp16_capture_v1").resolve()
    expected_fp16_verification = (repo / "results/measurement_audit_v1/server_native_size_v1").resolve()
    expected_accuracy = (repo / "results/measurement_audit_v1" / ACCURACY_DIR_NAME).resolve()
    if attempt2_root.resolve() != expected_attempt2 or fp16_capture.resolve() != expected_fp16_capture or fp16_verification.resolve() != expected_fp16_verification or accuracy_root.resolve() != expected_accuracy:
        raise ValueError("Latency inputs must use the locked attempt2, FP16, verification and accuracy paths")
    manifest_path = attempt2_root / "study_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing attempt2 study manifest: {manifest_path}")
    source_manifest = read_json(manifest_path)
    if source_manifest.get("study") != "yolo11n_precision_head_ablation_v1" or source_manifest.get("execution_attempt") != 2:
        raise ValueError("Latency inputs must be the accepted precision-head attempt2")
    if source_manifest.get("dataset_split") != DATASET_SPLIT:
        raise ValueError("Latency inputs are not dev-only")
    expected_env = source_manifest.get("environment", {})
    for key, value in EXPECTED_ENV.items():
        if expected_env.get(key) != value:
            raise ValueError(f"Attempt2 environment contract differs for {key}")
    expected_gpu = parse_gpu_identity(source_manifest.get("gpu_before"))

    specs = {}
    fp16_report_path = fp16_capture / "capture_report.json"
    fp16_report = read_json(fp16_report_path)
    expected_capture_runtime(fp16_report)
    if fp16_report.get("status") != "pass" or fp16_report.get("dataset_split") != DATASET_SPLIT:
        raise ValueError("FP16 reference capture is not a passed dev capture")
    _check_environment_subset(fp16_report.get("environment", {}), expected_env, "FP16 capture")
    fp16_engine = Path(fp16_report["runtime_arguments"]["model"])
    fp16_hash = validate_engine_file(fp16_engine, fp16_report["model_sha256"])
    fp16_summary = read_json(fp16_verification / "verification_summary.json")
    if fp16_summary.get("model_sha256") != fp16_hash or fp16_summary.get("native_matching_status") != "pass":
        raise ValueError("FP16 verification is not bound to the direct engine")
    specs["fp16"] = {
        "engine_key": "fp16", "model": "fp16", "arm": "fp16", "build_repeat": None,
        "engine_path": str(fp16_engine.resolve()), "engine_sha256": fp16_hash,
        "capture_report_sha256": sha256_file(fp16_report_path),
    }
    for arm in ARM_NAMES:
        arm_dir = attempt2_root / arm
        for repeat in (1, 2, 3):
            key = f"{arm}_{repeat}"
            build_path = arm_dir / f"repeat_{repeat}" / "build_manifest.json"
            capture_path = arm_dir / f"repeat_{repeat}" / "capture" / "capture_report.json"
            build = read_json(build_path)
            capture = read_json(capture_path)
            if build.get("study") != source_manifest.get("study") or build.get("arm") != arm or build.get("repeat") != repeat:
                raise ValueError(f"Build provenance differs for {key}")
            if build.get("termination_status") != "completed":
                raise ValueError(f"Build is not complete for {key}")
            if build.get("environment") != expected_env:
                raise ValueError(f"Build environment differs from attempt2 for {key}")
            for field in (
                "source_result_commit", "source_study_code_commit", "source_study_manifest_sha256",
                "source_summary_sha256", "source_weights_sha256", "frozen_weights_measured_sha256",
                "onnx_sha256", "calibration_cache_input_sha256", "timing_cache_input_sha256",
                "baseline_reference_study", "baseline_reference_result_commit", "builder_flags", "settings",
            ):
                expected_value = source_manifest.get(field)
                if expected_value is not None and build.get(field) != expected_value:
                    raise ValueError(f"Build source provenance differs for {key}: {field}")
            expected_capture_runtime(capture)
            if capture.get("status") != "pass" or capture.get("dataset_split") != DATASET_SPLIT:
                raise ValueError(f"Capture is not a passed dev capture for {key}")
            _check_environment_subset(capture.get("environment", {}), expected_env, f"{key} capture")
            if capture.get("model_sha256") != build.get("engine_sha256"):
                raise ValueError(f"Build/capture engine link differs for {key}")
            engine_path = Path(capture["runtime_arguments"]["model"])
            engine_hash = validate_engine_file(engine_path, build["engine_sha256"])
            specs[key] = {
                "engine_key": key, "model": arm, "arm": arm, "build_repeat": repeat,
                "engine_path": str(engine_path.resolve()), "engine_sha256": engine_hash,
                "build_manifest_sha256": sha256_file(build_path),
                "capture_report_sha256": sha256_file(capture_path),
            }
    if len(specs) != 13 or len({row["engine_path"] for row in specs.values()}) != 13 or len({row["engine_sha256"] for row in specs.values()}) != 13:
        raise ValueError("Latency inventory must bind 13 distinct existing engine binaries")

    payload = read_json(fp16_capture / "validator_predictions.json")
    names = [record.get("image") for record in payload.get("records", [])]
    if len(names) != EXPECTED_IMAGES or len(set(names)) != EXPECTED_IMAGES:
        raise ValueError("FP16 capture does not provide the locked 1636 unique dev image IDs")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing dev image directory: {images_dir}")
    pool = make_image_pool_manifest(names, images_dir)
    accuracy = _load_accuracy_links(accuracy_root, ["fp16", *ARM_NAMES])
    source_artifact_commit = None
    if accuracy_root.is_dir():
        source_artifact_commit = read_json(accuracy_root / "analysis_summary.json").get("artifact_commit")
    return {
        "source_manifest": source_manifest,
        "source_manifest_path": manifest_path,
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_artifact_commit": source_artifact_commit,
        "expected_environment": expected_env,
        "expected_gpu": expected_gpu,
        "specs": specs,
        "pool": pool,
        "accuracy": accuracy,
        "fp16_capture_path": fp16_capture,
        "fp16_verification_path": fp16_verification,
    }


def child_command(repo: Path, spec: dict, round_id: int, session_dir: Path,
                  run_manifest: Path, pool_manifest: Path, images_dir: Path,
                  device: str, confirmations: dict[int, str]) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "--child",
        "--engine-key", spec["engine_key"], "--engine", spec["engine_path"],
        "--engine-sha256", spec["engine_sha256"], "--round", str(round_id),
        "--session-dir", str(session_dir), "--run-manifest", str(run_manifest),
        "--pool-manifest", str(pool_manifest), "--images-dir", str(images_dir),
        "--device", device, *confirmation_cli_args(confirmations),
    ]


def _load_pool_images(pool_manifest_path: Path, images_dir: Path):
    import cv2
    manifest = read_json(pool_manifest_path)
    selected = manifest.get("selected_files", [])
    if len(selected) != POOL_SIZE or len({row.get("image") for row in selected}) != POOL_SIZE:
        raise ValueError("Image pool manifest is not 256 unique files")
    images = []
    shapes = []
    for row in selected:
        name = row["image"]
        path = images_dir / name
        if path.parent != images_dir or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Image bytes changed after pool preparation: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Image could not be decoded: {path}")
        images.append(image)
        shapes.append(list(image.shape))
    selected_names = [row["image"] for row in selected]
    expected_warmup = cyclic_sequence(selected_names, WARMUP_CALLS)
    expected_measured = cyclic_sequence(selected_names, MEASURED_CALLS)
    if manifest.get("warmup_sequence") != expected_warmup or manifest.get("measured_sequence") != expected_measured:
        raise ValueError("Image sequence differs from the canonical cyclic pool order")
    if (manifest.get("warmup_sequence_sha256") != sha256_bytes(canonical_json(expected_warmup)) or
            manifest.get("measured_sequence_sha256") != sha256_bytes(canonical_json(expected_measured))):
        raise ValueError("Image sequence hash differs from the canonical cyclic pool order")
    aspect_ratios = {round(shape[1] / shape[0], 8) for shape in shapes if len(shape) >= 2 and shape[0] > 0}
    if len(aspect_ratios) < 2:
        raise ValueError("Selected pool does not contain the required varied source aspect ratios")
    return manifest, images, shapes


def _runtime_environment() -> dict:
    from uniform_build_repeat import environment
    return environment()


def run_child(args) -> int:
    session_dir = args.session_dir.resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = read_json(args.run_manifest.resolve())
    expected_env = run_manifest["environment"]
    expected_gpu = run_manifest["gpu_identity"]
    expected_hash = args.engine_sha256
    engine = args.engine.resolve()
    expected_spec = run_manifest.get("engine_inventory", {}).get(args.engine_key)
    if not isinstance(expected_spec, dict) or expected_spec.get("engine_path") != str(engine) or expected_spec.get("engine_sha256") != expected_hash:
        raise ValueError("Child engine binding does not match the parent inventory")
    if not any(item.get("round") == args.round and item.get("engine_key") == args.engine_key for item in run_manifest.get("round_schedule", [])):
        raise ValueError("Child round/engine is not in the locked parent schedule")
    validate_engine_file(engine, expected_hash)
    confirmations = _parse_confirmations(args.confirm_desktop_process)

    from uniform_build_repeat import ensure_idle, snapshot
    gpu_before = snapshot(confirmations)
    try:
        ensure_idle(gpu_before)
    except Exception as error:
        write_json(session_dir / "telemetry_violation.json", {
            "schema_version": 1, "study": STUDY, "status": "blocked",
            "phase": "before_session", "error": str(error), "gpu_snapshot": gpu_before,
        })
        raise
    before_binding = validate_gpu_identity(gpu_before, expected_gpu)
    runtime_env = _runtime_environment()
    _check_environment_subset(runtime_env, expected_env, "Child", require_all=True)

    pool_manifest, images, shapes = _load_pool_images(args.pool_manifest.resolve(), args.images_dir.resolve())
    import torch
    import ultralytics
    from ultralytics import YOLO

    # Ultralytics receives the existing engine path and handles its own plan
    # metadata/header. This is intentionally not a direct TensorRT deserializer.
    model = YOLO(str(engine))
    runtime = {**RUNTIME_OPTIONS, "device": args.device}

    def predict(image):
        return model.predict(source=image, **runtime)

    def synchronize():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    raw = measure_predict(model, images, predict, synchronize)
    gpu_after = snapshot(confirmations)
    try:
        ensure_idle(gpu_after)
    except Exception as error:
        write_json(session_dir / "telemetry_violation.json", {
            "schema_version": 1, "study": STUDY, "status": "blocked",
            "phase": "after_session", "error": str(error),
            "gpu_before": gpu_before, "gpu_after": gpu_after,
        })
        raise
    after_binding = validate_gpu_identity(gpu_after, expected_gpu)
    peak = None
    if torch.cuda.is_available():
        peak = int(torch.cuda.max_memory_allocated())
    stats = latency_statistics(raw)
    measured_sequence = pool_manifest["measured_sequence"]
    session = {
        "schema_version": 1,
        "study": STUDY,
        "status": "completed",
        "round": args.round,
        "engine_key": args.engine_key,
        "engine_sha256": expected_hash,
        "engine_path": str(engine),
        "engine_bytes": engine.stat().st_size,
        "dataset_split": DATASET_SPLIT,
        "runtime": runtime,
        "image_pool": {
            "manifest_path": str(args.pool_manifest.resolve()),
            "pool_size": len(images),
            "pool_seed": pool_manifest["seed"],
            "pool_sequence_sha256": pool_manifest["measured_sequence_sha256"],
            "measured_sequence_start_index": 0,
            "measured_sequence_calls": len(measured_sequence),
            "decoded_shapes": shapes,
            "distinct_decoded_shapes": sorted({tuple(shape) for shape in shapes}),
            "distinct_source_aspect_ratios": sorted({round(shape[1] / shape[0], 8) for shape in shapes}),
            "same_image_bytes_checked": True,
        },
        "warmup_calls": WARMUP_CALLS,
        "measured_calls": MEASURED_CALLS,
        "warmup_in_timing": False,
        "disk_decode_in_timing": False,
        "model_load_in_timing": False,
        "initial_allocation_in_timing": False,
        "synchronize_before_timer": True,
        "synchronize_after_predict": True,
        "timer": "time.perf_counter_ns monotonic high-resolution clock",
        "latency_ms": stats,
        "raw_latency_ms": raw,
        "torch_allocated_peak_bytes": peak,
        "torch_allocated_peak_scope": "Torch allocator only; not total TensorRT memory" if peak is not None else "unavailable",
        "environment": runtime_env,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "gpu_identity_binding": {"before": before_binding, "after": after_binding},
        "scope": "Synchronous batch-1 end-to-end model.predict wall time from decoded CPU image through preprocessing/H2D/inference/postprocessing/NMS; not pure TensorRT kernel time.",
    }
    write_json(session_dir / "session.json", session)
    print(f"DONE SESSION: {session_dir / 'session.json'}", flush=True)
    return 0


def _parse_confirmations(values: list[str]) -> dict[int, str]:
    from uniform_build_repeat import parse_desktop_confirmations
    return parse_desktop_confirmations(values)


def _session_summary(session: dict) -> dict:
    return {key: session[key] for key in ("round", "engine_key", "engine_sha256", "engine_bytes", "latency_ms", "warmup_calls", "measured_calls", "gpu_identity_binding", "environment")}


def _aggregate_engine_sessions(sessions: list[dict]) -> dict:
    if len(sessions) != 3 or {row["round"] for row in sessions} != {1, 2, 3}:
        raise ValueError("Each engine must have exactly one completed session in every round")
    raw = [value for session in sessions for value in session["raw_latency_ms"]]
    return {
        "engine_key": sessions[0]["engine_key"],
        "engine_sha256": sessions[0]["engine_sha256"],
        "rounds": [_session_summary(session) for session in sorted(sessions, key=lambda row: row["round"])],
        "pooled_calls_3000": latency_statistics(raw),
        "aggregation_note": "Pooled calls summarize 3 measurement rounds for this one build; they are not 3,000 independent builds.",
    }


def _build_report(summary: dict) -> str:
    lines = [
        "# YOLO11n precision-head latency study v1",
        "",
        "Status: `latency_completed_review_required`.",
        "",
        f"- Study: `{STUDY}`; device identity: `{summary['gpu_identity']['name']}` / `{summary['gpu_identity']['uuid']}`",
        f"- Sessions: {summary['sessions_completed']}/39; each session has {WARMUP_CALLS} warmup and {MEASURED_CALLS} measured calls.",
        "- Timing is synchronous batch-1 `model.predict` wall time from decoded CPU image through preprocessing/H2D/inference/postprocessing/NMS. Disk decode, model load, initial allocation and warmup are excluded; this is not pure TensorRT kernel time.",
        "- Accuracy references are COCO/XML points and fixed paired CIs from the accepted analysis; no CI is attached to a latency point and no fastest engine is selected.",
        "",
        "## Per-engine latency",
        "",
        "| Engine | Arm/build | Mean ms (3,000 calls) | P50 | P95 | P99 | Serial FPS | Accuracy point/CI reference |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for key in ENGINE_KEYS:
        row = summary["per_engine"][key]
        stats = row["pooled_calls_3000"]
        spec = summary["engine_inventory"][key]
        accuracy = summary["accuracy_links"]["models"][spec["model"]]
        lines.append(
            f"| {key} | {spec['model']}/{spec['build_repeat'] or 'reference'} | {stats['mean_ms']:.4f} | {stats['median_ms']:.4f} | {stats['p95_ms']:.4f} | {stats['p99_ms']:.4f} | {stats['serial_fps']:.4f} | point `point_estimates.json#{spec['model']}`; CI refs `{', '.join(accuracy['contrast_ci_refs'])}` |"
        )
    lines.extend([
        "",
        "## Arm summaries",
        "",
        "Arm rows pool all three builds and all three rounds; build-level and round-level rows remain in `latency_summary.json` and each session JSON.",
        "",
        "| Arm | Builds | Sessions | Pooled calls | Mean ms | P95 ms | Serial FPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ("fp16", *ARM_NAMES):
        row = summary["arm_summary"][arm]
        stats = row["pooled_calls"]
        lines.append(f"| {arm} | {row['builds']} | {row['sessions']} | {stats['n_calls']} | {stats['mean_ms']:.4f} | {stats['p95_ms']:.4f} | {stats['serial_fps']:.4f} |")
    lines.extend([
        "",
        "## Limitations",
        "",
        "This is a single shared RTX8000/server diagnostic with sampled GPU telemetry and a finite fixed image pool. The three rounds reduce position confounding but are not a complete Latin square; results remain conditional on the fixed environment, runtime, engine bytes, image pool and postprocessing options. Desktop processes may remain visible under the operator-confirmed exception. Power/energy are not endpoints here, and no production-confidence benchmark, cross-device claim, engine rebuild, accuracy selection, official-test evaluation, retraining or 15-model expansion is performed.",
        "",
    ])
    return "\n".join(lines)


def _run_parent(args) -> int:
    repo = Path(__file__).resolve().parents[1]
    output = validate_output_target(repo, args.out_dir.resolve())
    ensure_output_absent(output)
    if str(args.device) != "0":
        raise ValueError("Latency study is locked to --device 0 on the accepted RTX8000")
    confirmations = _parse_confirmations(args.confirm_desktop_process)
    attempt2_root = (repo / args.attempt2_root).resolve()
    fp16_capture = (repo / args.fp16_capture).resolve()
    fp16_verification = (repo / args.fp16_verification).resolve()
    accuracy_root = (repo / args.accuracy_root).resolve()
    images_dir = args.images_dir.resolve()

    # Validate all 13 direct engine bytes and all non-GPU input contracts before
    # acquiring the phase or allowing a child to deserialize any engine.
    context = validate_inputs(repo, attempt2_root, fp16_capture, fp16_verification, accuracy_root, images_dir)
    schedule = engine_schedule()
    validate_schedule(schedule)

    from run_uniform_inference_repeat import run_environment_preflight
    preflight = run_environment_preflight(repo)
    _check_environment_subset(preflight["environment"], context["expected_environment"], "Parent", require_all=True)
    from uniform_build_repeat import ensure_idle, snapshot

    lock_path = repo / "results/architecture_matrix_v1/.gpu_phase.lock"
    with GpuPhaseLock(lock_path, STUDY):
        gpu_before = snapshot(confirmations)
        ensure_idle(gpu_before)
        gpu_binding = validate_gpu_identity(gpu_before, context["expected_gpu"])
        output.mkdir(parents=True)
        pool_path = output / "image_pool_manifest.json"
        write_json(pool_path, context["pool"])
        current_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        run_manifest_path = output / "study_manifest.json"
        manifest = {
            "schema_version": 1,
            "study": STUDY,
            "dataset_split": DATASET_SPLIT,
            "protocol_status": "server-run-authorized_only_after_Astra_code_review",
            "current_code_commit": current_commit,
            "source_attempt2_artifact_commit": context["source_artifact_commit"],
            "source_attempt2_manifest": {"path": str(context["source_manifest_path"]), "sha256": context["source_manifest_sha256"]},
            "source_execution_git_commit": context["source_manifest"].get("execution_git_commit"),
            "environment": context["expected_environment"],
            "environment_preflight": preflight,
            "gpu_identity": context["expected_gpu"],
            "gpu_before_parent": gpu_before,
            "gpu_identity_binding_parent": gpu_binding,
            "operator_confirmations": [{"pid": pid, "reported_path": path} for pid, path in sorted(confirmations.items())],
            "engine_inventory": context["specs"],
            "round_schedule": schedule,
            "runtime": {**RUNTIME_OPTIONS, "device": args.device},
            "warmup_calls": WARMUP_CALLS,
            "measured_calls_per_session": MEASURED_CALLS,
            "rounds": 3,
            "sessions": 39,
            "image_pool_manifest": {"path": str(pool_path), "sha256": sha256_file(pool_path)},
            "accuracy_links": context["accuracy"],
            "process_policy": "parent holds GPU phase lock; one fresh child process per engine per round; no concurrent engines",
            "engine_policy": "existing 13 engine binaries only; direct SHA256 checked before every deserialization; no build/export/rebuild",
            "telemetry_policy": "nvidia-smi snapshots before/after each child session; external competing CUDA workload blocks the run; desktop exception requires current PID/path confirmation",
            "scope": "Accuracy-latency diagnostic on accepted YOLO11n precision-head engines; no deployment selection or cross-device claim.",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(run_manifest_path, manifest)
        completed = []
        for item in schedule:
            spec = context["specs"][item["engine_key"]]
            session_dir = output / f"round_{item['round']}" / item["engine_key"]
            session_dir.mkdir(parents=True)
            command = child_command(repo, spec, item["round"], session_dir, run_manifest_path, pool_path, images_dir, args.device, confirmations)
            print(f"START SESSION {item['sequence']}/39 round {item['round']} {item['engine_key']}", flush=True)
            completed_process = subprocess.run(command, cwd=repo, capture_output=True, text=True, check=False)
            write_text(session_dir / "child.stdout.log", completed_process.stdout or "")
            write_text(session_dir / "child.stderr.log", completed_process.stderr or "")
            if completed_process.returncode:
                write_json(session_dir / "failure.json", {
                    "schema_version": 1, "study": STUDY, "status": "failed",
                    "sequence": item["sequence"], "command": command,
                    "returncode": completed_process.returncode,
                    "stdout_log": "child.stdout.log", "stderr_log": "child.stderr.log",
                })
                raise RuntimeError(f"Latency session failed; preserve partial output and inspect {session_dir}")
            session_path = session_dir / "session.json"
            if not session_path.is_file():
                raise RuntimeError(f"Latency child returned success without session artifact: {session_path}")
            session = read_json(session_path)
            _validate_session(session, item, spec, context["pool"], context["expected_environment"])
            write_json(session_dir / "execution_manifest.json", {
                "schema_version": 1, "study": STUDY, "sequence": item["sequence"],
                "round": item["round"], "engine_key": item["engine_key"],
                "engine_sha256": spec["engine_sha256"], "command": command,
                "returncode": completed_process.returncode,
                "session_sha256": sha256_file(session_path),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            })
            completed.append(session)
            print(f"FINISHED SESSION {item['sequence']}/39 round {item['round']} {item['engine_key']}", flush=True)

        per_engine = {key: _aggregate_engine_sessions([session for session in completed if session["engine_key"] == key]) for key in ENGINE_KEYS}
        arm_summary = {}
        for arm in ("fp16", *ARM_NAMES):
            rows = [session for session in completed if ENGINE_MODELS[session["engine_key"]] == arm]
            if arm == "fp16":
                expected_builds = 1
            else:
                expected_builds = 3
            arm_summary[arm] = {
                "arm": arm, "builds": expected_builds, "sessions": len(rows),
                "build_keys": [key for key in ENGINE_KEYS if ENGINE_MODELS[key] == arm],
                "pooled_calls": latency_statistics([value for session in rows for value in session["raw_latency_ms"]]),
                "aggregation_note": "Arm pooled calls include every listed build and round; call samples are not independent build replicates and no fastest session is selected.",
            }
        final_gpu = snapshot(confirmations)
        ensure_idle(final_gpu)
        summary = {
            "schema_version": 1,
            "study": STUDY,
            "status": "latency_completed_review_required",
            "global_g0": "review_required",
            "dataset_split": DATASET_SPLIT,
            "sessions_completed": len(completed),
            "schedule": schedule,
            "engine_inventory": context["specs"],
            "per_engine": per_engine,
            "arm_summary": arm_summary,
            "accuracy_links": context["accuracy"],
            "runtime": {**RUNTIME_OPTIONS, "device": args.device},
            "warmup_calls": WARMUP_CALLS,
            "measured_calls_per_session": MEASURED_CALLS,
            "image_pool_manifest": {"path": str(pool_path), "sha256": sha256_file(pool_path)},
            "gpu_identity": context["expected_gpu"],
            "gpu_before_parent": gpu_before,
            "gpu_after_parent": final_gpu,
            "environment": context["expected_environment"],
            "telemetry": {"external_workload_detected": False, "status": "complete", "limitation": "Sampled nvidia-smi snapshots cannot prove zero interference between observations."},
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "limitations": [
                "Single shared RTX8000/server diagnostic; desktop exception may remain visible and GPU isolation is not absolute.",
                "Three rounds reduce position confounding but are not a complete Latin-square design.",
                "Latency is synchronous Ultralytics model.predict wall time, not pure TensorRT kernel time; no disk decode/model load/initial allocation/warmup is timed.",
                "Pooled call samples are not independent build replicates; build and round hierarchy remains explicit.",
                "Power/energy are not required endpoints; Torch peak, when present, is allocator-only.",
                "Accuracy links point to COCO/XML points and fixed contrast CIs; no arm is selected and no latency success threshold is applied.",
            ],
        }
        write_json(output / "latency_summary.json", summary)
        write_text(output / "report.md", _build_report(summary))
        print(f"DONE: {output / 'latency_summary.json'}", flush=True)
        return 0


def _validate_session(session: dict, item: dict, spec: dict, pool: dict, expected_env: dict) -> None:
    if session.get("status") != "completed" or session.get("study") != STUDY:
        raise ValueError(f"Session is not completed: {item['engine_key']} round {item['round']}")
    for key, expected in (("round", item["round"]), ("engine_key", item["engine_key"]), ("engine_sha256", spec["engine_sha256"]),
                          ("warmup_calls", WARMUP_CALLS), ("measured_calls", MEASURED_CALLS)):
        if session.get(key) != expected:
            raise ValueError(f"Session contract differs for {item['engine_key']} round {item['round']}: {key}")
    if session.get("runtime") != {**RUNTIME_OPTIONS, "device": "0"}:
        raise ValueError("Session runtime shape/options differ from locked latency contract")
    if session.get("image_pool", {}).get("pool_sequence_sha256") != pool.get("measured_sequence_sha256"):
        raise ValueError("Session image sequence differs from locked pool")
    if session.get("image_pool", {}).get("measured_sequence_calls") != MEASURED_CALLS:
        raise ValueError("Session measured image sequence length differs")
    if session.get("warmup_in_timing") is not False or session.get("disk_decode_in_timing") is not False:
        raise ValueError("Session timing boundary includes excluded work")
    if session.get("synchronize_before_timer") is not True or session.get("synchronize_after_predict") is not True:
        raise ValueError("Session synchronization contract is incomplete")
    _check_environment_subset(session.get("environment", {}), expected_env, "Session", require_all=True)
    latency_statistics(session.get("raw_latency_ms", []))
    if session.get("latency_ms") != latency_statistics(session["raw_latency_ms"]):
        raise ValueError("Session latency summary does not reproduce raw samples")


def main(argv=None) -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=repo / "results/measurement_audit_v1" / OUTPUT_DIR_NAME)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confirm-desktop-process", action="append", default=[], metavar="PID=PATH")
    parser.add_argument("--attempt2-root", type=Path, default=Path("results/measurement_audit_v1") / ATTEMPT2_DIR_NAME)
    parser.add_argument("--fp16-capture", type=Path, default=Path("results/measurement_audit_v1/server_fp16_capture_v1"))
    parser.add_argument("--fp16-verification", type=Path, default=Path("results/measurement_audit_v1/server_native_size_v1"))
    parser.add_argument("--accuracy-root", type=Path, default=Path("results/measurement_audit_v1") / ACCURACY_DIR_NAME)
    parser.add_argument("--images-dir", type=Path, default=Path("../nighttime-tsd/data/processed/cctsdb2021_clean/dev/images"))
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--engine-key")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--engine-sha256")
    parser.add_argument("--round", type=int)
    parser.add_argument("--session-dir", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--pool-manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.child:
            required = (args.engine_key, args.engine, args.engine_sha256, args.round, args.session_dir, args.run_manifest, args.pool_manifest)
            if any(value is None for value in required):
                parser.error("child session arguments are incomplete")
            return run_child(args)
        return _run_parent(args)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
