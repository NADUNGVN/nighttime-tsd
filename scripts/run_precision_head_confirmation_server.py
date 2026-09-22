#!/usr/bin/env python3
"""Server-only executor for ST-SERVER-01.

The parent in this module is CPU orchestration. TensorRT, torch and
Ultralytics are imported only inside a short-lived child. ``--runtime-double``
is an external-runtime test adapter and is never accepted by the production
runbook.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from precision_head_confirmation_contract import ContractError, selected_targets, validate_arm_for_phase, validate_schedule


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def job_key(job: dict[str, Any]) -> str:
    selection = job["selection"] or "NA"
    return f"{job['sequence']:03d}_{job['model']}_{job['phase']}_{selection}_{job['arm']}_r{job['repeat'] or 0}"


def child_state(path: Path, job: dict[str, Any], **updates: Any) -> dict[str, Any]:
    state = {"schema_version": 1, "job": job, "status": "running", "stage": "started", "stage_history": [], "builder_attempted": False, "builder_completed": False, "capture_attempted": False, "capture_completed": False, "calibration_batches": 0, "calibration_read_calls": 0, "calibration_write_calls": 0, "engine_published": False, "raw_tensors_published": False, "owner_release_status": "not_attempted", "synchronization_completed": 0, **updates}
    atomic(path, state)
    return state


def update_state(path: Path, state: dict[str, Any], stage: str, **updates: Any) -> None:
    state["stage"] = stage
    state["stage_history"].append({"stage": stage, "utc": datetime.now(timezone.utc).isoformat()})
    state.update(updates)
    atomic(path, state)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _calibration_binding(plan: dict[str, Any], selection: str, repo: Path) -> dict[str, Any]:
    """Return the producer-verified materialization; never reconstruct it."""
    records = plan.get("calibration_producer", {}).get("selections", [])
    record = next((item for item in records if item.get("id") == selection), None)
    if not isinstance(record, dict) or record.get("status") != "verified":
        raise ContractError(f"calibration selection is not producer-verified: {selection}")
    materialization = record.get("materialization")
    contract = record.get("manifest_contract")
    if not isinstance(materialization, dict) or materialization.get("status") != "complete":
        raise ContractError(f"calibration materialization is not complete: {selection}")
    if not isinstance(contract, dict) or contract.get("status") != "canonical_manifest_valid":
        raise ContractError(f"calibration manifest contract is not canonical: {selection}")
    yaml_path = Path(materialization.get("yaml", ""))
    if not yaml_path.is_absolute():
        yaml_path = (repo / yaml_path).resolve()
    expected = [row.get("image") for row in materialization.get("image_bytes", [])]
    manifest_order = list(contract.get("image_ids", []))
    if expected != manifest_order or len(expected) != 1024 or len(set(expected)) != 1024:
        raise ContractError(f"calibration manifest/materialization order is not the locked 1024-image order: {selection}")
    if not yaml_path.is_file():
        raise FileNotFoundError(f"producer-verified calibration YAML is absent: {yaml_path}")
    image_names = [Path(item).name for item in expected]
    return {
        "id": selection,
        "manifest": record["manifest"],
        "manifest_sha256": record["canonical_sha256"],
        "yaml": yaml_path,
        "yaml_sha256": materialization.get("yaml_sha256"),
        "materialized_directory": materialization.get("resolved_directory"),
        "ordered_images": expected,
        "ordered_images_sha256": sha256_bytes(json.dumps(image_names, separators=(",", ":"), ensure_ascii=False).encode("utf-8")),
        "source_bytes": materialization.get("image_bytes", []),
    }


def verify_plan_inputs(repo: Path, plan: dict[str, Any]) -> None:
    """Recheck plan-bound bytes immediately before any production child."""
    for model, evidence in plan.get("models", {}).items():
        for key in ("checkpoint", "onnx"):
            item = evidence.get(key, {})
            path = repo / item.get("path", "")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
                raise ContractError(f"bound {model} {key} changed or is missing: {path}")
    for item in plan.get("provenance_files", []):
        path = repo / item.get("path", "")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
            raise ContractError(f"bound helper/config changed or is missing: {path}")
    dataset = plan.get("dataset", {})
    xml_validation = dataset.get("xml_validation")
    if xml_validation:
        xml_path = Path(xml_validation.get("path", ""))
        if not xml_path.is_file() or hashlib.sha256(xml_path.read_bytes()).hexdigest() != xml_validation.get("sha256"):
            raise ContractError(f"bound XML archive changed or is missing: {xml_path}")
    for selection in plan.get("calibration_producer", {}).get("selections", []):
        materialization = selection.get("materialization", {})
        yaml_path = Path(materialization.get("yaml", ""))
        if not yaml_path.is_file() or hashlib.sha256(yaml_path.read_bytes()).hexdigest() != materialization.get("yaml_sha256"):
            raise ContractError(f"bound calibration YAML changed or is missing: {yaml_path}")
        materialized_dir = Path(materialization.get("resolved_directory", ""))
        materialized_root = materialized_dir.resolve()
        image_dir = (materialized_root / "images").resolve()
        try:
            image_dir.relative_to(materialized_root)
        except ValueError as exc:
            raise ContractError(f"calibration image directory escapes producer root: {image_dir}") from exc
        for item in materialization.get("image_bytes", []):
            source = repo / "data/processed/cctsdb2021_clean" / item["image"]
            materialized = image_dir / Path(item["image"]).name
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
            materialized_hash = hashlib.sha256(materialized.read_bytes()).hexdigest() if materialized.is_file() else None
            if source_hash != item.get("source_sha256") or materialized_hash != item.get("materialized_sha256") or source_hash != materialized_hash:
                raise ContractError(f"bound calibration image changed or is missing: {item.get('image')}")


def validate_canonical_capture_records(records: list[dict[str, Any]], plan: dict[str, Any]) -> None:
    reference = plan.get("dataset", {}).get("canonical_dev_reference", {})
    expected = reference.get("records", [])
    if len(expected) != 1636:
        raise ContractError("plan is missing the canonical 1,636-image dev reference")
    def project(row: dict[str, Any]) -> dict[str, Any]:
        image = row.get("image")
        stem = row.get("stem", Path(image).stem if image else None)
        if stem != (Path(image).stem if image else None):
            raise ContractError(f"canonical dev stem does not agree with image: {image}")
        return {"image": image, "stem": stem, "orig_shape": row.get("orig_shape")}

    observed = [project(row) for row in records]
    canonical = [project(row) for row in expected]
    if observed != canonical:
        raise ContractError("capture image order or original-shape binding differs from canonical dev reference")


def _gpu_identity(device: str) -> dict[str, str]:
    index = str(device)
    result = subprocess.run(
        ["nvidia-smi", "-i", index, "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ContractError(f"unable to verify GPU identity for device {device}: {result.stderr.strip()}")
    fields = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
    if len(fields) != 3 or any(not item for item in fields):
        raise ContractError(f"invalid GPU identity response for device {device}: {result.stdout!r}")
    return {"uuid": fields[0], "name": fields[1], "driver_version": fields[2], "device": index}


def _release_state(path: Path, state: dict[str, Any], *, status: str, error: str | None = None) -> None:
    updates: dict[str, Any] = {"cleanup_started": True, "cleanup_completed": status == "released", "owner_release_status": status}
    if error:
        updates["cleanup_error"] = error
    state.update(updates)
    atomic(path, state)


def _auxiliary_cache_sha(out: Path, model: str, selection: str) -> str:
    manifest_path = out / "execution_manifest.json"
    if not manifest_path.is_file():
        raise ContractError("execution manifest is missing before scored cache reuse")
    manifest = read(manifest_path)
    matches = [item for item in manifest.get("jobs", []) if item.get("job", {}).get("model") == model and item.get("job", {}).get("selection") == selection and item.get("job", {}).get("phase") == "auxiliary_calibration"]
    if len(matches) != 1:
        raise ContractError(f"expected exactly one auxiliary cache producer for {model}/{selection}")
    state = matches[0].get("state", {})
    cache = state.get("calibration_cache", {})
    value = cache.get("after_sha256")
    if not value or state.get("calibration_batches") != 1024 or state.get("calibration_write_calls") != 1:
        raise ContractError(f"auxiliary cache provenance is incomplete for {model}/{selection}")
    return value


def _validate_completed_child(job: dict[str, Any], state: dict[str, Any]) -> None:
    if state.get("status") != "completed" or state.get("job") != job:
        raise ContractError("child completion state is missing or has the wrong job identity")
    if not state.get("builder_attempted") or not state.get("builder_completed"):
        raise ContractError("child builder lifecycle counters are incomplete")
    if job.get("capture_required"):
        if not state.get("capture_attempted") or not state.get("capture_completed"):
            raise ContractError("child capture lifecycle counters are incomplete")
        if state.get("forward_attempted") != 1636 or state.get("forward_completed") != 1636 or state.get("synchronization_completed") != 1636:
            raise ContractError("child forward/synchronization accounting is not exactly 1636")
        if state.get("warmup_forward_count") != 0:
            raise ContractError("child reported an uncounted warmup forward")
    if state.get("owner_release_status") != "released_after_child_return" or not state.get("cleanup_completed"):
        raise ContractError("child owner lifecycle was not released after cleanup")


def bind_gpu_identity(manifest: dict[str, Any], identity: dict[str, str]) -> None:
    """Bind the first production GPU identity and compare every later child."""
    if not identity or not identity.get("uuid") or not identity.get("name") or not identity.get("driver_version"):
        raise ContractError("production child did not publish a complete GPU identity")
    previous = manifest.get("gpu_identity")
    if previous is not None and previous != identity:
        raise ContractError("GPU identity changed within one confirmation study")
    manifest["gpu_identity"] = dict(identity)


class NoWarmupBackendAccounting:
    """Mixin for the actual validator backend wrapper.

    A framework warmup request is observable, but this study does not turn it
    into a forward. Data forwards and their completion/synchronization are
    counted separately by the concrete child wrapper below.
    """

    warmup_request_count = 0
    warmup_forward_count = 0
    forward_attempt_count = 0
    forward_completed_count = 0
    synchronization_completed_count = 0
    last_instance = None

    @classmethod
    def reset_accounting(cls) -> None:
        cls.warmup_request_count = 0
        cls.warmup_forward_count = 0
        cls.forward_attempt_count = 0
        cls.forward_completed_count = 0
        cls.synchronization_completed_count = 0
        cls.last_instance = None

    def warmup(self, *warmup_args: Any, **warmup_kwargs: Any) -> None:
        type(self).warmup_request_count += 1
        # Deliberately do not call the framework implementation: that would
        # issue an uncounted engine forward.

    def __call__(self, *call_args: Any, **call_kwargs: Any) -> Any:
        type(self).forward_attempt_count += 1
        result = super().__call__(*call_args, **call_kwargs)
        type(self).forward_completed_count += 1
        return result


def runtime_double(job: dict[str, Any], plan: dict[str, Any], out: Path, state_path: Path) -> None:
    """External-boundary fixture; production never enables this switch."""
    state = read(state_path) if state_path.is_file() else child_state(state_path, job)
    validate_arm_for_phase(job["phase"], job["arm"])
    model_spec = plan.get("models", {}).get(job["model"], {})
    engine_sha = hashlib.sha256(job_key(job).encode()).hexdigest()
    cache_sha = hashlib.sha256(f"cache:{job['model']}:{job['selection']}".encode()).hexdigest()
    timing_in = hashlib.sha256(b"").hexdigest()
    timing_out = hashlib.sha256(f"timing:{job_key(job)}".encode()).hexdigest()
    study_root = state_path.parents[2]
    public_inspector_path = study_root / "public" / "inspectors" / f"{job_key(job)}.json"
    public_inspector = {"schema_version": 1, "source": "external_runtime_boundary_double", "job_id": job_key(job), "requested_precision": "fixture-only", "effective_precision": "unknown"}
    atomic(public_inspector_path, public_inspector)
    public_inspector_sha256 = hashlib.sha256(public_inspector_path.read_bytes()).hexdigest()
    update_state(state_path, state, "external_plan_verified", boundary_fixture=True, producer_bindings_checked=True)
    state = read(state_path)
    state.update({"status": "ready_to_release", "builder_attempted": True, "builder_completed": True, "capture_attempted": bool(job["capture_required"]), "capture_completed": bool(job["capture_required"]), "calibration_batches": 1024 if job["phase"] == "auxiliary_calibration" else 0, "calibration_read_calls": 2 if job["phase"] == "scored_int8" else 0, "calibration_write_calls": 1 if job["phase"] == "auxiliary_calibration" else 0, "synchronization_completed": 1636 if job["capture_required"] else 0, "forward_attempted": 1636 if job["capture_required"] else 0, "forward_completed": 1636 if job["capture_required"] else 0, "warmup_request_count": 0, "warmup_forward_count": 0, "cache_consumed": job["phase"] == "scored_int8", "engine_sha256": engine_sha, "engine_inspector": {"path": str(public_inspector_path.relative_to(study_root).as_posix()), "sha256": public_inspector_sha256, "format": "JSON", "source": "external_runtime_boundary_double", "publication": "public_allowlist"}, "calibration_cache": {"selection": job.get("selection"), "before_sha256": cache_sha if job["phase"] == "scored_int8" else None, "after_sha256": cache_sha if job["phase"] in {"auxiliary_calibration", "scored_int8"} else None, "auxiliary_cache_sha256": cache_sha if job["phase"] == "scored_int8" else None, "immutable": job["phase"] == "scored_int8", "ordered_images_sha256": "runtime-double-ordered-images"}, "timing_cache": {"input_sha256": timing_in, "output_sha256": timing_out, "reused": False}, "precision_inspector": {"requested_targets": selected_targets(model_spec.get("mapping", {}), job["arm"]) if model_spec.get("mapping") else [], "requested_only": True}, "cleanup_started": False, "cleanup_completed": False, "owner_release_status": "pending_child_return", "boundary_fixture": True})
    test_identity = plan.get("_test_gpu_identity_by_sequence", {}).get(str(job.get("sequence")))
    if test_identity:
        state["gpu_identity"] = test_identity
    atomic(state_path, state)
    update_state(state_path, read(state_path), "external_build_complete", parser_completed=True, calibration_callbacks_observed=job["phase"] != "scored_fp16")
    if job["capture_required"]:
        canonical = plan.get("dataset", {}).get("canonical_dev_reference", {}).get("records", [])
        records = [{"image": row.get("image", f"{index:04d}.jpg"), "stem": row.get("stem", Path(row.get("image", f"{index:04d}.jpg")).stem), "orig_shape": row.get("orig_shape", [100, 100]), "xyxy": [], "confidence": [], "class_id": [], "validator_input": {"imgsz": [640, 640], "ratio_pad": [[1.0, 1.0], [0.0, 0.0]], "prediction_xyxy": [], "target_xyxy": [], "target_class_id": []}, "validator_statistics": {"tp": [], "confidence_dtype": "float64", "pred_class_dtype": "float64", "target_class_dtype": "int64"}} for index, row in enumerate(canonical if len(canonical) == 1636 else [{} for _ in range(1636)])]
        if len(canonical) == 1636:
            validate_canonical_capture_records(records, plan)
        first = records[0]
        records[0] = {**first, "xyxy": [[10.0, 10.0, 20.0, 20.0]], "confidence": [0.1], "class_id": [0], "validator_input": {"imgsz": [640, 640], "ratio_pad": [[1.0, 1.0], [0.0, 0.0]], "prediction_xyxy": [[10.0, 10.0, 20.0, 20.0]], "target_xyxy": [], "target_class_id": []}, "validator_statistics": {"tp": [[False] * 10], "confidence_dtype": "float32", "pred_class_dtype": "int64", "target_class_dtype": "int64"}}
        prediction = {"schema_version": 2, "capture_mode": "same_val_process_batch", "iou_thresholds": [0.5 + 0.05 * index for index in range(10)], "records": records, "model_sha256": state["engine_sha256"]}
        prediction_path = out / "predictions.json"
        atomic(prediction_path, prediction)
        prediction_sha256 = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        atomic(out / "cell_metrics.json", {"schema_version": 1, "job": job, "job_id": job_key(job), "status": "synthetic_external_runtime_double", "plan_sha256": plan.get("_plan_sha256", "runtime-double-plan-bound-at-test"), "checkpoint_sha256": model_spec.get("checkpoint", {}).get("sha256", "runtime-double-checkpoint"), "onnx_sha256": model_spec.get("onnx", {}).get("sha256", "runtime-double-onnx"), "engine_sha256": state["engine_sha256"], "metrics": {"full": {"ap50": 0.0, "ap50_95": 0.0}, "XS": {"ap50": 0.0, "ap50_95": 0.0}, "S": {"ap50": 0.0, "ap50_95": 0.0}}, "statistics_replay": {"status": "pass", "images": 1636}, "native_matching": {"status": "pass", "changed_tp_decisions": 0}, "xml_validation": {"path": "external-test-xml", "sha256": "external-test-xml", "images": 1636, "instances": 2706}, "prediction_path": "predictions.json", "prediction_sha256": prediction_sha256, "record_count": 1636, "raw_output_source": "external_runtime_double_capture_adapter", "postprocess_route": model_spec.get("postprocess", "double")})
        update_state(state_path, read(state_path), "external_capture_complete", capture_records_published=1636, synchronization_observed=1636)


def finalize_child_state(path: Path) -> None:
    """Mark release only after the production child function has returned."""
    state = read(path)
    if state.get("status") not in {"ready_to_release", "completed"}:
        return
    state.update({"status": "completed", "cleanup_started": True, "cleanup_completed": True, "owner_release_status": "released_after_child_return", "release_boundary": "run_child_after_build_real_return"})
    atomic(path, state)


def build_real(job: dict[str, Any], plan: dict[str, Any], repo: Path, out: Path, args: argparse.Namespace, state_path: Path, *, external_boundary: bool = False) -> None:
    """Build one real server cell and capture it with the locked validator."""
    validate_arm_for_phase(job["phase"], job["arm"])
    verify_plan_inputs(repo, plan)
    if external_boundary:
        # Tests enter through the same child production boundary.  The
        # external fixture replaces only TensorRT/device work; it is never
        # accepted by the production runbook or production analyzer.
        plan.setdefault("_plan_sha256", hashlib.sha256((out / "confirmation_plan.json").read_bytes()).hexdigest() if (out / "confirmation_plan.json").is_file() else "external-boundary-plan")
        update_state(state_path, read(state_path), "build_real_external_boundary", build_real_entry=True, boundary_fixture=True)
        runtime_double(job, plan, out / "jobs" / job_key(job), state_path)
        state = read(state_path)
        state["build_real_entry"] = True
        atomic(state_path, state)
        return
    import inspect
    import numpy as np
    import torch
    import tensorrt as trt
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.engine.exporter import Exporter

    from capture_cctsdb_validator import CaptureValidator, metric_summary
    from run_architecture_matrix import GpuPhaseLock
    from uniform_build_repeat import ensure_idle, parse_background_confirmations, parse_desktop_confirmations, snapshot

    model = job["model"]
    validate_arm_for_phase(job["phase"], job["arm"])
    verify_plan_inputs(repo, plan)
    model_evidence = plan["models"][model]
    checkpoint = repo / model_evidence["checkpoint"]["path"]
    onnx = repo / model_evidence["onnx"]["path"]
    mapping = model_evidence["mapping"]["targets"]
    gpu_identity = _gpu_identity(args.device)
    private = out / "private" / model / (job["selection"] or "fp16") / job_key(job)
    private.mkdir(parents=True, exist_ok=False)
    engine_path = private / "model.engine"
    cache_path = out / "private" / "calibration" / model / f"{job['selection']}.cache"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_binding = _calibration_binding(plan, job["selection"], repo) if job["phase"] in {"auxiliary_calibration", "scored_int8"} else None
    auxiliary_cache_sha = _auxiliary_cache_sha(out, model, job["selection"]) if job["phase"] == "scored_int8" else None
    desktop = parse_desktop_confirmations(args.confirm_desktop_process)
    if args.confirm_background_process:
        raise ContractError("confirmation build-variation study does not authorize competing background compute")
    background = parse_background_confirmations([])
    logger = builder = network = parser = config = timing = calibrator = exporter = loader = validator = None
    with GpuPhaseLock(repo / "results/architecture_matrix_v1/.gpu_phase.lock", f"confirmation_{job_key(job)}"):
        before = snapshot(desktop, background)
        ensure_idle(before, background)
        update_state(state_path, read(state_path), "telemetry_before_build", telemetry={"before_build": before})
        runtime_contract = plan.get("runtime", {})
        observed_runtime = {"torch": torch.__version__, "ultralytics": ultralytics.__version__, "tensorrt": trt.__version__, "cuda": torch.version.cuda, "gpu": gpu_identity["name"], "gpu_identity": gpu_identity}
        for key in ("torch", "ultralytics", "tensorrt", "cuda"):
            if runtime_contract.get(key) and observed_runtime[key] != runtime_contract[key]:
                raise ContractError(f"runtime {key} differs: expected={runtime_contract[key]!r} observed={observed_runtime[key]!r}")
        update_state(state_path, read(state_path), "runtime_loaded", runtime=observed_runtime, gpu_identity=gpu_identity, input_identity={"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest()})
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError("TensorRT ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        if hashlib.sha256(onnx.read_bytes()).hexdigest() != model_evidence["onnx"]["sha256"] or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != model_evidence["checkpoint"]["sha256"]:
            raise ContractError("bound checkpoint/ONNX bytes changed during parse")
        expected_shape = tuple(model_evidence["output_shape"])
        if network.num_inputs != 1 or tuple(network.get_input(0).shape) != (1, 3, 640, 640):
            raise RuntimeError(f"unexpected TensorRT input contract: {network.get_input(0).shape}")
        if network.num_outputs != 1 or tuple(network.get_output(0).shape) != expected_shape:
            raise RuntimeError(f"unexpected TensorRT output contract: {network.get_output(0).shape}, expected {expected_shape}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 ** 3)
        config.builder_optimization_level = 3
        config.avg_timing_iterations = 1
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        timing = config.create_timing_cache(b"")
        if not config.set_timing_cache(timing, False):
            raise RuntimeError("fresh timing cache was not accepted")
        config.clear_flag(trt.BuilderFlag.TF32)
        if job["phase"] == "scored_fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            config.clear_flag(trt.BuilderFlag.INT8)
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            config.clear_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            if job["phase"] == "scored_int8" and not cache_path.is_file():
                raise FileNotFoundError(f"auxiliary cache missing: {cache_path}")

            loader = None
            if job["phase"] == "auxiliary_calibration":
                from types import SimpleNamespace
                calibration_yaml = calibration_binding["yaml"]
                exporter = Exporter(overrides={"format": "engine", "data": str(calibration_yaml), "imgsz": 640, "batch": 1, "fraction": 1.0, "split": "val", "rect": False, "device": str(args.device)})
                exporter.imgsz = (640, 640)
                exporter.model = SimpleNamespace(task="detect")
                loader = iter(exporter.get_int8_calibration_dataloader())

            class CacheOnly(trt.IInt8Calibrator):
                def __init__(self) -> None:
                    super().__init__(); self.read_calls = 0; self.write_calls = 0; self.batch_calls = 0; self.eos_calls = 0; self.cache_consumed = False; self.tensor = None; self.observed_images = []; self.tensor_hash = hashlib.sha256()
                def get_batch_size(self) -> int: return 1
                def get_algorithm(self) -> Any: return trt.CalibrationAlgoType.MINMAX_CALIBRATION
                def read_calibration_cache(self) -> bytes:
                    self.read_calls += 1
                    if job["phase"] == "scored_int8":
                        self.cache_consumed = True
                        if not cache_path.is_file():
                            raise FileNotFoundError(f"scored cache missing: {cache_path}")
                        return cache_path.read_bytes()
                    return None
                def write_calibration_cache(self, data: Any) -> None:
                    self.write_calls += 1
                    if job["phase"] == "scored_int8": raise RuntimeError("scored cache write")
                    cache_path.write_bytes(bytes(data))
                def get_batch(self, names: Any) -> Any:
                    if job["phase"] == "scored_int8": raise RuntimeError("scored calibration batch")
                    try:
                        batch = next(loader)
                    except StopIteration:
                        self.eos_calls += 1
                        return None
                    self.batch_calls += 1
                    image_names = batch.get("im_file") or batch.get("im_files") or []
                    if isinstance(image_names, (str, Path)):
                        image_names = [str(image_names)]
                    if len(image_names) != 1:
                        raise ContractError(f"calibration batch is not batch=1: {image_names!r}")
                    observed_name = Path(str(image_names[0])).name
                    expected_name = Path(calibration_binding["ordered_images"][self.batch_calls - 1]).name
                    if observed_name != expected_name:
                        raise ContractError(f"calibration order mismatch at batch {self.batch_calls}: {observed_name} != {expected_name}")
                    image = batch["img"].to(device=torch.device(f"cuda:{args.device}"), dtype=torch.float32) / 255.0
                    self.tensor = image.contiguous()
                    self.observed_images.append(observed_name)
                    self.tensor_hash.update(self.tensor.detach().cpu().numpy().tobytes())
                    return [int(self.tensor.data_ptr())]
            calibrator = CacheOnly()
            config.int8_calibrator = calibrator
        cache_before_sha = hashlib.sha256(cache_path.read_bytes()).hexdigest() if job["phase"] == "scored_int8" and cache_path.is_file() else None
        if job["phase"] == "scored_int8" and cache_before_sha != auxiliary_cache_sha:
            raise ContractError("scored cache bytes do not match the auxiliary producer cache")
        targets = set(selected_targets(mapping, job["arm"]))
        observed_constraints = []
        effective_precision = {}
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            if layer.name in targets:
                if layer.type != trt.LayerType.CONVOLUTION:
                    raise RuntimeError(f"precision target is not convolution: {layer.name}")
                layer.precision = trt.float32
                for output_index in range(layer.num_outputs): layer.set_output_type(output_index, trt.float32)
                observed_constraints.append(layer.name)
                effective_precision[layer.name] = {"precision": str(layer.precision), "outputs": [str(layer.get_output_type(output_index)) for output_index in range(layer.num_outputs)]}
        if observed_constraints != sorted(targets) and set(observed_constraints) != targets:
            raise RuntimeError(f"precision target mismatch: expected {sorted(targets)}, observed {observed_constraints}")
        update_state(state_path, read(state_path), "build", builder_attempted=True, precision_constraints=observed_constraints, precision_inspector={"requested_targets": sorted(targets), "requested_layer_settings": effective_precision, "requested_only": True, "effective_precision": "unknown", "effective_engine_precision": "not_inferred_from_layer_flags"})
        serialized = builder.build_serialized_network(network, config)
        if serialized is None: raise RuntimeError("TensorRT builder returned no engine")
        if job["phase"] == "auxiliary_calibration":
            if calibrator.batch_calls != 1024 or calibrator.write_calls != 1:
                raise RuntimeError(f"auxiliary calibration audit mismatch: batches={calibrator.batch_calls}, writes={calibrator.write_calls}")
        elif job["phase"] == "scored_int8":
            if calibrator.read_calls < 1 or not calibrator.cache_consumed or calibrator.batch_calls != 0 or calibrator.write_calls != 0:
                raise RuntimeError("scored INT8 cache-only audit mismatch")
        engine_path.write_bytes(bytes(serialized))
        timing_path = private / "timing.cache"
        timing_path.write_bytes(bytes(timing.serialize()))
        timing_audit = {"input_sha256": hashlib.sha256(b"").hexdigest(), "output_sha256": hashlib.sha256(timing_path.read_bytes()).hexdigest(), "reused": False, "bytes": timing_path.stat().st_size}
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(bytes(serialized))
        if engine is None:
            raise RuntimeError("TensorRT runtime could not deserialize the just-built engine")
        inspector = engine.create_engine_inspector()
        inspector_path = private / "inspector.json"
        inspector_path.write_text(
            inspector.get_engine_information(trt.LayerInformationFormat.JSON), encoding="utf-8"
        )
        inspector_sha256 = hashlib.sha256(inspector_path.read_bytes()).hexdigest()
        public_inspector_path = out / "public" / "inspectors" / f"{job_key(job)}.json"
        public_inspector_path.parent.mkdir(parents=True, exist_ok=True)
        public_inspector_path.write_bytes(inspector_path.read_bytes())
        public_inspector_sha256 = hashlib.sha256(public_inspector_path.read_bytes()).hexdigest()
        del inspector, engine, runtime
        cache_after = hashlib.sha256(cache_path.read_bytes()).hexdigest() if cache_path.is_file() else None
        if job["phase"] == "scored_int8" and cache_before_sha != cache_after:
            raise ContractError("scored calibration cache changed during build")
        calibration_audit = None
        if calibration_binding is not None:
            calibration_audit = {"selection": calibration_binding["id"], "manifest_sha256": calibration_binding["manifest_sha256"], "yaml_sha256": calibration_binding["yaml_sha256"], "expected_images_sha256": calibration_binding["ordered_images_sha256"], "observed_images_sha256": sha256_bytes(json.dumps(getattr(calibrator, "observed_images", []), separators=(",", ":"), ensure_ascii=False).encode("utf-8")), "tensor_sequence_sha256": getattr(getattr(calibrator, "tensor_hash", None), "hexdigest", lambda: None)(), "batch_count": getattr(calibrator, "batch_calls", 0), "input_shape": [1, 3, 640, 640], "input_dtype": "torch.float32"}
            if job["phase"] == "auxiliary_calibration" and calibration_audit["observed_images_sha256"] != calibration_binding["ordered_images_sha256"]:
                raise ContractError("calibration loader order audit hash mismatch")
        update_state(state_path, read(state_path), "build_complete", builder_completed=True, engine_sha256=hashlib.sha256(engine_path.read_bytes()).hexdigest(), engine_inspector={"path": str(public_inspector_path.relative_to(out).as_posix()), "sha256": public_inspector_sha256, "private_source_sha256": inspector_sha256, "format": "JSON", "source": "TensorRT EngineInspector", "publication": "public_allowlist"}, calibration_batches=calibrator.batch_calls if job["phase"] != "scored_fp16" else 0, calibration_read_calls=calibrator.read_calls if job["phase"] != "scored_fp16" else 0, calibration_write_calls=calibrator.write_calls if job["phase"] != "scored_fp16" else 0, cache_consumed=getattr(calibrator, "cache_consumed", False) if job["phase"] != "scored_fp16" else False, calibration_cache={"selection": job.get("selection"), "before_sha256": cache_before_sha, "after_sha256": cache_after, "auxiliary_cache_sha256": auxiliary_cache_sha, "immutable": job["phase"] == "scored_int8", "ordered_images_sha256": calibration_audit["expected_images_sha256"] if calibration_audit else None, "audit": calibration_audit}, timing_cache=timing_audit)
        after_build = snapshot(desktop, background)
        ensure_idle(after_build, background)
        state_after_build = read(state_path)
        state_after_build.setdefault("telemetry", {}).update({"after_build": after_build})
        atomic(state_path, state_after_build)
        if job["capture_required"]:
            update_state(state_path, read(state_path), "capture", capture_attempted=True)
            data = private / "dev.yaml"
            dev = repo / "data/processed/cctsdb2021_clean/dev"
            data.write_text(f"path: {dev.as_posix()}\ntrain: images\nval: images\nnames:\n  0: prohibitory\n  1: mandatory\n  2: warning\nnc: 3\n", encoding="utf-8")
            from ultralytics.engine import validator as validator_module
            backend_base = validator_module.AutoBackend
            class CountingNoWarmupBackend(NoWarmupBackendAccounting, backend_base):
                def __init__(self, *backend_args: Any, **backend_kwargs: Any) -> None:
                    super().__init__(*backend_args, **backend_kwargs)
                    type(self).last_instance = self
                def __call__(self, *call_args: Any, **call_kwargs: Any) -> Any:
                    type(self).forward_attempt_count += 1
                    result = super(NoWarmupBackendAccounting, self).__call__(*call_args, **call_kwargs)
                    type(self).forward_completed_count += 1
                    torch.cuda.current_stream(torch.device(f"cuda:{args.device}")).synchronize()
                    type(self).synchronization_completed_count += 1
                    return result
            CountingNoWarmupBackend.reset_accounting()
            validator_module.AutoBackend = CountingNoWarmupBackend
            try:
                validator = CaptureValidator(args={"model": str(engine_path), "data": str(data), "split": "val", "device": args.device, "imgsz": 640, "batch": 1, "workers": 0, "conf": 0.001, "iou": 0.7, "max_det": 300, "rect": False, "plots": False, "verbose": False}, save_dir=private / "validator")
                try:
                    validator(model=str(engine_path))
                except Exception:
                    update_state(state_path, read(state_path), "capture_failed", forward_attempted=CountingNoWarmupBackend.forward_attempt_count, forward_completed=CountingNoWarmupBackend.forward_completed_count, synchronization_completed=CountingNoWarmupBackend.synchronization_completed_count, warmup_request_count=CountingNoWarmupBackend.warmup_request_count, warmup_forward_count=CountingNoWarmupBackend.warmup_forward_count, cleanup_started=False, cleanup_completed=False, owner_release_status="pending_child_return")
                    raise
            finally:
                validator_module.AutoBackend = backend_base
            records = validator.capture_records
            if len(records) != 1636 or len({record["image"] for record in records}) != 1636: raise RuntimeError("capture did not cover dev exactly once")
            validate_canonical_capture_records(records, plan)
            if CountingNoWarmupBackend.forward_attempt_count != 1636 or CountingNoWarmupBackend.forward_completed_count != 1636 or CountingNoWarmupBackend.synchronization_completed_count != 1636 or CountingNoWarmupBackend.warmup_forward_count != 0:
                raise RuntimeError(f"capture accounting mismatch: attempted={CountingNoWarmupBackend.forward_attempt_count}, completed={CountingNoWarmupBackend.forward_completed_count}, synchronized={CountingNoWarmupBackend.synchronization_completed_count}, warmup_forwards={CountingNoWarmupBackend.warmup_forward_count}")
            predictions = private / "validator_predictions.json"
            predictions.write_text(json.dumps({"schema_version": 2, "capture_mode": "same_val_process_batch", "iou_thresholds": [0.5 + 0.05 * index for index in range(10)], "records": records, "model_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest()}, allow_nan=False) + "\n", encoding="utf-8")
            from capture_cctsdb_validator import replay_statistics
            from verify_cctsdb_capture import rematch_native, validate_capture
            from audit_cctsdb_measurement import load_xml
            payload = json.loads(predictions.read_text(encoding="utf-8"))
            checked_records, thresholds = validate_capture(payload)
            if len(checked_records) != 1636 or sum(len(row["validator_input"]["target_class_id"]) for row in checked_records) != 2706:
                raise ContractError("capture membership/instance contract is not exactly 1636/2706")
            replay = replay_statistics(checked_records)
            matching = rematch_native(checked_records, thresholds)
            if matching["status"] != "pass":
                raise ContractError(f"native matching replay changed {matching['changed_tp_decisions']} TP decisions")
            xml_path = repo / plan.get("dataset", {}).get("xml_path", "data/raw/CCTSDB2021/xml.zip")
            xml = load_xml(xml_path, {row["image"]: row for row in checked_records})
            if set(xml) != {row["image"] for row in checked_records} or sum(len(value["rows"]) for value in xml.values()) != 2706:
                raise ContractError("XML/capture membership or 2706-instance contract failed")
            public_predictions = state_path.parent / "predictions.json"
            public_predictions.write_bytes(predictions.read_bytes())
            metrics = metric_summary(validator.metrics)
            replay_delta = {key: float(replay[key] - metrics[key]) for key in ("map50", "map50_95", "precision", "recall")}
            if any(abs(value) > 1e-12 for value in replay_delta.values()):
                raise ContractError(f"captured statistics replay differs from validator metrics: {replay_delta}")
            atomic(state_path.parent / "cell_metrics.json", {"schema_version": 1, "job": job, "job_id": job_key(job), "plan_sha256": hashlib.sha256((out / "confirmation_plan.json").read_bytes()).hexdigest(), "checkpoint_sha256": model_evidence["checkpoint"]["sha256"], "onnx_sha256": model_evidence["onnx"]["sha256"], "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(), "metrics": {"full": {"ap50": metrics["map50"], "ap50_95": metrics["map50_95"]}}, "statistics_replay": replay, "statistics_replay_delta": replay_delta, "statistics_replay_match": True, "native_matching": matching, "xml_validation": {"path": plan.get("dataset", {}).get("xml_path"), "sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(), "images": len(xml), "instances": sum(len(value["rows"]) for value in xml.values())}, "prediction_path": str(public_predictions.relative_to(out).as_posix()), "prediction_sha256": hashlib.sha256(public_predictions.read_bytes()).hexdigest(), "record_count": len(records), "raw_output_source": "CaptureValidator._process_batch", "postprocess_route": plan["models"][model]["postprocess"], "backend_metadata": {"format": str(getattr(CountingNoWarmupBackend.last_instance, "format", "unknown")), "stride": int(getattr(CountingNoWarmupBackend.last_instance, "stride", 0)), "end2end": bool(getattr(CountingNoWarmupBackend.last_instance, "end2end", False))}})
            update_state(state_path, read(state_path), "capture_complete", capture_completed=True, synchronization_completed=CountingNoWarmupBackend.synchronization_completed_count, forward_attempted=CountingNoWarmupBackend.forward_attempt_count, forward_completed=CountingNoWarmupBackend.forward_completed_count, warmup_request_count=CountingNoWarmupBackend.warmup_request_count, warmup_forward_count=CountingNoWarmupBackend.warmup_forward_count, backend_metadata={"format": str(getattr(CountingNoWarmupBackend.last_instance, "format", "unknown")), "stride": int(getattr(CountingNoWarmupBackend.last_instance, "stride", 0)), "end2end": bool(getattr(CountingNoWarmupBackend.last_instance, "end2end", False))})
            after_capture = snapshot(desktop, background)
            ensure_idle(after_capture, background)
            state_after_capture = read(state_path)
            state_after_capture.setdefault("telemetry", {}).update({"after_capture": after_capture})
            atomic(state_path, state_after_capture)
            CountingNoWarmupBackend.last_instance = None
            del validator, records, checked_records, xml, matching, replay
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device=torch.device(f"cuda:{args.device}"))
        after = snapshot(desktop, background)
        ensure_idle(after, background)
        logger = builder = network = parser = config = timing = calibrator = exporter = loader = validator = None
        gc.collect()
        state = read(state_path); state.update({"status": "ready_to_release", "telemetry": {**state.get("telemetry", {}), "final": after}, "cleanup_started": True, "cleanup_completed": False, "owner_release_status": "pending_child_return", "release_boundary": "after_local_owner_clear_before_child_return"}); atomic(state_path, state)


def run_child(args: argparse.Namespace) -> int:
    job = read(Path(args.job_json))
    out = Path(args.out_dir).resolve()
    plan_path = Path(args.plan).resolve()
    plan = read(plan_path)
    jobs = plan.get("schedule", {}).get("jobs", [])
    validate_schedule(jobs)
    if job not in jobs:
        raise ContractError("child job is not an exact member of the canonical plan schedule")
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    state_path = out / "jobs" / f"{job_key(job)}" / "child_state.json"
    job_out = state_path.parent
    job_out.mkdir(parents=True, exist_ok=False)
    child_state(state_path, job, plan_sha256=plan_sha256)
    try:
        plan["_plan_sha256"] = plan_sha256
        build_real(job, plan, Path(args.repo).resolve(), out, args, state_path, external_boundary=args.runtime_double)
        finalize_child_state(state_path)
    except Exception as exc:
        state = read(state_path) if state_path.exists() else {"schema_version": 1, "job": job, "status": "unknown", "stage": "unknown", "stage_history": [], "unknown_completion": True}
        state.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "no_retry": True, "cleanup_started": True, "cleanup_completed": False, "owner_release_status": "cleanup_failed_or_unknown"})
        atomic(state_path, state)
        traceback.clear_frames(exc.__traceback__)
        gc.collect()
        return 1
    return 0


def run_parent(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve(); out = Path(args.out_dir).resolve(); plan_path = Path(args.plan).resolve(); plan = read(plan_path)
    if args.confirm_background_process:
        raise ContractError("confirmation build-variation study does not authorize competing background compute")
    jobs = plan["schedule"]["jobs"]; validate_schedule(jobs)
    expected_schedule_sha256 = __import__("precision_head_confirmation_contract").canonical_json_sha256(jobs)
    if plan["schedule"].get("sha256") != expected_schedule_sha256:
        raise ContractError("plan schedule hash does not match its serialized jobs")
    if set(plan.get("models", {})) != {"yolov8n", "yolo26n"}:
        raise ContractError("plan model evidence must contain exactly both bound models")
    verify_plan_inputs(repo, plan)
    if out.exists():
        existing = {path.name for path in out.iterdir()}
        if existing - {"confirmation_plan.json", "schedule.json"}:
            raise FileExistsError(f"output exists or is partial; no resume/overwrite: {out}")
    else:
        out.mkdir(parents=True)
    atomic(out / "execution_manifest.json", {"schema_version": 1, "study": "precision_head_confirmation_v1", "status": "running", "plan_sha256": __import__("hashlib").sha256(plan_path.read_bytes()).hexdigest(), "expected_jobs": 84, "expected_captures": 78, "started_utc": datetime.now(timezone.utc).isoformat(), "jobs": []})
    for job in jobs:
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--job-json", "-", "--plan", str(plan_path), "--out-dir", str(out), "--repo", str(repo), "--device", args.device]
        if args.runtime_double: command.append("--runtime-double")
        command += sum((["--confirm-desktop-process", value] for value in args.confirm_desktop_process), [])
        command += sum((["--confirm-background-process", value] for value in args.confirm_background_process), [])
        payload = json.dumps(job)
        job_dir = out / "jobs" / job_key(job)
        try:
            result = subprocess.run(command, input=payload, text=True, capture_output=True, timeout=args.child_timeout)
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "child.log").write_text(result.stdout + ("\n" + result.stderr if result.stderr else ""), encoding="utf-8", newline="\n")
            inventory = read(job_dir / "child_state.json") if (job_dir / "child_state.json").is_file() else {"status": "unknown"}
            exit_code = result.returncode
        except subprocess.TimeoutExpired as exc:
            job_dir.mkdir(parents=True, exist_ok=True)
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            (job_dir / "child.log").write_text(stdout + "\nTIMEOUT\n" + stderr, encoding="utf-8", newline="\n")
            state_path = job_dir / "child_state.json"
            inventory = read(state_path) if state_path.is_file() else {"status": "timeout", "job": job, "no_retry": True}
            inventory.update({"status": "timeout", "error_type": "TimeoutExpired", "no_retry": True, "unknown_completion": True, "cleanup_started": True, "cleanup_completed": False, "owner_release_status": "timeout_unknown"})
            atomic(state_path, inventory)
            exit_code = 124
        if inventory.get("job") != job:
            exit_code = exit_code or 1
            inventory["identity_error"] = "child state job differs from dispatched job"
        manifest = read(out / "execution_manifest.json")
        if exit_code == 0:
            try:
                _validate_completed_child(job, inventory)
                identity = inventory.get("gpu_identity")
                if identity:
                    bind_gpu_identity(manifest, identity)
                elif not args.runtime_double:
                    raise ContractError("production child did not publish GPU identity")
            except Exception as exc:
                exit_code = 1
                inventory["contract_error"] = f"{type(exc).__name__}:{exc}"
        manifest["jobs"].append({"job": job, "exit_code": exit_code, "state": inventory}); atomic(out / "execution_manifest.json", manifest)
        if exit_code != 0:
            manifest["status"] = "incomplete_blocked"; atomic(out / "execution_manifest.json", manifest); return exit_code
    manifest = read(out / "execution_manifest.json"); manifest["status"] = "completed_review_required"; manifest["finished_utc"] = datetime.now(timezone.utc).isoformat(); atomic(out / "execution_manifest.json", manifest); return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--plan", type=Path, required=True); parser.add_argument("--out-dir", type=Path, required=True); parser.add_argument("--device", default="0"); parser.add_argument("--child-timeout", type=int, default=21600); parser.add_argument("--runtime-double", action="store_true"); parser.add_argument("--confirm-desktop-process", action="append", default=[]); parser.add_argument("--confirm-background-process", action="append", default=[]); parser.add_argument("--child", action="store_true"); parser.add_argument("--job-json")
    args = parser.parse_args(argv)
    if args.child:
        if args.job_json != "-": raise SystemExit("child requires --job-json -")
        payload = json.load(sys.stdin)
        descriptor, temporary_name = tempfile.mkstemp(prefix="confirmation_job_", suffix=".json")
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        args.job_json = str(temporary)
        try: return run_child(args)
        finally: temporary.unlink(missing_ok=True)
    return run_parent(args)


if __name__ == "__main__": raise SystemExit(main())
