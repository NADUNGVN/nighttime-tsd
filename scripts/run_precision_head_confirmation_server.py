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
        if state.get("warmup_calls") != 0:
            raise ContractError("child reported an uncounted warmup")
    if state.get("owner_release_status") != "released" or not state.get("cleanup_completed"):
        raise ContractError("child owner lifecycle was not released after cleanup")


def runtime_double(job: dict[str, Any], plan: dict[str, Any], out: Path, state_path: Path) -> None:
    """Synthetic external adapter used only by CPU integration tests."""
    state = read(state_path) if state_path.is_file() else child_state(state_path, job)
    validate_arm_for_phase(job["phase"], job["arm"])
    model_spec = plan.get("models", {}).get(job["model"], {})
    engine_sha = hashlib.sha256(job_key(job).encode()).hexdigest()
    cache_sha = hashlib.sha256(f"cache:{job['model']}:{job['selection']}".encode()).hexdigest()
    timing_in = hashlib.sha256(b"").hexdigest()
    timing_out = hashlib.sha256(f"timing:{job_key(job)}".encode()).hexdigest()
    state.update({"status": "completed", "builder_attempted": True, "builder_completed": True, "capture_attempted": bool(job["capture_required"]), "capture_completed": bool(job["capture_required"]), "calibration_batches": 1024 if job["phase"] == "auxiliary_calibration" else 0, "calibration_read_calls": 2 if job["phase"] == "scored_int8" else 0, "calibration_write_calls": 1 if job["phase"] == "auxiliary_calibration" else 0, "synchronization_completed": 1636 if job["capture_required"] else 0, "forward_attempted": 1636 if job["capture_required"] else 0, "forward_completed": 1636 if job["capture_required"] else 0, "warmup_calls": 0, "cache_consumed": job["phase"] == "scored_int8", "engine_sha256": engine_sha, "calibration_cache": {"selection": job.get("selection"), "before_sha256": cache_sha if job["phase"] == "scored_int8" else None, "after_sha256": cache_sha if job["phase"] in {"auxiliary_calibration", "scored_int8"} else None, "immutable": job["phase"] == "scored_int8", "ordered_images_sha256": "runtime-double-ordered-images"}, "timing_cache": {"input_sha256": timing_in, "output_sha256": timing_out, "reused": False}, "precision_inspector": {"requested_targets": selected_targets(model_spec.get("mapping", {}), job["arm"]) if model_spec.get("mapping") else [], "effective_targets": [], "effective_precision": "fp16" if job["phase"] == "scored_fp16" else "int8"}, "cleanup_started": True, "cleanup_completed": True, "owner_release_status": "released"})
    atomic(state_path, state)
    if job["capture_required"]:
        records = [{"image": f"{index:04d}.jpg", "orig_shape": [100, 100], "xyxy": [], "confidence": [], "class_id": [], "validator_input": {"imgsz": [640, 640], "ratio_pad": [[1.0, 1.0], [0.0, 0.0]], "prediction_xyxy": [], "target_xyxy": [], "target_class_id": []}, "validator_statistics": {"tp": [], "confidence_dtype": "float64", "pred_class_dtype": "float64", "target_class_dtype": "int64"}} for index in range(1636)]
        prediction = {"schema_version": 2, "capture_mode": "same_val_process_batch", "iou_thresholds": [0.5 + 0.05 * index for index in range(10)], "records": records, "model_sha256": state["engine_sha256"]}
        prediction_path = out / "predictions.json"
        atomic(prediction_path, prediction)
        prediction_sha256 = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        atomic(out / "cell_metrics.json", {"schema_version": 1, "job": job, "job_id": job_key(job), "status": "synthetic_external_runtime_double", "plan_sha256": plan.get("_plan_sha256", "runtime-double-plan-bound-at-test"), "checkpoint_sha256": model_spec.get("checkpoint", {}).get("sha256", "runtime-double-checkpoint"), "onnx_sha256": model_spec.get("onnx", {}).get("sha256", "runtime-double-onnx"), "engine_sha256": state["engine_sha256"], "metrics": {"full": {"ap50": 0.0, "ap50_95": 0.0}, "XS": {"ap50": 0.0, "ap50_95": 0.0}, "S": {"ap50": 0.0, "ap50_95": 0.0}}, "statistics_replay": {"status": "pass", "images": 1636}, "native_matching": {"status": "pass", "changed_tp_decisions": 0}, "xml_validation": {"path": "external-test-xml", "sha256": "external-test-xml", "images": 1636, "instances": 2706}, "prediction_path": "predictions.json", "prediction_sha256": prediction_sha256, "record_count": 1636, "raw_output_source": "external_runtime_double_capture_adapter", "postprocess_route": model_spec.get("postprocess", "double")})


def build_real(job: dict[str, Any], plan: dict[str, Any], repo: Path, out: Path, args: argparse.Namespace, state_path: Path) -> None:
    """Build one real server cell and capture it with the locked validator."""
    import hashlib
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
    desktop = parse_desktop_confirmations(args.confirm_desktop_process)
    if args.confirm_background_process:
        raise ContractError("confirmation build-variation study does not authorize competing background compute")
    background = parse_background_confirmations([])
    with GpuPhaseLock(repo / "results/architecture_matrix_v1/.gpu_phase.lock", f"confirmation_{job_key(job)}"):
        before = snapshot(desktop, background)
        ensure_idle(before, background)
        update_state(state_path, read(state_path), "telemetry_before_build", telemetry={"before_build": before})
        runtime_contract = plan.get("runtime", {})
        observed_runtime = {"torch": torch.__version__, "ultralytics": ultralytics.__version__, "tensorrt": trt.__version__, "cuda": torch.version.cuda, "gpu": gpu_identity["name"], "gpu_identity": gpu_identity}
        for key in ("torch", "ultralytics", "tensorrt", "cuda"):
            if runtime_contract.get(key) and observed_runtime[key] != runtime_contract[key]:
                raise ContractError(f"runtime {key} differs: expected={runtime_contract[key]!r} observed={observed_runtime[key]!r}")
        update_state(state_path, read(state_path), "runtime_loaded", runtime=observed_runtime, gpu_identity=gpu_identity)
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError("TensorRT ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
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
        update_state(state_path, read(state_path), "build", builder_attempted=True, precision_constraints=observed_constraints, precision_inspector={"requested_targets": sorted(targets), "effective_targets": sorted(effective_precision), "effective_precision": effective_precision})
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
        (private / "inspector.json").write_text(
            inspector.get_engine_information(trt.LayerInformationFormat.JSON), encoding="utf-8"
        )
        del inspector, engine, runtime
        cache_after = hashlib.sha256(cache_path.read_bytes()).hexdigest() if cache_path.is_file() else None
        if job["phase"] == "scored_int8" and cache_before_sha != cache_after:
            raise ContractError("scored calibration cache changed during build")
        calibration_audit = None
        if calibration_binding is not None:
            calibration_audit = {"selection": calibration_binding["id"], "manifest_sha256": calibration_binding["manifest_sha256"], "yaml_sha256": calibration_binding["yaml_sha256"], "expected_images_sha256": calibration_binding["ordered_images_sha256"], "observed_images_sha256": sha256_bytes(json.dumps(getattr(calibrator, "observed_images", []), separators=(",", ":"), ensure_ascii=False).encode("utf-8")), "tensor_sequence_sha256": getattr(getattr(calibrator, "tensor_hash", None), "hexdigest", lambda: None)(), "batch_count": getattr(calibrator, "batch_calls", 0), "input_shape": [1, 3, 640, 640], "input_dtype": "torch.float32"}
            if job["phase"] == "auxiliary_calibration" and calibration_audit["observed_images_sha256"] != calibration_binding["ordered_images_sha256"]:
                raise ContractError("calibration loader order audit hash mismatch")
        update_state(state_path, read(state_path), "build_complete", builder_completed=True, engine_sha256=hashlib.sha256(engine_path.read_bytes()).hexdigest(), calibration_batches=calibrator.batch_calls if job["phase"] != "scored_fp16" else 0, calibration_read_calls=calibrator.read_calls if job["phase"] != "scored_fp16" else 0, calibration_write_calls=calibrator.write_calls if job["phase"] != "scored_fp16" else 0, cache_consumed=getattr(calibrator, "cache_consumed", False) if job["phase"] != "scored_fp16" else False, calibration_cache={"selection": job.get("selection"), "before_sha256": cache_before_sha, "after_sha256": cache_after, "immutable": job["phase"] == "scored_int8", "ordered_images_sha256": calibration_audit["expected_images_sha256"] if calibration_audit else None, "audit": calibration_audit}, timing_cache=timing_audit)
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
            class CountingNoWarmupBackend(backend_base):
                total_calls = 0
                warmup_calls = 0
                last_instance = None
                def __init__(self, *backend_args: Any, **backend_kwargs: Any) -> None:
                    super().__init__(*backend_args, **backend_kwargs)
                    type(self).last_instance = self
                def warmup(self, *warmup_args: Any, **warmup_kwargs: Any) -> None:
                    type(self).warmup_calls += 1
                def __call__(self, *call_args: Any, **call_kwargs: Any) -> Any:
                    type(self).total_calls += 1
                    return super().__call__(*call_args, **call_kwargs)
            validator_module.AutoBackend = CountingNoWarmupBackend
            validator = CaptureValidator(args={"model": str(engine_path), "data": str(data), "split": "val", "device": args.device, "imgsz": 640, "batch": 1, "workers": 0, "conf": 0.001, "iou": 0.7, "max_det": 300, "rect": False, "plots": False, "verbose": False}, save_dir=private / "validator")
            try:
                validator(model=str(engine_path))
            finally:
                validator_module.AutoBackend = backend_base
            records = validator.capture_records
            if len(records) != 1636 or len({record["image"] for record in records}) != 1636: raise RuntimeError("capture did not cover dev exactly once")
            if CountingNoWarmupBackend.total_calls != 1636 or CountingNoWarmupBackend.warmup_calls != 0:
                raise RuntimeError(f"capture forward accounting mismatch: calls={CountingNoWarmupBackend.total_calls}, warmup={CountingNoWarmupBackend.warmup_calls}")
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
            atomic(state_path.parent / "cell_metrics.json", {"schema_version": 1, "job": job, "job_id": job_key(job), "plan_sha256": hashlib.sha256((out / "confirmation_plan.json").read_bytes()).hexdigest(), "checkpoint_sha256": model_evidence["checkpoint"]["sha256"], "onnx_sha256": model_evidence["onnx"]["sha256"], "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(), "metrics": {"full": {"ap50": metrics["map50"], "ap50_95": metrics["map50_95"]}}, "statistics_replay": replay, "native_matching": matching, "xml_validation": {"path": plan.get("dataset", {}).get("xml_path"), "sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(), "images": len(xml), "instances": sum(len(value["rows"]) for value in xml.values())}, "prediction_path": str(public_predictions.relative_to(out).as_posix()), "prediction_sha256": hashlib.sha256(public_predictions.read_bytes()).hexdigest(), "record_count": len(records), "raw_output_source": "CaptureValidator._process_batch", "postprocess_route": plan["models"][model]["postprocess"]})
            update_state(state_path, read(state_path), "capture_complete", capture_completed=True, synchronization_completed=1636, forward_attempted=CountingNoWarmupBackend.total_calls, forward_completed=CountingNoWarmupBackend.total_calls, warmup_calls=CountingNoWarmupBackend.warmup_calls, backend_metadata={"format": str(getattr(CountingNoWarmupBackend.last_instance, "format", "unknown")), "stride": int(getattr(CountingNoWarmupBackend.last_instance, "stride", 0)), "end2end": bool(getattr(CountingNoWarmupBackend.last_instance, "end2end", False))})
            after_capture = snapshot(desktop, background)
            ensure_idle(after_capture, background)
            state_after_capture = read(state_path)
            state_after_capture.setdefault("telemetry", {}).update({"after_capture": after_capture})
            atomic(state_path, state_after_capture)
            del validator, records, checked_records, xml, matching, replay
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device=torch.device(f"cuda:{args.device}"))
        after = snapshot(desktop, background)
        ensure_idle(after, background)
        state = read(state_path); state.update({"status": "completed", "telemetry": {**state.get("telemetry", {}), "final": after}, "cleanup_started": True, "cleanup_completed": True, "owner_release_status": "released"}); atomic(state_path, state)


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
        if args.runtime_double:
            plan["_plan_sha256"] = plan_sha256
            runtime_double(job, plan, job_out, state_path)
        else:
            build_real(job, plan, Path(args.repo).resolve(), out, args, state_path)
    except Exception as exc:
        state = read(state_path) if state_path.exists() else {"schema_version": 1, "job": job, "status": "unknown", "stage": "unknown", "stage_history": [], "unknown_completion": True}
        state.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "no_retry": True, "cleanup_started": True, "cleanup_completed": False, "owner_release_status": "cleanup_failed_or_unknown"})
        atomic(state_path, state)
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
        if exit_code == 0:
            try:
                _validate_completed_child(job, inventory)
                identity = inventory.get("gpu_identity")
                manifest = read(out / "execution_manifest.json")
                if identity:
                    prior = manifest.get("gpu_identity")
                    if prior and prior != identity:
                        raise ContractError("GPU identity changed within one confirmation study")
                    manifest["gpu_identity"] = identity
            except Exception as exc:
                exit_code = 1
                inventory["contract_error"] = f"{type(exc).__name__}:{exc}"
        manifest = read(out / "execution_manifest.json"); manifest["jobs"].append({"job": job, "exit_code": exit_code, "state": inventory}); atomic(out / "execution_manifest.json", manifest)
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
