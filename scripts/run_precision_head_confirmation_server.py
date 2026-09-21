#!/usr/bin/env python3
"""Server-only executor for ST-SERVER-01.

The parent in this module is CPU orchestration. TensorRT, torch and
Ultralytics are imported only inside a short-lived child. ``--runtime-double``
is an external-runtime test adapter and is never accepted by the production
runbook.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from precision_head_confirmation_contract import ContractError, selected_targets, validate_schedule


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


def runtime_double(job: dict[str, Any], out: Path, state_path: Path) -> None:
    """Synthetic external adapter used only by CPU integration tests."""
    state = child_state(state_path, job)
    state.update({"status": "completed", "builder_attempted": True, "builder_completed": True, "capture_attempted": bool(job["capture_required"]), "capture_completed": bool(job["capture_required"]), "calibration_batches": 1024 if job["phase"] == "auxiliary_calibration" else 0, "calibration_read_calls": 2 if job["phase"] == "scored_int8" else 0, "calibration_write_calls": 1 if job["phase"] == "auxiliary_calibration" else 0, "owner_release_status": "released", "synchronization_completed": 1636 if job["capture_required"] else 0})
    atomic(state_path, state)
    if job["capture_required"]:
        atomic(out / "cell_metrics.json", {"schema_version": 1, "job": job, "job_id": job_key(job), "status": "synthetic_external_runtime_double", "metrics": {"full": {"ap50": 0.0, "ap50_95": 0.0}, "XS": {"ap50": 0.0, "ap50_95": 0.0}, "S": {"ap50": 0.0, "ap50_95": 0.0}}, "predictions_source": "test_double_only"})


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
    model_evidence = plan["models"][model]
    checkpoint = repo / model_evidence["checkpoint"]["path"]
    onnx = repo / model_evidence["onnx"]["path"]
    mapping = model_evidence["mapping"]["targets"]
    private = out / "private" / model / (job["selection"] or "fp16") / job_key(job)
    private.mkdir(parents=True, exist_ok=False)
    engine_path = private / "model.engine"
    cache_path = out / "private" / "calibration" / model / f"{job['selection']}.cache"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    desktop = parse_desktop_confirmations(args.confirm_desktop_process)
    background = parse_background_confirmations(args.confirm_background_process)
    with GpuPhaseLock(repo / "results/architecture_matrix_v1/.gpu_phase.lock", f"confirmation_{job_key(job)}"):
        before = snapshot(desktop, background)
        ensure_idle(before, background)
        update_state(state_path, read(state_path), "runtime_loaded", runtime={"torch": torch.__version__, "ultralytics": ultralytics.__version__, "tensorrt": trt.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
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
                selection = {"U42": 42, "U43": 43, "U44": 44}[job["selection"]]
                calibration_yaml = repo / "data/processed/cctsdb2021_clean/calibration" / f"uniform_s{selection}_n1024" / "calibration.yaml"
                if not calibration_yaml.is_file():
                    raise FileNotFoundError(f"materialized calibration YAML missing: {calibration_yaml}")
                exporter = Exporter(overrides={"format": "engine", "data": str(calibration_yaml), "imgsz": 640, "batch": 1, "fraction": 1.0, "split": "val", "rect": False, "device": str(args.device)})
                exporter.imgsz = (640, 640)
                exporter.model = SimpleNamespace(task="detect")
                loader = iter(exporter.get_int8_calibration_dataloader())

            class CacheOnly(trt.IInt8Calibrator):
                def __init__(self) -> None:
                    super().__init__(); self.read_calls = 0; self.write_calls = 0; self.batch_calls = 0; self.eos_calls = 0; self.cache_consumed = False; self.tensor = None
                def get_batch_size(self) -> int: return 1
                def get_algorithm(self) -> Any: return trt.CalibrationAlgoType.MINMAX_CALIBRATION
                def read_calibration_cache(self) -> bytes:
                    self.read_calls += 1
                    if job["phase"] == "scored_int8":
                        self.cache_consumed = True
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
                    image = batch["img"].to(device=torch.device(f"cuda:{args.device}"), dtype=torch.float32) / 255.0
                    self.tensor = image.contiguous()
                    return [int(self.tensor.data_ptr())]
            calibrator = CacheOnly()
            config.int8_calibrator = calibrator
        targets = set(selected_targets(mapping, job["arm"]))
        observed_constraints = []
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            if layer.name in targets:
                if layer.type != trt.LayerType.CONVOLUTION:
                    raise RuntimeError(f"precision target is not convolution: {layer.name}")
                layer.precision = trt.float32
                for output_index in range(layer.num_outputs): layer.set_output_type(output_index, trt.float32)
                observed_constraints.append(layer.name)
        if observed_constraints != sorted(targets) and set(observed_constraints) != targets:
            raise RuntimeError(f"precision target mismatch: expected {sorted(targets)}, observed {observed_constraints}")
        update_state(state_path, read(state_path), "build", builder_attempted=True, precision_constraints=observed_constraints)
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
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(bytes(serialized))
        if engine is None:
            raise RuntimeError("TensorRT runtime could not deserialize the just-built engine")
        inspector = engine.create_engine_inspector()
        (private / "inspector.json").write_text(
            inspector.get_engine_information(trt.LayerInformationFormat.JSON), encoding="utf-8"
        )
        del inspector, engine, runtime
        update_state(state_path, read(state_path), "build_complete", builder_completed=True, engine_sha256=hashlib.sha256(engine_path.read_bytes()).hexdigest(), calibration_batches=calibrator.batch_calls if job["phase"] != "scored_fp16" else 0, calibration_read_calls=calibrator.read_calls if job["phase"] != "scored_fp16" else 0, calibration_write_calls=calibrator.write_calls if job["phase"] != "scored_fp16" else 0, cache_consumed=getattr(calibrator, "cache_consumed", False) if job["phase"] != "scored_fp16" else False)
        if job["capture_required"]:
            update_state(state_path, read(state_path), "capture", capture_attempted=True)
            data = private / "dev.yaml"
            dev = repo / "data/processed/cctsdb2021_clean/dev"
            data.write_text(f"path: {dev.as_posix()}\ntrain: images\nval: images\nnames:\n  0: prohibitory\n  1: mandatory\n  2: warning\nnc: 3\n", encoding="utf-8")
            validator = CaptureValidator(args={"model": str(engine_path), "data": str(data), "split": "val", "device": args.device, "imgsz": 640, "batch": 1, "workers": 0, "conf": 0.001, "iou": 0.7, "max_det": 300, "rect": False, "plots": False, "verbose": False}, save_dir=private / "validator")
            validator(model=str(engine_path))
            records = validator.capture_records
            if len(records) != 1636 or len({record["image"] for record in records}) != 1636: raise RuntimeError("capture did not cover dev exactly once")
            predictions = private / "validator_predictions.json"
            predictions.write_text(json.dumps({"schema_version": 2, "records": records, "model_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest()}, allow_nan=False) + "\n", encoding="utf-8")
            public_predictions = state_path.parent / "predictions.json"
            public_predictions.write_bytes(predictions.read_bytes())
            metrics = metric_summary(validator.metrics)
            atomic(state_path.parent / "cell_metrics.json", {"schema_version": 1, "job": job, "job_id": job_key(job), "plan_sha256": hashlib.sha256((out / "confirmation_plan.json").read_bytes()).hexdigest(), "checkpoint_sha256": model_evidence["checkpoint"]["sha256"], "onnx_sha256": model_evidence["onnx"]["sha256"], "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(), "metrics": {"full": {"ap50": metrics["map50"], "ap50_95": metrics["map50_95"]}}, "prediction_path": str(public_predictions.relative_to(out).as_posix()), "prediction_sha256": hashlib.sha256(public_predictions.read_bytes()).hexdigest(), "record_count": len(records), "raw_output_source": "CaptureValidator._process_batch", "postprocess_route": plan["models"][model]["postprocess"]})
            update_state(state_path, read(state_path), "capture_complete", capture_completed=True, synchronization_completed=1636)
        after = snapshot(desktop, background)
        state = read(state_path); state.update({"status": "completed", "telemetry": {"before": before, "after": after}, "owner_release_status": "released"}); atomic(state_path, state)


def run_child(args: argparse.Namespace) -> int:
    job = read(Path(args.job_json))
    out = Path(args.out_dir).resolve()
    state_path = out / "jobs" / f"{job_key(job)}" / "child_state.json"
    job_out = state_path.parent
    job_out.mkdir(parents=True, exist_ok=False)
    try:
        if args.runtime_double:
            runtime_double(job, job_out, state_path)
        else:
            build_real(job, read(Path(args.plan)), Path(args.repo).resolve(), out, args, state_path)
    except Exception as exc:
        state = read(state_path) if state_path.exists() else child_state(state_path, job)
        state.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "no_retry": True})
        atomic(state_path, state)
        return 1
    return 0


def run_parent(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve(); out = Path(args.out_dir).resolve(); plan_path = Path(args.plan).resolve(); plan = read(plan_path)
    jobs = plan["schedule"]["jobs"]; validate_schedule(jobs)
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
            inventory.update({"status": "timeout", "error_type": "TimeoutExpired", "no_retry": True})
            atomic(state_path, inventory)
            exit_code = 124
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
