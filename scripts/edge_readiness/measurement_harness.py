#!/usr/bin/env python3
"""Backend-neutral CPU/mock measurement contract for the edge readiness lane.

This module is deliberately independent of TensorRT, HailoRT, CUDA, image
decoding libraries, and model packages.  It defines the future session boundary
and can be exercised with a deterministic mock adapter only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol


SCHEMA_VERSION = "e2l1-edge-measurement-v1"
PROTOCOL_VERSION = "e2l1-edge-measurement-protocol-v1"
DEFAULT_WARMUP_CALLS = 200
DEFAULT_MEASURED_CALLS = 1000
DEFAULT_SESSION_COUNT = 3
DEFAULT_INPUT_SHAPE = (1, 3, 640, 640)
SHA256_RE = set("0123456789abcdef")


class Boundary(str, Enum):
    """Explicit timing boundary; no implicit backend-specific defaults."""

    INFERENCE_ONLY = "inference_only"
    DECODED_IMAGE_TO_DETECTIONS = "decoded_image_to_detections"


class HarnessError(ValueError):
    """Stable machine-readable validation error with a single error shape."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        return {"status": "error", "error": {"code": self.code, "message": self.message, "details": self.details}}


def unavailable(reason: str, *, scope: str | None = None) -> dict[str, Any]:
    """Represent unknown/unavailable evidence without inventing a numeric value."""
    result: dict[str, Any] = {"status": "unavailable", "reason": reason}
    if scope is not None:
        result["scope"] = scope
    return result


def permission_denied(reason: str, *, scope: str | None = None) -> dict[str, Any]:
    result = {"status": "permission_denied", "reason": reason}
    if scope is not None:
        result["scope"] = scope
    return result


@dataclass(frozen=True)
class DeviceBinding:
    device_id: str
    hardware_identity: str
    runtime_versions: dict[str, str]
    power_mode_before: dict[str, Any]
    power_mode_after: dict[str, Any]
    thermal_before: dict[str, Any]
    thermal_after: dict[str, Any]
    throttle_before: dict[str, Any]
    throttle_after: dict[str, Any]


@dataclass(frozen=True)
class MeasurementConfig:
    target_id: str
    backend: str
    model_id: str
    model_sha256: str
    boundary: Boundary = Boundary.INFERENCE_ONLY
    input_shape: tuple[int, int, int, int] = DEFAULT_INPUT_SHAPE
    batch: int = 1
    warmup_calls: int = DEFAULT_WARMUP_CALLS
    measured_calls: int = DEFAULT_MEASURED_CALLS
    session_count: int = DEFAULT_SESSION_COUNT
    decoded_from_disk_in_timing: bool = False
    pipelined_throughput_measured: bool = False

    def validate(self) -> None:
        if not self.target_id or self.target_id.startswith("-") or any(c.isspace() for c in self.target_id):
            raise HarnessError("INVALID_TARGET", "target_id must be a non-option stable alias")
        if not self.backend or not self.model_id:
            raise HarnessError("INVALID_BINDING", "backend and model_id are required")
        digest = self.model_sha256.lower()
        if len(digest) != 64 or any(c not in SHA256_RE for c in digest):
            raise HarnessError("INVALID_MODEL_HASH", "model_sha256 must be a 64-character lowercase hexadecimal SHA-256")
        if tuple(self.input_shape) != DEFAULT_INPUT_SHAPE:
            raise HarnessError("INVALID_INPUT_SHAPE", "E2L1-002 requires batch-1 input shape (1,3,640,640)")
        if self.batch != 1:
            raise HarnessError("INVALID_BATCH", "E2L1-002 requires batch=1")
        if self.warmup_calls < 0 or self.measured_calls <= 0 or self.session_count <= 0:
            raise HarnessError("INVALID_COUNTS", "warmup_calls >= 0, measured_calls > 0, and session_count > 0 are required")
        if self.decoded_from_disk_in_timing:
            raise HarnessError("DISK_DECODE_BOUNDARY", "disk decode is excluded; use a decoded image provider")


@dataclass(frozen=True)
class Preprocessed:
    payload: Any
    shape: tuple[int, int, int, int]


class BackendAdapter(Protocol):
    """Backend contract: adapter owns synchronization around its work."""

    backend_name: str
    synchronization_policy: str

    def synchronize(self, phase: str) -> None:
        ...

    def infer(self, preprocessed: Preprocessed) -> Any:
        ...


class MockBackend:
    """CPU-only adapter for contract tests; it performs no model inference."""

    backend_name = "mock_cpu"
    synchronization_policy = "no_device_sync; calls are recorded for ordering tests"

    def __init__(self, output: Any = None):
        self.output = output
        self.events: list[str] = []

    def synchronize(self, phase: str) -> None:
        self.events.append(f"sync:{phase}")

    def infer(self, preprocessed: Preprocessed) -> Any:
        self.events.append("infer")
        return self.output


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise HarnessError("NONFINITE_SAMPLE", "latency and telemetry samples must be finite")
    return value


def percentile_linear(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(_finite(v) for v in values)
    if not ordered or not 0.0 <= fraction <= 1.0:
        raise HarnessError("INVALID_PERCENTILE", "percentile requires non-empty values and fraction in [0,1]")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_latencies(latencies_ms: Iterable[float]) -> dict[str, Any]:
    values = [_finite(value) for value in latencies_ms]
    if not values or any(value <= 0 for value in values):
        raise HarnessError("INVALID_LATENCY_SAMPLES", "latency samples must be non-empty and positive")
    mean_ms = statistics.fmean(values)
    return {
        "n_calls": len(values),
        "unit": "ms",
        "mean": mean_ms,
        "median": statistics.median(values),
        "p50": percentile_linear(values, 0.50),
        "p95": percentile_linear(values, 0.95),
        "p99": percentile_linear(values, 0.99),
        "min": min(values),
        "max": max(values),
        "serial_fps": 1000.0 / mean_ms,
        "pipelined_throughput_fps": unavailable("not measured by this serial session"),
    }


def integrate_power_energy(samples: Iterable[dict[str, Any]], start_ns: int, end_ns: int) -> dict[str, Any]:
    """Integrate time-aligned whole-boundary power samples using trapezoids."""
    if end_ns <= start_ns:
        raise HarnessError("INVALID_TIME_RANGE", "end_ns must be greater than start_ns")
    rows = sorted(samples, key=lambda row: int(row["monotonic_ns"]))
    if len(rows) < 2:
        return unavailable("fewer than two time-aligned power samples", scope="unknown")
    previous = None
    joules = 0.0
    for row in rows:
        timestamp = int(row["monotonic_ns"])
        watts = _finite(row["power_w"])
        if watts < 0 or not row.get("boundary") or row.get("unit") != "W":
            raise HarnessError("INVALID_POWER_SAMPLE", "power samples require nonnegative power_w, boundary, and unit=W")
        if previous is not None:
            previous_ns, previous_watts = previous
            left = max(previous_ns, start_ns)
            right = min(timestamp, end_ns)
            if right > left:
                joules += ((previous_watts + watts) / 2.0) * ((right - left) / 1_000_000_000.0)
        previous = (timestamp, watts)
    if previous is None or previous[0] < start_ns or rows[0]["monotonic_ns"] > end_ns:
        return unavailable("power samples do not span the measured interval", scope="unknown")
    boundary = {str(row["boundary"]) for row in rows}
    if len(boundary) != 1:
        raise HarnessError("MIXED_POWER_BOUNDARY", "all energy samples must use one measurement boundary")
    return {"status": "measured", "energy_j": joules, "unit": "J", "boundary": next(iter(boundary)), "method": "trapezoidal"}


def _run_one_call(
    config: MeasurementConfig,
    backend: BackendAdapter,
    decoded_image: Any,
    preprocess: Callable[[Any], Preprocessed],
    postprocess: Callable[[Any], Any],
    *,
    timed: bool,
) -> tuple[float | None, dict[str, Any]]:
    start_ns: int | None = None
    if config.boundary is Boundary.INFERENCE_ONLY:
        prepared = preprocess(decoded_image)
        if prepared.shape != config.input_shape:
            raise HarnessError("SHAPE_MISMATCH", "preprocessing did not produce the locked input shape", {"observed": prepared.shape})
        backend.synchronize("before_timer")
        if timed:
            start_ns = time.perf_counter_ns()
        output = backend.infer(prepared)
        backend.synchronize("after_inference")
        if timed:
            elapsed = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        else:
            elapsed = None
        return elapsed, {"detections": unavailable("postprocess excluded by inference_only boundary")}

    backend.synchronize("before_timer")
    if timed:
        start_ns = time.perf_counter_ns()
    prepared = preprocess(decoded_image)
    if prepared.shape != config.input_shape:
        raise HarnessError("SHAPE_MISMATCH", "preprocessing did not produce the locked input shape", {"observed": prepared.shape})
    output = backend.infer(prepared)
    backend.synchronize("after_inference")
    detections = postprocess(output)
    if timed:
        elapsed = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    else:
        elapsed = None
    return elapsed, {"detections": detections}


def run_session(
    config: MeasurementConfig,
    backend: BackendAdapter,
    image_provider: Callable[[int], Any],
    preprocess: Callable[[Any], Preprocessed],
    postprocess: Callable[[Any], Any],
    *,
    session_index: int,
    device: DeviceBinding,
    power_samples: Iterable[dict[str, Any]] | None = None,
    memory_peak: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one session; intended for CPU/mock tests until separately authorized."""
    config.validate()
    if session_index < 1 or session_index > config.session_count:
        raise HarnessError("INVALID_SESSION", "session_index is outside configured session_count")
    if device.device_id != config.target_id:
        raise HarnessError("DEVICE_BINDING_MISMATCH", "device binding does not match target_id")
    for index in range(config.warmup_calls):
        _run_one_call(config, backend, image_provider(index % 256), preprocess, postprocess, timed=False)
    raw: list[float] = []
    measured_start = time.perf_counter_ns()
    for index in range(config.measured_calls):
        latency, _ = _run_one_call(config, backend, image_provider(index % 256), preprocess, postprocess, timed=True)
        assert latency is not None
        raw.append(latency)
    measured_end = time.perf_counter_ns()
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed_mock",
        "session_index": session_index,
        "target_id": config.target_id,
        "backend": config.backend,
        "backend_adapter": backend.backend_name,
        "synchronization_policy": backend.synchronization_policy,
        "model": {"model_id": config.model_id, "sha256": config.model_sha256},
        "boundary": config.boundary.value,
        "disk_decode_in_timing": config.decoded_from_disk_in_timing,
        "input_shape": list(config.input_shape),
        "batch": config.batch,
        "warmup_calls": config.warmup_calls,
        "measured_calls": config.measured_calls,
        "warmup_in_timing": False,
        "raw_latency_ms": raw,
        "latency_statistics": summarize_latencies(raw),
        "throughput_semantics": {
            "serial_fps": "reciprocal of mean synchronous latency",
            "pipelined_throughput_fps": unavailable("no pipelined producer/consumer measurement"),
        },
        "device": asdict(device),
        "memory_peak": memory_peak or unavailable("not measured by CPU/mock harness", scope="backend-specific"),
        "power_energy": integrate_power_energy(power_samples or [], measured_start, measured_end),
        "mock_only": True,
    }
    return result


def write_json_no_overwrite(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise HarnessError("OUTPUT_EXISTS", "refusing to overwrite existing output", {"path": str(path)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def mock_device(target_id: str = "E1") -> DeviceBinding:
    evidence = unavailable("CPU/mock test fixture; no device telemetry sampled")
    return DeviceBinding(target_id, "mock-cpu-fixture", {"runtime": "mock"}, evidence, evidence, evidence, evidence, evidence, evidence)


def run_mock_sessions(out_dir: Path, *, measured_calls: int = 5, warmup_calls: int = 2) -> dict[str, Any]:
    """Create a tiny local-only mock artifact for tests/manual contract review."""
    config = MeasurementConfig("E1", "cpu_fp32_reference", "mock-model", "0" * 64, warmup_calls=warmup_calls, measured_calls=measured_calls)
    session_paths: list[str] = []
    for session_index in range(1, config.session_count + 1):
        backend = MockBackend(output={"mock": True})
        session = run_session(config, backend, lambda index: {"index": index}, lambda image: Preprocessed(image, DEFAULT_INPUT_SHAPE), lambda output: output, session_index=session_index, device=mock_device())
        path = out_dir / f"session_{session_index}.json"
        write_json_no_overwrite(path, session)
        session_paths.append(str(path))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_mock",
        "mock_only": True,
        "config": asdict(config) | {"boundary": config.boundary.value, "input_shape": list(config.input_shape)},
        "sessions": session_paths,
    }
    write_json_no_overwrite(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local CPU/mock edge measurement harness; never touches a device")
    parser.add_argument("--mock", action="store_true", help="run the local mock contract")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.mock:
        raise SystemExit("Only --mock is available in E2L1-002; device inference is not authorized")
    print(json.dumps(run_mock_sessions(args.out_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
