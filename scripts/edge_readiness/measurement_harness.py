#!/usr/bin/env python3
"""Backend-neutral CPU/mock measurement contract for the edge readiness lane.

This module is deliberately independent of TensorRT, HailoRT, CUDA, image
decoding libraries, and model packages.  It defines the future session boundary
and can be exercised with a deterministic mock adapter only.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import inspect
import json
import math
import numbers
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
UNKNOWN_ALIGNMENT = {"", "unknown", "unverified", "none", "null"}
SESSION_CLOCK_IDENTITY = "host_perf_counter_ns"
SESSION_CLOCK_METHOD = "time.perf_counter_ns"
SESSION_ALIGNMENT_SOURCE = "same_process_run_session"


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
    pool_ids: tuple[str, ...] = ()
    pool_hash: str | None = None
    pool_order: tuple[int, ...] = ()

    def validate(self) -> None:
        if not self.target_id or self.target_id.startswith("-") or any(c.isspace() for c in self.target_id):
            raise HarnessError("INVALID_TARGET", "target_id must be a non-option stable alias")
        if not self.backend or not self.model_id:
            raise HarnessError("INVALID_BINDING", "backend and model_id are required")
        digest = self.model_sha256.lower()
        if len(digest) != 64 or any(c not in SHA256_RE for c in digest):
            raise HarnessError("INVALID_MODEL_HASH", "model_sha256 must be a 64-character lowercase hexadecimal SHA-256")
        if not isinstance(self.boundary, Boundary):
            raise HarnessError("INVALID_BOUNDARY", "boundary must be an explicit Boundary enum value")
        if tuple(self.input_shape) != DEFAULT_INPUT_SHAPE:
            raise HarnessError("INVALID_INPUT_SHAPE", "E2L1-002 requires batch-1 input shape (1,3,640,640)")
        if self.batch != 1:
            raise HarnessError("INVALID_BATCH", "E2L1-002 requires batch=1")
        if self.warmup_calls < 0 or self.measured_calls <= 0 or self.session_count <= 0:
            raise HarnessError("INVALID_COUNTS", "warmup_calls >= 0, measured_calls > 0, and session_count > 0 are required")
        if self.decoded_from_disk_in_timing:
            raise HarnessError("DISK_DECODE_BOUNDARY", "disk decode is excluded; use a decoded image provider")
        if self.pipelined_throughput_measured:
            raise HarnessError("PIPELINED_UNSUPPORTED", "this harness reports serial latency only")
        if not self.pool_ids or not self.pool_order:
            raise HarnessError("INVALID_POOL_BINDING", "pool_ids and pool_order are required")
        if any(not isinstance(image_id, str) or not image_id for image_id in self.pool_ids):
            raise HarnessError("INVALID_POOL_BINDING", "pool_ids must contain non-empty strings")
        if len(set(self.pool_ids)) != len(self.pool_ids):
            raise HarnessError("INVALID_POOL_BINDING", "pool_ids must be unique")
        if any(isinstance(index, bool) or not isinstance(index, numbers.Integral) for index in self.pool_order):
            raise HarnessError("INVALID_POOL_BINDING", "pool_order must contain integer, non-boolean indices")
        if sorted(self.pool_order) != list(range(len(self.pool_ids))):
            raise HarnessError("INVALID_POOL_BINDING", "pool_order must be a permutation of pool indices")
        if self.pool_hash is None or len(self.pool_hash) != 64 or any(c not in SHA256_RE for c in self.pool_hash.lower()):
            raise HarnessError("INVALID_POOL_BINDING", "pool_hash must be a 64-character hexadecimal SHA-256")


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
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise HarnessError("NONFINITE_SAMPLE", "numeric samples are required") from exc
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


def _coverage_metadata(
    *,
    requested_start_ns: int,
    requested_end_ns: int,
    first_sample_ns: int | None,
    last_sample_ns: int | None,
    coverage_start_ns: int | None,
    coverage_end_ns: int | None,
    sample_count: int,
    gaps_ns: list[int] | None = None,
    clock_identity: str | None = None,
    alignment_source: str | None = None,
    method: str = "trapezoidal_linear_interpolation",
) -> dict[str, Any]:
    requested_duration_ns = requested_end_ns - requested_start_ns
    covered_duration_ns = 0 if coverage_start_ns is None or coverage_end_ns is None else max(0, coverage_end_ns - coverage_start_ns)
    return {
        "requested_start_ns": requested_start_ns,
        "requested_end_ns": requested_end_ns,
        "requested_duration_ns": requested_duration_ns,
        "first_sample_ns": first_sample_ns,
        "last_sample_ns": last_sample_ns,
        "coverage_start_ns": coverage_start_ns,
        "coverage_end_ns": coverage_end_ns,
        "covered_duration_ns": covered_duration_ns,
        "coverage_fraction": covered_duration_ns / requested_duration_ns if requested_duration_ns else 0.0,
        "full_interval_covered": covered_duration_ns == requested_duration_ns,
        "sample_count": sample_count,
        "gaps_ns": gaps_ns or [],
        "max_gap_ns": max(gaps_ns or [0]),
        "clock_identity": clock_identity,
        "alignment_source": alignment_source,
        "method": method,
    }


def _validate_clock_conversion(conversion: dict[str, Any], expected_clock_identity: str) -> tuple[str, float, int]:
    if not isinstance(conversion, dict):
        raise HarnessError("INVALID_CLOCK_CONVERSION", "clock_conversion must be a validated mapping")
    source = conversion.get("source_clock_identity")
    target = conversion.get("target_clock_identity")
    evidence = conversion.get("evidence")
    validated = conversion.get("validated")
    offset_ns = conversion.get("offset_ns")
    scale = conversion.get("scale", 1.0)
    if not isinstance(source, str) or not source or target != expected_clock_identity:
        raise HarnessError("INVALID_CLOCK_CONVERSION", "clock conversion source/target must bind to the expected target clock")
    if validated is not True or not isinstance(evidence, str) or not evidence.strip():
        raise HarnessError("INVALID_CLOCK_CONVERSION", "clock conversion requires validated=true and explicit evidence")
    if isinstance(offset_ns, bool) or not isinstance(offset_ns, numbers.Integral):
        raise HarnessError("INVALID_CLOCK_CONVERSION", "clock conversion offset_ns must be an integer")
    if isinstance(scale, bool) or not isinstance(scale, numbers.Real) or not math.isfinite(float(scale)) or float(scale) <= 0:
        raise HarnessError("INVALID_CLOCK_CONVERSION", "clock conversion scale must be finite and positive")
    return source, float(scale), int(offset_ns)


def integrate_power_energy(
    samples: Iterable[dict[str, Any]],
    start_ns: int,
    end_ns: int,
    *,
    clock_identity: str | None = None,
    alignment_source: str | None = None,
    clock_conversion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Integrate aligned power samples without extrapolation.

    Samples must be supplied in strictly increasing timestamp order and must
    match the explicit expected ``clock_identity`` and ``alignment_source``.
    A different source clock is accepted only with a validated conversion
    mapping and evidence. The interval is measured only when samples bracket
    both requested endpoints; otherwise an overlapping result is explicitly
    ``partial``.
    """
    if not isinstance(start_ns, numbers.Integral) or not isinstance(end_ns, numbers.Integral):
        raise HarnessError("INVALID_TIME_RANGE", "start_ns and end_ns must be integer nanoseconds")
    if end_ns <= start_ns:
        raise HarnessError("INVALID_TIME_RANGE", "end_ns must be greater than start_ns")
    raw_rows = list(samples)
    if len(raw_rows) < 2:
        return {"status": "unavailable", "reason": "fewer than two time-aligned power samples", "coverage": _coverage_metadata(requested_start_ns=start_ns, requested_end_ns=end_ns, first_sample_ns=None, last_sample_ns=None, coverage_start_ns=None, coverage_end_ns=None, sample_count=len(raw_rows), clock_identity=clock_identity, alignment_source=alignment_source)}
    if clock_identity is None or str(clock_identity).lower() in UNKNOWN_ALIGNMENT or alignment_source is None or str(alignment_source).lower() in UNKNOWN_ALIGNMENT:
        raise HarnessError("UNBOUND_CLOCK_ALIGNMENT", "integration requires an explicit expected clock identity and alignment source")

    rows: list[tuple[int, float, str, str, str]] = []
    previous_ns: int | None = None
    boundaries: set[str] = set()
    units: set[str] = set()
    clocks: set[str] = set()
    alignments: set[str] = set()
    source_clocks: set[str] = set()
    conversion_spec: tuple[str, float, int] | None = None
    for row in raw_rows:
        timestamp = row.get("monotonic_ns")
        if isinstance(timestamp, bool) or not isinstance(timestamp, numbers.Real) or not math.isfinite(float(timestamp)) or int(timestamp) != timestamp:
            raise HarnessError("INVALID_TIMESTAMP", "timestamps must be finite integer nanoseconds")
        timestamp = int(timestamp)
        watts = _finite(row.get("power_w"))
        if watts < 0:
            raise HarnessError("INVALID_POWER_SAMPLE", "power_w must be nonnegative")
        boundary_value = row.get("boundary")
        unit_value = row.get("unit")
        row_clock_value = row.get("clock_identity", row.get("clock_id"))
        row_alignment_value = row.get("alignment_source")
        boundary = "" if boundary_value is None else str(boundary_value)
        unit = "" if unit_value is None else str(unit_value)
        row_clock = "" if row_clock_value is None else str(row_clock_value)
        row_alignment = "" if row_alignment_value is None else str(row_alignment_value)
        if not boundary or not unit:
            raise HarnessError("INVALID_POWER_SAMPLE", "power samples require boundary and unit")
        if row_clock.lower() in UNKNOWN_ALIGNMENT or row_alignment.lower() in UNKNOWN_ALIGNMENT:
            raise HarnessError("UNVERIFIED_CLOCK_ALIGNMENT", "power samples require verified clock identity and alignment source")
        source_clocks.add(row_clock)
        if row_clock != str(clock_identity):
            if clock_conversion is None:
                raise HarnessError("CLOCK_CONVERSION_REQUIRED", "power sample clock differs from the expected session clock; validated conversion evidence is required", {"expected": clock_identity, "observed": row_clock})
            if conversion_spec is None:
                conversion_spec = _validate_clock_conversion(clock_conversion, str(clock_identity))
            source_clock, scale, offset_ns = conversion_spec
            if row_clock != source_clock:
                raise HarnessError("MIXED_CLOCK_ALIGNMENT", "converted samples must use one source clock")
            timestamp = int(round(timestamp * scale + offset_ns))
            if timestamp < 0:
                raise HarnessError("INVALID_CLOCK_CONVERSION", "converted timestamps must be nonnegative")
        if row_alignment != str(alignment_source):
            raise HarnessError("ALIGNMENT_BINDING_MISMATCH", "power sample alignment does not match the expected session alignment", {"expected": alignment_source, "observed": row_alignment})
        if previous_ns is not None and timestamp <= previous_ns:
            raise HarnessError("NONMONOTONIC_TIMESTAMPS", "timestamps must be strictly increasing after clock binding; sorting is not implicit")
        rows.append((timestamp, watts, boundary, unit, row_alignment))
        previous_ns = timestamp
        boundaries.add(boundary)
        units.add(unit)
        clocks.add(str(clock_identity))
        alignments.add(row_alignment)
    if len(boundaries) != 1:
        raise HarnessError("MIXED_POWER_BOUNDARY", "all energy samples must use one measurement boundary")
    if len(units) != 1:
        raise HarnessError("MIXED_POWER_UNIT", "all energy samples must use one unit")
    if next(iter(units)) != "W":
        raise HarnessError("INVALID_POWER_SAMPLE", "power samples must use unit=W")
    if len(source_clocks) != 1:
        raise HarnessError("MIXED_CLOCK_ALIGNMENT", "all power samples must use one source clock")
    if len(clocks) != 1 or len(alignments) != 1:
        raise HarnessError("MIXED_CLOCK_ALIGNMENT", "all energy samples must use one verified clock identity and alignment source")

    timestamps = [row[0] for row in rows]
    first_ns, last_ns = timestamps[0], timestamps[-1]
    overlap_start = max(start_ns, first_ns)
    overlap_end = min(end_ns, last_ns)
    gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    coverage = _coverage_metadata(
        requested_start_ns=start_ns,
        requested_end_ns=end_ns,
        first_sample_ns=first_ns,
        last_sample_ns=last_ns,
        coverage_start_ns=overlap_start if overlap_start < overlap_end else None,
        coverage_end_ns=overlap_end if overlap_start < overlap_end else None,
        sample_count=len(rows),
        gaps_ns=gaps,
        clock_identity=next(iter(clocks)),
        alignment_source=next(iter(alignments)),
    )
    if overlap_start >= overlap_end:
        return {"status": "unavailable", "reason": "power samples do not overlap measured interval", "coverage": coverage}

    def interpolate(timestamp: int) -> float:
        if timestamp == first_ns:
            return rows[0][1]
        if timestamp == last_ns:
            return rows[-1][1]
        index = bisect.bisect_left(timestamps, timestamp)
        if index < len(timestamps) and timestamps[index] == timestamp:
            return rows[index][1]
        left_t, left_w = rows[index - 1][0], rows[index - 1][1]
        right_t, right_w = rows[index][0], rows[index][1]
        ratio = (timestamp - left_t) / (right_t - left_t)
        return left_w + (right_w - left_w) * ratio

    points = [(overlap_start, interpolate(overlap_start))]
    points.extend((timestamp, watts) for timestamp, watts, *_ in rows if overlap_start < timestamp < overlap_end)
    points.append((overlap_end, interpolate(overlap_end)))
    joules = sum((left_w + right_w) / 2.0 * (right_t - left_t) / 1_000_000_000.0 for (left_t, left_w), (right_t, right_w) in zip(points, points[1:]))
    status = "measured" if coverage["full_interval_covered"] else "partial"
    result = {"status": status, "energy_j": joules, "unit": "J", "boundary": next(iter(boundaries)), "coverage": coverage}
    if status != "measured":
        result["reason"] = "samples do not bracket the full measured interval; energy is covered overlap only"
    return result


def _run_one_call(
    config: MeasurementConfig,
    backend: BackendAdapter,
    decoded_image: Any,
    preprocess: Callable[[Any], Preprocessed],
    postprocess: Callable[[Any], Any],
    *,
    timed: bool,
    clock_ns: Callable[[], int],
) -> tuple[float | None, dict[str, Any]]:
    start_ns: int | None = None
    if config.boundary is Boundary.INFERENCE_ONLY:
        prepared = preprocess(decoded_image)
        if prepared.shape != config.input_shape:
            raise HarnessError("SHAPE_MISMATCH", "preprocessing did not produce the locked input shape", {"observed": prepared.shape})
        backend.synchronize("before_timer")
        if timed:
            start_ns = clock_ns()
        output = backend.infer(prepared)
        backend.synchronize("after_inference")
        if timed:
            elapsed = (clock_ns() - start_ns) / 1_000_000.0
        else:
            elapsed = None
        return elapsed, {"detections": unavailable("postprocess excluded by inference_only boundary")}

    backend.synchronize("before_timer")
    if timed:
        start_ns = clock_ns()
    prepared = preprocess(decoded_image)
    if prepared.shape != config.input_shape:
        raise HarnessError("SHAPE_MISMATCH", "preprocessing did not produce the locked input shape", {"observed": prepared.shape})
    output = backend.infer(prepared)
    backend.synchronize("after_inference")
    detections = postprocess(output)
    if inspect.isawaitable(detections):
        if inspect.iscoroutine(detections):
            detections.close()
        raise HarnessError("ASYNC_POSTPROCESS_UNSUPPORTED", "postprocessing must complete synchronously on the CPU/mock path")
    if timed:
        elapsed = (clock_ns() - start_ns) / 1_000_000.0
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
    include_warmup_energy: bool = False,
    clock_ns: Callable[[], int] | None = None,
) -> dict[str, Any]:
    """Run one session; intended for CPU/mock tests until separately authorized."""
    config.validate()
    if session_index < 1 or session_index > config.session_count:
        raise HarnessError("INVALID_SESSION", "session_index is outside configured session_count")
    if device.device_id != config.target_id:
        raise HarnessError("DEVICE_BINDING_MISMATCH", "device binding does not match target_id")
    clock = clock_ns or time.perf_counter_ns
    session_clock = {
        "identity": SESSION_CLOCK_IDENTITY,
        "method": SESSION_CLOCK_METHOD,
        "alignment_source": SESSION_ALIGNMENT_SOURCE,
        "monotonic": True,
        "verification": "harness_session_binding; not independent device-clock verification",
        "mock_only": True,
    }
    session_start_ns = clock()
    pool_indices = list(config.pool_order)
    consumed: dict[str, dict[str, Any]] = {}

    def sequence_record(indices: list[int]) -> dict[str, Any]:
        ids = [config.pool_ids[index] for index in indices]
        sequence = {"indices": indices, "ids": ids}
        return {
            **sequence,
            "count": len(indices),
            "sequence_hash": hashlib.sha256(json.dumps(sequence, separators=(",", ":")).encode("utf-8")).hexdigest(),
        }

    warmup_indices: list[int] = []
    for index in range(config.warmup_calls):
        pool_index = pool_indices[index % len(pool_indices)]
        warmup_indices.append(pool_index)
        _run_one_call(config, backend, image_provider(pool_index), preprocess, postprocess, timed=False, clock_ns=clock)
    consumed["warmup"] = sequence_record(warmup_indices)

    raw: list[float] = []
    measured_indices: list[int] = []
    measured_start = clock()
    for index in range(config.measured_calls):
        pool_index = pool_indices[index % len(pool_indices)]
        measured_indices.append(pool_index)
        latency, _ = _run_one_call(config, backend, image_provider(pool_index), preprocess, postprocess, timed=True, clock_ns=clock)
        assert latency is not None
        raw.append(latency)
    measured_end = clock()
    session_end_ns = measured_end
    measured_window = {
        "start_monotonic_ns": measured_start,
        "end_monotonic_ns": measured_end,
        "duration_ns": measured_end - measured_start,
        "image_count": config.measured_calls,
        "includes_warmup": False,
        "scope": "measured_loop_outer_window",
    }
    session_window = {
        "start_monotonic_ns": session_start_ns,
        "end_monotonic_ns": session_end_ns,
        "duration_ns": session_end_ns - session_start_ns,
        "image_count": config.warmup_calls + config.measured_calls,
        "scope": "total_session_window",
    }
    consumed["measured"] = sequence_record(measured_indices)
    power_sample_rows = list(power_samples or [])
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
        "session_window": session_window,
        "measured_window": measured_window,
        "session_clock": session_clock,
        "session_start_monotonic_ns": session_start_ns,
        "session_end_monotonic_ns": session_end_ns,
        "session_duration_ns": session_end_ns - session_start_ns,
        "warmup_in_timing": False,
        "raw_latency_ms": raw,
        "latency_statistics": summarize_latencies(raw),
        "throughput_semantics": {
            "serial_fps": "reciprocal of mean synchronous latency",
            "pipelined_throughput_fps": unavailable("no pipelined producer/consumer measurement"),
        },
        "device": asdict(device),
        "image_pool": {
            "pool_ids": list(config.pool_ids),
            "pool_hash": config.pool_hash,
            "pool_order": list(config.pool_order),
            "pool_size": len(config.pool_ids),
            "phase_policy": "restart_each_phase",
            "consumed": consumed,
        },
        "memory_peak": memory_peak or unavailable("not measured by CPU/mock harness", scope="backend-specific"),
        "energy_interval": {
            "start_monotonic_ns": measured_start,
            "end_monotonic_ns": measured_end,
            "duration_ns": measured_end - measured_start,
            "image_count": config.measured_calls,
            "includes_warmup": False,
            "boundary": "declared_by_power_samples_or_unavailable",
            "includes_provider_overhead": True,
            "scope": "measured_loop_outer_window",
        },
        "latency_interval": {
            "boundary": config.boundary.value,
            "image_count": config.measured_calls,
            "includes_warmup": False,
            "scope": "per_call_timing_boundary",
        },
        "power_energy": integrate_power_energy(power_sample_rows, measured_start, measured_end, clock_identity=session_clock["identity"], alignment_source=session_clock["alignment_source"]),
        "mock_only": True,
    }
    if include_warmup_energy:
        result["warmup_inclusive_energy_interval"] = {
            "start_monotonic_ns": session_start_ns,
            "end_monotonic_ns": session_end_ns,
            "duration_ns": session_end_ns - session_start_ns,
            "image_count": config.warmup_calls + config.measured_calls,
            "includes_warmup": True,
            "scope": "total_session_window",
        }
        result["warmup_inclusive_power_energy"] = integrate_power_energy(power_sample_rows, session_start_ns, session_end_ns, clock_identity=session_clock["identity"], alignment_source=session_clock["alignment_source"])
    return result


def write_json_no_overwrite(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise HarnessError("OUTPUT_EXISTS", "refusing to overwrite existing output", {"path": str(path)}) from exc


def mock_device(target_id: str = "E1") -> DeviceBinding:
    evidence = unavailable("CPU/mock test fixture; no device telemetry sampled")
    return DeviceBinding(target_id, "mock-cpu-fixture", {"runtime": "mock"}, evidence, evidence, evidence, evidence, evidence, evidence)


def mock_pool() -> tuple[tuple[str, ...], str, tuple[int, ...]]:
    pool_ids = tuple(f"mock-{index:03d}" for index in range(256))
    pool_hash = hashlib.sha256("\n".join(pool_ids).encode("utf-8")).hexdigest()
    return pool_ids, pool_hash, tuple(range(len(pool_ids)))


def run_mock_sessions(out_dir: Path, *, measured_calls: int = 5, warmup_calls: int = 2) -> dict[str, Any]:
    """Create a tiny local-only mock artifact for tests/manual contract review."""
    pool_ids, pool_hash, pool_order = mock_pool()
    config = MeasurementConfig("E1", "cpu_fp32_reference", "mock-model", "0" * 64, warmup_calls=warmup_calls, measured_calls=measured_calls, pool_ids=pool_ids, pool_hash=pool_hash, pool_order=pool_order)
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
