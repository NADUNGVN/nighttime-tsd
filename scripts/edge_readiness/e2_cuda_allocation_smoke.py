#!/usr/bin/env python3
"""Default-disabled E2 allocation-only CUDA copy smoke.

No engine, TensorRT, model, inference, benchmark or power collection is part
of this command. Real execution requires the explicit flag and remains a
separate authorization gate from this local implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Keep the documented ``python scripts/...`` invocation CPU-safe and usable
# from the repository root without requiring a pre-set PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_readiness.cuda_runtime_owner import (
    MAX_ALLOCATION_BYTES,
    CudaRuntime,
    CudaRuntimeMemoryOwner,
    CudaRuntimeStream,
    _validate_size,
)
from edge_readiness.jetson_adapter import AdapterError

SMOKE_PAYLOAD = bytes((index * 37 + 11) % 256 for index in range(4096))


def build_plan() -> dict[str, Any]:
    return {
        "schema_version": "e2l1-cuda-allocation-smoke-v1",
        "status": "proposed_default_disabled",
        "target": "E2",
        "real_device_execution": False,
        "authorization_required": "--execute-real-device plus separate Astra authorization",
        "payload_nbytes": len(SMOKE_PAYLOAD),
        "payload_sha256": hashlib.sha256(SMOKE_PAYLOAD).hexdigest(),
        "max_device_allocation_bytes": MAX_ALLOCATION_BYTES,
        "operations": ["one owned stream", "one device allocation", "one synchronous H2D", "one synchronous D2H", "one stream completion check", "exact-byte compare", "free allocation", "destroy stream"],
        "forbidden": ["TensorRT", "engine", "model", "inference", "benchmark", "power/thermal collection", "package installation", "device configuration"],
        "artifact_policy": "new scoped JSON manifest only; no payload bytes or device binary",
    }


def run_roundtrip(owner: CudaRuntimeMemoryOwner, stream: CudaRuntimeStream, payload: bytes = SMOKE_PAYLOAD) -> dict[str, Any]:
    _validate_size(len(payload), max_bytes=MAX_ALLOCATION_BYTES)
    pointer: int | None = None
    destination = bytearray(len(payload))
    primary_failure: BaseException | None = None
    try:
        pointer = owner.allocate_device(len(payload), "allocation_smoke")
        owner.copy_host_to_device(payload, pointer, stream.handle)
        owner.copy_device_to_host(pointer, destination, stream.handle)
        stream.synchronize("allocation_smoke_completion")
        if bytes(destination) != payload:
            raise AdapterError("CUDA_ROUNDTRIP_MISMATCH", "small allocation roundtrip bytes differ", {"expected_sha256": hashlib.sha256(payload).hexdigest(), "observed_sha256": hashlib.sha256(destination).hexdigest()})
        return {"status": "pass", "nbytes": len(payload), "payload_sha256": hashlib.sha256(payload).hexdigest(), "observed_sha256": hashlib.sha256(destination).hexdigest(), "stream_handle": stream.handle}
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        if pointer is not None:
            cleanup_errors: list[Exception] = []
            try:
                owner.synchronize(stream.handle)
            except Exception as exc:
                cleanup_errors.append(exc)
            try:
                owner.free_device(pointer)
            except Exception as exc:
                cleanup_errors.append(exc)
            if primary_failure is None and cleanup_errors:
                raise cleanup_errors[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2-only allocation/copy smoke; real execution is disabled by default")
    parser.add_argument("--target", default="E2")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--execute-real-device", action="store_true", help="explicitly opt into the separately authorized E2 CUDA roundtrip")
    args = parser.parse_args(argv)
    if args.target != "E2":
        print(json.dumps({"status": "blocked", "reason": "E2_ONLY"}, indent=2))
        return 2
    if not args.execute_real_device:
        print(json.dumps(build_plan(), indent=2))
        return 0
    if args.out_dir is None:
        print(json.dumps({"status": "blocked", "reason": "OUT_DIR_REQUIRED"}, indent=2))
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=False)
    runtime = CudaRuntime().open()
    stream = runtime.create_stream()
    owner = CudaRuntimeMemoryOwner(runtime, stream)
    try:
        result = {**build_plan(), "status": "executed_allocation_only", "real_device_execution": True, "roundtrip": run_roundtrip(owner, stream)}
        (args.out_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        result = {**build_plan(), "status": "failed_allocation_only", "real_device_execution": True, "error": {"code": getattr(exc, "code", type(exc).__name__), "message": str(exc)}}
        (args.out_dir / "failure.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(result, indent=2))
        return 2
    finally:
        stream.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
