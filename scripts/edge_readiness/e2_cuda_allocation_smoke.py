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
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from edge_readiness.edge_errors import AdapterError

SMOKE_PAYLOAD = bytes((index * 37 + 11) % 256 for index in range(4096))


def build_plan() -> Dict[str, Any]:
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


def _error_dict(exc: BaseException) -> Dict[str, Any]:
    if isinstance(exc, AdapterError):
        return exc.as_dict()["error"]
    return {"code": type(exc).__name__, "message": str(exc), "details": {}}


def _combine_failures(primary: Optional[BaseException], cleanup: List[Tuple[str, BaseException]]) -> Optional[BaseException]:
    if not cleanup:
        return primary
    details: Dict[str, Any] = {"cleanup": [{"operation": operation, "error": _error_dict(exc)} for operation, exc in cleanup]}
    if primary is not None:
        details["primary"] = _error_dict(primary)
    return AdapterError("SMOKE_CLEANUP_FAILED", "allocation smoke lifecycle failed", details)


def run_roundtrip(owner: CudaRuntimeMemoryOwner, stream: CudaRuntimeStream, payload: bytes = SMOKE_PAYLOAD) -> Dict[str, Any]:
    _validate_size(len(payload), max_bytes=MAX_ALLOCATION_BYTES)
    pointer: Optional[int] = None
    destination = bytearray(len(payload))
    primary_failure: Optional[BaseException] = None
    cleanup_failures: List[Tuple[str, BaseException]] = []
    result: Optional[Dict[str, Any]] = None
    try:
        pointer = owner.allocate_device(len(payload), "allocation_smoke")
        owner.copy_host_to_device(payload, pointer, stream.handle)
        owner.copy_device_to_host(pointer, destination, stream.handle)
        stream.synchronize("allocation_smoke_completion")
        if bytes(destination) != payload:
            raise AdapterError("CUDA_ROUNDTRIP_MISMATCH", "small allocation roundtrip bytes differ", {"expected_sha256": hashlib.sha256(payload).hexdigest(), "observed_sha256": hashlib.sha256(destination).hexdigest()})
        result = {"status": "pass", "nbytes": len(payload), "payload_sha256": hashlib.sha256(payload).hexdigest(), "observed_sha256": hashlib.sha256(destination).hexdigest(), "stream_handle": stream.handle}
    except BaseException as exc:
        primary_failure = exc
    finally:
        if pointer is not None:
            try:
                owner.synchronize(stream.handle)
            except Exception as exc:
                cleanup_failures.append(("allocation_stream_synchronize", exc))
            try:
                owner.free_device(pointer)
            except Exception as exc:
                cleanup_failures.append(("allocation_free", exc))
    combined = _combine_failures(primary_failure, cleanup_failures)
    if combined is not None:
        raise combined
    assert result is not None
    return result


def write_json_exclusive(path: Path, document: Dict[str, Any]) -> None:
    """Write strict JSON using only Python 3.8-compatible APIs."""
    with open(str(path), "x", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _runtime_identity(runtime: CudaRuntime) -> Dict[str, Any]:
    path = getattr(runtime, "library_path", None)
    realpath = os.path.realpath(path) if path else None
    return {"library_path": path, "library_realpath": realpath, "library_basename": os.path.basename(realpath) if realpath else None}


def execute_smoke(out_dir: Path, *, runtime_factory: Callable[[], CudaRuntime] = CudaRuntime, owner_factory: Callable[[CudaRuntime, CudaRuntimeStream], CudaRuntimeMemoryOwner] = CudaRuntimeMemoryOwner, writer: Callable[[Path, Dict[str, Any]], None] = write_json_exclusive) -> Tuple[int, Dict[str, Any]]:
    """Run one allocation smoke and record a terminal artifact after cleanup."""
    out_dir.mkdir(parents=True, exist_ok=False)
    runtime: Optional[CudaRuntime] = None
    stream: Optional[CudaRuntimeStream] = None
    runtime_opened = False
    runtime_identity: Optional[Dict[str, Any]] = None
    primary_failure: Optional[BaseException] = None
    cleanup_failures: List[Tuple[str, BaseException]] = []
    roundtrip: Optional[Dict[str, Any]] = None
    try:
        runtime = runtime_factory()
        runtime.open()
        runtime_opened = True
        runtime_identity = _runtime_identity(runtime)
        stream = runtime.create_stream()
        owner = owner_factory(runtime, stream)
        roundtrip = run_roundtrip(owner, stream)
    except BaseException as exc:
        primary_failure = exc
    finally:
        if stream is not None:
            try:
                stream.close()
            except BaseException as exc:
                cleanup_failures.append(("stream_close", exc))
        if runtime is not None and runtime_opened:
            try:
                runtime.close()
            except BaseException as exc:
                cleanup_failures.append(("runtime_close", exc))

    failure = _combine_failures(primary_failure, cleanup_failures)
    base: Dict[str, Any] = {
        **build_plan(),
        "real_device_execution": True,
        "cleanup": {"stream_close": "ok" if not any(op == "stream_close" for op, _ in cleanup_failures) else "failed", "runtime_close": "ok" if not any(op == "runtime_close" for op, _ in cleanup_failures) else "failed"},
        "runtime_identity": runtime_identity or {"library_path": None, "library_realpath": None, "library_basename": None},
    }
    if failure is not None:
        base["status"] = "failed_allocation_only"
        base["error"] = _error_dict(failure)
        writer(out_dir / "failure.json", base)
        return 2, base
    base["status"] = "executed_allocation_only"
    base["roundtrip"] = roundtrip
    writer(out_dir / "manifest.json", base)
    return 0, base


def main(argv: Optional[List[str]] = None, *, runtime_factory: Callable[[], CudaRuntime] = CudaRuntime, owner_factory: Callable[[CudaRuntime, CudaRuntimeStream], CudaRuntimeMemoryOwner] = CudaRuntimeMemoryOwner, writer: Callable[[Path, Dict[str, Any]], None] = write_json_exclusive) -> int:
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
    code, result = execute_smoke(args.out_dir, runtime_factory=runtime_factory, owner_factory=owner_factory, writer=writer)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
