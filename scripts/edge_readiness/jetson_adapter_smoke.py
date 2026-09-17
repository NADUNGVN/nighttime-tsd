#!/usr/bin/env python3
"""Executable local CPU/mock dry-run for the proposed Jetson smoke protocol.

This entrypoint intentionally has no device mode. It exercises the adapter
through the accepted measurement harness and writes only mock artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from edge_readiness.jetson_adapter import (  # noqa: E402
    AdapterHarnessBridge,
    AdapterConfig,
    AdapterError,
    DeviceTensor,
    EngineBinding,
    EngineDescriptor,
    JETSON_PROFILES,
    JetsonRuntimeAdapter,
    MockBuffers,
    MockContext,
    MockStream,
    build_smoke_plan,
    fixture_manifest,
    fixture_to_input_bytes,
    make_train_only_fixture,
    validate_yolo11n_native_output_contract,
)
from edge_readiness.measurement_harness import (  # noqa: E402
    Boundary,
    HarnessError,
    MeasurementConfig,
    Preprocessed,
    mock_device,
    run_session,
    write_json_no_overwrite,
)


def _mock_engine(runtime_version: str) -> EngineDescriptor:
    return EngineDescriptor(
        runtime_version=runtime_version,
        engine_sha256="b" * 64,
        bindings=(
            EngineBinding("images", "input", (1, 3, 640, 640), "float32"),
            EngineBinding("output0", "output", (1, 7, 8400), "float32"),
        ),
    )


def _adapter(target_id: str, events: list[str]) -> tuple[JetsonRuntimeAdapter, AdapterHarnessBridge]:
    profile = JETSON_PROFILES[target_id]
    engine = _mock_engine(profile.expected_runtime_prefix + ".mock")
    config = AdapterConfig(target_id, profile.expected_runtime_prefix + ".mock")
    validate_yolo11n_native_output_contract(engine, config)
    stream = MockStream(handle=91, events=events)
    allocations = {
        "images": DeviceTensor("images", (1, 3, 640, 640), "float32", 1 * 3 * 640 * 640 * 4, 0x7F000, "dry-run-buffer-owner", "dry-run-lifetime-001"),
        "output0": DeviceTensor("output0", (1, 7, 8400), "float32", 1 * 7 * 8400 * 4, 0x12000, "dry-run-buffer-owner", "dry-run-lifetime-001"),
    }
    buffers = MockBuffers(engine, events, allocations=allocations, stream_handle=91)
    adapter = JetsonRuntimeAdapter(config, engine, MockContext(events), stream, buffers)
    return adapter, AdapterHarnessBridge(adapter)


def run_cpu_mock_dry_run(out_dir: Path, *, target_id: str = "E2") -> dict[str, Any]:
    """Run two tiny harness sessions, one for each declared timing boundary."""
    if target_id not in JETSON_PROFILES:
        raise AdapterError("UNKNOWN_TARGET", "dry-run target must be E2, E3 or E5")
    images = make_train_only_fixture()
    pool_ids = tuple(image.image_id for image in images)
    pool_hash = hashlib.sha256("\n".join(pool_ids).encode("utf-8")).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": "e2l1-jetson-adapter-smoke-v1",
        "status": "completed_mock_dry_run",
        "mock_only": True,
        "real_device_execution": False,
        "target_id": target_id,
        "plan": build_smoke_plan(target_id),
        "fixture": fixture_manifest(images),
        "boundaries": {},
        "failure_artifact": "failure.json (written only if this orchestration fails)",
    }
    for boundary in (Boundary.INFERENCE_ONLY, Boundary.DECODED_IMAGE_TO_DETECTIONS):
        events: list[str] = []
        adapter, bridge = _adapter(target_id, events)
        config = MeasurementConfig(
            target_id=target_id,
            backend=bridge.backend_name,
            model_id=adapter.config.model_id,
            model_sha256=adapter.config.checkpoint_sha256,
            boundary=boundary,
            warmup_calls=1,
            measured_calls=3,
            session_count=1,
            pool_ids=pool_ids,
            pool_hash=pool_hash,
            pool_order=(0, 1, 2),
        )

        def image_provider(index: int):
            return images[index]

        def preprocess(image):
            return Preprocessed(fixture_to_input_bytes(image), (1, 3, 640, 640))

        def postprocess(result):
            return {"native_output_contract": result.output_contract, "nms_applied": False}

        session = run_session(config, bridge, image_provider, preprocess, postprocess, session_index=1, device=mock_device(target_id))
        session["adapter_events"] = events
        session["output_contract"] = validate_yolo11n_native_output_contract(adapter.engine, adapter.config)
        name = boundary.value
        session_path = out_dir / name / "session_1.json"
        write_json_no_overwrite(session_path, session)
        write_json_no_overwrite(out_dir / name / "adapter_events.json", {"mock_only": True, "events": events})
        manifest["boundaries"][name] = {"session": str(session_path), "mock_only": True, "warmup_excluded": True, "measured_calls": 3}

    write_json_no_overwrite(out_dir / "input_fixture_manifest.json", manifest["fixture"])
    write_json_no_overwrite(out_dir / "output_contract.json", {"name": "output0", "shape": [1, 7, 8400], "comparison_stage": "raw_tensor_before_decode_or_NMS"})
    write_json_no_overwrite(out_dir / "manifest.json", manifest)
    return manifest


def _error_payload(exc: Exception) -> dict[str, Any]:
    if hasattr(exc, "as_dict"):
        return exc.as_dict()  # type: ignore[no-any-return]
    return {"status": "error", "error": {"code": "MOCK_ORCHESTRATION_FAILED", "message": str(exc), "details": {}}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local CPU/mock Jetson adapter smoke; no device mode exists")
    parser.add_argument("--target", choices=tuple(JETSON_PROFILES), default="E2")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", required=True, help="required acknowledgement that only CPU/mock execution is available")
    args = parser.parse_args(argv)
    try:
        result = run_cpu_mock_dry_run(args.out_dir, target_id=args.target)
    except (AdapterError, HarnessError, OSError, ValueError) as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        failure = {"schema_version": "e2l1-jetson-adapter-smoke-v1", "status": "failed_mock", "mock_only": True, "real_device_execution": False, "target_id": args.target, "error": _error_payload(exc)}
        write_json_no_overwrite(args.out_dir / "failure.json", failure)
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        return 2
    print(json.dumps({"status": result["status"], "mock_only": True, "output_dir": str(args.out_dir)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
