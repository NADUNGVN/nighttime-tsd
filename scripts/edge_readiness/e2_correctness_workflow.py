#!/usr/bin/env python3
"""Staged E2 correctness-smoke workflow with device stages disabled.

``preflight`` and ``source-artifacts`` are local read-only preparation stages.
``target-build`` and ``inference-compare`` deliberately emit a structured
NO-GO artifact until Astra separately authorizes device side effects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Keep the documented ``python scripts/edge_readiness/...`` entrypoint usable
# without requiring an installation or mutating the caller's environment.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge_readiness.e2_source_fixture import verify_fixture
from edge_readiness.jetson_adapter import YOLO11N_CHECKPOINT_SHA256, YOLO11N_CHECKPOINT_PATH

CONFIG_PATH = Path("configs/deployment/yolo11n_fp16_trt10.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def run_stage(stage: str, repo_root: Path, source_root: Path, out_dir: Path) -> dict:
    if out_dir.exists():
        raise ValueError(f"refusing existing workflow output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    base = {"schema_version": "e2l1-e2-correctness-workflow-v1", "stage": stage, "target": "E2", "created_at_utc": datetime.now(timezone.utc).isoformat(), "real_device_execution": False, "model_load_or_forward": False}
    checkpoint = source_root / YOLO11N_CHECKPOINT_PATH
    config = repo_root / CONFIG_PATH
    if stage == "preflight":
        if not checkpoint.is_file():
            result = {**base, "status": "blocked", "reason": "CHECKPOINT_MISSING", "checkpoint": str(checkpoint)}
        elif file_sha256(checkpoint) != YOLO11N_CHECKPOINT_SHA256:
            result = {**base, "status": "blocked", "reason": "CHECKPOINT_HASH_MISMATCH", "observed": file_sha256(checkpoint), "expected": YOLO11N_CHECKPOINT_SHA256}
        elif not config.is_file():
            result = {**base, "status": "blocked", "reason": "DEPLOYMENT_CONFIG_MISSING", "config": str(config)}
        else:
            result = {**base, "status": "preflight_verified_read_only", "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)}, "deployment_config": {"path": str(config), "sha256": file_sha256(config)}, "required_checks": ["target identity/runtime/API", "target disk and scoped output absence", "accepted CCTSDB fixture hashes", "target-native engine hash/bindings", "final stream/event completion"], "disabled_stages": ["target-build", "inference-compare"]}
    elif stage == "source-artifacts":
        fixture = verify_fixture(source_root)
        result = {**base, "status": "source_artifacts_verified_read_only", "fixture": fixture, "writes": ["fixture_manifest.json only; no image/model bytes"]}
        write_once(out_dir / "fixture_manifest.json", fixture)
    elif stage in {"target-build", "inference-compare"}:
        result = {**base, "status": "blocked_not_authorized", "reason": "ASTRA_AUTHORIZATION_REQUIRED", "disabled_side_effects": ["model transfer", "TensorRT export/build", "CUDA context/allocation", "engine deserialize", "model load/forward", "scored timing", "power measurement", "SSH device execution"], "required_inputs": ["preflight_verified_read_only", "source_artifacts_verified_read_only", "target-local E2 TensorRT 8.5.2.2 API check", "target-native engine and hash", "accepted source-reference output"], "stage_entrypoint": "not enabled in E2L1-008"}
    else:
        raise ValueError(f"unknown stage: {stage}")
    write_once(out_dir / "stage_manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Staged E2 correctness workflow; device stages are disabled")
    parser.add_argument("--stage", choices=("preflight", "source-artifacts", "target-build", "inference-compare"), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_stage(args.stage, args.repo_root, args.source_root, args.out_dir)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=False)
        result = {"schema_version": "e2l1-e2-correctness-workflow-v1", "stage": args.stage, "status": "failed", "real_device_execution": False, "error": {"code": type(exc).__name__, "message": str(exc)}}
        write_once(args.out_dir / "failure.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] in {"preflight_verified_read_only", "source_artifacts_verified_read_only"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
