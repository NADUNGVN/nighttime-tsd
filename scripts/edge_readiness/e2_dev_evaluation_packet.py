#!/usr/bin/env python3
"""CPU/mock readiness packet for a future full-development E2 evaluation.

This module is intentionally incapable of loading TensorRT, CUDA, an ONNX
model, or an engine.  It validates the frozen study contract and exercises the
parent lifecycle with an injected runtime double.  A later, separately
approved runner can implement the runtime adapter behind the same boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

OUTPUT_ELEMENTS = 7 * 8400
OUTPUT_BYTES = OUTPUT_ELEMENTS * 4
DEV_IMAGES = 1636
DEV_INSTANCES = 2706
SOURCE_MANIFEST_SHA256 = "60744680973a73d2986bdf59ce3c6bc956119aeeebbd2106960df57947665c08"
SOURCE_ONNX_SHA256 = "bd20b36d640c502358c44edbbde51f05267eda2518b0c0a4cfe84ca18a00d4b7"
TARGET_MANIFEST_SHA256 = "29954570c5b1f44af928e283ce2ab89885c224b165ab85b4f69de6d5ae6a4e74"
ENGINE_SHA256 = "581a9ea2eafdae690f25f57ab88ba1bcffdafa99e5322ea50ee43678520e7f56"
ENGINE_BYTES = 7158097
DATASET_MANIFEST_COMMIT = "3154b7ad2308ff8802ea1c15532951c86ed66a96"
DATASET_MANIFEST_SHA256 = "5d1b6f1c6df475efe14c8bc6e41c6312b5121dcb828041ec81bbcb2733ac0a2e"
ULTRALYTICS_VERSION = "8.4.102"
EVALUATOR_ID = "coco_xml_paired_image_bootstrap_v1"


class PacketError(ValueError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def study_contract() -> Dict[str, Any]:
    return {
        "schema_version": "e2l1-dev-evaluation-packet-v1",
        "status": "prepared_not_executed",
        "terminal_status": "edge_dev_packet_implementation_review_required",
        "scope": {"model": "YOLO11n", "target": "E2", "split": "CCTSDB2021/dev", "images": DEV_IMAGES, "instances": DEV_INSTANCES, "official_test_used": False},
        "immutable_inputs": {
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_onnx_sha256": SOURCE_ONNX_SHA256,
            "target_manifest_sha256": TARGET_MANIFEST_SHA256,
            "engine_sha256": ENGINE_SHA256,
            "engine_bytes": ENGINE_BYTES,
            "engine_storage": "private E2 path only; never download/load locally",
            "dataset_manifest": "results/paper_artifacts/v1/metadata/dataset_manifest.json",
            "dataset_manifest_commit": DATASET_MANIFEST_COMMIT,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "dev_yaml": "results/measurement_audit_v1/server_fp16_capture_v1/dev_absolute.yaml",
            "dev_yaml_sha256": "ea8f40ba75920b2a67c2b5e1fd18c3d2eaf760686f11d0efea9bb7f26d5f5479",
        },
        "runtime_contract": {"existing_engine_only": True, "runtime": "E2 original TRT 8.5.2.2", "imgsz": 640, "batch": 1, "conf": 0.001, "iou": 0.7, "max_det": 300, "workers": 0, "rect": False, "warmup": 0, "debug_forwards": 0, "latency_benchmark": False, "energy_benchmark": False, "duplicate_nms": False, "postprocess_helper": {"package": "ultralytics", "version": ULTRALYTICS_VERSION, "symbol": "ultralytics.utils.ops.non_max_suppression", "device": "cpu/reference and E2 target-owned postprocess", "return_idxs": True}},
        "call_budget": {"target_image_passes": DEV_IMAGES, "target_passes_per_image": 1, "reference_image_passes_if_reuse_not_verified": DEV_IMAGES, "reference_passes_if_reuse_verified": 0, "warmup_calls": 0, "shape_probe_calls": 0, "debug_calls": 0, "automatic_retries": 0, "builder_invocations": 0},
        "endpoints": {"primary": ["paired_source_onnx_vs_e2_fp16_COCO_bbox_AP50", "paired_source_onnx_vs_e2_fp16_COCO_bbox_AP50_95"], "secondary": ["XS/S_AP50", "XS/S_AP50_95", "per_class_counts", "coordinate_and_threshold_diagnostics"], "bootstrap": {"estimator": EVALUATOR_ID, "seed": 20260916, "resamples": 1000, "paired_image_sampling": True, "noninferiority_margin": None}},
        "resource_requirements": {"fresh_e2_output_root": True, "private_output_storage_gib_min": 2, "available_memory_gib_min": 4, "per_image_target_output_bytes": OUTPUT_BYTES, "timeout_per_image_seconds": 30, "stage_timeout_seconds": 54000, "streaming_required": True, "retain_raw_tensors": "private optional; public hashes/metrics only"},
        "ownership": {"server_operator": ["materialize/verify source ONNX, dev images/labels/XML, input mapping and reference predictions when reusable", "publish hashes and private transfer package"], "luna1_operator": ["validate package on local CPU", "after later integrated GO, transfer only approved package to E2", "run one target pass per image with existing engine", "audit counters/hashes and publish sanitized metrics"], "astra_reviewer": ["review packet and prerequisites", "approve or reject later target execution", "review final paired metrics"]},
        "forbidden_now": ["SSH", "transfer", "source model forward", "ONNX export", "TensorRT build", "E2 inference", "benchmark", "install", "device configuration"],
    }


def validate_study_contract(contract: Mapping[str, Any]) -> Dict[str, Any]:
    if contract.get("schema_version") != "e2l1-dev-evaluation-packet-v1":
        raise PacketError("PACKET_SCHEMA_MISMATCH")
    if contract.get("status") != "prepared_not_executed":
        raise PacketError("PACKET_STATUS_INVALID")
    scope = contract.get("scope", {})
    if scope.get("images") != DEV_IMAGES or scope.get("instances") != DEV_INSTANCES or scope.get("split") != "CCTSDB2021/dev":
        raise PacketError("DEV_DATASET_CONTRACT_MISMATCH")
    budget = contract.get("call_budget", {})
    if budget.get("target_image_passes") != DEV_IMAGES or budget.get("target_passes_per_image") != 1 or budget.get("automatic_retries") != 0 or budget.get("builder_invocations") != 0:
        raise PacketError("CALL_BUDGET_INVALID")
    forbidden = contract.get("forbidden_now", [])
    if any(item not in forbidden for item in ("E2 inference", "TensorRT build", "benchmark")):
        raise PacketError("CURRENT_EXECUTION_GUARD_MISSING")
    if contract.get("endpoints", {}).get("bootstrap", {}).get("noninferiority_margin") is not None:
        raise PacketError("UNAUTHORIZED_NONINFERIORITY_MARGIN")
    return {"status": "validated", "target_image_passes": DEV_IMAGES, "reference_passes": budget.get("reference_passes_if_reuse_not_verified")}


def validate_package_inventory(inventory: Mapping[str, Any]) -> Dict[str, Any]:
    required = {"source_manifest_sha256": SOURCE_MANIFEST_SHA256, "source_onnx_sha256": SOURCE_ONNX_SHA256, "target_manifest_sha256": TARGET_MANIFEST_SHA256, "engine_sha256": ENGINE_SHA256, "engine_bytes": ENGINE_BYTES}
    for key, expected in required.items():
        if inventory.get(key) != expected:
            raise PacketError("PACKAGE_BINDING_MISMATCH: {}".format(key))
    if inventory.get("engine_public") is True or inventory.get("raw_tensors_public") is True:
        raise PacketError("PRIVATE_ARTIFACT_LEAK_POLICY")
    return {"status": "package_binding_verified", "engine_private": True, "raw_tensors_private": True}


def validate_output_payload(payload: bytes) -> None:
    if len(payload) != OUTPUT_BYTES:
        raise PacketError("OUTPUT_SHAPE_MISMATCH")
    values = struct.unpack("<{}f".format(OUTPUT_ELEMENTS), payload)
    if any(not math.isfinite(value) for value in values):
        raise PacketError("OUTPUT_NONFINITE")


class MockRuntime:
    """External runtime double used by tests; no model or CUDA calls."""

    def __init__(self, outputs: Mapping[str, bytes], fail_image: Optional[str] = None, timeout_image: Optional[str] = None, cleanup_error: bool = False):
        self.outputs = dict(outputs)
        self.fail_image = fail_image
        self.timeout_image = timeout_image
        self.cleanup_error = cleanup_error
        self.loaded = False
        self.target_calls = 0
        self.closed = False

    def load_engine(self) -> None:
        self.loaded = True

    def enqueue(self, image_id: str, input_bytes: bytes) -> bytes:
        if not self.loaded:
            raise PacketError("ENGINE_NOT_LOADED")
        self.target_calls += 1
        if image_id == self.timeout_image:
            raise TimeoutError("mock target timeout")
        if image_id == self.fail_image:
            raise RuntimeError("mock target failure")
        return self.outputs[image_id]

    def copy_output(self, payload: bytes) -> bytes:
        return bytes(payload)

    def close(self) -> None:
        self.closed = True
        if self.cleanup_error:
            raise RuntimeError("mock cleanup failure")


def mock_evaluator(image_id: str, output: bytes) -> Dict[str, Any]:
    validate_output_payload(output)
    return {"image_id": image_id, "output_sha256": sha256_bytes(output), "output_bytes": len(output), "metric_status": "synthetic_contract_only", "ap_available": False}


def run_mock_study(image_ids: Sequence[str], inputs: Mapping[str, bytes], runtime: MockRuntime, out_dir: Path, evaluator: Callable[[str, bytes], Dict[str, Any]] = mock_evaluator) -> Dict[str, Any]:
    if out_dir.exists():
        raise PacketError("OUTPUT_EXISTS")
    out_dir.mkdir(parents=True)
    events: List[Dict[str, Any]] = []
    counters = {"engine_load_attempted": 0, "engine_load_completed": 0, "target_calls_attempted": 0, "target_calls_completed": 0, "output_copies_completed": 0, "analysis_attempted": 0, "analysis_completed": 0, "unknown_completions": []}
    analyses: List[Dict[str, Any]] = []
    error: Optional[Dict[str, Any]] = None
    events.append({"event": "dispatch", "image_count": len(image_ids), "automatic_retries": 0})
    try:
        counters["engine_load_attempted"] += 1
        events.append({"event": "engine_load_attempted"})
        runtime.load_engine()
        counters["engine_load_completed"] += 1
        events.append({"event": "engine_load_completed"})
        for image_id in image_ids:
            if image_id not in inputs:
                raise PacketError("INPUT_BINDING_MISSING: {}".format(image_id))
            counters["target_calls_attempted"] += 1
            events.append({"event": "target_call_attempted", "image_id": image_id})
            try:
                payload = runtime.enqueue(image_id, inputs[image_id])
            except TimeoutError as exc:
                counters["unknown_completions"].append("target_call:{}".format(image_id))
                raise PacketError("STAGE_TIMEOUT: {}".format(image_id)) from exc
            counters["target_calls_completed"] += 1
            events.append({"event": "target_call_completed", "image_id": image_id})
            copied = runtime.copy_output(payload)
            validate_output_payload(copied)
            counters["output_copies_completed"] += 1
            counters["analysis_attempted"] += 1
            row = evaluator(image_id, copied)
            counters["analysis_completed"] += 1
            analyses.append(row)
            events.append({"event": "analysis_completed", "image_id": image_id, "output_sha256": row.get("output_sha256")})
    except PacketError as exc:
        error = {"code": str(exc).split(":", 1)[0], "message": str(exc)}
    except Exception as exc:
        error = {"code": "RUNTIME_FAILURE", "message": str(exc)}
    finally:
        try:
            runtime.close()
            events.append({"event": "cleanup_completed"})
        except Exception as exc:
            cleanup = {"code": "CLEANUP_FAILURE", "message": str(exc)}
            events.append({"event": "cleanup_failed", "error": cleanup})
            error = error or cleanup
    result = {"schema_version": "e2l1-dev-mock-run-v1", "status": "complete" if error is None else "failed_partial", "counters": counters, "analyses": analyses, "error": error, "no_retry": True, "target_calls_observed": runtime.target_calls}
    (out_dir / "events.jsonl").write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8", newline="\n")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return result


def packet_files(out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=False)
    contract = study_contract()
    validate_study_contract(contract)
    (out_dir / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    runbook = """# ST-EDGE-03 prospective full-dev E2 packet\n\nStatus: `prepared_not_executed`; current terminal status: `edge_dev_packet_implementation_review_required`.\n\n## Later operator chain (not executable under E2L1-024)\n\n1. Server operator verifies the canonical dev inventory at commit `3154b7ad2308ff8802ea1c15532951c86ed66a96`, exactly 1636 images/2706 instances, XML and image membership, and the bound source ONNX `bd20b36d640c502358c44edbbde51f05267eda2518b0c0a4cfe84ca18a00d4b7`. If the existing reference predictions do not bind to this exact ONNX, preprocessing, postprocess and dataset, materialize a separately counted CPU reference capture; do not substitute another ONNX.\n2. Server operator creates a private manifest containing source/input/reference hashes, the dev YAML hash, evaluator/helper versions and one-to-one image mapping. Public handoff contains hashes/metadata only; no engine, ONNX, input tensors or raw predictions.\n3. Luna1 validates the package in a fresh local root with `validate_package_inventory`, confirms no local engine load, and waits for a later integrated GO.\n4. After that GO only, Luna1 transfers the approved private package to a fresh E2 root, verifies `ENGINE_SHA256` and `SOURCE_ONNX_SHA256` on E2, and runs exactly one target pass per image (1636 calls) through the existing E2 engine. No warmup, shape probe, debug forward, retry, rebuild, latency or energy collection.\n5. Luna1 writes durable per-image counters/events and hashes, then invokes the accepted `coco_xml_paired_image_bootstrap_v1` evaluator on CPU with conf=0.001, IoU=0.7, max_det=300 and seed 20260916/1000 paired image resamples.\n6. A reviewer audits source/target identities, call counts, membership, postprocess ownership, AP50/AP50-95 plus XS/S and per-class diagnostics. No noninferiority margin or deployment claim is invented.\n\n## Ownership and stop conditions\n\nThe server owns reference/source materialization; Luna1 owns E2 execution and evidence audit; Astra owns the later GO and review. Missing source identity, dataset/XML membership, target engine hash, fresh root, or full call accounting is a blocker. Timeout/failure stops the run with partial evidence and no automatic retry. Existing engine unavailability is a new decision, not build permission.\n\nThis packet is an accuracy study design, not a benchmark. It does not authorize SSH, transfer, source forward, export, build, inference or benchmark now.\n"""
    (out_dir / "runbook.md").write_text(runbook, encoding="utf-8", newline="\n")
    (out_dir / "index.json").write_text(json.dumps({"schema_version": "e2l1-dev-evaluation-packet-v1", "status": contract["status"], "public_artifacts": ["contract.json", "runbook.md", "index.json"], "private_policy": "no engine/ONNX/input/raw prediction bytes"}, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return {"contract": out_dir / "contract.json", "runbook": out_dir / "runbook.md", "index": out_dir / "index.json"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Write the CPU/mock-only prospective E2 development evaluation packet")
    parser.add_argument("--write-packet", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.write_packet:
        parser.error("--write-packet is required; this module cannot execute E2")
    paths = packet_files(args.out_dir)
    print(json.dumps({key: str(value.resolve()) for key, value in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
