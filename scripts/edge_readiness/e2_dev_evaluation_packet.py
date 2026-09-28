#!/usr/bin/env python3
"""ST-EDGE-03 CPU/mock implementation boundary.

This module is intentionally incapable of loading TensorRT, CUDA, an ONNX
model, or an engine. It validates the frozen study contract and exercises the
existing runtime boundary with an external double. It also persists child
events and computes synthetic CPU AP/paired-bootstrap evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import random
import time
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from edge_readiness.jetson_adapter import (
        AdapterConfig, EngineBinding, EngineDescriptor, HostTensor,
        JetsonRuntimeAdapter, MockBuffers, MockContext, MockStream, expected_nbytes,
    )
except ModuleNotFoundError:  # direct CLI execution from scripts/edge_readiness
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from edge_readiness.jetson_adapter import (
        AdapterConfig, EngineBinding, EngineDescriptor, HostTensor,
        JetsonRuntimeAdapter, MockBuffers, MockContext, MockStream, expected_nbytes,
    )

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
NMS_SYMBOL = "ultralytics.utils.nms.non_max_suppression"
DEV_YAML_SHA256 = "ea8f40ba75920b2a67c2b5e1fd18c3d2eaf760686f11d0efea9bb7f26d5f5479"
SERVER_TRT_REFERENCE_PREDICTIONS_SHA256 = "5c23d0fa7d4bbf09858b2f1a4dbf45c35650c474aaedebe40d845e3c9acc410a"
CANONICAL_DEV_IMAGE_IDS_SHA256 = "14cedb5d6984d303c494a840fea9f3dc77d8f329d8f23bf086648e16ad6e4c1c"
CANONICAL_DEV_IMAGE_IDS_SOURCE = "results/measurement_audit_v1/server_fp16_capture_v1/validator_predictions.json:records[].image sorted UTF-8 newline-delimited"
INPUT_SHAPE = [1, 3, 640, 640]


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


def canonical_image_ids_digest(image_ids: Sequence[str]) -> str:
    payload = b"\n".join(str(image_id).encode("utf-8") for image_id in image_ids)
    return hashlib.new("sha256", payload).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def study_contract() -> Dict[str, Any]:
    return {
        "schema_version": "e2l1-dev-evaluation-packet-v3",
        "status": "prepared_not_executed",
        "terminal_status": "edge_dev_packet_implementation_review_required",
        "scope": {"model": "YOLO11n", "target": "E2", "split": "CCTSDB2021/dev", "images": DEV_IMAGES, "instances": DEV_INSTANCES, "official_test_used": False, "canonical_image_ids_sha256": CANONICAL_DEV_IMAGE_IDS_SHA256, "canonical_image_ids_source": CANONICAL_DEV_IMAGE_IDS_SOURCE},
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
            "dev_yaml_sha256": DEV_YAML_SHA256,
            "xml_identity_required": True,
            "source_onnx_cpu_reference_status": "pending_not_executed",
            "source_onnx_cpu_reference_producer": "scripts/edge_readiness/e2_dev_evaluation_packet.py:write_source_reference_pending",
            "server_trt_reference_predictions_sha256": SERVER_TRT_REFERENCE_PREDICTIONS_SHA256,
            "server_trt_reference_is_source_onnx": False,
        },
        "runtime_contract": {"existing_engine_only": True, "runtime": "E2 original TRT 8.5.2.2", "imgsz": 640, "batch": 1, "conf": 0.001, "iou": 0.7, "max_det": 300, "workers": 0, "rect": False, "warmup": 0, "debug_forwards": 0, "latency_benchmark": False, "energy_benchmark": False, "duplicate_nms": False, "postprocess_helper": {"package": "ultralytics", "version": ULTRALYTICS_VERSION, "symbol": NMS_SYMBOL, "device": "cpu/reference and E2 target-owned postprocess", "return_idxs": True, "input_shape": [1, 7, 8400], "input_dtype": "torch.float32"}},
        "call_budget": {"target_image_passes": DEV_IMAGES, "target_passes_per_image": 1, "reference_image_passes_if_reuse_not_verified": DEV_IMAGES, "reference_passes_if_reuse_verified": 0, "warmup_calls": 0, "shape_probe_calls": 0, "debug_calls": 0, "automatic_retries": 0, "builder_invocations": 0},
        "endpoints": {"primary": ["paired_source_onnx_vs_e2_fp16_COCO_bbox_AP50", "paired_source_onnx_vs_e2_fp16_COCO_bbox_AP50_95"], "secondary": ["XS/S_AP50", "XS/S_AP50_95", "per_class_counts", "coordinate_and_threshold_diagnostics"], "bootstrap": {"estimator": EVALUATOR_ID, "seed": 20260916, "resamples": 1000, "paired_image_sampling": True, "noninferiority_margin": None}},
        "resource_requirements": {"fresh_e2_output_root": True, "input_storage_mode": "streaming_manifest_only", "raw_input_tensor_bundle_forbidden": True, "private_output_storage_gib_min": 1, "available_memory_gib_min": 4, "per_image_target_output_bytes": OUTPUT_BYTES, "timeout_per_image_seconds": 30, "stage_timeout_seconds": 54000, "streaming_required": True, "retain_raw_tensors": "private optional; public hashes/metrics only", "bounded_input_bytes": 3 * 640 * 640 * 4},
        "ownership": {"server_operator": ["materialize/verify source ONNX, dev images/labels/XML, input mapping and reference predictions when reusable", "publish hashes and private transfer package"], "luna1_operator": ["validate package on local CPU", "after later integrated GO, transfer only approved package to E2", "run one target pass per image with existing engine", "audit counters/hashes and publish sanitized metrics"], "astra_reviewer": ["review packet and prerequisites", "approve or reject later target execution", "review final paired metrics"]},
        "forbidden_now": ["SSH", "transfer", "source model forward", "ONNX export", "TensorRT build", "E2 inference", "benchmark", "install", "device configuration"],
    }


def validate_study_contract(contract: Mapping[str, Any]) -> Dict[str, Any]:
    expected = study_contract()
    if contract.get("schema_version") != expected["schema_version"]:
        raise PacketError("PACKET_SCHEMA_MISMATCH")
    if contract.get("status") != expected["status"] or contract.get("terminal_status") != expected["terminal_status"]:
        raise PacketError("PACKET_STATUS_INVALID")
    for section in ("scope", "immutable_inputs", "runtime_contract", "call_budget", "endpoints", "resource_requirements"):
        actual = contract.get(section)
        if not isinstance(actual, Mapping):
            raise PacketError("PACKET_SECTION_MISSING:" + section)
        for key, value in expected[section].items():
            if actual.get(key) != value:
                raise PacketError("PACKET_CONTRACT_MISMATCH:" + section + "." + key)
    forbidden = contract.get("forbidden_now", [])
    if not isinstance(forbidden, list) or set(expected["forbidden_now"]) - set(forbidden):
        raise PacketError("CURRENT_EXECUTION_GUARD_MISSING")
    return {"status": "validated", "target_image_passes": DEV_IMAGES, "reference_passes": contract["call_budget"]["reference_image_passes_if_reuse_not_verified"], "strict_fields_checked": True}


def validate_package_inventory(inventory: Mapping[str, Any], package_root: Optional[Path] = None) -> Dict[str, Any]:
    required = {"source_manifest_sha256": SOURCE_MANIFEST_SHA256, "source_onnx_sha256": SOURCE_ONNX_SHA256, "target_manifest_sha256": TARGET_MANIFEST_SHA256, "engine_sha256": ENGINE_SHA256, "engine_bytes": ENGINE_BYTES}
    for key, expected in required.items():
        if inventory.get(key) != expected:
            raise PacketError("PACKAGE_BINDING_MISMATCH: {}".format(key))
    if inventory.get("engine_public") is True or inventory.get("raw_tensors_public") is True:
        raise PacketError("PRIVATE_ARTIFACT_LEAK_POLICY")
    image_ids = inventory.get("image_ids")
    if not isinstance(image_ids, list) or len(image_ids) != DEV_IMAGES or len(set(image_ids)) != DEV_IMAGES or image_ids != sorted(image_ids):
        raise PacketError("PACKAGE_IMAGE_MEMBERSHIP_MISMATCH")
    if canonical_image_ids_digest(image_ids) != CANONICAL_DEV_IMAGE_IDS_SHA256:
        raise PacketError("PACKAGE_CANONICAL_IMAGE_DIGEST_MISMATCH")
    canonical_path = Path(__file__).resolve().parents[2] / "results/measurement_audit_v1/server_fp16_capture_v1/validator_predictions.json"
    if canonical_path.is_file():
        canonical_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
        canonical_ids = sorted(record["image"] for record in canonical_payload.get("records", []))
        if image_ids != canonical_ids:
            raise PacketError("PACKAGE_CANONICAL_IMAGE_BINDING_MISMATCH")
    elif inventory.get("canonical_image_ids_sha256") != CANONICAL_DEV_IMAGE_IDS_SHA256:
        raise PacketError("PACKAGE_CANONICAL_IMAGE_BINDING_MISSING")
    if inventory.get("canonical_image_ids_sha256") != CANONICAL_DEV_IMAGE_IDS_SHA256 or inventory.get("canonical_image_ids_source") != CANONICAL_DEV_IMAGE_IDS_SOURCE:
        raise PacketError("PACKAGE_CANONICAL_IMAGE_BINDING_MISMATCH")
    identity = inventory.get("reference_identity", {})
    if not isinstance(identity, Mapping) or identity.get("source_onnx_sha256") != SOURCE_ONNX_SHA256 or identity.get("server_trt_predictions_sha256") != SERVER_TRT_REFERENCE_PREDICTIONS_SHA256 or identity.get("server_trt_is_source_onnx") is not False:
        raise PacketError("PACKAGE_REFERENCE_IDENTITY_MISMATCH")
    reference_status = identity.get("source_onnx_cpu_reference_status")
    if reference_status not in ("pending_not_executed", "verified"):
        raise PacketError("PACKAGE_SOURCE_REFERENCE_STATUS_INVALID")
    source_reference_hash = identity.get("source_onnx_cpu_predictions_sha256")
    if reference_status == "pending_not_executed" and source_reference_hash is not None:
        raise PacketError("PACKAGE_SOURCE_REFERENCE_HASH_PREMATURE")
    if reference_status == "verified" and (not isinstance(source_reference_hash, str) or len(source_reference_hash) != 64):
        raise PacketError("PACKAGE_SOURCE_REFERENCE_HASH_MISSING")
    if inventory.get("input_contract") != {"shape": INPUT_SHAPE, "dtype": "float32", "byteorder": "little", "finite": True, "bytes_per_image": 3 * 640 * 640 * 4}:
        raise PacketError("PACKAGE_INPUT_CONTRACT_MISMATCH")
    if inventory.get("input_storage_mode") != "streaming_manifest_only" or inventory.get("raw_input_tensor_bundle_forbidden") is not True:
        raise PacketError("PACKAGE_INPUT_STORAGE_POLICY")
    input_records = inventory.get("input_records")
    if not isinstance(input_records, list) or len(input_records) != DEV_IMAGES or [record.get("image_id") for record in input_records] != image_ids:
        raise PacketError("PACKAGE_INPUT_PRODUCER_MEMBERSHIP_MISMATCH")
    producer = inventory.get("input_producer")
    if not isinstance(producer, Mapping) or producer.get("mode") != "bound_image_preprocess_stream" or producer.get("preprocess") != "scripts.edge_readiness.e2_image_preprocess:preprocess_image_bytes" or producer.get("output_contract") != {"shape": INPUT_SHAPE, "dtype": "float32", "byteorder": "little", "finite": True}:
        raise PacketError("PACKAGE_INPUT_PRODUCER_BINDING_MISSING")
    image_records = producer.get("image_records")
    if not isinstance(image_records, list) or len(image_records) != DEV_IMAGES or [record.get("image_id") for record in image_records] != image_ids:
        raise PacketError("PACKAGE_IMAGE_PRODUCER_MEMBERSHIP_MISMATCH")
    for record in image_records:
        if (not isinstance(record, Mapping) or not isinstance(record.get("path"), str) or Path(record["path"]).is_absolute() or ".." in Path(record["path"]).parts or not isinstance(record.get("bytes"), int) or record.get("bytes") <= 0 or not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64 or not isinstance(record.get("orig_shape"), list) or len(record["orig_shape"]) != 2 or any(type(value) is not int or value <= 0 for value in record["orig_shape"]) or not isinstance(record.get("resized_shape", [640, 640]), list) or len(record.get("resized_shape", [640, 640])) != 2):
            raise PacketError("PACKAGE_IMAGE_PRODUCER_CONTRACT_MISMATCH")
    for record in input_records:
        if not isinstance(record, Mapping) or not isinstance(record.get("image_id"), str):
            raise PacketError("PACKAGE_INPUT_PRODUCER_CONTRACT_MISMATCH")
        if "path" in record or any(key in record for key in ("tensor_path", "payload_path")):
            raise PacketError("PACKAGE_PRECOMPUTED_INPUT_PATH_FORBIDDEN")
        if reference_status == "verified" and (record.get("bytes") != 3 * 640 * 640 * 4 or record.get("shape") != INPUT_SHAPE or record.get("dtype") != "float32" or record.get("byteorder") != "little" or record.get("finite") is not True or not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64 or not isinstance(record.get("preprocessing"), Mapping)):
            raise PacketError("PACKAGE_INPUT_FINGERPRINT_MISSING")
    xml_hash = identity.get("xml_sha256")
    if not isinstance(xml_hash, str) or len(xml_hash) != 64 or any(char not in "0123456789abcdef" for char in xml_hash.lower()):
        raise PacketError("PACKAGE_XML_IDENTITY_MISSING")
    allowed = inventory.get("allowed_files", [])
    if reference_status == "verified":
        for key in ("source_reference_file", "xml_file"):
            binding = identity.get(key)
            if not isinstance(binding, Mapping) or not isinstance(binding.get("path"), str) or binding.get("sha256") is None or len(str(binding.get("sha256"))) != 64:
                raise PacketError("PACKAGE_REFERENCE_FILE_BINDING_MISSING:" + key)
            if not any(isinstance(item, Mapping) and item.get("path") == binding["path"] and item.get("bytes") == binding.get("bytes") and item.get("sha256") == binding.get("sha256") for item in allowed):
                raise PacketError("PACKAGE_REFERENCE_FILE_NOT_ALLOWLISTED:" + key)
    if package_root is not None:
        if not isinstance(allowed, list) or len({item.get("path") for item in allowed if isinstance(item, Mapping)}) != len(allowed):
            raise PacketError("PACKAGE_DUPLICATE_FILE_ENTRY")
        root = package_root.resolve(); declared = set()
        for item in allowed:
            if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                raise PacketError("PACKAGE_PATH_INVALID")
            rel = Path(item["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise PacketError("PACKAGE_PATH_INVALID")
            candidate = root / rel
            if candidate.is_symlink():
                raise PacketError("PACKAGE_FILE_MISSING_OR_SYMLINK:" + item["path"])
            path = candidate.resolve(); declared.add(rel.as_posix())
            if root not in path.parents or not path.is_file():
                raise PacketError("PACKAGE_FILE_MISSING_OR_SYMLINK:" + item["path"])
            if item.get("bytes") != path.stat().st_size or item.get("sha256") != sha256_file(path):
                raise PacketError("PACKAGE_FILE_HASH_MISMATCH:" + item["path"])
        unexpected = [path for path in root.rglob("*") if path.is_file() and not path.is_symlink() and path.relative_to(root).as_posix() not in declared]
        if unexpected:
            raise PacketError("PACKAGE_UNDECLARED_FILE")
    source_input_records = None
    source_reference_manifest_sha256 = None
    if reference_status == "verified":
        if package_root is None:
            raise PacketError("PACKAGE_ROOT_REQUIRED_FOR_REFERENCE")
        for key in ("source_reference_file", "xml_file"):
            binding = identity.get(key)
            if not isinstance(binding, Mapping) or not isinstance(binding.get("path"), str) or not isinstance(binding.get("bytes"), int) or not isinstance(binding.get("sha256"), str) or len(binding["sha256"]) != 64:
                raise PacketError("PACKAGE_REFERENCE_FILE_BINDING_MISSING:" + key)
            if not any(isinstance(item, Mapping) and item.get("path") == binding["path"] and item.get("bytes") == binding["bytes"] and item.get("sha256") == binding["sha256"] for item in allowed):
                raise PacketError("PACKAGE_REFERENCE_FILE_NOT_ALLOWLISTED:" + key)
        root = package_root.resolve()
        source_binding = identity["source_reference_file"]
        source_relative = Path(source_binding["path"])
        xml_relative = Path(identity["xml_file"]["path"])
        if source_relative.is_absolute() or ".." in source_relative.parts or xml_relative.is_absolute() or ".." in xml_relative.parts:
            raise PacketError("PACKAGE_REFERENCE_PATH_INVALID")
        source_manifest_path = (root / source_relative).resolve()
        xml_path = (root / xml_relative).resolve()
        if root not in source_manifest_path.parents or root not in xml_path.parents or not source_manifest_path.is_file() or not xml_path.is_file():
            raise PacketError("PACKAGE_REFERENCE_FILE_MISSING")
        if (source_manifest_path.stat().st_size != source_binding["bytes"] or sha256_file(source_manifest_path) != source_binding["sha256"] or
                xml_path.stat().st_size != identity["xml_file"]["bytes"] or sha256_file(xml_path) != identity["xml_file"]["sha256"] or sha256_file(xml_path) != identity["xml_sha256"]):
            raise PacketError("PACKAGE_REFERENCE_FILE_HASH_MISMATCH")
        source_validation = validate_reference_artifact(source_manifest_path, package_root=root, allowed_files=allowed)
        if source_validation.get("status") != "verified" or source_validation.get("test_only") is not False:
            raise PacketError("PACKAGE_SOURCE_REFERENCE_NOT_PRODUCTION_VERIFIED")
        if source_validation.get("predictions_sha256") != source_reference_hash or source_validation.get("image_ids") != image_ids:
            raise PacketError("PACKAGE_SOURCE_REFERENCE_IDENTITY_MISMATCH")
        source_reference_manifest_sha256 = source_validation["artifact_sha256"]
        source_input_records = source_validation["source_input_records"]
        for image_record, input_record in zip(image_records, input_records):
            source_row = source_input_records.get(image_record["image_id"])
            if (not isinstance(source_row, Mapping) or input_record.get("sha256") != source_row.get("sha256") or input_record.get("preprocessing") != source_row.get("preprocessing") or
                    image_record.get("path") != source_row.get("image_path") or image_record.get("bytes") != source_row.get("image_bytes") or image_record.get("sha256") != source_row.get("image_sha256") or image_record.get("orig_shape") != source_row.get("orig_shape")):
                raise PacketError("PACKAGE_SOURCE_INPUT_PROVENANCE_MISMATCH:" + str(image_record.get("image_id")))
            image_path = (root / Path(image_record["path"])).resolve()
            if root not in image_path.parents or not image_path.is_file() or image_path.stat().st_size != image_record["bytes"] or sha256_file(image_path) != image_record["sha256"]:
                raise PacketError("PACKAGE_IMAGE_FILE_BINDING_MISMATCH:" + str(image_record.get("image_id")))
            relative_image = image_path.relative_to(root).as_posix()
            if not any(isinstance(item, Mapping) and item.get("path") == relative_image and item.get("bytes") == image_record["bytes"] and item.get("sha256") == image_record["sha256"] for item in allowed):
                raise PacketError("PACKAGE_IMAGE_FILE_NOT_ALLOWLISTED:" + str(image_record.get("image_id")))
    return {"status": "execution_ready" if reference_status == "verified" else "metadata_pending", "engine_private": True, "raw_tensors_private": True, "files_checked": len(allowed) if package_root is not None else 0, "image_count": len(image_ids), "source_reference_status": reference_status, "source_input_records": source_input_records, "source_reference_manifest_sha256": source_reference_manifest_sha256, "streaming_producer_verified": True}


def stream_input_records(input_records: Iterable[Mapping[str, Any]], image_ids: Sequence[str], payload_loader: Callable[[Mapping[str, Any]], bytes]) -> Iterable[Tuple[str, bytes]]:
    """Bounded producer/consumer interface: one input tensor at a time."""
    expected = list(image_ids)
    seen = []
    for record in input_records:
        image_id = record.get("image_id")
        if len(seen) >= len(expected) or image_id != expected[len(seen)]:
            raise PacketError("STREAM_INPUT_ORDER_MISMATCH")
        if record.get("bytes") != 3 * 640 * 640 * 4 or record.get("shape") != INPUT_SHAPE or record.get("dtype") != "float32" or record.get("byteorder") != "little" or record.get("finite") is not True:
            raise PacketError("STREAM_INPUT_CONTRACT_MISMATCH")
        payload = payload_loader(record)
        if not isinstance(payload, bytes) or len(payload) != record["bytes"] or sha256_bytes(payload) != record.get("sha256"):
            raise PacketError("STREAM_INPUT_HASH_MISMATCH:" + str(image_id))
        if any(not math.isfinite(value) for value in struct.unpack("<{}f".format(len(payload) // 4), payload)):
            raise PacketError("STREAM_INPUT_NONFINITE:" + str(image_id))
        seen.append(image_id)
        yield image_id, payload
    if seen != expected:
        raise PacketError("STREAM_INPUT_COUNT_MISMATCH")


def stream_bound_images(image_records: Iterable[Mapping[str, Any]], image_ids: Sequence[str], image_loader: Callable[[Mapping[str, Any]], bytes], preprocess: Callable[[str, bytes], bytes]) -> Iterable[Tuple[str, bytes]]:
    """Concrete image -> preprocessing -> input-tensor producer boundary."""
    expected = list(image_ids)
    seen: List[str] = []
    for record in image_records:
        image_id = record.get("image_id")
        if len(seen) >= len(expected) or image_id != expected[len(seen)]:
            raise PacketError("STREAM_IMAGE_ORDER_MISMATCH")
        path = record.get("path")
        if not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts:
            raise PacketError("STREAM_IMAGE_PATH_INVALID:" + str(image_id))
        declared_bytes = record.get("bytes")
        declared_hash = record.get("sha256")
        if not isinstance(declared_bytes, int) or declared_bytes <= 0 or not isinstance(declared_hash, str) or len(declared_hash) != 64:
            raise PacketError("STREAM_IMAGE_BINDING_INVALID:" + str(image_id))
        image_bytes = image_loader(record)
        if not isinstance(image_bytes, bytes) or len(image_bytes) != declared_bytes or sha256_bytes(image_bytes) != declared_hash:
            raise PacketError("STREAM_IMAGE_HASH_MISMATCH:" + str(image_id))
        tensor_bytes = preprocess(image_id, image_bytes)
        if not isinstance(tensor_bytes, bytes) or len(tensor_bytes) != 3 * 640 * 640 * 4:
            raise PacketError("STREAM_PREPROCESS_CONTRACT_MISMATCH:" + str(image_id))
        values = struct.unpack("<{}f".format(len(tensor_bytes) // 4), tensor_bytes)
        if any(not math.isfinite(value) for value in values):
            raise PacketError("STREAM_PREPROCESS_NONFINITE:" + str(image_id))
        seen.append(image_id)
        yield image_id, tensor_bytes
    if seen != expected:
        raise PacketError("STREAM_IMAGE_COUNT_MISMATCH")


def validate_reference_artifact(path: Path, *, package_root: Optional[Path] = None, allowed_files: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """Validate source identity, every saved output, and every input fingerprint."""
    if not path.is_file():
        raise PacketError("SOURCE_REFERENCE_MISSING:" + str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PacketError("SOURCE_REFERENCE_INVALID:" + str(path)) from exc
    if payload.get("schema_version") != "e2l1-source-onnx-reference-v2":
        raise PacketError("SOURCE_REFERENCE_SCHEMA_INVALID")
    status = payload.get("status")
    if status not in ("pending_not_executed", "verified"):
        raise PacketError("SOURCE_REFERENCE_STATUS_INVALID")
    if payload.get("source_onnx_sha256") != SOURCE_ONNX_SHA256:
        raise PacketError("SOURCE_REFERENCE_ONNX_MISMATCH")
    if status == "pending_not_executed" and payload.get("executed_image_passes") != 0:
        raise PacketError("SOURCE_REFERENCE_PENDING_COUNTER_INVALID")
    source_inputs: Dict[str, Dict[str, Any]] = {}
    if status == "verified":
        test_only = payload.get("test_only") is True
        image_ids = payload.get("image_ids")
        if not isinstance(image_ids, list) or len(image_ids) != len(set(image_ids)) or image_ids != [row.get("image_id") for row in payload.get("records", [])]:
            raise PacketError("SOURCE_REFERENCE_IMAGE_MEMBERSHIP_INVALID")
        if not test_only and (len(image_ids) != DEV_IMAGES or canonical_image_ids_digest(image_ids) != CANONICAL_DEV_IMAGE_IDS_SHA256):
            raise PacketError("SOURCE_REFERENCE_CANONICAL_SCOPE_MISMATCH")
        counters = payload.get("attempted_completed", {})
        if (payload.get("executed_image_passes") != len(image_ids) or payload.get("model_forward") is not True or
                counters.get("source_onnx_forwards_attempted") != len(image_ids) or
                counters.get("source_onnx_forwards_completed") != len(image_ids) or
                counters.get("native_forwards_attempted") != 0 or counters.get("native_forwards_completed") != 0):
            raise PacketError("SOURCE_REFERENCE_EXECUTION_INCOMPLETE")
        if payload.get("onnx_sha256_before") != SOURCE_ONNX_SHA256 or payload.get("onnx_sha256_after") != SOURCE_ONNX_SHA256 or payload.get("onnx_sha256") != SOURCE_ONNX_SHA256:
            raise PacketError("SOURCE_REFERENCE_ONNX_AFTER_HASH_MISMATCH")
        if payload.get("ort_session_providers") != ["CPUExecutionProvider"]:
            raise PacketError("SOURCE_REFERENCE_ORT_PROVIDER_MISMATCH")
        cleanup = payload.get("cleanup", {})
        if not isinstance(cleanup, Mapping) or cleanup.get("status") not in ("completed", "not_required"):
            raise PacketError("SOURCE_REFERENCE_CLEANUP_UNKNOWN")
        if not isinstance(payload.get("predictions_path"), str) or not isinstance(payload.get("predictions_sha256"), str) or len(payload["predictions_sha256"]) != 64:
            raise PacketError("SOURCE_REFERENCE_OUTPUT_BINDING_MISSING")
        bundle_root = path.parent.resolve()
        relative = Path(payload["predictions_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PacketError("SOURCE_REFERENCE_OUTPUT_PATH_INVALID")
        output = (bundle_root / relative).resolve()
        if bundle_root not in output.parents or not output.is_file() or sha256_file(output) != payload["predictions_sha256"]:
            raise PacketError("SOURCE_REFERENCE_OUTPUT_HASH_MISMATCH")
        try:
            inventory = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PacketError("SOURCE_REFERENCE_OUTPUT_INVENTORY_INVALID") from exc
        raw_rows = inventory.get("records")
        if not isinstance(raw_rows, list) or [row.get("image_id") for row in raw_rows] != image_ids:
            raise PacketError("SOURCE_REFERENCE_OUTPUT_MEMBERSHIP_MISMATCH")
        allowed = list(allowed_files) if allowed_files is not None else None
        root = package_root.resolve() if package_root is not None else None
        for row, raw_row in zip(payload["records"], raw_rows):
            image_id = row["image_id"]
            input_hash = row.get("input_sha256")
            preprocessing = row.get("preprocessing")
            if (not isinstance(input_hash, str) or len(input_hash) != 64 or
                    row.get("input_bytes") != 3 * 640 * 640 * 4 or row.get("input_shape") != INPUT_SHAPE or
                    row.get("input_dtype") != "float32" or row.get("input_byteorder") != "little" or
                    not isinstance(preprocessing, Mapping) or not isinstance(preprocessing.get("letterbox_transform"), Mapping) or
                    preprocessing.get("input_sha256") != input_hash or preprocessing.get("input_bytes") != row.get("input_bytes") or
                    not isinstance(preprocessing.get("original_shape"), list) or
                    list(preprocessing.get("original_shape", [])[:2]) != row.get("orig_shape") or
                    not isinstance(row.get("image_sha256"), str) or len(row["image_sha256"]) != 64 or
                    not isinstance(row.get("image_bytes"), int) or row["image_bytes"] <= 0 or
                    not isinstance(row.get("image_path"), str) or not row["image_path"] or
                    not isinstance(row.get("orig_shape"), list) or len(row["orig_shape"]) != 2 or
                    any(type(value) is not int or value <= 0 for value in row["orig_shape"])):
                raise PacketError("SOURCE_REFERENCE_INPUT_BINDING_INVALID:" + str(image_id))
            for key in ("path", "bytes", "sha256", "image_path", "image_bytes", "image_sha256", "orig_shape", "input_sha256", "input_bytes", "input_shape", "input_dtype", "input_byteorder", "preprocessing"):
                if raw_row.get(key) != row.get(key):
                    raise PacketError("SOURCE_REFERENCE_OUTPUT_INVENTORY_BINDING_MISMATCH:" + str(image_id))
            raw_relative = Path(str(row.get("path", "")))
            if raw_relative.is_absolute() or not raw_relative.parts or ".." in raw_relative.parts:
                raise PacketError("SOURCE_REFERENCE_RAW_PATH_INVALID:" + str(image_id))
            raw_output = (bundle_root / raw_relative).resolve()
            if bundle_root not in raw_output.parents or not raw_output.is_file() or raw_output.stat().st_size != row.get("bytes") or sha256_file(raw_output) != row.get("sha256"):
                raise PacketError("SOURCE_REFERENCE_RAW_HASH_MISMATCH:" + str(image_id))
            if root is not None:
                if root not in raw_output.parents:
                    raise PacketError("SOURCE_REFERENCE_PACKAGE_BINDING_MISSING")
                if allowed is not None:
                    package_relative = raw_output.relative_to(root).as_posix()
                    if not any(isinstance(item, Mapping) and item.get("path") == package_relative and item.get("bytes") == raw_output.stat().st_size and item.get("sha256") == row.get("sha256") for item in allowed):
                        raise PacketError("SOURCE_REFERENCE_RAW_FILE_NOT_ALLOWLISTED:" + str(image_id))
                    output_relative = output.relative_to(root).as_posix()
                    if not any(isinstance(item, Mapping) and item.get("path") == output_relative and item.get("bytes") == output.stat().st_size and item.get("sha256") == payload["predictions_sha256"] for item in allowed):
                        raise PacketError("SOURCE_REFERENCE_INVENTORY_NOT_ALLOWLISTED")
            source_inputs[image_id] = {"sha256": input_hash, "preprocessing": preprocessing, "image_path": row["image_path"], "image_sha256": row["image_sha256"], "image_bytes": row["image_bytes"], "orig_shape": row["orig_shape"]}
        if test_only:
            status_result = "test_fixture_verified"
        else:
            status_result = "verified"
    else:
        status_result = "metadata_pending"
    return {"status": status_result, "path": str(path.resolve()), "artifact_sha256": sha256_file(path), "predictions_sha256": payload.get("predictions_sha256"), "executed_image_passes": payload.get("executed_image_passes", 0), "test_only": payload.get("test_only", False), "image_ids": payload.get("image_ids", []), "source_input_records": source_inputs}


def _letterbox_to_original(box: Sequence[float], record: Mapping[str, Any]) -> List[float]:
    shape = record.get("orig_shape")
    if not isinstance(shape, Sequence) or len(shape) != 2:
        raise PacketError("POSTPROCESS_ORIGINAL_SHAPE_MISSING:" + str(record.get("image_id", record.get("image"))))
    preprocessing = record.get("preprocessing")
    transform = preprocessing.get("letterbox_transform") if isinstance(preprocessing, Mapping) else None
    if not isinstance(transform, Mapping):
        raise PacketError("POSTPROCESS_LETTERBOX_TRANSFORM_MISSING:" + str(record.get("image_id", record.get("image"))))
    ratio, pad = transform.get("gain"), transform.get("pad")
    if (not isinstance(ratio, Sequence) or len(ratio) != 2 or
            not isinstance(pad, Sequence) or len(pad) != 2 or
            any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in list(ratio) + list(pad)) or
            any(float(value) <= 0 for value in ratio)):
        raise PacketError("POSTPROCESS_LETTERBOX_TRANSFORM_INVALID")
    try:
        import torch
        from ultralytics.utils.ops import scale_boxes
    except ImportError as exc:
        raise PacketError("POSTPROCESS_TRANSFORM_DEPENDENCY_MISSING:" + str(exc)) from exc
    boxes = torch.tensor([list(box)], dtype=torch.float32)
    scale_boxes((640, 640), boxes, tuple(int(value) for value in shape), ratio_pad=(tuple(float(value) for value in ratio), tuple(float(value) for value in pad)))
    return [float(value) for value in boxes[0].tolist()]


def postprocess_saved_outputs(raw_manifest_path: Path, raw_root: Path, image_records: Sequence[Mapping[str, Any]], out_dir: Path, *, source_label: str) -> Dict[str, Any]:
    """Convert private raw output inventory into the canonical analyzer records."""
    if out_dir.exists():
        raise PacketError("OUTPUT_EXISTS:" + str(out_dir))
    try:
        from edge_readiness.e2_output_diagnostic import resolve_accepted_nms
        import torch
    except ImportError as exc:
        raise PacketError("POSTPROCESS_DEPENDENCY_MISSING:" + str(exc)) from exc
    payload = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    rows_by_id = {row.get("image_id"): row for row in payload.get("records", [])}
    metadata = {record.get("image_id"): record for record in image_records}
    expected = [record.get("image_id") for record in image_records]
    if expected != [row.get("image_id") for row in payload.get("records", [])]:
        raise PacketError("POSTPROCESS_MEMBERSHIP_MISMATCH")
    helper, binding = resolve_accepted_nms()
    records: List[Dict[str, Any]] = []
    helper_hashes = []
    out_dir.mkdir(parents=True)
    for image_id in expected:
        row = rows_by_id[image_id]
        raw_path = raw_root / row["path"]
        if not raw_path.is_file() or raw_path.stat().st_size != OUTPUT_BYTES or sha256_file(raw_path) != row.get("sha256"):
            raise PacketError("POSTPROCESS_RAW_HASH_MISMATCH:" + str(image_id))
        raw = raw_path.read_bytes()
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(1, 7, 8400).clone()
        before = tensor.detach().cpu().contiguous().numpy().tobytes()
        helper_input = tensor.clone()
        detections, indexes = helper(helper_input, conf_thres=0.001, iou_thres=0.7, max_det=300, nc=3, return_idxs=True)
        after = tensor.detach().cpu().contiguous().numpy().tobytes()
        if before != after:
            raise PacketError("POSTPROCESS_INPUT_MUTATED:" + str(image_id))
        detection_rows = detections[0].detach().cpu().tolist() if detections else []
        index_rows = indexes[0].detach().cpu().tolist() if indexes else []
        boxes, scores, classes = [], [], []
        for detection in detection_rows:
            if len(detection) != 6:
                raise PacketError("POSTPROCESS_DETECTION_SHAPE_INVALID:" + str(image_id))
            transform_record = dict(metadata[image_id])
            transform_record["preprocessing"] = row.get("preprocessing")
            if not isinstance(row.get("input_sha256"), str) or len(row["input_sha256"]) != 64:
                raise PacketError("POSTPROCESS_INPUT_HASH_MISSING:" + str(image_id))
            boxes.append(_letterbox_to_original(detection[:4], transform_record))
            scores.append(float(detection[4]))
            classes.append(int(detection[5]))
        record = {"image": image_id, "orig_shape": list(metadata[image_id]["orig_shape"]), "xyxy": boxes, "confidence": scores, "class_id": classes, "raw_output_sha256": row["sha256"], "raw_output_bytes": row["bytes"], "input_sha256": row["input_sha256"], "preprocessing": row["preprocessing"], "source_reference_manifest_sha256": payload.get("source_reference_manifest_sha256"), "postprocess": {"binding": binding, "input_shape": [1, 7, 8400], "input_dtype": str(tensor.dtype), "input_unchanged": True, "anchor_indexes": index_rows}}
        records.append(record)
        helper_hashes.append(sha256_bytes(json.dumps(record, sort_keys=True).encode("utf-8")))
    result = {"schema_version": "e2l1-postprocessed-records-v1", "status": "complete", "source": source_label, "records": records, "postprocess_binding": binding, "record_hashes_sha256": sha256_bytes("\n".join(helper_hashes).encode("utf-8"))}
    _write_json_once(out_dir / "records.json", result)
    return result


def write_source_reference_pending(out_path: Path) -> Dict[str, Any]:
    """Publish a truthful source-reference preparation record without a forward."""
    if out_path.exists():
        raise PacketError("OUTPUT_EXISTS")
    payload = {"schema_version": "e2l1-source-onnx-reference-v2", "status": "pending_not_executed", "source_onnx_sha256": SOURCE_ONNX_SHA256, "producer": "scripts/edge_readiness/e2_dev_evaluation_packet.py:write_source_reference_pending", "planned_image_passes": DEV_IMAGES, "executed_image_passes": 0, "model_forward": False, "cpu_evaluator": EVALUATOR_ID, "provenance_note": "The SERVER TensorRT FP16 prediction hash is retained separately and is not source-ONNX evidence."}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return payload


def validate_output_payload(payload: bytes) -> None:
    if len(payload) != OUTPUT_BYTES:
        raise PacketError("OUTPUT_SHAPE_MISMATCH")
    values = struct.unpack("<{}f".format(OUTPUT_ELEMENTS), payload)
    if any(not math.isfinite(value) for value in values):
        raise PacketError("OUTPUT_NONFINITE")


@dataclass(frozen=True)
class BoxPrediction:
    image_id: str
    class_id: int
    score: float
    box: Tuple[float, float, float, float]


@dataclass(frozen=True)
class GroundTruth:
    image_id: str
    class_id: int
    box: Tuple[float, float, float, float]


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - inter
    return inter / union if union else 0.0


def _average_precision(predictions: Sequence[BoxPrediction], truths: Sequence[GroundTruth], threshold: float) -> float:
    by_key: Dict[Tuple[str, int], List[GroundTruth]] = {}
    for truth in truths:
        by_key.setdefault((truth.image_id, truth.class_id), []).append(truth)
    ordered = sorted(predictions, key=lambda item: (-item.score, item.image_id, item.class_id, item.box))
    matched: Dict[Tuple[str, int], set] = {}
    tp: List[int] = []
    fp: List[int] = []
    for pred in ordered:
        candidates = by_key.get((pred.image_id, pred.class_id), [])
        used = matched.setdefault((pred.image_id, pred.class_id), set())
        best_iou, best_index = 0.0, -1
        for index, truth in enumerate(candidates):
            overlap = _iou(pred.box, truth.box)
            if index not in used and overlap > best_iou:
                best_iou, best_index = overlap, index
        if best_iou >= threshold:
            used.add(best_index); tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    if not truths or not predictions:
        return 0.0
    recalls = [0.0]
    precisions = [1.0]
    cumulative_tp = cumulative_fp = 0
    for true_positive, false_positive in zip(tp, fp):
        cumulative_tp += true_positive; cumulative_fp += false_positive
        recalls.append(cumulative_tp / float(len(truths)))
        precisions.append(cumulative_tp / float(cumulative_tp + cumulative_fp))
    return sum(max((precision for recall, precision in zip(recalls, precisions) if recall >= index / 100.0), default=0.0) for index in range(101)) / 101.0


def evaluate_predictions(predictions: Sequence[BoxPrediction], truths: Sequence[GroundTruth], image_ids: Sequence[str]) -> Dict[str, Any]:
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise PacketError("EVALUATION_IMAGE_IDS_INVALID")
    if not truths:
        raise PacketError("EVALUATION_NO_GROUND_TRUTH")
    thresholds = [0.50 + 0.05 * index for index in range(10)]
    ap50 = _average_precision(predictions, truths, 0.50)
    ap5095 = sum(_average_precision(predictions, truths, threshold) for threshold in thresholds) / 10.0
    return {"backend": "dependency_free_coco_style_101_point", "image_count": len(image_ids), "ground_truth_count": len(truths), "prediction_count": len(predictions), "metrics": {"AP50": ap50, "AP50_95": ap5095}, "nonzero_ground_truth": True}


def paired_bootstrap(predictions: Sequence[BoxPrediction], truths: Sequence[GroundTruth], image_ids: Sequence[str], seed: int = 20260916, resamples: int = 1000) -> Dict[str, Any]:
    if resamples <= 0:
        raise PacketError("BOOTSTRAP_RESAMPLES_INVALID")
    rng = random.Random(seed)
    pred_by_image = {image_id: [item for item in predictions if item.image_id == image_id] for image_id in image_ids}
    truth_by_image = {image_id: [item for item in truths if item.image_id == image_id] for image_id in image_ids}
    values = {"AP50": [], "AP50_95": []}
    for _ in range(resamples):
        draw = [image_ids[rng.randrange(len(image_ids))] for _ in image_ids]
        drawn_predictions: List[BoxPrediction] = []
        drawn_truths: List[GroundTruth] = []
        names: List[str] = []
        for index, image_id in enumerate(draw):
            name = "{}#{}".format(image_id, index); names.append(name)
            drawn_predictions.extend(BoxPrediction(name, item.class_id, item.score, item.box) for item in pred_by_image[image_id])
            drawn_truths.extend(GroundTruth(name, item.class_id, item.box) for item in truth_by_image[image_id])
        metrics = evaluate_predictions(drawn_predictions, drawn_truths, names)["metrics"]
        for key in values:
            values[key].append(metrics[key])
    ci: Dict[str, Dict[str, float]] = {}
    for key, series in values.items():
        ordered = sorted(series)
        ci[key] = {"low_2_5": ordered[max(0, int(resamples * 0.025) - 1)], "high_97_5": ordered[min(resamples - 1, int(resamples * 0.975))]}
    return {"seed": seed, "resamples": resamples, "paired_image_sampling": True, "duplicate_occurrences_preserved": True, "ci95": ci}


def _canonical_dependencies() -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Load the locked CPU evaluator only at the analysis boundary."""
    try:
        import numpy as np
        from analyze_dev_quantization import ap_values, ci, resample_ap
        from audit_cctsdb_measurement import load_xml
        from verify_cctsdb_capture import coco_size
    except ImportError as exc:
        raise PacketError("CANONICAL_EVALUATOR_DEPENDENCY_MISSING:" + str(exc)) from exc
    return np, ap_values, ci, resample_ap, load_xml, coco_size


def _canonical_record_map(records: Sequence[Mapping[str, Any]], label: str) -> Dict[str, Mapping[str, Any]]:
    """Normalize the accepted name-keyed loader contract at this boundary."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise PacketError("CANONICAL_EVALUATOR_RECORDS_INVALID:" + label)
    mapped: Dict[str, Mapping[str, Any]] = {}
    stems = set()
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("image"), str) or not record["image"]:
            raise PacketError("CANONICAL_EVALUATOR_RECORD_INVALID:" + label)
        image = record["image"]
        stem = Path(image).stem
        if image in mapped or stem in stems:
            raise PacketError("CANONICAL_EVALUATOR_DUPLICATE_IMAGE:" + image)
        shape = record.get("orig_shape")
        if not isinstance(shape, Sequence) or len(shape) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape):
            raise PacketError("CANONICAL_EVALUATOR_SHAPE_INVALID:" + image)
        for field in ("xyxy", "confidence", "class_id"):
            if not isinstance(record.get(field), Sequence):
                raise PacketError("CANONICAL_EVALUATOR_FIELD_INVALID:" + field)
        if len(record["xyxy"]) != len(record["confidence"]) or len(record["xyxy"]) != len(record["class_id"]):
            raise PacketError("CANONICAL_EVALUATOR_ARRAY_LENGTH_MISMATCH:" + image)
        mapped[image] = record
        stems.add(stem)
    return mapped


def _json_safe(value: Any) -> Any:
    """Serialize undefined metric support as JSON null, never NaN."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def evaluate_canonical_coco_xml_pair(source_records: Sequence[Mapping[str, Any]], target_records: Sequence[Mapping[str, Any]], xml_path: Path, resamples: int = 1000, seed: int = 20260916) -> Dict[str, Any]:
    """Delegate to the repository's locked COCO/XML evaluator and paired AP.

    This is lazy by design: the local readiness environment need not install
    scientific dependencies. When called in the approved analysis environment
    it reuses ``verify_cctsdb_capture.coco_size`` and
    ``analyze_dev_quantization.resample_ap``; no alternate AP implementation is
    used for production evidence.
    """
    np, ap_values, ci, resample_ap, load_xml, coco_size = _canonical_dependencies()
    if resamples < 2 or len(source_records) != len(target_records) or not source_records:
        raise PacketError("CANONICAL_EVALUATOR_SCOPE_INVALID")
    source_map = _canonical_record_map(source_records, "source")
    target_map = _canonical_record_map(target_records, "target")
    source_ids = sorted(source_map)
    target_ids = sorted(target_map)
    if source_ids != target_ids:
        raise PacketError("CANONICAL_EVALUATOR_IMAGE_MEMBERSHIP_MISMATCH")
    if any(list(source_map[image]["orig_shape"]) != list(target_map[image]["orig_shape"]) for image in source_ids):
        raise PacketError("CANONICAL_EVALUATOR_SHAPE_MISMATCH")
    ordered_source = [source_map[image] for image in source_ids]
    ordered_target = [target_map[image] for image in target_ids]
    xml = load_xml(xml_path, source_map)
    source_report, source_evaluator = coco_size(ordered_source, xml, return_evaluator=True)
    target_report, target_evaluator = coco_size(ordered_target, xml, return_evaluator=True)
    if source_report["metric_id"] != "COCO_bbox_AP_custom_CCTSDB_area_XML_original_coordinates_v1" or target_report["metric_id"] != source_report["metric_id"]:
        raise PacketError("CANONICAL_EVALUATOR_METRIC_ID_MISMATCH")
    if source_report["rules"] != target_report["rules"] or source_report["evaluator_source_sha256"] != target_report["evaluator_source_sha256"]:
        raise PacketError("CANONICAL_EVALUATOR_RULE_BINDING_MISMATCH")
    point_source = ap_values(source_evaluator)
    point_target = ap_values(target_evaluator)
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = rng.integers(0, len(source_ids), size=(resamples, len(source_ids)))
    samples.sort(axis=1)
    draws_source = np.empty((resamples, point_source.shape[0], point_source.shape[1]))
    draws_target = np.empty_like(draws_source)
    for index, sample in enumerate(samples):
        draws_source[index] = resample_ap(source_evaluator, sample)
        draws_target[index] = resample_ap(target_evaluator, sample)
    contrasts = {}
    labels = ("all", "xs", "s", "m", "l", "xl")
    for area_index, label in enumerate(labels):
        contrasts[label] = {metric: ci(100.0 * (draws_target[:, area_index, metric_index] - draws_source[:, area_index, metric_index]), 100.0 * (point_target[area_index, metric_index] - point_source[area_index, metric_index])) for metric_index, metric in enumerate(("AP50", "AP50_95"))}
    return _json_safe({"backend": EVALUATOR_ID, "metric_id": source_report["metric_id"], "evaluator_source_sha256": source_report["evaluator_source_sha256"], "image_count": len(source_ids), "source_point": point_source.tolist(), "target_point": point_target.tolist(), "paired_target_minus_source": contrasts, "bootstrap": {"seed": seed, "resamples": resamples, "generator": "numpy.random.Generator(PCG64)", "same_sorted_image_draw": True, "duplicate_occurrences_preserved": True}})


class AdapterRuntimeDouble:
    """External-call double that still exercises the real Jetson adapter."""
    def __init__(self, cleanup_error: bool = False) -> None:
        self.mock_only = True
        self.events: List[str] = []
        config = AdapterConfig("E2", "8.5.2.2")
        engine = EngineDescriptor("8.5.2.2", ENGINE_SHA256, (EngineBinding("images", "input", tuple(INPUT_SHAPE), "float32"), EngineBinding("output0", "output", (1, 7, 8400), "float32")))
        self.stream = MockStream(handle=91, events=self.events)
        self.buffers = MockBuffers(engine, self.events, stream_handle=self.stream.handle)
        self.adapter = JetsonRuntimeAdapter(config, engine, MockContext(self.events), self.stream, self.buffers, stage_observer=self.events.append)
        self.loaded = False
        self.closed = False
        self.target_calls = 0
        self.cleanup_error = cleanup_error

    def load_engine(self) -> None:
        self.events.append("provider_load_engine_completed")
        self.loaded = True

    def enqueue(self, image_id: str, input_bytes: bytes) -> bytes:
        if not self.loaded:
            raise PacketError("ENGINE_NOT_LOADED")
        expected = expected_nbytes(tuple(INPUT_SHAPE), "float32")
        if len(input_bytes) != expected:
            raise PacketError("INPUT_CONTRACT_MISMATCH:" + image_id)
        self.target_calls += 1
        host = HostTensor("images", tuple(INPUT_SHAPE), "float32", expected, bytes(input_bytes))
        outputs = self.adapter.infer(host)
        return outputs["output0"].payload

    def copy_output(self, payload: bytes) -> bytes:
        return bytes(payload)

    def close(self) -> None:
        errors = []
        try:
            self.buffers.release()
        except Exception as exc:
            errors.append(exc)
        try:
            self.stream.close()
        except Exception as exc:
            errors.append(exc)
        self.closed = not errors
        if errors:
            raise PacketError("ADAPTER_DOUBLE_CLEANUP_FAILED") from errors[0]
        self.events.append("adapter_double_cleanup_completed")
        if self.cleanup_error:
            raise PacketError("ADAPTER_DOUBLE_CLEANUP_INJECTED")


def make_lazy_e2_provider() -> Any:
    """Construct the real lazy provider without importing TensorRT or loading an engine."""
    from edge_readiness.jetson_runtime_provider import TensorRTProvider
    return TensorRTProvider(AdapterConfig("E2", "8.5.2.2"))


@dataclass(frozen=True)
class StagePermissions:
    """Explicit execution switches; false is the safe local default."""

    allow_source_forward: bool = False
    allow_target_inference: bool = False


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise PacketError("OUTPUT_EXISTS:" + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _validate_source_scope(image_records: Sequence[Mapping[str, Any]], *, test_only: bool) -> List[str]:
    if not image_records:
        raise PacketError("SOURCE_IMAGE_RECORDS_EMPTY")
    image_ids = [str(record.get("image_id")) for record in image_records]
    if len(set(image_ids)) != len(image_ids) or any(not value or value == "None" for value in image_ids):
        raise PacketError("SOURCE_IMAGE_IDS_INVALID")
    if not test_only and (len(image_ids) != DEV_IMAGES or image_ids != sorted(image_ids) or canonical_image_ids_digest(image_ids) != CANONICAL_DEV_IMAGE_IDS_SHA256):
        raise PacketError("SOURCE_CANONICAL_SCOPE_MISMATCH")
    if test_only and len(image_ids) > 5:
        raise PacketError("SOURCE_TEST_FIXTURE_SCOPE_TOO_LARGE")
    for record in image_records:
        relative = Path(str(record.get("path", "")))
        shape = record.get("orig_shape")
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise PacketError("SOURCE_IMAGE_PATH_INVALID:" + str(record.get("image_id")))
        if (not isinstance(record.get("bytes"), int) or record["bytes"] <= 0 or
                not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64 or
                any(char not in "0123456789abcdef" for char in record["sha256"].lower()) or
                not isinstance(shape, list) or len(shape) != 2 or
                any(type(value) is not int or value <= 0 for value in shape)):
            raise PacketError("SOURCE_IMAGE_METADATA_INVALID:" + str(record.get("image_id")))
    return image_ids


def _validated_input_tensor(tensor: Any, image_id: str) -> Tuple[bytes, Dict[str, Any]]:
    payload = bytes(getattr(tensor, "payload", b""))
    if (tuple(getattr(tensor, "shape", ())) != tuple(INPUT_SHAPE) or
            getattr(tensor, "dtype", None) != "float32" or
            getattr(tensor, "byteorder", None) != "little" or
            len(payload) != 3 * 640 * 640 * 4 or
            getattr(tensor, "finite", False) is not True):
        raise PacketError("SOURCE_PREPROCESS_CONTRACT_MISMATCH:" + image_id)
    metadata = getattr(tensor, "metadata", None)
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("letterbox_transform"), Mapping):
        raise PacketError("SOURCE_PREPROCESS_METADATA_MISSING:" + image_id)
    observed_hash = sha256_bytes(payload)
    if metadata.get("input_sha256") != observed_hash or metadata.get("input_bytes") != len(payload):
        raise PacketError("SOURCE_PREPROCESS_HASH_MISMATCH:" + image_id)
    return payload, dict(metadata)


def run_source_reference_stage(image_records: Sequence[Mapping[str, Any]], source_root: Path, onnx_path: Path, out_dir: Path, *, permissions: StagePermissions, runtime: Any = None, commit: str = "unknown", expected_onnx_sha256: str = SOURCE_ONNX_SHA256, test_only: bool = False) -> Dict[str, Any]:
    """Run the pinned ONNX CPU reference through the existing ORT boundary.

    This stage deliberately does not load the native PyTorch checkpoint. It
    hashes and validates the pinned ONNX first, uses CPUExecutionProvider, and
    keeps raw output bytes private while publishing only bound inventories.
    """
    if not permissions.allow_source_forward:
        raise PacketError("SOURCE_FORWARD_NOT_AUTHORIZED")
    if out_dir.exists():
        raise PacketError("OUTPUT_EXISTS:" + str(out_dir))
    expected_ids = _validate_source_scope(image_records, test_only=test_only)
    out_dir.mkdir(parents=True)
    private = out_dir / "private"
    onnx_dir = private / "onnx_reference"
    onnx_dir.mkdir(parents=True)
    counters = {"source_preprocess_attempted": 0, "source_preprocess_completed": 0, "source_onnx_forwards_attempted": 0, "source_onnx_forwards_completed": 0, "native_forwards_attempted": 0, "native_forwards_completed": 0}
    events: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    runtime = runtime
    cleanup: Dict[str, Any] = {"status": "not_started"}
    error: Optional[Dict[str, Any]] = None
    status = "failed_partial"
    ort_session = None
    onnx_sha256_before: Optional[str] = None
    onnx_sha256_after: Optional[str] = None
    ort_session_providers: List[str] = []
    try:
        if not onnx_path.is_file():
            raise PacketError("SOURCE_ONNX_MISSING:" + str(onnx_path))
        onnx_sha256_before = sha256_file(onnx_path)
        if onnx_sha256_before != expected_onnx_sha256:
            raise PacketError("SOURCE_ONNX_HASH_MISMATCH:" + onnx_sha256_before)

        # Validate the complete source scope and each decoded image shape
        # before opening the ONNX session or issuing the first ORT call.
        from edge_readiness.e2_image_preprocess import ImagePreprocessError, inspect_image_bytes
        for record in image_records:
            image_id = str(record["image_id"])
            path = source_root / str(record["path"])
            image_bytes = path.read_bytes()
            if len(image_bytes) != record["bytes"] or sha256_bytes(image_bytes) != record["sha256"]:
                raise PacketError("SOURCE_IMAGE_HASH_MISMATCH:" + image_id)
            try:
                decoded = inspect_image_bytes(image_bytes, image_id=image_id)
            except ImagePreprocessError as exc:
                raise PacketError(str(exc)) from exc
            if decoded["orig_shape"] != record["orig_shape"]:
                raise PacketError("SOURCE_IMAGE_SHAPE_MISMATCH:" + image_id)

        if runtime is None:
            from edge_readiness.e2_source_bundle import UltralyticsSourceRuntime
            runtime = UltralyticsSourceRuntime()
        environment = runtime.prepare()
        onnx_contract = runtime.validate_onnx(onnx_path)
        ort_session = runtime.open_ort(onnx_path)
        if list(environment.get("ort_providers_required", [])) != ["CPUExecutionProvider"]:
            raise PacketError("SOURCE_ORT_PROVIDER_POLICY_MISMATCH")
        if not hasattr(ort_session, "get_providers"):
            raise PacketError("SOURCE_ORT_SESSION_PROVIDER_UNAVAILABLE")
        ort_session_providers = list(ort_session.get_providers())
        if ort_session_providers != ["CPUExecutionProvider"]:
            raise PacketError("SOURCE_ORT_SESSION_PROVIDER_MISMATCH")
        for record in image_records:
            image_id = str(record["image_id"])
            image_path = source_root / str(record["path"])
            image_bytes = image_path.read_bytes()
            if len(image_bytes) != record["bytes"] or sha256_bytes(image_bytes) != record["sha256"]:
                raise PacketError("SOURCE_IMAGE_CHANGED_AFTER_PREFLIGHT:" + image_id)
            counters["source_preprocess_attempted"] += 1
            if hasattr(runtime, "preprocess_bytes"):
                tensor = runtime.preprocess_bytes(image_bytes, image_id=image_id)
            elif hasattr(runtime, "preprocess"):
                tensor = runtime.preprocess(image_path)
            else:
                from edge_readiness.e2_image_preprocess import preprocess_image_bytes
                tensor = preprocess_image_bytes(image_bytes, image_id=image_id)
            input_payload, preprocessing = _validated_input_tensor(tensor, image_id)
            if preprocessing.get("original_shape", [])[:2] != record["orig_shape"]:
                raise PacketError("SOURCE_PREPROCESS_SHAPE_BINDING_MISMATCH:" + image_id)
            counters["source_preprocess_completed"] += 1
            input_hash = sha256_bytes(input_payload)
            events.append({"event": "source_input_bound", "image_id": image_id, "input_sha256": input_hash, "preprocessing": preprocessing})
            counters["source_onnx_forwards_attempted"] += 1
            events.append({"event": "source_onnx_forward_attempted", "image_id": image_id})
            output = runtime.ort_forward(ort_session, tensor)
            payload = bytes(output.payload)
            validate_output_payload(payload)
            output_path = onnx_dir / (str(image_id) + ".bin")
            output_path.write_bytes(payload)
            rows.append({"image_id": image_id, "image_path": str(record["path"]), "image_bytes": record["bytes"], "image_sha256": record["sha256"], "orig_shape": record["orig_shape"], "input_sha256": input_hash, "input_bytes": len(input_payload), "input_shape": list(INPUT_SHAPE), "input_dtype": "float32", "input_byteorder": "little", "preprocessing": preprocessing, "path": str(output_path.relative_to(out_dir).as_posix()), "bytes": len(payload), "sha256": sha256_bytes(payload), "shape": list(getattr(output, "shape", (1, 7, 8400))), "dtype": getattr(output, "dtype", "float32")})
            counters["source_onnx_forwards_completed"] += 1
            events.append({"event": "source_onnx_forward_completed", "image_id": image_id, "sha256": sha256_bytes(payload)})
        status = "complete"
    except Exception as exc:
        status = "failed_partial"
        error = {"code": type(exc).__name__, "message": str(exc)}
        events.append({"event": "source_failed", "error": error})
    finally:
        ort_session = None
        if onnx_path.is_file():
            try:
                onnx_sha256_after = sha256_file(onnx_path)
                if onnx_sha256_before is not None and onnx_sha256_after != onnx_sha256_before:
                    status = "failed_partial"
                    error = error or {"code": "SOURCE_ONNX_CHANGED_DURING_RUN", "message": "ONNX bytes changed between before/after hashes"}
            except OSError as exc:
                status = "failed_partial"
                error = error or {"code": "SOURCE_ONNX_AFTER_HASH_FAILED", "message": str(exc)}
        else:
            status = "failed_partial"
            error = error or {"code": "SOURCE_ONNX_MISSING_AFTER_RUN", "message": "ONNX file disappeared during source reference stage"}
        if runtime is not None and hasattr(runtime, "close"):
            try:
                runtime.close()
                cleanup = {"status": "completed"}
            except Exception as exc:
                cleanup = {"status": "failed", "error": {"code": type(exc).__name__, "message": str(exc)}}
                error = error or {"code": "CLEANUP_FAILURE", "message": "source runtime cleanup failed", "details": cleanup["error"]}
                status = "failed_partial"
        else:
            cleanup = {"status": "not_required"}
    predictions_path = out_dir / "public/raw_predictions.json"
    prediction_payload = {"schema_version": "e2l1-raw-output-inventory-v2", "source": "onnx_cpu", "source_onnx_sha256": expected_onnx_sha256, "records": rows}
    predictions_hash = None
    if status == "complete":
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text(json.dumps(prediction_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        predictions_hash = sha256_file(predictions_path)
    manifest = {"schema_version": "e2l1-source-onnx-reference-v2", "status": "verified" if status == "complete" else status, "real_device_execution": False, "test_only": bool(test_only), "scope": "test_fixture" if test_only else "canonical_dev", "commit": commit, "source_onnx_sha256": expected_onnx_sha256, "onnx_sha256": onnx_sha256_after, "onnx_sha256_before": onnx_sha256_before, "onnx_sha256_after": onnx_sha256_after, "onnx_contract": onnx_contract if 'onnx_contract' in locals() else None, "ort_session_providers": ort_session_providers, "image_ids": expected_ids, "records": rows, "attempted_completed": counters, "executed_image_passes": counters["source_onnx_forwards_completed"], "environment": environment if 'environment' in locals() else None, "cleanup": cleanup, "model_forward": status == "complete", "native_model_forward": False, "predictions_path": str(predictions_path.relative_to(out_dir).as_posix()), "predictions_sha256": predictions_hash, "error": error}
    _write_json_once(out_dir / "manifest.json", manifest)
    _write_json_once(out_dir / "index.json", {"schema_version": "e2l1-source-reference-index-v2", "status": status, "public_artifacts": ["manifest.json", "index.json"], "private_policy": "raw source output tensors remain under private/"})
    (out_dir / "events.jsonl").write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8", newline="\n")
    return manifest


class ExistingE2TargetRuntime:
    """Production target factory reusing E2TensorRTRuntime/provider/owner/adapter."""

    def __init__(self, engine_path: Path, engine_sha256: str = ENGINE_SHA256, stage_observer: Optional[Callable[[str], None]] = None) -> None:
        self.engine_path = engine_path.resolve()
        self.engine_sha256 = engine_sha256
        self.stage_observer = stage_observer
        self._runtime = None
        self._execution = None
        self.loaded = False
        self.target_calls = 0
        self.mock_only = False

    def load_engine(self) -> None:
        from edge_readiness.e2_model_smoke import E2TensorRTRuntime, EngineArtifact
        from edge_readiness.jetson_adapter import AdapterConfig
        observed = sha256_file(self.engine_path)
        if observed != self.engine_sha256:
            raise PacketError("ENGINE_HASH_MISMATCH:" + observed)
        self._runtime = E2TensorRTRuntime(stage_observer=self.stage_observer)
        self._runtime.preflight()
        artifact = EngineArtifact(self.engine_path, observed, self.engine_path.stat().st_size, self._runtime.runtime_version, {"existing_engine_only": True, "warmup": 0})
        self._execution = self._runtime.open_execution(artifact)
        self.loaded = True

    def enqueue(self, image_id: str, input_bytes: bytes) -> bytes:
        if not self.loaded or self._execution is None:
            raise PacketError("ENGINE_NOT_LOADED")
        expected = expected_nbytes(tuple(INPUT_SHAPE), "float32")
        if len(input_bytes) != expected:
            raise PacketError("INPUT_CONTRACT_MISMATCH:" + image_id)
        host = HostTensor("images", tuple(INPUT_SHAPE), "float32", expected, bytes(input_bytes))
        self.target_calls += 1
        return self._execution.infer(host)["output0"].payload

    def copy_output(self, payload: bytes) -> bytes:
        return bytes(payload)

    def close(self) -> None:
        if self._execution is not None:
            self._execution.close()
            self._execution = None
        self._runtime = None
        self.loaded = False


@dataclass
class ImageStreamSource:
    """Pickle-safe bound image stream used by the target child."""

    image_records: List[Mapping[str, Any]]
    root: Path
    synthetic_inputs: Optional[Dict[str, bytes]] = None
    synthetic_image_bytes: Optional[Dict[str, bytes]] = None
    source_input_records: Optional[Dict[str, Mapping[str, Any]]] = None
    source_reference_manifest_sha256: Optional[str] = None

    def iter_inputs(self, image_ids: Sequence[str], event_sink: Optional[Callable[[Mapping[str, Any]], None]] = None) -> Iterable[Tuple[str, bytes, Dict[str, Any]]]:
        expected = list(image_ids)
        if [record.get("image_id") for record in self.image_records] != expected:
            raise PacketError("STREAM_IMAGE_ORDER_MISMATCH")
        seen: List[str] = []
        for record in self.image_records:
            image_id = str(record.get("image_id"))
            if len(seen) >= len(expected) or image_id != expected[len(seen)]:
                raise PacketError("STREAM_IMAGE_ORDER_MISMATCH")
            relative = Path(str(record.get("path")))
            if relative.is_absolute() or ".." in relative.parts:
                raise PacketError("STREAM_IMAGE_PATH_INVALID:" + str(image_id))
            if event_sink is not None:
                event_sink({"event": "image_attempt_started", "image_id": image_id})
            if self.synthetic_inputs is not None:
                image_bytes = (self.synthetic_image_bytes or self.synthetic_inputs)[image_id]
            else:
                image_bytes = (self.root / relative).read_bytes()
            if len(image_bytes) != record.get("bytes") or sha256_bytes(image_bytes) != record.get("sha256"):
                raise PacketError("STREAM_IMAGE_HASH_MISMATCH:" + str(image_id))
            if self.synthetic_inputs is not None:
                tensor_bytes = bytes(self.synthetic_inputs[image_id])
                preprocessing = record.get("preprocessing")
                if not isinstance(preprocessing, Mapping):
                    raise PacketError("STREAM_PREPROCESS_METADATA_MISSING:" + image_id)
            else:
                from edge_readiness.e2_image_preprocess import ImagePreprocessError, preprocess_image_bytes
                try:
                    tensor = preprocess_image_bytes(image_bytes, image_id=image_id)
                except ImagePreprocessError as exc:
                    raise PacketError(str(exc)) from exc
                tensor_bytes = bytes(tensor.payload)
                preprocessing = tensor.metadata
            if (len(tensor_bytes) != 3 * 640 * 640 * 4 or
                    preprocessing.get("input_sha256") != sha256_bytes(tensor_bytes) or
                    preprocessing.get("input_bytes") != len(tensor_bytes)):
                raise PacketError("STREAM_PREPROCESS_CONTRACT_MISMATCH:" + str(image_id))
            if list(preprocessing.get("original_shape", [])[:2]) != record.get("orig_shape"):
                raise PacketError("STREAM_IMAGE_SHAPE_MISMATCH:" + str(image_id))
            source_input = (self.source_input_records or {}).get(image_id)
            if self.source_input_records is not None and (not isinstance(source_input, Mapping) or source_input.get("sha256") != sha256_bytes(tensor_bytes) or source_input.get("preprocessing") != preprocessing):
                raise PacketError("STREAM_SOURCE_INPUT_MISMATCH:" + str(image_id))
            if event_sink is not None:
                event_sink({"event": "image_input_ready", "image_id": image_id, "input_sha256": sha256_bytes(tensor_bytes), "preprocessing": dict(preprocessing)})
            seen.append(image_id)
            yield image_id, tensor_bytes, dict(preprocessing)
        if seen != expected:
            raise PacketError("STREAM_IMAGE_COUNT_MISMATCH")


def _target_stage_child_main(image_ids: List[str], image_source: ImageStreamSource, runtime: Any, events_name: str, result_name: str) -> None:
    events_path = Path(events_name)
    result_path = Path(result_name)
    private_root = result_path.parent / "private" / "target_reference"
    private_root.mkdir(parents=True, exist_ok=True)
    counters = {"engine_load_attempted": 0, "engine_load_completed": 0, "image_inputs_attempted": 0, "image_inputs_completed": 0, "target_calls_attempted": 0, "target_calls_completed": 0, "output_copies_completed": 0, "unknown_completions": []}
    rows: List[Dict[str, Any]] = []
    result: Dict[str, Any] = {"status": "failed_partial", "counters": counters, "records": rows, "no_retry": True, "source_reference_manifest_sha256": image_source.source_reference_manifest_sha256}
    active_image: Optional[str] = None
    active_phase: Optional[str] = None
    if os.name != "nt" and bool(getattr(runtime, "ignore_terminate", False)):
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    _append_event(events_path, {"event": "child_started", "image_count": len(image_ids), "mock_only": bool(getattr(runtime, "mock_only", False)), "counters": dict(counters), "utc": _utc()})
    try:
        counters["engine_load_attempted"] = 1
        _append_event(events_path, {"event": "engine_load_attempted", "counters": dict(counters), "utc": _utc()})
        runtime.load_engine()
        counters["engine_load_completed"] = 1
        _append_event(events_path, {"event": "engine_load_completed", "counters": dict(counters), "utc": _utc()})
        image_rows = {str(row.get("image_id")): row for row in image_source.image_records}
        def stream_event(event: Mapping[str, Any]) -> None:
            nonlocal active_image, active_phase
            row = dict(event)
            if row.get("event") == "image_attempt_started":
                active_image = str(row.get("image_id"))
                active_phase = "image_preprocess"
                counters["image_inputs_attempted"] += 1
            elif row.get("event") == "image_input_ready":
                counters["image_inputs_completed"] += 1
                active_phase = "target_call"
            row["counters"] = dict(counters)
            row["utc"] = _utc()
            _append_event(events_path, row)
        for image_id, input_bytes, preprocessing in image_source.iter_inputs(image_ids, event_sink=stream_event):
            counters["target_calls_attempted"] += 1
            active_image = image_id
            active_phase = "target_call"
            _append_event(events_path, {"event": "target_call_attempted", "image_id": image_id, "counters": dict(counters), "utc": _utc()})
            output = runtime.enqueue(image_id, input_bytes)
            counters["target_calls_completed"] += 1
            copied = runtime.copy_output(output)
            validate_output_payload(copied)
            counters["output_copies_completed"] += 1
            output_path = private_root / (image_id + ".bin")
            output_path.write_bytes(copied)
            image_record = image_rows[image_id]
            record = {"image_id": image_id, "image_path": image_record.get("path"), "image_bytes": image_record.get("bytes"), "image_sha256": image_record.get("sha256"), "orig_shape": image_record.get("orig_shape"), "input_sha256": sha256_bytes(input_bytes), "input_bytes": len(input_bytes), "input_shape": list(INPUT_SHAPE), "input_dtype": "float32", "input_byteorder": "little", "preprocessing": preprocessing, "source_input_sha256": (image_source.source_input_records or {}).get(image_id, {}).get("sha256"), "source_reference_manifest_sha256": image_source.source_reference_manifest_sha256, "path": str(output_path.relative_to(result_path.parent).as_posix()), "bytes": len(copied), "sha256": sha256_bytes(copied), "shape": [1, 7, 8400], "dtype": "float32"}
            rows.append(dict(record, payload=copied))
            active_image = None
            active_phase = None
            _append_event(events_path, {"event": "target_call_completed", "image_id": image_id, "counters": dict(counters), "record": record, "utc": _utc()})
        result["status"] = "complete"
    except Exception as exc:
        error = {"code": type(exc).__name__, "message": str(exc)}
        if active_phase == "target_call" and counters["target_calls_attempted"] > counters["target_calls_completed"]:
            counters["unknown_completions"].append("target_call:" + str(active_image))
        result["error"] = error
        _append_event(events_path, {"event": "child_failed", "image_id": active_image, "phase": active_phase, "error": error, "counters": dict(counters), "utc": _utc()})
    finally:
        try:
            runtime.close()
            cleanup = {"status": "completed"}
        except Exception as exc:
            cleanup = {"status": "failed", "error": {"code": type(exc).__name__, "message": str(exc)}}
            result["error"] = result.get("error") or {"code": "CLEANUP_FAILURE", "message": "target cleanup failed", "details": cleanup["error"]}
            result["status"] = "failed_partial"
            _append_event(events_path, {"event": "cleanup_failed", "error": cleanup["error"], "counters": dict(counters), "utc": _utc()})
        result["counters"] = counters
        result["records"] = [{key: value for key, value in row.items() if key != "payload"} for row in rows]
        result["cleanup"] = cleanup
        result["durable_cleanup_known"] = True
        result["real_device_execution"] = not bool(getattr(runtime, "mock_only", False))
        result_path.write_bytes(json.dumps(result, sort_keys=True, default=_json_default).encode("utf-8"))
        if bool(getattr(runtime, "exit_after_result", False)):
            os._exit(int(getattr(runtime, "exit_after_result", 17)) if type(getattr(runtime, "exit_after_result", False)) is int else 17)
        _append_event(events_path, {"event": "child_finalized", "status": result["status"], "cleanup": cleanup, "counters": dict(counters), "utc": _utc()})


def _durable_events(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, Mapping):
            events.append(dict(value))
    return events


def _event_partial_state(events: Sequence[Mapping[str, Any]], expected_ids: Sequence[str]) -> Dict[str, Any]:
    counters = {"engine_load_attempted": 0, "engine_load_completed": 0, "image_inputs_attempted": 0, "image_inputs_completed": 0, "target_calls_attempted": 0, "target_calls_completed": 0, "output_copies_completed": 0, "unknown_completions": []}
    completed_records: Dict[str, Dict[str, Any]] = {}
    active_image = None
    active_since_ns = None
    active_phase = None
    for event in events:
        if isinstance(event.get("counters"), Mapping):
            counters.update(dict(event["counters"]))
        kind = event.get("event")
        if kind == "image_attempt_started":
            active_image = event.get("image_id")
            active_since_ns = event.get("monotonic_ns")
            active_phase = "image_preprocess"
        elif kind == "image_input_ready":
            active_phase = "target_call"
        elif kind == "target_call_attempted":
            active_image = event.get("image_id")
            if active_since_ns is None:
                active_since_ns = event.get("monotonic_ns")
            active_phase = "target_call"
        elif kind == "target_call_completed":
            if isinstance(event.get("record"), Mapping):
                completed_records[str(event.get("image_id"))] = dict(event["record"])
            if active_image == event.get("image_id"):
                active_image = None
                active_since_ns = None
                active_phase = None
    if active_image is not None:
        unknown = list(counters.get("unknown_completions", []))
        if active_phase == "target_call" and int(counters.get("target_calls_attempted", 0)) > int(counters.get("target_calls_completed", 0)):
            unknown.append("target_call:" + str(active_image))
        elif active_phase == "image_preprocess":
            unknown.append("image_preprocess:" + str(active_image))
        counters["unknown_completions"] = sorted(set(unknown))
    ordered_records = [completed_records[image_id] for image_id in expected_ids if image_id in completed_records]
    return {"counters": counters, "records": ordered_records, "active_image": active_image, "active_since_ns": active_since_ns, "active_phase": active_phase}


def _terminate_and_reap(process: Any, grace_seconds: float) -> Dict[str, Any]:
    termination = {"terminate_requested": False, "kill_escalated": False, "alive_after_grace": False, "reaped": False, "exit_code": None}
    if process.is_alive():
        termination["terminate_requested"] = True
        try:
            process.terminate()
        except Exception as exc:
            termination["terminate_error"] = {"code": type(exc).__name__, "message": str(exc)}
        process.join(grace_seconds)
    if process.is_alive():
        termination["alive_after_grace"] = True
        if hasattr(process, "kill"):
            termination["kill_escalated"] = True
            try:
                process.kill()
            except Exception as exc:
                termination["kill_error"] = {"code": type(exc).__name__, "message": str(exc)}
            process.join(grace_seconds)
    termination["reaped"] = not process.is_alive()
    termination["exit_code"] = process.exitcode
    return termination


def run_target_execution_stage(input_records: Sequence[Mapping[str, Any]], image_ids: Sequence[str], input_loader: Optional[Callable[[Mapping[str, Any]], bytes]], runtime: Any, out_dir: Path, *, permissions: StagePermissions, image_source: Optional[ImageStreamSource] = None, stage_timeout_seconds: Optional[float] = None, per_image_timeout_seconds: Optional[float] = None, termination_grace_seconds: float = 2.0, inventory: Optional[Mapping[str, Any]] = None, package_root: Optional[Path] = None, require_execution_ready: bool = False) -> Dict[str, Any]:
    """Parent-side target stage with mandatory bounded child and durable events."""
    if not permissions.allow_target_inference:
        raise PacketError("TARGET_INFERENCE_NOT_AUTHORIZED")
    if out_dir.exists():
        raise PacketError("OUTPUT_EXISTS:" + str(out_dir))
    validation = None
    if inventory is not None:
        validation = validate_package_inventory(inventory, package_root)
        if require_execution_ready and validation["status"] != "execution_ready":
            raise PacketError("PACKAGE_NOT_EXECUTION_READY")
        if require_execution_ready and image_source is not None:
            if image_source.source_input_records != validation.get("source_input_records") or image_source.source_reference_manifest_sha256 != validation.get("source_reference_manifest_sha256"):
                raise PacketError("TARGET_SOURCE_REFERENCE_NOT_BOUND")
    if image_source is None:
        if input_loader is None:
            raise PacketError("IMAGE_STREAM_SOURCE_REQUIRED")
        # This compatibility path is test-only; production dispatch supplies
        # ImageStreamSource and never consumes precomputed input tensors.
        payloads = {str(record["image_id"]): input_loader(record) for record in input_records}
        image_source = ImageStreamSource([dict(record) for record in input_records], Path("."), synthetic_inputs=payloads)
    out_dir.mkdir(parents=True)
    events_path = out_dir / "events.jsonl"
    child_result = out_dir / "child_result.json"
    resource_contract = study_contract()["resource_requirements"]
    stage_timeout_seconds = float(resource_contract["stage_timeout_seconds"] if stage_timeout_seconds is None else stage_timeout_seconds)
    per_image_timeout_seconds = float(resource_contract["timeout_per_image_seconds"] if per_image_timeout_seconds is None else per_image_timeout_seconds)
    if (not math.isfinite(stage_timeout_seconds) or not math.isfinite(per_image_timeout_seconds) or
            not math.isfinite(termination_grace_seconds) or stage_timeout_seconds <= 0 or
            per_image_timeout_seconds <= 0 or termination_grace_seconds <= 0 or
            stage_timeout_seconds > float(resource_contract["stage_timeout_seconds"]) or
            per_image_timeout_seconds > float(resource_contract["timeout_per_image_seconds"])):
        raise PacketError("TARGET_DEADLINE_INVALID")
    process = multiprocessing.get_context("spawn").Process(target=_target_stage_child_main, args=(list(image_ids), image_source, runtime, str(events_path), str(child_result)))
    process.start()
    stage_started = time.monotonic()
    timeout_code = None
    latest_events: List[Dict[str, Any]] = []
    while process.is_alive():
        process.join(0.05)
        latest_events = _durable_events(events_path)
        partial = _event_partial_state(latest_events, image_ids)
        if partial["active_since_ns"] is not None and time.monotonic_ns() - int(partial["active_since_ns"]) >= per_image_timeout_seconds * 1_000_000_000:
            timeout_code = "IMAGE_TIMEOUT"
            break
        if time.monotonic() - stage_started >= stage_timeout_seconds:
            timeout_code = "STAGE_TIMEOUT"
            break
    latest_events = _durable_events(events_path)
    termination = _terminate_and_reap(process, termination_grace_seconds) if timeout_code else {"terminate_requested": False, "kill_escalated": False, "alive_after_grace": False, "reaped": not process.is_alive(), "exit_code": process.exitcode}
    if timeout_code:
        # TERM handlers may flush one last event before exiting. Re-read only
        # after the bounded terminate/kill/reap sequence so partial evidence
        # includes every event that made it to durable storage.
        latest_events = _durable_events(events_path)
    partial = _event_partial_state(latest_events, image_ids)
    if timeout_code:
        result = {"schema_version": "e2l1-target-reference-v4", "status": "failed_partial", "real_device_execution": not bool(getattr(runtime, "mock_only", False)), "error": {"code": timeout_code, "message": "bounded target stage or per-image deadline expired"}, "counters": partial["counters"], "records": partial["records"], "unknown_completion_state": "unknown", "cleanup": {"status": "unknown_after_process_termination"}, "durable_cleanup_known": False, "termination": termination, "no_retry": True, "source_reference_manifest_sha256": image_source.source_reference_manifest_sha256}
    elif child_result.is_file():
        try:
            child_payload = json.loads(child_result.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            child_payload = None
            result = {"status": "failed_partial", "error": {"code": "CHILD_RESULT_INVALID", "message": str(exc)}, "counters": partial["counters"], "records": partial["records"], "unknown_completion_state": "unknown", "no_retry": True}
        if child_payload is not None:
            result = child_payload
            result["termination"] = termination
            result["records"] = result.get("records", partial["records"])
            if process.exitcode != 0:
                result.update({"status": "failed_partial", "error": {"code": "CHILD_EXIT_NONZERO_WITH_RESULT", "message": "child produced a result then exited abnormally", "details": {"exit_code": process.exitcode}}, "unknown_completion_state": "unknown", "durable_cleanup_known": False})
            elif result.get("durable_cleanup_known") is not True or not isinstance(result.get("cleanup"), Mapping) or result["cleanup"].get("status") != "completed":
                result.update({"status": "failed_partial", "error": {"code": "CHILD_CLEANUP_NOT_CONFIRMED", "message": "complete target evidence requires successful recorded cleanup"}, "unknown_completion_state": "unknown"})
            elif result.get("status") == "complete":
                finalized = [event for event in latest_events if event.get("event") == "child_finalized"]
                counters = result.get("counters", {})
                rows = result.get("records", [])
                if (not finalized or finalized[-1].get("status") != "complete" or
                        not latest_events or latest_events[-1].get("event") != "child_finalized" or
                        counters.get("target_calls_attempted") != len(image_ids) or counters.get("target_calls_completed") != len(image_ids) or
                        counters.get("output_copies_completed") != len(image_ids) or counters.get("image_inputs_completed") != len(image_ids) or
                        [row.get("image_id") for row in rows] != list(image_ids)):
                    result.update({"status": "failed_partial", "error": {"code": "TARGET_TERMINAL_EVIDENCE_INCOMPLETE", "message": "child completion lacks matching terminal event, counters or inventory"}, "unknown_completion_state": "unknown"})
                else:
                    for row in rows:
                        relative_raw_path = Path(str(row.get("path", "")))
                        raw_path = (out_dir / relative_raw_path).resolve()
                        if relative_raw_path.is_absolute() or ".." in relative_raw_path.parts or out_dir.resolve() not in raw_path.parents:
                            result.update({"status": "failed_partial", "error": {"code": "TARGET_ARTIFACT_PATH_INVALID", "message": "target row output path escapes attempt root", "details": {"image_id": row.get("image_id")}}, "unknown_completion_state": "unknown"})
                            break
                        expected_input = (image_source.source_input_records or {}).get(row.get("image_id"))
                        source_image = next((item for item in image_source.image_records if item.get("image_id") == row.get("image_id")), None)
                        input_binding_ok = (
                            isinstance(row.get("input_sha256"), str) and len(row["input_sha256"]) == 64 and
                            row.get("input_bytes") == 3 * 640 * 640 * 4 and row.get("input_shape") == INPUT_SHAPE and
                            row.get("input_dtype") == "float32" and row.get("input_byteorder") == "little" and
                            isinstance(row.get("preprocessing"), Mapping) and
                            row["preprocessing"].get("input_sha256") == row.get("input_sha256") and
                            row["preprocessing"].get("input_bytes") == row.get("input_bytes")
                        )
                        if isinstance(expected_input, Mapping):
                            input_binding_ok = input_binding_ok and row.get("input_sha256") == expected_input.get("sha256") and row.get("preprocessing") == expected_input.get("preprocessing") and row.get("source_input_sha256") == expected_input.get("sha256") and row.get("source_reference_manifest_sha256") == image_source.source_reference_manifest_sha256
                        else:
                            input_binding_ok = input_binding_ok and row.get("source_input_sha256") is None and row.get("source_reference_manifest_sha256") == image_source.source_reference_manifest_sha256
                        image_binding_ok = isinstance(source_image, Mapping) and all((
                            row.get("image_path") == source_image.get("path"),
                            row.get("image_bytes") == source_image.get("bytes"),
                            row.get("image_sha256") == source_image.get("sha256"),
                            row.get("orig_shape") == source_image.get("orig_shape"),
                        ))
                        output_binding_ok = raw_path.is_file() and not raw_path.is_symlink() and raw_path.stat().st_size == row.get("bytes") and sha256_file(raw_path) == row.get("sha256")
                        if not output_binding_ok or not input_binding_ok or not image_binding_ok:
                            result.update({"status": "failed_partial", "error": {"code": "TARGET_ARTIFACT_BINDING_INVALID", "message": "target row does not bind its private output, source image and source input", "details": {"image_id": row.get("image_id")}}, "unknown_completion_state": "unknown"})
                            break
    else:
        result = {"status": "failed_partial", "error": {"code": "CHILD_EXIT_WITHOUT_RESULT", "message": "target child exited before a durable result"}, "counters": partial["counters"], "records": partial["records"], "unknown_completion_state": "unknown", "durable_cleanup_known": False, "no_retry": True, "termination": termination}
    result["process"] = {"exit_code": process.exitcode, "reaped": not process.is_alive(), "alive_after_bounded_termination": process.is_alive()}
    result["schema_version"] = "e2l1-target-reference-v4"
    result["attempted_completed"] = result.get("counters", partial["counters"])
    result["durable_events"] = events_path.name
    result["no_retry"] = True
    _write_json_once(out_dir / "manifest.json", result)
    _write_json_once(out_dir / "index.json", {"schema_version": "e2l1-target-reference-index-v4", "status": result.get("status"), "public_artifacts": ["manifest.json", "index.json", "events.jsonl"], "private_policy": "raw target output tensors remain private", "stage_timeout_seconds": stage_timeout_seconds, "per_image_timeout_seconds": per_image_timeout_seconds, "termination": termination})
    return result


def run_canonical_analysis_stage(source_records_path: Path, target_records_path: Path, xml_path: Path, out_dir: Path, *, resamples: int = 1000, seed: int = 20260916) -> Dict[str, Any]:
    """Final analyzer CLI stage; it never loads a model or target runtime."""
    if out_dir.exists():
        raise PacketError("OUTPUT_EXISTS:" + str(out_dir))
    source_payload = json.loads(source_records_path.read_text(encoding="utf-8"))
    target_payload = json.loads(target_records_path.read_text(encoding="utf-8"))
    source_records = source_payload.get("records", source_payload) if isinstance(source_payload, Mapping) else source_payload
    target_records = target_payload.get("records", target_payload) if isinstance(target_payload, Mapping) else target_payload
    result = evaluate_canonical_coco_xml_pair(source_records, target_records, xml_path, resamples=resamples, seed=seed)
    out_dir.mkdir(parents=True)
    _write_json_once(out_dir / "analysis.json", result)
    _write_json_once(out_dir / "index.json", {"schema_version": "e2l1-canonical-analysis-index-v1", "status": "complete", "public_artifacts": ["analysis.json", "index.json"], "source_records_sha256": sha256_file(source_records_path), "target_records_sha256": sha256_file(target_records_path), "xml_sha256": sha256_file(xml_path)})
    return result


class MockRuntime:
    """External runtime double used by tests; no model or CUDA calls."""

    def __init__(self, outputs: Mapping[str, bytes], fail_image: Optional[str] = None, timeout_image: Optional[str] = None, cleanup_error: bool = False, hang_image: Optional[str] = None, ignore_terminate: bool = False, exit_after_result: Any = False):
        self.mock_only = True
        self.outputs = dict(outputs)
        self.fail_image = fail_image
        self.timeout_image = timeout_image
        self.cleanup_error = cleanup_error
        self.hang_image = hang_image
        self.ignore_terminate = ignore_terminate
        self.exit_after_result = exit_after_result
        self.loaded = False
        self.target_calls = 0
        self.closed = False

    def load_engine(self) -> None:
        self.loaded = True

    def enqueue(self, image_id: str, input_bytes: bytes) -> bytes:
        if not self.loaded:
            raise PacketError("ENGINE_NOT_LOADED")
        self.target_calls += 1
        if image_id == self.hang_image:
            while True:
                time.sleep(0.05)
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
    value = struct.unpack_from("<f", output)[0]
    correct = value >= 0.0
    prediction = BoxPrediction(image_id, 0, 0.9 if correct else 0.8, (0.0, 0.0, 10.0, 10.0) if correct else (20.0, 20.0, 30.0, 30.0))
    truth = GroundTruth(image_id, 0, (0.0, 0.0, 10.0, 10.0))
    return {"image_id": image_id, "output_sha256": sha256_bytes(output), "output_bytes": len(output), "predictions": [prediction], "ground_truth": [truth], "metric_status": "synthetic_cpu_evaluated", "ap_available": True}


def _json_default(value: Any) -> Any:
    if isinstance(value, (BoxPrediction, GroundTruth)):
        return {"image_id": value.image_id, "class_id": value.class_id, "score": getattr(value, "score", None), "box": list(value.box)}
    raise TypeError(type(value).__name__)


def _final_metrics(analyses: Sequence[Mapping[str, Any]], resamples: int) -> Dict[str, Any]:
    image_ids = [row["image_id"] for row in analyses]
    predictions = [item for row in analyses for item in row.get("predictions", [])]
    truths = [item for row in analyses for item in row.get("ground_truth", [])]
    if not image_ids or not truths:
        return {"status": "not_available"}
    result = evaluate_predictions(predictions, truths, image_ids)
    result["bootstrap"] = paired_bootstrap(predictions, truths, image_ids, resamples=resamples)
    return result


def run_mock_study(image_ids: Sequence[str], inputs: Mapping[str, bytes], runtime: MockRuntime, out_dir: Path, evaluator: Callable[[str, bytes], Dict[str, Any]] = mock_evaluator, bootstrap_resamples: int = 1000) -> Dict[str, Any]:
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
    result = {"schema_version": "e2l1-dev-mock-run-v2", "status": "complete" if error is None else "failed_partial", "counters": counters, "analyses": analyses, "metrics": _final_metrics(analyses, bootstrap_resamples) if error is None else {"status": "not_available_partial_run"}, "error": error, "no_retry": True, "target_calls_observed": runtime.target_calls}
    (out_dir / "events.jsonl").write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8", newline="\n")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8", newline="\n")
    return result


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    payload = dict(event)
    payload.setdefault("monotonic_ns", time.monotonic_ns())
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _child_main(image_ids: List[str], outputs: Dict[str, bytes], events_name: str, result_name: str, hang_at: Optional[str], fail_at: Optional[str], adapter_double: bool, adapter_cleanup_error: bool) -> None:
    events = Path(events_name)
    result_path = Path(result_name)
    counters = {"target_calls_attempted": 0, "target_calls_completed": 0, "output_copies_completed": 0, "analysis_completed": 0, "unknown_completions": []}
    analyses: List[Dict[str, Any]] = []
    runtime = AdapterRuntimeDouble(cleanup_error=adapter_cleanup_error) if adapter_double else None
    _append_event(events, {"event": "child_started", "image_count": len(image_ids), "adapter_double": adapter_double, "utc": _utc()})
    primary_error: Optional[Dict[str, Any]] = None
    try:
        if runtime is not None:
            runtime.load_engine()
        _append_event(events, {"event": "engine_load_completed", "utc": _utc()})
        for image_id in image_ids:
            counters["target_calls_attempted"] += 1
            _append_event(events, {"event": "target_call_attempted", "image_id": image_id, "counters": dict(counters), "utc": _utc()})
            if image_id == hang_at:
                _append_event(events, {"event": "target_call_hanging", "image_id": image_id, "counters": dict(counters), "utc": _utc()})
                while True:
                    time.sleep(0.05)
            if image_id == fail_at:
                raise RuntimeError("mock child failure:" + image_id)
            payload = runtime.enqueue(image_id, bytes(3 * 640 * 640 * 4)) if runtime is not None else bytes(outputs[image_id])
            validate_output_payload(payload)
            counters["target_calls_completed"] += 1
            counters["output_copies_completed"] += 1
            row = mock_evaluator(image_id, payload)
            counters["analysis_completed"] += 1
            analyses.append(row)
            _append_event(events, {"event": "analysis_completed", "image_id": image_id, "counters": dict(counters), "utc": _utc()})
        result = {"status": "complete", "counters": counters, "analyses": analyses, "metrics": _final_metrics(analyses, min(100, max(10, len(image_ids) * 10)))}
    except Exception as exc:
        counters["unknown_completions"].append("child_failure")
        primary_error = {"code": type(exc).__name__, "message": str(exc)}
        result = {"status": "failed_partial", "counters": counters, "analyses": analyses, "error": primary_error, "no_retry": True}
        _append_event(events, {"event": "child_failed", "error": primary_error, "counters": dict(counters), "utc": _utc()})
    finally:
        if runtime is not None:
            try:
                runtime.close()
                result["cleanup"] = {"status": "completed"}
                _append_event(events, {"event": "cleanup_completed", "counters": dict(counters), "utc": _utc()})
            except Exception as exc:
                cleanup_error = {"code": type(exc).__name__, "message": str(exc)}
                result["cleanup"] = {"status": "failed", "error": cleanup_error}
                if primary_error is None:
                    result["error"] = {"code": "CLEANUP_FAILURE", "message": "child cleanup failed", "details": cleanup_error}
                result["status"] = "failed_partial"
                _append_event(events, {"event": "cleanup_failed", "error": cleanup_error, "counters": dict(counters), "utc": _utc()})
        else:
            result["cleanup"] = {"status": "not_required"}
        result["durable_cleanup_known"] = True
        result_path.write_text(json.dumps(result, default=_json_default, sort_keys=True), encoding="utf-8", newline="\n")
        if result.get("status") == "complete":
            _append_event(events, {"event": "child_completed", "counters": dict(counters), "cleanup": result["cleanup"], "utc": _utc()})
        else:
            _append_event(events, {"event": "child_finalized_failed", "counters": dict(counters), "cleanup": result["cleanup"], "utc": _utc()})


def run_durable_child_study(image_ids: Sequence[str], outputs: Mapping[str, bytes], out_dir: Path, timeout_seconds: float = 30.0, hang_at: Optional[str] = None, fail_at: Optional[str] = None, adapter_double: bool = False, adapter_cleanup_error: bool = False) -> Dict[str, Any]:
    """Run the external double in a bounded process with durable partial evidence."""
    if out_dir.exists():
        raise PacketError("OUTPUT_EXISTS")
    out_dir.mkdir(parents=True)
    events_path = out_dir / "events.jsonl"
    child_result = out_dir / "child_result.json"
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_child_main, args=(list(image_ids), dict(outputs), str(events_path), str(child_result), hang_at, fail_at, adapter_double, adapter_cleanup_error))
    process.start()
    process.join(timeout_seconds)
    timed_out = process.is_alive()
    if timed_out:
        process.terminate()
        process.join(2.0)
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()] if events_path.is_file() else []
    counters = rows[-1].get("counters", {}) if rows else {}
    if timed_out:
        attempted = counters.get("target_calls_attempted", 0)
        completed = counters.get("target_calls_completed", 0)
        unknown = list(counters.get("unknown_completions", []))
        if completed < attempted and completed < len(image_ids):
            unknown.append("target_call:" + list(image_ids)[completed])
        counters["unknown_completions"] = unknown
        result = {"schema_version": "e2l1-dev-child-run-v1", "status": "failed_partial", "error": {"code": "STAGE_TIMEOUT", "message": "child exceeded bounded timeout"}, "counters": counters, "unknown_completion_state": "unknown", "cleanup": "child_terminated", "no_retry": True}
    elif child_result.is_file():
        result = json.loads(child_result.read_text(encoding="utf-8"))
        if result.get("durable_cleanup_known") is not True:
            result = {"schema_version": "e2l1-dev-child-run-v1", "status": "failed_partial", "error": {"code": "CLEANUP_STATE_UNKNOWN", "message": "child result did not record final cleanup"}, "counters": counters, "unknown_completion_state": "unknown", "no_retry": True}
        result.update({"schema_version": "e2l1-dev-child-run-v1", "durable_events": events_path.name, "no_retry": True})
    else:
        result = {"schema_version": "e2l1-dev-child-run-v1", "status": "failed_partial", "error": {"code": "CHILD_EXIT_WITHOUT_RESULT", "message": "child exited before final result"}, "counters": counters, "unknown_completion_state": "unknown", "no_retry": True}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8", newline="\n")
    return result


def existing_runtime_boundary() -> Dict[str, Any]:
    return {"provider": "scripts.edge_readiness.jetson_runtime_provider.TensorRTProvider", "owner": "scripts.edge_readiness.cuda_runtime_owner.CudaRuntimeMemoryOwner", "adapter": "scripts.edge_readiness.jetson_adapter.JetsonRuntimeAdapter", "production_factory": "make_lazy_e2_provider", "cpu_double": "AdapterRuntimeDouble uses JetsonRuntimeAdapter + MockBuffers/MockContext/MockStream", "child_process": "run_durable_child_study", "new_framework": False, "model_calls": False}


def packet_files(out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=False)
    contract = study_contract()
    validate_study_contract(contract)
    (out_dir / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    runbook = """# ST-EDGE-03 prospective full-dev E2 packet (R2 implementation)

Status: `prepared_not_executed`; terminal status: `edge_dev_packet_implementation_review_required`.

## Local CPU-only commands

`python scripts/edge_readiness/e2_dev_evaluation_packet.py --write-source-reference-pending --out-dir results/edge_readiness_v1/e2l1-026-source-reference-pending/source_reference.json`

This records source-ONNX CPU reference generation as pending and executes zero model forwards. A future operator must separately run the source CPU producer, publish its output hash, and update the manifest; the SERVER TensorRT FP16 hash `5c23d0fa7d4bbf09858b2f1a4dbf45c35650c474aaedebe40d845e3c9acc410a` is retained only as a distinct server reference.

`python scripts/edge_readiness/e2_dev_evaluation_packet.py --validate-contract`

`python scripts/edge_readiness/e2_dev_evaluation_packet.py --validate-package --inventory PRIVATE/package_manifest.json --package-root PRIVATE/package`

The package validator binds the exact canonical image-ID digest, producer-shaped streaming input records, source/runtime identities, XML identity, and actual allowlisted regular files. Input tensors are consumed one at a time; the full 7.49 GiB bundle is not materialized.

## Future integrated execution chain (not authorized now)

1. Server creates the source ONNX CPU reference separately from the existing SERVER TensorRT FP16 reference and counts all 1636 source passes. Missing source reference output remains a blocker.
2. After integrated GO, E2 verifies the existing engine hash on-device and the parent launches one bounded child. The child calls the existing `TensorRTProvider`/`CudaRuntimeMemoryOwner`/`JetsonRuntimeAdapter` path; no warmup, probe, debug forward, retry, build, latency or energy collection is permitted. `AdapterRuntimeDouble` tests the same adapter lifecycle without device calls.
3. Events are append+flush+fsync durable. Timeout terminates the child, preserves partial counters and marks completion unknown. Cleanup is recorded separately.
4. CPU postprocess uses the actual lazy `ultralytics.utils.nms.non_max_suppression` binding with a copied float32 tensor of shape `[1,7,8400]`; input bytes are checked unchanged and detections are recorded.
5. Once source and target records exist, `evaluate_canonical_coco_xml_pair` delegates to the accepted `coco_xml_paired_image_bootstrap_v1` implementation (`pycocotools==2.0.10`, PCG64 seed `20260916`, 1000 sorted paired image draws, duplicates retained) and reports source, target and target-minus-source AP50/AP50-95 for all/xs/s/m/l/xl. No custom AP result is substituted.

No SSH, transfer, build, inference or benchmark is authorized by this packet. Historical raw FAILs and accepted saved-output diagnostics remain unchanged.
"""
    (out_dir / "runbook.md").write_text(runbook, encoding="utf-8", newline="\n")
    (out_dir / "index.json").write_text(json.dumps({"schema_version": contract["schema_version"], "status": contract["status"], "public_artifacts": ["contract.json", "runbook.md", "index.json"], "runtime_boundary": existing_runtime_boundary(), "private_policy": "no engine/ONNX/input/raw prediction bytes"}, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return {"contract": out_dir / "contract.json", "runbook": out_dir / "runbook.md", "index": out_dir / "index.json"}


def closing_packet_files(out_dir: Path) -> Dict[str, Path]:
    """Publish the R2 closure packet without changing the historical R2 root."""
    if out_dir.exists():
        raise PacketError("OUTPUT_EXISTS:" + str(out_dir))
    out_dir.mkdir(parents=True)
    contract = study_contract()
    contract.update({"schema_version": "e2l1-dev-evaluation-packet-v4", "terminal_status": "edge_dev_packet_canonical_reference_pending", "execution_chain": {"source": {"entrypoint": "run_source_reference_stage", "runtime": "edge_readiness.e2_source_bundle.UltralyticsSourceRuntime", "permission": "--allow-source-forward", "public_outputs": ["manifest.json", "index.json", "events.jsonl"], "raw_outputs": "private/"}, "target": {"entrypoint": "run_target_execution_stage", "runtime": "ExistingE2TargetRuntime -> E2TensorRTRuntime -> TensorRTProvider/CudaRuntimeMemoryOwner/JetsonRuntimeAdapter", "permission": "--allow-target-inference", "one_pass_per_image": True, "public_outputs": ["manifest.json", "index.json", "events.jsonl"], "raw_outputs": "private/"}, "analysis": {"entrypoint": "run_canonical_analysis_stage", "runtime": EVALUATOR_ID, "permission": "CPU-only; source and target records must already exist", "undefined_metrics": "JSON null"}}, "source_reference_status": "pending_not_executed", "real_device_execution": False})
    validate_study_contract({**contract, "schema_version": "e2l1-dev-evaluation-packet-v3", "terminal_status": "edge_dev_packet_implementation_review_required"})
    (out_dir / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    runbook = """# E2L1-027 ST-EDGE-03 closing packet

Status: `prepared_not_executed`; source CPU reference is still `0/1636` and no
E2 forward is authorized by this packet.

## Guarded stage commands

Source CPU reference (only in a separately approved source-forward session):

`python scripts/edge_readiness/e2_dev_evaluation_packet.py --stage source --source-root SOURCE_ROOT --checkpoint SOURCE_ROOT/results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt --image-records PRIVATE/image_records.json --out-dir PRIVATE/source-reference --allow-source-forward --commit CODE_COMMIT`

The stage reuses `e2_source_bundle.UltralyticsSourceRuntime`, consumes the
bound image records in order, writes raw output only below `private/`, and
publishes hashes/counters. Without `--allow-source-forward` it fails closed.

Target E2 execution (only after a later integrated GO):

`python scripts/edge_readiness/e2_dev_evaluation_packet.py --stage target --engine PRIVATE/engine.plan --input-records PRIVATE/package_manifest.json --package-root PRIVATE/package --out-dir PRIVATE/target-reference --allow-target-inference`

The target stage reuses `E2TensorRTRuntime`, `TensorRTProvider`,
`CudaRuntimeMemoryOwner`, `OwnedBuffers` and `JetsonRuntimeAdapter`; it does
not build, warm up, retry or benchmark. Without `--allow-target-inference` it
fails closed. Parent cleanup is known before a child result is accepted.

CPU-only terminal analysis:

`python scripts/edge_readiness/e2_dev_evaluation_packet.py --stage analyze --source-records PRIVATE/source_records.json --target-records PRIVATE/target_records.json --xml PRIVATE/xml.zip --out-dir results/edge_readiness_v1/e2l1-027-canonical-analysis`

Analysis normalizes the accepted name-keyed XML loader contract, rejects
duplicate IDs/shape mismatches, delegates COCO/XML AP and PCG64 paired
resampling to the locked repository evaluator, and serializes undefined
support as `null`. It never loads a model or invokes a device runtime.

No SSH, transfer, source forward, build, E2 inference, retry or benchmark was
performed while producing this packet. Historical FAILs and private raw bytes
remain unchanged.
"""
    (out_dir / "runbook.md").write_text(runbook, encoding="utf-8", newline="\n")
    (out_dir / "index.json").write_text(json.dumps({"schema_version": contract["schema_version"], "status": contract["status"], "public_artifacts": ["contract.json", "runbook.md", "index.json"], "execution_chain": contract["execution_chain"], "private_policy": "no model/engine/input/raw tensor bytes"}, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return {"contract": out_dir / "contract.json", "runbook": out_dir / "runbook.md", "index": out_dir / "index.json"}


def c1c4_packet_files(out_dir: Path) -> Dict[str, Path]:
    """Publish one auditable C1-C4 package without executing an edge stage."""
    if out_dir.exists():
        raise PacketError("OUTPUT_EXISTS:" + str(out_dir))
    out_dir.mkdir(parents=True)
    contract = study_contract()
    contract.update({
        "schema_version": "e2l1-c1c4-execution-chain-packet-v1",
        "terminal_status": "edge_c1c4_chain_ready_no_execution",
        "source_reference_status": "pending_not_executed",
        "real_device_execution": False,
        "execution_chain": {
            "C1_source": {
                "entrypoint": "run_source_reference_stage",
                "runtime": "edge_readiness.e2_source_bundle.UltralyticsSourceRuntime",
                "backend": "onnxruntime",
                "provider_allowlist": ["CPUExecutionProvider"],
                "native_forward_forbidden": True,
                "onnx_sha256": SOURCE_ONNX_SHA256,
                "outputs": "private/onnx_reference/*.bin + public/raw_predictions.json",
            },
            "C2_postprocess": {
                "entrypoint": "postprocess_saved_outputs",
                "nms": NMS_SYMBOL,
                "input_shape": [1, 7, 8400],
                "coordinate_space": "original_image_xyxy",
                "output": "records.json canonical analyzer schema",
                "input_mutation_check": True,
            },
            "C3_target": {
                "entrypoint": "run_target_execution_stage",
                "child_process": True,
                "durable_events": "append+flush+fsync",
                "timeout_is_bounded": True,
                "retry": False,
                "cleanup_state_required": True,
            },
            "C4_package_input": {
                "validator": "validate_package_inventory",
                "image_source": "ImageStreamSource",
                "producer": "bound_image_preprocess_stream",
                "precomputed_tensor_bundle_production": False,
                "execution_ready_gate": True,
            },
        },
    })
    # The shared study validator remains the source of truth for immutable
    # contract fields; the packet adds only the C1-C4 execution-chain fields.
    validate_study_contract({**contract, "schema_version": "e2l1-dev-evaluation-packet-v3", "terminal_status": "edge_dev_packet_implementation_review_required"})
    (out_dir / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    runbook = """# E2L1-028 C1-C4 integrated execution-chain packet

Status: `prepared_not_executed`; this package completes the local interfaces
and tests. It does not authorize SSH, transfer, model build, source forward,
E2 inference, or benchmark.

## C1 — pinned ONNX CPU source

The source entrypoint hashes `best.onnx` before opening it, requires exactly
`CPUExecutionProvider`, calls `UltralyticsSourceRuntime.preprocess` and
`ort_forward`, and never loads the native PyTorch checkpoint. Raw output is
private; the public inventory binds each output hash. The canonical ONNX hash
is `bd20b36d640c502358c44edbbde51f05267eda2518b0c0a4cfe84ca18a00d4b7`.

```text
python scripts/edge_readiness/e2_dev_evaluation_packet.py --stage source --source-root PACKAGE_ROOT --onnx PACKAGE_ROOT/private/onnx_export/best.onnx --image-records PACKAGE_ROOT/image_records.json --out-dir ATTEMPT/source --allow-source-forward --commit CODE_COMMIT
```

This command is a future separately approved source-reference action; it was
not run while producing this packet.

## C2 — saved-output postprocess and canonical records

`postprocess_saved_outputs` verifies the raw manifest membership, bytes and
hashes, invokes the accepted Ultralytics NMS helper on a copied float32
`[1,7,8400]` tensor, checks that the saved tensor was not mutated, maps boxes
back to original-image coordinates, and writes the canonical analyzer
`records.json`. The integrated test covers producer → private raw output →
postprocess → canonical analysis.

```text
python scripts/edge_readiness/e2_dev_evaluation_packet.py --stage postprocess --raw-manifest ATTEMPT/manifest.json --raw-root ATTEMPT --image-records PACKAGE_ROOT/image_records.json --source-label onnx_cpu --out-dir ATTEMPT/records
```

## C3 — bounded target child and durable evidence

The target entrypoint validates the package gate before dispatch, starts one
bounded child, streams one image through preprocessing at a time, and persists
events with flush+fsync around load/call/copy/failure/cleanup. A timeout
terminates the child, retains counters and marks completion unknown; no retry
or late success is accepted. Cleanup must be known before a child result is
accepted.

```text
python scripts/edge_readiness/e2_dev_evaluation_packet.py --stage target --engine PACKAGE_ROOT/private/engine.plan --input-records PACKAGE_ROOT/package_manifest.json --package-root PACKAGE_ROOT --out-dir ATTEMPT/target --allow-target-inference
```

The command is not run by E2L1-028.

## C4 — execution-ready package and image stream

`validate_package_inventory` binds the canonical 1636-image digest, source
and target identities, allowlisted regular files, and source-reference status.
`ImageStreamSource` reads and hashes each image, preprocesses it through the
existing source runtime, validates shape/dtype/finite bytes, and yields one
input at a time. A precomputed tensor bundle cannot satisfy the production
path; synthetic inputs exist only for explicit unit tests.

```text
python scripts/edge_readiness/e2_dev_evaluation_packet.py --validate-package --inventory PACKAGE_ROOT/package_manifest.json --package-root PACKAGE_ROOT
```

The conditional smoke permission from E2L1-017 was consumed by the build
timeout. There is no unused smoke authorization. A future prospective dev
comparison requires integrated review and a separate explicit GO; it is not
authorized by this packet. Do not run model calls, transfer, build, inference
or benchmark from this runbook.
"""
    (out_dir / "runbook.md").write_text(runbook, encoding="utf-8", newline="\n")
    index = {
        "schema_version": contract["schema_version"],
        "status": contract["status"],
        "terminal_status": contract["terminal_status"],
        "public_artifacts": ["contract.json", "runbook.md", "index.json"],
        "execution_chain": contract["execution_chain"],
        "private_policy": "no edge binary, engine, raw input or raw prediction payload in this packet",
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return {"contract": out_dir / "contract.json", "runbook": out_dir / "runbook.md", "index": out_dir / "index.json"}


def p1p4_packet_files(out_dir: Path) -> Dict[str, Path]:
    """Write the E2L1-029 implementation contract/runbook; never executes a stage."""
    if out_dir.exists():
        raise PacketError("OUTPUT_EXISTS:" + str(out_dir))
    out_dir.mkdir(parents=True)
    contract = study_contract()
    contract.update({
        "schema_version": "e2l1-p1p4-prospective-dev-packet-v1",
        "terminal_status": "prepared_not_executed_integrated_review_required",
        "source_reference_status": "pending_not_executed",
        "real_device_execution": False,
        "execution_authorization": {"source_forward": False, "target_inference": False, "build": False, "benchmark": False, "unused_e2l1_017_smoke_go": False},
        "implementation_repairs": {
            "P1": "record actual LetterBox gain/padding/rounding per image; use scale_boxes with the frozen metadata",
            "P2": "source and target share numpy/cv2 CPU image-byte preprocessor; target does not execute source runtime preflight",
            "P3": "real CLI uses 54000-second stage ceiling and 30-second per-image watchdog, durable event reconstruction, bounded terminate/kill/reap",
            "P4": "verified source manifest binds canonical 1636 IDs, order, actual image hashes/shapes, input hashes/preprocess, ORT providers and ONNX before/after hashes",
        },
    })
    validate_study_contract({**contract, "schema_version": "e2l1-dev-evaluation-packet-v3", "terminal_status": "edge_dev_packet_implementation_review_required"})
    (out_dir / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    runbook = """# E2L1-029 P1-P4 prospective dev packet

Status: `prepared_not_executed`; local code/tests only. NO-GO source ONNX
forward, SSH, transfer, engine build/load, E2 inference or benchmark. The
single conditional smoke authorized by E2L1-017 was consumed by the 900s
build timeout; there is no unused smoke permission. This packet adds no GO.

## Frozen prospective scope

The proposed study remains CCTSDB2021/dev, exactly 1,636 canonical image IDs
in frozen order, pinned source ONNX CPU reference and existing E2 FP16/1 GiB
engine. Existing confidence/IoU/max-detection and paired AP/CI protocol stays
unchanged. A new dev comparison is prospective only and requires integrated
review plus explicit authorization. It is not an additional model call now.

## P1 — transform provenance

Preprocessing persists per-image resize gain, integer left/top padding,
unpadded dimensions, exact four borders and pinned LetterBox rounding rule.
Post-NMS boxes return to original-image coordinates through the installed
`ultralytics.utils.ops.scale_boxes` helper with that recorded ratio/pad. The
regression matrix covers portrait, landscape, odd rounding, clipping and the
actual pinned LetterBox output.

## P2 — real target image stream

`ImageStreamSource` reads bytes, verifies the image hash, decodes and validates
shape, then uses `e2_image_preprocess` (NumPy/OpenCV only) to generate one
little-endian float32 input at a time. The source ORT/export preflight is not
called by target preprocessing and CUDA visibility is not queried there.
Streamed image/input hashes and transform metadata must match the source
reference manifest before they are associated with target output.

The prior read-only E2 probe observed Python 3.8.10 and no Torch, Ultralytics,
ONNX or ONNX Runtime in that probed interpreter. NumPy/OpenCV availability was
not established. Do not install anything under this packet; an authorized
operator must verify the exact target interpreter's CPU preprocessing
dependencies before any future execution review.

## P3 — bounded target child

The CLI takes its deadlines from the frozen contract: 54,000 seconds maximum
for the stage and 30 seconds per image, including preprocessing and target
work. Every event is append/flush/fsync persisted. Timeout recovery rereads
events after bounded TERM, KILL escalation and reap, preserving counters,
completed artifact inventory and unknown in-flight completion. Result success
requires exit code zero, all expected counts and outputs, terminal event last,
and confirmed cleanup. No retry or late success is accepted.

## P4 — source and package binding

Production source mode requires the exact canonical 1,636 IDs/order and
validates every image's bytes/hash/decoded shape before ORT opens. It records
the actual CPU provider, input hash/transform and ONNX hashes before and after
the pass. The production package validator joins each image, preprocessing
fingerprint and saved source output to the actual source-reference manifest;
test-only subsets (at most five images) cannot pass the production gate.

Future local command templates remain guarded. Do not dispatch them based on
this packet:

```text
python scripts/edge_readiness/e2_dev_evaluation_packet.py --validate-package --inventory PACKAGE_ROOT/package_manifest.json --package-root PACKAGE_ROOT
python scripts/edge_readiness/e2_dev_evaluation_packet.py --stage source --source-root PACKAGE_ROOT --onnx PACKAGE_ROOT/private/onnx_export/best.onnx --image-records PACKAGE_ROOT/image_records.json --out-dir ATTEMPT/source --allow-source-forward --commit CODE_COMMIT
python scripts/edge_readiness/e2_dev_evaluation_packet.py --stage target --engine PACKAGE_ROOT/private/engine.plan --input-records PACKAGE_ROOT/package_manifest.json --package-root PACKAGE_ROOT --out-dir ATTEMPT/target --allow-target-inference
```

The final comparison, if separately approved later, is prospective on all
1,636 dev images. No smoke, source pass, build, inference or benchmark is
authorized by this runbook.
"""
    (out_dir / "runbook.md").write_text(runbook, encoding="utf-8", newline="\n")
    index = {"schema_version": contract["schema_version"], "status": contract["status"], "terminal_status": contract["terminal_status"], "public_artifacts": ["contract.json", "runbook.md", "index.json"], "execution_authorization": contract["execution_authorization"], "private_policy": "no image, ONNX, engine, input tensor or raw prediction payload"}
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return {"contract": out_dir / "contract.json", "runbook": out_dir / "runbook.md", "index": out_dir / "index.json"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="E2L1-029 prospective dev chain with explicit execution guards")
    parser.add_argument("--write-packet", action="store_true")
    parser.add_argument("--write-closing-packet", action="store_true")
    parser.add_argument("--write-c1c4-packet", action="store_true")
    parser.add_argument("--write-p1p4-packet", action="store_true")
    parser.add_argument("--write-source-reference-pending", action="store_true")
    parser.add_argument("--validate-contract", action="store_true")
    parser.add_argument("--validate-package", action="store_true")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--stage", choices=("source", "postprocess", "target", "analyze"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--source-records", type=Path)
    parser.add_argument("--target-records", type=Path)
    parser.add_argument("--image-records", type=Path)
    parser.add_argument("--input-records", type=Path)
    parser.add_argument("--raw-manifest", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--source-label")
    parser.add_argument("--xml", type=Path)
    parser.add_argument("--allow-source-forward", action="store_true")
    parser.add_argument("--allow-target-inference", action="store_true")
    parser.add_argument("--commit", default="unknown")
    args = parser.parse_args(argv)
    if args.write_packet:
        if args.out_dir is None:
            parser.error("--write-packet requires --out-dir")
        paths = packet_files(args.out_dir)
        print(json.dumps({key: str(value.resolve()) for key, value in paths.items()}, sort_keys=True))
        return 0
    if args.write_closing_packet:
        if args.out_dir is None:
            parser.error("--write-closing-packet requires --out-dir")
        paths = closing_packet_files(args.out_dir)
        print(json.dumps({key: str(value.resolve()) for key, value in paths.items()}, sort_keys=True))
        return 0
    if args.write_c1c4_packet:
        if args.out_dir is None:
            parser.error("--write-c1c4-packet requires --out-dir")
        paths = c1c4_packet_files(args.out_dir)
        print(json.dumps({key: str(value.resolve()) for key, value in paths.items()}, sort_keys=True))
        return 0
    if args.write_p1p4_packet:
        if args.out_dir is None:
            parser.error("--write-p1p4-packet requires --out-dir")
        paths = p1p4_packet_files(args.out_dir)
        print(json.dumps({key: str(value.resolve()) for key, value in paths.items()}, sort_keys=True))
        return 0
    if args.write_source_reference_pending:
        if args.out_dir is None:
            parser.error("--write-source-reference-pending requires --out-dir as the output file")
        print(json.dumps(write_source_reference_pending(args.out_dir), sort_keys=True))
        return 0
    if args.validate_contract:
        print(json.dumps(validate_study_contract(study_contract()), sort_keys=True))
        return 0
    if args.validate_package:
        if args.inventory is None:
            parser.error("--validate-package requires --inventory")
        inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        print(json.dumps(validate_package_inventory(inventory, args.package_root), sort_keys=True))
        return 0
    if args.stage == "source":
        required = (args.source_root, args.onnx, args.image_records, args.out_dir)
        if any(value is None for value in required):
            parser.error("--stage source requires --source-root, --onnx, --image-records and --out-dir")
        records_payload = json.loads(args.image_records.read_text(encoding="utf-8"))
        image_records = records_payload.get("records", records_payload) if isinstance(records_payload, Mapping) else records_payload
        result = run_source_reference_stage(image_records, args.source_root, args.onnx, args.out_dir, permissions=StagePermissions(allow_source_forward=args.allow_source_forward), commit=args.commit)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "verified" else 2
    if args.stage == "target":
        if args.engine is None or args.input_records is None or args.package_root is None or args.out_dir is None:
            parser.error("--stage target requires --engine, --input-records, --package-root and --out-dir")
        records_payload = json.loads(args.input_records.read_text(encoding="utf-8"))
        inventory = records_payload
        image_records = inventory.get("input_producer", {}).get("image_records", [])
        image_ids = [record["image_id"] for record in image_records]
        validation = validate_package_inventory(inventory, args.package_root)
        if validation["status"] != "execution_ready":
            raise PacketError("PACKAGE_NOT_EXECUTION_READY")
        runtime = ExistingE2TargetRuntime(args.engine)
        source = ImageStreamSource(image_records, args.package_root, source_input_records=validation["source_input_records"], source_reference_manifest_sha256=validation["source_reference_manifest_sha256"])
        resource_contract = study_contract()["resource_requirements"]
        result = run_target_execution_stage(image_records, image_ids, None, runtime, args.out_dir, permissions=StagePermissions(allow_target_inference=args.allow_target_inference), image_source=source, stage_timeout_seconds=resource_contract["stage_timeout_seconds"], per_image_timeout_seconds=resource_contract["timeout_per_image_seconds"], inventory=inventory, package_root=args.package_root, require_execution_ready=True)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "complete" else 2
    if args.stage == "postprocess":
        if args.raw_manifest is None or args.raw_root is None or args.image_records is None or args.source_label is None or args.out_dir is None:
            parser.error("--stage postprocess requires --raw-manifest, --raw-root, --image-records, --source-label and --out-dir")
        records_payload = json.loads(args.image_records.read_text(encoding="utf-8"))
        image_records = records_payload.get("records", records_payload) if isinstance(records_payload, Mapping) else records_payload
        result = postprocess_saved_outputs(args.raw_manifest, args.raw_root, image_records, args.out_dir, source_label=args.source_label)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    if args.stage == "analyze":
        if args.source_records is None or args.target_records is None or args.xml is None or args.out_dir is None:
            parser.error("--stage analyze requires --source-records, --target-records, --xml and --out-dir")
        result = run_canonical_analysis_stage(args.source_records, args.target_records, args.xml, args.out_dir)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    parser.error("select one packet/validation action or --stage analyze; source/target stages require the approved orchestrator contract")


if __name__ == "__main__":
    raise SystemExit(main())
