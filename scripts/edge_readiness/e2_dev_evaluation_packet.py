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
NMS_SYMBOL = "ultralytics.utils.nms.non_max_suppression"
DEV_YAML_SHA256 = "ea8f40ba75920b2a67c2b5e1fd18c3d2eaf760686f11d0efea9bb7f26d5f5479"
REFERENCE_PREDICTIONS_SHA256 = "5c23d0fa7d4bbf09858b2f1a4dbf45c35650c474aaedebe40d845e3c9acc410a"
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


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def study_contract() -> Dict[str, Any]:
    return {
        "schema_version": "e2l1-dev-evaluation-packet-v2",
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
            "dev_yaml_sha256": DEV_YAML_SHA256,
            "reference_predictions_sha256": REFERENCE_PREDICTIONS_SHA256,
            "xml_identity_required": True,
        },
        "runtime_contract": {"existing_engine_only": True, "runtime": "E2 original TRT 8.5.2.2", "imgsz": 640, "batch": 1, "conf": 0.001, "iou": 0.7, "max_det": 300, "workers": 0, "rect": False, "warmup": 0, "debug_forwards": 0, "latency_benchmark": False, "energy_benchmark": False, "duplicate_nms": False, "postprocess_helper": {"package": "ultralytics", "version": ULTRALYTICS_VERSION, "symbol": NMS_SYMBOL, "device": "cpu/reference and E2 target-owned postprocess", "return_idxs": True}},
        "call_budget": {"target_image_passes": DEV_IMAGES, "target_passes_per_image": 1, "reference_image_passes_if_reuse_not_verified": DEV_IMAGES, "reference_passes_if_reuse_verified": 0, "warmup_calls": 0, "shape_probe_calls": 0, "debug_calls": 0, "automatic_retries": 0, "builder_invocations": 0},
        "endpoints": {"primary": ["paired_source_onnx_vs_e2_fp16_COCO_bbox_AP50", "paired_source_onnx_vs_e2_fp16_COCO_bbox_AP50_95"], "secondary": ["XS/S_AP50", "XS/S_AP50_95", "per_class_counts", "coordinate_and_threshold_diagnostics"], "bootstrap": {"estimator": EVALUATOR_ID, "seed": 20260916, "resamples": 1000, "paired_image_sampling": True, "noninferiority_margin": None}},
        "resource_requirements": {"fresh_e2_output_root": True, "input_storage_mode": "streaming_manifest_only", "raw_input_tensor_bundle_forbidden": True, "private_output_storage_gib_min": 1, "available_memory_gib_min": 4, "per_image_target_output_bytes": OUTPUT_BYTES, "timeout_per_image_seconds": 30, "stage_timeout_seconds": 54000, "streaming_required": True, "retain_raw_tensors": "private optional; public hashes/metrics only"},
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
    if inventory.get("input_contract") != {"shape": INPUT_SHAPE, "dtype": "float32", "byteorder": "little", "finite": True, "bytes_per_image": 3 * 640 * 640 * 4}:
        raise PacketError("PACKAGE_INPUT_CONTRACT_MISMATCH")
    if inventory.get("input_storage_mode") != "streaming_manifest_only" or inventory.get("raw_input_tensor_bundle_forbidden") is not True:
        raise PacketError("PACKAGE_INPUT_STORAGE_POLICY")
    identity = inventory.get("reference_identity", {})
    if identity.get("source_onnx_sha256") != SOURCE_ONNX_SHA256 or identity.get("predictions_sha256") != REFERENCE_PREDICTIONS_SHA256:
        raise PacketError("PACKAGE_REFERENCE_IDENTITY_MISMATCH")
    xml_hash = identity.get("xml_sha256")
    if not isinstance(xml_hash, str) or len(xml_hash) != 64 or any(char not in "0123456789abcdef" for char in xml_hash.lower()):
        raise PacketError("PACKAGE_XML_IDENTITY_MISSING")
    allowed = inventory.get("allowed_files", [])
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
    return {"status": "package_binding_verified", "engine_private": True, "raw_tensors_private": True, "files_checked": len(allowed) if package_root is not None else 0, "image_count": len(image_ids)}


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
    if not truths:
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
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _child_main(image_ids: List[str], outputs: Dict[str, bytes], events_name: str, result_name: str, hang_at: Optional[str], fail_at: Optional[str]) -> None:
    events = Path(events_name)
    result_path = Path(result_name)
    counters = {"target_calls_attempted": 0, "target_calls_completed": 0, "output_copies_completed": 0, "analysis_completed": 0, "unknown_completions": []}
    analyses: List[Dict[str, Any]] = []
    _append_event(events, {"event": "child_started", "image_count": len(image_ids), "utc": _utc()})
    _append_event(events, {"event": "engine_load_completed", "utc": _utc()})
    try:
        for image_id in image_ids:
            counters["target_calls_attempted"] += 1
            _append_event(events, {"event": "target_call_attempted", "image_id": image_id, "counters": dict(counters), "utc": _utc()})
            if image_id == hang_at:
                _append_event(events, {"event": "target_call_hanging", "image_id": image_id, "counters": dict(counters), "utc": _utc()})
                while True:
                    time.sleep(0.05)
            if image_id == fail_at:
                raise RuntimeError("mock child failure:" + image_id)
            payload = bytes(outputs[image_id])
            validate_output_payload(payload)
            counters["target_calls_completed"] += 1
            counters["output_copies_completed"] += 1
            row = mock_evaluator(image_id, payload)
            counters["analysis_completed"] += 1
            analyses.append(row)
            _append_event(events, {"event": "analysis_completed", "image_id": image_id, "counters": dict(counters), "utc": _utc()})
        result = {"status": "complete", "counters": counters, "analyses": analyses, "metrics": _final_metrics(analyses, min(100, max(10, len(image_ids) * 10)))}
        result_path.write_text(json.dumps(result, default=_json_default, sort_keys=True), encoding="utf-8", newline="\n")
        _append_event(events, {"event": "child_completed", "counters": dict(counters), "utc": _utc()})
    except Exception as exc:
        counters["unknown_completions"].append("child_failure")
        _append_event(events, {"event": "child_failed", "error": str(exc), "counters": dict(counters), "utc": _utc()})


def run_durable_child_study(image_ids: Sequence[str], outputs: Mapping[str, bytes], out_dir: Path, timeout_seconds: float = 30.0, hang_at: Optional[str] = None, fail_at: Optional[str] = None) -> Dict[str, Any]:
    """Run the external double in a bounded process with durable partial evidence."""
    if out_dir.exists():
        raise PacketError("OUTPUT_EXISTS")
    out_dir.mkdir(parents=True)
    events_path = out_dir / "events.jsonl"
    child_result = out_dir / "child_result.json"
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_child_main, args=(list(image_ids), dict(outputs), str(events_path), str(child_result), hang_at, fail_at))
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
        result.update({"schema_version": "e2l1-dev-child-run-v1", "durable_events": events_path.name, "no_retry": True})
    else:
        result = {"schema_version": "e2l1-dev-child-run-v1", "status": "failed_partial", "error": {"code": "CHILD_EXIT_WITHOUT_RESULT", "message": "child exited before final result"}, "counters": counters, "unknown_completion_state": "unknown", "no_retry": True}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8", newline="\n")
    return result


def existing_runtime_boundary() -> Dict[str, Any]:
    return {"provider": "scripts.edge_readiness.jetson_runtime_provider.TensorRTProvider", "owner": "scripts.edge_readiness.cuda_runtime_owner.CudaRuntimeMemoryOwner", "adapter": "scripts.edge_readiness.jetson_adapter.JetsonRuntimeAdapter", "double": "MockRuntime/child process", "new_framework": False, "model_calls": False}


def packet_files(out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=False)
    contract = study_contract()
    validate_study_contract(contract)
    (out_dir / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    runbook = """# ST-EDGE-03 prospective full-dev E2 packet (R1 implementation)\n\nStatus: `prepared_not_executed`; terminal status: `edge_dev_packet_implementation_review_required`.\n\nThe local implementation validates every frozen identity/runtime field, checks actual package files, uses the existing Jetson provider/owner/adapter boundary through an external double, persists child events, bounds timeout and retains partial counters, and computes synthetic CPU AP50/AP50-95 plus paired duplicate-image bootstrap CI. It performs no model, CUDA, TensorRT, SSH, transfer, build or inference.\n\n## Future operator chain\n\n1. Server verifies the canonical 1636-image/2706-instance dev inventory, XML/image membership, source ONNX `bd20b36d640c502358c44edbbde51f05267eda2518b0c0a4cfe84ca18a00d4b7`, and reference prediction hash `5c23d0fa7d4bbf09858b2f1a4dbf45c35650c474aaedebe40d845e3c9acc410a`.\n2. Server publishes a private manifest with relative paths, bytes and SHA256; Luna1 runs `validate_package_inventory(package_root=...)`. Wrong engine/source/XML/hash/size/path/symlink/duplicate is a hard stop. Inputs are streaming manifest-only; a full 7.49 GiB float32 tensor bundle is forbidden.\n3. After a later integrated GO, the owner-side E2 preflight verifies the existing engine hash on E2. Luna1 runs one target pass per image through the existing provider/owner/adapter; warmup, shape probe, debug forward, retry, build, latency and energy are all zero.\n4. Child events are append+fsync durable. A timeout terminates the child, preserves counters and marks completion unknown; no automatic retry.\n5. CPU analysis then invokes `coco_xml_paired_image_bootstrap_v1` with seed 20260916 and 1000 paired image resamples for AP50/AP50-95 plus size/class diagnostics. The dependency-free evaluator here is only a synthetic workflow check, not the canonical XML evaluator.\n\nMissing identity, membership, exact call accounting, durable evidence or engine availability is a blocker. This packet does not authorize execution now.\n"""
    (out_dir / "runbook.md").write_text(runbook, encoding="utf-8", newline="\n")
    (out_dir / "index.json").write_text(json.dumps({"schema_version": contract["schema_version"], "status": contract["status"], "public_artifacts": ["contract.json", "runbook.md", "index.json"], "runtime_boundary": existing_runtime_boundary(), "private_policy": "no engine/ONNX/input/raw prediction bytes"}, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
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
