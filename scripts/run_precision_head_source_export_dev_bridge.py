#!/usr/bin/env python3
"""Compare frozen native-FP32 and accepted-ONNX-FP32 detection on CCTSDB dev.

This is a server-only application-level bridge.  The parent performs identity,
dataset, XML and output-protection checks without importing Torch or ONNX
Runtime, then dispatches one model-scoped CPU child at a time.  Children use
the pinned Ultralytics preprocessing, one frozen PyTorch FP32 forward and one
CPUExecutionProvider ONNX session call per image.  Only bounded detection
records are retained; raw model tensors are never written.

The bridge is descriptive evidence.  It does not alter the historical raw
numeric FAIL, export an ONNX graph, build TensorRT, run a test split or open a
precision matrix.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import prepare_precision_head_confirmation as readiness
import prepare_precision_head_confirmation_graph as graph
import verify_precision_head_confirmation_numeric as numeric


STUDY = "precision_head_source_export_dev_bridge_v1"
OUTPUT_ROOT_NAME = STUDY
DEFAULT_OUTPUT = Path("results/measurement_audit_v1") / OUTPUT_ROOT_NAME
DEFAULT_READINESS_ROOT = Path("results/measurement_audit_v1") / "server_precision_head_confirmation_readiness_v2"
DEFAULT_GRAPH_AUDIT_ROOT = Path("results/measurement_audit_v1") / "precision_head_confirmation_graph_audit_v4"
DEFAULT_ONNX_ROOT = Path("results/measurement_audit_v1") / "precision_head_confirmation_graph_prep_v2"
DEFAULT_DEV_ROOT = Path("data/processed/cctsdb2021_clean/dev")
MODEL_CHOICES = ("yolov8n", "yolo26n")
DEV_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEV_IMAGE_COUNT = 1636
DEV_INSTANCE_COUNT = 2706
NAMES = ("prohibitory", "mandatory", "warning")
CONF = 0.001
IOU = 0.7
MAX_DET = 300
IMG_SIZE = 640
SCHEMA_VERSION = 1


class BridgeUnresolved(RuntimeError):
    """Raised when a bridge prerequisite or semantic validity check fails."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value: Any) -> bytes:
    return readiness.canonical_json(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json_no_overwrite(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_text_no_overwrite(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.write_text(value, encoding="utf-8", newline="\n")


def write_jsonl_line(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


def repo_path(repo: Path, value: Path) -> Path:
    return numeric.repo_path(repo, value)


def relative_to_repo(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def git_head(repo: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("Cannot resolve Git HEAD")
    return result.stdout.strip()


def selected_models(value: str) -> list[str]:
    if value not in (*MODEL_CHOICES, "all"):
        raise ValueError(f"Unsupported model selection: {value}")
    return list(MODEL_CHOICES) if value == "all" else [value]


def call_budget(image_count: int = DEV_IMAGE_COUNT, models: int = 2) -> dict[str, Any]:
    if image_count <= 0 or models <= 0:
        raise ValueError("call budget dimensions must be positive")
    per_model = {"native_cpu_forward": image_count, "onnx_cpu_session_run": image_count}
    return {
        "images_per_model": image_count,
        "models": models,
        "per_model": per_model,
        "total_ordinary_forward_or_session_calls": models * image_count * 2,
        "session_construction_calls": models,
        "export_calls": 0,
        "tensorRT_build_calls": 0,
        "repeats": 1,
    }


def application_route(model: str) -> dict[str, Any]:
    if model == "yolov8n":
        return {
            "model": model,
            "input_representation": "[1,7,8400] decoded xywh plus post-sigmoid class probabilities",
            "postprocess": "ultralytics.utils.nms.non_max_suppression",
            "nms": True,
            "end2end": False,
            "conf": CONF,
            "iou": IOU,
            "max_det": MAX_DET,
            "nc": 3,
            "agnostic": False,
            "multi_label": False,
            "coordinate_scale": "ultralytics.utils.ops.scale_boxes from 640x640 input pixels to original image xyxy",
        }
    if model == "yolo26n":
        return {
            "model": model,
            "input_representation": "[1,300,6] decoded end-to-end xyxy, confidence, class id",
            "postprocess": "ultralytics.utils.nms.non_max_suppression end2end=True filtering path",
            "nms": False,
            "end2end": True,
            "conf": CONF,
            "iou": IOU,
            "max_det": MAX_DET,
            "nc": 3,
            "agnostic": False,
            "multi_label": False,
            "coordinate_scale": "ultralytics.utils.ops.scale_boxes from 640x640 input pixels to original image xyxy",
        }
    raise ValueError(f"Unsupported application route: {model}")


def validate_dev_names(image_names: list[str], label_names: list[str], expected_names: list[str]) -> None:
    if image_names != expected_names:
        raise ValueError("Current dev image order/membership differs from accepted canonical reference")
    image_stems = [Path(name).stem for name in image_names]
    label_stems = [Path(name).stem for name in label_names]
    if len(image_stems) != len(set(image_stems)):
        raise ValueError("Current dev image inventory contains duplicate stems")
    if image_stems != sorted(image_stems):
        raise ValueError("Current dev image inventory is not lexicographically ordered")
    if set(image_stems) != set(label_stems):
        raise ValueError("Current dev image/label stem sets differ")


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} is a symlink")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path.resolve(strict=True)


def dev_inventory(repo: Path, dev_root: Path, canonical: dict[str, Any]) -> dict[str, Any]:
    dev_root = repo_path(repo, dev_root)
    images_dir = dev_root / "images"
    labels_dir = dev_root / "labels"
    if images_dir.is_symlink() or labels_dir.is_symlink():
        raise ValueError("Dev images/labels directory must not be a symlink")
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Dev images/labels directories are missing under {dev_root}")
    image_paths = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in DEV_IMAGE_EXTENSIONS)
    label_paths = sorted(path for path in labels_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")
    image_names = [path.name for path in image_paths]
    label_names = [path.name for path in label_paths]
    expected_names = [row["image"] for row in canonical["records"]]
    validate_dev_names(image_names, label_names, expected_names)
    if len(image_names) != DEV_IMAGE_COUNT or len(label_names) != DEV_IMAGE_COUNT:
        raise ValueError(f"Dev inventory count differs: images={len(image_names)}, labels={len(label_names)}")
    canonical_shapes = {row["image"]: list(row["orig_shape"]) for row in canonical["records"]}
    rows = []
    for path in image_paths:
        resolved = _regular_file(path, f"dev image {path.name}")
        rows.append({
            "order": len(rows),
            "image": path.name,
            "source_path": relative_to_repo(repo, resolved),
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
            "canonical_orig_shape": canonical_shapes[path.name],
        })
    return {
        "root": relative_to_repo(repo, dev_root),
        "images_dir": relative_to_repo(repo, images_dir),
        "labels_dir": relative_to_repo(repo, labels_dir),
        "image_count": len(image_names),
        "label_count": len(label_names),
        "instances_expected_from_canonical_reference": DEV_INSTANCE_COUNT,
        "order_sha256": sha256_bytes(canonical_json(image_names)),
        "rows": rows,
        "label_names": label_names,
        "hash_scope": "current server dev image bytes; label names only in parent, XML supplies metric truth",
        "official_test_accessed": False,
    }


def external_file_evidence(path: Path, label: str) -> dict[str, Any]:
    resolved = _regular_file(path, label)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved), "regular_file": True}


def validate_xml_binding(xml_path: Path, image_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the requested dev XML members without importing model runtimes."""
    from audit_cctsdb_measurement import load_xml

    records = {row["image"]: row for row in image_rows}
    xml = load_xml(xml_path, records)
    if set(xml) != set(records):
        raise ValueError("XML archive membership differs from canonical dev images")
    class_counts: collections.Counter[str] = collections.Counter()
    instances = 0
    for image, row in xml.items():
        if list(row["shape"]) != list(records[image]["canonical_orig_shape"]):
            raise ValueError(f"XML/image shape differs for {image}")
        for class_id, _box in row["rows"]:
            class_counts[NAMES[int(class_id)]] += 1
            instances += 1
    if instances != DEV_INSTANCE_COUNT:
        raise ValueError(f"XML instance count differs: {instances} != {DEV_INSTANCE_COUNT}")
    return {
        "archive": external_file_evidence(xml_path, "XML annotation archive"),
        "images": len(xml),
        "instances": instances,
        "class_instances": dict(sorted(class_counts.items())),
        "member_order": "matched by basename stem; evaluator sorts image basename",
        "loader_source_sha256": sha256_bytes(inspect.getsource(load_xml).encode("utf-8")),
    }


def canonical_dev(repo: Path, config: dict[str, Any]) -> dict[str, Any]:
    reference = readiness._canonical_dev_reference(repo, config)
    if reference["images"] != DEV_IMAGE_COUNT or reference["instances"] != DEV_INSTANCE_COUNT:
        raise ValueError("Canonical dev reference count differs from locked contract")
    return reference


def model_binding(repo: Path, config: dict[str, Any], accepted: dict[str, Any], graph_audit: dict[str, Any], model: str, onnx_root: Path) -> dict[str, Any]:
    checkpoint = numeric.validate_checkpoint_contract(repo, config, accepted, model)
    accepted_contract = numeric.accepted_model_contracts(accepted, [model])[model]
    graph_doc = graph_audit["models"][model]
    onnx_path = repo_path(repo, onnx_root / "models" / model / "model.onnx")
    onnx_before = external_file_evidence(onnx_path, f"accepted ONNX {model}")
    if onnx_before["sha256"] != numeric.EXPECTED_ONNX_SHA256[model]:
        raise ValueError(f"Accepted ONNX hash differs for {model}: {onnx_before['sha256']}")
    graph_onnx = graph_doc.get("provenance", {}).get("onnx", {})
    if (
        graph_onnx.get("before", {}).get("sha256") != numeric.EXPECTED_ONNX_SHA256[model]
        or graph_onnx.get("after", {}).get("sha256") != numeric.EXPECTED_ONNX_SHA256[model]
        or graph_onnx.get("unchanged_during_audit") is not True
    ):
        raise ValueError(f"Graph-v4 does not bind unchanged accepted ONNX bytes for {model}")
    return {
        "model": model,
        "checkpoint": checkpoint,
        "onnx": {"path": str(onnx_path), "repo_relative_path": relative_to_repo(repo, onnx_path), "expected_sha256": numeric.EXPECTED_ONNX_SHA256[model], "before": onnx_before},
        "graph_contract": {"audit_model_report": f"precision_head_confirmation_graph_audit_v4/models/{model}/graph_audit.json", "mapping_status": graph_doc.get("mapping_status"), "output_shape": numeric.EXPECTED_OUTPUT_SHAPES[model]},
        "accepted_contract": accepted_contract,
        "application_route": application_route(model),
    }


def build_plan(repo: Path, args: argparse.Namespace, models: list[str]) -> dict[str, Any]:
    readiness_root = repo_path(repo, args.readiness_root)
    graph_root = repo_path(repo, args.graph_audit_root)
    onnx_root = repo_path(repo, args.onnx_root)
    accepted = graph.validate_readiness_artifact(repo, readiness_root)
    binding = graph.validate_config_binding(repo, accepted)
    config_path = repo_path(repo, Path(binding["path"]))
    config = readiness.load_config(config_path, repo)
    canonical = canonical_dev(repo, config)
    inventory = dev_inventory(repo, args.dev_root, canonical)
    image_rows = inventory["rows"]
    xml_binding = validate_xml_binding(args.xml.resolve(), image_rows)
    graph_audit = numeric.validate_graph_audit_artifact(repo, graph_root, models)
    model_plans = {model: model_binding(repo, config, accepted, graph_audit, model, onnx_root) for model in models}
    runtime = config.get("runtime", {})
    fixed_settings = {
        "imgsz": IMG_SIZE,
        "batch": 1,
        "rect": False,
        "conf": CONF,
        "iou": IOU,
        "max_det": MAX_DET,
        "workers": 0,
        "task": "detect",
        "preprocess": {
            "producer": "Ultralytics BasePredictor.pre_transform/preprocess",
            "LetterBox.auto": False,
            "LetterBox.scale_fill": False,
            "LetterBox.scaleup": True,
            "LetterBox.center": True,
            "LetterBox.padding_value": 114,
            "input_color_order": "decoded BGR -> RGB CHW -> float32 / 255",
        },
    }
    if any(runtime.get(key) != fixed_settings[key] for key in ("imgsz", "batch", "conf", "iou", "max_det", "workers")):
        raise ValueError(f"Locked runtime settings differ from config: {runtime}")
    return {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY,
        "status": "dispatching_cpu_children",
        "repo_head": git_head(repo),
        "output_root": relative_to_repo(repo, repo_path(repo, args.out_dir)),
        "config": {"path": relative_to_repo(repo, config_path), "file": numeric.file_evidence(repo, config_path), "readiness_binding": {key: value for key, value in binding.items() if key != "config"}},
        "readiness": {"root": relative_to_repo(repo, readiness_root), "accepted_git_commit": accepted["accepted_git_commit"], "accepted_execution_commit": accepted["accepted_execution_commit"], "files": accepted["files"]},
        "graph_audit": {"root": relative_to_repo(repo, graph_root), "commit": graph_audit["commit"], "files": graph_audit["files"]},
        "canonical_dev_reference": canonical,
        "dev_inventory": inventory,
        "xml_binding": xml_binding,
        "selected_models": models,
        "models": model_plans,
        "fixed_settings": fixed_settings,
        "call_contract": call_budget(len(image_rows), len(models)),
        "application_contract": {model: application_route(model) for model in models},
        "metric_contract": {
            "estimator": "COCO_bbox_AP_custom_CCTSDB_area_XML_original_coordinates_v1",
            "source": "scripts/verify_cctsdb_capture.py::coco_size",
            "areas": ["all", "xs", "s", "m", "l", "xl"],
            "values": "AP50 and AP50-95 in [0,1]; report signed ONNX-minus-native delta",
            "matching_diagnostic": "none; ordered application outputs are compared descriptively, no rematching replaces AP",
        },
        "environment": {
            "requested": numeric.CPU_ENVIRONMENT,
            "mode": "CPU children; CUDA hidden; ORT CPUExecutionProvider only",
            "expected_packages": {name: runtime.get(name) for name in ("torch", "ultralytics", "numpy", "pycocotools")},
        },
        "provenance_scope": "source checkpoint and accepted ONNX are read-only; source-native and exporter semantics remain separate",
        "audit_flags": {"export_performed": False, "onnx_modified": False, "tensorRT_imported": False, "tensorRT_build_performed": False, "gpu_used": False, "calibration_loader_called": False, "official_test_accessed": False, "training_performed": False, "matrix_opened": False, "scored_run_authorized": False},
        "no_overwrite": True,
        "no_resume": True,
    }


def new_child_state(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "status": "running",
        "stage": "not_started",
        "stage_history": [],
        "forward_counts": {"native_cpu_forward": {"attempted": 0, "completed": 0}, "onnx_cpu_session_run": {"attempted": 0, "completed": 0}},
        "finite_checks": {"native": {"attempted": 0, "failed": 0}, "onnx": {"attempted": 0, "failed": 0}},
        "records_written": {"native": 0, "onnx": 0},
    }


def stage(state: dict[str, Any], name: str, **details: Any) -> None:
    state["stage"] = name
    state["stage_history"].append({"sequence": len(state["stage_history"]), "stage": name, **details})


def _stride(network: Any) -> int:
    value = getattr(network, "stride", 32)
    try:
        return int(value.max().item())
    except AttributeError:
        return int(value)


def _array_contract(primary: Any, model: str, np: Any) -> dict[str, Any]:
    expected = numeric.EXPECTED_OUTPUT_SHAPES[model]
    array = np.ascontiguousarray(primary)
    if list(array.shape) != expected:
        raise BridgeUnresolved(f"{model} primary shape differs: {list(array.shape)} != {expected}")
    if array.dtype != np.dtype("float32"):
        raise BridgeUnresolved(f"{model} primary dtype differs: {array.dtype}")
    if not np.isfinite(array).all():
        raise BridgeUnresolved(f"{model} primary output is non-finite")
    return {"shape": list(array.shape), "dtype": str(array.dtype), "finite": True, "sha256": sha256_bytes(array.tobytes(order="C"))}


def _trace_public(trace: dict[str, Any], row: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = dict(trace)
    result["path"] = row["source_path"]
    result.update({"order": row["order"], "image": row["image"], "expected_source_sha256": row["sha256"], "expected_source_bytes": row["bytes"]})
    return result


def _ratio_pad(trace: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    letterbox = trace["letterbox"]
    ratio = tuple(float(value) for value in letterbox["ratio"])
    padding = (float(letterbox["padding"]["left"]), float(letterbox["padding"]["top"]))
    return ratio, padding


def application_postprocess(model: str, primary: Any, trace: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    """Apply the locked installed producer route to one primary output."""
    np = runtime["np"]
    torch = runtime["torch"]
    nms = runtime["nms"]
    ops = runtime["ops"]
    primary_np = np.ascontiguousarray(primary)
    _array_contract(primary_np, model, np)
    tensor = torch.from_numpy(primary_np)
    route = application_route(model)
    filtered = nms.non_max_suppression(
        tensor,
        conf_thres=CONF,
        iou_thres=IOU,
        classes=None,
        agnostic=False,
        max_det=MAX_DET,
        nc=3,
        end2end=(model == "yolo26n"),
    )
    if not isinstance(filtered, list) or len(filtered) != 1:
        raise BridgeUnresolved(f"{model} installed postprocess returned an invalid batch")
    detections = filtered[0]
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise BridgeUnresolved(f"{model} postprocess output shape differs: {list(detections.shape)}")
    if detections.numel() and not bool(torch.isfinite(detections).all()):
        raise BridgeUnresolved(f"{model} postprocess returned non-finite detections")
    boxes = detections[:, :4].clone()
    orig_shape = tuple(int(value) for value in trace["original_image_size_hw"])
    scaled = ops.scale_boxes((IMG_SIZE, IMG_SIZE), boxes, orig_shape, ratio_pad=_ratio_pad(trace))
    scaled_np = np.ascontiguousarray(scaled.detach().cpu().numpy())
    confidence = np.ascontiguousarray(detections[:, 4].detach().cpu().numpy())
    class_float = np.ascontiguousarray(detections[:, 5].detach().cpu().numpy())
    class_id = np.rint(class_float).astype(np.int64)
    if class_float.size and (not np.array_equal(class_float, class_id.astype(class_float.dtype)) or np.any((class_id < 0) | (class_id > 2))):
        raise BridgeUnresolved(f"{model} postprocess returned invalid class ids")
    if scaled_np.size and (not np.isfinite(scaled_np).all() or np.any(scaled_np[:, 2:] < scaled_np[:, :2])):
        raise BridgeUnresolved(f"{model} scaled detections have invalid geometry")
    if confidence.size and (not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1))):
        raise BridgeUnresolved(f"{model} detections have invalid confidence")
    return {
        "xyxy": scaled_np.astype(float).tolist(),
        "confidence": confidence.astype(float).tolist(),
        "class_id": class_id.astype(int).tolist(),
        "count": int(len(class_id)),
        "coordinate_contract": "original image xyxy after installed scale_boxes; clipped to image bounds",
        "route": route,
    }


def _source_evidence() -> dict[str, Any]:
    from ultralytics.engine.predictor import BasePredictor
    from ultralytics.utils import nms, ops

    return {
        "numeric_runtime_sources": numeric.runtime_source_evidence(),
        "application_sources": {
            "nms": {"module": nms.__file__, "callable": "ultralytics.utils.nms.non_max_suppression", "sha256": sha256_bytes(inspect.getsource(nms.non_max_suppression).encode("utf-8"))},
            "scale_boxes": {"module": ops.__file__, "callable": "ultralytics.utils.ops.scale_boxes", "sha256": sha256_bytes(inspect.getsource(ops.scale_boxes).encode("utf-8"))},
            "predictor_postprocess_reference": {"module": inspect.getsourcefile(BasePredictor), "callable": "ultralytics.engine.predictor.BasePredictor.preprocess", "sha256": sha256_bytes(inspect.getsource(BasePredictor.preprocess).encode("utf-8"))},
        },
    }


def _record(model: str, source: str, row: dict[str, Any], trace: dict[str, Any], primary_meta: dict[str, Any], detections: dict[str, Any]) -> dict[str, Any]:
    public_trace_hash = sha256_bytes(canonical_json(trace))
    return {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY,
        "model": model,
        "source": source,
        "order": row["order"],
        "image": row["image"],
        "orig_shape": list(trace["original_image_size_hw"]),
        "input_binding": {
            "source_path": row["source_path"],
            "image_sha256": row["sha256"],
            "image_bytes": row["bytes"],
            "preprocess_trace_sha256": public_trace_hash,
            "tensor_shape": trace["tensor"]["shape"],
            "tensor_dtype": trace["tensor"]["dtype"],
            "tensor_sha256": trace["tensor"]["bytes_sha256"],
            "same_input_contract_native_onnx": True,
        },
        "primary_output": primary_meta,
        "detections": detections,
    }


def _load_records(path: Path, expected_names: list[str]) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise BridgeUnresolved(f"Blank record line at {path}:{line_number}")
            records.append(json.loads(line))
    if [row.get("image") for row in records] != expected_names:
        raise BridgeUnresolved(f"Record image order/membership differs in {path}")
    if [row.get("order") for row in records] != list(range(len(expected_names))):
        raise BridgeUnresolved(f"Record order is not contiguous in {path}")
    for row in records:
        if row.get("source") not in {"native", "onnx"}:
            raise BridgeUnresolved(f"Invalid record source in {path}: {row.get('source')}")
        detections = row.get("detections", {})
        if len(detections.get("xyxy", [])) != len(detections.get("confidence", [])) or len(detections.get("xyxy", [])) != len(detections.get("class_id", [])):
            raise BridgeUnresolved(f"Detection array length mismatch in {path}: {row.get('image')}")
    return records


def ordered_output_comparison(native: list[dict[str, Any]], onnx: list[dict[str, Any]], limit: int = 20) -> dict[str, Any]:
    if [row.get("image") for row in native] != [row.get("image") for row in onnx]:
        raise BridgeUnresolved("Native/ONNX record memberships differ before comparison")
    examples = []
    equal_images = 0
    native_total = 0
    onnx_total = 0
    for left, right in zip(native, onnx):
        ld = left["detections"]
        rd = right["detections"]
        native_total += int(ld["count"])
        onnx_total += int(rd["count"])
        equal = canonical_json(ld) == canonical_json(rd)
        equal_images += int(equal)
        if not equal and len(examples) < limit:
            examples.append({
                "image": left["image"],
                "order": left["order"],
                "native_count": int(ld["count"]),
                "onnx_count": int(rd["count"]),
                "native_classes_first": ld["class_id"][:10],
                "onnx_classes_first": rd["class_id"][:10],
                "comparison": "ordered payload equality only; no rematching, sorting or replacement of AP estimator",
            })
    return {
        "images": len(native),
        "ordered_payload_equal_images": equal_images,
        "ordered_payload_different_images": len(native) - equal_images,
        "native_predictions": native_total,
        "onnx_predictions": onnx_total,
        "signed_prediction_count_delta_onnx_minus_native": onnx_total - native_total,
        "bounded_examples": examples,
        "matching": "not performed; examples retain ordered producer outputs and are not an AP estimator",
    }


def signed_metric_delta(native_report: dict[str, Any], onnx_report: dict[str, Any]) -> dict[str, Any]:
    labels = ("all", "xs", "s", "m", "l", "xl")
    result = {}
    for label in labels:
        left = native_report["metrics"][label]
        right = onnx_report["metrics"][label]
        result[label] = {
            "map50": None if left["map50"] is None or right["map50"] is None else float(right["map50"] - left["map50"]),
            "map50_95": None if left["map50_95"] is None or right["map50_95"] is None else float(right["map50_95"] - left["map50_95"]),
            "definition": "ONNX minus native, AP units [0,1]",
        }
    return result


def _metrics(native: list[dict[str, Any]], onnx: list[dict[str, Any]], xml_path: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    from audit_cctsdb_measurement import load_xml
    from verify_cctsdb_capture import coco_size

    def metric_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "image": row["image"],
                "orig_shape": row["orig_shape"],
                "xyxy": row["detections"]["xyxy"],
                "confidence": row["detections"]["confidence"],
                "class_id": row["detections"]["class_id"],
            }
            for row in records
        ]

    native_metric_records = metric_records(native)
    onnx_metric_records = metric_records(onnx)
    truth = load_xml(xml_path, {row["image"]: row for row in native_metric_records})
    native_report = coco_size(native_metric_records, truth)
    onnx_report = coco_size(onnx_metric_records, truth)
    for report in (native_report, onnx_report):
        report["xml_sha256"] = sha256_file(xml_path)
        report["runtime_provider"] = "CPUExecutionProvider"
    return {
        "estimator": "COCO_bbox_AP_custom_CCTSDB_area_XML_original_coordinates_v1",
        "native": native_report,
        "onnx": onnx_report,
        "signed_delta_onnx_minus_native": signed_metric_delta(native_report, onnx_report),
        "note": "Descriptive source/export drift on dev. No post-hoc equivalence threshold or PASS label is applied.",
    }


def _model_report(repo: Path, plan: dict[str, Any], model: str, out_dir: Path, state: dict[str, Any], runtime: dict[str, Any], source_evidence: dict[str, Any], native_records: list[dict[str, Any]], onnx_records: list[dict[str, Any]], xml_path: Path, checkpoint_before: dict[str, Any], onnx_before: dict[str, Any], checkpoint_after: dict[str, Any], onnx_after: dict[str, Any], inventory_before: list[dict[str, Any]], inventory_after: list[dict[str, Any]]) -> dict[str, Any]:
    expected_names = [row["image"] for row in plan["dev_inventory"]["rows"]]
    comparison = ordered_output_comparison(native_records, onnx_records)
    metric_result = _metrics(native_records, onnx_records, xml_path, runtime)
    if checkpoint_before != checkpoint_after or onnx_before != onnx_after:
        raise BridgeUnresolved(f"Accepted input binary changed during {model} bridge")
    if inventory_before != inventory_after:
        raise BridgeUnresolved(f"Dev image bytes changed during {model} bridge")
    if len(native_records) != len(expected_names) or len(onnx_records) != len(expected_names):
        raise BridgeUnresolved(f"Incomplete native/ONNX record count for {model}")
    state["status"] = "completed"
    stage(state, "completed")
    return {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY,
        "model": model,
        "status": "completed",
        "validity": "validity_checks_passed; scientific_assessment_descriptive",
        "runtime": runtime["packages"],
        "onnx_providers": runtime["available_providers"],
        "source_evidence": source_evidence,
        "checkpoint": {"before": checkpoint_before, "after": checkpoint_after, "unchanged": checkpoint_before == checkpoint_after},
        "onnx": {"before": onnx_before, "after": onnx_after, "unchanged": onnx_before == onnx_after},
        "records": {"native": "models/%s/native_records.jsonl" % model, "onnx": "models/%s/onnx_records.jsonl" % model, "preprocess": "models/%s/preprocess_trace.jsonl" % model, "images": len(expected_names)},
        "comparison": comparison,
        "metrics": metric_result,
        "forward_counts": {key: dict(value) for key, value in state["forward_counts"].items()},
        "finite_checks": state["finite_checks"],
        "lifecycle": {"status": state["status"], "stage": state["stage"], "stage_history": state["stage_history"], "forward_counts": state["forward_counts"]},
        "audit_flags": {"export_performed": False, "onnx_modified": False, "tensorRT_imported": False, "tensorRT_build_performed": False, "gpu_used": False, "calibration_loader_called": False, "official_test_accessed": False, "training_performed": False, "matrix_opened": False, "scored_run_authorized": False},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_model_child(repo: Path, plan: dict[str, Any], model: str, out_dir: Path, state: dict[str, Any] | None = None, runtime_loader: Any = numeric.load_child_runtime) -> dict[str, Any]:
    state = state or new_child_state(model)
    numeric.set_cpu_environment()
    out_dir.mkdir(parents=True, exist_ok=True)
    stage(state, "child_started")
    config = read_json(repo_path(repo, Path(plan["config"]["path"])))
    stage(state, "runtime_loading")
    runtime = runtime_loader(config)
    from ultralytics.utils import nms, ops

    runtime["nms"] = nms
    runtime["ops"] = ops
    stage(state, "runtime_loaded", providers=runtime["available_providers"])
    model_plan = plan["models"][model]
    checkpoint_path = repo_path(repo, Path(model_plan["checkpoint"]["expected"]["path"]))
    onnx_path = repo_path(repo, Path(model_plan["onnx"]["repo_relative_path"]))
    checkpoint_before = numeric.file_evidence(repo, checkpoint_path)
    onnx_before = external_file_evidence(onnx_path, f"accepted ONNX {model}")
    if checkpoint_before.get("sha256") != model_plan["checkpoint"]["expected"]["sha256"] or onnx_before["sha256"] != numeric.EXPECTED_ONNX_SHA256[model]:
        raise BridgeUnresolved(f"Input binary binding differs for {model}")
    stage(state, "binary_binding_complete")
    dev_rows = plan["dev_inventory"]["rows"]
    inventory_before = [{"image": row["image"], "bytes": _regular_file(repo_path(repo, Path(row["source_path"])), f"dev image {row['image']}").stat().st_size, "sha256": sha256_file(repo_path(repo, Path(row["source_path"]))) } for row in dev_rows]
    stage(state, "model_loading")
    loaded = runtime["YOLO"](str(checkpoint_path), task="detect")
    network = loaded.model.to("cpu").float().eval()
    native_head = numeric.validate_native_head(network, model, model_plan["accepted_contract"])
    model_stride = _stride(network)
    stage(state, "head_validated", head=native_head["head"], stride=model_stride)
    stage(state, "onnx_session_setup")
    session, onnx_contract = numeric.run_onnx_session(onnx_path, model, runtime)
    stage(state, "onnx_session_ready", contract=onnx_contract)
    source_evidence = _source_evidence()
    native_path = out_dir / "native_records.jsonl"
    onnx_records_path = out_dir / "onnx_records.jsonl"
    trace_path = out_dir / "preprocess_trace.jsonl"
    with native_path.open("x", encoding="utf-8", newline="\n") as native_handle, onnx_records_path.open("x", encoding="utf-8", newline="\n") as onnx_handle, trace_path.open("x", encoding="utf-8", newline="\n") as trace_handle:
        for index, row in enumerate(dev_rows):
            stage(state, "preprocess", image=row["image"], order=index)
            image_path = repo_path(repo, Path(row["source_path"]))
            current = external_file_evidence(image_path, f"dev image {row['image']}")
            if current["sha256"] != row["sha256"] or current["bytes"] != row["bytes"]:
                raise BridgeUnresolved(f"Dev image binding changed before forward: {row['image']}")
            tensor, trace = numeric.trace_preprocess(image_path, runtime, stride=model_stride)
            trace = _trace_public(trace, row, repo)
            if trace["original_image_size_hw"] != row["canonical_orig_shape"]:
                raise BridgeUnresolved(f"Decoded image shape differs from accepted reference: {row['image']}")
            input_np = runtime["np"].ascontiguousarray(tensor.detach().cpu().numpy())
            write_jsonl_line(trace_handle, trace)
            stage(state, "preprocess_complete", image=row["image"], order=index)
            stage(state, "native_forward", image=row["image"], order=index)
            state["forward_counts"]["native_cpu_forward"]["attempted"] += 1
            with runtime["torch"].no_grad():
                native_output = network(tensor)
            state["forward_counts"]["native_cpu_forward"]["completed"] += 1
            native_primary, native_contract = numeric.extract_native_primary(native_output, model, runtime["np"])
            state["finite_checks"]["native"]["attempted"] += 1
            native_meta = _array_contract(native_primary, model, runtime["np"])
            native_detections = application_postprocess(model, native_primary, trace, runtime)
            native_record = _record(model, "native", row, trace, native_meta, native_detections)
            native_record["native_output_contract"] = native_contract
            write_jsonl_line(native_handle, native_record)
            state["records_written"]["native"] += 1
            stage(state, "native_complete", image=row["image"], order=index)
            stage(state, "onnx_forward", image=row["image"], order=index)
            state["forward_counts"]["onnx_cpu_session_run"]["attempted"] += 1
            outputs = session.run(["output0"], {"images": input_np})
            if len(outputs) != 1:
                raise BridgeUnresolved(f"ONNX returned {len(outputs)} outputs for {model}")
            state["forward_counts"]["onnx_cpu_session_run"]["completed"] += 1
            onnx_primary = runtime["np"].ascontiguousarray(outputs[0])
            state["finite_checks"]["onnx"]["attempted"] += 1
            onnx_meta = _array_contract(onnx_primary, model, runtime["np"])
            onnx_detections = application_postprocess(model, onnx_primary, trace, runtime)
            onnx_record = _record(model, "onnx", row, trace, onnx_meta, onnx_detections)
            write_jsonl_line(onnx_handle, onnx_record)
            state["records_written"]["onnx"] += 1
            stage(state, "onnx_complete", image=row["image"], order=index)
            if (index + 1) % 100 == 0 or index + 1 == len(dev_rows):
                print(f"{model}: completed {index + 1}/{len(dev_rows)} images", flush=True)
    stage(state, "metric_evaluation")
    expected_names = [row["image"] for row in dev_rows]
    native_records = _load_records(native_path, expected_names)
    onnx_records = _load_records(onnx_records_path, expected_names)
    checkpoint_after = numeric.file_evidence(repo, checkpoint_path)
    onnx_after = external_file_evidence(onnx_path, f"accepted ONNX {model}")
    inventory_after = [{"image": row["image"], "bytes": _regular_file(repo_path(repo, Path(row["source_path"])), f"dev image {row['image']}").stat().st_size, "sha256": sha256_file(repo_path(repo, Path(row["source_path"]))) } for row in dev_rows]
    report = _model_report(repo, plan, model, out_dir, state, runtime, source_evidence, native_records, onnx_records, Path(plan["xml_binding"]["archive"]["path"]), checkpoint_before, onnx_before, checkpoint_after, onnx_after, inventory_before, inventory_after)
    report["native_head_contract"] = native_head
    report["onnx_session_contract"] = onnx_contract
    report["model_stride"] = model_stride
    return report


def publishable_inventory(output_root: Path) -> list[str]:
    allowed = {"bridge_plan.json", "bridge_manifest.json", "report.md"}
    if output_root.is_dir():
        for path in output_root.rglob("*"):
            if not path.is_file() or "private" in path.parts:
                continue
            relative = path.relative_to(output_root).as_posix()
            if path.suffix in {".json", ".jsonl", ".md", ".log"} or path.name.endswith(".log.gz"):
                allowed.add(relative)
    return sorted(allowed)


def child_owned_files(output_root: Path, model: str) -> list[str]:
    model_root = output_root / "models" / model
    if not model_root.is_dir():
        return []
    return sorted(path.relative_to(output_root).as_posix() for path in model_root.rglob("*") if path.is_file() and "private" not in path.parts)


def child_failure(plan: dict[str, Any], model: str, state: dict[str, Any], exc: Exception, output_root: Path) -> dict[str, Any]:
    state["status"] = "failed"
    return {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY,
        "model": model,
        "status": "failed",
        "validity": "blocked_not_interpretable",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "stage": state["stage"],
        "stage_history": state["stage_history"],
        "forward_counts": state["forward_counts"],
        "finite_checks": state["finite_checks"],
        "records_written": state["records_written"],
        "partial_files_scope": f"models/{model}/owned_paths_only",
        "partial_files": child_owned_files(output_root, model),
        "no_silent_resume": True,
        "audit_flags": {"export_performed": False, "onnx_modified": False, "tensorRT_imported": False, "tensorRT_build_performed": False, "gpu_used": False, "calibration_loader_called": False, "official_test_accessed": False, "training_performed": False, "matrix_opened": False, "scored_run_authorized": False},
    }


def _final_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Precision-head source/export dev bridge",
        "",
        f"- Study: `{manifest['study']}`",
        f"- Execution: `{manifest['status']}`; validity: `{manifest.get('validity', 'not_observed')}`.",
        "- This is a descriptive native-FP32 versus accepted-ONNX-FP32 application-level bridge on CCTSDB2021/dev. It does not replace the historical numeric FAIL.",
        "- No export, TensorRT import/build, GPU, calibration, training, official test, repeat or precision matrix was run by this package.",
        "",
        "## Model results",
        "",
    ]
    for row in manifest.get("models", []):
        lines.append(f"- `{row.get('model')}`: status `{row.get('status')}`, validity `{row.get('validity', 'blocked')}`, forward counts `{row.get('forward_counts', {})}`.")
        if row.get("status") == "completed":
            delta = row.get("metrics", {}).get("signed_delta_onnx_minus_native", {}).get("all", {})
            lines.append(f"  - All-size signed delta (ONNX − native): AP50 `{delta.get('map50')}`, AP50–95 `{delta.get('map50_95')}`; no pass threshold applied.")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- Hard validity failures (identity, split/XML membership, provider, non-finite output or incomplete records) block scientific interpretation.",
        "- A valid execution reports measured export drift only. Ordered examples do not perform rematching and do not replace the COCO/XML AP estimator.",
    ]
    return "\n".join(lines) + "\n"


def run_parent(args: argparse.Namespace, repo: Path) -> int:
    numeric.set_cpu_environment()
    models = selected_models(args.model)
    output_root = repo_path(repo, args.out_dir)
    if output_root.exists():
        raise FileExistsError(f"Bridge output exists; refusing overwrite/resume: {output_root}")
    if args.xml is None:
        raise ValueError("--xml is required: provide the raw CCTSDB XML archive used by the established COCO/XML estimator")
    args.xml = args.xml.expanduser().resolve()
    plan = build_plan(repo, args, models)
    output_root.mkdir(parents=True)
    write_json_no_overwrite(output_root / "bridge_plan.json", plan)
    model_rows = []
    failed = False
    script = Path(__file__).resolve()
    for model in models:
        model_dir = output_root / "models" / model
        model_dir.mkdir(parents=True)
        command = [sys.executable, str(script), "--child", "--repo-root", str(repo), "--model", model, "--plan", str(output_root / "bridge_plan.json")]
        result = subprocess.run(command, cwd=repo, env=os.environ.copy(), capture_output=True, text=True, check=False)
        (output_root / "logs").mkdir(parents=True, exist_ok=True)
        write_text_no_overwrite(output_root / "logs" / f"{model}.log", result.stdout + result.stderr)
        report_path = model_dir / "model_report.json"
        failure_path = model_dir / "failure.json"
        if report_path.is_file():
            row = read_json(report_path)
        elif failure_path.is_file():
            row = read_json(failure_path)
        else:
            row = child_failure(plan, model, new_child_state(model), RuntimeError(f"child exited {result.returncode} without a report"), output_root)
            write_json_no_overwrite(failure_path, row)
        if result.returncode != 0 or row.get("status") != "completed":
            failed = True
        model_rows.append(row)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY,
        "status": "failed" if failed else "completed",
        "validity": "blocked_not_interpretable" if failed else "validity_checks_passed; scientific_assessment_descriptive",
        "execution_status": "failed" if failed else "completed",
        "numeric_historical_verdict": "preserved_fail_not_replaced",
        "models": model_rows,
        "plan_path": "bridge_plan.json",
        "artifact_inventory": publishable_inventory(output_root),
        "artifact_inventory_scope": "parent_publishable_excludes_private",
        "call_contract": plan["call_contract"],
        "audit_flags": plan["audit_flags"],
        "run_inventory": {"owner": "parent", "scope": "whole_output_root_publishable", "files": publishable_inventory(output_root), "model_files": {model: child_owned_files(output_root, model) for model in models}, "log_files": [f"logs/{model}.log" for model in models]},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_no_overwrite(output_root / "bridge_manifest.json", manifest)
    write_text_no_overwrite(output_root / "report.md", _final_report(manifest))
    print(f"DONE: {output_root / 'bridge_manifest.json'}")
    return 1 if failed else 0


def run_child(args: argparse.Namespace) -> int:
    repo = args.repo_root.resolve()
    plan = read_json(repo_path(repo, Path(args.plan)))
    model = args.model
    output_root = repo_path(repo, Path(plan["output_root"]))
    out_dir = output_root / "models" / model
    state = new_child_state(model)
    try:
        report = run_model_child(repo, plan, model, out_dir, state=state)
        write_json_no_overwrite(out_dir / "model_report.json", report)
        print(f"DONE MODEL {model}: {out_dir / 'model_report.json'}", flush=True)
        return 0
    except Exception as exc:
        failure = child_failure(plan, model, state, exc, output_root)
        write_json_no_overwrite(out_dir / "failure.json", failure)
        print(f"FAILED MODEL {model}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-only native/accepted-ONNX CCTSDB2021 dev detection bridge")
    parser.add_argument("--readiness-root", type=Path, default=DEFAULT_READINESS_ROOT)
    parser.add_argument("--graph-audit-root", type=Path, default=DEFAULT_GRAPH_AUDIT_ROOT)
    parser.add_argument("--onnx-root", type=Path, default=DEFAULT_ONNX_ROOT)
    parser.add_argument("--dev-root", type=Path, default=DEFAULT_DEV_ROOT)
    parser.add_argument("--xml", type=Path, required=False, help="Raw CCTSDB XML archive used by the established COCO/XML estimator")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", choices=(*MODEL_CHOICES, "all"), default="all")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[1] if args.repo_root is None else args.repo_root.resolve()
    if args.child:
        if args.repo_root is None or args.plan is None or args.model == "all":
            raise ValueError("Internal child requires --repo-root, --plan and one model")
        return run_child(args)
    return run_parent(args, repo)


if __name__ == "__main__":
    raise SystemExit(main())
