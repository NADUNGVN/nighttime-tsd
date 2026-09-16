#!/usr/bin/env python3
"""CPU-only readiness for the YOLOv8n/YOLO26n confirmation protocol.

This module validates the frozen source contract and records a deterministic
84-invocation schedule.  It deliberately does not import TensorRT, call
``YOLO.export``, touch CUDA, build an engine, or run a scored matrix.  ONNX
export and graph-level mapping are deferred to the reviewed server prepare
phase; a successful local readiness report is therefore not a server-run
authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


STUDY = "precision_head_confirmation_v1"
PINNED_SOURCE_COMMIT = "53f12554761c80f367876756b4371a7729891d44"
DEFAULT_CONFIG = Path("configs/precision_head_confirmation_v1.json")
DEFAULT_OUTPUT = Path("results/measurement_audit_v1/precision_head_confirmation_readiness_v1")
ARM_NAMES = ("baseline_int8", "bbox_fp32", "classification_fp32", "both_fp32")
SELECTION_IDS = ("U42", "U43", "U44")
DEV_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
LOCKED_SECTION_SHA256 = {
    "models": "adfd8ce825b4625041d1c55ab65307364ad3a5561ce5b1a0297cb80818e28ac0",
    "calibration_selections": "56cb7d766e640dd73a3f88c784a1db756c11c9c55d2cac9ed240fb6b309cc8c6",
    "arms": "5edf62638f1973bf2b025988c49d8063ebe3096bb9b6544589deb1312a70b04b",
    "build_repeats": "a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4",
    "fp16_repeats": "a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4",
    "accounting": "6dcc615fc623904cd6d0908b13be4d0c46f0578d1f132f368ce7c95dcf35dd57",
    "runtime": "d5146a2c9cd01835412f70a4423f22f2f9517cec5fe9c124b47587981ea144ef",
    "calibration_recipe": "ce8a7b835a41b739b9971af18f4f2c0c342dcf65643a72596ccbf5b07ddcf712",
    "export_contract": "b1e5cd4c44c4a5b34e7aaf3240fbd900eee72a27beaf00dbfcd6749efe22f4c9",
    "execution_boundary": "f2b82ca16d6998e78e6f192dc9636568251f3eb4dd64fdff91a05fafadbadacc",
    "schedule": "66c8cc7ecae1b5db22feb8b973dcff55085119605663f1e95452e246b716c5cc",
    "decision_rules": "f514aef9ee7de1fc9fba9343293199326a4538ef87efc905d91c9798f4049e0f",
    "dev_contract": "21ea84e81bf91a0b5b0f6acf35dd1dbe14da36ae7fd92af1d72d5e99196d5c87",
    "canonical_dev_reference": "e9ad2e8ccc2fa21c74d300865db73d269ff7caec8298621da1ab8dbbb2bf84be",
}
LOCKED_CONFIG_SECTIONS = tuple(LOCKED_SECTION_SHA256)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_no_overwrite(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def write_text_no_overwrite(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def relative_git_path(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is outside repository: {path}") from exc


def git_blob(repo: Path, commit: str, path: Path) -> bytes:
    relative = relative_git_path(repo, path)
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{commit}:{relative}"],
        cwd=repo, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"Missing canonical input at {commit}: {relative}")
    return result.stdout


def canonical_sha256(repo: Path, commit: str, path: Path) -> str:
    return sha256_bytes(git_blob(repo, commit, path))


def load_config(config_path: Path, repo: Path | None = None) -> dict[str, Any]:
    config_path = config_path.resolve()
    repo = (repo or config_path.parent.parent).resolve()
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("Readiness config must be a JSON object")
    if config.get("schema_version") != 1 or config.get("study") != STUDY:
        raise ValueError("Readiness config schema/study is not locked")
    if config.get("canonical_input_commit") != PINNED_SOURCE_COMMIT:
        raise ValueError("Readiness config must pin the reviewed source commit")
    for section, expected_hash in LOCKED_SECTION_SHA256.items():
        if section not in config or sha256_bytes(canonical_json(config[section])) != expected_hash:
            raise ValueError(f"Readiness config immutable section hash mismatch: {section}")
    if config.get("canonical_dev_reference_commit") != "5eb7ec36da7eed1701f6383b9db37ca3cfe31186":
        raise ValueError("Readiness config must pin the accepted FP16 dev reference commit")
    if tuple(config.get("arms", ())) != ARM_NAMES:
        raise ValueError("Readiness config arms do not match the locked four-arm contract")
    if config.get("build_repeats") != [1, 2, 3] or config.get("fp16_repeats") != [1, 2, 3]:
        raise ValueError("Readiness config repeats must be exactly 1, 2, 3")
    models = config.get("models")
    if not isinstance(models, list) or [m.get("label") for m in models] != ["yolov8n", "yolo26n"]:
        raise ValueError("Readiness config must contain exactly YOLOv8n and YOLO26n")
    selections = config.get("calibration_selections")
    if not isinstance(selections, list) or [s.get("id") for s in selections] != list(SELECTION_IDS):
        raise ValueError("Readiness config selections must be U42, U43, U44 in order")
    accounting = config.get("accounting", {})
    expected_accounting = {
        "auxiliary_calibration_builds": 6,
        "scored_int8_builds": 72,
        "scored_fp16_builds": 6,
        "scored_builds": 78,
        "total_builder_invocations": 84,
        "captures": 78,
        "onnx_exports": 2,
        "dev_images": 1636,
        "dev_instances": 2706,
        "image_passes": 127608,
    }
    if accounting != expected_accounting:
        raise ValueError(f"Accounting mismatch: {accounting!r}")
    runtime = config.get("runtime", {})
    for key, value in {"imgsz": 640, "batch": 1, "task": "detect", "rect": False, "workers": 0}.items():
        if runtime.get(key) != value:
            raise ValueError(f"Runtime field {key} is not locked to {value!r}")
    schedule = config.get("schedule", {})
    if schedule.get("version") != "model_block_aux_then_interleaved_rounds_v2":
        raise ValueError("Unknown schedule version")
    if schedule.get("model_order") != ["yolov8n", "yolo26n"]:
        raise ValueError("Model schedule order is not locked")
    if schedule.get("selection_order") != list(SELECTION_IDS):
        raise ValueError("Selection schedule order is not locked")
    if schedule.get("arm_order") != list(ARM_NAMES):
        raise ValueError("Arm schedule order is not locked")
    if config["calibration_recipe"].get("preprocessing_status") != "unresolved_pending_server_prepare":
        raise ValueError("Calibration preprocessing must remain unresolved until server prepare evidence")
    return config


def normalize_split_relative(value: Any, split: str, kind: str) -> str:
    """Accept only ``split/<kind>/<basename>`` POSIX paths."""
    if not isinstance(value, str) or "\\" in value:
        raise ValueError(f"{kind} path must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 3 or path.parts[0:2] != (split, kind):
        raise ValueError(f"{kind} path is not strictly {split}/{kind}/basename: {value!r}")
    if path.parts[-1] in {"", ".", ".."} or ".." in path.parts:
        raise ValueError(f"{kind} path contains traversal: {value!r}")
    if path.name != path.parts[-1] or path.name != Path(path.name).name:
        raise ValueError(f"{kind} path is not a basename: {value!r}")
    return path.as_posix()


def validate_yolo_label_text(text: str, path_label: str) -> dict[str, Any]:
    """Validate YOLO normalized labels without reading any test data."""
    count = 0
    classes: list[int] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label field count at {path_label}:{line_number}")
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"Invalid numeric YOLO label at {path_label}:{line_number}") from exc
        if class_id not in {0, 1, 2} or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid class/non-finite YOLO label at {path_label}:{line_number}")
        center_x, center_y, width, height = values
        if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"Invalid normalized YOLO bbox at {path_label}:{line_number}")
        # Keep the producer's normalized center/size contract.  Rounded source
        # annotations can place a derived corner a few ulps outside [0, 1];
        # silently clipping or rejecting those corners would change the
        # accepted label inventory.  Corner clipping remains the evaluator's
        # coordinate responsibility and is recorded by downstream validators.
        classes.append(class_id)
        count += 1
    return {"count": count, "class_ids": classes}


def validate_calibration_payload(payload: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """Validate the canonical producer schema and normalized train IDs."""
    if not isinstance(payload, dict):
        raise ValueError("Calibration manifest must be an object")
    if payload.get("strategy") != "uniform":
        raise ValueError("Calibration manifest strategy is not uniform")
    if payload.get("seed") != selection["seed"]:
        raise ValueError("Calibration seed does not match selection")
    if payload.get("requested_size") != 1024 or payload.get("selected_size") != 1024:
        raise ValueError("Calibration size is not 1024")
    purpose = payload.get("purpose")
    if not isinstance(purpose, str) or "train/images" not in purpose or "calibration" not in purpose.lower():
        raise ValueError("Calibration purpose is not an explicit train-only calibration purpose")
    files = payload.get("files")
    if not isinstance(files, list) or len(files) != 1024:
        raise ValueError("Calibration manifest must contain exactly 1024 selected files")
    source_images: list[str] = []
    source_labels: list[str] = []
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("Calibration file entry must be an object")
        image = normalize_split_relative(row.get("source_image"), "train", "images")
        label = normalize_split_relative(row.get("source_label"), "train", "labels")
        if Path(image).stem != Path(label).stem:
            raise ValueError(f"Calibration image/label stems differ: {image!r}, {label!r}")
        source_images.append(image)
        source_labels.append(label)
    if len(set(source_images)) != 1024 or len(set(source_labels)) != 1024:
        raise ValueError("Calibration manifest contains duplicate image or label IDs")
    return {
        "status": "canonical_manifest_valid",
        "seed": selection["seed"],
        "selected_size": len(source_images),
        "train_only": True,
        "image_ids": source_images,
        "label_ids": source_labels,
    }


def file_inside_root(path: Path, root: Path, label: str, reject_symlink: bool = True) -> Path:
    if reject_symlink and path.is_symlink():
        raise ValueError(f"{label} is a symlink; expected a file in the intended split")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    resolved_root = root.resolve()
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes intended root: {path}") from exc
    return resolved


def _canonical_dev_reference(repo: Path, config: dict[str, Any]) -> dict[str, Any]:
    reference = config["canonical_dev_reference"]
    commit = config["canonical_dev_reference_commit"]
    report_path = repo / reference["capture_report"]
    predictions_path = repo / reference["predictions"]
    report_bytes = git_blob(repo, commit, report_path)
    predictions_bytes = git_blob(repo, commit, predictions_path)
    report = json.loads(report_bytes.decode("utf-8"))
    predictions = json.loads(predictions_bytes.decode("utf-8"))
    if report.get("dataset_split") != reference["dataset_split"]:
        raise ValueError("Canonical dev capture split mismatch")
    if report.get("images") != reference["images"] or report.get("instances") != reference["instances"]:
        raise ValueError("Canonical dev capture count mismatch")
    if report.get("predictions_sha256") != reference["predictions_sha256"]:
        raise ValueError("Canonical dev capture prediction hash mismatch")
    if sha256_bytes(predictions_bytes) != reference["predictions_sha256"]:
        raise ValueError("Canonical prediction bytes do not match capture report")
    records = predictions.get("records") if isinstance(predictions, dict) else None
    if not isinstance(records, list) or len(records) != reference["images"]:
        raise ValueError("Canonical dev prediction records do not contain 1636 images")
    normalized = []
    seen = set()
    for row in records:
        image = row.get("image") if isinstance(row, dict) else None
        shape = row.get("orig_shape") if isinstance(row, dict) else None
        if not isinstance(image, str) or Path(image).name != image or Path(image).suffix.lower() not in DEV_IMAGE_EXTENSIONS:
            raise ValueError(f"Canonical dev image ID is malformed: {image!r}")
        if image in seen or not isinstance(shape, list) or len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
            raise ValueError(f"Canonical dev image ID/shape is invalid: {image!r}")
        seen.add(image)
        normalized.append({"image": image, "stem": Path(image).stem, "orig_shape": shape})
    if [row["image"] for row in normalized] != sorted(row["image"] for row in normalized):
        raise ValueError("Canonical dev image order is not sorted")
    return {
        "commit": commit,
        "capture_report": reference["capture_report"],
        "capture_report_sha256": sha256_bytes(report_bytes),
        "predictions": reference["predictions"],
        "predictions_sha256": sha256_bytes(predictions_bytes),
        "dataset_split": report["dataset_split"],
        "images": report["images"],
        "instances": report["instances"],
        "records": normalized,
        "ordered_ids_sha256": sha256_bytes(canonical_json([row["image"] for row in normalized])),
        "shape_reference_sha256": sha256_bytes(canonical_json({row["image"]: row["orig_shape"] for row in normalized})),
        "hash_scope": "canonical accepted FP16 artifact bytes; not current image-byte hashes",
    }


def parse_materialized_calibration_yaml(
    yaml_path: Path,
    expected_dir: Path,
    rows: list[dict[str, Any]],
    source_root: Path,
) -> dict[str, Any]:
    """Validate a producer YAML and all materialized image bytes if present."""
    if not yaml_path.is_file():
        return {
            "status": "missing_materialization",
            "yaml": str(yaml_path),
            "yaml_exists": False,
            "missing": ["materialized_calibration_yaml"],
            "errors": [],
        }
    try:
        import yaml
        document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("calibration YAML is not a mapping")
        if document.get("train") != "images" or document.get("val") != "images":
            raise ValueError("calibration YAML train/val must point to images")
        if "test" in document and document.get("test") != "images":
            raise ValueError("calibration YAML test must point to images when present")
        if document.get("nc") != 3:
            raise ValueError("calibration YAML nc is not 3")
        names = document.get("names")
        expected_names = {0: "prohibitory", 1: "mandatory", 2: "warning"}
        if isinstance(names, list):
            names = {index: value for index, value in enumerate(names)}
        if names != expected_names:
            raise ValueError("calibration YAML class names differ from locked CCTSDB names")
        declared = document.get("path")
        if not isinstance(declared, str) or "\\" in declared:
            raise ValueError("calibration YAML path must be a POSIX path")
        declared_posix = PurePosixPath(declared)
        expected_name = expected_dir.name
        if declared_posix.name != expected_name:
            raise ValueError("calibration YAML path does not identify the selected calibration directory")
        materialized_dir = expected_dir
        image_dir = materialized_dir / "images"
        materialized_names = sorted(path.name for path in image_dir.glob("*") if path.is_file()) if image_dir.is_dir() else []
        expected_names = sorted(Path(row["source_image"]).name for row in rows)
        if materialized_names != expected_names:
            raise ValueError(f"materialized calibration image inventory mismatch: {len(materialized_names)} != {len(expected_names)}")
        byte_records = []
        for row in rows:
            name = Path(row["source_image"]).name
            source = file_inside_root(source_root / row["source_image"], source_root / "train" / "images", f"source calibration image {name}")
            materialized = file_inside_root(image_dir / name, image_dir, f"materialized calibration image {name}")
            source_hash = sha256_file(source)
            materialized_hash = sha256_file(materialized)
            if source_hash != materialized_hash:
                raise ValueError(f"materialized calibration image bytes differ for {name}")
            byte_records.append({"image": row["source_image"], "source_sha256": source_hash, "materialized_sha256": materialized_hash, "bytes": source.stat().st_size})
        return {
            "status": "complete",
            "yaml": str(yaml_path),
            "yaml_exists": True,
            "yaml_sha256": sha256_file(yaml_path),
            "resolved_directory": str(materialized_dir),
            "image_count": len(byte_records),
            "image_bytes": byte_records,
            "missing": [],
            "errors": [],
        }
    except (OSError, ValueError, ImportError, UnicodeError) as exc:
        return {
            "status": "invalid",
            "yaml": str(yaml_path),
            "yaml_exists": True,
            "missing": [],
            "errors": [f"{type(exc).__name__}:{exc}"],
        }


def validate_calibration_manifest(
    repo: Path,
    commit: str,
    selection: dict[str, Any],
    dataset_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_path = (repo / selection["manifest"]).resolve()
    record: dict[str, Any] = {"id": selection["id"], "manifest": selection["manifest"], "errors": [], "status": "invalid"}
    try:
        observed_sha = canonical_sha256(repo, commit, manifest_path)
        record["canonical_sha256"] = observed_sha
        if observed_sha != selection["canonical_sha256"]:
            raise ValueError(f"{selection['id']} canonical manifest hash mismatch")
        payload = json.loads(git_blob(repo, commit, manifest_path).decode("utf-8"))
        contract = validate_calibration_payload(payload, selection)
        record["manifest_contract"] = contract
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        record["errors"].append(f"{type(exc).__name__}:{exc}")
        record["manifest_contract"] = {"status": "invalid"}
        record["materialization"] = {"status": "invalid", "missing": [], "errors": record["errors"]}
        return record

    rows = payload["files"]
    source_root = repo / "data/processed/cctsdb2021_clean"
    train_image_ids = set(dataset_contract.get("train_image_ids", ())) if dataset_contract else set()
    dev_ids = set(dataset_contract.get("dev_image_ids", ())) if dataset_contract else set()
    test_ids = set(dataset_contract.get("test_image_ids", ())) if dataset_contract else set()
    selected_ids = {Path(row["source_image"]).stem for row in rows}
    selected_missing = sorted(selected_ids - train_image_ids) if train_image_ids else []
    overlap_dev = sorted(selected_ids & dev_ids)
    overlap_test = sorted(selected_ids & test_ids)
    if selected_missing:
        record["errors"].append(f"selected_train_ids_missing:{len(selected_missing)}")
    if overlap_dev or overlap_test:
        record["errors"].append(f"split_overlap:dev={len(overlap_dev)},test={len(overlap_test)}")
    selected_bytes = []
    for row in rows:
        try:
            image = file_inside_root(source_root / row["source_image"], source_root / "train" / "images", row["source_image"])
            label = file_inside_root(source_root / row["source_label"], source_root / "train" / "labels", row["source_label"])
            selected_bytes.append({
                "image": row["source_image"], "label": row["source_label"],
                "image_sha256": sha256_file(image), "label_sha256": sha256_file(label),
                "image_bytes": image.stat().st_size, "label_bytes": label.stat().st_size,
            })
        except (OSError, ValueError) as exc:
            record["errors"].append(f"selected_source_invalid:{type(exc).__name__}:{exc}")
            break
    record["selection_audit"] = {
        "selected_ids": sorted(selected_ids),
        "selected_train_ids_missing": selected_missing,
        "dev_overlap_ids": overlap_dev,
        "test_overlap_ids": overlap_test,
        "source_bytes": selected_bytes,
        "hash_scope": "current checkout source bytes, not historical server image-byte hashes",
    }
    materialized_yaml = manifest_path.parent / "calibration.yaml"
    materialization = parse_materialized_calibration_yaml(materialized_yaml, manifest_path.parent, rows, source_root)
    record["materialization"] = materialization
    record["status"] = "verified" if not record["errors"] and materialization["status"] == "complete" else ("missing" if not record["errors"] and materialization["status"] == "missing_materialization" else "invalid")
    return record


def calibration_overlap(records: list[dict[str, Any]]) -> dict[str, Any]:
    sets = {
        record["id"]: set(record.get("manifest_contract", {}).get("image_ids", ()))
        for record in records
    }
    pairs = {}
    ids = list(sets)
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            pairs[f"{left}:{right}"] = len(sets[left] & sets[right])
    return {"pairwise_image_id_overlap": pairs, "selection_sets_disjoint": all(value == 0 for value in pairs.values())}


def summarize_tensor(value: Any) -> dict[str, Any] | None:
    shape = getattr(value, "shape", None)
    if shape is None or not hasattr(value, "dtype"):
        return None
    return {"kind": "tensor", "shape": [int(item) for item in shape], "dtype": str(value.dtype)}


def summarize_output(value: Any, depth: int = 0) -> dict[str, Any]:
    tensor = summarize_tensor(value)
    if tensor is not None:
        return tensor
    if depth > 3:
        return {"kind": type(value).__name__, "truncated": True}
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "keys": list(value.keys()),
            "values": {str(key): summarize_output(item, depth + 1) for key, item in value.items()},
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "length": len(value),
            "items": [summarize_output(item, depth + 1) for item in value],
        }
    return {"kind": type(value).__name__}


def derive_active_branches(end2end: bool, available: list[str]) -> tuple[list[str], list[str]]:
    preferred = ["one2one_cv2", "one2one_cv3"] if end2end else ["cv2", "cv3"]
    alternate = ["cv2", "cv3"] if end2end else ["one2one_cv2", "one2one_cv3"]
    active = [name for name in preferred if name in available]
    inactive = [name for name in alternate if name in available]
    if len(active) != 2:
        raise ValueError(f"Ambiguous or incomplete active head branches: {available!r}")
    return active, inactive


def collect_branch_convolutions(torch_module: Any, branch: Any, prefix: str) -> list[dict[str, Any]]:
    result = []
    for name, module in branch.named_modules():
        if isinstance(module, torch_module.nn.Conv2d):
            full_name = f"{prefix}.{name}" if name else prefix
            result.append({
                "name": full_name,
                "module_type": type(module).__name__,
                "in_channels": int(module.in_channels),
                "out_channels": int(module.out_channels),
                "groups": int(module.groups),
                "kernel_size": [int(item) for item in module.kernel_size],
                "stride": [int(item) for item in module.stride],
            })
    if not result:
        raise ValueError(f"No Conv2d leaves found under active branch {prefix}")
    return result


def validate_model_contract(observed: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    errors = []
    for key in (
        "head_type", "head_index", "end2end", "active_branches", "inactive_branches",
        "output_kind", "primary_output_shape", "primary_output_dtype", "debug_paths",
        "boxes_shape", "scores_shape", "feature_shapes", "postprocess", "nms",
        "primary_output_representation",
    ):
        if observed.get(key) != expected.get(key):
            errors.append(f"{key}: expected {expected.get(key)!r}, observed {observed.get(key)!r}")
    if len(observed.get("active_convolution_mapping", {})) != 2:
        errors.append("active_convolution_mapping must contain exactly two active branches")
    if set(observed.get("active_convolution_mapping", {})) != set(expected.get("active_branches", ())):
        errors.append("active_convolution_mapping branches do not equal active_branches")
    if observed.get("mapping_status") != "pytorch_structure_verified_onnx_deferred":
        errors.append("mapping_status does not state the deferred ONNX boundary")
    return errors


def inspect_frozen_model(repo: Path, model_config: dict[str, Any], probe: bool = True) -> dict[str, Any]:
    checkpoint = (repo / model_config["checkpoint"]).resolve()
    result: dict[str, Any] = {
        "label": model_config["label"],
        "checkpoint": model_config["checkpoint"],
        "expected_sha256": model_config["sha256"],
        "expected_bytes": model_config["bytes"],
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
        "device": "cpu",
        "probe_requested": probe,
        "export_performed": False,
        "tensorrt_imported": False,
        "gpu_used": False,
        "mapping_status": "not_observed",
    }
    errors = []
    if not checkpoint.is_file():
        errors.append("checkpoint_missing")
    elif result["checkpoint_sha256"] != model_config["sha256"] or result["checkpoint_bytes"] != model_config["bytes"]:
        errors.append("checkpoint_hash_or_size_mismatch")
    if errors or not probe:
        result["status"] = "unresolved"
        result["errors"] = errors or ["model_probe_not_run"]
        result["probe_status"] = "not_run" if not probe else "blocked_by_checkpoint_error"
        return result

    try:
        import torch
        from ultralytics import YOLO

        loaded = YOLO(str(checkpoint), task="detect")
        network = loaded.model.to("cpu").eval()
        parameters_on_cpu = all(getattr(parameter, "device", None).type == "cpu" for parameter in network.parameters())
        head = network.model[-1]
        head_index = int(getattr(head, "i"))
        head_type = type(head).__name__
        end2end = bool(getattr(head, "end2end"))
        available = [name for name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3") if hasattr(head, name)]
        active, inactive = derive_active_branches(end2end, available)
        mapping = {
            branch_name: collect_branch_convolutions(
                torch, getattr(head, branch_name), f"model.{head_index}.{branch_name}"
            )
            for branch_name in active
        }
        import contextlib
        with torch.no_grad():
            output = network(torch.zeros((1, 3, 640, 640), dtype=torch.float32, device="cpu"))
        summary = summarize_output(output)
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise ValueError("Frozen model output is not the expected primary-plus-head tuple")
        primary = summarize_tensor(output[0])
        debug = output[1]
        if primary is None or not isinstance(debug, dict):
            raise ValueError("Frozen model output representation is not tensor-plus-dict")

        if end2end:
            output_kind = "end2end_topk_detections_plus_head_dict"
            postprocess = "topk_300_detections_in_model_head_output"
            nms = "postprocess_only_as_defined_by_locked_ultralytics_runtime"
            expected_debug = debug.get("one2one", {})
            boxes = expected_debug.get("boxes") if isinstance(expected_debug, dict) else None
            scores = expected_debug.get("scores") if isinstance(expected_debug, dict) else None
            feats = expected_debug.get("feats") if isinstance(expected_debug, dict) else None
        else:
            output_kind = "decoded_raw_plus_head_dict"
            postprocess = "none_in_model_head_output"
            nms = "not_in_model_head_output"
            boxes = debug.get("boxes")
            scores = debug.get("scores")
            feats = debug.get("feats")
        tensor_shape = lambda item: [int(value) for value in item.shape] if summarize_tensor(item) else None
        observed = {
            "head_type": head_type,
            "head_index": head_index,
            "end2end": end2end,
            "active_branches": active,
            "inactive_branches": inactive,
            "output_kind": output_kind,
            "primary_output_shape": primary["shape"],
            "primary_output_dtype": primary["dtype"],
            "debug_paths": list(debug.keys()),
            "boxes_shape": tensor_shape(boxes),
            "scores_shape": tensor_shape(scores),
            "feature_shapes": [tensor_shape(item) for item in feats] if isinstance(feats, (tuple, list)) else None,
            "postprocess": postprocess,
            "nms": nms,
            "primary_output_representation": "tuple_tensor_plus_dict",
            "active_convolution_mapping": mapping,
            "mapping_status": "pytorch_structure_verified_onnx_deferred",
        }
        errors = validate_model_contract(observed, model_config["expected_contract"])
        mapping_hash = sha256_bytes(canonical_json(mapping))
        result.update({
            "status": "verified" if not errors and parameters_on_cpu else "unresolved",
            "errors": errors + ([] if parameters_on_cpu else ["parameter_not_on_cpu"]),
            "head": {"type": head_type, "index": head_index, "end2end": end2end, "available_branches": available},
            "active_branches": active,
            "inactive_branches": inactive,
            "active_convolution_mapping": mapping,
            "mapping_hash": mapping_hash,
            "mapping_status": observed["mapping_status"],
            "forward_summary": summary,
            "head_output_evidence": observed,
            "probe": {
                "method": "Ultralytics frozen PyTorch model CPU eval/no_grad dummy input",
                "input_shape": [1, 3, 640, 640],
                "input_dtype": "torch.float32",
                "parameters_on_cpu": parameters_on_cpu,
                "cuda_calls": False,
            },
            "constraints": {
                "bbox_fp32_requested": "deferred_to_server_onnx_prepare",
                "classification_fp32_requested": "deferred_to_server_onnx_prepare",
                "effective_precision": "unknown",
            },
        })
    except Exception as exc:  # dependency/model incompatibility is an unresolved readiness result
        result.update({
            "status": "unresolved",
            "errors": [f"cpu_probe_failed:{type(exc).__name__}:{exc}"],
            "probe_error_type": type(exc).__name__,
            "mapping_status": "unresolved",
        })
    return result


def _directory_ids(directory: Path, extensions: set[str]) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def _ids_hash(values: list[str]) -> str:
    return sha256_bytes(canonical_json(values))


def validate_dev_inventory_identity(
    current_image_names: list[str],
    current_label_names: list[str],
    current_shapes: dict[str, list[int] | None],
    canonical_records: list[dict[str, Any]],
) -> list[str]:
    """Compare IDs, label stems and decoded shapes with accepted FP16 evidence."""
    errors: list[str] = []
    canonical_names = [row["image"] for row in canonical_records]
    if current_image_names != canonical_names:
        errors.append("dev_image_ids_differ_from_canonical_reference")
    current_stems = [Path(name).stem for name in current_image_names]
    label_stems = [Path(name).stem for name in current_label_names]
    if len(set(current_stems)) != len(current_stems) or current_stems != sorted(current_stems):
        errors.append("dev_image_stems_are_not_unique_sorted")
    if set(current_stems) != set(label_stems):
        errors.append("dev_image_label_stem_set_mismatch")
    canonical_by_name = {row["image"]: row for row in canonical_records}
    for name, shape in current_shapes.items():
        expected = canonical_by_name.get(name)
        if expected is None:
            errors.append(f"dev_image_not_in_canonical_reference:{name}")
        elif shape != expected["orig_shape"]:
            errors.append(f"dev_shape_differs_from_canonical_reference:{name}")
    return errors


def validate_dataset_contract(repo: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Audit current dev files against the canonical accepted FP16 reference."""
    expected = config["dev_contract"]
    errors: list[str] = []
    try:
        reference = _canonical_dev_reference(repo, config)
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "status": "unresolved",
            "split": expected["split"],
            "images_observed": 0,
            "labels_observed": 0,
            "instances_observed": 0,
            "expected": expected,
            "errors": [f"canonical_dev_reference:{type(exc).__name__}:{exc}"],
            "yaml_reference": expected["yaml_reference"],
            "canonical_reference": {"status": "unresolved"},
        }

    root = repo / "data/processed/cctsdb2021_clean"
    images_dir = root / "dev" / "images"
    labels_dir = root / "dev" / "labels"
    train_images_dir = root / "train" / "images"
    train_labels_dir = root / "train" / "labels"
    test_images_dir = root / "test" / "images"
    image_paths = sorted(path for path in images_dir.glob("*") if path.is_file() and path.suffix.lower() in DEV_IMAGE_EXTENSIONS)
    label_paths = sorted(path for path in labels_dir.glob("*.txt") if path.is_file())
    current_image_names = [path.name for path in image_paths]
    current_label_names = [path.name for path in label_paths]
    current_image_stems = [Path(name).stem for name in current_image_names]
    current_label_stems = [Path(name).stem for name in current_label_names]

    try:
        from PIL import Image
    except ImportError as exc:
        Image = None
        errors.append(f"image_decoder_missing:{exc}")
    current_shapes: dict[str, list[int] | None] = {}
    inventory: list[dict[str, Any]] = []
    instances = 0
    for image_path in image_paths:
        image_name = image_path.name
        label_path = labels_dir / f"{image_path.stem}.txt"
        row: dict[str, Any] = {"image": image_name, "label": label_path.name}
        try:
            image_resolved = file_inside_root(image_path, images_dir, f"dev image {image_name}")
            row["image_sha256"] = sha256_file(image_resolved)
            row["image_bytes"] = image_resolved.stat().st_size
            if Image is None:
                raise ValueError("image decoder unavailable")
            with Image.open(image_resolved) as decoded:
                decoded.load()
                row["orig_shape"] = [int(decoded.height), int(decoded.width)]
        except (OSError, ValueError) as exc:
            errors.append(f"dev_image_invalid:{image_name}:{type(exc).__name__}:{exc}")
            row["orig_shape"] = None
        try:
            label_resolved = file_inside_root(label_path, labels_dir, f"dev label {label_path.name}")
            label_text = label_resolved.read_text(encoding="utf-8")
            label_audit = validate_yolo_label_text(label_text, label_path.name)
            row["label_sha256"] = sha256_file(label_resolved)
            row["label_bytes"] = label_resolved.stat().st_size
            row["label_count"] = label_audit["count"]
            row["class_ids"] = label_audit["class_ids"]
            instances += label_audit["count"]
        except (OSError, ValueError, UnicodeError) as exc:
            errors.append(f"dev_label_invalid:{label_path.name}:{type(exc).__name__}:{exc}")
            row["label_count"] = None
            row["class_ids"] = None
        current_shapes[image_name] = row.get("orig_shape")
        inventory.append(row)
    errors.extend(validate_dev_inventory_identity(current_image_names, current_label_names, current_shapes, reference["records"]))
    if len(image_paths) != expected["images"]:
        errors.append(f"dev_image_count:{len(image_paths)}!={expected['images']}")
    if len(label_paths) != expected["images"]:
        errors.append(f"dev_label_count:{len(label_paths)}!={expected['images']}")
    if instances != expected["instances"]:
        errors.append(f"dev_instance_count:{instances}!={expected['instances']}")

    train_image_ids = [Path(name).stem for name in _directory_ids(train_images_dir, DEV_IMAGE_EXTENSIONS)]
    train_label_ids = [Path(name).stem for name in _directory_ids(train_labels_dir, {".txt"})]
    test_image_ids = [Path(name).stem for name in _directory_ids(test_images_dir, DEV_IMAGE_EXTENSIONS)]
    if set(train_image_ids) != set(train_label_ids):
        errors.append("train_image_label_stem_set_mismatch")
    dev_ids = set(current_image_stems)
    test_ids = set(test_image_ids)
    if dev_ids & test_ids:
        errors.append(f"dev_test_id_overlap:{len(dev_ids & test_ids)}")
    inventory_hash = sha256_bytes(canonical_json(inventory))
    return {
        "status": "verified" if not errors else "unresolved",
        "split": expected["split"],
        "images_observed": len(image_paths),
        "labels_observed": len(label_paths),
        "instances_observed": instances,
        "expected": expected,
        "errors": errors,
        "yaml_reference": expected["yaml_reference"],
        "canonical_reference": reference,
        "inventory": inventory,
        "inventory_sha256": inventory_hash,
        "inventory_hash_scope": "current checkout dev image/label bytes and decoded shapes; not historical server image-byte hashes",
        "dev_image_ids": current_image_stems,
        "train_image_ids": train_image_ids,
        "test_image_ids": test_image_ids,
        "train_inventory": {
            "images": len(train_image_ids),
            "labels": len(train_label_ids),
            "image_ids_sha256": _ids_hash(train_image_ids),
            "label_ids_sha256": _ids_hash(train_label_ids),
        },
        "test_exclusion_inventory": {
            "images": len(test_image_ids),
            "image_ids_sha256": _ids_hash(test_image_ids),
            "read_mode": "IDs only; no test pixels or labels read",
        },
        "dev_test_overlap": sorted(dev_ids & test_ids),
    }


def generate_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    schedule = []
    sequence = 1
    for model in config["schedule"]["model_order"]:
        for selection in config["schedule"]["selection_order"]:
            schedule.append({
                "sequence": sequence, "round": None, "model": model, "selection": selection,
                "phase": "auxiliary_calibration", "arm": "baseline_int8", "repeat": None,
                "scored": False, "capture_required": False, "calibration_cache_creation": True,
                "timing_cache_policy": "private_not_reused_by_scored",
            })
            sequence += 1
        canonical_cells = [
            {"selection": None, "phase": "scored_fp16", "arm": "fp16"},
            *[
                {"selection": selection, "phase": "scored_int8", "arm": arm}
                for selection in config["schedule"]["selection_order"]
                for arm in config["schedule"]["arm_order"]
            ],
        ]
        if len(canonical_cells) != 13:
            raise ValueError("Canonical interleaved schedule must contain 13 scored cells")
        for round_id, shift in enumerate((0, 4, 8), start=1):
            for cell in canonical_cells[shift:] + canonical_cells[:shift]:
                schedule.append({
                    "sequence": sequence, "round": round_id, "model": model,
                    "selection": cell["selection"], "phase": cell["phase"],
                    "arm": cell["arm"], "repeat": round_id,
                    "scored": True, "capture_required": True,
                    "calibration_cache_creation": False,
                    "timing_cache_policy": "fresh_empty_private",
                })
                sequence += 1
    return schedule


def validate_schedule(schedule: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if len(schedule) != 84:
        raise ValueError(f"Schedule length is {len(schedule)}, expected 84")
    if [row["sequence"] for row in schedule] != list(range(1, 85)):
        raise ValueError("Schedule sequence is not contiguous")
    expected = generate_schedule(config)
    exact_fields = (
        "sequence", "round", "model", "selection", "phase", "arm", "repeat", "scored",
        "capture_required", "calibration_cache_creation", "timing_cache_policy",
    )
    for index, (actual, expected_row) in enumerate(zip(schedule, expected), start=1):
        for field in exact_fields:
            if actual.get(field) != expected_row.get(field):
                raise ValueError(f"Schedule job {index} field {field} differs from canonical order")
    keys = [(row["model"], row["selection"], row["arm"], row["repeat"], row["round"]) for row in schedule]
    if len(keys) != len(set(keys)):
        raise ValueError("Schedule contains duplicate cells/jobs")
    aux = [row for row in schedule if row["phase"] == "auxiliary_calibration"]
    int8 = [row for row in schedule if row["phase"] == "scored_int8"]
    fp16 = [row for row in schedule if row["phase"] == "scored_fp16"]
    if len(aux) != 6 or len(int8) != 72 or len(fp16) != 6:
        raise ValueError("Schedule accounting is not 6 auxiliary + 72 INT8 + 6 FP16")
    if any(row["scored"] or row["capture_required"] for row in aux):
        raise ValueError("Auxiliary cache builds must not be scored or captured")
    if any(not row["scored"] or not row["capture_required"] for row in int8 + fp16):
        raise ValueError("Every scored build must require one capture")
    for model in config["schedule"]["model_order"]:
        block = [row for row in schedule if row["model"] == model]
        if len(block) != 42:
            raise ValueError(f"Model block {model} does not contain 42 invocations")
        aux_sequences = {row["selection"]: row["sequence"] for row in block if row["phase"] == "auxiliary_calibration"}
        for selection in config["schedule"]["selection_order"]:
            if selection not in aux_sequences:
                raise ValueError(f"Missing auxiliary build for {model}/{selection}")
            if any(row["sequence"] <= aux_sequences[selection] for row in block if row["phase"] == "scored_int8" and row["selection"] == selection):
                raise ValueError(f"Scored build precedes auxiliary build for {model}/{selection}")
        scored = [row for row in block if row["scored"]]
        for round_id, shift in enumerate((0, 4, 8), start=1):
            observed = [(row["selection"], row["arm"]) for row in scored if row["round"] == round_id]
            canonical = [(row["selection"], row["arm"]) for row in expected if row["model"] == model and row["round"] == round_id]
            if observed != canonical:
                raise ValueError(f"Round {round_id} order does not match canonical rotation {shift}")
    return {
        "status": "verified",
        "jobs": len(schedule),
        "auxiliary_calibration_builds": len(aux),
        "scored_int8_builds": len(int8),
        "scored_fp16_builds": len(fp16),
        "scored_builds": len(int8) + len(fp16),
        "captures": sum(1 for row in schedule if row["capture_required"]),
        "round_rotation": [0, 4, 8],
        "schedule_sha256": sha256_bytes(canonical_json(schedule)),
    }


def environment_evidence() -> dict[str, Any]:
    values: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": "not_observed_cpu_readiness_only",
        "cuda": "not_observed_cpu_readiness_only",
        "tensorrt": "not_imported_by_design",
    }
    for package in ("torch", "ultralytics", "numpy", "pycocotools"):
        try:
            values[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            values[package] = "unknown"
    return values


def git_head(repo: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def config_semantic_hash(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json({section: config[section] for section in LOCKED_CONFIG_SECTIONS}))


def build_readiness(
    repo: Path,
    config: dict[str, Any],
    probe_models: bool = True,
    config_path: Path | None = None,
) -> dict[str, Any]:
    config_path = (config_path or repo / DEFAULT_CONFIG).resolve()
    environment = environment_evidence()
    model_records = [inspect_frozen_model(repo, model, probe=probe_models) for model in config["models"]]
    dataset = validate_dataset_contract(repo, config)
    calibration_records = [
        validate_calibration_manifest(repo, config["canonical_input_commit"], selection, dataset)
        for selection in config["calibration_selections"]
    ]
    schedule = generate_schedule(config)
    schedule_evidence = validate_schedule(schedule, config)
    missing = [
        missing_item
        for record in calibration_records
        for missing_item in record.get("materialization", {}).get("missing", [])
    ]
    unresolved = [
        f"{record['label']}:{error}"
        for record in model_records
        for error in record.get("errors", [])
    ] + [f"dev_contract:{error}" for error in dataset.get("errors", [])]
    unresolved += [
        f"{record['id']}:{error}"
        for record in calibration_records
        for error in record.get("errors", [])
    ]
    contracts_verified = all(record.get("status") == "verified" for record in model_records)
    canonical_inputs_verified = all(record["manifest_contract"]["status"] == "canonical_manifest_valid" for record in calibration_records)
    raw_inputs_ready = contracts_verified and canonical_inputs_verified and dataset["status"] == "verified" and not missing and not unresolved
    if unresolved:
        overall_status = "unresolved"
    elif missing:
        overall_status = "readiness_complete_scored_matrix_blocked"
    else:
        overall_status = "ready_for_server_prepare_review"
    return {
        "schema_version": 1,
        "study": STUDY,
        "phase": "local_cpu_readiness",
        "status": overall_status,
        "scored_matrix_gate": "blocked_deferred_graph_validation",
        "scored_run_authorized": False,
        "gpu_used": False,
        "export_performed": False,
        "tensorrt_build_performed": False,
        "build_matrix_performed": False,
        "canonical_input_commit": config["canonical_input_commit"],
        "raw_inputs_ready_for_server_prepare": raw_inputs_ready,
        "config_identity": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "semantic_sha256": config_semantic_hash(config),
            "locked_sections": list(LOCKED_CONFIG_SECTIONS),
        },
        "execution_provenance": {
            "git_commit": git_head(repo),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "config": str(config_path),
        "environment_observed_before_model_probe": environment,
        "model_contracts": model_records,
        "calibration_readiness": {
            "selections": calibration_records,
            "overlap": calibration_overlap(calibration_records),
            "recipe": config["calibration_recipe"],
        },
        "export_contract": config["export_contract"],
        "execution_boundary": config["execution_boundary"],
        "dev_contract": dataset,
        "schedule": {
            **schedule_evidence,
            "version": config["schedule"]["version"],
            "jobs": schedule,
        },
        "accounting": config["accounting"],
        "environment": environment,
        "missing_prerequisites": missing,
        "unresolved_checks": unresolved,
        "deferred_to_server_prepare": [
            "architecture-specific ONNX export and export manifest/hash",
            "active ONNX graph dataflow mapping and parser/validator evidence",
            "TensorRT environment/GPU identity and direct binary checks",
            "six auxiliary cache-generation builds",
            "seventy-eight scored builds/captures",
        ],
        "decision_rules": config["decision_rules"],
        "limitations": config["limitations"],
    }


def readiness_report(manifest: dict[str, Any]) -> str:
    verified = [row["label"] for row in manifest["model_contracts"] if row.get("status") == "verified"]
    unresolved = manifest["unresolved_checks"]
    missing = manifest["missing_prerequisites"]
    lines = [
        "# Precision-head confirmation readiness",
        "",
        f"- Study: `{manifest['study']}`",
        f"- Status: `{manifest['status']}`",
        f"- Scored matrix gate: `{manifest['scored_matrix_gate']}`",
        "- Scope: CPU-only local readiness; no GPU, ONNX export, TensorRT import/build, or scored matrix.",
        "",
        "## Head/output evidence",
        "",
        f"Verified frozen model contracts: `{', '.join(verified) if verified else 'none'}`.",
        "The contract is checked from head flags, active branch selection, tensor-plus-dict output representation, semantic postprocess path and shapes. Shape alone is not accepted as routing evidence.",
        "",
    ]
    for row in manifest["model_contracts"]:
        evidence = row.get("head_output_evidence", {})
        lines.append(
            f"- `{row['label']}`: status `{row.get('status')}`, head `{evidence.get('head_type', 'unknown')}` index `{evidence.get('head_index', 'unknown')}`, "
            f"end2end `{evidence.get('end2end', 'unknown')}`, active `{evidence.get('active_branches', 'unknown')}`, "
            f"primary `{evidence.get('primary_output_shape', 'unknown')}`, output kind `{evidence.get('output_kind', 'unknown')}`."
        )
    lines += [
        "",
        "## Workload accounting",
        "",
        "The explicit schedule contains 6 auxiliary calibration invocations, 72 scored INT8 invocations, and 6 scored FP16 invocations: 84 builder invocations and 78 captures. Auxiliary engines are not scored and are not selected by AP.",
        "",
        "## Blockers and deferred artifacts",
        "",
    ]
    lines += [f"- Missing prerequisite: `{item}`" for item in missing] or ["- No missing local prerequisite recorded."]
    lines += [f"- Unresolved check: `{item}`" for item in unresolved] or ["- No unresolved check recorded."]
    lines += [
        "- ONNX export, graph-level mapping, TensorRT/GPU identity and all auxiliary/scored execution remain deferred to a separately reviewed server prepare/run phase.",
        "- `scored_run_authorized` is `false`; this report is not TensorRT end-to-end evidence.",
        "",
        "## Statistics protocol",
        "",
        "Report `within_selection_build_SD` across the three repeats and `between_selection_mean_SD` across the three selection means. Keep the +2 pp full-endpoint value only as a pre-set engineering screening target; report FP16 gaps and XS/S effects continuously without a non-inferiority or size gate.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    out_dir = (repo / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Readiness output already exists; refusing overwrite: {out_dir}")
    config = load_config(config_path, repo)
    manifest = build_readiness(repo, config, probe_models=True, config_path=config_path)
    write_json_no_overwrite(out_dir / "readiness_manifest.json", manifest)
    write_json_no_overwrite(out_dir / "model_contracts.json", {"study": STUDY, "models": manifest["model_contracts"]})
    write_json_no_overwrite(out_dir / "calibration_readiness.json", manifest["calibration_readiness"])
    write_json_no_overwrite(out_dir / "schedule.json", manifest["schedule"])
    write_text_no_overwrite(out_dir / "report.md", readiness_report(manifest))
    print(f"DONE: {out_dir / 'readiness_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
