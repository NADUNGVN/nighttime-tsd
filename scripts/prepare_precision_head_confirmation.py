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
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


STUDY = "precision_head_confirmation_v1"
PINNED_SOURCE_COMMIT = "53f12554761c80f367876756b4371a7729891d44"
DEFAULT_CONFIG = Path("configs/precision_head_confirmation_v1.json")
DEFAULT_OUTPUT = Path("results/measurement_audit_v1/precision_head_confirmation_readiness_v1")
ARM_NAMES = ("baseline_int8", "bbox_fp32", "classification_fp32", "both_fp32")
SELECTION_IDS = ("U42", "U43", "U44")
DEV_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


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
    if schedule.get("version") != "model_block_selection_aux_then_scored_v1":
        raise ValueError("Unknown schedule version")
    if schedule.get("model_order") != ["yolov8n", "yolo26n"]:
        raise ValueError("Model schedule order is not locked")
    if schedule.get("selection_order") != list(SELECTION_IDS):
        raise ValueError("Selection schedule order is not locked")
    if schedule.get("arm_order") != list(ARM_NAMES):
        raise ValueError("Arm schedule order is not locked")
    return config


def validate_calibration_payload(payload: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """Validate the canonical producer schema without resolving local paths."""
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
        image = row.get("source_image")
        label = row.get("source_label")
        if not isinstance(image, str) or not image.startswith("train/images/"):
            raise ValueError(f"Calibration image is outside train/images: {image!r}")
        if not isinstance(label, str) or not label.startswith("train/labels/"):
            raise ValueError(f"Calibration label is outside train/labels: {label!r}")
        if "/dev/" in image or "/test/" in image or "\\" in image:
            raise ValueError(f"Calibration image has forbidden split/path: {image!r}")
        if "/dev/" in label or "/test/" in label or "\\" in label:
            raise ValueError(f"Calibration label has forbidden split/path: {label!r}")
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


def validate_calibration_manifest(repo: Path, commit: str, selection: dict[str, Any]) -> dict[str, Any]:
    manifest_path = (repo / selection["manifest"]).resolve()
    observed_sha = canonical_sha256(repo, commit, manifest_path)
    if observed_sha != selection["canonical_sha256"]:
        raise ValueError(f"{selection['id']} canonical manifest hash mismatch")
    payload = json.loads(git_blob(repo, commit, manifest_path).decode("utf-8"))
    contract = validate_calibration_payload(payload, selection)

    materialized_yaml = manifest_path.parent / "calibration.yaml"
    source_root = repo / "data/processed/cctsdb2021_clean"
    missing_images = [
        item["source_image"] for item in payload["files"]
        if not (source_root / item["source_image"]).is_file()
    ]
    missing_labels = [
        item["source_label"] for item in payload["files"]
        if not (source_root / item["source_label"]).is_file()
    ]
    missing = []
    if not materialized_yaml.is_file():
        missing.append(f"{selection['id']}:materialized_calibration_yaml")
    if missing_images:
        missing.append(f"{selection['id']}:selected_train_images:{len(missing_images)}")
    if missing_labels:
        missing.append(f"{selection['id']}:selected_train_labels:{len(missing_labels)}")
    return {
        "id": selection["id"],
        "manifest": selection["manifest"],
        "canonical_sha256": observed_sha,
        "manifest_contract": contract,
        "materialization": {
            "status": "complete" if not missing else "missing_materialization",
            "yaml": str(materialized_yaml.relative_to(repo).as_posix()),
            "yaml_exists": materialized_yaml.is_file(),
            "selected_train_images_present": len(missing_images) == 0,
            "selected_train_labels_present": len(missing_labels) == 0,
            "missing": missing,
        },
    }


def calibration_overlap(records: list[dict[str, Any]]) -> dict[str, Any]:
    sets = {record["id"]: set(record["manifest_contract"]["image_ids"]) for record in records}
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
        result["status"] = "unresolved" if errors else "not_probed"
        result["errors"] = errors
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


def validate_dataset_contract(repo: Path, config: dict[str, Any]) -> dict[str, Any]:
    images_dir = repo / "data/processed/cctsdb2021_clean/dev/images"
    labels_dir = repo / "data/processed/cctsdb2021_clean/dev/labels"
    images = sorted(path for path in images_dir.glob("*") if path.suffix.lower() in DEV_IMAGE_EXTENSIONS)
    labels = sorted(labels_dir.glob("*.txt"))
    instances = 0
    for label in labels:
        instances += sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())
    expected = config["dev_contract"]
    errors = []
    if len(images) != expected["images"]:
        errors.append(f"dev_image_count:{len(images)}!={expected['images']}")
    if len(labels) != expected["images"]:
        errors.append(f"dev_label_count:{len(labels)}!={expected['images']}")
    if instances != expected["instances"]:
        errors.append(f"dev_instance_count:{instances}!={expected['instances']}")
    return {
        "status": "verified" if not errors else "unresolved",
        "split": expected["split"],
        "images_observed": len(images),
        "labels_observed": len(labels),
        "instances_observed": instances,
        "expected": expected,
        "errors": errors,
        "yaml_reference": expected["yaml_reference"],
    }


def generate_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    schedule = []
    sequence = 1
    for model in config["schedule"]["model_order"]:
        for selection in config["schedule"]["selection_order"]:
            schedule.append({
                "sequence": sequence, "model": model, "selection": selection,
                "phase": "auxiliary_calibration", "arm": "baseline_int8", "repeat": None,
                "scored": False, "capture_required": False, "calibration_cache_creation": True,
            })
            sequence += 1
            for arm in config["schedule"]["arm_order"]:
                for repeat in config["schedule"]["build_repeat_order"]:
                    schedule.append({
                        "sequence": sequence, "model": model, "selection": selection,
                        "phase": "scored_int8", "arm": arm, "repeat": repeat,
                        "scored": True, "capture_required": True, "calibration_cache_creation": False,
                    })
                    sequence += 1
        for repeat in config["schedule"]["fp16_repeat_order"]:
            schedule.append({
                "sequence": sequence, "model": model, "selection": None,
                "phase": "scored_fp16", "arm": "fp16", "repeat": repeat,
                "scored": True, "capture_required": True, "calibration_cache_creation": False,
            })
            sequence += 1
    return schedule


def validate_schedule(schedule: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if len(schedule) != 84:
        raise ValueError(f"Schedule length is {len(schedule)}, expected 84")
    if [row["sequence"] for row in schedule] != list(range(1, 85)):
        raise ValueError("Schedule sequence is not contiguous")
    keys = [(row["model"], row["selection"], row["arm"], row["repeat"]) for row in schedule]
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
        for selection in config["schedule"]["selection_order"]:
            cell = [row for row in block if row["selection"] == selection]
            if cell[0]["phase"] != "auxiliary_calibration":
                raise ValueError(f"Auxiliary build is not first in {model}/{selection}")
    return {
        "status": "verified",
        "jobs": len(schedule),
        "auxiliary_calibration_builds": len(aux),
        "scored_int8_builds": len(int8),
        "scored_fp16_builds": len(fp16),
        "scored_builds": len(int8) + len(fp16),
        "captures": sum(1 for row in schedule if row["capture_required"]),
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


def build_readiness(repo: Path, config: dict[str, Any], probe_models: bool = True) -> dict[str, Any]:
    model_records = [inspect_frozen_model(repo, model, probe=probe_models) for model in config["models"]]
    calibration_records = [
        validate_calibration_manifest(repo, config["canonical_input_commit"], selection)
        for selection in config["calibration_selections"]
    ]
    schedule = generate_schedule(config)
    schedule_evidence = validate_schedule(schedule, config)
    dataset = validate_dataset_contract(repo, config)
    missing = [
        missing_item
        for record in calibration_records
        for missing_item in record["materialization"]["missing"]
    ]
    unresolved = [
        f"{record['label']}:{error}"
        for record in model_records
        for error in record.get("errors", [])
    ] + [f"dev_contract:{error}" for error in dataset.get("errors", [])]
    contracts_verified = all(record.get("status") == "verified" for record in model_records)
    canonical_inputs_verified = all(record["manifest_contract"]["status"] == "canonical_manifest_valid" for record in calibration_records)
    raw_prereqs_verified = contracts_verified and canonical_inputs_verified and dataset["status"] == "verified"
    scored_gate = "blocked_missing_prepare_artifacts" if missing or unresolved else "ready_for_review"
    return {
        "schema_version": 1,
        "study": STUDY,
        "phase": "local_cpu_readiness",
        "status": "readiness_complete_scored_matrix_blocked" if raw_prereqs_verified and (missing or not contracts_verified) else ("ready_for_review" if raw_prereqs_verified else "unresolved"),
        "scored_matrix_gate": scored_gate,
        "scored_run_authorized": False,
        "gpu_used": False,
        "export_performed": False,
        "tensorrt_build_performed": False,
        "build_matrix_performed": False,
        "canonical_input_commit": config["canonical_input_commit"],
        "config": "configs/precision_head_confirmation_v1.json",
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
        "environment": environment_evidence(),
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
    manifest = build_readiness(repo, config, probe_models=True)
    write_json_no_overwrite(out_dir / "readiness_manifest.json", manifest)
    write_json_no_overwrite(out_dir / "model_contracts.json", {"study": STUDY, "models": manifest["model_contracts"]})
    write_json_no_overwrite(out_dir / "calibration_readiness.json", manifest["calibration_readiness"])
    write_json_no_overwrite(out_dir / "schedule.json", manifest["schedule"])
    write_text_no_overwrite(out_dir / "report.md", readiness_report(manifest))
    print(f"DONE: {out_dir / 'readiness_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
