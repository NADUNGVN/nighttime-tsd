#!/usr/bin/env python3
"""Prepare architecture-specific ONNX graphs for the precision-head study.

This is a prepare-only boundary.  The parent performs CPU/readiness checks and
dispatches one model-specific child at a time.  Only the child imports
Ultralytics/ONNX and exports a frozen model to a private, new output tree.  No
TensorRT module is imported, no CUDA API is touched, no calibration tensor is
materialized, and no scored job is dispatched here.

The runner is intentionally conservative about graph mapping.  A source
module name must match one exported Conv node, that node must reach an output,
and a branch-owned pre/post-processing merge must expose the bbox/class
channel spans.  Shape or prefix matches without dataflow evidence are
reported as ``mapping_unresolved`` rather than being guessed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import prepare_precision_head_confirmation as readiness


STUDY = "precision_head_confirmation_v1"
PREPARE_STUDY = "precision_head_confirmation_graph_prepare_v1"
ACCEPTED_READINESS_COMMIT = "7c0ea9e7fe86dfa6358ec1ee90473f9243a53f76"
ACCEPTED_READINESS_EXECUTION_COMMIT = "9c0597d1ec6f6d73c98fffa7cc3779a00eddec34"
READINESS_ROOT_NAME = "server_precision_head_confirmation_readiness_v2"
READINESS_FILES = (
    "readiness_manifest.json",
    "model_contracts.json",
    "calibration_readiness.json",
    "schedule.json",
    "report.md",
)
ACCEPTED_READINESS_FILE_SHA256 = {
    "readiness_manifest.json": "112afbd2384da4e381e3ba9f7293c2d7fdc22991448acabb66c1fa21b6276147",
    "model_contracts.json": "dac10919a336f115676bbe997e30ecd66419b4b99f5795f4499498f1b7fd4d51",
    "calibration_readiness.json": "34c258e518c826bae92ad880e76cb78977c6afd054943ae9ebb1fcfc49fec41c",
    "schedule.json": "531ae23f2269f862fd9141e0f85b567fc8d997933391f62cedc2e3195f84dd5b",
    "report.md": "7cf1f681d1cac6c3fe4b913b825ed088bcafe266e5874a1179d239d1aaeeaaf7",
}
MODEL_CHOICES = ("yolov8n", "yolo26n")
DEFAULT_READINESS_ROOT = Path("results/measurement_audit_v1") / READINESS_ROOT_NAME
DEFAULT_OUTPUT = Path("results/measurement_audit_v1/precision_head_confirmation_graph_prep_v1")
EXPORT_ARGUMENTS = {
    "format": "onnx",
    "imgsz": 640,
    "batch": 1,
    "opset": 17,
    "simplify": True,
    "dynamic": False,
    "half": False,
    "device": "cpu",
    "task": "detect",
}
GRAPH_OUTPUT_SHAPES = {
    "yolov8n": [1, 7, 8400],
    "yolo26n": [1, 300, 6],
}


def canonical_json(value: Any) -> bytes:
    return readiness.canonical_json(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return readiness.sha256_file(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_no_overwrite(path: Path, value: Any) -> None:
    readiness.write_json_no_overwrite(path, value)


def write_text_no_overwrite(path: Path, value: str) -> None:
    readiness.write_text_no_overwrite(path, value)


def repo_path(repo: Path, value: Path) -> Path:
    resolved = (value if value.is_absolute() else repo / value).resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes repository: {value}") from exc
    return resolved


def git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError("Cannot resolve current Git HEAD")
    return result.stdout.strip()


def git_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo, capture_output=True, check=False,
    )
    return result.returncode == 0


def _actual_files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def validate_readiness_artifact(repo: Path, root: Path) -> dict[str, Any]:
    """Validate the accepted five-file readiness artifact and its Git blobs."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Readiness root is missing: {root}")
    if root.name != READINESS_ROOT_NAME:
        raise ValueError(f"Readiness root must be named {READINESS_ROOT_NAME}")
    if _actual_files(root) != sorted(READINESS_FILES):
        raise ValueError("Readiness root must contain exactly the five accepted files")

    file_records: dict[str, dict[str, Any]] = {}
    for relative in READINESS_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Readiness artifact is not a regular file: {relative}")
        actual_bytes = path.read_bytes()
        actual_hash = sha256_bytes(actual_bytes)
        expected_hash = ACCEPTED_READINESS_FILE_SHA256[relative]
        canonical_bytes = readiness.git_blob(repo, ACCEPTED_READINESS_COMMIT, repo / "results/measurement_audit_v1" / READINESS_ROOT_NAME / relative)
        canonical_hash = sha256_bytes(canonical_bytes)
        if actual_hash != expected_hash or canonical_hash != expected_hash or actual_bytes != canonical_bytes:
            raise ValueError(f"Readiness artifact hash/blob mismatch: {relative}")
        file_records[relative] = {
            "path": str(path),
            "bytes": len(actual_bytes),
            "sha256": actual_hash,
            "accepted_git_blob_sha256": canonical_hash,
        }

    manifest = read_json(root / "readiness_manifest.json")
    model_contracts = read_json(root / "model_contracts.json")
    calibration = read_json(root / "calibration_readiness.json")
    schedule = read_json(root / "schedule.json")
    if manifest.get("study") != STUDY or manifest.get("status") != "ready_for_server_prepare_review":
        raise ValueError("Readiness manifest is not the accepted CPU-ready study")
    if manifest.get("raw_inputs_ready_for_server_prepare") is not True:
        raise ValueError("Readiness manifest does not accept raw inputs")
    if manifest.get("scored_run_authorized") is not False:
        raise ValueError("Readiness artifact must not authorize scored execution")
    for key in ("gpu_used", "export_performed", "tensorrt_build_performed", "build_matrix_performed"):
        if manifest.get(key) is not False:
            raise ValueError(f"Readiness artifact incorrectly records {key}=true")
    if manifest.get("missing_prerequisites") != [] or manifest.get("unresolved_checks") != []:
        raise ValueError("Accepted readiness artifact contains missing or unresolved checks")
    if manifest.get("execution_provenance", {}).get("git_commit") != ACCEPTED_READINESS_EXECUTION_COMMIT:
        raise ValueError("Readiness artifact execution commit is not the accepted inventory commit")
    if model_contracts != {"study": STUDY, "models": manifest.get("model_contracts")}:
        raise ValueError("Standalone model contracts do not equal the manifest section")
    if calibration != manifest.get("calibration_readiness"):
        raise ValueError("Standalone calibration readiness does not equal the manifest section")
    if schedule != manifest.get("schedule"):
        raise ValueError("Standalone schedule does not equal the manifest section")
    return {
        "root": str(root),
        "accepted_git_commit": ACCEPTED_READINESS_COMMIT,
        "accepted_execution_commit": ACCEPTED_READINESS_EXECUTION_COMMIT,
        "files": file_records,
        "manifest": manifest,
        "model_contracts": model_contracts,
        "calibration_readiness": calibration,
        "schedule": schedule,
    }


def validate_config_binding(repo: Path, accepted: dict[str, Any], config_path: Path | None = None) -> dict[str, Any]:
    config_path = (config_path or repo / readiness.DEFAULT_CONFIG).resolve()
    manifest = accepted["manifest"]
    identity = manifest.get("config_identity", {})
    actual_hash = sha256_file(config_path) if config_path.is_file() else None
    if actual_hash != identity.get("sha256"):
        raise ValueError(f"Current config hash differs from accepted readiness: {actual_hash} != {identity.get('sha256')}")
    config = readiness.load_config(config_path, repo)
    semantic_hash = readiness.config_semantic_hash(config)
    if semantic_hash != identity.get("semantic_sha256"):
        raise ValueError("Current config semantic hash differs from accepted readiness")
    canonical_script = readiness.git_blob(
        repo, ACCEPTED_READINESS_EXECUTION_COMMIT,
        repo / "scripts/prepare_precision_head_confirmation.py",
    )
    expected_script_hash = manifest.get("execution_provenance", {}).get("script_sha256")
    if sha256_bytes(canonical_script) != expected_script_hash:
        raise ValueError("Accepted readiness script hash does not match its execution commit")
    current_script = repo / "scripts/prepare_precision_head_confirmation.py"
    if sha256_file(current_script) != expected_script_hash:
        raise ValueError("Current readiness helper changed after the accepted inventory")
    return {
        "path": str(config_path),
        "sha256": actual_hash,
        "semantic_sha256": semantic_hash,
        "accepted_script_sha256": expected_script_hash,
        "accepted_execution_commit": ACCEPTED_READINESS_EXECUTION_COMMIT,
        "config": config,
    }


def selected_model_configs(config: dict[str, Any], model: str) -> list[dict[str, Any]]:
    if model not in (*MODEL_CHOICES, "all"):
        raise ValueError(f"Unsupported model selection: {model}")
    selected = list(config["models"]) if model == "all" else [row for row in config["models"] if row.get("label") == model]
    if len(selected) != (2 if model == "all" else 1):
        raise ValueError(f"Model selection does not match the locked two-model config: {model}")
    return selected


def validate_current_bindings(repo: Path, config: dict[str, Any], accepted: dict[str, Any], model: str) -> dict[str, Any]:
    """Recheck data/checkpoint bindings before any export is attempted."""
    selected = selected_model_configs(config, model)
    expected_models = {row["label"]: row for row in accepted["manifest"].get("model_contracts", [])}
    for model_config in selected:
        accepted_model = expected_models.get(model_config["label"])
        if accepted_model is None:
            raise ValueError(f"Accepted readiness has no model contract for {model_config['label']}")
        if accepted_model.get("expected_sha256") != model_config.get("sha256"):
            raise ValueError(f"Checkpoint contract mismatch for {model_config['label']}")
        checkpoint = repo_path(repo, Path(model_config["checkpoint"]))
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise FileNotFoundError(f"Frozen checkpoint is missing or symlinked: {checkpoint}")
        if sha256_file(checkpoint) != model_config["sha256"] or checkpoint.stat().st_size != model_config["bytes"]:
            raise ValueError(f"Frozen checkpoint hash/size mismatch for {model_config['label']}")

    dataset = readiness.validate_dataset_contract(repo, config)
    if dataset.get("status") != "verified":
        raise ValueError(f"Current dataset binding is unresolved: {dataset.get('errors')}")
    calibrations = [
        readiness.validate_calibration_manifest(repo, config["canonical_input_commit"], selection, dataset)
        for selection in config["calibration_selections"]
    ]
    for record in calibrations:
        if record.get("status") != "verified" or (record.get("materialization") or {}).get("status") != "complete":
            raise ValueError(f"Current calibration binding is unresolved for {record.get('id')}")
    return {
        "selected_models": [row["label"] for row in selected],
        "checkpoint_bindings": {
            row["label"]: {
                "path": row["checkpoint"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
            }
            for row in selected
        },
        "dataset": dataset,
        "calibrations": calibrations,
        "selected_calibration_ids": [row["id"] for row in config["calibration_selections"]],
    }


def package_evidence(package_names: tuple[str, ...] = ("torch", "ultralytics", "onnx", "onnxslim")) -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for name in package_names:
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = None
        record: dict[str, Any] = {"version": version, "module": None, "module_sha256": None}
        if version is not None:
            try:
                module = importlib.import_module(name)
                module_path = getattr(module, "__file__", None)
                record["module"] = str(module_path) if module_path else None
                if module_path and Path(module_path).is_file():
                    record["module_sha256"] = sha256_file(Path(module_path))
            except Exception as exc:  # producer diagnostics must preserve import failure, not hide it
                record["import_error"] = f"{type(exc).__name__}:{exc}"
        packages[name] = record
    return packages


def environment_evidence() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": None,
        "producer_packages": package_evidence(),
        "cuda_touched": False,
        "tensorrt_imported": False,
    }


def parent_environment_evidence() -> dict[str, Any]:
    """Return parent telemetry without importing a model/runtime module."""
    versions = {}
    for name in ("torch", "ultralytics", "onnx", "onnxslim"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions_from_metadata_only": versions,
        "imports_performed": [],
        "cuda_touched": False,
        "tensorrt_imported": False,
    }


def normalize_module_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().replace("\\", "/").strip("/")
    text = text.replace("/", ".")
    text = re.sub(r":\d+$", "", text)
    text = re.sub(r"\.Conv(?:Integer)?$", "", text)
    text = re.sub(r"\.+", ".", text)
    return text.strip(".")


def _node_field(node: Any, key: str, default: Any = None) -> Any:
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def graph_node_records(graph: Any) -> list[dict[str, Any]]:
    raw_nodes = graph.get("nodes", []) if isinstance(graph, dict) else getattr(graph, "node", [])
    records = []
    for index, node in enumerate(raw_nodes):
        name = _node_field(node, "name", "") or f"<unnamed:{index}>"
        inputs = list(_node_field(node, "inputs", _node_field(node, "input", [])) or [])
        outputs = list(_node_field(node, "outputs", _node_field(node, "output", [])) or [])
        attributes = _node_field(node, "attributes", {}) or {}
        if not isinstance(attributes, dict):
            attributes = {
                getattr(attr, "name", ""): getattr(attr, "i", None)
                for attr in attributes
            }
        records.append({
            "index": index,
            "name": str(name),
            "normalized_name": normalize_module_name(name),
            "op_type": str(_node_field(node, "op_type", _node_field(node, "opType", ""))),
            "inputs": [str(item) for item in inputs],
            "outputs": [str(item) for item in outputs],
            "attributes": attributes,
        })
    return records


def _value_shape(value: Any) -> list[Any] | None:
    if isinstance(value, dict):
        shape = value.get("shape")
        return list(shape) if isinstance(shape, list) else None
    type_proto = getattr(value, "type", None)
    tensor_type = getattr(type_proto, "tensor_type", None)
    shape_proto = getattr(tensor_type, "shape", None)
    if shape_proto is None:
        return None
    result: list[Any] = []
    for dimension in getattr(shape_proto, "dim", []):
        if getattr(dimension, "dim_value", 0):
            result.append(int(dimension.dim_value))
        elif getattr(dimension, "dim_param", ""):
            result.append(str(dimension.dim_param))
        else:
            result.append(None)
    return result


def graph_tensor_shapes(graph: Any) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    if isinstance(graph, dict):
        for name, shape in (graph.get("tensor_shapes", {}) or {}).items():
            result[str(name)] = list(shape)
        for row in graph.get("inputs", []) + graph.get("outputs", []) + graph.get("value_info", []):
            if isinstance(row, dict) and isinstance(row.get("name"), str) and isinstance(row.get("shape"), list):
                result[row["name"]] = list(row["shape"])
        return result
    graph_proto = graph
    for row in list(getattr(graph_proto, "input", [])) + list(getattr(graph_proto, "output", [])) + list(getattr(graph_proto, "value_info", [])):
        shape = _value_shape(row)
        if shape is not None:
            result[str(row.name)] = shape
    return result


def graph_output_records(graph: Any) -> list[dict[str, Any]]:
    raw_outputs = graph.get("outputs", []) if isinstance(graph, dict) else getattr(graph, "output", [])
    shapes = graph_tensor_shapes(graph)
    records = []
    for row in raw_outputs:
        if isinstance(row, dict):
            name = row.get("name")
            shape = row.get("shape", shapes.get(name))
        else:
            name = getattr(row, "name", None)
            shape = shapes.get(name)
        if not isinstance(name, str):
            continue
        records.append({"name": name, "shape": list(shape) if isinstance(shape, list) else None})
    return records


def graph_indices(graph: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    nodes = graph_node_records(graph)
    producers: dict[str, dict[str, Any]] = {}
    consumers: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        for output in node["outputs"]:
            if output:
                producers[output] = node
        for input_name in node["inputs"]:
            if input_name:
                consumers.setdefault(input_name, []).append(node)
    return nodes, producers, consumers


def _branch_from_name(name: str) -> str | None:
    normalized = normalize_module_name(name)
    if re.search(r"(?:^|\.)one2one_cv2(?:\.|$)", normalized) or re.search(r"(?:^|\.)cv2(?:\.|$)", normalized):
        return "bbox"
    if re.search(r"(?:^|\.)one2one_cv3(?:\.|$)", normalized) or re.search(r"(?:^|\.)cv3(?:\.|$)", normalized):
        return "classification"
    return None


def _ancestors(tensor: str, producers: dict[str, dict[str, Any]], memo: dict[str, set[str]]) -> set[str]:
    if tensor in memo:
        return memo[tensor]
    producer = producers.get(tensor)
    if producer is None:
        memo[tensor] = set()
        return set()
    result = {producer["name"]}
    for input_name in producer["inputs"]:
        result.update(_ancestors(input_name, producers, memo))
    memo[tensor] = result
    return result


def _forward(start_tensors: list[str], consumers: dict[str, list[dict[str, Any]]], output_names: set[str]) -> tuple[set[str], set[str]]:
    queue = list(start_tensors)
    seen_tensors: set[str] = set()
    seen_nodes: set[str] = set()
    reached_outputs: set[str] = set()
    while queue:
        tensor = queue.pop(0)
        if tensor in seen_tensors:
            continue
        seen_tensors.add(tensor)
        if tensor in output_names:
            reached_outputs.add(tensor)
        for node in consumers.get(tensor, []):
            seen_nodes.add(node["name"])
            queue.extend(node["outputs"])
    return seen_nodes, reached_outputs


def _attribute_int(attributes: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = attributes.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], int):
        return value[0]
    return default


def _find_branch_merge(
    nodes: list[dict[str, Any]],
    producers: dict[str, dict[str, Any]],
    shapes: dict[str, list[Any]],
    branch_targets: dict[str, set[str]],
    expected_shape: list[int],
) -> dict[str, Any] | None:
    ancestor_memo: dict[str, set[str]] = {}
    candidates = []
    for node in nodes:
        if node["op_type"] != "Concat":
            continue
        axis = _attribute_int(node["attributes"], "axis")
        if axis != 1 or not node["outputs"]:
            continue
        output_shape = shapes.get(node["outputs"][0])
        if output_shape != expected_shape:
            continue
        inputs = node["inputs"]
        owners: dict[str, list[str]] = {}
        input_spans = []
        channel_cursor = 0
        valid = True
        for input_name in inputs:
            input_shape = shapes.get(input_name)
            if not isinstance(input_shape, list) or len(input_shape) <= 1 or not isinstance(input_shape[1], int):
                valid = False
                break
            owner_names = []
            ancestors = _ancestors(input_name, producers, ancestor_memo)
            for owner, target_names in branch_targets.items():
                if ancestors & target_names:
                    owner_names.append(owner)
            if len(owner_names) != 1:
                valid = False
                break
            owner = owner_names[0]
            owners.setdefault(owner, []).append(input_name)
            width = int(input_shape[1])
            input_spans.append({"input": input_name, "owner": owner, "span": [channel_cursor, channel_cursor + width], "shape": input_shape})
            channel_cursor += width
        if valid and set(owners) == {"bbox", "classification"} and channel_cursor == expected_shape[1]:
            candidates.append({
                "node": node["name"],
                "output": node["outputs"][0],
                "axis": axis,
                "shape": output_shape,
                "input_spans": input_spans,
            })
    if len(candidates) != 1:
        return None
    return candidates[0]


def audit_graph_mapping(graph: Any, model_label: str, accepted_model: dict[str, Any]) -> dict[str, Any]:
    """Audit Conv ownership and actual graph dataflow for one exported model."""
    expected_contract = accepted_model.get("head_output_evidence") or {}
    active = list(expected_contract.get("active_branches") or accepted_model.get("active_branches") or [])
    inactive = list(expected_contract.get("inactive_branches") or accepted_model.get("inactive_branches") or [])
    source_mapping = accepted_model.get("active_convolution_mapping") or {}
    nodes, producers, consumers = graph_indices(graph)
    shapes = graph_tensor_shapes(graph)
    outputs = graph_output_records(graph)
    output_names = {row["name"] for row in outputs}
    expected_output_shape = GRAPH_OUTPUT_SHAPES[model_label]
    primary_outputs = [row for row in outputs if row.get("shape") == expected_output_shape]
    errors: list[str] = []
    if not primary_outputs:
        errors.append(f"primary_output_shape_not_observed:{expected_output_shape}")

    branch_targets: dict[str, set[str]] = {"bbox": set(), "classification": set()}
    branch_records: dict[str, Any] = {}
    all_target_nodes: set[str] = set()
    for branch in active:
        owner = "bbox" if branch.endswith("cv2") else "classification" if branch.endswith("cv3") else None
        if owner is None:
            errors.append(f"unsupported_active_branch:{branch}")
            continue
        expected_convs = source_mapping.get(branch, [])
        if not isinstance(expected_convs, list) or not expected_convs:
            errors.append(f"missing_source_convolution_mapping:{branch}")
            expected_convs = []
        head_prefix = expected_contract.get("head_prefix")
        branch_prefix = normalize_module_name(f"{head_prefix}.{branch}") if head_prefix else ""
        if not branch_prefix or branch_prefix.endswith("."):
            # The readiness record contains fully qualified module names; derive
            # the prefix from the first source convolution when head_prefix is
            # not separately recorded.
            first_name = expected_convs[0].get("name", "") if expected_convs else ""
            parts = normalize_module_name(first_name).split(".")
            branch_token = branch
            try:
                branch_prefix = ".".join(parts[:parts.index(branch_token) + 1])
            except ValueError:
                branch_prefix = ""
        prefix_matches = [node for node in nodes if branch_prefix and (node["normalized_name"] == branch_prefix or node["normalized_name"].startswith(branch_prefix + "."))]
        excluded_non_conv = [node["name"] for node in prefix_matches if node["op_type"] != "Conv"]
        matched = []
        for source_conv in expected_convs:
            source_name = normalize_module_name(source_conv.get("name"))
            candidates = [node for node in prefix_matches if node["op_type"] == "Conv" and node["normalized_name"] == source_name]
            if len(candidates) != 1:
                errors.append(f"{branch}:source_conv_match_count:{source_name}:{len(candidates)}")
                continue
            node = candidates[0]
            reachable_nodes, reachable_outputs = _forward(node["outputs"], consumers, output_names)
            if not reachable_outputs:
                errors.append(f"{branch}:unreachable_conv:{node['name']}")
            entry = {
                "source_module": source_conv.get("name"),
                "source_module_type": source_conv.get("module_type"),
                "export_node": node["name"],
                "export_op_type": node["op_type"],
                "match": "exact_normalized_module_name",
                "output_tensors": node["outputs"],
                "reachable_outputs": sorted(reachable_outputs),
                "downstream_node_count": len(reachable_nodes),
            }
            matched.append(entry)
            branch_targets[owner].add(node["name"])
            all_target_nodes.add(node["name"])
        branch_records[branch] = {
            "owner": owner,
            "source_prefix": branch_prefix,
            "candidate_count": len(prefix_matches),
            "candidate_nodes": [
                {"name": node["name"], "op_type": node["op_type"], "outputs": node["outputs"]}
                for node in prefix_matches
            ],
            "excluded_non_convolution_nodes": sorted(excluded_non_conv),
            "matched_convolutions": matched,
            "target_node_names": sorted(node["export_node"] for node in matched),
        }
        if not matched:
            errors.append(f"{branch}:no_matched_convolution")

    inactive_records = {}
    for branch in inactive:
        if not isinstance(branch, str):
            errors.append("inactive_branch_name_malformed")
            continue
        candidates = [node for node in nodes if re.search(rf"(?:^|\.){re.escape(branch)}(?:\.|$)", node["normalized_name"])]
        inactive_records[branch] = {
            "candidate_nodes": [{"name": node["name"], "op_type": node["op_type"]} for node in candidates],
            "excluded_from_precision_targets": True,
        }

    if branch_targets["bbox"] & branch_targets["classification"]:
        errors.append("bbox_and_classification_target_sets_overlap")
    boxes_shape = expected_contract.get("boxes_shape")
    scores_shape = expected_contract.get("scores_shape")
    merge_shape = None
    if (
        isinstance(boxes_shape, list) and isinstance(scores_shape, list)
        and len(boxes_shape) == 3 and len(scores_shape) == 3
        and boxes_shape[0] == scores_shape[0] and boxes_shape[2] == scores_shape[2]
        and isinstance(scores_shape[1], int)
    ):
        # The readiness ``boxes_shape`` is the DFL/debug tensor (64 bins),
        # while the decoded branch merge has four box coordinates.  Do not
        # mistake the debug representation for the exported primary layout.
        merge_shape = [boxes_shape[0], 4 + scores_shape[1], boxes_shape[2]]
    merge = _find_branch_merge(nodes, producers, shapes, branch_targets, merge_shape or [1, 7, 8400])
    if merge is None:
        errors.append("branch_owned_channel_merge_unresolved")
    else:
        spans = {row["owner"]: row["span"] for row in merge["input_spans"]}
        for branch, owner in (("bbox", "bbox"), ("classification", "classification")):
            for record in branch_records.values():
                if record.get("owner") == owner:
                    record["output_lineage"] = {
                        "branch_owner": owner,
                        "pre_postprocess_merge": merge["node"],
                        "merge_output": merge["output"],
                        "channel_span": spans.get(owner),
                        "semantic_role": "box_coordinates" if owner == "bbox" else "classification_scores",
                        "reachability_may_be_shared_after_merge": model_label == "yolo26n",
                    }
    for branch, record in branch_records.items():
        if not record.get("matched_convolutions"):
            errors.append(f"{branch}:empty_target_set")

    mapping_status = "verified" if not errors else "mapping_unresolved"
    mapping_payload = {
        "model": model_label,
        "active_branch_audit": branch_records,
        "inactive_branch_audit": inactive_records,
        "branch_owned_target_sets": {
            "bbox": sorted(branch_targets["bbox"]),
            "classification": sorted(branch_targets["classification"]),
        },
        "branch_merge": merge,
    }
    return {
        "status": mapping_status,
        "model": model_label,
        "mapping_status": mapping_status,
        "errors": errors,
        "graph_outputs": outputs,
        "expected_primary_output_shape": expected_output_shape,
        "primary_output_candidates": primary_outputs,
        "active_branches": active,
        "inactive_branches": inactive,
        "active_branch_audit": branch_records,
        "inactive_branch_audit": inactive_records,
        "branch_owned_target_sets": {
            "bbox": sorted(branch_targets["bbox"]),
            "classification": sorted(branch_targets["classification"]),
            "both_union": sorted(all_target_nodes),
            "disjoint": not bool(branch_targets["bbox"] & branch_targets["classification"]),
        },
        "output_ancestry_policy": "branch ownership is established before shared downstream reachability; YOLO26 does not require disjoint final-output ancestry",
        "source_to_export_policy": "exact normalized source-module match; fusion/renaming without explicit lineage is unresolved",
        "branch_merge": merge,
        "mapping_hash": sha256_bytes(canonical_json(mapping_payload)),
    }


def precision_target_sets(mapping: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm not in ("baseline_int8", "bbox_fp32", "classification_fp32", "both_fp32"):
        raise ValueError(f"Unknown precision arm: {arm}")
    bbox = list(mapping.get("branch_owned_target_sets", {}).get("bbox", []))
    classification = list(mapping.get("branch_owned_target_sets", {}).get("classification", []))
    if not bbox or not classification or not mapping.get("branch_owned_target_sets", {}).get("disjoint"):
        raise ValueError("Cannot create precision target sets from unresolved branch mapping")
    selected = {
        "baseline_int8": [],
        "bbox_fp32": bbox,
        "classification_fp32": classification,
        "both_fp32": sorted(set(bbox + classification)),
    }[arm]
    return {
        "arm": arm,
        "bbox_fp32_targets": bbox if arm in ("bbox_fp32", "both_fp32") else [],
        "classification_fp32_targets": classification if arm in ("classification_fp32", "both_fp32") else [],
        "target_layers": selected,
        "target_sets_disjoint": not bool(set(bbox) & set(classification)),
        "baseline_non_target_protection": "all active branch convolutions remain INT8 unless explicitly listed by the locked arm",
        "effective_precision": "must be verified from builder/inspector in the later server build phase",
    }


def _shape(value: Any) -> list[Any] | None:
    if isinstance(value, dict) and isinstance(value.get("shape"), list):
        return list(value["shape"])
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in shape]
    except (TypeError, ValueError):
        return None


def _rows(value: Any) -> list[list[Any]] | None:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return value
        return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        return converted if isinstance(converted, list) else None
    return None


def validate_native_output(model_label: str, output: Any, expected_contract: dict[str, Any]) -> dict[str, Any]:
    """Validate native output representation without applying a second NMS."""
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise ValueError("Native output must be the locked primary-plus-head pair")
    primary, head = output
    if _shape(primary) != list(expected_contract["primary_output_shape"]):
        raise ValueError(f"Native primary output shape mismatch for {model_label}")
    observed_dtype = str(getattr(primary, "dtype", "unknown"))
    expected_dtype = expected_contract.get("primary_output_dtype")
    if expected_dtype and observed_dtype != "unknown" and observed_dtype != expected_dtype:
        raise ValueError(f"Native primary output dtype mismatch for {model_label}: {observed_dtype}")
    device = getattr(primary, "device", None)
    if device is not None and str(getattr(device, "type", device)) != "cpu":
        raise ValueError(f"Native primary output is not on CPU for {model_label}")
    if not isinstance(head, dict):
        raise ValueError("Native head debug output must be a dictionary")
    if model_label == "yolov8n":
        if expected_contract.get("end2end") is not False or expected_contract.get("nms") != "not_in_model_head_output":
            raise ValueError("YOLOv8n adapter was given an end2end contract")
        return {
            "model": model_label,
            "representation": "tuple_tensor_plus_dict",
            "primary_shape": _shape(primary),
            "primary_dtype": observed_dtype,
            "box_channels": [0, 4],
            "score_channels": [4, 7],
            "postprocess": "none_in_model_head_output",
            "nms_applied": False,
            "double_nms": False,
            "native_head_keys": sorted(head),
        }
    if model_label == "yolo26n":
        if expected_contract.get("end2end") is not True or expected_contract.get("nms") == "not_in_model_head_output":
            raise ValueError("YOLO26n adapter was given a non-end2end contract")
        if not isinstance(head.get("one2one"), dict):
            raise ValueError("YOLO26n native output is missing one2one head evidence")
        rows = _rows(primary)
        if rows is not None:
            for row in rows:
                if len(row) != 6:
                    raise ValueError("YOLO26n detections must use [x1,y1,x2,y2,score,class_id]")
                if not all(isinstance(item, (int, float)) for item in row):
                    raise ValueError("YOLO26n detection values must be numeric")
                if not all(float(item) == float(item) for item in row):
                    raise ValueError("YOLO26n detection values must be finite")
                if not 0.0 <= float(row[4]) <= 1.0 or int(row[5]) not in (0, 1, 2):
                    raise ValueError("YOLO26n score/class semantics are invalid")
        return {
            "model": model_label,
            "representation": "tuple_tensor_plus_dict",
            "primary_shape": _shape(primary),
            "primary_dtype": observed_dtype,
            "detection_columns": {"boxes": [0, 4], "score": 4, "class_id": 5},
            "postprocess": "topk_300_detections_in_model_head_output",
            "nms_applied": False,
            "double_nms": False,
            "native_head_keys": sorted(head),
        }
    raise ValueError(f"Unsupported native adapter model: {model_label}")


def adapter_contract(model_label: str, expected_contract: dict[str, Any]) -> dict[str, Any]:
    if model_label not in MODEL_CHOICES:
        raise ValueError(f"Unsupported adapter model: {model_label}")
    output = validate_native_output(model_label, (type("ShapeOnly", (), {"shape": expected_contract["primary_output_shape"]})(), {} if model_label == "yolov8n" else {"one2one": {}}), expected_contract)
    return {
        "model": model_label,
        "primary_output_shape": expected_contract["primary_output_shape"],
        "output_kind": expected_contract["output_kind"],
        "postprocess": expected_contract["postprocess"],
        "nms": expected_contract["nms"],
        "native_output_validation": output,
        "scored_adapter_status": "deferred_until_real_TensorRT_forward_and_native_parser_validation",
        "double_nms_forbidden": True,
    }


def validate_native_contract_evidence(
    model_label: str, native_evidence: dict[str, Any], expected_contract: dict[str, Any]
) -> dict[str, Any]:
    """Bridge the reviewed CPU head probe to the model-specific adapter."""
    required = (
        "primary_output_shape", "primary_output_representation", "output_kind",
        "postprocess", "nms", "end2end",
    )
    missing = [key for key in required if key not in native_evidence]
    if missing:
        raise ValueError(f"Native contract evidence is incomplete: {missing}")
    for key in required:
        if native_evidence.get(key) != expected_contract.get(key):
            raise ValueError(f"Native contract evidence differs for {model_label}: {key}")
    dummy_primary = type(
        "ShapeOnly", (), {"shape": expected_contract["primary_output_shape"], "dtype": expected_contract.get("primary_output_dtype", "unknown")}
    )()
    dummy_head = {"one2one": {}} if model_label == "yolo26n" else {}
    return validate_native_output(model_label, (dummy_primary, dummy_head), expected_contract)


def calibration_recipe_evidence(repo: Path, config: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    helper = repo_path(repo, Path(config["calibration_recipe"]["preprocessing_helper"]))
    recipe = dict(config["calibration_recipe"])
    recipe.update({
        "status": "unresolved_pending_server_prepare",
        "helper_path": str(helper),
        "helper_sha256": sha256_file(helper),
        "materialization_checked": True,
        "materialized_tensor_file_created": False,
        "source_manifest_ids": [row["id"] for row in current["calibrations"]],
        "file_order_binding": "canonical manifest file order; producer preprocessing must be resolved before cache creation",
        "unresolved_until": "server preflight producer decoder/color/letterbox/pad/interpolation/layout/dtype/normalization/batch evidence",
    })
    return recipe


def _load_onnx(path: Path) -> tuple[Any, dict[str, Any]]:
    onnx = importlib.import_module("onnx")
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    graph = model.graph
    nodes = graph_node_records(graph)
    schema = {
        "inputs": [{"name": row.name, "shape": _value_shape(row)} for row in graph.input],
        "outputs": graph_output_records(graph),
        "value_info_count": len(graph.value_info),
        "node_count": len(graph.node),
        "op_types": sorted({node["op_type"] for node in nodes}),
        "quantization_nodes": [node["name"] for node in nodes if node["op_type"] in ("QuantizeLinear", "DequantizeLinear")],
    }
    if schema["quantization_nodes"]:
        raise ValueError("Prepare export contains Q/DQ; expected float ONNX")
    return model, schema


def producer_source_evidence() -> dict[str, Any]:
    evidence = package_evidence(("ultralytics", "onnx", "onnxslim"))
    try:
        ultralytics = importlib.import_module("ultralytics")
        from ultralytics import YOLO  # imported only in the model child
        evidence["ultralytics"]["YOLO_source_sha256"] = sha256_bytes(inspect.getsource(YOLO).encode("utf-8"))
        evidence["ultralytics"]["module_version"] = getattr(ultralytics, "__version__", None)
    except Exception as exc:
        evidence["ultralytics"]["source_error"] = f"{type(exc).__name__}:{exc}"
    return evidence


def default_exporter(staged_weights: Path, destination: Path) -> dict[str, Any]:
    """Export one frozen checkpoint from a private copy; never build an engine."""
    from ultralytics import YOLO

    model = YOLO(str(staged_weights), task="detect")
    reported = model.export(**EXPORT_ARGUMENTS)
    reported_path = Path(reported).resolve()
    private_root = staged_weights.parent.resolve()
    try:
        reported_path.relative_to(private_root)
    except ValueError as exc:
        raise ValueError(f"Exporter wrote outside private workspace: {reported_path}") from exc
    if not reported_path.is_file():
        raise FileNotFoundError(f"Ultralytics reported a missing ONNX export: {reported_path}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite ONNX output: {destination}")
    shutil.copy2(reported_path, destination)
    return {
        "reported_path": str(reported_path),
        "destination": str(destination),
        "effective_arguments": dict(EXPORT_ARGUMENTS),
    }


def prepare_model(
    repo: Path,
    config: dict[str, Any],
    accepted: dict[str, Any],
    current: dict[str, Any],
    model_label: str,
    model_dir: Path,
    exporter: Callable[[Path, Path], dict[str, Any]] | None = None,
    graph_loader: Callable[[Path], tuple[Any, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    model_config = next(row for row in config["models"] if row["label"] == model_label)
    accepted_model = next(row for row in accepted["manifest"]["model_contracts"] if row["label"] == model_label)
    if model_dir.exists():
        raise FileExistsError(f"Model output exists; refusing resume/overwrite: {model_dir}")
    model_dir.mkdir(parents=True)
    private_dir = model_dir / "private"
    private_dir.mkdir()
    checkpoint = repo_path(repo, Path(model_config["checkpoint"]))
    staged = private_dir / "frozen_source.pt"
    shutil.copy2(checkpoint, staged)
    if sha256_file(staged) != model_config["sha256"]:
        raise ValueError("Private frozen checkpoint copy hash mismatch")

    env = environment_evidence()
    env["git_commit"] = git_head(repo)
    current_probe = readiness.inspect_frozen_model(
        repo, model_config, probe=True, runtime_contract=config["runtime"], observed_environment=env
    )
    if current_probe.get("status") != "verified":
        raise ValueError(f"Native frozen model recheck failed for {model_label}: {current_probe.get('errors')}")
    accepted_evidence = accepted_model.get("head_output_evidence", {})
    for key in ("head_type", "head_index", "end2end", "active_branches", "primary_output_shape", "output_kind", "primary_output_representation"):
        observed = current_probe.get("head_output_evidence", {}).get(key)
        if observed != accepted_evidence.get(key):
            raise ValueError(f"Native model contract changed for {model_label}: {key}")
    native_adapter_validation = validate_native_contract_evidence(
        model_label, current_probe["head_output_evidence"], model_config["expected_contract"]
    )

    onnx_path = model_dir / "model.onnx"
    export_record = (exporter or default_exporter)(staged, onnx_path)
    if not onnx_path.is_file():
        raise FileNotFoundError("Exporter did not leave the expected model.onnx")
    loaded_object, schema = (graph_loader or _load_onnx)(onnx_path)
    # The real loader returns an ONNX ModelProto; tests may return a fixture
    # graph directly.  Resolve both forms without importing ONNX in the parent.
    loaded_graph = getattr(loaded_object, "graph", loaded_object)
    mapping = audit_graph_mapping(loaded_graph, model_label, accepted_model)
    if mapping["status"] != "verified":
        raise ValueError(f"Graph mapping unresolved for {model_label}: {mapping['errors']}")
    mapping["source_onnx_sha256"] = sha256_file(onnx_path)
    adapter = adapter_contract(model_label, model_config["expected_contract"])
    schema_path = model_dir / "graph_schema.json"
    write_json_no_overwrite(schema_path, {
        "schema_version": 1,
        "model": model_label,
        "onnx_sha256": sha256_file(onnx_path),
        "schema": schema,
        "mapping": mapping,
        "adapter": adapter,
    })
    record = {
        "schema_version": 1,
        "study": PREPARE_STUDY,
        "model": model_label,
        "status": "completed",
        "checkpoint": {
            "path": model_config["checkpoint"],
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "private_copy_sha256": sha256_file(staged),
        },
        "native_contract_recheck": current_probe,
        "native_adapter_validation": native_adapter_validation,
        "export": {
            **export_record,
            "source_weights_sha256": sha256_file(checkpoint),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_bytes": onnx_path.stat().st_size,
            "export_arguments": dict(EXPORT_ARGUMENTS),
            "preserved_end2end": bool(model_config["expected_contract"]["end2end"]),
        },
        "graph_schema": schema,
        "graph_mapping": mapping,
        "adapter": adapter,
        "calibration_recipe": calibration_recipe_evidence(repo, config, current),
        "environment": {
            **env,
            "producer_source_evidence": producer_source_evidence(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "tensorrt_imported": False,
        "cuda_touched": False,
        "build_performed": False,
        "capture_performed": False,
        "scored_run_authorized": False,
        "private_workspace": str(private_dir),
    }
    write_json_no_overwrite(model_dir / "model_prepare.json", record)
    return record


def output_inventory(root: Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = path.relative_to(root).as_posix()
        server_only = path.suffix.lower() in {".onnx", ".pt", ".engine", ".cache", ".npy"} or relative.startswith("models/") and "/private/" in relative
        inventory.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "server_only": server_only,
            "commit_allowed": not server_only,
        })
    return inventory


def preparation_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Precision-head graph preparation",
        "",
        f"- Study: `{manifest['study']}`",
        f"- Status: `{manifest['status']}`",
        f"- Models: `{', '.join(manifest['selected_models'])}`",
        "- Scope: prepare-only; no TensorRT import/build, no CUDA, no calibration tensor materialization, no scored matrix.",
        "",
        "## Model results",
        "",
    ]
    for row in manifest["models"]:
        mapping = row.get("graph_mapping", {})
        lines.append(
            f"- `{row['model']}`: status `{row.get('status')}`, ONNX `{row.get('export', {}).get('onnx_sha256', 'not-written')}`, "
            f"mapping `{mapping.get('mapping_status', 'unknown')}`, adapter `{row.get('adapter', {}).get('scored_adapter_status', 'unknown')}`."
        )
    lines += [
        "",
        "## Contract boundary",
        "",
        "- Native frozen head/output was rechecked before export; YOLOv8n remains raw/non-end2end and YOLO26n remains native end2end/top-k.",
        "- Branch ownership is derived from active source lineage and a branch-owned channel merge. Shared downstream reachability is recorded and is not treated as branch overlap for YOLO26n.",
        "- Real TensorRT parser/build/forward evidence is deferred. `scored_run_authorized` remains `false`.",
        "- Calibration preprocessing remains unresolved until the reviewed server preflight records the actual producer recipe; no tensor file was created here.",
        "",
        "## Output policy",
        "",
        "- JSON/report/log/schema files are the review artifacts. ONNX and private source copies are server-only binaries and must not be staged to Git.",
        "- A failed child keeps its partial model directory and writes a structured failure report. Existing output roots are never resumed or overwritten.",
    ]
    return "\n".join(lines) + "\n"


def _failure_payload(root: Path, stage: str, exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": PREPARE_STUDY,
        "status": "failed",
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "partial_files": _actual_files(root) if root.exists() else [],
        "no_silent_resume": True,
        "scored_run_authorized": False,
        "tensorrt_imported": False,
        "cuda_touched": False,
    }


def child_command(repo: Path, model: str, readiness_root: Path, out_dir: Path) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()),
        "--child", "--model", model,
        "--readiness-root", str(readiness_root),
        "--out-dir", str(out_dir),
    ]


def run_child(args: argparse.Namespace, repo: Path) -> int:
    out_dir = repo_path(repo, args.out_dir)
    readiness_root = repo_path(repo, args.readiness_root)
    model_dir = out_dir / "models" / args.model
    try:
        accepted = validate_readiness_artifact(repo, readiness_root)
        binding = validate_config_binding(repo, accepted)
        config = binding["config"]
        current = validate_current_bindings(repo, config, accepted, args.model)
        record = prepare_model(repo, config, accepted, current, args.model, model_dir)
        print(f"DONE MODEL {args.model}: {model_dir / 'model_prepare.json'}", flush=True)
        return 0 if record["status"] == "completed" else 1
    except Exception as exc:
        model_dir.mkdir(parents=True, exist_ok=True)
        write_json_no_overwrite(model_dir / "failure.json", _failure_payload(model_dir, "model_prepare", exc))
        print(f"FAILED MODEL {args.model}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


def run_parent(args: argparse.Namespace, repo: Path) -> int:
    out_dir = repo_path(repo, args.out_dir)
    readiness_root = repo_path(repo, args.readiness_root)
    if out_dir.exists():
        raise FileExistsError(f"Output exists; refusing overwrite/resume: {out_dir}")
    accepted = validate_readiness_artifact(repo, readiness_root)
    binding = validate_config_binding(repo, accepted)
    config = binding["config"]
    current = validate_current_bindings(repo, config, accepted, args.model)
    out_dir.mkdir(parents=True)
    write_json_no_overwrite(out_dir / "prepare_plan.json", {
        "schema_version": 1,
        "study": PREPARE_STUDY,
        "status": "dispatching_model_children",
        "selected_models": current["selected_models"],
        "readiness_artifact": accepted["files"],
        "config_binding": {key: value for key, value in binding.items() if key != "config"},
        "current_bindings": {
            "checkpoint_bindings": current["checkpoint_bindings"],
            "selected_calibration_ids": current["selected_calibration_ids"],
            "dataset_status": current["dataset"]["status"],
        },
        "dispatch": "one isolated CPU child per selected frozen model; no build/capture dispatch",
        "scored_run_authorized": False,
    })
    model_records = []
    for model_label in current["selected_models"]:
        completed = subprocess.run(
            child_command(repo, model_label, readiness_root, out_dir),
            cwd=repo, capture_output=True, text=True, check=False,
        )
        log_path = out_dir / "logs" / f"{model_label}.log"
        write_text_no_overwrite(log_path, (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else ""))
        model_json = out_dir / "models" / model_label / "model_prepare.json"
        if completed.returncode != 0 or not model_json.is_file():
            failure = _failure_payload(out_dir, f"child:{model_label}", RuntimeError(f"model child exited {completed.returncode}"))
            failure["child_stdout"] = completed.stdout or ""
            failure["child_stderr"] = completed.stderr or ""
            write_json_no_overwrite(out_dir / "failure.json", failure)
            raise RuntimeError(f"Prepare child failed for {model_label}; partial output preserved")
        model_records.append(read_json(model_json))

    manifest = {
        "schema_version": 1,
        "study": PREPARE_STUDY,
        "phase": "server_prepare_only",
        "status": "graph_preparation_completed_review_required" if all(row.get("status") == "completed" and row.get("graph_mapping", {}).get("status") == "verified" for row in model_records) else "unresolved",
        "selected_models": current["selected_models"],
        "model_count": len(model_records),
        "models": model_records,
        "readiness_artifact": {
            "accepted_git_commit": accepted["accepted_git_commit"],
            "root": accepted["root"],
            "files": accepted["files"],
        },
        "config_binding": {key: value for key, value in binding.items() if key != "config"},
        "current_bindings": {
            "checkpoint_bindings": current["checkpoint_bindings"],
            "calibration_ids": current["selected_calibration_ids"],
            "dataset_status": current["dataset"]["status"],
            "train_dev_overlap": current["dataset"].get("train_dev_overlap", []),
            "dev_test_overlap": current["dataset"].get("dev_test_overlap", []),
        },
        "export_contract": config["export_contract"],
        "calibration_recipe": calibration_recipe_evidence(repo, config, current),
        "environment": {
            **parent_environment_evidence(),
            "git_commit": git_head(repo),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "gpu_used": False,
        "cuda_touched": False,
        "tensorrt_imported": False,
        "tensorrt_build_performed": False,
        "build_matrix_performed": False,
        "capture_performed": False,
        "scored_run_authorized": False,
        "output_inventory": output_inventory(out_dir),
        "limitations": [
            "ONNX graph evidence is prepare-phase evidence, not TensorRT parser/build/forward evidence.",
            "Calibration preprocessing remains unresolved until a reviewed server preflight records the actual producer recipe.",
            "ONNX/private source binaries are server-only and must not be committed.",
        ],
    }
    write_text_no_overwrite(out_dir / "report.md", preparation_report(manifest))
    manifest["output_inventory"] = output_inventory(out_dir)
    write_json_no_overwrite(out_dir / "graph_preparation_manifest.json", manifest)
    print(f"DONE: {out_dir / 'graph_preparation_manifest.json'}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-root", type=Path, default=DEFAULT_READINESS_ROOT)
    parser.add_argument("--model", choices=(*MODEL_CHOICES, "all"), default="all")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if args.child:
        if args.model == "all":
            raise ValueError("Child dispatch requires exactly one model")
        return run_child(args, repo)
    return run_parent(args, repo)


if __name__ == "__main__":
    raise SystemExit(main())
