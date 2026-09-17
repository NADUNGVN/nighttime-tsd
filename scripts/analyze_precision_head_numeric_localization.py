#!/usr/bin/env python3
"""Localize the accepted numeric_v2 disagreement without changing its gate.

This is a bounded diagnostic for exactly two already-observed cases:
YOLOv8n/00009 and YOLO26n/00006.  It preserves the numeric_v2 strict result,
uses one ordinary native forward per model, one original ONNX run per model,
and one private derived-graph ONNX run for YOLO26n.  The derived graph only
adds existing intermediate tensors as outputs; it does not edit nodes,
weights, attributes or the accepted ONNX file.

The parent does input/inventory checks without importing Torch, ONNX Runtime
or ONNX.  Child failures are model-scoped.  Private arrays/derived ONNX are
never part of the publishable inventory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

import verify_precision_head_confirmation_numeric as numeric


STUDY = "precision_head_numeric_localization_v1"
OUTPUT_ROOT_NAME = "precision_head_numeric_localization_v1"
NUMERIC_ROOT = Path("results/measurement_audit_v1/precision_head_confirmation_numeric_v2")
NUMERIC_V1_ROOT = Path("results/measurement_audit_v1/precision_head_confirmation_numeric_v1")
DEFAULT_OUTPUT = Path("results/measurement_audit_v1") / OUTPUT_ROOT_NAME
TARGETS = {"yolov8n": "00009", "yolo26n": "00006"}
V8_OFFENDER_INDICES = ([0, 1, 8004], [0, 1, 8014])
V26_INTERNAL_OUTPUTS = {
    "preselection_merged": "/model.23/Concat_3_output_0",
    "preselection_transposed": "/model.23/Transpose_output_0",
    "preselection_boxes": "/model.23/Split_output_0",
    "preselection_scores": "/model.23/Split_output_1",
    "stage1_values": "/model.23/TopK_output_0",
    "stage1_indices": "/model.23/TopK_output_1",
    "stage2_values": "/model.23/TopK_1_output_0",
    "stage2_indices": "/model.23/TopK_1_output_1",
    "selected_class_scores": "/model.23/GatherElements_output_0",
    "selected_anchor_rank": "/model.23/Add_2_output_0",
    "selected_class_id": "/model.23/Mod_output_0",
    "selected_anchor_indices": "/model.23/Expand_1_output_0",
    "selected_boxes": "/model.23/GatherElements_1_output_0",
}
V26_TRACE_NODES = (
    "/model.23/Concat_3",
    "/model.23/Transpose",
    "/model.23/Split",
    "/model.23/ReduceMax",
    "/model.23/TopK",
    "/model.23/GatherElements",
    "/model.23/Flatten",
    "/model.23/TopK_1",
    "/model.23/Div_1",
    "/model.23/Mod",
    "/model.23/Gather_3",
    "/model.23/GatherElements_1",
    "/model.23/Concat_6",
)
V26_LINEAGE_OUTPUTS = {
    "/model.23/Concat_3": ("/model.23/Concat_3_output_0",),
    "/model.23/Transpose": ("/model.23/Transpose_output_0",),
    "/model.23/Split": ("/model.23/Split_output_0", "/model.23/Split_output_1"),
    "/model.23/TopK": ("/model.23/TopK_output_0", "/model.23/TopK_output_1"),
    "/model.23/GatherElements": ("/model.23/GatherElements_output_0",),
    "/model.23/TopK_1": ("/model.23/TopK_1_output_0", "/model.23/TopK_1_output_1"),
    "/model.23/Div_1": ("/model.23/Add_2_output_0",),
    "/model.23/Mod": ("/model.23/Mod_output_0",),
    "/model.23/Gather_3": ("/model.23/Expand_1_output_0",),
    "/model.23/GatherElements_1": ("/model.23/GatherElements_1_output_0",),
    "/model.23/Concat_6": ("output0",),
}
V26_EXPECTED_INTERNAL_METADATA = {
    "preselection_merged": {"shape": [1, 7, 8400], "dtype": "float32"},
    "preselection_transposed": {"shape": [1, 8400, 7], "dtype": "float32"},
    "preselection_boxes": {"shape": [1, 8400, 4], "dtype": "float32"},
    "preselection_scores": {"shape": [1, 8400, 3], "dtype": "float32"},
    "stage1_values": {"shape": [1, 300], "dtype": "float32"},
    "stage1_indices": {"shape": [1, 300], "dtype": "int64"},
    "stage2_values": {"shape": [1, 300], "dtype": "float32"},
    "stage2_indices": {"shape": [1, 300], "dtype": "int64"},
    "selected_class_scores": {"shape": [1, 300, 3], "dtype": "float32"},
    "selected_anchor_rank": {"shape": [1, 300], "dtype": "int64"},
    "selected_class_id": {"shape": [1, 300], "dtype": "int64"},
    "selected_anchor_indices": {"shape": [1, 300, 1], "dtype": "int64"},
    "selected_boxes": {"shape": [1, 300, 4], "dtype": "float32"},
}
CANONICAL_NUMERIC_V2_HASHES = {
    "numeric_manifest.json": "0d28fc3261ad4fa42baa8459c449c50e98777f9c3a43123f288109159523787d",
    "numeric_plan.json": "05216d73d69b0a9f5d621aee2fbd3a444d5e6cdc9a11050a4c6a709970c06831",
    "report.md": "5bcb70399bb634e6e449ea54ad36850405dae6e19d574afb259e88f74abb56eb",
}


class LocalizationUnresolved(RuntimeError):
    """Raised when the bounded localization cannot preserve an association."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_no_overwrite(path: Path, value: Any) -> None:
    numeric.write_json_no_overwrite(path, value)


def write_text_no_overwrite(path: Path, value: str) -> None:
    numeric.write_text_no_overwrite(path, value)


def repo_path(repo: Path, value: Path) -> Path:
    return numeric.repo_path(repo, value)


def file_evidence(repo: Path, path: Path) -> dict[str, Any]:
    return numeric.file_evidence(repo, path)


def canonical_json(value: Any) -> bytes:
    return numeric.canonical_json(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def bounded_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def array_summary(value: Any, role: str) -> dict[str, Any]:
    array = np.asarray(value)
    flat = array.reshape(-1)
    finite = flat[np.isfinite(flat)] if np.issubdtype(array.dtype, np.floating) else flat
    result: dict[str, Any] = {
        "role": role,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "element_count": int(array.size),
        "finite_count": int(np.isfinite(flat).sum()) if np.issubdtype(array.dtype, np.number) else None,
        "zero_count": int(np.equal(flat, 0).sum()) if np.issubdtype(array.dtype, np.number) else None,
        "sha256": sha256_bytes(np.ascontiguousarray(array).tobytes(order="C")),
    }
    if finite.size and np.issubdtype(array.dtype, np.number):
        result.update({"min": float(np.min(finite)), "max": float(np.max(finite)), "p50": float(np.quantile(finite, 0.5)), "mean": float(np.mean(finite))})
    else:
        result.update({"min": None, "max": None, "p50": None, "mean": None})
    return result


def to_numpy(value: Any, np_module: Any = np) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().to("cpu").contiguous().numpy()
    return np_module.ascontiguousarray(value)


def normalize_preselection(value: Any, channels: int, role: str, np_module: Any = np) -> tuple[Any, dict[str, Any]]:
    array = to_numpy(value, np_module)
    if array.ndim != 3 or array.shape[0] != 1:
        raise LocalizationUnresolved(f"{role} does not have batch shape 1: {list(array.shape)}")
    if array.shape[1] == channels:
        normalized = np_module.ascontiguousarray(array.transpose(0, 2, 1))
        layout = "BCHW-like channel-first to BNC"
    elif array.shape[2] == channels:
        normalized = array
        layout = "BNC channel-last"
    else:
        raise LocalizationUnresolved(f"{role} has no channel axis {channels}: {list(array.shape)}")
    if normalized.dtype != np_module.dtype("float32"):
        raise LocalizationUnresolved(f"{role} is not float32: {normalized.dtype}")
    return normalized, {"role": role, "original": array_summary(array, role), "normalized": array_summary(normalized, role), "layout": layout}


def allowed_error(reference: float, observed: float) -> dict[str, Any]:
    absolute = abs(observed - reference)
    allowance = numeric.LOCKED_TOLERANCES["float32"]["atol"] + numeric.LOCKED_TOLERANCES["float32"]["rtol"] * abs(reference)
    ratio = absolute / allowance if allowance else (0.0 if absolute == 0 else float("inf"))
    return {"reference": float(reference), "observed": float(observed), "absolute_error": float(absolute), "allowed_error": float(allowance), "error_allowance_ratio": bounded_float(ratio) if np.isfinite(ratio) else "inf", "status": "pass" if absolute <= allowance else "fail"}


def v8_offender_report(reference: Any, observed: Any, np_module: Any = np) -> dict[str, Any]:
    reference = np_module.asarray(reference)
    observed = np_module.asarray(observed)
    if reference.shape != observed.shape or list(reference.shape) != numeric.EXPECTED_OUTPUT_SHAPES["yolov8n"]:
        raise LocalizationUnresolved(f"YOLOv8 primary shape mismatch: {reference.shape} versus {observed.shape}")
    scalar_rows = []
    for index in V8_OFFENDER_INDICES:
        row = {"index": list(index), "box": allowed_error(float(reference[tuple(index)]), float(observed[tuple(index)]))}
        anchor = index[-1]
        row["anchor_class_scores"] = [{"class_index": class_index, **allowed_error(float(reference[0, 4 + class_index, anchor]), float(observed[0, 4 + class_index, anchor]))} for class_index in range(3)]
        scalar_rows.append(row)
    strict = numeric.compare_primary("yolov8n", reference, observed, np_module)
    box = strict["channel_results"]["boxes"]
    score = strict["channel_results"]["scores"]
    return {
        "historical_offender_indices": scalar_rows,
        "strict_rerun_comparison": strict,
        "all_new_mismatches": {"box_mismatch_count": box.get("mismatch_count"), "score_mismatch_count": score.get("mismatch_count"), "total_mismatch_count": strict.get("mismatch_count"), "first_box_offenders": box.get("first_offenders", []), "first_score_offenders": score.get("first_offenders", [])},
        "interpretation_boundary": "scalar and channel evidence is supplemental; the numeric_v2 strict FAIL remains unchanged",
    }


def stable_topk_numpy(values: Any, k: int) -> tuple[Any, Any]:
    array = np.asarray(values)
    order = np.argsort(-array, axis=-1, kind="stable")[..., :k]
    return np.take_along_axis(array, order, axis=-1), order.astype(np.int64)


def topk_with_runtime(values: Any, k: int, torch_module: Any | None = None, topk_impl: Callable[..., tuple[Any, Any]] | None = None) -> tuple[Any, Any, str]:
    if topk_impl is not None:
        result = topk_impl(values, k)
        return np.asarray(result[0]), np.asarray(result[1], dtype=np.int64), "injected_topk_for_local_test"
    if torch_module is None or not hasattr(torch_module, "topk") or not hasattr(torch_module, "from_numpy"):
        raise LocalizationUnresolved("Installed torch.topk is required to reconstruct native YOLO26 selection")
    tensor = torch_module.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    selected = torch_module.topk(tensor, k, dim=-1, largest=True, sorted=True)
    return to_numpy(selected.values), to_numpy(selected.indices).astype(np.int64), "installed_torch.topk(largest=true,sorted=true)"


def reconstruct_yolo26_selection(boxes_bnc: Any, scores_bnc: Any, *, torch_module: Any | None = None, topk_impl: Callable[..., tuple[Any, Any]] | None = None) -> dict[str, Any]:
    boxes = np.asarray(boxes_bnc)
    scores = np.asarray(scores_bnc)
    if boxes.shape[0] != 1 or scores.shape[0] != 1 or boxes.shape[1] != scores.shape[1] or scores.shape[2] != 3 or boxes.shape[2] != 4:
        raise LocalizationUnresolved(f"YOLO26 preselection shapes cannot be associated: boxes={boxes.shape}, scores={scores.shape}")
    anchor_scores = scores.max(axis=-1)
    stage1_values, stage1_indices, topk_source = topk_with_runtime(anchor_scores, min(300, scores.shape[1]), torch_module, topk_impl)
    selected_scores = np.take_along_axis(scores, stage1_indices[..., None], axis=1)
    flat_scores = selected_scores.reshape(1, -1)
    stage2_values, stage2_flat_indices, _ = topk_with_runtime(flat_scores, min(300, flat_scores.shape[-1]), torch_module, topk_impl)
    anchor_rank = stage2_flat_indices // 3
    class_id = stage2_flat_indices % 3
    original_anchor = np.take_along_axis(stage1_indices, anchor_rank, axis=1)
    selected_boxes = np.take_along_axis(boxes, np.repeat(original_anchor[..., None], 4, axis=-1), axis=1)
    primary = np.concatenate((selected_boxes, stage2_values[..., None], class_id[..., None].astype(np.float32)), axis=-1)
    return {
        "primary": primary.astype(np.float32, copy=False),
        "stage1_values": stage1_values,
        "stage1_indices": stage1_indices,
        "stage2_values": stage2_values,
        "stage2_flat_indices": stage2_flat_indices,
        "anchor_rank": anchor_rank,
        "class_id": class_id,
        "original_anchor": original_anchor,
        "topk_source": topk_source,
        "flattening": {"class_count": 3, "flat_index_formula": "anchor_rank * 3 + class_id", "class_formula": "flat_index % 3", "anchor_rank_formula": "flat_index // 3"},
    }


def selection_pairs(stage2_flat_indices: Any, stage1_indices: Any) -> list[tuple[int, int]]:
    flat = np.asarray(stage2_flat_indices).reshape(-1)
    anchors = np.asarray(stage1_indices).reshape(-1)
    pairs = []
    for value in flat:
        index = int(value)
        rank, class_id = divmod(index, 3)
        if rank >= anchors.size:
            raise LocalizationUnresolved(f"Flattened class index rank is out of range: {index}")
        pairs.append((int(anchors[rank]), int(class_id)))
    return pairs


def validate_selection_mapping(selection: dict[str, Any], label: str, np_module: Any = np) -> dict[str, Any]:
    stage1_indices = np_module.asarray(selection["stage1_indices"], dtype=np_module.int64)
    stage2_flat = np_module.asarray(selection["stage2_flat_indices"], dtype=np_module.int64)
    expected_rank = stage2_flat // 3
    expected_class = stage2_flat % 3
    expected_anchor = np_module.take_along_axis(stage1_indices, expected_rank, axis=1)
    actual_rank = np_module.asarray(selection["anchor_rank"], dtype=np_module.int64)
    actual_class = np_module.asarray(selection["class_id"], dtype=np_module.int64)
    actual_anchor = np_module.asarray(selection["original_anchor"], dtype=np_module.int64)
    if actual_anchor.ndim == 3 and actual_anchor.shape[-1] == 1:
        actual_anchor = actual_anchor[..., 0]
    if not np_module.array_equal(actual_rank, expected_rank):
        raise LocalizationUnresolved(f"{label} anchor-rank arithmetic disagrees with stage2 flat indices")
    if not np_module.array_equal(actual_class, expected_class):
        raise LocalizationUnresolved(f"{label} class-id arithmetic disagrees with stage2 flat indices")
    if not np_module.array_equal(actual_anchor, expected_anchor):
        raise LocalizationUnresolved(f"{label} original-anchor Gather disagrees with stage1 indices")
    return {"status": "exact", "class_formula": "flat_index % 3", "anchor_rank_formula": "flat_index // 3", "original_anchor_formula": "stage1_indices[anchor_rank]", "checked_count": int(stage2_flat.size)}


def reconstruct_yolo26_from_indices(boxes_bnc: Any, scores_bnc: Any, selection: dict[str, Any], np_module: Any = np) -> dict[str, Any]:
    mapping = validate_selection_mapping(selection, "YOLO26 direct selection", np_module)
    boxes = np_module.asarray(boxes_bnc)
    scores = np_module.asarray(scores_bnc)
    original_anchor = np_module.asarray(selection["original_anchor"], dtype=np_module.int64)
    if original_anchor.ndim == 3 and original_anchor.shape[-1] == 1:
        original_anchor = original_anchor[..., 0]
    class_id = np_module.asarray(selection["class_id"], dtype=np_module.int64)
    selected_boxes = np_module.take_along_axis(boxes, np_module.repeat(original_anchor[..., None], 4, axis=-1), axis=1)
    selected_scores = np_module.take_along_axis(scores, np_module.repeat(original_anchor[..., None], 3, axis=-1), axis=1)
    selected_class_scores = np_module.take_along_axis(selected_scores, class_id[..., None], axis=2)[..., 0]
    primary = np_module.concatenate((selected_boxes, selected_class_scores[..., None], class_id[..., None].astype(np_module.float32)), axis=-1)
    stage2_values = np_module.asarray(selection["stage2_values"], dtype=np_module.float32)
    value_consistency = numeric.compare_float_arrays(stage2_values, selected_class_scores, np_module, "yolo26_stage2_value_reconstruction")
    return {"primary": primary.astype(np_module.float32, copy=False), "selected_class_scores": selected_class_scores, "mapping_validation": mapping, "stage2_value_consistency": value_consistency}


def summarize_selection(stage1_indices: Any, stage2_flat_indices: Any, anchor_rank: Any, class_id: Any, original_anchor: Any, source: str) -> dict[str, Any]:
    pairs = selection_pairs(stage2_flat_indices, stage1_indices)
    return {
        "source": source,
        "stage1_count": int(np.asarray(stage1_indices).size),
        "stage2_count": int(np.asarray(stage2_flat_indices).size),
        "stage1_index_summary": array_summary(stage1_indices, "stage1 original-anchor candidates"),
        "stage2_flat_index_summary": array_summary(stage2_flat_indices, "stage2 flattened class indices"),
        "anchor_rank_summary": array_summary(anchor_rank, "stage2 selected anchor rank"),
        "class_id_summary": array_summary(class_id, "stage2 selected class id"),
        "original_anchor_summary": array_summary(original_anchor, "stage2 original anchor indices"),
        "first_pairs": [{"anchor": anchor, "class_id": class_id_value} for anchor, class_id_value in pairs[:10]],
        "pair_set_sha256": sha256_bytes(canonical_json(sorted([[anchor, class_id_value] for anchor, class_id_value in pairs]))),
    }


def compare_selection_alignment(native: dict[str, Any], onnx: dict[str, Any]) -> dict[str, Any]:
    native_pairs = selection_pairs(native["stage2_flat_indices"], native["stage1_indices"])
    onnx_pairs = selection_pairs(onnx["stage2_flat_indices"], onnx["stage1_indices"])
    native_set, onnx_set = set(native_pairs), set(onnx_pairs)
    overlap = native_set & onnx_set
    order_equal = native_pairs == onnx_pairs
    result = {
        "status": "resolved",
        "native_count": len(native_pairs),
        "onnx_count": len(onnx_pairs),
        "overlap_count": len(overlap),
        "native_unique_count": len(native_set),
        "onnx_unique_count": len(onnx_set),
        "same_selected_set": native_set == onnx_set,
        "same_selected_set_permutation": native_set == onnx_set and not order_equal,
        "same_order": order_equal,
        "different_selected_set_membership": native_set != onnx_set,
        "native_only_examples": [{"anchor": a, "class_id": c} for a, c in sorted(native_set - onnx_set)[:10]],
        "onnx_only_examples": [{"anchor": a, "class_id": c} for a, c in sorted(onnx_set - native_set)[:10]],
        "association_rule": "exact (original_anchor,class_id) pair; no nearest-neighbor/rematching",
    }
    return result


def compare_preselection(native_boxes: Any, native_scores: Any, onnx_boxes: Any, onnx_scores: Any, np_module: Any = np) -> dict[str, Any]:
    box = numeric.compare_float_arrays(np_module.asarray(native_boxes), np_module.asarray(onnx_boxes), np_module, "yolo26_same_anchor_decoded_boxes")
    score = numeric.compare_float_arrays(np_module.asarray(native_scores), np_module.asarray(onnx_scores), np_module, "yolo26_same_anchor_class_probabilities")
    return {
        "association": "same anchor index in each model's own pre-selection tensor; no cross-model anchor invention",
        "boxes": box,
        "class_probabilities": score,
        "same_anchor_numerical_error": {"boxes": box.get("status") == "fail", "class_probabilities": score.get("status") == "fail"},
        "logit_probability_boundary": "only mapped class probabilities are compared; raw logits are not compared to sigmoid probabilities",
    }


def _attribute_value(attribute: Any) -> Any:
    kind = getattr(attribute, "type", None)
    if kind == 2 or hasattr(attribute, "i") and not getattr(attribute, "ints", None):
        return int(attribute.i)
    if kind == 7 or getattr(attribute, "ints", None):
        return [int(value) for value in attribute.ints]
    if hasattr(attribute, "s") and attribute.s:
        return attribute.s.decode("utf-8", errors="replace")
    return None


def _node_record(node: Any) -> dict[str, Any]:
    return {"name": node.name, "op_type": node.op_type, "inputs": list(node.input), "outputs": list(node.output), "attributes": {attribute.name: _attribute_value(attribute) for attribute in node.attribute}}


def _value_info_metadata(value_info: Any) -> dict[str, Any]:
    tensor_type = getattr(getattr(value_info, "type", None), "tensor_type", None)
    if tensor_type is None:
        raise LocalizationUnresolved(f"Diagnostic tensor has no tensor type metadata: {getattr(value_info, 'name', '<unnamed>')}")
    dtype_code = int(getattr(tensor_type, "elem_type", 0))
    dtype = {1: "float32", 7: "int64"}.get(dtype_code, f"onnx_dtype_{dtype_code}")
    shape = []
    shape_proto = getattr(tensor_type, "shape", None)
    for dim in getattr(shape_proto, "dim", []):
        dim_value = int(getattr(dim, "dim_value", 0) or 0)
        dim_param = str(getattr(dim, "dim_param", "") or "")
        shape.append(dim_value if dim_value > 0 else (dim_param or None))
    return {"shape": shape, "dtype": dtype}


def validate_v26_graph_contract(model_proto: Any) -> dict[str, Any]:
    nodes = {node.name: node for node in model_proto.graph.node}
    missing_nodes = [name for name in V26_TRACE_NODES if name not in nodes]
    if missing_nodes:
        raise LocalizationUnresolved(f"YOLO26 diagnostic graph nodes missing: {missing_nodes}")
    tensors = set()
    for node in model_proto.graph.node:
        tensors.update(node.output)
    tensors.update(output.name for output in model_proto.graph.output)
    missing_outputs = [name for name in V26_INTERNAL_OUTPUTS.values() if name not in tensors]
    if missing_outputs:
        raise LocalizationUnresolved(f"YOLO26 diagnostic tensors missing: {missing_outputs}")
    if "output0" not in {output.name for output in getattr(model_proto.graph, "output", [])}:
        raise LocalizationUnresolved("YOLO26 original output0 is not a graph output")
    trace = [_node_record(nodes[name]) for name in V26_TRACE_NODES]
    lineage_errors = {name: [output for output in outputs if output not in nodes[name].output] for name, outputs in V26_LINEAGE_OUTPUTS.items()}
    lineage_errors = {name: missing for name, missing in lineage_errors.items() if missing}
    if lineage_errors:
        raise LocalizationUnresolved(f"YOLO26 diagnostic lineage is unresolved: {lineage_errors}")
    topk = {row["name"]: row["attributes"] for row in trace if row["op_type"] == "TopK"}
    if any(attrs.get("axis") != -1 or attrs.get("largest") != 1 or attrs.get("sorted") != 1 for attrs in topk.values()) or set(topk) != {"/model.23/TopK", "/model.23/TopK_1"}:
        raise LocalizationUnresolved(f"YOLO26 TopK attributes are not the accepted observed contract: {topk}")
    split = next(row for row in trace if row["name"] == "/model.23/Split")
    gather = [row for row in trace if row["op_type"] == "GatherElements"]
    if split["attributes"].get("axis") != -1 or any(row["attributes"].get("axis") != 1 for row in gather):
        raise LocalizationUnresolved("YOLO26 split/gather axis contract differs")
    value_infos = {}
    for value_info in list(getattr(model_proto.graph, "input", [])) + list(getattr(model_proto.graph, "value_info", [])) + list(getattr(model_proto.graph, "output", [])):
        value_infos[value_info.name] = value_info
    tensor_metadata = {}
    for key, name in V26_INTERNAL_OUTPUTS.items():
        value_info = value_infos.get(name)
        if value_info is None:
            raise LocalizationUnresolved(f"No shape/dtype metadata for diagnostic tensor: {name}")
        observed = _value_info_metadata(value_info)
        expected = V26_EXPECTED_INTERNAL_METADATA[key]
        if observed != expected:
            raise LocalizationUnresolved(f"YOLO26 diagnostic tensor metadata differs for {name}: observed={observed}, expected={expected}")
        tensor_metadata[key] = observed
    return {
        "model": "yolo26n",
        "original_output_name": "output0",
        "output_additions": dict(V26_INTERNAL_OUTPUTS),
        "trace_nodes": trace,
        "lineage": {name: list(outputs) for name, outputs in V26_LINEAGE_OUTPUTS.items()},
        "topk_contract": {name: attrs for name, attrs in topk.items()},
        "flattening_contract": {"class_count": 3, "stage1": "TopK(ReduceMax(class_probability), k=300, axis=-1)", "stage2": "TopK(flatten(selected_class_probabilities), k=300, axis=-1)", "class_index": "flat_index % 3", "anchor_rank": "flat_index // 3", "original_anchor": "Gather(stage1_indices, anchor_rank)"},
        "tensor_metadata": tensor_metadata,
        "score_semantics": "preselection_scores are the accepted post-Sigmoid class probabilities from the merged float32 graph; raw logits are not compared to probabilities",
        "instrumentation": "existing intermediate tensors exposed as additional graph outputs; nodes/weights/attributes unchanged",
    }


def make_v26_derived_graph(original_path: Path, derived_path: Path, onnx_module: Any) -> tuple[Any, dict[str, Any]]:
    original = onnx_module.load(str(original_path), load_external_data=True)
    onnx_module.checker.check_model(original)
    contract = validate_v26_graph_contract(original)
    existing_outputs = {output.name for output in original.graph.output}
    if existing_outputs != {"output0"}:
        raise LocalizationUnresolved(f"Original YOLO26 output set differs: {sorted(existing_outputs)}")
    derived = copy.deepcopy(original)
    value_infos = {}
    for value_info in list(derived.graph.input) + list(derived.graph.value_info) + list(derived.graph.output):
        value_infos[value_info.name] = value_info
    added = []
    for name in V26_INTERNAL_OUTPUTS.values():
        if name == "output0":
            continue
        value_info = value_infos.get(name)
        if value_info is None:
            raise LocalizationUnresolved(f"No shape/dtype metadata for diagnostic tensor: {name}")
        derived.graph.output.append(copy.deepcopy(value_info))
        added.append(name)
    if derived_path.exists():
        raise FileExistsError(f"Derived graph already exists; no overwrite: {derived_path}")
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_module.save(derived, str(derived_path))
    onnx_module.checker.check_model(derived)
    contract["added_output_names"] = added
    contract["derived_file"] = {"path": str(derived_path), "bytes": derived_path.stat().st_size, "sha256": hashlib.sha256(derived_path.read_bytes()).hexdigest()}
    return derived, contract


def _v2_target(plan: dict[str, Any], manifest: dict[str, Any], model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if manifest.get("status") != "completed" or manifest.get("execution_status") != "completed":
        raise LocalizationUnresolved(f"numeric_v2 execution is not completed: {manifest.get('status')}")
    if manifest.get("numeric_verdict", {}).get("status") != "fail":
        raise LocalizationUnresolved("numeric_v2 strict FAIL must be preserved before localization")
    row = next((item for item in manifest.get("models", []) if item.get("model") == model), None)
    if not row or row.get("status") != "completed" or row.get("numeric_status") != "fail":
        raise LocalizationUnresolved(f"numeric_v2 does not contain completed negative evidence for {model}")
    target_id = TARGETS[model]
    target = next((item for item in row.get("comparisons", []) if item.get("image_id") == target_id), None)
    if target is None or target.get("comparison", {}).get("status") != "fail":
        raise LocalizationUnresolved(f"numeric_v2 target comparison is not the accepted FAIL for {model}/{target_id}")
    return target, row


def build_plan(repo: Path, numeric_root: Path, *, file_evidence_fn=file_evidence) -> dict[str, Any]:
    if numeric_root.name != "precision_head_confirmation_numeric_v2" or not numeric_root.is_dir():
        raise FileNotFoundError(f"Accepted numeric_v2 root is missing or misnamed: {numeric_root}")
    manifest_path = numeric_root / "numeric_manifest.json"
    plan_path = numeric_root / "numeric_plan.json"
    report_path = numeric_root / "report.md"
    for path in (manifest_path, plan_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(f"Accepted numeric_v2 artifact is missing: {path}")
    manifest, numeric_plan = read_json(manifest_path), read_json(plan_path)
    v1_manifest = repo_path(repo, NUMERIC_V1_ROOT / "numeric_manifest.json")
    if not v1_manifest.is_file():
        raise FileNotFoundError(f"Preserved numeric_v1 manifest is missing: {v1_manifest}")
    targets = {}
    rows = {}
    for model in TARGETS:
        targets[model], rows[model] = _v2_target(numeric_plan, manifest, model)
        model_report = numeric_root / "models" / model / "numeric_report.json"
        if not model_report.is_file():
            raise FileNotFoundError(f"Accepted numeric_v2 model report missing: {model_report}")
    for relative, expected in numeric.EXPECTED_ONNX_SHA256.items():
        path = repo_path(repo, Path(f"results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2/models/{relative}/model.onnx"))
        evidence = file_evidence_fn(repo, path)
        if evidence.get("sha256") != expected:
            raise ValueError(f"Accepted ONNX hash differs for {relative}: {evidence.get('sha256')}")
    fixture_by_model = {}
    for model, target_id in TARGETS.items():
        fixture_by_model[model] = next(row for row in numeric_plan["fixture"]["forward_images"] if row["image_id"] == target_id)
        numeric.verify_bound_image(repo, fixture_by_model[model], file_evidence_fn)
    return {
        "schema_version": 1,
        "study": STUDY,
        "status": "dispatching_two_bounded_cases",
        "output_root": str(DEFAULT_OUTPUT.as_posix()),
        "numeric_v2_root": str(NUMERIC_ROOT.as_posix()),
        "numeric_v1_root": str(NUMERIC_V1_ROOT.as_posix()),
        "numeric_v2_execution_commit": numeric_plan.get("repo_head"),
        "numeric_v2_strict_verdict": manifest.get("numeric_verdict"),
        "numeric_v2_target_records": targets,
        "numeric_v2_model_records": {model: {"numeric_status": row.get("numeric_status"), "forward_counts": row.get("forward_counts"), "head": row.get("native_head_contract", {}).get("head")} for model, row in rows.items()},
        "fixture": fixture_by_model,
        "models": {model: numeric_plan["models"][model] for model in TARGETS},
        "canonical_numeric_v2_hashes": CANONICAL_NUMERIC_V2_HASHES,
        "numeric_v2_hash_representation": "canonical Git blob SHA256; checkout CRLF hashes must not be substituted",
        "forward_contract": {"native_model_forwards": {"yolov8n": 1, "yolo26n": 1}, "onnx_original_session_runs": {"yolov8n": 1, "yolo26n": 1}, "onnx_derived_session_runs": {"yolov8n": 0, "yolo26n": 1}, "total_native": 2, "total_onnx": 3, "failure_policy": "preserve partial; no retry or automatic extra call"},
        "diagnostic_cases": {"yolov8n": {"image_id": "00009", "selection": "U42", "purpose": "historical strict box offenders"}, "yolo26n": {"image_id": "00006", "selection": "U42", "purpose": "first canonical fixture; TopK/Gather localization"}},
        "audit_flags": {"export_performed": False, "tensorRT_imported": False, "build_performed": False, "accepted_onnx_modified": False, "gpu_used": False, "scored_run_authorized": False, "matrix_opened": False, "strict_verdict_changed": False},
        "no_overwrite": True,
    }


def _extract_native_preselection(output: Any, model: str, np_module: Any = np) -> tuple[Any, Any, dict[str, Any]]:
    if not isinstance(output, (tuple, list)) or len(output) != 2 or not isinstance(output[1], dict):
        raise LocalizationUnresolved(f"{model} native output is not the accepted tuple-plus-dict")
    debug = output[1]
    branch = debug if model == "yolov8n" else debug.get("one2one")
    if not isinstance(branch, dict) or any(key not in branch for key in ("boxes", "scores")):
        raise LocalizationUnresolved(f"{model} native returned branch lacks boxes/scores")
    boxes, box_meta = normalize_preselection(branch["boxes"], 4, f"{model}.native.boxes", np_module)
    scores, score_meta = normalize_preselection(branch["scores"], 3, f"{model}.native.class_probabilities", np_module)
    return boxes, scores, {"capture_site": "ordinary frozen model forward returned head dictionary", "branch": "raw" if model == "yolov8n" else "one2one", "boxes": box_meta, "class_probabilities": score_meta, "debug_keys": list(debug.keys())}


def _session_run(runtime: dict[str, Any], onnx_path: Path, input_np: Any, output_names: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    options = numeric.build_session_options(runtime["ort"])
    session = runtime["ort"].InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    providers = list(session.get_providers())
    if providers != ["CPUExecutionProvider"]:
        raise LocalizationUnresolved(f"ONNX provider is not CPU-only: {providers}")
    names = [item.name for item in session.get_outputs()]
    missing = [name for name in output_names if name not in names]
    if missing:
        raise LocalizationUnresolved(f"Requested ONNX outputs are unavailable: {missing}")
    values = session.run(output_names, {"images": input_np})
    return {name: runtime["np"].ascontiguousarray(value) for name, value in zip(output_names, values)}, {"providers_observed": providers, "providers_requested": ["CPUExecutionProvider"], "output_names": output_names, "session_output_names": names}


def _v26_onnx_selection(outputs: dict[str, Any]) -> dict[str, Any]:
    stage1_indices = np.asarray(outputs[V26_INTERNAL_OUTPUTS["stage1_indices"]], dtype=np.int64)
    stage2_flat = np.asarray(outputs[V26_INTERNAL_OUTPUTS["stage2_indices"]], dtype=np.int64)
    anchor_rank = np.asarray(outputs[V26_INTERNAL_OUTPUTS["selected_anchor_rank"]], dtype=np.int64)
    class_id = np.asarray(outputs[V26_INTERNAL_OUTPUTS["selected_class_id"]], dtype=np.int64)
    original_anchor = np.asarray(outputs[V26_INTERNAL_OUTPUTS["selected_anchor_indices"]], dtype=np.int64)
    selection = {"stage1_indices": stage1_indices, "stage2_flat_indices": stage2_flat, "anchor_rank": anchor_rank, "class_id": class_id, "original_anchor": original_anchor, "stage1_values": outputs[V26_INTERNAL_OUTPUTS["stage1_values"]], "stage2_values": outputs[V26_INTERNAL_OUTPUTS["stage2_values"]], "source": "direct ONNX TopK/Gather/Div/Mod outputs from private derived graph"}
    selection["mapping_validation"] = validate_selection_mapping(selection, "YOLO26 ONNX", np)
    return selection


def analyze_v8_case(native_primary: Any, native_boxes: Any, native_scores: Any, onnx_primary: Any, historical: dict[str, Any], np_module: Any = np) -> dict[str, Any]:
    current = v8_offender_report(native_primary, onnx_primary, np_module)
    current["historical_numeric_v2_record"] = historical
    current["native_preselection"] = {"boxes": array_summary(native_boxes, "native raw box tensor"), "class_probabilities": array_summary(native_scores, "native raw class score tensor"), "selection_stage": "not_applicable; YOLOv8 accepted primary is raw decoded fixed-anchor output"}
    current["onnx_preselection"] = {"boxes": array_summary(np_module.asarray(onnx_primary)[:, 0:4, :].transpose(0, 2, 1), "ONNX raw box tensor"), "class_probabilities": array_summary(np_module.asarray(onnx_primary)[:, 4:7, :].transpose(0, 2, 1), "ONNX raw class score tensor"), "selection_stage": "not_applicable"}
    return {"model": "yolov8n", "image_id": TARGETS["yolov8n"], "status": "completed", "numeric_status": "fail", "strict_verdict_preserved": True, "analysis": current, "forward_counts": {"native_model_forwards": 1, "onnx_original_session_runs": 1, "onnx_derived_session_runs": 0}}


def analyze_v26_case(native_primary: Any, native_boxes: Any, native_scores: Any, original_outputs: dict[str, Any], derived_outputs: dict[str, Any], graph_contract: dict[str, Any], historical: dict[str, Any], *, torch_module: Any | None = None, topk_impl: Callable[..., tuple[Any, Any]] | None = None, np_module: Any = np) -> dict[str, Any]:
    onnx_primary = original_outputs["output0"]
    native_reconstructed = reconstruct_yolo26_selection(native_boxes, native_scores, torch_module=torch_module, topk_impl=topk_impl)
    onnx_boxes, _ = normalize_preselection(derived_outputs[V26_INTERNAL_OUTPUTS["preselection_boxes"]], 4, "yolo26n.onnx.boxes", np_module)
    onnx_scores, _ = normalize_preselection(derived_outputs[V26_INTERNAL_OUTPUTS["preselection_scores"]], 3, "yolo26n.onnx.class_probabilities", np_module)
    onnx_selection = _v26_onnx_selection(derived_outputs)
    onnx_reconstructed = reconstruct_yolo26_from_indices(onnx_boxes, onnx_scores, onnx_selection, np_module)
    stage1_indices = np_module.asarray(onnx_selection["stage1_indices"], dtype=np_module.int64)
    expected_selected_scores = np_module.take_along_axis(onnx_scores, np_module.repeat(stage1_indices[..., None], 3, axis=-1), axis=1)
    expected_selected_boxes = onnx_reconstructed["primary"][..., :4]
    direct_selected_scores = np_module.asarray(derived_outputs[V26_INTERNAL_OUTPUTS["selected_class_scores"]])
    direct_selected_boxes = np_module.asarray(derived_outputs[V26_INTERNAL_OUTPUTS["selected_boxes"]])
    direct_intermediate_consistency = {
        "selected_class_scores": numeric.compare_float_arrays(expected_selected_scores, direct_selected_scores, np_module, "yolo26_direct_selected_class_scores"),
        "selected_boxes": numeric.compare_float_arrays(expected_selected_boxes, direct_selected_boxes, np_module, "yolo26_direct_selected_boxes"),
        "association": "stage1 original-anchor indices only; no rematching",
    }
    native_selection_summary = summarize_selection(native_reconstructed["stage1_indices"], native_reconstructed["stage2_flat_indices"], native_reconstructed["anchor_rank"], native_reconstructed["class_id"], native_reconstructed["original_anchor"], "reconstructed from native returned preselection tensors with installed PyTorch topk")
    onnx_selection_summary = summarize_selection(onnx_selection["stage1_indices"], onnx_selection["stage2_flat_indices"], onnx_selection["anchor_rank"], onnx_selection["class_id"], onnx_selection["original_anchor"], onnx_selection["source"])
    reconstructed_native_compare = numeric.compare_primary("yolo26n", native_reconstructed["primary"], np_module.asarray(native_primary), np_module)
    reconstructed_onnx_compare = numeric.compare_primary("yolo26n", onnx_reconstructed["primary"], np_module.asarray(derived_outputs["output0"]), np_module)
    original_derived_exact = bool(np_module.array_equal(np_module.asarray(original_outputs["output0"]), np_module.asarray(derived_outputs["output0"])))
    return {
        "model": "yolo26n", "image_id": TARGETS["yolo26n"], "status": "completed", "numeric_status": "fail", "strict_verdict_preserved": True,
        "analysis": {
            "historical_numeric_v2_record": historical,
            "strict_rerun_comparison": numeric.compare_primary("yolo26n", np_module.asarray(native_primary), np_module.asarray(onnx_primary), np_module),
            "native_preselection": {"boxes": array_summary(native_boxes, "native one2one decoded boxes"), "class_probabilities": array_summary(native_scores, "native one2one class probabilities"), "capture_site": "ordinary frozen model forward returned one2one head dictionary"},
            "onnx_preselection": {"boxes": array_summary(onnx_boxes, "ONNX pre-TopK decoded boxes"), "class_probabilities": array_summary(onnx_scores, "ONNX pre-TopK class probabilities"), "source_tensors": {key: array_summary(value, key) for key, value in derived_outputs.items() if key != "output0"}},
            "same_anchor_comparison": compare_preselection(native_boxes, native_scores, onnx_boxes, onnx_scores, np_module),
            "native_selection": native_selection_summary,
            "onnx_selection": onnx_selection_summary,
            "selection_alignment": compare_selection_alignment(native_reconstructed, onnx_selection),
            "reconstructed_outputs": {"native_from_own_tensors_and_reconstructed_indices": reconstructed_native_compare, "onnx_from_own_tensors_and_direct_indices": reconstructed_onnx_compare, "onnx_index_mapping": onnx_reconstructed["mapping_validation"], "onnx_stage2_value_consistency": onnx_reconstructed["stage2_value_consistency"], "onnx_direct_intermediate_consistency": direct_intermediate_consistency},
            "graph_trace": graph_contract,
            "instrumentation_sensitivity": {"status": "unchanged_exact" if original_derived_exact else "drift", "original_output_vs_derived_output_exact": original_derived_exact, "if_drift": "derived internals are not used as proof of original execution" if not original_derived_exact else None},
            "association_boundary": "exact anchor/class pairs only; no nearest-neighbor matching, rematching, score cutoff, NMS or row replacement",
        },
        "forward_counts": {"native_model_forwards": 1, "onnx_original_session_runs": 1, "onnx_derived_session_runs": 1},
    }


def new_child_state(model: str) -> dict[str, Any]:
    return {"model": model, "status": "running", "stage": "not_started", "stage_history": [], "forward_counts": {"native_model_forwards": {"attempted": 0, "completed": 0}, "onnx_original_session_runs": {"attempted": 0, "completed": 0}, "onnx_derived_session_runs": {"attempted": 0, "completed": 0}}}


def stage(state: dict[str, Any], name: str, **details: Any) -> None:
    state["stage"] = name
    state["stage_history"].append({"sequence": len(state["stage_history"]), "stage": name, **details})


def child_report(repo: Path, plan: dict[str, Any], model: str, out_dir: Path, *, runtime_loader=numeric.load_child_runtime, file_evidence_fn=file_evidence, source_evidence_fn=numeric.runtime_source_evidence, onnx_module_loader=None, state: dict[str, Any] | None = None) -> dict[str, Any]:
    state = state or new_child_state(model)
    stage(state, "child_started")
    numeric.set_cpu_environment()
    out_dir.mkdir(parents=True, exist_ok=True)
    private_dir = out_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    stage(state, "runtime_loading")
    config = read_json(repo_path(repo, Path(plan["models"][model].get("config_path", plan.get("config_path", "configs/precision_head_confirmation_v1.json"))))) if plan.get("config_path") else read_json(repo_path(repo, Path("configs/precision_head_confirmation_v1.json")))
    runtime = runtime_loader(config)
    stage(state, "runtime_loaded")
    model_plan = plan["models"][model]
    checkpoint_path = repo_path(repo, Path(model_plan["checkpoint"]["expected"]["path"]))
    onnx_path = repo_path(repo, Path(model_plan["onnx"]["path"]))
    stage(state, "binary_binding")
    checkpoint_before = file_evidence_fn(repo, checkpoint_path)
    onnx_before = file_evidence_fn(repo, onnx_path)
    if checkpoint_before.get("sha256") != numeric.EXPECTED_CHECKPOINTS[model]["sha256"] or onnx_before.get("sha256") != numeric.EXPECTED_ONNX_SHA256[model]:
        raise ValueError(f"Input binary binding differs for {model}")
    row = plan["fixture"][model]
    fixture_binding = numeric.verify_bound_image(repo, row, file_evidence_fn)
    source_path = numeric.resolve_bound_source_image(repo, row)
    tensor, preprocess_trace = numeric.trace_preprocess(source_path, runtime, stride=32)
    stage(state, "preprocessed", image=row["image"])
    producer_source_evidence = source_evidence_fn()
    model_loader = runtime["YOLO"](str(checkpoint_path), task="detect")
    network = model_loader.model.to("cpu").float().eval()
    native_head = numeric.validate_native_head(network, model, plan["models"][model]["accepted_contract"])
    stage(state, "head_validated", head=native_head["head"])
    state["forward_counts"]["native_model_forwards"]["attempted"] += 1
    with runtime["torch"].no_grad():
        native_output = network(tensor)
    state["forward_counts"]["native_model_forwards"]["completed"] += 1
    native_primary, native_contract = numeric.extract_native_primary(native_output, model, runtime["np"])
    native_boxes, native_scores, native_pre_meta = _extract_native_preselection(native_output, model, runtime["np"])
    stage(state, "native_forward_complete")
    input_np = runtime["np"].ascontiguousarray(tensor.detach().cpu().numpy())
    state["forward_counts"]["onnx_original_session_runs"]["attempted"] += 1
    original_outputs, original_session = _session_run(runtime, onnx_path, input_np, ["output0"])
    state["forward_counts"]["onnx_original_session_runs"]["completed"] += 1
    stage(state, "original_onnx_complete")
    if model == "yolov8n":
        analysis = analyze_v8_case(native_primary, native_boxes, native_scores, original_outputs["output0"], plan["numeric_v2_target_records"][model], runtime["np"])
        graph_contract = None
        derived_outputs, derived_session = {}, None
    else:
        if onnx_module_loader is None:
            try:
                import onnx as onnx_module_loader
            except ImportError as exc:
                raise LocalizationUnresolved(f"ONNX Python package is required for private diagnostic graph: {exc}") from exc
        derived_path = private_dir / "yolo26n_localization_view.onnx"
        _derived_model, graph_contract = make_v26_derived_graph(onnx_path, derived_path, onnx_module_loader)
        state["forward_counts"]["onnx_derived_session_runs"]["attempted"] += 1
        derived_names = ["output0", *V26_INTERNAL_OUTPUTS.values()]
        deduped_names = list(dict.fromkeys(derived_names))
        derived_outputs, derived_session = _session_run(runtime, derived_path, input_np, deduped_names)
        state["forward_counts"]["onnx_derived_session_runs"]["completed"] += 1
        stage(state, "derived_onnx_complete")
        analysis = analyze_v26_case(native_primary, native_boxes, native_scores, original_outputs, derived_outputs, graph_contract, plan["numeric_v2_target_records"][model], torch_module=runtime["torch"], np_module=runtime["np"])
    checkpoint_after = file_evidence_fn(repo, checkpoint_path)
    onnx_after = file_evidence_fn(repo, onnx_path)
    if checkpoint_before != checkpoint_after or onnx_before != onnx_after:
        raise ValueError(f"Accepted input binary changed during localization for {model}")
    if fixture_binding != numeric.verify_bound_image(repo, row, file_evidence_fn):
        raise ValueError(f"Target image binding changed during localization for {model}")
    state["status"] = "completed"
    stage(state, "completed")
    private = analysis["analysis"].get("graph_trace", {}).get("derived_file", {}) if model == "yolo26n" else {}
    private_artifacts = ([{"path": "models/yolo26n/private/yolo26n_localization_view.onnx", "retention": "server_private_not_publishable", "bytes": private.get("bytes"), "sha256": private.get("sha256")}] if model == "yolo26n" else [])
    return {"schema_version": 1, "study": STUDY, "model": model, "image_id": TARGETS[model], "status": "completed", "numeric_status": "fail", "strict_numeric_v2_verdict": "fail_preserved", "runtime": runtime["packages"], "producer_source_evidence": producer_source_evidence, "onnx_original_session": original_session, "onnx_accepted": {"before": onnx_before, "after": onnx_after, "unchanged": onnx_before == onnx_after}, "checkpoint": {"before": checkpoint_before, "after": checkpoint_after, "unchanged": checkpoint_before == checkpoint_after}, "fixture_binding": fixture_binding, "preprocess_trace": preprocess_trace, "native_head_contract": native_head, "native_output_contract": native_contract, "native_preselection_capture": native_pre_meta, "analysis": analysis["analysis"], "private_artifacts": private_artifacts, "forward_counts": {key: dict(value) for key, value in state["forward_counts"].items()}, "lifecycle": {"status": state["status"], "stage": state["stage"], "stage_history": state["stage_history"], "forward_counts": state["forward_counts"]}, "audit_flags": {"export_performed": False, "tensorRT_imported": False, "accepted_onnx_modified": False, "gpu_used": False, "scored_run_authorized": False, "matrix_opened": False}, "created_utc": datetime.now(timezone.utc).isoformat()}


def publishable_inventory(output_root: Path) -> list[str]:
    allowed = {"localization_plan.json", "localization_manifest.json", "report.md"}
    if output_root.is_dir():
        for path in output_root.rglob("*"):
            if not path.is_file() or "private" in path.parts:
                continue
            relative = path.relative_to(output_root).as_posix()
            if path.suffix in {".json", ".md", ".log"} or path.name.endswith(".log.gz"):
                allowed.add(relative)
    return sorted(allowed)


def child_owned_files(output_root: Path, model: str) -> list[str]:
    model_root = output_root / "models" / model
    if not model_root.is_dir():
        return []
    return sorted(path.relative_to(output_root).as_posix() for path in model_root.rglob("*") if path.is_file() and "private" not in path.parts)


def run_parent(args: argparse.Namespace, repo: Path) -> int:
    output_root = repo_path(repo, args.out_dir)
    if output_root.exists():
        raise FileExistsError(f"Localization output exists; refusing overwrite/resume: {output_root}")
    numeric_root = repo_path(repo, args.numeric_root)
    plan = build_plan(repo, numeric_root)
    plan["repo_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    plan["output_root"] = str(output_root.relative_to(repo.resolve()).as_posix())
    output_root.mkdir(parents=True)
    write_json_no_overwrite(output_root / "localization_plan.json", plan)
    rows = []
    failed = False
    script = Path(__file__).resolve()
    for model in TARGETS:
        model_dir = output_root / "models" / model
        model_dir.mkdir(parents=True)
        command = [sys.executable, str(script), "--child", "--repo-root", str(repo), "--model", model, "--plan", str(output_root / "localization_plan.json")]
        result = subprocess.run(command, cwd=repo, env=os.environ.copy(), capture_output=True, text=True, check=False)
        write_text_no_overwrite(output_root / "logs" / f"{model}.log", result.stdout + result.stderr)
        report_path = model_dir / "localization_report.json"
        failure_path = model_dir / "failure.json"
        if report_path.is_file():
            row = read_json(report_path)
        elif failure_path.is_file():
            row = read_json(failure_path)
        else:
            failed = True
            row = {"schema_version": 1, "study": STUDY, "model": model, "status": "failed", "numeric_status": "fail_preserved_not_observed", "error_type": "ChildProcessError", "error": f"model child exited {result.returncode}", "stage": "child_process_exit_without_record", "stage_history": [], "forward_counts": new_child_state(model)["forward_counts"], "partial_files_scope": f"models/{model}/owned_paths_only", "partial_files": child_owned_files(output_root, model), "no_silent_resume": True}
            write_json_no_overwrite(failure_path, row)
        if result.returncode != 0 or row.get("status") != "completed":
            failed = True
        rows.append(row)
    manifest = {"schema_version": 1, "study": STUDY, "status": "failed" if failed else "completed", "execution_status": "failed" if failed else "completed", "numeric_verdict": "fail_preserved", "models": rows, "plan_path": "localization_plan.json", "artifact_inventory": publishable_inventory(output_root), "artifact_inventory_scope": "parent_publishable_excludes_private", "forward_contract": plan["forward_contract"], "audit_flags": plan["audit_flags"], "run_inventory": {"owner": "parent", "scope": "whole_output_root_publishable", "files": publishable_inventory(output_root), "model_files": {model: child_owned_files(output_root, model) for model in TARGETS}, "log_files": [f"logs/{model}.log" for model in TARGETS]}, "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json_no_overwrite(output_root / "localization_manifest.json", manifest)
    write_text_no_overwrite(output_root / "report.md", _final_report(manifest))
    print(f"DONE: {output_root / 'localization_manifest.json'}")
    return 1 if failed else 0


def _final_report(manifest: dict[str, Any]) -> str:
    lines = ["# Precision-head numeric localization", "", f"- Study: `{manifest['study']}`", f"- Status: `{manifest['status']}`", "- The accepted numeric_v2 verdict remains `FAIL`; this package is supplemental localization evidence, not a replacement gate.", "", "## Cases", ""]
    for row in manifest.get("models", []):
        lines.append(f"- `{row.get('model')}/{row.get('image_id')}`: execution `{row.get('status')}`, strict verdict `{row.get('strict_numeric_v2_verdict', 'fail_preserved')}`, counts `{row.get('forward_counts', {})}`.")
    lines += ["", "## Boundary", "", "- No TensorRT, export, rebuild, optimization sweep, AP/test evaluation, threshold search, NMS or matrix was run.", "- Private full tensors and the YOLO26 derived graph are excluded from the publishable inventory.", "- Any association not resolved by exact anchor/class indices remains `unresolved`; no nearest-neighbor matching is used."]
    return "\n".join(lines) + "\n"


def run_child(args: argparse.Namespace) -> int:
    repo = args.repo_root.resolve()
    plan = read_json(repo_path(repo, Path(args.plan)))
    model = args.model
    output_root = repo_path(repo, Path(plan["output_root"]))
    out_dir = output_root / "models" / model
    state = new_child_state(model)
    try:
        result = child_report(repo, plan, model, out_dir, state=state)
        write_json_no_overwrite(out_dir / "localization_report.json", result)
        print(f"DONE MODEL {model}: {out_dir / 'localization_report.json'}")
        return 0
    except Exception as exc:
        failure = {"schema_version": 1, "study": STUDY, "model": model, "image_id": TARGETS.get(model), "status": "failed", "numeric_status": "fail_preserved_not_observed", "error_type": type(exc).__name__, "error": str(exc), "stage": state["stage"], "stage_history": state["stage_history"], "forward_counts": state["forward_counts"], "partial_files_scope": f"models/{model}/owned_paths_only", "partial_files": child_owned_files(output_root, model), "no_silent_resume": True, "audit_flags": {"export_performed": False, "tensorRT_imported": False, "accepted_onnx_modified": False, "gpu_used": False, "scored_run_authorized": False, "matrix_opened": False}}
        write_json_no_overwrite(out_dir / "failure.json", failure)
        print(f"FAILED MODEL {model}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded CPU localization of accepted numeric_v2 disagreement")
    parser.add_argument("--numeric-root", type=Path, default=NUMERIC_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", choices=(*numeric.MODEL_CHOICES, "all"), default="all")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[1] if not args.repo_root else args.repo_root.resolve()
    if args.child:
        if args.repo_root is None or args.plan is None or args.model == "all":
            raise ValueError("Internal child requires --repo-root, --plan and one model")
        return run_child(args)
    models = numeric.selected_models(args.model)
    if models != list(TARGETS):
        raise ValueError("The localization package is fixed to both bounded cases; use --model all")
    return run_parent(args, repo)


if __name__ == "__main__":
    raise SystemExit(main())
