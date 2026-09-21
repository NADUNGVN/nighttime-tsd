"""Pure-Python contract and accounting helpers for the confirmation study.

This module deliberately has no torch, CUDA, TensorRT, ONNX or Ultralytics
imports.  The parent process may import it on a CPU-only workstation; GPU
runtime code belongs to the isolated server child.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

MODELS = ("yolov8n", "yolo26n")
SELECTIONS = ("U42", "U43", "U44")
ARMS = ("baseline_int8", "bbox_fp32", "classification_fp32", "both_fp32")
ROUNDS = (1, 2, 3)
SHIFTS = (0, 4, 8)
EXPECTED = {
    "auxiliary_cache_builds": 6,
    "scored_int8_builds": 72,
    "scored_fp16_builds": 6,
    "total_builder_invocations": 84,
    "captures": 78,
}


class ContractError(ValueError):
    """A protocol or provenance violation that must stop the study."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_schedule() -> list[dict[str, Any]]:
    """Return the immutable 84-job schedule in execution order.

    The rotation changes execution order only.  It never changes the identity
    of a cell and never permits metric-based reordering.
    """
    jobs: list[dict[str, Any]] = []
    sequence = 0
    for model in MODELS:
        for selection in SELECTIONS:
            sequence += 1
            jobs.append({
                "sequence": sequence, "job_id": f"aux-{model}-{selection}",
                "kind": "auxiliary_cache", "model": model, "selection": selection,
                "arm": "cache_builder", "round": 0, "repeat": 1,
                "capture_required": False, "calibration_write_allowed": True,
            })
        for round_id, shift in zip(ROUNDS, SHIFTS):
            cells = [(selection, arm) for selection in SELECTIONS for arm in ARMS]
            cells.append((None, "fp16"))
            cells = cells[shift:] + cells[:shift]
            for position, (selection, arm) in enumerate(cells):
                sequence += 1
                kind = "scored_fp16" if arm == "fp16" else "scored_int8"
                jobs.append({
                    "sequence": sequence,
                    "job_id": f"{kind}-{model}-r{round_id:02d}-p{position:02d}",
                    "kind": kind, "model": model, "selection": selection,
                    "arm": arm, "round": round_id, "repeat": round_id,
                    "rotation_shift": shift, "rotation_position": position,
                    "capture_required": True, "calibration_write_allowed": False,
                })
    validate_schedule(jobs)
    return jobs


def validate_schedule(jobs: Iterable[dict[str, Any]]) -> None:
    rows = list(jobs)
    if len(rows) != EXPECTED["total_builder_invocations"]:
        raise ContractError(f"schedule has {len(rows)} jobs, expected 84")
    if [row.get("sequence") for row in rows] != list(range(1, 85)):
        raise ContractError("schedule sequence is not contiguous")
    ids = [row.get("job_id") for row in rows]
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ContractError("schedule contains duplicate or empty job IDs")
    counts = {key: sum(row.get("kind") == key for row in rows) for key in (
        "auxiliary_cache", "scored_int8", "scored_fp16")}
    if counts != {"auxiliary_cache": 6, "scored_int8": 72, "scored_fp16": 6}:
        raise ContractError(f"schedule kind counts differ: {counts}")
    if sum(bool(row.get("capture_required")) for row in rows) != EXPECTED["captures"]:
        raise ContractError("capture count differs from 78")
    for model in MODELS:
        model_rows = [row for row in rows if row.get("model") == model]
        if len(model_rows) != 42:
            raise ContractError(f"{model} must have 42 builder jobs")
        if model_rows != sorted(model_rows, key=lambda row: row["sequence"]):
            raise ContractError(f"{model} block is not contiguous")
        if sum(row["kind"] == "auxiliary_cache" for row in model_rows) != 3:
            raise ContractError(f"{model} auxiliary count differs")
        for round_id, shift in zip(ROUNDS, SHIFTS):
            cells = [row for row in model_rows if row.get("round") == round_id]
            if len(cells) != 13 or {row.get("rotation_shift") for row in cells} != {shift}:
                raise ContractError(f"{model} round {round_id} rotation differs")
            if sum(row["kind"] == "scored_int8" for row in cells) != 12:
                raise ContractError(f"{model} round {round_id} int8 count differs")
            if sum(row["kind"] == "scored_fp16" for row in cells) != 1:
                raise ContractError(f"{model} round {round_id} fp16 count differs")
            if any(row["capture_required"] is not True for row in cells):
                raise ContractError("every scored cell must have one capture")
    if any(row["kind"] == "auxiliary_cache" and row["capture_required"] for row in rows):
        raise ContractError("auxiliary cache builders must not capture")


def mapping_targets(graph_audit: dict[str, Any], model: str) -> dict[str, list[str]]:
    """Extract only the verified active convolution targets from graph audit."""
    if graph_audit.get("mapping_status") != "verified":
        raise ContractError(f"graph mapping is not verified for {model}")
    audit = graph_audit.get("active_branch_audit") or {}
    branch_names = {
        "yolov8n": ("cv2", "cv3"),
        "yolo26n": ("one2one_cv2", "one2one_cv3"),
    }[model]
    result: dict[str, list[str]] = {}
    for label, branch in zip(("bbox", "classification"), branch_names):
        row = audit.get(branch) or {}
        names = list(row.get("target_node_names") or [])
        if not names:
            raise ContractError(f"{model} {branch} has no target nodes")
        if len(names) != len(set(names)):
            raise ContractError(f"{model} {branch} target names are duplicated")
        if any("/act/" in name or not name.endswith("/Conv") for name in names):
            raise ContractError(f"{model} {branch} includes non-convolution target")
        result[label] = names
    if set(result["bbox"]) & set(result["classification"]):
        raise ContractError(f"{model} active target sets overlap")
    result["both"] = result["bbox"] + result["classification"]
    result["mapping_hash"] = graph_audit.get("mapping_hash")
    return result


def selected_targets(mapping: dict[str, list[str]], arm: str) -> list[str]:
    if arm not in ARMS:
        raise ContractError(f"unknown arm: {arm}")
    if arm == "baseline_int8":
        return []
    if arm == "bbox_fp32":
        return list(mapping["bbox"])
    if arm == "classification_fp32":
        return list(mapping["classification"])
    return list(mapping["both"])


def verify_cache_only_audit(audit: dict[str, Any], *, scored: bool) -> None:
    if scored:
        if audit.get("read_calls") != 1 or audit.get("write_calls") != 0 or audit.get("batch_calls") != 0:
            raise ContractError("scored INT8 build did not use read-only cache-only calibration")
    else:
        if audit.get("write_calls") != 1 or audit.get("batch_calls") != 1024:
            raise ContractError("auxiliary calibration build did not consume exactly 1024 batches")


def sample_sd(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    if len(numbers) < 2:
        return float("nan")
    mean = sum(numbers) / len(numbers)
    return math.sqrt(sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1))


def shared_bootstrap_indices(image_count: int, draws: int, seed: int) -> list[list[int]]:
    """Use NumPy only when analysis is explicitly invoked, never in the runner parent."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - server analysis dependency
        raise RuntimeError("analysis requires numpy") from exc
    generator = np.random.Generator(np.random.PCG64(seed))
    return generator.integers(0, image_count, size=(draws, image_count)).tolist()
