"""Pure-Python contract and accounting helpers for the confirmation study.

This module deliberately has no torch, CUDA, TensorRT, ONNX or Ultralytics
imports.  The parent process may import it on a CPU-only workstation; GPU
runtime code belongs to the isolated server child.
"""
from __future__ import annotations

import hashlib
import importlib
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
    """Return the schedule from the accepted readiness producer, verbatim."""
    producer = importlib.import_module("prepare_precision_head_confirmation")
    config_path = Path(__file__).resolve().parents[1] / "configs" / "precision_head_confirmation_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    jobs = producer.generate_schedule(config)
    validate_schedule(jobs)
    return jobs


def validate_schedule(jobs: Iterable[dict[str, Any]]) -> None:
    rows = list(jobs)
    if len(rows) != EXPECTED["total_builder_invocations"]:
        raise ContractError(f"schedule has {len(rows)} jobs, expected 84")
    if [row.get("sequence") for row in rows] != list(range(1, 85)):
        raise ContractError("schedule sequence is not contiguous")
    exact_fields = ("sequence", "round", "model", "selection", "phase", "arm", "repeat", "scored", "capture_required", "calibration_cache_creation", "timing_cache_policy")
    # Compare against a fresh canonical producer output, not only counts.
    canonical = _canonical_schedule_without_validation()
    if len(rows) != len(canonical) or any(any(row.get(field) != wanted.get(field) for field in exact_fields) for row, wanted in zip(rows, canonical)):
        raise ContractError("schedule differs from accepted canonical field order")
    counts = {key: sum(row.get("phase") == key for row in rows) for key in ("auxiliary_calibration", "scored_int8", "scored_fp16")}
    if counts != {"auxiliary_calibration": 6, "scored_int8": 72, "scored_fp16": 6}:
        raise ContractError(f"schedule kind counts differ: {counts}")
    if sum(bool(row.get("capture_required")) for row in rows) != EXPECTED["captures"]:
        raise ContractError("capture count differs from 78")
    for model in MODELS:
        model_rows = [row for row in rows if row.get("model") == model]
        if len(model_rows) != 42:
            raise ContractError(f"{model} must have 42 builder jobs")
        sequences = [row["sequence"] for row in model_rows]
        if sequences != list(range(sequences[0], sequences[0] + 42)):
            raise ContractError(f"{model} block is not contiguous")
        if sum(row["phase"] == "auxiliary_calibration" for row in model_rows) != 3:
            raise ContractError(f"{model} auxiliary count differs")
        for round_id, shift in zip(ROUNDS, SHIFTS):
            cells = [row for row in model_rows if row.get("round") == round_id]
            if len(cells) != 13:
                raise ContractError(f"{model} round {round_id} rotation differs")
            if sum(row["phase"] == "scored_int8" for row in cells) != 12:
                raise ContractError(f"{model} round {round_id} int8 count differs")
            if sum(row["phase"] == "scored_fp16" for row in cells) != 1:
                raise ContractError(f"{model} round {round_id} fp16 count differs")
            if any(row["capture_required"] is not True for row in cells):
                raise ContractError("every scored cell must have one capture")
    if any(row["phase"] == "auxiliary_calibration" and row["capture_required"] for row in rows):
        raise ContractError("auxiliary cache builders must not capture")


def _canonical_schedule_without_validation() -> list[dict[str, Any]]:
    """Load the accepted producer without validating any derived counts."""
    producer = importlib.import_module("prepare_precision_head_confirmation")
    config_path = Path(__file__).resolve().parents[1] / "configs" / "precision_head_confirmation_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return producer.generate_schedule(config)


def mapping_targets(graph_audit: dict[str, Any], model: str) -> dict[str, list[str]]:
    """Extract only the verified active convolution targets from graph audit."""
    mapping = graph_audit.get("mapping") or graph_audit
    if graph_audit.get("mapping_status", mapping.get("mapping_status")) != "verified":
        raise ContractError(f"graph mapping is not verified for {model}")
    audit = mapping.get("active_branch_audit") or {}
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
        matched = row.get("matched_convolutions") or row.get("matched_records") or []
        matched_by_name = {item.get("export_node", item.get("name")): item for item in matched if isinstance(item, dict)}
        if any("/act/" in name or not name.endswith("/Conv") for name in names):
            raise ContractError(f"{model} {branch} includes non-convolution target")
        if matched_by_name and any(matched_by_name.get(name, {}).get("export_op_type", "Conv") != "Conv" for name in names):
            raise ContractError(f"{model} {branch} target metadata is not Conv")
        result[label] = names
    if set(result["bbox"]) & set(result["classification"]):
        raise ContractError(f"{model} active target sets overlap")
    result["both"] = result["bbox"] + result["classification"]
    result["mapping_hash"] = mapping.get("mapping_hash")
    if not isinstance(result["mapping_hash"], str) or not result["mapping_hash"]:
        raise ContractError(f"{model} mapping hash is missing from the accepted mapping object")
    return result


def selected_targets(mapping: dict[str, list[str]], arm: str) -> list[str]:
    if arm == "fp16":
        return []
    if arm not in ARMS:
        raise ContractError(f"unknown arm: {arm}")
    if arm == "baseline_int8":
        return []
    if arm == "bbox_fp32":
        return list(mapping["bbox"])
    if arm == "classification_fp32":
        return list(mapping["classification"])
    return list(mapping["both"])


def validate_arm_for_phase(phase: str, arm: str) -> None:
    if phase == "scored_fp16":
        if arm != "fp16":
            raise ContractError(f"FP16 phase requires fp16 arm, got {arm!r}")
        return
    if phase in {"auxiliary_calibration", "scored_int8"} and arm not in ARMS:
        raise ContractError(f"INT8 phase requires one of {ARMS}, got {arm!r}")


def verify_cache_only_audit(audit: dict[str, Any], *, scored: bool) -> None:
    if scored:
        if audit.get("read_calls", 0) < 1 or audit.get("cache_consumed") is not True or audit.get("write_calls") != 0 or audit.get("batch_calls") != 0:
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
