#!/usr/bin/env python3
"""Audit and aggregate the published confirmation records on CPU.

The server runner writes one immutable ``cell_metrics.json`` per scored cell.
This analyzer never opens an engine or imports CUDA/TensorRT.  It first checks
the complete schedule and provenance, then computes descriptive variation and
shared-image bootstrap summaries from the published metric records.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from precision_head_confirmation_contract import (
    ContractError,
    build_schedule,
    sample_sd,
    shared_bootstrap_indices,
    validate_schedule,
)


def read(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def cell_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("models/*/round_*/**/cell_metrics.json")):
        row = read(path)
        if not isinstance(row, dict):
            raise ContractError(f"cell metric is not an object: {path}")
        row["_path"] = path.as_posix()
        rows.append(row)
    if len(rows) != 78:
        raise ContractError(f"expected 78 cell metrics, found {len(rows)}")
    ids = [row.get("job_id") for row in rows]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ContractError("cell metric job IDs are missing or duplicated")
    expected = {job["job_id"] for job in build_schedule() if job["capture_required"]}
    if set(ids) != expected:
        raise ContractError("cell metrics do not cover exactly the scheduled scored cells")
    return rows


def metric(row: dict[str, Any], endpoint: str = "full", name: str = "ap50_95") -> float:
    metrics = row.get("metrics", {})
    value = metrics.get(endpoint, {}).get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractError(f"missing finite {endpoint}.{name} in {row.get('job_id')}")
    return float(value)


def bootstrap_contrast(rows: list[dict[str, Any]], left_arm: str, right_arm: str, endpoint: str) -> dict[str, Any]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["selection"]), str(row.get("repeat", row.get("round"))))
        by_key[(key[0], key[1] + ":" + str(row["arm"]))] = row
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for selection in ("U42", "U43", "U44"):
        for repeat in ("1", "2", "3"):
            left = by_key.get((selection, repeat + ":" + left_arm))
            right = by_key.get((selection, repeat + ":" + right_arm))
            if left is not None and right is not None:
                pairs.append((left, right))
    if len(pairs) != 9:
        raise ContractError(f"contrast requires 9 paired selection/build cells, found {len(pairs)}")
    deltas = [metric(left, endpoint) - metric(right, endpoint) for left, right in pairs]
    # The published cell metric may contain per-image deltas.  If present, use
    # one shared resampling index for every arm/build; otherwise retain a
    # descriptive point estimate and state that conditional CI input is absent.
    pair_image_deltas = []
    for left, right in pairs:
        left_values = left.get("image_metric")
        right_values = right.get("image_metric")
        if not isinstance(left_values, list) or not isinstance(right_values, list):
            pair_image_deltas = []
            break
        if len(left_values) != len(right_values):
            raise ContractError("paired image metric lengths differ")
        pair_image_deltas.append([float(a) - float(b) for a, b in zip(left_values, right_values)])
    if not pair_image_deltas:
        return {"point_mean": sum(deltas) / len(deltas), "ci": None, "bootstrap_status": "not_available_without_image_metric"}
    images = len(pair_image_deltas[0])
    if images == 0 or any(len(value) != images for value in pair_image_deltas):
        raise ContractError("image metric lengths differ")
    indices = shared_bootstrap_indices(images, 1000, 20260916)
    draw_values = [
        sum(sum(pair[index] for index in sample) / images for pair in pair_image_deltas) / len(pair_image_deltas)
        for sample in indices
    ]
    draw_values.sort()
    return {
        "point_mean": sum(deltas) / len(deltas),
        "ci": [draw_values[25], draw_values[974]],
        "bootstrap_status": "completed_shared_image_resampling",
    }


def analyze(root: Path, out_dir: Path) -> dict[str, Any]:
    plan = read(root / "confirmation_plan.json")
    schedule = read(root / "schedule.json")
    validate_schedule(schedule["jobs"])
    rows = cell_rows(root)
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["selection"]), str(row["arm"]))
        groups.setdefault(key, []).append(metric(row))
    variation = [
        {"model": model, "selection": selection, "arm": arm, "n": len(values), "mean": sum(values) / len(values), "sample_sd_ddof1": sample_sd(values), "min": min(values), "max": max(values)}
        for (model, selection, arm), values in sorted(groups.items())
    ]
    summary = {
        "schema_version": 1,
        "study": "precision_head_confirmation_analysis_v1",
        "status": "completed_descriptive_analysis",
        "input_plan_sha256": plan.get("contract", {}).get("sha256"),
        "schedule_sha256": schedule.get("sha256"),
        "cell_count": len(rows),
        "variation": variation,
        "primary_contrast": "both_fp32_minus_baseline_int8_full_ap50_95",
        "bootstrap": {"generator": "PCG64", "seed": 20260916, "draws": 1000, "shared_image_resampling": True, "status": "conditional_on_paired_image_metrics"},
        "limitations": ["No result is inferred when a cell is missing.", "This report is descriptive and does not select a best build.", "FP16 has no non-inferiority margin."],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "variation_table.json").write_text(json.dumps(variation, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.root.resolve(), args.out_dir.resolve())
    print(f"DONE: {args.out_dir / 'analysis_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
