#!/usr/bin/env python3
"""CPU audit and pooled COCO/XML AP bootstrap for ST-SERVER-01."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from precision_head_confirmation_contract import ContractError, ARMS, build_schedule, sample_sd, shared_bootstrap_indices, validate_schedule

ENDPOINTS = ("all", "xs", "s")
METRICS = ("map50", "map50_95")


def read(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def percentile_ci(values: list[float], point: float) -> dict[str, Any]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {"point": None if not math.isfinite(point) else point, "ci95": [None, None], "valid_draws": 0, "undefined_draws": len(values)}
    def quantile(q: float) -> float:
        position = (len(finite) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return float(finite[low] if low == high else finite[low] + (finite[high] - finite[low]) * (position - low))
    return {"point": float(point) if math.isfinite(point) else None, "ci95": [quantile(.025), quantile(.975)], "valid_draws": len(finite), "undefined_draws": len(values) - len(finite)}


def _job_key(job: dict[str, Any]) -> str:
    return f"{job['sequence']:03d}_{job['model']}_{job['phase']}_{job['selection'] or 'NA'}_{job['arm']}_r{job['repeat'] or 0}"


def _cell_files(root: Path) -> tuple[list[dict[str, Any]], dict[tuple[int, str, int, str, str], dict[str, Any]]]:
    manifest_path = root / "execution_manifest.json"
    if not manifest_path.is_file():
        raise ContractError("execution manifest is missing")
    manifest = read(manifest_path)
    if manifest.get("status") != "completed_review_required":
        raise ContractError(f"execution is not complete: {manifest.get('status')}")
    if len(manifest.get("jobs", [])) != 84 or any(item.get("state", {}).get("status") != "completed" for item in manifest["jobs"]):
        raise ContractError("execution manifest does not contain 84 completed child states")
    plan_sha256 = hashlib.sha256((root / "confirmation_plan.json").read_bytes()).hexdigest()
    expected = [row for row in build_schedule() if row["capture_required"]]
    expected_keys = {(row["sequence"], row["model"], row["round"], row["selection"], row["arm"]): row for row in expected}
    cells: dict[tuple[int, str, int, str, str], dict[str, Any]] = {}
    for path in sorted(root.glob("jobs/*/cell_metrics.json")):
        row = read(path); job = row.get("job") or {}
        key = (job.get("sequence"), job.get("model"), job.get("round"), job.get("selection"), job.get("arm"))
        if key not in expected_keys or key in cells: raise ContractError(f"unexpected or duplicate scored cell: {key}")
        prediction_path = path.parent / "predictions.json"
        if not prediction_path.is_file(): raise ContractError(f"published predictions missing: {prediction_path}")
        if row.get("job_id") != _job_key(job):
            raise ContractError(f"cell job identity mismatch: {path}")
        if row.get("plan_sha256") != plan_sha256:
            raise ContractError(f"cell plan provenance mismatch: {path}")
        model_spec = plan.get("models", {}).get(job.get("model"), {})
        if row.get("checkpoint_sha256") != model_spec.get("checkpoint", {}).get("sha256") or row.get("onnx_sha256") != model_spec.get("onnx", {}).get("sha256"):
            raise ContractError(f"cell model provenance mismatch: {path}")
        if row.get("record_count") != 1636:
            raise ContractError(f"cell record count mismatch: {path}")
        actual_prediction_sha256 = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        if row.get("prediction_sha256") != actual_prediction_sha256:
            raise ContractError(f"prediction hash mismatch: {prediction_path}")
        row["_path"], row["_prediction_path"] = path, prediction_path
        cells[key] = row
    if set(cells) != set(expected_keys): raise ContractError(f"scored coverage differs: expected={len(expected_keys)} observed={len(cells)}")
    return expected, cells


def _endpoint_values(ap: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for endpoint in ENDPOINTS:
        for metric in METRICS:
            value = ap[0 if endpoint == "all" else (1 if endpoint == "xs" else 2), 0 if metric == "map50" else 1]
            values[endpoint + "." + metric] = float(value) if value is not None and math.isfinite(float(value)) else float("nan")
    return values


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _contrast(points: dict[tuple[str, str, str], dict[str, float]], draws: dict[tuple[str, str, str], list[dict[str, float]]], model: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for left, right in (("bbox_fp32", "baseline_int8"), ("classification_fp32", "baseline_int8"), ("both_fp32", "baseline_int8")):
        name = f"{left}_minus_{right}"; output[name] = {}
        for selection in ("U42", "U43", "U44"):
            output[name][selection] = {}
            for endpoint in ENDPOINTS:
                output[name][selection][endpoint] = {}
                for metric in METRICS:
                    field = endpoint + "." + metric
                    point = 100.0 * (points[(model, selection, left)][field] - points[(model, selection, right)][field])
                    values = [100.0 * (a[field] - b[field]) for a, b in zip(draws[(model, selection, left)], draws[(model, selection, right)])]
                    output[name][selection][endpoint][metric] = percentile_ci(values, point)
        output[name]["selection_mean"] = {}
        for endpoint in ENDPOINTS:
            output[name]["selection_mean"][endpoint] = {}
            for metric in METRICS:
                field = endpoint + "." + metric
                point = statistics.fmean(100.0 * (points[(model, selection, left)][field] - points[(model, selection, right)][field]) for selection in ("U42", "U43", "U44"))
                values = [statistics.fmean(100.0 * (draws[(model, selection, left)][i][field] - draws[(model, selection, right)][i][field]) for selection in ("U42", "U43", "U44")) for i in range(1000)]
                output[name]["selection_mean"][endpoint][metric] = percentile_ci(values, point)
    return output


def _fp16_contrasts(points: dict[tuple[str, str, str], dict[str, float]], draws: dict[tuple[str, str, str], list[dict[str, float]]], model: str, fp16_point: dict[str, float], fp16_draw: list[dict[str, float]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in ("baseline_int8", "bbox_fp32", "classification_fp32", "both_fp32"):
        name = f"{arm}_minus_fp16"; output[name] = {}
        for endpoint in ENDPOINTS:
            output[name][endpoint] = {}
            for metric in METRICS:
                field = endpoint + "." + metric
                point = statistics.fmean(points[(model, selection, arm)][field] for selection in ("U42", "U43", "U44")) - fp16_point[field]
                values = [statistics.fmean(draws[(model, selection, arm)][i][field] for selection in ("U42", "U43", "U44")) - fp16_draw[i][field] for i in range(1000)]
                output[name][endpoint][metric] = percentile_ci([100.0 * value for value in values], 100.0 * point)
    return output


def analyze(root: Path, out_dir: Path, xml_path: Path) -> dict[str, Any]:
    if out_dir.exists(): raise FileExistsError(f"analysis output exists: {out_dir}")
    plan, schedule = read(root / "confirmation_plan.json"), read(root / "schedule.json")
    validate_schedule(schedule["jobs"])
    if schedule.get("sha256") != plan.get("schedule", {}).get("sha256"): raise ContractError("schedule hash is not linked to plan")
    try:
        import numpy as np
        from audit_cctsdb_measurement import load_records, load_xml
        from analyze_dev_quantization import ap_values, resample_ap
        from verify_cctsdb_capture import coco_size
    except ImportError as exc:
        raise RuntimeError("analysis requires numpy, pycocotools, torch and ultralytics") from exc
    expected, cells = _cell_files(root)
    xml_cache: dict[str, Any] = {}
    points_repeat: dict[tuple[str, str, str, int], dict[str, float]] = {}
    evaluators: dict[tuple[str, str, str, int], Any] = {}
    for key, cell in cells.items():
        _, model, repeat, selection, arm = key
        payload = read(cell["_prediction_path"]); records = load_records(payload)
        if len(records) != 1636: raise ContractError(f"capture image count differs for {key}")
        if not xml_cache: xml_cache.update(load_xml(xml_path, records))
        _, evaluator = coco_size(list(records.values()), xml_cache, return_evaluator=True)
        points_repeat[(model, selection, arm, int(repeat))] = _endpoint_values(ap_values(evaluator))
        evaluators[(model, selection, arm, int(repeat))] = evaluator
    points: dict[tuple[str, str, str], dict[str, float]] = {}
    for model in ("yolov8n", "yolo26n"):
        for selection in ("U42", "U43", "U44"):
            for arm in ARMS:
                points[(model, selection, arm)] = {field: statistics.fmean(points_repeat[(model, selection, arm, repeat)][field] for repeat in (1, 2, 3)) for field in next(iter(points_repeat.values()))}
    indices = shared_bootstrap_indices(1636, 1000, 20260916)
    draws_repeat: dict[tuple[str, str, str, int], list[dict[str, float]]] = {}
    for key, evaluator in evaluators.items():
        draws_repeat[key] = [_endpoint_values(resample_ap(evaluator, sample)) for sample in indices]
    draws: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    for model in ("yolov8n", "yolo26n"):
        for selection in ("U42", "U43", "U44"):
            for arm in ARMS:
                draws[(model, selection, arm)] = [{field: statistics.fmean(draws_repeat[(model, selection, arm, repeat)][i][field] for repeat in (1, 2, 3)) for field in next(iter(draws_repeat.values()))[0]} for i in range(1000)]
    fp16_point = {model: {field: statistics.fmean(points_repeat[(model, None, "fp16", repeat)][field] for repeat in (1, 2, 3)) for field in next(iter(points_repeat.values()))} for model in ("yolov8n", "yolo26n")}
    fp16_draw = {model: [{field: statistics.fmean(draws_repeat[(model, None, "fp16", repeat)][i][field] for repeat in (1, 2, 3)) for field in next(iter(draws_repeat.values()))[0]} for i in range(1000)] for model in ("yolov8n", "yolo26n")}
    variation = []
    for (model, selection, arm), point in sorted(points.items()):
        for endpoint in ENDPOINTS:
            for metric in METRICS:
                values = [points_repeat[(model, selection, arm, repeat)][endpoint + "." + metric] for repeat in (1, 2, 3)]
                variation.append({"model": model, "selection": selection, "arm": arm, "endpoint": endpoint, "metric": metric, "build_mean": statistics.fmean(values), "within_selection_build_sd_ddof1": sample_sd(values)})
    for model in ("yolov8n", "yolo26n"):
        for arm in ARMS:
            for endpoint in ENDPOINTS:
                for metric in METRICS:
                    values = [points[(model, selection, arm)][endpoint + "." + metric] for selection in ("U42", "U43", "U44")]
                    variation.append({"model": model, "selection": "between_selection_means", "arm": arm, "endpoint": endpoint, "metric": metric, "between_selection_mean_sd_ddof1": sample_sd(values)})
    summary = {"schema_version": 2, "study": "precision_head_confirmation_analysis_v1", "status": "completed_descriptive_analysis", "estimator_id": "coco_xml_paired_image_bootstrap_v1", "schedule_sha256": schedule["sha256"], "plan_sha256": hashlib.sha256((root / "confirmation_plan.json").read_bytes()).hexdigest(), "images": 1636, "xml_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(), "bootstrap": {"generator": "PCG64", "seed": 20260916, "draws": 1000, "shared_image_resampling": True, "duplicates_preserved": True}, "points": points, "fp16_points": fp16_point, "contrasts": {model: _contrast(points, draws, model) for model in ("yolov8n", "yolo26n")}, "fp16_contrasts": {model: _fp16_contrasts(points, draws, model, fp16_point[model], fp16_draw[model]) for model in ("yolov8n", "yolo26n")}, "variation": variation, "limitations": ["AP is pooled from detection-level COCO/XML evaluator records; no per-image AP averaging.", "FP16 controls are three model-level repeats, not nine calibration samples.", "No best-build selection or post-hoc success threshold."]}
    out_dir.mkdir(parents=True)
    safe_summary = _json_safe(summary)
    (out_dir / "analysis_summary.json").write_text(json.dumps(safe_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "variation_table.json").write_text(json.dumps(_json_safe(variation), indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "contrast_table.json").write_text(json.dumps(_json_safe({"contrasts": summary["contrasts"], "fp16_contrasts": summary["fp16_contrasts"]}), indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "report.md").write_text("# Precision-head confirmation analysis v2\n\nAP is reconstructed with the pooled COCO/XML evaluator and shared-image PCG64 bootstrap; no per-image AP shortcut is used. The machine-readable outputs include all-cell points, primary/control/FP16 contrasts, within-selection build SD, between-selection-mean SD, and undefined-draw counts.\n", encoding="utf-8", newline="\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--xml", type=Path, required=True); parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv); analyze(args.root.resolve(), args.out_dir.resolve(), args.xml.resolve()); print(f"DONE: {args.out_dir / 'analysis_summary.json'}"); return 0


if __name__ == "__main__": raise SystemExit(main())
