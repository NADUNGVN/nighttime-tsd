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


def _cell_files(root: Path, plan: dict[str, Any] | None = None, *, allow_test_double: bool = False) -> tuple[list[dict[str, Any]], dict[tuple[int, str, int, str, str], dict[str, Any]]]:
    manifest_path = root / "execution_manifest.json"
    if not manifest_path.is_file():
        raise ContractError("execution manifest is missing")
    manifest = read(manifest_path)
    plan = plan or read(root / "confirmation_plan.json")
    if manifest.get("status") != "completed_review_required":
        raise ContractError(f"execution is not complete: {manifest.get('status')}")
    expected_all = build_schedule()
    manifest_jobs = manifest.get("jobs", [])
    if len(manifest_jobs) != 84 or any(item.get("exit_code") != 0 or item.get("state", {}).get("status") != "completed" for item in manifest_jobs):
        raise ContractError("execution manifest does not contain 84 completed child states")
    observed_all = [item.get("job") for item in manifest_jobs]
    if observed_all != expected_all or any(item.get("state", {}).get("job") != item.get("job") for item in manifest_jobs):
        raise ContractError("execution manifest job order or child identity differs from canonical schedule")
    state_by_job = {_job_key(item["job"]): item["state"] for item in manifest_jobs}
    if len(state_by_job) != 84:
        raise ContractError("execution manifest contains duplicate child identities")
    for item in manifest_jobs:
        state = item["state"]
        if state.get("owner_release_status") != "released_after_child_return" or not state.get("cleanup_completed"):
            raise ContractError(f"child lifecycle release is not verified: {_job_key(item['job'])}")
        if not state.get("engine_sha256"):
            raise ContractError(f"child engine identity is missing: {_job_key(item['job'])}")
        if item["job"].get("capture_required"):
            if state.get("forward_attempted") != 1636 or state.get("forward_completed") != 1636 or state.get("synchronization_completed") != 1636 or state.get("warmup_forward_count") != 0:
                raise ContractError(f"child capture accounting is invalid: {_job_key(item['job'])}")
            if not allow_test_double and not state.get("telemetry"):
                raise ContractError(f"child telemetry is missing: {_job_key(item['job'])}")
    plan_sha256 = hashlib.sha256((root / "confirmation_plan.json").read_bytes()).hexdigest()
    expected = [row for row in build_schedule() if row["capture_required"]]
    schedule = read(root / "schedule.json")
    validate_schedule(schedule.get("jobs", []))
    if schedule.get("sha256") != hashlib.sha256(json.dumps(schedule["jobs"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
        raise ContractError("schedule content hash is invalid")
    if schedule.get("jobs") != plan.get("schedule", {}).get("jobs") or schedule.get("sha256") != plan.get("schedule", {}).get("sha256"):
        raise ContractError("plan/schedule content binding differs")
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
        expected_checkpoint = model_spec.get("checkpoint", {}).get("sha256")
        expected_onnx = model_spec.get("onnx", {}).get("sha256")
        if not expected_checkpoint or not expected_onnx or row.get("checkpoint_sha256") != expected_checkpoint or row.get("onnx_sha256") != expected_onnx:
            raise ContractError(f"cell model provenance mismatch: {path}")
        if row.get("record_count") != 1636:
            raise ContractError(f"cell record count mismatch: {path}")
        child = state_by_job.get(_job_key(job))
        if child is None or row.get("engine_sha256") != child.get("engine_sha256"):
            raise ContractError(f"cell engine identity is not linked to child state: {path}")
        cache = child.get("calibration_cache", {})
        if job.get("phase") == "auxiliary_calibration":
            if child.get("calibration_batches") != 1024 or child.get("calibration_write_calls") != 1 or not cache.get("after_sha256"):
                raise ContractError(f"auxiliary cache evidence is incomplete: {path}")
        elif job.get("phase") == "scored_int8":
            if child.get("calibration_batches") != 0 or child.get("calibration_write_calls") != 0 or child.get("calibration_read_calls", 0) < 1 or cache.get("immutable") is not True or cache.get("before_sha256") != cache.get("after_sha256") or cache.get("auxiliary_cache_sha256") != cache.get("before_sha256"):
                raise ContractError(f"scored cache-only evidence is incomplete: {path}")
        if not allow_test_double:
            inspector = child.get("engine_inspector", {})
            inspector_rel = Path(inspector.get("path", "")).as_posix()
            if not inspector_rel.startswith("public/inspectors/") or not inspector_rel.endswith(".json") or "/private/" in f"/{inspector_rel}":
                raise ContractError(f"engine inspector is outside the public allowlist: {path}")
            inspector_path = root / inspector_rel
            if not child.get("precision_inspector", {}).get("requested_only") or child.get("precision_inspector", {}).get("effective_precision") != "unknown":
                raise ContractError(f"precision inspector evidence is not explicitly labeled requested-only/effective-unknown: {path}")
            if not inspector.get("sha256") or not inspector_path.is_file() or hashlib.sha256(inspector_path.read_bytes()).hexdigest() != inspector.get("sha256"):
                raise ContractError(f"public engine inspector evidence is missing or hash-mismatched: {path}")
        if row.get("status") == "synthetic_external_runtime_double" and not allow_test_double:
            raise ContractError(f"synthetic boundary evidence cannot be analyzed as production: {path}")
        actual_prediction_sha256 = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        if row.get("prediction_sha256") != actual_prediction_sha256:
            raise ContractError(f"prediction hash mismatch: {prediction_path}")
        row["_path"], row["_prediction_path"] = path, prediction_path
        payload = read(prediction_path)
        if payload.get("schema_version") != 2 or payload.get("capture_mode") != "same_val_process_batch":
            raise ContractError(f"capture schema mismatch: {prediction_path}")
        if payload.get("model_sha256") != row.get("engine_sha256"):
            raise ContractError(f"capture engine identity mismatch: {prediction_path}")
        if len(payload.get("iou_thresholds", [])) != 10:
            raise ContractError(f"capture IoU thresholds missing: {prediction_path}")
        if row.get("native_matching", {}).get("status") != "pass" or row.get("xml_validation", {}).get("instances") != 2706:
            raise ContractError(f"cell native/XML validation is incomplete: {path}")
        if not allow_test_double and row.get("statistics_replay_match") is not True:
            raise ContractError(f"cell statistics replay was not compared with validator metrics: {path}")
        expected_xml_sha = plan.get("dataset", {}).get("xml_sha256")
        if expected_xml_sha and row.get("xml_validation", {}).get("sha256") != expected_xml_sha:
            raise ContractError(f"cell XML provenance mismatch: {path}")
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


def _point_rows(points: dict[tuple[str, str, str], dict[str, float]], points_repeat: dict[tuple[str, str, str, int], dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for (model, selection, arm), point in sorted(points.items()):
        for repeat in (1, 2, 3):
            rows.append({"model": model, "selection": selection, "arm": arm, "repeat": repeat, "metrics": points_repeat[(model, selection, arm, repeat)]})
        rows.append({"model": model, "selection": selection, "arm": arm, "repeat": "mean", "metrics": point})
    return rows


def _fp16_rows(fp16_point: dict[str, dict[str, float]], fp16_repeat: dict[tuple[str, str, str, int], dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for model in sorted(fp16_point):
        for repeat in (1, 2, 3):
            rows.append({"model": model, "arm": "fp16", "repeat": repeat, "metrics": fp16_repeat[(model, None, "fp16", repeat)]})
        rows.append({"model": model, "arm": "fp16", "repeat": "mean", "metrics": fp16_point[model]})
    return rows


def _contrast(points: dict[tuple[str, str, str], dict[str, float]], draws: dict[tuple[str, str, str], list[dict[str, float]]], model: str, draw_count: int = 1000) -> dict[str, Any]:
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
                values = [statistics.fmean(100.0 * (draws[(model, selection, left)][i][field] - draws[(model, selection, right)][i][field]) for selection in ("U42", "U43", "U44")) for i in range(draw_count)]
                output[name]["selection_mean"][endpoint][metric] = percentile_ci(values, point)
    return output


def _fp16_contrasts(points: dict[tuple[str, str, str], dict[str, float]], draws: dict[tuple[str, str, str], list[dict[str, float]]], model: str, fp16_point: dict[str, float], fp16_draw: list[dict[str, float]], draw_count: int = 1000) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in ("baseline_int8", "bbox_fp32", "classification_fp32", "both_fp32"):
        name = f"{arm}_minus_fp16"; output[name] = {}
        for endpoint in ENDPOINTS:
            output[name][endpoint] = {}
            for metric in METRICS:
                field = endpoint + "." + metric
                point = statistics.fmean(points[(model, selection, arm)][field] for selection in ("U42", "U43", "U44")) - fp16_point[field]
                values = [statistics.fmean(draws[(model, selection, arm)][i][field] for selection in ("U42", "U43", "U44")) - fp16_draw[i][field] for i in range(draw_count)]
                output[name][endpoint][metric] = percentile_ci([100.0 * value for value in values], 100.0 * point)
    return output


def analyze(root: Path, out_dir: Path, xml_path: Path, *, allow_test_double: bool = False, bootstrap_draws: int = 1000) -> dict[str, Any]:
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
    if not allow_test_double and plan.get("dataset", {}).get("xml_sha256") and hashlib.sha256(xml_path.read_bytes()).hexdigest() != plan["dataset"]["xml_sha256"]:
        raise ContractError("analysis XML bytes differ from the plan-bound archive")
    expected, cells = _cell_files(root, plan, allow_test_double=allow_test_double)
    xml_cache: dict[str, Any] = {}
    points_repeat: dict[tuple[str, str, str, int], dict[str, float]] = {}
    evaluators: dict[tuple[str, str, str, int], Any] = {}
    for key, cell in cells.items():
        _, model, repeat, selection, arm = key
        payload = read(cell["_prediction_path"]); records = load_records(payload)
        if len(records) != 1636: raise ContractError(f"capture image count differs for {key}")
        try:
            from verify_cctsdb_capture import validate_capture
            validate_capture(payload)
        except (ValueError, KeyError, TypeError) as exc:
            raise ContractError(f"same-pass capture validation failed for {key}: {exc}") from exc
        if not xml_cache: xml_cache.update(load_xml(xml_path, records))
        _, evaluator = coco_size(list(records.values()), xml_cache, return_evaluator=True)
        points_repeat[(model, selection, arm, int(repeat))] = _endpoint_values(ap_values(evaluator))
        evaluators[(model, selection, arm, int(repeat))] = evaluator
    points: dict[tuple[str, str, str], dict[str, float]] = {}
    for model in ("yolov8n", "yolo26n"):
        for selection in ("U42", "U43", "U44"):
            for arm in ARMS:
                points[(model, selection, arm)] = {field: statistics.fmean(points_repeat[(model, selection, arm, repeat)][field] for repeat in (1, 2, 3)) for field in next(iter(points_repeat.values()))}
    if bootstrap_draws <= 0 or bootstrap_draws > 1000:
        raise ContractError("bootstrap_draws must be in 1..1000")
    indices = shared_bootstrap_indices(1636, bootstrap_draws, 20260916)
    draws_repeat: dict[tuple[str, str, str, int], list[dict[str, float]]] = {}
    for key, evaluator in evaluators.items():
        draws_repeat[key] = [_endpoint_values(resample_ap(evaluator, sample)) for sample in indices]
    draws: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    for model in ("yolov8n", "yolo26n"):
        for selection in ("U42", "U43", "U44"):
            for arm in ARMS:
                draws[(model, selection, arm)] = [{field: statistics.fmean(draws_repeat[(model, selection, arm, repeat)][i][field] for repeat in (1, 2, 3)) for field in next(iter(draws_repeat.values()))[0]} for i in range(bootstrap_draws)]
    fp16_point = {model: {field: statistics.fmean(points_repeat[(model, None, "fp16", repeat)][field] for repeat in (1, 2, 3)) for field in next(iter(points_repeat.values()))} for model in ("yolov8n", "yolo26n")}
    fp16_draw = {model: [{field: statistics.fmean(draws_repeat[(model, None, "fp16", repeat)][i][field] for repeat in (1, 2, 3)) for field in next(iter(draws_repeat.values()))[0]} for i in range(bootstrap_draws)] for model in ("yolov8n", "yolo26n")}
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
    contrast_tables = {model: _contrast(points, draws, model, bootstrap_draws) for model in ("yolov8n", "yolo26n")}
    fp16_contrast_tables = {model: _fp16_contrasts(points, draws, model, fp16_point[model], fp16_draw[model], bootstrap_draws) for model in ("yolov8n", "yolo26n")}
    sample_plan_bytes = json.dumps(indices, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    summary = {"schema_version": 2, "study": "precision_head_confirmation_analysis_v1", "status": "completed_descriptive_analysis", "estimator_id": "coco_xml_paired_image_bootstrap_v1", "schedule_sha256": schedule["sha256"], "plan_sha256": hashlib.sha256((root / "confirmation_plan.json").read_bytes()).hexdigest(), "images": 1636, "instances": 2706, "xml_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(), "bootstrap": {"generator": "PCG64", "seed": 20260916, "draws": bootstrap_draws, "shared_image_resampling": True, "duplicates_preserved": True, "sample_plan_sha256": hashlib.sha256(sample_plan_bytes).hexdigest()}, "all_cell_points": _point_rows(points, points_repeat), "fp16_points": _fp16_rows(fp16_point, points_repeat), "contrasts": contrast_tables, "fp16_contrasts": fp16_contrast_tables, "variation": variation, "limitations": ["AP is pooled from detection-level COCO/XML evaluator records; no per-image AP averaging.", "FP16 controls are three model-level repeats, not nine calibration samples.", "No best-build selection or post-hoc success threshold."]}
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
