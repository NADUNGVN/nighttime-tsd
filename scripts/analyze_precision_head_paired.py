#!/usr/bin/env python3
"""CPU-only paired uncertainty analysis for the accepted YOLO11n head diagnostic.

This runner reads only canonical Git blobs for the frozen captures and reports.
It does not load an engine, call TensorRT, run inference, or modify source
artifacts.  The output is a new, guarded directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_dev_quantization import LABELS, ap_values, ci, localization, resample_ap
from audit_cctsdb_measurement import load_records, load_xml, sha256, size_bin
from run_g0_int8_capture import check_same_targets
from run_uniform_inference_repeat import prediction_payload, prediction_payload_hash
from verify_cctsdb_capture import coco_size

ARM_NAMES = ("baseline_int8", "bbox_fp32", "classification_fp32", "both_fp32")
MODELS = ("fp16", *ARM_NAMES)
CONTRASTS = (
    ("bbox_minus_baseline", "bbox_fp32", "baseline_int8"),
    ("classification_minus_baseline", "classification_fp32", "baseline_int8"),
    ("both_minus_baseline", "both_fp32", "baseline_int8"),
    ("bbox_minus_classification", "bbox_fp32", "classification_fp32"),
    ("both_minus_bbox", "both_fp32", "bbox_fp32"),
    ("both_minus_classification", "both_fp32", "classification_fp32"),
    ("baseline_minus_fp16", "baseline_int8", "fp16"),
    ("bbox_minus_fp16", "bbox_fp32", "fp16"),
    ("classification_minus_fp16", "classification_fp32", "fp16"),
    ("both_minus_fp16", "both_fp32", "fp16"),
)
EXPECTED_RESAMPLES = 1000
EXPECTED_SEED = 20260916
EXPECTED_IMAGES = 1636
EXPECTED_INSTANCES = 2706
EXPECTED_SIZE_METRIC = "COCO_bbox_AP_custom_CCTSDB_area_XML_original_coordinates_v1"


def write_json(path: Path, value) -> None:
    """Write a new LF JSON file; never overwrite an analysis artifact."""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _relative_git_path(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Input is outside repository: {path}") from exc


def git_blob(repo: Path, path: Path) -> bytes:
    """Read the exact bytes stored at HEAD, not Windows checkout bytes."""
    rel = _relative_git_path(repo, path)
    result = subprocess.run(
        ["git", "cat-file", "blob", f"HEAD:{rel}"],
        cwd=repo,
        capture_output=True,
    )
    if result.returncode:
        raise ValueError(f"Missing/untracked canonical input at HEAD: {rel}")
    return result.stdout


def canonical_sha256(repo: Path, path: Path) -> str:
    return hashlib.sha256(git_blob(repo, path)).hexdigest()


def read_json_blob(repo: Path, path: Path):
    try:
        return json.loads(git_blob(repo, path).decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"Canonical input is not UTF-8 JSON: {_relative_git_path(repo, path)}") from exc


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def latest_touch_commit(repo: Path, path: Path) -> str:
    rel = _relative_git_path(repo, path)
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", rel],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise ValueError(f"No Git commit found for input: {rel}")
    return commit


def build_samples(n_images: int, resamples: int, seed: int) -> np.ndarray:
    if n_images <= 0 or resamples < 2:
        raise ValueError("Invalid bootstrap dimensions")
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = rng.integers(0, n_images, size=(resamples, n_images))
    # Sorting gives deterministic canonical image order for COCO tie handling;
    # duplicate occurrences remain present and are not deduplicated.
    samples.sort(axis=1)
    return samples


def paired_contrast_summary(draws: np.ndarray, points: np.ndarray) -> dict:
    draws = np.asarray(draws)
    points = np.asarray(points)
    if draws.ndim != 4 or draws.shape[1:] != (len(MODELS), len(LABELS), 2):
        raise ValueError(f"Unexpected bootstrap shape: {draws.shape}")
    if points.shape != (len(MODELS), len(LABELS), 2):
        raise ValueError(f"Unexpected point shape: {points.shape}")
    output = {}
    for name, left, right in CONTRASTS:
        i, j = MODELS.index(left), MODELS.index(right)
        output[name] = {
            label: {
                metric: ci(
                    100.0 * (draws[:, i, a, t] - draws[:, j, a, t]),
                    100.0 * (points[i, a, t] - points[j, a, t]),
                )
                for t, metric in enumerate(("map50", "map50_95"))
            }
            for a, label in enumerate(LABELS)
        }
    return output


def localization_summary_vs_reference(per_model: dict, reference_model: str) -> dict:
    """Count GT match transitions against a fixed reference, by size."""
    if reference_model not in per_model:
        raise ValueError(f"Missing localization reference: {reference_model}")
    reference = per_model[reference_model]
    output = {}
    for model, rows in per_model.items():
        if set(rows) != set(reference):
            raise ValueError("Localization GT identity mismatch")
        output[model] = {"reference_model": reference_model}
        for label in LABELS:
            ids = [key for key, row in rows.items() if label == "all" or row["size"] == label]
            metrics = {"instances": len(ids)}
            for threshold in (50, 75, 90):
                key = f"matched_iou{threshold}"
                metrics[f"recall_iou{threshold}"] = (
                    sum(rows[item][key] for item in ids) / len(ids) if ids else None
                )
                metrics[f"lost_vs_{reference_model}_iou{threshold}"] = sum(
                    reference[item][key] and not rows[item][key] for item in ids
                )
                metrics[f"gained_vs_{reference_model}_iou{threshold}"] = sum(
                    rows[item][key] and not reference[item][key] for item in ids
                )
            for field in ("matched50_pair_iou", "matched50_center_error_over_sqrt_gt_area"):
                values = [rows[item][field] for item in ids if rows[item][field] is not None]
                metrics[field + "_median"] = float(np.median(values)) if values else None
            output[model][label] = metrics
    return output


def installed_versions() -> dict:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pycocotools": importlib.metadata.version("pycocotools"),
    }
    for package in ("ultralytics", "torch", "tensorrt"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _validate_one_repeat(
    repo: Path,
    base: Path,
    repeat: int,
    reference_payload: dict,
    xml_sha256: str,
    expected_environment: dict | None,
):
    repeat_dir = base / f"repeat_{repeat}"
    capture = repeat_dir / "capture"
    verification = repeat_dir / "verification"
    paths = {
        "build_manifest": repeat_dir / "build_manifest.json",
        "inspector": repeat_dir / "inspector.json",
        "comparison": repeat_dir / "comparison.json",
        "capture_report": capture / "capture_report.json",
        "predictions": capture / "validator_predictions.json",
        "verification_summary": verification / "verification_summary.json",
        "native_matching": verification / "native_matching.json",
        "size_coco_xml": verification / "size_coco_xml.json",
    }
    contents = {key: read_json_blob(repo, path) for key, path in paths.items()}
    report = contents["capture_report"]
    payload = contents["predictions"]
    summary = contents["verification_summary"]
    native = contents["native_matching"]
    size_report = contents["size_coco_xml"]
    if report.get("status") != "pass" or report.get("dataset_split") != "CCTSDB2021/dev":
        raise ValueError(f"Unverified capture: {base.name}/repeat_{repeat}")
    require_equal(report.get("images"), EXPECTED_IMAGES, f"{base.name}/repeat_{repeat} images")
    require_equal(report.get("instances"), EXPECTED_INSTANCES, f"{base.name}/repeat_{repeat} instances")
    require_equal(summary.get("native_matching_status"), "pass", f"{base.name}/repeat_{repeat} native matching")
    require_equal(summary.get("size_diagnostic"), "completed", f"{base.name}/repeat_{repeat} size diagnostic")
    require_equal(native.get("changed_tp_decisions"), 0, f"{base.name}/repeat_{repeat} native decisions")
    require_equal(canonical_sha256(repo, paths["predictions"]), report["predictions_sha256"], f"{base.name}/repeat_{repeat} prediction file")
    require_equal(canonical_sha256(repo, paths["capture_report"]), summary["capture_report_sha256"], f"{base.name}/repeat_{repeat} capture report")
    require_equal(report["predictions_sha256"], summary["capture_prediction_sha256"], f"{base.name}/repeat_{repeat} prediction link")
    require_equal(report["model_sha256"], summary["model_sha256"], f"{base.name}/repeat_{repeat} engine link")
    require_equal(size_report.get("xml_sha256"), xml_sha256, f"{base.name}/repeat_{repeat} XML")
    require_equal(size_report.get("metric_id"), EXPECTED_SIZE_METRIC, f"{base.name}/repeat_{repeat} metric")
    require_equal(size_report.get("images"), EXPECTED_IMAGES, f"{base.name}/repeat_{repeat} size images")
    environment = report.get("environment", {})
    if expected_environment is not None:
        for key in ("python", "numpy", "ultralytics", "torch", "cuda", "tensorrt", "gpu"):
            if key in expected_environment and key in environment:
                require_equal(environment[key], expected_environment[key], f"{base.name}/repeat_{repeat} environment.{key}")
    # This check preserves target/preprocessing pairing against the FP16 reference.
    check_same_targets(reference_payload, payload)
    return {
        "report": report,
        "payload": payload,
        "size_report": size_report,
        "prediction_payload_hash": prediction_payload_hash(payload),
        "input_files": {
            key: {"path": _relative_git_path(repo, path), "sha256": canonical_sha256(repo, path)}
            for key, path in paths.items()
        },
    }


def _model_input_manifest(model: str, base: Path, repeats: list[dict]) -> dict:
    representative = repeats[0]
    payload_hashes = [item["prediction_payload_hash"] for item in repeats]
    return {
        "model": model,
        "arm": None if model == "fp16" else model,
        "capture_root": str(base).replace("\\", "/"),
        "selected_repeat": 1,
        "selection_reason": "repeat_1 representative after exact payload verification; no best-build selection",
        "repeat_payload_hashes": payload_hashes,
        "repeat_payload_exact": len(set(payload_hashes)) == 1,
        "engine_sha256": representative["report"]["model_sha256"],
        "environment": representative["report"]["environment"],
        "runtime_arguments": representative["report"].get("runtime_arguments"),
        "full_ultralytics_point": representative["report"]["metrics"],
        "repeat_inputs": [item["input_files"] for item in repeats],
    }


def _point_table(models: dict, points: np.ndarray, persisted_sizes: dict) -> dict:
    output = {"schema_version": 1, "metric_id": EXPECTED_SIZE_METRIC, "models": {}}
    for index, model in enumerate(MODELS):
        coco = {}
        max_error = 0.0
        for a, label in enumerate(LABELS):
            row = {}
            for t, metric in enumerate(("map50", "map50_95")):
                value = points[index, a, t]
                persisted = persisted_sizes[model]["metrics"][label][metric]
                error = abs(float(value) - float(persisted)) if persisted is not None else None
                if error is not None:
                    max_error = max(max_error, error)
                row[metric] = float(value) if np.isfinite(value) else None
                row[f"persisted_{metric}"] = persisted
                row[f"reproduction_error_{metric}"] = error
            row["instances"] = persisted_sizes[model]["metrics"][label]["instances"]
            coco[label] = row
        output["models"][model] = {
            "full_ultralytics": models[model]["full_ultralytics"],
            "coco_xml": coco,
            "coco_point_reproduction_max_abs_error": max_error,
            "coco_point_reproduction_exact_within_1e-12": max_error <= 1e-12,
        }
    return output


def _short_report(points: dict, contrasts: dict, localization: dict, metadata: dict) -> str:
    focus = (("all", "map50_95"), ("xs", "map50"), ("xs", "map50_95"), ("s", "map50"), ("s", "map50_95"))
    lines = [
        "# Precision-head paired dev analysis v1",
        "",
        "Status: `step_A_completed_review_required` (CPU replay; no model inference).",
        "",
        f"- Artifact input commit: `{metadata['artifact_commit']}`",
        f"- Analysis code commit: `{metadata['code_commit']}`",
        f"- Dataset: `CCTSDB2021/dev`, {EXPECTED_IMAGES} images, {EXPECTED_INSTANCES} GT instances",
        f"- Bootstrap: {metadata['resamples']} paired image draws, PCG64 seed `{metadata['seed']}`; duplicate occurrences retained",
        "- Full Ultralytics values below are persisted capture points; confidence intervals apply only to COCO/XML endpoints.",
        "",
        "## Fixed contrasts (percentage points)",
        "",
        "| Contrast | " + " | ".join(f"{label} {metric}" for label, metric in focus) + " |",
        "|---|" + "---:|" * len(focus),
    ]
    for name, _, _ in CONTRASTS:
        cells = []
        for label, metric in focus:
            row = contrasts[name][label][metric]
            low, high = row["ci95_percentile_pp"]
            if low is None:
                cells.append(f"{row['point_delta_pp']:.4f} [null]")
            else:
                cells.append(f"{row['point_delta_pp']:.4f} [{low:.4f}, {high:.4f}]")
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "Notation: each cell is `point_delta_pp [95% percentile CI]`; valid/undefined draw counts are in `contrast_ci.json`.",
        "",
        "## Interpretation limits",
        "",
        "These are exploratory paired dev contrasts conditional on frozen captures, one fixed dev sample, one calibration policy, and one server environment. They do not separate build/tactic variability, calibration variability, training variability, or deployment-device variability; they do not establish causal independence of bbox and classification errors. The localization table is a GT-centric conditional matching diagnostic against `baseline_int8`, not a causal decomposition. No official-test data, retraining, new inference, TensorRT build, or benchmark was run by this analysis.",
        "",
        "Point estimates and all ten contrasts must be reviewed together; this report does not select an arm or apply a post-hoc success threshold.",
        "",
    ])
    return "\n".join(lines)


def run(args) -> int:
    repo = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir.resolve()
    study = (repo / args.study_dir).resolve()
    fp16_capture = (repo / args.fp16_capture).resolve()
    fp16_verification = (repo / args.fp16_verification).resolve()
    xml_path = args.xml.resolve()
    if out_dir.exists():
        raise ValueError(f"Output directory already exists: {out_dir}")
    if args.resamples != EXPECTED_RESAMPLES or args.seed != EXPECTED_SEED:
        raise ValueError(f"Locked protocol requires resamples={EXPECTED_RESAMPLES}, seed={EXPECTED_SEED}")
    if not xml_path.is_file():
        raise ValueError(f"Missing XML input: {xml_path}")
    if not study.is_dir() or not fp16_capture.is_dir() or not fp16_verification.is_dir():
        raise ValueError("Missing locked study/capture/verification directory")

    runtime = installed_versions()
    require_equal(runtime["pycocotools"], "2.0.10", "installed pycocotools")
    xml_digest = sha256(xml_path)
    artifact_manifest = study / "study_manifest.json"
    artifact_commit = latest_touch_commit(repo, artifact_manifest)
    code_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()

    fp16_report_path = fp16_capture / "capture_report.json"
    fp16_payload_path = fp16_capture / "validator_predictions.json"
    fp16_summary_path = fp16_verification / "verification_summary.json"
    fp16_native_path = fp16_verification / "native_matching.json"
    fp16_size_path = fp16_verification / "size_coco_xml.json"
    fp16_report = read_json_blob(repo, fp16_report_path)
    fp16_payload = read_json_blob(repo, fp16_payload_path)
    fp16_summary = read_json_blob(repo, fp16_summary_path)
    fp16_native = read_json_blob(repo, fp16_native_path)
    fp16_size = read_json_blob(repo, fp16_size_path)
    if fp16_report.get("status") != "pass" or fp16_report.get("dataset_split") != "CCTSDB2021/dev":
        raise ValueError("FP16 reference is not a passed dev capture")
    require_equal(fp16_report.get("images"), EXPECTED_IMAGES, "FP16 images")
    require_equal(fp16_report.get("instances"), EXPECTED_INSTANCES, "FP16 instances")
    require_equal(fp16_summary.get("native_matching_status"), "pass", "FP16 native matching")
    require_equal(fp16_native.get("changed_tp_decisions"), 0, "FP16 native decisions")
    require_equal(canonical_sha256(repo, fp16_payload_path), fp16_report["predictions_sha256"], "FP16 prediction file")
    require_equal(canonical_sha256(repo, fp16_report_path), fp16_summary["capture_report_sha256"], "FP16 capture report")
    require_equal(fp16_report["predictions_sha256"], fp16_summary["capture_prediction_sha256"], "FP16 prediction link")
    require_equal(fp16_size.get("xml_sha256"), xml_digest, "FP16 XML")
    require_equal(fp16_size.get("metric_id"), EXPECTED_SIZE_METRIC, "FP16 metric")
    reference_records = load_records(fp16_payload)
    xml = load_xml(xml_path, reference_records)

    expected_environment = fp16_report.get("environment", {})
    models = {
        "fp16": {
            "full_ultralytics": fp16_report["metrics"],
            "repeats": [{
                "report": fp16_report,
                "payload": fp16_payload,
                "size_report": fp16_size,
                "prediction_payload_hash": prediction_payload_hash(fp16_payload),
                "input_files": {
                    key: {"path": _relative_git_path(repo, path), "sha256": canonical_sha256(repo, path)}
                    for key, path in {
                        "capture_report": fp16_report_path,
                        "predictions": fp16_payload_path,
                        "verification_summary": fp16_summary_path,
                        "native_matching": fp16_native_path,
                        "size_coco_xml": fp16_size_path,
                    }.items()
                },
            }],
        }
    }
    for arm in ARM_NAMES:
        base = study / arm
        repeats = [_validate_one_repeat(repo, base, repeat, fp16_payload, xml_digest, expected_environment) for repeat in (1, 2, 3)]
        if len({item["prediction_payload_hash"] for item in repeats}) != 1:
            raise ValueError(f"Three repeat payloads are not exact for {arm}")
        models[arm] = {"full_ultralytics": repeats[0]["report"]["metrics"], "repeats": repeats}
    for model in MODELS:
        check_same_targets(fp16_payload, models[model]["repeats"][0]["payload"])
    # Check the runtime against the captured dependency contract before CPU replay.
    if expected_environment.get("numpy") is not None:
        require_equal(runtime["numpy"], expected_environment["numpy"], "installed NumPy")
    if expected_environment.get("ultralytics") is not None:
        require_equal(runtime["ultralytics"], expected_environment["ultralytics"], "installed Ultralytics")

    evaluators = []
    points = []
    localization_rows = {}
    persisted_sizes = {}
    for model in MODELS:
        representative = models[model]["repeats"][0]
        records = load_records(representative["payload"])
        current_size, evaluator = coco_size(list(records.values()), xml, return_evaluator=True)
        persisted = representative["size_report"]
        require_equal(current_size["metric_id"], persisted["metric_id"], f"{model} metric convention")
        require_equal(current_size["rules"], persisted["rules"], f"{model} rules")
        require_equal(current_size["evaluator_source_sha256"], persisted["evaluator_source_sha256"], f"{model} evaluator source")
        current_points = ap_values(evaluator)
        expected_points = np.array([[persisted["metrics"][label][metric] for metric in ("map50", "map50_95")] for label in LABELS], dtype=float)
        if not np.allclose(current_points, expected_points, rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f"COCO point replay mismatch: {model}")
        evaluators.append(evaluator)
        points.append(current_points)
        localization_rows[model] = localization(evaluator)
        persisted_sizes[model] = persisted
        print(f"VERIFIED: {model}: COCO point replay exact within 1e-12", flush=True)
    points_array = np.asarray(points)

    out_dir.mkdir(parents=True)
    samples = build_samples(EXPECTED_IMAGES, args.resamples, args.seed)
    image_names = [evaluators[0].cocoGt.imgs[index]["file_name"] for index in evaluators[0].params.imgIds]
    sample_plan = {
        "schema_version": 1,
        "seed": args.seed,
        "generator": "numpy.random.Generator(PCG64)",
        "images": EXPECTED_IMAGES,
        "image_ids": list(range(1, EXPECTED_IMAGES + 1)),
        "image_names": image_names,
        "indices_zero_based": samples.tolist(),
        "same_draws_for_models": True,
        "duplicates_retained": True,
    }
    write_json(out_dir / "bootstrap_samples.json", sample_plan)
    write_json(out_dir / "localization_per_gt.json", localization_rows)
    write_json(out_dir / "localization_summary.json", {
        "reference_model": "baseline_int8",
        "metrics": localization_summary_vs_reference(localization_rows, "baseline_int8"),
        "scope": "GT recall at IoU thresholds, class-correct ALL-area COCO matching at captured conf>=0.001 and max_det=300. Pair-error medians condition on IoU50-matched GT; diagnostic only, not a causal bbox/classification decomposition.",
    })

    draws = np.empty((args.resamples, len(MODELS), len(LABELS), 2), dtype=float)
    start = time.monotonic()
    for index, sample in enumerate(samples):
        for model_index, evaluator in enumerate(evaluators):
            draws[index, model_index] = resample_ap(evaluator, sample)
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == args.resamples:
            elapsed = time.monotonic() - start
            remaining = elapsed / (index + 1) * (args.resamples - index - 1)
            print(f"BOOTSTRAP {index + 1}/{args.resamples}: {elapsed:.1f}s elapsed, estimated remaining {remaining:.1f}s", flush=True)
    serializable_draws = draws.astype(object)
    serializable_draws[~np.isfinite(draws)] = None
    write_json(out_dir / "bootstrap_draws.json", {
        "axes": ["resample", "model", "size", "metric"],
        "models": MODELS,
        "sizes": LABELS,
        "metrics": ["map50", "map50_95"],
        "values": serializable_draws.tolist(),
    })
    point_table = _point_table(models, points_array, persisted_sizes)
    contrast_table = {
        "schema_version": 1,
        "units": "percentage_points",
        "models": MODELS,
        "sizes": LABELS,
        "metrics": ["map50", "map50_95"],
        "contrasts": paired_contrast_summary(draws, points_array),
        "estimator_id": "coco_xml_paired_image_bootstrap_v1",
    }
    write_json(out_dir / "point_estimates.json", point_table)
    write_json(out_dir / "contrast_ci.json", contrast_table)

    input_manifest = {
        "schema_version": 1,
        "analysis": "precision_head_paired_analysis_v1",
        "dataset_split": "CCTSDB2021/dev",
        "artifact_commit": artifact_commit,
        "code_commit": code_commit,
        "study_manifest": {
            "path": _relative_git_path(repo, artifact_manifest),
            "sha256": canonical_sha256(repo, artifact_manifest),
        },
        "xml": {"path": str(xml_path).replace("\\", "/"), "sha256": xml_digest, "scope": "locked dev image IDs only"},
        "runtime_versions": runtime,
        "captured_environment_contract": expected_environment,
        "models": {model: _model_input_manifest(model, fp16_capture if model == "fp16" else study / model, models[model]["repeats"]) for model in MODELS},
        "representative_rule": "Use repeat_1 only after exact repeat payload verification; this is deduplication, not best-build selection.",
        "no_new_inference": True,
    }
    write_json(out_dir / "input_manifest.json", input_manifest)

    output_files = [
        "bootstrap_samples.json", "localization_per_gt.json", "localization_summary.json",
        "bootstrap_draws.json", "point_estimates.json", "contrast_ci.json", "input_manifest.json",
    ]
    metadata = {"artifact_commit": artifact_commit, "code_commit": code_commit, "seed": args.seed, "resamples": args.resamples}
    write_text(out_dir / "report.md", _short_report(point_table, contrast_table["contrasts"], localization_rows, metadata))
    output_files.append("report.md")
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "global_g0": "review_required",
        "execution_mode": "CPU replay from canonical Git capture blobs; no model inference, TensorRT build, or benchmark",
        "dataset_split": "CCTSDB2021/dev",
        "images": EXPECTED_IMAGES,
        "instances": EXPECTED_INSTANCES,
        "artifact_commit": artifact_commit,
        "code_commit": code_commit,
        "script_sha256": canonical_sha256(repo, Path(__file__)),
        "input_manifest_sha256": sha256(out_dir / "input_manifest.json"),
        "sample_plan_sha256": sha256(out_dir / "bootstrap_samples.json"),
        "seed": args.seed,
        "resamples": args.resamples,
        "models": MODELS,
        "contrasts": [name for name, _, _ in CONTRASTS],
        "estimator_id": "coco_xml_paired_image_bootstrap_v1",
        "repeat_deduplication": {model: models[model]["repeats"][0]["prediction_payload_hash"] for model in MODELS},
        "point_reproduction": {model: point_table["models"][model]["coco_point_reproduction_max_abs_error"] for model in MODELS},
        "outputs": {name: sha256(out_dir / name) for name in output_files},
        "limitations": [
            "Conditional on frozen captures, one dev sample, one calibration policy, and one server environment; not build/tactic, calibration, training, or device variability.",
            "Image IID bootstrap may understate dependence from sequences/cameras; CIs are exploratory and unadjusted for multiple contrasts.",
            "Full Ultralytics points are persisted capture values; percentile CIs are only for the COCO/XML endpoints.",
            "Localization gained/lost counts are conditional matching diagnostics, not a causal decomposition of bbox and classification errors.",
            "No official-test annotations, retraining, new inference, TensorRT build, or benchmark was used.",
            "The source server is a shared lab server; this CPU replay does not establish GPU isolation or end-to-end TensorRT validity.",
        ],
    }
    write_json(out_dir / "analysis_summary.json", summary)
    print(f"DONE: {out_dir / 'analysis_summary.json'}", flush=True)
    return 0


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, default=Path("results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1_attempt2"))
    parser.add_argument("--fp16-capture", type=Path, default=Path("results/measurement_audit_v1/server_fp16_capture_v1"))
    parser.add_argument("--fp16-verification", type=Path, default=Path("results/measurement_audit_v1/server_native_size_v1"))
    parser.add_argument("--xml", type=Path, default=(repo / "../nighttime-tsd/data/raw/CCTSDB2021/xml.zip").resolve())
    parser.add_argument("--resamples", type=int, default=EXPECTED_RESAMPLES)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
