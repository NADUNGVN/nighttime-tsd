#!/usr/bin/env python3
"""Export compact, provenance-bound paper tables from accepted study JSON."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "paper_core_v1" / "tables"

SOURCES = {
    "ablation_points": Path("results/measurement_audit_v1/precision_head_paired_analysis_v1/point_estimates.json"),
    "paired_ci": Path("results/measurement_audit_v1/precision_head_paired_analysis_v1/contrast_ci.json"),
    "paired_summary": Path("results/measurement_audit_v1/precision_head_paired_analysis_v1/analysis_summary.json"),
    "build_repeat": Path("results/measurement_audit_v1/server_uniform_build_repeat_v1/repeat_summary.json"),
    "latency": Path("results/measurement_audit_v1/server_yolo11n_precision_head_latency_v1/latency_summary.json"),
    "bridge_v8": Path("results/measurement_audit_v1/precision_head_source_export_dev_bridge_v1/models/yolov8n/model_report.json"),
    "bridge_26": Path("results/measurement_audit_v1/precision_head_source_export_dev_bridge_v1/models/yolo26n/model_report.json"),
    "feasibility": Path("results/measurement_audit_v1/precision_head_trt_feasibility_v1/smoke_manifest.json"),
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_git_blob_oid(relative_path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"HEAD:{relative_path.as_posix()}"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> float:
    return float(value) * 100.0


def main() -> int:
    missing = [str(path) for path in SOURCES.values() if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError("Required committed evidence is missing: " + ", ".join(missing))

    data = {name: read_json(ROOT / path) for name, path in SOURCES.items()}
    points = data["ablation_points"]
    paired = data["paired_summary"]
    ci = data["paired_ci"]
    repeat = data["build_repeat"]
    latency = data["latency"]

    source_hashes = {name: sha256(ROOT / path) for name, path in SOURCES.items()}
    source_paths = {name: path.as_posix() for name, path in SOURCES.items()}

    point_rows: list[dict[str, Any]] = []
    for arm, values in points["models"].items():
        for metric in ("map50", "map50_95"):
            point_rows.append({
                "study": "yolo11n_precision_head_ablation_v1",
                "model": "YOLO11n",
                "condition": arm,
                "evaluator": "Ultralytics validation capture",
                "size": "full",
                "metric": metric,
                "estimate": pct(values["full_ultralytics"][metric]),
                "unit": "percent",
                "images": paired["images"],
                "instances": paired["instances"],
                "source_path": source_paths["ablation_points"],
                "source_sha256_worktree": source_hashes["ablation_points"],
                "interpretation": "Persisted capture point; paired CI is not reported for this evaluator.",
            })
        for size in ("all", "xs", "s"):
            for metric in ("map50", "map50_95"):
                point_rows.append({
                    "study": "yolo11n_precision_head_ablation_v1",
                    "model": "YOLO11n",
                    "condition": arm,
                    "evaluator": "COCO/XML pycocotools custom CCTSDB area",
                    "size": size,
                    "metric": metric,
                    "estimate": pct(values["coco_xml"][size][metric]),
                    "unit": "percent",
                    "images": paired["images"],
                    "instances": values["coco_xml"][size]["instances"],
                    "source_path": source_paths["ablation_points"],
                    "source_sha256_worktree": source_hashes["ablation_points"],
                    "interpretation": "Point estimate; CI applies to paired contrasts, not each absolute point.",
                })

    contrast_rows: list[dict[str, Any]] = []
    for contrast, by_size in ci["contrasts"].items():
        for size, by_metric in by_size.items():
            for metric, result in by_metric.items():
                contrast_rows.append({
                    "study": "precision_head_paired_analysis_v1",
                    "model": "YOLO11n",
                    "contrast": contrast,
                    "size": size,
                    "metric": metric,
                    "point_delta_pp": result["point_delta_pp"],
                    "ci95_low_pp": result["ci95_percentile_pp"][0],
                    "ci95_high_pp": result["ci95_percentile_pp"][1],
                    "valid_resamples": result["valid_resamples"],
                    "undefined_resamples": result["undefined_resamples"],
                    "bootstrap_seed": paired["seed"],
                    "bootstrap_resamples": paired["resamples"],
                    "estimator": paired["estimator_id"],
                    "source_path": source_paths["paired_ci"],
                    "source_sha256_worktree": source_hashes["paired_ci"],
                })

    variability_rows: list[dict[str, Any]] = []
    build_count = len(repeat["repeats"])
    for size, by_metric in repeat["build_variability_coco_xml"].items():
        for metric, result in by_metric.items():
            variability_rows.append({
                "study": repeat["study"],
                "model": "YOLO11n",
                "condition": "Uniform calibration / repeated TensorRT builds",
                "size": size,
                "metric": metric,
                "builds": build_count,
                "mean_percent": pct(result["mean"]),
                "sample_sd_pp": pct(result["sample_std"]),
                "minimum_percent": pct(result["min"]),
                "maximum_percent": pct(result["max"]),
                "range_pp": result["range_pp"],
                "source_path": source_paths["build_repeat"],
                "source_sha256_worktree": source_hashes["build_repeat"],
                "interpretation": "Descriptive build variability; not calibration-selection or device variance.",
            })

    latency_rows: list[dict[str, Any]] = []
    for arm, result in latency["arm_summary"].items():
        pooled = result["pooled_calls"]
        latency_rows.append({
            "study": latency["study"],
            "model": "YOLO11n",
            "condition": arm,
            "builds": result["builds"],
            "sessions": result["sessions"],
            "timed_calls": pooled["n_calls"],
            "mean_ms": pooled["mean_ms"],
            "p50_ms": pooled["median_ms"],
            "p95_ms": pooled["p95_ms"],
            "p99_ms": pooled["p99_ms"],
            "serial_fps": pooled["serial_fps"],
            "source_path": source_paths["latency"],
            "source_sha256_worktree": source_hashes["latency"],
            "interpretation": "Full synchronous predict wall-time boundary; calls are not independent build replicates.",
        })

    bridge_rows: list[dict[str, Any]] = []
    for key in ("bridge_v8", "bridge_26"):
        report = data[key]
        delta = report["metrics"]["signed_delta_onnx_minus_native"]["all"]
        counts = report["forward_counts"]
        bridge_rows.append({
            "study": "precision_head_source_export_dev_bridge_v1",
            "model": report["model"],
            "status": report["status"],
            "validity": report["validity"],
            "images": report["metrics"]["native"]["images"],
            "onnx_minus_native_ap50_pp": pct(delta["map50"]),
            "onnx_minus_native_ap50_95_pp": pct(delta["map50_95"]),
            "native_cpu_calls_completed": counts["native_cpu_forward"]["completed"],
            "onnx_cpu_calls_completed": counts["onnx_cpu_session_run"]["completed"],
            "source_path": source_paths[key],
            "source_sha256_worktree": source_hashes[key],
            "interpretation": "Application-level AP bridge only; earlier strict numeric FAIL remains unchanged; no TensorRT inference.",
        })

    smoke = data["feasibility"]
    smoke_contract = smoke["call_contract"]
    smoke_rows = [{
        "study": smoke["study"],
        "status": smoke["status"],
        "validity": smoke["validity"],
        "models": smoke_contract["models"],
        "builders": smoke_contract["independent_builds"],
        "trt_enqueues": smoke_contract["trt_application_enqueues"],
        "ort_cpu_calls": smoke_contract["onnx_cpu_reference_calls"],
        "native_forwards": smoke_contract["native_forwards"],
        "warmup_calls": smoke_contract["warmup_calls"],
        "source_path": source_paths["feasibility"],
        "source_sha256_worktree": source_hashes["feasibility"],
        "interpretation": "Runtime feasibility only; not an accuracy matrix or benchmark.",
    }]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "precision_head_points.csv", list(point_rows[0]), point_rows)
    write_csv(OUTPUT / "precision_head_contrasts.csv", list(contrast_rows[0]), contrast_rows)
    write_csv(OUTPUT / "build_variability.csv", list(variability_rows[0]), variability_rows)
    write_csv(OUTPUT / "latency_summary.csv", list(latency_rows[0]), latency_rows)
    write_csv(OUTPUT / "source_export_bridge.csv", list(bridge_rows[0]), bridge_rows)
    write_csv(OUTPUT / "trt_feasibility.csv", list(smoke_rows[0]), smoke_rows)

    try:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = "unavailable"

    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/export_paper_core_evidence.py",
        "generator_sha256": sha256(Path(__file__).resolve()),
        "git_revision_at_generation": git_revision,
        "source_files": {
            name: {
                "path": path.as_posix(),
                "sha256_worktree_bytes": source_hashes[name],
                "canonical_git_blob_oid_at_generation": canonical_git_blob_oid(path),
            }
            for name, path in SOURCES.items()
        },
        "outputs": {
            path.name: {"sha256_worktree_bytes": sha256(path)}
            for path in sorted(OUTPUT.glob("*.csv"))
        },
        "hash_semantics": "SHA-256 fields cover raw worktree bytes; source files also record canonical Git blob object IDs at the generation revision.",
        "scope": "Existing committed evidence only; no inference, build, or new statistical estimation.",
        "status": "generated_from_committed_sources",
    }
    manifest_path = OUTPUT / "evidence_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "tables": manifest["outputs"], "manifest": manifest_path.relative_to(ROOT).as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
