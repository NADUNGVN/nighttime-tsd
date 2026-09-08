#!/usr/bin/env python3
"""Summarize the YOLO11n VCSC decision gate; never starts new experiments."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


DOMAINS = ("sunny", "cloud", "rain", "snow", "foggy", "night")
SIZES = ("xs", "s", "m", "l", "xl")
POLICIES = ("uniform", "low_luminance", "vcsc")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(prefix: str, values: list[float]) -> dict[str, float]:
    return {
        f"{prefix}_mean": mean(values),
        f"{prefix}_std": stdev(values),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def metric(path: Path) -> dict:
    return read(path)["metrics"]


def negative_metric(path: Path, threshold: str = "0.25") -> float | None:
    if not path.is_file():
        return None
    return float(read(path)["metrics"][threshold]["false_positives_per_image"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a compact policy/seed decision table from completed YOLO11n calibration results")
    parser.add_argument("--root", type=Path, default=Path("results/calibration_method_v1/rtx8000/yolo11n"))
    parser.add_argument("--config", type=Path, default=Path("configs/yolo11n_calibration_development_v1.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite summary directory: {args.out_dir}")
    config = read(args.config)
    seeds = [int(value) for value in config["calibration"]["stability_seeds"]]
    reference_dir = args.root / "eval" / "yolo11n_fp16_reference"
    reference = {split: metric(reference_dir / f"yolo11n_fp16_reference_{split}.json") for split in ("full", *DOMAINS) if (reference_dir / f"yolo11n_fp16_reference_{split}.json").is_file()}
    reference_size_path = args.root / "size" / "yolo11n_fp16_reference.json"
    reference_size = read(reference_size_path).get("metrics", {}) if reference_size_path.is_file() else {}
    required_reference = {"full", *DOMAINS}
    omissions: list[dict[str, str]] = []
    if required_reference - set(reference):
        omissions.extend({"policy": "fp16", "seed": "reference", "missing": split} for split in sorted(required_reference - set(reference)))
    rows: list[dict[str, object]] = []
    for policy in POLICIES:
        for seed in seeds:
            directory = args.root / "eval" / f"yolo11n_int8_{policy}_s{seed}"
            metrics = {}
            for split in ("full", *DOMAINS):
                path = directory / f"yolo11n_int8_{policy}_s{seed}_{split}.json"
                if path.is_file():
                    metrics[split] = metric(path)
                else:
                    omissions.append({"policy": policy, "seed": str(seed), "missing": split})
            size_path = args.root / "size" / f"yolo11n_int8_{policy}_s{seed}.json"
            size_metrics = read(size_path).get("metrics", {}) if size_path.is_file() else {}
            if "full" not in metrics or not required_reference.issubset(metrics) or not required_reference.issubset(reference) or not {"xs", "s"}.issubset(size_metrics) or not {"xs", "s"}.issubset(reference_size):
                if not size_path.is_file():
                    omissions.append({"policy": policy, "seed": str(seed), "missing": "size_metrics"})
                continue
            domain_deltas = [metrics[domain]["map50"] - reference[domain]["map50"] for domain in DOMAINS]
            domain_retention = [metrics[domain]["map50"] / reference[domain]["map50"] if reference[domain]["map50"] else float("nan") for domain in DOMAINS]
            negative = negative_metric(args.root / "negative" / f"yolo11n_int8_{policy}_s{seed}.json")
            rows.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "full_map50": metrics["full"]["map50"],
                    "full_map50_95": metrics["full"]["map50_95"],
                    "full_delta_map50": metrics["full"]["map50"] - reference["full"]["map50"],
                    "full_retention_map50": metrics["full"]["map50"] / reference["full"]["map50"],
                    "macro_domain_delta_map50": mean(domain_deltas),
                    "macro_domain_retention_map50": mean(domain_retention),
                    "worst_domain": DOMAINS[min(range(len(DOMAINS)), key=lambda index: domain_deltas[index])],
                    "worst_domain_delta_map50": min(domain_deltas),
                    "xs_map50": size_metrics["xs"]["map50"],
                    "xs_delta_map50": size_metrics["xs"]["map50"] - reference_size["xs"]["map50"],
                    "s_map50": size_metrics["s"]["map50"],
                    "s_delta_map50": size_metrics["s"]["map50"] - reference_size["s"]["map50"],
                    "negative_fp_per_image_at_025": negative,
                }
            )
    if omissions and not args.allow_incomplete:
        raise RuntimeError(f"Missing {len(omissions)} required results; use --allow-incomplete only for progress reporting")
    summaries: list[dict[str, object]] = []
    for policy in POLICIES:
        subset = [row for row in rows if row["policy"] == policy]
        if not subset:
            continue
        summary: dict[str, object] = {"policy": policy, "seeds_completed": len(subset)}
        for prefix, key in (
            ("full_map50", "full_map50"),
            ("full_map50_95", "full_map50_95"),
            ("full_delta_map50", "full_delta_map50"),
            ("macro_domain_delta_map50", "macro_domain_delta_map50"),
            ("macro_domain_retention_map50", "macro_domain_retention_map50"),
            ("worst_domain_delta_map50", "worst_domain_delta_map50"),
            ("xs_delta_map50", "xs_delta_map50"),
            ("s_delta_map50", "s_delta_map50"),
        ):
            summary.update(aggregate(prefix, [float(row[key]) for row in subset]))
        summary["negative_fp_per_image_at_025_mean"] = None if any(row["negative_fp_per_image_at_025"] is None for row in subset) else mean([float(row["negative_fp_per_image_at_025"]) for row in subset])
        summaries.append(summary)
    summary_by_policy = {row["policy"]: row for row in summaries}
    recommendation = "INCOMPLETE"
    explanation = "Five-seed results are not complete for Uniform, Low-Luminance, and VCSC."
    if all(policy in summary_by_policy and summary_by_policy[policy]["seeds_completed"] == len(seeds) for policy in POLICIES):
        vcsc, uniform = summary_by_policy["vcsc"], summary_by_policy["uniform"]
        advantage = float(vcsc["macro_domain_delta_map50_mean"]) - float(uniform["macro_domain_delta_map50_mean"])
        full_ok = float(vcsc["full_map50_mean"]) >= float(uniform["full_map50_mean"])
        stable = float(vcsc["macro_domain_delta_map50_std"]) <= float(uniform["macro_domain_delta_map50_std"])
        if advantage >= 0.005 and full_ok and stable:
            recommendation, explanation = "SCALE", "VCSC meets the preregistered macro-domain advantage (>=0.005), full-test non-degradation, and variability criteria."
        elif advantage <= -0.005 or float(vcsc["macro_domain_delta_map50_std"]) > float(uniform["macro_domain_delta_map50_std"]) + 0.005:
            recommendation, explanation = "STOP", "VCSC is materially worse than Uniform or materially less stable under the preregistered rule."
        else:
            recommendation, explanation = "MODIFY_VCSC", "VCSC approximately matches Uniform; inspect descriptor/cluster design before any 15-model scale-up."
    args.out_dir.mkdir(parents=True)
    for filename, fieldnames, records in (
        ("policy_seed_results.csv", sorted({key for row in rows for key in row}) or ["empty"], rows),
        ("policy_summary.csv", sorted({key for row in summaries for key in row}) or ["empty"], summaries),
        ("omissions.csv", ["policy", "seed", "missing"], omissions),
    ):
        with (args.out_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.root.resolve()),
        "config": str(args.config.resolve()),
        "reference": {"full_map50": reference.get("full", {}).get("map50"), "full_map50_95": reference.get("full", {}).get("map50_95")},
        "policy_summaries": summaries,
        "recommendation": recommendation,
        "explanation": explanation,
        "decision_rule": config["decision_rule"],
        "omissions": len(omissions),
    }
    (args.out_dir / "decision_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
