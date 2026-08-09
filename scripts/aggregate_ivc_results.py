#!/usr/bin/env python3
"""Create paper-ready accuracy and repeated-latency tables from one IVC target."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SPLITS = ("full", "daylike", "sunny", "cloud", "night", "rain", "snow", "foggy")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate one completed IVC target without changing source results")
    parser.add_argument("--root", type=Path, required=True, help="results/ivc_study_v1/<target>")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="Write available rows and an omissions CSV instead of failing")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.out_dir}")
    manifest_path = args.root / "execution_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing matrix manifest: {manifest_path}")
    manifest = read(manifest_path)
    accuracy_rows: list[dict[str, object]] = []
    benchmark_rows: list[dict[str, object]] = []
    omissions: list[dict[str, str]] = []

    for model in manifest["models"]:
        label = model["label"]
        mode_payload = model["modes"]
        fp16_by_split: dict[str, dict] = {}
        for split in SPLITS:
            fp16_path = args.root / "eval" / f"{label}_fp16" / f"{label}_fp16_{split}.json"
            if fp16_path.is_file():
                fp16_by_split[split] = read(fp16_path)
        for mode in mode_payload:
            for split in SPLITS:
                path = args.root / "eval" / f"{label}_{mode}" / f"{label}_{mode}_{split}.json"
                if not path.is_file():
                    omissions.append({"kind": "evaluation", "model": label, "mode": mode, "split_or_rep": split, "path": str(path)})
                    continue
                result = read(path)
                metrics = result["metrics"]
                fp16 = fp16_by_split.get(split)
                accuracy_rows.append(
                    {
                        "target": manifest["target"],
                        "model": label,
                        "mode": mode,
                        "split": split,
                        "map50": metrics["map50"],
                        "map50_95": metrics["map50_95"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "delta_map50_vs_fp16": None if fp16 is None else metrics["map50"] - fp16["metrics"]["map50"],
                        "delta_map50_95_vs_fp16": None if fp16 is None else metrics["map50_95"] - fp16["metrics"]["map50_95"],
                        "result_path": str(path),
                    }
                )
            directory = args.root / "benchmark" / f"{label}_{mode}"
            repetitions = sorted(directory.glob("rep*.json")) if directory.is_dir() else []
            if not repetitions:
                omissions.append({"kind": "benchmark", "model": label, "mode": mode, "split_or_rep": "all", "path": str(directory)})
                continue
            values = [read(path) for path in repetitions]
            benchmark_rows.append(
                {
                    "target": manifest["target"],
                    "model": label,
                    "mode": mode,
                    "repetitions": len(values),
                    "mean_latency_ms": average([value["latency_ms"]["mean"] for value in values]),
                    "p50_latency_ms": average([value["latency_ms"]["p50"] for value in values]),
                    "p95_latency_ms": average([value["latency_ms"]["p95"] for value in values]),
                    "mean_throughput_fps": average([value["throughput_fps"] for value in values]),
                    "benchmark_paths": ";".join(str(path) for path in repetitions),
                }
            )

    if omissions and not args.allow_incomplete:
        raise RuntimeError(f"Matrix is incomplete ({len(omissions)} missing cells); rerun with --allow-incomplete only for a progress report")
    args.out_dir.mkdir(parents=True)
    for name, rows in (("accuracy.csv", accuracy_rows), ("benchmark.csv", benchmark_rows), ("omissions.csv", omissions)):
        keys = sorted({key for row in rows for key in row}) or ["empty"]
        with (args.out_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    worst_rows = []
    for model in manifest["models"]:
        for mode in model["modes"]:
            if mode == "fp16":
                continue
            subset = [row for row in accuracy_rows if row["model"] == model["label"] and row["mode"] == mode and row["split"] not in {"full", "daylike"} and row["delta_map50_vs_fp16"] is not None]
            if subset:
                worst = min(subset, key=lambda row: float(row["delta_map50_vs_fp16"]))
                worst_rows.append({"target": manifest["target"], "model": model["label"], "mode": mode, "worst_domain": worst["split"], "worst_domain_delta_map50": worst["delta_map50_vs_fp16"]})
    with (args.out_dir / "worst_domain_delta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "model", "mode", "worst_domain", "worst_domain_delta_map50"])
        writer.writeheader()
        writer.writerows(worst_rows)
    print(json.dumps({"accuracy_rows": len(accuracy_rows), "benchmark_rows": len(benchmark_rows), "omissions": len(omissions), "worst_domain_rows": len(worst_rows), "out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
