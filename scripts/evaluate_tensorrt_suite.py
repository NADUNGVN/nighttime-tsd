#!/usr/bin/env python3
"""Evaluate one or more fixed-batch TensorRT engines on every CCTSDB test split."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SPLITS = {
    "full": "configs/cctsdb2021_test_full.yaml",
    "daylike": "configs/cctsdb2021_test_daylike.yaml",
    "sunny": "configs/cctsdb2021_test_sunny.yaml",
    "cloud": "configs/cctsdb2021_test_cloud.yaml",
    "night": "configs/cctsdb2021_test_night.yaml",
    "rain": "configs/cctsdb2021_test_rain.yaml",
    "snow": "configs/cctsdb2021_test_snow.yaml",
    "foggy": "configs/cctsdb2021_test_foggy.yaml",
}
SIZE_SPLITS = {
    "size_xs": "configs/cctsdb2021_test_size_xs.yaml",
    "size_s": "configs/cctsdb2021_test_size_s.yaml",
    "size_m": "configs/cctsdb2021_test_size_m.yaml",
    "size_l": "configs/cctsdb2021_test_size_l.yaml",
    "size_xl": "configs/cctsdb2021_test_size_xl.yaml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_engine(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("--engine must be LABEL=PATH")
    if not label.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError(f"Invalid engine label: {label}")
    return label, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate TensorRT engines on all official CCTSDB splits")
    parser.add_argument("--engine", type=parse_engine, action="append", required=True, help="LABEL=PATH; repeat for each engine")
    parser.add_argument("--out-dir", type=Path, default=Path("results/eval/tensorrt10"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1, help="Must match static TensorRT engine batch size")
    parser.add_argument("--predictions-full-dir", type=Path, help="Optional directory for raw per-image predictions on the full official split only")
    parser.add_argument("--include-size-splits", action="store_true", help="Also evaluate official CCTSDB XML-defined XS/S/M/L/XL positive-test subsets")
    args = parser.parse_args()

    engines = dict(args.engine)
    if len(engines) != len(args.engine):
        raise ValueError("Each --engine label must be unique")
    if args.batch != 1:
        raise ValueError("This protocol exports static batch-1 engines; use --batch 1")
    for label, engine in engines.items():
        if not engine.is_file():
            raise FileNotFoundError(f"Missing engine for {label}: {engine}")

    evaluator = Path(__file__).with_name("evaluate_cctsdb.py")
    if not evaluator.is_file():
        raise FileNotFoundError(f"Missing evaluator: {evaluator}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = {**SPLITS, **SIZE_SPLITS} if args.include_size_splits else SPLITS
    results: list[dict[str, str]] = []
    for label, engine in engines.items():
        for split, data in splits.items():
            output = args.out_dir / f"{label}_{split}.json"
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite an existing evaluation: {output}")
            command = [
                sys.executable,
                str(evaluator),
                "--engine",
                str(engine),
                "--data",
                data,
                "--out",
                str(output),
                "--device",
                str(args.device),
                "--imgsz",
                str(args.imgsz),
                "--batch",
                str(args.batch),
                "--label",
                label,
            ]
            if args.predictions_full_dir is not None and split == "full":
                prediction_path = args.predictions_full_dir / f"{label}_full_predictions.json"
                if prediction_path.exists():
                    raise FileExistsError(f"Refusing to overwrite predictions: {prediction_path}")
                command.extend(["--predictions-out", str(prediction_path)])
            print(f"\n=== {label}: {split} ===", flush=True)
            subprocess.run(command, check=True)
            results.append({"label": label, "split": split, "result": str(output), "result_sha256": sha256(output)})

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "engines": {label: {"path": str(path.resolve()), "sha256": sha256(path)} for label, path in engines.items()},
        "splits": splits,
        "results": results,
    }
    manifest_path = args.out_dir / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nDONE: {len(results)} evaluations")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
