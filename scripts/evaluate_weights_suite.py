#!/usr/bin/env python3
"""Evaluate one or more frozen PyTorch checkpoints on every CCTSDB test split."""
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_weights(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("--weights must be LABEL=PATH")
    if not label.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError(f"Invalid checkpoint label: {label}")
    return label, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PyTorch checkpoints on all official CCTSDB splits")
    parser.add_argument("--weights", type=parse_weights, action="append", required=True, help="LABEL=PATH; repeat for each checkpoint")
    parser.add_argument("--out-dir", type=Path, default=Path("results/eval/fp32_architecture"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    checkpoints = dict(args.weights)
    if len(checkpoints) != len(args.weights):
        raise ValueError("Each --weights label must be unique")
    for label, weights in checkpoints.items():
        if not weights.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {label}: {weights}")

    evaluator = Path(__file__).with_name("evaluate_cctsdb.py")
    if not evaluator.is_file():
        raise FileNotFoundError(f"Missing evaluator: {evaluator}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str]] = []
    for label, weights in checkpoints.items():
        for split, data in SPLITS.items():
            output = args.out_dir / f"{label}_{split}.json"
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite an existing evaluation: {output}")
            command = [
                sys.executable,
                str(evaluator),
                "--weights",
                str(weights),
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
            print(f"\n=== {label}: {split} ===", flush=True)
            subprocess.run(command, check=True)
            results.append({"label": label, "split": split, "result": str(output), "result_sha256": sha256(output)})

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "checkpoints": {label: {"path": str(path.resolve()), "sha256": sha256(path)} for label, path in checkpoints.items()},
        "splits": SPLITS,
        "results": results,
    }
    manifest_path = args.out_dir / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nDONE: {len(results)} evaluations")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
