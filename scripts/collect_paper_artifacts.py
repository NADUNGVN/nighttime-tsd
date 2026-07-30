#!/usr/bin/env python3
"""Collect small, versionable paper artifacts without copying CCTSDB images or engines."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise argparse.ArgumentTypeError(f"Invalid label: {label}")
    return label, Path(raw_path)


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout + completed.stderr).strip()
    return output or None


def copy_record(source: Path, destination: Path, records: list[dict[str, object]], purpose: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    records.append({"path": destination.as_posix(), "sha256": sha256(destination), "bytes": destination.stat().st_size, "purpose": purpose})


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect reproducibility metadata without copying dataset images or TensorRT engines")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/cctsdb2021_clean"))
    parser.add_argument("--run", type=parse_labeled_path, action="append", required=True, help="LABEL=results/run_directory")
    parser.add_argument("--engine", type=parse_labeled_path, action="append", default=[], help="LABEL=results/engines/file.engine")
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_artifacts/v1"))
    args = parser.parse_args()

    runs = dict(args.run)
    engines = dict(args.engine)
    if len(runs) != len(args.run) or len(engines) != len(args.engine):
        raise ValueError("Run and engine labels must be unique")
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact directory: {args.out_dir}")
    dataset_manifest = args.data_root / "manifests" / "dataset_manifest.json"
    if not dataset_manifest.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {dataset_manifest}")

    records: list[dict[str, object]] = []
    metadata_dir = args.out_dir / "metadata"
    copy_record(dataset_manifest, metadata_dir / "dataset_manifest.json", records, "prepared dataset split and label provenance")
    calibration_manifests = sorted((args.data_root / "calibration").glob("*/calibration_manifest.json"))
    if not calibration_manifests:
        raise FileNotFoundError(f"No calibration manifests found below {args.data_root / 'calibration'}")
    for manifest in calibration_manifests:
        copy_record(manifest, metadata_dir / "calibration" / manifest.parent.name / manifest.name, records, "INT8 calibration selection; contains paths and luminance only")

    inventory: dict[str, object] = {"runs": {}, "engines": {}}
    for label, run_dir in runs.items():
        best = run_dir / "weights" / "best.pt"
        provenance = run_dir / "provenance.json"
        if not best.is_file() or not provenance.is_file():
            raise FileNotFoundError(f"Run {label} must contain weights/best.pt and provenance.json: {run_dir}")
        inventory["runs"][label] = {"best_pt": str(best.resolve()), "best_pt_sha256": sha256(best), "best_pt_bytes": best.stat().st_size}
        copy_record(provenance, metadata_dir / "training" / label / "provenance.json", records, "training command and dataset hash")
        curves = run_dir / "results.csv"
        if curves.is_file():
            copy_record(curves, metadata_dir / "training" / label / "results.csv", records, "per-epoch training and development metrics")
        else:
            inventory["runs"][label]["results_csv"] = "not_found"

    for label, engine in engines.items():
        provenance = engine.with_suffix(engine.suffix + ".provenance.json")
        if not engine.is_file() or not provenance.is_file():
            raise FileNotFoundError(f"Engine {label} must have its .engine and .engine.provenance.json: {engine}")
        inventory["engines"][label] = {"engine": str(engine.resolve()), "engine_sha256": sha256(engine), "engine_bytes": engine.stat().st_size}
        copy_record(provenance, metadata_dir / "engines" / f"{label}.engine.provenance.json", records, "hardware-specific TensorRT engine provenance")

    try:
        import torch
        import ultralytics
    except ImportError:
        torch = None
        ultralytics = None
    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": None if torch is None else torch.__version__,
        "cuda": None if torch is None else torch.version.cuda,
        "ultralytics": None if ultralytics is None else ultralytics.__version__,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    records.append({"path": "environment.json", "sha256": sha256(args.out_dir / "environment.json"), "bytes": (args.out_dir / "environment.json").stat().st_size, "purpose": "runtime environment lock"})
    (args.out_dir / "model_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    records.append({"path": "model_inventory.json", "sha256": sha256(args.out_dir / "model_inventory.json"), "bytes": (args.out_dir / "model_inventory.json").stat().st_size, "purpose": "checkpoint and engine sizes/hashes"})
    for record in records:
        record_path = Path(str(record["path"]))
        if record_path != Path("environment.json") and record_path != Path("model_inventory.json"):
            record["path"] = record_path.relative_to(args.out_dir).as_posix()
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv, "contains_dataset_images": False, "contains_engine_binaries": False, "files": records}
    (args.out_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {args.out_dir}")
    print(json.dumps({"metadata_files": len(records), "calibration_manifests": len(calibration_manifests), "runs": sorted(runs), "engines": sorted(engines)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
