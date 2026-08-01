#!/usr/bin/env python3
"""Train a reproducible CCTSDB2021 baseline without touching official test data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path


def stage_data(src: Path, dst: Path) -> Path:
    if not (src / "train" / "images").exists() or not (src / "dev" / "images").exists():
        raise FileNotFoundError(f"Invalid CCTSDB root: {src}")
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return dst


def write_data_yaml(path: Path, data_root: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Auto-generated local CCTSDB2021 train config
path: {data_root.resolve().as_posix()}
train: train/images
val: dev/images
test: test/images
names:
  0: prohibitory
  1: mandatory
  2: warning
nc: 3
""",
        encoding="utf-8",
    )
    return path


def resolve_data_root(data_yaml: Path) -> Path:
    """Resolve CCTSDB's root relative to the YAML, not Ultralytics settings."""
    path_value: str | None = None
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*path:\s*(.+?)\s*(?:#.*)?$", line)
        if match:
            path_value = match.group(1).strip().strip("\"'")
            break
    if not path_value:
        raise ValueError(f"Missing path: entry in {data_yaml}")
    root = Path(path_value)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    if not (root / "train" / "images").exists() or not (root / "dev" / "images").exists():
        raise FileNotFoundError(f"Invalid CCTSDB root resolved from {data_yaml}: {root}")
    return root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: object) -> object:
    """Convert Ultralytics' configuration values into JSON-safe primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO on CCTSDB2021")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", type=Path, default=Path("configs/cctsdb2021_train.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="results")
    parser.add_argument("--name", default="yolo11n_cctsdb_clean_s42")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--cache",
        default=False,
        nargs="?",
        const="ram",
        help="Image cache: off by default; use --cache ram only on fast local disk/RAM.",
    )
    parser.add_argument(
        "--stage-to",
        type=Path,
        default=None,
        help="Optional local SSD/tmp target, e.g. /tmp/cctsdb2021_clean.",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: install dependencies first: pip install -r requirements.txt")
        return 1

    if not args.data.exists():
        print(f"ERROR: missing data config: {args.data}")
        return 1
    source_root = resolve_data_root(args.data)
    data_yaml = args.data
    if args.stage_to is not None:
        staged_root = stage_data(source_root, args.stage_to)
        data_yaml = write_data_yaml(Path("local/cctsdb2021_train_local.yaml"), staged_root)
    else:
        data_yaml = write_data_yaml(Path("local/cctsdb2021_train_absolute.yaml"), source_root)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    project_root = Path(args.project).resolve()
    print(f"model : {args.model}")
    print(f"data  : {data_yaml.resolve()}")
    print(f"run   : {project_root / args.name}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(project_root),
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        exist_ok=True,
        pretrained=True,
        resume=args.resume,
        cache=False if not args.cache else args.cache,
        plots=False,
    )

    run_dir = Path(model.trainer.save_dir) if getattr(model, "trainer", None) else project_root / args.name
    best = run_dir / "weights" / "best.pt"
    source_manifest = Path("data/processed/cctsdb2021_clean/manifests/dataset_manifest.json")
    trainer_args = getattr(getattr(model, "trainer", None), "args", None)
    effective_args = vars(trainer_args) if hasattr(trainer_args, "__dict__") else {}
    provenance = {
        "command": sys.argv,
        "requested_training": {
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "patience": args.patience,
            "seed": args.seed,
            "cache": args.cache,
        },
        "effective_ultralytics_args": json_safe(effective_args),
        "data_yaml": str(data_yaml.resolve()),
        "data_yaml_sha256": sha256(data_yaml),
        "dataset_manifest": str(source_manifest.resolve()) if source_manifest.exists() else None,
        "dataset_manifest_sha256": sha256(source_manifest) if source_manifest.exists() else None,
        "seed": args.seed,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("DONE")
    print(f"best: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
