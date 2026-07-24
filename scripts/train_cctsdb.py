#!/usr/bin/env python3
"""Train YOLO on CCTSDB2021 full train split."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def stage_data(src: Path, dst: Path) -> Path:
    if not (src / "train" / "images").exists() or not (src / "test" / "images").exists():
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
val: test/images
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO on CCTSDB2021")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", type=Path, default=Path("configs/cctsdb2021_train.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="yolo11n_cctsdb_full")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
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
        help="Optional local SSD/tmp target, e.g. /tmp/cctsdb2021_full.",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: install dependencies first: pip install -r requirements.txt")
        return 1

    data_yaml = args.data
    if args.stage_to is not None:
        src_root = Path("data/processed/cctsdb2021_full")
        staged_root = stage_data(src_root, args.stage_to)
        data_yaml = write_data_yaml(Path("configs/cctsdb2021_train_local.yaml"), staged_root)

    if not data_yaml.exists():
        print(f"ERROR: missing data config: {data_yaml}")
        return 1

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    print(f"model : {args.model}")
    print(f"data  : {data_yaml.resolve()}")
    print(f"run   : {args.project}/{args.name}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        exist_ok=True,
        pretrained=True,
        resume=args.resume,
        cache=False if not args.cache else args.cache,
        plots=False,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print("DONE")
    print(f"best: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
