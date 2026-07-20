#!/usr/bin/env python3
"""Pre-build labels.cache then train CCTSDB full — avoids Ultralytics scan hang.

Root cause: even with workers=0, Ultralytics cache_labels() uses ThreadPool(NUM_THREADS)
on every image; on slow disks this hangs or races on labels.cache.

This script:
  1) Forces NUM_THREADS=1
  2) Sequentially builds train/val label caches once (with progress)
  3) Starts training (workers=0, cache=False for images)

Usage:
  python scripts/precache_and_train_cctsdb.py
  python scripts/precache_and_train_cctsdb.py --epochs 100 --batch 64
  python scripts/precache_and_train_cctsdb.py --precache-only
  python scripts/precache_and_train_cctsdb.py --data-root /tmp/cctsdb2021_full
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def force_single_thread() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import ultralytics.utils as u

        u.NUM_THREADS = 1
    except Exception as e:
        print("WARN: could not set ultralytics NUM_THREADS:", e)


def resolve_data_yaml(data_yaml: Path, data_root: Path | None) -> Path:
    """Optionally rewrite yaml path to a local data root (e.g. /tmp copy)."""
    if data_root is None:
        return data_yaml
    data_root = data_root.resolve()
    if not (data_root / "train" / "images").exists():
        print(f"ERROR: {data_root}/train/images missing")
        sys.exit(1)
    out = Path("configs/cctsdb2021_full_local.yaml")
    text = f"""# auto-generated — local/fast disk root
path: {data_root.as_posix()}
train: train/images
val: test/images
test: test/images
names:
  0: prohibitory
  1: mandatory
  2: warning
nc: 3
"""
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out} → path={data_root}")
    return out


def wipe_partial_caches(data_root: Path) -> None:
    for p in data_root.rglob("*.cache"):
        try:
            p.unlink()
            print(f"  removed {p}")
        except OSError as e:
            print(f"  skip {p}: {e}")


def precache_split(img_path: str, data: dict, prefix: str) -> None:
    """Build labels.cache for one split using Ultralytics YOLODataset, 1 thread."""
    force_single_thread()
    from ultralytics.data.dataset import YOLODataset

    print(f"\n=== Precache {prefix}: {img_path} ===")
    # YOLODataset constructor calls get_labels() → cache_labels if no cache
    ds = YOLODataset(
        img_path=img_path,
        imgsz=640,
        batch_size=16,
        augment=False,
        hyp={},
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix=prefix,
        task="detect",
        classes=None,
        data=data,
        fraction=1.0,
    )
    n = len(ds.labels) if hasattr(ds, "labels") else "?"
    print(f"  OK {prefix}: labels loaded = {n}")


def precache(data_yaml: Path) -> dict:
    force_single_thread()
    from ultralytics.utils import YAML

    raw = YAML.load(data_yaml)
    # resolve path relative to yaml file
    path = Path(raw["path"])
    if not path.is_absolute():
        path = (data_yaml.parent / path).resolve()
    raw["path"] = str(path)

    train_imgs = str(path / raw["train"])
    val_imgs = str(path / raw.get("val", raw["train"]))

    print(f"data path : {path}")
    print(f"train imgs: {train_imgs}")
    print(f"val imgs  : {val_imgs}")
    n_tr = len(list(Path(train_imgs).glob("*.*")))
    n_va = len(list(Path(val_imgs).glob("*.*")))
    print(f"counts    : train={n_tr} val={n_va}")

    wipe_partial_caches(path)

    # Minimal data dict for YOLODataset
    data = {
        "names": raw["names"] if isinstance(raw["names"], dict) else {i: n for i, n in enumerate(raw["names"])},
        "channels": 3,
        "nc": int(raw.get("nc", 3)),
    }

    precache_split(train_imgs, data, "train: ")
    precache_split(val_imgs, data, "val: ")
    print("\nPrecache DONE — train should skip long rescan.")
    return raw


def train(data_yaml: Path, args: argparse.Namespace) -> None:
    force_single_thread()
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,  # critical
        cache=False,
        plots=False,
        exist_ok=True,
        project=args.project,
        name=args.name,
        pretrained=True,
        patience=args.patience,
        # slightly less disk thrash at start
        save_period=-1,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("configs/cctsdb2021_full.yaml"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="If set, use this as dataset path (copy data here first for speed)",
    )
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="yolo11n_cctsdb_full")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--precache-only", action="store_true")
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="rsync-like copy processed dataset to fast local path then train",
    )
    args = parser.parse_args()

    force_single_thread()

    data_root = args.data_root
    if args.copy_to is not None:
        src = Path("data/processed/cctsdb2021_full").resolve()
        dst = args.copy_to.resolve()
        print(f"Copying {src} → {dst} (may take a while)...")
        dst.mkdir(parents=True, exist_ok=True)
        # shutil.copytree merge
        if not (dst / "train").exists():
            shutil.copytree(src / "train", dst / "train")
        if not (dst / "test").exists():
            shutil.copytree(src / "test", dst / "test")
        data_root = dst
        print("Copy done.")

    data_yaml = resolve_data_yaml(args.data, data_root)

    print("Step A: precache labels (single-thread)...")
    try:
        precache(data_yaml)
    except TypeError as e:
        # API drift across ultralytics versions — fallback train-only with NUM_THREADS=1
        print("WARN: YOLODataset API mismatch:", e)
        print("Fallback: train with NUM_THREADS=1 only")
        if args.precache_only:
            return 1
        train(data_yaml, args)
        return 0
    except Exception as e:
        print("ERROR during precache:", e)
        import traceback

        traceback.print_exc()
        print("\nTry copy to local disk then re-run:")
        print("  python scripts/precache_and_train_cctsdb.py --copy-to $HOME/data_local/cctsdb_full")
        return 1

    if args.precache_only:
        print("Precache-only requested; exit.")
        return 0

    print("\nStep B: train...")
    train(data_yaml, args)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
