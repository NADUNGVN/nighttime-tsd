#!/usr/bin/env python3
"""Build day / night / mixed PTQ calibration image folders.

- night: subsample of CNTSSS train images
- day: subsample of CCTSDB2021 train (non-night) images
- mixed: half night + half day

Usage (after datasets are ready):
  python scripts/build_calibration_sets.py --n 256
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def copy_sample(paths: list[Path], out_dir: Path, n: int, seed: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    # clear previous soft copies only if empty policy: overwrite names
    rng = random.Random(seed)
    picks = paths if len(paths) <= n else rng.sample(paths, n)
    for p in picks:
        dst = out_dir / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
    return len(picks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256, help="Images per calibration set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cntsss", type=Path, default=Path("data/raw/CNTSSS/train/images"))
    parser.add_argument(
        "--cctsdb-day",
        type=Path,
        default=Path("data/raw/CCTSDB2021"),
        help="Root to search day train images (heuristic: train_img)",
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed/calibration"))
    args = parser.parse_args()

    night_imgs = list_images(args.cntsss)
    if not night_imgs:
        # try alternate layout
        alt = Path("data/raw/CNTSSS/train")
        night_imgs = list_images(alt)
    print(f"Night pool: {len(night_imgs)} from CNTSSS")

    day_root = args.cctsdb_day
    day_candidates = []
    for sub in ("train_img", "DATASET/train", "train/images", "train"):
        day_candidates.extend(list_images(day_root / sub))
    if not day_candidates:
        day_candidates = list_images(day_root)
    # crude filter: exclude paths with night in name
    day_imgs = [p for p in day_candidates if "night" not in str(p).lower()]
    print(f"Day pool: {len(day_imgs)} from CCTSDB")

    if not night_imgs:
        print("ERROR: no night images — download/verify CNTSSS first")
        return 1

    n_night = copy_sample(night_imgs, args.out / "night" / "images", args.n, args.seed)
    print(f"  night calib: {n_night} → {args.out / 'night' / 'images'}")

    if day_imgs:
        n_day = copy_sample(day_imgs, args.out / "day" / "images", args.n, args.seed + 1)
        print(f"  day calib:   {n_day} → {args.out / 'day' / 'images'}")
        half = max(1, args.n // 2)
        mixed_paths = random.Random(args.seed + 2).sample(night_imgs, min(half, len(night_imgs))) + (
            random.Random(args.seed + 3).sample(day_imgs, min(half, len(day_imgs)))
        )
        n_mix = copy_sample(mixed_paths, args.out / "mixed" / "images", args.n, args.seed + 4)
        print(f"  mixed calib: {n_mix} → {args.out / 'mixed' / 'images'}")
    else:
        print("WARN: no day images yet — only night calibration set built")
        # still create mixed as night-only placeholder
        copy_sample(night_imgs, args.out / "mixed" / "images", args.n, args.seed + 4)

    # empty labels dir so ultralytics data yaml val path works (calib only needs images)
    for split in ("night", "day", "mixed"):
        (args.out / split / "labels").mkdir(parents=True, exist_ok=True)

    print("OK. Use configs/calibration_{night,day,mixed}.yaml for export quantize=8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
