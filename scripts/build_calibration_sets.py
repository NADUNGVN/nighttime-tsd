#!/usr/bin/env python3
"""Build day / night / mixed PTQ calibration folders from CCTSDB only.

- night: images listed under weather .../night (test) + optional train subsample
- day:   weather non-night folders (sunny/foggy/...) or CCTSDB train subsample
- mixed: half night + half day

Usage (after CCTSDB prepared):
  python scripts/build_calibration_sets.py --n 256
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NIGHT_NAME = re.compile(r"night", re.I)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def copy_sample(paths: list[Path], out_dir: Path, n: int, seed: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    picks = paths if len(paths) <= n else rng.sample(paths, n)
    for p in picks:
        dst = out_dir / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
    return len(list(out_dir.glob("*")))


def weather_images_by_condition(cctsdb_root: Path) -> tuple[list[Path], list[Path]]:
    """Return (night_imgs, day_imgs) from weather package if present."""
    night, day = [], []
    # find dirs named night / sunny / etc under CCTSDB raw
    for d in cctsdb_root.rglob("*"):
        if not d.is_dir():
            continue
        name = d.name.lower()
        imgs = list_images(d)
        if not imgs:
            # XMLs only: map stems to test_img later
            continue
        if name == "night" or NIGHT_NAME.fullmatch(name):
            night.extend(imgs)
        elif name in {"sunny", "foggy", "rain", "cloud", "snow", "day"}:
            day.extend(imgs)
    return night, day


def stems_from_weather_xml(cctsdb_root: Path) -> tuple[set[str], set[str]]:
    night_stems, day_stems = set(), set()
    for xp in cctsdb_root.rglob("*.xml"):
        parts_lower = [p.lower() for p in xp.parts]
        if "night" in parts_lower:
            night_stems.add(xp.stem)
            # also resolved test id if 5-digit index (handled in extract script)
        elif any(x in parts_lower for x in ("sunny", "foggy", "rain", "cloud", "snow")):
            day_stems.add(xp.stem)
    return night_stems, day_stems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cctsdb-raw", type=Path, default=Path("data/raw/CCTSDB2021"))
    parser.add_argument(
        "--cctsdb-full",
        type=Path,
        default=Path("data/processed/cctsdb2021_full"),
        help="YOLO layout for train/test images",
    )
    parser.add_argument(
        "--cctsdb-night",
        type=Path,
        default=Path("data/processed/cctsdb2021_night"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed/calibration"))
    args = parser.parse_args()

    night_imgs = list_images(args.cctsdb_night / "images")
    if not night_imgs:
        night_imgs = list_images(args.cctsdb_night)

    # day: full test images not in night folder
    test_imgs = list_images(args.cctsdb_full / "test" / "images")
    night_names = {p.name for p in night_imgs}
    day_imgs = [p for p in test_imgs if p.name not in night_names]

    # fallback: train subsample as day-like if test day empty
    if not day_imgs:
        day_imgs = list_images(args.cctsdb_full / "train" / "images")

    print(f"Night pool: {len(night_imgs)} (from cctsdb night processed)")
    print(f"Day pool:   {len(day_imgs)} (test non-night or train fallback)")

    if not night_imgs and not day_imgs:
        print("ERROR: no CCTSDB images — run prepare_cctsdb_full + extract night first")
        return 1

    if night_imgs:
        print("night calib:", copy_sample(night_imgs, args.out / "night" / "images", args.n, args.seed))
    if day_imgs:
        print("day calib:  ", copy_sample(day_imgs, args.out / "day" / "images", args.n, args.seed + 1))

    if night_imgs and day_imgs:
        half = max(1, args.n // 2)
        rng = random.Random(args.seed + 2)
        mix = rng.sample(night_imgs, min(half, len(night_imgs))) + rng.sample(
            day_imgs, min(half, len(day_imgs))
        )
        print("mixed calib:", copy_sample(mix, args.out / "mixed" / "images", args.n, args.seed + 4))
    elif night_imgs:
        copy_sample(night_imgs, args.out / "mixed" / "images", args.n, args.seed + 4)

    for split in ("night", "day", "mixed"):
        (args.out / split / "labels").mkdir(parents=True, exist_ok=True)

    print("OK. Use configs/calibration_{night,day,mixed}.yaml for INT8 export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
