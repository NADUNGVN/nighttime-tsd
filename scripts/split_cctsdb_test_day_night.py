#!/usr/bin/env python3
"""Label CCTSDB test set as day/night and build 50/50 subsets for quant analysis.

Background:
  - Official weather package labels the 1500 *test* images only (not the 16k train).
  - Typically ~500 weather-night + ~1000 non-night (not exactly 50/50).
  - This script:
      1) Assigns night from weather folder XMLs (authoritative).
      2) Assigns remaining test images as day_candidates.
      3) Builds balanced eval sets: --balance 0.5 → 750 night + 750 day when possible.
         If weather-night < 750, promotes darkest day_candidates (mean luminance) to night.

Outputs:
  data/processed/cctsdb2021_test_day/{images,labels}
  data/processed/cctsdb2021_test_night/{images,labels}
  data/processed/cctsdb2021_test_day_night_manifest.csv
  configs/cctsdb2021_test_day.yaml
  configs/cctsdb2021_test_night.yaml

Usage:
  python scripts/split_cctsdb_test_day_night.py --list-only
  python scripts/split_cctsdb_test_day_night.py --balance 0.5
  python scripts/split_cctsdb_test_day_night.py --balance 0.5 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageStat

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
TEST_ID_START = 18992
TEST_COUNT = 1500
DAY_FOLDER_NAMES = {"sunny", "foggy", "rain", "cloud", "snow", "day", "clear"}


def find_test_images(full_root: Path, raw_root: Path) -> dict[str, Path]:
    """stem -> path for test images (prefer processed full/test)."""
    out: dict[str, Path] = {}
    candidates = [
        full_root / "test" / "images",
        raw_root / "test_img",
        raw_root / "test" / "images",
    ]
    for folder in candidates:
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                out[p.stem] = p
                if p.stem.isdigit():
                    out[str(int(p.stem))] = p
                    out[f"{int(p.stem):05d}"] = p
        if out:
            break
    return out


def find_test_labels(full_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    lab = full_root / "test" / "labels"
    if not lab.exists():
        return out
    for p in lab.glob("*.txt"):
        out[p.stem] = p
        if p.stem.isdigit():
            out[str(int(p.stem))] = p
            out[f"{int(p.stem):05d}"] = p
    return out


def resolve_stem_to_image(
    stem: str, filename_tag: str | None, images: dict[str, Path]
) -> str | None:
    candidates = []
    if filename_tag:
        candidates.append(Path(filename_tag.strip()).stem)
    candidates.append(stem)
    if stem.isdigit():
        i = int(stem)
        candidates += [str(i), f"{i:05d}"]
        if 0 <= i < TEST_COUNT:
            candidates += [str(TEST_ID_START + i), f"{TEST_ID_START + i:05d}"]
        if TEST_ID_START <= i <= TEST_ID_START + TEST_COUNT - 1:
            candidates += [str(i), f"{i:05d}"]
    for c in candidates:
        if c in images:
            return images[c].stem  # canonical stem of file
    return None


def weather_assign(raw_root: Path, images: dict[str, Path]) -> dict[str, str]:
    """image_stem -> 'night'|'day' from weather XML paths."""
    assign: dict[str, str] = {}
    for xp in raw_root.rglob("*.xml"):
        parts = [x.lower() for x in xp.parts]
        # only weather package paths
        if not any("weather" in p or "environment" in p or "光照" in p or "classif" in p for p in parts):
            # still allow .../night/... under raw
            if "night" not in parts and not any(d in parts for d in DAY_FOLDER_NAMES):
                continue
        label = None
        if "night" in parts:
            label = "night"
        elif any(d in parts for d in DAY_FOLDER_NAMES):
            label = "day"
        else:
            continue
        fn = None
        try:
            root = ET.parse(xp).getroot()
            fn = root.findtext("filename")
        except ET.ParseError:
            pass
        img_stem = resolve_stem_to_image(xp.stem, fn, images)
        if img_stem:
            # night wins if conflict
            if img_stem in assign and assign[img_stem] == "night":
                continue
            assign[img_stem] = label
    return assign


def mean_luminance(path: Path) -> float:
    with Image.open(path) as im:
        gray = im.convert("L")
        return float(ImageStat.Stat(gray).mean[0])


def copy_split(
    stems: list[str],
    images: dict[str, Path],
    labels: dict[str, Path],
    out_root: Path,
) -> int:
    img_out = out_root / "images"
    lab_out = out_root / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)
    n = 0
    for stem in stems:
        # resolve path
        src = None
        for k in (stem, str(int(stem)) if stem.isdigit() else stem, f"{int(stem):05d}" if stem.isdigit() else stem):
            if k in images:
                src = images[k]
                break
        if src is None:
            continue
        real_stem = src.stem
        dst = img_out / f"{real_stem}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)
        # label
        lab = None
        for k in (real_stem, stem):
            if k in labels:
                lab = labels[k]
                break
        lab_dst = lab_out / f"{real_stem}.txt"
        if lab and lab.exists():
            shutil.copy2(lab, lab_dst)
        else:
            lab_dst.write_text("", encoding="utf-8")
        n += 1
    return n


def write_yaml(path: Path, data_root: Path) -> None:
    path.write_text(
        f"""# Auto: CCTSDB test day/night split for quant analysis
path: {data_root.resolve().as_posix()}
train: images
val: images
test: images
names:
  0: prohibitory
  1: mandatory
  2: warning
nc: 3
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/CCTSDB2021"))
    parser.add_argument("--full", type=Path, default=Path("data/processed/cctsdb2021_full"))
    parser.add_argument("--out-day", type=Path, default=Path("data/processed/cctsdb2021_test_day"))
    parser.add_argument("--out-night", type=Path, default=Path("data/processed/cctsdb2021_test_night"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/cctsdb2021_test_day_night_manifest.csv"))
    parser.add_argument(
        "--balance",
        type=float,
        default=0.5,
        help="Target fraction night of *selected* eval set (0.5 → 50%% night / 50%% day)",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=1500,
        help="Ideal total images in balanced eval (default 1500 → 750+750)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--luma-threshold", type=float, default=None, help="Optional: force night if mean L < T")
    args = parser.parse_args()

    images = find_test_images(args.full, args.raw)
    # unique by real file
    unique_files = {id(p): p for p in images.values()}
    test_stems = sorted({p.stem for p in unique_files.values()})
    print(f"Test images found: {len(test_stems)}")
    if len(test_stems) < 1000:
        print("WARN: expected ~1500 test images — check prepare_cctsdb_full / test_img")

    labels = find_test_labels(args.full)
    print(f"Test labels found: {len(labels)}")

    weather = weather_assign(args.raw, images)
    weather_night = {s for s, v in weather.items() if v == "night"}
    weather_day = {s for s, v in weather.items() if v == "day"}
    print(f"Weather XML mapped: night={len(weather_night)} day={len(weather_day)}")

    # All test stems
    all_stems = set(test_stems)
    # Prefer real stems from files
    night_set = set()
    for s in weather_night:
        if s in all_stems:
            night_set.add(s)
        else:
            # try resolve
            r = resolve_stem_to_image(s, None, images)
            if r:
                night_set.add(r)

    day_set = all_stems - night_set
    # weather day only used for logging
    print(f"After weather: night={len(night_set)} day_candidates={len(day_set)}")

    # Luminance for all (for promote + manifest)
    luma: dict[str, float] = {}
    for stem in sorted(all_stems):
        p = images.get(stem) or images.get(str(int(stem)) if stem.isdigit() else stem)
        if p is None:
            continue
        try:
            luma[stem] = mean_luminance(p)
        except Exception:
            luma[stem] = 128.0

    if args.luma_threshold is not None:
        for stem, L in luma.items():
            if L < args.luma_threshold and stem not in night_set:
                night_set.add(stem)
                day_set.discard(stem)
        print(f"After luma<{args.luma_threshold}: night={len(night_set)} day={len(day_set)}")

    # Balance 50/50
    target_total = min(args.target_total, len(all_stems))
    if target_total % 2 == 1:
        target_total -= 1
    n_night_target = int(round(target_total * args.balance))
    n_day_target = target_total - n_night_target
    print(f"Target balance: night={n_night_target} day={n_day_target} (total={target_total})")

    # Rank day candidates by darkness (low L → more night-like)
    day_sorted_dark = sorted(day_set, key=lambda s: luma.get(s, 128.0))

    final_night = set(night_set)
    final_day = set(day_set)

    if len(final_night) < n_night_target:
        need = n_night_target - len(final_night)
        promote = [s for s in day_sorted_dark if s not in final_night][:need]
        for s in promote:
            final_night.add(s)
            final_day.discard(s)
        print(f"Promoted {len(promote)} darkest day-candidates → night (heuristic)")
    elif len(final_night) > n_night_target:
        # keep darkest official+current nights
        keep = sorted(final_night, key=lambda s: luma.get(s, 128.0))[:n_night_target]
        drop = final_night - set(keep)
        final_night = set(keep)
        final_day |= drop
        print(f"Subsampled night to {n_night_target} (kept darkest)")

    if len(final_day) > n_day_target:
        # keep brightest days
        keep = sorted(final_day, key=lambda s: -luma.get(s, 128.0))[:n_day_target]
        final_day = set(keep)
        print(f"Subsampled day to {n_day_target} (kept brightest)")
    elif len(final_day) < n_day_target:
        print(f"WARN: only {len(final_day)} day images < target {n_day_target}")

    print(f"FINAL: night={len(final_night)} day={len(final_day)}")

    # Manifest rows
    rows = []
    for stem in sorted(all_stems):
        if stem in final_night:
            split = "night"
        elif stem in final_day:
            split = "day"
        else:
            split = "unused"
        src = "weather" if stem in weather_night or stem in weather_day else "heuristic_or_residual"
        if stem in weather_night:
            src = "weather_night"
        elif stem in weather_day:
            src = "weather_day"
        elif stem in final_night and stem not in night_set:
            src = "luma_promote_to_night"
        rows.append(
            {
                "stem": stem,
                "split": split,
                "source": src,
                "mean_luminance": f"{luma.get(stem, -1):.2f}",
            }
        )

    if args.list_only:
        print("list-only: no copy. Sample night:", sorted(final_night)[:5])
        print("sample day:", sorted(final_day)[:5])
        return 0

    # Write outputs
    if args.out_night.exists():
        shutil.rmtree(args.out_night)
    if args.out_day.exists():
        shutil.rmtree(args.out_day)

    n_n = copy_split(sorted(final_night), images, labels, args.out_night)
    n_d = copy_split(sorted(final_day), images, labels, args.out_day)
    print(f"Copied night={n_n} → {args.out_night}")
    print(f"Copied day={n_d} → {args.out_day}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "split", "source", "mean_luminance"])
        w.writeheader()
        w.writerows(rows)
    print(f"Manifest: {args.manifest}")

    write_yaml(Path("configs/cctsdb2021_test_night.yaml"), args.out_night)
    write_yaml(Path("configs/cctsdb2021_test_day.yaml"), args.out_day)
    print("Wrote configs/cctsdb2021_test_night.yaml")
    print("Wrote configs/cctsdb2021_test_day.yaml")

    print("\nNext eval:")
    print("  python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_test_day.yaml")
    print("  python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_test_night.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
