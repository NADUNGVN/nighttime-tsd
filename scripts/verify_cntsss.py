#!/usr/bin/env python3
"""Verify CNTSSS structure, counts, and label quality sample.

Expected (from README_nighttime_2papers.md):
  train: 3,276 images | test: 786 images
  YOLO format labels, 3 classes (0/1/2)
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_TRAIN = 3276
EXPECTED_TEST = 786


def find_cntsss_root(candidates: list[Path]) -> Path | None:
    for root in candidates:
        if not root.exists():
            continue
        # Accept either root/train/images or root/CNTSSS/train/images
        for p in [root, *root.iterdir()] if root.is_dir() else []:
            if (p / "train" / "images").is_dir() and (p / "test" / "images").is_dir():
                return p
            if (p / "train").is_dir() and (p / "test").is_dir():
                # maybe images are directly under train/
                return p
    return None


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    # Prefer images/ subfolder
    img_dir = folder / "images" if (folder / "images").is_dir() else folder
    return sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)


def list_labels(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    lab_dir = folder / "labels" if (folder / "labels").is_dir() else folder
    return sorted(lab_dir.rglob("*.txt"))


def parse_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return rows
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            rows.append((cls, x, y, w, h))
        except ValueError:
            continue
    return rows


def check_split(name: str, split_dir: Path, expected: int) -> dict:
    images = list_images(split_dir)
    labels = list_labels(split_dir)
    img_stems = {p.stem for p in images}
    lab_stems = {p.stem for p in labels}
    missing_lab = sorted(img_stems - lab_stems)
    orphan_lab = sorted(lab_stems - img_stems)

    class_counts: Counter[int] = Counter()
    bad_boxes = 0
    empty_labels = 0
    for lp in labels:
        rows = parse_yolo_label(lp)
        if not rows:
            empty_labels += 1
        for cls, x, y, w, h in rows:
            class_counts[cls] += 1
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                bad_boxes += 1
            if cls not in (0, 1, 2):
                bad_boxes += 1  # unexpected class id

    ok_count = abs(len(images) - expected) <= 5  # small tolerance
    return {
        "name": name,
        "n_images": len(images),
        "n_labels": len(labels),
        "expected": expected,
        "count_ok": ok_count,
        "missing_labels": len(missing_lab),
        "orphan_labels": len(orphan_lab),
        "empty_labels": empty_labels,
        "bad_boxes": bad_boxes,
        "class_counts": dict(sorted(class_counts.items())),
        "sample_missing": missing_lab[:5],
        "images": images,
        "labels": labels,
    }


def sample_visual_report(images: list[Path], labels: list[Path], n: int, seed: int) -> None:
    """Text-only sample: print image path + label lines (no GUI)."""
    rng = random.Random(seed)
    lab_map = {p.stem: p for p in labels}
    pool = [im for im in images if im.stem in lab_map]
    if not pool:
        print("  (no matched image-label pairs for sample)")
        return
    picks = rng.sample(pool, min(n, len(pool)))
    print(f"\n--- Visual spot-check sample (n={len(picks)}, seed={seed}) ---")
    for im in picks:
        lp = lab_map[im.stem]
        rows = parse_yolo_label(lp)
        print(f"  {im.name}: {len(rows)} boxes → {rows[:3]}{'...' if len(rows) > 3 else ''}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CNTSSS dataset integrity")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw"),
        help="Search root (default: data/raw)",
    )
    parser.add_argument("--sample", type=int, default=10, help="Spot-check N labels")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = find_cntsss_root([args.root, args.root / "CNTSSS", Path("data/raw/CNTSSS")])
    if root is None:
        print("ERROR: CNTSSS not found. Expected train/ + test/ under data/raw/CNTSSS/")
        print("  Run: python scripts/download_cntsss.py --extract")
        return 1

    print(f"CNTSSS root: {root.resolve()}")
    train = check_split("train", root / "train", EXPECTED_TRAIN)
    test = check_split("test", root / "test", EXPECTED_TEST)

    all_ok = True
    for s in (train, test):
        print(f"\n=== {s['name']} ===")
        print(f"  images: {s['n_images']} (expected ~{s['expected']}) {'OK' if s['count_ok'] else 'MISMATCH'}")
        print(f"  labels: {s['n_labels']}")
        print(f"  missing labels: {s['missing_labels']} | orphan labels: {s['orphan_labels']}")
        print(f"  empty labels: {s['empty_labels']} | bad boxes/classes: {s['bad_boxes']}")
        print(f"  class instance counts: {s['class_counts']}")
        if s["sample_missing"]:
            print(f"  sample missing label stems: {s['sample_missing']}")
        if not s["count_ok"] or s["missing_labels"] or s["bad_boxes"]:
            all_ok = False

    # Class balance (train)
    total = sum(train["class_counts"].values()) or 1
    print("\n=== Train class balance (instances) ===")
    name_map = {0: "prohibitory", 1: "mandatory", 2: "warning"}
    for c, n in sorted(train["class_counts"].items()):
        print(f"  {c} ({name_map.get(c, '?')}): {n} ({100 * n / total:.1f}%)")
    print("  Paper prior: prohibitory ~64%, mandatory ~22%, warning ~14% (order depends on id mapping)")

    sample_visual_report(train["images"], train["labels"], args.sample, args.seed)

    print("\n" + ("PASS — CNTSSS looks usable" if all_ok else "WARN — review mismatches above"))
    print("Next: download CCTSDB2021 and run extract_cctsdb_night.py")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
