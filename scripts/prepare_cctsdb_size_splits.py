#!/usr/bin/env python3
"""Materialize the official CCTSDB2021 positive-test size subsets without leakage."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SIZE_BINS = ("xs", "s", "m", "l", "xl")
OFFICIAL_AREA_RULES = {
    "xs": "area <= 210 px^2",
    "s": "210 < area <= 400 px^2",
    "m": "400 < area <= 1000 px^2",
    "l": "1000 < area <= 2000 px^2",
    "xl": "area > 2000 px^2",
}
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_size_source(raw: Path) -> tuple[Path, bool]:
    directories = [path for path in raw.iterdir() if path.is_dir() and "size" in path.name.lower()]
    archives = [path for path in raw.iterdir() if path.is_file() and path.suffix.lower() == ".zip" and "size" in path.name.lower()]
    if len(directories) == 1:
        return directories[0], False
    if len(archives) == 1:
        return archives[0], True
    raise FileNotFoundError(
        "Could not uniquely locate the official 'classification based on size of traffic signs' directory/archive below "
        f"{raw}. Found directories={directories}, archives={archives}"
    )


def category_from_path(path: Path) -> str | None:
    for part in reversed(path.parts[:-1]):
        token = part.lower().strip()
        if token in SIZE_BINS:
            return token
    return None


def link(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create CCTSDB official positive-test size subsets from the official size XML package")
    parser.add_argument("--raw", type=Path, required=True, help="CCTSDB raw folder containing the official size archive/directory")
    parser.add_argument("--processed", type=Path, default=Path("data/processed/cctsdb2021_clean"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/cctsdb2021_clean/domains"))
    args = parser.parse_args()
    test_images = args.processed / "test" / "images"
    test_labels = args.processed / "test" / "labels"
    if not test_images.is_dir() or not test_labels.is_dir():
        raise FileNotFoundError("Missing processed positive test images/labels; run prepare_cctsdb.py first")
    source, is_archive = locate_size_source(args.raw)
    context = tempfile.TemporaryDirectory(prefix="cctsdb_size_") if is_archive else None
    try:
        root = Path(context.name) if context is not None else source
        if is_archive:
            with zipfile.ZipFile(source) as archive:
                archive.extractall(root)
        assignments: dict[str, str] = {}
        for xml in root.rglob("*.xml"):
            category = category_from_path(xml.relative_to(root))
            if category is None:
                continue
            stem = xml.stem
            if stem in assignments and assignments[stem] != category:
                raise ValueError(f"Conflicting official size categories for {stem}: {assignments[stem]} and {category}")
            assignments[stem] = category
        test_stems = {path.stem for path in test_images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES}
        if not assignments:
            raise ValueError("No size XML assignments found. Inspect the official archive layout; no output was created.")
        unknown = set(assignments) - test_stems
        if unknown:
            raise ValueError(f"Official size XML references {len(unknown)} images absent from processed positive test; first={sorted(unknown)[0]}")
        if len(assignments) >= len(test_stems):
            raise ValueError("Official size subsets must omit mixed-size images; assignment unexpectedly covers all positive test images")
        counts = Counter(assignments.values())
        missing_categories = set(SIZE_BINS) - set(counts)
        if missing_categories:
            raise ValueError(f"Official size XML lacks categories: {sorted(missing_categories)}")
        destinations = {size: args.out / f"size_{size}" for size in SIZE_BINS}
        existing = [path for path in destinations.values() if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing size split(s): " + ", ".join(str(path) for path in existing))
        modes: set[str] = set()
        rows: list[dict[str, str]] = []
        for size, destination in destinations.items():
            (destination / "images").mkdir(parents=True)
            (destination / "labels").mkdir(parents=True)
            for stem in sorted(stem for stem, assigned in assignments.items() if assigned == size):
                image_matches = [path for path in test_images.glob(f"{stem}.*") if path.suffix.lower() in IMAGE_SUFFIXES]
                label = test_labels / f"{stem}.txt"
                if len(image_matches) != 1 or not label.is_file():
                    raise FileNotFoundError(f"Missing processed image/label for official size XML stem {stem}")
                modes.add(link(image_matches[0], destination / "images" / image_matches[0].name))
                modes.add(link(label, destination / "labels" / label.name))
                rows.append({"stem": stem, "size": size, "xml_source": "official_size_package"})
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source.resolve()),
            "source_sha256": sha256(source) if is_archive else None,
            "processed_positive_test": str((args.processed / "test").resolve()),
            "construction": "Official size-package XML path category; no inferred threshold assignment. Images containing multiple official sizes are excluded by the official package.",
            "official_area_rules": OFFICIAL_AREA_RULES,
            "size_image_counts": {size: counts[size] for size in SIZE_BINS},
            "assigned_images": len(assignments),
            "unassigned_positive_test_images": len(test_stems - set(assignments)),
            "link_modes": sorted(modes),
        }
        manifest_path = args.out.parent / "manifests" / "official_size_splits.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with (manifest_path.parent / "official_size_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stem", "size", "xml_source"])
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: row["stem"]))
        print(json.dumps(manifest, indent=2))
        return 0
    finally:
        if context is not None:
            context.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
