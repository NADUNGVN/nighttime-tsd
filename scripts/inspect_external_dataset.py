#!/usr/bin/env python3
"""Create a lightweight, reproducible inventory for an externally acquired dataset.

This is deliberately format-agnostic.  It records the received release before
we assume an annotation layout or map its labels into CCTSDB super-classes.
It never alters the raw data and does not copy images or labels.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DATASETS = {"tt100k", "mtsd", "cure_tsd"}
ANNOTATION_SUFFIXES = {".json", ".xml", ".txt", ".csv", ".yaml", ".yml"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory an externally acquired TT100K, MTSD, or CURE-TSD release without modifying it")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--raw", type=Path, required=True, help="Root of the received official release")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path; must not be inside --raw")
    parser.add_argument("--sample-limit", type=int, default=40)
    args = parser.parse_args()

    raw = args.raw.resolve()
    out = args.out.resolve()
    if not raw.is_dir():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw}")
    if is_inside(out, raw):
        raise ValueError("--out must be outside --raw so the inventory cannot change its own counts")
    if args.sample_limit < 1:
        raise ValueError("--sample-limit must be positive")

    extensions: Counter[str] = Counter()
    bytes_by_extension: Counter[str] = Counter()
    annotation_candidates: list[str] = []
    archives: list[str] = []
    samples: list[str] = []
    file_count = 0
    directory_count = 0
    total_bytes = 0
    top_level = sorted(item.name for item in raw.iterdir())

    for root_text, dirs, files in os.walk(raw):
        root = Path(root_text)
        directory_count += len(dirs)
        for filename in files:
            path = root / filename
            relative = path.relative_to(raw).as_posix()
            try:
                size = path.stat().st_size
            except OSError as exc:
                raise RuntimeError(f"Could not stat {path}: {exc}") from exc
            suffix = path.suffix.lower() or "[no_extension]"
            file_count += 1
            total_bytes += size
            extensions[suffix] += 1
            bytes_by_extension[suffix] += size
            if len(samples) < args.sample_limit:
                samples.append(relative)
            if suffix in ANNOTATION_SUFFIXES and len(annotation_candidates) < args.sample_limit:
                annotation_candidates.append(relative)
            if suffix in ARCHIVE_SUFFIXES and len(archives) < args.sample_limit:
                archives.append(relative)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "raw_root": str(raw),
        "raw_data_modified": False,
        "top_level_entries": top_level,
        "counts": {
            "files": file_count,
            "directories_below_root": directory_count,
            "total_bytes": total_bytes,
            "by_extension": {suffix: {"files": extensions[suffix], "bytes": bytes_by_extension[suffix]} for suffix in sorted(extensions)},
        },
        "sample_relative_files": samples,
        "annotation_candidates": annotation_candidates,
        "archives_present": archives,
        "next_gate": "Review this manifest, the official terms, class taxonomy, and sequence metadata before implementing a converter or training.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "files": file_count, "bytes": total_bytes, "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
