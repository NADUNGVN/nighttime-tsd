#!/usr/bin/env python3
"""Audit the official TT100K release before any label remapping or training.

The audit is intentionally semantic-neutral: TT100K category codes are
recorded exactly as released.  A human-reviewed mapping file is required
before its categories can be collapsed into the CCTSDB three-class taxonomy.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def resolve_data_root(raw: Path) -> Path:
    raw = raw.resolve()
    candidates = [raw, raw / "data"]
    matches = [candidate for candidate in candidates if (candidate / "annotations.json").is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one TT100K data root with annotations.json below {raw}; found {matches}")
    return matches[0]


def read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing official split IDs: {path}")
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image ID in {path}")
    return ids


def lexical_family(category: str) -> str:
    match = re.match(r"[a-z]+", category)
    return match.group(0) if match else "[nonstandard]"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an official TT100K extraction without changing it")
    parser.add_argument("--raw", type=Path, required=True, help="TT100K release directory or its parent directory")
    parser.add_argument("--out", type=Path, required=True, help="Audit JSON output outside the raw release")
    parser.add_argument("--sample-limit", type=int, default=30)
    args = parser.parse_args()

    raw = args.raw.resolve()
    out = args.out.resolve()
    data_root = resolve_data_root(raw)
    try:
        out.relative_to(raw)
    except ValueError:
        pass
    else:
        raise ValueError("--out must not be inside --raw")
    if args.sample_limit < 1:
        raise ValueError("--sample-limit must be positive")

    annotation_path = data_root / "annotations.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if set(annotation) != {"imgs", "types"}:
        raise ValueError(f"Unexpected annotations.json top-level keys: {sorted(annotation)}")
    if not isinstance(annotation["imgs"], (list, dict)) or not isinstance(annotation["types"], list):
        raise ValueError("TT100K annotations must contain list/dictionary-valued imgs and list-valued types")
    # The official 2016 archive stores imgs as a dictionary keyed by image ID.
    # Accept a list as well so the audit remains compatible with equivalent
    # official serializations, but normalize before inspecting records.
    image_records = annotation["imgs"].values() if isinstance(annotation["imgs"], dict) else annotation["imgs"]

    split_ids = {split: read_ids(data_root / split / "ids.txt") for split in ("train", "test", "other")}
    id_sets = {split: set(ids) for split, ids in split_ids.items()}
    if any(id_sets[left] & id_sets[right] for left in id_sets for right in id_sets if left < right):
        raise ValueError("Official TT100K split ID files overlap")

    types = annotation["types"]
    if len(types) != len(set(types)):
        raise ValueError("Duplicate class code in annotations.json types")
    known_types = set(types)
    category_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    split_image_counts: Counter[str] = Counter()
    split_instance_counts: Counter[str] = Counter()
    missing_images: list[str] = []
    missing_image_count = 0
    unknown_categories: Counter[str] = Counter()
    invalid_boxes: list[dict[str, object]] = []
    image_paths: set[str] = set()

    for image in image_records:
        if not isinstance(image, dict) or "path" not in image or "objects" not in image:
            raise ValueError(f"Invalid image record: {image!r}")
        relative_path = str(image["path"])
        if relative_path in image_paths:
            raise ValueError(f"Duplicate annotation image path: {relative_path}")
        image_paths.add(relative_path)
        pieces = Path(relative_path).parts
        if len(pieces) != 2 or pieces[0] not in split_ids or Path(pieces[1]).suffix.lower() != ".jpg":
            raise ValueError(f"Unexpected TT100K image path: {relative_path}")
        split = pieces[0]
        image_id = Path(pieces[1]).stem
        if image_id not in id_sets[split]:
            raise ValueError(f"Annotated image not listed in {split}/ids.txt: {relative_path}")
        if not (data_root / relative_path).is_file():
            missing_image_count += 1
            if len(missing_images) < args.sample_limit:
                missing_images.append(relative_path)
        split_image_counts[split] += 1
        for obj in image["objects"]:
            if not isinstance(obj, dict):
                raise ValueError(f"Invalid object in {relative_path}: {obj!r}")
            category = str(obj.get("category", ""))
            if category not in known_types:
                unknown_categories[category] += 1
            category_counts[category] += 1
            family_counts[lexical_family(category)] += 1
            split_instance_counts[split] += 1
            bbox = obj.get("bbox")
            if not isinstance(bbox, dict):
                if len(invalid_boxes) < args.sample_limit:
                    invalid_boxes.append({"path": relative_path, "category": category, "reason": "missing_bbox"})
                continue
            try:
                xmin, ymin = float(bbox["xmin"]), float(bbox["ymin"])
                xmax, ymax = float(bbox["xmax"]), float(bbox["ymax"])
            except (KeyError, TypeError, ValueError):
                if len(invalid_boxes) < args.sample_limit:
                    invalid_boxes.append({"path": relative_path, "category": category, "reason": "malformed_bbox"})
                continue
            if not (xmax > xmin and ymax > ymin):
                if len(invalid_boxes) < args.sample_limit:
                    invalid_boxes.append({"path": relative_path, "category": category, "reason": "nonpositive_bbox", "bbox": bbox})

    expected_paths = {f"{split}/{image_id}.jpg" for split, image_ids in split_ids.items() for image_id in image_ids}
    missing_annotations = sorted(expected_paths - image_paths)
    extra_annotations = sorted(image_paths - expected_paths)
    if missing_annotations or extra_annotations:
        raise ValueError(f"Official image-ID and annotation mismatch: missing={len(missing_annotations)}, extra={len(extra_annotations)}")

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw),
        "data_root": str(data_root),
        "annotation_file": str(annotation_path),
        "raw_data_modified": False,
        "official_splits": {
            split: {"images_from_ids": len(ids), "images_annotated": split_image_counts[split], "instances": split_instance_counts[split]}
            for split, ids in split_ids.items()
        },
        "totals": {"images": len(image_paths), "instances": sum(category_counts.values()), "category_codes": len(types)},
        "category_codes": types,
        "instances_by_category": {category: category_counts[category] for category in sorted(category_counts)},
        "instances_by_lexical_family": {family: family_counts[family] for family in sorted(family_counts)},
        "integrity": {
            "missing_image_files": missing_image_count,
            "missing_image_files_sample": missing_images,
            "missing_annotations": len(missing_annotations),
            "extra_annotations": len(extra_annotations),
            "unknown_categories": {category: unknown_categories[category] for category in sorted(unknown_categories)},
            "invalid_bbox_sample": invalid_boxes,
        },
        "mapping_status": "No category-to-CCTSDB mapping is applied by this audit. Review the official taxonomy and approve a versioned mapping before conversion.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": manifest["totals"]["images"], "instances": manifest["totals"]["instances"], "category_codes": manifest["totals"]["category_codes"], "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
