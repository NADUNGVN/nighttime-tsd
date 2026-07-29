#!/usr/bin/env python3
"""Build a deterministic INT8 calibration subset from CCTSDB training images only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

NAMES = ["prohibitory", "mandatory", "warning"]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_root(data_yaml: Path) -> Path:
    text = data_yaml.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*path:\s*(.+?)\s*(?:#.*)?$", text)
    if match is None:
        raise ValueError(f"Missing path: entry in {data_yaml}")
    root = Path(match.group(1).strip().strip("\"'"))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    if not train_images.is_dir() or not train_labels.is_dir():
        raise FileNotFoundError(f"Expected train/images and train/labels below {root}")
    return root


def median_luminance(image_path: Path) -> float:
    """A robust, inexpensive brightness statistic in the 0--255 grayscale range."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("low_luminance calibration requires Pillow; install requirements.txt") from error
    with Image.open(image_path) as image:
        grayscale = image.convert("L").resize((64, 64))
        return float(statistics.median(grayscale.getdata()))


def materialize(source: Path, destination: Path) -> str:
    """Hard-link when possible; copy only when the source is on another filesystem."""
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_yaml(destination: Path) -> Path:
    yaml_path = destination / "calibration.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "# Auto-generated. Do not use this set for training or validation.",
                f"path: {destination.resolve().as_posix()}",
                "train: images",
                "val: images",
                "test: images",
                "names:",
                "  0: prohibitory",
                "  1: mandatory",
                "  2: warning",
                "nc: 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a CCTSDB INT8 calibration subset from training images only")
    parser.add_argument("--data", type=Path, default=Path("configs/cctsdb2021_train.yaml"))
    parser.add_argument("--strategy", choices=["uniform", "low_luminance"], required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", required=True, help="Directory name under data_root/calibration")
    args = parser.parse_args()

    if args.size <= 0:
        raise ValueError("--size must be positive")
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("--name must be a single safe directory name")
    if not args.data.is_file():
        raise FileNotFoundError(f"Missing data YAML: {args.data}")

    data_root = resolve_data_root(args.data)
    image_root = data_root / "train" / "images"
    label_root = data_root / "train" / "labels"
    candidates = sorted(path for path in image_root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.size > len(candidates):
        raise ValueError(f"Requested {args.size} images but only {len(candidates)} train images are available")
    missing_labels = [path for path in candidates if not (label_root / f"{path.stem}.txt").is_file()]
    if missing_labels:
        raise FileNotFoundError(f"Missing labels for {len(missing_labels)} train images; first: {missing_labels[0]}")

    luminance: dict[Path, float] = {}
    if args.strategy == "low_luminance":
        for index, image_path in enumerate(candidates, start=1):
            luminance[image_path] = median_luminance(image_path)
            if index % 1000 == 0:
                print(f"Measured luminance: {index}/{len(candidates)}")
        selected = sorted(candidates, key=lambda path: (luminance[path], path.name))[: args.size]
    else:
        selected = sorted(random.Random(args.seed).sample(candidates, args.size), key=lambda path: path.name)

    destination = data_root / "calibration" / args.name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing calibration set: {destination}")
    (destination / "images").mkdir(parents=True)
    (destination / "labels").mkdir(parents=True)

    selected_records = []
    link_modes: set[str] = set()
    for image_path in selected:
        label_path = label_root / f"{image_path.stem}.txt"
        image_target = destination / "images" / image_path.name
        label_target = destination / "labels" / label_path.name
        link_modes.add(materialize(image_path, image_target))
        link_modes.add(materialize(label_path, label_target))
        selected_records.append(
            {
                "source_image": image_path.relative_to(data_root).as_posix(),
                "source_label": label_path.relative_to(data_root).as_posix(),
                "median_luminance": luminance.get(image_path),
            }
        )

    yaml_path = write_yaml(destination)
    values = list(luminance.values())
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "INT8 calibration only; selected exclusively from CCTSDB train/images",
        "source_data_yaml": str(args.data.resolve()),
        "source_data_yaml_sha256": sha256(args.data),
        "source_data_root": str(data_root),
        "strategy": args.strategy,
        "seed": args.seed,
        "requested_size": args.size,
        "selected_size": len(selected_records),
        "candidate_size": len(candidates),
        "link_modes": sorted(link_modes),
        "luminance_definition": "median grayscale intensity after 64x64 resize, range 0-255",
        "candidate_luminance": None if not values else {"min": min(values), "median": float(statistics.median(values)), "max": max(values)},
        "selected_luminance": None if not values else {"min": min(luminance[path] for path in selected), "median": float(statistics.median([luminance[path] for path in selected])), "max": max(luminance[path] for path in selected)},
        "calibration_yaml": str(yaml_path.resolve()),
        "files": selected_records,
    }
    manifest_path = destination / "calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {destination}")
    print(json.dumps({key: manifest[key] for key in ("strategy", "selected_size", "candidate_size", "candidate_luminance", "selected_luminance", "calibration_yaml")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
