#!/usr/bin/env python3
"""Measure fixed-threshold false positives on official CCTSDB negative scenes only."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(source: Path) -> tuple[list[Path], tempfile.TemporaryDirectory[str] | None]:
    if source.is_dir():
        images = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        return images, None
    if source.is_file() and source.suffix.lower() == ".zip":
        temporary = tempfile.TemporaryDirectory(prefix="cctsdb_negative_eval_")
        with zipfile.ZipFile(source) as archive:
            archive.extractall(temporary.name)
        images = sorted(path for path in Path(temporary.name).rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        return images, temporary
    raise FileNotFoundError(f"--negative-source must be an extracted directory or official ZIP: {source}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate false detections on CCTSDB official negative scenes; this is not mAP evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--engine", type=Path)
    group.add_argument("--weights", type=Path)
    parser.add_argument("--negative-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.25,0.50,0.75")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite result: {args.out}")
    model_path = args.engine or args.weights
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model: {model_path}")
    thresholds = [float(value.strip()) for value in args.thresholds.split(",")]
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("--thresholds must be comma-separated values in [0, 1]")
    from ultralytics import YOLO
    import torch
    import ultralytics

    images, temporary = collect_images(args.negative_source)
    try:
        if len(images) != 500:
            raise ValueError(f"Expected 500 official negative images, found {len(images)} in {args.negative_source}")
        model = YOLO(str(model_path))
        results = {}
        for threshold in thresholds:
            counts: list[int] = []
            for prediction in model.predict(source=[str(path) for path in images], stream=True, imgsz=args.imgsz, batch=args.batch, device=args.device, conf=threshold, iou=args.iou, verbose=False):
                counts.append(0 if prediction.boxes is None else len(prediction.boxes))
            results[f"{threshold:.2f}"] = {
                "threshold": threshold,
                "false_positives": sum(counts),
                "false_positives_per_image": sum(counts) / len(counts),
                "images_with_at_least_one_false_detection": sum(count > 0 for count in counts),
                "fraction_images_with_at_least_one_false_detection": sum(count > 0 for count in counts) / len(counts),
            }
        payload = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Separate fixed-threshold official-negative false-positive stress test; not mixed with positive mAP benchmark and not used for threshold selection",
            "label": args.label,
            "model": str(model_path.resolve()),
            "model_sha256": sha256(model_path),
            "negative_source": str(args.negative_source.resolve()),
            "negative_source_sha256": sha256(args.negative_source) if args.negative_source.is_file() else None,
            "negative_images": len(images),
            "runtime": {"imgsz": args.imgsz, "batch": args.batch, "iou": args.iou, "device": args.device, "thresholds": thresholds},
            "metrics": results,
            "environment": {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__, "cuda": torch.version.cuda},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
