#!/usr/bin/env python3
"""Evaluate one frozen CCTSDB model and save a self-describing JSON result."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path


NAMES = ["prohibitory", "mandatory", "warning"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def absolute_data_yaml(source: Path) -> Path:
    """Rewrite only path: as absolute to avoid Ultralytics global path settings."""
    text = source.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*path:\s*(.+?)\s*(?:#.*)?$", text)
    if match is None:
        raise ValueError(f"Missing path: entry in {source}")
    raw_root = match.group(1).strip().strip("\"'")
    root = Path(raw_root)
    if not root.is_absolute():
        root = (source.parent / root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root resolved from {source} does not exist: {root}")
    rewritten = text[: match.start(1)] + root.as_posix() + text[match.end(1) :]
    destination = Path("local") / f"{source.stem}_absolute.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rewritten, encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a CCTSDB model without modifying its run directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--weights", type=Path)
    group.add_argument("--engine", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--label", default=None, help="e.g. fp32, fp16_trt, int8_mixed")
    args = parser.parse_args()
    from ultralytics import YOLO
    import torch
    import ultralytics

    model_path = args.engine or args.weights
    if not model_path.is_file() or not args.data.is_file():
        raise FileNotFoundError(f"Missing model or data YAML: {model_path}, {args.data}")
    resolved_data = absolute_data_yaml(args.data)
    metrics = YOLO(str(model_path)).val(data=str(resolved_data), split="val", imgsz=args.imgsz, batch=args.batch, device=args.device, plots=False, verbose=False)
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "model": str(model_path.resolve()),
        "model_sha256": sha256(model_path),
        "data": str(args.data.resolve()),
        "data_sha256": sha256(args.data),
        "resolved_data": str(resolved_data.resolve()),
        "resolved_data_sha256": sha256(resolved_data),
        "metrics": {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map), "precision": float(metrics.box.mp), "recall": float(metrics.box.mr), "per_class_ap50": {NAMES[index]: float(value) for index, value in enumerate(metrics.box.ap50.tolist())}},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__, "cuda": torch.version.cuda},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
