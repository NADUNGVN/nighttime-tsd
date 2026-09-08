#!/usr/bin/env python3
"""Recoverably quarantine invalid YOLO11n INT8 artifacts made with a shared TRT cache."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def move_matches(source: Path, destination: Path, pattern: str, kind: str, moved: list[str]) -> None:
    if not source.is_dir():
        return
    for item in sorted(source.glob(pattern)):
        if (kind == "file" and not item.is_file()) or (kind == "dir" and not item.is_dir()):
            continue
        target = destination / item.name
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite quarantine artifact: {target}")
        shutil.move(str(item), target)
        moved.append(str(item))


def main() -> int:
    parser = argparse.ArgumentParser(description="Quarantine, never delete, invalid shared-cache YOLO11n INT8 artifacts")
    parser.add_argument("--root", type=Path, default=Path("results/calibration_method_v1/rtx8000/yolo11n"))
    parser.add_argument("--tag", default="shared_trt_cache_20260909")
    args = parser.parse_args()
    root = args.root.resolve()
    if root.name != "yolo11n" or not (root / "engines").is_dir():
        raise ValueError(f"Unexpected study root; refusing quarantine: {root}")
    destination = root / f"quarantine_{args.tag}"
    if destination.exists():
        raise FileExistsError(f"Quarantine destination already exists: {destination}")
    for name in ("engines", "eval", "predictions", "size", "negative", "manifests"):
        (destination / name).mkdir(parents=True, exist_ok=False)
    moved: list[str] = []
    move_matches(root / "engines", destination / "engines", "yolo11n_int8_*", "file", moved)
    move_matches(root / "eval", destination / "eval", "yolo11n_int8_*", "dir", moved)
    move_matches(root / "predictions", destination / "predictions", "yolo11n_int8_*", "file", moved)
    move_matches(root / "size", destination / "size", "yolo11n_int8_*", "file", moved)
    move_matches(root / "negative", destination / "negative", "yolo11n_int8_*", "file", moved)
    for pattern in ("export_*.json", "evaluate_*.json", "negative_*.json"):
        move_matches(root / "manifests", destination / "manifests", pattern, "file", moved)
    legacy = root / "execution_manifest.json"
    if legacy.is_file():
        target = destination / "execution_manifest_pre_cache_fix.json"
        shutil.move(str(legacy), target)
        moved.append(str(legacy))
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "INT8 artifacts were exported while Ultralytics/TensorRT reused a calibration cache located beside a shared intermediate ONNX path. These artifacts are not valid evidence of their declared calibration policy.",
        "source_root": str(root),
        "quarantine": str(destination),
        "moved_count": len(moved),
        "moved": moved,
        "preserved": ["FP16 engine/evaluations/predictions/size/negative", "frozen best.pt", "all train-only calibration manifests and images", "raw CCTSDB data"],
    }
    (destination / "quarantine_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
