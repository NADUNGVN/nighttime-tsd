#!/usr/bin/env python3
"""Measure end-to-end batch-1 TensorRT inference latency without disk I/O."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_snapshot() -> str | None:
    command = ["nvidia-smi", "--query-gpu=name,power.draw,memory.used,memory.total", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark end-to-end batch-1 TensorRT engine latency on preloaded CCTSDB images")
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Image directory; labels are never read")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()

    if not args.engine.is_file() or not args.images.is_dir():
        raise FileNotFoundError(f"Missing engine or image directory: {args.engine}, {args.images}")
    if args.out.exists() or args.warmup < 0 or args.samples <= 0:
        raise ValueError("Output must not exist; --warmup must be nonnegative and --samples positive")
    import cv2
    import torch
    from ultralytics import YOLO
    import ultralytics

    paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No images found in {args.images}")
    preloaded = [cv2.imread(str(path)) for path in paths]
    if any(image is None for image in preloaded):
        raise ValueError("At least one benchmark image could not be decoded")
    model = YOLO(str(args.engine))

    def infer(image) -> None:
        model.predict(source=image, imgsz=args.imgsz, device=args.device, conf=args.conf, iou=args.iou, verbose=False)

    for index in range(args.warmup):
        infer(preloaded[index % len(preloaded)])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    before = gpu_snapshot()
    latencies_ms: list[float] = []
    for index in range(args.samples):
        image = preloaded[index % len(preloaded)]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        infer(image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    after = gpu_snapshot()
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": str(args.engine.resolve()),
        "engine_sha256": sha256(args.engine),
        "images": str(args.images.resolve()),
        "image_pool_size": len(preloaded),
        "image_decode_in_timing": False,
        "batch": 1,
        "warmup": args.warmup,
        "samples": args.samples,
        "runtime": {"imgsz": args.imgsz, "confidence": args.conf, "iou": args.iou, "device": args.device},
        "latency_ms": {"mean": sum(latencies_ms) / len(latencies_ms), "p50": percentile(latencies_ms, 0.50), "p90": percentile(latencies_ms, 0.90), "p95": percentile(latencies_ms, 0.95), "min": min(latencies_ms), "max": max(latencies_ms)},
        "throughput_fps": 1000.0 / (sum(latencies_ms) / len(latencies_ms)),
        "gpu_before": before,
        "gpu_after": after,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
