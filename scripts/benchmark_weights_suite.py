#!/usr/bin/env python3
"""Benchmark frozen PyTorch YOLO checkpoints on preloaded CCTSDB images."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
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


def parse_weights(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("--weights must be LABEL=PATH")
    if not label.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError(f"Invalid checkpoint label: {label}")
    return label, Path(raw_path)


def gpu_snapshot() -> str | None:
    command = ["nvidia-smi", "--query-gpu=name,power.draw,memory.used,memory.total", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark FP32 PyTorch checkpoints, batch one, with disk decoding excluded")
    parser.add_argument("--weights", type=parse_weights, action="append", required=True, help="LABEL=PATH; repeat for each checkpoint")
    parser.add_argument("--images", type=Path, required=True, help="Image directory; labels are never read")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()

    checkpoints = dict(args.weights)
    if len(checkpoints) != len(args.weights):
        raise ValueError("Each --weights label must be unique")
    if not args.images.is_dir() or args.warmup < 0 or args.samples <= 0:
        raise ValueError("--images must be a directory; --warmup must be nonnegative; --samples must be positive")
    for label, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {label}: {checkpoint}")
        output = args.out_dir / f"{label}.json"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite an existing benchmark: {output}")

    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No images found in {args.images}")
    preloaded = [cv2.imread(str(path)) for path in paths]
    if any(image is None for image in preloaded):
        raise ValueError("At least one benchmark image could not be decoded")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str]] = []
    for label, checkpoint in checkpoints.items():
        print(f"\n=== {label} ===", flush=True)
        model = YOLO(str(checkpoint))

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
        mean_ms = sum(latencies_ms) / len(latencies_ms)
        summary = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "model": str(checkpoint.resolve()),
            "model_sha256": sha256(checkpoint),
            "model_bytes": checkpoint.stat().st_size,
            "images": str(args.images.resolve()),
            "image_pool_size": len(preloaded),
            "image_decode_in_timing": False,
            "batch": 1,
            "warmup": args.warmup,
            "samples": args.samples,
            "runtime": {"imgsz": args.imgsz, "confidence": args.conf, "iou": args.iou, "device": args.device, "representation": "PyTorch FP32"},
            "latency_ms": {"mean": mean_ms, "p50": percentile(latencies_ms, 0.50), "p90": percentile(latencies_ms, 0.90), "p95": percentile(latencies_ms, 0.95), "min": min(latencies_ms), "max": max(latencies_ms)},
            "throughput_fps": 1000.0 / mean_ms,
            "gpu_before": before,
            "gpu_after": after,
            "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__},
        }
        output = args.out_dir / f"{label}.json"
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        results.append({"label": label, "result": str(output), "result_sha256": sha256(output)})
        print(json.dumps({"label": label, "mean_ms": mean_ms, "throughput_fps": summary["throughput_fps"]}, indent=2), flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "checkpoints": {label: {"path": str(path.resolve()), "sha256": sha256(path)} for label, path in checkpoints.items()},
        "results": results,
    }
    manifest_path = args.out_dir / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nDONE: {len(results)} benchmarks")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
