#!/usr/bin/env python3
"""Benchmark latency + power on Jetson (tegrastats / jtop).

Metrics for Bài 1: ms/frame, W steady-state, J/frame, FPS/W.

Typical usage ON Jetson:
  python scripts/bench_power.py --engine best.engine --source data/raw/CNTSSS/test/images --power-mode MAXN
  python scripts/bench_power.py --weights best.pt --source ... --half

Power sampling:
  - Prefer `jtop` (jetson-stats) if available
  - Fallback: parse `tegrastats` subprocess lines
  - Host without Jetson: latency-only mode with a warning
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


def sample_tegrastats(stop_event: threading.Event, samples: list[dict], interval_ms: int = 200) -> None:
    """Background thread reading tegrastats power if present."""
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return
    assert proc.stdout is not None
    for line in proc.stdout:
        if stop_event.is_set():
            break
        # Example fragment: VDD_GPU_SOC 2457mW/2457mW ... (varies by board)
        mw = []
        parts = line.replace("/", " ").split()
        for i, tok in enumerate(parts):
            if tok.endswith("mW"):
                try:
                    mw.append(float(tok.replace("mW", "")))
                except ValueError:
                    pass
        if mw:
            samples.append({"t": time.time(), "power_w": sum(mw) / 1000.0, "raw_n": len(mw)})
    proc.terminate()


def try_jtop_power() -> float | None:
    try:
        from jtop import jtop  # type: ignore

        with jtop() as jetson:
            if jetson.ok():
                # stats structure varies; try common keys
                p = jetson.power
                if isinstance(p, dict):
                    tot = p.get("tot") or p.get("total")
                    if isinstance(tot, dict) and "power" in tot:
                        return float(tot["power"]) / 1000.0  # mW → W
                    if isinstance(tot, (int, float)):
                        return float(tot) / 1000.0
    except Exception:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default=None, help="TensorRT .engine path")
    parser.add_argument("--weights", default=None, help=".pt path (fallback)")
    parser.add_argument("--source", required=True, help="Image folder or video")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--device", default=0)
    parser.add_argument("--power-mode", default="unknown", help="e.g. MAXN, 30W (label only)")
    parser.add_argument("--out", type=Path, default=Path("runs/bench/bench_result.json"))
    args = parser.parse_args()

    model_path = args.engine or args.weights
    if not model_path:
        print("Provide --engine or --weights")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics required")
        return 1

    model = YOLO(model_path)
    source = Path(args.source)
    if source.is_dir():
        imgs = sorted(
            [
                p
                for p in source.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )[: args.max_images]
    else:
        imgs = [source]

    if not imgs:
        print(f"No images under {source}")
        return 1

    # Warmup
    for im in imgs[: max(1, args.warmup)]:
        model.predict(source=str(im), imgsz=args.imgsz, half=args.half, device=args.device, verbose=False)

    power_samples: list[dict] = []
    stop = threading.Event()
    th = threading.Thread(target=sample_tegrastats, args=(stop, power_samples), daemon=True)
    th.start()

    lat_ms: list[float] = []
    t0 = time.perf_counter()
    for im in imgs:
        t_i = time.perf_counter()
        model.predict(source=str(im), imgsz=args.imgsz, half=args.half, device=args.device, verbose=False)
        lat_ms.append((time.perf_counter() - t_i) * 1000.0)
    elapsed = time.perf_counter() - t0
    stop.set()
    th.join(timeout=2)

    # Optional single jtop snapshot
    jtop_w = try_jtop_power()

    mean_ms = statistics.mean(lat_ms)
    p50 = statistics.median(lat_ms)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    power_vals = [s["power_w"] for s in power_samples]
    mean_w = statistics.mean(power_vals) if power_vals else (jtop_w or None)
    j_per_frame = (mean_w * mean_ms / 1000.0) if mean_w else None
    fps_per_w = (fps / mean_w) if mean_w and mean_w > 0 else None

    result = {
        "model": str(model_path),
        "source": str(source),
        "n_images": len(imgs),
        "power_mode_label": args.power_mode,
        "latency_ms_mean": mean_ms,
        "latency_ms_p50": p50,
        "latency_ms_std": statistics.pstdev(lat_ms) if len(lat_ms) > 1 else 0.0,
        "fps": fps,
        "wall_s": elapsed,
        "power_w_mean": mean_w,
        "power_samples": len(power_vals),
        "j_per_frame": j_per_frame,
        "fps_per_w": fps_per_w,
        "tegrastats_available": len(power_vals) > 0,
        "jtop_snapshot_w": jtop_w,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Also append CSV row for multi-run matrix
    csv_path = args.out.with_suffix(".csv")
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        if write_header:
            w.writeheader()
        w.writerow(result)

    print(json.dumps(result, indent=2))
    if mean_w is None:
        print(
            "\nNOTE: No power samples. On Jetson install jetson-stats (jtop) or ensure tegrastats is in PATH.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
