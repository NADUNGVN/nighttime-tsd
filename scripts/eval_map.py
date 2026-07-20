#!/usr/bin/env python3
"""Evaluate mAP on CNTSSS test and/or CCTSDB night subset.

  python scripts/eval_map.py --weights runs/detect/yolo11n_cntsss/weights/best.pt
  python scripts/eval_map.py --weights best.pt --data configs/cctsdb2021_night.yaml
  python scripts/eval_map.py --engine best.engine --data configs/cntsss.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None)
    parser.add_argument("--engine", default=None)
    parser.add_argument("--data", default="configs/cntsss.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default=0)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    model_path = args.engine or args.weights
    if not model_path:
        print("Provide --weights or --engine")
        return 1

    from ultralytics import YOLO

    model = YOLO(model_path)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        half=args.half,
        split="test" if "cntsss" in args.data else "val",
    )

    summary = {
        "model": str(model_path),
        "data": args.data,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    # per-class if available
    try:
        summary["per_class_ap50"] = [float(x) for x in metrics.box.ap50.tolist()]
    except Exception:
        pass

    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
