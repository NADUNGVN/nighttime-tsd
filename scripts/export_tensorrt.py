#!/usr/bin/env python3
"""Export TensorRT engines: FP32 / FP16 / INT8-PTQ with calibration variants.

IMPORTANT (Ultralytics 2026):
  - Use quantize=8 (int8=True is deprecated)
  - INT8 must be calibrated ON THE TARGET DEVICE (Jetson Orin)
  - Pass data=<yaml> + fraction for calibration subset

Examples (on Orin):
  python scripts/export_tensorrt.py --weights runs/detect/yolo11n_cntsss/weights/best.pt --precision fp16
  python scripts/export_tensorrt.py --weights .../best.pt --precision int8 --calib configs/calibration_night.yaml
  python scripts/export_tensorrt.py --weights .../best.pt --precision int8 --calib configs/calibration_day.yaml
  python scripts/export_tensorrt.py --weights .../best.pt --precision int8 --calib configs/calibration_mixed.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
    )
    parser.add_argument(
        "--calib",
        default="configs/calibration_night.yaml",
        help="Dataset yaml for INT8 calibration",
    )
    parser.add_argument("--fraction", type=float, default=0.5, help="Calib fraction of val set")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace GB")
    args = parser.parse_args()

    w = Path(args.weights)
    if not w.exists():
        print(f"ERROR: weights not found: {w}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics on the Jetson target")
        return 1

    model = YOLO(str(w))
    export_kwargs = {
        "format": "engine",
        "imgsz": args.imgsz,
        "device": args.device,
        "workspace": args.workspace,
    }

    if args.precision == "fp16":
        export_kwargs["quantize"] = 16
    elif args.precision == "int8":
        export_kwargs["quantize"] = 8
        export_kwargs["data"] = args.calib
        export_kwargs["fraction"] = args.fraction
        print(f"INT8 PTQ calibrate with data={args.calib} fraction={args.fraction}")
        print("Ensure this runs ON Jetson Orin (not host-only).")
    # fp32: leave quantize unset

    out = model.export(**export_kwargs)
    print(f"Exported: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
