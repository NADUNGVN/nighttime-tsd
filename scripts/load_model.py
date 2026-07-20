#!/usr/bin/env python3
"""Load YOLO weights on server (pretrained, checkpoint, or TensorRT engine).

Examples:
  # COCO pretrained nano (auto-download)
  python scripts/load_model.py --model yolo11n.pt --info

  # Resume / fine-tuned weights after training
  python scripts/load_model.py --model runs/detect/yolo11n_cntsss/weights/best.pt --info

  # Val on CNTSSS
  python scripts/load_model.py --model runs/detect/yolo11n_cntsss/weights/best.pt \\
      --val --data configs/cntsss.yaml

  # Predict on a folder / image
  python scripts/load_model.py --model best.pt --predict data/raw/CNTSSS/test/images --save

  # Export for Jetson later (FP16 engine — run INT8 calib on Orin)
  python scripts/load_model.py --model best.pt --export engine --half
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_yolo(model_path: str):
    from ultralytics import YOLO

    p = Path(model_path)
    if not p.exists() and not str(model_path).endswith((".pt", ".yaml", ".engine", ".onnx")):
        # allow bare names like yolo11n.pt (ultralytics downloads)
        pass
    elif not p.exists() and p.suffix in {".pt", ".engine", ".onnx"} and "/" in model_path.replace("\\", "/"):
        print(f"ERROR: weights not found: {p.resolve()}", file=sys.stderr)
        sys.exit(1)
    return YOLO(model_path)


def print_info(model) -> None:
    ckpt = getattr(model, "ckpt_path", None) or getattr(model, "model_name", None)
    print("=== Model loaded ===")
    print(f"  path/name : {ckpt}")
    try:
        import torch

        print(f"  torch     : {torch.__version__}")
        print(f"  cuda      : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  gpu       : {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  vram_gb   : {vram:.1f}")
    except ImportError:
        pass
    # task / names if available
    names = getattr(model, "names", None)
    if names:
        print(f"  classes   : {names}")
    print("OK — model is ready for train / val / predict / export")


def main() -> int:
    parser = argparse.ArgumentParser(description="Load YOLO model on training server")
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="yolo11n.pt | yolov8n.pt | path/to/best.pt | path/to.engine",
    )
    parser.add_argument("--info", action="store_true", help="Print device + class info")
    parser.add_argument("--val", action="store_true", help="Run validation")
    parser.add_argument("--data", default="configs/cntsss.yaml")
    parser.add_argument("--predict", default=None, help="Image / folder / video path")
    parser.add_argument("--save", action="store_true", help="Save predict outputs")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--export",
        default=None,
        choices=["onnx", "engine", "torchscript"],
        help="Export format after load",
    )
    parser.add_argument("--half", action="store_true", help="FP16 for val/predict/export")
    parser.add_argument("--out-json", type=Path, default=None, help="Write val metrics JSON")
    args = parser.parse_args()

    try:
        model = load_yolo(args.model)
    except ImportError:
        print("Install: pip install -r requirements.txt", file=sys.stderr)
        return 1

    if args.info or not any([args.val, args.predict, args.export]):
        print_info(model)
        if not any([args.val, args.predict, args.export]):
            return 0

    if args.val:
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            half=args.half,
            split="test",
        )
        summary = {
            "model": args.model,
            "data": args.data,
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        print(json.dumps(summary, indent=2))
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.predict:
        model.predict(
            source=args.predict,
            imgsz=args.imgsz,
            device=args.device,
            half=args.half,
            save=args.save,
        )

    if args.export:
        kw = {"format": args.export, "imgsz": args.imgsz, "device": args.device}
        if args.half and args.export in {"engine", "onnx"}:
            kw["half"] = True
        out = model.export(**kw)
        print(f"Exported: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
