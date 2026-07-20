#!/usr/bin/env python3
"""Fine-tune YOLO11n / YOLOv8n on CNTSSS.

Desktop (RTX 5060 8GB):
  python scripts/train_baseline.py --model yolo11n.pt --batch 16 --epochs 100

Server (RTX 3090 24GB):
  python scripts/train_baseline.py --model yolo11n.pt --batch 64 --epochs 100 --workers 8
  python scripts/train_baseline.py --model yolov8n.pt --batch 64 --epochs 100 --name yolov8n_cntsss

Resume:
  python scripts/train_baseline.py --model runs/detect/yolo11n_cntsss/weights/last.pt --resume

Load only (smoke):
  python scripts/load_model.py --model yolo11n.pt --info
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Pretrained: yolo11n.pt | yolov8n.pt  OR checkpoint path for resume",
    )
    parser.add_argument("--data", default="configs/cntsss.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="5060 8GB: 8–16 | 3090 24GB: 32–64 (nano)",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Resume from --model last.pt")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache",
        default=False,
        nargs="?",
        const="ram",
        help="Cache images: omit=off, --cache or --cache ram|disk (3090 often uses ram)",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics: pip install -r requirements.txt")
        return 1

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: {data_path} not found (cwd={Path.cwd()})")
        return 1

    # Sanity: dataset root must exist for train
    # ultralytics resolves path relative to yaml parent
    print(f"data yaml : {data_path.resolve()}")
    print(f"model     : {args.model}")
    print(f"batch     : {args.batch} | device={args.device} | epochs={args.epochs}")

    name = args.name or f"{Path(str(args.model).replace(chr(92), '/')).stem}_cntsss"
    if args.resume:
        name = args.name  # keep run name if resuming; ultralytics handles via resume

    model = YOLO(args.model)

    train_kw = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        exist_ok=True,
        pretrained=True,
        resume=args.resume,
    )
    if args.name is not None or not args.resume:
        train_kw["name"] = name if not args.resume else (args.name or name)
    if args.cache:
        train_kw["cache"] = args.cache

    results = model.train(**train_kw)

    best = Path(args.project) / (args.name or name) / "weights" / "best.pt"
    last = best.with_name("last.pt")
    print("=== Train finished ===")
    print(f"  best.pt : {best if best.exists() else '(see runs/detect/*/weights/)'}")
    print(f"  last.pt : {last if last.exists() else ''}")
    print("Load on this server:")
    print(f"  python scripts/load_model.py --model {best} --info")
    print(f"  python scripts/load_model.py --model {best} --val --data {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
