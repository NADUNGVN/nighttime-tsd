#!/usr/bin/env python3
"""Pretrain a YOLO detector on the prepared 221-class TT100K source dataset.

This script deliberately does not evaluate or read CCTSDB.  Its best
checkpoint is an input to the separately controlled CCTSDB fine-tuning stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return str(value)


def resolve_data_root(data_yaml: Path) -> Path:
    path_value: str | None = None
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*path:\s*(.+?)\s*(?:#.*)?$", line)
        if match:
            path_value = match.group(1).strip().strip("\"'")
            break
    if not path_value:
        raise ValueError(f"Missing path: entry in {data_yaml}")
    root = Path(path_value)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    for split in ("train", "val", "test"):
        if not (root / "images" / split).is_dir() or not (root / "labels" / split).is_dir():
            raise FileNotFoundError(f"Invalid prepared TT100K split {split}: {root}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain YOLO on prepared TT100K 221-class data")
    parser.add_argument("--model", default="yolo11n.pt", help="COCO-pretrained model, not a CCTSDB checkpoint")
    parser.add_argument("--data", type=Path, default=Path("data/processed/tt100k_221/tt100k_221.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="results")
    parser.add_argument("--name", default="yolo11n_tt100k221_s42_pretrain_v1")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None, help="Explicit TT100K last.pt path to resume")
    parser.add_argument("--cache", default=False, nargs="?", const="ram", help="Image cache: disabled by default")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: install dependencies first: pip install -r requirements.txt")
        return 1
    if not args.data.is_file():
        raise FileNotFoundError(f"Missing TT100K data YAML: {args.data}")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"Missing resume checkpoint: {args.resume}")
    data_root = resolve_data_root(args.data)
    source_manifest = data_root / "manifests" / "dataset_manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Missing TT100K preparation manifest: {source_manifest}")
    if len(re.findall(r"^\s+\d+:", args.data.read_text(encoding="utf-8"), flags=re.MULTILINE)) != 221:
        raise ValueError("TT100K YAML must preserve exactly 221 classes")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    project_root = Path(args.project).resolve()
    print(f"model : {args.model}")
    print(f"data  : {args.data.resolve()}")
    print(f"run   : {project_root / args.name}")
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(project_root),
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        exist_ok=True,
        pretrained=True,
        resume=str(args.resume) if args.resume is not None else False,
        cache=False if not args.cache else args.cache,
        plots=False,
    )
    run_dir = Path(model.trainer.save_dir) if getattr(model, "trainer", None) else project_root / args.name
    trainer_args = getattr(getattr(model, "trainer", None), "args", None)
    effective_args = vars(trainer_args) if hasattr(trainer_args, "__dict__") else {}
    source_model = Path(args.model)
    provenance = {
        "command": sys.argv,
        "stage": "tt100k_221_pretraining",
        "requested_training": {key: json_safe(getattr(args, key)) for key in ("model", "epochs", "imgsz", "batch", "workers", "device", "project", "name", "patience", "seed", "cache", "resume")},
        "effective_ultralytics_args": json_safe(effective_args),
        "data_yaml": str(args.data.resolve()),
        "data_yaml_sha256": sha256(args.data),
        "dataset_manifest": str(source_manifest.resolve()),
        "dataset_manifest_sha256": sha256(source_manifest),
        "source_model_sha256": sha256(source_model) if source_model.is_file() else None,
        "seed": args.seed,
        "next_stage": "Fine-tune this checkpoint on CCTSDB with scripts/train_cctsdb.py; the CCTSDB 3-class head replaces TT100K's 221-class head.",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("DONE")
    print(f"best: {run_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
