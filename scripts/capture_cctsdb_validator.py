#!/usr/bin/env python3
"""G0 single-pass frozen YOLO11n/dev capture. No training, export or policy selection."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ap_per_class

from audit_cctsdb_measurement import NAMES, SUFFIXES, sha256
from run_architecture_matrix import GpuPhaseLock

PINNED_VERSION = "8.4.102"
REPRESENTATIONS = ("fp16", "uniform", "low_luminance", "vcsc_proportional")
FROZEN_WEIGHTS_SHA256 = "3e5fc7a2148c16539cd9fb7cc7cacd81a4eec1dfc28143cdf9b6dcd872ba4ab8"


def capture_inputs(repo, representation):
    """Allow only the existing reference and initial v2 INT8 engines."""
    if representation not in REPRESENTATIONS:
        raise ValueError("Unsupported representation")
    base = repo / "results/calibration_method_v2/rtx8000/yolo11n"
    label = "yolo11n_fp16_reference" if representation == "fp16" else f"yolo11n_int8_{representation}_s42"
    engine = (repo / "results/calibration_method_v1/rtx8000/yolo11n/engines" if representation == "fp16" else base / "engines") / f"{label}.engine"
    provenance = Path(str(engine) + ".provenance.json")
    previous = base / "dev_eval" / f"{label}.json"
    for p in (engine, provenance, previous):
        if not p.is_file():
            raise FileNotFoundError(f"Missing: {p}")
    old = json.loads(previous.read_text(encoding="utf-8"))
    prov = json.loads(provenance.read_text(encoding="utf-8"))
    digest = sha256(engine)
    if digest != old["model_sha256"] or digest != prov["engine_sha256"]:
        raise ValueError("Engine hash differs from existing evaluation/provenance")
    if prov["source_weights_sha256"] != FROZEN_WEIGHTS_SHA256:
        raise ValueError("Source weights are not the frozen YOLO11n weights")
    if Path(old["data"]).name != "cctsdb2021_dev.yaml":
        raise ValueError("Historical reference is not dev")
    if prov["precision"] != ("fp16" if representation == "fp16" else "int8"):
        raise ValueError("Precision mismatch")
    if representation != "fp16":
        cache = prov.get("calibration_cache", {})
        if cache.get("isolation") != "fresh per-engine temporary workspace; never shared across policies or seeds" or not cache.get("sha256"):
            raise ValueError("INT8 calibration cache isolation evidence missing")
    return engine, provenance, previous, old, prov, digest


def serial(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    return value


def metric_summary(metrics):
    return {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "per_class_ap50": {NAMES[int(c)]: float(ap[0]) for c, ap in zip(metrics.box.ap_class_index, metrics.box.all_ap)}}


class CaptureValidator(DetectionValidator):
    """Intercept the parent's actual matching call, not a second predict/val pass."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_records = []

    def _process_batch(self, preds, batch):
        matched = super()._process_batch(preds, batch)
        # scale_preds clones bboxes; never mutate predictions used by metrics.
        original = self.scale_preds(preds, batch)
        no_pred = preds["cls"].shape[0] == 0
        self.capture_records.append({
            "image": Path(batch["im_file"]).name,
            "orig_shape": serial(batch["ori_shape"]),
            "xyxy": serial(original["bboxes"]),
            "confidence": serial(preds["conf"]), "class_id": [int(c) for c in serial(preds["cls"])],
            "validator_input": {"imgsz": serial(batch["imgsz"]), "ratio_pad": serial(batch["ratio_pad"]),
                "prediction_xyxy": serial(preds["bboxes"]), "target_xyxy": serial(batch["bboxes"]),
                "target_class_id": serial(batch["cls"])},
            "validator_statistics": {"tp": serial(matched["tp"]),
                # Parent update_metrics uses float64 empty arrays for these.
                "confidence_dtype": "float64" if no_pred else str(preds["conf"].cpu().numpy().dtype),
                "pred_class_dtype": "float64" if no_pred else str(preds["cls"].cpu().numpy().dtype),
                "target_class_dtype": str(batch["cls"].cpu().numpy().dtype)},
        })
        return matched


def replay_statistics(records):
    """Reproduce AP/P/R from saved actual TP decisions, preserving order/dtypes."""
    if not records:
        raise ValueError("No captured records")
    tp, conf, pred_cls, target_cls = [], [], [], []
    for r in records:
        st = r["validator_statistics"]
        n = len(r["confidence"])
        tp.append(np.asarray(st["tp"], dtype=bool).reshape(n, 10))
        conf.append(np.asarray(r["confidence"], dtype=st["confidence_dtype"]))
        pred_cls.append(np.asarray(r["class_id"], dtype=st["pred_class_dtype"]))
        target_cls.append(np.asarray(r["validator_input"]["target_class_id"], dtype=st["target_class_dtype"]))
    values = ap_per_class(np.concatenate(tp), np.concatenate(conf), np.concatenate(pred_cls), np.concatenate(target_cls), plot=False)
    p, recall, ap, ids = values[2], values[3], values[5], values[6]
    return {"map50": float(ap[:, 0].mean()), "map50_95": float(ap.mean()),
            "precision": float(p.mean()), "recall": float(recall.mean()),
            "per_class_ap50": {NAMES[int(c)]: float(row[0]) for c, row in zip(ids, ap)}}


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="fp16")
    parser.add_argument("--repeat-study", type=Path, help="Scoped Step-A Uniform build study; no arbitrary engine")
    parser.add_argument("--repeat-index", type=int, choices=(1,2,3))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if ultralytics.__version__ != PINNED_VERSION:
        parser.error(f"Activate nighttime-tsd: requires ultralytics {PINNED_VERSION}; do not upgrade")
    import tensorrt as trt
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; this capture requires an existing TensorRT engine")
    if args.repeat_study is not None:
        if args.repeat_index is None or args.representation != 'fp16':
            parser.error('Repeat capture requires --repeat-index and no --representation override')
        from uniform_build_repeat import repeat_capture_inputs
        engine, provenance, previous, old, prov, engine_hash = repeat_capture_inputs(repo,args.repeat_study,args.repeat_index)
        args.representation = f'uniform_build_repeat_{args.repeat_index}'
    else:
        if args.repeat_index is not None:
            parser.error('--repeat-index requires --repeat-study')
        engine, provenance, previous, old, prov, engine_hash = capture_inputs(repo, args.representation)
    if trt.__version__ != prov["environment"]["tensorrt_python"]:
        parser.error("TensorRT version differs from engine export; do not rebuild automatically")
    split = repo / "data/processed/cctsdb2021_clean/dev"
    if args.out_dir.exists():
        parser.error("Output exists; use a new version, never overwrite")
    for p in (engine, provenance, previous, split / "images", split / "labels"):
        if not p.exists():
            parser.error(f"Missing: {p}")
    expected = {p.name for p in (split / "images").iterdir() if p.suffix.lower() in SUFFIXES}
    if len(expected) != 1636 or {Path(n).stem for n in expected} != {p.stem for p in (split / "labels").glob("*.txt")}:
        parser.error("Expected exactly 1636 dev images with matching labels")
    with GpuPhaseLock(repo / "results/architecture_matrix_v1/.gpu_phase.lock", f"g0_{args.representation}_dev_capture"):
        args.out_dir.mkdir(parents=True)
        data = args.out_dir / "dev_absolute.yaml"
        data.write_text(f"path: {split.as_posix()}\ntrain: images\nval: images\nnames:\n  0: prohibitory\n  1: mandatory\n  2: warning\nnc: 3\n", encoding="utf-8")
        runtime = {"model": str(engine), "data": str(data.resolve()), "split": "val", "device": args.device,
            "imgsz": 640, "batch": 1, "workers": 0, "task": "detect", "mode": "val", "conf": 0.001,
            "iou": 0.7, "max_det": 300, "rect": False, "plots": False, "verbose": False, "save_json": False, "save_txt": False}
        validator = CaptureValidator(args=runtime, save_dir=args.out_dir / "validator")
        validator(model=str(engine))
        records = validator.capture_records
        names = [r["image"] for r in records]
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("Captured records do not cover dev exactly once")
        captured = {"schema_version": 2, "capture_mode": "same_val_process_batch", "model_sha256": engine_hash,
            "data_sha256": sha256(data), "iou_thresholds": serial(validator.iouv), "records": records,
            "coordinate_contract": "xyxy is scaled/clipped original-image coordinates for inspection; AP uses validator_input coordinates and recorded TP statistics. Do not substitute scaled boxes for matching."}
        predictions = args.out_dir / "validator_predictions.json"
        write_json(predictions, captured)
        # Read back from disk: validate the serialized artifact, not only RAM.
        replay = replay_statistics(json.loads(predictions.read_text(encoding="utf-8"))["records"])
        measured = metric_summary(validator.metrics)
        differences = {k: replay[k] - measured[k] for k in ("map50", "map50_95", "precision", "recall")}
        class_differences = {k: replay["per_class_ap50"][k]-v for k, v in measured["per_class_ap50"].items()}
        passed = all(abs(v) <= 1e-12 for v in [*differences.values(), *class_differences.values()])
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
        result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
            "representation": args.representation, "source_weights_sha256": prov["source_weights_sha256"],
            "script_sha256": sha256(Path(__file__)), "model_sha256": engine_hash, "engine_provenance_sha256": sha256(provenance),
            "dataset_split": "CCTSDB2021/dev", "runtime_arguments": runtime, "resolved_arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(validator.args).items()},
            "environment": {"python": platform.python_version(), "ultralytics": ultralytics.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__, "tensorrt": trt.__version__, "gpu": torch.cuda.get_device_name(validator.device)},
            "matching_source_sha256": hashlib.sha256(inspect.getsource(DetectionValidator._process_batch).encode()).hexdigest(),
            "ap_source_sha256": hashlib.sha256(inspect.getsource(ap_per_class).encode()).hexdigest(),
            "images": len(records), "instances": sum(len(r["validator_input"]["target_class_id"]) for r in records),
            "metrics": measured, "statistics_replay": replay, "delta_replay_minus_val": differences,
            "delta_per_class_ap50": class_differences, "replay_tolerance": 1e-12,
            "previous_evaluation_sha256": sha256(previous), "delta_current_minus_historical_val": {k: measured[k]-old["metrics"][k] for k in differences},
            "predictions_sha256": sha256(predictions), "status": "pass" if passed else "review_required",
            "scope": "Pass certifies same-run statistics replay only; XML/size convention and global G0 are not resolved."}
        write_json(args.out_dir / "capture_report.json", result)
        print(json.dumps({k: result[k] for k in ("status", "images", "instances", "metrics", "delta_replay_minus_val", "scope")}, indent=2))
        print(f"DONE: {args.out_dir / 'capture_report.json'}")
        return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
