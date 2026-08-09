#!/usr/bin/env python3
"""Export a fixed YOLO checkpoint to a TensorRT engine with provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_version(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout + completed.stderr).strip()
    return output or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a YOLO checkpoint as an FP16 or INT8 TensorRT engine")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp16", "int8"], required=True)
    parser.add_argument("--out", type=Path, required=True, help="Stable destination for the generated engine")
    parser.add_argument("--data", type=Path, help="Required INT8 calibration YAML produced by build_calibration_set.py")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workspace", type=float, default=None, help="TensorRT workspace limit in GiB, if needed")
    parser.add_argument("--expected-tensorrt-major", type=int, default=10, help="Fail if the local TensorRT major version differs; use the target device's recorded major version")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.weights}")
    if args.precision == "int8" and (args.data is None or not args.data.is_file()):
        raise ValueError("INT8 export requires --data pointing to a generated calibration YAML")
    if args.precision == "fp16" and args.data is not None:
        raise ValueError("FP16 export does not accept --data; calibration is INT8-only")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing engine: {args.out}")

    import tensorrt as trt
    from ultralytics import YOLO
    import torch
    import ultralytics

    trt_major = int(trt.__version__.split(".", 1)[0])
    if args.expected_tensorrt_major < 8 or args.expected_tensorrt_major > 10:
        raise ValueError("This protocol supports native TensorRT PTQ majors 8, 9, or 10 only")
    if trt_major != args.expected_tensorrt_major:
        raise RuntimeError(
            f"Expected TensorRT {args.expected_tensorrt_major}.x but found TensorRT {trt.__version__}. "
            "Record the target environment and pass its supported native PTQ major explicitly. TensorRT 11 is not "
            "supported by this protocol because Ultralytics routes INT8 export through ModelOpt."
        )

    export_args = {
        "format": "engine",
        "imgsz": args.imgsz,
        "device": args.device,
        "batch": args.batch,
        "half": args.precision == "fp16",
        "int8": args.precision == "int8",
    }
    if args.data is not None:
        export_args["data"] = str(args.data.resolve())
    if args.workspace is not None:
        export_args["workspace"] = args.workspace

    exported = Path(YOLO(str(args.weights)).export(**export_args))
    if not exported.is_file():
        raise FileNotFoundError(f"Ultralytics reported an engine that does not exist: {exported}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, args.out)

    calibration_manifest = args.data.parent / "calibration_manifest.json" if args.data is not None else None
    provenance = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "precision": args.precision,
        "source_weights": str(args.weights.resolve()),
        "source_weights_sha256": sha256(args.weights),
        "engine": str(args.out.resolve()),
        "engine_sha256": sha256(args.out),
        "calibration_yaml": None if args.data is None else str(args.data.resolve()),
        "calibration_yaml_sha256": None if args.data is None else sha256(args.data),
        "calibration_manifest": None if calibration_manifest is None or not calibration_manifest.is_file() else str(calibration_manifest.resolve()),
        "calibration_manifest_sha256": None if calibration_manifest is None or not calibration_manifest.is_file() else sha256(calibration_manifest),
        "export_args": export_args,
        "expected_tensorrt_major": args.expected_tensorrt_major,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda": torch.version.cuda,
            "tensorrt_python": trt.__version__,
            "trtexec_version": command_version(["trtexec", "--version"]),
        },
    }
    provenance_path = args.out.with_suffix(args.out.suffix + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("DONE")
    print(f"engine: {args.out}")
    print(f"provenance: {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
