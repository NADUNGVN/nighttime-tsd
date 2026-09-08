#!/usr/bin/env python3
"""Run the locked IVC TensorRT calibration matrix serially on one NVIDIA target."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_architecture_matrix import GpuPhaseLock


SPLITS = ("full", "daylike", "sunny", "cloud", "night", "rain", "snow", "foggy")
INT8_MODES = ("int8_uniform", "int8_low_luminance", "int8_luminance_stratified")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema in {path}: {payload.get('schema_version')}")
    return payload


def selected_models(models: list[dict[str, Any]], raw_only: str | None) -> list[dict[str, Any]]:
    requested = None if raw_only is None else {value.strip() for value in raw_only.split(",") if value.strip()}
    labels = {entry["label"] for entry in models}
    unknown = set() if requested is None else requested - labels
    if unknown:
        raise ValueError("Unknown labels in --only: " + ", ".join(sorted(unknown)))
    return [entry for entry in models if requested is None or entry["label"] in requested]


def modes_for(target: dict[str, Any]) -> tuple[str, ...]:
    return ("fp16", *INT8_MODES) if target["int8_supported"] else ("fp16",)


def engine_path(root: Path, label: str, mode: str) -> Path:
    return root / "engines" / f"{label}_{mode}_b1.engine"


def eval_dir(root: Path, label: str, mode: str) -> Path:
    return root / "eval" / f"{label}_{mode}"


def eval_complete(root: Path, label: str, mode: str) -> bool:
    directory = eval_dir(root, label, mode)
    return (directory / "suite_manifest.json").is_file() and all((directory / f"{label}_{mode}_{split}.json").is_file() for split in SPLITS)


def benchmark_complete(root: Path, label: str, mode: str, repetitions: int) -> bool:
    directory = root / "benchmark" / f"{label}_{mode}"
    return all((directory / f"rep{index:02d}.json").is_file() for index in range(1, repetitions + 1))


def calibration_yaml(repo: Path, study: dict[str, Any], mode: str) -> Path:
    key = mode.removeprefix("int8_")
    policy = study["calibration"]["policies"][key]
    return repo / "data" / "processed" / "cctsdb2021_clean" / "calibration" / policy["name"] / "calibration.yaml"


def write_manifest(root: Path, study_path: Path, target_name: str, trt_major: int, models: list[dict[str, Any]], modes: tuple[str, ...], phase: str, repetitions: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": str(study_path),
        "study_sha256": sha256(study_path),
        "target": target_name,
        "expected_tensorrt_major": trt_major,
        "phase": phase,
        "models": [
            {
                "label": model["label"],
                "run_name": model["run_name"],
                "modes": {
                    mode: {
                        "engine": str(engine_path(root, model["label"], mode)),
                        "engine_complete": engine_path(root, model["label"], mode).is_file() and engine_path(root, model["label"], mode).with_suffix(".engine.provenance.json").is_file(),
                        "evaluation_complete": eval_complete(root, model["label"], mode),
                        "benchmark_complete": benchmark_complete(root, model["label"], mode, repetitions),
                    }
                    for mode in modes
                },
            }
            for model in models
        ],
    }
    (root / "execution_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(command: list[str], repo: Path) -> None:
    print("START: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)


def prepare_calibrations(repo: Path, study: dict[str, Any]) -> int:
    builder = repo / "scripts" / "build_calibration_set.py"
    for policy in study["calibration"]["policies"].values():
        output = repo / "data" / "processed" / "cctsdb2021_clean" / "calibration" / policy["name"]
        if output.is_dir():
            manifest = output / "calibration_manifest.json"
            yaml_path = output / "calibration.yaml"
            if not manifest.is_file() or not yaml_path.is_file():
                raise RuntimeError(f"Partial calibration directory exists: {output}; inspect before continuing")
            print(f"SKIP calibration complete: {policy['name']}")
            continue
        run([sys.executable, str(builder), "--strategy", policy["strategy"], "--size", str(study["calibration"]["size"]), "--seed", str(study["calibration"]["seed"]), "--name", policy["name"], "--data", study["data"]["train_yaml"]], repo)
    return 0


def export_all(repo: Path, study: dict[str, Any], root: Path, models: list[dict[str, Any]], modes: tuple[str, ...], trt_major: int, device: str) -> int:
    exporter = repo / "scripts" / "export_tensorrt.py"
    for model in models:
        weights = repo / "results" / model["run_name"] / "weights" / "best.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"Missing frozen checkpoint: {weights}")
        for mode in modes:
            engine = engine_path(root, model["label"], mode)
            provenance = engine.with_suffix(".engine.provenance.json")
            if engine.is_file() and provenance.is_file():
                print(f"SKIP export complete: {model['label']} {mode}")
                continue
            if engine.exists() or provenance.exists():
                raise RuntimeError(f"Partial engine artifact exists: {engine}; quarantine it before rerunning")
            command = [sys.executable, str(exporter), "--weights", str(weights), "--precision", "fp16" if mode == "fp16" else "int8", "--out", str(engine), "--device", device, "--imgsz", str(study["runtime"]["imgsz"]), "--batch", str(study["runtime"]["batch"]), "--expected-tensorrt-major", str(trt_major)]
            if mode != "fp16":
                calibration = calibration_yaml(repo, study, mode)
                if not calibration.is_file():
                    raise FileNotFoundError(f"Missing calibration YAML for {mode}: {calibration}")
                command.extend(["--data", str(calibration)])
            run(command, repo)
    return 0


def evaluate_all(repo: Path, study: dict[str, Any], root: Path, models: list[dict[str, Any]], modes: tuple[str, ...], device: str, save_full_predictions: bool) -> int:
    suite = repo / "scripts" / "evaluate_tensorrt_suite.py"
    for model in models:
        for mode in modes:
            engine = engine_path(root, model["label"], mode)
            if not engine.is_file():
                raise FileNotFoundError(f"Missing engine: {engine}")
            if eval_complete(root, model["label"], mode):
                print(f"SKIP evaluation complete: {model['label']} {mode}")
                continue
            directory = eval_dir(root, model["label"], mode)
            if directory.exists() and any(directory.iterdir()):
                raise RuntimeError(f"Partial evaluation directory exists: {directory}; quarantine it before rerunning")
            command = [sys.executable, str(suite), "--engine", f"{model['label']}_{mode}={engine}", "--out-dir", str(directory), "--device", device, "--imgsz", str(study["runtime"]["imgsz"]), "--batch", str(study["runtime"]["batch"])]
            if save_full_predictions:
                command.extend(["--predictions-full-dir", str(root / "predictions")])
            run(command, repo)
    return 0


def benchmark_all(repo: Path, study: dict[str, Any], root: Path, models: list[dict[str, Any]], modes: tuple[str, ...], device: str, repetitions: int) -> int:
    benchmark = repo / "scripts" / "benchmark_tensorrt_engine.py"
    images = repo / study["data"]["benchmark_images"]
    for model in models:
        for mode in modes:
            engine = engine_path(root, model["label"], mode)
            if not engine.is_file():
                raise FileNotFoundError(f"Missing engine: {engine}")
            directory = root / "benchmark" / f"{model['label']}_{mode}"
            for index in range(1, repetitions + 1):
                output = directory / f"rep{index:02d}.json"
                if output.is_file():
                    print(f"SKIP benchmark complete: {model['label']} {mode} rep{index:02d}")
                    continue
                run([sys.executable, str(benchmark), "--engine", str(engine), "--images", str(images), "--out", str(output), "--device", device, "--imgsz", str(study["runtime"]["imgsz"]), "--conf", str(study["runtime"]["confidence"]), "--iou", str(study["runtime"]["iou"]), "--warmup", str(study["runtime"]["warmup"]), "--samples", str(study["runtime"]["benchmark_samples"])], repo)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute or audit the locked IVC TensorRT matrix on one NVIDIA device")
    parser.add_argument("--study", type=Path, default=Path("configs/ivc_study_v1.json"))
    parser.add_argument("--target", required=True, choices=("rtx8000", "xavier_nx", "agx_xavier", "jetson_nano"))
    parser.add_argument("--phase", choices=("status", "calibrations", "export", "evaluate", "benchmark"), default="status")
    parser.add_argument("--only", help="Comma-separated model labels")
    parser.add_argument("--device", default="0")
    parser.add_argument("--expected-tensorrt-major", type=int, default=10)
    parser.add_argument("--benchmark-repetitions", type=int, default=None)
    parser.add_argument("--save-full-predictions", action="store_true", help="Save raw per-image predictions for the full official test split")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    study_path = args.study if args.study.is_absolute() else repo / args.study
    study = read_json(study_path)
    if args.phase != "status" and not study.get("scale_up_authorized", False):
        raise RuntimeError(
            "15-model scale-up is blocked by the locked protocol. Complete the YOLO11n VCSC decision gate first, "
            "review its decision report, and record an explicit authorization in the study config before scaling."
        )
    target = study["nvidia_targets"][args.target]
    architecture_path = repo / study["architecture_plan"]
    architecture = read_json(architecture_path)
    models = selected_models(architecture["models"], args.only)
    modes = modes_for(target)
    repetitions = args.benchmark_repetitions or study["runtime"]["benchmark_repetitions"]
    if repetitions <= 0:
        raise ValueError("--benchmark-repetitions must be positive")
    root = repo / "results" / "ivc_study_v1" / args.target

    write_manifest(root, study_path, args.target, args.expected_tensorrt_major, models, modes, "status" if args.phase == "status" else args.phase, repetitions)
    if args.phase == "status":
        print(f"target: {args.target} | int8_supported={target['int8_supported']} | modes={','.join(modes)}")
        for model in models:
            cells = ", ".join(f"{mode}:E={'Y' if engine_path(root, model['label'], mode).is_file() else 'N'}/V={'Y' if eval_complete(root, model['label'], mode) else 'N'}/B={'Y' if benchmark_complete(root, model['label'], mode, repetitions) else 'N'}" for mode in modes)
            print(f"{model['label']}: {cells}")
        return 0
    if args.phase == "calibrations":
        return prepare_calibrations(repo, study)
    lock_path = repo / "results" / "architecture_matrix_v1" / ".gpu_phase.lock"
    with GpuPhaseLock(lock_path, f"ivc_{args.target}_{args.phase}"):
        if args.phase == "export":
            return export_all(repo, study, root, models, modes, args.expected_tensorrt_major, args.device)
        if args.phase == "evaluate":
            return evaluate_all(repo, study, root, models, modes, args.device, args.save_full_predictions)
        return benchmark_all(repo, study, root, models, modes, args.device, repetitions)


if __name__ == "__main__":
    raise SystemExit(main())
