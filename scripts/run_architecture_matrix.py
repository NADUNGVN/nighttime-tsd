#!/usr/bin/env python3
"""Run or audit the reproducible CCTSDB YOLO scale matrix serially."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1:
        raise ValueError(f"Unsupported plan schema: {plan.get('schema_version')}")
    if not isinstance(plan.get("training"), dict) or not isinstance(plan.get("models"), list):
        raise ValueError("Plan must contain training and models")
    labels = [entry.get("label") for entry in plan["models"]]
    runs = [entry.get("run_name") for entry in plan["models"]]
    if not all(isinstance(item, str) and item for item in labels + runs):
        raise ValueError("Every model must have a non-empty label and run_name")
    if len(set(labels)) != len(labels) or len(set(runs)) != len(runs):
        raise ValueError("Model labels and run names must be unique")
    for entry in plan["models"]:
        if not isinstance(entry.get("model"), str) or not entry["model"].endswith(".pt"):
            raise ValueError(f"Invalid pretrained model in plan: {entry}")
        if not isinstance(entry.get("batch"), int) or entry["batch"] <= 0:
            raise ValueError(f"Invalid batch in plan: {entry}")
    return plan


def selected_models(plan: dict[str, Any], raw_only: str | None) -> list[dict[str, Any]]:
    requested = None if raw_only is None else {item.strip() for item in raw_only.split(",") if item.strip()}
    available = {entry["label"] for entry in plan["models"]}
    unknown = set() if requested is None else requested - available
    if unknown:
        raise ValueError(f"Unknown labels in --only: {', '.join(sorted(unknown))}")
    return [entry for entry in plan["models"] if requested is None or entry["label"] in requested]


def status_for(repo: Path, entry: dict[str, Any]) -> tuple[str, Path]:
    run_dir = repo / "results" / entry["run_name"]
    if (run_dir / "weights" / "best.pt").is_file():
        return "complete", run_dir
    if (run_dir / "weights" / "last.pt").is_file():
        return "incomplete", run_dir
    if run_dir.exists():
        return "started_without_checkpoint", run_dir
    return "pending", run_dir


def write_manifest(repo: Path, plan_path: Path, plan: dict[str, Any], models: list[dict[str, Any]], phase: str) -> Path:
    output_dir = repo / "results" / "architecture_matrix_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(plan_path),
        "phase": phase,
        "training": plan["training"],
        "models": [
            {
                "label": entry["label"],
                "model": entry["model"],
                "run_name": entry["run_name"],
                "batch": entry["batch"],
                "reuse_complete": bool(entry.get("reuse_complete", False)),
                "status": status_for(repo, entry)[0],
                "run_dir": str(status_for(repo, entry)[1]),
            }
            for entry in models
        ],
    }
    path = output_dir / "execution_manifest.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def print_status(repo: Path, models: list[dict[str, Any]]) -> None:
    print(f"{'label':<10} {'batch':>5}  status")
    for entry in models:
        status, _ = status_for(repo, entry)
        print(f"{entry['label']:<10} {entry['batch']:>5}  {status}")


def training_command(repo: Path, plan: dict[str, Any], entry: dict[str, Any], resume: bool) -> list[str]:
    training = plan["training"]
    command = [
        sys.executable,
        "scripts/train_cctsdb.py",
        "--model", entry["model"],
        "--data", training["data"],
        "--epochs", str(training["epochs"]),
        "--imgsz", str(training["imgsz"]),
        "--batch", str(entry["batch"]),
        "--workers", str(training["workers"]),
        "--device", str(training["device"]),
        "--project", str(training["project"]),
        "--name", entry["run_name"],
        "--patience", str(training["patience"]),
        "--seed", str(training["seed"]),
    ]
    if resume:
        command.append("--resume")
    return command


def run_train(repo: Path, plan_path: Path, plan: dict[str, Any], models: list[dict[str, Any]], resume_incomplete: bool) -> int:
    for entry in models:
        status, _ = status_for(repo, entry)
        if status == "complete":
            print(f"SKIP complete: {entry['label']}", flush=True)
            continue
        if status in {"incomplete", "started_without_checkpoint"} and not resume_incomplete:
            print_status(repo, models)
            print(f"STOP: {entry['label']} is {status}. Inspect it, then rerun with --resume-incomplete if it has a valid last.pt.")
            return 2
        if status == "started_without_checkpoint":
            print(f"STOP: {entry['label']} has no last.pt and cannot be resumed safely.")
            return 2
        command = training_command(repo, plan, entry, resume=status == "incomplete")
        print("\nSTART " + entry["label"] + ": " + " ".join(command), flush=True)
        write_manifest(repo, plan_path, plan, models, "train")
        try:
            subprocess.run(command, cwd=repo, check=True)
        except subprocess.CalledProcessError as error:
            write_manifest(repo, plan_path, plan, models, "train_failed")
            print(f"FAILED: {entry['label']} (exit {error.returncode}). No later model was started.")
            return error.returncode or 1
        final_status, _ = status_for(repo, entry)
        if final_status != "complete":
            write_manifest(repo, plan_path, plan, models, "train_failed")
            print(f"FAILED: {entry['label']} finished without best.pt (status={final_status}).")
            return 1
        print(f"COMPLETE: {entry['label']}", flush=True)
        write_manifest(repo, plan_path, plan, models, "train")
    write_manifest(repo, plan_path, plan, models, "train_complete")
    print("\nALL SELECTED TRAINING RUNS COMPLETE")
    return 0


def run_eval(repo: Path, plan_path: Path, plan: dict[str, Any], models: list[dict[str, Any]], out_dir: Path) -> int:
    incomplete = [entry["label"] for entry in models if status_for(repo, entry)[0] != "complete"]
    if incomplete:
        print("STOP: cannot evaluate; missing complete checkpoints: " + ", ".join(incomplete))
        return 2
    command = [sys.executable, "scripts/evaluate_weights_suite.py"]
    for entry in models:
        checkpoint = Path("results") / entry["run_name"] / "weights" / "best.pt"
        command.extend(["--weights", f"fp32_{entry['label']}={checkpoint.as_posix()}"])
    command.extend(["--out-dir", str(out_dir), "--device", str(plan["training"]["device"]), "--imgsz", str(plan["training"]["imgsz"]), "--batch", "64"])
    print("START evaluation: " + " ".join(command), flush=True)
    write_manifest(repo, plan_path, plan, models, "evaluate")
    subprocess.run(command, cwd=repo, check=True)
    write_manifest(repo, plan_path, plan, models, "evaluate_complete")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Serial CCTSDB FP32 training/evaluation matrix for YOLOv8, YOLO11, and YOLO26")
    parser.add_argument("--plan", type=Path, default=Path("configs/architecture_matrix_v1.json"))
    parser.add_argument("--phase", choices=("status", "train", "evaluate"), default="status")
    parser.add_argument("--only", help="Comma-separated labels, for example yolo11s,yolo11m")
    parser.add_argument("--resume-incomplete", action="store_true", help="Resume only entries that contain weights/last.pt")
    parser.add_argument("--eval-out-dir", type=Path, default=Path("results/eval/fp32_architecture_matrix_v1"))
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    plan_path = args.plan if args.plan.is_absolute() else repo / args.plan
    plan = read_plan(plan_path)
    models = selected_models(plan, args.only)
    if args.phase == "status":
        print_status(repo, models)
        manifest = write_manifest(repo, plan_path, plan, models, "status")
        print(f"manifest: {manifest.relative_to(repo)}")
        return 0
    if args.phase == "train":
        return run_train(repo, plan_path, plan, models, args.resume_incomplete)
    return run_eval(repo, plan_path, plan, models, args.eval_out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
