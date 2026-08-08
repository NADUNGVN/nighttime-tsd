#!/usr/bin/env python3
"""Serial TT100K-221 to CCTSDB-3 transfer matrix for all 15 YOLO models."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_architecture_matrix import GpuPhaseLock


def read_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1:
        raise ValueError(f"Unsupported plan schema: {plan.get('schema_version')}")
    if not isinstance(plan.get("pretraining"), dict) or not isinstance(plan.get("fine_tuning"), dict) or not isinstance(plan.get("models"), list):
        raise ValueError("Plan must contain pretraining, fine_tuning, and models")
    required = {"label", "model", "rollout", "batch", "pretrain_run", "finetune_run"}
    labels: list[str] = []
    runs: list[str] = []
    for entry in plan["models"]:
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError(f"Invalid model entry: {entry}")
        if entry["rollout"] not in {"phase1", "phase2"} or not isinstance(entry["batch"], int) or entry["batch"] <= 0:
            raise ValueError(f"Invalid rollout or batch: {entry}")
        if not all(isinstance(entry[key], str) and entry[key] for key in required - {"batch"}):
            raise ValueError(f"Invalid string field: {entry}")
        labels.append(entry["label"])
        runs.extend([entry["pretrain_run"], entry["finetune_run"]])
    if len(plan["models"]) != 15 or len(set(labels)) != 15 or len(set(runs)) != 30:
        raise ValueError("Transfer matrix must contain 15 labels and 30 unique stage run names")
    if sum(entry["rollout"] == "phase1" for entry in plan["models"]) != 3:
        raise ValueError("phase1 must contain exactly three nano models")
    return plan


def run_status(repo: Path, run_name: str) -> tuple[str, Path]:
    run_dir = repo / "results" / run_name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    provenance = run_dir / "provenance.json"
    if best.is_file() and provenance.is_file():
        return "complete", run_dir
    if last.is_file():
        return "incomplete", run_dir
    if best.is_file():
        return "partial_best_without_last", run_dir
    if run_dir.exists():
        return "started_without_checkpoint", run_dir
    return "pending", run_dir


def selected_models(plan: dict[str, Any], rollout: str, only: str | None) -> list[dict[str, Any]]:
    requested = None if only is None else {item.strip() for item in only.split(",") if item.strip()}
    available = {entry["label"] for entry in plan["models"]}
    unknown = set() if requested is None else requested - available
    if unknown:
        raise ValueError("Unknown labels in --only: " + ", ".join(sorted(unknown)))
    return [entry for entry in plan["models"] if (rollout == "all" or entry["rollout"] == rollout) and (requested is None or entry["label"] in requested)]


def print_status(repo: Path, models: list[dict[str, Any]]) -> None:
    print(f"{'label':<10} {'rollout':<7} {'batch':>5}  {'pretrain':<28} finetune")
    for entry in models:
        pretrain = run_status(repo, entry["pretrain_run"])[0]
        finetune = run_status(repo, entry["finetune_run"])[0]
        print(f"{entry['label']:<10} {entry['rollout']:<7} {entry['batch']:>5}  {pretrain:<28} {finetune}")


def write_manifest(repo: Path, plan_path: Path, plan: dict[str, Any], models: list[dict[str, Any]], phase: str, rollout: str) -> Path:
    out_dir = repo / "results" / "tt100k_transfer_matrix_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(plan_path),
        "phase": phase,
        "rollout": rollout,
        "pretraining": plan["pretraining"],
        "fine_tuning": plan["fine_tuning"],
        "models": [
            {
                **entry,
                "pretrain_status": run_status(repo, entry["pretrain_run"])[0],
                "finetune_status": run_status(repo, entry["finetune_run"])[0],
            }
            for entry in models
        ],
    }
    path = out_dir / "execution_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def ensure_resumable(repo: Path, status: str, entry: dict[str, Any], stage: str, allow_resume: bool) -> Path | None:
    run_key = "pretrain_run" if stage == "pretrain" else "finetune_run"
    _, run_dir = run_status(repo, entry[run_key])
    if status == "pending":
        return None
    if status == "incomplete" and allow_resume:
        return run_dir / "weights" / "last.pt"
    if status == "incomplete":
        raise RuntimeError(f"{entry['label']} {stage} is incomplete; rerun with --resume-incomplete after inspection")
    if status in {"started_without_checkpoint", "partial_best_without_last"}:
        raise RuntimeError(f"{entry['label']} {stage} is {status}; it cannot be resumed safely")
    raise RuntimeError(f"Unexpected {stage} status {status} for {entry['label']}")


def pretrain_command(plan: dict[str, Any], entry: dict[str, Any], resume: Path | None) -> list[str]:
    config = plan["pretraining"]
    command = [
        sys.executable, "scripts/train_tt100k.py", "--model", entry["model"], "--data", config["data"],
        "--epochs", str(config["epochs"]), "--imgsz", str(config["imgsz"]), "--batch", str(entry["batch"]),
        "--workers", str(config["workers"]), "--device", str(config["device"]), "--project", str(config["project"]),
        "--name", entry["pretrain_run"], "--patience", str(config["patience"]), "--seed", str(config["seed"]),
    ]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def finetune_command(repo: Path, plan: dict[str, Any], entry: dict[str, Any], resume: Path | None) -> list[str]:
    config = plan["fine_tuning"]
    if run_status(repo, entry["pretrain_run"])[0] != "complete":
        raise RuntimeError(f"Cannot fine-tune {entry['label']}; TT100K pretraining is not complete")
    pretrain_best = repo / "results" / entry["pretrain_run"] / "weights" / "best.pt"
    if not pretrain_best.is_file():
        raise FileNotFoundError(f"Cannot fine-tune {entry['label']}; missing TT100K checkpoint {pretrain_best}")
    command = [
        sys.executable, "scripts/train_cctsdb.py", "--model", str(pretrain_best), "--data", config["data"],
        "--epochs", str(config["epochs"]), "--imgsz", str(config["imgsz"]), "--batch", str(entry["batch"]),
        "--workers", str(config["workers"]), "--device", str(config["device"]), "--project", str(config["project"]),
        "--name", entry["finetune_run"], "--patience", str(config["patience"]), "--seed", str(config["seed"]),
    ]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def run_stage(repo: Path, plan_path: Path, plan: dict[str, Any], models: list[dict[str, Any]], stage: str, rollout: str, resume_incomplete: bool) -> int:
    run_key = "pretrain_run" if stage == "pretrain" else "finetune_run"
    for entry in models:
        status, _ = run_status(repo, entry[run_key])
        if status == "complete":
            print(f"SKIP complete {stage}: {entry['label']}", flush=True)
            continue
        try:
            resume = ensure_resumable(repo, status, entry, stage, resume_incomplete)
            command = pretrain_command(plan, entry, resume) if stage == "pretrain" else finetune_command(repo, plan, entry, resume)
        except (FileNotFoundError, RuntimeError) as error:
            write_manifest(repo, plan_path, plan, models, f"{stage}_blocked", rollout)
            print(f"STOP: {error}")
            return 2
        print("\nSTART " + stage + " " + entry["label"] + ": " + " ".join(command), flush=True)
        write_manifest(repo, plan_path, plan, models, stage, rollout)
        try:
            subprocess.run(command, cwd=repo, check=True)
        except subprocess.CalledProcessError as error:
            write_manifest(repo, plan_path, plan, models, f"{stage}_failed", rollout)
            print(f"FAILED {stage}: {entry['label']} (exit {error.returncode}). No later model was started.")
            return error.returncode or 1
        final_status, _ = run_status(repo, entry[run_key])
        if final_status != "complete":
            write_manifest(repo, plan_path, plan, models, f"{stage}_failed", rollout)
            print(f"FAILED {stage}: {entry['label']} ended with status={final_status}")
            return 1
        print(f"COMPLETE {stage}: {entry['label']}", flush=True)
    write_manifest(repo, plan_path, plan, models, f"{stage}_complete", rollout)
    print(f"ALL SELECTED {stage.upper()} RUNS COMPLETE")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Serial 15-model TT100K-221 to CCTSDB-3 transfer matrix")
    parser.add_argument("--plan", type=Path, default=Path("configs/tt100k_transfer_matrix_v1.json"))
    parser.add_argument("--phase", choices=("status", "pretrain", "finetune"), default="status")
    parser.add_argument("--rollout", choices=("phase1", "phase2", "all"), default="phase1")
    parser.add_argument("--only", help="Comma-separated labels; cannot bypass their configured rollout")
    parser.add_argument("--resume-incomplete", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_path = args.plan if args.plan.is_absolute() else repo / args.plan
    plan = read_plan(plan_path)
    models = selected_models(plan, args.rollout, args.only)
    if not models:
        raise ValueError("No models selected")
    if args.phase == "status":
        print_status(repo, models)
        return 0
    lock_path = repo / "results" / "architecture_matrix_v1" / ".gpu_phase.lock"
    with GpuPhaseLock(lock_path, f"tt100k_transfer_{args.phase}_{args.rollout}"):
        return run_stage(repo, plan_path, plan, models, args.phase, args.rollout, args.resume_incomplete)


if __name__ == "__main__":
    raise SystemExit(main())
