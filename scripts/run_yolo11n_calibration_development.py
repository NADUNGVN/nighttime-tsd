#!/usr/bin/env python3
"""Execute the frozen-YOLO11n calibration-method decision gate, never the 15-model scale-up."""
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported config schema in {path}")
    return payload


def parse_seeds(raw: str, config: dict[str, Any]) -> list[int]:
    if raw == "initial":
        return [int(config["calibration"]["initial_seed"])]
    if raw == "stability":
        return [int(value) for value in config["calibration"]["stability_seeds"]]
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or len(set(values)) != len(values):
        raise ValueError("--seeds must be initial, stability, or unique comma-separated integers")
    return values


def calibration_name(config: dict[str, Any], policy: str, seed: int) -> str:
    item = config["calibration"]["policies"][policy]
    return item["name_template"].format(seed=seed, size=config["calibration"]["size"])


def root(repo: Path) -> Path:
    return repo / "results" / "calibration_method_v1" / "rtx8000" / "yolo11n"


def engine(root_dir: Path, mode: str, seed: int | None = None) -> Path:
    suffix = "reference" if seed is None else f"s{seed}"
    return root_dir / "engines" / f"yolo11n_{mode}_{suffix}.engine"


def eval_dir(root_dir: Path, mode: str, seed: int | None = None) -> Path:
    suffix = "reference" if seed is None else f"s{seed}"
    return root_dir / "eval" / f"yolo11n_{mode}_{suffix}"


def execute(command: list[str], repo: Path) -> None:
    print("START: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)


def write_manifest(repo: Path, config_path: Path, config: dict[str, Any], seeds: list[int], phase: str) -> None:
    destination = root(repo)
    manifest_dir = destination / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = "_".join(str(seed) for seed in seeds)
    manifest_path = manifest_dir / f"{phase}_seeds_{seed_tag}.json"
    if manifest_path.exists():
        print(f"SKIP manifest: {manifest_path}")
        return
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "phase": phase,
        "frozen_model": {"weights": config["model"]["weights"], "sha256": sha256(repo / config["model"]["weights"])},
        "seeds": seeds,
        "policies": config["calibration"]["policies"],
        "policy": "Development-only YOLO11n gate. Do not interpret this manifest as authorizing 15-model scale-up.",
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_calibrations(repo: Path, config: dict[str, Any], seeds: list[int]) -> None:
    script = repo / "scripts" / "build_calibration_set.py"
    data_root = repo / "data" / "processed" / "cctsdb2021_clean" / "calibration"
    for seed in seeds:
        for policy, item in config["calibration"]["policies"].items():
            name = calibration_name(config, policy, seed)
            destination = data_root / name
            if destination.is_dir():
                if not (destination / "calibration.yaml").is_file() or not (destination / "calibration_manifest.json").is_file():
                    raise RuntimeError(f"Partial calibration output exists: {destination}")
                manifest = load(destination / "calibration_manifest.json")
                if manifest.get("strategy") != item["strategy"] or int(manifest.get("seed", -1)) != seed or int(manifest.get("selected_size", -1)) != int(config["calibration"]["size"]):
                    raise RuntimeError(
                        f"Existing calibration does not match the locked protocol: {destination} "
                        f"(strategy={manifest.get('strategy')}, seed={manifest.get('seed')}, selected_size={manifest.get('selected_size')})"
                    )
                if item["strategy"] == "vcsc" and int((manifest.get("vcsc") or {}).get("clusters", -1)) != int(item["clusters"]):
                    raise RuntimeError(f"Existing VCSC calibration has the wrong K: {destination}")
                print(f"SKIP calibration: {policy} seed={seed}")
                continue
            command = [sys.executable, str(script), "--data", config["data"]["train_yaml"], "--strategy", item["strategy"], "--size", str(config["calibration"]["size"]), "--seed", str(seed), "--name", name]
            if "clusters" in item:
                command.extend(["--clusters", str(item["clusters"])])
            execute(command, repo)


def export(repo: Path, config: dict[str, Any], seeds: list[int]) -> None:
    script = repo / "scripts" / "export_tensorrt.py"
    output = root(repo)
    weights = repo / config["model"]["weights"]
    fp16 = engine(output, "fp16")
    if not fp16.exists():
        execute([sys.executable, str(script), "--weights", str(weights), "--precision", "fp16", "--out", str(fp16), "--device", str(config["runtime"]["device"]), "--imgsz", str(config["runtime"]["imgsz"]), "--batch", str(config["runtime"]["batch"]), "--expected-tensorrt-major", str(config["runtime"]["expected_tensorrt_major"])], repo)
    for seed in seeds:
        for policy in config["calibration"]["policies"]:
            output_engine = engine(output, f"int8_{policy}", seed)
            if output_engine.exists():
                if not output_engine.with_suffix(".engine.provenance.json").is_file():
                    raise RuntimeError(f"Partial engine output exists: {output_engine}")
                print(f"SKIP export: {policy} seed={seed}")
                continue
            yaml = repo / "data" / "processed" / "cctsdb2021_clean" / "calibration" / calibration_name(config, policy, seed) / "calibration.yaml"
            execute([sys.executable, str(script), "--weights", str(weights), "--precision", "int8", "--data", str(yaml), "--out", str(output_engine), "--device", str(config["runtime"]["device"]), "--imgsz", str(config["runtime"]["imgsz"]), "--batch", str(config["runtime"]["batch"]), "--expected-tensorrt-major", str(config["runtime"]["expected_tensorrt_major"])], repo)


def evaluate_one(repo: Path, config: dict[str, Any], mode: str, source_engine: Path, seed: int | None) -> None:
    directory = eval_dir(root(repo), mode, seed)
    label = f"yolo11n_{mode}_{'reference' if seed is None else f's{seed}'}"
    size_output = root(repo) / "size" / f"{label}.json"
    if directory.exists() and any(directory.iterdir()):
        if (directory / "suite_manifest.json").is_file():
            print(f"SKIP evaluation: {mode} seed={seed}")
            if size_output.is_file():
                return
            prediction = root(repo) / "predictions" / f"{label}_full_predictions.json"
            if not prediction.is_file():
                raise RuntimeError(f"Completed suite lacks full-test predictions required for size evaluation: {prediction}")
            execute([sys.executable, str(repo / "scripts" / "evaluate_cctsdb_size.py"), "--predictions", str(prediction), "--xml", str((repo / config["data"]["official_xml"]).resolve()), "--out", str(size_output)], repo)
            return
        # The only recoverable partial state is a first full-split evaluation
        # that failed while saving raw predictions. It has no suite manifest,
        # no JSON result, and no prediction payload; delete only this empty
        # runner directory so the immutable evaluation can be retried.
        leftovers = sorted(path.name for path in directory.iterdir())
        if not leftovers:
            directory.rmdir()
        else:
            raise RuntimeError(f"Partial evaluation output exists: {directory}; inspect and quarantine it before retrying: {leftovers}")
    command = [sys.executable, str(repo / "scripts" / "evaluate_tensorrt_suite.py"), "--engine", f"yolo11n_{mode}_{'reference' if seed is None else f's{seed}'}={source_engine}", "--out-dir", str(directory), "--device", str(config["runtime"]["device"]), "--imgsz", str(config["runtime"]["imgsz"]), "--batch", str(config["runtime"]["batch"]), "--predictions-full-dir", str(root(repo) / "predictions")]
    execute(command, repo)
    execute([sys.executable, str(repo / "scripts" / "evaluate_cctsdb_size.py"), "--predictions", str(root(repo) / "predictions" / f"{label}_full_predictions.json"), "--xml", str((repo / config["data"]["official_xml"]).resolve()), "--out", str(size_output)], repo)


def evaluate(repo: Path, config: dict[str, Any], seeds: list[int]) -> None:
    evaluate_one(repo, config, "fp16", engine(root(repo), "fp16"), None)
    for seed in seeds:
        for policy in config["calibration"]["policies"]:
            mode = f"int8_{policy}"
            evaluate_one(repo, config, mode, engine(root(repo), mode, seed), seed)


def negative(repo: Path, config: dict[str, Any], seeds: list[int], source: Path) -> None:
    script = repo / "scripts" / "evaluate_cctsdb_negatives.py"
    output = root(repo) / "negative"
    targets = [("fp16", engine(root(repo), "fp16"), None)] + [(f"int8_{policy}", engine(root(repo), f"int8_{policy}", seed), seed) for seed in seeds for policy in config["calibration"]["policies"]]
    for mode, source_engine, seed in targets:
        suffix = "reference" if seed is None else f"s{seed}"
        path = output / f"yolo11n_{mode}_{suffix}.json"
        if path.exists():
            print(f"SKIP negative: {mode} seed={seed}")
            continue
        execute([sys.executable, str(script), "--engine", str(source_engine), "--negative-source", str(source), "--out", str(path), "--thresholds", ",".join(str(value) for value in config["negative_thresholds"]), "--device", str(config["runtime"]["device"]), "--imgsz", str(config["runtime"]["imgsz"]), "--batch", str(config["runtime"]["batch"]), "--label", f"yolo11n_{mode}_{suffix}"], repo)


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO11n-only calibration methodology decision gate")
    parser.add_argument("--config", type=Path, default=Path("configs/yolo11n_calibration_development_v1.json"))
    parser.add_argument("--phase", choices=("status", "calibrations", "export", "evaluate", "negative"), default="status")
    parser.add_argument("--seeds", default="initial", help="initial, stability, or comma-separated calibration seeds")
    parser.add_argument("--negative-source", type=Path, help="Required only for --phase negative; official 500-image negative directory/ZIP")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else repo / args.config
    config = load(config_path)
    seeds = parse_seeds(args.seeds, config)
    weights = repo / config["model"]["weights"]
    if not weights.is_file():
        raise FileNotFoundError(f"Frozen YOLO11n weights not found: {weights}")
    write_manifest(repo, config_path, config, seeds, args.phase)
    if args.phase == "status":
        print(json.dumps({"development_model": config["model"]["label"], "seeds": seeds, "frozen_weights": str(weights), "results_root": str(root(repo)), "scale_up": "blocked pending decision gate"}, indent=2))
        return 0
    if args.phase == "calibrations":
        build_calibrations(repo, config, seeds)
        return 0
    lock = repo / "results" / "architecture_matrix_v1" / ".gpu_phase.lock"
    with GpuPhaseLock(lock, f"yolo11n_calibration_development_{args.phase}"):
        if args.phase == "export":
            export(repo, config, seeds)
        elif args.phase == "evaluate":
            evaluate(repo, config, seeds)
        else:
            if args.negative_source is None:
                raise ValueError("--negative-source is required for --phase negative")
            negative(repo, config, seeds, args.negative_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
