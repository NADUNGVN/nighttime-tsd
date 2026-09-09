#!/usr/bin/env python3
"""Run the dev-only VCSC-proportional decision gate for frozen YOLO11n.

The runner intentionally has no official-test or negative-test phase. Those
artifacts were locked after the VCSC-v1 pilot and cannot select VCSC-v2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

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


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported config schema: {path}")
    return payload


def parse_seeds(raw: str, config: dict) -> list[int]:
    if raw == "initial":
        return [int(config["calibration"]["initial_seed"])]
    if raw == "stability":
        return [int(seed) for seed in config["calibration"]["stability_seeds"]]
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("--seeds must be initial, stability, or unique comma-separated integers")
    return values


def parse_policies(raw: str, config: dict) -> list[str]:
    available = config["calibration"]["policies"]
    values = list(available) if raw == "all" else [item.strip() for item in raw.split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or any(item not in available for item in values):
        raise ValueError(f"--policies must be all or a subset of: {', '.join(available)}")
    return values


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def results_root(repo: Path) -> Path:
    return repo / "results" / "calibration_method_v2" / "rtx8000" / "yolo11n"


def calibration_name(config: dict, policy: str, seed: int) -> str:
    return config["calibration"]["policies"][policy]["name_template"].format(seed=seed, size=config["calibration"]["size"])


def execute(command: list[str], repo: Path) -> None:
    print("START: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)


def write_manifest(repo: Path, config_path: Path, config: dict, seeds: list[int], policies: list[str], phase: str) -> None:
    directory = results_root(repo) / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{phase}_seeds_{'_'.join(map(str, seeds))}.json"
    if path.exists():
        print(f"SKIP manifest: {path}")
        return
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo),
        "phase": phase,
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "frozen_weights": str((repo / config["model"]["weights"]).resolve()),
        "frozen_weights_sha256": sha256(repo / config["model"]["weights"]),
        "seeds": seeds,
        "policies": {name: config["calibration"]["policies"][name] for name in policies},
        "official_test_lock": config["data"]["official_test_policy"],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_calibrations(repo: Path, config: dict, seeds: list[int], policies: list[str]) -> None:
    calibration_root = repo / "data" / "processed" / "cctsdb2021_clean" / "calibration"
    for seed in seeds:
        for policy in policies:
            definition = config["calibration"]["policies"][policy]
            destination = calibration_root / calibration_name(config, policy, seed)
            manifest_path = destination / "calibration_manifest.json"
            if destination.exists():
                if not (destination / "calibration.yaml").is_file() or not manifest_path.is_file():
                    raise RuntimeError(f"Partial calibration output exists: {destination}")
                manifest = load(manifest_path)
                if manifest.get("strategy") != definition["strategy"] or int(manifest.get("seed", -1)) != seed or int(manifest.get("selected_size", -1)) != int(config["calibration"]["size"]):
                    raise RuntimeError(f"Existing calibration does not match VCSC-v2 protocol: {destination}")
                if definition["strategy"] == "vcsc_proportional" and (manifest.get("vcsc") or {}).get("quota_policy") != "population_proportional_largest_remainder":
                    raise RuntimeError(f"Existing VCSC-proportional calibration lacks the locked quota policy: {destination}")
                print(f"SKIP calibration: {policy} seed={seed}")
                continue
            command = [sys.executable, "scripts/build_calibration_set.py", "--data", config["data"]["train_yaml"], "--strategy", definition["strategy"], "--size", str(config["calibration"]["size"]), "--seed", str(seed), "--name", destination.name]
            for key, flag in (("clusters", "--clusters"), ("kmeans_restarts", "--kmeans-restarts"), ("min_cluster_quota", "--min-cluster-quota")):
                if key in definition:
                    command.extend([flag, str(definition[key])])
            execute(command, repo)


def engine_path(repo: Path, policy: str, seed: int) -> Path:
    return results_root(repo) / "engines" / f"yolo11n_int8_{policy}_s{seed}.engine"


def export(repo: Path, config: dict, seeds: list[int], policies: list[str]) -> None:
    weights = repo / config["model"]["weights"]
    for seed in seeds:
        for policy in policies:
            engine = engine_path(repo, policy, seed)
            provenance = engine.with_suffix(".engine.provenance.json")
            if engine.is_file():
                if not provenance.is_file():
                    raise RuntimeError(f"Partial engine output exists: {engine}")
                print(f"SKIP export: {policy} seed={seed}")
                continue
            yaml = repo / "data" / "processed" / "cctsdb2021_clean" / "calibration" / calibration_name(config, policy, seed) / "calibration.yaml"
            execute([sys.executable, "scripts/export_tensorrt.py", "--weights", str(weights), "--precision", "int8", "--data", str(yaml), "--out", str(engine), "--device", str(config["runtime"]["device"]), "--imgsz", str(config["runtime"]["imgsz"]), "--batch", str(config["runtime"]["batch"]), "--expected-tensorrt-major", str(config["runtime"]["expected_tensorrt_major"])], repo)


def dev_eval(repo: Path, config: dict, label: str, engine: Path) -> None:
    root = results_root(repo)
    output = root / "dev_eval" / f"{label}.json"
    predictions = root / "dev_predictions" / f"{label}.json"
    size = root / "dev_size" / f"{label}.json"
    if output.is_file() or predictions.is_file() or size.is_file():
        if output.is_file() and predictions.is_file() and size.is_file():
            print(f"SKIP dev evaluation: {label}")
            return
        raise RuntimeError(f"Partial VCSC-v2 dev evaluation exists for {label}; quarantine before retrying")
    execute([sys.executable, "scripts/evaluate_cctsdb.py", "--engine", str(engine), "--data", config["data"]["development_yaml"], "--out", str(output), "--device", str(config["runtime"]["device"]), "--imgsz", str(config["runtime"]["imgsz"]), "--batch", str(config["runtime"]["batch"]), "--label", label, "--predictions-out", str(predictions)], repo)
    execute([sys.executable, "scripts/evaluate_cctsdb_size.py", "--predictions", str(predictions), "--xml", str((repo / config["data"]["official_xml"]).resolve()), "--out", str(size), "--split-label", "development"], repo)


def preflight(repo: Path, config: dict) -> None:
    """Check only development IDs and the train-only policy before GPU work."""
    yaml_path = repo / config["data"]["development_yaml"]
    text = yaml_path.read_text(encoding="utf-8")
    path_match = re.search(r"(?m)^\s*path:\s*(.+?)\s*$", text)
    val_match = re.search(r"(?m)^\s*val:\s*(.+?)\s*$", text)
    if path_match is None or val_match is None:
        raise ValueError(f"Development YAML must define path and val: {yaml_path}")
    data_root = Path(path_match.group(1).strip().strip("\"'"))
    if not data_root.is_absolute():
        data_root = (yaml_path.parent / data_root).resolve()
    image_root = data_root / val_match.group(1).strip().strip("\"'")
    if image_root.name != "images":
        raise ValueError(f"Development val path must end in images for label alignment: {image_root}")
    label_root = image_root.parent / "labels"
    images = sorted(path for path in image_root.iterdir() if path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png"})
    if not images:
        raise FileNotFoundError(f"No development images found: {image_root}")
    missing_labels = [path.name for path in images if not (label_root / f"{path.stem}.txt").is_file()]
    xml_path = (repo / config["data"]["official_xml"]).resolve()
    with zipfile.ZipFile(xml_path) as archive:
        xml_stems = {Path(member).stem for member in archive.namelist() if member.lower().endswith(".xml")}
    dev_stems = {path.stem for path in images}
    missing_xml = sorted(dev_stems - xml_stems)
    if missing_labels or missing_xml:
        raise RuntimeError(f"VCSC-v2 development preflight failed: missing_labels={len(missing_labels)}, missing_xml={len(missing_xml)}")
    output = results_root(repo) / "preflight.json"
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "VCSC-v2 preflight; development-image identity and train-only calibration policy verification before GPU work",
        "development_yaml": str(yaml_path.resolve()),
        "development_yaml_sha256": sha256(yaml_path),
        "development_images": len(images),
        "development_labels_missing": 0,
        "development_xml_missing": 0,
        "official_test_lock": config["data"]["official_test_policy"],
        "status": "pass",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def evaluate_dev(repo: Path, config: dict, seeds: list[int], policies: list[str]) -> None:
    # Reuse the valid isolated-cache FP16 reference from v1. It has the same
    # frozen source-weight hash and is only evaluated here on development data.
    fp16 = repo / "results" / "calibration_method_v1" / "rtx8000" / "yolo11n" / "engines" / "yolo11n_fp16_reference.engine"
    if not fp16.is_file():
        raise FileNotFoundError(f"Missing frozen FP16 reference engine: {fp16}")
    dev_eval(repo, config, "yolo11n_fp16_reference", fp16)
    for seed in seeds:
        for policy in policies:
            dev_eval(repo, config, f"yolo11n_int8_{policy}_s{seed}", engine_path(repo, policy, seed))


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO11n VCSC-proportional dev-only decision gate")
    parser.add_argument("--config", type=Path, default=Path("configs/yolo11n_calibration_development_v2.json"))
    parser.add_argument("--phase", choices=("status", "preflight", "calibrations", "export", "evaluate-dev"), default="status")
    parser.add_argument("--seeds", default="initial")
    parser.add_argument("--policies", default="all")
    args = parser.parse_args()
    repo = repo_root()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    config = load(config_path)
    weights = repo / config["model"]["weights"]
    if not weights.is_file():
        raise FileNotFoundError(f"Missing frozen weights: {weights}")
    seeds = parse_seeds(args.seeds, config)
    policies = parse_policies(args.policies, config)
    if args.seeds == "stability":
        requested = set(seeds)
        incompatible = [
            policy
            for policy in policies
            if set(config["calibration"]["policy_stability_seeds"][policy]) != requested
        ]
        if incompatible:
            raise ValueError(
                "These policies do not have the requested five-seed stability design: "
                + ", ".join(incompatible)
                + ". Run their deterministic initial seed separately, or use --policies uniform,vcsc_proportional."
            )
    write_manifest(repo, config_path, config, seeds, policies, args.phase)
    if args.phase == "status":
        print(json.dumps({"model": config["model"]["label"], "seeds": seeds, "policies": policies, "results_root": str(results_root(repo)), "official_test": "locked; unavailable to this runner", "scale_up": "blocked pending VCSC-v2 development gate"}, indent=2))
        return 0
    if args.phase == "preflight":
        preflight(repo, config)
        return 0
    if args.phase == "calibrations":
        build_calibrations(repo, config, seeds, policies)
        return 0
    lock = repo / "results" / "architecture_matrix_v1" / ".gpu_phase.lock"
    with GpuPhaseLock(lock, f"yolo11n_calibration_development_v2_{args.phase}"):
        if args.phase == "export":
            export(repo, config, seeds, policies)
        else:
            evaluate_dev(repo, config, seeds, policies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
