#!/usr/bin/env python3
"""Build a deterministic INT8 calibration subset from CCTSDB training images only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

NAMES = ["prohibitory", "mandatory", "warning"]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_root(data_yaml: Path) -> Path:
    text = data_yaml.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*path:\s*(.+?)\s*(?:#.*)?$", text)
    if match is None:
        raise ValueError(f"Missing path: entry in {data_yaml}")
    root = Path(match.group(1).strip().strip("\"'"))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    if not train_images.is_dir() or not train_labels.is_dir():
        raise FileNotFoundError(f"Expected train/images and train/labels below {root}")
    return root


def median_luminance(image_path: Path) -> float:
    """A robust, inexpensive brightness statistic in the 0--255 grayscale range."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("low_luminance calibration requires Pillow; install requirements.txt") from error
    with Image.open(image_path) as image:
        grayscale = image.convert("L").resize((64, 64))
        pixels = grayscale.get_flattened_data() if hasattr(grayscale, "get_flattened_data") else grayscale.getdata()
        return float(statistics.median(pixels))


def visual_descriptor(image_path: Path) -> dict[str, float]:
    """Return deterministic, train-only low-cost scene descriptors for VCSC."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("VCSC requires Pillow; install requirements.txt") from error
    with Image.open(image_path) as source:
        rgb = source.convert("RGB").resize((128, 128))
        gray = np.asarray(rgb.convert("L"), dtype=np.float64)
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.float64)
        red_green_blue = np.asarray(rgb, dtype=np.float64)
    histogram = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / histogram.sum()
    entropy = -float(np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0])))
    laplacian = -4.0 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return {
        "mean_luminance": float(gray.mean()),
        "luminance_std": float(gray.std()),
        "mean_saturation": float(hsv[:, :, 1].mean()),
        "entropy_bits": entropy,
        "laplacian_variance": float(laplacian.var()),
        "dark_channel_mean": float(red_green_blue.min(axis=2).mean()),
    }


def git_commit() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, timeout=10, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def kmeans(features: np.ndarray, clusters: int, seed: int, iterations: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic K-means++ implementation; avoids a scikit-learn dependency."""
    if features.ndim != 2 or len(features) < clusters:
        raise ValueError("VCSC needs at least K feature rows")
    rng = np.random.default_rng(seed)
    centers = [features[int(rng.integers(len(features)))]]
    for _ in range(1, clusters):
        squared = np.min(np.sum((features[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=2), axis=1)
        total = float(squared.sum())
        if total <= 0:
            candidate = next(index for index in range(len(features)) if not any(np.array_equal(features[index], center) for center in centers))
        else:
            candidate = int(rng.choice(len(features), p=squared / total))
        centers.append(features[candidate])
    centers_array = np.asarray(centers, dtype=np.float64)
    assignments = np.full(len(features), -1, dtype=np.int64)
    for _ in range(iterations):
        distances = np.sum((features[:, None, :] - centers_array[None, :, :]) ** 2, axis=2)
        next_assignments = np.argmin(distances, axis=1)
        if np.array_equal(assignments, next_assignments):
            break
        assignments = next_assignments
        for cluster in range(clusters):
            members = features[assignments == cluster]
            if len(members) == 0:
                centers_array[cluster] = features[int(rng.integers(len(features)))]
            else:
                centers_array[cluster] = members.mean(axis=0)
    return assignments, centers_array


def balanced_cluster_sample(assignments: np.ndarray, size: int, clusters: int, seed: int) -> tuple[list[int], dict[int, int]]:
    """Sample as evenly as possible while retaining exactly `size` examples."""
    base, remainder = divmod(size, clusters)
    quotas = {cluster: base + (1 if cluster < remainder else 0) for cluster in range(clusters)}
    rng = random.Random(seed)
    chosen: list[int] = []
    for cluster in range(clusters):
        members = np.flatnonzero(assignments == cluster).tolist()
        if len(members) < quotas[cluster]:
            raise ValueError(f"VCSC cluster {cluster} contains {len(members)} images, below required quota {quotas[cluster]}")
        chosen.extend(rng.sample(members, quotas[cluster]))
    return sorted(chosen), quotas


def materialize(source: Path, destination: Path) -> str:
    """Hard-link when possible; copy only when the source is on another filesystem."""
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_yaml(destination: Path) -> Path:
    yaml_path = destination / "calibration.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "# Auto-generated. Do not use this set for training or validation.",
                f"path: {destination.resolve().as_posix()}",
                "train: images",
                "val: images",
                "test: images",
                "names:",
                "  0: prohibitory",
                "  1: mandatory",
                "  2: warning",
                "nc: 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a CCTSDB INT8 calibration subset from training images only")
    parser.add_argument("--data", type=Path, default=Path("configs/cctsdb2021_train.yaml"))
    parser.add_argument("--strategy", choices=["uniform", "low_luminance", "vcsc"], required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", required=True, help="Directory name under data_root/calibration")
    parser.add_argument("--clusters", type=int, default=8, help="VCSC K-means cluster count; ignored by other strategies")
    parser.add_argument("--kmeans-restarts", type=int, default=32, help="Deterministic VCSC K-means++ restarts; choose the first clustering that can satisfy all cluster quotas")
    args = parser.parse_args()

    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.clusters <= 1:
        raise ValueError("--clusters must be at least two")
    if args.kmeans_restarts <= 0:
        raise ValueError("--kmeans-restarts must be positive")
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("--name must be a single safe directory name")
    if not args.data.is_file():
        raise FileNotFoundError(f"Missing data YAML: {args.data}")

    data_root = resolve_data_root(args.data)
    image_root = data_root / "train" / "images"
    label_root = data_root / "train" / "labels"
    candidates = sorted(path for path in image_root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.size > len(candidates):
        raise ValueError(f"Requested {args.size} images but only {len(candidates)} train images are available")
    missing_labels = [path for path in candidates if not (label_root / f"{path.stem}.txt").is_file()]
    if missing_labels:
        raise FileNotFoundError(f"Missing labels for {len(missing_labels)} train images; first: {missing_labels[0]}")

    luminance: dict[Path, float] = {}
    descriptors: dict[Path, dict[str, float]] = {}
    standardized: dict[Path, dict[str, float]] = {}
    assignments: dict[Path, int] = {}
    centers: list[list[float]] | None = None
    quotas: dict[int, int] | None = None
    selected_restart: int | None = None
    kmeans_seed: int | None = None
    descriptor_names = ("mean_luminance", "luminance_std", "mean_saturation", "entropy_bits", "laplacian_variance", "dark_channel_mean")
    if args.strategy == "low_luminance":
        for index, image_path in enumerate(candidates, start=1):
            luminance[image_path] = median_luminance(image_path)
            if index % 1000 == 0:
                print(f"Measured luminance: {index}/{len(candidates)}")
        selected = sorted(candidates, key=lambda path: (luminance[path], path.name))[: args.size]
    elif args.strategy == "vcsc":
        for index, image_path in enumerate(candidates, start=1):
            descriptors[image_path] = visual_descriptor(image_path)
            if index % 1000 == 0:
                print(f"Measured VCSC descriptors: {index}/{len(candidates)}")
        matrix = np.asarray([[descriptors[path][name] for name in descriptor_names] for path in candidates], dtype=np.float64)
        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0)
        stds[stds == 0] = 1.0
        normalized = (matrix - means) / stds
        failures: list[dict[str, int]] = []
        for restart in range(args.kmeans_restarts):
            # Restart zero preserves the former single-start result whenever
            # it is quota-feasible. Later starts remain deterministic functions
            # of the calibration seed and never use test information.
            trial_seed = args.seed + restart * 1_000_003
            cluster_ids, cluster_centers = kmeans(normalized, args.clusters, trial_seed)
            try:
                selected_indices, quotas = balanced_cluster_sample(cluster_ids, args.size, args.clusters, args.seed)
            except ValueError:
                failures.append({str(cluster): int(np.count_nonzero(cluster_ids == cluster)) for cluster in range(args.clusters)})
                continue
            selected_restart = restart
            kmeans_seed = trial_seed
            break
        else:
            raise ValueError(
                f"No quota-feasible VCSC clustering in {args.kmeans_restarts} deterministic restarts; "
                f"required quota={args.size // args.clusters}, last_cluster_sizes={failures[-1] if failures else {}}"
            )
        selected = [candidates[index] for index in selected_indices]
        centers = cluster_centers.tolist()
        for index, image_path in enumerate(candidates):
            standardized[image_path] = {name: float(normalized[index, feature]) for feature, name in enumerate(descriptor_names)}
            assignments[image_path] = int(cluster_ids[index])
        feature_standardization = {name: {"mean": float(means[index]), "std": float(stds[index])} for index, name in enumerate(descriptor_names)}
    else:
        selected = sorted(random.Random(args.seed).sample(candidates, args.size), key=lambda path: path.name)

    destination = data_root / "calibration" / args.name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing calibration set: {destination}")
    (destination / "images").mkdir(parents=True)
    (destination / "labels").mkdir(parents=True)

    selected_records = []
    link_modes: set[str] = set()
    for image_path in selected:
        label_path = label_root / f"{image_path.stem}.txt"
        image_target = destination / "images" / image_path.name
        label_target = destination / "labels" / label_path.name
        link_modes.add(materialize(image_path, image_target))
        link_modes.add(materialize(label_path, label_target))
        selected_records.append(
            {
                "source_image": image_path.relative_to(data_root).as_posix(),
                "source_label": label_path.relative_to(data_root).as_posix(),
                "median_luminance": luminance.get(image_path),
                "descriptor": descriptors.get(image_path),
                "standardized_descriptor": standardized.get(image_path),
                "cluster": assignments.get(image_path),
            }
        )

    yaml_path = write_yaml(destination)
    values = list(luminance.values())
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "INT8 calibration only; selected exclusively from CCTSDB train/images",
        "source_data_yaml": str(args.data.resolve()),
        "source_data_yaml_sha256": sha256(args.data),
        "source_data_root": str(data_root),
        "strategy": args.strategy,
        "vcsc": None
        if args.strategy != "vcsc"
        else {
            "clusters": args.clusters,
            "sampling": "K-means++ initialized deterministic K-means over train-only standardized descriptors; first quota-feasible deterministic restart; equal quota per cluster",
            "kmeans_restarts": args.kmeans_restarts,
            "selected_restart": selected_restart,
            "kmeans_seed": kmeans_seed,
            "descriptor_names": list(descriptor_names),
            "descriptor_definitions": {
                "mean_luminance": "mean grayscale intensity on 128x128 RGB-to-L resize, range 0-255",
                "luminance_std": "standard deviation of grayscale intensity on the same resize",
                "mean_saturation": "mean HSV saturation on the same resize, range 0-255",
                "entropy_bits": "Shannon entropy of 256-bin grayscale histogram",
                "laplacian_variance": "variance of four-neighbor grayscale Laplacian; sharpness proxy",
                "dark_channel_mean": "mean per-pixel minimum RGB channel; visibility/haze-related proxy, not a ground-truth haze label",
            },
            "feature_standardization": feature_standardization,
            "cluster_centers_standardized": centers,
            "quota_by_cluster": {str(cluster): quota for cluster, quota in quotas.items()},
        },
        "seed": args.seed,
        "requested_size": args.size,
        "selected_size": len(selected_records),
        "candidate_size": len(candidates),
        "link_modes": sorted(link_modes),
        "luminance_definition": "median grayscale intensity after 64x64 resize, range 0-255",
        "candidate_luminance": None if not values else {"min": min(values), "median": float(statistics.median(values)), "max": max(values)},
        "selected_luminance": None if not values else {"min": min(luminance[path] for path in selected), "median": float(statistics.median([luminance[path] for path in selected])), "max": max(luminance[path] for path in selected)},
        "calibration_yaml": str(yaml_path.resolve()),
        "files": selected_records,
        "implementation": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)), "git_commit": git_commit()},
    }
    if args.strategy == "vcsc":
        candidate_path = destination / "vcsc_candidate_descriptors.json"
        candidate_payload = {
            "schema_version": 1,
            "purpose": "Complete train-only VCSC feature and cluster audit trail",
            "seed": args.seed,
            "clusters": args.clusters,
            "records": [
                {
                    "source_image": path.relative_to(data_root).as_posix(),
                    "descriptor": descriptors[path],
                    "standardized_descriptor": standardized[path],
                    "cluster": assignments[path],
                }
                for path in candidates
            ],
        }
        candidate_path.write_text(json.dumps(candidate_payload) + "\n", encoding="utf-8")
        manifest["vcsc"]["candidate_descriptor_file"] = str(candidate_path.resolve())
        manifest["vcsc"]["candidate_descriptor_file_sha256"] = sha256(candidate_path)
    manifest_path = destination / "calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {destination}")
    print(json.dumps({key: manifest[key] for key in ("strategy", "selected_size", "candidate_size", "candidate_luminance", "selected_luminance", "calibration_yaml")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
