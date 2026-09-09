#!/usr/bin/env python3
"""Train-only diagnosis for the frozen VCSC-v1 calibration design.

This script reads already-saved calibration manifests and their complete
train-pool descriptor records.  It performs no model inference, export, or
evaluation, and never reads the CCTSDB official test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import fmean, pstdev


DESCRIPTOR_NAMES = (
    "mean_luminance",
    "luminance_std",
    "mean_saturation",
    "entropy_bits",
    "laplacian_variance",
    "dark_channel_mean",
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_text(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def selected_images(manifest: dict) -> set[str]:
    images = {str(row["source_image"]) for row in manifest["files"]}
    if len(images) != int(manifest["selected_size"]):
        raise ValueError("Calibration manifest contains duplicate selected image IDs")
    if any(not image.startswith("train/images/") for image in images):
        raise ValueError("Calibration manifest contains a non-train image")
    return images


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right)


def descriptor_shift(reference: dict[str, dict], selected: set[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in DESCRIPTOR_NAMES:
        population = [float(row["descriptor"][name]) for row in reference.values()]
        chosen = [float(reference[image]["descriptor"][name]) for image in selected]
        sigma = pstdev(population)
        result[name] = {
            "candidate_mean": fmean(population),
            "candidate_std": sigma,
            "selected_mean": fmean(chosen),
            "selected_mean_shift_sd": 0.0 if sigma == 0 else (fmean(chosen) - fmean(population)) / sigma,
        }
    return result


def policy_manifest_paths(root: Path, policy: str, size: int) -> list[Path]:
    if policy == "vcsc":
        pattern = f"vcsc_k8_s*_n{size}/calibration_manifest.json"
    else:
        pattern = f"{policy}_s*_n{size}/calibration_manifest.json"
    return sorted(root.glob(pattern))


def policy_selection_summary(root: Path, policy: str, size: int, reference: dict[str, dict]) -> dict:
    entries = []
    for path in policy_manifest_paths(root, policy, size):
        manifest = load_json(path)
        selected = selected_images(manifest)
        entries.append(
            {
                "seed": int(manifest["seed"]),
                "selection_sha256": sha256_text(selected),
                "descriptor_shift": descriptor_shift(reference, selected),
                "selected_images": selected,
            }
        )
    entries.sort(key=lambda row: row["seed"])
    pairs = [
        {
            "seeds": [left["seed"], right["seed"]],
            "jaccard": jaccard(left["selected_images"], right["selected_images"]),
        }
        for left, right in combinations(entries, 2)
    ]
    return {
        "available_seeds": [entry["seed"] for entry in entries],
        "unique_selection_hashes": len({entry["selection_sha256"] for entry in entries}),
        "pairwise_selection_jaccard": pairs,
        "per_seed": [{key: value for key, value in entry.items() if key != "selected_images"} for entry in entries],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit VCSC-v1 design using train-only saved descriptors")
    parser.add_argument("--root", type=Path, default=Path("data/processed/cctsdb2021_clean/calibration"))
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifests = policy_manifest_paths(args.root, "vcsc", args.size)
    if not manifests:
        raise FileNotFoundError(f"No VCSC manifests under {args.root}")

    vcsc_rows = []
    reference_records: dict[str, dict] | None = None
    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        metadata = manifest.get("vcsc") or {}
        if int(metadata.get("clusters", -1)) != 8:
            raise ValueError(f"Expected K=8 in {manifest_path}")
        candidate_path = manifest_path.parent / "vcsc_candidate_descriptors.json"
        candidate_payload = load_json(candidate_path)
        records = candidate_payload["records"]
        candidate_by_image = {str(row["source_image"]): row for row in records}
        if len(candidate_by_image) != int(manifest["candidate_size"]):
            raise ValueError(f"Candidate descriptor count mismatch in {candidate_path}")
        if any(not image.startswith("train/images/") for image in candidate_by_image):
            raise ValueError(f"Candidate descriptor file is not train-only: {candidate_path}")
        selected = selected_images(manifest)
        if not selected <= candidate_by_image.keys():
            raise ValueError(f"Selected images missing from candidate descriptors: {manifest_path}")
        candidate_counts = Counter(int(row["cluster"]) for row in records)
        selected_counts = Counter(int(candidate_by_image[image]["cluster"]) for image in selected)
        clusters = list(range(int(metadata["clusters"])))
        quota = {int(key): int(value) for key, value in metadata["quota_by_cluster"].items()}
        if set(candidate_counts) != set(clusters) or set(selected_counts) != set(clusters):
            raise ValueError(f"Missing VCSC cluster in {manifest_path}")
        if sum(selected_counts.values()) != args.size:
            raise ValueError(f"Unexpected VCSC selected size in {manifest_path}")
        per_cluster = []
        for cluster in clusters:
            candidate_n = candidate_counts[cluster]
            selected_n = selected_counts[cluster]
            candidate_fraction = candidate_n / len(records)
            selected_fraction = selected_n / args.size
            per_cluster.append(
                {
                    "cluster": cluster,
                    "candidate_images": candidate_n,
                    "candidate_fraction": candidate_fraction,
                    "selected_images": selected_n,
                    "selected_fraction": selected_fraction,
                    "quota": quota[cluster],
                    "selection_overrepresentation_factor": selected_fraction / candidate_fraction,
                    "quota_feasibility_margin_images": candidate_n - quota[cluster],
                }
            )
        vcsc_rows.append(
            {
                "seed": int(manifest["seed"]),
                "selected_restart": metadata.get("selected_restart"),
                "kmeans_seed": metadata.get("kmeans_seed"),
                "selection_sha256": sha256_text(selected),
                "candidate_descriptor_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                "equal_quota_sampling": len(set(selected_counts.values())) == 1,
                "clusters": per_cluster,
                "selected_images": selected,
            }
        )
        if int(manifest["seed"]) == 42:
            reference_records = candidate_by_image

    if reference_records is None:
        raise ValueError("VCSC seed 42 is required as the common train-only descriptor reference")
    vcsc_rows.sort(key=lambda row: row["seed"])
    vcsc_pairs = [
        {"seeds": [left["seed"], right["seed"]], "jaccard": jaccard(left["selected_images"], right["selected_images"])}
        for left, right in combinations(vcsc_rows, 2)
    ]
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "VCSC-v1 design diagnosis; train-only manifests and train-only visual descriptors only",
        "root": str(args.root.resolve()),
        "calibration_size": args.size,
        "vcsc": {
            "seeds": [row["seed"] for row in vcsc_rows],
            "equal_quota": args.size // 8,
            "per_seed": [{key: value for key, value in row.items() if key != "selected_images"} for row in vcsc_rows],
            "pairwise_selection_jaccard": vcsc_pairs,
        },
        "train_descriptor_selection_shift": {
            "reference": "VCSC seed-42 complete train-pool descriptors; no official-test feature or label was read",
            "uniform": policy_selection_summary(args.root, "uniform", args.size, reference_records),
            "low_luminance": policy_selection_summary(args.root, "low_luminance", args.size, reference_records),
            "vcsc": policy_selection_summary(args.root, "vcsc", args.size, reference_records),
        },
        "interpretation_guardrail": "Overrepresentation factors quantify the mechanical effect of equal cluster quotas. They diagnose VCSC-v1 only and are not evidence that a replacement policy improves official-test accuracy.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "vcsc_seeds": output["vcsc"]["seeds"], "status": "pass"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
