#!/usr/bin/env python3
"""Read-only integrity audit for the five-seed YOLO11n calibration inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


POLICIES = ("uniform", "low_luminance", "vcsc")


def selection_hash(records: list[dict]) -> str:
    values = "\n".join(record["source_image"] for record in records)
    return hashlib.sha256(values.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify train-only, deterministic five-seed calibration manifests without writing files")
    parser.add_argument("--root", type=Path, default=Path("data/processed/cctsdb2021_clean/calibration"))
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    rows: list[dict] = []
    failures: list[str] = []
    for policy in POLICIES:
        for seed in seeds:
            name = f"vcsc_k8_s{seed}_n{args.size}" if policy == "vcsc" else f"{policy}_s{seed}_n{args.size}"
            path = args.root / name / "calibration_manifest.json"
            if not path.is_file():
                failures.append(f"missing manifest: {path}")
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            records = manifest.get("files", [])
            valid_train_only = all(record.get("source_image", "").startswith("train/images/") and record.get("source_label", "").startswith("train/labels/") for record in records)
            unique = len({record.get("source_image") for record in records})
            if manifest.get("strategy") != policy or manifest.get("seed") != seed or len(records) != args.size or unique != args.size or not valid_train_only:
                failures.append(f"invalid manifest: {path}")
            row = {"policy": policy, "seed": seed, "selected_size": len(records), "unique_images": unique, "train_only": valid_train_only, "selection_sha256": selection_hash(records)}
            if policy == "vcsc":
                clusters = Counter(record.get("cluster") for record in records)
                row.update({"cluster_counts": {str(key): value for key, value in sorted(clusters.items())}, "selected_restart": (manifest.get("vcsc") or {}).get("selected_restart"), "kmeans_seed": (manifest.get("vcsc") or {}).get("kmeans_seed")})
                if sorted(clusters.values()) != [args.size // 8] * 8:
                    failures.append(f"unbalanced VCSC clusters: {path}")
            rows.append(row)
    grouped = {policy: [row["selection_sha256"] for row in rows if row["policy"] == policy] for policy in POLICIES}
    payload = {"schema_version": 1, "root": str(args.root.resolve()), "seeds": seeds, "rows": rows, "unique_selection_hashes_by_policy": {policy: len(set(values)) for policy, values in grouped.items()}, "expected_interpretation": "Uniform and VCSC should normally have five distinct selection hashes. Low-Luminance is an exact ranked policy and may have one hash across seeds.", "failures": failures, "status": "pass" if not failures else "fail"}
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
