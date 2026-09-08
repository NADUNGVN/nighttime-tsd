#!/usr/bin/env python3
"""Paired 1,000-resample CIs for YOLO11n calibration policies, including macro-domain Delta mAP50."""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from bootstrap_cctsdb_map50 import load_ground_truth, map50, percentile, resolve_images


DOMAINS = ("sunny", "cloud", "rain", "snow", "foggy", "night")


def payload_records(path: Path) -> dict[str, dict]:
    return {record["image"]: record for record in json.loads(path.read_text(encoding="utf-8"))["records"]}


def names_for(data: Path) -> list[str]:
    images = resolve_images(data)
    return sorted(path.name for path in images.iterdir() if path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap paired full and macro-domain Delta mAP50 for one candidate policy")
    parser.add_argument("--fp16-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    args = parser.parse_args()
    if args.out.exists() or args.iterations <= 0:
        raise ValueError("Output must not exist and --iterations must be positive")
    fp16, candidate = payload_records(args.fp16_predictions), payload_records(args.candidate_predictions)
    split_paths = {"full": args.configs_dir / "cctsdb2021_test_full.yaml", **{name: args.configs_dir / f"cctsdb2021_test_{name}.yaml" for name in DOMAINS}}
    names = {name: names_for(path) for name, path in split_paths.items()}
    truth = {name: load_ground_truth(resolve_images(path), fp16) for name, path in split_paths.items()}
    for split, selected in names.items():
        missing = [image for image in selected if image not in fp16 or image not in candidate]
        if missing:
            raise ValueError(f"{split} predictions miss {len(missing)} images; first={missing[0]}")
    point = {split: {"fp16": map50(selected, fp16, truth[split]), "candidate": map50(selected, candidate, truth[split])} for split, selected in names.items()}
    rng = random.Random(args.seed)
    values = {split: [] for split in names}
    macro_values = []
    for _ in range(args.iterations):
        deltas = []
        for split, selected in names.items():
            sampled = [rng.choice(selected) for _ in selected]
            delta = map50(sampled, candidate, truth[split]) - map50(sampled, fp16, truth[split])
            values[split].append(delta)
            if split in DOMAINS:
                deltas.append(delta)
        macro_values.append(sum(deltas) / len(deltas))
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iterations": args.iterations,
        "seed": args.seed,
        "fp16_predictions": str(args.fp16_predictions.resolve()),
        "candidate_predictions": str(args.candidate_predictions.resolve()),
        "point_delta_map50": {split: point[split]["candidate"] - point[split]["fp16"] for split in point},
        "bootstrap_95ci_delta_map50": {split: [percentile(samples, 0.025), percentile(samples, 0.975)] for split, samples in values.items()},
        "point_macro_domain_delta_map50": sum(point[split]["candidate"] - point[split]["fp16"] for split in DOMAINS) / len(DOMAINS),
        "bootstrap_95ci_macro_domain_delta_map50": [percentile(macro_values, 0.025), percentile(macro_values, 0.975)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
