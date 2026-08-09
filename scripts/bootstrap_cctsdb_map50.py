#!/usr/bin/env python3
"""Paired image-bootstrap confidence intervals for CCTSDB mAP50 from saved predictions."""
from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path


NAMES = ("prohibitory", "mandatory", "warning")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return 0.0 if union <= 0 else intersection / union


def average_precision(tp: list[int], fp: list[int], positives: int) -> float | None:
    if positives == 0:
        return None
    cumulative_tp, cumulative_fp = [], []
    running_tp = running_fp = 0
    for true_positive, false_positive in zip(tp, fp):
        running_tp += true_positive
        running_fp += false_positive
        cumulative_tp.append(running_tp)
        cumulative_fp.append(running_fp)
    recalls = [value / positives for value in cumulative_tp]
    precisions = [true_positive / max(true_positive + false_positive, 1) for true_positive, false_positive in zip(cumulative_tp, cumulative_fp)]
    return sum(max((precision for recall, precision in zip(recalls, precisions) if recall >= threshold), default=0.0) for threshold in (index / 100.0 for index in range(101))) / 101.0


def resolve_images(data: Path) -> Path:
    text = data.read_text(encoding="utf-8")
    root_match = re.search(r"(?m)^\s*path:\s*(.+?)\s*(?:#.*)?$", text)
    val_match = re.search(r"(?m)^\s*val:\s*(.+?)\s*(?:#.*)?$", text)
    if root_match is None or val_match is None:
        raise ValueError(f"Missing path or val in {data}")
    root = Path(root_match.group(1).strip().strip("\"'"))
    if not root.is_absolute():
        root = (data.parent / root).resolve()
    return root / val_match.group(1).strip().strip("\"'")


def load_ground_truth(images: Path, available: dict[str, dict]) -> dict[str, dict[int, list[list[float]]]]:
    labels = images.parent / "labels"
    ground_truth: dict[str, dict[int, list[list[float]]]] = {}
    for name, record in available.items():
        label = labels / f"{Path(name).stem}.txt"
        height, width = record["orig_shape"]
        values: dict[int, list[list[float]]] = {index: [] for index in range(len(NAMES))}
        if label.is_file():
            for raw in label.read_text(encoding="utf-8").splitlines():
                cls, xc, yc, box_w, box_h = (float(value) for value in raw.split())
                index = int(cls)
                values[index].append([(xc - box_w / 2.0) * width, (yc - box_h / 2.0) * height, (xc + box_w / 2.0) * width, (yc + box_h / 2.0) * height])
        ground_truth[name] = values
    return ground_truth


def map50(sampled: list[str], predictions: dict[str, dict], truth: dict[str, dict[int, list[list[float]]]]) -> float:
    aps: list[float] = []
    for cls in range(len(NAMES)):
        gt_by_occurrence: dict[int, list[list[float]]] = {}
        detections: list[tuple[float, int, list[float]]] = []
        for occurrence, name in enumerate(sampled):
            gt_by_occurrence[occurrence] = list(truth[name][cls])
            record = predictions[name]
            for box, score, predicted_class in zip(record["xyxy"], record["confidence"], record["class_id"]):
                if predicted_class == cls:
                    detections.append((float(score), occurrence, box))
        positives = sum(len(boxes) for boxes in gt_by_occurrence.values())
        detections.sort(key=lambda item: item[0], reverse=True)
        used = {occurrence: [False] * len(boxes) for occurrence, boxes in gt_by_occurrence.items()}
        tp: list[int] = []
        fp: list[int] = []
        for _, occurrence, box in detections:
            matches = [(iou(box, candidate), index) for index, candidate in enumerate(gt_by_occurrence[occurrence]) if not used[occurrence][index]]
            best_iou, best_index = max(matches, default=(0.0, -1))
            if best_iou >= 0.5:
                used[occurrence][best_index] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
        ap = average_precision(tp, fp, positives)
        if ap is not None:
            aps.append(ap)
    if not aps:
        raise ValueError("No annotated ground-truth classes in selected images")
    return sum(aps) / len(aps)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired bootstrap mAP50 and Delta mAP50 from IVC full-split prediction JSON files")
    parser.add_argument("--data", type=Path, required=True, help="Official CCTSDB split YAML, for example configs/cctsdb2021_test_night.yaml")
    parser.add_argument("--fp16-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.out.exists() or args.iterations <= 0:
        raise ValueError("Output must not exist and --iterations must be positive")
    fp16_payload = json.loads(args.fp16_predictions.read_text(encoding="utf-8"))
    candidate_payload = json.loads(args.candidate_predictions.read_text(encoding="utf-8"))
    fp16 = {record["image"]: record for record in fp16_payload["records"]}
    candidate = {record["image"]: record for record in candidate_payload["records"]}
    images = resolve_images(args.data)
    selected_names = sorted(path.name for path in images.iterdir() if path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png"})
    missing = [name for name in selected_names if name not in fp16 or name not in candidate]
    if missing:
        raise ValueError(f"Predictions do not cover {len(missing)} selected images; first={missing[0]}")
    truth = load_ground_truth(images, fp16)
    point_fp16 = map50(selected_names, fp16, truth)
    point_candidate = map50(selected_names, candidate, truth)
    rng = random.Random(args.seed)
    fp16_values: list[float] = []
    candidate_values: list[float] = []
    delta_values: list[float] = []
    for _ in range(args.iterations):
        sampled = [rng.choice(selected_names) for _ in selected_names]
        fp16_value = map50(sampled, fp16, truth)
        candidate_value = map50(sampled, candidate, truth)
        fp16_values.append(fp16_value)
        candidate_values.append(candidate_value)
        delta_values.append(candidate_value - fp16_value)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data.resolve()),
        "images": len(selected_names),
        "iterations": args.iterations,
        "seed": args.seed,
        "fp16_predictions": str(args.fp16_predictions.resolve()),
        "candidate_predictions": str(args.candidate_predictions.resolve()),
        "point_map50": {"fp16": point_fp16, "candidate": point_candidate, "delta_candidate_minus_fp16": point_candidate - point_fp16},
        "bootstrap_95ci": {
            "fp16_map50": [percentile(fp16_values, 0.025), percentile(fp16_values, 0.975)],
            "candidate_map50": [percentile(candidate_values, 0.025), percentile(candidate_values, 0.975)],
            "delta_candidate_minus_fp16": [percentile(delta_values, 0.025), percentile(delta_values, 0.975)],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
