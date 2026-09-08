#!/usr/bin/env python3
"""Evaluate CCTSDB full-test predictions by the official traffic-sign area bins.

This is an instance-level diagnostic.  It does not create image subsets, because
an image may contain signs from several size bins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from bootstrap_cctsdb_map50 import NAMES, average_precision, iou


SIZE_RULES = {
    "xs": "area <= 210 px^2",
    "s": "210 < area <= 400 px^2",
    "m": "400 < area <= 1000 px^2",
    "l": "1000 < area <= 2000 px^2",
    "xl": "area > 2000 px^2",
}
NAME_TO_CLASS = {name: index for index, name in enumerate(NAMES)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def size_bin(area: float) -> str:
    if area <= 210:
        return "xs"
    if area <= 400:
        return "s"
    if area <= 1000:
        return "m"
    if area <= 2000:
        return "l"
    return "xl"


def xml_ground_truth(path: Path) -> dict[str, dict[int, list[tuple[list[float], str]]]]:
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise FileNotFoundError("--xml must be the official CCTSDB xml.zip archive")
    output: dict[str, dict[int, list[tuple[list[float], str]]]] = {}
    with zipfile.ZipFile(path) as archive:
        members = [member for member in archive.namelist() if member.lower().endswith(".xml")]
        if len(members) != 1500:
            raise ValueError(f"Expected 1,500 XML annotations in {path}, found {len(members)}")
        for member in members:
            root = ET.fromstring(archive.read(member))
            filename = (root.findtext("filename") or f"{Path(member).stem}.jpg").strip()
            image = Path(filename).name
            entries = {index: [] for index in range(len(NAMES))}
            for object_node in root.findall("object"):
                name = (object_node.findtext("name") or "").strip().lower()
                if name not in NAME_TO_CLASS:
                    raise ValueError(f"Unknown CCTSDB class {name!r} in {member}")
                box = object_node.find("bndbox")
                if box is None:
                    raise ValueError(f"Missing bndbox in {member}")
                xyxy = [float(box.findtext(field, "")) for field in ("xmin", "ymin", "xmax", "ymax")]
                area = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])
                entries[NAME_TO_CLASS[name]].append((xyxy, size_bin(area)))
            if image in output:
                raise ValueError(f"Duplicate XML image annotation: {image}")
            output[image] = entries
    return output


def metrics_for_bin(records: dict[str, dict], truth: dict[str, dict[int, list[tuple[list[float], str]]]], target_bin: str) -> dict:
    per_class: dict[str, float] = {}
    instances = 0
    ignored_predictions = 0
    for cls, name in enumerate(NAMES):
        target_by_image = {image: [box for box, assigned in classes[cls] if assigned == target_bin] for image, classes in truth.items()}
        other_by_image = {image: [box for box, assigned in classes[cls] if assigned != target_bin] for image, classes in truth.items()}
        positives = sum(len(boxes) for boxes in target_by_image.values())
        instances += positives
        detections = []
        for image, record in records.items():
            for box, confidence, predicted_class in zip(record["xyxy"], record["confidence"], record["class_id"]):
                if predicted_class == cls:
                    detections.append((float(confidence), image, box))
        detections.sort(key=lambda item: item[0], reverse=True)
        used = {image: [False] * len(boxes) for image, boxes in target_by_image.items()}
        tp: list[int] = []
        fp: list[int] = []
        for _, image, box in detections:
            candidates = [(iou(box, candidate), index) for index, candidate in enumerate(target_by_image[image]) if not used[image][index]]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= 0.5:
                used[image][best_index] = True
                tp.append(1)
                fp.append(0)
            elif any(iou(box, non_target) >= 0.5 for non_target in other_by_image[image]):
                ignored_predictions += 1
            else:
                tp.append(0)
                fp.append(1)
        ap = average_precision(tp, fp, positives)
        if ap is not None:
            per_class[name] = ap
    return {
        "instances": instances,
        "map50": sum(per_class.values()) / len(per_class) if per_class else None,
        "per_class_ap50": per_class,
        "ignored_predictions_matching_out_of_bin_same_class": ignored_predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Instance-level official CCTSDB XS/S/M/L/XL AP50 from saved full-test predictions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True, help="Official CCTSDB xml.zip")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite size result: {args.out}")
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    records = {record["image"]: record for record in payload["records"]}
    truth = xml_ground_truth(args.xml)
    if set(records) != set(truth):
        missing, extra = sorted(set(truth) - set(records)), sorted(set(records) - set(truth))
        raise ValueError(f"Full-test prediction/XML mismatch: missing={len(missing)}, extra={len(extra)}")
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Official CCTSDB size-bin diagnostic from full positive test; no image-level size subset is constructed.",
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256(args.predictions),
        "official_xml": str(args.xml.resolve()),
        "official_xml_sha256": sha256(args.xml),
        "size_rules": SIZE_RULES,
        "matching": "AP50 by class. A prediction matching a same-class ground-truth sign outside the target bin is ignored, rather than counted as a false positive for the target bin.",
        "metrics": {bin_name: metrics_for_bin(records, truth, bin_name) for bin_name in SIZE_RULES},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
