#!/usr/bin/env python3
"""CPU-only G0 audit. Never loads weights or alters historical evaluation files."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

NAMES = ("prohibitory", "mandatory", "warning")
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VERSION = "cctsdb_measurement_audit_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lf_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def size_bin(area):
    for name, upper in zip(("xs", "s", "m", "l"), (210, 400, 1000, 2000)):
        if area <= upper:
            return name
    return "xl"


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def valid_box(box):
    return len(box) == 4 and all(math.isfinite(v) for v in box) and box[2] > box[0] and box[3] > box[1]


def load_records(payload):
    records = {}
    stems = set()
    for record in payload["records"]:
        name = record["image"]
        if not isinstance(name, str) or "/" in name or "\\" in name or Path(name).suffix.lower() not in SUFFIXES:
            raise ValueError(f"Invalid image basename: {name!r}")
        if name in records or Path(name).stem in stems:
            raise ValueError(f"Duplicate prediction ID: {name}")
        shape = record["orig_shape"]
        if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError(f"Invalid orig_shape: {name}")
        if len({len(record[k]) for k in ("xyxy", "confidence", "class_id")}) != 1:
            raise ValueError(f"Prediction array length mismatch: {name}")
        for box, score, cls in zip(record["xyxy"], record["confidence"], record["class_id"]):
            # Clipping at image borders may collapse a detection to zero area.
            # Retain it for faithful metric replay, and report it separately.
            ordered_box = len(box) == 4 and all(math.isfinite(v) for v in box) and box[2] >= box[0] and box[3] >= box[1]
            if not ordered_box or not math.isfinite(score) or not 0 <= score <= 1 or type(cls) is not int or cls not in range(3):
                raise ValueError(f"Invalid prediction: {name}")
        records[name] = record
        stems.add(Path(name).stem)
    if not records:
        raise ValueError("No prediction records")
    return records


def load_labels(path, shape):
    height, width = shape
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(v) for v in line.split()]
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Invalid label row: {path}")
        cls, x, y, w, h = values
        if cls not in (0, 1, 2) or not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"Invalid normalized label: {path}")
        rows.append((int(cls), [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]))
    return rows


def load_xml(path, records):
    """Only parse requested member IDs; do not use misleading XML filename fields."""
    expected = {Path(name).stem: name for name in records}
    output = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            stem = Path(member).stem
            if not member.lower().endswith(".xml") or stem not in expected:
                continue
            name = expected[stem]
            if name in output:
                raise ValueError(f"Duplicate XML member stem: {stem}")
            root = ET.fromstring(archive.read(member))
            rows = []
            for obj in root.findall("object"):
                cls_name = (obj.findtext("name") or "").strip().lower()
                if cls_name not in NAMES:
                    raise ValueError(f"Unknown XML class: {member}: {cls_name}")
                box = [float(obj.findtext(f"bndbox/{k}", "nan")) for k in ("xmin", "ymin", "xmax", "ymax")]
                if not valid_box(box):
                    raise ValueError(f"Invalid XML box: {member}")
                rows.append((NAMES.index(cls_name), box))
            output[name] = {"rows": rows, "shape": [int(root.findtext("size/height", "0")), int(root.findtext("size/width", "0"))]}
    missing = sorted(set(records) - set(output))
    if missing:
        raise ValueError(f"Missing {len(missing)} XML IDs; first={missing[:5]}")
    return output


def match_rows(left, right, tolerance):
    """Bipartite matching within coordinate tolerance, invariant to annotation order."""
    if len(left) != len(right):
        return None
    candidates = [[j for j, (cls2, b2) in enumerate(right) if cls == cls2 and max(abs(a-b) for a, b in zip(box, b2)) <= tolerance] for cls, box in left]
    assigned = {}

    def augment(i, seen):
        for j in candidates[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    if not all(augment(i, set()) for i in range(len(left))):
        return None
    return [(i, j) for j, i in assigned.items()]


def replay_ultralytics(records, truth, expected_version):
    """Reuse installed validator/AP code on CPU; no model, export or prediction call."""
    import numpy as np
    import torch
    import ultralytics
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import ap_per_class, box_iou

    if ultralytics.__version__ != expected_version:
        raise ValueError(f"Replay requires recorded ultralytics={expected_version}, installed={ultralytics.__version__}; do not upgrade the server environment")
    proxy = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10, device="cpu"))
    tps, scores, classes, targets = [], [], [], []
    # Preserve input order: confidence ties can depend on ordering.
    for name, record in records.items():
        gt = truth[name]
        gb = torch.tensor([b for _, b in gt], dtype=torch.float32).reshape(-1, 4)
        gc = torch.tensor([c for c, _ in gt], dtype=torch.float32)
        pb = torch.tensor(record["xyxy"], dtype=torch.float32).reshape(-1, 4)
        pc = torch.tensor(record["class_id"], dtype=torch.float32)
        correct = BaseValidator.match_predictions(proxy, pc, gc, box_iou(gb, pb))
        tps.append(correct.cpu().numpy())
        scores.extend(record["confidence"])
        classes.extend(record["class_id"])
        targets.extend(c for c, _ in gt)
    values = ap_per_class(np.concatenate(tps), np.asarray(scores), np.asarray(classes), np.asarray(targets), plot=False)
    ap, ids = values[5], values[6]
    return {"map50": float(ap[:, 0].mean()), "map50_95": float(ap.mean()),
            "per_class_ap50": {NAMES[int(c)]: float(row[0]) for c, row in zip(ids, ap)},
            "ultralytics": ultralytics.__version__, "torch": torch.__version__, "numpy": np.__version__,
            "device": "cpu", "matching_source_sha256": hashlib.sha256(inspect.getsource(BaseValidator.match_predictions).encode()).hexdigest(),
            "ap_source_sha256": hashlib.sha256(inspect.getsource(ap_per_class).encode()).hexdigest()}


def audit(prediction, evaluation, split_dir, xml_path=None, replay=False, coordinate_tolerance=0.05, ap_tolerance=0.0001, allow_lf=False):
    from PIL import Image

    payload = json.loads(prediction.read_text(encoding="utf-8"))
    summary = json.loads(evaluation.read_text(encoding="utf-8"))
    records = load_records(payload)
    for key in ("model_sha256", "data_sha256"):
        if not payload.get(key) or payload[key] != summary.get(key):
            raise ValueError(f"Prediction/evaluation {key} mismatch")
    declared_hash = summary.get("per_image_predictions", {}).get("sha256")
    hash_mode = "exact_bytes" if declared_hash == sha256(prediction) else "LF_normalized_checkout"
    if declared_hash != sha256(prediction) and not (allow_lf and declared_hash == lf_sha256(prediction)):
        raise ValueError("Prediction file hash does not match evaluation")
    images = {p.name: p for p in (split_dir / "images").iterdir() if p.suffix.lower() in SUFFIXES}
    label_paths = {p.stem: p for p in (split_dir / "labels").glob("*.txt")}
    stems = {Path(name).stem for name in records}
    if set(images) != set(records) or len(stems) != len(images) or set(label_paths) != stems:
        raise ValueError("Prediction/image/label membership mismatch; exact split coverage required")
    truth = {}
    label_hashes = {}
    image_hashes = {}
    for name, record in records.items():
        with Image.open(images[name]) as im:
            if [im.height, im.width] != record["orig_shape"]:
                raise ValueError(f"Prediction/image dimension mismatch: {name}")
        label = label_paths[Path(name).stem]
        truth[name] = load_labels(label, record["orig_shape"])
        label_hashes[label.name] = sha256(label)
        image_hashes[name] = sha256(images[name])
    counts = Counter(size_bin(area(box)) for rows in truth.values() for _, box in rows)
    degenerate = [{"image": name, "box": b, "confidence": s} for name, r in records.items() for b, s in zip(r["xyxy"], r["confidence"]) if area(b) == 0]
    result = {"version": VERSION, "prediction_sha256": sha256(prediction), "evaluation_sha256": sha256(evaluation),
              "prediction_declared_sha256": declared_hash, "prediction_hash_match": hash_mode,
              "model_sha256": payload["model_sha256"], "data_sha256": payload["data_sha256"],
              "images": len(records), "instances": sum(len(v) for v in truth.values()),
              "labels_inventory_sha256": hashlib.sha256(json.dumps(label_hashes, sort_keys=True).encode()).hexdigest(),
              "images_inventory_sha256": hashlib.sha256(json.dumps(image_hashes, sort_keys=True).encode()).hexdigest(),
              "duplicate_image_content_count": len(image_hashes)-len(set(image_hashes.values())),
              "zero_area_predictions_retained": {"count": len(degenerate), "samples": degenerate[:20]},
              "class_instances": dict(Counter(NAMES[c] for rows in truth.values() for c, _ in rows)),
              "yolo_continuous_area_counts": dict(counts), "review_reasons": [],
              "coordinate_tolerance_px": coordinate_tolerance, "ap_tolerance_absolute": ap_tolerance}
    if degenerate:
        result["review_reasons"].append("Zero-area clipped predictions retained in replay; verify postprocessing convention")
    if xml_path:
        xml = load_xml(xml_path, records)
        mismatches, boundary_changes, bounds = [], [], []
        max_error = 0.0
        xml_counts = Counter()
        for name, entry in xml.items():
            xml_counts.update(size_bin(area(b)) for _, b in entry["rows"])
            matches = match_rows(truth[name], entry["rows"], coordinate_tolerance)
            if matches is None or entry["shape"] != records[name]["orig_shape"]:
                mismatches.append({"image": name, "yolo": truth[name], "xml": entry})
                continue
            for i, j in matches:
                b1, b2 = truth[name][i][1], entry["rows"][j][1]
                max_error = max(max_error, max(abs(a-b) for a, b in zip(b1, b2)))
                if size_bin(area(b1)) != size_bin(area(b2)):
                    boundary_changes.append({"image": name, "xml_box": b2, "yolo_area": area(b1), "xml_area": area(b2), "yolo_bin": size_bin(area(b1)), "xml_bin": size_bin(area(b2))})
                h, w = entry["shape"]
                if b2[0] < 0 or b2[1] < 0 or b2[2] > w or b2[3] > h:
                    bounds.append({"image": name, "box": b2, "shape": [h, w]})
        result["xml_alignment"] = {"path": str(xml_path.resolve()), "sha256": sha256(xml_path), "mismatched_images": len(mismatches), "mismatches": mismatches,
            "max_matched_coordinate_error_px": max_error, "xml_area_counts": dict(xml_counts),
            "yolo_vs_xml_bin_changes": boundary_changes, "out_of_image_xml_boxes": bounds}
        if mismatches:
            result["review_reasons"].append("XML/YOLO geometry, class or dimension mismatch")
        if boundary_changes:
            result["review_reasons"].append("YOLO rounding changes size assignment; do not silently mix area conventions")
        if bounds:
            result["review_reasons"].append("XML coordinates outside image require convention review")
    else:
        result["review_reasons"].append("XML alignment not checked: archive not supplied")
    if replay:
        reproduced = replay_ultralytics(records, truth, summary["environment"]["ultralytics"])
        deltas = {k: reproduced[k] - summary["metrics"][k] for k in ("map50", "map50_95")}
        result["replay"] = {"metrics": reproduced, "stored_val_metrics": summary["metrics"], "delta_replay_minus_val": deltas,
                            "note": "Same installed matching/AP implementation; original-image coordinates. Differences can also involve preprocessing, label scaling, ties or prediction path; this is not causal attribution."}
        if any(abs(v) > ap_tolerance for v in deltas.values()):
            result["review_reasons"].append("Saved predict() records do not reproduce val() AP within tolerance")
    else:
        result["review_reasons"].append("Ultralytics CPU AP replay not requested")
    result["status"] = "review_required" if result["review_reasons"] else "pass"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--evaluations-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--xml", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--replay-ultralytics", action="store_true")
    parser.add_argument("--label", help="Audit just this evaluation filename stem")
    parser.add_argument("--allow-lf-normalization", action="store_true", help="Windows checkout only: allow exact match after CRLF-to-LF conversion; recorded explicitly, never changes files")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Output directory already exists; use a new audit version (no overwrite)")
    evaluations = sorted(args.evaluations_dir.glob("*.json"))
    if args.label:
        evaluations = [p for p in evaluations if p.stem == args.label]
    predictions = sorted(args.predictions_dir.glob("*.json"))
    if not evaluations or not predictions:
        parser.error("No evaluation or prediction JSON files found")
    if args.xml and not zipfile.is_zipfile(args.xml):
        parser.error("XML archive is absent or incomplete; do not use a .part download")
    by_hash = {sha256(p): p for p in predictions}
    if args.allow_lf_normalization:
        by_hash.update({lf_sha256(p): p for p in predictions})
    args.out_dir.mkdir(parents=True)
    reports = []
    for ev in evaluations:
        try:
            summary = json.loads(ev.read_text(encoding="utf-8"))
            if "metrics" not in summary:
                continue
            phash = summary.get("per_image_predictions", {}).get("sha256")
            if phash not in by_hash:
                raise ValueError("No saved prediction with the evaluation's declared hash")
            print(f"AUDIT: {ev.stem}", flush=True)
            report = audit(by_hash[phash], ev, args.split_dir, args.xml, args.replay_ultralytics, allow_lf=args.allow_lf_normalization)
        except Exception as exc:
            report = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        report["evaluation"] = str(ev.resolve())
        (args.out_dir / ev.name).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        reports.append({"label": ev.stem, "status": report["status"], "review_reasons": report.get("review_reasons", []), "error": report.get("error")})
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    result = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit.stdout.strip() or None,
              "script_sha256": sha256(Path(__file__)), "python": platform.python_version(), "reports": reports,
              "scope": "Read-only measurement audit, no inference/training/export or policy selection",
              "status": "pass" if reports and all(r["status"] == "pass" for r in reports) else "review_required"}
    (args.out_dir / "audit_summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
