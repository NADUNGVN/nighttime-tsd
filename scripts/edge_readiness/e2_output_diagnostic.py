#!/usr/bin/env python3
"""CPU-only diagnosis of already-saved E2 output tensors.

This module never invokes TensorRT/CUDA and never writes or mutates tensor
buffers.  It replays the frozen raw comparator, then applies one explicit
YOLO11n-style application postprocess to the saved tensors for diagnostic
comparison only.  It is not an accuracy evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from .e2_output_compare import ComparisonPolicy, DomainTolerance, compare_output0
except ImportError:  # pragma: no cover - direct script execution on E2
    from e2_output_compare import ComparisonPolicy, DomainTolerance, compare_output0

OUTPUT_SHAPE = (1, 7, 8400)
OUTPUT_ELEMENTS = 7 * 8400
OUTPUT_BYTES = OUTPUT_ELEMENTS * 4
CHANNEL_NAMES = ("x", "y", "w", "h", "score_0", "score_1", "score_2")
CLASS_ORDER = ("prohibitory", "mandatory", "warning")
FIXTURE_IDS = ("00006", "00009", "00028")
POSTPROCESS_CONFIDENCE = 0.001
POSTPROCESS_IOU = 0.7
POSTPROCESS_MAX_DETECTIONS = 300

STRICT_POLICY = ComparisonPolicy(
    source_compute_mode="fp32_reference",
    target_compute_mode="onnx_fp32",
    box_tolerance=DomainTolerance(absolute=1e-5, relative=1e-4),
    score_tolerance=DomainTolerance(absolute=1e-5, relative=1e-4),
)
TARGET_POLICY = ComparisonPolicy()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tensor(path: Path, expected_sha256: str) -> Tuple[float, ...]:
    """Load exactly one little-endian float32 output after hash/size checks."""
    if path.stat().st_size != OUTPUT_BYTES:
        raise ValueError("TENSOR_SIZE_MISMATCH: expected {} bytes".format(OUTPUT_BYTES))
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError("TENSOR_HASH_MISMATCH: {} != {}".format(observed, expected_sha256))
    payload = path.read_bytes()
    values = struct.unpack("<{}f".format(OUTPUT_ELEMENTS), payload)
    if len(values) != OUTPUT_ELEMENTS:
        raise ValueError("TENSOR_ELEMENT_COUNT_MISMATCH")
    return values


def _quantile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _box(values: Sequence[float], anchor: int) -> Tuple[float, float, float, float]:
    return tuple(float(values[channel * 8400 + anchor]) for channel in range(4))  # type: ignore[return-value]


def _scores(values: Sequence[float], anchor: int) -> Tuple[float, float, float]:
    return tuple(float(values[channel * 8400 + anchor]) for channel in range(4, 7))  # type: ignore[return-value]


def _xyxy(xywh: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, width, height = xywh
    return (x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0)


def box_iou(left: Tuple[float, float, float, float], right: Tuple[float, float, float, float]) -> float:
    intersection_left = max(left[0], right[0])
    intersection_top = max(left[1], right[1])
    intersection_right = min(left[2], right[2])
    intersection_bottom = min(left[3], right[3])
    intersection = max(0.0, intersection_right - intersection_left) * max(0.0, intersection_bottom - intersection_top)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass(frozen=True)
class Detection:
    anchor: int
    class_id: int
    score: float
    box: Tuple[float, float, float, float]


def decode_candidates(values: Sequence[float], confidence: float = POSTPROCESS_CONFIDENCE) -> Tuple[Detection, ...]:
    if len(values) != OUTPUT_ELEMENTS:
        raise ValueError("OUTPUT_SHAPE_MISMATCH")
    candidates: List[Detection] = []
    for anchor in range(8400):
        scores = _scores(values, anchor)
        class_id = max(range(3), key=lambda index: (scores[index], -index))
        score = scores[class_id]
        if math.isfinite(score) and score >= confidence:
            candidates.append(Detection(anchor, class_id, score, _xyxy(_box(values, anchor))))
    return tuple(candidates)


def class_aware_nms(candidates: Sequence[Detection], iou_threshold: float = POSTPROCESS_IOU, max_detections: int = POSTPROCESS_MAX_DETECTIONS) -> Tuple[Detection, ...]:
    if iou_threshold < 0.0 or iou_threshold > 1.0 or max_detections < 0:
        raise ValueError("NMS_POLICY_INVALID")
    ordered = sorted(candidates, key=lambda item: (-item.score, item.class_id, item.anchor))
    kept: List[Detection] = []
    for candidate in ordered:
        if all(candidate.class_id != previous.class_id or box_iou(candidate.box, previous.box) <= iou_threshold for previous in kept):
            kept.append(candidate)
            if len(kept) == max_detections:
                break
    return tuple(kept)


def _domain_summary(errors: Sequence[float], mismatches: int) -> Dict[str, Any]:
    finite = [float(value) for value in errors if math.isfinite(value)]
    return {
        "elements": len(errors),
        "mismatches": int(mismatches),
        "max_absolute_error": max(finite) if finite else None,
        "absolute_error_quantiles": {"q50": _quantile(finite, 0.50), "q90": _quantile(finite, 0.90), "q95": _quantile(finite, 0.95), "q99": _quantile(finite, 0.99), "q100": _quantile(finite, 1.00)},
    }


def raw_strata(reference: Sequence[float], target: Sequence[float], policy: ComparisonPolicy = TARGET_POLICY) -> Dict[str, Any]:
    """Summarize fixed anchor strata without changing the acceptance policy."""
    all_anchors = tuple(range(8400))
    source_max = tuple(max(_scores(reference, anchor)) for anchor in all_anchors)
    target_max = tuple(max(_scores(target, anchor)) for anchor in all_anchors)
    strata = {
        "all_anchors": all_anchors,
        "source_onnx_max_score_ge_0.001": tuple(anchor for anchor in all_anchors if source_max[anchor] >= 0.001),
        "source_onnx_max_score_ge_0.25": tuple(anchor for anchor in all_anchors if source_max[anchor] >= 0.25),
        "union_source_target_max_score_ge_0.001": tuple(anchor for anchor in all_anchors if source_max[anchor] >= 0.001 or target_max[anchor] >= 0.001),
        "union_source_target_max_score_ge_0.25": tuple(anchor for anchor in all_anchors if source_max[anchor] >= 0.25 or target_max[anchor] >= 0.25),
    }
    output: Dict[str, Any] = {}
    for name, anchors in strata.items():
        domains: Dict[str, Any] = {}
        for domain, channels in (("boxes", range(4)), ("scores", range(4, 7))):
            errors = [abs(float(reference[channel * 8400 + anchor]) - float(target[channel * 8400 + anchor])) for anchor in anchors for channel in channels]
            mismatches = sum(1 for anchor in anchors for channel in channels if abs(float(reference[channel * 8400 + anchor]) - float(target[channel * 8400 + anchor])) > (policy.box_tolerance.absolute + policy.box_tolerance.relative * abs(float(reference[channel * 8400 + anchor])) if domain == "boxes" else policy.score_tolerance.absolute + policy.score_tolerance.relative * abs(float(reference[channel * 8400 + anchor]))))
            domains[domain] = _domain_summary(errors, mismatches)
            if domain == "boxes":
                domains[domain]["by_channel"] = {
                    CHANNEL_NAMES[channel]: _domain_summary(
                        [abs(float(reference[channel * 8400 + anchor]) - float(target[channel * 8400 + anchor])) for anchor in anchors],
                        sum(1 for anchor in anchors if abs(float(reference[channel * 8400 + anchor]) - float(target[channel * 8400 + anchor])) > policy.box_tolerance.absolute + policy.box_tolerance.relative * abs(float(reference[channel * 8400 + anchor]))),
                    ) for channel in range(4)
                }
        output[name] = {"anchor_count": len(anchors), "domains": domains}
    threshold_crossings: Dict[str, Any] = {}
    for threshold in (0.001, 0.25):
        source_selected = {anchor for anchor in all_anchors if source_max[anchor] >= threshold}
        target_selected = {anchor for anchor in all_anchors if target_max[anchor] >= threshold}
        union = source_selected | target_selected
        threshold_crossings[str(threshold)] = {
            "source_selected": len(source_selected),
            "target_selected": len(target_selected),
            "both_selected": len(source_selected & target_selected),
            "source_only": len(source_selected - target_selected),
            "target_only": len(target_selected - source_selected),
            "union": len(union),
        }
    class_changes = sum(max(range(3), key=lambda index: (_scores(reference, anchor)[index], -index)) != max(range(3), key=lambda index: (_scores(target, anchor)[index], -index)) for anchor in all_anchors)
    return {"strata": {name: {key: value for key, value in data.items() if key != "_anchors"} for name, data in ((name, {"anchor_count": len(anchors), "domains": output[name]["domains"]}) for name, anchors in strata.items())}, "threshold_crossings": threshold_crossings, "class_argmax_changes_all_anchors": class_changes, "score_semantics": "score values are compared under tolerance; equality is not required"}


def detection_summary(values: Sequence[float]) -> Dict[str, Any]:
    raw = decode_candidates(values)
    kept = class_aware_nms(raw)
    return {
        "policy": {"confidence": POSTPROCESS_CONFIDENCE, "iou": POSTPROCESS_IOU, "max_detections": POSTPROCESS_MAX_DETECTIONS, "coordinate_system": "640x640 letterboxed input pixels", "class_order": list(CLASS_ORDER), "nms": "class-aware single-label greedy NMS; sort (-score,class_id,anchor); suppress same-class IoU > threshold"},
        "raw_candidate_count": len(raw),
        "nms_kept_count": len(kept),
        "raw_class_counts": dict(sorted(Counter(item.class_id for item in raw).items())),
        "nms_class_counts": dict(sorted(Counter(item.class_id for item in kept).items())),
        "top_nms": [{"anchor": item.anchor, "class_id": item.class_id, "score": item.score, "box": list(item.box)} for item in kept[:20]],
        "_detections": kept,
    }


def _match_detections(left: Sequence[Detection], right: Sequence[Detection]) -> Dict[str, Any]:
    left_by_lineage = {(item.anchor, item.class_id): item for item in left}
    right_by_lineage = {(item.anchor, item.class_id): item for item in right}
    exact_keys = sorted(set(left_by_lineage) & set(right_by_lineage))
    exact_rows = [
        {
            "anchor": anchor,
            "class_id": class_id,
            "score_abs_delta": abs(right_by_lineage[(anchor, class_id)].score - left_by_lineage[(anchor, class_id)].score),
            "box_max_abs_delta": max(abs(a - b) for a, b in zip(left_by_lineage[(anchor, class_id)].box, right_by_lineage[(anchor, class_id)].box)),
            "box_iou": box_iou(left_by_lineage[(anchor, class_id)].box, right_by_lineage[(anchor, class_id)].box),
        }
        for anchor, class_id in exact_keys
    ]
    remaining_left = [item for item in left if (item.anchor, item.class_id) not in exact_keys]
    remaining_right = [item for item in right if (item.anchor, item.class_id) not in exact_keys]
    used: Set[int] = set()
    supplementary: List[Dict[str, Any]] = []
    for source in sorted(remaining_left, key=lambda item: (-item.score, item.class_id, item.anchor)):
        choices = [(box_iou(source.box, candidate.box), candidate.score, -candidate.anchor, index, candidate) for index, candidate in enumerate(remaining_right) if index not in used and candidate.class_id == source.class_id]
        if not choices:
            continue
        iou, _, _, index, candidate = max(choices)
        used.add(index)
        supplementary.append({"left_anchor": source.anchor, "right_anchor": candidate.anchor, "class_id": source.class_id, "iou": iou, "score_delta": candidate.score - source.score, "box_max_abs_delta": max(abs(a - b) for a, b in zip(source.box, candidate.box))})
    return {
        "same_origin_lineage_matches": len(exact_keys),
        "same_origin_lineage_union": len(set(left_by_lineage) | set(right_by_lineage)),
        "same_origin_score_abs_delta_quantiles": {key: _quantile([item["score_abs_delta"] for item in exact_rows], value) for key, value in (("q50", 0.5), ("q90", 0.9), ("q100", 1.0))},
        "same_origin_box_max_abs_delta_quantiles": {key: _quantile([item["box_max_abs_delta"] for item in exact_rows], value) for key, value in (("q50", 0.5), ("q90", 0.9), ("q100", 1.0))},
        "same_origin_box_iou_quantiles": {key: _quantile([item["box_iou"] for item in exact_rows], value) for key, value in (("q50", 0.5), ("q90", 0.9), ("q100", 1.0))},
        "same_origin_examples": exact_rows[:20],
        "supplementary_one_to_one_matches": len(supplementary),
        "left_unmatched": len(left) - len(exact_keys) - len(supplementary),
        "right_unmatched": len(right) - len(exact_keys) - len(supplementary),
        "supplementary_iou_quantiles": {key: _quantile([item["iou"] for item in supplementary], value) for key, value in (("q50", 0.5), ("q90", 0.9), ("q100", 1.0))},
        "supplementary": supplementary[:20],
    }


def compare_postprocess(reference: Sequence[float], target: Sequence[float]) -> Dict[str, Any]:
    left = detection_summary(reference)
    right = detection_summary(target)
    left_detections = left.pop("_detections")
    right_detections = right.pop("_detections")
    return {"reference": left, "target": right, "detection_matching": _match_detections(left_detections, right_detections), "interpretation": "postprocess comparison is descriptive only; no ground truth, AP, recall or safety claim"}


def _file_record(path: Path, label: str) -> Dict[str, Any]:
    return {"label": label, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def audit_logs(log_root: Path) -> Dict[str, Any]:
    records = []
    for relative in ("build_events.jsonl", "build_result.json", "build_stdout.log", "build_stderr.log", "inference_events.jsonl", "inference_result.json", "inference_stdout.log", "inference_stderr.log"):
        path = log_root / relative
        if not path.is_file():
            records.append({"label": relative, "status": "missing"})
            continue
        record = _file_record(path, relative)
        if relative.endswith("events.jsonl"):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            counts = Counter(row.get("event") for row in rows)
            record.update({"event_count": len(rows), "event_counts": dict(sorted(counts.items())), "first_utc": rows[0].get("utc_timestamp") if rows else None, "last_utc": rows[-1].get("utc_timestamp") if rows else None})
        records.append(record)
    return {"records": records, "private_content_policy": "full logs remain outside public Git; only size/hash/event metadata is published"}


def analyze_fixture(image_id: str, expected: Mapping[str, str], source_root: Path, target_root: Path) -> Dict[str, Any]:
    native = load_tensor(source_root / "private/native_reference/{}.bin".format(image_id), expected["native_sha256"])
    onnx = load_tensor(source_root / "private/onnx_reference/{}.bin".format(image_id), expected["onnx_sha256"])
    target = load_tensor(target_root / "{}.bin".format(image_id), expected["target_sha256"])
    source_result = compare_output0(native, onnx, STRICT_POLICY).as_dict()
    target_onnx = compare_output0(onnx, target, TARGET_POLICY).as_dict()
    target_native = compare_output0(native, target, TARGET_POLICY).as_dict()
    return {
        "image_id": image_id,
        "input_coordinate_contract": "xywh boxes in 640x640 letterboxed input pixels; output shape [1,7,8400]; little-endian float32",
        "verified_inputs": {"native": expected["native_sha256"], "source_onnx": expected["onnx_sha256"], "target": expected["target_sha256"]},
        "source_native_vs_onnx": source_result,
        "target_vs_source_onnx": target_onnx,
        "target_vs_source_native": target_native,
        "raw_strata_target_vs_source_onnx": raw_strata(onnx, target),
        "postprocess_target_vs_source_onnx": compare_postprocess(onnx, target),
        "postprocess_target_vs_source_native": compare_postprocess(native, target),
    }


def run_analysis(source_root: Path, target_root: Path, expected_by_fixture: Mapping[str, Mapping[str, str]], log_root: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=False)
    fixtures = {image_id: analyze_fixture(image_id, expected_by_fixture[image_id], source_root, target_root) for image_id in FIXTURE_IDS}
    analyzer_path = Path(__file__).resolve()
    test_path = analyzer_path.parents[2] / "tests/test_e2_output_diagnostic.py"
    payload = {
        "schema_version": "e2l1-output-diagnostic-v1",
        "status": "output_diagnostic_completed_review_required",
        "code_provenance": {"analyzer": {"path_label": "scripts/edge_readiness/e2_output_diagnostic.py", "sha256": sha256_file(analyzer_path)}, "tests": {"path_label": "tests/test_e2_output_diagnostic.py", "sha256": sha256_file(test_path)}},
        "execution_policy": {"cpu_only": True, "new_build": False, "new_inference": False, "benchmark": False, "postprocess": {"confidence": POSTPROCESS_CONFIDENCE, "iou": POSTPROCESS_IOU, "max_detections": POSTPROCESS_MAX_DETECTIONS}},
        "fixtures": fixtures,
        "logs": audit_logs(log_root),
        "limitations": ["Saved-target replay is not repeatability testing.", "Raw box mismatch does not establish task-level accuracy degradation.", "Three selected train fixtures provide no AP, dev accuracy, recall, safety or deployment claim.", "Postprocess uses frozen 640x640 input coordinates; no original-image scale-back was applied."],
    }
    (out_dir / "analysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    public_payload = {"schema_version": payload["schema_version"], "status": payload["status"], "fixtures": list(FIXTURE_IDS), "public_artifacts": ["analysis.json", "inventory.json", "report.md", "index.json"]}
    inventory = {"schema_version": payload["schema_version"], "status": payload["status"], "code_provenance": payload["code_provenance"], "target_outputs": [{"image_id": image_id, "bytes": OUTPUT_BYTES, "sha256": expected_by_fixture[image_id]["target_sha256"]} for image_id in FIXTURE_IDS], "logs": payload["logs"], "private_policy": "target tensors, engine and full logs are not published"}
    (out_dir / "inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "index.json").write_text(json.dumps(public_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    lines = ["# E2L1-023 saved-output diagnosis", "", "Status: `output_diagnostic_completed_review_required`.", "", "CPU-only replay of the saved target tensors used raw output0 comparison and fixed postprocess conf=0.001, IoU=0.7, max_det=300. The raw target FAIL is in box elements; score values are assessed by tolerance, not exact equality. This report does not claim accuracy impact, AP, recall, safety or deployment readiness.", ""]
    lines.append("| Fixture | Source export | TRT box/score mismatches | Selected anchors / box mismatches | NMS source -> target | Same-lineage box delta q100 / IoU q100 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for image_id in FIXTURE_IDS:
        row = fixtures[image_id]
        source = row["source_native_vs_onnx"]
        target = row["target_vs_source_onnx"]["domain_summary"]
        post = row["postprocess_target_vs_source_onnx"]
        selected = row["raw_strata_target_vs_source_onnx"]["strata"]["source_onnx_max_score_ge_0.001"]
        matching = post["detection_matching"]
        lines.append("| `{}` | {} | {}/{} | {}/{} | {} -> {} / {} -> {} | {:.6g} / {:.6g} |".format(image_id, source["status"], target["boxes"]["mismatches"], target["scores"]["mismatches"], selected["anchor_count"], selected["domains"]["boxes"]["mismatches"], post["reference"]["nms_kept_count"], post["target"]["nms_kept_count"], post["reference"]["raw_candidate_count"], post["target"]["raw_candidate_count"], matching["same_origin_box_max_abs_delta_quantiles"]["q100"] or 0.0, matching["same_origin_box_iou_quantiles"]["q100"] or 0.0))
    lines.extend(["", "Observed: the saved bytes replay the accepted raw comparison and permit deterministic CPU postprocess diagnostics. The score domain has zero policy violations in all three fixtures, and selected-anchor threshold membership is unchanged at 0.001 and 0.25; this is not exact-score equality. NMS counts and same-origin lineages are unchanged, while the retained box coordinates differ. Unknown: whether those coordinate changes alter final task detections or ground-truth accuracy under a valid evaluation protocol. Proposed next experiment: after scientific review, run one separately authorized FP16-reference/target postprocess comparison on a predeclared evaluation slice; do not relax the raw comparator or infer accuracy from this train-only trio.", ""])
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze saved E2 output tensors on CPU only")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = {image_id: {"native_sha256": sha256_file(args.source_root / "private/native_reference/{}.bin".format(image_id)), "onnx_sha256": sha256_file(args.source_root / "private/onnx_reference/{}.bin".format(image_id)), "target_sha256": manifest["target_outputs"][image_id]["sha256"]} for image_id in FIXTURE_IDS}
    payload = run_analysis(args.source_root, args.target_root, expected, args.log_root, args.out_dir)
    print(json.dumps({"status": payload["status"], "out_dir": str(args.out_dir.resolve()), "fixtures": list(FIXTURE_IDS)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
