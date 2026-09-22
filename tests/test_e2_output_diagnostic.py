import hashlib
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from edge_readiness.e2_output_diagnostic import (
    OUTPUT_BYTES,
    OUTPUT_ELEMENTS,
    class_aware_nms,
    compare_postprocess,
    decode_candidates,
    load_tensor,
    raw_strata,
    audit_logs,
    verify_pinned_manifest,
)
from edge_readiness.e2_output_compare import ComparisonPolicy


def tensor_bytes(values):
    return struct.pack("<{}f".format(OUTPUT_ELEMENTS), *values)


def synthetic(value=0.0):
    return [float(value)] * OUTPUT_ELEMENTS


class E2OutputDiagnosticTests(unittest.TestCase):
    def test_binary_shape_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "output.bin"
            path.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "TENSOR_SIZE_MISMATCH"):
                load_tensor(path, hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_bytes(tensor_bytes(synthetic()))
            with self.assertRaisesRegex(ValueError, "TENSOR_HASH_MISMATCH"):
                load_tensor(path, "0" * 64)
            self.assertEqual(len(load_tensor(path, hashlib.sha256(path.read_bytes()).hexdigest())), OUTPUT_ELEMENTS)
            self.assertEqual(path.stat().st_size, OUTPUT_BYTES)

            values = synthetic()
            values[0] = float("nan")
            path.write_bytes(tensor_bytes(values))
            with self.assertRaisesRegex(ValueError, "TENSOR_NONFINITE"):
                load_tensor(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_wrong_source_manifest_and_retained_log_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SOURCE_MANIFEST_HASH_MISMATCH"):
                verify_pinned_manifest(manifest, "0" * 64, "source")
            logs = root / "logs"
            logs.mkdir()
            (logs / "build_events.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "RETAINED_LOG_HASH_MISMATCH"):
                audit_logs(logs)

    def test_raw_replay_and_domain_strata_preserve_score_tolerance_semantics(self):
        reference = synthetic()
        target = synthetic()
        target[10] = 0.0005
        target[4 * 8400 + 10] = 0.101
        result = raw_strata(reference, target, ComparisonPolicy())
        self.assertIn("source_onnx_max_score_ge_0.001", result["strata"])
        self.assertEqual(result["threshold_crossings"]["0.001"]["target_only"], 1)
        self.assertIn("equality is not required", result["score_semantics"])

    def test_channel_anchor_association_and_threshold_crossing(self):
        reference = synthetic()
        target = synthetic()
        reference[4 * 8400 + 5] = 0.30
        target[5 * 8400 + 5] = 0.30
        result = raw_strata(reference, target)
        self.assertEqual(result["class_argmax_changes_all_anchors"], 1)
        self.assertEqual(result["threshold_crossings"]["0.25"]["both_selected"], 1)
        self.assertEqual(result["threshold_crossings"]["0.25"]["source_only"], 0)
        self.assertEqual(result["threshold_crossings"]["0.25"]["target_only"], 0)

    def test_nms_is_class_aware_and_does_not_mutate_input(self):
        values = synthetic()
        for channel, coordinate in enumerate((10.0, 10.0, 8.0, 8.0)):
            values[channel * 8400 + 1] = coordinate
            values[channel * 8400 + 2] = coordinate
            values[channel * 8400 + 3] = coordinate
        values[4 * 8400 + 1] = 0.9
        values[4 * 8400 + 2] = 0.8
        values[5 * 8400 + 3] = 0.7
        before = list(values)
        detections = decode_candidates(values)
        kept = class_aware_nms(detections, iou_threshold=0.7)
        self.assertEqual(len(detections), 3)
        self.assertEqual(len(kept), 2)
        self.assertEqual({item.class_id for item in kept}, {0, 1})
        self.assertEqual(values, before)

    def test_nms_tie_break_and_postprocess_unmatched(self):
        values = synthetic()
        for anchor, score in ((0, 0.9), (1, 0.9)):
            values[anchor] = 10.0
            values[8400 + anchor] = 10.0
            values[16800 + anchor] = 10.0
            values[4 * 8400 + anchor] = score
        detections = decode_candidates(values)
        kept = class_aware_nms(detections)
        self.assertEqual(kept[0].anchor, 0)
        changed = list(values)
        changed[0] = 30.0
        result = compare_postprocess(values, changed)
        self.assertGreaterEqual(result["detection_matching"]["right_unmatched"], 0)
        self.assertIn("supplementary_one_to_one_matches", result["detection_matching"])

    def test_nms_boundary_iou_and_max_det(self):
        values = synthetic()
        for anchor, score, x in ((0, 0.9, 10.0), (1, 0.8, 10.5), (2, 0.7, 30.0)):
            values[anchor] = x
            values[8400 + anchor] = 10.0
            values[2 * 8400 + anchor] = 8.0
            values[3 * 8400 + anchor] = 8.0
            values[4 * 8400 + anchor] = score
        self.assertEqual(len(class_aware_nms(decode_candidates(values), iou_threshold=0.7, max_detections=1)), 1)
        self.assertEqual(len(class_aware_nms(decode_candidates(values), iou_threshold=0.7, max_detections=3)), 2)
        self.assertEqual(len(class_aware_nms(decode_candidates(values), iou_threshold=1.0, max_detections=3)), 3)


if __name__ == "__main__":
    unittest.main()
