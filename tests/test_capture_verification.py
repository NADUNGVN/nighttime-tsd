import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_cctsdb_capture import coco_size, rematch_native, size_ranges, validate_capture


def example(boxes, predicted=None, scores=None):
    predicted = boxes if predicted is None else predicted
    record = {"image": "a.jpg", "orig_shape": [100, 100], "xyxy": copy.deepcopy(predicted),
        "confidence": scores if scores is not None else [.9] * len(predicted), "class_id": [0] * len(predicted),
        "validator_input": {"target_xyxy": copy.deepcopy(boxes), "target_class_id": [0] * len(boxes),
            "prediction_xyxy": copy.deepcopy(predicted)},
        "validator_statistics": {"tp": [[True]*10 for _ in predicted]}}
    xml = {"a.jpg": {"rows": [(0, b) for b in boxes], "shape": [100,100]}}
    return record, xml


class CaptureVerificationTests(unittest.TestCase):
    def test_native_rematch_detects_corruption(self):
        record, _ = example([[0,0,10,10]])
        thresholds = np.linspace(.5,.95,10).tolist()
        self.assertEqual(rematch_native([record], thresholds)["changed_tp_decisions"], 0)
        record["validator_statistics"]["tp"][0][3] = False
        self.assertEqual(rematch_native([record], thresholds)["changed_tp_decisions"], 1)

    def test_native_empty_and_invalid_capture(self):
        record, _ = example([[0,0,10,10]], [])
        thresholds = np.linspace(.5,.95,10).tolist()
        payload = {"schema_version": 2, "capture_mode": "same_val_process_batch",
            "records": [record], "iou_thresholds": thresholds}
        validate_capture(payload)
        self.assertEqual(rematch_native([record], thresholds)["status"], "pass")
        record["validator_input"]["target_xyxy"][0][2] = -1
        with self.assertRaises(ValueError):
            validate_capture(payload)

    def test_boundaries_perfect_and_absent_class(self):
        areas = [210., 211., 400., 401., 1000., 1001., 2000., 2001.]
        records, xml = [], {}
        for i, a in enumerate(areas):
            r, x = example([[0,0,50,a/50]])
            r["image"] = f"{i}.jpg"
            records.append(r)
            xml[r["image"]] = x["a.jpg"]
            ranges = size_ranges()[1:]
            self.assertEqual(sum(lo <= a <= hi for lo,hi in ranges), 1)
        before = copy.deepcopy((records, xml))
        result = coco_size(records, xml)
        self.assertEqual([result["metrics"][k]["instances"] for k in ("xs","s","m","l","xl")], [1,2,2,2,1])
        for metrics in result["metrics"].values():
            self.assertAlmostEqual(metrics["map50"], 1.)
            self.assertAlmostEqual(metrics["map50_95"], 1.)
            self.assertIsNone(metrics["per_class"]["warning"]["ap50"])
        self.assertEqual((records, xml), before)

    def test_no_predictions_and_shape_mismatch(self):
        r, xml = example([[0,0,10,10]], [])
        result = coco_size([r], xml)
        self.assertEqual(result["metrics"]["xs"]["map50"], 0.)
        self.assertIsNone(result["metrics"]["s"]["map50"])
        xml["a.jpg"]["shape"] = [99,100]
        with self.assertRaises(ValueError):
            coco_size([r], xml)

    def test_duplicate_is_fp_and_outside_gt_is_ignored(self):
        boxes = [[0,0,10,10],[20,0,30,10],[0,20,50,70]]
        # Large GT prediction is ignored for XS. Duplicate of first XS is FP
        # before second XS TP, so XS AP is below one.
        predictions = [boxes[2], boxes[0], boxes[0], boxes[1]]
        r, xml = example(boxes, predictions, [.99,.9,.8,.7])
        result = coco_size([r], xml)
        self.assertGreater(result["metrics"]["xs"]["map50"], .8)
        self.assertLess(result["metrics"]["xs"]["map50"], .9)
        r, xml = example(boxes, [boxes[2],boxes[0],boxes[1]], [.99,.9,.7])
        self.assertAlmostEqual(coco_size([r],xml)["metrics"]["xs"]["map50"], 1.)

    def test_tied_scores_are_repeatable(self):
        r, xml = example([[0,0,10,10]], [[50,50,60,60],[0,0,10,10]], [.8,.8])
        a = coco_size([r], xml)
        self.assertEqual(a, coco_size([r], xml))
        self.assertAlmostEqual(a["metrics"]["xs"]["map50"], .5)


if __name__ == "__main__":
    unittest.main()
