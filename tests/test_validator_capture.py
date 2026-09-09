import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@unittest.skipUnless(importlib.util.find_spec("ultralytics"), "Requires pinned Ultralytics test environment")
class ValidatorCaptureTests(unittest.TestCase):
    def test_parent_equivalence_and_json_replay(self):
        import numpy as np
        import torch
        from ultralytics.models.yolo.detect.val import DetectionValidator
        from capture_cctsdb_validator import CaptureValidator, replay_statistics, metric_summary, write_json

        with tempfile.TemporaryDirectory() as d:
            base = DetectionValidator(args={"plots": False}, save_dir=Path(d)/"base")
            captured = CaptureValidator(args={"plots": False}, save_dir=Path(d)/"capture")
            names = {0: "prohibitory", 1: "mandatory", 2: "warning"}
            for v in (base, captured):
                v.device = torch.device("cpu")
                v.data = {"val": "synthetic_dev/images"}
                v.training = False
                v.init_metrics(SimpleNamespace(names=names, end2end=False))
            # Three images: tied scores with FP/TP, an image with no predictions,
            # and an image without GT. This also exercises float64 empty stats.
            batch = {"img": torch.zeros(3, 3, 100, 100), "batch_idx": torch.tensor([0, 0, 1]),
                "cls": torch.tensor([[0.], [1.], [2.]]),
                "bboxes": torch.tensor([[.2,.2,.2,.2],[.6,.6,.2,.2],[.5,.5,.1,.1]]),
                "ori_shape": [(50,100)]*3, "ratio_pad": [((1.,1.),(0.,25.))]*3,
                "im_file": ["a.jpg","b.jpg","c.jpg"]}
            preds = [{"bboxes":torch.tensor([[10.,10.,30.,30.],[1.,1.,2.,2.],[50.,50.,70.,70.]]), "conf":torch.tensor([.8,.8,.7]),"cls":torch.tensor([0.,0.,1.])},
                {"bboxes":torch.empty(0,4),"conf":torch.empty(0),"cls":torch.empty(0)},
                {"bboxes":torch.tensor([[0.,0.,4.,4.]]),"conf":torch.tensor([.2]),"cls":torch.tensor([2.])}]
            original = preds[0]["bboxes"].clone()
            for v in (base,captured):
                v.update_metrics(preds,batch)
            self.assertTrue(torch.equal(preds[0]["bboxes"], original))
            # Capture must leave every statistic supplied to the parent unchanged.
            for key in base.metrics.stats:
                for a,b in zip(base.metrics.stats[key],captured.metrics.stats[key]):
                    np.testing.assert_array_equal(a,b)
            base.get_stats()
            captured.get_stats()
            self.assertEqual(metric_summary(base.metrics),metric_summary(captured.metrics))
            self.assertEqual(len(captured.capture_records),3)
            self.assertEqual(captured.capture_records[0]["xyxy"][0][1],0.)
            self.assertEqual(captured.capture_records[0]["validator_input"]["prediction_xyxy"][0][1],10.)
            p = Path(d)/"records.json"
            write_json(p,captured.capture_records)
            replay = replay_statistics(json.loads(p.read_text()))
            expected = metric_summary(base.metrics)
            for key in ("map50","map50_95","precision","recall"):
                self.assertAlmostEqual(replay[key],expected[key],places=14)
            self.assertEqual(replay["per_class_ap50"],expected["per_class_ap50"])
            with self.assertRaises(FileExistsError):
                write_json(p, {})

    def test_no_records_rejected(self):
        from capture_cctsdb_validator import replay_statistics
        with self.assertRaises(ValueError):
            replay_statistics([])


if __name__ == "__main__":
    unittest.main()
