import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from capture_cctsdb_validator import capture_inputs, FROZEN_WEIGHTS_SHA256
from audit_cctsdb_measurement import sha256
from run_g0_int8_capture import check_same_targets, comparison_row


class Int8CaptureTests(unittest.TestCase):
    def test_identity_and_cache_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            base = repo / "results/calibration_method_v2/rtx8000/yolo11n"
            (base / "engines").mkdir(parents=True)
            (base / "dev_eval").mkdir()
            engine = base / "engines/yolo11n_int8_uniform_s42.engine"
            engine.write_bytes(b"synthetic engine: never loaded")
            prov_path = Path(str(engine) + ".provenance.json")
            old = {"model_sha256": sha256(engine), "data": "/repo/configs/cctsdb2021_dev.yaml"}
            prov = {"engine_sha256": sha256(engine), "source_weights_sha256": FROZEN_WEIGHTS_SHA256,
                "precision": "int8", "calibration_cache": {"isolation": "fresh per-engine temporary workspace; never shared across policies or seeds", "sha256": "abc"}}
            (base / "dev_eval/yolo11n_int8_uniform_s42.json").write_text(json.dumps(old))
            prov_path.write_text(json.dumps(prov))
            self.assertEqual(capture_inputs(repo, "uniform")[-1], sha256(engine))
            del prov["calibration_cache"]
            prov_path.write_text(json.dumps(prov))
            with self.assertRaisesRegex(ValueError, "isolation"):
                capture_inputs(repo, "uniform")
            engine.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                capture_inputs(repo, "uniform")
            with self.assertRaises(ValueError):
                capture_inputs(repo, "test")

    def test_targets_must_match_but_predictions_can_differ(self):
        r = {"image": "a.jpg", "orig_shape": [100,100], "xyxy": [], "confidence": [], "class_id": [],
             "validator_input": {"imgsz": [640,640], "ratio_pad": [[6.4,6.4],[0,0]], "target_xyxy": [[0,0,64,64]], "target_class_id": [0]}}
        a = {"records": [r]}
        b = copy.deepcopy(a)
        b["records"][0].update(xyxy=[[0,0,10,10]], confidence=[.9], class_id=[0])
        check_same_targets(a,b)
        b["records"][0]["validator_input"]["target_xyxy"][0][2] += 1
        with self.assertRaisesRegex(ValueError, "target/preprocessing"):
            check_same_targets(a,b)

    def test_metric_delta_units_and_convention_guard(self):
        ref = {"metrics": {k:.8 for k in ("map50","map50_95","precision","recall")}}
        current = {"model_sha256":"abc", "metrics": {k:.7 for k in ref["metrics"]}}
        sizes = {"metric_id":"id", "rules":{"x":1}, "evaluator_source_sha256":"coco", "xml_sha256":"xml",
                 "metrics": {"xs":{"instances":10,"map50":.8,"map50_95":.5}}}
        other = copy.deepcopy(sizes)
        other["metrics"]["xs"]["map50"] = .7
        row = comparison_row("uniform",current,other,ref,sizes)
        self.assertAlmostEqual(row["coco_xml"]["xs"]["delta_map50_pp"], -10.)
        self.assertAlmostEqual(row["ultralytics_full"]["delta_pp"]["map50"], -10.)
        other["rules"]["x"] = 2
        with self.assertRaisesRegex(ValueError, "conventions"):
            comparison_row("uniform",current,other,ref,sizes)


if __name__ == "__main__":
    unittest.main()
