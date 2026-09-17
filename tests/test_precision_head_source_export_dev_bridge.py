import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_precision_head_source_export_dev_bridge as bridge  # noqa: E402


class PrecisionHeadSourceExportBridgeTests(unittest.TestCase):
    @staticmethod
    def _runtime():
        import torch
        from ultralytics.utils import nms, ops

        return {"torch": torch, "np": np, "nms": nms, "ops": ops}

    @staticmethod
    def _trace(orig=(100, 200), ratio=(2.0, 2.0), left=10, top=20):
        return {
            "original_image_size_hw": list(orig),
            "letterbox": {
                "ratio": list(ratio),
                "padding": {"left": left, "top": top, "right": 0, "bottom": 0},
            },
        }

    def test_call_budget_is_two_forwards_per_image_and_no_builds(self):
        budget = bridge.call_budget()
        self.assertEqual(budget["total_ordinary_forward_or_session_calls"], 6544)
        self.assertEqual(budget["per_model"], {"native_cpu_forward": 1636, "onnx_cpu_session_run": 1636})
        self.assertEqual(budget["export_calls"], 0)
        self.assertEqual(budget["tensorRT_build_calls"], 0)

    def test_routes_keep_v8_nms_and_v26_filtering_distinct(self):
        v8 = bridge.application_route("yolov8n")
        v26 = bridge.application_route("yolo26n")
        self.assertTrue(v8["nms"])
        self.assertFalse(v8["end2end"])
        self.assertFalse(v26["nms"])
        self.assertTrue(v26["end2end"])
        self.assertIn("non_max_suppression", v8["postprocess"])
        self.assertIn("end2end=True", v26["postprocess"])

    def test_real_installed_v8_postprocess_is_class_aware_and_strict_thresholded(self):
        primary = np.zeros((1, 7, 8400), dtype=np.float32)
        # Two identical class-0 boxes: the lower score must be suppressed.
        primary[0, 0:4, 0] = [50, 60, 20, 20]
        primary[0, 4, 0] = 0.90
        primary[0, 0:4, 1] = [50, 60, 20, 20]
        primary[0, 4, 1] = 0.80
        # Same geometry but class-1: class-aware NMS must retain it.
        primary[0, 0:4, 2] = [50, 60, 20, 20]
        primary[0, 5, 2] = 0.70
        # The producer uses a strict > confidence comparison.
        primary[0, 0:4, 3] = [30, 40, 10, 10]
        primary[0, 4, 3] = bridge.CONF
        primary[0, 0:4, 4] = [30, 40, 10, 10]
        primary[0, 4, 4] = bridge.CONF + 1e-4
        result = bridge.application_postprocess("yolov8n", primary, self._trace(), self._runtime())
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["class_id"].count(0), 2)
        self.assertIn(1, result["class_id"])
        self.assertNotIn(bridge.CONF, result["confidence"])
        self.assertEqual(result["xyxy"][0], [15.0, 15.0, 25.0, 25.0])

    def test_real_installed_v26_route_filters_without_second_nms_and_scales(self):
        primary = np.zeros((1, 300, 6), dtype=np.float32)
        primary[0, 0] = [40, 50, 60, 70, 0.90, 0]
        primary[0, 1] = [40, 50, 60, 70, 0.80, 0]
        primary[0, 2] = [40, 50, 60, 70, 0.70, 1]
        primary[0, 3] = [30, 40, 40, 50, bridge.CONF, 2]
        primary[0, 4] = [30, 40, 40, 50, bridge.CONF + 1e-4, 2]
        result = bridge.application_postprocess("yolo26n", primary, self._trace(), self._runtime())
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["class_id"], [0, 0, 1, 2])
        self.assertEqual(result["xyxy"][0], [15.0, 15.0, 25.0, 25.0])
        self.assertNotIn(bridge.CONF, result["confidence"])

    def test_empty_and_nonfinite_outputs_are_fail_closed(self):
        runtime = self._runtime()
        empty = np.zeros((1, 7, 8400), dtype=np.float32)
        result = bridge.application_postprocess("yolov8n", empty, self._trace(), runtime)
        self.assertEqual(result["count"], 0)
        invalid = empty.copy()
        invalid[0, 4, 0] = np.nan
        with self.assertRaises(bridge.BridgeUnresolved):
            bridge.application_postprocess("yolov8n", invalid, self._trace(), runtime)

    def test_dev_inventory_order_and_label_membership_are_hard_checks(self):
        names = ["00001.jpg", "00002.jpg"]
        labels = ["00001.txt", "00002.txt"]
        bridge.validate_dev_names(names, labels, names)
        with self.assertRaises(ValueError):
            bridge.validate_dev_names(names[::-1], labels, names)
        with self.assertRaises(ValueError):
            bridge.validate_dev_names(names, ["00001.txt", "00003.txt"], names)

    def test_ordered_comparison_does_not_rematch(self):
        def record(source, image, boxes):
            return {"source": source, "image": image, "order": 0, "detections": {"count": len(boxes), "xyxy": boxes, "confidence": [0.5] * len(boxes), "class_id": [0] * len(boxes)}}

        native = [record("native", "00001.jpg", [[0, 0, 1, 1]])]
        onnx = [record("onnx", "00001.jpg", [[1, 1, 2, 2]])]
        result = bridge.ordered_output_comparison(native, onnx)
        self.assertEqual(result["ordered_payload_different_images"], 1)
        self.assertEqual(result["bounded_examples"][0]["image"], "00001.jpg")
        self.assertIn("not performed", result["matching"])

    def test_signed_delta_is_onnx_minus_native_for_all_size_bins(self):
        labels = ("all", "xs", "s", "m", "l", "xl")
        def report(offset):
            return {"metrics": {label: {"map50": 0.1 + offset, "map50_95": 0.2 + offset} for label in labels}}

        result = bridge.signed_metric_delta(report(0.0), report(0.03))
        self.assertEqual(set(result), set(labels))
        self.assertAlmostEqual(result["all"]["map50"], 0.03)
        self.assertAlmostEqual(result["xl"]["map50_95"], 0.03)
        self.assertIn("ONNX minus native", result["all"]["definition"])

    def test_jsonl_writer_is_stream_friendly_and_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                row = {"count": 0, "xyxy": [], "confidence": [], "class_id": []}
                bridge.write_jsonl_line(handle, {"source": "native", "image": "00001.jpg", "order": 0, "detections": row})
                bridge.write_jsonl_line(handle, {"source": "native", "image": "00002.jpg", "order": 1, "detections": row})
            rows = bridge._load_records(path, ["00001.jpg", "00002.jpg"])
            self.assertEqual([row["order"] for row in rows], [0, 1])

    def test_bounded_mock_child_runs_both_sources_and_writes_separate_records(self):
        import torch

        class FakeNetwork:
            stride = 32

            def to(self, _device):
                return self

            def float(self):
                return self

            def eval(self):
                return self

            def __call__(self, _tensor):
                return None

        class FakeSession:
            def run(self, _outputs, _inputs):
                return [np.zeros((1, 7, 8400), dtype=np.float32)]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "checkpoint.pt").write_bytes(b"checkpoint")
            (root / "model.onnx").write_bytes(b"onnx")
            image = root / "dev" / "images" / "00001.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            xml_path = root / "truth.zip"
            xml = "<annotation><size><width>200</width><height>100</height></size><object><name>prohibitory</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object></annotation>"
            with zipfile.ZipFile(xml_path, "w") as archive:
                archive.writestr("00001.xml", xml)
            checkpoint_sha = bridge.sha256_file(root / "checkpoint.pt")
            image_sha = bridge.sha256_file(image)
            plan = {
                "output_root": "output",
                "config": {"path": "config.json"},
                "models": {"yolov8n": {"checkpoint": {"expected": {"path": "checkpoint.pt", "sha256": checkpoint_sha}}, "onnx": {"repo_relative_path": "model.onnx"}, "accepted_contract": {}}},
                "dev_inventory": {"rows": [{"order": 0, "image": "00001.jpg", "source_path": "dev/images/00001.jpg", "bytes": 5, "sha256": image_sha, "canonical_orig_shape": [100, 200]}]},
                "xml_binding": {"archive": {"path": str(xml_path)}},
            }
            output = root / "output" / "models" / "yolov8n"
            trace = {"original_image_size_hw": [100, 200], "letterbox": {"ratio": [2.0, 2.0], "padding": {"left": 10, "top": 20, "right": 0, "bottom": 0}}, "tensor": {"shape": [1, 3, 640, 640], "dtype": "float32", "bytes_sha256": "tensor"}}
            fake_runtime = {"torch": torch, "np": np, "YOLO": lambda *_args, **_kwargs: SimpleNamespace(model=FakeNetwork()), "packages": {"observed": {"mock": True}}, "available_providers": ["CPUExecutionProvider"]}
            with patch.object(bridge.numeric, "validate_native_head", return_value={"head": {"type": "Detect", "index": 22, "end2end": False}}), patch.object(bridge.numeric, "extract_native_primary", return_value=(np.zeros((1, 7, 8400), dtype=np.float32), {"shape": [1, 7, 8400]})), patch.object(bridge.numeric, "trace_preprocess", return_value=(torch.zeros((1, 3, 640, 640), dtype=torch.float32), trace)), patch.object(bridge.numeric, "run_onnx_session", return_value=(FakeSession(), {"providers_observed": ["CPUExecutionProvider"]})), patch.object(bridge, "external_file_evidence", side_effect=lambda path, label: {"path": str(path), "bytes": 4, "sha256": bridge.numeric.EXPECTED_ONNX_SHA256["yolov8n"], "regular_file": True} if path.name == "model.onnx" else {"path": str(path), "bytes": path.stat().st_size, "sha256": bridge.sha256_file(path), "regular_file": True}):
                report = bridge.run_model_child(root, plan, "yolov8n", output, runtime_loader=lambda _config: fake_runtime)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["forward_counts"]["native_cpu_forward"]["completed"], 1)
            self.assertEqual(report["forward_counts"]["onnx_cpu_session_run"]["completed"], 1)
            self.assertTrue((output / "native_records.jsonl").is_file())
            self.assertTrue((output / "onnx_records.jsonl").is_file())
            self.assertTrue((output / "preprocess_trace.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
