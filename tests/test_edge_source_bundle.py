import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from edge_readiness import e2_source_bundle as bundle  # noqa: E402
from edge_readiness.e2_source_fixture import preprocess_decoded_bgr  # noqa: E402
from edge_readiness.edge_errors import AdapterError  # noqa: E402


class FakeSourceRuntime:
    def __init__(self, fail_prepare=False):
        self.fail_prepare = fail_prepare
        self.calls = []
        self.input_payload = b"\0" * (1 * 3 * 640 * 640 * 4)
        self.output_payload = b"\0" * (1 * 7 * 8400 * 4)

    def prepare(self):
        self.calls.append("prepare")
        if self.fail_prepare:
            raise bundle.SourceBundleError("SOURCE_DEPENDENCY_MISSING", "fake missing dependency")
        return {"python": "3.8.10", "torch": "fake", "onnxruntime": "fake", "ort_providers_required": ["CPUExecutionProvider"]}

    def load_model(self, checkpoint):
        self.calls.append(("load_model", checkpoint))
        return object()

    def model_flags(self, model):
        self.calls.append("model_flags")
        return {"class_order": list(bundle.EXPECTED_CLASS_ORDER), "nc": 3, "reg_max": 16, "end2end": False, "export": False, "stride": [8, 16, 32]}

    def preprocess(self, image_path):
        self.calls.append(("preprocess", image_path.name))
        return bundle.TensorArtifact(self.input_payload, bundle.INPUT_SHAPE, "float32", "little", True, self.input_payload)

    def native_forward(self, model, tensor):
        self.calls.append("native_forward")
        return bundle.TensorArtifact(self.output_payload, bundle.OUTPUT_SHAPE, "float32", "little", True, self.output_payload)

    def export(self, checkpoint, export_dir, options):
        self.calls.append(("export", checkpoint, options))
        export_dir.mkdir(parents=True, exist_ok=False)
        path = export_dir / "best.onnx"
        path.write_bytes(b"fake-onnx")
        return path, 2

    def validate_onnx(self, onnx_path):
        self.calls.append("validate_onnx")
        return {"input_name": "images", "output_name": "output0", "input_shape": list(bundle.INPUT_SHAPE), "output_shape": list(bundle.OUTPUT_SHAPE), "input_dtype": "float32", "output_dtype": "float32", "checker": "pass"}

    def open_ort(self, onnx_path):
        self.calls.append("open_ort")
        return object()

    def ort_forward(self, session, tensor):
        self.calls.append("ort_forward")
        return bundle.TensorArtifact(self.output_payload, bundle.OUTPUT_SHAPE, "float32", "little", True, self.output_payload)


def fake_fixture_manifest():
    return {"schema_version": "fake-fixture", "status": "verified_read_only", "canonical_order": list(bundle.FIXTURE_IDS), "sequence_sha256": "fixture-sequence", "records": []}


class SourceBundleTests(unittest.TestCase):
    def test_helper_and_source_packer_have_identical_synthetic_bgr_semantics(self):
        pixels = bytes((index * 13 + 7) % 256 for index in range(640 * 640 * 3))
        helper = preprocess_decoded_bgr(pixels, 640, 640, lambda rgb, _w, _h, _new_w, _new_h: rgb)
        actual = bundle.pack_letterboxed_bgr(pixels, 640, 640)
        self.assertEqual(helper.payload, actual.payload)
        self.assertEqual(actual.shape, bundle.INPUT_SHAPE)
        self.assertEqual(actual.dtype, "float32")
        self.assertEqual(actual.byteorder, "little")

    def test_injected_runtime_produces_private_bundle_and_exact_source_compare(self):
        runtime = FakeSourceRuntime()
        with tempfile.TemporaryDirectory() as temp:
            source_root = Path(temp) / "source"
            checkpoint = source_root / bundle.YOLO11N_CHECKPOINT_PATH
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"frozen-checkpoint")
            out_dir = Path(temp) / "e2l1-012-source-v1"
            with patch.object(bundle, "verify_fixture", side_effect=[fake_fixture_manifest(), fake_fixture_manifest()]), patch.object(bundle, "file_sha256", side_effect=lambda path: bundle.YOLO11N_CHECKPOINT_SHA256 if path.name == "best.pt" else hashlib.sha256(path.read_bytes()).hexdigest()):
                code, manifest = bundle.SourceBundleRunner(source_root, out_dir, runtime, commit="test-commit").run()
            self.assertEqual(code, 0)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["attempted_completed"]["native_forwards_completed"], 3)
            self.assertEqual(manifest["attempted_completed"]["ort_forwards_completed"], 3)
            self.assertEqual(manifest["export"]["internal_forward_calls"], 2)
            self.assertTrue((out_dir / "public" / "manifest.json").is_file())
            self.assertTrue((out_dir / "public" / "index.json").is_file())
            self.assertEqual((source_root / bundle.YOLO11N_CHECKPOINT_PATH).read_bytes(), b"frozen-checkpoint")
            self.assertEqual(runtime.calls.count("native_forward"), 3)
            self.assertEqual(runtime.calls.count("ort_forward"), 3)

    def test_missing_dependency_is_fail_closed_and_records_counters(self):
        runtime = FakeSourceRuntime(fail_prepare=True)
        with tempfile.TemporaryDirectory() as temp:
            source_root = Path(temp) / "source"
            checkpoint = source_root / bundle.YOLO11N_CHECKPOINT_PATH
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"frozen-checkpoint")
            out_dir = Path(temp) / "e2l1-012-source-v1"
            with patch.object(bundle, "verify_fixture", return_value=fake_fixture_manifest()), patch.object(bundle, "file_sha256", return_value=bundle.YOLO11N_CHECKPOINT_SHA256):
                code, failure = bundle.SourceBundleRunner(source_root, out_dir, runtime, commit="test-commit").run()
            self.assertEqual(code, 2)
            self.assertEqual(failure["error"]["code"], "SOURCE_DEPENDENCY_MISSING")
            self.assertEqual(failure["attempted_completed"]["native_forwards_completed"], 0)
            self.assertTrue((out_dir / "public" / "failure.json").is_file())
            self.assertIn("plan_written", failure["events"])

    def test_existing_root_is_refused_without_runtime_side_effect(self):
        runtime = FakeSourceRuntime()
        with tempfile.TemporaryDirectory() as temp:
            out_dir = Path(temp) / "e2l1-012-source-v1"
            out_dir.mkdir()
            marker = out_dir / "marker.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing existing source bundle root"):
                bundle.SourceBundleRunner(Path(temp), out_dir, runtime).run()
            self.assertEqual(runtime.calls, [])
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_shape_and_semantic_mismatch_fail_closed(self):
        good = bundle.TensorArtifact(b"\0" * (1 * 7 * 8400 * 4), bundle.OUTPUT_SHAPE, "float32", "little", True)
        bad_shape = bundle.TensorArtifact(b"\0" * 4, (1, 7, 1), "float32", "little", True)
        with self.assertRaisesRegex(AdapterError, "SOURCE_TENSOR_SHAPE_MISMATCH"):
            bundle.compare_source_outputs(good, bad_shape)
        bad_value = bytearray(good.payload)
        bad_value[0:4] = b"\0\0\x80?"
        mismatch = bundle.compare_source_outputs(good, bundle.TensorArtifact(bytes(bad_value), bundle.OUTPUT_SHAPE, "float32", "little", True))
        self.assertEqual(mismatch["status"], "fail")
        self.assertGreater(mismatch["failing_elements"], 0)

    def test_actual_runtime_preflight_does_not_auto_install_missing_package(self):
        runtime = bundle.UltralyticsSourceRuntime()
        original = bundle.importlib.util.find_spec

        def missing(name):
            return None if name == "onnxslim" else original(name)

        with patch.object(bundle.importlib.util, "find_spec", side_effect=missing):
            with self.assertRaisesRegex(AdapterError, "SOURCE_DEPENDENCY_MISSING"):
                runtime.prepare()


if __name__ == "__main__":
    unittest.main()
