import hashlib
import json
import struct
import os
import subprocess
import tempfile
import types
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
    def __init__(self, fail_prepare=False, fail_export=False):
        self.fail_prepare = fail_prepare
        self.fail_export = fail_export
        self.calls = []
        self.input_payload = b"\0" * (1 * 3 * 640 * 640 * 4)
        self.output_payload = b"\0" * (1 * 7 * 8400 * 4)
        self.ort_payload = self.output_payload
        self.last_export_internal_forwards = 0

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
        return {"head_class": "Detect", "head_module": "fake.head", "class_order": list(bundle.EXPECTED_CLASS_ORDER), "nc": 3, "reg_max": 16, "end2end": False, "export": False, "training": False, "xyxy": False, "decoded_boxes": True, "coordinate_format": "xywh_pixels_of_640_letterboxed_input", "score_semantics": "sigmoid class probabilities", "stride": [8, 16, 32]}

    def preprocess(self, image_path):
        self.calls.append(("preprocess", image_path.name))
        return bundle.TensorArtifact(self.input_payload, bundle.INPUT_SHAPE, "float32", "little", True, self.input_payload)

    def native_forward(self, model, tensor):
        self.calls.append("native_forward")
        return bundle.TensorArtifact(self.output_payload, bundle.OUTPUT_SHAPE, "float32", "little", True, self.output_payload)

    def export(self, checkpoint, export_dir, options):
        self.calls.append(("export", checkpoint, options))
        if self.fail_export:
            self.last_export_internal_forwards = 5
            raise bundle.SourceBundleError("EXPORT_FAILED", "fake exporter failed after internal forwards")
        self.export_checkpoint = checkpoint
        path = checkpoint.with_suffix(".onnx")
        path.write_bytes(b"fake-onnx")
        return path, 2, {"before": {"export": False}, "after": {"export": True}, "checkpoint": str(checkpoint)}

    def validate_onnx(self, onnx_path):
        self.calls.append("validate_onnx")
        return {"input_name": "images", "output_name": "output0", "input_shape": list(bundle.INPUT_SHAPE), "output_shape": list(bundle.OUTPUT_SHAPE), "input_dtype": "float32", "output_dtype": "float32", "opset": 17, "forbidden_nodes": [], "checker": "pass"}

    def open_ort(self, onnx_path):
        self.calls.append("open_ort")
        return object()

    def ort_forward(self, session, tensor):
        self.calls.append("ort_forward")
        return bundle.TensorArtifact(self.ort_payload, bundle.OUTPUT_SHAPE, "float32", "little", True, self.ort_payload)


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
                code, manifest = bundle.SourceBundleRunner(source_root, out_dir, runtime, commit="unknown").run()
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
                code, failure = bundle.SourceBundleRunner(source_root, out_dir, runtime, commit="unknown").run()
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

    def test_compare_counts_every_failure_by_domain_and_rejects_nonfinite_bytes(self):
        count = bundle.OUTPUT_SHAPE[1] * bundle.OUTPUT_SHAPE[2]
        expected = [0.0] * count
        actual = list(expected)
        for index in range(25):
            actual[index] = 1.0
        for index in range(30):
            actual[4 * bundle.OUTPUT_SHAPE[2] + index] = 1.0
        good = bundle.TensorArtifact(struct.pack("<{}f".format(count), *expected), bundle.OUTPUT_SHAPE, "float32", "little", True)
        observed = bundle.TensorArtifact(struct.pack("<{}f".format(count), *actual), bundle.OUTPUT_SHAPE, "float32", "little", True)
        comparison = bundle.compare_source_outputs(good, observed)
        self.assertEqual(comparison["failing_elements"], 55)
        self.assertEqual(comparison["box_failing_elements"], 25)
        self.assertEqual(comparison["score_failing_elements"], 30)
        self.assertEqual(len(comparison["bounded_failures"]), 20)

        nonfinite = bytearray(good.payload)
        nonfinite[:4] = struct.pack("<f", float("inf"))
        with self.assertRaisesRegex(AdapterError, "SOURCE_TENSOR_NONFINITE"):
            bundle.compare_source_outputs(good, bundle.TensorArtifact(bytes(nonfinite), bundle.OUTPUT_SHAPE, "float32", "little", True))

    def test_numerical_fail_writes_terminal_provenance(self):
        runtime = FakeSourceRuntime()
        bad = bytearray(runtime.output_payload)
        bad[:4] = struct.pack("<f", 1.0)
        runtime.ort_payload = bytes(bad)
        with tempfile.TemporaryDirectory() as temp:
            source_root = Path(temp) / "source"
            checkpoint = source_root / bundle.YOLO11N_CHECKPOINT_PATH
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"frozen-checkpoint")
            out_dir = Path(temp) / "e2l1-012-source-v1"
            with patch.object(bundle, "verify_fixture", side_effect=[fake_fixture_manifest(), fake_fixture_manifest()]), patch.object(bundle, "file_sha256", side_effect=lambda path: bundle.YOLO11N_CHECKPOINT_SHA256 if path.name == "best.pt" else hashlib.sha256(path.read_bytes()).hexdigest()):
                code, manifest = bundle.SourceBundleRunner(source_root, out_dir, runtime, commit="unknown").run()
            self.assertEqual(code, 3)
            self.assertEqual(manifest["status"], "execution_complete_numerical_fail")
            self.assertEqual(manifest["execution_status"], "complete")
            self.assertIn("environment", manifest)
            self.assertIn("model_flags", manifest)
            self.assertIn("onnx", manifest)
            self.assertEqual(len(manifest["records"]), 3)
            self.assertIn("private_inventory", manifest)
            self.assertTrue((out_dir / "public" / "manifest.json").is_file())
            self.assertFalse((out_dir / "public" / "failure.json").exists())

    def test_export_exception_preserves_internal_forward_counter(self):
        runtime = FakeSourceRuntime(fail_export=True)
        with tempfile.TemporaryDirectory() as temp:
            source_root = Path(temp) / "source"
            checkpoint = source_root / bundle.YOLO11N_CHECKPOINT_PATH
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"frozen-checkpoint")
            out_dir = Path(temp) / "e2l1-012-source-v1"
            with patch.object(bundle, "verify_fixture", side_effect=[fake_fixture_manifest(), fake_fixture_manifest()]), patch.object(bundle, "file_sha256", side_effect=lambda path: bundle.YOLO11N_CHECKPOINT_SHA256 if path.name == "best.pt" else hashlib.sha256(path.read_bytes()).hexdigest()):
                code, failure = bundle.SourceBundleRunner(source_root, out_dir, runtime, commit="unknown").run()
            self.assertEqual(code, 2)
            self.assertEqual(failure["attempted_completed"]["export_invocations_attempted"], 1)
            self.assertEqual(failure["attempted_completed"]["exporter_internal_forwards"], 5)
            self.assertIn("private_inventory", failure)

    def test_terminal_writer_failure_keeps_primary_error_and_writer_error(self):
        runtime = FakeSourceRuntime(fail_prepare=True)

        def writer(path, payload):
            if path.name == "report.md":
                raise OSError("report sink unavailable")
            bundle.write_once(path, payload)

        with tempfile.TemporaryDirectory() as temp:
            source_root = Path(temp) / "source"
            checkpoint = source_root / bundle.YOLO11N_CHECKPOINT_PATH
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"frozen-checkpoint")
            out_dir = Path(temp) / "e2l1-012-source-v1"
            with patch.object(bundle, "verify_fixture", return_value=fake_fixture_manifest()), patch.object(bundle, "file_sha256", return_value=bundle.YOLO11N_CHECKPOINT_SHA256):
                code, failure = bundle.SourceBundleRunner(source_root, out_dir, runtime, commit="unknown", writer=writer).run()
            self.assertEqual(code, 2)
            self.assertEqual(failure["error"]["code"], "SOURCE_DEPENDENCY_MISSING")
            self.assertEqual(failure["artifact_write_errors"][0]["artifact"], "report.md")
            persisted = json.loads((out_dir / "public" / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["error"]["code"], "SOURCE_DEPENDENCY_MISSING")
            self.assertEqual(persisted["artifact_write_errors"][0]["artifact"], "report.md")

    def test_actual_runtime_preprocess_handles_nonsquare_padding_and_frozen_bytes(self):
        env_python = Path(r"D:\Research\paper\local\measurement_audit_env\Scripts\python.exe")
        if not env_python.is_file():
            self.skipTest("Astra local measurement environment is unavailable")
        code = r'''
import json, tempfile
from pathlib import Path
import cv2, numpy as np, ultralytics
from edge_readiness.e2_source_bundle import UltralyticsSourceRuntime
with tempfile.TemporaryDirectory() as temp:
    path = Path(temp) / "nonsquare.png"
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    image[:, :, 0] = 11
    image[:, :, 1] = 37
    image[:, :, 2] = 203
    if not cv2.imwrite(str(path), image):
        raise AssertionError("synthetic image write failed")
    runtime = UltralyticsSourceRuntime()
    runtime.modules = {"numpy": np, "cv2": cv2, "ultralytics": ultralytics}
    tensor = runtime.preprocess(path)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == "float32" and tensor.byteorder == "little"
    assert tensor.value.flags["C_CONTIGUOUS"]
    assert tensor.value.tobytes(order="C") == tensor.payload
    assert tensor.metadata["original_shape"] == [3, 5, 3]
    assert tensor.metadata["color"] == "BGR_to_RGB"
    assert tensor.metadata["letterbox"]["auto"] is False
    assert tensor.metadata["resized_shape"] == [640, 640, 3]
    print(json.dumps({"nbytes": tensor.nbytes, "metadata": tensor.metadata}, sort_keys=True))
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "scripts")
        result = subprocess.run([str(env_python), "-c", code], env=environment, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"original_shape": [3, 5, 3]', result.stdout)
        self.assertIn('"color": "BGR_to_RGB"', result.stdout)

    def test_actual_runtime_native_packaging_and_ort_dispatch_use_real_runtime_class(self):
        env_python = Path(r"D:\Research\paper\local\measurement_audit_env\Scripts\python.exe")
        if not env_python.is_file():
            self.skipTest("Astra local measurement environment is unavailable")
        code = r'''
import json
import numpy as np, torch
from edge_readiness.e2_source_bundle import UltralyticsSourceRuntime, TensorArtifact, OUTPUT_SHAPE
class Model(torch.nn.Module):
    def forward(self, value):
        assert tuple(value.shape) == (1, 3, 640, 640)
        return torch.zeros(OUTPUT_SHAPE, dtype=torch.float32)
class Wrapper:
    model = Model()
class Input:
    name = "images"
class Session:
    def get_inputs(self):
        return [Input()]
    def run(self, _outputs, feeds):
        assert list(feeds) == ["images"]
        assert feeds["images"].dtype == np.float32
        return [np.zeros(OUTPUT_SHAPE, dtype=np.float32)]
runtime = UltralyticsSourceRuntime()
runtime.modules = {"numpy": np, "torch": torch}
source = np.zeros((1, 3, 640, 640), dtype=np.dtype("<f4"))
tensor = TensorArtifact(source.tobytes(order="C"), source.shape, "float32", "little", True, source)
native = runtime.native_forward(Wrapper(), tensor)
ort = runtime.ort_forward(Session(), tensor)
assert native.payload == native.value.tobytes(order="C")
assert ort.payload == ort.value.tobytes(order="C")
assert native.shape == OUTPUT_SHAPE == ort.shape
print(json.dumps({"native_bytes": native.nbytes, "ort_bytes": ort.nbytes, "equal": native.payload == ort.payload}))
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "scripts")
        result = subprocess.run([str(env_python), "-c", code], env=environment, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"equal": true', result.stdout)

    def test_actual_runtime_ort_session_options_are_cpu_bounded(self):
        class Options:
            intra_op_num_threads = None
            inter_op_num_threads = None
            execution_mode = None

        class Session:
            def __init__(self, path, sess_options, providers):
                self.path = path
                self.options = sess_options
                self.providers = providers

            def get_providers(self):
                return self.providers

        fake_ort = types.SimpleNamespace(
            SessionOptions=Options,
            ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="sequential"),
            InferenceSession=Session,
        )
        runtime = bundle.UltralyticsSourceRuntime()
        runtime.modules = {"onnxruntime": fake_ort}
        session = runtime.open_ort(Path("private.onnx"))
        self.assertEqual(session.options.intra_op_num_threads, 2)
        self.assertEqual(session.options.inter_op_num_threads, 1)
        self.assertEqual(session.options.execution_mode, "sequential")
        self.assertEqual(session.providers, ["CPUExecutionProvider"])

    def test_actual_runtime_preflight_does_not_auto_install_missing_package(self):
        runtime = bundle.UltralyticsSourceRuntime()
        original = bundle.importlib.util.find_spec

        def missing(name):
            return None if name == "onnxslim" else original(name)

        with patch.object(bundle.importlib.util, "find_spec", side_effect=missing):
            with self.assertRaisesRegex(AdapterError, "SOURCE_DEPENDENCY_MISSING"):
                runtime.prepare()

    def test_actual_runtime_preflight_assigns_numpy_and_records_controls(self):
        runtime = bundle.UltralyticsSourceRuntime()
        numpy_module = types.SimpleNamespace(__version__="2.4.2")
        torch_module = types.SimpleNamespace(
            __version__="2.8.0",
            cuda=types.SimpleNamespace(is_available=lambda: False),
            set_num_threads=lambda value: setattr(torch_module, "threads", value),
            set_num_interop_threads=lambda value: setattr(torch_module, "interop", value),
            get_num_threads=lambda: getattr(torch_module, "threads", 2),
            get_num_interop_threads=lambda: getattr(torch_module, "interop", 1),
        )
        modules = {name: types.SimpleNamespace(__version__="installed") for name in bundle.REQUIRED_MODULES}
        modules["numpy"] = numpy_module
        modules["torch"] = torch_module
        modules["ultralytics"] = types.SimpleNamespace(__version__="8.4.102")
        with patch.object(bundle.importlib.util, "find_spec", return_value=object()), patch.object(bundle.importlib, "import_module", side_effect=lambda name: modules[name]):
            environment = runtime.prepare()
        self.assertIs(runtime.modules["numpy"], numpy_module)
        self.assertEqual(environment["auto_install_control"], "false")
        self.assertEqual(environment["torch_threads"], {"intra_op": 2, "interop": 1})


if __name__ == "__main__":
    unittest.main()
