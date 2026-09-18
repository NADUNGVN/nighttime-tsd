import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_precision_head_trt_feasibility as smoke  # noqa: E402


class FakeConfig:
    def __init__(self):
        self.flags = set()
        self.timing_cache_empty = False
        self.profiling_verbosity = None

    def set_memory_pool_limit(self, _pool, value):
        self.workspace = value

    def set_flag(self, flag):
        self.flags.add(flag)

    def clear_flag(self, flag):
        self.flags.discard(flag)

    def get_flag(self, flag):
        return flag in self.flags

    def create_timing_cache(self, value):
        self.timing_cache_empty = value == b""
        return object()

    def set_timing_cache(self, _cache, ignore_mismatch):
        self.ignore_mismatch = ignore_mismatch


class FakeNetwork:
    num_inputs = 1
    num_outputs = 1

    def get_input(self, _index):
        return SimpleNamespace(name="images", shape=(1, 3, 640, 640), dtype="DataType.FLOAT")

    def get_output(self, _index):
        return SimpleNamespace(name="output0", shape=(1, 7, 8400), dtype="DataType.FLOAT")


class FakeYolo26Network(FakeNetwork):
    def get_output(self, _index):
        return SimpleNamespace(name="output0", shape=(1, 300, 6), dtype="DataType.FLOAT")


class FakeEngine:
    num_io_tensors = 2

    def get_tensor_name(self, index):
        return ("images", "output0")[index]

    def get_tensor_mode(self, name):
        return "TensorIOMode.INPUT" if name == "images" else "TensorIOMode.OUTPUT"

    def get_tensor_shape(self, name):
        return (1, 3, 640, 640) if name == "images" else (1, 7, 8400)

    def get_tensor_dtype(self, _name):
        return "DataType.FLOAT"

    def create_engine_inspector(self):
        return SimpleNamespace(get_engine_information=lambda _format: '{"layers": []}')


class FakeTrt:
    class Logger:
        VERBOSE = "verbose"

        def __init__(self, _level):
            pass

    class MemoryPoolType:
        WORKSPACE = "workspace"

    class BuilderFlag:
        FP16 = "fp16"
        INT8 = "int8"
        TF32 = "tf32"

    class ProfilingVerbosity:
        DETAILED = "detailed"

    class LayerInformationFormat:
        JSON = "json"

    class OnnxParser:
        def __init__(self, network, _logger):
            self.network = network
            self.num_errors = 0

        def parse_from_file(self, _path):
            return True

        def get_error(self, _index):
            return "unused"

    class Builder:
        last_config = None

        def __init__(self, _logger):
            pass

        def create_network(self, _flags):
            return FakeNetwork()

        def create_builder_config(self):
            self.__class__.last_config = FakeConfig()
            return self.__class__.last_config

        def build_serialized_network(self, _network, _config):
            return b"private-engine-bytes"

    class Runtime:
        def __init__(self, _logger):
            pass

        def deserialize_cuda_engine(self, _serialized):
            return FakeEngine()


class FakeYolo26Trt(FakeTrt):
    class Builder(FakeTrt.Builder):
        def create_network(self, _flags):
            return FakeYolo26Network()

    class Runtime(FakeTrt.Runtime):
        def deserialize_cuda_engine(self, _serialized):
            engine = FakeEngine()
            engine.get_tensor_shape = lambda name: (1, 3, 640, 640) if name == "images" else (1, 300, 6)
            return engine


class FailingParserTrt(FakeTrt):
    class OnnxParser(FakeTrt.OnnxParser):
        def __init__(self, network, logger):
            super().__init__(network, logger)
            self.num_errors = 1

        def parse_from_file(self, _path):
            return False

        def get_error(self, _index):
            return "synthetic parser failure"


class PrecisionHeadTrtFeasibilityTests(unittest.TestCase):
    def test_contract_is_exactly_two_builds_and_sixteen_each_call(self):
        self.assertEqual(smoke.call_contract(), {
            "models": 2,
            "independent_builds": 2,
            "trt_application_enqueues": 16,
            "onnx_cpu_reference_calls": 16,
            "native_forwards": 0,
            "warmup_calls": 0,
            "retries": 0,
            "calibration_batches": 0,
            "dev_captures": 0,
            "test_captures": 0,
        })

    def test_fixture_order_is_the_locked_u42_eight(self):
        self.assertEqual(len(smoke.FIXTURE_NAMES), 8)
        self.assertEqual(smoke.FIXTURE_NAMES[0], "train/images/00006.jpg")
        self.assertEqual(smoke.FIXTURE_NAMES[-1], "train/images/00104.jpg")
        self.assertTrue(all(name.startswith("train/images/") for name in smoke.FIXTURE_NAMES))

    def test_builder_settings_enable_only_fp16_and_use_fresh_cache(self):
        self.assertTrue(smoke.BUILD_SETTINGS["fp16"])
        self.assertFalse(smoke.BUILD_SETTINGS["int8"])
        self.assertFalse(smoke.BUILD_SETTINGS["tf32"])
        self.assertFalse(smoke.BUILD_SETTINGS["precision_overrides"])
        self.assertEqual(smoke.BUILD_SETTINGS["workspace_bytes"], 4 << 30)
        self.assertIn("fresh_empty", smoke.BUILD_SETTINGS["timing_cache"])

    def test_application_contract_preserves_v8_and_v26_routes(self):
        v8 = smoke.application_contract("yolov8n")
        v26 = smoke.application_contract("yolo26n")
        self.assertEqual(v8["output"]["shape"], [1, 7, 8400])
        self.assertEqual(v26["output"]["shape"], [1, 300, 6])
        self.assertTrue(v8["route"]["nms"])
        self.assertTrue(v26["route"]["end2end"])
        self.assertIn("no second NMS", v26["comparison"]["v26"])

    def test_single_model_selection_cannot_change_reviewed_budget(self):
        with self.assertRaises(ValueError):
            smoke.selected_models("yolov8n")
        self.assertEqual(smoke.selected_models("all"), ["yolov8n", "yolo26n"])

    def test_gpu_identity_parser_requires_uuid_name_and_nine_fields(self):
        row = "GPU-1, Quadro RTX 8000, 595.71.05, P8, 32, 10 W, 300 MHz, 405 MHz, 29 MiB"
        identity = smoke.parse_gpu_identity(row)
        self.assertEqual(identity["uuid"], "GPU-1")
        self.assertEqual(identity["name"], "Quadro RTX 8000")
        with self.assertRaises(smoke.FeasibilityUnresolved):
            smoke.parse_gpu_identity("GPU-1, Quadro RTX 8000")

    def test_gpu_identity_check_rejects_different_device(self):
        expected = {"uuid": "GPU-1", "name": "RTX"}
        with self.assertRaises(smoke.FeasibilityUnresolved):
            smoke.validate_same_gpu(expected, {"uuid": "GPU-2", "name": "RTX"}, "test")

    def test_desktop_exception_requires_exact_current_pid_path(self):
        path = "/snap/snapd-desktop-integration/391/usr/bin/snapd-desktop-integration"
        self.assertEqual(smoke.parse_desktop_confirmations([f"644963={path}"]), {644963: path})
        with self.assertRaises(ValueError):
            smoke.parse_desktop_confirmations(["644963=/usr/bin/python"])
        blocked = smoke.classify_gpu_processes(f"644963, {path}, 5 MiB", {}, {}, {})
        self.assertEqual(blocked[0]["classification"], "blocked_desktop_unconfirmed")
        allowed = smoke.classify_gpu_processes(f"644963, {path}, 5 MiB", {644963: path}, {}, {})
        self.assertTrue(allowed[0]["allowed"])

    def test_low_memory_unknown_process_is_still_blocked(self):
        detail = smoke.classify_gpu_processes("123, /home/other/job, 1 MiB", {}, {}, {})
        self.assertFalse(detail[0]["allowed"])
        self.assertEqual(detail[0]["classification"], "blocked_non_allowlisted_process")
        with self.assertRaises(smoke.FeasibilityUnresolved):
            smoke.ensure_gpu_guard({"process_guard": {"blocked_processes": detail, "unmatched_confirmations": [], "background_workload": []}})

    def test_parser_and_engine_contracts_reject_wrong_output_shape_or_dtype(self):
        with self.assertRaises(smoke.FeasibilityUnresolved):
            smoke.validate_io_contract("yolo26n", {"input": dict(smoke.EXPECTED_INPUT), "output": {"name": "output0", "shape": [1, 300, 6], "dtype": "float16"}}, "fake")
        with self.assertRaises(smoke.FeasibilityUnresolved):
            smoke.validate_io_contract("yolov8n", {"input": dict(smoke.EXPECTED_INPUT), "output": {"name": "other", "shape": [1, 7, 8400], "dtype": "float32"}}, "fake")

    def test_ort_contract_rejects_gpu_or_mixed_provider(self):
        with self.assertRaises(smoke.FeasibilityUnresolved):
            smoke.validate_ort_contract({"providers_observed": ["CUDAExecutionProvider"], "input": dict(smoke.EXPECTED_INPUT)})
        with self.assertRaises(smoke.FeasibilityUnresolved):
            smoke.validate_ort_contract({"providers_observed": ["CPUExecutionProvider"], "input": {"name": "wrong"}})

    def test_descriptive_comparison_has_no_tolerance_or_pass_verdict(self):
        class TinyArray:
            shape = (1, 7, 8400)
            dtype = "float32"

            def astype(self, _dtype):
                return self

            def __sub__(self, _other):
                return self

        class TinyFinite:
            def all(self):
                return True

        class TinyDelta:
            def max(self):
                return 1

            def mean(self):
                return 0.1

        class TinyNumpy:
            float64 = "float64"

            @staticmethod
            def ascontiguousarray(value):
                return value

            @staticmethod
            def dtype(_name):
                return "float32"

            @staticmethod
            def isfinite(_value):
                return TinyFinite()

            @staticmethod
            def abs(_value):
                return TinyDelta()

            @staticmethod
            def array_equal(_left, _right):
                return False

        reference = TinyArray()
        observed = TinyArray()
        result = smoke.descriptive_primary_comparison(TinyNumpy, reference, observed, "yolov8n")
        self.assertEqual(result["assessment"], "descriptive_only")
        self.assertIsNone(result["tolerance"])
        self.assertFalse(result["exact_equal"])

    def test_external_fake_runtime_build_records_parser_flags_and_engine_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, evidence = smoke.build_engine(FakeTrt, Path(tmp) / "accepted.onnx", "yolov8n")
        self.assertIsInstance(engine, FakeEngine)
        self.assertEqual(evidence["builder"]["flags_observed"], {"FP16": True, "INT8": False, "TF32": False})
        self.assertEqual(evidence["parser"]["io"]["observed"]["output"]["shape"], [1, 7, 8400])
        self.assertFalse(evidence["builder"]["serialized_engine"]["published"])
        self.assertEqual(FakeTrt.Builder.last_config.workspace, 4 << 30)
        self.assertTrue(FakeTrt.Builder.last_config.timing_cache_empty)
        self.assertFalse(FakeTrt.Builder.last_config.ignore_mismatch)

    def test_external_fake_runtime_build_covers_yolo26_fixed_row_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            _engine, evidence = smoke.build_engine(FakeYolo26Trt, Path(tmp) / "accepted.onnx", "yolo26n")
        self.assertEqual(evidence["parser"]["io"]["observed"]["output"]["shape"], [1, 300, 6])
        self.assertEqual(evidence["engine"]["io"]["observed"]["output"]["shape"], [1, 300, 6])

    def test_parser_failure_is_hard_failure_with_error(self):
        with self.assertRaises(smoke.FeasibilityUnresolved) as raised:
            smoke.build_engine(FailingParserTrt, Path("accepted.onnx"), "yolov8n")
        self.assertIn("synthetic parser failure", str(raised.exception))

    def test_parent_module_does_not_import_tensor_rt(self):
        tree = ast.parse(Path(smoke.__file__).read_text(encoding="utf-8"))
        top_level_trt = [node for node in tree.body if isinstance(node, ast.Import) and any(alias.name == "tensorrt" for alias in node.names)]
        self.assertEqual(top_level_trt, [])
        self.assertNotIn("tensorrt", sys.modules)

    def test_dispatch_runs_each_model_once_and_preserves_order(self):
        calls = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            root.mkdir()
            plan = {"output_root": "out", "models": {}}

            def fake_run(command, **kwargs):
                model = command[command.index("--model") + 1]
                calls.append((model, kwargs["timeout"]))
                model_dir = root / "models" / model
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "model_report.json").write_text(json.dumps({"model": model, "status": "completed"}), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            rows, failed = smoke.dispatch_children(Path(tmp), plan, root, ["yolov8n", "yolo26n"], 17, run_fn=fake_run)
        self.assertFalse(failed)
        self.assertEqual([row["model"] for row in rows], ["yolov8n", "yolo26n"])
        self.assertEqual(calls, [("yolov8n", 17), ("yolo26n", 17)])

    def test_child_timeout_preserves_partial_files_and_does_not_retry(self):
        calls = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            root.mkdir()
            plan = {"output_root": "out", "models": {}}

            def fake_run(command, **_kwargs):
                model = command[command.index("--model") + 1]
                calls.append(model)
                model_dir = root / "models" / model
                model_dir.mkdir(parents=True, exist_ok=True)
                if model == "yolov8n":
                    (model_dir / "partial.jsonl").write_text("partial\n", encoding="utf-8")
                    raise subprocess.TimeoutExpired(command, 17, output="partial output", stderr="timeout")
                (model_dir / "failure.json").write_text(json.dumps({"model": model, "status": "failed"}), encoding="utf-8")
                return SimpleNamespace(returncode=1, stdout="failed", stderr="")

            rows, failed = smoke.dispatch_children(Path(tmp), plan, root, ["yolov8n", "yolo26n"], 17, run_fn=fake_run)
            first = json.loads((root / "models" / "yolov8n" / "failure.json").read_text(encoding="utf-8"))
        self.assertTrue(failed)
        self.assertEqual(calls, ["yolov8n", "yolo26n"])
        self.assertEqual(rows[0]["status"], "failed")
        self.assertTrue(any(path.endswith("models/yolov8n/partial.jsonl") for path in first["partial_files"]))
        self.assertTrue(first["no_silent_resume"])
        self.assertTrue(first["no_retry"])


if __name__ == "__main__":
    unittest.main()
