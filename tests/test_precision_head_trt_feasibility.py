import ast
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


class FakeDType:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return str(other) == self.name


class FakeMask:
    def __init__(self, value=True):
        self.value = value

    def all(self):
        return self.value

    def sum(self):
        return 0 if self.value else 1


class FakeArray:
    def __init__(self, shape, dtype="float32", payload=None):
        self.shape = tuple(shape)
        self.dtype = FakeDType(dtype) if isinstance(dtype, str) else dtype
        self._payload = payload or (f"{self.shape}:{self.dtype}".encode("ascii"))

    def copy(self):
        return FakeArray(self.shape, self.dtype, self._payload)

    def astype(self, dtype):
        return FakeArray(self.shape, str(dtype), self._payload)

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        keys = list(keys) + [slice(None)] * (len(self.shape) - len(keys))
        shape = []
        for size, item in zip(self.shape, keys):
            if isinstance(item, int):
                continue
            if isinstance(item, slice):
                start, stop, step = item.indices(size)
                shape.append(len(range(start, stop, step)))
            else:
                shape.append(size)
        return FakeArray(shape, self.dtype, self._payload)

    def __sub__(self, _other):
        return FakeArray(self.shape, "float64", self._payload)

    def __ne__(self, _other):
        return FakeMask(True)

    def max(self):
        return 1.0

    def mean(self):
        return 0.1

    def tobytes(self, order="C"):
        return self._payload

    def mutate(self):
        self._payload = b"mutated-by-consumer"


class FakeNumpy:
    float64 = "float64"

    @staticmethod
    def ascontiguousarray(value):
        return value

    @staticmethod
    def dtype(name):
        return FakeDType(str(name))

    @staticmethod
    def isfinite(_value):
        return FakeMask(True)

    @staticmethod
    def array_equal(left, right):
        return left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()

    @staticmethod
    def abs(value):
        return value


class FakeTensor:
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class NonfiniteNumpy(FakeNumpy):
    @staticmethod
    def isfinite(_value):
        return FakeMask(False)


class FakeOrt:
    class SessionOptions:
        pass

    class GraphOptimizationLevel:
        ORT_ENABLE_BASIC = "basic"

    class ExecutionMode:
        ORT_SEQUENTIAL = "sequential"

    class InferenceSession:
        def __new__(cls, _path, sess_options=None, providers=None):
            return SimpleNamespace(
                get_providers=lambda: ["CPUExecutionProvider"],
                get_inputs=lambda: [SimpleNamespace(name="images", type="tensor(float)", shape=[1, 3, 640, 640])],
                get_outputs=lambda: [SimpleNamespace(name="output0", type="tensor(float)", shape=list(smoke.EXPECTED_OUTPUT_SHAPES[cls.model]))],
            )

        model = "yolov8n"


class PrecisionHeadTrtFeasibilityTests(unittest.TestCase):
    def _fake_child_plan(self, root, model):
        checkpoint = root / f"{model}.pt"
        onnx = root / f"{model}.onnx"
        config = root / f"{model}.json"
        checkpoint.write_bytes(f"checkpoint-{model}".encode("ascii"))
        onnx.write_bytes(f"onnx-{model}".encode("ascii"))
        config.write_text(json.dumps({"model": model}), encoding="utf-8")
        checkpoint_sha = smoke.sha256_file(checkpoint)
        onnx_sha = smoke.sha256_file(onnx)
        old_sha = smoke.EXPECTED_ONNX_SHA256[model]
        smoke.EXPECTED_ONNX_SHA256[model] = onnx_sha
        source = root / f"{model}.jpg"
        source.write_bytes(b"fixture-image")
        source_sha = smoke.sha256_file(source)
        rows = [{"selection": "u42", "manifest_order": index, "image": name, "expected_sha256": source_sha, "expected_bytes": source.stat().st_size} for index, name in enumerate(smoke.FIXTURE_NAMES)]
        plan = {
            "output_root": "out",
            "config": {"path": config.name},
            "runtime_requirements": {"device_argument": "0", "logical_cuda_index": 0},
            "gpu_preflight": {"identity": {"uuid": "GPU-test", "name": "Fake GPU"}, "confirmed_desktop": [], "confirmed_background": []},
            "fixture": {"forward_images": rows},
            "models": {model: {"checkpoint": {"expected": {"path": checkpoint.name, "sha256": checkpoint_sha}}, "onnx": {"path": onnx.name, "expected_sha256": onnx_sha}}},
        }
        return plan, source, old_sha

    def _fake_snapshot(self, _desktop, _background, _device):
        return {"gpu_identity": {"uuid": "GPU-test", "name": "Fake GPU"}, "process_guard": {"blocked_processes": [], "unmatched_confirmations": [], "background_workload": []}}

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
            smoke.validate_ort_contract({"providers_observed": ["CUDAExecutionProvider"], "input": {"name": "images", "type": "tensor(float)", "shape": [1, 3, 640, 640]}}, "yolov8n")
        with self.assertRaises(smoke.FeasibilityUnresolved):
            smoke.validate_ort_contract({"providers_observed": ["CPUExecutionProvider"], "input": {"name": "wrong"}}, "yolov8n")

    def test_internal_child_cli_accepts_concrete_models_but_public_selection_stays_all(self):
        args = smoke.build_parser().parse_args(["--child", "--repo-root", ".", "--plan", "plan.json", "--model", "yolov8n"])
        self.assertEqual(args.model, "yolov8n")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            smoke.build_parser().parse_args(["--child", "--repo-root", ".", "--plan", "plan.json", "--model", "not-a-model"])
        with self.assertRaises(ValueError):
            smoke.selected_models("yolov8n")

    def test_ort_producer_contract_is_tensor_float_and_cpu_for_both_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model in smoke.MODEL_CHOICES:
                FakeOrt.InferenceSession.model = model
                _session, contract = smoke.numeric.run_onnx_session(Path(tmp) / f"{model}.onnx", model, {"ort": FakeOrt, "np": FakeNumpy})
                smoke.validate_ort_contract(contract, model)
                self.assertEqual(contract["input"]["type"], "tensor(float)")
                self.assertEqual(contract["output"]["type"], "tensor(float)")

    def test_pointer_binding_and_nonfinite_output_are_hard_failures(self):
        class BadContext:
            def set_tensor_address(self, _name, _pointer):
                return False

        with self.assertRaises(smoke.FeasibilityUnresolved):
            smoke.bind_tensor_addresses(BadContext(), "yolov8n", 1, 2)
        with self.assertRaises(smoke.FeasibilityUnresolved):
            smoke._finite_output(NonfiniteNumpy, FakeArray((1, 7, 8400)), "yolov8n", "synthetic")

    def test_integrated_child_fixture_both_models_preserves_raw_hashes_and_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model in smoke.MODEL_CHOICES:
                plan, source, old_sha = self._fake_child_plan(root, model)
                try:
                    class Session:
                        def run(self, _outputs, _feeds):
                            return [FakeArray(smoke.EXPECTED_OUTPUT_SHAPES[model])]

                    def fake_loader(_config, _device, model=model):
                        return {"np": FakeNumpy, "trt": FakeYolo26Trt if model == "yolo26n" else FakeTrt, "packages": {"observed": {"fake": "runtime"}}}

                    def fake_trace(_path, _runtime, stride=32):
                        return FakeTensor(FakeArray((1, 3, 640, 640))), {"original_image_size_hw": [480, 640], "stride": stride}

                    def fake_postprocess(_model, primary, _trace, _runtime):
                        primary.mutate()
                        return {"count": 0, "xyxy": [], "confidence": [], "class_id": []}

                    def fake_trt(_engine, _runtime, _input, _model, _device):
                        return FakeArray(smoke.EXPECTED_OUTPUT_SHAPES[model])

                    state = smoke._child_state(model)
                    out_dir = root / "out" / "models" / model
                    with patch.object(smoke.numeric, "verify_bound_image", return_value={}), patch.object(smoke.numeric, "resolve_bound_source_image", return_value=source), patch.object(smoke.numeric, "trace_preprocess", side_effect=fake_trace), patch.object(smoke.numeric, "run_onnx_session", return_value=(Session(), {"providers_observed": ["CPUExecutionProvider"], "input": {"name": "images", "type": "tensor(float)", "shape": [1, 3, 640, 640]}, "output": {"name": "output0", "type": "tensor(float)", "shape": list(smoke.EXPECTED_OUTPUT_SHAPES[model])}})), patch.object(smoke.bridge, "application_postprocess", side_effect=fake_postprocess), patch.object(smoke, "execute_trt_once", side_effect=fake_trt):
                        report = smoke.run_model_child(root, plan, model, out_dir, state=state, runtime_loader=fake_loader, snapshot_fn=self._fake_snapshot)
                    self.assertEqual(report["status"], "completed")
                    self.assertTrue(state["parser_attempted"] and state["parser_completed"])
                    self.assertTrue(state["build_attempted"] and state["build_completed"])
                    records = [json.loads(line) for line in (out_dir / "smoke_records.jsonl").read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(len(records), 8)
                    for record in records:
                        self.assertEqual(record["tensorrt"]["sha256"], record["comparison"]["observed_raw"]["sha256"])
                        self.assertEqual(record["onnx_cpu_reference"]["sha256"], record["comparison"]["reference_raw"]["sha256"])
                        self.assertEqual(record["onnx_cpu_reference"]["sha256"], record["private_materialization"]["onnx_reference_bytes"]["sha256"])
                        self.assertEqual(record["tensorrt"]["raw_units"], "immutable pre-postprocess TensorRT output")
                    self.assertEqual(state["forward_counts"]["trt_application_enqueue"]["attempted"], 8)
                    self.assertEqual(state["forward_counts"]["trt_application_enqueue"]["completed"], 8)
                    self.assertEqual(state["forward_counts"]["onnx_cpu_reference_call"]["attempted"], 8)
                    self.assertEqual(state["forward_counts"]["onnx_cpu_reference_call"]["completed"], 8)
                    self.assertEqual(state["ownership_status"], "released_after_final_synchronize")
                finally:
                    smoke.EXPECTED_ONNX_SHA256[model] = old_sha

    def test_integrated_child_second_image_failure_persists_attempted_completed_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = "yolov8n"
            plan, source, old_sha = self._fake_child_plan(root, model)
            try:
                class FailingSession:
                    def __init__(self):
                        self.calls = 0

                    def run(self, _outputs, _feeds):
                        self.calls += 1
                        if self.calls == 2:
                            raise RuntimeError("synthetic second-image ORT failure")
                        return [FakeArray(smoke.EXPECTED_OUTPUT_SHAPES[model])]

                session = FailingSession()

                def fake_loader(_config, _device):
                    return {"np": FakeNumpy, "trt": FakeTrt, "packages": {"observed": {"fake": "runtime"}}}

                def fake_trace(_path, _runtime, stride=32):
                    return FakeTensor(FakeArray((1, 3, 640, 640))), {"original_image_size_hw": [480, 640], "stride": stride}

                def fake_postprocess(_model, primary, _trace, _runtime):
                    primary.mutate()
                    return {"count": 0, "xyxy": [], "confidence": [], "class_id": []}

                def fake_trt(_engine, _runtime, _input, _model, _device):
                    return FakeArray(smoke.EXPECTED_OUTPUT_SHAPES[model])

                state = smoke._child_state(model)
                out_dir = root / "out" / "models" / model
                with patch.object(smoke.numeric, "verify_bound_image", return_value={}), patch.object(smoke.numeric, "resolve_bound_source_image", return_value=source), patch.object(smoke.numeric, "trace_preprocess", side_effect=fake_trace), patch.object(smoke.numeric, "run_onnx_session", return_value=(session, {"providers_observed": ["CPUExecutionProvider"], "input": {"name": "images", "type": "tensor(float)", "shape": [1, 3, 640, 640]}, "output": {"name": "output0", "type": "tensor(float)", "shape": [1, 7, 8400]}})), patch.object(smoke.bridge, "application_postprocess", side_effect=fake_postprocess), patch.object(smoke, "execute_trt_once", side_effect=fake_trt):
                    with self.assertRaises(RuntimeError):
                        smoke.run_model_child(root, plan, model, out_dir, state=state, runtime_loader=fake_loader, snapshot_fn=self._fake_snapshot)
                failure = smoke.child_failure(plan, model, state, RuntimeError("synthetic second-image ORT failure"), root / "out")
                self.assertEqual(state["forward_counts"]["trt_application_enqueue"], {"attempted": 2, "completed": 2})
                self.assertEqual(state["forward_counts"]["onnx_cpu_reference_call"], {"attempted": 2, "completed": 1})
                self.assertEqual(state["records_written"], 1)
                self.assertTrue(any(path.endswith("smoke_records.jsonl") for path in failure["partial_files"]))
                self.assertEqual(failure["audit_flags"]["tensorrt_build_performed"], True)
            finally:
                smoke.EXPECTED_ONNX_SHA256[model] = old_sha

    def test_terminal_inventory_includes_manifest_report_logs_and_child_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in ("smoke_plan.json", "smoke_manifest.json", "report.md", "logs/yolov8n.log", "models/yolov8n/child_state.json", "models/yolov8n/model_report.json"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            inventory = smoke.publishable_inventory(root)
        self.assertEqual(inventory, ["logs/yolov8n.log", "models/yolov8n/child_state.json", "models/yolov8n/model_report.json", "report.md", "smoke_manifest.json", "smoke_plan.json"])

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
        self.assertIsInstance(engine, smoke.OwnedEngine)
        self.assertIsInstance(engine.engine, FakeEngine)
        self.assertEqual(evidence["builder"]["flags_observed"], {"FP16": True, "INT8": False, "TF32": False})
        self.assertEqual(evidence["parser"]["io"]["observed"]["output"]["shape"], [1, 7, 8400])
        self.assertFalse(evidence["builder"]["serialized_engine"]["published"])
        self.assertEqual(FakeTrt.Builder.last_config.workspace, 4 << 30)
        self.assertTrue(FakeTrt.Builder.last_config.timing_cache_empty)
        self.assertFalse(FakeTrt.Builder.last_config.ignore_mismatch)
        engine.close()

    def test_external_fake_runtime_build_covers_yolo26_fixed_row_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, evidence = smoke.build_engine(FakeYolo26Trt, Path(tmp) / "accepted.onnx", "yolo26n")
        self.assertEqual(evidence["parser"]["io"]["observed"]["output"]["shape"], [1, 300, 6])
        self.assertEqual(evidence["engine"]["io"]["observed"]["output"]["shape"], [1, 300, 6])
        engine.close()

    def test_parser_failure_is_hard_failure_with_error(self):
        with self.assertRaises(smoke.FeasibilityUnresolved) as raised:
            smoke.build_engine(FailingParserTrt, Path("accepted.onnx"), "yolov8n")
        self.assertIn("synthetic parser failure", str(raised.exception))

    def test_owned_engine_keeps_logger_and_runtime_until_explicit_close(self):
        class LifetimeTrt(FakeTrt):
            logger_ref = None
            runtime_ref = None

            class Logger(FakeTrt.Logger):
                def __init__(self, level):
                    super().__init__(level)
                    LifetimeTrt.logger_ref = weakref.ref(self)

            class Runtime(FakeTrt.Runtime):
                def __init__(self, logger):
                    super().__init__(logger)
                    LifetimeTrt.runtime_ref = weakref.ref(self)

        with tempfile.TemporaryDirectory() as tmp:
            owned, _evidence = smoke.build_engine(LifetimeTrt, Path(tmp) / "accepted.onnx", "yolov8n")
            self.assertIsNotNone(LifetimeTrt.logger_ref())
            self.assertIsNotNone(LifetimeTrt.runtime_ref())
            owned.close()
            self.assertTrue(owned.closed)
        import gc
        gc.collect()
        self.assertIsNone(LifetimeTrt.logger_ref())
        self.assertIsNone(LifetimeTrt.runtime_ref())

    def test_build_failure_releases_runtime_and_logger(self):
        class FailingRuntimeTrt(FakeTrt):
            logger_ref = None
            runtime_ref = None

            class Logger(FakeTrt.Logger):
                def __init__(self, level):
                    super().__init__(level)
                    FailingRuntimeTrt.logger_ref = weakref.ref(self)

            class Runtime(FakeTrt.Runtime):
                def __init__(self, logger):
                    super().__init__(logger)
                    FailingRuntimeTrt.runtime_ref = weakref.ref(self)

                def deserialize_cuda_engine(self, _serialized):
                    return None

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(smoke.FeasibilityUnresolved):
                smoke.build_engine(FailingRuntimeTrt, Path(tmp) / "accepted.onnx", "yolov8n")
        import gc
        gc.collect()
        self.assertIsNone(FailingRuntimeTrt.logger_ref())
        self.assertIsNone(FailingRuntimeTrt.runtime_ref())

    def test_primary_child_error_survives_cleanup_error(self):
        class CleanupFailure:
            def close(self):
                raise RuntimeError("synthetic cleanup failure")

        state = smoke._child_state("yolov8n")
        state["_engine_owner"] = CleanupFailure()
        with patch.object(smoke, "_run_model_child_impl", side_effect=RuntimeError("synthetic primary failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic primary failure"):
                smoke.run_model_child(Path("."), {}, "yolov8n", Path("."), state=state)
        self.assertEqual(state["cleanup_error"]["error"], "synthetic cleanup failure")

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
            plan = {"output_root": "out", "models": {}, "runtime_requirements": {"logical_cuda_index": 0}}

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
            plan = {"output_root": "out", "models": {}, "runtime_requirements": {"logical_cuda_index": 0}}

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
        self.assertEqual(calls, ["yolov8n"])
        self.assertEqual(rows[0]["status"], "failed")
        self.assertTrue(any(path.endswith("models/yolov8n/partial.jsonl") for path in first["partial_files"]))
        self.assertTrue(first["no_silent_resume"])
        self.assertTrue(first["no_retry"])

    def test_timeout_with_truncated_state_preserves_corruption_and_reports_unknown_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            root.mkdir()
            plan = {"output_root": "out", "models": {}, "runtime_requirements": {"logical_cuda_index": 0}}

            def fake_run(command, **_kwargs):
                model_dir = root / "models" / command[command.index("--model") + 1]
                (model_dir / "child_state.json").write_text('{"stage":', encoding="utf-8")
                raise subprocess.TimeoutExpired(command, 17, output=b"partial stdout", stderr=b"partial stderr")

            rows, failed = smoke.dispatch_children(Path(tmp), plan, root, ["yolov8n", "yolo26n"], 17, run_fn=fake_run)
            failure = json.loads((root / "models" / "yolov8n" / "failure.json").read_text(encoding="utf-8"))
            state_bytes = (root / "models" / "yolov8n" / "child_state.json").read_text(encoding="utf-8")
        self.assertTrue(failed)
        self.assertEqual(len(rows), 1)
        self.assertEqual(state_bytes, '{"stage":')
        self.assertEqual(failure["completion_status"], "unknown_after_timeout")
        self.assertIsNone(failure["forward_counts"]["trt_application_enqueue"]["attempted"])
        self.assertEqual(failure["state_recovery"]["status"], "unknown")

    def test_timeout_with_valid_partial_state_preserves_observed_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            root.mkdir()
            plan = {"output_root": "out", "models": {}, "runtime_requirements": {"logical_cuda_index": 0}}

            def fake_run(command, **_kwargs):
                model_dir = root / "models" / command[command.index("--model") + 1]
                state = smoke._child_state("yolov8n")
                state.update({"stage": "build", "records_written": 1, "forward_counts": {"trt_application_enqueue": {"attempted": 2, "completed": 2}, "onnx_cpu_reference_call": {"attempted": 1, "completed": 1}, "native_forward": {"attempted": 0, "completed": 0}}})
                (model_dir / "child_state.json").write_text(json.dumps(state), encoding="utf-8")
                raise subprocess.TimeoutExpired(command, 17, output=b"partial", stderr=b"")

            rows, failed = smoke.dispatch_children(Path(tmp), plan, root, ["yolov8n", "yolo26n"], 17, run_fn=fake_run)
            failure = rows[0]
        self.assertTrue(failed)
        self.assertEqual(failure["state_recovery"]["status"], "valid_partial_state")
        self.assertEqual(failure["forward_counts"]["trt_application_enqueue"], {"attempted": 2, "completed": 2})
        self.assertEqual(failure["records_written"], 1)

    def test_timeout_does_not_overwrite_existing_failure_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            root.mkdir()
            plan = {"output_root": "out", "models": {}, "runtime_requirements": {"logical_cuda_index": 0}}
            existing = {"model": "yolov8n", "status": "failed", "error": "existing-terminal-record"}

            def fake_run(command, **_kwargs):
                model_dir = root / "models" / command[command.index("--model") + 1]
                (model_dir / "failure.json").write_text(json.dumps(existing), encoding="utf-8")
                raise subprocess.TimeoutExpired(command, 17, output=b"partial", stderr=b"")

            rows, failed = smoke.dispatch_children(Path(tmp), plan, root, ["yolov8n", "yolo26n"], 17, run_fn=fake_run)
            preserved = json.loads((root / "models" / "yolov8n" / "failure.json").read_text(encoding="utf-8"))
        self.assertTrue(failed)
        self.assertEqual(len(rows), 1)
        self.assertEqual(preserved, existing)


if __name__ == "__main__":
    unittest.main()
