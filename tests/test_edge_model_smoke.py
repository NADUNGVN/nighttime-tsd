import ast
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from edge_readiness import e2_model_smoke as smoke  # noqa: E402
from edge_readiness.e2_output_compare import compare_output0  # noqa: E402
from edge_readiness.edge_errors import AdapterError  # noqa: E402
from edge_readiness.jetson_adapter import HostTensor  # noqa: E402
from edge_readiness.jetson_runtime_provider import OwnedBuffers, TensorRTProvider  # noqa: E402


def _pack(values):
    return struct.pack("<{}f".format(len(values)), *values)


class FakeConfig:
    def __init__(self):
        self.flags = []
        self.workspace = None
        self.cleared = []

    def set_memory_pool_limit(self, pool, value):
        self.workspace = (pool, value)

    def set_flag(self, flag):
        self.flags.append(flag)

    def clear_flag(self, flag):
        self.cleared.append(flag)


class FakeParser:
    def __init__(self, ok=True):
        self.ok = ok
        self.num_errors = 0 if ok else 1

    def parse_from_file(self, path):
        self.path = path
        return self.ok

    def get_error(self, index):
        return "fake parser error {}".format(index)


class FakeContext:
    def __init__(self, engine, fail_on_call=None):
        self.engine = engine
        self.calls = 0
        self.fail_on_call = fail_on_call

    def execute_async_v2(self, bindings, stream_handle):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("injected enqueue failure")
        self.engine.current_output = self.engine.outputs[self.calls - 1]
        return True


class FakeEngine:
    num_bindings = 2

    def __init__(self, outputs, fail_on_call=None):
        self.outputs = outputs
        self.current_output = outputs[0]
        self.context = FakeContext(self, fail_on_call)

    def get_binding_name(self, index):
        return ("images", "output0")[index]

    def get_binding_shape(self, index):
        return ((1, 3, 640, 640), (1, 7, 8400))[index]

    def get_binding_dtype(self, index):
        return "float32"

    def binding_is_input(self, index):
        return index == 0

    def get_location(self, index):
        return "TensorLocation.DEVICE"

    def create_execution_context(self):
        return self.context


class FakeTensorRT:
    __version__ = "8.5.2.2"

    class Logger:
        ERROR = 0

        def __init__(self, level):
            self.level = level

    class NetworkDefinitionCreationFlag:
        EXPLICIT_BATCH = 0

    class MemoryPoolType:
        WORKSPACE = 0

    class BuilderFlag:
        FP16 = 1
        TF32 = 2

    def __init__(self, outputs=None, parse_ok=True, fail_on_call=None):
        self.outputs = outputs or [_pack([0.0] * smoke.OUTPUT_ELEMENTS)] * 3
        self.parse_ok = parse_ok
        self.fail_on_call = fail_on_call
        self.configs = []
        self.engine = FakeEngine(self.outputs, fail_on_call)

    def Builder(self, logger):
        module = self

        class Builder:
            def create_network(self, flags):
                module.network_flags = flags
                return object()

            def create_builder_config(self):
                config = FakeConfig()
                module.configs.append(config)
                return config

            def build_serialized_network(self, network, config):
                module.builder_network = network
                module.builder_config = config
                return b"fake-serialized-engine"

        return Builder()

    def OnnxParser(self, network, logger):
        return FakeParser(self.parse_ok)

    def Runtime(self, logger):
        module = self

        class Runtime:
            def deserialize_cuda_engine(self, payload):
                module.deserialized_payload = payload
                return module.engine

        return Runtime()


class FakeMemory:
    def __init__(self, engine):
        self.engine = engine
        self.next_pointer = 1000
        self.freed = []

    def allocate_device(self, nbytes, name):
        pointer = self.next_pointer
        self.next_pointer += 1
        return pointer

    def copy_host_to_device(self, payload, pointer, stream_handle):
        self.last_input = payload

    def copy_device_to_host(self, pointer, destination, stream_handle):
        destination[:] = self.engine.current_output

    def synchronize(self, stream_handle):
        pass

    def free_device(self, pointer):
        self.freed.append(pointer)


class FakeStream:
    handle = 77

    def synchronize(self, stage):
        pass

    def close(self):
        pass


class FakeExecution:
    def __init__(self, descriptor, adapter, buffers, provider, stream):
        self.descriptor = descriptor
        self.adapter = adapter
        self.buffers = buffers
        self.provider = provider
        self.stream = stream

    def infer(self, host_input):
        return self.adapter.infer(host_input)

    def close(self):
        self.buffers.free()
        self.provider.close()
        self.stream.close()


class ProductionFakeTarget:
    def __init__(self, outputs, parse_ok=True, fail_on_call=None, timeout=False):
        self.trt = FakeTensorRT(outputs, parse_ok=parse_ok, fail_on_call=fail_on_call)
        self.timeout = timeout
        self.calls = []

    def preflight(self):
        self.calls.append("preflight")
        return {"target_id": "E2", "runtime_version": "8.5.2.2", "execution_api": "legacy_binding_execute_async_v2", "hostname": smoke.EXPECTED_HOSTNAME, "machine": smoke.EXPECTED_ARCHITECTURE, "hardware_model": smoke.EXPECTED_MODEL_TOKEN}

    def build(self, onnx_path, engine_path, workspace_bytes):
        self.calls.append("build")
        if self.timeout:
            raise smoke.TargetTimeout("STAGE_TIMEOUT", "injected build timeout", {"stage": "build"})
        return smoke.TensorRTOnnxBuilder(module_loader=lambda _name: self.trt).build(onnx_path, engine_path, workspace_bytes)

    def open_execution(self, engine):
        self.calls.append("open_execution")
        provider = TensorRTProvider(smoke.AdapterConfig("E2", "8.5.2.2"), module_loader=lambda _name: self.trt)
        descriptor = provider.load_engine(engine.path.read_bytes(), engine.sha256)
        stream = FakeStream()
        memory = FakeMemory(self.trt.engine)
        buffers = OwnedBuffers(descriptor, memory, stream.handle)
        adapter = provider.create_adapter(stream, buffers)
        return FakeExecution(descriptor, adapter, buffers, provider, stream)


def make_bundle(temp_root):
    root = Path(temp_root)
    private = root / "private"
    (private / "onnx_export").mkdir(parents=True)
    (private / "inputs").mkdir()
    (private / "native_reference").mkdir()
    (private / "onnx_reference").mkdir()
    onnx_path = private / "onnx_export" / "best.onnx"
    onnx_path.write_bytes(b"accepted-onnx")
    records = {}
    onnx_values = {}
    native_values = {}
    for index, image_id in enumerate(smoke.FIXTURE_IDS):
        input_values = [0.0] * (smoke.INPUT_NBYTES // 4)
        input_values[0] = 0.1 * (index + 1)
        input_payload = _pack(input_values)
        native = [0.0] * smoke.OUTPUT_ELEMENTS
        native[0] = 0.1 * (index + 1)
        onnx = list(native)
        if image_id == "00028":
            onnx[smoke.OUTPUT_SHAPE[2] + 8002] += 2e-5
        native_payload = _pack(native)
        onnx_payload = _pack(onnx)
        input_path = private / "inputs" / (image_id + ".bin")
        native_path = private / "native_reference" / (image_id + ".bin")
        onnx_output_path = private / "onnx_reference" / (image_id + ".bin")
        input_path.write_bytes(input_payload)
        native_path.write_bytes(native_payload)
        onnx_output_path.write_bytes(onnx_payload)
        records[image_id] = {
            "input": {"path": str(input_path), "nbytes": len(input_payload), "sha256": hashlib.sha256(input_payload).hexdigest(), "shape": list(smoke.INPUT_SHAPE), "dtype": "float32", "byteorder": "little"},
            "native": {"path": str(native_path), "nbytes": len(native_payload), "sha256": hashlib.sha256(native_payload).hexdigest(), "shape": list(smoke.OUTPUT_SHAPE), "dtype": "float32", "byteorder": "little"},
            "onnx": {"path": str(onnx_output_path), "nbytes": len(onnx_payload), "sha256": hashlib.sha256(onnx_payload).hexdigest(), "shape": list(smoke.OUTPUT_SHAPE), "dtype": "float32", "byteorder": "little"},
        }
        native_values[image_id] = native
        onnx_values[image_id] = onnx
    declared = {image_id: compare_output0(native_values[image_id], onnx_values[image_id], smoke.STRICT_SOURCE_POLICY).as_dict() for image_id in smoke.FIXTURE_IDS}
    manifest = {
        "schema_version": smoke.SOURCE_MANIFEST_SCHEMA,
        "status": "execution_complete_numerical_fail",
        "execution_status": "complete",
        "onnx": {"path": str(onnx_path), "sha256": smoke.SOURCE_ONNX_SHA256, "contract": {"input_name": "images", "output_name": "output0", "input_shape": list(smoke.INPUT_SHAPE), "output_shape": list(smoke.OUTPUT_SHAPE), "input_dtype": "float32", "output_dtype": "float32", "opset": 17, "forbidden_nodes": []}},
        "records": records,
        "comparisons": declared,
    }
    public = root / "public"
    public.mkdir()
    manifest_path = public / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sha = lambda path: smoke.SOURCE_ONNX_SHA256 if path == onnx_path else hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    original_sha = smoke.CANONICAL_SOURCE_MANIFEST_SHA256
    smoke.CANONICAL_SOURCE_MANIFEST_SHA256 = manifest_sha
    with patch.object(smoke, "file_sha256", side_effect=sha):
        try:
            bundle = smoke.load_source_bundle(manifest_path, root)
            bundle.test_manifest_sha256 = manifest_sha
            return bundle
        finally:
            smoke.CANONICAL_SOURCE_MANIFEST_SHA256 = original_sha


class ModelSmokeTests(unittest.TestCase):
    def test_canonical_manifest_hash_matches_raw_git_blob(self):
        blob = subprocess.run(["git", "cat-file", "blob", smoke.CANONICAL_SOURCE_COMMIT + ":results/edge_readiness_v1/e2l1-013-source-v2/public/manifest.json"], capture_output=True, check=True).stdout
        self.assertEqual(len(blob), 20346)
        self.assertEqual(hashlib.sha256(blob).hexdigest(), smoke.CANONICAL_SOURCE_MANIFEST_SHA256)

    def test_archive_provenance_works_without_git_and_rejects_changed_helper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "export"
            for relative in smoke.PROVENANCE_FILES + smoke.PACKAGE_FILES:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            archive_manifest = Path(temp) / "code-export.json"
            payload = {"schema_version": "e2l1-code-export-v1", "commit": "34a542b2f787d7ef60dc3d3125cecad78e0c16f9", "files": {relative: smoke.file_sha256(root / relative) for relative in smoke.PROVENANCE_FILES}, "package_files": {relative: smoke.file_sha256(root / relative) for relative in smoke.PACKAGE_FILES}}
            archive_manifest.write_text(json.dumps(payload), encoding="utf-8")
            observed = smoke.code_provenance(payload["commit"], archive_manifest, root)
            self.assertEqual(observed["provenance_mode"], "reviewed_archive_manifest")
            (root / "scripts/edge_readiness/edge_errors.py").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(AdapterError, "CODE_ARCHIVE_BYTES_MISMATCH"):
                smoke.code_provenance(payload["commit"], archive_manifest, root)

    def test_parent_uses_production_child_stage_contract_and_publishes_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = make_bundle(temp)

            class ChildContract:
                def build(self, source_bundle, engine_path, timeout_seconds):
                    engine_path.parent.mkdir(parents=True, exist_ok=True)
                    engine_path.write_bytes(b"child-engine")
                    artifact = smoke.EngineArtifact(engine_path, hashlib.sha256(b"child-engine").hexdigest(), len(b"child-engine"), "8.5.2.2", {"workspace_bytes": smoke.WORKSPACE_BYTES, "fp16_enabled": True})
                    return artifact, {"parse_attempted": 1, "parse_completed": 1, "build_attempted": 1, "build_completed": 1}, engine_path.parent / "build_events.jsonl"

                def infer(self, source_bundle, engine, target_dir, timeout_seconds):
                    target_dir.mkdir(parents=True)
                    outputs = {}
                    for image_id in smoke.FIXTURE_IDS:
                        payload = source_bundle.onnx_outputs[image_id].path.read_bytes()
                        path = target_dir / (image_id + ".bin")
                        path.write_bytes(payload)
                        outputs[image_id] = {"path": str(path), "nbytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "shape": list(smoke.OUTPUT_SHAPE), "dtype": "float32"}
                    return outputs, {"engine_load_attempted": 1, "engine_load_completed": 1, "enqueues_attempted": 3, "enqueues_completed": 3, "output_copies_attempted": 3, "output_copies_completed": 3}, target_dir.parent / "inference_events.jsonl"

            target = ProductionFakeTarget([])
            code, manifest = smoke.ModelSmokeRunner(bundle, Path(temp) / "out", target, execute=True, stage_executor=ChildContract()).run()
            self.assertEqual(code, 0)
            self.assertEqual(manifest["attempted_completed"]["parse_completed"], 1)
            self.assertIn("code_provenance", manifest)
            self.assertIn("stage_evidence", manifest)

    def test_child_entry_records_parse_and_build_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            onnx = root / "source.onnx"
            engine = root / "engine.plan"
            result = root / "result.json"
            events = root / "events.jsonl"
            onnx.write_bytes(b"onnx")

            class ChildRuntime:
                def __init__(self, stage_observer=None):
                    self.runtime_version = "8.5.2.2"

                def preflight(self):
                    return {}

            class ChildBuilder:
                last_event = {"parser_completed": True, "build_completed": True}

                def __init__(self, stage_observer=None):
                    self.stage_observer = stage_observer
                    self.last_event = {"parser_completed": True, "build_completed": True}

                def build(self, onnx_path, engine_path, workspace):
                    for event in ("parser_attempted", "parser_completed", "build_attempted", "build_completed"):
                        if self.stage_observer:
                            self.stage_observer(event)
                    engine_path.write_bytes(b"engine")
                    return smoke.EngineArtifact(engine_path, hashlib.sha256(b"engine").hexdigest(), 6, "8.5.2.2", {"fp16_enabled": True})

            with patch.object(smoke, "E2TensorRTRuntime", ChildRuntime), patch.object(smoke, "TensorRTOnnxBuilder", ChildBuilder):
                self.assertEqual(smoke._child_stage_main(["--child-stage", "build", "--onnx", str(onnx), "--engine", str(engine), "--result", str(result), "--events", str(events)]), 0)
            self.assertEqual(json.loads(result.read_text(encoding="utf-8"))["counters"]["parse_completed"], 1)
            self.assertIn("parser_completed", events.read_text(encoding="utf-8"))

            failed_result = root / "failed-result.json"
            failed_events = root / "failed-events.jsonl"

            class FailingBuilder(ChildBuilder):
                last_event = {"parser_completed": True, "build_completed": False}

                def __init__(self, stage_observer=None):
                    super().__init__(stage_observer)
                    self.last_event = {"parser_completed": True, "build_completed": False}

                def build(self, onnx_path, engine_path, workspace):
                    if self.stage_observer:
                        self.stage_observer("parser_attempted")
                        self.stage_observer("parser_completed")
                        self.stage_observer("build_attempted")
                    raise smoke.ModelSmokeError("ENGINE_BUILD_FAILED", "injected build failure")

            with patch.object(smoke, "E2TensorRTRuntime", ChildRuntime), patch.object(smoke, "TensorRTOnnxBuilder", FailingBuilder):
                self.assertEqual(smoke._child_stage_main(["--child-stage", "build", "--onnx", str(onnx), "--engine", str(root / "failed.plan"), "--result", str(failed_result), "--events", str(failed_events)]), 2)
            failed = json.loads(failed_result.read_text(encoding="utf-8"))
            self.assertEqual(failed["counters"]["parse_completed"], 1)
            self.assertEqual(failed["counters"]["build_completed"], 0)

    def test_parent_reconciles_partial_outputs_and_counters_after_child_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = make_bundle(temp)

            class TimedChild:
                def build(self, source_bundle, engine_path, timeout_seconds):
                    engine_path.parent.mkdir(parents=True, exist_ok=True)
                    engine_path.write_bytes(b"engine")
                    return smoke.EngineArtifact(engine_path, hashlib.sha256(b"engine").hexdigest(), 6, "8.5.2.2", {}), {"parse_attempted": 1, "parse_completed": 1, "build_attempted": 1, "build_completed": 1}, engine_path.parent / "build_events.jsonl"

                def infer(self, source_bundle, engine, target_dir, timeout_seconds):
                    target_dir.mkdir(parents=True)
                    partial = target_dir / "00006.bin"
                    partial.write_bytes(source_bundle.onnx_outputs["00006"].path.read_bytes())
                    raise smoke.TargetTimeout("STAGE_TIMEOUT", "injected parent-visible timeout", {"termination_confirmed": True, "completion": "unknown", "counters": {"engine_load_attempted": 1, "engine_load_completed": 1, "enqueues_attempted": 2, "enqueues_completed": 1, "output_copies_attempted": 1, "output_copies_completed": 1}})

            code, failure = smoke.ModelSmokeRunner(bundle, Path(temp) / "out", ProductionFakeTarget([]), execute=True, stage_executor=TimedChild()).run()
            self.assertEqual(code, 2)
            self.assertEqual(failure["execution_state"], "unknown")
            self.assertEqual(failure["attempted_completed"]["enqueues_attempted"], 2)
            self.assertEqual(failure["provenance"]["target_outputs"]["00006"]["completion"], "partial")
            self.assertTrue(failure["error"]["details"]["termination_confirmed"])

    def test_event_only_timeout_reconciles_counters_and_unknown_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = make_bundle(temp)
            event_path = Path(temp) / "out" / "private" / "inference_events.jsonl"

            class EventOnlyChild:
                def build(self, source_bundle, engine_path, timeout_seconds):
                    engine_path.parent.mkdir(parents=True, exist_ok=True)
                    engine_path.write_bytes(b"engine")
                    return smoke.EngineArtifact(engine_path, hashlib.sha256(b"engine").hexdigest(), 6, "8.5.2.2", {}), {"parse_attempted": 1, "parse_completed": 1, "build_attempted": 1, "build_completed": 1}, engine_path.parent / "build_events.jsonl"

                def infer(self, source_bundle, engine, target_dir, timeout_seconds):
                    target_dir.mkdir(parents=True)
                    (target_dir / "00006.bin").write_bytes(source_bundle.onnx_outputs["00006"].path.read_bytes())
                    event_path.parent.mkdir(parents=True, exist_ok=True)
                    event_path.write_text(json.dumps({"event": "enqueue_attempted", "counters": {"engine_load_attempted": 1, "engine_load_completed": 1, "enqueues_attempted": 1, "enqueues_completed": 0}}) + "\n", encoding="utf-8")
                    raise smoke.TargetTimeout("STAGE_TIMEOUT", "event-only timeout", {"termination_confirmed": True, "completion": "unknown"})

            code, failure = smoke.ModelSmokeRunner(bundle, Path(temp) / "out", ProductionFakeTarget([]), execute=True, stage_executor=EventOnlyChild()).run()
            self.assertEqual(code, 2)
            self.assertEqual(failure["attempted_completed"]["enqueues_attempted"], 1)
            self.assertIn("enqueue_completion", failure["attempted_completed"]["unknown_completions"])
            self.assertIn("enqueue_attempted", failure["provenance"]["stage_evidence_snapshot"]["inference"]["events"]["content"])

    def test_child_preflight_failures_write_original_structured_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = root / "result.json"
            events = root / "events.jsonl"
            with patch.object(smoke, "code_provenance", side_effect=smoke.ModelSmokeError("CODE_ARCHIVE_BYTES_MISMATCH", "injected provenance failure")):
                self.assertEqual(smoke._child_stage_main(["--child-stage", "build", "--onnx", str(root / "missing.onnx"), "--engine", str(root / "engine.plan"), "--result", str(result), "--events", str(events)]), 2)
            payload = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(payload["error"]["code"], "CODE_ARCHIVE_BYTES_MISMATCH")
            self.assertEqual(payload["counters"]["build_attempted"], 0)

    def test_manifest_bytes_are_canonical_even_when_onnx_reference_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = make_bundle(temp)
            manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
            manifest["operator_note"] = "edited"
            bundle.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(AdapterError, "SOURCE_MANIFEST_HASH_MISMATCH"):
                smoke.load_source_bundle(bundle.manifest_path, bundle.root)

    def test_target_identity_rejects_before_tensor_rt_import(self):
        calls = []
        identity = smoke._target_identity()
        identity["observed"]["hostname"] = "server-host"
        runtime = smoke.E2TensorRTRuntime(module_loader=lambda name: calls.append(name))
        with patch.object(smoke, "_target_identity", return_value=identity):
            with self.assertRaisesRegex(AdapterError, "TARGET_IDENTITY_MISMATCH"):
                runtime.preflight()
        self.assertEqual(calls, [])

    def test_owned_stage_timeout_is_bounded_and_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            event_path = Path(temp) / "events.jsonl"
            started = time.monotonic()
            with self.assertRaisesRegex(AdapterError, "STAGE_TIMEOUT") as context:
                smoke.run_bounded_process([sys.executable, "-c", "import time; time.sleep(5)"], 0.15, "inference", event_path)
            self.assertLess(time.monotonic() - started, 3.0)
            self.assertEqual(context.exception.details["completion"], "unknown")
            self.assertTrue(context.exception.details["termination_confirmed"])
            events = event_path.read_text(encoding="utf-8")
            self.assertIn('"event": "timeout"', events)

    def test_invalid_timeout_is_rejected_before_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            for value in (0, -1, float("nan"), float("inf")):
                with self.assertRaisesRegex(AdapterError, "TIMEOUT_INVALID"):
                    smoke.run_bounded_process([sys.executable, "-c", "raise SystemExit(99)"], value, "build", Path(temp) / (str(value) + ".jsonl"))

    def test_cleanup_attempts_every_resource_and_reports_all_failures(self):
        calls = []

        class Resource:
            def __init__(self, name, method):
                self.name, self.method = name, method

            def free(self):
                calls.append(self.name)
                raise RuntimeError(self.name)

            def close(self):
                calls.append(self.name)
                raise RuntimeError(self.name)

        execution = smoke._OwnedTargetExecution(None, None, Resource("buffers", "free"), Resource("provider", "close"), Resource("stream", "close"), Resource("cuda", "close"))
        with self.assertRaisesRegex(AdapterError, "TARGET_CLEANUP_FAILED") as context:
            execution.close()
        self.assertEqual(calls, ["buffers", "provider", "stream", "cuda"])
        self.assertEqual([item["resource"] for item in context.exception.details["errors"]], calls)

    def test_allowlist_is_exactly_ten_private_files_and_source_fail_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = make_bundle(temp)
            self.assertEqual(len(bundle.allowlist), 10)
            self.assertTrue(all(item["relative_path"].startswith("private/") for item in bundle.allowlist))
            self.assertEqual(bundle.source_comparisons["00028"]["status"], "fail")

    def test_allowlist_rejects_traversal_and_tampered_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = make_bundle(temp)
            manifest_path = bundle.manifest_path
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["records"]["00006"]["input"]["path"] = str(root / "private" / "inputs" / ".." / "native_reference" / "00006.bin")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(AdapterError, "BUNDLE_PATH_TRAVERSAL"):
                with patch.object(smoke, "CANONICAL_SOURCE_MANIFEST_SHA256", hashlib.sha256(manifest_path.read_bytes()).hexdigest()):
                    with patch.object(smoke, "file_sha256", side_effect=lambda path: smoke.SOURCE_ONNX_SHA256 if path.suffix == ".onnx" else hashlib.sha256(path.read_bytes()).hexdigest()):
                        smoke.load_source_bundle(manifest_path, root)

    def test_allowlist_rejects_bytes_tampered_after_manifest_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = make_bundle(temp)
            input_path = bundle.inputs["00006"].path
            payload = bytearray(input_path.read_bytes())
            payload[0] ^= 1
            input_path.write_bytes(bytes(payload))
            with self.assertRaisesRegex(AdapterError, "BUNDLE_TENSOR_HASH_MISMATCH"):
                with patch.object(smoke, "CANONICAL_SOURCE_MANIFEST_SHA256", bundle.test_manifest_sha256):
                    with patch.object(smoke, "file_sha256", side_effect=lambda path: smoke.SOURCE_ONNX_SHA256 if path.suffix == ".onnx" else hashlib.sha256(path.read_bytes()).hexdigest()):
                        smoke.load_source_bundle(bundle.manifest_path, bundle.root)

    def test_builder_uses_parser_fp16_and_bounded_workspace(self):
        trt = FakeTensorRT()
        with tempfile.TemporaryDirectory() as temp:
            onnx = Path(temp) / "source.onnx"
            engine = Path(temp) / "engine.plan"
            onnx.write_bytes(b"onnx")
            artifact = smoke.TensorRTOnnxBuilder(module_loader=lambda _name: trt).build(onnx, engine)
            self.assertEqual(artifact.runtime_version, "8.5.2.2")
            self.assertEqual(artifact.nbytes, len(b"fake-serialized-engine"))
            self.assertEqual(trt.builder_config.workspace, (0, smoke.WORKSPACE_BYTES))
            self.assertIn(1, trt.builder_config.flags)
            self.assertIn(2, trt.builder_config.cleared)
            self.assertTrue(artifact.build_contract["builder_flags"]["fp16_enabled"])

    def test_builder_parser_error_is_structured_and_no_engine_is_written(self):
        trt = FakeTensorRT(parse_ok=False)
        with tempfile.TemporaryDirectory() as temp:
            onnx = Path(temp) / "source.onnx"
            engine = Path(temp) / "engine.plan"
            onnx.write_bytes(b"onnx")
            with self.assertRaisesRegex(AdapterError, "ONNX_PARSE_FAILED") as context:
                smoke.TensorRTOnnxBuilder(module_loader=lambda _name: trt).build(onnx, engine)
            self.assertEqual(context.exception.details["parser_errors"], ["fake parser error 0"])
            self.assertFalse(engine.exists())

    def test_builder_rejects_wrong_runtime_before_parser_call(self):
        trt = FakeTensorRT()
        trt.__version__ = "8.6.0"
        with tempfile.TemporaryDirectory() as temp:
            onnx = Path(temp) / "source.onnx"
            engine = Path(temp) / "engine.plan"
            onnx.write_bytes(b"onnx")
            with self.assertRaisesRegex(AdapterError, "RUNTIME_VERSION_MISMATCH"):
                smoke.TensorRTOnnxBuilder(module_loader=lambda _name: trt).build(onnx, engine)
            self.assertFalse(engine.exists())

    def test_production_parser_provider_owner_adapter_separates_source_fail_from_target_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = make_bundle(temp)
            outputs = [_pack(([0.1 * (index + 1)] + [0.0] * (smoke.OUTPUT_ELEMENTS - 1))) for index in range(3)]
            target = ProductionFakeTarget(outputs)
            code, manifest = smoke.ModelSmokeRunner(bundle, Path(temp) / "out", target, execute=True).run()
            self.assertEqual(code, 0)
            self.assertEqual(manifest["status"], "execution_complete_source_strict_fail_target_pass")
            self.assertEqual(manifest["source_strict_status"], "fail")
            self.assertEqual(manifest["export_discrepancy"]["status"], "fail")
            self.assertEqual(manifest["tensorrt_discrepancy"], {"target_vs_source_onnx": "pass", "target_vs_source_native": "pass"})
            self.assertEqual(manifest["target_vs_source_onnx_status"], "pass")
            self.assertEqual(manifest["target_vs_source_native_status"], "pass")
            self.assertEqual(manifest["attempted_completed"]["enqueues_completed"], 3)
            self.assertEqual(len(manifest["target_outputs"]), 3)
            self.assertNotEqual(manifest["target_outputs"]["00006"]["sha256"], manifest["target_outputs"]["00009"]["sha256"])
            self.assertIn("neighborhood", manifest["target_outputs"]["00028"])

    def test_second_enqueue_failure_preserves_partial_outputs_and_counters(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = make_bundle(temp)
            outputs = [_pack(([0.1 * (index + 1)] + [0.0] * (smoke.OUTPUT_ELEMENTS - 1))) for index in range(3)]
            target = ProductionFakeTarget(outputs, fail_on_call=2)
            code, failure = smoke.ModelSmokeRunner(bundle, Path(temp) / "out", target, execute=True).run()
            self.assertEqual(code, 2)
            self.assertEqual(failure["execution_state"], "failed")
            self.assertEqual(failure["attempted_completed"]["enqueues_attempted"], 2)
            self.assertEqual(failure["attempted_completed"]["enqueues_completed"], 1)
            self.assertEqual(len(failure["provenance"]["target_outputs"]), 1)

    def test_build_timeout_is_unknown_without_global_cleanup_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = make_bundle(temp)
            target = ProductionFakeTarget([], timeout=True)
            code, failure = smoke.ModelSmokeRunner(bundle, Path(temp) / "out", target, execute=True).run()
            self.assertEqual(code, 2)
            self.assertEqual(failure["execution_state"], "unknown")
            self.assertEqual(failure["error"]["code"], "STAGE_TIMEOUT")
            self.assertEqual(failure["attempted_completed"]["build_attempted"], 1)
            self.assertEqual(failure["attempted_completed"]["build_completed"], 0)

    def test_default_disabled_and_output_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = make_bundle(temp)
            target = ProductionFakeTarget([])
            out = Path(temp) / "dry"
            code, manifest = smoke.ModelSmokeRunner(bundle, out, target).run()
            self.assertEqual(code, 3)
            self.assertEqual(manifest["status"], "proposed_not_executed")
            self.assertEqual(target.calls, [])
            with self.assertRaisesRegex(ValueError, "refusing existing model-smoke output root"):
                smoke.ModelSmokeRunner(bundle, out, target).run()

    def test_python38_parse_contract_for_workflow_and_imported_boundaries(self):
        for relative in ("scripts/edge_readiness/e2_model_smoke.py", "scripts/edge_readiness/e2_output_compare.py", "scripts/edge_readiness/jetson_adapter.py", "scripts/edge_readiness/jetson_runtime_provider.py", "scripts/edge_readiness/e2_source_fixture.py"):
            ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative, feature_version=(3, 8))


if __name__ == "__main__":
    unittest.main()
