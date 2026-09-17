import hashlib
import ctypes
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from edge_readiness.e2_source_fixture import (  # noqa: E402
    FixtureError,
    preprocess_decoded_bgr,
    verify_fixture,
)
from edge_readiness.e2_output_compare import (  # noqa: E402
    OUTPUT_ELEMENTS,
    ComparisonPolicy,
    compare_output0,
)
from edge_readiness.jetson_adapter import (  # noqa: E402
    AdapterConfig,
    AdapterError,
    EngineBinding,
    EngineDescriptor,
    HostTensor,
    JetsonRuntimeAdapter,
    MockBuffers,
    MockStream,
    expected_nbytes,
    make_host_tensor,
)
from edge_readiness.jetson_runtime_provider import (  # noqa: E402
    OwnedBuffers,
    TensorRTProvider,
)
from edge_readiness.cuda_runtime_owner import (  # noqa: E402
    CudaRuntime,
    CudaRuntimeMemoryOwner,
)
from edge_readiness.e2_cuda_allocation_smoke import (  # noqa: E402
    build_plan,
    run_roundtrip,
)


class FakeContext:
    def __init__(self):
        self.calls = []

    def execute_async_v2(self, bindings, stream_handle):
        self.calls.append((tuple(bindings), stream_handle))
        return True


class FakeEngine:
    num_bindings = 2

    def __init__(self):
        self.context = FakeContext()

    def get_binding_name(self, index):
        return ("images", "output0")[index]

    def get_binding_shape(self, index):
        return ((1, 3, 640, 640), (1, 7, 8400))[index]

    def get_binding_dtype(self, index):
        return "float32"

    def binding_is_input(self, index):
        return index == 0

    def get_location(self, _index):
        return "TensorLocation.DEVICE"

    def create_execution_context(self):
        return self.context


class FakeTensorRT:
    __version__ = "8.5.2.2"

    class Logger:
        ERROR = 0

        def __init__(self, _level):
            pass

    class Runtime:
        engine = FakeEngine()

        def __init__(self, _logger):
            pass

        def deserialize_cuda_engine(self, _bytes):
            return self.engine


class FakeMemory:
    def __init__(self, fail_after=None):
        self.next_pointer = 0x700000
        self.fail_after = fail_after
        self.allocations = []
        self.freed = []
        self.copies = []
        self.pending_d2h = []
        self.completion_value = 42
        self.fail_h2d = False
        self.fail_sync = False
        self.fail_free = False
        self.sync_calls = []

    def allocate_device(self, nbytes, name):
        if self.fail_after is not None and len(self.allocations) >= self.fail_after:
            raise RuntimeError("injected allocation failure")
        pointer = self.next_pointer
        self.next_pointer += 0x1000
        self.allocations.append((name, nbytes, pointer))
        return pointer

    def copy_host_to_device(self, payload, device_pointer, stream_handle):
        if self.fail_h2d:
            raise RuntimeError("injected H2D failure")
        self.copies.append(("h2d", len(payload), device_pointer, stream_handle))

    def copy_device_to_host(self, device_pointer, destination, stream_handle):
        self.copies.append(("d2h", len(destination), device_pointer, stream_handle))
        self.pending_d2h.append(destination)

    def complete_pending(self):
        if not self.pending_d2h:
            return
        for destination in self.pending_d2h:
            destination[:] = bytes([self.completion_value]) * len(destination)
        self.pending_d2h.clear()
        self.completion_value += 1

    def synchronize(self, stream_handle):
        self.sync_calls.append(stream_handle)
        if self.fail_sync:
            raise RuntimeError("injected synchronize failure")

    def free_device(self, device_pointer):
        if self.fail_free:
            raise RuntimeError("injected free failure")
        self.freed.append(device_pointer)


class FakeStream:
    handle = 13

    def __init__(self, on_synchronize=None):
        self.stages = []
        self.on_synchronize = on_synchronize

    def synchronize(self, stage):
        if self.on_synchronize is not None:
            self.on_synchronize()
        self.stages.append(stage)


class FakeCFunction:
    def __init__(self, function):
        self.function = function
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.function(*args)


class FakeCudaLibrary:
    def __init__(self, fail_operation=None):
        self.device_memory = {}
        self.next_pointer = 0x5000
        self.freed = []
        self.destroyed_streams = []
        self.fail_operation = fail_operation
        self.cudaMalloc = FakeCFunction(self._cuda_malloc)
        self.cudaFree = FakeCFunction(self._cuda_free)
        self.cudaMemcpy = FakeCFunction(self._cuda_memcpy)
        self.cudaStreamCreate = FakeCFunction(self._cuda_stream_create)
        self.cudaStreamDestroy = FakeCFunction(self._cuda_stream_destroy)
        self.cudaStreamSynchronize = FakeCFunction(self._cuda_stream_synchronize)
        self.cudaGetErrorString = FakeCFunction(lambda code: f"fake-{code}".encode())

    def _maybe_fail(self, operation):
        return 7 if self.fail_operation == operation else 0

    @staticmethod
    def _value(value):
        return value.value if hasattr(value, "value") else int(value)

    def _cuda_malloc(self, output, size):
        failure = self._maybe_fail("cudaMalloc")
        if failure:
            return failure
        pointer = self.next_pointer
        self.next_pointer += 0x1000
        output._obj.value = pointer
        self.device_memory[pointer] = bytearray(self._value(size))
        return 0

    def _cuda_free(self, pointer):
        failure = self._maybe_fail("cudaFree")
        if failure:
            return failure
        self.freed.append(int(pointer.value))
        self.device_memory.pop(int(pointer.value), None)
        return 0

    def _cuda_memcpy(self, destination, source, size, kind):
        failure = self._maybe_fail("cudaMemcpy")
        if failure:
            return failure
        count = self._value(size)
        if self._value(kind) == 1:
            self.device_memory[int(destination.value)][:count] = ctypes.string_at(source, count)
        elif int(kind) == 2:
            ctypes.memmove(destination, bytes(self.device_memory[int(source.value)][:count]), count)
        else:
            return 9
        return 0

    def _cuda_stream_create(self, output):
        output._obj.value = 0x7000
        return 0

    def _cuda_stream_destroy(self, stream):
        self.destroyed_streams.append(int(stream.value))
        return 0

    def _cuda_stream_synchronize(self, _stream):
        return self._maybe_fail("cudaStreamSynchronize")


class E2RuntimeProviderTests(unittest.TestCase):
    def test_provider_is_lazy_and_loads_only_on_explicit_engine_call(self):
        loaded = []

        def loader(name):
            loaded.append(name)
            return FakeTensorRT

        provider = TensorRTProvider(AdapterConfig("E2", "8.5.2.2"), module_loader=loader)
        self.assertEqual(loaded, [])
        engine_bytes = b"fake-engine"
        descriptor = provider.load_engine(engine_bytes, hashlib.sha256(engine_bytes).hexdigest())
        self.assertEqual(loaded, ["tensorrt"])
        self.assertEqual([binding.name for binding in descriptor.bindings], ["images", "output0"])
        self.assertEqual(descriptor.runtime_version, "8.5.2.2")
        self.assertTrue(all(binding.location == "device" for binding in descriptor.bindings))
        provider.close()
        self.assertFalse(provider.loaded)

    def test_provider_rejects_missing_observed_runtime_version_before_deserialize(self):
        class NoVersionTensorRT:
            class Logger:
                ERROR = 0

                def __init__(self, _level):
                    pass

            class Runtime:
                def __init__(self, _logger):
                    raise AssertionError("deserialization must not start")

        provider = TensorRTProvider(AdapterConfig("E2", "8.5.2.2"), module_loader=lambda _name: NoVersionTensorRT)
        with self.assertRaisesRegex(AdapterError, "RUNTIME_VERSION_UNAVAILABLE"):
            provider.load_engine(b"fake-engine", hashlib.sha256(b"fake-engine").hexdigest())

    def test_provider_rejects_missing_binding_location(self):
        class NoLocationEngine(FakeEngine):
            get_location = None

        class NoLocationTensorRT(FakeTensorRT):
            __version__ = "8.5.2.2"

            class Runtime:
                def __init__(self, _logger):
                    self.engine = NoLocationEngine()

                def deserialize_cuda_engine(self, _bytes):
                    return self.engine

        provider = TensorRTProvider(AdapterConfig("E2", "8.5.2.2"), module_loader=lambda _name: NoLocationTensorRT)
        with self.assertRaisesRegex(AdapterError, "BINDING_LOCATION_UNKNOWN|BINDING_LOCATION_UNAVAILABLE"):
            provider.load_engine(b"fake-engine", hashlib.sha256(b"fake-engine").hexdigest())

    def test_provider_rejects_null_execution_context(self):
        class NullContextEngine(FakeEngine):
            def create_execution_context(self):
                return None

        class NullContextTensorRT(FakeTensorRT):
            class Runtime:
                def __init__(self, _logger):
                    self.engine = NullContextEngine()

                def deserialize_cuda_engine(self, _bytes):
                    return self.engine

        provider = TensorRTProvider(AdapterConfig("E2", "8.5.2.2"), module_loader=lambda _name: NullContextTensorRT)
        descriptor = provider.load_engine(b"fake-engine", hashlib.sha256(b"fake-engine").hexdigest())
        buffers = OwnedBuffers(descriptor, FakeMemory(), stream_handle=13)
        with self.assertRaisesRegex(AdapterError, "EXECUTION_CONTEXT_UNAVAILABLE"):
            provider.create_adapter(FakeStream(), buffers)

    def test_real_bridge_uses_owned_allocations_and_frees_cleanly(self):
        engine_bytes = b"fake-engine"
        provider = TensorRTProvider(AdapterConfig("E2", "8.5.2.2"), module_loader=lambda _name: FakeTensorRT)
        descriptor = provider.load_engine(engine_bytes, hashlib.sha256(engine_bytes).hexdigest())
        memory = FakeMemory()
        buffers = OwnedBuffers(descriptor, memory, stream_handle=13)
        stream = FakeStream(on_synchronize=memory.complete_pending)
        adapter = provider.create_adapter(stream, buffers)
        host = make_host_tensor(descriptor.input_binding(), b"\0" * expected_nbytes((1, 3, 640, 640), "float32"))
        first = adapter.infer(host)
        second = adapter.infer(host)
        self.assertIsNot(first["output0"], second["output0"])
        self.assertEqual(set(first["output0"].payload), {42})
        self.assertEqual(set(second["output0"].payload), {43})
        self.assertNotEqual(set(first["output0"].payload), {0})
        self.assertEqual(memory.copies[0][2], memory.allocations[0][2])
        self.assertEqual(memory.copies[-1][2], memory.allocations[1][2])
        self.assertEqual(len(first["output0"].payload), expected_nbytes((1, 7, 8400), "float32"))
        buffers.free()
        self.assertEqual(set(memory.freed), {allocation[2] for allocation in memory.allocations})
        self.assertIn(13, memory.sync_calls)
        provider.close()

    def test_failed_second_call_cannot_expose_previous_output_and_free_rejects_reuse(self):
        engine_bytes = b"fake-engine"
        provider = TensorRTProvider(AdapterConfig("E2", "8.5.2.2"), module_loader=lambda _name: FakeTensorRT)
        descriptor = provider.load_engine(engine_bytes, hashlib.sha256(engine_bytes).hexdigest())
        memory = FakeMemory()
        buffers = OwnedBuffers(descriptor, memory, stream_handle=13)
        stream = FakeStream(on_synchronize=memory.complete_pending)
        adapter = provider.create_adapter(stream, buffers)
        host = make_host_tensor(descriptor.input_binding(), b"\0" * expected_nbytes((1, 3, 640, 640), "float32"))
        adapter.infer(host)
        memory.fail_h2d = True
        with self.assertRaisesRegex(AdapterError, "COPY_H2D_FAILED"):
            adapter.infer(host)
        with self.assertRaisesRegex(AdapterError, "OUTPUT_NOT_READY"):
            buffers.output_tensors()
        buffers.free()
        with self.assertRaisesRegex(AdapterError, "BUFFER_OWNER_FREED"):
            adapter.infer(host)
        provider.close()

    def test_output_is_not_available_before_readiness_or_after_free(self):
        descriptor = FakeEngineDescriptor()
        buffers = OwnedBuffers(descriptor, FakeMemory(), stream_handle=13)
        output = descriptor.output_bindings()[0]
        device = buffers.binding_buffers()[output.name]
        buffers.begin_inference()
        buffers.copy_device_to_host(device, 13)
        with self.assertRaisesRegex(AdapterError, "OUTPUT_NOT_READY"):
            buffers.output_tensors()
        buffers.mark_outputs_ready(13)
        self.assertIn(output.name, buffers.output_tensors())
        buffers.free()
        with self.assertRaisesRegex(AdapterError, "OUTPUT_NOT_READY"):
            buffers.output_tensors()

    def test_partial_allocation_frees_prior_buffers(self):
        descriptor = FakeEngineDescriptor()
        memory = FakeMemory(fail_after=1)
        with self.assertRaisesRegex(AdapterError, "ALLOCATION_FAILED"):
            OwnedBuffers(descriptor, memory, stream_handle=13)
        self.assertEqual(len(memory.freed), 1)

    def test_output_descriptor_mismatch_fails_before_compare(self):
        class WrongDescriptorBuffers(MockBuffers):
            def output_tensors(self):
                outputs = super().output_tensors()
                outputs["output0"] = HostTensor("output0", (1, 7, 1), "float32", 28, b"\0" * 28)
                return outputs

        fake_descriptor = FakeEngineDescriptor()
        descriptor = EngineDescriptor(fake_descriptor.runtime_version, fake_descriptor.engine_sha256, fake_descriptor.bindings)
        buffers = WrongDescriptorBuffers(descriptor, [], stream_handle=13)
        adapter = JetsonRuntimeAdapter(AdapterConfig("E2", "8.5.2.2"), descriptor, FakeContext(), FakeStream(), buffers)
        host = make_host_tensor(descriptor.input_binding(), b"\0" * expected_nbytes((1, 3, 640, 640), "float32"))
        with self.assertRaisesRegex(AdapterError, "OUTPUT_CONTRACT_MISMATCH"):
            adapter.infer(host)

    def test_engine_hash_mismatch_does_not_load_runtime(self):
        loaded = []
        provider = TensorRTProvider(AdapterConfig("E2", "8.5.2.2"), module_loader=lambda name: loaded.append(name))
        with self.assertRaisesRegex(AdapterError, "ENGINE_HASH_MISMATCH"):
            provider.load_engine(b"fake-engine", "0" * 64)
        self.assertEqual(loaded, [])


class ConcreteCudaOwnerTests(unittest.TestCase):
    def test_ctypes_owner_is_lazy_and_roundtrip_is_exact_with_fake_functions(self):
        library = FakeCudaLibrary()
        runtime = CudaRuntime(library_loader=lambda _candidate: library, library_candidates=("fake-libcudart",))
        self.assertFalse(runtime.loaded)
        runtime.open()
        stream = runtime.create_stream()
        owner = CudaRuntimeMemoryOwner(runtime, stream)
        result = run_roundtrip(owner, stream)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["observed_sha256"], result["payload_sha256"])
        self.assertEqual(library.device_memory, {})
        stream.close()
        runtime.close()
        self.assertEqual(library.destroyed_streams, [0x7000])

    def test_ctypes_owner_checks_return_codes_and_does_not_claim_load_success(self):
        library = FakeCudaLibrary(fail_operation="cudaMalloc")
        runtime = CudaRuntime(library_loader=lambda _candidate: library, library_candidates=("fake-libcudart",))
        runtime.open()
        with self.assertRaisesRegex(AdapterError, "CUDA_CALL_FAILED"):
            runtime.allocate_device(64)
        runtime.close()

    def test_ctypes_owner_copy_sync_and_free_failures_are_structured(self):
        for operation, expected in (("cudaMemcpy", "CUDA_CALL_FAILED"), ("cudaStreamSynchronize", "CUDA_CALL_FAILED"), ("cudaFree", "CUDA_CALL_FAILED")):
            library = FakeCudaLibrary(fail_operation=operation)
            runtime = CudaRuntime(library_loader=lambda _candidate, lib=library: lib, library_candidates=("fake-libcudart",))
            runtime.open()
            stream = runtime.create_stream()
            owner = CudaRuntimeMemoryOwner(runtime, stream)
            if operation == "cudaFree":
                pointer = owner.allocate_device(64, "failure-test")
                with self.assertRaisesRegex(AdapterError, expected):
                    owner.free_device(pointer)
            else:
                with self.assertRaisesRegex(AdapterError, expected):
                    run_roundtrip(owner, stream)
            stream.close()
            runtime.close()

    def test_owned_buffers_attempts_sync_and_every_free_even_when_cleanup_fails(self):
        descriptor = FakeEngineDescriptor()
        memory = FakeMemory()
        buffers = OwnedBuffers(descriptor, memory, stream_handle=13)
        memory.fail_sync = True
        memory.fail_free = True
        with self.assertRaisesRegex(AdapterError, "FREE_FAILED"):
            buffers.free()
        self.assertEqual(len(memory.sync_calls), 1)
        self.assertEqual(memory.freed, [])
        buffers.free()

    def test_allocation_smoke_is_default_disabled(self):
        plan = build_plan()
        self.assertEqual(plan["status"], "proposed_default_disabled")
        self.assertFalse(plan["real_device_execution"])
        self.assertLessEqual(plan["payload_nbytes"], 1024 * 1024)


class FakeEngineDescriptor:
    runtime_version = "8.5.2.2"
    engine_sha256 = "b" * 64
    bindings = (
        EngineBinding("images", "input", (1, 3, 640, 640), "float32"),
        EngineBinding("output0", "output", (1, 7, 8400), "float32"),
    )

    def input_binding(self):
        return self.bindings[0]

    def output_bindings(self):
        return (self.bindings[1],)


class FixturePreparationTests(unittest.TestCase):
    def test_preprocess_injected_resize_has_explicit_contract(self):
        def identity_resize(rgb, width, height, new_width, new_height):
            self.assertEqual((width, height, new_width, new_height), (2, 1, 640, 320))
            return bytes(pixel for row in range(new_height) for pixel in rgb[(row % height) * width * 3:((row % height) + 1) * width * 3] * (new_width // width))

        result = preprocess_decoded_bgr(bytes([10, 20, 30, 40, 50, 60]), 2, 1, identity_resize)
        self.assertEqual(result.shape, (1, 3, 640, 640))
        self.assertEqual(result.dtype, "float32")
        self.assertEqual(len(result.payload), expected_nbytes(result.shape, result.dtype))
        first_resized_pixel = (160 * 640) * 4
        self.assertAlmostEqual(struct.unpack("<f", result.payload[first_resized_pixel:first_resized_pixel + 4])[0], 30 / 255.0, places=7)
        self.assertEqual(result.metadata["layout"], "NCHW")
        self.assertEqual(result.metadata["interpolation"], "injected_source_resize")

    def test_preprocess_rejects_wrong_decoded_bytes(self):
        with self.assertRaisesRegex(FixtureError, "DECODED_IMAGE_INVALID"):
            preprocess_decoded_bgr(b"short", 2, 1, lambda *_args: b"")

    def test_actual_fixture_manifest_is_verified_without_copying_images(self):
        source_root = Path(r"D:\Research\paper")
        if not (source_root / "data/processed/cctsdb2021_clean/train/images/00006.jpg").exists():
            self.skipTest("canonical local fixture source is not mounted")
        manifest = verify_fixture(source_root)
        self.assertEqual(manifest["status"], "verified_read_only")
        self.assertEqual(manifest["canonical_order"], ["00006", "00009", "00028"])
        self.assertFalse(manifest["official_test_used"])
        self.assertEqual(manifest["derived_input"]["status"], "not_materialized_by_manifest")


class E2OutputComparatorTests(unittest.TestCase):
    def test_compute_mode_is_separate_from_io_dtype_and_domains_are_separate(self):
        reference = [0.0] * OUTPUT_ELEMENTS
        target = [0.0] * OUTPUT_ELEMENTS
        target[0] = 0.004  # within box absolute tolerance
        target[4 * 8400] = 0.003  # outside score absolute tolerance
        policy = ComparisonPolicy(target_compute_mode="fp16", target_binding_dtype="float32")
        result = compare_output0(reference, target, policy)
        self.assertFalse(result.passed)
        self.assertEqual(result.domain_summary["boxes"]["mismatches"], 0)
        self.assertEqual(result.domain_summary["scores"]["mismatches"], 1)
        self.assertEqual(result.policy["target_compute_mode"], "fp16")
        self.assertEqual(result.policy["target_binding_dtype"], "float32")
        self.assertEqual(result.policy["comparison_stage"], "native output0 before decode/NMS")

    def test_nonfinite_values_fail_closed_and_mismatch_summary_is_bounded(self):
        reference = [0.0] * OUTPUT_ELEMENTS
        target = [0.0] * OUTPUT_ELEMENTS
        reference[2] = float("nan")
        target[3] = float("inf")
        for index in range(4, 30):
            target[index] = 1.0
        result = compare_output0(reference, target)
        self.assertFalse(result.passed)
        self.assertEqual(result.nonfinite_reference, 1)
        self.assertEqual(result.nonfinite_target, 1)
        self.assertEqual(len(result.mismatches), 20)
        self.assertGreaterEqual(result.mismatch_count, 2)
        json.dumps(result.as_dict(), allow_nan=False)


class E2WorkflowTests(unittest.TestCase):
    def test_local_preflight_and_source_stages_write_scoped_manifests(self):
        from edge_readiness.e2_correctness_workflow import run_stage

        source_root = Path(r"D:\Research\paper")
        if not (source_root / "data/processed/cctsdb2021_clean/train/images/00006.jpg").exists():
            self.skipTest("canonical local fixture source is not mounted")
        with tempfile.TemporaryDirectory(prefix="e2l1-008-workflow-") as temp_dir:
            temp_root = Path(temp_dir)
            preflight = run_stage("preflight", ROOT, source_root, temp_root / "preflight")
            source = run_stage("source-artifacts", ROOT, source_root, temp_root / "source-artifacts")
            self.assertEqual(preflight["status"], "preflight_verified_read_only")
            self.assertEqual(source["status"], "source_artifacts_verified_read_only")
            self.assertTrue((temp_root / "source-artifacts" / "fixture_manifest.json").is_file())

    def test_device_stages_are_structurally_blocked_without_side_effects(self):
        from edge_readiness.e2_correctness_workflow import run_stage

        with tempfile.TemporaryDirectory(prefix="e2l1-008-workflow-") as temp_dir:
            result = run_stage("target-build", ROOT, ROOT, Path(temp_dir) / "target-build")
            self.assertEqual(result["status"], "blocked_not_authorized")
            self.assertFalse(result["real_device_execution"])
            self.assertIn("TensorRT export/build", result["disabled_side_effects"])

    def test_cli_records_fixture_failure_inside_new_root_and_does_not_mask_it(self):
        from edge_readiness.e2_correctness_workflow import main

        with tempfile.TemporaryDirectory(prefix="e2l1-009-workflow-") as temp_dir:
            temp_root = Path(temp_dir)
            output = temp_root / "failed-source"
            source_root = temp_root / "missing-source"
            self.assertEqual(main(["--stage", "source-artifacts", "--repo-root", str(ROOT), "--source-root", str(source_root), "--out-dir", str(output)]), 2)
            failure = output / "failure.json"
            self.assertTrue(failure.is_file())
            self.assertIn("FIXTURE_MISSING", failure.read_text(encoding="utf-8"))
            before = failure.read_bytes()
            self.assertEqual(main(["--stage", "source-artifacts", "--repo-root", str(ROOT), "--source-root", str(source_root), "--out-dir", str(output)]), 2)
            self.assertEqual(failure.read_bytes(), before)

    def test_cli_rejects_second_invocation_without_mutating_success_root(self):
        from edge_readiness.e2_correctness_workflow import main

        source_root = Path(r"D:\Research\paper")
        if not (source_root / "data/processed/cctsdb2021_clean/train/images/00006.jpg").exists():
            self.skipTest("canonical local fixture source is not mounted")
        with tempfile.TemporaryDirectory(prefix="e2l1-009-workflow-") as temp_dir:
            output = Path(temp_dir) / "preflight"
            args = ["--stage", "preflight", "--repo-root", str(ROOT), "--source-root", str(source_root), "--out-dir", str(output)]
            self.assertEqual(main(args), 0)
            before = sorted((path.relative_to(output).as_posix(), path.read_bytes()) for path in output.rglob("*" ) if path.is_file())
            self.assertEqual(main(args), 2)
            after = sorted((path.relative_to(output).as_posix(), path.read_bytes()) for path in output.rglob("*") if path.is_file())
            self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
