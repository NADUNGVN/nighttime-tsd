import hashlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from edge_readiness.e5_tensorrt_compat import (  # noqa: E402
    CUDA12_LIBRARY_CANDIDATES,
    CUDA_REQUIRED_SYMBOLS,
    E5CompatibilityError,
    E5_INPUT_SHAPE,
    E5_OUTPUT_SHAPE,
    Cuda12LibraryResolver,
    execute_e5_v3_once,
    inspect_e5_named_io,
    require_opencv,
    validate_hash_bindings,
)


class MockEngine:
    num_io_tensors = 2

    def __init__(self, overrides=None):
        self.rows = {
            "images": {"mode": "TensorIOMode.INPUT", "shape": E5_INPUT_SHAPE, "dtype": "DataType.FLOAT", "location": "TensorLocation.DEVICE"},
            "output0": {"mode": "TensorIOMode.OUTPUT", "shape": E5_OUTPUT_SHAPE, "dtype": "DataType.FLOAT", "location": "TensorLocation.DEVICE"},
        }
        for name, values in (overrides or {}).items():
            self.rows[name].update(values)

    def get_tensor_name(self, index):
        return ("images", "output0")[index]

    def get_tensor_mode(self, name):
        return self.rows[name]["mode"]

    def get_tensor_shape(self, name):
        return self.rows[name]["shape"]

    def get_tensor_dtype(self, name):
        return self.rows[name]["dtype"]

    def get_tensor_location(self, name):
        return self.rows[name]["location"]


class MockMemory:
    def __init__(self, fail_sync_calls=(), fail_allocate_at=None, fail_free_at=()):
        self.events = []
        self.next_pointer = 100
        self.pending_copies = []
        self.output_generation = 0
        self.sync_calls = 0
        self.fail_sync_calls = set(fail_sync_calls)
        self.fail_allocate_at = fail_allocate_at
        self.allocate_calls = 0
        self.fail_free_at = set(fail_free_at)
        self.free_calls = 0

    def allocate_device(self, nbytes, name):
        self.allocate_calls += 1
        self.events.append(("allocate", name, nbytes))
        if self.fail_allocate_at == self.allocate_calls:
            raise RuntimeError("mock allocation failure")
        self.next_pointer += 1
        return self.next_pointer

    def copy_host_to_device(self, payload, device_pointer, stream_handle):
        self.events.append(("h2d", device_pointer, len(payload), stream_handle))

    def copy_device_to_host(self, device_pointer, destination, stream_handle):
        self.output_generation += 1
        self.events.append(("d2h_enqueued", device_pointer, len(destination), stream_handle))
        marker = self.output_generation % 251 + 1
        self.pending_copies.append((destination, bytes([marker]) * len(destination)))

    def synchronize(self, stream_handle):
        self.sync_calls += 1
        self.events.append(("synchronize", stream_handle))
        if self.sync_calls in self.fail_sync_calls:
            raise RuntimeError("mock stream synchronization failure")
        pending, self.pending_copies = self.pending_copies, []
        for destination, payload in pending:
            destination[:] = payload

    def free_device(self, device_pointer):
        self.free_calls += 1
        self.events.append(("free", device_pointer))
        if self.free_calls in self.fail_free_at:
            raise RuntimeError("mock device free failure")


class MockContext:
    def __init__(self, events, enqueue_result=True, reject_address_name=None):
        self.events = events
        self.enqueue_result = enqueue_result
        self.reject_address_name = reject_address_name

    def set_tensor_address(self, name, pointer):
        self.events.append(("address", name, pointer))
        return name != self.reject_address_name

    def execute_async_v3(self, stream_handle):
        self.events.append(("execute_v3", stream_handle))
        return self.enqueue_result


def make_hash_manifest(source, input_payload, engine):
    source_hash = hashlib.sha256(source).hexdigest()
    return {
        "source_onnx_sha256": source_hash,
        "engine_source_onnx_sha256": source_hash,
        "input_sha256": hashlib.sha256(input_payload).hexdigest(),
        "engine_sha256": hashlib.sha256(engine).hexdigest(),
    }


def approved_payloads():
    source = b"mock-source-onnx"
    input_payload = b"\x11" * (1 * 3 * 640 * 640 * 4)
    engine = b"mock-e5-engine"
    return source, input_payload, engine, make_hash_manifest(source, input_payload, engine)


class E5TensorRTCompatibilityTests(unittest.TestCase):
    def test_named_io_mode_shape_dtype_location_and_runtime_profile(self):
        actual = inspect_e5_named_io(MockEngine(), "10.3.0.30")
        self.assertEqual(actual["images"].mode, "INPUT")
        self.assertEqual(actual["images"].shape, E5_INPUT_SHAPE)
        self.assertEqual(actual["images"].dtype, "float32")
        self.assertEqual(actual["images"].location, "DEVICE")
        self.assertEqual(actual["output0"].shape, E5_OUTPUT_SHAPE)
        with self.assertRaisesRegex(E5CompatibilityError, "TensorRT 10.3"):
            inspect_e5_named_io(MockEngine(), "8.5.2.2")

    def test_named_io_fails_closed_on_dtype_location_shape_or_mode_drift(self):
        cases = (
            ({"images": {"dtype": "DataType.HALF"}}, "NAMED_IO_CONTRACT_MISMATCH"),
            ({"images": {"location": "TensorLocation.HOST"}}, "NAMED_IO_CONTRACT_MISMATCH"),
            ({"output0": {"shape": (1, 7, -1)}}, "NAMED_IO_DYNAMIC_SHAPE"),
            ({"output0": {"mode": "TensorIOMode.INPUT"}}, "NAMED_IO_CONTRACT_MISMATCH"),
        )
        for overrides, expected_code in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(E5CompatibilityError) as caught:
                    inspect_e5_named_io(MockEngine(overrides), "10.3.0.30")
                self.assertEqual(caught.exception.code, expected_code)

    def test_hash_bindings_cover_source_input_engine_and_engine_source(self):
        source, input_payload, engine, manifest = approved_payloads()
        result = validate_hash_bindings(manifest, source, input_payload, engine)
        self.assertEqual(result["source_onnx_sha256"], manifest["source_onnx_sha256"])
        self.assertEqual(result["input_sha256"], manifest["input_sha256"])
        self.assertEqual(result["engine_sha256"], manifest["engine_sha256"])
        wrong_source_binding = dict(manifest, engine_source_onnx_sha256="0" * 64)
        with self.assertRaisesRegex(E5CompatibilityError, "manifest"):
            validate_hash_bindings(wrong_source_binding, source, input_payload, engine)
        with self.assertRaisesRegex(E5CompatibilityError, "manifest"):
            validate_hash_bindings(manifest, source, input_payload + b"x", engine)

    def test_v3_order_sync_and_fresh_host_output_per_invocation(self):
        source, input_payload, engine_bytes, manifest = approved_payloads()
        memory = MockMemory()
        context = MockContext(memory.events)
        outputs = []
        for _index in range(2):
            result = execute_e5_v3_once(
                MockEngine(), context, memory, 77, input_payload, source, engine_bytes, manifest, "10.3.0.30"
            )
            self.assertTrue(result["output_copy_synchronized"])
            self.assertTrue(result["fresh_host_output_buffer"])
            self.assertEqual(len(result["output_bytes"]), 1 * 7 * 8400 * 4)
            outputs.append(result["output_bytes"])
        self.assertNotEqual(outputs[0], outputs[1])
        kinds = [event[0] for event in memory.events]
        address_indices = [index for index, kind in enumerate(kinds) if kind == "address"]
        h2d_indices = [index for index, kind in enumerate(kinds) if kind == "h2d"]
        enqueue_indices = [index for index, kind in enumerate(kinds) if kind == "execute_v3"]
        d2h_indices = [index for index, kind in enumerate(kinds) if kind == "d2h_enqueued"]
        sync_indices = [index for index, kind in enumerate(kinds) if kind == "synchronize"]
        free_indices = [index for index, kind in enumerate(kinds) if kind == "free"]
        self.assertEqual(len(address_indices), 4)
        self.assertTrue(max(address_indices[:2]) < h2d_indices[0] < enqueue_indices[0] < d2h_indices[0] < sync_indices[0] < free_indices[0])
        self.assertTrue(max(address_indices[2:]) < h2d_indices[1] < enqueue_indices[1] < d2h_indices[1] < sync_indices[2] < free_indices[2])

    def test_bad_hash_is_rejected_before_any_device_allocation(self):
        source, input_payload, engine_bytes, manifest = approved_payloads()
        memory = MockMemory()
        with self.assertRaises(E5CompatibilityError) as caught:
            execute_e5_v3_once(
                MockEngine(), MockContext(memory.events), memory, 0, input_payload + b"x", source, engine_bytes, manifest, "10.3.0.30"
            )
        self.assertEqual(caught.exception.code, "HASH_BINDING_MISMATCH")
        self.assertEqual(memory.allocate_calls, 0)

    def test_enqueue_failure_withholds_output_and_releases_after_stream_drain(self):
        source, input_payload, engine_bytes, manifest = approved_payloads()
        memory = MockMemory()
        context = MockContext(memory.events, enqueue_result=False)
        with self.assertRaises(E5CompatibilityError) as caught:
            execute_e5_v3_once(MockEngine(), context, memory, 8, input_payload, source, engine_bytes, manifest, "10.3.0.30")
        self.assertEqual(caught.exception.code, "EXECUTE_ASYNC_V3_FAILED")
        self.assertNotIn("d2h_enqueued", [event[0] for event in memory.events])
        self.assertEqual(len([event for event in memory.events if event[0] == "free"]), 2)

    def test_named_address_rejection_releases_allocations_without_dispatch(self):
        source, input_payload, engine_bytes, manifest = approved_payloads()
        memory = MockMemory()
        context = MockContext(memory.events, reject_address_name="output0")
        with self.assertRaises(E5CompatibilityError) as caught:
            execute_e5_v3_once(MockEngine(), context, memory, 8, input_payload, source, engine_bytes, manifest, "10.3.0.30")
        self.assertEqual(caught.exception.code, "TENSOR_ADDRESS_FAILED")
        self.assertNotIn("execute_v3", [event[0] for event in memory.events])
        self.assertEqual(len([event for event in memory.events if event[0] == "free"]), 2)

    def test_sync_failure_withholds_output_even_if_cleanup_later_succeeds(self):
        source, input_payload, engine_bytes, manifest = approved_payloads()
        memory = MockMemory(fail_sync_calls=(1,))
        with self.assertRaises(E5CompatibilityError) as caught:
            execute_e5_v3_once(MockEngine(), MockContext(memory.events), memory, 8, input_payload, source, engine_bytes, manifest, "10.3.0.30")
        self.assertEqual(caught.exception.code, "OUTPUT_SYNC_FAILED")
        self.assertEqual(len([event for event in memory.events if event[0] == "free"]), 2)

    def test_unconfirmed_stream_drain_retains_live_allocations(self):
        source, input_payload, engine_bytes, manifest = approved_payloads()
        memory = MockMemory(fail_sync_calls=(1, 2))
        with self.assertRaises(E5CompatibilityError) as caught:
            execute_e5_v3_once(MockEngine(), MockContext(memory.events), memory, 8, input_payload, source, engine_bytes, manifest, "10.3.0.30")
        self.assertEqual(caught.exception.code, "OPERATION_AND_CLEANUP_FAILED")
        self.assertTrue(caught.exception.details["live_allocations_retained"])
        self.assertFalse(any(event[0] == "free" for event in memory.events))

    def test_free_failure_withholds_output_and_reports_incomplete_lifetime(self):
        source, input_payload, engine_bytes, manifest = approved_payloads()
        memory = MockMemory(fail_free_at=(1,))
        with self.assertRaises(E5CompatibilityError) as caught:
            execute_e5_v3_once(MockEngine(), MockContext(memory.events), memory, 8, input_payload, source, engine_bytes, manifest, "10.3.0.30")
        self.assertEqual(caught.exception.code, "CLEANUP_FAILED")
        self.assertTrue(caught.exception.details["live_allocations_retained"])
        self.assertEqual(len([event for event in memory.events if event[0] == "free"]), 2)

    def test_partial_allocation_failure_drains_and_releases_prior_allocations(self):
        source, input_payload, engine_bytes, manifest = approved_payloads()
        memory = MockMemory(fail_allocate_at=2)
        with self.assertRaises(E5CompatibilityError) as caught:
            execute_e5_v3_once(MockEngine(), MockContext(memory.events), memory, 8, input_payload, source, engine_bytes, manifest, "10.3.0.30")
        self.assertEqual(caught.exception.code, "E5_RUNTIME_CALL_FAILED")
        self.assertEqual(len([event for event in memory.events if event[0] == "free"]), 1)

    def test_missing_opencv_fails_closed_without_fallback(self):
        def missing(_name):
            raise ModuleNotFoundError("mock cv2 missing")
        with self.assertRaises(E5CompatibilityError) as caught:
            require_opencv(missing)
        self.assertEqual(caught.exception.code, "OPENCV_REQUIRED")
        self.assertIn("no fallback", str(caught.exception))

    def test_opencv_incomplete_api_is_rejected(self):
        with self.assertRaises(E5CompatibilityError) as caught:
            require_opencv(lambda _name: SimpleNamespace(imdecode=lambda *_args: None))
        self.assertEqual(caught.exception.code, "OPENCV_API_INCOMPLETE")

    def test_cuda12_candidate_selection_uses_injected_filesystem_and_loader(self):
        attempts = []
        chosen = CUDA12_LIBRARY_CANDIDATES[1]
        library = SimpleNamespace(**{name: (lambda *_args: 0) for name in CUDA_REQUIRED_SYMBOLS})
        resolver = Cuda12LibraryResolver(
            file_exists=lambda path: path == chosen,
            library_loader=lambda path: (attempts.append(path) or library),
            candidates=CUDA12_LIBRARY_CANDIDATES[:2],
        )
        path, observed = resolver.open()
        self.assertEqual(path, chosen)
        self.assertIs(observed, library)
        self.assertEqual(attempts, [chosen])
        self.assertFalse(any("11.4" in candidate for candidate in CUDA12_LIBRARY_CANDIDATES))

    def test_cuda12_candidate_with_missing_symbols_is_not_accepted(self):
        incomplete = SimpleNamespace(cudaMalloc=lambda *_args: 0)
        resolver = Cuda12LibraryResolver(
            file_exists=lambda _path: True,
            library_loader=lambda _path: incomplete,
            candidates=(CUDA12_LIBRARY_CANDIDATES[0],),
        )
        with self.assertRaises(E5CompatibilityError) as caught:
            resolver.open()
        self.assertEqual(caught.exception.code, "CUDA12_LIBRARY_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
