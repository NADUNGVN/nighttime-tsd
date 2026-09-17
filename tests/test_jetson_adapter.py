import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from edge_readiness.jetson_adapter import (  # noqa: E402
    AdapterHarnessBridge,
    AdapterConfig,
    AdapterError,
    DeviceTensor,
    EngineBinding,
    EngineDescriptor,
    JetsonRuntimeAdapter,
    MockBuffers,
    MockContext,
    MockStream,
    build_smoke_plan,
    expected_nbytes,
    fixture_manifest,
    fixture_to_input_bytes,
    make_host_tensor,
    make_train_only_fixture,
    validate_yolo11n_native_output_contract,
)
from edge_readiness.measurement_harness import (  # noqa: E402
    Boundary,
    MeasurementConfig,
    Preprocessed,
    mock_device,
    run_session,
)
import edge_readiness.jetson_adapter_smoke as smoke_orchestrator  # noqa: E402
from edge_readiness.jetson_adapter_smoke import run_cpu_mock_dry_run  # noqa: E402


def make_engine(runtime_version="8.5.2.2", input_dtype="float32", output_shape=(1, 7, 8400), output_name="output0", location="device"):
    return EngineDescriptor(
        runtime_version=runtime_version,
        engine_sha256="b" * 64,
        bindings=(
            EngineBinding("images", "input", (1, 3, 640, 640), input_dtype, location),
            EngineBinding(output_name, "output", output_shape, "float32", location),
        ),
    )


def make_adapter(target_id="E2", runtime_version=None, engine=None, allocations=None, context=None, stream_handle=91, buffer_stream_handle=None, events=None, **buffer_kwargs):
    events = events if events is not None else []
    runtime_version = runtime_version or ("8.5.2.2.mock" if target_id != "E5" else "10.3.0.mock")
    engine = engine or make_engine(runtime_version)
    stream = MockStream(handle=stream_handle, events=events)
    buffers = MockBuffers(engine, events, allocations=allocations, stream_handle=stream_handle if buffer_stream_handle is None else buffer_stream_handle, **buffer_kwargs)
    context = context or MockContext(events)
    return JetsonRuntimeAdapter(AdapterConfig(target_id, runtime_version), engine, context, stream, buffers), buffers, events


class JetsonAdapterTests(unittest.TestCase):
    def test_smoke_plan_labels_synthetic_fixture_and_unavailable_energy(self):
        plan = build_smoke_plan("E2")
        self.assertEqual(plan["status"], "proposed_not_executed")
        self.assertEqual(plan["native_output_contract"]["shape"], [1, 7, 8400])
        self.assertEqual(plan["fixture"]["split"], "synthetic_mock_fixture")
        self.assertEqual(plan["fixture"]["accepted_cctsdb_materialization"]["status"], "missing_prerequisite")
        self.assertEqual(plan["energy"]["status"], "unavailable_without_validated_power_boundary_and_clock_alignment")
        self.assertNotIn("missing power boundary or clock alignment", plan["stop_conditions"])

    def test_profiles_accept_observed_runtime_versions(self):
        AdapterConfig("E2", "8.5.2.2").validate()
        AdapterConfig("E3", "8.5.2.2-jetson").validate()
        AdapterConfig("E5", "10.3.0.30").validate()

    def test_config_rejects_unapproved_target_runtime_and_checkpoint(self):
        cases = (
            (AdapterConfig("E4", "8.5.2.2"), "UNKNOWN_TARGET"),
            (AdapterConfig("E2", "10.3.0"), "RUNTIME_VERSION_MISMATCH"),
            (AdapterConfig("E2", "8.5.2.2", checkpoint_sha256="c" * 64), "CHECKPOINT_HASH_MISMATCH"),
        )
        for config, code in cases:
            with self.subTest(code=code), self.assertRaises(AdapterError) as raised:
                config.validate()
            self.assertEqual(raised.exception.code, code)
            self.assertEqual(raised.exception.as_dict()["status"], "error")

    def test_buffer_owner_addresses_are_used_for_e2_v2_copy_and_enqueue(self):
        engine = make_engine()
        allocations = {
            "images": DeviceTensor("images", (1, 3, 640, 640), "float32", expected_nbytes((1, 3, 640, 640), "float32"), 0x7F000, "owner-a", "life-1"),
            "output0": DeviceTensor("output0", (1, 7, 8400), "float32", expected_nbytes((1, 7, 8400), "float32"), 0x12000, "owner-a", "life-1"),
        }
        adapter, _, events = make_adapter(engine=engine, allocations=allocations)
        adapter.infer(make_host_tensor(engine.input_binding(), b"\0" * expected_nbytes((1, 3, 640, 640), "float32")))
        self.assertIn("copy_h2d:images:520192:4915200", events)
        self.assertIn("execute_async_v2:520192,73728:91", events)
        self.assertIn("copy_d2h:output0:73728:235200", events)

    def test_buffer_owner_addresses_are_used_for_e5_v3(self):
        engine = make_engine("10.3.0.mock")
        allocations = {
            "images": DeviceTensor("images", (1, 3, 640, 640), "float32", 4915200, 0xABC000, "owner-e5", "life-5"),
            "output0": DeviceTensor("output0", (1, 7, 8400), "float32", 235200, 0xDEF000, "owner-e5", "life-5"),
        }
        adapter, _, events = make_adapter("E5", engine=engine, allocations=allocations)
        adapter.infer(make_host_tensor(engine.input_binding(), b"\0" * 4915200))
        self.assertIn("set_tensor_address:images:11255808", events)
        self.assertIn("set_tensor_address:output0:14610432", events)
        self.assertIn("copy_h2d:images:11255808:4915200", events)
        self.assertIn("copy_d2h:output0:14610432:235200", events)

    def test_output_payload_is_unavailable_until_final_completion(self):
        engine = make_engine()
        events = []
        stream = MockStream(handle=91, events=events)
        buffers = MockBuffers(engine, events, stream_handle=91)
        with self.assertRaises(AdapterError) as raised:
            buffers.output_tensors()
        self.assertEqual(raised.exception.code, "OUTPUT_NOT_READY")
        adapter = JetsonRuntimeAdapter(AdapterConfig("E2", "8.5.2.2.mock"), engine, MockContext(events), stream, buffers)
        outputs = adapter.infer(make_host_tensor(engine.input_binding(), b"\0" * 4915200))
        self.assertEqual(len(outputs["output0"].payload), 235200)
        self.assertIn("sync:after_output_copy", events)

    def test_constructor_rejects_missing_duplicate_wrong_role_location_and_stream(self):
        engine = make_engine()
        base = {
            "images": DeviceTensor("images", (1, 3, 640, 640), "float32", 4915200, 1, "owner", "life"),
            "output0": DeviceTensor("output0", (1, 7, 8400), "float32", 235200, 2, "owner", "life"),
        }
        with self.assertRaisesRegex(AdapterError, "BUFFER_BINDING_SET_MISMATCH"):
            make_adapter(engine=engine, allocations={"images": base["images"]})
        with self.assertRaisesRegex(AdapterError, "DUPLICATE_DEVICE_POINTER"):
            make_adapter(engine=engine, allocations={**base, "output0": DeviceTensor("output0", (1, 7, 8400), "float32", 235200, 1, "owner", "life")})
        with self.assertRaisesRegex(AdapterError, "BUFFER_LIFETIME_MISSING"):
            make_adapter(engine=engine, allocations={**base, "images": DeviceTensor("images", (1, 3, 640, 640), "float32", 4915200, 1, "", "life")})
        with self.assertRaisesRegex(AdapterError, "COPY_STREAM_MISMATCH"):
            make_adapter(engine=engine, buffer_stream_handle=92)

        bad_role = EngineDescriptor(engine.runtime_version, engine.engine_sha256, (EngineBinding("images", "input", (1, 3, 640, 640), "float32"), EngineBinding("output0", "scratch", (1, 7, 8400), "float32")))
        with self.assertRaisesRegex(AdapterError, "UNSUPPORTED_BINDING_ROLE"):
            make_adapter(engine=bad_role)
        bad_location = make_engine(location="host")
        with self.assertRaisesRegex(AdapterError, "UNSUPPORTED_BINDING_LOCATION"):
            make_adapter(engine=bad_location)

    def test_named_address_failure_and_copy_failures_are_structured(self):
        engine = make_engine("10.3.0.mock")
        events = []
        stream = MockStream(handle=91, events=events)
        buffers = MockBuffers(engine, events, stream_handle=91)
        adapter = JetsonRuntimeAdapter(AdapterConfig("E5", "10.3.0.mock"), engine, MockContext(events, fail_named_address="output0"), stream, buffers)
        with self.assertRaisesRegex(AdapterError, "SET_TENSOR_ADDRESS_FAILED"):
            adapter.infer(make_host_tensor(engine.input_binding(), b"\0" * 4915200))
        adapter, _, _ = make_adapter(fail_h2d=True)
        with self.assertRaisesRegex(AdapterError, "MOCK_H2D_FAILED"):
            adapter.infer(make_host_tensor(adapter.engine.input_binding(), b"\0" * 4915200))
        adapter, _, _ = make_adapter(fail_d2h=True)
        with self.assertRaisesRegex(AdapterError, "MOCK_D2H_FAILED"):
            adapter.infer(make_host_tensor(adapter.engine.input_binding(), b"\0" * 4915200))

    def test_model_output_contract_rejects_generic_80_class_shape(self):
        engine = make_engine(output_shape=(1, 84, 8400))
        config = AdapterConfig("E2", "8.5.2.2")
        with self.assertRaisesRegex(AdapterError, "MODEL_OUTPUT_CONTRACT_MISMATCH"):
            validate_yolo11n_native_output_contract(engine, config)
        self.assertEqual(validate_yolo11n_native_output_contract(make_engine(), config)["native_output_shape"], [1, 7, 8400])

    def test_adapter_bridge_runs_through_harness_both_boundaries(self):
        images = make_train_only_fixture()
        pool_ids = tuple(image.image_id for image in images)
        import hashlib
        pool_hash = hashlib.sha256("\n".join(pool_ids).encode()).hexdigest()
        for boundary in (Boundary.INFERENCE_ONLY, Boundary.DECODED_IMAGE_TO_DETECTIONS):
            adapter, _, events = make_adapter()
            bridge = AdapterHarnessBridge(adapter)
            config = MeasurementConfig("E2", bridge.backend_name, adapter.config.model_id, adapter.config.checkpoint_sha256, boundary=boundary, warmup_calls=1, measured_calls=3, session_count=1, pool_ids=pool_ids, pool_hash=pool_hash, pool_order=(0, 1, 2))
            seen = []
            result = run_session(config, bridge, lambda index: seen.append(index) or images[index], lambda image: Preprocessed(fixture_to_input_bytes(image), (1, 3, 640, 640)), lambda output: {"nms": False}, session_index=1, device=mock_device("E2"))
            self.assertEqual(result["status"], "completed_mock")
            self.assertEqual(result["measured_window"]["image_count"], 3)
            self.assertFalse(result["measured_window"]["includes_warmup"])
            self.assertEqual(seen, [0, 0, 1, 2])
            self.assertEqual(result["image_pool"]["consumed"]["measured"]["ids"], ["mock-synthetic-00", "mock-synthetic-01", "mock-synthetic-02"])
            self.assertEqual(result["boundary"], boundary.value)
            self.assertTrue(any(item.startswith("sync:after_output_copy") for item in events))

    def test_cli_dry_run_emits_manifest_layout_without_real_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_cpu_mock_dry_run(Path(temp), target_id="E2")
            self.assertEqual(result["status"], "completed_mock_dry_run")
            self.assertFalse(result["real_device_execution"])
            self.assertTrue((Path(temp) / "manifest.json").exists())
            self.assertTrue((Path(temp) / "inference_only" / "session_1.json").exists())
            self.assertTrue((Path(temp) / "decoded_image_to_detections" / "adapter_events.json").exists())

    def test_cli_failure_writes_structured_failure_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            original = smoke_orchestrator.run_cpu_mock_dry_run
            smoke_orchestrator.run_cpu_mock_dry_run = lambda *_args, **_kwargs: (_ for _ in ()).throw(AdapterError("MOCK_FAILURE", "injected test failure"))
            try:
                code = smoke_orchestrator.main(["--target", "E2", "--out-dir", temp, "--dry-run"])
            finally:
                smoke_orchestrator.run_cpu_mock_dry_run = original
            self.assertEqual(code, 2)
            failure = (Path(temp) / "failure.json").read_text(encoding="utf-8")
            self.assertIn("MOCK_FAILURE", failure)

    def test_fixture_manifest_is_deterministic_and_not_train_evidence(self):
        first = make_train_only_fixture()
        second = make_train_only_fixture()
        self.assertEqual(first, second)
        manifest = fixture_manifest(first)
        self.assertEqual(manifest["image_order"], ["mock-synthetic-00", "mock-synthetic-01", "mock-synthetic-02"])
        self.assertEqual(manifest["images"][0]["derived_input_sha256"], __import__("hashlib").sha256(fixture_to_input_bytes(first[0])).hexdigest())
        self.assertFalse(manifest["official_test_used"])


if __name__ == "__main__":
    unittest.main()
