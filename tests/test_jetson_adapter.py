import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from edge_readiness.jetson_adapter import (  # noqa: E402
    AdapterConfig,
    AdapterError,
    EngineBinding,
    EngineDescriptor,
    JetsonRuntimeAdapter,
    MockBuffers,
    MockContext,
    MockStream,
    build_smoke_plan,
    expected_nbytes,
    fixture_manifest,
    make_host_tensor,
    make_train_only_fixture,
)


def make_engine(runtime_version="8.5.2.2", input_dtype="float32"):
    return EngineDescriptor(
        runtime_version=runtime_version,
        engine_sha256="b" * 64,
        bindings=(
            EngineBinding("images", "input", (1, 3, 640, 640), input_dtype),
            EngineBinding("output0", "output", (1, 84, 8400), "float32"),
        ),
    )


class JetsonAdapterTests(unittest.TestCase):
    def test_smoke_plan_is_proposed_and_train_only(self):
        plan = build_smoke_plan("E2")
        self.assertEqual(plan["status"], "proposed_not_executed")
        self.assertEqual(plan["target"]["id"], "E2")
        self.assertEqual(plan["model"]["checkpoint_sha256"], "3e5fc7a2148c16539cd9fb7cc7cacd81a4eec1dfc28143cdf9b6dcd872ba4ab8")
        self.assertFalse(plan["fixture"]["official_test_used"])
        self.assertIn("model load/forward", plan["forbidden_actions"])
        self.assertEqual(plan["fixture"]["image_order"], ["train-fixture-00", "train-fixture-01", "train-fixture-02"])

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
            with self.subTest(code=code):
                with self.assertRaises(AdapterError) as raised:
                    config.validate()
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(raised.exception.as_dict()["status"], "error")

    def test_e2_mock_binding_copy_enqueue_and_final_sync(self):
        events = []
        engine = make_engine()
        adapter = JetsonRuntimeAdapter(
            AdapterConfig("E2", "8.5.2.2"),
            engine,
            MockContext(events),
            MockStream(events=events),
            MockBuffers(engine, events),
        )
        binding = engine.input_binding()
        host = make_host_tensor(binding, b"\0" * expected_nbytes(binding.shape, binding.dtype))

        outputs = adapter.infer(host)

        self.assertIn("output0", outputs)
        self.assertEqual(
            events,
            [
                "sync:before_input_copy",
                "copy_h2d:images:4915200",
                "sync:before_enqueue",
                "execute_async_v2:2:7",
                "sync:after_enqueue",
                "copy_d2h:output0:2822400",
                "sync:after_output_copy",
            ],
        )

    def test_e5_uses_named_tensor_v3_path(self):
        events = []
        engine = make_engine("10.3.0.30", input_dtype="float16")
        adapter = JetsonRuntimeAdapter(
            AdapterConfig("E5", "10.3.0.30", input_dtype="float16"),
            engine,
            MockContext(events),
            MockStream(events=events),
            MockBuffers(engine, events),
        )
        binding = engine.input_binding()
        adapter.infer(make_host_tensor(binding, b"\0" * expected_nbytes(binding.shape, binding.dtype)))

        self.assertEqual(
            events,
            [
                "sync:before_input_copy",
                "copy_h2d:images:2457600",
                "sync:before_enqueue",
                "set_tensor_address:images:1",
                "set_tensor_address:output0:2",
                "execute_async_v3:7",
                "sync:after_enqueue",
                "copy_d2h:output0:2822400",
                "sync:after_output_copy",
            ],
        )

    def test_input_contract_rejects_wrong_bytes(self):
        engine = make_engine()
        binding = engine.input_binding()
        with self.assertRaises(AdapterError) as raised:
            make_host_tensor(binding, b"short", nbytes=5)
        self.assertEqual(raised.exception.code, "BUFFER_SIZE_MISMATCH")

        events = []
        adapter = JetsonRuntimeAdapter(
            AdapterConfig("E2", "8.5.2.2"), engine, MockContext(events), MockStream(events=events), MockBuffers(engine, events)
        )
        bad = make_host_tensor(binding, b"\0" * expected_nbytes(binding.shape, binding.dtype))
        object.__setattr__(bad, "dtype", "float16")
        with self.assertRaises(AdapterError) as raised:
            adapter.infer(bad)
        self.assertEqual(raised.exception.code, "INPUT_CONTRACT_MISMATCH")

    def test_enqueue_failure_stops_before_output_copy(self):
        events = []
        engine = make_engine()
        adapter = JetsonRuntimeAdapter(
            AdapterConfig("E2", "8.5.2.2"),
            engine,
            MockContext(events, enqueue_result=False),
            MockStream(events=events),
            MockBuffers(engine, events),
        )
        binding = engine.input_binding()
        with self.assertRaises(AdapterError) as raised:
            adapter.infer(make_host_tensor(binding, b"\0" * expected_nbytes(binding.shape, binding.dtype)))
        self.assertEqual(raised.exception.code, "ENQUEUE_FAILED")
        self.assertNotIn("copy_d2h:output0:2822400", events)

    def test_fixture_manifest_is_deterministic_and_order_bound(self):
        first = make_train_only_fixture()
        second = make_train_only_fixture()
        self.assertEqual(first, second)
        manifest = fixture_manifest(first)
        self.assertEqual(len(manifest["images"]), 3)
        self.assertEqual(manifest["image_order"], [image.image_id for image in first])
        self.assertEqual(manifest["images"][0]["sha256"], first[0].sha256)
        self.assertFalse(manifest["official_test_used"])


if __name__ == "__main__":
    unittest.main()
