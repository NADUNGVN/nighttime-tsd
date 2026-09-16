import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from edge_readiness.measurement_harness import (  # noqa: E402
    Boundary,
    DeviceBinding,
    HarnessError,
    MeasurementConfig,
    MockBackend,
    Preprocessed,
    DEFAULT_INPUT_SHAPE,
    integrate_power_energy,
    mock_device,
    percentile_linear,
    run_mock_sessions,
    run_session,
    summarize_latencies,
    unavailable,
    write_json_no_overwrite,
)


class EdgeMeasurementHarnessTests(unittest.TestCase):
    def config(self, **kwargs):
        values = {"target_id": "E1", "backend": "cpu_fp32_reference", "model_id": "fixture", "model_sha256": "a" * 64}
        values.update(kwargs)
        return MeasurementConfig(**values)

    def test_config_fails_closed_for_shape_batch_hash_and_disk_decode(self):
        for kwargs, code in (
            ({"input_shape": (1, 3, 320, 320)}, "INVALID_INPUT_SHAPE"),
            ({"batch": 2}, "INVALID_BATCH"),
            ({"model_sha256": "bad"}, "INVALID_MODEL_HASH"),
            ({"decoded_from_disk_in_timing": True}, "DISK_DECODE_BOUNDARY"),
        ):
            with self.subTest(code=code), self.assertRaisesRegex(HarnessError, code):
                self.config(**kwargs).validate()

    def test_percentiles_are_linear_and_summary_distinguishes_throughput(self):
        self.assertAlmostEqual(percentile_linear([1, 2, 3, 4], 0.95), 3.85)
        summary = summarize_latencies([1, 2, 3, 4])
        self.assertEqual(summary["n_calls"], 4)
        self.assertIn("serial_fps", summary)
        self.assertEqual(summary["pipelined_throughput_fps"]["status"], "unavailable")

    def test_power_integration_requires_actual_boundary_and_returns_joules(self):
        rows = [
            {"monotonic_ns": 0, "power_w": 10.0, "unit": "W", "boundary": "whole_device"},
            {"monotonic_ns": 1_000_000_000, "power_w": 20.0, "unit": "W", "boundary": "whole_device"},
        ]
        result = integrate_power_energy(rows, 0, 1_000_000_000)
        self.assertEqual(result["status"], "measured")
        self.assertAlmostEqual(result["energy_j"], 15.0)
        self.assertEqual(integrate_power_energy([], 0, 1)["status"], "unavailable")
        with self.assertRaisesRegex(HarnessError, "MIXED_POWER_BOUNDARY"):
            integrate_power_energy(rows + [{"monotonic_ns": 2_000_000_000, "power_w": 20, "unit": "W", "boundary": "module"}], 0, 2_000_000_000)

    def test_inference_only_excludes_postprocess_and_records_sync_order(self):
        backend = MockBackend(output={"ok": True})
        result = run_session(
            self.config(measured_calls=3, warmup_calls=2, boundary=Boundary.INFERENCE_ONLY),
            backend,
            lambda index: index,
            lambda image: Preprocessed(image, DEFAULT_INPUT_SHAPE),
            lambda output: {"detections": output},
            session_index=1,
            device=mock_device(),
        )
        self.assertEqual(result["warmup_calls"], 2)
        self.assertEqual(result["measured_calls"], 3)
        self.assertEqual(len(result["raw_latency_ms"]), 3)
        self.assertFalse(result["warmup_in_timing"])
        self.assertEqual(result["power_energy"]["status"], "unavailable")
        self.assertEqual(backend.events.count("infer"), 5)
        self.assertEqual(backend.events[:3], ["sync:before_timer", "infer", "sync:after_inference"])

    def test_decoded_to_detections_includes_pre_and_postprocess(self):
        backend = MockBackend(output={"boxes": 1})
        seen = []
        result = run_session(
            self.config(measured_calls=2, warmup_calls=0, boundary=Boundary.DECODED_IMAGE_TO_DETECTIONS),
            backend,
            lambda index: {"image": index},
            lambda image: (seen.append("pre"), Preprocessed(image, DEFAULT_INPUT_SHAPE))[1],
            lambda output: (seen.append("post"), output)[1],
            session_index=1,
            device=mock_device(),
        )
        self.assertEqual(len(result["raw_latency_ms"]), 2)
        self.assertEqual(seen, ["pre", "post", "pre", "post"])
        self.assertEqual(result["boundary"], Boundary.DECODED_IMAGE_TO_DETECTIONS.value)

    def test_mock_cli_shape_creates_three_sessions_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = run_mock_sessions(root, measured_calls=2, warmup_calls=1)
            self.assertEqual(len(manifest["sessions"]), 3)
            self.assertTrue((root / "manifest.json").is_file())
            with self.assertRaisesRegex(HarnessError, "OUTPUT_EXISTS"):
                write_json_no_overwrite(root / "manifest.json", {})
            json.loads((root / "session_1.json").read_text(encoding="utf-8"))

    def test_negative_states_are_structured(self):
        self.assertEqual(unavailable("not measured", scope="board")["status"], "unavailable")
        wrong_device = DeviceBinding("E2", "wrong", {}, {}, {}, {}, {}, {}, {})
        with self.assertRaisesRegex(HarnessError, "DEVICE_BINDING_MISMATCH"):
            run_session(self.config(measured_calls=1, warmup_calls=0), MockBackend(), lambda _: 1, lambda _: Preprocessed(1, DEFAULT_INPUT_SHAPE), lambda x: x, session_index=1, device=wrong_device)


if __name__ == "__main__":
    unittest.main()
