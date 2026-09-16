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
    mock_pool,
)


class EdgeMeasurementHarnessTests(unittest.TestCase):
    def config(self, **kwargs):
        pool_ids, pool_hash, pool_order = mock_pool()
        values = {"target_id": "E1", "backend": "cpu_fp32_reference", "model_id": "fixture", "model_sha256": "a" * 64, "pool_ids": pool_ids, "pool_hash": pool_hash, "pool_order": pool_order}
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
            {"monotonic_ns": 0, "power_w": 10.0, "unit": "W", "boundary": "whole_device", "clock_identity": "mono-1", "alignment_source": "same-process"},
            {"monotonic_ns": 1_000_000_000, "power_w": 20.0, "unit": "W", "boundary": "whole_device", "clock_identity": "mono-1", "alignment_source": "same-process"},
        ]
        result = integrate_power_energy(rows, 0, 1_000_000_000)
        self.assertEqual(result["status"], "measured")
        self.assertAlmostEqual(result["energy_j"], 15.0)
        self.assertEqual(integrate_power_energy([], 0, 1)["status"], "unavailable")
        with self.assertRaisesRegex(HarnessError, "MIXED_POWER_BOUNDARY"):
            integrate_power_energy(rows + [{"monotonic_ns": 2_000_000_000, "power_w": 20, "unit": "W", "boundary": "module", "clock_identity": "mono-1", "alignment_source": "same-process"}], 0, 2_000_000_000)

    def test_power_full_and_clipped_ramp_use_interpolated_boundary_values(self):
        rows = [
            {"monotonic_ns": 0, "power_w": 0.0, "unit": "W", "boundary": "whole_device", "clock_identity": "mono-1", "alignment_source": "same-process"},
            {"monotonic_ns": 10_000_000_000, "power_w": 10.0, "unit": "W", "boundary": "whole_device", "clock_identity": "mono-1", "alignment_source": "same-process"},
        ]
        full = integrate_power_energy(rows, 0, 10_000_000_000)
        clipped = integrate_power_energy(rows, 0, 2_000_000_000)
        self.assertEqual(full["status"], "measured")
        self.assertAlmostEqual(full["energy_j"], 50.0)
        self.assertEqual(clipped["status"], "measured")
        self.assertAlmostEqual(clipped["energy_j"], 2.0)

    def test_power_partial_coverage_never_claims_measured_full_interval(self):
        rows = [
            {"monotonic_ns": 2_000_000_000, "power_w": 10.0, "unit": "W", "boundary": "whole_device", "clock_identity": "mono-1", "alignment_source": "same-process"},
            {"monotonic_ns": 8_000_000_000, "power_w": 10.0, "unit": "W", "boundary": "whole_device", "clock_identity": "mono-1", "alignment_source": "same-process"},
        ]
        result = integrate_power_energy(rows, 0, 10_000_000_000)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["coverage"]["full_interval_covered"])
        self.assertEqual(result["coverage"]["covered_duration_ns"], 6_000_000_000)
        self.assertAlmostEqual(result["energy_j"], 60.0)
        self.assertNotIn("average_power_w", result)

    def test_power_no_overlap_and_invalid_order_or_values_fail_closed(self):
        base = {"unit": "W", "boundary": "whole_device", "clock_identity": "mono-1", "alignment_source": "same-process"}
        no_overlap = [{**base, "monotonic_ns": 20, "power_w": 1}, {**base, "monotonic_ns": 30, "power_w": 1}]
        self.assertEqual(integrate_power_energy(no_overlap, 0, 10)["status"], "unavailable")
        for rows, code in (
            ([{**base, "monotonic_ns": 0, "power_w": 1}, {**base, "monotonic_ns": 0, "power_w": 1}], "NONMONOTONIC_TIMESTAMPS"),
            ([{**base, "monotonic_ns": 10, "power_w": 1}, {**base, "monotonic_ns": 0, "power_w": 1}], "NONMONOTONIC_TIMESTAMPS"),
            ([{**base, "monotonic_ns": 0, "power_w": -1}, {**base, "monotonic_ns": 1, "power_w": 1}], "INVALID_POWER_SAMPLE"),
            ([{**base, "monotonic_ns": 0, "power_w": float("nan")}, {**base, "monotonic_ns": 1, "power_w": 1}], "NONFINITE_SAMPLE"),
            ([{**base, "monotonic_ns": 0, "power_w": 1}, {**base, "monotonic_ns": 1, "power_w": 1, "clock_identity": "other"}], "MIXED_CLOCK_ALIGNMENT"),
            ([{**base, "monotonic_ns": 0, "power_w": 1}, {**base, "monotonic_ns": 1, "power_w": 1, "alignment_source": "unverified"}], "UNVERIFIED_CLOCK_ALIGNMENT"),
            ([{**base, "monotonic_ns": 0, "power_w": 1}, {**base, "monotonic_ns": 1, "power_w": 1, "unit": "mW"}], "MIXED_POWER_UNIT"),
            ([{**base, "monotonic_ns": 0, "power_w": 1}, {**base, "monotonic_ns": 1, "power_w": 1, "clock_identity": None}], "UNVERIFIED_CLOCK_ALIGNMENT"),
        ):
            with self.subTest(code=code), self.assertRaisesRegex(HarnessError, code):
                integrate_power_energy(rows, 0, 1)

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
        self.assertGreater(result["session_duration_ns"], 0)
        self.assertTrue(result["energy_interval"]["includes_provider_overhead"])
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
