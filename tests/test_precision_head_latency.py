import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_precision_head_latency import (  # noqa: E402
    ENGINE_KEYS,
    MEASURED_CALLS,
    RUNTIME_OPTIONS,
    WARMUP_CALLS,
    _aggregate_engine_sessions,
    child_command,
    cyclic_sequence,
    engine_schedule,
    ensure_output_absent,
    latency_statistics,
    make_image_pool_manifest,
    measure_predict,
    percentile_linear,
    select_image_pool,
    sha256_file,
    validate_engine_file,
    validate_output_target,
    validate_schedule,
    write_json,
)


class PrecisionHeadLatencyTests(unittest.TestCase):
    def test_schedule_has_39_serial_sessions_and_locked_rotations(self):
        schedule = engine_schedule()
        validate_schedule(schedule)
        self.assertEqual(len(schedule), 39)
        self.assertEqual([row["engine_key"] for row in schedule[:13]], list(ENGINE_KEYS))
        self.assertEqual([row["engine_key"] for row in schedule[13:26]], list(ENGINE_KEYS[4:]) + list(ENGINE_KEYS[:4]))
        self.assertEqual([row["engine_key"] for row in schedule[26:]], list(ENGINE_KEYS[8:]) + list(ENGINE_KEYS[:8]))
        for round_id in (1, 2, 3):
            self.assertEqual({row["engine_key"] for row in schedule if row["round"] == round_id}, set(ENGINE_KEYS))

    def test_selection_is_deterministic_without_replacement_and_sequence_is_shared(self):
        names = [f"{index:04d}.jpg" for index in range(1636)]
        first = select_image_pool(names)
        second = select_image_pool(names)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 256)
        self.assertEqual(len(set(first)), 256)
        self.assertEqual(cyclic_sequence(first, 1000), cyclic_sequence(first, 1000))

    def test_pool_manifest_records_image_hashes_and_fixed_stream(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = [f"{index:04d}.jpg" for index in range(1636)]
            for index, name in enumerate(names):
                (root / name).write_bytes(f"image-{index}".encode())
            manifest = make_image_pool_manifest(names, root)
            self.assertEqual(manifest["pool_size"], 256)
            self.assertEqual(len(manifest["selected_files"]), 256)
            selected = manifest["selected_files"][0]
            self.assertEqual(selected["sha256"], sha256_file(root / selected["image"]))
            self.assertEqual(len(manifest["measured_sequence"]), 1000)
            self.assertTrue(manifest["same_sequence_for_every_engine"])

    def test_timer_boundary_excludes_warmup_and_synchronizes(self):
        events = []
        clock = iter([100, 1_100, 2_100, 3_100, 4_100, 5_100])
        calls = []

        def predict(image):
            calls.append(image)
            events.append("predict")

        def sync():
            events.append("sync")

        values = measure_predict(None, ["a", "b"], predict, sync, timer_ns=lambda: next(clock), warmup=2, samples=2)
        self.assertEqual(calls, ["a", "b", "a", "b"])
        self.assertEqual(values, [0.001, 0.001])
        self.assertEqual(events, ["predict", "predict", "sync", "sync", "predict", "sync", "sync", "predict", "sync"])

    def test_percentile_and_raw_aggregation_are_linear_and_include_all_samples(self):
        values = [1.0, 2.0, 4.0, 8.0]
        self.assertEqual(percentile_linear(values, 0.50), 3.0)
        stats = latency_statistics(values)
        self.assertEqual(stats["n_calls"], 4)
        self.assertEqual(stats["p95_ms"], float(np.quantile(values, 0.95, method="linear")))
        self.assertAlmostEqual(stats["serial_fps"], 1000.0 / 3.75)

    def test_runtime_options_are_explicit_and_not_old_benchmark_defaults(self):
        self.assertEqual(RUNTIME_OPTIONS, {
            "imgsz": 640, "batch": 1, "conf": 0.001, "iou": 0.7,
            "max_det": 300, "rect": False, "task": "detect", "verbose": False,
        })

    def test_engine_missing_and_changed_bytes_are_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            missing = root / "missing.engine"
            with self.assertRaises(FileNotFoundError):
                validate_engine_file(missing, "x")
            engine = root / "engine.engine"
            engine.write_bytes(b"original")
            digest = sha256_file(engine)
            self.assertEqual(validate_engine_file(engine, digest), digest)
            engine.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_engine_file(engine, digest)

    def test_parent_child_command_is_one_fresh_session_and_not_concurrent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = {"engine_key": "bbox_fp32_1", "engine_path": "/srv/bbox.engine", "engine_sha256": "hash"}
            command = child_command(root, spec, 2, root / "round_2/bbox_fp32_1", root / "study_manifest.json", root / "image_pool_manifest.json", root / "images", "0", {7: "/usr/bin/Xorg"})
            self.assertIn("--child", command)
            self.assertEqual(command.count("--engine-key"), 1)
            self.assertEqual(command.count("--round"), 1)
            self.assertEqual(command.count("--confirm-desktop-process"), 1)

    def test_engine_session_aggregation_requires_three_rounds(self):
        base = {"engine_key": "bbox_fp32_1", "engine_sha256": "h", "engine_bytes": 4,
                "warmup_calls": WARMUP_CALLS, "measured_calls": MEASURED_CALLS,
                "gpu_identity_binding": {}, "environment": {}}
        sessions = [dict(base, round=round_id, raw_latency_ms=[float(round_id), float(round_id + 1)]) for round_id in (1, 2, 3)]
        for session in sessions:
            session["latency_ms"] = latency_statistics(session["raw_latency_ms"])
        result = _aggregate_engine_sessions(sessions)
        self.assertEqual(result["pooled_calls_3000"]["n_calls"], 6)
        with self.assertRaises(ValueError):
            _aggregate_engine_sessions(sessions[:2])

    def test_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "output"
            path.mkdir()
            with self.assertRaises(FileExistsError):
                ensure_output_absent(path)
            target = Path(temp) / "new.json"
            write_json(target, {"ok": True})
            with self.assertRaises(FileExistsError):
                write_json(target, {"changed": True})

    def test_output_path_is_locked_to_latency_study_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            with self.assertRaises(ValueError):
                validate_output_target(repo, repo / "other")


if __name__ == "__main__":
    unittest.main()
