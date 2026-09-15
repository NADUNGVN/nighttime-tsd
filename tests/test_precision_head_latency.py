import sys
import json
import copy
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

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
    _validate_session,
    validate_complete_session_set,
    validate_engine_file,
    validate_build_provenance,
    validate_output_target,
    validate_schedule,
    parse_locked_dev_yaml,
    _validate_persisted_gpu_evidence,
    write_json,
)
import run_precision_head_latency as latency  # noqa: E402


def producer_snapshot_fixture():
    """Create a real uniform_build_repeat.snapshot payload without touching a GPU."""
    import importlib

    # uniform_build_repeat imports the GPU capture producer at module import time;
    # replace only that unrelated heavy module so this producer contract test
    # remains CPU-only on the local inspection environment.
    producer_import = ModuleType("capture_cctsdb_validator")
    producer_import.FROZEN_WEIGHTS_SHA256 = "test-frozen-weights"
    producer_import.write_json = lambda *_args, **_kwargs: None
    with patch.dict(sys.modules, {"capture_cctsdb_validator": producer_import}):
        uniform_build_repeat = importlib.import_module("uniform_build_repeat")
    sys.modules["uniform_build_repeat"] = uniform_build_repeat
    snapshot = uniform_build_repeat.snapshot

    device = "GPU-test, Quadro RTX 8000, 595.71.05, P8, 30, 9 W, 300 MHz, 405 MHz, 29 MiB"
    with patch.object(uniform_build_repeat.subprocess, "run", side_effect=[
        SimpleNamespace(stdout=device),
        SimpleNamespace(stdout=""),
    ]):
        return snapshot({})


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

    def test_image_path_resolution_is_repo_anchored_and_shape_validation_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            default_path, default_binding = latency.resolve_images_dir(repo, None)
            self.assertEqual(default_path, (repo / latency.DEFAULT_IMAGES_DIR).resolve())
            self.assertEqual(default_binding["resolution"], "repo-relative locked default")
            relocated, binding = latency.resolve_images_dir(repo, Path("relocated/images"))
            self.assertEqual(relocated, (repo / "relocated/images").resolve())
            self.assertEqual(binding["declared"], "relocated/images")
            self.assertEqual(latency.validate_decoded_shape(Path("image.jpg"), [720, 1280], lambda _: [720, 1280]), [720, 1280])
            with self.assertRaises(ValueError):
                latency.validate_decoded_shape(Path("image.jpg"), [720, 1280], lambda _: [1280, 720])
            with self.assertRaises(ValueError):
                latency.validate_decoded_shape(Path("image.jpg"), [720, 1280], lambda _: [720, 1280, 3])

    def test_pinned_canonical_dev_yaml_is_parsed_with_posix_semantics(self):
        repo = Path(__file__).resolve().parents[1]
        raw = latency.git_blob(repo, latency.PINNED_ATTEMPT2_COMMIT, repo / latency.FP16_DATA_REFERENCE)
        contract = parse_locked_dev_yaml(raw)
        self.assertEqual(contract["train"], "images")
        self.assertEqual(contract["val"], "images")
        self.assertEqual(contract["names"], latency.LOCKED_DEV_NAMES)
        self.assertTrue(contract["path"].endswith("/data/processed/cctsdb2021_clean/dev"))
        for bad in (raw.decode().replace("val: images", "val: test"),
                    raw.decode().replace("cctsdb2021_clean/dev", "cctsdb2021_clean/test")):
            with self.assertRaises(ValueError):
                parse_locked_dev_yaml(bad)

    def test_real_producer_snapshot_schema_reaches_consumer_and_missing_key_fails(self):
        state = producer_snapshot_fixture()
        expected_gpu = {"uuid": "GPU-test", "name": "Quadro RTX 8000", "driver_version": "595.71.05"}
        binding = latency.validate_gpu_identity(state, expected_gpu)
        _validate_persisted_gpu_evidence(state, binding, expected_gpu, "Producer snapshot")
        self.assertFalse(state["process_guard"]["external_workload_detected"])
        self.assertNotIn("external_gpu_workload_detected", state["process_guard"])
        for mutation in ("missing", "true", "blocked", "limited"):
            bad = copy.deepcopy(state)
            if mutation == "missing":
                del bad["process_guard"]["external_workload_detected"]
            elif mutation == "true":
                bad["process_guard"]["external_workload_detected"] = True
            elif mutation == "blocked":
                bad["process_guard"]["blocked_processes"] = [{"allowed": False}]
            else:
                bad["process_guard"]["telemetry_status"] = "limited"
            with self.assertRaises(ValueError):
                _validate_persisted_gpu_evidence(bad, binding, expected_gpu, f"bad-{mutation}")

    def test_accepted_canonical_snapshot_reaches_consumer(self):
        repo = Path(__file__).resolve().parents[1]
        manifest = latency.read_json_blob(
            repo,
            latency.PINNED_ATTEMPT2_COMMIT,
            repo / "results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1_attempt2/study_manifest.json",
        )
        snapshot = manifest["gpu_before"]
        expected_gpu = latency.parse_gpu_identity(snapshot)
        binding = latency.validate_gpu_identity(snapshot, expected_gpu)
        _validate_persisted_gpu_evidence(snapshot, binding, expected_gpu, "Accepted canonical snapshot")
        self.assertFalse(snapshot["process_guard"]["external_workload_detected"])

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

    def test_build_provenance_checks_flat_and_nested_cache_links(self):
        source = {
            "study": "accepted", "environment": {"gpu": "Quadro RTX 8000"},
            "source_result_commit": "source", "calibration_cache_input": {"sha256": "cal"},
            "timing_cache_input": {"sha256": "timing"}, "settings": {"imgsz": 640},
        }
        build = {
            "study": "accepted", "arm": "bbox_fp32", "repeat": 1, "termination_status": "completed",
            "environment": {"gpu": "Quadro RTX 8000"}, "source_result_commit": "source",
            "settings": {"imgsz": 640}, "calibration_cache_input_sha256": "cal",
            "timing_cache_input_sha256": "timing", "calibration_cache_input": {"sha256": "cal"},
            "timing_cache_input": {"sha256": "timing"},
        }
        validate_build_provenance(build, source, "bbox_fp32", 1)
        for field, value in (("timing_cache_input_sha256", "changed"), ("source_result_commit", "mutated")):
            mutated = copy.deepcopy(build)
            mutated[field] = value
            with self.assertRaises(ValueError):
                validate_build_provenance(mutated, source, "bbox_fp32", 1)
        mutated = copy.deepcopy(build)
        mutated["calibration_cache_input"] = {"sha256": "changed"}
        with self.assertRaises(ValueError):
            validate_build_provenance(mutated, source, "bbox_fp32", 1)

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
        sessions = [dict(base, round=round_id, raw_latency_ms=[float(round_id)] * MEASURED_CALLS) for round_id in (1, 2, 3)]
        for session in sessions:
            session["latency_ms"] = latency_statistics(session["raw_latency_ms"])
        result = _aggregate_engine_sessions(sessions)
        self.assertEqual(result["pooled_calls_3000"]["n_calls"], 3000)
        with self.assertRaises(ValueError):
            _aggregate_engine_sessions(sessions[:2])

    def test_complete_session_set_rejects_missing_duplicate_and_wrong_rounds(self):
        schedule = engine_schedule()
        sessions = [{"round": item["round"], "engine_key": item["engine_key"]} for item in schedule]
        validate_complete_session_set(sessions, schedule)
        with self.assertRaises(ValueError):
            validate_complete_session_set(sessions[:-1], schedule)
        duplicate = sessions[:-1] + [dict(sessions[0])]
        with self.assertRaises(ValueError):
            validate_complete_session_set(duplicate, schedule)

    def test_parent_cpu_mock_completes_all_39_sessions_without_environment_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "latency-output"
            expected_gpu = {"uuid": "GPU-test", "name": "Quadro RTX 8000", "driver_version": "595.71.05"}
            expected_env = {
                "torch": "2.5.1+cu121", "ultralytics": "8.4.102", "tensorrt": "10.16.1.11",
                "numpy": "2.4.4", "cuda": "12.1", "gpu": "Quadro RTX 8000", "pycocotools": "2.0.10",
            }
            state = producer_snapshot_fixture()
            guard = state["process_guard"]
            pool = {"measured_sequence_sha256": "pool-sequence", "pool_size": 256,
                    "selected_files": [{"image": "a.jpg"}] * 256}
            specs = {}
            for index, key in enumerate(ENGINE_KEYS):
                specs[key] = {"engine_key": key, "model": latency.ENGINE_MODELS[key],
                              "arm": latency.ENGINE_MODELS[key], "build_repeat": latency.ENGINE_REPEATS[key],
                              "engine_path": str(root / f"{key}.engine"),
                              "engine_sha256": f"{index:064x}", "engine_bytes": 1}
            accuracy = {"models": {model: {"contrast_ci_refs": []} for model in ("fp16", *latency.ARM_NAMES)}}
            context = {
                "source_manifest": {"execution_git_commit": "accepted-source-code"},
                "source_manifest_path": root / "source.json", "source_manifest_sha256": "source-manifest",
                "source_artifact_commit": latency.PINNED_ATTEMPT2_COMMIT,
                "expected_environment": expected_env, "expected_gpu": expected_gpu,
                "specs": specs, "pool": pool, "accuracy": accuracy,
                "images_binding": {"declared": None, "resolved": str(root / "images"), "resolution": "test"},
            }
            events = []
            process_calls = []

            class NoLock:
                def __init__(self, *_args):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *_args):
                    return False

            def fake_child_command(_repo, spec, round_id, session_dir, run_manifest, pool_manifest, images_dir, device, confirmations):
                return ["mock-child", "--engine-key", spec["engine_key"], "--round", str(round_id),
                        "--session-dir", str(session_dir), "--run-manifest", str(run_manifest),
                        "--pool-manifest", str(pool_manifest), "--device", device]

            def fake_run(command, **_kwargs):
                def option(name):
                    return command[command.index(name) + 1]
                key = option("--engine-key")
                round_id = int(option("--round"))
                session_dir = Path(option("--session-dir"))
                pool_path = Path(option("--pool-manifest"))
                spec = specs[key]
                events.append(("start", round_id, key))
                binding = {"expected": expected_gpu, "actual": expected_gpu, "matched": True}
                session_state = {**state, "process_guard": guard}
                raw = [1.0] * MEASURED_CALLS
                session = {
                    "schema_version": 1, "study": latency.STUDY, "status": "completed",
                    "round": round_id, "engine_key": key, "engine_sha256": spec["engine_sha256"],
                    "engine_path": spec["engine_path"], "engine_bytes": 1, "dataset_split": latency.DATASET_SPLIT,
                    "runtime": {**RUNTIME_OPTIONS, "device": "0"},
                    "image_pool": {"manifest_path": str(pool_path.resolve()), "manifest_sha256": sha256_file(pool_path),
                                   "pool_size": 256, "pool_seed": latency.SEED, "pool_sequence_sha256": pool["measured_sequence_sha256"],
                                   "measured_sequence_start_index": 0, "measured_sequence_calls": MEASURED_CALLS,
                                   "decoded_shapes": [[720, 1280]], "distinct_decoded_shapes": [[720, 1280]],
                                   "distinct_source_aspect_ratios": [1.77777778], "same_image_bytes_checked": True},
                    "warmup_calls": WARMUP_CALLS, "measured_calls": MEASURED_CALLS,
                    "warmup_in_timing": False, "disk_decode_in_timing": False, "model_load_in_timing": False,
                    "initial_allocation_in_timing": False,
                    "preprocessed_input_shape_observation": {"expected_shape": [1, 3, 640, 640],
                        "by_source_aspect_ratio": {"1.77777778": {"image_index": 0, "shape": [1, 3, 640, 640]}},
                        "timing_excluded": True, "probe_calls": 2},
                    "synchronize_before_timer": True, "synchronize_after_predict": True,
                    "latency_ms": latency_statistics(raw), "raw_latency_ms": raw,
                    "environment": expected_env, "gpu_before": session_state, "gpu_after": session_state,
                    "gpu_identity_binding": {"before": binding, "after": binding},
                    "process_guard_evidence": {"before": guard, "after": guard},
                    "session_output_dir": str(session_dir.resolve()),
                }
                write_json(session_dir / "session.json", session)
                events.append(("finish", round_id, key))
                process_calls.append(command)
                return SimpleNamespace(returncode=0, stdout="mock stdout", stderr="")

            def fake_snapshot(_confirmations):
                return state

            args = SimpleNamespace(
                out_dir=output, device="0", confirm_desktop_process=[],
                attempt2_root=Path("attempt2"), fp16_capture=Path("fp16"),
                fp16_verification=Path("verification"), accuracy_root=Path("accuracy"), images_dir=None,
            )
            result = latency._run_parent(args, {
                "repo": root, "validate_output_target": lambda _repo, path: path,
                "validate_inputs": lambda *_args: context, "lock_class": NoLock,
                "snapshot": fake_snapshot, "ensure_idle": lambda _state: None,
                "child_command": fake_child_command, "run_process": fake_run,
                "current_commit": "test-code-commit",
            })
            self.assertEqual(result, 0)
            self.assertTrue((output / "latency_summary.json").is_file())
            self.assertTrue((output / "report.md").is_file())
            summary = json.loads((output / "latency_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["sessions_completed"], 39)
            self.assertEqual(summary["arm_summary"]["fp16"]["pooled_calls"]["n_calls"], 3000)
            self.assertEqual(summary["arm_summary"]["baseline_int8"]["pooled_calls"]["n_calls"], 9000)
            self.assertEqual(len(process_calls), 39)
            self.assertEqual([event[0] for event in events], [phase for _ in range(39) for phase in ("start", "finish")])
            self.assertEqual(json.loads((output / "study_manifest.json").read_text(encoding="utf-8")).get("study"), latency.STUDY)

    def test_child_mock_uses_locked_runtime_and_records_shape_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            engine = root / "engine.engine"
            engine.write_bytes(b"x")
            engine_hash = sha256_file(engine)
            session_dir = root / "output/round_1/fp16"
            pool_path = root / "output/image_pool_manifest.json"
            pool_path.parent.mkdir(parents=True)
            pool = {"selected_files": [{"image": "a.jpg"}] * 256, "seed": latency.SEED,
                    "measured_sequence": ["a.jpg"] * MEASURED_CALLS,
                    "warmup_sequence": ["a.jpg"] * WARMUP_CALLS,
                    "measured_sequence_sha256": "pool-sequence", "warmup_sequence_sha256": "warmup-sequence"}
            write_json(pool_path, pool)
            expected_gpu = {"uuid": "GPU-test", "name": "Quadro RTX 8000", "driver_version": "595.71.05"}
            env = {"torch": "2.5.1+cu121", "ultralytics": "8.4.102", "tensorrt": "10.16.1.11",
                   "numpy": "2.4.4", "cuda": "12.1", "gpu": "Quadro RTX 8000", "pycocotools": "2.0.10"}
            state = producer_snapshot_fixture()
            guard = state["process_guard"]
            schedule = latency.engine_schedule()
            manifest_path = root / "output/study_manifest.json"
            write_json(manifest_path, {"study": latency.STUDY, "environment": env, "gpu_identity": expected_gpu,
                                       "output_dir": str(root / "output"), "images": {"resolved": str(root / "images")},
                                       "image_pool_manifest": {"path": str(pool_path), "sha256": sha256_file(pool_path)},
                                       "engine_inventory": {"fp16": {"engine_path": str(engine.resolve()), "engine_sha256": engine_hash, "engine_bytes": 1}},
                                       "round_schedule": schedule})

            class FakeCuda:
                @staticmethod
                def is_available():
                    return False

            class FakeTorch:
                cuda = FakeCuda()

            class FakeModel:
                def predict(self, **kwargs):
                    calls.append(kwargs)
                    return []

            calls = []
            observed = []

            def fake_loader(_pool_path, _images_path):
                return pool, [object()] * 256, [[720, 1280]] * 256

            def fake_observe(model, images, shapes, runtime):
                observed.append((model, len(images), shapes, runtime))
                return {"expected_shape": [1, 3, 640, 640],
                        "by_source_aspect_ratio": {"1.77777778": {"image_index": 0, "shape": [1, 3, 640, 640]}},
                        "timing_excluded": True, "probe_calls": 1}

            args = SimpleNamespace(engine_key="fp16", engine=engine, engine_sha256=engine_hash, round=1,
                                   session_dir=session_dir, run_manifest=manifest_path, pool_manifest=pool_path,
                                   images_dir=root / "images", device="0", confirm_desktop_process=[])
            with patch.object(latency, "_parse_confirmations", return_value={}):
                result = latency.run_child(args, {
                    "validate_engine_file": lambda path, digest: digest,
                    "snapshot": lambda _confirmations: state, "ensure_idle": lambda _state: None,
                    "runtime_environment": lambda: env, "load_pool_images": fake_loader,
                    "torch": FakeTorch(), "YOLO": lambda _path: FakeModel(), "observe_shapes": fake_observe,
                })
            self.assertEqual(result, 0)
            self.assertTrue((session_dir / "session.json").is_file())
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0][3], {**RUNTIME_OPTIONS, "device": "0"})
            session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(session["preprocessed_input_shape_observation"]["expected_shape"], [1, 3, 640, 640])
            self.assertEqual(session["image_pool"]["pool_size"], 256)
            spec = {"engine_path": str(engine.resolve()), "engine_sha256": engine_hash, "engine_bytes": 1}
            item = {"round": 1, "engine_key": "fp16"}
            latency._validate_session(session, item, spec, pool, env, expected_gpu,
                                      pool_path, session_dir, sha256_file(pool_path))
            short = copy.deepcopy(session)
            short["raw_latency_ms"] = [1.0]
            with self.assertRaises(ValueError):
                latency._validate_session(short, item, spec, pool, env, expected_gpu,
                                          pool_path, session_dir, sha256_file(pool_path))
            missing_guard = copy.deepcopy(session)
            del missing_guard["gpu_after"]["process_guard"]
            with self.assertRaises(ValueError):
                latency._validate_session(missing_guard, item, spec, pool, env, expected_gpu,
                                          pool_path, session_dir, sha256_file(pool_path))

    def test_parent_failed_child_keeps_partial_output_without_completed_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "latency-output"
            expected_gpu = {"uuid": "GPU-test", "name": "Quadro RTX 8000", "driver_version": "595.71.05"}
            state = {"device": "GPU-test, Quadro RTX 8000, 595.71.05, P8, 30, 9 W, 300 MHz, 405 MHz, 29 MiB"}
            specs = {key: {"engine_key": key, "model": latency.ENGINE_MODELS[key],
                           "engine_path": str(root / f"{key}.engine"), "engine_sha256": f"{index:064x}"}
                     for index, key in enumerate(ENGINE_KEYS)}
            context = {
                "source_manifest": {}, "source_manifest_path": root / "source.json", "source_manifest_sha256": "source",
                "source_artifact_commit": latency.PINNED_ATTEMPT2_COMMIT,
                "expected_environment": latency.EXPECTED_ENV, "expected_gpu": expected_gpu,
                "specs": specs, "pool": {"selected_files": [], "measured_sequence_sha256": "pool"},
                "accuracy": {"models": {}}, "images_binding": {"resolved": str(root / "images")},
            }

            class NoLock:
                def __init__(self, *_args):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *_args):
                    return False

            def fake_command(_repo, spec, round_id, session_dir, *_args):
                return ["mock-child", "--engine-key", spec["engine_key"], "--round", str(round_id),
                        "--session-dir", str(session_dir)]

            def failed_run(_command, **_kwargs):
                return SimpleNamespace(returncode=7, stdout="partial", stderr="failed child")

            args = SimpleNamespace(
                out_dir=output, device="0", confirm_desktop_process=[], attempt2_root=Path("attempt2"),
                fp16_capture=Path("fp16"), fp16_verification=Path("verification"), accuracy_root=Path("accuracy"),
                images_dir=None,
            )
            with self.assertRaises(RuntimeError):
                latency._run_parent(args, {
                    "repo": root, "validate_output_target": lambda _repo, path: path,
                    "validate_inputs": lambda *_args: context, "lock_class": NoLock,
                    "snapshot": lambda _confirmations: state, "ensure_idle": lambda _state: None,
                    "child_command": fake_command, "run_process": failed_run, "current_commit": "test-code",
                })
            self.assertTrue((output / "study_manifest.json").is_file())
            self.assertTrue((output / "round_1/fp16/failure.json").is_file())
            self.assertFalse((output / "latency_summary.json").exists())
            with self.assertRaises(FileExistsError):
                latency._run_parent(args, {
                    "repo": root, "validate_output_target": lambda _repo, path: path,
                    "validate_inputs": lambda *_args: context, "lock_class": NoLock,
                })

    def test_session_validator_rejects_short_samples_and_missing_telemetry(self):
        with self.assertRaises(ValueError):
            latency._validate_session(
                {"status": "completed", "study": latency.STUDY, "round": 1, "engine_key": "fp16",
                 "engine_sha256": "h", "warmup_calls": WARMUP_CALLS, "measured_calls": MEASURED_CALLS,
                 "runtime": {**RUNTIME_OPTIONS, "device": "0"}, "image_pool": {"pool_sequence_sha256": "p",
                 "measured_sequence_calls": MEASURED_CALLS, "pool_size": 256}, "warmup_in_timing": False,
                 "disk_decode_in_timing": False, "synchronize_before_timer": True,
                 "synchronize_after_predict": True, "environment": {}, "raw_latency_ms": [1.0]},
                {"round": 1, "engine_key": "fp16"}, {"engine_sha256": "h", "engine_path": "x"},
                {"measured_sequence_sha256": "p"}, latency.EXPECTED_ENV,
            )

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
