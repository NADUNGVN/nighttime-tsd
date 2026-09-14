import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_uniform_timing_cache_replay as replay
from audit_cctsdb_measurement import sha256


ENVIRONMENT = {
    "torch": "2.5.1+cu121", "ultralytics": "8.4.102", "tensorrt": "10.16.1.11",
    "numpy": "2.4.4", "python": "test-python", "cuda": "12.1",
    "gpu": "Quadro RTX 8000", "pycocotools": "2.0.10",
}
GPU = {"device": "GPU-test, Quadro RTX 8000, 595.71.05, P8, 35, 9 W, 300 MHz, 405 MHz, 32 MiB"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, allow_nan=False)
        handle.write("\n")


def source_fixture(repo):
    source, _ = replay.resolve_protocol_paths(repo)
    source.mkdir(parents=True)
    (source / "source.onnx").write_bytes(b"test-onnx")
    (source / "repeat_1").mkdir()
    (source / "repeat_1/calibration.cache").write_bytes(b"test-calibration")
    (source / "repeat_1/timing.cache").write_bytes(b"test-timing")
    (source / "repeat_1/inspector.json").write_text(json.dumps({"Layers": [{
        "Name": "sigmoid", "LayerType": "Activation", "Inputs": [], "Outputs": [],
        "Weights": {}, "TacticName": "t", "ParameterType": "Activation",
    }]}), encoding="utf-8")
    settings = copy.deepcopy(replay.SOURCE_SETTINGS)
    write_json(source / "study_manifest.json", {
        "study": replay.SOURCE_STUDY, "git_commit": replay.SOURCE_CODE_COMMIT,
        "source_weights_sha256": replay.FROZEN_WEIGHTS_SHA256,
        "onnx_sha256": sha256(source / "source.onnx"), "settings": settings,
        "environment": ENVIRONMENT, "gpu_before": GPU, "engine_metadata": {"task": "detect"},
    })
    write_json(source / "repeat_summary.json", {"build_variability_coco_xml": {}})
    constraints = ["sigmoid"]
    for repeat in replay.ENGINE_REPEATS:
        if repeat != 1:
            (source / f"repeat_{repeat}").mkdir()
        write_json(source / f"repeat_{repeat}/build_manifest.json", {
            "study": replay.SOURCE_STUDY, "repeat": repeat,
            "source_weights_sha256": replay.FROZEN_WEIGHTS_SHA256,
            "onnx_sha256": sha256(source / "source.onnx"), "settings": settings,
            "calibration_cache_sha256": sha256(source / "repeat_1/calibration.cache"),
            "sigmoid_fp32_constraints": constraints,
        })
    return source


class UniformTimingCacheReplayTests(unittest.TestCase):
    def test_protocol_path_is_exact_and_separate_from_step_a(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source, output = replay.resolve_protocol_paths(repo)
            self.assertEqual(source.name, "server_uniform_build_repeat_v1")
            self.assertEqual(output.name, "server_uniform_timing_cache_replay_v1")
            self.assertEqual(replay.validate_output_target(repo, output), (source, output))
            with self.assertRaisesRegex(ValueError, "server_uniform_timing_cache_replay_v1"):
                replay.validate_output_target(repo, repo / "results/measurement_audit_v1/timing-cache")

    def test_source_contract_locks_onnx_calibration_and_timing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = source_fixture(repo)
            with patch.object(replay, "SOURCE_ONNX_SHA256", sha256(source / "source.onnx")), \
                 patch.object(replay, "SOURCE_CALIBRATION_CACHE_SHA256", sha256(source / "repeat_1/calibration.cache")), \
                 patch.object(replay, "SOURCE_TIMING_CACHE_SHA256", sha256(source / "repeat_1/timing.cache")):
                contract = replay.validate_source_contract(repo)
            self.assertEqual(contract["timing_cache"], source / "repeat_1/timing.cache")
            self.assertEqual(contract["source_result_commit"], replay.SOURCE_RESULT_COMMIT)

    def test_source_contract_rejects_changed_timing_input(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = source_fixture(repo)
            (source / "repeat_1/timing.cache").write_bytes(b"changed")
            with patch.object(replay, "SOURCE_ONNX_SHA256", sha256(source / "source.onnx")), \
                 patch.object(replay, "SOURCE_CALIBRATION_CACHE_SHA256", sha256(source / "repeat_1/calibration.cache")), \
                 patch.object(replay, "SOURCE_TIMING_CACHE_SHA256", "locked-timing"):
                with self.assertRaisesRegex(ValueError, "input hash mismatch"):
                    replay.validate_source_contract(repo)

    def test_capture_dispatch_is_scoped_and_does_not_use_step_a_flag(self):
        command = replay.timing_capture_command(
            Path("/repo"), Path("/repo/results/measurement_audit_v1/server_uniform_timing_cache_replay_v1"),
            2, Path("/repo/results/measurement_audit_v1/server_uniform_timing_cache_replay_v1/repeat_2"),
            "0", {445: "/usr/bin/Xorg"})
        self.assertIn("--timing-cache-study", command)
        self.assertIn("--repeat-index", command)
        self.assertNotIn("--repeat-study", command)
        self.assertNotIn("--representation", command)

    def test_replay_classification_is_locked_to_exact_payload_and_metrics(self):
        base = {"prediction_payload_sha256": "a", "capture": {"metrics": {"map50": 1}},
                "size": {"metrics": {"all": {"map50": 1}}}}
        self.assertEqual(replay.classify_replay([base, copy.deepcopy(base), copy.deepcopy(base)]),
                         "replay_exact_observed")
        changed = copy.deepcopy(base)
        changed["prediction_payload_sha256"] = "b"
        self.assertEqual(replay.classify_replay([base, changed, copy.deepcopy(base)]),
                         "replay_variation_observed")
        self.assertEqual(replay.classify_replay([base], ["missing capture"]), "incomplete_or_invalid")

    def test_replay_capture_inputs_rejects_arbitrary_study_and_binds_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = source_fixture(repo)
            output = replay.resolve_protocol_paths(repo)[1]
            output.mkdir(parents=True)
            write_json(output / "study_manifest.json", {"study": replay.STUDY})
            engine = output / "repeat_1/model.engine"
            engine.parent.mkdir(parents=True)
            engine.write_bytes(b"engine")
            write_json(output / "repeat_1/build_manifest.json", {
                "study": replay.STUDY, "repeat": 1, "onnx_sha256": sha256(source / "source.onnx"),
                "calibration_cache_input_sha256": sha256(source / "repeat_1/calibration.cache"),
                "engine_sha256": sha256(engine), "environment": {"tensorrt": "10.16.1.11"},
            })
            write_json(source / "repeat_1/capture/capture_report.json", {"status": "pass"})
            with patch.object(replay, "SOURCE_ONNX_SHA256", sha256(source / "source.onnx")), \
                 patch.object(replay, "SOURCE_CALIBRATION_CACHE_SHA256", sha256(source / "repeat_1/calibration.cache")), \
                 patch.object(replay, "SOURCE_TIMING_CACHE_SHA256", sha256(source / "repeat_1/timing.cache")):
                result = replay.replay_capture_inputs(repo, output, 1)
                self.assertEqual(result[0], engine)
                with self.assertRaisesRegex(ValueError, "exactly"):
                    replay.replay_capture_inputs(repo, output / "other", 1)

    def test_parent_build_children_are_sequential_and_before_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = source_fixture(repo)
            output = replay.resolve_protocol_paths(repo)[1]
            events = []
            snapshot = {"device": GPU["device"], "process_guard": {
                "telemetry_status": "complete", "external_workload_detected": False,
                "blocked_processes": [], "unmatched_confirmations": [],
            }}
            helpers = {
                "environment_preflight": lambda _repo: {"status": "ok", "environment": ENVIRONMENT},
                "parse_desktop_confirmations": lambda _values: {},
                "snapshot": lambda _confirmations: snapshot,
                "ensure_idle": lambda _state: None,
            }
            with patch.object(replay, "SOURCE_ONNX_SHA256", sha256(source / "source.onnx")), \
                 patch.object(replay, "SOURCE_CALIBRATION_CACHE_SHA256", sha256(source / "repeat_1/calibration.cache")), \
                 patch.object(replay, "SOURCE_TIMING_CACHE_SHA256", sha256(source / "repeat_1/timing.cache")), \
                 patch.object(replay, "git_value", return_value="test-git"), \
                 patch.object(replay, "evaluate", side_effect=lambda *_args: events.append("evaluate") or 0), \
                 patch.object(replay.subprocess, "run", side_effect=lambda command, **_kwargs: events.append(command) or SimpleNamespace(returncode=0)):
                result = replay.main(["--out-dir", str(output)], repo_override=repo, helpers=helpers)
            self.assertEqual(result, 0)
            build_commands = [event for event in events if isinstance(event, list) and "--phase" in event]
            self.assertEqual([event[event.index("--repeat") + 1] for event in build_commands], ["1", "2", "3"])
            self.assertEqual(events[-1], "evaluate")

    def test_settings_do_not_add_tactic_or_precision_intervention(self):
        self.assertEqual(replay.REPLAY_SETTINGS["int8"], replay.SOURCE_SETTINGS["int8"])
        self.assertFalse(replay.REPLAY_SETTINGS["fp16"])
        self.assertFalse(replay.REPLAY_SETTINGS["tf32"])
        self.assertEqual(replay.REPLAY_SETTINGS["builder_optimization_level"], 3)
        self.assertEqual(replay.REPLAY_SETTINGS["timing_cache"], "source_repeat_1_replay_per_process")


if __name__ == "__main__":
    unittest.main()
