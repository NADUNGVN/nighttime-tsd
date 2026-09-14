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
GUARD = {"telemetry_status": "complete", "external_workload_detected": False,
         "blocked_processes": [], "unmatched_confirmations": []}
GPU_WITH_GUARD = {"device": GPU["device"], "process_guard": GUARD}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, allow_nan=False)
        handle.write("\n")


def source_fixture(repo):
    source, _ = replay.resolve_protocol_paths(repo)
    frozen_source = Path(__file__).resolve().parents[1] / replay.FROZEN_WEIGHTS_PATH
    frozen_target = repo / replay.FROZEN_WEIGHTS_PATH
    frozen_target.parent.mkdir(parents=True, exist_ok=True)
    frozen_target.write_bytes(frozen_source.read_bytes())
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
    sigmoid_constraints = ["sigmoid"]
    write_json(source / "study_manifest.json", {
        "study": replay.SOURCE_STUDY, "git_commit": replay.SOURCE_CODE_COMMIT,
        "source_weights_sha256": replay.FROZEN_WEIGHTS_SHA256,
        "onnx_sha256": sha256(source / "source.onnx"), "settings": settings,
        "environment": ENVIRONMENT, "gpu_before": GPU, "engine_metadata": {"task": "detect"},
    })
    write_json(source / "repeat_summary.json", {"build_variability_coco_xml": {}})
    for repeat in replay.ENGINE_REPEATS:
        if repeat != 1:
            (source / f"repeat_{repeat}").mkdir()
        write_json(source / f"repeat_{repeat}/build_manifest.json", {
            "study": replay.SOURCE_STUDY, "repeat": repeat,
            "source_weights_sha256": replay.FROZEN_WEIGHTS_SHA256,
            "onnx_sha256": sha256(source / "source.onnx"), "settings": settings,
            "calibration_cache_sha256": sha256(source / "repeat_1/calibration.cache"),
            "builder_flags": 514,
            "sigmoid_fp32_constraints": sigmoid_constraints,
        })
    return source


def replay_study_fixture(repo, source):
    output = replay.resolve_protocol_paths(repo)[1]
    output.mkdir(parents=True)
    write_json(output / "study_manifest.json", {
        "study": replay.STUDY,
        "source_study": replay.SOURCE_STUDY,
        "source_result_commit": replay.SOURCE_RESULT_COMMIT,
        "source_study_code_commit": replay.SOURCE_CODE_COMMIT,
        "source_study_manifest_sha256": sha256(source / "study_manifest.json"),
        "source_weights_sha256": replay.FROZEN_WEIGHTS_SHA256,
        "frozen_weights_path": replay.FROZEN_WEIGHTS_PATH.as_posix(),
        "frozen_weights_measured_sha256": sha256(repo / replay.FROZEN_WEIGHTS_PATH),
        "onnx_sha256": sha256(source / "source.onnx"),
        "settings": copy.deepcopy(replay.REPLAY_SETTINGS),
        "protocol_variant": replay.REPLAY_VARIANT,
        "builder_flags": 514,
        "sigmoid_fp32_constraints": ["sigmoid"],
        "calibration_cache_input": {"source_repeat": 1, "sha256": sha256(source / "repeat_1/calibration.cache"), "copy_per_build": True},
        "timing_cache_input": {"source_repeat": 1, "sha256": sha256(source / "repeat_1/timing.cache"), "copy_per_build": True, "ignore_mismatch": False},
    })
    for repeat in replay.ENGINE_REPEATS:
        dest = output / f"repeat_{repeat}"
        dest.mkdir()
        engine = dest / "model.engine"
        engine.write_bytes(f"engine-{repeat}".encode())
        (dest / "calibration_cache_input.cache").write_bytes((source / "repeat_1/calibration.cache").read_bytes())
        (dest / "timing_cache_input.cache").write_bytes((source / "repeat_1/timing.cache").read_bytes())
        (dest / "timing_cache_output.cache").write_bytes(b"timing-output")
        write_json(dest / "inspector.json", {"Layers": []})
        write_json(dest / "build_manifest.json", {
            "study": replay.STUDY, "repeat": repeat, "source_study": replay.SOURCE_STUDY,
            "source_result_commit": replay.SOURCE_RESULT_COMMIT,
            "source_study_code_commit": replay.SOURCE_CODE_COMMIT,
            "source_study_manifest_sha256": sha256(source / "study_manifest.json"),
            "source_weights_sha256": replay.FROZEN_WEIGHTS_SHA256,
            "frozen_weights_path": replay.FROZEN_WEIGHTS_PATH.as_posix(),
            "frozen_weights_measured_sha256": sha256(repo / replay.FROZEN_WEIGHTS_PATH),
            "onnx_sha256": sha256(source / "source.onnx"),
            "calibration_cache_input_sha256": sha256(dest / "calibration_cache_input.cache"),
            "calibration_cache_read": True, "calibration_cache_read_calls": 2,
            "calibration_cache_read_violations": [], "calibration_batches_consumed": 0,
            "calibration_cache_write_calls": 0, "calibration_cache_write_violations": [],
            "calibration_cache_output_changed": None,
            "timing_cache_input_sha256": sha256(dest / "timing_cache_input.cache"),
            "timing_cache_output_sha256": sha256(dest / "timing_cache_output.cache"),
            "timing_cache_attach": {"called": True, "ignore_mismatch": False, "return_value": True},
            "timing_cache_coverage": replay.CACHE_COVERAGE_UNKNOWN,
            "timing_cache_output_changed": True,
            "settings": copy.deepcopy(replay.REPLAY_SETTINGS), "builder_flags": 514,
            "sigmoid_fp32_constraints": ["sigmoid"],
            "inspector_signature": "replay-inspector", "source_inspector_signature": "source-inspector",
            "engine_sha256": sha256(engine), "inspector_sha256": sha256(dest / "inspector.json"),
            "environment": {"tensorrt": "10.16.1.11"},
            "gpu_before": copy.deepcopy(GPU_WITH_GUARD), "gpu_after": copy.deepcopy(GPU_WITH_GUARD),
            "study_manifest_sha256": sha256(output / "study_manifest.json"),
            "termination_status": "completed",
        })
    return output


def evaluation_payload(confidence=0.5):
    records = []
    for index in range(1636):
        target_count = 2 if index < 1070 else 1
        records.append({
            "image": f"{index:04d}.jpg", "orig_shape": [10, 10], "xyxy": [[1, 1, 4, 4]],
            "confidence": [confidence], "class_id": [0],
            "validator_input": {
                "imgsz": 640, "ratio_pad": [[1.0, 1.0], [0.0, 0.0]],
                "target_xyxy": [[1, 1, 2, 2]] * target_count,
                "target_class_id": [0] * target_count,
            },
            "validator_statistics": {},
        })
    return {"schema_version": 2, "capture_mode": "same_val_process_batch",
            "iou_thresholds": [0.5], "coordinate_contract": "xyxy", "records": records,
            "confidence_marker": confidence}


def evaluation_metrics():
    return {"map50": 0.9, "map50_95": 0.8, "precision": 0.7, "recall": 0.6}


def evaluation_size():
    return {"metric_id": "test-size", "rules": "test-rules",
            "evaluator_source_sha256": "evaluator", "xml_sha256": "xml",
            "metrics": {label: {"instances": 2706 if label == "all" else 1,
                                  "map50": 0.9, "map50_95": 0.8}
                        for label in ("all", "xs", "s", "m", "l", "xl")}}


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

    def test_source_contract_hashes_frozen_weights_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = source_fixture(repo)
            (repo / replay.FROZEN_WEIGHTS_PATH).write_bytes(b"changed-weights")
            with patch.object(replay, "SOURCE_ONNX_SHA256", sha256(source / "source.onnx")), \
                 patch.object(replay, "SOURCE_CALIBRATION_CACHE_SHA256", sha256(source / "repeat_1/calibration.cache")), \
                 patch.object(replay, "SOURCE_TIMING_CACHE_SHA256", sha256(source / "repeat_1/timing.cache")):
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
        exact_records = []
        for repeat in replay.ENGINE_REPEATS:
            record = copy.deepcopy(base)
            record["repeat"] = repeat
            exact_records.append(record)
        self.assertEqual(replay.classify_replay(exact_records),
                         "replay_exact_observed")
        changed = copy.deepcopy(exact_records[1])
        changed["prediction_payload_sha256"] = "b"
        self.assertEqual(replay.classify_replay([exact_records[0], changed, exact_records[2]]),
                         "replay_variation_observed")
        self.assertEqual(replay.classify_replay([exact_records[0]]), "incomplete_or_invalid")
        self.assertEqual(replay.classify_replay([exact_records[0]], ["missing capture"]), "incomplete_or_invalid")

    def test_comparison_separates_capture_report_and_prediction_payload(self):
        def payload(confidence):
            return {"schema_version": 2, "capture_mode": "same_val_process_batch",
                    "iou_thresholds": [0.5], "coordinate_contract": "xyxy",
                    "records": [{"image": "one.jpg", "orig_shape": [10, 10],
                                 "xyxy": [[1, 1, 4, 4]], "confidence": [confidence], "class_id": [0],
                                 "validator_input": {"imgsz": 640, "ratio_pad": [[1, 1], [0, 0]],
                                                      "target_xyxy": [], "target_class_id": []},
                                 "validator_statistics": {}}]}

        report = {"metrics": {"map50": 0.9, "map50_95": 0.8, "precision": 0.7, "recall": 0.6}}
        size = {"metrics": {"all": {"map50": 0.9, "map50_95": 0.8}}}
        row = {"predictions": payload(0.5), "capture": report, "size": size}
        exact = replay._build_comparison_row(row, None, report, payload(0.5), size)
        changed = replay._build_comparison_row(row, None, report, payload(0.4), size)
        self.assertTrue(exact["prediction_payload_exact_vs_step_a_repeat_1"])
        self.assertFalse(changed["prediction_payload_exact_vs_step_a_repeat_1"])
        self.assertTrue(exact["metrics_exact_vs_step_a_repeat_1"])

    def test_calibration_callback_contract_and_timing_attach_branches(self):
        audit = replay.CalibrationCacheAudit(b"cache")
        self.assertEqual(audit.read(), b"cache")
        self.assertEqual(audit.read(), b"cache")
        audit.validate()
        wrong_read = replay.CalibrationCacheAudit(b"cache")
        wrong_read.read(b"wrong")
        wrong_read.read(b"cache")
        with self.assertRaisesRegex(RuntimeError, "read bytes differed"):
            wrong_read.validate()

        with self.assertRaisesRegex(RuntimeError, "not read"):
            replay.CalibrationCacheAudit(b"cache").validate()
        batch_audit = replay.CalibrationCacheAudit(b"cache")
        batch_audit.read()
        with self.assertRaisesRegex(RuntimeError, "recalibration"):
            batch_audit.record_batch()
        with self.assertRaisesRegex(RuntimeError, "consumed"):
            batch_audit.validate()

        write_audit = replay.CalibrationCacheAudit(b"cache")
        write_audit.read()
        write_audit.write(b"wrong")
        write_audit.write(b"cache")
        with self.assertRaisesRegex(RuntimeError, "write bytes differed"):
            write_audit.validate()

        class Config:
            def __init__(self, result):
                self.result = result
                self.created = []
                self.attached = []

            def create_timing_cache(self, data):
                self.created.append(data)
                return {"bytes": data}

            def set_timing_cache(self, cache, ignore_mismatch):
                self.attached.append((cache, ignore_mismatch))
                return self.result

        config = Config(True)
        self.assertEqual(replay.attach_timing_cache(config, b"timing"),
                         {"called": True, "ignore_mismatch": False, "return_value": True})
        self.assertEqual(config.created, [b"timing"])
        self.assertEqual(len(config.attached), 1)
        with self.assertRaisesRegex(RuntimeError, "ignore_mismatch=False"):
            replay.attach_timing_cache(Config(False), b"timing")

    def test_build_phase_requires_prepared_study_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            output = repo / "results/measurement_audit_v1/server_uniform_timing_cache_replay_v1"
            with self.assertRaises(SystemExit):
                replay.main(["--phase", "build", "--repeat", "1", "--out-dir", str(output)],
                            repo_override=repo)

    def test_replay_capture_inputs_rejects_arbitrary_study_and_binds_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = source_fixture(repo)
            output = replay_study_fixture(repo, source)
            engine = output / "repeat_1/model.engine"
            write_json(source / "repeat_1/capture/capture_report.json", {"status": "pass"})
            with patch.object(replay, "SOURCE_ONNX_SHA256", sha256(source / "source.onnx")), \
                 patch.object(replay, "SOURCE_CALIBRATION_CACHE_SHA256", sha256(source / "repeat_1/calibration.cache")), \
                 patch.object(replay, "SOURCE_TIMING_CACHE_SHA256", sha256(source / "repeat_1/timing.cache")):
                result = replay.replay_capture_inputs(repo, output, 1)
                self.assertEqual(result[0], engine)
                mutated = json.loads((output / "study_manifest.json").read_text(encoding="utf-8"))
                mutated["settings"]["builder_optimization_level"] = 2
                (output / "study_manifest.json").unlink()
                write_json(output / "study_manifest.json", mutated)
                with self.assertRaisesRegex(ValueError, "exactly"):
                    replay.replay_capture_inputs(repo, output / "other", 1)
                with self.assertRaisesRegex(ValueError, "settings"):
                    replay.replay_capture_inputs(repo, output, 1)

    def test_evaluate_cpu_mocked_children_runs_all_three_repeats_and_detects_payload_change(self):
        for changed_repeat, expected_status in ((None, "replay_exact_observed"),
                                                 (2, "replay_variation_observed")):
            with self.subTest(changed_repeat=changed_repeat):
                with tempfile.TemporaryDirectory() as directory:
                    repo = Path(directory)
                    source = source_fixture(repo)
                    output = replay_study_fixture(repo, source)
                    source_predictions = evaluation_payload()
                    source_capture = {
                        "dataset_split": replay.DATASET_SPLIT, "images": 1636, "instances": 2706,
                        "status": "pass", "runtime_arguments": copy.deepcopy(replay.EXPECTED_RUNTIME),
                        "metrics": evaluation_metrics(),
                    }
                    source_size = evaluation_size()
                    write_json(source / "repeat_1/capture/validator_predictions.json", source_predictions)
                    write_json(source / "repeat_1/capture/capture_report.json", source_capture)
                    write_json(source / "repeat_1/verification/size_coco_xml.json", source_size)
                    args = SimpleNamespace(device="0", confirmed_desktop={})
                    events = []

                    def run_child(command, _repo):
                        events.append(Path(command[1]).name if len(command) > 1 else "child")
                        if "verify_cctsdb_capture.py" in str(command[1]):
                            capture_dir = Path(command[command.index("--capture-dir") + 1])
                            verification_dir = Path(command[command.index("--out-dir") + 1])
                            capture_report = json.loads((capture_dir / "capture_report.json").read_text(encoding="utf-8"))
                            predictions = capture_dir / "validator_predictions.json"
                            write_json(verification_dir / "verification_summary.json", {
                                "dataset_split": replay.DATASET_SPLIT, "model_sha256": capture_report["model_sha256"],
                                "native_matching_status": "pass", "size_diagnostic": "completed",
                                "capture_hash_match": "exact_bytes",
                                "capture_prediction_sha256": sha256(predictions),
                                "capture_report_sha256": sha256(capture_dir / "capture_report.json"),
                            })
                            write_json(verification_dir / "size_coco_xml.json", source_size)
                            return 22

                        run_dir = Path(command[command.index("--out-dir") + 1])
                        repeat = int(command[command.index("--repeat-index") + 1])
                        run_dir.mkdir(parents=True, exist_ok=True)
                        predictions = copy.deepcopy(source_predictions)
                        if repeat == changed_repeat:
                            predictions["records"][0]["confidence"] = [0.4]
                        write_json(run_dir / "validator_predictions.json", predictions)
                        build_manifest = json.loads((output / f"repeat_{repeat}/build_manifest.json").read_text(encoding="utf-8"))
                        report = {
                            "dataset_split": replay.DATASET_SPLIT, "images": 1636, "instances": 2706,
                            "status": "pass", "runtime_arguments": copy.deepcopy(replay.EXPECTED_RUNTIME),
                            "metrics": evaluation_metrics(), "model_sha256": build_manifest["engine_sha256"],
                            "engine_provenance_sha256": sha256(output / f"repeat_{repeat}/build_manifest.json"),
                            "predictions_sha256": sha256(run_dir / "validator_predictions.json"),
                            "gpu_before": copy.deepcopy(GPU_WITH_GUARD), "gpu_after": copy.deepcopy(GPU_WITH_GUARD),
                        }
                        write_json(run_dir / "capture_report.json", report)
                        return 11

                    with patch.object(replay, "SOURCE_ONNX_SHA256", sha256(source / "source.onnx")), \
                         patch.object(replay, "SOURCE_CALIBRATION_CACHE_SHA256", sha256(source / "repeat_1/calibration.cache")), \
                         patch.object(replay, "SOURCE_TIMING_CACHE_SHA256", sha256(source / "repeat_1/timing.cache")):
                        result = replay.evaluate(repo, output, args,
                                                 replay.validate_source_contract(repo),
                                                 {"run_child": run_child})
                    self.assertEqual(result, 0)
                    summary = json.loads((output / "repeat_summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(summary["classification"], expected_status)
                    self.assertEqual(len(summary["records"]), 3)
                    self.assertEqual(events, ["capture_cctsdb_validator.py", "verify_cctsdb_capture.py"] * 3)

    def test_capture_dispatch_rejects_locked_study_and_build_mutations(self):
        mutations = (
            ("study", "settings", lambda value: {**value, "builder_optimization_level": 2}),
            ("study", "builder_flags", lambda _value: 0),
            ("study", "sigmoid_fp32_constraints", lambda _value: ["changed"]),
            ("study", "timing_cache_input", lambda value: {**value, "ignore_mismatch": True}),
            ("build", "termination_status", lambda _value: "failed"),
            ("build", "timing_cache_input_sha256", lambda _value: "changed"),
        )
        for location, key, mutate in mutations:
            with self.subTest(location=location, key=key):
                with tempfile.TemporaryDirectory() as directory:
                    repo = Path(directory)
                    source = source_fixture(repo)
                    output = replay_study_fixture(repo, source)
                    target = output / "study_manifest.json" if location == "study" else output / "repeat_1/build_manifest.json"
                    value = json.loads(target.read_text(encoding="utf-8"))
                    value[key] = mutate(value[key])
                    target.unlink()
                    write_json(target, value)
                    write_json(source / "repeat_1/capture/capture_report.json", {"status": "pass"})
                    with patch.object(replay, "SOURCE_ONNX_SHA256", sha256(source / "source.onnx")), \
                         patch.object(replay, "SOURCE_CALIBRATION_CACHE_SHA256", sha256(source / "repeat_1/calibration.cache")), \
                         patch.object(replay, "SOURCE_TIMING_CACHE_SHA256", sha256(source / "repeat_1/timing.cache")):
                        with self.assertRaises(ValueError):
                            replay.replay_capture_inputs(repo, output, 1)

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
