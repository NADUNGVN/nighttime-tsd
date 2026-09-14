import copy
from contextlib import ExitStack
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_yolo11n_precision_head_ablation as ablation
import run_uniform_timing_cache_replay as replay
from audit_cctsdb_measurement import sha256


GPU = {"device": "GPU-test, Quadro RTX 8000, 595.71.05, P8, 35, 9 W, 300 MHz, 405 MHz, 32 MiB",
       "process_guard": {"telemetry_status": "complete", "external_workload_detected": False,
                          "blocked_processes": [], "unmatched_confirmations": []}}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, allow_nan=False)
        handle.write("\n")


def minimal_payload():
    return {
        "schema_version": 2, "capture_mode": "same_val_process_batch",
        "iou_thresholds": [0.5], "coordinate_contract": "xyxy",
        "records": [{
            "image": "one.jpg", "orig_shape": [10, 10], "xyxy": [[1, 1, 4, 4]],
            "confidence": [0.5], "class_id": [0],
            "validator_input": {"imgsz": 640, "ratio_pad": [[1, 1], [0, 0]],
                                 "prediction_xyxy": [[1, 1, 4, 4]],
                                 "target_xyxy": [[1, 1, 4, 4]], "target_class_id": [0]},
            "validator_statistics": {"tp": [[True]]},
        }],
    }


def minimal_size():
    return {
        "metric_id": "test-size", "rules": "test-rules",
        "evaluator_source_sha256": "evaluator", "xml_sha256": "xml",
        "metrics": {label: {"instances": 2706 if label == "all" else 1,
                             "map50": 0.9, "map50_95": 0.8}
                    for label in ablation.SIZE_LABELS},
    }


def contract_fixture(repo):
    """Create only the JSON/cache boundary needed by source validation tests."""
    source, _ = ablation.resolve_protocol_paths(repo)
    source.mkdir(parents=True)
    weights = repo / replay.FROZEN_WEIGHTS_PATH
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"test-frozen-weights")
    (source / "source.onnx").write_bytes(b"test-onnx")
    (source / "repeat_1").mkdir()
    (source / "repeat_1/calibration.cache").write_bytes(b"test-calibration")
    (source / "repeat_1/timing.cache").write_bytes(b"test-timing")
    write_json(source / "repeat_summary.json", {"classification": "replay_exact_observed"})
    weights_hash = sha256(weights)
    onnx_hash = sha256(source / "source.onnx")
    calibration_hash = sha256(source / "repeat_1/calibration.cache")
    timing_hash = sha256(source / "repeat_1/timing.cache")
    write_json(source / "study_manifest.json", {
        "study": replay.SOURCE_STUDY, "git_commit": "code",
        "source_weights_sha256": weights_hash, "onnx_sha256": onnx_hash,
        "settings": copy.deepcopy(replay.SOURCE_SETTINGS), "environment": {"gpu": "test"},
        "gpu_before": GPU, "engine_metadata": {"task": "detect"},
    })
    for repeat in replay.ENGINE_REPEATS:
        (source / f"repeat_{repeat}").mkdir(exist_ok=True)
        write_json(source / f"repeat_{repeat}/build_manifest.json", {
            "study": replay.SOURCE_STUDY, "repeat": repeat,
            "source_weights_sha256": weights_hash, "onnx_sha256": onnx_hash,
            "settings": copy.deepcopy(replay.SOURCE_SETTINGS),
            "calibration_cache_sha256": calibration_hash, "builder_flags": ablation.EXPECTED_BUILDER_FLAGS,
            "sigmoid_fp32_constraints": ["/model.23/Sigmoid"],
        })

    baseline = repo / "results/measurement_audit_v1" / ablation.BASELINE_STUDY_DIR
    (baseline / "repeat_1/capture").mkdir(parents=True)
    (baseline / "repeat_1/verification").mkdir(parents=True)
    write_json(baseline / "study_manifest.json", {"study": ablation.BASELINE_STUDY})
    write_json(baseline / "repeat_summary.json", {"classification": "replay_exact_observed"})
    payload = minimal_payload()
    write_json(baseline / "repeat_1/capture/validator_predictions.json", payload)
    write_json(baseline / "repeat_1/capture/capture_report.json", {
        "status": "pass", "predictions_sha256": sha256(baseline / "repeat_1/capture/validator_predictions.json"),
        "metrics": {"map50": 0.9, "map50_95": 0.8, "precision": 0.7, "recall": 0.6},
        "dataset_split": "CCTSDB2021/dev", "images": 1636, "instances": 2706,
        "runtime_arguments": copy.deepcopy(replay.EXPECTED_RUNTIME),
    })
    write_json(baseline / "repeat_1/verification/size_coco_xml.json", minimal_size())
    write_json(baseline / "repeat_1/build_manifest.json", {
        "study": ablation.BASELINE_STUDY, "repeat": 1,
        "source_result_commit": "result", "source_study_code_commit": "code",
        "source_weights_sha256": weights_hash, "onnx_sha256": onnx_hash,
        "calibration_cache_input_sha256": calibration_hash, "timing_cache_input_sha256": timing_hash,
        "settings": copy.deepcopy(replay.REPLAY_SETTINGS), "builder_flags": ablation.EXPECTED_BUILDER_FLAGS,
        "timing_cache_attach": {"called": True, "ignore_mismatch": False, "return_value": True},
    })
    return source, weights


def locked_contract_constants(repo, source, weights):
    stack = ExitStack()
    hashes = {
        "SOURCE_ONNX_SHA256": sha256(source / "source.onnx"),
        "SOURCE_CALIBRATION_CACHE_SHA256": sha256(source / "repeat_1/calibration.cache"),
        "SOURCE_TIMING_CACHE_SHA256": sha256(source / "repeat_1/timing.cache"),
        "FROZEN_WEIGHTS_SHA256": sha256(weights),
    }
    for module in (replay, ablation):
        for name, value in hashes.items():
            stack.enter_context(patch.object(module, name, value))
        stack.enter_context(patch.object(module, "SOURCE_CODE_COMMIT", "code"))
        stack.enter_context(patch.object(module, "SOURCE_RESULT_COMMIT", "result"))
    return stack


class FakeTRT:
    float32 = "FP32"

    class LayerType:
        CONVOLUTION = "CONVOLUTION"
        ACTIVATION = "ACTIVATION"


class FakeLayer:
    def __init__(self, name, layer_type, outputs=1, precision="INT8"):
        self.name = name
        self.type = layer_type
        self.num_outputs = outputs
        self.precision = precision
        self._outputs = [precision] * outputs

    def get_output_type(self, index):
        return self._outputs[index]

    def set_output_type(self, index, value):
        self._outputs[index] = value


class FakeNetwork:
    def __init__(self):
        self._layers = [
            FakeLayer("/model.23/cv2.0.0.conv", FakeTRT.LayerType.CONVOLUTION, 2),
            FakeLayer("/model.23/cv3.0.0.conv", FakeTRT.LayerType.CONVOLUTION, 2),
            FakeLayer("/model.23/dfl.conv", FakeTRT.LayerType.CONVOLUTION, 1),
            FakeLayer("/model.23/Sigmoid", FakeTRT.LayerType.ACTIVATION, 1),
        ]
        self.num_layers = len(self._layers)

    def get_layer(self, index):
        return self._layers[index]


class PrecisionHeadAblationTests(unittest.TestCase):
    def test_plan_is_four_arm_major_groups_and_twelve_operations(self):
        plan = ablation.build_plan()
        self.assertEqual(len(plan), 12)
        self.assertEqual([item["sequence"] for item in plan], list(range(1, 13)))
        self.assertEqual([item["arm"] for item in plan],
                         [arm for arm in ablation.ARMS for _ in ablation.REPEATS])
        self.assertEqual([item["repeat"] for item in plan], list(ablation.REPEATS) * 4)
        self.assertEqual(plan, ablation.build_plan())

    def test_output_path_is_exact_and_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source, output = ablation.resolve_protocol_paths(repo)
            self.assertEqual(source.name, "server_uniform_build_repeat_v1")
            self.assertEqual(output.name, "server_yolo11n_precision_head_ablation_v1")
            self.assertEqual(ablation.validate_output_target(repo, output), (source, output))
            with self.assertRaisesRegex(ValueError, "server_yolo11n_precision_head_ablation_v1"):
                ablation.validate_output_target(repo, repo / "results/measurement_audit_v1/other")

    def test_exact_prefix_selection_rejects_non_convolution_and_missing_nodes(self):
        specs = [
            {"name": "/model.23/cv2.0.conv", "is_convolution": True},
            {"name": "/model.23/cv3.0.conv", "is_convolution": True},
            {"name": "/model.23/cv2.extra/activation", "is_convolution": False},
        ]
        self.assertEqual(ablation.select_precision_layers(specs, "baseline_int8"), [])
        with self.assertRaisesRegex(ValueError, "non-convolution"):
            ablation.select_precision_layers(specs, "bbox_fp32")
        self.assertEqual(ablation.select_precision_layers(specs[:2], "bbox_fp32"),
                         ["/model.23/cv2.0.conv"])
        with self.assertRaisesRegex(ValueError, "No parsed"):
            ablation.select_precision_layers(specs[:1], "classification_fp32")
        with self.assertRaisesRegex(ValueError, "differs"):
            ablation.validate_selected_precision_layers(specs[:2], "bbox_fp32",
                                                        ["/model.23/cv3.0.conv"])

    def test_constraints_set_all_outputs_fp32_and_preserve_baseline_sigmoid(self):
        for arm, expected in (("baseline_int8", []),
                              ("bbox_fp32", ["/model.23/cv2.0.0.conv"]),
                              ("classification_fp32", ["/model.23/cv3.0.0.conv"]),
                              ("both_fp32", ["/model.23/cv2.0.0.conv", "/model.23/cv3.0.0.conv"])):
            network = FakeNetwork()
            audit = ablation.apply_precision_constraints(network, FakeTRT(), arm,
                                                          ["/model.23/Sigmoid"])
            self.assertEqual(audit["matched_layer_names"], expected)
            self.assertTrue(audit["obey_precision_constraints"])
            self.assertTrue(all(value == {"precision_fp32": True, "all_outputs_fp32": True}
                                for value in audit["effective"].values()))
            sigmoid = network.get_layer(3)
            self.assertEqual(sigmoid.precision, "FP32")
            self.assertEqual(sigmoid._outputs, ["FP32"])
            for layer in network._layers[:2]:
                if layer.name in expected:
                    self.assertEqual(layer.precision, "FP32")
                    self.assertEqual(layer._outputs, ["FP32", "FP32"])
                else:
                    self.assertEqual(layer.precision, "INT8")

    def test_four_arm_layer_sets_must_be_repeat_stable_and_union_consistent(self):
        layer_sets = {
            "baseline_int8": [], "bbox_fp32": ["bbox"],
            "classification_fp32": ["class"], "both_fp32": ["bbox", "class"],
        }
        manifests = {
            arm: {repeat: {"constraint_audit": {"matched_layer_names": list(names)}}
                  for repeat in ablation.REPEATS}
            for arm, names in layer_sets.items()
        }
        ablation.validate_arm_layer_contract(manifests)
        manifests["both_fp32"][2]["constraint_audit"]["matched_layer_names"] = ["bbox"]
        with self.assertRaisesRegex(ValueError, "differs across repeats"):
            ablation.validate_arm_layer_contract(manifests)

    def test_constraint_manifest_rejects_missing_requested_or_effective_evidence(self):
        network = FakeNetwork()
        audit = ablation.apply_precision_constraints(network, FakeTRT(), "bbox_fp32",
                                                     ["/model.23/Sigmoid"])
        self.assertEqual(audit["matched_layer_types"], {"/model.23/cv2.0.0.conv": "CONVOLUTION"})
        ablation.validate_constraint_audit(audit, "bbox_fp32", ["/model.23/Sigmoid"])
        bad = copy.deepcopy(audit)
        bad["effective"] = {}
        with self.assertRaisesRegex(ValueError, "incomplete"):
            ablation.validate_constraint_audit(bad, "bbox_fp32", ["/model.23/Sigmoid"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            ablation.validate_manifest_layer_names("bbox_fp32", audit["matched_layer_names"], {})

    def test_cache_audit_and_timing_attach_have_no_fallback(self):
        audit = ablation.CalibrationCacheAudit(b"cache")
        self.assertEqual(audit.read(), b"cache")
        self.assertEqual(audit.read(), b"cache")
        audit.validate()
        wrong = ablation.CalibrationCacheAudit(b"cache")
        wrong.read(b"wrong")
        with self.assertRaisesRegex(RuntimeError, "differed"):
            wrong.validate()
        batch = ablation.CalibrationCacheAudit(b"cache")
        batch.read()
        with self.assertRaisesRegex(RuntimeError, "recalibration"):
            batch.record_batch()
        output = ablation.CalibrationCacheAudit(b"cache")
        output.read()
        output.write(b"wrong")
        with self.assertRaisesRegex(RuntimeError, "write bytes"):
            output.validate()

        class Config:
            def __init__(self, result):
                self.result = result
                self.created = []
                self.attached = []

            def create_timing_cache(self, data):
                self.created.append(data)
                return data

            def set_timing_cache(self, cache, ignore_mismatch):
                self.attached.append((cache, ignore_mismatch))
                return self.result

        config = Config(True)
        self.assertEqual(ablation.attach_timing_cache(config, b"timing"),
                         {"called": True, "ignore_mismatch": False, "return_value": True})
        self.assertEqual(config.created, [b"timing"])
        self.assertEqual(config.attached, [(b"timing", False)])
        with self.assertRaisesRegex(RuntimeError, "ignore_mismatch=False"):
            ablation.attach_timing_cache(Config(False), b"timing")

    def test_classification_exact_variation_and_incomplete(self):
        def row(repeat, payload, value=0.9):
            return {"repeat": repeat, "prediction_payload_sha256": payload,
                    "capture": {"metrics": {"map50": value}},
                    "size": {"metrics": {"all": {"map50": value}}}}

        exact = [row(1, "same"), row(2, "same"), row(3, "same")]
        self.assertEqual(ablation.classify_arm(exact), "replay_exact_observed")
        changed = [row(1, "a"), row(2, "b"), row(3, "a")]
        self.assertEqual(ablation.classify_arm(changed), "replay_variation_observed")
        self.assertEqual(ablation.classify_arm(exact[:2]), "incomplete_or_invalid")
        self.assertEqual(ablation.classify_arm(exact, ["workload"]), "incomplete_or_invalid")

    def test_commands_bind_arm_repeat_and_do_not_use_arbitrary_engine_dispatch(self):
        repo = Path("/repo")
        root = repo / "results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1"
        command = ablation.ablation_capture_command(repo, root, "bbox_fp32", 2,
                                                     root / "bbox_fp32/repeat_2", "0",
                                                     {644963: "/snap/snapd-desktop-integration/391/usr/bin/snapd-desktop-integration"})
        self.assertIn("--precision-head-ablation-study", command)
        self.assertIn("--ablation-arm", command)
        self.assertIn("bbox_fp32", command)
        self.assertIn("--repeat-index", command)
        self.assertIn("2", command)
        self.assertNotIn("--repeat-study", command)
        self.assertNotIn("--timing-cache-study", command)
        build_command = ablation._build_child_command(repo, root, "classification_fp32", 3, {})
        self.assertEqual(build_command[build_command.index("--arm") + 1], "classification_fp32")
        self.assertEqual(build_command[build_command.index("--repeat") + 1], "3")

    def test_capture_dispatch_has_exact_study_path_arm_repeat_scope(self):
        from capture_cctsdb_validator import validate_precision_head_ablation_scope

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            root = repo / "results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1"
            self.assertEqual(validate_precision_head_ablation_scope(
                repo, root, "classification_fp32", 3, "fp16"), root.resolve())
            with self.assertRaisesRegex(ValueError, "exactly"):
                validate_precision_head_ablation_scope(repo, repo / "other",
                                                       "classification_fp32", 3, "fp16")
            with self.assertRaisesRegex(ValueError, "valid study and arm"):
                validate_precision_head_ablation_scope(repo, root, "other", 3, "fp16")
            with self.assertRaisesRegex(ValueError, "repeat-index"):
                validate_precision_head_ablation_scope(repo, root, "classification_fp32", 3, "uniform")

    def test_settings_and_flags_are_step_a_values_with_only_locked_precision_intervention(self):
        self.assertEqual(ablation.REPLAY_SETTINGS, replay.REPLAY_SETTINGS)
        self.assertTrue(ablation.REPLAY_SETTINGS["int8"])
        self.assertFalse(ablation.REPLAY_SETTINGS["fp16"])
        self.assertFalse(ablation.REPLAY_SETTINGS["tf32"])
        self.assertEqual(ablation.REPLAY_SETTINGS["workspace_bytes"], 4 << 30)
        self.assertEqual(ablation.REPLAY_SETTINGS["builder_optimization_level"], 3)
        self.assertEqual(ablation.REPLAY_SETTINGS["avg_timing_iterations"], 1)
        self.assertEqual(ablation.EXPECTED_BUILDER_FLAGS, 2 + 512)  # INT8 + OBEY; no FP16/TF32 flag is introduced.

    def test_external_workload_or_incomplete_telemetry_becomes_review_flag(self):
        guard = {"telemetry_status": "limited", "external_workload_detected": True,
                 "blocked_processes": [{"pid": 9}], "unmatched_confirmations": []}
        row = {
            "arm": "bbox_fp32", "repeat": 1,
            "build": {"gpu_before": {"process_guard": guard},
                       "gpu_after": {"process_guard": guard}},
            "capture": {"gpu_before": {"process_guard": guard},
                        "gpu_after": {"process_guard": guard}, "status": "pass"},
            "verification": {"native_matching_status": "pass", "size_diagnostic": "completed"},
            "build_manifest": {"calibration_batches_consumed": 0,
                                "calibration_cache_read": True,
                                "calibration_cache_read_violations": [],
                                "calibration_cache_write_violations": []},
        }
        reasons = ablation._guard_invalid_reasons(row)
        self.assertTrue(any("telemetry" in reason for reason in reasons))
        self.assertTrue(any("external_workload" in reason for reason in reasons))

    def test_study_manifest_contract_binds_source_hashes_arms_order_and_dev_only_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source_manifest = repo / "source-study.json"
            baseline_manifest = repo / "baseline-study.json"
            source_manifest.write_text("source", encoding="utf-8")
            baseline_manifest.write_text("baseline", encoding="utf-8")
            source = {
                "manifest": {"git_commit": replay.SOURCE_CODE_COMMIT},
                "manifest_path": source_manifest,
                "baseline_study_path": baseline_manifest,
                "build_manifests": {1: {"builder_flags": ablation.EXPECTED_BUILDER_FLAGS,
                                        "sigmoid_fp32_constraints": ["sigmoid"]}},
                "input_hashes": {"weights_sha256": ablation.FROZEN_WEIGHTS_SHA256},
            }
            manifest = ablation.expected_study_manifest(source)
            ablation.validate_ablation_study_manifest(manifest, source)
            self.assertEqual(manifest["arms"], list(ablation.ARMS))
            self.assertEqual(manifest["dev_images"], 1636)
            self.assertEqual(manifest["dev_instances"], 2706)
            self.assertIn("no retraining", manifest["scope"])
            bad = copy.deepcopy(manifest)
            bad["capture_order"] = list(reversed(bad["capture_order"]))
            with self.assertRaisesRegex(ValueError, "capture_order"):
                ablation.validate_ablation_study_manifest(bad, source)

    def test_ablation_capture_input_rejects_wrong_root_before_engine_access(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            with self.assertRaisesRegex(ValueError, "exactly"):
                ablation.ablation_capture_inputs(
                    repo, repo / "results/measurement_audit_v1/other", "baseline_int8", 1)

    def test_source_and_accepted_baseline_hashes_are_required_and_direct_weights_are_measured(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source, weights = contract_fixture(repo)
            with locked_contract_constants(repo, source, weights) as _:
                contract = ablation.validate_ablation_sources(repo)
                self.assertEqual(contract["baseline_study"]["study"], ablation.BASELINE_STUDY)
                self.assertEqual(contract["input_hashes"]["weights_sha256"], sha256(weights))
                self.assertEqual(contract["baseline_capture"]["status"], "pass")
                weights.write_bytes(b"changed-weights")
                with self.assertRaisesRegex(ValueError, "input hash mismatch"):
                    ablation.validate_ablation_sources(repo)

    def test_main_all_builds_all_twelve_children_before_evaluate_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            output = repo / "results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1"
            source = {"input_hashes": {"locked": "yes"}}
            events = []

            def fake_run(command, **_kwargs):
                events.append(list(command))
                return SimpleNamespace(returncode=0)

            helpers = {"parse_desktop_confirmations": lambda _values: {}}
            with patch.object(ablation, "validate_ablation_sources",
                              return_value=source), \
                 patch.object(ablation, "preflight", return_value=(
                     {"environment": {}}, GPU, {"matched": True})), \
                 patch.object(ablation, "_write_study_manifest"), \
                 patch.object(ablation, "evaluate",
                              side_effect=lambda *_args: events.append("evaluate") or 0), \
                 patch.object(ablation.subprocess, "run", side_effect=fake_run):
                result = ablation.main(["--out-dir", str(output)], repo_override=repo,
                                       helpers=helpers)
            self.assertEqual(result, 0)
            build_commands = [event for event in events if isinstance(event, list)]
            self.assertEqual(len(build_commands), 12)
            self.assertEqual(
                [(event[event.index("--arm") + 1], int(event[event.index("--repeat") + 1]))
                 for event in build_commands],
                [(item["arm"], item["repeat"]) for item in ablation.build_plan()])
            self.assertEqual(events[-1], "evaluate")

            output.mkdir(parents=True, exist_ok=True)
            with patch.object(ablation, "validate_ablation_sources", return_value=source):
                with self.assertRaises(SystemExit):
                    ablation.main(["--out-dir", str(output)], repo_override=repo,
                                  helpers=helpers)

    def test_build_phase_requires_prepared_manifest_and_never_resumes_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            output = repo / "results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1"
            with self.assertRaises(SystemExit):
                ablation.main(["--phase", "build", "--arm", "bbox_fp32", "--repeat", "1",
                               "--out-dir", str(output)], repo_override=repo,
                              helpers={"parse_desktop_confirmations": lambda _values: {}})
            output.mkdir(parents=True)
            with patch.object(ablation, "validate_ablation_sources", return_value={"input_hashes": {}}):
                with self.assertRaises(SystemExit):
                    ablation.main(["--phase", "all", "--out-dir", str(output)], repo_override=repo,
                                  helpers={"parse_desktop_confirmations": lambda _values: {}})

    def test_evaluate_runs_all_twelve_captures_after_build_validation_and_aggregates_three_per_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source_root, weights = contract_fixture(repo)
            output = repo / "results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1"
            output.mkdir(parents=True)
            events = []
            args = SimpleNamespace(device="0", confirmed_desktop={})

            def fake_validate(_output, _source, arm, repeat):
                dest = output / arm / f"repeat_{repeat}"
                dest.mkdir(parents=True, exist_ok=True)
                engine = dest / "model.engine"
                if not engine.exists():
                    engine.write_bytes(f"{arm}-{repeat}".encode())
                manifest_path = dest / "build_manifest.json"
                if not manifest_path.exists():
                    write_json(manifest_path, {"arm": arm, "repeat": repeat})
                selected = {
                    "baseline_int8": [],
                    "bbox_fp32": ["/model.23/cv2.0.conv"],
                    "classification_fp32": ["/model.23/cv3.0.conv"],
                    "both_fp32": ["/model.23/cv2.0.conv", "/model.23/cv3.0.conv"],
                }[arm]
                return {
                    "study": ablation.STUDY, "arm": arm, "repeat": repeat,
                    "engine_sha256": sha256(engine), "gpu_before": copy.deepcopy(GPU),
                    "gpu_after": copy.deepcopy(GPU),
                    "calibration_batches_consumed": 0, "calibration_cache_read": True,
                    "calibration_cache_read_violations": [], "calibration_cache_write_violations": [],
                    "constraint_audit": {"matched_layer_names": selected,
                                         "matched_layer_count": len(selected),
                                         "matched_layer_types": {name: "CONVOLUTION" for name in selected}},
                }

            def fake_run_child(command, _repo):
                command_text = " ".join(str(item) for item in command)
                if "capture_cctsdb_validator.py" in command_text:
                    capture_dir = Path(command[command.index("--out-dir") + 1])
                    run_dir = capture_dir.parent
                    arm = command[command.index("--ablation-arm") + 1]
                    repeat = int(command[command.index("--repeat-index") + 1])
                    events.append(("capture", arm, repeat))
                    capture_dir.mkdir(parents=True, exist_ok=True)
                    build_manifest_path = run_dir / "build_manifest.json"
                    build_manifest_path.write_text(
                        json.dumps({"arm": arm, "repeat": repeat}), encoding="utf-8")
                    payload = minimal_payload()
                    predictions_path = capture_dir / "validator_predictions.json"
                    write_json(predictions_path, payload)
                    build = fake_validate(output, None, arm, repeat)
                    write_json(capture_dir / "capture_report.json", {
                        "dataset_split": "CCTSDB2021/dev", "images": 1636, "instances": 2706,
                        "status": "pass", "runtime_arguments": copy.deepcopy(replay.EXPECTED_RUNTIME),
                        "metrics": {"map50": 0.9, "map50_95": 0.8, "precision": 0.7, "recall": 0.6},
                        "model_sha256": build["engine_sha256"],
                        "engine_provenance_sha256": sha256(run_dir / "build_manifest.json"),
                        "predictions_sha256": sha256(predictions_path),
                        "gpu_before": copy.deepcopy(GPU), "gpu_after": copy.deepcopy(GPU),
                    })
                    return 100 + repeat
                events.append(("verify",))
                capture_dir = Path(command[command.index("--capture-dir") + 1])
                verification_dir = Path(command[command.index("--out-dir") + 1])
                report = json.loads((capture_dir / "capture_report.json").read_text(encoding="utf-8"))
                verification_dir.mkdir(parents=True, exist_ok=True)
                write_json(verification_dir / "verification_summary.json", {
                    "dataset_split": "CCTSDB2021/dev", "model_sha256": report["model_sha256"],
                    "native_matching_status": "pass", "size_diagnostic": "completed",
                    "capture_hash_match": "exact_bytes",
                    "capture_prediction_sha256": report["predictions_sha256"],
                    "capture_report_sha256": sha256(capture_dir / "capture_report.json"),
                })
                write_json(verification_dir / "native_matching.json", {"status": "pass"})
                write_json(verification_dir / "size_coco_xml.json", minimal_size())
                return 200

            with locked_contract_constants(repo, source_root, weights) as _:
                contract = ablation.validate_ablation_sources(repo)
                write_json(output / "study_manifest.json", ablation.expected_study_manifest(contract))
                with patch.object(ablation, "_validate_ablation_build", side_effect=fake_validate):
                    result = ablation.evaluate(repo, output, args, contract,
                                               {"run_child": fake_run_child})
            self.assertEqual(result, 0)
            expected_events = []
            for item in ablation.build_plan():
                expected_events.extend([("capture", item["arm"], item["repeat"]), ("verify",)])
            self.assertEqual(events, expected_events)
            summary = json.loads((output / "comparison_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "replay_exact_observed")
            self.assertEqual(summary["classification"], "replay_exact_observed")
            self.assertEqual(summary["diagnostic_branch_classification"], "no_branch_signal")
            self.assertEqual({arm: len(summary["arms"][arm]["records"]) for arm in ablation.ARMS},
                             {arm: 3 for arm in ablation.ARMS})
            self.assertEqual(summary["build_order"], ablation.build_plan())
            self.assertEqual(summary["capture_order"], ablation.build_plan())


if __name__ == "__main__":
    unittest.main()
