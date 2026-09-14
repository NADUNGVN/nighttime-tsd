import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_cctsdb_measurement import sha256
from run_uniform_inference_repeat import (
    DATASET_SPLIT,
    EXPECTED_RUNTIME,
    PREDICTION_PAYLOAD_VERSION,
    ROUND_ORDER,
    aggregate_within_engine,
    BACKGROUND_WORKLOAD_VARIANT,
    capture_command,
    ensure_output_absent,
    environment_probe_command,
    main,
    parse_gpu_identity,
    parse_environment_probe,
    prediction_difference,
    prediction_payload_hash,
    resolve_protocol_paths,
    run_capture_pair,
    run_environment_preflight,
    round_plan,
    validate_device_argument,
    validate_capture_contract,
    validate_engine_manifest,
    validate_gpu_identity,
)


def prediction_fixture():
    return {
        "schema_version": 2,
        "capture_mode": "same_val_process_batch",
        "iou_thresholds": [0.5, 0.55, 0.6],
        "coordinate_contract": "xyxy original coordinates; validator input remains authoritative",
        "model_sha256": "engine-identity-is-not-in-payload-hash",
        "data_sha256": "data-identity-is-not-in-payload-hash",
        "created_utc": "one timestamp",
        "records": [
            {"image": "a.jpg", "orig_shape": [100, 120],
             "xyxy": [[1.0, 2.0, 11.0, 12.0], [20.0, 21.0, 30.0, 31.0]],
             "confidence": [0.9, 0.8], "class_id": [0, 1],
             "validator_input": {"target_class_id": [0], "target_xyxy": [[1, 2, 11, 12]], "imgsz": [640, 640], "ratio_pad": [[1, 1], [0, 0]]},
             "validator_statistics": {"tp": [[True], [False]]}},
            {"image": "b.jpg", "orig_shape": [100, 120],
             "xyxy": [], "confidence": [], "class_id": [],
             "validator_input": {"target_class_id": [], "target_xyxy": [], "imgsz": [640, 640], "ratio_pad": [[1, 1], [0, 0]]},
             "validator_statistics": {"tp": []}},
        ],
    }


def metric_capture(map50, map50_95, precision=0.8, recall=0.7):
    return {"map50": map50, "map50_95": map50_95, "precision": precision, "recall": recall,
            "per_class_ap50": {"prohibitory": map50, "mandatory": map50, "warning": map50}}


def metric_size(map50, map50_95):
    return {"instances": 1, "map50": map50, "map50_95": map50_95,
            "per_class": {"prohibitory": {"ap50": map50, "ap50_95": map50_95}}}


def aggregate_row(round_id, value, payload_hash):
    return {
        "round": round_id, "engine_sha256": "engine",
        "prediction_payload_sha256": payload_hash, "metrics_exact": round_id == 1,
        "capture_report": {"metrics": metric_capture(value, value - 0.1)},
        "size_report": {"metrics": {label: metric_size(value, value - 0.1)
                                      for label in ("all", "xs", "s", "m", "l", "xl")}},
    }


class UniformInferenceRepeatTests(unittest.TestCase):
    def test_payload_hash_ignores_metadata_but_detects_bbox_and_confidence(self):
        baseline = prediction_fixture()
        changed_metadata = copy.deepcopy(baseline)
        changed_metadata["created_utc"] = "another timestamp"
        changed_metadata["runtime_arguments"] = {"data": "/different/server/path"}
        changed_metadata["model_sha256"] = "another-engine"
        self.assertEqual(prediction_payload_hash(baseline), prediction_payload_hash(changed_metadata))

        changed_bbox = copy.deepcopy(baseline)
        changed_bbox["records"][0]["xyxy"][0][0] = 1.25
        result = prediction_difference(baseline, changed_bbox)
        self.assertFalse(result["exact"])
        self.assertIn("a.jpg", result["differences"]["bbox_changed"])

        changed_confidence = copy.deepcopy(baseline)
        changed_confidence["records"][0]["confidence"][0] = 0.91
        result = prediction_difference(baseline, changed_confidence)
        self.assertIn("a.jpg", result["differences"]["confidence_changed"])

    def test_payload_difference_detects_order_membership_and_duplicate(self):
        baseline = prediction_fixture()
        reordered = copy.deepcopy(baseline)
        reordered["records"][0], reordered["records"][1] = reordered["records"][1], reordered["records"][0]
        result = prediction_difference(baseline, reordered)
        self.assertIn("image_order_changed", result["differences"])

        missing = copy.deepcopy(baseline)
        missing["records"].pop()
        self.assertIn("image_membership_changed", prediction_difference(baseline, missing)["differences"])

        duplicate = copy.deepcopy(baseline)
        duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
        result = prediction_difference(baseline, duplicate)
        self.assertEqual(result["differences"]["current_duplicate_images"], ["a.jpg"])

    def test_detection_order_uses_complete_triple_multiset(self):
        baseline = prediction_fixture()
        reordered = copy.deepcopy(baseline)
        record = reordered["records"][0]
        for key in ("xyxy", "confidence", "class_id"):
            record[key].reverse()
        result = prediction_difference(baseline, reordered)
        self.assertEqual(result["differences"], {"detection_order_changed": ["a.jpg"]})

    def test_engine_hash_mismatch_is_rejected(self):
        study = {"source_weights_sha256": "weights", "onnx_sha256": "onnx", "settings": {"batch": 1}}
        manifest = {"study": "uniform_build_repeat_v1", "repeat": 2,
                    "source_weights_sha256": "weights", "onnx_sha256": "onnx",
                    "settings": {"batch": 1}, "engine_sha256": "expected"}
        self.assertEqual(validate_engine_manifest(manifest, 2, study, "expected"), "expected")
        with self.assertRaisesRegex(ValueError, "engine hash"):
            validate_engine_manifest(manifest, 2, study, "other")

    def test_protocol_paths_keep_logical_ids_separate_from_server_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source, output = resolve_protocol_paths(repo)
            self.assertEqual(source, repo.resolve() / "results/measurement_audit_v1/server_uniform_build_repeat_v1")
            self.assertEqual(output, repo.resolve() / "results/measurement_audit_v1/server_uniform_inference_repeat_v1")
            self.assertEqual((source, output), resolve_protocol_paths(repo))
            with self.assertRaisesRegex(ValueError, "server_uniform_inference_repeat_v1"):
                from run_uniform_inference_repeat import validate_output_target
                validate_output_target(repo, repo / "results/measurement_audit_v1/uniform_inference_repeat_v1")
            concurrent_source, concurrent_output = resolve_protocol_paths(repo, concurrent=True)
            self.assertEqual(concurrent_source, source)
            self.assertEqual(concurrent_output.name, "server_uniform_inference_repeat_concurrent_v1")
            self.assertEqual(validate_output_target(repo, concurrent_output, concurrent=True),
                             (source, concurrent_output))

    def test_gpu_uuid_driver_and_device_binding_are_strict(self):
        device = "GPU-test, Quadro RTX 8000, 595.71.05, P8, 35, 9 W, 300 MHz, 405 MHz, 32 MiB"
        study = {"gpu_before": {"device": device}}
        current = {"device": device}
        self.assertEqual(parse_gpu_identity(current), {"uuid": "GPU-test", "name": "Quadro RTX 8000", "driver_version": "595.71.05"})
        self.assertTrue(validate_gpu_identity(current, study)["matched"])
        for changed in (device.replace("GPU-test", "GPU-other"), device.replace("595.71.05", "580.178.04")):
            with self.assertRaisesRegex(ValueError, "GPU identity"):
                validate_gpu_identity({"device": changed}, study)
        with self.assertRaisesRegex(ValueError, "--device 0"):
            validate_device_argument("1")

    def test_malformed_detection_arrays_are_rejected_before_order_comparison(self):
        malformed = prediction_fixture()
        malformed["records"][0]["confidence"].pop()
        with self.assertRaisesRegex(ValueError, "different lengths"):
            prediction_payload_hash(malformed)
        malformed = prediction_fixture()
        malformed["records"][0]["xyxy"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "malformed bbox"):
            prediction_payload_hash(malformed)

    def test_output_protection_and_round_order(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            ensure_output_absent(output)
            output.mkdir()
            with self.assertRaises(FileExistsError):
                ensure_output_absent(output)
            plan = round_plan(Path(directory) / "planned")
        self.assertEqual([item["engine_repeat"] for item in plan], [1, 2, 3, 2, 3, 1, 3, 1, 2])
        self.assertEqual(ROUND_ORDER, ((1, 2, 3), (2, 3, 1), (3, 1, 2)))

    def test_within_engine_aggregation_has_sample_sd_and_no_metric_selection(self):
        payload_hash = "payload"
        rows = [aggregate_row(1, 0.8, payload_hash), aggregate_row(2, 0.6, "changed"), aggregate_row(3, 1.0, "changed-again")]
        aggregate = aggregate_within_engine(rows)
        self.assertEqual(aggregate["baseline"], "round_1_first_capture; not selected by metric")
        self.assertFalse(aggregate["prediction_payload_exact_all"])
        self.assertAlmostEqual(aggregate["coco_xml"]["xs"]["map50"]["mean"], 0.8)
        self.assertAlmostEqual(aggregate["coco_xml"]["xs"]["map50"]["range_pp"], 40.0)
        self.assertAlmostEqual(aggregate["coco_xml"]["xs"]["map50"]["sample_std_pp"], 20.0)

    def test_test_split_and_runtime_contract_are_rejected(self):
        capture = {"dataset_split": DATASET_SPLIT, "images": 1636, "instances": 2706,
                   "status": "pass", "runtime_arguments": dict(EXPECTED_RUNTIME)}
        verification = {"dataset_split": DATASET_SPLIT, "native_matching_status": "pass", "size_diagnostic": "completed"}
        validate_capture_contract(capture, verification)
        bad = copy.deepcopy(capture)
        bad["dataset_split"] = "CCTSDB2021/test"
        with self.assertRaisesRegex(ValueError, "CCTSDB2021/dev"):
            validate_capture_contract(bad, verification)

    def test_capture_command_is_repeat_scoped_and_has_no_representation_override(self):
        command = capture_command(Path("/repo"), Path("/repo/results/measurement_audit_v1/uniform_build_repeat_v1"),
                                  3, Path("/repo/results/round_1/engine_3"), "0", {445: "/usr/bin/Xorg"})
        self.assertIn("--repeat-study", command)
        self.assertIn("--repeat-index", command)
        self.assertNotIn("--representation", command)
        self.assertIn("445=/usr/bin/Xorg", command)
        concurrent = capture_command(Path("/repo"), Path("/repo/results/measurement_audit_v1/server_uniform_build_repeat_v1"),
                                     2, Path("/repo/results/measurement_audit_v1/server_uniform_inference_repeat_concurrent_v1/round_1/engine_2"),
                                     "0", {}, {446: "python opcm_full_bgfg.py"}, True)
        self.assertIn("--allow-confirmed-background-workload", concurrent)
        self.assertIn("446=python opcm_full_bgfg.py", concurrent)
        self.assertEqual(BACKGROUND_WORKLOAD_VARIANT, "operator_confirmed_background_compute_v1")
        self.assertEqual(PREDICTION_PAYLOAD_VERSION, "uniform_inference_repeat_prediction_payload_v1")

    def test_environment_preflight_requires_one_structured_json_child_report(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            stdout = json.dumps({"schema_version": 1, "status": "ok",
                                 "environment": {"gpu": "Quadro RTX 8000"}})
            calls = []

            def runner(command, **kwargs):
                calls.append((command, kwargs))
                return SimpleNamespace(stdout=stdout, stderr="runtime warning", returncode=0)

            report = run_environment_preflight(repo, runner)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["environment"]["gpu"], "Quadro RTX 8000")
        self.assertEqual(report["child"]["stderr"], "runtime warning")
        self.assertEqual(calls[0][0], environment_probe_command(repo))
        self.assertEqual(calls[0][1], {"cwd": repo, "capture_output": True, "text": True, "check": False})
        with self.assertRaisesRegex(RuntimeError, "pure JSON"):
            parse_environment_probe("warning\n" + stdout, "", 0)
        with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
            parse_environment_probe(json.dumps({"status": "error", "error": "CUDA unavailable"}), "", 2)

    def test_run_child_waits_before_returning(self):
        events = []

        class Process:
            pid = 987

            def wait(self):
                events.append("wait")
                return 0

        with patch("run_uniform_inference_repeat.subprocess.Popen", return_value=Process()):
            from run_uniform_inference_repeat import _run_child
            self.assertEqual(_run_child(["child"], Path("/repo")), 987)
        self.assertEqual(events, ["wait"])

    def test_capture_pair_keeps_capture_then_verify_child_order(self):
        events = []

        def run_child(command, repo):
            events.append((Path(command[1]).name, command))
            return len(events)

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            run_dir = repo / "results/round_1/engine_2"
            result = run_capture_pair(repo, repo / "source", 2, run_dir, "0", {},
                                      repo / "xml.zip", run_child)

        self.assertEqual(result[:2], (1, 2))
        self.assertEqual([item[0] for item in events],
                         ["capture_cctsdb_validator.py", "verify_cctsdb_capture.py"])
        self.assertEqual(events[0][1][events[0][1].index("--repeat-index") + 1], "2")
        self.assertIn("--capture-dir", events[1][1])

    def test_main_preflight_failure_creates_no_output_and_starts_no_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            _source, output = resolve_protocol_paths(repo)
            events = []

            def fail_preflight(_repo):
                events.append("preflight")
                raise RuntimeError("probe child failed")

            def unexpected_child(_command, _repo):
                events.append("capture")
                raise AssertionError("capture must not start after failed preflight")

            with self.assertRaisesRegex(RuntimeError, "probe child failed"):
                main(["--out-dir", str(output)], repo_override=repo,
                     helpers={"environment_preflight": fail_preflight,
                              "run_child": unexpected_child})

        self.assertEqual(events, ["preflight"])
        self.assertFalse(output.exists())

    def test_main_cpu_mock_runs_preflight_before_all_capture_children_in_fixed_round_order(self):
        device = "GPU-test, Quadro RTX 8000, 595.71.05, P8, 35, 9 W, 300 MHz, 405 MHz, 32 MiB"
        environment = {"torch": "test", "ultralytics": "test", "tensorrt": "test", "numpy": "test",
                       "python": "test", "cuda": "12.1", "gpu": "Quadro RTX 8000", "pycocotools": "test"}
        gpu_snapshot = {
            "device": device,
            "process_guard": {"telemetry_status": "complete", "external_workload_detected": True,
                               "external_workload_authorized": True, "blocked_processes": [],
                               "unmatched_confirmations": [], "background_workload": [{
                                   "pid": 446, "expected_command": "python opcm_full_bgfg.py",
                                   "observed_command": "python opcm_full_bgfg.py", "status": "running",
                                   "observed_on_gpu": True, "authorized": True, "blocking": False,
                               }]},
        }
        size_report = {
            "metric_id": "coco_xml_size_v1", "rules": {"xs": 210},
            "evaluator_source_sha256": "evaluator", "xml_sha256": "xml",
            "metrics": {label: metric_size(0.5, 0.4) for label in ("all", "xs", "s", "m", "l", "xl")},
        }
        prediction = prediction_fixture()
        child_events = []
        helper_events = []

        def write_json(path, value):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(value, handle, allow_nan=False)
                handle.write("\n")

        def preflight(_repo):
            helper_events.append("preflight")
            return {"status": "ok", "environment": environment,
                    "child": {"return_code": 0, "stdout_sha256": "probe", "stderr": ""}}

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source_root, output = resolve_protocol_paths(repo, concurrent=True)
            source_root.mkdir(parents=True)
            write_json(source_root / "study_manifest.json", {
                "study": "uniform_build_repeat_v1", "source_weights_sha256": "weights",
                "onnx_sha256": "onnx", "settings": {"batch": 1}, "environment": environment,
                "gpu_before": gpu_snapshot, "gpu_after": gpu_snapshot,
            })
            engine_hashes = {}
            for engine_id in (1, 2, 3):
                engine_path = source_root / f"repeat_{engine_id}/model.engine"
                engine_path.parent.mkdir(parents=True, exist_ok=True)
                engine_path.write_bytes(f"engine-{engine_id}".encode("ascii"))
                engine_hashes[engine_id] = sha256(engine_path)
                write_json(source_root / f"repeat_{engine_id}/build_manifest.json", {
                    "study": "uniform_build_repeat_v1", "repeat": engine_id,
                    "source_weights_sha256": "weights", "onnx_sha256": "onnx", "settings": {"batch": 1},
                    "engine_sha256": engine_hashes[engine_id], "calibration_cache_sha256": "cache",
                    "cache_matches_historical_uniform": True,
                })

            reference_capture = repo / "results/measurement_audit_v1/server_fp16_capture_v1"
            reference_verification = repo / "results/measurement_audit_v1/server_native_size_v1"
            write_json(reference_capture / "capture_report.json", {})
            write_json(reference_capture / "validator_predictions.json", prediction)
            write_json(reference_verification / "size_coco_xml.json", size_report)

            def repeat_capture_inputs(_repo, _root, engine_id):
                return (source_root / f"repeat_{engine_id}/model.engine", None, None, None, None,
                        engine_hashes[engine_id])

            def run_child(command, _repo):
                command_text = " ".join(command)
                if "capture_cctsdb_validator.py" in command_text:
                    self.assertIn("--allow-confirmed-background-workload", command)
                    self.assertIn("446=python opcm_full_bgfg.py", command)
                    engine_id = int(command[command.index("--repeat-index") + 1])
                    out_dir = Path(command[command.index("--out-dir") + 1])
                    capture_dir = out_dir
                    capture_dir.mkdir(parents=True, exist_ok=True)
                    predictions_path = capture_dir / "validator_predictions.json"
                    write_json(predictions_path, prediction)
                    report = {
                        "dataset_split": DATASET_SPLIT, "images": 1636, "instances": 2706, "status": "pass",
                        "runtime_arguments": dict(EXPECTED_RUNTIME), "model_sha256": engine_hashes[engine_id],
                        "predictions_sha256": sha256(predictions_path), "metrics": metric_capture(0.5, 0.4),
                        "gpu_before": gpu_snapshot, "gpu_after": gpu_snapshot,
                        "external_gpu_workload_detected": True,
                    }
                    write_json(capture_dir / "capture_report.json", report)
                    child_events.append(("capture", engine_id))
                else:
                    capture_dir = Path(command[command.index("--capture-dir") + 1])
                    verification_dir = Path(command[command.index("--out-dir") + 1])
                    verification_dir.mkdir(parents=True, exist_ok=True)
                    capture_report_path = capture_dir / "capture_report.json"
                    pred_hash = sha256(capture_dir / "validator_predictions.json")
                    capture_report = json.loads(capture_report_path.read_text(encoding="utf-8"))
                    write_json(verification_dir / "verification_summary.json", {
                        "dataset_split": DATASET_SPLIT, "native_matching_status": "pass",
                        "size_diagnostic": "completed", "model_sha256": capture_report["model_sha256"],
                        "capture_hash_match": "exact_bytes", "capture_prediction_sha256": pred_hash,
                        "capture_report_sha256": sha256(capture_report_path), "global_g0": "review_required",
                    })
                    write_json(verification_dir / "size_coco_xml.json", size_report)
                    child_events.append(("verify", None))
                return len(child_events)

            helpers = {
                "environment_preflight": preflight,
                "write_json": write_json,
                "check_reference": lambda *_args: ({}, size_report),
                "check_same_targets": lambda *_args: None,
                "ensure_idle": lambda _state, _background=None: None,
                "parse_desktop_confirmations": lambda _values: {},
                "parse_background_confirmations": lambda _values: {446: "python opcm_full_bgfg.py"},
                "repeat_capture_inputs": repeat_capture_inputs,
                "snapshot": lambda _confirmations, _background=None: gpu_snapshot,
                "load_records": lambda _payload: {},
                "load_xml": lambda *_args: None,
                "run_child": run_child,
            }

            with patch("uniform_build_repeat.environment",
                       side_effect=AssertionError("parent called CUDA environment")), \
                 patch("run_uniform_inference_repeat.subprocess.run",
                       return_value=SimpleNamespace(stdout="test-head\n")):
                result = main(["--out-dir", str(output),
                               "--confirm-background-process", "446=python opcm_full_bgfg.py"],
                              repo_override=repo, helpers=helpers)

            self.assertEqual(result, 0)
            self.assertEqual(helper_events, ["preflight"])
            self.assertEqual(child_events,
                             [("capture", 1), ("verify", None), ("capture", 2), ("verify", None),
                              ("capture", 3), ("verify", None), ("capture", 2), ("verify", None),
                              ("capture", 3), ("verify", None), ("capture", 1), ("verify", None),
                              ("capture", 3), ("verify", None), ("capture", 1), ("verify", None),
                              ("capture", 2), ("verify", None)])
            summary = json.loads((output / "repeat_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "inference_repeatability_completed_review_required")
            manifest = json.loads((output / "study_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["environment_preflight"]["status"], "ok")
            self.assertEqual(manifest["protocol_variant"], BACKGROUND_WORKLOAD_VARIANT)
            self.assertEqual(manifest["background_workload_authorization"]["confirmed_processes"],
                             [{"pid": 446, "expected_command": "python opcm_full_bgfg.py"}])
            self.assertTrue(any("external_workload" in flag for flag in summary["review_flags"]))


if __name__ == "__main__":
    unittest.main()
