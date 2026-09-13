import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_uniform_inference_repeat import (
    DATASET_SPLIT,
    EXPECTED_RUNTIME,
    PREDICTION_PAYLOAD_VERSION,
    ROUND_ORDER,
    aggregate_within_engine,
    capture_command,
    ensure_output_absent,
    parse_gpu_identity,
    prediction_difference,
    prediction_payload_hash,
    resolve_protocol_paths,
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
        self.assertEqual(PREDICTION_PAYLOAD_VERSION, "uniform_inference_repeat_prediction_payload_v1")


if __name__ == "__main__":
    unittest.main()
