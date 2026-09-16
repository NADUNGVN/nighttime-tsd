import copy
import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import prepare_precision_head_confirmation as readiness  # noqa: E402


class PrecisionHeadConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = readiness.load_config(REPO / "configs/precision_head_confirmation_v1.json", REPO)

    def test_config_locks_models_contracts_and_accounting(self):
        self.assertEqual([row["label"] for row in self.config["models"]], ["yolov8n", "yolo26n"])
        self.assertEqual(self.config["arms"], list(readiness.ARM_NAMES))
        self.assertEqual(self.config["accounting"]["total_builder_invocations"], 84)
        self.assertEqual(self.config["accounting"]["scored_builds"], 78)
        self.assertEqual(self.config["accounting"]["captures"], 78)
        self.assertIsNone(self.config["decision_rules"]["fp16_noninferiority_margin"])
        self.assertIsNone(self.config["decision_rules"]["size_gate_pp"])

    def test_schedule_has_two_model_blocks_six_auxiliary_and_78_scored_jobs(self):
        schedule = readiness.generate_schedule(self.config)
        evidence = readiness.validate_schedule(schedule, self.config)
        self.assertEqual(len(schedule), 84)
        self.assertEqual(evidence["auxiliary_calibration_builds"], 6)
        self.assertEqual(evidence["scored_int8_builds"], 72)
        self.assertEqual(evidence["scored_fp16_builds"], 6)
        self.assertEqual(evidence["captures"], 78)
        self.assertEqual([row["model"] for row in schedule[:42]], ["yolov8n"] * 42)
        self.assertEqual([row["model"] for row in schedule[42:]], ["yolo26n"] * 42)
        self.assertTrue(all(not row["scored"] and not row["capture_required"] for row in schedule if row["phase"] == "auxiliary_calibration"))
        self.assertTrue(all(row["scored"] and row["capture_required"] for row in schedule if row["phase"] != "auxiliary_calibration"))
        keys = [(row["model"], row["selection"], row["arm"], row["repeat"]) for row in schedule]
        self.assertEqual(len(keys), len(set(keys)))

    def test_schedule_auxiliary_precedes_each_selection_cell(self):
        schedule = readiness.generate_schedule(self.config)
        for model in ("yolov8n", "yolo26n"):
            for selection in readiness.SELECTION_IDS:
                rows = [row for row in schedule if row["model"] == model and row["selection"] == selection]
                self.assertEqual(rows[0]["phase"], "auxiliary_calibration")
                self.assertEqual(len(rows), 13)
                self.assertEqual({row["arm"] for row in rows[1:]}, set(readiness.ARM_NAMES))

    def test_canonical_calibration_metadata_and_train_only_paths(self):
        records = [
            readiness.validate_calibration_manifest(REPO, self.config["canonical_input_commit"], selection)
            for selection in self.config["calibration_selections"]
        ]
        self.assertEqual([record["manifest_contract"]["selected_size"] for record in records], [1024] * 3)
        self.assertTrue(all(record["manifest_contract"]["train_only"] for record in records))
        overlap = readiness.calibration_overlap(records)
        self.assertEqual(set(overlap["pairwise_image_id_overlap"]), {"U42:U43", "U42:U44", "U43:U44"})

    def test_calibration_wrong_seed_dev_path_and_duplicate_are_rejected(self):
        selection = self.config["calibration_selections"][0]
        payload = json.loads(
            readiness.git_blob(REPO, self.config["canonical_input_commit"], REPO / selection["manifest"]).decode("utf-8")
        )
        bad_seed = copy.deepcopy(payload)
        bad_seed["seed"] = 43
        with self.assertRaises(ValueError):
            readiness.validate_calibration_payload(bad_seed, selection)
        bad_split = copy.deepcopy(payload)
        bad_split["files"][0]["source_image"] = "dev/images/00006.jpg"
        with self.assertRaises(ValueError):
            readiness.validate_calibration_payload(bad_split, selection)
        bad_duplicate = copy.deepcopy(payload)
        bad_duplicate["files"][1]["source_image"] = bad_duplicate["files"][0]["source_image"]
        with self.assertRaises(ValueError):
            readiness.validate_calibration_payload(bad_duplicate, selection)

    def test_head_contract_rejects_ambiguous_branch_and_wrong_representation(self):
        self.assertEqual(readiness.derive_active_branches(False, ["cv2", "cv3"]), (["cv2", "cv3"], []))
        self.assertEqual(
            readiness.derive_active_branches(True, ["cv2", "cv3", "one2one_cv2", "one2one_cv3"]),
            (["one2one_cv2", "one2one_cv3"], ["cv2", "cv3"]),
        )
        with self.assertRaises(ValueError):
            readiness.derive_active_branches(True, ["cv2", "cv3"])

        expected = copy.deepcopy(self.config["models"][1]["expected_contract"])
        actual = copy.deepcopy(expected)
        actual.update({
            "primary_output_representation": "tuple_tensor_plus_dict",
            "active_convolution_mapping": {name: [{"name": f"{name}.0"}] for name in expected["active_branches"]},
            "mapping_status": "pytorch_structure_verified_onnx_deferred",
        })
        self.assertEqual(readiness.validate_model_contract(actual, expected), [])
        wrong_representation = copy.deepcopy(actual)
        wrong_representation["primary_output_representation"] = "raw_tensor_only"
        self.assertTrue(readiness.validate_model_contract(wrong_representation, expected))
        wrong_branch = copy.deepcopy(actual)
        wrong_branch["active_branches"] = ["cv2", "cv3"]
        self.assertTrue(readiness.validate_model_contract(wrong_branch, expected))
        wrong_semantics_same_shape = copy.deepcopy(actual)
        wrong_semantics_same_shape["output_kind"] = "decoded_raw_plus_head_dict"
        self.assertTrue(readiness.validate_model_contract(wrong_semantics_same_shape, expected))

    def test_no_fallback_or_gpu_matrix_side_effects_in_readiness_module(self):
        source = inspect.getsource(readiness)
        self.assertNotIn("torch.cuda", source)
        schedule = readiness.generate_schedule(self.config)
        for row in schedule:
            if row["phase"] == "scored_int8":
                self.assertFalse(row["calibration_cache_creation"])
                self.assertTrue(row["capture_required"])
            if row["phase"] == "scored_fp16":
                self.assertIsNone(row["selection"])
        self.assertTrue(self.config["execution_boundary"]["parent_must_not_import_tensorrt_or_touch_cuda"])
        self.assertEqual(self.config["execution_boundary"]["scored_execution"], "one_isolated_child_per_builder_or_capture_job")
        self.assertEqual(self.config["export_contract"]["end2end"], "preserve_frozen_model_head_flag")

    def test_output_writer_is_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            readiness.write_json_no_overwrite(path, {"first": True})
            with self.assertRaises(FileExistsError):
                readiness.write_json_no_overwrite(path, {"second": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"first": True})

    def test_dataset_contract_uses_actual_canonical_dev_counts(self):
        evidence = readiness.validate_dataset_contract(REPO, self.config)
        self.assertEqual(evidence["status"], "verified")
        self.assertEqual(evidence["images_observed"], 1636)
        self.assertEqual(evidence["instances_observed"], 2706)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("ultralytics") is not None,
        "CPU model dependencies are not installed",
    )
    def test_actual_frozen_yolov8n_and_yolo26n_head_output_contracts(self):
        for model_config in self.config["models"]:
            record = readiness.inspect_frozen_model(REPO, model_config, probe=True)
            self.assertEqual(record["status"], "verified", record.get("errors"))
            self.assertFalse(record["gpu_used"])
            self.assertFalse(record["export_performed"])
            self.assertFalse(record["tensorrt_imported"])
            self.assertEqual(record["mapping_status"], "pytorch_structure_verified_onnx_deferred")

    def test_build_readiness_preserves_missing_prepare_artifacts_without_authorizing_run(self):
        manifest = readiness.build_readiness(REPO, self.config, probe_models=False)
        self.assertFalse(manifest["scored_run_authorized"])
        self.assertFalse(manifest["gpu_used"])
        self.assertFalse(manifest["export_performed"])
        self.assertFalse(manifest["tensorrt_build_performed"])
        self.assertEqual(manifest["accounting"]["total_builder_invocations"], 84)
        self.assertTrue(any("materialized_calibration_yaml" in item for item in manifest["missing_prerequisites"]))


if __name__ == "__main__":
    unittest.main()
