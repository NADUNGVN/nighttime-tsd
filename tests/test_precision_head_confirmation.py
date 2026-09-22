import copy
import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_schedule_rounds_use_exact_canonical_interleaving_and_reject_mutations(self):
        schedule = readiness.generate_schedule(self.config)
        scored = [row for row in schedule if row["model"] == "yolov8n" and row["scored"]]
        expected_cells = [
            (None, "fp16"),
            *[(selection, arm) for selection in readiness.SELECTION_IDS for arm in readiness.ARM_NAMES],
        ]
        self.assertEqual([(row["selection"], row["arm"]) for row in scored[:13]], expected_cells)
        self.assertEqual([(row["selection"], row["arm"]) for row in scored[13:26]], expected_cells[4:] + expected_cells[:4])
        self.assertEqual([(row["selection"], row["arm"]) for row in scored[26:]], expected_cells[8:] + expected_cells[:8])
        self.assertEqual({row["repeat"] for row in scored if row["round"] == 1}, {1})
        self.assertEqual({row["repeat"] for row in scored if row["round"] == 2}, {2})
        self.assertEqual({row["repeat"] for row in scored if row["round"] == 3}, {3})
        mutated = copy.deepcopy(schedule)
        mutated[4]["repeat"] = 99
        with self.assertRaisesRegex(ValueError, "canonical order"):
            readiness.validate_schedule(mutated, self.config)
        mutated_selection = copy.deepcopy(schedule)
        mutated_selection[4]["selection"] = "U44"
        with self.assertRaisesRegex(ValueError, "canonical order"):
            readiness.validate_schedule(mutated_selection, self.config)

    def test_canonical_calibration_metadata_and_train_only_paths(self):
        records = [
            readiness.validate_calibration_manifest(REPO, self.config["canonical_input_commit"], selection)
            for selection in self.config["calibration_selections"]
        ]
        self.assertEqual([record["manifest_contract"]["selected_size"] for record in records], [1024] * 3)
        self.assertTrue(all(record["manifest_contract"]["train_only"] for record in records))
        overlap = readiness.calibration_overlap(records)
        self.assertEqual(set(overlap["pairwise_image_id_overlap"]), {"U42:U43", "U42:U44", "U43:U44"})

    def test_calibration_recipe_evidence_reads_size_from_selection_contract(self):
        with patch.object(
            readiness,
            "validate_calibration_manifest",
            side_effect=lambda *_args, **_kwargs: {"status": "verified"},
        ):
            evidence = readiness.calibration_recipe_evidence(REPO, self.config)
        self.assertEqual(evidence["requested_size"], 1024)
        self.assertTrue(evidence["train_only"])
        self.assertEqual(len(evidence["selections"]), 3)

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

        bad_nested = copy.deepcopy(payload)
        bad_nested["files"][0]["source_image"] = "train/images/nested/00006.jpg"
        with self.assertRaises(ValueError):
            readiness.validate_calibration_payload(bad_nested, selection)
        bad_stem = copy.deepcopy(payload)
        bad_stem["files"][0]["source_label"] = "train/labels/different.txt"
        with self.assertRaises(ValueError):
            readiness.validate_calibration_payload(bad_stem, selection)

    def test_same_count_wrong_dev_ids_and_shapes_are_rejected(self):
        canonical = [{"image": f"{index:05d}.jpg", "orig_shape": [720, 1280]} for index in range(1636)]
        current_names = [row["image"] for row in canonical]
        current_labels = [f"{Path(name).stem}.txt" for name in current_names]
        shapes = {name: [720, 1280] for name in current_names}
        self.assertEqual(readiness.validate_dev_inventory_identity(current_names, current_labels, shapes, canonical), [])
        wrong_ids = list(current_names)
        wrong_ids[-1] = "99999.jpg"
        self.assertIn("dev_image_ids_differ_from_canonical_reference", readiness.validate_dev_inventory_identity(wrong_ids, current_labels, shapes, canonical))
        wrong_shapes = dict(shapes)
        wrong_shapes["00000.jpg"] = [1080, 1920]
        self.assertIn("dev_shape_differs_from_canonical_reference:00000.jpg", readiness.validate_dev_inventory_identity(current_names, current_labels, wrong_shapes, canonical))
        wrong_labels = list(current_labels)
        wrong_labels[0] = "different.txt"
        self.assertIn("dev_image_label_stem_set_mismatch", readiness.validate_dev_inventory_identity(current_names, wrong_labels, shapes, canonical))

    def test_materialized_calibration_yaml_and_bytes_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "data"
            expected_dir = root / "calibration" / "uniform_test_n2"
            image_dir = expected_dir / "images"
            (source_root / "train/images").mkdir(parents=True)
            image_dir.mkdir(parents=True)
            rows = []
            for name, payload in (("00001.jpg", b"one"), ("00002.jpg", b"two")):
                source = source_root / "train/images" / name
                source.write_bytes(payload)
                (image_dir / name).write_bytes(payload)
                rows.append({"source_image": f"train/images/{name}", "source_label": f"train/labels/{Path(name).stem}.txt"})
            yaml_path = expected_dir / "calibration.yaml"
            yaml_path.write_text(
                f"path: {str(expected_dir).replace(chr(92), '/') }\ntrain: images\nval: images\nnames:\n  0: prohibitory\n  1: mandatory\n  2: warning\nnc: 3\n",
                encoding="utf-8",
            )
            complete = readiness.parse_materialized_calibration_yaml(yaml_path, expected_dir, rows, source_root)
            self.assertEqual(complete["status"], "complete")
            (image_dir / "00002.jpg").write_bytes(b"substituted")
            invalid_bytes = readiness.parse_materialized_calibration_yaml(yaml_path, expected_dir, rows, source_root)
            self.assertEqual(invalid_bytes["status"], "invalid")
            (image_dir / "00002.jpg").write_bytes(b"two")
            (image_dir / "extra.jpg").write_bytes(b"extra")
            invalid_extra = readiness.parse_materialized_calibration_yaml(yaml_path, expected_dir, rows, source_root)
            self.assertEqual(invalid_extra["status"], "invalid")
            yaml_path.write_text("path: wrong\ntrain: images\nval: images\nnc: 3\n", encoding="utf-8")
            invalid_yaml = readiness.parse_materialized_calibration_yaml(yaml_path, expected_dir, rows, source_root)
            self.assertEqual(invalid_yaml["status"], "invalid")
            same_basename_parent = root / "wrong-parent" / expected_dir.name
            yaml_path.write_text(
                f"path: {str(same_basename_parent).replace(chr(92), '/') }\ntrain: images\nval: images\nnames:\n  0: prohibitory\n  1: mandatory\n  2: warning\nnc: 3\n",
                encoding="utf-8",
            )
            invalid_parent = readiness.parse_materialized_calibration_yaml(yaml_path, expected_dir, rows, source_root)
            self.assertEqual(invalid_parent["status"], "invalid")
            self.assertTrue(any("expected" in error for error in invalid_parent["errors"]))
            yaml_path.write_text("path: [unterminated\n", encoding="utf-8")
            malformed = readiness.parse_materialized_calibration_yaml(yaml_path, expected_dir, rows, source_root)
            self.assertEqual(malformed["status"], "invalid")
            self.assertTrue(any(error.startswith("YAMLError:") for error in malformed["errors"]))

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

    def test_config_mutation_fails_immutable_identity_check(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            mutated = copy.deepcopy(self.config)
            mutated["runtime"]["conf"] = 0.9
            path.write_text(json.dumps(mutated), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "immutable section hash mismatch: runtime"):
                readiness.load_config(path, REPO)

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
        model_records = [
            {"label": model["label"], "status": "verified", "errors": []}
            for model in self.config["models"]
        ]
        dataset = {"status": "verified", "errors": []}
        missing = [
            {
                "id": selection["id"],
                "status": "missing",
                "errors": [],
                "manifest_contract": {"status": "canonical_manifest_valid", "image_ids": []},
                "materialization": {
                    "status": "missing_materialization",
                    "missing": ["materialized_calibration_yaml"],
                    "errors": [],
                },
            }
            for selection in self.config["calibration_selections"]
        ]
        # Use an explicit fixture instead of relying on whether a developer's
        # checkout happens to contain the server materializations.
        with patch.object(readiness, "inspect_frozen_model", side_effect=model_records), \
             patch.object(readiness, "validate_dataset_contract", return_value=dataset), \
             patch.object(readiness, "validate_calibration_manifest", side_effect=missing):
            manifest = readiness.build_readiness(REPO, self.config, probe_models=True)
        self.assertFalse(manifest["scored_run_authorized"])
        self.assertFalse(manifest["gpu_used"])
        self.assertFalse(manifest["export_performed"])
        self.assertFalse(manifest["tensorrt_build_performed"])
        self.assertEqual(manifest["scored_matrix_gate"], "blocked_deferred_graph_validation")
        self.assertFalse(manifest["raw_inputs_ready_for_server_prepare"])
        self.assertFalse(any("model_probe_not_run" in item for item in manifest["unresolved_checks"]))
        self.assertIn("semantic_sha256", manifest["config_identity"])
        self.assertIn("script_sha256", manifest["execution_provenance"])
        self.assertEqual(manifest["accounting"]["total_builder_invocations"], 84)
        self.assertTrue(any("materialized_calibration_yaml" in item for item in manifest["missing_prerequisites"]))

    def test_nested_materialization_status_controls_raw_readiness(self):
        model_records = [
            {"label": model["label"], "status": "verified", "errors": []}
            for model in self.config["models"]
        ]
        dataset = {
            "status": "verified",
            "errors": [],
            "train_image_ids": ["train"],
            "dev_image_ids": ["dev"],
            "test_image_ids": ["test"],
            "train_inventory": {"status": "verified"},
        }
        complete = [
            {
                "id": selection["id"],
                "status": "verified",
                "errors": [],
                "manifest_contract": {"status": "canonical_manifest_valid", "image_ids": []},
                "materialization": {"status": "complete", "missing": [], "errors": []},
            }
            for selection in self.config["calibration_selections"]
        ]
        with patch.object(readiness, "inspect_frozen_model", side_effect=model_records), \
             patch.object(readiness, "validate_dataset_contract", return_value=dataset), \
             patch.object(readiness, "validate_calibration_manifest", side_effect=complete):
            positive = readiness.build_readiness(REPO, self.config, probe_models=True)
        self.assertTrue(positive["raw_inputs_ready_for_server_prepare"])
        self.assertEqual(positive["status"], "ready_for_server_prepare_review")

        invalid = copy.deepcopy(complete)
        invalid[1]["status"] = "invalid"
        invalid[1]["materialization"] = {"status": "invalid", "missing": [], "errors": ["image bytes mismatch"]}
        with patch.object(readiness, "inspect_frozen_model", side_effect=model_records), \
             patch.object(readiness, "validate_dataset_contract", return_value=dataset), \
             patch.object(readiness, "validate_calibration_manifest", side_effect=invalid):
            negative = readiness.build_readiness(REPO, self.config, probe_models=True)
        self.assertFalse(negative["raw_inputs_ready_for_server_prepare"])
        self.assertEqual(negative["status"], "unresolved")
        self.assertIn("U43:materialization_status:invalid", negative["unresolved_checks"])
        self.assertIn("U43:materialization_error:image bytes mismatch", negative["unresolved_checks"])

        missing = copy.deepcopy(complete)
        missing[0]["status"] = "missing"
        missing[0]["materialization"] = {
            "status": "missing_materialization",
            "missing": ["materialized_calibration_yaml"],
            "errors": [],
        }
        with patch.object(readiness, "inspect_frozen_model", side_effect=model_records), \
             patch.object(readiness, "validate_dataset_contract", return_value=dataset), \
             patch.object(readiness, "validate_calibration_manifest", side_effect=missing):
            missing_manifest = readiness.build_readiness(REPO, self.config, probe_models=True)
        self.assertFalse(missing_manifest["raw_inputs_ready_for_server_prepare"])
        self.assertEqual(missing_manifest["status"], "readiness_complete_scored_matrix_blocked")
        self.assertIn("U42:materialized_calibration_yaml", missing_manifest["missing_prerequisites"])

    def test_split_inventory_missing_and_train_dev_overlap_are_not_verified(self):
        source_image = next((REPO / "data/processed/cctsdb2021_clean/dev/images").glob("*.jpg"))
        source_label = REPO / "data/processed/cctsdb2021_clean/dev/labels" / f"{source_image.stem}.txt"
        from PIL import Image
        with Image.open(source_image) as image:
            shape = [image.height, image.width]
        config = copy.deepcopy(self.config)
        config["dev_contract"] = copy.deepcopy(config["dev_contract"])
        config["dev_contract"].update({"images": 1, "instances": len(source_label.read_text(encoding="utf-8").splitlines())})
        canonical = [{"image": source_image.name, "stem": source_image.stem, "orig_shape": shape}]
        split_inventory = {
            "status": "verified",
            "images": {"train": 1, "dev": 1, "official_test": 1},
            "scope": "fixture",
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dev_images = root / "data/processed/cctsdb2021_clean/dev/images"
            dev_labels = root / "data/processed/cctsdb2021_clean/dev/labels"
            dev_images.mkdir(parents=True)
            dev_labels.mkdir(parents=True)
            (dev_images / source_image.name).write_bytes(source_image.read_bytes())
            (dev_labels / source_label.name).write_bytes(source_label.read_bytes())
            with patch.object(readiness, "_canonical_dev_reference", return_value={"records": canonical}), \
                 patch.object(readiness, "_canonical_dataset_inventory", return_value=split_inventory):
                missing = readiness.validate_dataset_contract(root, config)
            self.assertEqual(missing["status"], "unresolved")
            self.assertIn("train_image_inventory_missing", missing["errors"])
            self.assertIn("test_exclusion_image_inventory_missing", missing["errors"])
            self.assertEqual(missing["test_exclusion_inventory"]["status"], "missing")

            train_images = root / "data/processed/cctsdb2021_clean/train/images"
            train_labels = root / "data/processed/cctsdb2021_clean/train/labels"
            test_images = root / "data/processed/cctsdb2021_clean/test/images"
            train_images.mkdir(parents=True)
            train_labels.mkdir(parents=True)
            test_images.mkdir(parents=True)
            (train_images / source_image.name).write_bytes(source_image.read_bytes())
            (train_labels / source_label.name).write_bytes(source_label.read_bytes())
            (test_images / source_image.name).write_bytes(source_image.read_bytes())
            with patch.object(readiness, "_canonical_dev_reference", return_value={"records": canonical}), \
                 patch.object(readiness, "_canonical_dataset_inventory", return_value=split_inventory):
                overlap = readiness.validate_dataset_contract(root, config)
            self.assertIn("train_dev_id_overlap:1", overlap["errors"])
            self.assertIn("dev_test_id_overlap:1", overlap["errors"])
            self.assertEqual(overlap["train_inventory"]["status"], "verified")
            self.assertEqual(overlap["test_exclusion_inventory"]["status"], "verified")

    def test_unsupported_ultralytics_is_unresolved_but_other_package_diffs_are_recorded(self):
        observed = {"torch": "2.8.0+cu129", "ultralytics": "8.4.101", "numpy": "2.4.2", "pycocotools": "2.0.9"}
        evidence = readiness.runtime_compatibility_evidence(observed, self.config["runtime"])
        self.assertEqual(evidence["status"], "unresolved")
        self.assertFalse(evidence["native_head_probe_supported"])
        self.assertTrue(any("ultralytics_unsupported" in error for error in evidence["blocking_errors"]))
        self.assertEqual(len(evidence["package_mismatches"]), 4)
        self.assertEqual(evidence["server_runtime_status"], "not_verified_by_cpu_readiness")


if __name__ == "__main__":
    unittest.main()
