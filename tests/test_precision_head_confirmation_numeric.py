import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import verify_precision_head_confirmation_numeric as numeric  # noqa: E402


def _selection(selection_id="U42", count=8):
    images = [f"train/images/{index:05d}.jpg" for index in range(count)]
    source = [
        {"image": image, "source_sha256": f"{index + 1:064x}", "bytes": 100 + index}
        for index, image in enumerate(images)
    ]
    materialized = [
        {"image": image, "source_sha256": row["source_sha256"], "materialized_sha256": row["source_sha256"], "bytes": row["bytes"]}
        for image, row in zip(images, source)
    ]
    return {
        "id": selection_id,
        "manifest_contract": {"status": "canonical_manifest_valid", "train_only": True, "image_ids": images},
        "selection_audit": {"source_bytes": source},
        "materialization": {"image_bytes": materialized},
    }


class PrecisionHeadNumericTests(unittest.TestCase):
    def test_protocol_locks_cpu_tolerances_and_fixture(self):
        accepted = {"manifest": {"calibration_readiness": {"selections": [_selection("U42"), _selection("U43", 2), _selection("U44", 2)]}}}
        plan = numeric.build_fixture_plan(accepted)
        self.assertEqual(plan["forward_image_count"], 8)
        self.assertEqual([row["image_id"] for row in plan["forward_images"]], [f"{index:05d}" for index in range(8)])
        self.assertEqual([row["selection"] for row in plan["preprocess_trace_images"]], ["U42", "U43", "U44"])
        self.assertEqual(numeric.LOCKED_TOLERANCES["float32"], {"rtol": 1e-4, "atol": 1e-5})
        self.assertTrue(plan["no_labels_read"])

    def test_fixture_rejects_non_train_or_reordered_content(self):
        bad = _selection()
        bad["manifest_contract"]["image_ids"][0] = "dev/images/00000.jpg"
        with self.assertRaises(ValueError):
            numeric._selection_rows(bad, 8)
        bad = _selection()
        bad["selection_audit"]["source_bytes"].reverse()
        with self.assertRaises(ValueError):
            numeric._selection_rows(bad, 8)

    def test_fixture_rejects_duplicate_ids(self):
        bad = _selection()
        bad["manifest_contract"]["image_ids"][1] = bad["manifest_contract"]["image_ids"][0]
        bad["selection_audit"]["source_bytes"][1]["image"] = bad["selection_audit"]["source_bytes"][0]["image"]
        bad["materialization"]["image_bytes"][1]["image"] = bad["materialization"]["image_bytes"][0]["image"]
        with self.assertRaises(ValueError):
            numeric._selection_rows(bad, 8)

    def test_bound_image_uses_canonical_dataset_root_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            canonical = repo / "data/processed/cctsdb2021_clean/train/images/00001.jpg"
            materialized = repo / "data/processed/cctsdb2021_clean/calibration/uniform_s42_n1024/images/00001.jpg"
            decoy = repo / "train/images/00001.jpg"
            canonical.parent.mkdir(parents=True)
            materialized.parent.mkdir(parents=True)
            decoy.parent.mkdir(parents=True)
            content = b"canonical image bytes"
            canonical.write_bytes(content)
            materialized.write_bytes(content)
            decoy.write_bytes(b"wrong-root decoy")
            row = {"selection": "U42", "image": "train/images/00001.jpg", "expected_sha256": numeric.sha256_bytes(content), "expected_bytes": len(content), "materialized_expected_sha256": numeric.sha256_bytes(content)}
            evidence = numeric.verify_bound_image(repo, row)
            self.assertEqual(Path(evidence["source"]["path"]), Path("data/processed/cctsdb2021_clean/train/images/00001.jpg"))
            self.assertEqual(numeric.resolve_bound_source_image(repo, row), canonical.resolve())
            canonical.write_bytes(b"changed image bytes")
            with self.assertRaises(ValueError):
                numeric.verify_bound_image(repo, row)
            bad = dict(row, image="train/images/../secret.jpg")
            with self.assertRaises(ValueError):
                numeric.resolve_bound_source_image(repo, bad)
            bad = dict(row, image="../train/images/00001.jpg")
            with self.assertRaises(ValueError):
                numeric.resolve_bound_materialized_image(repo, bad)

    def test_fixture_verification_preserves_selection_bindings_when_image_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            content = b"shared anchor"
            source = repo / "data/processed/cctsdb2021_clean/train/images/00001.jpg"
            source.parent.mkdir(parents=True)
            source.write_bytes(content)
            for selection in ("U42", "U43"):
                materialized = repo / f"data/processed/cctsdb2021_clean/calibration/uniform_s{selection[1:]}_n1024/images/00001.jpg"
                materialized.parent.mkdir(parents=True)
                materialized.write_bytes(content)
            digest = numeric.sha256_bytes(content)
            row42 = {"selection": "U42", "manifest_order": 0, "image": "train/images/00001.jpg", "image_id": "00001", "expected_sha256": digest, "expected_bytes": len(content), "materialized_expected_sha256": digest}
            row43 = dict(row42, selection="U43", manifest_order=1)
            result = numeric.verify_fixture_files(repo, {"forward_images": [row42], "preprocess_trace_images": [row43]})
            self.assertEqual(result["unique_image_count"], 1)
            self.assertEqual([item["selection"] for item in result["images"]["train/images/00001.jpg"]["selection_bindings"]], ["U42", "U43"])

    def test_preprocess_array_contract_is_rgb_chw_float32_and_batched(self):
        bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        tensor = numeric.preprocess_array_contract(bgr, np)
        self.assertEqual(tensor.shape, (1, 3, 1, 2))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_allclose(tensor[0, :, 0, 0], [3 / 255, 2 / 255, 1 / 255])
        self.assertTrue(tensor.flags.c_contiguous)

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in ("torch", "ultralytics", "onnxruntime", "cv2")),
        "pinned CPU producer dependencies are unavailable",
    )
    def test_actual_ultralytics_cpu_preprocess_trace_uses_pinned_path(self):
        numeric.set_cpu_environment()
        import cv2

        runtime = numeric.load_child_runtime({"runtime": {"torch": "2.8.0+cu129", "ultralytics": "8.4.102", "numpy": "2.4.2", "pycocotools": "2.0.10"}})
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "fixture.jpg"
            image = np.zeros((480, 800, 3), dtype=np.uint8)
            image[:, :, 0] = 10
            image[:, :, 1] = 20
            image[:, :, 2] = 30
            self.assertTrue(cv2.imwrite(str(image_path), image))
            tensor, trace = numeric.trace_preprocess(image_path, runtime, stride=32)
        self.assertEqual(tuple(tensor.shape), (1, 3, 640, 640))
        self.assertEqual(str(tensor.dtype), "torch.float32")
        self.assertEqual(trace["decoded"]["color_order"], "BGR")
        self.assertEqual(trace["letterbox"]["padding"], {"top": 128, "bottom": 128, "left": 0, "right": 0})
        self.assertTrue(trace["same_tensor_for_source_and_onnx"])

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in ("torch", "ultralytics", "onnxruntime", "cv2")),
        "pinned CPU producer dependencies are unavailable",
    )
    def test_actual_calibration_component_trace_is_bounded_and_separate(self):
        numeric.set_cpu_environment()
        import cv2

        runtime = numeric.load_child_runtime({"runtime": {"torch": "2.8.0+cu129", "ultralytics": "8.4.102", "numpy": "2.4.2", "pycocotools": "2.0.10"}})
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "calibration_anchor.jpg"
            image = np.zeros((480, 800, 3), dtype=np.uint8)
            image[:, :, 0] = 10
            image[:, :, 1] = 20
            image[:, :, 2] = 30
            self.assertTrue(cv2.imwrite(str(image_path), image))
            trace = numeric.trace_calibration_preprocess(image_path, runtime, stride=32)
        self.assertEqual(trace["stage"], "bounded_calibration_component_trace")
        self.assertFalse(trace["loader_dispatch"]["Exporter.get_int8_calibration_dataloader"])
        self.assertFalse(trace["dataset_side_effects"]["labels_read"])
        self.assertEqual(trace["load_image"]["resized_hw"], [384, 640])
        self.assertEqual(trace["letterbox"]["padding"], {"top": 128, "bottom": 128, "left": 0, "right": 0})
        self.assertEqual(trace["formatted_uint8"]["layout"], "RGB_CHW")
        self.assertEqual(trace["normalization"]["operation"], "float32(rgb_chw) / 255.0")

    def test_v8_raw_comparison_reports_pass_and_same_shape_corruption(self):
        reference = np.zeros((1, 7, 8400), dtype=np.float32)
        observed = reference.copy()
        observed[0, 3, 4] = 1e-5
        passed = numeric.compare_primary("yolov8n", reference, observed, np)
        self.assertEqual(passed["status"], "pass")
        observed[0, 3, 4] = 1.0
        failed = numeric.compare_primary("yolov8n", reference, observed, np)
        self.assertEqual(failed["status"], "fail")
        self.assertGreater(failed["mismatch_count"], 0)
        self.assertEqual(set(failed["channel_results"]), {"boxes", "scores"})

    def test_zero_reference_relative_diagnostics_are_json_safe(self):
        reference = np.zeros((1, 7, 8400), dtype=np.float32)
        observed = reference.copy()
        observed[0, 0, 0] = 1.0
        result = numeric.compare_primary("yolov8n", reference, observed, np)
        self.assertEqual(result["status"], "fail")
        self.assertGreater(result["channel_results"]["boxes"]["relative_infinite_count"], 0)
        self.assertIsNone(result["channel_results"]["boxes"]["max_relative"])
        json.dumps(result, allow_nan=False)

    def test_v8_wrong_shape_and_nonfinite_are_unresolved(self):
        reference = np.zeros((1, 7, 8400), dtype=np.float32)
        wrong = np.zeros((1, 7, 10), dtype=np.float32)
        self.assertEqual(numeric.compare_primary("yolov8n", reference, wrong, np)["status"], "unresolved")
        reference[0, 0, 0] = np.inf
        self.assertEqual(numeric.compare_primary("yolov8n", reference, reference.copy(), np)["status"], "unresolved")

    def test_v26_fixed_row_comparison_reports_ties_without_rematching(self):
        reference = np.zeros((1, 300, 6), dtype=np.float32)
        reference[..., 0:4] = [1, 2, 3, 4]
        reference[..., 4] = 0.5
        reference[..., 5] = 1
        observed = reference.copy()
        result = numeric.compare_primary("yolo26n", reference, observed, np)
        self.assertEqual(result["status"], "pass")
        self.assertGreater(result["tie_count_reference_scores"], 0)
        self.assertIn("fixed native row index", result["tie_handling"])

    def test_v26_wrong_class_and_fractional_class(self):
        reference = np.zeros((1, 300, 6), dtype=np.float32)
        reference[..., 5] = 0
        wrong_class = reference.copy()
        wrong_class[..., 5] = 1
        self.assertEqual(numeric.compare_primary("yolo26n", reference, wrong_class, np)["status"], "fail")
        fractional = reference.copy()
        fractional[..., 5] = 0.5
        self.assertEqual(numeric.compare_primary("yolo26n", reference, fractional, np)["status"], "unresolved")

    def test_native_output_rejects_wrong_debug_branch(self):
        class FakeTensor:
            def detach(self):
                return self

            def to(self, _device):
                return self

            def contiguous(self):
                return self

            def numpy(self):
                return np.zeros((1, 300, 6), dtype=np.float32)

        with self.assertRaises(numeric.NumericUnresolved):
            numeric.extract_native_primary((FakeTensor(), {"one2many": {}}), "yolo26n", np)

    def test_child_command_is_model_specific_and_no_shell(self):
        command = numeric.child_command(Path("scripts/verify.py"), Path("/repo"), "yolov8n", Path("/repo/out/numeric_plan.json"))
        self.assertIn("--child", command)
        self.assertIn("yolov8n", command)
        self.assertNotIn("--model all", " ".join(command))

    def test_source_has_no_trt_or_export_dispatch(self):
        source = Path(numeric.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import tensorrt", source)
        self.assertNotIn("from tensorrt", source)
        self.assertNotIn(".export(", source)
        self.assertNotIn("uniform_build_repeat.prepare", source)
        self.assertIn("CPUExecutionProvider", source)

    def test_graph_audit_manifest_rejects_unverified_mapping(self):
        # Keep this boundary test independent of the real five-file artifact.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / numeric.GRAPH_AUDIT_ROOT_NAME
            (root / "models/yolov8n").mkdir(parents=True)
            (root / "models/yolo26n").mkdir(parents=True)
            for relative in numeric.GRAPH_AUDIT_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            expected_blob_hash = numeric.sha256_bytes(b"x")
            manifest = {"status": "audit_only_completed", "audit_flags": {"export_performed": False}, "models": [{"model": "yolov8n", "status": "audit_only_completed", "mapping_status": "mapping_unresolved"}, {"model": "yolo26n", "status": "audit_only_completed", "mapping_status": "verified"}]}
            model_docs = {
                model: {"status": "audit_only_completed", "mapping_status": "verified", "audit_flags": {"export_performed": False}, "provenance": {"onnx": {"before": {"sha256": numeric.EXPECTED_ONNX_SHA256[model]}}}, "onnx_schema": {"outputs": [{"name": "output0", "shape": numeric.EXPECTED_OUTPUT_SHAPES[model]}]}}
                for model in numeric.MODEL_CHOICES
            }
            with patch.object(numeric, "GRAPH_AUDIT_FILES", {key: expected_blob_hash for key in numeric.GRAPH_AUDIT_FILES}), patch.object(numeric.readiness, "git_blob", return_value=b"x"), patch.object(numeric, "read_json", side_effect=lambda path: manifest if Path(path).name == "graph_audit_manifest.json" else model_docs[Path(path).parent.name]):
                with self.assertRaises(numeric.NumericUnresolved):
                    numeric.validate_graph_audit_artifact(Path(temp), root, list(numeric.MODEL_CHOICES))

    def test_single_model_selection_binds_full_accepted_graph_artifact(self):
        root = REPO / "results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4"
        accepted = numeric.validate_graph_audit_artifact(REPO, root, ["yolov8n"])
        self.assertEqual(set(accepted["accepted_models"]), set(numeric.MODEL_CHOICES))
        self.assertEqual(set(accepted["models"]), {"yolov8n"})

    def test_no_overwrite_is_enforced_by_parent_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            output = repo / "output"
            output.mkdir()
            args = SimpleNamespace(model="all", out_dir=Path("output"), readiness_root=Path("readiness"), graph_audit_root=Path("graph"), source_root=Path("source"))
            with self.assertRaises(FileExistsError):
                numeric.run_parent(args, repo)

    def test_parent_dispatches_two_isolated_children_sequentially(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            (repo / "configs").mkdir()
            (repo / "configs/config.json").write_text("{}", encoding="utf-8")
            (repo / "source").mkdir()
            fixture = {
                "forward_images": [],
                "preprocess_trace_images": [],
                "forward_image_count": 8,
                "selection_rule": "fixed",
            }
            accepted = {"manifest": {}, "files": {}, "accepted_git_commit": "r", "accepted_execution_commit": "e"}
            binding = {"path": str(repo / "configs/config.json"), "sha256": "c", "semantic_sha256": "s"}
            config = {"runtime": {}, "models": []}
            graph_audit = {"files": {}, "models": {model: {"mapping_status": "verified"} for model in numeric.MODEL_CHOICES}}
            args = SimpleNamespace(model="all", out_dir=Path("output"), readiness_root=Path("readiness"), graph_audit_root=Path("graph"), source_root=Path("source"))
            calls = []

            def fake_run(command, **kwargs):
                calls.append(command)
                if command[:3] == ["git", "rev-parse", "HEAD"]:
                    return SimpleNamespace(returncode=0, stdout="head\n", stderr="")
                model = command[command.index("--model") + 1]
                model_dir = repo / "output/models" / model
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "numeric_report.json").write_text(json.dumps({"model": model, "status": "completed", "numeric_status": "pass"}), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="child\n", stderr="")

            with patch.object(numeric.graph, "validate_readiness_artifact", return_value=accepted), patch.object(numeric.graph, "validate_config_binding", return_value=binding), patch.object(numeric.readiness, "load_config", return_value=config), patch.object(numeric, "validate_graph_audit_artifact", return_value=graph_audit), patch.object(numeric, "build_fixture_plan", return_value=fixture), patch.object(numeric, "verify_bound_image"), patch.object(numeric, "_model_plan", return_value={"accepted_contract": {}}), patch.object(numeric, "file_evidence", return_value={"exists": True}), patch.object(numeric.subprocess, "run", side_effect=fake_run):
                result = numeric.run_parent(args, repo)
            self.assertEqual(result, 0)
            self.assertEqual([command[command.index("--model") + 1] for command in calls[1:]], ["yolov8n", "yolo26n"])
            self.assertEqual(json.loads((repo / "output/numeric_manifest.json").read_text(encoding="utf-8"))["status"], "completed")
            self.assertTrue((repo / "output/logs/yolov8n.log").is_file())
            self.assertTrue((repo / "output/logs/yolo26n.log").is_file())

    def test_parent_consumes_existing_child_failure_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            (repo / "configs").mkdir()
            (repo / "configs/config.json").write_text("{}", encoding="utf-8")
            (repo / "source").mkdir()
            fixture = {"forward_images": [], "preprocess_trace_images": [], "forward_image_count": 8, "selection_rule": "fixed"}
            accepted = {"manifest": {}, "files": {}, "accepted_git_commit": "r", "accepted_execution_commit": "e"}
            binding = {"path": str(repo / "configs/config.json"), "sha256": "c", "semantic_sha256": "s"}
            config = {"runtime": {}, "models": []}
            graph_audit = {"files": {}, "models": {model: {"mapping_status": "verified"} for model in numeric.MODEL_CHOICES}}
            args = SimpleNamespace(model="all", out_dir=Path("output"), readiness_root=Path("readiness"), graph_audit_root=Path("graph"), source_root=Path("source"))
            original_failure = {"model": "yolov8n", "status": "failed", "numeric_status": "not_observed", "error": "preserved child failure", "no_silent_resume": True}

            def fake_run(command, **kwargs):
                if command[:3] == ["git", "rev-parse", "HEAD"]:
                    return SimpleNamespace(returncode=0, stdout="head\n", stderr="")
                model = command[command.index("--model") + 1]
                model_dir = repo / "output/models" / model
                model_dir.mkdir(parents=True, exist_ok=True)
                if model == "yolov8n":
                    (model_dir / "failure.json").write_text(json.dumps(original_failure), encoding="utf-8")
                    return SimpleNamespace(returncode=1, stdout="failed\n", stderr="")
                (model_dir / "numeric_report.json").write_text(json.dumps({"model": model, "status": "completed", "numeric_status": "pass"}), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="child\n", stderr="")

            with patch.object(numeric.graph, "validate_readiness_artifact", return_value=accepted), patch.object(numeric.graph, "validate_config_binding", return_value=binding), patch.object(numeric.readiness, "load_config", return_value=config), patch.object(numeric, "validate_graph_audit_artifact", return_value=graph_audit), patch.object(numeric, "build_fixture_plan", return_value=fixture), patch.object(numeric, "verify_bound_image"), patch.object(numeric, "_model_plan", return_value={"accepted_contract": {}}), patch.object(numeric, "file_evidence", return_value={"exists": True}), patch.object(numeric.subprocess, "run", side_effect=fake_run):
                result = numeric.run_parent(args, repo)
            manifest = json.loads((repo / "output/numeric_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(result, 1)
            self.assertEqual(manifest["models"][0], original_failure)
            self.assertEqual(manifest["numeric_verdict"]["status"], "not_observed")
            self.assertIn("models/yolov8n/failure.json", manifest["artifact_inventory"])
            self.assertFalse((repo / "output/models/yolov8n/numeric_report.json").exists())

    def test_parent_writes_fallback_failure_when_child_crashes_before_record(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            (repo / "configs").mkdir()
            (repo / "configs/config.json").write_text("{}", encoding="utf-8")
            (repo / "source").mkdir()
            fixture = {"forward_images": [], "preprocess_trace_images": [], "forward_image_count": 8, "selection_rule": "fixed"}
            accepted = {"manifest": {}, "files": {}, "accepted_git_commit": "r", "accepted_execution_commit": "e"}
            binding = {"path": str(repo / "configs/config.json"), "sha256": "c", "semantic_sha256": "s"}
            config = {"runtime": {}, "models": []}
            graph_audit = {"files": {}, "models": {model: {"mapping_status": "verified"} for model in numeric.MODEL_CHOICES}}
            args = SimpleNamespace(model="yolov8n", out_dir=Path("output"), readiness_root=Path("readiness"), graph_audit_root=Path("graph"), source_root=Path("source"))

            def fake_run(command, **kwargs):
                if command[:3] == ["git", "rev-parse", "HEAD"]:
                    return SimpleNamespace(returncode=0, stdout="head\n", stderr="")
                return SimpleNamespace(returncode=1, stdout="crash\n", stderr="before report\n")

            with patch.object(numeric.graph, "validate_readiness_artifact", return_value=accepted), patch.object(numeric.graph, "validate_config_binding", return_value=binding), patch.object(numeric.readiness, "load_config", return_value=config), patch.object(numeric, "validate_graph_audit_artifact", return_value=graph_audit), patch.object(numeric, "build_fixture_plan", return_value=fixture), patch.object(numeric, "verify_bound_image"), patch.object(numeric, "_model_plan", return_value={"accepted_contract": {}}), patch.object(numeric, "file_evidence", return_value={"exists": True}), patch.object(numeric.subprocess, "run", side_effect=fake_run):
                result = numeric.run_parent(args, repo)
            failure = json.loads((repo / "output/models/yolov8n/failure.json").read_text(encoding="utf-8"))
            self.assertEqual(result, 1)
            self.assertEqual(failure["numeric_status"], "not_observed")
            self.assertTrue(failure["no_silent_resume"])

    def test_child_failure_persists_partial_failure_without_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            output = repo / "output"
            (output / "models/yolov8n").mkdir(parents=True)
            plan = {"output_root": "output"}
            (output / "numeric_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            args = SimpleNamespace(repo_root=repo, plan=Path("output/numeric_plan.json"), model="yolov8n")
            with patch.object(numeric, "run_model_child", side_effect=RuntimeError("stub failure")):
                result = numeric.run_child(args)
            self.assertEqual(result, 1)
            failure = json.loads((output / "models/yolov8n/failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "failed")
            self.assertTrue(failure["no_silent_resume"])


if __name__ == "__main__":
    unittest.main()
