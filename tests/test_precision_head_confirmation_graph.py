from __future__ import annotations

import copy
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import prepare_precision_head_confirmation as readiness
import prepare_precision_head_confirmation_graph as graph


def expected_model(label: str) -> dict:
    if label == "yolov8n":
        active = ["cv2", "cv3"]
        inactive = []
        head_index = 22
        output = [1, 7, 8400]
        boxes = [1, 4, 8400]
        scores = [1, 3, 8400]
        output_kind = "decoded_raw_plus_head_dict"
        postprocess = "none_in_model_head_output"
        nms = "not_in_model_head_output"
    else:
        active = ["one2one_cv2", "one2one_cv3"]
        inactive = ["cv2", "cv3"]
        head_index = 23
        output = [1, 300, 6]
        boxes = [1, 4, 8400]
        scores = [1, 3, 8400]
        output_kind = "end2end_topk_detections_plus_head_dict"
        postprocess = "topk_300_detections_in_model_head_output"
        nms = "postprocess_only_as_defined_by_locked_ultralytics_runtime"
    mappings = {
        active[0]: [{"name": f"model.{head_index}.{active[0]}.0", "module_type": "Conv2d"}],
        active[1]: [{"name": f"model.{head_index}.{active[1]}.0", "module_type": "Conv2d"}],
    }
    return {
        "label": label,
        "head_output_evidence": {
            "head_type": "Detect",
            "head_index": head_index,
            "end2end": label == "yolo26n",
            "active_branches": active,
            "inactive_branches": inactive,
            "output_kind": output_kind,
            "primary_output_shape": output,
            "primary_output_representation": "tuple_tensor_plus_dict",
            "boxes_shape": boxes,
            "scores_shape": scores,
            "nms": nms,
            "postprocess": postprocess,
        },
        "active_convolution_mapping": mappings,
    }


def graph_fixture(label: str, *, add_nonconv: bool = True, shared_downstream: bool = False) -> dict:
    model = expected_model(label)
    active = model["head_output_evidence"]["active_branches"]
    index = model["head_output_evidence"]["head_index"]
    bbox, classification = active
    bbox_name = f"model.{index}/{bbox}.0/Conv"
    class_name = f"model.{index}/{classification}.0/Conv"
    bbox_tensor = "bbox_features"
    class_tensor = "class_features"
    merged = "merged_head"
    output = "output0"
    nodes = [
        {"name": bbox_name, "op_type": "Conv", "inputs": ["features"], "outputs": [bbox_tensor]},
        {"name": class_name, "op_type": "Conv", "inputs": ["features"], "outputs": [class_tensor]},
    ]
    if add_nonconv:
        nodes.append({
            "name": f"model.{index}/{bbox}.0/act/Sigmoid",
            "op_type": "Sigmoid",
            "inputs": [bbox_tensor],
            "outputs": ["bbox_after_helper"],
        })
        # The synthetic graph retains the Conv tensor as the branch-owned
        # source so the helper is auditable but not a precision target.
    nodes.append({
        "name": f"model.{index}/head/Concat",
        "op_type": "Concat",
        "inputs": [bbox_tensor, class_tensor],
        "outputs": [merged],
        "attributes": {"axis": 1},
    })
    if shared_downstream:
        nodes.append({"name": "head/TopK", "op_type": "TopK", "inputs": [merged], "outputs": [output]})
    else:
        nodes[-1]["outputs"] = [output]
    if label == "yolo26n":
        nodes.extend([
            {"name": f"model.{index}/cv2.0/Conv", "op_type": "Conv", "inputs": ["features"], "outputs": ["inactive_bbox"]},
            {"name": f"model.{index}/cv3.0/Conv", "op_type": "Conv", "inputs": ["features"], "outputs": ["inactive_class"]},
        ])
    return {
        "nodes": nodes,
        "inputs": [{"name": "features", "shape": [1, 64, 80, 80]}],
        "outputs": [{"name": output, "shape": model["head_output_evidence"]["primary_output_shape"]}],
        "tensor_shapes": {
            bbox_tensor: [1, 4, 8400],
            class_tensor: [1, 3, 8400],
            merged: [1, 7, 8400],
            output: model["head_output_evidence"]["primary_output_shape"],
        },
    }, model


class TensorLike:
    def __init__(self, shape, rows=None):
        self.shape = tuple(shape)
        self._rows = rows

    def tolist(self):
        if self._rows is None:
            return []
        return self._rows


class PrecisionHeadGraphTests(unittest.TestCase):
    def test_yolov8_dataflow_mapping_excludes_nonconv_helpers_and_records_spans(self):
        fixture, model = graph_fixture("yolov8n")
        result = graph.audit_graph_mapping(fixture, "yolov8n", model)
        self.assertEqual(result["mapping_status"], "verified", result["errors"])
        self.assertEqual(result["branch_owned_target_sets"]["bbox"], ["model.22/cv2.0/Conv"])
        self.assertEqual(result["branch_owned_target_sets"]["classification"], ["model.22/cv3.0/Conv"])
        self.assertEqual(result["branch_merge"]["shape"], [1, 7, 8400])
        self.assertEqual(result["active_branch_audit"]["cv2"]["output_lineage"]["channel_span"], [0, 4])
        self.assertEqual(result["active_branch_audit"]["cv3"]["output_lineage"]["channel_span"], [4, 7])
        self.assertIn("model.22/cv2.0/act/Sigmoid", result["active_branch_audit"]["cv2"]["excluded_non_convolution_nodes"])

    def test_yolo26_keeps_inactive_branches_out_of_targets_and_allows_shared_output_ancestry(self):
        fixture, model = graph_fixture("yolo26n", shared_downstream=True)
        result = graph.audit_graph_mapping(fixture, "yolo26n", model)
        self.assertEqual(result["mapping_status"], "verified", result["errors"])
        self.assertEqual(result["branch_owned_target_sets"]["bbox"], ["model.23/one2one_cv2.0/Conv"])
        self.assertEqual(result["branch_owned_target_sets"]["classification"], ["model.23/one2one_cv3.0/Conv"])
        self.assertTrue(result["active_branch_audit"]["one2one_cv2"]["output_lineage"]["reachability_may_be_shared_after_merge"])
        self.assertTrue(result["inactive_branch_audit"]["cv2"]["excluded_from_precision_targets"])
        self.assertNotIn("model.23/cv2.0/Conv", result["branch_owned_target_sets"]["both_union"])

    def test_missing_ambiguous_and_unreachable_mapping_is_fail_closed(self):
        fixture, model = graph_fixture("yolov8n")
        missing = copy.deepcopy(fixture)
        missing["nodes"][0]["name"] = "model.22/cv2.0/Conv_renamed"
        result = graph.audit_graph_mapping(missing, "yolov8n", model)
        self.assertEqual(result["mapping_status"], "mapping_unresolved")
        self.assertTrue(any("source_conv_match_count" in error for error in result["errors"]))

        ambiguous = copy.deepcopy(fixture)
        ambiguous["nodes"].insert(1, copy.deepcopy(ambiguous["nodes"][0]))
        ambiguous["nodes"][1]["outputs"] = ["bbox_features_duplicate"]
        ambiguous["tensor_shapes"]["bbox_features_duplicate"] = [1, 4, 8400]
        result = graph.audit_graph_mapping(ambiguous, "yolov8n", model)
        self.assertEqual(result["mapping_status"], "mapping_unresolved")
        self.assertTrue(any("match_count" in error for error in result["errors"]))

        unreachable = copy.deepcopy(fixture)
        unreachable["nodes"] = [node for node in unreachable["nodes"] if node["name"] != "model.22/head/Concat"]
        result = graph.audit_graph_mapping(unreachable, "yolov8n", model)
        self.assertEqual(result["mapping_status"], "mapping_unresolved")
        self.assertTrue(any("unreachable_conv" in error or "merge_unresolved" in error for error in result["errors"]))

    def test_nonconv_prefix_and_duplicate_final_merges_are_not_guessed(self):
        fixture, model = graph_fixture("yolov8n")
        duplicate = copy.deepcopy(fixture)
        duplicate["nodes"].append({
            "name": "model.22/head/Concat_duplicate",
            "op_type": "Concat",
            "inputs": ["bbox_features", "class_features"],
            "outputs": ["merged_duplicate"],
            "attributes": {"axis": 1},
        })
        duplicate["tensor_shapes"]["merged_duplicate"] = [1, 7, 8400]
        result = graph.audit_graph_mapping(duplicate, "yolov8n", model)
        self.assertEqual(result["mapping_status"], "mapping_unresolved")
        self.assertIn("branch_owned_channel_merge_unresolved", result["errors"])

    def test_precision_arm_target_sets_are_owned_and_baseline_is_protected(self):
        fixture, model = graph_fixture("yolov8n")
        mapping = graph.audit_graph_mapping(fixture, "yolov8n", model)
        self.assertEqual(graph.precision_target_sets(mapping, "baseline_int8")["target_layers"], [])
        self.assertEqual(graph.precision_target_sets(mapping, "bbox_fp32")["target_layers"], ["model.22/cv2.0/Conv"])
        self.assertEqual(graph.precision_target_sets(mapping, "classification_fp32")["target_layers"], ["model.22/cv3.0/Conv"])
        self.assertEqual(len(graph.precision_target_sets(mapping, "both_fp32")["target_layers"]), 2)
        with self.assertRaises(ValueError):
            graph.precision_target_sets({"branch_owned_target_sets": {"bbox": [], "classification": []}}, "bbox_fp32")

    def test_native_adapters_distinguish_raw_and_end2end_and_do_not_double_nms(self):
        v8 = expected_model("yolov8n")["head_output_evidence"]
        v8_result = graph.validate_native_output("yolov8n", (TensorLike([1, 7, 8400]), {"boxes": object()}), v8)
        self.assertFalse(v8_result["nms_applied"])
        self.assertEqual(v8_result["box_channels"], [0, 4])

        v26 = expected_model("yolo26n")["head_output_evidence"]
        rows = [[10.0, 20.0, 30.0, 40.0, 0.5, 1.0] for _ in range(300)]
        v26_result = graph.validate_native_output("yolo26n", (TensorLike([1, 300, 6], rows), {"one2one": {}}), v26)
        self.assertFalse(v26_result["double_nms"])
        self.assertEqual(v26_result["detection_columns"]["class_id"], 5)

        with self.assertRaises(ValueError):
            graph.validate_native_output("yolov8n", [], v8)
        with self.assertRaises(ValueError):
            graph.validate_native_output("yolo26n", (TensorLike([1, 300, 6], [[0, 0, 1, 1, 0.5, 3]] * 300), {"one2one": {}}), v26)
        wrong_contract = copy.deepcopy(v8)
        wrong_contract["end2end"] = True
        with self.assertRaises(ValueError):
            graph.validate_native_output("yolov8n", (TensorLike([1, 7, 8400]), {}), wrong_contract)

    def test_adapter_contract_is_deferred_until_real_parser_and_forward_evidence(self):
        for label in ("yolov8n", "yolo26n"):
            contract = expected_model(label)["head_output_evidence"]
            adapter = graph.adapter_contract(label, contract)
            self.assertTrue(adapter["double_nms_forbidden"])
            self.assertTrue(adapter["scored_adapter_status"].startswith("deferred"))

    def test_prepare_source_has_no_tensorRT_or_cuda_import_and_child_dispatch_is_scoped(self):
        source = inspect.getsource(graph)
        self.assertNotRegex(source, r"(?m)^\s*import\s+tensorrt")
        self.assertNotRegex(source, r"(?m)^\s*from\s+tensorrt")
        self.assertNotIn("torch.cuda", source)
        command = graph.child_command(REPO, "yolov8n", Path("readiness"), Path("new-output"))
        self.assertIn("--child", command)
        self.assertIn("--model", command)
        self.assertNotIn("--checkpoint", command)
        self.assertNotIn("--engine", command)

    def test_parent_dispatches_one_isolated_child_per_model_without_importing_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            accepted = {"accepted_git_commit": "accepted", "root": str(root / "readiness"), "files": {}, "manifest": {}}
            config = {
                "models": [{"label": "yolov8n"}, {"label": "yolo26n"}],
                "export_contract": {},
                "calibration_recipe": {"preprocessing_helper": "helper.py"},
            }
            current = {
                "selected_models": ["yolov8n", "yolo26n"],
                "checkpoint_bindings": {},
                "selected_calibration_ids": ["U42", "U43", "U44"],
                "calibrations": [{"id": "U42"}, {"id": "U43"}, {"id": "U44"}],
                "dataset": {"status": "verified", "train_dev_overlap": [], "dev_test_overlap": []},
            }
            binding = {"config": config, "path": "config", "sha256": "c", "semantic_sha256": "s"}
            (root / "helper.py").write_text("helper", encoding="utf-8")
            args = SimpleNamespace(model="all", out_dir=output, readiness_root=root / "readiness")
            calls = []

            def fake_run(command, **_kwargs):
                calls.append(command)
                model = command[command.index("--model") + 1]
                model_dir = output / "models" / model
                model_dir.mkdir(parents=True)
                (model_dir / "model_prepare.json").write_text(
                    json.dumps({
                        "model": model,
                        "status": "completed",
                        "graph_mapping": {"status": "verified"},
                        "export": {"onnx_sha256": "x"},
                        "adapter": {"scored_adapter_status": "deferred"},
                    }),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout=f"done {model}", stderr="")

            with patch.object(graph, "validate_readiness_artifact", return_value=accepted), \
                 patch.object(graph, "validate_config_binding", return_value=binding), \
                 patch.object(graph, "validate_current_bindings", return_value=current), \
                 patch.object(graph, "git_head", return_value="test-parent-commit"), \
                 patch.object(graph, "subprocess") as subprocess_mock:
                subprocess_mock.run.side_effect = fake_run
                subprocess_mock.run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
                # Keep Git lookup deterministic while proving that the parent
                # does not import the model/export producer modules.
                subprocess_mock.run.side_effect = fake_run
                result = graph.run_parent(args, root)
            self.assertEqual(result, 0)
            self.assertEqual(len(calls), 2)
            self.assertEqual([cmd[cmd.index("--model") + 1] for cmd in calls], ["yolov8n", "yolo26n"])
            self.assertTrue((output / "graph_preparation_manifest.json").is_file())
            self.assertTrue((output / "logs/yolov8n.log").is_file())
            self.assertTrue((output / "logs/yolo26n.log").is_file())

    def test_prepare_model_preserves_private_copy_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"frozen")
            model_config = {
                "label": "yolov8n",
                "checkpoint": "model.pt",
                "sha256": readiness.sha256_file(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "expected_contract": {
                    "end2end": False,
                    "head_type": "Detect",
                    "head_index": 22,
                    "active_branches": ["cv2", "cv3"],
                    "inactive_branches": [],
                    "output_kind": "decoded_raw_plus_head_dict",
                    "primary_output_shape": [1, 7, 8400],
                    "primary_output_dtype": "torch.float32",
                    "debug_paths": ["boxes", "scores", "feats"],
                    "boxes_shape": [1, 64, 8400],
                    "scores_shape": [1, 3, 8400],
                    "feature_shapes": [[1, 64, 80, 80], [1, 128, 40, 40], [1, 256, 20, 20]],
                    "postprocess": "none_in_model_head_output",
                    "nms": "not_in_model_head_output",
                    "primary_output_representation": "tuple_tensor_plus_dict",
                },
            }
            config = {"models": [model_config], "runtime": {}, "export_contract": {}, "calibration_recipe": {"preprocessing_helper": "helper.py"}}
            (root / "helper.py").write_text("helper", encoding="utf-8")
            accepted_model = expected_model("yolov8n")
            accepted_model["expected_sha256"] = model_config["sha256"]
            accepted_model["expected_bytes"] = checkpoint.stat().st_size
            accepted_model["head_output_evidence"].update(model_config["expected_contract"])
            accepted = {"manifest": {"model_contracts": [accepted_model]}}
            current = {"calibrations": [{"id": "U42"}, {"id": "U43"}, {"id": "U44"}]}
            model_dir = root / "output" / "yolov8n"
            fixture, _ = graph_fixture("yolov8n")
            onnx_file = root / "fake.onnx"

            def fake_exporter(_staged, destination):
                destination.write_bytes(b"onnx")
                return {"effective_arguments": dict(graph.EXPORT_ARGUMENTS)}

            def fake_loader(_path):
                return fixture, {"outputs": [{"name": "output0", "shape": [1, 7, 8400]}]}

            probe = {"status": "verified", "head_output_evidence": accepted_model["head_output_evidence"]}
            with patch.object(graph.readiness, "inspect_frozen_model", return_value=probe), \
                 patch.object(graph, "environment_evidence", return_value={"git_commit": "test", "producer_packages": {}}), \
                 patch.object(graph, "producer_source_evidence", return_value={}), \
                 patch.object(graph, "git_head", return_value="test-child-commit"):
                record = graph.prepare_model(root, config, accepted, current, "yolov8n", model_dir, fake_exporter, fake_loader)
            self.assertEqual(record["status"], "completed")
            self.assertTrue((model_dir / "private/frozen_source.pt").is_file())
            self.assertTrue((model_dir / "model.onnx").is_file())
            with self.assertRaises(FileExistsError):
                graph.prepare_model(root, config, accepted, current, "yolov8n", model_dir, fake_exporter, fake_loader)

    def test_wrong_config_readiness_and_checkpoint_bindings_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            accepted = {"manifest": {"config_identity": {"sha256": "wrong", "semantic_sha256": "wrong"}}}
            with self.assertRaises(ValueError):
                graph.validate_config_binding(root, accepted, config_path)

            bad_readiness_root = root / "not-the-accepted-root"
            bad_readiness_root.mkdir()
            with self.assertRaises(ValueError):
                graph.validate_readiness_artifact(root, bad_readiness_root)

            checkpoint = root / "frozen.pt"
            checkpoint.write_bytes(b"frozen")
            config = {
                "models": [{
                    "label": "yolov8n",
                    "checkpoint": "frozen.pt",
                    "sha256": "not-the-checkpoint-hash",
                    "bytes": checkpoint.stat().st_size,
                }],
                "calibration_selections": [],
            }
            accepted = {"manifest": {"model_contracts": [{"label": "yolov8n", "expected_sha256": config["models"][0]["sha256"]}]}}
            with self.assertRaises(ValueError):
                graph.validate_current_bindings(root, config, accepted, "yolov8n")


if __name__ == "__main__":
    unittest.main()
