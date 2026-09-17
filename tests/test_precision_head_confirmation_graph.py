from __future__ import annotations

import copy
import importlib.util
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
        nodes[-1]["attributes"] = {"axis": 2, "k": 300}
    else:
        nodes[-1]["outputs"] = [output]
    if label == "yolo26n":
        nodes.extend([
            {"name": f"model.{index}/cv2.0/Conv", "op_type": "Conv", "inputs": ["features"], "outputs": ["inactive_bbox"]},
            {"name": f"model.{index}/cv3.0/Conv", "op_type": "Conv", "inputs": ["features"], "outputs": ["inactive_class"]},
        ])
    return {
        "nodes": nodes,
        "inputs": [{"name": "images", "shape": [1, 3, 640, 640], "dtype": "float32"}],
        "outputs": [{"name": output, "shape": model["head_output_evidence"]["primary_output_shape"], "dtype": "float32"}],
        "tensor_shapes": {
            bbox_tensor: [1, 4, 8400],
            class_tensor: [1, 3, 8400],
            merged: [1, 7, 8400],
            output: model["head_output_evidence"]["primary_output_shape"],
        },
        "tensor_dtypes": {
            "images": "float32",
            "features": "float32",
            bbox_tensor: "float32",
            class_tensor: "float32",
            merged: "float32",
            output: "float32",
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


class FakeDim:
    def __init__(self, value=None, param=""):
        self.dim_value = value
        self.dim_param = param


class FakeTensorProto:
    def __init__(self, dims, data_type, value):
        self.dims = list(dims)
        self.data_type = data_type
        self.value = value


class FakeNumpyHelper:
    @staticmethod
    def to_array(tensor):
        return TensorLike(tensor.dims, tensor.value)


class FakeTensorAttribute:
    def __init__(self, tensor):
        self.type = 4
        self.t = tensor

    def HasField(self, name):
        return name == "t"


def fake_value_info(name, shape, elem_type):
    shape_proto = None if shape is None else SimpleNamespace(dim=[FakeDim(value=item) for item in shape])
    tensor_type = SimpleNamespace(shape=shape_proto, elem_type=elem_type)
    return SimpleNamespace(name=name, type=SimpleNamespace(tensor_type=tensor_type))


class FakeMetadataGraph:
    def __init__(self, *, value_info, initializer, nodes=()):
        self.input = []
        self.output = []
        self.value_info = list(value_info)
        self.initializer = list(initializer)
        self.node = list(nodes)


class PrecisionHeadGraphTests(unittest.TestCase):
    def test_onnx_metadata_preserves_scalar_vector_zero_and_missing_rank_and_conflicts(self):
        scalar = FakeTensorProto([], 7, 3)
        vector = FakeTensorProto([1], 7, [3])
        zero = FakeTensorProto([0], 7, [])
        conflict = FakeTensorProto([3], 7, [1, 2, 3])
        constant = FakeTensorProto([], 7, 3)
        metadata_graph = FakeMetadataGraph(
            value_info=[
                fake_value_info("missing_rank", None, 7),
                fake_value_info("conflict", [2], 1),
            ],
            initializer=[
                SimpleNamespace(name="scalar", **scalar.__dict__),
                SimpleNamespace(name="vector", **vector.__dict__),
                SimpleNamespace(name="zero", **zero.__dict__),
                SimpleNamespace(name="conflict", **conflict.__dict__),
            ],
            nodes=[SimpleNamespace(
                op_type="Constant",
                output=["constant_scalar"],
                attribute=[FakeTensorAttribute(constant)],
            )],
        )
        tensor_metadata, initializer_metadata, initializers, conflicts = graph._collect_onnx_tensor_metadata(
            metadata_graph, FakeNumpyHelper
        )
        self.assertEqual(tensor_metadata["scalar"]["shape"], [])
        self.assertEqual(tensor_metadata["vector"]["shape"], [1])
        self.assertEqual(tensor_metadata["zero"]["shape"], [0])
        self.assertEqual(tensor_metadata["missing_rank"]["shape_status"], "missing_rank")
        self.assertEqual(tensor_metadata["conflict"]["shape_status"], "conflict")
        self.assertEqual(tensor_metadata["conflict"]["dtype_status"], "conflict")
        self.assertEqual(initializer_metadata["constant_scalar"]["shape"], [])
        self.assertEqual(initializer_metadata["constant_scalar"]["dtype"], "int64")
        self.assertEqual(initializers["scalar"], 3)
        self.assertEqual(tensor_metadata["scalar"]["dtype"], "int64")
        self.assertEqual(graph.graph_tensor_shapes({"tensor_metadata": tensor_metadata})["scalar"], [])
        self.assertEqual(graph.graph_tensor_shapes({"tensor_metadata": tensor_metadata})["zero"], [0])
        self.assertNotIn("missing_rank", graph.graph_tensor_shapes({"tensor_metadata": tensor_metadata}))
        self.assertGreaterEqual(len(conflicts), 2)

    def test_real_onnx_loader_records_scalar_initializer_metadata_when_available(self):
        if importlib.util.find_spec("onnx") is None:
            self.skipTest("onnx is not installed in the local measurement environment")
        import onnx
        from onnx import TensorProto, helper

        graph_proto = helper.make_graph(
            [helper.make_node("Mod", ["indices", "divisor"], ["mod_out"], fmod=0)],
            "scalar-mod",
            [helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 300])],
            [helper.make_tensor_value_info("mod_out", TensorProto.INT64, [1, 300])],
            initializer=[helper.make_tensor("divisor", TensorProto.INT64, [], [3])],
        )
        model = helper.make_model(graph_proto, opset_imports=[helper.make_operatorsetid("", 17)])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scalar_mod.onnx"
            onnx.save(model, path)
            graph_doc, _ = graph._load_onnx(path)
        self.assertEqual(graph_doc["initializer_metadata"]["divisor"]["shape"], [])
        self.assertEqual(graph_doc["initializer_metadata"]["divisor"]["dtype"], "int64")
        self.assertEqual(graph_doc["tensor_shapes"]["divisor"], [])
        self.assertEqual(graph_doc["tensor_dtypes"]["divisor"], "int64")

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

    def test_graph_contract_rejects_reversed_spans_and_non_primary_or_ambiguous_outputs(self):
        fixture, model = graph_fixture("yolov8n")
        reversed_inputs = copy.deepcopy(fixture)
        reversed_inputs["nodes"][3]["inputs"] = ["class_features", "bbox_features"]
        result = graph.audit_graph_mapping(reversed_inputs, "yolov8n", model)
        self.assertEqual(result["mapping_status"], "mapping_unresolved")
        self.assertIn("branch_owned_channel_merge_unresolved", result["errors"])

        debug_only = copy.deepcopy(fixture)
        debug_only["nodes"][3]["outputs"] = ["debug_output"]
        debug_only["outputs"] = [{"name": "output0", "shape": [1, 7, 8400], "dtype": "float32"}]
        debug_only["tensor_shapes"]["debug_output"] = [1, 7, 8400]
        debug_only["nodes"].append({"name": "unrelated", "op_type": "Identity", "inputs": ["unrelated_input"], "outputs": ["output0"]})
        result = graph.audit_graph_mapping(debug_only, "yolov8n", model)
        self.assertEqual(result["mapping_status"], "mapping_unresolved")
        self.assertTrue(any("unreachable_conv" in error or "merge_unresolved" in error for error in result["errors"]))

        multiple = copy.deepcopy(fixture)
        multiple["outputs"].append({"name": "debug_same_shape", "shape": [1, 7, 8400], "dtype": "float32"})
        multiple["tensor_shapes"]["debug_same_shape"] = [1, 7, 8400]
        result = graph.audit_graph_mapping(multiple, "yolov8n", model)
        self.assertEqual(result["mapping_status"], "mapping_unresolved")
        self.assertIn("primary_output_must_be_unique", " ".join(result["errors"]))

        wrong_input = copy.deepcopy(fixture)
        wrong_input["inputs"][0] = {"name": "images", "shape": [1, 3, 320, 320], "dtype": "float16"}
        result = graph.audit_graph_mapping(wrong_input, "yolov8n", model)
        self.assertEqual(result["mapping_status"], "mapping_unresolved")
        self.assertIn("input_schema_mismatch", " ".join(result["errors"]))

    def test_pinned_ultralytics_wrapper_alias_maps_real_yolov8_conv_name(self):
        fixture, model = graph_fixture("yolov8n")
        model["active_convolution_mapping"]["cv2"] = [{"name": "model.22.cv2.0.0.conv", "module_type": "Conv2d"}]
        model["active_convolution_mapping"]["cv3"] = [{"name": "model.22.cv3.0.0.conv", "module_type": "Conv2d"}]
        fixture["nodes"][0]["name"] = "/model.22/cv2.0/cv2.0.0/conv/Conv"
        fixture["nodes"][1]["name"] = "/model.22/cv3.0/cv3.0.0/conv/Conv"
        result = graph.audit_graph_mapping(fixture, "yolov8n", model)
        self.assertEqual(result["mapping_status"], "verified", result["errors"])
        self.assertEqual(result["active_branch_audit"]["cv2"]["matched_convolutions"][0]["match"], "explicit_ultralytics_exporter_wrapper_alias")
        self.assertEqual(result["active_branch_audit"]["cv3"]["matched_convolutions"][0]["match"], "explicit_ultralytics_exporter_wrapper_alias")
        self.assertEqual(graph._exporter_wrapper_aliases("model.22.cv2.0.0.conv"), {"model.22.cv2.0.cv2.0.0.conv"})

    def test_pinned_ultralytics_wrapper_alias_covers_yolo26_one2one_branches(self):
        fixture, model = graph_fixture("yolo26n", shared_downstream=True)
        model["active_convolution_mapping"]["one2one_cv2"] = [{"name": "model.23.one2one_cv2.0.0.conv", "module_type": "Conv2d"}]
        model["active_convolution_mapping"]["one2one_cv3"] = [{"name": "model.23.one2one_cv3.0.0.0.conv", "module_type": "Conv2d"}]
        fixture["nodes"][0]["name"] = "/model.23/one2one_cv2.0/one2one_cv2.0.0/conv/Conv"
        fixture["nodes"][1]["name"] = "/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv"
        result = graph.audit_graph_mapping(fixture, "yolo26n", model)
        self.assertEqual(result["mapping_status"], "verified", result["errors"])
        self.assertEqual(result["active_branch_audit"]["one2one_cv2"]["matched_convolutions"][0]["match"], "explicit_ultralytics_exporter_wrapper_alias")
        self.assertEqual(result["active_branch_audit"]["one2one_cv3"]["matched_convolutions"][0]["match"], "explicit_ultralytics_exporter_wrapper_alias")
        self.assertEqual(
            graph._exporter_wrapper_aliases("model.23.one2one_cv3.0.0.0.conv"),
            {
                "model.23.one2one_cv3.0.one2one_cv3.0.0.0.conv",
                "model.23.one2one_cv3.0.one2one_cv3.0.0.one2one_cv3.0.0.0.conv",
            },
        )
        self.assertIn(
            "model.23.one2one_cv3.0.one2one_cv3.0.2",
            graph._exporter_wrapper_aliases("model.23.one2one_cv3.0.2"),
        )
        for scale in range(3):
            for block in range(2):
                for leaf in range(2):
                    source = f"model.23.one2one_cv3.{scale}.{block}.{leaf}.conv"
                    expected_nested = (
                        f"model.23.one2one_cv3.{scale}.one2one_cv3.{scale}.{block}."
                        f"one2one_cv3.{scale}.{block}.{leaf}.conv"
                    )
                    self.assertIn(expected_nested, graph._exporter_wrapper_aliases(source))
            terminal_source = f"model.23.one2one_cv3.{scale}.2"
            terminal_alias = f"model.23.one2one_cv3.{scale}.one2one_cv3.{scale}.2"
            self.assertIn(terminal_alias, graph._exporter_wrapper_aliases(terminal_source))
        self.assertEqual(graph._exporter_wrapper_aliases("model.23.one2one_cv3.bad.0.0.conv"), set())
        self.assertEqual(graph._exporter_wrapper_aliases("model.23.one2one_cv3.0.bad.0.conv"), set())

    def test_yolo26_alias_duplicate_is_ambiguous_and_inactive_branches_stay_excluded(self):
        fixture, model = graph_fixture("yolo26n")
        model["active_convolution_mapping"]["one2one_cv3"] = [{"name": "model.23.one2one_cv3.1.0.0.conv", "module_type": "Conv2d"}]
        fixture["nodes"][1]["name"] = "/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/conv/Conv"
        duplicate = copy.deepcopy(fixture["nodes"][1])
        duplicate["outputs"] = ["duplicate_class_features"]
        fixture["nodes"].append(duplicate)
        fixture["tensor_shapes"]["duplicate_class_features"] = [1, 3, 8400]
        result = graph.audit_graph_mapping(fixture, "yolo26n", model)
        self.assertEqual(result["mapping_status"], "mapping_unresolved")
        self.assertTrue(any("one2one_cv3:source_conv_match_count" in error for error in result["errors"]))
        self.assertTrue(result["inactive_branch_audit"]["cv2"]["excluded_from_precision_targets"])

    def test_yolo26_pinned_end2end_postprocess_ops_require_explicit_semantics(self):
        fixture, model = graph_fixture("yolo26n")
        merge_node = next(node for node in fixture["nodes"] if node["op_type"] == "Concat")
        merge_node["outputs"] = ["merged_head"]
        fixture["nodes"].extend([
            {
                "name": "model.23/Transpose",
                "op_type": "Transpose",
                "inputs": ["merged_head"],
                "outputs": ["transposed"],
                "attributes": {"perm": [0, 2, 1]},
            },
            {
                "name": "model.23/Split",
                "op_type": "Split",
                "inputs": ["transposed"],
                "outputs": ["split_boxes", "split_scores"],
                "attributes": {"axis": 2, "split": [4, 3]},
            },
            {
                "name": "model.23/ReduceMax",
                "op_type": "ReduceMax",
                "inputs": ["split_scores"],
                "outputs": ["max_scores"],
                "attributes": {"axes": [2], "keepdims": 1},
            },
            {
                "name": "model.23/Flatten",
                "op_type": "Flatten",
                "inputs": ["split_boxes"],
                "outputs": ["flat_boxes"],
                "attributes": {"axis": 2},
            },
            {
                "name": "model.23/Unsqueeze",
                "op_type": "Unsqueeze",
                "inputs": ["max_scores"],
                "outputs": ["expanded_scores"],
                "attributes": {"axes": [2]},
            },
            {
                "name": "model.23/Tile",
                "op_type": "Tile",
                "inputs": ["max_scores", "tile_repeats"],
                "outputs": ["tiled_scores"],
            },
            {
                "name": "model.23/Mod",
                "op_type": "Mod",
                "inputs": ["max_scores", "mod_divisor"],
                "outputs": ["mod_scores"],
                "attributes": {"fmod": 0},
            },
            {
                "name": "model.23/TopK",
                "op_type": "TopK",
                "inputs": ["max_scores"],
                "outputs": ["top_values", "top_indices"],
                "attributes": {"axis": 1, "k": 300},
            },
            {
                "name": "model.23/Concat_6",
                "op_type": "Concat",
                "inputs": ["top_values", "top_indices"],
                "outputs": ["output0"],
                "attributes": {"axis": 2},
            },
        ])
        fixture["tensor_shapes"].update({
            "transposed": [1, 8400, 7],
            "split_boxes": [1, 8400, 4],
            "split_scores": [1, 8400, 3],
            "max_scores": [1, 8400, 1],
            "flat_boxes": [8400, 4],
            "expanded_scores": [1, 8400, 1, 1],
            "tile_repeats": [3],
            "tiled_scores": [3, 8400, 1],
            "mod_divisor": [],
            "mod_scores": [1, 8400, 1],
            "top_values": [1, 300, 1],
            "top_indices": [1, 300, 1],
        })
        fixture["tensor_dtypes"].update({
            "max_scores": "int64",
            "mod_divisor": "int64",
            "mod_scores": "int64",
        })
        fixture["initializers"] = {"tile_repeats": [3, 1, 1], "mod_divisor": [3]}
        result = graph.audit_graph_mapping(fixture, "yolo26n", model)
        self.assertEqual(result["mapping_status"], "verified", result["errors"])
        audited_ops = {row["op_type"] for row in result["branch_merge"]["downstream_semantic_audit"]["visited_nodes"]}
        self.assertTrue({"Split", "ReduceMax", "Flatten", "Unsqueeze", "Tile", "Mod"}.issubset(audited_ops))

        mod_semantics = next(
            row for row in result["branch_merge"]["downstream_semantic_audit"]["visited_nodes"]
            if row["op_type"] == "Mod"
        )
        self.assertEqual(mod_semantics["divisor"]["shape"], [])
        self.assertEqual(mod_semantics["divisor"]["dtype"], "int64")
        self.assertEqual(mod_semantics["divisor"]["value"], [3])
        self.assertEqual(mod_semantics["expected_class_count"], 3)

        singleton = copy.deepcopy(fixture)
        singleton["tensor_shapes"]["mod_divisor"] = [1]
        singleton_result = graph.audit_graph_mapping(singleton, "yolo26n", model)
        self.assertEqual(singleton_result["mapping_status"], "verified", singleton_result["errors"])

        invalid = copy.deepcopy(fixture)
        split = next(node for node in invalid["nodes"] if node["op_type"] == "Split")
        split["attributes"].pop("split")
        invalid_result = graph.audit_graph_mapping(invalid, "yolo26n", model)
        self.assertEqual(invalid_result["mapping_status"], "mapping_unresolved")
        self.assertIn("split_semantics_unresolved", " ".join(invalid_result["errors"]))

        for divisor, divisor_shape, divisor_dtype in ((0, [], "int64"), (3, None, "int64"), (3, [], "float32")):
            invalid_mod = copy.deepcopy(fixture)
            invalid_mod["initializers"]["mod_divisor"] = [divisor]
            if divisor_shape is None:
                invalid_mod["tensor_shapes"].pop("mod_divisor", None)
            else:
                invalid_mod["tensor_shapes"]["mod_divisor"] = divisor_shape
            invalid_mod["tensor_dtypes"]["mod_divisor"] = divisor_dtype
            invalid_mod_result = graph.audit_graph_mapping(invalid_mod, "yolo26n", model)
            self.assertEqual(invalid_mod_result["mapping_status"], "mapping_unresolved")
            self.assertIn("mod_semantics_unresolved", " ".join(invalid_mod_result["errors"]))

        unsupported_mode = copy.deepcopy(fixture)
        mod_node = next(node for node in unsupported_mode["nodes"] if node["op_type"] == "Mod")
        mod_node["attributes"]["fmod"] = 1
        unsupported_result = graph.audit_graph_mapping(unsupported_mode, "yolo26n", model)
        self.assertEqual(unsupported_result["mapping_status"], "mapping_unresolved")
        self.assertIn("mod_semantics_unresolved", " ".join(unsupported_result["errors"]))

    def test_precision_targets_require_verified_mapping_and_consistent_merge(self):
        fixture, model = graph_fixture("yolov8n")
        mapping = graph.audit_graph_mapping(fixture, "yolov8n", model)
        unresolved = copy.deepcopy(mapping)
        unresolved["status"] = "mapping_unresolved"
        with self.assertRaises(ValueError):
            graph.precision_target_sets(unresolved, "bbox_fp32")
        inconsistent = copy.deepcopy(mapping)
        inconsistent["branch_owned_target_sets"]["classification"] = inconsistent["branch_owned_target_sets"]["bbox"]
        with self.assertRaises(ValueError):
            graph.precision_target_sets(inconsistent, "both_fp32")

    def test_unknown_post_merge_semantics_are_unresolved_even_with_matching_shape(self):
        fixture, model = graph_fixture("yolov8n", shared_downstream=True)
        fixture["nodes"][-1]["op_type"] = "MysterySelection"
        result = graph.audit_graph_mapping(fixture, "yolov8n", model)
        self.assertEqual(result["mapping_status"], "mapping_unresolved")
        self.assertTrue(any("unknown_post_merge_semantics" in error for error in result["errors"]))

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
        v8_result = graph.validate_native_output("yolov8n", (TensorLike([1, 7, 8400]), {"boxes": object(), "scores": object(), "feats": object()}), v8)
        self.assertFalse(v8_result["nms_applied"])
        self.assertEqual(v8_result["box_channels"], [0, 4])

        v26 = expected_model("yolo26n")["head_output_evidence"]
        rows = [[10.0, 20.0, 30.0, 40.0, 0.5, 1.0] for _ in range(300)]
        v26_result = graph.validate_native_output("yolo26n", (TensorLike([1, 300, 6], rows), {"one2one": {}, "one2many": {}}), v26)
        self.assertFalse(v26_result["double_nms"])
        self.assertEqual(v26_result["detection_columns"]["class_id"], 5)

        with self.assertRaises(ValueError):
            graph.validate_native_output("yolov8n", [], v8)
        with self.assertRaises(ValueError):
            graph.validate_native_output("yolo26n", (TensorLike([1, 300, 6], [[0, 0, 1, 1, 0.5, 3]] * 300), {"one2one": {}, "one2many": {}}), v26)
        with self.assertRaises(ValueError):
            graph.validate_native_output("yolo26n", (TensorLike([1, 300, 6], [[0, 0, 1, 1, float("inf"), 1]] * 300), {"one2one": {}, "one2many": {}}), v26)
        with self.assertRaises(ValueError):
            graph.validate_native_output("yolo26n", (TensorLike([1, 300, 6], [[0, 0, 1, 1, 0.5, 1.5]] * 300), {"one2one": {}, "one2many": {}}), v26)
        with self.assertRaises(ValueError):
            graph.validate_native_output("yolo26n", (TensorLike([1, 300, 6], [[30, 20, 10, 40, 0.5, 1]] * 300), {"one2one": {}, "one2many": {}}), v26)
        with self.assertRaises(ValueError):
            graph.validate_native_output("yolo26n", (TensorLike([1, 300, 6], []), {"one2one": {}, "one2many": {}}), v26)
        wrong_contract = copy.deepcopy(v8)
        wrong_contract["end2end"] = True
        with self.assertRaises(ValueError):
            graph.validate_native_output("yolov8n", (TensorLike([1, 7, 8400]), {"boxes": object(), "scores": object(), "feats": object()}), wrong_contract)

    def test_adapter_contract_is_deferred_until_real_parser_and_forward_evidence(self):
        for label in ("yolov8n", "yolo26n"):
            contract = expected_model(label)["head_output_evidence"]
            adapter = graph.adapter_contract(label, contract)
            self.assertTrue(adapter["double_nms_forbidden"])
            self.assertTrue(adapter["scored_adapter_status"].startswith("deferred"))
            self.assertFalse(adapter["observed_native_numeric_validation"])

    def test_environment_schema_is_accepted_by_real_readiness_compatibility_boundary(self):
        environment = graph.environment_evidence()
        runtime = {"ultralytics": "8.4.102", "torch": None, "numpy": None, "pycocotools": None}
        compatibility = readiness.runtime_compatibility_evidence(environment, runtime)
        self.assertEqual(environment["ultralytics"], environment["producer_packages"]["ultralytics"]["version"])
        self.assertEqual(compatibility["status"], "cpu_probe_supported", compatibility)
        self.assertTrue(compatibility["native_head_probe_supported"])

    def test_dependency_preflight_blocks_wrong_or_missing_locked_prerequisites(self):
        def fake_version(name):
            if name == "ultralytics":
                return "wrong-version"
            if name == "onnxruntime":
                return "1.0.0"
            raise graph.metadata.PackageNotFoundError(name)

        with patch.object(graph.metadata, "version", side_effect=fake_version):
            result = graph.preflight_producer_dependencies({"runtime": {"ultralytics": "8.4.102"}})
        self.assertEqual(result["status"], "unresolved")
        self.assertIn("onnx", result["missing"])
        self.assertTrue(any(row["package"] == "ultralytics" for row in result["mismatches"]))
        self.assertFalse(result["network_or_install_mutation"])

    def test_accepted_binding_snapshot_rejects_same_shape_source_substitution(self):
        dataset = {
            "status": "verified", "dev_image_ids": ["a"],
            "inventory": [{"image": "a.jpg", "image_sha256": "image-a", "label_sha256": "label-a"}],
            "inventory_sha256": "inventory-a", "train_image_ids": ["train-a"],
            "train_inventory": {"status": "verified", "image_ids_sha256": "train-a"},
            "test_image_ids": ["test-a"], "test_exclusion_inventory": {"status": "verified"},
            "train_dev_overlap": [], "dev_test_overlap": [],
        }
        calibrations = [{"id": "U42", "manifest_contract": {"status": "canonical"}, "selection_audit": {"selected_ids": ["train-a"], "source_bytes": [{"image_sha256": "source-a"}]}, "materialization": {"status": "complete", "image_bytes": [{"image_sha256": "material-a"}]}}]
        self.assertEqual(graph.compare_current_to_accepted_snapshot(dataset, calibrations, copy.deepcopy(dataset), copy.deepcopy(calibrations))["status"], "matched")
        changed = copy.deepcopy(calibrations)
        changed[0]["materialization"]["image_bytes"][0]["image_sha256"] = "substituted"
        comparison = graph.compare_current_to_accepted_snapshot(dataset, changed, dataset, calibrations)
        self.assertEqual(comparison["status"], "mismatch")
        self.assertTrue(any("calibrations.U42" in row["path"] for row in comparison["differences"]))

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
        child_env = graph.cpu_child_environment()
        self.assertEqual(child_env["YOLO_AUTOINSTALL"], "0")
        self.assertEqual(child_env["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"], "1")
        self.assertEqual(child_env["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(child_env["OMP_NUM_THREADS"], "2")
        self.assertEqual(child_env["MKL_NUM_THREADS"], "2")

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
            child_envs = []

            def fake_run(command, **kwargs):
                calls.append(command)
                child_envs.append(kwargs.get("env", {}))
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
            self.assertEqual([env["CUDA_VISIBLE_DEVICES"] for env in child_envs], ["-1", "-1"])
            self.assertEqual([env["YOLO_AUTOINSTALL"] for env in child_envs], ["0", "0"])
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
            config = {"models": [model_config], "runtime": {"ultralytics": "8.4.102"}, "export_contract": {}, "calibration_recipe": {"preprocessing_helper": "helper.py"}}
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
                export_calls.append(destination)
                destination.write_bytes(b"onnx")
                return {"effective_arguments": dict(graph.EXPORT_ARGUMENTS)}

            def fake_loader(_path):
                return fixture, {"outputs": [{"name": "output0", "shape": [1, 7, 8400]}]}

            expected_probe = {"status": "verified", "head_output_evidence": accepted_model["head_output_evidence"]}
            real_environment = graph.environment_evidence()

            def real_boundary_probe(_repo, _model, *, probe, runtime_contract, observed_environment):
                compatibility = readiness.runtime_compatibility_evidence(observed_environment, runtime_contract)
                self.assertEqual(compatibility["status"], "cpu_probe_supported", compatibility)
                return {**expected_probe, "head_output_evidence": accepted_model["head_output_evidence"]}

            export_calls = []
            with patch.object(graph.readiness, "inspect_frozen_model", side_effect=real_boundary_probe), \
                 patch.object(graph, "environment_evidence", return_value=real_environment), \
                 patch.object(graph, "producer_source_evidence", return_value={}), \
                 patch.object(graph, "git_head", return_value="test-child-commit"):
                with self.assertRaises(ValueError):
                    graph.prepare_model(root, config, accepted, current, "yolov8n", root / "output_fail" / "yolov8n", fake_exporter, fake_loader, {"status": "unresolved", "missing": ["onnx"]})
                self.assertEqual(export_calls, [])
                record = graph.prepare_model(root, config, accepted, current, "yolov8n", model_dir, fake_exporter, fake_loader, {"status": "verified"})
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
