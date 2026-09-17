import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import analyze_precision_head_numeric_localization as localization  # noqa: E402

try:
    import torch
    from ultralytics.nn.modules.head import Detect
except ImportError:
    torch = None
    Detect = None


def stable_topk(values, k):
    return localization.stable_topk_numpy(values, k)


def v26_capture(drift=False):
    boxes = np.zeros((1, 4, 300), dtype=np.float32)
    for index in range(300):
        boxes[0, :, index] = [index, index + 1, index + 2, index + 3]
    scores = np.zeros((1, 3, 300), dtype=np.float32)
    for index in range(300):
        scores[0, :, index] = [0.001 * index, 0.002 * index + 0.1, 0.003 * index + 0.2]
    native = localization.reconstruct_yolo26_selection(boxes.transpose(0, 2, 1), scores.transpose(0, 2, 1), topk_impl=stable_topk)
    merged = np.concatenate((boxes, scores), axis=1)
    derived = {
        "output0": native["primary"].copy(),
        localization.V26_INTERNAL_OUTPUTS["preselection_merged"]: merged,
        localization.V26_INTERNAL_OUTPUTS["preselection_transposed"]: merged.transpose(0, 2, 1),
        localization.V26_INTERNAL_OUTPUTS["preselection_boxes"]: boxes,
        localization.V26_INTERNAL_OUTPUTS["preselection_scores"]: scores,
        localization.V26_INTERNAL_OUTPUTS["stage1_values"]: native["stage1_values"],
        localization.V26_INTERNAL_OUTPUTS["stage1_indices"]: native["stage1_indices"],
        localization.V26_INTERNAL_OUTPUTS["stage2_values"]: native["stage2_values"],
        localization.V26_INTERNAL_OUTPUTS["stage2_indices"]: native["stage2_flat_indices"],
        localization.V26_INTERNAL_OUTPUTS["selected_class_scores"]: np.take_along_axis(scores.transpose(0, 2, 1), native["original_anchor"][..., None], axis=1),
        localization.V26_INTERNAL_OUTPUTS["selected_anchor_rank"]: native["anchor_rank"],
        localization.V26_INTERNAL_OUTPUTS["selected_class_id"]: native["class_id"],
        localization.V26_INTERNAL_OUTPUTS["selected_anchor_indices"]: native["original_anchor"][..., None],
        localization.V26_INTERNAL_OUTPUTS["selected_boxes"]: native["primary"][..., :4],
    }
    if drift:
        derived["output0"] = derived["output0"].copy()
        derived["output0"][0, 0, 0] += 0.25
    return boxes.transpose(0, 2, 1), scores.transpose(0, 2, 1), native["primary"], {"output0": native["primary"]}, derived


def real_detect(reg_max, end2end):
    head = Detect(nc=3, reg_max=reg_max, end2end=end2end, ch=(16, 32, 64))
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    head.export = False
    head.dynamic = False
    head.agnostic_nms = False
    head.max_det = 300
    if reg_max == 1:
        head.dfl = torch.nn.Identity()
    return head.float().eval()


def real_payload(reg_max):
    feats = [torch.full((1, 16, 80, 80), 0.25), torch.full((1, 32, 40, 40), 0.5), torch.full((1, 64, 20, 20), 0.75)]
    boxes = torch.zeros((1, 4 * reg_max, 8400), dtype=torch.float32)
    if reg_max > 1:
        for side in range(4):
            boxes[:, side * reg_max + side + 1, :] = 2.0
            boxes[:, side * reg_max + side + 5, :] = -0.5
    else:
        boxes[:] = torch.linspace(-2.0, 2.0, 8400, dtype=torch.float32).reshape(1, 1, 8400)
    scores = torch.linspace(-6.0, 6.0, 3 * 8400, dtype=torch.float32).reshape(1, 3, 8400)
    return {"boxes": boxes, "scores": scores, "feats": feats}


class PrecisionHeadLocalizationTests(unittest.TestCase):
    @unittest.skipUnless(Detect is not None and torch is not None, "pinned Ultralytics CPU producer environment is unavailable")
    def test_real_yolov8_decoder_primary_and_raw_roles(self):
        head = real_detect(reg_max=16, end2end=False)
        payload = real_payload(reg_max=16)
        with torch.no_grad():
            primary = head._inference(payload)
        state = localization.new_child_state("yolov8n")
        boxes, scores, meta = localization.extract_native_semantic_preselection((primary, payload), "yolov8n", primary, head, torch, np, state)
        self.assertEqual(tuple(boxes.shape), (1, 8400, 4))
        self.assertEqual(tuple(scores.shape), (1, 8400, 3))
        self.assertTrue(np.isfinite(boxes).all())
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreaterEqual(float(scores.min()), 0.0)
        self.assertLessEqual(float(scores.max()), 1.0)
        self.assertEqual(meta["raw_debug"]["raw_box_parameters"]["shape"], [1, 64, 8400])
        self.assertEqual(meta["raw_debug"]["raw_roles"]["scores"], "class logits before installed sigmoid")
        self.assertEqual(meta["box_format"], "xywh")
        self.assertEqual(state["forward_counts"]["native_decoder_replays"]["completed"], 0)

    @unittest.skipUnless(Detect is not None and torch is not None, "pinned Ultralytics CPU producer environment is unavailable")
    def test_real_yolo26_decoder_replay_and_postprocess_are_exact(self):
        head = real_detect(reg_max=1, end2end=True)
        payload = real_payload(reg_max=1)
        with torch.no_grad():
            primary = head.postprocess(head._inference(payload).permute(0, 2, 1))
        state = localization.new_child_state("yolo26n")
        boxes, scores, meta = localization.extract_native_semantic_preselection((primary, {"one2one": payload}), "yolo26n", primary, head, torch, np, state)
        self.assertEqual(tuple(boxes.shape), (1, 8400, 4))
        self.assertEqual(tuple(scores.shape), (1, 8400, 3))
        self.assertGreaterEqual(float(scores.min()), 0.0)
        self.assertLessEqual(float(scores.max()), 1.0)
        self.assertEqual(meta["raw_debug"]["raw_box_parameters"]["shape"], [1, 4, 8400])
        self.assertEqual(meta["box_format"], "xyxy")
        self.assertTrue(meta["postprocess_replay"]["exact_primary_match"])
        self.assertEqual(state["forward_counts"]["native_decoder_replays"]["completed"], 1)
        self.assertEqual(state["forward_counts"]["native_postprocess_replays"]["completed"], 1)
        self.assertEqual(meta["decoder_replay"]["state_before"], meta["decoder_replay"]["state_after"])

    @unittest.skipUnless(Detect is not None and torch is not None, "pinned Ultralytics CPU producer environment is unavailable")
    def test_real_decoder_rejects_wrong_coordinate_or_agnostic_flags(self):
        head = real_detect(reg_max=16, end2end=False)
        payload = real_payload(reg_max=16)
        with torch.no_grad():
            primary = head._inference(payload)
        head.xyxy = True
        with self.assertRaises(localization.LocalizationUnresolved):
            localization.extract_native_semantic_preselection((primary, payload), "yolov8n", primary, head, torch, np, localization.new_child_state("yolov8n"))
        head = real_detect(reg_max=1, end2end=True)
        payload = real_payload(reg_max=1)
        with torch.no_grad():
            primary = head.postprocess(head._inference(payload).permute(0, 2, 1))
        head.agnostic_nms = True
        with self.assertRaises(localization.LocalizationUnresolved):
            localization.extract_native_semantic_preselection((primary, {"one2one": payload}), "yolo26n", primary, head, torch, np, localization.new_child_state("yolo26n"))

    @unittest.skipUnless(Detect is not None and torch is not None, "pinned Ultralytics CPU producer environment is unavailable")
    def test_decoder_replay_drift_suppresses_semantic_acceptance(self):
        head = real_detect(reg_max=1, end2end=True)
        payload = real_payload(reg_max=1)
        original = head._inference
        with torch.no_grad():
            primary = head.postprocess(original(payload).permute(0, 2, 1))
        calls = {"count": 0}

        def drift(branch):
            with torch.no_grad():
                result = original(branch)
            calls["count"] += 1
            if calls["count"] == 1:
                result = result.clone()
                result[0, 0, :] += 1.0
            return result

        head._inference = drift
        with self.assertRaises(localization.LocalizationUnresolved):
            localization.extract_native_semantic_preselection((primary, {"one2one": payload}), "yolo26n", primary, head, torch, np, localization.new_child_state("yolo26n"))

    @unittest.skipUnless(Detect is not None and torch is not None, "pinned Ultralytics CPU producer environment is unavailable")
    def test_real_decoder_rejects_missing_features_nonfinite_raw_and_double_sigmoid(self):
        head = real_detect(reg_max=16, end2end=False)
        payload = real_payload(reg_max=16)
        with torch.no_grad():
            primary = head._inference(payload)
        missing_feats = dict(payload)
        del missing_feats["feats"]
        with self.assertRaises(localization.LocalizationUnresolved):
            localization.extract_native_semantic_preselection((primary, missing_feats), "yolov8n", primary, head, torch, np, localization.new_child_state("yolov8n"))

        nonfinite = dict(payload)
        nonfinite["scores"] = payload["scores"].clone()
        nonfinite["scores"][0, 0, 0] = float("nan")
        with self.assertRaises(localization.LocalizationUnresolved):
            localization.extract_native_semantic_preselection((primary, nonfinite), "yolov8n", primary, head, torch, np, localization.new_child_state("yolov8n"))

        head = real_detect(reg_max=1, end2end=True)
        payload = real_payload(reg_max=1)
        original = head._inference
        with torch.no_grad():
            primary = head.postprocess(original(payload).permute(0, 2, 1))

        def double_sigmoid(branch):
            with torch.no_grad():
                result = original(branch).clone()
            result[:, 4:7, :] = torch.sigmoid(result[:, 4:7, :])
            return result

        head._inference = double_sigmoid
        state = localization.new_child_state("yolo26n")
        with self.assertRaises(localization.LocalizationUnresolved):
            localization.extract_native_semantic_preselection((primary, {"one2one": payload}), "yolo26n", primary, head, torch, np, state)
        self.assertIn("raw_debug", state["semantic_adapter"])

    def test_wrong_coordinate_metadata_is_rejected_before_cross_side_analysis(self):
        metadata = {"box_format": "xyxy", "coordinate_space": "640x640 input pixels", "score_semantics": "native primary post-sigmoid class probabilities", "semantic_admissibility": {"status": "native_self_consistency_pass"}}
        with self.assertRaises(localization.LocalizationUnresolved):
            localization.validate_native_semantic_metadata(metadata, "yolov8n")

    def test_v2_runner_root_preserves_v1_and_strict_verdict(self):
        self.assertEqual(localization.STUDY, "precision_head_numeric_localization_v2")
        self.assertEqual(localization.DEFAULT_OUTPUT.as_posix(), "results/measurement_audit_v1/precision_head_numeric_localization_v2")
        self.assertEqual(localization.NUMERIC_V1_ROOT.as_posix(), "results/measurement_audit_v1/precision_head_confirmation_numeric_v1")

    def test_v8_offender_scalar_allowance_and_new_mismatch_count(self):
        reference = np.zeros((1, 7, 8400), dtype=np.float32)
        observed = reference.copy()
        reference[0, 1, 8004] = 0.1
        observed[0, 1, 8004] = 0.1005493
        reference[0, 1, 8014] = 0.2
        observed[0, 1, 8014] = 0.2005493
        result = localization.v8_offender_report(reference, observed)
        self.assertEqual(result["strict_rerun_comparison"]["status"], "fail")
        self.assertEqual(result["all_new_mismatches"]["box_mismatch_count"], 2)
        self.assertEqual(result["all_new_mismatches"]["score_mismatch_count"], 0)
        self.assertEqual(result["historical_offender_indices"][0]["box"]["status"], "fail")
        self.assertEqual(result["historical_offender_indices"][0]["anchor_class_scores"][0]["status"], "pass")

    def test_near_zero_reference_is_json_safe(self):
        result = localization.allowed_error(0.0, 1.0)
        self.assertGreater(result["error_allowance_ratio"], 1e4)
        self.assertEqual(result["status"], "fail")
        relative = localization.numeric.compare_float_arrays(np.array([0.0], dtype=np.float32), np.array([1e-4], dtype=np.float32), np, "near_zero")
        self.assertEqual(relative["relative_infinite_count"], 1)
        json.dumps(relative)

    def test_v26_reconstruction_uses_own_tensors_and_class_flattening(self):
        boxes = np.zeros((1, 4, 4), dtype=np.float32)
        scores = np.array([[[0.1, 0.9, 0.2, 0.3], [0.8, 0.1, 0.7, 0.4], [0.2, 0.3, 0.6, 0.5]]], dtype=np.float32).transpose(0, 2, 1)
        result = localization.reconstruct_yolo26_selection(boxes, scores, topk_impl=stable_topk)
        self.assertEqual(result["flattening"]["class_formula"], "flat_index % 3")
        self.assertEqual(result["flattening"]["anchor_rank_formula"], "flat_index // 3")
        self.assertEqual(result["primary"].shape, (1, 12, 6))
        self.assertEqual(result["topk_source"], "injected_topk_for_local_test")

    def test_permutation_only_is_separated_from_membership_change(self):
        native = {"stage1_indices": np.array([[10, 11]]), "stage2_flat_indices": np.array([[0, 4]])}
        permutation = {"stage1_indices": np.array([[10, 11]]), "stage2_flat_indices": np.array([[4, 0]])}
        changed = {"stage1_indices": np.array([[10, 12]]), "stage2_flat_indices": np.array([[0, 4]])}
        perm_result = localization.compare_selection_alignment(native, permutation)
        changed_result = localization.compare_selection_alignment(native, changed)
        self.assertTrue(perm_result["same_selected_set_permutation"])
        self.assertFalse(perm_result["different_selected_set_membership"])
        self.assertTrue(changed_result["different_selected_set_membership"])
        self.assertEqual(changed_result["overlap_count"], 1)

    def test_same_anchor_coordinate_error_and_scores_within_tolerance_are_separate(self):
        native_boxes = np.zeros((1, 4, 3), dtype=np.float32)
        onnx_boxes = native_boxes.copy()
        onnx_boxes[0, 1, 0] = 0.1
        native_scores = np.full((1, 3, 3), 0.5, dtype=np.float32)
        onnx_scores = native_scores.copy()
        onnx_scores[0, 0, 0] += 1e-6
        result = localization.compare_preselection(native_boxes, native_scores, onnx_boxes, onnx_scores)
        self.assertTrue(result["same_anchor_numerical_error"]["boxes"])
        self.assertFalse(result["same_anchor_numerical_error"]["class_probabilities"])

    def test_wrong_index_mapping_fails_closed_without_rematching(self):
        with self.assertRaises(localization.LocalizationUnresolved):
            localization.selection_pairs(np.array([[9]]), np.array([[2]]))
        native = {"stage1_indices": np.array([[2]]), "stage2_flat_indices": np.array([[0]])}
        onnx = {"stage1_indices": np.array([[3]]), "stage2_flat_indices": np.array([[0]])}
        result = localization.compare_selection_alignment(native, onnx)
        self.assertTrue(result["different_selected_set_membership"])
        self.assertEqual(result["association_rule"], "exact (original_anchor,class_id) pair; no nearest-neighbor/rematching")

    def test_instrumented_output_drift_is_flagged(self):
        native_boxes, native_scores, native_primary, original, derived = v26_capture(drift=True)
        graph_contract = {"derived_file": {"path": "private/view.onnx", "sha256": "x", "bytes": 1}, "output_additions": localization.V26_INTERNAL_OUTPUTS}
        result = localization.analyze_v26_case(native_primary, native_boxes, native_scores, original, derived, graph_contract, {"comparison": {"status": "fail"}}, topk_impl=stable_topk)
        self.assertEqual(result["analysis"]["instrumentation_sensitivity"]["status"], "drift")
        self.assertEqual(result["analysis"]["cross_side_admissibility"]["endpoint_status"], "unresolved/not_admissible")
        self.assertEqual(result["analysis"]["same_anchor_comparison"]["status"], "unresolved")
        self.assertEqual(result["analysis"]["selection_alignment"]["same_selected_set"], True)

    def test_v26_localization_analysis_reconstructs_both_sides(self):
        native_boxes, native_scores, native_primary, original, derived = v26_capture()
        graph_contract = {"derived_file": {"path": "private/view.onnx", "sha256": "x", "bytes": 1}, "output_additions": localization.V26_INTERNAL_OUTPUTS}
        result = localization.analyze_v26_case(native_primary, native_boxes, native_scores, original, derived, graph_contract, {"comparison": {"status": "fail"}}, topk_impl=stable_topk)
        self.assertEqual(result["analysis"]["same_anchor_comparison"]["boxes"]["status"], "pass")
        self.assertTrue(result["analysis"]["selection_alignment"]["same_order"])
        self.assertEqual(result["analysis"]["reconstructed_outputs"]["onnx_from_own_tensors_and_direct_indices"]["status"], "pass")
        self.assertEqual(result["analysis"]["reconstructed_outputs"]["onnx_stage2_value_consistency"]["status"], "pass")
        self.assertEqual(result["forward_counts"], {"native_model_forwards": 1, "onnx_original_session_runs": 1, "onnx_derived_session_runs": 1})

    def test_direct_index_arithmetic_mismatch_fails_closed(self):
        _, _, _, _, derived = v26_capture()
        broken = dict(derived)
        anchor_rank_name = localization.V26_INTERNAL_OUTPUTS["selected_anchor_rank"]
        broken[anchor_rank_name] = derived[anchor_rank_name].copy()
        broken[anchor_rank_name][0, 0] += 1
        with self.assertRaises(localization.LocalizationUnresolved):
            localization._v26_onnx_selection(broken)

    def test_graph_contract_records_topk_gather_attributes(self):
        class Attr:
            def __init__(self, name, value, kind=2):
                self.name, self.type, self.i, self.ints = name, kind, value if kind == 2 else 0, value if kind == 7 else []

        class Node:
            def __init__(self, name, op_type, outputs, attrs=()):
                self.name, self.op_type, self.input, self.output, self.attribute = name, op_type, [], outputs, list(attrs)

        nodes = []
        for name in localization.V26_TRACE_NODES:
            op = "TopK" if name in ("/model.23/TopK", "/model.23/TopK_1") else "GatherElements" if name in ("/model.23/GatherElements", "/model.23/GatherElements_1") else "Split" if name == "/model.23/Split" else "Unknown"
            attrs = (Attr("axis", -1), Attr("largest", 1), Attr("sorted", 1)) if op == "TopK" else (Attr("axis", 1),) if op == "GatherElements" else (Attr("axis", -1),) if op == "Split" else ()
            outputs = list(localization.V26_LINEAGE_OUTPUTS.get(name, (name + "_output",)))
            nodes.append(Node(name, op, outputs, attrs))
        class Graph:
            pass
        graph = Graph()
        graph.node = nodes
        graph.output = [SimpleNamespace(name="output0")]
        class Dim:
            def __init__(self, value):
                self.dim_value, self.dim_param = value, ""
        class TensorType:
            def __init__(self, shape, dtype):
                self.elem_type = 1 if dtype == "float32" else 7
                self.shape = SimpleNamespace(dim=[Dim(value) for value in shape])
        graph.input = []
        graph.value_info = [SimpleNamespace(name=localization.V26_INTERNAL_OUTPUTS[key], type=SimpleNamespace(tensor_type=TensorType(spec["shape"], spec["dtype"]))) for key, spec in localization.V26_EXPECTED_INTERNAL_METADATA.items()]
        model = SimpleNamespace(graph=graph)
        contract = localization.validate_v26_graph_contract(model)
        self.assertEqual(contract["topk_contract"]["/model.23/TopK"]["axis"], -1)
        self.assertEqual(contract["flattening_contract"]["class_count"], 3)
        graph.value_info[0].type.tensor_type.shape.dim[0].dim_value = 2
        with self.assertRaises(localization.LocalizationUnresolved):
            localization.validate_v26_graph_contract(model)

    def test_publishable_inventory_excludes_private_derived_graph(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "out"
            (root / "models/yolo26n/private").mkdir(parents=True)
            (root / "models/yolo26n/private/view.onnx").write_bytes(b"private")
            (root / "models/yolo26n/localization_report.json").write_text("{}", encoding="utf-8")
            inventory = localization.publishable_inventory(root)
            self.assertIn("models/yolo26n/localization_report.json", inventory)
            self.assertNotIn("models/yolo26n/private/view.onnx", inventory)

    def test_child_failure_state_is_model_scoped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "models/yolov8n/partial").mkdir(parents=True)
            (root / "models/yolo26n/partial").mkdir(parents=True)
            (root / "models/yolov8n/partial/a.json").write_text("{}", encoding="utf-8")
            (root / "models/yolo26n/partial/b.json").write_text("{}", encoding="utf-8")
            self.assertEqual(localization.child_owned_files(root, "yolo26n"), ["models/yolo26n/partial/b.json"])
            state = localization.new_child_state("yolo26n")
            localization.stage(state, "original_onnx_complete")
            state["forward_counts"]["native_model_forwards"]["attempted"] = 1
            self.assertEqual(state["stage"], "original_onnx_complete")
            self.assertNotIn("models/yolov8n", localization.child_owned_files(root, "yolo26n"))


if __name__ == "__main__":
    unittest.main()
