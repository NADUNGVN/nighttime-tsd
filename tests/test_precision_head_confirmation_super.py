from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from precision_head_confirmation_contract import (  # noqa: E402
    ContractError,
    ARMS,
    build_schedule,
    mapping_targets,
    sample_sd,
    selected_targets,
    shared_bootstrap_indices,
    validate_schedule,
    verify_cache_only_audit,
)


def graph_fixture(model: str) -> dict:
    if model == "yolov8n":
        branches = ("cv2", "cv3")
    else:
        branches = ("one2one_cv2", "one2one_cv3")
    return {
        "mapping_status": "verified",
        "mapping_hash": "fixture",
        "active_branch_audit": {
            branch: {"target_node_names": [f"/model.head/{branch}/{i}/Conv" for i in range(9)]}
            for branch in branches
        },
    }


class PrecisionHeadConfirmationSuperTests(unittest.TestCase):
    def test_schedule_has_locked_accounting_and_rotated_model_blocks(self):
        jobs = build_schedule()
        validate_schedule(jobs)
        self.assertEqual(len(jobs), 84)
        self.assertEqual(sum(job["capture_required"] for job in jobs), 78)
        self.assertEqual([job["model"] for job in jobs[:42]], ["yolov8n"] * 42)
        self.assertEqual([job["model"] for job in jobs[42:]], ["yolo26n"] * 42)
        self.assertEqual(
            [job["rotation_shift"] for job in jobs if job["kind"] == "scored_fp16"],
            [0, 4, 8, 0, 4, 8],
        )

    def test_schedule_rejects_duplicate_or_missing_cell(self):
        jobs = build_schedule()
        jobs[-1] = copy.deepcopy(jobs[-2])
        with self.assertRaises(ContractError):
            validate_schedule(jobs)

    def test_architecture_specific_mapping_and_arms(self):
        for model in ("yolov8n", "yolo26n"):
            mapping = mapping_targets(graph_fixture(model), model)
            self.assertEqual(len(mapping["bbox"]), 9)
            self.assertEqual(len(mapping["classification"]), 9)
            self.assertEqual(selected_targets(mapping, "baseline_int8"), [])
            self.assertEqual(len(selected_targets(mapping, "both_fp32")), 18)

    def test_mapping_rejects_non_convolution_and_ambiguous_targets(self):
        graph = graph_fixture("yolov8n")
        graph["active_branch_audit"]["cv2"]["target_node_names"][0] = "/model.head/cv2/0/act/Sigmoid"
        with self.assertRaises(ContractError):
            mapping_targets(graph, "yolov8n")
        graph = graph_fixture("yolov8n")
        graph["active_branch_audit"]["cv3"]["target_node_names"].append(graph["active_branch_audit"]["cv2"]["target_node_names"][0])
        with self.assertRaises(ContractError):
            mapping_targets(graph, "yolov8n")

    def test_cache_only_contract_distinguishes_auxiliary_and_scored(self):
        verify_cache_only_audit({"read_calls": 1, "write_calls": 0, "batch_calls": 0}, scored=True)
        verify_cache_only_audit({"read_calls": 0, "write_calls": 1, "batch_calls": 1024}, scored=False)
        with self.assertRaises(ContractError):
            verify_cache_only_audit({"read_calls": 1, "write_calls": 1, "batch_calls": 0}, scored=True)

    def test_statistics_are_sample_sd_and_bootstrap_is_reproducible(self):
        self.assertAlmostEqual(sample_sd([1.0, 2.0, 3.0]), 1.0)
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy is unavailable; server analysis uses NumPy PCG64")
        left = shared_bootstrap_indices(8, 5, 20260916)
        right = shared_bootstrap_indices(8, 5, 20260916)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 5)
        self.assertEqual(len(left[0]), 8)

    def test_model_26_does_not_use_inactive_cv2_cv3(self):
        graph = graph_fixture("yolo26n")
        mapping = mapping_targets(graph, "yolo26n")
        self.assertTrue(all("one2one_" in item for item in mapping["both"]))
        self.assertFalse(any("/cv2/" in item or "/cv3/" in item for item in mapping["both"]))

    def test_parent_source_has_no_runtime_import_and_server_phase_is_gate_closed(self):
        source = (REPO / "scripts/run_precision_head_confirmation.py").read_text(encoding="utf-8")
        self.assertNotIn("import tensorrt", source)
        self.assertNotIn("import torch", source)
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts/run_precision_head_confirmation.py"), "--phase", "scored"],
            cwd=REPO, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("integrated Astra GO", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
