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
from run_precision_head_confirmation_server import run_parent as run_server_parent  # noqa: E402


def graph_fixture(model: str) -> dict:
    if model == "yolov8n":
        branches = ("cv2", "cv3")
    else:
        branches = ("one2one_cv2", "one2one_cv3")
    return {
        "mapping_status": "verified",
        "mapping_hash": "fixture",
        "mapping": {"mapping_status": "verified", "active_branch_audit": {
            branch: {"target_node_names": [f"/model.head/{branch}/{i}/Conv" for i in range(9)]}
            for branch in branches
        }},
    }


class PrecisionHeadConfirmationSuperTests(unittest.TestCase):
    def test_schedule_has_locked_accounting_and_rotated_model_blocks(self):
        jobs = build_schedule()
        validate_schedule(jobs)
        import prepare_precision_head_confirmation as accepted_producer
        readiness_config = json.loads((REPO / "configs/precision_head_confirmation_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(jobs, accepted_producer.generate_schedule(readiness_config))
        self.assertEqual(len(jobs), 84)
        self.assertEqual(sum(job["capture_required"] for job in jobs), 78)
        self.assertEqual([job["model"] for job in jobs[:42]], ["yolov8n"] * 42)
        self.assertEqual([job["model"] for job in jobs[42:]], ["yolo26n"] * 42)
        fp16_positions = [job["sequence"] for job in jobs if job["phase"] == "scored_fp16"]
        self.assertEqual([jobs[position - 1]["round"] for position in fp16_positions], [1, 2, 3, 1, 2, 3])

    def test_schedule_rejects_duplicate_or_missing_cell(self):
        jobs = build_schedule()
        jobs[-1] = copy.deepcopy(jobs[-2])
        with self.assertRaises(ContractError):
            validate_schedule(jobs)
        jobs = build_schedule()
        jobs[6]["selection"] = "U999"
        with self.assertRaises(ContractError):
            validate_schedule(jobs)
        jobs = build_schedule()
        jobs[6]["arm"] = "not-an-arm"
        with self.assertRaises(ContractError):
            validate_schedule(jobs)
        jobs = build_schedule()
        jobs[42], jobs[43] = jobs[43], jobs[42]
        with self.assertRaises(ContractError):
            validate_schedule(jobs)

    def test_architecture_specific_mapping_and_arms(self):
        for model in ("yolov8n", "yolo26n"):
            mapping = mapping_targets(graph_fixture(model), model)
            self.assertEqual(len(mapping["bbox"]), 9)
            self.assertEqual(len(mapping["classification"]), 9)
            self.assertEqual(selected_targets(mapping, "baseline_int8"), [])
            self.assertEqual(len(selected_targets(mapping, "both_fp32")), 18)

    def test_real_accepted_graph_artifacts_use_nested_mapping_schema(self):
        for model in ("yolov8n", "yolo26n"):
            path = REPO / "results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4/models" / model / "graph_audit.json"
            if not path.is_file():
                self.skipTest("accepted graph artifact is not in this checkout")
            mapping = mapping_targets(json.loads(path.read_text(encoding="utf-8")), model)
            self.assertTrue(mapping["bbox"])
            self.assertTrue(mapping["classification"])

    def test_mapping_rejects_non_convolution_and_ambiguous_targets(self):
        graph = graph_fixture("yolov8n")
        graph["mapping"]["active_branch_audit"]["cv2"]["target_node_names"][0] = "/model.head/cv2/0/act/Sigmoid"
        with self.assertRaises(ContractError):
            mapping_targets(graph, "yolov8n")
        graph = graph_fixture("yolov8n")
        graph["mapping"]["active_branch_audit"]["cv3"]["target_node_names"].append(graph["mapping"]["active_branch_audit"]["cv2"]["target_node_names"][0])
        with self.assertRaises(ContractError):
            mapping_targets(graph, "yolov8n")

    def test_cache_only_contract_distinguishes_auxiliary_and_scored(self):
        verify_cache_only_audit({"read_calls": 2, "cache_consumed": True, "write_calls": 0, "batch_calls": 0}, scored=True)
        verify_cache_only_audit({"read_calls": 0, "write_calls": 1, "batch_calls": 1024}, scored=False)
        with self.assertRaises(ContractError):
            verify_cache_only_audit({"read_calls": 2, "cache_consumed": True, "write_calls": 1, "batch_calls": 0}, scored=True)

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

    def test_external_runtime_double_executes_full_parent_child_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_path = root / "confirmation_plan.json"
            plan = {"schedule": {"jobs": build_schedule()}, "models": {}}
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            out = root / "run"
            args = type("Args", (), {"repo": REPO, "plan": plan_path, "out_dir": out, "device": "0", "child_timeout": 30, "runtime_double": True, "confirm_desktop_process": [], "confirm_background_process": []})()
            self.assertEqual(run_server_parent(args), 0)
            manifest = json.loads((out / "execution_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed_review_required")
            self.assertEqual(len(manifest["jobs"]), 84)
            self.assertEqual(len(list(out.glob("jobs/*/cell_metrics.json"))), 78)

    def test_parent_timeout_preserves_blocked_partial_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_path = root / "confirmation_plan.json"
            plan_path.write_text(json.dumps({"schedule": {"jobs": build_schedule()}, "models": {}}), encoding="utf-8")
            out = root / "run"
            args = type("Args", (), {"repo": REPO, "plan": plan_path, "out_dir": out, "device": "0", "child_timeout": 0, "runtime_double": True, "confirm_desktop_process": [], "confirm_background_process": []})()
            self.assertEqual(run_server_parent(args), 124)
            manifest = json.loads((out / "execution_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "incomplete_blocked")
            self.assertEqual(manifest["jobs"][0]["state"]["status"], "timeout")
            self.assertTrue((out / "jobs").exists())

    def test_analyzer_rejects_missing_cell_and_provenance_mutation(self):
        from analyze_precision_head_confirmation import _cell_files
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "confirmation_plan.json").write_text("{}", encoding="utf-8")
            jobs = build_schedule()
            manifest = {"status": "completed_review_required", "jobs": [{"state": {"status": "completed"}} for _ in jobs]}
            (root / "execution_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            expected_plan_sha = __import__("hashlib").sha256((root / "confirmation_plan.json").read_bytes()).hexdigest()
            capture_jobs = [job for job in jobs if job["capture_required"]]
            for index, job in enumerate(capture_jobs):
                directory = root / "jobs" / f"cell-{index:03d}"
                directory.mkdir(parents=True)
                prediction = directory / "predictions.json"
                prediction.write_text("{}", encoding="utf-8")
                cell = {"job": job, "job_id": f"{job['sequence']:03d}_{job['model']}_{job['phase']}_{job['selection'] or 'NA'}_{job['arm']}_r{job['repeat']}", "plan_sha256": expected_plan_sha, "record_count": 1636, "prediction_sha256": __import__("hashlib").sha256(prediction.read_bytes()).hexdigest()}
                (directory / "cell_metrics.json").write_text(json.dumps(cell), encoding="utf-8")
            (root / "jobs" / "cell-077" / "cell_metrics.json").unlink()
            with self.assertRaises(ContractError):
                _cell_files(root)
            directory = root / "jobs" / "cell-077"
            directory.mkdir(exist_ok=True)
            prediction = directory / "predictions.json"
            prediction.write_text("{}", encoding="utf-8")
            last = capture_jobs[-1]
            bad = {"job": last, "job_id": f"{last['sequence']:03d}_{last['model']}_{last['phase']}_{last['selection'] or 'NA'}_{last['arm']}_r{last['repeat']}", "plan_sha256": "mutated", "record_count": 1636, "prediction_sha256": __import__("hashlib").sha256(prediction.read_bytes()).hexdigest()}
            (directory / "cell_metrics.json").write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ContractError):
                _cell_files(root)

    def test_synthetic_pooled_ap_is_not_mean_of_per_image_ap(self):
        if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("pycocotools") is None:
            self.skipTest("NumPy/pycocotools unavailable for pooled AP fixture")
        sys.path.insert(0, str(REPO / "scripts"))
        from verify_cctsdb_capture import coco_size
        from analyze_dev_quantization import ap_values, resample_ap

        def record(name, boxes, scores):
            return {"image": name, "orig_shape": [100, 100], "xyxy": boxes, "confidence": scores, "class_id": [0] * len(boxes)}

        records = [record("a.jpg", [[10, 10, 30, 30]], [0.9]), record("b.jpg", [[10, 10, 30, 30], [60, 60, 70, 70], [70, 70, 80, 80]], [0.9, 0.8, 0.7])]
        xml = {"a.jpg": {"shape": [100, 100], "rows": [(0, [10, 10, 30, 30])]}, "b.jpg": {"shape": [100, 100], "rows": [(0, [10, 10, 30, 30]), (0, [40, 40, 50, 50]), (0, [20, 70, 30, 80])]}}
        pooled, evaluator = coco_size(records, xml, return_evaluator=True)
        single = [coco_size([row], {row["image"]: xml[row["image"]]})["metrics"]["all"]["map50_95"] for row in records]
        self.assertNotAlmostEqual(pooled["metrics"]["all"]["map50_95"], sum(single) / len(single), places=8)
        duplicated = resample_ap(evaluator, [0, 0])
        self.assertTrue(duplicated.shape[0] >= 1)


if __name__ == "__main__":
    unittest.main()
