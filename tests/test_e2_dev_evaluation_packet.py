import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from edge_readiness.e2_dev_evaluation_packet import (  # noqa: E402
    ENGINE_BYTES,
    ENGINE_SHA256,
    SOURCE_MANIFEST_SHA256,
    SOURCE_ONNX_SHA256,
    TARGET_MANIFEST_SHA256,
    MockRuntime,
    PacketError,
    DEV_IMAGES,
    CANONICAL_DEV_IMAGE_IDS_SHA256,
    CANONICAL_DEV_IMAGE_IDS_SOURCE,
    SERVER_TRT_REFERENCE_PREDICTIONS_SHA256,
    AdapterRuntimeDouble,
    BoxPrediction,
    GroundTruth,
    evaluate_predictions,
    evaluate_canonical_coco_xml_pair,
    mock_evaluator,
    run_durable_child_study,
    run_mock_study,
    stream_input_records,
    study_contract,
    validate_output_payload,
    validate_package_inventory,
    validate_study_contract,
    write_source_reference_pending,
)


def output(seed):
    return struct.pack("<{}f".format(7 * 8400), *([float(seed)] * (7 * 8400)))


def canonical_ids():
    payload = json.loads((ROOT / "results/measurement_audit_v1/server_fp16_capture_v1/validator_predictions.json").read_text(encoding="utf-8"))
    return sorted(record["image"] for record in payload["records"])


def package_inventory():
    ids = canonical_ids()
    return {"source_manifest_sha256": SOURCE_MANIFEST_SHA256, "source_onnx_sha256": SOURCE_ONNX_SHA256, "target_manifest_sha256": TARGET_MANIFEST_SHA256, "engine_sha256": ENGINE_SHA256, "engine_bytes": ENGINE_BYTES, "engine_public": False, "raw_tensors_public": False, "image_ids": ids, "canonical_image_ids_sha256": CANONICAL_DEV_IMAGE_IDS_SHA256, "canonical_image_ids_source": CANONICAL_DEV_IMAGE_IDS_SOURCE, "input_contract": {"shape": [1, 3, 640, 640], "dtype": "float32", "byteorder": "little", "finite": True, "bytes_per_image": 3 * 640 * 640 * 4}, "input_storage_mode": "streaming_manifest_only", "raw_input_tensor_bundle_forbidden": True, "input_records": [{"image_id": image_id, "path": "private/inputs/{}.bin".format(image_id), "bytes": 3 * 640 * 640 * 4, "sha256": "a" * 64, "shape": [1, 3, 640, 640], "dtype": "float32", "byteorder": "little", "finite": True} for image_id in ids], "reference_identity": {"source_onnx_sha256": SOURCE_ONNX_SHA256, "server_trt_predictions_sha256": SERVER_TRT_REFERENCE_PREDICTIONS_SHA256, "server_trt_is_source_onnx": False, "source_onnx_cpu_reference_status": "pending_not_executed", "source_onnx_cpu_predictions_sha256": None, "xml_sha256": "a" * 64}}


class E2DevEvaluationPacketTests(unittest.TestCase):
    def test_contract_freezes_counts_and_no_execution(self):
        contract = study_contract()
        self.assertEqual(validate_study_contract(contract)["target_image_passes"], DEV_IMAGES)
        self.assertEqual(contract["call_budget"]["builder_invocations"], 0)
        self.assertEqual(contract["runtime_contract"]["warmup"], 0)
        bad = dict(contract, scope=dict(contract["scope"], images=1635))
        with self.assertRaisesRegex(PacketError, "PACKET_CONTRACT_MISMATCH"):
            validate_study_contract(bad)

    def test_contract_rejects_engine_hash_warmup_and_reference_count_changes(self):
        contract = study_contract()
        for changed in (
            dict(contract, immutable_inputs=dict(contract["immutable_inputs"], engine_sha256="0" * 64)),
            dict(contract, runtime_contract=dict(contract["runtime_contract"], warmup=1)),
            dict(contract, call_budget=dict(contract["call_budget"], warmup_calls=1)),
        ):
            with self.assertRaisesRegex(PacketError, "PACKET_CONTRACT_MISMATCH"):
                validate_study_contract(changed)

    def test_package_binding_rejects_wrong_source_or_public_engine(self):
        inventory = package_inventory()
        self.assertEqual(validate_package_inventory(inventory)["status"], "package_binding_verified")
        bad = dict(inventory, source_onnx_sha256="0" * 64)
        with self.assertRaisesRegex(PacketError, "PACKAGE_BINDING_MISMATCH"):
            validate_package_inventory(bad)
        with self.assertRaisesRegex(PacketError, "PRIVATE_ARTIFACT_LEAK_POLICY"):
            validate_package_inventory(dict(inventory, engine_public=True))

    def test_package_checks_actual_file_hash_and_unexpected_file(self):
        inventory = package_inventory()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / "manifest.json"; path.write_text("ok", encoding="utf-8")
            inventory["allowed_files"] = [{"path": "manifest.json", "bytes": path.stat().st_size, "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}]
            self.assertEqual(validate_package_inventory(inventory, root)["files_checked"], 1)
            (root / "unexpected.bin").write_bytes(b"x")
            with self.assertRaisesRegex(PacketError, "PACKAGE_UNDECLARED_FILE"):
                validate_package_inventory(inventory, root)

    def test_output_shape_and_nonfinite_guards(self):
        with self.assertRaisesRegex(PacketError, "OUTPUT_SHAPE_MISMATCH"):
            validate_output_payload(b"short")
        bad = bytearray(output(1))
        bad[:4] = struct.pack("<f", float("nan"))
        with self.assertRaisesRegex(PacketError, "OUTPUT_NONFINITE"):
            validate_output_payload(bytes(bad))

    def test_successful_end_to_end_synthetic_artifact_to_analysis(self):
        ids = ("00006", "00009", "00028")
        payloads = {image_id: output(index + 1) for index, image_id in enumerate(ids)}
        runtime = MockRuntime(payloads)
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_study(ids, {image_id: b"input-" + image_id.encode() for image_id in ids}, runtime, Path(temp) / "run")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["counters"]["target_calls_attempted"], 3)
            self.assertEqual(result["counters"]["target_calls_completed"], 3)
            self.assertEqual(result["counters"]["analysis_completed"], 3)
            self.assertEqual(result["target_calls_observed"], 3)
            self.assertTrue(runtime.closed)
            self.assertTrue((Path(temp) / "run" / "result.json").is_file())
            self.assertEqual(len((Path(temp) / "run" / "events.jsonl").read_text().splitlines()), 13)

    def test_timeout_publishes_partial_evidence_without_retry(self):
        ids = ("00006", "00009", "00028")
        payloads = {image_id: output(index + 1) for index, image_id in enumerate(ids)}
        runtime = MockRuntime(payloads, timeout_image="00009")
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_study(ids, {image_id: b"input" for image_id in ids}, runtime, Path(temp) / "run")
            self.assertEqual(result["status"], "failed_partial")
            self.assertEqual(result["error"]["code"], "STAGE_TIMEOUT")
            self.assertEqual(result["counters"]["target_calls_attempted"], 2)
            self.assertEqual(result["counters"]["target_calls_completed"], 1)
            self.assertEqual(result["counters"]["unknown_completions"], ["target_call:00009"])
            self.assertEqual(result["target_calls_observed"], 2)
            self.assertEqual(result["counters"]["analysis_completed"], 1)

    def test_cleanup_failure_preserves_primary_result_and_no_retry(self):
        runtime = MockRuntime({"00006": output(1)}, cleanup_error=True)
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_study(("00006",), {"00006": b"input"}, runtime, Path(temp) / "run")
            self.assertEqual(result["status"], "failed_partial")
            self.assertEqual(result["error"]["code"], "CLEANUP_FAILURE")
            events = (Path(temp) / "run" / "events.jsonl").read_text()
            self.assertIn("cleanup_failed", events)
            self.assertNotIn("retry", events.lower())

    def test_mock_evaluator_and_ap_ci_are_cpu_artifact_analysis(self):
        row = mock_evaluator("00006", output(1))
        self.assertEqual(row["metric_status"], "synthetic_cpu_evaluated")
        self.assertTrue(row["ap_available"])
        predictions = [BoxPrediction("a", 0, 0.9, (0.0, 0.0, 10.0, 10.0)), BoxPrediction("b", 0, 0.8, (20.0, 20.0, 30.0, 30.0))]
        truths = [GroundTruth("a", 0, (0.0, 0.0, 10.0, 10.0)), GroundTruth("b", 0, (0.0, 0.0, 10.0, 10.0))]
        metrics = evaluate_predictions(predictions, truths, ["a", "b"])
        self.assertGreater(metrics["metrics"]["AP50"], 0.0)
        self.assertIn("AP50_95", metrics["metrics"])
        empty = evaluate_predictions([], [GroundTruth("a", 0, (0.0, 0.0, 10.0, 10.0))], ["a"])
        self.assertEqual(empty["metrics"]["AP50"], 0.0)

    def test_durable_child_timeout_preserves_partial_events(self):
        ids = ("00006", "00009")
        payloads = {image_id: output(index + 1) for index, image_id in enumerate(ids)}
        with tempfile.TemporaryDirectory() as temp:
            result = run_durable_child_study(ids, payloads, Path(temp) / "child", timeout_seconds=3.0, hang_at="00009")
            self.assertEqual(result["status"], "failed_partial")
            self.assertEqual(result["error"]["code"], "STAGE_TIMEOUT")
            self.assertEqual(result["unknown_completion_state"], "unknown")
            events = (Path(temp) / "child" / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn("target_call_hanging", events)

    def test_streaming_producer_checks_one_bounded_float32_payload(self):
        image_id = "00014.jpg"
        payload = b"\x00" * (3 * 640 * 640 * 4)
        record = {"image_id": image_id, "bytes": len(payload), "shape": [1, 3, 640, 640], "dtype": "float32", "byteorder": "little", "finite": True, "sha256": __import__("hashlib").sha256(payload).hexdigest()}
        rows = list(stream_input_records([record], [image_id], lambda item: payload))
        self.assertEqual(rows, [(image_id, payload)])

    def test_existing_adapter_double_runs_real_adapter_lifecycle(self):
        runtime = AdapterRuntimeDouble()
        with tempfile.TemporaryDirectory() as temp:
            input_bytes = b"\x00" * (3 * 640 * 640 * 4)
            result = run_mock_study(("00014.jpg",), {"00014.jpg": input_bytes}, runtime, Path(temp) / "adapter", bootstrap_resamples=10)
            self.assertEqual(result["status"], "complete")
            self.assertTrue(runtime.closed)
            self.assertIn("h2d_copy_completed", runtime.events)
            self.assertIn("d2h_copy_synchronized", runtime.events)

    def test_source_reference_is_explicitly_pending_and_canonical_evaluator_has_no_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            record = write_source_reference_pending(Path(temp) / "source-reference.json")
            self.assertEqual(record["status"], "pending_not_executed")
            self.assertEqual(record["executed_image_passes"], 0)
        with self.assertRaisesRegex(PacketError, "CANONICAL_EVALUATOR_DEPENDENCY_MISSING"):
            evaluate_canonical_coco_xml_pair([{"image": "00014.jpg", "orig_shape": [1, 1], "xyxy": [], "confidence": [], "class_id": []}], [{"image": "00014.jpg", "orig_shape": [1, 1], "xyxy": [], "confidence": [], "class_id": []}], Path("missing.xml"), resamples=2)


if __name__ == "__main__":
    unittest.main()
