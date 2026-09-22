import json
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

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
    StagePermissions,
    run_source_reference_stage,
    run_target_execution_stage,
    ImageStreamSource,
    postprocess_saved_outputs,
    validate_reference_artifact,
    run_canonical_analysis_stage,
    stream_bound_images,
)
import edge_readiness.e2_dev_evaluation_packet as packet_module


def output(seed):
    return struct.pack("<{}f".format(7 * 8400), *([float(seed)] * (7 * 8400)))


def canonical_ids():
    payload = json.loads((ROOT / "results/measurement_audit_v1/server_fp16_capture_v1/validator_predictions.json").read_text(encoding="utf-8"))
    return sorted(record["image"] for record in payload["records"])


def package_inventory():
    ids = canonical_ids()
    return {"source_manifest_sha256": SOURCE_MANIFEST_SHA256, "source_onnx_sha256": SOURCE_ONNX_SHA256, "target_manifest_sha256": TARGET_MANIFEST_SHA256, "engine_sha256": ENGINE_SHA256, "engine_bytes": ENGINE_BYTES, "engine_public": False, "raw_tensors_public": False, "image_ids": ids, "canonical_image_ids_sha256": CANONICAL_DEV_IMAGE_IDS_SHA256, "canonical_image_ids_source": CANONICAL_DEV_IMAGE_IDS_SOURCE, "input_contract": {"shape": [1, 3, 640, 640], "dtype": "float32", "byteorder": "little", "finite": True, "bytes_per_image": 3 * 640 * 640 * 4}, "input_storage_mode": "streaming_manifest_only", "raw_input_tensor_bundle_forbidden": True, "input_records": [{"image_id": image_id, "path": "private/inputs/{}.bin".format(image_id), "bytes": 3 * 640 * 640 * 4, "sha256": "a" * 64, "shape": [1, 3, 640, 640], "dtype": "float32", "byteorder": "little", "finite": True} for image_id in ids], "input_producer": {"mode": "bound_image_preprocess_stream", "preprocess": "scripts.edge_readiness.e2_source_bundle:UltralyticsSourceRuntime.preprocess", "output_contract": {"shape": [1, 3, 640, 640], "dtype": "float32", "byteorder": "little", "finite": True}, "image_records": [{"image_id": image_id, "path": "private/images/{}.jpg".format(image_id), "bytes": 1, "sha256": "b" * 64, "orig_shape": [640, 640], "resized_shape": [640, 640]} for image_id in ids]}, "reference_identity": {"source_onnx_sha256": SOURCE_ONNX_SHA256, "server_trt_predictions_sha256": SERVER_TRT_REFERENCE_PREDICTIONS_SHA256, "server_trt_is_source_onnx": False, "source_onnx_cpu_reference_status": "pending_not_executed", "source_onnx_cpu_predictions_sha256": None, "xml_sha256": "a" * 64}}


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
        self.assertEqual(validate_package_inventory(inventory)["status"], "metadata_pending")
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

    def test_durable_child_does_not_publish_success_before_cleanup(self):
        ids = ("00006",)
        payloads = {ids[0]: output(1)}
        with tempfile.TemporaryDirectory() as temp:
            result = run_durable_child_study(ids, payloads, Path(temp) / "child", timeout_seconds=3.0, adapter_double=True, adapter_cleanup_error=True)
            self.assertEqual(result["status"], "failed_partial")
            self.assertEqual(result["error"]["code"], "CLEANUP_FAILURE")
            events = (Path(temp) / "child" / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn("cleanup_failed", events)
            self.assertNotIn('"event": "child_completed"', events)

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
        original = packet_module._canonical_dependencies
        packet_module._canonical_dependencies = lambda: (_ for _ in ()).throw(PacketError("CANONICAL_EVALUATOR_DEPENDENCY_MISSING:forced"))
        try:
            with self.assertRaisesRegex(PacketError, "CANONICAL_EVALUATOR_DEPENDENCY_MISSING"):
                evaluate_canonical_coco_xml_pair([{"image": "00014.jpg", "orig_shape": [1, 1], "xyxy": [], "confidence": [], "class_id": []}], [{"image": "00014.jpg", "orig_shape": [1, 1], "xyxy": [], "confidence": [], "class_id": []}], Path("missing.xml"), resamples=2)
        finally:
            packet_module._canonical_dependencies = original

    def test_canonical_evaluator_real_xml_and_paired_resampling(self):
        ids = ["00001.jpg", "00002.jpg", "00003.jpg", "00004.jpg"]
        def row(image, box):
            return {"image": image, "orig_shape": [100, 100], "xyxy": [list(box)], "confidence": [0.9], "class_id": [0]}
        source = [row(image, (10, 10, 30, 30)) for image in ids]
        target = [row(image, (10, 10, 30, 30)) for image in ids]
        with tempfile.TemporaryDirectory() as temp:
            xml_path = Path(temp) / "labels.zip"
            with zipfile.ZipFile(xml_path, "w") as archive:
                for image in ids:
                    archive.writestr(Path(image).stem + ".xml", "<annotation><size><width>100</width><height>100</height></size><object><name>prohibitory</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>30</ymax></bndbox></object></annotation>")
            result = evaluate_canonical_coco_xml_pair(source, target, xml_path, resamples=4)
            self.assertEqual(result["backend"], "coco_xml_paired_image_bootstrap_v1")
            self.assertEqual(result["image_count"], 4)
            self.assertEqual(result["paired_target_minus_source"]["all"]["AP50"]["point_delta_pp"], 0.0)
            self.assertEqual(result["paired_target_minus_source"]["all"]["AP50"]["valid_resamples"], 4)
            self.assertNotIn("NaN", json.dumps(result))
            self.assertEqual(set(result["paired_target_minus_source"]["all"]), {"AP50", "AP50_95"})
            changed = list(source)
            changed[0] = row(ids[0], (60, 60, 80, 80))
            changed_result = evaluate_canonical_coco_xml_pair(source, changed, xml_path, resamples=4)
            self.assertLessEqual(changed_result["target_point"][0][0], changed_result["source_point"][0][0])
            empty = [{"image": image, "orig_shape": [100, 100], "xyxy": [], "confidence": [], "class_id": []} for image in ids]
            empty_result = evaluate_canonical_coco_xml_pair(empty, empty, xml_path, resamples=4)
            self.assertNotIn("NaN", json.dumps(empty_result))
            self.assertIn("xs", empty_result["paired_target_minus_source"])
            with self.assertRaisesRegex(PacketError, "CANONICAL_EVALUATOR_DUPLICATE_IMAGE"):
                evaluate_canonical_coco_xml_pair(source + [source[0]], target + [target[0]], xml_path, resamples=4)

    def test_bound_image_producer_rejects_changed_bytes_and_preserves_contract(self):
        image = b"jpeg-like"
        record = {"image_id": "00001.jpg", "path": "images/00001.jpg", "bytes": len(image), "sha256": __import__("hashlib").sha256(image).hexdigest()}
        tensor = b"\x00" * (3 * 640 * 640 * 4)
        self.assertEqual(list(stream_bound_images([record], ["00001.jpg"], lambda item: image, lambda image_id, payload: tensor))[0][0], "00001.jpg")
        with self.assertRaisesRegex(PacketError, "STREAM_IMAGE_HASH_MISMATCH"):
            list(stream_bound_images([record], ["00001.jpg"], lambda item: b"changed", lambda image_id, payload: tensor))

    def test_canonical_evaluator_covers_class_and_size_support(self):
        cases = [("00001.jpg", 0, "prohibitory", (10, 10, 20, 20)), ("00002.jpg", 1, "mandatory", (10, 10, 30, 20)), ("00003.jpg", 2, "warning", (10, 10, 35, 35)), ("00004.jpg", 0, "prohibitory", (10, 10, 50, 50)), ("00005.jpg", 1, "mandatory", (10, 10, 60, 60))]
        records = [{"image": image, "orig_shape": [100, 100], "xyxy": [list(box)], "confidence": [0.9], "class_id": [class_id]} for image, class_id, _name, box in cases]
        with tempfile.TemporaryDirectory() as temp:
            xml_path = Path(temp) / "labels.zip"
            with zipfile.ZipFile(xml_path, "w") as archive:
                for image, _class_id, name, box in cases:
                    archive.writestr(Path(image).stem + ".xml", "<annotation><size><width>100</width><height>100</height></size><object><name>{}</name><bndbox><xmin>{}</xmin><ymin>{}</ymin><xmax>{}</xmax><ymax>{}</ymax></bndbox></object></annotation>".format(name, *box))
            result = evaluate_canonical_coco_xml_pair(records, records, xml_path, resamples=3)
            self.assertEqual(result["image_count"], 5)
            self.assertEqual(len(result["source_point"]), 6)
            self.assertNotIn("NaN", json.dumps(result))

    def test_source_and_target_stage_guards_and_external_double(self):
        with self.assertRaisesRegex(PacketError, "SOURCE_FORWARD_NOT_AUTHORIZED"):
            run_source_reference_stage([], Path("."), Path("missing"), Path(tempfile.gettempdir()) / "unused-e2l1-027", permissions=StagePermissions())
        ids = ["00001.jpg"]
        tensor = b"\x00" * (3 * 640 * 640 * 4)
        image = tensor
        input_record = {"image_id": ids[0], "path": "images/00001.jpg", "bytes": len(image), "sha256": __import__("hashlib").sha256(image).hexdigest(), "orig_shape": [640, 640], "resized_shape": [640, 640]}
        runtime = AdapterRuntimeDouble()
        with tempfile.TemporaryDirectory() as temp:
            result = run_target_execution_stage([input_record], ids, lambda item: image, runtime, Path(temp) / "target", permissions=StagePermissions(allow_target_inference=True), image_source=ImageStreamSource([input_record], Path(temp), synthetic_inputs={ids[0]: image}))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["attempted_completed"]["target_calls_completed"], 1)
            self.assertIn('"event": "child_finalized"', (Path(temp) / "target" / "events.jsonl").read_text())

    def test_source_stage_binds_image_bytes_and_private_output_with_double(self):
        tensor = b"\x00" * (7 * 8400 * 4)
        image = b"bound-image"
        record = {"image_id": "00001.jpg", "path": "images/00001.jpg", "bytes": len(image), "sha256": __import__("hashlib").sha256(image).hexdigest()}
        class SourceDouble:
            def prepare(self): return {"device": "cpu", "ort_providers_required": ["CPUExecutionProvider"]}
            def validate_onnx(self, path): return {"input_name": "images", "output_name": "output0", "input_shape": [1, 3, 640, 640], "output_shape": [1, 7, 8400]}
            def open_ort(self, path): self.ort_opened = True; return object()
            def preprocess(self, path): return SimpleNamespace(payload=b"\x00" * (3 * 640 * 640 * 4))
            def ort_forward(self, session, input_tensor): self.ort_called = True; return SimpleNamespace(payload=tensor, shape=(1, 7, 8400), dtype="float32")
            def close(self): self.closed = True
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            (root / "images/00001.jpg").write_bytes(image)
            onnx = root / "best.onnx"
            onnx.write_bytes(b"onnx")
            runtime = SourceDouble()
            result = run_source_reference_stage([record], root, onnx, root / "source", permissions=StagePermissions(allow_source_forward=True), runtime=runtime, commit="test", expected_onnx_sha256=__import__("hashlib").sha256(b"onnx").hexdigest())
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["attempted_completed"]["source_onnx_forwards_completed"], 1)
            self.assertTrue(runtime.ort_opened)
            self.assertTrue(runtime.ort_called)
            self.assertTrue((root / "source/private/onnx_reference/00001.jpg.bin").is_file())

    def test_source_stage_rejects_hash_or_provider_before_forward(self):
        image = b"guarded-image"
        record = {"image_id": "00001.jpg", "path": "images/00001.jpg", "bytes": len(image), "sha256": __import__("hashlib").sha256(image).hexdigest()}
        class GuardedRuntime:
            def __init__(self, providers): self.providers = providers; self.forwarded = False
            def prepare(self): return {"device": "cpu", "ort_providers_required": self.providers}
            def validate_onnx(self, path): return {}
            def open_ort(self, path): return object()
            def ort_forward(self, session, tensor): self.forwarded = True; raise AssertionError("forward must be guarded")
            def close(self): pass
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "images").mkdir(); (root / "images/00001.jpg").write_bytes(image)
            onnx = root / "best.onnx"; onnx.write_bytes(b"onnx")
            wrong_hash_runtime = GuardedRuntime(["CPUExecutionProvider"])
            wrong_hash = run_source_reference_stage([record], root, onnx, root / "wrong-hash", permissions=StagePermissions(allow_source_forward=True), runtime=wrong_hash_runtime)
            self.assertEqual(wrong_hash["status"], "failed_partial")
            self.assertIn("SOURCE_ONNX_HASH_MISMATCH", wrong_hash["error"]["message"])
            self.assertFalse(wrong_hash_runtime.forwarded)
            provider_runtime = GuardedRuntime(["CUDAExecutionProvider"])
            onnx_hash = __import__("hashlib").sha256(b"onnx").hexdigest()
            wrong_provider = run_source_reference_stage([record], root, onnx, root / "wrong-provider", permissions=StagePermissions(allow_source_forward=True), runtime=provider_runtime, expected_onnx_sha256=onnx_hash)
            self.assertEqual(wrong_provider["status"], "failed_partial")
            self.assertIn("SOURCE_ORT_PROVIDER_POLICY_MISMATCH", wrong_provider["error"]["message"])
            self.assertFalse(provider_runtime.forwarded)

    def test_raw_output_postprocess_feeds_canonical_record_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_root = root / "raw"
            raw_root.mkdir()
            raw_path = raw_root / "00001.jpg.bin"
            values = [0.0] * (7 * 8400)
            values[0] = 10.0; values[8400] = 10.0; values[2 * 8400] = 4.0; values[3 * 8400] = 4.0; values[4 * 8400] = 0.9
            raw_path.write_bytes(struct.pack("<{}f".format(len(values)), *values))
            raw_manifest = root / "raw_manifest.json"
            raw_manifest.write_text(json.dumps({"records": [{"image_id": "00001.jpg", "path": "00001.jpg.bin", "bytes": raw_path.stat().st_size, "sha256": __import__("hashlib").sha256(raw_path.read_bytes()).hexdigest()}]}), encoding="utf-8")
            result = postprocess_saved_outputs(raw_manifest, raw_root, [{"image_id": "00001.jpg", "orig_shape": [640, 640], "resized_shape": [640, 640]}], root / "post", source_label="synthetic_ort")
            record = result["records"][0]
            self.assertEqual(record["image"], "00001.jpg")
            self.assertEqual(len(record["xyxy"]), 1)
            self.assertEqual(record["postprocess"]["input_shape"], [1, 7, 8400])
            self.assertTrue(record["postprocess"]["input_unchanged"])

    def test_target_stage_exact_child_timeout_and_no_late_success(self):
        ids = ["00001.jpg"]
        image = b"\x00" * (3 * 640 * 640 * 4)
        record = {"image_id": ids[0], "path": "images/00001.jpg", "bytes": len(image), "sha256": __import__("hashlib").sha256(image).hexdigest(), "orig_shape": [640, 640], "resized_shape": [640, 640]}
        with tempfile.TemporaryDirectory() as temp:
            runtime = MockRuntime({ids[0]: output(1)}, hang_image=ids[0])
            result = run_target_execution_stage([record], ids, None, runtime, Path(temp) / "target", permissions=StagePermissions(allow_target_inference=True), image_source=ImageStreamSource([record], Path(temp), synthetic_inputs={ids[0]: image}), timeout_seconds=0.5)
            self.assertEqual(result["status"], "failed_partial")
            self.assertEqual(result["error"]["code"], "STAGE_TIMEOUT")
            self.assertEqual(result["unknown_completion_state"], "unknown")
            self.assertNotIn('"status": "complete"', (Path(temp) / "target" / "manifest.json").read_text())

    def test_target_dispatch_rejects_pending_package_before_runtime(self):
        inventory = package_inventory()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(PacketError, "PACKAGE_NOT_EXECUTION_READY"):
                run_target_execution_stage([], [], None, AdapterRuntimeDouble(), Path(temp) / "target", permissions=StagePermissions(allow_target_inference=True), image_source=ImageStreamSource([], Path(temp), synthetic_inputs={}), inventory=inventory, package_root=Path(temp), require_execution_ready=True)

    def test_full_chain_ort_postprocess_package_target_and_canonical_analysis(self):
        raw_values = [0.0] * (7 * 8400)
        raw_values[0] = 10.0; raw_values[8400] = 10.0; raw_values[2 * 8400] = 4.0; raw_values[3 * 8400] = 4.0; raw_values[4 * 8400] = 0.9
        raw = struct.pack("<{}f".format(len(raw_values)), *raw_values)
        image = b"chain-image"
        input_tensor = b"\x00" * (3 * 640 * 640 * 4)
        image_record = {"image_id": "00001.jpg", "path": "images/00001.jpg", "bytes": len(image), "sha256": __import__("hashlib").sha256(image).hexdigest(), "orig_shape": [640, 640], "resized_shape": [640, 640]}
        class OrtDouble:
            def prepare(self): return {"device": "cpu", "ort_providers_required": ["CPUExecutionProvider"]}
            def validate_onnx(self, path): return {"input_name": "images", "output_name": "output0", "input_shape": [1, 3, 640, 640], "output_shape": [1, 7, 8400]}
            def open_ort(self, path): return object()
            def preprocess(self, path): return SimpleNamespace(payload=input_tensor)
            def ort_forward(self, session, tensor): return SimpleNamespace(payload=raw, shape=(1, 7, 8400), dtype="float32")
            def close(self): pass
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "images").mkdir(); (root / "images/00001.jpg").write_bytes(image); onnx = root / "best.onnx"; onnx.write_bytes(b"onnx")
            source_dir = root / "source"
            test_onnx_hash = __import__("hashlib").sha256(b"onnx").hexdigest()
            with mock.patch.object(packet_module, "SOURCE_ONNX_SHA256", test_onnx_hash), mock.patch.object(packet_module, "DEV_IMAGES", 1):
                source = run_source_reference_stage([image_record], root, onnx, source_dir, permissions=StagePermissions(allow_source_forward=True), runtime=OrtDouble(), expected_onnx_sha256=test_onnx_hash)
                self.assertEqual(validate_reference_artifact(source_dir / "manifest.json", package_root=source_dir)["status"], "verified")
            source_post = postprocess_saved_outputs(source_dir / "manifest.json", source_dir, [image_record], root / "source-post", source_label="onnx_cpu")
            target_dir = root / "target"
            target = run_target_execution_stage([image_record], ["00001.jpg"], None, MockRuntime({"00001.jpg": raw}), target_dir, permissions=StagePermissions(allow_target_inference=True), image_source=ImageStreamSource([image_record], root, synthetic_inputs={"00001.jpg": input_tensor}, synthetic_image_bytes={"00001.jpg": image}), timeout_seconds=3.0)
            self.assertEqual(target["status"], "complete")
            target_post = postprocess_saved_outputs(target_dir / "manifest.json", target_dir, [image_record], root / "target-post", source_label="e2_target")
            xml = root / "labels.zip"
            with zipfile.ZipFile(xml, "w") as archive:
                archive.writestr("00001.xml", "<annotation><size><width>640</width><height>640</height></size><object><name>prohibitory</name><bndbox><xmin>8</xmin><ymin>8</ymin><xmax>12</xmax><ymax>12</ymax></bndbox></object></annotation>")
            analysis = run_canonical_analysis_stage(root / "source-post/records.json", root / "target-post/records.json", xml, root / "analysis", resamples=3)
            self.assertEqual(analysis["backend"], "coco_xml_paired_image_bootstrap_v1")
            self.assertEqual(analysis["image_count"], 1)
            self.assertEqual(analysis["source_point"][0][0], analysis["target_point"][0][0])
if __name__ == "__main__":
    unittest.main()
