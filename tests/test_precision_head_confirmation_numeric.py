import copy
import io
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import verify_precision_head_confirmation_numeric as numeric  # noqa: E402


class _FakeTensor:
    def __init__(self, value):
        self._value = np.ascontiguousarray(value)

    @property
    def device(self):
        return SimpleNamespace(type="cpu")

    @property
    def dtype(self):
        return "torch.float32"

    def detach(self):
        return self

    def to(self, _device):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self._value


class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeTorch:
    Tensor = _FakeTensor

    @staticmethod
    def device(_name):
        return SimpleNamespace(type="cpu")

    @staticmethod
    def no_grad():
        return _FakeNoGrad()


class _FakeLetterBox:
    interpolation = 1

    def __init__(self, new_shape, **_kwargs):
        self.new_shape = tuple(new_shape)

    def get_params(self, sample):
        image = sample["img"]
        return {
            "orig_shape": tuple(image.shape[:2]),
            "ratio": (1.0, 1.0),
            "new_unpad": tuple(image.shape[1::-1]),
            "top": 0,
            "bottom": 0,
            "left": 0,
            "right": 0,
        }

    def __call__(self, image=None, **kwargs):
        value = image if image is not None else kwargs["img"]
        result = np.zeros((*self.new_shape, 3), dtype=np.uint8)
        height = min(result.shape[0], value.shape[0])
        width = min(result.shape[1], value.shape[1])
        result[:height, :width] = value[:height, :width]
        return result


class _FakeBasePredictor:
    @staticmethod
    def pre_transform(predictor, images):
        return [_FakeLetterBox((640, 640))(image=image) for image in images]

    @staticmethod
    def preprocess(predictor, images):
        transformed = predictor.pre_transform(images)
        return _FakeTensor(numeric.preprocess_array_contract(transformed[0], np))


class _FakeYOLODataset:
    @staticmethod
    def load_image(_self, _index, rect_mode, resize_short):
        assert rect_mode is True
        assert resize_short is False
        return np.zeros((384, 640, 3), dtype=np.uint8), (480, 800), (384, 640)


class _FakeSessionOptions:
    pass


class _FakeSession:
    def __init__(self, model, error=None):
        self.model = model
        self.error = error

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [SimpleNamespace(name="images", type="tensor(float)", shape=[1, 3, 640, 640])]

    def get_outputs(self):
        return [SimpleNamespace(name="output0", type="tensor(float)", shape=numeric.EXPECTED_OUTPUT_SHAPES[self.model])]

    def run(self, _outputs, _inputs):
        if self.error is not None:
            raise self.error
        return [np.zeros(numeric.EXPECTED_OUTPUT_SHAPES[self.model], dtype=np.float32)]


class _FakeORT:
    GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_BASIC="basic")
    ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")

    def __init__(self, model, error=None):
        self.model = model
        self.error = error

    def SessionOptions(self):
        return _FakeSessionOptions()

    def InferenceSession(self, _path, sess_options, providers):
        assert providers == ["CPUExecutionProvider"]
        return _FakeSession(self.model, self.error)


def _fake_head(model, wrong=False):
    head_type = "WrongHead" if wrong else "Detect"
    head = type(head_type, (), {})()
    head.i = 999 if wrong else numeric.EXPECTED_HEAD_IDENTITIES[model]["head_index"]
    head.end2end = not numeric.EXPECTED_HEAD_IDENTITIES[model]["end2end"] if wrong else numeric.EXPECTED_HEAD_IDENTITIES[model]["end2end"]
    return head


class _FakeNetwork:
    def __init__(self, model, wrong_head=False):
        self.model_name = model
        self.model = [_fake_head(model, wrong_head)]

    def to(self, _device):
        return self

    def float(self):
        return self

    def eval(self):
        return self

    def parameters(self):
        return [SimpleNamespace(device=SimpleNamespace(type="cpu"), dtype="torch.float32")]

    def __call__(self, _tensor):
        primary = _FakeTensor(np.zeros(numeric.EXPECTED_OUTPUT_SHAPES[self.model_name], dtype=np.float32))
        boxes = _FakeTensor(np.zeros((1, 64 if self.model_name == "yolov8n" else 4, 8400), dtype=np.float32))
        scores = _FakeTensor(np.zeros((1, 3, 8400), dtype=np.float32))
        feats = [_FakeTensor(np.zeros((1, 1, 1, 1), dtype=np.float32)) for _ in range(3)]
        branch = {"boxes": boxes, "scores": scores, "feats": feats}
        if self.model_name == "yolov8n":
            return primary, branch
        return primary, {"one2many": branch, "one2one": branch}


def _fake_runtime(model, *, wrong_head=False, onnx_error=None):
    network = _FakeNetwork(model, wrong_head=wrong_head)

    def yolo(_path, task):
        assert task == "detect"
        return SimpleNamespace(model=network)

    return {
        "np": np,
        "ort": _FakeORT(model, onnx_error),
        "torch": _FakeTorch(),
        "YOLO": yolo,
        "YOLODataset": _FakeYOLODataset,
        "LetterBox": _FakeLetterBox,
        "imread": lambda _path: np.zeros((480, 800, 3), dtype=np.uint8),
        "BasePredictor": _FakeBasePredictor,
        "packages": {"expected_from_config": {}, "observed": {"test_runtime": True}, "mode": "CPU double"},
        "available_providers": ["CPUExecutionProvider"],
    }


def _selection(selection_id="U42", count=8):
    images = [f"train/images/{index:05d}.jpg" for index in range(count)]
    source = [
        {"image": image, "image_sha256": f"{index + 1:064x}", "image_bytes": 100 + index}
        for index, image in enumerate(images)
    ]
    materialized = [
        {
            "image": image,
            "source_sha256": row["image_sha256"],
            "materialized_sha256": row["image_sha256"],
            "bytes": row["image_bytes"],
        }
        for image, row in zip(images, source)
    ]
    return {
        "id": selection_id,
        "manifest_contract": {"status": "canonical_manifest_valid", "train_only": True, "image_ids": images},
        "selection_audit": {"source_bytes": source},
        "materialization": {"image_bytes": materialized},
    }


class PrecisionHeadNumericTests(unittest.TestCase):
    def _external_child_plan(self, temp, model):
        plan = json.loads((REPO / "results/measurement_audit_v1/precision_head_confirmation_numeric_v1/numeric_plan.json").read_text(encoding="utf-8"))
        plan["output_root"] = "output"
        graph_doc = json.loads((REPO / f"results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4/models/{model}/graph_audit.json").read_text(encoding="utf-8"))
        plan["models"][model]["reference_semantics"] = numeric.build_reference_semantics(model, graph_doc, plan["models"][model]["accepted_contract"], {})
        (temp / "configs/precision_head_confirmation_v1.json").parent.mkdir(parents=True)
        (temp / "configs/precision_head_confirmation_v1.json").write_text("{}", encoding="utf-8")
        evidence = {}

        def add_external_file(path, sha256, byte_count):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
            evidence[str(path.resolve())] = {"path": path.relative_to(temp).as_posix(), "exists": True, "regular_file": True, "bytes": byte_count, "sha256": sha256}

        checkpoint = plan["models"][model]["checkpoint"]["expected"]
        add_external_file(temp / checkpoint["path"], checkpoint["sha256"], checkpoint["bytes"])
        onnx = plan["models"][model]["onnx"]
        add_external_file(temp / onnx["path"], onnx["expected_sha256"], 1)
        for row in plan["fixture"]["forward_images"] + plan["fixture"]["preprocess_trace_images"]:
            source = numeric.resolve_bound_source_image(temp, row)
            materialized = numeric.resolve_bound_materialized_image(temp, row)
            add_external_file(source, row["expected_sha256"], row["expected_bytes"])
            add_external_file(materialized, row["materialized_expected_sha256"], row["expected_bytes"])

        def evidence_fn(repo, path):
            return dict(evidence[str(path.resolve())])

        return plan, evidence_fn

    def test_canonical_plan_runs_both_child_loops_with_faithful_cpu_doubles(self):
        rows = []
        for model in numeric.MODEL_CHOICES:
            with tempfile.TemporaryDirectory() as temp:
                repo = Path(temp)
                plan, evidence_fn = self._external_child_plan(repo, model)
                result = numeric.run_model_child(
                    repo,
                    plan,
                    model,
                    repo / "output/models" / model,
                    runtime_loader=lambda _config, model=model: _fake_runtime(model),
                    source_evidence_fn=lambda: {"boundary": "external_cpu_runtime_double"},
                    file_evidence_fn=evidence_fn,
                )
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["numeric_status"], "pass")
                self.assertEqual(len(result["comparisons"]), 8)
                self.assertEqual(len(result["calibration_preprocess_traces"]), 3)
                self.assertEqual(result["forward_counts"], {"source_cpu_fp32": {"attempted": 8, "completed": 8}, "onnx_cpu": {"attempted": 8, "completed": 8}})
                self.assertEqual(result["lifecycle"]["status"], "completed")
                stages = [event["stage"] for event in result["lifecycle"]["stage_history"]]
                self.assertIn("head_validation_complete", stages)
                self.assertIn("source_forward_complete", stages)
                self.assertIn("onnx_forward_complete", stages)
                self.assertIn("comparison_complete", stages)
                self.assertEqual(stages[-1], "completed")
                rows.append(result)
        aggregate = numeric.aggregate_numeric_verdict(rows)
        self.assertEqual(aggregate["status"], "pass")
        self.assertEqual(aggregate["counts"], {"pass": 2, "fail": 0, "unresolved": 0, "not_observed": 0})

    def test_child_lifecycle_preserves_failure_stage_and_forward_counters(self):
        failure_cases = ("runtime", "head", "preprocess", "source_after", "onnx", "after_onnx")
        for failure_case in failure_cases:
            with self.subTest(failure_case=failure_case), tempfile.TemporaryDirectory() as temp:
                repo = Path(temp)
                plan, evidence_fn = self._external_child_plan(repo, "yolov8n")
                state = numeric.new_child_state("yolov8n")
                runtime_loader = lambda _config: _fake_runtime("yolov8n")
                patches = []
                if failure_case == "runtime":
                    runtime_loader = lambda _config: (_ for _ in ()).throw(RuntimeError("runtime failure"))
                elif failure_case == "head":
                    runtime_loader = lambda _config: _fake_runtime("yolov8n", wrong_head=True)
                elif failure_case == "onnx":
                    runtime_loader = lambda _config: _fake_runtime("yolov8n", onnx_error=RuntimeError("onnx call failure"))
                elif failure_case in ("preprocess", "source_after", "after_onnx"):
                    if failure_case == "preprocess":
                        patches.append(patch.object(numeric, "trace_preprocess", side_effect=RuntimeError("preprocess failure")))
                    elif failure_case == "source_after":
                        patches.append(patch.object(numeric, "extract_native_primary", side_effect=RuntimeError("source output failure")))
                    else:
                        patches.append(patch.object(numeric, "compare_primary", side_effect=RuntimeError("comparison failure after onnx")))
                context = patches[0] if patches else nullcontext()
                with context:
                    with self.assertRaises(Exception):
                        numeric.run_model_child(repo, plan, "yolov8n", repo / "output/models/yolov8n", runtime_loader=runtime_loader, source_evidence_fn=lambda: {"boundary": "external_cpu_runtime_double"}, file_evidence_fn=evidence_fn, state=state)
                if failure_case == "runtime":
                    self.assertEqual(state["stage"], "runtime_loading")
                    self.assertEqual(state["forward_counts"]["source_cpu_fp32"], {"attempted": 0, "completed": 0})
                elif failure_case == "head":
                    self.assertEqual(state["stage"], "head_validation")
                elif failure_case == "preprocess":
                    self.assertEqual(state["stage"], "preprocess")
                elif failure_case == "source_after":
                    self.assertEqual(state["stage"], "source_forward")
                    self.assertEqual(state["forward_counts"]["source_cpu_fp32"], {"attempted": 1, "completed": 1})
                else:
                    self.assertEqual(state["stage"], "comparison" if failure_case == "after_onnx" else "onnx_forward")
                    self.assertEqual(state["forward_counts"]["source_cpu_fp32"], {"attempted": 1, "completed": 1})
                    expected_onnx = {"attempted": 1, "completed": 0 if failure_case == "onnx" else 1}
                    self.assertEqual(state["forward_counts"]["onnx_cpu"], expected_onnx)

    def test_canonical_head_adapter_and_reference_use_real_readiness_metadata(self):
        readiness_manifest = json.loads((REPO / "results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2/readiness_manifest.json").read_text(encoding="utf-8"))
        contracts = {row["label"]: row for row in readiness_manifest["model_contracts"]}
        graph_root = REPO / "results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4"
        for model, expected in numeric.EXPECTED_HEAD_IDENTITIES.items():
            contract = contracts[model]
            self.assertEqual(numeric.canonical_head_contract(contract, model), expected)
            graph_doc = json.loads((graph_root / f"models/{model}/graph_audit.json").read_text(encoding="utf-8"))
            reference = numeric.build_reference_semantics(model, graph_doc, contract, {})
            self.assertEqual(reference["native_reference"]["head_contract"], expected)

    def test_model_plan_uses_canonical_head_contract_from_real_artifacts(self):
        readiness_root = REPO / "results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2"
        graph_root = REPO / "results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4"
        readiness_prefix = REPO / "results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2"
        canonical = {}
        for relative in numeric.graph.READINESS_FILES:
            blob = numeric.readiness.git_blob(REPO, numeric.graph.ACCEPTED_READINESS_COMMIT, readiness_prefix / relative)
            canonical[relative] = json.loads(blob.decode("utf-8")) if relative.endswith(".json") else blob.decode("utf-8")
        accepted = {
            "manifest": canonical["readiness_manifest.json"],
            "model_contracts": canonical["model_contracts.json"],
            "calibration_readiness": canonical["calibration_readiness.json"],
            "schedule": canonical["schedule.json"],
            "accepted_git_commit": numeric.graph.ACCEPTED_READINESS_COMMIT,
            "accepted_execution_commit": numeric.graph.ACCEPTED_READINESS_EXECUTION_COMMIT,
            "files": {},
        }
        config = numeric.read_json(REPO / "configs/precision_head_confirmation_v1.json")
        graph_audit = numeric.validate_graph_audit_artifact(REPO, graph_root, list(numeric.MODEL_CHOICES))
        real_file_evidence = numeric.file_evidence

        def external_onnx_evidence(repo, path):
            if path.name == "model.onnx":
                model = path.parent.name
                return {"path": str(path), "exists": True, "regular_file": True, "bytes": 1, "sha256": numeric.EXPECTED_ONNX_SHA256[model]}
            return real_file_evidence(repo, path)

        with patch.object(numeric, "file_evidence", side_effect=external_onnx_evidence):
            for model, expected in numeric.EXPECTED_HEAD_IDENTITIES.items():
                plan = numeric._model_plan(REPO, config, accepted, graph_audit, model)
                self.assertEqual(plan["reference_semantics"]["native_reference"]["head_contract"], expected)

    def test_native_head_adapter_rejects_malformed_metadata_and_wrong_actual_head(self):
        readiness_manifest = json.loads((REPO / "results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2/readiness_manifest.json").read_text(encoding="utf-8"))
        contract = next(row for row in readiness_manifest["model_contracts"] if row["label"] == "yolov8n")
        malformed = copy.deepcopy(contract)
        malformed["head"].pop("index")
        with self.assertRaisesRegex(ValueError, "Accepted canonical head schema invalid"):
            numeric.canonical_head_contract(malformed, "yolov8n")

        WrongHead = type("WrongHead", (), {})
        wrong_head = WrongHead()
        wrong_head.i = 99
        wrong_head.end2end = False

        class FakeNetwork:
            model = [wrong_head]

            def to(self, _device):
                return self

            def eval(self):
                return self

            def parameters(self):
                return []

        with self.assertRaisesRegex(numeric.NumericUnresolved, "expected=.*observed=.*"):
            numeric.validate_native_head(FakeNetwork(), "yolov8n", contract)

    def test_protocol_locks_cpu_tolerances_and_fixture(self):
        accepted = {"manifest": {"calibration_readiness": {"selections": [_selection("U42"), _selection("U43", 2), _selection("U44", 2)]}}}
        plan = numeric.build_fixture_plan(accepted)
        self.assertEqual(plan["forward_image_count"], 8)
        self.assertEqual([row["image_id"] for row in plan["forward_images"]], [f"{index:05d}" for index in range(8)])
        self.assertEqual([row["selection"] for row in plan["preprocess_trace_images"]], ["U42", "U43", "U44"])
        self.assertEqual(plan["forward_images"][0]["expected_sha256"], f"{1:064x}")
        self.assertEqual(plan["forward_images"][0]["expected_bytes"], 100)
        self.assertEqual(numeric.LOCKED_TOLERANCES["float32"], {"rtol": 1e-4, "atol": 1e-5})
        self.assertTrue(plan["no_labels_read"])

    def test_fixture_requires_canonical_source_image_binding_fields(self):
        bad = _selection()
        for row in bad["selection_audit"]["source_bytes"]:
            row.pop("image_sha256")
            row["source_sha256"] = "legacy-field"
        with self.assertRaisesRegex(ValueError, "Canonical source image binding is incomplete"):
            numeric._selection_rows(bad, 8)

    def test_fixture_requires_canonical_materialized_image_binding_fields(self):
        bad = _selection()
        for row in bad["materialization"]["image_bytes"]:
            row.pop("materialized_sha256")
        with self.assertRaisesRegex(ValueError, "Canonical materialized image binding is incomplete"):
            numeric._selection_rows(bad, 8)

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

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in ("torch", "ultralytics", "onnxruntime", "cv2")),
        "pinned CPU producer dependencies are unavailable",
    )
    def test_calibration_trace_ignores_valid_stale_wrong_channel_and_corrupt_adjacent_npy(self):
        numeric.set_cpu_environment()
        import cv2

        runtime = numeric.load_child_runtime({"runtime": {"torch": "2.8.0+cu129", "ultralytics": "8.4.102", "numpy": "2.4.2", "pycocotools": "2.0.10"}})
        adjacent_payloads = {"valid": io.BytesIO(), "wrong_channel": io.BytesIO(), "corrupt": io.BytesIO(b"not-a-valid-npy")}
        np.save(adjacent_payloads["valid"], np.zeros((4, 4, 3), dtype=np.uint8))
        np.save(adjacent_payloads["wrong_channel"], np.zeros((4, 4, 1), dtype=np.uint8))
        for name, payload in adjacent_payloads.items():
            with tempfile.TemporaryDirectory() as temp:
                image_path = Path(temp) / f"anchor_{name}.jpg"
                image = np.full((480, 800, 3), 17, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(image_path), image))
                adjacent = image_path.with_suffix(".npy")
                adjacent.write_bytes(payload.getvalue())
                before = adjacent.read_bytes()
                with patch.object(runtime["np"], "load", side_effect=AssertionError("adjacent .npy must not be read")):
                    trace = numeric.trace_calibration_preprocess(image_path, runtime, stride=32)
                self.assertEqual(adjacent.read_bytes(), before)
                self.assertFalse(trace["cache_isolation"]["original_adjacent_npy_used"])
                self.assertFalse(trace["cache_isolation"]["original_adjacent_npy_deleted"])
                self.assertNotEqual(Path(trace["cache_isolation"]["isolated_npy_candidate"]).resolve(), adjacent.resolve())

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

    def test_asymmetric_boundary_uses_reference_relative_isclose_order(self):
        reference = np.zeros((1, 7, 8400), dtype=np.float32)
        observed = reference.copy()
        reference[0, 0, 0] = np.float32(0.0010002674534916878)
        observed[0, 0, 0] = np.float32(0.0010103675303980708)
        self.assertTrue(np.isclose(reference[0, 0, 0], observed[0, 0, 0], rtol=1e-4, atol=1e-5))
        self.assertFalse(np.isclose(observed[0, 0, 0], reference[0, 0, 0], rtol=1e-4, atol=1e-5))
        result = numeric.compare_primary("yolov8n", reference, observed, np)
        self.assertEqual(result["status"], "fail")
        self.assertIn("observed first", result["tolerance_order"])

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

    def test_child_failure_inventory_is_model_scoped_and_persists_lifecycle(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            output = repo / "output"
            (output / "models/yolo26n/partial").mkdir(parents=True)
            (output / "models/yolo26n/partial/owned.json").write_text("{}", encoding="utf-8")
            (output / "models/yolov8n/failure.json").parent.mkdir(parents=True)
            (output / "models/yolov8n/failure.json").write_text("{}", encoding="utf-8")
            (output / "logs").mkdir()
            (output / "logs/yolov8n.log").write_text("old child", encoding="utf-8")
            (output / "numeric_plan.json").write_text(json.dumps({"output_root": "output"}), encoding="utf-8")
            args = SimpleNamespace(repo_root=repo, plan=Path("output/numeric_plan.json"), model="yolo26n")

            def fail_with_state(_repo, _plan, _model, _out_dir, **kwargs):
                state = kwargs["state"]
                numeric.record_child_stage(state, "head_validation")
                raise numeric.NumericUnresolved("head failure")

            with patch.object(numeric, "run_model_child", side_effect=fail_with_state):
                self.assertEqual(numeric.run_child(args), 1)
            failure = json.loads((output / "models/yolo26n/failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["stage"], "head_validation")
            self.assertEqual(failure["forward_counts"]["source_cpu_fp32"], {"attempted": 0, "completed": 0})
            self.assertEqual(failure["partial_files_scope"], "models/yolo26n/owned_paths_only")
            self.assertEqual(failure["partial_files"], ["models/yolo26n/partial/owned.json"])
            self.assertNotIn("yolov8n", " ".join(failure["partial_files"]))

    def test_source_has_no_trt_or_export_dispatch(self):
        source = Path(numeric.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import tensorrt", source)
        self.assertNotIn("from tensorrt", source)
        self.assertNotIn(".export(", source)
        self.assertNotIn("expected_contract", source)
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

    def test_v2_links_preserved_v1_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            prior = repo / numeric.PRESERVED_V1_OUTPUT
            prior.mkdir(parents=True)
            (prior / "numeric_manifest.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
            (prior / "numeric_plan.json").write_text("{}", encoding="utf-8")
            (prior / "report.md").write_text("# preserved v1\n", encoding="utf-8")
            evidence = numeric.preserved_v1_evidence(repo, repo / "results/measurement_audit_v1/precision_head_confirmation_numeric_v2")
            self.assertEqual(evidence["status"], "preserved_not_overwritten")
            self.assertEqual(evidence["path"], "results/measurement_audit_v1/precision_head_confirmation_numeric_v1")
            self.assertEqual(evidence["artifact_inventory"], ["numeric_manifest.json", "numeric_plan.json", "report.md"])

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
            self.assertEqual(manifest["run_inventory"]["owner"], "parent")
            self.assertIn("models/yolov8n/failure.json", manifest["run_inventory"]["model_files"]["yolov8n"])
            self.assertIn("models/yolo26n/numeric_report.json", manifest["run_inventory"]["model_files"]["yolo26n"])
            self.assertNotIn("models/yolov8n/failure.json", manifest["run_inventory"]["model_files"]["yolo26n"])

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
