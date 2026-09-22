from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from precision_head_confirmation_contract import (  # noqa: E402
    ContractError,
    ARMS,
    build_schedule,
    canonical_json_sha256,
    mapping_targets,
    sample_sd,
    selected_targets,
    shared_bootstrap_indices,
    validate_schedule,
    verify_cache_only_audit,
)
from run_precision_head_confirmation_server import (  # noqa: E402
    NoWarmupBackendAccounting,
    bind_gpu_identity,
    build_real,
    child_state,
    finalize_child_state,
    job_key,
    validate_canonical_capture_records,
    verify_plan_inputs,
    run_parent as run_server_parent,
)


def graph_fixture(model: str) -> dict:
    if model == "yolov8n":
        branches = ("cv2", "cv3")
    else:
        branches = ("one2one_cv2", "one2one_cv3")
    return {
        "mapping_status": "verified",
        "mapping_hash": "fixture",
        "mapping": {"mapping_status": "verified", "mapping_hash": "fixture", "active_branch_audit": {
            branch: {"target_node_names": [f"/model.head/{branch}/{i}/Conv" for i in range(9)]}
            for branch in branches
        }},
    }


class PrecisionHeadConfirmationSuperTests(unittest.TestCase):
    def runtime_plan(self, root: Path, *, identity_by_sequence: dict[str, dict[str, str]] | None = None) -> tuple[dict, Path]:
        jobs = build_schedule()
        model_evidence = {}
        for model in ("yolov8n", "yolo26n"):
            checkpoint = root / f"{model}.pt"
            onnx = root / f"{model}.onnx"
            checkpoint.write_bytes(f"checkpoint-{model}".encode())
            onnx.write_bytes(f"onnx-{model}".encode())
            model_evidence[model] = {"checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}, "onnx": {"path": str(onnx), "sha256": hashlib.sha256(onnx.read_bytes()).hexdigest()}, "postprocess": f"route-{model}"}
        plan = {"schedule": {"jobs": jobs, "sha256": canonical_json_sha256(jobs)}, "models": model_evidence}
        if identity_by_sequence is not None:
            plan["_test_gpu_identity_by_sequence"] = identity_by_sequence
        plan_path = root / "confirmation_plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        return plan, plan_path

    def normal_path_fixture(self, root: Path, model: str, job: dict, *, parser_ok: bool = True, output_ok: bool = True, capture_ok: bool = True):
        """Producer-shaped plan plus external-library doubles for build_real."""
        source_root = root / "data/processed/cctsdb2021_clean/train/images"
        source_root.mkdir(parents=True, exist_ok=True)
        calibration_selections = []
        for selection in ("U42", "U43", "U44"):
            materialized_root = root / "calibration" / selection
            image_root = materialized_root / "images"
            image_root.mkdir(parents=True, exist_ok=True)
            rows = []
            for index in range(1024):
                name = f"{index:04d}.jpg"
                source = source_root / name
                source.write_bytes(b"calibration-image-" + name.encode())
                (image_root / name).write_bytes(source.read_bytes())
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                rows.append({"image": f"train/images/{name}", "source_sha256": digest, "materialized_sha256": digest})
            yaml_path = materialized_root / "calibration.yaml"
            yaml_path.write_text("path: .\n", encoding="utf-8")
            calibration_selections.append({"id": selection, "status": "verified", "canonical_sha256": f"manifest-{selection}", "manifest": f"manifest-{selection}", "manifest_contract": {"status": "canonical_manifest_valid", "image_ids": [row["image"] for row in rows]}, "materialization": {"status": "complete", "yaml": str(yaml_path), "yaml_sha256": hashlib.sha256(yaml_path.read_bytes()).hexdigest(), "resolved_directory": str(materialized_root), "image_bytes": rows}})
        checkpoint = root / f"{model}.pt"
        onnx = root / f"{model}.onnx"
        checkpoint.write_bytes(f"checkpoint-{model}".encode())
        onnx.write_bytes(f"onnx-{model}".encode())
        target_names = ["/model.22/cv2.0/cv2.0.0/conv/Conv", "/model.22/cv3.0/cv3.0.0/conv/Conv"] if model == "yolov8n" else ["/model.23/one2one_cv2.0/one2one_cv2.0.0/conv/Conv", "/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv"]
        canonical = [{"image": f"dev/{index:04d}.jpg", "stem": f"{index:04d}", "orig_shape": [100, 100]} for index in range(1636)]
        plan = {"runtime": {}, "models": {model: {"checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}, "onnx": {"path": str(onnx), "sha256": hashlib.sha256(onnx.read_bytes()).hexdigest()}, "output_shape": [1, 7, 8400] if model == "yolov8n" else [1, 300, 6], "mapping": {"targets": {"bbox": target_names[:1], "classification": target_names[1:], "both": target_names}}, "postprocess": f"route-{model}"}}, "provenance_files": [], "dataset": {"canonical_dev_reference": {"records": canonical}, "xml_path": str(root / "xml.zip")}, "calibration_producer": {"selections": calibration_selections}, "schedule": {"jobs": [job], "sha256": canonical_json_sha256([job])}}
        (root / "xml.zip").write_bytes(b"synthetic-xml")

        class FakeTensor:
            def to(self, **_kwargs): return self
            def __truediv__(self, _value): return self
            def contiguous(self): return self
            def data_ptr(self): return 1234
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return SimpleNamespace(tobytes=lambda: b"calibration-tensor")

        class FakeStream:
            def synchronize(self): self.synced = True

        class FakeCuda:
            @staticmethod
            def current_stream(_device): return FakeStream()
            @staticmethod
            def is_available(): return False

        class FakeTorch:
            __version__ = "2.5.1+cu121"
            version = SimpleNamespace(cuda="12.1")
            cuda = FakeCuda()
            float32 = "float32"
            @staticmethod
            def device(value): return value

        class FakeLayer:
            type = "CONVOLUTION"
            num_outputs = 1
            def __init__(self, name): self.name = name; self.precision = None; self.output_types = []
            def set_output_type(self, _index, value): self.output_types.append(value)
            def get_output_type(self, _index): return self.output_types[0] if self.output_types else "unknown"

        class FakeNetwork:
            num_inputs = 1
            num_outputs = 1
            def __init__(self):
                self.layers = [FakeLayer(name) for name in target_names]
            def get_input(self, _index): return SimpleNamespace(shape=(1, 3, 640, 640))
            def get_output(self, _index): return SimpleNamespace(shape=plan["models"][model]["output_shape"] if output_ok else [1, 1, 1])
            @property
            def num_layers(self): return len(self.layers)
            def get_layer(self, index): return self.layers[index]

        class FakeConfig:
            def __init__(self): self.flags = set(); self.int8_calibrator = None
            def set_memory_pool_limit(self, _pool, value): self.workspace = value
            def set_flag(self, flag): self.flags.add(flag)
            def clear_flag(self, flag): self.flags.discard(flag)
            def set_timing_cache(self, _cache, ignore_mismatch): self.ignore_mismatch = ignore_mismatch; return True
            def create_timing_cache(self, value): return SimpleNamespace(serialize=lambda: b"timing", initial=value)

        class FakeEngine:
            def create_engine_inspector(self): return SimpleNamespace(get_engine_information=lambda _format: '{"layers": []}')

        class FakeTrt:
            __version__ = "10.16.1.11"
            float32 = "float32"
            class Logger:
                VERBOSE = "verbose"
                def __init__(self, _level): pass
            class LayerType: CONVOLUTION = "CONVOLUTION"
            class MemoryPoolType: WORKSPACE = "workspace"
            class BuilderFlag: TF32 = "tf32"; FP16 = "fp16"; INT8 = "int8"; OBEY_PRECISION_CONSTRAINTS = "obey"
            class ProfilingVerbosity: DETAILED = "detailed"
            class LayerInformationFormat: JSON = "json"
            class CalibrationAlgoType: MINMAX_CALIBRATION = "minmax"
            class IInt8Calibrator: pass
            class Builder:
                def __init__(self, _logger): pass
                def create_network(self, _flags): return FakeNetwork()
                def create_builder_config(self): return FakeConfig()
                def build_serialized_network(self, _network, config):
                    calibrator = getattr(config, "int8_calibrator", None)
                    if calibrator is not None:
                        if job["phase"] == "auxiliary_calibration":
                            self.assertion = calibrator.read_calibration_cache()
                            for _ in range(1024):
                                calibrator.get_batch(["input"])
                            calibrator.get_batch(["input"])
                            calibrator.write_calibration_cache(b"cache-" + model.encode())
                        else:
                            calibrator.read_calibration_cache()
                    return b"serialized-" + model.encode() + job["phase"].encode()
            class OnnxParser:
                def __init__(self, network, _logger): self.network = network; self.num_errors = 0
                def parse_from_file(self, _path): return parser_ok
                def get_error(self, _index): return "synthetic parser failure"
            class Runtime:
                def __init__(self, _logger): pass
                def deserialize_cuda_engine(self, _serialized): return FakeEngine()

        class FakeExporter:
            def __init__(self, **_kwargs): pass
            def get_int8_calibration_dataloader(self):
                return ({"im_file": [f"{index:04d}.jpg"], "img": FakeTensor()} for index in range(1024))

        records = []
        for index, row in enumerate(canonical):
            target_count = 2 if index < 1070 else 1
            records.append({"image": row["image"], "stem": row["stem"], "orig_shape": row["orig_shape"], "xyxy": [], "confidence": [], "class_id": [], "validator_input": {"imgsz": [640, 640], "ratio_pad": [[1.0, 1.0], [0.0, 0.0]], "prediction_xyxy": [], "target_xyxy": [[1.0, 1.0, 10.0, 10.0]] * target_count, "target_class_id": [0] * target_count}, "validator_statistics": {"tp": [], "confidence_dtype": "float64", "pred_class_dtype": "float64", "target_class_dtype": "int64"}})

        class FixtureBackend:
            def __init__(self, *args, **kwargs):
                self.format = "fixture"
                self.stride = 32
                self.end2end = False
            def warmup(self, *_args, **_kwargs):
                return None
            def __call__(self, _value):
                return None

        validator_module = SimpleNamespace(AutoBackend=FixtureBackend)

        class FakeValidator:
            def __init__(self, *args, **kwargs): self.capture_records = records; self.metrics = SimpleNamespace(box=SimpleNamespace(map50=0.0, map=0.0, mp=0.0, mr=0.0))
            def __call__(self, **_kwargs):
                if not capture_ok: raise RuntimeError("synthetic capture failure")
                backend = validator_module.AutoBackend(model="fixture")
                backend.warmup(None)
                for _ in records:
                    backend(None)
                return None

        class Lock:
            def __init__(self, *args): pass
            def __enter__(self): return self
            def __exit__(self, *_args): return False

        def load_xml(_path, requested): return {name: {"shape": [100, 100], "rows": [(0, [1.0, 1.0, 10.0, 10.0])] * (2 if index < 1070 else 1)} for index, name in enumerate(requested)}
        dependencies = {"np": SimpleNamespace(), "torch": FakeTorch, "trt": FakeTrt, "ultralytics": SimpleNamespace(__version__="8.4.102"), "Exporter": FakeExporter, "CaptureValidator": FakeValidator, "metric_summary": lambda _metrics: {"map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0}, "replay_statistics": lambda _records: {"map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0}, "GpuPhaseLock": Lock, "ensure_idle": lambda *_args: None, "parse_background_confirmations": lambda _args: [], "parse_desktop_confirmations": lambda _args: [], "snapshot": lambda *_args: {"gpu": "fixture"}, "rematch_native": lambda _records, _thresholds: {"status": "pass", "changed_tp_decisions": 0}, "validate_capture": lambda payload: (payload["records"], payload["iou_thresholds"]), "load_xml": load_xml, "SimpleNamespace": SimpleNamespace, "validator_module": validator_module}
        return plan, dependencies

    def test_production_backend_wrapper_skips_warmup_forward_and_counts_data_completion(self):
        class DummyBackend:
            def __call__(self, value):
                return value + 1
        class Wrapped(NoWarmupBackendAccounting, DummyBackend):
            pass
        Wrapped.reset_accounting()
        backend = Wrapped()
        backend.warmup("dummy")
        self.assertEqual(Wrapped.warmup_request_count, 1)
        self.assertEqual(Wrapped.warmup_forward_count, 0)
        self.assertEqual(backend(4), 5)
        self.assertEqual(Wrapped.forward_attempt_count, 1)
        self.assertEqual(Wrapped.forward_completed_count, 1)

    def test_normal_build_real_path_runs_both_models_all_three_phases(self):
        from run_precision_head_confirmation_server import read as read_state
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "study"
            out.mkdir()
            (out / "public").mkdir()
            identity = {"uuid": "GPU-fixture", "name": "fixture-gpu", "driver_version": "fixture-driver", "device": "0"}
            args = SimpleNamespace(device="0", confirm_desktop_process=[], confirm_background_process=[])
            all_states = []
            for model in ("yolov8n", "yolo26n"):
                jobs = [job for job in build_schedule() if job["model"] == model]
                aux_job = next(job for job in jobs if job["phase"] == "auxiliary_calibration" and job["selection"] == "U42")
                aux_plan, aux_deps = self.normal_path_fixture(root, model, aux_job)
                plan_path = root / f"{model}-plan.json"
                plan_path.write_text(json.dumps(aux_plan), encoding="utf-8")
                (out / "confirmation_plan.json").write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
                aux_state_path = out / "jobs" / f"{job_key(aux_job)}" / "child_state.json"
                aux_state_path.parent.mkdir(parents=True)
                child_state(aux_state_path, aux_job, plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())
                with patch("run_precision_head_confirmation_server.load_production_dependencies", return_value=aux_deps), patch("run_precision_head_confirmation_server._gpu_identity", return_value=identity):
                    build_real(aux_job, aux_plan, root, out, args, aux_state_path, external_boundary=False)
                finalize_child_state(aux_state_path)
                aux_state = read_state(aux_state_path)
                self.assertEqual(aux_state["calibration_batches"], 1024)
                self.assertEqual(aux_state["calibration_write_calls"], 1)
                self.assertTrue(aux_state["builder_completed"])
                all_states.append(aux_state)
                manifest = {"jobs": [{"job": aux_job, "state": aux_state}]}
                (out / "execution_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

                int8_job = next(job for job in jobs if job["phase"] == "scored_int8" and job["selection"] == "U42" and job["arm"] == "baseline_int8")
                int8_plan, int8_deps = self.normal_path_fixture(root, model, int8_job)
                int8_plan_path = root / f"{model}-int8-plan.json"
                int8_plan_path.write_text(json.dumps(int8_plan), encoding="utf-8")
                (out / "confirmation_plan.json").write_text(int8_plan_path.read_text(encoding="utf-8"), encoding="utf-8")
                int8_state_path = out / "jobs" / f"{job_key(int8_job)}" / "child_state.json"
                int8_state_path.parent.mkdir(parents=True)
                child_state(int8_state_path, int8_job, plan_sha256=hashlib.sha256(int8_plan_path.read_bytes()).hexdigest())
                with patch("run_precision_head_confirmation_server.load_production_dependencies", return_value=int8_deps), patch("run_precision_head_confirmation_server._gpu_identity", return_value=identity):
                    build_real(int8_job, int8_plan, root, out, args, int8_state_path, external_boundary=False)
                finalize_child_state(int8_state_path)
                int8_state = read_state(int8_state_path)
                self.assertEqual(int8_state["calibration_batches"], 0)
                self.assertEqual(int8_state["calibration_write_calls"], 0)
                self.assertGreaterEqual(int8_state["calibration_read_calls"], 1)
                self.assertTrue(int8_state["calibration_cache"]["immutable"])
                self.assertEqual(int8_state["forward_completed"], 1636)
                all_states.append(int8_state)

                fp16_job = next(job for job in jobs if job["phase"] == "scored_fp16" and job["repeat"] == 1)
                fp16_plan, fp16_deps = self.normal_path_fixture(root, model, fp16_job)
                fp16_plan_path = root / f"{model}-fp16-plan.json"
                fp16_plan_path.write_text(json.dumps(fp16_plan), encoding="utf-8")
                (out / "confirmation_plan.json").write_text(fp16_plan_path.read_text(encoding="utf-8"), encoding="utf-8")
                fp16_state_path = out / "jobs" / f"{job_key(fp16_job)}" / "child_state.json"
                fp16_state_path.parent.mkdir(parents=True)
                child_state(fp16_state_path, fp16_job, plan_sha256=hashlib.sha256(fp16_plan_path.read_bytes()).hexdigest())
                with patch("run_precision_head_confirmation_server.load_production_dependencies", return_value=fp16_deps), patch("run_precision_head_confirmation_server._gpu_identity", return_value=identity):
                    build_real(fp16_job, fp16_plan, root, out, args, fp16_state_path, external_boundary=False)
                finalize_child_state(fp16_state_path)
                fp16_state = read_state(fp16_state_path)
                self.assertEqual(fp16_state["calibration_batches"], 0)
                self.assertEqual(fp16_state["forward_completed"], 1636)
                self.assertEqual(fp16_state["warmup_forward_count"], 0)
                self.assertTrue(fp16_state["engine_inspector"]["path"].startswith("public/inspectors/"))
                all_states.append(fp16_state)
            self.assertEqual(len(all_states), 6)

    def test_normal_build_real_parser_and_output_failures_are_not_success(self):
        from run_precision_head_confirmation_server import run_child
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "study"
            out.mkdir()
            job = next(job for job in build_schedule() if job["model"] == "yolov8n" and job["phase"] == "scored_fp16")
            plan, deps = self.normal_path_fixture(root, "yolov8n", job, parser_ok=False)
            plan["schedule"] = {"jobs": build_schedule(), "sha256": canonical_json_sha256(build_schedule())}
            plan_path = root / "plan.json"
            job_path = root / "job.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            job_path.write_text(json.dumps(job), encoding="utf-8")
            args = SimpleNamespace(job_json=job_path, out_dir=out, plan=plan_path, repo=root, device="0", runtime_double=False, confirm_desktop_process=[], confirm_background_process=[])
            with patch("run_precision_head_confirmation_server.load_production_dependencies", return_value=deps), patch("run_precision_head_confirmation_server._gpu_identity", return_value={"uuid": "GPU-fixture", "name": "fixture-gpu", "driver_version": "fixture-driver", "device": "0"}):
                self.assertEqual(run_child(args), 1)
            states = list((out / "jobs").glob("*/child_state.json"))
            self.assertEqual(len(states), 1)
            state = json.loads(states[0].read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertIn("TensorRT ONNX parse failed", state["error"])

            out2 = root / "study-output-failure"
            out2.mkdir()
            plan2, deps2 = self.normal_path_fixture(root, "yolov8n", job, capture_ok=False)
            plan2["schedule"] = {"jobs": build_schedule(), "sha256": canonical_json_sha256(build_schedule())}
            plan2_path = root / "plan2.json"
            job2_path = root / "job2.json"
            plan2_path.write_text(json.dumps(plan2), encoding="utf-8")
            job2_path.write_text(json.dumps(job), encoding="utf-8")
            args2 = SimpleNamespace(job_json=job2_path, out_dir=out2, plan=plan2_path, repo=root, device="0", runtime_double=False, confirm_desktop_process=[], confirm_background_process=[])
            with patch("run_precision_head_confirmation_server.load_production_dependencies", return_value=deps2), patch("run_precision_head_confirmation_server._gpu_identity", return_value={"uuid": "GPU-fixture", "name": "fixture-gpu", "driver_version": "fixture-driver", "device": "0"}):
                self.assertEqual(run_child(args2), 1)
            state2 = json.loads(next((out2 / "jobs").glob("*/child_state.json")).read_text(encoding="utf-8"))
            self.assertEqual(state2["status"], "failed")
            self.assertIn("synthetic capture failure", state2["error"])
            self.assertEqual(state2["forward_completed"], 0)

    def test_gpu_identity_is_persisted_and_changed_identity_blocks(self):
        manifest = {}
        first = {"uuid": "GPU-A", "name": "Quadro RTX 8000", "driver_version": "595.71.05", "device": "0"}
        bind_gpu_identity(manifest, first)
        bind_gpu_identity(manifest, dict(first))
        self.assertEqual(manifest["gpu_identity"], first)
        with self.assertRaises(ContractError):
            bind_gpu_identity(manifest, {**first, "uuid": "GPU-B"})
        with self.assertRaises(ContractError):
            bind_gpu_identity({}, {"uuid": "GPU-A"})

    def test_calibration_verification_uses_producer_nested_images_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "data/processed/cctsdb2021_clean/train/images/00001.jpg"
            materialized = root / "calibration/U42/images/00001.jpg"
            source.parent.mkdir(parents=True)
            materialized.parent.mkdir(parents=True)
            source.write_bytes(b"calibration-image")
            materialized.write_bytes(source.read_bytes())
            yaml_path = root / "calibration/U42/calibration.yaml"
            yaml_path.write_text("path: .\n", encoding="utf-8")
            yaml_hash = __import__("hashlib").sha256(yaml_path.read_bytes()).hexdigest()
            image_hash = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
            plan = {"models": {}, "provenance_files": [], "dataset": {}, "calibration_producer": {"selections": [{"id": "U42", "materialization": {"yaml": str(yaml_path), "yaml_sha256": yaml_hash, "resolved_directory": str(yaml_path.parent), "image_bytes": [{"image": "train/images/00001.jpg", "source_sha256": image_hash, "materialized_sha256": image_hash}]}}]}}
            verify_plan_inputs(root, plan)
            materialized.write_bytes(b"changed")
            with self.assertRaises(ContractError):
                verify_plan_inputs(root, plan)
            materialized.write_bytes(source.read_bytes())
            plan["calibration_producer"]["selections"][0]["materialization"]["resolved_directory"] = str(root / "calibration/U42/wrong")
            with self.assertRaises(ContractError):
                verify_plan_inputs(root, plan)

    def test_canonical_capture_validation_projects_actual_readiness_records(self):
        readiness_path = REPO / "results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2/readiness_manifest.json"
        if not readiness_path.is_file():
            self.skipTest("accepted readiness manifest is not in this checkout")
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        records = readiness["dev_contract"]["canonical_reference"]["records"]
        plan = {"dataset": {"canonical_dev_reference": {"records": records}}}
        observed = [{"image": row["image"], "orig_shape": row["orig_shape"]} for row in records]
        validate_canonical_capture_records(observed, plan)
        bad = list(observed)
        bad[0] = {**bad[0], "orig_shape": [1, 1]}
        with self.assertRaises(ContractError):
            validate_canonical_capture_records(bad, plan)

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
            self.assertEqual(mapping["mapping_hash"], "fixture")
            self.assertEqual(len(mapping["bbox"]), 9)
            self.assertEqual(len(mapping["classification"]), 9)
            self.assertEqual(selected_targets(mapping, "baseline_int8"), [])
            self.assertEqual(selected_targets(mapping, "fp16"), [])
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
            plan, plan_path = self.runtime_plan(root)
            jobs = plan["schedule"]["jobs"]
            out = root / "run"
            out.mkdir()
            (out / "confirmation_plan.json").write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
            (out / "schedule.json").write_text(json.dumps({"schema_version": 1, "jobs": jobs, "sha256": canonical_json_sha256(jobs)}), encoding="utf-8")
            args = type("Args", (), {"repo": REPO, "plan": plan_path, "out_dir": out, "device": "0", "child_timeout": 30, "runtime_double": True, "confirm_desktop_process": [], "confirm_background_process": []})()
            self.assertEqual(run_server_parent(args), 0)
            manifest = json.loads((out / "execution_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed_review_required")
            self.assertEqual(len(manifest["jobs"]), 84)
            self.assertEqual(len(list(out.glob("jobs/*/cell_metrics.json"))), 78)
            self.assertTrue(all(item["state"].get("build_real_entry") for item in manifest["jobs"]))
            self.assertTrue(all(item["state"].get("release_boundary") == "run_child_after_build_real_return" for item in manifest["jobs"]))
            self.assertTrue(all(item["state"].get("forward_completed", 0) == 1636 for item in manifest["jobs"] if item["job"]["capture_required"]))
            self.assertTrue(all("external_plan_verified" in [row["stage"] for row in item["state"]["stage_history"]] for item in manifest["jobs"]))
            inspector_paths = [item["state"]["engine_inspector"]["path"] for item in manifest["jobs"]]
            self.assertEqual(len(inspector_paths), 84)
            self.assertTrue(all(path.startswith("public/inspectors/") and "/private/" not in f"/{path}" for path in inspector_paths))
            self.assertTrue(all((out / path).is_file() for path in inspector_paths))
            self.assertFalse((out / "private").exists())

    def test_gpu_identity_is_persisted_across_children_and_changed_identity_stops_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = {"uuid": "GPU-A", "name": "Quadro RTX 8000", "driver_version": "595.71.05", "device": "0"}
            second = {**first, "uuid": "GPU-B"}
            plan, plan_path = self.runtime_plan(root, identity_by_sequence={"1": first, "2": second})
            jobs = plan["schedule"]["jobs"]
            out = root / "run"
            args = type("Args", (), {"repo": REPO, "plan": plan_path, "out_dir": out, "device": "0", "child_timeout": 30, "runtime_double": True, "confirm_desktop_process": [], "confirm_background_process": []})()
            self.assertNotEqual(run_server_parent(args), 0)
            manifest = json.loads((out / "execution_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "incomplete_blocked")
            self.assertEqual(manifest["gpu_identity"], first)
            self.assertEqual(len(manifest["jobs"]), 2)
            self.assertIn("GPU identity changed", manifest["jobs"][1]["state"].get("contract_error", ""))

    def test_external_runtime_double_artifacts_complete_production_analyzer_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, plan_path = self.runtime_plan(root)
            jobs = plan["schedule"]["jobs"]
            out = root / "run"
            args = type("Args", (), {"repo": REPO, "plan": plan_path, "out_dir": out, "device": "0", "child_timeout": 30, "runtime_double": True, "confirm_desktop_process": [], "confirm_background_process": []})()
            self.assertEqual(run_server_parent(args), 0)
            (out / "confirmation_plan.json").write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
            (out / "schedule.json").write_text(json.dumps({"schema_version": 1, "jobs": jobs, "sha256": canonical_json_sha256(jobs)}), encoding="utf-8")
            xml_path = root / "xml.zip"
            def annotation(index: int) -> str:
                count = 2 if index < 1070 else 1
                objects = "".join(
                    f"<object><name>{'prohibitory' if object_index == 0 else 'mandatory'}</name><bndbox><xmin>{1 + object_index * 20}</xmin><ymin>1</ymin><xmax>{10 + object_index * 20}</xmax><ymax>10</ymax></bndbox></object>"
                    for object_index in range(count)
                )
                return f"<annotation><size><width>100</width><height>100</height></size>{objects}</annotation>"
            with zipfile.ZipFile(xml_path, "w") as archive:
                for index in range(1636):
                    archive.writestr(f"{index:04d}.xml", annotation(index))
            from analyze_precision_head_confirmation import analyze
            with self.assertRaises(ContractError):
                analyze(out, root / "production-analysis", xml_path)
            summary = analyze(out, root / "analysis", xml_path, allow_test_double=True, bootstrap_draws=2)
            self.assertEqual(summary["status"], "completed_descriptive_analysis")
            self.assertEqual(len(summary["all_cell_points"]), 96)
            self.assertEqual(len(summary["fp16_points"]), 8)
            self.assertTrue((root / "analysis" / "analysis_summary.json").is_file())

    def test_parent_timeout_preserves_blocked_partial_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, plan_path = self.runtime_plan(root)
            out = root / "run"
            args = type("Args", (), {"repo": REPO, "plan": plan_path, "out_dir": out, "device": "0", "child_timeout": 0, "runtime_double": True, "confirm_desktop_process": [], "confirm_background_process": []})()
            self.assertEqual(run_server_parent(args), 124)
            manifest = json.loads((out / "execution_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "incomplete_blocked")
            self.assertEqual(manifest["jobs"][0]["state"]["status"], "timeout")
            self.assertTrue((out / "jobs").exists())

    def test_confirmation_study_rejects_competing_background_workload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, plan_path = self.runtime_plan(root)
            args = type("Args", (), {"repo": REPO, "plan": plan_path, "out_dir": root / "run", "device": "0", "child_timeout": 30, "runtime_double": True, "confirm_desktop_process": [], "confirm_background_process": ["123=python"]})()
            with self.assertRaises(ContractError):
                run_server_parent(args)

    def test_analyzer_rejects_missing_cell_and_provenance_mutation(self):
        from analyze_precision_head_confirmation import _cell_files
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            jobs = build_schedule()
            (root / "confirmation_plan.json").write_text(json.dumps({"schedule": {"jobs": jobs, "sha256": canonical_json_sha256(jobs)}, "models": {}}), encoding="utf-8")
            (root / "schedule.json").write_text(json.dumps({"jobs": jobs, "sha256": canonical_json_sha256(jobs)}), encoding="utf-8")
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
