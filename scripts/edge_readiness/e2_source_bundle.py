#!/usr/bin/env python3
"""CPU-only frozen YOLO11n source/reference bundle producer.

The module has no ML imports at import time.  The real runtime is loaded only
after a complete dependency preflight and is forced to CPU.  Tests inject a
small runtime double so lifecycle, hashes and contracts are testable without
copying the checkpoint or running a model locally.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge_readiness.edge_errors import AdapterError
from edge_readiness.e2_source_fixture import FIXTURE_IDS, FIXTURE_RELATIVE_PATHS, verify_fixture

INPUT_SHAPE = (1, 3, 640, 640)
OUTPUT_SHAPE = (1, 7, 8400)
YOLO11N_CHECKPOINT_PATH = "results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt"
YOLO11N_CHECKPOINT_SHA256 = "3e5fc7a2148c16539cd9fb7cc7cacd81a4eec1dfc28143cdf9b6dcd872ba4ab8"
EXPECTED_CLASS_ORDER = ("prohibitory", "mandatory", "warning")
SOURCE_ABS_TOL = 1e-5
SOURCE_REL_TOL = 1e-4
REQUIRED_MODULES = ("torch", "ultralytics", "onnx", "onnxruntime", "onnxslim", "cv2", "PIL")
EXPORT_OPTIONS = {
    "format": "onnx",
    "device": "cpu",
    "half": False,
    "opset": 17,
    "imgsz": 640,
    "batch": 1,
    "dynamic": False,
    "nms": False,
    "simplify": True,
    "int8": False,
}


class SourceBundleError(AdapterError):
    pass


@dataclass
class TensorArtifact:
    payload: bytes
    shape: Tuple[int, ...]
    dtype: str
    byteorder: str
    finite: bool
    value: Any = field(default=None, repr=False, compare=False)

    @property
    def nbytes(self) -> int:
        return len(self.payload)


@dataclass
class BundleCounters:
    fixture_images_attempted: int = 0
    fixture_images_verified: int = 0
    inputs_attempted: int = 0
    inputs_completed: int = 0
    native_forwards_attempted: int = 0
    native_forwards_completed: int = 0
    export_invocations_attempted: int = 0
    export_invocations_completed: int = 0
    exporter_internal_forwards: int = 0
    ort_forwards_attempted: int = 0
    ort_forwards_completed: int = 0
    comparisons_attempted: int = 0
    comparisons_completed: int = 0

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "x", encoding="utf-8", newline="\n") as handle:
        if isinstance(payload, str):
            handle.write(payload)
        else:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")


def _error_dict(exc: BaseException) -> Dict[str, Any]:
    if isinstance(exc, AdapterError):
        return exc.as_dict()["error"]
    return {"code": type(exc).__name__, "message": str(exc), "details": {}}


def _ensure_finite(value: Any) -> bool:
    return bool(value)


def _validate_tensor(tensor: TensorArtifact, expected_shape: Tuple[int, ...], label: str) -> None:
    if tensor.shape != expected_shape:
        raise SourceBundleError("SOURCE_TENSOR_SHAPE_MISMATCH", "{} shape differs from the frozen contract".format(label), {"expected": list(expected_shape), "observed": list(tensor.shape)})
    if tensor.dtype != "float32" or tensor.byteorder != "little":
        raise SourceBundleError("SOURCE_TENSOR_DTYPE_MISMATCH", "{} must be little-endian float32".format(label), {"dtype": tensor.dtype, "byteorder": tensor.byteorder})
    expected_nbytes = 4
    for dimension in expected_shape:
        expected_nbytes *= dimension
    if len(tensor.payload) != expected_nbytes:
        raise SourceBundleError("SOURCE_TENSOR_BYTES_MISMATCH", "{} byte length differs from shape".format(label), {"expected": expected_nbytes, "observed": len(tensor.payload)})
    if not _ensure_finite(tensor.finite):
        raise SourceBundleError("SOURCE_TENSOR_NONFINITE", "{} contains non-finite values".format(label))


def pack_letterboxed_bgr(pixels_bgr: bytes, width: int, height: int) -> TensorArtifact:
    """Pack an already letterboxed 640x640 BGR image like the source path."""
    if width != 640 or height != 640 or len(pixels_bgr) != width * height * 3:
        raise SourceBundleError("LETTERBOX_CONTRACT_MISMATCH", "source packer requires a 640x640 BGR image")
    floats = bytearray()
    plane = width * height
    for channel in (2, 1, 0):
        for index in range(plane):
            floats.extend(struct.pack("<f", pixels_bgr[index * 3 + channel] / 255.0))
    return TensorArtifact(bytes(floats), INPUT_SHAPE, "float32", "little", True)


def tensor_record(tensor: TensorArtifact, private_path: Path) -> Dict[str, Any]:
    return {"path": str(private_path.resolve()), "nbytes": tensor.nbytes, "sha256": hashlib.sha256(tensor.payload).hexdigest(), "shape": list(tensor.shape), "dtype": tensor.dtype, "byteorder": tensor.byteorder, "finite": tensor.finite}


def compare_source_outputs(reference: TensorArtifact, observed: TensorArtifact) -> Dict[str, Any]:
    _validate_tensor(reference, OUTPUT_SHAPE, "native reference")
    _validate_tensor(observed, OUTPUT_SHAPE, "ONNX output")
    count = len(reference.payload) // 4
    reference_values = struct.unpack("<{}f".format(count), reference.payload)
    observed_values = struct.unpack("<{}f".format(count), observed.payload)
    maximum_abs = 0.0
    maximum_relative = 0.0
    failures = []
    for index, (expected, actual) in enumerate(zip(reference_values, observed_values)):
        absolute = abs(actual - expected)
        relative = absolute / max(abs(expected), 1e-30)
        limit = SOURCE_ABS_TOL + SOURCE_REL_TOL * abs(expected)
        maximum_abs = max(maximum_abs, absolute)
        maximum_relative = max(maximum_relative, relative)
        if absolute > limit:
            if len(failures) < 20:
                failures.append({"flat_index": index, "reference": expected, "observed": actual, "abs_error": absolute, "limit": limit})
    return {"status": "pass" if not failures else "fail", "equation": "abs(observed-reference) <= 1e-5 + 1e-4*abs(reference)", "elements": count, "failing_elements": len(failures), "max_abs_error": maximum_abs, "max_relative_error": maximum_relative, "bounded_failures": failures}


def _first_tensor(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                shape = tuple(int(dimension) for dimension in item.shape)
            except AttributeError:
                continue
            if shape:
                return item
    return value


class SourceRuntime(Protocol):
    def prepare(self) -> Dict[str, Any]: ...
    def load_model(self, checkpoint: Path) -> Any: ...
    def model_flags(self, model: Any) -> Dict[str, Any]: ...
    def preprocess(self, image_path: Path) -> TensorArtifact: ...
    def native_forward(self, model: Any, tensor: TensorArtifact) -> TensorArtifact: ...
    def export(self, checkpoint: Path, export_dir: Path, options: Dict[str, Any]) -> Tuple[Path, int]: ...
    def validate_onnx(self, onnx_path: Path) -> Dict[str, Any]: ...
    def open_ort(self, onnx_path: Path) -> Any: ...
    def ort_forward(self, session: Any, tensor: TensorArtifact) -> TensorArtifact: ...


class UltralyticsSourceRuntime:
    """Actual source runtime; all third-party imports are deferred to prepare."""

    def __init__(self) -> None:
        self.modules: Dict[str, Any] = {}

    def prepare(self) -> Dict[str, Any]:
        missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
        if missing:
            raise SourceBundleError("SOURCE_DEPENDENCY_MISSING", "complete source environment is unavailable; refusing auto-install", {"missing": missing, "required": list(REQUIRED_MODULES)})
        for name in REQUIRED_MODULES:
            self.modules[name] = importlib.import_module(name)
        torch = self.modules["torch"]
        if torch.cuda.is_available():
            raise SourceBundleError("CPU_PROVIDER_REQUIRED", "CUDA is visible in the source environment after CPU gating")
        versions = {}
        for name, module in self.modules.items():
            versions[name] = str(getattr(module, "__version__", "unknown"))
        versions["python"] = sys.version.split()[0]
        versions["ort_providers_required"] = ["CPUExecutionProvider"]
        return versions

    def load_model(self, checkpoint: Path) -> Any:
        YOLO = self.modules["ultralytics"].YOLO
        model = YOLO(str(checkpoint))
        model.model.to("cpu")
        model.model.eval()
        return model

    def model_flags(self, model: Any) -> Dict[str, Any]:
        detect = model.model.model[-1] if hasattr(model.model, "model") else model.model[-1]
        names = getattr(model, "names", getattr(model.model, "names", {}))
        if isinstance(names, dict):
            class_order = [str(names[index]) for index in sorted(names)]
        else:
            class_order = [str(item) for item in names]
        flags: Dict[str, Any] = {"class_order": class_order, "nc": getattr(detect, "nc", None), "reg_max": getattr(detect, "reg_max", None), "end2end": getattr(detect, "end2end", None), "export": getattr(detect, "export", None)}
        stride = getattr(detect, "stride", None)
        if stride is not None:
            flags["stride"] = stride.detach().cpu().tolist() if hasattr(stride, "detach") else list(stride)
        return flags

    def preprocess(self, image_path: Path) -> TensorArtifact:
        np = self.modules["numpy"] if "numpy" in self.modules else importlib.import_module("numpy")
        cv2 = self.modules["cv2"]
        LetterBox = self.modules["ultralytics"].data.augment.LetterBox
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SourceBundleError("SOURCE_IMAGE_DECODE_FAILED", "OpenCV could not decode fixture image", {"path": str(image_path)})
        letterbox = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, center=True, stride=32, padding_value=114, interpolation=cv2.INTER_LINEAR)
        letterboxed_bgr = letterbox(image=image)
        rgb = letterboxed_bgr[:, :, ::-1]
        array = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...], dtype=np.float32) / 255.0
        packed = pack_letterboxed_bgr(np.ascontiguousarray(letterboxed_bgr).tobytes(order="C"), 640, 640)
        return TensorArtifact(packed.payload, packed.shape, packed.dtype, packed.byteorder, packed.finite, array)

    def _numpy_tensor(self, array: Any) -> TensorArtifact:
        np = self.modules["numpy"]
        array = np.ascontiguousarray(array, dtype=np.float32)
        return TensorArtifact(array.tobytes(order="C"), tuple(int(item) for item in array.shape), "float32", "little", bool(np.isfinite(array).all()), array)

    def native_forward(self, model: Any, tensor: TensorArtifact) -> TensorArtifact:
        torch = self.modules["torch"]
        with torch.inference_mode():
            raw = model.model(torch.from_numpy(tensor.value))
        output = _first_tensor(raw)
        if tuple(int(item) for item in output.shape) != OUTPUT_SHAPE:
            raise SourceBundleError("NATIVE_OUTPUT_CONTRACT_MISMATCH", "native pre-NMS output is not [1,7,8400]", {"observed": list(output.shape)})
        return self._numpy_tensor(output.detach().cpu().numpy())

    def export(self, checkpoint: Path, export_dir: Path, options: Dict[str, Any]) -> Tuple[Path, int]:
        YOLO = self.modules["ultralytics"].YOLO
        export_dir.mkdir(parents=True, exist_ok=False)
        exporter = YOLO(str(checkpoint))
        counter = {"calls": 0}
        hook = None
        if hasattr(exporter.model, "register_forward_pre_hook"):
            hook = exporter.model.register_forward_pre_hook(lambda *_args: counter.__setitem__("calls", counter["calls"] + 1))
        try:
            result = exporter.export(project=str(export_dir), name="onnx", exist_ok=False, **options)
        finally:
            if hook is not None:
                hook.remove()
        path = Path(str(result)).resolve()
        private_root = export_dir.resolve()
        if private_root not in path.parents or path.suffix.lower() != ".onnx" or not path.is_file():
            raise SourceBundleError("EXPORT_PATH_INVALID", "exporter did not produce an ONNX file inside the private root", {"observed": str(path), "private_root": str(private_root)})
        return path, counter["calls"]

    def validate_onnx(self, onnx_path: Path) -> Dict[str, Any]:
        onnx = self.modules["onnx"]
        model = onnx.load(str(onnx_path), load_external_data=False)
        onnx.checker.check_model(model)
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise SourceBundleError("ONNX_IO_CONTRACT_MISMATCH", "ONNX must expose exactly one input and one output")
        input_value = model.graph.input[0]
        output_value = model.graph.output[0]
        input_shape = [dimension.dim_value for dimension in input_value.type.tensor_type.shape.dim]
        output_shape = [dimension.dim_value for dimension in output_value.type.tensor_type.shape.dim]
        if input_shape != list(INPUT_SHAPE) or output_shape != list(OUTPUT_SHAPE):
            raise SourceBundleError("ONNX_SHAPE_CONTRACT_MISMATCH", "ONNX I/O shape differs from the frozen contract", {"input": input_shape, "output": output_shape})
        if input_value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT or output_value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise SourceBundleError("ONNX_DTYPE_CONTRACT_MISMATCH", "ONNX input/output must be float32")
        return {"input_name": input_value.name, "output_name": output_value.name, "input_shape": input_shape, "output_shape": output_shape, "input_dtype": "float32", "output_dtype": "float32", "checker": "pass"}

    def open_ort(self, onnx_path: Path) -> Any:
        ort = self.modules["onnxruntime"]
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise SourceBundleError("ORT_PROVIDER_CONTRACT_MISMATCH", "ORT session is not CPU-only", {"providers": session.get_providers()})
        return session

    def ort_forward(self, session: Any, tensor: TensorArtifact) -> TensorArtifact:
        np = self.modules["numpy"]
        output = session.run(None, {session.get_inputs()[0].name: tensor.value})[0]
        return self._numpy_tensor(np.asarray(output))


class SourceBundleRunner:
    def __init__(self, source_root: Path, out_dir: Path, runtime: SourceRuntime, *, commit: str = "unknown") -> None:
        self.source_root = source_root.resolve()
        self.out_dir = out_dir.resolve()
        self.runtime = runtime
        self.commit = commit
        self.counters = BundleCounters()
        self.events: List[str] = []

    def _log(self, message: str) -> None:
        self.events.append(message)

    def _plan(self) -> Dict[str, Any]:
        return {"schema_version": "e2l1-source-bundle-v1", "status": "planned", "target": "source-cpu-only", "commit": self.commit, "source_root": str(self.source_root), "fixture_ids": list(FIXTURE_IDS), "input_shape": list(INPUT_SHAPE), "output_shape": list(OUTPUT_SHAPE), "dtype": "float32", "export_options": EXPORT_OPTIONS, "source_equivalence": {"absolute": SOURCE_ABS_TOL, "relative": SOURCE_REL_TOL}, "execution_policy": {"device": "cpu", "cpu_threads": 2, "cuda_visible_devices": "", "ort_providers": ["CPUExecutionProvider"], "downloads": False, "auto_install": False, "test_split_used": False, "dev_split_used": False}, "private_policy": "checkpoint copy, ONNX and tensor bytes remain only under private/ and are never published"}

    def _write_failure(self, exc: BaseException) -> Dict[str, Any]:
        failure = {"schema_version": "e2l1-source-bundle-v1", "status": "failed", "real_device_execution": False, "commit": self.commit, "attempted_completed": self.counters.as_dict(), "events": self.events, "error": _error_dict(exc)}
        try:
            write_once(self.out_dir / "public" / "failure.json", failure)
            write_once(self.out_dir / "public" / "run.log", "\n".join(self.events) + "\n")
        except FileExistsError:
            failure["artifact_write_error"] = "failure artifact path already existed; root was not overwritten"
        return failure

    def run(self) -> Tuple[int, Dict[str, Any]]:
        if self.out_dir.exists():
            raise ValueError("refusing existing source bundle root: {}".format(self.out_dir))
        self.out_dir.mkdir(parents=True, exist_ok=False)
        private = self.out_dir / "private"
        public = self.out_dir / "public"
        public.mkdir()
        try:
            write_once(public / "plan.json", self._plan())
            self._log("plan_written")
            fixture_before = verify_fixture(self.source_root)
            self.counters.fixture_images_attempted = len(FIXTURE_IDS)
            self.counters.fixture_images_verified = len(FIXTURE_IDS)
            checkpoint = self.source_root / YOLO11N_CHECKPOINT_PATH
            if not checkpoint.is_file() or file_sha256(checkpoint) != YOLO11N_CHECKPOINT_SHA256:
                raise SourceBundleError("CHECKPOINT_HASH_MISMATCH", "frozen YOLO11n checkpoint is missing or differs from accepted SHA-256", {"path": str(checkpoint), "expected": YOLO11N_CHECKPOINT_SHA256, "observed": file_sha256(checkpoint) if checkpoint.is_file() else None})
            checkpoint_private = private / "checkpoint" / "best.pt"
            checkpoint_private.parent.mkdir(parents=True)
            shutil.copy2(str(checkpoint), str(checkpoint_private))
            if file_sha256(checkpoint_private) != YOLO11N_CHECKPOINT_SHA256:
                raise SourceBundleError("CHECKPOINT_COPY_HASH_MISMATCH", "private checkpoint copy differs from accepted checkpoint")
            environment = self.runtime.prepare()
            native_model = self.runtime.load_model(checkpoint_private)
            flags = self.runtime.model_flags(native_model)
            if tuple(flags.get("class_order", ())) != EXPECTED_CLASS_ORDER:
                raise SourceBundleError("CLASS_ORDER_CONTRACT_MISMATCH", "model class order differs from the frozen source contract", {"expected": list(EXPECTED_CLASS_ORDER), "observed": flags.get("class_order")})
            inputs: Dict[str, TensorArtifact] = {}
            native_records: Dict[str, Dict[str, Any]] = {}
            native_output_dir = private / "native_reference"
            input_dir = private / "inputs"
            native_output_dir.mkdir()
            input_dir.mkdir()
            for image_id, relative in zip(FIXTURE_IDS, FIXTURE_RELATIVE_PATHS):
                self.counters.inputs_attempted += 1
                image_path = self.source_root / relative
                tensor = self.runtime.preprocess(image_path)
                _validate_tensor(tensor, INPUT_SHAPE, "input tensor {}".format(image_id))
                input_path = input_dir / (image_id + ".bin")
                input_path.write_bytes(tensor.payload)
                inputs[image_id] = tensor
                self.counters.inputs_completed += 1
                self.counters.native_forwards_attempted += 1
                native = self.runtime.native_forward(native_model, tensor)
                _validate_tensor(native, OUTPUT_SHAPE, "native output {}".format(image_id))
                native_path = native_output_dir / (image_id + ".bin")
                native_path.write_bytes(native.payload)
                native_records[image_id] = {"input": tensor_record(tensor, input_path), "native": tensor_record(native, native_path)}
                self.counters.native_forwards_completed += 1
            export_dir = private / "onnx_export"
            self.counters.export_invocations_attempted += 1
            onnx_path, internal_calls = self.runtime.export(checkpoint_private, export_dir, dict(EXPORT_OPTIONS))
            self.counters.export_invocations_completed += 1
            self.counters.exporter_internal_forwards = internal_calls
            onnx_contract = self.runtime.validate_onnx(onnx_path)
            ort_session = self.runtime.open_ort(onnx_path)
            ort_dir = private / "onnx_reference"
            ort_dir.mkdir()
            comparisons: Dict[str, Any] = {}
            for image_id in FIXTURE_IDS:
                self.counters.ort_forwards_attempted += 1
                observed = self.runtime.ort_forward(ort_session, inputs[image_id])
                _validate_tensor(observed, OUTPUT_SHAPE, "ONNX output {}".format(image_id))
                ort_path = ort_dir / (image_id + ".bin")
                ort_path.write_bytes(observed.payload)
                self.counters.ort_forwards_completed += 1
                self.counters.comparisons_attempted += 1
                comparison = compare_source_outputs(_tensor_from_record_path(native_records[image_id]["native"]["path"], native_records[image_id]["native"]), observed)
                comparisons[image_id] = comparison
                self.counters.comparisons_completed += 1
                native_records[image_id]["onnx"] = tensor_record(observed, ort_path)
            fixture_after = verify_fixture(self.source_root)
            if fixture_before["sequence_sha256"] != fixture_after["sequence_sha256"]:
                raise SourceBundleError("FIXTURE_CHANGED_DURING_RUN", "fixture sequence changed between protected hash checks")
            if any(item["status"] != "pass" for item in comparisons.values()):
                raise SourceBundleError("SOURCE_EQUIVALENCE_FAILED", "native and ONNX CPU outputs differ under the predeclared equation", {"comparisons": comparisons})
            manifest = {"schema_version": "e2l1-source-bundle-v1", "status": "complete", "real_device_execution": False, "commit": self.commit, "checkpoint": {"source_path": str(checkpoint), "private_path": str(checkpoint_private.resolve()), "sha256": YOLO11N_CHECKPOINT_SHA256}, "fixture": fixture_before, "fixture_after": fixture_after, "environment": environment, "model_flags": flags, "onnx": {"path": str(onnx_path), "sha256": file_sha256(onnx_path), "contract": onnx_contract}, "export": {"options": EXPORT_OPTIONS, "invocations": 1, "internal_forward_calls": internal_calls}, "records": native_records, "comparisons": comparisons, "attempted_completed": self.counters.as_dict(), "private_retention": {"root": str(private.resolve()), "published": False}}
            report = "# L1A-012 source bundle\n\nStatus: complete; CPU-only.\n\n" + json.dumps({"commit": self.commit, "counters": self.counters.as_dict(), "comparisons": comparisons}, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            index = {"schema_version": "e2l1-source-bundle-index-v1", "status": "complete", "public_artifacts": ["plan.json", "manifest.json", "report.md", "run.log", "index.json"], "private_root": str(private.resolve())}
            write_once(public / "report.md", report)
            write_once(public / "run.log", "\n".join(self.events + ["bundle_complete"]) + "\n")
            write_once(public / "index.json", index)
            write_once(public / "manifest.json", manifest)
            return 0, manifest
        except BaseException as exc:
            self._log("failed:{}".format(type(exc).__name__))
            failure = self._write_failure(exc)
            return 2, failure


def _tensor_from_record_path(path: str, record: Dict[str, Any]) -> TensorArtifact:
    payload = Path(path).read_bytes()
    return TensorArtifact(payload, tuple(int(item) for item in record["shape"]), record["dtype"], record["byteorder"], bool(record["finite"]))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CPU-only frozen YOLO11n source/reference bundle")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--commit", default="unknown")
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["ULTRALYTICS_AUTOUPDATE"] = "false"
    runner = SourceBundleRunner(args.source_root, args.out_dir, UltralyticsSourceRuntime(), commit=args.commit)
    code, result = runner.run()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
