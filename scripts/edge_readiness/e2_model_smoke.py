#!/usr/bin/env python3
"""E2 target-model diagnostic workflow.

The workflow consumes one hash-bound private source bundle and keeps three
outcomes separate: source native-vs-ONNX strict equivalence, E2-vs-ONNX, and
E2-vs-native.  Device execution is disabled unless ``--execute-real-device``
is explicitly supplied.  TensorRT/CUDA imports are lazy and the production
target path uses the existing parser/builder, TensorRTProvider, OwnedBuffers,
and JetsonRuntimeAdapter boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge_readiness.edge_errors import AdapterError
from edge_readiness.e2_output_compare import ComparisonPolicy, DomainTolerance, compare_output0
from edge_readiness.jetson_adapter import (
    AdapterConfig,
    EngineDescriptor,
    HostTensor,
    JetsonRuntimeAdapter,
    make_host_tensor,
)
from edge_readiness.jetson_runtime_provider import OwnedBuffers, TensorRTProvider
from edge_readiness.cuda_runtime_owner import CudaRuntime, CudaRuntimeMemoryOwner


TARGET_ID = "E2"
EXPECTED_RUNTIME_PREFIX = "8.5.2.2"
EXPECTED_HARDWARE = "Jetson Xavier NX"
SOURCE_MANIFEST_SCHEMA = "e2l1-source-bundle-v1"
SOURCE_ONNX_SHA256 = "bd20b36d640c502358c44edbbde51f05267eda2518b0c0a4cfe84ca18a00d4b7"
FIXTURE_IDS = ("00006", "00009", "00028")
INPUT_SHAPE = (1, 3, 640, 640)
OUTPUT_SHAPE = (1, 7, 8400)
OUTPUT_ELEMENTS = 1 * 7 * 8400
INPUT_NBYTES = 1 * 3 * 640 * 640 * 4
OUTPUT_NBYTES = 1 * 7 * 8400 * 4
WORKSPACE_BYTES = 1024 * 1024 * 1024
STRICT_SOURCE_POLICY = ComparisonPolicy(
    source_compute_mode="fp32_reference",
    target_compute_mode="onnx_fp32",
    source_dtype="float32",
    target_binding_dtype="float32",
    box_tolerance=DomainTolerance(absolute=1e-5, relative=1e-4),
    score_tolerance=DomainTolerance(absolute=1e-5, relative=1e-4),
)
TARGET_POLICY = ComparisonPolicy()


class ModelSmokeError(AdapterError):
    pass


class TargetTimeout(ModelSmokeError):
    pass


class TargetExecutionUnknown(ModelSmokeError):
    pass


@dataclass(frozen=True)
class TensorReference:
    role: str
    image_id: str
    path: Path
    shape: Tuple[int, ...]
    dtype: str
    byteorder: str
    nbytes: int
    sha256: str


@dataclass
class SourceBundle:
    root: Path
    manifest_path: Path
    manifest: Dict[str, Any]
    onnx: Path
    inputs: Dict[str, TensorReference]
    native: Dict[str, TensorReference]
    onnx_outputs: Dict[str, TensorReference]
    source_comparisons: Dict[str, Dict[str, Any]]
    allowlist: List[Dict[str, Any]]


@dataclass(frozen=True)
class EngineArtifact:
    path: Path
    sha256: str
    nbytes: int
    runtime_version: str
    build_contract: Dict[str, Any]


@dataclass
class SmokeCounters:
    parse_attempted: int = 0
    parse_completed: int = 0
    build_attempted: int = 0
    build_completed: int = 0
    engine_load_attempted: int = 0
    engine_load_completed: int = 0
    enqueues_attempted: int = 0
    enqueues_completed: int = 0
    output_copies_attempted: int = 0
    output_copies_completed: int = 0
    comparisons_attempted: int = 0
    comparisons_completed: int = 0

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


@dataclass
class StageBudget:
    name: str
    seconds: float
    started: float = field(default_factory=time.monotonic)

    def check(self) -> None:
        if self.seconds <= 0:
            raise ModelSmokeError("TIMEOUT_INVALID", "stage timeout must be positive", {"stage": self.name})
        if time.monotonic() - self.started > self.seconds:
            raise TargetTimeout("STAGE_TIMEOUT", "stage deadline exceeded", {"stage": self.name, "seconds": self.seconds})


class TargetExecution(Protocol):
    descriptor: EngineDescriptor

    def infer(self, host_input: HostTensor) -> Dict[str, HostTensor]: ...

    def close(self) -> None: ...


class TargetRuntime(Protocol):
    def preflight(self) -> Dict[str, Any]: ...

    def build(self, onnx_path: Path, engine_path: Path, workspace_bytes: int) -> EngineArtifact: ...

    def open_execution(self, engine: EngineArtifact) -> TargetExecution: ...


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        if isinstance(payload, str):
            handle.write(payload)
        else:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")


def _error_dict(exc: BaseException) -> Dict[str, Any]:
    if isinstance(exc, AdapterError):
        return exc.as_dict()["error"]
    return {"code": type(exc).__name__, "message": str(exc), "details": {}}


def _private_path(raw: Any, bundle_root: Path) -> Path:
    if not isinstance(raw, str):
        raise ModelSmokeError("BUNDLE_PATH_INVALID", "manifest path must be a string")
    normalized = raw.replace("\\", "/")
    marker = "/private/"
    marker_index = normalized.find(marker)
    if marker_index < 0:
        if normalized.startswith("private/"):
            relative = normalized[len("private/"):]
        else:
            raise ModelSmokeError("BUNDLE_PATH_OUTSIDE_PRIVATE", "manifest path is not explicitly under private/")
    else:
        relative = normalized[marker_index + len(marker):]
    relative_path = Path(relative)
    if relative_path.is_absolute() or not relative or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ModelSmokeError("BUNDLE_PATH_TRAVERSAL", "manifest private path is not a normalized relative path", {"path": raw})
    private_root = (bundle_root / "private").resolve()
    candidate = (private_root / relative_path).resolve(strict=False)
    if private_root not in candidate.parents:
        raise ModelSmokeError("BUNDLE_PATH_OUTSIDE_PRIVATE", "manifest path escapes private root", {"path": raw})
    current = bundle_root.resolve()
    for part in (Path("private") / relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise ModelSmokeError("BUNDLE_SYMLINK_REJECTED", "private bundle path contains a symlink", {"path": str(current)})
    if not candidate.is_file():
        raise ModelSmokeError("BUNDLE_FILE_MISSING", "manifest-bound private file is missing", {"path": str(candidate)})
    return candidate


def _tensor_reference(record: Any, role: str, image_id: str, bundle_root: Path, expected_shape: Tuple[int, ...]) -> TensorReference:
    if not isinstance(record, dict):
        raise ModelSmokeError("BUNDLE_RECORD_INVALID", "manifest tensor record must be an object", {"role": role, "image_id": image_id})
    path = _private_path(record.get("path"), bundle_root)
    shape = tuple(record.get("shape", ()))
    if shape != expected_shape or record.get("dtype") != "float32" or record.get("byteorder") != "little":
        raise ModelSmokeError("BUNDLE_TENSOR_CONTRACT_MISMATCH", "manifest tensor does not match frozen float32 contract", {"role": role, "image_id": image_id, "shape": list(shape), "dtype": record.get("dtype"), "byteorder": record.get("byteorder")})
    expected_nbytes = INPUT_NBYTES if expected_shape == INPUT_SHAPE else OUTPUT_NBYTES
    observed_hash = file_sha256(path)
    if path.stat().st_size != expected_nbytes or record.get("nbytes") != expected_nbytes or record.get("sha256") != observed_hash:
        raise ModelSmokeError("BUNDLE_TENSOR_HASH_MISMATCH", "manifest tensor size or SHA-256 does not match private bytes", {"role": role, "image_id": image_id, "path": str(path), "expected_nbytes": expected_nbytes, "observed_nbytes": path.stat().st_size, "expected_sha256": record.get("sha256"), "observed_sha256": observed_hash})
    payload = path.read_bytes()
    values = struct.unpack("<{}f".format(expected_nbytes // 4), payload)
    if not all(math.isfinite(value) for value in values):
        raise ModelSmokeError("BUNDLE_TENSOR_NONFINITE", "manifest-bound tensor contains non-finite bytes", {"role": role, "image_id": image_id})
    return TensorReference(role, image_id, path, expected_shape, "float32", "little", expected_nbytes, observed_hash)


def _tensor_values(reference: TensorReference) -> List[float]:
    payload = reference.path.read_bytes()
    if len(payload) != reference.nbytes or hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise ModelSmokeError("BUNDLE_TENSOR_CHANGED", "private reference changed after validation", {"path": str(reference.path)})
    values = list(struct.unpack("<{}f".format(reference.nbytes // 4), payload))
    if not all(math.isfinite(value) for value in values):
        raise ModelSmokeError("BUNDLE_TENSOR_NONFINITE", "private reference contains non-finite bytes at comparison time", {"path": str(reference.path)})
    return values


def _record_allowlist(reference: TensorReference) -> Dict[str, Any]:
    return {"role": reference.role, "image_id": reference.image_id, "relative_path": "private/" + reference.path.relative_to((reference.path.parents[1]).resolve()).as_posix() if reference.path.parents[1].name == "private" else reference.path.relative_to(reference.path.parents[2]).as_posix(), "nbytes": reference.nbytes, "sha256": reference.sha256}


def load_source_bundle(manifest_path: Path, bundle_root: Path) -> SourceBundle:
    manifest_path = manifest_path.resolve()
    bundle_root = bundle_root.resolve()
    if not manifest_path.is_file():
        raise ModelSmokeError("BUNDLE_MANIFEST_MISSING", "source manifest is missing", {"path": str(manifest_path)})
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelSmokeError("BUNDLE_MANIFEST_INVALID", "source manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise ModelSmokeError("BUNDLE_SCHEMA_MISMATCH", "source manifest schema is not accepted")
    if manifest.get("status") not in {"complete", "execution_complete_numerical_fail"} or manifest.get("execution_status") != "complete":
        raise ModelSmokeError("BUNDLE_EXECUTION_INCOMPLETE", "source bundle is not a completed diagnostic input")
    onnx_meta = manifest.get("onnx")
    if not isinstance(onnx_meta, dict) or onnx_meta.get("sha256") != SOURCE_ONNX_SHA256:
        raise ModelSmokeError("SOURCE_ONNX_HASH_MISMATCH", "source ONNX is not the accepted frozen artifact")
    contract = onnx_meta.get("contract", {})
    expected_contract = {"input_name": "images", "output_name": "output0", "input_shape": list(INPUT_SHAPE), "output_shape": list(OUTPUT_SHAPE), "input_dtype": "float32", "output_dtype": "float32", "opset": 17, "forbidden_nodes": []}
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise ModelSmokeError("SOURCE_ONNX_CONTRACT_MISMATCH", "source manifest ONNX contract is not frozen")
    onnx_path = _private_path(onnx_meta.get("path"), bundle_root)
    if file_sha256(onnx_path) != SOURCE_ONNX_SHA256:
        raise ModelSmokeError("SOURCE_ONNX_HASH_MISMATCH", "private ONNX bytes differ from the accepted SHA-256")
    records = manifest.get("records")
    if not isinstance(records, dict) or tuple(records.keys()) != FIXTURE_IDS:
        raise ModelSmokeError("BUNDLE_FIXTURE_ORDER_MISMATCH", "source records must contain exactly the canonical three fixtures in order")
    inputs = {}
    native = {}
    onnx_outputs = {}
    allowlist = [{"role": "source_onnx", "relative_path": "private/" + onnx_path.relative_to((bundle_root / "private").resolve()).as_posix(), "nbytes": onnx_path.stat().st_size, "sha256": SOURCE_ONNX_SHA256}]
    source_comparisons = {}
    for image_id in FIXTURE_IDS:
        record = records[image_id]
        inputs[image_id] = _tensor_reference(record.get("input"), "input", image_id, bundle_root, INPUT_SHAPE)
        native[image_id] = _tensor_reference(record.get("native"), "source_native", image_id, bundle_root, OUTPUT_SHAPE)
        onnx_outputs[image_id] = _tensor_reference(record.get("onnx"), "source_onnx_output", image_id, bundle_root, OUTPUT_SHAPE)
        allowlist.extend(_record_allowlist(item) for item in (inputs[image_id], native[image_id], onnx_outputs[image_id]))
        source_result = compare_output0(_tensor_values(native[image_id]), _tensor_values(onnx_outputs[image_id]), STRICT_SOURCE_POLICY)
        source_comparisons[image_id] = source_result.as_dict()
        declared = manifest.get("comparisons", {}).get(image_id, {})
        declared_status = "pass" if declared.get("status") == "pass" else "fail"
        if declared_status != source_comparisons[image_id]["status"]:
            raise ModelSmokeError("SOURCE_COMPARISON_RECORD_MISMATCH", "declared source strict status differs from reloaded private bytes", {"image_id": image_id, "declared": declared_status, "observed": source_comparisons[image_id]["status"]})
    if len({item["relative_path"] for item in allowlist}) != 10:
        raise ModelSmokeError("BUNDLE_ALLOWLIST_COLLISION", "source bundle allowlist must contain exactly ten distinct private binaries")
    return SourceBundle(bundle_root, manifest_path, manifest, onnx_path, inputs, native, onnx_outputs, source_comparisons, allowlist)


def _collect_parser_errors(parser: Any) -> List[str]:
    errors = []
    count = int(getattr(parser, "num_errors", 0))
    for index in range(count):
        try:
            errors.append(str(parser.get_error(index)))
        except Exception as exc:
            errors.append("parser_error_unavailable:{}".format(exc))
    return errors


class TensorRTOnnxBuilder:
    def __init__(self, module_loader: Optional[Callable[[str], Any]] = None) -> None:
        self.module_loader = module_loader or importlib.import_module
        self.last_event: Dict[str, Any] = {}

    def build(self, onnx_path: Path, engine_path: Path, workspace_bytes: int = WORKSPACE_BYTES) -> EngineArtifact:
        if engine_path.exists():
            raise ModelSmokeError("ENGINE_OUTPUT_EXISTS", "engine output must be absent before build", {"path": str(engine_path)})
        trt = self.module_loader("tensorrt")
        runtime_version = str(getattr(trt, "__version__", ""))
        if not runtime_version.startswith(EXPECTED_RUNTIME_PREFIX):
            raise ModelSmokeError("RUNTIME_VERSION_MISMATCH", "TensorRT runtime is not the accepted E2 8.5.2.2 family", {"expected_prefix": EXPECTED_RUNTIME_PREFIX, "observed": runtime_version})
        logger = trt.Logger(trt.Logger.ERROR)
        builder = None
        parser = None
        try:
            builder = trt.Builder(logger)
            flags = 0
            if hasattr(trt, "NetworkDefinitionCreationFlag") and hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
                flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(flags)
            parser = trt.OnnxParser(network, logger)
            parsed = parser.parse_from_file(str(onnx_path))
            if parsed is not True:
                errors = _collect_parser_errors(parser)
                self.last_event = {"parser_errors": errors}
                raise ModelSmokeError("ONNX_PARSE_FAILED", "TensorRT ONNX parser rejected the accepted source ONNX", {"parser_errors": errors})
            config = builder.create_builder_config()
            if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
            else:
                config.max_workspace_size = workspace_bytes
            flags_record = {"workspace_bytes": workspace_bytes, "fp16_enabled": False, "tf32_disabled": False}
            if not hasattr(trt, "BuilderFlag") or not hasattr(trt.BuilderFlag, "FP16") or not hasattr(config, "set_flag"):
                raise ModelSmokeError("FP16_BUILDER_FLAG_UNAVAILABLE", "TensorRT builder cannot enforce the required FP16-enabled diagnostic build")
            config.set_flag(trt.BuilderFlag.FP16)
            flags_record["fp16_enabled"] = True
            if hasattr(trt, "BuilderFlag") and hasattr(config, "clear_flag") and hasattr(trt.BuilderFlag, "TF32"):
                config.clear_flag(trt.BuilderFlag.TF32)
                flags_record["tf32_disabled"] = True
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise ModelSmokeError("ENGINE_BUILD_FAILED", "TensorRT returned no serialized engine")
            payload = bytes(serialized)
            if not payload:
                raise ModelSmokeError("ENGINE_BUILD_FAILED", "TensorRT returned an empty serialized engine")
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            with engine_path.open("xb") as handle:
                handle.write(payload)
            digest = hashlib.sha256(payload).hexdigest()
            self.last_event = {"parser_errors": [], "builder_flags": flags_record, "logger_lifetime": "held through parser and builder", "engine_sha256": digest}
            return EngineArtifact(engine_path.resolve(), digest, len(payload), runtime_version, {"builder_flags": flags_record, "parser_errors": [], "source_onnx_sha256": SOURCE_ONNX_SHA256})
        except AdapterError:
            raise
        except ModelSmokeError:
            raise
        except Exception as exc:
            errors = _collect_parser_errors(parser) if parser is not None else []
            self.last_event = {"parser_errors": errors}
            raise ModelSmokeError("ENGINE_BUILD_FAILED", "TensorRT parser/builder raised an exception", {"parser_errors": errors}) from exc


def _observed_trt_version(trt: Any) -> str:
    version = str(getattr(trt, "__version__", "")).strip()
    if not version:
        raise ModelSmokeError("RUNTIME_VERSION_UNAVAILABLE", "TensorRT module does not expose an observed version")
    return version


class E2TensorRTRuntime:
    def __init__(self, module_loader: Optional[Callable[[str], Any]] = None, cuda_factory: Optional[Callable[[], CudaRuntime]] = None) -> None:
        self.module_loader = module_loader or importlib.import_module
        self.cuda_factory = cuda_factory or CudaRuntime
        self.runtime_version = ""

    def preflight(self) -> Dict[str, Any]:
        trt = self.module_loader("tensorrt")
        self.runtime_version = _observed_trt_version(trt)
        if not self.runtime_version.startswith(EXPECTED_RUNTIME_PREFIX):
            raise ModelSmokeError("RUNTIME_VERSION_MISMATCH", "observed TensorRT version is not the E2 profile", {"expected_prefix": EXPECTED_RUNTIME_PREFIX, "observed": self.runtime_version})
        return {"target_id": TARGET_ID, "hardware_profile": EXPECTED_HARDWARE, "runtime_version": self.runtime_version, "execution_api": "legacy_binding_execute_async_v2", "hostname": platform.node(), "machine": platform.machine(), "identity_policy": "E2 profile and installed runtime must be independently operator-verified", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    def build(self, onnx_path: Path, engine_path: Path, workspace_bytes: int) -> EngineArtifact:
        if not self.runtime_version:
            self.preflight()
        return TensorRTOnnxBuilder(self.module_loader).build(onnx_path, engine_path, workspace_bytes)

    def open_execution(self, engine: EngineArtifact) -> TargetExecution:
        if not self.runtime_version:
            self.preflight()
        config = AdapterConfig(TARGET_ID, self.runtime_version, input_dtype="float32")
        provider = TensorRTProvider(config, module_loader=self.module_loader)
        cuda = self.cuda_factory()
        stream = None
        buffers = None
        try:
            descriptor = provider.load_engine(engine.path.read_bytes(), engine.sha256)
            if len(descriptor.output_bindings()) != 1 or descriptor.output_bindings()[0].name != "output0" or descriptor.output_bindings()[0].dtype != "float32":
                raise ModelSmokeError("OUTPUT_BINDING_CONTRACT_MISMATCH", "E2 smoke requires one float32 output0 binding")
            cuda.open()
            stream = cuda.create_stream()
            owner = CudaRuntimeMemoryOwner(cuda, stream)
            buffers = OwnedBuffers(descriptor, owner, stream.handle)
            adapter = provider.create_adapter(stream, buffers)
            return _OwnedTargetExecution(descriptor, adapter, buffers, provider, stream, cuda)
        except BaseException:
            if buffers is not None:
                try:
                    buffers.free()
                except BaseException:
                    pass
            provider.close()
            if stream is not None:
                try:
                    stream.close()
                except BaseException:
                    pass
            try:
                cuda.close()
            except BaseException:
                pass
            raise


class _OwnedTargetExecution:
    def __init__(self, descriptor: EngineDescriptor, adapter: JetsonRuntimeAdapter, buffers: OwnedBuffers, provider: TensorRTProvider, stream: Any, cuda: CudaRuntime) -> None:
        self.descriptor = descriptor
        self.adapter = adapter
        self.buffers = buffers
        self.provider = provider
        self.stream = stream
        self.cuda = cuda

    def infer(self, host_input: HostTensor) -> Dict[str, HostTensor]:
        return self.adapter.infer(host_input)

    def close(self) -> None:
        errors = []
        try:
            self.buffers.free()
        except BaseException as exc:
            errors.append(exc)
        self.provider.close()
        try:
            self.stream.close()
        except BaseException as exc:
            errors.append(exc)
        try:
            self.cuda.close()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise ModelSmokeError("TARGET_CLEANUP_FAILED", "target execution cleanup was not fully successful", {"errors": [_error_dict(item) for item in errors]})


def _raw_values(payload: bytes, expected_nbytes: int) -> List[float]:
    if len(payload) != expected_nbytes:
        raise ModelSmokeError("OUTPUT_BYTES_MISMATCH", "target output byte length differs from frozen contract", {"expected": expected_nbytes, "observed": len(payload)})
    values = list(struct.unpack("<{}f".format(expected_nbytes // 4), payload))
    if not all(math.isfinite(value) for value in values):
        raise ModelSmokeError("TARGET_OUTPUT_NONFINITE", "target output contains non-finite float32 bytes")
    return values


def _neighborhood(image_id: str, native: Sequence[float], source_onnx: Sequence[float], target: Sequence[float]) -> Dict[str, Any]:
    anchor = 8002
    values = {}
    for channel in range(7):
        values[str(channel)] = {}
        for offset in range(-2, 3):
            index = channel * OUTPUT_SHAPE[2] + anchor + offset
            values[str(channel)][str(anchor + offset)] = {"native": native[index], "source_onnx": source_onnx[index], "target": target[index]}
    return {"image_id": image_id, "channel": 1, "anchor": anchor, "flat_index": OUTPUT_SHAPE[2] + anchor, "values": values, "note": "diagnostic neighborhood; no score cutoff or NMS applied"}


class ModelSmokeRunner:
    def __init__(self, bundle: SourceBundle, out_dir: Path, target: TargetRuntime, *, execute: bool = False, build_timeout_seconds: float = 900.0, inference_timeout_seconds: float = 180.0) -> None:
        self.bundle = bundle
        self.out_dir = out_dir.resolve()
        self.target = target
        self.execute = execute
        self.build_timeout_seconds = build_timeout_seconds
        self.inference_timeout_seconds = inference_timeout_seconds
        self.counters = SmokeCounters()
        self.events: List[str] = []
        self.evidence: Dict[str, Any] = {"source_bundle": {"manifest": str(bundle.manifest_path), "onnx_sha256": SOURCE_ONNX_SHA256, "allowlist": bundle.allowlist}, "out_dir": str(self.out_dir)}

    def _log(self, event: str) -> None:
        self.events.append(event)

    def _plan(self) -> Dict[str, Any]:
        return {"schema_version": "e2l1-model-smoke-v1", "status": "planned", "target": {"id": TARGET_ID, "hardware": EXPECTED_HARDWARE, "runtime_prefix": EXPECTED_RUNTIME_PREFIX}, "source_bundle": {"manifest": str(self.bundle.manifest_path), "onnx_sha256": SOURCE_ONNX_SHA256, "allowlist": self.bundle.allowlist}, "builder": {"workspace_bytes": WORKSPACE_BYTES, "fp16_enabled": True, "output_dtype": "float32", "timing_cache": "none", "retries": 0}, "execution": {"device": "E2", "enqueues": 3, "warmup": 0, "repeats": 0, "latency_benchmark": False, "energy_benchmark": False, "input_policy": "exact frozen float32 input bytes; no E2 decode/preprocess"}, "comparisons": {"source_strict": STRICT_SOURCE_POLICY.as_dict(), "target_diagnostic": TARGET_POLICY.as_dict(), "domains": ["boxes", "scores"], "source_failure_preserved": True}, "private_policy": "engine and full tensor outputs remain private; publish only JSON/text evidence"}

    def _write_failure(self, exc: BaseException) -> Dict[str, Any]:
        failure = {"schema_version": "e2l1-model-smoke-v1", "status": "failed", "execution_state": self.evidence.get("execution_state", "not_started"), "real_device_execution": self.execute, "source_strict_status": "fail" if any(item["status"] == "fail" for item in self.bundle.source_comparisons.values()) else "pass", "attempted_completed": self.counters.as_dict(), "events": self.events, "provenance": self.evidence, "error": _error_dict(exc)}
        write_once(self.out_dir / "public" / "failure.json", failure)
        write_once(self.out_dir / "public" / "run.log", "\n".join(self.events) + "\n")
        write_once(self.out_dir / "public" / "index.json", {"schema_version": "e2l1-model-smoke-index-v1", "status": "failed", "public_artifacts": ["plan.json", "failure.json", "run.log", "index.json"]})
        return failure

    def run(self) -> Tuple[int, Dict[str, Any]]:
        if self.out_dir.exists():
            raise ValueError("refusing existing model-smoke output root: {}".format(self.out_dir))
        self.out_dir.mkdir(parents=True, exist_ok=False)
        public = self.out_dir / "public"
        private = self.out_dir / "private"
        public.mkdir()
        private.mkdir()
        try:
            write_once(public / "plan.json", self._plan())
            self._log("plan_written")
            source_status = "fail" if any(item["status"] == "fail" for item in self.bundle.source_comparisons.values()) else "pass"
            self.evidence["source_strict_comparisons"] = self.bundle.source_comparisons
            if not self.execute:
                self.evidence["execution_state"] = "disabled"
                manifest = {"schema_version": "e2l1-model-smoke-v1", "status": "proposed_not_executed", "execution_status": "disabled", "source_strict_status": source_status, "export_discrepancy": {"status": source_status, "source_native_vs_source_onnx": self.bundle.source_comparisons}, "target_diagnostic_status": "not_executed", "tensorrt_discrepancy": {"target_vs_source_onnx": "not_executed", "target_vs_source_native": "not_executed"}, "target_vs_source_onnx_status": "not_executed", "target_vs_source_native_status": "not_executed", "real_device_execution": False, "source_bundle": self.evidence["source_bundle"], "source_comparisons": self.bundle.source_comparisons, "attempted_completed": self.counters.as_dict(), "private_retention": {"root": str(private.resolve()), "published": False}}
                write_once(public / "manifest.json", manifest)
                write_once(public / "report.md", "# E2 model smoke\n\nStatus: proposed_not_executed; device execution remains disabled.\n")
                write_once(public / "run.log", "\n".join(self.events + ["device_execution_disabled"]) + "\n")
                write_once(public / "index.json", {"schema_version": "e2l1-model-smoke-index-v1", "status": "proposed_not_executed", "public_artifacts": ["plan.json", "manifest.json", "report.md", "run.log", "index.json"]})
                return 3, manifest
            self.evidence["execution_state"] = "started"
            target_info = self.target.preflight()
            if target_info.get("target_id") != TARGET_ID or not str(target_info.get("runtime_version", "")).startswith(EXPECTED_RUNTIME_PREFIX):
                raise ModelSmokeError("TARGET_IDENTITY_MISMATCH", "target preflight does not match the E2 TensorRT 8.5.2.2 profile", {"observed": target_info})
            self.evidence["target_preflight"] = target_info
            build_budget = StageBudget("build", self.build_timeout_seconds)
            engine_path = private / "engine" / "e2_yolo11n_fp16.engine"
            self.counters.parse_attempted += 1
            self.counters.build_attempted += 1
            build_budget.check()
            engine = self.target.build(self.bundle.onnx, engine_path, WORKSPACE_BYTES)
            build_budget.check()
            self.counters.parse_completed += 1
            self.counters.build_completed += 1
            self.evidence["engine"] = {"path": str(engine.path), "sha256": engine.sha256, "nbytes": engine.nbytes, "runtime_version": engine.runtime_version, "build_contract": engine.build_contract}
            self.counters.engine_load_attempted += 1
            execution = self.target.open_execution(engine)
            self.counters.engine_load_completed += 1
            target_outputs: Dict[str, Dict[str, Any]] = {}
            try:
                inference_budget = StageBudget("inference", self.inference_timeout_seconds)
                target_dir = private / "target_output"
                target_dir.mkdir()
                self.evidence["target_outputs"] = target_outputs
                for image_id in FIXTURE_IDS:
                    inference_budget.check()
                    self.counters.enqueues_attempted += 1
                    input_reference = self.bundle.inputs[image_id]
                    input_payload = input_reference.path.read_bytes()
                    if hashlib.sha256(input_payload).hexdigest() != input_reference.sha256:
                        raise ModelSmokeError("INPUT_CHANGED", "frozen input changed before target dispatch", {"image_id": image_id})
                    binding = execution.descriptor.input_binding()
                    host_input = make_host_tensor(binding, input_payload)
                    outputs = execution.infer(host_input)
                    self.counters.enqueues_completed += 1
                    self.counters.output_copies_attempted += 1
                    if set(outputs) != {"output0"}:
                        raise ModelSmokeError("OUTPUT_BINDING_CONTRACT_MISMATCH", "target returned an unexpected output binding", {"observed": list(outputs)})
                    output = outputs["output0"]
                    if output.shape != OUTPUT_SHAPE or output.dtype != "float32" or output.nbytes != OUTPUT_NBYTES or not isinstance(output.payload, bytes):
                        raise ModelSmokeError("TARGET_OUTPUT_CONTRACT_MISMATCH", "target output0 is not the frozen float32 [1,7,8400] contract", {"image_id": image_id, "shape": list(output.shape), "dtype": output.dtype, "nbytes": output.nbytes})
                    target_payload = output.payload
                    _raw_values(target_payload, OUTPUT_NBYTES)
                    target_path = target_dir / (image_id + ".bin")
                    with target_path.open("xb") as handle:
                        handle.write(target_payload)
                    target_hash = hashlib.sha256(target_payload).hexdigest()
                    self.counters.output_copies_completed += 1
                    native_values = _tensor_values(self.bundle.native[image_id])
                    source_onnx_values = _tensor_values(self.bundle.onnx_outputs[image_id])
                    target_values = list(struct.unpack("<{}f".format(OUTPUT_NBYTES // 4), target_payload))
                    self.counters.comparisons_attempted += 1
                    target_vs_onnx = compare_output0(source_onnx_values, target_values, TARGET_POLICY).as_dict()
                    target_vs_native = compare_output0(native_values, target_values, TARGET_POLICY).as_dict()
                    self.counters.comparisons_completed += 1
                    target_outputs[image_id] = {"path": str(target_path.resolve()), "nbytes": len(target_payload), "sha256": target_hash, "shape": list(OUTPUT_SHAPE), "dtype": "float32", "target_vs_source_onnx": target_vs_onnx, "target_vs_source_native": target_vs_native, "neighborhood": _neighborhood(image_id, native_values, source_onnx_values, target_values)}
                inference_budget.check()
                self.evidence["execution_state"] = "complete"
            finally:
                try:
                    execution.close()
                except BaseException as close_exc:
                    self.evidence["cleanup_error"] = _error_dict(close_exc)
                    raise
            target_onnx_status = "pass" if all(item["target_vs_source_onnx"]["status"] == "pass" for item in target_outputs.values()) else "fail"
            target_native_status = "pass" if all(item["target_vs_source_native"]["status"] == "pass" for item in target_outputs.values()) else "fail"
            target_status = "pass" if target_onnx_status == "pass" and target_native_status == "pass" else "fail"
            if source_status == "fail" and target_status == "pass":
                status = "execution_complete_source_strict_fail_target_pass"
            elif target_status == "pass":
                status = "execution_complete_all_diagnostic_comparisons_pass"
            else:
                status = "execution_complete_target_mismatch"
            manifest = {"schema_version": "e2l1-model-smoke-v1", "status": status, "execution_status": "complete", "source_strict_status": source_status, "export_discrepancy": {"status": source_status, "source_native_vs_source_onnx": self.bundle.source_comparisons}, "target_diagnostic_status": "complete", "tensorrt_discrepancy": {"target_vs_source_onnx": target_onnx_status, "target_vs_source_native": target_native_status}, "target_vs_source_onnx_status": target_onnx_status, "target_vs_source_native_status": target_native_status, "target_comparison_outcome": target_status, "real_device_execution": True, "source_bundle": self.evidence["source_bundle"], "target_preflight": self.evidence["target_preflight"], "engine": self.evidence["engine"], "source_comparisons": self.bundle.source_comparisons, "target_outputs": target_outputs, "attempted_completed": self.counters.as_dict(), "private_retention": {"root": str(private.resolve()), "published": False}}
            write_once(public / "manifest.json", manifest)
            write_once(public / "report.md", "# E2 model smoke\n\nStatus: {}; source strict: {}; target-vs-ONNX: {}; target-vs-native: {}.\n".format(status, source_status, target_onnx_status, target_native_status))
            write_once(public / "run.log", "\n".join(self.events + ["model_smoke_complete"]) + "\n")
            write_once(public / "index.json", {"schema_version": "e2l1-model-smoke-index-v1", "status": status, "public_artifacts": ["plan.json", "manifest.json", "report.md", "run.log", "index.json"], "private_root": str(private.resolve())})
            return 0 if target_status == "pass" else 3, manifest
        except BaseException as exc:
            self._log("failed:{}".format(type(exc).__name__))
            if self.evidence.get("execution_state") == "started":
                self.evidence["execution_state"] = "unknown" if isinstance(exc, (TargetTimeout, TargetExecutionUnknown, TimeoutError)) else "failed"
            failure = self._write_failure(exc)
            return 2, failure


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="E2 TensorRT model diagnostic smoke; device execution is disabled by default")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", choices=("E2",), default="E2")
    parser.add_argument("--execute-real-device", action="store_true")
    parser.add_argument("--build-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--inference-timeout-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.target != TARGET_ID:
        raise SystemExit("only E2 is in E2L1-014")
    if args.execute_real_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        bundle = load_source_bundle(args.manifest, args.bundle_root)
        runner = ModelSmokeRunner(bundle, args.out_dir, E2TensorRTRuntime(), execute=args.execute_real_device, build_timeout_seconds=args.build_timeout_seconds, inference_timeout_seconds=args.inference_timeout_seconds)
        code, result = runner.run()
    except BaseException as exc:
        result = {"schema_version": "e2l1-model-smoke-v1", "status": "failed_before_output", "error": _error_dict(exc)}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
