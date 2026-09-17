#!/usr/bin/env python3
"""Local-only Jetson TensorRT adapter contract and protocol planning.

This module deliberately imports no TensorRT, CUDA, NumPy, image decoder or
model runtime. Device runtime objects are injected behind small protocols so
the contract can be tested with CPU mocks before any target-side authorization.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

INPUT_SHAPE = (1, 3, 640, 640)
YOLO11N_MODEL_ID = "yolo11n_cctsdb_clean_s42_v2"
YOLO11N_CHECKPOINT_PATH = "results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt"
YOLO11N_CHECKPOINT_SHA256 = "3e5fc7a2148c16539cd9fb7cc7cacd81a4eec1dfc28143cdf9b6dcd872ba4ab8"
DTYPE_ITEMSIZE = {"float16": 2, "float32": 4}

class AdapterError(ValueError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.details = code, message, details or {}

    def as_dict(self) -> dict[str, Any]:
        return {"status": "error", "error": {"code": self.code, "message": self.message, "details": self.details}}

@dataclass(frozen=True)
class JetsonProfile:
    target_id: str
    hardware: str
    expected_runtime_prefix: str
    execution_api: str
    compatibility_status: str

JETSON_PROFILES = {
    "E2": JetsonProfile("E2", "Jetson Xavier NX", "8.5.2.2", "legacy_binding_execute_async_v2", "target smoke validation required"),
    "E3": JetsonProfile("E3", "Jetson AGX Xavier", "8.5.2.2", "legacy_binding_execute_async_v2", "target smoke validation required"),
    "E5": JetsonProfile("E5", "Jetson Orin Nano Super", "10.3.0", "named_tensor_execute_async_v3", "target smoke validation required"),
}

@dataclass(frozen=True)
class AdapterConfig:
    target_id: str
    runtime_version: str
    model_id: str = YOLO11N_MODEL_ID
    checkpoint_sha256: str = YOLO11N_CHECKPOINT_SHA256
    input_shape: tuple[int, int, int, int] = INPUT_SHAPE
    input_dtype: str = "float32"

    def validate(self) -> JetsonProfile:
        profile = JETSON_PROFILES.get(self.target_id)
        if profile is None:
            raise AdapterError("UNKNOWN_TARGET", "only E2, E3 and E5 are in this adapter package")
        if not self.runtime_version.startswith(profile.expected_runtime_prefix):
            raise AdapterError("RUNTIME_VERSION_MISMATCH", "runtime version does not match observed target profile", {"expected_prefix": profile.expected_runtime_prefix, "observed": self.runtime_version})
        if self.model_id != YOLO11N_MODEL_ID:
            raise AdapterError("MODEL_NOT_APPROVED", "only the frozen YOLO11n engineering reference is in this smoke plan")
        if self.checkpoint_sha256.lower() != YOLO11N_CHECKPOINT_SHA256:
            raise AdapterError("CHECKPOINT_HASH_MISMATCH", "checkpoint hash is not the accepted frozen YOLO11n reference")
        if tuple(self.input_shape) != INPUT_SHAPE:
            raise AdapterError("INPUT_SHAPE_MISMATCH", "adapter requires batch-1 640x640 input")
        if self.input_dtype not in DTYPE_ITEMSIZE:
            raise AdapterError("UNSUPPORTED_DTYPE", "input dtype must be float16 or float32")
        return profile

@dataclass(frozen=True)
class EngineBinding:
    name: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    location: str = "device"

@dataclass(frozen=True)
class EngineDescriptor:
    runtime_version: str
    engine_sha256: str
    bindings: tuple[EngineBinding, ...]

    def validate(self, config: AdapterConfig) -> None:
        profile = config.validate()
        if not self.engine_sha256 or len(self.engine_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.engine_sha256.lower()):
            raise AdapterError("INVALID_ENGINE_HASH", "engine_sha256 must be a hexadecimal SHA-256")
        if not self.runtime_version.startswith(profile.expected_runtime_prefix):
            raise AdapterError("RUNTIME_VERSION_MISMATCH", "engine runtime version does not match target profile")
        if not self.bindings:
            raise AdapterError("MISSING_BINDINGS", "engine must expose at least one I/O binding")
        names = [binding.name for binding in self.bindings]
        if len(set(names)) != len(names) or any(not name for name in names):
            raise AdapterError("INVALID_BINDINGS", "binding names must be unique and non-empty")
        inputs = [binding for binding in self.bindings if binding.role == "input"]
        outputs = [binding for binding in self.bindings if binding.role == "output"]
        if len(inputs) != 1 or not outputs:
            raise AdapterError("INVALID_BINDINGS", "smoke contract requires exactly one input and at least one output")
        input_binding = inputs[0]
        if tuple(input_binding.shape) != config.input_shape:
            raise AdapterError("INPUT_SHAPE_MISMATCH", "engine input shape is not (1,3,640,640)", {"observed": input_binding.shape})
        if input_binding.dtype != config.input_dtype:
            raise AdapterError("INPUT_DTYPE_MISMATCH", "engine input dtype differs from the declared adapter dtype")
        for binding in self.bindings:
            if binding.dtype not in DTYPE_ITEMSIZE or binding.location not in {"host", "device"} or not binding.shape or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in binding.shape):
                raise AdapterError("INVALID_BINDING", "binding dtype, location and static positive shape must be explicit", {"binding": binding.name})

    def input_binding(self) -> EngineBinding:
        return next(binding for binding in self.bindings if binding.role == "input")

@dataclass(frozen=True)
class HostTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    payload: Any

@dataclass(frozen=True)
class DeviceTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    device_pointer: int

class StreamLike(Protocol):
    handle: int
    def synchronize(self, stage: str) -> None: ...

class ContextLike(Protocol):
    def execute_async_v2(self, bindings: list[int], stream_handle: int) -> bool: ...
    def set_tensor_address(self, name: str, device_pointer: int) -> bool: ...
    def execute_async_v3(self, stream_handle: int) -> bool: ...

class BufferManagerLike(Protocol):
    def copy_host_to_device(self, host: HostTensor, device: DeviceTensor) -> None: ...
    def copy_device_to_host(self, device: DeviceTensor) -> None: ...
    def binding_pointers(self) -> list[int]: ...
    def output_tensors(self) -> dict[str, DeviceTensor]: ...

def expected_nbytes(shape: tuple[int, ...], dtype: str) -> int:
    if dtype not in DTYPE_ITEMSIZE or not shape or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise AdapterError("INVALID_TENSOR", "tensor shape and dtype must be explicit")
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * DTYPE_ITEMSIZE[dtype]

def make_host_tensor(binding: EngineBinding, payload: Any, *, nbytes: int | None = None) -> HostTensor:
    size = expected_nbytes(binding.shape, binding.dtype) if nbytes is None else nbytes
    if size != expected_nbytes(binding.shape, binding.dtype):
        raise AdapterError("BUFFER_SIZE_MISMATCH", "host tensor byte size does not match binding")
    return HostTensor(binding.name, binding.shape, binding.dtype, size, payload)

class JetsonRuntimeAdapter:
    """A narrow injected-runtime adapter; it never imports a device runtime."""

    def __init__(self, config: AdapterConfig, engine: EngineDescriptor, context: ContextLike, stream: StreamLike, buffers: BufferManagerLike):
        engine.validate(config)
        self.config, self.engine, self.context, self.stream, self.buffers = config, engine, context, stream, buffers
        self.profile = config.validate()
        self._device_bindings = {
            binding.name: DeviceTensor(binding.name, binding.shape, binding.dtype, expected_nbytes(binding.shape, binding.dtype), index + 1)
            for index, binding in enumerate(engine.bindings)
        }

    def synchronize(self, stage: str) -> None:
        self.stream.synchronize(stage)

    def infer(self, host_input: HostTensor) -> dict[str, DeviceTensor]:
        input_binding = self.engine.input_binding()
        if host_input.name != input_binding.name or host_input.shape != input_binding.shape or host_input.dtype != input_binding.dtype or host_input.nbytes != expected_nbytes(input_binding.shape, input_binding.dtype):
            raise AdapterError("INPUT_CONTRACT_MISMATCH", "host input does not match the engine input binding")
        input_device = self._device_bindings[input_binding.name]
        self.synchronize("before_input_copy")
        self.buffers.copy_host_to_device(host_input, input_device)
        self.synchronize("before_enqueue")
        pointers = self.buffers.binding_pointers()
        try:
            if self.profile.execution_api == "legacy_binding_execute_async_v2":
                enqueue_result = self.context.execute_async_v2(pointers, self.stream.handle)
            else:
                for binding in self.engine.bindings:
                    if not self.context.set_tensor_address(binding.name, self._device_bindings[binding.name].device_pointer):
                        raise AdapterError("SET_TENSOR_ADDRESS_FAILED", "runtime rejected a tensor address", {"tensor": binding.name})
                enqueue_result = self.context.execute_async_v3(self.stream.handle)
        except AttributeError as exc:
            raise AdapterError("RUNTIME_API_UNAVAILABLE", "injected context lacks the API required by the target profile", {"api": self.profile.execution_api}) from exc
        if enqueue_result is not True:
            raise AdapterError("ENQUEUE_FAILED", "runtime enqueue did not return true")
        self.synchronize("after_enqueue")
        for binding in self.engine.bindings:
            if binding.role == "output":
                self.buffers.copy_device_to_host(self._device_bindings[binding.name])
        self.synchronize("after_output_copy")
        return self.buffers.output_tensors()

@dataclass
class MockStream:
    handle: int = 7
    events: list[str] | None = None
    def __post_init__(self) -> None:
        if self.events is None:
            self.events = []
    def synchronize(self, stage: str) -> None:
        self.events.append(f"sync:{stage}")

@dataclass
class MockContext:
    events: list[str]
    enqueue_result: bool = True
    supports_v2: bool = True
    supports_v3: bool = True
    def execute_async_v2(self, bindings: list[int], stream_handle: int) -> bool:
        if not self.supports_v2:
            raise AttributeError("execute_async_v2 unavailable")
        self.events.append(f"execute_async_v2:{len(bindings)}:{stream_handle}")
        return self.enqueue_result
    def set_tensor_address(self, name: str, device_pointer: int) -> bool:
        if not self.supports_v3:
            raise AttributeError("set_tensor_address unavailable")
        self.events.append(f"set_tensor_address:{name}:{device_pointer}")
        return True
    def execute_async_v3(self, stream_handle: int) -> bool:
        if not self.supports_v3:
            raise AttributeError("execute_async_v3 unavailable")
        self.events.append(f"execute_async_v3:{stream_handle}")
        return self.enqueue_result

@dataclass
class MockBuffers:
    engine: EngineDescriptor
    events: list[str]
    def __post_init__(self) -> None:
        self.devices = {
            binding.name: DeviceTensor(binding.name, binding.shape, binding.dtype, expected_nbytes(binding.shape, binding.dtype), index + 1)
            for index, binding in enumerate(self.engine.bindings)
        }
    def copy_host_to_device(self, host: HostTensor, device: DeviceTensor) -> None:
        self.events.append(f"copy_h2d:{host.name}:{host.nbytes}")
    def copy_device_to_host(self, device: DeviceTensor) -> None:
        self.events.append(f"copy_d2h:{device.name}:{device.nbytes}")
    def binding_pointers(self) -> list[int]:
        return [self.devices[binding.name].device_pointer for binding in self.engine.bindings]
    def output_tensors(self) -> dict[str, DeviceTensor]:
        return {name: tensor for name, tensor in self.devices.items() if next(binding for binding in self.engine.bindings if binding.name == name).role == "output"}

@dataclass(frozen=True)
class FixtureImage:
    image_id: str
    content: bytes
    sha256: str

def make_train_only_fixture(count: int = 3) -> tuple[FixtureImage, ...]:
    if count <= 0:
        raise AdapterError("INVALID_FIXTURE", "fixture count must be positive")
    images = []
    for index in range(count):
        content = b"P6\n4 4\n255\n" + bytes(((index * 53 + pixel * 17 + channel * 29) % 256) for pixel in range(16) for channel in range(3))
        images.append(FixtureImage(f"train-fixture-{index:02d}", content, hashlib.sha256(content).hexdigest()))
    return tuple(images)

def fixture_manifest(images: tuple[FixtureImage, ...]) -> dict[str, Any]:
    sequence = [{"image_id": image.image_id, "sha256": image.sha256} for image in images]
    return {
        "split": "train_only_engineering_fixture",
        "image_order": [image.image_id for image in images],
        "images": sequence,
        "sequence_sha256": hashlib.sha256(json.dumps(sequence, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "official_test_used": False,
        "content_provenance": "deterministic local PPM bytes; same bytes/order are supplied to source-reference and adapter contract tests",
    }
def build_smoke_plan(target_id: str = "E2") -> dict[str, Any]:
    if target_id not in JETSON_PROFILES:
        raise AdapterError("UNKNOWN_TARGET", "smoke plan target must be E2, E3 or E5")
    profile = JETSON_PROFILES[target_id]
    fixture = make_train_only_fixture()
    return {
        "status": "proposed_not_executed",
        "target": {"id": profile.target_id, "hardware": profile.hardware, "observed_runtime_prefix": profile.expected_runtime_prefix, "execution_api_candidate": profile.execution_api, "compatibility_status": profile.compatibility_status},
        "model": {"id": YOLO11N_MODEL_ID, "family": "YOLO11n", "precision": "FP16", "checkpoint_path": YOLO11N_CHECKPOINT_PATH, "checkpoint_sha256": YOLO11N_CHECKPOINT_SHA256, "engineering_reference_only": True},
        "input": {"shape": list(INPUT_SHAPE), "batch": 1, "dtype": "float32", "preprocessing": "OpenCV-style BGR decoded image -> RGB -> letterbox preserving aspect ratio -> contiguous NCHW float scaling [0,1]", "disk_decode_in_timing": False},
        "postprocessing": {"contract": "native engine output must be identified before adapter binding; one NMS owner only; no double NMS", "confidence_threshold": 0.25, "iou_threshold": 0.7, "max_detections": 300, "threshold_status": "provisional deployment config; not tuned in this smoke plan"},
        "fixture": fixture_manifest(fixture),
        "proposed_target_path": "target-local TensorRT FP16 build from the frozen checkpoint; never transfer the server RTX engine",
        "checks_before_execution": ["target identity/UUID and runtime versions", "direct checkpoint hash", "target-built engine hash and engine/runtime compatibility", "input/output binding names, shape, dtype and location", "explicit stream/event completion for copies, enqueue and output copy", "raw telemetry source and clock/boundary evidence"],
        "allowed_resources": ["existing JetPack-provided TensorRT runtime on E2", "existing approved checkpoint bytes", "train-only deterministic fixture", "new target-scoped output directory"],
        "forbidden_actions": ["package installation", "sudo or mode/clock/fan changes", "server-engine transfer", "engine export/build", "model load/forward", "official-test images", "AP or FPS/energy scoring"],
        "stop_conditions": ["runtime/profile mismatch", "checkpoint or engine hash mismatch", "binding shape/dtype mismatch", "missing final synchronization", "unknown/native output contract", "missing power boundary or clock alignment"],
    }
