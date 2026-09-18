#!/usr/bin/env python3
"""Local-only Jetson TensorRT adapter contract and CPU/mock orchestration.

No TensorRT, CUDA, NumPy, image decoder, or model runtime is imported here.
Runtime, stream, allocation, and copy ownership are injected so the contract
can be verified without a target device.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from edge_readiness.edge_errors import AdapterError

INPUT_SHAPE = (1, 3, 640, 640)
YOLO11N_MODEL_ID = "yolo11n_cctsdb_clean_s42_v2"
YOLO11N_CHECKPOINT_PATH = "results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt"
YOLO11N_CHECKPOINT_SHA256 = "3e5fc7a2148c16539cd9fb7cc7cacd81a4eec1dfc28143cdf9b6dcd872ba4ab8"
YOLO11N_NATIVE_OUTPUT_SHAPE = (1, 7, 8400)
YOLO11N_NATIVE_OUTPUT_NAME = "output0"
DTYPE_ITEMSIZE = {"float16": 2, "float32": 4}


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
        if any(binding.role not in {"input", "output"} for binding in self.bindings):
            raise AdapterError("UNSUPPORTED_BINDING_ROLE", "only input and output bindings are supported")
        if any(binding.location != "device" for binding in self.bindings):
            raise AdapterError("UNSUPPORTED_BINDING_LOCATION", "host-location bindings are not accepted by this device-copy contract")
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
            if binding.dtype not in DTYPE_ITEMSIZE or not binding.shape or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in binding.shape):
                raise AdapterError("INVALID_BINDING", "binding dtype and static positive shape must be explicit", {"binding": binding.name})

    def input_binding(self) -> EngineBinding:
        return next(binding for binding in self.bindings if binding.role == "input")

    def output_bindings(self) -> tuple[EngineBinding, ...]:
        return tuple(binding for binding in self.bindings if binding.role == "output")


@dataclass(frozen=True)
class HostTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    payload: bytes


@dataclass(frozen=True)
class DeviceTensor:
    """Allocation descriptor owned by the injected buffer manager."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    device_pointer: int
    owner: str
    lifetime_id: str


class StreamLike(Protocol):
    handle: int

    def synchronize(self, stage: str) -> None: ...


class ContextLike(Protocol):
    def execute_async_v2(self, bindings: list[int], stream_handle: int) -> bool: ...
    def set_tensor_address(self, name: str, device_pointer: int) -> bool: ...
    def execute_async_v3(self, stream_handle: int) -> bool: ...


class BufferManagerLike(Protocol):
    """Owner contract: reset state before each call and publish after sync."""

    def begin_inference(self) -> None: ...
    def copy_stream_handle(self) -> int: ...
    def binding_buffers(self) -> dict[str, DeviceTensor]: ...
    def copy_host_to_device(self, host: HostTensor, device: DeviceTensor, stream_handle: int) -> None: ...
    def copy_device_to_host(self, device: DeviceTensor, stream_handle: int) -> None: ...
    def mark_outputs_ready(self, stream_handle: int) -> None: ...
    def output_tensors(self) -> dict[str, HostTensor]: ...


def expected_nbytes(shape: tuple[int, ...], dtype: str) -> int:
    if dtype not in DTYPE_ITEMSIZE or not shape or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise AdapterError("INVALID_TENSOR", "tensor shape and dtype must be explicit")
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * DTYPE_ITEMSIZE[dtype]


def make_host_tensor(binding: EngineBinding, payload: bytes, *, nbytes: int | None = None) -> HostTensor:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise AdapterError("INVALID_HOST_PAYLOAD", "host payload must be bytes-like")
    expected = expected_nbytes(binding.shape, binding.dtype)
    size = expected if nbytes is None else nbytes
    if size != expected or len(payload) != expected:
        raise AdapterError("BUFFER_SIZE_MISMATCH", "host tensor byte size does not match binding", {"expected": expected, "declared": size, "actual": len(payload)})
    return HostTensor(binding.name, binding.shape, binding.dtype, size, bytes(payload))


def validate_yolo11n_native_output_contract(engine: EngineDescriptor, config: AdapterConfig) -> dict[str, Any]:
    """Reject a generic positive shape that cannot represent this checkpoint."""
    engine.validate(config)
    outputs = engine.output_bindings()
    if len(outputs) != 1 or outputs[0].name != YOLO11N_NATIVE_OUTPUT_NAME or tuple(outputs[0].shape) != YOLO11N_NATIVE_OUTPUT_SHAPE:
        raise AdapterError("MODEL_OUTPUT_CONTRACT_MISMATCH", "YOLO11n no-NMS engineering output must be output0 [1,7,8400]", {"expected": {"name": YOLO11N_NATIVE_OUTPUT_NAME, "shape": list(YOLO11N_NATIVE_OUTPUT_SHAPE)}, "observed": [{"name": item.name, "shape": list(item.shape)} for item in outputs]})
    return {
        "model_id": config.model_id,
        "native_output_name": YOLO11N_NATIVE_OUTPUT_NAME,
        "native_output_shape": list(YOLO11N_NATIVE_OUTPUT_SHAPE),
        "source_tensor": "source raw prediction tensor -> output0; compare before decode/NMS",
        "nms_owner": "none in this raw contract; exactly one later postprocess owner",
        "coordinate_mapping": "compare output0 in the same [1,7,8400] coordinate order before decoding",
    }


class JetsonRuntimeAdapter:
    """Narrow injected-runtime adapter; never allocates device addresses."""

    def __init__(self, config: AdapterConfig, engine: EngineDescriptor, context: ContextLike, stream: StreamLike, buffers: BufferManagerLike, stage_observer: Optional[Callable[[str], None]] = None):
        engine.validate(config)
        self.config, self.engine, self.context, self.stream, self.buffers = config, engine, context, stream, buffers
        self.stage_observer = stage_observer
        self.profile = config.validate()
        if buffers.copy_stream_handle() != stream.handle:
            raise AdapterError("COPY_STREAM_MISMATCH", "buffer copies and runtime synchronization must use the same stream", {"copy_stream": buffers.copy_stream_handle(), "runtime_stream": stream.handle})
        allocations = buffers.binding_buffers()
        names = [binding.name for binding in engine.bindings]
        if set(allocations) != set(names) or len(allocations) != len(names):
            raise AdapterError("BUFFER_BINDING_SET_MISMATCH", "buffer owner must supply exactly one allocation for every engine binding", {"expected": names, "observed": list(allocations)})
        pointers = [allocations[name].device_pointer for name in names]
        if any(isinstance(pointer, bool) or not isinstance(pointer, int) or pointer <= 0 for pointer in pointers):
            raise AdapterError("INVALID_DEVICE_POINTER", "device pointers must be positive non-boolean integers")
        if len(set(pointers)) != len(pointers):
            raise AdapterError("DUPLICATE_DEVICE_POINTER", "each binding must own a distinct device pointer")
        for binding in engine.bindings:
            allocation = allocations[binding.name]
            if allocation.name != binding.name or tuple(allocation.shape) != tuple(binding.shape) or allocation.dtype != binding.dtype or allocation.nbytes != expected_nbytes(binding.shape, binding.dtype):
                raise AdapterError("BUFFER_DESCRIPTOR_MISMATCH", "allocation descriptor does not match engine binding", {"binding": binding.name})
            if not allocation.owner or not allocation.lifetime_id:
                raise AdapterError("BUFFER_LIFETIME_MISSING", "allocation owner and lifetime_id are required", {"binding": binding.name})
        self._device_bindings = dict(allocations)

    def _observe(self, event: str) -> None:
        if self.stage_observer is not None:
            self.stage_observer(event)

    def synchronize(self, stage: str) -> None:
        self.stream.synchronize(stage)

    def infer(self, host_input: HostTensor) -> dict[str, HostTensor]:
        input_binding = self.engine.input_binding()
        expected = expected_nbytes(input_binding.shape, input_binding.dtype)
        if host_input.name != input_binding.name or host_input.shape != input_binding.shape or host_input.dtype != input_binding.dtype or host_input.nbytes != expected or len(host_input.payload) != expected:
            raise AdapterError("INPUT_CONTRACT_MISMATCH", "host input does not match the engine input binding")
        self.buffers.begin_inference()
        input_device = self._device_bindings[input_binding.name]
        self.synchronize("before_input_copy")
        self._observe("h2d_copy_attempted")
        try:
            self.buffers.copy_host_to_device(host_input, input_device, self.stream.handle)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("COPY_H2D_FAILED", "buffer owner rejected host-to-device copy") from exc
        self._observe("h2d_copy_completed")
        self.synchronize("before_enqueue")
        pointers = [self._device_bindings[binding.name].device_pointer for binding in self.engine.bindings]
        self._observe("enqueue_attempted")
        try:
            if self.profile.execution_api == "legacy_binding_execute_async_v2":
                enqueue_result = self.context.execute_async_v2(pointers, self.stream.handle)
            else:
                for binding in self.engine.bindings:
                    if not self.context.set_tensor_address(binding.name, self._device_bindings[binding.name].device_pointer):
                        raise AdapterError("SET_TENSOR_ADDRESS_FAILED", "runtime rejected a tensor address", {"tensor": binding.name})
                enqueue_result = self.context.execute_async_v3(self.stream.handle)
        except AdapterError:
            raise
        except AttributeError as exc:
            raise AdapterError("RUNTIME_API_UNAVAILABLE", "injected context lacks the API required by the target profile", {"api": self.profile.execution_api}) from exc
        except Exception as exc:
            raise AdapterError("ENQUEUE_FAILED", "runtime enqueue raised an exception") from exc
        if enqueue_result is not True:
            raise AdapterError("ENQUEUE_FAILED", "runtime enqueue did not return true")
        self._observe("enqueue_completed")
        self.synchronize("after_enqueue")
        self._observe("enqueue_synchronized")
        try:
            for binding in self.engine.output_bindings():
                self._observe("d2h_copy_attempted")
                self.buffers.copy_device_to_host(self._device_bindings[binding.name], self.stream.handle)
                self._observe("d2h_copy_completed")
            self.synchronize("after_output_copy")
            self._observe("d2h_copy_synchronized")
            self.buffers.mark_outputs_ready(self.stream.handle)
            outputs = self.buffers.output_tensors()
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("COPY_D2H_FAILED", "buffer owner failed to complete device-to-host output copy") from exc
        expected_names = {binding.name for binding in self.engine.output_bindings()}
        if set(outputs) != expected_names:
            raise AdapterError("OUTPUT_CONTRACT_MISMATCH", "buffer owner did not return complete host output payloads")
        for binding in self.engine.output_bindings():
            tensor = outputs[binding.name]
            expected_nbytes_value = expected_nbytes(binding.shape, binding.dtype)
            if (
                tensor.name != binding.name
                or tuple(tensor.shape) != tuple(binding.shape)
                or tensor.dtype != binding.dtype
                or tensor.nbytes != expected_nbytes_value
                or not isinstance(tensor.payload, bytes)
                or len(tensor.payload) != expected_nbytes_value
            ):
                raise AdapterError("OUTPUT_CONTRACT_MISMATCH", "host output descriptor or payload does not match engine binding", {"binding": binding.name})
        return outputs


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
    fail_named_address: str | None = None

    def execute_async_v2(self, bindings: list[int], stream_handle: int) -> bool:
        if not self.supports_v2:
            raise AttributeError("execute_async_v2 unavailable")
        self.events.append(f"execute_async_v2:{','.join(str(pointer) for pointer in bindings)}:{stream_handle}")
        return self.enqueue_result

    def set_tensor_address(self, name: str, device_pointer: int) -> bool:
        if not self.supports_v3:
            raise AttributeError("set_tensor_address unavailable")
        self.events.append(f"set_tensor_address:{name}:{device_pointer}")
        return name != self.fail_named_address

    def execute_async_v3(self, stream_handle: int) -> bool:
        if not self.supports_v3:
            raise AttributeError("execute_async_v3 unavailable")
        self.events.append(f"execute_async_v3:{stream_handle}")
        return self.enqueue_result


@dataclass
class MockBuffers:
    engine: EngineDescriptor
    events: list[str]
    allocations: dict[str, DeviceTensor] | None = None
    stream_handle: int = 7
    fail_h2d: bool = False
    fail_d2h: bool = False

    def __post_init__(self) -> None:
        if self.allocations is None:
            self.allocations = {
                binding.name: DeviceTensor(binding.name, binding.shape, binding.dtype, expected_nbytes(binding.shape, binding.dtype), 4096 + index * 4096, "mock-buffer-owner", "mock-session-lifetime")
                for index, binding in enumerate(self.engine.bindings)
            }
        self._pending_outputs: set[str] = set()
        self._ready = False
        self._host_outputs: dict[str, HostTensor] = {}

    def copy_stream_handle(self) -> int:
        return self.stream_handle

    def begin_inference(self) -> None:
        self._pending_outputs.clear()
        self._ready = False
        self._host_outputs.clear()

    def binding_buffers(self) -> dict[str, DeviceTensor]:
        return dict(self.allocations or {})

    def copy_host_to_device(self, host: HostTensor, device: DeviceTensor, stream_handle: int) -> None:
        if self.fail_h2d:
            raise AdapterError("MOCK_H2D_FAILED", "mock requested host-to-device failure")
        if stream_handle != self.stream_handle:
            raise AdapterError("COPY_STREAM_MISMATCH", "mock copy used the wrong stream")
        self.events.append(f"copy_h2d:{host.name}:{device.device_pointer}:{host.nbytes}")

    def copy_device_to_host(self, device: DeviceTensor, stream_handle: int) -> None:
        if self.fail_d2h:
            raise AdapterError("MOCK_D2H_FAILED", "mock requested device-to-host failure")
        if stream_handle != self.stream_handle:
            raise AdapterError("COPY_STREAM_MISMATCH", "mock copy used the wrong stream")
        self.events.append(f"copy_d2h:{device.name}:{device.device_pointer}:{device.nbytes}")
        self._pending_outputs.add(device.name)

    def mark_outputs_ready(self, stream_handle: int) -> None:
        if stream_handle != self.stream_handle:
            raise AdapterError("COPY_STREAM_MISMATCH", "mock completion used the wrong stream")
        expected = {binding.name for binding in self.engine.output_bindings()}
        if self._pending_outputs != expected:
            raise AdapterError("OUTPUT_NOT_READY", "not every output copy completed")
        self._host_outputs = {
            name: HostTensor(name, tensor.shape, tensor.dtype, tensor.nbytes, bytes(tensor.nbytes))
            for name, tensor in self.binding_buffers().items()
            if name in expected
        }
        self._ready = True

    def output_tensors(self) -> dict[str, HostTensor]:
        if not self._ready:
            raise AdapterError("OUTPUT_NOT_READY", "host outputs are unavailable before final stream completion")
        return dict(self._host_outputs)


@dataclass(frozen=True)
class AdapterInferenceResult:
    outputs: dict[str, HostTensor]
    output_contract: dict[str, Any]


class AdapterHarnessBridge:
    """Duck-typed BackendAdapter bridge for the accepted measurement harness."""

    backend_name = "jetson_adapter_cpu_mock"
    synchronization_policy = "adapter-owned-copy-stream; final output readiness is explicit"

    def __init__(self, adapter: JetsonRuntimeAdapter):
        self.adapter = adapter
        self.events: list[str] = []

    def synchronize(self, phase: str) -> None:
        self.events.append(f"harness_sync:{phase}")
        self.adapter.synchronize(f"harness_{phase}")

    def infer(self, preprocessed: Any) -> AdapterInferenceResult:
        if tuple(preprocessed.shape) != self.adapter.config.input_shape:
            raise AdapterError("BRIDGE_SHAPE_MISMATCH", "harness prepared shape differs from adapter input")
        host = make_host_tensor(self.adapter.engine.input_binding(), preprocessed.payload)
        outputs = self.adapter.infer(host)
        contract = {
            "outputs": [{"name": name, "shape": list(tensor.shape), "dtype": tensor.dtype, "nbytes": tensor.nbytes, "payload_sha256": hashlib.sha256(tensor.payload).hexdigest()} for name, tensor in outputs.items()],
            "available_after": "adapter.after_output_copy_completion",
        }
        return AdapterInferenceResult(outputs, contract)


@dataclass(frozen=True)
class FixtureImage:
    image_id: str
    content: bytes
    sha256: str


def make_train_only_fixture(count: int = 3) -> tuple[FixtureImage, ...]:
    """Synthetic bytes for contract tests; not CCTSDB train evidence."""
    if count <= 0:
        raise AdapterError("INVALID_FIXTURE", "fixture count must be positive")
    images = []
    for index in range(count):
        content = b"P6\n4 4\n255\n" + bytes(((index * 53 + pixel * 17 + channel * 29) % 256) for pixel in range(16) for channel in range(3))
        images.append(FixtureImage(f"mock-synthetic-{index:02d}", content, hashlib.sha256(content).hexdigest()))
    return tuple(images)


def fixture_to_input_bytes(image: FixtureImage, *, nbytes: int = expected_nbytes(INPUT_SHAPE, "float32")) -> bytes:
    """Deterministically derive mock tensor bytes from fixture bytes, no decode."""
    seed = hashlib.sha256(image.content).digest()
    return (seed * ((nbytes + len(seed) - 1) // len(seed)))[:nbytes]


def fixture_manifest(images: tuple[FixtureImage, ...]) -> dict[str, Any]:
    sequence = [{"image_id": image.image_id, "sha256": image.sha256, "derived_input_sha256": hashlib.sha256(fixture_to_input_bytes(image)).hexdigest()} for image in images]
    return {
        "split": "synthetic_mock_fixture",
        "image_order": [image.image_id for image in images],
        "images": sequence,
        "sequence_sha256": hashlib.sha256(json.dumps(sequence, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "official_test_used": False,
        "content_provenance": "deterministic local synthetic PPM bytes for CPU/mock contract tests only; not CCTSDB evidence",
        "accepted_cctsdb_materialization": {"status": "missing_prerequisite", "required": "first three U42 CCTSDB train IDs in canonical order with accepted source content hashes", "synthetic_fixture_is_not_substitute": True},
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
        "source_reference": {"checkpoint": YOLO11N_CHECKPOINT_PATH, "deployment_policy": "configs/deployment/yolo11n_fp16_trt10.json", "accepted_server_fp16_evidence": "results/calibration_method_v1/rtx8000/yolo11n/eval/yolo11n_fp16_reference/", "callable_source_forward": "missing_prerequisite; current package does not import a model runtime"},
        "input": {"shape": list(INPUT_SHAPE), "batch": 1, "dtype": "float32", "preprocessing": "OpenCV-style BGR decoded image -> RGB -> letterbox preserving aspect ratio -> contiguous NCHW float scaling [0,1]", "disk_decode_in_timing": False},
        "native_output_contract": {"name": YOLO11N_NATIVE_OUTPUT_NAME, "shape": list(YOLO11N_NATIVE_OUTPUT_SHAPE), "source_to_engine_mapping": "source raw prediction tensor -> output0; compare before decode/NMS"},
        "postprocessing": {"contract": "native engine output must be identified before adapter binding; one NMS owner only; no double NMS", "confidence_threshold": 0.25, "iou_threshold": 0.7, "max_detections": 300, "threshold_status": "provisional deployment config; not tuned in this smoke plan"},
        "fixture": fixture_manifest(fixture),
        "proposed_target_path": "target-local TensorRT FP16 build from the frozen checkpoint; never transfer the server RTX engine",
        "required_cctsdb_fixture": "first three U42 train images and accepted content hashes are not materialized in this local package; obtain before real smoke",
        "checks_before_execution": ["target identity/UUID and runtime versions", "direct checkpoint hash", "target-built engine hash and engine/runtime compatibility", "input/output binding names, shape, dtype and location", "explicit stream/event completion for copies, enqueue and output copy", "raw telemetry source and clock/boundary evidence"],
        "read_only_preflight": ["target-local TensorRT/Python availability without installation", "target disk space and output-directory absence", "checkpoint/config/fixture path existence and hashes", "no-write permission/path check for every allowed output"],
        "exact_allowed_writes": ["results/edge_readiness_v1/e2l1-007/<run_id>/manifest.json", "results/edge_readiness_v1/e2l1-007/<run_id>/input_fixture_manifest.json", "results/edge_readiness_v1/e2l1-007/<run_id>/adapter_events.json", "results/edge_readiness_v1/e2l1-007/<run_id>/output_contract.json", "results/edge_readiness_v1/e2l1-007/<run_id>/failure.json on failure"],
        "energy": {"status": "unavailable_without_validated_power_boundary_and_clock_alignment", "does_not_block": ["correctness", "synchronized latency"]},
        "allowed_resources": ["existing JetPack-provided TensorRT runtime on target", "existing approved checkpoint bytes", "accepted CCTSDB train-only fixture after materialization", "new target-scoped output directory"],
        "forbidden_actions": ["package installation", "sudo or mode/clock/fan changes", "server-engine transfer", "engine export/build", "model load/forward", "official-test images", "AP or FPS/energy scoring"],
        "stop_conditions": ["runtime/profile mismatch", "checkpoint or engine hash mismatch", "binding shape/dtype mismatch", "missing final synchronization", "wrong/native output contract", "missing CCTSDB fixture materialization"],
    }
