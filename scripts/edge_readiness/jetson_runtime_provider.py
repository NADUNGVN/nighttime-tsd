#!/usr/bin/env python3
"""Lazy real-runtime bridge for the approved E2 TensorRT stack.

Importing this module is CPU-safe. TensorRT is loaded only by ``load_engine``;
CUDA allocators/streams are supplied by an injected memory owner. This keeps
real mode separate from the CPU/mock bridge and makes partial cleanup testable.
"""
from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from edge_readiness.jetson_adapter import (
    AdapterConfig,
    AdapterError,
    DeviceTensor,
    EngineBinding,
    EngineDescriptor,
    HostTensor,
    JetsonRuntimeAdapter,
    expected_nbytes,
    validate_yolo11n_native_output_contract,
)


class CudaMemoryOwner(Protocol):
    def allocate_device(self, nbytes: int, name: str) -> int: ...
    def copy_host_to_device(self, payload: bytes, device_pointer: int, stream_handle: int) -> None: ...
    def copy_device_to_host(self, device_pointer: int, destination: bytearray, stream_handle: int) -> None: ...
    def synchronize(self, stream_handle: int) -> None: ...
    def free_device(self, device_pointer: int) -> None: ...


@dataclass
class OwnedBuffers:
    """BufferManagerLike implementation backed by an injected CUDA owner."""

    engine: EngineDescriptor
    memory: CudaMemoryOwner
    stream_handle: int
    owner: str = "jetson-cuda-owner"
    lifetime_id: str = "unassigned"

    def __post_init__(self) -> None:
        self.lifetime_id = f"{self.owner}:{id(self)}"
        self._allocations: dict[str, DeviceTensor] = {}
        self._host_outputs: dict[str, HostTensor] = {}
        self._pending_outputs: set[str] = set()
        self._pending_destinations: dict[str, bytearray] = {}
        self._ready = False
        self._freed = False
        try:
            for binding in self.engine.bindings:
                pointer = self.memory.allocate_device(expected_nbytes(binding.shape, binding.dtype), binding.name)
                self._allocations[binding.name] = DeviceTensor(binding.name, binding.shape, binding.dtype, expected_nbytes(binding.shape, binding.dtype), pointer, self.owner, self.lifetime_id)
        except Exception as exc:
            self.free()
            if isinstance(exc, AdapterError):
                raise
            raise AdapterError("ALLOCATION_FAILED", "CUDA memory owner failed during partial allocation") from exc

    def copy_stream_handle(self) -> int:
        return self.stream_handle

    def begin_inference(self) -> None:
        if self._freed:
            raise AdapterError("BUFFER_OWNER_FREED", "buffer owner cannot be reused after free")
        self._host_outputs.clear()
        self._pending_outputs.clear()
        self._pending_destinations.clear()
        self._ready = False

    def binding_buffers(self) -> dict[str, DeviceTensor]:
        return dict(self._allocations)

    def copy_host_to_device(self, host: HostTensor, device: DeviceTensor, stream_handle: int) -> None:
        if stream_handle != self.stream_handle:
            raise AdapterError("COPY_STREAM_MISMATCH", "H2D copy stream differs from owned stream")
        self.memory.copy_host_to_device(host.payload, device.device_pointer, stream_handle)

    def copy_device_to_host(self, device: DeviceTensor, stream_handle: int) -> None:
        if stream_handle != self.stream_handle:
            raise AdapterError("COPY_STREAM_MISMATCH", "D2H copy stream differs from owned stream")
        destination = bytearray(device.nbytes)
        self.memory.copy_device_to_host(device.device_pointer, destination, stream_handle)
        # The owner may enqueue an asynchronous D2H copy. Do not snapshot the
        # destination here: at this point it can still contain its zero-fill.
        # ``mark_outputs_ready`` is called by the adapter only after the
        # runtime stream's completion boundary.
        self._pending_destinations[device.name] = destination
        self._pending_outputs.add(device.name)
        self._ready = False

    def mark_outputs_ready(self, stream_handle: int) -> None:
        if stream_handle != self.stream_handle:
            raise AdapterError("COPY_STREAM_MISMATCH", "completion stream differs from owned stream")
        expected = {binding.name for binding in self.engine.output_bindings()}
        if self._pending_outputs != expected:
            raise AdapterError("OUTPUT_NOT_READY", "not every output copy completed")
        if set(self._pending_destinations) != expected:
            raise AdapterError("OUTPUT_NOT_READY", "not every output destination is staged")
        self._host_outputs = {}
        for binding in self.engine.output_bindings():
            destination = self._pending_destinations[binding.name]
            expected_size = expected_nbytes(binding.shape, binding.dtype)
            if len(destination) != expected_size:
                raise AdapterError("OUTPUT_CONTRACT_MISMATCH", "completed host destination size does not match engine binding", {"binding": binding.name})
            # Snapshot only after stream completion. This is the ownership
            # handoff from mutable async staging memory to an immutable result.
            self._host_outputs[binding.name] = HostTensor(binding.name, binding.shape, binding.dtype, expected_size, bytes(destination))
        self._pending_destinations.clear()
        self._pending_outputs.clear()
        self._ready = True

    def output_tensors(self) -> dict[str, HostTensor]:
        if not self._ready:
            raise AdapterError("OUTPUT_NOT_READY", "host outputs are unavailable before final completion")
        return {name: tensor for name, tensor in self._host_outputs.items() if name in {binding.name for binding in self.engine.output_bindings()}}

    def free(self) -> None:
        if self._freed and not self._allocations:
            return
        errors: list[Exception] = []
        try:
            self.memory.synchronize(self.stream_handle)
        except Exception as exc:
            errors.append(exc)
        for allocation in list(self._allocations.values()):
            try:
                self.memory.free_device(allocation.device_pointer)
            except Exception as exc:
                errors.append(exc)
        self._allocations.clear()
        self._host_outputs.clear()
        self._pending_outputs.clear()
        self._pending_destinations.clear()
        self._ready = False
        self._freed = True
        if errors:
            raise AdapterError("FREE_FAILED", "CUDA memory owner failed while releasing allocations", {"count": len(errors)})


class TensorRTProvider:
    """Lazy TensorRT 8.x-compatible provider; no import occurs at construction."""

    def __init__(self, config: AdapterConfig, *, module_loader: Callable[[str], Any] | None = None):
        self.config = config
        self._module_loader = module_loader or importlib.import_module
        self._trt: Any = None
        self._logger: Any = None
        self._runtime: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._descriptor: EngineDescriptor | None = None
        self._native_output_contract: dict[str, Any] | None = None

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def load_engine(self, engine_bytes: bytes, engine_sha256: str) -> EngineDescriptor:
        if hashlib.sha256(engine_bytes).hexdigest() != engine_sha256.lower():
            raise AdapterError("ENGINE_HASH_MISMATCH", "engine bytes do not match supplied SHA-256")
        if self.loaded:
            raise AdapterError("PROVIDER_ALREADY_LOADED", "provider accepts one immutable engine per lifetime")
        self.config.validate()
        try:
            self._trt = self._module_loader("tensorrt")
            observed_runtime_version = _observed_runtime_version(self._trt)
            if not observed_runtime_version.startswith(self.config.validate().expected_runtime_prefix):
                raise AdapterError("RUNTIME_VERSION_MISMATCH", "imported TensorRT version does not match target profile", {"expected_prefix": self.config.validate().expected_runtime_prefix, "observed": observed_runtime_version})
            self._logger = self._trt.Logger(self._trt.Logger.ERROR)
            self._runtime = self._trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
            if self._engine is None:
                raise AdapterError("ENGINE_DESERIALIZE_FAILED", "TensorRT returned no engine")
            self._descriptor = self._describe_engine(engine_sha256)
            self._native_output_contract = validate_yolo11n_native_output_contract(self._descriptor, self.config)
            return self._descriptor
        except AdapterError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise AdapterError("ENGINE_DESERIALIZE_FAILED", "TensorRT engine deserialization failed") from exc

    def _describe_engine(self, engine_sha256: str) -> EngineDescriptor:
        if self._engine is None:
            raise AdapterError("ENGINE_NOT_LOADED", "load_engine is required before introspection")
        bindings: list[EngineBinding] = []
        if hasattr(self._engine, "num_bindings"):
            for index in range(int(self._engine.num_bindings)):
                name = str(self._engine.get_binding_name(index))
                shape = tuple(int(dim) for dim in self._engine.get_binding_shape(index))
                dtype = _dtype_name(self._engine.get_binding_dtype(index))
                location_method = getattr(self._engine, "get_location", None)
                location = _location_name(location_method(index)) if callable(location_method) else _missing_location()
                bindings.append(EngineBinding(name, "input" if self._engine.binding_is_input(index) else "output", shape, dtype, location))
        elif hasattr(self._engine, "num_io_tensors"):
            for index in range(int(self._engine.num_io_tensors)):
                name = str(self._engine.get_tensor_name(index))
                shape = tuple(int(dim) for dim in self._engine.get_tensor_shape(name))
                dtype = _dtype_name(self._engine.get_tensor_dtype(name))
                mode = str(self._engine.get_tensor_mode(name)).lower()
                location_method = getattr(self._engine, "get_tensor_location", None)
                location = _location_name(location_method(name)) if callable(location_method) else _missing_location()
                bindings.append(EngineBinding(name, "input" if "input" in mode else "output", shape, dtype, location))
        else:
            raise AdapterError("BINDING_API_UNAVAILABLE", "TensorRT engine exposes neither legacy nor named tensor introspection")
        descriptor = EngineDescriptor(self.config.runtime_version, engine_sha256.lower(), tuple(bindings))
        descriptor.validate(self.config)
        return descriptor

    def create_adapter(self, stream: Any, buffers: OwnedBuffers) -> JetsonRuntimeAdapter:
        if self._engine is None or self._descriptor is None:
            raise AdapterError("ENGINE_NOT_LOADED", "load_engine is required before creating an adapter")
        if self._context is None:
            self._context = self._engine.create_execution_context()
        if self._context is None:
            raise AdapterError("EXECUTION_CONTEXT_UNAVAILABLE", "TensorRT returned no execution context")
        return JetsonRuntimeAdapter(self.config, self._descriptor, self._context, stream, buffers)

    def close(self) -> None:
        self._context = None
        self._engine = None
        self._runtime = None
        # Keep logger alive until Runtime is released; TensorRT may retain the
        # logger through the runtime lifetime.
        self._logger = None
        self._trt = None
        self._descriptor = None
        self._native_output_contract = None

    def __enter__(self) -> "TensorRTProvider":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _dtype_name(value: Any) -> str:
    text = str(value).lower()
    if "half" in text or "float16" in text:
        return "float16"
    if "float" in text or "float32" in text:
        return "float32"
    raise AdapterError("UNSUPPORTED_ENGINE_DTYPE", "only float16 and float32 bindings are supported", {"observed": str(value)})


def _observed_runtime_version(module: Any) -> str:
    version = getattr(module, "__version__", None)
    if not isinstance(version, str) or not version.strip():
        raise AdapterError("RUNTIME_VERSION_UNAVAILABLE", "imported TensorRT module does not expose an observed version")
    return version.strip()


def _location_name(value: Any) -> str:
    text = str(value).lower()
    if "device" in text:
        return "device"
    if "host" in text:
        return "host"
    raise AdapterError("BINDING_LOCATION_UNKNOWN", "TensorRT returned an unknown binding location", {"observed": str(value)})


def _missing_location() -> str:
    raise AdapterError("BINDING_LOCATION_UNAVAILABLE", "TensorRT engine does not expose binding location introspection")
