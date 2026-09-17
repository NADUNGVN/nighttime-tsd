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
)


class CudaMemoryOwner(Protocol):
    def allocate_device(self, nbytes: int, name: str) -> int: ...
    def copy_host_to_device(self, payload: bytes, device_pointer: int, stream_handle: int) -> None: ...
    def copy_device_to_host(self, device_pointer: int, destination: bytearray, stream_handle: int) -> None: ...
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
        self._ready = False
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
        self._host_outputs[device.name] = HostTensor(device.name, device.shape, device.dtype, device.nbytes, bytes(destination))
        self._pending_outputs.add(device.name)
        self._ready = False

    def mark_outputs_ready(self, stream_handle: int) -> None:
        if stream_handle != self.stream_handle:
            raise AdapterError("COPY_STREAM_MISMATCH", "completion stream differs from owned stream")
        expected = {binding.name for binding in self.engine.output_bindings()}
        if self._pending_outputs != expected:
            raise AdapterError("OUTPUT_NOT_READY", "not every output copy completed")
        self._ready = True

    def output_tensors(self) -> dict[str, HostTensor]:
        if not self._ready:
            raise AdapterError("OUTPUT_NOT_READY", "host outputs are unavailable before final completion")
        return {name: tensor for name, tensor in self._host_outputs.items() if name in {binding.name for binding in self.engine.output_bindings()}}

    def free(self) -> None:
        errors: list[Exception] = []
        for allocation in list(self._allocations.values()):
            try:
                self.memory.free_device(allocation.device_pointer)
            except Exception as exc:
                errors.append(exc)
        self._allocations.clear()
        self._host_outputs.clear()
        self._pending_outputs.clear()
        self._ready = False
        if errors:
            raise AdapterError("FREE_FAILED", "CUDA memory owner failed while releasing allocations", {"count": len(errors)})


class TensorRTProvider:
    """Lazy TensorRT 8.x-compatible provider; no import occurs at construction."""

    def __init__(self, config: AdapterConfig, *, module_loader: Callable[[str], Any] | None = None):
        self.config = config
        self._module_loader = module_loader or importlib.import_module
        self._trt: Any = None
        self._runtime: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._descriptor: EngineDescriptor | None = None

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
            logger = self._trt.Logger(self._trt.Logger.ERROR)
            self._runtime = self._trt.Runtime(logger)
            self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
            if self._engine is None:
                raise AdapterError("ENGINE_DESERIALIZE_FAILED", "TensorRT returned no engine")
            self._descriptor = self._describe_engine(engine_sha256)
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
                bindings.append(EngineBinding(name, "input" if self._engine.binding_is_input(index) else "output", shape, dtype, "device"))
        elif hasattr(self._engine, "num_io_tensors"):
            for index in range(int(self._engine.num_io_tensors)):
                name = str(self._engine.get_tensor_name(index))
                shape = tuple(int(dim) for dim in self._engine.get_tensor_shape(name))
                dtype = _dtype_name(self._engine.get_tensor_dtype(name))
                mode = str(self._engine.get_tensor_mode(name)).lower()
                bindings.append(EngineBinding(name, "input" if "input" in mode else "output", shape, dtype, "device"))
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
        return JetsonRuntimeAdapter(self.config, self._descriptor, self._context, stream, buffers)

    def close(self) -> None:
        self._context = None
        self._engine = None
        self._runtime = None
        self._trt = None
        self._descriptor = None

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
