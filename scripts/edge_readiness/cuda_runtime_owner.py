#!/usr/bin/env python3
"""Lazy CUDA 11.x runtime owner for small, explicitly bounded transfers.

The module imports only Python's ctypes. Loading libcudart and every CUDA call
are deferred until an explicit ``CudaRuntime.open``/owner operation. Copies are
deliberately synchronous: the contract does not silently treat a bytearray as
pinned host memory. The owner still exposes one stream and synchronizes it
before cleanup through the accepted ``OwnedBuffers`` boundary.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, Set, Tuple

from edge_readiness.edge_errors import AdapterError

CUDA_SUCCESS = 0
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
MAX_ALLOCATION_BYTES = 1024 * 1024
DEFAULT_LIBRARY_CANDIDATES = (
    "/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so.11.4.298",
    "/usr/local/cuda-11.4/lib64/libcudart.so.11.0",
    "/usr/local/cuda/lib64/libcudart.so.11.0",
    "libcudart.so.11.0",
)


class StreamLike(Protocol):
    handle: int

    def synchronize(self, stage: str) -> None: ...


class CudaRuntime:
    """Lazy libcudart loader with explicit symbol and return-code checks."""

    def __init__(self, *, library_loader: Optional[Callable[[str], Any]] = None, library_candidates: Tuple[str, ...] = DEFAULT_LIBRARY_CANDIDATES):
        self._library_loader = library_loader or ctypes.CDLL
        self.library_candidates = library_candidates
        self.library: Any = None
        self.library_path: Optional[str] = None
        self._streams: Set[int] = set()

    @property
    def loaded(self) -> bool:
        return self.library is not None

    def open(self) -> "CudaRuntime":
        if self.loaded:
            return self
        failures: List[str] = []
        for candidate in self.library_candidates:
            try:
                library = self._library_loader(candidate)
            except (OSError, AttributeError) as exc:
                failures.append(f"{candidate}:{type(exc).__name__}")
                continue
            self.library = library
            self.library_path = candidate
            try:
                self._configure_symbols()
            except AdapterError:
                self.library = None
                self.library_path = None
                raise
            return self
        raise AdapterError("CUDA_LIBRARY_UNAVAILABLE", "could not load any approved libcudart candidate", {"candidates": list(self.library_candidates), "failures": failures})

    def _configure_symbols(self) -> None:
        for name, argtypes, restype in (
            ("cudaMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int),
            ("cudaFree", [ctypes.c_void_p], ctypes.c_int),
            ("cudaMemcpy", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_int),
            ("cudaStreamCreate", [ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            ("cudaStreamDestroy", [ctypes.c_void_p], ctypes.c_int),
            ("cudaStreamSynchronize", [ctypes.c_void_p], ctypes.c_int),
            ("cudaGetErrorString", [ctypes.c_int], ctypes.c_char_p),
        ):
            function = getattr(self.library, name, None)
            if function is None:
                raise AdapterError("CUDA_SYMBOL_MISSING", "libcudart lacks a required symbol", {"symbol": name, "library": self.library_path})
            try:
                function.argtypes = argtypes
                function.restype = restype
            except (AttributeError, TypeError) as exc:
                raise AdapterError("CUDA_SYMBOL_NOT_CONFIGURABLE", "CUDA function does not accept a ctypes signature", {"symbol": name}) from exc

    def _require_open(self) -> Any:
        if not self.loaded:
            raise AdapterError("CUDA_RUNTIME_NOT_OPEN", "call CudaRuntime.open before using libcudart")
        return self.library

    def _error_text(self, code: int) -> str:
        try:
            raw = self._require_open().cudaGetErrorString(int(code))
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="replace")
            return str(raw)
        except Exception:
            return "cuda error text unavailable"

    def check(self, code: int, operation: str) -> None:
        if int(code) != CUDA_SUCCESS:
            raise AdapterError("CUDA_CALL_FAILED", f"{operation} returned CUDA error", {"operation": operation, "code": int(code), "message": self._error_text(int(code))})

    def allocate_device(self, nbytes: int) -> int:
        _validate_size(nbytes)
        pointer = ctypes.c_void_p()
        self.check(self._require_open().cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(nbytes)), "cudaMalloc")
        if not pointer.value:
            raise AdapterError("CUDA_NULL_POINTER", "cudaMalloc returned a null device pointer")
        return int(pointer.value)

    def free_device(self, pointer: int) -> None:
        _validate_pointer(pointer)
        self.check(self._require_open().cudaFree(ctypes.c_void_p(pointer)), "cudaFree")

    def copy_host_to_device(self, payload: bytes, pointer: int, nbytes: int) -> None:
        _validate_pointer(pointer)
        if len(payload) != nbytes:
            raise AdapterError("CUDA_COPY_SIZE_MISMATCH", "host payload size differs from requested copy", {"actual": len(payload), "expected": nbytes})
        source = ctypes.create_string_buffer(payload, nbytes)
        self.check(self._require_open().cudaMemcpy(ctypes.c_void_p(pointer), ctypes.cast(source, ctypes.c_void_p), ctypes.c_size_t(nbytes), CUDA_MEMCPY_HOST_TO_DEVICE), "cudaMemcpyHostToDevice")

    def copy_device_to_host(self, pointer: int, destination: bytearray, nbytes: int) -> None:
        _validate_pointer(pointer)
        if len(destination) != nbytes:
            raise AdapterError("CUDA_COPY_SIZE_MISMATCH", "host destination size differs from requested copy", {"actual": len(destination), "expected": nbytes})
        try:
            destination_pointer = ctypes.addressof((ctypes.c_ubyte * nbytes).from_buffer(destination))
        except (TypeError, ValueError) as exc:
            raise AdapterError("CUDA_HOST_BUFFER_INVALID", "D2H destination must be a writable contiguous bytearray") from exc
        self.check(self._require_open().cudaMemcpy(ctypes.c_void_p(destination_pointer), ctypes.c_void_p(pointer), ctypes.c_size_t(nbytes), CUDA_MEMCPY_DEVICE_TO_HOST), "cudaMemcpyDeviceToHost")

    def create_stream(self) -> "CudaRuntimeStream":
        pointer = ctypes.c_void_p()
        self.check(self._require_open().cudaStreamCreate(ctypes.byref(pointer)), "cudaStreamCreate")
        if not pointer.value:
            raise AdapterError("CUDA_NULL_STREAM", "cudaStreamCreate returned a null stream")
        handle = int(pointer.value)
        self._streams.add(handle)
        return CudaRuntimeStream(self, handle)

    def synchronize(self, handle: int, stage: str) -> None:
        _validate_pointer(handle)
        self.check(self._require_open().cudaStreamSynchronize(ctypes.c_void_p(handle)), f"cudaStreamSynchronize:{stage}")

    def destroy_stream(self, handle: int) -> None:
        if handle not in self._streams:
            return
        self.check(self._require_open().cudaStreamDestroy(ctypes.c_void_p(handle)), "cudaStreamDestroy")
        self._streams.discard(handle)

    def close(self) -> None:
        if self._streams:
            raise AdapterError("CUDA_STREAMS_ACTIVE", "close streams before closing the CUDA runtime", {"streams": sorted(self._streams)})
        self.library = None
        self.library_path = None


@dataclass
class CudaRuntimeStream:
    runtime: CudaRuntime
    handle: int
    _closed: bool = False

    def synchronize(self, stage: str) -> None:
        if self._closed:
            raise AdapterError("CUDA_STREAM_CLOSED", "cannot synchronize a closed CUDA stream")
        self.runtime.synchronize(self.handle, stage)

    def close(self) -> None:
        if not self._closed:
            self.runtime.destroy_stream(self.handle)
            self._closed = True


@dataclass
class CudaRuntimeMemoryOwner:
    """Concrete CudaMemoryOwner using synchronous, pageable-safe CUDA copies."""

    runtime: CudaRuntime
    stream: CudaRuntimeStream
    owner: str = "cuda-runtime-owner"

    context_policy: str = "uses the caller's current CUDA primary context; does not create or switch contexts"

    def __post_init__(self) -> None:
        if self.stream.runtime is not self.runtime:
            raise AdapterError("CUDA_RUNTIME_STREAM_MISMATCH", "stream belongs to a different CUDA runtime")
        # Addresses can be reused by CUDA. Keep only live generations plus a
        # short-lived idempotency set; allocation of a reused address starts a
        # fresh generation and removes it from the released set.
        self._live_allocations: Set[int] = set()
        self._released_allocations: Set[int] = set()

    def allocate_device(self, nbytes: int, name: str) -> int:
        del name
        pointer = self.runtime.allocate_device(nbytes)
        if pointer in self._live_allocations:
            raise AdapterError("CUDA_POINTER_REUSED_LIVE", "CUDA returned an address already live in this owner", {"pointer": pointer})
        self._live_allocations.add(pointer)
        self._released_allocations.discard(pointer)
        return pointer

    def copy_host_to_device(self, payload: bytes, device_pointer: int, stream_handle: int) -> None:
        self._check_stream(stream_handle)
        self._require_live(device_pointer)
        self.runtime.copy_host_to_device(payload, device_pointer, len(payload))

    def copy_device_to_host(self, device_pointer: int, destination: bytearray, stream_handle: int) -> None:
        self._check_stream(stream_handle)
        self._require_live(device_pointer)
        self.runtime.copy_device_to_host(device_pointer, destination, len(destination))

    def synchronize(self, stream_handle: int) -> None:
        self._check_stream(stream_handle)
        self.stream.synchronize("owner_cleanup")

    def free_device(self, device_pointer: int) -> None:
        if device_pointer in self._released_allocations:
            return
        self._require_live(device_pointer)
        self.runtime.free_device(device_pointer)
        self._live_allocations.remove(device_pointer)
        self._released_allocations.add(device_pointer)

    def _require_live(self, device_pointer: int) -> None:
        if device_pointer not in self._live_allocations:
            raise AdapterError("CUDA_POINTER_NOT_OWNED", "CUDA pointer is not live in this owner", {"pointer": device_pointer})

    def _check_stream(self, stream_handle: int) -> None:
        if stream_handle != self.stream.handle:
            raise AdapterError("COPY_STREAM_MISMATCH", "CUDA owner received a non-owned stream", {"expected": self.stream.handle, "observed": stream_handle})


def _validate_size(nbytes: int, *, max_bytes: Optional[int] = None) -> None:
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0 or (max_bytes is not None and nbytes > max_bytes):
        limit = max_bytes if max_bytes is not None else "unbounded-by-smoke"
        raise AdapterError("CUDA_ALLOCATION_LIMIT", "size must be a positive integer within the configured limit", {"nbytes": nbytes, "max": limit})


def _validate_pointer(pointer: int) -> None:
    if isinstance(pointer, bool) or not isinstance(pointer, int) or pointer <= 0:
        raise AdapterError("CUDA_POINTER_INVALID", "CUDA pointer/stream handle must be a positive integer")
