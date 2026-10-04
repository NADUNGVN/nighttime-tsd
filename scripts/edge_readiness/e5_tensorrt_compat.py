"""Isolated CPU/mock contract for a prospective Jetson Orin Nano Super lane.

This module deliberately does not import TensorRT, CUDA, OpenCV, or the E2
adapter. Hardware-facing objects are injected so the contract can be tested
without loading a device runtime or dispatching a model.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple


E5_TENSORRT_VERSION_PREFIX = "10.3"
E5_CUDA_MAJOR = 12
E5_INPUT_NAME = "images"
E5_INPUT_SHAPE = (1, 3, 640, 640)
E5_OUTPUT_NAME = "output0"
E5_OUTPUT_SHAPE = (1, 7, 8400)
E5_DTYPE = "float32"
E5_LOCATION = "DEVICE"

CUDA12_LIBRARY_CANDIDATES = (
    "/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12",
    "/usr/local/cuda-12.6/lib64/libcudart.so.12",
    "/usr/local/cuda-12/targets/aarch64-linux/lib/libcudart.so.12",
    "/usr/lib/aarch64-linux-gnu/libcudart.so.12",
    "libcudart.so.12",
)

CUDA_REQUIRED_SYMBOLS = (
    "cudaMalloc",
    "cudaFree",
    "cudaMemcpy",
    "cudaStreamCreate",
    "cudaStreamDestroy",
    "cudaStreamSynchronize",
    "cudaGetErrorString",
)


class E5CompatibilityError(RuntimeError):
    """Structured failure from the isolated E5 compatibility contract."""

    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class NamedTensor:
    name: str
    mode: str
    shape: Tuple[int, ...]
    dtype: str
    location: str

    @property
    def nbytes(self) -> int:
        sizes = {"float32": 4, "float16": 2, "int32": 4, "int8": 1, "bool": 1}
        return _element_count(self.shape) * sizes[self.dtype]


class E5MemoryOwner(Protocol):
    def allocate_device(self, nbytes: int, name: str) -> int: ...
    def copy_host_to_device(self, payload: bytes, device_pointer: int, stream_handle: int) -> None: ...
    def copy_device_to_host(self, device_pointer: int, destination: bytearray, stream_handle: int) -> None: ...
    def synchronize(self, stream_handle: int) -> None: ...
    def free_device(self, device_pointer: int) -> None: ...


def validate_tensorrt_runtime_version(version: str) -> str:
    observed = str(version).strip()
    if not observed.startswith(E5_TENSORRT_VERSION_PREFIX + ".") and observed != E5_TENSORRT_VERSION_PREFIX:
        raise E5CompatibilityError(
            "TENSORRT_VERSION_MISMATCH",
            "E5 named-I/O profile requires TensorRT 10.3.x",
            {"expected_prefix": E5_TENSORRT_VERSION_PREFIX, "observed": observed},
        )
    return observed


def inspect_e5_named_io(engine: Any, runtime_version: str) -> Dict[str, NamedTensor]:
    """Require the frozen two-tensor, device-resident TensorRT 10.3 contract."""
    validate_tensorrt_runtime_version(runtime_version)
    required = (
        "num_io_tensors",
        "get_tensor_name",
        "get_tensor_mode",
        "get_tensor_shape",
        "get_tensor_dtype",
        "get_tensor_location",
    )
    if any(not hasattr(engine, name) for name in required):
        raise E5CompatibilityError("NAMED_IO_API_MISSING", "engine lacks TensorRT named-I/O introspection")
    try:
        count = int(engine.num_io_tensors)
    except Exception as exc:
        raise E5CompatibilityError("NAMED_IO_COUNT_INVALID", "engine I/O tensor count is unreadable") from exc
    if count != 2:
        raise E5CompatibilityError("NAMED_IO_COUNT_MISMATCH", "E5 profile requires exactly one input and one output", {"observed": count})

    observed: Dict[str, NamedTensor] = {}
    for index in range(count):
        try:
            name = str(engine.get_tensor_name(index))
            if name in observed:
                raise E5CompatibilityError("NAMED_IO_DUPLICATE_NAME", "engine repeats a tensor name", {"name": name})
            raw_shape = tuple(engine.get_tensor_shape(name))
            if not raw_shape or any(isinstance(dimension, bool) or int(dimension) <= 0 for dimension in raw_shape):
                raise E5CompatibilityError("NAMED_IO_DYNAMIC_SHAPE", "dynamic or non-positive tensor dimensions are not accepted", {"name": name, "shape": list(raw_shape)})
            shape = tuple(int(dimension) for dimension in raw_shape)
            mode = _enum_token(engine.get_tensor_mode(name))
            dtype = _normalize_dtype(engine.get_tensor_dtype(name))
            location = _normalize_location(engine.get_tensor_location(name))
        except E5CompatibilityError:
            raise
        except Exception as exc:
            raise E5CompatibilityError("NAMED_IO_INTROSPECTION_FAILED", "could not inspect every engine tensor") from exc
        observed[name] = NamedTensor(name, mode, shape, dtype, location)

    expected = {
        E5_INPUT_NAME: NamedTensor(E5_INPUT_NAME, "INPUT", E5_INPUT_SHAPE, E5_DTYPE, E5_LOCATION),
        E5_OUTPUT_NAME: NamedTensor(E5_OUTPUT_NAME, "OUTPUT", E5_OUTPUT_SHAPE, E5_DTYPE, E5_LOCATION),
    }
    if observed != expected:
        raise E5CompatibilityError(
            "NAMED_IO_CONTRACT_MISMATCH",
            "engine named-I/O mode, dtype, location, or shape differs from the frozen E5 profile",
            {"expected": _tensor_summary(expected), "observed": _tensor_summary(observed)},
        )
    return observed


def validate_hash_bindings(
    manifest: Mapping[str, Any],
    source_onnx_bytes: bytes,
    input_bytes: bytes,
    engine_bytes: bytes,
) -> Dict[str, str]:
    """Bind the engine to its source ONNX and the actual per-call input bytes."""
    if not isinstance(manifest, Mapping):
        raise E5CompatibilityError("HASH_MANIFEST_INVALID", "hash binding manifest must be a mapping")
    actual = {
        "source_onnx_sha256": _payload_sha256(source_onnx_bytes, "source ONNX"),
        "input_sha256": _payload_sha256(input_bytes, "input tensor"),
        "engine_sha256": _payload_sha256(engine_bytes, "engine"),
    }
    declared: Dict[str, str] = {}
    for name in ("source_onnx_sha256", "input_sha256", "engine_sha256", "engine_source_onnx_sha256"):
        value = manifest.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
            raise E5CompatibilityError("HASH_MANIFEST_INVALID", "manifest contains a missing or invalid SHA-256", {"field": name})
        declared[name] = value.lower()
    mismatches = {
        name: {"declared": declared[name], "actual": digest}
        for name, digest in actual.items()
        if declared[name] != digest
    }
    if declared["engine_source_onnx_sha256"] != actual["source_onnx_sha256"]:
        mismatches["engine_source_onnx_sha256"] = {
            "declared": declared["engine_source_onnx_sha256"],
            "actual_source_onnx_sha256": actual["source_onnx_sha256"],
        }
    if mismatches:
        raise E5CompatibilityError("HASH_BINDING_MISMATCH", "source, input, or engine bytes do not match the manifest", mismatches)
    return actual


def execute_e5_v3_once(
    engine: Any,
    context: Any,
    memory: E5MemoryOwner,
    stream_handle: int,
    input_bytes: bytes,
    source_onnx_bytes: bytes,
    engine_bytes: bytes,
    manifest: Mapping[str, Any],
    runtime_version: str,
) -> Dict[str, Any]:
    """Run one injected named-I/O v3 transaction and expose only synchronized output.

    All copies and execution use one stream. If a final drain cannot be
    confirmed, live allocations are intentionally retained rather than freed
    while asynchronous work may still reference them. This synchronous
    contract does not impose a wall-clock deadline; a future device runner
    must supervise it in a killable child process before any target GO.
    """
    if isinstance(stream_handle, bool) or not isinstance(stream_handle, int) or stream_handle < 0:
        raise E5CompatibilityError("STREAM_HANDLE_INVALID", "stream handle must be a non-negative integer")
    needed_memory_methods = ("allocate_device", "copy_host_to_device", "copy_device_to_host", "synchronize", "free_device")
    if any(not callable(getattr(memory, name, None)) for name in needed_memory_methods):
        raise E5CompatibilityError("MEMORY_OWNER_API_MISSING", "injected memory owner is incomplete")
    if not callable(getattr(context, "set_tensor_address", None)) or not callable(getattr(context, "execute_async_v3", None)):
        raise E5CompatibilityError("EXECUTION_CONTEXT_API_MISSING", "context lacks named address or execute_async_v3")

    tensors = inspect_e5_named_io(engine, runtime_version)
    hashes = validate_hash_bindings(manifest, source_onnx_bytes, input_bytes, engine_bytes)
    input_tensor = tensors[E5_INPUT_NAME]
    output_tensor = tensors[E5_OUTPUT_NAME]
    if len(input_bytes) != input_tensor.nbytes:
        raise E5CompatibilityError("INPUT_BYTE_COUNT_MISMATCH", "input bytes do not fill the exact named-I/O tensor", {"expected": input_tensor.nbytes, "observed": len(input_bytes)})

    allocations: Dict[str, int] = {}
    output_host = bytearray(output_tensor.nbytes)
    result: Optional[Dict[str, Any]] = None
    primary_error: Optional[BaseException] = None
    cleanup_errors: List[Tuple[str, BaseException]] = []
    try:
        for tensor in (input_tensor, output_tensor):
            pointer = memory.allocate_device(tensor.nbytes, tensor.name)
            if isinstance(pointer, bool) or not isinstance(pointer, int) or pointer <= 0 or pointer in allocations.values():
                raise E5CompatibilityError("DEVICE_POINTER_INVALID", "memory owner returned an invalid or duplicate pointer", {"tensor": tensor.name})
            allocations[tensor.name] = pointer

        for tensor in (input_tensor, output_tensor):
            try:
                accepted = context.set_tensor_address(tensor.name, allocations[tensor.name])
            except Exception as exc:
                raise E5CompatibilityError("TENSOR_ADDRESS_FAILED", "TensorRT rejected a named device address", {"tensor": tensor.name}) from exc
            if accepted is not True:
                raise E5CompatibilityError("TENSOR_ADDRESS_FAILED", "TensorRT did not accept a named device address", {"tensor": tensor.name})

        memory.copy_host_to_device(input_bytes, allocations[E5_INPUT_NAME], stream_handle)
        try:
            enqueued = context.execute_async_v3(stream_handle)
        except Exception as exc:
            raise E5CompatibilityError("EXECUTE_ASYNC_V3_FAILED", "TensorRT v3 enqueue raised an exception") from exc
        if enqueued is not True:
            raise E5CompatibilityError("EXECUTE_ASYNC_V3_FAILED", "TensorRT v3 enqueue returned false")

        memory.copy_device_to_host(allocations[E5_OUTPUT_NAME], output_host, stream_handle)
        try:
            memory.synchronize(stream_handle)
        except Exception as exc:
            raise E5CompatibilityError("OUTPUT_SYNC_FAILED", "output copy completion could not be confirmed") from exc
        fresh_output = bytes(output_host)
        result = {
            "output_bytes": fresh_output,
            "output_sha256": hashlib.sha256(fresh_output).hexdigest(),
            "source_onnx_sha256": hashes["source_onnx_sha256"],
            "input_sha256": hashes["input_sha256"],
            "engine_sha256": hashes["engine_sha256"],
            "output_shape": list(output_tensor.shape),
            "output_dtype": output_tensor.dtype,
            "stream_handle": stream_handle,
            "output_copy_synchronized": True,
            "fresh_host_output_buffer": True,
        }
    except BaseException as exc:
        primary_error = exc
    finally:
        try:
            memory.synchronize(stream_handle)
        except BaseException as exc:
            cleanup_errors.append(("synchronize_before_free", exc))
        if not cleanup_errors:
            for pointer in reversed(list(allocations.values())):
                try:
                    memory.free_device(pointer)
                except BaseException as exc:
                    cleanup_errors.append(("free_device", exc))

    if primary_error is not None:
        if cleanup_errors:
            raise E5CompatibilityError(
                "OPERATION_AND_CLEANUP_FAILED",
                "E5 transaction failed and cleanup was incomplete",
                {
                    "primary_code": getattr(primary_error, "code", type(primary_error).__name__),
                    "cleanup_failures": [stage for stage, _error in cleanup_errors],
                    "live_allocations_retained": bool(allocations),
                },
            ) from primary_error
        if isinstance(primary_error, E5CompatibilityError):
            raise primary_error
        raise E5CompatibilityError("E5_RUNTIME_CALL_FAILED", "an injected CUDA/runtime call failed") from primary_error
    if cleanup_errors:
        raise E5CompatibilityError(
            "CLEANUP_FAILED",
            "E5 transaction output is withheld because cleanup did not complete",
            {
                "cleanup_failures": [stage for stage, _error in cleanup_errors],
                "live_allocations_retained": bool(allocations),
            },
        ) from cleanup_errors[0][1]
    if result is None:
        raise E5CompatibilityError("RESULT_MISSING", "E5 transaction ended without a completed result")
    return result


def require_opencv(module_loader: Optional[Callable[[str], Any]] = None) -> Any:
    """Fail closed if the approved OpenCV preprocessing implementation is absent."""
    loader = module_loader or importlib.import_module
    try:
        module = loader("cv2")
    except Exception as exc:
        raise E5CompatibilityError("OPENCV_REQUIRED", "E5 preprocessing requires OpenCV; no fallback is accepted") from exc
    required = ("imdecode", "IMREAD_COLOR", "resize", "INTER_LINEAR", "copyMakeBorder", "BORDER_CONSTANT")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise E5CompatibilityError("OPENCV_API_INCOMPLETE", "OpenCV preprocessing API is incomplete", {"missing": missing})
    return module


class Cuda12LibraryResolver:
    """Injectable CUDA 12.x candidate selector; never loads a library on import."""

    def __init__(
        self,
        file_exists: Optional[Callable[[str], bool]] = None,
        library_loader: Optional[Callable[[str], Any]] = None,
        candidates: Tuple[str, ...] = CUDA12_LIBRARY_CANDIDATES,
    ):
        self.file_exists = file_exists or os.path.isfile
        self.library_loader = library_loader or ctypes.CDLL
        self.candidates = tuple(candidates)
        self.library = None
        self.library_path = None
        for candidate in self.candidates:
            if not _looks_like_cuda12_candidate(candidate):
                raise E5CompatibilityError("CUDA_CANDIDATE_NOT_12X", "candidate list contains a non-CUDA-12 library", {"candidate": candidate})

    def open(self) -> Tuple[str, Any]:
        failures: List[str] = []
        for candidate in self.candidates:
            is_soname = "/" not in candidate and "\\" not in candidate
            if not is_soname and not self.file_exists(candidate):
                failures.append(candidate + ":missing")
                continue
            try:
                library = self.library_loader(candidate)
            except Exception as exc:
                failures.append(candidate + ":" + type(exc).__name__)
                continue
            missing_symbols = [name for name in CUDA_REQUIRED_SYMBOLS if not callable(getattr(library, name, None))]
            if missing_symbols:
                failures.append(candidate + ":missing_symbols=" + ",".join(missing_symbols))
                continue
            self.library = library
            self.library_path = candidate
            return candidate, library
        raise E5CompatibilityError(
            "CUDA12_LIBRARY_UNAVAILABLE",
            "no approved CUDA 12.x candidate loaded with the required runtime symbols",
            {"candidates": list(self.candidates), "failures": failures},
        )


def _element_count(shape: Tuple[int, ...]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def _enum_token(value: Any) -> str:
    return str(value).rsplit(".", 1)[-1].strip().upper()


def _normalize_dtype(value: Any) -> str:
    token = _enum_token(value)
    aliases = {"FLOAT": "float32", "FLOAT32": "float32", "HALF": "float16", "FLOAT16": "float16", "INT32": "int32", "INT8": "int8", "BOOL": "bool"}
    return aliases.get(token, token.lower())


def _normalize_location(value: Any) -> str:
    token = _enum_token(value)
    if token in ("GPU", "DEVICE"):
        return "DEVICE"
    if token == "HOST":
        return "HOST"
    return token


def _payload_sha256(payload: bytes, label: str) -> str:
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        raise E5CompatibilityError("HASH_PAYLOAD_INVALID", "hash-bound payload must be non-empty bytes", {"payload": label})
    return hashlib.sha256(bytes(payload)).hexdigest()


def _tensor_summary(tensors: Mapping[str, NamedTensor]) -> Dict[str, Dict[str, Any]]:
    return {
        name: {"mode": tensor.mode, "shape": list(tensor.shape), "dtype": tensor.dtype, "location": tensor.location}
        for name, tensor in sorted(tensors.items())
    }


def _looks_like_cuda12_candidate(candidate: str) -> bool:
    normalized = candidate.replace("\\", "/").lower()
    return "/cuda-12" in normalized or ".so.12" in normalized or normalized.endswith("libcudart.so.12")
