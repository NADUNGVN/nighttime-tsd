#!/usr/bin/env python3
"""Shared CPU-only YOLO11n image preprocessing boundary.

This module intentionally imports only NumPy and OpenCV at call time. It does
not inspect CUDA, load a model, initialize ORT, or run source-runtime
preflight. The resize/padding steps mirror the pinned Ultralytics LetterBox
8.4.102 path and are shared by source and target stages.
"""
from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


IMAGE_SIZE = (640, 640)
PREPROCESS_ID = "yolo11n-letterbox-ultralytics-8.4.102-v1"


class ImagePreprocessError(ValueError):
    pass


@dataclass
class PreprocessedImage:
    payload: bytes
    shape: Tuple[int, ...]
    dtype: str
    byteorder: str
    finite: bool
    value: Any
    metadata: Dict[str, Any]


def _modules(np_module: Any = None, cv2_module: Any = None) -> Tuple[Any, Any]:
    try:
        np = np_module if np_module is not None else importlib.import_module("numpy")
        cv2 = cv2_module if cv2_module is not None else importlib.import_module("cv2")
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise ImagePreprocessError("IMAGE_PREPROCESS_DEPENDENCY_MISSING:" + str(missing)) from exc
    return np, cv2


def decode_image_bytes(image_bytes: bytes, *, image_id: str = "unknown", np_module: Any = None, cv2_module: Any = None) -> Any:
    np, cv2 = _modules(np_module, cv2_module)
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ImagePreprocessError("IMAGE_DECODE_FAILED:" + str(image_id))
    return image


def inspect_image_bytes(image_bytes: bytes, *, image_id: str = "unknown", np_module: Any = None, cv2_module: Any = None) -> Dict[str, Any]:
    image = decode_image_bytes(image_bytes, image_id=image_id, np_module=np_module, cv2_module=cv2_module)
    return {"orig_shape": [int(image.shape[0]), int(image.shape[1])], "channels": int(image.shape[2]), "image_sha256": hashlib.sha256(image_bytes).hexdigest(), "image_bytes": len(image_bytes)}


def preprocess_image_bytes(image_bytes: bytes, *, image_id: str = "unknown", np_module: Any = None, cv2_module: Any = None) -> PreprocessedImage:
    """Decode BGR bytes and return the exact frozen 1x3x640x640 CPU tensor."""
    np, cv2 = _modules(np_module, cv2_module)
    image = decode_image_bytes(image_bytes, image_id=image_id, np_module=np, cv2_module=cv2)
    height, width = int(image.shape[0]), int(image.shape[1])
    target_h, target_w = IMAGE_SIZE

    # These are the pinned LetterBox.get_params equations. Keep the rounding
    # and split-border rules identical to that implementation.
    gain = min(target_h / height, target_w / width)
    resized_w, resized_h = round(width * gain), round(height * gain)
    dw, dh = target_w - resized_w, target_h - resized_h
    dw /= 2.0
    dh /= 2.0
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    if (width, height) != (resized_w, resized_h):
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image
    letterboxed_bgr = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if tuple(letterboxed_bgr.shape[:2]) != IMAGE_SIZE:
        raise ImagePreprocessError("LETTERBOX_OUTPUT_SHAPE_MISMATCH:" + str(image_id))

    rgb = letterboxed_bgr[:, :, ::-1]
    array = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...], dtype=np.float32) / 255.0
    array = np.ascontiguousarray(array, dtype=np.dtype("<f4"))
    payload = array.tobytes(order="C")
    metadata = {
        "preprocess_id": PREPROCESS_ID,
        "original_shape": [height, width, int(image.shape[2])],
        "resized_shape": [int(letterboxed_bgr.shape[0]), int(letterboxed_bgr.shape[1]), int(letterboxed_bgr.shape[2])],
        "letterbox_transform": {
            "gain": [float(gain), float(gain)],
            "pad": [int(left), int(top)],
            "resized_unpadded_shape": [int(resized_h), int(resized_w)],
            "output_shape": [int(letterboxed_bgr.shape[0]), int(letterboxed_bgr.shape[1])],
            "borders": [int(left), int(top), int(right), int(bottom)],
            "rounding": "round(dimension*gain); split border with round(delta-0.1)/round(delta+0.1)",
        },
        "letterbox": {"new_shape": [target_h, target_w], "auto": False, "scale_fill": False, "scaleup": True, "center": True, "stride": 32, "padding_value": 114, "interpolation": "cv2.INTER_LINEAR"},
        "color": "BGR_to_RGB",
        "layout": "NCHW",
        "normalization": "pixel/255.0",
        "contiguous": bool(array.flags["C_CONTIGUOUS"]),
        "dtype_endian": "<f4",
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "input_bytes": len(payload),
    }
    if payload != array.tobytes(order="C"):
        raise ImagePreprocessError("INPUT_FREEZE_MISMATCH:" + str(image_id))
    return PreprocessedImage(payload, tuple(int(value) for value in array.shape), "float32", "little", bool(np.isfinite(array).all()), array, metadata)


def preprocess_image(image_path: Path, *, image_id: Optional[str] = None) -> PreprocessedImage:
    path = Path(image_path)
    try:
        image_bytes = path.read_bytes()
    except OSError as exc:
        raise ImagePreprocessError("IMAGE_READ_FAILED:{}:{}".format(image_id or path.name, exc)) from exc
    return preprocess_image_bytes(image_bytes, image_id=image_id or path.name)
