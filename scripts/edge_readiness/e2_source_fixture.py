#!/usr/bin/env python3
"""Read-only CCTSDB fixture binding and decoder-independent preprocessing.

The fixture manifest binds actual JPEG bytes without copying them into Git.
Image decoding/resizing is injected because this local environment does not
ship the source reference's Pillow/OpenCV stack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

FIXTURE_IDS = ("00006", "00009", "00028")
FIXTURE_RELATIVE_PATHS = tuple(f"data/processed/cctsdb2021_clean/train/images/{image_id}.jpg" for image_id in FIXTURE_IDS)
ACCEPTED_IMAGE_SHA256 = {
    "00006": "a4bdd9e4968a005f2c8223d0b10adcf8104f0c95c1086b1631721d940aa55434",
    "00009": "48b81a7b018827fcb92b589eee6ed389134bc9e711ba5751347849f49d370354",
    "00028": "e42cc181f5e8f6bb9f4349625b68cb65b94ac42e01a16726be6bc35446c31cde",
}
CANONICAL_BINDING_MANIFEST = "results/measurement_audit_v1/precision_head_confirmation_readiness_r1r3_v2/readiness_manifest.json"


class FixtureError(ValueError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.details = code, message, details or {}

    def as_dict(self) -> dict[str, Any]:
        return {"status": "error", "error": {"code": self.code, "message": self.message, "details": self.details}}


@dataclass(frozen=True)
class FixtureRecord:
    image_id: str
    relative_path: str
    absolute_path: str
    nbytes: int
    sha256: str
    accepted_sha256: str


@dataclass(frozen=True)
class PreprocessedTensor:
    payload: bytes
    shape: tuple[int, int, int, int]
    dtype: str
    metadata: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_fixture(source_root: Path) -> dict[str, Any]:
    records: list[FixtureRecord] = []
    for image_id, relative in zip(FIXTURE_IDS, FIXTURE_RELATIVE_PATHS):
        path = source_root / Path(relative)
        if not path.is_file():
            raise FixtureError("FIXTURE_MISSING", "required CCTSDB train image is missing", {"image_id": image_id, "path": str(path)})
        actual = sha256_file(path)
        expected = ACCEPTED_IMAGE_SHA256[image_id]
        if actual != expected:
            raise FixtureError("FIXTURE_HASH_MISMATCH", "image bytes differ from accepted canonical binding", {"image_id": image_id, "expected": expected, "observed": actual})
        records.append(FixtureRecord(image_id, relative, str(path.resolve()), path.stat().st_size, actual, expected))
    sequence = [{"image_id": record.image_id, "relative_path": record.relative_path, "sha256": record.sha256} for record in records]
    return {
        "schema_version": "e2l1-cctsdb-fixture-binding-v1",
        "status": "verified_read_only",
        "split": "CCTSDB2021/train",
        "official_test_used": False,
        "negative_test_used": False,
        "canonical_order": list(FIXTURE_IDS),
        "records": [record.__dict__ for record in records],
        "sequence_sha256": hashlib.sha256(json.dumps(sequence, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "accepted_binding_manifest": CANONICAL_BINDING_MANIFEST,
        "copy_policy": "bind actual source bytes by path/hash; do not copy image/model bytes into Git",
        "derived_input": {"status": "not_materialized_by_manifest", "same_bytes_required_for_source_and_target": True, "preprocess_helper": "scripts/edge_readiness/e2_source_fixture.py"},
    }


def preprocess_decoded_bgr(
    pixels_bgr: bytes,
    width: int,
    height: int,
    resize_rgb: Callable[[bytes, int, int, int, int], bytes],
    *,
    image_size: int = 640,
    pad_value: int = 114,
) -> PreprocessedTensor:
    """Convert decoded BGR bytes to deterministic NCHW float32 bytes.

    ``resize_rgb`` is the source-approved resize implementation supplied by
    the later source environment. It receives RGB bytes and must return RGB
    bytes; keeping it injected prevents a silent OpenCV/Pillow mismatch.
    """
    if width <= 0 or height <= 0 or len(pixels_bgr) != width * height * 3:
        raise FixtureError("DECODED_IMAGE_INVALID", "decoded BGR dimensions and bytes are inconsistent")
    if not 0 <= pad_value <= 255:
        raise FixtureError("INVALID_PAD_VALUE", "pad value must be an 8-bit scalar")
    rgb = bytes(channel for pixel in range(width * height) for channel in (pixels_bgr[pixel * 3 + 2], pixels_bgr[pixel * 3 + 1], pixels_bgr[pixel * 3]))
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = resize_rgb(rgb, width, height, resized_width, resized_height)
    if len(resized) != resized_width * resized_height * 3:
        raise FixtureError("RESIZE_OUTPUT_INVALID", "injected resize did not return expected RGB bytes")
    canvas = bytearray([pad_value] * (image_size * image_size * 3))
    pad_x = (image_size - resized_width) // 2
    pad_y = (image_size - resized_height) // 2
    for y in range(resized_height):
        source_start = y * resized_width * 3
        target_start = ((pad_y + y) * image_size + pad_x) * 3
        canvas[target_start:target_start + resized_width * 3] = resized[source_start:source_start + resized_width * 3]
    plane = image_size * image_size
    floats = bytearray()
    for channel in range(3):
        for index in range(plane):
            floats.extend(struct.pack("<f", canvas[index * 3 + channel] / 255.0))
    return PreprocessedTensor(bytes(floats), (1, 3, image_size, image_size), "float32", {"source_color": "BGR", "model_color": "RGB", "layout": "NCHW", "normalization": "pixel/255.0", "letterbox_pad_value": pad_value, "interpolation": "injected_source_resize", "disk_decode_in_timing": False})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the three E2 CCTSDB train fixture bytes without copying them")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    try:
        manifest = verify_fixture(args.source_root)
    except FixtureError as exc:
        failure = {"schema_version": "e2l1-cctsdb-fixture-binding-v1", "status": "failed", "error": exc.as_dict()}
        (args.out_dir / "failure.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        return 2
    (args.out_dir / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
