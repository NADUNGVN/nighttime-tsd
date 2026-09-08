#!/usr/bin/env python3
"""Audit, but never mix, the official CCTSDB2021 negative test images."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate(raw: Path) -> tuple[Path, bool]:
    folders = [path for path in raw.iterdir() if path.is_dir() and "negative" in path.name.lower()]
    archives = [path for path in raw.iterdir() if path.is_file() and path.suffix.lower() == ".zip" and "negative" in path.name.lower()]
    if len(folders) == 1:
        return folders[0], False
    if len(archives) == 1:
        return archives[0], True
    raise FileNotFoundError(f"Could not uniquely locate official negative samples below {raw}; folders={folders}, archives={archives}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the official CCTSDB negative-image release without evaluating or copying it")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/calibration_method_v1/rtx8000/yolo11n/negative_set_audit.json"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {args.out}")
    source, is_archive = locate(args.raw)
    context = tempfile.TemporaryDirectory(prefix="cctsdb_negative_") if is_archive else None
    try:
        root = Path(context.name) if context is not None else source
        if is_archive:
            with zipfile.ZipFile(source) as archive:
                archive.extractall(root)
        images = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        payload = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source.resolve()),
            "source_sha256": sha256(source) if is_archive else None,
            "negative_images": len(images),
            "expected_official_negative_images": 500,
            "status": "available" if len(images) == 500 else "unexpected_count",
            "sample_relative_paths": [str(path.relative_to(root).as_posix()) for path in images[:20]],
            "policy": "Negative images are a separate fixed-threshold false-positive stress test and are never included in the 1,500-image positive mAP benchmark or used for threshold selection.",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        if context is not None:
            context.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
