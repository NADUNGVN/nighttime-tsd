"""Manifest-indexed INT8 calibration input construction.

This module deliberately mirrors the pinned Ultralytics detection-export
dataset setup, but owns the final index order and passes ``shuffle=False``
explicitly. It never changes the installed Ultralytics module or global RNG.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Sequence


PINNED_ULTRALYTICS_VERSION = "8.4.102"
PINNED_SOURCE_SHA256 = {
    "exporter_calibration_method": "c3083ef89201276d5ea73e217846a0ddcaeed92b4909f75a2545c4ef2016b44c",
    "build_yolo_dataset": "cc30939a9a049a201481a43b84f1b41eea32f0e68bfe57aba3bbada5c9433e49",
    "check_det_dataset": "180a0628c120d35a930fb2ba7681c6bc646658ab7555e66357aa611c53d04ee2",
    "build_dataloader": "3526a3fbca169de6bdfde5e6128a5be128d16135725458475728a3d145a65164",
}


class CalibrationOrderError(RuntimeError):
    """The installed dataset cannot satisfy the locked calibration sequence."""


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def path_key(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def source_fingerprints(
    exporter_class: type,
    build_yolo_dataset: Callable[..., Any],
    check_det_dataset: Callable[..., Any],
    build_dataloader: Callable[..., Any],
) -> dict[str, str]:
    objects = {
        "exporter_calibration_method": exporter_class.get_int8_calibration_dataloader,
        "build_yolo_dataset": build_yolo_dataset,
        "check_det_dataset": check_det_dataset,
        "build_dataloader": build_dataloader,
    }
    return {
        name: hashlib.sha256(inspect.getsource(obj).encode("utf-8")).hexdigest()
        for name, obj in objects.items()
    }


def verify_pinned_source_contract(
    ultralytics_version: str,
    exporter_class: type,
    build_yolo_dataset: Callable[..., Any],
    check_det_dataset: Callable[..., Any],
    build_dataloader: Callable[..., Any],
) -> dict[str, str]:
    if ultralytics_version != PINNED_ULTRALYTICS_VERSION:
        raise CalibrationOrderError(
            f"Ultralytics version mismatch: expected {PINNED_ULTRALYTICS_VERSION}, observed {ultralytics_version}"
        )
    observed = source_fingerprints(
        exporter_class, build_yolo_dataset, check_det_dataset, build_dataloader
    )
    if observed != PINNED_SOURCE_SHA256:
        raise CalibrationOrderError(
            f"pinned Ultralytics producer source changed: expected={PINNED_SOURCE_SHA256}, observed={observed}"
        )
    return observed


class ManifestIndexedDataset:
    """Expose an existing Ultralytics dataset through canonical manifest indices."""

    def __init__(self, dataset: Any, indices: Sequence[int], ordered_paths: Sequence[Path]):
        self.dataset = dataset
        self.indices = tuple(indices)
        self.im_files = [str(path) for path in ordered_paths]
        self.collate_fn = dataset.collate_fn

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]


def build_manifest_ordered_calibration_dataloader(
    exporter: Any,
    *,
    ordered_image_ids: Sequence[str],
    expected_image_paths: Sequence[str | Path],
    check_det_dataset: Callable[..., Any],
    build_yolo_dataset: Callable[..., Any],
    build_dataloader: Callable[..., Any],
    expected_count: int = 1024,
) -> tuple[Any, dict[str, Any]]:
    """Build the pinned detection dataset and explicitly index it by manifest.

    ``expected_image_paths`` are the accepted, materialized files in the same
    order as ``ordered_image_ids``. The underlying YOLO dataset may discover
    files in any order; duplicates, omissions and extras are rejected before
    the dataloader is constructed.
    """
    ids = list(ordered_image_ids)
    expected_paths = [Path(path).resolve() for path in expected_image_paths]
    if len(ids) != expected_count or len(expected_paths) != expected_count:
        raise CalibrationOrderError(
            f"manifest calibration count must be {expected_count}: ids={len(ids)} paths={len(expected_paths)}"
        )
    if len(set(ids)) != expected_count:
        raise CalibrationOrderError("manifest calibration IDs contain duplicates")
    expected_keys = [path_key(path) for path in expected_paths]
    if len(set(expected_keys)) != expected_count:
        raise CalibrationOrderError("expected materialized calibration paths contain duplicates")
    missing_files = [str(path) for path in expected_paths if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"manifest calibration images are missing: {missing_files[:3]}")

    args = exporter.args
    if int(args.batch) != 1 or args.split != "val" or float(args.fraction) != 1.0 or bool(getattr(args, "rect", False)):
        raise CalibrationOrderError(
            f"calibration preprocessing contract changed: batch={args.batch}, split={args.split}, fraction={args.fraction}, rect={getattr(args, 'rect', None)}"
        )
    cfg = deepcopy(args)
    cfg.imgsz = max(exporter.imgsz)
    data = check_det_dataset(args.data, split=args.split)
    dataset = build_yolo_dataset(
        cfg,
        data[args.split or "val"],
        args.batch,
        data,
        mode="val",
        fraction=args.fraction,
    )
    transform_ops = getattr(getattr(dataset, "transforms", None), "transforms", [])
    if transform_ops and hasattr(transform_ops[0], "new_shape"):
        transform_ops[0].new_shape = exporter.imgsz
    if bool(getattr(dataset, "augment", False)) or bool(getattr(dataset, "rect", False)):
        raise CalibrationOrderError(
            f"calibration dataset preprocessing flags changed: augment={getattr(dataset, 'augment', None)}, rect={getattr(dataset, 'rect', None)}"
        )

    discovered = [Path(item).resolve() for item in dataset.im_files]
    discovered_keys = [path_key(item) for item in discovered]
    if len(discovered_keys) != len(set(discovered_keys)):
        raise CalibrationOrderError("Ultralytics dataset discovered duplicate image paths")
    if len(discovered_keys) != expected_count:
        raise CalibrationOrderError(
            f"Ultralytics dataset count differs from locked selection: expected={expected_count} observed={len(discovered_keys)}"
        )
    discovered_set = set(discovered_keys)
    expected_set = set(expected_keys)
    if discovered_set != expected_set:
        missing = [str(path) for path, key in zip(expected_paths, expected_keys) if key not in discovered_set]
        extra = [str(path) for path, key in zip(discovered, discovered_keys) if key not in expected_set]
        raise CalibrationOrderError(
            f"Ultralytics dataset membership differs from manifest: missing={missing[:3]} extra={extra[:3]}"
        )
    dataset_index = {key: index for index, key in enumerate(discovered_keys)}
    ordered_indices = [dataset_index[key] for key in expected_keys]
    view = ManifestIndexedDataset(dataset, ordered_indices, expected_paths)
    loader = build_dataloader(
        view,
        batch=1,
        workers=0,
        shuffle=False,
        drop_last=True,
    )
    audit = {
        "status": "manifest_ordered_loader_built",
        "expected_count": expected_count,
        "discovered_count": len(discovered),
        "yield_order_ids_sha256": canonical_json_sha256(ids),
        "yield_order_paths_sha256": canonical_json_sha256([str(path) for path in expected_paths]),
        "discovered_path_set_sha256": canonical_json_sha256(sorted(discovered_keys)),
        "manifest_index_sha256": canonical_json_sha256(ordered_indices),
        "shuffle": False,
        "batch": 1,
        "workers": 0,
        "drop_last": True,
        "augment": bool(getattr(dataset, "augment", False)),
        "dataset_rect": bool(getattr(dataset, "rect", False)),
    }
    return loader, audit
