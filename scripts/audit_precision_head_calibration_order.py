#!/usr/bin/env python3
"""CPU-only, full-image audit of the accepted U42/U43/U44 calibration order.

No model forward, TensorRT import, calibration cache, engine build, or CUDA
allocation is performed. If the accepted server materialization is not
available, verified local source bytes are copied into an isolated temporary
materialization and that limitation is recorded in the receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from precision_head_calibration_order import (
    CalibrationOrderError,
    build_manifest_ordered_calibration_dataloader,
    canonical_json_sha256,
    verify_pinned_source_contract,
)


SELECTIONS = ("U42", "U43", "U44")
EXPECTED_COUNT = 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CalibrationOrderError(f"expected JSON object: {path}")
    return value


def verify_source_rows(selection: dict[str, Any], dataset_root: Path) -> tuple[list[str], list[dict[str, Any]]]:
    ids = list(selection.get("manifest_contract", {}).get("image_ids", []))
    materialized_rows = list(selection.get("materialization", {}).get("image_bytes", []))
    source_rows = list(selection.get("selection_audit", {}).get("source_bytes", []))
    if selection.get("status") != "verified" or len(ids) != EXPECTED_COUNT or len(set(ids)) != EXPECTED_COUNT:
        raise CalibrationOrderError(f"accepted selection is not a unique 1024-image verified manifest: {selection.get('id')}")
    if len(materialized_rows) != EXPECTED_COUNT or [row.get("image") for row in materialized_rows] != ids:
        raise CalibrationOrderError(f"materialization order/count differs from accepted manifest: {selection.get('id')}")
    source_by_image = {row.get("image"): row for row in source_rows}
    if len(source_by_image) != EXPECTED_COUNT or set(source_by_image) != set(ids):
        raise CalibrationOrderError(f"source inventory membership differs from accepted manifest: {selection.get('id')}")
    receipts = []
    for image_id, materialized in zip(ids, materialized_rows):
        source = source_by_image[image_id]
        image_path = dataset_root / Path(image_id)
        label_path = dataset_root / Path(source["label"])
        if not image_path.is_file():
            raise FileNotFoundError(f"accepted calibration image is missing: {image_path}")
        image_observed_hash = sha256_file(image_path)
        if image_observed_hash != source.get("image_sha256") or image_path.stat().st_size != source.get("image_bytes"):
            raise CalibrationOrderError(f"accepted source image binding differs for {image_id}: sha256={image_observed_hash}, bytes={image_path.stat().st_size}")
        if not label_path.is_file():
            raise FileNotFoundError(f"accepted calibration label is missing: {label_path}")
        label_bytes = label_path.read_bytes()
        label_observed_hash = hashlib.sha256(label_bytes).hexdigest()
        label_eol_normalized_hash = hashlib.sha256(label_bytes.replace(b"\r\n", b"\n")).hexdigest()
        label_exact = label_observed_hash == source.get("label_sha256") and len(label_bytes) == source.get("label_bytes")
        label_eol_equivalent = label_eol_normalized_hash == source.get("label_sha256")
        if not label_exact and not label_eol_equivalent:
            raise CalibrationOrderError(f"accepted source label content differs beyond CRLF/LF normalization for {image_id}: sha256={label_observed_hash}, normalized_sha256={label_eol_normalized_hash}")
        image_hash = source["image_sha256"]
        if materialized.get("source_sha256") != image_hash or materialized.get("materialized_sha256") != image_hash or materialized.get("bytes") != source["image_bytes"]:
            raise CalibrationOrderError(f"accepted source/materialization image binding differs for {image_id}")
        receipts.append({
            "image_id": image_id,
            "source_image_sha256": image_hash,
            "source_image_bytes": source["image_bytes"],
            "source_label_sha256": source["label_sha256"],
            "source_label_bytes": source["label_bytes"],
            "local_label_sha256": label_observed_hash,
            "local_label_bytes": len(label_bytes),
            "local_label_exact_hash_match": label_exact,
            "local_label_crlf_lf_normalized_hash_match": label_eol_equivalent,
            "materialized_image_sha256": materialized["materialized_sha256"],
        })
    return ids, receipts


def make_scratch_materialization(
    selection: dict[str, Any], dataset_root: Path, ids: list[str], scratch: Path
) -> tuple[Path, Path, str]:
    materialized = selection["materialization"]
    accepted_yaml = Path(materialized.get("yaml", ""))
    accepted_root = Path(materialized.get("resolved_directory", ""))
    if accepted_yaml.is_file() and accepted_root.is_dir():
        expected_yaml_sha = materialized.get("yaml_sha256")
        if sha256_file(accepted_yaml) != expected_yaml_sha:
            raise CalibrationOrderError(f"accepted materialization YAML hash differs: {accepted_yaml}")
        root = accepted_root.resolve()
        for image_id, row in zip(ids, materialized["image_bytes"]):
            image = root / "images" / Path(image_id).name
            if not image.is_file() or image.stat().st_size != row["bytes"] or sha256_file(image) != row["materialized_sha256"]:
                raise CalibrationOrderError(f"accepted materialized image differs: {image_id}")
        return accepted_yaml.resolve(), root, "accepted_server_materialization_verified_read_only"

    root = scratch / selection["id"]
    image_dir, label_dir = root / "images", root / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    source_by_image = {row["image"]: row for row in selection["selection_audit"]["source_bytes"]}
    for image_id in ids:
        row = source_by_image[image_id]
        shutil.copyfile(dataset_root / Path(image_id), image_dir / Path(image_id).name)
        shutil.copyfile(dataset_root / Path(row["label"]), label_dir / Path(row["label"]).name)
    data_yaml = root / "calibration.yaml"
    data_yaml.write_text(
        "path: " + json.dumps(str(root.resolve())) + "\n"
        "train: images\nval: images\n"
        "names:\n  0: prohibitory\n  1: mandatory\n  2: warning\nnc: 3\n",
        encoding="utf-8",
        newline="\n",
    )
    return data_yaml, root, "isolated_temporary_materialization_from_hash_verified_local_source"


def audit_selection(
    selection: dict[str, Any], dataset_root: Path, torch: Any, Exporter: Any,
    exporter_module: Any, source_fingerprints: dict[str, str], scratch_parent: Path,
) -> dict[str, Any]:
    ids, source_receipts = verify_source_rows(selection, dataset_root)
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{selection['id']}_", dir=scratch_parent) as temporary:
        scratch = Path(temporary)
        yaml_path, materialized_root, materialization_basis = make_scratch_materialization(
            selection, dataset_root, ids, scratch
        )
        exporter = Exporter(overrides={
            "format": "engine", "data": str(yaml_path), "imgsz": 640,
            "batch": 1, "fraction": 1.0, "split": "val", "rect": False,
            "device": "cpu",
        })
        exporter.imgsz = (640, 640)
        exporter.model = SimpleNamespace(task="detect")
        expected_paths = [materialized_root / "images" / Path(image_id).name for image_id in ids]
        loader, loader_audit = build_manifest_ordered_calibration_dataloader(
            exporter,
            ordered_image_ids=ids,
            expected_image_paths=expected_paths,
            check_det_dataset=exporter_module.check_det_dataset,
            build_yolo_dataset=exporter_module.build_yolo_dataset,
            build_dataloader=exporter_module.build_dataloader,
            expected_count=EXPECTED_COUNT,
        )
        observed_rows: list[dict[str, Any]] = []
        tensor_sequence = hashlib.sha256()
        count = 0
        loader_iterator = iter(loader)
        for batch in loader_iterator:
            index = count
            names = batch.get("im_file") or batch.get("im_files") or []
            if isinstance(names, (str, Path)):
                names = [str(names)]
            if len(names) != 1:
                raise CalibrationOrderError(f"batch is not batch=1 at index {index}: {names!r}")
            observed_path = Path(str(names[0])).resolve()
            expected_path = expected_paths[index].resolve() if index < len(expected_paths) else None
            if index >= EXPECTED_COUNT or observed_path != expected_path:
                raise CalibrationOrderError(f"manifest order mismatch at index {index}: expected={expected_path}, observed={observed_path}")
            image_tensor = batch.get("img")
            if tuple(image_tensor.shape) != (1, 3, 640, 640):
                raise CalibrationOrderError(f"preprocessed shape differs at {ids[index]}: {tuple(image_tensor.shape)}")
            if image_tensor.dtype != torch.uint8:
                raise CalibrationOrderError(f"preprocessed dtype differs at {ids[index]}: {image_tensor.dtype}")
            normalized = (image_tensor.to(dtype=torch.float32) / 255.0).contiguous()
            tensor_bytes = normalized.numpy().tobytes()
            tensor_sequence.update(tensor_bytes)
            source = source_receipts[index]
            observed_rows.append({
                **source,
                "observed_index": index,
                "observed_path": str(observed_path),
                "input_shape": list(image_tensor.shape),
                "input_dtype": str(image_tensor.dtype),
                "normalized_dtype": str(normalized.dtype),
                "normalized_tensor_sha256": hashlib.sha256(tensor_bytes).hexdigest(),
            })
            count += 1
        if count != EXPECTED_COUNT:
            raise CalibrationOrderError(f"loader ended early: expected={EXPECTED_COUNT}, observed={count}")
        receipt = next(loader_iterator, None)
        if receipt is not None:
            raise CalibrationOrderError("calibration loader produced items beyond the locked 1024-image selection")
        observed_ids = [row["image_id"] for row in observed_rows]
        if observed_ids != ids:
            raise CalibrationOrderError("observed calibration IDs differ from the canonical manifest sequence")
        return {
            "schema_version": 1,
            "selection": selection["id"],
            "status": "pass",
            "materialization_basis": materialization_basis,
            "accepted_manifest": selection["manifest"],
            "accepted_manifest_sha256": selection["canonical_sha256"],
            "accepted_yaml_sha256": sha256_file(yaml_path),
            "source_inventory_count": len(source_receipts),
            "yielded_count": count,
            "validated_count": count,
            "tensors_delivered_to_test_consumer": count,
            "end_of_stream_verified": True,
            "canonical_ids_sha256": canonical_json_sha256(ids),
            "observed_ids_sha256": canonical_json_sha256(observed_ids),
            "normalized_tensor_sequence_sha256": tensor_sequence.hexdigest(),
            "loader": loader_audit,
            "ultralytics_source_sha256": source_fingerprints,
            "items": observed_rows,
            "label_byte_limit": "Some Windows checkout labels use CRLF; all label contents are verified equivalent after CRLF-to-LF normalization. Calibration consumes only the image tensor; label bytes are not passed to the calibrator.",
            "execution_limits": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_cuda_allocations": 0,
                "model_forward_calls": 0,
                "tensorrt_imported": False,
                "engine_builds": 0,
                "calibration_cache_reads": 0,
                "calibration_cache_writes": 0,
            },
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--readiness-root", type=Path, default=repo / "results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2")
    parser.add_argument("--dataset-root", type=Path, default=repo / "data/processed/cctsdb2021_clean")
    parser.add_argument("--out-dir", type=Path, default=repo / "results/measurement_audit_v1/precision_head_calibration_order_audit_v1")
    args = parser.parse_args(argv)
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"fresh audit output root required: {out}")
    out.mkdir(parents=True)
    manifest_path = args.readiness_root.resolve() / "calibration_readiness.json"
    readiness_path = args.readiness_root.resolve() / "readiness_manifest.json"
    manifest = load_json(manifest_path)
    readiness = load_json(readiness_path)
    if readiness.get("status") not in {"ready_for_server_prepare_review", "ready_for_server_run"}:
        raise CalibrationOrderError(f"readiness status is not accepted for audit: {readiness.get('status')}")
    selection_by_id = {row.get("id"): row for row in manifest.get("selections", [])}
    if set(selection_by_id) != set(SELECTIONS):
        atomic_json(out / "failure.json", {"status": "failed", "error": "readiness selection IDs differ", "observed": sorted(selection_by_id)})
        return 1
    dataset_root = args.dataset_root.resolve()
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise CalibrationOrderError("CPU audit requires CUDA_VISIBLE_DEVICES=-1 before importing torch")
        import numpy as np
        import torch
        import ultralytics
        from ultralytics.engine import exporter as exporter_module
        from ultralytics.engine.exporter import Exporter

        if "tensorrt" in __import__("sys").modules:
            raise CalibrationOrderError("TensorRT was imported during CPU-only audit startup")
        if torch.cuda.device_count() != 0:
            raise CalibrationOrderError("CUDA device is visible during CPU-only order audit")
        torch.set_num_threads(2)
        fingerprints = verify_pinned_source_contract(
            ultralytics.__version__, Exporter,
            exporter_module.build_yolo_dataset,
            exporter_module.check_det_dataset,
            exporter_module.build_dataloader,
        )
        python_packages = {
            "python": __import__("platform").python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "numpy": np.__version__,
        }
        manifest_value = {
            "schema_version": 1,
            "study": "precision_head_calibration_order_cpu_audit_v1",
            "status": "running",
            "readiness_manifest_sha256": sha256_file(readiness_path),
            "calibration_readiness_sha256": sha256_file(manifest_path),
            "dataset_root": str(dataset_root),
            "runtime": python_packages,
            "ultralytics_source_sha256": fingerprints,
            "execution_limits": {"cuda_visible_devices": "-1", "torch_cuda_allocations": 0, "model_forward_calls": 0, "tensorrt_imported": False, "engine_builds": 0, "cache_reads": 0, "cache_writes": 0},
            "selections": [],
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(out / "audit_manifest.json", manifest_value)
        scratch_parent = Path(tempfile.gettempdir())
        for selection_id in SELECTIONS:
            receipt = audit_selection(
                selection_by_id[selection_id], dataset_root, torch, Exporter,
                exporter_module, fingerprints, scratch_parent,
            )
            atomic_json(out / f"selection_{selection_id}.json", receipt)
            manifest_value["selections"].append({
                "id": selection_id,
                "status": receipt["status"],
                "count": receipt["yielded_count"],
                "canonical_ids_sha256": receipt["canonical_ids_sha256"],
                "observed_ids_sha256": receipt["observed_ids_sha256"],
                "normalized_tensor_sequence_sha256": receipt["normalized_tensor_sequence_sha256"],
                "receipt": f"selection_{selection_id}.json",
                "receipt_sha256": sha256_file(out / f"selection_{selection_id}.json"),
            })
            manifest_value["completed_selections"] = len(manifest_value["selections"])
            atomic_json(out / "audit_manifest.json", manifest_value)
        manifest_value["status"] = "completed_review_required"
        manifest_value["completed_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(out / "audit_manifest.json", manifest_value)
        lines = [
            "# Precision-head calibration order CPU audit",
            "",
            "No TensorRT build, model forward, calibration cache operation, or CUDA allocation was run.",
            "The audit verifies the actual pinned Ultralytics dataset/dataloader path over every accepted image in U42/U43/U44.",
            "Local label files have CRLF/LF byte differences from the Linux source inventory; all normalize to the accepted LF hashes, and labels are not passed as calibration tensors.",
            "",
            "| Selection | Expected | Yielded | Validated | IDs match | Materialization |",
            "|---|---:|---:|---:|---|---|",
        ]
        for item in manifest_value["selections"]:
            receipt = load_json(out / item["receipt"])
            lines.append(f"| {item['id']} | 1024 | {receipt['yielded_count']} | {receipt['validated_count']} | {receipt['canonical_ids_sha256'] == receipt['observed_ids_sha256']} | {receipt['materialization_basis']} |")
        lines.extend(["", f"Ultralytics source fingerprints: `{json.dumps(fingerprints, sort_keys=True)}`", ""])
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
        print(f"DONE: {out / 'audit_manifest.json'}")
        return 0
    except BaseException as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_selections": len(manifest_value.get("selections", [])) if "manifest_value" in locals() else 0,
            "traceback": traceback.format_exc(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "no_model_forward": True,
            "no_tensorrt_build": True,
        }
        atomic_json(out / "failure.json", failure)
        if "manifest_value" in locals():
            manifest_value["status"] = "failed"
            manifest_value["failure"] = "failure.json"
            atomic_json(out / "audit_manifest.json", manifest_value)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
