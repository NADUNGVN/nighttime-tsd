#!/usr/bin/env python3
"""Verify frozen PyTorch FP32 outputs against the accepted ONNX graphs on CPU.

This is a bounded, non-scored diagnostic.  The parent performs all input and
hash checks without importing a model runtime, then dispatches one model-only
CPU child at a time.  Children use the pinned Ultralytics preprocessing
functions, a PyTorch CPU reference, and an explicitly selected ONNX Runtime
CPU provider.  No exporter, TensorRT, calibration loader, GPU or NMS path is
available in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import prepare_precision_head_confirmation as readiness
import prepare_precision_head_confirmation_graph as graph


STUDY = "precision_head_confirmation_numeric_v1"
MODEL_CHOICES = ("yolov8n", "yolo26n")
SELECTION_IDS = ("U42", "U43", "U44")
FIXTURE_SELECTION = "U42"
FIXTURE_COUNT = 8
READINESS_ROOT_NAME = "server_precision_head_confirmation_readiness_v2"
GRAPH_AUDIT_ROOT_NAME = "precision_head_confirmation_graph_audit_v4"
DEFAULT_READINESS_ROOT = Path("results/measurement_audit_v1") / READINESS_ROOT_NAME
DEFAULT_GRAPH_AUDIT_ROOT = Path("results/measurement_audit_v1") / GRAPH_AUDIT_ROOT_NAME
DEFAULT_SOURCE_ROOT = Path("results/measurement_audit_v1") / "precision_head_confirmation_graph_prep_v2"
DEFAULT_OUTPUT = Path("results/measurement_audit_v1") / STUDY
GRAPH_AUDIT_COMMIT = "6780b813c5f1cb2d915b72832eedd79525eaecbd"
GRAPH_AUDIT_FILES = {
    "audit_plan.json": "cfe5333e5caf816511060eb81b3575d8390e8adbac3acb19278120a87fa5c65a",
    "graph_audit_manifest.json": "c5f538aa5cb390d68e272cec9fad22a455ac8397fdb351f5c93c1e79e4382d65",
    "models/yolov8n/graph_audit.json": "5cf9d9b7c39c8d7607352492ee2e89276915104bee93ce661876864a5fb97ccb",
    "models/yolo26n/graph_audit.json": "2c8d97c10d12323862831a2e58c6aa7a3c79e2fe9bc424560db206fc707b14cb",
    "report.md": "71489f4c12607321c61cd18e4363a7d037cd318aaaa3823a7271e3ede5c33a2e",
}
EXPECTED_ONNX_SHA256 = {
    "yolov8n": "e22d53bbeb333f44783535d911d5e318ebb7d500e8fcb3d1cbb1284f5248d603",
    "yolo26n": "1b2467ccd62bd1e53f3bde3e3f22e1b42129711d3e368a4b4666d025099ce5cc",
}
EXPECTED_OUTPUT_SHAPES = {"yolov8n": [1, 7, 8400], "yolo26n": [1, 300, 6]}
EXPECTED_CHECKPOINTS = {
    "yolov8n": {
        "path": "results/yolov8n_cctsdb_clean_s42_v1/weights/best.pt",
        "sha256": "b2b7a1c77a19499ded33c9cc11c621757077aa871f4e7f7a1fcdbf94f53b383b",
        "bytes": 6260963,
    },
    "yolo26n": {
        "path": "results/yolo26n_cctsdb_clean_s42_v1/weights/best.pt",
        "sha256": "2bb49f85f581469fc7942652d5fda4da44278d57fa8363e8f7295daa49f0d01e",
        "bytes": 5399038,
    },
}
LOCKED_TOLERANCES = {
    "float32": {"rtol": 1e-4, "atol": 1e-5},
    "class_id": {"rule": "exact integer equality, inclusive range 0..2"},
    "ranking": {"rule": "fixed native row index; no rematching, sorting, NMS or tie breaking"},
}
CPU_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "YOLO_AUTOINSTALL": "0",
    "ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS": "1",
    "PIP_NO_INDEX": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
}
GRAPH_AUDIT_MODEL_FILES = {model: f"models/{model}/graph_audit.json" for model in MODEL_CHOICES}
DATASET_ROOT = Path("data/processed/cctsdb2021_clean")
CALIBRATION_SELECTION_ROOT = DATASET_ROOT / "calibration"
ACCEPTED_EXPORT_ARGUMENTS = {
    "format": "onnx",
    "imgsz": 640,
    "batch": 1,
    "opset": 17,
    "simplify": True,
    "dynamic": False,
    "half": False,
    "device": "cpu",
    "task": "detect",
}
CALIBRATION_TRACE_ARGUMENTS = {
    "dataset_mode": "val",
    "rect": False,
    "fraction": 1.0,
    "workers": 0,
    "batch": 1,
    "imgsz": 640,
    "YOLODataset.load_image.rect_mode": True,
    "YOLODataset.load_image.resize_short": False,
    "LetterBox.auto": False,
    "LetterBox.scale_fill": False,
    "LetterBox.scaleup": False,
    "LetterBox.center": True,
    "LetterBox.padding_value": 114,
}


class NumericUnresolved(RuntimeError):
    """Raised when prerequisites do not allow a meaningful numeric comparison."""


def canonical_json(value: Any) -> bytes:
    return readiness.canonical_json(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return readiness.sha256_file(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_no_overwrite(path: Path, value: Any) -> None:
    readiness.write_json_no_overwrite(path, value)


def write_text_no_overwrite(path: Path, value: str) -> None:
    readiness.write_text_no_overwrite(path, value)


def repo_path(repo: Path, value: Path) -> Path:
    resolved = (value if value.is_absolute() else repo / value).resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes repository: {value}") from exc
    return resolved


def file_evidence(repo: Path, path: Path) -> dict[str, Any]:
    path = path.resolve()
    record = {
        "path": path.relative_to(repo.resolve()).as_posix() if path.is_relative_to(repo.resolve()) else str(path),
        "exists": path.is_file(),
        "regular_file": path.is_file() and not path.is_symlink(),
    }
    if path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return record


def git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError("Cannot resolve current Git HEAD")
    return result.stdout.strip()


def set_cpu_environment() -> None:
    os.environ.update(CPU_ENVIRONMENT)


def selected_models(value: str) -> list[str]:
    if value not in (*MODEL_CHOICES, "all"):
        raise ValueError(f"Unsupported model selection: {value}")
    return list(MODEL_CHOICES) if value == "all" else [value]


def validate_graph_audit_artifact(repo: Path, root: Path, models: list[str]) -> dict[str, Any]:
    """Bind the exact accepted five-file graph-v4 artifact."""
    root = root.resolve()
    if root.name != GRAPH_AUDIT_ROOT_NAME or not root.is_dir():
        raise FileNotFoundError(f"Accepted graph audit root is missing or misnamed: {root}")
    actual = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    if actual != sorted(GRAPH_AUDIT_FILES):
        raise ValueError(f"Graph-v4 artifact file inventory differs: {actual}")
    records = {}
    for relative, expected in GRAPH_AUDIT_FILES.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Graph-v4 artifact is not a regular file: {relative}")
        canonical = readiness.git_blob(repo, GRAPH_AUDIT_COMMIT, repo / "results/measurement_audit_v1" / GRAPH_AUDIT_ROOT_NAME / relative)
        canonical_hash = sha256_bytes(canonical)
        working_bytes = path.read_bytes()
        if canonical_hash != expected or (working_bytes != canonical and working_bytes.replace(b"\r\n", b"\n") != canonical):
            raise ValueError(f"Graph-v4 file does not equal accepted Git content: {relative}")
        records[relative] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_bytes(working_bytes),
            "accepted_git_blob_sha256": canonical_hash,
            "content_match_mode": "exact" if working_bytes == canonical else "LF_normalized_for_core_autocrlf",
        }
    manifest = read_json(root / "graph_audit_manifest.json")
    if manifest.get("status") != "audit_only_completed":
        raise ValueError("Graph-v4 manifest is not completed audit-only evidence")
    if manifest.get("audit_flags", {}).get("export_performed") is not False:
        raise ValueError("Graph-v4 manifest incorrectly records export")
    rows = {row.get("model"): row for row in manifest.get("models", [])}
    # The accepted graph artifact is a two-model, five-file contract.  A
    # single-model numeric selection may select from it, but may not turn the
    # accepted artifact into a model-specific substitute.
    if set(rows) != set(MODEL_CHOICES):
        raise ValueError(f"Graph-v4 model set differs from accepted full set: {sorted(rows)}")
    unknown = set(models) - set(MODEL_CHOICES)
    if unknown:
        raise ValueError(f"Unsupported graph model selection: {sorted(unknown)}")
    model_docs = {}
    for model in models:
        row = rows[model]
        doc = read_json(root / GRAPH_AUDIT_MODEL_FILES[model])
        if row.get("status") != "audit_only_completed" or row.get("mapping_status") != "verified":
            raise NumericUnresolved(f"Graph-v4 {model} is not verified structural evidence")
        if doc.get("status") != "audit_only_completed" or doc.get("mapping_status") != "verified":
            raise NumericUnresolved(f"Graph-v4 model report is not verified for {model}")
        flags = doc.get("audit_flags", {})
        if any(flags.get(key) is not False for key in ("export_performed", "build_performed", "capture_performed", "scored_run_authorized")):
            raise ValueError(f"Graph-v4 flags are unsafe for {model}")
        observed_sha = (row.get("provenance", {}).get("onnx", {}).get("before", {}) or {}).get("sha256")
        if observed_sha != EXPECTED_ONNX_SHA256[model] or doc.get("provenance", {}).get("onnx", {}).get("before", {}).get("sha256") != observed_sha:
            raise ValueError(f"Graph-v4 ONNX hash binding differs for {model}")
        schema_outputs = doc.get("onnx_schema", {}).get("outputs", [])
        shape_value = schema_outputs[0].get("shape") if schema_outputs else None
        if not schema_outputs or schema_outputs[0].get("name") != "output0" or shape_value != EXPECTED_OUTPUT_SHAPES[model]:
            raise ValueError(f"Graph-v4 output schema differs for {model}: {shape_value}")
        model_docs[model] = doc
    return {"root": str(root), "commit": GRAPH_AUDIT_COMMIT, "files": records, "manifest": manifest, "accepted_models": sorted(rows), "models": model_docs}


def accepted_model_contracts(accepted: dict[str, Any], models: list[str]) -> dict[str, dict[str, Any]]:
    rows = {row.get("label"): row for row in accepted["manifest"].get("model_contracts", [])}
    missing = [model for model in models if model not in rows]
    if missing:
        raise ValueError(f"Accepted readiness has no model contracts: {missing}")
    return {model: rows[model] for model in models}


def validate_checkpoint_contract(repo: Path, config: dict[str, Any], accepted: dict[str, Any], model: str) -> dict[str, Any]:
    config_rows = {row.get("label"): row for row in config.get("models", [])}
    config_row = config_rows.get(model)
    accepted_row = accepted_model_contracts(accepted, [model])[model]
    expected = EXPECTED_CHECKPOINTS[model]
    if not config_row or config_row.get("checkpoint") != expected["path"] or config_row.get("sha256") != expected["sha256"] or config_row.get("bytes") != expected["bytes"]:
        raise ValueError(f"Current config checkpoint contract differs for {model}")
    if accepted_row.get("checkpoint") != expected["path"] or accepted_row.get("expected_sha256") != expected["sha256"] or accepted_row.get("expected_bytes") != expected["bytes"]:
        raise ValueError(f"Accepted readiness checkpoint contract differs for {model}")
    path = repo_path(repo, Path(expected["path"]))
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Frozen checkpoint is missing or symlinked: {path}")
    observed = file_evidence(repo, path)
    if observed.get("sha256") != expected["sha256"] or observed.get("bytes") != expected["bytes"]:
        raise ValueError(f"Frozen checkpoint hash/size mismatch for {model}")
    return {"expected": expected, "observed": observed}


def _selection_rows(selection: dict[str, Any], count: int | None = None) -> list[dict[str, Any]]:
    contract = selection.get("manifest_contract", {})
    if contract.get("status") != "canonical_manifest_valid" or contract.get("train_only") is not True:
        raise ValueError(f"Selection {selection.get('id')} is not canonical train-only readiness")
    image_ids = contract.get("image_ids")
    source_bytes = selection.get("selection_audit", {}).get("source_bytes")
    materialized_bytes = selection.get("materialization", {}).get("image_bytes")
    if not isinstance(image_ids, list) or not isinstance(source_bytes, list) or not isinstance(materialized_bytes, list):
        raise ValueError(f"Selection {selection.get('id')} lacks ordered content bindings")
    if len(image_ids) != len(source_bytes) or len(image_ids) != len(materialized_bytes):
        raise ValueError(f"Selection {selection.get('id')} ordered content binding length mismatch")
    if [row.get("image") for row in source_bytes] != image_ids or [row.get("image") for row in materialized_bytes] != image_ids:
        raise ValueError(f"Selection {selection.get('id')} content binding order differs from manifest order")
    limit = len(image_ids) if count is None else count
    rows = []
    seen = set()
    for index, image in enumerate(image_ids[:limit]):
        parsed = PurePosixPath(image)
        if len(parsed.parts) != 3 or parsed.parts[:2] != ("train", "images"):
            raise ValueError(f"Non-train image in fixed fixture: {image}")
        image_id = parsed.stem
        if image_id in seen:
            raise ValueError(f"Duplicate image ID in fixed fixture: {image_id}")
        seen.add(image_id)
        source = source_bytes[index]
        materialized = materialized_bytes[index]
        if source.get("image") != image or materialized.get("image") != image:
            raise ValueError(f"Content binding row mismatch for {image}")
        if source.get("source_sha256") != materialized.get("materialized_sha256") or source.get("bytes") != materialized.get("bytes"):
            raise ValueError(f"Source/materialization accepted hashes differ for {image}")
        rows.append({
            "selection": selection["id"],
            "manifest_order": index,
            "image": image,
            "image_id": image_id,
            "expected_sha256": source.get("source_sha256"),
            "expected_bytes": source.get("bytes"),
            "materialized_expected_sha256": materialized.get("materialized_sha256"),
            "rule": "first 8 distinct train image IDs in canonical U42 manifest order" if selection["id"] == FIXTURE_SELECTION and count == FIXTURE_COUNT else "first image in canonical train-only selection order",
        })
    if count is not None and len(rows) != count:
        raise ValueError(f"Selection {selection.get('id')} has only {len(rows)} usable rows")
    return rows


def build_fixture_plan(accepted: dict[str, Any]) -> dict[str, Any]:
    selections = {row.get("id"): row for row in accepted["manifest"].get("calibration_readiness", {}).get("selections", [])}
    if set(selections) != set(SELECTION_IDS):
        raise ValueError("Accepted readiness does not contain exactly U42/U43/U44")
    fixture = _selection_rows(selections[FIXTURE_SELECTION], FIXTURE_COUNT)
    traces = [_selection_rows(selections[selection], 1)[0] for selection in SELECTION_IDS]
    return {
        "selection_rule": "first 8 distinct train image IDs in canonical U42 manifest order",
        "forward_selection": FIXTURE_SELECTION,
        "forward_image_count": len(fixture),
        "forward_images": fixture,
        "preprocess_trace_rule": "first image in each canonical U42/U43/U44 train-only manifest order",
        "preprocess_trace_images": traces,
        "no_labels_read": True,
        "no_test_images_or_labels_read": True,
    }


def _safe_train_image_parts(image: Any) -> tuple[PurePosixPath, tuple[str, ...]]:
    if not isinstance(image, str) or not image:
        raise ValueError(f"Invalid bound image path: {image!r}")
    parsed = PurePosixPath(image)
    parts = parsed.parts
    if parsed.is_absolute() or len(parts) != 3 or parts[:2] != ("train", "images") or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Bound image is not a safe canonical train image: {image!r}")
    return parsed, parts


def resolve_bound_source_image(repo: Path, row: dict[str, Any]) -> Path:
    """Resolve a readiness POSIX image ID against the canonical dataset root."""
    _parsed, parts = _safe_train_image_parts(row.get("image"))
    return repo_path(repo, DATASET_ROOT / Path(*parts))


def resolve_bound_materialized_image(repo: Path, row: dict[str, Any]) -> Path:
    """Resolve the separate materialized calibration copy without reusing source root."""
    parsed, _parts = _safe_train_image_parts(row.get("image"))
    selection = row.get("selection")
    if selection not in SELECTION_IDS:
        raise ValueError(f"Unknown calibration selection for bound image: {selection!r}")
    root = CALIBRATION_SELECTION_ROOT / f"uniform_s{selection[1:]}_n1024" / "images"
    return repo_path(repo, root / parsed.name)


def verify_bound_image(repo: Path, row: dict[str, Any]) -> dict[str, Any]:
    source = resolve_bound_source_image(repo, row)
    materialized = resolve_bound_materialized_image(repo, row)
    if source.is_symlink() or not source.is_file() or materialized.is_symlink() or not materialized.is_file():
        raise FileNotFoundError(f"Bound source/materialized image missing: {row['image']}")
    source_record = file_evidence(repo, source)
    materialized_record = file_evidence(repo, materialized)
    if source_record.get("sha256") != row["expected_sha256"] or source_record.get("bytes") != row["expected_bytes"]:
        raise ValueError(f"Current source image binding differs for {row['image']}")
    if materialized_record.get("sha256") != row["materialized_expected_sha256"] or materialized_record.get("bytes") != row["expected_bytes"]:
        raise ValueError(f"Current materialized image binding differs for {row['image']}")
    return {"source": source_record, "materialized": materialized_record}


def verify_fixture_files(repo: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    rows = fixture["forward_images"] + fixture["preprocess_trace_images"]
    unique = {}
    for row in rows:
        evidence = verify_bound_image(repo, row)
        image = row["image"]
        if image not in unique:
            unique[image] = {"source": evidence["source"], "materialized": evidence["materialized"], "materialized_by_selection": {}, "selection_bindings": []}
        elif any(unique[image]["source"].get(key) != evidence["source"].get(key) for key in ("sha256", "bytes")):
            raise ValueError(f"Deduplicated fixture image has inconsistent source byte evidence: {image}")
        selection = row["selection"]
        prior_materialized = unique[image]["materialized_by_selection"].get(selection)
        if prior_materialized is not None and any(prior_materialized.get(key) != evidence["materialized"].get(key) for key in ("sha256", "bytes")):
            raise ValueError(f"Repeated fixture selection has inconsistent materialized byte evidence: {selection}/{image}")
        unique[image]["materialized_by_selection"][selection] = evidence["materialized"]
        unique[image]["selection_bindings"].append({"selection": row["selection"], "manifest_order": row["manifest_order"], "expected_sha256": row["expected_sha256"], "materialized_expected_sha256": row["materialized_expected_sha256"]})
    return {"status": "matched", "images": unique, "unique_image_count": len(unique)}


def callable_source_evidence(label: str, value: Any) -> dict[str, Any]:
    try:
        source = inspect.getsource(value).encode("utf-8")
        source_file = inspect.getsourcefile(value)
    except (OSError, TypeError) as exc:
        raise NumericUnresolved(f"Cannot inspect pinned producer source for {label}: {exc}") from exc
    return {
        "label": label,
        "source_file": source_file,
        "source_sha256": sha256_bytes(source),
        "source_bytes": len(source),
    }


def producer_file_evidence(label: str, module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin or not Path(spec.origin).is_file():
        raise NumericUnresolved(f"Pinned producer module source is unavailable: {module_name}")
    path = Path(spec.origin)
    return {"label": label, "module": module_name, "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def runtime_source_evidence() -> dict[str, Any]:
    from ultralytics.data.augment import LetterBox
    from ultralytics.data.augment import Format
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.engine.exporter import Exporter
    from ultralytics.data.loaders import imread
    from ultralytics.engine.predictor import BasePredictor

    return {
        "inference_producer": [
            callable_source_evidence("ultralytics.data.loaders.imread", imread),
            callable_source_evidence("ultralytics.data.augment.LetterBox.__init__", LetterBox.__init__),
            callable_source_evidence("ultralytics.data.augment.LetterBox.get_params", LetterBox.get_params),
            callable_source_evidence("ultralytics.data.augment.LetterBox.apply_image", LetterBox.apply_image),
            callable_source_evidence("ultralytics.engine.predictor.BasePredictor.pre_transform", BasePredictor.pre_transform),
            callable_source_evidence("ultralytics.engine.predictor.BasePredictor.preprocess", BasePredictor.preprocess),
            callable_source_evidence("ultralytics.data.augment.Format._format_img", Format._format_img),
        ],
        "calibration_producer_source": [
            producer_file_evidence("calibration helper source", "ultralytics.engine.exporter"),
            producer_file_evidence("dataset source", "ultralytics.data.dataset"),
            producer_file_evidence("augmentation source", "ultralytics.data.augment"),
            callable_source_evidence("ultralytics.engine.exporter.Exporter.get_int8_calibration_dataloader", Exporter.get_int8_calibration_dataloader),
            callable_source_evidence("ultralytics.engine.exporter.Exporter.__init__", Exporter.__init__),
            callable_source_evidence("ultralytics.engine.exporter.Exporter.export_onnx", Exporter.export_onnx),
            callable_source_evidence("ultralytics.data.dataset.YOLODataset.load_image", YOLODataset.load_image),
            callable_source_evidence("ultralytics.data.dataset.YOLODataset.build_transforms", YOLODataset.build_transforms),
            callable_source_evidence("ultralytics.data.build.build_yolo_dataset", build_yolo_dataset),
            callable_source_evidence("ultralytics.data.augment.Format._format_img", Format._format_img),
            {"label": "repository calibration helper", "path": "scripts/uniform_build_repeat.py"},
        ],
        "calibration_loader_called": False,
        "calibration_component_called": True,
        "calibration_recipe_status": "source_inspected_scoped_image_trace; calibration loader dispatch forbidden",
        "export_semantics_status": "accepted_settings_and_observed_ONNX_schema_recorded; native_reference_is_not_an_export_copy",
    }


def array_digest(array: Any) -> str:
    return sha256_bytes(array.tobytes(order="C"))


def quantile_summary(values: Any, np: Any) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise NumericUnresolved("Cannot summarize empty numeric values")
    finite = flat[np.isfinite(flat)]
    result: dict[str, Any] = {
        "element_count": int(flat.size),
        "finite_count": int(finite.size),
        "infinite_count": int(np.isinf(flat).sum()),
        "nan_count": int(np.isnan(flat).sum()),
    }
    if finite.size:
        q = np.quantile(finite, [0.0, 0.5, 0.95, 0.99, 1.0], method="linear")
        result.update({key: float(value) for key, value in zip(("min", "p50", "p95", "p99", "max"), q)})
    else:
        result.update({key: None for key in ("min", "p50", "p95", "p99", "max")})
    return result


def compare_float_arrays(reference: Any, observed: Any, np: Any, domain: str) -> dict[str, Any]:
    if reference.shape != observed.shape:
        return {"status": "unresolved", "domain": domain, "reason": "shape_mismatch", "reference_shape": list(reference.shape), "observed_shape": list(observed.shape), "mismatch_count": None, "element_count": None}
    if reference.dtype != np.dtype("float32") or observed.dtype != np.dtype("float32"):
        return {"status": "unresolved", "domain": domain, "reason": "dtype_not_float32", "reference_dtype": str(reference.dtype), "observed_dtype": str(observed.dtype), "mismatch_count": None, "element_count": int(reference.size)}
    if not np.isfinite(reference).all() or not np.isfinite(observed).all():
        return {"status": "unresolved", "domain": domain, "reason": "nonfinite_output"}
    delta = np.abs(observed.astype(np.float64) - reference.astype(np.float64))
    abs_reference = np.abs(reference.astype(np.float64))
    relative = np.zeros_like(delta, dtype=np.float64)
    np.divide(delta, abs_reference, out=relative, where=abs_reference > 0)
    relative[(abs_reference == 0) & (delta > 0)] = np.inf
    close = np.isclose(reference, observed, rtol=LOCKED_TOLERANCES["float32"]["rtol"], atol=LOCKED_TOLERANCES["float32"]["atol"])
    bad = np.argwhere(~close)
    result = {
        "status": "pass" if bool(close.all()) else "fail",
        "domain": domain,
        "shape": list(reference.shape),
        "dtype": "float32",
        "rtol": LOCKED_TOLERANCES["float32"]["rtol"],
        "atol": LOCKED_TOLERANCES["float32"]["atol"],
        "mismatch_count": int((~close).sum()),
        "element_count": int(close.size),
        "max_abs": float(delta.max()),
        "max_relative": float(relative.max()) if np.isfinite(relative.max()) else None,
        "relative_infinite_count": int(np.isinf(relative).sum()),
        "abs_quantiles": quantile_summary(delta, np),
        "relative_quantiles": quantile_summary(relative, np),
        "comparison_equation": "abs(observed-reference) <= atol + rtol*abs(reference)",
        "first_offenders": [list(map(int, item)) for item in bad[:10]],
    }
    return result


def _tie_count(values: Any, np: Any) -> int:
    flat = np.asarray(values).reshape(-1)
    if flat.size < 2:
        return 0
    ordered = np.sort(flat)
    return int(np.sum(ordered[1:] == ordered[:-1]))


def compare_primary(model: str, reference: Any, observed: Any, np: Any) -> dict[str, Any]:
    expected = tuple(EXPECTED_OUTPUT_SHAPES[model])
    if tuple(reference.shape) != expected or tuple(observed.shape) != expected:
        return {
            "status": "unresolved",
            "domain": "native_primary",
            "reason": "primary_shape_mismatch",
            "expected_shape": list(expected),
            "reference_shape": list(reference.shape),
            "observed_shape": list(observed.shape),
        }
    if not np.isfinite(reference).all() or not np.isfinite(observed).all():
        return {"status": "unresolved", "domain": "native_primary", "reason": "nonfinite_primary_output"}
    if model == "yolov8n":
        boxes = compare_float_arrays(reference[:, 0:4, :], observed[:, 0:4, :], np, "yolov8n_raw_boxes_channels")
        scores = compare_float_arrays(reference[:, 4:7, :], observed[:, 4:7, :], np, "yolov8n_raw_score_channels")
        overall = "fail" if "fail" in (boxes["status"], scores["status"]) else ("unresolved" if "unresolved" in (boxes["status"], scores["status"]) else "pass")
        return {
            "status": overall,
            "domain": "yolov8n_raw_primary_fixed_channel_spans",
            "shape": list(reference.shape),
            "channel_spans": {"boxes": [0, 4], "scores": [4, 7]},
            "channel_results": {"boxes": boxes, "scores": scores},
            "mismatch_count": (boxes.get("mismatch_count") or 0) + (scores.get("mismatch_count") or 0),
            "element_count": (boxes.get("element_count") or 0) + (scores.get("element_count") or 0),
            "tolerance_order": "reference first in numpy.isclose(reference, observed, rtol, atol)",
        }
    if reference.dtype != np.dtype("float32") or observed.dtype != np.dtype("float32"):
        return {"status": "unresolved", "domain": "yolo26n_native_detections", "reason": "dtype_not_float32"}
    ref_classes = reference[..., 5]
    obs_classes = observed[..., 5]
    valid_class = (
        np.equal(ref_classes, np.rint(ref_classes)).all()
        and np.equal(obs_classes, np.rint(obs_classes)).all()
        and (ref_classes >= 0).all() and (ref_classes <= 2).all()
        and (obs_classes >= 0).all() and (obs_classes <= 2).all()
    )
    if not valid_class:
        return {"status": "unresolved", "domain": "yolo26n_native_detections", "reason": "class_id_not_finite_integer_in_range"}
    boxes = compare_float_arrays(reference[..., :4], observed[..., :4], np, "yolo26n_fixed_row_boxes")
    scores = compare_float_arrays(reference[..., 4:5], observed[..., 4:5], np, "yolo26n_fixed_row_scores")
    class_equal = bool(np.array_equal(ref_classes, obs_classes))
    numeric_statuses = (boxes["status"], scores["status"], "pass" if class_equal else "fail")
    numeric = {
        "status": "fail" if "fail" in numeric_statuses else ("unresolved" if "unresolved" in numeric_statuses else "pass"),
        "domain": "yolo26n_native_detections_fixed_row",
        "shape": list(reference.shape),
        "channel_spans": {"boxes": [0, 4], "score": [4, 5], "class_id": [5, 6]},
        "channel_results": {"boxes": boxes, "score": scores},
        "mismatch_count": (boxes.get("mismatch_count") or 0) + (scores.get("mismatch_count") or 0) + int(np.sum(ref_classes != obs_classes)),
        "element_count": (boxes.get("element_count") or 0) + (scores.get("element_count") or 0) + int(ref_classes.size),
        "tolerance_order": "reference first in numpy.isclose(reference, observed, rtol, atol)",
    }
    numeric["class_ids_exact"] = class_equal
    numeric["class_mismatch_count"] = int(np.sum(ref_classes != obs_classes))
    numeric["tie_count_reference_scores"] = _tie_count(reference[..., 4], np)
    numeric["tie_count_observed_scores"] = _tie_count(observed[..., 4], np)
    numeric["tie_handling"] = "fixed native row index; ties are reported and never rematched"
    return numeric


def extract_native_primary(output: Any, model: str, np: Any) -> tuple[Any, dict[str, Any]]:
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise NumericUnresolved(f"{model} native output is not primary-tensor-plus-head-dict")
    primary, debug = output
    if not hasattr(primary, "detach") or not isinstance(debug, dict):
        raise NumericUnresolved(f"{model} native output branch is not the accepted tuple contract")
    if model == "yolov8n":
        required = ("boxes", "scores", "feats")
        if any(key not in debug for key in required):
            raise NumericUnresolved("YOLOv8 native debug branch is not the accepted raw-head branch")
    else:
        branch = debug.get("one2one")
        if not isinstance(branch, dict) or any(key not in branch for key in ("boxes", "scores", "feats")):
            raise NumericUnresolved("YOLO26 native debug branch is not the accepted one2one branch")
    array = primary.detach().to("cpu").contiguous().numpy()
    if list(array.shape) != EXPECTED_OUTPUT_SHAPES[model]:
        raise NumericUnresolved(f"{model} native primary shape differs: {list(array.shape)}")
    return array, {"representation": "tuple_tensor_plus_dict", "debug_keys": list(debug.keys()), "shape": list(array.shape), "dtype": str(primary.dtype)}


def preprocess_array_contract(transformed: Any, np: Any) -> Any:
    """The pinned BasePredictor list path after LetterBox, for CPU-stub tests."""
    if transformed.ndim != 3 or transformed.shape[-1] != 3:
        raise NumericUnresolved("Preprocessed image is not HWC three-channel")
    rgb = transformed[..., ::-1]
    chw = np.ascontiguousarray(rgb.transpose((2, 0, 1)))
    return (chw.astype(np.float32) / 255.0)[None, ...]


def build_session_options(ort: Any) -> Any:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = True
    options.log_severity_level = 3
    return options


def child_command(script: Path, repo: Path, model: str, plan: Path) -> list[str]:
    return [sys.executable, str(script), "--child", "--repo-root", str(repo), "--model", model, "--plan", str(plan)]


def _runtime_packages(config: dict[str, Any], metadata: Any, torch: Any, ultralytics: Any, np: Any, ort: Any) -> dict[str, Any]:
    expected = config.get("runtime", {})
    observed = {
        "torch": getattr(torch, "__version__", None),
        "ultralytics": getattr(ultralytics, "__version__", None),
        "numpy": getattr(np, "__version__", None),
        "onnxruntime": getattr(ort, "__version__", None),
        "pycocotools": None,
    }
    try:
        observed["pycocotools"] = metadata.version("pycocotools")
    except metadata.PackageNotFoundError:
        observed["pycocotools"] = None
    mismatches = [name for name in ("torch", "ultralytics", "numpy", "pycocotools") if expected.get(name) and observed.get(name) != expected.get(name)]
    if mismatches:
        raise NumericUnresolved(f"Pinned runtime mismatch: {mismatches}; observed={observed}")
    return {"expected_from_config": {name: expected.get(name) for name in ("torch", "ultralytics", "numpy", "pycocotools")}, "observed": observed, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "mode": "CPUExecutionProvider; CUDA hidden"}


def load_child_runtime(config: dict[str, Any]) -> dict[str, Any]:
    import importlib.metadata as metadata

    try:
        import numpy as np
        import onnxruntime as ort
        import torch
        import ultralytics
        from ultralytics.data.augment import LetterBox
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.data.loaders import imread
        from ultralytics.engine.predictor import BasePredictor
        from ultralytics import YOLO
    except ImportError as exc:
        raise NumericUnresolved(f"Required pinned CPU runtime import failed: {exc}") from exc
    packages = _runtime_packages(config, metadata, torch, ultralytics, np, ort)
    providers = list(ort.get_available_providers())
    if "CPUExecutionProvider" not in providers:
        raise NumericUnresolved(f"ONNX Runtime CPU provider is unavailable: {providers}")
    return {"np": np, "ort": ort, "torch": torch, "YOLO": YOLO, "YOLODataset": YOLODataset, "LetterBox": LetterBox, "imread": imread, "BasePredictor": BasePredictor, "packages": packages, "available_providers": providers}


def trace_preprocess(path: Path, runtime: dict[str, Any], stride: int = 32) -> tuple[Any, dict[str, Any]]:
    np = runtime["np"]
    torch = runtime["torch"]
    image = runtime["imread"](str(path))
    if image is None or image.ndim != 3 or image.shape[-1] != 3:
        raise NumericUnresolved(f"Pinned producer failed to decode BGR three-channel image: {path}")
    predictor_cls = runtime["BasePredictor"]

    class TracePredictor:
        def __init__(self) -> None:
            self.imgsz = (640, 640)
            self.args = SimpleNamespace(rect=False)
            self.model = SimpleNamespace(format="pt", dynamic=False, stride=int(stride), fp16=False)
            self.device = torch.device("cpu")
            self.last_transformed = None

        def pre_transform(self, images: list[Any]) -> list[Any]:
            transformed = predictor_cls.pre_transform(self, images)
            self.last_transformed = transformed[0]
            return transformed

    predictor = TracePredictor()
    letterbox = runtime["LetterBox"](new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, center=True, stride=int(stride), padding_value=114)
    params = letterbox.get_params({"img": image.copy()})
    tensor = predictor_cls.preprocess(predictor, [image])
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or str(tensor.dtype) != "torch.float32":
        raise NumericUnresolved("Pinned preprocessing did not produce CPU float32 tensor")
    tensor_np = np.ascontiguousarray(tensor.detach().cpu().numpy())
    transformed = predictor.last_transformed
    if transformed is None or tuple(tensor_np.shape) != (1, 3, 640, 640) or not np.isfinite(tensor_np).all():
        raise NumericUnresolved("Pinned preprocessing output shape/finite contract is invalid")
    trace = {
        "path": str(path),
        "decoded": {"shape": list(image.shape), "dtype": str(image.dtype), "color_order": "BGR", "bytes_sha256": array_digest(np.ascontiguousarray(image))},
        "letterbox": {
            "new_shape": [640, 640],
            "auto": False,
            "scale_fill": False,
            "scaleup": True,
            "center": True,
            "stride": int(stride),
            "padding_value": 114,
            "interpolation": int(letterbox.interpolation),
            "orig_shape": [int(value) for value in params["orig_shape"]],
            "ratio": [float(value) for value in params["ratio"]],
            "new_unpad": [int(value) for value in params["new_unpad"]],
            "padding": {key: int(params[key]) for key in ("top", "bottom", "left", "right")},
        },
        "transformed": {"shape": list(transformed.shape), "dtype": str(transformed.dtype), "bytes_sha256": array_digest(np.ascontiguousarray(transformed))},
        "tensor": {"shape": list(tensor_np.shape), "dtype": str(tensor_np.dtype), "range": [float(tensor_np.min()), float(tensor_np.max())], "bytes_sha256": array_digest(tensor_np)},
        "original_image_size_hw": [int(image.shape[0]), int(image.shape[1])],
        "operations": ["decode pinned BGR", "LetterBox via BasePredictor.pre_transform", "BGR_to_RGB", "BHWC_to_BCHW", "contiguous", "uint8_to_float32", "divide_by_255", "batch_axis"],
        "same_tensor_for_source_and_onnx": True,
    }
    return tensor, trace


def trace_calibration_preprocess(path: Path, runtime: dict[str, Any], stride: int = 32) -> dict[str, Any]:
    """Trace only the pinned calibration image components on one bounded image.

    This deliberately calls ``YOLODataset.load_image`` and the validation
    ``LetterBox``/``Format`` semantics directly on a small stub.  It does not
    instantiate a dataset, read labels, use an image cache, build a dataloader,
    or dispatch ``Exporter.get_int8_calibration_dataloader``.
    """
    np = runtime["np"]
    dataset_cls = runtime["YOLODataset"]
    image_stub = SimpleNamespace(
        ims=[None],
        im_files=[str(path)],
        npy_files=[path.with_suffix(".npy")],
        channels=3,
        cv2_flag=1,
        prefix="bounded calibration trace: ",
        imgsz=640,
        augment=False,
        buffer=[],
        max_buffer_length=0,
        cache=False,
    )
    loaded, original_hw, resized_hw = dataset_cls.load_image(image_stub, 0, rect_mode=True, resize_short=False)
    if loaded is None or loaded.ndim != 3 or loaded.shape[-1] != 3:
        raise NumericUnresolved("Calibration component loader did not produce BGR HWC image")
    letterbox = runtime["LetterBox"](
        new_shape=(640, 640), auto=False, scale_fill=False, scaleup=False,
        center=True, stride=int(stride), padding_value=114,
    )
    params = letterbox.get_params({"img": loaded.copy()})
    transformed = letterbox(image=loaded.copy())
    if transformed.ndim != 3 or tuple(transformed.shape[:2]) != (640, 640):
        raise NumericUnresolved("Calibration LetterBox did not produce 640x640 image")
    # Validation Format._format_img emits uint8 RGB CHW; the repository
    # calibration helper performs /255 on that stream before TensorRT consumes
    # it.  Keep both stages explicit without constructing the loader.
    rgb_chw = np.ascontiguousarray(transformed[..., ::-1].transpose((2, 0, 1)))
    if rgb_chw.dtype != np.dtype("uint8"):
        raise NumericUnresolved(f"Calibration formatted image dtype differs: {rgb_chw.dtype}")
    normalized = rgb_chw.astype(np.float32) / 255.0
    return {
        "path": str(path),
        "stage": "bounded_calibration_component_trace",
        "producer_calls": ["YOLODataset.load_image", "LetterBox.__call__", "Format._format_img_equivalent"],
        "loader_dispatch": {"Exporter.get_int8_calibration_dataloader": False, "build_yolo_dataset": False, "dataloader": False},
        "dataset_side_effects": {"labels_read": False, "image_cache_read": False, "dataset_instantiated": False},
        "load_image": {
            "rect_mode": True,
            "resize_short": False,
            "input_path_kind": "materialized calibration image",
            "original_hw": [int(value) for value in original_hw],
            "resized_hw": [int(value) for value in resized_hw],
            "dtype": str(loaded.dtype),
            "color_order": "BGR",
            "bytes_sha256": array_digest(np.ascontiguousarray(loaded)),
        },
        "letterbox": {
            "new_shape": [640, 640],
            "auto": False,
            "scale_fill": False,
            "scaleup": False,
            "center": True,
            "stride": int(stride),
            "padding_value": 114,
            "interpolation": int(letterbox.interpolation),
            "orig_shape": [int(value) for value in params["orig_shape"]],
            "ratio": [float(value) for value in params["ratio"]],
            "new_unpad": [int(value) for value in params["new_unpad"]],
            "padding": {key: int(params[key]) for key in ("top", "bottom", "left", "right")},
        },
        "formatted_uint8": {"shape": list(rgb_chw.shape), "dtype": str(rgb_chw.dtype), "layout": "RGB_CHW", "bytes_sha256": array_digest(rgb_chw)},
        "normalization": {"applied": True, "operation": "float32(rgb_chw) / 255.0", "shape": list(normalized.shape), "dtype": str(normalized.dtype), "range": [float(normalized.min()), float(normalized.max())], "bytes_sha256": array_digest(normalized)},
        "calibration_trace_scope": "three accepted anchors only; no full calibration set, cache, test or negative-test data",
    }


def validate_native_head(network: Any, model: str, accepted_contract: dict[str, Any]) -> dict[str, Any]:
    network = network.to("cpu").eval()
    head = network.model[-1]
    expected = accepted_contract.get("expected_contract", {})
    if type(head).__name__ != expected.get("head_type") or int(getattr(head, "i")) != expected.get("head_index") or bool(getattr(head, "end2end")) != expected.get("end2end"):
        raise NumericUnresolved(f"Native head identity differs for {model}")
    if any(getattr(parameter, "device", None).type != "cpu" for parameter in network.parameters()):
        raise NumericUnresolved(f"Native parameters are not on CPU for {model}")
    parameter_dtypes = {str(getattr(parameter, "dtype", None)) for parameter in network.parameters()}
    if parameter_dtypes != {"torch.float32"}:
        raise NumericUnresolved(f"Native parameters are not explicitly float32 for {model}: {sorted(parameter_dtypes)}")
    return {"head": {"type": type(head).__name__, "index": int(head.i), "end2end": bool(head.end2end)}, "parameters_dtype": "torch.float32", "float_explicit": True}


def verify_native_model(network: Any, model: str, accepted_contract: dict[str, Any], tensor: Any, runtime: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    torch = runtime["torch"]
    head_contract = validate_native_head(network, model, accepted_contract)
    with torch.no_grad():
        output = network(tensor)
    primary, contract = extract_native_primary(output, model, runtime["np"])
    if str(primary.dtype) != "float32" and str(primary.dtype) != "torch.float32":
        raise NumericUnresolved(f"Native primary dtype differs for {model}: {primary.dtype}")
    return primary, {**head_contract, "output": contract}


def run_onnx_session(onnx_path: Path, model: str, runtime: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    ort = runtime["ort"]
    np = runtime["np"]
    options = build_session_options(ort)
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    actual_providers = list(session.get_providers())
    if actual_providers != ["CPUExecutionProvider"]:
        raise NumericUnresolved(f"ONNX session provider selection was not CPU-only: {actual_providers}")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    expected_shape = EXPECTED_OUTPUT_SHAPES[model]
    if len(inputs) != 1 or inputs[0].name != "images" or inputs[0].type != "tensor(float)" or list(inputs[0].shape) != [1, 3, 640, 640]:
        raise NumericUnresolved(f"ONNX input contract differs for {model}: {[(x.name, x.type, x.shape) for x in inputs]}")
    if len(outputs) != 1 or outputs[0].name != "output0" or outputs[0].type != "tensor(float)" or list(outputs[0].shape) != expected_shape:
        raise NumericUnresolved(f"ONNX output branch/schema differs for {model}: {[(x.name, x.type, x.shape) for x in outputs]}")
    return session, {
        "providers_requested": ["CPUExecutionProvider"],
        "providers_observed": actual_providers,
        "options": {"graph_optimization_level": "ORT_ENABLE_BASIC", "intra_op_num_threads": 2, "inter_op_num_threads": 1, "execution_mode": "ORT_SEQUENTIAL", "enable_mem_pattern": False, "enable_cpu_mem_arena": True},
        "input": {"name": inputs[0].name, "type": inputs[0].type, "shape": list(inputs[0].shape)},
        "output": {"name": outputs[0].name, "type": outputs[0].type, "shape": list(outputs[0].shape)},
    }


def run_model_child(repo: Path, plan: dict[str, Any], model: str, out_dir: Path) -> dict[str, Any]:
    set_cpu_environment()
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = out_dir / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(repo_path(repo, Path(plan["config_path"])))
    runtime = load_child_runtime(config)
    model_plan = plan["models"][model]
    checkpoint_path = repo_path(repo, Path(model_plan["checkpoint"]["expected"]["path"]))
    onnx_path = repo_path(repo, Path(model_plan["onnx"]["path"]))
    checkpoint_before = file_evidence(repo, checkpoint_path)
    onnx_before = file_evidence(repo, onnx_path)
    if checkpoint_before.get("sha256") != model_plan["checkpoint"]["expected"]["sha256"] or onnx_before.get("sha256") != EXPECTED_ONNX_SHA256[model]:
        raise ValueError(f"Child binary binding differs before forward for {model}")
    fixture_binding = verify_fixture_files(repo, plan["fixture"])
    source_evidence = runtime_source_evidence()
    model_loader = runtime["YOLO"](str(checkpoint_path), task="detect")
    network = model_loader.model.to("cpu").float().eval()
    native_head_contract = validate_native_head(network, model, model_plan["accepted_contract"])
    raw_stride = getattr(network, "stride", 32)
    try:
        model_stride = int(raw_stride.max().item())
    except AttributeError:
        model_stride = int(raw_stride)
    unique_rows: dict[str, list[dict[str, Any]]] = {}
    for row in plan["fixture"]["forward_images"] + plan["fixture"]["preprocess_trace_images"]:
        unique_rows.setdefault(row["image"], []).append(row)
    traces = {}
    calibration_traces = {}
    tensors = {}
    for image, bound_rows in unique_rows.items():
        row = bound_rows[0]
        source_path = resolve_bound_source_image(repo, row)
        tensor, trace = trace_preprocess(source_path, runtime, stride=model_stride)
        trace.update({"selection": row["selection"], "image_id": row["image_id"], "expected_source_sha256": row["expected_sha256"], "selection_bindings": [{"selection": item["selection"], "manifest_order": item["manifest_order"], "image_id": item["image_id"], "expected_source_sha256": item["expected_sha256"]} for item in bound_rows]})
        traces[image] = trace
        tensors[image] = tensor
        write_json_no_overwrite(partial_dir / f"inference_trace_{row['selection']}_{row['image_id']}.json", trace)
    for row in plan["fixture"]["preprocess_trace_images"]:
        image = row["image"]
        materialized_path = resolve_bound_materialized_image(repo, row)
        calibration_trace = trace_calibration_preprocess(materialized_path, runtime, stride=model_stride)
        calibration_trace.update({"selection": row["selection"], "image_id": row["image_id"], "image": image, "expected_source_sha256": row["expected_sha256"], "materialized_expected_sha256": row["materialized_expected_sha256"]})
        calibration_traces[f"{row['selection']}::{image}"] = calibration_trace
        write_json_no_overwrite(partial_dir / f"calibration_trace_{row['selection']}_{row['image_id']}.json", calibration_trace)

    session, onnx_contract = run_onnx_session(onnx_path, model, runtime)
    comparisons = []
    native_contract = None
    for row in plan["fixture"]["forward_images"]:
        image = row["image"]
        tensor = tensors[image]
        input_np = runtime["np"].ascontiguousarray(tensor.detach().cpu().numpy())
        with runtime["torch"].no_grad():
            native_output = network(tensor)
        native_primary, native_contract_row = extract_native_primary(native_output, model, runtime["np"])
        if native_contract is None:
            native_contract = native_contract_row
        onnx_outputs = session.run(["output0"], {"images": input_np})
        if len(onnx_outputs) != 1:
            raise NumericUnresolved(f"ONNX returned an unexpected output count for {model}")
        onnx_primary = runtime["np"].ascontiguousarray(onnx_outputs[0])
        comparison = compare_primary(model, native_primary, onnx_primary, runtime["np"])
        comparison_record = {
            "selection": row["selection"],
            "manifest_order": row["manifest_order"],
            "image": image,
            "image_id": row["image_id"],
            "input_tensor": {"shape": list(input_np.shape), "dtype": str(input_np.dtype), "sha256": array_digest(input_np)},
            "source_primary": {"shape": list(native_primary.shape), "dtype": str(native_primary.dtype), "sha256": array_digest(runtime["np"].ascontiguousarray(native_primary))},
            "onnx_primary": {"shape": list(onnx_primary.shape), "dtype": str(onnx_primary.dtype), "sha256": array_digest(onnx_primary)},
            "same_verified_input": True,
            "comparison": comparison,
        }
        comparisons.append(comparison_record)
        write_json_no_overwrite(partial_dir / f"comparison_{row['selection']}_{row['image_id']}.json", comparison_record)
    checkpoint_after = file_evidence(repo, checkpoint_path)
    onnx_after = file_evidence(repo, onnx_path)
    if checkpoint_before != checkpoint_after or onnx_before != onnx_after:
        raise ValueError(f"Input binary changed during numeric verification for {model}")
    fixture_binding_after = verify_fixture_files(repo, plan["fixture"])
    if fixture_binding != fixture_binding_after:
        raise ValueError(f"Bound source/materialized image bytes changed during numeric verification for {model}")
    statuses = [row["comparison"]["status"] for row in comparisons]
    overall = "pass" if statuses and all(status == "pass" for status in statuses) else ("fail" if any(status == "fail" for status in statuses) else "unresolved")
    return {
        "schema_version": 1,
        "study": STUDY,
        "model": model,
        "status": "completed",
        "numeric_status": overall,
        "runtime": runtime["packages"],
        "onnxruntime": onnx_contract,
        "producer_source_evidence": source_evidence,
        "reference_semantics": model_plan["reference_semantics"],
        "checkpoint": {"before": checkpoint_before, "after": checkpoint_after, "unchanged": checkpoint_before == checkpoint_after},
        "onnx": {"before": onnx_before, "after": onnx_after, "unchanged": onnx_before == onnx_after, "expected_sha256": EXPECTED_ONNX_SHA256[model]},
        "fixture_binding": fixture_binding,
        "fixture_binding_after": fixture_binding_after,
        "fixture_binding_unchanged": fixture_binding == fixture_binding_after,
        "preprocess_traces": traces,
        "calibration_preprocess_traces": calibration_traces,
        "native_head_contract": native_head_contract,
        "model_stride": model_stride,
        "native_output_contract": native_contract,
        "comparisons": comparisons,
        "forward_counts": {"source_cpu_fp32": len(comparisons), "onnx_cpu": len(comparisons)},
        "audit_flags": {"export_performed": False, "build_performed": False, "calibration_loader_called": False, "calibration_component_called": True, "gpu_used": False, "scored_run_authorized": False},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_reference_semantics(model: str, graph_doc: dict[str, Any], accepted_contract: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    observed_schema = graph_doc.get("onnx_schema", {})
    requested = dict(ACCEPTED_EXPORT_ARGUMENTS)
    preserved = graph_doc.get("provenance", {}).get("preserved_records", {})
    for record in preserved.values():
        export = (record.get("summary", {}) or {}).get("export") or {}
        if isinstance(export.get("requested_arguments"), dict):
            requested = dict(export["requested_arguments"])
            break
    return {
        "model": model,
        "native_reference": {
            "mode": "frozen PyTorch model.model.to(cpu).float().eval() under torch.no_grad",
            "dtype": "explicit float32 primary output",
            "head_contract": accepted_contract.get("expected_contract", {}),
            "forward_counts": {"native_primary": 0, "export_reference": 0},
            "fusion": "not_reproduced_in_native_reference; any historical exporter fusion belongs to the ONNX artifact and is not inferred from numeric agreement",
        },
        "accepted_export_settings": requested,
        "observed_onnx_schema": {
            "inputs": observed_schema.get("inputs", []),
            "outputs": observed_schema.get("outputs", []),
            "opset_imports": observed_schema.get("opset_imports", []),
            "static_shapes": observed_schema.get("effective_observations", {}).get("static_shapes"),
            "dynamic_shape_observed": observed_schema.get("effective_observations", {}).get("dynamic_shape_observed"),
            "float_graph_expected": observed_schema.get("effective_observations", {}).get("float_graph_expected"),
            "quantization_nodes": observed_schema.get("quantization_nodes", []),
        },
        "head_output_semantics": {
            "output_name": "output0",
            "output_shape": EXPECTED_OUTPUT_SHAPES[model],
            "max_det": config.get("runtime", {}).get("max_det", 300),
            "postprocess": "no additional NMS/rematching in this diagnostic; YOLO26 fixed native row index",
            "coordinate_or_packing": "accepted graph schema/structural audit is referenced; native-to-ONNX numeric comparison is measured separately",
        },
        "source_inspection_boundary": "Exporter and dataset/augmentation source hashes are recorded at runtime; source inspection is evidence of implementation, not proof of a producer call unless the corresponding call flag is true",
    }


def _model_plan(repo: Path, config: dict[str, Any], accepted: dict[str, Any], graph_audit: dict[str, Any], model: str) -> dict[str, Any]:
    checkpoint = validate_checkpoint_contract(repo, config, accepted, model)
    graph_doc = graph_audit["models"][model]
    accepted_contract = accepted_model_contracts(accepted, [model])[model]
    onnx_relative = f"results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2/models/{model}/model.onnx"
    onnx_path = repo_path(repo, Path(onnx_relative))
    onnx_record = file_evidence(repo, onnx_path)
    if onnx_record.get("sha256") != EXPECTED_ONNX_SHA256[model]:
        raise ValueError(f"Existing ONNX hash differs for {model}: {onnx_record.get('sha256')}")
    graph_onnx = graph_doc.get("provenance", {}).get("onnx", {})
    before = graph_onnx.get("before", {})
    after = graph_onnx.get("after", {})
    if before.get("sha256") != EXPECTED_ONNX_SHA256[model] or after.get("sha256") != EXPECTED_ONNX_SHA256[model] or graph_onnx.get("unchanged_during_audit") is not True:
        raise ValueError(f"Graph-v4 does not bind unchanged accepted ONNX bytes for {model}")
    return {
        "checkpoint": checkpoint,
        "onnx": {"path": onnx_relative, "expected_sha256": EXPECTED_ONNX_SHA256[model], "before": onnx_record},
        "graph_contract": {"audit_model_report": f"{GRAPH_AUDIT_ROOT_NAME}/{GRAPH_AUDIT_MODEL_FILES[model]}", "mapping_status": graph_doc.get("mapping_status"), "output_shape": EXPECTED_OUTPUT_SHAPES[model]},
        "accepted_contract": accepted_contract,
        "reference_semantics": build_reference_semantics(model, graph_doc, accepted_contract, config),
    }


def _plan_report(plan: dict[str, Any]) -> str:
    lines = [
        "# Precision-head CPU numeric verification plan",
        "",
        f"- Study: `{plan['study']}`",
        f"- Status: `{plan['status']}`",
        "- Scope: frozen PyTorch FP32 versus existing ONNX, CPU only, eight U42 train images plus three preprocessing trace anchors.",
        "- No export, TensorRT, calibration loader, GPU, NMS, AP, official test or negative-test access is part of this diagnostic.",
        "- The command in the protocol is a candidate only until Astra reviews this package.",
        "",
        "## Fixed bindings",
        "",
        f"- Readiness: `{plan['readiness']['root']}` at `{plan['readiness']['accepted_git_commit']}`.",
        f"- Graph audit: `{plan['graph_audit']['root']}` at `{plan['graph_audit']['commit']}`.",
        f"- Forward fixture: `{plan['fixture']['forward_image_count']}` images; `{plan['fixture']['selection_rule']}`.",
        f"- Preprocessing anchors: `{', '.join(row['selection'] for row in plan['fixture']['preprocess_trace_images'])}`.",
        "",
        "## Tolerances",
        "",
        "- Float32 raw boxes/scores/channels: `rtol=1e-4`, `atol=1e-5`; report max, quantiles and offending locations.",
        "- YOLO26 class IDs: finite integers in 0..2 and exact equality at the native row index.",
        "- Ties: report them; never rematch, sort, clip, NMS or tune thresholds.",
    ]
    return "\n".join(lines) + "\n"


def _final_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Precision-head CPU numeric verification",
        "",
        f"- Study: `{manifest['study']}`",
        f"- Status: `{manifest['status']}`",
        f"- Execution status: `{manifest.get('execution_status', manifest['status'])}`; numeric verdict: `{manifest.get('numeric_verdict', {}).get('status', 'not_observed')}`.",
        "- This diagnostic is not a scored accuracy test and does not authorize TensorRT work.",
        "- Source and existing ONNX bytes are checked before and after; raw arrays are not published.",
        "",
        "## Results",
        "",
    ]
    for row in manifest.get("models", []):
        lines.append(f"- `{row.get('model')}`: execution `{row.get('status')}`, numeric status `{row.get('numeric_status', 'not_observed')}`, forward counts `{row.get('forward_counts', {})}`.")
    lines += [
        "",
        "## Boundary",
        "",
        "- A pass means the accepted native output contract agrees with the existing ONNX output on this eight-image CPU fixture under the locked tensor path.",
        "- It does not prove dataset-wide accuracy, calibration equivalence, TensorRT compatibility, GPU behavior or engine reproducibility.",
        "- Any fail or unresolved result remains evidence for review; no model, output or preprocessing change is applied automatically.",
    ]
    return "\n".join(lines) + "\n"


def aggregate_numeric_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = ("pass", "fail", "unresolved", "not_observed")
    counts = {status: 0 for status in allowed}
    for row in rows:
        status = row.get("numeric_status") if row.get("status") == "completed" else "not_observed"
        if status not in counts:
            status = "unresolved"
        counts[status] += 1
    if counts["fail"]:
        overall = "fail"
    elif counts["unresolved"]:
        overall = "unresolved"
    elif counts["not_observed"]:
        overall = "not_observed"
    else:
        overall = "pass"
    return {"status": overall, "counts": counts, "execution_and_numeric_are_separate": True}


def publishable_artifact_inventory(output_root: Path) -> list[str]:
    """List JSON/text evidence, including partial lifecycle evidence, not binaries."""
    inventory = {"numeric_plan.json", "numeric_manifest.json", "report.md"}
    if output_root.is_dir():
        for path in output_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(output_root).as_posix()
            if path.suffix in (".json", ".md", ".log") or path.name.endswith(".log.gz"):
                inventory.add(relative)
    return sorted(inventory)


def run_parent(args: argparse.Namespace, repo: Path) -> int:
    set_cpu_environment()
    models = selected_models(args.model)
    output_root = repo_path(repo, args.out_dir)
    if output_root.exists():
        raise FileExistsError(f"Numeric output root exists; refusing overwrite/resume: {output_root}")
    readiness_root = repo_path(repo, args.readiness_root)
    graph_root = repo_path(repo, args.graph_audit_root)
    source_root = repo_path(repo, args.source_root)
    accepted = graph.validate_readiness_artifact(repo, readiness_root)
    binding = graph.validate_config_binding(repo, accepted)
    config_path = repo_path(repo, Path(binding["path"]))
    config = readiness.load_config(config_path, repo)
    graph_audit = validate_graph_audit_artifact(repo, graph_root, models)
    fixture = build_fixture_plan(accepted)
    for row in fixture["forward_images"] + fixture["preprocess_trace_images"]:
        verify_bound_image(repo, row)
    model_plans = {model: _model_plan(repo, config, accepted, graph_audit, model) for model in models}
    if not source_root.is_dir():
        raise FileNotFoundError(f"Existing graph source root is missing: {source_root}")
    plan = {
        "schema_version": 1,
        "study": STUDY,
        "status": "dispatching_cpu_children",
        "repo_head": git_head(repo),
        "output_root": str(output_root.relative_to(repo.resolve()).as_posix()),
        "config_path": str(config_path.relative_to(repo.resolve()).as_posix()),
        "source_root": str(source_root.relative_to(repo.resolve()).as_posix()),
        "selected_models": models,
        "readiness": {"root": str(readiness_root.relative_to(repo.resolve()).as_posix()), "accepted_git_commit": accepted["accepted_git_commit"], "accepted_execution_commit": accepted["accepted_execution_commit"], "files": accepted["files"]},
        "graph_audit": {"root": str(graph_root.relative_to(repo.resolve()).as_posix()), "commit": GRAPH_AUDIT_COMMIT, "files": graph_audit["files"]},
        "config_binding": {key: value for key, value in binding.items() if key != "config"},
        "fixture": fixture,
        "calibration_trace_contract": {"anchors": fixture["preprocess_trace_images"], "arguments": CALIBRATION_TRACE_ARGUMENTS, "full_loader_called": False, "cache_created": False, "labels_read": False},
        "models": model_plans,
        "environment": {"requested": CPU_ENVIRONMENT, "mode": "CPU parent; model-specific children import Torch/ONNX Runtime", "expected_packages": {name: config.get("runtime", {}).get(name) for name in ("torch", "ultralytics", "numpy", "pycocotools")}, "onnx_provider": "CPUExecutionProvider"},
        "tolerances": LOCKED_TOLERANCES,
        "code_provenance": {
            "numeric_script": file_evidence(repo, Path(__file__)),
            "graph_script": file_evidence(repo, repo / "scripts/prepare_precision_head_confirmation_graph.py"),
            "readiness_helper": file_evidence(repo, repo / "scripts/prepare_precision_head_confirmation.py"),
            "repository_calibration_helper": file_evidence(repo, repo / "scripts/uniform_build_repeat.py"),
            "config": file_evidence(repo, config_path),
        },
        "audit_flags": {"export_performed": False, "build_performed": False, "calibration_loader_called": False, "calibration_component_called": True, "gpu_used": False, "scored_run_authorized": False},
        "no_overwrite": True,
    }
    output_root.mkdir(parents=True)
    write_json_no_overwrite(output_root / "numeric_plan.json", plan)
    model_rows = []
    failed = False
    script = Path(__file__).resolve()
    for model in models:
        model_dir = output_root / "models" / model
        model_dir.mkdir(parents=True)
        command = child_command(script, repo, model, output_root / "numeric_plan.json")
        result = subprocess.run(command, cwd=repo, env=os.environ.copy(), capture_output=True, text=True, check=False)
        write_text_no_overwrite(output_root / "logs" / f"{model}.log", result.stdout + result.stderr)
        report_path = model_dir / "numeric_report.json"
        if report_path.is_file():
            row = read_json(report_path)
        elif (model_dir / "failure.json").is_file():
            row = read_json(model_dir / "failure.json")
            if row.get("model") != model or row.get("status") != "failed":
                raise ValueError(f"Existing child failure record is invalid for {model}")
        else:
            failed = True
            row = {"schema_version": 1, "study": STUDY, "model": model, "status": "failed", "numeric_status": "not_observed", "error_type": "ChildProcessError", "error": f"model child exited {result.returncode}", "command": command, "partial_files": [], "no_silent_resume": True, "audit_flags": {"export_performed": False, "build_performed": False, "calibration_loader_called": False, "gpu_used": False, "scored_run_authorized": False}}
            write_json_no_overwrite(model_dir / "failure.json", row)
        if result.returncode != 0 or row.get("status") != "completed":
            failed = True
        model_rows.append(row)
    manifest = {
        "schema_version": 1,
        "study": STUDY,
        "status": "failed" if failed else "completed",
        "models": model_rows,
        "execution_status": "failed" if failed else "completed",
        "numeric_verdict": aggregate_numeric_verdict(model_rows),
        "plan_path": "numeric_plan.json",
        "artifact_inventory": publishable_artifact_inventory(output_root),
        "audit_flags": {"export_performed": False, "build_performed": False, "calibration_loader_called": False, "gpu_used": False, "scored_run_authorized": False},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_no_overwrite(output_root / "numeric_manifest.json", manifest)
    write_text_no_overwrite(output_root / "report.md", _final_report(manifest))
    print(f"DONE: {output_root / 'numeric_manifest.json'}")
    return 1 if failed else 0


def run_child(args: argparse.Namespace) -> int:
    repo = repo_path(Path(args.repo_root).resolve(), Path("."))
    plan = read_json(repo_path(repo, Path(args.plan)))
    model = args.model
    out_dir = repo_path(repo, Path(plan["output_root"])) / "models" / model
    try:
        result = run_model_child(repo, plan, model, out_dir)
        write_json_no_overwrite(out_dir / "numeric_report.json", result)
        print(f"DONE MODEL {model}: {out_dir / 'numeric_report.json'}")
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "study": STUDY,
            "model": model,
            "status": "failed",
            "numeric_status": "not_observed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "partial_files": sorted(path.relative_to(repo_path(repo, Path(plan["output_root"]))).as_posix() for path in repo_path(repo, Path(plan["output_root"])).rglob("*") if path.is_file()),
            "no_silent_resume": True,
            "audit_flags": {"export_performed": False, "build_performed": False, "calibration_loader_called": False, "gpu_used": False, "scored_run_authorized": False},
        }
        write_json_no_overwrite(out_dir / "failure.json", failure)
        print(f"FAILED MODEL {model}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-only frozen FP32 to accepted ONNX numeric/preprocessing verification")
    parser.add_argument("--readiness-root", type=Path, default=DEFAULT_READINESS_ROOT)
    parser.add_argument("--graph-audit-root", type=Path, default=DEFAULT_GRAPH_AUDIT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", choices=(*MODEL_CHOICES, "all"), default="all")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.child:
        if args.repo_root is None or args.plan is None or args.model == "all":
            raise ValueError("Internal child requires --repo-root, --plan and one model")
        return run_child(args)
    repo = Path(__file__).resolve().parents[1]
    return run_parent(args, repo)


if __name__ == "__main__":
    raise SystemExit(main())
