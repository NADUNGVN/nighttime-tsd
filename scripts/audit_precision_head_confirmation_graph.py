#!/usr/bin/env python3
"""Audit preserved precision-head ONNX files without exporting or executing models.

This is the bounded G5 diagnostic boundary.  It reads only an existing v2
ONNX tree, its preserved JSON/log provenance, and the accepted readiness/config
binding.  ONNX checker, shape inference and the graph mapper run in memory.
No Ultralytics model is loaded, no forward/exporter/calibration/TensorRT path
is called, and no source artifact is modified.  Unresolved mapping is evidence
and is persisted rather than raised before the diagnostic is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import prepare_precision_head_confirmation_graph as graph


AUDIT_STUDY = "precision_head_confirmation_graph_audit_v1"
SOURCE_ROOT_DEFAULT = Path("results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2")
OUTPUT_ROOT_DEFAULT = Path("results/measurement_audit_v1/precision_head_confirmation_graph_audit_v3")
MODEL_CHOICES = graph.MODEL_CHOICES


def sha256_file(path: Path) -> str:
    return graph.sha256_file(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_no_overwrite(path: Path, value: Any) -> None:
    graph.write_json_no_overwrite(path, value)


def write_text_no_overwrite(path: Path, value: str) -> None:
    graph.write_text_no_overwrite(path, value)


def relative_path(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def file_evidence(repo: Path, path: Path) -> dict[str, Any]:
    path = path.resolve()
    record: dict[str, Any] = {
        "path": relative_path(repo, path),
        "absolute_path": str(path),
        "exists": path.is_file(),
    }
    if path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return record


def _record_summary(repo: Path, source_root: Path, model_dir: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    paths = [
        (f"models/{model_dir.name}/model_prepare.json", model_dir / "model_prepare.json"),
        (f"models/{model_dir.name}/failure.json", model_dir / "failure.json"),
        ("prepare_plan.json", source_root / "prepare_plan.json"),
        ("failure.json", source_root / "failure.json"),
    ]
    for key, path in paths:
        if not path.is_file():
            continue
        row: dict[str, Any] = file_evidence(repo, path)
        try:
            value = read_json(path)
            row["summary"] = {
                "status": value.get("status"),
                "stage": value.get("stage"),
                "error_type": value.get("error_type"),
                "error": value.get("error"),
                "partial_files": value.get("partial_files"),
                "export": value.get("export"),
                "checkpoint": value.get("checkpoint"),
                "no_silent_resume": value.get("no_silent_resume"),
                "scored_run_authorized": value.get("scored_run_authorized"),
            }
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            row["parse_error"] = f"{type(exc).__name__}:{exc}"
        records[key] = row
    log = model_dir.parent.parent / "logs" / f"{model_dir.name}.log"
    if log.is_file():
        records["log"] = file_evidence(repo, log)
    return records


def _node_with_shapes(node: dict[str, Any], shapes: dict[str, list[Any]]) -> dict[str, Any]:
    return {
        "index": node.get("index"),
        "name": node["name"],
        "normalized_name": node.get("normalized_name", graph.normalize_module_name(node["name"])),
        "op_type": node["op_type"],
        "inputs": node["inputs"],
        "outputs": node["outputs"],
        "input_shapes": [shapes.get(name) for name in node["inputs"]],
        "output_shapes": [shapes.get(name) for name in node["outputs"]],
        "attributes": node["attributes"],
    }


def _head_nodes(graph_doc: dict[str, Any], model: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = model.get("head_output_evidence") or {}
    prefix = f"model.{evidence.get('head_index')}."
    shapes = graph_doc.get("tensor_shapes", {})
    return [
        _node_with_shapes(node, shapes)
        for node in graph_doc.get("nodes", [])
        if node.get("normalized_name", "").startswith(prefix)
    ]


def _post_merge_nodes(graph_doc: dict[str, Any], mapping: dict[str, Any]) -> list[dict[str, Any]]:
    merge = mapping.get("branch_merge") or {}
    audit = merge.get("downstream_semantic_audit") or {}
    names = {row.get("node") for row in audit.get("visited_nodes", []) if row.get("node")}
    if merge.get("node"):
        names.add(merge["node"])
    shapes = graph_doc.get("tensor_shapes", {})
    return [
        _node_with_shapes(node, shapes)
        for node in graph_doc.get("nodes", [])
        if node.get("name") in names
    ]


def _source_to_node_matches(
    accepted_model: dict[str, Any], mapping: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    audits = mapping.get("active_branch_audit", {})
    source_mapping = accepted_model.get("active_convolution_mapping", {})
    for branch, source_convs in source_mapping.items():
        branch_audit = audits.get(branch, {})
        candidates = branch_audit.get("candidate_nodes", [])
        rows = []
        for source in source_convs:
            source_name = graph.normalize_module_name(source.get("name"))
            accepted_names = {source_name, *graph._exporter_wrapper_aliases(source_name)}
            candidate_rows = [
                row for row in candidates
                if row.get("op_type") == "Conv"
                and graph.normalize_module_name(row.get("name")) in accepted_names
            ]
            matched_rows = [
                row for row in branch_audit.get("matched_convolutions", [])
                if row.get("source_normalized_name") == source_name
            ]
            rows.append({
                "source_module": source.get("name"),
                "source_module_type": source.get("module_type"),
                "source_normalized_name": source_name,
                "accepted_export_normalized_names": sorted(accepted_names),
                "candidate_export_nodes": candidate_rows,
                "matched_records": matched_rows,
                "status": "matched" if len(matched_rows) == 1 else "unresolved",
            })
        result[branch] = rows
    return result


def _relevant_initializers(graph_doc: dict[str, Any], post_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    initializers = graph_doc.get("initializers", {})
    names = {
        input_name
        for node in post_nodes
        for input_name in node.get("inputs", [])
        if input_name in initializers
    }
    return {name: initializers[name] for name in sorted(names)}


def _provenance(repo: Path, source_root: Path, model: str) -> dict[str, Any]:
    model_dir = source_root / "models" / model
    onnx_path = model_dir / "model.onnx"
    before = file_evidence(repo, onnx_path)
    record_files = _record_summary(repo, source_root, model_dir)
    if not before["exists"]:
        raise FileNotFoundError(f"Existing ONNX is missing; refusing substitute export: {onnx_path}")
    if not any(
        name in {f"models/{model_dir.name}/model_prepare.json", f"models/{model_dir.name}/failure.json"}
        for name in record_files
    ):
        raise FileNotFoundError(f"Preserved prepare/failure record is missing: {model_dir}")
    return {
        "source_root": relative_path(repo, source_root),
        "model_dir": relative_path(repo, model_dir),
        "onnx": {"before": before},
        "preserved_records": record_files,
    }


def audit_model(
    repo: Path,
    source_root: Path,
    model_label: str,
    accepted_model: dict[str, Any],
    model_output: Path,
) -> dict[str, Any]:
    provenance = _provenance(repo, source_root, model_label)
    onnx_path = source_root / "models" / model_label / "model.onnx"
    graph_doc, schema = graph._load_onnx(onnx_path)
    mapping = graph.audit_graph_mapping(graph_doc, model_label, accepted_model)
    after = file_evidence(repo, onnx_path)
    provenance["onnx"].update({
        "after": after,
        "unchanged_during_audit": provenance["onnx"]["before"] == after,
    })
    head_nodes = _head_nodes(graph_doc, accepted_model)
    post_nodes = _post_merge_nodes(graph_doc, mapping)
    adapter = graph.adapter_contract(model_label, accepted_model["head_output_evidence"])
    return {
        "schema_version": 1,
        "study": AUDIT_STUDY,
        "model": model_label,
        "status": "audit_only_completed",
        "mapping_status": mapping.get("mapping_status"),
        "mapping_errors": mapping.get("errors", []),
        "source_to_export_matches": _source_to_node_matches(accepted_model, mapping),
        "onnx_schema": schema,
        "actual_head_conv_nodes": head_nodes,
        "post_merge_topology": post_nodes,
        "relevant_small_shape_index_constants": _relevant_initializers(graph_doc, post_nodes),
        "mapping": mapping,
        "adapter_limitations": {
            "declared_contract": adapter,
            "numeric_forward_observed": False,
            "producer_forward_called": False,
            "tensorrt_parser_or_build_called": False,
            "calibration_loader_called": False,
            "scored_run_authorized": False,
            "semantic_boundary": "mapping evidence is structural; producer-output equivalence and numeric semantics remain deferred",
            "recipe_boundary": "calibration preprocessing and producer trace remain unresolved",
        },
        "provenance": provenance,
        "audit_flags": {
            "audit_only": True,
            "export_performed": False,
            "build_performed": False,
            "capture_performed": False,
            "scored_run_authorized": False,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def _report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Precision-head preserved graph audit",
        "",
        f"- Study: `{manifest['study']}`",
        f"- Status: `{manifest['status']}`",
        f"- Source root: `{manifest['source_root']}`",
        "- Mode: audit-only; existing ONNX read/check/shape-inference/mapping in memory.",
        "- Exporter, model forward, calibration loader, TensorRT and scored execution were not called.",
        "",
        "## Model results",
        "",
    ]
    for row in manifest["models"]:
        lines.append(
            f"- `{row['model']}`: audit `{row['status']}`, mapping `{row.get('mapping_status', 'not_observed')}`, "
            f"errors `{len(row.get('mapping_errors', []))}`."
        )
    lines += [
        "",
        "## Boundary",
        "",
        "- Unresolved mapping is persisted as evidence; it does not authorize a retry or scored run.",
        "- ONNX/PT/engine/cache binaries are source/server-only and are not diagnostic artifacts for Git.",
        "- A newly computed ONNX hash describes the bytes observed during this audit and does not establish historical export provenance.",
    ]
    return "\n".join(lines) + "\n"


def run_audit(args: argparse.Namespace, repo: Path) -> int:
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "-1",
        "YOLO_AUTOINSTALL": "0",
        "ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS": "1",
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    })
    source_root = graph.repo_path(repo, args.source_root)
    output_root = graph.repo_path(repo, args.out_dir)
    if output_root.exists():
        raise FileExistsError(f"Diagnostic output exists; refusing overwrite: {output_root}")

    selected = list(MODEL_CHOICES) if args.model == "all" else [args.model]
    if args.model not in (*MODEL_CHOICES, "all"):
        raise ValueError(f"Unsupported model selection: {args.model}")

    # Resolve every source path before creating the output tree.  A missing
    # binary stops the diagnostic; it must never trigger an export substitute.
    source_provenance = {model: _provenance(repo, source_root, model) for model in selected}
    readiness_root = graph.repo_path(repo, args.readiness_root)
    accepted = graph.validate_readiness_artifact(repo, readiness_root)
    binding = graph.validate_config_binding(repo, accepted)
    accepted_models = {row["label"]: row for row in accepted["manifest"]["model_contracts"]}
    missing_contracts = [model for model in selected if model not in accepted_models]
    if missing_contracts:
        raise ValueError(f"Accepted readiness has no model contract: {missing_contracts}")

    code_files = {
        "diagnostic_script": file_evidence(repo, Path(__file__)),
        "graph_script": file_evidence(repo, repo / "scripts/prepare_precision_head_confirmation_graph.py"),
        "readiness_helper": file_evidence(repo, repo / "scripts/prepare_precision_head_confirmation.py"),
        "config": file_evidence(repo, Path(binding["path"])),
    }
    plan = {
        "schema_version": 1,
        "study": AUDIT_STUDY,
        "status": "audit_only_dispatching",
        "source_root": relative_path(repo, source_root),
        "output_root": relative_path(repo, output_root),
        "selected_models": selected,
        "source_provenance_resolved": source_provenance,
        "readiness": {
            "root": accepted["root"],
            "accepted_git_commit": accepted["accepted_git_commit"],
            "accepted_execution_commit": accepted["accepted_execution_commit"],
            "files": accepted["files"],
        },
        "config_binding": {key: value for key, value in binding.items() if key != "config"},
        "code_provenance": code_files,
        "environment": {
            "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "onnx_runtime_mode": "CPU/CUDA-hidden; no model runtime import",
        },
        "audit_flags": {
            "audit_only": True,
            "export_performed": False,
            "build_performed": False,
            "capture_performed": False,
            "scored_run_authorized": False,
        },
        "no_overwrite": True,
    }
    output_root.mkdir(parents=True)
    write_json_no_overwrite(output_root / "audit_plan.json", plan)

    model_rows: list[dict[str, Any]] = []
    fatal = False
    for model_label in selected:
        model_output = output_root / "models" / model_label
        model_output.mkdir(parents=True)
        try:
            row = audit_model(repo, source_root, model_label, accepted_models[model_label], model_output)
        except Exception as exc:
            fatal = True
            row = {
                "schema_version": 1,
                "study": AUDIT_STUDY,
                "model": model_label,
                "status": "audit_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "provenance": source_provenance[model_label],
                "audit_flags": {
                    "audit_only": True,
                    "export_performed": False,
                    "build_performed": False,
                    "capture_performed": False,
                    "scored_run_authorized": False,
                },
            }
        write_json_no_overwrite(model_output / "graph_audit.json", row)
        model_rows.append(row)

    manifest = {
        "schema_version": 1,
        "study": AUDIT_STUDY,
        "status": "audit_only_completed_with_failures" if fatal else "audit_only_completed",
        "source_root": relative_path(repo, source_root),
        "output_root": relative_path(repo, output_root),
        "selected_models": selected,
        "models": model_rows,
        "readiness": plan["readiness"],
        "config_binding": plan["config_binding"],
        "code_provenance": code_files,
        "environment": plan["environment"],
        "audit_flags": plan["audit_flags"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_no_overwrite(output_root / "graph_audit_manifest.json", manifest)
    write_text_no_overwrite(output_root / "report.md", _report(manifest))
    print(f"DONE: {output_root / 'graph_audit_manifest.json'}", flush=True)
    return 1 if fatal else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT_DEFAULT)
    parser.add_argument("--readiness-root", type=Path, default=graph.DEFAULT_READINESS_ROOT)
    parser.add_argument("--model", choices=(*MODEL_CHOICES, "all"), default="all")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT_DEFAULT)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    return run_audit(args, repo)


if __name__ == "__main__":
    raise SystemExit(main())
