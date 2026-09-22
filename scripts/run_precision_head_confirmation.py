#!/usr/bin/env python3
"""CPU parent/orchestration contract for the two-model confirmation study.

The parent is intentionally safe to run without CUDA/TensorRT.  TensorRT
build/capture work is dispatched only by a reviewed server child after the
integrated Astra gate.  This module therefore validates accounting, input
identity and graph ownership before it can create any scored output root.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from precision_head_confirmation_contract import (
    ARMS,
    MODELS,
    SELECTIONS,
    ContractError,
    build_schedule,
    canonical_json_sha256,
    mapping_targets,
    selected_targets,
    sha256_file,
    validate_schedule,
)

SCHEMA_VERSION = 1
CONTRACT_PATH = Path("configs/precision_head_confirmation_execution_v1.json")
READINESS_PATH = Path("results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2/readiness_manifest.json")
GRAPH_ROOT = Path("results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4")
PREP_ROOT = Path("results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def relative(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def git_head(repo: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def validate_files(repo: Path, contract: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for model, spec in contract["models"].items():
        if model not in MODELS:
            raise ContractError(f"unknown model in execution contract: {model}")
        for key in ("checkpoint", "onnx"):
            path = repo / spec[key]
            if not path.is_file():
                raise FileNotFoundError(f"missing {model} {key}: {path}")
            actual = sha256_file(path)
            if actual != spec[f"{key}_sha256"]:
                raise ContractError(f"{model} {key} hash differs: {actual}")
            evidence.setdefault(model, {})[key] = {"path": spec[key], "sha256": actual, "bytes": path.stat().st_size}
        graph_path = repo / GRAPH_ROOT / "models" / model / "graph_audit.json"
        graph = read_json(graph_path)
        targets = mapping_targets(graph, model)
        expected_hash = spec["mapping_sha256"]
        if targets["mapping_hash"] != expected_hash:
            raise ContractError(f"{model} mapping hash differs")
        evidence[model]["mapping"] = {"path": relative(repo, graph_path), "sha256": expected_hash, "targets": targets}
        evidence[model]["output_shape"] = spec["output_shape"]
        evidence[model]["postprocess"] = spec["postprocess"]
        evidence[model]["head_index"] = spec["head_index"]
        evidence[model]["active_branches"] = spec["active_branches"]
    return evidence


def validate_xml_binding(repo: Path, xml_path: Path, expected_images: int, expected_instances: int) -> dict[str, Any]:
    xml_path = xml_path.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"missing canonical XML archive: {xml_path}")
    dev_dir = repo / "data/processed/cctsdb2021_clean/dev/images"
    image_names = sorted(path.name for path in dev_dir.iterdir() if path.is_file()) if dev_dir.is_dir() else []
    if len(image_names) != expected_images:
        raise ContractError(f"dev image inventory is not {expected_images}: {len(image_names)}")
    from audit_cctsdb_measurement import load_xml
    xml = load_xml(xml_path, {name: {} for name in image_names})
    instances = sum(len(value["rows"]) for value in xml.values())
    if len(xml) != expected_images or instances != expected_instances:
        raise ContractError(f"XML binding is not exactly {expected_images}/{expected_instances}: {len(xml)}/{instances}")
    return {"path": str(xml_path), "sha256": sha256_file(xml_path), "images": len(xml), "instances": instances}


def build_plan(repo: Path, *, out_dir: Path, contract_path: Path = CONTRACT_PATH, xml_path: Path | None = None) -> dict[str, Any]:
    contract = read_json(repo / contract_path)
    producer = __import__("prepare_precision_head_confirmation")
    readiness_config_path = repo / contract["accepted_readiness_config"]["path"]
    readiness_config = read_json(readiness_config_path)
    jobs = build_schedule()
    validate_schedule(jobs)
    evidence = validate_files(repo, contract)
    readiness_path = repo / READINESS_PATH
    if not readiness_path.is_file():
        raise FileNotFoundError(f"missing accepted readiness manifest: {readiness_path}")
    readiness = read_json(readiness_path)
    if readiness.get("status") not in {"ready_for_server_prepare_review", "ready_for_server_run"}:
        raise ContractError(f"readiness status is not accepted: {readiness.get('status')}")
    calibration = producer.calibration_recipe_evidence(repo, readiness_config)
    xml_binding = validate_xml_binding(repo, xml_path, contract["analysis"]["dev_images"], contract["analysis"]["dev_instances"]) if xml_path is not None else None
    provenance_paths = [
        contract_path,
        readiness_path,
        Path("scripts/prepare_precision_head_confirmation.py"),
        Path("scripts/prepare_precision_head_confirmation_graph.py"),
        Path("scripts/capture_cctsdb_validator.py"),
        Path("scripts/verify_cctsdb_capture.py"),
        Path("scripts/analyze_dev_quantization.py"),
    ]
    provenance = [{"path": relative(repo, repo / path), "sha256": sha256_file(repo / path)} for path in provenance_paths]
    canonical_reference = readiness.get("dev_contract", {}).get("canonical_reference")
    if not isinstance(canonical_reference, dict) or len(canonical_reference.get("records", [])) != contract["analysis"]["dev_images"]:
        raise ContractError("accepted readiness manifest does not contain the canonical dev image/shape reference")
    return {
        "schema_version": SCHEMA_VERSION,
        "study": "precision_head_confirmation_v1",
        "status": "implementation_incomplete_remediation_in_progress",
        "repo_head": git_head(repo),
        "contract": {"path": relative(repo, repo / contract_path), "sha256": sha256_file(repo / contract_path)},
        "readiness": {"path": relative(repo, readiness_path), "status": readiness.get("status")},
        "calibration_producer": calibration,
        "graph_audit_root": relative(repo, repo / GRAPH_ROOT),
        "prepare_root": relative(repo, repo / PREP_ROOT),
        "models": evidence,
        "runtime": contract["runtime"],
        "dataset": {"dev_images": contract["analysis"]["dev_images"], "dev_instances": contract["analysis"]["dev_instances"], "xml_path": xml_binding["path"] if xml_binding else "data/raw/CCTSDB2021/xml.zip", "xml_sha256": xml_binding["sha256"] if xml_binding else None, "xml_validation": xml_binding, "canonical_dev_reference": {"commit": canonical_reference.get("commit"), "capture_report_sha256": canonical_reference.get("capture_report_sha256"), "predictions_sha256": canonical_reference.get("predictions_sha256"), "ordered_ids_sha256": canonical_reference.get("ordered_ids_sha256"), "shape_reference_sha256": canonical_reference.get("shape_reference_sha256"), "records": canonical_reference.get("records")}},
        "provenance_files": provenance,
        "schedule": {"sha256": canonical_json_sha256(jobs), "jobs": jobs},
        "accounting": {"auxiliary_cache_builds": 6, "scored_int8_builds": 72, "scored_fp16_builds": 6, "total_builder_invocations": 84, "captures": 78, "dev_image_model_passes": 127608},
        "execution_boundary": {"parent_imports_cuda": False, "parent_imports_tensorrt": False, "server_children_only": True, "go_required": True, "no_resume": True, "no_retry_or_replacement": True},
        "output": {"root": relative(repo, out_dir), "no_overwrite": True, "private_engines_caches_tensors": True, "published": ["manifests", "reports", "jsonl", "logs", "hashes", "public/inspectors/*.json"], "public_inspector_contract": "nonbinary EngineInspector JSON only; no engine/checkpoint/ONNX/cache/raw tensors"},
        "limitations": ["Shared lab GPU telemetry is sampled, not isolation proof.", "The 84-job budget is not evidence until server artifacts complete.", "This plan does not select a best build or promise a positive result."],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_plan(repo: Path, out_dir: Path, *, xml_path: Path | None = None) -> Path:
    if out_dir.exists():
        raise FileExistsError(f"fresh output root required: {out_dir}")
    plan = build_plan(repo, out_dir=out_dir, xml_path=xml_path)
    out_dir.mkdir(parents=True)
    atomic_json(out_dir / "confirmation_plan.json", plan)
    atomic_json(out_dir / "schedule.json", {"schema_version": SCHEMA_VERSION, "jobs": plan["schedule"]["jobs"], "sha256": plan["schedule"]["sha256"]})
    return out_dir / "confirmation_plan.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=Path("results/measurement_audit_v1/server_precision_head_confirmation_v1"))
    parser.add_argument("--phase", choices=("plan", "scored", "analysis"), default="plan")
    parser.add_argument("--go-token", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="0")
    parser.add_argument("--child-timeout", type=int, default=21600)
    parser.add_argument("--confirm-desktop-process", action="append", default=[])
    parser.add_argument("--confirm-background-process", action="append", default=[])
    parser.add_argument("--runtime-double", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--xml", type=Path, help="Canonical read-only XML archive; required for a production plan")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    if args.phase == "plan":
        print(f"PLAN: {write_plan(repo, out_dir, xml_path=args.xml)}")
        return 0
    if args.go_token != "ASTRA_INTEGRATED_GO_REQUIRED":
        raise SystemExit("Refusing server phase: integrated Astra GO is required after local implementation review")
    if args.phase == "scored":
        plan_data = read_json(out_dir / "confirmation_plan.json")
        if not args.runtime_double and not plan_data.get("dataset", {}).get("xml_validation"):
            raise SystemExit("Production scored phase requires a plan bound to --xml with 1636/2706 validation")
        from run_precision_head_confirmation_server import run_parent as run_server_parent
        plan_path = out_dir / "confirmation_plan.json"
        if not plan_path.is_file():
            raise SystemExit(f"Missing preflight plan: {plan_path}")
        server_args = argparse.Namespace(
            repo=repo, plan=plan_path, out_dir=out_dir, device=args.device,
            child_timeout=args.child_timeout, runtime_double=args.runtime_double,
            confirm_desktop_process=args.confirm_desktop_process,
            confirm_background_process=args.confirm_background_process,
        )
        return run_server_parent(server_args)
    raise SystemExit("CPU analysis is a separate post-artifact command; use scripts/analyze_precision_head_confirmation.py")


if __name__ == "__main__":
    raise SystemExit(main())
