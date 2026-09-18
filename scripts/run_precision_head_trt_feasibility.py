#!/usr/bin/env python3
"""Bounded TensorRT FP16 feasibility smoke for the two accepted ONNX graphs.

The parent process is deliberately runtime-free: it validates accepted
readiness/graph/input bindings, records the shared-GPU preflight and dispatches
one child per model.  Only a child imports TensorRT/CUDA, builds an engine in a
private temporary directory and executes the locked eight-image fixture.  The
smoke is descriptive feasibility evidence, not a scored accuracy or timing
matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import prepare_precision_head_confirmation as readiness
import prepare_precision_head_confirmation_graph as graph
import run_precision_head_source_export_dev_bridge as bridge
import verify_precision_head_confirmation_numeric as numeric


STUDY = "precision_head_trt_feasibility_v1"
SCHEMA_VERSION = 1
MODEL_CHOICES = ("yolov8n", "yolo26n")
FIXTURE_NAMES = (
    "train/images/00006.jpg",
    "train/images/00009.jpg",
    "train/images/00028.jpg",
    "train/images/00036.jpg",
    "train/images/00054.jpg",
    "train/images/00061.jpg",
    "train/images/00098.jpg",
    "train/images/00104.jpg",
)
DEFAULT_READINESS_ROOT = Path("results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2")
DEFAULT_GRAPH_AUDIT_ROOT = Path("results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4")
DEFAULT_ONNX_ROOT = Path("results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2")
DEFAULT_OUTPUT = Path("results/measurement_audit_v1") / STUDY
EXPECTED_OUTPUT_SHAPES = numeric.EXPECTED_OUTPUT_SHAPES
EXPECTED_ONNX_SHA256 = numeric.EXPECTED_ONNX_SHA256
EXPECTED_INPUT = {"name": "images", "shape": [1, 3, 640, 640], "dtype": "float32"}
EXPECTED_OUTPUT = {"name": "output0", "dtype": "float32"}
CHILD_ENVIRONMENT = {
    "YOLO_AUTOINSTALL": "0",
    "ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS": "1",
    "PIP_NO_INDEX": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
}
BUILD_SETTINGS = {
    "imgsz": 640,
    "batch": 1,
    "workspace_bytes": 4 << 30,
    "builder_optimization_level": 3,
    "avg_timing_iterations": 1,
    "fp16": True,
    "int8": False,
    "tf32": False,
    "profiling_verbosity": "DETAILED",
    "timing_cache": "fresh_empty_in_memory_per_model; never serialized or reused",
    "precision_overrides": False,
}
CALL_CONTRACT = {
    "models": 2,
    "independent_builds": 2,
    "trt_application_enqueues": 16,
    "onnx_cpu_reference_calls": 16,
    "native_forwards": 0,
    "warmup_calls": 0,
    "retries": 0,
    "calibration_batches": 0,
    "dev_captures": 0,
    "test_captures": 0,
}
RUNTIME_REQUIREMENTS = {
    "torch": "2.5.1+cu121",
    "ultralytics": "8.4.102",
    "tensorrt": "10.16.1.11",
    "numpy": "2.4.4",
    "pycocotools": "2.0.10",
    "cuda": "12.1",
    "device_argument": "0",
    "physical_device_index": 0,
    "logical_cuda_index": 0,
    "gpu_identity": "same UUID/name at preflight, child start and post-run snapshots",
    "gpu_isolation": "not claimed; shared lab desktop is recorded and unknown workloads block/review",
}


class FeasibilityUnresolved(RuntimeError):
    """Raised when feasibility evidence cannot satisfy the hard contract."""


GPU_PROCESS_GUARD_VERSION = "operator-confirmed-nvidia-smi-path-v1"
DESKTOP_ALLOWLIST = {
    "snapd-desktop-integration": "/snap/snapd-desktop-integration/<numeric-revision>/usr/bin/snapd-desktop-integration",
    "xorg": "/usr/lib/xorg/Xorg or /usr/bin/Xorg",
    "gnome-shell": "/usr/bin/gnome-shell or /usr/libexec/gnome-shell",
    "gnome-session": "/usr/bin/gnome-session-binary or /usr/libexec/gnome-session-binary",
}
_DESKTOP_EXECUTABLES = (
    ("snapd-desktop-integration", re.compile(r"^/snap/snapd-desktop-integration/[1-9][0-9]*/usr/bin/snapd-desktop-integration$")),
    ("xorg", re.compile(r"^/(?:usr/lib/xorg/Xorg|usr/bin/Xorg)$")),
    ("gnome-shell", re.compile(r"^/(?:usr/bin|usr/libexec)/gnome-shell$")),
    ("gnome-session", re.compile(r"^/(?:usr/bin|usr/libexec)/gnome-session-binary$")),
)


def desktop_allowlist_id(path: str) -> str | None:
    if not isinstance(path, str):
        return None
    for allowlist_id, pattern in _DESKTOP_EXECUTABLES:
        if pattern.fullmatch(path):
            return allowlist_id
    return None


def parse_desktop_confirmations(values: list[str] | None) -> dict[int, str]:
    confirmations: dict[int, str] = {}
    for value in values or []:
        pid_text, separator, path = value.partition("=")
        pid = int(pid_text) if separator and pid_text.isdigit() else 0
        if not separator or pid <= 0 or not path or pid in confirmations:
            raise ValueError(f"Invalid desktop confirmation {value!r}; expected unique PID=PATH")
        if desktop_allowlist_id(path) is None:
            raise ValueError(f"Desktop confirmation path is outside the allowlist: {path!r}")
        confirmations[pid] = path
    return confirmations


def parse_background_confirmations(values: list[str] | None) -> dict[int, str]:
    confirmations: dict[int, str] = {}
    for value in values or []:
        pid_text, separator, command = value.partition("=")
        pid = int(pid_text) if separator and pid_text.isdigit() else 0
        command = " ".join(command.split())
        if not separator or pid <= 0 or not command or pid in confirmations:
            raise ValueError(f"Invalid background confirmation {value!r}; expected unique PID=COMMAND")
        if any(token in command for token in ("*", "?", "[", "]")):
            raise ValueError("Background workload command must be exact, not a pattern")
        confirmations[pid] = command
    return confirmations


def inspect_background_processes(confirmations: dict[int, str]) -> dict[int, dict[str, Any]]:
    checks: dict[int, dict[str, Any]] = {}
    for pid, expected in sorted(confirmations.items()):
        try:
            result = subprocess.run(["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True, timeout=15, check=False)
        except OSError as exc:
            checks[pid] = {"status": "unverifiable", "expected_command": expected, "observed_command": None, "error": str(exc)}
            continue
        observed = " ".join((result.stdout or "").split())
        status = "running" if observed == expected else ("exited" if not observed and result.returncode in (0, 1) else "command_mismatch")
        checks[pid] = {"status": status, "expected_command": expected, "observed_command": observed or None}
    return checks


def classify_gpu_processes(raw: str, confirmed_desktop: dict[int, str], confirmed_background: dict[int, str], background_checks: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    if raw.strip() in ("", "No running processes found"):
        return []
    details = []
    for row in raw.splitlines():
        fields = [item.strip() for item in row.split(",", 2)]
        if len(fields) != 3 or not fields[0].isdigit():
            details.append({"pid": None, "raw": row, "allowed": False, "classification": "blocked_unverifiable_process", "reason": "cannot parse nvidia-smi process row"})
            continue
        pid = int(fields[0])
        process_name, used_gpu_memory = fields[1:]
        detail: dict[str, Any] = {"pid": pid, "process_name": process_name, "used_gpu_memory": used_gpu_memory, "raw": row, "resolved_executable": process_name if process_name.startswith("/") else None, "allowed": False, "classification": None}
        allowlist_id = desktop_allowlist_id(process_name)
        if pid == os.getpid():
            detail.update({"allowed": True, "classification": "current_runner_process", "reason": "current child process"})
        elif pid in confirmed_background and background_checks.get(pid, {}).get("status") == "running":
            detail.update({"allowed": True, "classification": "allowed_background_operator_confirmed", "operator_confirmed": True, "expected_command": confirmed_background[pid], "reason": "exact current ps command and nvidia-smi PID were explicitly confirmed"})
        elif allowlist_id is not None and confirmed_desktop.get(pid) == process_name:
            detail.update({"allowed": True, "classification": "allowed_desktop_operator_confirmed", "operator_confirmed": True, "allowlist_exception": allowlist_id, "desktop_allowlist_id": allowlist_id, "reason": "exact current nvidia-smi PID/path was explicitly confirmed; /proc was not read"})
        elif allowlist_id is not None:
            detail.update({"classification": "blocked_desktop_unconfirmed", "desktop_allowlist_id": allowlist_id, "reason": "desktop path lacks explicit current operator confirmation"})
        elif process_name in ("", "[Not Found]"):
            detail.update({"classification": "blocked_unverifiable_process", "reason": "nvidia-smi did not provide a usable process path"})
        else:
            detail.update({"classification": "blocked_non_allowlisted_process", "reason": "reported path is not in the narrow desktop allowlist"})
        details.append(detail)
    return details


def gpu_snapshot(confirmed_desktop: dict[int, str], confirmed_background: dict[int, str]) -> dict[str, Any]:
    device_cmd = ["nvidia-smi", "--query-gpu=uuid,name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used", "--format=csv,noheader"]
    process_cmd = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"]
    device = subprocess.run(device_cmd, capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    processes = subprocess.run(process_cmd, capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    background_checks = inspect_background_processes(confirmed_background)
    process_details = classify_gpu_processes(processes, confirmed_desktop, confirmed_background, background_checks)
    observed_pids = {item.get("pid") for item in process_details if isinstance(item.get("pid"), int)}
    unmatched = [{"pid": pid, "reported_path": path} for pid, path in sorted(confirmed_desktop.items()) if pid not in observed_pids]
    blocked = [item for item in process_details if item.get("allowed") is not True]
    background_workload = []
    for pid, check in sorted(background_checks.items()):
        on_gpu = pid in observed_pids
        status = check["status"] if on_gpu else ("present_not_observed_on_gpu" if check["status"] == "running" else check["status"])
        blocking = check["status"] in ("command_mismatch", "unverifiable")
        background_workload.append({"pid": pid, "expected_command": check["expected_command"], "observed_command": check.get("observed_command"), "status": status, "observed_on_gpu": on_gpu, "authorized": status == "running" and on_gpu, "blocking": blocking, "verification_method": "ps -o args= PID plus exact PID in nvidia-smi snapshot; /proc/<pid>/exe was not read"})
    return {"device": device, "processes": processes, "process_details": process_details, "process_guard": {"version": GPU_PROCESS_GUARD_VERSION, "allowlist": DESKTOP_ALLOWLIST, "verification_method": "exact PID/path comparison against current nvidia-smi plus explicit operator confirmation; /proc was not read", "blocked_processes": blocked, "unmatched_confirmations": unmatched, "background_workload": background_workload, "external_workload_detected": bool(blocked or unmatched or any(item["authorized"] for item in background_workload)), "telemetry_status": "complete"}}


def ensure_gpu_guard(snapshot: dict[str, Any]) -> None:
    guard = snapshot.get("process_guard", {})
    blocked = list(guard.get("blocked_processes", [])) + list(guard.get("unmatched_confirmations", [])) + [item for item in guard.get("background_workload", []) if item.get("blocking")]
    if blocked:
        raise FeasibilityUnresolved(f"Other CUDA processes active or unverifiable; no process was changed: {blocked}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json_no_overwrite(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_json_overwrite(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_text_no_overwrite(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def write_jsonl_line(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()


def repo_path(repo: Path, value: Path) -> Path:
    return numeric.repo_path(repo, value)


def relative_to_repo(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def file_evidence(repo: Path, path: Path) -> dict[str, Any]:
    path = path.resolve()
    result: dict[str, Any] = {
        "path": relative_to_repo(repo, path) if path.is_relative_to(repo.resolve()) else str(path),
        "exists": path.is_file(),
        "regular_file": path.is_file() and not path.is_symlink(),
    }
    if path.is_file():
        result.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return result


def git_head(repo: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False)
    if result.returncode:
        raise FeasibilityUnresolved("Cannot resolve current Git HEAD")
    return result.stdout.strip()


def call_contract(model_count: int = 2, fixture_count: int = 8) -> dict[str, Any]:
    if model_count != 2 or fixture_count != 8:
        raise ValueError("The feasibility smoke is locked to two models and eight fixtures")
    return dict(CALL_CONTRACT)


def selected_models(value: str) -> list[str]:
    if value != "all":
        raise ValueError("This bounded smoke requires --model all; single-model execution would change the reviewed call contract")
    return list(MODEL_CHOICES)


def validate_device(device: int) -> None:
    if device != 0:
        raise ValueError("This reviewed smoke binds physical GPU 0 to logical CUDA device 0; other device indices are not authorized")


def application_contract(model: str) -> dict[str, Any]:
    route = bridge.application_route(model)
    return {
        "model": model,
        "input": dict(EXPECTED_INPUT),
        "output": {**EXPECTED_OUTPUT, "shape": list(EXPECTED_OUTPUT_SHAPES[model])},
        "route": route,
        "comparison": {
            "primary": "raw output array; descriptive only",
            "v8": "decoded xywh plus probabilities; installed application NMS/scale_boxes route",
            "v26": "fixed [1,300,6] row order; installed end2end=True filtering route; no second NMS or sorting",
        },
    }


def parse_gpu_identity(device_csv: str, device: int = 0) -> dict[str, Any]:
    rows = [line.strip() for line in (device_csv or "").splitlines() if line.strip()]
    if device < 0 or device >= len(rows):
        raise FeasibilityUnresolved(f"nvidia-smi returned no row for device {device}: {rows!r}")
    fields = [item.strip() for item in rows[device].split(",")]
    if len(fields) != 9:
        raise FeasibilityUnresolved(f"Cannot parse GPU identity row: {rows[device]!r}")
    names = ("uuid", "name", "driver_version", "pstate", "temperature_gpu", "power_draw", "clocks_sm", "clocks_mem", "memory_used")
    result = dict(zip(names, fields))
    if not result["uuid"] or not result["name"]:
        raise FeasibilityUnresolved(f"GPU UUID/name is missing: {result!r}")
    result["device_index"] = device
    return result


def snapshot_gpu(confirm_desktop: dict[int, str], confirm_background: dict[int, str], device: int) -> dict[str, Any]:
    snapshot = gpu_snapshot(confirm_desktop, confirm_background)
    snapshot["gpu_identity"] = parse_gpu_identity(snapshot.get("device", ""), device)
    return snapshot


def snapshot_violations(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    guard = snapshot.get("process_guard", {})
    violations = list(guard.get("blocked_processes", []))
    violations.extend(guard.get("unmatched_confirmations", []))
    violations.extend(item for item in guard.get("background_workload", []) if item.get("blocking"))
    return violations


def validate_same_gpu(expected: dict[str, Any], observed: dict[str, Any], phase: str) -> None:
    for key in ("uuid", "name"):
        if observed.get(key) != expected.get(key):
            raise FeasibilityUnresolved(f"GPU {key} differs at {phase}: expected={expected.get(key)!r}, observed={observed.get(key)!r}")


def _expected_fixture_plan(accepted: dict[str, Any]) -> dict[str, Any]:
    fixture = numeric.build_fixture_plan(accepted)
    actual = tuple(row.get("image") for row in fixture.get("forward_images", []))
    if actual != FIXTURE_NAMES:
        raise FeasibilityUnresolved(f"U42 fixture order differs from A2L-041: {actual!r}")
    if fixture.get("forward_image_count") != len(FIXTURE_NAMES):
        raise FeasibilityUnresolved("U42 fixture count is not eight")
    return fixture


def model_plan(repo: Path, config: dict[str, Any], accepted: dict[str, Any], graph_audit: dict[str, Any], model: str, onnx_root: Path) -> dict[str, Any]:
    checkpoint = numeric.validate_checkpoint_contract(repo, config, accepted, model)
    graph_doc = graph_audit["models"][model]
    onnx_path = repo_path(repo, onnx_root / "models" / model / "model.onnx")
    onnx = file_evidence(repo, onnx_path)
    if onnx.get("sha256") != EXPECTED_ONNX_SHA256[model]:
        raise FeasibilityUnresolved(f"Accepted ONNX hash differs for {model}: {onnx.get('sha256')}")
    graph_onnx = graph_doc.get("provenance", {}).get("onnx", {})
    if graph_onnx.get("before", {}).get("sha256") != EXPECTED_ONNX_SHA256[model] or graph_onnx.get("after", {}).get("sha256") != EXPECTED_ONNX_SHA256[model] or graph_onnx.get("unchanged_during_audit") is not True:
        raise FeasibilityUnresolved(f"Graph audit does not bind unchanged ONNX bytes for {model}")
    accepted_contract = numeric.accepted_model_contracts(accepted, [model])[model]
    return {
        "model": model,
        "checkpoint": checkpoint,
        "onnx": {"path": relative_to_repo(repo, onnx_path), "expected_sha256": EXPECTED_ONNX_SHA256[model], "before": onnx},
        "accepted_contract": accepted_contract,
        "graph_contract": {"mapping_status": graph_doc.get("mapping_status"), "output_shape": list(EXPECTED_OUTPUT_SHAPES[model])},
        "application_contract": application_contract(model),
    }


def build_plan(repo: Path, args: argparse.Namespace, models: list[str], gpu_preflight: dict[str, Any]) -> dict[str, Any]:
    readiness_root = repo_path(repo, args.readiness_root)
    graph_root = repo_path(repo, args.graph_audit_root)
    onnx_root = repo_path(repo, args.onnx_root)
    accepted = graph.validate_readiness_artifact(repo, readiness_root)
    config_binding = graph.validate_config_binding(repo, accepted)
    config_path = Path(config_binding["path"]).resolve()
    config = readiness.load_config(config_path, repo)
    fixture = _expected_fixture_plan(accepted)
    fixture_bindings = {}
    for row in fixture["forward_images"]:
        fixture_bindings[row["image"]] = numeric.verify_bound_image(repo, row)
    graph_audit = numeric.validate_graph_audit_artifact(repo, graph_root, models)
    model_rows = {model: model_plan(repo, config, accepted, graph_audit, model, onnx_root) for model in models}
    return {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY,
        "status": "local_preparation_only; actual_smoke_pending_review",
        "repo_head": git_head(repo),
        "output_root": relative_to_repo(repo, repo_path(repo, args.out_dir)),
        "config": {"path": relative_to_repo(repo, config_path), "file": file_evidence(repo, config_path), "binding": config_binding},
        "readiness": {"root": relative_to_repo(repo, readiness_root), "accepted_git_commit": accepted["accepted_git_commit"], "accepted_execution_commit": accepted["accepted_execution_commit"], "files": accepted["files"]},
        "graph_audit": {"root": relative_to_repo(repo, graph_root), "commit": graph_audit["commit"], "files": graph_audit["files"]},
        "models": model_rows,
        "fixture": fixture,
        "fixture_bindings_preflight": fixture_bindings,
        "fixture_contract": {"selection": "U42", "images": list(FIXTURE_NAMES), "count": 8, "source_and_materialized_hashes_required": True},
        "build_settings": dict(BUILD_SETTINGS),
        "runtime_requirements": dict(RUNTIME_REQUIREMENTS),
        "call_contract": call_contract(),
        "gpu_preflight": gpu_preflight,
        "process_guard_contract": {"version": GPU_PROCESS_GUARD_VERSION, "desktop_allowlist": DESKTOP_ALLOWLIST, "unknown_process_policy": "block before dispatch; if newly observed during/after, preserve output and mark review", "no_kill_pause_permission_clock_or_power_changes": True},
        "numerical_policy": {"assessment": "descriptive_only", "hard_checks": ["shape", "dtype", "finite", "input identity"], "raw_output_tolerance": None, "reason": "no justified application-equivalence margin was pre-registered; no post-hoc threshold", "v26": "fixed row index; no rematching/sorting/NMS to manufacture agreement"},
        "audit_flags": {"export_performed": False, "onnx_modified": False, "tensorrt_imported": False, "tensorrt_build_performed": False, "gpu_used": False, "calibration_loader_called": False, "official_test_accessed": False, "training_performed": False, "matrix_opened": False, "scored_run_authorized": False},
        "historical_numeric_verdict": "preserved_fail_not_replaced",
        "no_overwrite": True,
        "no_resume": True,
    }


def _shape(value: Any) -> list[int]:
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise FeasibilityUnresolved(f"Tensor shape is not concrete: {value!r}") from exc


def dtype_name(value: Any) -> str:
    text = str(value).lower()
    if "float32" in text or text.endswith("float") or text == "dataType.float" or text == "datatype.float":
        return "float32"
    if "float16" in text or "half" in text:
        return "float16"
    if "int8" in text:
        return "int8"
    return text


def validate_io_contract(model: str, contract: dict[str, Any], phase: str) -> dict[str, Any]:
    expected_shape = list(EXPECTED_OUTPUT_SHAPES[model])
    expected = {"input": {**EXPECTED_INPUT}, "output": {**EXPECTED_OUTPUT, "shape": expected_shape}}
    for role in ("input", "output"):
        observed = contract.get(role, {})
        wanted = expected[role]
        for key, value in wanted.items():
            if observed.get(key) != value:
                raise FeasibilityUnresolved(f"{phase} {role} contract differs for {model}: {key} expected={value!r}, observed={observed.get(key)!r}")
    return {"phase": phase, "expected": expected, "observed": contract}


def validate_ort_contract(contract: dict[str, Any], model: str) -> None:
    if contract.get("providers_observed") != ["CPUExecutionProvider"]:
        raise FeasibilityUnresolved(f"ORT reference provider is not CPU-only: {contract.get('providers_observed')!r}")
    input_schema = contract.get("input", {})
    expected_input_schema = {"name": "images", "type": "tensor(float)", "shape": list(EXPECTED_INPUT["shape"])}
    if input_schema != expected_input_schema:
        raise FeasibilityUnresolved(f"ORT input schema differs: expected={expected_input_schema!r}, observed={input_schema!r}")
    output_schema = contract.get("output", {})
    expected_output_schema = {"name": "output0", "type": "tensor(float)", "shape": list(EXPECTED_OUTPUT_SHAPES[model])}
    if output_schema.get("name") != expected_output_schema["name"] or output_schema.get("type") != expected_output_schema["type"]:
        raise FeasibilityUnresolved(f"ORT output schema name/type differs: expected={expected_output_schema!r}, observed={output_schema!r}")
    if output_schema.get("shape") != expected_output_schema["shape"]:
        raise FeasibilityUnresolved(f"ORT output shape differs: expected={expected_output_schema!r}, observed={output_schema!r}")


def _tensor_contract(tensor: Any) -> dict[str, Any]:
    return {"name": str(tensor.name), "shape": _shape(tensor.shape), "dtype": dtype_name(tensor.dtype)}


def network_io_contract(network: Any) -> dict[str, Any]:
    inputs = [_tensor_contract(network.get_input(index)) for index in range(int(network.num_inputs))]
    outputs = [_tensor_contract(network.get_output(index)) for index in range(int(network.num_outputs))]
    if len(inputs) != 1 or len(outputs) != 1:
        raise FeasibilityUnresolved(f"TensorRT parser produced unexpected IO counts: inputs={inputs}, outputs={outputs}")
    return {"input": inputs[0], "output": outputs[0]}


def engine_io_contract(engine: Any) -> dict[str, Any]:
    names = [str(engine.get_tensor_name(index)) for index in range(int(engine.num_io_tensors))]
    inputs = [name for name in names if str(engine.get_tensor_mode(name)).lower().endswith("input")]
    outputs = [name for name in names if str(engine.get_tensor_mode(name)).lower().endswith("output")]
    if inputs != [EXPECTED_INPUT["name"]] or outputs != [EXPECTED_OUTPUT["name"]]:
        raise FeasibilityUnresolved(f"TensorRT engine IO names/modes differ: names={names}, inputs={inputs}, outputs={outputs}")
    return {
        "input": {"name": inputs[0], "shape": _shape(engine.get_tensor_shape(inputs[0])), "dtype": dtype_name(engine.get_tensor_dtype(inputs[0]))},
        "output": {"name": outputs[0], "shape": _shape(engine.get_tensor_shape(outputs[0])), "dtype": dtype_name(engine.get_tensor_dtype(outputs[0]))},
    }


def _parser_errors(parser: Any) -> list[str]:
    errors = []
    for index in range(int(getattr(parser, "num_errors", 0))):
        errors.append(str(parser.get_error(index)))
    return errors


def build_engine(trt: Any, onnx_path: Path, model: str, state: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """Build one private engine with the locked flags; never writes the engine."""
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if state is not None:
        state["parser_attempted"] = True
        _persist_state(state)
    parsed = bool(parser.parse_from_file(str(onnx_path)))
    errors = _parser_errors(parser)
    if not parsed:
        raise FeasibilityUnresolved(f"TensorRT parser failed for {model}: {errors}")
    parser_contract = network_io_contract(network)
    parser_binding = validate_io_contract(model, parser_contract, "parser")
    if state is not None:
        state["parser_completed"] = True
        _persist_state(state)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, BUILD_SETTINGS["workspace_bytes"])
    config.builder_optimization_level = BUILD_SETTINGS["builder_optimization_level"]
    config.avg_timing_iterations = BUILD_SETTINGS["avg_timing_iterations"]
    config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.INT8)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    timing_cache = config.create_timing_cache(b"")
    config.set_timing_cache(timing_cache, False)
    flag_observed = {name: bool(config.get_flag(flag)) for name, flag in (("FP16", trt.BuilderFlag.FP16), ("INT8", trt.BuilderFlag.INT8), ("TF32", trt.BuilderFlag.TF32))}
    if flag_observed != {"FP16": True, "INT8": False, "TF32": False}:
        raise FeasibilityUnresolved(f"TensorRT builder flags were not observed as locked: {flag_observed}")
    if state is not None:
        state["build_attempted"] = True
        _persist_state(state)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise FeasibilityUnresolved(f"TensorRT build returned no serialized engine for {model}")
    serialized_bytes = bytes(serialized)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise FeasibilityUnresolved(f"TensorRT runtime could not deserialize engine for {model}")
    engine_contract = engine_io_contract(engine)
    engine_binding = validate_io_contract(model, engine_contract, "engine")
    inspector = engine.create_engine_inspector()
    inspector_json = None
    try:
        inspector_json = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    except Exception as exc:  # inspector availability is recorded, not silently treated as layer precision proof
        inspector_json = {"unavailable": type(exc).__name__ + ": " + str(exc)}
    if state is not None:
        state["build_completed"] = True
        _persist_state(state)
    return engine, {
        "parser": {"parse_returned_true": parsed, "errors": errors, "io": parser_binding},
        "builder": {"settings": dict(BUILD_SETTINGS), "flags_observed": flag_observed, "timing_cache": "created from empty bytes and not serialized/reused", "serialized_engine": {"bytes": len(serialized_bytes), "sha256": sha256_bytes(serialized_bytes), "published": False}},
        "engine": {"io": engine_binding, "inspector_format": "JSON", "inspector": inspector_json, "fp16_all_layers_not_inferred": True},
        "ownership": {"logger": "child-local; released at child termination", "builder": "child-local; build completed before execution", "parser": "child-local; parse evidence retained", "config": "child-local; timing cache not serialized", "runtime": "child-local; engine deserialized before enqueue", "engine": "child-local; no engine file published", "context": "child-local context per enqueue; synchronize completed before output copy"},
    }


def _finite_output(np: Any, value: Any, model: str, source: str) -> Any:
    array = np.ascontiguousarray(value)
    expected = list(EXPECTED_OUTPUT_SHAPES[model])
    if list(array.shape) != expected or array.dtype != np.dtype("float32"):
        raise FeasibilityUnresolved(f"{source} output contract differs for {model}: shape={list(array.shape)}, dtype={array.dtype}")
    if not bool(np.isfinite(array).all()):
        raise FeasibilityUnresolved(f"{source} output is non-finite for {model}")
    return array


def descriptive_primary_comparison(np: Any, reference: Any, observed: Any, model: str) -> dict[str, Any]:
    reference = _finite_output(np, reference, model, "ORT reference")
    observed = _finite_output(np, observed, model, "TensorRT")
    delta = np.abs(observed.astype(np.float64) - reference.astype(np.float64))
    return {"assessment": "descriptive_only", "shape": list(reference.shape), "dtype": "float32", "exact_equal": bool(np.array_equal(observed, reference)), "max_abs": float(delta.max()), "mean_abs": float(delta.mean()), "tolerance": None, "v26_row_semantics": "fixed row index; no rematching/sorting/NMS" if model == "yolo26n" else None}


def _component_comparison(np: Any, reference: Any, observed: Any, label: str) -> dict[str, Any]:
    reference = np.ascontiguousarray(reference)
    observed = np.ascontiguousarray(observed)
    if reference.shape != observed.shape:
        raise FeasibilityUnresolved(f"Raw {label} shapes differ: {reference.shape} != {observed.shape}")
    delta = np.abs(observed.astype(np.float64) - reference.astype(np.float64))
    return {"label": label, "shape": list(reference.shape), "dtype": str(reference.dtype), "exact_equal": bool(np.array_equal(observed, reference)), "max_abs": float(delta.max()), "mean_abs": float(delta.mean()), "tolerance": None}


def descriptive_raw_semantics(np: Any, reference: Any, observed: Any, model: str) -> dict[str, Any]:
    """Compare immutable raw units separately before any application consumer."""
    reference = _finite_output(np, reference, model, "ORT reference")
    observed = _finite_output(np, observed, model, "TensorRT")
    result: dict[str, Any] = {"assessment": "descriptive_only", "reference_raw": {"shape": list(reference.shape), "dtype": str(reference.dtype), "sha256": numeric.array_digest(reference)}, "observed_raw": {"shape": list(observed.shape), "dtype": str(observed.dtype), "sha256": numeric.array_digest(observed)}, "tolerance": None}
    if model == "yolov8n":
        result["components"] = {"boxes_xywh": _component_comparison(np, reference[:, 0:4, :], observed[:, 0:4, :], "boxes_xywh"), "class_probabilities": _component_comparison(np, reference[:, 4:7, :], observed[:, 4:7, :], "class_probabilities")}
    else:
        result["components"] = {"boxes_xyxy": _component_comparison(np, reference[:, :, 0:4], observed[:, :, 0:4], "boxes_xyxy"), "confidence": _component_comparison(np, reference[:, :, 4], observed[:, :, 4], "confidence"), "class_ids": {"label": "class_ids", "shape": list(reference[:, :, 5].shape), "exact_equal": bool(np.array_equal(reference[:, :, 5], observed[:, :, 5])), "mismatch_count": int((reference[:, :, 5] != observed[:, :, 5]).sum()) if hasattr((reference[:, :, 5] != observed[:, :, 5]), "sum") else None, "rule": "fixed row index exact integer comparison; no rematching/sorting/NMS"}}
    return result


def execute_trt_once(engine: Any, runtime: dict[str, Any], input_np: Any, model: str, device: int) -> Any:
    torch = runtime["torch"]
    np = runtime["np"]
    input_tensor = torch.as_tensor(input_np, dtype=torch.float32, device=torch.device(f"cuda:{device}"))
    output_tensor = torch.empty(tuple(EXPECTED_OUTPUT_SHAPES[model]), dtype=torch.float32, device=input_tensor.device)
    context = engine.create_execution_context()
    bind_tensor_addresses(context, model, int(input_tensor.data_ptr()), int(output_tensor.data_ptr()))
    stream = torch.cuda.current_stream(input_tensor.device)
    if not bool(context.execute_async_v3(stream.cuda_stream)):
        raise FeasibilityUnresolved(f"TensorRT execute_async_v3 returned false for {model}")
    torch.cuda.synchronize(input_tensor.device)
    return _finite_output(np, output_tensor.detach().cpu().numpy(), model, "TensorRT")


def bind_tensor_addresses(context: Any, model: str, input_ptr: int, output_ptr: int) -> None:
    """Bind both TensorRT pointers and treat a false return as a hard failure."""
    if not context.set_tensor_address(EXPECTED_INPUT["name"], input_ptr):
        raise FeasibilityUnresolved(f"TensorRT input pointer binding returned false for {model}")
    if not context.set_tensor_address(EXPECTED_OUTPUT["name"], output_ptr):
        raise FeasibilityUnresolved(f"TensorRT output pointer binding returned false for {model}")


def runtime_loader(config: dict[str, Any], device: int) -> dict[str, Any]:
    validate_device(device)
    os.environ.update(CHILD_ENVIRONMENT)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    import importlib.metadata as metadata
    import numpy as np
    import tensorrt as trt
    import torch
    import ultralytics

    runtime = numeric.load_child_runtime(config)
    expected = {key: RUNTIME_REQUIREMENTS[key] for key in ("torch", "ultralytics", "numpy", "pycocotools")}
    observed = dict(runtime["packages"].get("observed", {}))
    observed["tensorrt"] = getattr(trt, "__version__", None)
    observed["cuda"] = getattr(torch.version, "cuda", None)
    observed["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if any(observed.get(key) != value for key, value in expected.items()) or observed.get("tensorrt") != RUNTIME_REQUIREMENTS["tensorrt"] or observed.get("cuda") != RUNTIME_REQUIREMENTS["cuda"]:
        raise FeasibilityUnresolved(f"Pinned runtime mismatch: expected={expected | {'tensorrt': RUNTIME_REQUIREMENTS['tensorrt'], 'cuda': RUNTIME_REQUIREMENTS['cuda']}}, observed={observed}")
    if not torch.cuda.is_available():
        raise FeasibilityUnresolved("CUDA is unavailable in TensorRT child")
    runtime.update({"trt": trt, "np": np, "torch": torch, "packages": {"expected": expected, "observed": observed, "execution": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "mode": "GPU TensorRT child; ORT CPUExecutionProvider only", "physical_device_index": device, "logical_cuda_index": 0}}, "device": device})
    from ultralytics.utils import nms, ops
    runtime.update({"nms": nms, "ops": ops})
    return runtime


def _public_trace(trace: dict[str, Any], row: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = dict(trace)
    result.update({"selection": row["selection"], "manifest_order": row["manifest_order"], "image": row["image"], "expected_source_sha256": row["expected_sha256"], "expected_source_bytes": row["expected_bytes"]})
    return result


def _child_state(model: str) -> dict[str, Any]:
    return {"model": model, "status": "running", "stage": "not_started", "stage_history": [], "forward_counts": {"trt_application_enqueue": {"attempted": 0, "completed": 0}, "onnx_cpu_reference_call": {"attempted": 0, "completed": 0}, "native_forward": {"attempted": 0, "completed": 0}}, "records_written": 0, "parser_attempted": False, "parser_completed": False, "build_attempted": False, "build_completed": False, "dispatch_attempted": 0, "dispatch_completed": 0, "tensorrt_imported": False, "tensorrt_build_performed": False, "gpu_used": False}


def _persist_state(state: dict[str, Any]) -> None:
    path_value = state.get("state_path")
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    public = {key: value for key, value in state.items() if not key.startswith("_")}
    path.write_text(json.dumps(public, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _stage(state: dict[str, Any], stage: str, **details: Any) -> None:
    state["stage"] = stage
    state["stage_history"].append({"sequence": len(state["stage_history"]), "stage": stage, **details})
    _persist_state(state)


def _owned_files(output_root: Path, model: str) -> list[str]:
    model_root = output_root / "models" / model
    if not model_root.is_dir():
        return []
    return sorted(path.relative_to(output_root).as_posix() for path in model_root.rglob("*") if path.is_file())


def child_failure(plan: dict[str, Any], model: str, state: dict[str, Any], exc: Exception, output_root: Path) -> dict[str, Any]:
    state["status"] = "failed"
    _persist_state(state)
    return {"schema_version": SCHEMA_VERSION, "study": STUDY, "model": model, "status": "failed", "validity": "blocked_not_interpretable", "error_type": type(exc).__name__, "error": str(exc), "stage": state["stage"], "stage_history": state["stage_history"], "forward_counts": state["forward_counts"], "records_written": state["records_written"], "partial_files": _owned_files(output_root, model), "partial_files_scope": f"models/{model}/owned_paths_only", "no_silent_resume": True, "no_retry": True, "audit_flags": {"export_performed": False, "onnx_modified": False, "tensorrt_imported": bool(state.get("tensorrt_imported")), "tensorrt_build_performed": bool(state.get("tensorrt_build_performed")), "gpu_used": bool(state.get("gpu_used")), "calibration_loader_called": False, "native_forward": False, "official_test_accessed": False, "training_performed": False, "matrix_opened": False, "scored_run_authorized": False}}


def run_model_child(repo: Path, plan: dict[str, Any], model: str, out_dir: Path, state: dict[str, Any] | None = None, runtime_loader: Callable[..., dict[str, Any]] = runtime_loader, snapshot_fn: Callable[..., dict[str, Any]] = snapshot_gpu) -> dict[str, Any]:
    state = state or _child_state(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    state["state_path"] = str(out_dir / "child_state.json")
    _persist_state(state)
    model_plan = plan["models"][model]
    desktop = {int(row["pid"]): row["reported_path"] for row in plan["gpu_preflight"].get("confirmed_desktop", [])}
    background = {int(row["pid"]): row["expected_command"] for row in plan["gpu_preflight"].get("confirmed_background", [])}
    expected_gpu = plan["gpu_preflight"]["identity"]
    device = int(plan["runtime_requirements"]["device_argument"])
    validate_device(device)
    _stage(state, "child_started")
    telemetry_before = snapshot_fn(desktop, background, device)
    ensure_gpu_guard(telemetry_before)
    validate_same_gpu(expected_gpu, telemetry_before["gpu_identity"], "child_start")
    if snapshot_violations(telemetry_before):
        raise FeasibilityUnresolved(f"Unexpected workload at child start: {snapshot_violations(telemetry_before)}")
    checkpoint_path = repo_path(repo, Path(model_plan["checkpoint"]["expected"]["path"]))
    onnx_path = repo_path(repo, Path(model_plan["onnx"]["path"]))
    checkpoint_before = file_evidence(repo, checkpoint_path)
    onnx_before = file_evidence(repo, onnx_path)
    if checkpoint_before.get("sha256") != model_plan["checkpoint"]["expected"]["sha256"] or onnx_before.get("sha256") != EXPECTED_ONNX_SHA256[model]:
        raise FeasibilityUnresolved(f"Child binary identity differs for {model}")
    fixture_rows = plan["fixture"]["forward_images"]
    for row in fixture_rows:
        numeric.verify_bound_image(repo, row)
    _stage(state, "runtime_loading")
    config = read_json(repo_path(repo, Path(plan["config"]["path"])))
    runtime = runtime_loader(config, device)
    state["tensorrt_imported"] = True
    state["gpu_used"] = True
    _stage(state, "runtime_loaded", packages=runtime["packages"])
    _stage(state, "ort_session_setup")
    session, ort_contract = numeric.run_onnx_session(onnx_path, model, runtime)
    validate_ort_contract(ort_contract, model)
    _stage(state, "ort_session_ready", contract=ort_contract)
    _stage(state, "build")
    with tempfile.TemporaryDirectory(prefix=f"{STUDY}_{model}_") as scratch:
        scratch_path = Path(scratch)
        engine, build_evidence = build_engine(runtime["trt"], onnx_path, model, state=state)
        state["tensorrt_build_performed"] = True
        _persist_state(state)
        _stage(state, "build_complete", engine_sha256=build_evidence["builder"]["serialized_engine"]["sha256"])
        telemetry_during_build = snapshot_fn(desktop, background, device)
        validate_same_gpu(expected_gpu, telemetry_during_build["gpu_identity"], "after_build")
        records_path = out_dir / "smoke_records.jsonl"
        trace_path = out_dir / "preprocess_trace.jsonl"
        with records_path.open("x", encoding="utf-8", newline="\n") as records_handle, trace_path.open("x", encoding="utf-8", newline="\n") as trace_handle:
            for index, row in enumerate(fixture_rows):
                _stage(state, "preprocess", image=row["image"], order=index)
                source_path = numeric.resolve_bound_source_image(repo, row)
                current = file_evidence(repo, source_path)
                if current.get("sha256") != row["expected_sha256"] or current.get("bytes") != row["expected_bytes"]:
                    raise FeasibilityUnresolved(f"Fixture source changed before execution: {row['image']}")
                tensor, trace = numeric.trace_preprocess(source_path, runtime, stride=32)
                trace = _public_trace(trace, row, repo)
                input_np = runtime["np"].ascontiguousarray(tensor.detach().cpu().numpy())
                if list(input_np.shape) != EXPECTED_INPUT["shape"] or input_np.dtype != runtime["np"].dtype("float32") or not bool(runtime["np"].isfinite(input_np).all()):
                    raise FeasibilityUnresolved(f"Preprocessed input contract differs for {row['image']}")
                input_before_consumers = input_np.copy()
                input_private_path = scratch_path / f"input_{index:02d}.bin"
                input_private_path.write_bytes(input_np.tobytes(order="C"))
                write_jsonl_line(trace_handle, trace)
                _stage(state, "execution", image=row["image"], order=index)
                state["forward_counts"]["trt_application_enqueue"]["attempted"] += 1
                state["dispatch_attempted"] += 1
                _persist_state(state)
                trt_output = execute_trt_once(engine, runtime, input_np, model, device)
                state["forward_counts"]["trt_application_enqueue"]["completed"] += 1
                state["dispatch_completed"] += 1
                _persist_state(state)
                trt_raw = runtime["np"].ascontiguousarray(trt_output).copy()
                if not bool(runtime["np"].array_equal(input_before_consumers, input_np)):
                    raise FeasibilityUnresolved(f"Input tensor bytes changed during TensorRT consumer for {row['image']}")
                state["forward_counts"]["onnx_cpu_reference_call"]["attempted"] += 1
                _persist_state(state)
                ort_outputs = session.run(["output0"], {"images": input_np})
                if len(ort_outputs) != 1:
                    raise FeasibilityUnresolved(f"ORT returned {len(ort_outputs)} outputs for {model}")
                state["forward_counts"]["onnx_cpu_reference_call"]["completed"] += 1
                _persist_state(state)
                ort_raw = runtime["np"].ascontiguousarray(_finite_output(runtime["np"], ort_outputs[0], model, "ORT reference")).copy()
                if not bool(runtime["np"].array_equal(input_before_consumers, input_np)):
                    raise FeasibilityUnresolved(f"Input tensor bytes changed during ORT consumer for {row['image']}")
                raw_comparison = descriptive_raw_semantics(runtime["np"], ort_raw, trt_raw, model)
                reference_private_path = scratch_path / f"onnx_reference_{index:02d}.bin"
                reference_private_path.write_bytes(ort_raw.tobytes(order="C"))
                trt_postprocess_input = trt_raw.copy()
                ort_postprocess_input = ort_raw.copy()
                trt_detections = bridge.application_postprocess(model, trt_postprocess_input, trace, runtime)
                ort_detections = bridge.application_postprocess(model, ort_postprocess_input, trace, runtime)
                if not bool(runtime["np"].array_equal(trt_raw, trt_output)) or not bool(runtime["np"].array_equal(ort_raw, ort_outputs[0])):
                    raise FeasibilityUnresolved(f"Postprocess changed an immutable raw output for {row['image']}")
                if sha256_file(reference_private_path) != raw_comparison["reference_raw"]["sha256"]:
                    raise FeasibilityUnresolved(f"Materialized ORT reference hash differs from published raw hash for {row['image']}")
                write_jsonl_line(records_handle, {"schema_version": SCHEMA_VERSION, "study": STUDY, "model": model, "order": index, "image": row["image"], "input": {"source_sha256": row["expected_sha256"], "source_bytes": row["expected_bytes"], "shape": list(input_np.shape), "dtype": str(input_np.dtype), "sha256": numeric.array_digest(input_np)}, "private_materialization": {"scratch_scope": "private temporary directory; deleted after child", "input_bytes": {"bytes": input_private_path.stat().st_size, "sha256": sha256_file(input_private_path)}, "onnx_reference_bytes": {"bytes": reference_private_path.stat().st_size, "sha256": sha256_file(reference_private_path)}, "raw_bytes_published": False}, "tensorrt": {"shape": list(trt_raw.shape), "dtype": str(trt_raw.dtype), "finite": True, "sha256": numeric.array_digest(trt_raw), "raw_units": "immutable pre-postprocess TensorRT output", "detections": trt_detections}, "onnx_cpu_reference": {"shape": list(ort_raw.shape), "dtype": str(ort_raw.dtype), "finite": True, "sha256": numeric.array_digest(ort_raw), "raw_units": "immutable pre-postprocess CPU ORT output", "detections": ort_detections}, "comparison": raw_comparison, "native_forward": False})
                state["records_written"] += 1
                _persist_state(state)
        telemetry_after = snapshot_fn(desktop, background, device)
    validate_same_gpu(expected_gpu, telemetry_after["gpu_identity"], "after_run")
    checkpoint_after = file_evidence(repo, checkpoint_path)
    onnx_after = file_evidence(repo, onnx_path)
    if checkpoint_before != checkpoint_after or onnx_before != onnx_after:
        raise FeasibilityUnresolved(f"Input binary changed during {model}")
    if state["forward_counts"]["trt_application_enqueue"]["completed"] != 8 or state["forward_counts"]["onnx_cpu_reference_call"]["completed"] != 8:
        raise FeasibilityUnresolved(f"Call count differs from eight fixtures for {model}")
    workload_violations = snapshot_violations(telemetry_during_build) + snapshot_violations(telemetry_after)
    state["status"] = "completed"
    _stage(state, "completed")
    return {"schema_version": SCHEMA_VERSION, "study": STUDY, "model": model, "status": "completed", "validity": "review_required_external_workload" if workload_violations else "validity_checks_passed; scientific_assessment_descriptive", "runtime": runtime["packages"], "checkpoint": {"before": checkpoint_before, "after": checkpoint_after, "unchanged": checkpoint_before == checkpoint_after}, "onnx": {"before": onnx_before, "after": onnx_after, "unchanged": onnx_before == onnx_after, "expected_sha256": EXPECTED_ONNX_SHA256[model]}, "build": build_evidence, "records": {"smoke": f"models/{model}/smoke_records.jsonl", "preprocess": f"models/{model}/preprocess_trace.jsonl", "fixtures": 8, "raw_tensors_published": False, "engine_published": False}, "forward_counts": state["forward_counts"], "native_forwards": 0, "warmup": 0, "retries": 0, "calibration_batches": 0, "telemetry": {"before": telemetry_before, "during_build": telemetry_during_build, "after": telemetry_after, "external_workload_violations": workload_violations, "shared_lab_disclaimer": "Desktop/background visibility is not proof of GPU isolation."}, "lifecycle": {"status": state["status"], "stage": state["stage"], "stage_history": state["stage_history"]}, "audit_flags": {"export_performed": False, "onnx_modified": False, "tensorrt_imported": True, "tensorrt_build_performed": True, "gpu_used": True, "calibration_loader_called": False, "official_test_accessed": False, "training_performed": False, "matrix_opened": False, "scored_run_authorized": False}, "created_utc": datetime.now(timezone.utc).isoformat()}


def publishable_inventory(output_root: Path) -> list[str]:
    return sorted(path.relative_to(output_root).as_posix() for path in output_root.rglob("*") if path.is_file() and path.suffix in {".json", ".jsonl", ".md", ".log"})


def _decode_stream(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _write_json_if_absent(path: Path, value: Any) -> None:
    if not path.exists():
        write_json_no_overwrite(path, value)


def _write_text_if_absent(path: Path, value: str) -> None:
    if not path.exists():
        write_text_no_overwrite(path, value)


def _state_from_disk(path: Path, model: str) -> dict[str, Any]:
    if not path.is_file():
        state = _child_state(model)
        state["completion_status"] = "unknown_after_timeout"
        return state
    state = read_json(path)
    state["completion_status"] = "unknown_after_timeout"
    return state


def dispatch_children(repo: Path, plan: dict[str, Any], output_root: Path, models: list[str], timeout_seconds: int, run_fn: Callable[..., Any] = subprocess.run) -> tuple[list[dict[str, Any]], bool]:
    rows = []
    failed = False
    script = Path(__file__).resolve()
    for model in models:
        model_dir = output_root / "models" / model
        model_dir.mkdir(parents=True, exist_ok=True)
        state_path = model_dir / "child_state.json"
        if not state_path.exists():
            initial_state = _child_state(model)
            initial_state["state_path"] = str(state_path)
            _persist_state(initial_state)
        command = [sys.executable, str(script), "--child", "--repo-root", str(repo), "--model", model, "--device", str(plan["runtime_requirements"]["logical_cuda_index"]), "--plan", str(output_root / "smoke_plan.json")]
        try:
            result = run_fn(command, cwd=repo, env={**os.environ, **CHILD_ENVIRONMENT}, capture_output=True, text=True, check=False, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            state = _state_from_disk(state_path, model)
            row = child_failure(plan, model, state, TimeoutError(f"child timeout after {timeout_seconds}s"), output_root)
            _write_json_if_absent(model_dir / "failure.json", row)
            output = _decode_stream(exc.stdout) + _decode_stream(exc.stderr)
            if not output:
                output = "child timed out; owned child termination was not independently confirmed; stopped further GPU dispatch\n"
            result = None
            _write_text_if_absent(output_root / "logs" / f"{model}.log", output)
            rows.append(read_json(model_dir / "failure.json"))
            failed = True
            break
        else:
            output = _decode_stream(result.stdout) + _decode_stream(result.stderr)
        _write_text_if_absent(output_root / "logs" / f"{model}.log", output)
        report_path = model_dir / "model_report.json"
        failure_path = model_dir / "failure.json"
        if report_path.is_file():
            row = read_json(report_path)
        elif failure_path.is_file():
            row = read_json(failure_path)
        else:
            row = child_failure(plan, model, _state_from_disk(state_path, model), RuntimeError("child exited without report"), output_root)
            _write_json_if_absent(failure_path, row)
        if (result is not None and result.returncode != 0) or row.get("status") != "completed":
            failed = True
        rows.append(row)
    return rows, failed


def final_report(manifest: dict[str, Any]) -> str:
    lines = ["# TensorRT FP16 feasibility smoke", "", f"- Study: `{manifest['study']}`", f"- Status: `{manifest['status']}`; validity: `{manifest.get('validity')}`.", "- This is a bounded feasibility smoke, not a scored accuracy result or timing benchmark.", "- Historical numeric/localization `FAIL` verdicts are preserved.", "", "## Model status", ""]
    for row in manifest.get("models", []):
        lines.append(f"- `{row.get('model')}`: `{row.get('status')}`; validity `{row.get('validity', 'blocked')}`; counts `{row.get('forward_counts', {})}`.")
    lines += ["", "## Boundary", "", "- FP16 builder enablement is recorded with parser, binding and inspector evidence; it is not interpreted as every layer executing FP16.", "- No raw tensors or engine binaries are publishable. Numerical comparison is descriptive only; no equivalence threshold is applied."]
    return "\n".join(lines) + "\n"


def run_parent(args: argparse.Namespace, repo: Path) -> int:
    models = selected_models(args.model)
    validate_device(args.device)
    output_root = repo_path(repo, args.out_dir)
    if output_root.exists():
        raise FileExistsError(f"Output exists; refusing overwrite/resume: {output_root}")
    desktop = parse_desktop_confirmations(args.confirm_desktop_process)
    background = parse_background_confirmations(args.confirm_background_process)
    preflight = snapshot_gpu(desktop, background, args.device)
    ensure_gpu_guard(preflight)
    gpu_preflight = {"identity": preflight["gpu_identity"], "snapshot": preflight, "confirmed_desktop": [{"pid": pid, "reported_path": path} for pid, path in sorted(desktop.items())], "confirmed_background": [{"pid": pid, "expected_command": command} for pid, command in sorted(background.items())]}
    plan = build_plan(repo, args, models, gpu_preflight)
    output_root.mkdir(parents=True)
    write_json_no_overwrite(output_root / "smoke_plan.json", plan)
    rows, failed = dispatch_children(repo, plan, output_root, models, args.child_timeout_seconds)
    review_required = any(row.get("validity") == "review_required_external_workload" for row in rows)
    manifest = {"schema_version": SCHEMA_VERSION, "study": STUDY, "status": "failed" if failed else "completed", "validity": "blocked_not_interpretable" if failed else ("review_required_external_workload" if review_required else "validity_checks_passed; scientific_assessment_descriptive"), "execution_status": "failed" if failed else "completed", "models": rows, "plan_path": "smoke_plan.json", "artifact_inventory": [], "terminal_artifact_inventory_complete": False, "call_contract": plan["call_contract"], "build_settings": plan["build_settings"], "audit_flags": {"parent_tensorrt_imported": False, "parent_gpu_used": False, "export_performed": False, "calibration_loader_called": False, "matrix_opened": False, "scored_run_authorized": False}, "gpu_preflight": gpu_preflight, "historical_numeric_verdict": "preserved_fail_not_replaced", "no_retry": True, "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json_no_overwrite(output_root / "smoke_manifest.json", manifest)
    write_text_no_overwrite(output_root / "report.md", final_report(manifest))
    manifest["artifact_inventory"] = publishable_inventory(output_root)
    manifest["terminal_artifact_inventory_complete"] = True
    write_json_overwrite(output_root / "smoke_manifest.json", manifest)
    print(f"DONE: {output_root / 'smoke_manifest.json'}")
    return 1 if failed else 0


def run_child(args: argparse.Namespace) -> int:
    repo = args.repo_root.resolve()
    plan = read_json(repo_path(repo, args.plan))
    model = args.model
    if model not in MODEL_CHOICES:
        raise ValueError(f"Internal child requires one concrete model: {model!r}")
    if model not in plan.get("models", {}):
        raise FeasibilityUnresolved(f"Child model is not bound by the parent plan: {model}")
    validate_device(args.device)
    expected_logical_device = int(plan.get("runtime_requirements", {}).get("logical_cuda_index", -1))
    if args.device != expected_logical_device:
        raise FeasibilityUnresolved(f"Child device differs from plan: expected logical CUDA {expected_logical_device}, observed {args.device}")
    output_root = repo_path(repo, Path(plan["output_root"]))
    out_dir = output_root / "models" / model
    state = _child_state(model)
    try:
        report = run_model_child(repo, plan, model, out_dir, state=state)
        write_json_no_overwrite(out_dir / "model_report.json", report)
        print(f"DONE MODEL {model}: {out_dir / 'model_report.json'}", flush=True)
        return 0
    except Exception as exc:
        failure = child_failure(plan, model, state, exc, output_root)
        if not (out_dir / "failure.json").exists():
            write_json_no_overwrite(out_dir / "failure.json", failure)
        print(f"FAILED MODEL {model}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded two-model TensorRT FP16 feasibility smoke; server-only execution")
    parser.add_argument("--readiness-root", type=Path, default=DEFAULT_READINESS_ROOT)
    parser.add_argument("--graph-audit-root", type=Path, default=DEFAULT_GRAPH_AUDIT_ROOT)
    parser.add_argument("--onnx-root", type=Path, default=DEFAULT_ONNX_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", choices=("all", *MODEL_CHOICES), default="all")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--confirm-desktop-process", action="append", default=[], metavar="PID=PATH")
    parser.add_argument("--confirm-background-process", action="append", default=[], metavar="PID=COMMAND")
    parser.add_argument("--child-timeout-seconds", type=int, default=3600)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device < 0 or args.child_timeout_seconds <= 0:
        raise ValueError("device must be non-negative and child timeout must be positive")
    validate_device(args.device)
    repo = Path(__file__).resolve().parents[1] if args.repo_root is None else args.repo_root.resolve()
    if args.child:
        if args.repo_root is None or args.plan is None:
            raise ValueError("Internal child requires --repo-root and --plan")
        return run_child(args)
    return run_parent(args, repo)


if __name__ == "__main__":
    raise SystemExit(main())
