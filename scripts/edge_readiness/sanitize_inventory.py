#!/usr/bin/env python3
"""Create a public-safe summary from a private read-only device inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

IPV4_RE = re.compile(
    r"(?i)(\b(?:ip|ipv4|address|gateway|route|inet)\s*(?:=|:|\s)\s*)"
    r"(?:\d{1,3}\.){3}\d{1,3}\b"
)
MAC_RE = re.compile(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b")
IDENTIFIER_RE = re.compile(
    r"(?im)^.*(?:serial(?:\s+number)?|mac(?:\s+address)?|uuid|device\s+id)\s*[:=].*$"
)
NVIDIA_DRIVER_RE = re.compile(r"NVRM version:.*?aarch64\s+([^\s]+)", re.IGNORECASE)
ALLOWED_COMMANDS = (
    "model", "os_release", "l4t_release_file", "uname", "architecture",
    "nvidia_driver_proc", "lscpu", "memory", "root_filesystem", "mounts", "nvidia_smi", "nvcc",
    "trtexec", "hailortcli_version", "hailo_identify", "l4t_package",
    "python", "python_runtime_packages", "python_numpy_opencv",
    "backend_package_inventory", "power_mode", "clock_state",
    "thermal_sources", "telemetry_and_power_sources", "current_workloads",
)
LSCPU_FIELDS = (
    "Architecture:", "CPU(s):", "On-line CPU(s) list:", "Model name:",
    "Vendor ID:", "Core(s) per socket:", "Socket(s):", "CPU max MHz:",
    "CPU min MHz:",
)


def _safe_output(name: str, raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw
    if name == "lscpu":
        text = "\n".join(
            line for line in text.splitlines()
            if any(line.startswith(field) for field in LSCPU_FIELDS)
        )
    elif name == "trtexec":
        text = "\n".join(text.splitlines()[:1])
    elif name == "nvcc":
        text = "\n".join(
            line for line in text.splitlines()
            if "release " in line.lower() or "cuda compilation tools" in line.lower()
        )
    elif name == "nvidia_driver_proc":
        match = NVIDIA_DRIVER_RE.search(text)
        text = f"NVRM driver version={match.group(1)}" if match else "NVRM driver version=unparsed"
    elif name == "current_workloads":
        text = "\n".join(line for line in text.splitlines() if not re.match(r"^\s*ps\s", line))
    text = IDENTIFIER_RE.sub("[REDACTED DEVICE IDENTIFIER]", text)
    text = MAC_RE.sub("[REDACTED MAC]", text)
    text = IPV4_RE.sub(r"\1[REDACTED IP]", text)
    return text or None


def sanitize_inventory(payload: dict[str, Any]) -> dict[str, Any]:
    commands = payload.get("commands")
    if not isinstance(commands, dict):
        raise ValueError("inventory commands must be an object")
    safe_commands: dict[str, Any] = {}
    for name in ALLOWED_COMMANDS:
        record = commands.get(name)
        if not isinstance(record, dict):
            continue
        raw_output = record.get("output")
        encoded = raw_output.encode("utf-8") if isinstance(raw_output, str) else b""
        safe_commands[name] = {
            "status": record.get("status"),
            "returncode": record.get("returncode"),
            "output": _safe_output(name, raw_output if isinstance(raw_output, str) else None),
            "raw_output_bytes": len(encoded),
            "raw_output_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    transport = payload.get("transport")
    completeness = payload.get("inventory_completeness")
    if not isinstance(transport, dict) or not isinstance(completeness, dict):
        raise ValueError("inventory transport/completeness must be objects")
    unmarked = transport.get("stderr_or_unmarked_output")
    unmarked_bytes = unmarked.encode("utf-8") if isinstance(unmarked, str) else b""
    return {
        "schema_version": payload.get("schema_version"),
        "created_utc": payload.get("created_utc"),
        "target": payload.get("target"),
        "collector_sha256": payload.get("collector_sha256"),
        "transport": {
            "status": transport.get("status"),
            "returncode": transport.get("returncode"),
            "timed_out": transport.get("timed_out"),
            "unmarked_output_bytes": len(unmarked_bytes),
            "unmarked_output_sha256": hashlib.sha256(unmarked_bytes).hexdigest(),
        },
        "inventory_completeness": completeness,
        "commands": safe_commands,
        "sanitization": {
            "policy": "allowlisted read-only outputs; hostname/SSH route omitted; IP/MAC/device identifiers redacted",
            "raw_command_output_retained": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="private inventory JSON")
    parser.add_argument("--output", type=Path, required=True, help="new sanitized JSON output")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite sanitized inventory: {args.output}")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    sanitized = sanitize_inventory(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(sanitized, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "target": sanitized["target"],
        "inventory_status": sanitized["inventory_completeness"]["status"],
        "commands": len(sanitized["commands"]),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
