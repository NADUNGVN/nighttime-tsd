#!/usr/bin/env python3
"""Collect a bounded, read-only inventory from one authorized SSH alias.

This collector intentionally does not install packages, change power/clock/fan
state, build artifacts, run inference, or benchmark.  It sends a fixed shell
fixture to the remote host and stores the raw command evidence in a new JSON
file.  Device aliases are used instead of committing private network details.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "e2l1-edge-inventory-v1"
BEGIN = "__E2L1_COMMAND__"
END = "__E2L1_EXIT__"
TARGET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SSH_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# These are queries only.  Keep this list small and review changes before use.
REMOTE_COMMANDS: tuple[tuple[str, str], ...] = (
    ("hostname", "hostname"),
    ("model", "sh -c 'tr -d \"\\000\" < /proc/device-tree/model'"),
    ("os_release", "cat /etc/os-release"),
    ("l4t_release_file", "sh -c '[ -r /etc/nv_tegra_release ] && cat /etc/nv_tegra_release || exit 127'"),
    ("nvidia_driver_proc", "sh -c '[ -r /proc/driver/nvidia/version ] && cat /proc/driver/nvidia/version || exit 127'"),
    ("uname", "uname -a"),
    ("architecture", "uname -m"),
    ("lscpu", "lscpu"),
    ("memory", "free -h"),
    ("root_filesystem", "df -hT /"),
    ("mounts", "findmnt -no SOURCE,FSTYPE,SIZE,USED,AVAIL,TARGET /"),
    ("pci", "sh -c 'command -v lspci >/dev/null && lspci -nn || exit 127'"),
    (
        "nvidia_smi",
        "sh -c 'command -v nvidia-smi >/dev/null || exit 127; nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,temperature.gpu --format=csv,noheader'",
    ),
    (
        "nvcc",
        "sh -c 'for p in /usr/local/cuda/bin/nvcc /usr/local/cuda-11.4/bin/nvcc /usr/local/cuda-12.6/bin/nvcc; do [ -x \"$p\" ] && \"$p\" --version && exit 0; done; exit 127'",
    ),
    (
        "trtexec",
        "sh -c 'for p in \"$(command -v trtexec 2>/dev/null)\" /usr/src/tensorrt/bin/trtexec; do if [ -x \"$p\" ]; then \"$p\" --help; exit $?; fi; done; exit 127'",
    ),
    (
        "hailortcli_version",
        "sh -c 'command -v hailortcli >/dev/null || exit 127; hailortcli --version'",
    ),
    (
        "hailo_identify",
        "sh -c 'command -v hailortcli >/dev/null || exit 127; hailortcli fw-control identify'",
    ),
    (
        "l4t_package",
        "sh -c 'command -v dpkg-query >/dev/null || exit 127; for p in nvidia-l4t-core libcudnn8 tensorrt python3-libnvinfer; do dpkg-query -W -f=\"" + chr(92) + "${Package}" + "\\t" + chr(92) + "${Version}" + "\\n\" \"$p\" 2>/dev/null || printf \"%s\\n\" \"$p=missing\"; done'",
    ),
    ("python", "python3 --version"),
    (
        "python_runtime_packages",
        "python3 -c 'import importlib.metadata as m; names=(\"torch\",\"torchvision\",\"ultralytics\",\"onnxruntime\",\"tensorrt\",\"hailort\"); d={str(x.metadata.get(\"Name\",\"\")).lower(): x.version for x in m.distributions()}; print(\"\\n\".join(f\"{n}={d[n]}\" for n in names if n in d))'",
    ),
    ("power_mode", "sh -c 'command -v nvpmodel >/dev/null || exit 127; nvpmodel -q'"),
    ("clock_state", "sh -c 'command -v jetson_clocks >/dev/null || exit 127; jetson_clocks --show'"),
    (
        "thermal_sources",
        "sh -c 'for f in /sys/class/thermal/thermal_zone*/type; do [ -r \"$f\" ] && printf \"%s: \" \"$f\" && cat \"$f\"; done; for f in /sys/class/thermal/thermal_zone*/temp; do [ -r \"$f\" ] && printf \"%s: \" \"$f\" && cat \"$f\"; done'",
    ),
    (
        "telemetry_and_power_sources",
        "sh -c 'for c in tegrastats jtop vcgencmd sensors ipmitool; do if command -v \"$c\" >/dev/null; then printf \"command:%s=%s\\n\" \"$c\" \"$(command -v \"$c\")\"; else printf \"command:%s=missing\\n\" \"$c\"; fi; done; for d in /sys/class/powercap /sys/bus/iio/devices; do if [ -d \"$d\" ]; then printf \"path:%s=present\\n\" \"$d\"; find \"$d\" -maxdepth 2 -type f -name \"*power*\" -o -name \"*energy*\" 2>/dev/null; else printf \"path:%s=missing\\n\" \"$d\"; fi; done'",
    ),
)

REMOTE_SCRIPT = "#!/bin/sh\nrun() {\n  name=\"$1\"\n  shift\n  printf '%s %s\\n' \"" + BEGIN + "\" \"$name\"\n  \"$@\" 2>&1\n  rc=$?\n  printf '%s %s %s\\n' \"" + END + "\" \"$name\" \"$rc\"\n}\n" + "\n".join(
    f"run {name} {command}" for name, command in REMOTE_COMMANDS
) + "\n"

# Some commands (notably device-tree reads) do not emit a trailing newline.
# Keep the end marker on its own line for the parser.
REMOTE_SCRIPT = REMOTE_SCRIPT.replace(
    "  rc=$?" + chr(10),
    "  rc=$?" + chr(10) + "  printf '\\n'" + chr(10),
)
REMOTE_SCRIPT = REMOTE_SCRIPT.replace(
    "  \"$@\" 2>&1" + chr(10),
    "  if command -v timeout >/dev/null 2>&1; then" + chr(10)
    + "    timeout 8 \"$@\" 2>&1" + chr(10)
    + "  else" + chr(10)
    + "    \"$@\" 2>&1" + chr(10)
    + "  fi" + chr(10),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_target(value: str) -> str:
    if not TARGET_RE.fullmatch(value):
        raise ValueError("target must contain only letters, digits, hyphens, and underscores")
    return value


def command_status(returncode: int | None, timed_out: bool = False) -> str:
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "ok"
    if returncode in (124, 137):
        return "timeout"
    if returncode == 127:
        return "missing"
    if returncode in (126, 13):
        return "permission_denied"
    return "unavailable"


def validate_ssh_alias(value: str) -> str:
    if not SSH_ALIAS_RE.fullmatch(value):
        raise ValueError("ssh_alias must be a safe existing SSH alias, not an option")
    return value


def parse_remote_output(
    raw: str,
    expected_names: set[str] | None = None,
    *,
    strict: bool = True,
) -> dict[str, dict[str, object]]:
    """Parse marker output and optionally enforce the complete command contract."""
    records: dict[str, dict[str, object]] = {}
    current: str | None = None
    buffer: list[str] = []
    duplicate_names: list[str] = []
    unexpected_names: list[str] = []
    for line in raw.splitlines():
        if line.startswith(BEGIN + " "):
            if current is not None:
                raise ValueError(f"nested command marker for {current}")
            current = line[len(BEGIN) + 1 :].strip()
            if expected_names is not None and current not in expected_names:
                unexpected_names.append(current)
            buffer = []
        elif line.startswith(END + " "):
            parts = line.split()
            if current is None or len(parts) != 3 or parts[1] != current:
                raise ValueError("unmatched command end marker")
            if current in records:
                duplicate_names.append(current)
            records[current] = {
                "status": command_status(int(parts[2])),
                "returncode": int(parts[2]),
                "output": "\n".join(buffer).strip() or None,
            }
            current = None
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        if strict:
            raise ValueError(f"unterminated command marker for {current}")
        records[current] = {
            "status": "partial",
            "returncode": None,
            "output": "\n".join(buffer).strip() or None,
        }
    if strict and duplicate_names:
        raise ValueError(f"duplicate command markers: {sorted(set(duplicate_names))}")
    if strict and unexpected_names:
        raise ValueError(f"unexpected command markers: {sorted(set(unexpected_names))}")
    if strict and expected_names is not None:
        missing = sorted(expected_names.difference(records))
        if missing:
            raise ValueError(f"missing command markers: {missing}")
    return records


def collect(target: str, ssh_alias: str, out: Path, notes: str = "") -> dict[str, object]:
    validate_target(target)
    validate_ssh_alias(ssh_alias)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite inventory: {out}")

    ssh_command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=yes",
        ssh_alias,
        "sh",
        "-s",
    ]
    started = datetime.now(timezone.utc).isoformat()
    try:
        completed = subprocess.run(
            ssh_command,
            input=REMOTE_SCRIPT.encode("utf-8"),
            capture_output=True,
            timeout=60,
            check=False,
        )
        timed_out = False
        raw = (
            completed.stdout.decode("utf-8", errors="replace")
            + completed.stderr.decode("utf-8", errors="replace")
        ).strip()
        returncode: int | None = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        raw = ((exc.stdout or b"") + (exc.stderr or b"")).decode(errors="replace").strip() if isinstance(exc.stdout, bytes) else ((exc.stdout or "") + (exc.stderr or "")).strip()
        returncode = None

    parse_error: str | None = None
    records: dict[str, dict[str, object]] = {}
    if BEGIN in raw:
        try:
            records = parse_remote_output(
                raw,
                {name for name, _ in REMOTE_COMMANDS},
                strict=not timed_out and returncode == 0,
            )
        except ValueError as exc:
            parse_error = str(exc)
            try:
                records = parse_remote_output(raw, strict=False)
            except ValueError:
                records = {}
    expected_names = {name for name, _ in REMOTE_COMMANDS}
    missing_commands = sorted(expected_names.difference(records))
    complete = not timed_out and returncode == 0 and not parse_error and not missing_commands and len(records) == len(expected_names)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": started,
        "target": target,
        "ssh_alias": ssh_alias,
        "ssh_command": ssh_command,
        "collector_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "transport": {
            "status": "ok" if returncode == 0 and not timed_out else "unavailable",
            "returncode": returncode,
            "timed_out": timed_out,
            "stderr_or_unmarked_output": raw if not records else None,
        },
        "inventory_completeness": {
            "status": "complete" if complete else ("partial" if records else "unavailable"),
            "expected_count": len(expected_names),
            "recorded_count": len(records),
            "missing_commands": missing_commands,
            "parse_error": parse_error,
        },
        "operator_notes": notes or None,
        "commands": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2L1 bounded read-only SSH inventory")
    parser.add_argument("--target", required=True)
    parser.add_argument("--ssh-alias", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    payload = collect(args.target, args.ssh_alias, args.out, args.notes)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["transport"]["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
