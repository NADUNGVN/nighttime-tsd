#!/usr/bin/env python3
"""Bounded, raw, read-only telemetry feasibility sampling over SSH."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "e2l1-edge-telemetry-v1"
SAMPLE_MARKER = "__E2L1_SAMPLE__"
SOURCE_MARKER = "__E2L1_SOURCE__"
TARGET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SSH_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SAMPLE_RE = re.compile(r"^" + re.escape(SAMPLE_MARKER) + r" device_monotonic_ns=(\d+)$")
KNOWN_DEVICES = {"E1": "pi5", "E2": "nx", "E3": "agx", "E5": "nano"}

REMOTE_SCRIPT = r'''#!/bin/sh
# E2L1-005: foreground, bounded, read-only sensor snapshots only.
sample_once() {
  ts="unknown"
  if command -v python3 >/dev/null 2>&1; then
    ts=$(python3 -c 'import time; print(time.monotonic_ns())' 2>/dev/null) || ts="unknown"
  fi
  printf '%s device_monotonic_ns=%s\n' "__E2L1_SAMPLE__" "$ts"
  printf '%s source=free_bytes\n' "__E2L1_SOURCE__"
  free -b 2>&1
  printf '%s source=thermal_sysfs\n' "__E2L1_SOURCE__"
  for f in /sys/class/thermal/thermal_zone*/type; do
    [ -r "$f" ] && printf '%s: ' "$f" && cat "$f"
  done
  for f in /sys/class/thermal/thermal_zone*/temp; do
    [ -r "$f" ] && printf '%s: ' "$f" && cat "$f"
  done
  printf '%s source=tegrastats\n' "__E2L1_SOURCE__"
  if command -v tegrastats >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
    timeout 2s tegrastats --interval 1000 2>&1
  else
    printf 'unavailable: tegrastats or timeout is missing\n'
  fi
  printf '%s source=nvidia_smi\n' "__E2L1_SOURCE__"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,temperature.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits 2>&1
  else
    printf 'unavailable: nvidia-smi is missing\n'
  fi
  printf '%s source=vcgencmd\n' "__E2L1_SOURCE__"
  if command -v vcgencmd >/dev/null 2>&1; then
    vcgencmd measure_temp 2>&1
    vcgencmd get_throttled 2>&1
  else
    printf 'unavailable: vcgencmd is missing\n'
  fi
  printf '%s source=power_files\n' "__E2L1_SOURCE__"
  found=0
  for f in /sys/class/powercap/*/energy_uj /sys/bus/iio/devices/*/in_power*_input; do
    if [ -r "$f" ]; then
      found=1
      printf '%s=' "$f"
      cat "$f"
    fi
  done
  [ "$found" -eq 1 ] || printf 'unavailable: no readable power files found\n'
}
printf '%s command=source_inventory\n' "__E2L1_SOURCE__"
for c in python3 free tegrastats timeout nvidia-smi vcgencmd sensors; do
  if command -v "$c" >/dev/null 2>&1; then
    printf 'command:%s=%s\n' "$c" "$(command -v "$c")"
  else
    printf 'command:%s=missing\n' "$c"
  fi
done
for d in /sys/class/powercap /sys/bus/iio/devices /sys/class/thermal; do
  if [ -d "$d" ]; then printf 'path:%s=present\n' "$d"; else printf 'path:%s=missing\n' "$d"; fi
done
i=0
while [ "$i" -lt 5 ]; do
  sample_once
  i=$((i + 1))
  if [ "$i" -lt 5 ]; then sleep 1; fi
done
'''

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def validate_target(value: str) -> str:
    if not TARGET_RE.fullmatch(value):
        raise ValueError("target must contain only letters, digits, hyphens, and underscores")
    return value

def validate_ssh_alias(value: str) -> str:
    if not SSH_ALIAS_RE.fullmatch(value):
        raise ValueError("ssh_alias must be a safe existing SSH alias, not an option")
    return value

def command_status(returncode: int | None, timed_out: bool = False) -> str:
    if timed_out or returncode in (124, 137):
        return "timeout"
    if returncode == 0:
        return "ok"
    if returncode == 127:
        return "missing"
    if returncode in (13, 126):
        return "permission_denied"
    return "unavailable"

def parse_sample_blocks(raw: str) -> list[dict[str, Any]]:
    """Parse sample boundaries and preserve every source line verbatim."""
    samples: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in raw.splitlines():
        match = SAMPLE_RE.fullmatch(line)
        if match:
            if current is not None:
                samples.append(current)
            current = {"device_monotonic_ns": int(match.group(1)), "raw_lines": []}
        elif current is not None:
            current["raw_lines"].append(line)
    if current is not None:
        samples.append(current)
    for sample in samples:
        sample["raw"] = "\n".join(sample.pop("raw_lines"))
    return samples

def parse_source_inventory(raw: str) -> list[str]:
    return [line for line in raw.splitlines() if line.startswith(SOURCE_MARKER)]

def parse_channel_observations(raw: str) -> dict[str, list[dict[str, Any]]]:
    """Parse explicit memory/thermal/power labels without merging rails."""
    channels: dict[str, list[dict[str, Any]]] = {"memory": [], "thermal": [], "power": [], "throttle": []}
    sample_index = -1
    source = "unknown"
    for line in raw.splitlines():
        sample_match = SAMPLE_RE.fullmatch(line)
        if sample_match:
            sample_index += 1
            source = "unknown"
            continue
        if line.startswith(SOURCE_MARKER + " source="):
            source = line.split(" source=", 1)[1]
            continue
        if source == "free_bytes" and line.startswith("Mem:"):
            fields = line.split()
            if len(fields) >= 7 and all(field.isdigit() for field in fields[1:7]):
                labels = ("total", "used", "free", "shared", "buff_cache", "available")
                channels["memory"].append({"sample_index": sample_index, "source": source, "unit": "bytes", **dict(zip(labels, map(int, fields[1:7])))})
        if source == "thermal_sysfs" and "/temp: " in line:
            path, value = line.rsplit(": ", 1)
            if value.isdigit():
                channels["thermal"].append({"sample_index": sample_index, "source": source, "label": path, "value_millidegrees_c": int(value), "unit": "millidegrees_C"})
        if source == "tegrastats":
            ram = re.search(r"\bRAM (\d+)/(\d+)MB", line)
            if ram:
                channels["memory"].append({"sample_index": sample_index, "source": source, "label": "RAM", "used": int(ram.group(1)), "total": int(ram.group(2)), "unit": "MB"})
            for label, value in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)@([0-9]+(?:\.[0-9]+)?)C", line):
                channels["thermal"].append({"sample_index": sample_index, "source": source, "label": label, "value_c": float(value), "unit": "C"})
            for label, first, second in re.findall(r"\b([A-Z][A-Z0-9_]*) (\d+)mW/(\d+)mW", line):
                channels["power"].append({"sample_index": sample_index, "source": source, "label": label, "instantaneous_mw": int(first), "averaged_mw": int(second), "unit": "mW", "boundary": "unknown_rail_boundary"})
        if source == "nvidia_smi" and line and not line.startswith("unavailable:"):
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 5:
                try:
                    temperature, power, used, total = float(fields[1]), float(fields[2]), int(fields[3]), int(fields[4])
                except ValueError:
                    continue
                channels["thermal"].append({"sample_index": sample_index, "source": source, "label": "temperature.gpu", "value_c": temperature, "unit": "C"})
                channels["power"].append({"sample_index": sample_index, "source": source, "label": "power.draw", "value_w": power, "unit": "W", "boundary": "unknown_gpu_boundary"})
                channels["memory"].append({"sample_index": sample_index, "source": source, "label": "memory.used", "used": used, "total": total, "unit": "MiB"})
        if source == "vcgencmd":
            temp = re.fullmatch(r"temp=([0-9]+(?:\.[0-9]+)?)'C", line)
            if temp:
                channels["thermal"].append({"sample_index": sample_index, "source": source, "label": "vcgencmd_temp", "value_c": float(temp.group(1)), "unit": "C"})
            if line.startswith("throttled="):
                channels["throttle"].append({"sample_index": sample_index, "source": source, "label": "vcgencmd_throttled", "raw_value": line.split("=", 1)[1]})
    return channels

def summarize_samples(raw: str) -> dict[str, Any]:
    samples = parse_sample_blocks(raw)
    timestamps = [sample["device_monotonic_ns"] for sample in samples]
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    channels = parse_channel_observations(raw)
    return {
        "sample_count": len(samples),
        "samples": samples,
        "source_markers": parse_source_inventory(raw),
        "timestamp_origin": "device_monotonic_ns from device python3 time.monotonic_ns; raw source lines are unmodified",
        "timestamp_status": "available" if samples else "unavailable_or_partial",
        "timestamp_order": "strictly_increasing" if all(right > left for left, right in zip(timestamps, timestamps[1:])) else "not_verified",
        "sample_gaps_ns": gaps,
        "max_gap_ns": max(gaps or [0]),
        "channels": channels,
        "available_channels": [name for name, observations in channels.items() if observations],
        "power_boundary_note": "Original rail/source labels are preserved; rails are not summed and boundary is unknown unless separately established.",
        "ssh_receive_timestamp_note": "SSH receipt timestamps are local transport times and are not sensor acquisition times",
    }

def write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)

def collect(target: str, ssh_alias: str, results_root: Path, *, local_timeout_s: float = 28.0) -> dict[str, Any]:
    validate_target(target)
    validate_ssh_alias(ssh_alias)
    if target not in KNOWN_DEVICES:
        raise ValueError(f"target is not an authorized E1/E2/E3/E5 telemetry target: {target}")
    if KNOWN_DEVICES[target] != ssh_alias:
        raise ValueError(f"ssh alias does not match authorized target binding: {target} -> {KNOWN_DEVICES[target]}")
    started_wall = datetime.now(timezone.utc)
    started_mono = time.monotonic_ns()
    run_id = started_wall.strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = results_root / target / run_id
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite telemetry output: {out_dir}")
    ssh_command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=yes", ssh_alias, "sh", "-s"]
    timed_out = False
    try:
        completed = subprocess.run(ssh_command, input=REMOTE_SCRIPT.encode("utf-8"), capture_output=True, timeout=local_timeout_s, check=False)
        returncode: int | None = completed.returncode
        stdout, stderr = completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout, stderr = exc.stdout or b"", exc.stderr or b""
    finished_mono = time.monotonic_ns()
    if isinstance(stdout, str):
        stdout = stdout.encode("utf-8", errors="replace")
    if isinstance(stderr, str):
        stderr = stderr.encode("utf-8", errors="replace")
    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": started_wall.isoformat(),
        "target": target,
        "ssh_alias": ssh_alias,
        "ssh_command": ssh_command,
        "collector_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "remote_script_sha256": sha256_bytes(REMOTE_SCRIPT.encode("utf-8")),
        "transport": {
            "status": command_status(returncode, timed_out),
            "returncode": returncode,
            "timed_out": timed_out,
            "local_receive_start_monotonic_ns": started_mono,
            "local_receive_end_monotonic_ns": finished_mono,
            "local_duration_ns": finished_mono - started_mono,
        },
        "scope": {
            "read_only": True,
            "max_device_sample_window_seconds": 30,
            "sample_plan": "five foreground snapshots; tegrastats child bounded to 2 seconds per snapshot",
            "workload_generated": False,
            "model_loaded": False,
            "configuration_changed": False,
        },
        "telemetry": summarize_samples(stdout_text),
        "stderr_or_ssh_diagnostics": stderr_text or None,
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    stdout_path, stderr_path, summary_path = out_dir / "telemetry.stdout", out_dir / "telemetry.stderr", out_dir / "telemetry.json"
    write_exclusive(stdout_path, stdout)
    write_exclusive(stderr_path, stderr)
    summary_bytes = (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    write_exclusive(summary_path, summary_bytes)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "run_id": run_id,
        "artifacts": {
            "telemetry.stdout": {"sha256": sha256_bytes(stdout_path.read_bytes()), "bytes": stdout_path.stat().st_size},
            "telemetry.stderr": {"sha256": sha256_bytes(stderr_path.read_bytes()), "bytes": stderr_path.stat().st_size},
            "telemetry.json": {"sha256": sha256_bytes(summary_path.read_bytes()), "bytes": summary_path.stat().st_size},
        },
        "manifest_hash_note": "Hashes are computed from bytes after exclusive writes on Windows; manifest is not self-hashed.",
    }
    write_exclusive(out_dir / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return summary | {"run_id": run_id, "output_dir": str(out_dir), "manifest": manifest}

def enrich_existing_run(run_dir: Path) -> dict[str, Any]:
    """Parse an existing capture locally without re-contacting the device."""
    stdout_path = run_dir / "telemetry.stdout"
    if not stdout_path.is_file():
        raise FileNotFoundError(stdout_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_stdout_sha256": sha256_bytes(stdout_path.read_bytes()),
        "parser_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "telemetry": summarize_samples(stdout_path.read_bytes().decode("utf-8", errors="replace")),
        "note": "Local post-capture parse; no SSH/device contact was made.",
    }
    parsed_path = run_dir / "parsed_channels.json"
    write_exclusive(parsed_path, (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    parsed_manifest = {
        "source_stdout_sha256": payload["source_stdout_sha256"],
        "parsed_channels": {"sha256": sha256_bytes(parsed_path.read_bytes()), "bytes": parsed_path.stat().st_size},
        "manifest_hash_note": "Hash is computed from parsed_channels.json bytes after exclusive write; this manifest is not self-hashed.",
    }
    write_exclusive(run_dir / "parsed_manifest.json", (json.dumps(parsed_manifest, indent=2) + "\n").encode("utf-8"))
    return payload | {"parsed_path": str(parsed_path), "parsed_manifest": parsed_manifest}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2L1-005 bounded read-only SSH telemetry collector")
    parser.add_argument("--target", required=True, choices=sorted(KNOWN_DEVICES))
    parser.add_argument("--ssh-alias", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--parse-run", type=Path, help="locally parse an existing run without SSH")
    args = parser.parse_args(argv)
    if args.parse_run is not None:
        print(json.dumps(enrich_existing_run(args.parse_run), indent=2, ensure_ascii=False))
        return 0
    payload = collect(args.target, args.ssh_alias, args.results_root)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["transport"]["status"] == "ok" else 2

if __name__ == "__main__":
    raise SystemExit(main())
