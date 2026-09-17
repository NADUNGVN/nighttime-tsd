#!/usr/bin/env python3
"""Bounded read-only E2 CUDA runtime/header feasibility inspection.

This probe intentionally does not load libcudart, call ctypes.CDLL, initialize
CUDA, allocate memory, create a stream, or inspect TensorRT/model state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REMOTE_CHILD_TIMEOUT_SECONDS = 50
REMOTE_SCRIPT = r'''#!/bin/sh
set +e
printf '%s\n' '__E2L1_CUDA_PROBE__ begin'
printf 'hostname='; hostname 2>&1
printf 'nvcc_path='; command -v nvcc 2>&1 || true
printf 'nvcc_version='; nvcc --version 2>&1 | tail -n 1 || true
printf 'ldconfig_libcudart='; ldconfig -p 2>&1 | grep -E 'libcudart\.so(\.11|\.so)' | head -n 3 || true
for lib in /usr/local/cuda-11.4/lib64/libcudart.so.11.0 /usr/local/cuda/lib64/libcudart.so.11.0; do
  if [ -r "$lib" ]; then
    printf 'library_path=%s\n' "$lib"
    printf 'library_realpath='; readlink -f "$lib" 2>&1
    printf 'library_stat='; stat -c '%A %U %G %s %n' "$lib" 2>&1
    printf 'library_symbols=\n'
    nm -D --defined-only "$lib" 2>&1 | grep -E 'cuda(Malloc|Free|MemcpyAsync|StreamSynchronize|GetErrorString|MallocHost|FreeHost)' | head -n 30 || true
  fi
done
for header in /usr/local/cuda-11.4/include/cuda_runtime_api.h /usr/local/cuda/include/cuda_runtime_api.h; do
  if [ -r "$header" ]; then
    printf 'header_path=%s\n' "$header"
    printf 'header_realpath='; readlink -f "$header" 2>&1
    printf 'header_stat='; stat -c '%A %U %G %s %n' "$header" 2>&1
    grep -nE 'cuda(Malloc|Free|MemcpyAsync|StreamSynchronize|GetErrorString|MallocHost|FreeHost)[[:space:]]*\(' "$header" | head -n 40 || true
  fi
done
printf '%s\n' '__E2L1_CUDA_PROBE__ end'
'''


def script_sha256() -> str:
    return hashlib.sha256(REMOTE_SCRIPT.encode("utf-8")).hexdigest()


def run_probe(output_root: Path, *, alias: str = "nx", timeout_seconds: int = 60) -> dict[str, object]:
    if alias != "nx" or not re.fullmatch(r"[A-Za-z0-9_-]+", alias):
        raise ValueError("E2_ALIAS_REQUIRED")
    if timeout_seconds <= REMOTE_CHILD_TIMEOUT_SECONDS:
        raise ValueError("LOCAL_TIMEOUT_MUST_EXCEED_REMOTE_TIMEOUT")
    output_root.mkdir(parents=True, exist_ok=False)
    command = [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=yes", alias, "timeout",
        f"{REMOTE_CHILD_TIMEOUT_SECONDS}s", "sh", "-s",
    ]
    started = time.monotonic_ns()
    try:
        completed = subprocess.run(command, input=REMOTE_SCRIPT.encode("utf-8"), capture_output=True, timeout=timeout_seconds, check=False)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(command, 124, exc.stdout or b"", exc.stderr or b"")
        timed_out = True
    ended = time.monotonic_ns()
    stdout = completed.stdout if isinstance(completed.stdout, bytes) else str(completed.stdout).encode("utf-8")
    stderr = completed.stderr if isinstance(completed.stderr, bytes) else str(completed.stderr).encode("utf-8")
    (output_root / "remote.stdout").write_bytes(stdout)
    (output_root / "remote.stderr").write_bytes(stderr)
    manifest: dict[str, object] = {
        "schema_version": "e2l1-edge-cuda-runtime-probe-v1",
        "status": "timeout" if timed_out else ("ok" if completed.returncode == 0 else "remote_error"),
        "scope": "E2 only; read-only libcudart/header/symbol inspection; no CUDA load or initialization",
        "alias": alias,
        "remote_child_timeout_seconds": REMOTE_CHILD_TIMEOUT_SECONDS,
        "local_timeout_seconds": timeout_seconds,
        "sanitized_command": f"ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes <alias> timeout {REMOTE_CHILD_TIMEOUT_SECONDS}s sh -s",
        "command_sha256": hashlib.sha256(" ".join(command).encode("utf-8")).hexdigest(),
        "remote_script_sha256": script_sha256(),
        "returncode": completed.returncode,
        "timed_out": timed_out,
        "local_elapsed_ns": ended - started,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {"stdout": "remote.stdout", "stderr": "remote.stderr"},
        "forbidden_actions": ["ctypes.CDLL", "CUDA load/init", "context/stream creation", "allocation", "memory copy", "TensorRT/model operation", "package installation", "sudo", "device writes"],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded read-only E2 CUDA runtime/header probe")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alias", default="nx")
    args = parser.parse_args(argv)
    result = run_probe(args.out_dir, alias=args.alias)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
