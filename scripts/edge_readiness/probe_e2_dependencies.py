#!/usr/bin/env python3
"""One bounded, read-only E2 dependency/API metadata probe.

The remote script never imports TensorRT/CUDA; it uses module metadata and
source-token discovery to avoid import-time CUDA initialization. It writes a
new local artifact root and never writes to the device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REMOTE_SCRIPT = r'''#!/bin/sh
export PYTHONDONTWRITEBYTECODE=1
printf '%s\n' '__E2L1_PROBE__ begin'
printf 'hostname='; hostname 2>&1
printf 'python3='; python3 --version 2>&1
printf 'disk_root='; df -hT / 2>&1
printf 'home_path='; printf '%s\n' "$HOME"
if [ -d "$HOME" ]; then printf 'home_directory=present\n'; else printf 'home_directory=missing\n'; fi
if [ -w "$HOME" ]; then printf 'home_permission=writable\n'; else printf 'home_permission=not_writable\n'; fi
python3 -B - <<'PY'
import importlib.metadata as metadata
import importlib.util
import pathlib
import re
import sys

print('python_sys_version=' + sys.version.replace('\n', ' '))
names = ('numpy', 'tensorrt', 'pycuda', 'cuda', 'cuda.cudart', 'onnx', 'onnxruntime', 'torch', 'ultralytics')
for name in names:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:
        print('module=' + name + ';spec_error=' + type(exc).__name__)
        continue
    if spec is None:
        print('module=' + name + ';status=missing')
        continue
    origin = spec.origin or ''
    print('module=' + name + ';status=present;origin=' + origin)
    try:
        print('package_version=' + name + ';version=' + metadata.version(name))
    except metadata.PackageNotFoundError:
        print('package_version=' + name + ';version=unresolved')
    except Exception as exc:
        print('package_version=' + name + ';version_error=' + type(exc).__name__)
    if name == 'tensorrt' and origin and origin != 'built-in':
        try:
            source = pathlib.Path(origin).read_text(errors='replace')
            tokens = ('Runtime', 'deserialize_cuda_engine', 'execute_async_v2', 'set_tensor_address', 'execute_async_v3')
            print('tensorrt_source_tokens=' + ','.join(token + ':' + ('present' if re.search(re.escape(token), source) else 'absent') for token in tokens))
        except Exception as exc:
            print('tensorrt_source_read=error:' + type(exc).__name__)
print('__E2L1_PROBE__ end')
PY
'''


def script_sha256() -> str:
    return hashlib.sha256(REMOTE_SCRIPT.encode("utf-8")).hexdigest()


def run_probe(output_root: Path, *, alias: str = "nx", timeout_seconds: int = 60) -> dict:
    output_root.mkdir(parents=True, exist_ok=False)
    command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes", alias, "sh", "-s"]
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
    manifest = {
        "schema_version": "e2l1-edge-dependency-probe-v1",
        "status": "timeout" if timed_out else ("ok" if completed.returncode == 0 else "remote_error"),
        "scope": "E2 only; read-only metadata/source discovery; no TensorRT/CUDA import or initialization",
        "alias": alias,
        "sanitized_command": "ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes <alias> sh -s",
        "command_sha256": hashlib.sha256(" ".join(command).encode("utf-8")).hexdigest(),
        "remote_script_sha256": script_sha256(),
        "returncode": completed.returncode,
        "timed_out": timed_out,
        "local_elapsed_ns": ended - started,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {"stdout": "remote.stdout", "stderr": "remote.stderr"},
        "forbidden_actions": ["package installation", "sudo", "chmod", "clock/fan/power changes", "CUDA context/allocation", "engine deserialize/build", "model load/forward", "benchmark", "device writes"],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded read-only E2 dependency probe")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alias", default="nx")
    args = parser.parse_args(argv)
    result = run_probe(args.out_dir, alias=args.alias)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
