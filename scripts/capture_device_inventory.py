#!/usr/bin/env python3
"""Capture an immutable hardware/software inventory before an IVC benchmark campaign."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(command: list[str]) -> dict[str, object]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"command": command, "available": False, "output": None}
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
    except subprocess.TimeoutExpired:
        return {"command": command, "available": True, "timeout": True, "output": None}
    return {
        "command": command,
        "available": True,
        "returncode": completed.returncode,
        "output": (completed.stdout + completed.stderr).strip() or None,
    }


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture the environment of a target benchmark device")
    parser.add_argument("--target", required=True, help="Stable identifier, e.g. xavier_nx_8gb")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--notes", default="", help="Cooling, power-supply, RAM, and board-revision notes")
    args = parser.parse_args()
    if not args.target.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--target must contain only letters, digits, hyphens, and underscores")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite inventory: {args.out}")

    commands = {
        "uname": ["uname", "-a"],
        "os_release": ["cat", "/etc/os-release"],
        "nvidia_smi": ["nvidia-smi"],
        "tegrastats_help": ["tegrastats", "--help"],
        "nvpmodel_query": ["nvpmodel", "-q"],
        "jetson_clocks_show": ["jetson_clocks", "--show"],
        "vcgencmd_throttled": ["vcgencmd", "get_throttled"],
        "lscpu": ["lscpu"],
        "free": ["free", "-h"],
        "df": ["df", "-h"],
    }
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "notes": args.notes or None,
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "packages": {name: package_version(name) for name in ("torch", "torchvision", "ultralytics", "tensorrt", "onnxruntime", "hailort")},
        "commands": {name: run(command) for name, command in commands.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
