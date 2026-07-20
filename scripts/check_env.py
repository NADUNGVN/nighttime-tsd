#!/usr/bin/env python3
"""Environment smoke check (host or Jetson)."""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def main() -> int:
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")

    try:
        import torch

        print(f"torch: {torch.__version__} | cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  device0: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("torch: NOT installed")

    try:
        import ultralytics

        print(f"ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("ultralytics: NOT installed")

    for cmd in ("tegrastats", "jtop"):
        path = shutil.which(cmd)
        print(f"{cmd}: {path or 'not in PATH'}")

    # Jetson model
    for p in (
        "/etc/nv_tegra_release",
        "/proc/device-tree/model",
    ):
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                print(f"{p}: {f.read().strip()[:200]}")
        except OSError:
            pass

    try:
        out = subprocess.check_output(
            ["dpkg-query", "-W", "-f=${Package} ${Version}\n", "tensorrt"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        print(f"tensorrt package: {out.strip()}")
    except Exception:
        print("tensorrt package: (not queried — non-Debian or not installed)")

    print("OK — review missing pieces above before Orin export.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
