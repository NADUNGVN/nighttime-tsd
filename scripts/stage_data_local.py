#!/usr/bin/env python3
"""Copy YOLO dataset from NFS home to local disk (biggest train speedup).

Usage:
  python scripts/stage_data_local.py
  python scripts/stage_data_local.py --dst /tmp/cctsdb2021_full
  python scripts/stage_data_local.py --dst /var/tmp/$USER/cctsdb2021_full --force
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def fs_type(path: Path) -> str:
    try:
        out = subprocess.check_output(["df", "-T", str(path)], text=True)
        line = out.strip().splitlines()[-1]
        return line.split()[1].lower()
    except Exception:
        return "unknown"


def dir_size_gb(path: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total / 1e9


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage dataset to local disk")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/processed/cctsdb2021_full"),
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/tmp") / f"cctsdb2021_full_{os.environ.get('USER', 'user')}",
    )
    parser.add_argument("--force", action="store_true", help="Delete dst and recopy")
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()

    if not src.exists():
        print(f"ERROR: src missing: {src}")
        return 1

    print(f"src: {src}  fs={fs_type(src)}")
    print(f"dst: {dst}  parent_fs={fs_type(dst.parent)}")

    pfs = fs_type(dst.parent)
    if pfs in {"nfs", "nfs4", "cifs", "smb"}:
        print(
            f"WARN: destination parent is still '{pfs}'. "
            "Speedup will be small. Prefer true local disk."
        )
    else:
        print(f"OK: destination looks local-ish (fs={pfs})")

    if dst.exists() and not args.force:
        n = len(list((dst / "train" / "images").glob("*"))) if (dst / "train" / "images").exists() else 0
        if n >= 10000:
            print(f"dst already has ~{n} train images — skip copy (use --force to redo)")
            print(f"\nTrain with:\n  python scripts/train_3090.py --local-data {dst}")
            return 0
        print("dst incomplete — will recopy")
        shutil.rmtree(dst)

    if dst.exists() and args.force:
        shutil.rmtree(dst)

    free = shutil.disk_usage(str(dst.parent)).free / 1e9
    need = dir_size_gb(src) * 1.05
    print(f"size src ≈ {need/1.05:.1f} GB | free on dst parent ≈ {free:.1f} GB")
    if free < need:
        print("ERROR: not enough free space on destination")
        return 1

    print("Copying (rsync if available, else shutil)...")
    t0 = time.time()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        subprocess.check_call(
            ["rsync", "-a", "--info=progress2", f"{src}/", f"{dst}/"]
        )
    else:
        shutil.copytree(src, dst)
    print(f"Copy done in {time.time()-t0:.0f}s → {dst}")

    # drop label caches from NFS copy (rebuild local is fine/faster)
    for p in dst.rglob("*.cache"):
        p.unlink(missing_ok=True)
        print(f"  removed cache {p.name}")

    n_tr = len(list((dst / "train" / "images").glob("*")))
    n_te = len(list((dst / "test" / "images").glob("*")))
    print(f"counts: train={n_tr} test={n_te}")

    # write local yaml
    yaml_path = Path("configs/cctsdb2021_full_local.yaml")
    yaml_path.write_text(
        f"""# auto: local staged CCTSDB full
path: {dst.as_posix()}
train: train/images
val: test/images
test: test/images
names:
  0: prohibitory
  1: mandatory
  2: warning
nc: 3
""",
        encoding="utf-8",
    )
    print(f"wrote {yaml_path}")
    print("\nNext:")
    print(f"  python scripts/train_3090.py --local-data {dst} --model yolo11n.pt")
    print("  # or")
    print(f"  python scripts/train_baseline.py --data {yaml_path} --workers 6 --batch 96 --cache ram \\")
    print("      --model yolo11n.pt --name yolo11n_cctsdb_full_fast")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
