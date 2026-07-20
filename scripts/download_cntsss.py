#!/usr/bin/env python3
"""Download CNTSSS from Google Drive (YOLO-LLTS release).

File ID from: https://github.com/linzy88/YOLO-LLTS
Expected: CNTSSS.zip ~885 MB
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Google Drive file id for CNTSSS.zip
DEFAULT_FILE_ID = "1A-7t-Wb5rjUZslUJ_1tltlUUvtSxBXdX"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CNTSSS.zip via gdown")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    parser.add_argument("--file-id", default=DEFAULT_FILE_ID, help="Google Drive file id")
    parser.add_argument("--extract", action="store_true", help="Unzip after download")
    args = parser.parse_args()

    try:
        import gdown
    except ImportError:
        print("ERROR: gdown not installed. Run: pip install gdown", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.out_dir / "CNTSSS.zip"
    url = f"https://drive.google.com/uc?id={args.file_id}"

    print(f"[1/2] Downloading CNTSSS → {zip_path}")
    print(f"      URL: {url}")
    if zip_path.exists() and zip_path.stat().st_size > 100_000_000:
        print(f"      Already present ({zip_path.stat().st_size / 1e6:.1f} MB) — skip download")
    else:
        gdown.download(url, str(zip_path), quiet=False)
        if not zip_path.exists():
            print("ERROR: download failed", file=sys.stderr)
            return 1
        print(f"      Done: {zip_path.stat().st_size / 1e6:.1f} MB")

    if args.extract:
        import zipfile

        extract_to = args.out_dir / "CNTSSS"
        print(f"[2/2] Extracting → {extract_to}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(args.out_dir)
        # Normalize nested folder names if zip has CNTSSS/ root
        if extract_to.exists():
            print(f"      Extracted to {extract_to}")
        else:
            print(f"      Extracted under {args.out_dir} — run verify_cntsss.py to locate")
    else:
        print("[2/2] Skip extract (pass --extract to unzip)")

    print("OK. Next: python scripts/verify_cntsss.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
