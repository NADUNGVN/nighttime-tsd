#!/usr/bin/env python3
"""Preflight checks BEFORE long CCTSDB-full training.

Run this first. Exit code 0 = safe to start full train.
Exit code != 0 = fix listed issues first.

Usage:
  python scripts/preflight_cctsdb.py
  python scripts/preflight_cctsdb.py --smoke-train   # 1 mini epoch (~few min)
  python scripts/preflight_cctsdb.py --data-root $HOME/data_local/cctsdb_full
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Report:
    def __init__(self) -> None:
        self.ok: list[str] = []
        self.warn: list[str] = []
        self.fail: list[str] = []

    def add_ok(self, msg: str) -> None:
        self.ok.append(msg)
        print(f"  [OK]   {msg}")

    def add_warn(self, msg: str) -> None:
        self.warn.append(msg)
        print(f"  [WARN] {msg}")

    def add_fail(self, msg: str) -> None:
        self.fail.append(msg)
        print(f"  [FAIL] {msg}")

    def summary(self) -> int:
        print("\n" + "=" * 60)
        print(f"PREFLIGHT: {len(self.ok)} OK | {len(self.warn)} WARN | {len(self.fail)} FAIL")
        if self.fail:
            print("RESULT: NOT READY — fix FAIL items before full train.")
            return 1
        if self.warn:
            print("RESULT: READY WITH WARNINGS — full train possible, review WARNs.")
            return 0
        print("RESULT: READY — you can start full training.")
        return 0


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def check_env(r: Report) -> None:
    section("1) Python / packages")
    r.add_ok(f"python {sys.version.split()[0]} @ {sys.executable}")
    try:
        import torch

        r.add_ok(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            r.add_ok(f"CUDA GPU: {name} ({mem:.1f} GB)")
            # tiny tensor test
            t0 = time.time()
            x = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            r.add_ok(f"CUDA tensor OK ({(time.time()-t0)*1000:.0f} ms)")
            del x
        else:
            r.add_fail("CUDA not available — train will be extremely slow on CPU")
    except ImportError:
        r.add_fail("torch not installed")
    try:
        import ultralytics

        r.add_ok(f"ultralytics {ultralytics.__version__}")
    except ImportError:
        r.add_fail("ultralytics not installed")
    try:
        from PIL import Image  # noqa: F401

        r.add_ok("Pillow OK")
    except ImportError:
        r.add_fail("Pillow not installed")


def check_disk(r: Report, data_root: Path) -> None:
    section("2) Disk / path")
    if not data_root.exists():
        r.add_fail(f"data root missing: {data_root}")
        return
    r.add_ok(f"data root exists: {data_root.resolve()}")
    try:
        usage = shutil.disk_usage(str(data_root))
        free_gb = usage.free / 1e9
        if free_gb < 5:
            r.add_fail(f"free disk < 5 GB ({free_gb:.1f} GB) — need space for runs/")
        elif free_gb < 20:
            r.add_warn(f"free disk only {free_gb:.1f} GB — tight for cache/runs")
        else:
            r.add_ok(f"free disk {free_gb:.1f} GB")
    except OSError as e:
        r.add_warn(f"disk_usage failed: {e}")

    # filesystem type (Linux)
    try:
        out = subprocess.check_output(
            ["df", "-T", str(data_root)], text=True, stderr=subprocess.DEVNULL
        )
        line = out.strip().splitlines()[-1]
        parts = line.split()
        fstype = parts[1] if len(parts) > 1 else "?"
        r.add_ok(f"filesystem: {fstype}  ({line[:80]}...)")
        if fstype.lower() in {"nfs", "cifs", "smb", "fuse", "overlay"}:
            r.add_warn(
                f"FS type '{fstype}' may be slow/network — prefer copy to local SSD "
                "($HOME/data_local or /tmp) before full train"
            )
    except Exception:
        r.add_warn("could not detect filesystem type (df -T)")


def count_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def check_dataset(r: Report, data_root: Path) -> tuple[list[Path], list[Path]]:
    section("3) Dataset counts & pairing")
    train_img = data_root / "train" / "images"
    train_lab = data_root / "train" / "labels"
    test_img = data_root / "test" / "images"
    test_lab = data_root / "test" / "labels"

    tr_imgs = count_images(train_img)
    te_imgs = count_images(test_img)
    tr_labs = list(train_lab.glob("*.txt")) if train_lab.exists() else []
    te_labs = list(test_lab.glob("*.txt")) if test_lab.exists() else []

    print(f"  train images: {len(tr_imgs)}  labels: {len(tr_labs)}")
    print(f"  test  images: {len(te_imgs)}  labels: {len(te_labs)}")

    if len(tr_imgs) < 10000:
        r.add_fail(f"train images {len(tr_imgs)} < 10000 — not full CCTSDB")
    else:
        r.add_ok(f"train images {len(tr_imgs)} >= 10000")

    if len(te_imgs) < 1000:
        r.add_warn(f"test images {len(te_imgs)} (expected ~1500)")
    else:
        r.add_ok(f"test images {len(te_imgs)}")

    # pairing sample
    tr_lab_stems = {p.stem for p in tr_labs}
    missing = [p for p in tr_imgs if p.stem not in tr_lab_stems]
    if missing:
        r.add_fail(f"train images missing labels: {len(missing)} e.g. {missing[0].name}")
    else:
        r.add_ok("all train images have matching label txt")

    te_lab_stems = {p.stem for p in te_labs}
    missing_te = [p for p in te_imgs if p.stem not in te_lab_stems]
    if missing_te:
        r.add_warn(f"test images missing labels: {len(missing_te)}")
    else:
        r.add_ok("all test images have matching label txt")

    # cache files
    caches = list(data_root.rglob("*.cache"))
    if caches:
        r.add_warn(f"found {len(caches)} .cache file(s) — delete if previous hang: "
                   f"{caches[0].name}")
    else:
        r.add_ok("no leftover .cache files")

    return tr_imgs, te_imgs


def check_labels_sample(r: Report, data_root: Path, n: int = 200) -> None:
    section("4) Label format sample")
    lab_dir = data_root / "train" / "labels"
    files = list(lab_dir.glob("*.txt"))[:n] if lab_dir.exists() else []
    if not files:
        r.add_fail("no train label files to sample")
        return

    bad = 0
    empty = 0
    cls_counts: Counter[int] = Counter()
    for lp in files:
        text = lp.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            empty += 1
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5:
                bad += 1
                continue
            try:
                cid = int(float(parts[0]))
                vals = list(map(float, parts[1:5]))
                if cid not in (0, 1, 2):
                    bad += 1
                if not all(0 <= v <= 1.5 for v in vals):  # allow slight overflow
                    bad += 1
                cls_counts[cid] += 1
            except ValueError:
                bad += 1

    r.add_ok(f"sampled {len(files)} labels; empty={empty} bad_lines={bad}")
    r.add_ok(f"class instances in sample: {dict(sorted(cls_counts.items()))}")
    if bad > len(files) * 0.05:
        r.add_fail("too many bad label lines in sample (>5%)")
    if empty > len(files) * 0.5:
        r.add_warn("many empty labels in sample")


def check_read_speed(r: Report, images: list[Path], n: int = 50) -> None:
    section("5) Image read speed (I/O)")
    if not images:
        r.add_fail("no images for speed test")
        return
    from PIL import Image

    sample = images[:n]
    t0 = time.time()
    ok = 0
    for p in sample:
        try:
            with Image.open(p) as im:
                im.load()
            ok += 1
        except Exception as e:
            r.add_warn(f"cannot read {p.name}: {e}")
    dt = time.time() - t0
    ips = ok / dt if dt > 0 else 0
    ms = (dt / ok * 1000) if ok else 0
    msg = f"read {ok}/{len(sample)} images in {dt:.2f}s → {ips:.1f} img/s ({ms:.0f} ms/img)"
    if ips < 5:
        r.add_warn(msg + " — VERY SLOW disk; copy to local SSD before full train")
    elif ips < 30:
        r.add_warn(msg + " — moderate I/O; expect long label scan")
    else:
        r.add_ok(msg)


def check_ultralytics_load(r: Report, data_yaml: Path, n_probe: int = 8) -> None:
    section("6) Ultralytics mini load (not full 16k scan)")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import ultralytics.utils as u

        u.NUM_THREADS = 1
    except Exception:
        pass

    try:
        from ultralytics import YOLO
        from ultralytics.utils import YAML
    except ImportError as e:
        r.add_fail(f"ultralytics import: {e}")
        return

    if not data_yaml.exists():
        r.add_fail(f"yaml missing: {data_yaml}")
        return

    # load yaml path
    raw = YAML.load(data_yaml)
    path = Path(raw["path"])
    if not path.is_absolute():
        path = (data_yaml.parent / path).resolve()
    train_dir = path / raw["train"]
    imgs = count_images(train_dir)[:n_probe]
    if not imgs:
        r.add_fail(f"no images under {train_dir}")
        return

    try:
        model = YOLO("yolo11n.pt")
        t0 = time.time()
        # predict on a few images — does NOT scan full dataset
        model.predict(source=[str(p) for p in imgs], imgsz=640, verbose=False, device=0)
        r.add_ok(f"predict on {len(imgs)} images OK ({time.time()-t0:.1f}s)")
    except Exception as e:
        r.add_fail(f"ultralytics predict failed: {e}")


def check_smoke_train(r: Report, data_yaml: Path) -> None:
    section("7) Smoke train (fraction=0.01, 1 epoch) — optional")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import ultralytics.utils as u

        u.NUM_THREADS = 1
    except Exception:
        pass

    try:
        from ultralytics import YOLO

        t0 = time.time()
        YOLO("yolo11n.pt").train(
            data=str(data_yaml),
            epochs=1,
            imgsz=640,
            batch=16,
            workers=0,
            cache=False,
            plots=False,
            fraction=0.01,  # ~1% ≈ 160 images
            device=0,
            project="runs/detect",
            name="preflight_smoke",
            exist_ok=True,
            verbose=False,
            patience=1,
        )
        r.add_ok(f"smoke train 1 epoch finished in {time.time()-t0:.0f}s")
    except Exception as e:
        r.add_fail(f"smoke train failed: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight before CCTSDB full train")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/cctsdb2021_full"),
        help="YOLO layout root (train/images, test/images)",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("configs/cctsdb2021_full.yaml"),
    )
    parser.add_argument(
        "--smoke-train",
        action="store_true",
        help="Also run 1-epoch mini train on 1% data (recommended once)",
    )
    parser.add_argument("--read-n", type=int, default=50, help="Images for I/O speed test")
    args = parser.parse_args()

    print("CCTSDB FULL — PREFLIGHT")
    print(f"cwd: {Path.cwd()}")
    r = Report()

    check_env(r)
    check_disk(r, args.data_root)
    tr_imgs, te_imgs = check_dataset(r, args.data_root)
    check_labels_sample(r, args.data_root)
    check_read_speed(r, tr_imgs, n=args.read_n)
    check_ultralytics_load(r, args.data_yaml)

    if args.smoke_train:
        # ensure yaml path points at data_root if custom
        yaml_path = args.data_yaml
        if args.data_root.resolve() != Path("data/processed/cctsdb2021_full").resolve():
            yaml_path = Path("configs/cctsdb2021_full_preflight.yaml")
            yaml_path.write_text(
                f"""path: {args.data_root.resolve().as_posix()}
train: train/images
val: test/images
names:
  0: prohibitory
  1: mandatory
  2: warning
nc: 3
""",
                encoding="utf-8",
            )
            print(f"  using temp yaml {yaml_path}")
        check_smoke_train(r, yaml_path)

    code = r.summary()
    print("\nNext if READY:")
    print("  # optional: copy to fast disk first")
    print("  python scripts/precache_and_train_cctsdb.py --copy-to $HOME/data_local/cctsdb_full")
    print("  # or direct:")
    print("  python scripts/precache_and_train_cctsdb.py --epochs 100 --batch 64")
    print("\nIf FAIL on I/O or smoke: copy dataset to local disk, re-run preflight --data-root ...")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
