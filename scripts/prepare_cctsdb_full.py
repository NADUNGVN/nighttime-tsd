#!/usr/bin/env python3
"""Prepare CCTSDB2021 FULL into YOLO layout for training (>10k images).

Expected after unzip under data/raw/CCTSDB2021/:
  train_img/   or train_img/**/*.jpg
  test_img/
  train_labels/*.txt   (YOLO txt)  OR  xml/**/*.xml
  test_labels/*.txt    OR  xml/**/*.xml

Output:
  data/processed/cctsdb2021_full/
    train/{images,labels}
    test/{images,labels}

Class ids remapped to CNTSSS order:
  0 prohibitory | 1 mandatory | 2 warning

Usage:
  # 1) unzip zips first (see docs below)
  python scripts/prepare_cctsdb_full.py
  python scripts/prepare_cctsdb_full.py --verify-only
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# name / possible source class id → target id (CNTSSS)
NAME_TO_ID = {
    "prohibitory": 0,
    "prohibition": 0,
    "forbidden": 0,
    "mandatory": 1,
    "mandatorysign": 1,
    "warning": 2,
    "warning sign": 2,
    "禁止": 0,
    "指示": 1,
    "警告": 2,
    "0": 0,
    "1": 1,
    "2": 2,
}

# Some CCTSDB releases use different id order; adjust via --id-map if needed
# Default assume already 0/1/2 = prohibitory/mandatory/warning (common)
DEFAULT_ID_MAP = {0: 0, 1: 1, 2: 2}


def find_images(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p
            if p.stem.isdigit():
                out[str(int(p.stem))] = p
                out[f"{int(p.stem):05d}"] = p
    return out


def find_txt_labels(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for p in root.rglob("*.txt"):
        if p.name.lower() in {"classes.txt", "readme.txt"}:
            continue
        out[p.stem] = p
        if p.stem.isdigit():
            out[str(int(p.stem))] = p
            out[f"{int(p.stem):05d}"] = p
    return out


def find_xml_labels(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for p in root.rglob("*.xml"):
        # skip weather classification-only trees if full xml exists elsewhere
        out[p.stem] = p
        if p.stem.isdigit():
            out[str(int(p.stem))] = p
            out[f"{int(p.stem):05d}"] = p
    return out


def parse_yolo_txt(path: Path, id_map: dict[int, int]) -> list[str]:
    lines_out = []
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return lines_out
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cid = id_map.get(cid, cid)
            if cid not in (0, 1, 2):
                # try name
                continue
            x, y, w, h = map(float, parts[1:5])
            lines_out.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        except ValueError:
            continue
    return lines_out


def parse_voc_xml(path: Path, img_w: int, img_h: int) -> list[str]:
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return []
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bb = obj.find("bndbox")
        if name_el is None or bb is None or not name_el.text:
            continue
        name = name_el.text.strip()
        key = name.lower()
        cid = NAME_TO_ID.get(key)
        if cid is None:
            for k, v in NAME_TO_ID.items():
                if k in key:
                    cid = v
                    break
        if cid is None and name.isdigit():
            cid = int(name)
            if cid not in (0, 1, 2):
                continue
        if cid is None:
            continue
        try:
            xmin = float(bb.findtext("xmin", "0"))
            ymin = float(bb.findtext("ymin", "0"))
            xmax = float(bb.findtext("xmax", "0"))
            ymax = float(bb.findtext("ymax", "0"))
        except (TypeError, ValueError):
            continue
        bw = max(0.0, xmax - xmin)
        bh = max(0.0, ymax - ymin)
        cx = xmin + bw / 2
        cy = ymin + bh / 2
        lines.append(
            f"{cid} {cx / img_w:.6f} {cy / img_h:.6f} {bw / img_w:.6f} {bh / img_h:.6f}"
        )
    return lines


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def resolve_split_dirs(src: Path) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """Return train_img, test_img, train_lab, test_lab roots (best effort)."""
    candidates_train_img = [
        src / "train_img",
        src / "train" / "images",
        src / "DATASET" / "train",
        src / "train",
    ]
    candidates_test_img = [
        src / "test_img",
        src / "test" / "images",
        src / "test",
    ]
    candidates_train_lab = [
        src / "train_labels",
        src / "train" / "labels",
        src / "xml" / "xml",
        src / "xml",
    ]
    candidates_test_lab = [
        src / "test_labels",
        src / "test" / "labels",
        src / "xml" / "xml",
        src / "xml",
    ]

    def first_with_files(paths: list[Path], pattern: str) -> Path | None:
        for p in paths:
            if p.exists() and any(p.rglob(pattern)):
                return p
        return None

    tr_i = first_with_files(candidates_train_img, "*.jpg") or first_with_files(
        candidates_train_img, "*.png"
    )
    te_i = first_with_files(candidates_test_img, "*.jpg") or first_with_files(
        candidates_test_img, "*.png"
    )
    tr_l = first_with_files(candidates_train_lab, "*.txt") or first_with_files(
        candidates_train_lab, "*.xml"
    )
    te_l = first_with_files(candidates_test_lab, "*.txt") or first_with_files(
        candidates_test_lab, "*.xml"
    )
    return tr_i, te_i, tr_l, te_l


def copy_split(
    name: str,
    img_dir: Path,
    lab_dir: Path | None,
    out_root: Path,
    id_map: dict[int, int],
    xml_fallback: Path | None,
) -> dict:
    images = find_images(img_dir)
    # unique by path
    unique_imgs = {p.resolve(): s for s, p in images.items()}
    # prefer 5-digit stem keys from path
    stem_to_img: dict[str, Path] = {}
    for p in unique_imgs:
        stem_to_img[p.stem] = p

    txt_labs = find_txt_labels(lab_dir) if lab_dir else {}
    xml_labs = find_xml_labels(lab_dir) if lab_dir else {}
    if xml_fallback and xml_fallback != lab_dir:
        for k, v in find_xml_labels(xml_fallback).items():
            xml_labs.setdefault(k, v)

    out_img = out_root / name / "images"
    out_lab = out_root / name / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_no_lab = 0
    n_empty = 0
    class_counts: Counter[int] = Counter()

    for stem, src_im in sorted(stem_to_img.items(), key=lambda x: x[0]):
        keys = [stem]
        if stem.isdigit():
            keys += [str(int(stem)), f"{int(stem):05d}"]

        lines: list[str] = []
        # prefer txt
        for k in keys:
            if k in txt_labs:
                lines = parse_yolo_txt(txt_labs[k], id_map)
                break
        if not lines:
            for k in keys:
                if k in xml_labs:
                    w, h = image_size(src_im)
                    lines = parse_voc_xml(xml_labs[k], w, h)
                    break

        if not any(k in txt_labs or k in xml_labs for k in keys):
            n_no_lab += 1
            continue

        dst = out_img / f"{src_im.stem}{src_im.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src_im, dst)
        (out_lab / f"{src_im.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        if not lines:
            n_empty += 1
        for ln in lines:
            class_counts[int(ln.split()[0])] += 1
        n_ok += 1

    return {
        "split": name,
        "images": n_ok,
        "no_label_skipped": n_no_lab,
        "empty_label": n_empty,
        "class_instances": dict(sorted(class_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("data/raw/CCTSDB2021"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/cctsdb2021_full"))
    parser.add_argument(
        "--id-map",
        default="0:0,1:1,2:2",
        help="source_id:target_id pairs, e.g. 0:0,1:1,2:2",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--symlink", action="store_true", help="symlink images if possible (Linux)")
    args = parser.parse_args()

    id_map = DEFAULT_ID_MAP.copy()
    for pair in args.id_map.split(","):
        a, b = pair.split(":")
        id_map[int(a)] = int(b)

    if not args.src.exists():
        print(f"ERROR: {args.src} missing")
        print("Unzip train_img.zip test_img.zip train_labels.zip test_labels.zip xml.zip first")
        return 1

    tr_i, te_i, tr_l, te_l = resolve_split_dirs(args.src)
    print("Resolved paths:")
    print(f"  train images : {tr_i}")
    print(f"  test images  : {te_i}")
    print(f"  train labels : {tr_l}")
    print(f"  test labels  : {te_l}")

    if tr_i is None or te_i is None:
        print("\nERROR: need train_img + test_img unpacked.")
        print("  cd data/raw/CCTSDB2021")
        print("  unzip -q train_img.zip -d train_img")
        print("  unzip -q test_img.zip -d test_img")
        print("  unzip -q train_labels.zip -d train_labels")
        print("  unzip -q test_labels.zip -d test_labels")
        print("  unzip -q xml.zip -d xml")
        return 1

    if args.verify_only:
        print(f"train images count: {len(set(find_images(tr_i).values()))}")
        print(f"test images count : {len(set(find_images(te_i).values()))}")
        return 0

    xml_fb = args.src / "xml"
    if not xml_fb.exists():
        xml_fb = None

    if args.out.exists():
        print(f"Output exists, merging/updating under {args.out}")

    r_train = copy_split("train", tr_i, tr_l, args.out, id_map, xml_fb)
    r_test = copy_split("test", te_i, te_l or tr_l, args.out, id_map, xml_fb)

    print("\n=== RESULT ===")
    for r in (r_train, r_test):
        print(
            f"{r['split']}: images={r['images']}  "
            f"no_label_skip={r['no_label_skipped']}  empty={r['empty_label']}  "
            f"cls={r['class_instances']}"
        )

    n_train = r_train["images"]
    n_test = r_test["images"]
    print(f"\nTOTAL usable: train={n_train} test={n_test} sum={n_train + n_test}")
    if n_train < 10000:
        print("WARN: train < 10k — check train_img fully unzipped and labels match.")
    else:
        print("OK: train >= 10k images.")

    print("\nTrain next:")
    print(
        "  python scripts/train_baseline.py --model yolo11n.pt "
        "--data configs/cctsdb2021_full.yaml --batch 64 --epochs 100 "
        "--name yolo11n_cctsdb_full --cache ram"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
