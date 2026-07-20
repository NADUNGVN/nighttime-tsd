#!/usr/bin/env python3
"""Extract CCTSDB2021 night-condition test images into YOLO layout.

CCTSDB2021 ships a package:
  "classification based on weather and environment"
with XML annotations for the 1500 positive test images, labeled by
weather/lighting. This script finds night/dark/low-light XMLs and
copies matching images + converts boxes to YOLO txt (3 classes).

Manual step first:
  1. Download from https://github.com/csust7zhangjm/CCTSDB2021
     Google Drive: https://drive.google.com/drive/folders/14Km2W-5hbixXDfz7WSqW_Rx7O5m8ZMFn
  2. Unpack into data/raw/CCTSDB2021/ with at least:
       - test images (test_img or similar)
       - weather/environment classification XMLs
       - optional: train_labels for class-name mapping

Class names in CCTSDB are typically: prohibitory, mandatory, warning
(same 3 super-classes as CNTSSS).
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Keywords in path/filename/XML that suggest nighttime
NIGHT_PATTERNS = re.compile(
    r"(night|dark|low.?light|evening|dusk|dim|nighttime|nigh)",
    re.IGNORECASE,
)

# Map common Chinese/English class names → id 0/1/2 (align with CNTSSS)
CLASS_ALIASES = {
    "prohibitory": 0,
    "prohibition": 0,
    "forbidden": 0,
    "mandatory": 1,
    "mandatorysign": 1,
    "warning": 2,
    "warning sign": 2,
    # Chinese common names in CCTSDB papers
    "禁止": 0,
    "指示": 1,
    "警告": 2,
}


def find_images(root: Path) -> dict[str, Path]:
    """stem -> path for all images under root."""
    out: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            out[p.stem] = p
    return out


def find_xmls(root: Path) -> list[Path]:
    return sorted(root.rglob("*.xml"))


def xml_is_night(xml_path: Path, tree: ET.ElementTree) -> bool:
    """Heuristic: night if path or any text node matches night patterns."""
    if NIGHT_PATTERNS.search(str(xml_path)):
        return True
    root = tree.getroot()
    # folder / filename / condition tags
    for tag in ("folder", "filename", "condition", "weather", "time", "lighting", "scene"):
        el = root.find(tag)
        if el is not None and el.text and NIGHT_PATTERNS.search(el.text):
            return True
    # any descendant text
    for el in root.iter():
        if el.text and NIGHT_PATTERNS.search(el.text):
            return True
        if el.tail and NIGHT_PATTERNS.search(el.tail):
            return True
    return False


def parse_voc_boxes(tree: ET.ElementTree) -> list[tuple[str, int, int, int, int]]:
    """Return list of (class_name, xmin, ymin, xmax, ymax)."""
    boxes = []
    root = tree.getroot()
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bb = obj.find("bndbox")
        if name_el is None or bb is None or name_el.text is None:
            continue
        name = name_el.text.strip()
        try:
            xmin = int(float(bb.findtext("xmin", "0")))
            ymin = int(float(bb.findtext("ymin", "0")))
            xmax = int(float(bb.findtext("xmax", "0")))
            ymax = int(float(bb.findtext("ymax", "0")))
        except (TypeError, ValueError):
            continue
        boxes.append((name, xmin, ymin, xmax, ymax))
    return boxes


def class_to_id(name: str) -> int | None:
    key = name.strip().lower()
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    # fuzzy
    for k, v in CLASS_ALIASES.items():
        if k in key or key in k:
            return v
    return None


def voc_to_yolo(
    boxes: list[tuple[str, int, int, int, int]], w: int, h: int
) -> list[str]:
    lines = []
    for name, xmin, ymin, xmax, ymax in boxes:
        cid = class_to_id(name)
        if cid is None:
            continue
        bw = max(0, xmax - xmin)
        bh = max(0, ymax - ymin)
        cx = xmin + bw / 2
        cy = ymin + bh / 2
        lines.append(
            f"{cid} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"
        )
    return lines


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size  # w, h
    except Exception:
        import cv2

        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Cannot read image: {path}")
        h, w = img.shape[:2]
        return w, h


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract CCTSDB2021 night test subset")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/raw/CCTSDB2021"),
        help="Unpacked CCTSDB2021 root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/cctsdb2021_night"),
        help="Output YOLO-layout folder",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list night candidates, do not copy",
    )
    parser.add_argument(
        "--force-all-test-xml",
        action="store_true",
        help="If night heuristics fail, print structure help",
    )
    args = parser.parse_args()

    if not args.src.exists():
        print(f"ERROR: {args.src} not found.")
        print("Download CCTSDB2021 (Google Drive / Baidu) and unpack there.")
        print("  https://github.com/csust7zhangjm/CCTSDB2021")
        return 1

    print(f"Scanning images under {args.src} ...")
    images = find_images(args.src)
    print(f"  found {len(images)} images")

    print(f"Scanning XMLs under {args.src} ...")
    xmls = find_xmls(args.src)
    print(f"  found {len(xmls)} XML files")

    # Prefer weather/environment package if present
    weather_xmls = [
        x
        for x in xmls
        if re.search(r"weather|environment|condition|classif", str(x), re.I)
    ]
    scan_xmls = weather_xmls if weather_xmls else xmls
    print(f"  scanning {len(scan_xmls)} XMLs for night condition")

    night_stems: list[str] = []
    night_xml_map: dict[str, Path] = {}
    unmapped_classes: set[str] = set()
    parent_folders: dict[str, int] = {}

    for xp in scan_xmls:
        try:
            tree = ET.parse(xp)
        except ET.ParseError:
            continue
        if not xml_is_night(xp, tree):
            continue
        stem = xp.stem
        # VOC often uses filename tag
        fn = tree.getroot().findtext("filename")
        if fn:
            stem = Path(fn).stem
        night_stems.append(stem)
        night_xml_map[stem] = xp
        parent = str(xp.parent.relative_to(args.src)) if xp.is_relative_to(args.src) else str(xp.parent)
        parent_folders[parent] = parent_folders.get(parent, 0) + 1

    night_stems = sorted(set(night_stems))
    print(f"\nNight candidates: {len(night_stems)}")
    if parent_folders:
        print("  by folder:")
        for k, v in sorted(parent_folders.items(), key=lambda x: -x[1])[:15]:
            print(f"    {v:4d}  {k}")

    matched = [s for s in night_stems if s in images]
    missing_img = [s for s in night_stems if s not in images]
    print(f"  matched images: {len(matched)} | missing image files: {len(missing_img)}")
    if missing_img[:5]:
        print(f"  sample missing: {missing_img[:5]}")

    if len(matched) == 0:
        print("\nWARN: No night images matched. Inspect folder names under:")
        # show unique parent names of all xml
        parents = sorted({str(x.parent) for x in xmls})[:30]
        for p in parents:
            print(f"  {p}")
        print("\nTip: open a few XMLs and note the tag used for lighting condition,")
        print("then adjust NIGHT_PATTERNS in this script.")
        if args.force_all_test_xml:
            print("--force-all-test-xml set but still no match heuristic — aborting copy.")
        return 2

    if args.list_only:
        for s in matched[:20]:
            print(f"  {s} <- {night_xml_map[s]}")
        if len(matched) > 20:
            print(f"  ... +{len(matched) - 20} more")
        return 0

    out_img = args.out / "images"
    out_lab = args.out / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_cls = 0
    for stem in matched:
        src_im = images[stem]
        xp = night_xml_map[stem]
        try:
            tree = ET.parse(xp)
            boxes = parse_voc_boxes(tree)
            for name, *_ in boxes:
                if class_to_id(name) is None:
                    unmapped_classes.add(name)
            w, h = image_size(src_im)
            lines = voc_to_yolo(boxes, w, h)
            if not lines and boxes:
                skipped_cls += 1
            # copy image
            dst_im = out_img / f"{stem}{src_im.suffix.lower()}"
            if not dst_im.exists():
                shutil.copy2(src_im, dst_im)
            (out_lab / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            written += 1
        except Exception as e:
            print(f"  FAIL {stem}: {e}", file=sys.stderr)

    # write list file
    list_path = args.out / "night_list.txt"
    list_path.write_text("\n".join(matched) + "\n", encoding="utf-8")

    print(f"\nWrote {written} images+labels → {args.out}")
    print(f"  Expected ~500 night test (paper plan); got {written}")
    if unmapped_classes:
        print(f"  Unmapped class names (fix CLASS_ALIASES): {sorted(unmapped_classes)}")
    if skipped_cls:
        print(f"  Images with boxes but no mapped class: {skipped_cls}")
    print("Next: point configs/cctsdb2021_night.yaml path and run val.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
