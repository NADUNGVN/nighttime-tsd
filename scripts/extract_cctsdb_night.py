#!/usr/bin/env python3
"""Extract CCTSDB2021 night-condition test images into YOLO layout.

CCTSDB2021 layout (after unzip on server):
  test_img/*.jpg          — test images numbered ~18992–20491 (1500)
  weather_env/.../night/  — 500 VOC XMLs (stems may be 00552… not image ids)
  xml/xml/*.xml           — full VOC for all images (prefer for boxes)

Night XML stems often do NOT match image filenames. Resolution order:
  1) <filename> inside night XML
  2) exact stem match
  3) if stem is integer i in [0, 1500): image id 18992+i (official test range start)
  4) if stem is integer, try raw int as image stem (zero-pad variants)

Usage:
  python scripts/extract_cctsdb_night.py --list-only
  python scripts/extract_cctsdb_night.py
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

NIGHT_PATTERNS = re.compile(
    r"(night|dark|low.?light|evening|dusk|dim|nighttime|/night[\\/])",
    re.IGNORECASE,
)

# CCTSDB2021 positive test set numbering (README)
TEST_ID_START = 18992
TEST_ID_END = 20491  # inclusive
TEST_COUNT = 1500

CLASS_ALIASES = {
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
    # numeric class ids sometimes appear
    "0": 0,
    "1": 1,
    "2": 2,
}


def find_images(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            out[p.stem] = p
            # also index unpadded / padded numeric stems
            if p.stem.isdigit():
                out[str(int(p.stem))] = p
                out[f"{int(p.stem):05d}"] = p
    return out


def find_xmls(root: Path) -> list[Path]:
    return sorted(root.rglob("*.xml"))


def is_night_path(xml_path: Path) -> bool:
    """True if path clearly under a night condition folder."""
    parts = [x.lower() for x in xml_path.parts]
    # exact folder name night (weather package)
    if "night" in parts:
        return True
    return bool(NIGHT_PATTERNS.search(str(xml_path)))


def parse_voc_boxes(tree: ET.ElementTree) -> list[tuple[str, int, int, int, int]]:
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
        lines.append(f"{cid} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    return lines


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        import cv2

        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Cannot read image: {path}")
        h, w = img.shape[:2]
        return w, h


def resolve_image_stem(
    xml_path: Path, tree: ET.ElementTree, images: dict[str, Path]
) -> tuple[str | None, str]:
    """Return (matched_stem_in_images_dict, method) or (None, reason)."""
    candidates: list[tuple[str, str]] = []

    fn = tree.getroot().findtext("filename")
    if fn:
        candidates.append((Path(fn.strip()).stem, "xml.filename"))

    candidates.append((xml_path.stem, "xml.stem"))

    # numeric strategies
    stem = xml_path.stem
    if stem.isdigit():
        i = int(stem)
        candidates.append((str(i), "int"))
        candidates.append((f"{i:05d}", "int05"))
        # test-set index → absolute id (18992 + i)
        if 0 <= i < TEST_COUNT:
            candidates.append((str(TEST_ID_START + i), "test_offset"))
            candidates.append((f"{TEST_ID_START + i:05d}", "test_offset05"))
        # already absolute test id
        if TEST_ID_START <= i <= TEST_ID_END:
            candidates.append((str(i), "test_abs"))
            candidates.append((f"{i:05d}", "test_abs05"))

    for cand, method in candidates:
        if cand in images:
            return cand, method
    return None, "no_match:" + ",".join(c for c, _ in candidates[:6])


def find_label_xml(
    img_stem: str, night_xml: Path, all_xml_by_stem: dict[str, Path]
) -> Path:
    """Prefer full VOC under xml/ for boxes; fall back to night xml."""
    for key in (img_stem, f"{int(img_stem):05d}" if img_stem.isdigit() else img_stem):
        if key in all_xml_by_stem:
            # prefer non-weather full annotations if multiple
            p = all_xml_by_stem[key]
            return p
    # any xml with this stem
    return night_xml


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract CCTSDB2021 night test subset")
    parser.add_argument("--src", type=Path, default=Path("data/raw/CCTSDB2021"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/cctsdb2021_night"))
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--night-glob",
        default="**/night/**/*.xml",
        help="Glob under --src for night XMLs (default: **/night/**/*.xml)",
    )
    args = parser.parse_args()

    if not args.src.exists():
        print(f"ERROR: {args.src} not found.")
        return 1

    print(f"Scanning images under {args.src} ...")
    images = find_images(args.src)
    # unique paths
    n_unique = len({id(p) for p in images.values()})
    print(f"  indexed {len(images)} stem keys → {n_unique} unique files")
    sample_imgs = sorted({p.name for p in images.values()})[:5]
    print(f"  sample images: {sample_imgs}")

    # Night XMLs: prefer folder named night
    night_xmls = sorted(args.src.glob(args.night_glob))
    if not night_xmls:
        # fallback: any xml whose path matches night
        night_xmls = [x for x in find_xmls(args.src) if is_night_path(x)]
    print(f"Night XMLs: {len(night_xmls)}")

    if night_xmls:
        # debug first xml
        try:
            t0 = ET.parse(night_xmls[0])
            print(
                f"  sample night xml: {night_xmls[0].name} "
                f"filename={t0.getroot().findtext('filename')!r} "
                f"objects={len(t0.getroot().findall('object'))}"
            )
        except ET.ParseError as e:
            print(f"  sample parse fail: {e}")

    all_xml_by_stem: dict[str, Path] = {}
    for xp in find_xmls(args.src):
        # prefer xml/ tree over weather for same stem (longer path with /xml/)
        stem = xp.stem
        prev = all_xml_by_stem.get(stem)
        if prev is None:
            all_xml_by_stem[stem] = xp
        elif "weather" in str(prev).lower() and "weather" not in str(xp).lower():
            all_xml_by_stem[stem] = xp
        if stem.isdigit():
            all_xml_by_stem[str(int(stem))] = all_xml_by_stem.get(
                str(int(stem)), all_xml_by_stem[stem]
            )

    matched: list[tuple[str, Path, str]] = []  # img_stem, night_xml, method
    missing = []
    methods: dict[str, int] = {}

    for xp in night_xmls:
        try:
            tree = ET.parse(xp)
        except ET.ParseError:
            missing.append((xp.stem, "parse_error"))
            continue
        img_stem, method = resolve_image_stem(xp, tree, images)
        if img_stem is None:
            missing.append((xp.stem, method))
            continue
        methods[method] = methods.get(method, 0) + 1
        matched.append((img_stem, xp, method))

    # dedupe by image stem (keep first)
    seen = set()
    deduped = []
    for item in matched:
        if item[0] in seen:
            continue
        seen.add(item[0])
        deduped.append(item)
    matched = deduped

    print(f"\nMatched images: {len(matched)} | unresolved XMLs: {len(missing)}")
    print(f"  resolve methods: {methods}")
    if missing[:5]:
        print(f"  sample unresolved: {missing[:5]}")
    if matched[:5]:
        print("  sample matched:")
        for s, xp, m in matched[:5]:
            print(f"    {s}  via {m}  <- {xp.name}")

    if not matched:
        print("\nFAIL: still 0 matches. Debug with:")
        print("  ls data/raw/CCTSDB2021/test_img | head")
        print("  ls \"data/raw/CCTSDB2021/weather_env\"/*/night | head")
        print("  head -30 one night xml")
        return 2

    if args.list_only:
        return 0

    out_img = args.out / "images"
    out_lab = args.out / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    unmapped_classes: set[str] = set()
    written = 0
    empty_boxes = 0

    for img_stem, night_xml, _method in matched:
        src_im = images[img_stem]
        # label xml: try full VOC first
        label_xml = night_xml
        for key in (
            img_stem,
            f"{int(img_stem):05d}" if img_stem.isdigit() else None,
            night_xml.stem,
        ):
            if key and key in all_xml_by_stem:
                cand = all_xml_by_stem[key]
                # prefer non-weather
                if "weather" not in str(cand).lower() or "weather" in str(label_xml).lower():
                    label_xml = cand
                    if "weather" not in str(cand).lower():
                        break

        try:
            tree = ET.parse(label_xml)
            boxes = parse_voc_boxes(tree)
            if not boxes:
                # retry night xml
                tree = ET.parse(night_xml)
                boxes = parse_voc_boxes(tree)
            for name, *_ in boxes:
                if class_to_id(name) is None:
                    unmapped_classes.add(name)
            w, h = image_size(src_im)
            lines = voc_to_yolo(boxes, w, h)
            if not lines:
                empty_boxes += 1
            dst_im = out_img / f"{img_stem}{src_im.suffix.lower()}"
            if not dst_im.exists():
                shutil.copy2(src_im, dst_im)
            (out_lab / f"{img_stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            written += 1
        except Exception as e:
            print(f"  FAIL {img_stem}: {e}", file=sys.stderr)

    (args.out / "night_list.txt").write_text(
        "\n".join(s for s, _, _ in matched) + "\n", encoding="utf-8"
    )

    print(f"\nWrote {written} images+labels → {args.out}")
    print(f"  Expected ~500 night; got {written}")
    print(f"  empty labels: {empty_boxes}")
    if unmapped_classes:
        print(f"  Unmapped class names: {sorted(unmapped_classes)}")
        print("  → fix CLASS_ALIASES if many empty labels")
    print("Next:")
    print("  python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_night.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
