#!/usr/bin/env python3
"""Create the clean, reproducible CCTSDB2021 layout from official raw data.

The official raw YOLO IDs are 0=mandatory, 1=prohibitory, 2=warning.  This
project deliberately standardizes them as 0=prohibitory, 1=mandatory,
2=warning, and records the conversion in a manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path


RAW_TO_TARGET = {0: 1, 1: 0, 2: 2}
NAMES = {0: "prohibitory", 1: "mandatory", 2: "warning"}
NAME_TO_TARGET = {name: index for index, name in NAMES.items()}
EXPECTED_DOMAINS = {"sunny": 400, "cloud": 300, "rain": 160, "snow": 100, "foggy": 40, "night": 500}
ARCHIVES = {
    "train_images": "train_img.zip",
    "train_labels": "train_labels.zip",
    "test_images": "test_img.zip",
    "test_labels": "test_labels.zip",
    "weather": "Classification based on weather and environment.zip",
}


def archive_metadata(raw: Path) -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    for label, name in ARCHIVES.items():
        path = raw / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing official archive: {path}")
        stat = path.stat()
        result[label] = {"file": name, "bytes": stat.st_size, "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}
    return result


def extracted_roots(raw: Path) -> dict[str, Path]:
    roots = {
        "train_images": raw / "train_img",
        "train_labels": raw / "train_labels",
        "test_images": raw / "test_img",
        "test_labels": raw / "test_labels",
        "weather": raw / "测试集天气光照情况分类",
    }
    missing = [str(path) for path in roots.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Raw CCTSDB must be either official archives or extracted folders; missing: " + ", ".join(missing))
    return roots


def source_description(raw: Path) -> tuple[str, dict[str, object], dict[str, Path] | None]:
    if all((raw / name).is_file() for name in ARCHIVES.values()):
        return "official_archives", archive_metadata(raw), None
    roots = extracted_roots(raw)
    details: dict[str, object] = {}
    for label, path in roots.items():
        details[label] = {"path": str(path.resolve()), "files": sum(1 for item in path.rglob("*") if item.is_file())}
    return "official_extracted", details, roots


def extract_archives(raw: Path, work: Path) -> dict[str, Path]:
    extracted: dict[str, Path] = {}
    for label, name in ARCHIVES.items():
        destination = work / label
        destination.mkdir()
        with zipfile.ZipFile(raw / name) as archive:
            archive.extractall(destination)
        extracted[label] = destination
    return extracted


def find_dir(work: Path, name: str) -> Path:
    direct = work / name
    if direct.is_dir():
        return direct
    matches = [path for path in work.rglob(name) if path.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not uniquely locate {name} below {work}")
    return matches[0]


def image_files(directory: Path) -> dict[str, Path]:
    files = {path.stem: path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if not files:
        raise RuntimeError(f"No images found in {directory}")
    return files


def label_files(directory: Path) -> dict[str, Path]:
    files = {path.stem: path for path in directory.rglob("*.txt") if path.name.lower() != "classes.txt"}
    if not files:
        raise RuntimeError(f"No labels found in {directory}")
    return files


def remap_label(source: Path) -> tuple[str, Counter[int]]:
    output: list[str] = []
    counts: Counter[int] = Counter()
    for line in source.read_text(encoding="utf-8", errors="strict").splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label row in {source}: {line!r}")
        raw_id = int(parts[0])
        if raw_id not in RAW_TO_TARGET:
            raise ValueError(f"Unexpected raw class {raw_id} in {source}")
        target = RAW_TO_TARGET[raw_id]
        values = [float(value) for value in parts[1:]]
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Out-of-range YOLO coordinates in {source}")
        output.append(f"{target} " + " ".join(f"{value:.6f}" for value in values))
        counts[target] += 1
    return "\n".join(output) + ("\n" if output else ""), counts


def weather_assignments(weather_root: Path, test_stems: set[str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for xml_path in weather_root.rglob("*.xml"):
        domain = next((part.lower() for part in xml_path.parts if part.lower() in EXPECTED_DOMAINS), None)
        if domain is None:
            continue
        stem = xml_path.stem
        if stem not in test_stems:
            raise RuntimeError(f"Weather XML has no matching test image: {xml_path}")
        prior = assignments.setdefault(stem, domain)
        if prior != domain:
            raise RuntimeError(f"Conflicting weather labels for {stem}: {prior} and {domain}")
    counts = Counter(assignments.values())
    if dict(counts) != EXPECTED_DOMAINS or len(assignments) != len(test_stems):
        raise RuntimeError(f"Unexpected weather partition: {dict(counts)}, assigned={len(assignments)}, test={len(test_stems)}")
    return assignments


def validate_mapping(weather_root: Path, raw_test_labels: dict[str, Path]) -> dict[str, dict[str, int]]:
    """Verify raw IDs against named objects in official weather XML annotations."""
    raw_counts: Counter[int] = Counter()
    xml_counts: Counter[int] = Counter()
    for xml_path in weather_root.rglob("*.xml"):
        stem = xml_path.stem
        label = raw_test_labels.get(stem)
        if label is None:
            raise RuntimeError(f"Missing raw test label for weather XML {xml_path}")
        _, remapped = remap_label(label)
        raw_counts.update(remapped)
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name not in NAME_TO_TARGET:
                raise RuntimeError(f"Unknown XML class {name!r} in {xml_path}")
            xml_counts[NAME_TO_TARGET[name]] += 1
    if raw_counts != xml_counts:
        raise RuntimeError(f"Raw-ID mapping does not match XML semantics: raw={dict(raw_counts)}, xml={dict(xml_counts)}")
    return {"remapped_raw": {str(k): raw_counts[k] for k in sorted(raw_counts)}, "xml_semantic": {str(k): xml_counts[k] for k in sorted(xml_counts)}}


def choose_dev(stems: list[str], seed: int, ratio: float) -> set[str]:
    count = round(len(stems) * ratio)
    if not 1 <= count < len(stems):
        raise ValueError("dev ratio must produce a non-empty proper subset")
    ordered = sorted(stems, key=lambda stem: (hashlib.sha256(f"{seed}:{stem}".encode()).hexdigest(), stem))
    return set(ordered[:count])


def materialize_split(images: dict[str, Path], labels: dict[str, Path], stems: list[str], destination: Path, move_images: bool) -> Counter[int]:
    image_out, label_out = destination / "images", destination / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    class_counts: Counter[int] = Counter()
    for stem in stems:
        image, label = images.get(stem), labels.get(stem)
        if image is None or label is None:
            raise RuntimeError(f"Missing image or label for {stem}")
        target_image = image_out / f"{stem}{image.suffix.lower()}"
        if move_images:
            shutil.move(str(image), target_image)
        else:
            try:
                os.link(image, target_image)
            except OSError:
                shutil.copy2(image, target_image)
        converted, counts = remap_label(label)
        (label_out / f"{stem}.txt").write_text(converted, encoding="utf-8")
        class_counts.update(counts)
    return class_counts


def link_domain(source: Path, destination: Path, stems: list[str]) -> Counter[int]:
    return materialize_split(
        {path.stem: path for path in (source / "images").glob("*")},
        {path.stem: path for path in (source / "labels").glob("*.txt")},
        stems,
        destination,
        move_images=False,
    )


def remove_output(path: Path) -> None:
    if not path.exists():
        return
    expected = (Path("data") / "processed" / "cctsdb2021_clean").resolve()
    if path.resolve() != expected:
        raise RuntimeError(f"Refusing to remove unexpected output path: {path.resolve()}")
    shutil.rmtree(path)


def assert_split_integrity(root: Path, expected_images: int) -> None:
    images = list((root / "images").glob("*"))
    labels = list((root / "labels").glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    if len(images) != expected_images or len(labels) != expected_images or image_stems != label_stems:
        raise RuntimeError(
            f"Incomplete materialization at {root}: images={len(images)}, labels={len(labels)}, "
            f"missing_images={len(label_stems - image_stems)}, missing_labels={len(image_stems - label_stems)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare clean CCTSDB2021 data from official archives or extracted folders")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/cctsdb2021_clean"))
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Replace only the default clean output directory.")
    args = parser.parse_args()

    source_kind, source_metadata, direct_roots = source_description(args.raw)
    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"{args.output} exists; rerun with --force")
        remove_output(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="cctsdb_build_", dir=args.output.parent) if source_kind == "official_archives" else nullcontext(None)
    with temporary as temp_name:
        extracted = extract_archives(args.raw, Path(temp_name)) if temp_name is not None else direct_roots
        assert extracted is not None
        train_images = image_files(extracted["train_images"])
        train_labels = label_files(extracted["train_labels"])
        test_images = image_files(extracted["test_images"])
        test_labels = label_files(extracted["test_labels"])
        weather_root = find_dir(extracted["weather"], "测试集天气光照情况分类")
        if len(train_images) != 16356 or len(train_labels) != 16356 or len(test_images) != 1500 or len(test_labels) != 1500:
            raise RuntimeError(f"Unexpected raw counts: train images/labels={len(train_images)}/{len(train_labels)}, test={len(test_images)}/{len(test_labels)}")
        assignments = weather_assignments(weather_root, set(test_images))
        mapping_check = validate_mapping(weather_root, test_labels)
        dev_stems = choose_dev(sorted(train_images), args.seed, args.dev_ratio)
        train_stems = sorted(set(train_images) - dev_stems)
        args.output.mkdir(parents=True)
        move_temporary_images = source_kind == "official_archives"
        split_counts = {
            "train": materialize_split(train_images, train_labels, train_stems, args.output / "train", move_images=move_temporary_images),
            "dev": materialize_split(train_images, train_labels, sorted(dev_stems), args.output / "dev", move_images=move_temporary_images),
            "test": materialize_split(test_images, test_labels, sorted(test_images), args.output / "test", move_images=move_temporary_images),
        }
        assert_split_integrity(args.output / "train", len(train_stems))
        assert_split_integrity(args.output / "dev", len(dev_stems))
        assert_split_integrity(args.output / "test", len(test_images))
        domain_counts: dict[str, Counter[int]] = {}
        for domain in EXPECTED_DOMAINS:
            stems = sorted(stem for stem, assigned in assignments.items() if assigned == domain)
            domain_counts[domain] = link_domain(args.output / "test", args.output / "domains" / domain, stems)
            assert_split_integrity(args.output / "domains" / domain, len(stems))
        daylike = sorted(stem for stem, assigned in assignments.items() if assigned != "night")
        domain_counts["daylike"] = link_domain(args.output / "test", args.output / "domains" / "daylike", daylike)
        assert_split_integrity(args.output / "domains" / "daylike", len(daylike))
        manifest_dir = args.output / "manifests"
        manifest_dir.mkdir()
        with (manifest_dir / "test_weather.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stem", "domain", "aggregate"])
            writer.writeheader()
            for stem, domain in sorted(assignments.items()):
                writer.writerow({"stem": stem, "domain": domain, "aggregate": "night" if domain == "night" else "daylike"})
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": {"kind": source_kind, "items": source_metadata},
            "class_mapping": {"source_raw": {"0": "mandatory", "1": "prohibitory", "2": "warning"}, "target": {str(k): v for k, v in NAMES.items()}, "raw_to_target": {str(k): v for k, v in RAW_TO_TARGET.items()}},
            "mapping_validation": mapping_check,
            "development_split": {"seed": args.seed, "ratio": args.dev_ratio, "method": "deterministic SHA-256 ranking by image stem; not a video-grouped split"},
            "images": {"train": len(train_stems), "dev": len(dev_stems), "official_test": len(test_images)},
            "class_instances": {name: {str(k): counts[k] for k in sorted(counts)} for name, counts in split_counts.items()},
            "weather_images": {**EXPECTED_DOMAINS, "daylike": len(daylike)},
            "weather_class_instances": {name: {str(k): counts[k] for k in sorted(counts)} for name, counts in domain_counts.items()},
        }
        (manifest_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared {args.output}")
    print(json.dumps(manifest["images"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
