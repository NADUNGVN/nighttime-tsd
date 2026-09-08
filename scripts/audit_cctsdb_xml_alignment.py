#!/usr/bin/env python3
"""Read-only audit of the ID alignment between CCTSDB xml.zip and full-test predictions."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path


def normalized(value: str | None) -> str | None:
    if not value:
        return None
    return Path(value.strip()).stem or None


def sample(values: set[str], limit: int = 10) -> list[str]:
    return sorted(values)[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CCTSDB XML-to-prediction identifier alignment without creating results")
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    prediction_names = [str(record["image"]) for record in records]
    prediction_stems = {normalized(name) for name in prediction_names}
    if None in prediction_stems or len(prediction_stems) != len(prediction_names):
        raise ValueError("Prediction image IDs are missing or non-unique by stem")
    filename_stems: set[str] = set()
    path_stems: set[str] = set()
    member_stems: set[str] = set()
    samples: list[dict[str, str | None]] = []
    with zipfile.ZipFile(args.xml) as archive:
        members = [member for member in archive.namelist() if member.lower().endswith(".xml")]
        for member in members:
            root = ET.fromstring(archive.read(member))
            filename = root.findtext("filename")
            image_path = root.findtext("path")
            filename_stem = normalized(filename)
            path_stem = normalized(image_path)
            member_stem = normalized(member)
            if filename_stem:
                filename_stems.add(filename_stem)
            if path_stem:
                path_stems.add(path_stem)
            if member_stem:
                member_stems.add(member_stem)
            if len(samples) < 10:
                samples.append({"member": member, "filename": filename, "path": image_path, "filename_stem": filename_stem, "path_stem": path_stem, "member_stem": member_stem})
    payload = {
        "prediction_records": len(prediction_names),
        "prediction_name_sample": prediction_names[:10],
        "prediction_stem_sample": sample(prediction_stems),
        "xml_members": len(members),
        "xml_first_entries": samples,
        "overlap": {
            "prediction_vs_filename_stem": len(prediction_stems & filename_stems),
            "prediction_vs_path_stem": len(prediction_stems & path_stems),
            "prediction_vs_member_stem": len(prediction_stems & member_stems),
            "unmatched_prediction_stem_sample": sample(prediction_stems - filename_stems),
            "filename_stem_sample": sample(filename_stems),
            "path_stem_sample": sample(path_stems),
            "member_stem_sample": sample(member_stems),
        },
        "xml_field_presence": {"filename": len(filename_stems), "path": len(path_stems), "member": len(member_stems)},
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
