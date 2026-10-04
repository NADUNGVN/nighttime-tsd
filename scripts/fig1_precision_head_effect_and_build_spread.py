#!/usr/bin/env python
"""Render Figure 1 from the committed paper-core evidence tables (CPU only)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _style import paper_style, save  # noqa: E402


FIGURE_ID = "fig1_precision_head_effect_and_build_spread"
CORE_CLAIM = (
    "In the exploratory YOLO11n captures, both-branch FP32 exceeded baseline "
    "INT8 across the registered COCO/XML AP50–95 size strata; the separate "
    "three-build Uniform range is descriptive and is not a confidence interval."
)
SIZES = ("all", "xs", "s", "m", "l", "xl")
SIZE_LABELS = ("All", "XS", "S", "M", "L", "XL")
CONTRAST_PATH = Path("results/paper_core_v1/tables/precision_head_contrasts.csv")
BUILD_PATH = Path("results/paper_core_v1/tables/build_variability.csv")
MANUAL_QA_PATH = Path("docs/paper_core_v1/fig1_precision_head_qa.md")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _require_finite(df: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        if not np.isfinite(pd.to_numeric(df[column], errors="coerce")).all():
            raise ValueError(f"Non-finite {label}.{column}")


def load_inputs(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate only the locked YOLO11n table rows for this figure."""
    contrast_file = repo / CONTRAST_PATH
    build_file = repo / BUILD_PATH
    if not contrast_file.is_file() or not build_file.is_file():
        raise FileNotFoundError("Committed paper-core CSV input is missing")

    contrasts = pd.read_csv(contrast_file)
    builds = pd.read_csv(build_file)
    c = contrasts.loc[
        (contrasts["study"] == "precision_head_paired_analysis_v1")
        & (contrasts["model"] == "YOLO11n")
        & (contrasts["contrast"] == "both_minus_baseline")
        & (contrasts["metric"] == "map50_95")
    ].copy()
    b = builds.loc[
        (builds["study"] == "uniform_build_repeat_v1")
        & (builds["model"] == "YOLO11n")
        & (builds["metric"] == "map50_95")
    ].copy()

    if tuple(c["size"].tolist()) != SIZES:
        raise ValueError(f"Expected paired contrast sizes in order {SIZES}, got {tuple(c['size'])}")
    if tuple(b["size"].tolist()) != SIZES:
        raise ValueError(f"Expected build-range sizes in order {SIZES}, got {tuple(b['size'])}")
    if c["size"].duplicated().any() or b["size"].duplicated().any():
        raise ValueError("Figure inputs contain duplicate size rows")

    _require_finite(
        c,
        ("point_delta_pp", "ci95_low_pp", "ci95_high_pp", "valid_resamples", "undefined_resamples"),
        "contrasts",
    )
    _require_finite(
        b,
        ("builds", "mean_percent", "minimum_percent", "maximum_percent", "range_pp"),
        "builds",
    )
    if not ((c["ci95_low_pp"] <= c["point_delta_pp"]) & (c["point_delta_pp"] <= c["ci95_high_pp"])).all():
        raise ValueError("A paired point estimate falls outside its stored interval")
    if not (c["ci95_low_pp"] > 0).all():
        raise ValueError("The locked source table no longer supports the contracted all-positive caption")
    if not ((c["valid_resamples"] == 1000) & (c["undefined_resamples"] == 0)).all():
        raise ValueError("The expected 1,000 finite paired bootstrap draws are absent")
    if not (b["builds"] == 3).all():
        raise ValueError("The descriptive range panel requires exactly three recorded builds")
    if not ((b["minimum_percent"] <= b["mean_percent"]) & (b["mean_percent"] <= b["maximum_percent"])).all():
        raise ValueError("Stored build mean is outside its observed min–max")
    if not np.allclose(b["maximum_percent"] - b["minimum_percent"], b["range_pp"], atol=1e-8, rtol=0):
        raise ValueError("Stored build range does not equal maximum minus minimum")
    if not ((c["bootstrap_seed"] == 20260916) & (c["bootstrap_resamples"] == 1000)).all():
        raise ValueError("Unexpected paired bootstrap seed or draw count")
    return c, b


def build_figure(
    contrasts: pd.DataFrame,
    builds: pd.DataFrame,
    palette: tuple[str, ...],
    markers: tuple[str, ...],
) -> plt.Figure:
    """Make a hero paired-contrast panel and a distinct descriptive range panel."""
    fig, (ax_contrast, ax_range) = plt.subplots(
        1,
        2,
        figsize=(6.85, 3.55),
        sharey=True,
        gridspec_kw={"width_ratios": (1.8, 1.2), "wspace": 0.10},
    )
    y = np.arange(len(SIZES))
    c_color = palette[0]
    range_color = palette[1]
    ink = palette[6]
    grid = palette[7]

    fig.suptitle(
        "YOLO11n exploratory AP50–95",
        x=0.055,
        y=0.985,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.915,
        "Separate discovery and build-repeat studies; neither identifies a causal mechanism or cross-model effect",
        ha="left",
        va="top",
        fontsize=7.0,
        color=ink,
    )

    contrast_points = contrasts["point_delta_pp"].to_numpy(float)
    contrast_low = contrasts["ci95_low_pp"].to_numpy(float)
    contrast_high = contrasts["ci95_high_pp"].to_numpy(float)
    ax_contrast.errorbar(
        contrast_points,
        y,
        xerr=np.vstack((contrast_points - contrast_low, contrast_high - contrast_points)),
        fmt=markers[0],
        color=c_color,
        ecolor=c_color,
        markerfacecolor=palette[8],
        markeredgecolor=c_color,
        markeredgewidth=1.0,
        markersize=4.2,
        elinewidth=1.15,
        capsize=2.6,
        capthick=0.9,
        zorder=3,
    )
    ax_contrast.axvline(0, color=ink, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
    ax_contrast.set_xlim(-2, 14)
    ax_contrast.set_xticks((-2, 0, 4, 8, 12))
    ax_contrast.set_xlabel("Δ COCO/XML AP50–95 (percentage points)")
    fig.text(0.105, 0.815, "(a) Both FP32 − baseline INT8", ha="left", va="bottom", fontsize=8.1, fontweight="bold")
    fig.text(
        0.105,
        0.780,
        "Whiskers: paired 95% image-bootstrap CI · N=1,636 images / 2,706 XML instances",
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=ink,
    )
    ax_contrast.text(
        9.55,
        y[0] + 0.22,
        "All: +8.51 [7.71, 9.00] pp",
        ha="left",
        va="center",
        fontsize=5.8,
        color=c_color,
    )
    ax_contrast.grid(axis="x", color=grid, linewidth=0.45, alpha=0.85, zorder=0)

    minima = builds["minimum_percent"].to_numpy(float)
    maxima = builds["maximum_percent"].to_numpy(float)
    ranges = builds["range_pp"].to_numpy(float)
    ax_range.barh(y, ranges, height=0.24, color=range_color, edgecolor=range_color, linewidth=0.7, zorder=2)
    ax_range.plot(
        ranges,
        y,
        linestyle="none",
        marker=markers[1],
        markersize=3.6,
        markerfacecolor=palette[8],
        markeredgecolor=range_color,
        markeredgewidth=0.9,
        zorder=4,
    )
    for row, low, high, span in zip(y, minima, maxima, ranges, strict=True):
        ax_range.text(span + 0.10, row, f"{low:.1f}–{high:.1f}%", ha="left", va="center", fontsize=6.0, color=ink)
    ax_range.set_xlim(0, 6.2)
    ax_range.set_xticks((0, 1, 2, 3, 4, 5, 6))
    ax_range.set_xlabel("Observed range width (max−min, pp)")
    fig.text(0.665, 0.815, "(b) Three-build range width", ha="left", va="bottom", fontsize=8.1, fontweight="bold")
    fig.text(
        0.665,
        0.780,
        "Bar: max−min (pp); label: AP% min–max; n=3; no CI",
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=ink,
    )
    ax_range.grid(axis="x", color=grid, linewidth=0.45, alpha=0.85, zorder=0)

    ax_contrast.set_yticks(y, SIZE_LABELS)
    ax_contrast.set_ylabel("CCTSDB size bin")
    ax_range.tick_params(axis="y", left=False, labelleft=False)
    ax_contrast.set_ylim(len(SIZES) - 0.45, -0.55)
    for ax in (ax_contrast, ax_range):
        ax.tick_params(direction="in", length=2.8, width=0.65)
        ax.spines["left"].set_color(ink)
        ax.spines["bottom"].set_color(ink)
        ax.spines["left"].set_linewidth(0.65)
        ax.spines["bottom"].set_linewidth(0.65)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.70, bottom=0.22, wspace=0.13)
    return fig


def _versions() -> dict[str, str]:
    from importlib.metadata import version

    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "matplotlib": version("matplotlib"),
        "Pillow": version("Pillow"),
    }


def _write_manifest(repo: Path, out_dir: Path, outputs: dict[str, Path]) -> dict[str, Any]:
    input_paths = (repo / CONTRAST_PATH, repo / BUILD_PATH)
    output_records = {
        fmt: {
            "path": path.relative_to(repo).as_posix() if path.is_relative_to(repo) else str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for fmt, path in outputs.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "figure_id": FIGURE_ID,
        "status": "rendered_draft_review_required",
        "core_claim": CORE_CLAIM,
        "contract": {
            "path": "docs/paper_core_v1/fig1_precision_head_contract.md",
            "sha256": sha256_file(repo / "docs/paper_core_v1/fig1_precision_head_contract.md"),
        },
        "generator": {
            "path": Path(__file__).relative_to(repo).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "shared_style_export_helper": {
            "path": Path(__file__).with_name("_style.py").relative_to(repo).as_posix(),
            "sha256": sha256_file(Path(__file__).with_name("_style.py").resolve()),
            "status": "repository_local_fallback_not_canonical_icarus_style",
        },
        "inputs": [
            {
                "path": p.relative_to(repo).as_posix(),
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
            for p in input_paths
        ],
        "outputs": output_records,
        "render": {
            "backend": "matplotlib",
            "formats": ["pdf", "svg", "png"],
            "png_dpi": 600,
            "vector_text": "editable SVG text; PDF uses TrueType font embedding",
        },
        "environment": _versions(),
        "checks": {
            "data_contract": "pass",
            "three_format_export": "pass",
            "png_resolution": "pass_600_dpi",
            "canonical_icarus_critique_gate": "unavailable_not_run",
            "manual_four_axis_review": "recorded_in_docs/paper_core_v1/fig1_precision_head_qa.md",
            "release_label": "rendered_draft_review_required",
        },
    }
    manifest_path = out_dir / f"{FIGURE_ID}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = out_dir / "figure_manifest.csv"
    row = {
        "id": FIGURE_ID,
        "path": ";".join(v["path"] for v in output_records.values()),
        "claim": CORE_CLAIM,
        "source_data": ";".join(v.relative_to(repo).as_posix() for v in input_paths),
        "generation_script": Path(__file__).relative_to(repo).as_posix(),
        "status": "rendered_draft_review_required",
        "input_sha256": ";".join(v["sha256"] for v in manifest["inputs"]),
        "output_sha256": ";".join(v["sha256"] for v in output_records.values()),
        "style_helper": Path(__file__).with_name("_style.py").relative_to(repo).as_posix(),
        "dependency_provenance": json.dumps(manifest["environment"], sort_keys=True),
        "manual_critique": MANUAL_QA_PATH.as_posix(),
    }
    existing: list[dict[str, str]] = []
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            existing = [r for r in csv.DictReader(f) if r.get("id") != FIGURE_ID]
    fields = list(row)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)
    return manifest


def generate(repo: Path, out_dir: Path) -> dict[str, Any]:
    repo = repo.resolve()
    out_dir = out_dir.resolve()
    contrasts, builds = load_inputs(repo)
    palette, markers, _linestyles = paper_style(font="sans")
    fig = build_figure(contrasts, builds, palette, markers)
    fig.canvas.draw()
    outputs = save(fig, out_dir / FIGURE_ID, dpi=600)
    plt.close(fig)
    manifest = _write_manifest(repo, out_dir, outputs)
    print(f"[fig] saved {', '.join(str(p) for p in outputs.values())}")
    print(f"[fig] status={manifest['status']} manifest={out_dir / (FIGURE_ID + '_manifest.json')}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    out_dir = args.out_dir if args.out_dir is not None else repo / "outputs/figures"
    generate(repo, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
