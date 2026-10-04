"""Shared local paper-figure style/export fallback for this repository.

This is intentionally a small project-local substitute because the canonical
Icarus ``paperfig`` package and its critique gate are unavailable here. It is
not represented as the canonical preset.
"""

from __future__ import annotations

from pathlib import Path

PALETTE = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00",
    "#56B4E9", "#CC79A7", "#333333", "#D9DEE5", "#FFFFFF",
)
MARKERS = ("o", "s", "D", "^", "v", "P")
LINESTYLES = ("-", "--", "-.", ":")


def paper_style(font: str = "sans"):
    """Apply the shared local style once and return palette encodings."""
    if font != "sans":
        raise ValueError("The local fallback currently supports only sans")
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.titlesize": 8.7,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            # Stable element IDs make byte-level SVG hashes reproducible.
            "svg.hashsalt": "nighttime-tsd-paper-core-v1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
        }
    )
    return PALETTE, MARKERS, LINESTYLES


def save(fig, output_stem: str | Path, *, dpi: int = 600) -> dict[str, Path]:
    """Write editable PDF/SVG plus a 600-dpi PNG using stable metadata."""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "pdf": {"Title": stem.name, "Creator": "repository-local _style.py fallback", "CreationDate": None, "ModDate": None},
        "svg": {"Title": stem.name, "Creator": "repository-local _style.py fallback", "Date": None},
        "png": {"Software": "repository-local _style.py fallback"},
    }
    outputs: dict[str, Path] = {}
    for fmt in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight", metadata=metadata[fmt])
        if fmt == "svg":
            # Matplotlib emits trailing spaces in path data lines; they are
            # valid SVG but fail repository whitespace checks. Normalize only
            # line endings/trailing horizontal whitespace, preserving markup.
            content = path.read_text(encoding="utf-8")
            normalized = "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
            with path.open("w", encoding="utf-8", newline="\n") as svg_file:
                svg_file.write(normalized)
        outputs[fmt] = path
    return outputs
