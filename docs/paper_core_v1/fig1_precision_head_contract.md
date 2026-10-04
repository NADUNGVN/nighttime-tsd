# Figure 1 contract and render record — precision-head effect beside build spread

Status: **`rendered_draft_review_required`**. PDF, editable SVG and 600-dpi
PNG were generated from the committed tables with the existing CPU-only
measurement environment. The canonical Icarus style package and critique gate
are unavailable, so no canonical-gate pass or publication-ready claim is made.

```yaml
figure_id: fig1_precision_head_effect_and_build_spread
core_claim: "In the exploratory YOLO11n captures, both-branch FP32 exceeded baseline INT8 across the registered COCO/XML AP50–95 size strata; the separate three-build Uniform range is descriptive and is not a confidence interval."
backend: matplotlib
panels:
  - id: a
    type: forest
    defends: "Conditional paired both_fp32-minus-baseline_int8 AP50-95 point contrasts and 95% percentile intervals across full/XS/S/M/L/XL."
  - id: b
    type: range_width_with_endpoint_labels
    defends: "Observed AP50–95 max-minus-min span in percentage points across three Uniform builds for each size stratum; direct labels retain the absolute AP endpoints."
archetype: quantitative_composite
hierarchy:
  overview: "Panel a is the large hero: signed contrast with a visible zero reference."
  deviation: "Panel b is smaller context: observed three-build range, clearly labeled as non-CI."
  relationship: "Same AP50-95 endpoint and size strata connect policy contrast to build spread; separate studies and sampling units remain explicit."
hero_panel: a
export: [pdf, svg, png]
source_data:
  - results/paper_core_v1/tables/precision_head_contrasts.csv
  - results/paper_core_v1/tables/build_variability.csv
stats_on_figure: "Panel a: 1,000 shared-image PCG64 bootstrap draws (seed 20260916), 1,636 images, 2,706 instances; one captured build/condition and one calibration selection. Panel b: n=3 builds under one Uniform selection; bars encode the observed max-minus-min span (pp) and labels show the actual min–max AP endpoints; this is not a confidence interval."
```

## Intended design and interpretation safeguards

- Use a dominant forest panel with horizontal 95% percentile intervals, point
  markers and a zero line. Keep signed values in pp; do not truncate the axis
  near the effect.
- Use a smaller range-width panel with explicit “observed min–max, 3 builds”
  wording and numeric AP endpoint labels. Bars start at zero and encode only
  max-minus-min span; they are not styled or labeled as confidence intervals.
- Encode condition/panel by marker and linestyle as well as the color-safe
  palette so the result survives grayscale printing.
- Annotate the one-checkpoint/one-selection scope; do not imply an independent
  calibration sample or a causal mechanism.
- Caption: “Both-FP32 minus baseline-INT8 COCO/XML AP50–95 was positive in the
  one-selection YOLO11n discovery across all six registered size strata; the
  full-set contrast was +8.51 percentage points (pp; paired image-bootstrap
  95% percentile interval +7.71 to +9.00 pp). Panel (a) shows conditional
  intervals from 1,000 shared-image PCG64 draws (seed 20260916; 1,636 images,
  2,706 XML instances) with one captured build per condition. Panel (b) shows
  the observed max-minus-min span across three builds in a separate Uniform
  repeat study (n=3), with labels giving the absolute AP endpoints; these
  spans are descriptive, not confidence intervals.
  Cross-model confirmation remains pending.”

## Render, environment and review record

Run from the repository root with the preinstalled interpreter (no package
installation):

```powershell
& 'D:\Research\paper\local\measurement_audit_env\Scripts\python.exe' scripts/fig1_precision_head_effect_and_build_spread.py
& 'D:\Research\paper\local\measurement_audit_env\Scripts\python.exe' -m unittest discover -s tests -p 'test_fig1_precision_head_effect_and_build_spread.py' -v
```

The run used Python 3.11.9, NumPy 2.4.2, pandas 3.0.1, Matplotlib 3.10.8,
Pillow 12.3.0, pycocotools 2.0.10, ONNX Runtime 1.24.3 and Ultralytics 8.4.102.
The figure itself uses NumPy/pandas/Matplotlib/Pillow only. This checkout has
no `paperfig`; neither `scripts/critique.py` nor the skill's canonical
`scripts/_style.py` is installed. `scripts/_style.py` in this repository is a
small shared local style/export fallback, not the canonical Icarus preset.

The generator validates the locked six-row CSV selections, finite numeric
values, source bootstrap count/seed, `point ∈ CI`, positive stored lower bounds,
three-build count and `range = max − min`. Its tests exercise those checks and
the PDF/SVG/PNG export; the output manifest records source, script, helper,
contract and export SHA-256 values. The image was visually inspected; the
manual four-axis review and remaining limitations are in
`fig1_precision_head_qa.md`. Canonical mechanical critique is unavailable and
was not run. The figure remains `rendered_draft_review_required` pending
independent review and canonical tooling.
