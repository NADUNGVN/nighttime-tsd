# Figure 1 contract — precision-head effect beside observed build spread

Status: **inputs ready; rendering/gate blocked in this Windows environment**.
No PDF/PNG/SVG is claimed to exist.

```yaml
figure_id: fig1_precision_head_effect_and_build_spread
core_claim: "In one exploratory YOLO11n setting, the both-branch FP32 minus baseline INT8 COCO/XML AP50-95 contrast is positive across registered size strata, while three-build spread is also size-dependent and is not a calibration-selection interval."
backend: matplotlib
panels:
  - id: a
    type: forest
    defends: "Conditional paired both_fp32-minus-baseline_int8 AP50-95 point contrasts and 95% percentile intervals across full/XS/S/M/L/XL."
  - id: b
    type: range_points
    defends: "Observed AP50-95 min-to-max range across three Uniform builds for each size stratum; descriptive range only."
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
stats_on_figure: "Panel a: 1,000 shared-image PCG64 bootstrap draws (seed 20260916), 1,636 images, 2,706 instances; one captured build/condition and one calibration selection. Panel b: n=3 builds under one selection; line spans observed minimum-to-maximum, not a confidence interval."
```

## Intended design and interpretation safeguards

- Use a dominant forest panel with horizontal 95% percentile intervals, point
  markers and a zero line. Keep signed values in pp; do not truncate the axis
  near the effect.
- Use a smaller range panel with an explicit “observed min–max, 3 builds” label.
  Never style it as a CI or put it on the same visual legend as the bootstrap
  interval.
- Encode condition/panel by marker and linestyle as well as the color-safe
  palette so the result survives grayscale printing.
- Annotate the one-checkpoint/one-selection scope; do not imply an independent
  calibration sample or a causal mechanism.
- Provisional caption: “Both-FP32 exceeded baseline INT8 in the exploratory
  YOLO11n captures across the registered size strata, but the comparison is
  conditional on one build per condition and one calibration selection
  (panel a; paired image-bootstrap 95% percentile intervals, 1,000 draws,
  seed 20260916, 1,636 images and 2,706 XML instances). Panel b gives the
  observed full/size-stratified AP50–95 range across three builds of a separate
  Uniform-calibration repeat study; it is a descriptive range, not a CI. The
  cross-model confirmation remains pending.”

## Render gate

The current local Python 3.12 environment has none of `paperfig`, `matplotlib`,
`numpy` or `pandas`; the `icarus-figures` critique script is not present in the
available skill package or repository. No package installation is authorized
by this research handoff. Therefore the actual vector figure, `figure_manifest`
entry, mechanical critique and visual four-axis review remain open. The table
inputs are committed-generation candidates only; a renderer must read the CSV
files and call the shared `paper_style()`/`save()` preset, emit PDF+SVG+PNG, and
pass `critique.py` before the figure can be called publication-ready.
