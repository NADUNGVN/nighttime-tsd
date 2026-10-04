# Figure 1 manual QA — 2026-10-04

Status: **`rendered_draft_review_required`**. The figure is rendered and
locally checked, but the canonical Icarus critique package/gate is unavailable;
this document is a manual review record, not a canonical pass or publication
approval.

## Contract and reproducible output

- Claim: in the exploratory YOLO11n captures, both-branch FP32 exceeded
  baseline INT8 across the registered COCO/XML AP50–95 strata; the separate
  three-build Uniform span is descriptive, not a confidence interval.
- Source rows: `precision_head_contrasts.csv` and `build_variability.csv`,
  both committed under `results/paper_core_v1/tables/`; source, code, helper,
  contract and export hashes are recorded in
  `outputs/figures/fig1_precision_head_effect_and_build_spread_manifest.json`.
- Generator: `scripts/fig1_precision_head_effect_and_build_spread.py`;
  local shared style/export fallback: `scripts/_style.py`. The fallback is
  explicitly not the canonical Icarus `_style.py` or `paperfig` package.
- The fallback fixes Matplotlib's `svg.hashsalt`; identical inputs/settings
  therefore regenerate byte-identical PDF, SVG, PNG and figure manifest in
  the checked environment. It also strips Matplotlib's trailing whitespace
  from SVG lines without changing markup; a regression checks cleanliness and
  repeat-render stability.
- Outputs: PDF, editable-text SVG and 600-dpi PNG under `outputs/figures/`.
  The PNG was re-rendered in the pinned existing CPU environment and inspected
  in both color and a generated grayscale preview. No packages were installed.
- Local checks: fixed row/claim contract, valid values, contrast-in-CI,
  `range = max − min`, three export formats, high-resolution PNG and
  byte-stable repeat rendering. Focused tests: 4/4 PASS.
  `scripts/critique.py` is absent and was not run.

## Visual inspection

The figure reads from a dominant left forest panel to a smaller right range
panel. The left panel gives the signed zero reference, six stratum labels,
point estimates and percentile whiskers. The right panel starts at zero and
labels the absolute AP endpoints beside each observed max–min span. The subtitle
states that the panels are separate studies and do not identify a causal
mechanism or cross-model effect. The grayscale preview retains readable
whiskers/circle markers on the left and bar/endpoint-square encoding on the
right; the story does not depend on color alone. Axis labels state AP50–95 and
percentage-point units. N, CI definition/seed, and `n=3; no CI` are present on
the image.

## Four-axis manual critique

1. **Depth — limited, not a mechanism pass.** The image makes the size-stratified
   conditional contrast and the separate build spread legible without the
   caption, but it cannot explain why either occurred. The claim is
   deliberately descriptive and the subtitle rejects a causal interpretation;
   adding a mechanistic annotation would exceed the evidence.
2. **Elegance — acceptable draft hierarchy.** The primary contrast is the
   larger hero panel; the range panel is a smaller but nondecorative companion
   that records a separate registered source of observed variation. Removing
   panel (b) would erase that second evidence channel, not merely remove
   decoration. The panels use separate horizontal scales and are not treated
   as commensurate uncertainty.
3. **Unimpeachable — manual checks pass for the displayed claim.** The paired
   95% percentile CI, seed/draw count, image and annotation N, build count,
   units and endpoint values are specified. Both panels survive grayscale.
   The range bars start at zero, the forest zero line is explicit, and no
   truncated bar baseline or cross-panel numeric comparison is implied.
4. **Visible gap — polished draft, independent review still needed.** Typography,
   whitespace, direct endpoint labels and visual hierarchy read as a paper
   figure at presentation scale. The wide composite needs final target-journal
   column-size review, and no independent reviewer has signed off. This axis
   is not marked accepted.

## Caption decision

The manuscript caption states the positive full-set result and its interval,
defines the bootstrap and N, and describes panel (b) as a three-build
max–min span with no CI. It does not call the figure a causal explanation or
claim cross-model replication. Overall status remains
`rendered_draft_review_required` until the canonical critique gate and an
independent paper review are available.
