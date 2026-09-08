# Calibration methodology audit before YOLO11n development

Audit date: 2026-09-08. This record precedes any VCSC engine export.

## Reusable evidence

- Frozen checkpoints: all 15 CCTSDB models exist under `results/*/weights/best.pt`.
  The development checkpoint is `yolo11n_cctsdb_clean_s42_v2`.
- FP32 evaluations: 120 result JSON files cover 15 models × full/daylike/six
  official weather-light splits. The result schema records model/data SHA-256,
  mAP50, mAP50-95, precision, recall, per-class AP50, and environment.
- TensorRT pilot: YOLO11n FP16 and two 512-image INT8 policies are available
  as historical feasibility evidence only. They do not meet the new 1,024-image
  VCSC or repeated-seed protocol.
- Existing scripts: `prepare_cctsdb.py` creates the immutable train/dev/positive
  test and weather split layout; `build_calibration_set.py`,
  `export_tensorrt.py`, `evaluate_cctsdb.py`, and
  `evaluate_tensorrt_suite.py` provide the reusable calibration/export/eval
  pipeline. Existing benchmark scripts measure preloaded-image latency only.

## New requirements and modifications

- Replace the previous four-bin luminance policy with VCSC: train-only visual
  descriptors, training-pool standardization, deterministic K-means (`K=8`),
  and near-equal cluster sampling.
- Build official size subsets only from the CCTSDB size XML package, not from
  thresholds applied to labels or predictions. The official package excludes
  mixed-size images.
- Audit 500 official negatives separately and measure fixed-threshold false
  positives. They must never enter the 1,500-positive-image mAP benchmark.
- Run only frozen YOLO11n through FP16, Uniform INT8, Low-Luminance INT8, and
  VCSC INT8 before any all-model export.

## Leakage controls

- Calibration builder resolves and reads only `train/images` and
  `train/labels`; it rejects missing labels and records source paths.
- VCSC descriptors, standardization, clusters, and sampling use only that
  train pool. No test-domain folder, test XML, or test metric is an input.
- Size XML and negative assets are accessed only by post-export evaluators.
- The 15-model runner now refuses non-status phases unless an explicit
  post-gate authorization is written into its config.

## Hard-coded assumptions to preserve or review

- CCTSDB image size is fixed at 640 for all exports/evaluations.
- TensorRT static batch is one; the reference RTX environment expects major
  version 10.
- Latency confidence/IoU are 0.25/0.70 and are not selected on official test.
- Low-Luminance is a deterministic ranked set. Across multiple seeds its
  selection may be identical except at a luminance tie boundary; report that
  explicitly rather than claiming sampling variability where none exists.
- Foggy has only 40 images; it is a bootstrap diagnostic, never a standalone
  primary selection criterion.
