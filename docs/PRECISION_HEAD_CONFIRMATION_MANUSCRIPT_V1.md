# Manuscript insert — precision-head confirmation (draft)

## Methods

We evaluated two frozen, task-specific checkpoints, YOLOv8n and YOLO26n, on
the CCTSDB2021 development split under a pre-registered TensorRT build
contract. Uniform train-only calibration selections U42/U43/U44 each contained
1,024 images and used the same MinMax recipe. For each model we built three
independent plans for each selection and arm: baseline INT8, bbox FP32,
classification FP32 and both FP32. Three architecture-specific FP16 controls
were also built. Each model's 42 builders were executed as one block, with
rotated 13-cell rounds and fresh timing inputs.

The v8 decoded head and v26 end-to-end top-k head were evaluated with their
locked, different postprocessing routes. Predictions were linked to the
identical 1,636-image dev inventory and XML annotations. The primary contrast
was both-FP32 minus baseline-INT8 full AP50-95. We report full, XS and S
endpoints, image-paired PCG64 bootstrap intervals (seed 20260916, 1,000
draws), within-selection build SD, and between-selection-mean SD.

## Results

This section remains a template until the complete server artifact set is
audited. No estimate is inserted from the historical YOLO11n discovery,
feasibility smoke, CPU bridge, or partial confirmation output. The final table
will list all 78 scored cells, cache/build identity, full/XS/S metrics,
primary/control/FP16 contrasts, image-bootstrap intervals, and the two
distinct variability summaries.

## Limitations

The architecture sample contains two frozen models rather than a random model
sample. Shared-lab GPU telemetry is sampled and does not establish isolation;
thermal/order and implementation variability remain execution conditions.
The three calibration selections are fixed design cells, not nine independent
calibration samples. FP16 is a control/reference condition, not a
non-inferiority claim. COCO/XML and Ultralytics metrics are separate channels.
No test, negative, weather, latency, energy, deployment or causal tactic claim
is made. A negative, mixed, architecture-sensitive or incomplete confirmation
is scientifically valid and must remain visible.
