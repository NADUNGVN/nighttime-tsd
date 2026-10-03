# Manuscript insert — precision-head confirmation (prospective protocol)

**Execution status:** approved conditional GO; no v2 scored run has been
audited. This text describes a planned protocol, not completed work.

## Methods

The approved protocol will evaluate two frozen, task-specific checkpoints,
YOLOv8n and YOLO26n, on the CCTSDB2021 development split under the locked
TensorRT build contract. Uniform train-only calibration selections U42/U43/U44
each contain 1,024 manifest-ordered images and use the same MinMax recipe. For
each model, the schedule includes three repeats for each selection and INT8
arm (baseline, bbox FP32, classification FP32 and both FP32), plus the
architecture-specific FP16 controls. If the single authorized run proceeds,
each model's 42 scheduled builders will remain one contiguous block with
rotated 13-cell rounds and fresh timing inputs.

The plan binds the v8 decoded head and v26 end-to-end Top-K head to their
different locked postprocessing routes. Captures will be checked against the
same 1,636-image dev inventory and XML annotations. The primary contrast is
both-FP32 minus baseline-INT8 full AP50–95. The planned analysis reports full,
XS and S endpoints, image-paired PCG64 bootstrap intervals (seed 20260916,
1,000 draws), within-selection build SD, and between-selection-mean SD.

## Results

No cross-model confirmation estimate is available. This section will be
completed only after the user-operated artifact is pushed and all scheduled
cells, hashes, counters, lifecycle evidence and dataset bindings are audited.
Historical YOLO11n discovery, feasibility smoke, CPU bridge and failed partial
v1 are not substitutes. The final table is expected to include all 78 capture
cells, cache/build identity, full/XS/S metrics, primary/control/FP16
contrasts, image-bootstrap intervals and both variability summaries; any
missing/invalid cells will be reported as incomplete, with no replacement run.

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
