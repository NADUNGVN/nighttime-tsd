# Precision-head confirmation execution v1

This is the implementation contract for ST-SERVER-01. It is deliberately
separate from the historical YOLO11n discovery, the CPU source/ONNX bridge,
and the two-model FP16 feasibility smoke.

## Locked question and inputs

The study asks whether the YOLO11n precision-head observation is reproduced by
the frozen YOLOv8n and YOLO26n checkpoints. It does not promise a positive
answer. The accepted checkpoints, ONNX files, graph mappings, and hashes are
bound in `configs/precision_head_confirmation_execution_v1.json`. A changed
checkpoint, ONNX, graph mapping, readiness manifest, or runtime is a hard
failure; no silent export, fallback, or replacement is permitted.

Calibration is train-only Uniform U42/U43/U44, 1,024 images, MinMax, using the
accepted producer/preprocessing contract. Each `(model, selection)` has one
auxiliary cache builder. Its cache is read-only for the twelve scored INT8
cells. Scored builds consume zero calibration batches and perform zero cache
writes. Every builder starts with a fresh empty timing cache; a serialized
timing cache is evidence only and is never an input to a later builder.

## Immutable schedule

There are 6 auxiliary cache builders, 72 scored INT8 builders, and 6 scored
FP16 builders: 84 builder invocations and 78 captures. Each model is a
contiguous 42-job block: three auxiliary jobs followed by three 13-cell
rounds. The rounds use rotation shifts 0, 4 and 8. The cells are the twelve
selection/arm INT8 combinations plus one architecture-specific FP16 control.
All cells are retained. No build is selected, reordered, retried, or replaced
after looking at metrics. A timeout or invalid cell stops that model block.

The parent writes `confirmation_plan.json` and `schedule.json` before any
server child. The schedule hash is part of every child identity. Build and
capture children have separate state files, deadlines, atomic writes and
owned-process cleanup. Private engines, calibration/timing caches and raw
tensors remain on the producing server and are not published.

## Architecture contract

YOLOv8n targets only the verified `model.22/cv2` bbox and `model.22/cv3`
classification convolution nodes. Its decoded `[1,7,8400]` head output uses
the locked v8 NMS route. YOLO26n targets only verified
`model.23/one2one_cv2` and `model.23/one2one_cv3` nodes; inactive `cv2`/`cv3`
are never used. Its `[1,300,6]` end-to-end top-k output receives no second
NMS. Any ambiguous, missing, overlapping, or non-convolution target blocks the
affected model.

Each arm records requested, observed and effective precision/output types and
the OBEY constraint. Layer counts or inspector labels are not treated as a
proof of arithmetic precision.

## Capture and analysis

The dev split is 1,636 images and 2,706 XML instances. Inputs are batch 1,
640px, `conf=0.001`, `iou=0.7`, `max_det=300`, workers 0. Raw output hashes
are recorded before consumer postprocessing. Native matching, COCO/XML
metrics and the Ultralytics channel are separate evidence; no invented
native/ORT/TRT tensor equality is used.

The primary contrast is `both_fp32 - baseline_int8` on full AP50-95. Full,
XS and S endpoints are reported. Bootstrap uses PCG64 seed 20260916 with
1,000 draws and shared image resampling across arms, builds and selections;
duplicates remain in each resample. Build SD within a selection and SD across
the three selection means are separate ddof=1 descriptive quantities. The
nine scored engines are not treated as nine independent calibration samples.
No FP16 non-inferiority margin or post-hoc significance threshold is created.

## State and terminal statuses

The parent is CPU-only and must not import CUDA/TensorRT. The server child is
the only process allowed to import them. State is written atomically and
contains phase, identity, counters, telemetry, lifecycle and partial-file
inventory. Existing output roots are never resumed or overwritten.

The current pre-execution state is
`implementation_incomplete_remediation_in_progress`; it becomes
`implementation_complete_integrated_review_required` only after Astra accepts
this integrated R1 packet. Only after that review may the operator pass the GO
token and run the foreground server sequence. A complete artifact set ends in
`confirmation_completed_review_required`; missing, invalid, timed-out or
contaminated cells end in `confirmation_incomplete_blocked`. Neither status
selects a deployment arm or opens B/C/15-model work.
