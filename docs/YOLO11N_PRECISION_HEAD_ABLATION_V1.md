# YOLO11n precision-head ablation v1

Status: implementation/protocol prepared locally for A2L-013. This document does not authorize a server run. Server execution requires the subsequent Astra code review decision (A2L-014).

## Scope

This is a dev-only diagnostic intervention on the frozen YOLO11n CCTSDB2021 model. It uses the already accepted Uniform seed-42 calibration/cache inputs and the same `CCTSDB2021/dev` validator. It does not retrain, change calibration images, use the official test, benchmark a device, open the 15-model matrix, or select a best model.

The locked study ID is `yolo11n_precision_head_ablation_v1`, with output exactly at:

`results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1/`

The source is the accepted Step-A study `server_uniform_build_repeat_v1`. The accepted timing-replay study `server_uniform_timing_cache_replay_v1` is the baseline reference for comparison only; its accepted result commit is recorded in every manifest.

## Arms and layer contract

| Arm | Additional selected layers |
|---|---|
| `baseline_int8` | none; preserve the 77 baseline Sigmoid FP32 constraints and output constraints |
| `bbox_fp32` | convolution layers whose parsed ONNX/TensorRT name starts exactly `/model.23/cv2.` |
| `classification_fp32` | convolution layers whose parsed ONNX/TensorRT name starts exactly `/model.23/cv3.` |
| `both_fp32` | the union of the two prefixes above |

Selection is performed after parsing the actual network and checks layer type. Every selected convolution receives requested FP32 precision and FP32 output types with `OBEY_PRECISION_CONSTRAINTS`. The builder settings remain Step A: INT8 on, FP16/TF32 off, workspace 4 GiB, optimization level 3, average timing iterations 1, detailed inspector. No `/model.23/dfl`, `/model.23/Sigmoid`, backbone, neck, or decode layer is newly selected.

Each build manifest records full matched names, counts, layer types, branch prefixes, and before/requested/after constraint states. These records describe requested/effective builder evidence; they do not prove that every branch arithmetic operation ran in FP32, and counts are not FLOP ratios.

## Build and capture order

There are four arms × three independent builds: 12 builds total. The parent runs them sequentially in this fixed order:

1. `baseline_int8` repeats 1, 2, 3
2. `bbox_fp32` repeats 1, 2, 3
3. `classification_fp32` repeats 1, 2, 3
4. `both_fp32` repeats 1, 2, 3

Only after all 12 build manifests validate does the parent run one capture and CPU verification for each build in the same order. A child owns one build; the parent remains CPU orchestration. Existing output and partial output are preserved and never overwritten or silently resumed.

Every build receives independent copies of the same Step-A repeat-1 calibration and timing cache bytes:

- calibration cache SHA-256: `31e9d0b3f69470ac20f7380d8887c6dc47954afbe85d84be39870ba44e01a502`;
- timing cache SHA-256: `4c765a0224845e9ddc537253878c696e56c11045369aef224cff6cc0c4178f38`.

There is no cache chaining. Timing-cache attachment uses `ignore_mismatch=False`; if TensorRT rejects the common cache, the run keeps its partial evidence and is invalid/review-required. It must not fall back to an empty or arm-specific cache. Cache coverage is recorded as `unknown`.

All builds/captures use one GPU identity and one compatible environment, with the existing GPU lock, UUID/name/driver checks, current telemetry before/after, and workload guard. Confirmed desktop processes are scoped operator exceptions. New, unconfirmed, unverifiable, or competing compute workload is recorded and makes the affected run review-required; no process is killed or paused and no permissions, clocks, or power limits are changed. Shared-server telemetry does not establish absolute GPU isolation.

## Evaluation

Each capture uses the same-pass validator with `imgsz=640`, `batch=1`, `workers=0`, `conf=0.001`, NMS IoU `0.7`, `max_det=300`, `rect=False`, and the fixed dev split of 1,636 images/2,706 instances. Native matching and COCO/XML size verification are reused. Endpoints are Ultralytics full plus COCO/XML `all`, `XS`, `S`, `M`, `L`, and `XL`; full, XS, and S are explicitly surfaced in the arm deltas. No official test is read.

The summary reports for each arm and endpoint the mean, sample SD (`ddof=1`), min, max, and range in percentage points across its three builds. It reports arm-minus-baseline-arm-mean deltas separately for Ultralytics and COCO/XML, plus `bbox_delta`, `classification_delta`, and `both_delta`. It also keeps raw per-build comparisons against the accepted timing-replay repeat-1 reference and repeat-1 within-arm comparisons; timestamps and paths are not used as payload identity.

The fixed descriptive classifications are:

- `diagnostic_branch_sensitive`: at least one intervention arm improves outside the larger observed baseline/current build range at at least two of full, XS, and S;
- `no_branch_signal`: no intervention satisfies that descriptive condition;
- `incomplete_or_invalid`: a build/capture is missing or any cache, calibration, layer-selection, provenance, native/size, telemetry, or workload guard contract fails.

The classification is descriptive evidence, not a policy selector, ground truth claim, p-value, confidence interval, or proof that precision alone caused an AP change. A common timing cache selected after Step-A/timing-replay results is a feasibility intervention, not a tactic lock or preregistered universal pairing.

## Artifact contract

The output must contain `study_manifest.json`, `comparison_summary.json`, `repeat_summary.json`, 12 build logs and compressed logs, and four arm directories containing three repeats each. Each repeat contains `build_manifest.json`, `model.engine` (server-only), input/output cache evidence, raw `inspector.json`, `capture/capture_report.json`, `capture/validator_predictions.json`, `verification/verification_summary.json`, `verification/native_matching.json`, `verification/size_coco_xml.json`, `execution_manifest.json`, and `comparison.json`.

Engine binaries, ONNX, and other server-only binary inputs are not pushed to Git. JSON/log/cache artifacts required by the handoff are pushed after the server run. Local review checks source hashes, manifest bindings, engine/capture provenance hashes, cache contract, payload membership/preprocessing, inspector signatures, telemetry/workload flags, all 12 build/capture records, and the summary classification. The study stops at `step_A_completed_review_required`; no follow-up research step is launched automatically.
