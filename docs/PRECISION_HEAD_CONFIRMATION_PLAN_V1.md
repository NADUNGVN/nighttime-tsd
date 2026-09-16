# Precision-Head Confirmation Plan V1

Status: `protocol_only_review_required`

This document is a protocol proposal following A2L-021. It does not authorize a
server run, create a new runner, export an ONNX model, build TensorRT engines,
or select a deployment method. Astra must review and lock this plan before any
implementation or GPU execution.

## 1. Purpose and claim boundary

The purpose is to test whether the precision-head accuracy recovery observed in
the frozen YOLO11n/Uniform diagnostic is reproducible on two other already
trained and frozen nano checkpoints: YOLOv8n and YOLO26n. The design also
estimates calibration-selection variation separately from independent TensorRT
build variation.

The confirmation target is narrow:

- same CCTSDB2021 training provenance and frozen checkpoint for each model;
- three pre-registered train-only Uniform calibration selections;
- four INT8 representations: baseline, bbox-only FP32, classification-only
  FP32, and both branches FP32;
- an architecture-specific FP16 reference;
- dev-only accuracy capture and the accepted COCO/XML diagnostic estimator.

This plan does not claim calibration-policy superiority, universal TensorRT
determinism, causal separation of localization and classification errors,
deployment superiority over FP16, or journal acceptance/quartile. The YOLO11n
precision-head result is evidence motivating this confirmation, not a result to
be treated as already generalised. Latency is not an endpoint of this plan;
the previous YOLO11n latency study remains a separate diagnostic.

## 2. Frozen source inventory

The following checkpoint paths and hashes are present in the local repository
and were checked from the local files. They are source identities, not an
authorization to retrain or download a replacement checkpoint.

| Model | Frozen checkpoint | SHA256 | Bytes | Training run |
|---|---|---|---:|---|
| YOLOv8n | `results/yolov8n_cctsdb_clean_s42_v1/weights/best.pt` | `b2b7a1c77a19499ded33c9cc11c621757077aa871f4e7f7a1fcdbf94f53b383b` | 6,260,963 | `yolov8n_cctsdb_clean_s42_v1` |
| YOLO26n | `results/yolo26n_cctsdb_clean_s42_v1/weights/best.pt` | `2bb49f85f581469fc7942652d5fda4da44278d57fa8363e8f7295daa49f0d01e` | 5,399,038 | `yolo26n_cctsdb_clean_s42_v1` |

Both provenance records use `scripts/train_cctsdb.py`, seed `42`, the same
`configs/cctsdb2021_train.yaml` logical source and dataset manifest
`5d1b6f1c6df475efe14c8bc6e41c6312b5121dcb828041ec81bbcb2733ac0a2e`. The
architecture matrix records both runs as complete. The historical IVC
execution manifest contains paths for old YOLOv8n/YOLO26n engines, but those
records are not automatically reusable: each candidate must pass current
checkpoint, ONNX, TensorRT version, GPU identity, build-settings and direct
engine-hash checks. A serialized engine is never copied between architectures
or devices as if it were the same engine.

The three proposed calibration-selection manifests are existing train-only
Uniform selections, used as image-ID anchors rather than as reusable model
caches:

| Selection | Manifest | Canonical Git-blob SHA256 | Size |
|---|---|---|---:|
| U42 | `data/processed/cctsdb2021_clean/calibration/uniform_s42_n1024/calibration_manifest.json` | `ef1c4416986118814ed52367e9e04a935dfe2f2bf200f3752a4b5beb11b5ca18` | 1,024 |
| U43 | `data/processed/cctsdb2021_clean/calibration/uniform_s43_n1024/calibration_manifest.json` | `0ad1dfe04c0a983babed1ac4409ed199d260a2310653dc5b11527c9ee12b5889` | 1,024 |
| U44 | `data/processed/cctsdb2021_clean/calibration/uniform_s44_n1024/calibration_manifest.json` | `100914438e4d1b037e2b1bd046442d881b089ca6ba6a7f2ecab4a6270fb05a3c` | 1,024 |

Each manifest must be rechecked on the execution server for exact image bytes,
train-only source paths, no dev/test overlap and a materialized calibration
YAML. The local checkout does not contain all materialized calibration images
and YAML files for these selections, so server execution must stop if the
manifest cannot be resolved exactly. Calibration caches are architecture- and
model-specific; no YOLO11n cache is reused for YOLOv8n or YOLO26n.

## 3. Factorial design

Let `m` be model (`yolov8n`, `yolo26n`), `c` be calibration selection
(`U42`, `U43`, `U44`), `a` be representation and `r` be build repeat.

### INT8 cells

For each `(m, c, a)` cell, build three independent TensorRT engines:

- `baseline_int8`: no bbox/classification precision-head override;
- `bbox_fp32`: only the architecture-derived bbox-head convolution set is
  requested FP32, including FP32 outputs and OBEY evidence;
- `classification_fp32`: only the architecture-derived classification-head
  convolution set is requested FP32, including FP32 outputs and OBEY evidence;
- `both_fp32`: the union of the two disjoint target sets is requested FP32,
  including FP32 outputs and OBEY evidence.

There are `2 models × 3 selections × 4 arms × 3 repeats = 72` INT8 builds and
one dev capture for each build.

### FP16 control

For each model, build three independent FP16 reference engines with no
calibration selection (`c=NA`) and the same locked export/build shape and
runtime settings. These are six additional builds and six captures. The three
FP16 repeats estimate build variation of the reference; they do not introduce
a calibration factor.

Total planned workload is therefore **78 engine builds and 78 dev captures**:

| Component | Builds/captures per model | Both models |
|---|---:|---:|
| FP16 reference, 3 repeats | 3 | 6 |
| Four INT8 arms × 3 selections × 3 repeats | 36 | 72 |
| Total | 39 | **78** |

Each capture uses the full `CCTSDB2021/dev` split: 1,636 images and 2,706
instances. The plan therefore contains 127,608 model-image passes before any
calibration preprocessing. Six model/selection calibration-cache creation
phases are required; each cache is created once and then consumed read-only by
the 12 INT8 builds in that model/selection block.

### Isolation rules

1. Generate one private calibration cache for each `(model, c)` from the
   selected train-only IDs. Record cache bytes/hash and a zero-write,
   cache-only build audit for every consuming build. Do not regenerate the
   cache per repeat; otherwise calibration and build variation would be
   confounded.
2. Use a fresh empty private timing-cache input for every build. Persist the
   output timing cache and its hash, but never feed it to a later build. Do not
   use one ordinary timing cache across YOLOv8n and YOLO26n, or treat a shared
   timing cache as a tactic lock.
3. Export one architecture-specific ONNX from each frozen checkpoint with
   locked settings (`imgsz=640`, batch 1, opset 17, simplified, static,
   non-half). All arms and calibration selections for a model use that exact
   ONNX hash. A changed export requires a new reviewed protocol/version.
4. Keep builder settings, flags, workspace, optimization level, TensorRT and
   CUDA environment identical within a matched model block. Record any
   architecture-required exception explicitly and apply it to all four INT8
   arms for that architecture.
5. Pre-register a deterministic, interleaved order across calibration
   selections, arms and repeats before the first build. Do not reorder after
   observing AP, temperature or engine size. Persist sequence, timestamps and
   GPU telemetry; a single server should run all blocks serially.
6. Capture every completed engine once with the same dev image order and
   preprocessing. A payload that is identical across repeats is retained as
   evidence; it is not collapsed into one build or counted as proof that all
   future builds are byte-identical.

## 4. Architecture-specific head mapping

The YOLO11n names `/model.23/cv2.` and `/model.23/cv3.` must not be reused for
either target architecture without evidence. Before any build, the execution
record must derive the actual detection-head namespace from the model's frozen
ONNX/TRT network and persist:

- all candidate names matched by the architecture-specific head discovery;
- layer type for every candidate;
- selected bbox convolution names and selected classification convolution
  names;
- excluded helper, activation, reshape, concat and other non-convolution
  names;
- requested precision, output precision, effective precision and OBEY result
  for every selected layer;
- a stable mapping hash and the source ONNX hash.

Only actual convolution layers may be precision targets. A prefix that also
matches a helper or activation is not silently accepted. The two target sets
must be disjoint, nonempty and stable across all three repeats for a model.
If the network has no unambiguous mapping, the block stops before building and
is reported as `mapping_unresolved`; no guessed layer name, generic pretrained
`*.pt`, or YOLO11 naming shortcut is allowed.

The baseline and all three intervention arms must preserve the same
architecture-specific non-target constraints. The arm name alone is not proof
that branch arithmetic or accumulators used FP32; builder/effective evidence
and inspector output must be reported without over-interpreting it.

## 5. Dataset, runtime and evaluation contract

The scope is `CCTSDB2021/dev` only. Calibration uses only the locked training
IDs; no dev image, test image, test label, official negative or weather/official
size subset may influence calibration construction, arm definition, build
selection or stopping. The official test was used historically in pilot v1 and
must be described as historical provenance if relevant; this confirmation plan
does not use it to tune precision or calibration.

All representations within a model use the same capture contract:

- `imgsz=640`, batch 1, `task=detect`, `rect=false`, workers 0;
- `conf=0.001`, `iou=0.7`, `max_det=300`, no plots/JSON side effects;
- same sorted dev image IDs, exact image bytes and native matching checks;
- same `CCTSDB2021/dev` YAML/data provenance and frozen checkpoint binding.

The primary evaluator is the accepted `coco_xml_paired_image_bootstrap_v1`
diagnostic. Keep `ultralytics_full` and `coco_xml` as separate evaluators; do
not mix historical Ultralytics AP with COCO/XML size AP. Required endpoints are
full, XS and S:

- primary: full AP50–95 and `both_fp32 − baseline_int8`;
- secondary: full AP50, XS AP50/AP50–95 and S AP50/AP50–95;
- controls: bbox-only, classification-only and FP16 contrasts against the
  same model's baseline;
- diagnostic: native prediction matching and size-bin verification before
  any aggregation.

Use the accepted XML/COCO area convention and fixed image-level paired
bootstrap: PCG64 seed `20260916`, 1,000 resamples, same resampled image indices
for all arms, selections and repeats within a model. Preserve duplicate image
occurrences in each resample and report valid/undefined draws. Do not bootstrap
boxes independently or average per-image AP.

## 6. Separating build and calibration variation

For each model, arm and selection, retain all three build-level point estimates
and prediction payload hashes. Define:

- `cell_mean(m,c,a)`: arithmetic mean of the three build-repeat metrics;
- `build_SD(m,c,a)`: sample SD (`ddof=1`) across the three repeats;
- `selection_mean(m,a)`: arithmetic mean of the three `cell_mean` values;
- `calibration_SD(m,a)`: sample SD (`ddof=1`) across U42/U43/U44 selection
  means;
- selection range and per-selection build range.

Report these quantities for full, XS and S endpoints and for every contrast.
The calibration SD describes variation from the pre-registered image
selection; build SD describes repeated TensorRT construction under the same
selection/cache. Do not pool all 9 INT8 engines as independent calibration
replicates, and do not treat 3,000 calls or 3 captures as 3,000 builds.

Image-bootstrap CIs are conditional on the observed build and calibration
factors. The report must show them beside, not in place of, the empirical
build/calibration SDs. With only three selections and three repeats per cell,
variance components are descriptive pilot estimates; no formal variance claim
or universal determinism claim is allowed.

Use the same bootstrap draw for all build repeats before averaging at the cell
and selection levels. This preserves pairing while making the aggregation
rule auditable. Do not choose a seed, selection, build or arm by AP before
reporting the complete table.

## 7. Proposed decision margins, fixed before new data

These are proposed practical margins for Astra to lock or amend before
implementation; they are not thresholds inferred from the YOLO11n result and
are not substitutes for statistical testing.

- `δ_primary = 2.0` percentage points for full AP50–95. This is a deliberately
  visible improvement margin, larger than reporting-rounding noise, intended
  to distinguish a useful recovery from a negligible fluctuation.
- `δ_size = 3.0` percentage points for XS/S AP50. Smaller subsets have fewer
  instances and higher conditional variability, so the secondary margin is
  wider and is reported as a practical diagnostic, not a superiority test.
- `δ_fp16 = 1.0` percentage point for full AP50–95 and `2.0` points for XS/S
  AP50 when describing practical non-inferiority to the FP16 reference. This
  does not assert that INT8 must match FP16 on every endpoint.

The proposed confirmation disposition for each model is:

1. `both_fp32 − baseline_int8` has a selection-aggregated point estimate at
   least `δ_primary` on full AP50–95 and the paired image-bootstrap 95% CI does
   not cross zero;
2. at least two of the three pre-registered selections have a positive
   full-endpoint contrast and no selection is a negative reversal larger than
   `δ_primary`; this is a consistency description, not seed selection;
3. XS and S contrasts are reported; a loss larger than `δ_size` is flagged and
   prevents a claim of broad size robustness, even if full AP passes;
4. the FP16 contrast is reported with the `δ_fp16` non-inferiority description;
   failure means an accuracy trade-off, not automatic rejection of the
   precision-head mechanism;
5. bbox-only and classification-only are retained as mechanism controls. Their
   deltas are not subtracted to claim a causal decomposition, and `both_fp32`
   is not accepted merely because it has the largest AP.

If the conditions are not met, use `confirmation_not_established` or
`architecture_or_calibration_sensitive`, preserve all artifacts and stop.
Do not tune a new calibration policy, change nodes, select a favorable build,
or open the main 15-model matrix from an unfavorable result. Passing the
conditions supports a narrowly stated cross-architecture diagnostic; it does
not by itself authorize main15 or deployment selection.

## 8. One-server and multi-server execution policy

One server/GPU is preferred for all 78 builds and captures. If multiple
servers are available, split only by source model block: all YOLOv8n cells on
one matched host and all YOLO26n cells on another matched host. This keeps
every within-model arm/selection/repeat comparison on one device and runtime.

Before assignment, each host must record GPU UUID/model/driver, CUDA, Python,
TensorRT, Torch, Ultralytics, NumPy, dependencies, frozen checkpoint hash,
ONNX hash, calibration-manifest hashes and dataset hashes. A host with a
different GPU or incompatible TensorRT engine must rebuild its own reviewed
architecture-specific engines; it must not receive serialized engines from a
different host. If a model's repeats are split across devices, the result is
not a controlled build-variance estimate and must be marked invalid.

No server is presumed isolated. The current shared-lab desktop exception is
allowed only for exact current PID/path confirmation under the existing guard.
Any new training, inference, build or unknown compute process blocks that host;
low VRAM usage is not a waiver. Do not kill, pause, reprioritize, change
permissions, clocks or power limits. If a workload appears between snapshots,
record the violation and mark affected cells for review.

## 9. Prerequisites and artifact contract

Before implementation, the operator must be able to provide or verify:

1. the reviewed code/protocol commit and clean destination paths;
2. both frozen checkpoint files with the hashes in Section 2;
3. exact train YAML and dataset-manifest hashes;
4. U42/U43/U44 manifest files, materialized YAMLs and all image-byte hashes,
   with train-only/no-overlap validation;
5. one model-specific ONNX per frozen checkpoint and its export manifest/hash;
6. a compatible TensorRT environment and one GPU identity per host;
7. architecture-specific detection-head mapping evidence before the first
   precision-constrained build;
8. separate output roots for this plan, with no overwrite or implicit resume.

The eventual runner, which is explicitly out of scope here, must write a
study manifest, model/calibration inventory, mapping audit, per-build manifest,
calibration/timing cache audits, direct engine hash evidence, per-engine
capture/prediction/verification artifacts, telemetry snapshots, and a final
analysis summary. Each record must bind model, selection, arm, repeat, source
hashes, cache hashes, engine hash, environment and GPU identity. Push JSON,
YAML/manifests and logs only; do not push weights, ONNX, engine binaries or
raw image data.

The successful output state must remain review-required and must include all
cells. Missing cells, duplicate repeats, changed cache bytes, cache fallback,
unresolved mapping, wrong engine/GPU, incomplete telemetry, native matching
failure or new workload produces a preserved partial/invalid report rather
than a completed confirmation summary.

## 10. Workload, stopping and reporting

The count estimate is fixed at 6 calibration-cache creation phases, 78 engine
builds, 78 dev captures and 127,608 dev image-model passes, excluding warmups
and calibration preprocessing. No wall-clock estimate is supplied: the local
environment cannot benchmark TensorRT and previous latency timings are not a
build-time predictor.

Stop before any build if a prerequisite or mapping is missing. Stop the
affected block and preserve logs if a child fails; do not resume into the same
destination or bypass a failed guard. At completion, report all cell rows,
per-selection and pooled arm summaries, build SD, calibration SD/range,
image-bootstrap CI, FP16 contrasts, control contrasts, mapping/inspector
evidence, telemetry limitations and invalid flags.

The only permitted next decision is Astra's review of whether this protocol is
locked and worth implementing. This document does not provide a server
command, does not authorize GPU execution, does not open B/C or main15, and
does not authorize official-test or edge-device evaluation.
