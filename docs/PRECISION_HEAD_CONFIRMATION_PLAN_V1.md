# Precision-Head Confirmation Plan V1

Status: `design_locked_readiness_implementation`

This document is the design-locked readiness protocol following A2L-022. It
authorizes only local CPU readiness implementation. It does not authorize a
server run, scored build/capture matrix, ONNX export, TensorRT build, or
deployment selection. The reviewed preparation output must be accepted before
an operator receives a server execution command.

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

The scored workload is **78 engine builds and 78 dev captures**. In addition,
six auxiliary baseline cache-generation builder invocations are required and
are never scored or captured. The total builder workload is therefore **84
invocations**, not 78:

| Component | Builds/captures per model | Both models |
|---|---:|---:|
| FP16 reference, 3 repeats | 3 | 6 |
| Four INT8 arms × 3 selections × 3 repeats | 36 | 72 |
| Total | 39 | **78** |

Each capture uses the full `CCTSDB2021/dev` split: 1,636 images and 2,706
instances. The plan therefore contains 127,608 model-image passes before any
calibration preprocessing. Six model/selection auxiliary cache-generation
builds are required; each cache is created once and then consumed read-only by
the 12 scored INT8 builds in that model/selection block. The auxiliary build
has its own provenance/log/cache, but its engine is not an accuracy observation
and cannot be selected by AP.

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

The local CPU readiness probe against the frozen checkpoints provides a
pre-export contract check, not an ONNX mapping claim. Under the locked
Ultralytics `8.4.102` runtime it observed:

- YOLOv8n: `Detect` index `22`, `end2end=false`, active `cv2`/`cv3`, no
  `one2one_*` branches, and a tuple of raw decoded primary tensor
  `[1,7,8400]` plus the head dictionary (`boxes`, `scores`, `feats`). The
  head output contains no NMS/top-k postprocess.
- YOLO26n: `Detect` index `23`, `end2end=true`, active `one2one_cv2`/
  `one2one_cv3`, with `cv2`/`cv3` retained as inactive auxiliary branches, and
  a tuple of postprocessed top-k primary detections `[1,300,6]` plus
  `one2many`/`one2one` head dictionaries. The active one-to-one branch is
  selected by the frozen head's end-to-end path.

These observations are checked together with head flags, branch availability,
output representation, nested tensor paths and postprocess semantics; a shape
match alone never establishes the contract. The final server prepare phase
must repeat this evidence on the architecture-specific exported ONNX and
parser/validator path before any constrained build. Until then, effective
precision and ONNX graph-level mapping remain `unknown`/deferred.

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
- `within_selection_build_SD(m,c,a)`: sample SD (`ddof=1`) across the three
  independent repeats;
- `selection_mean(m,a)`: arithmetic mean of the three `cell_mean` values;
- `between_selection_mean_SD(m,a)`: sample SD (`ddof=1`) across U42/U43/U44
  selection means;
- selection range and per-selection build range.

Report these quantities for full, XS and S endpoints and for every contrast.
`between_selection_mean_SD` still contains residual build variation and is not
purified calibration variance; `within_selection_build_SD` describes repeated
TensorRT construction under the same selection/cache. Do not subtract SDs,
claim causal isolation, pool all 9 INT8 engines as independent calibration
replicates, or treat 3,000 calls or 3 captures as 3,000 builds.

Image-bootstrap CIs are conditional on the observed build and calibration
factors. The report must show them beside, not in place of, the empirical
build/calibration SDs. With only three selections and three repeats per cell,
variance components are descriptive pilot estimates; no formal variance claim
or universal determinism claim is allowed.

Use the same bootstrap draw for all build repeats before averaging at the cell
and selection levels. This preserves pairing while making the aggregation
rule auditable. Do not choose a seed, selection, build or arm by AP before
reporting the complete table.

## 7. Decision reporting, fixed before new data

Keep `+2.0` percentage points on full AP50–95 only as a reviewer-set
engineering screening target. It is not a domain-validated utility threshold,
significance threshold or paper/deployment gate. Keep the descriptive rule of
at least two of three selections positive and no reversal larger than 2.0 pp,
but do not turn it into a hypothesis test or select favorable selections.

Report continuous point estimates and paired image-bootstrap CIs for full, XS
and S endpoints. The FP16 gap and CI are reported continuously; there is no
FP16 non-inferiority/equivalence margin. XS/S effects are reported with their
point estimates, CIs and adverse effects; there is no `δ_size` gate and a full
endpoint result does not establish size robustness.

The primary contrast remains `both_fp32 − baseline_int8` on full AP50–95, with
bbox-only and classification-only as mandatory controls. Do not subtract
control effects as a causal decomposition and do not accept `both_fp32` merely
because it has the largest AP. If the descriptive evidence is not coherent,
use `confirmation_not_established` or
`architecture_or_calibration_sensitive`, preserve all artifacts and stop.
Passing a narrow cross-model diagnostic does not authorize main15,
deployment selection or another research phase.

## 8. One-server and multi-server execution policy

One server/GPU is preferred for all 78 builds and captures. If multiple
servers are available, split only by source model block: all YOLOv8n cells on
one matched host and all YOLO26n cells on another matched host. This keeps
every within-model arm/selection/repeat comparison on one device and runtime.

Before scored assignment, each host must record GPU UUID/model/driver, CUDA,
Python, TensorRT, Torch, Ultralytics, NumPy, dependencies, frozen checkpoint
hash, calibration-manifest hashes and dataset hashes. After the reviewed
prepare/export phase, it must also bind the model-specific ONNX hash and
mapping evidence. A host with a
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

The local readiness phase requires only raw, immutable inputs and the reviewed
runtime/config contract. The reviewed server prepare phase creates the
architecture-specific ONNX and graph mapping evidence; the operator must not
perform an unreviewed export merely to satisfy a circular prerequisite.

Before server preparation, the operator must be able to provide or verify:

1. the reviewed code/protocol commit and clean destination paths;
2. both frozen checkpoint files with the hashes in Section 2;
3. exact train YAML and dataset-manifest hashes;
4. U42/U43/U44 manifest files, materialized YAMLs and all image-byte hashes,
   with train-only/no-overlap validation;
5. a compatible TensorRT environment and one GPU identity per host;
6. separate output roots for readiness, prepare and scored work, with no
   overwrite or implicit resume.

The reviewed prepare phase must then create and validate one model-specific
ONNX per frozen checkpoint, export manifest/hash, active output dataflow
mapping, parser/validator evidence and model-specific constraints. A missing
or ambiguous head/output contract is `head_contract_unresolved` and blocks
all builds; it is not repaired by changing output shape or adding a second
postprocess step.

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

The count estimate is fixed at 6 auxiliary calibration-cache builder
invocations, 78 scored engine builds, 78 dev captures, **84 total builder
invocations** and 127,608 dev image-model passes, excluding warmups and
calibration preprocessing. No wall-clock estimate is supplied: the local
environment cannot benchmark TensorRT and previous latency timings are not a
build-time predictor.

Stop before any build if a prerequisite or mapping is missing. Stop the
affected block and preserve logs if a child fails; do not resume into the same
destination or bypass a failed guard. At completion, report all cell rows,
per-selection and pooled arm summaries, build SD, calibration SD/range,
image-bootstrap CI, FP16 contrasts, control contrasts, mapping/inspector
evidence, telemetry limitations and invalid flags.

The only permitted next decision is Astra's review of the readiness
implementation and its model-specific head/output evidence before authorizing
server preparation. This document does not authorize GPU execution, does not
open B/C or main15, and does not authorize official-test or edge-device
evaluation.

## Appendix A. D1–D3 readiness correction (A2L-023)

The readiness implementation now applies the following pre-server checks. This
appendix records the implementation contract; it does not authorize export,
TensorRT build, capture, or scored execution.

### D1. Canonical split and calibration materialization

The accepted FP16 dev capture at commit
`5eb7ec36da7eed1701f6383b9db37ca3cfe31186` is the canonical reference for
the `CCTSDB2021/dev` image IDs, label stems, decoded shapes, and 2,706 label
instances. Readiness compares the current checkout's ordered IDs, label stems,
decoded shapes, class IDs, finite normalized YOLO boxes, and per-file byte
inventory. The resulting inventory hash is explicitly scoped to the current
checkout; it is not relabeled as a historical server image-byte hash. Derived
box corners are not used as an additional rejection rule because the canonical
label format is center/width/height and rounded edge arithmetic can exceed one
slightly; downstream evaluator clipping remains the contract.

U42, U43, and U44 remain the only selections. Each selected path must normalize
to `train/images/<basename>` or `train/labels/<basename>`, remain inside the
intended train root, have a matching image/label stem, and be present in the
actual train inventory. Dev and test exclusion IDs are checked by membership;
test pixels and labels are not read to construct the policy. If a producer
`calibration.yaml` is present, readiness parses its schema, resolves the exact
1,024 selected source images, and compares materialized bytes to train source
bytes. Missing materialization is reported as missing, while malformed YAML,
extra/substituted images, wrong selection, or byte mismatch is invalid. No
materialization is rebuilt or repaired locally. The preprocessing recipe stays
`unresolved_pending_server_prepare` until the concrete server producer/helper
and source hash are observed.

### D2. Immutable identity, provenance, and fail-closed status

Locked scientific sections are checked against the reviewed immutable section
hashes, so mutating the same JSON used as both expectation and observation
cannot silently pass. The readiness manifest records the actual config path,
config-byte SHA256, semantic SHA256, execution Git commit, script SHA256, and
UTC creation time. Environment/package observations are recorded before any
model probe. A disabled model probe is an explicit unresolved
`model_probe_not_run` state, not an empty-error ready state.

`raw_inputs_ready_for_server_prepare` is separate from scored authorization.
The latter remains false and `scored_matrix_gate` remains
`blocked_deferred_graph_validation` until the reviewed server prepare phase
has produced model-specific ONNX/dataflow mapping and parser evidence. Local
readiness completion therefore cannot be reported as TensorRT or end-to-end
verification.

### D3. Exact model-block and interleaved schedule

Schedule version `model_block_aux_then_interleaved_rounds_v2` contains, per
model, auxiliary U42/U43/U44 calibration-cache creation jobs first. These jobs
do not capture, score, or provide timing outputs to scored jobs. Each model then
has three rounds of 13 scored cells: FP16 followed by the four arms for U42,
U43, and U44. Round 1 uses shift 0, round 2 shift 4, and round 3 shift 8 over
the canonical 13-cell order; round number is also the build repeat number.
This yields 3 auxiliary + 39 scored jobs per model, 84 builder invocations and
78 captures across both models.

Validation compares every expected model/selection/arm/repeat key and every
phase, cache, capture, timing-policy, dependency, and sequence field. Missing,
extra, wrong-selection, wrong-repeat, or malformed jobs are rejected even when
aggregate counts happen to match. The schedule hash is new and supersedes the
unrun grouped schedule; no historical result is rewritten.
