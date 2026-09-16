# Precision-head confirmation graph preparation v1

Status: implementation ready for reviewer confirmation; this document does
not authorize server export or any TensorRT run.

## Purpose and boundary

This phase prepares one architecture-specific static float ONNX graph for each
frozen model in the accepted CPU-readiness artifact. It verifies the exported
input/output schema and records active detection-head dataflow so that the
later INT8 precision arms target owned convolution layers rather than names
copied from YOLO11.

The phase is prepare-only:

- YOLOv8n and YOLO26n weights remain frozen; no generic download or retraining.
- Export settings are `imgsz=640`, batch `1`, float input, opset `17`,
  `simplify=true`, `dynamic=false`, `half=false`, with the frozen model's
  `end2end` flag preserved.
- No TensorRT import, engine build, calibration-cache creation, capture,
  benchmark, official test, bootstrap, or 15-model expansion occurs.
- ONNX/private source binaries stay server-only. JSON, report, log and schema
  files are the review artifacts to push.

The parent process performs only filesystem/Git/CPU-readiness checks and
dispatches one isolated child per selected model. The child is the only place
that imports Ultralytics/ONNX and calls the ONNX exporter. A successful
manifest still has `scored_run_authorized=false`.

Before those imports each child enforces `YOLO_AUTOINSTALL=0`,
`ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1`, `PIP_NO_INDEX=1`, and
`PIP_DISABLE_PIP_VERSION_CHECK=1`. It also inherits
`CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=2`, and `MKL_NUM_THREADS=2`.
The manifest records that CUDA API-query status is not instrumented; it does
not claim that imported libraries made zero internal CUDA queries. Metadata-only
checks require the locked Torch/Ultralytics/NumPy/pycocotools versions, `onnx`,
`onnxslim`, and one ONNX Runtime distribution before the exporter is imported.
No install or network mutation is attempted.

## Locked inputs

The runner accepts only:

```text
--readiness-root <accepted readiness directory>
--model yolov8n|yolo26n|all
--out-dir <new output directory>
```

It does not accept arbitrary checkpoint, engine, ONNX or precision arguments.
The readiness root must be the accepted five-file artifact from Git commit
`7c0ea9e7fe86dfa6358ec1ee90473f9243a53f76`, with the raw hashes recorded in
the runner. Its manifest must be `ready_for_server_prepare_review`, have no
missing/unresolved checks, and retain `scored_run_authorized=false`.

Before export the runner rechecks, for the selected model block:

1. current config bytes and semantic hash against the accepted readiness
   manifest and the reviewed readiness helper hash;
2. frozen checkpoint path, byte count and SHA256 against both config and
   accepted model contract;
3. current dev/train/test-exclusion inventory, split intersections and all
   U42/U43/U44 canonical manifests/materializations, compared exactly with the
   accepted snapshot, including IDs/order and recorded source/materialized byte
   hashes;
4. frozen PyTorch head flags, active branches, native output shape/meaning and
   model-specific adapter compatibility.

The current host, Git revision, package versions/module hashes, selected
checkpoint hashes, input manifest hashes and timestamp are written to the
provenance. A different host must carry its own accepted CPU-readiness
artifact; SERVER-01 evidence is not silently reused for another server.

## ONNX and graph audit

The child copies the verified checkpoint into a private model workspace and
exports from that copy. It refuses an existing output root/model directory and
keeps partial output on failure. The reported exporter path must remain inside
the private workspace before the ONNX is copied to the new model output.

`onnx.checker.check_model` is required. The audit records all graph inputs and
outputs, input/output dtypes, static shapes, small constant initializers, node
count/op types, Q/DQ nodes and the ONNX SHA256.
Q/DQ is rejected for this float export. A graph output shape alone is not a
mapping proof.

For every active source branch the audit:

- derives the fully qualified source convolution list from the accepted
  frozen-model contract;
- matches each source module to exactly one exported `Conv` node after only
  separator/terminal-`Conv` normalization;
- records all prefix candidates and excludes `Sigmoid`, `Reshape`, `Concat`,
  and other non-convolution helpers from precision targets;
- traces each matched node through consumers to the unique relevant primary
  output, not to a debug-only output;
- finds exactly one output-linked branch-owned channel merge with explicit
  axis, input shapes, order and locked spans `bbox=[0,4]`,
  `classification=[4,7]`;
- audits post-merge `Reshape`, `Transpose`, `Slice`, `Gather` and `TopK`
  semantics from explicit attributes/constant evidence. Unknown semantics are
  `mapping_unresolved`; shape alone is never treated as proof;
- records source-to-export node names, reachability, downstream operations and
  a stable mapping payload.

Missing, duplicate, renamed/fused-without-lineage, unreachable or ambiguous
matches produce `mapping_unresolved`; the runner never guesses a layer name or
forces YOLO11 counts. Bbox and classification target sets must be non-empty
and disjoint by branch ownership, and target generation requires an internally
consistent verified mapping. `baseline_int8` has no active branch FP32
override; its INT8 eligibility and separate sigmoid-FP32 protection remain
builder/inspector constraints, not claims about every other convolution. The
three intervention sets are derived only from the verified ownership map.

For YOLO26n, the active `one2one_cv2/one2one_cv3` branches own the pre/post-
processing channel spans while later TopK/Gather nodes may share final-output
ancestry. Shared reachability is recorded as expected and is not treated as a
precision-target overlap. Inactive `cv2/cv3` branches remain explicitly
audited and excluded.

## Adapter contract

The native adapter is model-specific:

| Model | Native primary output | Semantics | Second NMS |
|---|---|---|---|
| YOLOv8n | `[1,7,8400]` | four box-coordinate channels followed by three class-score channels; no model-head NMS | forbidden |
| YOLO26n | `[1,300,6]` | `[x1,y1,x2,y2,score,class_id]` after the frozen end-to-end TopK path | forbidden |

The runner validates tuple-tensor-plus-dictionary representation, shape/dtype,
CPU placement, output-specific head keys and coordinate/class semantics in
numeric probes when such output is available. Finite coordinates in `[0,640]`,
ordered corners, finite scores in `[0,1]`, and integer class IDs in `{0,1,2}`
are required; invalid and empty paths are explicit tests. The prepare adapter
records a declared contract only, with numeric forward validation deferred.
Real TensorRT parser/forward compatibility must not be called
`parser-ready-to-score` from shape evidence alone.

## Calibration recipe boundary

The runner binds existing U42/U43/U44 manifests and materialized YAML/hash
records and records the current SHA256 of `scripts/uniform_build_repeat.py`.
It records the concrete locked helper-level recipe: Ultralytics validation
decode, BGR source image, centered `LetterBox(640,640, scaleup=false,
auto=false, padding=114, INTER_LINEAR)`, HWC-uint8 to CHW-uint8 batching,
batch `1`, workers `0`, `drop_last=true`, manifest order, and float32
`/255.0` streaming in the existing helper. Because the producer source and
per-image trace are not executed in graph preparation, the status is accurately
`graph_only_completed_recipe_unresolved`; a reviewed server child must bind the
producer source hashes and one-image trace before any cache build. No tensor
file or official-test pixels/labels are created/read here.

## Outputs and failure handling

For each selected model the output contains `model_prepare.json`,
`graph_schema.json`, `model.onnx` and a retained private workspace. The root
contains `prepare_plan.json`, one child log per model,
`graph_preparation_manifest.json` and `report.md`. The manifest lists hashes
and marks ONNX/private source copies `server_only`; only JSON/report/log/schema
files are candidates for Git.

If a child fails, its model directory and any ONNX/partial files remain, and a
structured `failure.json` records stage, exception, partial files and the
no-resume rule. The parent writes a root failure report and exits non-zero. An
existing root is never silently resumed or overwritten.

The completed status is exactly
`graph_preparation_completed_review_required`. It is valid only when every
selected model has verified graph mapping and adapter preparation. It never
changes the later 84-build schedule and never authorizes scored execution.

## Candidate operator command (not yet authorized)

After Astra reviews the implementation and the exact hashes, an operator may
run the following foreground command on a host with its accepted readiness
artifact. This is a candidate command only; it must not be run from this
document before review:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git rev-parse HEAD && local/g0_size_env/bin/python scripts/prepare_precision_head_confirmation_graph.py --readiness-root results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2 --model all --out-dir results/measurement_audit_v1/precision_head_confirmation_graph_prep_v1
```

No `nohup` is implied. The operator must push only the allowed JSON/report/log/
schema artifacts after a successful prepare, preserving ONNX/private binaries
on the server for the subsequent review.
