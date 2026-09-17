# Precision-head numeric localization v2

Status: local implementation and CPU-test package for Astra review. This is a
new bounded diagnostic, not a replacement for either historical result. The
strict `precision_head_confirmation_numeric_v2` verdict remains `FAIL`, and
`precision_head_numeric_localization_v1` is preserved unchanged.

## Scope and fixed call budget

The diagnostic covers only YOLOv8n/U42 image `00009` and YOLO26n/U42 image
`00006`, using the already accepted checkpoint, ONNX, fixture, preprocessing,
input binding and numerical equation. It does not export, rebuild, import
TensorRT, use a GPU, run AP/official test data, search thresholds, use NMS or
open a matrix. The new output is
`results/measurement_audit_v1/precision_head_numeric_localization_v2/`; it
refuses overwrite and links the preserved v1 output.

The maximum is two ordinary native model forwards and three ONNX sessions:

| Operation | YOLOv8n | YOLO26n | Total |
| --- | ---: | ---: | ---: |
| Ordinary native model forward | 1 | 1 | 2 |
| Same-head decoder replay | 0 | 1 | 1 |
| Same-head postprocess replay | 0 | 1 | 1 |
| Original ONNX session | 1 | 1 | 2 |
| Private derived ONNX session | 0 | 1 | 1 |

Decoder/postprocess replays are not additional model forwards. A failure
preserves the partial child state and never retries or silently adds calls.

## Native semantic adapter

The installed Ultralytics `Detect` source is recorded by method evidence for
`forward_head`, `_inference`, `_get_decode_boxes`, `decode_bboxes`,
`postprocess` and `get_topk_index`. Actual head flags, `reg_max`, strides,
anchors and shape state are recorded and checked.

For YOLOv8n, the one ordinary forward's `[1,7,8400]` primary is used directly:
channels `0:4` are decoded `xywh` boxes and channels `4:7` are post-sigmoid
class probabilities in 640x640 input-pixel space. The retained debug
dictionary is recorded only as raw regression parameters/logits and feature
map evidence; raw 64-channel box parameters are never relabelled as decoded
boxes.

For YOLO26n, the actual returned `one2one` dictionary is retained. The same
head object's installed `_inference(one2one)` is replayed once under CPU
`no_grad` to produce decoded `[1,7,8400]` values: `xyxy` boxes and sigmoid
probabilities. The installed `postprocess` is then replayed once on that
decoded tensor and must equal the ordinary primary `[1,300,6]` exactly. The
replay records source/method hashes, raw input and decoded output summaries,
and before/after anchor/stride/shape state. No NumPy sigmoid, manual decoder,
head mutation, weight change, reload, fusion or export-mode switch is allowed.

Raw logits, decoded boxes and probabilities retain distinct roles. Missing
features, non-finite tensors, unsupported agnostic/single-label flags, wrong
coordinate metadata, double sigmoid, replay drift or state drift fail closed.
Raw summaries and failure state are retained; cross-side endpoints are
explicitly `unresolved/not_admissible` when a semantic guard fails.

## Cross-side admissibility

The original native-versus-ONNX strict comparison remains an independent
supplemental `FAIL`. YOLO26 Top-K/Gather analysis reconstructs each side from
its own tensors and exact indices, with no nearest-neighbor rematching. Only
after native decoder/postprocess self-consistency and exact unchanged
original-vs-derived ONNX output pass may same-anchor and selection-alignment
results be labelled admissible. Derived-graph output drift leaves raw
diagnostics but marks those cross-side endpoints `unresolved/not_admissible`.

## Tests and execution boundary

The critical tests use the existing pinned CPU environment
`local/measurement_audit_env` with real Ultralytics 8.4.102 `Detect` methods
and deterministic synthetic multiscale tensors: reg_max16/raw64 V8-like
parameters, reg_max1/raw4 end2end V26-like parameters, nonzero strides and
anchors, negative/extreme logits and ties. They do not load or forward a
frozen checkpoint. Additional tests cover malformed producer branches,
non-finite values, coordinate/flag errors, double sigmoid, replay drift,
Top-K/index reconstruction, instrumentation drift and model-scoped child
failure state. Existing numeric and graph-audit regressions remain required.

The server command is foreground CPU-only with CUDA hidden and the existing
two-thread ONNX Runtime contract. The operator must verify the new output root
is absent and the pinned binary/input hashes before running. Luna does not SSH
or run the server diagnostic; server artifacts are pushed for post-run audit.
