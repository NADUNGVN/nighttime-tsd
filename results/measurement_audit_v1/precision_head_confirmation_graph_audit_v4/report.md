# Precision-head preserved graph audit

- Study: `precision_head_confirmation_graph_audit_v1`
- Status: `audit_only_completed`
- Source root: `results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2`
- Mode: audit-only; existing ONNX read/check/shape-inference/mapping in memory.
- Exporter, model forward, calibration loader, TensorRT and scored execution were not called.
- Initializer/Constant protobuf shape and dtype metadata are recorded; explicit scalar shape `[]` is not treated as unknown rank.

## Model results

- `yolov8n`: audit `audit_only_completed`, mapping `verified`, errors `0`.
- `yolo26n`: audit `audit_only_completed`, mapping `verified`, errors `0`.

## Boundary

- Unresolved mapping is persisted as evidence; it does not authorize a retry or scored run.
- ONNX/PT/engine/cache binaries are source/server-only and are not diagnostic artifacts for Git.
- A newly computed ONNX hash describes the bytes observed during this audit and does not establish historical export provenance.
