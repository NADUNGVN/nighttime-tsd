# Precision-head numeric localization

- Study: `precision_head_numeric_localization_v1`
- Status: `failed`
- The accepted numeric_v2 verdict remains `FAIL`; this package is supplemental localization evidence, not a replacement gate.

## Cases

- `yolov8n/00009`: execution `failed`, strict verdict `fail_preserved`, counts `{'native_model_forwards': {'attempted': 1, 'completed': 1}, 'onnx_original_session_runs': {'attempted': 0, 'completed': 0}, 'onnx_derived_session_runs': {'attempted': 0, 'completed': 0}}`.
- `yolo26n/00006`: execution `completed`, strict verdict `fail_preserved`, counts `{'native_model_forwards': {'attempted': 1, 'completed': 1}, 'onnx_original_session_runs': {'attempted': 1, 'completed': 1}, 'onnx_derived_session_runs': {'attempted': 1, 'completed': 1}}`.

## Boundary

- No TensorRT, export, rebuild, optimization sweep, AP/test evaluation, threshold search, NMS or matrix was run.
- Private full tensors and the YOLO26 derived graph are excluded from the publishable inventory.
- Any association not resolved by exact anchor/class indices remains `unresolved`; no nearest-neighbor matching is used.
