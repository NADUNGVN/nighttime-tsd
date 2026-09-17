# Precision-head CPU numeric verification

- Study: `precision_head_confirmation_numeric_v1`
- Status: `failed`
- Execution status: `failed`; numeric verdict: `not_observed`.
- This diagnostic is not a scored accuracy test and does not authorize TensorRT work.
- Source and existing ONNX bytes are checked before and after; raw arrays are not published.

## Results

- `yolov8n`: execution `failed`, numeric status `not_observed`, forward counts `{}`.
- `yolo26n`: execution `failed`, numeric status `not_observed`, forward counts `{}`.

## Boundary

- A pass means the accepted native output contract agrees with the existing ONNX output on this eight-image CPU fixture under the locked tensor path.
- It does not prove dataset-wide accuracy, calibration equivalence, TensorRT compatibility, GPU behavior or engine reproducibility.
- Any fail or unresolved result remains evidence for review; no model, output or preprocessing change is applied automatically.
