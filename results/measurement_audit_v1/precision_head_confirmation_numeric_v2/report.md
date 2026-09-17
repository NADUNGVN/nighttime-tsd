# Precision-head CPU numeric verification

- Study: `precision_head_confirmation_numeric_v1`
- Status: `completed`
- Execution status: `completed`; numeric verdict: `fail`.
- This diagnostic is not a scored accuracy test and does not authorize TensorRT work.
- Source and existing ONNX bytes are checked before and after; raw arrays are not published.

## Results

- `yolov8n`: execution `completed`, numeric status `fail`, stage `completed`, forward counts `{'source_cpu_fp32': {'attempted': 8, 'completed': 8}, 'onnx_cpu': {'attempted': 8, 'completed': 8}}`.
- `yolo26n`: execution `completed`, numeric status `fail`, stage `completed`, forward counts `{'source_cpu_fp32': {'attempted': 8, 'completed': 8}, 'onnx_cpu': {'attempted': 8, 'completed': 8}}`.

## Boundary

- A pass means the accepted native output contract agrees with the existing ONNX output on this eight-image CPU fixture under the locked tensor path.
- It does not prove dataset-wide accuracy, calibration equivalence, TensorRT compatibility, GPU behavior or engine reproducibility.
- Any fail or unresolved result remains evidence for review; no model, output or preprocessing change is applied automatically.
