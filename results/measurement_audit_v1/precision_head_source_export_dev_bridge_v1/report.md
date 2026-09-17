# Precision-head source/export dev bridge

- Study: `precision_head_source_export_dev_bridge_v1`
- Execution: `completed`; validity: `validity_checks_passed; scientific_assessment_descriptive`.
- This is a descriptive native-FP32 versus accepted-ONNX-FP32 application-level bridge on CCTSDB2021/dev. It does not replace the historical numeric FAIL.
- No export, TensorRT import/build, GPU, calibration, training, official test, repeat or precision matrix was run by this package.

## Model results

- `yolov8n`: status `completed`, validity `validity_checks_passed; scientific_assessment_descriptive`, forward counts `{'native_cpu_forward': {'attempted': 1636, 'completed': 1636}, 'onnx_cpu_session_run': {'attempted': 1636, 'completed': 1636}}`.
  - All-size signed delta (ONNX − native): AP50 `0.0`, AP50–95 `9.09797841508464e-08`; no pass threshold applied.
- `yolo26n`: status `completed`, validity `validity_checks_passed; scientific_assessment_descriptive`, forward counts `{'native_cpu_forward': {'attempted': 1636, 'completed': 1636}, 'onnx_cpu_session_run': {'attempted': 1636, 'completed': 1636}}`.
  - All-size signed delta (ONNX − native): AP50 `0.0`, AP50–95 `-1.0617890777719907e-07`; no pass threshold applied.

## Interpretation boundary

- Hard validity failures (identity, split/XML membership, provider, non-finite output or incomplete records) block scientific interpretation.
- A valid execution reports measured export drift only. Ordered examples do not perform rematching and do not replace the COCO/XML AP estimator.
