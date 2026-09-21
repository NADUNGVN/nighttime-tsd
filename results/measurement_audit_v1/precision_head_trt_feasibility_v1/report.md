# TensorRT FP16 feasibility smoke

- Study: `precision_head_trt_feasibility_v1`
- Status: `completed`; validity: `validity_checks_passed; scientific_assessment_descriptive`.
- This is a bounded feasibility smoke, not a scored accuracy result or timing benchmark.
- Historical numeric/localization `FAIL` verdicts are preserved.

## Model status

- `yolov8n`: `completed`; validity `validity_checks_passed; scientific_assessment_descriptive`; counts `{'native_forward': {'attempted': 0, 'completed': 0}, 'onnx_cpu_reference_call': {'attempted': 8, 'completed': 8}, 'trt_application_enqueue': {'attempted': 8, 'completed': 8}}`.
- `yolo26n`: `completed`; validity `validity_checks_passed; scientific_assessment_descriptive`; counts `{'native_forward': {'attempted': 0, 'completed': 0}, 'onnx_cpu_reference_call': {'attempted': 8, 'completed': 8}, 'trt_application_enqueue': {'attempted': 8, 'completed': 8}}`.

## Boundary

- FP16 builder enablement is recorded with parser, binding and inspector evidence; it is not interpreted as every layer executing FP16.
- No raw tensors or engine binaries are publishable. Numerical comparison is descriptive only; no equivalence threshold is applied.
