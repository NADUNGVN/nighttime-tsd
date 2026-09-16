# Precision-head confirmation readiness

- Study: `precision_head_confirmation_v1`
- Status: `ready_for_server_prepare_review`
- Scored matrix gate: `blocked_deferred_graph_validation`
- Scope: CPU-only local readiness; no GPU, ONNX export, TensorRT import/build, or scored matrix.

## Head/output evidence

Verified frozen model contracts: `yolov8n, yolo26n`.
The contract is checked from head flags, active branch selection, tensor-plus-dict output representation, semantic postprocess path and shapes. Shape alone is not accepted as routing evidence.

- `yolov8n`: status `verified`, head `Detect` index `22`, end2end `False`, active `['cv2', 'cv3']`, primary `[1, 7, 8400]`, output kind `decoded_raw_plus_head_dict`.
- `yolo26n`: status `verified`, head `Detect` index `23`, end2end `True`, active `['one2one_cv2', 'one2one_cv3']`, primary `[1, 300, 6]`, output kind `end2end_topk_detections_plus_head_dict`.

## Workload accounting

The explicit schedule contains 6 auxiliary calibration invocations, 72 scored INT8 invocations, and 6 scored FP16 invocations: 84 builder invocations and 78 captures. Auxiliary engines are not scored and are not selected by AP.

## Blockers and deferred artifacts

- No missing local prerequisite recorded.
- No unresolved check recorded.
- ONNX export, graph-level mapping, TensorRT/GPU identity and all auxiliary/scored execution remain deferred to a separately reviewed server prepare/run phase.
- `scored_run_authorized` is `false`; this report is not TensorRT end-to-end evidence.

## Runtime compatibility

- Native CPU head probe compatibility: `cpu_probe_supported`; supported Ultralytics: `8.4.102`.
- Server CUDA/TensorRT/GPU identity remains `not_verified_by_cpu_readiness`; local Torch/NumPy differences are recorded as package observations, not silently treated as server verification.

## Statistics protocol

Report `within_selection_build_SD` across the three repeats and `between_selection_mean_SD` across the three selection means. Keep the +2 pp full-endpoint value only as a pre-set engineering screening target; report FP16 gaps and XS/S effects continuously without a non-inferiority or size gate.
