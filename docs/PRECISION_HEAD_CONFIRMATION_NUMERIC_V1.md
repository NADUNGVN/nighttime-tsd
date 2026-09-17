# Precision-head CPU numeric confirmation v1

Status: implementation package submitted for Astra review. This document defines a bounded CPU diagnostic; it does not authorize server execution until the reviewer accepts this package.

## Scope and boundary

The runner compares the frozen YOLOv8n and YOLO26n PyTorch FP32 native primary output with the already accepted ONNX graph on CPU. It uses eight fixed train images and three preprocessing trace anchors. It does not train, export, rebuild, calibrate, score AP, read official-test/negative-test images or labels, run TensorRT, use a GPU, apply NMS, or create a calibration cache.

The existing `84`-builder/`78`-capture schedule is unchanged. A numeric pass is only evidence that this small input fixture follows the same verified tensor path and that the selected native output agrees with the existing ONNX output. It is not dataset-wide accuracy, TensorRT compatibility, calibration equivalence or engine reproducibility.

## Fixed inputs

The parent binds, before dispatching children:

- accepted readiness root `results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2`;
- accepted graph-v4 five-file artifact at commit `6780b813c5f1cb2d915b72832eedd79525eaecbd`;
- frozen checkpoints and exact hashes already accepted by readiness;
- existing ONNX files from `precision_head_confirmation_graph_prep_v2`, with expected hashes `e22d53bbeb333f44783535d911d5e318ebb7d500e8fcb3d1cbb1284f5248d603` (YOLOv8n) and `1b2467ccd62bd1e53f3bde3e3f22e1b42129711d3e368a4b4666d025099ce5cc` (YOLO26n).

The forward fixture is the first eight distinct train image IDs in canonical U42 manifest order:

`00006, 00009, 00028, 00036, 00054, 00061, 00098, 00104`.

The runner verifies the current source and materialized image bytes against the accepted readiness rows before any forward. Canonical source IDs are resolved under `data/processed/cctsdb2021_clean`; materialized calibration copies are resolved separately under the corresponding `calibration/uniform_s*_n1024/images` directory. The resolver rejects absolute/traversal paths and wrong-root decoys. It records the IDs, manifest positions, original source paths, byte hashes and sizes, and repeats the source/materialized hash check after the child work. The first image of each U42/U43/U44 selection is also traced. If a trace ID overlaps the forward fixture, its inference tensor trace is reused while its selection binding remains separately recorded. Labels are not read.

The output root is exclusive and no-overwrite: `results/measurement_audit_v1/precision_head_confirmation_numeric_v1`. Existing output means stop; partial output is preserved and is not resumed silently.

## Producer and preprocessing contract

The child uses the pinned Ultralytics 8.4.102 inference producer path and records source hashes for the relevant functions. The observed list-input path is:

1. Ultralytics `imread`, producing a three-channel BGR array;
2. `BasePredictor.pre_transform` with the model's observed stride, `new_shape=(640,640)`, `auto=False`, `scale_fill=False`, `scaleup=True`, `center=True`, padding value `114` and `cv2.INTER_LINEAR`;
3. BGR to RGB;
4. BHWC to BCHW and contiguous layout;
5. CPU `float32`, divide by `255`, and batch dimension.

The exact decoded shape, resize ratio, rounded resized shape, each padding side, stage dtype/shape/range and tensor byte hash are recorded per trace. The same one verified `float32` tensor object/bytes is supplied to the PyTorch and ONNX paths.

The calibration producer is inspected but its full loader is never dispatched. On exactly three accepted anchors, the child exercises the pinned `YOLODataset.load_image` component with `rect_mode=True, resize_short=False`, then validation `LetterBox(new_shape=(640,640), auto=False, scale_fill=False, scaleup=False, center=True, padding_value=114)`, followed by the actual `Format._format_img`-equivalent RGB/CHW uint8 layout and the repository helper's explicit `float32 / 255` stream representation. The report stores these calibration stages separately from inference stages, including preliminary resize, ratio/padding, dtypes, ranges and hashes. It records `calibration_component_called=true`, `calibration_loader_called=false`, `dataset_instantiated=false`, `labels_read=false` and `image_cache_read=false`; source hashes are `inspected` evidence and are not promoted to producer execution claims. No full 3,072-image materialization, cache or calibration cache is created.

Native/export semantics are explicit in the plan and per-model report: accepted ONNX settings are `format=onnx, imgsz=640, batch=1, opset=17, simplify=true, dynamic=false, half=false, device=cpu, task=detect`, with the observed static float input/output schemas and `max_det=300` recorded. The native reference is `model.model.to(cpu).float().eval()` under `torch.no_grad`; it is not an in-memory exported/fused copy. Exporter, dataset and augmentation callable source hashes are recorded at runtime. Fusion, coordinate/packing and end-to-end behavior are described as accepted graph/reference semantics, not inferred from output agreement; no extra export-equivalent forward is introduced.

## Numeric comparison contract

Both runtimes are eval/no-grad CPU paths. The ONNX Runtime session requests exactly `CPUExecutionProvider`; the observed provider list, options and version are recorded. No GPU provider is selected and no automatic installation/network fallback is allowed. Required Torch, Ultralytics, NumPy and pycocotools versions must match the accepted config; ONNX Runtime version is recorded as an observed server prerequisite because it was not part of the historical readiness hash.

Native output selection is explicit:

- YOLOv8n uses the tuple primary tensor `[1,7,8400]` and reports separate raw box-channel `[0:4]` and score-channel `[4:7]` comparisons plus the overall verdict. Its head dictionary must contain `boxes`, `scores` and `feats`.
- YOLO26n uses the tuple primary tensor `[1,300,6]`, corresponding to the one2one end-to-end top-k output. Its head dictionary must contain the `one2one` branch. The one2many debug output is never compared.

The locked numeric rule is `rtol=1e-4`, `atol=1e-5` for finite `float32` boxes/scores/raw channels, evaluated in the order `np.isclose(reference, observed, rtol, atol)` and the equation `abs(observed-reference) <= atol + rtol*abs(reference)`. The report includes mismatch count, element count, max absolute/relative error, finite/infinite diagnostic counts, linear `min/p50/p95/p99/max` quantiles over finite relative values and up to ten offending locations. A zero reference with nonzero disagreement is a finite `fail` with JSON-safe infinite-relative diagnostics; it is not allowed to overflow the report. Shape mismatch, wrong dtype, wrong output branch or non-finite values are `unresolved`; finite numeric disagreement is `fail`; all required comparisons passing is `pass`.

For YOLO26n, coordinates `[0:4]` and score `[4:5]` columns are reported separately and compared at the original native row index. Class IDs `[5:6]` must be finite integer values in inclusive range `0..2` and must match exactly. Ties are counted and reported, but there is no sorting, rematching, tie breaking, clipping, NMS, confidence threshold, max-det change or policy tuning. Coordinates are not required to be inside the image because this is the pre-clipping native output stage.

## Parent/child lifecycle

The parent performs readiness, the complete two-model graph-v4 artifact validation, checkpoint, ONNX, fixture and no-overwrite checks without importing Torch or ONNX Runtime. A single-model CLI selects from that immutable full artifact; it never substitutes a model-specific graph artifact. It writes `numeric_plan.json`, then runs one child for each selected model sequentially with CPU-hidden/thread-limited environment variables. A child imports the pinned runtime only after the plan exists, checks model head identity, traces inference and bounded calibration components, runs eight source forwards and eight ONNX forwards, checks all input bytes again, and writes only JSON evidence. Per-image inference/calibration/comparison files are written under `models/<model>/partial/` as work progresses. A child failure writes `failure.json`; the parent consumes that existing record without overwriting it, creates a fallback only when no report/failure exists, and still writes a failed manifest and report with an accurate partial inventory. Execution status and numeric verdict are separate; the aggregate explicitly reports `pass`, `fail`, `unresolved` or `not_observed`.

The runner records `export_performed=false`, `build_performed=false`, `calibration_loader_called=false`, `calibration_component_called=true`, `gpu_used=false` and `scored_run_authorized=false` in plan, model and manifest evidence. It contains no TensorRT import or exporter dispatch and never calls `scripts/uniform_build_repeat.py`.

## Candidate server command — not authorized until Astra review

After Astra accepts this implementation, the operator may pull the reviewed commit, verify that the output root is absent and run the following single foreground command in the already provisioned environment:

Pull/check one line: `cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git rev-parse HEAD && test -d results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2 && test -d results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4 && test -f results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2/models/yolov8n/model.onnx && test -f results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2/models/yolo26n/model.onnx && if [ -e results/measurement_audit_v1/precision_head_confirmation_numeric_v1 ]; then echo OUTPUT_EXISTS; else echo OUTPUT_ABSENT; fi`

Run one line: `cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 YOLO_AUTOINSTALL=0 ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1 PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 local/g0_size_env/bin/python scripts/verify_precision_head_confirmation_numeric.py --readiness-root results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2 --graph-audit-root results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4 --source-root results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2 --model all --out-dir results/measurement_audit_v1/precision_head_confirmation_numeric_v1`

No GPU-idle or desktop confirmation is required for this CPU diagnostic. The command is still foreground and must not be wrapped with `nohup` by default.

## Publishable artifact procedure after an authorized run

Before staging, inspect the output tree. Push JSON/report/log evidence only:

- `numeric_plan.json`;
- `numeric_manifest.json`;
- `report.md`;
- `models/yolov8n/numeric_report.json` and `models/yolo26n/numeric_report.json`, or the corresponding `failure.json` when a child failed;
- any generated `models/<model>/partial/*.json` files, when present, so a late child failure remains auditable;
- `logs/yolov8n.log` and `logs/yolo26n.log`.

Do not push checkpoint, ONNX, raw tensor, cache, private binary or generated model files. Luna will pull the commit, check the exact artifact inventory, hashes, input bindings, provider/options, forward counts, preprocessing traces and all pass/fail/unresolved diagnostics, then stop at reviewer handoff. No follow-up study is launched automatically.
