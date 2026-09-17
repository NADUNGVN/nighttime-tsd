# Precision-head CPU numeric confirmation v1

Status: implementation package submitted for Astra review. This document defines a bounded CPU diagnostic; it does not authorize server execution until the reviewer accepts this package.

## Scope and boundary

The runner compares the frozen YOLOv8n and YOLO26n PyTorch FP32 native primary output with the already accepted ONNX graph on CPU. It uses eight fixed train images and three preprocessing trace anchors. It does not train, export, rebuild, calibrate, score AP, read official-test/negative-test images or labels, run TensorRT, use a GPU, apply NMS, or create a calibration cache.

The existing \`84\`-builder/\`78\`-capture schedule is unchanged. A numeric pass is only evidence that this small input fixture follows the same verified tensor path and that the selected native output agrees with the existing ONNX output. It is not dataset-wide accuracy, TensorRT compatibility, calibration equivalence or engine reproducibility.

## Fixed inputs

The parent binds, before dispatching children:

- accepted readiness root \`results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2\`;
- accepted graph-v4 five-file artifact at commit \`6780b813c5f1cb2d915b72832eedd79525eaecbd\`;
- frozen checkpoints and exact hashes already accepted by readiness;
- existing ONNX files from \`precision_head_confirmation_graph_prep_v2\`, with expected hashes \`e22d53bbeb333f44783535d911d5e318ebb7d500e8fcb3d1cbb1284f5248d603\` (YOLOv8n) and \`1b2467ccd62bd1e53f3bde3e3f22e1b42129711d3e368a4b4666d025099ce5cc\` (YOLO26n).

The forward fixture is the first eight distinct train image IDs in canonical U42 manifest order:

\`00006, 00009, 00028, 00036, 00054, 00061, 00098, 00104\`.

The runner verifies the current source and materialized image bytes against the accepted readiness rows before any forward. It records the IDs, manifest positions, original source paths, byte hashes and sizes. The first image of each U42/U43/U44 selection is also traced. If a trace ID overlaps the forward fixture, its tensor trace is reused while its selection binding remains separately recorded. Labels are not read.

The output root is exclusive and no-overwrite: \`results/measurement_audit_v1/precision_head_confirmation_numeric_v1\`. Existing output means stop; partial output is preserved and is not resumed silently.

## Producer and preprocessing contract

The child uses the pinned Ultralytics 8.4.102 inference producer path and records source hashes for the relevant functions. The observed list-input path is:

1. Ultralytics \`imread\`, producing a three-channel BGR array;
2. \`BasePredictor.pre_transform\` with the model's observed stride, \`new_shape=(640,640)\`, \`auto=False\`, \`scale_fill=False\`, \`scaleup=True\`, \`center=True\`, padding value \`114\` and \`cv2.INTER_LINEAR\`;
3. BGR to RGB;
4. BHWC to BCHW and contiguous layout;
5. CPU \`float32\`, divide by \`255\`, and batch dimension.

The exact decoded shape, resize ratio, rounded resized shape, each padding side, stage dtype/shape/range and tensor byte hash are recorded per trace. The same one verified \`float32\` tensor object/bytes is supplied to the PyTorch and ONNX paths.

The calibration producer is inspected but never dispatched. The report records source evidence for \`Exporter.get_int8_calibration_dataloader\`, \`YOLODataset.load_image\`, \`build_yolo_dataset\`, the augmentation module and the repository helper. This separates the dev/inference trace from the calibration loader and does not silently claim that they are equivalent. No full 3,072-image materialization is performed.

## Numeric comparison contract

Both runtimes are eval/no-grad CPU paths. The ONNX Runtime session requests exactly \`CPUExecutionProvider\`; the observed provider list, options and version are recorded. No GPU provider is selected and no automatic installation/network fallback is allowed. Required Torch, Ultralytics, NumPy and pycocotools versions must match the accepted config; ONNX Runtime version is recorded as an observed server prerequisite because it was not part of the historical readiness hash.

Native output selection is explicit:

- YOLOv8n uses the tuple primary tensor \`[1,7,8400]\` and retains the raw channel-wise comparison. Its head dictionary must contain \`boxes\`, \`scores\` and \`feats\`.
- YOLO26n uses the tuple primary tensor \`[1,300,6]\`, corresponding to the one2one end-to-end top-k output. Its head dictionary must contain the \`one2one\` branch. The one2many debug output is never compared.

The locked numeric rule is \`rtol=1e-4\`, \`atol=1e-5\` for finite \`float32\` boxes/scores/raw channels. The report includes mismatch count, element count, max absolute/relative error, linear \`min/p50/p95/p99/max\` quantiles and up to ten offending locations. Shape mismatch, wrong dtype, wrong output branch or non-finite values are \`unresolved\`; finite numeric disagreement is \`fail\`; all required comparisons passing is \`pass\`.

For YOLO26n, coordinates and score columns are compared at the original native row index. Class IDs must be finite integer values in inclusive range \`0..2\` and must match exactly. Ties are counted and reported, but there is no sorting, rematching, tie breaking, clipping, NMS, confidence threshold, max-det change or policy tuning. Coordinates are not required to be inside the image because this is the pre-clipping native output stage.

## Parent/child lifecycle

The parent performs readiness, graph-v4, checkpoint, ONNX, fixture and no-overwrite checks without importing Torch or ONNX Runtime. It writes \`numeric_plan.json\`, then runs one child for each selected model sequentially with CPU-hidden/thread-limited environment variables. A child imports the pinned runtime only after the plan exists, checks model head identity, traces preprocessing, runs eight source forwards and eight ONNX forwards, checks all input bytes again, and writes only JSON evidence. A child failure writes \`failure.json\`; the parent still writes a failed manifest and report with the partial inventory.

The runner records \`export_performed=false\`, \`build_performed=false\`, \`calibration_loader_called=false\`, \`gpu_used=false\` and \`scored_run_authorized=false\` in plan, model and manifest evidence. It contains no TensorRT import or exporter dispatch and never calls \`scripts/uniform_build_repeat.py\`.

## Candidate server command — not authorized until Astra review

After Astra accepts this implementation, the operator may pull the reviewed commit, verify that the output root is absent and run the following single foreground command in the already provisioned environment:

Pull/check one line: \`cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git rev-parse HEAD && test -d results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2 && test -d results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4 && test -f results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2/models/yolov8n/model.onnx && test -f results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2/models/yolo26n/model.onnx && if [ -e results/measurement_audit_v1/precision_head_confirmation_numeric_v1 ]; then echo OUTPUT_EXISTS; else echo OUTPUT_ABSENT; fi\`

Run one line: \`cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 YOLO_AUTOINSTALL=0 ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1 PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 local/g0_size_env/bin/python scripts/verify_precision_head_confirmation_numeric.py --readiness-root results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2 --graph-audit-root results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4 --source-root results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2 --model all --out-dir results/measurement_audit_v1/precision_head_confirmation_numeric_v1\`

No GPU-idle or desktop confirmation is required for this CPU diagnostic. The command is still foreground and must not be wrapped with \`nohup\` by default.

## Publishable artifact procedure after an authorized run

Before staging, inspect the output tree. Push JSON/report/log evidence only:

- \`numeric_plan.json\`;
- \`numeric_manifest.json\`;
- \`report.md\`;
- \`models/yolov8n/numeric_report.json\` and \`models/yolo26n/numeric_report.json\`, or the corresponding \`failure.json\` when a child failed;
- \`logs/yolov8n.log\` and \`logs/yolo26n.log\`.

Do not push checkpoint, ONNX, raw tensor, cache, private binary or generated model files. Luna will pull the commit, check the exact artifact inventory, hashes, input bindings, provider/options, forward counts, preprocessing traces and all pass/fail/unresolved diagnostics, then stop at reviewer handoff. No follow-up study is launched automatically.
