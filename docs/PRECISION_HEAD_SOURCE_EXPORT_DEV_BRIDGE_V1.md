# Precision-head source/export dev bridge v1

Status: local implementation and CPU/mock tests only. Server execution is `NO-GO` until Astra reviews this numerical-work contract.

This bridge was added after the YOLOv8n/YOLO26n raw-output and localization diagnostics were accepted as bounded, strict `FAIL` evidence. It is a new, dated review gate; it does not rewrite, relax, or replace any earlier verdict.

## Question and scope

Measure the application-level difference between each frozen PyTorch FP32 checkpoint and its already-accepted ONNX FP32 graph on the canonical CCTSDB2021/dev split. The result is intended to separate source/native-versus-export drift from any later INT8 head-precision attribution.

The two independent model lifecycles are YOLOv8n and YOLO26n. The workload is the existing dev split only: 1,636 images and the accepted 2,706 XML instances. It excludes official positive/negative test images, training, calibration, TensorRT, new exports, derived graphs, repeats, bootstrap, model sweeps and the scored 78-capture matrix.

The candidate workload is 6,544 ordinary calls:

- each model: 1,636 frozen native CPU FP32 forwards and 1,636 ORT CPU session runs;
- one ORT session construction per model; no export/build calls;
- models run sequentially in independent child processes so one failed model cannot silently contaminate the other.

The runner streams bounded JSONL summaries. It retains no unbounded native/ONNX output tensors and publishes no raw tensor dump.

## Locked input and producer contracts

The parent binds the accepted readiness artifact, graph-audit-v4 artifact, configuration and canonical dev image order from the accepted FP16 capture metadata. It checks the current dev image bytes, exact image membership/order, label-stem membership, image dimensions during child preprocessing, XML membership/dimensions/instance count, checkpoint hashes and accepted ONNX hashes. The XML archive is supplied explicitly by the operator because it is a raw dataset input and is not assumed to be in Git.

The frozen model and ONNX bindings are:

| Model | Frozen checkpoint | Accepted ONNX output | Expected checkpoint SHA256 | Expected ONNX SHA256 |
|---|---|---|---|---|
| YOLOv8n | `results/yolov8n_cctsdb_clean_s42_v1/weights/best.pt` | `[1,7,8400]` | `b2b7a1c77a19499ded33c9cc11c621757077aa871f4e7f7a1fcdbf94f53b383b` | `e22d53bbeb333f44783535d911d5e318ebb7d500e8fcb3d1cbb1284f5248d603` |
| YOLO26n | `results/yolo26n_cctsdb_clean_s42_v1/weights/best.pt` | `[1,300,6]` | `2bb49f85f581469fc7942652d5fda4da44278d57fa8363e8f7295daa49f0d01e` | `1b2467ccd62bd1e53f3bde3e3f22e1b42129711d3e368a4b4666d025099ce5cc` |

Preprocessing is the installed Ultralytics CPU path: decoded BGR, `BasePredictor.pre_transform/preprocess`, fixed 640×640 centered LetterBox, RGB CHW, contiguous float32 and division by 255. The same tensor and input hash are bound to both producers for each image. The locked settings are `rect=false`, `conf=0.001`, `iou=0.7`, `max_det=300`, `workers=0`, batch 1 and no image augmentation.

Application postprocessing is intentionally model-specific and shared by native/ONNX outputs:

- YOLOv8n uses `ultralytics.utils.nms.non_max_suppression` with `end2end=False`, class-aware (`agnostic=False`) NMS, `nc=3`, the locked confidence/IoU/max-detection values, then installed `scale_boxes` to original-image `xyxy`.
- YOLO26n uses the installed `non_max_suppression` end-to-end filtering branch with `end2end=True`; it does not run a second NMS. The returned decoded `xyxy`, confidence and class ID rows are then scaled with installed `scale_boxes`.

The source method hashes for preprocessing, NMS/end-to-end filtering and coordinate scaling are recorded at runtime. Native source semantics remain separate from exporter semantics; no native fusion or post-hoc sorting is added to make outputs agree.

## Records and metrics

Each model child writes separate `native_records.jsonl` and `onnx_records.jsonl`, plus one `preprocess_trace.jsonl`. A record binds image order/name, original dimensions, image bytes, tensor bytes, output shape/dtype/finite check/hash and application detections in original-image coordinates. No raw model tensor is written.

The child reuses `scripts/verify_cctsdb_capture.py::coco_size` and its `COCO_bbox_AP_custom_CCTSDB_area_XML_original_coordinates_v1` estimator. It reports all, XS, S, M, L and XL AP50/AP50–95 for native and ONNX, prediction counts and the signed delta defined as `ONNX minus native`. Bounded examples only report ordered-output equality/count/class-prefix differences. No one-to-one matching is used in the AP estimator or introduced as a hidden reconciliation step.

## Validity versus assessment

Hard validity failures block interpretation: wrong identity/hash, wrong split or XML membership, wrong dimensions, non-CPU ORT provider, non-finite output, invalid application detections, incomplete/order-mismatched JSONL, changed input binaries or changed dev image bytes. A failed child preserves its model-scoped partial records and failure state; the parent does not resume or overwrite it.

If all validity checks pass, the result is descriptive measured source/export drift on this dev split. No application-equivalence PASS threshold is proposed here and no threshold may be selected from the observed result. Astra/reviewer decides whether a future margin is scientifically useful before any precision matrix. The historical numeric/localization `FAIL` remains visible and unchanged.

## Output and operator boundary

The proposed output is:

`results/measurement_audit_v1/precision_head_source_export_dev_bridge_v1`

Publishable artifacts are the bridge plan/manifest/report, model reports or model-scoped failure records, model native/ONNX/preprocess JSONL and child logs. Private tensor/derived-graph directories are not part of this runner and are not to be pushed. The output root is absence-protected; a partial root is preserved for review, not resumed.

The operator must provide the actual raw XML archive path after checking its location. The following are candidate commands for the server after Astra reviews the implementation and after the exact commit is substituted in the read-only check:

```text
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master && test "$(env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git rev-parse HEAD)" = "<L2A-046-COMMIT>" && test ! -e results/measurement_audit_v1/precision_head_source_export_dev_bridge_v1 && test -f results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2/readiness_manifest.json && test -f results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4/graph_audit_manifest.json && test -f results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2/models/yolov8n/model.onnx && test -f results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2/models/yolo26n/model.onnx && echo READY
```

```text
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 YOLO_AUTOINSTALL=0 ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1 PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 local/g0_size_env/bin/python scripts/run_precision_head_source_export_dev_bridge.py --xml /ABSOLUTE/SERVER/PATH/TO/CCTSDB_RAW_XML_ARCHIVE.zip --model all --out-dir results/measurement_audit_v1/precision_head_source_export_dev_bridge_v1
```

The foreground command is one run, not a benchmark and not a TensorRT validation. It does not require an idle GPU because it hides CUDA and uses CPU; it must still run only after the operator has confirmed the server environment and raw XML path. Expected resource is one bounded CPU child at a time and 6,544 ordinary calls; no local wall-time estimate is claimed.

## Local verification boundary

Local verification covers call accounting, output protection helpers, canonical order/label membership guards, real installed Ultralytics NMS/end-to-end filtering on synthetic tensors (ties, strict threshold, overlap, class-aware behavior), coordinate scaling, empty/non-finite outputs, JSONL streaming and signed metric-delta direction. It does not load the frozen checkpoints, run full dev, invoke ONNX Runtime on the frozen graphs, export, build TensorRT or touch GPU.
