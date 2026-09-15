# YOLO11n precision-head latency v1

Status: `implementation-review-required`. A2L-017 authorizes local implementation/tests and protocol only. Astra must review this runner before any server/GPU run is authorized.

## Scope

Study ID: `yolo11n_precision_head_latency_v1`.

Destination: `results/measurement_audit_v1/server_yolo11n_precision_head_latency_v1/`.

This is a diagnostic accuracy–latency study on one accepted RTX8000 environment. It uses the 12 existing YOLO11n precision-head attempt2 engines plus the one existing FP16 reference engine. The runner never exports, builds, rebuilds, or directly deserializes a new engine. It passes the existing engine path to `ultralytics.YOLO`, after checking the direct binary SHA256 against the captured manifest.

No server command is issued by this implementation handoff. A server command may be supplied only after Astra reviews and authorizes this code.

## Locked measurement contract

- Existing binaries only: FP16 reference plus `baseline_int8`, `bbox_fp32`, `classification_fp32`, and `both_fp32`, each with build repeats 1–3. Direct bytes are hashed before any child loads an engine; missing/changed/duplicate engine bindings fail closed.
- Environment/GPU: captured attempt2 contract (`torch 2.5.1+cu121`, Ultralytics `8.4.102`, TensorRT `10.16.1.11`, NumPy `2.4.4`, CUDA `12.1`, `pycocotools 2.0.10`, Quadro RTX 8000) and the exact UUID/name/driver identity from the accepted attempt2 snapshot.
- Runtime: synchronous batch-1 `model.predict` wall-clock from a decoded CPU image through preprocessing/H2D/inference/postprocessing/NMS; `imgsz=640`, `conf=0.001`, `iou=0.7`, `max_det=300`, `rect=False`, `task=detect`, `verbose=False`. This is not pure TensorRT kernel time and is not the old benchmark script's default contract.
- Exclusions: disk decode, model load, initial allocation and warmup. GPU synchronization occurs before each timer and after each predict; timer is `time.perf_counter_ns` converted to milliseconds.
- Images: sorted 1,636 dev IDs from the locked FP16 capture; PCG64 seed `20260916`, sample 256 IDs without replacement, then sort selected IDs. The runner records selected file hashes, decoded shapes and the exact cyclic input sequence. Every session uses the same 256 image bytes; no annotations or official-test images are read.
- Per session: 200 warmup calls followed by 1,000 measured calls, cyclic pool order beginning at index 0. Persist all 1,000 raw samples plus mean/median/p95/p99/min/max and serial FPS (`1000/mean_ms`); p95/p99 use NumPy linear quantiles.
- Schedule: 3 rounds × 13 engines, 39 sessions. Round 1 uses `FP16, baseline1, bbox1, classification1, both1, baseline2, ... both3`; round 2 rotates this list left by 4; round 3 rotates it left by 8. One fresh child process per session, parent orchestration, no concurrent engine sessions. The rotation reduces position confounding but is not claimed to be a complete Latin square.
- Summaries: per engine/per round, per engine pooled 3,000 calls, and arm summaries containing all builds and rounds. Call samples are not treated as independent build replicates; no fastest build or arm is selected.
- Accuracy links: `precision_head_paired_analysis_v1/point_estimates.json` and `contrast_ci.json` are linked by model/evaluator and hash. Full Ultralytics points remain separate; CIs are the ten fixed COCO/XML contrast CIs and are not attached to latency points.
- Telemetry: parent and every child record GPU/process snapshots before/after. The existing desktop exception is allowed only through current `PID=PATH` confirmation. Any unconfirmed CUDA workload, new competing workload, identity mismatch or incomplete telemetry stops the run; no kill/pause/permissions/clock/power changes are permitted. Shared-server observations do not establish GPU isolation.

## Output contract

The runner writes a new `study_manifest.json`, `image_pool_manifest.json`, 39 session directories with `session.json`, `execution_manifest.json` and child logs, then `latency_summary.json` and `report.md`. It refuses an existing destination and preserves partial output on failure. The final state is `latency_completed_review_required`; no deployment choice or next study is started automatically.

## Local checks required before review

Run CPU/mock tests only. Tests must not import TensorRT, load an engine or call GPU. A passing local test is not a server benchmark result and must not be described as TensorRT end-to-end validation.
