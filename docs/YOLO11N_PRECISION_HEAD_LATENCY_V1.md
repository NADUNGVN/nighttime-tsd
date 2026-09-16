# YOLO11n precision-head latency v1

Status: `implementation_accepted_operator_run_authorized`. Astra accepted implementation commit `2d8f5af84efda4be273cc7f9eca9becbdf202bf1` under A2L-020 in `docs/ASTRA_TO_LUNA.md`. The operator may run the existing-engine study after current server preflight passes; no server execution or latency result is claimed by this status.

## Scope

Study ID: `yolo11n_precision_head_latency_v1`.

Destination: `results/measurement_audit_v1/server_yolo11n_precision_head_latency_v1/`.

This is a diagnostic accuracy–latency study on one accepted RTX8000 environment. It uses the 12 existing YOLO11n precision-head attempt2 engines plus the one existing FP16 reference engine. The runner never exports, builds, rebuilds, or directly deserializes a new engine. It passes the existing engine path to `ultralytics.YOLO`, after checking the direct binary SHA256 against the captured manifest.

See A2L-020 for read-only server checks, the foreground base command, current desktop-confirmation handling, completion criteria and artifact handoff. Luna supplies the complete command using the operator's current snapshot, not historical PIDs. Missing or changed binaries are reported rather than rebuilt; partial output is preserved.

## Locked measurement contract

- Existing binaries only: FP16 reference plus `baseline_int8`, `bbox_fp32`, `classification_fp32`, and `both_fp32`, each with build repeats 1–3. Direct bytes are hashed before any child loads an engine; missing/changed/duplicate engine bindings fail closed.
- Environment/GPU: captured attempt2 contract (`torch 2.5.1+cu121`, Ultralytics `8.4.102`, TensorRT `10.16.1.11`, NumPy `2.4.4`, CUDA `12.1`, `pycocotools 2.0.10`, Quadro RTX 8000) and the exact UUID/name/driver identity from the accepted attempt2 snapshot. The parent does not run a CUDA-touching environment probe; every child must record and validate it.
- Runtime: synchronous batch-1 `model.predict` wall-clock from a decoded CPU image through preprocessing/H2D/inference/postprocessing/NMS; `imgsz=640`, `conf=0.001`, `iou=0.7`, `max_det=300`, `rect=False`, `task=detect`, `verbose=False`. This is not pure TensorRT kernel time and is not the old benchmark script's default contract.
- Exclusions: disk decode, model load, initial allocation and warmup. GPU synchronization occurs before each timer and after each predict; timer is `time.perf_counter_ns` converted to milliseconds.
- Images: sorted 1,636 dev IDs from the locked FP16 capture; PCG64 seed `20260916`, sample 256 IDs without replacement, then sort selected IDs. The default directory is repo-root-relative `data/processed/cctsdb2021_clean/dev/images`; an explicit relocation is resolved from the repository when relative and is persisted as declared/resolved path. The runner records selected file hashes, compares decoded `(H,W)` against the accepted capture `orig_shape`, and records direct filesystem bytes actually used. This is not a claim that new image hashes are content-identical to the historical capture when historical image-byte hashes were not stored. Every session uses the same 256 image bytes; no annotations or official-test images are read.
- Per session: one non-timed preprocessing probe per distinct selected source aspect ratio, then 200 warmup calls followed by exactly 1,000 measured calls, cyclic pool order beginning at index 0. The probe must observe `(1,3,640,640)` and is persisted outside the timer. Persist all 1,000 raw samples plus mean/median/p95/p99/min/max and serial FPS (`1000/mean_ms`); p95/p99 use NumPy linear quantiles.
- Schedule: 3 rounds × 13 engines, 39 sessions. Round 1 uses `FP16, baseline1, bbox1, classification1, both1, baseline2, ... both3`; round 2 rotates this list left by 4; round 3 rotates it left by 8. One fresh child process per session, parent orchestration, no concurrent engine sessions. The rotation reduces position confounding but is not claimed to be a complete Latin square.
- Summaries: per engine/per round, per engine pooled 3,000 calls, and arm summaries containing all builds and rounds. Call samples are not treated as independent build replicates; no fastest build or arm is selected.
- Accuracy links: `precision_head_paired_analysis_v1/point_estimates.json` and `contrast_ci.json` are read from and pinned to accepted analysis commit `4846c73ddd2cbb2bd522caa0e1a1eb4598deb1e3`; attempt2 metadata and all consumed JSON are read from accepted commit `839acdcb6a09569d1e6e130aa523c38d960dabd5`. The manifest records each canonical path/commit/blob SHA256. Full Ultralytics points remain separate; CIs are the ten fixed COCO/XML contrast CIs and are not attached to latency points.
- Telemetry: parent and every child record GPU/process snapshots before/after. The existing desktop exception is allowed only through current `PID=PATH` confirmation. Any unconfirmed CUDA workload, new competing workload, identity mismatch or incomplete telemetry stops the run; no kill/pause/permissions/clock/power changes are permitted. Shared-server observations do not establish GPU isolation.

## Output contract

The runner writes a new `study_manifest.json`, `image_pool_manifest.json`, 39 session directories with `session.json`, `execution_manifest.json` and child logs, then `latency_summary.json` and `report.md`. The child validates its scheduled output directory and parent pool hash before GPU work. Each session persists independently checked before/after GPU identity and process-guard evidence; malformed telemetry, changed workload, missing/duplicate session, wrong raw-sample cardinality or wrong arm call count fails closed. It refuses an existing destination and preserves partial output on failure. The final state is `latency_completed_review_required`; no deployment choice or next study is started automatically.

## Local checks required before review

Run CPU/mock tests only. Tests must not import TensorRT, load an engine or call GPU. A passing local test is not a server benchmark result and must not be described as TensorRT end-to-end validation.

## A2L-018 corrections

- R1: image-path resolution is anchored to the repository rather than shell cwd. The selected files are checked for direct SHA256 and decoded dimensions against the accepted FP16 `orig_shape`; relocation metadata is persisted and does not claim historical byte identity.
- R2: accepted attempt2 JSON is read from canonical commit `839acdcb6a09569d1e6e130aa523c38d960dabd5`, and paired-analysis JSON from canonical commit `4846c73ddd2cbb2bd522caa0e1a1eb4598deb1e3`. Prediction/report/verification links, model bindings, nested and flat cache input hashes, point/contrast schema and canonical blob hashes are checked before use. Server engine bytes remain direct filesystem checks before deserialization.
- R3: a session must contain exactly 1,000 positive finite raw samples, independent before/after GPU identity and process-guard evidence, parent pool path/hash binding, scheduled output location, and an observed `(1,3,640,640)` preprocessing shape for every sampled aspect-ratio group. Parent waits for every child and emits no completed summary unless all 39 unique sessions, 3 rounds per engine, 3,000 pooled calls per build and 9,000 pooled calls per INT8 arm validate. The CUDA-touching environment probe is deferred to children, so parent orchestration itself does not run it.

## A2L-019 compatibility corrections

- The accepted FP16 `dev_absolute.yaml` is read from the pinned Git blob and parsed with `yaml.safe_load`. The runner validates the POSIX absolute `path` suffix `data/processed/cctsdb2021_clean/dev`, `train: images`, `val: images`, the three locked class names and `nc: 3`; it does not identify the dataset through an uppercase substring. Historical Linux paths remain valid when the runner is inspected on Windows. Repository-anchored default image resolution and explicit relocation binding are unchanged.
- Persisted GPU snapshots are validated against the actual `uniform_build_repeat.snapshot` producer schema and version. The consumer reads `process_guard.external_workload_detected`, requires complete telemetry and the producer's required guard fields, and rejects missing/unknown/true workload states, blocked or unmatched processes, and authorized background compute outside this latency contract. The producer field is not renamed and no missing field defaults to clean.
- These are schema-compatibility fixes only. The 13 existing engines, 39 serial sessions, runtime options, image pool, warmup/timed-call boundary, shape probe, telemetry policy and aggregation design are unchanged.
