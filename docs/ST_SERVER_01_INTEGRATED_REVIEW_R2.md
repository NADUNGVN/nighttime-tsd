# ST-SERVER-01 integrated review R2

Reviewed 2026-09-22 at 187126071ce2041925497f0289541b48c570ce50. Decision: NO-GO server matrix; continue the same supertask locally to one genuinely integrated packet. No changes to models, selections, endpoints, tolerances or budget. This is implementation repair, not a new scientific design.

## Independent checks

Astra ran `D:/Research/paper/local/measurement_audit_env/Scripts/python.exe -m unittest discover -s tests -p test_precision_head_confirmation_super.py -v`: 13 tests, **12 PASS, 1 ERROR, zero skips**. Error: `_cell_files` uses undefined `plan` at analyzer line 62. Both NumPy-dependent tests passed. Correct the reported 11 PASS/2 SKIP with the exact command/environment/commit and actual captured output; missing dependencies do not explain this NameError.

The canonical schedule and nested target-name extraction are repaired. The external runtime double does exercise 84 child launches and 78 synthetic files, but bypasses the production `build_real` path entirely. It cannot certify the interfaces below.

## R2.1 — executable producer/consumer contracts (blocking)

- `mapping_targets` extracts nested target names but still reads `mapping_hash` at the root. Both accepted real graph artifacts return None; their hashes are at `mapping.mapping_hash`. `validate_files` therefore rejects both models. Test the full real mapping/hash binding, not just nonempty lists.
- `validate_files` returns flat keys such as `yolov8n.checkpoint`; `build_plan` stores them as `models`. `build_real` and analyzer expect `models['yolov8n']['checkpoint']`. Output shape and postprocess fields are not copied into that evidence either. Establish one serialized schema and consume the actual producer result throughout tests.
- `run_child` does not create `child_state.json` before `build_real` calls `read(state_path)` at its first update. Initialize state before runtime import/side effects; preserve stage and error if imports fail.
- Scored FP16 calls `selected_targets(mapping, 'fp16')`, which deterministically raises `ContractError: unknown arm: fp16`. Explicitly represent FP16 no-target behavior without weakening INT8 arm validation.
- `_cell_files` uses undefined `plan`. Repair and run the existing regression; then test successful analyzer completion, not only invalid inputs.

## R2.2 — calibration, immutable inputs and identity (blocking)

`build_plan` records calibration recipe evidence, but the child reconstructs a calibration directory rather than consuming/binding the verified materialization. It neither audits ordered calibration image/tensor consumption nor publishes the shared calibration-cache hash. Passing 1024 batch calls does not prove the correct ordered 1024 tensors. Trace the actual locked Ultralytics loader order, preprocessing/dtype/shape and manifest; reject mismatch before the scored consumers. Scored consumers must verify immutable cache bytes and model/selection identity on every read, record read calls, zero supplied batches/writes, and before/after hashes. Preserve separate fresh timing-cache input/output hashes and requested/effective precision-inspector evidence in public reports.

Enforce accepted readiness/config/helper/graph/data/XML bindings, not unverified self-declared hash fields. Recheck actual ONNX/checkpoint inputs at the execution boundary and after use; parent-to-child plan/schedule consistency must be validated. Current child only copies plan checkpoint/ONNX hashes into reports. Current runtime versions and GPU name are recorded but not checked against the frozen contract/UUID/driver, and device-name lookup hardcodes device 0. Wrong runtime/GPU/cache/config must fail in CPU tests before a mocked build.

## R2.3 — actual capture route and numerical accounting (blocking)

The direct `CaptureValidator(model=engine_path)` path invokes framework backend loading and warmup. Local installed Ultralytics 8.4.102 validator source calls `model.warmup`; AutoBackend warmup performs an extra forward for an engine. The locked budget prohibits uncounted forwards. Provide an explicit tested route with true counted enqueues and no hidden warmup; do not report 1636 syncs by assigning a constant.

Audit raw engine packaging versus the actual TensorRT backend reader, metadata/names/shape/stride/end2end behavior and owner lifetime. Do not assume a serialized plan plus a recorded string `postprocess_route` establishes v26 semantics. Use saved synthetic raw outputs to exercise the real v8 decode/NMS and v26 end-to-end route, including detection counts, coordinates and class mapping; no real model forward needed locally. The present schema-2 prediction payload omits required same-pass capture fields (e.g. capture_mode/iou_thresholds) and cannot pass the existing `validate_capture` contract. Integrate native-statistics replay, matching and size verification explicitly, keep Ultralytics and COCO/XML metrics separate, validate the canonical 1636-image membership/2706 GT instances, not just record count.

## R2.4 — lifecycle, telemetry and budget evidence (blocking)

`owner_release_status='released'` is assigned while validator/builder/context objects can still be live. No production finally cleanup verifies all owners and preserves primary/cleanup errors. The real child must record observed attempts/completions, synchronization and release, including failures and timeout/unknown completion. Reuse the previously reviewed lifetime/timeout approach rather than regress it.

Telemetry is captured before build and after build+capture, not at each build/capture boundary; after-state is not checked. `--confirm-background-process` feeds an existing helper that allows confirmed competing compute, contrary to this confirmation study's locked build-variation rule (the earlier feasibility smoke had a different permission). Keep desktop exceptions; reject competing compute for this study. Preserve telemetry on failure, enforce device/runtime identity, and never alter other users' processes.

Bound termination of only owned child/process descendants and confirm exit before subsequent dispatch. Parent success must validate child counters/identity/schema, not merely exit code. Retain public inspector/cache provenance, failure inventory and publication allowlist, which the runbook currently promises but executor does not generate. No binaries in Git.

## R2.5 — completed analysis proof (blocking)

The switch to pooled COCO/XML AP and duplicate-preserving resampling is correct in direction. Test the **whole** analyzer success path with production-shaped synthetic capture artifacts and all model/FP16/selection/repeat combinations. Fix undefined `plan` and reject None/None provenance equality, duplicate/incorrect child jobs, nonzero exit codes, synthetic/test-double data, wrong engine/cache/model, count-only image substitutions and tampered schedule content whose declared hash was copied unchanged. Recompute hashes, bind XML to the expected dataset, verify points from captures and preserve the shared sample plan/hash.

The summary stringifies tuple keys in `_json_safe`; use an explicit documented output schema rather than opaque Python tuple strings. Produce all-cell points, full/XS/S control and FP16 contrasts, CI valid/undefined counts, within-build and between-selection SD/ranges, point reproduction and manuscript tables. Test full JSON serialization, publication and input invariance. An isolated pooled-AP fixture is insufficient proof of this integrated analyzer.

## R2.6 — real CLI/runbook and regression strategy

The runbook describes splitting two hosts but the implementation has no model-block selector and always requires all 84 jobs. Implement complete-model partition plus merged audit with globally unique canonical cell identities, or explicitly leave two-host execution unavailable and remove the claim; preserve the locked model-local GPU/runtime comparison. Do not silently change the scientific design.

Replace the fabricated runtime-double completion shortcut as the principal integration evidence. Inject doubles at external TensorRT/CUDA/filesystem/runtime boundaries while executing **the real producer, child state machine, capture dispatch and analyzer code**. Cover auxiliary, each INT8 arm, FP16, both output shapes, cache-only reuse, owner failure and timeout; then all 84/78 jobs and successful CPU analysis from the same generated artifacts. No actual CUDA import, TensorRT build or frozen forward is required. Keep the existing lightweight schedule test, labeled appropriately.

## Single resubmission gate

Work autonomously through all six groups; no new Astra message per bug. Use L2A-053 remediation milestones and attach a closure table mapping each item to code, actual test command/output and remaining limitations. Run in an existing local environment with the statistical dependencies available (Astra used the path above); do not modify runtime environments on a server. No skipped core analyzer/integration gate and no silent test failures. If unable to implement a contract, report the precise limitation in the consolidated packet rather than claim completion.

Then submit one code/protocol/test/runbook packet for the original integrated gate. NO server execution, export, GPU, new calibration/build/inference until GO. Preserve unrelated dirty checkpoint and previous evidence. Commit/push Astra inbox/review unchanged with scoped work via NADUNGVN; Astra does not push.
