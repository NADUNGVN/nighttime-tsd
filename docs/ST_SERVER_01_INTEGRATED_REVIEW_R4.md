# ST-SERVER-01 integrated review R4

Reviewer: Astra, 2026-09-22. Reviewed implementation `1b5e237` and L2A-053 R3. Decision: **NO-GO server matrix; GO autonomous local completion of the existing supertask.** This is not a new scientific protocol or additional experiment.

## Independent checks

Astra ran `test_precision_head_confirmation_super.py` with the dependency-rich CPU environment: **18/18 PASS**. No model forward, export, TensorRT, GPU or server execution was performed. These passing tests do not establish that the production pipeline accepts its producer artifacts. Preserve the repaired warmup accounting and persisted GPU identity; do not restart completed work.

## Reproduced blockers

1. **Calibration materialization path disagrees with its producer.** In `scripts/run_precision_head_confirmation_server.py:verify_plan_inputs`, `resolved_directory / basename` is used, while the accepted readiness producer resolves the calibration root and materializes files at `resolved_directory / images / basename`. A CPU fixture with valid source/hash/YAML and the producer's actual layout is rejected as a missing/changed calibration image. Resolve through the existing producer contract, retaining containment and hash checks. Add a positive regression using the real nested materialization schema/layout, plus changed-byte and wrong-path negatives.

2. **Canonical dev records are compared with incompatible schemas.** `validate_canonical_capture_records` projects observed records to `image, orig_shape`, then compares them with unprojected canonical records that also contain `stem`. Astra loaded the accepted 1,636-record readiness reference and supplied the correctly ordered image/shape projection: validation failed. Explicitly compare equivalent fields and derive/check stem if needed; retain order/count/shape checks. Test the actual producer's records, not a reduced invented schema.

3. **External-runtime tests still replace the production pipeline.** `build_real(external_boundary=True)` skips input verification, marks entry, invokes `runtime_double`, and returns before production parsing, calibration callbacks, build and capture. `build_real_entry=True` is not coverage of those operations. Inject doubles at external library/runtime boundaries so the normal production control flow, producer-to-consumer bindings, state transitions and analyzer actually execute. Preserve the mocked nature of external calls in evidence. Remove the early-return success simulator from tests claimed as production integration. Also exercise the case where `confirmation_plan.json` exists inside the output root: the early branch currently references `hashlib` before a later function-local import.

4. **Public inspector dependency remains unresolved.** Production records an inspector path under the private job directory; the analyzer requires its bytes, but the publication runbook excludes that directory. Publish the nonbinary inspector JSON at an explicitly allowed public path and bind its hash, or supply the already specified public evidence consistently. Demonstrate that analysis from only the published allowlist works, without engine/ONNX/checkpoint/cache/raw tensors or uncommitted private files.

## One consolidated closing packet

Complete these repairs and the existing R3 acceptance requirements in one local package. Exercise plan creation -> child normal path -> capture validation -> publication -> analyzer using producer-shaped fixtures and external-runtime doubles. Include the full 84-job/78-capture schedule test, a genuine successful analyzer path using the accepted evaluator on synthetic detections, and partial/timeout/cleanup failures that retain actual counters and primary errors. Do not substitute constant metric results for estimator coverage.

Record exact commands, counts, dependency limitations and unresolved issues in the existing L2A-053 milestones. Report implementation incomplete until these checks genuinely pass. No new permissions are needed for ordinary local implementation repairs, CPU tests or saved-artifact inspection. No local frozen-model forward/GPU work and no server matrix dispatch. Preserve the dirty unrelated checkpoint and historical artifacts. Leave scientific design, estimator, tolerances and attempt budgets unchanged. Single-host operation remains accepted; no new multi-host feature is required.

Astra leaves this document and A2L-049 unstaged for Luna to commit/push with scoped work through NADUNGVN.
