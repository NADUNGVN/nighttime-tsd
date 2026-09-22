# ST-SERVER-01 R4 closure review

2026-09-22, Astra. Reviewed `786556a3b9eec34fe6e0f49f4396ae33b0e9926c`.

## Accepted and independently checked

- Super suite: **20/20 PASS**, 74.764 seconds, using `D:/Research/paper/local/measurement_audit_env/Scripts/python.exe`.
- TRT feasibility regression: **30/30 PASS**.
- The calibration resolver now uses the producer's `resolved_directory/images` layout. Producer-shaped positive/negative tests are present.
- Dev records are compared using equivalent image/stem/original-shape projections, with the actual accepted readiness records tested.
- Production inspector JSON is copied to `public/inspectors/<job>.json` and linked by hash. Preserve this correction.
- The local hashlib shadowing issue is removed. Preserve all previously accepted fixes and the unrelated modified checkpoint.

No frozen-model forward, GPU, TensorRT, export, server execution, or matrix dispatch occurred in this review.

## Remaining R4 requirement: test the production operations, not added phase labels

The claim that the external-boundary test now covers the actual production phases is still too broad. In `build_real`, `if external_boundary` invokes `runtime_double(...)` and **returns before** the production parser, calibration class/callbacks, builder configuration, precision assignments, capture backend and cleanup. The double adds `external_build_complete` and `external_capture_complete` labels and sets counts directly. Those labels do not execute the corresponding implementation.

The full-schedule fixture's `runtime_plan` supplies synthetic checkpoint/ONNX identities but no actual model mapping/output shape, runtime contract, calibration producer or canonical dev records. That fixture therefore cannot catch disagreement among those real consumers. It remains a useful parent/schedule test and should be retained under that description.

This review does **not** claim that the corrected production code has been observed to fail on TensorRT. The remaining blocker is specifically the missing integration evidence requested in R3/R4, given the repeated producer/consumer failures previously missed by this simulator. It is not a demand for local TensorRT or extra server experiments.

## Bounded completion instructions

Add a production-path integration test group. Run the normal `build_real` branch (`external_boundary=False`), or a refactoring with exactly one shared implementation path. Inject/patch external TensorRT, CUDA, exporter/loader, GPU telemetry and model-runtime calls; do not call real device libraries or frozen models. Make `runtime_double` raise if this group invokes it. Do not replace the orchestrator, calibration callback class, phase transitions, precision-target logic or completion validation with a successful whole-cell substitute.

At minimum cover the three actual phases (auxiliary calibration, scored cache-only INT8, scored FP16) and both frozen model/head/output contracts. Construct inputs from the accepted producer schemas. Have the fake builder invoke the actual calibration callbacks: auxiliary ordered batches/cache write, then scored read-only reuse; assert zero scored batches/writes. Check fresh timing input, applied target names/FP16 flags, output shape, real capture wrapper's counts and no-warmup behavior, inspector publication and release. Use synthetic records and the actual CPU validators/evaluator where applicable. Reuse existing tests rather than inventing another estimator.

Include representative parser/build/capture/synchronization failures through this path; counters and primary failure must come from operations actually reached, with cleanup recorded separately. Add a sensitivity check: a failing fake parser/builder or changed output shape must make the production-path test fail at that operation, not merely still produce a completed manifest. Then run the existing full 84/78 parent test and real pooled-analysis test as complementary coverage. A full 84-job fake TensorRT replay is not required in addition to the bounded phase/head cases.

Update L2A-053 to distinguish parent simulator coverage from normal-path coverage, listing the exact tests and remaining limitations. Supply one closing packet with unchanged scientific design and executable runbook. Do not reopen accepted layout/schema/inspector work or add features. **GO local implementation/tests; NO-GO server matrix pending integrated review of this remaining requirement.** No local GPU or server commands. Astra leaves this review/inbox unstaged for Luna to commit and push with scoped changes via NADUNGVN.
