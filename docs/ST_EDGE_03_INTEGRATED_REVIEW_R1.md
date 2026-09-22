# ST-EDGE-03 integrated R1 — accept diagnostic evidence; complete the executable dev workflow

2026-09-22 review at eab65f5. **Saved-output diagnostic accepted with the reporting corrections below. Dev execution NO-GO; local implementation of the existing supertask GO.** Do not change engine, precision, tolerance, boxes, dataset or endpoints. No SSH/transfer/model forward/build/inference/benchmark/install is authorized by this review.

## Independent evidence

Astra ran focused tests: analyzer 7/7 and dev packet 7/7 PASS. Canonical hashes/sizes of all seven audit/packet files match L1A-024. Astra replayed all three published fixture analyses from the already-existing private bytes and compared JSON-normalized results exactly; retained log hashes/metadata also match. Existing five detection rows and corrected min/max IoU remain valid descriptive evidence. Raw source/target FAILs are not converted into PASS.

## E1 — helper binding must describe a real executed symbol

Both packet and audit name `ultralytics.utils.ops.non_max_suppression`. In the installed pinned Ultralytics 8.4.102 that symbol **does not exist**; `ultralytics.utils.nms.non_max_suppression` does. Astra checked both attributes on CPU. The analyzer only records a helper-binding dictionary; it still calls its custom Python postprocess. It does not execute the claimed accepted helper or store its code hash/backend outputs.

Keep custom tables as explicitly labeled historical diagnostics. Add an executable CPU comparison of saved outputs through the actual pinned helper, with source/function/package and selected NMS backend hashes, float32 input, cloned buffer, fixed conf/IoU/max_det/single-label and return_idxs. Attribute prior independent helper comparison to Astra's review, not to a helper call absent from this runner. Test exact confidence boundary and positive-area/tie cases. No frozen model forward is required.

The input before/after audit currently hashes twice consecutively **before** analyze_fixture. Move the after check after processing, include the consumed native/ONNX/target bytes, and state precisely whether checking input bytes or output payloads. Preserve v1 and E2L1-024 artifacts; publish an addendum/fresh root.

## E2 — the dev packet is a design/mock scaffold, not a completed execution implementation

`e2_dev_evaluation_packet.py` explicitly defers a real runtime adapter to a later runner; CLI only writes three packet files. `run_mock_study` processes three arbitrary IDs with arbitrary input bytes through MockRuntime. `mock_evaluator` computes hashes and returns `ap_available=False`. There is no actual dev source materializer/reference-reuse validator, existing-engine E2 adapter dispatch, scale-back/detection capture, COCO/XML AP/bootstrap analyzer or executed CLI command chain. The runbook is prose. Correct packet-ready/implementation-complete implications in L1A-024; the approved ST-EDGE-03 implementation deliverables are not complete.

Complete one narrow workflow using the existing E2 runtime/provider/owner components, with no new framework and no device execution locally. Separate:

1. CPU inventory/preparation and conditional reference reuse, with explicit frozen ONNX identity, canonical dev IDs/shapes/XML/bytes, letterbox/preprocess details and private package manifest. Actual reference forward remains disabled before GO. Reuse only proven identical references; otherwise provide a separately counted future CPU capture mode.
2. E2 execution CLI for the existing engine only, lazy runtime initialization behind validated identity/package/authorization boundaries; exactly 1636 target passes, no hidden warmup/probes/retry/build. Use reviewed owner/buffer/stream/completion lifecycle, bounded child timeout and durable evidence. It must be testable locally by replacing external runtime calls, without executing a real engine.
3. Local CPU capture conversion and real COCO/XML/paired-bootstrap analysis with fixed original-coordinate geometry and supported helper versions. Prefer target raw-output capture on E2 with postprocess/evaluation on an existing compatible CPU host if that avoids unsupported packages on E2; explicitly bind identical reference/target processing, no E2 install. This does not authorize any new model calls.
4. Scoped package/public artifact writer and operator runbook with real command lines and roles. The user executes any server CPU command; Luna1 does not gain SERVER-01 SSH authority. GPU/edge execution remains off until one integrated review.

## E3 — validation and lifecycle must enforce the frozen contract

Astra changed engine_sha256 to WRONG, runtime warmup to 999 and call_budget warmup_calls to 999; `validate_study_contract` still returned validated. Its reference count is None due to reading `reference_passes_if_reuse_not_verified` while the contract stores `reference_image_passes_if_reuse_not_verified`. Reject these mismatches and verify every immutable required field; validation cannot be a handful of constant checks on otherwise untrusted configuration.

`validate_package_inventory` compares claimed hash strings, not files. Add actual allowed-file bytes/hash/size/path/symlink/duplicate checks, all canonical image memberships, input shape/dtype/finiteness and byte budgets, consistent reference source identity and XML hash. Verify the existing engine on its owning E2 only in a future gated preflight; never download/load it locally. No substitution with a different ONNX or engine.

The mock timeout is merely a raised exception, with no enforceable wall-clock timeout; events exist only in RAM and are written after completion. A hang or killed process loses the evidence. Reuse bounded child-process timeout, durable per-attempt/per-completion events, unknown-completion state, ownership cleanup and partial publication. Distinguish enqueue/copy/sync attempted/completed; preserve primary and cleanup failures. Test abrupt exit/hang and output-root collision, not only a cooperative exception.

Resource plan must match actual transfer format: 1636 float32 input tensors alone require 8,041,267,200 bytes (about 7.49 GiB), before target/reference outputs and archives. A stated 2 GiB storage minimum is insufficient for a materialized tensor bundle. Choose a documented streaming/batching package or realistic disk bound with preflight; do not silently encode/resize inputs differently or regenerate on E2 without a bound preprocessing contract. Assess timeout/RAM from the chosen design; these are planning limits, not benchmark claims.

## One complete resubmission, no per-helper approval loop

Run the real internal CLI/parent/child state machine with external runtime doubles; exercise all 1636 IDs/count accounting without real inference. A small synthetic dataset may test numerical evaluator/bootstrap, but must use real postprocess/geometry/COCO and duplicate-image resampling, with known positive/false-positive examples and independent point checks. Mocking the evaluator to hashes or fixed AP does not satisfy artifact-to-analysis. Production mode must reject mock/scored evidence confusion.

Include negative tests for wrong helper/engine/ONNX/manifest/XML/input/dataset, wrong budget, missing references, NaN/Inf, duplicate IDs, wrong output geometry, timeout/copy/cleanup failure and accurate partial counters. Demonstrate success with nonzero GT/AP and verify saved raw inputs unchanged. Correct the L1A-024 milestone to distinguish accepted diagnostic, completed design and unfinished executable implementation; continue the same ST-EDGE-03 package through a single integrated handoff.

Do not wait for main Luna or request another task for routine implementation. If an external prerequisite cannot be resolved read-only locally, finish independent local work and report one concrete prerequisite list. No new device attempts, no changing raw verdicts, no accuracy/deployment claim. Commit/push scoped changes and unchanged inbox/R1 via NADUNGVN, preserving artifacts and private bytes.
