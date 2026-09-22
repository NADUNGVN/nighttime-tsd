# ST-SERVER-01 R5 integrated decision

Reviewer Astra, 2026-09-22. Reviewed executable revision `46f695acf60b010173222eb666da89f6e0322087` and L2A-053 R5.

## Decision

**Conditional GO for the existing single-host confirmation study**, after the operator verifies the execution prerequisites below. This closes the local integrated implementation gate. It does not certify TensorRT end-to-end correctness, a positive numerical result, or paper acceptance. Luna does not SSH servers; only the user dispatches server commands. The edge lane is independent and remains under its own gate.

## Evidence and limitations

The production dependency seam now permits tests to call the normal `build_real(..., external_boundary=False)` path. CPU tests cover both frozen model contracts across auxiliary calibration, scored cache-only INT8 and scored FP16, actual calibration callback bodies, capture wrapper accounting, public inspectors, and parser/capture failure propagation. This is materially different from the earlier whole-cell simulator. Existing schedule/analysis tests are complementary, not device validation.

Astra's first full super-suite run had **21 PASS, 1 FAIL**, taking 206.802 seconds. The failure was the existing full-schedule test receiving exit 124 from its 30-second synthetic child deadline. Astra reran that exact test alone: **1/1 PASS in 17.134 seconds**. This is consistent with transient local resource contention, not proof of its cause. Preserve both observations; do not report Astra's first run as 22/22. The new normal-path tests passed in that full run. Independent TensorRT-feasibility regression: **30/30 PASS**. No actual model forward, TensorRT, GPU, export or server operation was performed by Astra.

The fixtures still replace the capture runtime and some numerical verification functions; they do not establish real backend metadata, executed arithmetic precision or numerical equivalence. The existing real pooled-evaluator tests and server runtime guards remain necessary. The normal-path test selects baseline INT8; do not claim it independently exercised every nonempty precision override or every possible failure. These limits must stay in L2A-053. They are not authorization to add canary builds or hidden model calls.

## Operator prerequisites and authorized work

1. Luna prepares complete foreground commands, one command per line, with the real checked-out/pushed revision. Record exact executed source/config/helper bytes against the reviewed executable, not only ancestry. Include the runner, child, contract, analyzer, accepted readiness config, calibration producer, graph/capture/XML/matching helpers and lifecycle/telemetry dependencies. Documentation-only handoff commits do not reopen code review. Material executable or scientific changes do.
2. Preserve the unrelated dirty checkpoint. Verify the actual server checkpoint, ONNX and accepted graph/calibration/dataset bindings against the frozen hashes; never fix a mismatch by overwriting user files or loosening a check. Do not export, train or rebuild inputs.
3. Obtain a fresh snapshot of GPU UUID/name/driver, runtime versions, processes, resource availability and disk space. Use current exact desktop PID/path confirmations, never historical PIDs. This build-variation study requires no competing compute under its existing rule. Do not kill or pause other users' processes. Sampled telemetry is not isolation proof. If a prerequisite fails, report it; no blanket new review is needed merely because a machine is temporarily busy.
4. Use fresh root `results/measurement_audit_v1/server_precision_head_confirmation_v1`. The CPU plan may create only its plan/schedule there; before scored dispatch no previous jobs/private execution output may exist. Bind the real XML archive and the exact 1,636-image/2,706-instance dev inventory. No resume, overwrite or replacement attempt.
5. Run the reviewed runbook's plan and scored phases sequentially, with actual paths and confirmations. The existing GO token is permitted for this bounded study after the preceding checks. Do not leave placeholder PID/path/commit values in the operator command. Do not use the runtime-double switch in production.

Locked maximum: **84 builder invocations = 6 auxiliary calibration + 72 scored INT8 + 6 FP16; 78 dev captures = 127,608 dev-image model passes.** Preserve U42/U43/U44, all four INT8 arms, three repeats, model-local FP16 references, canonical rotation/order, same single GPU/runtime, immutable selection caches and fresh timing-cache inputs. No extra smoke, warmup forward, retry/replacement, export, training, benchmark, official-test evaluation or best-build selection. First scheduled jobs are the runtime integration check and count within the budget.

If a job fails or times out, the parent stops; preserve logs, actual counters and unknown completion. Do not rerun automatically or reinterpret a failed job as success. A material protocol deviation requires one evidence-backed exception decision. Unfavorable but valid accuracy is not a reason to stop/reorder/drop a cell.

After execution, the user publishes only approved nonbinary artifacts. Luna audits canonical hashes, coverage, provenance, cache/runtime evidence and partial states, then runs the locked CPU analysis and prepares methods/results/limitations. Continue L2A-053 milestones through this final handoff; the supertask is not complete merely because its implementation gate passed. No approval per model/arm/repeat is needed within this authorization.

Astra leaves this decision and A2L-051 unstaged for Luna to commit/push through NADUNGVN. No server commands were executed by Astra.
