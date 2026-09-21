# ST-EDGE-01 — finish one instrumented E2 smoke and its research handoff

Owner/executor: Luna1. Reviewer: Astra. Issued2026-09-22 under E2L1-022. This is a single end-to-end package: repair diagnostic logging, test/package, stage, run one authorized new attempt, audit, interpret and report. Do not request separate approval after each local file/test/transfer if the conditions below are met.

## Starting evidence and goal

Read L1A-020/021 and canonical commits ec804d9/65632a1. Attempt1 used the authorized900s build deadline, parsed successfully, timed out during build, terminated with-15 and never loaded an engine or performed inference. Subsequent read-only inspection found no surviving matching process, no engine and no build_result.json. Preserve attempt1 and its unknown build completion. Current resource observations do not reconstruct conditions during that build; unaligned NVRM/kernel lines are not a causal finding. No proven OOM, workspace shortage or model incompatibility exists.

Goal: obtain the first real E2 TensorRT outputs under the frozen contract, or a useful instrumented failure report establishing the next concrete decision. Success is not mandatory for task completion; truthful terminal evidence is. No accuracy, latency, power or Q2 claim follows from three fixture outputs.

Accepted source: local `D:/Research/luna1-e2l1-017-source-v2`, archive SHA256 `bfbd48198faad06dc45a3c1c969e14ace3f1d7c34500ef4715d3bb6997edb28b`, canonical manifest SHA256 `60744680973a73d2986bdf59ce3c6bc956119aeeebbd2106960df57947665c08`, ONNX `bd20b36d640c502358c44edbbde51f05267eda2518b0c0a4cfe84ca18a00d4b7`. No new export, source regeneration or SERVER-01 access is needed. Existing HF download and allowlist validation remain evidence; recheck local/target bytes without repeating uploads.

## Autonomous local implementation and preflight

Improve only observability/lifecycle of the reviewed E2 workflow, retaining Python3.8 compatibility, actual adapter/buffer ownership, comparison policy, precision flags,1GiB workspace, static batch1/640, TRT8.5.2.2 and source identities.

- Use informative TensorRT builder logging (INFO plus timestamped stage events; do not require verbose tactic dumps for every operation). Persist stdout/stderr directly to owned files while the child runs, so timeout/disconnect does not erase evidence. JSON failure reports can include tails plus paths/hashes; preserve full scoped logs. Flush UTC/monotonic stage boundaries and actual counters. A heartbeat is liveness evidence, not proof of tactic progress.
- Independent build and inference child deadlines: new build ceiling **3600 seconds**, inference ceiling **180 seconds**.3600 is an operational bound, not a predicted duration. Keep bounded owned-process-group shutdown; record leader/group/cleanup observations honestly. No reset/kill of unrelated workloads and no unlimited retries.
- Keep optional low-rate existing tegrastats/resource recording across the attempt (e.g.5-second samples), with collectors owned, bounded and cleaned up. Record current read-only power mode and resources. Unavailable power/clock telemetry alone does not block correctness; report it unavailable. No power-mode/clock/swap changes, installation or privileged log access.
- Test durable logging on timeout, actual child argv/path/provenance, successful build then three distinct outputs, second-image failure, unknown versus zero completion, cleanup preserving primary failure, and no implicit fourth enqueue/retry. Run focused and existing edge regressions plus Python3.8 compatibility checks with CPU doubles; no local GPU, real graph or extra device run.
- Freeze and push executable/protocol/test revisions and this inbox using NADUNGVN. Produce the code package from canonical Git bytes and bind all imported helpers/initializer files. Verify manifest/hash on local and target; no hidden worktree changes. Document observability/deadline changes from attempt1, without pretending build artifacts from two different attempts are interchangeable.

## Conditional GO — one additional attempt, no intermediate reviewer gate

This supertask **authorizes exactly ONE additional E2 diagnostic attempt** once the above tests/package checks pass and fresh E2 identity/runtime/resource checks match the approved device. This supersedes the previous NO-GO for a second build only within this exact bounded scope. No additional Astra message is needed between successful local tests, staging, execution and audit.

Use existing nx alias only; no network scanning, host substitution or SERVER-01 SSH. If network is down, record network_wait and resume preflight after a reported change; do not consume another model attempt or tight-loop probes. Read-only probes/CPU tests do not count as model builds. Verify no old owned build survives and no resource-incompatible workload is present. Do not infer availability from free VRAM alone or invent an isolated-device claim.

Create new absent code/output paths, e.g. `/tmp/luna1-e2l1-022-code` and `/tmp/luna1-e2l1-022-model-smoke`; reference the verified existing source root. Do not overwrite/resume/delete attempt1. On collision, inspect status and distinguish a never-dispatched staging collision from an already-consumed attempt; choose a documented fresh staging suffix only before dispatch. Never use path renaming to obtain another build budget.

Budget for attempt2: **one FP16-enabled TensorRT builder invocation**,1GiB workspace; if build succeeds and engine/runtime/binding hashes validate, **one inference child with at most three application enqueues**, fixtures00006,00009,00028 in that order. Zero warmup, calibration, INT8, native/ORT source forward, benchmark, export, retraining or alternate engine selection. One build failure/timeout ends this attempt; do not run inference on partial bytes. Successful build automatically permits the already-budgeted inference stage; do not stop to ask permission just because an engine exists.

After timeout/disconnect, inspect the existing owned run read-only and establish its state before any further command. No second invocation of an ambiguous stage and no third attempt. If inference fails partway, preserve actual attempts/completions and stop. A target numerical mismatch is a reportable diagnostic result, not permission to relax tolerance, change decoder, select another build or rerun the fixture.

## Audit and final deliverables

Bring back scoped public JSON/JSONL/log evidence, not engine/tensor bytes. Preserve engine and raw outputs privately on E2 for audit; hash them. Verify package/source/engine/input/output chain and exact counters against logs, full log hashes, comparison status and cleanup. Keep export_discrepancy (native versus source ONNX) separate from target comparisons (TRT versus ONNX/native); source strict FAIL for00028 remains even if target diagnostic passes.

Write one final report covering both attempts: deadline/settings/code, parse/build/engine/inference counts, observed timing/resource/log evidence, source versus target comparisons and limitations. Build wall time is not inference latency. Three train fixtures do not establish dev accuracy or deployment performance.

Deliver public artifact inventory, per-fixture comparison table if available, methods/limitations paragraph ready for the research evidence ledger and one evidence-backed next-step recommendation. Terminal status is edge_smoke_completed_review_required or edge_smoke_failed_diagnosed_review_required. If failure remains unresolved, say what is unknown and propose one specific bounded next change with rationale; do not execute it.

Use **L1A-022** for milestones: implementation_verified, staged, running, artifact_audited, terminal. Push this unchanged inbox and scoped results via NADUNGVN. Do not mark the whole task done after code, instructions or a forwarded message. Do not wait for main Luna. Do not start full-dev capture, another edge, FP16/INT8 comparison, latency/energy sessions or paper deployment claims without the next scientific decision.
