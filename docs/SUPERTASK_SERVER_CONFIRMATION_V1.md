# ST-SERVER-01 — complete the cross-model precision-head confirmation package

Owner: main Luna. Reviewer: Astra. SERVER-01/other server executor: the user only. Issued2026-09-22 under A2L-045. This is one work package spanning implementation, integrated review, operator execution, audit, analysis and manuscript-ready reporting, not a request for a separate approval after each file or unit test.

## Outcome and accepted starting point

Deliver an auditable answer to whether the YOLO11n precision-head observation extends to the frozen YOLOv8n and YOLO26n checkpoints under the locked calibration/build design. A null, negative or architecture-sensitive result is a valid completed deliverable. Do not promise a positive result or Q2 acceptance.

Astra inspected canonical artifacts at03af60ff4a639818240b760849f3655331e9f972 and L2A-052 atb3741da41d3aa95e11716d7cdc8cbb8eef44c3a6: inventory13/13, two models completed, one build/model, eight TRT and eight CPU ORT calls/model, eight records/traces/model, finite outputs and expected shapes, released owners and eight synchronizations/model. Raw ORT reference hashes match private-materialization hash evidence. Smoke manifest SHA256:4c9ea719236569550e26743d5e07cf8f8a3a849f72b065a4357af0630232a71d. This accepts bounded execution feasibility, not INT8 validity or numerical equivalence. Binary invariance remains server-recorded evidence; no private engine was locally rerun by Astra.

Keep numerical limitations visible: recorded maximum v8 raw box difference2.663726806640625 pixels and score difference0.005711495876312256; v26 fixed-row box difference638.68994140625 and up to173 class-row mismatches per fixture. The v26 fixed-row statistic is not a matched-detection localization error. It cannot prove failure or success of task accuracy by itself. The completed full-dev source/ONNX bridge and historical strict FAILs remain separate evidence. Do not manufacture an equivalence threshold from these observations.

## Locked scope and work budget

Use docs/PRECISION_HEAD_CONFIRMATION_PLAN_V1.md, its D3 schedule, configs/precision_head_confirmation_v1.json and the accepted graph/numeric/bridge evidence. Do not rerun those studies merely to start this task.

- Frozen models: YOLOv8n and YOLO26n; accepted per-model ONNX hashes, no new export/training/checkpoint replacement.
- Train-only Uniform U42/U43/U44,1024 images each, manifest order, same declared MinMax recipe. No calibration-policy search.
- Four arms: baseline_int8, bbox_fp32, classification_fp32, both_fp32; three builds per selection/arm. Three FP16 references/model.
- Total future confirmation budget:6 auxiliary cache-generation builder invocations plus72 scored INT8 plus6 scored FP16 builds = **84 builder invocations**, **78 dev captures**, **127,608 dev image-model passes**. The completed feasibility smoke is historical and not one of these scored cells. No extra canary builds or hidden warmup/reference forwards outside the approved accounting.
- Per-model schedule: three auxiliary jobs, then three rotated13-cell rounds, shifts0/4/8. Preserve all cells and observations; no best-build selection or AP-based stopping/reordering.
- One model/selection calibration cache shared read-only across its12 scored INT8 builds; zero consuming calibration batches/writes. Fresh empty timing input for every build; record outputs but never reuse them. No cross-model cache or engine reuse.
- Full1636-image dev split and2706 XML instances; full/XS/S endpoints. Primary contrast both_fp32 minus baseline_int8, full AP50-95. Mandatory branch controls and architecture-specific FP16 references. No official-test/negative/weather evaluation or latency/energy benchmark in this task.

## Phase1 — implement and self-review the entire package locally

Read applicable repository instructions and reuse existing builder, calibration, graph-audit, capture and evaluator components where compatible. Do not copy YOLO11n layer names into the new architectures or create a generic framework beyond this study. Finish these deliverables together:

1. A versioned execution contract/config referencing the immutable accepted input/schedule hashes. Preserve historical readiness config; explicitly resolve its deferred calibration preprocessing and graph fields in the execution contract with actual producer/helper hashes. Trace the existing accepted preprocessing, do not guess. Freeze builder flags, FP32 target/output constraints, OBEY behavior, workspace and runtime consistently before data collection. Use accepted architecture-specific active convolution mappings, with non-convolution helpers excluded and requested/effective evidence separated. Unsupported or ambiguous constraints stop the affected block rather than silently falling back.
2. CPU-only orchestration, isolated build/capture children, source/cache/engine identity chain, durable counters, atomic state, bounded child deadlines and owned-process lifecycle. Include fresh-root protection and complete partial/failure inventory. Retain private engines/caches/tensors on their producing host for audit; publish only allowed reports/logs/hashes. Plan disk/RAM needs and deadlines in advance; no infinite waits or automatic GPU retries.
3. Architecture-correct capture and verifier. Freeze preprocessing and postprocessing before scoring: v8 raw decoded output needs the locked NMS path; v26 already has its accepted end-to-end representation and must not receive a second NMS. Explicitly reconcile native matching with the earlier source strict FAILs: check the implemented capture route on identical saved raw data with approved helper behavior, not invented tensor equality between native/ORT/TRT. Keep COCO/XML and Ultralytics channels distinct; do not quietly change the estimator or mix historical settings. Any unresolved scientific contract is grouped into the integrated review, not bypassed.
4. CPU analysis that reproduces points and paired image-bootstrap CIs using PCG64 seed20260916,1000 draws, shared image resampling across arms/builds/selections before averaging. Preserve duplicate sampled images. Report ddof1 within-selection build SD and between-selection-mean SD as distinct descriptive quantities; nine engines are not nine independent calibration samples. No positive-result filter, invented significance or FP16 noninferiority margin.
5. Tests through actual CLI/parent/child orchestration with external-runtime doubles, both output shapes and semantics,84-job schedule accounting, calibration isolation/cache-only consumption, precision-target discovery, immutable raw outputs, mapping ambiguity, timeout/partial/crash, wrong model/cache/GPU, duplicate/missing cells, matching and analysis/bootstrap reproducibility. Run all relevant tests; separate genuine failures from unavailable real fixtures/dependencies without calling skipped end-to-end checks verified. Local code/tests require no GPU or actual frozen-model inference.
6. One operator runbook: exact revisions and executed-helper byte checks, fresh output roots, inventory/resource preflight, foreground commands, status/exit evidence, narrowly scoped artifact publication commands, and CPU analysis instructions. No SERVER-01 SSH by Luna. Do not edit dirty checkpoint bytes or stage unrelated work.

## One integrated pre-execution gate

After the whole implementation/protocol/test/runbook package is complete, push it and append **L2A-053** with one consolidated review packet: executable revision, changed paths, contract hashes, test results/limitations, schedule/accounting, exact remaining decisions and ready-to-run commands. Status: implementation_complete_integrated_review_required. Astra reviews the whole package once before scored GPU dispatch; **84-job execution is NOT authorized merely by the old FP16 smoke passing**. Do not stop for review after individual helpers or routine bug fixes. Do not call this supertask fully completed at this gate.

Scientific/runtime changes, unsupported TRT target constraints or uncertain data semantics must be surfaced in that packet. Purely local code repairs and tests within the fixed contract are autonomous. If a critical issue prevents finishing, report one evidence-backed blocker list with proposed resolutions, not successive tiny permission requests.

## Phase2 — user-operated server execution after integrated GO

Once Astra accepts the implementation and grants scored execution, coordinate directly with the user for commands/snapshots and continue through the locked sequence without new approval per model, selection, arm or repeat. First scheduled auxiliary/scored jobs serve as runtime integration checks and count inside the84; do not add a separate smoke. Runtime gates concern correctness, provenance and resources, not favorable AP. Preserve unfavorable metrics and finish valid scheduled cells.

Multiple servers may run in parallel only after the user offers them and compatibility is established, split by complete model block:42 builders/39 captures on each of two hosts. Keep all arms/selections/repeats and FP16 controls of a model on the same GPU/runtime. Never migrate a running model block or copy server engines across devices. Otherwise run both blocks on SERVER-01. CPU preparation/analysis and the edge lane may proceed independently.

Distinguish this repeated-build study from the prior shared-workload feasibility smoke: competing compute can confound the build-variation endpoint. Use the existing confirmation-plan workload rule, offer another available host or defer that model block instead of disabling guards or changing others' processes. Known desktop exceptions remain explicit and sampled telemetry does not prove isolation. A timed-out or invalid cell stops the affected block; no implicit retries/replacements, no opportunistic seed/model change. A separately assigned unaffected block may continue. Record budget consumption and partial results; seek a single exception decision only if the frozen design cannot continue.

## Phase3 — audit, analyze and hand off research results

After the user pushes evidence, inspect canonical Git bytes, engine/cache links, all job identities/counters, dataset/input bindings, output semantics, telemetry and completion. Check against the full expected schedule, not just file counts. Missing/invalid cells yield an explicit incomplete report, never a positive confirmation verdict.

Run the locked CPU analysis on available authorized local resources; if server CPU is needed, give the user bounded commands. Produce machine-readable summary JSON and Markdown tables: all cell metrics, primary/control/FP16 contrasts with conditional image CIs, build and selection SD/ranges, full/XS/S adverse effects, cache/precision/telemetry caveats. Reproduce point estimates independently from stored predictions with the accepted evaluator where data are available. No need for additional GPU runs to beautify a report.

Deliver a manuscript-ready methods/results/limitations section and updated evidence ledger separating YOLO11n discovery, cross-model confirmation, CPU export bridge and E2 engineering smoke. Classify evidence as supported, mixed, not established or incomplete; do not automatically select a deployment arm or launch main15/B/C. Use already-registered screening rules only as engineering descriptors, not guaranteed paper acceptance.

Terminal status: confirmation_completed_review_required with full audit/analysis, or confirmation_incomplete_blocked with preserved evidence and one concrete next-decision packet. Mark the supertask complete only at that terminal handoff, not when code is pushed.

## Coordination and autonomy

Keep this main lane on its own worktree and inbox/outbox. Luna1 owns E2 paths only. Both Lunas commit/push Astra inbox entries unchanged along with scoped work using NADUNGVN; Astra does not push. Use L2A-053 milestones rather than new task numbers for each fix. Report concise progress by phase and actual completed counts, not speculative percentages. Waiting for a server command/result is operator_pending, not a need for another design approval. No token, checkpoint or private archive in public Git.
