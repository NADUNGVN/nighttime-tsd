# ST-SERVER-01 integrated review R1 — implementation incomplete, return the whole package

2026-09-22. Reviewed228a6a3a0d091350dd037e8eeb40a1d38fa7b501/L2A-053. **NO-GO scored server execution.** This is the existing integrated gate not yet satisfied, not a new scientific gate. Continue the same supertask autonomously and return one complete remediation packet; do not ask for approval after each fix below.

Astra read the three new scripts, contract, tests, protocol and runbook, applied interface-design review, and ran the new suite using the existing `D:/Research/paper/local/measurement_audit_env/Scripts/python.exe`: **8/8 PASS, no skip**. NumPy is available in that environment. Those tests pass while the actual execution path is absent and accepted artifact interfaces fail. No GPU, TensorRT, frozen-model forward or actual ONNX graph was run in this review. Preserve the dirty checkpoint.

## Blocking findings, with bounded reproductions

### 1. No builder/capture executor exists in this packet

`run_precision_head_confirmation.py` implements plan writing only. Calling its real CLI with `--phase scored --go-token ASTRA_INTEGRATED_GO_REQUIRED` exits1: `Server phases are dispatched by the post-GO runbook; no GPU work is permitted in the local parent`. There is no child dispatch, build/calibration/capture implementation, telemetry lifecycle, or producer of the analyzer's cell_metrics.json. A CPU-only parent must dispatch runtime children; it must not refuse all execution forever. A runbook referring to a later command is not an executor.

Implement the complete production CLI and bounded children for the84-job/78-capture design using compatible existing components. Prove the actual CLI/parent/child path with external runtime doubles, not a fabricated completed report. Cover two models and the complete schedule through final inventory; failures must preserve real partial counters, stop the affected model block and never invent successful cells. Do not release a server command until this integrated gate passes. A magic string is not authorization by itself.

### 2. Schedule differs from the accepted design and its validator accepts invalid cells

New build_schedule places FP16 last in the unrotated13-cell list. Accepted prepare_precision_head_confirmation.py `_expected_schedule` logic and design AppendixA place FP16 FIRST, followed by U42/U43/U44 arms, then apply shifts0/4/8. This changes actual job order despite matching aggregate counts. Astra observed the new first scored cell U42/baseline_int8 instead of FP16.

Astra changed the first scored row to selection=U999 and arm=not-an-arm; validate_schedule still accepted it. Comparing model_rows to their sequence sort also does not prove model-block contiguity. Replace count-only validation with exact canonical job-field validation, reusing the accepted schedule producer/validator or comparing to its immutable output. Include identities, order, repeat, dependencies, cache/capture policy and allowed models/arms/selections. Test wrong arm/selection, duplicate semantic cells with unique job IDs, model interleaving, moved FP16, mutated dependencies and rotation positions. Do not redefine the locked schedule to make the new code pass.

### 3. Actual accepted mapping/schema and calibration interfaces are not wired

The committed graph_audit.json stores active_branch_audit inside `mapping`, not at the root expected by new mapping_targets. Passing either real accepted graph artifact to that function raises no-target-nodes. Do not change historical artifacts to fit invented test fixtures. Read the real producer schema, map active Conv nodes explicitly using op_type/source ownership rather than only a /Conv suffix, and validate canonical mapping/provenance hashes against accepted ONNX identities. Add regression tests using canonical Git JSON artifacts, which are available without model binaries or GPU.

The execution config names `scripts/uniform_build_repeat.py::prepare_calibration`, but that function does not exist. Inspect and bind the actual producer/preprocessing implementation. Resolve train selection IDs/bytes/materialized YAML, deterministic order, MinMax tensor recipe, six isolated calibration caches and cache-only consumers in executable code, not just prose. `verify_cache_only_audit` assumes exactly one read callback; previous replay evidence observed more than one read. Require the reviewed same-cache identity, actual successful consumption, zero batches/writes and recorded read count, not an invented exact callback count. Separate end-of-stream callbacks from actual consumed batches.

Validate execution contract fields against accepted immutable scientific sections and actual readiness/calibration/dev/runtime evidence. Reading a status string or echoing a mapping hash does not verify current source bytes, no-overlap, helpers or runtime. Record and test source/helper/contract identities, flags/constraints and failure semantics across the complete pipeline.

### 4. Analyzer implements the wrong bootstrap estimator and does not report required contrasts

`bootstrap_contrast` averages supplied per-image scalar metric differences. That is not the locked COCO/XML AP estimator: AP is not an average of per-image AP. The accepted paired evaluator uses resample_ap on duplicated image records/matching evidence, then aggregates build/selection metrics for each shared draw. Reuse/extract that actual estimator logic; do not copy its historical assumption that repeats are exact, because this confirmation intentionally measures non-exact build variation.

Moreover analyze never calls bootstrap_contrast, computes only full AP50-95 grouped variation, omits primary/control/FP16 contrast tables, XS/S endpoints and between-selection-mean SD, and nevertheless reports completed_descriptive_analysis. Implement all endpoints/contrasts/variation/CIs as specified in ST-SERVER-01. Preserve image multiplicities, valid/undefined counts, fixed PCG64 sample plan and point reproduction. FP16 has three model-level repeats, not nine independent selection-specific controls; define aggregation from the locked design and show it explicitly. Do not fabricate per-image metrics or silently emit completed results without required CI inputs.

Audit every cell's semantic identity, hash/provenance/validity and observed capture before analyzing; matching78 arbitrary job IDs is insufficient. Hash the actual plan instead of mislabeling its contract hash input_plan_sha256; verify schedule/content hashes rather than copying fields. Reject existing output roots. Add a small CPU synthetic detection/GT fixture where mean per-image AP differs from pooled AP, plus duplicate-resample, absent-class/undefined draw, missing/invalid cell, provenance mutation and FP16/control contrast tests. An independently replayed accepted estimator output is required, not only RNG repeatability and sample-SD unit tests.

### 5. Runbook/protocol overstate implemented behavior

The runbook proposes rerunning graph preparation even though ST-SERVER-01 explicitly reuses frozen accepted ONNX and prohibits a new export. Replace that with read-only verification. Provide executable plan/build/capture/analysis commands, supported CLI options for model-block host assignment/current desktop confirmations, complete checked-out helper binding and actual publication manifest generation. Dynamic PID fields can be filled after snapshot; executable implementation must already exist before review.

Correct L2A-053's implementation_complete claim to implementation_incomplete_remediation_in_progress until the above exists. The limitation is not merely missing local model data/NumPy. Templates are acceptable for unobserved manuscript results, but not as substitutes for code paths or numerical outputs. Keep previous reports as history with explicit addenda, not silent rewrites of findings.

## Single resubmission acceptance checklist

- Production CLI executes the full84/78 orchestration under external doubles; actual TensorRT/build/capture adapters exist for the eventual server path.
- Canonical schedule and real graph artifacts pass without binary fixtures; all corruption tests reject.
- Real calibration producer, graph targets, constraints, bindings, cache audits, child lifecycle and capture/evaluator producer chain are connected and tested.
- Correct paired detection-level AP resampling and all required outputs run on synthetic CPU fixtures, with independent point/CI replay and no per-image AP shortcut.
- Targeted/relevant regressions pass; give exact interpreter/dependencies and explain genuinely unavailable external fixtures without claiming them verified.
- One complete protocol/runbook with fresh roots, accounting, deadlines, no hidden calls, immutable input/helper hashes, scoped artifact push and failure paths; no new export or study.
- Publish one executable commit and L2A-053 remediation addendum linking every acceptance item to code/tests/evidence. Do not mark implementation complete before these checks run.

Luna owns all local repairs and may work across these components without another microtask. The user still runs servers only after Astra's integrated GO. Keep84 builders/78 captures unstarted; no calibration-policy/arm/tolerance change, checkpoint replacement or private-binary publication. This packet consolidates the blockers of the original assignment; no extra research branch is opened.
