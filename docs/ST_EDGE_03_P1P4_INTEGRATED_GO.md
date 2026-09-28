# ST-EDGE-03: P1-P4 review and conditional dev execution

## Review — 2026-09-28

Reviewed `988bd6f715df99da6998914404ce507204e10790` and L1A-029. Accept P1-P4 for the existing prospective YOLO11n FP16 source-ONNX versus E2 TensorRT dev comparison, subject to the operational prerequisites below. This is not acceptance of historical raw numerical equivalence and not a benchmark authorization.

Astra independently ran the packet suite with the existing dependency-rich CPU environment: **31 PASS, 1 POSIX-only SKIP** (32 collected). Saved-output diagnostic suite: **8/8 PASS**. Unlike Luna's earlier environment, the review environment has pinned Ultralytics 8.4.102, so its three helper-dependent cases executed. Additional synthetic CPU comparison found byte-identical preprocessing against the pinned LetterBox implementation for all five shapes: 480x640, 641x319, 319x641, 720x1280 and 640x640. This is synthetic preprocessing evidence, not actual dev inference or proof of byte equality across different OpenCV installations.

Code inspection confirms explicit per-image transform restoration, target NumPy/OpenCV preprocessing without source CUDA rejection, canonical source scope/provider and before/after ONNX checks, source-bound input hashes, normal CLI deadlines of 54000 seconds/stage and 30 seconds/whole image, and rejection of abnormal child termination or incomplete terminal artifacts. The POSIX TERM-ignore escalation test remains unexecuted by Astra on Windows. E2 NumPy/OpenCV availability is not yet established. No actual source forwards or E2 execution occurred in this review.

## One end-to-end assignment: E2L1-030

**GO local CPU tests/packaging and bounded read-only E2 dependency/resource checks. Conditional GO one source CPU pass and one E2 dev pass after their prerequisites pass.** This supersedes E2L1-029's execution HOLD only within this assignment. No extra Astra response is needed between stages if all stated conditions pass. Complete preparation, operator handoff, execution, artifact audit and paired analysis as one supertask, not another proposal-only packet.

### 1. Freeze and prepare without model calls

- Commit this unchanged inbox/decision with the scoped runbook through NADUNGVN, record the resulting full revision and actual helper hashes. Preserve the reviewed numerical paths; administrative commands/receipts may be added without changing algorithms.
- Keep the existing canonical 1636 ordered dev images, 2706 instances, source ONNX, existing E2 engine, preprocessing, NMS/evaluator and statistical design. Validate all frozen hashes from the existing contract. Do not export, rebuild or copy a server engine to E2.
- Use fresh absent roots identifying `e2l1-030-source-reference-v1`, `e2l1-030-target-dev-v1`, and `e2l1-030-analysis-v1`; record full actual paths before execution. Do not overwrite any prior output, including source-bundle numerical FAILs.
- Through the already approved `nx` alias only, perform a bounded read-only dependency/resource check: device identity, existing engine hash, Python compatibility, NumPy/OpenCV import versions, required runtime libraries, free storage/memory and other workloads. No package installation, power/clock changes or process interference. Network failure is a bounded operational hold, not permission to scan/guess hosts or loop retries.
- Run the existing POSIX termination test with a CPU/mock child in a suitable Linux environment before target dispatch. This is test code, not model inference. Record the result; do not claim the Windows skip passed. Missing dependencies or unavailable engine remain an explicit hold; do not install or rebuild automatically.
- Use source CPU postprocessing/analysis in its pinned dependency-complete environment. E2 needs its target runtime and NumPy/OpenCV, not an installation of the source Torch/Ultralytics/ORT stack.

### 2. Source reference — user-operated server, CPU only

Give the user complete foreground commands with resolved paths and full revision, CUDA hidden and existing thread/auto-install controls. Luna1 does **not** SSH to SERVER-01. The user can run this CPU lane while unrelated GPU work proceeds, subject to available CPU/RAM; do not gate CPU work on an empty GPU.

Authorize at most **1636 source ONNX CPU forwards**, one per canonical dev image, and **zero native model forwards**. Use the already accepted source ONNX; no new export or calibration. Before the first forward, require all image/hash/shape/scope checks and actual CPUExecutionProvider. Require finite output, complete counts, unchanged ONNX hash and source input/preprocessing records before declaring the reference verified. Do not call a partial manifest execution-ready.

The user publishes sanitized reports and transfers private package bytes using the existing approved transfer workflow/location, including the already approved private HF repository when needed. Pin archive hash/revision and verify exact allowed members and hashes locally before transfer. Raw tensors/images/models stay out of public Git. Do not invent SSH access to SERVER-01 or use public hosting for private bytes.

### 3. Target E2 — one bounded existing-engine pass

Only after the source manifest and package gate are verified, stage the scoped package to the approved E2 location and verify it there. Require current E2 identity, correct engine/runtime binding, sufficient disk and memory, no conflicting study-owned process and the existing workload policy. The guard must compare each real preprocessed input's hash/metadata to its source reference before its enqueue; any cross-platform mismatch stops the pass, not a tolerance or preprocessing adjustment mid-run.

Authorize at most **1636 target enqueues**, exactly one per dev image if complete, using the existing FP16 engine. **Zero builds, warmups, extra smoke calls, retry/resume or benchmark sessions.** The former smoke authorization is consumed and irrelevant. Keep 54000-second stage and 30-second whole-image limits; they are operational ceilings, not latency estimates. Preserve durable partial counts, primary errors, termination and unknown cleanup if failure occurs. No rerun of a failed image or failed stage under this authorization.

### 4. Audit, paired analysis and terminal handoff

Retrieve already-produced scoped evidence; check hashes, source/target identity, 1636 ordered rows per side, per-image input binding, actual counters, finite outputs, terminal exit/cleanup and immutable source/engine identities. Run the existing CPU postprocessing and canonical pooled COCO/XML size evaluation, paired 1000-draw bootstrap with seed 20260916, and locked contrasts. Keep source-export discrepancy separate from target-source discrepancy. Do not replace historical strict FAILs, tune boxes/tolerances, select favorable images or claim deployment/latency/energy readiness from AP.

Append L1A-030 milestones: `implementation_review_accepted`, `operator_pending`, `source_verified`, `target_running`, `artifact_audited`, then `analyzed_review_required` or precise `failed_partial/operational_hold`. Push sanitized artifacts and the final report through NADUNGVN. Stop after the completed analysis or first genuine execution failure; a failed stage consumes its attempted calls. A package/dependency gate stopping before calls is not scientific failure. Missing power telemetry alone is not a correctness blocker because energy is outside this task.

## Independence and limits

Main Luna is repairing the server calibration order; this edge lane need not wait for that repair or the 84/78 study. Coordinate only real shared resources with the user. No third E2 build, engine change, precision-arm expansion, accuracy acceptance threshold invented after results, or new experiment is authorized. Any numerical outcome is reportable; the task is a correctly executed comparison, not achieving a preferred result.

Astra used the API/interface-design skill to review the actual producer/consumer contracts and failure boundaries, not merely test counts. Astra did not SSH, transfer private files, perform model inference, build or benchmark during this review.
