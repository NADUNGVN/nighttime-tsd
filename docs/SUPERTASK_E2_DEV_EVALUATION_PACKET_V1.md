# ST-EDGE-03 — close saved-output diagnosis and prepare a task-level evaluation packet

Issued 2026-09-22 under E2L1-024. Owner Luna1. Work independently of main Luna's server confirmation. One consolidated local implementation/protocol/test handoff, not approvals per helper. Current authority: local CPU/source inspection/saved-output replay only. No new SSH, transfer, source model forward, export, build, target inference, benchmark, installs or device changes in this packet.

## Reviewed evidence and scope

Astra inspected d723cdf's four canonical public artifacts and independently replayed all three fixture analyses from local private native/ONNX/target bytes, verifying their hashes. After JSON normalization all fixture results exactly reproduce. Five focused analyzer tests pass. Raw FAILs remain unchanged; source 00028 also retains its strict export FAIL. Accept the completed saved-output diagnostic as descriptive evidence, not numerical equivalence or deployment readiness.

For TRT versus ONNX, kept anchor/class lineages are identical: one detection for 00006, two each for 00009 and 00028. Maximum same-lineage coordinate changes are 0.0971832275, 0.2123107910 and 0.2636260986 input pixels. Across all five detections IoU ranges from 0.9793940798 to 0.9946785885. The earlier prose pairs per-fixture maximum coordinate difference with maximum IoU, but those need not belong to the same detection: for 00009 the 0.21231-pixel row has IoU 0.979394, not 0.987351. Fix the prose/table interpretation; do not present maximum IoU as a worst-case measure.

No evidence currently establishes a coordinate-decoding bug or justifies 'correcting boxes'. Do not shift/round/clip/fit target boxes to references, change comparator tolerances or rebuild a more favorable engine. Raw numeric and task-level results answer different questions.

## Deliverable 1 — finalize CPU replay and evidence bindings

1. Preserve published v1 unchanged and add an audit/addendum using a fresh root. Correct L1A-023 analysis SHA256 typo: actual canonical blob is `b9df6015385a5595997e1eefdc5253553638bd2a39237ac22c1163cd7c8c2058`. Keep old L1A-022 'reproducible discrepancy' qualified as saved-byte replay, not repeat device execution.
2. Replace self-derived expected source hashes in diagnostic CLI with the pinned accepted source manifest and verify its hash. Bind target manifest and full retained log hashes to accepted evidence. Recheck before/after input hashes. Published code/helper/runtime bindings must be complete.
3. The dependency-free postprocess is a custom diagnostic, not yet identical to the accepted helper: it computes Python float64 box conversion and uses >=confidence, whereas installed Ultralytics 8.4.102 uses float32 operations and >confidence. Astra compared the saved nine outputs against the real CPU Ultralytics non_max_suppression at conf .001/IoU .7/max_det300/nc3/single-label, with return_idxs: all anchors matched; maximum coordinate arithmetic difference was 1.52587890625e-05. Record this as evidence, not exact helper equivalence. Bind/version the actual helper and explicit CPU backend; retain legacy custom tables as diagnostic history. No model inference is needed to replay saved raw tensors.
4. Tests must use positive-area boxes to prove same-class suppression and different-class retention. Existing NMS fixtures leave height zero and therefore do not demonstrate suppression. Add boundary confidence, IoU/ties, immutable inputs, non-finite outputs, max_det, unmatched detections and wrong source/log hash cases. Supplementary matching without an IoU cutoff is descriptive assignment, not proof of detection agreement; disclose it and retain all IoU/unmatched rows.
5. Publish all five detection rows (paired coordinate delta AND IoU), min/max IoU, class/confidence/count differences and corrected limitations. Preserve all comparator FAILs. No claim about accuracy from these three training fixtures.

## Deliverable 2 — prospective FP16 dev evaluation design and executable CPU/mock packet

Prepare, but do not execute, a single study to evaluate the existing YOLO11n E2 FP16 engine against its bound source ONNX on the canonical full 1636-image development split. Reuse the existing engine on its original E2/runtime; do not download/load it locally or rebuild it. Freeze its known hash and source ONNX identity. If the engine is unavailable later, that is a separate decision, not implicit build permission.

Resolve source ONNX and preprocessing identity explicitly: the E2 bundle source is not interchangeable with another ONNX merely because checkpoint/output shape agree. Reuse existing full-dev reference predictions only if actual ONNX/input/postprocess/dataset bindings match; otherwise propose a separately counted CPU reference capture. No native/reference forward is authorized by this packet. Main Luna's v8/v26 bridge is not a YOLO11n reference.

Design the command chain around user-operated server CPU materialization when local prerequisites are unavailable, approved private transfer, then Luna1-operated E2 only after a later integrated GO. No assumed SERVER-01 SSH from local. Define private package inventory/hashes and storage/memory/timeout requirements, strict ownership/fresh roots, bounded interruption and no automatic retry/rebuild. Keep public reports/predictions separate from private engine/ONNX/input tensors.

Freeze canonical dev image membership, original-image/letterbox transforms, names, output semantics and original-coordinate matching; no duplicate NMS. Primary descriptive endpoints: paired source-ONNX versus existing E2 FP16 full COCO/XML AP50/AP50-95, secondary XS/S, plus counts/class/coordinate diagnostics. Use the already accepted evaluator and paired image-bootstrap convention, with fixed sample plan and no invented success/noninferiority margin. Retain raw export/target strict FAILs separately. Dev is development evidence, not untouched final-test confirmation.

Explicitly enumerate future reference/target calls (one counted target pass per dev image = 1636; reference calls only if needed). Warmup, shape probes and debug/reference forwards cannot be hidden. No latency/energy benchmark or precision-arm sweep in this accuracy packet. Missing power telemetry does not block correctness; no energy claim.

Implement/test local CPU parent, package validation, mocked E2 engine loader/runtime child, observed counters, buffer ownership, postprocess/evaluator, failure/timeout/partial publication and result audit using existing lane components rather than new framework design. Test real internal orchestration with external runtime doubles; do not manufacture terminal counters as the only integration evidence. Include successful end-to-end synthetic artifact-to-analysis and negative hash/shape/dataset/cleanup tests. Tests use synthetic/saved outputs, never a new frozen-model forward.

## One integrated handoff

Use L1A-024 milestones for diagnostic closure and prospective packet readiness. Provide a single code/protocol/test/runbook package with immutable inputs, all unresolved prerequisites, exact future counts/resources and user/server/edge division. Do not ask for approval per local repair. Stop at `edge_dev_packet_implementation_review_required`; server/E2 execution remains NO-GO until review. If the full prospective implementation is blocked, give one concrete evidence-backed blocker list, not a claim of task completion.

Commit/push scoped work and unchanged Astra inbox/supertask via NADUNGVN; preserve old artifacts and private bytes. Main lane proceeds independently. No claim of AP improvement, Q2 acceptance or that edge correctness is solved before task-level evidence exists.
