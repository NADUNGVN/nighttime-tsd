# ST-EDGE-02 — explain the existing E2 mismatch without another model run

Owner Luna1, issued2026-09-22 under E2L1-023. End-to-end scope: acquire the three already-produced private output tensors and retained logs, verify hashes, replay comparisons, analyze raw versus application-level consequences on CPU, and deliver an evidence-backed report. No review after each helper; follow the fixed scope through terminal handoff.

## Accepted execution and limits

Astra inspected the five canonical public artifacts at97c2dcc and verified their SHA256 against L1A-022. Manifest SHA25629954570c5b1f44af928e283ce2ab89885c224b165ab85b4f69de6d5ae6a4e74. Recorded parse/build/load each1/1; enqueue/copy/synchronize/compare each3/3; no unknown completions. Engine SHA256581a9ea2eafdae690f25f57ab88ba1bcffdafa99e5322ea50ee43678520e7f56,7,158,097 bytes. **Accept ST-EDGE-01 execution-completion evidence, retain target numerical FAIL.** One observed execution is not demonstrated repeatability; correct L1A-022's phrase reproducible target discrepancy to observed target discrepancy unless referring specifically to deterministic replay of the saved bytes.

For TRT versus source ONNX the recorded results are:

| Fixture | Box mismatches/33600 | Score mismatches/25200 | Maximum box absolute difference, input pixels |
|---|---:|---:|---:|
|00006|965|0|4.2297821044921875|
|00009|1226|0|1.342620849609375|
|00028|564|0|0.954437255859375|

All source/target values were reported finite. Score PASS means within the existing abs+relative policy, not exact scores. Raw box FAIL neither establishes task-level accuracy degradation nor licenses a larger tolerance. The16m35s second-build stage exceeded attempt1's15-minute ceiling; that makes a deadline limitation plausible but does not prove the uninstrumented first attempt would have completed identically. Workspace/subnormal warnings are observations, not identified causes.

## Authorized acquisition, no runtime execution

Use bounded read-only nx access to the known `/tmp/luna1-e2l1-022-model-smoke` root. Download only existing `private/target_output/00006.bin`,00009.bin,00028.bin (235200 bytes each), and scoped build/inference result/event/stdout/stderr logs referenced by the accepted manifest. Preserve original E2 files. No engine download/deserialization, rebuild, warmup, allocation smoke, source forward, inference, alternate device or runtime install. If a required file is absent or hash differs, report it; never regenerate it.

Keep the three raw tensors in a fresh private local staging root outside public Git. Verify exact sizes and manifest SHA256 before processing:00006 d22fdb65d0affaae165c527c2f7d067e989911ae3f42391ada6387aed596bd9e;00009 d67bba2dbcc840a09f3eca943352f653f15160caefb3ff6d7c350236d7837320;00028 cab60f96c43af4835493d90b73052334e121b2b285be41216a088c73347b9a3c. Reuse verified local native/ONNX reference bytes; no HF re-upload or SERVER-01 access. Hash full retrieved logs against recorded evidence; publish sanitized scoped text/JSON evidence only. Temporary network loss is an operational wait, not permission for another model attempt.

## CPU analysis, fixed before examining full outputs

1. Replay the existing strict source and target comparators exactly from saved bytes. Preserve all prior FAILs and original tolerances, channel order, float32 shape[1,7,8400] and xywh input-pixel coordinates. Verify little-endian decoding and immutable hashes before/after processing. Report discrepancies in replay before interpreting model behavior.
2. For native/ONNX/TRT, summarize box absolute differences by x/y/w/h and the known8400-anchor scales only where the accepted mapping supports that interpretation. Report max and quantiles, violation counts and near-zero-denominator cautions. Associate box violations with scores at the same anchors; include all anchors and descriptive subsets at fixed source-ONNX max-score cutoffs0.001 and0.25 plus the union of source/target selected anchors. These are diagnostic strata, not new acceptance thresholds; do not select the cutoff that makes the result look best. Count class argmax/threshold crossings separately from tolerated score differences.
3. Replay one documented existing YOLO11n application postprocess on all three saved outputs identically, CPU-only, conf0.001/iou0.7/max_det300, class-aware single-label NMS. Reuse a verified helper and bind its version/source. Work in the640x640 input coordinate system unless canonical original-image/letterbox metadata are already available and verified. Do not invent scale-back dimensions, apply duplicate decoding, or modify raw buffers through in-place NMS.
4. Report postprocess detection counts/classes/confidence and coordinate changes. Prefer same originating-anchor/class lineage for pre/post-NMS comparison when retrievable. Any supplementary IoU matching must be deterministic, one-to-one, documented and report unmatched detections; it must not replace the failed raw comparison or imply matching against ground truth. No AP, dev accuracy, recall/safety or deployment claim from three selected train fixtures without a scoped ground-truth evaluation protocol.
5. Audit retained stage/log evidence to distinguish observed warnings, actual completion timestamps and unavailable build-time resources. Offline rounding/FP16-cast simulations, if included, are exploratory arithmetic diagnostics only, not proof of internal TensorRT precision or root cause. Do not change build precision/workspace, comparator margins or input semantics to explain away failures.

## Tests, report and completion

Implement a narrow CPU analyzer with tests for binary shape/dtype/hash rejection, original comparison replay, channel/anchor association, threshold crossing, mutation safety, NMS behavior and unmatched/tie handling. Use an existing compatible local environment; no automatic changes to server/E2 inference environments. Freeze the above analysis settings and code revision before real-payload analysis. Pure CPU bugs may be fixed/retested autonomously with revision history; they do not consume a new model attempt.

Deliver `results/edge_readiness_v1/e2l1-023-output-diagnostic/` public inventory/hash bindings, comparator replay, raw-domain tables, fixed-strata tables, postprocess differences, both-attempt chronology and a concise manuscript-ready limitation paragraph. Use a fresh root, preserve source bytes and past artifacts; do not publish private tensors/engines. State what is established, what is unknown, and propose ONE next experiment only if the existing evidence justifies it. Do not execute a third build or benchmark automatically.

Use **L1A-023** for acquisition/implementation_verified/analyzed/terminal milestones; commit/push this unchanged inbox/supertask and scoped code/tests/public report via NADUNGVN. Finish with output_diagnostic_completed_review_required, or a concrete missing-evidence blocker. Main Luna's confirmation implementation proceeds independently. No new Astra message is needed for the bounded read-only acquisition and CPU workflow in this document.
