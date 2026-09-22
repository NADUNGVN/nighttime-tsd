# E2L1-027 actual execution-chain review

Astra, 2026-09-22. Reviewed `af34707`. Decision: **GO local completion; NO-GO source forwards or E2 execution.** No new scientific scope.

## Accepted work

Astra independently ran the packet suite **19/19 PASS** and diagnostic suite **8/8 PASS** (27 total). The 108-test wider regression is Luna1-reported, not independently rerun in this review. Canonical XML record-map normalization and real synthetic COCO/paired-resampling tests now pass in the existing dependency-rich CPU environment. Accept these fixes, the pending-versus-verified distinction, canonical ID digest recomputation, and cleanup-before-result correction in the mock durable child. Preserve them.

## Remaining implementation defects in the actual CLI chain

### C1 — the supposed source ONNX reference invokes native PyTorch

`run_source_reference_stage` loads a checkpoint through `UltralyticsSourceRuntime.load_model`, then calls `native_forward`. It neither accepts nor opens the pinned ONNX nor invokes ORT. Its manifest nevertheless includes the constant `source_onnx_sha256`. This is not the prospective source-ONNX versus E2 comparison. The stage also does not verify the checkpoint's frozen hash before loading it.

Implement the intended **existing pinned ONNX CPU reference**, with actual ONNX before/after hash and observed CPUExecutionProvider, using the accepted preprocessing and output contract. Reuse the existing source-bundle ORT methods where compatible; do not export or build anything. Native forward is not needed for this dev contrast and must not be added as an extra 1,636 calls. Keep 0/1636 as the executed source count until separately authorized. Update the CLI/runbook to accept the real pinned ONNX rather than pretending a checkpoint call is ORT evidence. Test that the normal stage calls the fake ORT session and cannot call a fake native model, and rejects hash/provider mismatches before forwarding.

### C2 — producer outputs do not feed the analyzer

Source and target stages output raw-tensor inventory rows (`image_id`, private path, bytes, hash). `run_canonical_analysis_stage` expects detection records (`image`, `orig_shape`, `xyxy`, `confidence`, `class_id`). There is no implemented conversion stage between them. Astra supplied a producer-shaped row to `_canonical_record_map` and reproduced `CANONICAL_EVALUATOR_RECORD_INVALID:source`.

Implement a shared saved-output processing stage using the accepted helper, original image/letterbox metadata and the frozen coordinate convention. It must produce exactly the schema the actual analyzer consumes, with postprocess and tensor/input hashes linked. Preserve raw FAIL and separate export/target discrepancies. Do not alter boxes to match references; scaling/clipping must be the existing declared coordinate transform. Source-reference manifest schema must also match `validate_reference_artifact`: the new producer currently writes `e2l1-source-reference-v2/complete`, while that consumer only accepts `e2l1-source-onnx-reference-v1/verified`.

Close this with **one synthetic producer -> private output -> postprocess -> validated reference/package -> target -> canonical analyzer integration test**, not independently fabricated analyzer records. Synthetic XML and external ORT/CUDA calls may be doubled; use the actual CPU postprocessing and AP implementation. The test must verify correct membership, shapes and hashes and reject wrong source or coordinates.

### C3 — target CLI bypasses the bounded child and durable evidence

`--stage target` calls `run_target_execution_stage` directly in the CLI process. That function calls load/enqueue synchronously without a deadline, holds events in memory and writes them only after completion/exception. The tested bounded `_child_main` path is still a different mock workflow. Thus a real hang can bypass the promised timeout and lose attempted/completed evidence.

Wire the actual target stage through a CPU parent and owned bounded child, reusing existing lifecycle utilities. Flush/fsync events and durable counters around real operations; preserve primary error, cleanup errors, partial outputs and unknown completion. The parent's success requires child exit plus successful terminal cleanup. Exercise this exact stage/CLI path with a hanging/failing external runtime double and verify no retries, no late success, and no leaked owned process. Do not test a separate simulator as a substitute. Do not label a mock run `real_device_execution=true` (the current target manifest sets this unconditionally).

### C4 — execution dispatch bypasses package validation and image streaming

The target CLI reads `input_records` and resolves their paths directly; it does not call `validate_package_inventory`. `stream_bound_images` exists but is unused by the real source/target chain, while target still loads precomputed tensor files. This does not implement the declared bound-image streaming workflow. The source stage also accepts arbitrary nonempty image lists rather than enforcing the 1,636 canonical inventory for production.

Wire mandatory execution-ready package/source/XML/code identity checks into dispatch before runtime imports/loads. Consume the bounded image/preprocess stream (or provide the already approved concrete bounded producer/transfer contract); do not silently switch to a full precomputed tensor bundle. Enforce source/target inventory, count, path containment and input-hash agreement at the actual execution boundary. Small synthetic fixtures remain allowed only under explicit test injection, never the production CLI mode. Ensure published execution evidence can be distinguished from synthetic evidence.

## Closing deliverable and authority

Complete C1-C4 together locally; there is no permission request per helper. Use `D:/Research/paper/local/measurement_audit_env/Scripts/python.exe` for the existing CPU dependencies, without installing or running models. Keep source/target runtime calls injected for tests. Produce one accurate executable runbook and a full-chain integration result, preserving accepted evaluator work. Continue L1A-027 with a correction and append L1A-028 for the consolidated closing packet. Do not call this merely a pending-reference wait: these are code/contract gaps before any reference execution.

No SSH, transfer, model forward, export, build, E2 inference, benchmark, retry, tolerance change or new research task is authorized here. Main lane's separate GO is not edge GO. Commit/push unchanged review/inbox and scoped changes via NADUNGVN. Astra leaves documentation unstaged.
