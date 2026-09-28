# E2L1-028 C1-C4 implementation review

Astra, 2026-09-28. Reviewed `d41ac7f89630c77aaf7b66e89742df3917e35bca`. Independently ran packet tests **24/24 PASS** and diagnostic tests **8/8 PASS**. Wider 108-test regression remains Luna1-reported. No SSH, model forward, export, build or device execution in this review.

Accept the move from native forward to pinned ONNX/ORT, the actual source/target/postprocess/analyzer connection on synthetic fixtures, and use of a target child with durable events. Preserve those improvements and the repaired canonical evaluator. **NO-GO execution remains**, for the concrete real-input/runtime defects below. Fix as one local package, not a new research design.

## P1 — original-image box transform is wrong for accepted non-square metadata

`_letterbox_to_original` accepts/defaults `resized_shape=[640,640]` and computes padding as `(640-resized_shape)/2`. But the existing source preprocess reports the padded letterbox result as 640x640, not the unpadded resized image. Thus padding is lost for non-square originals. Current tests only use square 640x640 originals.

Independent CPU reproduction: original shape `[480,640]`, supplied/default resized shape `[640,640]`, predicted letterbox box `[64,192,192,320]`. Current function returns `[64,192,192,320]`; the pinned Ultralytics `scale_boxes((640,640), ..., (480,640))` returns `[64,112,192,240]`. The 80-pixel y offset materially changes AP.

Use the accepted transform with actual preprocessing gain/padding/shape metadata, including rounding, rather than inventing a second box transform or treating padded shape as unpadded. Bind this metadata to the input producer. Add landscape, portrait, odd-size rounding and clipping cases against the real helper plus a full-chain non-square synthetic image test. Do not relax numerical tolerances or correct boxes toward references.

## P2 — real target image preprocessing invokes a source-only preflight

`ImageStreamSource.iter_inputs` calls `UltralyticsSourceRuntime.prepare()` on E2 after the target engine is loaded. That method requires the complete source/export environment (including ONNX Runtime and onnxslim) and explicitly raises `CPU_PROVIDER_REQUIRED` when `torch.cuda.is_available()`. It is incompatible with the intended CUDA-capable target process; synthetic_inputs bypasses it in tests.

Separate the pure CPU image preprocessing dependency boundary from source model/export/ORT preflight. Reuse the same actual resize/color/normalization operations and bind their resulting bytes, without requiring the entire server environment on E2 or hiding CUDA from its TensorRT runtime. Do not install packages or change device configuration. Use existing recorded E2 dependency evidence; report a specific missing prerequisite if needed. Test the non-synthetic image-loader/preprocess path with CUDA reported available and a small real decoded synthetic image, without any model/device calls. Verify target input hashes against the source producer's exact input hashes; current source/target output rows do not establish those links.

## P3 — deadlines and terminal evidence disagree with the declared contract

The production target CLI hardcodes `timeout_seconds=180.0` for the entire 1636-image child, while the packet declares stage timeout 54000 seconds and per-image timeout 30 seconds. No matching per-image watchdog exists. Use the already declared bounded deadline semantics consistently in the actual parent, driven by the frozen execution contract; do not silently borrow the old three-image smoke deadline or change numerical/call budgets.

After terminate/join(2), the parent does not check whether the child is still alive or escalate owned-process termination. Add bounded termination/reaping and record actual outcomes. Validate child exit code and terminal cleanup before accepting a complete result. On timeout, reconstruct available last counters from durable events, preserving unknown completion and partial inventory instead of publishing an empty attempted/completed object. Test a real child that ignores graceful termination with external-runtime doubles where supported, plus abnormal exit after a result file.

## P4 — bind actual source scope and input identity; correct the runbook

Source dispatch still accepts arbitrary nonempty image lists. For production, require exact canonical 1636 IDs/order and bound image/shape metadata before any ORT calls; small fixtures must be explicitly test-only. Record actual session providers, input tensor hashes/preprocessing metadata and ONNX before/after hashes (currently only before is computed). Connect package reference evidence to the produced manifest and detections rather than trusting `verified` strings and unrelated allowed hashes. These are existing provenance requirements, not new data collection.

The generated runbook incorrectly says the next permitted execution is an already authorized single E2 smoke. **That smoke authorization was consumed historically. There is no unused smoke permission.** The current candidate is a prospective 1636-image dev comparison using the existing engine, with zero builds, separately counted CPU source calls and no retry/benchmark. Remove stale smoke/revalidation wording from generator and new packet. Source and E2 model calls remain NO-GO until integrated review; main server GO does not apply to edge.

## Handoff

GO local implementation/CPU tests and saved-artifact inspection only. Complete P1-P4 together with tests through the normal real-image stage, actual helper/evaluator and external runtime doubles. Use the existing dependency-rich CPU environment; no installation, SSH, transfer, forward, build, inference or benchmark. Return one consolidated L1A-029 with corrections to the prior closure claim, observed results and executable runbook. Preserve accepted fixes, historical raw FAILs, private binaries and existing attempts. Commit/push this unchanged review/inbox with scoped work through NADUNGVN. Astra leaves documentation unstaged.
