# A2L-052 recovery review: one production binding defect

Reviewed 2026-09-29 at `0b030bd4df4e879e583469cea9ccea468b57fe04`.

## Accepted evidence

Astra independently ran the calibration-order tests: 5/5 PASS with CUDA_VISIBLE_DEVICES=-1; confirmation super-suite: 24/24 PASS; TensorRT-feasibility mock regression: 30/30 PASS. The first loader-test invocation omitted CUDA_VISIBLE_DEVICES and correctly failed its explicit environment assertion; the corrected CPU-only invocation passed. No GPU, model forward or TensorRT execution was performed.

All three committed selection receipt hashes match their canonical Git blobs. Each has 1024 unique IDs, sequential observed indices, matching recomputed ID digest and recorded EOF. Accept these as local real-image loader audit evidence, with the disclosed temporary Windows materialization, label CRLF normalization and different local Torch/NumPy runtime. They do not establish server tensor-byte equality or successful server execution. Astra checked receipts, not an independent full-image rerun. Preserve them; do not repeat this expensive audit solely because of the binding defect below.

Accept the deterministic manifest-indexed producer, unchanged numerical selection/preprocessing intent, prospective v2/85-total-attempt accounting, and improved callback error evidence. Do not reopen those accepted pieces.

## Blocking defect — real server dependency injection

`load_production_dependencies()` in `scripts/run_precision_head_confirmation_server.py` assigns `build_calibration_loader` directly to `build_manifest_ordered_calibration_dataloader`. That function requires three keyword-only dependencies: `check_det_dataset`, `build_yolo_dataset`, and `build_dataloader`.

The call in `build_real()` passes only exporter, ordered_image_ids, expected_image_paths and expected_count. Therefore it raises TypeError before entering the loader. The CPU audit passes all three dependencies explicitly; it is a different call boundary. The super-suite substitutes a reduced-signature fake loader and consequently misses this defect even though every test passes.

Astra reproduced this without any device import using `inspect.signature(real_loader).bind(exporter, ordered_image_ids=[], expected_image_paths=[], expected_count=1024)`: `TypeError: missing a required argument: 'check_det_dataset'`. Python will require the other two dependency parameters as well. This is a deterministic production failure, not a conjecture about GPU behavior.

## Required closure, as one bounded local package

1. Bind the three actual pinned exporter helpers in the production dependency adapter (for example an explicit wrapper or functools.partial), or pass them explicitly from the real dependency mapping at the call site. Keep loader implementation and numerical behavior unchanged. Do not remove required dependency checks or change the installed library.
2. Add regression coverage for the actual production callable construction and invocation. Keep the real shared loader in the test and replace only external dataset/device/library operations as necessary. Include a signature-contract test that fails on the current implementation and a normal auxiliary child path that reaches the real loader for both model routes. Do not replace this seam with the same permissive fake that caused the gap. This needs no TensorRT installation or GPU import; separate the CPU adapter factory if useful.
3. Run loader, super and feasibility regressions. Correct the coverage claim: prior full-order receipts prove the shared producer, not its server injection. No new full-image audit is required if producer/preprocessing code remains byte-unchanged; rebind its evidence to the fix and record that fact.
4. Append one L2A-053 recovery closure, preserve v1, unchanged scientific design, v2 root and 85 cumulative attempted-builder cap, and unrelated dirty checkpoint. Commit/push this unchanged review and inbox with the scoped fix through NADUNGVN.

**GO local closure/tests; NO-GO GPU dispatch until the corrected binding is reviewed.** No new experiment, canary, source export, cache build, numerical tolerance or budget change. Return the concrete fix/tests once, not a new plan for each helper. Edge source CPU work remains independently authorized.
