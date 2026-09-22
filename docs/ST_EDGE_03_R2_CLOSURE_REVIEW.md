# ST-EDGE-03 R2 closure review

2026-09-22, Astra. Reviewed `9d2ddf1` on `luna1/e2l1-006-jetson-adapter-smoke`.

## Independent results and accepted corrections

Using existing `D:/Research/paper/local/measurement_audit_env/Scripts/python.exe`, without installing anything:

- Packet suite: **12 PASS, 1 ERROR**.
- Diagnostic suite: **8/8 PASS**.
- Thus these two focused suites are **20 PASS, 1 ERROR**, not 21/21 in the dependency-equipped environment.
- The actual Ultralytics 8.4.102 helper now accepts the adapter's float32 CPU `[1,7,8400]` synthetic zero output. No model was run. Preserve the tensor conversion fix.
- The SERVER TensorRT reference is now explicitly distinguished from the pending source-ONNX CPU reference. Accept this provenance correction; source forwards remain **0/1636**.
- The canonical AP/resampling implementation is reused rather than replaced for the proposed real analysis. With only XML loading substituted by a synthetic annotation mapping, the actual COCO/resampling path produced the expected zero paired difference for identical synthetic detections. This is a narrow unit check, not a successful end-to-end evaluator run.

No SSH, transfer, frozen-model forward, export, build, inference or benchmark was performed.

## Concrete remaining work

### 1. Fix and test the actual canonical evaluator boundary

`evaluate_canonical_coco_xml_pair` calls `load_xml(xml_path, source_records)` with a list of dictionaries. The accepted loader iterates image names from a name-keyed mapping. The dependency-equipped test fails at this call with:

`TypeError: expected str, bytes or os.PathLike object, not dict`

Pass the accepted name-keyed record contract, first rejecting duplicate image identities and incompatible source/target shapes. Test using real synthetic XML content plus the actual pycocotools evaluator and resampling, not an expected missing-dependency exception. Cover empty/all-wrong/perfect predictions, class/size support and identical versus changed paired outputs. Preserve absent-support/undefined-resample semantics; serialize undefined values as explicit null, not nonstandard JSON NaN. No model calls are needed. Run in the existing dependency-rich CPU environment above. Retain an independently injected missing-dependency test, but do not make the successful path depend on packages being absent.

### 2. Implement the execution chain; metadata and an adapter mock are not the real factory path

Current CLI actions only write metadata or validate it. There is no command to produce CPU ONNX references, run the existing E2 engine, or perform final paired analysis. `make_lazy_e2_provider` constructs a provider but is not used by a production execution path. `_child_main` instantiates `AdapterRuntimeDouble` or reads supplied output bytes; it does not use the real provider/owner chain. Therefore this is not merely a reference-generation wait: implementation remains incomplete.

**Implementing these real paths locally is authorized. Executing them on models/devices is not.** Reuse the accepted source preprocessing/provider/owner/adapter components and lifecycle; do not write a new runtime framework. Supply executable stage commands for source preparation, package validation, target child/parent and analysis. Each stage must have explicit permission/mode checks and fresh outputs. Keep real calls lazy and test normal orchestration with only external calls doubled. Model/mock selection must not replace the entire pipeline or generate successful counters without invoking operations. Preserve bounded timeouts, counters, partial artifacts and cleanup errors, including failures after an apparent successful computation.

The parent must never return success before the child's cleanup result is known. Currently `_child_main` writes `child_result.json` and `child_completed` before its `finally` cleanup; `run_durable_child_study` accepts that result even if cleanup later emits `cleanup_failed`. Fix this shared lifecycle before using it for real execution; the mock adapter's `close` also only appends an event rather than exercising resource release.

### 3. Finish the actual source/package binding

The manifest-only streaming declaration must have a concrete producer/consumer: bound images/preprocessing -> one input tensor -> CPU source and E2 target contract. Do not require a full precomputed tensor bundle while advertising image-streaming. Keep source call budget explicit and pending until authorized execution.

On an archive/device without the repo's `validator_predictions.json`, the validator currently accepts a caller-supplied canonical image digest without recomputing it from the supplied IDs. Compute the digest using the declared canonical serialization, or package and hash-check the canonical inventory. Tie the verified source-reference and XML fields to required, actually hashed allowed files, not merely arbitrary 64-character strings. Distinguish metadata-pending validation from execution-ready validation. Add positive producer-shaped and negative changed/missing tests.

## Closing handoff

Complete the entire remaining local package autonomously, not a pending-reference document alone. Update L1A-026 with corrected test results, then a consolidated L1A-027 closing report listing concrete implemented CLI stages, CPU integration tests and execution prerequisites. Use the existing environment for CPU postprocessing/evaluation; do not install dependencies or run frozen models. Preserve old raw FAILs, the accepted diagnosis, engine/source identities, budgets and all historical outputs. No tolerance relaxation, box correction, additional build, retry or benchmark.

**GO local code/CPU tests; NO-GO source model calls and E2 execution pending integrated review.** Main Luna proceeds independently. Astra leaves this document/inbox unstaged; Luna1 commits/pushes scoped work through NADUNGVN.
