# CPU reproduction and evidence-audit recipe

## Source-reference handoff

The one approved full source ONNX CPU stage is user-operated on SERVER-01. Use the complete foreground command in [`docs/E2L1-030_SOURCE_CPU_RUNBOOK.md`](../../docs/E2L1-030_SOURCE_CPU_RUNBOOK.md) unchanged. It pins code revision `988bd6f715df99da6998914404ce507204e10790`, ONNX SHA-256 `bd20b36d640c502358c44edbbde51f05267eda2518b0c0a4cfe84ca18a00d4b7`, all eleven helper hashes, canonical dev image IDs/bytes/shapes, CUDA-hidden ONNX Runtime `CPUExecutionProvider`, a fresh absent output root, one POSIX mock test and at most 1,636 source ONNX forwards with zero native forwards. Luna-EDGE does not SSH to SERVER-01.

Return only sanitized state/counters, actual provider/version, POSIX test summary with zero skips, and hashes/sizes for the approved manifests, index/events and stdout/stderr. Preserve a failed/partial stage and never retry or resume it. The current full-dev source count is still `0/1636`.

## CPU-only postprocess and paired analysis after both sides exist

Run from the reviewed repository in the existing dependency-complete CPU environment. These stages must read only saved outputs; they do not call a model or device. Replace the uppercase paths with the audited private roots and use fresh, absent output directories.

```bash
python scripts/edge_readiness/e2_dev_evaluation_packet.py \
  --stage postprocess \
  --raw-manifest SOURCE_ROOT/manifest.json \
  --raw-root SOURCE_ROOT \
  --image-records PACKAGE_ROOT/image_records.json \
  --source-label onnx_cpu \
  --out-dir SOURCE_ROOT/postprocess

python scripts/edge_readiness/e2_dev_evaluation_packet.py \
  --stage postprocess \
  --raw-manifest TARGET_ROOT/manifest.json \
  --raw-root TARGET_ROOT \
  --image-records PACKAGE_ROOT/image_records.json \
  --source-label e2_fp16 \
  --out-dir TARGET_ROOT/postprocess

python scripts/edge_readiness/e2_dev_evaluation_packet.py \
  --stage analyze \
  --source-records SOURCE_ROOT/postprocess/records.json \
  --target-records TARGET_ROOT/postprocess/records.json \
  --xml CANONICAL_DEV_XML \
  --out-dir ANALYSIS_ROOT
```

The analyzer delegates COCO/XML metrics and paired resampling to the locked repository evaluator (1,000 paired resamples, seed `20260916`). Do not write an alternate AP implementation or use the smoke fixtures to fill the full-dev tables.

## Boundary checks and data policy

Before CPU analysis, verify the real source-manifest-to-private-package binding, exact canonical order, source ONNX identity/provider and before/after hash, actual image/input byte hashes, output tensor count/shape/dtype, the private allowlist and cleanup/terminal status. Before any already-authorized E2 target stage, verify the same input bytes per image, current E2/runtime/engine identity and fresh resource gate; target calls remain capped at 1,636 with no build, warmup, smoke, retry/resume or benchmark.

Public Git allowlist: sanitized manifests/indexes, source/runtime identities, counters, hashes, diagnostic tables, paired metrics, tests and this recipe. Private only: checkpoint/ONNX/engine, images and labels/XML, preprocessed input tensors, raw source/target tensors, full unredacted process logs and private transfer archives. Do not put these bytes in Git or public HF. Any private transfer must follow the already approved exact allowlist/hash workflow after source artifacts arrive.

The dependency-complete local environment at `D:/Research/paper/local/measurement_audit_env/Scripts/python.exe` was used without installing packages (Python 3.11.9, NumPy 2.4.2, Pillow 12.3.0, pycocotools 2.0.10, ONNX Runtime 1.24.3, Ultralytics 8.4.102, Torch 2.8.0+cu129, OpenCV import available). The complete `test_e2_dev_evaluation_packet.py` CPU fixture/evaluator suite ran 32 tests: 31 passed and one POSIX-only child-termination test skipped on Windows. Both previously skipped pinned-helper boundaries—saved-output postprocess/NMS and Ultralytics letterbox/box-scaling equivalence—passed. `test_e2_output_diagnostic.py` passed 8/8. These tests use synthetic artifacts or runtime doubles; no model forward was run. The separate E5 contract tests passed 15/15 with CPU doubles. The Linux POSIX prerequisite remains user-operated and is not satisfied by the Windows run. Full-dev source output remains pending; see L1A-032 for exact counts and limits.
