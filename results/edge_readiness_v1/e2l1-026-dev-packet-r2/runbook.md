# ST-EDGE-03 prospective full-dev E2 packet (R2 implementation)

Status: `prepared_not_executed`; terminal status: `edge_dev_packet_implementation_review_required`.

## Local CPU-only commands

`python scripts/edge_readiness/e2_dev_evaluation_packet.py --write-source-reference-pending --out-dir results/edge_readiness_v1/e2l1-026-source-reference-pending/source_reference.json`

This records source-ONNX CPU reference generation as pending and executes zero model forwards. A future operator must separately run the source CPU producer, publish its output hash, and update the manifest; the SERVER TensorRT FP16 hash `5c23d0fa7d4bbf09858b2f1a4dbf45c35650c474aaedebe40d845e3c9acc410a` is retained only as a distinct server reference.

`python scripts/edge_readiness/e2_dev_evaluation_packet.py --validate-contract`

`python scripts/edge_readiness/e2_dev_evaluation_packet.py --validate-package --inventory PRIVATE/package_manifest.json --package-root PRIVATE/package`

The package validator binds the exact canonical image-ID digest, producer-shaped streaming input records, source/runtime identities, XML identity, and actual allowlisted regular files. Input tensors are consumed one at a time; the full 7.49 GiB bundle is not materialized.

## Future integrated execution chain (not authorized now)

1. Server creates the source ONNX CPU reference separately from the existing SERVER TensorRT FP16 reference and counts all 1636 source passes. Missing source reference output remains a blocker.
2. After integrated GO, E2 verifies the existing engine hash on-device and the parent launches one bounded child. The child calls the existing `TensorRTProvider`/`CudaRuntimeMemoryOwner`/`JetsonRuntimeAdapter` path; no warmup, probe, debug forward, retry, build, latency or energy collection is permitted. `AdapterRuntimeDouble` tests the same adapter lifecycle without device calls.
3. Events are append+flush+fsync durable. Timeout terminates the child, preserves partial counters and marks completion unknown. Cleanup is recorded separately.
4. CPU postprocess uses the actual lazy `ultralytics.utils.nms.non_max_suppression` binding with a copied float32 tensor of shape `[1,7,8400]`; input bytes are checked unchanged and detections are recorded.
5. Once source and target records exist, `evaluate_canonical_coco_xml_pair` delegates to the accepted `coco_xml_paired_image_bootstrap_v1` implementation (`pycocotools==2.0.10`, PCG64 seed `20260916`, 1000 sorted paired image draws, duplicates retained) and reports source, target and target-minus-source AP50/AP50-95 for all/xs/s/m/l/xl. No custom AP result is substituted.

No SSH, transfer, build, inference or benchmark is authorized by this packet. Historical raw FAILs and accepted saved-output diagnostics remain unchanged.
