# ST-EDGE-03 prospective full-dev E2 packet (R1 implementation)

Status: `prepared_not_executed`; terminal status: `edge_dev_packet_implementation_review_required`.

The local implementation validates every frozen identity/runtime field, checks actual package files, uses the existing Jetson provider/owner/adapter boundary through an external double, persists child events, bounds timeout and retains partial counters, and computes synthetic CPU AP50/AP50-95 plus paired duplicate-image bootstrap CI. It performs no model, CUDA, TensorRT, SSH, transfer, build or inference.

## Future operator chain

1. Server verifies the canonical 1636-image/2706-instance dev inventory, XML/image membership, source ONNX `bd20b36d640c502358c44edbbde51f05267eda2518b0c0a4cfe84ca18a00d4b7`, and reference prediction hash `5c23d0fa7d4bbf09858b2f1a4dbf45c35650c474aaedebe40d845e3c9acc410a`.
2. Server publishes a private manifest with relative paths, bytes and SHA256; Luna1 runs `validate_package_inventory(package_root=...)`. Wrong engine/source/XML/hash/size/path/symlink/duplicate is a hard stop. Inputs are streaming manifest-only; a full 7.49 GiB float32 tensor bundle is forbidden.
3. After a later integrated GO, the owner-side E2 preflight verifies the existing engine hash on E2. Luna1 runs one target pass per image through the existing provider/owner/adapter; warmup, shape probe, debug forward, retry, build, latency and energy are all zero.
4. Child events are append+fsync durable. A timeout terminates the child, preserves counters and marks completion unknown; no automatic retry.
5. CPU analysis then invokes `coco_xml_paired_image_bootstrap_v1` with seed 20260916 and 1000 paired image resamples for AP50/AP50-95 plus size/class diagnostics. The dependency-free evaluator here is only a synthetic workflow check, not the canonical XML evaluator.

Missing identity, membership, exact call accounting, durable evidence or engine availability is a blocker. This packet does not authorize execution now.
