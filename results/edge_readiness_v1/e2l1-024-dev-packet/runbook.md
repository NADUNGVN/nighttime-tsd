# ST-EDGE-03 prospective full-dev E2 packet

Status: `prepared_not_executed`; current terminal status: `edge_dev_packet_implementation_review_required`.

## Later operator chain (not executable under E2L1-024)

1. Server operator verifies the canonical dev inventory at commit `3154b7ad2308ff8802ea1c15532951c86ed66a96`, exactly 1636 images/2706 instances, XML and image membership, and the bound source ONNX `bd20b36d640c502358c44edbbde51f05267eda2518b0c0a4cfe84ca18a00d4b7`. If the existing reference predictions do not bind to this exact ONNX, preprocessing, postprocess and dataset, materialize a separately counted CPU reference capture; do not substitute another ONNX.
2. Server operator creates a private manifest containing source/input/reference hashes, the dev YAML hash, evaluator/helper versions and one-to-one image mapping. Public handoff contains hashes/metadata only; no engine, ONNX, input tensors or raw predictions.
3. Luna1 validates the package in a fresh local root with `validate_package_inventory`, confirms no local engine load, and waits for a later integrated GO.
4. After that GO only, Luna1 transfers the approved private package to a fresh E2 root, verifies `ENGINE_SHA256` and `SOURCE_ONNX_SHA256` on E2, and runs exactly one target pass per image (1636 calls) through the existing E2 engine. No warmup, shape probe, debug forward, retry, rebuild, latency or energy collection.
5. Luna1 writes durable per-image counters/events and hashes, then invokes the accepted `coco_xml_paired_image_bootstrap_v1` evaluator on CPU with conf=0.001, IoU=0.7, max_det=300 and seed 20260916/1000 paired image resamples.
6. A reviewer audits source/target identities, call counts, membership, postprocess ownership, AP50/AP50-95 plus XS/S and per-class diagnostics. No noninferiority margin or deployment claim is invented.

## Ownership and stop conditions

The server owns reference/source materialization; Luna1 owns E2 execution and evidence audit; Astra owns the later GO and review. Missing source identity, dataset/XML membership, target engine hash, fresh root, or full call accounting is a blocker. Timeout/failure stops the run with partial evidence and no automatic retry. Existing engine unavailability is a new decision, not build permission.

This packet is an accuracy study design, not a benchmark. It does not authorize SSH, transfer, source forward, export, build, inference or benchmark now.
