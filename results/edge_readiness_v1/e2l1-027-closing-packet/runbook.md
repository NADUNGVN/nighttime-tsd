# E2L1-027 ST-EDGE-03 closing packet

Status: `prepared_not_executed`; source CPU reference is still `0/1636` and no
E2 forward is authorized by this packet.

## Guarded stage commands

Source CPU reference (only in a separately approved source-forward session):

`python scripts/edge_readiness/e2_dev_evaluation_packet.py --stage source --source-root SOURCE_ROOT --checkpoint SOURCE_ROOT/results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt --image-records PRIVATE/image_records.json --out-dir PRIVATE/source-reference --allow-source-forward --commit CODE_COMMIT`

The stage reuses `e2_source_bundle.UltralyticsSourceRuntime`, consumes the
bound image records in order, writes raw output only below `private/`, and
publishes hashes/counters. Without `--allow-source-forward` it fails closed.

Target E2 execution (only after a later integrated GO):

`python scripts/edge_readiness/e2_dev_evaluation_packet.py --stage target --engine PRIVATE/engine.plan --input-records PRIVATE/package_manifest.json --package-root PRIVATE/package --out-dir PRIVATE/target-reference --allow-target-inference`

The target stage reuses `E2TensorRTRuntime`, `TensorRTProvider`,
`CudaRuntimeMemoryOwner`, `OwnedBuffers` and `JetsonRuntimeAdapter`; it does
not build, warm up, retry or benchmark. Without `--allow-target-inference` it
fails closed. Parent cleanup is known before a child result is accepted.

CPU-only terminal analysis:

`python scripts/edge_readiness/e2_dev_evaluation_packet.py --stage analyze --source-records PRIVATE/source_records.json --target-records PRIVATE/target_records.json --xml PRIVATE/xml.zip --out-dir results/edge_readiness_v1/e2l1-027-canonical-analysis`

Analysis normalizes the accepted name-keyed XML loader contract, rejects
duplicate IDs/shape mismatches, delegates COCO/XML AP and PCG64 paired
resampling to the locked repository evaluator, and serializes undefined
support as `null`. It never loads a model or invokes a device runtime.

No SSH, transfer, source forward, build, E2 inference, retry or benchmark was
performed while producing this packet. Historical FAILs and private raw bytes
remain unchanged.
