# YOLO11n precision-head latency study v1

Status: `latency_completed_review_required`.

- Study: `yolo11n_precision_head_latency_v1`; device identity: `Quadro RTX 8000` / `GPU-9850d121-55dc-e752-ffaa-df19e7585eb4`
- Sessions: 39/39; each session has 200 warmup and 1000 measured calls.
- Timing is synchronous batch-1 `model.predict` wall time from decoded CPU image through preprocessing/H2D/inference/postprocessing/NMS. Disk decode, model load, initial allocation and warmup are excluded; this is not pure TensorRT kernel time.
- Accuracy references are COCO/XML points and fixed paired CIs from the accepted analysis; no CI is attached to a latency point and no fastest engine is selected.

## Per-engine latency

| Engine | Arm/build | Mean ms (3,000 calls) | P50 | P95 | P99 | Serial FPS | Accuracy point/CI reference |
|---|---|---:|---:|---:|---:|---:|---|
| fp16 | fp16/reference | 3.5917 | 3.4418 | 4.2922 | 4.8098 | 278.4224 | point `point_estimates.json#fp16`; CI refs `baseline_minus_fp16, bbox_minus_fp16, classification_minus_fp16, both_minus_fp16` |
| baseline_int8_1 | baseline_int8/1 | 3.5981 | 3.5056 | 3.9626 | 5.6790 | 277.9242 | point `point_estimates.json#baseline_int8`; CI refs `bbox_minus_baseline, classification_minus_baseline, both_minus_baseline, baseline_minus_fp16` |
| bbox_fp32_1 | bbox_fp32/1 | 3.6993 | 3.5925 | 4.3197 | 4.6613 | 270.3213 | point `point_estimates.json#bbox_fp32`; CI refs `bbox_minus_baseline, bbox_minus_classification, both_minus_bbox, bbox_minus_fp16` |
| classification_fp32_1 | classification_fp32/1 | 3.6594 | 3.5821 | 4.0500 | 4.5344 | 273.2675 | point `point_estimates.json#classification_fp32`; CI refs `classification_minus_baseline, bbox_minus_classification, both_minus_classification, classification_minus_fp16` |
| both_fp32_1 | both_fp32/1 | 3.6160 | 3.5527 | 3.9391 | 4.3524 | 276.5487 | point `point_estimates.json#both_fp32`; CI refs `both_minus_baseline, both_minus_bbox, both_minus_classification, both_minus_fp16` |
| baseline_int8_2 | baseline_int8/2 | 3.4853 | 3.4338 | 3.8046 | 3.9002 | 286.9187 | point `point_estimates.json#baseline_int8`; CI refs `bbox_minus_baseline, classification_minus_baseline, both_minus_baseline, baseline_minus_fp16` |
| bbox_fp32_2 | bbox_fp32/2 | 3.5827 | 3.5203 | 3.9234 | 4.0139 | 279.1206 | point `point_estimates.json#bbox_fp32`; CI refs `bbox_minus_baseline, bbox_minus_classification, both_minus_bbox, bbox_minus_fp16` |
| classification_fp32_2 | classification_fp32/2 | 3.5471 | 3.4982 | 3.8786 | 3.9786 | 281.9227 | point `point_estimates.json#classification_fp32`; CI refs `classification_minus_baseline, bbox_minus_classification, both_minus_classification, classification_minus_fp16` |
| both_fp32_2 | both_fp32/2 | 3.6816 | 3.6252 | 4.0003 | 4.2633 | 271.6187 | point `point_estimates.json#both_fp32`; CI refs `both_minus_baseline, both_minus_bbox, both_minus_classification, both_minus_fp16` |
| baseline_int8_3 | baseline_int8/3 | 3.5143 | 3.4590 | 3.8571 | 3.9622 | 284.5492 | point `point_estimates.json#baseline_int8`; CI refs `bbox_minus_baseline, classification_minus_baseline, both_minus_baseline, baseline_minus_fp16` |
| bbox_fp32_3 | bbox_fp32/3 | 3.6041 | 3.5526 | 3.9233 | 4.2082 | 277.4634 | point `point_estimates.json#bbox_fp32`; CI refs `bbox_minus_baseline, bbox_minus_classification, both_minus_bbox, bbox_minus_fp16` |
| classification_fp32_3 | classification_fp32/3 | 3.5681 | 3.5188 | 3.8806 | 4.0380 | 280.2612 | point `point_estimates.json#classification_fp32`; CI refs `classification_minus_baseline, bbox_minus_classification, both_minus_classification, classification_minus_fp16` |
| both_fp32_3 | both_fp32/3 | 3.6690 | 3.5958 | 4.0029 | 4.1623 | 272.5513 | point `point_estimates.json#both_fp32`; CI refs `both_minus_baseline, both_minus_bbox, both_minus_classification, both_minus_fp16` |

## Arm summaries

Arm rows pool all three builds and all three rounds; build-level and round-level rows remain in `latency_summary.json` and each session JSON.

| Arm | Builds | Sessions | Pooled calls | Mean ms | P95 ms | Serial FPS |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 1 | 3 | 3000 | 3.5917 | 4.2922 | 278.4224 |
| baseline_int8 | 3 | 9 | 9000 | 3.5326 | 3.8718 | 283.0792 |
| bbox_fp32 | 3 | 9 | 9000 | 3.6287 | 3.9921 | 275.5818 |
| classification_fp32 | 3 | 9 | 9000 | 3.5915 | 3.9297 | 278.4329 |
| both_fp32 | 3 | 9 | 9000 | 3.6556 | 3.9817 | 273.5562 |

## Limitations

This is a single shared RTX8000/server diagnostic with sampled GPU telemetry and a finite fixed image pool. The three rounds reduce position confounding but are not a complete Latin square; results remain conditional on the fixed environment, runtime, engine bytes, image pool and postprocessing options. Desktop processes may remain visible under the operator-confirmed exception. Power/energy are not endpoints here, and no production-confidence benchmark, cross-device claim, engine rebuild, accuracy selection, official-test evaluation, retraining or 15-model expansion is performed.
