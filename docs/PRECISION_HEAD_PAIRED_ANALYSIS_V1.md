# Precision-head paired analysis v1

Status: `server-cpu-authorized` under A2L-016. This is a dev-only CPU replay of the accepted YOLO11n precision-head diagnostic. It does not load an engine, invoke TensorRT, run inference, rebuild anything, or benchmark a device.

## Locked inputs and estimator

- Inputs are the four arms in `results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1_attempt2/`, plus the historical FP16 capture `server_fp16_capture_v1` and verification `server_native_size_v1`.
- Each arm must have three completed repeats with exact prediction payloads. `repeat_1` is used once as the representative after exactness verification; this is deduplication, not best-build selection.
- JSON inputs are read from canonical `HEAD` Git blobs so Windows checkout line endings cannot silently change linked hashes. XML is read from the locked server path and its raw SHA256 must equal each persisted size report.
- The runner requires `numpy` and `ultralytics` to match the captured environment and `pycocotools==2.0.10`. The source XML must contain only the locked dev IDs requested by the captures.
- The estimator is `coco_xml_paired_image_bootstrap_v1`: 1,000 draws, `numpy.random.Generator(PCG64)`, seed `20260916`, sampling with replacement from the sorted 1,636 dev images. Each row is sorted for canonical image order, but duplicate occurrences are retained. The same draw is applied to FP16 and all four arms; matching is reused and COCO accumulation is rerun.
- Endpoints are `all`, `xs`, `s`, `m`, `l`, `xl` × `AP50`, `AP50-95` under `COCO_bbox_AP_custom_CCTSDB_area_XML_original_coordinates_v1`.
- Contrasts are fixed: bbox−baseline, classification−baseline, both−baseline, bbox−classification, both−bbox, both−classification, baseline−FP16, bbox−FP16, classification−FP16, both−FP16. CIs are percentile 95% intervals with valid/undefined counts; they are exploratory.
- Full Ultralytics points remain separate persisted capture values. COCO/XML CIs must not be attached to those full points.
- Localization is a GT-centric gained/lost matching diagnostic at IoU 0.50/0.75/0.90 versus `baseline_int8`, grouped by size. It is not a causal bbox/classification decomposition.

## Output contract

The destination must be new:

`results/measurement_audit_v1/precision_head_paired_analysis_v1/`

The runner writes `input_manifest.json`, `bootstrap_samples.json`, `bootstrap_draws.json`, `point_estimates.json`, `contrast_ci.json`, `localization_per_gt.json`, `localization_summary.json`, `report.md`, and `analysis_summary.json`. It refuses to overwrite an existing destination and fails on missing, changed, unlinked, incompatible, or non-canonical inputs.

The analysis state is `step_A_completed_review_required`. It does not select an arm, open a calibration experiment, start the 15-model matrix, run official-test evaluation, retrain, or benchmark.

## Foreground server CPU command

Run after pulling the Luna commit and checking the artifact/output paths. The XML path below is the locked sibling dataset path used by the existing server audits; adjust only if the server operator has the same XML under another path and the resulting SHA256 is already the persisted one.

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/analyze_precision_head_paired.py --out-dir results/measurement_audit_v1/precision_head_paired_analysis_v1 --xml /home/ubuntu/Dung_TDTU/nighttime-tsd/data/raw/CCTSDB2021/xml.zip
```

This command is foreground and CPU-only. It may run alongside an unrelated GPU job because it does not open CUDA or TensorRT; it still must not share or mutate the input artifact directories. If it fails, preserve the error and any partial output for review; do not resume or overwrite the destination.
