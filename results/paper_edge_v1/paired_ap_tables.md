# Paired development-set AP tables — pending execution

No cells below are populated from the three-fixture smoke. Fill only after hash-auditing complete source and E2 outputs with exactly 1,636 canonical image IDs and 2,706 instances on each side.

Primary table uses the locked COCO/XML evaluator. The paired target-minus-source confidence interval uses 1,000 image-paired resamples, seed `20260916`; do not introduce a post-hoc noninferiority margin.

| Dataset / model | Images | Instances | Source ONNX CPU AP50 | E2 FP16 AP50 | Δ AP50 (E2 − source) | Paired 95% CI | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| CCTSDB2021 dev / frozen YOLO11n | 1,636 | 2,706 | pending | pending | pending | pending | waiting for source operator output, package audit, E2 prerequisites and one-pass target stage |

| COCO size stratum | Source AP | E2 AP | Δ AP (E2 − source) | Paired 95% CI | Support / validity |
| --- | ---: | ---: | ---: | --- | --- |
| Small | pending | pending | pending | pending | pending |
| Medium | pending | pending | pending | pending | pending |
| Large | pending | pending | pending | pending | pending |

Required audit before publication: each side has 1,636 unique ordered IDs, matching per-image source/target input bindings, finite outputs and valid terminal/cleanup state; the target pass has no more than 1,636 enqueues and no retries. Preserve undefined strata as JSON `null` and report support rather than substituting a metric. Raw outputs, images, labels, ONNX/checkpoint and engine bytes are private; publish sanitized counts, hashes and metrics only.
