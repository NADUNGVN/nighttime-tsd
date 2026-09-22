# E2L1-024 saved-output audit addendum

Status: `output_diagnostic_completed_review_required`.

CPU-only replay of the saved target tensors used raw output0 comparison and fixed postprocess conf=0.001, IoU=0.7, max_det=300. The raw target FAIL is in box elements; score values are assessed by tolerance, not exact equality. This report does not claim accuracy impact, AP, recall, safety or deployment readiness.

The row-level maximum box delta and IoU minimum are separate extrema; they are not asserted to belong to the same detection.

| Fixture | Source export | TRT box/score mismatches | Selected anchors / box mismatches | NMS source -> target | Score Δmax | Box Δmax | IoU min–max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `00006` | pass | 965/0 | 10/0 | 1 -> 1 / 10 -> 10 | 0.000801444 | 0.0971832 | 0.99244–0.99244 |
| `00009` | pass | 1226/0 | 16/5 | 2 -> 2 / 16 -> 16 | 0.000768185 | 0.212311 | 0.979394–0.987351 |
| `00028` | fail | 564/0 | 20/0 | 2 -> 2 / 20 -> 20 | 8.9407e-05 | 0.263626 | 0.993441–0.994679 |

All retained same-origin rows (source ONNX -> TRT) are:

| Fixture | Anchor | Class | Score abs delta | Box max abs delta | Box IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| `00006` | 3369 | 1 | 0.000801444054 | 0.0971832275 | 0.992439788 |
| `00009` | 2619 | 0 | 0.000170230865 | 0.0797729492 | 0.987351184 |
| `00009` | 2662 | 1 | 0.000768184662 | 0.212310791 | 0.97939408 |
| `00028` | 6980 | 0 | 8.94069672e-05 | 0.263626099 | 0.994678588 |
| `00028` | 7300 | 2 | 2.44379044e-05 | 0.200805664 | 0.993440814 |

Observed: the saved bytes replay the accepted raw comparison and permit deterministic CPU postprocess diagnostics. The score domain has zero policy violations in all three fixtures, and selected-anchor threshold membership is unchanged at 0.001 and 0.25; this is not exact-score equality. NMS counts and same-origin lineages are unchanged, while the retained box coordinates differ. Unknown: whether those coordinate changes alter final task detections or ground-truth accuracy under a valid evaluation protocol. Proposed next experiment: after scientific review, run one separately authorized FP16-reference/target postprocess comparison on a predeclared evaluation slice; do not relax the raw comparator or infer accuracy from this train-only trio.
