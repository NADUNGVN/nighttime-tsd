# Draft manuscript section — edge execution and numerical limitations

## Edge execution and output agreement

We evaluated the frozen batch-1, 640×640 YOLO11n path on a Jetson Xavier NX using TensorRT 8.5.2.2 and an FP16 engine configured with a 1-GiB workspace. The first instrumented build attempt reached its 900-s deadline and produced no engine. A separately counted second attempt parsed the ONNX, built and loaded one engine, and completed three ordered fixture inferences with synchronized output copies. The second build and inference stages took approximately 16m35s and 13.3s, respectively; these are workflow wall times, not latency measurements. No additional build or inference was performed after that attempt.

We compared source-native, source-ONNX and TensorRT raw outputs before decode or non-maximum suppression (NMS), keeping source-export disagreement distinct from target-runtime disagreement. Source-native versus source-ONNX met the frozen strict criterion on two fixtures and retained one box-element mismatch on the third. TensorRT versus source-ONNX exceeded the same frozen box criterion on all three fixtures, with 965, 1,226 and 564 box-element violations for fixtures `00006`, `00009` and `00028`; score-domain violations were zero under the predeclared absolute-plus-relative tolerance. Zero score violations indicate tolerance compliance, not bitwise score equality.

CPU replay with the accepted postprocessing helper retained one, two and two same-origin detections for the respective fixtures in both source-ONNX and TensorRT outputs. The five matched rows had maximum score differences from 0.0000244 to 0.0008014, maximum coordinate differences from 0.07977 to 0.26363 input pixels, and pairwise box IoUs from 0.97939 to 0.99468. Thus, this saved-output analysis localizes the observed discrepancy to box coordinates without showing a changed retained-detection count in these fixtures.

| Fixture | Source native → ONNX | TRT → ONNX raw box / score violations | NMS detections, ONNX → TRT | Max score Δ | Max box Δ (input px) | Matched-box IoU range |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `00006` | pass | 965 / 0 | 1 → 1 | 0.0008014 | 0.09718 | 0.99244–0.99244 |
| `00009` | pass | 1,226 / 0 | 2 → 2 | 0.0007682 | 0.21231 | 0.97939–0.98735 |
| `00028` | fail (one box element) | 564 / 0 | 2 → 2 | 0.0000894 | 0.26363 | 0.99344–0.99468 |

These three fixtures are diagnostic-only and do not provide ground-truth AP, recall, a deployment acceptance result, or evidence that the observed coordinate differences do or do not affect accuracy. We therefore retain the raw numerical FAIL and make no accuracy-impact conclusion from tolerance compliance or post-NMS count agreement.

## Full development-set analysis status

The predeclared full-development comparison uses 1,636 ordered images and 2,706 instances. At this checkpoint, neither side has produced a full-dev prediction set (`0/1,636` source; `0/1,636` E2 target). The user-operated source ONNX CPU reference remains pending, and E2 was reported offline. Accordingly, all paired COCO/XML AP values and confidence intervals remain pending; no latency or energy benchmark was collected. The complete one-pass, no-rebuild/no-retry conditions remain those of E2L1-030.

This section is an evidence-constrained draft for review, not a claim of deployable edge performance. The separate alternative-device inventory is readiness evidence only and does not replace the E2 comparison.
