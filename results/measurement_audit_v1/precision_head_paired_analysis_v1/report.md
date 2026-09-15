# Precision-head paired dev analysis v1

Status: `step_A_completed_review_required` (CPU replay; no model inference).

- Artifact input commit: `839acdcb6a09569d1e6e130aa523c38d960dabd5`
- Analysis code commit: `8b5a3420f1eeb460b58198647cbdbe83f1ab1045`
- Dataset: `CCTSDB2021/dev`, 1636 images, 2706 GT instances
- Bootstrap: 1000 paired image draws, PCG64 seed `20260916`; duplicate occurrences retained
- Full Ultralytics values below are persisted capture points; confidence intervals apply only to COCO/XML endpoints.

## Fixed contrasts (percentage points)

| Contrast | all map50_95 | xs map50 | xs map50_95 | s map50 | s map50_95 |
|---|---:|---:|---:|---:|---:|
| bbox_minus_baseline | 5.4263 [4.9118, 5.8991] | 6.7094 [2.1525, 11.2544] | 7.3861 [4.9635, 10.1734] | 0.5272 [-0.0354, 1.1975] | 6.7072 [5.6116, 7.7726] |
| classification_minus_baseline | 3.6271 [2.9389, 4.1490] | 2.8362 [-0.3497, 5.3691] | 1.3778 [0.1425, 2.5680] | 1.7645 [0.7447, 2.8701] | 1.7893 [0.7073, 2.8024] |
| both_minus_baseline | 8.5067 [7.7104, 8.9999] | 8.9983 [3.1418, 15.1011] | 8.9474 [5.7214, 12.6664] | 2.2296 [1.0662, 3.2093] | 8.7087 [7.3844, 9.9855] |
| bbox_minus_classification | 1.7993 [1.2384, 2.5085] | 3.8732 [-0.7783, 9.3488] | 6.0083 [3.4869, 8.9770] | -1.2373 [-2.2879, -0.3152] | 4.9178 [3.7078, 6.2846] |
| both_minus_bbox | 3.0804 [2.4585, 3.4520] | 2.2889 [-1.5951, 5.4316] | 1.5612 [-0.0082, 3.2275] | 1.7024 [0.7675, 2.5102] | 2.0016 [1.0364, 2.9070] |
| both_minus_classification | 4.8797 [4.4281, 5.2982] | 6.1621 [1.8981, 11.7489] | 7.5696 [4.9545, 10.9084] | 0.4651 [-0.0850, 0.8020] | 6.9194 [5.9152, 8.0245] |
| baseline_minus_fp16 | -9.1116 [-9.6821, -8.3370] | -10.1958 [-16.0900, -4.3431] | -10.1978 [-13.1254, -7.2703] | -2.1238 [-3.1518, -0.9625] | -9.5003 [-10.7273, -8.0705] |
| bbox_minus_fp16 | -3.6853 [-4.1638, -3.0439] | -3.4864 [-6.8747, 0.6081] | -2.8117 [-4.6041, -0.7969] | -1.5967 [-2.4735, -0.6599] | -2.7931 [-3.6966, -1.8071] |
| classification_minus_fp16 | -5.4846 [-5.9896, -5.0174] | -7.3595 [-12.8061, -2.8792] | -8.8200 [-11.7381, -6.3637] | -0.3593 [-0.8469, 0.2664] | -7.7109 [-8.9126, -6.6272] |
| both_minus_fp16 | -0.6049 [-0.9361, -0.3670] | -1.1975 [-2.9487, 0.6160] | -1.2504 [-2.7020, 0.4311] | 0.1058 [-0.2458, 0.4121] | -0.7915 [-1.4773, -0.1414] |

Notation: each cell is `point_delta_pp [95% percentile CI]`; valid/undefined draw counts are in `contrast_ci.json`.

## Interpretation limits

These are exploratory paired dev contrasts conditional on frozen captures, one fixed dev sample, one calibration policy, and one server environment. They do not separate build/tactic variability, calibration variability, training variability, or deployment-device variability; they do not establish causal independence of bbox and classification errors. The localization table is a GT-centric conditional matching diagnostic against `baseline_int8`, not a causal decomposition. No official-test data, retraining, new inference, TensorRT build, or benchmark was run by this analysis.

Point estimates and all ten contrasts must be reviewed together; this report does not select an arm or apply a post-hoc success threshold.
