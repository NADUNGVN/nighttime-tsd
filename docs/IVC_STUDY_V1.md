# Locked IVC study: domain-robust INT8 calibration

## Claim and scope

The paper investigates whether the selection of **CCTSDB training images** for
post-training INT8 calibration changes accuracy retention across environmental
domains. It is a controlled detector-deployment study, not a new detector or
an external-data transfer-learning paper.

The study contains the completed 15-model matrix: YOLOv8, YOLO11, and YOLO26
at `n`, `s`, `m`, `l`, and `x` scale. TT100K, MTSD, and CURE-TSD are excluded
from this manuscript. The CCTSDB official test is evaluation-only.

## Fixed calibration policies

All policies select exactly 1,024 images from `CCTSDB train/images`, at image
size 640 and seed 42:

- `uniform_s42_n1024`: seeded uniform random sample.
- `low_luminance_n1024`: 1,024 lowest median grayscale-luminance images.
- `luminance_stratified_s42_n1024`: sort all train images by the same
  luminance statistic, divide into four equal-count strata, and sample 256
  images from each stratum using seed 42.

No policy uses an official-test image, weather label, or test result. The
already committed 512-image YOLO11n engines are pilots and are not the final
experiment.

## Target execution matrix

| Target | Required representations | Interpretation |
|---|---|---|
| RTX 8000 reference | FP16 + all three INT8 policies, all 15 models | calibration-effect reference |
| Xavier NX | FP16 + all three INT8 policies, all 15 models | target-device calibration evidence |
| AGX Xavier | FP16 + all three INT8 policies, all 15 models | target-device calibration evidence |
| Jetson Nano | TensorRT FP16, attempt all 15 | constrained feasibility baseline; no INT8 |
| Raspberry Pi 5 CPU | one pinned FP32 CPU backend, attempt all 15 | CPU baseline |
| Raspberry Pi 5 + Hailo | Hailo INT8 with LSC, attempt all 15 | accelerator feasibility/deployment |

For every successful engine, evaluate `full`, `daylike`, `sunny`, `cloud`,
`night`, `rain`, `snow`, and `foggy`. Every attempted but unsupported build or
run is retained as a versioned failure record. TensorRT engines are built on
the target device and never copied from the RTX server.

Jetson Nano is intentionally FP16-only: it has no native TensorRT INT8
hardware support. Xavier-class devices support FP16 and INT8. See NVIDIA's
[TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/support-matrix/)
and [Jetson Nano guidance](https://forums.developer.nvidia.com/t/jetson-nano-tensorrt-engine-int8/344803/6).

## Measurement contract

- Use GPU TensorRT, not DLA, on Xavier NX and AGX Xavier.
- Use static batch one and 640 input everywhere.
- Fix confidence `0.25` and IoU `0.70` for latency tests. Disk decoding is
  excluded; preprocessing, inference, and post-processing are timed.
- Use 50 warm-up inferences, 1,000 timed preloaded images, and three fresh
  process repetitions per successful model/representation/device.
- Run each board in its highest supported stable performance mode, with fixed
  cooling and no competing workload. Record the exact `nvpmodel`, clock,
  governor, software versions, RAM, and power supply.
- Report mAP50, mAP50-95, `Delta mAP = INT8 - FP16`, worst-domain Delta mAP,
  p50/p95 latency, throughput, memory, build success, and energy/image where
  a reliable device power measurement exists.

## Commands and artifacts

Capture inventory before any build. Do not overwrite an existing inventory.

```bash
python scripts/capture_device_inventory.py --target xavier_nx_8gb --out results/ivc_study_v1/xavier_nx/device_inventory.json --notes "8GB; active fan; official PSU; max stable nvpmodel"
```

The NVIDIA runner is serial, restart-safe, and records status under
`results/ivc_study_v1/<target>/`. Use the TensorRT major version actually
recorded on that target; TensorRT 8--10 native PTQ is supported by the runner.

```bash
python scripts/run_ivc_quantization_matrix.py --target rtx8000 --phase calibrations --expected-tensorrt-major 10
python scripts/run_ivc_quantization_matrix.py --target rtx8000 --phase export --expected-tensorrt-major 10
python scripts/run_ivc_quantization_matrix.py --target rtx8000 --phase evaluate --expected-tensorrt-major 10 --save-full-predictions
python scripts/run_ivc_quantization_matrix.py --target rtx8000 --phase benchmark --expected-tensorrt-major 10
```

Do not run two NVIDIA matrix phases concurrently. If a partial engine or
evaluation directory remains after a failure, inspect and move it to a clearly
named quarantine directory before retrying; the runner deliberately refuses
to overwrite partial evidence.

After a target matrix is complete, generate tables without modifying the raw
evidence. Use the saved full-test predictions with any official split YAML for
a paired FP16-versus-INT8 bootstrap interval.

```bash
python scripts/aggregate_ivc_results.py --root results/ivc_study_v1/rtx8000 --out-dir results/ivc_study_v1/rtx8000/tables
python scripts/bootstrap_cctsdb_map50.py --data configs/cctsdb2021_test_night.yaml --fp16-predictions results/ivc_study_v1/rtx8000/predictions/yolo11n_fp16_full_predictions.json --candidate-predictions results/ivc_study_v1/rtx8000/predictions/yolo11n_int8_luminance_stratified_full_predictions.json --out results/ivc_study_v1/rtx8000/bootstrap/yolo11n_lsc_night.json --iterations 2000 --seed 42
```

## Completion criteria

The study is complete when every planned cell has either a valid result or a
recorded reason for failure, every successful deployment has three benchmark
replicates, and result tables include domain-wise confidence intervals from
saved per-image predictions. The outcome is not pre-committed: LSC may win,
tie, or lose; the contribution is the controlled evidence.
