# YOLO11n VCSC-proportional development gate

VCSC-v1 is rejected and remains frozen. VCSC-v2 changes one mechanism only:
the visual descriptors and deterministic K-means remain train-only, while the
calibration quota is allocated approximately in proportion to the K-means
cluster populations. The allocation uses deterministic largest remainder with
a documented minimum of one image for every non-empty cluster.

The runner is intentionally unable to evaluate official positive test images,
weather domains, or official negative scenes. VCSC-v2 selection is restricted
to `CCTSDB2021/dev` after calibration images are selected solely from `train`.
No FP32 weight is retrained.

## Execution order

Run each command as one physical line. Calibration construction is CPU-only;
export and dev evaluation require the GPU phase lock and must not run together.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && python scripts/run_yolo11n_calibration_development_v2.py --phase status --seeds initial --policies all
```

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && python scripts/run_yolo11n_calibration_development_v2.py --phase preflight --seeds initial --policies all
```

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && python scripts/run_yolo11n_calibration_development_v2.py --phase calibrations --seeds initial --policies all
```

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && python scripts/run_yolo11n_calibration_development_v2.py --phase calibrations --seeds stability --policies uniform,vcsc_proportional
```

Before the GPU phases, inspect the saved VCSC-proportional calibration
manifests: every selected image must be train-only, the quota policy must be
`population_proportional_largest_remainder`, and selected size must be 1,024.

After this verification, export and evaluate the initial seed first. Do not
start the five-seed GPU experiment until the initial dev-only pipeline is
checked and its outputs are pushed for review.
