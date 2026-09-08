# YOLO11n calibration-method decision gate

This is the only active INT8 experiment. The 15 FP32 checkpoints are frozen;
there is no retraining, no external training data, and no 15-model TensorRT
export until this gate is reviewed.

## Fixed model and leakage boundary

- Model: `results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt`.
- Calibration candidates: only the 14,720 CCTSDB train images.
- Development set: 1,636 images; it may support operational configuration but
  never calibration construction.
- Official 1,500 positive test, official XML annotations for instance-level
  size evaluation, and 500 official
  negative scenes are evaluation-only. They are never used for model choice,
  early stopping, calibration construction, or policy design.

## Calibration policies

All use 1,024 train-only images. Uniform and low-luminance retain their prior
definitions. VCSC creates a six-dimensional visual descriptor per training
image: mean luminance, luminance standard deviation, mean saturation, entropy,
Laplacian variance, and dark-channel mean. Features are standardized using
only the training pool, clustered with deterministic K-means++ (`K=8`), and
sampled equally at 128 images/cluster. To avoid a pathological initialization
that makes this quota impossible, VCSC tries up to 32 deterministic restarts
derived only from the calibration seed and accepts the first quota-feasible
result; the chosen restart and K-means seed are saved.

The VCSC manifest saves every candidate ID, raw and standardized descriptors,
cluster assignment, selected IDs, seed, K, feature definitions, source hashes,
script hash, and Git commit.

Each TensorRT INT8 engine is exported in a fresh engine-specific workspace.
This is mandatory because TensorRT 8--10 calibration caches are adjacent to the
intermediate ONNX file; no calibration cache may be shared by two policies or
seeds. The provenance records the fresh cache hash.

If a log reports that it is reading a cache while a new policy/seed is being
exported, stop the run and quarantine those INT8 artifacts. They are not valid
evidence of their declared calibration sets.

## Official size and negative protocols

CCTSDB2021 officially defines sign-size thresholds:
XS ≤210 px²; S (210,400]; M (400,1000]; L (1000,2000]; XL >2000 px². The
CCTSDB's released `xml.zip` provides one XML annotation per positive-test
image. Because an image can contain multiple sign sizes, the implementation
calculates AP50 at the **instance** level from complete full-test predictions;
it does not fabricate image subsets or discard mixed-size scenes. CCTSDB2021
also provides 500 negative images. They remain separate from
mAP and are evaluated only for fixed-threshold false positives at 0.25, 0.50,
and 0.75. [Official release README](https://github.com/csust7zhangjm/CCTSDB2021),
[dataset paper](https://centaur.reading.ac.uk/106129/1/12-23.pdf).

## Commands

All commands are one physical shell line. First audit the official negative
release. The output location has a default to avoid long command paths.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && python scripts/audit_cctsdb_negative_set.py --raw ../nighttime-tsd/data/raw/CCTSDB2021
```

Run the initial four-way seed-42 pilot in discrete phases. No command below
trains a model. The evaluation phase saves complete full-test predictions and
automatically calculates XML-based XS/S/M/L/XL metrics.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && conda activate nighttime-tsd && python scripts/run_yolo11n_calibration_development.py --phase calibrations --seeds initial
```

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && conda activate nighttime-tsd && python scripts/run_yolo11n_calibration_development.py --phase export --seeds initial
```

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && conda activate nighttime-tsd && python scripts/run_yolo11n_calibration_development.py --phase evaluate --seeds initial
```

After the negative audit reports exactly 500 files, replace `NEGATIVE_SOURCE`
with the audited archive or extracted folder and run:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && conda activate nighttime-tsd && python scripts/run_yolo11n_calibration_development.py --phase negative --seeds initial --negative-source NEGATIVE_SOURCE
```

For the five-seed stability campaign, run Uniform and VCSC with
`--seeds stability --policies uniform,vcsc`. Existing seed-42 artifacts are
never overwritten. Low-Luminance is an exact ranked baseline: its selection is
identical across seeds, so it is retained once at seed 42 rather than being
misreported as five sampling replicates.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && python scripts/run_yolo11n_calibration_development.py --phase calibrations --seeds stability --policies uniform,vcsc
```

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && python scripts/run_yolo11n_calibration_development.py --phase export --seeds stability --policies uniform,vcsc
```

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && python scripts/run_yolo11n_calibration_development.py --phase evaluate --seeds stability --policies uniform,vcsc
```

Create the decision table only after all five seeds are complete:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && conda activate nighttime-tsd && python scripts/summarize_yolo11n_calibration_development.py --out-dir results/calibration_method_v1/rtx8000/yolo11n/summary
```

The bootstrap command produces paired 1,000-resample intervals for full,
weather, and macro-domain Delta mAP50. Change only the candidate filename for
Uniform or Low-Luminance.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && conda activate nighttime-tsd && python scripts/bootstrap_yolo11n_calibration.py --fp16-predictions results/calibration_method_v1/rtx8000/yolo11n/predictions/yolo11n_fp16_reference_full_predictions.json --candidate-predictions results/calibration_method_v1/rtx8000/yolo11n/predictions/yolo11n_int8_vcsc_s42_full_predictions.json --out results/calibration_method_v1/rtx8000/yolo11n/bootstrap/vcsc_s42.json --iterations 1000 --seed 42
```

## Decision rule

The summary emits one result:

- `SCALE`: VCSC improves mean macro-domain Delta mAP50 over Uniform by ≥0.005,
  does not reduce mean full-test mAP50, and is no more variable across five
  calibration seeds.
- `MODIFY_VCSC`: its macro-domain advantage is within ±0.005 of Uniform;
  inspect descriptors/K/clusters instead of scaling.
- `STOP`: VCSC is ≥0.005 worse in macro-domain Delta mAP50 or materially more
  variable.
- `INCOMPLETE`: missing evidence; scale-up remains blocked.
