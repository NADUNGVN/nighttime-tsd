# CCTSDB2021 protocol

This repository uses only CCTSDB2021. Historical CNTSSS, multi-model, and
night-subset artifacts were removed because their label semantics and/or
provenance were not compatible with this protocol.

## Immutable rules

- Raw CCTSDB labels are `0=mandatory`, `1=prohibitory`, `2=warning`.
- The processed benchmark uses `0=prohibitory`, `1=mandatory`, `2=warning`.
  `scripts/prepare_cctsdb.py` is the only conversion path and verifies this
  mapping against the official weather XML annotations.
- The 1,500-image official test set is never used as training validation,
  early stopping input, model-selection input, or INT8 calibration input.
- Development validation is a deterministic 10% split of the 16,356-image
  training set, seed 42. It is not a video-grouped split because the release
  does not expose video grouping metadata.
- INT8 calibration may use only `train/images`. A luminance-selected subset is
  called *low-luminance*, never *official night*.
- FP32, FP16 TensorRT, and each INT8 calibration strategy must be evaluated
  with the identical official-test YAMLs. The checkpoint is frozen before
  export; only the inference representation changes.

## Build the dataset

Either download the official archive folder into `data/raw/`, or point to an
already extracted official release, then run:

```bash
python scripts/prepare_cctsdb.py --seed 42 --dev-ratio 0.10

# Example: data lives in a separate server checkout.
python scripts/prepare_cctsdb.py \
  --raw ../nighttime-tsd/data/raw/CCTSDB2021 --seed 42 --dev-ratio 0.10
```

The builder expects `train_img.zip`, `train_labels.zip`, `test_img.zip`,
`test_labels.zip`, and `Classification based on weather and environment.zip`.
It creates `data/processed/cctsdb2021_clean/` with train, dev, official test,
six official domains, and a provenance manifest.
When `--raw` points to an extracted release outside this checkout, source
images are hard-linked when possible (otherwise copied); raw data is never
moved or modified.

## Train and evaluate

```bash
python scripts/train_cctsdb.py --model yolo11n.pt --epochs 100 --batch 64 \
  --seed 42 --name yolo11n_cctsdb_clean_s42

python scripts/evaluate_cctsdb.py \
  --weights runs/detect/yolo11n_cctsdb_clean_s42/weights/best.pt \
  --data configs/cctsdb2021_test_night.yaml --label fp32 \
  --out results/fp32/yolo11n_night.json
```

Evaluate full, daylike, and all six domains. Report per-class results only
when that class is represented in the corresponding domain; `foggy` has 40
images and needs an explicit uncertainty caveat.

For architecture comparisons, train every candidate with the same fixed split,
seed, epochs, image size, and nominal batch size. Per-device mini-batch may
decrease for larger scales; keep Ultralytics' nominal batch size fixed through
gradient accumulation. Evaluate frozen checkpoints through
the suite below; it refuses to overwrite results and records checkpoint and
result hashes in a manifest.

```bash
python scripts/evaluate_weights_suite.py --weights fp32_yolov8n=results/yolov8n_cctsdb_clean_s42_v1/weights/best.pt --weights fp32_yolo26n=results/yolo26n_cctsdb_clean_s42_v1/weights/best.pt --out-dir results/eval/fp32_architecture --batch 64
```

### Full-scale architecture matrix

`configs/architecture_matrix_v1.json` defines all five detection scales
(`n`, `s`, `m`, `l`, `x`) for YOLOv8, YOLO11, and YOLO26. The completed `n`
checkpoints are reused; the remaining twelve runs are serial so one server GPU
is never shared by concurrent training jobs. Batch size is reduced by scale to
fit the 48 GB RTX 8000; Ultralytics keeps its nominal batch size at 64 through
gradient accumulation. The server uses two dataloader workers because eight
workers exhausted its shared-memory allocation. Each newly trained run records
both requested and effective Ultralytics arguments in its provenance file.

First inspect the plan and server state, then start the serial queue. The
runner stops on the first failure and never overwrites a finished `best.pt`.
Only use `--resume-incomplete` when the corresponding `last.pt` exists and
you intentionally want to resume that exact run.

```bash
python scripts/run_architecture_matrix.py --phase status
python scripts/run_architecture_matrix.py --phase train
python scripts/run_architecture_matrix.py --phase train --resume-incomplete
```

After all 15 checkpoints are complete, evaluate each frozen FP32 model across
the same eight official-test splits. This command refuses to evaluate an
incomplete matrix and writes a separate manifest and 120 result JSON files.

```bash
python scripts/run_architecture_matrix.py --phase evaluate
```

Benchmark the same frozen FP32 checkpoints separately from accuracy. This is a
batch-one server-runtime comparison: CCTSDB development images are preloaded,
disk decoding is excluded, and each measured call includes preprocessing,
inference, and post-processing. It is not an edge-device or energy result and
must not be reported as a Jetson measurement.

```bash
python scripts/run_architecture_matrix.py --phase benchmark
```

## TensorRT calibration and export

The server export stack is pinned to Python 3.11, PyTorch `2.5.1+cu121`,
Ultralytics `8.4.102`, and TensorRT `10.16.1.11`. Do **not** install an
unbounded `tensorrt-cu12` package: TensorRT 11 routes Ultralytics INT8 export
through NVIDIA ModelOpt, whose current Torch requirement is incompatible with
the frozen baseline environment. TensorRT 10 uses the native calibration path.

Install or repair this stack with the following exact commands in the active
`nighttime-tsd` Conda environment:

```bash
python -m pip install --force-reinstall --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip uninstall -y nvidia-modelopt tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs
python -m pip install --no-cache-dir tensorrt-cu12==10.16.1.11
```

Create two 512-image calibration sets from `train/images` only. The first is
a seeded uniform reference; the second deliberately selects the darkest train
images by median grayscale luminance. These are calibration policies, not
training subsets and not official-night data.

```bash
python scripts/build_calibration_set.py --strategy uniform --size 512 --seed 42 --name uniform_s42_n512
python scripts/build_calibration_set.py --strategy low_luminance --size 512 --seed 42 --name low_luminance_n512
```

Export the same frozen checkpoint once as FP16 and once per INT8 calibration
policy. Each export emits a JSON provenance record with hashes for the source
checkpoint, calibration manifest, and generated engine. Engines are hardware
specific and are not committed; commit only the JSON results and provenance.

```bash
python scripts/export_tensorrt.py --weights results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt --precision fp16 --out results/engines/yolo11n_cctsdb_clean_s42_v2_fp16_trt1016.engine
python scripts/export_tensorrt.py --weights results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt --precision int8 --data data/processed/cctsdb2021_clean/calibration/uniform_s42_n512/calibration.yaml --out results/engines/yolo11n_cctsdb_clean_s42_v2_int8_uniform_s42_n512_trt1016.engine
```

Evaluate all official splits at TensorRT's static export batch size of one.
The suite refuses to overwrite any result, and writes a manifest containing
every engine/result hash.

```bash
python scripts/evaluate_tensorrt_suite.py --engine fp16_trt1016=results/engines/yolo11n_cctsdb_clean_s42_v2_fp16_trt1016.engine --engine int8_uniform_trt1016=results/engines/yolo11n_cctsdb_clean_s42_v2_int8_uniform_s42_n512_trt1016.engine --engine int8_low_luminance_trt1016=results/engines/yolo11n_cctsdb_clean_s42_v2_int8_low_luminance_n512_trt1016.engine --out-dir results/eval/tensorrt10 --batch 1
```

## Paper artifacts and deployment benchmark

Do not commit images, labels, or TensorRT engine binaries. Instead, collect
the split/calibration manifests, training CSVs, environment lock, and model
inventory; these small artifacts make the experiments auditable without
duplicating CCTSDB. The collector copies manifests and CSV/JSON only.

```bash
python scripts/collect_paper_artifacts.py --run yolo11n=results/yolo11n_cctsdb_clean_s42_v2 --run yolov8n=results/yolov8n_cctsdb_clean_s42_v1 --run yolo26n=results/yolo26n_cctsdb_clean_s42_v1 --engine fp16_trt1016=results/engines/yolo11n_cctsdb_clean_s42_v2_fp16_trt1016.engine --engine int8_uniform_trt1016=results/engines/yolo11n_cctsdb_clean_s42_v2_int8_uniform_s42_n512_trt1016.engine --engine int8_low_luminance_trt1016=results/engines/yolo11n_cctsdb_clean_s42_v2_int8_low_luminance_n512_trt1016.engine --out-dir results/paper_artifacts/v1
```

`configs/deployment/yolo11n_fp16_trt10.json` is the current FP16 deployment
contract. Its confidence threshold is provisional and must be selected on the
development split, never on the official test. Benchmark end-to-end batch-one
latency on preloaded images (disk decode is excluded):

```bash
python scripts/benchmark_tensorrt_engine.py --engine results/engines/yolo11n_cctsdb_clean_s42_v2_fp16_trt1016.engine --images data/processed/cctsdb2021_clean/dev/images --out results/benchmark/fp16_trt1016_server.json --warmup 50 --samples 500
```
