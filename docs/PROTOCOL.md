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
