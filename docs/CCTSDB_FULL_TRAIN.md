# Train / eval trên CCTSDB2021 (dataset duy nhất)

Xem số ảnh Drive/train/test: [`DATASET_CCTSDB2021.md`](DATASET_CCTSDB2021.md).

```text
Train: 16_356 mixed
Test:  1_500  → weather: night ~500 | non-night ~1000
```

```bash
python scripts/prepare_cctsdb_full.py
python scripts/stage_data_local.py --dst /tmp/cctsdb2021_full
python scripts/train_baseline.py --model yolo11n.pt \
  --data configs/cctsdb2021_full_local.yaml \
  --epochs 100 --batch 64 --workers 8 --cache ram --name yolo11n_cctsdb_full

python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_full_local.yaml
python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_night.yaml
python scripts/build_calibration_sets.py --n 256
```
