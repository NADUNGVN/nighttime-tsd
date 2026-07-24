# Train / eval trên CCTSDB2021 (dataset duy nhất)

Xem số ảnh Drive/train/test: [`DATASET_CCTSDB2021.md`](DATASET_CCTSDB2021.md).

```text
Train: 16_356 mixed
Test:  1_500  → weather: sunny 400 | cloud 300 | rain 160 | snow 100 | foggy 40 | night 500
```

```bash
python scripts/train_cctsdb.py --model yolo11n.pt \
  --data configs/cctsdb2021_train.yaml \
  --epochs 100 --batch 64 --workers 8 --cache ram --name yolo11n_cctsdb_full

# Nếu server/ổ mạng chậm:
python scripts/train_cctsdb.py --model yolo11n.pt \
  --stage-to /tmp/cctsdb2021_full \
  --epochs 100 --batch 64 --workers 8 --cache ram --name yolo11n_cctsdb_full
```
