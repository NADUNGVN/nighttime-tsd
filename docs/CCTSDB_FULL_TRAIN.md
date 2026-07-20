# Train trên CCTSDB2021 Full (>10k ảnh)

Protocol mới (thay train-only CNTSSS):

| | |
|---|---|
| **Train** | CCTSDB2021 full (~16k train, day+night+weather) |
| **Val/Test in-domain** | CCTSDB test 1.5k |
| **Cross night** | CNTSSS test 786 (và/hoặc CCTSDB night 500) |
| **Classes** | 3: prohibitory / mandatory / warning (cùng CNTSSS) |

> Paper framing: mixed-condition large-scale train + night cross-eval (không còn “train night-only”).

## Trên server 3090

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nighttime-tsd
cd ~/Dung_TDTU/nighttime-tsd
git pull origin master

# 1) Unzip phần còn thiếu (train_img ~1.2G)
cd data/raw/CCTSDB2021
unzip -q train_img.zip -d train_img
unzip -q train_labels.zip -d train_labels
# test_img / test_labels / xml đã có thì bỏ qua
cd ~/Dung_TDTU/nighttime-tsd

# 2) Build YOLO layout
python scripts/prepare_cctsdb_full.py
# Kỳ vọng: train ≈ 16356, test ≈ 1500

# 3) Train (screen khuyến nghị)
screen -S cctsdb_full
python scripts/train_baseline.py \
  --model yolo11n.pt \
  --data configs/cctsdb2021_full.yaml \
  --epochs 100 \
  --batch 64 \
  --workers 8 \
  --device 0 \
  --name yolo11n_cctsdb_full \
  --cache ram

# đối chứng
python scripts/train_baseline.py \
  --model yolov8n.pt \
  --data configs/cctsdb2021_full.yaml \
  --epochs 100 \
  --batch 64 \
  --name yolov8n_cctsdb_full \
  --cache ram
```

## Eval sau train

```bash
# In-domain CCTSDB
python scripts/eval_map.py \
  --weights runs/detect/runs/detect/yolo11n_cctsdb_full/weights/best.pt \
  --data configs/cctsdb2021_full.yaml --batch 32 \
  --out runs/eval_yolo11n_cctsdb_full.json

# Cross: CNTSSS night
python scripts/eval_map.py \
  --weights runs/detect/runs/detect/yolo11n_cctsdb_full/weights/best.pt \
  --data configs/cntsss.yaml --batch 32 \
  --out runs/eval_yolo11n_cctsdbfull_on_cntsss.json

# Cross: CCTSDB night only
python scripts/eval_map.py \
  --weights runs/detect/runs/detect/yolo11n_cctsdb_full/weights/best.pt \
  --data configs/cctsdb2021_night.yaml --batch 32 \
  --out runs/eval_yolo11n_cctsdbfull_on_night.json
```

## So với baseline cũ (CNTSSS-train)

| | Train CNTSSS (đã xong) | Train CCTSDB full (mới) |
|---|---|---|
| # train | 3.276 đêm | ~16k mixed |
| In-domain | CNTSSS strong | CCTSDB strong |
| Night cross | CCTSDB night | CNTSSS night |

Giữ weights cũ `*_cntsss` để so bảng ablation train-set.
