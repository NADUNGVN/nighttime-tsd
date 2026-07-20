# Train trên CCTSDB2021 Full (>10k ảnh)

Protocol mới (thay train-only CNTSSS):

| | |
|---|---|
| **Train** | CCTSDB2021 full (~16k train, day+night+weather) |
| **Val/Test in-domain** | CCTSDB test 1.5k |
| **Cross night** | CNTSSS test 786 (và/hoặc CCTSDB night 500) |
| **Classes** | 3: prohibitory / mandatory / warning (cùng CNTSSS) |

> Paper framing: mixed-condition large-scale train + night cross-eval (không còn “train night-only”).

## Trên server 3090 (gỡ nghẽn NFS)

Phần cứng: [`Server_Hardware.md`](Server_Hardware.md) · tối ưu: [`BOTTLENECK_TRAIN_3090.md`](BOTTLENECK_TRAIN_3090.md)

**Nút thắt:** data trên NFS + `workers=0` → GPU util ~0%.  
**Cách gỡ:** copy data ra local (`/tmp`) → `workers=6` + `cache=ram` + `batch=96`.

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nighttime-tsd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd ~/Dung_TDTU/nighttime-tsd
git pull origin master

# 0) (một lần) prepare YOLO layout nếu chưa có
# python scripts/prepare_cctsdb_full.py   # train≈16356

# 1) Stage khỏi NFS → local (QUAN TRỌNG)
python scripts/stage_data_local.py --dst /tmp/cctsdb2021_full

# 2) Train tối ưu 3090
screen -S cctsdb_full
python scripts/train_3090.py \
  --local-data /tmp/cctsdb2021_full \
  --model yolo11n.pt \
  --name yolo11n_cctsdb_full

# 3) Sau khi 11n xong — YOLOv8n (không song song 1 GPU)
python scripts/train_3090.py \
  --local-data /tmp/cctsdb2021_full \
  --model yolov8n.pt \
  --name yolov8n_cctsdb_full
```

Nếu **không** stage được (hết chỗ /tmp):  
`python scripts/train_3090.py --preset nfs --model yolo11n.pt` (chậm hơn, workers=2).

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
