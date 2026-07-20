# Train trên server RTX 3090 24GB

## 1. Clone code (không kèm data)

```bash
git clone <YOUR_REPO_URL> nighttime-tsd
cd nighttime-tsd
python -m venv .venv && source .venv/bin/activate   # Linux
pip install -r requirements.txt
# PyTorch CUDA (nếu pip chưa kéo đúng):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 2. Data CNTSSS trên server

**Không push lên git** (~1 GB, trong `.gitignore`).

```bash
# Cách A — tải lại từ Drive (cùng script)
pip install gdown
python scripts/download_cntsss.py --extract

# Cách B — scp từ máy local
# scp data/raw/CNTSSS.zip user@server:~/nighttime-tsd/data/raw/
# unzip data/raw/CNTSSS.zip -d data/raw/
```

Kiểm tra:

```bash
python scripts/verify_cntsss.py
python scripts/check_env.py
```

## 3. Load model (pretrained / checkpoint)

```bash
# Pretrained YOLO11n (tự download weights COCO)
python scripts/load_model.py --model yolo11n.pt --info

# Pretrained YOLOv8n
python scripts/load_model.py --model yolov8n.pt --info

# Sau khi train xong
python scripts/load_model.py --model runs/detect/yolo11n_cntsss/weights/best.pt --info
python scripts/load_model.py --model runs/detect/yolo11n_cntsss/weights/best.pt \
    --val --data configs/cntsss.yaml --batch 32
```

API tương đương trong Python:

```python
from ultralytics import YOLO

# 1) Pretrained (download tự động lần đầu)
model = YOLO("yolo11n.pt")

# 2) Checkpoint đã fine-tune
model = YOLO("runs/detect/yolo11n_cntsss/weights/best.pt")

# 3) Resume mid-run
model = YOLO("runs/detect/yolo11n_cntsss/weights/last.pt")
model.train(resume=True)
```

## 4. Train (3090)

```bash
# Main model
python scripts/train_baseline.py \
  --model yolo11n.pt \
  --data configs/cntsss.yaml \
  --epochs 100 \
  --batch 64 \
  --workers 8 \
  --device 0 \
  --name yolo11n_cntsss \
  --cache ram

# Control
python scripts/train_baseline.py \
  --model yolov8n.pt \
  --batch 64 \
  --epochs 100 \
  --name yolov8n_cntsss \
  --cache ram
```

| GPU | batch gợi ý (nano, imgsz=640) |
|---|---|
| 5060 8GB | 8–16 |
| **3090 24GB** | **32–64** (có thể 96 nếu ổn) |

## 5. Lấy weights về máy / Orin

```bash
# weights
scp user@server:~/nighttime-tsd/runs/detect/yolo11n_cntsss/weights/best.pt ./weights/

# trên Orin: export INT8 (calibrate on device)
python scripts/export_tensorrt.py --weights best.pt --precision int8 \
  --calib configs/calibration_night.yaml
```

## Layout mong đợi

```
nighttime-tsd/
  configs/cntsss.yaml          # path: ../data/raw/CNTSSS
  data/raw/CNTSSS/{train,test}/{images,labels}
  scripts/load_model.py
  scripts/train_baseline.py
  runs/detect/<name>/weights/{best.pt,last.pt}
```
