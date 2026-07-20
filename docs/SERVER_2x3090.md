# Server 2× RTX 3090 — hướng dẫn train / gỡ nghẽn

> File riêng cho cấu hình **hai GPU 3090**.  
> Khác với máy 1 GPU (xem thêm `Server_Hardware.md`, `BOTTLENECK_TRAIN_3090.md`).

---

## 1. Phần cứng mục tiêu

| Hạng mục | Giá trị (mục tiêu) |
|---|---|
| GPU | **2× NVIDIA GeForce RTX 3090** (mỗi card **24 GB** VRAM) |
| GPU index | `0` và `1` (`nvidia-smi` / `nvitop`) |
| CPU / RAM | Máy host mạnh (vd. nhiều nhân, RAM lớn — kiểm tra bằng `htop` / `lscpu`) |
| Storage | `$HOME` có thể là **NFS** → vẫn nên **stage data ra local** trước train nặng |

### Kiểm tra 2 GPU

```bash
nvidia-smi -L
# Kỳ vọng 2 dòng: GPU 0 ... 3090, GPU 1 ... 3090

nvitop
# hoặc
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv
```

Nếu chỉ thấy **1** GPU: driver / cáp / `CUDA_VISIBLE_DEVICES` đang ẩn card thứ hai.

---

## 2. Chiến lược train (khuyến nghị)

Với **2× 3090**, chạy **song song 2 model** là hợp lý (mỗi model 1 GPU):

| GPU | Job |
|---|---|
| **GPU 0** | YOLO11n @ CCTSDB full |
| **GPU 1** | YOLOv8n @ CCTSDB full |

**Không** đặt 2 job lên cùng 1 GPU.  
**Có** thể 2 job cùng đọc dataset đã stage local (hoặc mỗi job 1 copy nếu NFS vẫn chậm).

### Nút thắt còn lại (dù có 2 GPU)

| Nút thắt | Cách xử lý |
|---|---|
| Data trên **NFS** | `stage_data_local.py` → `/tmp` hoặc SSD local |
| Dataloader yếu | Sau khi local: `workers=4–8`, `cache=ram` (RAM máy đủ) |
| 2 job × cache RAM × 16k ảnh | Theo dõi RAM host; thiếu thì `cache=false` hoặc chỉ 1 job cache |

---

## 3. Chuẩn bị data (một lần)

```bash
cd ~/Dung_TDTU/nighttime-tsd   # hoặc path repo
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nighttime-tsd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

git pull origin master

# YOLO layout (nếu chưa)
# python scripts/prepare_cctsdb_full.py

# Stage khỏi NFS → local (bắt buộc nếu home = nfs)
python scripts/stage_data_local.py --dst /tmp/cctsdb2021_full
```

Kiểm tra:

```bash
find /tmp/cctsdb2021_full/train/images -type f | wc -l   # ~16356
df -T /tmp                                               # không nên là nfs
```

---

## 4. Train song song 2× 3090

### Cách A — hai `screen` (đơn giản, dễ theo dõi)

**Terminal / screen 1 — GPU 0 — YOLO11n**

```bash
screen -S train_gpu0
conda activate nighttime-tsd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd ~/Dung_TDTU/nighttime-tsd

python scripts/train_3090.py \
  --local-data /tmp/cctsdb2021_full \
  --model yolo11n.pt \
  --device 0 \
  --name yolo11n_cctsdb_full \
  --preset local
```

`Ctrl+A` `D` detach.

**Terminal / screen 2 — GPU 1 — YOLOv8n**

```bash
screen -S train_gpu1
conda activate nighttime-tsd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd ~/Dung_TDTU/nighttime-tsd

python scripts/train_3090.py \
  --local-data /tmp/cctsdb2021_full \
  --model yolov8n.pt \
  --device 1 \
  --name yolov8n_cctsdb_full \
  --preset local
```

### Cách B — một lệnh nền (không screen)

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/train_3090.py \
  --local-data /tmp/cctsdb2021_full --model yolo11n.pt \
  --device 0 --name yolo11n_cctsdb_full --preset local \
  > logs/train_yolo11n.log 2>&1 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/train_3090.py \
  --local-data /tmp/cctsdb2021_full --model yolov8n.pt \
  --device 1 --name yolov8n_cctsdb_full --preset local \
  > logs/train_yolov8n.log 2>&1 &
```

(`mkdir -p logs` trước.)

### Cách C — `train_baseline.py` thuần

```bash
# GPU 0
python scripts/train_baseline.py \
  --model yolo11n.pt \
  --data configs/cctsdb2021_full_local.yaml \
  --epochs 100 --batch 96 --workers 6 --cache ram \
  --device 0 --name yolo11n_cctsdb_full

# GPU 1 (shell khác)
python scripts/train_baseline.py \
  --model yolov8n.pt \
  --data configs/cctsdb2021_full_local.yaml \
  --epochs 100 --batch 96 --workers 6 --cache ram \
  --device 1 --name yolov8n_cctsdb_full
```

Lưu ý: `train_3090.py` cần nhận `--device` (đã hỗ trợ). Nếu preset local ghi đè device, luôn truyền `--device 0` / `1` như trên.

---

## 5. Preset gợi ý theo từng GPU

| Tham số | Giá trị gợi ý (mỗi job / 1× 3090) |
|---|---|
| Model | YOLO11n hoặc YOLOv8n (nano) |
| imgsz | 640 |
| batch | **64–96** (có thể 128 nếu VRAM trống) |
| workers | **4–8** sau khi data **local** |
| cache | **`ram`** nếu host RAM lớn; NFS thì `false` |
| epochs | 100 |
| AMP | bật (mặc định Ultralytics) |

**Hai job cùng lúc:** mỗi GPU một process; **không** dùng DataParallel cho 2 model khác nhau.

---

## 6. Theo dõi

```bash
nvitop
# Kỳ vọng:
#   GPU0 util cao + process yolo11n
#   GPU1 util cao + process yolov8n

screen -ls
screen -r train_gpu0
screen -r train_gpu1
```

Weights:

```text
runs/detect/.../yolo11n_cctsdb_full/weights/best.pt
runs/detect/.../yolov8n_cctsdb_full/weights/best.pt
```

(Đường dẫn có thể lồng `runs/detect/runs/detect/` tùy version Ultralytics — dùng `find runs -name best.pt`.)

---

## 7. Eval sau train (1 GPU là đủ)

```bash
# In-domain CCTSDB
python scripts/eval_map.py \
  --weights runs/detect/runs/detect/yolo11n_cctsdb_full/weights/best.pt \
  --data configs/cctsdb2021_full_local.yaml --batch 32 --device 0 \
  --out runs/eval_yolo11n_cctsdb_full.json

python scripts/eval_map.py \
  --weights runs/detect/runs/detect/yolov8n_cctsdb_full/weights/best.pt \
  --data configs/cctsdb2021_full_local.yaml --batch 32 --device 0 \
  --out runs/eval_yolov8n_cctsdb_full.json

# Cross night CNTSSS
python scripts/eval_map.py \
  --weights runs/detect/runs/detect/yolo11n_cctsdb_full/weights/best.pt \
  --data configs/cntsss.yaml --batch 32 --device 0 \
  --out runs/eval_yolo11n_cctsdbfull_on_cntsss.json
```

---

## 8. Việc **không** nên làm

| Không | Lý do |
|---|---|
| 2 job trên **cùng** GPU 0 | OOM / chậm hơn nối tiếp |
| 2 job đọc thẳng **NFS** full 16k | I/O nghẽn, GPU util thấp |
| `workers=0` lâu dài khi đã có local + nhiều CPU | Lãng phí GPU |
| Giả định DDP 2 GPU cho **1** model nano | Không cần; song song **2 model** đơn giản hơn |

---

## 9. Checklist nhanh 2× 3090

- [ ] `nvidia-smi -L` → 2× 3090  
- [ ] Data staged local (`/tmp/cctsdb2021_full`, ≥10k train)  
- [ ] `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` (tránh lỗi OpenCV/libstdc++)  
- [ ] Screen `train_gpu0` (device 0) + `train_gpu1` (device 1)  
- [ ] `nvitop`: cả 2 GPU util cao  
- [ ] Xong → eval + `git add -f` JSON/weights nếu cần  

---

## 10. Liên kết file khác

| File | Nội dung |
|---|---|
| `docs/BOTTLENECK_TRAIN_3090.md` | Gỡ nghẽn NFS / workers / cache (áp dụng từng GPU) |
| `docs/CCTSDB_FULL_TRAIN.md` | Protocol CCTSDB full |
| `docs/Server_Hardware.md` | Snapshot 1 GPU / host khác (nếu có) |
| `scripts/stage_data_local.py` | Copy data → local |
| `scripts/train_3090.py` | Preset train |

---

*Cập nhật: dùng khi host có đúng 2× RTX 3090. Nếu `nvidia-smi` chỉ 1 card, quay lại train tuần tự 1 GPU.*
