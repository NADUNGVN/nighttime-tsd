# Quantized Traffic Sign Detection Across Weather/Light Domains on Edge (CCTSDB2021)

Bài hướng tới: **YOLO + quant (FP32→FP16→INT8)** trên **CCTSDB2021**, đo **accuracy theo 6 điều kiện weather/light chính thức trên test** và sau này **latency–power trên Jetson Orin**.

Dataset chi tiết: [`docs/DATASET_CCTSDB2021.md`](docs/DATASET_CCTSDB2021.md)  
Plan lịch sử (đã thu hẹp): [`docs/README_nighttime_2papers.md`](docs/README_nighttime_2papers.md) *(archive — không còn multi-dataset)*.

---

## 1. Dataset — chỉ **CCTSDB2021**

Drive: https://drive.google.com/drive/folders/14Km2W-5hbixXDfz7WSqW_Rx7O5m8ZMFn

| Phần | Số ảnh | Ghi chú |
|---|---|---|
| **Train** | **16 356** | Mixed day/night/weather — **không** có nhãn sáng/tối từng ảnh |
| **Test (positive)** | **1 500** | In-domain hold-out |
| **Negative** (optional) | 500 | Không dùng train hiện tại |
| **Tổng positive** | **17 856** | = 16356 + 1500 — đây là con số “hơn 16k” trên Drive |
| Weather split (test only) | 1 500 XML | **sunny 400** · **cloud 300** · **rain 160** · **snow 100** · **foggy 40** · **night 500** |

**16k = chỉ train.** Cả bộ labeled positive ≈ **17.9k** ảnh (+ zip phụ: XML, weather, size — không nhân đôi số ảnh train).

### 3 lớp

`0` prohibitory · `1` mandatory · `2` warning

### Đã loại khỏi plan

CNTSSS, INTSD, TT100K(-night), CURE-TSD, NTS-YOLO — quá nhỏ / không public / lệch trục. Không mở rộng protocol mới trên các bộ này.

---

## 2. Câu hỏi nghiên cứu (quant theo domain)

Train: **16 356 mixed** (không biết % night trong train).

Đo sau quant trên **test 1500 đã biết weather/light domain**:

| Subset test | Số ảnh | Mục đích |
|---|---|---|
| Full test | 1500 | mAP tổng in-domain |
| **Sunny** | 400 | domain sáng rõ |
| **Cloud** | 300 | domain trời nhiều mây |
| **Rain** | 160 | domain mưa |
| **Snow** | 100 | domain tuyết |
| **Foggy** | 40 | domain sương mù, report kèm caveat vì nhỏ |
| **Night** | 500 | domain tối |
| **Day-like** | 1000 | aggregate phụ = sunny + cloud + rain + snow + foggy |

→ INT8 làm **mAP từng domain** giảm bao nhiêu, domain nào nhạy nhất, lớp nào (0/1/2) yếu hơn — **không** phải “giảm lr/batch”.

---

## 3. Kết quả FP32 hiện tại (train CCTSDB full)

| Model | Test full mAP50 | Test night mAP50 | Ghi chú |
|---|---|---|---|
| YOLOv8n | **0.781** | 0.265 | |
| YOLO11n | 0.780 | 0.241 | |
| YOLO26n | 0.780 | 0.253 | mAP50-95 full ~0.50 |

Finding hiện tại: in-domain tốt (~0.78); **subset night trong cùng CCTSDB đã rơi mạnh** (~0.24–0.26); per-class night: class 0/1 ~0, warning còn cao. Cần re-eval FP32 theo 6 domain official để chuẩn hóa bảng chính.

JSON: `runs/eval_yolo*_cctsdb_full.json`, `runs/eval_*_on_cctsdb_night.json`.

---

## 4. Pipeline hiện tại

| Bước | Việc |
|---|---|
| ✅ | Train 11n / 8n / 26n trên CCTSDB full |
| ✅ | Eval test full + night subset |
| ✅ | **Tách test thành 6 weather/light domain official** (`split_cctsdb_test_weather_domains.py`) |
| ⏳ | Eval FP32 trên 6 domain official |
| ⏳ | Calib + **FP16/INT8** → Δ mAP theo domain |
| ⏳ | Orin: latency / power |

### Active train/test files

Train/val:

```text
configs/cctsdb2021_train.yaml      # train=16356, val/test=1500 official positive test
configs/cctsdb2021_test_full.yaml  # full positive test
```

Domain test configs:

```text
configs/cctsdb2021_test_sunny.yaml    # 400
configs/cctsdb2021_test_cloud.yaml    # 300
configs/cctsdb2021_test_rain.yaml     # 160
configs/cctsdb2021_test_snow.yaml     # 100
configs/cctsdb2021_test_foggy.yaml    # 40
configs/cctsdb2021_test_night.yaml    # 500
configs/cctsdb2021_test_daylike.yaml  # 1000 = non-night aggregate
```

Chi tiết: [`docs/DATASET_CCTSDB2021.md`](docs/DATASET_CCTSDB2021.md).

---

## 5. Repo (gọn)

```
configs/     cctsdb2021_train.yaml, cctsdb2021_test_*.yaml
data/        raw/CCTSDB2021, processed/cctsdb2021_full, cctsdb2021_test_*
scripts/     train_cctsdb.py
docs/        DATASET_CCTSDB2021.md (chính), plan archive
```

### Server sync

Code/config đi bằng git:

```bash
git pull
```

Dataset đi bằng Drive/gdown vì `data/` không commit vào repo.

### Lệnh chính

```bash
python scripts/train_cctsdb.py --model yolo11n.pt \
  --data configs/cctsdb2021_train.yaml \
  --epochs 100 --batch 64 --workers 8 --name yolo11n_cctsdb_full

# Nếu server/ổ mạng chậm, stage data sang /tmp trong cùng train script:
python scripts/train_cctsdb.py --model yolo11n.pt \
  --stage-to /tmp/cctsdb2021_full \
  --epochs 100 --batch 64 --workers 8 --name yolo11n_cctsdb_full
```

Server git (tránh libffi): `ntd-git push|pull|status`  
Torch + driver 535: **cu121** (không cu130).

---

## 6. Cite / license

CCTSDB2021 — GPL-3.0; cite papers trên [GitHub CCTSDB2021](https://github.com/csust7zhangjm/CCTSDB2021).
