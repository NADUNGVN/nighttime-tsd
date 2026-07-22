# Quantized Traffic Sign Detection on Edge (CCTSDB2021)

Bài hướng tới: **YOLO + quant (FP32→FP16→INT8)** trên **CCTSDB2021**, đo **accuracy theo điều kiện sáng/tối (test)** và sau này **latency–power trên Jetson Orin**.

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
| Weather split (test only) | 1 500 XML | **night ~500** · **non-night ~1000** (sunny/fog/rain/cloud/snow) |

**16k = chỉ train.** Cả bộ labeled positive ≈ **17.9k** ảnh (+ zip phụ: XML, weather, size — không nhân đôi số ảnh train).

### 3 lớp

`0` prohibitory · `1` mandatory · `2` warning

### Đã loại khỏi plan

CNTSSS, INTSD, TT100K(-night), CURE-TSD, NTS-YOLO — quá nhỏ / không public / lệch trục. Không mở rộng protocol mới trên các bộ này.

---

## 2. Câu hỏi nghiên cứu (quant)

Train: **16 356 mixed** (không biết % night trong train).

Đo sau quant trên **test 1500 đã biết sáng/tối**:

| Subset test | ~Số | Mục đích |
|---|---|---|
| Full test | 1500 | mAP tổng in-domain |
| **Night** | ~500 | mAP tối |
| **Day-like** (non-night) | ~1000 | mAP sáng / điều kiện khác đêm |

→ INT8 làm **mAP_night** vs **mAP_day** giảm bao nhiêu, lớp nào (0/1/2) yếu hơn — **không** phải “giảm lr/batch”.

---

## 3. Kết quả FP32 hiện tại (train CCTSDB full)

| Model | Test full mAP50 | Test night mAP50 | Ghi chú |
|---|---|---|---|
| YOLOv8n | **0.781** | 0.265 | |
| YOLO11n | 0.780 | 0.241 | |
| YOLO26n | 0.780 | 0.253 | mAP50-95 full ~0.50 |

Finding: in-domain tốt (~0.78); **subset night trong cùng CCTSDB đã rơi mạnh** (~0.24–0.26); per-class night: class 0/1 ~0, warning còn cao.

JSON: `runs/eval_yolo*_cctsdb_full.json`, `runs/eval_*_on_cctsdb_night.json`.

---

## 4. Pipeline còn lại

| Bước | Việc |
|---|---|
| ✅ | Train 11n / 8n / 26n trên CCTSDB full |
| ✅ | Eval test full + night subset |
| ⏳ | Tách eval **test day-like** (~1000) tường minh |
| ⏳ | Calib + **FP16/INT8** → Δ mAP day vs night |
| ⏳ | Orin: latency / power |

---

## 5. Repo (gọn)

```
configs/     cctsdb2021_full.yaml, cctsdb2021_night.yaml, calibration_*.yaml
data/        raw/CCTSDB2021, processed/cctsdb2021_full, cctsdb2021_night
scripts/     prepare_cctsdb_full, train_baseline, eval_map, stage_data_local, export_tensorrt, …
docs/        DATASET_CCTSDB2021.md (chính), plan archive
```

### Lệnh chính

```bash
python scripts/prepare_cctsdb_full.py
python scripts/stage_data_local.py --dst /tmp/cctsdb2021_full
python scripts/train_baseline.py --model yolo11n.pt \
  --data configs/cctsdb2021_full_local.yaml \
  --epochs 100 --batch 64 --workers 8 --cache ram --name yolo11n_cctsdb_full
python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_full_local.yaml
python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_night.yaml
```

Server git (tránh libffi): `ntd-git push|pull|status`  
Torch + driver 535: **cu121** (không cu130).

---

## 6. Cite / license

CCTSDB2021 — GPL-3.0; cite papers trên [GitHub CCTSDB2021](https://github.com/csust7zhangjm/CCTSDB2021).
