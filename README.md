# Quantized Night-Time Traffic Sign Detection (Edge / Orin)

Nghiên cứu **phát hiện biển báo giao thông ban đêm** + (mục tiêu Bài 1) **đo accuracy–latency–power khi lượng tử hóa INT8** trên Jetson.

Kế hoạch 2 bài chi tiết: [`docs/README_nighttime_2papers.md`](docs/README_nighttime_2papers.md).

---

## 1. Ý tưởng / lý thuyết (đang hướng tới)

### Câu hỏi nghiên cứu

Khi nén model từ **FP32 → FP16 → INT8 (PTQ)**:

1. **Accuracy (mAP) giảm ở đâu?** — dataset nào, điều kiện đêm/ngày, lớp nào (prohibitory / mandatory / warning)?
2. **Đổi lại được gì trên edge?** — ms/frame, W, J/frame, FPS/W (Jetson Orin).

### Quant không phải gì?

| Không phải | Mà là |
|---|---|
| Xóa bớt hyperparameter (lr, batch…) | Cùng kiến trúc YOLO, **số bit** weight/activation giảm |
| Chỉ “tối ưu tốc độ trên giấy” | Đo **mAP thực** + **latency/power** trên device |

### Gap paper (claim đúng)

Chưa có bài kết hợp đủ ba: **(a) nighttime TSD + (b) INT8/quant + (c) edge power/latency**.  
Phân biệt: YOLO-LLTS = night, chưa quant/power; IET ITS 2026 = quant edge, chủ yếu ban ngày / không biển báo đêm.

| Bài | Trục | Deadline gợi ý |
|---|---|---|
| **Bài 1** hội nghị | PTQ INT8 + accuracy–latency–power trên **Orin** | VEHITS 17/11/2026 · WACV’27 WS · ITSC 2027 |
| **Bài 2** journal | QAT, Orin vs Thor, domain×quant đầy đủ, streaming | Sau Bài 1 |

---

## 2. Dữ liệu

| Dataset | Quy mô | Tính chất | Vai trò hiện tại |
|---|---|---|---|
| **CCTSDB2021 full** | Train ~**16 356** · Test ~**1 500** | Ngày + đêm + thời tiết (mixed), 3 lớp | **Train chính** (large-scale) |
| **CCTSDB2021 night** | ~**500** test | Chỉ đêm (XML weather) | Cross / night subset |
| **CNTSSS** | Train **3 276** · Test **786** | **Chỉ đêm** thật, 3 lớp | Train ablation night-only + **cross-eval đêm** |

Cùng taxonomy 3 super-class:

| id | Tên |
|---|---|
| 0 | prohibitory |
| 1 | mandatory |
| 2 | warning |

Data **không** commit git (`data/` gitignore). Trên server: `data/raw/`, `data/processed/`.

---

## 3. Protocol thí nghiệm (hiện tại vs README gốc)

| | README gốc (plan) | **Đang chạy thực tế** |
|---|---|---|
| Train main | CNTSSS night 3 276 | **CCTSDB full ~16k** (mixed) |
| Models | YOLO11n + YOLOv8n | + **YOLO26n** |
| Precision đã đo | FP32 / FP16 / INT8 | **Mới FP32** (eval JSON) |
| Test | CNTSSS + CCTSDB-night | CCTSDB-test + **CNTSSS cross** (+ night subset tùy run) |
| Device quant | Jetson Orin | **Chưa** — train/eval trên server GPU (RTX 8000 / 3090) |

Paper cần **nói rõ** protocol main (CCTSDB full) vs ablation (CNTSSS night-only).

---

## 4. Kết quả hiện tại (FP32) — đã push `runs/eval_*.json`

Weights (trên server train):

```text
runs/detect/runs/detect/yolo11n_cctsdb_full/weights/best.pt
runs/detect/runs/detect/yolov8n_cctsdb_full/weights/best.pt
runs/detect/runs/detect/yolo26n_cctsdb_full/weights/best.pt
runs/detect/runs/detect/yolo11n_cntsss/weights/best.pt   # ablation night-only
runs/detect/runs/detect/yolov8n_cntsss/weights/best.pt
```

### 4.1. Train **CCTSDB full** → test **CCTSDB** (in-domain)

| Model | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| YOLOv8n | **0.781** | 0.492 | 0.875 | **0.733** |
| YOLO11n | 0.780 | 0.493 | 0.879 | 0.712 |
| YOLO26n | 0.780 | **0.500** | **0.890** | 0.714 |

→ Ba nano **tương đương, tốt** trên miền train.

Per-class AP50 (in-domain, 0/1/2):

| Model | prohibitory | mandatory | warning |
|---|---|---|---|
| v8n | 0.717 | 0.767 | **0.860** |
| 11n | 0.704 | 0.792 | 0.845 |
| 26n | 0.703 | 0.786 | 0.852 |

### 4.2. Train **CCTSDB full** → test **CNTSSS night** (cross-domain)

| Model | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| YOLOv8n | **0.233** | **0.134** | 0.296 | 0.259 |
| YOLO11n | 0.223 | 0.132 | 0.321 | 0.223 |
| YOLO26n | 0.212 | 0.120 | 0.277 | 0.222 |

Per-class AP50 trên CNTSSS:

| Model | prohibitory | mandatory | warning |
|---|---|---|---|
| v8n | **0.007** | 0.018 | **0.673** |
| 11n | 0.010 | 0.013 | 0.646 |
| 26n | 0.012 | 0.010 | 0.613 |

### 4.3. Finding chính (FP32)

```text
CCTSDB in-domain  mAP50 ~0.78
        ↓ cross
CNTSSS night      mAP50 ~0.22   (−~0.56)
  · prohibitory / mandatory ~ sập (~0.01)
  · warning còn ~0.61–0.67
```

| Finding | Ý nghĩa cho paper |
|---|---|
| Domain gap đêm rất lớn | Mixed-day CCTSDB **không** chuyển tốt sang CNTSSS night |
| Class 0/1 yếu trên đêm cross | Điểm yếu **trước quant** — INT8 có thể làm tệ thêm |
| 3 model nano gần nhau | Chọn 1 main + control; không khác biệt kiến trúc lớn ở FP32 |
| Train CNTSSS only (trước) mAP50 ~0.67–0.68 trên CNTSSS | Night-only train **cần** nếu muốn giỏi CNTSSS |

### 4.4. Chưa có (README Bài 1 còn thiếu)

- [ ] Eval **CCTSDB-night** (~500) cho 3 model full (nếu chưa đủ file)
- [ ] **FP16 / INT8** mAP (TensorRT, calib day/night/mixed)
- [ ] **Latency / power** trên **Jetson Orin**
- [ ] So YOLO-LLTS FPS trên Orin

---

## 5. Trạng thái pipeline

| Hạng mục | Status |
|---|---|
| Data CNTSSS + CCTSDB + night extract | ✅ |
| Train CCTSDB full × 3 models (100 ep) | ✅ |
| Train CNTSSS × 2 models | ✅ |
| Eval FP32 in-domain + cross CNTSSS | ✅ (`runs/eval_*.json`) |
| Calib sets day/night/mixed | ⏳ |
| Orin INT8 + power | ❌ |
| Viết paper / nộp | ⏳ |

---

## 6. Repo layout

```
configs/          # dataset yaml (+ cctsdb2021_full_local trên server)
data/raw/         # CNTSSS, CCTSDB2021 (gitignore)
data/processed/   # cctsdb full / night, calibration
docs/             # plan 2 papers, server, bottleneck
scripts/          # train, eval, stage local, export TRT, bench
runs/             # weights + eval JSON (một phần eval đã track)
```

---

## 7. Lệnh thường dùng (server)

```bash
conda activate nighttime-tsd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd ~/Dung_TDTU/nighttime-tsd

# Stage data local (tránh NFS)
python scripts/stage_data_local.py --dst /tmp/cctsdb2021_full

# Train (SERVER-02: workers 6–8, batch 64, RTX 8000)
python scripts/train_baseline.py \
  --model yolo11n.pt \
  --data configs/cctsdb2021_full_local.yaml \
  --epochs 100 --batch 64 --workers 8 --cache ram \
  --device 0 --name yolo11n_cctsdb_full

# Eval
python scripts/eval_map.py --weights runs/detect/runs/detect/yolo11n_cctsdb_full/weights/best.pt \
  --data configs/cctsdb2021_full_local.yaml --batch 32 --out runs/eval_yolo11n_cctsdb_full.json
python scripts/eval_map.py --weights .../best.pt --data configs/cntsss.yaml --batch 32 \
  --out runs/eval_yolo11n_cctsdb_full_on_cntsss.json

# Git trên SERVER-02 (tránh libffi): ~/bin/ntd-git push | pull | status
```

Torch trên driver 535: dùng **cu121**, không cu130.

Chi tiết server / nghẽn I/O:

- [`docs/SERVER_TRAIN.md`](docs/SERVER_TRAIN.md) · [`docs/CONDA_ENV.md`](docs/CONDA_ENV.md)
- [`docs/BOTTLENECK_TRAIN_3090.md`](docs/BOTTLENECK_TRAIN_3090.md)
- [`docs/CCTSDB_FULL_TRAIN.md`](docs/CCTSDB_FULL_TRAIN.md)

---

## 8. Bước tiếp theo (theo README)

1. **Accuracy:** eval thêm CCTSDB-night; (tuỳ) bảng train-CNTSSS vs train-CCTSDB.  
2. **Calib:** `build_calibration_sets.py` day/night/mixed.  
3. **Orin:** export FP16/INT8 + mAP + power/latency → claim quant.  
4. **Viết:** Intro gap 3 yếu tố + Results FP32 domain gap + placeholder quant.

---

## 9. Citation / license

- YOLO-LLTS / CNTSSS — IEEE TIM 2025, [github.com/linzy88/YOLO-LLTS](https://github.com/linzy88/YOLO-LLTS)  
- CCTSDB2021 — GPL-3.0, [github.com/csust7zhangjm/CCTSDB2021](https://github.com/csust7zhangjm/CCTSDB2021)  
- CNTSSS: GitHub không có license file — nên xin phép tác giả trước khi redistrib derivative  

---

*README cập nhật theo trạng thái thực nghiệm FP32 (eval push 2026-07). Plan gốc đầy đủ vẫn nằm ở `docs/README_nighttime_2papers.md`.*
