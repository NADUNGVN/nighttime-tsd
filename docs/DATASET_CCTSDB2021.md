# Dataset chính: CCTSDB2021 (duy nhất trong plan hiện tại)

Drive chính thức:  
https://drive.google.com/drive/folders/14Km2W-5hbixXDfz7WSqW_Rx7O5m8ZMFn

Repo: https://github.com/csust7zhangjm/CCTSDB2021 · **GPL-3.0**

---

## 1. Vì sao Drive “nhiều hơn 16k”?

**16 356 chỉ là tập train.** Cả gói Drive gồm nhiều zip **cùng một benchmark**, không phải 16k ảnh thôi.

| Thành phần (zip / gói) | Nội dung | Số ảnh (chính thức) |
|---|---|---|
| `train_img` + `train_labels` | Ảnh + nhãn train | **16 356** (id 00000–18991) |
| `test_img` + labels/XML | Ảnh test **positive** | **1 500** (id 18992–20491) |
| `negative samples` | Ảnh không biển (optional) | **500** |
| **Tổng ảnh có nhãn positive (train+test)** | | **17 856** |
| **+ negative** | | **18 356** |
| `classification based on weather and environment` | **Chỉ gắn điều kiện cho 1 500 test** (night/sunny/…) | 1 500 XML (không thêm ảnh mới) |
| `classification based on size of traffic signs` | Gắn theo kích thước biển (test) | Không thêm ảnh mới |
| `xml` | VOC XML train+test | Trùng ảnh trên |

→ Trên Drive thấy **nhiều file/zip** vì: ảnh train, ảnh test, nhãn txt, XML, phân loại weather, size, negative — **không** phải 30k ảnh train khác.

Số repo HELP: *“17856 images in training set and positive sample test set”* = **16356 + 1500**.

---

## 2. Sáng / tối trên **test 1500** (bổ sung cho quant)

| Split | Day/night? |
|---|---|
| Train 16 356 | **Không** metadata chính thức (mixed) |
| Test 1 500 | Weather XML + script cân bằng 50/50 |

### Weather chính thức (không đủ 50–50)

Thường: **night ~500** · **non-night ~1000** (sunny/fog/rain/cloud/snow).

### Script chuẩn hóa 50% đêm / 50% sáng (trên test)

```bash
# Xem thống kê weather + kế hoạch balance
python scripts/split_cctsdb_test_day_night.py --list-only

# Tạo 750 night + 750 day (từ 1500 test)
# - night: weather night + (nếu thiếu) ảnh tối nhất theo mean luminance
# - day:   ảnh sáng nhất trong phần còn lại
python scripts/split_cctsdb_test_day_night.py --balance 0.5 --target-total 1500
```

Output:

| Path | Nội dung |
|---|---|
| `data/processed/cctsdb2021_test_night/` | ~750 ảnh + labels YOLO |
| `data/processed/cctsdb2021_test_day/` | ~750 ảnh + labels YOLO |
| `data/processed/cctsdb2021_test_day_night_manifest.csv` | stem, split, source (weather/heuristic), luminance |
| `configs/cctsdb2021_test_night.yaml` | eval night |
| `configs/cctsdb2021_test_day.yaml` | eval day |

Eval:

```bash
python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_test_day.yaml
python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_test_night.yaml
```

Sau quant: so Δ mAP **day vs night** (và per-class).

---

## 3. 3 lớp (super-class)

| id | Tên |
|---|---|
| 0 | prohibitory |
| 1 | mandatory |
| 2 | warning |

---

## 4. Đã loại khỏi plan (dataset nhỏ / không dùng)

| Dataset | Lý do bỏ khỏi plan active |
|---|---|
| CNTSSS (~4k đêm) | Nhỏ hơn CCTSDB; cross-eval đã xong — **không** còn trục chính |
| INTSD | On-request, 41 lớp, Bài 2 cũ — **bỏ** |
| TT100K / TT100K-night | Không đêm thật / synthetic không release — **bỏ** |
| CURE-TSD, NTS-YOLO | Phụ — **bỏ** |

Weights/eval CNTSSS cũ có thể còn trong `runs/` (lịch sử) — **không** mở rộng protocol mới.

---

## 5. Protocol repo (chốt)

```text
Train:  CCTSDB train 16 356 (mixed)
Test:   CCTSDB test 1 500 (in-domain)
        └─ tách weather: night ~500 | day-like ~1000  → phân tích quant
Models: YOLO11n, YOLOv8n, YOLO26n (FP32 done)
Next:   FP16/INT8 + (Orin) power/latency
```
