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

## 2. Sáng / tối — chỉ biết trên **test 1500**

| Split | Biết day/night? |
|---|---|
| Train 16 356 | **Không** (mixed, không metadata) |
| Test 1 500 | **Có** — folder weather: night / sunny / foggy / rain / cloud / snow |

Ước lượng đã gặp khi extract: **night ~500**, **non-night ~1000** (đếm lại bằng script trên server).

Dùng cho quant: eval **test-night** vs **test-day (non-night)** sau FP16/INT8.

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
