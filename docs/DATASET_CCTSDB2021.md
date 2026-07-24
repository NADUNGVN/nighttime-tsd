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

## 2. Weather/light domain trên **test 1500** (bắt buộc cho quant)

| Split | Weather/light domain? |
|---|---|
| Train 16 356 | **Không** metadata chính thức (mixed) |
| Test 1 500 | Weather XML chính thức, 6 domain |

### Weather chính thức

| Domain | Số ảnh |
|---|---:|
| sunny | 400 |
| cloud | 300 |
| rain | 160 |
| snow | 100 |
| foggy | 40 |
| night | 500 |
| **Tổng** | **1500** |

`daylike` là aggregate phụ: `sunny + cloud + rain + snow + foggy = 1000`.
`foggy` chỉ có 40 ảnh nên report kèm caveat.

### Domain official đã được tách sẵn

Trong bundle train server, các split này đã được chuẩn bị sẵn dưới `data/processed/`.
Script tạo split được lưu ở `scripts_legacy/split_cctsdb_test_weather_domains.py` để tham chiếu, không cần chạy khi train.

Output:

| Path | Nội dung |
|---|---|
| `data/processed/cctsdb2021_test_sunny/` | 400 ảnh + labels YOLO |
| `data/processed/cctsdb2021_test_cloud/` | 300 ảnh + labels YOLO |
| `data/processed/cctsdb2021_test_rain/` | 160 ảnh + labels YOLO |
| `data/processed/cctsdb2021_test_snow/` | 100 ảnh + labels YOLO |
| `data/processed/cctsdb2021_test_foggy/` | 40 ảnh + labels YOLO |
| `data/processed/cctsdb2021_test_night/` | 500 ảnh + labels YOLO |
| `data/processed/cctsdb2021_test_daylike/` | 1000 ảnh + labels YOLO |
| `data/processed/cctsdb2021_test_weather_manifest.csv` | stem, domain, aggregate, source_xml |
| `configs/cctsdb2021_test_{domain}.yaml` | eval từng domain |

Eval:

```bash
python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_test_sunny.yaml
python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_test_cloud.yaml
python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_test_rain.yaml
python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_test_snow.yaml
python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_test_foggy.yaml
python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_test_night.yaml
```

Sau quant: so Δ mAP **theo từng domain** và report thêm **daylike vs night** nếu cần.

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
        └─ tách weather: sunny/cloud/rain/snow/foggy/night → phân tích quant
Models: YOLO11n, YOLOv8n, YOLO26n (FP32 done)
Next:   FP16/INT8 + (Orin) power/latency
```
