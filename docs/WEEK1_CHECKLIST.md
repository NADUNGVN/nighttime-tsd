# Checklist tuần 1 — Nighttime INT8 TSD

> Bắt đầu mục tiêu: **~01/08/2026**. Hôm nay (đóng gói plan): **20/07/2026**.  
> Cập nhật cột Status: `TODO` → `DOING` → `DONE` / `BLOCKED`.

| # | Việc | Status | Ghi chú |
|---|---|---|---|
| 1 | Tải CNTSSS.zip (~885 MB) + giải nén | DONE | 20/07/2026 — 928 MB, `data/raw/CNTSSS/` |
| 2 | Verify format YOLO + đếm 3276/786 + spot-check nhãn | DONE | PASS: 3276/786, 0 bad boxes, class ~64.8/22.5/12.7% |
| 3 | Tải CCTSDB2021 (Drive/Baidu) → `data/raw/CCTSDB2021/` | TODO | [GitHub](https://github.com/csust7zhangjm/CCTSDB2021) |
| 4 | Trích night-test từ XML điều kiện (~500) | TODO | `python scripts/extract_cctsdb_night.py` |
| 5 | Email YOLO-LLTS (license CNTSSS + TT100K-night) | TODO | Draft: `docs/emails/email_yolo_llts_cntsss.md` → **linzy88@mail2.sysu.edu.cn** |
| 6 | Email INTSD (full 41 lớp, Bài 2) | TODO | Draft: `docs/emails/email_intsd.md` · [project](https://adityamishra-ml.github.io/INTSD/) |
| 7 | Theo dõi CFP: ITSC 2027, WACV’27 workshops, VEHITS | TODO | VEHITS regular: **17/11/2026** (đã xác minh) |
| 8 | Xác nhận JetPack/TensorRT trên Orin | TODO | `jetson_release` / `dpkg -l \| grep tensorrt` |
| 9 | Smoke export INT8 1 model (`quantize=8`, calibrate on device) | TODO | `python scripts/export_tensorrt.py --precision int8 ...` |
| 10 | (Bài 2 sớm) Thor: ModelOpt + stronglyTyped thử | TODO | Không block Bài 1 |

## Verify gates (không qua thì không cam kết full pipeline)

- [ ] CNTSSS: train≈3276, test≈786, class ids ∈ {0,1,2}, <1% bad boxes  
- [ ] CCTSDB night: ≥300 ảnh khớp (target ~500)  
- [ ] 1 lần `yolo11n` train 3 epoch smoke test chạy xong  
- [ ] 1 engine FP16 + 1 engine INT8 export trên Orin  

## Lệnh tuần 1 (copy-paste)

```bash
# Host (Windows PowerShell / Linux)
cd D:\Research\paper
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt

python scripts/download_cntsss.py --extract
python scripts/verify_cntsss.py

# Sau khi tự unpack CCTSDB2021:
python scripts/extract_cctsdb_night.py
python scripts/build_calibration_sets.py --n 256

# Smoke train (GPU)
python scripts/train_baseline.py --model yolo11n.pt --epochs 3 --name smoke_yolo11n
```

```bash
# Trên Jetson Orin (sau khi có best.pt)
python scripts/export_tensorrt.py --weights runs/detect/.../best.pt --precision fp16
python scripts/export_tensorrt.py --weights runs/detect/.../best.pt --precision int8 --calib configs/calibration_night.yaml
python scripts/bench_power.py --engine best.engine --source data/raw/CNTSSS/test/images --power-mode MAXN
```
