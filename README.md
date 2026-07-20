# Quantized Night-Time Traffic Sign Detection (Orin / Thor)

Hai bài theo kế hoạch trong [`docs/README_nighttime_2papers.md`](docs/README_nighttime_2papers.md):

| | Bài 1 (hội nghị) | Bài 2 (journal) |
|---|---|---|
| **Trục** | INT8 PTQ night TSD + accuracy–latency–power trên **Jetson AGX Orin** | QAT + **Orin vs Thor** + domain-gap × quant + streaming/thermal |
| **Data** | CNTSSS (3-class night) + CCTSDB2021-night | + INTSD 41-class (nếu xin được) |
| **Deadline gần** | VEHITS **17/11/2026** · WACV’27 workshops · ITSC 2027 | Sau Bài 1 (~T5/2027) |

## Repo layout

```
configs/          # YOLO data yaml + calibration yaml
data/raw/         # CNTSSS, CCTSDB2021 (gitignored)
data/processed/   # night subset, calibration sets
docs/             # plan, checklist, email drafts
scripts/          # download, verify, train, export, bench
runs/             # ultralytics outputs (gitignored)
```

## Quick start (tuần 1)

```bash
pip install -r requirements.txt
python scripts/download_cntsss.py --extract
python scripts/verify_cntsss.py
# unpack CCTSDB2021 into data/raw/CCTSDB2021/ then:
python scripts/extract_cctsdb_night.py
python scripts/build_calibration_sets.py --n 256
```

### Server RTX 3090 — conda + load + train

Chi tiết: [`docs/SERVER_TRAIN.md`](docs/SERVER_TRAIN.md) · [`docs/CONDA_ENV.md`](docs/CONDA_ENV.md)

```bash
# Env riêng (không đụng env AI khác): nighttime-tsd
bash scripts/setup_conda_env.sh --with-torch-cu124 --install-htop
conda activate nighttime-tsd
nvitop   # GPU
htop     # CPU/RAM (system)

# Data (không có trong git)
python scripts/download_cntsss.py --extract && python scripts/verify_cntsss.py

# Load pretrained
python scripts/load_model.py --model yolo11n.pt --info

# Train
python scripts/train_baseline.py --model yolo11n.pt --batch 64 --epochs 100 --workers 8 --name yolo11n_cntsss
```

Checklist: [`docs/WEEK1_CHECKLIST.md`](docs/WEEK1_CHECKLIST.md) · Verify: [`docs/VERIFY_STATUS_2026-07-20.md`](docs/VERIFY_STATUS_2026-07-20.md)

## Experiment matrix (Bài 1)

- Models: YOLO11n (main), YOLOv8n (control)  
- Precision: FP32, FP16, INT8-PTQ (TensorRT, calibrate on Orin)  
- Calibration: day / night / mixed  
- Hardware: AGX Orin @ MAXN + 30W  
- Metrics: mAP@50, mAP@50-95, per-class, ms/frame, W, J/frame, FPS/W  

```bash
python scripts/train_baseline.py --model yolo11n.pt --epochs 100
python scripts/export_tensorrt.py --weights .../best.pt --precision int8 --calib configs/calibration_night.yaml
python scripts/bench_power.py --engine .../best.engine --source data/raw/CNTSSS/test/images
python scripts/eval_map.py --weights .../best.pt --data configs/cntsss.yaml
python scripts/eval_map.py --weights .../best.pt --data configs/cctsdb2021_night.yaml
```

## Citation anchors

- YOLO-LLTS / CNTSSS — IEEE TIM 2025, [github.com/linzy88/YOLO-LLTS](https://github.com/linzy88/YOLO-LLTS)  
- CCTSDB2021 — [github.com/csust7zhangjm/CCTSDB2021](https://github.com/csust7zhangjm/CCTSDB2021)  
- Ultralytics Jetson / TensorRT — [docs.ultralytics.com](https://docs.ultralytics.com/guides/nvidia-jetson/)  

## License note

CNTSSS has no explicit license on GitHub — send the week-1 permission email before public redistribution of derivatives. CCTSDB2021 is GPL-3.0.
