# [ARCHIVE] Plan cũ — multi-dataset night TSD + INT8

> **Trạng thái: LƯU TRỮ.**  
> Plan active hiện tại chỉ còn **CCTSDB2021** — xem [`DATASET_CCTSDB2021.md`](DATASET_CCTSDB2021.md) và [`../README.md`](../README.md).  
> File này giữ để tham chiếu lịch sử (venue, gap INT8+night, Orin/Thor). **Không** mở rộng CNTSSS / INTSD / TT100K trong protocol mới.

---

## Đã loại khỏi plan active

| Dataset | Lý do |
|---|---|
| CNTSSS | Nhỏ hơn CCTSDB; bỏ trục chính |
| INTSD | On-request / Bài 2 — bỏ |
| TT100K-night | Synthetic không release — bỏ |
| CURE-TSD, NTS-YOLO | Phụ — bỏ |

## Vẫn giữ ý tưởng kỹ thuật (áp dụng trên CCTSDB)

- Quant FP32 → FP16 → INT8 PTQ  
- Đo mAP theo **điều kiện** (test weather: night vs non-night)  
- Latency–power trên Jetson Orin (khi có board)  
- Calib day / night / mixed cho PTQ (ảnh lấy từ CCTSDB weather / heuristic, **không** bắt buộc CNTSSS)

## Venue (tham khảo)

VEHITS 17/11/2026 · WACV’27 workshops · ITSC 2027 · Journal TIM / JRTIP / JSA (Bài 2 nếu còn)

## CCTSDB số ảnh (chính thức)

- Train **16 356** + test positive **1 500** = **17 856**  
- Negative 500 (optional)  
- Weather XML: **chỉ** 1500 test (night ~500)

Drive: https://drive.google.com/drive/folders/14Km2W-5hbixXDfz7WSqW_Rx7O5m8ZMFn
