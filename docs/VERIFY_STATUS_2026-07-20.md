# Verify status — 20/07/2026

Đối chiếu lại các claim quan trọng trong `README_nighttime_2papers.md` trước khi execute.

| Claim | Status | Evidence / note |
|---|---|---|
| CNTSSS tải được (Drive) | ✅ Repo live | [linzy88/YOLO-LLTS](https://github.com/linzy88/YOLO-LLTS) · file id `1A-7t-Wb5rjUZslUJ_1tltlUUvtSxBXdX` · contact `linzy88@mail2.sysu.edu.cn` |
| CCTSDB2021 tải được | ✅ Repo live | [csust7zhangjm/CCTSDB2021](https://github.com/csust7zhangjm/CCTSDB2021) · GPL-3.0 · Drive folder + Baidu · có gói weather/environment XML |
| INTSD full public | ⚠️ On request | [project page](https://adityamishra-ml.github.io/INTSD/) · arXiv 2511.17183 · email tuần 1 |
| VEHITS 2027 deadline 17/11/2026 | ✅ | [vehits.scitevents.org](https://vehits.scitevents.org/) Regular Paper Submission: November 17, 2026 |
| Ultralytics INT8 flag `quantize=8` | ✅ | [docs TensorRT export](https://docs.ultralytics.com/integrations/tensorrt/) · calibrate với `data=` + `fraction` |
| Workspace code sẵn sàng | ✅ scaffold | Scripts + configs tạo 20/07/2026 |
| Dataset files on disk | ✅ CNTSSS | `data/raw/CNTSSS` verified 3276/786; CCTSDB2021 chưa tải |
| Jetson Orin env verified | ❓ manual | Cần chạy trên máy Orin (ngoài repo này) |

## Gap framing (giữ đúng giọng Intro)

Không claim “chưa ai làm night TSD” hay “chưa ai quantize YOLO trên Orin”.  
Claim đúng: **chưa có bài kết hợp đủ (a) night TSD + (b) INT8/quant + (c) edge power/latency**, và phân biệt với IET ITS 2026 (day, traffic light/obstacle) + YOLO-LLTS (night, no TRT/quant/power).

## Cascade nộp Bài 1

1. Thí nghiệm T8–T10/2026  
2. **WACV’27 workshop** (list ~sau 11/08) **hoặc VEHITS 17/11/2026**  
3. **ITSC 2027** (~T2–T3/2027) là đích flagship  
4. **Không rush RIVF 31/07** (11 ngày, chưa có thí nghiệm)

## Ranh giới 2 bài (nhắc nhanh)

| Bài 1 | Bài 2 |
|---|---|
| PTQ INT8, Orin, 3-class, CNTSSS+CCTSDB night | QAT, Orin vs Thor, streaming/thermal, YOLO26n, INTSD |
