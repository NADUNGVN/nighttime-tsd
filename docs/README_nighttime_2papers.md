# Kế hoạch 2 bài: Nighttime Traffic Sign Detection + INT8 trên Jetson Orin/Thor

> Đóng gói ngày 20/07/2026 từ Gap 1 trong `README_conference_gaps.md`. **Mọi fact quan trọng dưới đây đã được kiểm chứng trực tiếp trên web ngày 20/07/2026** (4 luồng kiểm chứng song song: dataset, độ mở của gap, deadline hội nghị, journal đích). Chiến lược: **Bài 1** = bài hội nghị 6–10 trang (PTQ INT8, Orin, 3-class night benchmark) → **Bài 2** = bản mở rộng journal Q1/Q2 (QAT, Orin vs Thor, domain-gap × quantization, streaming/thermal).

---

## PHẦN A — KẾT QUẢ THẨM ĐỊNH (20/07/2026)

### A1. Dataset: KHẢ THI ✅ (có một ràng buộc quan trọng)

| Dataset | Trạng thái | Chi tiết đã xác minh |
|---|---|---|
| **CNTSSS** (đêm thật, TQ) | ✅ Tải được ngay | Repo `github.com/linzy88/YOLO-LLTS` tồn tại; link Google Drive **sống, tải ẩn danh được**, file `CNTSSS.zip` 885 MB. 4.062 ảnh đêm (3.276 train / 13 thành phố; 786 test / 4 thành phố giữ riêng: Thành Đô, Thượng Hải, Thâm Quyến, Thiên Tân). Dusk→đêm sâu, có mưa, đô thị/cao tốc/nông thôn. Cấu trúc thư mục YOLO-format. |
| **CCTSDB2021 night subset** | ✅ Tải được | Repo chính thức `csust7zhangjm/CCTSDB2021`, GPL-3.0. ~700 ảnh đêm trong 16.356 train; test có gói XML **gán nhãn theo điều kiện thời tiết/ánh sáng** → trích được ~500 ảnh test đêm. Cùng taxonomy 3 lớp với CNTSSS → ghép làm cross-dataset sạch. |
| **INTSD** (đêm, Ấn Độ, 41 lớp) | ⚠️ Mới mở một phần | v3 arXiv (14/07/2026) tuyên bố "publicly available", có project page — nhưng trang download ghi *"full dataset will be made available soon... access on request"*. Hiện chỉ tải được subset + annotations.json. **→ Email tác giả NGAY tuần 1** để xin bản full cho Bài 2 (fine-grained 41 lớp). |
| **TT100K** | ❌ Không có đêm thật | Chính YOLO-LLTS khẳng định TT100K/GTSRB không có ảnh đêm; TT100K-night/GTSDB-night của họ là **CycleGAN synthetic và KHÔNG được release** → nếu dùng phải tự tái tạo (có thể làm ablation, nói rõ là synthetic tự dựng). |
| CURE-TSD, NTS-YOLO (figshare) | Phụ | CURE-TSD: darkening synthetic 5 mức, xin qua form. NTS-YOLO: CC BY 4.0 nhưng chỉ đóng gói lại ảnh đêm CCTSDB. |

**Ràng buộc quan trọng nhất:** CNTSSS chỉ có **3 super-class** (prohibitory ~64%, mandatory ~22%, warning ~14%) — KHÔNG có nhãn fine-grained. → Bài 1 phải định vị là **3-class nighttime benchmark** (nhất quán với CCTSDB2021, hợp lệ và đủ cho hội nghị); fine-grained 41 lớp để dành cho Bài 2 nếu xin được INTSD full.

**License:** CNTSSS **không có license** (GitHub `license: null`) — dùng nghiên cứu de facto ổn (tác giả công bố "code and dataset are available") nhưng nên email xin phép chính thức, tiện thể hỏi luôn file TT100K-night synthetic.

### A2. Gap: VẪN MỞ ✅ (kiểm chứng đối kháng 13 truy vấn, đến 20/07/2026)

Không có bài nào kết hợp đủ 3 yếu tố: **(a) nighttime/low-light TSD + (b) INT8/quantization + (c) edge power/latency**. Nhưng vòng vây đang khép — Introduction phải chuyển giọng từ "chưa ai làm gì" sang **"chưa ai kết hợp cả ba"**, và phải cite + phân biệt rõ với các bài gần nhất:

1. **IET ITS 2026 (DOI 10.1049/itr2.70135)** — ĐE DỌA GẦN NHẤT: quantized YOLOv8n INT8 TensorRT trên Orin Nano, ROS 2, phân tích deadline/latency. Nhưng làm **đèn giao thông + chướng ngại vật, ban ngày**. → Phân biệt: biển báo + ban đêm + đo power.
2. **YOLO-LLTS** (IEEE TIM 2025) — SOTA low-light TSD, 44,9 ms trên AGX Orin **không TensorRT**, không quantization, không power. Đã rà **cả 17 bài citing** — không bài nào thêm quantization/power/edge.
3. **INTSD/LENS-Net** (arXiv 2511.17183) — dataset + detector đêm, không quantization/edge/power.
4. **MVA 04/2026 (10.1007/s00138-026-01816-x)** — driver safety trên Orin Nano 15 FPS, theo abstract không night/INT8/power (paywall — nên skim PDF trước khi nộp).
5. **FEBG-YOLOv8s** (WEVJ 08/2025) — "edge deployment" chỉ trên giấy (params/GFLOPs), không device thật.

**Bắt buộc cite (mới, sau 05/2026):** IET ITS 10.1049/itr2.70135; Edge-TSR (arXiv 2606.17241 — phương pháp đo latency/thermal trên Orin Nano); YOLO-MAFF (IEEE T-ITS 01/07/2026). Cận kề: TS-1M, INTSD v2/v3, bài MVA, SciRep energy review (s41598-026-46453-6).

**13 truy vấn để rà lại trước khi nộp** (lưu trong phần E — checklist).

### A3. Phần cứng: CẢNH BÁO LỚN VỀ THOR ⚠️ → biến thành cơ hội

- Ultralytics hỗ trợ export TensorRT INT8 PTQ chính chủ (`format="engine"`, `quantize=8` — flag `int8=True` cũ đã đổi tên; calibration bằng `data=<yaml>` + `fraction`, **phải calibrate trên chính thiết bị đích**). Jetson guide chính thức phủ cả **AGX Thor (JetPack 7.2, Blackwell)** lẫn AGX Orin / Orin NX / Orin Nano Super với benchmark FP32/FP16/INT8. [docs.ultralytics.com/guides/nvidia-jetson/]
- **NHƯNG:** NVIDIA forum (03/2026, chưa fix): trên **Thor, INT8 gần như KHÔNG nhanh hơn FP16** (+2,7% qps, trong khi Orin +36%) — NVIDIA tái hiện được, quy cho node quantization của Myelin compiler; workaround: **ModelOpt explicit quantization + `--stronglyTyped`**. Ngoài ra FP8/FP4 trên Thor SM110 âm thầm rơi về FP32 (TensorRT issue #4590), DLA chưa hỗ trợ trong TensorRT 11.0.
- **Hệ quả cho kế hoạch:** Bài 1 chạy chính trên **Orin** (câu chuyện INT8 rõ ràng). Thor để dành cho Bài 2 với phát hiện "riêng có": *lợi ích INT8 KHÔNG chuyển giao qua thế hệ phần cứng mới* — một finding đo lường thực nghiệm mới mẻ, rất hợp journal đo lường (TIM-style), và anh là người có sẵn cả 2 máy để đo.

### A4. Venue hội nghị (deadline đã xác minh trên trang chính thức, 20/07/2026)

| Venue | Deadline | Ghi chú |
|---|---|---|
| RIVF 2026 (Hà Nội, 18–20/12) | **31/07/2026** — 11 ngày! | IEEE Xplore, 6pp. Chỉ khả thi nếu rush (không khuyến nghị — chưa có thí nghiệm). |
| WACV 2027 main R2 | 28/08/2026 | 8pp, Xplore + CVF. Quá gấp cho bài này. |
| SoICT 2026 (TP.HCM) | 16/09/2026 | Nay là Springer CCIS (không còn IEEE), Scopus. |
| VISAPP 2027 (Malta) | 15/09/2026 | Scopus/WoS, không Xplore. |
| **WACV 2027 workshops** | ~T10–T11/2026 (list công bố sau 11/08) | **Cơ hội tốt** — Xplore/CVF, hợp automotive vision. Theo dõi từ giữa T8. |
| **VEHITS 2027** (16–18/04/2027) | **17/11/2026 — ĐÃ XÁC MINH** | 12pp regular, Scopus/WoS (không Xplore). Fit hoàn hảo (vehicle + ITS). Deadline duy nhất đã xác minh nằm đúng cửa sổ. |
| **ITSC 2027** (Boston, 09/2027) | ~cuối T2–T3/2027 (pattern: ITSC 2026 là 01/03) | **MỤC TIÊU CHÍNH** — flagship ITS, Xplore, 6pp+2. CFP ra ~T11–12/2026, theo dõi ieee-itsc.org. |
| ICIP 2027 (Singapore, 29/11–03/12/2027!) | ~T2–T5/2027 (pattern) | Xplore, 5pp — backup nếu trượt ITSC. |
| ATC/MAPR 2026 | ĐÃ ĐÓNG | ATC 2027/MAPR 2027 (~T5/2027) là fallback khu vực. NICS có vẻ ngủ đông. |

**Cascade khuyến nghị:** thí nghiệm T8–T10/2026 → nộp **WACV'27 workshop (T10–11) hoặc VEHITS (17/11)** → dù đậu hay trượt, bản nâng cấp nộp **ITSC 2027 (~01/03/2027)** nếu bài chưa vào Xplore; nếu đã đậu workshop/VEHITS thì chuyển thẳng sang Bài 2 journal.

### A5. Journal Q1/Q2 cho Bài 2 — ĐÃ KIỂM CHỨNG ✅ (quartile SJR 2024 + JIF từ JCR 06/2026)

**Top 3 khuyến nghị:** ① **IEEE TIM** (Q1, IF 7.0, đăng miễn phí theo đường subscription) — chính là venue của YOLO-LLTS nên có tiền lệ biên tập trực tiếp; có quy trình chính thức cho extended conference paper ("List of Extensions"); ĐIỀU KIỆN SỐNG CÒN: phải viết theo khung *instrumentation & measurement* (đặc trưng hóa hệ đo thị giác + năng lượng), không viết như bài detection-app. ② **J. Real-Time Image Processing** (Q2, IF 4.9) — fit đúng thể loại nhất, xác suất đậu thực tế cao nhất, ~100 ngày review. ③ **Journal of Systems Architecture** (Q1, IF 5.3) — nếu framing theo góc kiến trúc hệ thống edge (cross-generation Orin/Thor); là journal duy nhất có luật extension tường minh "≥30% mới + đổi title/abstract"; pipeline nhanh (~138 ngày đến acceptance). Chi tiết đầy đủ ở Phần C.

**Tránh:** Neural Computing & Applications (**đã bị WoS delist**), IET ITS (nay Q3), Machine Vision & Applications (JCR ~Q3). **Rủi ro desk-reject cho bài dạng benchmark:** ESWA (triage 5 ngày), Measurement (có luật thành văn loại bài "ML áp dụng công cụ sẵn có"), IoT-J (nếu thiếu góc hệ thống IoT).

---

## PHẦN B — BÀI 1: HỘI NGHỊ (6–10 trang)

### Tiêu đề dự kiến
**"Quantized Night-Time Traffic Sign Detection at the Edge: An Accuracy–Latency–Power Study of INT8 YOLO on Jetson"**

### Elevator pitch (3 câu, dùng cho Abstract/Intro)
Biển báo ban đêm là điều kiện nguy hiểm nhất nhưng các detector đêm SOTA (YOLO-LLTS) chưa hề được lượng tử hóa hay đo năng lượng; ngược lại các bài quantized-edge (IET ITS 2026) chỉ làm ban ngày và không làm biển báo. Chúng tôi cung cấp nghiên cứu đầu tiên kết hợp cả ba: INT8 PTQ cho nighttime TSD, đo đầy đủ accuracy–latency–power trên Jetson Orin, với cross-dataset protocol trên 2 bộ đêm thật. Kết quả cho thấy [X]% năng lượng tiết kiệm đổi lấy [Y] điểm mAP trong điều kiện đêm — và độ rơi do quantization ban đêm [khác/không khác] ban ngày.

### Contributions (4)
1. Benchmark đầu tiên về INT8 quantization cho nighttime TSD với đo power/latency thực trên thiết bị (Jetson AGX Orin, tegrastats/jtop; ms/frame, W steady-state, J/frame, FPS/W).
2. Cross-dataset night protocol: train CNTSSS (3.276), test CNTSSS-test (786, 4 thành phố chưa thấy) + CCTSDB2021-night (~500, trích theo XML điều kiện) — cùng taxonomy 3 lớp.
3. Ablation calibration set cho PTQ: calibrate bằng ảnh ngày vs đêm vs trộn — ảnh hưởng đến accuracy đêm (mini-study, mở đường cho Bài 2).
4. So sánh trực tiếp với YOLO-LLTS (22,3 FPS không TensorRT trên AGX Orin) → cho thấy model nano chuẩn + TensorRT INT8 đạt [nhanh hơn nhiều lần] với accuracy [tương đương/thấp hơn Z điểm].

### Ma trận thí nghiệm (giữ gọn cho 6–8 trang)

| Chiều | Giá trị |
|---|---|
| Model | YOLO11n (chính), YOLOv8n (đối chứng) |
| Precision | FP32, FP16, INT8-PTQ (TensorRT, calibrate trên Orin) |
| Calibration | day / night / mixed (3 biến thể) |
| Train | CNTSSS train (3.276) — fine-tune từ COCO pretrained |
| Test | CNTSSS test (786) + CCTSDB2021-night (~500); ablation: TT100K-night synthetic tự dựng (tùy thời gian) |
| Phần cứng | Jetson AGX Orin, 2 power mode (MAXN + 30W); *(Thor để dành Bài 2)* |
| Metrics | mAP@50, mAP@50-95, per-class (3 lớp), ms/frame, W, J/frame, FPS/W; Δ(FP32→INT8) so sánh ngày vs đêm |

### Lịch 10 tuần (bắt đầu ~01/08/2026)

| Tuần | Việc |
|---|---|
| 1 | Tải CNTSSS (885MB) + CCTSDB2021; kiểm tra nhãn thực tế; trích night-test theo XML; **email tác giả YOLO-LLTS** (license + xin TT100K-night) và **tác giả INTSD** (xin full 41 lớp — cho Bài 2); dựng repo + môi trường Jetson (JetPack, TensorRT, jtop). |
| 2–3 | Fine-tune YOLO11n/YOLOv8n trên CNTSSS; baseline FP32/FP16 trên Orin; dựng harness đo power (kế thừa script AETA2026). |
| 4–5 | Export INT8 PTQ + 3 biến thể calibration; đo đủ ma trận trên Orin; phân tích Δ ngày–đêm. |
| 6 | Cross-dataset CCTSDB-night; (nếu kịp) dựng TT100K-night CycleGAN ablation. |
| 7–8 | Viết bài (dùng skill research-paper-writing), vẽ figures (Pareto accuracy–power, bar Δ theo lớp). |
| 9 | Tự review + **rà lại arXiv bằng 13 truy vấn** (phần E); skim PDF bài MVA paywall. |
| 10 | Buffer + nộp: WACV'27 workshop (nếu list T8 có workshop hợp) hoặc **VEHITS 17/11**. |

### Rủi ro & phương án B (Bài 1)
- **CNTSSS nhãn lỗi/format lạ sau khi tải** → tuần 1 phát hiện ngay; phương án B: dùng CCTSDB2021 night làm train phụ + NTS-YOLO (CC BY) bổ sung.
- **INT8 mất quá nhiều mAP ban đêm** → đó CHÍNH LÀ finding (negative result vẫn là đóng góp — đã có tiền lệ arXiv 2508.19600); thêm QAT nhẹ nếu cần cứu.
- **Trượt WACV workshop/VEHITS** → ITSC 2027 (~01/03) là đích chính, thời gian dư để polish.

---

## PHẦN C — BÀI 2: JOURNAL Q1/Q2 (bản mở rộng)

### Tiêu đề dự kiến
**"Night-Time Traffic Sign Detection on Embedded AI Platforms: A Cross-Generation Study of Quantization, Domain Shift, and Deployment-Aware Evaluation"**

### Nội dung MỚI so với Bài 1 (mục tiêu ≥50–60% mới)

1. **QAT vs PTQ** (Bài 1 chỉ PTQ): QAT trên CNTSSS, kỳ vọng thu hẹp Δ đêm; thêm ModelOpt explicit quantization.
2. **Cross-generation Orin vs Thor** — điểm nhấn riêng có: đo và phân tích việc **INT8 không tăng tốc trên Thor** (Myelin issue, NVIDIA xác nhận 03/2026) vs +36% trên Orin; đánh giá workaround ModelOpt + stronglyTyped; trả lời câu hỏi "INT8 có còn đáng đầu tư trên phần cứng edge thế hệ mới?" → góc đo lường thực nghiệm mới, chưa có bài TSD nào chạm tới.
3. **Domain-gap × quantization đầy đủ** (nâng cấp contribution 3 của Bài 1): train day-only vs mixed; calibration study đầy đủ; breakdown theo điều kiện (dusk/đêm sâu/mưa — CNTSSS có đa dạng điều kiện).
4. **Deployment-aware evaluation**: streaming video đêm thật, sustained load 30–60 phút, log nhiệt độ/tần số/FPS (thermal throttling), so static-mAP vs streaming (motivation từ Edge-TSR 2606.17241).
5. **Mở rộng model**: + YOLO26n (thiết kế native cho quantization/edge, không NMS) → câu chuyện tiến hóa kiến trúc.
6. **Fine-grained 41 lớp trên INTSD** (nếu email tuần 1 thành công) → nâng từ 3-class lên fine-grained, giá trị thực tiễn cao hơn hẳn.
7. **Năng lượng chi tiết**: J/frame theo power mode, FPS/W cross-generation, chi phí năng lượng của enhancement preprocessing (CLAHE) nếu thêm.

### Journal đích (kiểm chứng 20/07/2026; quartile SJR 2024, JIF từ JCR ra 06/2026; APC = phí open-access, mọi journal "hybrid" đều có đường subscription MIỄN PHÍ)

**Chiến lược nộp:** TIM (Q1, framing đo lường) → nếu bị desk-reject vì framing → JSA (Q1, framing kiến trúc hệ thống) → JRTIP (Q2, chắc ăn nhất). Ba đích này cùng dùng chung một bộ thí nghiệm, chỉ đổi cách kể chuyện.

| # | Journal | Quartile | IF | APC (có đường free?) | First decision / tổng | Ghi chú fit |
|---|---|---|---|---|---|---|
| 1 | **IEEE Trans. Instrumentation & Measurement** | Q1/Q1 | 7.0 | $2.800 (✅ free route) | ~4–6 tuần (có khi 7 tháng) | Tiền lệ YOLO-LLTS; quy trình extension chính thức (cover letter + List of Extensions); PHẢI framing I&M; luật tối đa 1 vòng Major Revision |
| 2 | **J. Systems Architecture** (Elsevier) | Q1/Q1 | 5.3 | $2.450 (✅) | 2 ngày screen, ~138 ngày accept | Framing hệ thống edge (cross-gen Orin/Thor, thermal, energy); luật extension rõ: ≥30% mới + đổi title/abstract |
| 3 | **J. Real-Time Image Processing** (Springer) | Q2/~Q2 | 4.9 | $3.390 (✅) | 2 ngày screen, ~100 ngày | Fit thể loại tốt nhất (~160 bài Jetson; có bài energy-aware quantization 2025, Tiny-YOLOv8 embedded 2026); cửa cao nhất |
| 4 | EAAI (Elsevier) | Q1/Q1 | 9.0 | $3.040 (✅) | ~207 ngày accept ⚠️ chậm | IF cao, có đăng benchmark biển báo điều kiện khắc nghiệt (2026); yêu cầu cover letter nêu delta |
| 5 | IEEE Sensors Journal | Q1/Q1 | 4.5 | $2.800 (✅) | ~8,8 tuần đến ePub | Framing camera-as-sensor; có guideline mở rộng conference paper chính thức |
| 6 | IEEE Trans. Consumer Electronics | Q1/Q1 | 9.9 (tăng nóng, cộng đồng có nghi ngại) | $2.800 (✅) | ≤10 tuần | "Dark horse": mục Industry Paper chuộng deployment/benchmark; cần framing dashcam/driver-assist |
| 7 | Internet of Things (Elsevier) | Q1 | 7.1 | $3.020 (✅) | 4 ngày screen, ~116 ngày | Có đăng bài Jetson/edge-DL; cần khoác nhẹ góc IoT |
| 8 | Applied Intelligence (Springer) | Q2/Q2 | 3.5 | $3.290 (✅) | ~2,6 tháng — nhanh nhất | Backup Q2 nhanh, có đăng ADAS/biển báo |
| 9 | Scientific Reports | Q1 (đa ngành) | 4.9 | $2.850 **bắt buộc** | ~4,3 tháng, ~57% accept | Đăng bài YOLO biển báo as-is; điều kiện: kết quả chính KHÔNG được lộ từ bài hội nghị + khai trong cover letter |
| 10 | IEEE Access | Q1 (đa ngành) | 4.2 | $2.160 **bắt buộc** | ~4–6 tuần | Fallback cuối: xét soundness, prestige thấp nhất |

**Loại khỏi danh sách:** Measurement (luật thành văn desk-reject bài "ML áp dụng tool sẵn có" — đúng hồ sơ bài này), ESWA (triage 5 ngày, đòi novelty phương pháp), IoT-J (rủi ro desk-reject nếu chỉ 1 thiết bị), Neural Computing & Applications (**WoS delisted** — dù SJR còn ghi Q1), IET ITS (rớt xuống Q3), Machine Vision & Applications (JCR ~Q3), MDPI Sensors (chỉ dùng nếu cần cực nhanh, APC bắt buộc ~$2.600).

**Luật extension cần nhớ:** IEEE = "substantial additional technical material", cite bài hội nghị, TIM đòi delta là *kết quả kỹ thuật mới* (không phải giải thích thêm); JSA = ≥30% mới + title/abstract khác; Springer ≈ 30% theo thông lệ; SciRep = kết luận chính không được suy ra từ bài proceedings. → 7 hạng mục delta ở trên (QAT, Thor cross-gen, domain-gap, streaming/thermal, YOLO26n, INTSD, energy) vượt chuẩn này thoải mái.

### Lịch dự kiến
Nộp Bài 1 (T11/2026) → thí nghiệm mở rộng T12/2026–T3/2027 (QAT, Thor, streaming) → viết T4/2027 → nộp T5/2027.

---

## PHẦN D — RANH GIỚI 2 BÀI (chống self-plagiarism)

| Nội dung | Bài 1 (hội nghị) | Bài 2 (journal) |
|---|---|---|
| PTQ INT8 trên Orin, 3-class, CNTSSS+CCTSDB night | ✅ TRỤC CHÍNH | Tóm tắt lại + cite Bài 1 |
| QAT, ModelOpt | ❌ | ✅ |
| Jetson Thor, cross-generation | ❌ (chỉ nhắc future work) | ✅ TRỤC CHÍNH |
| Domain-gap × quantization | Mini-ablation (3 calibration set) | ✅ Đầy đủ |
| Streaming/thermal | ❌ | ✅ |
| YOLO26n, INTSD 41 lớp | ❌ | ✅ (tùy dữ liệu) |

Quy tắc: Bài 2 **cite Bài 1 tường minh**, viết lại toàn bộ text (không copy đoạn), tái dùng tối đa ~1–2 bảng nền có ghi nguồn; phần mới ≥50% (chuẩn IEEE cho extended conference paper).

---

## PHẦN E — CHECKLIST TUẦN 1 (trước khi cam kết toàn bộ)

- [ ] Tải `CNTSSS.zip` (Google Drive, 885MB), xác nhận format YOLO + đếm lại 3.276/786 + xem chất lượng nhãn bằng mắt (~50 ảnh).
- [ ] Tải CCTSDB2021, trích night-test từ XML điều kiện, xác nhận ~500 ảnh.
- [ ] Email tác giả YOLO-LLTS: xin phép dùng CNTSSS (không license) + xin TT100K-night synthetic.
- [ ] Email tác giả INTSD (project page adityamishra-ml.github.io/INTSD): xin full dataset 41 lớp cho Bài 2.
- [ ] Đăng ký theo dõi: ieee-itsc.org (CFP ITSC 2027), wacv.thecvf.com (workshop list ~11/08), vehits.scitevents.org.
- [ ] Xác nhận JetPack/TensorRT trên Orin; thử export INT8 1 model bằng `quantize=8` (lưu ý flag `int8=True` đã deprecated).
- [ ] Trên Thor: thử ModelOpt explicit quant + `--stronglyTyped` sớm (cho Bài 2) — nếu NVIDIA fix Myelin trước đó thì finding cross-gen phải điều chỉnh giọng.

**13 truy vấn rà gap trước khi nộp** (chạy lại trên Google Scholar/arXiv, cả tuần 9 Bài 1 lẫn trước khi nộp Bài 2):
1. nighttime traffic sign detection quantization INT8
2. low-light traffic sign detection TensorRT Jetson
3. YOLO traffic sign night edge power consumption
4. quantized object detection low-light traffic Jetson Orin
5. CNTSSS dataset citing papers
6. VISAT benchmark quantized YOLO evaluation
7. "Quantized YOLOv8n" "Jetson Orin Nano" traffic nighttime
8. traffic sign detection edge deployment power latency INT8
9. arXiv "traffic sign" nighttime dataset new benchmark
10. "quantization-aware training" traffic sign nighttime low-light
11. Raspberry Pi TFLite INT8 traffic sign night energy
12. arxiv.org full-text: "traffic sign" "quantization"
13. Semantic Scholar: citations của 2503.13883 (YOLO-LLTS) và 2510.26833 (VISAT)

---

*Nguồn kiểm chứng chính: github.com/linzy88/YOLO-LLTS · arxiv.org/abs/2503.13883 (v4) · arxiv.org/abs/2511.17183 (v3) · github.com/csust7zhangjm/CCTSDB2021 · docs.ultralytics.com/guides/nvidia-jetson/ · forums.developer.nvidia.com/t/363884 (Thor INT8) · ietresearch.onlinelibrary.wiley.com/doi/10.1049/itr2.70135 · arxiv.org/pdf/2606.17241 · vehits.scitevents.org · ieee-itsc.org · wacv.thecvf.com/Conferences/2027/Dates*
