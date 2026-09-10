# G0 — Kiểm chứng phép đo CCTSDB

## Phạm vi

Chỉ audit ảnh/labels/XML/prediction đã có và tái tính AP trên CPU. Không train, không export, không load engine, không chọn policy theo test. Evaluator và kết quả lịch sử được giữ nguyên; chưa scale 15 model.

## Đã triển khai

`scripts/audit_cctsdb_measurement.py` kiểm tra:

- ID ảnh không trùng (kể cả trùng stem nhưng khác extension), đủ membership của split và labels.
- Shape prediction khớp kích thước ảnh thực; class, confidence, tọa độ hữu hạn và thứ tự bbox hợp lệ.
- Hash model/data giữa prediction và evaluation; hash nội dung prediction đúng provenance.
- XML ghép bằng archive member stem; đối chiếu class/bbox không phụ thuộc thứ tự object bằng bipartite matching, tolerance 0,05 pixel.
- Đếm size theo XML và YOLO chưa làm tròn lại; ghi mọi trường hợp thay bin do rounding, không tự sửa thresholds.
- CPU replay bằng `BaseValidator.match_predictions` và `ap_per_class` của đúng phiên bản Ultralytics ghi trong result. Ghi phiên bản NumPy/Torch và hash code matching/AP để truy nguyên.
- Giữ prediction có diện tích 0 sau clipping trong replay, nhưng gắn cờ review; không xóa prediction để làm đẹp metric.

Đây là công cụ kiểm tra, không phải evaluator size mới. `pass` chỉ có nghĩa các kiểm tra được cấu hình đã đạt, không chứng minh toàn bộ G0 hoặc paper đạt. Chưa có audit nội bộ calibrator tiêu thụ ảnh, precision layer coverage hay leakage theo hash giữa mọi split.

## Kết quả local đã xác minh

Mốc code nguồn trước thay đổi: `88045db`. Môi trường replay local tách riêng tại `local/measurement_audit_env`, Ultralytics 8.4.102, Torch 2.8.0, NumPy 2.4.2; không sửa environment server. Torch/NumPy chưa khớp server nên không quy mọi sai khác cho `predict()`/`val()`.

Bốn bộ prediction v2 dev đủ 1.636 ảnh và 2.706 GT. Hash server khớp sau khi chuẩn hóa CRLF thành LF trong bộ nhớ; không chỉnh file. Trên server dùng kiểm tra hash byte chính xác, không cần flag chuẩn hóa.

| Dev representation | Delta AP50 replay − val (điểm %) | Delta AP50–95 (điểm %) | Prediction diện tích 0 |
|---|---:|---:|---:|
| FP16 | −0,0302 | −0,0565 | 0 |
| Low-Luminance INT8 | −0,0023 | +0,0103 | 7 |
| Uniform INT8 | −0,0760 | +0,2669 | 98 |
| VCSC proportional INT8 | −0,0069 | +0,1957 | 0 |

Tolerance kiểm tra là 0,0001 AP, tức 0,01 điểm phần trăm. Đây là ngưỡng phát hiện sai khác phép đo, không phải ngưỡng chọn policy. Sai khác hiện tại cần giải thích, không đồng nghĩa toàn bộ mAP cũ sai. Replay dùng tọa độ ảnh gốc; validator có thể tính ở tọa độ ảnh input và có khác biệt về scaling/ties/preprocessing.

Local không có `xml.zip` đầy đủ: file tải dở là 0 byte. Với positive test, có thể đối chiếu bằng XML trong archive weather đã có local:

- Đủ 1.500 ảnh, 3.228 GT, không ảnh nào sai class/bbox/dimensions ở tolerance 0,05 pixel.
- Sai số tọa độ lớn nhất giữa XML và YOLO: khoảng 0,00128 pixel.
- XML đếm XS/S/M/L/XL = 813/823/812/408/372; chưa giải quyết chênh lệch S/M so với số trong bài gốc.
- Tính diện tích trực tiếp từ YOLO float cho 778/845/825/408/372; có 48 object đổi bin so với XML vì gần biên. Vì vậy không thay area từ XML sang YOLO float một cách âm thầm.

Đây là integrity audit trên test đã có, không dùng để chọn calibration. Kết quả local chi tiết:

- `results/measurement_audit_v1/local_dev_replay_v2/`
- `results/measurement_audit_v1/local_test_xml/`

Các thư mục audit thử nghiệm trước đó được giữ nguyên. Chỉ hai thư mục trên là bằng chứng dùng trong bảng này. Bộ kiểm thử có 13 test, gồm CPU replay thực trên trường hợp tổng hợp perfect/empty predictions.

## Chạy trên server

Trước hết đồng bộ code mới qua quy trình Git local → GitHub → server. Không chạy lệnh này khi script chưa có trên server. Không nâng cấp Torch/Ultralytics/TensorRT.

Mỗi block là một dòng vật lý. Chạy foreground, không `nohup`. Audit CPU này không cần export hoặc inference lại.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master
```

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && python scripts/audit_cctsdb_measurement.py --predictions-dir results/calibration_method_v2/rtx8000/yolo11n/dev_predictions --evaluations-dir results/calibration_method_v2/rtx8000/yolo11n/dev_eval --split-dir data/processed/cctsdb2021_clean/dev --xml ../nighttime-tsd/data/raw/CCTSDB2021/xml.zip --out-dir results/measurement_audit_v1/server_dev_v1 --replay-ultralytics
```

Kết thúc sẽ in `audit_summary.json` ra terminal và ghi report riêng cho từng representation. Exit code 2 / `review_required` là audit đã hoàn tất nhưng còn vấn đề cần xem, không phải job GPU bị treo. `status: error` trong một report là input/dependency không đạt; đọc trường `error`. Không tự tăng tolerance để chuyển sang pass. Script không ghi đè thư mục đã tồn tại; nếu cần chạy lại, dùng tên output phiên bản mới.

Push riêng output audit, không thêm engine/weights/data:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git add -f results/measurement_audit_v1/server_dev_v1/*.json && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git commit -m "results: G0 CCTSDB measurement audit on server"
```

Chạy push riêng để vẫn dùng được nếu commit báo `nothing to commit`:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git push origin master
```

## Gate tiếp theo

Đối chiếu replay cùng environment server và XML dev. Nếu vẫn lệch, kiểm tra hoặc bổ sung capture prediction ngay trong validator để dùng cùng một lượt inference cho mAP/size/bootstrap; version kết quả mới, không ghi đè số cũ. Sau đó mới kiểm chứng evaluator size với matching/ignore/IoU thresholds chuẩn và chuyển sang phân tích lỗi YOLO11n trên dev. G0 hiện chưa pass, chưa triển khai ba-nano pilot hoặc matrix 15 model.

## Kết quả server và bước capture cùng lượt val

Đã nhận audit tại commit `b4b110a`. Cả bốn report hoàn tất, trạng thái `review_required`, không có `error`. `00751.jpg` có một bbox XML khác YOLO khoảng 0,5 pixel; raw YOLO archive cũng có tọa độ khác XML, nên không quy lỗi này cho training. `04492.jpg` có một bbox vượt chiều rộng ảnh 1 pixel. Có 40 object đổi size-bin khi dùng diện tích YOLO float thay cho XML. Không sửa annotation hoặc đổi ngưỡng để ép pass.

Source code Ultralytics 8.4.102 xác nhận `_process_batch()` tính IoU/matching trong tọa độ ảnh input; `scale_preds()` đưa bbox về ảnh gốc và clip biên. Công cụ mới `scripts/capture_cctsdb_validator.py` intercept chính `_process_batch()` trong một lượt validator và trả nguyên kết quả cho parent. Nó lưu prediction cả hai hệ tọa độ, target bbox thực tế, TP cho 10 IoU, thứ tự và dtype thống kê. Không gọi `model.predict()` hoặc chạy inference lần hai.

Sau khi ghi JSON, script đọc lại TP/confidence/class/target và tính lại AP/P/R bằng cùng `ap_per_class`. Dtype được giữ vì confidence INT8 có nhiều ties. Chênh lệch từng metric và per-class AP phải không quá `1e-12`. Đây là kiểm chứng serialization/aggregation trong cùng run; không phải xác minh độc lập thuật toán matching hoặc giải quyết hết G0. Capture vẫn lưu native bbox để kiểm chứng geometry/matching độc lập ở bước sau.

Script khóa vào engine FP16 YOLO11n cũ (kiểm tra hash), dev 1.636 ảnh, batch 1, imgsz 640, cùng GPU phase lock. Không có tùy chọn test hoặc policy. Kết quả vào thư mục mới. Bộ test so sánh parent validator với capture validator, bao gồm confidence ties, ảnh không detection, ảnh không GT, clipping và replay từ JSON; tổng 15 test đạt local. Chưa chạy TensorRT thật ở local.

Sau khi pull code mới, chạy trực tiếp:

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && python scripts/capture_cctsdb_validator.py --out-dir results/measurement_audit_v1/server_fp16_capture_v1 --device 0
```

Khi thấy `DONE` và `status: pass`, chỉ phần same-run replay đã đạt. Nếu `review_required` hoặc lỗi, không train/export lại. Thư mục chứa `capture_report.json`, `validator_predictions.json` và `dev_absolute.yaml`.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git add -f results/measurement_audit_v1/server_fp16_capture_v1/capture_report.json results/measurement_audit_v1/server_fp16_capture_v1/validator_predictions.json results/measurement_audit_v1/server_fp16_capture_v1/dev_absolute.yaml && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git commit -m "results: G0 single-pass FP16 dev validation capture"
```

Sau đó chạy lệnh push riêng ở mục trên. Không stage engine, weights hoặc dữ liệu raw.

## Bước hiện tại: native rematch và diagnostic COCO/XML theo kích thước

Đã nhận capture server tại commit `5eb7ec3`: 1.636 ảnh, 2.706 GT; mAP50 = 0,9777499256550611 và mAP50-95 = 0,7630247217006284. Replay từ TP đã serialize trên server khớp chính xác các metric. `scripts/verify_cctsdb_capture.py` tính lại IoU và assignment từ bbox native, không dùng TP đã lưu làm đầu vào, rồi đối chiếu TP. Thử local trên toàn bộ capture cho 0 quyết định TP khác nhau. Đây là kiểm chứng đường dữ liệu/geometry bằng cùng thuật toán Ultralytics, không phải chứng minh độc lập rằng thiết kế matching của Ultralytics là chuẩn COCO.

Script mới còn tạo diagnostic riêng bằng **pycocotools 2.0.10**: score-greedy matching, one-to-one cho GT không crowd; bỏ qua GT ngoài size-bin và prediction không match nằm ngoài khoảng diện tích theo quy ước COCO. AP dùng 101 recall points và 10 IoU 0,50–0,95. Nó không tương đương chính xác AP nội suy của Ultralytics hoặc evaluator size cũ. Tham chiếu implementation: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py.

Các quy tắc được ghi trong JSON:

- XML theo member ID, tọa độ ảnh gốc; diện tích liên tục `(xmax-xmin)*(ymax-ymin)`, không `+1`, không làm tròn/clip GT.
- Giữ các biên đang có: XS ≤210; S (210,400]; M (400,1000]; L (1000,2000]; XL >2000 pixel². Đây là diagnostic có version, không tự tuyên bố đã khớp toàn bộ thống kê bài gốc.
- Prediction từ cùng lượt val, scaled/clipped về ảnh gốc; tối đa 300 detection/ảnh, giữ score ties ổn định và box suy biến do clipping.
- Class không có GT trong bin nhận `null` và không vào trung bình; không tự thay bằng 0.
- Tính `all` và XS/S/M/L/XL dưới cùng một quy ước COCO/XML. Không so size mới với size cũ như thể chỉ model thay đổi. Không dùng full COCO/XML thay thế âm thầm full Ultralytics.
- Giữ nguyên XML khác YOLO ở `00751` và box ngoài biên ở `04492`, ghi nhận thay vì sửa dữ liệu huấn luyện. Chưa tính size INT8 hoặc quyết định policy.

Sáu test mới kiểm tra biên diện tích, perfect/empty predictions, missing class, duplicate FP, ignore ngoài bin, score ties, phát hiện TP bị sửa, geometry lỗi và không làm thay đổi input. Tổng 21 test đạt local. Native matching chạy được local; phần dev XML đầy đủ phải chạy server. Không cần GPU, weights, export hay inference mới.

Sau khi pull code, tạo môi trường audit phụ để không thay dependency của conda training. Chỉ bổ sung pycocotools vào venv phụ; không upgrade Torch/Ultralytics/TensorRT:

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && python -m venv --system-site-packages local/g0_size_env && local/g0_size_env/bin/python -m pip install --no-deps pycocotools==2.0.10
```

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/verify_cctsdb_capture.py --capture-dir results/measurement_audit_v1/server_fp16_capture_v1 --xml ../nighttime-tsd/data/raw/CCTSDB2021/xml.zip --out-dir results/measurement_audit_v1/server_native_size_v1
```

Output: `native_matching.json`, `size_coco_xml.json`, `verification_summary.json`. Không ghi đè output đã tồn tại. `DONE` nghĩa audit hoàn tất; `native_matching_status: pass` chỉ xác nhận matching, còn `global_g0: review_required` được giữ có chủ ý để review quy ước annotation/size và capture INT8. Nếu có khác biệt TP thì trả exit 2 với chi tiết, không tự nới tolerance.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git add -f results/measurement_audit_v1/server_native_size_v1/native_matching.json results/measurement_audit_v1/server_native_size_v1/size_coco_xml.json results/measurement_audit_v1/server_native_size_v1/verification_summary.json && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git commit -m "results: G0 native matching and versioned size diagnostic"
```

Sau đó push riêng như trên. Không chạy thêm calibration hay scale 15 model trước khi review kết quả.

## INT8 capture trên dev sau native FP16 pass

Commit kết quả server `2f3d496` xác nhận 63.530 quyết định TP trên 1.636 ảnh FP16 không khác CPU rematch. Các hash capture/code/engine khớp. Diagnostic COCO/XML có 2.706 GT; XS/S/M/L/XL = 190/642/995/498/381. Full AP50/AP50-95 COCO/XML = 0,9728170266748176 / 0,756546108865103. Đây không thay thế full Ultralytics 0,9777499256550611 / 0,7630247217006284.

`scripts/run_g0_int8_capture.py` chạy đúng ba engine INT8 v2 seed42 hiện có: Uniform, Low-Luminance, VCSC proportional. Trình tự cho từng engine: same-pass dev capture → native rematch → COCO/XML size → đối chiếu với FP16. Không retrain/export, không calibration mới, không test, không chọn seed/policy. Môi trường và engine hash được kiểm tra trước inference; output luôn mới, không auto-skip kết quả dở dang. Nếu thất bại giữa chừng, giữ output và đọc lỗi trước khi chọn tên phiên bản mới để chạy lại.

Capture mặc định vẫn là FP16, nhưng nhận `--representation` từ allowlist cố định. Hash engine phải khớp evaluation và provenance; source hash phải là frozen YOLO11n; INT8 phải có bằng chứng cache isolation. Runner kiểm tra Torch/NumPy/Ultralytics/TensorRT cùng phiên bản FP16, dùng GPU phase lock ở mỗi capture, chạy tuần tự, và đối chiếu từng image ID, kích thước, ratio_pad, native GT/class với FP16. Mỗi capture lưu same-run replay; matching có sai khác thì dừng để review.

Bảng cuối lưu riêng hai quy ước: `ultralytics_full` và `coco_xml` (all + 5 bins). Delta tính theo điểm phần trăm (INT8 − FP16), không trộn giữa evaluator. Gate G0 vẫn `review_required`; một seed chưa đủ kết luận superiority hoặc ổn định sampling. Evaluator cũ, annotation và engine không bị sửa. 24 test local đạt, gồm kiểm tra hash/cache/allowlist, sai lệch target và metric convention. Chưa chạy TensorRT thực trên local.

Sau khi pull code, dùng venv audit phụ đã tạo, chạy foreground:

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/run_g0_int8_capture.py --out-dir results/measurement_audit_v1/server_int8_capture_v1 --device 0
```

Chờ `DONE: .../comparison_summary.json`. Trong mỗi thư mục policy có `capture/` (report, predictions, YAML), `verification/` (native, COCO/XML, summary) và `comparison.json`. Capture sử dụng GPU; không chạy train/export/benchmark đồng thời. Các bước verification dùng CPU.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git add -f -- ':(glob)results/measurement_audit_v1/server_int8_capture_v1/**/*.json' ':(glob)results/measurement_audit_v1/server_int8_capture_v1/**/*.yaml' && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git commit -m "results: G0 same-pass INT8 dev capture and size comparison"
```

Sau đó push riêng theo lệnh ở trên. Chỉ push artifact audit; không thêm engine/weights/raw data.

## Phân tích dev từ capture đã kiểm chứng: localization và paired bootstrap

Đã nhận `b40c6db`: cả ba INT8 seed42 đều capture pass; CPU rematch local và server không có quyết định TP khác nhau. Replay AP/P/R và historical validation khớp chính xác. Scope giữ nguyên frozen YOLO11n/dev, không official test hoặc calibration mới.

Quy ước phép đo được cố định cho bước phân tích này: full Ultralytics vẫn giữ nguyên làm metric validation; all + size COCO/XML là hệ diagnostic riêng. CI ở bước này **chỉ áp dụng COCO/XML**, không ghép vào điểm AP Ultralytics. XML nguyên bản, area liên tục không +1/round/clip GT, boundary XS/S/M/L/XL và matching/ignore giữ như bước trước. Các khác biệt annotation đã audit vẫn được ghi nhận; bước này không tuyên bố size evaluator là evaluator chính thức của tác giả dataset hay đóng toàn bộ G0.

`scripts/analyze_dev_quantization.py` chỉ đọc capture và XML dev rồi:

1. Kiểm tra liên kết hash, cùng GT/preprocessing, cùng evaluator/config; tính lại COCO point estimate và bắt buộc khớp size report trước ở tolerance 1e-12.
2. Phân tích GT-centric bằng ALL-area same-class COCO matching tại IoU 0,50/0,75/0,90. Báo recall theo size, số GT mất/được match so với FP16; median IoU và sai số tâm chuẩn hóa của cặp đã match ở IoU50. Median có selection bias do chỉ gồm cặp đã match, không phải causal decomposition. Confidence là ngưỡng lưu prediction 0,001, không phải ngưỡng deployment.
3. Paired image bootstrap mặc định 1.000 mẫu, PCG64 seed 20260910; mỗi mẫu N ảnh lấy có hoàn lại, cùng indices cho FP16 và cả ba INT8. Giữ cả các lần xuất hiện lặp, sắp theo source filename để tie-breaking ổn định. Tái sử dụng matching trong từng ảnh và chạy lại official COCO accumulation trên mẫu; không trung bình AP từng ảnh.
4. Lưu mAP50 và mAP50-95 cho all/XS/S/M/L/XL từng draw; percentile 95% CI của chênh lệch INT8−FP16 và Low-Luminance/VCSC−Uniform, đơn vị điểm phần trăm. Nếu một class có GT ở endpoint gốc biến mất trong bootstrap draw, endpoint nhận null; báo số draw hợp lệ/không xác định, không tự đổi macro class set.

Đây là exploratory dev inference, CI không hiệu chỉnh multiple comparisons, không phải bằng chứng superiority xác nhận. CI điều kiện trên bốn engine seed42 cố định; không đo calibration-seed variance. IID image bootstrap có thể đánh giá thấp uncertainty nếu có scene/camera/sequence dependence. Không chọn policy hoặc scale từ CI này tự động.

29 test local đạt, gồm so cached-bootstrap với fresh COCO evaluation khi nhân bản image occurrences, trường hợp không prediction/thiếu GT class/ties, zero paired delta khi engine giống nhau, percentile/missing endpoints và localization transitions. Toàn bộ dev XML chỉ có server, nên chưa chạy full-data bootstrap ở local. Trước bootstrap script luôn kiểm chứng lại point estimates trên server, rồi mới ghi output mới. Bản evaluator cũ không bị thay kết quả: thay đổi helper chỉ thêm tùy chọn trả về đối tượng COCO để tái sử dụng matching.

Sau khi pull code, chạy foreground bằng venv audit đã có (CPU, không inference, không GPU):

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/analyze_dev_quantization.py --out-dir results/measurement_audit_v1/server_dev_error_bootstrap_v1 --resamples 1000 --seed 20260910
```

Đầu tiên chờ bốn dòng `VERIFIED`, sau đó `BOOTSTRAP .../1000`, và cuối cùng `DONE: .../analysis_summary.json`. Script in thời gian đã chạy và ETA; không cần nohup. Không ghi đè thư mục có sẵn. Chạy thử nhanh có thể dùng `--resamples 10` với một output khác; kết quả đó ghi `smoke_only`, không dùng làm CI cuối.

Artifact: `bootstrap_samples.json` (image names + indices), `bootstrap_draws.json`, `localization_per_gt.json`, `localization_summary.json`, `analysis_summary.json` (CIs, provenance và limitations). G0 vẫn `review_required` chờ xem report, chưa thêm train/export/INT8 matrix.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git add -f results/measurement_audit_v1/server_dev_error_bootstrap_v1/*.json && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git commit -m "results: dev localization diagnostics and paired image bootstrap"
```

Push riêng theo quy trình phía trên; chỉ audit JSON, không weights/engine/raw data.
