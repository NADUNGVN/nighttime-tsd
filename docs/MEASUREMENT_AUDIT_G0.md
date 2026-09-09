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
