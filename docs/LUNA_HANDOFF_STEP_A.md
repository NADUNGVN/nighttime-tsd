# Luna handoff — bối cảnh nghiên cứu và Uniform build-repeat Step A

> Cập nhật sau Step A: đọc [ASTRA_TO_LUNA.md](ASTRA_TO_LUNA.md), entry A2L-001 trước khi làm. Báo cáo tiếp theo ghi vào [LUNA_TO_ASTRA.md](LUNA_TO_ASTRA.md). Các mục dưới đây là context ban đầu 12/09; trạng thái “chưa hoàn tất” và yêu cầu bắt buộc `/proc` đã được thay thế bởi code d9378cb, results a5e79e7 và quyết định review mới. Không chạy lại Step A từ lệnh lịch sử bên dưới.

Mốc bàn giao: **12/09/2026**. Code local được kiểm tra tại commit **`a5f84fb`**, branch `master`, working tree sạch trước khi thêm tài liệu này.

Đây là tài liệu tiếp nhận công việc, không phải báo cáo một thí nghiệm mới đã hoàn thành. Luna phải đọc trạng thái Git và artifact thực tế khi bắt đầu; không mặc định mốc này là HEAD mới nhất. Nếu có mâu thuẫn giữa tài liệu, code và chỉ đạo mới của người dùng, nêu rõ trước khi thay protocol.

## 1. Phân vai và quy trình bắt buộc

- **Người phụ trách nghiên cứu/reviewer:** chốt scope, giả thuyết, điều kiện nghiệm thu và quyết định bước tiếp theo.
- **Luna:** triển khai code, tests và tài liệu trên local; push code; sau khi người dùng push kết quả thì pull về kiểm tra và báo cáo.
- **Người dùng:** pull code và chạy lệnh trên server; push kết quả về Git.
- **Luna không có SSH. Không tìm cách SSH hoặc giả định đã truy cập server. Không chạy TensorRT build/benchmark trên local.**

Luồng công việc:

1. Luna đọc context, kiểm tra code, sửa preflight và chạy tests local.
2. Luna review diff, commit/push đúng file trong task; báo commit cụ thể.
3. Luna cung cấp lệnh để người dùng pull và kiểm tra server. Mỗi lệnh là **một dòng vật lý**; không mặc định dùng `nohup`.
4. Người dùng chạy study trên server, gửi log hoặc push artifact.
5. Luna pull kết quả khi local an toàn, xác minh artifact/hash/protocol, tổng hợp báo cáo.
6. Dừng chờ reviewer. Không tự triển khai bước B/C hoặc mở rộng nghiên cứu.

Nếu chưa có kết quả server, chỉ báo “code/tests local đã kiểm tra; server chưa kiểm chứng”. Unit tests không chứng minh TensorRT end-to-end đã chạy thành công.

## 2. Nghiên cứu này là gì?

Tên làm việc: **Suy giảm chất lượng phát hiện biển báo do lượng tử hóa INT8 theo kích thước đối tượng và điều kiện ảnh, hướng tới triển khai edge có kiểm soát.**

Hướng này liên quan đề tài CalibShift-INT8 trong tài liệu ý tưởng của người dùng, nhưng repo **chưa triển khai calibration bank + condition selector động**. Không mô tả đó là phương pháp hiện có.

Vấn đề đang kiểm chứng: chênh lệch chất lượng sau lượng tử hóa đến từ lựa chọn calibration images, lựa chọn precision/tactic khi build, hay độ nhạy của các thành phần detector? Các nguyên nhân chưa được tách đầy đủ.

Mục tiêu journal từng thảo luận gồm IVC và các ứng viên phù hợp khác. Không có cam kết chấp nhận hoặc bảo đảm Q2. Số model/device không tự tạo tính mới khoa học.

## 3. Scope đã khóa

| Thành phần | Phạm vi |
|---|---|
| Dataset | CCTSDB2021; prohibitory, mandatory, warning |
| Train | 14.720 ảnh |
| Dev | 1.636 ảnh, split seed 42 |
| Positive official test | 1.500 ảnh |
| Negative official test | 500 ảnh, stress test riêng |
| Frozen models | YOLOv8, YOLO11, YOLO26; mỗi họ n/s/m/l/x: tổng 15 checkpoint |
| Training đã thực hiện | 640×640, 100 epochs theo protocol chung; batch vật lý khác theo quy mô, không nói mọi hyperparameter giống hệt |
| Development hiện tại | Chỉ frozen YOLO11n, RTX8000 |
| Step A evaluation | Chỉ dev, không official test/domain/negative |

Không retrain 15 model. Không thêm external training dataset. TT100K/MTSD/CURE-TSD không thuộc manuscript đang thực hiện. Không mở rộng INT8 15 model trước review/gate.

Official test đã được đánh giá trong lịch sử. Không mô tả là hoàn toàn chưa từng xem. Không tiếp tục dùng kết quả official test để chọn/tinh chỉnh calibration policy, thresholds hoặc phương pháp. Calibration chỉ lấy ảnh train; dev dùng cho phát triển và kiểm chứng.

Hardware tương lai người dùng xác nhận: Jetson Xavier NX, Jetson AGX Xavier, Jetson Nano, Raspberry Pi 5 CPU, Pi 5 + Hailo 26 TOPS. Đó không phải danh sách thiết bị đã benchmark xong. Backend-native quantization khác nhau; cùng source weights/calibration IDs không đồng nghĩa TensorRT và Hailo là cùng một mô hình lượng tử hóa về số học.

## 4. Lịch sử và bằng chứng đã có

### Calibration v1 và v2

- Uniform: lấy mẫu train-only theo seed.
- Low-Luminance: ranking độ sáng; các seed có thể chọn cùng một tập ảnh. Một selection không trở thành năm mẫu độc lập chỉ vì đổi tên seed; SD của một run không chứng minh không có biến thiên.
- VCSC v1 equal-quota: K-means K=8, chia quota gần đều. Không vượt decision gate; giữ làm kết quả lịch sử, không scale.
- VCSC v2 proportional: quota theo tỷ lệ cụm; đã thử trên dev. Bằng chứng hiện có không hỗ trợ tuyên bố vượt Uniform.
- VCSC v1 thay clustering theo seed; SD lịch sử không phải chỉ riêng sampling noise, còn có clustering và có thể có build noise.

Đã phát hiện Ultralytics/TensorRT đọc calibration cache cạnh ONNX dùng chung. Các artifact bị ảnh hưởng được quarantine. **Không tái sử dụng số liệu quarantine làm bằng chứng policy.** Export sau đó dùng workspace riêng mỗi engine và ghi cache hash/provenance.

### Measurement audit G0

- XML archive có cả train/test; tên `<filename>` bên trong XML không đáng tin để ghép full test. Đã dùng XML member stem với kiểm tra coverage.
- Đã phát hiện khác biệt giữa prediction từ lượt `predict` riêng và lượt `val`. Hiện dùng same-pass validator capture, replay và native rematch đã kiểm chứng.
- Hai trường hợp annotation cần ghi nhận: `00751.jpg` có bbox XML/YOLO lệch khoảng 0,5 pixel; `04492.jpg` có bbox vượt chiều rộng 1 pixel. Không sửa dữ liệu để ép audit pass.
- Phân tích size dùng COCO/XML riêng; không trộn mAP của evaluator này với Ultralytics full hoặc evaluator size lịch sử.
- Size bins theo diện tích gốc XML: XS ≤210; S (210,400]; M (400,1000]; L (1000,2000]; XL >2000 px². Dev có 190/642/995/498/381 objects, tổng 2.706. Xem implementation/protocol để hiểu matching, ignore và coordinate convention.
- Đã có 1.000 paired image bootstrap trên dev và phân tích localization. Các CI có điều kiện trên engine cố định; không phải CI cho retraining hoặc mọi calibration seed. Kết quả cho thấy suy giảm localization/small-object, chưa chứng minh cơ chế duy nhất.
- Candidate diagnostic ghi số GT có nhiều candidate cùng class, không được gọi trực tiếp đó là số duplicate false positives.

### Engine inspection và lý do cần Step A

Inspector của ba INT8 có thông tin chi tiết; FP16 chỉ names-only. Conv weight types Int8/Float ghi nhận:

| Engine | Int8 weights | Float weights |
|---|---:|---:|
| Uniform | 64 | 21 |
| Low-Luminance | 74 | 11 |
| VCSC proportional | 78 | 7 |

Các số này **không chứng minh precision mọi phép tính/accumulator**, không phải tỷ lệ FLOPs INT8. TensorRT có thể lựa chọn tactic/precision khác giữa các build. Cần kiểm soát calibration trước khi gán chênh lệch cho policy hoặc head.

## 5. Đọc gì trước khi làm?

Đọc `AGENTS.md` áp dụng nếu có, sau đó theo thứ tự:

1. [MEASUREMENT_AUDIT_G0.md](MEASUREMENT_AUDIT_G0.md), đặc biệt mục “Bước A đã được duyệt”.
2. [uniform_build_repeat.py](../scripts/uniform_build_repeat.py) và [test_uniform_build_repeat.py](../tests/test_uniform_build_repeat.py).
3. [capture_cctsdb_validator.py](../scripts/capture_cctsdb_validator.py), [run_g0_int8_capture.py](../scripts/run_g0_int8_capture.py), GPU lock trong [run_architecture_matrix.py](../scripts/run_architecture_matrix.py).
4. [YOLO11N_CALIBRATION_DEVELOPMENT_V2.md](YOLO11N_CALIBRATION_DEVELOPMENT_V2.md) và config v2 để hiểu lineage.
5. [RESEARCH_VIABILITY_Q2_20260909.md](RESEARCH_VIABILITY_Q2_20260909.md): bối cảnh thẩm định ngày 09/09, không thay thế báo cáo mới hơn.

[IVC_STUDY_V1.md](IVC_STUDY_V1.md) là **template scale-up đã bị chặn/superseded**, không phải chỉ đạo chạy 15 model.

Artifact tham chiếu (đọc, không sửa/ghi đè):

- `results/measurement_audit_v1/server_fp16_capture_v1/`
- `results/measurement_audit_v1/server_native_size_v1/`
- `results/measurement_audit_v1/server_int8_capture_v1/`
- `results/measurement_audit_v1/server_dev_error_bootstrap_v1/`
- `results/measurement_audit_v1/server_engine_inspection_v1/`
- `results/measurement_audit_v1/local_candidate_diagnostic_v1/`
- `results/calibration_method_v1/` và `results/calibration_method_v2/`: giữ đúng version/lineage, không nhập chung kết quả.

## 6. Trạng thái Step A tại thời điểm bàn giao

**Code đã triển khai; chưa có kết quả build-repeat hoàn tất được xác nhận.**

- `4f6cae8`: triển khai controlled three-build repeat.
- `a5f84fb`: preflight chấp nhận distribution `onnxruntime`, `onnxruntime-gpu`, `onnxruntime-qnn`, kiểm tra import/providers thật. Không tự cài package.
- Protocol ghi 41 tests local đạt ở mốc đó. Luna phải chạy lại và báo test command/count của phiên làm việc mới, không dùng số 41 làm kết quả của mình.
- Server từng dừng vì preflight chỉ tìm metadata `onnxruntime`; sửa tại `a5f84fb`.
- Sau đó server dừng vì các CUDA process khác. Chưa có bằng chứng Step A đã chạy sau lỗi đó.

Log ngày 11/09 (chỉ lịch sử, không phải PID/trạng thái hiện tại):

- PID `2925282`, user ubuntu: `python scripts/run_channel_ablation.py ...`, CHBMIT, GPU SM khoảng 8–27% ở các mẫu cuối. Đây là workload cạnh tranh thực sự dù chỉ khoảng 312 MiB VRAM.
- PID `3129226`: training segmentation; đã không còn trong snapshot sau. Phải kiểm tra lại thực tế.
- PID `644963`/`644977`, user archlab/duy: `/snap/snapd-desktop-integration/391/usr/bin/snapd-desktop-integration`, C+G khoảng 5 MiB mỗi process. Preflight hiện chặn cả hai.
- Snapshot server ghi driver `595.71.05`, CUDA maximum hiển thị `13.2`; đó không phải CUDA runtime của PyTorch. Không đổi environment theo dòng CUDA của `nvidia-smi`.

## 7. Task Luna thực hiện ngay trên local

### Sửa preflight một cách hạn chế

1. Kiểm tra `snapshot()`/`ensure_idle()` và mọi call site chuẩn bị/build/evaluate. Không làm mất GPU lock.
2. Thiết kế phân loại process có thể kiểm thử offline. Ngoại lệ desktop phải dựa trên executable path xác minh, không dựa riêng tên process, UID, VRAM thấp hoặc PID cũ.
3. Nếu cho phép snap desktop qua allowlist, giới hạn đúng package/executable structure; đường dẫn revision có thể thay đổi nhưng không được mở thành allowlist mọi `/snap/*`. Ghi cách xác minh, resolved executable và lý do ngoại lệ vào artifact.
4. Python training, process không xác định hoặc không xác minh được phải tiếp tục chặn. Xử lý PID race/permission error minh bạch, fail closed khi không đủ chứng cứ.
5. Desktop ngoại lệ vẫn phải xuất hiện trong snapshot/report. Allowlist không chứng minh GPU hoàn toàn không có nhiễu; không viết như vậy.
6. Không có blanket bypass, không tự kill/pause process, không đổi clock/power limit hoặc nâng cấp package.

Tests tối thiểu: desktop hợp lệ; training ít VRAM vẫn bị chặn; tên giống desktop nhưng executable khác; không đọc được executable; process race; log ngoại lệ; giữ kiểm tra lock/hash và chống ghi đè. Không thay test để bỏ guard.

Nếu chưa đủ thông tin để xác minh ngoại lệ trên server, cung cấp lệnh read-only một dòng cho người dùng lấy thông tin rồi chờ. Không đoán và tự whitelist.

### Giới hạn file thay đổi

Ưu tiên script Step A, tests liên quan và tài liệu hướng dẫn. Nếu phải sửa shared helper, giải thích tác động và chạy regression tests. Không sửa artifact lịch sử, evaluator/size thresholds, frozen calibration selection hoặc source weights.

## 8. Contract thí nghiệm không được thay

- Weights: `results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt`.
- SHA256: `3e5fc7a2148c16539cd9fb7cc7cacd81a4eec1dfc28143cdf9b6dcd872ba4ab8`.
- Calibration: `data/processed/cctsdb2021_clean/calibration/uniform_v2_s42_n1024/`, train-only, không duplicate, khớp manifest historical và source images.
- Export ONNX một lần: opset17, simplify, static `[1,3,640,640]`, không Q/DQ.
- Freeze image order và preprocessed uint8 tensors; hash float32 input stream. Không decode/sample lại giữa các build.
- Build1 tạo MinMax calibration cache; build2/3 dùng bản sao cùng cache theo contract, không recalibrate. Hash cache historical chỉ đối chiếu, không ép khớp.
- Timing cache rỗng riêng mỗi repeat để quan sát retiming. Mỗi repeat là process riêng.
- Workspace 4 GiB; optimization level3; average timing iterations1; INT8 bật, FP16 tắt, TF32 tắt; Sigmoid FP32 với OBEY như protocol; không thêm bbox/classification constraint.
- Dev capture/replay/native matching/COCO-size verification cùng reference. Không official test/negative, không bootstrap mới.
- Lưu log, inspector, hashes và GPU/environment snapshots. Không tự xóa artifact hoặc auto-resume lên output cũ.

Đây là **baseline kiểm soát mới**, không tuyên bố bitwise recreation engine cũ. Ba repeat là chẩn đoán giới hạn, không chứng minh tính xác định hoặc ước lượng chính xác toàn bộ variance.

## 9. Server workflow — người dùng chạy, Luna hướng dẫn

Local: `D:\Research\paper`, Windows PowerShell. Server: Ubuntu, `/home/ubuntu/Dung_TDTU/nighttime-tsd-new`. Không gửi cú pháp PowerShell cho Bash hoặc ngược lại.

Server interpreter: `local/g0_size_env/bin/python`, kế thừa `nighttime-tsd`. Code ở mốc bàn giao kiểm tra torch `2.5.1+cu121`, Ultralytics `8.4.102`, TensorRT `10.16.1.11`, NumPy `2.4.4`, pycocotools `2.0.10`. Nếu không khớp, báo để review, không tự nâng/hạ package.

Sau khi Luna push code, cung cấp commit và lệnh pull riêng. Mẫu sau chỉ áp dụng nếu người dùng vẫn ở branch `master`; xác minh branch/status trước, không tự chuyển/ghi đè nhánh research mới:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master
```

Kiểm tra GPU/workload và output trước. Lệnh chạy khi đủ điều kiện:

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/uniform_build_repeat.py --out-dir results/measurement_audit_v1/server_uniform_build_repeat_v1
```

Đây là tác vụ nhiều phút, chạy foreground. Theo dõi `FROZEN CALIBRATION`, build1/3 đến3/3, validation/verifications, rồi `DONE` và `repeat_summary.json`. Không coi im terminal là treo; không mở thêm job tương tự.

Nếu output tồn tại, kiểm tra partial artifact và báo trước. Không xóa, không tự đổi tên để che lỗi hoặc tự tiếp tục phase chưa review. Nếu workload cạnh tranh xuất hiện trong run, báo vi phạm/hạn chế và review tính hợp lệ; không tự công nhận run.

Sau khi xác minh đủ artifact, Luna cung cấp lệnh Git add đúng các JSON, cache và `.log.gz` được study tạo; kiểm tra file thực tế trước khi dùng glob. Commit và push là hai lệnh riêng. Nếu `nothing to commit`, kiểm tra commit/status trước; không để `&& commit` khiến bỏ qua push cần thiết.

Không stage `source.pt`, `source.onnx`, `calibration_uint8.npy`, engine hoặc toàn bộ `results` trong task này. Binary giữ server, hash trong provenance. Không dùng `git add .`. Dùng workaround Git server ở trên nếu cần tránh lỗi thư viện conda. Không thay Git identity/account toàn cục.

## 10. Nghiệm thu và báo cáo

### Trước khi người dùng chạy server

- Diff chỉ gồm phạm vi đã duyệt; tests guard và regression liên quan đạt.
- Báo tests đã chạy, dependency/local limitations và commit đã push.
- Cung cấp lệnh kiểm tra + chạy + hướng dẫn artifact, mỗi lệnh một dòng.
- Chưa đánh dấu thí nghiệm hoàn tất.

### Sau khi người dùng push kết quả

- Đủ ba engine provenance và dev evaluation; không trộn artifact study/policy khác.
- ONNX/calibration input/cache contract giống nhau đúng thiết kế; build2/3 không recalibrate.
- Đủ logs, inspector, snapshots và hash. Engine hash khác không đủ kết luận predictions/accuracy khác.
- Replay/native matching pass; xem mọi warning/partial/error trước khi tổng hợp.
- Báo từng build và mean/sample SD/min/max/range **điểm phần trăm** cho COCO/XML full, XS/S/M/L/XL AP50/AP50–95; Ultralytics báo riêng.
- So sánh inspector signatures với metric, mô tả association, không tự suy causal attribution.
- Ghi GPU/environment/noise limitations; before/after snapshots không chứng minh không có workload ở mọi thời điểm giữa run.

Định dạng bàn giao cuối:

1. Commit và files đổi, lý do.
2. Tests và phần chưa kiểm chứng.
3. Trạng thái server: chưa chạy / bị chặn / partial / hoàn tất có review.
4. Artifact locations, integrity/protocol checks.
5. Bảng ba build và diễn giải giới hạn.
6. Vấn đề còn mở và đề nghị reviewer quyết định.

**Điểm dừng bắt buộc:** `step_A_completed_review_required` khi đủ điều kiện. Không tự chạy B (bbox FP32), C (classification control), thêm calibration policy, INT8 matrix 15 model hoặc hardware benchmark. Reviewer quyết định bước tiếp theo dựa trên bằng chứng Step A.
