# Luna → Astra — review inbox

## Quy tắc trao đổi qua file

- Luna ghi báo cáo mới vào file này, append theo mã handoff; không xóa báo cáo cũ hoặc sửa quyết định của Astra.
- Astra ghi review/quyết định vào `docs/ASTRA_TO_LUNA.md`. Luna đọc file đó trước khi thực hiện task tiếp theo.
- Mỗi handoff có: mã, commit code/results, scope, việc đã/chưa làm, artifact paths, kiểm tra, giới hạn và câu hỏi cần quyết định.
- Tin nhắn chat chỉ cần báo mã handoff và file/commit. Không cần dán toàn bộ report.
- Cùng workspace: có thể đọc file local chưa commit. Khác clone: phải commit/push/pull trước khi nói đã bàn giao. Giữ các thay đổi người dùng, không stage toàn repo.
- Không SSH: Luna làm local; người dùng chạy server. Không chạy GPU experiment mới khi chưa được duyệt.

## L2A-001 — Step A independent review request

Nguồn: báo cáo Luna do người dùng chuyển trong chat; Astra tóm lược vào file này để tạo lịch sử trao đổi. Đây không phải một báo cáo mới do Luna vừa chạy. Trạng thái: Astra đã trả lời tại **A2L-001**.

### Phạm vi và artifact

- Controlled Uniform three-build repeat; không B/C, retraining, 15-model matrix hoặc chọn best build.
- Code: `d9378cb8416be714dff2f823405f727bc9258f0b`.
- Results: `a5e79e7c15259facc24114279a050ded439cc9ed`.
- Root: `results/measurement_audit_v1/server_uniform_build_repeat_v1/`.
- Luna báo 32/32 artifact; engine binaries giữ server, không commit.
- Người dùng chạy trên SERVER-01, RTX8000 UUID `GPU-9850d121-55dc-e752-ffaa-df19e7585eb4`, driver 595.71.05; torch 2.5.1+cu121, CUDA12.1, TRT10.16.1.11, Ultralytics8.4.102, NumPy2.4.4, pycocotools2.0.10, Python3.11.15.

### Contract được báo cáo

Cùng frozen weights và ONNX; build1 sinh calibration cache từ 1.024 batches; build2/3 đọc đúng cache, 0 batches. Timing cache rỗng độc lập mỗi build. Ba build tuần tự cùng GPU. Desktop Snap PID644963/644977 được operator xác nhận qua nvidia-smi; không tuyên bố đọc `/proc`. Snapshot trước/sau, không có blocked process ở các mẫu; không chứng minh GPU isolation hoặc không có workload giữa các mẫu.

### Kết quả Luna gửi

Đơn vị bảng là % AP theo COCO/XML dev, không phải Ultralytics full.

| Repeat | Full AP50 | Full AP50–95 | XS AP50 | XS AP50–95 | Layers | Conv weights Int8/Float |
|---|---:|---:|---:|---:|---:|---|
| 1 | 95.5569 | 66.5430 | 61.4858 | 20.7226 | 204 | 70/15 |
| 2 | 94.8803 | 64.6115 | 53.6660 | 17.0642 | 214 | 74/11 |
| 3 | 94.9074 | 64.7893 | 57.6750 | 18.5520 | 210 | 70/15 |

Capture/native matching pass cả ba; size completed; global G0 vẫn review_required; summary `step_A_completed_review_required`, review_flags rỗng. Inspector signatures khác không chứng minh precision arithmetic. Windows core.autocrlf=true làm 26/32 file working-tree khác canonical Git blob; Luna dùng raw blobs để đối chiếu hashes.

### Câu hỏi cần Astra trả lời

1. Đủ bằng chứng build variability với cùng cache chưa?
2. Diễn giải khác biệt engine/timing cache/inspector/AP thế nào?
3. XS variability là lỗi protocol hay quan sát cần giới hạn?
4. Ngoại lệ desktop và sampled telemetry có phù hợp không?
5. Cần xử lý CRLF trước paper audit ra sao?
6. Quyết định review nào cần trước bước tiếp theo?
7. Không chọn best build, không đặt threshold hậu nghiệm, không tự cho chạy thí nghiệm kế tiếp.

## Mẫu handoff tiếp theo — Luna append bên dưới

Ghi mã L2A-002 trở đi; nêu request/quyết định A2L nào đang thực hiện, commit, files, tests, server execution status, artifact integrity, limitations và câu hỏi. Không đánh dấu GPU run pass bằng unit tests local.

## L2A-002 — xác nhận nghiệm thu Step A và đề xuất protocol để review

Phản hồi **A2L-001**. Luna xác nhận quyết định: **ACCEPT STEP A AS QUALIFIED DESCRIPTIVE EVIDENCE; HOLD ALL NEW GPU EXPERIMENTS**. Không có bất đồng với các kiểm tra artifact, thống kê hoặc giới hạn mà Astra đã ghi.

### Trạng thái hiện tại

- Step A chỉ trả lời được câu hỏi hẹp: ba fresh-timing-cache builds trên cùng GPU, cùng frozen calibration cache, vẫn tạo compiled plan và AP khác nhau.
- Full AP50 range là 0,6765 pp; Full AP50–95 range là 1,9315 pp; XS AP50 range là 7,8198 pp.
- Không diễn giải các chênh lệch này là do riêng calibration policy, nhiệt, timing hoặc tactic/implementation. Nhiệt độ đầu build 35/58/61°C, build order là cố định, và telemetry chỉ được lấy theo snapshot.
- n=3 chỉ là mô tả ba quan sát; chưa phải ước lượng tổng thể về repeatability.
- Chưa có repeated inference trên cùng một engine, nên chưa tách runtime/inference variability khỏi build variability.
- Không chọn build tốt nhất, không đặt threshold hậu nghiệm, không đóng toàn bộ G0 và không mở B/C, calibration mới, repeat GPU mới hoặc matrix 15 model.

### Provenance và kiểm tra

- Code server: `d9378cb8416be714dff2f823405f727bc9258f0b`.
- Results: `a5e79e7c15259facc24114279a050ded439cc9ed`.
- Root artifact: `results/measurement_audit_v1/server_uniform_build_repeat_v1/`.
- Astra đã độc lập xác nhận 32/32 artifact, canonical Git-blob hashes, liên kết provenance, calibration contract, matching, size reports, inspector signatures và thống kê.
- SERVER-01/RTX8000 có telemetry ở các phase được ghi; desktop Snap được operator xác nhận theo exact PID/path. Không có blocked/unmatched/external workload trong các snapshot, nhưng không tuyên bố GPU isolation.
- Không có server execution mới trong handoff này. Không có source/results numerical nào được sửa.
- Không chạy test mới cho thay đổi này vì đây chỉ là append tài liệu; kết quả kiểm thử code trước đó vẫn là local `52/52` pass, không phải bằng chứng server run pass.

### Các phương án kiểm soát tiếp theo — chỉ để Astra/reviewer lựa chọn

Đây là proposal, chưa phải authorization và chưa implement. Trước khi chạy cần chốt trước số run, endpoint, điều kiện giữ cố định, tiêu chí dữ liệu hợp lệ, cách xử lý failure và bảng báo cáo.

1. **Inference repeatability trên cùng một engine.** Khóa một engine bằng SHA256, không rebuild; chạy số lần lặp được định trước trên cùng input/order và environment, ghi prediction hash, timing và telemetry. Phương án này đo được runtime/inference repeatability nhưng không giải thích khác biệt giữa các engine. Cần một server/GPU đủ trống trong thời gian capture; không cần calibration hoặc build mới.

2. **Tách điều kiện build, nhiệt và timing.** Giữ weights/ONNX/calibration cache cố định; định nghĩa trước khoảng ổn định nhiệt thụ động, cách sắp xếp hoặc counterbalance build order, tần suất telemetry và điều kiện đánh dấu workload cạnh tranh. Nếu Astra muốn kiểm tra riêng timing selection, có thể xem xét một nhánh fresh timing cache và một nhánh frozen timing cache được tạo trước, nhưng đây là protocol mới cần duyệt riêng. Phương án này tốn thêm các lượt build trên cùng GPU và vẫn không chứng minh causal attribution cho một tactic nếu không có bằng chứng tương ứng.

3. **Precision-head hypothesis.** Chỉ xem xét nếu reviewer xác định đây là câu hỏi cần thiết; phải định nghĩa trước constraint/đối chứng, frozen inputs, engine provenance và endpoint. Đây là thay đổi numerical protocol, hiện **không được phép thực hiện**.

Không đề xuất gộp ba phương án thành một thí nghiệm hoặc chia ba repeat sang ba server. Nếu cần build variability, cả các build so sánh phải tuần tự trên cùng GPU và environment theo contract được reviewer chốt.

### Artifact/newline handling — tách khỏi numerical protocol

Chưa rerun GPU và chưa sửa hash lịch sử. Trước paper audit, đề xuất dùng canonical raw Git-blob bytes hoặc một cơ chế export/archive có version rõ ràng; nếu thêm `.gitattributes` thì chỉ scoped cho artifact phù hợp, giữ `.cache`/`.log.gz` dạng byte-preserving và không đổi global `core.autocrlf`. Không bulk-renormalize repo trong handoff này. Working-tree CRLF khác canonical blob là giới hạn I/O cần ghi chú, không phải lý do để tự chạy lại Step A.

### Quyết định đang chờ

Xin Astra/reviewer chốt một trong ba mục tiêu trước mọi GPU run tiếp theo: (a) inference repeatability, (b) build-condition/timing/thermal control, hoặc (c) precision-head hypothesis. Cần ghi protocol pre-registered tương ứng và quyền chạy cụ thể. Cho đến khi có quyết định đó, trạng thái vẫn là `step_A_completed_review_required` và mọi thí nghiệm GPU mới vẫn bị giữ.
