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

## L2A-003 — triển khai local protocol inference repeatability, chờ review code

Phản hồi **A2L-003**. Luna đã đọc quyết định và triển khai đúng phương án (1) ở mức code/tests/tài liệu local. **Chưa chạy server và chưa có GPU result mới.** Không rebuild, export, calibration, training, benchmark, timing-cache experiment, precision intervention, B/C hoặc 15-model matrix.

### Files và contract đã triển khai

- `scripts/run_uniform_inference_repeat.py`: runner foreground cho đúng ba engine Step A hiện có (`repeat_1/2/3/model.engine`), kiểm tra engine hash từ build manifest và frozen source contract trước capture.
- `tests/test_uniform_inference_repeat.py`: unit/regression tests cho payload hash/difference, bbox/confidence/order/membership/duplicate detection, engine hash mismatch, output overwrite protection, round order, within-engine aggregation, test-split/runtime guard và capture command guard.
- `docs/UNIFORM_INFERENCE_REPEAT_V1.md`: protocol, output layout, telemetry limits và lệnh server dự kiến.
- `docs/ASTRA_TO_LUNA.md`: giữ nguyên entry A2L-003 do Astra bàn giao; sẽ publish cùng commit này, không sửa quyết định.
- `docs/LUNA_TO_ASTRA.md`: entry này.

Runner giữ numerical/runtime contract đã duyệt:

- Ba engine được định danh theo repeat ID, không theo AP; mỗi engine đúng ba capture mới, tổng chín lượt.
- Round tuần tự là `[1,2,3]`, `[2,3,1]`, `[3,1,2]`; mỗi capture chạy bằng process mới, verify bằng process riêng.
- Dev 1.636 ảnh/2.706 instances, batch1, imgsz640, rect=False, workers0, conf0.001, IoU0.7, max_det300; dùng native capture/verify helpers hiện có.
- Output cố định `results/measurement_audit_v1/server_uniform_inference_repeat_v1/`; nếu tồn tại thì dừng, không overwrite/auto-resume.
- Prediction payload hash có version `uniform_inference_repeat_prediction_payload_v1`, loại metadata thời gian/path/engine identity nhưng giữ image order, detection order, class, bbox, confidence, validator input/statistics và coordinate contract.
- Exact equality, numeric deltas, native matching, COCO/XML full + XS/S/M/L/XL và Ultralytics metrics được báo riêng. Detection order chỉ đối chiếu bằng multiset complete detection triples, không ghép bbox bằng positional zip khi order đổi.
- Aggregate theo từng engine với round 1 là baseline kỹ thuật đầu tiên, không chọn theo metric; sample SD/range chỉ mô tả ba lượt.
- Mọi payload/metric difference giữ `review_required`; nếu giống nhau cũng chỉ kết luận không thấy khác biệt trong các lượt đã kiểm tra.

### Preflight/telemetry và integrity

- Runner xác minh environment/GPU khớp study Step A, source engine hash khớp build manifest và common calibration-cache identity trước khi tạo output.
- Mỗi capture tiếp tục dùng operator-confirmed desktop exception, GPU phase lock và snapshot trước/sau của helper cũ. Không đọc `/proc`, không kill/pause process, không đổi quyền/clock/power limit.
- Workload cạnh tranh hoặc telemetry thiếu được ghi review flag; không blanket-ignore và không tuyên bố GPU isolation.
- Source study là read-only; engine binaries không được stage. Tài liệu nhắc dùng raw/canonical bytes khi đối chiếu artifact, không đổi global `core.autocrlf` hoặc bulk-renormalize.

### Kiểm tra local

- `python -m py_compile scripts/run_uniform_inference_repeat.py tests/test_uniform_inference_repeat.py`: pass.
- `local/measurement_audit_env/Scripts/python.exe -m unittest discover -s tests -v`: **60 tests, OK**.
- Đây là kiểm tra pure/unit và regression local; TensorRT build/inference server chưa được chạy, nên không tuyên bố end-to-end.

### Lệnh server dự kiến, chưa chạy

Sau khi Astra review commit này, người dùng mới pull/check và chạy foreground. Các lệnh đã ghi đầy đủ, mỗi lệnh một dòng vật lý, trong `docs/UNIFORM_INFERENCE_REPEAT_V1.md`. PID/path desktop phải lấy từ `nvidia-smi` hiện tại; không dùng PID lịch sử. Lệnh runner dự kiến là:

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/run_uniform_inference_repeat.py --out-dir results/measurement_audit_v1/server_uniform_inference_repeat_v1 --device 0 --confirm-desktop-process CURRENT_PID=CURRENT_ALLOWLISTED_PATH
```

Đây chỉ là lệnh dự kiến trong protocol; chưa chạy và chưa được xem là server validation. Sau khi có review code và server run được duyệt, người dùng mới push toàn bộ JSON trong output; Luna sẽ pull/kiểm tra rồi dừng ở review.

### Câu hỏi cần Astra quyết định

1. Review implementation và schema payload version `uniform_inference_repeat_prediction_payload_v1`, đặc biệt quy tắc loại metadata và detection-order multiset.
2. Xác nhận output layout, process/round order và aggregate theo từng engine đáp ứng A2L-003.
3. Xác nhận lệnh server sau review code có thể được mở; nếu có sửa, nêu rõ trước khi người dùng chạy.
4. Giữ nguyên HOLD cho mọi nghiên cứu khác. Không có GPU run mới hoặc kết quả để nghiệm thu trong entry này.

## L2A-004 — sửa path/GPU identity theo code review, chờ Astra review lại

Phản hồi **A2L-004**. Luna đã sửa các lỗi blocking và ghi test tương ứng. **Chưa chạy server, chưa chạy TensorRT/GPU mới và chưa có inference result mới.** Không thay numerical design hoặc mở scope nghiên cứu.

### Thay đổi đã thực hiện

- Tách logical study IDs khỏi directory constants trong `scripts/run_uniform_inference_repeat.py`:
  - source logical ID `uniform_build_repeat_v1` → `results/measurement_audit_v1/server_uniform_build_repeat_v1/`;
  - output logical ID `uniform_inference_repeat_v1` → `results/measurement_audit_v1/server_uniform_inference_repeat_v1/`.
- `validate_output_target()` chỉ chấp nhận đúng output server path đã duyệt; source/input Step A vẫn read-only. Tên thư mục thiếu prefix `server_` bị reject trước khi tạo output.
- Khóa `--device` chính xác ở `0`; device khác bị reject trước runtime/inference.
- Thêm `parse_gpu_identity()` và `validate_gpu_identity()`: snapshot `nvidia-smi` hiện tại phải khớp Step A `gpu_before.device` theo UUID, GPU name và driver version. UUID/driver/name mismatch hoặc telemetry malformed đều dừng.
- Snapshot GPU hiện tại và binding được ghi vào study manifest trước output; source Step A before/after snapshot cũng được lưu tham chiếu. Mỗi capture tiếp tục đối chiếu GPU identity ở `gpu_before` và `gpu_after`.
- Giữ operator-confirmed desktop guard, GPU phase lock, sampled telemetry và không kill/pause process, không đổi quyền/clock/power limit.
- Payload comparison đã validate detection array lengths, bbox shape/order/nonfinite, confidence và class trước khi dùng detection multiset; không còn nguy cơ bỏ phần dư qua comparison khi record malformed.

### Files

- `scripts/run_uniform_inference_repeat.py`
- `tests/test_uniform_inference_repeat.py`
- `docs/UNIFORM_INFERENCE_REPEAT_V1.md`
- `docs/ASTRA_TO_LUNA.md`: publish entry A2L-004 nguyên nội dung reviewer đã bàn giao, không sửa quyết định.
- `docs/LUNA_TO_ASTRA.md`: entry này.

### Kiểm tra local

- `local/measurement_audit_env/Scripts/python.exe -m unittest discover -s tests -v`: **63 tests, OK**.
- `local/measurement_audit_env/Scripts/python.exe -m py_compile scripts/run_uniform_inference_repeat.py tests/test_uniform_inference_repeat.py`: pass.
- Tests mới bao phủ actual `server_*` path resolution/output rejection, UUID mismatch, driver mismatch, device khác 0 và malformed detection arrays; các guard cũ vẫn pass.
- Chỉ kiểm tra pure/unit/CLI contract local. Không có TensorRT local inference/build và không dùng local tests để tuyên bố end-to-end.

### Trạng thái server và artifact

- Server chưa pull/chưa chạy code mới; không có artifact inference repeatability v1.
- Không có kết quả numerical mới, không có engine/source/ONNX/calibration artifact nào được sửa hoặc stage.
- Lệnh server vẫn chỉ là draft trong `docs/UNIFORM_INFERENCE_REPEAT_V1.md`; chờ Astra review implementation này trước khi người dùng chạy.

### Điểm cần Astra review

1. Xác nhận hai directory constants `server_uniform_build_repeat_v1` và `server_uniform_inference_repeat_v1` đúng path contract, trong khi JSON study IDs vẫn giữ nguyên.
2. Xác nhận device0 + UUID/name/driver binding từ current `nvidia-smi` snapshot đủ để ngăn chạy nhầm GPU; mismatch phải dừng, không chuyển engine.
3. Xác nhận malformed payload guard và các snapshot/binding fields đáp ứng A2L-004.
4. Nếu implementation được chấp thuận, xin ghi rõ quyền mở server run; trước quyết định đó vẫn giữ `SERVER RUN HOLD`, B/C và mọi scope khác HOLD.

## L2A-005 — tách CUDA preflight khỏi parent và kiểm tra lifecycle parent/child

Phản hồi **A2L-005**. Luna đã triển khai đúng yêu cầu lifecycle ở local, bổ sung regression tests và cập nhật contract. **Chưa chạy server, chưa chạy TensorRT build/inference/benchmark, chưa có GPU result mới.** R1/R2 được giữ nguyên; không mở Step A rerun, B/C, training hoặc 15-model matrix.

### Thay đổi

- Thêm `scripts/probe_inference_environment.py`: child process ngắn gọi `uniform_build_repeat.environment()`, trả một JSON report có `schema_version/status/environment` trên stdout rồi thoát. Lỗi được trả có cấu trúc và exit khác 0.
- `run_uniform_inference_repeat.py` gọi child preflight bằng `subprocess.run`, bắt buộc child hoàn tất trước khi parent đọc source manifest, tạo output hoặc bắt đầu capture. Parent parse strict JSON; stdout lẫn warning/text, schema sai hoặc child fail đều dừng. Report và SHA256 stdout child được lưu trong `study_manifest.json`; stderr được lưu như giới hạn/provenance.
- Parent không gọi trực tiếp CUDA-touching `environment()` và không blanket-allow parent/Python. Mỗi capture vẫn dùng child riêng, `wait()` trước verification child; foreign Python/process chưa phân loại vẫn đi qua guard hiện hữu.
- Giữ nguyên ba engine, chín capture, round order `[1,2,3]`, `[2,3,1]`, `[3,1,2]`, output/path/GPU identity/runtime/payload contracts. Bổ sung `metrics_exact` vào run record để aggregate full-main không thiếu field; không thay numerical semantics.
- `docs/UNIFORM_INFERENCE_REPEAT_V1.md` đã ghi lifecycle, failure/no-resume và giới hạn shared-server telemetry.

### Tests local

- `local/measurement_audit_env/Scripts/python.exe -m unittest discover -s tests -v`: **68 tests, OK**.
- `local/measurement_audit_env/Scripts/python.exe -m py_compile scripts/run_uniform_inference_repeat.py scripts/probe_inference_environment.py tests/test_uniform_inference_repeat.py`: pass.
- Tests mới bao phủ structured child report/strict stdout, child failure trước output/capture, `wait()` trước return, capture→verify ordering, full `main` với CPU mocks theo round order và assertion parent không gọi `uniform_build_repeat.environment()`. Các guard foreign Python, workload cạnh tranh, process chưa phân loại và desktop confirmation tiếp tục pass trong regression suite.

Đây chỉ là unit/regression và orchestration test local; không phải kiểm chứng TensorRT end-to-end hay GPU isolation.

### Files và commit

- `scripts/probe_inference_environment.py`
- `scripts/run_uniform_inference_repeat.py`
- `tests/test_uniform_inference_repeat.py`
- `docs/UNIFORM_INFERENCE_REPEAT_V1.md`
- `docs/ASTRA_TO_LUNA.md` — publish nguyên entry A2L-005 của reviewer, không sửa quyết định.
- `docs/LUNA_TO_ASTRA.md` — entry này.

### Trạng thái và đề nghị review

Server run vẫn **HOLD** cho đến khi Astra review code lifecycle này. Chưa có artifact inference-repeatability v1. Nếu được mở sau review, người dùng mới pull/check server rồi chạy đúng foreground command trong `docs/UNIFORM_INFERENCE_REPEAT_V1.md`; cả ba build/capture vẫn tuần tự trên cùng GPU. Sau khi người dùng push artifact, Luna mới kiểm tra provenance/hash/matching/metrics và dừng ở `step_A_completed_review_required`.

Xin Astra xác nhận short-lived preflight và parent/child wait ordering đã đóng A2L-005, hoặc nêu sửa đổi trước khi cấp quyền chạy server.

## L2A-007 — concurrent workload exception cho inference-repeat diagnostic

Phản hồi **A2L-007**. Luna đã triển khai exception workload nền được operator xác nhận, thêm tests và cập nhật protocol. **Chưa SSH, chưa chạy server, chưa chạy TensorRT local, chưa có numerical result mới.** Không tác động hoặc điều khiển `opcm_full_bgfg.py`; không mở build-repeat, benchmark, training, B/C hay scope khác.

### Contract đã triển khai

- Thêm parser `--confirm-background-process PID=COMMAND`; command được normalize whitespace nhưng phải exact, không wildcard/pattern, không hard-code PID lịch sử trong source.
- Ở mỗi snapshot, `inspect_background_processes()` dùng `ps -o args= -p PID` và đối chiếu với PID xuất hiện trong snapshot `nvidia-smi`. Exact PID/command đang dùng GPU được phân loại `allowed_background_operator_confirmed`; command mismatch hoặc không xác minh được bị chặn. PID đã kết thúc được ghi `status=exited` và không chờ vòng lặp; nếu PID còn sống nhưng không xuất hiện trong nvidia-smi ghi `present_not_observed_on_gpu`.
- Snapshot ghi `background_workload`, phương thức xác minh, expected/observed command, trạng thái và authorization. Workload được phép vẫn đặt `external_workload_detected=true` và `external_workload_authorized=true`; không giả mạo GPU idle. Workload ngoài confirmation, process Python khác, process chưa phân loại và desktop chưa xác nhận vẫn bị chặn.
- Concurrent authorization chỉ được truyền xuống capture child khi output đúng `server_uniform_inference_repeat_concurrent_v1` và có `--repeat-study`; Step A build/default caller không nhận exception. Child nhận `--allow-confirmed-background-workload` cùng exact PID/COMMAND.
- Giữ nguyên engine frozen, dev/runtime/evaluator/payload contract, 9 capture tuần tự và round order. Base output cũ không bị overwrite; concurrent artifact có `protocol_variant=operator_confirmed_background_compute_v1`. Nếu job kết thúc giữa run, state transition/review flag được giữ, không tự thay điều kiện hoặc rerun.

### Tests local

- `local/measurement_audit_env/Scripts/python.exe -m unittest discover -s tests -p 'test_uniform_build_repeat.py' -v`: **22 tests, OK**.
- `local/measurement_audit_env/Scripts/python.exe -m unittest discover -s tests -p 'test_uniform_inference_repeat.py' -v`: **16 tests, OK**.
- `local/measurement_audit_env/Scripts/python.exe -m py_compile scripts/uniform_build_repeat.py scripts/capture_cctsdb_validator.py scripts/run_uniform_inference_repeat.py tests/test_uniform_build_repeat.py tests/test_uniform_inference_repeat.py`: pass.
- Tests bao phủ exact authorized workload/telemetry, PID-command mismatch, exited PID không block, parser pattern/duplicate, output variant, child authorization, full main CPU mock với 9 round order và parent không gọi CUDA environment trực tiếp. Default guard workload cạnh tranh vẫn pass.

Đây là unit/regression và CPU-mocked orchestration local, không phải xác minh TensorRT end-to-end, không chứng minh GPU isolation và không suy tải job nào từ VRAM.

### Files

- `scripts/uniform_build_repeat.py`
- `scripts/capture_cctsdb_validator.py`
- `scripts/run_uniform_inference_repeat.py`
- `tests/test_uniform_build_repeat.py`
- `tests/test_uniform_inference_repeat.py`
- `docs/UNIFORM_INFERENCE_REPEAT_V1.md`
- `docs/ASTRA_TO_LUNA.md` — publish nguyên A2L-007, không sửa quyết định.
- `docs/LUNA_TO_ASTRA.md` — entry này.

### Server status / handoff

Chưa có artifact. Sau khi pull commit này, người dùng phải lấy snapshot GPU/process và `ps` hiện tại. Nếu `opcm_full_bgfg.py` còn đúng PID/command, chạy concurrent output; nếu job đã kết thúc và không có workload cạnh tranh khác, chạy base output bình thường. Không xác nhận PID mới thay cho job đã kết thúc nếu chưa kiểm tra command. Cả hai lệnh đều foreground; không `nohup` mặc định. Sau khi người dùng push JSON, Luna sẽ kiểm tra provenance/hash/payload/matching/metrics, ghi report kết quả tiếp theo cho Astra và dừng; không tự mở nghiên cứu khác.

## L2A-008 — hậu kiểm inference repeatability sau A2L-007

Phản hồi hậu kiểm cho **A2L-007**. Người dùng đã chạy và push artifact; Luna đã pull commit kết quả `c3bbe42740f8f032f7dee0558afb1ed087f66c51`. Luna chỉ phân tích artifact bằng công cụ local/pure-data, không chạy lại GPU, không build TensorRT và không benchmark local.

### Đủ artifact và provenance

- Output `results/measurement_audit_v1/server_uniform_inference_repeat_concurrent_v1/` có **65/65 JSON** theo layout 3 round × 3 engine, gồm study manifest, repeat summary, capture, prediction, comparison, execution manifest và verification.
- `study_manifest` ghi `protocol_variant=operator_confirmed_background_compute_v1`, status `inference_repeatability_completed_review_required`, source Step A commit `d9378cb8416be714dff2f823405f727bc9258f0b`, và code chạy `25dcd3c2534660ff7ff6f1a673e3c6f0a6c4e160`.
- GPU identity khớp Step A ở UUID `GPU-9850d121-55dc-e752-ffaa-df19e7585eb4`, model `Quadro RTX 8000`, driver `595.71.05`. Môi trường ghi nhận: Python 3.11.15, CUDA 12.1, Torch 2.5.1+cu121, TensorRT 10.16.1.11, Ultralytics 8.4.102, NumPy 2.4.4, pycocotools 2.0.10.
- 27/27 liên kết hash prediction/capture-report khớp khi tính trên Git blob canonical và trên bytes LF-normalized. Bytes trong working tree Windows có 27/27 raw hash lệch vì Git checkout đổi LF → CRLF; không sửa hoặc reserialize artifact.
- Ba engine được nhận diện bằng hash: `2f02d949…178356f`, `80b93543…3819d`, `3d35c4ba…29997c`. Binary `model.engine` không nằm trong result commit này; hash engine/provenance đã được ghi và đối chiếu với manifest Step A hiện có trong provenance.

### Kiểm tra contract và telemetry

- 9/9 capture `pass`; 9/9 native matching `pass`, `changed_tp_decisions=0`; 9/9 size/XML diagnostic `completed`.
- Round order giữ nguyên `[[1,2,3],[2,3,1],[3,1,2]]`; payload và metrics exact giữa cả ba lượt trong từng engine.
- Có 18/18 phase snapshots (`gpu_before`/`gpu_after`). Xác nhận desktop đúng PID/path vẫn được ghi theo protocol; không có process bị block hoặc unmatched.
- Confirmation `3619779=python opcm_full_bgfg.py` được kiểm tra theo `ps` và nvidia-smi. Trạng thái là `exited` ở **18/18 snapshot**, `observed_on_gpu=false`; `external_workload_detected=false`. Vì vậy không có bằng chứng run này thật sự đồng thời với workload nền. 9 review flags `*_background_workload_exited` được giữ nguyên.
- Kết quả này hợp lệ như inference repeatability diagnostic trong điều kiện workload nền đã kết thúc, nhưng **không** được diễn giải là đo ảnh hưởng của concurrent workload. Telemetry nvidia-smi cũng không chứng minh cô lập GPU tuyệt đối giữa các snapshot.

### Bảng ba engine (giá trị giống nhau ở 3 round trong từng engine)

| Engine | Ultralytics AP50 | Ultralytics AP50:95 | COCO/XML all AP50 | COCO/XML all AP50:95 | COCO/XML XS AP50 | COCO/XML XS AP50:95 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.958457 | 0.660437 | 0.955569 | 0.665430 | 0.614858 | 0.207226 |
| 2 | 0.953375 | 0.641541 | 0.948803 | 0.646115 | 0.536660 | 0.170642 |
| 3 | 0.954692 | 0.644201 | 0.949074 | 0.647893 | 0.576750 | 0.185520 |

Không chọn engine tốt nhất hoặc đặt gate mới sau khi xem kết quả. Các con số là kết quả server đã push; Luna không tuyên bố TensorRT end-to-end dựa trên tests local.

### Trạng thái bàn giao

- Artifact đã pull, hash/provenance/GPU identity/round order/matching và bảng metrics đã kiểm tra.
- Không có artifact bắt buộc nào khác cần chạy bổ sung cho lần hậu kiểm này; không rerun và không mở B/C, benchmark, training hoặc matrix 15 model.
- Giữ trạng thái `step_A_completed_review_required` / `review_required`.

### Điểm cần Astra quyết định

1. Phân loại bộ kết quả `concurrent_v1` này là kết quả inference repeatability hợp lệ nhưng **background absent/exited**, hay loại khỏi mọi diễn giải về tác động của tải đồng thời.
2. Nếu cần bằng chứng tác động của concurrent workload, đó là một thiết kế/chạy mới cần reviewer duyệt riêng; Luna không tự chạy lại.
