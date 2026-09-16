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

## L2A-009 — triển khai Uniform timing-cache replay, chờ review code

Phản hồi **A2L-009**. Luna đã triển khai local theo contract feasibility check đã khóa. **Chưa chạy server, chưa build TensorRT local, chưa capture GPU local và chưa mở precision/calibration intervention, B/C hoặc 15-model matrix.**

### Đã triển khai

- Thêm `scripts/run_uniform_timing_cache_replay.py` với study/path riêng `uniform_timing_cache_replay_v1` / `server_uniform_timing_cache_replay_v1`.
- Parent orchestration giữ CPU-only: chạy structured environment probe child, preflight GPU/process read-only, rồi spawn ba build child độc lập tuần tự 1→2→3. Mỗi build child dùng `uniform_build_repeat` environment/guard và builder logic tương ứng Step A.
- Khóa source đúng `server_uniform_build_repeat_v1`, source result commit `a5e79e7c15259facc24114279a050ded439cc9ed`, source code commit `d9378cb8416be714dff2f823405f727bc9258f0b`, ONNX hash `d187dc23430cbe227594f6ac28b2a5d88793adc1bcde72f4b7b4b7ddc94557e4`, frozen weights và calibration hash Step A.
- Mỗi process nhận bản copy riêng của `repeat_1/timing.cache` hash `4c765a0224845e9ddc537253878c696e56c11045369aef224cff6cc0c4178f38`; không đọc output build trước. `set_timing_cache(input, ignore_mismatch=False)` được ghi evidence; output cache được lưu/hash riêng và chỉ gắn cờ `timing_cache_output_changed`, không tự diễn giải thành tactic evidence.
- Cache-only calibrator đọc đúng `repeat_1/calibration.cache`, ghi `calibration_cache_read`, số lần read/write và bắt buộc `calibration_batches_consumed=0`; `get_batch()` ném lỗi nếu TensorRT yêu cầu recalibration. Calibration table khác input giữ partial và dừng.
- Giữ INT8, FP16/TF32 off, workspace 4 GiB, optimization level3, avg timing iterations1, detailed inspector, OBEY và danh sách Sigmoid FP32 của Step A. Không thêm editable cache, algorithm selector, `ERROR_ON_TIMING_CACHE_MISS`, Q/DQ hay precision/calibration policy mới.
- Thêm provenance dispatch `--timing-cache-study` trong `scripts/capture_cctsdb_validator.py`; capture chỉ nhận đúng study/path mới và build manifest tương ứng, không biến output thành `uniform_build_repeat_v1` hay mở arbitrary engine path.
- Evaluate thực hiện đúng 3 capture dev theo thứ tự 1→2→3, verify CPU, payload/metrics comparison giữa ba build và so với Step A repeat_1; summary có mean, sample SD (ddof1), min/max/range pp cho Ultralytics và COCO/XML all/XS/S/M/L/XL.
- Phân loại khóa trước kết quả: `replay_exact_observed`, `replay_variation_observed` hoặc `incomplete_or_invalid`. Không chọn best build và không tự tăng repeats/tuning.

### Tests và kiểm tra local

- `local/measurement_audit_env/Scripts/python.exe -m unittest discover -s tests -p 'test_uniform_timing_cache_replay.py' -v`: **8 tests, OK**.
- `local/measurement_audit_env/Scripts/python.exe -m unittest discover -s tests -v`: **80 tests, OK**.
- `python -m py_compile scripts/run_uniform_timing_cache_replay.py scripts/capture_cctsdb_validator.py tests/test_uniform_timing_cache_replay.py`: pass.
- `git diff --check`: pass, chỉ cảnh báo newline CRLF chuẩn của working tree Windows.
- Tests mới bao phủ exact output path, source ONNX/calibration/timing hash lock, changed timing-cache rejection, no Step-A capture flag/arbitrary engine dispatch, exact/variation/invalid classification, build order 1→2→3 trước evaluate và không thêm precision/tactic intervention.
- Đây là unit/mock/CPU orchestration test; không phải TensorRT end-to-end, không chứng minh build thành công hoặc GPU isolation trên server.

### Files

- `scripts/run_uniform_timing_cache_replay.py`
- `scripts/capture_cctsdb_validator.py`
- `tests/test_uniform_timing_cache_replay.py`
- `docs/UNIFORM_TIMING_CACHE_REPLAY_V1.md`
- `docs/ASTRA_TO_LUNA.md` — publish nguyên A2L-009, không sửa quyết định reviewer.
- `docs/LUNA_TO_ASTRA.md` — entry này.

### Lệnh server dự kiến, chưa cấp quyền chạy

Sau khi Astra review code, người dùng sẽ pull commit Luna hiện tại, kiểm tra GPU/environment/input/output rồi chạy foreground đúng một dòng theo docs:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master
```

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/run_uniform_timing_cache_replay.py --out-dir results/measurement_audit_v1/server_uniform_timing_cache_replay_v1 --device 0 --confirm-desktop-process DESKTOP_PID_1=DESKTOP_PATH_1 --confirm-desktop-process DESKTOP_PID_2=DESKTOP_PATH_2
```

Các placeholder desktop phải được thay bằng PID/path hiện tại sau snapshot server; không dùng PID lịch sử. Chưa có quyền server run trong entry này. Nếu code được review, sau khi người dùng push artifact Luna sẽ kiểm tra canonical blobs/cache/provenance/telemetry/capture/metrics và dừng tại review decision; không tự triển khai editable-tactic control, đổi calibration, retrain hay scale15.

## L2A-010 — sửa contract theo A2L-010, chờ review trước server

Phản hồi **A2L-010**. Luna đã triển khai và kiểm tra local; **chưa chạy server, chưa chạy TensorRT build/benchmark/capture trên local, chưa tạo artifact GPU mới**. Thiết kế A2L-009 và numerical contract không thay đổi.

### Đã sửa

- Sửa R1 bằng cách tách rõ `source_report` (metrics từ `capture_report.json`) và `source_predictions` (payload từ `validator_predictions.json`) trong `_build_comparison_row()`. Bổ sung CPU-mocked toàn bộ evaluate 3 repeats, kiểm tra cả exact payload và changed payload; helper aggregation/comparison thật vẫn được chạy.
- Sửa R2 bằng `CalibrationCacheAudit`: cho phép ít nhất một callback read, kiểm tra mọi read trả đúng input, giữ zero-batch contract, ghi từng callback write và không để write đúng sau đó che write sai. Callback conversion lỗi cũng tạo violation để hậu kiểm phát hiện. Tách `attach_timing_cache()` để `set_timing_cache(..., False)` trả false dừng ngay, không có fallback.
- Sửa R3 bằng direct hash đúng file `results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt`, ghi path và measured hash vào manifest; phase build bắt buộc study manifest đã prepare/preflight hợp lệ; build snapshot kiểm tra UUID/name/driver bằng `validate_gpu_identity()` trước builder.
- Capture dispatch và `_validate_replay_build()` dùng cùng study/build contract: settings, flags, Sigmoid constraints, source/cache hashes, engine hash, study-manifest binding, timing output và `termination_status=completed` đều được kiểm tra trước capture. Final classification đọc guard build-before/after và capture-before/after; cache coverage được ghi `unknown` khi chưa có bằng chứng API/log.
- Cập nhật `docs/UNIFORM_TIMING_CACHE_REPLAY_V1.md` theo contract mới; không đổi quyết định trong A2L-010.

### Tests/checks

- `local/measurement_audit_env/Scripts/python.exe -m unittest discover -s tests -p 'test_uniform_timing_cache_replay.py' -v`: **14 tests, OK**.
- `local/measurement_audit_env/Scripts/python.exe -m unittest discover -s tests -v`: **86 tests, OK**.
- `local/measurement_audit_env/Scripts/python.exe -m py_compile scripts/run_uniform_timing_cache_replay.py scripts/capture_cctsdb_validator.py tests/test_uniform_timing_cache_replay.py`: pass.
- `git diff --check`: pass; chỉ có cảnh báo newline CRLF→LF chuẩn của working tree Windows.
- Tests mới bao phủ direct weight mismatch, thiếu/trùng repeat classification, hai schema report/payload, full evaluate 3 repeats với payload exact/changed, read một/nhiều lần, no-read, recalibration batch, write sai bị giữ violation dù write sau đúng, timing attach false/no fallback, build phase thiếu prepared manifest và mutation các field study/build trước capture.

Các builder/calibrator tests là pure contract + fake config/CPU orchestration; không phải TensorRT end-to-end. Không dùng test local để tuyên bố build/capture GPU thành công hoặc GPU isolation.

### Files/commit handoff

- `scripts/run_uniform_timing_cache_replay.py`
- `tests/test_uniform_timing_cache_replay.py`
- `docs/UNIFORM_TIMING_CACHE_REPLAY_V1.md`
- `docs/LUNA_TO_ASTRA.md` — entry này.
- `docs/ASTRA_TO_LUNA.md` — publish nguyên A2L-010, không sửa nội dung quyết định.

Chưa có server status hoặc artifact mới để báo cáo. Sau khi Astra review commit này, người dùng mới pull và chạy đúng protocol foreground; Luna chưa cấp lại quyền chạy trong entry này và sẽ dừng ở bước review tiếp theo.

## L2A-011 — publish authorization và chuẩn bị lệnh server

Phản hồi **A2L-011**. Luna đã đọc quyết định **CODE REVIEW ACCEPTED; OPERATOR SERVER RUN AUTHORIZED** cho đúng Uniform timing-cache replay v1. Không sửa numerical code, không chạy GPU/TensorRT local và chưa có artifact server mới.

### Đã commit/push

- Publish nguyên `docs/ASTRA_TO_LUNA.md`, không sửa nội dung quyết định A2L-011.
- Cập nhật `docs/UNIFORM_TIMING_CACHE_REPLAY_V1.md` sang trạng thái server-run-authorized và thêm snapshot command không đọc `/proc`, không dùng PID lịch sử.
- Entry này trong `docs/LUNA_TO_ASTRA.md`.

### Lệnh server

Sau khi pull, người dùng chạy từng lệnh foreground một dòng:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master
```

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git status --short --branch && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git rev-parse HEAD && test "$(env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git rev-parse HEAD)" = "$(env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git rev-parse origin/master)" && if [ -e results/measurement_audit_v1/server_uniform_timing_cache_replay_v1 ]; then echo OUTPUT_EXISTS; else echo OUTPUT_ABSENT; fi
```

```bash
hostname && nvidia-smi --query-gpu=uuid,name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used --format=csv,noheader && nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader && for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk '$1 ~ /^[0-9]+$/ {print $1}'); do printf 'PID %s | ' "$p"; ps -o user=,comm=,args= -p "$p"; done && if [ -e results/measurement_audit_v1/server_uniform_timing_cache_replay_v1 ]; then echo OUTPUT_EXISTS; else echo OUTPUT_ABSENT; fi
```

Local không có live snapshot của SERVER-01 nên Luna **chưa thể điền PID/path hiện tại một cách trung thực** và không dùng lại PID 644963/644977. Người dùng gửi output snapshot; nếu có desktop GPU rows, operator đối chiếu và xác nhận đúng PID/path hiện tại. Nếu không có desktop rows cần xác nhận, bỏ các option desktop khỏi lệnh chạy. Sau khi có snapshot, Luna sẽ gửi đúng một lệnh foreground hoàn chỉnh:

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/run_uniform_timing_cache_replay.py --out-dir results/measurement_audit_v1/server_uniform_timing_cache_replay_v1 --device 0 --confirm-desktop-process CURRENT_PID=CURRENT_PATH
```

Lệnh trên là mẫu chờ thay bằng confirmation đã đối chiếu, không chạy với chuỗi placeholder. Giá trị commit kiểm tra sẽ được thay bằng commit thực tế sau khi push. Chỉ chạy khi output absent, HEAD đúng commit, environment/GPU/input hashes khớp và không có compute workload cạnh tranh. Study giữ 3 build tuần tự rồi 3 capture/CPU verify; không `nohup`, không kill/pause workload, không mở B/C hay study khác.

### Trạng thái bàn giao

- Server: chưa chạy; artifact: chưa có.
- Vấn đề cần người dùng cung cấp: snapshot GPU/process hiện tại để xác định có cần desktop confirmation và lấy đúng PID/path.
- Sau khi có kết quả server, Luna sẽ bổ sung phần kết quả vào chuỗi bàn giao này, pull/hậu kiểm và dừng để Astra review; không tự triển khai bước nghiên cứu tiếp theo.

### L2A-011 — server result addendum

Người dùng đã chạy đúng study trên **SERVER-01**, GPU Quadro RTX 8000 UUID `GPU-9850d121-55dc-e752-ffaa-df19e7585eb4`, driver `595.71.05`, rồi push artifact commit **`3797075c934ca5f88c0d64b998c38def0deef49f`**. Luna đã pull và hậu kiểm local bằng dữ liệu/Git blob; không chạy lại GPU, TensorRT build hoặc benchmark.

#### Artifact và provenance

- Đủ **41/41 file** theo inventory: study/summary, 3 compressed build logs, và đủ repeat 1–3 gồm build manifest, input caches, timing output cache, inspector, capture/predictions, comparison, execution manifest và 3 verification JSON. Không có `model.engine` trong commit, đúng policy.
- Source/result/code binding, study-manifest binding, settings, builder flags `514`, Sigmoid constraints, frozen-weight path/hash, calibration/timing input hashes, engine hash, completed termination và timing attach `ignore_mismatch=false` đều khớp contract cho cả ba repeat.
- Direct frozen weight hash được ghi là `3e5fc7a2…ba4ab8` và khớp locked hash. Canonical Git blob của source calibration cache khớp `31e9d0b3…e01a502`; timing cache khớp `4c765a02…178f38`. Source `source.onnx` và `model.engine` không có trong local/result commit theo policy server-only, nên Luna không tự re-hash binary đó local; các hash liên kết đã được kiểm tra trong manifest/capture/verification.
- Working tree Windows làm **32/41 artifact raw bytes** khác Git blob do LF→CRLF. Luna dùng canonical Git blob cho hash integrity và không reserialize/chỉnh artifact.

#### Contract, telemetry và verification

- Cả ba build: calibration cache read **2 lần**, `calibration_batches_consumed=0`, write callback `0`, không có read/write violation; timing input cùng hash, timing attach thành công, output timing hash `2283d70e…c362ce48` giống nhau giữa ba repeat nhưng khác input, và coverage được ghi `unknown`.
- Engine hashes khác nhau ở cả ba repeat: `a732cd85…9f4e6a`, `a0298de9…f6b419`, `f9279fc3…b3440c`. Inspector signature/plan signature quan sát được giống nhau giữa ba engine và source (`204` layers, Conv weights `Int8=70`, `Float=15`, plan signature `369ceedc…d03fe9`); không suy engine bytes giống hoặc full tactic lock từ đó.
- 3/3 capture `pass`, mỗi capture 1.636 ảnh/2.706 instances; 3/3 native matching `pass`, `changed_tp_decisions=0`, 3/3 size diagnostic `completed`, capture/verification prediction và report hashes liên kết exact.
- Tất cả 12 snapshot build/capture có telemetry `complete`, GPU UUID/name/driver khớp, không blocked/unmatched/external workload tại các điểm quan sát. Desktop Snap confirmations được ghi; đây vẫn là sampled telemetry trên shared server, không phải GPU isolation tuyệt đối.
- Nhiệt độ build before/after lần lượt 32→34°C, 35→35°C, 36→36°C; capture 37→38°C, 37→39°C, 38→40°C. Đây là giới hạn điều kiện, không tách thermal/timing/implementation causality.
- Ba log có `DONE BUILD`, không có `Traceback`, `[E]` hoặc `ERROR` theo marker scan; đây không phải chứng minh mọi warning đều vô hại.

#### Kết quả metrics

`repeat_summary.json` phân loại **`replay_exact_observed`**, `global_g0=review_required`; payload và metrics đều exact giữa ba replay và Step-A repeat_1. Mean/sample SD/range đều 0 trong ba repeat:

| Endpoint | AP50 | AP50–95 |
|---|---:|---:|
| Ultralytics full | 0.9584570 | 0.6604367 |
| COCO/XML all | 0.9555686 | 0.6654298 |
| COCO/XML XS | 0.6148577 | 0.2072260 |
| COCO/XML S | 0.9530828 | 0.5743576 |
| COCO/XML M | 0.9807243 | 0.6816387 |
| COCO/XML L | 0.9824451 | 0.7774076 |
| COCO/XML XL | 0.9928438 | 0.8420439 |

Summary flags giữ nguyên `repeat_1/2/3_timing_cache_output_changed`. Kết quả cho thấy trong ba lần quan sát này ordinary timing-cache replay cho payload/metrics ổn định dù serialized engine hashes khác; không chứng minh timing cache khóa tactic, không chọn engine tốt nhất và không dùng làm lý do mở calibration intervention.

#### Bàn giao và quyết định cần review

- Step A2L-011 đã có kết quả server hợp lệ ở mức descriptive diagnostic với các giới hạn trên; không có artifact bắt buộc nào thiếu và không cần rerun.
- Astra cần quyết định cách diễn giải `replay_exact_observed` cùng engine-byte variability/timing output changed/coverage unknown và thermal sampled conditions trước mọi protocol tiếp theo.
- Luna dừng tại đây: không mở B/C, editable tactics, calibration mới, retraining, matrix 15 model hoặc benchmark latency/energy.

## L2A-013 — triển khai YOLO11n precision-head ablation, chờ review code

Phản hồi **A2L-013**. Luna đã đọc quyết định tại A2L-013 và triển khai local theo đúng diagnostic protocol. **Chưa chạy server, chưa chạy TensorRT build/capture/benchmark trên local, chưa tạo artifact GPU mới và chưa cung cấp lệnh server.** Numerical contract, frozen weights, Uniform calibration và dev-only scope được giữ nguyên.

### Đã triển khai

- Thêm `scripts/run_yolo11n_precision_head_ablation.py` với study/path riêng `yolo11n_precision_head_ablation_v1` / `server_yolo11n_precision_head_ablation_v1`.
- Khóa bốn arm: `baseline_int8`, `bbox_fp32`, `classification_fp32`, `both_fp32`. Layer selection chạy trên layer names/types thực tế sau ONNX parse; bbox dùng chính xác `/model.23/cv2.`, classification dùng chính xác `/model.23/cv3.`. Mỗi layer được chọn ghi before/requested/after, đặt FP32 precision + mọi output FP32, và yêu cầu OBEY. Baseline không thêm cv2/cv3 constraint và giữ danh sách Sigmoid FP32 của Step A.
- Parent giữ CPU orchestration: preflight rồi chạy đủ 12 build child tuần tự theo arm-major/repeat `1,2,3`; chỉ sau khi toàn bộ build manifest validate mới chạy 12 capture + CPU verification theo cùng thứ tự. Output/partial output không bị overwrite hoặc tự resume.
- Mỗi build copy độc lập cùng Step-A repeat-1 calibration/timing cache; không chain cache. `set_timing_cache(..., ignore_mismatch=False)` false dừng, không fallback. Calibration audit bắt buộc đọc cache, zero batch, ghi violation nếu read/write khác bytes; timing coverage ghi `unknown`.
- Giữ Step-A settings/flags: INT8, FP16/TF32 off, workspace 4 GiB, optimization level 3, avg timing iterations 1, detailed inspector, flags `514`. Build manifest ghi source/result/code/hash, frozen-weight measured hash/path, cache hashes, engine/inspector/provenance, environment/GPU snapshots, workload/telemetry và layer evidence.
- Thêm dispatch capture bounded trong `scripts/capture_cctsdb_validator.py`: `--precision-head-ablation-study` + `--ablation-arm` chỉ nhận đúng study path, arm, repeat và không mở arbitrary engine path; các dispatch cũ không đổi.
- Thêm `docs/YOLO11N_PRECISION_HEAD_ABLATION_V1.md` ghi scope, order, cache/precision/evaluation/artifact contract, shared-server limitation và điểm dừng `step_A_completed_review_required`.
- Summary giữ mean, sample SD (ddof1), min/max/range pp của ba build mỗi arm, delta so baseline arm mean/reference, full/XS/S branch deltas, inspector evidence và invalid flags. Classification khóa trước là `diagnostic_branch_sensitive`, `no_branch_signal` hoặc `incomplete_or_invalid`; không chọn best arm và không suy diễn arithmetic precision chỉ từ tên/count.

Trong lúc test CPU/mock tôi phát hiện một lỗi thật ở phần lập chỉ mục endpoint full: Ultralytics full dùng metric group trực tiếp, không có khóa `all`. Runner đã được sửa để phân biệt Ultralytics với COCO/XML `all` trước khi aggregate/diagnostic.

### Tests/checks local

- `local/measurement_audit_env\Scripts\python.exe -m unittest discover -s tests -p 'test_yolo11n_precision_head_ablation.py' -v`: **18 tests, OK**.
- `local/measurement_audit_env\Scripts\python.exe -m unittest discover -s tests -v`: **104 tests, OK**.
- `local/measurement_audit_env\Scripts\python.exe -m py_compile scripts/run_yolo11n_precision_head_ablation.py scripts/capture_cctsdb_validator.py tests/test_yolo11n_precision_head_ablation.py`: pass.
- `git diff --check`: pass; chỉ có cảnh báo LF→CRLF chuẩn của working tree Windows.
- Test mới thực thi prefix exact/non-convolution/missing, all outputs FP32 + effective evidence + OBEY, baseline Sigmoid, arm union/repeat stability, Step-A settings/flags, common cache/no-chain/no-fallback, calibration multi-read/zero-batch/write violation, exact study dispatch, source/frozen-weight/baseline hash lock, 12 build order trước 12 capture, output overwrite/partial protection, telemetry/workload review flags và exact/variation/incomplete classification.
- Đây là CPU/mock/contract/orchestration evidence; local tests **không** chứng minh TensorRT build hoặc end-to-end GPU thành công.

### Files/commit handoff

- `scripts/run_yolo11n_precision_head_ablation.py`
- `scripts/capture_cctsdb_validator.py`
- `tests/test_yolo11n_precision_head_ablation.py`
- `docs/YOLO11N_PRECISION_HEAD_ABLATION_V1.md`
- `docs/ASTRA_TO_LUNA.md` — publish nguyên A2L-013, không sửa quyết định reviewer.
- `docs/LUNA_TO_ASTRA.md` — entry này.

### Trạng thái và điểm chờ review

- Server: **chưa chạy**; GPU/TensorRT artifact mới: **chưa có**.
- L2A-013 chỉ bàn giao code/protocol/tests để Astra review implementation. Sau khi Astra review và cấp authorization riêng, mới được tạo lệnh server; entry này không mở server run.
- Chưa mở calibration policy, retrain, 15-model matrix, official-test evaluation, cross-device hoặc benchmark. Luna chờ quyết định A2L-014.

## L2A-014 — publish authorization và chuẩn bị operator server run

Phản hồi **A2L-014**. Luna đã đọc quyết định `CODE REVIEW ACCEPTED; SERVER RUN AUTHORIZED` cho đúng `yolo11n_precision_head_ablation_v1`. Luna cập nhật `docs/YOLO11N_PRECISION_HEAD_ABLATION_V1.md` sang `server-run-authorized` và giữ nguyên numerical/provenance contract: frozen YOLO11n, Uniform calibration, bốn arm, 12 build tuần tự rồi 12 capture dev, common timing cache không fallback và không chain.

### Trạng thái bàn giao

- Đã publish nguyên nội dung A2L-014 trong `docs/ASTRA_TO_LUNA.md`.
- Đã cập nhật protocol với điều kiện authorized, server handoff, desktop/workload guard, artifact push và điểm dừng `step_A_completed_review_required`.
- Server/GPU/workload: **chưa kiểm tra trong lượt này**; Luna không SSH và chưa chạy study. Chưa có artifact GPU mới.
- Người dùng cần chạy pull/check/snapshot trên server. Chỉ sau khi snapshot xác nhận GPU/environment phù hợp, output absent và không có workload compute cạnh tranh chưa được xác nhận mới chạy lệnh study foreground.

### Lệnh operator tiếp theo

Các lệnh dưới đây được cung cấp để người dùng chạy trực tiếp trên server; mỗi lệnh là một dòng. Lệnh pull phải hoàn tất trước check. Sau khi có output snapshot, thay confirmation bằng PID/path desktop hiện tại nếu có; không dùng PID lịch sử. Nếu không có desktop GPU row cần xác nhận thì bỏ các option confirmation.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master
```

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git status --short --branch && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git rev-parse HEAD && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git rev-parse origin/master && if [ -e results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1 ]; then echo OUTPUT_EXISTS; else echo OUTPUT_ABSENT; fi
```

```bash
hostname && nvidia-smi --query-gpu=uuid,name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used --format=csv,noheader && nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader && for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk '$1 ~ /^[0-9]+$/ {print $1}'); do printf 'PID %s | ' "$p"; ps -o user=,comm=,args= -p "$p"; done
```

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/run_yolo11n_precision_head_ablation.py --phase all --out-dir results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1 --device 0
```

Không dùng `nohup`, không kill/pause process, không đổi quyền/clock/power. Nếu snapshot có desktop allowlist hiện hữu, lệnh cuối append từng `--confirm-desktop-process CURRENT_PID=CURRENT_ALLOWLISTED_PATH`; nếu có Python/build/inference hoặc process không phân loại, không chạy. Nếu study lỗi, giữ partial/log, không resume/evaluate bypass. Sau khi hoàn tất, người dùng push JSON/cache/log.gz theo protocol, không push engine/ONNX/weights; Luna sẽ pull và hậu kiểm rồi ghi addendum, không tự mở bước nghiên cứu tiếp theo.

## L2A-015 — sửa lỗi chọn layer sau partial server failure, chờ review trước rerun

Phản hồi sự cố A2L-014 trên server. Luna đã đọc log `build_bbox_fp32_1`: TensorRT parse và load timing cache thành công, nhưng runner dừng trong `select_precision_layers()` vì `/model.23/cv2.` cũng khớp node helper non-convolution `/model.23/cv2.0/cv2.0.0/act/Sigmoid`. Đây là lỗi implementation của layer-selection boundary, không phải lỗi GPU, workload, calibration hoặc timing-cache incompatibility.

### Đã sửa local

- `select_precision_layers()` nay chỉ chọn các layer convolution có tên bắt đầu đúng prefix; các node non-convolution cùng namespace không bị chọn.
- Build constraint audit nay ghi additive `prefix_candidate_names` và `excluded_prefix_non_convolution_names`, để phân biệt toàn bộ namespace thực tế với tập precision target.
- `validate_constraint_audit()` kiểm tra candidate = selected ∪ excluded, không overlap, selected vẫn là convolution đúng prefix; `validate_arm_layer_contract()` kiểm tra candidate/excluded/selected ổn định qua ba repeat.
- Protocol được bổ sung giải thích rằng helper activation dưới namespace `cv2/cv3` là candidate bị loại, không phải target FP32.
- Không đổi numerical settings, flags, frozen source, cache bytes, arm names, build/capture order hoặc evaluator.

### Tests/checks local

- Bổ sung regression test tái hiện đúng tên node server `/model.23/cv2.0/cv2.0.0/act/Sigmoid`, xác nhận node bị loại và evidence được validate.
- Targeted suite: **19/19 pass**; full regression suite: **105/105 pass**; `py_compile` cho runner và test: **pass**; `git diff --check`: **pass**.
- Không chạy TensorRT/GPU trên local.

### Server partial và điều kiện tiếp theo

- Partial `results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1/` phải giữ nguyên: baseline builds 1–3 đã hoàn tất, bbox repeat 1 thất bại, chưa có capture.
- Không resume/overwrite partial và không chạy `--phase evaluate`.
- Commit sửa khác commit đã được A2L-014 review, nên cần Astra review/re-authorize trước rerun. Vì output v1 đã tồn tại, rerun cũng cần Astra chốt output version/path mới; Luna không tự xóa, di chuyển hoặc ghi đè artifact partial.

## L2A-015 addendum — triển khai attempt2

Phản hồi yêu cầu triển khai A2L-015. Luna giữ nguyên scientific study ID `yolo11n_precision_head_ablation_v1` và chuyển output của execution attempt 2 sang:

`results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1_attempt2/`

### Thay đổi đã triển khai

- Runner `run_yolo11n_precision_head_ablation.py` chỉ chấp nhận destination attempt2; output v1, path khác và output attempt2 đã tồn tại đều bị từ chối trước khi ghi.
- Capture dispatch `validate_precision_head_ablation_scope()` và `ablation_capture_inputs()` cùng khóa đúng destination attempt2; destination v1/ngoài scope bị từ chối.
- `study_manifest.json` của attempt2 ghi `execution_attempt: 2`, relative `previous_attempt_path`, `execution_reason: implementation_fix_non_convolution_namespace_selection`, `execution_git_commit` hiện hành và `runner_script_sha256`. Source Step-A code commit vẫn được ghi riêng trong `source_study_code_commit`.
- Protocol active và foreground command đã chuyển sang attempt2. Partial v1 được ghi rõ là read-only, không xóa/di chuyển/overwrite/resume/copy.
- CPU/mock evaluate test đã chuyển sang destination attempt2 và vẫn yêu cầu đủ 12 build/capture theo thứ tự khóa. Không đổi flags, cache, weights, evaluator, numerical settings hoặc GPU policy.

### Kiểm tra và provenance

- Targeted ablation: **19/19 pass**; full regression: **105/105 pass**; `py_compile` runner/capture/tests: **pass**; `git diff --check`: **pass**.
- Đây là CPU/mock evidence; không chạy TensorRT build/benchmark/capture trên local và chưa tuyên bố native TensorRT end-to-end.
- Implementation commit: `a5abd14f444449c691395a02c5a8e2cd4513e43f`. Push dùng tài khoản GitHub `NADUNGVN` có quyền ghi; addendum này được cập nhật bằng commit tài liệu kế tiếp.

### Trạng thái server và bàn giao

- Chưa chạy server và chưa tạo artifact attempt2. Partial v1 vẫn giữ nguyên.
- Sau khi push, người dùng pull đúng commit, kiểm tra `attempt2` absent và snapshot GPU/process/environment; sau đó mới chạy foreground command trong protocol, bổ sung desktop confirmations theo snapshot hiện tại nếu cần.
- Nếu attempt2 lỗi, giữ partial/log và báo cụ thể; không resume/evaluate bypass. Sau khi hoàn tất, chỉ push JSON/cache/log.gz theo artifact contract để Luna hậu kiểm đủ 12 build/capture rồi dừng ở `step_A_completed_review_required`.

## L2A-016 — hậu kiểm attempt2, bàn giao Astra review

Luna đã pull artifact server từ commit `839acdcb6a09569d1e6e130aa523c38d960dabd5` bằng Git, không SSH và không chạy lại GPU. Study execution commit trong `study_manifest.json` là `97683e649b8afca23f37dfe56a2412479c19c627`; runner hash là `577e053300172947134bc3d85eb74d323da0460dba5798b38768fb717e4cb3fe`.

### Contract/provenance đã kiểm tra

- Đủ **159 artifact**: 111 JSON, 36 cache, 12 `.log.gz`; đủ 12 repeat directories, 12 build manifests, 12 capture reports, 12 prediction payloads và 12 verification summaries. Không có `model.engine` được track, đúng server-only binary policy.
- `study_manifest`: `execution_attempt=2`, previous path đúng `results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1`, reason đúng `implementation_fix_non_convolution_namespace_selection`; source Step-A code/result commit và frozen weights/ONNX/calibration/timing hashes khớp locked contract.
- Cả 12 build: `termination_status=completed`, `cache_chain=false`, calibration batches `0`, calibration read/write violations rỗng, timing attach `called=true`, `ignore_mismatch=false`, không fallback. Input timing hash đều là `4c765a02…178f38`.
- Constraint audit hợp lệ cho cả 4 arm và ổn định qua 3 repeat: baseline `0`, bbox `9`, classification `15`, both `24` convolution targets; candidate/excluded namespace evidence được validate.
- Cả 12 capture `pass`, 12/12 native matching `pass`, 12/12 size diagnostic `completed`; các liên kết prediction/capture/build provenance/verification khớp theo canonical Git blobs.
- Có 48 điểm quan sát telemetry build/capture before/after; tất cả `telemetry_status=complete`, không `external_workload_detected`, blocked process hoặc unmatched confirmation. Đây vẫn là sampled telemetry trên shared lab server, không phải GPU isolation tuyệt đối.
- Có khác biệt line ending giữa một số working-tree bytes Windows và canonical Git blobs; hậu kiểm dùng canonical blob để kiểm tra hash server, không reserialize artifact. `source.onnx` và 12 engine binaries không có local theo policy nên không direct re-hash binary; chỉ kiểm tra hash liên kết trong manifest/report.

### Kết quả descriptive từ summary đã push

| Arm | Targets | Full AP50 | Full AP50:95 | COCO/XML all AP50 | XS AP50 | S AP50 | Classification |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline_int8 | 0 | 95.846% | 66.044% | 95.557% | 61.486% | 95.308% | replay_exact_observed |
| bbox_fp32 | 9 | 96.550% | 71.837% | 96.164% | 68.195% | 95.835% | replay_exact_observed |
| classification_fp32 | 15 | 97.003% | 70.672% | 96.543% | 64.322% | 97.073% | replay_exact_observed |
| both_fp32 | 24 | 97.599% | 75.707% | 97.127% | 70.484% | 97.538% | replay_exact_observed |

Arm-minus-baseline-arm-mean deltas from the persisted summary are: bbox full AP50 `+0.704 pp`, full AP50:95 `+5.793 pp`, XS AP50 `+6.709 pp`; classification `+1.157 pp`, `+4.628 pp`, `+2.836 pp`; both `+1.753 pp`, `+9.663 pp`, `+8.998 pp`. These are descriptive values only; Luna does not select a best arm or infer causality.

### Flags and stopping point

- Overall persisted classification: `replay_exact_observed`.
- Persisted diagnostic label: `diagnostic_branch_sensitive`.
- Review flags: `baseline_int8_timing_cache_output_changed`, `bbox_fp32_timing_cache_output_changed`, `classification_fp32_timing_cache_output_changed`, `both_fp32_timing_cache_output_changed`. All 12 timing-cache outputs differ from the common input according to build manifests; coverage remains `unknown`.
- State: **`step_A_completed_review_required`**. Astra cần quyết định cách diễn giải precision-head diagnostic cùng timing-cache output variability, implementation/tactic ambiguity và điều kiện telemetry shared server. Luna không chạy B/C, calibration matrix, 15-model matrix, retraining, official test hoặc benchmark tiếp theo.

## L2A-017 — triển khai paired dev analysis, chờ CPU server replay

Phản hồi **A2L-016**. Luna đã giữ nguyên entry giao việc trong `docs/ASTRA_TO_LUNA.md` và triển khai runner CPU riêng cùng protocol cho paired uncertainty analysis trên artifact attempt2 đã được Astra nghiệm thu.

### Đã hoàn thành

- Thêm `scripts/analyze_precision_head_paired.py`, chỉ đọc canonical Git blobs của capture/verification JSON; không load engine, không gọi TensorRT, không inference, không benchmark và không sửa input artifact.
- Khóa đúng bốn arm `baseline_int8`, `bbox_fp32`, `classification_fp32`, `both_fp32`, FP16 historical reference, ba repeat exact trước khi deduplicate về repeat_1, 1.636 dev images, seed `20260916`, PCG64 và 1.000 paired draws.
- Giữ duplicate image occurrences khi COCO accumulate; dùng chung draw cho năm representation; phát hiện missing/changed/unlinked input, sai XML hash, sai environment, sai evaluator convention và output directory đã tồn tại.
- Tạo đủ 10 contrast cố định, sáu size endpoints × AP50/AP50-95, point reproduction độc lập, full Ultralytics point riêng và localization gained/lost theo baseline ở IoU 0.50/0.75/0.90.
- Thêm `docs/PRECISION_HEAD_PAIRED_ANALYSIS_V1.md` với contract/output/foreground command server.
- Thêm 5 tests cho duplicate draw, identical-arm delta=0, đủ 10 contrast và missing/changed canonical input.

### Kết quả kiểm tra local

- Targeted paired-analysis tests: **5/5 pass**.
- Full regression suite: **111/111 pass**.
- `py_compile` runner/tests: **pass**; `git diff --check`: **pass**.
- Phân tích số liệu **chưa chạy local**: local không có `xml.zip` CCTSDB ở path locked và local `numpy=2.4.2` không khớp capture contract `numpy=2.4.4`. Runner fail-closed ở các điều kiện này, không tạo CIs giả.
- Không có thư mục output mới `results/measurement_audit_v1/precision_head_paired_analysis_v1/` và chưa có artifact analysis để báo cáo khoa học.

### Commit và lệnh server

- Commit chứa code, tests, protocol, L2A-017 và nguyên văn entry A2L-016 sẽ được push lên branch `master` bằng tài khoản GitHub `NADUNGVN`; hash commit được báo ngay sau push.
- Sau khi pull commit đó, người dùng chạy một lệnh foreground CPU trong protocol:

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/analyze_precision_head_paired.py --out-dir results/measurement_audit_v1/precision_head_paired_analysis_v1 --xml /home/ubuntu/Dung_TDTU/nighttime-tsd/data/raw/CCTSDB2021/xml.zip
```

Lệnh không mở CUDA/TensorRT và không cần GPU idle guard, nhưng phải dùng đúng XML có SHA256 `35c1f3b7cdfde8e5ddded9c186e16335b2f24364ebd00c2be95bdcfca4051329`. Nếu lỗi, giữ nguyên thông báo và partial output, không resume/overwrite. Sau khi người dùng push output, Luna sẽ pull canonical blobs, kiểm tra đủ manifest/sample/draws/point/CI/localization và cập nhật bảng thực tế.

### Trạng thái bàn giao

`code_ready_server_cpu_required`; chưa có contrast/CI numerical để Astra nghiệm thu. Các giới hạn đã khóa: exploratory dev-only, conditional on frozen captures/one server environment, không tách build/tactic hoặc calibration variability, không causal bbox/classification decomposition, không official test và không mở study tiếp theo. Chưa chọn arm hay đặt success threshold hậu nghiệm.

### L2A-017 addendum — CPU replay đã hoàn tất và hậu kiểm local

Người dùng đã chạy đúng runner CPU foreground và push artifact ở commit `4846c73ddd2cbb2bd522caa0e1a1eb4598deb1e3`. Luna đã pull về local; input artifact commit được runner ghi là `839acdcb6a09569d1e6e130aa523c38d960dabd5`, code commit là `8b5a3420f1eeb460b58198647cbdbe83f1ab1045`.

#### Kiểm tra artifact/provenance

- Đủ **9/9 file**: input manifest, sample plan, bootstrap draws, point estimates, contrast CI, localization per-GT/summary, report Markdown và analysis summary.
- Tất cả output hash trong `analysis_summary.json` khớp canonical Git blobs; 13 nhóm input (FP16 một repeat và bốn arm × ba repeat) khớp hash manifest. Ba repeat mỗi arm có prediction payload exact; repeat_1 được dùng một lần, không chọn best build.
- Sample plan tái lập exact với PCG64 seed `20260916`, 1.000 × 1.636, cùng draw cho năm representation, duplicate retained. Tính lại `contrast_ci` khớp artifact; tất cả endpoint có `valid_resamples=1000`, `undefined_resamples=0`.
- COCO/XML point reproduction khớp persisted size reports với sai số tối đa `0.0`; không dùng CIs COCO/XML cho full Ultralytics points.

#### COCO/XML paired contrasts (percentage points)

Mỗi ô là `point [95% percentile CI]`; bảng đầy đủ sáu size × hai metric nằm trong `contrast_ci.json`. Đây là các endpoint ưu tiên theo A2L-016: all AP50-95, XS AP50/AP50-95, S AP50/AP50-95.

| Contrast | All AP50-95 | XS AP50 | XS AP50-95 | S AP50 | S AP50-95 |
|---|---:|---:|---:|---:|---:|
| bbox − baseline | +5.4263 [4.9118, 5.8991] | +6.7094 [2.1525, 11.2544] | +7.3861 [4.9635, 10.1734] | +0.5272 [-0.0354, 1.1975] | +6.7072 [5.6116, 7.7726] |
| classification − baseline | +3.6271 [2.9389, 4.1490] | +2.8362 [-0.3497, 5.3691] | +1.3778 [0.1425, 2.5680] | +1.7645 [0.7447, 2.8701] | +1.7893 [0.7073, 2.8024] |
| both − baseline | +8.5067 [7.7104, 8.9999] | +8.9983 [3.1418, 15.1011] | +8.9474 [5.7214, 12.6664] | +2.2296 [1.0662, 3.2093] | +8.7087 [7.3844, 9.9855] |
| bbox − classification | +1.7993 [1.2384, 2.5085] | +3.8732 [-0.7783, 9.3488] | +6.0083 [3.4869, 8.9770] | -1.2373 [-2.2879, -0.3152] | +4.9178 [3.7078, 6.2846] |
| both − bbox | +3.0804 [2.4585, 3.4520] | +2.2889 [-1.5951, 5.4316] | +1.5612 [-0.0082, 3.2275] | +1.7024 [0.7675, 2.5102] | +2.0016 [1.0364, 2.9070] |
| both − classification | +4.8797 [4.4281, 5.2982] | +6.1621 [1.8981, 11.7489] | +7.5696 [4.9545, 10.9084] | +0.4651 [-0.0850, 0.8020] | +6.9194 [5.9152, 8.0245] |
| baseline − FP16 | -9.1116 [-9.6821, -8.3370] | -10.1958 [-16.0900, -4.3431] | -10.1978 [-13.1254, -7.2703] | -2.1238 [-3.1518, -0.9625] | -9.5003 [-10.7273, -8.0705] |
| bbox − FP16 | -3.6853 [-4.1638, -3.0439] | -3.4864 [-6.8747, 0.6081] | -2.8117 [-4.6041, -0.7969] | -1.5967 [-2.4735, -0.6599] | -2.7931 [-3.6966, -1.8071] |
| classification − FP16 | -5.4846 [-5.9896, -5.0174] | -7.3595 [-12.8061, -2.8792] | -8.8200 [-11.7381, -6.3637] | -0.3593 [-0.8469, 0.2664] | -7.7109 [-8.9126, -6.6272] |
| both − FP16 | -0.6049 [-0.9361, -0.3670] | -1.1975 [-2.9487, 0.6160] | -1.2504 [-2.7020, 0.4311] | +0.1058 [-0.2458, 0.4121] | -0.7915 [-1.4773, -0.1414] |

Full Ultralytics points remain separate in `point_estimates.json` and reproduce the persisted captures: FP16 `97.7750/76.3025`, baseline `95.8457/66.0437`, bbox `96.5501/71.8370`, classification `97.0031/70.6718`, both `97.5986/75.7066` for AP50/AP50-95 (%). These must not be read as the COCO/XML all endpoints above.

Localization is available in `localization_summary.json`; for example at IoU50/all, bbox has 10 gained/8 lost GT matches, classification 0 gained/11 lost, and both 11 gained/18 lost versus baseline. At IoU75/0.90 there are further gained/lost transitions. This remains a conditional GT-centric matching diagnostic, not a causal bbox/classification decomposition.

#### Kết luận và điểm dừng

Kết quả đã hoàn tất đúng estimator và artifact contract, nhưng trạng thái vẫn là **`step_A_completed_review_required`**. Các CIs là exploratory, conditional on frozen captures, một dev sample và một server environment; chưa tách build/tactic variability, calibration variability hoặc thiết bị. Không chọn arm, không đặt success threshold hậu nghiệm, không mở B/C, calibration matrix, 15-model matrix, retraining, official test hoặc benchmark. Astra cần review diễn giải bảng paired contrasts cùng các giới hạn này.

## L2A-018 — triển khai latency runner/protocol, chờ Astra review implementation

Phản hồi **A2L-017**. Luna đã đọc quyết định nghiệm thu paired analysis và triển khai phần implementation local cho study `yolo11n_precision_head_latency_v1`. Theo giới hạn của A2L-017, chưa chạy server/GPU, chưa build/rebuild engine và chưa cung cấp lệnh operator chạy study.

### Đã triển khai

- Thêm `scripts/run_precision_head_latency.py` với parent/child lifecycle: parent giữ host GPU phase lock, mỗi session là một child mới, chờ child hoàn tất rồi mới chuyển session kế tiếp; không chạy đồng thời nhiều engine.
- Parent kiểm tra trước khi child load engine: accepted attempt2 provenance, FP16 reference, 13 direct engine paths/bytes/SHA256, 13 hash/path duy nhất, runtime capture contract, frozen environment/GPU identity, 256-image pool và paired accuracy links. Child kiểm tra lại engine hash trước `YOLO(...)` deserialization.
- Khóa synchronous batch-1 `model.predict` với `imgsz=640`, `conf=0.001`, `iou=0.7`, `max_det=300`, `rect=False`, `task=detect`, `verbose=False`; loại disk decode/model load/initial allocation/warmup khỏi timer; sync trước timer và sau predict; dùng `perf_counter_ns`.
- Khóa 256 ảnh từ sorted 1,636 dev IDs bằng PCG64 seed `20260916`, 200 warmup và 1,000 raw measured calls/session; lưu hash file, decoded shape, cyclic sequence/hash và raw latency samples.
- Khóa 39 session theo 3 round rotation left 4/8; báo per-engine/per-round, pooled 3,000 calls/build và arm summaries gồm mọi build/round; không chọn fastest run, không xem call samples là build replicates.
- Ghi telemetry GPU/process trước/sau ở parent và child; giữ desktop exception chỉ qua PID/path hiện tại; workload CUDA cạnh tranh hoặc identity/telemetry không hợp lệ làm run fail-closed. Không kill/pause process, đổi quyền, clock hay power.
- Liên kết accuracy points và 10 fixed COCO/XML contrast CIs đúng evaluator; không gắn COCO CI lên full Ultralytics points. Output mới được bảo vệ không overwrite, giữ partial/log khi lỗi.
- Thêm `docs/YOLO11N_PRECISION_HEAD_LATENCY_V1.md` mô tả contract/output và nêu rõ chưa có server command trước Astra review.

### Tests/checks local

- Thêm 10 CPU/mock tests cho schedule 39 session, pool deterministic/no-replacement/shared sequence, image hashes, warmup/timer/synchronization boundary, percentile/raw aggregation, runtime options, missing/changed engine bytes, child lifecycle command, engine-session aggregation và no-overwrite.
- Targeted latency suite: **11/11 pass**; full regression suite: **122/122 pass**; `py_compile` runner/tests và `git diff --check`: **pass**. Đây chỉ là CPU/mock evidence; không chạy TensorRT/GPU local.

### Trạng thái bàn giao

`implementation_review_required`. Luna chưa chạy server/GPU, chưa tạo `server_yolo11n_precision_head_latency_v1/` và chưa chọn engine/arm. Astra cần review runner parent/child, direct engine hash-before-load, timestamp/synchronization boundary, schedule/aggregation, accuracy-link contract và telemetry guard trước khi cấp server-run authorization.

## L2A-019 — hoàn tất sửa R1/R2/R3 và integration mock 39 sessions

Phản hồi **A2L-018**. Luna đã triển khai các sửa đổi local trên runner latency; giữ nguyên 13 engine, 3 round, 39 session, image pool, runtime options, sample plan và mọi kết quả nghiên cứu đã nghiệm thu. Chưa chạy GPU/server, không rebuild engine và không chạy lại ablation, paired analysis hay nghiên cứu nào khác.

### R1 — path và ảnh được khóa đúng

- Default `--images-dir` nay là `data/processed/cctsdb2021_clean/dev/images`, resolve từ repository root thay vì shell cwd. Relocation explicit được ghi cả declared/resolved path và canonical `dev_absolute.yaml` reference.
- Parent đọc 1.636 image IDs và `orig_shape` từ FP16 capture blob đã pin; pool preparation hash trực tiếp 256 bytes được chọn và decode kiểm đúng `(H,W)` với `orig_shape`. Child kiểm lại hash/shape bytes trước model work. Không diễn đạt các hash mới là content-identical với historical capture nếu historical capture không lưu image-byte hashes.

### R2 — canonical provenance/binding

- Attempt2 metadata và consumed JSON được đọc từ canonical Git commit `839acdcb6a09569d1e6e130aa523c38d960dabd5`; paired-analysis summary/points/CI được đọc từ canonical commit `4846c73ddd2cbb2bd522caa0e1a1eb4598deb1e3`. Manifest/spec ghi path, commit và blob SHA256.
- Kiểm prediction payload ↔ capture report ↔ verification, source weight/model/build bindings, 13 engine hashes, nested/flat calibration-timing input hashes và exact model/ten-contrast mapping. Engine server vẫn được hash trực tiếp trước mỗi `YOLO(...)`; không yêu cầu binary local.
- Checkout JSON bị thay đổi không trở thành expected baseline vì validator dùng pinned blobs; các test provenance kiểm thay đổi source binding, cache hash và engine bytes phải fail.

### R3 — session thực và lifecycle

- Child kiểm scheduled output path, parent pool path/hash, image relocation binding, engine path/hash/bytes và round/engine schedule trước GPU snapshot. Parent chờ từng child tuần tự; child environment probe CUDA chỉ chạy trong child, không chạy ở parent.
- Session bắt buộc dataset/runtime/pool binding, preprocessing probe ngoài timer cho từng source-aspect group với shape `(1,3,640,640)`, đúng 1.000 raw positive finite samples, `n_calls=1000`, independently persisted before/after GPU identity và process-guard evidence.
- Parent chỉ tạo `latency_summary.json`/`report.md` sau 39 cặp `(round, engine)` duy nhất, mỗi engine đủ round 1–3; pooled call count là 3.000/build, FP16 arm 3.000 và mỗi INT8 arm 9.000. Child fail/missing/malformed hoặc telemetry/workload mismatch giữ partial output và không tạo completed summary.

### Tests/checks local

- Targeted latency suite: **18/18 pass**; trong đó parent CPU/mock orchestration chạy đủ **39/39 sessions** đến `latency_summary.json` và `report.md`, kiểm thứ tự start/finish tuần tự; child mock kiểm runtime options và shape observation; negative tests kiểm short raw vector, missing telemetry, missing/duplicate session, provenance/cache mutation và failed-child partial output.
- Full regression suite: **129/129 pass**.
- `py_compile` runner/tests: **pass**; `git diff --check`: **pass**.
- Đây chỉ là CPU/mock evidence; không phải kiểm chứng TensorRT end-to-end và chưa có server artifact latency.

### Trạng thái bàn giao

`implementation_review_required`. Astra cần review corrected R1/R2/R3 và quyết định có cấp server-run authorization hay không. Luna chưa cung cấp lệnh server và không tự mở bước nghiên cứu tiếp theo.

## L2A-020 — sửa F1/F2 theo A2L-019, chờ review implementation

Phản hồi **A2L-019**. Luna đã sửa đúng hai lỗi schema tương thích trong runner latency; giữ nguyên thiết kế latency, không chạy server/GPU, không rebuild engine và không chạy lại bất kỳ nghiên cứu đã hoàn tất nào.

### F1 — canonical YAML metadata

- Thay check substring `"CCTSDB2021"`/`"val: images"` bằng `yaml.safe_load` trên exact Git blob của `results/measurement_audit_v1/server_fp16_capture_v1/dev_absolute.yaml` tại pinned commit `839acdcb6a09569d1e6e130aa523c38d960dabd5`.
- Validator kiểm tra absolute POSIX path đã normalize và có suffix chính xác `data/processed/cctsdb2021_clean/dev`, `train: images`, `val: images`, `names` `{0: prohibitory, 1: mandatory, 2: warning}` và `nc: 3`. Vì dùng `PurePosixPath`/POSIX semantics, historical Linux path được kiểm đúng trên Windows local; không sửa canonical YAML artifact và không dùng uppercase substring làm dataset identity.
- Default repo-anchored image path, explicit relocation metadata, canonical blob/hash binding và direct filesystem image hash/shape checks vẫn giữ nguyên. `validate_inputs` trả thêm parsed canonical data contract nhưng không đổi input/data scope.

### F2 — producer/consumer telemetry schema

- Consumer nay đọc đúng `process_guard.external_workload_detected` do `uniform_build_repeat.snapshot` ghi; không rename producer field và không mặc định missing field thành clean.
- Consumer kiểm schema/version producer thật (`GPU_PROCESS_GUARD_VERSION`), complete `device`/`processes` telemetry, required guard fields, boolean workload/authorization states, blocked/unmatched lists và process details. Missing/unknown/`True`, blocked/unmatched hoặc authorized background workload ngoài latency contract đều fail-closed.
- Parent summary aggregation cũng dùng đúng `external_workload_detected`; clean desktop exception vẫn đi qua schema producer, còn latency study không mở background-compute exception.

### Kiểm tra trực tiếp và tests

- Canonical metadata test đọc trực tiếp pinned Git blob, parse đúng YAML canonical và reject wrong `val`/wrong dev path.
- Producer-consumer contract test gọi trực tiếp `uniform_build_repeat.snapshot` với `nvidia-smi` I/O mock, đưa nguyên payload vào `_validate_persisted_gpu_evidence`; negative cases missing/true/blocked/limited đều reject. Không có CUDA/GPU call.
- Parent CPU/mock integration vẫn chạy đủ **39/39 session** tuần tự đến summary/report; fixtures telemetry lấy từ output schema của producer thay vì tự tạo key sai.
- Targeted latency suite: **21/21 pass**.
- Full regression suite: **132/132 pass** trên Python 3.10 local với `pycocotools==2.0.10`.
- `py_compile` runner/tests: **pass**; `git diff --check`: **pass**. Không chạy TensorRT build/benchmark và không tuyên bố TensorRT end-to-end.

### Files và trạng thái

- Sửa `scripts/run_precision_head_latency.py`, thêm canonical YAML parser và producer-schema validation.
- Bổ sung `tests/test_precision_head_latency.py` với canonical metadata và producer-consumer contract/negative tests.
- Cập nhật `docs/YOLO11N_PRECISION_HEAD_LATENCY_V1.md` mô tả hai compatibility corrections.
- Giữ nguyên entry A2L-019 trong `docs/ASTRA_TO_LUNA.md` và push cùng commit này.

Trạng thái: `implementation_review_required`. Chưa cung cấp lệnh server; Astra cần review F1/F2 và quyết định authorization trước mọi server/GPU run.

## L2A-021 — handoff operator latency run authorized, snapshot pending

Phản hồi **A2L-020**. Luna giữ nguyên runner và numerical contract của commit `2d8f5af84efda4be273cc7f9eca9becbdf202bf1`; đã cập nhật protocol sang `implementation_accepted_operator_run_authorized` và push nguyên entry A2L-020 cùng tài liệu bàn giao. Không ghi study completed, chưa có server artifact và chưa chạy GPU từ local.

### Bàn giao operator

- Server target theo authorization: `SERVER-01`, GPU identity locked: `GPU-9850d121-55dc-e752-ffaa-df19e7585eb4`, `Quadro RTX 8000`, driver `595.71.05`.
- Luna không SSH/server-run theo handoff protocol, nên snapshot **hiện tại** và trạng thái output chưa được Luna tự tuyên bố. Operator cần chạy bốn lệnh read-only trong A2L-020 và gửi nguyên output; không dùng lại desktop PID/path lịch sử.
- Sau khi snapshot cho thấy đúng GPU, không có workload compute cạnh tranh/unknown, desktop rows hiện tại được operator xác nhận và output `OUTPUT_ABSENT`, Luna sẽ cung cấp một lệnh foreground hoàn chỉnh với từng `--confirm-desktop-process PID=PATH` hiện tại. Không thêm background-compute exception và không dùng `nohup` mặc định.
- Lệnh foreground chỉ dùng 13 engine đã tồn tại, không rebuild/export; nếu thiếu hoặc hash binary sai thì dừng và báo exact path. Partial output sau lỗi phải được giữ nguyên.

### Hậu kiểm sau operator run

Operator cần push đúng study artifacts dưới `results/measurement_audit_v1/server_yolo11n_precision_head_latency_v1/`, không commit engine/model/image và không dùng `git add .`. Luna sẽ pull canonical artifacts, kiểm 4 root files + 39 session directories, 39 unique sessions, 1.000 raw samples/session, 39.000 timed calls, engine/pool/hash links, shape observations, telemetry và round/build hierarchy; sau đó ghi addendum L2A-021 hoặc entry tiếp theo và dừng cho Astra review.

Trạng thái: `handoff_ready_server_snapshot_required`. Chưa có latency result, chưa chọn arm/engine và chưa mở study tiếp theo.

### Addendum L2A-021 — hậu kiểm latency artifact hoàn tất

Operator đã chạy đúng study trên `SERVER-01` và push artifact trong commit `c6e61dea6127525933edd48af783f497445ed2ff`. Luna đã pull commit này về local; không chạy GPU/server và không rebuild engine.

#### Kết quả integrity/provenance

- Hậu kiểm: **PASS**.
- Layout đúng: `160/160` files gồm `81` JSON, `78` child logs và `1` report Markdown; 4 root files và 3 round × 13 engine directories đúng contract.
- Có `39/39` session duy nhất `(round, engine)`, mỗi session `1.000` raw latency samples; tổng `39.000` timed calls. Mỗi session có 200 warmup calls, shape probe loại khỏi timing, và return code `0`.
- Recompute mean/median/p95/p99/min/max/FPS từ raw vectors khớp summary với `numpy.quantile(method='linear')`; schedule, round/build hierarchy và arm pooling khớp.
- Pool `CCTSDB2021/dev`: source `1.636`, fixed pool `256`, seed `20260916`, warmup `200`, measured `1.000`, cùng sequence và `disk_decode_in_timing=false`.
- Engine inventory đủ 13 engine; engine SHA/bytes và accepted canonical blob refs khớp Git. Canonical attempt-2 commit là `839acdcb6a09569d1e6e130aa523c38d960dabd5`; runner code commit trong manifest là `a5f5523a78c8f9de6a3d5cc96a1ea2376bab7d21`. Luna không tuyên bố đã rehash binary engine trên local vì binary không thuộc artifact checkout.
- Accuracy links vẫn trỏ đúng paired-analysis canonical inputs; không thực hiện accuracy analysis mới.

GPU/environment được ghi trong artifact: `GPU-9850d121-55dc-e752-ffaa-df19e7585eb4`, `Quadro RTX 8000`, driver `595.71.05`, CUDA `12.1`, Torch `2.5.1+cu121`, TensorRT `10.16.1.11`, Ultralytics `8.4.102`, NumPy `2.4.4`, Python `3.11.15` và `pycocotools 2.0.10`.

#### Telemetry/process guard

- Có `78` child before/after guard snapshots (39 × 2) và `2` parent snapshots; GPU UUID/name/driver đều match locked identity.
- Mọi snapshot có telemetry status `complete`, `external_workload_detected=false`, `external_workload_authorized=false`, không blocked/unmatched/background workload.
- Desktop exception được ghi đúng bằng xác nhận hiện tại của operator: PID `644963` và `644977`, cả hai path `/snap/snapd-desktop-integration/391/usr/bin/snapd-desktop-integration`. Verification method ghi rõ đối chiếu PID/path từ `nvidia-smi` và **không đọc được `/proc/<pid>/exe`**; không ghi thành đã xác minh qua `/proc`.
- Đây vẫn là shared lab server: desktop processes được cho phép tồn tại; `nvidia-smi` snapshots không chứng minh zero interference giữa các snapshot. Không có workload compute cạnh tranh nào được guard phát hiện trong artifact.
- Tất cả child stdout có `DONE SESSION`, stderr rỗng. Một warning Ultralytics về tự đoán task xuất hiện 39 lần trong stdout; không tạo violation và không làm thay đổi numerical contract, nhưng được giữ lại để reviewer biết.

#### Latency pooled theo arm

| Arm | Builds | Sessions | Calls | Mean ms | Median ms | P95 ms | P99 ms | Serial FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 1 | 3 | 3,000 | 3.591666 | 3.441813 | 4.292193 | 4.809802 | 278.422357 |
| baseline INT8 | 3 | 9 | 9,000 | 3.532581 | 3.456276 | 3.871791 | 4.246000 | 283.079163 |
| bbox FP32 | 3 | 9 | 9,000 | 3.628687 | 3.554793 | 3.992141 | 4.503302 | 275.581761 |
| classification FP32 | 3 | 9 | 9,000 | 3.591530 | 3.523714 | 3.929726 | 4.211887 | 278.432873 |
| both FP32 | 3 | 9 | 9,000 | 3.655555 | 3.588497 | 3.981748 | 4.243174 | 273.556248 |

Arm pooled rows include all listed builds and rounds; they are not independent build replicates and Luna không chọn arm/build nhanh nhất.

#### Latency pooled theo build

| Engine | Mean ms | Median ms | P95 ms | P99 ms | Serial FPS |
|---|---:|---:|---:|---:|---:|
| fp16 | 3.591666 | 3.441813 | 4.292193 | 4.809802 | 278.422357 |
| baseline_int8_1 | 3.598104 | 3.505625 | 3.962573 | 5.678998 | 277.924177 |
| baseline_int8_2 | 3.485308 | 3.433825 | 3.804647 | 3.900226 | 286.918702 |
| baseline_int8_3 | 3.514331 | 3.459008 | 3.857122 | 3.962243 | 284.549200 |
| bbox_fp32_1 | 3.699302 | 3.592532 | 4.319698 | 4.661255 | 270.321256 |
| bbox_fp32_2 | 3.582681 | 3.520282 | 3.923370 | 4.013859 | 279.120615 |
| bbox_fp32_3 | 3.604079 | 3.552563 | 3.923320 | 4.208183 | 277.463412 |
| classification_fp32_1 | 3.659418 | 3.582078 | 4.049955 | 4.534376 | 273.267461 |
| classification_fp32_2 | 3.547071 | 3.498234 | 3.878610 | 3.978568 | 281.922734 |
| classification_fp32_3 | 3.568100 | 3.518809 | 3.880609 | 4.038037 | 280.261191 |
| both_fp32_1 | 3.616000 | 3.552680 | 3.939100 | 4.352448 | 276.548705 |
| both_fp32_2 | 3.681632 | 3.625221 | 4.000266 | 4.263316 | 271.618689 |
| both_fp32_3 | 3.669035 | 3.595803 | 4.002861 | 4.162267 | 272.551257 |

Timing scope là synchronous batch-1 `model.predict` wall time từ decoded CPU image qua preprocessing/H2D/inference/postprocessing/NMS; đây không phải pure TensorRT kernel time. Kết quả không phải deployment choice, cross-device claim hay causal claim về precision.

Trạng thái bàn giao: `latency_completed_review_required` / `step_A_completed_review_required`. Astra cần review accuracy–latency trade-off, mức biến thiên giữa round/build và giới hạn telemetry của shared lab server. Luna không mở B/C, không mở rộng 15 model và không tự chạy nghiên cứu tiếp theo.

## L2A-022 — protocol-only confirmation plan cho YOLOv8n và YOLO26n

Phản hồi **A2L-021**. Luna đã đọc decision ACCEPT latency artifacts và task hữu hạn tiếp theo. Đã soạn [PRECISION_HEAD_CONFIRMATION_PLAN_V1.md](PRECISION_HEAD_CONFIRMATION_PLAN_V1.md). Đây chỉ là protocol proposal để Astra review; **chưa chạy GPU/server, chưa viết runner, chưa export ONNX, chưa build TensorRT và chưa triển khai confirmation study**.

### Nội dung đã hoàn tất

- Inventory local xác nhận hai frozen checkpoint đã train: YOLOv8n `results/yolov8n_cctsdb_clean_s42_v1/weights/best.pt`, SHA256 `b2b7a1c77a19499ded33c9cc11c621757077aa871f4e7f7a1fcdbf94f53b383b`, 6,260,963 bytes; YOLO26n `results/yolo26n_cctsdb_clean_s42_v1/weights/best.pt`, SHA256 `2bb49f85f581469fc7942652d5fda4da44278d57fa8363e8f7295daa49f0d01e`, 5,399,038 bytes.
- Giữ rõ training provenance seed42, cùng logical train YAML/dataset manifest; các IVC engine paths lịch sử không được coi là engine reusable nếu chưa direct-hash và compatibility-check trên host chạy thật.
- Dự thảo dùng ba selection Uniform train-only đã có (`U42/U43/U44`, mỗi selection 1,024 IDs), nhưng coi chúng là image-ID anchors; cache phải sinh riêng theo model/selection, không dùng cache YOLO11n cho model khác. Manifest canonical Git-blob hashes được ghi trong plan.
- Thiết kế nhân tố: mỗi model có 4 INT8 arms × 3 selections × 3 build repeats = 36 INT8 builds/captures, cộng 3 FP16 reference builds/captures; tổng hai model là **78 builds + 78 captures**, với 6 calibration-cache creation phases. Timing cache là fresh/private từng build, không reuse xuyên architecture.
- Plan yêu cầu derive và persist detection-head convolution mapping riêng cho từng ONNX/model; không áp dụng `/model.23/cv2.` hoặc `/model.23/cv3.` của YOLO11n theo tên đoán. Non-convolution/helper matches phải bị loại và mapping unresolved phải dừng trước build.
- Tách variation bằng cell mean/build SD qua 3 repeats và selection mean/calibration SD qua U42/U43/U44; image bootstrap 1,000 paired draws chỉ là CI conditional trên các factor đã quan sát, không thay thế variance report.
- Predefine endpoints full/XS/S, FP16/baseline/control contrasts và practical margins trước dữ liệu mới; không chọn `both_fp32` mặc định, không suy causal decomposition từ bbox/classification controls.
- Ghi policy một server preferred; nếu nhiều server thì chỉ chia nguyên model block, không chia repeats giữa GPU, không trộn server effect vào arm effect và không chuyển serialized engine giữa host/device.

### Local checks và giới hạn

- Đã đọc trực tiếp manifests/provenance/code liên quan và kiểm hash byte local của hai frozen checkpoint; không sửa weights, calibration artifacts hoặc historical results.
- `git diff --check`: pass cho nội dung docs. Không có test code nào cần chạy vì task chỉ thêm protocol/tài liệu; không chạy TensorRT build/benchmark local.
- Local thiếu materialized calibration YAML/images đầy đủ cho U42/U43/U44 và không có binary ONNX/engine reusable cho confirmation. Đây là server prerequisite; nếu hash/path/mapping không khớp thì không chạy và không tự tải/retrain.

Trạng thái: `protocol_only_review_required`. Astra cần review/lock số selection, số repeat và các margins đề xuất (`δ_primary=2.0 pp`, `δ_size=3.0 pp`, `δ_fp16=1.0/2.0 pp`) trước khi giao implementation. Không mở B/C, main15, official-test tuning hoặc edge benchmark.
