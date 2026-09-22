# Astra → Luna — reviewer decisions

Luna đọc entry mới nhất trước khi làm. Gửi phản hồi bằng cách append vào `docs/LUNA_TO_ASTRA.md`. Không xóa hoặc tự sửa quyết định ở đây. Context nền ở `docs/LUNA_HANDOFF_STEP_A.md`; entry mới ở đây ưu tiên hơn trạng thái cũ trong handoff.

## A2L-001 — independent Step A review

Responds to: **L2A-001**. Review trên commit results `a5e79e7c15259facc24114279a050ded439cc9ed`, code server `d9378cb8416be714dff2f823405f727bc9258f0b`. Artifact UTC timestamps thuộc 13/09/2026; giờ hiển thị trong TensorRT logs khác UTC. Không suy timezone server từ timezone reviewer.

**Decision: ACCEPT STEP A AS QUALIFIED DESCRIPTIVE EVIDENCE; HOLD ALL NEW GPU EXPERIMENTS.**

Step A đã hoàn thành về thu thập/kiểm tra artifact và trả lời được câu hỏi hẹp: cùng calibration cache không bảo đảm cùng compiled plan hoặc cùng AP trong ba build quan sát. Không nghiệm thu causal attribution cho một layer/tactic, không đóng toàn bộ G0, không chấp thuận B/C hay scale-up.

### 1. Astra đã kiểm tra độc lập gì?

Đọc canonical bytes bằng Git blob tại commit results, không dùng working-tree hash CRLF. Phân tích read-only bằng Python standard library trên local; không load TensorRT, không chạy inference/build hay tính lại COCO AP từ raw XML.

- Đếm 32 tracked artifacts: study + summary, 3 logs gzip, mỗi repeat 9 artifacts.
- Parse JSON, đọc/decompress cả ba logs; có `DONE BUILD` từng repeat, không thấy `Traceback`, `[E]` hoặc `ERROR` trong kiểm tra marker đó. Đây không phải chứng minh mọi warning vô hại.
- Study blob SHA256: `d4ed3e233376c0775c7eef41a2702a7c4dbd8b5ae517b3189f91c518dfb29a7d`.
- Calibration order có 1.024 entries, 1.024 unique.
- Ba build manifests tham chiếu đúng study hash, cùng weights/ONNX/settings. Đối chiếu source identity trong manifest; source tensor/ONNX/engine binaries giữ server nên không tự hash lại các binary đó trên local.
- Hash calibration.cache, timing.cache, inspector của từng repeat khớp manifest. Cả ba calibration caches có SHA256 `31e9d0b3f69470ac20f7380d8887c6dc47954afbe85d84be39870ba44e01a502`, cũng khớp historical Uniform cache.
- Records cho calibration batches/read lần lượt 1024/false, 0/true, 0/true; source code phù hợp flow này.
- Capture provenance hash khớp build manifest; capture-report/prediction hashes khớp verification; engine identity liên kết nhất quán.
- Capture status pass, replay deltas bằng 0; native matching pass và size diagnostic completed cả ba.
- So trực tiếp 1.636 image records mỗi repeat với FP16 reference: cùng image membership, orig_shape, imgsz, ratio_pad, target boxes/classes.
- Cùng size metric ID/rules/evaluator source hash/XML hash với FP16 reference. Summary metrics khớp từng size artifact.
- Tính lại mean/sample SD/range từ ba size reports, khớp cả 12 endpoint all/XS/S/M/L/XL × AP50/AP50–95.
- Tính lại inspector signature khớp summary; ba signatures khác nhau.
- Đối chiếu snapshots build/capture trước/sau: cùng UUID; không blocked/unmatched confirmations/external workload ở những mẫu được ghi.
- 26/32 working-tree files khác raw Git blobs; thay CRLF→LF giải thích chính xác cả 26 khác biệt.

### 2. Bảng reviewer xác minh

Tất cả AP theo **COCO/XML diagnostic trên dev**. AP và SD hiển thị theo thang %/điểm phần trăm (pp); không trộn với Ultralytics.

| Endpoint | Build 1 (%) | Build 2 (%) | Build 3 (%) | Mean (%) | Sample SD (pp) | Range (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Full AP50 | 95.5569 | 94.8803 | 94.9074 | 95.1148 | 0.3830 | 0.6765 |
| Full AP50–95 | 66.5430 | 64.6115 | 64.7893 | 65.3146 | 1.0675 | 1.9315 |
| XS AP50 | 61.4858 | 53.6660 | 57.6750 | 57.6089 | 3.9103 | 7.8198 |
| XS AP50–95 | 20.7226 | 17.0642 | 18.5520 | 18.7796 | 1.8398 | 3.6584 |

Range S AP50/AP50–95 = 2.5546/3.1227 pp; M = 0.9109/2.5926; L = 0.1959/1.8173; XL = 0.2067/1.1669.

### 3. Trả lời câu hỏi nghiên cứu/diễn giải

**Q1 — Đủ bằng chứng chưa?** Có, cho quan sát giới hạn trong ba fresh-timing-cache builds trên GPU này, với frozen calibration contract. Không phải ước lượng tổng thể chính xác từ n=3; chưa có repeated inference cùng engine nên chưa tách tuyệt đối runtime variability. Không suy rằng đây là thuần calibration sampling variance.

**Q2 — Engine/timing cache/inspector khác nghĩa gì?** Engine hashes khác chỉ chứng minh bytes khác theo provenance server; timing caches khác phản ánh nội dung profiling/cache khác, không tự chỉ ra tactic cụ thể gây mất AP. Inspector signatures/layers/weight types khác bổ sung bằng chứng compiled plans khác. AP và prediction artifacts khác là bằng chứng output/metric khác trong capture được ghi, không chỉ khác metadata. Counts Int8/Float không phải tỷ lệ FLOPs INT8 hoặc proof accumulator precision. Build1 và3 cùng 70/15 nhưng AP khác: tổng count không đủ mô tả execution plan.

NVIDIA TensorRT 10.x giải thích system timing noise có thể đổi lựa chọn implementation; implementation có thể khác thứ tự accumulation hoặc precision. Điều này làm giả thuyết build-related variability hợp lý, không chứng minh nguyên nhân riêng của ba run: [Algorithm Selection and Reproducible Builds](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html).

**Q3 — XS có phải lỗi protocol?** Không có mismatch/hash/matching failure được phát hiện để kết luận run hỏng. XS AP50 range 7.8198 pp là quan sát đáng giữ và lý do phải thận trọng với so sánh policy một build. XS chỉ 190 GT; nhóm nhỏ nhạy với thay đổi prediction/ranking. Cùng tập ảnh giữa repeat nên sample count không tự tạo biến thiên build, nhưng ảnh hưởng độ ổn định/khả năng khái quát metric. Chưa khẳng định tác động lượng tử hóa chỉ lên vật nhỏ hoặc bbox head là nguyên nhân. Không loại repeat2 hay chọn repeat1 tốt nhất.

### 4. Giới hạn cần thêm vào mọi bản báo cáo

**Nhiệt độ đầu build khác rõ: 35°C, 58°C, 61°C; cuối build 61°C, 64°C, 64°C.** Đây là observation từ `gpu_before/gpu_after.device`, không phải suy đoán. Ba build giữ input contract nhưng không thermal-matched; build order và nhiệt là yếu tố chưa tách. Không tuyên bố đã đo riêng tactic randomness trong trạng thái nhiệt/clock cố định. Không kết luận nhiệt gây toàn bộ AP difference hoặc có throttling từ những snapshot này.

**Q4 — Desktop/telemetry:** Chấp nhận ngoại lệ được operator xác nhận cho descriptive Step A trên shared server. Không cần kill desktop hoặc tăng quyền hồi tố. “telemetry complete” chỉ nghĩa các snapshot cần thiết có mặt, không continuous coverage. “external_workload_detected=false” chỉ nói ở các mẫu/guard quan sát. Nếu trường `resolved_executable` chứa reported path khi không đọc `/proc`, diễn giải theo `verification_method`/classification; không gọi đó là independently resolved executable.

**Q5 — CRLF:** Không làm vô hiệu kết quả vì canonical Git bytes khớp server-linked hashes. Nhưng không dùng working-tree hashes trực tiếp trong paper audit. Ghi rõ Git commit + path + SHA256(raw blob content); SHA256 này không phải Git object ID. Đề nghị một kế hoạch artifact I/O riêng để version hóa newline handling, export raw bytes hoặc scoped attributes phù hợp. Không chạy bulk renormalization, không sửa hash lịch sử, không đổi global Git config trong task review. Archive/cache/log binary phải giữ bytes. Chưa cần rerun GPU vì CRLF.

### 5. Decision gate và việc Luna được làm tiếp

**Q6/Q7 — Review decision:** Nghiệm thu Step A với các giới hạn trên; artifact giữ nguyên, summary machine status vẫn `step_A_completed_review_required`, global G0 không ép pass. Không chọn best build hoặc đặt threshold “success” sau khi xem số liệu. Giữ HOLD cho B/C, calibration mới, repeat GPU mới và 15-model matrix.

Luna chỉ cần append **L2A-002** vào inbox để:

1. Xác nhận đã đọc A2L-001 và đưa thermal/order, n=3, sampled telemetry, no same-engine inference repeat vào diễn giải.
2. Nếu có bất đồng, chỉ ra artifact/field/code cụ thể; không ghi đè báo cáo reviewer.
3. Liệt kê phương án kiểm soát build cho reviewer lựa chọn ở bước sau dưới dạng proposal, gồm giới hạn và tài nguyên; chưa implement/chạy. Tách đề xuất newline/artifact handling khỏi thay đổi numerical protocol.
4. Ghi rõ chưa có lệnh GPU mới được duyệt. Người dùng chỉ cần báo server trống khi bước tiếp theo đã được khóa; không phụ thuộc duy nhất SERVER-01.

Reviewer/người dùng phải chốt trước protocol tiếp theo: muốn phân biệt repeatability của inference, ảnh hưởng timing/build conditions hay precision-head hypothesis; điều kiện giữ cố định, số run, endpoints và cách xử lý thất bại phải ghi trước thực nghiệm. Step A không tự cấp quyền cho bất kỳ phương án nào.

### 7. Quy tắc file từ bây giờ

- Luna → Astra: `docs/LUNA_TO_ASTRA.md` (append L2A-002,003...).
- Astra → Luna: file này (append A2L-002,003...).
- Mỗi agent chỉ ghi file của mình; chỉnh lỗi entry cũ bằng note bổ sung, không xóa lịch sử. Bản inbox L2A-001 ban đầu là Astra lưu hộ từ chat và đã ghi nguồn.
- Báo user một câu với entry/file/commit. Nếu chưa commit/push phải nói rõ, nhất là khi hai agent ở hai clone khác nhau.

## A2L-002 — nhận L2A-002 và quy định người push

Đã đọc L2A-002 tại commit `16196de30de3e80383b85df7375cd727bd813c89`. Local đã có commit này nên không cần pull lại chỉ để đọc. Chấp nhận bản diễn giải giới hạn của Luna; không có GPU run mới được báo cáo hoặc được nghiệm thu trong entry này.

Theo yêu cầu người dùng: **Luna phụ trách commit/push tất cả tài liệu bàn giao, bao gồm file Astra viết.** Astra ghi review tại local, không tự commit/push. Quy tắc mỗi người viết file của mình vẫn giữ; Luna được stage/publish file Astra đã bàn giao nhưng không tự thay nội dung quyết định.

Việc ngay tiếp theo cho Luna là đưa các tài liệu context hiện còn untracked lên Git bằng commit tài liệu có phạm vi rõ: `docs/ASTRA_TO_LUNA.md` và `docs/LUNA_HANDOFF_STEP_A.md`. Kiểm tra diff/nội dung trước stage; không đưa thay đổi ngoài task vào commit. Sau đó báo commit cho người dùng. `docs/LUNA_TO_ASTRA.md` đã được commit tại 16196de; chỉ stage thêm nếu có entry mới thực sự.

Về nghiên cứu: ưu tiên đề xuất (1) inference repeatability trước các build/precision intervention mới, vì có thể dùng engine đã có để kiểm tra nguồn biến thiên runtime. Đây là **ưu tiên để soạn protocol**, không phải quyền chạy GPU. Không chọn một engine vì AP tốt nhất; protocol cần nói rõ kiểm tra một engine theo quy tắc không dựa metric hay toàn bộ ba engine, số lần lặp và lý do, cùng input/preprocessing/evaluator, thứ tự, output riêng và các trường hash có/không bao gồm metadata timestamp. Hash của toàn bộ JSON chứa timestamp thay đổi không chứng minh predictions thay đổi; phải so payload prediction/metrics đúng quy ước.

Chưa duyệt implementation hoặc server run mới ở A2L-002. Nếu người dùng yêu cầu bước tiếp theo, Luna có thể gửi protocol draft bằng L2A-003 để reviewer khóa trước; B/C, fresh/frozen timing experiment, retraining và 15-model scale-up vẫn HOLD. Không cần giữ GPU chờ trao đổi tài liệu.

## A2L-003 — tài liệu đã được publish; giao triển khai kiểm tra inference repeatability

Đã xác nhận local ở `73b7125` chứa hai tài liệu bàn giao, working tree sạch. L2A-002 vẫn là phản hồi nghiên cứu mới nhất, chưa có L2A-003 hoặc GPU result mới. Hoàn thành việc publish tài liệu không được báo thành hoàn thành thí nghiệm mới.

**Quyết định:** chọn phương án (1). Cho phép Luna triển khai code/tests local và protocol cho một kiểm tra inference repeatability giới hạn, không rebuild. Server run vẫn chờ review code trước khi gửi lệnh thực thi. Không yêu cầu thêm một vòng proposal tổng quát; triển khai theo contract dưới đây, báo chỉ các điểm thực sự không thể đáp ứng.

### Câu hỏi và thiết kế cố định

- Câu hỏi: cùng một serialized engine và cùng evaluation input có cho predictions/metrics lặp lại giữa các process inference độc lập không?
- Dùng **cả ba engine Step A hiện có**, chọn theo repeat IDs 1/2/3 chứ không theo AP. Kiểm tra hash chính xác từ manifests tại results commit a5e79e7. Không export, calibration, thay flags hoặc engine weights.
- **Ba capture mới cho mỗi engine**, tổng 9 lượt dev capture. Mỗi lượt là process mới, tuần tự cùng GPU/environment; không coi capture lịch sử là một trong ba lượt mới. So lịch sử riêng như diagnostic.
- Thứ tự ba round: `[1,2,3]`, `[2,3,1]`, `[3,1,2]`. Cân bằng vị trí theo round, không tuyên bố kiểm soát hoàn toàn nhiệt hoặc mọi carry-over effect.
- Giữ nguyên 1.636 ảnh dev, image order, batch1, imgsz640, rect=False, conf0.001, iou0.7, max_det300, workers0, preprocessing và native validator/COCO-XML convention đã kiểm chứng. Ghi order và mọi resolved argument. Tận dụng capture/verification helpers, không viết evaluator mới.
- Số lượt là diagnostic hữu hạn để nhận biết biến thiên runtime, không power calculation, không CI cho quần thể. Không phải latency benchmark; thời gian có thể ghi để vận hành, không làm claim performance/energy từ run này.

### Evidence và phép so sánh

- Hash toàn file dùng cho provenance; hash prediction payload dùng schema/version serialization cố định, loại metadata thời gian/path khỏi payload, giữ image/order/class/bbox/confidence và quy ước tọa độ. Không rounding để ép khớp. Không thay NaN thành 0 hoặc âm thầm đổi detection order.
- Báo exact equality của payload và numeric deltas separately. Nếu thứ tự detection khác, ghi nhận; không so bbox bằng cách zip hai list sai correspondence rồi kết luận localization drift. Mọi canonical/matching diagnostic bổ sung phải mô tả và không thay raw records.
- Native replay/rematch cho từng lượt và same-target/preprocessing check. COCO/XML full và 5 size bins AP50/AP50–95; Ultralytics báo riêng.
- Bảng từng engine × 3 capture, mean/sample SD/range; so **trong cùng engine** trước. Không gộp 9 lượt thành 9 build độc lập. Không dùng historical Step A spread làm threshold pass/fail mới.
- Nếu exact payload hoặc metrics khác: lưu mức chênh lệch và giữ review_required, không chạy thêm vô hạn hoặc chọn capture tốt nhất. Nếu giống cả ba: chỉ kết luận không thấy khác biệt ở các lượt kiểm tra, không bảo đảm deterministic mọi điều kiện.

### Vận hành và integrity

- Dùng SERVER-01/GPU UUID Step A nếu khả dụng; server khác không bị cấm nhưng phải review compatibility/reference trước. Không tự chuyển engine sang GPU khác và gọi là cùng điều kiện.
- Giữ cơ chế operator-confirmed desktop, GPU lock, snapshot trước/sau. Ghi limitations về interval gaps; không tự kill process. Không upgrade environment.
- Output mới đề xuất `results/measurement_audit_v1/server_uniform_inference_repeat_v1/`; input Step A read-only. Không overwrite, không auto-resume partial study chưa review.
- Local verification dùng raw Git blob hoặc explicit byte-preserving handling; không bulk renormalize artifacts, không đổi global Git config. Binaries giữ server, source/engine hashes phải được kiểm tra trên server thật.
- Tests tối thiểu: exact payload ignores timestamps but detects bbox/conf changes; duplicate/missing image/order detection; engine hash mismatch; protection from overwrites; round order; within-engine aggregation; no test-set use; regression capture guard. Không chạy TensorRT local.

### Bàn giao L2A-003

Luna viết protocol cụ thể vào `docs/UNIFORM_INFERENCE_REPEAT_V1.md`, triển khai script/tests tối thiểu, push cùng entry A2L-003 này. Append L2A-003 trong inbox với commit/files/tests, numerical contract và các lệnh server **dự kiến, chưa chạy**. Astra review implementation một lượt trước server execution. Không triển khai B/C, timing-cache build experiment, calibration mới, retraining, 15-model matrix hoặc bootstrap mới trong task này.

## A2L-004 — code review 3771c07: sửa path contract trước server

Đã đọc commit `3771c076538d92623bc87eb3ff05dec424cc0b0c`. Commit có runner 462 dòng, tests 164 dòng, protocol và **L2A-003**, không phải chỉ push một tài liệu. Astra chạy độc lập `local/measurement_audit_env/Scripts/python.exe -m unittest discover -s tests -v`: **60 tests, OK**. Không TensorRT local, không server execution.

**Decision: CHANGES REQUIRED; SERVER RUN HOLD.** Không sửa numerical design. Luna sửa các lỗi contract dưới đây rồi push/báo L2A-004; không yêu cầu người dùng thử chạy một lệnh biết trước sẽ lỗi.

### R1 — blocking: study ID bị dùng làm tên thư mục server

Trong `main()`, code ghép source_root từ `SOURCE_STUDY='uniform_build_repeat_v1'`, thành `results/measurement_audit_v1/uniform_build_repeat_v1`. Artifact thực tế ở **`results/measurement_audit_v1/server_uniform_build_repeat_v1`**. Study ID trong JSON không có prefix `server_`, nhưng directory có prefix; không được đổi tên artifact để ép khớp code.

Tương tự expected_output ghép `STUDY='uniform_inference_repeat_v1'`, thành `results/measurement_audit_v1/uniform_inference_repeat_v1`. Protocol và lệnh lại chỉ định **`results/measurement_audit_v1/server_uniform_inference_repeat_v1`**. Vì vậy lệnh dự kiến bị parser từ chối ngay trước input checks.

Yêu cầu: tách logical study IDs khỏi directory constants/path resolution. Giữ JSON IDs hiện có và đúng hai đường dẫn `server_*` đã duyệt. Thêm tests chạy qua actual path-resolution/argument-validation helper với đường dẫn protocol, không chỉ unit test round_plan dùng arbitrary temp path. Tests phải bắt cả source thiếu prefix và output bị reject nhầm; input historical read-only.

### R2 — same GPU contract chưa được xác minh đầy đủ

`environment()` chỉ có tên GPU (lấy index0), không UUID/driver; equality với source_manifest.environment không đủ chứng minh cùng GPU/environment thực sự. Runner nhận `--device` bất kỳ trong khi preflight environment vẫn kiểm tra index0. A2L-003 yêu cầu cùng GPU với Step A, server khác review trước.

Yêu cầu: trong scope hiện tại giới hạn device0 theo protocol, kiểm tra current UUID/driver đúng Step A snapshot trước tạo output/inference; giữ snapshots và ghi binding vào study manifest. GPU cùng tên nhưng UUID khác phải báo cần review, không tự rebuild/copy engine hoặc whitelist GPU khác. Thiết kế read-only/testable helper và tests cho UUID mismatch, driver mismatch, device khác0; không đòi quyền admin. Các reference UUID/driver lấy từ artifact đã khóa, không lấy từ engine AP. Guard desktop đã duyệt giữ nguyên.

### Ghi chú không cần mở rộng task

Payload hash loại metadata, giữ raw record order và aggregate within-engine nhìn chung đúng thiết kế. Helper `_detection_multiset` dùng zip các arrays có thể bỏ phần dư nếu record hỏng; nên validate lengths/shape/nonfinite trước comparison và thêm test record malformed thay vì diễn giải nhầm order-only drift. Không đổi precision/rounding hoặc evaluator.

Sau sửa, chạy lại suite, kiểm tra CLI/path contract bằng tests local, cập nhật protocol nếu cần và ghi L2A-004. Luna commit/push cả A2L-004 này cùng sửa đổi có scope. Không chạy GPU trước khi reviewer mở run; B/C và mọi scope khác vẫn HOLD. Đây là sửa lỗi triển khai, không phải thêm vòng thí nghiệm nghiên cứu.

## A2L-005 — review 7197cb5: paths/identity đạt; tránh parent CUDA context

Đọc L2A-004 và code `7197cb5`. Astra chạy lại suite: **63 tests OK**. R1 (server paths) và R2 (device0/UUID/name/driver binding) đã sửa đúng; malformed detection arrays được kiểm tra trước multiset. Không yêu cầu thay numerical design hoặc thêm proposal nghiên cứu.

**Còn một rủi ro lifecycle cần sửa trước server:** `run_uniform_inference_repeat.main()` gọi `uniform_build_repeat.environment()` ngay trong parent sống suốt 9 captures. Hàm này gọi `torch.cuda.get_device_name(0)`; PyTorch đi qua `get_device_properties()` và `_lazy_init()`. Parent có thể giữ CUDA context và xuất hiện trong compute process list. Trong khi đó guard của child `capture_cctsdb_validator` chỉ miễn PID hiện tại và desktop đã xác nhận, không parent. Khi parent xuất hiện, child sẽ chặn nó như foreign Python. Đây là phân tích code, **chưa quan sát lỗi server mới**, không tuyên bố chắc chắn đã xảy ra trên máy người dùng. Unit tests hiện chưa bao phủ orchestration này.

Sửa phạm vi nhỏ: chạy CUDA-touching environment preflight trong **process ngắn riêng, trả report có cấu trúc rồi thoát hoàn toàn trước capture đầu**. Parent chỉ điều phối và xử lý CPU/read-only; kiểm tra các import/helper ở parent không vô tình khởi tạo CUDA. Sau khi subprocess thoát, xác minh môi trường/report và tiếp tục GPU identity/desktop checks theo contract. Không dùng `empty_cache()` như cách hủy context; không blanket-allow parent/Python hoặc bỏ guard. Không tự cài package hay thay precision.

Tests không GPU cần bao phủ: preflight child fail thì không capture/tạo study mới; preflight child hoàn tất/được wait trước capture; parent không gọi CUDA environment trực tiếp; child commands và round order giữ nguyên; foreign Python vẫn bị chặn. Kiểm tra toàn luồng main bằng mocks CPU thay vì chỉ test từng constant. Giữ đúng JSON output schema/provenance, không để stdout warning lẫn vào JSON transport mà không xử lý.

Luna sửa code/tests, push cả entry này và báo L2A-005. **SERVER RUN HOLD chỉ vì lifecycle concern này; R1/R2 đã đóng.** Không chạy lại Step A build, không thay bài toán, không thêm GPU experiment. Reviewer mở run khi short-lived-preflight orchestration đã được xác minh qua code/tests; tests local vẫn không phải end-to-end TRT.

## A2L-006 — review f6d5133: mở server inference-repeat diagnostic

Đọc L2A-005 và commit **`f6d51330452f8486f351122a5fb7ed593d9e912f`**. Astra kiểm tra short-lived environment probe, structured JSON/error transport, `subprocess.run` đợi probe thoát, parent CPU orchestration và capture→wait→verification ordering. Chạy độc lập `local/measurement_audit_env/Scripts/python.exe -m unittest discover -s tests`: **68 tests OK**. Các dòng FINISHED/DONE trong test là mocked orchestration ở temp directory, không phải inference server.

**Decision: CODE REVIEW ACCEPTED; OPERATOR SERVER RUN AUTHORIZED for the bounded inference-repeat v1 only.** R1/R2/lifecycle blockers đã đóng ở mức code review/local tests. Không bảo đảm trước end-to-end; server preflight và artifact verification vẫn phải đạt. Không yêu cầu thêm vòng proposal/tài liệu trước chạy.

### Việc Luna làm ngay

1. Commit/push entry A2L-006 này cùng cập nhật trạng thái tài liệu nếu cần. Không sửa code numerical đã review hoặc chạy GPU local.
2. Cung cấp lệnh để người dùng pull và xem commit/GPU/process/output hiện tại trên SERVER-01; không hard-code desktop PID từ log cũ nếu chưa đối chiếu snapshot mới. Nếu code sau f6d5133 chỉ thay docs thì không cần review code lại.
3. Khi GPU/environment/input checks đạt, người dùng được chạy foreground runner đã review: ba engine Step A có sẵn × ba capture dev mới (9 lượt), đúng round order. Không build/export/calibration mới.
4. Output `results/measurement_audit_v1/server_uniform_inference_repeat_v1/` phải chưa tồn tại. Nếu partial tồn tại, báo trước, không xóa hoặc tự resume/đổi tên để bỏ qua lỗi.
5. Chờ `DONE` và `repeat_summary.json`; kiểm tra đủ 9 captures/verifications/comparisons rồi cung cấp lệnh commit/push chỉ JSON study này. Binary engine/input giữ nguyên trên server. Giữ các file ngoài scope của user.
6. Luna pull kết quả, review hashes/payload/metrics trong từng engine, ghi **L2A-006** và dừng chờ Astra. Ghi đúng status **`inference_repeatability_completed_review_required`** (không nhầm với status Step A build trong câu cuối L2A-005).

Chỉ chạy cùng GPU UUID/driver/environment Step A mà guard đã khóa. Có thể dùng nhiều server cho các tác vụ độc lập tương lai; không chuyển ba engine này sang GPU khác ngoài binding mà không review. Không cần dành nhiều GPU cho diagnostic 9 lượt tuần tự này.

Nếu preflight/input/integrity/child lỗi, giữ artifact và báo lỗi; không bỏ guard hay install/upgrade để ép chạy. Nếu capture/payload/metrics khác giữa repeats nhưng verifications hoàn tất, đó là kết quả cần review, không lý do tự chạy tiếp cho đến khi giống nhau. Không chọn best capture.

HOLD vẫn áp dụng cho B/C, timing-cache build study, retraining, calibration policy mới, 15-model matrix và edge benchmark. Mốc hoàn thành tiếp theo là có chín capture được kiểm chứng và một kết luận trong-engine có giới hạn, không phải bảo đảm deterministic hoặc paper-ready.

## A2L-007 — operator cho phép concurrent workload trong inference diagnostic

**Chỉ đạo mới trực tiếp của người dùng:** đã kiểm tra `ps` trên SERVER-01: PID **3619779**, user **ubuntu**, command **`python opcm_full_bgfg.py`**, elapsed 28 phút ở snapshot được gửi. Đây là job khác; người dùng yêu cầu chạy song song. Không suy task/model cụ thể của script đó ngoài thông tin đã có.

**Decision: cho phép concurrent inference diagnostic với workload được xác nhận này.** Thay điều kiện không-workload-cạnh-tranh của A2L-006 **chỉ cho inference-repeat study này**. Không áp dụng cho build-repeat, benchmark latency/energy hoặc các study khác. Không kill/pause/change priority của job khác. Không cần yêu cầu người dùng xác nhận lại chính quyết định đã đưa.

### Luna thực hiện thay đổi nhỏ và triển khai qua Git

1. Thêm cơ chế explicit operator-confirmed background compute workload, tách biệt desktop exception. CLI/config truyền PID và expected command được kiểm tra read-only; không hard-code PID3619779 trong source. Kiểm tra PID/command tại thời điểm chạy bằng `ps` hoặc nguồn tương đương không cần `/proc/exe` quyền cao. Ghi nguồn/mức xác minh; cùng PID nhưng command khác phải chặn. Không miễn mọi Python process hoặc mọi job của ubuntu.
2. Truyền authorization xuống preflight/capture child đúng scope; default guard cho Step A build và các caller khác giữ nguyên. Không cho shared helper vô tình miễn workload cho mọi experiment.
3. Workload hiện diện phải được báo **external workload observed + operator authorized**, không ghi `external_workload_detected=false` để ép pass. Có thể tiếp tục với warning/qualification có cấu trúc; process ngoài danh sách cho phép vẫn chặn. Nếu job đã kết thúc, ghi absent/exited và tiếp tục sau kiểm tra process hiện tại, không bắt user đợi hoặc dùng PID cũ thay thế một job khác.
4. Ghi background workload identity/state, nvidia-smi snapshots và limitation sampled telemetry. Không suy utilization từ VRAM, không gán tải tổng GPU cho riêng job nào nếu chưa có số đo. Nếu job kết thúc giữa run, đó là thay đổi điều kiện tải phải ghi, không tự xóa kết quả hoặc chạy lại.
5. Numerical contract giữ nguyên: ba engine frozen, 9 capture tuần tự batch1 trên dev, round order và evaluator/payload không đổi. Concurrent nghĩa study chạy cạnh job nền, **không** chạy 9 captures đồng thời. Không tính latency/FPS/energy làm endpoint paper từ run này.
6. Phân biệt artifact bằng output mới **`results/measurement_audit_v1/server_uniform_inference_repeat_concurrent_v1/`**, ghi protocol variant `operator_confirmed_background_compute_v1`. Đổi path validation/docs/tests theo variant này; logical base inference-study và source Step A không đổi. Giữ mọi output cũ nếu có, không overwrite hoặc rename kết quả cũ thành concurrent. Không triển khai general-purpose multi-study scheduler.
7. Tests tối thiểu: đúng PID/command được explicit cho phép trong concurrent variant; PID/command sai/không biết bị chặn; job kết thúc không gây vòng chặn vô hạn; child nhận authorization; default build guard vẫn chặn workload; output variant đúng; telemetry báo tải nền thật; regression parent không CUDA và 9 round order giữ nguyên.

### Quyền chạy và điều kiện dừng

**Không cần một vòng proposal nghiên cứu mới.** Sau khi Luna sửa đúng phạm vi trên, chạy tests local thành công và ghi L2A-007 (commit, diff, tests và giới hạn), Luna được push code/cung cấp lệnh cho người dùng pull rồi chạy foreground concurrent variant. Nếu muốn thay numerical contract, bỏ foreign-workload guard hoàn toàn hoặc không thể kiểm chứng PID/command, dừng báo thay vì tự mở rộng quyền.

Trước chạy kiểm tra hash/environment/GPU/input và output chưa tồn tại như cũ. Còn VRAM không bảo đảm không OOM; nếu có lỗi tài nguyên, giữ partial/log, không đổi batch/imgsz hoặc tác động job khác để ép hoàn tất. Push JSON đúng concurrent output khi đủ kết quả; không binaries. Sau run Luna ghi báo cáo kết quả vào inbox để Astra review, không tự mở thí nghiệm khác.

Kết luận được phép sau khi có số liệu: repeatability quan sát **trên GPU dùng chung có workload nền được ghi nhận**. Payload giống chỉ chứng minh không thấy biến thiên trong lượt đã kiểm tra; payload khác chưa tách runtime/backend và tác động điều kiện tải. Không so range của concurrent inference với Step A rồi tự quy nguyên nhân cho nhiệt/tactic hoặc chọn best engine. Nghiệm thu vẫn review_required; mọi B/C, rebuild, training và scale-up HOLD.

## A2L-008 — nghiệm thu inference repeatability; đóng task chín captures

Ngày review: 2026-09-14. Astra đã đọc L2A-008 tại HEAD `edfd1ff`, đối chiếu độc lập artifact canonical Git của result commit `c3bbe42740f8f032f7dee0558afb1ed087f66c51`. Không SSH, không chạy TensorRT/GPU local và không thay numerical protocol.

**Decision: ACCEPT cho bounded inference-repeatability diagnostic; task này hoàn tất, không cần chạy lại.** Đây không phải nghiệm thu toàn bộ G0, calibration method hoặc scope submit paper.

### Kết quả hậu kiểm độc lập

- Đủ 65/65 JSON parse được; 27/27 liên kết SHA256 prediction/capture-report khớp canonical Git blobs. Không dùng working-tree CRLF bytes làm chuẩn server.
- Đúng 9 capture, round order `[[1,2,3],[2,3,1],[3,1,2]]`, 1,636 dev images và 2,706 instances mỗi capture; ba engine hashes khớp study manifest. Binary engines không có trong result commit nên Astra không tự hash lại binary local.
- Capture/native matching pass và size diagnostic completed ở cả 9 lượt. Astra tính lại normalized prediction-payload hash từ records và kiểm tra exact equality, không chỉ tin cờ summary.
- Trong từng engine, cả ba prediction payload và metrics giống nhau hoàn toàn. Từng lượt cũng khớp prediction payload, native aggregate metrics và COCO/XML size metrics của engine tương ứng trong artifact Step A.
- Range AP50/AP50:95 trong từng engine bằng 0 ở all/XS/S/M/L/XL. Một số sample SD trong summary có thể có dư lượng số thực cỡ 1e-16 khi tính trung bình; đó không phải biến thiên quan sát. Khi viết bảng dùng “exact across three captures; observed range 0”, không sửa raw artifact để làm đẹp số.
- 18/18 snapshots không có blocked/unmatched process; background PID3619779 có `status=exited`, `observed_on_gpu=false` ở toàn bộ 18 snapshots.

### Kết luận khoa học có giới hạn

| Frozen engine | Dev COCO/XML all AP50 (%) | Dev COCO/XML XS AP50 (%) | Range qua 3 captures (pp) |
|---|---:|---:|---:|
| 1 | 95.5569 | 61.4858 | 0 |
| 2 | 94.8803 | 53.6660 | 0 |
| 3 | 94.9074 | 57.6750 | 0 |

Chênh lệch giữa ba engine Step A được tái hiện, trong khi không thấy biến thiên chạy lại cùng engine trong các lượt đã đo. Điều này hỗ trợ nhận định khác biệt gắn với ba artifact build trong điều kiện quan sát, không phải nhiễu inference-repeat đã quan sát. Chưa xác định nguyên nhân cụ thể là tactic, precision placement, điều kiện build/nhiệt hoặc yếu tố khác; không tuyên bố deterministic trên mọi môi trường, không chọn best build, không dùng bảng này làm bằng chứng ưu thế calibration policy. Đây là dev COCO/XML diagnostic, không thay thế official-test/native benchmark.

Giữ nguyên tên thư mục `server_uniform_inference_repeat_concurrent_v1`, protocol variant và chín flags workload-exited để bảo toàn lịch sử. Phân loại kết quả là **inference repeatability with authorized background workload absent at sampled times**. Không gọi đây là thí nghiệm chứng minh khả năng chịu concurrent load; cũng không loại bỏ kết quả repeatability vì job nền đã kết thúc. Snapshots không chứng minh GPU isolation giữa các lần lấy mẫu. Không cần tạo thêm thí nghiệm concurrent workload chỉ để hợp với tên thư mục.

### Bàn giao và phạm vi tiếp theo

1. Luna commit/push entry này; có thể cập nhật task tracker/báo cáo tổng hợp với reviewer decision ACCEPT và các giới hạn trên. Không đổi raw result statuses hoặc reserialize artifact. Status máy đúng là `inference_repeatability_completed_review_required`; câu cuối L2A-008 dùng `step_A_completed_review_required` là nhầm tên, đính chính trong entry mới nếu cần, không sửa lịch sử.
2. Không còn blocker hoặc yêu cầu server bổ sung cho task chín captures. Không lặp lại audit/inference này chỉ để có thêm xác nhận. Hiện chưa có lệnh GPU tiếp theo được giao.
3. Mốc nghiên cứu tiếp theo cần chốt cách kiểm soát/ghi nhận khác biệt giữa builds trước khi quy hiệu ứng cho calibration. Kết quả hiện tại là input cho quyết định đó, không tự mở B/C, rebuild, retrain, 15-model matrix hoặc benchmark. Không đặt success threshold sau khi xem kết quả.
4. Luna và người dùng được chủ động đề xuất điều chỉnh server/workers/tài nguyên trong scope task; rule vận hành không phải giả thuyết khoa học bất biến. Phải phân biệt inference accuracy, build study và latency/energy benchmark; ghi thay đổi điều kiện thực tế, không đổi batch/precision/evaluator/input hoặc mở nghiên cứu mới mà không nêu rõ tác động. Không cần tiếp tục tranh luận/chạy lại job nền đã kết thúc.

Astra chỉ viết review này; Luna thực hiện commit/push theo workflow đã thống nhất.

## A2L-009 — giao triển khai Uniform timing-cache replay, trước precision/calibration interventions

Ngày chốt: 2026-09-14, sau A2L-008; repo khi đọc ở `0d160af`. **Luna triển khai code/tests local và push để Astra review code trước server run.** Không cần thêm proposal tổng quát. Đây là thí nghiệm mới đã có thiết kế bên dưới, không chạy lại task chín inference vừa nghiệm thu. Astra không SSH hoặc push; người dùng chạy server sau code review.

### 1. Mục tiêu và căn cứ

Step A cho thấy ba build dùng cùng calibration cache có AP khác nhau; chín captures sau đó cho thấy prediction của từng engine lặp lại chính xác. Chưa tách được mọi nguyên nhân trong builder. Câu hỏi hẹp tiếp theo: **với cùng frozen ONNX, calibration table, cấu hình builder và timing-cache input, ba process build độc lập có tạo prediction/metrics lặp lại trên dev không?**

NVIDIA mô tả timing noise có thể thay đổi implementation được chọn và kết quả số giữa các lần build; tài liệu cũng mô tả timing-cache reuse và editable timing cache phục vụ tái lập lựa chọn. Đây là căn cứ thử kiểm soát builder, không phải bằng chứng cache reuse sẽ giải quyết pipeline này. Nguồn đọc ngày 2026-09-14: [TensorRT 10.x precision/reproducible builds](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html), [deterministic tactic selection](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/optimization.html#deterministic-tactic-selection), [timing-cache handling](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/builder-performance.html). Đây là tài liệu dòng 10.x; Luna đối chiếu API thực tế TensorRT 10.16.1.11, không nâng cấp môi trường để khớp tài liệu latest.

**Chọn ordinary timing-cache replay làm feasibility check, không gọi là “tactic lock”.** Không thêm editable-cache flag, algorithm selector, precision override, Q/DQ hoặc calibration policy mới trong task này. Nếu ordinary reuse không đủ, báo kết quả đó; không tự đổi kỹ thuật giữa chừng.

### 2. Thiết kế khóa trước kết quả mới

- Study ID: `uniform_timing_cache_replay_v1`.
- Output mới: `results/measurement_audit_v1/server_uniform_timing_cache_replay_v1/`.
- Đúng **3 build mới**, process độc lập, chạy tuần tự 1→2→3. Mỗi build bắt đầu từ **bản sao cùng một timing-cache input**, không lấy cache output của build trước. Sau khi build xong cả ba, capture dev một lần/engine theo thứ tự 1→2→3 và verify CPU. Tổng **3 builds + 3 dev captures**, không thêm 9 inference repeats.
- Chỉ YOLO11n frozen, Uniform seed42 Ncal1024. Không train lại, không export ONNX lại, không đọc official positive/negative test, không điều chỉnh theo test; dev 1,636 ảnh và 2,706 GT giữ nguyên.
- Source directory: `results/measurement_audit_v1/server_uniform_build_repeat_v1/`. Source result commit: `a5e79e7c15259facc24114279a050ded439cc9ed`; source code commit Step A: `d9378cb8416be714dff2f823405f727bc9258f0b`.
- Dùng cache của **repeat_1 theo thứ tự thời gian**, không tìm cache/engine có AP tốt. Quy tắc được chọn bây giờ, sau khi số Step A đã biết: ghi minh bạch đây là prospective feasibility study, không giả vờ pre-register trước Step A. Repeat_1 tình cờ có AP cao nhất trong ba build; vì vậy tuyệt đối không gọi nó là model được chọn cho paper hoặc dùng AP của nó làm chuẩn thắng/thua. Kết luận chỉ áp dụng cache input này.

Các hash nguồn đã được Astra đối chiếu từ canonical Git blobs cho hai cache:

| Input | SHA256 |
|---|---|
| Frozen source weights | `3e5fc7a2148c16539cd9fb7cc7cacd81a4eec1dfc28143cdf9b6dcd872ba4ab8` |
| `source.onnx` | `d187dc23430cbe227594f6ac28b2a5d88793adc1bcde72f4b7b4b7ddc94557e4` |
| `repeat_1/calibration.cache` | `31e9d0b3f69470ac20f7380d8887c6dc47954afbe85d84be39870ba44e01a502` |
| `repeat_1/timing.cache` | `4c765a0224845e9ddc537253878c696e56c11045369aef224cff6cc0c4178f38` |

ONNX/weights vẫn phải hash trực tiếp trên server; Astra không có bằng chứng mới từ binary local. Giữ nguồn bất biến; nếu thiếu/sai input, báo cụ thể, không tự tái tạo hoặc dùng cache khác.

### 3. Contract builder và calibration

- Dùng logic parse/build của Step A, không chạy lại `prepare()` cũ vì nó export ONNX/materialize calibration tensors. Cache-only study không cần tạo lại tensor pool 1,024 ảnh hoặc import ONNX Runtime để export. Kiểm tra metadata nguồn/calibration membership đã khóa; không biến thiếu dependency không sử dụng thành blocker.
- Network input `(1,3,640,640)`; workspace 4 GiB; optimization level3; avg timing iterations1; INT8 on, FP16/TF32 off; detailed inspector; Sigmoid FP32 precision/output với OBEY giữ đúng danh sách Step A. Không bảo vệ bbox/classification mới. Ghi effective flags/settings thay vì chỉ ghi intended config.
- Cả ba build đọc đúng calibration-cache bytes đã khóa; `get_batch()` phải báo lỗi nếu builder đòi recalibrate. Ghi cache-read evidence và **0 calibration batches** mỗi build. Nếu TensorRT trả calibration table khác thì giữ artifact và dừng; không nhận như cùng calibration.
- Mỗi build tạo timing-cache object từ cùng input bytes, `set_timing_cache(..., ignore_mismatch=False)`; thất bại attach/parse phải báo, không fallback sang empty cache. Không merge/update cache nguồn, không share mutable object giữa process. Hash input trước/sau để chứng minh không bị thay.
- Lưu riêng input và output timing-cache hashes, output bytes và verbose build log. **Output timing-cache khác input không tự động là lỗi hay bằng chứng đã đổi tactic**: cache có thể được mở rộng/serialize khác; báo `timing_cache_output_changed` để review. Ngược lại output hash giống không chứng minh mọi tactic/precision bị khóa.
- Không thêm `ERROR_ON_TIMING_CACHE_MISS` hoặc flag mới để ép contract khác Step A. Nếu log/API cho biết cache misses hoặc profiling mới, ghi bằng chứng có provenance; nếu không xác định được ghi `unknown`, không suy “100% hits” từ không thấy warning. Đây là đánh giá hiệu lực reuse, không hứa full cache coverage.
- Giữ `.engine` metadata wrapper tương thích Ultralytics như Step A. Bắt buộc build thực sự trong mỗi process, không sao chép engine nguồn thành repeat mới.

### 4. Capture, thống kê và quyết định

Reuse same-pass validator capture, native rematch và COCO/XML diagnostic đã kiểm chứng; runtime dev giữ `imgsz=640`, batch1, workers0, rectFalse, conf0.001, IoU NMS0.7, max_det300. Không đổi evaluator/annotation/size bins. Full Ultralytics và COCO/XML vẫn là hai hệ metric riêng.

Summary phải có:

1. Mỗi build: source hashes, engine hash, calibration read/batches, input/output timing hashes, flags, inspector signature + conv-weight-type counts, telemetry, capture/native/size statuses.
2. Exact prediction-payload comparison giữa cả ba build bằng payload version của inference-repeat; giữ order/shape/dtype contract, không so raw JSON chứa timestamp. Report metrics exact/delta riêng, đếm khác biệt payload; không dùng AP giống để suy bbox giống.
3. Full Ultralytics AP50/AP50:95/P/R và COCO/XML all/XS/S/M/L/XL AP50/AP50:95: từng build, mean, sample SD (ddof1), min/max/range pp. Không chọn best build; không bootstrap image để giả làm build-sampling CI với n=3.
4. So với Step A repeat_1 để kiểm tra reconstruction, và bảng mô tả range Step A fresh-cache bên cạnh range mới. Historical comparison **không phải thí nghiệm causal fresh-vs-reused random hóa/cùng điều kiện nhiệt**; không tính p-value hay kết luận cache là nguyên nhân duy nhất.

Phân loại đã khóa trước run:

- `replay_exact_observed`: đủ ba build/capture hợp lệ, prediction payload và metrics exact giữa ba build. Có thể khác repeat_1 cũ; ghi riêng. Đây là candidate reproducible build procedure cho đúng ONNX/cache/config/device, không tự chứng minh transferable sang policy/head/model khác.
- `replay_variation_observed`: build/capture hợp lệ nhưng payload hoặc metrics khác. Báo toàn bộ spread; không tăng repeats, đổi cache hoặc tuning đến khi giống. Cũng là task hoàn tất có kết quả, không phải lỗi cần chạy mãi.
- `incomplete_or_invalid`: input/cache identity sai, build/calibration/capture/verification lỗi hoặc điều kiện thực thi không đúng. Giữ partial/log; không tổng hợp như đủ n=3 và không overwrite.

Engine hashes/inspector signatures khác nhau là evidence bổ sung, không tự phủ định `replay_exact_observed`. Nếu exact payload nhưng native/COCO metrics khác thì kiểm tra aggregation/evaluator, không gán cho TensorRT. Nếu phát hiện workload nền trong build, ghi qualification/invalidity theo protocol, không silently loại một repeat khỏi bảng.

### 5. Server/tài nguyên — thực dụng, không mở vòng guard mới

Task reuse cache nguồn gắn với GPU UUID `GPU-9850d121-55dc-e752-ffaa-df19e7585eb4`, RTX8000, driver595.71.05 và environment Step A. Ba build này chạy cùng GPU đó; không chia mỗi repeat sang một server. Việc này **không cấm nhiều server**: CPU/tests/artifact review chạy nơi khác, các experiment độc lập tương lai chia server; muốn chuyển cache study sang GPU khác phải lập provenance/cache nguồn mới trước, không bypass mismatch.

Đây là builder study, khác task inference concurrent trước. Chọn khoảng không có training/build/benchmark cạnh tranh trên GPU đó; không cần trống hoàn toàn VRAM hoặc kill desktop. Giữ exception desktop được operator xác nhận theo nvidia-smi, không đòi `/proc/exe` hoặc quyền admin. Không hard-code PID lịch sử. Luna phối hợp người dùng phân loại workload và thời điểm chạy; không chỉ lặp thông báo “rule cấm”. Không kill/pause/thay priority job khác, không khóa clock/power.

Reuse snapshots trước/sau build/capture, ghi temperature/clock/power/process/UUID và sampled-telemetry limitation. Không mở thêm thermal-control experiment hoặc vòng chờ nguội vô hạn; không tuyên bố thermal-matched hoặc GPU-isolated. Nếu người dùng yêu cầu chạy builder cạnh workload thật, giải thích tác động và báo đổi điều kiện study trước, không ngụy trang thành desktop/idle.

Parent orchestration phải CPU-only; CUDA environment probe/build/capture chạy child kết thúc hoàn toàn trước child kế tiếp. Giữ lifecycle fix đã nghiệm thu. Bounded locks/output checks giữ nguyên, không dùng GPUtil hay lượng VRAM như bằng chứng độc quyền GPU.

### 6. Deliverables triển khai cho Luna

- Runner mới đề nghị `scripts/run_uniform_timing_cache_replay.py`; protocol `docs/UNIFORM_TIMING_CACHE_REPLAY_V1.md`; tests tương ứng. Có thể reuse/extract helper nhỏ, không rewrite pipeline. Capture phải nhận study mới bằng provenance dispatch được kiểm tra, **không đổi manifest giả thành `uniform_build_repeat_v1`** hoặc mở arbitrary engine/test paths để vượt allowlist.
- Tests CPU/mock: ba child builds đúng order; mỗi lần input cache giống, không chain output; nguồn không bị ghi; cache mismatch/attach failure không fallback; recalibration forbidden; settings/Sigmoid invariants; engine/capture binding; dev-only; missing/partial output; parent/child lifecycle; payload/metric comparison; regression Step A và inference-repeat behavior. Chạy relevant tests và full suite, ghi kết quả thực, không gọi mock output là server success.
- Output gồm study manifest, ba build manifests + inspector + timing output caches + verbose logs, ba captures/predictions và verifications, summary/comparison. Giữ engines và source weights/ONNX lớn trên server. Lập artifact inventory rõ ngay trong protocol/tests, lưu SHA256; JSON LF và binary cache/log giữ nguyên bytes. Luna hậu kiểm canonical Git blobs, không bulk renormalize artifact lịch sử.
- Manifest ghi source commit/result commit, code commit/diff state, host/GPU/environment/runtime, input/config/evaluator hashes, exact commands, timestamps và actual termination statuses. Các source paths phân biệt server/local; không hard-code Windows path vào command server.
- CLI server chỉ cung cấp sau khi code được Astra review: mỗi lệnh một dòng, foreground, progress BUILD 1/3→3/3 rồi CAPTURE/VERIFY, DONE + summary location. Chưa ước lượng thời gian như số đo chắc chắn; cache replay có thể nhanh hơn fresh build nhưng phải đo thực tế. Không gộp Git push sau failed run bằng pipeline che exit code.

Luna append **L2A-009**: commit, files, test results, contract deviations (nếu có) và lệnh server dự kiến. Push cả A2L-009 này; Astra review implementation một lần trước server run, chỉ yêu cầu sửa lỗi thực sự ảnh hưởng contract. **Không cần một vòng proposal khác; chưa có quyền GPU run trong entry này.**

Sau kết quả, Astra quyết định đã đủ cơ sở dùng build control cho precision-head/calibration comparison hay phải ghi build variance vào thiết kế. Không tự mở B/C, đổi calibration, train hoặc scale15. Nếu ordinary timing replay còn biến thiên, lựa chọn editable-tactic control hoặc repeated-build design sẽ được quyết định từ bằng chứng; không tự triển khai cả hai. Mục tiêu là đóng một câu hỏi kiểm soát phép đo có giới hạn rồi trở lại nghiên cứu chính, không biến paper thành chuỗi audit vô tận.

## A2L-010 — review b5b4a38: sửa lỗi comparison trước server run

Ngày review: 2026-09-14. Astra đã đọc L2A-009, toàn bộ runner 834 dòng, capture dispatch diff, protocol và tám tests mới của commit `b5b4a38e5cbf229b7ed18c581ca04250e5cd969d`. Chạy độc lập `local/measurement_audit_env/Scripts/python.exe -m unittest discover -s tests`: **80 tests OK**. Chưa chạy TensorRT/GPU. Các dòng BUILD/DONE trong test là mock, không phải kết quả server.

**Decision: SERVER RUN HOLD vì lỗi triển khai đã tái hiện; thiết kế A2L-009 không thay đổi.** Không cần proposal hoặc audit nghiên cứu mới. Luna sửa các điểm dưới đây trong một lượt và bổ sung behavioral tests, rồi push L2A-010 để review mở run.

### R1 — lỗi chắc chắn ở comparison sau capture đầu tiên

`evaluate()` đọc `source_capture` từ `capture/capture_report.json` (dòng 588), nhưng truyền nó vào `_build_comparison_row()` (dòng 652); helper gọi `prediction_difference(source_capture, row['predictions'])` (dòng 548). Capture report chứa metrics/provenance, không có records. Object đúng đã được đọc ở biến `source_predictions` nhưng chưa được truyền vào helper.

Astra gọi helper bằng artifact Step A repeat_1 thực, chỉ CPU/read-only, tái hiện:

```text
ValueError: Prediction payload missing fields: ['capture_mode', 'iou_thresholds', 'records', 'coordinate_contract']
```

Vì tất cả builds chạy trước evaluate, code hiện tại có thể tốn ba lượt build rồi mới lỗi ở comparison đầu tiên. Tách rõ tham số `source_report` và `source_predictions`; metrics lấy report, payload comparison lấy prediction object. Không bổ sung records giả vào report, bỏ comparison hoặc bắt exception rồi coi pass.

Test bắt buộc: chạy thực helper với đúng hai schema; CPU-mocked **toàn bộ evaluate 3 repeats** đến ghi summary/comparison/execution manifests. Mock child GPU thôi, không mock mất helper/aggregation cần kiểm tra. Bao phủ exact và changed-payload cases. Test orchestration hiện tại mock nguyên `evaluate()` nên không phát hiện lỗi này.

### R2 — callback cache cần đúng contract, tests hiện chưa chạy builder path

Ở dòng 373/418, code yêu cầu `read_calls == 1`. A2L-009 chỉ khóa việc đọc cùng bytes và zero batches, không khóa số callback đúng một. Sửa thành có ít nhất một lần đọc, mọi lần trả đúng input, giữ count thực; `batch_calls == 0` vẫn bắt buộc. Nếu muốn giữ exactly-one thì phải có căn cứ API cụ thể, không lấy số lần quan sát Step A làm bảo đảm.

`write_calibration_cache()` hiện chỉ lưu lần write đầu và kiểm tra biến `output` cuối cùng. Cần kiểm tra **mỗi** callback write với input, không để một write khác bytes rồi write sau đúng bytes che mất vi phạm. Ghi evidence vi phạm ngay; dừng có lỗi, giữ partial, không thay calibration table nguồn. Nếu exception callback bị binding xử lý, post-build vẫn phải phát hiện bằng violation flag/count. Không yêu cầu TensorRT nhất thiết phải write cache khi đã đọc cache hợp lệ.

Thêm mocked builder/calibrator tests thực thi các nhánh: read một/nhiều lần cùng bytes + zero batches đạt; không read fail; yêu cầu batch fail; bất kỳ write khác bytes fail; `set_timing_cache` trả false fail và không fallback; mỗi build nhận input nguồn chứ không output trước. Tests hiện tại chỉ kiểm tra constants/path/command/main với mocked subprocess, chưa kiểm chứng các nhánh này. Không cần GPU để test contract này.

### R3 — hoàn tất source/phase validation đã có trong thiết kế

- `validate_source_contract()` hiện xác nhận weight hash từ manifest nhưng không hash `best.pt` thực tế. A2L-009 yêu cầu direct server hash: kiểm tra đúng frozen file `results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt` trước output/build và ghi path + measured hash. Không load/train/export model để làm việc này; thêm mismatch test.
- `--phase build` có thể chạy khi output chưa tồn tại: điều kiện dòng 743 chỉ chặn trường hợp output đã tồn tại mà thiếu manifest, sau đó `build()` tự tạo cây output. Require study manifest hợp lệ cho build child; không cho standalone build vượt bước prepare/preflight đã khóa. Dùng GPU identity helper trên snapshot build trước khi build để giữ đúng UUID/driver cả với phase riêng; không thêm yêu cầu admin hay rule desktop mới.
- `_validate_replay_build()`/capture dispatch nên dùng cùng kiểm tra contract của study mới: effective settings/flags/Sigmoid, timing/calibration input hashes, engine hash, study-manifest binding và completed build. Hiện capture dispatch chưa kiểm timing/settings; không mở rộng arbitrary engine path hoặc giả manifest Step A. Test mutation các trường đã khóa bị từ chối trước capture. Không cần framework provenance tổng quát.

### Hoàn thiện nhỏ cùng lượt sửa

`classify_replay()` hiện trả `replay_exact_observed` cho một record khi không truyền invalid_reasons; Astra đã tái hiện bằng một record tổng hợp. Validate đúng ba unique repeat IDs trước phân loại, thiếu/trùng trả incomplete/invalid. Main hiện lặp ba nên đây chưa phải lỗi server đã quan sát, nhưng summary helper không nên công bố n=3 từ input thiếu.

Khi viết final classification, đọc cả build-before/after guards, không chỉ capture guards; không phụ thuộc hoàn toàn caller đã kiểm trước đó. Cache coverage nếu chưa xác định thì ghi rõ `unknown` như A2L-009. Các bổ sung này không thay số build/capture hoặc numerical design.

### Phần đã đạt và cách bàn giao

Hướng triển khai đúng: ordinary cache reuse, source cache riêng từng process, không editable flag/precision mới, cache-only calibration, separate timing-study capture dispatch, ba build rồi ba captures và giữ source artifacts. Không yêu cầu rewrite 834 dòng hay tạo nghiên cứu khác. Chỉ sửa lỗi/contract trên, chạy targeted + full suite, ghi L2A-010 với tests mới thực sự bao phủ đường dữ liệu/build callbacks.

Luna commit/push code/tests/protocol cần cập nhật và nguyên entry A2L-010 này. Sau khi reviewer xác nhận sửa đạt, người dùng mới chạy foreground. **Hiện chưa cần pull để chạy server, không có artifact GPU mới được yêu cầu trong lượt sửa này.**

## A2L-011 — review eb7b452: mở chạy timing-cache replay trên server

Ngày review: 2026-09-14. Đã đọc L2A-010 và diff commit `eb7b45272328d3d697cf0770096bb1d48765d170`, đối chiếu R1/R2/R3 với A2L-009/010. **Decision: CODE REVIEW ACCEPTED; OPERATOR SERVER RUN AUTHORIZED cho đúng Uniform timing-cache replay v1.** Không cần proposal hoặc vòng xác nhận tài liệu khác trước khi chạy.

### Bằng chứng reviewer kiểm tra độc lập

- Full suite local: **86 tests OK**; `py_compile` ba file runner/capture/tests đạt; `git diff --check` đạt. Tests là CPU/mock, không phải GPU execution.
- Chạy lại `_build_comparison_row()` bằng report/predictions/size Step A repeat_1 thực, đủ 1,636 prediction records: comparison hoàn tất, exact payload/metrics; lỗi thiếu records R1 đã đóng.
- Cache audit chấp nhận nhiều read cùng bytes với zero batches; thử write sai rồi write đúng vẫn bị từ chối. Callback adapter gọi audit này và validate sau build; attach timing cache không có fallback.
- Direct weight hash, prepared study manifest, build UUID/driver check, shared study/build validation trước capture, missing/duplicate-repeat classification và guards cả build/capture đã được bổ sung. Các source caches/settings/precision và giới hạn ba builds + ba captures giữ nguyên.
- Tests evaluate mới thực thi comparison/aggregation/summary cho ba repeats, gồm exact và changed payload; không còn mock mất toàn bộ evaluate như trước. Vẫn chưa kiểm chứng native TensorRT end-to-end trước server run.

### Luna bàn giao ngay cho người dùng

1. Commit/push A2L-011 này và cập nhật trạng thái protocol thành server-run-authorized nếu cần. Không sửa numerical code sau review trừ khi báo diff cụ thể. Không SSH/GPU local.
2. Cung cấp lệnh pull/check commit, snapshot GPU/process hiện tại, rồi lệnh foreground trong `docs/UNIFORM_TIMING_CACHE_REPLAY_V1.md`. Thay placeholder desktop bằng current PID/path nếu có; không mặc định phải tồn tại đúng hai desktop process hoặc lấy PID từ log cũ. Nếu không cần confirmation thì bỏ các option đó. Không mặc định nohup.
3. Chạy default `--phase all` vào output mới `results/measurement_audit_v1/server_uniform_timing_cache_replay_v1/`. Đúng **3 build độc lập tuần tự + 3 dev captures**, mỗi capture tiếp theo CPU verification. Dùng source/cache/GPU/environment đã khóa, không tạo lại ONNX hoặc calibration table.
4. Chọn khoảng không có compute workload cạnh tranh cho builder study; desktop đã xác nhận được giữ, không đòi GPU trống hoàn toàn, không kill/pause/đổi quyền/clock. Luna cùng người dùng kiểm tra và sắp xếp thực tế, không yêu cầu quyền đọc `/proc/exe`. Study này gắn cache với GPU Step A, không chia ba repeat sang ba server khác nhau; các tác vụ CPU/độc lập vẫn có thể dùng server khác.
5. Nếu input/preflight/build/capture lỗi, giữ partial/log và báo cụ thể. Không overwrite, đổi batch/precision/cache, hoặc tự resume/chạy lại cho tới khi kết quả đẹp. Native error/runtime incompatibility nếu xuất hiện vẫn phải xử lý từ log; code review không bảo đảm không có lỗi môi trường.
6. Khi DONE, kiểm tra đủ ba build/capture/verifications và summary; push JSON/cache/log.gz đúng scope theo inventory, không engine/weights/data. Bổ sung optional calibration output cache files nếu callback có ghi. Commit và push tách lệnh để không che failure hoặc mắc ở `nothing to commit`.
7. Luna pull artifact, hậu kiểm canonical Git blob hashes, cùng source/cache input, zero batches, flags/inspector, telemetry, same-pass/native matching và payload/metrics/range n=3. Ghi **L2A-011** với result commit, inventory, kết quả exact/variation/invalid và mọi giới hạn. Không suy timing-cache output hash thành full tactic lock.

**Mốc tiếp theo là kết quả server của task này, không phải thêm review code nếu chỉ publish docs.** `replay_variation_observed` cũng là kết quả hợp lệ nếu integrity đạt; không tăng số lượt chỉ vì chưa exact. Sau report dừng để Astra quyết định cách kiểm soát build cho nghiên cứu tiếp theo. Chưa mở B/C, editable tactics, calibration mới, retrain, 15-model matrix hoặc benchmark latency/energy. Không chọn best engine hoặc tuyên bố paper-ready từ study này.

Astra chỉ ghi quyết định review; Luna phụ trách commit/push như workflow đã thống nhất.

## A2L-013 — giao protocol precision-head ablation YOLO11n (code review trước server)

Ngày chốt: 2026-09-14, sau khi A2L-012 nghiệm thu `replay_exact_observed`. Đây là bước quay lại câu hỏi khoa học chính: **suy giảm INT8 nằm ở nhánh regression (bbox), nhánh classification, hay ở phần còn lại của graph?** Chỉ thay đổi constraint precision ở detection head; không retrain, không đổi calibration images, không dùng official test.

**Luna được giao triển khai code/tests local và protocol; chưa được chạy GPU/server.** Sau L2A-013, Astra review implementation một lần rồi mới cấp quyền chạy.

### 1. Arms và ranh giới graph khóa trước

Từ YOLO11 v8.4.102, detection head là `/model.23`; `cv2` là box-regression branch, `cv3` là classification branch, còn `dfl`/decode và output Sigmoid giữ theo baseline. Node selection phải dựa trên **network layer names thực tế sau ONNX parse**, không dựa tên file hoặc inspector text. Prefix chính xác:

- `bbox_fp32`: mọi convolution layer có tên bắt đầu `/model.23/cv2.`;
- `classification_fp32`: mọi convolution layer có tên bắt đầu `/model.23/cv3.`;
- `both_fp32`: hợp của hai tập trên, là control dương/upper-bound chẩn đoán;
- `baseline_int8`: không thêm constraint cv2/cv3, chỉ giữ 77 Sigmoid FP32 + output constraints của baseline Step A.

Mỗi layer được chọn phải đặt precision **FP32** và mọi output của layer thành FP32, với `OBEY_PRECISION_CONSTRAINTS`; không bật FP16/TF32. Không constrain `/model.23/dfl`, `/model.23/Sigmoid`, backbone/neck hoặc decode ngoài các node baseline đã bảo vệ. Nếu prefix không match, match layer ngoài prefix, hoặc danh sách layer của bốn arm không được ghi/không ổn định, dừng trước build. Manifest phải lưu full matched names, count theo arm, layer types và before/after constraint state; không chỉ lưu count.

Lý do chọn branch này: mã nguồn head định nghĩa `cv2` cho 4×reg_max box outputs và `cv3` cho class scores; TensorRT cho phép constraint layer precision/output, nhưng builder vẫn có thể chèn reformat và chọn implementation phù hợp. Do đó kết quả sẽ là **diagnostic intervention**, không phải chứng minh mọi phép tính của branch chạy FP32 hay quy toàn bộ AP delta cho branch nếu graph còn fused/unsupported.

### 2. Thiết kế build để không lẫn build variance

- Study ID: `yolo11n_precision_head_ablation_v1`.
- Output mới: `results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1/`.
- Frozen YOLO11n weights, ONNX, Uniform seed42 calibration cache và dev split giữ nguyên các hash đã khóa trong A2L-009/012. Không đọc official positive/negative test.
- Bốn arm × **ba build độc lập tuần tự** (12 builds), sau đó capture dev **một lần cho mỗi build** (12 captures) và CPU verification. Không chọn build theo AP; summary arm dùng mean/sample SD/min/max/range của cả ba.
- Mỗi arm nhận **bản sao độc lập của cùng baseline Step-A timing-cache input** `server_uniform_build_repeat_v1/repeat_1/timing.cache`, hash `4c765a0224845e9ddc537253878c696e56c11045369aef224cff6cc0c4178f38`; không chain output giữa các build/arm. Đây là cách giữ timing bytes chung để so arm; coverage cache là `unknown`, `set_timing_cache(..., ignore_mismatch=False)` false phải dừng, không fallback sang empty.
- Nếu TensorRT từ chối timing cache chung vì precision constraints, giữ partial và báo `timing_cache_incompatible`; không tự tạo cache mới rồi trộn vào arm này. Việc tạo arm-specific timing cache sẽ là protocol follow-up, không tự mở trong runner.
- Calibration cache luôn là cùng bytes `31e9d0b3…e01a502`; calibrator cache-only, mọi read phải đúng bytes, `get_batch()` đúng 0, write callback nếu có phải đúng bytes. Không dùng Ncal khác, policy khác, hoặc output calibration mới.
- Builder settings giữ Step A: workspace4GiB, optimization level3, avg timing iterations1, INT8 on, FP16/TF32 off, detailed inspector, OBEY; chỉ thay precision constraints theo arm. Ghi effective flags/settings và matched layer list.
- Build child/process lifecycle CPU parent→short probe→one child per build; GPU lock, UUID/name/driver/environment checks và process guard giữ nguyên. Không cần kill desktop; không chạy cạnh compute workload chưa được operator xác nhận. Không khóa clock/power hoặc chờ thermal vô hạn.

Lưu ý phương pháp: chọn common baseline timing cache **sau khi biết** Step A/A2L-012 là một feasibility intervention, không phải preregistration trước mọi kết quả. Ghi rõ trong manifest/paper; không tuyên bố cache này khóa tactic hoặc làm bốn arm perfectly paired ở mọi TensorRT layer. Nếu cache coverage không đo được, giữ `unknown`.

### 3. Capture/evaluation và endpoints

Reuse same-pass validator, native matching và COCO/XML size diagnostic đã kiểm chứng: dev 1,636 images/2,706 GT, imgsz640, batch1, workers0, conf0.001, NMS IoU0.7, max_det300, rectFalse. Không official test, không threshold tuning.

Mỗi build phải lưu:

1. capture report + prediction payload và native/size verification;
2. full Ultralytics AP50/AP50–95/precision/recall;
3. COCO/XML all và XS/S/M/L/XL AP50/AP50–95;
4. inspector raw/signature, matched precision constraints, calibration/timing evidence, GPU snapshots;
5. exact payload/metrics comparison với các build cùng arm và baseline timing-replay reference (không so raw JSON timestamp).

`comparison_summary.json` phải báo cho mỗi arm và mỗi size:

- mean, sample SD (ddof1), min, max, range pp của ba builds;
- delta so với **baseline arm mean/reference**: `AP_arm − AP_baseline`, giữ riêng Ultralytics và COCO/XML;
- `bbox_delta`/`classification_delta`/`both_delta` theo full, XS và S tối thiểu; không gọi upper-bound `both_fp32` là “ground truth”;
- inspector counts và matched names, nhưng không diễn giải counts thành tỷ lệ FLOPs hoặc proof arithmetic precision;
- invalid/review flags nếu cache mismatch, workload, telemetry thiếu, node selection sai, native/size fail hoặc payload khác.

Để tránh p-hacking, quyết định trước:

- `diagnostic_branch_sensitive`: một arm FP32 có full/XS/S delta so baseline cùng arm ngoài sai số build và cải thiện nhất quán ở ít nhất hai endpoints (full + XS hoặc S). Đây chỉ là descriptive evidence, không đặt ngưỡng mới sau khi xem kết quả.
- `no_branch_signal`: các arm không khác baseline ngoài build variability hoặc `both_fp32` không cải thiện; không dựng câu chuyện branch.
- `incomplete_or_invalid`: thiếu một build/capture, common timing cache không attach, calibration violation, node list mismatch hoặc guard fail. Giữ partial, không tổng hợp.

Không dùng “branch_sensitive” để chọn policy/model tốt nhất; cần xem raw arm table và uncertainty. Với n=3/build arm, không dùng p-value hoặc CI bootstrap ảnh để giả làm build CI. Bootstrap image-level nếu cần sẽ là phân tích phụ sau khi arm integrity pass, không thay variability n=3.

### 4. Deliverables/tests Luna phải hoàn tất trước mở server

- Runner đề nghị `scripts/run_yolo11n_precision_head_ablation.py`; protocol `docs/YOLO11N_PRECISION_HEAD_ABLATION_V1.md`. Có thể trích helper precision selection nhưng không sửa evaluator/payload global ngoài dispatch study mới.
- Tests CPU/mock phải thực thi: exact prefix matching và reject missing/extra nodes; all outputs FP32 + OBEY; baseline constraints unchanged; four arms/settings/flags; common cache copy/no chain; timing attach false/no fallback; calibration read multi/zero-batch/write violation; full 12-build order trước captures; output overwrite/partial; source weights/ONNX/cache/dev-only/hash; build+capture provenance and effective layer list; exact/variation/incomplete classification. Không chỉ test constants hoặc mock nguyên evaluate.
- Manifest ghi arm, source commits/hashes, frozen weight measured hash/path, common timing/calibration hashes, layer names/counts, requested/effective constraints, builder flags/settings, engine/provenance/inspector hashes, environment/GPU, commands/timestamps, telemetry và limitations. Engine binaries không push.
- Chạy targeted tests + full suite + py_compile; report L2A-013 gồm diff thực, test count và deviations. **Không có lệnh server trong L2A-013**; chỉ sau A2L-014 review code mới cung cấp command.

### 5. Diễn giải và điểm dừng

Nếu một nhánh FP32 phục hồi XS/S nhưng làm latency/engine size tăng, đó là trade-off accuracy–precision để báo cáo; chưa benchmark latency ở task này. Nếu không phục hồi, không kết luận lượng tử hóa không ảnh hưởng branch vì constraints có thể không được thực thi đầy đủ hoặc lỗi nằm ở activation/calibration/tactic khác. Nếu common cache incompatible, đó là thông tin về giới hạn thiết kế, không được âm thầm đổi protocol.

Sau khi Luna push code/tests/protocol và Astra review, người dùng chạy đúng 12 builds + 12 captures trên GPU tương thích Step A. Luna hậu kiểm artifact rồi dừng; Astra mới quyết định có đủ bằng chứng giao precision-head insight cho paper hay cần một protocol follow-up. Chưa mở calibration policy mới, retrain, 15-model matrix, cross-device benchmark hoặc official-test evaluation.

## A2L-012 — nghiệm thu timing-cache replay; kết thúc diagnostic này

Ngày review: 2026-09-14. Đã đọc L2A-011 server addendum ở report commit `652a82aa653cdaf51ea6761ce90c6f4a4a342a14`; đối chiếu độc lập raw Git blobs từ artifact commit `3797075c934ca5f88c0d64b998c38def0deef49f`. Không chạy GPU/TensorRT local, không sửa raw results.

**Decision: ACCEPT — `replay_exact_observed` trong phạm vi đã đo. Task A2L-009/011 hoàn tất; không cần rerun, tăng repeats hoặc triển khai editable timing cache chỉ để bổ sung xác nhận.** Giữ global_g0 review_required: nghiệm thu này không đồng nghĩa toàn bộ measurement/paper đã hoàn tất.

### Bằng chứng Astra kiểm tra

- Đủ 41 file, không có engine binary; JSON parse được và ba gzip logs giải nén được, có DONE BUILD tương ứng, không thấy Traceback/[E]/ERROR qua marker scan. Scan không phải bảo đảm mọi warning vô hại.
- Tính lại 27 liên kết hash trên canonical Git blob bytes: input/output caches, inspector, study-manifest binding, build-manifest→capture provenance, prediction→capture/verification, capture-report→verification. Tất cả khớp; hai cache input cũng khớp source Step A.
- Mỗi build read calibration cache 2 lần, zero batch/write và không violation; flags514, timing attach ignore_mismatchFalse, termination completed.
- Cả ba capture có 1,636 records/2,706 GT, capture/native pass, size completed; engine hash liên kết nhất quán. Astra tính lại prediction-payload hash và so metrics với artifact Step A repeat_1: khớp ở cả ba lượt. Inspector signature tính lại khớp manifest và source.
- 12 snapshots build/capture có telemetry complete, UUID/name/driver đúng source, không blocked/unmatched/external workload tại mẫu quan sát. Engine hashes có ba giá trị khác nhau; output timing cache có một hash chung, khác input như flags đã báo.
- Engine/ONNX binary server-only không được Astra hash lại tại local. Những kết luận về binary identity dựa vào provenance/server capture đã liên kết, không phải kiểm tra binary local.

### Bảng kết quả và diễn giải

| Dev endpoint | AP50 (%) | AP50–95 (%) | Range giữa 3 builds (pp) |
|---|---:|---:|---:|
| Ultralytics full | 95.8457 | 66.0437 | 0 / 0 |
| COCO/XML all | 95.5569 | 66.5430 | 0 / 0 |
| COCO/XML XS | 61.4858 | 20.7226 | 0 / 0 |

Payload equality là bằng chứng mạnh hơn chỉ AP bằng nhau trong sample này. Có thể ghi: **ba rebuild độc lập với cùng ordinary timing-cache input tái tạo đúng prediction/metrics trên dev trong cấu hình nguồn này**. Chưa chứng minh mọi tactic bị khóa, mọi build tương lai deterministic, hoặc quy trình giữ nguyên hiệu lực khi đổi calibration table/precision branch/model/device. Không gán byte difference của engine cho nguyên nhân cụ thể khi chưa phân tích serialization.

Khác output timing-cache hash không làm vô hiệu kết quả; giữ coverage unknown. Đối chiếu range với Step A fresh-cache là descriptive historical comparison, không phải causal experiment đã kiểm soát nhiệt/build order. Không dùng thành công của cache repeat_1 để chọn build có AP tốt cho báo cáo hiệu năng hoặc kết luận ưu thế calibration.

Đính chính L2A-011: câu “Mean/sample SD/range đều 0” không chính xác. Mean là các AP trong bảng; range bằng0. SD COCO/XML L và XL AP50–95 trong JSON là khoảng `1.35974e-16` (AP units), do aggregation rounding dù input metrics exact. Báo “exact predictions/metrics; observed range0” hoặc SD làm tròn0 với chú thích; không sửa raw numbers/historical report để che điều này. Luna ghi correction trong entry tiếp theo, không cần GPU hay sửa evaluator.

### Quyết định nghiên cứu tiếp theo

Đã đủ cơ sở **kết thúc nhánh kiểm tra inference-repeat và ordinary timing-cache replay** cho baseline này. Bước hợp lý tiếp theo là thiết kế ablation precision có đối chứng trên dev: baseline không bảo vệ head mới, bbox-branch protection và classification-branch control. Đây là định hướng để chốt protocol, **chưa cấp quyền implementation/GPU B/C trong entry này**.

Trước khi giao runner mới, Astra cần khóa chính xác các node/ranh giới intervention và cách xử lý build variation cho từng arm. Không được giả định cache/tactic của baseline tái sử dụng đầy đủ sau đổi constraint, hoặc dùng một build/arm rồi quy mọi AP delta cho branch đó. Giữ same frozen weights, Uniform calibration input và dev-only; không dùng official test để chọn nodes/flags. Không mở calibration mới, train lại, 15-model matrix hoặc hardware benchmark từ kết quả này.

Luna chỉ cần commit/push A2L-012 và cập nhật trạng thái task/correction ở L2A-012; không phải chạy thêm để giải quyết một blocker. Sau đó lượt tiếp theo thuộc Astra: chốt protocol ablation có đối chứng và giao task cụ thể, thay vì yêu cầu Luna tự mở nghiên cứu. Có thể bắt đầu viết phần measurement/reproducibility từ bằng chứng đã có, nhưng chưa gọi đây là đóng góp cải thiện INT8 đã được chứng minh.

## A2L-014 — review commit 6ac5dae và mở server precision-head ablation

Ngày review: 2026-09-14. Astra đã pull/đọc commit `6ac5dae2f42245da9864a3499ff4820fbed40d38`, protocol `docs/YOLO11N_PRECISION_HEAD_ABLATION_V1.md`, runner, capture dispatch và test mới. Không chạy TensorRT/GPU local.

**Decision: CODE REVIEW ACCEPTED; SERVER RUN AUTHORIZED cho đúng `yolo11n_precision_head_ablation_v1`.** Đây là authorization có phạm vi hẹp cho 12 build + 12 dev capture của YOLO11n; không mở calibration policy, retraining, official-test evaluation, 15-model matrix, cross-device hay latency/energy benchmark.

### Bằng chứng kiểm tra độc lập

- Local full suite: **104 tests OK**; targeted ablation tests **18 tests OK**; `py_compile` và `git diff --check` đạt. Các kiểm tra TensorRT trong test là mock/CPU, không thay thế server execution.
- Layer selection dựa trên layer names/types sau ONNX parse, với prefix chính xác `/model.23/cv2.` và `/model.23/cv3.`; reject non-convolution/missing/ngoài prefix; `both_fp32` phải đúng union và bbox/classification không overlap. Baseline arm không thêm head layer.
- Mỗi intervention layer được đặt requested precision FP32 và mọi output FP32; OBEY được bật. Manifest lưu matched names/types, before/requested/after/effective state và bảo toàn 77 Sigmoid constraints của Step A. Đây là requested/effective builder evidence, không phải proof toàn bộ branch arithmetic chạy FP32.
- Runner khóa source weights/ONNX, Step-A Uniform calibration cache, Step-A repeat-1 timing-cache input, flags/settings, `ignore_mismatch=False`, cache-only zero-batch, không chain và output path riêng. Build/capture dispatch không nhận arbitrary engine path.
- Orchestration chạy arm-major 4 arm × 3 repeat (12 build) trước khi capture; capture dispatch yêu cầu study path, arm và repeat hợp lệ. Existing/partial output không overwrite hoặc tự resume.
- Tests thực thi đường aggregation/evaluate mock, cache callback read/write/zero-batch, source/hash/manifest binding, workload flags, exact/variation/incomplete classification và 12-build/12-capture order; không chỉ kiểm tra hằng số.

### Điều kiện chạy server

Luna cập nhật protocol thành `server-run-authorized`, push commit của entry L2A-014 và gửi người dùng lệnh foreground. Người dùng chỉ chạy khi đã pull đúng commit, output `results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1/` chưa tồn tại, frozen source/cache hashes khớp và GPU/environment phù hợp Step A. Dùng đúng một server/GPU identity cho toàn bộ 12 builds + 12 captures; không chia repeats giữa server khác nhau.

Trên shared server, workload compute hiện hữu phải được xác định/xác nhận trước snapshot theo policy; không kill, pause, reprioritize, đổi clock/power hoặc bỏ guard. Desktop confirmation chỉ dùng PID/path hiện tại nếu cần, không lấy PID lịch sử. Nếu có workload Python/compute chưa xác nhận, giữ partial/log và dừng; không suy “VRAM còn trống” thành điều kiện đủ. Không dùng `nohup` cho study này.

Lệnh chính sau khi Luna cập nhật protocol (thay confirmation bằng snapshot hiện tại, hoặc bỏ option nếu không có desktop GPU row cần xác nhận):

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/run_yolo11n_precision_head_ablation.py --phase all --out-dir results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1 --device 0
```

Không chạy `--phase evaluate` trên partial nếu một build lỗi; giữ toàn bộ output và log để review. Nếu study hoàn tất, Luna hậu kiểm canonical Git blobs, inventory, source/cache/provenance hashes, layer evidence, calibration/timing guards, telemetry, native/size verification và summary classification; sau đó ghi L2A-014 addendum rồi dừng. Astra sẽ quyết định tiếp theo từ artifact, không chọn arm tốt nhất tự động.

### Cách diễn giải kết quả được phép

Chỉ báo cáo `diagnostic_branch_sensitive`, `no_branch_signal` hoặc `incomplete_or_invalid` theo rule đã khóa trong protocol. Không gọi `both_fp32` là ground truth, không quy mọi AP delta cho branch nếu timing-cache coverage vẫn unknown, và không biến khác biệt engine hash/inspector thành bằng chứng tactic hoặc FLOP ratio. Nếu common timing cache bị TensorRT từ chối, đó là kết quả giới hạn feasibility; không fallback sang cache mới trong study này.

## A2L-015 — nghiệm thu sửa selector fb49587; giao chạy lại attempt2

Ngày review: 2026-09-15. Astra đã đọc L2A-015 và diff commit `fb495870317b51bbe7188fc57319030a2c644fca`. **Decision: ACCEPT bản sửa selector; AUTHORIZE chuẩn bị và operator chạy attempt2 theo các thay đổi cụ thể dưới đây.** Luna triển khai local, kiểm tra và push; người dùng chạy server. Không cần một vòng xin authorization nữa nếu diff chỉ thực hiện entry này và các checks đạt.

### Kết quả review và đính chính

Astra chạy độc lập full suite: **105 tests OK**, gồm 19 tests ablation. `py_compile` runner/capture/tests và `git diff --check fb49587^ fb49587` đạt. Đây là CPU/mock evidence; chưa kiểm chứng native TensorRT trên local.

Selector sửa đúng yêu cầu: convolution có prefix cv2/cv3 được chọn; helper non-convolution cùng namespace được ghi candidate/excluded audit. Regression test dùng đúng tên Sigmoid gây lỗi được nêu trong log. Sigmoid bị loại khỏi tập intervention bổ sung nhưng vẫn được bảo vệ theo baseline Sigmoid constraints. Không thay flags, cache, weights hay evaluator.

Astra đính chính review A2L-014: việc reject mọi non-convolution cùng prefix là giả định sai mà reviewer đã bỏ sót. Test cũ cũng củng cố giả định đó. Không quy sự cố selector cho tài nguyên GPU hoặc operator. Thông tin baseline builds 1–3 hoàn tất và bbox repeat 1 thất bại hiện dựa trên L2A-015; Astra chưa hậu kiểm trực tiếp partial server artifacts.

### Path và phạm vi attempt2 đã chốt

- Scientific study ID giữ `yolo11n_precision_head_ablation_v1`; đây là lần chạy lại sau sửa implementation, không phải phương pháp nghiên cứu v2.
- Output duy nhất cho lượt mới: `results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1_attempt2/`.
- Giữ nguyên `results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1/` và mọi partial/log trong đó. Không xóa, di chuyển, ghi đè, resume hoặc chép ba baseline builds cũ vào attempt2.
- Attempt2 chạy mới cả 4 arms × 3 builds, rồi 12 dev captures/CPU verification theo thứ tự đã khóa. Dùng cả ba builds mỗi arm để tổng hợp. Không thay source Step-A/timing-replay reference paths hoặc scientific settings.

### Luna triển khai ngay trong một lượt

1. Đồng bộ output target trong `run_yolo11n_precision_head_ablation.py`, capture dispatch `capture_cctsdb_validator.py`, tests và protocol sang attempt2. Chỉ đổi destination của ablation mới; không bulk-replace historical paths trong handoff/artifact. Giữ CLI bounded, không mở arbitrary output/engine hoặc fallback về v1.
2. Study manifest bổ sung `execution_attempt: 2`, relative `previous_attempt_path`, lý do `implementation_fix_non_convolution_namespace_selection`, current execution `git_commit` và runner `script_sha256`. Phân biệt current commit với source Step-A code commit; giữ binding study manifest → build → capture. Không sửa manifest lịch sử.
3. Tests phải kiểm tra runner và capture cùng chấp nhận attempt2, từ chối destination v1/ngoài scope, và existing attempt2 không bị overwrite. Chạy regression selector helper hiện có, full suite, py_compile, diff-check. CPU/mock evaluate phải đi hết 12 captures đến summary với destination mới.
4. Cập nhật protocol active status và lệnh foreground sang attempt2; ghi L2A-015 addendum gồm commit, diff, checks, current output path và điều kiện authorization đã đáp ứng. Commit/push entry A2L-015 nguyên văn cùng thay đổi. Không SSH hoặc chạy TensorRT/GPU local.
5. Nếu diff đúng phạm vi trên và checks đạt, Luna bàn giao ngay lệnh pull/check/snapshot/run cho người dùng; không chờ Astra cấp số giao việc mới chỉ để đổi path. Nếu phát hiện cần đổi numerical protocol/cache/GPU policy hoặc native error khác, báo căn cứ cụ thể để review.

### Operator run và hậu kiểm

Sau khi Luna push bản cập nhật, người dùng pull đúng commit; kiểm tra attempt2 chưa tồn tại và snapshot GPU/process hiện tại. Dùng cùng GPU/environment nguồn đã khóa cho lượt controlled build này. Runner hiện chỉ hỗ trợ desktop confirmations, **không có option xác nhận background compute cho ablation**: không được diễn giải xác nhận tên Python là đã mở chạy concurrent. Nếu cần đổi điều kiện concurrent, phải có variant được ghi nhận; entry này chỉ xử lý selector/path. Không yêu cầu GPU trống desktop hoặc quyền đọc /proc/exe, không kill/pause process khác.

Lệnh foreground sau khi destination update đã được push/pull (Luna bổ sung desktop confirmations đúng snapshot hiện tại nếu cần):

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/run_yolo11n_precision_head_ablation.py --phase all --out-dir results/measurement_audit_v1/server_yolo11n_precision_head_ablation_v1_attempt2 --device 0
```

Không chạy lệnh này với commit fb49587 chưa cập nhật destination. Nếu attempt2 có lỗi, giữ partial/log và báo lỗi cụ thể. Nếu hoàn tất, người dùng push JSON/cache/log evidence của attempt2 (không engine/ONNX/weights); Luna hậu kiểm canonical Git blobs và bảng đủ 12 builds/captures, rồi báo Astra để phân tích. Kết quả scientific vẫn chờ review; không tự mở 15-model/calibration mới/official test/hardware benchmark.

Astra chỉ sửa tài liệu giao việc trong lượt này; Luna phụ trách code/tests và commit/push theo workflow đã thống nhất.

## A2L-016 — nghiệm thu attempt2 và giao phân tích paired dev trên artifact

Ngày review: 2026-09-16. Astra đọc L2A-016 ở `822cdd9b7b14d2a475ab5b17b7405831d44478ea`, kiểm tra canonical Git blobs của artifact commit `839acdcb6a09569d1e6e130aa523c38d960dabd5`. **Decision: ACCEPT attempt2 như một precision-head diagnostic hoàn chỉnh trên dev.** Không cần build/capture lại để xác nhận thêm. `diagnostic_branch_sensitive` được chấp nhận là mô tả dữ liệu hiện có; chưa phải bằng chứng tổng quát cho mọi model/calibration/device hoặc paper-ready.

### Kiểm tra độc lập của Astra

- Đủ 159 artifact, tất cả JSON parse được; 12 gzip logs giải nén được, có DONE BUILD đúng arm/repeat, không thấy Traceback/[E]/ERROR qua marker scan. Không engine/ONNX/weights trong tập artifact này.
- 111 liên kết SHA256 đã tính lại từ canonical bytes: runner tại execution commit, source study manifests và 9 liên kết mỗi repeat gồm study/cache/inspector/build/capture/predictions/verification. Khớp toàn bộ. Không direct re-hash engine/ONNX server-only tại local.
- 12 builds completed, cùng frozen weight/ONNX/calibration/timing input hashes và settings/flags514; calibration read >=1, zero batches, zero write, không violations. Cache attach thành công với ignore_mismatch=False; không chain; coverage unknown.
- Constraint audit/union và repeat stability đạt, target counts 0/9/15/24. Inspector signatures tính lại khớp raw inspector; Conv weight type Int8/Float lần lượt 70/15, 61/24, 55/30, 46/39. Đây là metadata evidence, không phải FLOP ratio hoặc proof precision của mọi operation.
- 12 captures đều 1,636 images/2,706 GT, native matching pass và size verification completed. Prediction payload hashes tính lại đúng comparison records; metrics/payload exact giữa ba builds trong mỗi arm. Baseline payload cũng exact với timing-replay reference trước đó. Tính lại full AP means/deltas và kiểm tra range full/size bằng0.
- 48 build/capture snapshots có GPU identity đúng source, telemetry complete, không blocked/unmatched/external workload tại thời điểm quan sát. Wording đúng là không phát hiện competing workload trong snapshots; không khẳng định không có workload giữa các mẫu.
- FP16 reference `server_fp16_capture_v1` và `server_native_size_v1` có 3 prediction/report hash links khớp, cùng captured targets/preprocessing và size convention với attempt2. Không lấy số liệu từ custom-size evaluator cũ để so sánh.

### Kết quả và diễn giải đã chốt

Các giá trị dưới đây là phần trăm AP; full dùng Ultralytics, XS/S dùng COCO/XML diagnostic đã version hóa. FP16 là engine reference lịch sử tương thích, không phải arm mới được rebuild trong attempt2.

| Representation | Full AP50 | Full AP50–95 | XS AP50 | S AP50 |
|---|---:|---:|---:|---:|
| FP16 reference | 97.7750 | 76.3025 | 71.6815 | 97.4321 |
| Baseline INT8 | 95.8457 | 66.0437 | 61.4858 | 95.3083 |
| Bbox FP32 | 96.5501 | 71.8370 | 68.1952 | 95.8354 |
| Classification FP32 | 97.0031 | 70.6718 | 64.3220 | 97.0728 |
| Both FP32 | 97.5986 | 75.7066 | 70.4841 | 97.5379 |

Bbox intervention tăng full AP50–95 5.7933 pp và XS AP50 6.7094 pp so baseline; classification tăng tương ứng 4.6281/2.8362 pp; both tăng 9.6630/8.9983 pp. Both còn thấp hơn FP16 reference 0.5958 pp full AP50–95 và 1.1975 pp XS AP50. Đây là point estimates trên dev, chưa có CI cho contrast mới.

Diễn giải: bảo vệ bbox có tín hiệu phục hồi XS và AP ở IoU chặt hơn mạnh hơn classification-only trong cấu hình này; classification cũng đóng góp, nên chưa quy toàn bộ suy giảm cho bbox. Both có AP cao nhất trong bốn arm hiện tại nhưng chưa được chọn làm deployment policy vì chưa đo chi phí runtime. Không gọi both là upper bound được bảo đảm.

Tất cả 12 engine hashes khác nhau. Baseline/bbox mỗi arm có một timing-output hash và một inspector signature; classification/both mỗi arm có ba timing-output hashes và ba inspector signatures, dù prediction/metrics exact. Ghi rõ chi tiết này bổ sung L2A-016: exact output không đồng nghĩa engine/tactic identity. Mọi output timing cache khác input; giữ coverage unknown. Mean không bằng0; observed within-arm AP range bằng0, sample SD có thể có rounding cỡ machine precision.

### Task tiếp theo cho Luna — phân tích CPU, không cần GPU trống

**AUTHORIZE Luna triển khai và thực hiện paired uncertainty analysis trên các captures đã có.** Đây là task tiếp nối cụ thể; không chỉ publish quyết định rồi chờ một lệnh mới. Output mới: `results/measurement_audit_v1/precision_head_paired_analysis_v1/`. Protocol/analysis note: `docs/PRECISION_HEAD_PAIRED_ANALYSIS_V1.md`.

1. Inputs khóa ở artifact commit trên: bốn arm attempt2, FP16 capture/verification đã nêu. Verify đủ ba repeats exact trước khi dùng repeat_1 làm đại diện tính toán mỗi arm; đây là deduplication của identical payload, không chọn best build. n ảnh là1,636, không nhân thành4,908 vì có ba builds.
2. Tạo runner CPU riêng, tái sử dụng helper `resample_ap`/`coco_size` trong `analyze_dev_quantization.py` và `verify_cctsdb_capture.py` khi phù hợp. Không sửa số liệu nguồn hoặc estimator lịch sử. Đọc/hash canonical Git bytes trên Windows; nếu cần materialize, dùng staging riêng với bytes nguyên vẹn. XML chỉ lấy các dev IDs đã khóa, cùng XML hash/convention; không đọc official-test annotations để tính metric.
3. Khóa trước khi chạy: 1,000 paired image bootstrap draws, seed20260916, sample có hoàn lại từ sorted 1,636 dev image IDs, dùng cùng draw cho mọi representation. Giữ duplicate image occurrences khi accumulate AP; không tính trung bình AP từng ảnh, không bootstrap từng box. Báo percentile95% CI và số valid/undefined draws; giữ nguyên quy tắc class support của helper đã có. CIs exploratory, conditional on fixed captures/dev sample; không phải build/calibration/training CI hoặc xác nhận trên official test.
4. Báo toàn bộ sáu size endpoints all/XS/S/M/L/XL, AP50 và AP50–95 cùng COCO/XML convention. Contrasts cố định: bbox−baseline, classification−baseline, both−baseline, bbox−classification, both−bbox, both−classification, và mỗi arm−FP16 (10 contrasts). Ưu tiên all AP50–95, XS AP50/AP50–95, S AP50/AP50–95 trong narrative, không bỏ contrasts bất lợi hoặc đặt success threshold sau khi thấy CI. Full Ultralytics point estimates lưu riêng; không dán COCO CIs lên Ultralytics points.
5. Tái sử dụng GT-centric localization diagnostic để đếm gained/lost matches ở IoU0.50/0.75/0.90 so baseline theo size, giữ matching convention và confidence settings. Report đây là localization diagnostic có điều kiện trên predictions/scores, không chứng minh lỗi classification và bbox độc lập. Không cần render ảnh hay thêm inference.
6. Deliverables: input manifest/hash/versions, sample plan+hash, bootstrap draws/samples, point/CI tables JSON, localization summary và báo cáo Markdown ngắn. Ghi code commit, seed, paired contrasts, repeat deduplication, estimator ID và mọi limitation. Point estimates phải khớp persisted size reports trước bootstrap; tests cần có duplicate-draw handling, identical-arm delta0, missing/changed input bị phát hiện và đủ10 contrasts. Chạy targeted/full relevant tests và diff-check; Luna commit/push như workflow đã thống nhất.
7. Nếu local có XML và environment phù hợp, Luna được chạy CPU analysis ngay. Nếu thiếu XML hoặc environment cần dùng server, hoàn tất code/tests và push rồi đưa người dùng một lệnh CPU foreground; chạy được trên server khác hoặc cạnh GPU job vì task này không build/inference/benchmark, không áp GPU-idle guard. Ghi đúng nơi thực thi và giới hạn tương thích, không đòi SSH hoặc chuyển engine.

Luna ghi L2A-017 với artifact commit, bảng contrasts/CI, point-estimate reproduction và giới hạn rồi bàn giao Astra. Task được nghiệm thu không có nghĩa toàn bộ paper đã đủ dữ liệu. Bằng chứng này dùng để quyết định thiết kế xác nhận precision/calibration và sau đó main15-model study; chưa tự động mở server experiments mới, retrain, official test hoặc latency/energy benchmark trong entry này.

## A2L-017 — nghiệm thu paired analysis; giao chuẩn bị đo accuracy–latency

Ngày review: 2026-09-16. Reviewed L2A-017 ở report commit `bb3e280731c076da1c55d2cfbedece9d6c40fbb5`, artifact `4846c73ddd2cbb2bd522caa0e1a1eb4598deb1e3`, code `8b5a3420f1eeb460b58198647cbdbe83f1ab1045`. **Decision: ACCEPT paired dev analysis; kết thúc bước phân tích uncertainty này.** Không yêu cầu bootstrap lại hoặc tăng resamples.

### Kiểm tra độc lập

- Đủ9 artifact; 111 liên kết hash output/code/study/13 nhóm input khớp canonical Git blobs. Ba repeat mỗi arm có payload hashes exact; representative repeat1 là deduplication, không chọn theo AP.
- Tạo lại PCG64 seed20260916, ma trận1,000×1,636 và sorted sample indices: exact; image names khớp sorted dev capture. Draw array shape1,000×5×6×2, finite toàn bộ.
- Tính độc lập point deltas và percentile2.5/97.5 từ saved draws: đủ120 ô (10 contrasts×6 sizes×2 metrics), max absolute difference0. Tất cả1,000 valid/0 undefined. Point estimates khớp các persisted size reports, reproduction error0.
- Tính lại180 gained/lost counts từ localization_per_gt ở 5 representations×6 sizes×3 IoU×2 transitions: khớp. Targeted paired tests5/5 đạt.
- Reviewer local NumPy2.4.2, server2.4.4: tái lập indices/CI exact trong kiểm tra này. Không tuyên bố đã chạy lại toàn bộ1,000 COCO/XML evaluations ở local; kiểm tra này tái tính CI từ saved draws và hậu kiểm point/input bindings, đọc implementation duplicate handling. XML và TensorRT binaries không được chạy tại local.

### Kết luận khoa học được phép

Tất cả số liệu dưới đây là COCO/XML diagnostic, delta theo điểm phần trăm, với exploratory paired percentile95% CI có điều kiện trên fixed captures/dev sample; không phải CI cho build/calibration/training variance hoặc kiểm định nhiều contrasts đã hiệu chỉnh.

| Contrast | All AP50–95 | XS AP50 | XS AP50–95 |
|---|---:|---:|---:|
| Bbox − baseline | +5.4263 [4.9118,5.8991] | +6.7094 [2.1525,11.2544] | +7.3861 [4.9635,10.1734] |
| Classification − baseline | +3.6271 [2.9389,4.1490] | +2.8362 [-0.3497,5.3691] | +1.3778 [0.1425,2.5680] |
| Both − baseline | +8.5067 [7.7104,8.9999] | +8.9983 [3.1418,15.1011] | +8.9474 [5.7214,12.6664] |

Bbox/both có tín hiệu phục hồi XS và localization đủ rõ để chuyển sang đánh giá chi phí chạy. Classification-only cũng cải thiện all AP50–95; chưa có bằng chứng XS AP50 tăng nhất quán vì CI chứa0. Bbox−classification XS AP50 CI cũng chứa0, nên không viết bbox chắc chắn tốt hơn classification ở mọi small-object endpoint. Both−bbox XS AP50/AP50–95 CIs chứa0; chưa chứng minh bảo vệ cả hai luôn tốt hơn bbox-only cho XS.

Both−FP16 all AP50–95 = −0.6049 pp [−0.9361,−0.3670]; vẫn còn gap theo metric này. XS CIs so FP16 chứa0 không phải chứng minh equivalence/non-inferiority, vì chưa khóa equivalence margin. S AP50 của bbox−baseline cũng chưa chắc chắn. Giữ cả10 contrasts trong supplementary tables.

Localization all IoU75: bbox gained141/lost37, classification42/31, both143/43; IoU90 tương ứng461/115,185/164,472/111. Các chuyển đổi này hỗ trợ giả thuyết độ nhạy localization; chịu ảnh hưởng matching/scores/NMS nên không phải causal decomposition. Không dùng riêng số match IoU50 để suy toàn bộ AP.

Các con số all COCO/XML trên khác full Ultralytics (ví dụ both−baseline +8.5067 vs +9.6630 pp) vì estimator khác, không phải kết quả mâu thuẫn. Giữ nhãn evaluator tách biệt. Ba build exact trên dev và CI conditional chưa chứng minh chuyển được sang model/calibration/device khác. Kết quả VCSC trước đó vẫn là kết quả riêng, không bị thay bằng kết quả precision-head này.

### Bước tiếp theo đã giao Luna: implementation đo latency các engine có sẵn

**AUTHORIZE local implementation/tests và protocol cho cost study; chưa chạy GPU/server trong entry này.** Câu hỏi: phần accuracy phục hồi có giữ được lợi ích latency so với FP16 không? Cần trả lời trước khi nhân rộng chính sách FP32 head. Đây là benchmark chẩn đoán trên RTX8000 hiện tại, không phải cross-device deployment matrix.

Study ID `yolo11n_precision_head_latency_v1`; output mới `results/measurement_audit_v1/server_yolo11n_precision_head_latency_v1/`; protocol `docs/YOLO11N_PRECISION_HEAD_LATENCY_V1.md`. Runner đề nghị `scripts/run_precision_head_latency.py`. Giữ frozen engines; không export/build mới.

1. Inputs: đủ12 engines của attempt2 (4 arms×3 builds) và1 FP16 engine có hash trong `server_fp16_capture_v1/capture_report.json`: tổng13 binaries. Khóa metadata ở attempt2 input commit839acdc và FP16 reference đã kiểm. Direct server binary hashes phải khớp manifest trước deserialization, source/dev provenance được giữ. Không loại builds chỉ vì AP exact: classification/both có inspector khác nhau, latency có thể khác. FP16 có1 build, báo rõ khác với3 builds/INT8 arm.
2. Metric chính: synchronous batch1 pipeline wall-clock latency từ decoded CPU image vào `model.predict` đến khi kết quả hoàn tất, gồm preprocessing/H2D/inference/postprocessing/NMS; loại disk decode, model load, allocation ban đầu và warmup. Dùng GPU synchronize trước timer và sau predict, timer monotonic độ phân giải cao. Không gọi đây là pure TensorRT kernel time. Không dùng thông số speed của validator hoặc old benchmark JSON để thay lượt đo mới.
3. Giữ Ultralytics8.4.102, TensorRT10.16.1.11, environment/GPU của attempt2. Runtime imgsz640, batch1, rectFalse, conf0.001, iou0.7, max_det300, task detect, verboseFalse. Conf này giữ workload postprocessing gần accuracy capture; kết quả được ghi rõ không phải production benchmark conf0.25. Kiểm tra input640×640 cả ảnh có aspect ratio khác nhau. Tái sử dụng framework predict để nhận engine metadata header; không tự deserialize cả Ultralytics header như TensorRT plan.
4. Image pool cố định256 ảnh từ dev IDs đã khóa, chọn PCG64 seed20260916 không hoàn lại trên sorted IDs rồi sort selected IDs; cùng danh sách/nội dung ảnh cho mọi engine/session, chỉ đọc images, không annotations/test. Lưu image IDs, file hashes và input sequence. Mỗi session preload/decode trước warmup;200 warmup calls rồi1,000 measured calls theo cyclic pool order bắt đầu index0. Các số này khóa trước kết quả, không adaptive stopping theo độ đẹp latency.
5. Đo3 rounds với mỗi engine xuất hiện1 lần/round,39 sessions. Canonical list: FP16, baseline1,bbox1,classification1,both1, baseline2,bbox2,classification2,both2, baseline3,bbox3,classification3,both3. Round1 theo list, round2 rotate-left4, round3 rotate-left8; persist schedule. Thứ tự này giảm confounding vị trí, không tuyên bố Latin-square hoàn chỉnh. Mỗi session một child process mới để release model/resources; parent CPU orchestration. Không chạy nhiều engine cùng GPU đồng thời.
6. Lưu1,000 raw per-call latency samples/session, mean/median/p95/p99/min/max (percentile method linear ghi rõ), serial FPS=1000/mean_ms và engine bytes. Tách variability giữa3 measurement rounds khỏi3 builds; không coi3,000 inference timings là3,000 independent builds. Báo per-engine/per-round và arm summaries gồm mọi builds, không chọn fastest run. Accuracy points/CI được link vào bảng với đúng evaluator. Torch allocated peak nếu thu được chỉ là Torch allocator, không gọi total TensorRT memory; snapshot memory/power/thermal chỉ là observations, không suy peak tuyệt đối/average power/energy từ hai điểm. Power/energy chưa phải required endpoint của task này.
7. Dùng GPU lock/identity và process guard hiện hành cho latency session, desktop confirmation PID/path hiện tại được phép. Không cần GPU không còn desktop. Workload compute cạnh tranh làm không đủ điều kiện so controlled latency; ghi partial/log và báo thực tế, không kill/pause/chỉnh quyền/clock. Ghi thermal/power/clock snapshots trước/sau session; không chờ cooldown vô hạn. Một server/GPU cho bảng13 engines này; các CPU task khác có thể dùng server khác.
8. Audit `benchmark_tensorrt_engine.py` hiện có trước reuse: script này có warmup và sync nhưng thiếu raw samples,p99,round/build hierarchy, exact engine bindings và rectFalse/max_det lock. Sửa qua wrapper/helper có tests hoặc runner riêng; không âm thầm dùng defaults cũ. Tests CPU/mock bao phủ13 engine hashes/missing inputs,39 sessions/schedule, lifecycle, warmup excluded, sync/timer order, percentile/raw aggregation, same image stream, runtime shape/options, partial/no-overwrite và complete summary path. Tests không khẳng định GPU benchmark end-to-end.

Luna hoàn tất code/tests/protocol trong một lượt, ghi L2A-018 và push cả entry này; Astra review implementation rồi mới đưa lệnh operator server. Nếu binary bị thiếu ở server, báo engine/path cụ thể, không tự rebuild vì sẽ thay đối tượng đang đo. Khi có latency, reviewer mới cân nhắc accuracy–latency trade-off và thiết kế calibration/architecture confirmation để quay lại main15-model scope. Không mở retraining, external datasets, official test hoặc edge-device matrix trong task implementation này.

Nguồn tham khảo cho ranh giới phép đo: [NVIDIA TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html), phần Benchmarking mô tả wall-clock/CUDA events và kiểm soát môi trường. Đây là nguyên tắc tham khảo; implementation phải tương thích environment10.16 đã khóa, không nâng runtime theo trang latest.

## A2L-018 — review latency implementation; sửa ba lỗ hổng trước server

Ngày 2026-09-16. Reviewed L2A-018 và commit `f62e84fa20e296a749221d6af067238ed92ab576`. **Decision: CHANGES REQUESTED; authorize local fixes/tests only.** Chưa chạy latency trên server. Không thay đổi nghiên cứu, số engine, số round, sample plan hoặc endpoints đã khóa ở A2L-017.

### Reviewer verification

- Working tree sạch lúc bắt đầu; local HEAD đúng commit được gửi. Astra tự chạy lại targeted tests **11/11** và full regression **122/122**, đều pass bằng `local/measurement_audit_env/Scripts/python.exe`. Python mặc định của MSYS thiếu NumPy; không cài/nâng package, dùng environment local sẵn có. Không load TensorRT/engine, không chạy GPU.
- Đúng: lịch 13 engine × 3 round, rotations 0/4/8; 200 warmup và 1.000 measured calls; sync trước timer/sau predict; raw samples và linear percentiles; engine bytes được kiểm trước load; parent gọi child tuần tự; không export/build.
- Mock diagnostic độc lập trên artifact thực (chỉ mock sự hiện diện/hash binary và filesystem ảnh do local không có engine/dataset) cho thấy `validate_inputs` chấp nhận cả input nguyên bản và input đã đổi một dev image ID cùng bbox accuracy point table. `_validate_session` chấp nhận một raw sample với metadata `measured_calls=1000`, thậm chí không có GPU evidence. Đây là lỗ hổng validator/tests, **không phải bằng chứng artifact hiện tại bị hỏng**.

### R1 — đường dẫn ảnh phải khớp dataset đã capture

`main` đang default `../nighttime-tsd/data/processed/cctsdb2021_clean/dev/images`, nhưng accepted `server_fp16_capture_v1/dev_absolute.yaml` chỉ tới `/home/ubuntu/Dung_TDTU/nighttime-tsd-new/data/processed/cctsdb2021_clean/dev` + `val: images`. Default hiện tại chỉ tới repo sibling; có thể fail missing hoặc đọc bản processed khác cùng filename. Reviewer không biết sibling server có tồn tại hay không và không khẳng định nó đã thiếu.

- Default resolve tương đối với repo thành `data/processed/cctsdb2021_clean/dev/images`, hoặc resolve từ accepted capture data reference. Không phụ thuộc shell cwd.
- Nếu giữ explicit `--images-dir` cho relocation, ghi declared/resolved path và kiểm đủ selected IDs, decoded dimensions theo captured `orig_shape`, hashes của bytes thực dùng. Không diễn đạt hash mới của ảnh là bằng chứng content-identical với historical capture nếu capture không lưu image-byte hashes. Không tạo dataset mới hoặc đọc test.
- Test default path dưới cwd khác repo, missing image, sai dimensions và relocation (nếu hỗ trợ).

### R2 — bind inputs vào artifact đã nghiệm thu, không chỉ ghi hash hiện tại

`validate_inputs` và `_load_accuracy_links` đang đọc JSON working tree rồi hash/log chính file đang đọc; chưa pin accepted blobs. `validator_predictions.json` không được check against `capture_report.predictions_sha256`; accuracy `point_estimates/contrast_ci` không đối chiếu `analysis_summary.outputs`. Một số cache fields bị bỏ qua vì study lưu `calibration_cache_input`/`timing_cache_input` dạng nested nhưng loop tìm flat `*_input_sha256`.

- Pin attempt2 source artifact commit `839acdcb6a09569d1e6e130aa523c38d960dabd5`, paired-analysis artifact commit `4846c73ddd2cbb2bd522caa0e1a1eb4598deb1e3`. Read exact canonical Git blobs ở commit này cho consumed input metadata (bao gồm historical FP16 files được lưu trong tree), hoặc so working-tree data với các canonical records trước dùng. Không dùng HEAD tùy ý làm expected baseline. Ghi path/commit/blob SHA256 rõ ràng; giữ CRLF caveat.
- Kiểm prediction/report/verification links và source weight/model bindings; giữ đủ 13 fixed engine hashes từ accepted records. Bind paired points/CIs với accepted summary output hashes và kiểm đúng model/contrast mapping. Compare nested calibration/timing cache input hashes với build records nếu tiếp tục kiểm chain này. Không yêu cầu binary ONNX hoặc chạy cache/build mới.
- Binary trên server vẫn phải hash trực tiếp, khớp accepted engine hash trước load. Image hashes của phiên latency vẫn là direct filesystem hashes.
- Negative tests: thay prediction ID, accuracy value/CI, source/build binding, cache hash và engine hash phải reject hoặc tuyệt đối không sử dụng mutated checkout do đọc pinned blob. Test unchanged accepted fixture vẫn pass. Không bắt Git working tree toàn repo sạch vì có thể có artifact không liên quan của người dùng.

### R3 — nghiệm thu session phải kiểm dữ liệu thực, không tin metadata tự khai

Ở `_validate_session`, `latency_statistics` chỉ yêu cầu vector nonempty; nó không bắt `len(raw_latency_ms)==1000`. Test aggregation hiện dùng 2 samples/round nhưng output key vẫn `pooled_calls_3000`, cho thấy thiếu test cardinality.

- Require đúng 1.000 positive finite raw samples/session, stats `n_calls=1000`, đúng engine/build/round/path/hash, pool manifest hash + sequence binding và runtime options. Require và independently validate cả before/after GPU identity/process guard evidence đã persisted, thay vì mặc định summary telemetry clean. Same sampled telemetry limitation vẫn giữ nguyên.
- Check 39 unique complete sessions; mỗi engine đúng 3 rounds, 3.000 calls; FP16 arm 3.000 và mỗi INT8 arm 9.000 calls. Không thay sample counts hoặc coi calls là independent builds. Thiếu artifact/session, duplicate round, wrong count hoặc telemetry mismatch không được xuất completed summary.
- Bind child pool file với hash trong parent run manifest; validate session output location/round/engine against schedule và existing output trước GPU work. Thêm observed preprocessed input-shape check `(1,3,640,640)` cho các aspect ratios có trong fixed pool **ngoài measured timer**, lưu bằng chứng; chỉ assert `rect=False` trong constant test chưa đủ xác minh pipeline. Không thêm instrumentation nặng vào 1.000 timed calls.

### Completion tests và bàn giao

Luna bổ sung CPU/mock integration test chạy parent từ accepted-shaped fixtures qua **đủ 39 child sessions đến latency_summary/report**, kiểm các counts/hash links/round hierarchy. Test failed child, missing session, malformed raw/telemetry và partial output được giữ, không completed summary; kiểm child thực sự được wait trước session sau và GPU-touching environment probe không chạy trong parent. Test `run_child` với mocked runtime/model/image loader phải xác nhận options và observed shape, không chỉ test argv builder. Không tuyên bố CPU mocks chứng minh TensorRT end-to-end.

Không cần thêm features, metric mới, statistical threshold, benchmark dataset, hoặc thay desktop/shared-lab policy. Update protocol cho path/binding/validation; ghi L2A-019 với tests và sửa từng R1/R2/R3. Push bằng NADUNGVN cả entry này, không commit scratch `local/astra_latency_review.py`. Astra sẽ review corrected implementation để quyết định lệnh server foreground. Không cần người dùng chạy lại Step A/ablation/bootstrap; toàn bộ kết quả đã nghiệm thu vẫn giữ nguyên.

## A2L-019 — review eb30a08: hai lỗi schema thực cần sửa, không đổi protocol

Ngày 2026-09-16. Reviewed L2A-019 và commit `eb30a08f4b61fdf653532c362f531897560ec4f9`. **Decision: CHANGES REQUESTED, local sửa hai lỗi tương thích dưới đây; chưa cấp server run.** Đây không phải yêu cầu thêm nghiên cứu hay siết resource guard. Không chạy lại build/ablation/bootstrap; không đổi 13 engine, 39 sessions, runtime, data hoặc telemetry policy.

### Những gì đã kiểm tra và chấp nhận

Astra chạy lại targeted **18/18** và full regression **129/129**, đều pass trên local CPU. Đã đọc code path resolution, pinned canonical input reads, nested cache checks, raw-sample cardinality, shape probe ngoài timer, child output bindings và parent serial aggregation. Các thay đổi này xử lý đúng hướng R1/R2/R3. Tuy nhiên tests integration mock tự tạo metadata/guard schema nên chưa chứng minh code nối được với producer/artifact thật.

### F1 — valid accepted YAML bị reject trước khi kiểm binary

Ở `scripts/run_precision_head_latency.py:518`, điều kiện yêu cầu substring viết hoa `CCTSDB2021`. Canonical `server_fp16_capture_v1/dev_absolute.yaml` tại chính pinned commit chứa:

```yaml
path: /home/ubuntu/Dung_TDTU/nighttime-tsd-new/data/processed/cctsdb2021_clean/dev
train: images
val: images
names:
  0: prohibitory
  1: mandatory
  2: warning
nc: 3
```

Astra gọi `validate_inputs` trên canonical metadata thật, không mock, đã tái hiện `ValueError: Accepted FP16 data reference does not bind the locked dev image directory`. Lỗi xảy ra trước chỗ local thiếu engine. Không được sửa YAML artifact đã nghiệm thu để chiều theo check.

**Sửa:** parse YAML và validate các field/path theo accepted dev reference, dùng POSIX path semantics cho historical Linux path trên Windows nếu cần. Không dùng uppercase substring làm dataset identity. Giữ canonical commit/hash binding, repo-anchored default và explicit relocation đã có. Test nguyên canonical YAML phải pass phần metadata; wrong split/path phải reject, không chỉ đổi `.lower()` rồi coi là validated dev.

### F2 — tên trường guard không khớp producer, clean GPU sẽ bị báo nhầm

Producer `uniform_build_repeat.snapshot` ghi `process_guard.external_workload_detected` (line 301). Consumer mới ở runner lines 918 và 1140 đọc `external_gpu_workload_detected`, key không tồn tại. Vì `.get(...) is not False`, missing key bị coi là workload. Integration fixtures cũng dùng sai key, nên 39 mock sessions vẫn pass nhưng real session sẽ fail tại parent validation sau lượt đầu.

Astra lấy nguyên `gpu_before` từ accepted attempt2 study manifest: `external_workload_detected=false`, complete telemetry, desktop confirmations hợp lệ; GPU identity validation pass nhưng `_validate_persisted_gpu_evidence` báo `accepted snapshot records an external GPU workload`. Đây là lỗi consumer, **không phải snapshot server mới hay bằng chứng GPU đang bận**.

**Sửa:** đọc đúng `external_workload_detected` ở validation lẫn summary aggregation. Không rename producer/old artifact. Không default missing key thành False: missing/unknown vẫn incomplete, True vẫn rejected cho controlled latency. Fixture phải dùng schema thật. Thêm contract test gọi producer `snapshot` với OS/nvidia-smi I/O được mock (không GPU), đưa output trực tiếp qua consumer; thêm regression với canonical accepted snapshot. Negative cases True/missing/blocked/incomplete vẫn reject.

### Hậu kiểm và bàn giao lần sửa này

- Chạy `validate_inputs` qua toàn bộ canonical metadata thật của 13 engines và accuracy refs, chỉ mock filesystem engine stat/hash và ảnh vì không có ở local. **Không mock `validate_inputs`, canonical readers, provenance validators hoặc sửa nội dung blob.** Hiện Astra đã kiểm đường downstream 13 records có thể hoàn tất khi bypass riêng F1 trong diagnostic in-memory; đó chỉ giúp khoanh vùng, không phải production fix hoặc server validation.
- Giữ test 39-session integration, chuyển guard fixture sang output/schema producer thực. Giữ tests thiếu sample, hash mutation, failed-child/partial. Không mở thêm chức năng hay thay thiết kế đo.
- Ghi L2A-020, kết quả targeted/full và exact canonical/producer-consumer checks; push cả A2L-019. Không commit scratch `local/astra_latency_review_eb30.py`. Chưa đưa lệnh GPU/server cho người dùng ở commit đang lỗi này. Sau sửa, Astra review hai điểm này và quyết định authorization; không yêu cầu người dùng trả giá bằng một lần chạy server thất bại để phát hiện lỗi đã tái hiện local.

## A2L-020 — ACCEPT implementation 2d8f5af; authorize operator latency run

Ngày 2026-09-16. Reviewed L2A-020 và commit `2d8f5af84efda4be273cc7f9eca9becbdf202bf1`. **Decision: ACCEPT implementation; AUTHORIZE người dùng chạy latency study theo protocol trên SERVER-01 khi preflight hiện tại đạt.** Không cần thêm vòng phê duyệt code nếu Luna chỉ push entry/protocol handoff này, không đổi runner hoặc numerical contract.

### Evidence và giới hạn

- Astra chạy lại full regression **132/132 pass**, bao gồm 21 latency tests và mock integration 39 sessions. Đây là CPU/mock tests, không phải TensorRT end-to-end.
- Astra gọi `validate_inputs` xuyên suốt canonical metadata thật: 13 engine records và paired accuracy refs pass. Không mock canonical readers/YAML/provenance validators và không sửa nội dung blobs; chỉ mock binary stat/hash và image filesystem operations không có tại local. Direct server binary/image validation vẫn phải chạy thật.
- Nguyên accepted attempt2 `gpu_before` đi qua corrected telemetry consumer thành công; producer field `external_workload_detected` và YAML/POSIX checks đã sửa đúng. Snapshot historical không chứng minh server hiện đang rảnh.
- Không thay nghiên cứu, không invalidate kết quả đã nghiệm thu. Tạm dừng vòng sửa implementation này và chuyển sang thu latency thực. GPU/runtime errors chưa thể loại trừ hoàn toàn bằng local mocks.

### Luna thực hiện bàn giao operator, không tự SSH/GPU

1. Push entry này cùng cập nhật trạng thái protocol sang `implementation_accepted_operator_run_authorized` (runner không cần sửa). Ghi L2A-021 là **handoff ready**, không ghi study completed khi chưa có server artifacts. Không stage scratch local hoặc thay đổi không liên quan.
2. Cho người dùng pull fast-forward, xác nhận có reviewed commit và cung cấp preflight read-only dưới đây, mỗi lệnh một dòng. Không dùng nohup mặc định; không kill/pause/đổi quyền/clock tiến trình khác. Không dùng desktop PID cũ nếu chưa đối chiếu snapshot hiện tại.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && git merge-base --is-ancestor 2d8f5af84efda4be273cc7f9eca9becbdf202bf1 HEAD && git log -1 --oneline
hostname && nvidia-smi --query-gpu=uuid,name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used --format=csv,noheader && nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && if [ -e results/measurement_audit_v1/server_yolo11n_precision_head_latency_v1 ]; then echo OUTPUT_EXISTS_PRESERVE; else echo OUTPUT_ABSENT; fi
```

3. GPU identity của bảng này: `GPU-9850d121-55dc-e752-ffaa-df19e7585eb4`, Quadro RTX 8000, driver `595.71.05`. Giữ existing g0_size_env/runtime của attempt2, đủ 13 original binaries. Desktop đúng narrow allowlist được operator xác nhận theo current PID/path thì được phép; không yêu cầu tắt desktop. Workload compute cạnh tranh không đủ điều kiện controlled latency. Không tự chuyển serialized engines sang GPU khác; nếu server khác rảnh, báo để bố trí study khác hoặc thiết kế riêng, không trộn vào bảng latency này.
4. Khi output absent và snapshot phù hợp, Luna cung cấp lệnh foreground dựa trên lệnh cơ sở dưới đây, append một `--confirm-desktop-process PID=PATH` cho mỗi current eligible desktop row mà operator đã xác nhận. Nếu query compute-apps rỗng thì không thêm flags. Không thêm background-compute exception. Không cần hỏi Astra duyệt lại chỉ để điền current desktop confirmations.

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/run_precision_head_latency.py --out-dir results/measurement_audit_v1/server_yolo11n_precision_head_latency_v1
```

Đây là lệnh cơ sở: nếu có desktop compute rows mà thiếu confirmation, runner sẽ chặn; Luna phải chỉ rõ complete command theo snapshot mới. Runner tự kiểm canonical metadata, direct engine hashes, image hashes/shapes, env/identity trước/sau sessions. Missing engine hoặc hash mismatch thì báo exact path, **không rebuild**. Nếu failure tạo partial output, giữ nguyên và báo traceback/log; không xóa output/resume tùy tiện. Nếu cần sửa code/retry sau failure, quay lại review và output version mới.

### Khi nào done và hậu kiểm

- Console phải đến `FINISHED SESSION 39/39` và `DONE: .../latency_summary.json`, exit code 0. Đồng thời summary `status=latency_completed_review_required`, `sessions_completed=39`; đúng 13 engine × 3 rounds, 39.000 timed calls (FP16 3.000, mỗi INT8 arm 9.000), raw mỗi session 1.000. Không chỉ dựa vào file tồn tại.
- Expected output: 4 root files (`study_manifest.json`, `image_pool_manifest.json`, `latency_summary.json`, `report.md`) và 39 session directories, mỗi directory có `session.json`, `execution_manifest.json`, `child.stdout.log`, `child.stderr.log`: tổng **160 files**, trong đó **81 JSON**, 78 logs và 1 Markdown nếu thành công không có violation/extra file. Không commit engine/model binaries hay image files; push study artifacts trong output này theo scope, không `git add .`.
- Luna pull về: kiểm canonical artifact hashes và accepted engine links; 39 complete/unique sessions, raw timings/statistics/pool hash/shape observations, sampled telemetry và round/build hierarchy. Báo median/p95/p99/serial FPS cho tất cả arms và FP16, kèm per-build/per-round; không lấy fastest run hoặc chọn arm. Ghép accuracy theo đúng evaluator đã locked. Ghi L2A-021 addendum sau hậu kiểm (hoặc entry kế tiếp nếu cần), rồi dừng cho Astra xem accuracy–latency trade-off.
- Không export/retrain, không mở 15-model/calibration/device matrix, không dùng official test ở task này. Local source review hoàn tất; server measurement chưa chạy/chưa có kết quả ở thời điểm authorization.

## A2L-021 — ACCEPT latency artifacts; đóng diagnostic, chuẩn bị protocol confirmation

Review artifact commit `c6e61dea6127525933edd48af783f497445ed2ff` và L2A-021 report `69313a46e2cc1a4fd11e3dbe6881e483386f529a`. **Decision: ACCEPT completed latency study với giới hạn bên dưới.** Không cần chạy lại 39 sessions hoặc sửa raw summary status chỉ để đổi `review_required` thành `pass`; entry này là reviewer disposition riêng. Không còn blocker của study latency đã hoàn tất.

### Hậu kiểm độc lập của Astra

- Đọc canonical Git blobs, đủ **160 files, 39 sessions, 39.000 positive finite raw samples**. Tái tính mean/median/p95/p99/min/max/serial FPS cho từng session, từng engine và pooled arms: max absolute difference **0.0** trong local replay.
- **172 hash-link checks** khớp: canonical input refs, accuracy refs, session JSON và pool bindings. Runner blob ở execution commit `a5f5523a78c8f9de6a3d5cc96a1ea2376bab7d21` identical với reviewed implementation `2d8f5af84efda4be273cc7f9eca9becbdf202bf1`.
- Tạo lại PCG64 seed20260916 pool256 từ1636 IDs, warmup200 và measured1000 cyclic stream: exact. Captured dimensions và observed input `(1,3,640,640)` bao phủ đủ selected aspect-ratio groups. 13 engine records match accepted capture hashes/paths; local không có binary nên không tuyên bố trực tiếp rehash engine hoặc image bytes.
- 80 snapshots (78 child + 2 parent) đạt expected GPU identity và producer guard validation. Recorded temperatures 32–62°C. Không phát hiện competing compute trong snapshots; không suy zero interference giữa snapshots, không suy nguyên nhân biến thiên từ nhiệt độ.
- 39 stderr rỗng, stdout có DONE SESSION; cùng một warning auto-guess `task=detect` xuất hiện39 lần. Constructor warning phù hợp inference task đã locked; không có evidence đổi task/shape. Giữ warning trong logs, không chạy lại chỉ để loại warning.

### Accuracy–latency: kết luận được phép

Accuracy là COCO/XML dev diagnostic từ paired analysis; latency là synchronous decoded-image `model.predict`, conf0.001, batch1, RTX8000. Không gọi đây là pure TensorRT kernel latency hoặc FPS toàn hệ thống camera.

| Representation | Δ all AP50–95 vs baseline (pp) | Mean ms | Median ms | p95 ms | Mean latency change vs baseline |
|---|---:|---:|---:|---:|---:|
| FP16 reference | +9.1116 | 3.5917 | 3.4418 | 4.2922 | +1.67% |
| Baseline INT8 | 0 | 3.5326 | 3.4563 | 3.8718 | 0 |
| Bbox FP32 | +5.4263 | 3.6287 | 3.5548 | 3.9921 | +2.72% |
| Classification FP32 | +3.6271 | 3.5915 | 3.5237 | 3.9297 | +1.67% |
| Both FP32 | +8.5067 | 3.6556 | 3.5885 | 3.9817 | +3.48% |

Both−baseline AP50–95 CI vẫn [+7.7104,+8.9999] pp; both−FP16 là −0.6049 pp [−0.9361,−0.3670]. Both có mean latency cao hơn FP161.78% trong bảng này, không phải giải pháp đã chứng minh thắng FP16 đồng thời accuracy và speed. Không chọn một engine/build nhanh nhất để đảo kết luận. Latency differences chỉ descriptive, chưa có CI/independence model phù hợp để gọi significant hoặc equivalent.

**Round variation cần nêu rõ:** mean FP16 round1/2/3 = 3.3795/3.9745/3.4210 ms; baseline tương ứng3.4647/3.5912/3.5418 ms. Baseline chậm hơn FP16 ở rounds1 và3, nhanh hơn ở round2. FP16 pooled median3.4418 ms cũng thấp hơn baseline3.4563 ms. Do đó pooled mean advantage1.65% của INT8 không chứng minh speedup ổn định. Không bỏ round2 hậu nghiệm, không suy đó là workload lạ hoặc throttling; sampled guard không phát hiện competing workload và không đủ dữ liệu để xác định nguyên nhân. Các p95/p99 thấp hơn FP16 chỉ là quan sát của workload/session hiện có, không khẳng định tail-latency superiority tổng quát.

**Research disposition:** giữ precision-head protection là hướng confirmation có triển vọng về phục hồi accuracy/localization, nhưng **chưa khóa thành deployment method vượt FP16 và chưa scale15**. Chuỗi diagnostic đo lường này kết thúc ở đây. Không mở thêm vòng audit/repeat latency chỉ để đạt kết quả mong muốn. Tiếp theo cần kiểm khả năng lặp lại hiệu ứng trên source models/calibration selections khác, không tiếp tục tối ưu trên cùng YOLO11n/dev rồi gọi là confirmatory.

### Task hữu hạn tiếp theo cho Luna — protocol-only, không GPU

Authorize đọc repo và soạn **`docs/PRECISION_HEAD_CONFIRMATION_PLAN_V1.md`**, ghi L2A-022. Không viết/chạy runner thí nghiệm mới trong task này. Nội dung cần đủ để Astra chọn một confirmation study, không tạo hàng loạt alternatives:

1. Tóm tắt evidence và những claim chưa được chứng minh: calibration-policy superiority chưa đạt; precision-head accuracy recovery quan sát trên YOLO11n/Uniform seed42; latency không có FP16 speedup ổn định; không hứa journal acceptance/quartile.
2. Inventory source artifacts/model mappings cho **YOLOv8n và YOLO26n đã train/frozen**. Chỉ đọc manifest/code để xác định checkpoint path/hash nếu hiện có và detection-head convolution mappings cần verify; không giả định tên layer YOLO11 áp dụng cho họ khác, không dùng pretrained generic `*.pt` thay frozen trained checkpoint. Missing binaries local là operator prerequisite, không tự tải/retrain.
3. Đề xuất một protocol confirmation dev-only trên hai nano còn lại, giữ FP16 reference, baseline INT8, bbox-only, classification-only và both controls. Tách calibration-selection variance khỏi build variance; nêu số calibration selections/build repeats, tổng build/capture cần thiết và phần artifact nào reuse hợp lệ. Không chọn seed/build theo AP; không dùng cùng ordinary timing cache xuyên architecture như thể là tactic lock. Cần protocol đủ hẹp để quyết định trước main15, không yêu cầu hoàn tất mọi edge device ở pilot này.
4. Dùng source weights/protocol đã frozen, train-only calibration IDs và estimator đã nghiệm thu; không retrain, không thêm dataset. Main15-model scope vẫn giữ cho giai đoạn sau khi method/controls được khóa, không âm thầm thu hẹp paper thành riêng YOLO11n. Official test đã từng được dùng ở pilot v1: mô tả lịch sử trung thực, không gọi nó chưa từng được xem; không dùng test để sửa precision/calibration trong task này.
5. Predefine endpoints accuracy/XS/S và cách báo CIs/variability; chỉ đề xuất practical decision margins với lý do trước số liệu mới, không suy margins từ các kết quả mong muốn. Giữ classification control và FP16; không chọn both mặc định vì AP cao nhất.
6. Nêu cách phân công nhiều server nếu operator báo có máy trống: có thể chia theo source model, mỗi matched within-model comparison phải cùng device/runtime/calibration/preprocessing control; pin manifest riêng mỗi host. Không trộn server effects vào arm effect, không copy serialized TensorRT engines và coi là cùng benchmark. Nếu chỉ1server khả dụng, protocol vẫn chạy được tuần tự.
7. Ghi exact prerequisites, estimated workload bằng counts (ước lượng thời gian nếu có phải ghi giả định), stopping/reporting condition và giới hạn. Đây là plan để review; chưa có server command hoặc authorization cho build/capture mới.

Luna push A2L-021 cùng plan/L2A-022, không stage scratch `local/astra_review_latency_artifacts.py`. Chưa mở B/C/main15/official-test/edge benchmarks tự động. Sau review plan, Astra quyết định protocol và giao implementation một lượt; không yêu cầu người dùng chạy thêm gì trên server trong lúc Luna chuẩn bị tài liệu này.

## A2L-022 — review confirmation plan; khóa design và giao implementation readiness

Reviewed L2A-022/plan commit `53f12554761c80f367876756b4371a7729891d44`. **Decision: ACCEPT core factorial design, với các chỉnh sửa bắt buộc dưới đây; AUTHORIZE local implementation của preparation/readiness phase, chưa authorize 78 scored builds/captures hoặc server run.** Luna sửa plan và triển khai readiness trong cùng task, không cần thêm một lượt chỉ để sửa câu chữ. Đây là bước chuyển sang source-model confirmation, không tiếp tục latency audit.

### Design được giữ

- Hai frozen source models YOLOv8n/YOLO26n; Astra đã direct-hash hai checkpoint local, đều khớp Section2. Không retrain/download generic weights.
- Uniform train-only U42/U43/U44, 1.024 IDs/selection; 4 INT8 arms baseline/bbox/classification/both, 3 independent builds/cell; 3 FP16 builds/model. **72 INT8 + 6 FP16 = 78 scored builds/captures**, 127.608 dev image passes. Đây không phải 78 lần train.
- Giữ cả ba selections và ba builds: bỏ một factor sẽ không trả lời được câu hỏi selection variability vs repeated construction. n=3 vẫn chỉ là small-sample descriptive evidence, không chứng minh đã ước lượng variance chính xác. Không cần tăng n hoặc mở15 models ở bước này.
- Fresh empty timing input cho mỗi scored build được chấp thuận **cho study mới này**, nhằm khảo sát độ nhạy dưới independent retiming. Nó khác timing-cache replay của diagnostic YOLO11n; không pool các repeat YOLO11 cũ vào study mới, không diễn giải chênh lệch giữa studies như chỉ do architecture. Common calibration cache/model/selection chỉ dùng trong matched cells đúng source ONNX và preprocess.

### Chỉnh sửa protocol bắt buộc

**C1. Chốt inference head/output contract trước build, đặc biệt YOLO26.** Generic `imgsz/conf/iou` không đủ bảo đảm cùng inference path. Source Ultralytics **v8.4.102** phân biệt `cv2/cv3` và `one2one_cv2/one2one_cv3`; exporter hỗ trợ explicit `end2end`. Không lấy default của latest docs để suy runtime cũ. Plan chưa chứng minh frozen YOLO26 checkpoint/export dùng path nào.

- Readiness phải ghi source checkpoint/model head flags, export arguments, expected ONNX output semantics, active convolution-to-output mapping, NMS/postprocess behavior và parser/validator support. Không chọn nhánh theo AP; không âm thầm tắt end2end, đổi head hoặc thêm/double NMS để làm runner cũ chạy được.
- Giữ native inference path của frozen checkpoint dưới runtime locked, xác minh cùng path cho FP16 và mọi INT8 arm. Nếu checkpoint state/history không cho phép xác định unambiguous path, report `head_contract_unresolved` và reviewer quyết định trước export. Chưa khóa literal layer prefixes/counts từ suy đoán.
- Mapping phải theo active output dataflow và layer types, không chỉ string match. Helpers được audit/exclude, không reject chỉ vì nằm cùng namespace. Không ép YOLO26 có output `(1,7,8400)` nếu graph thực là postprocessed detections; shape một mình cũng chưa chứng minh đúng path. Capture/native matching adapter phải phù hợp và có tests trước scored matrix.
- Nguồn primary đã kiểm: [v8.4.102 head.py](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.102/ultralytics/nn/modules/head.py), [v8.4.102 exporter.py](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.102/ultralytics/engine/exporter.py). Runtime package/source hashes trên server mới là execution evidence; không upgrade theo documentation mới.

**C2. Tính đủ cache-creation workload và giữ preparation ngoài scored repeats.** Calibration cache creation bằng TensorRT thường gắn với một builder invocation. Plan hiện ghi6 creation phases nhưng78 builds không tính rõ những invocation này. Khóa cách triển khai: mỗi model/selection có một auxiliary baseline cache-generation build (6 total), giữ provenance/log/cache nhưng không capture/đưa AP vào scored tables, không chọn dùng auxiliary engine vì AP tốt. Sau đó cả72 scored INT8 builds đều cache-only, zero batches/zero cache writes; FP16 không calibration. Do đó budget là **78 scored + 6 auxiliary = 84 builder invocations**, 78 captures; 2 ONNX exports được tính riêng. Nếu API thực không cần tạo auxiliary engine, phải ghi rõ bằng chứng và thực tế invocations, không sửa counts tùy kết quả.

Calibrator algorithm/flag, deterministic input order, preprocessing và batch size phải explicit và dùng thống nhất; ưu tiên giữ audited MinMax/batch1 stream của study cũ. Table compatibility được bind model/ONNX/preprocess/version. Không reuse cache xuyên model, không dùng timing output auxiliary làm input cho scored build. FP16 settings/non-target constraints và INT8 sigmoid constraints phải khóa bằng config, không copy số77 layers của YOLO11 sang architecture khác.

**C3. Diễn giải statistics và margins.** `SD` giữa ba selection means còn chứa residual build variation, không phải purified calibration variance. Đổi tên/report thành `between_selection_mean_SD`; `within_selection_build_SD` mô tả3 independent builds. Không trừ hai SD hoặc tuyên bố đã causal-isolate variance. Build repeat index giữa arms là schedule index, không phải cùng tactic/plan; pairing chắc chắn là images/selection, không suy matched build randomness.

Khóa primary question `both-baseline` full COCO/XML AP50–95 trên từng model vì kiểm full head protection; bbox/classification vẫn mandatory controls, không chọn deployment arm bằng primary endpoint. Aggregate mỗi bootstrap draw bằng mean AP qua3 builds rồi3 selections, không average predictions hoặc pool detections từ nhiều engines. Image CIs conditional trên các engines/selection đã quan sát; 2models không là random architecture sample.

Giữ **+2 pp** chỉ như reviewer-set engineering screening target trước dữ liệu mới, không gọi là domain-validated utility threshold hay statistically significant effect size. Ghi rõ đề xuất sau pilot YOLO11, trước confirmation data. Report continuous point/CI và per-selection contrasts cho mọi endpoint. Không diễn giải CIs này thành unconditional calibration/build uncertainty hoặc paper acceptance gate. Tiêu chí consistency2/3 positive và không reversal quá2pp được giữ như descriptive gate, không kiểm định.

**Bỏ non-inferiority/equivalence claim và δ_fp16=1/2pp**, vì chưa có căn cứ application-specific hoặc power/analysis phù hợp. Báo FP16 gap/CI liên tục. **Không dùng δ_size=3pp làm ranh giới đã được xác nhận**; report XS/S point/CI và adverse effects, không tự gọi broad size robustness khi một size chưa chắc chắn. Positive full endpoint không đảm bảo XS/S. Phạm vi gọi là cross-model replication trên cùng dev set đã dùng exploratory, không gọi validation trên independent held-out data. Không đổi margins sau kết quả hoặc chọn seeds thuận lợi.

**C4. Multi-server được phép nhưng giới hạn claim rõ.** Có thể giao nguyên YOLOv8n block cho server A, YOLO26n cho server B nếu operator có máy phù hợp; không bắt cả hai phải dùng UUID cũ. Mỗi within-model arm/selection/repeat và FP16 reference phải cùng GPU/runtime đã bind. Lock TensorRT10.16.1.11/Ultralytics8.4.102 và numerical configuration để reuse implementation; nếu môi trường không tương thích thì báo prerequisite, không auto-upgrade. Nếu models nằm trên GPUs khác nhau, model×host confounded: chỉ khẳng định within-model contrasts trên từng host, không suy khác biệt effect giữa models là architecture-only và không gộp absolute latency. Unknown sampled interference vẫn là limitation; không đổi shared-lab desktop policy.

**C5. Readiness không phải prerequisite vòng tròn.** Section9 đang yêu cầu operator có ONNX/mapping trước implementation, dù chưa có runner export. Tách prerequisite raw weights/data/env khỏi artifacts do reviewed prepare/export phase sẽ tạo. Không yêu cầu người dùng tự export để lấp chỗ này.

### Task implementation được giao Luna lần này

1. Update `PRECISION_HEAD_CONFIRMATION_PLAN_V1.md` theo C1–C5, status `design_locked_readiness_implementation`; không tuyên bố full study GPU-authorized. Không sửa numerical data/historical artifacts.
2. Implement readiness/preparation module + tests, config khóa model/checkpoint/selection hashes và counts/schedule. Local chỉ CPU đọc metadata/trusted frozen model structure khi dependency sẵn có; không GPU/TensorRT build/ONNX export ở local, không tự cài/nâng runtime. Phần readiness server dự kiến gồm direct files/env/overlap audit, model-head/export contract evidence, kiến trúc các outputs/mapping checks; graph mapping nếu cần export chỉ chạy trong server prepare phase sau reviewer authorization. Không gọi build/capture matrix tự động khi readiness pass.
3. Readiness output machine-readable mới, model-specific và no-overwrite; gồm ready/unresolved/missing prerequisites, source refs/hashes, same dev/2706GT contract, calibration IDs/order/tensor recipe và train-only checks, environment identity, native head/output evidence, proposed per-model constraints/schedule, 84 total/78 scored accounting. Các trường chưa observable phải ghi unknown, không fabricate layer list hoặc effective precision.
4. Tests phải dùng actual canonical metadata shapes/schemas; checksums pin commit, not only checkout self-hash. Test 2models×39 scored cells +6 auxiliary jobs, disjoint arms, missing/ambiguous active-head mapping, calibration overlap/mismatch, no implicit fallback, wrong output representation, parent CPU/child separation, partial/no-overwrite. Reuse established helper khi phù hợp, không nhân bản một runner YOLO11 chỉ bằng replace model name.
5. Ghi L2A-023 với code/files/tests, fixed inference-path evidence hoặc unresolved details, protocol changes và candidate server prepare command (chưa chạy). Push cả A2L-022. Astra review readiness implementation trước authorizing operator prepare; sau actual graph/metadata evidence mới triển khai/authorize full scored runner. Không yêu cầu thêm latency/YOLO11 diagnostic, không train, không chạy official test hoặc main15.

Completion của task này là readiness code/tests/config và corrected protocol, không phải78 engines. Đây là gate để tránh build cả matrix với sai detection head/output. Không giữ các blocker giả chỉ vì thiếu engine/ONNX vốn chưa được phép tạo; báo rõ phase nào sẽ tạo artifact nào.

## A2L-023 — accept CPU head evidence; hoàn thiện data readiness/schedule trước server

Reviewed commit `73ab3cf2e066cd856e054ed07803af11c6d902a7` và L2A-023. **Decision: ACCEPT frozen PyTorch head/output evidence; CHANGES REQUESTED cho data readiness/config/schedule.** Chưa authorize export/build hoặc scored matrix. Không yêu cầu chạy lại các study cũ; đây là bổ sung kiểm tra còn thiếu trong preparation đã giao, không thêm nghiên cứu.

### Những gì đã được xác nhận độc lập

Astra chạy lại targeted **11/11** và full **143/143**, đều pass bằng existing local measurement environment. Actual-model CPU test không bị skip: hai frozen checkpoint đi qua `inspect_frozen_model` thành công. Đây là forward trên CPU, không phải graph/export/TRT validation.

- YOLOv8n: Detect index22, `end2end=false`, active `cv2/cv3`, primary `[1,7,8400]`.
- YOLO26n: Detect index23, `end2end=true`, active `one2one_cv2/one2one_cv3`, primary `[1,300,6]`; `cv2/cv3` là auxiliary không cấp primary inference output, **không có nghĩa chúng không được tính trong unfused PyTorch forward**.
- Local report environment Torch2.8.0+cu129/NumPy2.4.2 khác server locked Torch2.5.1+cu121/NumPy2.4.4; Ultralytics8.4.102 đúng. Accept architecture evidence với giới hạn này, không coi server env đã verified.
- Các local readiness files đang ở ignored output, không thuộc six-file implementation commit. L2A report là tracked evidence; nếu bàn giao local artifact cho reviewer khác thì push JSON/Markdown có scope, không claim chúng đã nằm trong commit hiện tại.
- Pairwise selection overlap76/75/73 không tự làm invalid ba independent seeded sampling runs; image sets được phép giao nhau khi sampling từ cùng train pool. Nó chỉ bác bỏ claim ba subsets disjoint, không chứng minh RNG draws phụ thuộc hoặc buộc resample selections.

### D1 — kiểm đúng ảnh/split/materialization, không chỉ counts và existence

`validate_calibration_manifest` hiện chỉ `is_file()` cho YAML và source image/label. Nó không parse YAML, resolve selected calibration image list, hash source/materialized bytes hoặc đối chiếu dev/test IDs. `validate_dataset_contract` chỉ đếm1636 files và2706 dòng: bộ ảnh/label khác nhưng cùng count có thể vẫn `verified`. `train_only=true` hiện chỉ được suy từ text/path prefix, chưa là leakage audit thực tế. Đây là thiếu validation, **không phải phát hiện dataset hiện tại bị leak**.

Yêu cầu sửa trong readiness, chưa materialize/export:

1. Đọc canonical dev IDs/shape reference đã dùng ở accepted FP16 capture; đối chiếu exact current dev image IDs và label stems, kiểm label class/finite normalized bbox hợp lệ. Counts chỉ là kiểm bổ sung. Record actual file hashes hoặc ordered inventory hash + per-file inventory để lần prepare sau bind cùng dữ liệu. Không mô tả newly measured hashes là historical image-byte hashes.
2. Selected calibration IDs phải nằm trong actual train inventory, loại dev/test ID overlaps qua membership checks. Chỉ đọc exclusion inventory/IDs cho test, không đọc test labels/images để xây policy. Chuẩn hóa path thành `train/images/basename` và `train/labels/basename`, reject traversal/symlink escape khỏi intended split; image-label stem phải match. Giữ đúng U42/U43/U44 IDs và deterministic order, không tạo lựa chọn khác để làm check pass.
3. Nếu calibration.yaml tồn tại, parse YAML theo producer schema, resolve calibration source đúng và kiểm **đúng1024 IDs**, materialized image bytes khớp corresponding train files. Một YAML trỏ test/selection khác/empty/extra image phải bị ghi invalid chứ không `complete`. Nếu YAML/images thiếu, ghi missing; không tự rebuild/move/repair. Bind actual bytes/order, không chỉ manifest metadata.
4. Định nghĩa recipe bằng helper/source hash và concrete decoder/color/letterbox/resize/normalization/order parameters, hoặc đánh dấu recipe `unresolved` nếu chưa kiểm; chuỗi `locked CCTSDB train image decode/resize/normalize path` chưa khóa preprocessing. Reuse audited preparation logic `uniform_build_repeat.py` nếu phù hợp nhưng không import GPU `environment()` vào CPU readiness. Không tạo multi-GB tensors trong task local này.

### D2 — config/provenance/status phải thể hiện đúng mức đã kiểm

Reviewer đã tái hiện `load_config` nhận `runtime.conf=0.9` mà không reject. Model path/hash và expected head contracts cũng lấy từ cùng mutable JSON; config path trong report luôn hard-coded dù CLI cho phép file khác. Output chưa lưu run timestamp/code/script/config hashes. Do đó config tự khai chưa đủ để gọi protocol locked.

- Validate tất cả locked scientific fields (source model identity/hash, U42/U43/U44 identity/hash, runtime conf/iou/max_det, counts, head/export semantics và bootstrap/decision settings) against reviewed configuration identity hoặc explicit immutable contract, không lấy observed và expected từ cùng một mutable object. Record config path thực, bytes/canonical semantic hash, execution Git commit, script hash và UTC timestamp. Không ép toàn working tree sạch nếu chỉ có unrelated user results.
- Record observed CPU package versions trước probe; phân biệt architecture evidence trên compatible local env với server-runtime-ready. Unsupported Ultralytics/head behavior là unresolved, không tự thay đổi mode. `probe_models=False` phải có explicit missing probe reason, không có trạng thái gate ready chỉ vì error list rỗng.
- Scored matrix gate luôn blocked/deferred ở runner readiness này, kể cả raw inputs đầy đủ, vì ONNX/TRT graph/mapping chưa được kiểm. Có thể thêm `raw_inputs_ready_for_server_prepare` riêng. `scored_run_authorized=false` giữ nguyên. Main readiness DONE chỉ có nghĩa report đã viết, không chứng minh mọi prerequisite pass.

### D3 — lịch đang group theo arm, trái protocol interleaving; validator bỏ lọt cell sai

Current generator chạy U42 baseline repeats1/2/3 rồi bbox repeats1/2/3, rồi classification và both; FP16 đều cuối model block. Đây không phải interleaving đã yêu cầu. Astra đã đổi một scored repeat1 thành99 trong memory và `validate_schedule` vẫn trả verified vì kiểm counts/uniqueness chưa đủ.

Khóa revised deterministic schedule **trước scored data mới**, giữ tổng84/78:

- Mỗi model: ba auxiliary cache creations U42/U43/U44 trước, không capture/scored; timing outputs của chúng không được reuse.
- Canonical13 scored cells/round: FP16 rồi `[U42 × baseline,bbox,classification,both]`, `[U43 × four arms]`, `[U44 × four arms]`. Round r dùng build repeat r; rounds1/2/3 rotate-left0/4/8 trên canonical13 cells. Tổng39 scored/model và42invocations/model. Đây là reproducible interleaving, không claim full Latin square hoặc randomized causal assignment.
- Validate exact model×selection×arm×repeat expected keys, phase/scored/cache flags, order và dependency auxiliary-before-use; reject missing/repeat99/wrong selection/extra arm ngay cả khi tổngcounts không đổi. Publish schedule version/hash mới; supersede proposed old schedule chưa chạy, không sửa historical study.

### Tests và bàn giao

Luna sửa D1–D3 và thêm fixtures cho same-count wrong dev IDs, wrong/malformed calibration YAML, substituted materialized bytes, train/dev overlap, wrong config hash/runtime, model-not-probed và malformed schedule. Tests filesystem nên dùng temporary fixtures hoặc canonical producer inputs để pass cả local thiếu YAML lẫn server đã có YAML; không assert mặc định rằng mọi môi trường đều thiếu calibration.yaml.

Giữ CPU-only boundary, no export/build/GPU, no auto-install. Chạy targeted/full, cập nhật protocol/config/L2A-024, push cả A2L-023. Sau review bản sửa, CPU inventory trên server **không cần GPU idle** và có thể chạy trên server được operator chọn; chỉ phase GPU sau mới cần workload controls. Không viết thêm full84-job runner ở task sửa readiness này. Không yêu cầu người dùng giải quyết local missing YAML bằng tải dataset/train lại; server audit sẽ xác định vật liệu có sẵn.

## A2L-024 — review 902c46e; sửa false-ready và bàn giao CPU inventory cho operator

Reviewed `902c46e9a02a27e57b99f411137bca9e7228e0c9`, L2A-024. Astra xác nhận local HEAD, origin/master và remote master cùng commit này; cả A2L-023 và L2A-024 đã được push. Astra independently reran targeted **15/15**, full **147/147**, CPU-only; actual frozen-model tests không skip. Không chạy TensorRT/server. **ACCEPT D3 interleaving và immutable config/provenance improvements; còn các lỗi readiness cụ thể dưới đây.** Không thay đổi numerical design, weights, seeds, arms hoặc 84/78 accounting.

### R1 — resolve YAML thật, không chỉ basename

Astra tái hiện bằng temporary fixture: `path: <temp>/WRONG/uniform_test` với expected directory `<temp>/calibration/uniform_test` vẫn trả `complete` nếu expected directory có ảnh đúng. Nguyên nhân `parse_materialized_calibration_yaml` so sánh `.name`, sau đó tự gán `materialized_dir = expected_dir`, bỏ qua declared path.

- Producer `build_calibration_set.write_yaml` hiện viết absolute POSIX directory. Với contract này phải resolve chính declared absolute path trên host và so sánh exact intended directory; reject same-basename/different-parent, traversal, foreign-host/stale root, unsupported relative path hoặc symlink escape. Không silently rebase YAML sang expected directory. Nếu cần hỗ trợ relocation phải là explicit producer contract, chưa tự sửa dữ liệu trong task này.
- Astra cũng tái hiện malformed YAML `names: [broken` ném `yaml.parser.ParserError` ra ngoài vì không catch `yaml.YAMLError`. Ghi structured `invalid`/reason trong report thay vì crash trước khi ghi audit. Không catch-all rồi coi complete.
- Tests: correct producer YAML pass; wrong parent with same basename invalid; syntactically malformed YAML invalid; substituted bytes/extra IDs vẫn invalid.

### R2 — lỗi nested phải chặn raw readiness tổng

Astra tái hiện integration bằng mocked materialization: cả ba record `status=invalid`, nested `materialization.errors=['image bytes mismatch']`, nhưng outer `errors=[]`; `build_readiness` trả `raw_inputs_ready_for_server_prepare=true`, `status=ready_for_server_prepare_review`, unresolved rỗng. Đây là false-ready thực, không chỉ thiếu câu chữ.

- Aggregate nested materialization errors/missing với selection ID. Require từng calibration record `verified` và materialization `complete` trước khi raw-ready; unknown/invalid/missing status không được pass chỉ vì errors rỗng. Matrix gate vẫn deferred và unauthorized.
- Add end-to-end readiness integration fixture cho all-complete positive case, invalid YAML, mismatched bytes, unknown status, missing YAML, probe disabled; assert cả raw flag, overall status và reason. Không chỉ unit-test parser.
- Test cuối hiện vẫn assert mọi checkout có missing YAML; bỏ assumption này, dùng temporary fixture hoặc mock explicit missing/complete states để tests pass trên server đã materialize đủ.

### R3 — chứng minh đủ exclusion inventory và runtime compatibility

Code review thấy `_directory_ids` cho missing directory trả rỗng; train empty và labels empty có thể bằng nhau, test exclusion empty được báo như zero-overlap. Missing inventory không chứng minh không leak. Kiểm existence/nonempty, unique stems và expected split inventory (train 14720, positive-test exclusion 1500); ưu tiên bind canonical split IDs từ existing approved manifests nếu có. Không đọc test pixels/labels; chỉ directory entries/canonical IDs. Record thiếu inventory là missing/unresolved, không true train-only proof. Selected membership không skip check khi train set rỗng. Add fixture missing exclusion/train inventory và actual train/dev overlap. Không resample hoặc sửa splits để pass.

Observed package versions đang được record trước probe nhưng chưa so sánh với supported Ultralytics8.4.102. Unsupported/missing Ultralytics phải explicit unresolved và không cấp native-head verified status từ probe đó. Torch/NumPy local khác server vẫn được dùng CPU structural evidence như đã chấp thuận; không ép local thành server runtime hoặc tự cài packages. Server runtime compatibility được report riêng, không claim CUDA/TRT GPU verification từ metadata. Test mismatch bằng mocks.

### Gói bàn giao và authorization giới hạn để giảm vòng chờ

1. Luna sửa R1–R3 trên local, cập nhật tests/protocol và ghi **L2A-025** nêu regression reproductions đã đóng. Push cả entry A2L-024 này; không commit local/astra_review_readiness_902c.py (scratch ignored). Astra không push thay Luna.
2. **Sau khi fixes và targeted/full tests pass, Luna được cung cấp lệnh CPU inventory cho operator ngay, không cần một vòng chờ riêng chỉ để cấp lệnh read-only.** Đây là authorization chỉ cho CPU audit, không phải chấp thuận readiness implementation cuối cùng hoặc export. Nếu thay đổi vượt R1–R3/numerical design, dừng hỏi trước.
3. Workflow bắt buộc: **Luna không SSH và không tự chạy server.** Luna push code, báo commit và các lệnh foreground mỗi lệnh một dòng; người dùng pull/chạy/push artifacts; Luna pull hậu kiểm rồi Astra review. Không mặc định nohup. Có thể dùng server khác với đủ frozen inputs; CPU inventory không cần GPU idle và không block vì desktop/compute GPU job khác. Giới hạn CPU threads nếu cần để không làm ảnh hưởng lab.
4. Candidate operator CPU command, chỉ giao sau fixes/tests/push; dùng output mới, tuyệt đối không overwrite nếu đã tồn tại:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 local/g0_size_env/bin/python scripts/prepare_precision_head_confirmation.py --config configs/precision_head_confirmation_v1.json --out-dir results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2
```

Luna kèm git pull/commit check và scoped artifact add/commit/push commands phù hợp, không stage unrelated files. Năm outputs dự kiến: readiness_manifest.json, model_contracts.json, calibration_readiness.json, schedule.json, report.md. DONE là report written, không phải raw inputs pass; hậu kiểm nested statuses, provenance và raw flags. Giữ artifact report kể cả unresolved; không tự sửa YAML/dataset trên server để làm pass, không yêu cầu train lại/download generic weights.
5. Đồng thời Luna có thể soạn checklist/thiết kế server ONNX mapping preparation (không export, không build, chưa triển khai full matrix), để tránh chờ rảnh CPU. Artifact readiness và fixes sẽ được Astra review cùng gói trước bước graph export. Chưa mở 6 auxiliary builds/78 scored builds, official test, main15 hoặc edge benchmarks. Không đổi protocol hoặc chọn arm.

Đây là sửa validator có counterexamples và một CPU inventory hữu ích, không yêu cầu lặp lại YOLO11n, latency hoặc bootstrap. Các kết quả nghiên cứu cũ không bị invalidated bởi các lỗi readiness mới này.

## A2L-025 — review 9c0597d; APPROVE operator CPU inventory

Reviewed L2A-025 và commit `9c0597d1ec6f6d73c98fffa7cc3779a00eddec34`. Astra xác nhận HEAD/origin/master/remote master đồng nhất tại thời điểm review; A2L-024 và L2A-025 đã tracked/pushed. Astra independently reran **18/18 targeted**, **150/150 full regression**, CPU-only (`CUDA_VISIBLE_DEVICES=-1`, OMP/MKL threads2); exit0, không skip actual frozen-model test. Full regression đã được xác nhận, bất kể recap chat ghi thiếu thông tin. Mock build/session logs trong tests không phải GPU execution evidence.

**Decision: ACCEPT corrections for CPU inventory use; AUTHORIZE operator chạy CPU readiness trên server ngay.** Không yêu cầu thêm vòng sửa trước bước read-only này. R1 đã resolve actual declared absolute YAML path và catch YAML parse errors; R2 raw-ready yêu cầu mọi materialization verified/complete và promote nested errors; R3 missing split inventory/runtime mismatch không còn được ngầm coi verified. D3 schedule và frozen design giữ nguyên.

### Execution và hậu kiểm

- Luna không SSH, không tự chạy server. Người dùng pull code, chạy foreground CPU command đã giao ở A2L-024 với output `results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2`, rồi push đúng5 report files. Không cần GPU idle, không kill/pause workload hoặc desktop; không tự install/upgrade environment. Nếu output đã tồn tại, giữ nguyên và hậu kiểm report trước, không overwrite/rerun tùy tiện.
- Command được bọc dòng trong chat dễ làm hỏng `/usr/bin` hoặc `scripts/prepare_precision_head_confirmation.py`. Giao các lệnh ngắn riêng: cd, git pull, rev-parse, rồi CPU invocation. Kiểm code revision; docs-only descendant không đổi reviewed runner/config cũng có thể dùng nếu Luna chứng minh hashes và ghi actual execution commit. Không reset worktree để ép HEAD.
- Khi DONE, inspect `status`, `raw_inputs_ready_for_server_prepare`, `missing_prerequisites`, `unresolved_checks`, từng U42/U43/U44 materialization, model-head contracts, dataset1636/2706/train14720/test-exclusion1500, schedule84/78 và provenance. `scored_run_authorized=false` và graph gate deferred là trạng thái đúng; không coi đó là thất bại CPU inventory. Nếu missing/invalid, preserve/push report và báo nguyên nhân, chưa sửa source data/YAML.
- Luna pull canonical artifacts, hậu kiểm và ghi **L2A-026** với commit/file inventory/hash links, current-host observations, unresolved details. Đồng thời có thể chuẩn bị checklist ONNX mapping phase như A2L-024; chưa export/build hoặc viết/chạy full scored matrix.

### Giới hạn còn lại, không block CPU inventory

Approved dataset manifest mới bind counts/provenance, không chứa historical per-image train/test IDs (`ids_available=false`). Report current-directory IDs/hashes đúng phạm vi, không gọi là historical byte identity hoặc exhaustive leakage proof. Before export review, đối chiếu current train/test ID intersection từ inventories đã lưu (không đọc test pixels/labels); selected calibration/test overlap đã được validator kiểm riêng.

Test `test_build_readiness_preserves_missing_prepare_artifacts_without_authorizing_run` vẫn assume local thiếu calibration YAML. Luna nên đổi thành explicit fixture/mocked missing khi lần tới chạm tests; không yêu cầu chạy full suite đó trên server đã materialized để chứng minh readiness. Current production command không chạy unittest nên đây không cản CPU audit.

Chỉ CPU inventory được duyệt: **không ONNX export, không auxiliary/scored TensorRT build, không capture/benchmark mới, không main15/official-test/edge study**. Sau server artifact review mới chốt graph-preparation implementation/authorization. Astra để entry này unstaged để Luna commit/push cùng bàn giao, không yêu cầu operator đợi docs-only commit mới được chạy reviewed9c0597d.

## A2L-026 — ACCEPT server readiness; GO local graph-preparation implementation

Reviewed artifact `7c0ea9e7fe86dfa6358ec1ee90473f9243a53f76` và L2A-026/report commit `96836a9db9c9672d7d89b0f2fe3dacdbe7e5bcad`. Astra independently đọc canonical Git blobs (không checkout CRLF hashes): đúng5files, cả5SHA256 khớp bảng L2A-026, subordinate JSON sections exact với manifest, script/config canonical hashes khớp execution commit9c0597d, semantic config và schedule hashes reproduce. 3.072 selected source/materialized hash links nhất quán; inventories train14720/dev1636/test1500, dev2706GT, train-dev/train-test/dev-test ID intersections đều0. Đây là kiểm chứng liên kết artifact, không claim Astra trực tiếp rehash ảnh server hoặc xác nhận content-duplicate absence.

**Decision: ACCEPT CPU readiness. GO bước triển khai local ONNX/graph preparation; chưa GO execution export hoặc84builds.** Không sửa study design, không audit lại các study cũ. Native CPU contracts YOLOv8n raw[1,7,8400], YOLO26n native end2end[1,300,6] được giữ; server GPU/TRT và exported dataflow chưa có evidence.

### Task Luna — một gói implementation hoàn chỉnh

Owner Luna: code/tests/protocol cho **prepare-only** runner, gợi ý `scripts/prepare_precision_head_confirmation_graph.py`, `tests/test_precision_head_confirmation_graph.py`, `docs/PRECISION_HEAD_CONFIRMATION_GRAPH_PREP_V1.md`. Reuse reviewed readiness config/helpers, không nhân bản numerical protocol. CLI scope gồm reviewed readiness-root, model selection `yolov8n|yolo26n|all`, new output-root; không có arbitrary checkpoint/engine argument và không auto-run matrix khi prepare pass. Model filter để operator có thể chia nguyên model block giữa servers; host khác cần current-host CPU readiness tương ứng, không gọi SERVER-01 artifact là verification cho server khác.

Inputs: canonical accepted readiness artifact/hash, frozen checkpoint hashes, runtime/export config, U42/U43/U44 manifests và materialization bindings. Before execution recheck actual selected model/data/config bindings; record host/env actual, actual producer source/version hashes và timestamp/git. Preserve originals, export từ private workspace/copy để không ghi đè best.pt/best.onnx cũ; no-overwrite output, structured failure report giữ partials, không silent resume/fallback.

Prepare outputs/model:

1. Native ONNX export (640, batch1, float input, opset17, simplify=true, dynamic=false, preserve end2end as locked), source/ONNX hashes, exact effective export args and environment. Không generic downloaded model, không disable end2end hoặc thêm double NMS. Đếm2 exports chính total, không build engine hoặc tạo calibration cache trong phase này.
2. ONNX checker + input/output schema và **active graph dataflow mapping**: trace bbox/class Conv outputs tới relevant output coordinates/scores/classes, audit helpers/non-Conv/inactive branches; shape/name prefix không đủ. Preserve source-to-export mapping evidence qua simplification/fusion; nếu ambiguous record unresolved, không gán tên branch bằng suy đoán hoặc force counts từYOLO11. Với YOLO26 end2end, class selection/TopK/Gather có thể ảnh hưởng cả selected box outputs: distinguish branch ownership/source lineage với downstream reachability, không require disjoint output ancestry một cách sai. INT8 arm target sets disjoint theo branch ownership; both union; baseline protection giữ đã khóa.
3. Adapter contract cho model-specific primary outputs, postprocess/filter semantics và native validation compatibility. Synthetic fixture tests cho raw/non-end2end versus postprocessed/end2end, empty/invalid output, coordinate/class semantics, no double NMS. Evidence chưa có real TensorRT phải ghi deferred; không gọi parser-ready-to-score chỉ từ shape.
4. Concrete frozen calibration tensor recipe và deterministic file order bind existing audited helper/source hash, decoder/color/letterbox/pad/interpolation/layout/dtype/normalization/batch settings. Không tự đổi preprocess; server preflight resolve previously unresolved recipe before cache creation. Không materialize full multi-GB tensors chỉ để viết report này nếu chưa cần; không read official test pixels/labels.
5. Summary status `graph_preparation_completed_review_required` chỉ khi actual checks đủ; unresolved/errors ghi cụ thể, `scored_run_authorized=false`. Manifest file inventory/hashes cho JSON/report/log/schema; ONNX binaries ở server theo existing binary policy, không commit weights/engine/private data. CPU-generated fixtures không giả làm ONNX/TRT server evidence.

Scope boundaries: local chỉ CPU code/mocked lifecycle/synthetic graph tests, không export frozen models/TensorRT/GPU. Server execution do operator sau Astra review implementation, foreground, new model-specific output; Luna không SSH. Có thể thiết kế ONNX checker/source parity diagnostics trên CPU server nhưng mọi actual export/forward phải được ghi, không phát sinh scored evaluation hoặc bootstrap mới. TRT parser inspection nếu implementation cần chỉ network parse, không builder execution, phải explicit phase và prerequisite trong review; không tự mở6aux builds để có mapping.

Tests tối thiểu: supported head/output modes, bad/ambiguous/missing mapping, nonConv helpers excluded, unreachable nodes, output ancestry versus owner distinction, wrong checkpoint/readiness/config/hash, native/end2end not silently switched, no overwrite/partial failure, CPU parent/isolated child boundaries, single-model accounting, exact no-build dispatch. Targeted/full regression, py_compile/diff check; fix remaining old readiness test local-missing-YAML assumption với fixture khi chạm suite. Không thêm scientific thresholds sau data.

Luna ghi **L2A-027**: files, tests, exact candidate operator commands, expected outputs, unresolved semantics, server resource needs; push bằng workflow hiện tại. Gửi một gói để Astra review interface + implementation + commands cùng lúc, không xin từng lệnh nhỏ. **Chưa giao lệnh export cho operator chạy trước review.** Không cần đợi edge inventory để implement phần này.

## A2L-027 — review graph6425b1a; fix execution adapter and graph validation before export

Reviewed commit `6425b1a1df3f45ac9a798c237d3d8211190a0dc4`, L2A-027. Astra reran full suite **166/166 PASS** on checkout `5d407f1` (graph code unchanged; edge lane adds its5tests). Mock session/build output is not real GPU evidence. **CHANGES REQUESTED, no server export yet.** Preserve frozen numerical design and accepted readiness; no repeat of older studies.

### G1 — production environment adapter deterministically blocks native probe

`prepare_model` passes graph `environment_evidence()` to readiness `inspect_frozen_model`. Graph returns `producer_packages.ultralytics.version`; readiness expects top-level `ultralytics`. With actual producer schema, `runtime_compatibility_evidence` returns `ultralytics_missing_for_native_head_probe` even when nested version is8.4.102. Existing mocked tests bypassed this boundary.

Normalize explicit version-only environment for the readiness helper, retain full producer metadata separately. Check locked server Torch/Ultralytics dependencies before export without installing/upgrading; no changing helper's accepted hash merely to work around adapter. Add integration test using real graph environment schema and real readiness compatibility function, not mocking away the function that failed. Missing/wrong version must still stop; matching8.4.102 must reach export stub. Test one full prepare_model success with CPU stub export/graph, and dependency failures before exporter call.

### G2 — graph semantics can currently false-pass

Astra reversed Concat inputs of existing YOLOv8 fixture. Audit still `verified` with class span[0,3] and bbox[3,7], while adapter claims bbox[0,4]/class[4,7]. Also `precision_target_sets` returns targets for a mapping with `status=mapping_unresolved` if target sets are nonempty/disjoint.

- Require expected exact branch spans/order at decoded merge (box0:4, class4:7), output-linked merge, unique primary output and locked input/output rank/shape/dtype. Do not require disjoint final ancestors for YOLO26 TopK/Gather; branch ownership and downstream selection remain different notions.
- `_forward` currently accepts reaching any graph output; require relevant primary output, not an unused/debug output. Verify merge-to-primary path; unknown reshape/transposition/slicing semantics must be unresolved rather than taking shape as semantic proof. Do not fabricate a universally verified YOLO26 adapter from shape-only dummy tensors; label declared contract versus observed numeric evidence distinctly. Real TRT forward stays deferred.
- `precision_target_sets` must require verified status plus internally consistent sets/mapping. Baseline wording should describe INT8 eligibility/constraints, not assert every other convolution executes INT8; sigmoid FP32 baseline constraints remain separate from intervention targets.
- Native output validator: `x==x` does not reject infinity and `int(class_id)` accepts fractional classes. Use finite-value and integer-class checks in allowed range, appropriate coordinate/order validation; invalid/empty outputs tested according to declared contract. Do not silently clamp/change detections.
- Tests: reversed spans, debug-only reachability, disconnected merge, multiple same-shape outputs, wrong input dtype/shape, unresolved mapping target request, nonfinite/fractional-class detections, plus valid raw/end2end paths. Synthetic fixtures must not make invalid dataflow look like real export evidence.

### G3 — bind current inputs and contain exporter side effects

`validate_current_bindings` recalculates dataset/calibration records but checks their self-consistency, not equality of current dev inventory and selected source/materialized hashes against accepted snapshot. A same-shape image substituted in source+materialized can pass. Compare accepted inventories/selection IDs/order/byte hashes where recorded; no image/label modifications to pass. Record current bindings in output, not just `dataset_status=verified`.

Disable Ultralytics automatic dependency installation before any import in child (review actual pinned exporter dependency behavior; e.g. enforced process environment plus prerequisite checks). CPU child must inherit explicit CUDA-hidden/thread controls, without claiming zero CUDA API queries if library internally queries availability. `device=cpu` is not an auto-install guard. Check required onnx/onnxslim/runtime prerequisites and report missing, no pip/network mutation. Record actual effective exporter args/end2end state before/after and actual producer implementation hashes, not simply relabel requested args as effective.

Calibration recipe currently remains generic unresolved after server prepare. Finish the concrete recipe audit requested A2L-026 (read producer/helper source and specify decoder/color/padding/resize/layout/normalization/order; bind source/hash), or accurately mark `graph_only_completed_recipe_unresolved` with explicit prerequisite before cache build. No multi-GB tensor creation required; no auxiliary build here.

### Handoff and ownership

Luna fixes G1–G3, adds meaningful boundary/integration tests, updates protocol, records **L2A-028**, pushes scoped files. Candidate CPU server commands remain unexecuted until Astra review; no GPU/TensorRT/ONNX export locally. Preserve partials if any (none reported). Avoid further mock-only success claims.

At review the shared working directory was on **luna1/e2l1-001-edge-readiness**, HEAD5d407f1, although Luna's graph belongs on master6425b1a. Do not checkout/reset/merge here while another lane is active. Create/use separate clone/worktree; coordinate copying only this entry to main-lane docs before commit. No force push or broad staging. Astra leaves docs edits unstaged; each lane owns its inbox/file updates. Edge branch push URL is a PR creation link, not evidence a PR was opened/merged.

Luna1 inventory accepted separately in E2L1-002; it can proceed with local harness/protocol preparation while graph is fixed. Operator need not run a graph command that currently fails its own environment adapter. No source-model retraining or change of research scope.

## A2L-028 — ae638cc reviewed; GO operator CPU ONNX preparation only

Reviewed `ae638ccb6af1ccfea908d28f68e5730691b7cca2` / L2A-028. Astra reran graph17/17PASS and the two previously data-blocked readiness tests2/2PASS using graph-worktree code with the readiness test module's REPO redirected in memory to `D:/Research/paper` (existing dataset), no file edits/copy and no export/GPU. This resolves those two missing-data failures, not proof of server export. Full regression verification is reported separately after run; do not count absent dataset tests as passes in the original163/165 run.

Full discovery reviewer run completed **167/167PASS** in40.573s with that in-memory readiness test REPO binding. Record the actual observed count (different from Luna's earlier165), no claim it was an unmodified clean-worktree/data-free invocation. No server/GPU/ONNX export was executed by Astra.

**Decision: GO operator-run CPU ONNX preparation at reviewed ae638cc; NO-GO TensorRT build/capture/scored matrix.** Accept environment adapter fix, current-to-accepted byte snapshot binding, no-auto-install guard, isolated CUDA-hidden child, input/output schema checks and improved mapping diagnostics for this bounded preparation. Do not repeat old YOLO11 studies or collect edge metrics to unlock this CPU job.

### What this GO does and does not certify

Purpose: produce actual ONNX/checker/source-lineage artifacts from the2frozen models on SERVER-01, with private copies and no overwrite. Both models sequentially, CPU only, no GPU-idle requirement. The allowed outcome includes structured `mapping_unresolved`; preserve that evidence, do not change model/end2end/seed to force pass. File creation/finite CPU forward for export is authorized, not training/calibration/evaluation.

Graph mapping `verified` here remains provisional structural evidence, not a scored-parser certificate. `_downstream_semantic_audit` still recognizes operators mostly by names/shape/attributes; e.g. Gather index lineage or same-shape arithmetic can change semantics. Actual graph review and subsequent numerical/source-output checks remain mandatory before scoring. Source model end2end before/after is not identical to observing the exporter's internal copied model; graph/schema evidence must corroborate. Recipe is explicitly unresolved; resolve actual producer trace before cache generation. No local frozen ONNX export required to prove server readiness.

Do not block this bounded collection merely because no real ONNX yet exists. If actual graph is rejected, retain generated ONNX privately and failure logs; next step is read-only analysis of that graph, not repeated full export without diagnosis. Current runner may not save full mapping/schema before raising; Luna should retrieve a read-only schema/mapping diagnostic from the preserved ONNX in a follow-up if needed, with no new builder invocation.

### Operator command and artifact procedure

Luna supplies separate one-line commands: cd, safe git pull, HEAD check, then invocation below. Expected reviewed code ae638cc; a docs-only descendant is acceptable after confirming runner/config/helper unchanged. Preserve dirty unrelated work; no reset/clean. Foreground, no nohup. New output root must be absent; if present, inspect existing completion/failure rather than delete/resume.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new
```

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 YOLO_AUTOINSTALL=0 PIP_NO_INDEX=1 local/g0_size_env/bin/python scripts/prepare_precision_head_confirmation_graph.py --readiness-root results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2 --model all --out-dir results/measurement_audit_v1/precision_head_confirmation_graph_prep_v1
```

Do not install missing dependencies automatically. If preflight fails, report exact missing/version and preserve partial report. No arbitrary runtime upgrade. Operator pushes scoped JSON/schema/report/log only, never ONNX/PT/engine/private binary copies; Luna provides verified explicit staging paths after checking actual output tree. Do not git-add whole output recursively without excluding private files.

Luna does not SSH: user executes, pushes artifacts, Luna audits canonical files/source/input links and records **L2A-029**, then Astra reviews. Required evidence: successful requested model count OR precise partial failure, input/ONNX hashes, schema/shape/dtype/opset, source-to-Conv targets, end2end/effective arg evidence and unresolved semantics, exact current bindings, dependency/producer hashes and no-build flags. No reported success can automatically authorize84builds. Edge lane continues independently.

User xác nhận5edge devices sẵn và có SSH. Task/authorization riêng nằm `docs/ASTRA_TO_LUNA1_EDGE.md`, entry E2L1-001. Luna1 owner edge-only runner/tests/docs/results, không sửa confirmation runner/config/inbox Luna. Luna chính commit/push A2L-026 này; Luna1 làm branch/worktree riêng, không switch branch trong shared active worktree. Mỗi bên push explicit scoped files, không git add-all/reset/clean. Luna1 có thể chạy SSH read-only inventory ngay khi operator cung cấp aliases; server Luna vẫn operator-mediated.

### Parallel lane Luna 1 — edge inventory GO, không tranh file ownership

Không chờ nhau: Luna graph implementation / operator artifact transfer / Luna1 edge inventory. Chỉ gate phụ thuộc bằng chứng mới phải chờ: graph execution sau code review; confirmation sau graph/parser evidence; edge scored benchmark sau model/backend/protocol selection. Không dùng readiness thành lý do ép benchmark mọi15models×5devices.

## A2L-029 — review d0eff75: fix nested indices; inspect existing v2 ONNX before another export

Reviewed `d0eff75b7042a209501061ce712c2f68c6fc73d6`, L2A-029/030 on 2026-09-16. Astra independently reran graph targeted tests: **20/20 PASS**. This is CPU fixture evidence only. No server SSH, frozen-model export, ONNX binary inspection, GPU or TensorRT execution was performed by Astra. Actual v2 ONNX/schema artifacts are not present in this local graph worktree; the server observations in L2A-030 remain operator/Luna-reported evidence.

**Decision: CHANGES REQUESTED for wrapper mapping; NO-GO another full export at this revision. GO local repair/tests and bounded operator-run read-only analysis of the preserved v2 ONNX.** This is not a GPU-availability restriction. Do not spend another export attempt on a mapping bug that can be tested from names and the existing graph.

### G4 — nested wrapper uses scale index in place of block index

At `_exporter_wrapper_aliases`, the new nested alias inserts `[scale_index, branch, scale_index, scale_index, branch]`. The second wrapper must represent the actual nested source container, not repeat the scale index. The current fixture only covers scale=0/block=0, where the two indices happen to agree.

Astra extended the existing synthetic fixture with the same reported wrapper structure across scale=0/1/2 and block=0/1 (inner Conv index=0). Four of six cases fail with `source_conv_match_count:...:0`; only (0,0) and (1,1) pass. Examples:

- Source `model.23.one2one_cv3.1.0.0.conv`; container-form node `/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/conv/Conv`. Code incorrectly generates intermediate `one2one_cv3.1.1`.
- Source `model.23.one2one_cv3.0.1.0.conv`; container-form node `/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/conv/Conv`. Code incorrectly generates intermediate `one2one_cv3.0.0`.

These are independently reproduced **fixture counterexamples**, not claims that Astra inspected those exact node bytes on SERVER-01. Derive nested aliases from source path segments at their actual levels. Keep explicit locked branch patterns, exact-one-Conv matching, output reachability, and terminal `.2` handling; no substring/shape fallback. Reject malformed or ambiguous aliases. Test all three scales, both nested blocks, both inner Conv leaves when present in accepted source inventory, terminal leaves, duplicate matches, and inactive branches. Check the complete accepted source inventory rather than only one convenient leaf. Do not alter weights, end2end, export arguments or calibration design.

### G5 — use existing exported graph as evidence, not another blind retry

Operator-mediated CPU diagnostic is authorized after Luna's local fixes/tests, **without authorizing a new export**. Luna may supply a small reviewed-in-scope diagnostic command/helper that only:

1. Reads the existing `results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2/models/{yolov8n,yolo26n}/model.onnx`, corresponding preserved prepare-plan/model/failure records, and canonical accepted readiness/config. Resolve exact paths first; if absent, stop rather than export a substitute. Do not modify v1/v2 or their failure records.
2. Runs existing ONNX checker/shape inference and `audit_graph_mapping` in memory, with CPU/CUDA-hidden environment. Record actual ONNX SHA256 before/after, file size, code/helper/config/readiness hashes, source path and available original-export provenance. A newly computed hash is an observation, not proof of missing historical export provenance.
3. Writes exclusive new JSON diagnostics under `results/measurement_audit_v1/precision_head_confirmation_graph_audit_v3/`: actual Conv names and source-to-node matches, I/O/opsets, post-merge node topology/attributes/shapes and relevant small shape/index constants, mapping errors even when unresolved, and adapter/recipe limitations. Do not publish trained weight tensors or model binaries. Save diagnostics before raising for unresolved mapping. Status must say audit-only, `export_performed=false`, `build_performed=false`, `scored_run_authorized=false`; do not manufacture a successful original prepare manifest.
4. Does not invoke YOLO/exporter/model forward, calibration loader, TensorRT, inference, package installation or download. No GPU-idle prerequisite for this CPU inspection. Fresh output root, foreground, commands each on one physical line; Luna supplies commands, user executes and pushes scoped JSON/logs, Luna audits canonical artifacts.

Prefer a short diagnostic helper reusing existing functions over a new orchestration framework. Tests must prove the diagnostic does not dispatch exporter/build, preserves failed inputs and persists unresolved evidence. This narrow read-only-input diagnostic is GO once those local checks pass; an unbounded rerun of `prepare_precision_head_confirmation_graph.py` is not covered. If binary/structural evidence is already available through the operator, inspect it directly instead of adding redundant tooling.

The six additional end2end operators are accepted as a **provisional structural audit extension**, not numeric proof. Existing fixture output shapes/selection paths are synthetic and do not certify box/score/class semantics. Real graph mapping, producer-output equivalence, preprocessing trace and TensorRT compatibility remain later evidence requirements before scored work; no extra numerical threshold is introduced here.

### Handoff

Luna fixes G4, adds complete index tests and the bounded diagnostic path if needed, updates protocol, records **L2A-031**, and pushes scoped files including this entry. Provide exact CPU diagnostic commands and artifact staging paths, not a v3 full-export command. After operator diagnostic artifacts, append a separate evidence report for Astra. Full export v3, calibration-cache generation, 84 builders/78 captures and all-15 scaling remain NO-GO pending that review. Do not repeat successful v8 export merely to investigate v26 naming. Edge Luna1 work is independent. Astra leaves this file unstaged; Luna owns commit/push.

## A2L-030 — accept G5 collection; resolve missing scalar metadata, not a model change

Review 2026-09-17: artifact commit `5b410204d12751845c9f6f38b7d9d4c883ab8c5e`, report `5ff517e71bc96d0580b006dc3a05dbd669b8c492`, implementation `ab8dbfb`. Astra reran graph + diagnostic tests **24/24 PASS**. Exactly five Git artifacts; per-model reports equal the corresponding manifest objects. Four recorded source/config hashes match canonical code at ab8dbfb. ONNX before/after hash and size equality is present for both models; no local binary exists to independently rehash. No export/forward/GPU/TensorRT was run by Astra.

**ACCEPT audit-only evidence collection.** v8 structural mapping verified; v26 source-to-Conv matches now complete (bbox9/class15), with one remaining structural error at `/model.23/Mod`. This is not evidence of numerical model failure. GO bounded local loader repair and subsequent operator CPU re-audit of the same binaries; no full export, no scored matrix.

### G6 — the direct failure is missing constant shape metadata

Actual v3 artifact records Mod inputs `/model.23/TopK_1_output_1` shape `[1,300]` and `/model.23/Constant_20_output_0` shape `null`, output `[1,300]`, `fmod=0`. The second input has recorded constant value **3**. `_load_onnx` reads initializer values but `graph_tensor_shapes`/dtype collection only walk graph input/output/value_info; initializer metadata can therefore be omitted. Do not replace unknown shapes with scalars globally.

Astra replayed the exact Mod node and recorded constant through `_downstream_semantic_audit`: null shape reproduces `mod_semantics_unresolved`; adding only `shape=[]` to this constant in memory makes the isolated structural check pass. This is a JSON-based diagnostic, not binary verification or producer numeric equivalence. Local reviewer environment lacks ONNX, so a real-protobuf fixture was not executed by Astra.

Official ONNX references checked for this diagnosis: [Mod version13, applicable at opset17](https://onnx.ai/onnx/operators/onnx__Mod.html#mod-13) supports multidirectional broadcasting; [ONNX IR tensor/static-shape definition](https://onnx.ai/onnx/repo-docs/IR.html#static-tensor-shapes) distinguishes explicit scalar `[]` from missing/unknown rank. Scalar3 modulo on nonnegative integer class-selection indices is consistent with extracting one of three class indices, but full selection/output equivalence remains to be validated, not inferred from shape alone.

Luna implementation scope:

1. Collect initializer and Constant tensor **actual dims and data_type** from protobuf into shape/dtype metadata, preserving explicit `[]`, `[1]`, zero dimensions, and missing rank as different cases. Handle metadata conflicts explicitly instead of silently overriding. Do not derive dims from a flattened value list or hardcode `/Constant_20` as scalar. Preserve existing numerical graph and binary bytes.
2. Persist small constant metadata and Mod input/output types/value/lineage evidence in the diagnostic. Verify the observed supported integer mode and divisor against the actual three-class contract; unknown types/controls remain unresolved. Do not relax unknown-op or shape rules globally. Static structural acceptance is not numeric certification.
3. Tests: scalar initializer3 + `[1,300]` indices; one-element vector; missing shape; zero dimension; conflicting metadata; wrong/zero divisor; unsupported mode/type; real checker/shape-inference synthetic ONNX fixture when existing local dependencies permit. No frozen-model export, no auto-install to get this test. Keep explicit skips reported rather than treating them as passes. Retain all G4 index/ambiguity tests.
4. After local tests pass, GO operator-run **same audit-only helper** against unchanged v2 binaries, output `results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4/`, exclusive fresh output, CPU hidden/foreground. Bind expected v8 ONNX `e22d53bbeb333f44783535d911d5e318ebb7d500e8fcb3d1cbb1284f5248d603` and v26 `1b2467ccd62bd1e53f3bde3e3f22e1b42129711d3e368a4b4666d025099ce5cc` before and after; mismatch stops. No forward/export/cache/build/scoring or v2/v3 overwrite. This entry authorizes that bounded re-audit after the fix/tests, not a new full export; Luna supplies one-line commands, user runs server and pushes artifacts.

### Correct L2A-032 hash wording without rewriting historical artifacts

All five SHA256 values quoted in L2A-032 match **Windows working-tree CRLF bytes**, not canonical Git blobs. Astra computed the canonical hashes from raw `git show 5b410204:<path>` bytes:

| Relative file | Canonical Git blob SHA256 |
|---|---|
| audit_plan.json | b6a4ec4ba2e122cf988915dd3103f0e80c2967849499c4c0fab98f0629e9e9d6 |
| graph_audit_manifest.json | e7aaa6a80158b1b341aee0867b4adf870bbb6b3b07fa47d704d8b70f8ec69d8d |
| models/yolov8n/graph_audit.json | 588364af696fc53953b9a00bc1d89eb55cad349e667ffded789c31627f76dd04 |
| models/yolo26n/graph_audit.json | 0112682f9295b65d9d3d371a85e6c09b76fd9946e508109b6d354407bbe175ab |
| report.md | 60341ba0aa8547e29e23f14eb5a64df391a41eb3d56cabaf3b1fb25021509f84 |

Append a correction distinguishing canonical Git, local checkout and server bytes. Do not claim direct original-server file-byte equality without a server file hash inventory; internal ONNX hash observations and matching recorded code hashes remain valid evidence. No mass line-ending rewrite or global Git config change needed.

Luna records **L2A-033** for implementation/tests/hash correction and exact diagnostic command, then an artifact addendum after operator run. Push scoped files including this entry. No additional full-export gate cycle is needed for this bounded read-only-input diagnostic. Next numerical/preprocess/TRT work needs a separate reviewed package; plan it in parallel if useful but do not execute it automatically. Astra leaves the handoff unstaged.

## A2L-031 — accept graph-v4 structural gate; implement bounded CPU numerical/preprocessing verification

Review 2026-09-17: artifact `6780b813c5f1cb2d915b72832eedd79525eaecbd`, implementation `0210a26`, report `a292c956a5ea8ba21c7b5ff727c62de3dc7e5dc3` / L2A-034. Astra independently verified all **5 canonical artifact SHA256 values** against the report; both standalone model reports equal their manifest entries; all4 recorded implementation/config hashes match canonical code. Both models record mapping verified, no mapping errors and expected/before/after ONNX hashes equal. YOLO26 Mod has explicit scalar3/int64, fmod0 and compatible integer input/output lineage. Astra reran graph/diagnostic suite: **27 run, 26 passed, 1 explicit skip** (local ONNX dependency absent). No local real-protobuf execution, frozen-model forward, server SSH or GPU execution was performed by Astra.

**ACCEPT G5/G6 and close the structural graph-mapping gate for these exact ONNX binaries.** Do not repeat graph exports/audits absent a changed input or new concrete issue. Acceptance is structural, not proof of source-output numerical equivalence, calibration preprocessing equivalence or TensorRT execution. Preserve v2/v3/v4 and existing frozen weights unchanged.

### Next task: one complete implementation + protocol package, not another report-only round

Owner main Luna. **GO local code/CPU fixture tests/protocol; actual frozen-model numerical execution remains server-operator work after Astra implementation review.** This package must cover both CPU numerical checks and preprocessing evidence, so the operator can eventually run one foreground command rather than a sequence of separate experiments. No TensorRT import/parser/build in this package, no new ONNX export, cache generation, AP evaluation or bootstrap. No official test or negative-test image/label reads.

Proposed owned files: `scripts/verify_precision_head_confirmation_numeric.py`, dedicated tests, `docs/PRECISION_HEAD_CONFIRMATION_NUMERIC_V1.md`, additive config if needed. Reuse audited helpers without invoking their GPU/export lifecycle or changing accepted historical helper/config hashes to make results pass. Keep parent CPU-only and model-specific isolated children. Existing config/schedule remains84 builders/78 scored captures; this CPU diagnostic is separately accounted and does not open them.

### Inputs and fixed workload

- Bind accepted readiness and graph-v4 commit/files, exact frozen checkpoints and the two accepted ONNX hashes from A2L-030. Verify actual bytes before/after; no substitute export or model download. Recheck required current source/runtime versions and CPU execution provider availability; no dependency auto-install or fallback to GPU. CPU-only does not require GPU idle.
- Use a small deterministic fixture of **the first8 distinct train image IDs in canonical U42 manifest order**, bound to accepted train-only materialization/content hashes, for both models. No image/label/outcome-based selection. Record IDs, original image sizes, content hashes and selection rule before any forward. Do not claim this small fixture is a dataset-wide accuracy test.
- Preprocessing trace additionally covers the first image in each of U42/U43/U44 using the actual pinned producer path. Reuse tensor traces when IDs coincide, but keep selection bindings. Inspect full list ordering through metadata if needed; do not materialize all3072 calibration tensors just to resolve the recipe. No calibration cache is created.

### P1 — preprocessing equivalence and concrete recipe

Trace actual decode/color order, resize/letterbox ratio and padding, interpolation/rounding, layout, dtype/range/normalization, batch shape and deterministic ordering. Bind exact source/version hashes of functions used. Capture stage shapes/dtypes and tensor-byte hashes; retain small diagnostic tensors server-side only where needed. Compare the tensor supplied to the source reference with that supplied to ONNX; they must be the same verified tensor for the numerical comparison.

Do not assume a conventional LetterBox recipe matches the pinned Ultralytics producer. Inspect the real code path and separate calibration loader preprocessing from dev/inference preprocessing; if they legitimately differ, record both and the intended eventual pipeline. Do not silently change preprocessing to force parity or call unresolved fields verified. The old `uniform_build_repeat.prepare` invokes exports/environment logic: never dispatch it as a convenient loader. Use scoped producer functionality and no writable cache side effects in original data directories; private scratch if unavoidable.

### P2 — source FP32 versus existing ONNX on CPU

Implement PyTorch CPU reference and explicitly selected ONNX CPU runtime, both eval/no-grad, same input tensor, pinned versions/providers/options. Work on an in-memory/private reference only; preserve checkpoint and graph bytes. Distinguish native head output from export-mode output. Inspect the pinned export semantics (including fusion, output packing, coordinate convention and end2end behavior) and record any reference-side export-mode flags needed for a like-for-like comparison. No regeneration of the ONNX is authorized. Do not compare v8 raw channels with post-NMS detections, or v26 one2many debug output with its one2one primary output.

For v8 record raw `[1,7,8400]` channel-wise box/score differences; for v26 record `[1,300,6]` detections and class/score/box differences. Define ranking/tie handling before results: raw row comparison must be retained, and any justified tie-aware matching is separately reported, class-aware and one-to-one, not arbitrary rematching to hide changed outputs. Preserve confidence and max-det semantics; no double NMS or model-policy tuning. Coordinates may be outside image bounds before native clipping: validate at the correct pipeline stage, do not silently clamp raw output just to meet the earlier declared `[0,640]` assumption. Clarify any stage-specific validator correction in this package.

Protocol must propose exact numerical tolerances, comparison domains, finite/dtype/class/integer requirements, and pass/unresolved/fail rules **before server outputs are observed**, with justification. No equality-of-engine-hash requirement and no AP threshold for this diagnostic. Report maxima/quantiles/counts and offending locations, not only allclose. Any tie-dependent ambiguity remains explicit. Shape or aggregate mAP equality alone cannot certify equivalence. Small CPU synthetic fixtures may exercise matching/math; frozen-model forward still waits for server authorization.

### Outputs, tests and handoff

Exclusive new root proposed `results/measurement_audit_v1/precision_head_confirmation_numeric_v1/`, manifest linking all reviewed inputs/code, fixture plan, actual provider/environment, source/ONNX hash observations, preprocessing traces, per-model/per-image comparison summaries and complete failure diagnostics. Small tensors/raw outputs may remain private server artifacts with hashes; JSON/report/log and explicit file inventory are publishable. Record forward counts separately; `export_performed=false`, `build_performed=false`, `scored_run_authorized=false`. Do not promote graph-only fields retroactively in earlier artifacts.

Tests should cover real helper boundaries with CPU stubs, identical tensor input to both paths, provider/weight/ONNX/fixture mismatches, wrong native-output branch, same-shape numerical corruption, ranking ties and wrong classes, finite values, preprocessing discrepancies, no-overwrite, partial failure persistence, no-export/build dispatch and CPU parent/child lifecycle. Test intentional failures as well as success. Missing ONNX fixture dependency is an explicit skip, not pass; dataset-dependent old suite failures remain distinguished from scoped test results. Do not spend this task rewriting unrelated tests or older studies.

Luna writes **L2A-035** with implementation commit, tests (pass/fail/skip separately), locked proposed tolerances, CPU resource/time estimate, exact candidate foreground command and scoped artifact instructions. Include this entry and previously unstaged Astra instructions in the scoped push, without modifying their content. **Do not give a command as authorized for execution until Astra reviews this complete package.** User runs server; Luna cannot SSH. Astra owns review/gates, Luna implementation/push.

Luna1 continues E2L1-006 local Jetson adapter/smoke package independently; no new edge task needed from this graph acceptance. No device build/inference or78-scored matrix is authorized by A2L-031. After numerical/preprocess evidence is accepted, TensorRT compatibility and bounded pilot can be considered together rather than automatically scaling.

## A2L-032 — numeric implementation review: repair runtime boundaries before operator execution

Reviewed 2026-09-17: implementation `4b9dc401143fae1ca5b200019ac725ffa567adbd`, handoff `b6eb1741237e09c8c24d3b0c85c972c443f308d1`, L2A-035. Astra independently reran **16/16 numeric tests PASS** with CUDA hidden. No frozen-model forward, ONNX session, server execution, export or GPU work was performed. The tests establish the covered CPU contracts, not server readiness. **CHANGES REQUESTED; candidate server command is not authorized yet.** This is implementation repair, not a change to the research design or another graph-audit round.

### N1 — canonical image resolution and real-layout regression

`verify_bound_image()` and the child trace loop currently resolve `train/images/00006.jpg` against the repository root. Accepted selection image paths are relative to `data/processed/cctsdb2021_clean`, not the repository. Astra reproduced `FileNotFoundError` from the function while `D:/Research/paper/data/processed/cctsdb2021_clean/train/images/00006.jpg` exists and `D:/Research/paper/train/images/00006.jpg` does not. The parent integration test mocks `verify_bound_image`, so it misses this failure.

Use one resolver bound to the accepted dataset root in BOTH parent verification and child decoding; keep materialized-selection bindings separate. Do not copy images into a wrong root to satisfy this code. Add real-layout CPU fixture tests (small synthetic bytes in the canonical directory layout are sufficient), wrong-root decoy, traversal/escape and changed-byte rejection. Preserve the first-eight rule and all U42/U43/U44 bindings even when tensor computation is deduplicated. Verify source/materialized bytes again after work, or correct the protocol's unsupported claim that all input bytes are rechecked.

### N2 — preserve the actual failed-child evidence and finish the run manifest

On a child exception, `run_child()` exclusively creates `models/<model>/failure.json`. When `numeric_report.json` is absent, `run_parent()` attempts to exclusively create that SAME failure path. That second write will raise instead of producing the promised final failure manifest/report. Existing tests exercise parent success and child failure separately, not this combined path.

Parent must consume/validate an existing child failure record without overwriting it, and only create a fallback when the child produced no report at all. Test actual parent-to-failing-child artifact lifecycle, crash-before-report and success paths, with a final accurate inventory in each case. Keep partial per-image/trace evidence if a later comparison raises. Separate execution completion from numeric verdict in the aggregate manifest: execution `completed` must not be interpreted as a numeric pass or authorization. Aggregate `pass/fail/unresolved/not_observed` explicitly; a completed diagnostic containing disagreement remains review evidence, not successful equivalence. No resume/overwrite of old roots.

Also fix the advertised single-model CLI: the immutable graph artifact legitimately contains BOTH models, but `set(rows) != set(models)` currently rejects either single-model selection. Validate the full accepted artifact, then select the requested subset (or remove the unsupported option consistently). Add both-model and single-model tests without replacing the graph validator with an unconditional mock.

### N3 — finish P1/P2 evidence, not just labels for it

1. Current traces execute inference preprocessing only. Hashing calibration producer source and setting `calibration_loader_called=false` is honest but does not fulfill the requested scoped calibration preprocessing trace. Inspect the pinned recipe, then exercise its actual image/transform components on the three accepted anchors in a bounded CPU fixture without dispatching an exporter/full calibration loader, generating a cache, reading test/negative data or writing dataset caches. Private scratch is allowed. Record inference and calibration stage outputs/parameters separately, including preliminary resize and final normalization where applicable; differences are observations, not reasons to alter either recipe. Keep each selection-to-image binding. Source inspection alone must remain `inspected`, not `traced/verified`.
2. Record native versus export-reference semantics explicitly. Astra inspected the installed Ultralytics 8.4.102 source: exporter uses a private model copy, `float()`, fusion and head export/format/dynamic/max-det settings; current child records none of that and uses only `.to('cpu').eval()`. This is NOT evidence that native outputs necessarily differ: at 8400 anchors the observed native/export top-k limits can agree. Inspect and document the exact accepted ONNX settings, fusion/coordinate/packing behavior and reference flags. Keep the existing native-reference comparison if justified; if an export-equivalent in-memory reference is necessary, propose the exact change and forward counts before running. No new ONNX export and no extra frozen forward automatically authorized. Ensure FP32 explicitly. Preserve one2one selection and fixed-row diagnostics; ties must not become an unsupported claim of arithmetic failure.
3. `yolov8n_raw_primary_channelwise` currently labels one whole-array statistic, not per-channel results; v26 boxes and score are also pooled. Report box channels and score channels separately, preserving overall verdict. Define the tolerance equation against the REFERENCE magnitude (`abs(observed-reference) <= atol + rtol*abs(reference)`) and use consistent `isclose` argument order. Keep finite disagreement diagnostics representable when reference is zero; do not lose an entire report to overflowing relative-error quantiles. Add zero/near-zero and same-shape corruption tests.

The proposed `rtol=1e-4, atol=1e-5` may remain the fixed strict diagnostic rule, not a general guarantee of cross-runtime equivalence. ONNX Runtime version/provider/options may be recorded as the observed CPU prerequisite for this small diagnostic; do not install or silently substitute a provider. No post-output tolerance widening. No alteration of accepted readiness/graph/config historical hashes.

### Deliver once, then review for the single operator run

GO Luna local code, source inspection, CPU synthetic/component tests and protocol repair for N1-N3 together. No server SSH or frozen-model execution. Record **L2A-036** with addressed findings, tests pass/fail/skip, implementation commit, exact unchanged/proposed numerical decisions, and candidate foreground command. Include this entry unchanged in the scoped push; Astra leaves it unstaged. No need to ask the operator to rerun inventory or wait for GPU idle: the eventual diagnostic is CPU-only. Output v1 remains a candidate only if absent; preserve any historical/partial artifacts.

Luna1 receives independent E2L1-007 adapter repairs and can work in parallel. Graph-v4 structural acceptance stands. Numerical execution, TensorRT compatibility/builds, the scored matrix and edge inference remain unopened until their respective reviewed implementation is ready.

## A2L-033 — accept core N1-N3 repairs; two bounded corrections and conditional CPU-run GO

Review 2026-09-17 of implementation `6495abd`, handoff `f2be9b653f9c9c35bf2c8ada0c808be3b3bad601` / L2A-036. Astra independently ran **23/23 numeric tests PASS; graph 22 PASS + 1 explicit skip; graph-audit 4/4 PASS; py_compile and git diff --check PASS**. Tests include actual pinned preprocessing components on synthetic CPU fixtures, not frozen-model execution or server evidence. No server SSH, model forward, ONNX session, export, TensorRT or GPU execution by Astra.

**ACCEPT canonical source resolution, retained selection bindings, failed-child consumption, aggregate execution/numeric separation, single-model graph selection, partial artifacts, FP32 native reference and split box/score reporting.** Structural graph acceptance remains closed. No new model/reference/export-mode design is requested. Two concrete residual errors below must be corrected before the bounded operator run; do not rerun graph preparation or rework the research protocol.

### C1 — enforce the already declared reference-relative equation

The implementation still calls `np.isclose(reference, observed, ...)`, whereas NumPy scales relative tolerance by its SECOND argument. The report claims `abs(observed-reference) <= atol + rtol*abs(reference)`. Astra reproduced a contradictory verdict using float32 reference `0.0010002674534916878`, observed `0.0010103675303980708`: current helper returns pass, but `np.isclose(observed, reference, rtol=1e-4, atol=1e-5)` is false.

Change to `np.isclose(observed, reference, ...)` (or an explicitly tested implementation of the same declared equation); correct `tolerance_order` metadata, protocol and L2A-036's "reference first" statement through an additive correction. Do not change `rtol=1e-4` or `atol=1e-5`. Add the exact asymmetric-boundary regression, in addition to zero/near-zero diagnostics. This repairs the specified rule, not a post-result change; no server numerical results have been observed.

### C2 — isolate the loader's .npy cache branch

`trace_calibration_preprocess` passes `npy_files=[path.with_suffix('.npy')]` to `YOLODataset.load_image`. In pinned 8.4.102 that method checks the .npy path independently of `cache=False`, may load it, and may unlink it if corrupt or wrong-channel. Astra's CPU mock supplied a .npy result: `np.load` was called once while the emitted evidence said `image_cache_read=false`. Current JPEG content binding therefore does not necessarily bind the consumed image, and a corrupt adjacent cache could be modified.

Keep the actual bounded `load_image` resize path, but direct its .npy candidate into a fresh private temporary directory under this run's scratch, guaranteed absent for the call. Never pass an original dataset/materialization cache path, never delete an original .npy, and never dispatch the full loader/exporter. Preserve the real verified JPEG path for image decoding. Alternatively a clearly scoped, tested no-cache producer wrapper may be used without globally monkeypatching Ultralytics. Record the actual cache isolation mechanism and call evidence; test valid, stale/wrong-channel and corrupt adjacent dataset .npy fixtures all remain byte-identical and are not consumed. Original image bytes and transformations must still be the source. Temporary scratch may be excluded from publishable artifacts, with its role recorded.

Keep `Format._format_img_equivalent` labelled equivalent/manual, not an actual Format producer call. The current RGB/CHW arithmetic is a bounded recipe trace, not proof of the full future calibration loader's tensor stream; no new full-loader execution is required for this CPU diagnostic. Keep native-versus-fused-export limitations explicit. Historical graph/config/checkpoint bytes stay frozen.

### Conditional authorization — no extra report-only review round for these two exact repairs

Luna may implement C1/C2, tests and consistent documentation locally, then push **L2A-037** plus this entry unchanged. **After C1/C2 behavioral tests and the numeric/graph/audit regression suites pass (same existing ONNX-fixture skip allowed), with no other numerical/input/runtime changes, Luna may provide the operator the reviewed CPU command below, pinned to the resulting pushed commit.** This is Astra's conditional GO for that specific small diagnostic; it is not a claim that the as-reviewed `6495abd` is runnable. If repair expands beyond C1/C2 or tests fail for a new reason, stop and report before server execution. Luna must not SSH the server.

Before execution, operator pulls fast-forward, verifies exact HEAD against Luna's supplied full commit and confirms the new numeric output directory is absent. Do not use `git reset/clean`, overwrite an existing run or install packages. Bind unchanged accepted readiness/graph-v4, two checkpoint/ONNX hashes, first eight U42 train IDs and three U42/U43/U44 anchors. Permit only **8 native CPU + 8 ONNX CPU forwards per model (32 total)** and the bounded preprocessing traces. No GPU-idle/desktop guard is needed: CUDA hidden, OMP/MKL2, ORT CPU provider, same existing pinned environment; no TensorRT, export, cache generation, training, AP or test/negative images. Runtime mismatch/missing package is a reported prerequisite, not permission to install or relax version locks.

Foreground command (Luna supplies the separate exact-commit pull/check line after pushing the fixes):

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 YOLO_AUTOINSTALL=0 ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1 PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 local/g0_size_env/bin/python scripts/verify_precision_head_confirmation_numeric.py --readiness-root results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2 --graph-audit-root results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4 --source-root results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2 --model all --out-dir results/measurement_audit_v1/precision_head_confirmation_numeric_v1
```

No nohup by default. Retain partial failures; no automatic retry/new model/reference or tolerance adjustment. Operator pushes publishable JSON/report/log evidence including partials, NOT model/ONNX/tensor binaries or caches. Luna checks canonical artifact hashes, actual providers/versions, before/after inputs, all forward counts, C1 equation/C2 isolation, preprocessing and per-image verdicts, then reports an L2A-037 artifact addendum for Astra. Execution `completed` and exit0 alone are not numerical PASS. Any numeric fail/unresolved is a diagnostic result requiring interpretation; no automatic TensorRT/scored matrix authorization follows.

Luna1's next local real-adapter preparation can proceed simultaneously; neither lane waits for the other. Astra leaves this entry unstaged for Luna to commit/push.

## A2L-034 — accept canonical schema repair; GO bounded operator CPU diagnostic

Reviewed 2026-09-17: `e8918de783d575116398b632e2b25d07f9d47f12`, L2A-038, incorporating `ea56053` / L2A-037 C1/C2 repairs. The latest code is in `D:/Research/paper-numeric-fix`; `paper-luna-graph` was still at `e7f4fa7` during review. Astra fetched the reported commit without switching/resetting either worktree and reviewed/tested the actual numeric-fix worktree. Keep that ownership distinction when committing this inbox.

**ACCEPT the schema correction and C1/C2. GO the bounded CPU run specified in A2L-033.** Source metadata uses `image_sha256/image_bytes`, materialized metadata uses `materialized_sha256/bytes`; these are not interchangeable schemas. The earlier pre-forward failure was an implementation/schema mismatch, not evidence of changed image contents. Do not rerun graph exports or relax accepted artifact hashes.

Independent evidence:

- Numeric suite: **27/27 PASS**, including actual synthetic-image preprocessing components in Astra's existing local environment, canonical field validation, asymmetric reference-relative tolerance and isolated .npy cases. This supersedes neither Luna's own environment-specific skip report nor server runtime checks; report each environment separately.
- Graph regression: **22 PASS / 1 explicit skip**; graph-audit regression: **4/4 PASS**.
- Astra read all five accepted readiness files directly from canonical Git blobs and verified their accepted SHA256 values. Feeding the REAL canonical manifest into the new fixture adapter yields exactly `00006,00009,00028,00036,00054,00061,00098,00104`; anchors U42=`00006`, U43=`00029`, U44=`00000`.
- All those source JPEG files under `D:/Research/paper/data/processed/cctsdb2021_clean` match the selected canonical source hashes AND sizes. No frozen-model forward, ONNX runtime session, server execution or GPU work was performed by Astra. Materialized server copies remain subject to the runner's checks.
- The direct Windows checkout readiness validator rejects CRLF-normalized checkout bytes, as expected from the historical strict byte binding. Astra used canonical Git blobs for the above audit and did not rewrite files, relax the old helper or claim a full local parent run. The Linux server runner must validate its actual accepted artifact bytes normally.

### Operator handoff, now authorized

Main Luna supplies the foreground pull/exact-commit check and A2L-033 run command to the user. Reviewed executable code is `e8918de783d575116398b632e2b25d07f9d47f12`; if Luna subsequently pushes this DOC-ONLY entry, use the actual resulting commit and verify numeric runner/tests/helper/config code is unchanged from the reviewed commit. No invented commit placeholder in the executable handoff.

Output `results/measurement_audit_v1/precision_head_confirmation_numeric_v1` must still be absent. According to L2A-038 the earlier attempt stopped before output creation; independently check that on server. If any output now exists, preserve it and inspect rather than delete, overwrite, silently resume or rerun a completed diagnostic. No `nohup` by default; no GPU-idle wait. CUDA hidden, CPU provider, threads limited as already specified. Scope remains **32 total model forwards: eight source + eight ONNX for each of two models**, plus bounded preprocessing anchors. No package auto-install, ONNX export, calibration cache, TensorRT, AP, test/negative-set use or matrix expansion.

After running, user pushes publishable evidence including partials/logs. Luna checks canonical hash/provenance/actual runtime, source and materialized before/after bindings, input tensor identity, numerical verdict separate from execution status, preprocessing and forward counts; append **L2A-039** with actual result, including fail/unresolved evidence. Do not treat DONE/exit0 as equivalence PASS, widen tolerances, change reference mode or launch follow-up experiments automatically.

Astra leaves this entry unstaged for main Luna to commit/push. Luna1's runtime repair E2L1-009 is independent and does not block this GO.

## A2L-035 — accept failed-run audit; repair canonical head adapter and exercise the complete CPU path

Review 2026-09-17: artifact `22c67a910b4d82ef7482ec3ed752723406856057`, L2A-039 `613fd8c504662e69994c9572fb31a99ff68f4f94`. Astra independently enumerated/read the **7 canonical Git artifacts**, confirmed plan code `e8918de`, two pre-forward head-identity failures, `execution_status=failed` and numeric verdict `not_observed`. No numerical equivalence result exists; these are implementation failures, not evidence against either model/ONNX. Keep numeric_v1 immutable.

Astra reproduced the guard failure on BOTH models with CPU doubles whose head identity exactly matches the REAL accepted contract. Canonical `.head` fields are `type/index/end2end` (v8 Detect/22/false, v26 Detect/23/true); the consumer requests `.expected_contract` with different keys. `build_reference_semantics` repeats the same erroneous access and emits an empty head contract. The previous review validated the image-schema boundary but did not cover this child boundary; review/test coverage was insufficient. Do not fix this by weakening identity validation or editing accepted metadata.

### H1 — one canonical contract adapter, no invented fallback schema

Use one explicit adapter from accepted `head.type/index/end2end` to the guard and reference metadata. Validate required fields/types/model association before any runtime dispatch; missing/malformed accepted metadata must produce a schema-specific error with expected and observed values, not an empty-dict default. Preserve canonical branch identity, class/output contracts and frozen input hashes. Cross-check any config identity separately rather than treating config and readiness records as the same shape. Find and replace every inappropriate `expected_contract` consumer in this runner, not just the first failing line. Tests must use the canonical readiness/failed-plan records from Git, not a hand-written record matching the bug.

### H2 — distinguish child partials from run inventory and record stage/count evidence

`run_child` enumerates the WHOLE output root into each child's `partial_files`. That explains why yolo26 reports the previous model's failure/log: it is an inventory scope bug, not evidence of shared model tensors, overwritten artifacts or cross-model computation. Enumerate only the current model's owned partial/report paths there. Keep `numeric_plan.json` as an explicitly shared reference and let the parent own the complete run inventory/log index. Retain the original historical failure records unchanged.

Add success/failure stage markers and explicit attempted/completed source/ONNX forward counts that survive exceptions. In v1 the counts are absent; zero is inferred from the error location and code ordering, not directly measured counters. Test failure before runtime, at head validation, during preprocessing, after a source forward and during/after an ONNX call, plus two sequential failing/successful children. A failed second child must not list the first child's files as its own; final parent inventory must still include both. Preserve no-overwrite and complete failure/report persistence.

### H3 — test the real metadata-to-child path before another operator run

Use the accepted canonical five-file readiness/graph evidence and v1 numeric plan as regression inputs; use actual schema adapters, head validation, fixture planning, output extraction, comparison and report writers. For local integration, inject only the external runtimes/files that cannot be materialized safely: a CPU model/session double returns arrays shaped/structured exactly as canonical `forward_summary` records. Exercise the actual `run_model_child` loop for BOTH models, all eight comparisons and three calibration anchors, then parent aggregation. Do not replace `run_model_child`, `validate_native_head`, `_model_plan`, fixture verification or comparison with unconditional success stubs in the test used to claim this boundary passes. Test malformed head keys and a wrong actual head as intentional failures.

Also GO a bounded **local CPU checkpoint load/structure inspection only**, on the two already available accepted frozen checkpoints after verifying exact SHA256/size, to exercise the corrected guard against real objects. This is not permission for frozen-model forward/export and not a claim of server pinned-runtime equivalence. Record local versions and before/after hashes. If available checkpoint bytes or dependencies do not match, report and use canonical contracts plus faithful doubles; do not download, install, substitute or save/rewrite a checkpoint. No numerical inference is needed to catch this identity error.

Current worktree note: `D:/Research/paper-review039` already showed `M results/yolov8n_cctsdb_clean_s42_v1/weights/best.pt` before Astra edits. Astra did not modify or stage it. Preserve this unrelated state, check bytes before using, and never stage it with the handoff or restore/reset it automatically. Other older worktrees remain separate; avoid branch/worktree churn and record the actual implementation location in the response.

### Conditional rerun gate and fresh output

GO main Luna H1-H3 local implementation/tests and protocol correction together. Append **L2A-040** with exact commit, canonical-fixture end-to-end test evidence, actual versus mocked boundaries, head values and counters. If all H1-H3 tests pass (old explicit graph ONNX-fixture skip remains allowed), the guarded source/runtime/input/tolerance design is unchanged and the only changes are these schema/lifecycle repairs, **Astra conditionally authorizes ONE operator foreground CPU rerun** under the A2L-034 resource limits. No extra report-only review round is required merely to relay these fixes. Any new numerical/reference/input design or failed integration requires review first.

Use fresh output **`results/measurement_audit_v1/precision_head_confirmation_numeric_v2`**, not v1. Keep study/protocol identity separate from attempt/output version and link the preserved v1 failure in the new manifest. Luna supplies exact pushed-commit pull/check plus the existing command with ONLY the reviewed code and `--out-dir .../precision_head_confirmation_numeric_v2`. User runs server; Luna does not SSH. Verify root absence, existing pinned environment and inputs; no nohup, GPU-idle guard, package installation, model substitution or automatic retries. Total planned model calls remain 8 source + 8 ONNX per model, CPU only. No TensorRT/matrix/AP/test-set work.

After operator pushes artifacts, audit canonical evidence and stage/count/partial isolation, then report outcome without equating execution completion to numeric pass. A real numeric disagreement is to be interpreted, not automatically repaired by changing tolerance or reference. Astra leaves this entry unstaged for Luna to commit/push; independent edge work need not wait for this rerun.

## A2L-036 — accept executed numeric evidence; localize disagreement without changing the equivalence gate

Review 2026-09-17: L2A-041 `770a4eb2cee14366e5c8d1b83be3522cd6734aac`, canonical server artifact commit `171f058520f1d8dadbcf7c1633a31c2c7e4327ad`. Astra inspected canonical reports/inventory and reran **34/34 numeric CPU tests PASS** in the existing local environment; no frozen-model forward, server, GPU or export was run by Astra. Worktree: `D:/Research/paper-review039`. Preserve the pre-existing modified YOLOv8n best.pt; it is not part of this handoff.

**ACCEPT numeric_v2 as completed, negative numerical evidence. Do not accept native/ONNX equivalence.** Both models completed eight source and eight ONNX calls with valid reported identities/input bindings. The previous head/schema failure is resolved. Preserve v1 and v2; do not relabel either historical verdict or run the full 32-forward diagnostic again unchanged. TensorRT, calibration, scored confirmation matrix, AP/test evaluation and retraining remain NO-GO.

### F1 — correct the hash reporting boundary, not the artifacts

The three hashes in L2A-041 are Windows CRLF checkout hashes. Astra read canonical bytes with `git show` from the artifact commit and verified that checkout/canonical contents become equal after LF normalization. Append a correction distinguishing these representations; do not rewrite the accepted server evidence or claim those checkout hashes were canonical:

| File | Canonical Git SHA256 |
| --- | --- |
| numeric_manifest.json | `0d28fc3261ad4fa42baa8459c449c50e98777f9c3a43123f288109159523787d` |
| numeric_plan.json | `05216d73d69b0a9f5d621aee2fbd3a444d5e6cdc9a11050a4c6a709970c06831` |
| report.md | `5bcb70399bb634e6e449ea54ad36850405dae6e19d574afb259e88f74abb56eb` |

Keep original checkout hashes labelled as such. Check future linked hashes against their stated byte representation, not normalized text silently substituted for a byte hash. This documentation correction does not invalidate the observed numerical comparisons.

### F2 — interpretation and the next bounded question

YOLOv8n image00009 has two failing box elements `[0,1,8004]` and `[0,1,8014]` out of33,600; all its score elements pass. Its reported box max absolute error is0.00054931640625. This is a small absolute discrepancy, but remains FAIL under the existing elementwise reference-relative criterion. It is not evidence that the discrepancy is clinically/practically irrelevant, nor grounds to widen tolerance. Note that passing other images may have larger maximum absolute error at larger reference values; do not use a maximum alone to decide the relative-tolerance gate.

YOLO26n has 176–240 observed tied scores versus zero reference ties, 815–1,056 box mismatches and92–180 class mismatches per image. Scores passing tolerance means approximate numerical agreement, NOT equal scores or equal selected anchors. Hypotheses include rank permutation, selection changes near ties/small scores, upstream numerical differences, or an incorrect semantic association. These can coexist. Do not state that this is merely harmless row order or that the box decoder itself is wrong before testing corresponding anchors. ONNX TopK specifies lower-index tie breaking: [official TopK specification](https://onnx.ai/onnx/operators/onnx__TopK.html). Check the installed PyTorch/Ultralytics implementation and actual ONNX operator attributes rather than imposing that rule on the reference silently.

### F3 — GO local implementation of a two-case CPU localization package

Implement a separate diagnostic runner/protocol/tests, retaining the accepted numeric runner and strict verdict. Proposed cases are fixed now: **YOLOv8n/00009** (the failing case) and **YOLO26n/00006** (first canonical fixture, not a selected best/worst case). Reuse accepted hashes, actual preprocessing/input-tensor bindings and pinned server CPU runtime. No new images, model/reference mode, fusion changes, export, optimization sweep or threshold search.

The candidate server plan is at most **two native model forwards plus three ONNX session runs**: original v8 once, original v26 once, v26 diagnostic-view once. The diagnostic view may expose existing intermediate graph tensors in a private derived graph; it is not permission to re-export, edit nodes/weights/attributes, replace operators or overwrite the accepted ONNX. Keep original output alongside exposed tensors, hash the derivative and record the exact output additions plus original before/after hashes. Validate graph mapping/shape/dtype/lineage before running. If the derivative changes the original final output relative to the unmodified session, flag instrumentation sensitivity and do not use it as proof of the original execution's internals. No operator-optimization sweeps or additional model calls on failure.

For the native side, inspect the installed source first and obtain intermediate data from that one ordinary forward, through its returned structures or observational hooks. Do not mutate head flags, patch computations, fuse the model or silently run another model pass. Record the exact source code/version and capture sites. Postprocessing of retained tensors is allowed, but label reconstructed selections separately from indices directly observed in the ordinary forward. If the data cannot be obtained within this limit without altering semantics, report the concrete boundary and proposed count before executing.

Required diagnostic outputs:

1. V8: reference/observed scalar values, absolute error, allowed error `1e-5 + 1e-4*abs(reference)`, error/allowance ratio and anchor/class scores for both historical offender indices; all new mismatches must also be counted. Original strict comparison remains visible, even if this rerun differs.
2. V26: corresponding pre-TopK decoded boxes, class logits/probabilities where mapped, shapes/dtypes, finite/zero counts and score magnitude/tie summaries. Compare anchor-aligned tensors BEFORE selection. Do not invent a cross-model anchor contract or compare raw logits to sigmoid scores.
3. Trace both stages of TopK/Gather/class-index arithmetic where present, including actual attributes, class flattening and original anchor indices. Reproduce each side's final boxes/classes/scores from its OWN captured tensors and indices; report exactness/error. Separate same-anchor numerical error, same-selected-set permutation, different-selected-set membership, and unresolved association. Include overlap/counts and bounded examples; no nearest-neighbor match that reuses rows or conceals missing detections.
4. Preserve full fixed-row comparison. Supplemental anchor/class alignment is a diagnostic, not a replacement PASS. Do not discard low-confidence rows, apply NMS, select a score cutoff after seeing results or publish AP from these two train images. Never infer correctness of all eight fixtures from two cases.

Keep full tensors/derived ONNX private on server, with hashes and explicit retention locations; publish bounded strict-JSON summaries, plan, logs, provenance and inventories only. Use a new output root `results/measurement_audit_v1/precision_head_numeric_localization_v1`; preserve partials and original exceptions, never overwrite or auto-retry. Tests must cover permutation-only, changed membership from ties, genuine same-anchor coordinate errors, unequal scores within tolerance, wrong index/class mapping, instrumented-output drift, near-zero reference errors and parent/child failures. Exercise actual analysis/writers with faithful CPU doubles, not unconditional PASS mocks.

### Gate and handoff

GO F1–F3 source inspection, local code, CPU tests and protocol now. Append **L2A-042**, commit/push this entry unchanged with the scoped implementation using NADUNGVN, and supply an exact forward-count/artifact contract plus candidate operator command for review. **No actual frozen-model forward or diagnostic ONNX session yet**; Astra reviews the instrumentation boundary before the user runs the bounded CPU command. Main Luna still does not SSH. CPU execution will not require GPU idle. No full export/numeric rerun or matrix is authorized by this entry.

Luna1's allocation-only edge path proceeds independently under E2L1-011. Neither lane waits for the other's result. Astra leaves this handoff unstaged; Luna owns commit/push.

## A2L-037 — repair native semantic adapter using the real decoder; conditional one-shot CPU localization v2

Reviewed 2026-09-17: L2A-043 `bba984c5b434d0ce2779863a9de75124822f6bd2`, artifact `b8ae581b68863a648bf5377cf5beeaf2e3f0953a`, implementation `c732a4f`. Astra read the seven canonical Git artifacts and actual installed Ultralytics8.4.102 head source, and reran **12/12 localization tests PASS**. Those tests validate arithmetic on fabricated decoded arrays but do not test the actual raw-return-to-decoded adapter; they are insufficient acceptance coverage. No frozen-model inference or GPU was performed by Astra.

Accept L2A-043's qualification. V8 stopped after one native call and zero ONNX calls. V26 completed one native, one original ONNX and one derived ONNX call; original/derived final outputs were recorded exact. Its ONNX-internal structural reconstruction can be retained, but cross-side score/box comparisons and the278/300 selection overlap are inadmissible because their semantic domains differ. Preserve the failed localization_v1 and strict numeric_v2 FAIL; do not turn this diagnostic implementation error into an additional claim that either exported model is incorrect.

Review accountability: A2L-036 explicitly required source inspection and like-for-like tensors; the implemented `_extract_native_preselection` instead renamed raw `boxes/scores` as decoded/probabilities. The review/test boundary did not prevent that error. Fix the producer boundary rather than adding more shape-only mocks. The current inbox also has no intervening recorded run-GO after A2L-036's implementation-only gate. Record the actual operator approval/reference if one exists; do not fabricate or retrospectively backdate authorization. This evidence-recording gap does not justify deleting the run or pretending it did not happen.

### S1 — exact semantics and selected adapter design

Inspected installed source: `ultralytics.nn.modules.head.Detect.forward_head`, `forward`, `_inference`, `_get_decode_boxes`, `decode_bboxes`, `postprocess`, `get_topk_index`. `forward_head` returns raw box parameters and class logits. `_inference` decodes boxes with DFL/anchors/strides and concatenates `scores.sigmoid()`. Shape4 alone does not prove pixel coordinates: YOLO26's raw4-channel box parameters still require decoding. Its end2end decoded format is xyxy; ordinary v8 defaults to xywh, subject to actual head flags. Both are in640x640 input pixel space, not original JPEG coordinates. Record actual reg_max, DFL module, stride/anchors, xyxy, end2end, export, training and agnostic_nms flags; do not overwrite them to match an assumption.

Approved design for this repair:

1. **V8:** use `native_primary [1,7,8400]` from the one ordinary forward directly: channels0:4 are its decoded box reference and4:7 are class probabilities. Do not require raw debug boxes to have4 channels, reshape64 into4, or run an extra model/decoder just for v8. Compare its primary directly to the original ONNX and retain scalar evidence for00009's historical offenders.
2. **V26:** retain the Torch tensors in the actual returned `one2one` dictionary (`boxes`, `scores`, `feats`). Under no_grad on CPU float32, replay the **same head object's installed `_inference(one2one)`** once to obtain `[1,7,8400]` decoded xyxy/probabilities. This is explicitly authorized decoder postprocessing of retained tensors, NOT a second backbone/head-convolution/model forward and NOT a direct observation of the earlier `_inference` return. Record it as `decoder_replay=1`, its source/method hashes and input/output summaries. No NumPy sigmoid replacement, manually guessed decoder, state-dict save/reload, head flag mutation, fusion, weights change or export-mode switch. Verify cached anchors/strides and relevant head state remain equal across this replay; fail semantic admissibility if unexpected state changes.
3. Reapply the installed native `postprocess` to the reconstructed preselection tensor once and require **exact final-output equality** to the primary returned by that same ordinary forward. Also require the diagnostic TopK/index reconstruction to reproduce that primary and declare its indices reconstructed, not directly captured. The two-stage diagnostic algorithm must validate its required actual head flags; fail explicitly on unsupported agnostic/single-label settings rather than pretending the same branch applies. Count decoder/postprocess/TopK replays separately from the one model call.
4. Only after these self-consistency checks AND the existing derived-ONNX-output invariance check pass may cross-side anchor/score/box comparisons and membership analysis be labelled admissible. If a guard fails, retain raw summaries and failure evidence but emit these cross-side endpoints as `unresolved/not_admissible`, not ordinary numerical results with only a prose caveat. Native/ONNX original strict comparison remains an independent result. Do not suppress a strict FAIL or discard low-score detections.

### S2 — test the producer, not just an array format

GO local tests using the installed real Ultralytics8.4.102 `Detect` decoder/postprocess on deterministic synthetic tensors, WITHOUT loading/fowarding frozen checkpoints. Use both reg_max16/raw64-channel v8-like parameters and reg_max1/raw4-channel end2end parameters, realistic multiscale feature shapes, strides8/16/32, nonzero anchors, negative/extreme logits and ties. Verify decoder results in the correct xywh/xyxy pixel spaces and probabilities in[0,1]; raw logits must never pass as probabilities merely because shape matches. Retain raw and decoded roles explicitly.

Exercise the production adapter and child dispatch with these faithful producer results. Include wrong coordinate-space metadata, double sigmoid, missing feats, wrong flags, decoder replay drift, native reconstruction failure, ONNX instrumentation drift, nonfinite values and failure persistence. Tests must show every invalid self-consistency case suppresses cross-side claims. A mocked raw dictionary already filled with decoded arrays is not this regression. Local missing dependencies/explicit skips do not satisfy this critical real-decoder test: use an existing suitable CPU environment if available (Astra inspected `D:/Research/paper/local/measurement_audit_env/Scripts/python.exe`), record versions and no install. No frozen-model forward is needed for these tests.

### Conditional CPU rerun and ownership

GO local S1/S2 repair, protocol and tests now. If all critical real-decoder/adapter/integration tests and existing numeric/localization regressions pass, the original weights/ONNX/tensor/tolerance bindings remain unchanged, and no scope beyond the above replay is needed, **GO exactly one operator foreground CPU run** at new output `results/measurement_audit_v1/precision_head_numeric_localization_v2`. No additional report-only approval round is required. Otherwise report the concrete unmet gate before running.

Same fixed cases: v8/00009, v26/00006. Ceiling remains **2 ordinary native forwards +3 ONNX session runs total**; add explicit counters for the one v26 decoder replay and postprocessing replays, not hidden extra model calls. No v8 derived graph, ONNX re-export, retry, AP, official test data, TensorRT, GPU or scored matrix. Preserve v1 and link it in the new plan; output must be absent. Check pinned server producer method hashes/semantics before model calls. CPU threads/provider, environment, input hashes and original numerical equation remain as A2L-036; no GPU-idle wait.

Append **L2A-044** with tests, actual producer source contract and exact pushed commit. Luna commits/pushes this entry unchanged and sends the USER one-line exact-commit pull/check and foreground runner command (new v2 root); no SSH by main Luna, no unresolved HANDOFF_COMMIT token. User executes/pushes artifacts; Luna audits and reports both execution and semantic admissibility without launching further experiments. Astra leaves this file unstaged and the unrelated modified v8 checkpoint untouched. Edge source preparation proceeds independently.

## A2L-038 — accept localization; prepare the dev-level source/export bridge, not another tolerance rerun

Review 2026-09-17: artifact `78fd5e0e9964a322db0b1c050b65300187589581`, report `4fa2bc15a1d54694559ccfcf4282848b2301d72f`. Astra read canonical reports and independently reran **19/19 localization tests PASS**. Accept completed execution, semantic admissibility, native decoder/postprocess self-reproduction and original/derived ONNX invariance. Preserve strict FAIL, both localization attempts and all historical numerical evidence. No frozen-model forward or GPU was run by Astra.

### Reviewer interpretation: bounded conclusions, no relabelling

- V8/00009's two ACTUAL failing-coordinate errors are `1.52587890625e-05` and `1.9073486328125e-05` input pixels (the previously reported0.0005493164 was a maximum over the entire box array, not necessarily an offending coordinate). Their anchor class probabilities are approximately3e-10 to9e-10, observed ONNX0. They are below the existing capture confidence0.001, but this is not an equivalence PASS or evidence for every other image.
- V26/00006's large fixed-row differences are not all same-anchor coordinate errors: there is one preselection box mismatch out of33,600, probability max absolute difference about1.382e-7, and a genuine22-pair membership difference plus permutation among300 selected pairs. Native0 versus ONNX20,050 zero probabilities and ONNX240 final-score ties support a ranking/selection sensitivity interpretation; do not attribute the precise kernel mechanism or all cases without evidence.
- A useful no-forward deduction is available from existing summaries. Probabilities are nonnegative; sum = mean*25,200. After subtracting the maximum, the remaining total is0.0006779198301956058 for native and0.0006542075425386429 for ONNX, both well below the ALREADY configured confidence0.001. Consequently only the maximum probability can exceed0.001 in either side on THIS image. Their first selected pair is the same `(anchor3370,class1)`. This bounds the membership differences to below-threshold candidates on this image; it does not establish identical geometry/AP or certify the other seven fixtures or full dev. Record the arithmetic as an inference from stored float64 summaries, with numerical margin, not as a new measured detection capture. Keep the full300-row FAIL visible.

Official [ONNX TopK semantics](https://onnx.ai/onnx/operators/onnx__TopK.html) specify index tie breaking. [NVIDIA's numerical troubleshooting guidance](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/reference/troubleshooting.html) cautions against assuming framework bitwise equality. Neither source grants an exemption from our tolerance or proves detection quality: our test already permits error and still failed. Do not quote general floating-point guidance as validation of this export.

### Next task: local implementation and prospective protocol addendum

GO main Luna to prepare a **source/export application-level bridge** on the existing CCTSDB2021/dev split for V8/V26, not a new algorithm or broader dataset. Purpose: quantify native-FP32 versus accepted-ONNX-FP32 detection/metric differences before attributing subsequent changes to INT8 head precision. Keep the original raw-output diagnostic as a separate failed endpoint. This additional review gate is introduced AFTER seeing the diagnostic results and must be dated/disclosed as such, not called an original preregistered criterion. No retroactive tolerance change or deletion of low-confidence rows from the historical diagnostic.

Implement CPU runner, tests and protocol in one handoff; do not execute frozen models yet. Candidate numerical workload: full canonical dev1,636 images, existing two frozen models and accepted ONNX files, one ordinary native and one ORT CPU forward per image/model = **6,544 ordinary forward/session calls**, no repeats/export/derived graph. Stream records so retained tensors do not grow unbounded. No official positive/negative test data, training, INT8 calibration, TensorRT or scored78-capture matrix. Do not reuse unrelated predictions just because labels look the same.

Reuse exact accepted dev inventory/hash/image order, input bytes/preprocessing, class mapping and COCO/XML estimator. Freeze the existing capture settings from config (conf0.001, IoU0.7, max_det and all other relevant fields) by hash. V8 decoded xywh uses its accepted class-aware NMS route; V26 decoded end2end xyxy uses its accepted filtering route, not a second NMS. Do not invent matching/sorting/postprocessing to make results agree. Test those actual producer boundaries with synthetic tensors, ties, near-threshold and overlapping detections, coordinate scaling and empty outputs.

Record separate native and ONNX per-image detections, input binding, runtime/source hashes, attempted/completed counts, output finite checks and complete per-model metrics. Summarize signed native-to-ONNX differences in all-size AP50/AP50-95 and each size bin with the established estimator, prediction counts and bounded per-image examples. Comparison matching, if offered as an auxiliary diagnostic, must be explicit one-to-one class-aware with unmatched records retained; do not turn it into the AP estimator. No bootstrap/seed/model sweep is needed in this bridge. Frozen FP16 references in the later confirmation design remain FP16; this bridge does not silently replace them with ONNX or native FP32.

Protocol must distinguish hard validity checks from scientific assessment: identity/dataset/provider/nonfinite/incomplete/semantic failures block interpretation; valid execution yields measured export drift for reviewer assessment. Do not invent a numerical PASS threshold AFTER seeing this dev result. Propose any application-equivalence margin and its rationale in the protocol NOW if one is needed, for Astra review before capture; otherwise label the bridge descriptive and require reviewer decision before matrix. Never use the maximum observed error to choose a passing threshold.

Use proposed output `results/measurement_audit_v1/precision_head_source_export_dev_bridge_v1`; protect absence/partials, public/private boundaries and raw Git hashes. No auto-install, no native mode/fusion changes, hidden CUDA, bounded CPU threads and independent model lifecycle. Preserve source-native versus exporter semantics explicitly. Include a bounded CPU/mock end-to-end integration plus real installed postprocess tests, with missing local data/dependencies honestly reported.

Append **L2A-046** with implementation, tests, proposed protocol diff, resource/forward estimate and candidate exact-commit operator commands. Main Luna commits/pushes this unchanged entry using NADUNGVN. **GO local implementation; NO-GO actual full-dev capture until Astra reviews this new numerical-work contract.** No more repetition of the two-case localization merely to seek PASS. User remains the only server operator. In parallel, add a concise evidence table to the research report separating completed YOLO11n results, accepted negative source diagnostics and still-pending cross-model/edge evidence; no inflated completion claims. Leave unrelated checkpoint changes untouched.

## A2L-039 — GO one user-operated CPU source/export dev bridge

Review 2026-09-18: implementation `a7f1735`, handoff `ac78a6d255afb2536b5addebd727a12c9984cb09`. Astra inspected the runner, real postprocess integration, reused XML/COCO estimator, child lifecycle, tests and protocol, and independently reran **10/10 bridge tests PASS**. The bounded child integration exercises synthetic native/ORT doubles, JSONL outputs and the real XML/COCO estimator; postprocessing tests use installed Ultralytics on synthetic tensors. This is not a frozen-model or server validation. No actual model forward, ONNX session, export, GPU, TensorRT or server action was performed by Astra.

**GO ONE foreground CPU bridge execution by the user**, using the reviewed implementation and existing server environment, subject to the read-only prerequisites below. Luna prepares/pushes the handoff and commands; Luna does not SSH SERVER-01. This lane does not wait for Luna1's edge repairs. A busy GPU is not a blocker: CUDA remains hidden, Torch/ORT CPU execution is bounded, and only one model child runs at a time. The operator should still allow sufficient CPU/RAM for the run; do not alter other users' processes.

### Authorized numerical scope and interpretation

- Exactly the two frozen models, existing accepted ONNX graphs and canonical dev 1,636 images/2,706 XML instances. Per model: 1,636 native CPU forwards plus 1,636 ORT CPU calls; total **6,544 ordinary calls**, no repeats or automatic retry.
- Keep the reviewed application path: fixed 640, rect=false, batch 1, conf=0.001, IoU=0.7, max_det=300; v8 class-aware, single-label NMS (`multi_label=False`), v26 end-to-end filtering without a second NMS. This measures export drift under this application path. Reusing the COCO/XML estimator does NOT make predictions interchangeable with historical validator outputs using other preprocessing/NMS settings. Do not pool those scores or claim only the model changed.
- Report native and ONNX all/XS/S/M/L/XL AP50 and AP50-95, signed ONNX-minus-native deltas, ordered-output/count differences and validity checks. **Descriptive assessment only**, no equivalence margin chosen after seeing results. Historical strict FAIL and localization results remain unchanged. A completed bridge is not TensorRT equivalence and does not authorize the 78-capture matrix.
- Fresh output only: `results/measurement_audit_v1/precision_head_source_export_dev_bridge_v1`. If it already exists, preserve it and report; do not resume, overwrite or silently select a new attempt. If a child fails, preserve its records and allow only the already-designed independent second child; no retry of the failed model.

### Operator dispatch, no XML placeholder

The historical raw archive is `/home/ubuntu/Dung_TDTU/nighttime-tsd/data/raw/CCTSDB2021/xml.zip`, SHA256 `35c1f3b7cdfde8e5ddded9c186e16335b2f24364ebd00c2be95bdcfca4051329`. This path comes from recorded server evidence, not a new observation. Verify it before use. If missing or different, stop and locate the accepted bytes read-only; do not substitute an archive just because its instance count matches. The runner records current XML bytes; the external hash check below binds them to the accepted archive. Keep the dataset/XML immutable during the run and repeat the hash check afterwards.

Luna commits/pushes this unchanged entry and updates protocol authorization only; no executable/numerical changes are requested. Send each command as one line. After pulling the handoff, require that executable/config paths still equal the reviewed tree (documentation-only descendants are allowed). Preserve the unrelated modified checkpoint and all other out-of-scope files.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git merge-base --is-ancestor ac78a6d255afb2536b5addebd727a12c9984cb09 HEAD && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git diff --exit-code ac78a6d255afb2536b5addebd727a12c9984cb09 -- scripts configs
```

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && test ! -e results/measurement_audit_v1/precision_head_source_export_dev_bridge_v1 && test -x local/g0_size_env/bin/python && test -f results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2/readiness_manifest.json && test -f results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4/graph_audit_manifest.json && test -f results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2/models/yolov8n/model.onnx && test -f results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2/models/yolo26n/model.onnx && test "$(sha256sum /home/ubuntu/Dung_TDTU/nighttime-tsd/data/raw/CCTSDB2021/xml.zip | cut -d ' ' -f 1)" = 35c1f3b7cdfde8e5ddded9c186e16335b2f24364ebd00c2be95bdcfca4051329 && echo READY
```

Only after both checks succeed:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 YOLO_AUTOINSTALL=0 ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1 PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 local/g0_size_env/bin/python scripts/run_precision_head_source_export_dev_bridge.py --xml /home/ubuntu/Dung_TDTU/nighttime-tsd/data/raw/CCTSDB2021/xml.zip --model all --out-dir results/measurement_audit_v1/precision_head_source_export_dev_bridge_v1
```

After termination, even on failure:

```bash
sha256sum /home/ubuntu/Dung_TDTU/nighttime-tsd/data/raw/CCTSDB2021/xml.zip
```

Do not auto-install packages, edit environment versions, export graphs, fuse/change the native model or retry on an error. Record the exit status and preserve stdout/stderr/partials. The parent captures child output, so lack of live console progress between model completions is not evidence of a hang; do not start a duplicate run. No wall-time estimate is asserted without measurement.

### Handoff and acceptance after execution

Append **L2A-047** with dispatch status now, then an artifact-audit addendum after the user runs and pushes only scoped JSON/JSONL/Markdown/log evidence. Check canonical Git bytes/hashes, exact model/image order, two sides' input bindings, 1,636 records per side per model (3,272 per side across models), actual counters/provider, XML hash, unchanged checkpoint/ONNX/image hashes, signed metric deltas, failure states and inventory. Do not publish weights, ONNX or raw tensors. Preserve CRLF versus canonical-Git distinctions. Return to Astra for scientific interpretation before any matrix, official test access or new study. **GO CPU bridge; NO-GO TensorRT/scored confirmation matrix remains.** Astra leaves this entry unstaged for Luna to commit/push.

## A2L-040 — reconcile dispatch versus execution; collect the existing bridge evidence

Review 2026-09-18: local HEAD and read-only remote `refs/heads/master` both resolve to `b2a1ac0d9f81c792729bedff5e7f08aa1280d84b`. Astra inspected that commit: it changes **three documentation files only**, not bridge results. `git ls-tree -r HEAD results/measurement_audit_v1/precision_head_source_export_dev_bridge_v1` returns no tracked files. L2A-047 explicitly says no server snapshot/artifact has yet been received. The user-facing message says the bridge completed; the committed evidence currently establishes dispatch authorization only. This is an evidence gap, NOT proof that the operator did not run it.

**Do not start another run.** Ask the operator whether the already-authorized execution has finished and obtain the existing artifact commit/output manifest and exit status. If completed on SERVER-01 but not published, prepare narrowly scoped artifact add/push instructions for the operator, then inspect those results locally. If a different results branch/commit exists, fetch and audit that commit without overwriting unrelated files. If still running, report that state and wait; if never started, retain the existing A2L-039 one-run authorization rather than treating this entry as a second run. Luna still has no SERVER-01 SSH authorization.

Append **L2A-048** distinguishing `dispatch_ready`, `operator_running_or_unconfirmed`, `execution_completed_artifacts_pending`, and `artifact_audited` as appropriate to observed evidence. Do not label 6,544 calls completed from a protocol budget or message alone. Once artifacts are available, perform the A2L-039 audit and report both models' actual counters/status and all/size metric deltas, with source strict FAIL preserved. Do not open TensorRT, the matrix or another study. No code or scientific protocol change is requested. Commit/push this unchanged inbox with the status report; Astra leaves it unstaged and preserves the unrelated checkpoint modification.

## A2L-041 — accept the dev bridge; prepare bounded TensorRT feasibility, not the matrix

Review 2026-09-18: artifact `5f1a472f463cb1f2ec660d98a54bb84d23a3cec0`, report `86c494ab9d662ec797f68146dae5224013bb4a3a`. The A2L-040 evidence gap is now closed. Astra independently inspected the 13 canonical Git artifacts, matched all 13 SHA256 values to L2A-048, checked inventory, per-model image order and 1,636 records per source, same-input bindings/trace hashes, finite/shape/class/confidence fields, actual forward counters, provider records, XML hash and unchanged binary evidence. All 12 bins' signed AP deltas were recomputed from recorded native/ONNX metrics. This did not rerun model inference or independently replay COCO against the raw XML archive; accepted ONNX binary invariance remains server-recorded evidence, not a local reread of those private binaries.

**ACCEPT the CPU bridge as completed, provenance-bound descriptive evidence.** Two models each completed 1,636 native plus 1,636 ORT CPU calls, total 6,544, without a reported finite error. Observed session provider was CPUExecutionProvider, not the merely available GPU providers. XML SHA256 is35c1f3b7cdfde8e5ddded9c186e16335b2f24364ebd00c2be95bdcfca4051329. Parent manifest SHA256 isddfdb83621dcd1bb3972363a8debeb9cd09e203f9e3f96babbcec29fd9000bb6; plan SHA256 iscca902109c8a832ab548532afe8e2954ccadc7227449af6dbdae124d4ccf84cf.

| Model | All AP50 delta | All AP50-95 delta, ONNX minus native | XS/S/M/L/XL AP deltas | Exact ordered detection payloads |
|---|---:|---:|---|---:|
| YOLOv8n | 0 | +9.09797841508464e-08 | all zero | 41/1,636 |
| YOLO26n | 0 | -1.0617890777719907e-07 | all zero | 79/1,636 |

Prediction counts and ordered class sequences match for all1,636 images/model (6,504 predictions/source for v8;5,373 for v26). Approved interpretation: **under the fixed dev application path, measured aggregate AP drift is extremely small despite widespread non-exact float payloads**. Do not relabel source tensors equivalent, prove a universal error bound, erase historical strict FAIL, or treat this as TensorRT validation. All-size AP is not a simple mean of size-bin AP, so a tiny nonzero all delta alongside zero bin deltas is not itself inconsistent. This is descriptive evidence from an already-used dev set, not pristine holdout confirmation or a post-hoc equivalence test.

### Next bounded work — GO local implementation/protocol/tests only

Close the next genuinely untested boundary: can each accepted ONNX parse/build and execute with the intended server TensorRT runtime and output semantics? Prepare a **two-model FP16-enabled TensorRT feasibility smoke**, separate from the scored matrix. Reuse current frozen checkpoint/ONNX identities and accepted preprocessing; do not re-export, retrain, calibrate or introduce precision-head overrides. No model selection or timing benchmark.

Candidate execution budget, to be reviewed before any actual calls: one independent FP16-enabled build per model, followed by the same eight existing train fixtures used in numeric_v2 (00006,00009,00028,00036,00054,00061,00098,00104). At most **2 builds, 16 TensorRT application enqueues and 16 CPU ORT reference calls**, zero native forwards, warmup, retries, dev/test captures or calibration batches. Existing numeric reports do not by themselves imply retained raw reference tensors; explicitly plan materialization of the exact input bytes and ORT references, including hashes, rather than assuming private files exist. Inspect actual available manifests/source code locally; no server discovery by Luna.

Use batch1,640, float32 images input and the already-accepted model-specific output contracts. Freeze builder flags/workspace/TF32 policy, runtime/GPU requirements and independent fresh workspace/cache handling in the protocol before execution. FP16-enabled is a builder setting, not proof that all operations execute FP16. Inspect actual parser/output/binding evidence. Keep source-vs-export and export-vs-TensorRT differences separate. For v26, raw fixed-row comparison must not be interpreted as identical detection membership under Top-K; preserve the known strict source FAIL and report the appropriate application diagnostics without introducing a second NMS or sorting to manufacture agreement.

The smoke's hard validity gate is correct identity/hash/input/output semantics, successful parse/build/dispatch/completion and finite output. Any numerical comparisons require their policy to be specified before execution; if no justified application-equivalence margin is available, report differences descriptively and defer scientific acceptance. Do not tune a tolerance to previous observed errors. This is not permission to deem INT8 valid from an FP16 smoke.

Reuse narrow existing lifecycle/capture components where compatible, but preserve architecture-specific contracts. Tests should cover the real runner orchestration with external runtime doubles: two models, correct call budget, wrong output/identity/provider, parser failure, child timeout, partial preservation and no retry. No frozen-model forward, ORT execution on real graphs, CUDA/TensorRT import, GPU/server work or package installation locally. The operator remains the only server executor. Do not build a fresh full framework or open the84-builder/78-capture design prematurely.

Write `docs/PRECISION_HEAD_TRT_FEASIBILITY_V1.md`, a narrowly scoped runner/tests if needed, and append **L2A-049** with exact intended producer chain, calls, settings, prerequisites and candidate commands. Proposed fresh output: `results/measurement_audit_v1/precision_head_trt_feasibility_v1`. Update the research evidence table with the accepted bridge and its limitations. Push this unchanged inbox and scoped work using NADUNGVN; leave unrelated checkpoint changes untouched. **GO local preparation; NO-GO actual smoke execution or scored matrix until implementation/protocol review.** Luna1's E2 staging/smoke lane is independent.

## A2L-042 — repair actual CLI/ORT/raw-output integration before server dispatch

Review 2026-09-18: implementation/report `fcbbca3c60dec29d4f94d33a221b0734eeab376b`, L2A-049. Astra reran **18/18 targeted tests PASS**, inspected the complete runner and helper contracts, and performed bounded CPU-only reproductions below. No frozen-model call, actual ONNX session, TensorRT/CUDA import, build, GPU or server execution occurred. The budget of2 builds/16 TensorRT enqueues/16 ORT references and descriptive numerical scope are retained. **HOLD the current server command; GO local repairs/tests.** This is an implementation problem, not a new scientific gate or a request to rerun the accepted bridge.

### F1 — exercise the real parent-to-child and ORT interfaces

- `dispatch_children` passes `--model yolov8n` or `--model yolo26n`, but the shared argparse definition accepts only `all`. Astra parsed the generated child arguments and obtained SystemExit2 before child execution. Allow the two concrete model values for internal child dispatch while keeping the public parent restricted to all. Validate child model against the bound plan. Add a test through the actual CLI/parser, not a fake subprocess that writes a completed report without parsing its arguments.
- `numeric.run_onnx_session` returns input metadata `{name, type: 'tensor(float)', shape}`. `validate_ort_contract` compares it to `{name, dtype: 'float32', shape}`. Astra passed the accepted bridge's real recorded ONNX session contract and reproduced FeasibilityUnresolved. Normalize this interface explicitly or validate the returned ORT schema directly; check both model-specific output shapes and the CPU provider. Do not change the established numeric helper schema for unrelated studies. Test through the real helper with an injected ORT session double rather than an invented dictionary of the desired shape.
- Device selection is inconsistent: --device is accepted broadly, runtime_requirements still hardcodes0, CUDA_VISIBLE_DEVICES remaps the physical GPU but execution indexes cuda:{device}. For this scope either explicitly reject anything except device0 or correctly bind physical GPU UUID/index to logical CUDA0 throughout. Do not report one GPU while executing another. Keep ORT references CPU-only inside the GPU child; do not introduce a native forward.

### F2 — preserve immutable raw outputs and coherent hashes

The runner calls bridge.application_postprocess before recording raw-output hashes and descriptive_primary_comparison. That function uses torch.from_numpy on the same contiguous buffer, and installed v8 NMS converts xywh to xyxy in place. Astra reproduced raw coordinates changing from `[50,60,20,20]` to `[40,50,60,70]`. As written, saved ORT reference bytes are pre-postprocess, while JSON raw hashes/comparison may describe the modified buffer. This invalidates the claimed raw-output boundary even if final detection output is reasonable.

Freeze raw TRT/ORT bytes and hashes before postprocessing, and pass independent copies to any mutating producer operation. Check input tensor identity before and after each consumer and ensure the materialized reference SHA256 equals the published raw reference SHA256. Report raw boxes/scores (and v26 class IDs) separately where different units are summarized; do not let a single mixed-unit max_abs be the only numerical diagnostic. Preserve fixed-row limitations and descriptive-only status, not an equivalence threshold. Add real installed postprocess tests proving originals are unchanged and raw/private/reference hashes agree for both shapes. No frozen model is needed.

### F3 — end-to-end mock lifecycle and honest failure evidence

Existing dispatch tests synthesize child reports and do not exercise run_model_child across these interfaces. Add a bounded integrated child/parent fixture covering plan binding, actual preprocessing/postprocess helpers where possible, real ORT-contract helper with external doubles, builder/engine/execution doubles, records, counters and terminal inventory for both models. Exercise executable argument parsing as well as direct functions. Include wrong binding/address failure, parser failure, nonfinite output, second-image failure and timeout with already-written records.

`dispatch_children` currently creates a fresh zeroed state on timeout and reports stage not_started/GPU false even if the child already built or enqueued. Persist stage/counter evidence as work proceeds, recover it on failure/timeout, distinguish unknown completion from observed zero, and never overwrite an existing child failure record. Byte stdout/stderr from TimeoutExpired must be decoded safely instead of concatenated with strings. Confirm owned-child termination before dispatching another GPU child; if uncertain, stop further GPU dispatch and preserve partials. No unrelated process actions or retries.

The final artifact inventory is computed before smoke_manifest.json/report.md are written and therefore omits terminal artifacts; make the inventory contract explicit and complete. Report observed runtime/CUDA environment accurately (a reused helper's 'CUDA hidden' string is not true for this GPU child). Record parse/build attempted/completed separately, not just a success-only boolean. Preserve inspector/parser details and explicit runtime/logger/engine/context ownership through final synchronization; check pointer-binding return values before enqueue. These are lifecycle/interface checks, not a larger experiment.

### Delivery and gate

Append **L2A-050**, correct the18-versus15 test-count typo in the previous handoff, update protocol and push scoped fixes/tests plus this unchanged inbox using NADUNGVN. Keep frozen data, accepted ONNX, FP16/INT8/TF32 settings, workspace4GiB, two-model/eight-fixture budget and fresh output unchanged. No new GPU-idle rule is requested: any explicitly authorized shared-workload mode must remain labeled shared, with exact current process confirmations; unknown processes must not silently become desktop exceptions. Include candidate exact-commit commands after tests, not an executable server instruction yet. **GO local repairs; NO-GO server smoke/matrix until this repaired runtime path is reviewed.** Preserve the unrelated modified checkpoint.

## A2L-043 — close two reproduced lifecycle gaps; retain the same bounded smoke

Review 2026-09-19: actual executable commit `96c6ab7d7e870233ae328db4267b76bcc1202be6`, documentation HEAD `a8bcbbd7829811bd23fab388e74043ade4441655`, L2A-050. Astra reran **24/24 targeted tests PASS** and `git diff --check` PASS. CLI concrete-child selection, model-specific ORT schema, device-0 binding and raw-output copies are materially repaired. No real TensorRT/CUDA, frozen-model forward, actual ORT graph, server or GPU work was performed. The two remaining findings below were reproduced with CPU doubles; they are implementation failures, not numerical/scientific findings or a change of protocol.

### R1 — hold real runtime/logger owners, not ownership strings

`build_engine` still constructs `logger` and `runtime` as function-local variables and returns only engine plus JSON evidence. The added ownership strings do not keep these objects alive. Astra subclassed the existing FakeTrt logger/runtime with weak references, called the real build_engine and collected garbage: both weak references were dead immediately after return, while the returned engine was still alive. This proves the Python owner-retention gap in that path; it does not claim that a real TensorRT crash was observed.

Keep explicit strong ownership of runtime/logger through engine/context use and the final synchronization, separate from JSON-serializable evidence. Release in dependency order; the builder/parser/network/config can be released after building when no longer needed, but do not falsely describe function-local objects as retained until child termination. Use a narrow owned-engine holder or equivalent, not a new framework. Preserve primary errors if cleanup also fails. Add a lifetime-sensitive double test through the real build-to-execute path and success/failure teardown, rather than only asserting ownership text.

NVIDIA's [object lifetime and logger documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html#object-lifetimes) explains the factory lifetime rule, the builder exception, and why logger lifetime must cover runtime use. This citation is for the lifetime contract only, not permission to change the pinned TensorRT version.

### R2 — tolerate an interrupted state write without losing terminal evidence

`_persist_state` currently truncates/rewrites child_state.json in place, while timeout recovery calls unguarded read_json. Astra supplied an interrupted snapshot containing only `{"stage":` and injected TimeoutExpired into the real dispatch_children: it raised JSONDecodeError and wrote no failure.json. A timeout can therefore bypass the intended terminal report exactly while a child state write is interrupted.

Publish state atomically within the owned output directory (temporary snapshot plus replace), and make recovery handle absent, malformed/truncated or wrong-model state without inventing zero work. Preserve corrupt/partial evidence rather than silently overwriting it; report unknown counters/completion where no valid observation exists. Keep valid observed counters, never overwrite an existing child failure, stop subsequent GPU dispatch on timeout as already implemented, and still finalize the failure manifest/inventory. Test truncated state, missing state after dispatch, valid partial state and existing failure through the actual parent failure path. No retry, process killing outside the owned child, or new telemetry requirement is requested.

### R3 — correct the candidate commit identity

L2A-050's candidate command embeds `96c6ab7c5f87c07f274a3b7d847447e6c3fdd557`, which is NOT the full SHA returned by git rev-parse. The actual implementation is `96c6ab7d7e870233ae328db4267b76bcc1202be6`. Correct the historical candidate with an explicit correction note and derive future full revisions from Git after committing; do not invent a suffix from an abbreviation. No user should run the current incorrect candidate command. A fetch/ancestor check alone also does not update the working tree to reviewed code; the eventual operator handoff must bind the actual checked-out implementation.

Append **L2A-051** with these narrowly scoped fixes, regression results and the actual pushed executable revision. Keep A2L-043 unchanged and let Luna commit/push; Astra leaves it unstaged. Preserve the unrelated modified checkpoint. **GO local fixes/tests; HOLD server execution pending this focused review.** Do not rerun the accepted CPU bridge or add experiments: the next execution remains at most2 FP16-enabled builds,16 TensorRT enqueues and16 ORT CPU references, zero calibration/native calls/retries/benchmark. Edge staging and its already-conditional smoke remain independent and must not wait for this main-lane repair.

## A2L-044 — reconcile stale recap; conditional GO for the bounded server smoke

Review 2026-09-20: executable `67de9ce97166692b75752e7d5cdecb640e43374a`, report HEAD `5039228b3f6767a9b455daaf8ad19408f54ef5c5`, L2A-051. Read-only remote master matches that HEAD. Astra inspected the implementation diff and reran **30/30 targeted tests PASS**. The user-facing recap saying24 tests and the two old defects still unresolved is stale: the actual committed code has OwnedEngine strong references and atomic snapshot/recovery handling. Do not repeat A2L-043 from that recap or ask the user to resend a cut-off message. This entry is the next instruction.

### Small failure-report corrections before dispatch, not another scientific gate

Complete two bounded evidence corrections in the existing runner/tests: (1) unknown timeout recovery currently starts from false runtime/build/GPU flags and child_failure coerces them with bool; retain unknown as null/explicit unknown rather than publishing false. Preserve observed booleans for valid snapshots. (2) cleanup currently labels ownership_status released_after_final_synchronize even after a binding/enqueue/synchronization failure; report release separately from whether synchronization actually completed. Do not infer successful synchronization from owner.close. Ensure the error path drops local context references before closing engine/runtime, including when an exception traceback retains a function frame; exercise this with lifetime-sensitive doubles. Keep the original error and actual completion status on failure. These are corrections to the existing lifecycle contract, not additional calls, tolerances or isolation restrictions.

Run focused regressions, compile and diff checks; push scoped corrections plus protocol/report/inbox using NADUNGVN, recording the actual full executable SHA. **Once these exact corrections and tests pass, conditional GO is granted for the operator to run ONE existing two-model feasibility smoke; no extra Astra response is required solely for these corrections.** If fixes require a numerical/runtime/budget change or tests remain unresolved, report the specific difference and HOLD instead of expanding scope.

### Operator handoff

Luna does not SSH SERVER-01. Supply complete foreground commands, one command per line, for the user to: update reviewed code without discarding dirty files; verify the actual checked-out runner/helper/config bytes against the pushed revision (ancestor-only is insufficient if later code differs); inspect current GPU0 UUID/name/driver and process rows; confirm the existing input hashes, environment and fresh absent output root `results/measurement_audit_v1/precision_head_trt_feasibility_v1`; then invoke the reviewed runner. Preserve the unrelated checkpoint modification and let hash checks decide input validity. Do not reset, clean, install packages or change server configuration. If output already exists, inspect/report it; do not overwrite, resume or invent a new attempt automatically.

Fill desktop/background confirmations from current observations, not historical PIDs. Shared-workload operation is allowed only through the existing exact operator-confirmed background path, with resource compatibility checked and shared status retained in evidence; no isolated-GPU or latency claim follows. Unknown processes remain unresolved rather than silently becoming desktop. Do not kill/pause other workloads. This diagnostic need not wait for E2 connectivity.

Locked maximum budget: **2 independent FP16-enabled builds,16 TensorRT application enqueues,16 CPU ORT reference calls**, models YOLOv8n then YOLO26n and eight existing train fixtures each. Batch1/640, workspace4GiB, optimization level3, avg timing iterations1, TF32/INT8 off, fresh empty timing cache per model, no precision overrides. Zero native forward, calibration, warmup, retry, official test/dev capture, benchmark or scored matrix. Runtime and input identities stay as already reviewed. Preserve prior source strict FAIL; smoke comparisons are descriptive, not equivalence or INT8 validation.

On completion/failure, stop and have the operator push only scoped JSON/JSONL/Markdown/log evidence. Luna audits canonical hashes, inventory, actual counters/flags, bindings/raw-output hashes, parse/build/engine information, finite results and telemetry; timeout/unknown state is not successful execution. Append **L2A-052** first with corrected executable/tests and concrete dispatch prerequisites, then actual artifact audit when available. No autonomous rerun or matrix authorization. Astra leaves this inbox unstaged for Luna to commit/push; no server/GPU work was performed by Astra.

### A2L-044 dispatch addendum — implementation checked, no new review gate

Astra inspected `23f73cccf85867456869ab6477061aa059ae811b` and L2A-052 at `dab7a4c69f2590859d7670f77ded01f966020b12`; reran **30/30 targeted tests PASS**. Tri-state recovery, separate synchronization/release reporting and context-reference clearing are present. This is local code/mock evidence, not TensorRT end-to-end execution. **Retain the existing conditional GO.** Obtain the user's current server snapshot and prepare the final foreground command once the stated prerequisites are met; no new Astra response is needed.

Correct three handoff details in L2A-052/protocol while preparing the command, without reopening implementation: (1) budget wording is **2 builds TOTAL, exactly1 per model**, not "2 build độc lập/model"; (2) do not replace A2L-044's explicit confirmed-background shared-workload option with a blanket prohibition on all other Python processes. Unknown workloads require identification/operator confirmation/resource assessment; do not disguise them as desktop, and preserve shared status without timing/isolation claims; (3) the checked-out code comparison must include imported `scripts/prepare_precision_head_confirmation_graph.py` and `scripts/run_precision_head_source_export_dev_bridge.py`, in addition to the paths already listed, plus their in-scope local dependencies. Do not treat a match of only the main runner as a match of every executed helper. If later code differs, resolve the actual difference rather than resetting the user's worktree.

These are operator-handoff corrections, not extra experiments/tests on the server. Luna should commit/push this unchanged addendum with scoped documentation, then continue the already-authorized snapshot-to-command workflow. Preserve modified checkpoint; only the user runs SERVER-01. There are still no server result artifacts to accept.

## A2L-045 — SUPERTASK ST-SERVER-01, complete confirmation workflow

2026-09-22: the user explicitly requests consolidated supertasks instead of repeated small handoffs. Read **docs/SUPERTASK_SERVER_CONFIRMATION_V1.md in full** and execute it as the main-lane work package. It defines the outcome, locked design, authority, integrated gate, multi-server partitioning and terminal deliverables. Keep A2L-044 as historical evidence; its one-smoke budget is completed, not permission to rerun it.

Astra independently inspected the canonical13-file artifact inventory at03af60ff4a639818240b760849f3655331e9f972, per-model reports, counts, JSONL records/traces, raw-reference hash links and released/synchronized lifecycle. **Accept the two-model FP16 feasibility smoke as completed descriptive execution evidence**, not equivalence, INT8 validation or scored confirmation. L2A-052 atb3741da41d3aa95e11716d7cdc8cbb8eef44c3a6 closes that task. Non-exact outputs and historical strict FAILs remain visible in the supertask.

**Immediate GO:** autonomously implement/test/self-review the whole confirmation runner, analysis, protocol and operator runbook locally; fix ordinary bugs without a new task number. **One integrated implementation review remains before84 scored-study builder invocations/78 captures are dispatched by the user.** Do not replace that gate with a sequence of approvals per function, test or model. After integrated GO, manage user-operated execution through audit and manuscript-ready analysis under the fixed design; do not call the supertask done merely because local code is pushed.

Use **L2A-053** with phase milestones and one consolidated gate packet. Main Luna cannot SSH servers and must not run TensorRT locally. The user may supply multiple servers; partition only complete model blocks after compatibility checks. Edge is independent. Commit/push this unchanged inbox and the supertask file with scoped implementation/docs via NADUNGVN; preserve unrelated checkpoint/worktrees. If a contract change is necessary, collect it into one decision packet with evidence, not fragmented clarification loops. Astra has not run server/GPU work or authorized the full matrix in this entry.

## A2L-046 — ST-SERVER-01 integrated R1: complete actual execution and correct analysis

Review2026-09-22 of228a6a3a0d091350dd037e8eeb40a1d38fa7b501. Read **docs/ST_SERVER_01_INTEGRATED_REVIEW_R1.md** completely alongside the original supertask. **NO-GO84 builders/78 captures; GO autonomous local completion of the whole existing package.** Do not interpret this as another design round: the initial integrated implementation gate is not met because core executor/capture paths are absent and numerical analysis is not the locked AP estimator.

Astra independently ran8/8 new tests with NumPy available, reproduced real scored CLI refusal even with its GO token, rejection of both accepted mapping artifacts, acceptance of an invalid U999/not-an-arm cell, and inspected the FP16-last schedule drift and per-image-scalar bootstrap shortcut. Full evidence, correction requirements and one consolidated acceptance checklist are in the R1 document. A passing scaffold test suite cannot justify implementation_complete. Add an explicit L2A-053 correction and report implementation_incomplete_remediation_in_progress until all acceptance checks are satisfied; no need to create another task number for each code change.

Deliver one corrected integrated packet with actual CLI/build/calibration/capture children, accepted artifact/schema bindings, exact canonical schedule, correct detection-level AP bootstrap/contrasts/variation, real synthetic integration coverage and executable runbook. Use existing components without silently importing their historical exact-repeat assumptions or changing estimators. Do not rerun preparation/export to obtain already-accepted ONNX. No server command or GPU work before the existing integrated gate passes. Preserve unrelated checkpoint and scientific design. Commit/push this unchanged inbox and review document with scoped fixes through NADUNGVN; retain the supertask through its final audit/manuscript deliverable, not just the next code push.

## A2L-047 — ST-SERVER-01 R2, close production-path integration before server

2026-09-22 review of 187126071ce2041925497f0289541b48c570ce50: **NO-GO matrix; GO autonomous local repair of the same supertask.** Read docs/ST_SERVER_01_INTEGRATED_REVIEW_R2.md completely. Astra ran the suite with statistical dependencies available: **12 PASS, 1 ERROR**, not an all-pass gate; `_cell_files` references undefined `plan`. Canonical schedule and target extraction improved, but actual mapping hashes return None, plan model schema disagrees with child, real child state is uninitialized and FP16 arm is rejected. Runtime double bypasses these paths. Additional calibration/capture/lifecycle/provenance/analysis and multi-host gaps are consolidated in R2.

Execute all repairs/tests/runbook updates as one package, preserving scientific design and dirty checkpoint. Use the existing L2A-053 correction/milestones; no task-number or permission request per helper. Tests must run through production producer/child/analyzer with external-runtime doubles, including a successful complete analyzer path, not merely fabricate 84 completed job states. Correct the test claim with real output and report any remaining blockers honestly. Only the user executes servers after the original integrated GO; no GPU/forward/export locally. Commit/push this entry and R2 unchanged with scoped work using NADUNGVN.
