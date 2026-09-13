# Uniform inference repeatability v1

Đây là protocol được Astra duyệt tại **A2L-003** để kiểm tra repeatability của inference trên **ba serialized engine Step A hiện có**. Tài liệu này là contract triển khai, không phải báo cáo kết quả. Code/tests local phải được review trước khi người dùng chạy server. A2L-004 yêu cầu giữ tách biệt study ID logic với tên thư mục server và khóa GPU identity trước khi tạo output.

## Câu hỏi và phạm vi

Kiểm tra câu hỏi hẹp: cùng một serialized engine và cùng evaluation input có cho predictions/metrics lặp lại giữa ba process inference độc lập hay không?

Runner chỉ inference/verify trên `CCTSDB2021/dev`. Không build TensorRT, export ONNX, calibration, training, benchmark latency/energy, test split, policy selection, B/C, precision intervention hoặc retraining. Không chọn engine theo AP; engine được chọn theo repeat ID 1/2/3 từ Step A.

## Input identity

- Logical source study ID là `uniform_build_repeat_v1`; thư mục source server cố định: `results/measurement_audit_v1/server_uniform_build_repeat_v1/`.
- Logical output study ID là `uniform_inference_repeat_v1`; thư mục output server cố định: `results/measurement_audit_v1/server_uniform_inference_repeat_v1/`. Không dùng tên thư mục thiếu prefix `server_`.
- Engine `engine_1`, `engine_2`, `engine_3` lần lượt là `repeat_1/model.engine`, `repeat_2/model.engine`, `repeat_3/model.engine`.
- Runner đọc `engine_sha256` từ từng build manifest và hash lại engine trên server trước capture. Mismatch dừng, không export/rebuild và không chuyển engine sang GPU khác.
- `source_weights_sha256`, ONNX hash, settings, environment/GPU và common calibration-cache hash phải nhất quán với study manifest. `repeat_capture_inputs()` tiếp tục kiểm tra frozen ONNX/tensor contract của Step A.
- Chỉ chấp nhận `--device 0`. Trước khi tạo output hoặc bắt đầu inference, runner lấy snapshot `nvidia-smi` và yêu cầu UUID, GPU name và driver version khớp `gpu_before.device` của Step A. UUID hoặc driver khác phải dừng để review; không chuyển/copy engine và không tự rebuild.
- Engine binaries không được commit vào Git; engine hash và inspector/provenance được lưu trong output JSON.

## Numerical/runtime contract cố định

Mỗi engine có đúng ba capture mới, tổng cộng chín capture. Mỗi capture là một process mới; verify chạy ở process riêng. Không dùng capture lịch sử làm một trong ba lượt mới.

- Dev đúng 1.636 ảnh và 2.706 instances; image order do native validator giữ nguyên.
- `batch=1`, `imgsz=640`, `rect=False`, `workers=0`, `conf=0.001`, `iou=0.7`, `max_det=300`.
- Dùng preprocessing, native validator, native rematching và COCO/XML evaluator helper hiện có; không viết evaluator mới.
- Ba round tuần tự, cân bằng vị trí engine: `[1,2,3]`, `[2,3,1]`, `[3,1,2]`.
- So sánh trong cùng engine trước; không gộp chín lượt thành chín build độc lập. Round 1 là baseline kỹ thuật đầu tiên, không được chọn theo metric.
- Không dùng historical Step A spread làm threshold pass/fail mới. Kết quả luôn dừng ở `inference_repeatability_completed_review_required`.

## Payload và phép so sánh

`validator_predictions.json` được chiếu vào schema `uniform_inference_repeat_prediction_payload_v1` trước khi hash. Payload giữ `image`, thứ tự image, detection order, class, bbox, confidence, validator input/statistics và coordinate contract. Metadata thời gian, path, engine identity và data path không nằm trong payload hash.

Runner báo riêng:

1. exact equality của canonical payload hash;
2. thay đổi membership/duplicate/order của image;
3. bbox, confidence, class hoặc detection-order drift. Detection order chỉ được nhận diện bằng multiset của complete detection triples, không ghép bbox bằng zip positional khi thứ tự đổi;
4. numeric deltas của Ultralytics full metrics và COCO/XML full + XS/S/M/L/XL AP50/AP50–95;
5. mean, sample SD, min, max và range theo từng engine × ba capture.

Nếu payload hoặc metric khác, lưu chênh lệch và giữ `review_required`; không chạy thêm vô hạn và không chọn capture tốt nhất. Nếu cả ba giống nhau, chỉ kết luận “không thấy khác biệt trong các lượt đã kiểm tra”, không tuyên bố deterministic trong mọi điều kiện.

## Telemetry và shared server

Mỗi capture dùng GPU phase lock và snapshot `nvidia-smi` trước/sau. Runner cũng kiểm tra UUID/name/driver ở cả hai snapshot của từng capture so với Step A binding. Giữ nguyên operator-confirmed desktop exception của Step A: người vận hành phải lấy PID/path hiện tại từ snapshot rồi truyền đúng `--confirm-desktop-process PID=PATH`; không dùng PID lịch sử, không đọc `/proc`, không kill/pause process, không đổi quyền/clock/power limit.

Telemetry sampled không chứng minh GPU isolation hoặc loại workload xuất hiện giữa hai snapshot. Process mới không được phân loại hoặc workload cạnh tranh phải được ghi vào review flags; không blanket-ignore. SERVER-01/RTX8000 là mặc định vì engine compatibility/reference đã được kiểm tra ở Step A; server khác cần review riêng trước khi dùng.

## Output contract

Output cố định và không overwrite:

```text
results/measurement_audit_v1/server_uniform_inference_repeat_v1/
  study_manifest.json
  round_1/engine_1/{capture,verification,comparison.json,execution_manifest.json}
  round_1/engine_2/{capture,verification,comparison.json,execution_manifest.json}
  round_1/engine_3/{capture,verification,comparison.json,execution_manifest.json}
  round_2/engine_2/{capture,verification,comparison.json,execution_manifest.json}
  round_2/engine_3/{capture,verification,comparison.json,execution_manifest.json}
  round_2/engine_1/{capture,verification,comparison.json,execution_manifest.json}
  round_3/engine_3/{capture,verification,comparison.json,execution_manifest.json}
  round_3/engine_1/{capture,verification,comparison.json,execution_manifest.json}
  round_3/engine_2/{capture,verification,comparison.json,execution_manifest.json}
  repeat_summary.json
```

Mỗi `capture/` phải có `validator_predictions.json` và `capture_report.json`; mỗi `verification/` phải có `native_matching.json`, `size_coco_xml.json`, `verification_summary.json`. `study_manifest.json` lưu round order, engine hashes, environment, Step A/current GPU snapshots và UUID/name/driver binding, reference hashes, payload schema và operator confirmations. `repeat_summary.json` lưu chín record, aggregate theo engine, review flags và limitations.

## Lệnh server dự kiến — chưa được chạy trước code review

Sau khi Astra review code commit, người dùng pull/check code. Mỗi lệnh dưới đây là một dòng vật lý; runner chạy foreground, không dùng `nohup` mặc định. Lấy PID/path desktop mới từ `nvidia-smi` hiện tại; không thay placeholder bằng PID lịch sử.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git rev-parse HEAD && hostname && nvidia-smi --query-gpu=uuid,name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used --format=csv,noheader && nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
```

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/run_uniform_inference_repeat.py --out-dir results/measurement_audit_v1/server_uniform_inference_repeat_v1 --device 0 --confirm-desktop-process CURRENT_PID=CURRENT_ALLOWLISTED_PATH
```

Nếu có nhiều desktop process đã được đối chiếu, thêm mỗi cặp `--confirm-desktop-process PID=PATH` trên cùng một dòng. Không truyền confirmation cho process không đúng allowlist. Nếu preflight gặp training/inference/build hoặc process chưa phân loại, dừng và gửi log; không kill process và không chạy lại trên output partial.

Sau khi runner in `DONE`, kiểm tra danh sách JSON rồi push riêng artifact; không stage engine/source/ONNX/calibration tensor:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git add -f -- ':(glob)results/measurement_audit_v1/server_uniform_inference_repeat_v1/**/*.json' && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git commit -m "results: Uniform inference repeatability diagnostic" && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git push origin master
```

Chưa có server result hoặc lệnh thực thi nào được coi là đã chạy trong commit protocol này. Sau khi người dùng push artifact, Luna mới pull/kiểm tra hash, payload, matching, metrics và bàn giao reviewer; không tự triển khai bước nghiên cứu tiếp theo.
