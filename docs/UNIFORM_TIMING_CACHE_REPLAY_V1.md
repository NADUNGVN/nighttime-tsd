# Uniform timing-cache replay v1 (A2L-010)

Đây là feasibility check trước mọi precision-head/calibration intervention. Runner không chạy lại Step A và không mở 15 model.

## Contract khóa trước server run

- Study ID: `uniform_timing_cache_replay_v1`.
- Output chính xác: `results/measurement_audit_v1/server_uniform_timing_cache_replay_v1/`.
- Source bất biến: `results/measurement_audit_v1/server_uniform_build_repeat_v1/`.
- GPU/environment phải khớp Step A: UUID `GPU-9850d121-55dc-e752-ffaa-df19e7585eb4`, Quadro RTX 8000, driver `595.71.05`, và các package/runtime đã ghi trong source manifest.
- Frozen source là cùng `source.onnx`, frozen weights, Uniform seed42 Ncal1024 và `repeat_1/calibration.cache`. Timing-cache input cho cả ba process là bản sao riêng của `repeat_1/timing.cache`; không dùng output của build trước.
- Preflight hash trực tiếp `results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt`, ghi `frozen_weights_path` và `frozen_weights_measured_sha256`; manifest hash không thay thế phép đo file thực tế.
- Ba process build độc lập chạy tuần tự 1→2→3. Sau đó capture dev đúng một lần/engine theo thứ tự 1→2→3, rồi verify CPU. Không có 9 inference repeats.
- Runtime capture giữ nguyên dev 1,636 ảnh/2,706 GT, `imgsz=640`, batch1, workers0, `rect=false`, conf0.001, IoU0.7, max_det300 và evaluator hiện hành.
- Builder giữ INT8, FP16/TF32 off, workspace 4 GiB, optimization level3, average timing iterations1, detailed inspector, OBEY và danh sách Sigmoid FP32 của Step A.
- Ordinary timing-cache replay chỉ gọi `set_timing_cache(input, ignore_mismatch=False)`. Không dùng editable cache, algorithm selector, `ERROR_ON_TIMING_CACHE_MISS`, Q/DQ, precision override mới hay calibration policy mới. Cache miss chỉ được báo nếu log/API cung cấp bằng chứng; không suy full cache hit từ việc không có warning.
- Calibration cache phải được đọc ít nhất một lần; mọi lần đọc phải trả đúng input, `get_batch()` không được gọi (`0` calibration batches), và mọi callback write phải khớp input. Vi phạm được ghi theo callback và hậu kiểm, nên write hợp lệ sau đó không che được write sai. Nếu builder yêu cầu recalibration hoặc trả calibration table khác input, giữ partial/log và dừng.
- `--phase build` chỉ hợp lệ sau một `study_manifest.json` đã prepare/preflight và phải khớp toàn bộ settings, flags, Sigmoid, source/cache hashes và protocol variant. Capture dispatch dùng lại cùng build/study contract trước khi chạm engine.
- GPU guard được kiểm ở snapshot build trước khi build và ở snapshot capture; classification cuối đọc cả hai phase. Timing-cache coverage không được suy đoán từ output/hash và được ghi là `unknown` nếu API/log không cung cấp coverage.
- Output timing cache được lưu riêng và so hash với input. Khác hash là `timing_cache_output_changed` để review, không tự diễn giải là tactic đã đổi; giống hash cũng không chứng minh mọi tactic đã bị khóa.

## Lệnh server dự kiến (chỉ sau code review)

Các PID desktop phải lấy từ snapshot hiện tại của server; không dùng PID lịch sử nếu chưa đối chiếu. Thay `DESKTOP_PID_1=DESKTOP_PATH_1` và `DESKTOP_PID_2=DESKTOP_PATH_2` bằng các confirmation hợp lệ. Lệnh foreground, mỗi lệnh một dòng:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master
```

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/run_uniform_timing_cache_replay.py --out-dir results/measurement_audit_v1/server_uniform_timing_cache_replay_v1 --device 0 --confirm-desktop-process DESKTOP_PID_1=DESKTOP_PATH_1 --confirm-desktop-process DESKTOP_PID_2=DESKTOP_PATH_2
```

Không chạy cạnh training/build/benchmark khác trên GPU. Không cần GPU idle tuyệt đối nếu chỉ có desktop đã xác nhận; process không phân loại phải chặn. Không kill/pause/đổi priority/clock/power. Nếu lỗi tài nguyên hoặc partial output, giữ nguyên và báo, không resume/đổi batch/imgsz.

## Artifact inventory để push sau run

Giữ `model.engine`, source ONNX/weights và calibration source lớn trên server. Push JSON, binary cache và compressed verbose logs của study này; không push engine binaries. Các artifact bắt buộc:

- `study_manifest.json` và `repeat_summary.json`;
- mỗi `repeat_1..3`: `build_manifest.json`, `inspector.json`, `calibration_cache_input.cache`, `timing_cache_input.cache`, `timing_cache_output.cache`; nếu TensorRT gọi callback write thì thêm `calibration_cache_output.cache` và các file hậu tố `_2`, `_3`, … tương ứng từng callback;
- `build_1.log.gz`, `build_2.log.gz`, `build_3.log.gz`;
- mỗi repeat: `capture/capture_report.json`, `capture/validator_predictions.json`, `verification/*.json`, `comparison.json`, `execution_manifest.json`.

Sau khi đủ output, có thể kiểm tra danh sách trước khi stage:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && find results/measurement_audit_v1/server_uniform_timing_cache_replay_v1 -type f \( -name '*.json' -o -name '*.cache' -o -name '*.log.gz' \) -printf '%P\n' | sort
```

Chỉ stage artifact của study, không stage `model.engine`:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && find results/measurement_audit_v1/server_uniform_timing_cache_replay_v1 -type f \( -name '*.json' -o -name '*.cache' -o -name '*.log.gz' \) -print0 | xargs -0 env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git add -f --
```

Luna sẽ pull commit artifact, kiểm tra canonical Git blobs, input/cache hashes, calibration batches, timing attach/output, inspector, telemetry, capture/native/size và bảng range n=3. Kết luận chỉ là `replay_exact_observed`, `replay_variation_observed` hoặc `incomplete_or_invalid`; không chọn best build và không tự mở study tiếp theo.
