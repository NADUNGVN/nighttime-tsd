# Thẩm định hướng nghiên cứu Edge INT8 Traffic-Sign Detection và lộ trình journal Q2

## 1. Kết luận điều hành

Hướng nghiên cứu còn khả thi, nhưng bằng chứng hiện tại chưa đủ để khẳng định có một đóng góp mới về calibration, cũng chưa đủ để xem bản thảo là sẵn sàng nộp journal. Quy mô 15 checkpoint và nhiều thiết bị là tài sản thực nghiệm; nó không tự tạo ra tính mới khoa học. Không có cơ sở để bảo đảm chấp nhận Q2 hoặc đưa ra một tỷ lệ chấp nhận đáng tin cậy.

Khuyến nghị là **tiếp tục có điều kiện ở giai đoạn xác minh phép đo và giả thuyết; chưa mở rộng INT8 ra 15 model**. Giữ nguyên toàn bộ FP32 weights, không retrain, không bổ sung dữ liệu training. Không tiếp tục đổi calibration policy chỉ để tìm một kết quả tốt trên official test.

Hướng có triển vọng hơn để kiểm chứng là: **suy giảm chất lượng phát hiện theo kích thước biển báo và điều kiện ảnh khi lượng tử hóa, và ảnh hưởng của suy giảm đó tới lựa chọn model–precision–backend dưới giới hạn độ trễ và năng lượng**. Đây là đề xuất định vị nghiên cứu, chưa phải một kết luận đã chứng minh tính mới. Một bài chỉ có bảng mAP/FPS của 15 model vẫn có rủi ro trùng lặp cao.

Có một lựa chọn Q2 đã xác minh được từ thông báo nhà xuất bản: Journal of Imaging công bố Q2, hạng 18/39 ngành Imaging Science and Photographic Technology, với chỉ số năm 2025 được công bố tháng 6/2026. Cần kiểm tra đơn vị có chấp nhận journal, ngành xếp hạng và chi phí open access hay không. [1 — Journal of Imaging](https://www.mdpi.com/journal/jimaging/announcements/17195)

## 2. Tài sản thực nghiệm và sức mạnh của bằng chứng

Nghiên cứu đang có 15 checkpoint thuộc ba họ YOLOv8, YOLO11 và YOLO26, mỗi họ gồm n/s/m/l/x. CCTSDB2021 được chia 14.720 ảnh train và 1.636 ảnh dev; positive official test có 1.500 ảnh và negative stress test có 500 ảnh riêng biệt. Đây là bài toán phát hiện ba nhóm biển báo, không phải nhận dạng đầy đủ mã/ngữ nghĩa của từng loại biển giao thông.

Các checkpoint được giữ cố định. So sánh trước/sau lượng tử hóa trên cùng checkpoint là thiết kế phù hợp với câu hỏi deployment. Tuy nhiên, 15 checkpoint không phải 15 mẫu độc lập của một quần thể kiến trúc: chúng là ba họ ở năm quy mô, cùng một dataset và một training seed. Không thể từ đó suy ra độ biến thiên do retraining hay kết luận một họ luôn tốt hơn trên mọi lần train.

Config architecture matrix có batch vật lý thay đổi theo quy mô: 64, 48, 32, 24, 16. Config có chủ đích dùng gradient accumulation với nominal batch 64, nhưng điều đó không chứng minh tất cả động lực tối ưu hóa hoàn toàn giống nhau. Cần mô tả đúng đây là checkpoint được train theo protocol chung với điều chỉnh tài nguyên, không viết “mọi hyperparameter giống hệt”. Không cần retrain để sửa cách trình bày này.

### 2.1. VCSC v1: kết quả hiện có

Các số dưới đây lấy từ báo cáo fresh-cache, không lấy từ những engine đã bị cách ly vì tái sử dụng calibration cache. Đơn vị mAP và SD trong bảng là phần trăm; Delta là điểm phần trăm so với FP16 trên cùng positive test.

| Chính sách | Số selection độc lập | Full mAP50 | Full mAP50–95 | Macro-domain Delta mAP50 | SD macro Delta |
|---|---:|---:|---:|---:|---:|
| FP16 reference | Không áp dụng | 76,765 | 48,726 | 0 | Không áp dụng |
| Uniform | 5 | 71,689 ± 0,481 | 39,142 ± 1,331 | −6,347 | 1,126 |
| Low-Luminance | 1 | 72,479 | 40,049 | −5,205 | Chưa ước lượng |
| VCSC equal-quota | 5 | 71,623 ± 1,089 | 39,091 ± 1,140 | −6,228 | 1,830 |

VCSC v1 chỉ tốt hơn Uniform khoảng 0,119 điểm ở macro Delta, trong khi full mAP50 hơi thấp hơn và SD macro lớn hơn khoảng 0,703 điểm. Nó không vượt qua decision gate đang lưu trong dự án. Điều được hỗ trợ là **không scale VCSC v1 theo protocol đó**, không phải “mọi phương pháp calibration theo điều kiện đều vô ích”.

Low-Luminance chọn đúng nhóm ảnh tối nhất bằng ranking. Năm tên seed có cùng selection hash không tạo thành năm selection độc lập. Giá trị SD bằng 0 trong JSON tổng hợp một run không phải bằng chứng phương pháp không biến thiên; trong manuscript nên ghi không áp dụng/chưa ước lượng, và tách build-repeat variability nếu đo thêm.

VCSC v1 còn đổi khởi tạo K-means theo seed. Vì vậy, biến thiên đang bao gồm thay đổi phân cụm lẫn lấy mẫu trong cụm; nếu build engine riêng, còn có thể có biến thiên do compiler/tactic. Không gọi toàn bộ SD này là thuần túy calibration-image sampling noise.

### 2.2. VCSC v2: chưa đủ bằng chứng để bác bỏ thống kê

V2 chuyển từ quota bằng nhau sang quota gần tỷ lệ số ảnh trong cụm. Đây là thay đổi cách tái trọng số phân phối train, không phải một phương pháp lượng tử hóa mới. Mục đích kiểm chứng hợp lý là xem việc lấy quá nhiều ảnh ở cụm hiếm có liên quan đến kết quả v1 hay không; chưa thể khẳng định đó là nguyên nhân nếu chưa có kiểm soát.

Kết quả dev hiện có chỉ cho seed 42:

| Representation/policy | mAP50 (%) | mAP50–95 (%) | XS AP50 (%) | S AP50 (%) |
|---|---:|---:|---:|---:|
| FP16 | 97,775 | 76,302 | 50,482 | 94,782 |
| Uniform INT8 | 96,036 | 66,230 | 36,583 | 91,133 |
| Low-Luminance INT8 | 95,492 | 64,851 | 39,259 | 89,515 |
| VCSC-proportional INT8 | 95,007 | 64,358 | 38,261 | 86,793 |

V2 thấp hơn Uniform khoảng 1,029 điểm mAP50 và 1,873 điểm mAP50–95 ở seed này; XS lại cao hơn khoảng 1,678 điểm. Không có cơ sở gọi đây là ưu thế chung, cũng không có cơ sở kết luận đã hoàn thành kiểm định năm seed. Có thể tạm dừng vì lợi ích kỳ vọng thấp so với chi phí, nhưng phải ghi rõ đó là quyết định phân bổ nguồn lực.

Các cột XS/S hiện là diagnostic từ evaluator riêng, cần xác minh ở mục 4. Không so trực tiếp 95% trên dev với 72% trên official test như thể đó là hiệu quả thay đổi phương pháp: hai tập khác nhau.

## 3. Đối chiếu tính mới với nghiên cứu đã có

### 3.1. Công trình gần nhất

Karimov, Imani và Kazakov nghiên cứu YOLO12 n/s/m/l/x, nhiều precision, ảnh suy giảm và calibration pha ảnh sạch/ảnh suy giảm; kết quả calibration mới không cải thiện nhất quán. Bản hiện hành trên arXiv là v3 tháng 5/2026, lần đầu công bố tháng 8/2025. Đây là công trình đối chiếu trực tiếp cho ý tưởng hiện tại, dù không nên coi trạng thái arXiv tự nó chứng minh đã peer review. [2 — Quantization Robustness to Input Degradations for Object Detection](https://arxiv.org/html/2508.19600v3)

Hệ quả: chỉ thay COCO bằng CCTSDB, YOLO12 bằng ba họ YOLO, hoặc tăng số thiết bị chưa đủ để tuyên bố một gap mới. Cần chỉ ra câu hỏi khác và bằng chứng mới, không chỉ mở rộng bảng benchmark. Không cần chấp nhận mọi kết luận của công trình này là chân lý để nhận ra phần giao nhau về ý tưởng.

### 3.2. Ma trận liên quan và ranh giới đóng góp

| Công trình | Nội dung liên quan đã được nghiên cứu | Ranh giới cần giữ cho paper này |
|---|---|---|
| QRID, 2025; v3 năm 2026 | Quy mô YOLO, precision, suy giảm ảnh và calibration theo suy giảm | Không nhận tính mới cho tổ hợp các yếu tố này; cần phân tích riêng về lỗi/kích thước và quyết định deployment. [2](https://arxiv.org/abs/2508.19600) |
| Benchmarking the Reliability of PTQ, 2023 | Độ tin cậy PTQ và worst-case performance | Không gọi worst-domain retention là ý tưởng hoàn toàn mới. [3](https://arxiv.org/abs/2303.13003) |
| Reg-PTQ, CVPR 2024 | Khó khăn lượng tử hóa nhánh regression và giải pháp chuyên biệt | “Localization nhạy với lượng tử hóa” không mới nếu đứng một mình. Nếu đề xuất giải pháp, phải đối chiếu nhóm phương pháp này. [4](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_Reg-PTQ_Regression-specialized_Post-training_Quantization_for_Fully_Quantized_Object_Detector_CVPR_2024_paper.html) |
| InlierQ, preprint 2026 | Calibration/activation hướng tới thông tin hữu ích cho detection, thay vì clutter/noise | Không xem global visual diversity tự động là tiêu chí calibration tốt nhất. [5](https://arxiv.org/abs/2602.03472) |
| TIDE, ECCV 2020 | Công cụ phân rã các loại lỗi detection | Dùng như phương pháp phân tích có sẵn; không nhận đóng góp thuật toán cho việc phân nhóm lỗi. [6](https://arxiv.org/abs/2008.08115) |
| AgriJetsonBench, preprint 2026 | TensorRT, Jetson, đo nguồn ngoài và hiệu quả năng lượng | Nhiều Jetson + FPS + J/inference cũng đã có tiền lệ. [7](https://arxiv.org/abs/2608.00927) |

Tổng hợp: một hướng thực nghiệm có thể đóng góp nếu tìm ra và kiểm chứng quan hệ **model scale × precision × kích thước vật thể × điều kiện ảnh**, rồi chứng minh quan hệ đó thay đổi lựa chọn triển khai. Trong các nguồn đối chiếu ở đây chưa thấy một nghiên cứu hoàn toàn trùng tổ hợp CCTSDB ba lớp, 15 checkpoint cố định và TensorRT/Hailo với phân tích lỗi theo kích thước. Điều này là khoảng trống ứng viên, không chứng minh quyền tuyên bố “first”.

Không nên tiếp tục “sáng chế” các tên calibration policy khi chưa có giả thuyết về activation hoặc lỗi tác vụ. Nếu vẫn muốn bài phương pháp, cần một cơ chế mới cùng ablation và baseline mạnh hơn Uniform/Low-Luminance; chi phí và rủi ro tăng đáng kể so với bài empirical deployment được thiết kế tốt.

## 4. Những vấn đề phải giải quyết trước khi tăng số thí nghiệm

### 4.1. Tính nhất quán của evaluator

`evaluate_cctsdb.py` gọi `model.val()` để lấy mAP, sau đó gọi `model.predict()` riêng để lưu prediction cho size/bootstrap. Chưa thể mặc định hai đường này có preprocessing, letterbox, postprocessing và matching hoàn toàn giống nhau. Cần tái tính mAP từ prediction đã lưu, đối chiếu với kết quả gốc và giải thích sai khác; tốt hơn là dùng một nguồn prediction chuẩn cho các phân tích liên quan.

`evaluate_cctsdb_size.py` hiện tính AP50 và bỏ qua prediction khớp với bất kỳ GT cùng lớp ngoài bin. Nó không quản lý one-to-one matching cho nhóm GT bị ignore theo cùng cách với GT trong bin. Đây là khác biệt cần kiểm tra bằng unit test và evaluator tham chiếu, không phải bằng chứng mọi số liệu size đều sai. COCO evaluator cung cấp triển khai công khai của matching, ignore, area ranges và nhiều IoU thresholds để đối chiếu. [8 — COCOeval](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)

Giữ ngưỡng diện tích CCTSDB thay vì thay bằng COCO small/medium/large. Tuy nhiên, bài gốc ghi XS/S/M/L/XL lần lượt 813/807/828/408/372, còn kết quả hiện tại là 813/823/812/408/372. Tổng cùng 3.228 nhưng có lệch ở S/M. Cần đối chiếu phiên bản XML, convention diện tích, ngưỡng biên và annotation chuyển đổi; không sửa ngưỡng để ép số đếm khớp. [9 — CCTSDB 2021, mục 3.3.2 và Bảng 1](https://centaur.reading.ac.uk/106129/1/12-23.pdf)

Join XML theo tên member đã khắc phục lỗi ID nhưng chưa đủ chứng minh geometry đúng. Phải so bbox XML với YOLO label trên đúng kích thước ảnh, class mapping và việc clipping. Bổ sung AP50–95 theo size nếu lấy small-object localization làm câu hỏi chính.

### 4.2. Tách export loss và quantization loss

Một đối chứng hữu ích là source FP32 → TensorRT FP32 → TensorRT FP16 → TensorRT INT8, cùng ảnh và cùng quy trình tính metric. So sánh trực tiếp PyTorch FP32 với TensorRT INT8 có thể trộn ảnh hưởng precision, export, preprocessing và backend.

Các log hiện có cho biết một số Sigmoid giữ FP32. Do đó dùng nhãn “TensorRT INT8 mixed-precision engine”, không viết “toàn bộ model W8A8” khi chưa có layer inventory. Input/output FLOAT không đồng nghĩa engine không có INT8 nội bộ. Tài liệu NVIDIA phân biệt cơ chế implicit quantization với explicit quantization và lựa chọn precision trong engine. [10 — TensorRT 10.x, Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)

Fresh per-engine workspace cùng calibration-cache hash khác nhau là tiến bộ quan trọng về traceability. Vẫn cần kiểm tra calibrator thực sự đọc đúng 1.024 ID, không bỏ ảnh do batch/drop-last, không dùng nhầm split, và không mang cache cũ vào một lần build mới. Không sử dụng các artifact đã cách ly làm bằng chứng policy.

### 4.3. Giới hạn của official test

Không thấy cơ sở để kết luận train đã dùng official test từ các log hiện có. Tuy nhiên, kết quả official test đã được xem để đánh giá v1 và định hướng bước tiếp theo. Đây là rủi ro adaptive method selection, khác với việc trực tiếp đưa test vào training hoặc calibration.

Vì vậy, không tiếp tục gọi mọi kết quả tiếp theo trên cùng test là hoàn toàn untouched confirmatory evidence. Khóa thiết kế từ bây giờ là cần thiết nhưng không xóa lịch sử thích nghi. Báo cáo v1 là exploratory; phát triển tiếp trên dev, ghi mốc đóng băng và giới hạn suy luận.

Một tập evaluation độc lập sẽ tăng sức thuyết phục, nhưng không có cơ sở nói mọi journal bắt buộc hai dataset. Nếu giữ đúng một dataset, phải định vị là nghiên cứu có điều kiện trên CCTSDB và chấp nhận rủi ro generalization lớn hơn. External evaluation-only có thể được cân nhắc sau khi có chấp thuận; nó không phải external training và không yêu cầu retrain. Mapping nhãn và chất lượng annotation phải được kiểm tra trước, không tự gộp mọi mã biển theo prefix.

### 4.4. Không trộn các nguồn biến thiên

Phân biệt bốn nguồn: mẫu ảnh evaluation, selection calibration, khởi tạo clustering, và build/runtime. Muốn đo riêng sampling stability của VCSC thì giữ partition train cố định và thay sampling seed; muốn đo toàn thuật toán thì thay cả partition nhưng phải đặt tên đúng. Giữ nguyên báo cáo v1, không viết lại lịch sử như thể đã dùng thiết kế mới.

Bootstrap phải paired theo ảnh: FP16 và INT8 dùng cùng resample, sau đó tính lại AP của tập resample. Không trung bình “AP của từng ảnh”. Dùng khoảng 1.000 resample ban đầu; lưu seed, quy tắc khi domain không có một class, CI và số ảnh/instances. Nếu ảnh thuộc chuỗi gần nhau, cần cân nhắc resampling theo nhóm/sequence thay vì giả định ảnh độc lập.

Macro-domain gồm sáu domain cơ bản, không cộng daylike như domain độc lập thứ bảy. Foggy ít ảnh cần CI và sensitivity analysis, không dùng riêng raw foggy mAP để chọn phương pháp. Negative test dùng confidence khóa trước, chẳng hạn 0,25/0,50/0,75; không chọn threshold tốt nhất từ chính negative test.

## 5. Scope đề xuất và phần thực sự thay đổi

### 5.1. Phần giữ nguyên

Giữ dataset training, 15 source checkpoint, bài toán ba lớp, kích thước input 640 và hướng post-training deployment. Uniform và Low-Luminance vẫn là baseline có ích. VCSC v1 là một negative result có giá trị nếu trình bày đúng; v2 là exploratory dev pilot chưa hoàn tất đánh giá stability.

Giữ toàn bộ 15 model trong accuracy study sau khi qua gate, nhưng không bắt buộc mọi model phải chạy trên mọi thiết bị. Phạm vi phần cứng là Xavier NX, AGX Xavier, Pi5 CPU, Pi5 + Hailo 26 TOPS và Jetson Nano; lựa chọn subset cho thí nghiệm chính dựa trên câu hỏi và tương thích, không dựa trên thiết bị nào cho số đẹp nhất.

### 5.2. Phần thay đổi cần được thống nhất trước triển khai

Thay đổi chính là **không đặt thành công của paper vào tuyên bố VCSC vượt Uniform**. Thay vào đó kiểm chứng ba câu hỏi:

1. **RQ1 — Accuracy/error:** Với cùng checkpoint, PTQ ảnh hưởng tới lỗi localization, miss và background false positives như thế nào khi kiểm soát kích thước và điều kiện ảnh?
2. **RQ2 — Selection:** Model tốt nhất theo full FP32 mAP có còn là lựa chọn tốt dưới giới hạn tail latency, năng lượng và small-sign performance không?
3. **RQ3 — Backend:** Những kết luận nào giữ được, và những kết luận nào đổi, khi chuyển từ TensorRT sang quantization native của Hailo với cùng source weights và calibration IDs?

Một câu trả lời âm tính nhưng chặt chẽ vẫn có thể hữu ích: chẳng hạn kết luận xếp hạng không thay đổi trong miền khảo sát. Tuy nhiên, nếu tất cả kết quả chỉ lặp lại kiến thức “INT8 nhanh hơn và kém chính xác hơn”, tính mới vẫn yếu. Không được định trước rằng INT8 nhất định phải làm biển nhỏ giảm nhiều hơn, hoặc nhất định phải có rank reversal.

Đóng góp ứng viên có thể gồm một protocol kiểm chứng sai số theo tác vụ và một quy tắc chọn cấu hình deployment trên dev. Quy tắc phải so với baseline đơn giản như chọn full-mAP cao nhất trong cùng budget. Pareto plot đơn thuần là công cụ phân tích, không phải thuật toán mới; cần cho thấy nó dẫn tới quyết định có ích với bằng chứng kiểm chứng.

Nếu chọn nhánh intervention, chỉ thử sau khi phân tích lỗi chỉ ra mục tiêu: ví dụ giữ precision cao ở một phần head và đo mức phục hồi AP so với chi phí latency/energy. Đây là ablation đề xuất, chưa được thực hiện, không được trình bày là phát minh mới; các công trình regression-aware PTQ đã tồn tại. Nhánh này không cần cập nhật source weights nhưng vẫn cần thống nhất scope và chi phí trước khi làm.

## 6. Thiết kế thực nghiệm tối thiểu có các điểm dừng

Các KPI dưới đây là tiêu chuẩn chất lượng nội bộ đề xuất, không phải điều kiện chính thức của journal và không bảo đảm acceptance.

| Giai đoạn | Công việc | Điều kiện hoàn tất / quyết định |
|---|---|---|
| G0: correctness | Audit label/XML/prediction, evaluator, model hash, calibration IDs, cache và precision | Không còn sai khác không giải thích được; unit tests matching/ignore đạt; mọi artifact có provenance; không chạy matrix khi chưa đạt |
| G1: scientific pilot | YOLO11n hiện có; phân rã lỗi dev, paired CI, kiểm soát pipeline | Có giả thuyết đo được và loại được nguyên nhân phần mềm; nếu chỉ có lỗi pipeline thì sửa phép đo trước, không rút kết luận khoa học |
| G2: replication nhỏ | Thêm YOLOv8n và YOLO26n với weights có sẵn, cùng ID Uniform và FP16 | Xem quan sát có lặp lại giữa họ hay chỉ là checkpoint cụ thể; không đòi hiệu ứng phải cùng dấu để “cho phép” báo cáo |
| G3: accuracy matrix | Mở rộng 15 checkpoint với protocol precision đã khóa | Full/domain/class/size metrics, raw predictions và failure registry đủ; sampling repeats ưu tiên model đại diện, rồi mở rộng nếu kết luận phụ thuộc seed |
| G4: edge study | Hai model đại diện trên một Xavier và Pi5 + Hailo trước | Model thực sự tương thích; end-to-end latency, power boundary, memory, thermal và accuracy sau compile đều được kiểm tra |
| G5: submission gate | Tổng hợp novelty, CI, limitation, code và data manifests | Có câu trả lời vượt bảng benchmark mô tả; mọi claim truy được đến số liệu; coauthor duyệt journal/ranking và tính nhất quán của protocol |

Không cần lập tức tạo thêm mọi calibration size và mọi policy trên 15 model. Ncal ablation 128/256/512/1024 chỉ nên chạy nếu calibration size là một câu hỏi của bản thảo; ưu tiên YOLO11n và một baseline rõ ràng. Không cần tạo lại engine Uniform đã hợp lệ chỉ vì thư mục đổi tên: trước hết kiểm tra weight hash, selection IDs, cache provenance và runtime/export arguments có cho phép tái sử dụng không.

Nếu tiếp tục tuyên bố VCSC v2 là một phương pháp tốt hơn, bắt buộc phải đánh giá đủ repeats đúng thiết kế và sửa các vấn đề ở G0 trước. Nếu paper chuyển trọng tâm, có thể dừng v2 với nhãn pilot chưa đủ bằng chứng, không cần tiêu GPU chỉ để hoàn thiện một nhánh không còn phục vụ câu hỏi chính.

### 6.1. Endpoint đề xuất

Giữ Delta mAP50 và mAP50–95 so với FP16 làm kết quả paired. Báo cáo cả giá trị tuyệt đối và retention; tỷ số cần xử lý khi reference thấp hoặc bằng 0. Small-object endpoint cần XS/S AP50–95 và recall được định nghĩa ở threshold cố định trên dev; không thay endpoint sau khi xem test để có significance.

Với phân tích điều kiện, cần bảng domain × size × class và số mẫu từng ô. Domain tối có thể chứa nhiều biển nhỏ hơn; raw mAP giảm không tự chứng minh tác động của ánh sáng. Chỉ nói association khi không có thiết kế kiểm soát; mọi kết luận về cơ chế phải dựa trên kiểm tra riêng.

Statistical significance không thay thế ý nghĩa thực tiễn. Trước khi chạy lớn, xác định mức chênh lệch đủ ảnh hưởng deployment, ví dụ một mức giảm năng lượng có ý nghĩa trong khi không mất quá tolerance AP đã khóa. Các mức như 0,5 điểm AP hoặc 10% năng lượng chỉ là lựa chọn thiết kế cần lý giải, không phải “ngưỡng Q2”.

### 6.2. Hardware protocol

Không đưa Jetson Nano bản gốc vào nhóm TensorRT INT8 GPU như Xavier. NVIDIA hướng dẫn Nano dùng FP32/FP16; dùng thiết bị này làm baseline khả năng thấp nếu cần. Không nhầm Jetson Nano với Orin Nano. [11 — NVIDIA, Can Jetson Nano import INT8?](https://forums.developer.nvidia.com/t/can-jetson-nano-import-int8/178500)

TensorRT engine trên RTX8000 không được mặc định mang sang Xavier chạy nguyên trạng. Build và kiểm tra backend trên nền tảng đích, lưu JetPack/CUDA/TensorRT và các precision fallback. Hailo dùng pipeline native; thống nhất source weights và calibration IDs chứ không tuyên bố hai backend là cùng một quantized numerical model. [12 — Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)

Thiết kế đo khởi điểm đề xuất: batch 1, ít nhất 100 warm-up, 1.000 inference được đo cho mỗi run, tối thiểu ba run độc lập, thêm kiểm tra sustained khoảng 15 phút cho nhiệt. P99 với 1.000 mẫu còn nhạy nên tăng mẫu/repeats nếu là endpoint chính. Ghi median/p95/p99, throughput thực đo, deadline misses, peak memory và nhiệt độ/throttling.

Tách kernel/model latency khỏi end-to-end latency gồm preprocess, chuyển dữ liệu và postprocess. Nguồn điện đo tại cùng boundary, ưu tiên đầu vào toàn thiết bị; trên Pi+Hailo phải tính cả Pi nếu so với Jetson toàn board. J/image lấy năng lượng tích phân chia số ảnh xử lý, nêu rõ có trừ idle hay không. Không suy năng lượng bằng công suất danh nghĩa TOPS/TDP hoặc coi 1/median latency là throughput đo trực tiếp.

Không gọi hệ thống đáp ứng an toàn giao thông chỉ từ ảnh tĩnh và độ trễ. Ngưỡng deadline phải gắn với workload; không suy khoảng cách cảnh báo từ diện tích bbox nếu không có camera geometry và khoảng cách ground truth.

## 7. Journal và cách hiểu mục tiêu Q2

Q2 phải kèm hệ xếp hạng, năm và subject category. Clarivate quy định journal có rank/quartile cho từng category; một journal có thể có nhiều quartile. JCR/JIF, SJR và CiteScore là các hệ khác nhau, không nên dùng thay thế nhau khi trường chỉ định một hệ. [13 — Clarivate, A Primer on Ties in the JCR](https://clarivate.com/academia-government/blog/a-primer-on-ties-in-the-jcr/)

| Journal | Độ phù hợp với bài đề xuất | Trạng thái xác minh và quyết định |
|---|---|---|
| Journal of Real-Time Image Processing | Phù hợp mạnh nếu có timing/energy, thiết kế triển khai và trade-off thực sự | Aims/scope chính thức đã xác minh. Chưa xác nhận quartile hiện hành bằng bản ghi JCR chính thức trong hồ sơ này; cần trường/thư viện xác nhận trước khi coi là lựa chọn đáp ứng Q2. [14](https://link.springer.com/journal/11554/aims-and-scope) |
| Journal of Imaging | Phù hợp ứng viên cho nghiên cứu imaging/detection thực nghiệm chặt chẽ | Nhà xuất bản công bố Q2 năm chỉ số 2025, ngành Imaging Science and Photographic Technology. Là lựa chọn có chứng cứ xếp hạng cụ thể; không đồng nghĩa dễ được nhận. Kiểm tra chính sách đơn vị và APC. [1](https://www.mdpi.com/journal/jimaging/announcements/17195), [15](https://www.mdpi.com/journal/jimaging/about) |
| Image and Vision Computing | Có thể cân nhắc nếu đóng góp về vision/analysis đủ rõ, không chỉ deployment bảng số | Chưa có cơ sở xem là lựa chọn “an toàn”. Không gán quartile hiện hành khi chưa có xác nhận đúng hệ/ngành/năm; trang aims/scope ScienceDirect không truy cập đầy đủ trong hồ sơ nguồn này |
| Computer Vision and Image Understanding | Chỉ nên giữ là mục tiêu mở rộng nếu có đóng góp về hiểu cơ chế/vision đủ mạnh | Không ưu tiên cho bản benchmark mô tả hiện tại; đây là đánh giá rủi ro nội dung, không phải tuyên bố journal cấm benchmark |

JRTIP nêu rõ manuscript cần giải quyết vấn đề real-time; một bài chỉ accuracy không nằm đúng trọng tâm. Trang scope còn công bố giới hạn 12 trang double-column, gồm references và bio; cần kiểm tra lại hướng dẫn tại thời điểm nộp. Do đó chọn JRTIP kéo theo yêu cầu hardware/implementation có substance, không thể chỉ đổi tiêu đề. [14 — JRTIP scope](https://link.springer.com/journal/11554/aims-and-scope)

Không nên lập danh sách journal Q2 từ các website tổng hợp rồi mặc định đáp ứng quy định trường. Với mục tiêu tối thiểu Q2, nên có ít nhất hai journal được đơn vị xác nhận trước khi đầu tư phần thực nghiệm lớn. Hồ sơ hiện mới có chứng cứ publisher cụ thể cho Journal of Imaging; JRTIP là ứng viên rất phù hợp về scope nhưng còn bước xác minh hành chính.

## 8. Mức sẵn sàng và quyết định cuối

| Hạng mục | Hiện trạng |
|---|---|
| Frozen training assets | Có cơ sở tái sử dụng; không cần retrain |
| Train-only calibration và cache isolation | Đã tiến bộ rõ; cần audit sự tiêu thụ ảnh thực tế và provenance đầy đủ |
| VCSC vượt Uniform | Chưa được chứng minh; v1 không qua gate, v2 một seed |
| Small-object result đáng dùng làm main claim | Chưa; còn reconciliation annotation và evaluator |
| Causal explanation của accuracy loss | Chưa; hiện mới có tín hiệu từ chênh lệch metrics |
| Statistical reliability | Có repeats của một số policy; chưa đủ CI và tách nguồn biến thiên |
| Cross-device accuracy/energy evidence | Chưa có bộ kết quả kiểm chứng hoàn chỉnh trong hồ sơ hiện tại |
| Distinct contribution so với prior art | Có hướng ứng viên; chưa đủ để khẳng định novelty |
| Journal Q2 candidate | Có một ứng viên xác minh được từ publisher; acceptance không bảo đảm |

**Quyết định: GO có điều kiện cho G0–G1; NO-GO cho mở rộng INT8 15 model ngay lúc này.** Không tiếp tục “train thêm để có paper”. Công việc có giá trị cao nhất tiếp theo là xác nhận phép đo từ artifact hiện có, sau đó chốt câu hỏi và pilot nhỏ trên dev.

Nếu sau audit không có quan sát hoặc quyết định triển khai vượt khỏi prior art, cần dừng tăng số model/device và xem lại đóng góp trước khi chi thêm thời gian. Nếu có bằng chứng lặp lại, evaluator đáng tin và hardware study giải thích được trade-off, có cơ sở xây dựng bản thảo hướng journal Q2; đó vẫn là triển vọng có điều kiện, không phải cam kết xuất bản.

## 9. Hồ sơ nguồn và giới hạn thẩm định

Mốc thẩm định: 09/09/2026. Local repository tại commit `88045db`; báo cáo không thay đổi code, checkpoint, protocol đang chạy hoặc kết quả lịch sử. Các nhận xét code là kiểm tra tĩnh; chưa thực hiện lại inference hoặc xác minh toàn bộ dữ liệu raw trên server.

Các bản ghi nội bộ đối chiếu gồm `configs/architecture_matrix_v1.json`, `configs/yolo11n_calibration_development_v2.json`, `scripts/evaluate_cctsdb.py`, `scripts/evaluate_cctsdb_size.py`, `scripts/bootstrap_cctsdb_map50.py`, `scripts/export_tensorrt.py`, báo cáo v1 tại `results/calibration_method_v1/rtx8000/yolo11n/summary_freshcache_v2/decision_report.json`, và bốn kết quả `dev_eval`/`dev_size` của v2. Log server về cache quarantine và calibration stability bổ sung ngữ cảnh provenance.

Tìm kiếm tập trung vào PTQ cho object detection, calibration robustness, regression/localization, traffic signs và heterogeneous edge deployment. Nguồn chủ yếu là công trình gốc, tài liệu nhà cung cấp và trang nhà xuất bản. Đây không phải systematic review có đăng ký protocol; không phát hiện một bài trùng hoàn toàn không chứng minh không tồn tại bài đó. Các preprint được ghi rõ trạng thái, không đánh đồng với công trình peer-reviewed.

Một số trang trả lỗi truy cập hoặc giới hạn request; khi có bản chỉ mục nội dung từ chính nguồn gốc, chỉ sử dụng phần được truy xuất, không suy rằng đã kiểm tra đầy đủ PDF. Chênh lệch số đếm size CCTSDB cần xác minh lại trên bản PDF/archive chính thức trước khi sửa evaluator. Thứ hạng journal phải kiểm tra lại theo quy định đơn vị và năm xét công bố thực tế.

### Nguồn công khai

1. MDPI. [Journal of Imaging Receives an Updated Impact Factor of 3.8](https://www.mdpi.com/journal/jimaging/announcements/17195), 25/06/2026. Xác nhận publisher về JIF/quartile năm 2025.
2. Karimov, T.; Imani, H.; Kazakov, A. [Quantization Robustness to Input Degradations for Object Detection](https://arxiv.org/html/2508.19600v3), arXiv:2508.19600, 2025; v3 01/05/2026. Công trình gần nhất về scope.
3. [Benchmarking the Reliability of Post-training Quantization: a Particular Focus on Worst-case Performance](https://arxiv.org/abs/2303.13003), 2023. PTQ reliability/worst-case.
4. Ding, Y.; Feng, W.; Chen, C.; Guo, J.; Liu, X. [Reg-PTQ: Regression-specialized Post-training Quantization for Fully Quantized Object Detector](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_Reg-PTQ_Regression-specialized_Post-training_Quantization_for_Fully_Quantized_Object_Detector_CVPR_2024_paper.html), CVPR 2024, pp. 16174–16184.
5. Kim, M. và cộng sự. [Inlier-Centric Post-Training Quantization for Object Detection Models](https://arxiv.org/abs/2602.03472), preprint, 03/02/2026.
6. Bolya, D. và cộng sự. [TIDE: A General Toolbox for Identifying Object Detection Errors](https://arxiv.org/abs/2008.08115), ECCV 2020.
7. [AgriJetsonBench: External-Power-Referenced TensorRT Benchmarking of Agricultural Vision Models on Jetson Edge Platforms](https://arxiv.org/abs/2608.00927), preprint, 2026.
8. COCO Consortium. [Python COCO evaluation implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py). Tham chiếu matching/ignore/area metrics; cần pin commit khi triển khai.
9. [CCTSDB 2021: A More Comprehensive Traffic Sign Detection Benchmark](https://centaur.reading.ac.uk/106129/1/12-23.pdf), Human-centric Computing and Information Sciences, 2022, bản lưu University of Reading. Mục 3.3.2 và Bảng 1.
10. NVIDIA. [TensorRT 10.x — Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html). Precision và calibration.
11. NVIDIA Developer Forums. [Can Jetson Nano import INT8?](https://forums.developer.nvidia.com/t/can-jetson-nano-import-int8/178500), phản hồi NVIDIA 20/05/2021. Chỉ áp dụng Jetson Nano gốc, không phải Orin Nano.
12. Hailo. [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo). Quy trình backend-native; không dùng trang này để khẳng định mọi model YOLO26 đều tương thích.
13. Clarivate. [A Primer on Ties in the JCR](https://clarivate.com/academia-government/blog/a-primer-on-ties-in-the-jcr/), 2023. Rank/quartile theo category.
14. Springer Nature. [Journal of Real-Time Image Processing — Aims and scope](https://link.springer.com/journal/11554/aims-and-scope). Phù hợp nội dung và giới hạn manuscript.
15. MDPI. [Journal of Imaging — Aims and scope](https://www.mdpi.com/journal/jimaging/about). Kiểm tra phạm vi imaging và chính sách trước submission.
