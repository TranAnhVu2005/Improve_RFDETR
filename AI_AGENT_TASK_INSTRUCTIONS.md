# AI Agent Task Instructions

## Mục tiêu và thứ tự đọc

Đọc `AI_AGENT_CONTEXT.md` trước, rồi chỉ đọc phần code/paper cần cho nhiệm vụ. Người dùng muốn **tài liệu hướng dẫn và context được thiết kế lại** để agent triển khai nhánh RF-DETR cải tiến dạng ablation. Không sửa code/training khi chưa được giao cụ thể. Mọi run cải tiến phải giữ protocol bất biến trong context; toàn bộ thuật toán được điều khiển bằng config boolean.

## Nguyên tắc

1. Audit repo, `git status`, config thật, entry point và API trước khi sửa. Giữ nguyên thay đổi có sẵn của người dùng; không revert/ghi đè chúng.
2. Không bịa công thức từ tên module. Đối chiếu paper đúng section/equation với code; ghi `Exact`, `Adaptation`, `Incorrect` hoặc `Unverified`.
3. Xác minh API RF-DETR thực tế theo đúng version cài đặt (model, matcher, criterion, loss keys, train, checkpoint). Không dựa vào API phiên bản khác.
4. Giữ baseline riêng. Baseline và improved phải cùng dataset snapshot/splits, class mapping, 640×640, A100, patience 10, epoch cap 80, checkpoint-selection rule và đánh giá.
5. Mọi kỹ thuật phải bật/tắt được qua config mà không code edit; config phải là nguồn sự thật duy nhất cho toggle và hyperparameters. Thiếu field/giá trị sai phải gây lỗi rõ, không fallback âm thầm.
6. Không double-count/nuốt loss gốc, không để loss key mồ côi, không dùng `except Exception: pass`, không tuyên bố cải thiện nếu thiếu benchmark.
7. Đánh giá mAP@0.5:0.95 với bước IoU 0.05, confidence 0.01 và confusion matrix được nhóm chạy bằng script riêng sau khi nhận best checkpoint. Agent chỉ tích hợp/đảm bảo tạo và bàn giao best checkpoint đúng; không thay thế script/không báo metric giả.
8. Không chạy training dài hoặc GPU/Modal tốn phí nếu chưa có sự cho phép rõ ràng. Static checks và test cục bộ chỉ khi phù合 phạm vi công việc.
9. Tài liệu paper gốc trong `BaiBaoDangMarkDown/` là nguồn, không rút gọn hoặc sửa để thay thế toàn văn. Dùng `README_PAPERS.md` để định tuyến tới paper/section.
10. Không lộ secrets, không commit API key/token. Mọi thay đổi phải có diff nhỏ và truy vết được.

## Protocol bất biến — kiểm tra trước mọi run

- Cùng dataset, split train/val/test và cùng kích thước ảnh 640×640.
- Split ratio hiện chưa được xác định bởi chủ đề/tài liệu. Trước run, đọc split hiện có hoặc được phê duyệt; ghi chính xác số lượng, tỉ lệ, seed/phương thức và xác nhận giống nhau ở mọi run. Không tự đoán tỉ lệ. Không dùng test để chọn model/hyperparameter.
- 32 lớp, báo cáo 1-based: index 1 `beef`, index 32 `tomato`; khóa đầy đủ mapping từ dataset gốc và đối chiếu COCO IDs trước training. Nếu annotation không khớp, dừng và báo cáo.
- A100, patience 10, tối đa 80 epochs, image size 640×640 cho baseline/improved/ablation.
- Cùng seed và mọi training kwargs không nằm trong danh sách ablation. RFO không được tự đổi các ràng buộc protocol này.
- Best checkpoint theo một quy tắc validation cố định, ghi trong config/log. Sau đó script nhóm đánh giá với confidence 0.01, IoU sweep bước 0.05, confusion matrix.

## Kế hoạch triển khai khi được giao code

### A. Audit và ma trận paper → code

Trước patch, lập bảng cho QAR, DQCD, IQCM, HIFRM, SCU, LUE, Weighted Tversky, HNM, RFO:

| Kỹ thuật | Paper/section/equation | Tensor & mục tiêu | Điểm hook/loss trong RF-DETR | Khác biệt/giả định | Trạng thái | Bằng chứng cần |
|---|---|---|---|---|---|---|

Đọc `BaiBaoDangMarkDown/README_PAPERS.md` để tìm đúng paper. Paper về ảnh dưới nước, ảnh suy giảm, bệnh học/urine microscopy không tương đương food dataset; nêu rõ transfer/adaptation. Với mỗi loss, xác minh assignments, targets, normalization/reduction, gradient và empty-target behavior.

### B. Audit tích hợp RF-DETR

- Xác minh version, signatures/source của model, criterion, matcher, train API, loss keys/weights, checkpoint save/load.
- Kiểm tra loss gốc: custom loss bổ sung hay thay thế? Không thay `loss_bbox`, `loss_giou`, classification loss ngoài chủ đích được chứng minh.
- Kiểm tra shape/dtype/device, tọa độ box, labels/COCO mapping, no-object, empty targets, mixed precision, gradients và optimizer inclusion cho mọi head.
- Kiểm tra forward → loss → backward, finite gradients, hook registration không nhân đôi, checkpoint reload trong process mới và inference sau reload.
- So sánh kwargs baseline/improved; chỉ khác ở module được ablate cùng hyperparameters riêng đã định trước. Ghi rõ nếu API buộc phải có khác biệt.

### C. Config/ablation

- Mỗi toggle độc lập: `qar`, `dqcd`, `iqcm`, `hifrm`, `scu`, `lue`, `weighted_tversky`, `hard_negative_mining`; `rfo.enabled` điều khiển tìm kiếm.
- Chỉ bật mặc định các kỹ thuật có đường triển khai được đối chiếu trong code. Trong revision hiện tại, `qar`, `dqcd`, `iqcm`, `hifrm`, `scu`, `lue` bật; `weighted_tversky`, `hard_negative_mining`, `rfo.enabled` tắt và fail fast nếu bật vì thiếu phương trình/dữ liệu/thuật toán đã xác minh.
- Validate dependencies rõ ràng (ví dụ SCU/LUE); cấm tự tắt module khác khi phụ thuộc.
- RFO chỉ search LR, num_queries, weight_decay, batch_size theo yêu cầu. Ghi candidate, bounds, seed, fitness, epochs/trial, compute budget và config áp dụng thật. Candidate dùng validation; tuyệt đối không dùng test. RFO không chạy dài/tốn phí nếu chưa được cho phép.
- Ghi run ID, toggle vector, config snapshot/hash, dataset/split ID, seed, version, checkpoint path, best epoch và validation criterion.
- Lưu best checkpoint ở đường dẫn phân biệt baseline/improved/ablation; xác nhận tồn tại và không ghi đè run khác.

### D. Kiểm tra trước khi báo hoàn tất

Chỉ chạy các kiểm tra phù hợp nhiệm vụ; không tự khởi động training trả phí. Báo rõ `PASS`, `PARTIAL`, `BLOCKED` hoặc `FAIL` và không đánh dấu các kiểm tra chưa chạy là đạt.

- [ ] Config toggles có hiệu lực; kỹ thuật đã triển khai bật, phần chưa đủ căn cứ tắt và fail fast.
- [ ] Dataset split/ratio được ghi nhận thực tế và đồng nhất; class mapping 32 lớp khớp nguồn.
- [ ] Protocol cố định ở mọi route: 640, A100, patience 10, cap 80.
- [ ] Paper/code matrix có trích section/equation, phân loại exact/adaptation/unverified.
- [ ] Loss keys, weights, matching/targets và optimizer/head parameters hợp lệ.
- [ ] Test loss/edge cases và forward/backward nhỏ (nếu được yêu cầu/khả dụng).
- [ ] Hook/checkpoint save-load/inference (nếu được yêu cầu/khả dụng).
- [ ] Baseline không đổi ngoài phạm vi.
- [ ] Best checkpoint bàn giao; đánh giá ngoài bằng script riêng với đúng metric/threshold/CM.
- [ ] Không có kết luận accuracy tốt hơn nếu chưa có thí nghiệm so sánh cùng protocol.

## Phản hồi sau mỗi nhiệm vụ

Ngắn gọn bằng tiếng Việt:

1. Phát hiện và bằng chứng (file/section/symbol).
2. File đã thay đổi và lý do.
3. Kiểm tra đã chạy và kết quả; ghi `NOT RUN` cho phần chưa chạy.
4. Trạng thái `PASS` / `PARTIAL` / `BLOCKED` / `FAIL`.
5. Giả định/việc còn lại, nhất là split ratio, mapping lớp hoặc API chưa xác minh.

Không yêu cầu người dùng xác nhận lại các ràng buộc đã nêu trong context. Nếu split ratio hoặc class map thiếu, tiếp tục công việc độc lập; chỉ chặn run/claim phụ thuộc vào dữ liệu còn thiếu.
