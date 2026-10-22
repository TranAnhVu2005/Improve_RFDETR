# RF-DETR cho nhận diện nguyên liệu nấu ăn

## Mục tiêu

So sánh baseline RF-DETR với nhánh cải tiến có thể bật/tắt từng kỹ thuật bằng config. Detector nhận diện nguyên liệu trong ảnh; kết quả detection là đầu vào cho bước ánh xạ nguyên liệu sang công thức và món ăn đề xuất. Tài liệu này hướng dẫn huấn luyện/checkpoint, không thay thế pipeline gợi ý món ăn.

## Protocol cố định cho mọi run

- Cùng dataset snapshot và cùng train/val/test split. Ghi số ảnh, tỉ lệ thực tế và cách chia/seed; tỷ lệ hiện chưa được ghi nhận, không tự suy đoán.
- Ảnh 640×640; A100; patience 10; tối đa 80 epochs.
- 32 lớp theo mapping gốc; index báo cáo 1 là `beef`, index 32 là `tomato`. Không tự điền lớp còn thiếu; xác minh annotation category IDs và thứ tự trước train.
- Dùng cùng baseline protocol, seed, checkpoint selection và training parameters cho các run; chỉ toggle kỹ thuật được phép khác.
- Bàn giao best checkpoint. Nhóm chạy script đánh giá riêng sau đó: confidence 0.01, mAP@0.5:0.95 với bước IoU 0.05, confusion matrix. Không báo metric nếu chưa có output script.

Các ràng buộc chi tiết và xử lý khi thiếu thông tin: [AI_AGENT_CONTEXT.md](AI_AGENT_CONTEXT.md). Quy trình agent: [AI_AGENT_TASK_INSTRUCTIONS.md](AI_AGENT_TASK_INSTRUCTIONS.md).

## Cấu hình và ablation

`config/rfdetr_baseline.yaml` là baseline; `config/rfdetr_improve_v1.yaml` là cấu hình improved. Các kỹ thuật đã tích hợp và bật mặc định:

- `qar`
- `dqcd`
- `iqcm`
- `hifrm`
- `scu`
- `lue`
- `weighted_tversky`, `hard_negative_mining` và `rfo.enabled` mặc định `false`. Chúng chưa đủ căn cứ để triển khai trung thực cho bài toán/dataset này; nếu bật, code dừng với thông báo thay vì giả lập kỹ thuật.

Mỗi ablation tắt đúng cờ cần kiểm tra, giữ các cờ khác và protocol cố định. DQCD dùng nhánh nhiệt độ cố định của paper khi không có DAFD; vì vậy đây là adaptation và không gồm nhánh lọc DAFD. HIFRM gamma và IQCM top-M/weight là giá trị cấu hình đề tài vì bài báo không quy định số cụ thể. Paper được nghiên cứu trên miền khác nên chưa có bằng chứng tăng độ chính xác trên nguyên liệu nấu ăn.

### Kỹ thuật được triển khai

- **QAR**: hệ số query học được, giới hạn [0.5, 2], khởi tạo 1.
- **DQCD**: InfoNCE trên query với positive liên tầng/cùng lớp và hard-negative query theo ambiguity; nhiệt độ fallback `τ=0.15`, không dùng FFT ảnh thay cho DAFD. `hard_negative_mining` là kỹ thuật mining crop MIDOG riêng, hiện tắt.
- **IQCM**: nhánh dự đoán IoU quality, BCE và hiệu chỉnh score inference.
- **HIFRM**: focal loss trên `1−GIoU` ở decoder cuối và các decoder phụ.
- **SCU + LUE**: uncertainty head, hiệu chuẩn uncertainty và precision-weighted L1 theo kích thước hộp; có cờ riêng.
- **Bỏ qua**: Weighted Tversky không có công thức phù hợp query classification trong PDF; MIDOG hard-negative mining cần dữ liệu crop mô học không có trong bộ nguyên liệu; RFO chưa xác minh đủ thuật toán/fitness.

## Tài liệu tham khảo

`BaiBaoDangMarkDown/README_PAPERS.md` là bản đồ ngắn tới năm bài nghiên cứu và giới hạn chuyển giao. Các file paper gốc trong thư mục đó được giữ nguyên để tra section/equation. Tên kỹ thuật không phải bằng chứng implementation trùng paper; ghi rõ `Exact`, `Adaptation`, hoặc `Unverified`.

## Chạy Modal

Entrypoint được thấy trong repo:

```powershell
modal run .\train_model\modal_train.py --mode baseline
modal run .\train_model\modal_train.py --mode improve
```

Modal hiện khai báo A100 cho cả hai mode. Kiểm tra config/path thực tế trước khi chạy. Các lệnh trên khởi chạy job GPU có thể tính phí. Không chạy job dài trong công việc audit/tài liệu.

Config mặc định: `config/rfdetr_baseline.yaml`, `config/rfdetr_improve_v1.yaml`. Trước khi tính phí, xác minh Modal account/Volume/secret, dataset split và mapping đủ 32 class theo thứ tự bắt đầu `beef`, kết thúc `tomato`. Có thể truyền `--dataset-dir` và `--config-path`; kiểm tra `modal_train.py` để biết cú pháp hiện hành.

Improved dùng `resume: true`: RF-DETR ghi full `last.ckpt` mỗi epoch; trainer commit checkpoint ổn định lên Modal Volume và lần chạy lại ưu tiên khôi phục `last.ckpt` (có optimizer/scheduler state), rồi mới tìm checkpoint lưu theo epoch. Có thể mất phần epoch chưa commit nếu container bị kill đột ngột. Sau khi run hoàn tất, không chạy lại cùng output path nếu muốn bắt đầu một thí nghiệm mới; dùng output path riêng hoặc tắt resume.

## Checkpoint và đánh giá

Trước khi bàn giao, ghi đường dẫn best checkpoint, run/config snapshot, split ID, best epoch và tiêu chí chọn checkpoint. Không gộp/ghi đè output baseline, improved và từng ablation. Đánh giá mAP/CM thuộc script riêng của nhóm; tham số phải giữ confidence 0.01, IoU step 0.05 và confusion matrix.

## Trạng thái và giới hạn

Đường dẫn, class mapping đầy đủ và split ratio cần được xác nhận từ dataset thực tế. Chỉ cập nhật kết quả benchmark sau khi có log/artifact có thể truy vết. Xem `git status` trước khi chỉnh sửa; giữ nguyên thay đổi người dùng đang có.
