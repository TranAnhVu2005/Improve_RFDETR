# AI Agent Context

> Đọc file này trước. Đây là context ngắn, ưu tiên sự thật đã xác nhận và các quyết định cố định. Chỉ mở tài liệu paper hoặc code liên quan đúng nhiệm vụ.

## Mục tiêu

Nghiên cứu cải tiến RF-DETR cho phát hiện nguyên liệu nấu ăn; đầu ra detector phục vụ bước ánh xạ nguyên liệu → công thức/món ăn. Scope hiện tại là thiết kế và tích hợp nhánh improved theo dạng ablation, giữ baseline độc lập để so sánh. Mỗi kỹ thuật phải có cờ boolean trong config; bật/tắt không cần sửa code.

## Ràng buộc thí nghiệm cố định

- Dataset: cùng một snapshot và cùng train/val/test split cho baseline, improved và mọi ablation. Ghi lại chính xác nguồn, số ảnh từng split, tỉ lệ thực tế, seed/cách chia và hash/định danh annotation. **Tỉ lệ hiện chưa được xác nhận trong tài liệu; không tự đặt.** Nếu dataset không có test split thì phải nêu rõ và thống nhất cách tạo trước khi chạy.
- Input: 640×640.
- Early stopping patience: 10; tối đa 80 epochs.
- GPU: NVIDIA A100 cho mọi run so sánh.
- Class order: cố định 32 lớp; class index 1-based bắt đầu `beef`, index 32 là `tomato`. Lấy toàn bộ tên/thứ tự lớp từ annotation/dataset gốc, xác minh mapping COCO category ID (thường 0-based) sang chỉ số báo cáo 1-based; không tự suy diễn 30 tên còn lại.
- Đánh giá sau checkpoint: dùng script đánh giá riêng của nhóm. Ngưỡng confidence 0.01; mAP@0.5:0.95 quét IoU threshold với bước 0.05; xuất confusion matrix. Không thay metric, threshold hay quy trình chọn best checkpoint giữa các run.
- Best checkpoint là artifact bàn giao cho script đánh giá. Không tuyên bố metric/kết quả nếu chưa chạy script trên checkpoint.
- Dataset split, thứ tự class, kích thước ảnh, patience, GPU và epoch tối đa không phải ablation knobs.

## Kỹ thuật cần có toggle

Tất cả bật mặc định trong config improved; mỗi nhóm là cờ độc lập, có thể tắt riêng:

`qar`, `dqcd`, `iqcm`, `hifrm`, `scu`, `lue`, `weighted_tversky`, `hard_negative_mining`, `rfo`.

IQCM và HIFRM là hai toggle riêng. SCU và LUE là hai toggle riêng để có thể ablate; nếu một phần phụ thuộc phần kia, validate config và báo lỗi/định nghĩa hành vi rõ ràng, không âm thầm bỏ qua. RFO chỉ điều khiển tìm kiếm siêu tham số (learning rate, số queries, weight decay, batch size) và có thể tắt độc lập; không được thay đổi các ràng buộc cố định phía trên. Mọi knob/weight/giới hạn thuật toán phải nằm trong config, không giấu ở Python defaults.

Ví dụ schema dự kiến (đối chiếu với implementation trước khi sửa code):

```yaml
improvements:
  qar: true
  dqcd: true
  iqcm: true
  hifrm: true
  scu: true
  lue: true
  weighted_tversky: true
  hard_negative_mining: true
rfo:
  enabled: true
  search: [lr, num_queries, weight_decay, batch_size]
```

## Repo hiện được thấy

- Entry point Modal: `train_model/modal_train.py`; mode `baseline` / `improve`; code chọn `gpu="A100"` cho cả hai.
- Config: `config/rfdetr_baseline.yaml`, `config/rfdetr_improve_v1.yaml`.
- Trainers: `train_model/train_baseline.py`, `train_model/train_improve_v1.py`.
- Paper Markdown: `BaiBaoDangMarkDown/`; bản gốc là nguồn tham khảo, giữ nguyên trừ khi được yêu cầu sửa.
- Các file trên được xác nhận tồn tại trong workspace khi context này được viết. Luôn kiểm tra `git status` và đọc file trước khi thay đổi.

## Tài liệu để tra cứu

- `AI_AGENT_TASK_INSTRUCTIONS.md`: workflow, tiêu chuẩn audit/tích hợp và báo cáo.
- `BaiBaoDangMarkDown/README_PAPERS.md`: bản đồ paper → ý tưởng → giới hạn áp dụng.
- `README.md`: hướng dẫn chạy và giao thức dự án.
- `BaiBaoDangMarkDown/*.md`: toàn văn đã cung cấp; chỉ đọc section có liên quan, không nạp toàn bộ nếu không cần.

## Bằng chứng và trạng thái

Dùng nhãn riêng: `CODE-OBSERVED`, `PAPER-VERIFIED`, `UNIT-TESTED`, `RUNTIME-VALIDATED`, `EXPERIMENT-VALIDATED`. Không chuyển trạng thái nếu chưa có artifact/bằng chứng tương ứng. Paper trên domain khác chỉ là nguồn ý tưởng; áp dụng vào món ăn/dataset này phải ghi `ADAPTATION` khi khác công thức, dữ liệu, tầng tích hợp hay protocol. Model chính, checkpoint, matcher, criterion, loss key và loader phải được xác minh trên version runtime thật trước khi patch.
