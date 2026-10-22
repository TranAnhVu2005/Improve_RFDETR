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

## Trạng thái implementation hiện tại

Trong `config/rfdetr_improve_v1.yaml`, các toggle chạy được mặc định bật: `qar`, `dqcd`, `iqcm`, `hifrm`, `scu`, `lue`. SCU và LUE có cờ độc lập. DQCD dùng nhánh fixed-temperature `m=1` trong paper khi DAFD vắng mặt; DAFD chưa được tích hợp. IQCM `top-M=100`, IQCM loss weight `0.25`, và HIFRM `gamma=2` là lựa chọn cấu hình của đề tài do paper không ấn định số tương ứng.

Các cờ `weighted_tversky`, `hard_negative_mining`, `rfo.enabled` mặc định tắt. Bật lên sẽ fail fast: chưa có phương trình Weighted Tversky phù hợp query classification; HNM trong MIDOG đòi hỏi dữ liệu crop mô học; thuật toán/fitness RFO chưa được xác minh đủ. Không bật các mục này để chạy ablation giả.

Mọi ablation đổi đúng cờ cần nghiên cứu; giữ nguyên protocol, data split và các hyperparameter còn lại. Không tuyên bố cải thiện độ chính xác nếu chưa chạy cùng split và script đánh giá của nhóm. Class mapping và dataset phải qua kiểm tra trước khi train.

Improved config bật `resume: true`. Full Lightning checkpoint `last.ckpt` được lưu mỗi epoch và commit ổn định lên Modal Volume; khi chạy lại, ưu tiên `last.ckpt`, fallback checkpoint lưu theo epoch. Resume khôi phục optimizer/scheduler nếu checkpoint đầy đủ còn tồn tại. Có thể mất epoch đang chạy khi bị kill; thí nghiệm mới cần output directory riêng hoặc `resume: false`.

Schema đang dùng:

```yaml
improvements:
  qar: true
  dqcd: true
  iqcm: true
  hifrm: true
  scu: true
  lue: true
  weighted_tversky: false
  hard_negative_mining: false
rfo:
  enabled: false
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
