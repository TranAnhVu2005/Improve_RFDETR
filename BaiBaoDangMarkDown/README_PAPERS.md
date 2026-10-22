# Bản đồ paper cho các kỹ thuật RF-DETR

> Tài liệu định tuyến nhanh; không thay cho toàn văn. Khi triển khai, tìm equation/algorithm và section chính xác trong file nguồn. Tất cả nguồn dưới đây thuộc domain khác food detection; mọi thay đổi công thức hoặc điểm tích hợp phải ghi `Adaptation`.

| File | Chủ đề liên quan | Cách dùng trong dự án | Cảnh báo khi chuyển giao |
|---|---|---|---|
| [`fmars-13-1885615.md`](fmars-13-1885615.md) | RF-DETR underwater plastic; Query Adaptive Reweighting (QAR), frequency-aware feature weighting | QAR; tham khảo ablation và thực nghiệm query | Underwater domain, không chứng minh hiệu quả trên food; kiểm tra section 3.4 và công thức/chiều query |
| [`jimaging-12-00401-v2.md`](jimaging-12-00401-v2.md) | Spectral-DETR; DQCD, SCU+LUE, DAFD | DQCD (section 3.3), uncertainty/precision weighted localization SCU+LUE (3.4), objective (3.5) | DQCD phụ thuộc DAFD gate trong paper; bỏ gate/thay bằng proxy là adaptation. Không mặc định phải mang cả DAFD vào danh sách kỹ thuật hiện tại |
| [`EL_34_10_13.md`](EL_34_10_13.md) | DARQ-Det; HIFRM và IoU Quality Calibration Module (IQCM) | HIFRM section III-C; IQCM section III-D | Hazy remote sensing domain. Xác minh equation, matched/unmatched targets, score calibration và loss trước tích hợp |
| [`2509.md`](2509.md) | MIDOG 2025 RF-DETR; hard negative mining | Tham khảo hard negative mining và training observations | Histopathology; bài mô tả mining/dataset workflow, không mặc nhiên định nghĩa query-level mining cho food detection |
| [`s41598-025-11725-0.md`](s41598-025-11725-0.md) | RF-DETR + Red Fox Optimization (RFO) cho urine microscopy | RFO hyperparameter search | Bài dùng dataset/domain và protocol khác. Đối chiếu algorithm/bounds/fitness; không sao chép mAP/kết quả như kỳ vọng của food dataset |

## Weighted Tversky

Trong năm file được liệt kê, chưa xác định được nguồn primary cụ thể cho **Weighted Tversky Loss** áp dụng cho DETR object queries. Không suy ra công thức classification detection từ segmentation Tversky. Ghi `Unverified` cho tới khi có nguồn/equation phù hợp và cách ánh xạ query labels/no-object được định nghĩa.

## Tra cứu/ghi nhận

Với mỗi module ghi paper section/equation, ký hiệu và reduction, code location, target/tensor, khác biệt, test và trạng thái `Exact` / `Adaptation` / `Incorrect` / `Unverified`. DOI/title trong file gốc cần được xác nhận nếu trích dẫn học thuật; bản Markdown có thể chứa OCR lỗi.
