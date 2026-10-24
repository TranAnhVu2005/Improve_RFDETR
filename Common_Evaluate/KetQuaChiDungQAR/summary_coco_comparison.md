# BẢNG TỔNG HỢP SO SÁNH CÁC MÔ HÌNH OBJECT DETECTION (COCO BENCHMARK)

- **Đề tài**: Nghiên cứu khoa học - Nhận diện nguyên liệu nấu ăn (32 lớp)
- **Thời gian**: 23/10/2026 13:18:07
- **Chuẩn đánh giá**: pycocotools COCOeval, 1481 ảnh, protocol `820af4eb847fa99688dc54e230ef9bf12f1525a6a361999e261da163a3cece71`
- **P/R/F1**: suy ra từ COCOeval matches tại conf=0.55, IoU=0.5; các cột mAP/AR là COCO metrics.

| Mô hình | Precision | Recall | F1 | mAP@50 | mAP@50-95 | mAP@75 | AR@100 | Patience | GFLOPs | Parameters | Latency | FPS | Trạng thái |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF-DETR Medium | 0.9753 | 0.9706 | 0.9730 | 0.9861 | 0.8764 | 0.9533 | 0.9165 | 10 | 39.09 | 33.60 | 29.98 | 33.35 | Hoàn tất đánh giá |

*Ghi chú: Không đồng nhất Precision/Recall tại một ngưỡng cố định với COCO AP/AR tích phân.*
