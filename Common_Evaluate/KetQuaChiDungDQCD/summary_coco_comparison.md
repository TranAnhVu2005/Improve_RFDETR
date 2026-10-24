# BẢNG TỔNG HỢP SO SÁNH CÁC MÔ HÌNH OBJECT DETECTION (COCO BENCHMARK)

- **Đề tài**: Nghiên cứu khoa học - Nhận diện nguyên liệu nấu ăn (32 lớp)
- **Thời gian**: 24/10/2026 07:56:54
- **Chuẩn đánh giá**: pycocotools COCOeval, 1481 ảnh, protocol `820af4eb847fa99688dc54e230ef9bf12f1525a6a361999e261da163a3cece71`
- **P/R/F1**: suy ra từ COCOeval matches tại conf=0.55, IoU=0.5; các cột mAP/AR là COCO metrics.

| Mô hình | Precision | Recall | F1 | mAP@50 | mAP@50-95 | mAP@75 | AR@100 | Patience | GFLOPs | Parameters | Latency | FPS | Trạng thái |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF-DETR Medium | 0.9812 | 0.9719 | 0.9765 | 0.9873 | 0.8784 | 0.9557 | 0.9110 | 10 | 39.09 | 33.60 | 28.06 | 35.64 | Hoàn tất đánh giá |

*Ghi chú: Không đồng nhất Precision/Recall tại một ngưỡng cố định với COCO AP/AR tích phân.*
