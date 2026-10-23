# BẢNG TỔNG HỢP SO SÁNH CÁC MÔ HÌNH OBJECT DETECTION (COCO BENCHMARK)

- **Đề tài**: Nghiên cứu khoa học - Nhận diện nguyên liệu nấu ăn (32 lớp)
- **Thời gian**: 22/10/2026 12:15:46
- **Chuẩn đánh giá**: pycocotools COCOeval, 1481 ảnh, protocol `266a48399627faf496f892c8a6192811669cdb9ed8b3c135511bce61ab710ddd`
- **P/R/F1**: suy ra từ COCOeval matches tại conf=0.55, IoU=0.5; các cột mAP/AR là COCO metrics.

| Mô hình | Precision | Recall | F1 | mAP@50 | mAP@50-95 | mAP@75 | AR@100 | Patience | GFLOPs | Parameters | Latency | FPS | Trạng thái |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF-DETR Medium | 0.9768 | 0.9683 | 0.9725 | 0.9873 | 0.7817 | 0.9305 | 0.8327 | 10 | 39.18 | 33.69 | 31.91 | 31.34 | Hoàn tất đánh giá |

*Ghi chú: Không đồng nhất Precision/Recall tại một ngưỡng cố định với COCO AP/AR tích phân.*
