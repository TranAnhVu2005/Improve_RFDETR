# AI_AGENT_CONTEXT.md — Bối cảnh dự án RF-DETR

> Mục đích: đọc file này trước khi đọc code để nắm nhanh trạng thái dự án, tránh quét lại toàn bộ repository và tránh suy diễn rằng các module đã được xác minh đúng theo paper.

## 1. Mục tiêu dự án

Dự án nghiên cứu cải tiến RF-DETR cho bài toán phát hiện đối tượng trên dataset COCO được tải từ Roboflow:

- Workspace/project theo README: `nckhcict2025/completed-project`, version `5`.
- Dataset được lưu trên Modal Volume tại `/vol/datasets/completed-project-5/coco`.
- Model mặc định của nhánh cải tiến: `RFDETRMedium`, pretrained.
- Hạ tầng dự kiến: Modal, GPU A10G.
- Thư viện RF-DETR được pin trong README ở phiên bản `rfdetr==1.11.2` vì code cải tiến móc vào API nội bộ.
- Baseline và improved là hai đường chạy riêng; giữ baseline nguyên trạng để so sánh có ý nghĩa.

Mục tiêu học thuật là kiểm tra và tích hợp các ý tưởng từ năm nhóm bài báo. Không được tuyên bố “reproduce chính xác paper” chỉ vì tên module hoặc ý tưởng tương tự đã xuất hiện trong code.

## 2. Cấu trúc và file quan trọng

Các file nhìn thấy trong working directory tại thời điểm tạo context:

- `train_improve_v1.py` — mã huấn luyện cải tiến; file trọng tâm khi audit.
- `train_baseline.py` — đường baseline; không sửa trừ khi nhiệm vụ yêu cầu rõ ràng.
- `rfdetr_improve_v1(1).yaml` — cấu hình cải tiến đang có trong working directory. Trong repo thực tế có thể được đặt/đổi tên thành `config/rfdetr_improve_v1.yaml`; xác minh đường dẫn thực trước khi sửa.
- `rfdetr_baseline(1).yaml` — cấu hình baseline.
- `modal_train.py` — entry point chạy Modal.
- `epoch_logger.py` — logging epoch.
- `README.md` — cách chạy, dataset, Modal, mô tả các adaptation hiện có.
- `donwload_dataset.py` — tên file hiện có bị viết là `donwload_dataset.py`; không tự đổi tên khi chưa kiểm tra import/reference.
- `2509.md`, `EL_34_10_13.md`, `fmars-13-1885615.md`, `jimaging-12-00401-v2.md`, `s41598-025-11725-0.md` — các bản Markdown của tài liệu/paper đã cung cấp. Các bản có hậu tố `(1)` có vẻ là bản trùng; xác minh nội dung/hash trước khi xử lý trùng lặp. Đọc metadata/title/abstract để lập bảng ánh xạ paper ↔ kỹ thuật, không ánh xạ chỉ dựa vào tên file.
- `image.png` — hình ảnh tài liệu/ghi chú; chỉ dùng nếu liên quan.

Lưu ý: đường dẫn thực tế có thể khác giữa bản local, Modal container và repo Git. Trước khi chạy lệnh hoặc sửa file, kiểm tra `pwd`, `find`, `git status` và các đường dẫn được import.

## 3. Kỹ thuật dự định tích hợp

Tên/nhóm kỹ thuật trong phạm vi dự án:

1. **QAR** — ý tưởng điều chỉnh/tái cân bằng query.
2. **DQCD** — ý tưởng contrastive/query discrimination từ Spectral-DETR.
3. **IQCM** — quality/IoU-aware query confidence/module.
4. **HIFRM** — focal-style refinement cho regression/GIoU.
5. **SCU + LUE** — uncertainty/calibration và weighting.
6. **Weighted Tversky Loss** — bổ sung tín hiệu cho phân loại/mất cân bằng.
7. **Hard Negative Mining (HNM)** — ưu tiên hard negatives.
8. **RFO** — tìm kiếm siêu tham số theo ý tưởng Red Fox Optimization.

Không được xem danh sách này là bằng chứng các kỹ thuật đã được triển khai chính xác. Cần xác minh công thức, tensor shape, gradient, điểm tích hợp, loss key và thí nghiệm đối chiếu với từng paper.

## 4. Trạng thái đã biết từ lần review tĩnh trước

Đây là các nhận xét từ đọc code, **chưa phải xác nhận runtime**:

- `train_improve_v1.py` có các hàm liên quan đến cấu hình cải tiến, custom losses, gắn hook vào model/builder và RFO.
- Các điểm cần ưu tiên audit:
  1. `_improvement_config()` có ánh xạ `loss_vfl` sang `loss_ce`. Phải xác minh key thực sự do criterion RF-DETR `1.11.2` trả về và key được `weight_dict` sử dụng. Không tự đổi tên loss dựa vào phỏng đoán.
  2. `_custom_losses()` có thể thay thế `loss_bbox` và `loss_giou` từ criterion gốc. Cần chứng minh loss gốc được giữ/được thay có chủ đích và không bị tính hai lần hoặc bị loại bỏ ngoài ý muốn.
  3. HIFRM đang có dạng focal-style dựa trên `(1 - giou).clamp_min(0).pow(gamma)` theo lần đọc trước. Cần đối chiếu chính xác công thức và định nghĩa GIoU của paper, bao gồm dấu, reduction, gamma và xử lý cặp matched boxes.
  4. IQCM dùng IoU với ground truth và chọn top-M query theo confidence theo lần đọc trước. Cần kiểm tra target cho query matched/unmatched có nhất quán với matcher của RF-DETR và với paper không.
  5. QAR có learnable query scale bị clamp trong `[0.5, 2.0]` theo lần đọc trước. Hook có thể scale input hoặc decoder states tùy signature. Phải xác minh hook tác động đúng tensor và gradient vẫn đi qua.
  6. DQCD hiện dùng proxy năng lượng tần số cao cho temperature/gating thay vì learned DAFD gate của paper Spectral-DETR. Đây là adaptation, không được mô tả là reproduction nguyên bản nếu chưa thay đổi và chứng minh.
  7. SCU/LUE dùng uncertainty head/loss tự thiết kế theo lần đọc trước. Cần so từng phương trình, target, detach/gradient, normalization và calibration với paper; nếu khác phải ghi rõ là adaptation.
  8. HNM là query-level, không mặc nhiên tương đương patch-level mining trong paper MIDOG.
  9. RFO mặc định được README mô tả là `population=3`, `generations=2`, `epochs_per_trial=3` — khoảng sáu candidate runs theo thiết kế hiện tại. Đây là tìm kiếm thực dụng rút gọn, không phải thí nghiệm đầy đủ của paper.
  10. Custom heads và monkey-patching có thể làm checkpoint khó load/inference bằng RF-DETR stock. Phải kiểm tra save → load trong tiến trình mới và inference sau reload.
  11. Baseline YAML có `lr_encoder`, nhưng `_training_kwargs()` của nhánh improved theo review trước có thể không truyền tham số tương đương. Phải kiểm tra lại và làm điều kiện so sánh baseline/improved nhất quán.
  12. YAML hiện tại có `model`, `training`, `evaluation`, `loss_weights`, `outputs`, nhưng không có phần `improvements` hoặc `rfo`; Python defaults có thể đang quyết định hành vi. Kiểm tra field nào thực sự được đọc. `evaluation`/`eval_dir` có thể chưa được sử dụng đầy đủ.
- README ghi cách chạy nhánh cải tiến:
  `modal run .\train_model\modal_train.py --mode improve`
- README mô tả dataset tự tải/được cache trên Modal Volume và A10G dùng physical batch tối đa 4 với gradient accumulation. Xác minh code thực tế trước khi coi các chi tiết đó là trạng thái runtime hiện tại.

## 5. Cấu hình cải tiến đã thấy

```yaml
model:
  name: "RF-DETR"
  variant: "RFDETRMedium"
  pretrained: true

training:
  epochs: 80
  patience: 10
  imgsz: 640
  batch_size: 16
  grad_accum_steps: 1
  workers: 4
  lr: 0.0001
  weight_decay: 0.0001
  device: 0
  seed: 42
  eval_interval: 1
  use_balanced_loss: true

evaluation:
  eval_conf: 0.01
  cm_conf: 0.25
  iou_step: 0.05
  benchmark_warmup: 20

loss_weights:
  loss_vfl: 1.0
  loss_bbox: 5.0
  loss_giou: 2.0

outputs:
  checkpoint_dir: "/vol/checkpoints/rfdetr_improve_v1"
  log_dir: "/vol/logs/rfdetr_improve_v1"
  eval_dir: "/vol/outputs/rfdetr_eval_improve_v1"
```

Đây là nội dung YAML được thấy trong working directory, không phải bằng chứng file này chính là file mà entry point Modal đang dùng. Phải kiểm tra `modal_train.py`, đối số `--config-path`, default config path và cách merge defaults.

## 6. Quy tắc kỹ thuật không được vi phạm

- Không tự tạo API của `rfdetr==1.11.2`. Xác minh signature và source code trong môi trường đúng phiên bản.
- Không dùng `except Exception: pass` hoặc fallback âm thầm để che lỗi tích hợp.
- Không nuốt mất loss gốc, không đổi tên loss tùy tiện, không để loss không có gradient.
- Không thay đổi baseline khi đang sửa nhánh improved, trừ khi nhiệm vụ yêu cầu.
- Không làm thay đổi dataset split, preprocessing, seed, số epoch, checkpoint selection hoặc metric giữa baseline và improved mà không ghi rõ.
- Không commit/API key/secret; không in `ROBOFLOW_API_KEY`.
- Không chạy job Modal/GPU có chi phí hoặc training dài nếu chưa được người dùng cho phép rõ ràng.
- Không tuyên bố “đã chạy thành công”, “đúng paper”, “tốt hơn baseline” nếu chưa có log/test/metric chứng minh.
- Ưu tiên thay đổi nhỏ, có test, dễ rollback. Tránh refactor toàn bộ file chỉ để làm đẹp.
- Ghi rõ mọi khác biệt giữa công thức paper và code triển khai.

## 7. Trạng thái xác minh

Tại thời điểm viết file này, các nhận xét nêu trên là **static-review findings**. Chưa có bằng chứng được lưu trong context này rằng đã hoàn thành:
- test import/runtime với đúng `rfdetr==1.11.2`;
- forward/backward trên batch thật;
- kiểm tra gradient từng loss/head;
- save/load checkpoint trong process mới;
- inference sau reload;
- training hoàn chỉnh trên Modal;
- so sánh mAP/FPS với baseline bằng cùng protocol;
- xác minh từng công thức với toàn văn năm paper.

AI agent phải phân biệt bốn mức: `CODE-OBSERVED`, `PAPER-VERIFIED`, `UNIT-TESTED`, `RUNTIME-VALIDATED`. Chỉ nâng trạng thái khi có bằng chứng cụ thể.
