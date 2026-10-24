# Common_Evaluate: hướng dẫn vận hành

Tài liệu này là điểm bắt đầu cho mọi lần chạy hoặc chỉnh sửa thư mục đánh giá. Đọc file này trước; chỉ mở `common_evaluate.py` khi cần sửa hành vi evaluator hoặc tra lỗi cụ thể.

## Mục đích và giới hạn

`common_evaluate.py` chạy inference bằng adapter tương ứng với framework, đổi prediction sang COCO, rồi dùng `pycocotools.COCOeval` để tính metric. Script không train model và không lấy metric từ log train. Checkpoint được chọn phải tồn tại thật; thiếu checkpoint sẽ dừng, không fallback sang pretrained hoặc random weights.

Protocol đang khóa trong `models_config.json`: 32 lớp đã cấu hình, ảnh 640, batch 1, ngưỡng prediction 0.0, operating point confidence 0.55 / IoU 0.50, tối đa 100 detections và kỳ vọng 1.481 ảnh test. Script kiểm tra annotation/dataset lúc chạy; nếu snapshot hiện tại không khớp thì phải dừng và sửa cấu hình theo dataset đã được xác nhận, không sửa số để lách kiểm tra. Không tuyên bố kết quả trước khi có artifact sinh ra từ lần chạy thật.

Lưu ý: code `Common_Evaluate` hiện xuất metric COCO và per-class CSV, nhưng không có bước xuất confusion matrix. Nếu cần ma trận nhầm lẫn theo protocol nghiên cứu (confidence 0.01 / IoU sweep 0.05), hãy dùng script đánh giá riêng của nhóm; không coi báo cáo Common_Evaluate là đã có confusion matrix.

## Checkpoint RF-DETR QAR-only đang được cấu hình

Ảnh Modal do người dùng cung cấp cho thấy Volume `rfdetr-storage`, thư mục `checkpoints/rfdetr_improve_v1_qar_only`, file `best.pth`. Cấu hình trỏ tới `/vol/checkpoints/rfdetr_improve_v1_qar_only/best.pth`; `common_evaluate.py` mount volume đó tại `/vol`. Tên và đường dẫn này dựa theo ảnh; evaluator sẽ tự xác minh file tồn tại trên Modal trước inference.

Đây là checkpoint train chỉ bật QAR. Cấu hình riêng `models_config_qar_only.json` bật `use_qar` và đặt scale bounds `0.5`–`2.0` theo cấu hình train; cấu hình benchmark dùng chung `models_config.json` vẫn giữ nguyên checkpoint trước đó. Adapter lấy các logits QAR đã học từ checkpoint rồi áp dụng scale lên `query_feat` trước transformer decoder; hỗ trợ cả query cơ sở và query nhân theo `group_detr`, theo logic của code train. Nếu checkpoint thiếu tensor QAR hoặc số query không khớp, evaluator dừng với lỗi rõ ràng thay vì âm thầm chấm như model không QAR. Cờ `use_rfdetr_123` tắt trong cấu hình QAR-only vì nhánh đó là thay đổi kiến trúc/ranking riêng, không có trong lần train này.

RF-DETR nạp checkpoint trước khi adapter đăng ký tham số QAR riêng, nên log thư viện có thể báo `transformer.improvement_query_scale_logits` là khóa chưa được tiêu thụ. Với profile này, adapter đọc tensor đó từ checkpoint, đăng ký vào transformer và in `[+] Restored QAR ...`; chỉ tiếp tục khi bước khôi phục này thành công.

## Profile đánh giá DQCD-only

Ảnh Modal mới cho thấy checkpoint `/vol/checkpoints/rfdetr_improve_v1_dqcd_only/best.pth` trên Volume `rfdetr-storage`. Cấu hình riêng `models_config_dqcd_only.json` trỏ tới checkpoint đó, tắt `use_qar` và `use_rfdetr_123`. DQCD trong code train chỉ thêm loss tương phản khi huấn luyện; không có nhánh hiệu chỉnh inference riêng cần gắn vào evaluator.

Dataset được cấu hình ở `/vol/datasets/completed-project-5/coco`, cùng vị trí volume với entrypoint train. Kết quả cloud được ghi dưới `/vol/runs/Common_COCO_Evaluation_Results/<run-id>` và tải về `Common_Evaluate/Results/` ở máy chạy lệnh.
Evaluator pin `rfdetr==1.11.2`, trùng phiên bản được pin trong Modal image của trainer improved. RF-DETR 1.11.2 cần Transformers 5.1 trở lên; evaluator ghim `torch==2.6.0` / `torchvision==0.21.0` để Transformers không tắt PyTorch vì phiên bản quá cũ. Cặp phiên bản Torch/Torchvision khớp bảng tương thích chính thức của PyTorch.

### Nếu gặp lỗi `BackboneConfigMixin`

Log ngày 22/10/2026 đã dừng ở `import rfdetr` với `ImportError: cannot import name 'BackboneConfigMixin'`. Dataset (32 lớp, 1.481 ảnh) và checkpoint đã được tìm thấy; inference chưa bắt đầu và chưa có metric. Nguyên nhân là image evaluator khi đó cài Torch 2.5.1, trong khi Transformers trong container báo cần Torch ít nhất 2.6 và tự vô hiệu hóa PyTorch. Dependency đã được sửa lên Torch 2.6.0 / Torchvision 0.21.0. Chạy lại lệnh bên dưới để Modal build image mới; nếu vẫn thấy log phiên bản cũ, kiểm tra lỗi build/cache thay vì diễn giải đó là lỗi checkpoint.

## Cách chạy RF-DETR đã train

Tại thư mục gốc repo, dùng PowerShell:

```powershell
modal run .\Common_Evaluate\common_evaluate.py::evaluate_single --model-name RFDETR_Medium --config-path .\Common_Evaluate\models_config_qar_only.json
```

Lệnh chỉ đánh giá `RFDETR_Medium` với checkpoint QAR-only; không chạy các model khác đang tắt/bật trong file config. Các lệnh thường dùng vẫn đọc `models_config.json`, còn `--config-path` chọn profile riêng cho lần QAR này. Cần đăng nhập Modal trên máy chạy lệnh và truy cập được volume `rfdetr-storage`. Lần đầu Modal có thể build image và tải dependencies. A100 được cấp phát cho evaluator và có thể phát sinh phí. Lệnh chạy ở foreground; chờ đến khi hoàn tất để kết quả được tải về `Common_Evaluate/Results/`.

Đánh giá checkpoint DQCD-only bằng profile riêng:

```powershell
modal run .\Common_Evaluate\common_evaluate.py::evaluate_single --model-name RFDETR_Medium --config-path .\Common_Evaluate\models_config_dqcd_only.json
```

Nếu dùng Roboflow để tải dataset khi đường dẫn cloud chưa có dữ liệu, cung cấp `ROBOFLOW_API_KEY` qua môi trường hoặc file `Common_Evaluate/.env` (tạo từ `.env.example`). Không ghi API key vào JSON hay commit `.env`.

Các lệnh khác:

```powershell
# Chạy tất cả model được đánh dấu enabled=true
modal run .\Common_Evaluate\common_evaluate.py::evaluate_my_models

# Trên máy local, chỉ dùng khi đã có dataset và checkpoint local tương ứng
python .\Common_Evaluate\common_evaluate.py --data-dir <COCO_DATASET_DIR> --models RFDETR_Medium --device cuda:0

# Gộp các result_<model>.json đã được tạo thật
python .\Common_Evaluate\common_evaluate.py --merge-dir <THU_MUC_KET_QUA> --output-dir <THU_MUC_BAO_CAO>
```

## Đầu ra và kiểm tra

Mỗi model chạy tạo `result_<model>.json`, CSV metric từng lớp, prediction COCO nén và báo cáo tổng hợp. Giữ nguyên các artifact này để truy vết `protocol_id`, hash dataset/checkpoint và inference config. Khi merge, model chưa có kết quả hiển thị trạng thái chờ và dấu `-`; không được xem là metric bằng 0.

Trước khi báo cáo, xác nhận process hoàn tất, `result_RFDETR_Medium.json` có provenance/hash, `protocol` đúng, và các output được sinh trong đúng run. Không tự điền/ước tính metric, không dùng test để chọn checkpoint/hyperparameter. README không khẳng định lần đánh giá đã chạy.

## Cấu hình liên quan

- `models_config.json`: cấu hình benchmark dùng chung.
- `models_config_qar_only.json`: profile riêng cho checkpoint QAR-only; chỉ chọn bằng `--config-path` khi cần đánh giá checkpoint này.
- `models_config_dqcd_only.json`: profile riêng cho checkpoint DQCD-only; chỉ chọn bằng `--config-path` khi cần đánh giá checkpoint này.
- `training_config.json`: snapshot cấu hình train được lưu riêng; evaluator không đọc file này.
- `.env.example`: tên biến bí mật cần thiết. File `.env` thật không được commit.

Giữ thay đổi evaluator nhỏ và theo yêu cầu. Trước khi chỉnh `common_evaluate.py`, tìm hàm liên quan bằng `rg` (ví dụ `resolve_checkpoint_path`, `RFDETRAdapter`, `merge_team_results`, `evaluate_models_cloud`) thay vì đọc/viết lại toàn file.
