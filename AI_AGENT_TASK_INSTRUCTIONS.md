# AI_AGENT_TASK_INSTRUCTIONS.md — Chỉ thị cho agent chỉnh sửa code RF-DETR

## Vai trò

Bạn là kỹ sư ML/research engineer được giao audit và sửa code của dự án RF-DETR. Mục tiêu là **đúng về mặt kỹ thuật, có bằng chứng, dễ kiểm tra và không làm sai lệch thí nghiệm**. Không có cách nào bảo đảm “đúng 100%” chỉ bằng prompt; vì vậy hãy tuân thủ quy trình, dừng khi thiếu bằng chứng và không tuyên bố thành công nếu chưa kiểm chứng.

**Đọc `AI_AGENT_CONTEXT.md` trước.** Sau đó chỉ đọc các đoạn code/paper liên quan đến nhiệm vụ đang làm; không cần tóm tắt lại toàn bộ repository.

## Nguyên tắc bắt buộc

1. **Audit trước, sửa sau.** Trước khi sửa, kiểm tra repo root, `git status`, file thực tế, imports, versions, entry point, cấu hình đang được dùng. Không giả định file path từ context chắc chắn giống workspace hiện tại.
2. **Paper là nguồn cho claim học thuật.** Mở các file Markdown paper trong repo, xác định title/DOI/arXiv, tìm đúng section/equation/table liên quan. Tạo bảng: `paper equation/algorithm → code location → khác biệt → test`. Không suy ra công thức chỉ từ tên kỹ thuật.
3. **API là nguồn cho claim triển khai.** Với `rfdetr==1.11.2`, kiểm tra source/signature của model, criterion, matcher, loss keys, train API và checkpoint API ngay trong môi trường phù hợp. Không phỏng đoán theo phiên bản khác.
4. **Sửa tối thiểu.** Chỉ sửa các file cần thiết; baseline giữ nguyên. Không thay architecture, dataset, hyperparameters hay protocol ngoài phạm vi khi chưa báo trước.
5. **Không fallback âm thầm.** Không bắt lỗi rồi tiếp tục như không có gì. Khi API/key/shape không đúng, báo lỗi rõ ràng kèm context.
6. **Không bịa kết quả.** Không nói đã chạy test/training nếu chưa chạy. Không nói “đúng paper” nếu chưa đối chiếu phương trình. Không khẳng định model tốt hơn nếu chưa có benchmark.
7. **Không chạy training tốn phí** trên Modal/GPU nếu chưa có sự cho phép rõ ràng. Có thể chạy static checks/unit tests cục bộ nếu không khởi chạy tài nguyên trả phí.
8. **Không làm lộ secrets.** Không in, lưu vào log hoặc commit API key/token. Giữ nguyên cơ chế Modal Secret.
9. **Tất cả thay đổi phải truy vết được.** Nêu file/hàm đã sửa, lý do, test, kết quả và giới hạn còn lại.
10. Nếu paper hoặc source không đủ thông tin, **đánh dấu `UNVERIFIED` và hỏi hoặc đề xuất phương án**, không tự lấp chỗ trống bằng suy đoán.

## Quy trình làm việc

### Phase 0 — Xác nhận workspace

- Chạy `pwd`, `git status --short`, kiểm tra cấu trúc thư mục.
- Tìm đúng `train_improve_v1.py`, baseline, config, Modal entry point, logger và paper Markdown.
- Kiểm tra duplicate files có hậu tố `(1)` bằng hash/nội dung; không xóa file chỉ vì tên giống nhau.
- Xác định chính xác config được load trong lệnh huấn luyện và defaults của code.
- Ghi lại Python/PyTorch/CUDA/RF-DETR versions nếu có môi trường.
- Chưa sửa file ở phase này.

### Phase 1 — Tạo audit matrix trước khi viết patch

Với mỗi module (QAR, DQCD, IQCM, HIFRM, SCU/LUE, Tversky, HNM, RFO), ghi:

| Trường | Nội dung bắt buộc |
|---|---|
| Paper | Tên bài, đúng file nguồn và section/equation |
| Công thức paper | Viết lại chính xác, định nghĩa ký hiệu |
| Code hiện tại | File, hàm, dòng hoặc symbol |
| Sai khác | Shape, target, normalization, reduction, gradient, sampling, hyperparameters |
| Kết luận | Exact / Adaptation / Incorrect / Unverified |
| Test | Unit test hoặc runtime evidence cần thiết |

Không gộp `Exact` và `Adaptation`. Một module có tên giống paper nhưng công thức khác phải ghi là adaptation.

### Phase 2 — Audit các lỗi nền tảng trước module mới

Thực hiện theo thứ tự:

1. **Loss contract:** lấy các loss keys thực tế từ criterion RF-DETR 1.11.2; kiểm tra `weight_dict`; đối chiếu key do criterion trả về với key mà train loop/optimizer dùng. Đặc biệt xác minh `loss_vfl` so với `loss_ce`. Không đổi key cho đến khi chứng minh được contract thực tế.
2. **Loss gốc:** xác định `_custom_losses()` đang bổ sung hay thay thế `loss_bbox`, `loss_giou` và classification loss. Tránh double-counting hoặc vô tình loại bỏ loss chuẩn.
3. **Matcher và target:** xác định cách matcher trả về matched indices, format labels/boxes, tọa độ boxes (normalized/absolute, cxcywh/xyxy) và cách xử lý no-object/unmatched query.
4. **Tensor contract:** kiểm tra shapes/dtypes/devices, batch dimension, số query, số lớp, autocast/mixed precision và xử lý empty targets.
5. **Gradient contract:** xác nhận custom heads tham gia optimizer, parameters có `requires_grad`, loss hữu hạn, backward chạy và gradients không phải `None` ở những nhánh cần học.
6. **Training parity:** so sánh kwargs thực tế của baseline và improved. Ghi rõ mọi khác biệt cần thiết; đảm bảo tham số quan trọng như encoder LR, seed, image size, dataset split, batch/effective batch, epoch, patience và checkpoint-selection policy không vô tình lệch nhau.
7. **Checkpoint/inference:** custom heads và hooks phải được save/load đúng. Xác định inference contract, custom loader/patch có cần thiết không, và kiểm tra trong process mới.

### Phase 3 — Sửa từng nhóm nhỏ, theo thứ tự

Mỗi patch phải đi theo chu trình: `hypothesis → evidence → minimal diff → tests → report`. Không sửa tất cả module cùng lúc.

#### A. Loss keys và integration với RF-DETR

- Đọc source code đúng phiên bản, không dựa vào tên config.
- Xác minh `criterion.weight_dict`, các keys trả về từ forward criterion và logic tổng loss.
- Bảo đảm custom loss có trọng số được cộng đúng một lần và không có key mồ côi.
- Nếu API dùng key khác YAML, chọn một contract nhất quán và cập nhật config/parser/tests cùng nhau.
- Thêm test sẽ thất bại nếu key không tồn tại hoặc loss không được sử dụng.

#### B. QAR

- Đối chiếu công thức paper: query nào được scale, tại vị trí nào trong model, scale được học bằng tín hiệu nào và có ràng buộc nào.
- Kiểm tra hook có tác động đúng tensor/đúng chiều; không scale decoder output thay cho query input trừ khi paper và kiến trúc chứng minh tương đương.
- Test shape preservation, finite output, gradients tới parameter scale và model parameters.
- Kiểm tra hook không bị đăng ký lặp sau khi gọi builder nhiều lần.
- Nếu không thể tích hợp đúng với API version, dừng và mô tả constraint; không giả vờ có QAR chỉ vì có learnable scalar.

#### C. DQCD

- Xác minh đúng objective contrastive, positive/negative pairs, mask, temperature, normalization và reduction của paper.
- Xác định feature/decoder layer nào được dùng và có gradient qua backbone/decoder không.
- Xử lý batch/query không có positive/negative hợp lệ, tránh NaN khi denominator rỗng.
- Nếu sử dụng high-frequency-energy proxy thay learned DAFD, phải giữ nhãn `Adaptation`, giải thích ảnh hưởng và thêm ablation/toggle. Không gọi là exact reproduction.

#### D. IQCM

- Đối chiếu định nghĩa quality score/target và cách kết hợp với class confidence.
- Dùng matched indices để gán target cho positive queries nếu paper/criterion yêu cầu. Không tùy tiện lấy max IoU với bất kỳ GT cho mọi query nếu có thể gán target sai.
- Xác định xử lý unmatched queries, no-GT images, top-M selection, detach hay không detach, loss reduction.
- Test trường hợp một GT, nhiều GT, không GT, query trùng và batch có số target khác nhau.

#### E. HIFRM và box losses

- Đối chiếu chính xác công thức paper: dùng IoU hay GIoU, dấu, focal exponent/gamma, clamp, weight, reduction và matched pairs.
- Xác minh box format và range. Test box trùng, box không giao nhau, box hợp lệ sát biên, empty targets và mixed precision.
- Không thay thế loss bbox/GIoU gốc nếu chưa chứng minh loss mới là phần thay thế có chủ đích.
- Thêm unit tests cho công thức trên các tensor nhỏ có expected value tính tay.

#### F. SCU/LUE

- Đối chiếu uncertainty definition, calibration target, variance/precision parameterization, regularization và loss equation của paper.
- Xác minh numerical stability, clamp/log/exp, detach policy, batch normalization/centering và gradients.
- Không dùng target tự đặt mà không ghi là adaptation.
- Test finite loss/gradients với confidence cực nhỏ/lớn, batch size 1 và batch không có positive target.

#### G. Weighted Tversky Loss

- Xác minh Tversky được áp dụng cho đúng logits/labels/queries; định nghĩa TP/FP/FN, alpha/beta, epsilon và weighting.
- Không áp dụng trực tiếp segmentation formula lên detection queries nếu chưa định nghĩa phép chuyển đổi hợp lệ.
- Xác minh tương thích multi-class, no-object class và label assignment của RF-DETR.
- Test alpha/beta, empty positives, numerical stability, finite gradients.
- Bảo đảm không đếm đôi classification objective với VFL/CE ngoài chủ đích.

#### H. Hard Negative Mining

- Xác định paper gốc dùng patch/image/instance/query-level mining. Ghi chính xác khác biệt với implementation.
- Kiểm tra hardness score, ambiguity score, top-k/fraction, tie handling và mask.
- Không để hard-negative selection loại bỏ toàn bộ positives hoặc tạo loss rỗng.
- Test số negatives được chọn, không trùng index, không vượt bounds và xử lý batch không có negative.
- Nếu dùng query-level adaptation, mô tả rõ và không ghi là patch-level method.

#### I. RFO

- Đối chiếu algorithm của paper: population initialization, update equations, bounds, stopping criteria, fitness selection, randomness và number of evaluations.
- Kiểm tra mỗi candidate thực sự truyền hyperparameters vào model/trainer; không chỉ ghi log candidate mà không áp dụng.
- Dùng validation mAP@0.5:0.95 nhất quán, cùng split và protocol; xử lý metric thiếu/NaN.
- Ghi rõ số trial, epochs/trial, compute budget và seed.
- Giữ chế độ RFO có thể tắt; unit-test deterministic behavior với seed cố định và bounds hợp lệ.
- Không chạy tìm kiếm GPU/Modal có chi phí nếu chưa được cho phép.

### Phase 4 — Config, logging và README

- Mỗi config field phải được đọc và có hiệu lực thực tế hoặc bị loại bỏ kèm giải thích. Không để cấu hình gây hiểu nhầm.
- Kiểm tra `evaluation`, `eval_dir`, `loss_weights`, `improvements`, `rfo`, `grad_accum_steps` có được sử dụng đúng không.
- Kiểm tra batch size physical/effective và gradient accumulation; không chỉ dựa vào README.
- Ghi log riêng cho loss gốc, từng custom loss, LR, RFO candidate, validation metric và checkpoint path.
- README phải phân biệt rõ: exact implementation, adaptation, chưa kiểm chứng, cách chạy test, và lệnh training có chi phí.
- Không ghi kết quả mAP/FPS nếu chưa có output thực.

## Tiêu chí nghiệm thu bắt buộc

Không được đánh dấu “hoàn tất” nếu chưa báo cáo từng mục:

- [ ] Import/syntax checks pass.
- [ ] Đúng RF-DETR version/API đã được xác minh.
- [ ] Unit tests cho từng custom loss và edge cases pass.
- [ ] Một forward + loss + backward nhỏ chạy được trên thiết bị khả dụng.
- [ ] Tất cả loss hữu hạn; gradient tới tham số cần học; optimizer chứa custom-head parameters.
- [ ] Không có loss key mồ côi, không có loss bị cộng hai lần ngoài chủ đích.
- [ ] Hook không bị đăng ký nhiều lần và không làm thay đổi shape contract.
- [ ] Save checkpoint → process mới → load → inference thành công.
- [ ] Baseline path vẫn hoạt động và không bị sửa ngoài phạm vi.
- [ ] Config được dùng đúng và mọi option có hiệu lực.
- [ ] Bảng đối chiếu paper/code hoàn chỉnh; từng module được phân loại Exact/Adaptation/Incorrect/Unverified.
- [ ] Nếu có benchmark, baseline và improved dùng cùng dataset split, seed/protocol và metric.
- [ ] README/changelog cập nhật và nêu rõ test nào đã chạy, test nào chưa chạy.

Training dài hoặc benchmark tốn phí là bước riêng, cần người dùng phê duyệt trước.

## Định dạng phản hồi sau mỗi nhiệm vụ

Trả lời ngắn gọn theo mẫu:

1. **Phát hiện:** lỗi/điểm chưa chắc chắn, kèm file/hàm.
2. **Bằng chứng:** paper section/equation hoặc source API/version.
3. **Thay đổi:** file/hàm và hành vi trước/sau.
4. **Kiểm thử:** lệnh chạy và kết quả thực tế; ghi `NOT RUN` nếu chưa chạy.
5. **Tình trạng:** `PASS`, `PARTIAL`, `BLOCKED` hoặc `FAIL`.
6. **Còn lại:** giới hạn/giả định/việc cần người dùng quyết định.
7. **Không khẳng định tuyệt đối:** chỉ nói “đã xác minh trong phạm vi các test liệt kê”, không nói “đúng 100%”.

## Lệnh bắt đầu khuyến nghị cho agent

> Đọc `AI_AGENT_CONTEXT.md` và file này. Chỉ thực hiện Phase 0 và Phase 1 trước: xác minh workspace/config/API version, lập audit matrix cho các module dựa trên paper Markdown có trong repo, không sửa code và không khởi chạy Modal/GPU. Trả về phát hiện kèm đường dẫn/hàm/phương trình nguồn, phân loại Exact/Adaptation/Incorrect/Unverified và đề xuất thứ tự patch. Chờ xác nhận trước khi sửa nhiều module.
