"""RF-DETR Training Module
Huấn luyện mô hình RF-DETR 50 epochs với cơ chế bảo vệ checkpoint toàn diện khi hết giờ (Timeout / SIGTERM).
Tự động đồng bộ và trích xuất toàn diện 6+ chỉ số loss vào file rfdetr_epoch_logs.csv chuẩn hóa.
"""
import os
import glob
import signal
import shutil
import yaml
import pandas as pd
from pathlib import Path
from typing import Callable, Optional

from train_model.epoch_logger import EpochLogger


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "rfdetr_baseline.yaml"
)


def sync_and_export_rfdetr_logs(output_dir: str, log_dir: str, epoch_logger: EpochLogger):
    """Trích xuất tự động và đầy đủ các chỉ số từ metrics.csv và TensorBoard event files
    để ghi vào rfdetr_epoch_logs.csv theo đúng chuẩn yêu cầu đề tài."""
    extracted_rows = []

    # 1. Đọc từ TensorBoard Event Files (chính xác và đầy đủ nhất cho từng step/epoch)
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_files = sorted(list(set(glob.glob(os.path.join(output_dir, "events*")) +
                             glob.glob(os.path.join(output_dir, "**", "events*"), recursive=True))))
        if event_files:
            all_events_by_step = {}
            step_durations = {}

            for ef in event_files:
                try:
                    ea = EventAccumulator(ef)
                    ea.Reload()
                    scalars = ea.Tags().get("scalars", [])

                    # Xác định thời điểm bắt đầu session huấn luyện trong file này
                    first_time = None
                    for tag_cand in ["train/lr", "train/loss", "train/loss_ce"]:
                        if tag_cand in scalars:
                            ev_list = ea.Scalars(tag_cand)
                            if ev_list:
                                first_time = ev_list[0].wall_time
                                break
                    if first_time is None and scalars:
                        first_time = ea.Scalars(scalars[0])[0].wall_time

                    # Thu thập các step có validation metric trong file này để tính duration
                    file_val_steps = []
                    if "val/mAP_50_95" in scalars:
                        for ev in ea.Scalars("val/mAP_50_95"):
                            file_val_steps.append((ev.step, ev.wall_time))
                    elif "val/loss" in scalars:
                        for ev in ea.Scalars("val/loss"):
                            file_val_steps.append((ev.step, ev.wall_time))

                    file_val_steps.sort(key=lambda x: x[0])
                    prev_t = first_time
                    for s, wt in file_val_steps:
                        if prev_t is not None and wt >= prev_t:
                            step_durations[s] = round(wt - prev_t, 2)
                        else:
                            step_durations[s] = 0.0
                        prev_t = wt

                    for tag in scalars:
                        for ev in ea.Scalars(tag):
                            step = ev.step
                            if step not in all_events_by_step:
                                all_events_by_step[step] = {}
                            all_events_by_step[step][tag] = ev.value
                except Exception as ee:
                    print(f"[!] Warning reading event file {ef}: {ee}")

            # Lọc các step có validation metric
            val_steps = sorted([s for s, d in all_events_by_step.items() if any(
                k in d for k in ["val/mAP_50_95", "val/loss", "val/mAP_50"])])
            if val_steps:
                for ep_idx, step in enumerate(val_steps, start=1):
                    d = all_events_by_step[step]

                    train_loss = float(
                        d.get("train/loss", d.get("val/loss", 0.0)))
                    cls_loss = float(
                        d.get("train/loss_ce", d.get("val/loss_ce", 0.0)))
                    box_loss = float(
                        d.get("train/loss_bbox", d.get("val/loss_bbox", 0.0)))
                    giou_loss = float(
                        d.get("train/loss_giou", d.get("val/loss_giou", 0.0)))
                    lr = float(d.get("train/lr", d.get("lr", 1e-4)))
                    val_prec = float(d.get("val/precision", 0.0))
                    val_rec = float(d.get("val/recall", 0.0))
                    val_map50 = float(d.get("val/mAP_50", 0.0))
                    val_map50_95 = float(d.get("val/mAP_50_95", 0.0))
                    epoch_time = step_durations.get(step, 0.0)

                    extracted_rows.append({
                        "Epoch": ep_idx,
                        "Train loss": round(train_loss, 5),
                        "Class loss": round(cls_loss, 5),
                        "Box loss": round(box_loss, 5),
                        "GIoU": round(giou_loss, 5),
                        "Learning rate": f"{lr:.6e}",
                        "Val Precision": round(val_prec, 4),
                        "Val Recall": round(val_rec, 4),
                        "val_mAP50": round(val_map50, 4),
                        "val_mAP50_95": round(val_map50_95, 4),
                        "Epoch Time (s)": epoch_time,
                    })
    except Exception as e:
        print(f"[!] Warning extracting from TensorBoard: {e}")

    # 2. Nếu TensorBoard chưa có, đọc từ metrics.csv
    if not extracted_rows:
        metrics_csv = Path(output_dir) / "metrics.csv"
        if metrics_csv.exists():
            try:
                df = pd.read_csv(metrics_csv)
                # Lọc các dòng có epoch hoặc val metrics
                if "epoch" in df.columns:
                    for _, row in df.iterrows():
                        ep = int(row.get("epoch", len(extracted_rows) + 1))
                        train_loss = float(
                            row.get("train/loss", row.get("loss", 0.0)))
                        cls_loss = float(
                            row.get("train/loss_ce", row.get("val/loss_ce", 0.0)))
                        box_loss = float(
                            row.get("train/loss_bbox", row.get("val/loss_bbox", 0.0)))
                        giou_loss = float(
                            row.get("train/loss_giou", row.get("val/loss_giou", 0.0)))
                        lr = float(row.get("train/lr", row.get("lr", 1e-4)))
                        val_prec = float(
                            row.get("val/precision", row.get("precision", 0.0)))
                        val_rec = float(
                            row.get("val/recall", row.get("recall", 0.0)))
                        val_map50 = float(
                            row.get("val/mAP_50", row.get("mAP_50", 0.0)))
                        val_map50_95 = float(
                            row.get("val/mAP_50_95", row.get("mAP_50_95", 0.0)))

                        extracted_rows.append({
                            "Epoch": ep,
                            "Train loss": round(train_loss, 5),
                            "Class loss": round(cls_loss, 5),
                            "Box loss": round(box_loss, 5),
                            "GIoU": round(giou_loss, 5),
                            "Learning rate": f"{lr:.6e}",
                            "Val Precision": round(val_prec, 4),
                            "Val Recall": round(val_rec, 4),
                            "val_mAP50": round(val_map50, 4),
                            "val_mAP50_95": round(val_map50_95, 4),
                            "Epoch Time (s)": 0.0,
                        })
            except Exception as me:
                print(f"[!] Warning reading metrics.csv: {me}")

    # 3. Ghi vào epoch_logger và lưu file rfdetr_epoch_logs.csv
    if extracted_rows:
        for r in extracted_rows:
            epoch_logger.logs_dict[int(r["Epoch"])] = r
        epoch_logger.save_csv()
        print(
            f"[+] [LOG EXPORT] Successfully exported {len(extracted_rows)} epochs to: {epoch_logger.log_file_path}")
    else:
        epoch_logger.save_csv()


def train_rfdetr(
    dataset_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    commit_fn: Optional[Callable[[], None]] = None,
    **kwargs,
):
    config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not config_file.is_file():
        raise FileNotFoundError(f"RF-DETR config not found: {config_file}")
    with config_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    outputs_cfg = cfg.get("outputs", {})

    dataset_dir = (
        dataset_dir
        or data_dir
        or data_cfg.get("dataset_dir")
        or "/vol/datasets/completed-project-5/coco"
    )
    output_dir = output_dir or outputs_cfg.get(
        "checkpoint_dir", "/vol/checkpoints/rfdetr_baseline"
    )
    log_dir = log_dir or outputs_cfg.get("log_dir", "/vol/logs/rfdetr_baseline")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    has_coco = all(
        os.path.isfile(
            os.path.join(dataset_dir, split, "_annotations.coco.json")
        )
        for split in ("train", "valid")
    )
    has_yolo = (
        os.path.isfile(os.path.join(dataset_dir, "data.yaml"))
        and os.path.isdir(os.path.join(dataset_dir, "train", "images"))
    )
    if not has_coco and not has_yolo:
        if "coco" in dataset_dir:
            yolo_alt = dataset_dir.replace("coco", "yolo")
            if os.path.isfile(os.path.join(yolo_alt, "data.yaml")) and os.path.isdir(
                os.path.join(yolo_alt, "train", "images")
            ):
                print(
                    f"[*] [AUTO SWITCH] Chuyển tự động sang thư mục YOLO hợp lệ: {yolo_alt}")
                dataset_dir = yolo_alt
                has_yolo = True
        elif "yolo" in dataset_dir:
            coco_alt = dataset_dir.replace("yolo", "coco")
            if all(
                os.path.isfile(
                    os.path.join(coco_alt, split, "_annotations.coco.json")
                )
                for split in ("train", "valid")
            ):
                print(
                    f"[*] [AUTO SWITCH] Chuyển tự động sang thư mục COCO hợp lệ: {coco_alt}")
                dataset_dir = coco_alt
                has_coco = True

    if not has_coco and not has_yolo:
        raise FileNotFoundError(
            "Dataset must be COCO (train/valid/_annotations.coco.json) or "
            f"YOLO (data.yaml and train/images): {dataset_dir}"
        )

    model_variant = model_cfg.get("variant", "RFDETRMedium")
    epochs = train_cfg.get("epochs", 80)
    patience = train_cfg.get("patience", 10)  # Cấu hình chuẩn patience = 10
    imgsz = train_cfg.get("imgsz", 640)
    batch_size = train_cfg.get("batch_size", 16)
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
    lr = float(train_cfg.get("lr", 0.0001))

    # 3. Kiểm tra checkpoint dở dang để resume nếu trước đó bị hết giờ (Timeout)
    resume_candidates = [
        Path(output_dir) / "checkpoint.pth",
        Path(output_dir) / "last.pth",
    ]
    ckpt_files = sorted(
        Path(output_dir).glob("checkpoint_*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for ckpt in ckpt_files:
        resume_candidates.append(ckpt)

    resume_path = None
    for cand in resume_candidates:
        if cand.exists():
            resume_path = str(cand)
            break

    if resume_path:
        print(
            f"[*] [RESUME SAFE] Tìm thấy checkpoint RF-DETR dở dang tại: {resume_path}. Tự động RESUME từ epoch dở dang...")
        metrics_file = Path(output_dir) / "metrics.csv"
        if metrics_file.exists():
            backup_metrics = Path(output_dir) / "metrics_epoch_backup.csv"
            shutil.copy(metrics_file, backup_metrics)
            print(
                f"[+] [BACKUP AN TOÀN] Đã sao lưu lịch sử metrics sang: {backup_metrics.name}")
    else:
        print(
            f"[*] [NEW RUN] Khởi tạo RF-DETR ({model_variant}) mới cho {epochs} epochs (patience={patience})...")

    log_file_csv = os.path.join(log_dir, "rfdetr_epoch_logs.csv")
    epoch_logger = EpochLogger(
        log_file_path=log_file_csv,
        model_name="RF-DETR",
        commit_fn=commit_fn,
    )

    def sync_rfdetr_checkpoints():
        """Đồng bộ các checkpoint của RF-DETR sang best.pth và last.pth."""
        best_candidates = [
            Path(output_dir) / "checkpoint_best_ema.pth",
            Path(output_dir) / "checkpoint_best_regular.pth",
            Path(output_dir) / "checkpoint_best.pth",
            Path(output_dir) / "best.pth",
        ]
        dest_best = Path(output_dir) / "best.pth"
        dest_last = Path(output_dir) / "last.pth"

        for cand in best_candidates:
            if cand.exists():
                if cand != dest_best:
                    shutil.copy(cand, dest_best)
                break

        current_last = Path(output_dir) / "checkpoint.pth"
        if current_last.exists() and current_last != dest_last:
            shutil.copy(current_last, dest_last)

        # Xuất và đồng bộ bảng log csv
        sync_and_export_rfdetr_logs(
            output_dir=output_dir, log_dir=log_dir, epoch_logger=epoch_logger)

        if commit_fn is not None:
            try:
                commit_fn()
            except Exception as e:
                print(f"[!] Warning commit storage: {e}")

    main_pid = os.getpid()

    # Xử lý Signal khi Modal gửi tín hiệu Timeout (SIGTERM)
    def handle_rfdetr_termination(signum, frame):
        if os.getpid() != main_pid:
            return
        print(
            f"\n[!] Nhận tín hiệu dừng (Signal {signum}) do hết thời gian. Đang lưu checkpoint RF-DETR...")
        sync_rfdetr_checkpoints()
        if commit_fn is not None:
            commit_fn()
        print("[+] Đã lưu toàn bộ tiến độ RF-DETR vào Storage trước khi dừng container.")
        raise SystemExit(128 + signum)

    try:
        signal.signal(signal.SIGTERM, handle_rfdetr_termination)
    except Exception:
        pass

    import rfdetr

    model_class = getattr(rfdetr, model_variant, None)
    if model_class is None:
        raise ValueError(
            f"Unsupported RF-DETR model variant {model_variant!r}. "
            "Check the installed rfdetr package and config."
        )
    if model_cfg.get("pretrained", True):
        model = model_class()
    else:
        model = model_class(pretrain_weights=None)

    print(
        f"[*] Bắt đầu huấn luyện RF-DETR trên {dataset_dir} (epochs={epochs}, patience={patience}, batch_size={batch_size}, imgsz={imgsz}, resume={bool(resume_path)})...")

    train_kwargs = {
        "dataset_dir": dataset_dir,
        "epochs": epochs,
        "early_stopping": True,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "lr": lr,
        "output_dir": output_dir,
        "resolution": imgsz,
        "num_workers": train_cfg.get("workers", 4),
        "weight_decay": train_cfg.get("weight_decay", 1e-4),
        "device": train_cfg.get("device", "cuda"),
        "seed": train_cfg.get("seed", 42),
        "eval_interval": train_cfg.get("eval_interval", 1),
        "checkpoint_interval": train_cfg.get("checkpoint_interval", 1),
        "tensorboard": train_cfg.get("tensorboard", True),
    }
    if "lr_encoder" in train_cfg:
        train_kwargs["lr_encoder"] = float(train_cfg["lr_encoder"])
    if resume_path:
        train_kwargs["resume"] = resume_path

    try:
        model.train(**train_kwargs)
    except Exception as exc:
        import traceback
        print(f"[!] Lỗi khi huấn luyện RF-DETR: {exc}")
        traceback.print_exc()
        raise exc
    finally:
        sync_rfdetr_checkpoints()
        if commit_fn is not None:
            commit_fn()

    dest_best = Path(output_dir) / "best.pth"
    if not dest_best.is_file():
        raise FileNotFoundError(
            f"RF-DETR training finished without a best checkpoint at {dest_best}"
        )
    print(f"[+] RF-DETR hoàn tất! Best Checkpoint: {dest_best}")
    return str(dest_best)


if __name__ == "__main__":
    train_rfdetr()
