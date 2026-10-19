# RF-DETR training on Modal

## 1. Install local tools and configure the Roboflow secret

Install the Modal CLI/SDK if it is not already installed:

```powershell
py -m pip install modal
modal setup
```

In the Modal dashboard, create a Secret named `roboflow-api-key` containing the
environment variable `ROBOFLOW_API_KEY` with your Roboflow API key. Do not put
the key in source files. If a key was previously shared, revoke it and create a
replacement.

## 2. Start training (dataset setup is automatic)

From the project root, run:

```powershell
modal run .\train_model\modal_train.py --mode improve
```

Before training, the code checks `/vol/datasets/completed-project-5/coco` on
the `rfdetr-storage` Modal Volume. If train/valid COCO annotations are missing,
it downloads `nckhcict2025/completed-project` version 5 from Roboflow, saves it
to that Volume, and commits it. Later runs reuse the saved dataset. No local
download or manual `modal volume put` is needed.

To use a different volume, set `RFDETR_VOLUME_NAME` before running the command.
The Modal function uses an A10G GPU, accumulates four batches of four samples
(effective batch size 16), and persists checkpoints/logs to the volume.
Training may incur Modal GPU and storage charges; the command above starts a
real training job.

To copy the best checkpoint back to the machine after the run:

```powershell
modal volume get rfdetr-storage checkpoints/rfdetr_baseline/best.pth .\best.pth
```

The baseline uses RF-DETR's supported training arguments. The previous
`use_balanced_loss` and custom loss-weight YAML entries were removed because the
stock RF-DETR training API does not accept those fields.

## 3. Train the improved model

The baseline remains the default. Select the improved implementation explicitly:

```powershell
modal run .\train_model\modal_train.py --mode improve
```

To override its dataset or configuration:

```powershell
modal run .\train_model\modal_train.py --mode improve `
  --dataset-dir /vol/datasets/completed-project-5/coco `
  --config-path /workspace/config/rfdetr_improve_v1.yaml
```

The improved route pins `rfdetr==1.11.2` because its criterion/model integration
uses checked internal APIs. It runs a small RFO-style validation search (six
candidate runs of three epochs by default), then trains the selected
configuration for the configured full epoch count. It selects candidates using
validation mAP@0.5:0.95 and moves the next population toward the best observed
candidate. This is a bounded practical search, not a reproduction of the paper's
20-fox/100-iteration experiment. Search adds GPU time; the `rfo` section can be
added to `config/rfdetr_improve_v1.yaml` to change `population`,
`generations`, and `epochs_per_trial`, or set `enabled: false` to skip it.
Results are written to `rfo_results.csv` and `rfo_best.json` in the configured
log directory.

The training script integrates QAR, IQCM, HIFRM, SCU/LUE, DQCD, Tversky, and
query-level hard-negative mining into RF-DETR's actual model/criterion and
optimizer. Two scope adaptations are intentional: the Spectral-DETR paper's
learned DAFD gate is not part of the requested module set, so DQCD temperature
uses a high-frequency-energy proxy; and query-level hard-negative mining is an
adaptation, not the patch-level mining method in the cited arXiv paper. The
improved route caps the physical batch at four samples on its A10G and uses
gradient accumulation to preserve the selected effective batch size. The
baseline route and its trainer are unchanged.

For the current improvement config, download the checkpoint with:

```powershell
modal volume get rfdetr-storage checkpoints/rfdetr_improve_v1/best.pth .\best_improved.pth
```
