# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-notebook project (`bc_classification.ipynb`) that trains a semantic-segmentation model on breast ultrasound images to label each pixel as **background / benign / malignant**. Despite the name "classification," the task is multi-class segmentation built on `segmentation-models-pytorch` (SMP) + PyTorch Lightning + Albumentations. There is no `src/`, no test suite, and no package — all code lives in the notebook cells.

## Running

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or a CUDA index URL
pip install -r requirements.txt
jupyter notebook bc_classification.ipynb
```

Install `torch`/`torchvision` first (their index URL depends on your CPU/CUDA hardware), then `requirements.txt` for everything else. The notebook is run top-to-bottom; there is no CLI entry point.

## Data layout (the key thing to understand)

Data is from a Kaggle competition, expected under `./data/complete_set` (train/test pre-split). The on-disk format is unusual and most custom code exists to adapt it:

- Images sit in class folders (`benign/`, `malignant/`) so `torchvision.datasets.ImageFolder` can read the **image-level** class.
- Pixel masks are **not** in a separate label tree — they live alongside each image as `<stem>_mask*.png` (an image may have several: `_mask.png`, `_mask_1.png`, …).

The data root is set once via `DATA_DIR = Path(os.environ.get("BC_DATA_DIR", "./data")) / "complete_set"`. It defaults to `./data/complete_set`; override the `BC_DATA_DIR` env var if your dataset lives elsewhere. The dataset is read from `DATA_DIR/training_set`, and `get_mean_std` runs over all of `DATA_DIR`.

## Architecture / data pipeline

1. **`SegmentationImageFolder(ImageFolder)`** — loads via ImageFolder, then **filters out `*_mask*` files** from `self.samples`, and re-attaches masks through `_find_masks` at `__getitem__` time. Multiple masks per image are unioned with `ImageChops.lighter` (per-pixel max). `__getitem__` returns a **3-tuple `(sample, mask, target)`** — not the usual `(sample, target)` — so the whole pipeline is built around 3-tuples. Also provides a static `get_mean_std(root)` used to normalize.

2. **Class-index offset (easy to get wrong):** ImageFolder `target` is `0=benign, 1=malignant`. Segmentation pixel labels are `0=background, 1=benign, 2=malignant`, i.e. `seg_label = target + 1`. Mask pixels are re-encoded with `np.where(mask>0, target+1, 0)` in `SubsetWithTransform`. Any code touching class names/IDs must account for this `+1`.

3. **Split + transforms:** `random_split` (seed 42) produces train/val `Subset`s of the *transform-less* dataset, then each is wrapped in **`SubsetWithTransform`**, which applies Albumentations (`A.Compose` with `mask=` so the same geometric ops apply to image and mask). Train gets augmentation (brightness/contrast, CLAHE, gamma, flips, rotate); val gets only resize+normalize+`ToTensorV2`.

4. **Model — `BreastCancerMammogramModel(pl.LightningModule)`:** `smp.create_model("UNET", encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=3)`. Loss = **Dice (multiclass, from_logits) + weighted CrossEntropy**, where the CE class weights are inverse-frequency weights computed from the train benign/malignant counts (background weight fixed at 1.0) to counter class imbalance. Metrics use `smp.metrics` IoU (`micro-imagewise` and `micro`), aggregated in `shared_epoch_end`. Optimizer Adam (lr 2e-4) + `CosineAnnealingLR`.

## Conventions / gotchas

- Reproducibility seeds (42) are set globally in an early cell and inside the split helper; keep them if editing.
- DataLoaders use `batch_size=32, num_workers=23` — tuned for a specific machine; lower `num_workers` on smaller boxes.
- `EPOCHS=50` and the scheduler `T_max` are set independently in two places — keep them consistent if changing epoch count.
- Class imbalance handling is partly a work-in-progress (see the `FIXME` markdown cell about the loss function).
