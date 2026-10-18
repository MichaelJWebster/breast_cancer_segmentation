"""Dataset classes for bc_classification.ipynb.

These live in a module rather than in the notebook so that DataLoader worker
processes can import them. On macOS (and Windows) workers are started with
"spawn", which re-imports everything from scratch and cannot see classes defined
inside a Jupyter kernel; with num_workers > 0 the workers would crash.
"""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops
from torch.utils.data import Dataset as BaseDataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import pil_loader


def default_num_workers() -> int:
    """One DataLoader worker per CPU core, leaving one core free."""
    return max(0, (os.cpu_count() or 1) - 1)


class SegmentationImageFolder(ImageFolder):
    # Class labels for multi-class segmentation
    # 0 = background, 1 = benign, 2 = malignant
    SEGMENTATION_CLASSES = ['background', 'benign', 'malignant']
    NUM_CLASSES = 3

    def __init__(self, root, transform=None, loader=None):
        if loader is None:
            super().__init__(root, transform=None)  # Don't pass transform to parent
        else:
            super().__init__(root, transform=None, loader=loader)

        # Store albumentations transform separately
        self.albu_transform = transform

        # Filter out masks from the self.samples list
        self.samples = [(path, target) for path, target in self.samples
                        if not '_mask' in path]
        self.imgs = self.samples  # For compatibility with ImageFolder

        # Number of segmentation classes (background + benign + malignant)
        self.num_classes = self.NUM_CLASSES

        self.class_to_name = {i: j for i, j in enumerate(self.SEGMENTATION_CLASSES)}

    @staticmethod
    def get_mean_std(root):
        """Calculate mean and std of the dataset after applying transform."""
        means = []
        stds = []

        pngs = list(Path(root).rglob("*.png"))
        samples = [png for png in pngs if not '_mask' in png.name]
        for path in samples:
            img = pil_loader(path)
            img_np = np.array(img)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            img_np = img_tensor.numpy()
            means.append(np.mean(img_np, axis=(1, 2)))  # Mean per channel: (C, H, W) -> (C,)
            stds.append(np.std(img_np, axis=(1, 2)))    # Std per channel: (C, H, W) -> (C,)

        mean = np.mean(means, axis=0)
        std = np.mean(stds, axis=0)

        return mean, std

    def __getitem__(self, index):
        # Get original image path and class label
        path, target = self.samples[index]

        # Load the image
        sample = self.loader(path)

        # Find all corresponding masks
        masks = self._find_masks(path)

        # Combine masks if multiple exist using max (union of all masks)
        if len(masks) > 1:
            combined = masks[0]
            for mask in masks[1:]:
                combined = ImageChops.lighter(combined, mask)  # Takes max of each pixel
            masks = combined
        elif len(masks) == 1:
            masks = masks[0]
        else:
            # No mask found - create empty mask
            masks = None

        return sample, masks, target

    def _find_masks(self, image_path):
        """Find all mask files corresponding to an image."""
        image_path = Path(image_path)

        # Extract the base name, e.g., "benign (4).png" -> "benign (4)"
        mask_patt = f"{image_path.stem}_mask*.png"

        # Find all masks: base_name + "_mask.png" and base_name + "_mask_N.png"
        mask_files = list(image_path.parent.glob(mask_patt))

        masks = [Image.open(mask_file).convert('L') for mask_file in mask_files]

        return masks


class SubsetWithTransform(BaseDataset):
    """A subset of a dataset with a specific transform applied."""

    def __init__(self, subset: Subset, transform=None):
        # subset should be a subset WITHOUT transform
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        sample, mask, target = self.subset[idx]
        # Convert to numpy arrays for albumentations
        sample_np = np.array(sample)
        mask_np = np.array(mask) if mask is not None else np.zeros(sample_np.shape[:2], dtype=np.uint8)

        # Encode class into mask for multi-class segmentation:
        # 0 = background, 1 = benign (target=0), 2 = malignant (target=1)
        mask_np = np.where(mask_np > 0, target + 1, 0).astype(np.uint8)
        if self.transform:
            transformed = self.transform(image=sample_np, mask=mask_np)
            sample = transformed['image']
            mask = transformed['mask']
        else:
            sample = torch.from_numpy(sample_np).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask_np)
        return sample, mask, target
