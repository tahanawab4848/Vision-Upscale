"""
dataset.py — PyTorch Dataset for ESRGAN training and inference.
"""
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SRDataset(Dataset):
    """
    Paired SR dataset.  Loads HR images and produces LR counterparts
    on the fly via bicubic downscaling.

    Directory layout expected:
        root/
            train/hr/  *.png  *.jpg  ...
            val/hr/    *.png  ...

    Args:
        hr_dir      : path to the folder containing HR images.
        patch_size  : HR patch size cropped during training (default 128).
        scale       : downscale factor for LR (default 4).
        augment     : apply random flip / rotation when True.
    """

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        hr_dir: str | Path,
        patch_size: int = 128,
        scale: int = 4,
        augment: bool = True,
    ):
        super().__init__()
        self.hr_dir = Path(hr_dir)
        self.patch_size = patch_size
        self.lr_size = patch_size // scale
        self.scale = scale
        self.augment = augment

        self.hr_paths = sorted(
            p for p in self.hr_dir.rglob("*") if p.suffix.lower() in self.IMG_EXTS
        )
        if not self.hr_paths:
            raise FileNotFoundError(f"No images found under {self.hr_dir}")

        self.to_tensor = transforms.ToTensor()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.hr_paths)

    # ------------------------------------------------------------------
    def _random_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            img = img.resize(
                (max(w, self.patch_size), max(h, self.patch_size)),
                Image.BICUBIC,
            )
            w, h = img.size
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        return img.crop((x, y, x + self.patch_size, y + self.patch_size))

    def _augment(self, hr: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
        k = random.randint(0, 3)
        if k:
            hr = hr.rotate(90 * k)
        return hr

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        hr = Image.open(self.hr_paths[idx]).convert("RGB")

        # Crop
        hr = self._random_crop(hr)

        # Augment
        if self.augment:
            hr = self._augment(hr)

        # Degrade to LR
        lr = hr.resize((self.lr_size, self.lr_size), Image.BICUBIC)

        return {
            "lr": self.to_tensor(lr),   # [3, H/scale, W/scale]  float32 in [0,1]
            "hr": self.to_tensor(hr),   # [3, H, W]
            "path": str(self.hr_paths[idx]),
        }


# ---------------------------------------------------------------------------
# Inference-only dataset (single folder, no pairing needed)
# ---------------------------------------------------------------------------

class InferenceDataset(Dataset):
    """
    Loads LR images from a folder for inference.

    Args:
        lr_dir : folder containing low-resolution images.
    """

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    def __init__(self, lr_dir: str | Path):
        self.lr_dir = Path(lr_dir)
        self.paths = sorted(
            p for p in self.lr_dir.rglob("*") if p.suffix.lower() in self.IMG_EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.lr_dir}")
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.paths[idx]).convert("RGB")
        return {"lr": self.to_tensor(img), "path": str(self.paths[idx])}
