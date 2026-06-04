"""
utils.py — Metrics, image helpers, and checkpoint utilities.
"""
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.utils import save_image


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "esrgan") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB)."""
    with torch.no_grad():
        mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(max_val / math.sqrt(mse))


def ssim(
    sr: torch.Tensor, hr: torch.Tensor, window_size: int = 11
) -> float:
    """
    Structural Similarity Index (simplified, per-channel mean).
    Both tensors: [B, C, H, W] in [0, 1].
    """
    C1, C2 = 0.01**2, 0.03**2
    import torch.nn.functional as F

    def _gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

    win = _gaussian_window(window_size).to(sr.device)
    win = win.expand(sr.shape[1], 1, window_size, window_size)

    pad = window_size // 2
    mu1 = F.conv2d(sr, win, padding=pad, groups=sr.shape[1])
    mu2 = F.conv2d(hr, win, padding=pad, groups=hr.shape[1])

    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(sr * sr, win, padding=pad, groups=sr.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(hr * hr, win, padding=pad, groups=hr.shape[1]) - mu2_sq
    sigma12 = F.conv2d(sr * hr, win, padding=pad, groups=sr.shape[1]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert [C, H, W] float tensor in [0, 1] to PIL Image."""
    t = t.clamp(0, 1).cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


def save_sr_image(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a [C, H, W] or [1, C, H, W] SR tensor as PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    save_image(tensor.clamp(0, 1), path)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    save_dir: str | Path,
    prefix: str = "esrgan",
) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{prefix}_epoch{epoch:04d}.pth"
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module | None = None,
    opt_g: torch.optim.Optimizer | None = None,
    opt_d: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> int:
    """Load checkpoint and return the epoch number."""
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    if discriminator is not None and "discriminator" in ckpt:
        discriminator.load_state_dict(ckpt["discriminator"])
    if opt_g is not None and "opt_g" in ckpt:
        opt_g.load_state_dict(ckpt["opt_g"])
    if opt_d is not None and "opt_d" in ckpt:
        opt_d.load_state_dict(ckpt["opt_d"])
    return ckpt.get("epoch", 0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
