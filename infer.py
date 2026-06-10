"""
infer.py — Run ESRGAN super-resolution on one image or a folder.

Usage:
    # Single image
    python infer.py --input photo.jpg --checkpoint checkpoints/esrgan_epoch0100.pth

    # Folder
    python infer.py --input lr_images/ --checkpoint checkpoints/esrgan_epoch0100.pth \
                    --output outputs/sr/

Options:
    --input       Path to LR image or folder of LR images
    --checkpoint  Generator checkpoint (.pth)
    --output      Output folder  (default: outputs/infer)
    --tile        Tile size for large images (0 = no tiling, default 512)
    --tile_pad    Padding around each tile to remove seams (default 32)
    --scale       Upscale factor (default 4, must match the checkpoint)
    --num_blocks  RRDB block count used when the model was trained (default 23)
    --fp16        Use half-precision inference (faster on modern GPUs)
"""

import argparse
import time
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import RRDBNet
from utils import get_logger, save_sr_image


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ESRGAN Inference")
    p.add_argument("--input", required=True, help="LR image or folder")
    p.add_argument("--checkpoint", required=True, help="Generator .pth checkpoint")
    p.add_argument("--output", default="outputs/infer", help="Output folder")
    p.add_argument("--tile", type=int, default=512, help="Tile size (0 = disabled)")
    p.add_argument("--tile_pad", type=int, default=32, help="Tile overlap padding")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--num_blocks", type=int, default=23)
    p.add_argument("--fp16", action="store_true", help="Half-precision inference")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tiled inference (avoids OOM on large images)
# ---------------------------------------------------------------------------

def infer_tiled(
    model: RRDBNet,
    lr: torch.Tensor,
    scale: int,
    tile: int,
    pad: int,
    device: torch.device,
    dtype: torch.dtype,
    logger: logging.Logger = None,
) -> torch.Tensor:
    """
    Split a large LR tensor into tiles, upscale each, stitch back together.
    Avoids edge artifacts by using padding and cropping out the center core region.

    lr: [1, 3, H, W] on CPU.
    Returns: [1, 3, H*scale, W*scale] on CPU.
    """
    _, _, h, w = lr.shape
    sr = torch.zeros(1, 3, h * scale, w * scale)

    stride = tile - pad * 2
    xs = list(range(0, w, stride))
    ys = list(range(0, h, stride))

    total_tiles = len(ys) * len(xs)
    current_tile = 0

    for y in ys:
        for x in xs:
            current_tile += 1
            if logger:
                logger.info(f"Processing tile {current_tile}/{total_tiles}...")
            
            # Core region (the part of the image this tile is responsible for)
            core_x_start = x
            core_x_end = min(x + stride, w)
            core_y_start = y
            core_y_end = min(y + stride, h)

            # Padded patch bounds (context for the model)
            patch_x_start = max(core_x_start - pad, 0)
            patch_x_end = min(core_x_end + pad, w)
            patch_y_start = max(core_y_start - pad, 0)
            patch_y_end = min(core_y_end + pad, h)

            patch = lr[:, :, patch_y_start:patch_y_end, patch_x_start:patch_x_end].to(device, dtype)
            with torch.no_grad():
                sr_patch = model(patch).cpu().float()

            # Offsets of the core region within the padded patch
            offset_x = core_x_start - patch_x_start
            offset_y = core_y_start - patch_y_start

            # Extract core region from the upscaled SR patch
            sr_offset_x = offset_x * scale
            sr_offset_y = offset_y * scale
            sr_core_w = (core_x_end - core_x_start) * scale
            sr_core_h = (core_y_end - core_y_start) * scale

            sr_core = sr_patch[:, :, sr_offset_y : sr_offset_y + sr_core_h, sr_offset_x : sr_offset_x + sr_core_w]

            # Destination in full SR canvas
            dx, dy = core_x_start * scale, core_y_start * scale
            sr[:, :, dy : dy + sr_core_h, dx : dx + sr_core_w] = sr_core

    return sr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def upscale(path: Path, model, args, device, dtype, out_dir, logger):
    to_tensor = transforms.ToTensor()

    img = Image.open(path).convert("RGB")
    lr = to_tensor(img).unsqueeze(0)

    t0 = time.perf_counter()
    if args.tile > 0 and (lr.shape[2] > args.tile or lr.shape[3] > args.tile):
        sr = infer_tiled(model, lr, args.scale, args.tile, args.tile_pad, device, dtype, logger)
    else:
        with torch.no_grad():
            sr = model(lr.to(device, dtype)).cpu().float()

    elapsed = time.perf_counter() - t0
    out_path = out_dir / (path.stem + "_SR.png")
    save_sr_image(sr[0], out_path)
    logger.info(f"{path.name} → {out_path.name}  [{elapsed:.2f}s]")


def main():
    args = parse_args()
    logger = get_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    logger.info(f"Device: {device}  dtype: {dtype}")

    # Build model
    model = RRDBNet(
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_blocks=args.num_blocks,
        scale=args.scale,
    ).to(device, dtype)
    model.eval()

    # Load weights
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("params_ema", ckpt.get("generator", ckpt))
    
    # Handle old ESRGAN / RealESRGAN checkpoint keys automatically
    new_state = {}
    for k, v in state.items():
        k = k.replace("RRDB_trunk.", "body.")
        k = k.replace(".RDB1.", ".db1.")
        k = k.replace(".RDB2.", ".db2.")
        k = k.replace(".RDB3.", ".db3.")
        k = k.replace(".rdb1.", ".db1.")
        k = k.replace(".rdb2.", ".db2.")
        k = k.replace(".rdb3.", ".db3.")
        k = k.replace("trunk_conv.", "conv_body.")
        k = k.replace("upconv1.", "upsample.1.")
        k = k.replace("upconv2.", "upsample.4.")
        k = k.replace("conv_up1.", "upsample.1.")
        k = k.replace("conv_up2.", "upsample.4.")
        k = k.replace("HRconv.", "conv_hr.")
        new_state[k] = v
    state = new_state

    model.load_state_dict(state, strict=True)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_file():
        upscale(input_path, model, args, device, dtype, out_dir, logger)
    elif input_path.is_dir():
        paths = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in IMG_EXTS)
        logger.info(f"Found {len(paths)} image(s) in {input_path}")
        for p in paths:
            upscale(p, model, args, device, dtype, out_dir, logger)
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
