"""
pretrain.py — Phase-1 PSNR pre-training (L1 loss only, no GAN).

Pre-training a stable generator before adversarial fine-tuning avoids
training instability and produces better textures in the GAN stage.

Usage:
    python pretrain.py --hr_dir data/train/hr --val_dir data/val/hr

The saved checkpoint can be passed to train.py via --pretrain.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import SRDataset
from models import RRDBNet
from utils import get_logger, psnr, save_checkpoint, load_checkpoint, count_parameters


def parse_args():
    p = argparse.ArgumentParser(description="ESRGAN PSNR Pre-training")
    p.add_argument("--hr_dir", required=True)
    p.add_argument("--val_dir", default=None)
    p.add_argument("--checkpoint", default=None, help="Resume checkpoint")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--num_blocks", type=int, default=23)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save_dir", default="checkpoints/pretrain")
    p.add_argument("--output_dir", default="outputs/pretrain")
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    logger = get_logger("pretrain")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_ds = SRDataset(args.hr_dir, args.patch_size, args.scale, augment=True)
    loader = DataLoader(
        train_ds, args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = RRDBNet(3, 3, 64, args.num_blocks, 32, args.scale).to(device)
    logger.info(f"Generator params: {count_parameters(model):,}")

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(args.checkpoint, model, device=str(device))
        logger.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(loader, 1):
            lr_img = batch["lr"].to(device)
            hr_img = batch["hr"].to(device)

            sr = model(lr_img)
            loss = criterion(sr, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % args.log_every == 0:
                logger.info(
                    f"[Pretrain] Epoch {epoch} [{step}/{len(loader)}] "
                    f"L1={loss.item():.5f}"
                )

        scheduler.step()
        avg = running_loss / len(loader)
        logger.info(f"Epoch {epoch}/{args.epochs} — avg L1={avg:.5f}")

        # Validation
        if args.val_dir and epoch % args.val_every == 0:
            model.eval()
            val_ds = SRDataset(args.val_dir, args.patch_size, args.scale, augment=False)
            val_loader = DataLoader(val_ds, 1, shuffle=False, num_workers=2)
            total_psnr = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    lr_img = batch["lr"].to(device)
                    hr_img = batch["hr"].to(device)
                    sr = model(lr_img).clamp(0, 1)
                    total_psnr += psnr(sr, hr_img)
            logger.info(f"[Val] PSNR = {total_psnr / len(val_loader):.2f} dB")

        # Save
        if epoch % 10 == 0:
            path = Path(args.save_dir) / f"pretrain_epoch{epoch:04d}.pth"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "generator": model.state_dict()}, path)
            logger.info(f"Saved: {path}")

    logger.info("Pre-training complete.")


if __name__ == "__main__":
    main()
