"""
train.py — ESRGAN training loop.

Usage:
    python train.py --hr_dir data/train/hr --val_dir data/val/hr

Key arguments:
    --hr_dir       Path to folder with HR training images
    --val_dir      Path to folder with HR validation images
    --checkpoint   Resume from an existing checkpoint (.pth)
    --pretrain     Path to a pre-trained generator (PSNR stage) to warm-start
    --epochs       Total training epochs  (default 100)
    --batch_size   Batch size             (default 16)
    --patch_size   HR patch size          (default 128)
    --scale        Super-resolution scale (default 4)
    --lr_g         Generator LR           (default 1e-4)
    --lr_d         Discriminator LR       (default 1e-4)
    --save_dir     Where to store checkpoints (default checkpoints/)
    --output_dir   Where to save validation SR images (default outputs/)
    --log_every    Print log every N batches (default 50)
    --val_every    Run validation every N epochs (default 5)
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import SRDataset
from models import RRDBNet, VGGDiscriminator, ESRGANLoss, GANLoss
from utils import (
    get_logger,
    psnr,
    ssim,
    save_sr_image,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ESRGAN Training")
    p.add_argument("--hr_dir", required=True, help="HR training images folder")
    p.add_argument("--val_dir", default=None, help="HR validation images folder")
    p.add_argument("--checkpoint", default=None, help="Resume checkpoint path")
    p.add_argument("--pretrain", default=None, help="Pre-trained generator weights")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--num_blocks", type=int, default=23)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--w_pixel", type=float, default=0.01)
    p.add_argument("--w_perceptual", type=float, default=1.0)
    p.add_argument("--w_gan", type=float, default=0.005)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    generator,
    discriminator,
    loader,
    opt_g,
    opt_d,
    criterion_g,
    criterion_d,
    device,
    epoch,
    args,
    logger,
):
    generator.train()
    discriminator.train()

    total_g, total_d = 0.0, 0.0

    for step, batch in enumerate(loader, 1):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        # ---- Train Discriminator ----------------------------------------
        with torch.no_grad():
            sr = generator(lr)

        d_real = discriminator(hr)
        d_fake = discriminator(sr.detach())
        loss_d = criterion_d.discriminator_loss(d_real, d_fake)

        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # ---- Train Generator --------------------------------------------
        sr = generator(lr)
        d_real = discriminator(hr).detach()
        d_fake = discriminator(sr)
        losses = criterion_g(sr, hr, d_real, d_fake)

        opt_g.zero_grad()
        losses["total"].backward()
        opt_g.step()

        total_g += losses["total"].item()
        total_d += loss_d.item()

        if step % args.log_every == 0:
            logger.info(
                f"Epoch {epoch} [{step}/{len(loader)}] "
                f"G={losses['total'].item():.4f} "
                f"(pix={losses['pixel'].item():.4f} "
                f"perc={losses['perceptual'].item():.4f} "
                f"gan={losses['gan'].item():.4f}) "
                f"D={loss_d.item():.4f}"
            )

    return total_g / len(loader), total_d / len(loader)


@torch.no_grad()
def validate(generator, val_dir, patch_size, scale, device, output_dir, epoch, logger):
    from data import SRDataset
    from torch.utils.data import DataLoader

    generator.eval()
    ds = SRDataset(val_dir, patch_size=patch_size, scale=scale, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    total_psnr, total_ssim = 0.0, 0.0
    out_dir = Path(output_dir) / f"epoch_{epoch:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(loader):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        sr = generator(lr).clamp(0, 1)

        total_psnr += psnr(sr, hr)
        total_ssim += ssim(sr, hr)
        save_sr_image(sr[0], out_dir / f"{i:04d}_sr.png")

    n = len(loader)
    avg_psnr = total_psnr / n
    avg_ssim = total_ssim / n
    logger.info(f"[Val epoch {epoch}] PSNR={avg_psnr:.2f} dB  SSIM={avg_ssim:.4f}")
    return avg_psnr, avg_ssim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    logger = get_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Datasets / Loaders
    train_ds = SRDataset(
        args.hr_dir, patch_size=args.patch_size, scale=args.scale, augment=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"Training samples: {len(train_ds)}")

    # Models
    generator = RRDBNet(
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_blocks=args.num_blocks,
        scale=args.scale,
    ).to(device)

    discriminator = VGGDiscriminator(in_channels=3, num_features=64).to(device)

    logger.info(f"Generator params  : {count_parameters(generator):,}")
    logger.info(f"Discriminator params: {count_parameters(discriminator):,}")

    # Losses
    criterion_g = ESRGANLoss(
        w_pixel=args.w_pixel,
        w_perceptual=args.w_perceptual,
        w_gan=args.w_gan,
    ).to(device)
    criterion_d = GANLoss(loss_type="ragan")

    # Optimisers
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.9, 0.99))
    opt_d = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.99)
    )

    # LR schedulers (halve every 50k iters ~ 25 epochs for batch=16)
    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(
        opt_g, milestones=[50, 100, 150, 200], gamma=0.5
    )
    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
        opt_d, milestones=[50, 100, 150, 200], gamma=0.5
    )

    start_epoch = 0

    # Load pre-trained / checkpoint
    if args.checkpoint:
        start_epoch = load_checkpoint(
            args.checkpoint, generator, discriminator, opt_g, opt_d, device=str(device)
        )
        logger.info(f"Resumed from epoch {start_epoch}: {args.checkpoint}")
    elif args.pretrain:
        ckpt = torch.load(args.pretrain, map_location=device)
        key = "generator" if "generator" in ckpt else "params"
        generator.load_state_dict(ckpt[key])
        logger.info(f"Loaded pre-trained generator: {args.pretrain}")

    # Training loop
    for epoch in range(start_epoch + 1, args.epochs + 1):
        avg_g, avg_d = train_one_epoch(
            generator, discriminator, train_loader,
            opt_g, opt_d,
            criterion_g, criterion_d,
            device, epoch, args, logger,
        )
        scheduler_g.step()
        scheduler_d.step()
        logger.info(
            f"Epoch {epoch}/{args.epochs} done — avg G={avg_g:.4f} D={avg_d:.4f}"
        )

        # Validation
        if args.val_dir and epoch % args.val_every == 0:
            validate(
                generator, args.val_dir, args.patch_size, args.scale,
                device, args.output_dir, epoch, logger,
            )

        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = save_checkpoint(
                epoch, generator, discriminator, opt_g, opt_d, args.save_dir
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
