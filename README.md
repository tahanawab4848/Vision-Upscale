# ESRGAN — Enhanced Super-Resolution GAN

A clean PyTorch implementation of **ESRGAN** (Wang et al., 2018) with:

- **RRDB Generator** — Residual-in-Residual Dense Blocks with 23 RRDB units
- **VGG Discriminator** — relativistic average GAN (RaGAN)
- **Perceptual + GAN + Pixel loss** — combined generator objective
- **Tiled inference** — handles arbitrarily large images without OOM
- **Two-phase training** — PSNR pre-train → GAN fine-tune

---

## Project structure

```
esrgan_project/
├── models/
│   ├── generator.py       # RRDBNet (RRDB + pixel-shuffle upsampling)
│   ├── discriminator.py   # VGG-style discriminator
│   └── losses.py          # Perceptual, GAN (RaGAN), combined ESRGAN loss
├── data/
│   └── dataset.py         # SRDataset (paired), InferenceDataset
├── utils/
│   └── utils.py           # PSNR, SSIM, checkpointing, image I/O
├── train.py               # GAN training loop
├── pretrain.py            # Phase-1 PSNR pre-training
├── infer.py               # Single image / folder inference
├── gui.py                 # Interactive Tkinter Desktop GUI
├── fingerprint_forensics.py # Latent fingerprint recovery & matching
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Data preparation

Organise your HR images:

```
data/
  train/hr/   ← high-resolution training images
  val/hr/     ← high-resolution validation images
```

LR counterparts are generated **on the fly** via bicubic downscaling (×4).

---

## Training

### Phase 1 — PSNR pre-training (recommended)

```bash
python pretrain.py \
    --hr_dir  data/train/hr \
    --val_dir data/val/hr \
    --epochs  50 \
    --batch_size 16
```

Saves checkpoints to `checkpoints/pretrain/`.

### Phase 2 — GAN fine-tuning

```bash
python train.py \
    --hr_dir  data/train/hr \
    --val_dir data/val/hr \
    --pretrain checkpoints/pretrain/pretrain_epoch0050.pth \
    --epochs  100 \
    --batch_size 16
```

Resume from a GAN checkpoint:

```bash
python train.py --hr_dir data/train/hr --checkpoint checkpoints/esrgan_epoch0060.pth
```

---

## Inference

```bash
# Single image
python infer.py \
    --input photo.jpg \
    --checkpoint checkpoints/esrgan_epoch0100.pth

# Whole folder
python infer.py \
    --input lr_images/ \
    --checkpoint checkpoints/esrgan_epoch0100.pth \
    --output outputs/sr/

# Half-precision (faster on CUDA)
python infer.py --input photo.jpg --checkpoint ... --fp16

# Tiled inference for large images (tile=0 disables tiling)
python infer.py --input large.png --checkpoint ... --tile 512 --tile_pad 32
```

---

## Desktop GUI

You can launch a fully featured desktop client built with Tkinter for an easy-to-use, visual upscaling experience:

```bash
python gui.py
```

Features include:
- Visual drag-and-drop file selection
- Model checkpoint switching
- Real-time side-by-side comparison of LR and SR images
- Integration with standard outputs

---

## Fingerprint Forensics Integration

This repository includes a forensic-grade biometric application demonstrating real-world use cases for neural super-resolution:

```bash
python fingerprint_forensics.py
```

This pipeline automatically:
1. Generates synthetic degraded "latent" fingerprints.
2. Uses ESRGAN to reconstruct the fingerprint.
3. Applies OpenCV biometric minutiae extraction to both.
4. Performs objective minutiae matching (BFM) and provides a visual forensic comparison.

---

## Key hyper-parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_blocks` | 23 | RRDB count in generator |
| `--scale` | 4 | Upscale factor |
| `--patch_size` | 128 | HR crop size during training |
| `--w_pixel` | 0.01 | L1 pixel loss weight |
| `--w_perceptual` | 1.0 | VGG perceptual loss weight |
| `--w_gan` | 0.005 | Adversarial loss weight |

---

## References

- Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks" (ECCVW 2018)
- Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data" (ICCVW 2021)
