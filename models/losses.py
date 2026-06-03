"""
losses.py — Perceptual loss, GAN loss, and pixel loss for ESRGAN.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 feature maps (relu3_4 by default).

    The network weights are frozen — only used for feature extraction.

    Args:
        layer_idx : index into the VGG19 features sequential up to which
                    features are extracted (default 26 = relu3_4).
    """

    def __init__(self, layer_idx: int = 26):
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:layer_idx])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # ImageNet normalisation (applied before VGG)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.criterion = nn.L1Loss()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_feat = self.feature_extractor(self._normalize(sr))
        hr_feat = self.feature_extractor(self._normalize(hr.detach()))
        return self.criterion(sr_feat, hr_feat)


class GANLoss(nn.Module):
    """
    Relativistic average GAN loss (RaGAN) used in ESRGAN.

    For the generator   : fool discriminator on fake + real samples.
    For the discriminator: distinguish real from fake.

    Args:
        loss_type : 'ragan' (default) or 'vanilla'
    """

    def __init__(self, loss_type: str = "ragan"):
        super().__init__()
        self.loss_type = loss_type
        self.criterion = nn.BCEWithLogitsLoss()

    def _real_label(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    def _fake_label(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def generator_loss(
        self, d_real: torch.Tensor, d_fake: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_type == "ragan":
            loss_real = self.criterion(
                d_real - d_fake.mean(), self._fake_label(d_real)
            )
            loss_fake = self.criterion(
                d_fake - d_real.mean(), self._real_label(d_fake)
            )
            return (loss_real + loss_fake) / 2
        return self.criterion(d_fake, self._real_label(d_fake))

    def discriminator_loss(
        self, d_real: torch.Tensor, d_fake: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_type == "ragan":
            loss_real = self.criterion(
                d_real - d_fake.mean().detach(), self._real_label(d_real)
            )
            loss_fake = self.criterion(
                d_fake - d_real.mean().detach(), self._fake_label(d_fake)
            )
            return (loss_real + loss_fake) / 2
        loss_real = self.criterion(d_real, self._real_label(d_real))
        loss_fake = self.criterion(d_fake, self._fake_label(d_fake))
        return (loss_real + loss_fake) / 2


class ESRGANLoss(nn.Module):
    """
    Combined ESRGAN loss for the generator:
        L = w_pixel * L1 + w_perceptual * VGG + w_gan * RaGAN

    Args:
        w_pixel       : weight for pixel-level L1 loss (default 0.01)
        w_perceptual  : weight for perceptual loss   (default 1.0)
        w_gan         : weight for adversarial loss  (default 0.005)
    """

    def __init__(
        self,
        w_pixel: float = 0.01,
        w_perceptual: float = 1.0,
        w_gan: float = 0.005,
    ):
        super().__init__()
        self.w_pixel = w_pixel
        self.w_perceptual = w_perceptual
        self.w_gan = w_gan

        self.pixel_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.gan_loss = GANLoss(loss_type="ragan")

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        d_real: torch.Tensor,
        d_fake: torch.Tensor,
    ) -> dict:
        l_pixel = self.pixel_loss(sr, hr)
        l_perceptual = self.perceptual_loss(sr, hr)
        l_gan = self.gan_loss.generator_loss(d_real, d_fake)

        total = (
            self.w_pixel * l_pixel
            + self.w_perceptual * l_perceptual
            + self.w_gan * l_gan
        )
        return {
            "total": total,
            "pixel": l_pixel,
            "perceptual": l_perceptual,
            "gan": l_gan,
        }
