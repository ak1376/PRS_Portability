# src/vae/model.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class VAEConfig:
    input_len: int
    latent_dim: int = 32
    hidden_channels: Tuple[int, ...] = (32, 64, 128)
    kernel_size: int = 9
    stride: int = 2
    padding: Optional[int] = 4  # Can be None for auto-calculation in FullyConvVAE1D
    use_batchnorm: bool = False  # <- default OFF for baseline
    model_type: str = "conv"  # "conv" or "fully_conv"

    lr: float = 1e-3
    beta: float = 0.01
    weight_decay: float = 0.0

    # ---- masking / inpainting ----
    mask_enabled: bool = False
    mask_block_len: int = 0
    mask_fill_value: str = "zero"     # {"mean","zero"}
    weight_masked: float = 1.0
    weight_unmasked: float = 0.0


class ConvVAE1D(nn.Module):
    """
    Minimal 1D Conv VAE for genotype vectors.
    Input:  x (B, L) or (B, 1, L)
    Output: recon (B, L), mu (B, Z), logvar (B, Z)
    """

    def __init__(
        self,
        input_len: int,
        latent_dim: int = 32,
        hidden_channels=(32, 64, 128),
        kernel_size: int = 9,
        stride: int = 2,
        padding: int = 4,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.latent_dim = int(latent_dim)
        self.hidden_channels = tuple(hidden_channels)
        self.k = int(kernel_size)
        self.s = int(stride)
        self.p = int(padding)
        self.use_bn = bool(use_batchnorm)

        # ---- encoder conv stack ----
        enc = []
        in_ch = 1
        for out_ch in self.hidden_channels:
            enc.append(nn.Conv1d(in_ch, out_ch, self.k, stride=self.s, padding=self.p))
            # batchnorm intentionally omitted for baseline
            enc.append(nn.ELU(inplace=True))
            in_ch = out_ch
        self.enc_conv = nn.Sequential(*enc)

        # infer encoded shape with a dummy forward
        with torch.no_grad():
            h = self.enc_conv(torch.zeros(2, 1, self.input_len))
            self.enc_ch, self.enc_len = h.shape[1], h.shape[2]
            flat_dim = self.enc_ch * self.enc_len

        self.fc_mu = nn.Linear(flat_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, self.latent_dim)

        # ---- decoder: z -> feature map ----
        self.fc_dec = nn.Linear(self.latent_dim, flat_dim)

        # ---- decoder convtranspose stack (mirror) ----
        dec = []
        rev = list(reversed(self.hidden_channels))
        in_ch = rev[0]

        for out_ch in rev[1:]:
            dec.append(
                nn.ConvTranspose1d(
                    in_ch, out_ch, self.k, stride=self.s, padding=self.p,
                    output_padding=self.s - 1
                )
            )
            # batchnorm intentionally omitted
            dec.append(nn.ELU(inplace=True))
            in_ch = out_ch

        dec.append(
            nn.ConvTranspose1d(
                in_ch, 1, self.k, stride=self.s, padding=self.p,
                output_padding=self.s - 1
            )
        )
        self.dec_conv = nn.Sequential(*dec)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Stable reparameterization:
        - clamp logvar
        - compute exp in float32 without autocast
        """
        # avoid importing amp globally if you want
        from torch.cuda.amp import autocast

        with autocast(enabled=False):
            mu32 = mu.float()
            logvar32 = logvar.float().clamp(-10.0, 10.0)
            std = torch.exp(0.5 * logvar32)
            eps = torch.randn_like(std)
            z = mu32 + std * eps
        return z.to(mu.dtype)

    def encode(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        h = self.enc_conv(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Optional but very effective “baseline safety”:
        logvar = logvar.clamp(-10.0, 10.0)

        return mu, logvar

    def decode(self, z: torch.Tensor):
        h = self.fc_dec(z).view(z.size(0), self.enc_ch, self.enc_len)
        y = self.dec_conv(h).squeeze(1)

        if y.size(1) > self.input_len:
            y = y[:, : self.input_len]
        elif y.size(1) < self.input_len:
            y = F.pad(y, (0, self.input_len - y.size(1)))

        # For HWE-normalized inputs, recon should be real-valued
        return y

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class FullyConvVAE1D(nn.Module):
    """
    Fully convolutional VAE - NO linear bottleneck layers.
    
    The latent space is a spatial tensor (B, latent_channels, spatial_len).
    Variational inference is done with 1x1 convolutions for mu and logvar.
    
    This avoids the training difficulties of linear bottlenecks.
    """

    def __init__(
        self,
        input_len: int,
        latent_dim: int = 32,  # interpreted as latent_channels
        hidden_channels: Tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 33,
        stride: int = 4,
        padding: Optional[int] = None,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.latent_channels = int(latent_dim)  # reuse latent_dim as channels
        self.hidden_channels = tuple(hidden_channels)
        self.k = int(kernel_size)
        self.s = int(stride)
        self.p = int(padding) if padding is not None else self.k // 2
        self.use_bn = bool(use_batchnorm)

        # ---- encoder conv stack ----
        enc = []
        in_ch = 1
        for out_ch in self.hidden_channels:
            enc.append(nn.Conv1d(in_ch, out_ch, self.k, stride=self.s, padding=self.p))
            if self.use_bn:
                enc.append(nn.BatchNorm1d(out_ch))
            enc.append(nn.ELU(inplace=True))
            in_ch = out_ch
        self.enc_conv = nn.Sequential(*enc)

        # Final conv to get mu and logvar (1x1 convolutions)
        # Use last hidden channel size
        last_ch = self.hidden_channels[-1]
        self.conv_mu = nn.Conv1d(last_ch, self.latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv1d(last_ch, self.latent_channels, kernel_size=1)

        # Infer encoded spatial length with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(2, 1, self.input_len)
            h = self.enc_conv(dummy)
            self.enc_spatial_len = h.shape[2]
            # Total latent dims for reference
            self.latent_dim = self.latent_channels * self.enc_spatial_len

        # ---- decoder: project back and upsample ----
        # First project latent channels back to decoder channels
        self.dec_proj = nn.Conv1d(self.latent_channels, last_ch, kernel_size=1)

        # Decoder convtranspose stack (mirror of encoder)
        dec = []
        rev = list(reversed(self.hidden_channels))  # e.g., [128, 64, 32]
        in_ch = rev[0]

        for out_ch in rev[1:]:
            dec.append(
                nn.ConvTranspose1d(
                    in_ch, out_ch, self.k, stride=self.s, padding=self.p,
                    output_padding=self.s - 1
                )
            )
            if self.use_bn:
                dec.append(nn.BatchNorm1d(out_ch))
            dec.append(nn.ELU(inplace=True))
            in_ch = out_ch

        # Final layer to get back to 1 channel
        dec.append(
            nn.ConvTranspose1d(
                in_ch, 1, self.k, stride=self.s, padding=self.p,
                output_padding=self.s - 1
            )
        )
        self.dec_conv = nn.Sequential(*dec)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization for spatial latent (B, C, L).
        """
        logvar = logvar.clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x: torch.Tensor):
        """
        Returns mu, logvar as (B, latent_channels, spatial_len) tensors.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        h = self.enc_conv(x)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h).clamp(-10.0, 10.0)
        return mu, logvar

    def decode(self, z: torch.Tensor):
        """
        z: (B, latent_channels, spatial_len)
        Returns: (B, input_len)
        """
        h = self.dec_proj(z)
        y = self.dec_conv(h).squeeze(1)

        # Adjust output size to match input
        if y.size(1) > self.input_len:
            y = y[:, :self.input_len]
        elif y.size(1) < self.input_len:
            y = F.interpolate(y.unsqueeze(1), size=self.input_len, mode='linear', align_corners=False).squeeze(1)

        return y

    def forward(self, x: torch.Tensor):
        """
        Returns: recon (B, L), mu (B, C, S), logvar (B, C, S)
        where C = latent_channels, S = spatial_len
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar