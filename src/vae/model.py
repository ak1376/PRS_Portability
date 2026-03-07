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
    padding: Optional[int] = 4
    use_batchnorm: bool = False
    model_type: str = "conv"

    lr: float = 1e-3
    beta: float = 0.01
    weight_decay: float = 0.0

    # ---- masking / inpainting ----
    mask_enabled: bool = False
    mask_block_len: int = 0
    mask_fill_value: str = "zero"
    weight_masked: float = 1.0
    weight_unmasked: float = 0.0


class ConvVAE1D(nn.Module):
    """
    1D Conv VAE for genotype classification.

    Input:
      x: (B, L) or (B, 1, L)

    Output:
      logits: (B, 3, L)
      mu:     (B, Z)
      logvar: (B, Z)
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

        enc = []
        in_ch = 1
        for out_ch in self.hidden_channels:
            enc.append(nn.Conv1d(in_ch, out_ch, self.k, stride=self.s, padding=self.p))
            enc.append(nn.ELU(inplace=True))
            in_ch = out_ch
        self.enc_conv = nn.Sequential(*enc)

        with torch.no_grad():
            h = self.enc_conv(torch.zeros(2, 1, self.input_len))
            self.enc_ch, self.enc_len = h.shape[1], h.shape[2]
            flat_dim = self.enc_ch * self.enc_len

        self.fc_mu = nn.Linear(flat_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, self.latent_dim)
        self.fc_dec = nn.Linear(self.latent_dim, flat_dim)

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
            dec.append(nn.ELU(inplace=True))
            in_ch = out_ch

        # final 3 channels = logits for genotype classes 0/1/2
        dec.append(
            nn.ConvTranspose1d(
                in_ch, 3, self.k, stride=self.s, padding=self.p,
                output_padding=self.s - 1
            )
        )
        self.dec_conv = nn.Sequential(*dec)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
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
            x = x.unsqueeze(1)
        h = self.enc_conv(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10.0, 10.0)
        return mu, logvar

    def decode(self, z: torch.Tensor):
        h = self.fc_dec(z).view(z.size(0), self.enc_ch, self.enc_len)
        logits = self.dec_conv(h)   # (B, 3, L_out)

        if logits.size(2) > self.input_len:
            logits = logits[:, :, : self.input_len]
        elif logits.size(2) < self.input_len:
            logits = F.pad(logits, (0, self.input_len - logits.size(2)))

        return logits

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


class FullyConvVAE1D(nn.Module):
    """
    Fully convolutional VAE for genotype classification.

    Input:
      x: (B, L) or (B, 1, L)

    Output:
      logits: (B, 3, L)
      mu:     (B, C, S)
      logvar: (B, C, S)
    """

    def __init__(
        self,
        input_len: int,
        latent_dim: int = 32,
        hidden_channels: Tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 33,
        stride: int = 4,
        padding: Optional[int] = None,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.latent_channels = int(latent_dim)
        self.hidden_channels = tuple(hidden_channels)
        self.k = int(kernel_size)
        self.s = int(stride)
        self.p = int(padding) if padding is not None else self.k // 2
        self.use_bn = bool(use_batchnorm)

        enc = []
        in_ch = 1
        for out_ch in self.hidden_channels:
            enc.append(nn.Conv1d(in_ch, out_ch, self.k, stride=self.s, padding=self.p))
            if self.use_bn:
                enc.append(nn.BatchNorm1d(out_ch))
            enc.append(nn.ELU(inplace=True))
            in_ch = out_ch
        self.enc_conv = nn.Sequential(*enc)

        last_ch = self.hidden_channels[-1]
        self.conv_mu = nn.Conv1d(last_ch, self.latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv1d(last_ch, self.latent_channels, kernel_size=1)

        with torch.no_grad():
            dummy = torch.zeros(2, 1, self.input_len)
            h = self.enc_conv(dummy)
            self.enc_spatial_len = h.shape[2]
            self.latent_dim = self.latent_channels * self.enc_spatial_len

        self.dec_proj = nn.Conv1d(self.latent_channels, last_ch, kernel_size=1)

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
            if self.use_bn:
                dec.append(nn.BatchNorm1d(out_ch))
            dec.append(nn.ELU(inplace=True))
            in_ch = out_ch

        dec.append(
            nn.ConvTranspose1d(
                in_ch, 3, self.k, stride=self.s, padding=self.p,
                output_padding=self.s - 1
            )
        )
        self.dec_conv = nn.Sequential(*dec)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = logvar.clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.enc_conv(x)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h).clamp(-10.0, 10.0)
        return mu, logvar

    def decode(self, z: torch.Tensor):
        h = self.dec_proj(z)
        logits = self.dec_conv(h)   # (B, 3, L_out)

        if logits.size(2) > self.input_len:
            logits = logits[:, :, :self.input_len]
        elif logits.size(2) < self.input_len:
            logits = F.interpolate(logits, size=self.input_len, mode="linear", align_corners=False)

        return logits

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar