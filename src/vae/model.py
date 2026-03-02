# src/vae/model.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


@dataclass
class VAEConfig:
    input_len: int
    latent_dim: int = 32
    hidden_channels: Tuple[int, ...] = (32, 64, 128)
    kernel_size: int = 9
    stride: int = 2
    padding: int = 4
    use_batchnorm: bool = False  # <- default OFF for baseline

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

        # Bound recon to [0,2] (dosage-like)
        y = 2.0 * torch.sigmoid(y)
        return y

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar