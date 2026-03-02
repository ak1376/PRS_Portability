# src/vae/lit_model.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import torch
import pytorch_lightning as pl

from src.vae.model import ConvVAE1D
from src.vae.loss import VAELoss


class LitVAE(pl.LightningModule):
    def __init__(self, cfg: Any):
        super().__init__()

        if is_dataclass(cfg):
            self.save_hyperparameters(asdict(cfg))
        elif isinstance(cfg, dict):
            self.save_hyperparameters(cfg)
        else:
            self.save_hyperparameters({"cfg": str(cfg)})

        self.cfg = cfg

        self.model = ConvVAE1D(
            input_len=cfg.input_len,
            latent_dim=getattr(cfg, "latent_dim", 32),
            hidden_channels=getattr(cfg, "hidden_channels", (32, 64, 128)),
            kernel_size=getattr(cfg, "kernel_size", 9),
            stride=getattr(cfg, "stride", 2),
            padding=getattr(cfg, "padding", 4),
            use_batchnorm=False,
        )

        self.loss_fn = VAELoss(beta=getattr(cfg, "beta", 0.01))
        self.example_input_array = torch.zeros(2, getattr(cfg, "input_len", 128))

    def _unpack_batch(self, batch: Any) -> torch.Tensor:
        return batch[0] if isinstance(batch, (tuple, list)) else batch

    def _assert_finite(self, name: str, t: torch.Tensor) -> None:
        if not torch.isfinite(t).all():
            raise RuntimeError(f"[NaN/Inf] {name} became non-finite")

    def step(self, batch: Any) -> Dict[str, torch.Tensor]:
        x = self._unpack_batch(batch).float()
        self._assert_finite("x", x)

        recon, mu, logvar = self.model(x)
        self._assert_finite("recon", recon)
        self._assert_finite("mu", mu)
        self._assert_finite("logvar", logvar)

        total, metrics = self.loss_fn(x, recon, mu, logvar)
        self._assert_finite("loss", total)

        return {
            "loss": total,
            "recon": metrics["recon"],
            "kl": metrics["kl"],
        }

    def training_step(self, batch, batch_idx):
        out = self.step(batch)

        # batchwise
        self.log("train/loss_step", out["loss"], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/recon_step", out["recon"], on_step=True, on_epoch=False)
        self.log("train/kl_step", out["kl"], on_step=True, on_epoch=False)

        # epochwise
        self.log("train/loss_epoch", out["loss"], on_step=False, on_epoch=True)
        self.log("train/recon_epoch", out["recon"], on_step=False, on_epoch=True)
        self.log("train/kl_epoch", out["kl"], on_step=False, on_epoch=True)

        return out["loss"]


    def validation_step(self, batch, batch_idx):
        out = self.step(batch)

        # batchwise
        self.log("val/loss_step", out["loss"], on_step=True, on_epoch=False, prog_bar=True)
        self.log("val/recon_step", out["recon"], on_step=True, on_epoch=False)
        self.log("val/kl_step", out["kl"], on_step=True, on_epoch=False)

        # epochwise
        self.log("val/loss_epoch", out["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon_epoch", out["recon"], on_step=False, on_epoch=True)
        self.log("val/kl_epoch", out["kl"], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        lr = float(getattr(self.cfg, "lr", 1e-3))
        wd = float(getattr(self.cfg, "weight_decay", 0.0))
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def forward(self, x):
        return self.model(x)