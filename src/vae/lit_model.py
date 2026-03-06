# src/vae/lit_model.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.vae.model import ConvVAE1D, FullyConvVAE1D
from src.vae.loss import mse_masked
from src.masking import make_mask_and_apply


class LitVAE(pl.LightningModule):
    """
    Two-pass metrics:
      (1) clean pass: recon_clean = model(x); mse_clean_all = MSE(recon_clean, x) over all positions
      (2) masked/corrupted pass (optional): x_in, mask = corrupt(x); recon = model(x_in)
          - mse_masked   = MSE on masked positions only
          - mse_unmasked = MSE on unmasked positions only
          - recon_objective = alpha*mse_masked + (1-alpha)*mse_unmasked
          - total_loss = recon_objective + beta*KL

    Logging (Option A):
      - train_step/* logged per step
      - train/* logged once per epoch (manual mean)
      - val/* and target/* logged once per epoch (manual mean)
      - add_dataloader_idx=False everywhere; val vs target distinguished by prefix
    """

    def __init__(self, cfg: Any):
        super().__init__()

        # Save hyperparams
        if is_dataclass(cfg):
            self.save_hyperparameters(asdict(cfg))
        elif isinstance(cfg, dict):
            self.save_hyperparameters(cfg)
        else:
            self.save_hyperparameters({"cfg": str(cfg)})

        self.cfg = cfg

        # Choose model type: "conv" (original) or "fully_conv" (no linear bottleneck)
        model_type = getattr(cfg, "model_type", "conv")
        
        if model_type == "fully_conv":
            self.model = FullyConvVAE1D(
                input_len=cfg.input_len,
                latent_dim=getattr(cfg, "latent_dim", 32),  # interpreted as latent_channels
                hidden_channels=getattr(cfg, "hidden_channels", (32, 64, 128)),
                kernel_size=getattr(cfg, "kernel_size", 33),
                stride=getattr(cfg, "stride", 4),
                padding=getattr(cfg, "padding", None),
                use_batchnorm=getattr(cfg, "use_batchnorm", True),
            )
        else:
            self.model = ConvVAE1D(
                input_len=cfg.input_len,
                latent_dim=getattr(cfg, "latent_dim", 32),
                hidden_channels=getattr(cfg, "hidden_channels", (32, 64, 128)),
                kernel_size=getattr(cfg, "kernel_size", 9),
                stride=getattr(cfg, "stride", 2),
                padding=getattr(cfg, "padding", 4),
                use_batchnorm=getattr(cfg, "use_batchnorm", False),
            )

        self.example_input_array = torch.zeros(2, getattr(cfg, "input_len", 128))

        # Option A buffers
        self._train_buf: Dict[str, List[torch.Tensor]] = {}
        self._val_buf: Dict[str, List[torch.Tensor]] = {}
        self._target_buf: Dict[str, List[torch.Tensor]] = {}

    # -------------------------
    # helpers
    # -------------------------
    def _unpack_batch(self, batch: Any) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return x.float()

    def _to_2d(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(1) if x.dim() == 3 else x

    def _assert_finite(self, name: str, t: torch.Tensor) -> None:
        if not torch.isfinite(t).all():
            raise RuntimeError(f"[NaN/Inf] {name} became non-finite")

    @staticmethod
    def _mse_all(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, x, reduction="mean")

    def _kl_standard_normal(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence from standard normal.
        Works with both 2D (B, Z) and 3D (B, C, L) tensors.
        """
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        kl_per = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        # Flatten all dims except batch and sum
        kl_per_sample = kl_per.view(kl_per.shape[0], -1).sum(dim=1)
        return kl_per_sample.mean()

    def _training_cfg(self) -> Any:
        return getattr(self.cfg, "training", None)

    def _get_beta(self) -> float:
        t = self._training_cfg()
        return float(getattr(t, "beta", getattr(self.cfg, "beta", 0.01)))

    def _get_lr(self) -> float:
        t = self._training_cfg()
        return float(getattr(t, "lr", getattr(self.cfg, "lr", 1e-3)))

    def _get_wd(self) -> float:
        t = self._training_cfg()
        return float(getattr(t, "weight_decay", getattr(self.cfg, "weight_decay", 0.0)))

    def _mask_cfg(self) -> Dict[str, Any]:
        """
        Reads nested cfg.masking.* if present, else falls back to legacy flat keys.
        """
        m = getattr(self.cfg, "masking", None)
        if m is not None:
            return {
                "enabled": bool(getattr(m, "enabled", False)),
                "alpha_masked": float(getattr(m, "alpha_masked", 1.0)),
                "n_blocks": int(getattr(m, "n_blocks", 1)),
                "block_len": getattr(m, "block_len", None),
                "mask_frac": getattr(m, "mask_frac", None),
                "allow_overlap": bool(getattr(m, "allow_overlap", True)),
                "fill": str(getattr(m, "fill", getattr(m, "fill_value", "zero"))),
                "gaussian_std": float(getattr(m, "gaussian_std", 0.1)),
                "constant_value": float(getattr(m, "constant_value", 0.0)),
            }

        # legacy fallback
        return {
            "enabled": bool(getattr(self.cfg, "mask_enabled", False)),
            "alpha_masked": float(getattr(self.cfg, "alpha_masked", 1.0)),
            "n_blocks": 1,
            "block_len": int(getattr(self.cfg, "mask_block_len", 0)) or None,
            "mask_frac": None,
            "allow_overlap": True,
            "fill": str(getattr(self.cfg, "mask_fill_value", "zero")),
            "gaussian_std": float(getattr(self.cfg, "mask_gaussian_std", 0.1)),
            "constant_value": 0.0,
        }

    def _make_step_seed(self, stage: str, batch_idx: int) -> int:
        base = int(getattr(self.cfg, "seed", 0))
        stage_salt = 0 if stage == "train" else (1 if stage == "val" else 2)

        if stage in {"val", "target"}:
            # fixed masks across epochs -> stable validation curves
            return base + 1_000_000 * stage_salt + int(batch_idx)

        # training keeps changing each epoch
        return base + 1_000_000 * stage_salt + 10_000 * int(self.current_epoch) + int(batch_idx)

    def _forward(self, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon, mu, logvar = self.model(x_in)
        return recon, mu, logvar

    @staticmethod
    def _buf_append(buf: Dict[str, List[torch.Tensor]], out: Dict[str, torch.Tensor], keys: List[str]) -> None:
        for k in keys:
            buf.setdefault(k, []).append(out[k].detach())

    def _flush_buf(self, buf: Dict[str, List[torch.Tensor]], prefix: str) -> None:
        for k, vs in buf.items():
            if len(vs) == 0:
                continue
            mean_v = torch.stack(vs).mean()
            self.log(
                f"{prefix}/{k}",
                mean_v,
                on_step=False,
                on_epoch=True,
                prog_bar=(prefix == "val" and k == "loss"),
                logger=True,
                add_dataloader_idx=False,
                sync_dist=False,
            )
        buf.clear()

    # -------------------------
    # core step
    # -------------------------
    def _shared_step(self, batch: Any, batch_idx: int, stage: str) -> Dict[str, torch.Tensor]:
        x = self._unpack_batch(batch).to(self.device)
        x = self._to_2d(x)
        self._assert_finite("x", x)

        B, L = x.shape
        beta = self._get_beta()
        eps = 1e-12

        # (1) clean pass
        recon_clean, mu_clean, logvar_clean = self._forward(x)
        recon_clean = self._to_2d(recon_clean)
        self._assert_finite("recon_clean", recon_clean)

        mse_clean_all = self._mse_all(x, recon_clean)
        kl_clean = self._kl_standard_normal(mu_clean, logvar_clean)

        # (2) masked/corrupted pass
        mcfg = self._mask_cfg()
        seed = self._make_step_seed(stage, batch_idx)

        if mcfg["enabled"] and (mcfg["block_len"] is not None or mcfg["mask_frac"] is not None):
            x_in, mask, used_block_len = make_mask_and_apply(
                x,
                enabled=True,
                n_blocks=int(mcfg["n_blocks"]),
                block_len=mcfg["block_len"],
                mask_frac=mcfg["mask_frac"],
                allow_overlap=bool(mcfg["allow_overlap"]),
                seed=int(seed),
                fill=str(mcfg["fill"]),
                gaussian_std=float(mcfg["gaussian_std"]),
                constant_value=float(mcfg["constant_value"]),
            )
        else:
            x_in = x
            mask = torch.zeros((B, L), dtype=torch.bool, device=self.device)
            used_block_len = 0

        self._assert_finite("x_in", x_in)

        recon, mu, logvar = self._forward(x_in)
        recon = self._to_2d(recon)
        self._assert_finite("recon", recon)

        kl = self._kl_standard_normal(mu, logvar)

        mse_corrupt_all = self._mse_all(x, recon)

        if mask.any():
            mse_mask = mse_masked(x, recon, mask)
            mse_unmask = mse_masked(x, recon, ~mask)
        else:
            mse_mask = torch.tensor(0.0, device=self.device)
            mse_unmask = mse_corrupt_all

        alpha = float(mcfg["alpha_masked"])
        # make alpha safe
        alpha = max(0.0, min(1.0, alpha))

        recon_objective = alpha * mse_mask + (1.0 - alpha) * mse_unmask
        loss = recon_objective + beta * kl
        self._assert_finite("loss", loss)

        # debug
        mask_frac = mask.float().mean()
        delta_in_l1 = (x_in - x).abs().mean()

        ratio_masked_over_clean = (mse_mask / (mse_clean_all + eps)) if mask.any() else torch.tensor(0.0, device=self.device)

        return {
            "loss": loss,
            "kl": kl,
            "recon_objective": recon_objective,
            "mse_corrupt_all": mse_corrupt_all,
            "mse_masked": mse_mask,
            "mse_unmasked": mse_unmask,
            "mse_clean_all": mse_clean_all,
            "kl_clean": kl_clean,
            "ratio_masked_over_clean": ratio_masked_over_clean,
            "mask_frac": mask_frac,
            "delta_in_l1": delta_in_l1,
            "used_block_len": torch.tensor(float(used_block_len), device=self.device),
            "x_max": x.max(),
            "x_in_max": x_in.max(),
        }

    # -------------------------
    # Lightning hooks
    # -------------------------
    def on_train_epoch_start(self) -> None:
        self._train_buf.clear()

    def on_validation_epoch_start(self) -> None:
        self._val_buf.clear()
        self._target_buf.clear()

    def training_step(self, batch: Any, batch_idx: int):
        out = self._shared_step(batch, batch_idx=batch_idx, stage="train")

        # train_step/* logs every step (no epoch)
        step_keys = [
            "loss", "kl", "recon_objective",
            "mse_corrupt_all", "mse_masked", "mse_unmasked",
            "mse_clean_all", "ratio_masked_over_clean",
        ]
        for k in step_keys:
            self.log(
                f"train_step/{k}",
                out[k],
                on_step=True,
                on_epoch=False,
                prog_bar=(k == "loss"),
                logger=True,
                add_dataloader_idx=False,
            )

        # debug step logs
        for k in ["mask_frac", "delta_in_l1", "used_block_len", "x_max", "x_in_max"]:
            self.log(
                f"debug/{k}",
                out[k],
                on_step=True,
                on_epoch=False,
                logger=True,
                add_dataloader_idx=False,
            )

        # accumulate for epoch means
        self._buf_append(
            self._train_buf,
            out,
            keys=step_keys + ["mask_frac", "delta_in_l1"],
        )

        return out["loss"]

    def on_train_epoch_end(self) -> None:
        self._flush_buf(self._train_buf, prefix="train")

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        prefix = "val" if dataloader_idx == 0 else "target"
        stage = prefix  # seed salt

        out = self._shared_step(batch, batch_idx=batch_idx, stage=stage)

        keys = [
            "loss", "kl", "recon_objective",
            "mse_corrupt_all", "mse_masked", "mse_unmasked",
            "mse_clean_all", "ratio_masked_over_clean",
            "mask_frac", "delta_in_l1",
        ]
        buf = self._val_buf if prefix == "val" else self._target_buf
        self._buf_append(buf, out, keys=keys)

        return out["loss"]

    def on_validation_epoch_end(self) -> None:
        self._flush_buf(self._val_buf, prefix="val")
        self._flush_buf(self._target_buf, prefix="target")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._get_lr(), weight_decay=self._get_wd())

    def forward(self, x):
        return self.model(x)