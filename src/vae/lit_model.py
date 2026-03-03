# src/vae/lit_model.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.vae.model import ConvVAE1D
from src.vae.loss import VAELoss, mse_masked


class LitVAE(pl.LightningModule):
    """
    Masked-inpainting VAE with *explicit* logging for:
      - recon_masked (MSE on masked positions, evaluated vs original x)
      - recon_unmasked (MSE on unmasked positions, evaluated vs original x)
      - recon_weighted = w_m*recon_masked + w_u*recon_unmasked (the objective recon term)
      - recon_nomask_all (baseline MSE when you feed x directly, no corruption)
      - ratio_masked_over_nomask = recon_masked / recon_nomask_all  (best sanity plot)

    This makes it easy to compare "masked training" vs "normal reconstruction".
    """

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
            use_batchnorm=getattr(cfg, "use_batchnorm", False),
        )

        self.loss_fn = VAELoss(beta=float(getattr(cfg, "beta", 0.01)))
        self.example_input_array = torch.zeros(2, getattr(cfg, "input_len", 128))

    # -------------------------
    # utilities
    # -------------------------
    def _unpack_batch(self, batch: Any) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return x.float()

    def _mask_enabled(self) -> bool:
        return bool(getattr(self.cfg, "mask_enabled", False)) and int(getattr(self.cfg, "mask_block_len", 0)) > 0

    def _make_contiguous_mask(self, B: int, L: int, *, seed: int) -> torch.Tensor:
        """
        Bool mask (B,L) with one contiguous True block per sample.
        Deterministic given seed.
        """
        block_len = int(getattr(self.cfg, "mask_block_len", 0))
        block_len = max(0, min(block_len, L))
        if block_len == 0:
            return torch.zeros((B, L), dtype=torch.bool, device=self.device)

        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        starts = torch.randint(low=0, high=L - block_len + 1, size=(B,), generator=g, device=self.device)

        ar = torch.arange(L, device=self.device).view(1, -1)
        mask = (ar >= starts.view(-1, 1)) & (ar < (starts + block_len).view(-1, 1))
        return mask

    def _apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B,L) float
        mask: (B,L) bool
        """
        fill = str(getattr(self.cfg, "mask_fill_value", "mean")).lower()

        if fill == "zero":
            fill_value = torch.zeros_like(x)

        elif fill == "mean":
            # per-SNP batch mean (detached)
            fill_value = x.mean(dim=0, keepdim=True).expand_as(x).detach()

        elif fill in ("random_af", "rand_af", "random"):
            # sample corrupted genotypes from per-SNP allele frequency estimated from the batch
            # assumes x is on {0,1,2} scale
            p = (x.mean(dim=0).clamp(0.0, 2.0) / 2.0).detach()  # (L,)
            b1 = torch.bernoulli(p.expand(x.size(0), -1))
            b2 = torch.bernoulli(p.expand(x.size(0), -1))
            fill_value = (b1 + b2).to(x.dtype)

        else:
            raise ValueError(f"Unknown mask_fill_value={fill!r}. Use 'zero', 'mean', or 'random_af'.")

        xm = x.clone()
        xm[mask] = fill_value[mask]
        return xm

    def _mse_all(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        if recon.dim() == 3:
            recon = recon.squeeze(1)
        return F.mse_loss(recon, x, reduction="mean")

    def _assert_finite(self, name: str, t: torch.Tensor) -> None:
        if not torch.isfinite(t).all():
            raise RuntimeError(f"[NaN/Inf] {name} became non-finite")

    # -------------------------
    # core forward+loss
    # -------------------------
    def _forward_vae(self, x_in: torch.Tensor):
        recon, mu, logvar = self.model(x_in)
        return recon, mu, logvar

    def _compute_metrics_one_pass(
        self,
        *,
        x: torch.Tensor,
        x_in: torch.Tensor,
        mask: Optional[torch.Tensor],
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes recon_masked/unmasked (if mask is not None),
        plus weighted recon objective and total loss (with KL).
        """
        if x.dim() == 3:
            x0 = x.squeeze(1)
        else:
            x0 = x
        if recon.dim() == 3:
            r0 = recon.squeeze(1)
        else:
            r0 = recon

        # KL
        _, base = self.loss_fn(x0, r0, mu, logvar)
        kl = base["kl"]

        if mask is None:
            recon_all = self._mse_all(x0, r0)
            recon_masked = torch.tensor(0.0, device=self.device)
            recon_unmasked = recon_all
            recon_weighted = recon_all
        else:
            m = mask.bool()
            recon_masked = mse_masked(x0, r0, m)
            recon_unmasked = mse_masked(x0, r0, ~m)

            w_m = float(getattr(self.cfg, "weight_masked", 1.0))
            w_u = float(getattr(self.cfg, "weight_unmasked", 0.0))
            recon_weighted = w_m * recon_masked + w_u * recon_unmasked
            recon_all = self._mse_all(x0, r0)

        beta = float(getattr(self.cfg, "beta", 0.01))
        loss = recon_weighted + beta * kl

        return {
            "loss": loss,
            "kl": kl,
            "recon_all": recon_all,              # MSE over all sites for THIS pass
            "recon_masked": recon_masked,        # MSE only on masked sites (0 if no mask)
            "recon_unmasked": recon_unmasked,    # MSE only on unmasked sites
            "recon_weighted": recon_weighted,    # the recon term used in objective
        }

    def _shared_step(self, batch: Any, batch_idx: int, stage: str) -> Dict[str, torch.Tensor]:
        x = self._unpack_batch(batch).to(self.device)
        self._assert_finite("x", x)

        # ensure 2D (B,L)
        if x.dim() == 3:
            x = x.squeeze(1)

        B, L = x.shape

        # --- masked pass ---
        mask = None
        x_in = x
        if self._mask_enabled():
            # deterministic-but-changing masks across steps
            base = int(getattr(self.cfg, "seed", 0))
            # stage salt makes train/val masks differ but deterministic
            stage_salt = 0 if stage == "train" else (1 if stage == "val" else 2)
            seed = base + 1_000_000 * stage_salt + 10_000 * int(self.current_epoch) + int(batch_idx)
            mask = self._make_contiguous_mask(B, L, seed=seed)
            x_in = self._apply_mask(x, mask)

        recon, mu, logvar = self._forward_vae(x_in)
        self._assert_finite("recon(masked-pass)", recon)
        self._assert_finite("mu", mu)
        self._assert_finite("logvar", logvar)

        masked_metrics = self._compute_metrics_one_pass(
            x=x, x_in=x_in, mask=mask, recon=recon, mu=mu, logvar=logvar
        )

        # --- no-mask baseline pass (for clean comparison) ---
        recon_nm, mu_nm, logvar_nm = self._forward_vae(x)
        nomask_all = self._mse_all(x, recon_nm)

        # --- debug: prove mask applied ---
        if mask is None:
            mask_frac = torch.tensor(0.0, device=self.device)
            mask_delta_l1 = torch.tensor(0.0, device=self.device)
        else:
            mask_frac = mask.float().mean()
            # how much did inputs change?
            mask_delta_l1 = (x_in - x).abs().mean()

        out = dict(masked_metrics)
        out.update(
            {
                "recon_nomask_all": nomask_all,
                "ratio_masked_over_nomask": (
                    masked_metrics["recon_masked"] / (nomask_all + 1e-12)
                    if mask is not None
                    else torch.tensor(0.0, device=self.device)
                ),
                "debug_mask_frac": mask_frac,
                "debug_mask_delta_l1": mask_delta_l1,
                "debug_x_max": x.max(),
                "debug_xin_max": x_in.max(),
            }
        )

        self._assert_finite("loss", out["loss"])
        return out

    # -------------------------
    # lightning hooks
    # -------------------------
    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch, batch_idx=batch_idx, stage="train")

        self.log("train/loss", out["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/kl", out["kl"], on_step=True, on_epoch=True)
        self.log("train/recon_weighted", out["recon_weighted"], on_step=True, on_epoch=True)
        self.log("train/recon_all", out["recon_all"], on_step=True, on_epoch=True)
        self.log("train/recon_masked", out["recon_masked"], on_step=True, on_epoch=True)
        self.log("train/recon_unmasked", out["recon_unmasked"], on_step=True, on_epoch=True)

        # baseline + ratio (THIS is the plot you want)
        self.log("train/recon_nomask_all", out["recon_nomask_all"], on_step=True, on_epoch=True)
        self.log("train/ratio_masked_over_nomask", out["ratio_masked_over_nomask"], on_step=True, on_epoch=True)

        # debugging proof
        self.log("debug/mask_frac", out["debug_mask_frac"], on_step=True, on_epoch=True)
        self.log("debug/mask_delta_l1", out["debug_mask_delta_l1"], on_step=True, on_epoch=True)
        self.log("debug/x_max", out["debug_x_max"], on_step=True, on_epoch=True)
        self.log("debug/xin_max", out["debug_xin_max"], on_step=True, on_epoch=True)

        return out["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        prefix = "val" if dataloader_idx == 0 else "target"
        stage = "val" if dataloader_idx == 0 else "target"
        out = self._shared_step(batch, batch_idx=batch_idx, stage=stage)

        self.log(f"{prefix}/loss", out["loss"], on_step=True, on_epoch=True, prog_bar=(dataloader_idx == 0))
        self.log(f"{prefix}/kl", out["kl"], on_step=True, on_epoch=True)
        self.log(f"{prefix}/recon_weighted", out["recon_weighted"], on_step=True, on_epoch=True)
        self.log(f"{prefix}/recon_all", out["recon_all"], on_step=True, on_epoch=True)
        self.log(f"{prefix}/recon_masked", out["recon_masked"], on_step=True, on_epoch=True)
        self.log(f"{prefix}/recon_unmasked", out["recon_unmasked"], on_step=True, on_epoch=True)

        self.log(f"{prefix}/recon_nomask_all", out["recon_nomask_all"], on_step=True, on_epoch=True)
        self.log(f"{prefix}/ratio_masked_over_nomask", out["ratio_masked_over_nomask"], on_step=True, on_epoch=True)

        # same debug metrics, but namespaced (so they appear with dataloader_idx)
        self.log("debug/mask_frac", out["debug_mask_frac"], on_step=True, on_epoch=True)
        self.log("debug/mask_delta_l1", out["debug_mask_delta_l1"], on_step=True, on_epoch=True)
        self.log("debug/x_max", out["debug_x_max"], on_step=True, on_epoch=True)
        self.log("debug/xin_max", out["debug_xin_max"], on_step=True, on_epoch=True)

        return out["loss"]

    def configure_optimizers(self):
        lr = float(getattr(self.cfg, "lr", 1e-3))
        wd = float(getattr(self.cfg, "weight_decay", 0.0))
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def forward(self, x):
        return self.model(x)