from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.vae.model import ConvVAE1D, FullyConvVAE1D
from src.vae.loss import cross_entropy_masked
from src.masking import make_mask_and_apply


class LitVAE(pl.LightningModule):
    """
    Classification version of LitVAE.

    Expected model behavior:
      - input: x_in with shape (B, L) or (B, C, L)
      - output: logits with shape (B, 3, L)
      - targets: integer genotype classes in {0,1,2}

    Two-pass metrics:
      (1) clean pass: logits_clean = model(x_clean_in)
          - ce_clean_all = cross-entropy over all positions
      (2) masked/corrupted pass: x_in, mask = corrupt(x); logits = model(model_in)
          - ce_masked   = CE on masked positions only
          - ce_unmasked = CE on unmasked positions only
          - recon_objective = alpha * ce_masked + (1-alpha) * ce_unmasked
          - total_loss = recon_objective + beta * KL

    If use_mask_channel=True, model input is (B, 2, L):
      - channel 0: genotype/corrupted values
      - channel 1: binary mask indicator
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

        model_type = getattr(cfg, "model_type", "conv")
        in_channels = 2 if self._use_mask_channel() else 1

        if model_type == "fully_conv":
            self.model = FullyConvVAE1D(
                input_len=cfg.input_len,
                latent_dim=getattr(cfg, "latent_dim", 32),
                hidden_channels=getattr(cfg, "hidden_channels", (32, 64, 128)),
                kernel_size=getattr(cfg, "kernel_size", 33),
                stride=getattr(cfg, "stride", 4),
                padding=getattr(cfg, "padding", None),
                use_batchnorm=getattr(cfg, "use_batchnorm", True),
                in_channels=in_channels,
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
                in_channels=in_channels,
            )

        if self._use_mask_channel():
            self.example_input_array = torch.zeros(2, 2, getattr(cfg, "input_len", 128))
        else:
            self.example_input_array = torch.zeros(2, getattr(cfg, "input_len", 128))

        self._train_buf: Dict[str, List[torch.Tensor]] = {}
        self._val_buf: Dict[str, List[torch.Tensor]] = {}
        self._target_buf: Dict[str, List[torch.Tensor]] = {}

        # Optional class weights for genotype classes {0,1,2}.
        self.class_weights: Optional[torch.Tensor] = None

    # -------------------------
    # helpers
    # -------------------------
    def _unpack_batch(self, batch: Any) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return x.float()

    def _to_2d(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(1) if x.dim() == 3 and x.shape[1] == 1 else x

    def _assert_finite(self, name: str, t: torch.Tensor) -> None:
        if not torch.isfinite(t).all():
            raise RuntimeError(f"[NaN/Inf] {name} became non-finite")

    def _assert_valid_targets(self, target: torch.Tensor) -> None:
        bad = ~((target == 0) | (target == 1) | (target == 2))
        if bad.any():
            vals = torch.unique(target[bad]).detach().cpu().tolist()
            raise RuntimeError(f"Targets must be in {{0,1,2}}. Found invalid values: {vals[:10]}")

    def set_class_weights(self, class_weights: torch.Tensor | None) -> None:
        """
        Store global class weights for genotype classes {0,1,2}.
        Expected shape: (3,)
        """
        if class_weights is None:
            self.class_weights = None
            return

        if class_weights.numel() != 3:
            raise ValueError(f"class_weights must have shape (3,), got {tuple(class_weights.shape)}")

        self.class_weights = class_weights.detach().float()

    def _get_class_weights(self) -> torch.Tensor | None:
        if self.class_weights is None:
            return None
        return self.class_weights.to(self.device)

    def _use_mask_channel(self) -> bool:
        m = getattr(self.cfg, "masking", None)
        if m is not None:
            return bool(getattr(m, "use_mask_channel", False))
        return bool(getattr(self.cfg, "use_mask_channel", False))

    def _make_model_input(
        self,
        x_values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build actual model input.

        Args:
            x_values: (B, L) float
            mask:     (B, L) bool or float, True = masked

        Returns:
            if use_mask_channel=False: (B, L)
            if use_mask_channel=True:  (B, 2, L)
                channel 0 = genotype/corrupted values
                channel 1 = mask indicator in {0,1}
        """
        if not self._use_mask_channel():
            return x_values

        if mask is None:
            mask = torch.zeros_like(x_values, dtype=torch.bool)

        mask_f = mask.to(dtype=x_values.dtype)
        return torch.stack([x_values, mask_f], dim=1)

    @staticmethod
    def _ce_all(
        logits: torch.Tensor,
        target: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        logits: (B, 3, L)
        target: (B, L)
        """
        if class_weights is not None:
            class_weights = class_weights.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits, target, weight=class_weights, reduction="mean")

    @staticmethod
    def _masked_accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 3, L)
        target: (B, L)
        mask:   (B, L) bool
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)

        pred = logits.argmax(dim=1)  # (B, L)
        return (pred[mask] == target[mask]).float().mean()

    def _kl_standard_normal(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        kl_per = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
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
                "use_mask_channel": bool(getattr(m, "use_mask_channel", False)),
            }

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
            "use_mask_channel": bool(getattr(self.cfg, "use_mask_channel", False)),
        }

    def _make_step_seed(self, stage: str, batch_idx: int) -> int:
        base = int(getattr(self.cfg, "seed", 0))
        stage_salt = 0 if stage == "train" else (1 if stage == "val" else 2)

        if stage in {"val", "target"}:
            return base + 1_000_000 * stage_salt + int(batch_idx)

        return base + 1_000_000 * stage_salt + 10_000 * int(self.current_epoch) + int(batch_idx)

    def _forward(self, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, mu, logvar = self.model(x_in)
        return logits, mu, logvar

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
        x = self._unpack_batch(batch).to(self.device)   # raw 0/1/2 genotypes as float
        x = self._to_2d(x)
        self._assert_finite("x", x)

        target = x.long()
        self._assert_valid_targets(target)

        B, L = x.shape
        beta = self._get_beta()
        eps = 1e-12
        class_weights = self._get_class_weights()

        # (1) clean pass
        clean_mask = torch.zeros((B, L), dtype=torch.bool, device=self.device)
        x_clean_in = self._make_model_input(x, clean_mask)

        logits_clean, mu_clean, logvar_clean = self._forward(x_clean_in)
        self._assert_finite("logits_clean", logits_clean)

        ce_clean_all = self._ce_all(logits_clean, target, class_weights=class_weights)
        acc_clean_all = (logits_clean.argmax(dim=1) == target).float().mean()
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

        model_in = self._make_model_input(x_in, mask)
        self._assert_finite("model_in", model_in)

        logits, mu, logvar = self._forward(model_in)
        self._assert_finite("logits", logits)

        kl = self._kl_standard_normal(mu, logvar)

        ce_corrupt_all = self._ce_all(logits, target, class_weights=class_weights)
        acc_corrupt_all = (logits.argmax(dim=1) == target).float().mean()

        if mask.any():
            ce_mask = cross_entropy_masked(
                logits,
                target,
                mask,
                class_weights=class_weights,
            )
            ce_unmask = cross_entropy_masked(
                logits,
                target,
                ~mask,
                class_weights=class_weights,
            )

            acc_mask = self._masked_accuracy(logits, target, mask)
            acc_unmask = self._masked_accuracy(logits, target, ~mask)
        else:
            ce_mask = torch.tensor(0.0, device=self.device)
            ce_unmask = ce_corrupt_all
            acc_mask = torch.tensor(0.0, device=self.device)
            acc_unmask = acc_corrupt_all

        alpha = float(mcfg["alpha_masked"])
        alpha = max(0.0, min(1.0, alpha))

        if mask.any():
            recon_objective = alpha * ce_mask + (1.0 - alpha) * ce_unmask
        else:
            recon_objective = ce_corrupt_all

        loss = recon_objective + beta * kl
        self._assert_finite("loss", loss)

        mask_frac = mask.float().mean()
        delta_in_l1 = (x_in - x).abs().mean()

        ratio_masked_over_clean = (
            ce_mask / (ce_clean_all + eps)
            if mask.any()
            else torch.tensor(0.0, device=self.device)
        )

        return {
            "loss": loss,
            "kl": kl,
            "recon_objective": recon_objective,
            "ce_corrupt_all": ce_corrupt_all,
            "ce_masked": ce_mask,
            "ce_unmasked": ce_unmask,
            "ce_clean_all": ce_clean_all,
            "acc_corrupt_all": acc_corrupt_all,
            "acc_masked": acc_mask,
            "acc_unmasked": acc_unmask,
            "acc_clean_all": acc_clean_all,
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

        step_keys = [
            "loss", "kl", "recon_objective",
            "ce_corrupt_all", "ce_masked", "ce_unmasked", "ce_clean_all",
            "acc_corrupt_all", "acc_masked", "acc_unmasked", "acc_clean_all",
            "ratio_masked_over_clean",
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

        for k in ["mask_frac", "delta_in_l1", "used_block_len", "x_max", "x_in_max"]:
            self.log(
                f"debug/{k}",
                out[k],
                on_step=True,
                on_epoch=False,
                logger=True,
                add_dataloader_idx=False,
            )

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
        stage = prefix

        out = self._shared_step(batch, batch_idx=batch_idx, stage=stage)

        keys = [
            "loss", "kl", "recon_objective",
            "ce_corrupt_all", "ce_masked", "ce_unmasked", "ce_clean_all",
            "acc_corrupt_all", "acc_masked", "acc_unmasked", "acc_clean_all",
            "ratio_masked_over_clean",
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