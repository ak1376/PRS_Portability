# src/vae/diagnostics_callback.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl

from src.masking import make_mask_and_apply


class ReconDiagnosticsCallback(pl.Callback):
    def __init__(self, X_val_diag, outdir, mask_cfg, seed=0):
        super().__init__()
        self.X_val_diag = np.asarray(X_val_diag, dtype=np.float32)
        self.outdir = Path(outdir)
        self.mask_cfg = dict(mask_cfg)
        self.seed = int(seed)

        self.epoch_dir = self.outdir / "epoch_recon"
        self.epoch_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.X_val_diag.size == 0:
            return

        device = pl_module.device
        epoch = int(trainer.current_epoch)

        x_true = torch.from_numpy(self.X_val_diag).to(device=device, dtype=torch.float32)

        x_masked, mask, _ = make_mask_and_apply(
            x_true,
            enabled=self.mask_cfg.get("enabled", False),
            n_blocks=self.mask_cfg.get("n_blocks", 1),
            block_len=self.mask_cfg.get("block_len"),
            mask_frac=self.mask_cfg.get("mask_frac"),
            allow_overlap=self.mask_cfg.get("allow_overlap", True),
            seed=self.seed,  # fixed every epoch
            fill=self.mask_cfg.get("fill", "zero"),
            gaussian_std=self.mask_cfg.get("gaussian_std", 0.1),
            constant_value=self.mask_cfg.get("constant_value", 0.0),
        )

        out = pl_module.model(x_masked)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)

        x_true_np = x_true.cpu().numpy().astype(np.int64)
        x_masked_np = x_masked.cpu().numpy().astype(np.float32)
        mask_np = mask.cpu().numpy().astype(bool)
        pred_np = pred.cpu().numpy().astype(np.int64)
        probs_np = probs.cpu().numpy().astype(np.float32)

        np.savez_compressed(
            self.epoch_dir / f"epoch_{epoch:03d}_val_recon.npz",
            x_true=x_true_np,
            x_masked=x_masked_np,
            mask=mask_np,
            pred=pred_np,
            prob_0=probs_np[:, 0, :],
            prob_1=probs_np[:, 1, :],
            prob_2=probs_np[:, 2, :],
        )