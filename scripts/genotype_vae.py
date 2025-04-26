#!/usr/bin/env python
"""
LD-aware VAE  ·  genotype → z → reconstruction + phenotype
───────────────────────────────────────────────────────────
• Reconstruction head : CE over 0/1/2 genotypes
• Phenotype     head : MSE
• Cohort (adv) head  : CE with gradient-reversal
  → use λ_adv > 0 and *subtract*  λ_adv · adv_loss in total loss
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt


# ───────────────────────── Dataset ─────────────────────────
class GenoPhenoDataset(Dataset):
    def __init__(self, geno_npy: str, meta_csv: str,
                 train: bool, test_frac: float = 0.2, seed: int = 0):
        G = np.load(geno_npy).astype(np.int64)        # (N,S)
        meta = pd.read_csv(meta_csv)
        idx = np.arange(len(meta))
        np.random.default_rng(seed).shuffle(idx)
        split = int(len(idx) * (1 - test_frac))

        self.mask = idx[:split] if train else idx[split:]
        self.G = torch.from_numpy(G[self.mask])
        self.pheno = torch.from_numpy(
            meta.loc[self.mask, "phenotype"].values.astype(np.float32))
        self.cohort = torch.from_numpy(
            (meta.loc[self.mask, "population"] == "EUR").astype(np.int64).values)

    def __len__(self) -> int:
        return len(self.mask)

    def __getitem__(self, i):
        return {
            "geno":   self.G[i],
            "pheno":  self.pheno[i],
            "cohort": self.cohort[i],
        }


# ────────────── Gradient-reversal layer ───────────────
class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lam * grad_out, None


# ────────────────────────── VAE ─────────────────────────
class LDVAE(pl.LightningModule):
    def __init__(self, S: int, H: int, Z: int, lam_adv: float, lr: float):
        super().__init__()
        self.save_hyperparameters()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(S, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU()
        )
        self.mu      = nn.Linear(H, Z)
        self.logvar  = nn.Linear(H, Z)

        # decoder & heads
        self.decoder      = nn.Sequential(
            nn.Linear(Z, H), nn.ReLU(),
            nn.Linear(H, S * 3)     # logits for 3 genotype states
        )
        self.pheno_head   = nn.Linear(Z, 1)
        self.cohort_head  = nn.Linear(Z, 2)

        # misc
        self.lam_adv = lam_adv
        self.lr      = lr
        self.ce      = nn.CrossEntropyLoss()
        self.mse     = nn.MSELoss()

    # helpers ------------------------------------------------
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # forward ------------------------------------------------
    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterise(mu, lv)
        recon  = self.decoder(z).view(x.size(0), -1, 3)
        pheno  = self.pheno_head(z).squeeze(1)
        cohort = self.cohort_head(GradRev.apply(z, self.lam_adv))
        return recon, pheno, cohort, mu, lv

    # shared step -------------------------------------------
    def _step(self, batch, stage: str):
        xi = batch["geno"]          # long
        x  = xi.float()             # float input
        y, pop = batch["pheno"], batch["cohort"]

        recon, y_hat, pop_logits, mu, lv = self(x)

        recon_loss = self.ce(recon.view(-1, 3), xi.view(-1))
        mse_loss   = self.mse(y_hat, y)
        adv_loss   = self.ce(pop_logits, pop)
        kld        = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / len(x)
        loss       = recon_loss + 10*mse_loss + kld - self.lam_adv * adv_loss

        self.log_dict(
            {f"{stage}_{k}": v for k, v in dict(
                total=loss, recon=recon_loss, mse=mse_loss,
                adv=adv_loss, kld=kld).items()},
            prog_bar=(stage == "train"),
            batch_size=len(x)
        )
        return {"loss": loss, "pred": y_hat.detach(), "true": y.detach()}

    def training_step  (self, b, _): return self._step(b, "train")
    def validation_step(self, b, _): return self._step(b, "val")
    def test_step      (self, b, _): return self._step(b, "test")

    # collect preds → scatter
    def on_test_epoch_start(self):
        self._preds, self._trues = [], []

    def on_test_batch_end(self, outputs, *args, **kwargs):
        self._preds.append(outputs["pred"])
        self._trues.append(outputs["true"])

    def on_test_epoch_end(self):
        preds = torch.cat(self._preds)
        trues = torch.cat(self._trues)
        r = np.corrcoef(preds.cpu(), trues.cpu())[0, 1]

        Path("results/vae").mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.scatter(preds.cpu(), trues.cpu(), alpha=.4)
        plt.xlabel("Predicted")
        plt.ylabel("True phenotype")
        plt.title(f"Test r = {r:.3f}")
        plt.tight_layout()
        plt.savefig("results/vae/pheno_scatter.png")
        plt.close()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ──────────── Callback to draw loss curves ───────────────
class LossPlotter(pl.Callback):
    def __init__(self):
        self.tr_recon, self.va_recon = [], []
        self.tr_mse,   self.va_mse   = [], []

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        if "train_recon" in m: self.tr_recon.append(m["train_recon"].item())
        if "val_recon"   in m: self.va_recon.append(m["val_recon"].item())
        if "train_mse"   in m: self.tr_mse  .append(m["train_mse" ].item())
        if "val_mse"     in m: self.va_mse  .append(m["val_mse"  ].item())

    def on_train_end(self, trainer, pl_module):
        Path("results/vae").mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        for y, lab, ls in zip(
            [self.tr_recon, self.va_recon, self.tr_mse, self.va_mse],
            ["train_recon", "val_recon", "train_mse", "val_mse"],
            ["-", "--", "-", "--"]
        ):
            if y: plt.plot(y, ls, label=lab)
        plt.yscale("log")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig("results/vae/loss_curves.png")
        plt.close()

# ─────────────────────────── main ─────────────────────────
def main(a):
    S = np.load(a.genotype).shape[1]

    ds_train = GenoPhenoDataset(a.genotype, a.meta, train=True)
    ds_test  = GenoPhenoDataset(a.genotype, a.meta, train=False)

    dl_train = DataLoader(ds_train, batch_size=a.batch,
                          shuffle=True, num_workers=4)
    dl_val   = DataLoader(ds_test,  batch_size=a.batch, num_workers=4)

    model = LDVAE(S, a.hidden, a.latent, a.lambda_adv, a.lr)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath="results/vae",
        filename="vae_epoch{epoch}",
        save_top_k=-1
    )
    trainer = pl.Trainer(
        max_epochs=a.epochs,
        callbacks=[checkpoint_cb, LossPlotter()],
        log_every_n_steps=1,
        devices=1,
        accelerator="auto"
    )

    trainer.fit(model, dl_train, dl_val)
    trainer.test(model, dl_val)

    # overwrite final ckpt exactly where Snakemake expects it
    trainer.save_checkpoint(f"results/vae/vae_epoch{a.epochs-1}.ckpt",
                            weights_only=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--genotype", required=True)
    ap.add_argument("--meta",     required=True)
    ap.add_argument("--latent",   type=int, default=32)
    ap.add_argument("--hidden",   type=int, default=256)
    ap.add_argument("--batch",    type=int, default=64)
    ap.add_argument("--epochs",   type=int, default=50)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--lambda_adv", type=float, default=1.0)
    args = ap.parse_args()
    main(args)
