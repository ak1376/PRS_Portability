#!/usr/bin/env python
"""
LD-aware VAE: genotype → latent → reconstruction + phenotype + cohort (adv)
----------------------------------------------------------------------------
* CE reconstruction loss over 0/1/2 genotypes
* MSE loss on phenotype
* CE adversarial loss for cohort (EUR vs. non-EUR) via gradient reversal
* Logging via Lightning + custom callback for plotting
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt

# ───────────── Dataset ─────────────
class GenoPhenoDataset(Dataset):
    def __init__(self, geno_npy, meta_csv, train, test_frac=0.2, seed=0):
        G = np.load(geno_npy).astype(np.int64)
        meta = pd.read_csv(meta_csv)
        idx = np.arange(len(meta))
        np.random.default_rng(seed).shuffle(idx)
        split = int(len(idx) * (1 - test_frac))
        self.mask = idx[:split] if train else idx[split:]
        self.G = torch.from_numpy(G[self.mask])
        self.pheno = torch.from_numpy(meta.loc[self.mask, "phenotype"].values.astype(np.float32))
        self.cohort = torch.from_numpy((meta.loc[self.mask, "population"] == "EUR").astype(np.int64).values)

    def __len__(self): return len(self.mask)
    def __getitem__(self, i): return {
        "geno": self.G[i], "pheno": self.pheno[i], "cohort": self.cohort[i]
    }

# ───────────── GradReversal ─────────────
class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_out): return -ctx.lam * grad_out, None

# ───────────── VAE Module ─────────────
class LDVAE(pl.LightningModule):
    def __init__(self, S, H, Z, lr, lambda_adv, recon_wt, mse_wt, beta_kl, warm_adv):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(nn.Linear(S, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.mu      = nn.Linear(H, Z)
        self.logvar  = nn.Linear(H, Z)
        self.decoder = nn.Sequential(nn.Linear(Z, H), nn.ReLU(), nn.Linear(H, S * 3))
        self.pheno_head  = nn.Linear(Z, 1)
        self.cohort_head = nn.Linear(Z, 2)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, lv):
        return mu + torch.randn_like(lv) * torch.exp(0.5 * lv)

    def _lam(self):
        frac = self.current_epoch / max(1, self.hparams.warm_adv)
        return self.hparams.lambda_adv * min(1.0, frac)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        recon = self.decoder(z).view(x.size(0), -1, 3)
        pheno = self.pheno_head(z).squeeze(1)
        # logits = self.cohort_head(GradRev.apply(z, self._lam()))
        logits = self.cohort_head(z)  # GRL disabled

        return recon, pheno, logits, mu, lv

    def _step(self, batch, stage):
        xi = batch["geno"]
        x = xi.float()
        y, pop = batch["pheno"], batch["cohort"]
        recon, y_hat, pop_logits, mu, lv = self(x)

        R = self.ce(recon.view(-1, 3), xi.view(-1))
        M = self.mse(y_hat, y)
        A = self.ce(pop_logits, pop)
        K = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / len(x)

        loss = (self.hparams.recon_wt * R + self.hparams.mse_wt * M +
                self.hparams.beta_kl * K - self._lam() * A)

        acc = (pop_logits.argmax(1) == pop).float().mean()
        self.log_dict({
            f"{stage}_recon": R,
            f"{stage}_mse": M,
            f"{stage}_adv": A,
            f"{stage}_kld": K,
            f"{stage}_total": loss,
            f"{stage}_pop_acc": acc
        }, batch_size=len(x))
        return {"loss": loss, "pred": y_hat.detach(), "true": y.detach()}

    def training_step(self, b, _): return self._step(b, "train")
    def validation_step(self, b, _): return self._step(b, "val")
    def test_step(self, b, _): return self._step(b, "test")

    def on_test_epoch_start(self): self._preds, self._trues = [], []
    def on_test_batch_end(self, out, *a): self._preds.append(out["pred"]); self._trues.append(out["true"])
    def on_test_epoch_end(self):
        preds = torch.cat(self._preds).cpu()
        trues = torch.cat(self._trues).cpu()
        r = np.corrcoef(preds.numpy(), trues.numpy())[0, 1]

        outpath = self.hparams.outdir / "pheno_scatter_all.png"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.scatter(trues.numpy(), preds.numpy(), alpha=.4)
        plt.xlabel("Predicted")
        plt.ylabel("True phenotype")
        plt.title(f"Test r = {r:.3f}")
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# ───────────── Callback for loss curves ─────────────
class LossPlotter(pl.Callback):
    def __init__(self): self.metrics = []

    def on_validation_epoch_end(self, trainer, module):
        cpu_metrics = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v
            for k, v in trainer.callback_metrics.items()
            if isinstance(v, (torch.Tensor, float, int))
        }
        self.metrics.append(cpu_metrics)

    def on_train_end(self, trainer, module):
        df = pd.DataFrame(self.metrics)
        P = module.hparams.outdir
        for name, fname in [("recon", "recon_loss_curves.png"),
                            ("mse", "phenotype_loss_curves.png"),
                            ("adv", "cohort_adv_loss_curves.png"),
                            ("pop_acc", "pop_acc_curve.png")]:
            plt.figure()
            for phase in ["train", "val"]:
                col = f"{phase}_{name}"
                if col in df.columns:
                    plt.plot(df[col], label=col)
            plt.xlabel("Epoch")
            plt.ylabel(name)
            plt.title(name.replace("_", " ").title())
            plt.legend()
            plt.savefig(P / fname)
            plt.close()

# ───────────── Entrypoint ─────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genotype", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_adv", type=float, default=1.0)
    ap.add_argument("--recon_wt", type=float, default=0.1)
    ap.add_argument("--mse_wt", type=float, default=10.0)
    ap.add_argument("--beta_kl", type=float, default=1.0)
    ap.add_argument("--warm_adv", type=int, default=5)
    ap.add_argument("--outdir", type=str, default="results/vae")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    S = np.load(args.genotype).shape[1]

    model = LDVAE(S, args.hidden, args.latent, args.lr, args.lambda_adv,
                  args.recon_wt, args.mse_wt, args.beta_kl, args.warm_adv)
    model.hparams.outdir = outdir  # make sure plots go to correct folder

    ds_tr = GenoPhenoDataset(args.genotype, args.meta, train=True)
    ds_va = GenoPhenoDataset(args.genotype, args.meta, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=4)
    dl_va = DataLoader(ds_va, batch_size=args.batch, num_workers=4)

    cb_ckpt = pl.callbacks.ModelCheckpoint(
    dirpath=args.outdir,
    filename="vae_epoch{epoch}",   # <<< THIS IS GOOD
    save_top_k=-1,
    auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[cb_ckpt, LossPlotter()],
        log_every_n_steps=1,
        devices=1,
        accelerator="auto",
        num_sanity_val_steps=0  # ← disables the sanity check
    )
    trainer.fit(model, dl_tr, dl_va)
    trainer.test(model, dl_va)
    trainer.save_checkpoint(outdir / f"vae_epoch{args.epochs-1}.ckpt", weights_only=True)

    from sklearn.decomposition import PCA
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Gather latent embeddings and cohort labels
    model.eval()
    all_mu, all_cohorts = [], []
    for batch in dl_va:
        xi = batch["geno"].float().to(model.device)
        with torch.no_grad():
            mu, _ = model.encode(xi)
        all_mu.append(mu.cpu())
        all_cohorts.append(batch["cohort"])

    Z = torch.cat(all_mu).numpy()
    coh = torch.cat(all_cohorts).numpy()
    pca = PCA(n_components=2).fit_transform(Z)

    # latent_pca.png with legend
    plt.figure()
    for value, label, color in zip([0, 1], ["non-EUR", "EUR"], ["blue", "red"]):
        idx = coh == value
        plt.scatter(pca[idx, 0], pca[idx, 1], label=label, color=color, alpha=0.5)
    plt.title("Latent PCA by Cohort")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.savefig(outdir / "latent_pca.png")
    plt.close()

    # cohort_confusion.png
    all_logits = []
    for batch in dl_va:
        xi = batch["geno"].float().to(model.device)
        with torch.no_grad():
            _, _, logits, _, _ = model(xi)
        all_logits.append(logits.cpu())

    preds = torch.cat(all_logits).argmax(dim=1).numpy()
    cm = confusion_matrix(coh, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["non-EUR", "EUR"])
    disp.plot(cmap="Blues")
    plt.title("Cohort Prediction Confusion Matrix")
    plt.savefig(outdir / "cohort_confusion.png")
    plt.close()

if __name__ == "__main__":
    main()
