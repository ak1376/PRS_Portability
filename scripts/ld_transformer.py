# ld_transformer_train.py

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tskit

MASK_TOKEN = 3  # index 3 in allele embedding table (0/1/2/MASK)

def make_windows(num_sites: int, L: int, stride: int | None = None) -> List[Tuple[int, int]]:
    if stride is None:
        stride = L
    return [(s, s + L) for s in range(0, num_sites - L + 1, stride)]

class GenotypeWindowDataset(Dataset):
    def __init__(self, ts_path: str | Path, meta_csv: str | Path,
                 L: int = 128, p_mask: float = 0.10, stride: int | None = None,
                 seed: int = 0):
        super().__init__()
        ts = tskit.load(ts_path)
        meta = pd.read_csv(meta_csv).set_index("individual_id")

        self.L = L
        self.p_mask = p_mask
        self.rng = np.random.default_rng(seed)
        self.meta = meta

        non_biallelic = [i for i, site in enumerate(ts.sites()) if len(site.alleles) != 2]
        ts = ts.delete_sites(non_biallelic)
        self.ts = ts

        G_hap = ts.genotype_matrix().T
        num_inds = ts.num_individuals
        assert G_hap.shape[0] == 2 * num_inds
        self.G_full = G_hap.reshape(num_inds, 2, -1).sum(axis=1).astype(np.int8)

        pop_labels = [meta.loc[ind.id, "population"] for ind in ts.individuals()]
        self.pop_vec = np.array([0 if p == "AFR" else 1 for p in pop_labels], dtype=np.int64)

        self.windows = make_windows(self.G_full.shape[1], L, stride)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        lo, hi = self.windows[idx]
        G_slice = self.G_full[:, lo:hi]
        mask = self.rng.random(G_slice.shape) < self.p_mask
        labels = np.where(mask, G_slice, -1)
        tokens = G_slice.copy()
        tokens[mask] = MASK_TOKEN
        return {
            "tokens": torch.from_numpy(tokens.astype(np.int64)),
            "labels": torch.from_numpy(labels.astype(np.int64)),
            "pop": torch.from_numpy(self.pop_vec)
        }

class LDTransformerModel(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=4, max_len=256, use_pop_emb=True):
        super().__init__()
        self.allele_emb = nn.Embedding(4, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.use_pop = use_pop_emb
        if use_pop_emb:
            self.pop_emb = nn.Embedding(2, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, 4*d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(d_model, 3)

    def forward(self, x: torch.Tensor, pop: torch.Tensor):
        B, N, L = x.shape
        tok = self.allele_emb(x)
        pos = self.pos_emb(torch.arange(L, device=x.device))[None, None, :, :]
        h = tok + pos
        if self.use_pop:
            h += self.pop_emb(pop)[:, :, None, :]
        h = h.view(B*N, L, -1)
        h = self.encoder(h)
        logits = self.head(h)
        return logits.view(B, N, L, 3)

def compute_ld_matrix(G):
    corr = np.corrcoef(G.T)
    r2 = corr**2
    np.fill_diagonal(r2, np.nan)
    return np.nanmean(r2, axis=1)

# ------------------------------------------------------------------#
#  Training & validation loop (Option A)                            #
# ------------------------------------------------------------------#
def train(args):
    """
    • Builds ONE GenotypeWindowDataset (windows are the “samples”).
    • Splits individuals (rows) into train / test *stratified by population*.
    • During optimisation:
        – loss is computed only on the train‑row labels
        – for validation, the *same* DataLoader is reused but the mask is flipped
    • Tracks average loss per epoch and saves loss curves +
      LD‑vs‑accuracy plot (see eval block).
    """
    # ---------------- dataset & dataloader ------------------------
    ds = GenotypeWindowDataset(args.ts, args.meta,
                               L=args.L, p_mask=args.p_mask,
                               stride=args.stride)
    loader = DataLoader(ds,
                        batch_size=args.B,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

    # ---------------- stratified individ split -------------------
    pop = ds.meta["population"].values         # (N,)
    afr_idx = np.where(pop == "AFR")[0]
    eur_idx = np.where(pop == "EUR")[0]

    rng = np.random.default_rng(0)
    afr_train = rng.choice(afr_idx,
                           int(len(afr_idx) * 0.8),
                           replace=False)
    eur_train = rng.choice(eur_idx,
                           int(len(eur_idx) * 0.8),
                           replace=False)

    train_rows = np.zeros_like(pop, dtype=bool)
    train_rows[afr_train] = True
    train_rows[eur_train] = True
    test_rows = ~train_rows
    train_rows_t = torch.from_numpy(train_rows)
    test_rows_t  = torch.from_numpy(test_rows)

    # ---------------- model & optim ------------------------------
    model = LDTransformerModel(d_model=args.d,
                               n_heads=args.h,
                               n_layers=args.layers,
                               max_len=args.L,
                               use_pop_emb=not args.no_pop).cuda()

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-1)

    train_curve, val_curve = [], []

    # =============================================================
    for epoch in range(args.epochs):
        # ----------- TRAIN ---------------------------------------
        model.train()
        running = 0.0
        for batch in loader:
            tok   = batch["tokens"].cuda()            # (B,N,L)
            lab   = batch["labels"].clone()           # keep on CPU for masking
            popid = batch["pop"].cuda()               # (N,)

            # mask out *test* rows for TRAIN
            lab[:, ~train_rows, :] = -1
            lab = lab.cuda()

            logits = model(tok, popid)                # (B,N,L,3)
            loss = lossf(logits.view(-1, 3), lab.view(-1))

            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()

        train_loss = running / len(loader)
        train_curve.append(train_loss)

        # ----------- VALIDATE ------------------------------------
        model.eval()
        with torch.no_grad():
            running = 0.0
            for batch in loader:
                tok   = batch["tokens"].cuda()
                lab   = batch["labels"].clone()
                popid = batch["pop"].cuda()
                # mask out *train* rows for VAL
                lab[:, train_rows, :] = -1
                lab = lab.cuda()

                logits = model(tok, popid)
                loss = lossf(logits.view(-1, 3), lab.view(-1))
                running += loss.item()

        val_loss = running / len(loader)
        val_curve.append(val_loss)

        print(f"epoch {epoch:02d}  train {train_loss:.4f}  val {val_loss:.4f}")

        # ---- save checkpoint each epoch -------------------------
        torch.save(model.state_dict(),
                   f"ld_transformer_epoch{epoch}.pt")

    # ------------------------------------------------------------
    #  Plot learning curves
    # ------------------------------------------------------------
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_curve, label="train")
    plt.plot(val_curve,   label="val")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title("LD‑Transformer learning curves")
    plt.legend(); plt.tight_layout()
    plt.savefig("ld_transformer_loss_curves.png")
    plt.close()

    # ------------------------------------------------------------
    #  Evaluate LD‑vs‑accuracy correlation on *test* individuals
    # ------------------------------------------------------------
    model.eval()
    snp_acc, ld_vec = [], []
    with torch.no_grad():
        for batch in loader:
            tok   = batch["tokens"].cuda()
            lab   = batch["labels"]                   # CPU
            popid = batch["pop"].cuda()

            # keep only test rows for evaluation
            tok_test = tok[:, test_rows_t, :]
            lab_test = lab[:, test_rows, :]

            logits = model(tok_test, popid[test_rows_t])
            pred   = logits.argmax(-1).cpu()

            # per‑SNP accuracy within the window
            correct = (pred == lab_test) & (lab_test != -1)
            acc = correct.sum(0) / (lab_test != -1).sum(0)   # (L,)
            snp_acc.extend(acc.numpy())

            # ground‑truth LD (r²) matrix for the window
            # get raw genotypes from dataset (CPU)
            G_slice = ds.G_full[:, batch["window_lo"]:batch["window_hi"]]
            # LD for test rows only
            G_t = G_slice[test_rows].astype(float)
            G_t -= G_t.mean(0)
            r = np.corrcoef(G_t.T)
            r2 = r**2
            # average LD of each SNP with all others
            ld_mean = r2.mean(1)
            ld_vec.extend(ld_mean)

    snp_acc = np.array(snp_acc)
    ld_vec  = np.array(ld_vec)

    r2_score = np.corrcoef(ld_vec, snp_acc)[0, 1]
    print(f"LD‑vs‑accuracy Pearson r = {r2_score:.3f}")

    plt.figure()
    plt.scatter(ld_vec, snp_acc, s=8, alpha=0.5)
    m, b = np.polyfit(ld_vec, snp_acc, 1)
    xs = np.linspace(0, ld_vec.max(), 100)
    plt.plot(xs, m*xs + b, c="red")
    plt.xlabel("mean r² of SNP within window")
    plt.ylabel("MLM accuracy (test inds.)")
    plt.title(f"LD ≙ MLM prediction accuracy  (r = {r2_score:.2f})")
    plt.tight_layout(); plt.savefig("ld_vs_accuracy.png"); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--p_mask", type=float, default=0.1)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--h", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--no_pop", action="store_true")
    args = ap.parse_args()
    train(args)
