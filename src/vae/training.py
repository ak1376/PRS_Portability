# training.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from lit_model import LitVAE
from model import ConvVAE1D  # only if you need it elsewhere
from loss import VAELoss
from model import VAEConfig  # if you defined your dataclass there


def make_loader(X: np.ndarray, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    ds = TensorDataset(X_t)  # yields (x,)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def main():
    # -------------------------
    # 1) Load data
    # -------------------------
    # TODO: replace with your real loading logic
    X_train = np.load("train.npy")  # shape (N, L), values in {0,1,2}
    X_val = np.load("val.npy")

    input_len = X_train.shape[1]

    # -------------------------
    # 2) Config
    # -------------------------
    cfg = VAEConfig(
        input_len=input_len,
        latent_dim=32,
        hidden_channels=(32, 64, 128),
        kernel_size=9,
        stride=2,
        padding=4,
        use_batchnorm=True,
        lr=1e-3,
        beta=1.0,
        weight_decay=0.0,
    )

    batch_size = 256
    train_loader = make_loader(X_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, batch_size=batch_size, shuffle=False)

    # -------------------------
    # 3) Lightning module
    # -------------------------
    lit = LitVAE(cfg)

    # -------------------------
    # 4) Callbacks / logging
    # -------------------------
    ckpt_cb = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="vae-{epoch:03d}-{val_loss:.4f}",
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    # -------------------------
    # 5) Trainer
    # -------------------------
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",  # optional; good on GPU
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=50,
        default_root_dir="runs/vae",
        enable_progress_bar=True,
    )

    # -------------------------
    # 6) Fit
    # -------------------------
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Best checkpoint:", ckpt_cb.best_model_path)


if __name__ == "__main__":
    main()