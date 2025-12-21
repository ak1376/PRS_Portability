# src/transformer/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_losses(
    path: Path,
    epochs: list[int],
    train: list[float],
    val: list[float],
    *,
    ylabel: str = "loss",
    title: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="val")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_contrastive_geometry(
    path: Path,
    epochs: list[int],
    pos_cos: list[float],
    neg_cos: list[float],
    perm_neg_cos: Optional[list[float]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(epochs, pos_cos, label="pos_cos (z1·z2)")
    plt.plot(epochs, neg_cos, label="neg_cos (z1·z2_shuf)")
    if perm_neg_cos is not None:
        plt.plot(epochs, perm_neg_cos, label="perm_neg_cos (z1·zperm)")
    plt.xlabel("epoch")
    plt.ylabel("cosine")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_confusion_matrix_png(cm: np.ndarray, out_png: Path, *, title: str) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
