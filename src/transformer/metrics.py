# src/transformer/metrics.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score

from src.transformer.masking import mask_haplotype
from src.transformer.model import HapMaskTransformer
from src.transformer.plots import plot_confusion_matrix_png


@torch.no_grad()
def eval_masked_acc_auc(
    model: HapMaskTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    mask_id: int,
    p_mask_site: float,
    class_balance: bool = True,
) -> dict[str, float]:
    """
    Accuracy + AUC on masked sites (MLM masking).

    Returns keys:
      - val_accuracy, val_auc, val_avg_loss  (kept for backwards compat)
      - accuracy, auc, avg_loss              (preferred)
    """
    from src.transformer.train import _compute_masked_loss  # local import

    model.eval()

    all_targets: list[int] = []
    all_preds: list[int] = []
    all_p1: list[float] = []
    total_loss = 0.0
    total_sites = 0

    for batch in loader:
        hap = batch.hap.to(device)
        pad_mask = batch.pad_mask.to(device) if getattr(batch, "pad_mask", None) is not None else None

        hap_true = hap.clone()  # IMPORTANT: prevent in-place masking from corrupting targets
        hap_masked, masked_sites = mask_haplotype(hap.clone(), mask_id=mask_id, p_mask_site=p_mask_site)

        logits, _z = model(hap_masked, pad_mask=pad_mask)

        loss_mask = masked_sites
        if pad_mask is not None:
            loss_mask = loss_mask & (~pad_mask)

        loss, n = _compute_masked_loss(logits, hap_true, loss_mask, class_balance=class_balance)
        if n == 0:
            continue

        probs = torch.softmax(logits, dim=-1)  # (B,L,2)
        pred = torch.argmax(logits, dim=-1)    # (B,L)

        t = hap_true[loss_mask].detach().cpu().numpy().astype(np.int64)
        p = pred[loss_mask].detach().cpu().numpy().astype(np.int64)
        p1 = probs[loss_mask, 1].detach().cpu().numpy().astype(np.float64)

        all_targets.extend(t.tolist())
        all_preds.extend(p.tolist())
        all_p1.extend(p1.tolist())

        total_loss += float(loss.item()) * n
        total_sites += n

    if len(all_targets) == 0:
        return {
            "val_accuracy": float("nan"),
            "val_auc": float("nan"),
            "val_avg_loss": float("nan"),
            "accuracy": float("nan"),
            "auc": float("nan"),
            "avg_loss": float("nan"),
        }

    targets = np.asarray(all_targets, dtype=np.int64)
    preds = np.asarray(all_preds, dtype=np.int64)
    p1 = np.asarray(all_p1, dtype=np.float64)

    acc = float(accuracy_score(targets, preds))
    try:
        auc = float(roc_auc_score(targets, p1))
    except Exception:
        auc = float("nan")

    avg_loss = float(total_loss / max(total_sites, 1))

    return {
        "val_accuracy": acc,
        "val_auc": auc,
        "val_avg_loss": avg_loss,
        "accuracy": acc,
        "auc": auc,
        "avg_loss": avg_loss,
    }


def comprehensive_validation_analysis(
    model: HapMaskTransformer,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    *,
    mask_id: int,
    p_mask_site: float = 0.15,
    class_balance: bool = True,
    split_name: str = "VAL",  # "VAL" or "TEST" (only affects plot titles + prints)
) -> dict[str, Any]:
    """
    Final analysis on masked sites for a given split (VAL/TEST).
    Writes into `output_dir`:
      - predictions.npy / targets.npy / probabilities.npy
      - confusion_matrix.png
      - roc_curve.png
      - metrics.json
    Returns: dict with accuracy/auc/etc.
    """
    from src.transformer.train import _compute_masked_loss  # local import
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score

    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: list[int] = []
    all_targets: list[int] = []
    all_probabilities: list[float] = []

    total_loss = 0.0
    total_sites = 0

    print(f"Collecting {split_name} predictions...")

    with torch.no_grad():
        for batch in loader:
            hap = batch.hap.to(device)
            pad_mask = batch.pad_mask.to(device) if getattr(batch, "pad_mask", None) is not None else None

            hap_true = hap.clone()  # IMPORTANT: prevent in-place masking from corrupting targets
            hap_masked, masked_sites = mask_haplotype(hap.clone(), mask_id=mask_id, p_mask_site=p_mask_site)

            logits, _z = model(hap_masked, pad_mask=pad_mask)

            loss_mask = masked_sites
            if pad_mask is not None:
                loss_mask = loss_mask & (~pad_mask)

            loss, n = _compute_masked_loss(logits, hap_true, loss_mask, class_balance=class_balance)
            if n == 0:
                continue

            probs = torch.softmax(logits, dim=-1)      # (B,L,2)
            pred_class = torch.argmax(logits, dim=-1)  # (B,L)

            masked_targets = hap_true[loss_mask].detach().cpu().numpy().astype(np.int64)
            masked_predictions = pred_class[loss_mask].detach().cpu().numpy().astype(np.int64)
            masked_probabilities = probs[loss_mask, 1].detach().cpu().numpy().astype(np.float64)

            all_targets.extend(masked_targets.tolist())
            all_predictions.extend(masked_predictions.tolist())
            all_probabilities.extend(masked_probabilities.tolist())

            total_loss += float(loss.item()) * n
            total_sites += n

    targets = np.asarray(all_targets, dtype=np.int64)
    predictions = np.asarray(all_predictions, dtype=np.int64)
    probabilities = np.asarray(all_probabilities, dtype=np.float64)

    if targets.size == 0:
        raise RuntimeError("No masked sites collected. Check masking / pad handling.")

    accuracy = float(accuracy_score(targets, predictions))
    try:
        auc_score = float(roc_auc_score(targets, probabilities))
    except Exception:
        auc_score = float("nan")

    avg_loss = float(total_loss / max(total_sites, 1))

    # Save arrays
    np.save(output_dir / "predictions.npy", predictions)
    np.save(output_dir / "targets.npy", targets)
    np.save(output_dir / "probabilities.npy", probabilities)

    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    plot_confusion_matrix_png(
        cm,
        output_dir / "confusion_matrix.png",
        title=f"{split_name} Confusion Matrix (acc={accuracy:.4f})",
    )

    # ROC curve
    try:
        fpr, tpr, _thr = roc_curve(targets, probabilities)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={auc_score:.4f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {split_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    metrics: dict[str, Any] = {
        "split": split_name,
        "accuracy": accuracy,
        "auc": auc_score,
        "average_loss": avg_loss,
        "total_masked_sites": int(targets.size),
        "class_distribution": {
            "class_0_count": int(np.sum(targets == 0)),
            "class_1_count": int(np.sum(targets == 1)),
            "class_0_fraction": float(np.mean(targets == 0)),
            "class_1_fraction": float(np.mean(targets == 1)),
        },
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"{split_name} analysis saved to: {output_dir}")
    return metrics
