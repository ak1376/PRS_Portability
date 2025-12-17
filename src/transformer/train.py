# src/transformer/train.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.transformer.masking import mask_haplotype
from src.transformer.model import HapMaskTransformer
from src.transformer.contrastive_losses import permute_columns_across_batch, info_nce

def _mean_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Mean cosine similarity over batch between vectors a and b.
    a,b: (B,d)
    returns: scalar tensor
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1).mean()



# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
def _compute_masked_loss(
    logits: torch.Tensor,        # (B, L, 2)
    targets: torch.Tensor,       # (B, L) in {0,1}
    loss_mask: torch.Tensor,     # (B, L) bool
    *,
    class_balance: bool = True,
) -> tuple[torch.Tensor, int]:
    """
    Cross-entropy over masked sites only.

    If class_balance=True, uses per-batch weights computed from masked targets
    to avoid collapsing to all-0 when allele-1 is rare.
    """
    n = int(loss_mask.sum().item())
    if n == 0:
        return torch.tensor(0.0, device=logits.device), 0

    logits_m = logits[loss_mask]      # (n, 2)
    targets_m = targets[loss_mask]    # (n,)

    if not class_balance:
        loss = F.cross_entropy(logits_m, targets_m)
        return loss, n

    # class balancing within masked sites
    pi = targets_m.float().mean().clamp(1e-4, 1 - 1e-4)
    w0 = 0.5 / (1.0 - pi)
    w1 = 0.5 / pi
    weights = torch.tensor([w0, w1], device=logits_m.device, dtype=logits_m.dtype)

    loss = F.cross_entropy(logits_m, targets_m, weight=weights)
    return loss, n


# -----------------------------------------------------------------------------
# Debug metrics + PNG
# -----------------------------------------------------------------------------
@torch.no_grad()
def _masked_site_metrics(
    *,
    hap_true: torch.Tensor,     # (B, L)
    hap_pred: torch.Tensor,     # (B, L)
    loss_mask: torch.Tensor,    # (B, L) bool
) -> dict[str, float]:
    """
    Computes accuracy + dumb baselines on the masked sites only.
    """
    m = loss_mask
    n_masked = int(m.sum().item())
    out: dict[str, float] = {"masked_sites": float(n_masked)}

    if n_masked == 0:
        out["acc"] = float("nan")
        out["pi_masked"] = float("nan")
        out["baseline_always0"] = float("nan")
        out["baseline_always1"] = float("nan")
        out["baseline_majority"] = float("nan")
        out["delta_vs_majority"] = float("nan")
        return out

    targets = hap_true[m]  # (n_masked,)
    preds = hap_pred[m]

    pi = float(targets.float().mean().item())
    b0 = float((targets == 0).float().mean().item())
    b1 = float((targets == 1).float().mean().item())
    bmaj = max(b0, b1)
    acc = float((preds == targets).float().mean().item())

    out["acc"] = acc
    out["pi_masked"] = pi
    out["baseline_always0"] = b0
    out["baseline_always1"] = b1
    out["baseline_majority"] = bmaj
    out["delta_vs_majority"] = acc - bmaj
    return out


@torch.no_grad()
def save_debug_png_top_only(
    out_png: Path,
    *,
    hap_true_1d: torch.Tensor,   # (L,)
    logits_1d: torch.Tensor,     # (L,2)
    loss_mask_1d: torch.Tensor,  # (L,) bool
    title: str,
    max_sites: int = 256,
):
    """
    Single-panel visualization:
      - line: P(allele=1) across window
      - vertical faint ticks: masked sites (loss sites)
      - colored points at masked sites at y=pred (0/1): green correct, red wrong
      - black dots at masked sites at y=true (0/1)

    This is “top panel only”, but actually answers: what are you predicting where it matters?
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    L = min(int(hap_true_1d.numel()), int(max_sites))
    t = hap_true_1d[:L].detach().cpu().long()
    m = loss_mask_1d[:L].detach().cpu().bool()
    lg = logits_1d[:L].detach().cpu()

    probs = torch.softmax(lg, dim=-1).numpy()  # (L,2)
    p1 = probs[:, 1]                           # P(allele=1)
    pred = probs.argmax(axis=1).astype(np.int64)  # predicted allele (0/1)
    conf = probs.max(axis=1)                      # confidence

    idx = np.arange(L)
    midx = idx[m.numpy()]

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.2))

    # belief curve
    ax.plot(idx, p1, linewidth=1.2)

    if len(midx) > 0:
        # faint vlines at masked sites
        ax.vlines(midx, 0, 1, colors="0.88", linewidth=0.8, zorder=0)

        # masked-site arrays
        t_m = t[m].numpy()         # true at masked sites
        p_m = pred[m.numpy()]      # pred at masked sites
        c_m = conf[m.numpy()]      # conf at masked sites
        correct = (t_m == p_m)

        # size by confidence (normalized)
        sizes = 20 + 120 * (c_m - c_m.min()) / (c_m.max() - c_m.min() + 1e-8)

        # predicted allele points (green/red)
        ax.scatter(
            midx[correct],
            p_m[correct],
            s=sizes[correct],
            alpha=0.95,
            color="green",
            label="pred @ masked (correct)",
            zorder=4,
        )
        ax.scatter(
            midx[~correct],
            p_m[~correct],
            s=sizes[~correct],
            alpha=0.95,
            color="red",
            label="pred @ masked (wrong)",
            zorder=4,
        )

        # true allele points (black)
        ax.scatter(
            midx,
            t_m,
            s=12,
            alpha=0.85,
            color="black",
            label="true @ masked",
            zorder=5,
        )

        acc_shown = float(correct.mean()) if len(correct) else float("nan")
        ax.set_title(title + f" | shown_masked={len(midx)} | acc_shown={acc_shown:.3f}")
    else:
        ax.set_title(title + " | shown_masked=0")

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("site (within window)")
    ax.set_ylabel("P(allele=1)")
    ax.legend(loc="upper right", frameon=True)

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def debug_snapshot_and_pngs(
    model: HapMaskTransformer,
    batch: Any,
    device: torch.device,
    *,
    mask_id: int,
    p_mask_site: float,
    out_dir: Path,
    step_tag: str,
    n_show: int = 2,
    max_sites: int = 256,
) -> dict[str, float]:
    """
    Runs ONE masked forward pass and:
      - returns metrics on masked sites
      - writes a few PNGs
    """
    model.eval()

    hap = batch.hap.to(device)  # (B,L)
    pad_mask = batch.pad_mask.to(device) if getattr(batch, "pad_mask", None) is not None else None

    hap_true = hap
    hap_masked, masked_sites = mask_haplotype(hap, mask_id=mask_id, p_mask_site=p_mask_site)

    logits, _z = model(hap_masked, pad_mask=pad_mask)  # (B,L,2)
    pred = logits.argmax(dim=-1)                        # (B,L)

    # loss mask = masked sites excluding PAD
    loss_mask = masked_sites
    if pad_mask is not None:
        loss_mask = loss_mask & (~pad_mask)

    metrics = _masked_site_metrics(hap_true=hap_true, hap_pred=pred, loss_mask=loss_mask)

    # write PNGs
    out_dir.mkdir(parents=True, exist_ok=True)
    B = int(hap.size(0))
    for i in range(min(int(n_show), B)):
        save_debug_png_top_only(
            out_dir / f"{step_tag}_ex{i}.png",
            hap_true_1d=hap_true[i],
            logits_1d=logits[i],
            loss_mask_1d=loss_mask[i],
            title=f"{step_tag} ex{i} | acc(masked)={metrics['acc']:.3f} | ΔvsMaj={metrics['delta_vs_majority']:.3f}",
            max_sites=max_sites,
        )

    return metrics


# -----------------------------------------------------------------------------
# Train / Eval epochs
# -----------------------------------------------------------------------------
def train_epoch(
    model: HapMaskTransformer,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    *,
    mask_id: int,
    p_mask_site: float = 0.15,              # MLM masking prob
    p_mask_site_ctr: float | None = None,   # contrastive masking prob (defaults to p_mask_site)
    grad_clip: float | None = 1.0,
    class_balance: bool = True,
    # ---- contrastive knobs ----
    use_contrastive: bool = True,
    contrastive_lambda: float = 0.1,
    contrastive_tau: float = 0.2,
    use_perm_negatives: bool = True,
    permute_every_k: int = 5,
) -> dict[str, float]:
    model.train()

    # --- totals for MLM loss ---
    total_mlm = 0.0
    total_sites = 0

    # --- totals for contrastive loss ---
    total_ctr = 0.0
    n_ctr_steps = 0

    # --- debug geometry metrics ---
    sum_pos_cos = 0.0          # mean cos(z1,z2)
    sum_neg_cos = 0.0          # mean cos(z1,z2_shuffled)
    sum_perm_neg_cos = 0.0     # mean cos(z1,zperm) if computed
    n_cos_steps = 0
    n_perm_cos_steps = 0

    # choose masking rate for contrastive views
    p_ctr = float(p_mask_site if p_mask_site_ctr is None else p_mask_site_ctr)

    step = 0
    for batch in loader:
        step += 1

        hap = batch.hap.to(device)  # (B,L)
        pad_mask = batch.pad_mask.to(device) if getattr(batch, "pad_mask", None) is not None else None
        hap_true = hap

        # -------------------------
        # View 1: MLM + z1
        # -------------------------
        hap_masked1, masked_sites1 = mask_haplotype(hap, mask_id=mask_id, p_mask_site=float(p_mask_site))
        logits1, z1 = model(hap_masked1, pad_mask=pad_mask)  # logits1: (B,L,2), z1: (B,d)

        loss_mask1 = masked_sites1
        if pad_mask is not None:
            loss_mask1 = loss_mask1 & (~pad_mask)

        mlm_loss, n = _compute_masked_loss(
            logits1, hap_true, loss_mask1, class_balance=class_balance
        )
        if n == 0:
            continue

        # -------------------------
        # Contrastive (optional)
        # -------------------------
        ctr_loss = torch.zeros((), device=device)

        if use_contrastive:
            # view 2 (positive)
            hap_masked2, _ = mask_haplotype(hap, mask_id=mask_id, p_mask_site=p_ctr)
            _logits2, z2 = model(hap_masked2, pad_mask=pad_mask)  # z2: (B,d)

            # --- cosine debug metrics (cheap) ---
            z1n = F.normalize(z1, dim=-1)
            z2n = F.normalize(z2, dim=-1)
            pos_cos = (z1n * z2n).sum(dim=-1).mean()

            perm = torch.randperm(z2.size(0), device=z2.device)
            neg_cos = (z1n * z2n[perm]).sum(dim=-1).mean()

            sum_pos_cos += float(pos_cos.item())
            sum_neg_cos += float(neg_cos.item())
            n_cos_steps += 1

            # optional extra negatives: LD-breaking permutation across batch
            zneg = None
            do_perm = (use_perm_negatives and permute_every_k > 0 and (step % int(permute_every_k) == 0))
            if do_perm:
                # NOTE: if you truly have PAD tokens, you may want to preserve them here.
                # If you don't have PAD, keep it simple:
                hap_perm = permute_columns_across_batch(hap)  # (B,L)

                hap_perm_masked, _ = mask_haplotype(hap_perm, mask_id=mask_id, p_mask_site=p_ctr)
                _logits_p, zperm = model(hap_perm_masked, pad_mask=pad_mask)
                zneg = zperm

                zperm_n = F.normalize(zperm, dim=-1)
                perm_neg_cos = (z1n * zperm_n).sum(dim=-1).mean()
                sum_perm_neg_cos += float(perm_neg_cos.item())
                n_perm_cos_steps += 1

            ctr_loss = info_nce(z1, z2, zneg=zneg, tau=float(contrastive_tau))
            total_ctr += float(ctr_loss.item())
            n_ctr_steps += 1

        # -------------------------
        # Combined loss + step
        # -------------------------
        loss = mlm_loss + (float(contrastive_lambda) * ctr_loss if use_contrastive else 0.0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and float(grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        total_mlm += float(mlm_loss.item()) * n
        total_sites += n

    out: dict[str, float] = {
        "mlm_loss": total_mlm / max(total_sites, 1),
    }

    if use_contrastive:
        out["ctr_loss"] = total_ctr / max(n_ctr_steps, 1)
        out["ctr_pos_cos"] = sum_pos_cos / max(n_cos_steps, 1)
        out["ctr_neg_cos"] = sum_neg_cos / max(n_cos_steps, 1)
        if n_perm_cos_steps > 0:
            out["ctr_perm_neg_cos"] = sum_perm_neg_cos / n_perm_cos_steps

    return out

@torch.no_grad()
def eval_epoch(
    model: HapMaskTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    mask_id: int,
    p_mask_site: float = 0.15,
    class_balance: bool = True,
) -> float:
    model.eval()
    total_loss = 0.0
    total_sites = 0

    for batch in loader:
        hap = batch.hap.to(device)
        pad_mask = batch.pad_mask.to(device) if getattr(batch, "pad_mask", None) is not None else None

        hap_true = hap
        hap_masked, masked_sites = mask_haplotype(hap, mask_id=mask_id, p_mask_site=p_mask_site)

        logits, _z = model(hap_masked, pad_mask=pad_mask)

        loss_mask = masked_sites
        if pad_mask is not None:
            loss_mask = loss_mask & (~pad_mask)

        loss, n = _compute_masked_loss(logits, hap_true, loss_mask, class_balance=class_balance)
        if n == 0:
            continue

        total_loss += float(loss.item()) * n
        total_sites += n

    return total_loss / max(total_sites, 1)
