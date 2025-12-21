# src/transformer/train.py
from __future__ import annotations

from typing import Any
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.transformer.model import HapMaskTransformer
from src.transformer.contrastive_losses import permute_columns_across_batch, info_nce
from src.transformer.masking import mask_haplotype


@torch.no_grad()
def masked_site_metrics(
    *,
    hap_true: torch.Tensor,     # (B,L)
    hap_pred: torch.Tensor,     # (B,L)
    loss_mask: torch.Tensor,    # (B,L) bool
) -> dict[str, float]:
    if loss_mask.numel() == 0:
        return {
            "acc": float("nan"),
            "baseline_majority": float("nan"),
            "delta_vs_majority": float("nan"),
            "pi_masked": float("nan"),
            "masked_sites": 0.0,
        }

    t = hap_true[loss_mask]
    p = hap_pred[loss_mask]
    n = int(t.numel())
    if n == 0:
        return {
            "acc": float("nan"),
            "baseline_majority": float("nan"),
            "delta_vs_majority": float("nan"),
            "pi_masked": float("nan"),
            "masked_sites": 0.0,
        }

    t01 = t.float()
    pi = float(t01.mean().item())
    maj = float(max(pi, 1.0 - pi))
    acc = float((p == t).float().mean().item())

    return {
        "acc": acc,
        "baseline_majority": maj,
        "delta_vs_majority": acc - maj,
        "pi_masked": pi,
        "masked_sites": float(n),
    }


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
    Single-panel debug plot:
      - line: P(allele=1) across sites
      - faint vertical ticks: masked sites (drawn BEHIND everything)
      - colored points at masked sites at y=pred (0/1): green correct, red wrong
      - black points at masked sites at y=true (0/1)
      - point size scales with confidence at masked sites
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    L = min(int(hap_true_1d.numel()), int(max_sites))
    t = hap_true_1d[:L].detach().cpu().long()
    m = loss_mask_1d[:L].detach().cpu().bool()
    lg = logits_1d[:L].detach().cpu()

    probs = torch.softmax(lg, dim=-1).numpy()  # (L,2)
    p1 = probs[:, 1]                           # P(allele=1)
    pred = probs.argmax(axis=1).astype(np.int64)
    conf = probs.max(axis=1).astype(np.float64)

    idx = np.arange(L, dtype=np.int64)
    midx = idx[m.numpy()]

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.2))
    ax.set_axisbelow(True)  # helps keep background stuff behind

    # --- masked-site vlines FIRST, sent to back ---
    if len(midx) > 0:
        ax.vlines(midx, 0, 1, colors="0.88", linewidth=0.8, zorder=0)

    # --- belief curve above vlines ---
    ax.plot(idx, p1, linewidth=1.2, zorder=2)

    if len(midx) > 0:
        t_m = t[m].numpy().astype(np.int64)     # true at masked sites
        p_m = pred[m.numpy()]                   # pred at masked sites
        c_m = conf[m.numpy()]                   # conf at masked sites
        correct = (t_m == p_m)

        # size by confidence (robust even if all conf equal)
        cmin = float(np.min(c_m)) if len(c_m) else 0.0
        cmax = float(np.max(c_m)) if len(c_m) else 1.0
        denom = (cmax - cmin) + 1e-8
        sizes = 20.0 + 120.0 * (c_m - cmin) / denom

        # predicted allele points (green/red) on top
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

        # true allele points (black) on very top
        ax.scatter(
            midx,
            t_m,
            s=12,
            alpha=0.85,
            color="black",
            label="true @ masked",
            zorder=5,
        )

        acc_shown = float(np.mean(correct)) if len(correct) else float("nan")
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
    *,
    model,
    batch: Any,
    device: torch.device,
    mask_id: int,
    p_mask_site: float,
    out_dir: Path,
    step_tag: str,
    n_show: int = 2,
    max_sites: int = 256,
) -> dict[str, float]:
    model.eval()

    hap = batch.hap.to(device)
    pad_mask = batch.pad_mask.to(device) if getattr(batch, "pad_mask", None) is not None else None

    hap_true = hap
    hap_masked, masked_sites = mask_haplotype(hap, mask_id=mask_id, p_mask_site=p_mask_site)

    logits, _ = model(hap_masked, pad_mask=pad_mask)
    pred = logits.argmax(dim=-1)

    loss_mask = masked_sites
    if pad_mask is not None:
        loss_mask = loss_mask & (~pad_mask)

    metrics = masked_site_metrics(
        hap_true=hap_true,
        hap_pred=pred,
        loss_mask=loss_mask,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    B = int(hap.size(0))
    for i in range(min(n_show, B)):
        save_debug_png_top_only(
            out_dir / f"{step_tag}_ex{i}.png",
            hap_true_1d=hap_true[i],
            logits_1d=logits[i],
            loss_mask_1d=loss_mask[i],
            title=f"{step_tag} ex{i} | ΔvsMaj={metrics['delta_vs_majority']:.3f}",
            max_sites=max_sites,
        )

    return metrics

# -----------------------------------------------------------------------------
# Loss (MLM over masked sites)
# -----------------------------------------------------------------------------
def _compute_masked_loss(
    logits: torch.Tensor,        # (B, L, 2)
    targets: torch.Tensor,       # (B, L) in {0,1}
    loss_mask: torch.Tensor,     # (B, L) bool
    *,
    class_balance: bool = True,
) -> tuple[torch.Tensor, int]:
    """
    Cross-entropy over masked (non-pad) sites only.

    Returns:
      loss: scalar tensor
      n: number of masked sites used
    """
    n = int(loss_mask.sum().item())
    if n == 0:
        return torch.tensor(0.0, device=logits.device), 0

    logits_m = logits[loss_mask]      # (n, 2)
    targets_m = targets[loss_mask]    # (n,)

    if not class_balance:
        return F.cross_entropy(logits_m, targets_m), n

    # class balancing within masked sites
    pi = targets_m.float().mean().clamp(1e-4, 1 - 1e-4)
    w0 = 0.5 / (1.0 - pi)
    w1 = 0.5 / pi
    weights = torch.tensor([w0, w1], device=logits_m.device, dtype=logits_m.dtype)

    return F.cross_entropy(logits_m, targets_m, weight=weights), n


def _batch_cosines(z1: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (pos_cos, neg_cos) using random permutation for negatives.
    z1,z2: (B,d)
    """
    z1n = F.normalize(z1, dim=-1)
    z2n = F.normalize(z2, dim=-1)

    pos_cos = (z1n * z2n).sum(dim=-1).mean()

    perm = torch.randperm(z2.size(0), device=z2.device)
    neg_cos = (z1n * z2n[perm]).sum(dim=-1).mean()

    return pos_cos, neg_cos


# -----------------------------------------------------------------------------
# Train epoch
# -----------------------------------------------------------------------------
def train_epoch(
    model: HapMaskTransformer,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    *,
    mask_id: int,
    p_mask_site: float = 0.15,
    p_mask_site_ctr: float | None = None,
    grad_clip: float | None = 1.0,
    class_balance: bool = True,
    # contrastive knobs
    use_contrastive: bool = True,
    contrastive_lambda: float = 0.1,
    contrastive_tau: float = 0.2,
    use_perm_negatives: bool = True,
    permute_every_k: int = 5,
    # determinism (optional)
    rng: torch.Generator | None = None,
) -> dict[str, float]:
    """
    Returns dict with:
      mlm_loss   : site-weighted avg CE over masked sites
      ctr_loss   : avg InfoNCE over steps where computed (nan if disabled)
      total_loss : avg (mlm + λ·ctr) over steps
      ctr_pos_cos / ctr_neg_cos / ctr_perm_neg_cos : diagnostics (nan if not computed)
    """
    model.train()

    # MLM totals (site-weighted)
    total_mlm = 0.0
    total_sites = 0

    # CTR totals (step-weighted)
    total_ctr = 0.0
    n_ctr_steps = 0

    # Total loss totals (step-weighted)
    total_total = 0.0
    n_total_steps = 0

    # geometry debug
    sum_pos_cos = 0.0
    sum_neg_cos = 0.0
    n_cos_steps = 0

    sum_perm_neg_cos = 0.0
    n_perm_cos_steps = 0

    p_ctr = float(p_mask_site if p_mask_site_ctr is None else p_mask_site_ctr)
    do_ctr = bool(use_contrastive and float(contrastive_lambda) > 0.0)

    step = 0
    for batch in loader:
        step += 1

        hap = batch.hap.to(device)
        pad_mask = batch.pad_mask.to(device) if getattr(batch, "pad_mask", None) is not None else None
        hap_true = hap

        # ---- view 1 (MLM) ----
        hap_masked1, masked_sites1 = mask_haplotype(
            hap, mask_id=mask_id, p_mask_site=float(p_mask_site), rng=rng
        )
        logits1, z1 = model(hap_masked1, pad_mask=pad_mask)

        loss_mask1 = masked_sites1
        if pad_mask is not None:
            loss_mask1 = loss_mask1 & (~pad_mask)

        mlm_loss, n = _compute_masked_loss(logits1, hap_true, loss_mask1, class_balance=class_balance)
        if n == 0:
            continue

        # ---- contrastive (optional) ----
        ctr_loss = torch.zeros((), device=device)

        if do_ctr:
            hap_masked2, _ = mask_haplotype(hap, mask_id=mask_id, p_mask_site=p_ctr, rng=rng)
            _logits2, z2 = model(hap_masked2, pad_mask=pad_mask)

            # geometry diagnostics
            pos_cos, neg_cos = _batch_cosines(z1, z2)
            sum_pos_cos += float(pos_cos.item())
            sum_neg_cos += float(neg_cos.item())
            n_cos_steps += 1

            zneg = None
            do_perm = bool(use_perm_negatives and permute_every_k > 0 and (step % int(permute_every_k) == 0))
            if do_perm:
                hap_perm = permute_columns_across_batch(hap)
                hap_perm_masked, _ = mask_haplotype(hap_perm, mask_id=mask_id, p_mask_site=p_ctr, rng=rng)
                _logits_p, zperm = model(hap_perm_masked, pad_mask=pad_mask)
                zneg = zperm

                # perm-neg cosine diagnostic
                z1n = F.normalize(z1, dim=-1)
                zpn = F.normalize(zperm, dim=-1)
                perm_neg_cos = (z1n * zpn).sum(dim=-1).mean()
                sum_perm_neg_cos += float(perm_neg_cos.item())
                n_perm_cos_steps += 1

            ctr_loss = info_nce(z1, z2, zneg=zneg, tau=float(contrastive_tau))
            total_ctr += float(ctr_loss.item())
            n_ctr_steps += 1

        # ---- combined ----
        total_loss = mlm_loss + float(contrastive_lambda) * ctr_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if grad_clip is not None and float(grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        # aggregates
        total_mlm += float(mlm_loss.item()) * n
        total_sites += n

        total_total += float(total_loss.item())
        n_total_steps += 1

    out: dict[str, float] = {
        "mlm_loss": total_mlm / max(total_sites, 1),
        "total_loss": total_total / max(n_total_steps, 1),
        "ctr_loss": float("nan"),
        "ctr_pos_cos": float("nan"),
        "ctr_neg_cos": float("nan"),
        "ctr_perm_neg_cos": float("nan"),
    }

    if do_ctr:
        out["ctr_loss"] = total_ctr / max(n_ctr_steps, 1)
        out["ctr_pos_cos"] = sum_pos_cos / max(n_cos_steps, 1)
        out["ctr_neg_cos"] = sum_neg_cos / max(n_cos_steps, 1)
        out["ctr_perm_neg_cos"] = (sum_perm_neg_cos / n_perm_cos_steps) if n_perm_cos_steps > 0 else float("nan")

    return out


# -----------------------------------------------------------------------------
# Eval epoch losses: MLM / CTR / TOTAL
# -----------------------------------------------------------------------------
@torch.no_grad()
def eval_epoch_losses(
    model: HapMaskTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    mask_id: int,
    p_mask_site: float = 0.15,
    class_balance: bool = True,
    # contrastive (optional)
    use_contrastive: bool = False,
    contrastive_lambda: float = 0.0,
    contrastive_tau: float = 0.2,
    p_mask_site_ctr: float | None = None,
    use_perm_negatives: bool = True,
    permute_every_k: int = 5,
    rng: torch.Generator | None = None,
) -> dict[str, float]:
    model.eval()

    total_mlm = 0.0
    total_sites = 0

    total_ctr = 0.0
    n_ctr_steps = 0

    total_total = 0.0
    n_total_steps = 0

    p_ctr = float(p_mask_site if p_mask_site_ctr is None else p_mask_site_ctr)
    do_ctr = bool(use_contrastive and float(contrastive_lambda) > 0.0)

    step = 0
    for batch in loader:
        step += 1

        hap = batch.hap.to(device)
        pad_mask = batch.pad_mask.to(device) if getattr(batch, "pad_mask", None) is not None else None
        hap_true = hap

        # view 1 (MLM)
        hap_masked1, masked_sites1 = mask_haplotype(
            hap, mask_id=mask_id, p_mask_site=float(p_mask_site), rng=rng
        )
        logits1, z1 = model(hap_masked1, pad_mask=pad_mask)

        loss_mask1 = masked_sites1
        if pad_mask is not None:
            loss_mask1 = loss_mask1 & (~pad_mask)

        mlm_loss, n = _compute_masked_loss(logits1, hap_true, loss_mask1, class_balance=class_balance)
        if n == 0:
            continue

        # contrastive
        ctr_loss = torch.zeros((), device=device)
        if do_ctr:
            hap_masked2, _ = mask_haplotype(hap, mask_id=mask_id, p_mask_site=p_ctr, rng=rng)
            _logits2, z2 = model(hap_masked2, pad_mask=pad_mask)

            zneg = None
            do_perm = bool(use_perm_negatives and permute_every_k > 0 and (step % int(permute_every_k) == 0))
            if do_perm:
                hap_perm = permute_columns_across_batch(hap)
                hap_perm_masked, _ = mask_haplotype(hap_perm, mask_id=mask_id, p_mask_site=p_ctr, rng=rng)
                _logits_p, zperm = model(hap_perm_masked, pad_mask=pad_mask)
                zneg = zperm

            ctr_loss = info_nce(z1, z2, zneg=zneg, tau=float(contrastive_tau))
            total_ctr += float(ctr_loss.item())
            n_ctr_steps += 1

        total = mlm_loss + float(contrastive_lambda) * ctr_loss

        total_mlm += float(mlm_loss.item()) * n
        total_sites += n

        total_total += float(total.item())
        n_total_steps += 1

    return {
        "mlm_loss": total_mlm / max(total_sites, 1),
        "ctr_loss": (total_ctr / max(n_ctr_steps, 1)) if do_ctr else float("nan"),
        "total_loss": total_total / max(n_total_steps, 1),
    }


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
    """Back-compat: returns MLM loss only (site-weighted)."""
    out = eval_epoch_losses(
        model=model,
        loader=loader,
        device=device,
        mask_id=mask_id,
        p_mask_site=p_mask_site,
        class_balance=class_balance,
        use_contrastive=False,
        contrastive_lambda=0.0,
    )
    return float(out["mlm_loss"])
