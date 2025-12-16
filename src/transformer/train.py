# src/transformer/train.py
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.transformer.masking import mask_haplotypes
from src.transformer.model import HapMaskTransformer


def load_checkpoint_model(ckpt_path: str, device: str = "cpu"):
    """
    Loads checkpoints saved by your training script:
      torch.save({"state_dict": model.state_dict(), "config": {...}}, out_model)

    Returns (model, ckpt_dict).
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    state = ckpt.get("state_dict")
    if state is None:
        raise ValueError("Checkpoint missing 'state_dict'.")

    cfg = ckpt.get("config", {})
    # Minimal required for model construction
    model_kwargs = dict(
        vocab_size=int(cfg.get("vocab_size", 3)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 8)),
        n_layers=int(cfg.get("n_layers", 6)),
        dropout=float(cfg.get("dropout", 0.1)),
        pad_id=cfg.get("pad_id", None),
    )

    model = HapMaskTransformer(**model_kwargs).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


@torch.no_grad()
def reconstruct_from_haps(
    model: HapMaskTransformer,
    hap1_tokens: torch.Tensor,   # (B, L) int64
    hap2_tokens: torch.Tensor,   # (B, L) int64
    *,
    pad_mask: torch.Tensor | None = None,
):
    """
    Forward pass and return predicted hap alleles (0/1) as int64 tensors:
      pred1, pred2: (B, L)

    Note: This will predict *all* sites. If you want imputation, you should mask
    positions in hap*_tokens BEFORE calling this.
    """
    hap1_logits, hap2_logits, _z = model(hap1_tokens, hap2_tokens, pad_mask=pad_mask)
    pred1 = torch.argmax(hap1_logits, dim=-1).long()
    pred2 = torch.argmax(hap2_logits, dim=-1).long()
    return pred1, pred2


def _compute_masked_hap_loss(
    hap_logits: torch.Tensor,      # (B, L, 2)
    hap_targets: torch.Tensor,     # (B, L) in {0,1}
    loss_mask: torch.Tensor,       # (B, L) bool
) -> tuple[torch.Tensor, int]:
    """
    Returns (loss, n_sites) where loss is mean CE over masked sites.
    """
    if loss_mask.sum() == 0:
        return torch.tensor(0.0, device=hap_logits.device), 0
    logits_m = hap_logits[loss_mask]      # (N_masked, 2)
    targets_m = hap_targets[loss_mask]    # (N_masked,)
    loss = F.cross_entropy(logits_m, targets_m)
    return loss, int(loss_mask.sum().item())


def train_epoch(
    model: HapMaskTransformer,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    mask_id: int,
    p_mask_site: float = 0.15,
    mask_both_prob: float = 1.0,
    grad_clip: float | None = 1.0,
):
    """
    One epoch of masked haplotype allele prediction.
    Loss is computed ONLY on masked sites (and not PAD).
    """
    model.train()
    total_loss = 0.0
    total_sites = 0

    for batch in loader:
        hap1 = batch.hap1.to(device)  # (B, L) tokens, typically {0,1,...}
        hap2 = batch.hap2.to(device)
        pad_mask = batch.pad_mask.to(device) if batch.pad_mask is not None else None

        # targets are ORIGINAL hap alleles (0/1) before masking
        # If your hap tokens include special tokens in the unmasked data, you should
        # ensure targets are {0,1} at those sites. With your current dataset, hap1/hap2
        # are allele calls, so this is fine.
        hap1_t = hap1
        hap2_t = hap2

        # Step 0: mask inputs (produces masked_sites bool mask)
        hap1_m, hap2_m, masked_sites = mask_haplotypes(
            hap1, hap2,
            mask_id=mask_id,
            p_mask_site=p_mask_site,
            mask_both_prob=mask_both_prob,
        )

        # Step 1: forward to hap logits
        hap1_logits, hap2_logits, _z = model(
            hap1_m, hap2_m,
            pad_mask=pad_mask,
            return_site_features=False,
        )  # each (B, L, 2)

        # Step 2: build loss mask (masked sites, excluding PAD)
        loss_mask = masked_sites
        if pad_mask is not None:
            loss_mask = loss_mask & (~pad_mask)

        # Step 3: CE loss on masked sites for each haplotype
        loss1, n1 = _compute_masked_hap_loss(hap1_logits, hap1_t, loss_mask)
        loss2, n2 = _compute_masked_hap_loss(hap2_logits, hap2_t, loss_mask)

        if (n1 + n2) == 0:
            continue

        loss = 0.5 * (loss1 + loss2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # count masked sites once per locus (not twice), for reporting stability
        n_sites = int(loss_mask.sum().item())
        total_loss += float(loss.item()) * n_sites
        total_sites += n_sites

    return total_loss / max(total_sites, 1)


@torch.no_grad()
def eval_epoch(
    model: HapMaskTransformer,
    loader: DataLoader,
    device: torch.device,
    mask_id: int,
    p_mask_site: float = 0.15,
    mask_both_prob: float = 1.0,
):
    """
    Validation epoch with the SAME masking objective, but no optimization.
    """
    model.eval()
    total_loss = 0.0
    total_sites = 0

    for batch in loader:
        hap1 = batch.hap1.to(device)
        hap2 = batch.hap2.to(device)
        pad_mask = batch.pad_mask.to(device) if batch.pad_mask is not None else None

        hap1_t = hap1
        hap2_t = hap2

        hap1_m, hap2_m, masked_sites = mask_haplotypes(
            hap1, hap2,
            mask_id=mask_id,
            p_mask_site=p_mask_site,
            mask_both_prob=mask_both_prob,
        )

        hap1_logits, hap2_logits, _z = model(hap1_m, hap2_m, pad_mask=pad_mask)

        loss_mask = masked_sites
        if pad_mask is not None:
            loss_mask = loss_mask & (~pad_mask)

        loss1, n1 = _compute_masked_hap_loss(hap1_logits, hap1_t, loss_mask)
        loss2, n2 = _compute_masked_hap_loss(hap2_logits, hap2_t, loss_mask)
        if (n1 + n2) == 0:
            continue

        loss = 0.5 * (loss1 + loss2)

        n_sites = int(loss_mask.sum().item())
        total_loss += float(loss.item()) * n_sites
        total_sites += n_sites

    return total_loss / max(total_sites, 1)


@torch.no_grad()
def embed_individuals(model: HapMaskTransformer, loader: DataLoader, device: torch.device):
    """
    Produce z embeddings for each individual WITHOUT masking.
    """
    model.eval()
    zs = []
    for batch in loader:
        hap1 = batch.hap1.to(device)
        hap2 = batch.hap2.to(device)
        pad_mask = batch.pad_mask.to(device) if batch.pad_mask is not None else None

        _hap1_logits, _hap2_logits, z = model(hap1, hap2, pad_mask=pad_mask)
        zs.append(z.cpu())
    return torch.cat(zs, dim=0)  # (N, d)
