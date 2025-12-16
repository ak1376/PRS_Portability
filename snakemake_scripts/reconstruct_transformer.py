#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transformer.train import load_checkpoint_model
from src.transformer.masking import mask_haplotypes


def _load_cfg(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--config not found: {path}")
    return yaml.safe_load(p.read_text()) or {}


def _choose_windows(L_total: int, window_len: int) -> list[tuple[int, int]]:
    if window_len <= 0:
        raise ValueError("window_len must be > 0")
    if window_len >= L_total:
        return [(0, L_total)]
    starts = list(range(0, L_total, window_len))
    windows = []
    for s in starts:
        e = s + window_len
        if e <= L_total:
            windows.append((s, e))
        else:
            windows.append((L_total - window_len, L_total))
            break
    return sorted(set(windows))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--hap1", required=True)
    p.add_argument("--hap2", required=True)
    p.add_argument("--out-hap1", required=True)
    p.add_argument("--out-hap2", required=True)
    p.add_argument("--out-geno", required=True)

    # controls
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # override masking params if you want
    p.add_argument("--p-mask-site", type=float, default=None)
    p.add_argument("--mask-both-prob", type=float, default=None)
    p.add_argument("--mask-id", type=int, default=None)

    args = p.parse_args()

    cfg = _load_cfg(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    window_len = int(data_cfg.get("window_len", 1024))
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 32))

    p_mask_site = float(args.p_mask_site if args.p_mask_site is not None else train_cfg.get("p_mask_site", 0.15))
    mask_both_prob = float(args.mask_both_prob if args.mask_both_prob is not None else train_cfg.get("mask_both_prob", 1.0))
    mask_id = int(args.mask_id if args.mask_id is not None else train_cfg.get("mask_id", 2))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path(args.out_hap1).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    # reproducible masking
    g = torch.Generator(device="cpu")
    g.manual_seed(int(args.seed))

    # Load haplotypes (must be 0/1 tokens already)
    hap1_np = np.load(args.hap1).astype(np.int64)
    hap2_np = np.load(args.hap2).astype(np.int64)
    if hap1_np.shape != hap2_np.shape:
        raise ValueError(f"hap1 shape {hap1_np.shape} != hap2 shape {hap2_np.shape}")
    N, L_total = hap1_np.shape

    # outputs start as original (so unmasked sites stay unchanged)
    recon_h1 = hap1_np.copy()
    recon_h2 = hap2_np.copy()

    # Load model
    model, _ckpt = load_checkpoint_model(args.model, device=str(device))
    model.eval()

    windows = _choose_windows(L_total, window_len)
    print(
        f"[masked_impute] N={N} L_total={L_total} window_len={window_len} "
        f"n_windows={len(windows)} batch_size={batch_size} "
        f"p_mask_site={p_mask_site} mask_both_prob={mask_both_prob} mask_id={mask_id}"
    )

    total_masked = 0
    total_correct1 = 0
    total_correct2 = 0
    total_ce1 = 0.0
    total_ce2 = 0.0

    for wi, (s, e) in enumerate(windows, start=1):
        h1w = torch.from_numpy(hap1_np[:, s:e]).long()
        h2w = torch.from_numpy(hap2_np[:, s:e]).long()
        Lw = e - s

        # batch over individuals
        for i0 in range(0, N, batch_size):
            i1 = min(N, i0 + batch_size)

            b1 = h1w[i0:i1].to(device, non_blocking=True)
            b2 = h2w[i0:i1].to(device, non_blocking=True)

            # mask like training
            b1m, b2m, masked_sites = mask_haplotypes(
                b1, b2,
                mask_id=mask_id,
                p_mask_site=p_mask_site,
                mask_both_prob=mask_both_prob,
                rng=None,   # if you want GPU RNG reproducibility, do it differently; CPU seed is enough for now
            )

            if masked_sites.sum().item() == 0:
                continue

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    hap1_logits, hap2_logits, _z = model(b1m, b2m, pad_mask=None)
            else:
                hap1_logits, hap2_logits, _z = model(b1m, b2m, pad_mask=None)

            # targets are the original alleles at masked sites
            t1 = b1[masked_sites]  # (n_masked,)
            t2 = b2[masked_sites]

            l1 = hap1_logits[masked_sites]  # (n_masked, 2)
            l2 = hap2_logits[masked_sites]

            ce1 = F.cross_entropy(l1, t1)
            ce2 = F.cross_entropy(l2, t2)

            p1 = torch.argmax(l1, dim=-1)
            p2 = torch.argmax(l2, dim=-1)

            nmask = int(masked_sites.sum().item())
            total_masked += nmask
            total_ce1 += float(ce1.item()) * nmask
            total_ce2 += float(ce2.item()) * nmask
            total_correct1 += int((p1 == t1).sum().item())
            total_correct2 += int((p2 == t2).sum().item())

            # IMPORTANT: only write predictions back at masked sites
            # Update numpy arrays for this batch/window slice
            pred1_full = b1.detach().clone()
            pred2_full = b2.detach().clone()
            pred1_full[masked_sites] = p1
            pred2_full[masked_sites] = p2

            recon_h1[i0:i1, s:e] = pred1_full.cpu().numpy().astype(np.int64)
            recon_h2[i0:i1, s:e] = pred2_full.cpu().numpy().astype(np.int64)

        if wi % 10 == 0 or wi == 1 or wi == len(windows):
            print(f"[masked_impute] window {wi}/{len(windows)} s={s} e={e} (Lw={Lw})")

    # summary
    if total_masked > 0:
        acc1 = total_correct1 / total_masked
        acc2 = total_correct2 / total_masked
        ce1 = total_ce1 / total_masked
        ce2 = total_ce2 / total_masked
        print(f"[masked_impute] masked_sites={total_masked}")
        print(f"[masked_impute] hap1: acc={acc1:.4f} CE={ce1:.4f}")
        print(f"[masked_impute] hap2: acc={acc2:.4f} CE={ce2:.4f}")
    else:
        print("[masked_impute] WARNING: no masked sites were sampled (p_mask_site too small?)")

    recon_g = (recon_h1 + recon_h2).clip(0, 2).astype(np.int8)

    np.save(args.out_hap1, recon_h1.astype(np.float32))
    np.save(args.out_hap2, recon_h2.astype(np.float32))
    np.save(args.out_geno, recon_g.astype(np.float32))

    print(f"[masked_impute] wrote:\n  {args.out_hap1}\n  {args.out_hap2}\n  {args.out_geno}")


if __name__ == "__main__":
    main()
