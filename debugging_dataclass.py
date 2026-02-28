#!/usr/bin/env python3
"""
debug_windows_hapdataset.py

Verifies HapDataset window selection behavior:
A) Same epoch + same individual -> same window
B) Different epoch + same individual -> (usually) different window
C) Different individuals -> different windows
D) What batches look like under DataLoader (are window starts mixed within batch?)

Run (example):
  python debug_windows_hapdataset.py --hap /sietch_colab/akapoor/PRS_Portability/experiments/out_of_africa/processed_data/hap1.npy --window-len 512 --seed 42 --ind 17 --batch-size 8

Optional:
  python debug_windows_hapdataset.py --hap /sietch_colab/akapoor/PRS_Portability/experiments/out_of_africa/processed_data/hap1.npy --window-len 512 --seed 42 --ind 17 --epoch0 0 --epoch1 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

# Adjust this import if your project layout differs
from src.transformer.data_class import HapDataset


def infer_start(full_row: torch.Tensor, window: torch.Tensor) -> int | None:
    """
    Find the start index where `window` occurs inside `full_row`.
    Returns the first match or None if not found.

    This is O(L_total * window_len) worst-case, but fine for debugging.
    """
    full_row = full_row.detach().cpu()
    window = window.detach().cpu()
    L = full_row.numel()
    w = window.numel()
    if w > L:
        return None

    # quick exact scan
    for s in range(0, L - w + 1):
        if torch.equal(full_row[s : s + w], window):
            return s
    return None


def summarize_one(ds: HapDataset, hap_all: torch.Tensor, ind: int) -> tuple[int | None, torch.Tensor]:
    """
    Fetch ds[ind] and infer start position by searching inside hap_all[ind].
    Returns (start, window_tensor).
    """
    item = ds[ind]
    win = item.hap
    start = infer_start(hap_all[ind], win)
    return start, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hap", type=str, required=True, help="Path to hap .npy (N,L_total)")
    ap.add_argument("--window-len", type=int, default=512)
    ap.add_argument("--window-mode", type=str, default="random", choices=["random", "first", "middle", "fixed"])
    ap.add_argument("--fixed-start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ind", type=int, default=17, help="Individual index to probe")
    ap.add_argument("--ind2", type=int, default=18, help="Second individual index to probe")
    ap.add_argument("--epoch0", type=int, default=0)
    ap.add_argument("--epoch1", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    hap_np = np.load(args.hap)
    if hap_np.ndim != 2:
        raise ValueError(f"--hap must be 2D (N,L). got {hap_np.shape}")
    hap_all = torch.from_numpy(hap_np).long()

    print(f"[load] hap_all shape = {tuple(hap_all.shape)}")

    ds = HapDataset(
        hap_all=hap_all,
        pad_id=None,
        window_len=int(args.window_len),
        window_mode=str(args.window_mode),
        fixed_start=int(args.fixed_start),
        seed=int(args.seed),
    )

    # -------------------------
    # Check A: same epoch, same ind -> same window
    # -------------------------
    print("\n=== Check A: same epoch, same individual twice ===")
    ds.set_epoch(int(args.epoch0))
    s1, w1 = summarize_one(ds, hap_all, int(args.ind))
    s2, w2 = summarize_one(ds, hap_all, int(args.ind))

    print(f"epoch={args.epoch0} ind={args.ind} start1={s1} start2={s2} equal_windows={torch.equal(w1, w2)}")
    if s1 is None or s2 is None:
        print("  NOTE: Could not infer start by exact matching; this can happen if the window occurs multiple times or data is weird.")
    else:
        print(f"  window snippet (first 12 tokens): {w1[:12].tolist()}")

    # -------------------------
    # Check B: different epochs, same ind -> different window (usually)
    # -------------------------
    print("\n=== Check B: different epochs, same individual ===")
    ds.set_epoch(int(args.epoch0))
    s0, w0 = summarize_one(ds, hap_all, int(args.ind))
    ds.set_epoch(int(args.epoch1))
    s1b, w1b = summarize_one(ds, hap_all, int(args.ind))

    print(f"ind={args.ind} epoch0={args.epoch0} start={s0}")
    print(f"ind={args.ind} epoch1={args.epoch1} start={s1b}")
    print(f"equal_windows={torch.equal(w0, w1b)} (expect False often if window_mode=random)")

    # -------------------------
    # Check C: same epoch, different inds -> likely different starts
    # -------------------------
    print("\n=== Check C: same epoch, different individuals ===")
    ds.set_epoch(int(args.epoch0))
    sa, _ = summarize_one(ds, hap_all, int(args.ind))
    sb, _ = summarize_one(ds, hap_all, int(args.ind2))
    print(f"epoch={args.epoch0} ind={args.ind} start={sa}")
    print(f"epoch={args.epoch0} ind={args.ind2} start={sb}")

    # -------------------------
    # Check D: DataLoader batch -> do we see mixed starts within one batch?
    # (This answers: "does a batch have SAME WINDOW?")
    # -------------------------
    print("\n=== Check D: DataLoader batch composition ===")
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,              # random individuals
        num_workers=int(args.num_workers),
        drop_last=True,
        collate_fn=None,           # ds returns HapBatch; default collate will try to collate dataclasses badly
    )

    # We can't use default collate on your dataclass; we'll manually pull items like DataLoader would.
    # Easiest is to sample indices ourselves like a batch.
    g = torch.Generator().manual_seed(0)
    n = len(ds)
    batch_inds = torch.randint(0, n, (int(args.batch_size),), generator=g).tolist()
    ds.set_epoch(int(args.epoch0))
    starts = []
    for ind in batch_inds:
        start, _ = summarize_one(ds, hap_all, int(ind))
        starts.append(start)
    print(f"sampled batch inds: {batch_inds}")
    print(f"inferred starts:    {starts}")
    if all(s is not None for s in starts):
        uniq = len(set(starts))
        print(f"unique starts in batch = {uniq}/{len(starts)}")
        if uniq == 1:
            print("  -> This batch appears to share the SAME window start (unexpected unless you force same-window batching).")
        else:
            print("  -> This batch has MIXED window starts (expected for your current HapDataset + random individuals).")
    else:
        print("  NOTE: Some starts were None (couldn't infer by exact matching). Still useful to inspect.")

    print("\nDone.")


if __name__ == "__main__":
    main()
