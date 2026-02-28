# src/transformer/splitting.py
from __future__ import annotations

from typing import Callable, Tuple
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def make_train_val_test_loaders(
    ds: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_frac: float | None = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    device: torch.device,
    collate_fn: Callable,
) -> Tuple[DataLoader, DataLoader, DataLoader | None, int, int, int]:
    """
    Create train/val/test DataLoaders with:
      - deterministic split via seed
      - train batches shuffled (random individuals)
      - val/test not shuffled
    """

    n = len(ds)
    if n < 2:
        raise ValueError(f"Need at least 2 samples to split, got n={n}")

    val_frac = float(val_frac)
    test_frac = float(test_frac)
    if train_frac is None:
        train_frac = 1.0 - val_frac - test_frac
    train_frac = float(train_frac)

    if train_frac <= 0 or val_frac < 0 or test_frac < 0:
        raise ValueError(f"Invalid fracs: train={train_frac} val={val_frac} test={test_frac}")

    s = train_frac + val_frac + test_frac
    if abs(s - 1.0) > 1e-6:
        train_frac /= s
        val_frac /= s
        test_frac /= s

    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    n_train = n - n_val - n_test

    # Ensure non-empty train/val
    if n_train <= 0:
        raise ValueError(f"Not enough samples for a non-empty train split: n={n}")
    if n_val <= 0:
        # steal one from train
        if n_train <= 1:
            raise ValueError(f"Not enough samples for non-empty train+val splits: n={n}")
        n_train -= 1
        n_val += 1

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()

    test_idx = perm[:n_test]
    val_idx  = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx)
    test_ds  = Subset(ds, test_idx) if n_test > 0 else None

    def _loader(subset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            subset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=int(num_workers),
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
            persistent_workers=(int(num_workers) > 0),
        )

    train_dl = _loader(train_ds, shuffle=True)
    val_dl   = _loader(val_ds, shuffle=False)
    test_dl  = (_loader(test_ds, shuffle=False) if test_ds is not None else None)

    return train_dl, val_dl, test_dl, len(train_ds), len(val_ds), (len(test_ds) if test_ds is not None else 0)
