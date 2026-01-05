# src/transformer/splitting.py
from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.transformer.window_batching import SameWindowBatchSampler


class IndexMapWindowDataset(Dataset):
    """
    Wraps a base dataset + a list of base indices (like Subset),
    BUT preserves the ability to pass (local_ind, start) tuples
    down to the base dataset as (base_ind, start).
    """
    def __init__(self, base_ds: Dataset, index_map: list[int]):
        self.base_ds = base_ds
        self.index_map = list(map(int, index_map))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            local_ind, s = int(item[0]), int(item[1])
            base_ind = self.index_map[local_ind]
            return self.base_ds[(base_ind, s)]
        base_ind = self.index_map[int(item)]
        return self.base_ds[base_ind]


def _make_loader(
    subset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
    collate_fn: Callable,
    batch_sampler=None,
) -> DataLoader:
    if batch_sampler is not None:
        # When using batch_sampler, you must NOT pass batch_size or shuffle
        return DataLoader(
            subset,
            batch_sampler=batch_sampler,
            num_workers=int(num_workers),
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
            persistent_workers=(int(num_workers) > 0),
        )

    return DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=(int(num_workers) > 0),
    )


def make_train_val_test_loaders(
    ds: torch.utils.data.Dataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_frac: float | None = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    device: torch.device,
    collate_fn: Callable,
    # NEW:
    same_window_batches: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader | None, int, int, int]:

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
    if not (abs(s - 1.0) < 1e-6):
        train_frac /= s
        val_frac /= s
        test_frac /= s

    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    n_train = n - n_val - n_test

    if n_train <= 0:
        need = 1 - n_train
        take = min(need, n_val - 1) if n_val > 1 else 0
        n_val -= take
        n_train += take
        need = 1 - n_train
        take = min(need, n_test - 1) if n_test > 1 else 0
        n_test -= take
        n_train += take

    if n_val <= 0:
        take = 1 - n_val
        if n_train - take < 1:
            raise ValueError(f"Not enough samples to make non-empty train+val splits: n={n}")
        n_train -= take
        n_val += take

    if test_frac > 0.0 and n_test <= 0:
        if n_train > 1:
            n_train -= 1
            n_test += 1
        else:
            n_test = 0

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()

    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    # IMPORTANT: use IndexMapWindowDataset so (local_ind, start) can flow through
    train_ds = IndexMapWindowDataset(ds, train_idx)
    val_ds = IndexMapWindowDataset(ds, val_idx)

    # TRAIN LOADER
    train_batch_sampler = None
    if same_window_batches:
        # ds must expose L_total and window_len for this to work
        if not hasattr(ds, "L_total") or not hasattr(ds, "window_len"):
            raise ValueError("same_window_batches=True requires dataset to have L_total and window_len attributes")
        if ds.window_len is None:
            raise ValueError("same_window_batches=True requires ds.window_len to be set (not None)")

        train_batch_sampler = SameWindowBatchSampler(
            n_samples=len(train_ds),
            L_total=int(ds.L_total),
            window_len=int(ds.window_len),
            batch_size=int(batch_size),
            seed=int(seed),
            drop_last=True,
        )

    train_dl = _make_loader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(train_batch_sampler is None),
        device=device,
        collate_fn=collate_fn,
        batch_sampler=train_batch_sampler,
    )

    # VAL LOADER (keep normal batching)
    val_dl = _make_loader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        device=device,
        collate_fn=collate_fn,
    )

    test_dl = None
    if n_test > 0:
        test_ds = IndexMapWindowDataset(ds, test_idx)
        test_dl = _make_loader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            device=device,
            collate_fn=collate_fn,
        )

    return train_dl, val_dl, test_dl, len(train_ds), len(val_ds), int(n_test)
