# src/transformer/data_class.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

'''
Window start indices are chosen randomly per individual and per epoch (if window_mode="random"), but in a deterministic way using a seed.
We do not want to use deterministic seeds. 
'''


def collate_hapbatch(items):
    hap = torch.stack([x.hap for x in items], dim=0)
    pad_mask = None
    if items[0].pad_mask is not None:
        pad_mask = torch.stack([x.pad_mask for x in items], dim=0)
    return type(items[0])(hap=hap, pad_mask=pad_mask)


@dataclass
class HapBatch:
    hap: torch.Tensor
    pad_mask: torch.Tensor | None = None


class HapDataset(Dataset):
    """
    Stores haplotypes:
      hap_all: (N, L_total) long

    Supports two kinds of indexing:
      - idx: int -> chooses window start according to window_mode (deterministic if seed provided)
      - idx: (int_ind, int_start) -> uses provided start (for same-window batching)
    """
    def __init__(
        self,
        hap_all: torch.Tensor,
        pad_id: int | None = None,
        *,
        window_len: int | None = None,
        window_mode: str = "random",   # "random" | "first" | "middle" | "fixed"
        fixed_start: int = 0,

        # NEW: deterministic random windows
        seed: int = 0,
    ):
        self.hap_all = hap_all.long()
        self.pad_id = pad_id
        self.L_total = self.hap_all.shape[1]
        self.window_len = window_len
        self.window_mode = window_mode
        self.fixed_start = fixed_start

        # NEW
        self.seed = int(seed)
        self.epoch = 0  # can be set by training loop for reproducible “new windows each epoch”

        if self.window_len is not None:
            if self.window_len <= 0:
                raise ValueError("window_len must be > 0")
            if self.window_len > self.L_total:
                raise ValueError(f"window_len ({self.window_len}) > L_total ({self.L_total})")
            if self.window_mode not in {"random", "first", "middle", "fixed"}:
                raise ValueError(f"Unknown window_mode: {self.window_mode}")

    # NEW
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.hap_all.size(0)

    # NEW: deterministic start for a specific individual index
    def _choose_start_for_index(self, ind: int) -> int:
        if self.window_len is None or self.window_len == self.L_total:
            return 0
        max_start = self.L_total - self.window_len
        if max_start <= 0:
            return 0

        if self.window_mode == "first":
            return 0
        if self.window_mode == "middle":
            return max_start // 2
        if self.window_mode == "fixed":
            return int(max(0, min(self.fixed_start, max_start)))

        # window_mode == "random" but deterministic:
        # start = f(seed, epoch, ind)
        g = torch.Generator()
        g.manual_seed(self.seed + 1_000_003 * self.epoch + 9_176 * int(ind))
        return int(torch.randint(0, max_start + 1, (1,), generator=g).item())

    def __getitem__(self, idx: int | tuple[int, int]) -> HapBatch:
        # allow (ind, start) indexing for same-window batching
        if isinstance(idx, tuple):
            ind = int(idx[0])
            start = int(idx[1])
        else:
            ind = int(idx)
            start = None

        h = self.hap_all[ind]

        if self.window_len is not None and self.window_len < self.L_total:
            max_start = self.L_total - self.window_len
            if start is None:
                start = self._choose_start_for_index(ind)

            start = int(max(0, min(start, max_start)))
            end = start + self.window_len
            h = h[start:end]

        pad_mask = None
        if self.pad_id is not None:
            pad_mask = (h == self.pad_id)

        return HapBatch(hap=h, pad_mask=pad_mask)
