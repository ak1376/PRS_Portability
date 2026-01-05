# src/transformer/window_batching.py
from __future__ import annotations

import math
import torch
from torch.utils.data import Sampler


class SameWindowBatchSampler(Sampler[list[tuple[int, int]]]):
    """
    Produces batches where EVERY element shares the same window start `s`.

    Each yielded batch is a list of (local_index, start) tuples.
    The dataset must support __getitem__((ind, start)).
    """

    def __init__(
        self,
        *,
        n_samples: int,
        L_total: int,
        window_len: int,
        batch_size: int,
        seed: int,
        drop_last: bool = True,
    ):
        self.n_samples = int(n_samples)
        self.L_total = int(L_total)
        self.window_len = int(window_len)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        if self.window_len <= 0:
            raise ValueError("window_len must be > 0")
        if self.window_len > self.L_total:
            raise ValueError(f"window_len ({self.window_len}) > L_total ({self.L_total})")

        self.max_start = self.L_total - self.window_len
        self.n_starts = self.max_start + 1

        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self.n_samples // self.batch_size
        return math.ceil(self.n_samples / self.batch_size)

    def __iter__(self):
        # Deterministic per-epoch randomness
        g = torch.Generator().manual_seed(self.seed + self._epoch)

        # Randomize sample order
        perm = torch.randperm(self.n_samples, generator=g).tolist()

        # Chunk into batches
        for i in range(0, len(perm), self.batch_size):
            batch_inds = perm[i : i + self.batch_size]
            if self.drop_last and len(batch_inds) < self.batch_size:
                continue

            # Pick ONE start for the whole batch
            if self.n_starts <= 1:
                s = 0
            else:
                s = int(torch.randint(0, self.n_starts, (1,), generator=g).item())

            yield [(int(ind), int(s)) for ind in batch_inds]
