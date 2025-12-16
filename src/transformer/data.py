from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

def collate_happairbatch(items):
    hap1 = torch.stack([x.hap1 for x in items], dim=0)
    hap2 = torch.stack([x.hap2 for x in items], dim=0)
    genotype = torch.stack([x.genotype for x in items], dim=0)

    pad_mask = None
    if items[0].pad_mask is not None:
        pad_mask = torch.stack([x.pad_mask for x in items], dim=0)

    return type(items[0])(hap1=hap1, hap2=hap2, genotype=genotype, pad_mask=pad_mask)


@dataclass
class HapPairBatch:
    hap1: torch.Tensor
    hap2: torch.Tensor
    genotype: torch.Tensor
    pad_mask: torch.Tensor | None = None


class HapPairDataset(Dataset):
    """
    Stores diploid individuals as two haplotype matrices:
      hap1_all: (N, L_total) long
      hap2_all: (N, L_total) long

    Optionally returns a contiguous window of length window_len.
    """
    def __init__(
        self,
        hap1_all: torch.Tensor,
        hap2_all: torch.Tensor,
        pad_id: int | None = None,
        *,
        window_len: int | None = None,          # e.g. 512, 1024, 2048
        window_mode: str = "random",            # "random" | "first" | "middle" | "fixed"
        fixed_start: int = 0,                   # used if window_mode="fixed"
        rng: torch.Generator | None = None,     # optional reproducibility
    ):
        assert hap1_all.shape == hap2_all.shape
        self.hap1_all = hap1_all.long()
        self.hap2_all = hap2_all.long()
        self.pad_id = pad_id

        self.L_total = self.hap1_all.shape[1]
        self.window_len = window_len
        self.window_mode = window_mode
        self.fixed_start = fixed_start
        self.rng = rng  # can be None

        if self.window_len is not None:
            if self.window_len <= 0:
                raise ValueError("window_len must be > 0")
            if self.window_len > self.L_total:
                raise ValueError(f"window_len ({self.window_len}) > L_total ({self.L_total})")
            if self.window_mode not in {"random", "first", "middle", "fixed"}:
                raise ValueError(f"Unknown window_mode: {self.window_mode}")

    def __len__(self) -> int:
        return self.hap1_all.size(0)

    def _choose_start(self) -> int:
        if self.window_len is None or self.window_len == self.L_total:
            return 0

        max_start = self.L_total - self.window_len

        if self.window_mode == "first":
            return 0
        if self.window_mode == "middle":
            return max_start // 2
        if self.window_mode == "fixed":
            return int(max(0, min(self.fixed_start, max_start)))
        # random
        if max_start == 0:
            return 0
        if self.rng is None:
            return int(torch.randint(0, max_start + 1, (1,)).item())
        return int(torch.randint(0, max_start + 1, (1,), generator=self.rng).item())

    def __getitem__(self, idx: int) -> HapPairBatch:
        h1 = self.hap1_all[idx]
        h2 = self.hap2_all[idx]

        # Windowing happens here
        if self.window_len is not None and self.window_len < self.L_total:
            s = self._choose_start()
            e = s + self.window_len
            h1 = h1[s:e]
            h2 = h2[s:e]

        g = (h1 + h2).clamp_min(0).clamp_max(2)

        pad_mask = None
        if self.pad_id is not None:
            pad_mask = (h1 == self.pad_id) | (h2 == self.pad_id)

        return HapPairBatch(hap1=h1, hap2=h2, genotype=g, pad_mask=pad_mask)
