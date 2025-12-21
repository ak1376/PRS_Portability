# src/transformer/early_stopping.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class EarlyStopConfig:
    enabled: bool = True
    monitor: str = "val_total_loss"  # val_total_loss, val_mlm_loss, val_accuracy, val_auc
    mode: str = "min"                # "min" or "max"
    patience: int = 25
    min_delta: float = 1e-4
    burn_in: int = 0                 # epochs to wait before stopping


def is_improvement(curr: float, best: float, *, mode: str, min_delta: float) -> bool:
    if mode not in ("min", "max"):
        raise ValueError(f"early_stopping.mode must be 'min' or 'max', got {mode}")
    if np.isnan(curr):
        return False
    if np.isnan(best):
        return True
    if mode == "min":
        return curr < (best - float(min_delta))
    return curr > (best + float(min_delta))


class EarlyStopper:
    """
    Tracks best metric + CPU snapshot of best model weights.

    Intended use (matches snakemake_scripts/train_transformer.py):
        should_stop = stopper.step(curr_metric=metric, epoch=ep, model=model)
        ...
        stopper.restore_best(model)

    Also supports legacy positional:
        should_stop = stopper.step(epoch, curr, model)
    """

    def __init__(self, cfg: EarlyStopConfig):
        self.cfg = cfg
        self.best_metric: float = float("nan")
        self.best_epoch: int = 0
        self.best_state_cpu: Optional[dict[str, torch.Tensor]] = None
        self.bad_epochs: int = 0

    def step(self, *args, **kwargs) -> bool:
        """
        Returns:
            should_stop (bool)

        Accepts either:
          - step(curr_metric=..., epoch=..., model=...)
          - step(epoch, curr, model)   # legacy positional
        """
        if not self.cfg.enabled:
            return False

        # ---- Parse inputs ----
        if "curr_metric" in kwargs:
            curr = float(kwargs["curr_metric"])
            epoch = int(kwargs["epoch"])
            model = kwargs["model"]
        elif len(args) == 3:
            epoch = int(args[0])
            curr = float(args[1])
            model = args[2]
        else:
            raise TypeError(
                "EarlyStopper.step expects (curr_metric=..., epoch=..., model=...) "
                "or legacy positional (epoch, curr, model)."
            )

        # ---- Check improvement ----
        improved = is_improvement(curr, self.best_metric, mode=self.cfg.mode, min_delta=self.cfg.min_delta)
        if improved:
            self.best_metric = float(curr)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            # store on CPU to avoid GPU memory growth
            self.best_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False  # don't stop if improved

        # ---- Not improved ----
        if epoch >= int(self.cfg.burn_in):
            self.bad_epochs += 1
            if self.bad_epochs >= int(self.cfg.patience):
                return True

        return False

    def restore_best(self, model: torch.nn.Module) -> bool:
        if not self.cfg.enabled:
            return False
        if self.best_state_cpu is None:
            return False
        model.load_state_dict(self.best_state_cpu)
        return True
