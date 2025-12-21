# src/transformer/io_utils.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def write_losses_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows with union-of-keys header (robust if some rows have extra columns)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")

    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def save_checkpoint(
    out_model: Path,
    *,
    model_state_dict: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> None:
    out_model = Path(out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model_state_dict, "config": config}, out_model)


def safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        return float(x)
    except Exception:
        return None
