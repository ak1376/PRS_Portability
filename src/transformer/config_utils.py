# src/transformer/config_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def maybe_load_cfg(cfg_path: str | None) -> dict:
    if cfg_path is None:
        return {}
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(f"--config not found: {cfg_path}")
    return yaml.safe_load(p.read_text()) or {}


def get_nested(cfg: dict, keys: list[str], default):
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def pick(cli_val, yaml_val, default):
    return default if (cli_val is None and yaml_val is None) else (yaml_val if cli_val is None else cli_val)
