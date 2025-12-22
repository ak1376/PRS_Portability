#!/usr/bin/env python3
"""
Generate comparative plots across all trained transformer models.

Robust to:
- param_id strings with extra suffixes after lam (e.g. _lam0p1_pmc0p2_pms0p75)
- varying CSV filenames (auto-detect)
- varying column names (substring matching)
- tiny number of models / missing columns (prints explicit warnings)

Outputs:
- training_loss_comparison.png
- validation_loss_comparison.png
- contrastive_pos.png
- contrastive_neg.png
- contrastive_perm_neg.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import re
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make src/ importable (kept for consistency with your repo pattern)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------
# Parsing helpers
# -----------------------------
RE_LAM = re.compile(r"(?:^|_)lam([0-9ep\-m]+)(?:_|$)")

def tok_to_float(tok: str) -> float:
    # "0p1" -> 0.1, "1e-4" -> 1e-4, "m" indicates negative
    return float(tok.replace("p", ".").replace("m", "-"))

def extract_lambda(param_id: str) -> float | None:
    m = RE_LAM.search(param_id)
    if not m:
        return None
    return tok_to_float(m.group(1))

def lam_label(lam: float | None) -> str:
    return "λ=NA" if lam is None else f"λ={lam:.2f}"


# -----------------------------
# CSV discovery + column picking
# -----------------------------
def find_model_csv(model_dir: Path) -> Path | None:
    """
    Find a CSV in the model_dir that likely contains epoch-wise losses/metrics.
    Prefer train_losses.csv or losses.csv if present; otherwise fall back to newest *loss*.csv.
    """
    preferred = ["train_losses.csv", "losses.csv", "metrics.csv"]
    for name in preferred:
        p = model_dir / name
        if p.exists():
            return p

    patterns = ["*loss*.csv", "*metric*.csv", "*.csv"]
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(model_dir.glob(pat))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    # Prefer files with "loss" or "metric" in name, then newest
    candidates.sort(
        key=lambda p: (
            ("loss" not in p.name.lower() and "metric" not in p.name.lower()),
            -p.stat().st_mtime,
        )
    )
    return candidates[0]

def load_csv(csv_path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed reading {csv_path}: {e}")
        return None

def pick_col(df: pd.DataFrame, exact: list[str], must_contain: tuple[str, ...] = (), any_contain: tuple[str, ...] = ()) -> str | None:
    cols = list(df.columns)

    # exact names first
    for c in exact:
        if c in cols:
            return c

    # substring heuristics
    for c in cols:
        cl = c.lower()
        if must_contain and not all(s in cl for s in must_contain):
            continue
        if any_contain and not any(s in cl for s in any_contain):
            continue
        return c
    return None

def finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


# -----------------------------
# Plotters
# -----------------------------
def plot_loss_comparison(
    model_data: dict[str, dict[str, Any]],
    out_path: Path,
    which: str,  # "train" or "val"
) -> None:
    plt.figure(figsize=(12, 6))

    n_plotted = 0

    for pid, info in model_data.items():
        df: pd.DataFrame = info["df"]
        lam: float | None = info["lam"]

        epoch_col = info["epoch_col"]
        if epoch_col is None:
            print(f"[WARN] {pid}: no epoch/step column")
            continue

        # Find loss col for train/val
        if which == "train":
            # common: train_mlm_loss, train_loss, mlm_train_loss, etc
            loss_col = pick_col(
                df,
                exact=["train_mlm_loss", "train_loss"],
                must_contain=(),
                any_contain=("loss", "train", "mlm"),
            )
            title = "Training Loss Across Models"
        else:
            loss_col = pick_col(
                df,
                exact=["val_mlm_loss", "valid_mlm_loss", "val_loss", "valid_loss"],
                must_contain=(),
                any_contain=("loss", "val", "valid", "mlm"),
            )
            title = "Validation Loss Across Models"

        if loss_col is None:
            print(f"[WARN] {pid}: no {which} loss column found (cols={list(df.columns)})")
            continue

        x, y = finite_xy(df[epoch_col].to_numpy(), df[loss_col].to_numpy())
        if len(x) == 0:
            print(f"[WARN] {pid}: {loss_col} has no finite values")
            continue

        plt.plot(x, y, linewidth=2, label=lam_label(lam))
        n_plotted += 1

    if n_plotted == 0:
        plt.axis("off")
        plt.text(
            0.5, 0.5,
            f"No valid {which} loss series found.\nSee console warnings for missing columns/files.",
            ha="center", va="center", fontsize=14
        )
    else:
        plt.title(title, fontsize=16)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote {out_path} (series plotted: {n_plotted})")


def plot_contrastive_comparison(
    model_data: dict[str, dict[str, Any]],
    out_path: Path,
    kind: str,  # "pos", "neg", "perm_neg"
) -> None:
    title_map = {
        "pos": "Contrastive Positive Cosine Similarity Across Models",
        "neg": "Contrastive Negative Cosine Similarity Across Models",
        "perm_neg": "Contrastive Hard Negative Cosine Similarity Across Models",
    }
    ylabel_map = {
        "pos": "Positive Cosine Similarity",
        "neg": "Negative Cosine Similarity",
        "perm_neg": "Hard Negative Cosine Similarity",
    }
    token_map = {
        "pos": ("pos", "positive"),
        "neg": ("neg", "negative"),
        "perm_neg": ("perm", "hard", "permutation"),
    }

    plt.figure(figsize=(12, 6))
    n_plotted = 0

    for pid, info in model_data.items():
        df: pd.DataFrame = info["df"]
        lam: float | None = info["lam"]
        epoch_col = info["epoch_col"]

        # Only plot for contrastive models (lam > 0)
        if lam is None or lam <= 0:
            continue

        if epoch_col is None:
            continue

        # Prefer your old fixed names, but fall back to substring search
        exact_candidates = {
            "pos": ["ctr_pos_cos", "ctr_pos_cosine"],
            "neg": ["ctr_neg_cos", "ctr_neg_cosine"],
            "perm_neg": ["ctr_perm_neg_cos", "ctr_perm_neg_cosine"],
        }[kind]

        # Must contain cosine-ish token AND the kind token
        kind_tokens = token_map[kind]
        cos_col = pick_col(
            df,
            exact=exact_candidates,
            must_contain=("cos",),  # ensure cosine-ish
            any_contain=kind_tokens,
        )

        # If that failed because column uses "cosine" not "cos", try cosine too
        if cos_col is None:
            for c in df.columns:
                cl = c.lower()
                if ("cos" in cl or "cosine" in cl) and any(t in cl for t in kind_tokens):
                    cos_col = c
                    break

        if cos_col is None:
            print(f"[WARN] {pid}: no contrastive {kind} cosine column found (cols={list(df.columns)})")
            continue

        x, y = finite_xy(df[epoch_col].to_numpy(), df[cos_col].to_numpy())
        if len(x) == 0:
            print(f"[WARN] {pid}: {cos_col} has no finite values")
            continue

        plt.plot(x, y, linewidth=2, label=lam_label(lam))
        n_plotted += 1

    if n_plotted == 0:
        plt.axis("off")
        plt.text(
            0.5, 0.5,
            f"No valid contrastive ({kind}) series found.\n"
            f"Either lam<=0 for all models, or the CSV lacks the cosine columns.\n"
            f"See console warnings.",
            ha="center", va="center", fontsize=14
        )
    else:
        plt.title(title_map[kind], fontsize=16)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(ylabel_map[kind], fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote {out_path} (series plotted: {n_plotted})")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--losses_dir", type=str, required=True,
        help="Base directory containing param_id subdirectories (each with a CSV of epoch-wise logs)."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for comparison plots."
    )
    parser.add_argument(
        "--param_ids", nargs="+", required=True,
        help="List of param_ids to compare."
    )
    args = parser.parse_args()

    losses_dir = Path(args.losses_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load per-model CSVs
    model_data: dict[str, dict[str, Any]] = {}

    for pid in args.param_ids:
        model_dir = losses_dir / pid
        if not model_dir.exists():
            print(f"[WARN] Missing model dir: {model_dir}")
            continue

        csv_path = find_model_csv(model_dir)
        if csv_path is None:
            print(f"[WARN] {pid}: no CSV found in {model_dir}")
            continue

        df = load_csv(csv_path)
        if df is None or df.empty:
            print(f"[WARN] {pid}: CSV empty/unreadable: {csv_path}")
            continue

        epoch_col = pick_col(df, exact=["epoch", "Epoch", "step", "Step"], any_contain=("epoch", "step"))
        lam = extract_lambda(pid)

        model_data[pid] = {
            "df": df,
            "csv_path": csv_path,
            "epoch_col": epoch_col,
            "lam": lam,
        }

        print(f"[OK] {pid}: loaded {csv_path.name} rows={len(df)} epoch_col={epoch_col} lam={lam_label(lam)}")

    if not model_data:
        print("[ERROR] No valid model CSVs found.")
        return

    # Plots
    plot_loss_comparison(model_data, output_dir / "training_loss_comparison.png", which="train")
    plot_loss_comparison(model_data, output_dir / "validation_loss_comparison.png", which="val")

    plot_contrastive_comparison(model_data, output_dir / "contrastive_pos.png", kind="pos")
    plot_contrastive_comparison(model_data, output_dir / "contrastive_neg.png", kind="neg")
    plot_contrastive_comparison(model_data, output_dir / "contrastive_perm_neg.png", kind="perm_neg")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
