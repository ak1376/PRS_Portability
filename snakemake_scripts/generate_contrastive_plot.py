#!/usr/bin/env python3
"""
Generate contrastive geometry plot from a trained transformer model.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

# Make src/ importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def plot_contrastive_geometry(
    out_path: Path,
    epochs: list,
    pos_cos: list,
    neg_cos: list,
    perm_neg_cos: list = None,
):
    """Plot contrastive geometry metrics over training."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.plot(epochs, pos_cos, label="pos cosine", marker="o")
    ax.plot(epochs, neg_cos, label="neg cosine", marker="s")
    if perm_neg_cos is not None:
        ax.plot(epochs, perm_neg_cos, label="perm neg cosine", marker="^")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Contrastive Geometry")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_losses_csv(csv_path: Path):
    """Load training losses from CSV file."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    epochs = df['epoch'].tolist()
    pos_cos = df['ctr_pos_cos'].tolist() if 'ctr_pos_cos' in df.columns else []
    neg_cos = df['ctr_neg_cos'].tolist() if 'ctr_neg_cos' in df.columns else []
    perm_neg_cos = df['ctr_perm_neg_cos'].tolist() if 'ctr_perm_neg_cos' in df.columns else []
    
    return epochs, pos_cos, neg_cos, perm_neg_cos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--losses_csv", type=str, required=True, help="Path to train_losses.csv")
    parser.add_argument("--output", type=str, required=True, help="Output plot path")
    args = parser.parse_args()
    
    losses_path = Path(args.losses_csv)
    output_path = Path(args.output)
    
    if not losses_path.exists():
        print(f"Error: losses file not found: {losses_path}")
        sys.exit(1)
    
    # Load training metrics
    epochs, pos_cos, neg_cos, perm_neg_cos = load_losses_csv(losses_path)
    
    # Check if contrastive data exists
    if not pos_cos or not neg_cos:
        print("Warning: No contrastive data found in losses CSV. Creating empty placeholder.")
        output_path.touch()
        return
    
    # Generate plot
    perm_ok = perm_neg_cos and not np.all(np.isnan(perm_neg_cos))
    plot_contrastive_geometry(
        output_path,
        epochs,
        pos_cos,
        neg_cos,
        perm_neg_cos if perm_ok else None,
    )
    
    print(f"Generated contrastive plot: {output_path}")


if __name__ == "__main__":
    main()