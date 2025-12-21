#!/usr/bin/env python3
"""
Generate comparative plots across all trained transformer models.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make src/ importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_losses_csv(csv_path: Path):
    """Load training losses from CSV file."""
    if not csv_path.exists():
        print(f"Warning: losses file not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def parse_param_id(param_id: str) -> dict:
    """Parse parameter ID to extract hyperparameters."""
    import re
    m = re.match(
        r"tx_e(\d+)_h(\d+)_L(\d+)_ff(\d+)_do([0-9ep\-m]+)_pool([A-Za-z0-9]+)_lr([0-9ep\-m]+)_lam([0-9ep\-m]+)$",
        param_id,
    )
    if not m:
        return {}

    dropout = float(m.group(5).replace("p", ".").replace("m", "-"))
    pool    = m.group(6)
    lr      = float(m.group(7).replace("p", ".").replace("m", "-"))
    lam     = float(m.group(8).replace("p", ".").replace("m", "-"))

    return dict(
        embed_dim=int(m.group(1)),
        n_heads=int(m.group(2)),
        n_layers=int(m.group(3)),
        ff_dim=int(m.group(4)),
        dropout=dropout,
        pool=pool,
        lr=lr,
        contrastive_lambda=lam,
    )


def plot_validation_accuracy(data_dict: dict, output_path: Path):
    """Plot validation masked loss across all models."""
    plt.figure(figsize=(10, 6))
    
    for param_id, data in data_dict.items():
        if data is None or 'val_mlm_loss' not in data:
            print(f"Warning: No val_mlm_loss column found for {param_id}")
            continue
        
        params = parse_param_id(param_id)
        lam = params.get('contrastive_lambda', 0.0)
        
        # Plot validation loss (lower is better)
        label = f"λ={lam:.1f}"
        epochs = data['epoch']
        val_losses = data['val_mlm_loss']
        plt.plot(epochs, val_losses, marker='o', label=label, linewidth=2, markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation MLM Loss', fontsize=12)
    plt.title('Validation Masked Language Model Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Generated validation loss plot: {output_path}")


def plot_training_loss(data_dict: dict, output_path: Path):
    """Plot training masked loss across all models."""
    plt.figure(figsize=(10, 6))
    
    for param_id, data in data_dict.items():
        if data is None or 'train_mlm_loss' not in data:
            print(f"Warning: No train_mlm_loss column found for {param_id}")
            continue
        
        params = parse_param_id(param_id)
        lam = params.get('contrastive_lambda', 0.0)
        
        # Plot training loss (lower is better)
        label = f"λ={lam:.1f}"
        epochs = data['epoch']
        train_losses = data['train_mlm_loss']
        plt.plot(epochs, train_losses, marker='o', label=label, linewidth=2, markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training MLM Loss', fontsize=12)
    plt.title('Training Masked Language Model Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Generated training loss plot: {output_path}")


def plot_contrastive_metrics(data_dict: dict, output_dir: Path, metric_name: str, ylabel: str):
    """Plot contrastive metrics across all models."""
    plt.figure(figsize=(10, 6))
    
    for param_id, data in data_dict.items():
        if data is None or metric_name not in data:
            continue
        
        params = parse_param_id(param_id)
        lam = params.get('contrastive_lambda', 0.0)
        
        # Skip models with lambda=0 for contrastive plots
        if lam <= 0:
            continue
        
        label = f"λ={lam:.1f}"
        epochs = data['epoch']
        metric_values = data[metric_name]
        plt.plot(epochs, metric_values, marker='o', label=label, linewidth=2, markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'Contrastive {ylabel} Across Models', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create output filename based on metric
    metric_clean = metric_name.replace('ctr_', '').replace('_cos', '')
    output_path = output_dir / f"contrastive_{metric_clean}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Generated {ylabel} plot: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--losses_dir", type=str, required=True, 
                       help="Base directory containing param_id subdirectories with train_losses.csv")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Output directory for comparison plots")
    parser.add_argument("--param_ids", nargs='+', required=True,
                       help="List of param_ids to compare")
    args = parser.parse_args()
    
    losses_dir = Path(args.losses_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all loss CSV files
    data_dict = {}
    for param_id in args.param_ids:
        csv_path = losses_dir / param_id / "train_losses.csv"
        data = load_losses_csv(csv_path)
        if data is not None:
            data_dict[param_id] = data
            print(f"Loaded losses for {param_id}: {len(data['epoch'])} epochs")
        else:
            print(f"Could not load losses for {param_id}")
    
    if not data_dict:
        print("No valid loss files found!")
        return
    
    # Generate plots
    print("\\nGenerating comparison plots...")
    
    # 1. Training loss
    plot_training_loss(data_dict, output_dir / "training_loss_comparison.png")
    
    # 2. Validation loss
    plot_validation_accuracy(data_dict, output_dir / "validation_loss_comparison.png")
    
    # 3-5. Contrastive metrics (only for models with lambda > 0)
    contrastive_metrics = [
        ('ctr_pos_cos', 'Positive Cosine Similarity'),
        ('ctr_neg_cos', 'Negative Cosine Similarity'), 
        ('ctr_perm_neg_cos', 'Hard Negative Cosine Similarity'),
    ]
    
    for metric_name, ylabel in contrastive_metrics:
        plot_contrastive_metrics(data_dict, output_dir, metric_name, ylabel)
    
    print(f"\\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()