#!/usr/bin/env python3
"""
Generate box and whisker plots for test metrics across models.
"""
import argparse
from pathlib import Path
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make src/ importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def load_test_metrics(param_ids: list[str], base_dir: Path) -> dict[str, dict]:
    """Load test metrics for all param_ids."""
    metrics_data = {}
    
    for param_id in param_ids:
        metrics_path = base_dir / param_id / "final_test_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                metrics_data[param_id] = metrics
                print(f"Loaded metrics for {param_id}")
            except Exception as e:
                print(f"Error loading metrics for {param_id}: {e}")
        else:
            print(f"Metrics file not found for {param_id}: {metrics_path}")
    
    return metrics_data


def create_box_plots(metrics_data: dict[str, dict], output_dir: Path):
    """Create box and whisker plots for test metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize data by lambda values
    lambda_groups = {}
    
    for param_id, metrics in metrics_data.items():
        params = parse_param_id(param_id)
        lam = params.get('contrastive_lambda', 0.0)
        
        if lam not in lambda_groups:
            lambda_groups[lam] = {
                'auc_scores': [],
                'accuracies': [],
                'param_ids': []
            }
        
        lambda_groups[lam]['auc_scores'].append(metrics['final_test_auc'])
        lambda_groups[lam]['accuracies'].append(metrics['final_test_accuracy'])
        lambda_groups[lam]['param_ids'].append(param_id)
    
    # Prepare data for plotting
    lambda_values = sorted(lambda_groups.keys())
    auc_data = []
    acc_data = []
    lambda_labels = []
    
    for lam in lambda_values:
        group = lambda_groups[lam]
        auc_data.extend([(f"λ={lam:.1f}", score) for score in group['auc_scores']])
        acc_data.extend([(f"λ={lam:.1f}", score) for score in group['accuracies']])
        lambda_labels.append(f"λ={lam:.1f}")
    
    # Create box plots for AUC
    plt.figure(figsize=(10, 6))
    auc_values_by_group = [lambda_groups[lam]['auc_scores'] for lam in lambda_values]
    
    bp = plt.boxplot(auc_values_by_group, labels=lambda_labels, patch_artist=True, 
                     notch=True, showmeans=True)
    
    # Customize colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_values)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xlabel('Contrastive Lambda', fontsize=12)
    plt.ylabel('test AUC', fontsize=12)
    plt.title('test AUC Distribution Across Contrastive Lambda Values', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "test_auc_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create box plots for Accuracy
    plt.figure(figsize=(10, 6))
    acc_values_by_group = [lambda_groups[lam]['accuracies'] for lam in lambda_values]
    
    bp = plt.boxplot(acc_values_by_group, labels=lambda_labels, patch_artist=True, 
                     notch=True, showmeans=True)
    
    # Customize colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xlabel('Contrastive Lambda', fontsize=12)
    plt.ylabel('test Accuracy', fontsize=12)
    plt.title('test Accuracy Distribution Across Contrastive Lambda Values', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "test_accuracy_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics table
    summary_stats = {}
    for lam in lambda_values:
        group = lambda_groups[lam]
        auc_scores = np.array(group['auc_scores'])
        acc_scores = np.array(group['accuracies'])
        
        summary_stats[f"lambda_{lam:.1f}"] = {
            'auc_mean': float(np.mean(auc_scores)),
            'auc_std': float(np.std(auc_scores)),
            'auc_min': float(np.min(auc_scores)),
            'auc_max': float(np.max(auc_scores)),
            'acc_mean': float(np.mean(acc_scores)),
            'acc_std': float(np.std(acc_scores)),
            'acc_min': float(np.min(acc_scores)),
            'acc_max': float(np.max(acc_scores)),
            'n_models': len(auc_scores)
        }
    
    # Save summary statistics
    with open(output_dir / "test_metrics_summary.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Print summary
    print("\\ntest Metrics Summary:")
    print("-" * 80)
    print(f"{'Lambda':<8} {'AUC Mean±Std':<15} {'Acc Mean±Std':<15} {'N Models':<8}")
    print("-" * 80)
    for lam in lambda_values:
        stats = summary_stats[f"lambda_{lam:.1f}"]
        auc_str = f"{stats['auc_mean']:.4f}±{stats['auc_std']:.4f}"
        acc_str = f"{stats['acc_mean']:.4f}±{stats['acc_std']:.4f}"
        print(f"{lam:<8.1f} {auc_str:<15} {acc_str:<15} {stats['n_models']:<8}")
    
    print(f"\\nBox plots saved to:")
    print(f"  - {output_dir}/test_auc_boxplot.png")
    print(f"  - {output_dir}/test_accuracy_boxplot.png")
    print(f"  - {output_dir}/test_metrics_summary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing param_id subdirectories")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for box plots")
    parser.add_argument("--param_ids", nargs='+', required=True,
                       help="List of param_ids to analyze")
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    
    # Load test metrics
    metrics_data = load_test_metrics(args.param_ids, base_dir)
    
    if not metrics_data:
        print("No test metrics found!")
        return
    
    print(f"\\nFound metrics for {len(metrics_data)} models")
    
    # Create box plots
    create_box_plots(metrics_data, output_dir)


if __name__ == "__main__":
    main()