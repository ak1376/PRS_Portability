#!/usr/bin/env python3
"""
Generate box-and-whisker plots for test metrics across models.

Robust to:
- param_id strings with extra suffixes (e.g. _lam0p0_pmc0p2_pms0p75)
- missing / non-finite metric values
- tiny group sizes (no notches unless enough points)

Always writes outputs (png/png/json) so Snakemake won't fail due to missing files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json
import re
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Make src/ importable (kept for consistency with your repo pattern)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _tok_to_float(tok: str) -> float:
    # Your encoding: "0p1" -> 0.1, "1e-4" -> 1e-4, "m" indicates negative
    return float(tok.replace("p", ".").replace("m", "-"))


# Flexible patterns: find tokens anywhere in the id
RE_LAM = re.compile(r"(?:^|_)lam([0-9ep\-m]+)(?:_|$)")
RE_DO  = re.compile(r"(?:^|_)do([0-9ep\-m]+)(?:_|$)")
RE_LR  = re.compile(r"(?:^|_)lr([0-9ep\-m]+)(?:_|$)")
RE_POOL = re.compile(r"(?:^|_)pool([A-Za-z0-9]+)(?:_|$)")

# If you still want these (optional)
RE_E = re.compile(r"(?:^|_)e(\d+)(?:_|$)")
RE_H = re.compile(r"(?:^|_)h(\d+)(?:_|$)")
RE_L = re.compile(r"(?:^|_)L(\d+)(?:_|$)")
RE_FF = re.compile(r"(?:^|_)ff(\d+)(?:_|$)")


def parse_param_id(param_id: str) -> dict[str, Any]:
    """
    Parse a param_id directory name. Only 'lam' is required for grouping.
    Everything else is best-effort.
    """
    out: dict[str, Any] = {}

    m = RE_LAM.search(param_id)
    if m:
        out["contrastive_lambda_token"] = m.group(1)
        out["contrastive_lambda"] = _tok_to_float(m.group(1))

    m = RE_DO.search(param_id)
    if m:
        out["dropout"] = _tok_to_float(m.group(1))

    m = RE_LR.search(param_id)
    if m:
        out["lr"] = _tok_to_float(m.group(1))

    m = RE_POOL.search(param_id)
    if m:
        out["pool"] = m.group(1)

    # Optional architectural fields (best-effort)
    m = RE_E.search(param_id)
    if m:
        out["embed_dim"] = int(m.group(1))
    m = RE_H.search(param_id)
    if m:
        out["n_heads"] = int(m.group(1))
    m = RE_L.search(param_id)
    if m:
        out["n_layers"] = int(m.group(1))
    m = RE_FF.search(param_id)
    if m:
        out["ff_dim"] = int(m.group(1))

    return out


def _finite_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def load_test_metrics(param_ids: list[str], base_dir: Path) -> dict[str, dict[str, Any]]:
    """Load test metrics for all param_ids."""
    metrics_data: dict[str, dict[str, Any]] = {}

    for param_id in param_ids:
        metrics_path = base_dir / param_id / "final_test_metrics.json"
        if not metrics_path.exists():
            print(f"[WARN] Metrics file not found: {metrics_path}")
            continue

        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            metrics_data[param_id] = metrics
            print(f"[OK] Loaded metrics for {param_id}")
        except Exception as e:
            print(f"[WARN] Error loading {metrics_path}: {e}")

    return metrics_data


def _write_empty_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    """Write placeholder outputs so Snakemake doesn't fail."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder AUC plot
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.text(
        0.5, 0.5,
        "No valid metrics found to plot.\n(See test_metrics_summary.json for details.)",
        ha="center", va="center"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "test_auc_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Placeholder Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.text(
        0.5, 0.5,
        "No valid metrics found to plot.\n(See test_metrics_summary.json for details.)",
        ha="center", va="center"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "test_accuracy_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(output_dir / "test_metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[WARN] Wrote placeholder outputs (no valid metrics).")


def create_box_plots(metrics_data: dict[str, dict[str, Any]], output_dir: Path) -> None:
    """Create box-and-whisker plots for test metrics across contrastive lambda groups."""
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, dict[str, Any]] = {}

    bad_parse = 0
    missing_auc = 0
    missing_acc = 0

    for param_id, metrics in metrics_data.items():
        params = parse_param_id(param_id)

        if "contrastive_lambda" not in params:
            bad_parse += 1
            continue

        lam_val = float(params["contrastive_lambda"])
        lam_label = f"λ={lam_val:g}"  # stable label: 0.1, 0.05, 1e-06, etc.

        auc = _finite_float(metrics.get("final_test_auc"))
        acc = _finite_float(metrics.get("final_test_accuracy"))

        if auc is None:
            missing_auc += 1
        if acc is None:
            missing_acc += 1
        if auc is None or acc is None:
            continue

        if lam_label not in groups:
            groups[lam_label] = {"lam_val": lam_val, "auc": [], "acc": [], "param_ids": []}

        groups[lam_label]["auc"].append(auc)
        groups[lam_label]["acc"].append(acc)
        groups[lam_label]["param_ids"].append(param_id)

    if not groups:
        summary = {
            "status": "no_valid_metrics",
            "n_models_loaded": len(metrics_data),
            "n_groups": 0,
            "bad_parse_no_lam": bad_parse,
            "missing_or_nonfinite_auc": missing_auc,
            "missing_or_nonfinite_acc": missing_acc,
            "note": "Check that param_id strings contain `_lam...` and metrics JSON contains "
                    "`final_test_auc` and `final_test_accuracy` with finite numeric values.",
        }
        print("[ERROR] No valid metrics found after filtering missing/NaN values.")
        print(f"        bad_parse={bad_parse} missing_auc={missing_auc} missing_acc={missing_acc}")
        _write_empty_outputs(output_dir, summary)
        return

    labels = sorted(groups.keys(), key=lambda k: groups[k]["lam_val"])
    auc_values = [groups[k]["auc"] for k in labels]
    acc_values = [groups[k]["acc"] for k in labels]

    min_n = min(len(v) for v in auc_values)
    use_notch = (min_n >= 5)

    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    rng = np.random.default_rng(0)

    def _plot(values_by_group: list[list[float]], ylabel: str, title: str, out_name: str) -> None:
        plt.figure(figsize=(10, 6))

        bp = plt.boxplot(
            values_by_group,
            labels=labels,
            patch_artist=True,
            notch=use_notch,
            showmeans=True,
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Overlay points (so 1–2 models don't look "broken")
        for i, vals in enumerate(values_by_group, start=1):
            x = np.full(len(vals), i, dtype=float)
            jitter = (rng.random(len(vals)) - 0.5) * 0.12
            plt.scatter(x + jitter, vals, s=18, alpha=0.7)

        plt.xlabel("Contrastive Lambda", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / out_name, dpi=150, bbox_inches="tight")
        plt.close()

    _plot(auc_values, "test AUC", "test AUC Distribution Across Contrastive Lambda Values", "test_auc_boxplot.png")
    _plot(acc_values, "test Accuracy", "test Accuracy Distribution Across Contrastive Lambda Values", "test_accuracy_boxplot.png")

    # Summary stats JSON
    summary: dict[str, Any] = {"status": "ok", "groups": {}}
    for lab in labels:
        auc = np.asarray(groups[lab]["auc"], dtype=float)
        acc = np.asarray(groups[lab]["acc"], dtype=float)
        summary["groups"][lab] = {
            "lambda": float(groups[lab]["lam_val"]),
            "auc_mean": float(np.mean(auc)),
            "auc_std": float(np.std(auc)),
            "auc_min": float(np.min(auc)),
            "auc_max": float(np.max(auc)),
            "acc_mean": float(np.mean(acc)),
            "acc_std": float(np.std(acc)),
            "acc_min": float(np.min(acc)),
            "acc_max": float(np.max(acc)),
            "n_models": int(len(auc)),
            "param_ids": groups[lab]["param_ids"],
        }

    summary["n_models_loaded"] = len(metrics_data)
    summary["n_groups"] = len(labels)
    summary["bad_parse_no_lam"] = bad_parse
    summary["missing_or_nonfinite_auc"] = missing_auc
    summary["missing_or_nonfinite_acc"] = missing_acc

    with open(output_dir / "test_metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs:")
    print(f"  - {output_dir}/test_auc_boxplot.png")
    print(f"  - {output_dir}/test_accuracy_boxplot.png")
    print(f"  - {output_dir}/test_metrics_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base directory containing param_id subdirectories")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for box plots")
    parser.add_argument("--param_ids", nargs="+", required=True,
                        help="List of param_ids to analyze")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)

    metrics_data = load_test_metrics(args.param_ids, base_dir)
    if not metrics_data:
        summary = {
            "status": "no_metrics_files",
            "n_models_loaded": 0,
            "note": "No final_test_metrics.json files were found for the provided param_ids/base_dir."
        }
        _write_empty_outputs(output_dir, summary)
        return

    print(f"\nFound metrics JSONs for {len(metrics_data)} models")
    create_box_plots(metrics_data, output_dir)


if __name__ == "__main__":
    main()
