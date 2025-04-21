import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.stats import t
from scipy import stats

def main(args):
    # Load data
    G = np.load(args.genotype_matrix)
    merged_sorted = pd.read_csv(args.sorted_phenotype)
    trait_df = pd.read_csv(args.trait_info)
    y = merged_sorted["phenotype"].values
    X = G.astype(float)
    num_inds, num_sites = G.shape

    ##############################################
    # Center phenotype and genotypes
    ##############################################
    y_centered = y - y.mean()
    X_centered = X - X.mean(axis=0)

    numerator = (X_centered * y_centered[:, None]).sum(axis=0)
    denominator = (X_centered ** 2).sum(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = numerator / denominator
        slopes[np.isnan(slopes)] = 0.0

    y_var = np.var(y, ddof=1)
    residual_var = y_var * (1 - (numerator**2 / (denominator * (y_centered**2).sum())))
    df = num_inds - 2
    stderr = np.sqrt(residual_var / denominator)
    t_stats = slopes / stderr
    pvals = 2 * t.sf(np.abs(t_stats), df)
    pvals[np.isnan(pvals)] = 1.0

    ##############################################
    # Determine causal SNPs
    ##############################################
    causal_site_ids = set(trait_df["site_id"])
    causal_mask = np.zeros(num_sites, dtype=bool)
    causal_mask[list(causal_site_ids)] = True

    ##############################################
    # Bonferroni threshold
    ##############################################
    bonf = 0.05 / m
    sig_mask = pvals < bonf

    # causal indicator
    causal_mask = np.zeros(m, bool)
    causal_mask[trait_df["site_id"].values] = True

    # Colours & sizes
    point_colors = np.full(m, "#333333", dtype=object)          # dark gray
    point_colors[sig_mask & ~causal_mask] = "#1f77b4"           # spurious
    point_colors[sig_mask &  causal_mask] = "#d62728"           # causal
    point_sizes  = np.full(m, 6)
    point_sizes[sig_mask & ~causal_mask] = 12
    point_sizes[sig_mask &  causal_mask] = 20

    # ------------------------------------------------- Manhattan plot
    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(m), -np.log10(pvals),
                c=point_colors, s=point_sizes, alpha=0.8, lw=0)

    # outline causal significant
    causal_sig = np.where(sig_mask & causal_mask)[0]
    if causal_sig.size:
        plt.scatter(causal_sig, -np.log10(pvals[causal_sig]),
                    c="#d62728", s=point_sizes[causal_sig],
                    edgecolors="black", lw=0.5)

    plt.axhline(-np.log10(bonf), ls="--", c="gray", lw=1.2)
    plt.text(m*0.95, -np.log10(bonf)+0.2, "Bonferroni", ha="right", va="bottom", color="gray")

    plt.title(f"GWAS with Top {a.num_pcs} PCs", fontsize=14)
    plt.xlabel("SNP index"); plt.ylabel("-log₁₀(p)")
    plt.tight_layout(); plt.savefig(a.manhattan_plot); plt.close()

    # ----------------------------------------- AF‑difference plot
    pop = merged["population"].values
    afr, eur = (pop == "AFR"), (pop == "EUR")
    AF_diff = np.abs(G[afr].sum(0)/(2*afr.sum()) - G[eur].sum(0)/(2*eur.sum()))
    norm = Normalize(vmin=AF_diff.min(), vmax=AF_diff.max())

    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(m), -np.log10(pvals),
                c=AF_diff, cmap="coolwarm", norm=norm, s=6, lw=0, alpha=0.8)
    plt.scatter(np.where(sig_mask)[0], -np.log10(pvals[sig_mask]),
                c=AF_diff[sig_mask], cmap="coolwarm", norm=norm,
                s=20, edgecolors="black", lw=0.3)
    plt.axhline(-np.log10(bonf), ls="--", c="gray")
    plt.colorbar(label="|AF diff (AFR – EUR)|")
    plt.title("GWAS with PCs: AF‑diff colouring")
    plt.xlabel("SNP index"); plt.ylabel("-log₁₀(p)")
    plt.tight_layout(); plt.savefig(a.af_diff_plot); plt.close()

    # --------------------------------------------------- QQ plot
    exp = -np.log10(np.linspace(1/(m+1), 1, m))
    obs = -np.log10(np.sort(pvals))
    lam = np.median(stats.chi2.isf(np.sort(pvals), 1)) / stats.chi2.ppf(0.5, 1)

    plt.figure(figsize=(6, 6))
    plt.scatter(exp, obs, s=8, alpha=0.6, color="#1f77b4")
    mx = max(exp.max(), obs.max()); plt.plot([0, mx], [0, mx], ls="--", c="gray")
    plt.title(f"QQ Plot (λ={lam:.2f})")
    plt.xlabel("Expected -log₁₀(p)"); plt.ylabel("Observed -log₁₀(p)")
    plt.tight_layout(); plt.savefig(a.qq_plot); plt.close()

    ##############################################
    # QQ Plot of GWAS p-values
    ##############################################
    observed_pvals = np.sort(pvals)
    expected_pvals = np.linspace(1 / (num_sites + 1), 1, num_sites)
    expected_quantiles = -np.log10(expected_pvals)
    observed_quantiles = -np.log10(observed_pvals)

    # Genomic inflation factor (lambda)
    chi2 = stats.chi2.isf(observed_pvals, df=1)
    lambda_gc = np.median(chi2) / stats.chi2.ppf(0.5, df=1)

    plt.figure(figsize=(6, 6))
    plt.scatter(expected_quantiles, observed_quantiles, s=8, alpha=0.6, color="#1f77b4", edgecolor="none")
    plt.plot([0, max(expected_quantiles.max(), observed_quantiles.max())],
            [0, max(expected_quantiles.max(), observed_quantiles.max())],
            linestyle="--", color="gray", linewidth=1)

    plt.title(f"QQ Plot of GWAS P-values\n(Expected vs. Observed Under Null, λ={lambda_gc:.2f})", fontsize=13)
    plt.xlabel("Expected -log₁₀(p)", fontsize=11)
    plt.ylabel("Observed -log₁₀(p)", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.qq_plot)
    plt.close()


    ##############################################
    # Save GWAS results
    ##############################################
    df = pd.DataFrame({
        "snp_index": np.arange(num_sites),
        "beta": slopes,
        "t_stat": t_stats,
        "pval": pvals,
        "is_causal": causal_mask,
        "is_significant": sig_mask,
        "AF_diff": AF_diff
    })
    df.to_csv(args.output_csv, index=False)

    ##############################################
    # Summary
    ##############################################
    print(f"Total SNPs: {num_sites}")
    print(f"Significant SNPs: {n_total_sig}")
    print(f"Causal among those: {n_sig_causal}")
    print(f"Spurious among those: {n_sig_spurious} ({spurious_pct:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run naive GWAS on genotype and phenotype data")
    parser.add_argument("--genotype_matrix", type=str, required=True)
    parser.add_argument("--sorted_phenotype", type=str, required=True)
    parser.add_argument("--trait_info", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--manhattan_plot", type=str, required=True)
    parser.add_argument("--af_diff_plot", type=str, required=True)
    parser.add_argument("--qq_plot", type=str, required=True, help="Path to save QQ plot image")
    args = parser.parse_args()
    main(args)
