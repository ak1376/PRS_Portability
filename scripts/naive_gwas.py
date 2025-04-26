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
    full_G = G.copy()
    full_meta = merged_sorted.copy()
    num_inds, num_sites = G.shape

    if args.discovery_pop:
        keep = merged_sorted["population"] == args.discovery_pop
        G = G[keep]
        merged_sorted = merged_sorted[keep]
        print(f"Running GWAS on discovery population: {args.discovery_pop} ({keep.sum()} individuals)")
    else:
        print("Running GWAS across all individuals")

    y = merged_sorted["phenotype"].values
    X = G.astype(float)

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
    df = len(y) - 2
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
    bonf = 0.05 / num_sites
    sig_mask = pvals < bonf

    n_total_sig = sig_mask.sum()
    n_sig_causal = np.sum(sig_mask & causal_mask)
    n_sig_spur = n_total_sig - n_sig_causal
    spur_pct = 100 * n_sig_spur / max(n_total_sig, 1)

    # Colours & sizes
    point_colors = np.full(num_sites, "#333333", dtype=object)
    point_colors[sig_mask & ~causal_mask] = "#1f77b4"
    point_colors[sig_mask &  causal_mask] = "#d62728"
    point_sizes = np.full(num_sites, 6)
    point_sizes[sig_mask & ~causal_mask] = 12
    point_sizes[sig_mask &  causal_mask] = 20

    # Manhattan plot
    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(num_sites), -np.log10(pvals),
                c=point_colors, s=point_sizes, alpha=0.8, lw=0)

    causal_sig = np.where(sig_mask & causal_mask)[0]
    if causal_sig.size:
        plt.scatter(causal_sig, -np.log10(pvals[causal_sig]),
                    c="#d62728", s=point_sizes[causal_sig],
                    edgecolors="black", lw=0.5)

    plt.axhline(-np.log10(bonf), ls="--", c="gray", lw=1.2)
    plt.text(num_sites * 0.95, -np.log10(bonf) + 0.2, "Bonferroni",
             ha="right", va="bottom", color="gray")
    plt.title(f"Spurious GWAS Hits\n({spur_pct:.1f}% Spurious of {n_total_sig} Significant SNPs)", fontsize=14)
    plt.xlabel("SNP index"); plt.ylabel("-log₁₀(p)")

    legend = []
    if (sig_mask & causal_mask).any():
        legend.append(Line2D([0], [0], marker='o', color='w', label='Causal SNP (Significant)',
                             markerfacecolor='#d62728', markeredgecolor='black', markersize=7))
    if (sig_mask & ~causal_mask).any():
        legend.append(Line2D([0], [0], marker='o', color='w', label='Spurious SNP (Significant)',
                             markerfacecolor='#1f77b4', markersize=7))
    legend.append(Line2D([0], [0], marker='o', color='w', label='Non‑significant SNP',
                         markerfacecolor='#333333', markersize=5))
    plt.legend(handles=legend, frameon=False, fontsize=9)

    plt.tight_layout(); plt.savefig(args.manhattan_plot); plt.close()

    # AF-difference plot
    pop = full_meta["population"].values
    afr, eur = (pop == "AFR"), (pop == "EUR")
    AF_diff = np.abs(full_G[afr].sum(0)/(2*afr.sum()) - full_G[eur].sum(0)/(2*eur.sum()))
    norm = Normalize(vmin=AF_diff.min(), vmax=AF_diff.max())
    spearman_r, spearman_p = stats.spearmanr(AF_diff, -np.log10(pvals))

    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(num_sites), -np.log10(pvals),
                c=AF_diff, cmap="coolwarm", norm=norm, s=6, lw=0, alpha=0.8)
    plt.scatter(np.where(sig_mask)[0], -np.log10(pvals[sig_mask]),
                c=AF_diff[sig_mask], cmap="coolwarm", norm=norm,
                s=20, edgecolors="black", lw=0.3)
    plt.axhline(-np.log10(bonf), ls="--", c="gray")
    plt.colorbar(label="|AF diff (AFR – EUR)|")
    plt.title(f"Standard GWAS: Allele Frequencies \nSpearman r = {spearman_r:.3f}")
    plt.xlabel("SNP index"); plt.ylabel("-log₁₀(p)")
    plt.tight_layout(); plt.savefig(args.af_diff_plot); plt.close()

    # QQ plot
    exp = -np.log10(np.linspace(1 / (num_sites + 1), 1, num_sites))
    obs = -np.log10(np.sort(pvals))
    lam = np.median(stats.chi2.isf(pvals, 1)) / stats.chi2.ppf(0.5, 1)

    plt.figure(figsize=(6, 6))
    plt.scatter(exp, obs, s=8, alpha=0.6, color="#1f77b4")
    mx = max(exp.max(), obs.max()); plt.plot([0, mx], [0, mx], ls="--", c="gray")
    plt.title(f"QQ Plot (λ={lam:.2f})")
    plt.xlabel("Expected -log₁₀(p)"); plt.ylabel("Observed -log₁₀(p)")
    plt.tight_layout(); plt.savefig(args.qq_plot); plt.close()

    # Save GWAS results
    gwas_df = pd.DataFrame({
        "snp_index": np.arange(num_sites),
        "beta": slopes,
        "t_stat": t_stats,
        "pval": pvals,
        "is_causal": causal_mask,
        "is_significant": sig_mask,
        "AF_diff": AF_diff
    })
    gwas_df.to_csv(args.output_csv, index=False)

    # Compute PRS for all individuals using GWAS beta from discovery pop
    full_meta["PRS"] = full_G @ np.nan_to_num(slopes)
    for group in ["AFR", "EUR"]:
        sub = full_meta[full_meta.population == group]
        r = np.corrcoef(sub["PRS"], sub["phenotype"])[0, 1]
        plt.figure()
        plt.scatter(sub["PRS"], sub["phenotype"], alpha=0.5)
        plt.xlabel("PRS"); plt.ylabel("Phenotype")
        plt.title(f"PRS vs. Phenotype in {group} (r = {r:.3f})")
        plt.tight_layout()
        plt.savefig(f"results/gwas/prs_scatter_{group}.png")
        sub[["individual_id", "PRS", "phenotype"]].to_csv(f"results/gwas/prs_values_{group}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run naive GWAS on genotype and phenotype data")
    parser.add_argument("--genotype_matrix", type=str, required=True)
    parser.add_argument("--sorted_phenotype", type=str, required=True)
    parser.add_argument("--trait_info", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--manhattan_plot", type=str, required=True)
    parser.add_argument("--af_diff_plot", type=str, required=True)
    parser.add_argument("--qq_plot", type=str, required=True)
    parser.add_argument("--discovery_pop", type=str, default=None, help="Run GWAS only on a specified population (e.g. 'EUR')")
    args = parser.parse_args()
    main(args)
