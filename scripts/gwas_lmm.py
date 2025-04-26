import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy import stats
import statsmodels.api as sm


def main(args):
    G = np.load(args.genotype_matrix)  # (n_indivs, n_snps)
    meta = pd.read_csv(args.sorted_phenotype)
    trait_df = pd.read_csv(args.trait_info)
    y_all = meta["phenotype"].values.astype(float)
    n, m = G.shape

    if args.discovery_pop:
        keep = meta["population"] == args.discovery_pop
        G_used = G[keep]
        y = y_all[keep]
        print(f"Running LMM GWAS on {args.discovery_pop} population ({keep.sum()} individuals)")
    else:
        G_used = G
        y = y_all
        print("Running LMM GWAS across all individuals")

    Z = pd.get_dummies(meta["population"]).values if not args.discovery_pop else None

    pvals = np.zeros(m)
    betas = np.zeros(m)
    t_stats = np.zeros(m)

    for j in range(m):
        g = G_used[:, j].astype(float)
        if np.std(g) == 0:
            pvals[j] = 1.0
            betas[j] = np.nan
            t_stats[j] = np.nan
            continue
        X = sm.add_constant(g)
        try:
            if Z is not None:
                model = sm.MixedLM(y, X, groups=Z.argmax(1))
                result = model.fit()
            else:
                model = sm.OLS(y, X)
                result = model.fit()
            pvals[j] = result.pvalues[1]
            betas[j] = result.params[1]
            t_stats[j] = result.tvalues[1]
        except Exception:
            pvals[j] = 1.0
            betas[j] = np.nan
            t_stats[j] = np.nan

    bonf = 0.05 / m
    sig = pvals < bonf
    causal = np.zeros(m, bool)
    causal[trait_df["site_id"].values] = True
    n_total_sig = sig.sum()
    n_sig_causal = np.sum(sig & causal)
    n_sig_spur = n_total_sig - n_sig_causal
    spur_pct = 100 * n_sig_spur / max(n_total_sig, 1)

    # Manhattan plot with legend
    colours = np.where(sig & causal, "#d62728", np.where(sig, "#1f77b4", "#333333"))
    sizes = np.where(sig & causal, 20, np.where(sig, 12, 6))
    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(m), -np.log10(pvals), c=colours, s=sizes, lw=0, alpha=0.8)
    idx = np.where(sig & causal)[0]
    if idx.size:
        plt.scatter(idx, -np.log10(pvals[idx]), c="#d62728", s=sizes[idx], edgecolors="black", lw=0.5)
    plt.axhline(-np.log10(bonf), ls="--", c="gray")
    plt.xlabel("SNP index"); plt.ylabel("-log₁₀(p)")
    plt.title(f"Spurious GWAS Hits\n({spur_pct:.1f}% Spurious of {n_total_sig} Significant SNPs)")
    legend = []
    if (sig & causal).any():
        legend.append(Line2D([0],[0],marker='o',color='w',label='Causal SNP (Significant)',
                             markerfacecolor='#d62728',markeredgecolor='black',markersize=7))
    if (sig & ~causal).any():
        legend.append(Line2D([0],[0],marker='o',color='w',label='Spurious SNP (Significant)',
                             markerfacecolor='#1f77b4',markersize=7))
    legend.append(Line2D([0],[0],marker='o',color='w',label='Non‑significant SNP',
                         markerfacecolor='#333333',markersize=5))
    plt.legend(handles=legend, frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(args.manhattan_plot); plt.close()

    # AF diff
    pop = meta["population"].values
    afr = pop == "AFR"
    eur = pop == "EUR"
    af_diff = np.abs(G[afr].sum(0)/(2*afr.sum()) - G[eur].sum(0)/(2*eur.sum()))
    norm = Normalize(vmin=af_diff.min(), vmax=af_diff.max())
    spearman_r, spearman_p = stats.spearmanr(af_diff, -np.log10(pvals))
    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(m), -np.log10(pvals), c=af_diff, cmap="coolwarm", norm=norm, s=6, lw=0, alpha=0.8)
    plt.scatter(np.where(sig)[0], -np.log10(pvals[sig]), c=af_diff[sig], cmap="coolwarm",
                norm=norm, s=20, edgecolors="black", lw=0.3)
    plt.axhline(-np.log10(bonf), ls="--", c="gray")
    plt.colorbar(label="|AF diff| AFR - EUR")
    plt.xlabel("SNP index"); plt.ylabel("-log₁₀(p)")
    plt.title(f"Linear Mixed Model GWAS: Allele Frequencies \nSpearman r = {spearman_r:.3f}")
    plt.tight_layout(); plt.savefig(args.af_diff_plot); plt.close()

    # QQ plot
    expected = -np.log10(np.linspace(1/(m+1), 1, m))
    observed = -np.log10(np.sort(pvals))
    chisq = stats.chi2.isf(pvals, 1)
    lam = np.median(chisq) / stats.chi2.ppf(0.5, 1)
    plt.figure(figsize=(6, 6))
    plt.scatter(expected, observed, s=8, alpha=0.6, color="#1f77b4")
    mx = max(expected.max(), observed.max())
    plt.plot([0, mx], [0, mx], ls="--", c="gray")
    plt.xlabel("Expected -log₁₀(p)")
    plt.ylabel("Observed -log₁₀(p)")
    plt.title(f"LMM GWAS QQ Plot (λ={lam:.2f})")
    plt.tight_layout(); plt.savefig(args.qq_plot); plt.close()

    # Save results
    df = pd.DataFrame({
        "snp_index": np.arange(m),
        "beta": betas,
        "t_stat": t_stats,
        "pval": pvals,
        "is_causal": causal,
        "is_significant": sig,
        "AF_diff": af_diff
    })
    df.to_csv(args.output_csv, index=False)

    # Compute PRS for all individuals using GWAS beta from discovery pop
    meta["PRS"] = G @ np.nan_to_num(betas)
    for group in ["AFR", "EUR"]:
        sub = meta[meta.population == group]
        r = np.corrcoef(sub["PRS"], sub["phenotype"])[0, 1]
        plt.figure()
        plt.scatter(sub["PRS"], sub["phenotype"], alpha=0.5)
        plt.xlabel("PRS"); plt.ylabel("Phenotype")
        plt.title(f"PRS vs. Phenotype in {group} (r = {r:.3f})")
        plt.tight_layout()
        plt.savefig(f"results/gwas_lmm/prs_scatter_{group}.png")
        sub[["individual_id", "PRS", "phenotype"]].to_csv(f"results/gwas_lmm/prs_values_{group}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genotype_matrix", type=str, required=True)
    parser.add_argument("--sorted_phenotype", type=str, required=True)
    parser.add_argument("--trait_info", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--manhattan_plot", type=str, required=True)
    parser.add_argument("--af_diff_plot", type=str, required=True)
    parser.add_argument("--qq_plot", type=str, required=True)
    parser.add_argument("--discovery_pop", default=None, help="Restrict GWAS to a population")
    main(parser.parse_args())