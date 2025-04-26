import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import t
from sklearn.decomposition import PCA

"""
GWAS + top-k PCs (clean, minimal).
• PCA is run on a *copy* of the genotype matrix, dropping monomorphic columns **only for the PCA step**.
• Ridge-stabilised OLS per SNP.
• Optionally restricts GWAS to a specific population (e.g., EUR).
• Outputs: CSV, Manhattan, AF-diff, QQ, Scree.
• Also computes PRS scores for EUR and AFR and saves scatterplots.
"""

# ------------------------------------------------------------------
# helper ------------------------------------------------------------
# ------------------------------------------------------------------

def safe_ols(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6):
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    beta = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta
    df = X.shape[0] - X.shape[1]
    s2 = resid.dot(resid) / df
    se2 = s2 * np.linalg.inv(XtX).diagonal()
    t_val = beta[0] / np.sqrt(se2[0])
    pval = 2 * t.sf(abs(t_val), df)
    return beta[0], t_val, pval

# ------------------------------------------------------------------
# main --------------------------------------------------------------
# ------------------------------------------------------------------

def main(a):
    G = np.load(a.genotype_matrix)
    meta = pd.read_csv(a.sorted_phenotype)
    trait = pd.read_csv(a.trait_info)
    y_all = meta["phenotype"].to_numpy()
    n, m = G.shape

    if a.discovery_pop:
        keep = meta["population"] == a.discovery_pop
        G_discovery = G[keep]
        y = y_all[keep]
        print(f"Running GWAS on discovery population: {a.discovery_pop} ({keep.sum()} individuals)")
    else:
        G_discovery = G
        y = y_all
        print("Running GWAS across all individuals")

    # PCA on COPY
    keep_cols = G_discovery.std(0) > 0
    Xpca = (G_discovery[:, keep_cols] - G_discovery[:, keep_cols].mean(0)) / G_discovery[:, keep_cols].std(0)
    PCs = PCA(a.num_pcs).fit_transform(Xpca)

    # scree plot
    plt.figure()
    cum = np.cumsum(PCA().fit(Xpca).explained_variance_ratio_) * 100
    plt.plot(range(1, len(cum)+1), cum, marker="o")
    plt.xlabel("# PCs"); plt.ylabel("Cumulative variance (%)")
    plt.title("Scree plot"); plt.tight_layout()
    plt.savefig(a.scree_plot); plt.close()

    Z = (PCs - PCs.mean(0)) / PCs.std(0)
    design_const = np.hstack([Z, np.ones((len(y),1))])

    betas = np.empty(m); tvals = np.empty(m); pvals = np.empty(m)
    for j in range(m):
        g = G_discovery[:, j].astype(float)
        sd = g.std()
        if sd == 0:
            betas[j] = np.nan; tvals[j] = np.nan; pvals[j] = 1.0; continue
        g = (g - g.mean()) / sd
        X = np.column_stack([g, design_const])
        betas[j], tvals[j], pvals[j] = safe_ols(X, y)

    bonf = 0.05 / m
    sig = pvals < bonf
    causal = np.zeros(m, bool); causal[trait["site_id"].values] = True

    n_total_sig = sig.sum()
    n_sig_causal = np.sum(sig & causal)
    n_sig_spur = n_total_sig - n_sig_causal
    spur_pct = 100 * n_sig_spur / max(n_total_sig, 1)

    # Manhattan
    colours = np.where(sig & causal, "#d62728", np.where(sig, "#1f77b4", "#333333"))
    sizes = np.where(sig & causal, 20, np.where(sig, 12, 4))
    plt.figure(figsize=(10,5))
    plt.scatter(range(m), -np.log10(pvals), c=colours, s=sizes, lw=0, alpha=0.8)
    if (sig & causal).any():
        idx = np.where(sig & causal)[0]
        plt.scatter(idx, -np.log10(pvals[idx]), c="#d62728", s=sizes[idx],
                    edgecolors="black", lw=0.5)
    plt.axhline(-np.log10(bonf), ls="--", c="gray")
    plt.title(f"Spurious GWAS Hits\n({spur_pct:.1f}% Spurious of {n_total_sig} Significant SNPs)")
    plt.xlabel("SNP index"); plt.ylabel("-log₁₀(p)")
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
    plt.tight_layout(); plt.savefig(a.manhattan_plot); plt.close()

    # AF-diff
    pop = meta["population"].to_numpy(); afr = pop=="AFR"; eur = pop=="EUR"
    AF = np.abs(G[afr].sum(0)/(2*afr.sum()) - G[eur].sum(0)/(2*eur.sum()))
    norm = Normalize(vmin=0, vmax=0.5)
    spearman_r, spearman_p = stats.spearmanr(AF, -np.log10(pvals))
    plt.figure(figsize=(10,5))
    plt.scatter(range(m), -np.log10(pvals), c=AF, cmap="coolwarm", norm=norm,
                s=4, lw=0, alpha=0.75)
    plt.scatter(np.where(sig)[0], -np.log10(pvals[sig]), c=AF[sig], cmap="coolwarm",
                norm=norm, s=18, edgecolors="black", lw=0.3)
    plt.axhline(-np.log10(bonf), ls="--", c="gray")
    plt.colorbar(label="|AF diff| AFR‑EUR")
    plt.title(f"GWAS with PCs: Allele Frequencies \nSpearman r = {spearman_r:.3f}")
    plt.xlabel("SNP index"); plt.ylabel("-log₁₀(p)")
    plt.tight_layout(); plt.savefig(a.af_diff_plot); plt.close()

    # QQ plot
    exp = -np.log10(np.linspace(1/(m+1), 1, m))
    obs = -np.log10(np.sort(pvals))
    lam = np.median(stats.chi2.isf(np.sort(pvals),1)) / stats.chi2.ppf(0.5,1)
    plt.figure(figsize=(6,6))
    plt.scatter(exp, obs, s=8, alpha=0.6)
    mx = max(exp.max(), obs.max()); plt.plot([0,mx],[0,mx], ls="--", c="gray")
    plt.title(f"QQ plot (λ={lam:.2f})"); plt.xlabel("Expected -log₁₀(p)");
    plt.ylabel("Observed -log₁₀(p)")
    plt.tight_layout(); plt.savefig(a.qq_plot); plt.close()

    # Save GWAS results
    gwas_df = pd.DataFrame({
        "snp_index": np.arange(m),
        "beta": betas,
        "t_stat": tvals,
        "pval": pvals,
        "is_causal": causal,
        "is_significant": sig,
        "AF_diff": AF
    })
    gwas_df.to_csv(a.output_csv, index=False)

    # Compute PRS using valid SNPs only
    valid = np.isfinite(betas)
    betas_clean = betas.copy()
    betas_clean[~valid] = 0

    prs = G @ betas_clean
    meta["PRS"] = prs

    for group in ["AFR", "EUR"]:
        sub = meta[meta.population == group]
        r = np.corrcoef(sub["PRS"], sub["phenotype"])[0, 1]
        plt.figure()
        plt.scatter(sub["PRS"], sub["phenotype"], alpha=0.5)
        plt.xlabel("PRS"); plt.ylabel("Phenotype")
        plt.title(f"PRS vs. Phenotype in {group} (r = {r:.3f})")
        plt.tight_layout()
        plt.savefig(f"results/gwas_pcs/prs_scatter_{group}.png")
        sub[["individual_id", "PRS", "phenotype"]].to_csv(f"results/gwas_pcs/prs_values_{group}.csv", index=False)

# CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--genotype_matrix", required=True)
    p.add_argument("--sorted_phenotype", required=True)
    p.add_argument("--trait_info", required=True)
    p.add_argument("--num_pcs", type=int, default=5)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--manhattan_plot", required=True)
    p.add_argument("--af_diff_plot", required=True)
    p.add_argument("--qq_plot", required=True)
    p.add_argument("--scree_plot", required=True)
    p.add_argument("--discovery_pop", default=None, help="Restrict GWAS to one population (e.g. EUR)")
    main(p.parse_args())
