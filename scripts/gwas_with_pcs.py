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
• Outputs: CSV, Manhattan, AF-diff, QQ, Scree — identical styling to the naïve GWAS, **including the “Spurious Hits” percentage in the title**.
"""

# ------------------------------------------------------------------
# helper ------------------------------------------------------------
# ------------------------------------------------------------------

def safe_ols(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6):
    """Return (beta, t‑stat, p‑value) for the first column of X using a tiny ridge."""
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
    y = meta["phenotype"].to_numpy()
    n, m = G.shape

    keep = G.std(0) > 0
    Xpca = (G[:, keep] - G[:, keep].mean(0)) / G[:, keep].std(0)
    PCs = PCA(a.num_pcs).fit_transform(Xpca)

    # scree plot
    plt.figure()
    cum = np.cumsum(PCA().fit(Xpca).explained_variance_ratio_) * 100
    plt.plot(range(1, len(cum)+1), cum, marker="o")
    plt.xlabel("# PCs"); plt.ylabel("Cumulative variance (%)")
    plt.title("Scree plot")
    plt.tight_layout(); plt.savefig(a.scree_plot); plt.close()

    Z = (PCs - PCs.mean(0)) / PCs.std(0)
    design_const = np.hstack([Z, np.ones((n,1))])

    betas = np.empty(m); tvals = np.empty(m); pvals = np.empty(m)
    for j in range(m):
        g = G[:, j].astype(float)
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
    plt.figure(figsize=(10,5))
    plt.scatter(range(m), -np.log10(pvals), c=AF, cmap="coolwarm", norm=norm,
                s=4, lw=0, alpha=0.75)
    plt.scatter(np.where(sig)[0], -np.log10(pvals[sig]), c=AF[sig], cmap="coolwarm",
                norm=norm, s=18, edgecolors="black", lw=0.3)
    plt.axhline(-np.log10(bonf), ls="--", c="gray")
    plt.colorbar(label="|AF diff| AFR‑EUR")
    plt.title("GWAS with PCs: AF‑diff colouring")
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

    # Save results
    pd.DataFrame({
        "snp_index": np.arange(m),
        "beta": betas,
        "t_stat": tvals,
        "pval": pvals,
        "is_causal": causal,
        "is_significant": sig,
        "AF_diff": AF
    }).to_csv(a.output_csv, index=False)

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
    main(p.parse_args())