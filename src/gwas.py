
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.stats import t
from scipy import stats
import tskit
import pickle
from pathlib import Path

def load_genotype_matrix(path):
    """
    Load genotype matrix from .npy or .trees file.
    Returns (num_inds, num_sites) matrix.
    If .trees, aggregates haplotypes to individual genotypes (sum).
    """
    path = str(path)
    if path.endswith(".trees") or path.endswith(".ts"):
        ts = tskit.load(path)
        G_hap = ts.genotype_matrix() # (sites, samples)
        
        if ts.num_individuals > 0:
            # Aggregate haplotypes to individuals
            num_inds = ts.num_individuals
            num_sites = ts.num_sites
            # Use float to avoid overflow if ploidy is high, though usually int8/16 is fine
            # But we cast to float later anyway
            G_ind = np.zeros((num_inds, num_sites), dtype=np.float32)
            
            for i, ind in enumerate(ts.individuals()):
                nodes = ind.nodes
                if len(nodes) > 0:
                    # Sum across the samples for this individual
                    G_ind[i] = G_hap[:, nodes].sum(axis=1)
            return G_ind
        else:
            # No individuals defined, return haplotypes as "individuals"
            return G_hap.T
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError(f"Unknown genotype file format: {path}")

def load_dataframe(path):
    """Load DataFrame from .csv or .pkl."""
    path = str(path)
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unknown dataframe file format: {path}")

def run_gwas(
    genotype_path,
    phenotype_path,
    trait_path,
    output_prefix,
    discovery_pop=None
):
    # Load data
    print(f"Loading genotypes from {genotype_path}...")
    G = load_genotype_matrix(genotype_path)
    
    print(f"Loading phenotypes from {phenotype_path}...")
    merged_sorted = load_dataframe(phenotype_path)
    
    # Ensure phenotypes are sorted by individual_id to match genotype matrix order
    if "individual_id" in merged_sorted.columns:
        merged_sorted = merged_sorted.sort_values("individual_id").reset_index(drop=True)
    
    # Verify shapes
    if G.shape[0] != len(merged_sorted):
        print(f"Warning: Genotype matrix has {G.shape[0]} individuals but phenotype file has {len(merged_sorted)}.")
        # If mismatch, try to intersect based on individual_id if possible?
        # But G doesn't have IDs attached easily unless we return them.
        # For now, assume 1-to-1 mapping if counts match, or error if not.
        if G.shape[0] == 2 * len(merged_sorted):
             raise ValueError(f"Genotype matrix has {G.shape[0]} rows (likely haplotypes) while phenotypes have {len(merged_sorted)} rows. Genotype aggregation failed or ploidy mismatch.")
        elif G.shape[0] != len(merged_sorted):
             raise ValueError(f"Shape mismatch: G={G.shape}, phenotypes={len(merged_sorted)}")
    
    print(f"Loading trait info from {trait_path}...")
    trait_df = load_dataframe(trait_path)
    
    full_G = G.copy()
    full_meta = merged_sorted.copy()
    num_inds, num_sites = G.shape

    if discovery_pop:
        keep = merged_sorted["population"] == discovery_pop
        if keep.sum() == 0:
            print(f"Warning: No individuals found for population {discovery_pop}. Available: {merged_sorted['population'].unique()}")
        G = G[keep]
        merged_sorted = merged_sorted[keep]
        print(f"Running GWAS on discovery population: {discovery_pop} ({keep.sum()} individuals)")
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
    # Avoid division by zero if denominator is 0
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = (numerator**2) / (denominator * (y_centered**2).sum())
        r2[np.isnan(r2)] = 0.0
        
    residual_var = y_var * (1 - r2)
    df = len(y) - 2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        stderr = np.sqrt(residual_var / denominator)
        t_stats = slopes / stderr
        t_stats[np.isnan(t_stats)] = 0.0
        
    pvals = 2 * t.sf(np.abs(t_stats), df)
    pvals[np.isnan(pvals)] = 1.0

    ##############################################
    # Determine causal SNPs
    ##############################################
    causal_site_ids = set(trait_df["site_id"])
    causal_mask = np.zeros(num_sites, dtype=bool)
    # Ensure site_ids are within range
    valid_site_ids = [sid for sid in causal_site_ids if sid < num_sites]
    causal_mask[valid_site_ids] = True

    ##############################################
    # Bonferroni threshold
    ##############################################
    bonf = 0.05 / max(num_sites, 1)
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

    manhattan_path = f"{output_prefix}_manhattan.png"
    plt.tight_layout(); plt.savefig(manhattan_path); plt.close()
    print(f"Saved Manhattan plot to {manhattan_path}")

    # AF-difference plot
    # Determine populations for AF diff
    pops = full_meta["population"].unique()
    if len(pops) >= 2:
        # Try to find AFR/EUR or YRI/CEU, otherwise take first two
        if "AFR" in pops and "EUR" in pops:
            pop1, pop2 = "AFR", "EUR"
        elif "YRI" in pops and "CEU" in pops:
            pop1, pop2 = "YRI", "CEU"
        else:
            pop1, pop2 = pops[0], pops[1]
            
        pop_vals = full_meta["population"].values
        mask1, mask2 = (pop_vals == pop1), (pop_vals == pop2)
        
        # Calculate AF
        # full_G is (num_inds, num_sites)
        # sum over inds (axis 0)
        af1 = full_G[mask1].sum(0) / (2 * mask1.sum())
        af2 = full_G[mask2].sum(0) / (2 * mask2.sum())
        
        AF_diff = np.abs(af1 - af2)
        norm = Normalize(vmin=AF_diff.min(), vmax=AF_diff.max())
        spearman_r, spearman_p = stats.spearmanr(AF_diff, -np.log10(pvals))

        plt.figure(figsize=(10, 5))
        plt.scatter(np.arange(num_sites), -np.log10(pvals),
                    c=AF_diff, cmap="coolwarm", norm=norm, s=6, lw=0, alpha=0.8)
        plt.scatter(np.where(sig_mask)[0], -np.log10(pvals[sig_mask]),
                    c=AF_diff[sig_mask], cmap="coolwarm", norm=norm,
                    s=20, edgecolors="black", lw=0.3)
        plt.axhline(-np.log10(bonf), ls="--", c="gray")
        plt.colorbar(label=f"|AF diff ({pop1} – {pop2})|")
        plt.title(f"Standard GWAS: Allele Frequencies \nSpearman r = {spearman_r:.3f}")
        plt.xlabel("SNP index"); plt.ylabel("-log₁₀(p)")
        af_diff_path = f"{output_prefix}_af_diff.png"
        plt.tight_layout(); plt.savefig(af_diff_path); plt.close()
        print(f"Saved AF diff plot to {af_diff_path}")
    else:
        print("Not enough populations for AF difference plot.")
        AF_diff = np.zeros(num_sites)

    # QQ plot
    exp = -np.log10(np.linspace(1 / (num_sites + 1), 1, num_sites))
    obs = -np.log10(np.sort(pvals))
    # Avoid infs
    obs = np.nan_to_num(obs, posinf=300) 
    
    if len(pvals) > 1:
        lam = np.median(stats.chi2.isf(pvals, 1)) / stats.chi2.ppf(0.5, 1)
    else:
        lam = 0

    plt.figure(figsize=(6, 6))
    plt.scatter(exp, obs, s=8, alpha=0.6, color="#1f77b4")
    mx = max(exp.max(), obs.max())
    plt.plot([0, mx], [0, mx], ls="--", c="gray")
    plt.title(f"QQ Plot (λ={lam:.2f})")
    plt.xlabel("Expected -log₁₀(p)"); plt.ylabel("Observed -log₁₀(p)")
    qq_path = f"{output_prefix}_qq.png"
    plt.tight_layout(); plt.savefig(qq_path); plt.close()
    print(f"Saved QQ plot to {qq_path}")

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
    csv_path = f"{output_prefix}_results.csv"
    gwas_df.to_csv(csv_path, index=False)
    print(f"Saved GWAS results to {csv_path}")

    # Compute PRS for all individuals using GWAS beta from discovery pop
    # Slopes are from the discovery pop (or all if not specified)
    # We apply these betas to ALL individuals in full_G
    full_meta["PRS"] = full_G @ np.nan_to_num(slopes)
    
    # Plot PRS vs Phenotype for each population
    for group in full_meta["population"].unique():
        sub = full_meta[full_meta.population == group]
        if len(sub) < 2:
            continue
        r = np.corrcoef(sub["PRS"], sub["phenotype"])[0, 1]
        plt.figure()
        plt.scatter(sub["PRS"], sub["phenotype"], alpha=0.5)
        plt.xlabel("PRS"); plt.ylabel("Phenotype")
        plt.title(f"PRS vs. Phenotype in {group} (r = {r:.3f})")
        plt.tight_layout()
        
        prs_plot_path = f"{output_prefix}_prs_scatter_{group}.png"
        plt.savefig(prs_plot_path)
        plt.close()
        
        prs_csv_path = f"{output_prefix}_prs_values_{group}.csv"
        sub[["individual_id", "PRS", "phenotype"]].to_csv(prs_csv_path, index=False)
        print(f"Saved PRS results for {group}")

