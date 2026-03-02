from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.stats import t
from scipy import stats


def _load_dataframe(path: Union[str, Path]) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unknown dataframe format: {path}")


def _load_npy_1d(path: Union[str, Path], name: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D array, got shape {arr.shape} from {path}")
    return arr


def _load_npy_2d(path: Union[str, Path], name: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D array, got shape {arr.shape} from {path}")
    return arr


def run_gwas(
    genotype_path: Union[str, Path],
    phenotype_path: Union[str, Path],
    trait_path: Union[str, Path],
    variant_site_ids_path: Union[str, Path],
    output_prefix: str,
    discovery_pop: Optional[str] = None,
) -> None:
    """
    Naive per-SNP linear regression GWAS:
      phenotype ~ SNP

    Inputs:
      genotype_path: (N, L) float or int dosage in {0,1,2}
      phenotype_path: meta.pkl with columns at least: ["individual_id","population","phenotype"]
      trait_path: effect_sizes.pkl with column ["site_id"] (tskit site IDs of causal variants)
      variant_site_ids_path: (L,) mapping from SNP index -> tskit site ID
    """

    print(f"[gwas] Loading genotypes from {genotype_path} ...")
    G = _load_npy_2d(genotype_path, "genotype").astype(np.float32)
    N, L = G.shape

    print(f"[gwas] Loading meta/phenotype from {phenotype_path} ...")
    meta = _load_dataframe(phenotype_path).copy()

    required_cols = {"population", "phenotype"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(
            f"meta file {phenotype_path} must contain columns {sorted(required_cols)}; missing {sorted(missing)}. "
            "Fix build_inputs to include phenotype in meta.pkl."
        )

    # Align length: build_inputs already aligned genotype rows to meta rows; enforce here.
    if len(meta) != N:
        raise ValueError(f"[gwas] Row mismatch: G has N={N} rows but meta has {len(meta)} rows.")

    print(f"[gwas] Loading variant_site_ids from {variant_site_ids_path} ...")
    variant_site_ids = _load_npy_1d(variant_site_ids_path, "variant_site_ids").astype(np.int64)
    if variant_site_ids.shape[0] != L:
        raise ValueError(f"[gwas] variant_site_ids length {variant_site_ids.shape[0]} != num_snps {L}")

    print(f"[gwas] Loading trait info from {trait_path} ...")
    trait_df = _load_dataframe(trait_path)
    if "site_id" not in trait_df.columns:
        raise ValueError(f"[gwas] trait file {trait_path} must contain a 'site_id' column (tskit site ids).")

    # Build causal mask in SNP-index space by mapping tskit site IDs -> indices in our filtered/subsetted matrix
    causal_site_ids = set(trait_df["site_id"].astype(int).tolist())
    causal_mask = np.isin(variant_site_ids, np.fromiter(causal_site_ids, dtype=np.int64, count=len(causal_site_ids)))

    # Keep copy for AF-diff + PRS evaluation across all individuals
    full_G = G
    full_meta = meta

    # Discovery subset
    if discovery_pop is not None:
        keep = (meta["population"].astype(str) == str(discovery_pop)).to_numpy()
        if keep.sum() < 3:
            raise ValueError(f"[gwas] Too few individuals in discovery_pop={discovery_pop}: n={keep.sum()}")
        G = G[keep]
        meta = meta.loc[keep].reset_index(drop=True)
        print(f"[gwas] Discovery pop = {discovery_pop} | n={keep.sum()} individuals")
    else:
        print("[gwas] Discovery pop = ALL individuals")

    y = meta["phenotype"].to_numpy(dtype=np.float64)
    X = G.astype(np.float64)

    # Center
    y_centered = y - y.mean()
    X_centered = X - X.mean(axis=0)

    numerator = (X_centered * y_centered[:, None]).sum(axis=0)
    denominator = (X_centered**2).sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = numerator / denominator
        beta[~np.isfinite(beta)] = 0.0

    # R^2 for simple regression with intercept = (corr)^2; compute robustly
    y_ss = (y_centered**2).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = (numerator**2) / (denominator * y_ss)
        r2[~np.isfinite(r2)] = 0.0

    y_var = np.var(y, ddof=1)
    residual_var = y_var * (1.0 - r2)

    df = len(y) - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        stderr = np.sqrt(residual_var / denominator)
        t_stat = beta / stderr
        t_stat[~np.isfinite(t_stat)] = 0.0

    pvals = 2 * t.sf(np.abs(t_stat), df)
    pvals[~np.isfinite(pvals)] = 1.0

    # Bonferroni
    bonf = 0.05 / max(L, 1)
    sig_mask = pvals < bonf

    n_total_sig = int(sig_mask.sum())
    n_sig_causal = int((sig_mask & causal_mask).sum())
    n_sig_spur = int(n_total_sig - n_sig_causal)
    spur_pct = 100.0 * n_sig_spur / max(n_total_sig, 1)

    # Manhattan
    point_colors = np.full(L, "#333333", dtype=object)
    point_colors[sig_mask & ~causal_mask] = "#1f77b4"
    point_colors[sig_mask &  causal_mask] = "#d62728"
    point_sizes = np.full(L, 6, dtype=float)
    point_sizes[sig_mask & ~causal_mask] = 12
    point_sizes[sig_mask &  causal_mask] = 20

    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(L), -np.log10(pvals), c=point_colors, s=point_sizes, alpha=0.8, lw=0)
    plt.axhline(-np.log10(bonf), ls="--", c="gray", lw=1.2)
    plt.title(f"Spurious GWAS Hits\n({spur_pct:.1f}% Spurious of {n_total_sig} Significant SNPs)", fontsize=14)
    plt.xlabel("SNP index"); plt.ylabel("-log₁₀(p)")

    legend = []
    if (sig_mask & causal_mask).any():
        legend.append(Line2D([0], [0], marker="o", color="w", label="Causal SNP (Significant)",
                             markerfacecolor="#d62728", markeredgecolor="black", markersize=7))
    if (sig_mask & ~causal_mask).any():
        legend.append(Line2D([0], [0], marker="o", color="w", label="Spurious SNP (Significant)",
                             markerfacecolor="#1f77b4", markersize=7))
    legend.append(Line2D([0], [0], marker="o", color="w", label="Non-significant SNP",
                         markerfacecolor="#333333", markersize=5))
    plt.legend(handles=legend, frameon=False, fontsize=9)

    manhattan_path = f"{output_prefix}_manhattan.png"
    plt.tight_layout(); plt.savefig(manhattan_path); plt.close()
    print(f"[gwas] Wrote {manhattan_path}")

    # AF diff plot (computed over FULL data, p-values from discovery)
    pops = full_meta["population"].astype(str).unique()
    AF_diff = np.zeros(L, dtype=np.float64)
    if len(pops) >= 2:
        if "YRI" in pops and "CEU" in pops:
            pop1, pop2 = "YRI", "CEU"
        else:
            pop1, pop2 = pops[0], pops[1]

        pop_vals = full_meta["population"].astype(str).to_numpy()
        m1, m2 = (pop_vals == pop1), (pop_vals == pop2)
        af1 = full_G[m1].sum(0) / (2.0 * max(m1.sum(), 1))
        af2 = full_G[m2].sum(0) / (2.0 * max(m2.sum(), 1))
        AF_diff = np.abs(af1 - af2)

        norm = Normalize(vmin=float(AF_diff.min()), vmax=float(AF_diff.max()))
        spearman_r, _ = stats.spearmanr(AF_diff, -np.log10(pvals))

        plt.figure(figsize=(10, 5))
        plt.scatter(np.arange(L), -np.log10(pvals), c=AF_diff, cmap="coolwarm", norm=norm, s=6, lw=0, alpha=0.8)
        plt.scatter(np.where(sig_mask)[0], -np.log10(pvals[sig_mask]),
                    c=AF_diff[sig_mask], cmap="coolwarm", norm=norm, s=20, edgecolors="black", lw=0.3)
        plt.axhline(-np.log10(bonf), ls="--", c="gray")
        plt.colorbar(label=f"|AF diff ({pop1} – {pop2})|")
        plt.title(f"Standard GWAS: Allele Frequencies\nSpearman r = {spearman_r:.3f}")
        plt.xlabel("SNP index"); plt.ylabel("-log₁₀(p)")
        af_diff_path = f"{output_prefix}_af_diff.png"
        plt.tight_layout(); plt.savefig(af_diff_path); plt.close()
        print(f"[gwas] Wrote {af_diff_path}")
    else:
        print("[gwas] Not enough populations for AF difference plot.")
        af_diff_path = f"{output_prefix}_af_diff.png"
        plt.figure(figsize=(10, 5))
        plt.text(0.1, 0.5, "Not enough populations for AF diff plot.", fontsize=12)
        plt.axis("off")
        plt.tight_layout(); plt.savefig(af_diff_path); plt.close()

    # QQ
    exp = -np.log10(np.linspace(1 / (L + 1), 1, L))
    obs = -np.log10(np.sort(pvals))
    obs = np.nan_to_num(obs, posinf=300)

    if L > 1:
        lam = np.median(stats.chi2.isf(pvals, 1)) / stats.chi2.ppf(0.5, 1)
    else:
        lam = 0.0

    plt.figure(figsize=(6, 6))
    plt.scatter(exp, obs, s=8, alpha=0.6)
    mx = float(max(exp.max(), obs.max()))
    plt.plot([0, mx], [0, mx], ls="--", c="gray")
    plt.title(f"QQ Plot (λ={lam:.2f})")
    plt.xlabel("Expected -log₁₀(p)"); plt.ylabel("Observed -log₁₀(p)")
    qq_path = f"{output_prefix}_qq.png"
    plt.tight_layout(); plt.savefig(qq_path); plt.close()
    print(f"[gwas] Wrote {qq_path}")

    # Save table
    gwas_df = pd.DataFrame({
        "snp_index": np.arange(L, dtype=int),
        "ts_site_id": variant_site_ids.astype(int),
        "beta": beta.astype(float),
        "t_stat": t_stat.astype(float),
        "pval": pvals.astype(float),
        "is_causal": causal_mask.astype(bool),
        "is_significant": sig_mask.astype(bool),
        "AF_diff": AF_diff.astype(float),
    })
    csv_path = f"{output_prefix}_results.csv"
    gwas_df.to_csv(csv_path, index=False)
    print(f"[gwas] Wrote {csv_path}")

    # Also write the name Snakemake expects if you prefer gwas_results.csv
    # (optional; remove if you don't want two files)
    out_alias = Path(output_prefix).parent / "gwas_results.csv"
    gwas_df.to_csv(out_alias, index=False)

    print("[gwas] Done.")