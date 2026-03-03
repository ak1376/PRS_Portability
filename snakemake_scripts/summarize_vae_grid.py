#!/usr/bin/env python3
"""
summarize_vae_grid.py

Summarize VAE grid experiments for easy comparison across model configurations.

Expected directory layout under --vae-basedir:

  <vae-basedir>/<exp>/<sid>/rep<rep>/
      hparams.resolved.yaml
      train_summary.json            (optional)
      logs/metrics.csv              (from Lightning CSVLogger)
      diagnostics/recon_summary.txt (from plot_vae_diagnostics.py)

Outputs to --outdir:
  - vae_grid_long.csv          (per-run recon metrics, long format)
  - vae_grid_summary.csv       (aggregated over runs)
  - vae_grid_diagnostics.csv   (diagnostic summary from metrics.csv)
  - vae_grid_joined.csv        (long merged with diag + hparams)

Notes:
- Robust to missing files.
- Parses recon_summary even if header line is commented with '#'.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# -------------------------
# utils
# -------------------------

def _safe_read_text(p: Path) -> str:
    return p.read_text() if p.exists() else ""

def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        return float(s)
    except Exception:
        return None

def _clean_header_line(ln: str) -> str:
    ln = ln.strip()
    if ln.startswith("#"):
        ln = ln[1:].strip()
    return ln

def _is_delimited_header(ln: str) -> bool:
    # Accept both:
    #   split metric model maf_baseline where
    #   split\tmetric\tmodel\tmaf_baseline\twhere
    ln = _clean_header_line(ln)
    toks = re.split(r"[,\t ]+", ln.strip())
    toks = [t for t in toks if t]
    need = {"split", "metric", "model", "maf_baseline", "where"}
    return need.issubset(set(toks))

def _split_tokens(ln: str) -> List[str]:
    # robust split for comma OR tab OR whitespace
    ln = ln.strip()
    if "," in ln and ("\t" not in ln):
        toks = [t.strip() for t in ln.split(",")]
        return [t for t in toks if t != ""]
    toks = re.split(r"[\t ]+", ln)
    return [t for t in toks if t != ""]

def _tagify(v: Any) -> str:
    if isinstance(v, float):
        s = f"{v:g}"
        return s.replace(".", "p")
    return str(v)


# -------------------------
# YAML flattening
# -------------------------

def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=key))
        elif isinstance(v, (list, tuple)):
            # store list as comma-separated for easy viewing
            out[key] = ",".join(map(str, v))
        else:
            out[key] = v
    return out


# -------------------------
# recon_summary parsing
# -------------------------

RECON_LONG_COLS = [
    "exp", "sid", "rep",
    "split", "where", "metric",
    "model", "maf_baseline",
    "delta_vs_maf", "ratio_vs_maf",
    "recon_path",
]

def parse_recon_summary(path: Path, *, exp: str, sid: int, rep: int) -> pd.DataFrame:
    """
    recon_summary.txt expected example:

      # split metric model maf_baseline where
      CEU_train mse 0.146926 0.14381 all
      CEU_train mse 0.143026 0.142823 masked_only
      ...

    We accept:
      - header commented with '#'
      - delimiter: whitespace, tab, or comma
    """
    txt = _safe_read_text(path)
    if not txt.strip():
        return pd.DataFrame(columns=RECON_LONG_COLS)

    lines0 = [ln for ln in txt.splitlines() if ln.strip()]
    # Find header line
    header_idx = None
    for i, ln in enumerate(lines0):
        if _is_delimited_header(ln):
            header_idx = i
            break

    if header_idx is None:
        # Try: if first non-empty line already looks like data (5 tokens), treat it as data without header
        # But require at least 5 columns.
        data_start = 0
        # Synthetic header
        header = ["split", "metric", "model", "maf_baseline", "where"]
    else:
        header = _split_tokens(_clean_header_line(lines0[header_idx]))
        data_start = header_idx + 1

    # Normalize header to required columns (keep order if present)
    # Some files may have extra columns; we only use required five.
    required = ["split", "metric", "model", "maf_baseline", "where"]
    # Build index map
    hmap = {h: j for j, h in enumerate(header)}
    if not all(k in hmap for k in required):
        # maybe header had weird spacing; fall back to required positions
        hmap = {k: i for i, k in enumerate(required)}

    rows: List[Dict[str, Any]] = []
    for ln in lines0[data_start:]:
        # skip pure comments
        if ln.strip().startswith("#"):
            continue
        toks = _split_tokens(_clean_header_line(ln))
        if len(toks) < 5:
            continue

        split = toks[hmap["split"]] if hmap["split"] < len(toks) else None
        metric = toks[hmap["metric"]] if hmap["metric"] < len(toks) else None
        model = _as_float(toks[hmap["model"]]) if hmap["model"] < len(toks) else None
        maf = _as_float(toks[hmap["maf_baseline"]]) if hmap["maf_baseline"] < len(toks) else None
        where = toks[hmap["where"]] if hmap["where"] < len(toks) else None

        if split is None or metric is None or where is None:
            continue
        if model is None or maf is None:
            # allow ratio-only rows etc, but keep numeric fields if present
            pass

        delta = (model - maf) if (model is not None and maf is not None) else None
        ratio = (model / (maf + 1e-12)) if (model is not None and maf is not None) else None

        rows.append({
            "exp": exp,
            "sid": sid,
            "rep": rep,
            "split": split,
            "where": where,
            "metric": metric,
            "model": model,
            "maf_baseline": maf,
            "delta_vs_maf": delta,
            "ratio_vs_maf": ratio,
            "recon_path": str(path),
        })

    if not rows:
        return pd.DataFrame(columns=RECON_LONG_COLS)
    return pd.DataFrame(rows, columns=RECON_LONG_COLS)


# -------------------------
# Lightning metrics parsing
# -------------------------

def load_metrics_csv(logdir: Path) -> Optional[pd.DataFrame]:
    # Lightning CSVLogger writes logs/version_*/metrics.csv sometimes.
    # You used logs/metrics.csv directly in your wrapper, but we handle both.
    candidates = []
    direct = logdir / "metrics.csv"
    if direct.exists():
        candidates.append(direct)
    for p in logdir.glob("**/metrics.csv"):
        candidates.append(p)

    for p in candidates:
        try:
            df = pd.read_csv(p)
            if "step" in df.columns:
                return df
        except Exception:
            continue
    return None

def summarize_diagnostics_from_metrics(df: pd.DataFrame, exp: str, sid: int, rep: int) -> pd.DataFrame:
    """
    Produces one-row-per-(exp,sid,rep) summary that helps catch:
      - loss blowups
      - KL dominating
      - masked/unmasked weirdness
      - NaNs
    """
    out: Dict[str, Any] = {"exp": exp, "sid": sid, "rep": rep}

    if df is None or df.empty:
        return pd.DataFrame([out])

    # Common metric names you used in LitVAE:
    # train/loss, val/loss, train/kl, val/kl,
    # train/recon_weighted, val/recon_weighted,
    # train/recon_masked, val/recon_masked,
    # train/recon_unmasked, val/recon_unmasked,
    # val/recon_nomask_all, val/ratio_masked_over_nomask
    # But Lightning may log without prefix depending on your log calls.
    cols = list(df.columns)

    def pick(prefix: str, name: str) -> List[str]:
        # allow exact or with stage prefixes
        wanted = [
            f"{prefix}/{name}",
            f"{prefix}_{name}",
            f"{name}_{prefix}",
            name,  # last resort
        ]
        return [c for c in cols if c in wanted] or [c for c in cols if c.endswith(f"/{name}") or c.endswith(f"_{name}")]

    def last_value(candidates: List[str]) -> Optional[float]:
        for c in candidates:
            s = df[c].dropna()
            if len(s) > 0:
                return _as_float(s.iloc[-1])
        return None

    def max_value(candidates: List[str]) -> Optional[float]:
        for c in candidates:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) > 0:
                return float(np.nanmax(s.values))
        return None

    def min_value(candidates: List[str]) -> Optional[float]:
        for c in candidates:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) > 0:
                return float(np.nanmin(s.values))
        return None

    # Final values
    out["train_loss_last"] = last_value(pick("train", "loss"))
    out["val_loss_last"] = last_value(pick("val", "loss"))
    out["train_kl_last"] = last_value(pick("train", "kl"))
    out["val_kl_last"] = last_value(pick("val", "kl"))
    out["train_recon_weighted_last"] = last_value(pick("train", "recon_weighted"))
    out["val_recon_weighted_last"] = last_value(pick("val", "recon_weighted"))
    out["val_ratio_masked_over_nomask_last"] = last_value(pick("val", "ratio_masked_over_nomask"))

    # Extremes (blowups)
    out["train_loss_max"] = max_value(pick("train", "loss"))
    out["val_loss_max"] = max_value(pick("val", "loss"))
    out["train_kl_max"] = max_value(pick("train", "kl"))
    out["val_kl_max"] = max_value(pick("val", "kl"))

    # NaN detection
    numeric = df.apply(pd.to_numeric, errors="coerce")

    # Only consider columns that actually have at least one numeric value
    numeric_cols = [c for c in numeric.columns if numeric[c].notna().any()]

    if numeric_cols:
        num = numeric[numeric_cols].to_numpy()
        out["any_nan_numeric"] = bool(np.isnan(num).any())
        out["any_inf_numeric"] = bool(np.isinf(num).any())
    else:
        out["any_nan_numeric"] = None
        out["any_inf_numeric"] = None

    # Approx “KL share” at end if available
    if out["val_loss_last"] is not None and out["val_kl_last"] is not None and out["val_loss_last"] != 0:
        out["val_kl_over_loss_last"] = float(out["val_kl_last"] / (out["val_loss_last"] + 1e-12))
    else:
        out["val_kl_over_loss_last"] = None

    return pd.DataFrame([out])


# -------------------------
# resolved hparams parsing
# -------------------------

def parse_resolved_hparams(path: Path, exp: str, sid: int, rep: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"exp": exp, "sid": sid, "rep": rep}
    if not path.exists():
        return out
    try:
        d = yaml.safe_load(path.read_text()) or {}
        flat = flatten_dict(d)
        # Prefix hparams columns to avoid collisions
        for k, v in flat.items():
            out[f"hparam.{k}"] = v
    except Exception as e:
        out["hparam_parse_error"] = str(e)
    return out


# -------------------------
# scanning runs
# -------------------------

@dataclass(frozen=True)
class Run:
    exp: str
    sid: int
    rep: int
    run_dir: Path

def iter_runs(vae_basedir: Path) -> List[Run]:
    runs: List[Run] = []
    if not vae_basedir.exists():
        return runs

    # Expect: vae_basedir/<exp>/<sid>/rep<rep>/
    for exp_dir in sorted([p for p in vae_basedir.iterdir() if p.is_dir()]):
        exp = exp_dir.name
        # skip summaries folder
        if exp.startswith("_"):
            continue
        for sid_dir in sorted([p for p in exp_dir.iterdir() if p.is_dir()]):
            # sid should be int
            try:
                sid = int(sid_dir.name)
            except Exception:
                continue
            for rep_dir in sorted([p for p in sid_dir.iterdir() if p.is_dir() and p.name.startswith("rep")]):
                m = re.match(r"rep(\d+)$", rep_dir.name)
                if not m:
                    continue
                rep = int(m.group(1))
                runs.append(Run(exp=exp, sid=sid, rep=rep, run_dir=rep_dir))
    return runs


# -------------------------
# main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-basedir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--only-mse", action="store_true", help="Keep only metric==mse rows in outputs")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    runs = iter_runs(args.vae_basedir)

    recon_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    hparam_rows: List[Dict[str, Any]] = []

    n_runs = 0
    n_recon_files = 0
    n_recon_rows = 0
    n_metrics_files = 0
    n_resolved_yaml = 0
    n_train_summary = 0

    for run in runs:
        n_runs += 1
        exp, sid, rep, d = run.exp, run.sid, run.rep, run.run_dir

        recon_p = d / "diagnostics" / "recon_summary.txt"
        metrics_dir = d / "logs"
        metrics_df = load_metrics_csv(metrics_dir)
        resolved_p = d / "hparams.resolved.yaml"
        train_summary_p = d / "train_summary.json"

        # recon
        if recon_p.exists():
            n_recon_files += 1
            df_r = parse_recon_summary(recon_p, exp=exp, sid=sid, rep=rep)
            if not df_r.empty:
                recon_rows.extend(df_r.to_dict(orient="records"))
                n_recon_rows += len(df_r)

        # diagnostics (metrics.csv)
        if metrics_df is not None and not metrics_df.empty:
            n_metrics_files += 1
            df_d = summarize_diagnostics_from_metrics(metrics_df, exp=exp, sid=sid, rep=rep)
            diag_rows.extend(df_d.to_dict(orient="records"))
        else:
            # still create a stub row so you know it was missing
            diag_rows.append({"exp": exp, "sid": sid, "rep": rep, "metrics_missing": True})

        # hparams
        if resolved_p.exists():
            n_resolved_yaml += 1
        hparam_rows.append(parse_resolved_hparams(resolved_p, exp=exp, sid=sid, rep=rep))

        # train_summary (optional)
        if train_summary_p.exists():
            n_train_summary += 1
            try:
                j = json.loads(train_summary_p.read_text())
                # store a few common things if present
                # avoid stomping keys by prefixing
                for k, v in (j or {}).items():
                    # scalars only
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        hparam_rows[-1][f"train_summary.{k}"] = v
            except Exception:
                hparam_rows[-1]["train_summary_parse_error"] = True

    # ---- build DFs (always with columns) ----
    df_recon = pd.DataFrame(recon_rows, columns=RECON_LONG_COLS)
    if args.only_mse and not df_recon.empty:
        df_recon = df_recon[df_recon["metric"] == "mse"].copy()

    df_diag = pd.DataFrame(diag_rows)
    if df_diag.empty:
        df_diag = pd.DataFrame(columns=["exp", "sid", "rep"])

    df_hp = pd.DataFrame(hparam_rows)
    if df_hp.empty:
        df_hp = pd.DataFrame(columns=["exp", "sid", "rep"])

    # Ensure join keys exist
    for df in (df_diag, df_hp):
        for c in ["exp", "sid", "rep"]:
            if c not in df.columns:
                df[c] = pd.NA

    # ---- write outputs ----
    out_long = args.outdir / "vae_grid_long.csv"
    out_diag = args.outdir / "vae_grid_diagnostics.csv"
    out_hp   = args.outdir / "vae_grid_hparams.csv"
    out_join = args.outdir / "vae_grid_joined.csv"
    out_sum  = args.outdir / "vae_grid_summary.csv"

    df_recon.to_csv(out_long, index=False)
    df_diag.to_csv(out_diag, index=False)
    df_hp.to_csv(out_hp, index=False)

    df_join = df_recon.merge(df_diag, on=["exp", "sid", "rep"], how="left").merge(df_hp, on=["exp", "sid", "rep"], how="left")
    df_join.to_csv(out_join, index=False)

    # ---- aggregated summary over runs ----
    if df_recon.empty:
        df_summary = pd.DataFrame(columns=[
            "exp", "split", "where", "metric",
            "model_mean", "model_std",
            "maf_mean", "maf_std",
            "delta_mean", "delta_std",
            "ratio_mean", "ratio_std",
            "n",
        ])
    else:
        g = df_recon.groupby(["exp", "split", "where", "metric"], dropna=False)

        def _agg(x: pd.Series) -> Tuple[float, float]:
            xs = pd.to_numeric(x, errors="coerce").dropna().astype(float)
            if len(xs) == 0:
                return (np.nan, np.nan)
            if len(xs) == 1:
                return (float(xs.iloc[0]), np.nan)
            return (float(xs.mean()), float(xs.std(ddof=1)))

        rows = []
        for keys, sub in g:
            exp, split, where, metric = keys
            mm, ms = _agg(sub["model"])
            bm, bs = _agg(sub["maf_baseline"])
            dm, ds = _agg(sub["delta_vs_maf"])
            rm, rs = _agg(sub["ratio_vs_maf"])
            rows.append({
                "exp": exp,
                "split": split,
                "where": where,
                "metric": metric,
                "model_mean": mm,
                "model_std": ms,
                "maf_mean": bm,
                "maf_std": bs,
                "delta_mean": dm,
                "delta_std": ds,
                "ratio_mean": rm,
                "ratio_std": rs,
                "n": int(len(sub)),
            })

        df_summary = pd.DataFrame(rows)
        # Default sort: prefer beating MAF (delta negative) and smaller model MSE
        if args.only_mse:
            df_summary = df_summary.sort_values(
                by=["split", "where", "delta_mean", "model_mean"],
                ascending=[True, True, True, True],
                na_position="last",
            )

    df_summary.to_csv(out_sum, index=False)

    # ---- console report ----
    if not args.quiet:
        print("Wrote:")
        print(f"  {out_long}")
        print(f"  {out_sum}")
        print(f"  {out_diag}")
        print(f"  {out_hp}")
        print(f"  {out_join}")
        print()
        print("Scan report:")
        print(f"  runs scanned:               {n_runs}")
        print(f"  recon_summary.txt found:    {n_recon_files}")
        print(f"  recon rows parsed:          {n_recon_rows}")
        print(f"  metrics.csv found:          {n_metrics_files}")
        print(f"  hparams.resolved.yaml found:{n_resolved_yaml}")
        print(f"  train_summary.json found:   {n_train_summary}")
        print()

        if df_recon.empty:
            print("No recon rows parsed.")
            print("Likely causes:")
            print("  - diagnostics did not run for some experiments, OR")
            print("  - recon_summary.txt header format differs from expected.")
            print("Open one recon_summary.txt and ensure it has columns including:")
            print("  split metric model maf_baseline where")
            return

        # A helpful quick view: focus on YRI_target masked_only mse if present
        focus = df_summary
        if args.only_mse:
            focus = df_summary[(df_summary["metric"] == "mse")].copy()

        def show_block(split: str, where: str, k: int = 12) -> None:
            sub = focus[(focus["split"] == split) & (focus["where"] == where)]
            if sub.empty:
                return
            # Sort by delta then model_mean
            sub = sub.sort_values(by=["delta_mean", "model_mean"], ascending=[True, True], na_position="last").head(k)
            print(f"Top {k}: split={split} where={where} (best = most negative delta_mean, then smallest model_mean)")
            cols = ["exp", "model_mean", "maf_mean", "delta_mean", "ratio_mean", "n"]
            print(sub[cols].to_string(index=False))
            print()

        show_block("CEU_val", "masked_only", k=12)
        show_block("YRI_target", "masked_only", k=12)
        show_block("YRI_target", "all", k=12)


if __name__ == "__main__":
    main()