#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import is_dataclass
from pathlib import Path
import sys
import yaml
import numpy as np
import torch

# --- ensure "import src.*" works ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transformer.data import HapPairDataset, collate_happairbatch
from src.transformer.model import GenotypeTransformer


def load_cfg(cfg_path: str) -> dict:
    cfg = yaml.safe_load(Path(cfg_path).read_text()) or {}
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg.setdefault("masking", {})
    cfg.setdefault("data", {})
    return cfg


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_to_tensors(batch):
    """
    HapPairBatch in your project seems to have: hap1, hap2, pad_mask.
    Support dataclass-like or dict-like.
    """
    if isinstance(batch, dict):
        hap1 = batch["hap1"]
        hap2 = batch["hap2"]
        pad_mask = batch.get("pad_mask", None)
    else:
        hap1 = getattr(batch, "hap1")
        hap2 = getattr(batch, "hap2")
        pad_mask = getattr(batch, "pad_mask", None)
    return hap1, hap2, pad_mask


def _tensor_summary(x: torch.Tensor) -> dict:
    # Avoid expensive ops on huge tensors unless asked
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
    }


def _maybe_stats(x: torch.Tensor) -> dict:
    # compute basic stats safely (float only)
    if not x.is_floating_point():
        return {}
    with torch.no_grad():
        xf = x.detach()
        return {
            "mean": float(xf.mean().cpu()),
            "std": float(xf.std(unbiased=False).cpu()),
            "min": float(xf.min().cpu()),
            "max": float(xf.max().cpu()),
        }


def _as_list_shapes(obj):
    """
    Convert arbitrary module output to a serializable description:
    tensors -> summary, tuples/lists -> recurse, dict -> recurse, other -> type str.
    """
    if torch.is_tensor(obj):
        return {"tensor": _tensor_summary(obj)}
    if isinstance(obj, (tuple, list)):
        return {"type": type(obj).__name__, "items": [_as_list_shapes(o) for o in obj]}
    if isinstance(obj, dict):
        return {"type": "dict", "items": {k: _as_list_shapes(v) for k, v in obj.items()}}
    return {"type": str(type(obj))}


class HookRecorder:
    def __init__(self, save_dir: Path | None, save_tensors: bool, save_max_mb: float, with_stats: bool):
        self.save_dir = save_dir
        self.save_tensors = save_tensors
        self.save_max_bytes = int(save_max_mb * 1024 * 1024)
        self.with_stats = with_stats
        self.records = []
        self.handles = []

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / "tensors").mkdir(parents=True, exist_ok=True)

    def _can_save_tensor(self, t: torch.Tensor) -> bool:
        return t.numel() * t.element_size() <= self.save_max_bytes

    def add_hook(self, name: str, module: torch.nn.Module):
        def hook(mod, inp, out):
            rec = {"name": name, "module": mod.__class__.__name__}

            # inputs: tuple of args
            inp_desc = []
            for x in inp:
                if torch.is_tensor(x):
                    d = _tensor_summary(x)
                    if self.with_stats:
                        d.update(_maybe_stats(x))
                    inp_desc.append(d)
                else:
                    inp_desc.append({"type": str(type(x))})
            rec["inputs"] = inp_desc

            # outputs: may be tensor/tuple/dict
            rec["output_desc"] = _as_list_shapes(out)

            # optionally save output tensor(s)
            if self.save_tensors and self.save_dir is not None:
                saved = []
                # only save direct tensors (and maybe first tensor in tuple)
                to_save = []
                if torch.is_tensor(out):
                    to_save = [("out", out)]
                elif isinstance(out, (tuple, list)):
                    for i, o in enumerate(out):
                        if torch.is_tensor(o):
                            to_save.append((f"out_{i}", o))
                elif isinstance(out, dict):
                    for k, o in out.items():
                        if torch.is_tensor(o):
                            to_save.append((f"out_{k}", o))

                for tag, t in to_save:
                    t_det = t.detach()
                    if self._can_save_tensor(t_det):
                        fn = f"{name.replace('.', '__')}__{tag}.pt"
                        path = self.save_dir / "tensors" / fn
                        torch.save(t_det.cpu(), path)
                        saved.append(str(path))
                    else:
                        saved.append(f"{tag}:SKIPPED_TOO_LARGE({t_det.numel()*t_det.element_size()/1e6:.1f}MB)")
                rec["saved"] = saved

            self.records.append(rec)

        self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hap1", required=True)
    ap.add_argument("--hap2", required=True)

    ap.add_argument("--sample_index", type=int, default=0, help="Which individual/window sample to trace (index into dataset).")
    ap.add_argument("--batch_size", type=int, default=1, help="Use 1 for clean tracing.")
    ap.add_argument("--force_cpu", action="store_true")

    ap.add_argument("--save_dir", type=str, default=None, help="If set, write trace.json and (optionally) tensors/")
    ap.add_argument("--save_tensors", action="store_true", help="Save intermediate outputs as .pt (can be large).")
    ap.add_argument("--save_max_mb", type=float, default=50.0, help="Max size per saved tensor.")
    ap.add_argument("--with_stats", action="store_true", help="Compute mean/std/min/max for floating outputs (slower).")

    ap.add_argument("--print_modules", action="store_true", help="Print module name list and exit (to decide what to hook).")
    ap.add_argument("--hook_filter", type=str, default=None,
                    help="Regex: only hook modules whose name matches (e.g. 'encoder|attn|embed|mlp').")

    args = ap.parse_args()

    cfg = load_cfg(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    device = get_device(force_cpu=args.force_cpu)
    print(f"[trace] device={device}")

    # Load haps
    hap1_all = torch.from_numpy(np.load(args.hap1)).long()
    hap2_all = torch.from_numpy(np.load(args.hap2)).long()
    N, Ltot = hap1_all.shape
    print(f"[trace] hap arrays: N={N} Ltot={Ltot}")

    # Dataset windowing
    window_len = int(data_cfg.get("window_len", 512))
    window_mode = str(data_cfg.get("window_mode", "random"))
    fixed_start = int(data_cfg.get("fixed_start", 0))
    print(f"[trace] window_len={window_len} window_mode={window_mode} fixed_start={fixed_start}")

    g = torch.Generator()
    g.manual_seed(int(train_cfg.get("seed", 42)))

    ds = HapPairDataset(
        hap1_all, hap2_all,
        pad_id=None,
        window_len=window_len,
        window_mode=window_mode,
        fixed_start=fixed_start,
        rng=g,
    )

    # Take one sample, wrap into a 1-item batch via collate
    idx = int(args.sample_index)
    sample = ds[idx]               # likely returns (hap1,hap2,...) object
    batch = collate_happairbatch([sample])  # make a batch of size 1
    hap1, hap2, pad_mask = batch_to_tensors(batch)

    print(f"[trace] one batch: hap1={tuple(hap1.shape)} hap2={tuple(hap2.shape)} pad_mask={None if pad_mask is None else tuple(pad_mask.shape)}")

    # Build model
    model = GenotypeTransformer(
        vocab_size=int(model_cfg.get("vocab_size", 3)),
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers=int(model_cfg["n_layers"]),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)
    model.eval()

    if args.print_modules:
        for name, mod in model.named_modules():
            if name == "":
                continue
            print(f"{name:60s}  {mod.__class__.__name__}")
        return

    # Recorder + hooks
    save_dir = Path(args.save_dir) if args.save_dir else None
    rec = HookRecorder(save_dir=save_dir, save_tensors=args.save_tensors, save_max_mb=args.save_max_mb, with_stats=args.with_stats)

    import re
    filt = re.compile(args.hook_filter) if args.hook_filter else None

    # Hook common “interesting” modules, or everything matched by filter
    for name, mod in model.named_modules():
        if name == "":
            continue
        # If no filter: hook high-signal modules only (keeps output readable)
        if filt is None:
            if any(k in name.lower() for k in ["embed", "encoder", "attn", "mlp", "ff", "norm", "head"]):
                rec.add_hook(name, mod)
        else:
            if filt.search(name):
                rec.add_hook(name, mod)

    # Forward pass
    with torch.no_grad():
        x1 = hap1.to(device)
        x2 = hap2.to(device)
        if pad_mask is not None:
            pad_mask = pad_mask.to(device)

        # call forward in the style your model expects
        try:
            out = model(x1, x2, pad_mask=pad_mask)
        except TypeError:
            out = model(x1, x2, pad_mask)

    # Print a compact trace
    print("\n=== TRACE (module -> output) ===")
    for r in rec.records:
        # show only module name and output desc in a compact way
        out_desc = r["output_desc"]
        print(f"{r['name']:55s} {r['module']:25s} -> {json.dumps(out_desc)[:200]}")

    # Save trace
    if save_dir is not None:
        trace = {
            "config": str(args.config),
            "sample_index": idx,
            "batch_shapes": {
                "hap1": list(hap1.shape),
                "hap2": list(hap2.shape),
                "pad_mask": None if pad_mask is None else list(pad_mask.shape),
            },
            "records": rec.records,
        }
        (save_dir / "trace.json").write_text(json.dumps(trace, indent=2))
        print(f"\n[trace] wrote {save_dir/'trace.json'}")
        if args.save_tensors:
            print(f"[trace] wrote tensors into {save_dir/'tensors'} (skipped anything > {args.save_max_mb}MB)")

    rec.close()


if __name__ == "__main__":
    main()
