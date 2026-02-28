#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# --- ensure "import src.*" works ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transformer.data import HapPairDataset, collate_happairbatch
from src.transformer.model import GenotypeTransformer


# -----------------------------
# helpers
# -----------------------------
def load_cfg(cfg_path: str) -> dict:
    cfg = yaml.safe_load(Path(cfg_path).read_text()) or {}
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg.setdefault("masking", {})
    cfg.setdefault("data", {})
    return cfg


def make_splits(n: int, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(val_frac * n))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_batch_fields(batch):
    """
    HapPairBatch fields: genotype, hap1, hap2, pad_mask
    Support dict-like or dataclass-like.
    """
    if isinstance(batch, dict):
        hap1 = batch["hap1"]
        hap2 = batch["hap2"]
        pad_mask = batch.get("pad_mask", None)
    else:
        hap1 = batch.hap1
        hap2 = batch.hap2
        pad_mask = getattr(batch, "pad_mask", None)
    return hap1, hap2, pad_mask


def make_fixed_mask(B, L, p_mask_site, device, seed=123):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return (torch.rand((B, L), device=device, generator=g) < float(p_mask_site))


def apply_mask(hap: torch.Tensor, mask: torch.Tensor, mask_id: int):
    x = hap.clone()
    x[mask] = int(mask_id)
    return x


def parse_model_output(out):
    """
    Your model returns (token_logits, pooled_embedding) OR dict-like outputs.
    We only want token logits of shape (B, L, V).
    """
    if isinstance(out, (tuple, list)):
        if len(out) >= 1:
            return out[0]
    if isinstance(out, dict):
        for k in ["logits", "token_logits", "mlm_logits", "logits1"]:
            if k in out:
                return out[k]
    raise RuntimeError(f"Unexpected model output type from forward(): {type(out)}")


def forward_logits(model, x1, x2, pad_mask):
    """Call model forward with best-effort pad_mask handling."""
    try:
        out = model(x1, x2, pad_mask=pad_mask)
    except TypeError:
        out = model(x1, x2, pad_mask)
    logits = parse_model_output(out)
    if logits.ndim != 3:
        raise RuntimeError(f"Expected token logits (B,L,V). Got {tuple(logits.shape)}")
    return logits


def loss_mlm_fixedmask(model, hap1, hap2, pad_mask, device, mask_id, mask1, mask2):
    """
    Masked loss: replace masked tokens with mask_id, compute CE only at masked sites.
    Uses ONE logits head (B,L,V) and scores it against hap1 AND hap2 masked targets.
    """
    model.train()
    hap1 = hap1.to(device)
    hap2 = hap2.to(device)
    if pad_mask is not None:
        pad_mask = pad_mask.to(device)

    y1 = hap1
    y2 = hap2

    x1 = apply_mask(hap1, mask1, mask_id)
    x2 = apply_mask(hap2, mask2, mask_id)

    logits = forward_logits(model, x1, x2, pad_mask)
    V = logits.shape[-1]

    loss1 = F.cross_entropy(
        logits[mask1].reshape(-1, V),
        y1[mask1].reshape(-1),
        reduction="mean",
    ) if mask1.any() else torch.tensor(0.0, device=device)

    loss2 = F.cross_entropy(
        logits[mask2].reshape(-1, V),
        y2[mask2].reshape(-1),
        reduction="mean",
    ) if mask2.any() else torch.tensor(0.0, device=device)

    return (loss1 + loss2) / 2.0


def loss_identity_alltokens(model, hap1, hap2, pad_mask, device, target: str):
    """
    Identity/memorization test:
      - NO masking
      - CE over ALL positions vs hap1 or hap2
    This is the real "can it overfit a batch?" sanity check.
    """
    model.train()
    hap1 = hap1.to(device)
    hap2 = hap2.to(device)
    if pad_mask is not None:
        pad_mask = pad_mask.to(device)

    logits = forward_logits(model, hap1, hap2, pad_mask)
    V = logits.shape[-1]

    y = hap1 if target == "hap1" else hap2
    return F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="mean")


@torch.no_grad()
def eval_loss(model, loader, device, objective: str, mask_id: int, p_mask_site: float, mask_seed: int):
    """
    Validation loss.
    - For MLM: uses a fixed mask seed for determinism-ish.
    - For identity: CE on all tokens vs chosen hap.
    """
    model.eval()
    losses = []

    for batch in loader:
        hap1, hap2, pad_mask = get_batch_fields(batch)
        hap1 = hap1.to(device)
        hap2 = hap2.to(device)
        if pad_mask is not None:
            pad_mask = pad_mask.to(device)

        B, L = hap1.shape

        if objective == "mlm":
            mask = make_fixed_mask(B, L, p_mask_site, device=device, seed=mask_seed)
            mask1 = mask
            mask2 = mask

            x1 = apply_mask(hap1, mask1, mask_id)
            x2 = apply_mask(hap2, mask2, mask_id)

            logits = forward_logits(model, x1, x2, pad_mask)
            V = logits.shape[-1]

            loss1 = F.cross_entropy(
                logits[mask1].reshape(-1, V),
                hap1[mask1].reshape(-1),
                reduction="mean",
            ) if mask1.any() else torch.tensor(0.0, device=device)

            loss2 = F.cross_entropy(
                logits[mask2].reshape(-1, V),
                hap2[mask2].reshape(-1),
                reduction="mean",
            ) if mask2.any() else torch.tensor(0.0, device=device)

            loss = (loss1 + loss2) / 2.0

        elif objective == "identity_hap1":
            logits = forward_logits(model, hap1, hap2, pad_mask)
            V = logits.shape[-1]
            loss = F.cross_entropy(logits.reshape(-1, V), hap1.reshape(-1), reduction="mean")

        elif objective == "identity_hap2":
            logits = forward_logits(model, hap1, hap2, pad_mask)
            V = logits.shape[-1]
            loss = F.cross_entropy(logits.reshape(-1, V), hap2.reshape(-1), reduction="mean")

        else:
            raise ValueError(f"Unknown objective: {objective}")

        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hap1", required=True)
    ap.add_argument("--hap2", required=True)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--objective",
        type=str,
        default="mlm",
        choices=["mlm", "identity_hap1", "identity_hap2"],
        help="mlm=masked loss. identity_* = no masking, CE on all tokens vs hap1/hap2.",
    )

    ap.add_argument("--mask_seed", type=int, default=123, help="mask seed for training batch (mlm)")
    ap.add_argument("--val_mask_seed", type=int, default=999, help="mask seed for validation (mlm)")
    ap.add_argument("--no_val", action="store_true", help="skip validation loss computation (faster)")

    ap.add_argument("--override_dropout", type=float, default=None, help="override model dropout (debug)")
    ap.add_argument("--override_weight_decay", type=float, default=None, help="override weight_decay (debug)")
    ap.add_argument("--override_lr", type=float, default=None, help="override lr (debug)")

    args = ap.parse_args()

    cfg = load_cfg(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    mask_cfg = cfg["masking"]

    device = get_device()
    print(f"[debug] device={device}")

    # load hap arrays
    hap1_all = torch.from_numpy(np.load(args.hap1)).long()
    hap2_all = torch.from_numpy(np.load(args.hap2)).long()
    N, Ltot = hap1_all.shape
    print(f"[debug] hap shape N={N} Ltot={Ltot}")

    # windowing
    window_len = int(data_cfg.get("window_len", 512))
    window_mode = str(data_cfg.get("window_mode", "random"))
    fixed_start = int(data_cfg.get("fixed_start", 0))
    print(f"[debug] window_len={window_len} window_mode={window_mode} fixed_start={fixed_start}")

    # dataset rng for reproducible windows
    g = torch.Generator()
    g.manual_seed(int(train_cfg.get("seed", args.seed)))

    ds = HapPairDataset(
        hap1_all,
        hap2_all,
        pad_id=None,
        window_len=window_len,
        window_mode=window_mode,
        fixed_start=fixed_start,
        rng=g,
    )

    train_idx, val_idx = make_splits(len(ds), args.val_frac, seed=args.seed)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    num_workers = int(train_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,  # important: stable batch for overfit test
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_happairbatch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_happairbatch,
    )

    # grab ONE fixed batch
    one_batch = next(iter(train_loader))
    hap1, hap2, pad_mask = get_batch_fields(one_batch)
    print(
        f"[debug] one_batch shapes: hap1={tuple(hap1.shape)} hap2={tuple(hap2.shape)} "
        f"pad_mask={None if pad_mask is None else tuple(pad_mask.shape)}"
    )

    # build model
    dropout = float(model_cfg.get("dropout", 0.1))
    if args.override_dropout is not None:
        dropout = float(args.override_dropout)

    model = GenotypeTransformer(
        vocab_size=int(model_cfg.get("vocab_size", 3)),
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers=int(model_cfg["n_layers"]),
        dropout=dropout,
    ).to(device)

    lr = float(train_cfg.get("lr", 3e-4))
    if args.override_lr is not None:
        lr = float(args.override_lr)

    wd = float(train_cfg.get("weight_decay", 0.0))
    if args.override_weight_decay is not None:
        wd = float(args.override_weight_decay)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # masking params (for mlm only)
    mask_id = int(mask_cfg.get("mask_id", 2))
    p_mask_site = float(mask_cfg.get("p_mask_site", 0.15))

    # fixed mask for the one-batch training loop
    B, L = hap1.shape
    fixed_mask = make_fixed_mask(B, L, p_mask_site, device=device, seed=args.mask_seed)
    mask1 = fixed_mask
    mask2 = fixed_mask

    if args.objective == "mlm":
        print(f"[debug] objective=mlm fixed mask frac={fixed_mask.float().mean().item():.3f}")
    else:
        print(f"[debug] objective={args.objective} (no masking)")

    # -----------------------------
    # overfit loop
    # -----------------------------
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)

        if args.objective == "mlm":
            loss = loss_mlm_fixedmask(
                model=model,
                hap1=hap1,
                hap2=hap2,
                pad_mask=pad_mask,
                device=device,
                mask_id=mask_id,
                mask1=mask1,
                mask2=mask2,
            )
        elif args.objective == "identity_hap1":
            loss = loss_identity_alltokens(
                model=model, hap1=hap1, hap2=hap2, pad_mask=pad_mask, device=device, target="hap1"
            )
        elif args.objective == "identity_hap2":
            loss = loss_identity_alltokens(
                model=model, hap1=hap1, hap2=hap2, pad_mask=pad_mask, device=device, target="hap2"
            )
        else:
            raise ValueError(f"Unknown objective: {args.objective}")

        loss.backward()
        opt.step()

        if step == 1 or step % args.log_every == 0:
            train_loss = float(loss.detach().cpu())

            if args.no_val:
                dt = time.time() - t0
                print(f"[step {step:04d}] train_onebatch_loss={train_loss:.6f}  elapsed={dt:.1f}s")
            else:
                val_loss = eval_loss(
                    model=model,
                    loader=val_loader,
                    device=device,
                    objective=args.objective,
                    mask_id=mask_id,
                    p_mask_site=p_mask_site,
                    mask_seed=args.val_mask_seed,
                )
                dt = time.time() - t0
                print(
                    f"[step {step:04d}] train_onebatch_loss={train_loss:.6f}  "
                    f"val_loss={val_loss:.6f}  elapsed={dt:.1f}s"
                )

    print("[debug] done")


if __name__ == "__main__":
    main()
