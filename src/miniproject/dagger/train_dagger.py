"""Supervised training / re-training for the DAgger vision policy.

Usage
-----
Initial training on demo set:
    python -m miniproject.dagger.train_dagger \
        --dataset miniproject/dagger/data/demos_0.npz \
        --out     miniproject/dagger/models/policy_0.pt

Aggregated re-training (DAgger step):
    python -m miniproject.dagger.train_dagger \
        --dataset miniproject/dagger/data/demos_0.npz \
                  miniproject/dagger/data/demos_1.npz \
        --out     miniproject/dagger/models/policy_1.pt

Key hyperparameters (CLI flags):
    --epochs 50        number of epochs
    --lr 1e-3          Adam learning rate
    --batch 128        mini-batch size
    --hidden 64        MLP width
    --l2-out 0.01      L2 penalty on the policy *outputs* (regularise
                       the action space -- reduces zig-zag behaviour)
    --val-frac 0.1     validation split fraction
    --turn-weight 1.0  loss weight for turn_bias component
    --speed-weight 1.0 loss weight for speed_scale component
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dagger_dataset import DaggerDataset, as_torch_dataset
from .vision_features import N_FEATURES
from .vision_policy import VisionPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the DAgger vision policy (MLP) on collected demos."
    )
    p.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="One or more .npz dataset files (will be concatenated).",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output path for the trained policy checkpoint (.pt).",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--l2-out", type=float, default=0.01)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--turn-weight", type=float, default=1.0)
    p.add_argument("--speed-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _train_one_epoch(
    model: VisionPolicy,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    *,
    turn_w: float,
    speed_w: float,
    l2_out: float,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    total = 0.0
    turn_sum = 0.0
    speed_sum = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        # Weighted MSE per output channel.
        turn_mse = F.mse_loss(pred[..., 0], y[..., 0])
        speed_mse = F.mse_loss(pred[..., 1], y[..., 1])
        loss = turn_w * turn_mse + speed_w * speed_mse
        # L2 regularisation on outputs (not weights) to discourage
        # large-amplitude commands; weighted uniformly across samples.
        if l2_out > 0.0:
            loss = loss + l2_out * (pred.pow(2).mean())
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        total += float(loss.item()) * x.size(0)
        turn_sum += float(turn_mse.item()) * x.size(0)
        speed_sum += float(speed_mse.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1), turn_sum / max(n, 1), speed_sum / max(n, 1)


@torch.no_grad()
def _val(
    model: VisionPolicy,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    turn_sum = 0.0
    speed_sum = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        turn_sum += float(F.mse_loss(pred[..., 0], y[..., 0]).item()) * x.size(0)
        speed_sum += float(F.mse_loss(pred[..., 1], y[..., 1]).item()) * x.size(0)
        n += x.size(0)
    return turn_sum / max(n, 1), speed_sum / max(n, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load + aggregate ---
    paths = [Path(p) for p in args.dataset]
    ds = DaggerDataset.load_many(paths)
    if len(ds) == 0:
        raise SystemExit("Empty dataset -- nothing to train on.")
    print(f"Loaded {len(ds)} labelled samples from {len(paths)} file(s).")
    print(f"  expert={int((ds.source == 0).sum())}  dagger={int((ds.source == 1).sum())}")

    # --- Split + normalisation ---
    train_ds, val_ds = ds.train_val_split(val_frac=args.val_frac, seed=args.seed)
    mean, std = train_ds.feature_stats()

    train_torch = as_torch_dataset(train_ds, mean, std)
    val_torch = as_torch_dataset(val_ds, mean, std) if len(val_ds) > 0 else None

    train_loader = DataLoader(
        train_torch, batch_size=args.batch, shuffle=True, drop_last=False
    )
    val_loader = (
        DataLoader(val_torch, batch_size=args.batch, shuffle=False)
        if val_torch is not None
        else None
    )

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionPolicy(n_features=N_FEATURES, hidden=args.hidden).to(device)
    model.set_normalisation(mean, std)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training loop ---
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_turn, train_speed = _train_one_epoch(
            model,
            train_loader,
            optim,
            turn_w=args.turn_weight,
            speed_w=args.speed_weight,
            l2_out=args.l2_out,
            device=device,
        )
        msg = (
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"train_turn_mse={train_turn:.4f}  train_speed_mse={train_speed:.4f}"
        )
        if val_loader is not None:
            val_turn, val_speed = _val(model, val_loader, device)
            val_total = args.turn_weight * val_turn + args.speed_weight * val_speed
            msg += f"  val_turn={val_turn:.4f}  val_speed={val_speed:.4f}"
            if val_total < best_val:
                best_val = val_total
                model.save(
                    args.out,
                    extra={
                        "best_val": best_val,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "dataset_paths": [str(p) for p in paths],
                        "n_samples": len(ds),
                    },
                )
                msg += "  [saved]"
        print(msg, flush=True)

    # Always save the final model as well (in case no val split was used
    # or the last epoch happened to be the best one).
    if val_loader is None:
        model.save(
            args.out,
            extra={
                "epoch": args.epochs,
                "dataset_paths": [str(p) for p in paths],
                "n_samples": len(ds),
            },
        )
    print(f"Saved policy to {args.out}")


if __name__ == "__main__":
    main()
