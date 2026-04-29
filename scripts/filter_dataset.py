"""Filter a DAgger demo dataset by episode id.

Usage examples
--------------
Inspect the per-episode breakdown of an existing dataset:

    python scripts/filter_dataset.py --in miniproject/dagger/data/demos_scripted_1.npz --info

Keep only specific episode ids (the ones that succeeded), and write a new file:

    python scripts/filter_dataset.py \\
        --in  miniproject/dagger/data/demos_scripted_1.npz \\
        --out miniproject/dagger/data/demos_scripted_1_clean.npz \\
        --keep-episodes 0,2,13

Or drop specific episode ids (everything else is kept):

    python scripts/filter_dataset.py \\
        --in  miniproject/dagger/data/demos_scripted_1.npz \\
        --out miniproject/dagger/data/demos_scripted_1_clean.npz \\
        --drop-episodes 1,3,4,5,6,7,8,9,10,11,12,14,15,16,17

Episodes are renumbered to stay 0..K-1 in the output file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from miniproject.dagger import DaggerDataset


def parse_id_list(spec: str) -> list[int]:
    """Parse '0,2,13' or '0-3' into a list[int]."""
    spec = spec.strip()
    if not spec:
        return []
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return out


def print_info(ds: DaggerDataset, path: Path) -> None:
    print(f"Dataset: {path}")
    print(f"  total samples : {len(ds)}")
    print(f"  feature dim   : {ds.features.shape[1]}")
    if len(ds) == 0:
        return

    eps = np.unique(ds.episode)
    print(f"  episodes      : {len(eps)}  (ids {int(eps.min())}..{int(eps.max())})")
    print()
    print(f"  {'ep_id':>5}  {'n':>6}  {'turn_mean':>9}  {'turn_std':>8}  "
          f"{'speed_mean':>10}  {'speed_std':>9}")
    for ep in eps:
        mask = ds.episode == ep
        turns = ds.labels[mask, 0]
        speeds = ds.labels[mask, 1]
        print(
            f"  {int(ep):>5}  {int(mask.sum()):>6}  "
            f"{turns.mean():+9.3f}  {turns.std():>8.3f}  "
            f"{speeds.mean():>10.3f}  {speeds.std():>9.3f}"
        )


def filter_dataset(
    ds: DaggerDataset,
    keep: list[int] | None,
    drop: list[int] | None,
) -> DaggerDataset:
    if keep is not None and drop is not None:
        raise ValueError("Use --keep-episodes OR --drop-episodes, not both.")

    eps = np.unique(ds.episode)
    if keep is not None:
        keep_set = set(int(x) for x in keep)
        unknown = keep_set - set(int(e) for e in eps)
        if unknown:
            print(f"[WARN] --keep-episodes contains unknown ids: {sorted(unknown)}")
        mask = np.isin(ds.episode, np.array(sorted(keep_set), dtype=np.int32))
    elif drop is not None:
        drop_set = set(int(x) for x in drop)
        unknown = drop_set - set(int(e) for e in eps)
        if unknown:
            print(f"[WARN] --drop-episodes contains unknown ids: {sorted(unknown)}")
        mask = ~np.isin(ds.episode, np.array(sorted(drop_set), dtype=np.int32))
    else:
        mask = np.ones(len(ds), dtype=bool)

    feats = ds.features[mask]
    labels = ds.labels[mask]
    source = ds.source[mask]
    episode = ds.episode[mask]

    if episode.size > 0:
        unique_eps = np.unique(episode)
        remap = {int(old): new for new, old in enumerate(unique_eps)}
        episode = np.array([remap[int(e)] for e in episode], dtype=np.int32)

    return DaggerDataset(feats, labels, source, episode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--in", dest="in_path", type=str, required=True)
    p.add_argument("--out", dest="out_path", type=str, default=None)
    p.add_argument(
        "--info",
        action="store_true",
        help="Only print per-episode stats, do not write any output.",
    )
    p.add_argument(
        "--keep-episodes",
        type=str,
        default=None,
        help="Comma-separated list / range of episode ids to KEEP, e.g. '0,2,13' or '0-3'.",
    )
    p.add_argument(
        "--drop-episodes",
        type=str,
        default=None,
        help="Comma-separated list / range of episode ids to DROP.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path)
    ds = DaggerDataset.load(in_path)

    if args.info:
        print_info(ds, in_path)
        return

    if args.out_path is None:
        raise SystemExit("--out is required unless --info is set")
    if args.keep_episodes is None and args.drop_episodes is None:
        raise SystemExit(
            "Provide --keep-episodes or --drop-episodes "
            "(or use --info to inspect first)."
        )

    keep = parse_id_list(args.keep_episodes) if args.keep_episodes else None
    drop = parse_id_list(args.drop_episodes) if args.drop_episodes else None

    print_info(ds, in_path)
    print()

    filtered = filter_dataset(ds, keep=keep, drop=drop)

    out_path = Path(args.out_path)
    filtered.save(out_path)

    print(f"Filtered: {len(ds)} -> {len(filtered)} samples")
    print(f"Saved to: {out_path}")
    print()
    print_info(filtered, out_path)


if __name__ == "__main__":
    main()
