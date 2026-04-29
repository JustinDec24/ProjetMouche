"""Npz-backed dataset for DAgger demonstrations.

File format (npz):
    features      : (N, D) float32  -- feature vectors at each labelled step
    labels        : (N, 2) float32  -- (turn_label, speed_label) in conv. below
    source        : (N,)   int32    -- 0 = pure expert, 1 = DAgger correction
    episode       : (N,)   int32    -- episode id (monotonically increasing)
    meta (kv)     : feature_names, turn_scale, speed_scale, schema_version

Label convention (set by the interactive collector):
    turn_label  = 0.5 * (gain_right - gain_left)
    speed_label = clip(0.5 * (gain_left + gain_right), 0.0, 1.0)
    Bounds used for target squashing in the policy live in
    `VisionPolicy.TURN_SCALE` / `VisionPolicy.SPEED_SCALE`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .vision_features import FEATURE_NAMES, N_FEATURES

SCHEMA_VERSION = 1


class DaggerDataset:
    """In-memory dataset of (features, labels, source, episode)."""

    def __init__(
        self,
        features: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        source: np.ndarray | None = None,
        episode: np.ndarray | None = None,
    ) -> None:
        if features is None:
            features = np.zeros((0, N_FEATURES), dtype=np.float32)
        if labels is None:
            labels = np.zeros((0, 2), dtype=np.float32)
        if source is None:
            source = np.zeros((0,), dtype=np.int32)
        if episode is None:
            episode = np.zeros((0,), dtype=np.int32)
        assert features.shape[0] == labels.shape[0] == source.shape[0] == episode.shape[0]
        self.features = features.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.source = source.astype(np.int32, copy=False)
        self.episode = episode.astype(np.int32, copy=False)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            labels=self.labels,
            source=self.source,
            episode=self.episode,
            feature_names=np.array(FEATURE_NAMES, dtype=object),
            schema_version=np.int32(SCHEMA_VERSION),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DaggerDataset":
        path = Path(path)
        with np.load(path, allow_pickle=True) as data:
            feats = np.asarray(data["features"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.float32)
            source = np.asarray(data["source"], dtype=np.int32)
            episode = np.asarray(data["episode"], dtype=np.int32)
        if feats.ndim != 2 or feats.shape[1] != N_FEATURES:
            raise ValueError(
                f"Dataset at {path} has features shape {feats.shape}; "
                f"expected (N, {N_FEATURES})"
            )
        return cls(feats, labels, source, episode)

    @classmethod
    def load_many(cls, paths: Iterable[str | Path]) -> "DaggerDataset":
        parts = [cls.load(p) for p in paths]
        return cls.concat(parts)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def append(
        self, feature_vec: np.ndarray, label: tuple[float, float], source: int, episode: int
    ) -> None:
        fv = np.asarray(feature_vec, dtype=np.float32).reshape(1, -1)
        if fv.shape[1] != N_FEATURES:
            raise ValueError(
                f"feature_vec has {fv.shape[1]} dims, expected {N_FEATURES}"
            )
        lbl = np.asarray(label, dtype=np.float32).reshape(1, 2)
        self.features = np.concatenate([self.features, fv], axis=0)
        self.labels = np.concatenate([self.labels, lbl], axis=0)
        self.source = np.concatenate([self.source, np.array([source], dtype=np.int32)])
        self.episode = np.concatenate([self.episode, np.array([episode], dtype=np.int32)])

    def extend(self, other: "DaggerDataset") -> None:
        self.features = np.concatenate([self.features, other.features], axis=0)
        self.labels = np.concatenate([self.labels, other.labels], axis=0)
        self.source = np.concatenate([self.source, other.source])
        # Renumber episodes to stay monotonic.
        offset = int(self.episode.max()) + 1 if self.episode.size > 0 else 0
        self.episode = np.concatenate(
            [self.episode, other.episode + np.int32(offset)]
        )

    @classmethod
    def concat(cls, parts: Iterable["DaggerDataset"]) -> "DaggerDataset":
        out = cls()
        for p in parts:
            out.extend(p)
        return out

    # ------------------------------------------------------------------
    # Stats / normalisation
    # ------------------------------------------------------------------
    def feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) with std floored at 1e-6 to prevent div-by-zero."""
        if len(self) == 0:
            return (
                np.zeros((N_FEATURES,), dtype=np.float32),
                np.ones((N_FEATURES,), dtype=np.float32),
            )
        mean = self.features.mean(axis=0).astype(np.float32)
        std = self.features.std(axis=0).astype(np.float32)
        std = np.maximum(std, 1e-6).astype(np.float32)
        return mean, std

    def train_val_split(
        self, val_frac: float = 0.1, seed: int = 0
    ) -> tuple["DaggerDataset", "DaggerDataset"]:
        n = len(self)
        if n == 0:
            return DaggerDataset(), DaggerDataset()
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        k = max(1, int(round(val_frac * n))) if n > 1 else 0
        val_idx = idx[:k]
        train_idx = idx[k:]
        return (
            DaggerDataset(
                self.features[train_idx],
                self.labels[train_idx],
                self.source[train_idx],
                self.episode[train_idx],
            ),
            DaggerDataset(
                self.features[val_idx],
                self.labels[val_idx],
                self.source[val_idx],
                self.episode[val_idx],
            ),
        )


# ----------------------------------------------------------------------
# Optional torch wrapper (only imported if available, keeps import cheap).
# ----------------------------------------------------------------------
def as_torch_dataset(ds: DaggerDataset, mean: np.ndarray, std: np.ndarray):
    """Wrap a DaggerDataset as a torch.utils.data.Dataset.

    Note: features are returned RAW (un-normalised). The policy normalises
    internally via its `feat_mean`/`feat_std` buffers (set via
    `model.set_normalisation(mean, std)` before training). This matches the
    inference path (`policy.act(feat)` is called on raw features).
    `mean` / `std` are still accepted for API compatibility but unused here.
    Imported lazily so using the npz helpers alone doesn't force torch.
    """
    import torch
    from torch.utils.data import Dataset

    del mean, std  # normalisation lives inside the model

    feats = ds.features.astype(np.float32, copy=False)
    labels = ds.labels.astype(np.float32)

    class _Wrap(Dataset):
        def __len__(self) -> int:
            return feats.shape[0]

        def __getitem__(self, idx: int):
            return (
                torch.from_numpy(feats[idx]),
                torch.from_numpy(labels[idx]),
            )

    return _Wrap()
