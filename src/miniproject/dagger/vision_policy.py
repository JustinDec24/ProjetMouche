"""Small MLP policy mapping bio-inspired features to (turn_bias, speed_scale).

Design choices:
    * Compact 2-layer MLP (64 hidden units) -- fast CPU inference,
      no CNN because the features are already visual summaries.
    * Output squashing: turn via `TURN_SCALE * tanh(...)`, speed via
      `SPEED_SCALE * sigmoid(...)`. Keeps the policy output in the same
      numerical range as the scripted vision module it replaces.
    * Feature normalisation stats (mean/std) live inside the checkpoint,
      so `load()` returns a self-contained callable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .vision_features import N_FEATURES


class VisionPolicy(nn.Module):
    """MLP: (N_FEATURES,) -> (turn, speed).

    turn  in [-TURN_SCALE,  TURN_SCALE]   via tanh
    speed in [0,            SPEED_SCALE]  via sigmoid
    """

    TURN_SCALE = 2.0
    SPEED_SCALE = 1.0

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.n_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        # Normalisation stats; overwritten on load(). Registered as buffers
        # so they move with .to(device) and are saved in state_dict.
        self.register_buffer(
            "feat_mean", torch.zeros(self.n_features, dtype=torch.float32)
        )
        self.register_buffer(
            "feat_std", torch.ones(self.n_features, dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.feat_mean) / self.feat_std
        raw = self.net(x)
        turn = self.TURN_SCALE * torch.tanh(raw[..., 0:1])
        speed = self.SPEED_SCALE * torch.sigmoid(raw[..., 1:2])
        return torch.cat([turn, speed], dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, features: np.ndarray) -> tuple[float, float]:
        """Run the policy on a single feature vector (CPU is fine).

        Returns
        -------
        (turn_bias, speed_scale) as python floats.
        """
        x = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0)
        if next(self.parameters()).device.type != "cpu":
            x = x.to(next(self.parameters()).device)
        out = self.forward(x)[0].detach().cpu().numpy()
        return float(out[0]), float(out[1])

    # ------------------------------------------------------------------
    def set_normalisation(self, mean: np.ndarray, std: np.ndarray) -> None:
        m = torch.as_tensor(mean, dtype=torch.float32).view(-1)
        s = torch.as_tensor(std, dtype=torch.float32).view(-1)
        if m.numel() != self.n_features or s.numel() != self.n_features:
            raise ValueError(
                f"Normalisation must have {self.n_features} dims; "
                f"got mean={m.numel()}, std={s.numel()}"
            )
        self.feat_mean.copy_(m)
        self.feat_std.copy_(torch.clamp(s, min=1e-6))

    # ------------------------------------------------------------------
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "state_dict": self.state_dict(),
            "n_features": self.n_features,
            "hidden": self.hidden,
            "turn_scale": self.TURN_SCALE,
            "speed_scale": self.SPEED_SCALE,
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "VisionPolicy":
        path = Path(path)
        payload = torch.load(path, map_location=map_location, weights_only=False)
        n_features = int(payload.get("n_features", N_FEATURES))
        hidden = int(payload.get("hidden", 64))
        model = cls(n_features=n_features, hidden=hidden)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
