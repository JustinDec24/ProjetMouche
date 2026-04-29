"""DAgger vision policy for the miniproject fly controller.

This package contains everything needed to collect human demonstrations,
train a small vision policy, and plug it back into the scripted controller
as a drop-in replacement for the hand-crafted vision avoidance module.

Public API
----------
- VisionFeatureExtractor : builds a compact feature vector from sim state.
- VisionPolicy           : small MLP mapping features -> (turn_bias, speed).
- DaggerDataset          : npz-backed dataset (features, labels, meta).
"""

from .vision_features import VisionFeatureExtractor, FEATURE_NAMES, N_FEATURES
from .vision_policy import VisionPolicy
from .dagger_dataset import DaggerDataset

__all__ = [
    "VisionFeatureExtractor",
    "FEATURE_NAMES",
    "N_FEATURES",
    "VisionPolicy",
    "DaggerDataset",
]
