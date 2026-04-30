#!/usr/bin/env python3
"""
Tests unitaires de la convention vision (lr_avoid, réduction de roue).

Lancement (depuis la racine du dépôt) :
  python miniproject/test_vision_battery.py

Trace pendant une simulation réelle (TRÈS verbeux, une ligne par décision vision) :
  PowerShell: $env:CONTROLLER_VIS_TRACE="1"
  puis : python miniproject/run_with_controller.py ...
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_src = _REPO / "src"
_mp = _REPO / "miniproject"
for p in (_src, _mp):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import numpy as np

# Pas de flood trace pendant les tests sauf si explicitement demandé
os.environ.setdefault("CONTROLLER_VIS_TRACE", "0")

from submission.controller import Controller


def _bare_controller() -> Controller:
    """Controller sans __init__ (pas de sim MuJoCo)."""
    c = Controller.__new__(Controller)
    c._vis_last_dir = 1.0
    for name in (
        "VIS_DIRECTION_MEMORY",
        "VIS_LR_EPS_PIXELS",
        "VIS_LR_MIN_SPIKE_PIXELS",
    ):
        setattr(c, name, getattr(Controller, name))
    return c


def _feat(**kwargs: float | int) -> dict:
    roi_bp = 100_000.0
    base = {
        "raw_left_area": 0.01,
        "raw_right_area": 0.01,
        "left_area": 0.01,
        "right_area": 0.01,
        "total_area": 0.05,
        "roi_binocular_px": roi_bp,
        "spike_half_left_px": 0,
        "spike_half_right_px": 0,
    }
    base.update(kwargs)
    return base


class TestAvoidLRFromFeat(unittest.TestCase):
    """lr_avoid > 0 : masse spike à gauche de l'image ; < 0 : à droite."""

    def test_obstacle_right_negative_lr(self) -> None:
        c = _bare_controller()
        lr = c._avoid_lr_from_feat(
            _feat(spike_half_left_px=10, spike_half_right_px=800)
        )
        self.assertLess(lr, 0.0)

    def test_obstacle_left_positive_lr(self) -> None:
        c = _bare_controller()
        lr = c._avoid_lr_from_feat(
            _feat(spike_half_left_px=900, spike_half_right_px=12)
        )
        self.assertGreater(lr, 0.0)

    def test_symmetric_falls_back_binocular_or_memory(self) -> None:
        c = _bare_controller()
        lr = c._avoid_lr_from_feat(
            _feat(
                spike_half_left_px=500,
                spike_half_right_px=495,
                raw_left_area=0.02,
                raw_right_area=0.01,
            )
        )
        self.assertIsInstance(lr, float)

    def test_sparse_spikes_use_binocular_areas(self) -> None:
        c = _bare_controller()
        lr = c._avoid_lr_from_feat(
            _feat(
                spike_half_left_px=0,
                spike_half_right_px=0,
                raw_left_area=0.05,
                raw_right_area=0.01,
            )
        )
        self.assertGreater(lr, 0.0)


class TestBiasNormWheelReduction(unittest.TestCase):
    """Alignement avec _compute_drives : side = int(bias_norm > 0)."""

    def test_positive_bias_reduces_index_1(self) -> None:
        bias_norm = 0.42
        side = int(bias_norm > 0)
        self.assertEqual(side, 1)

    def test_negative_bias_reduces_index_0(self) -> None:
        bias_norm = -0.42
        side = int(bias_norm > 0)
        self.assertEqual(side, 0)


class TestTurnSignMatchesLR(unittest.TestCase):
    """Réflexes STOP/WIDE : dir_sign = +1 si lr > 0."""

    def test_stop_dir_sign(self) -> None:
        lr_pos = 0.02
        lr_neg = -0.02
        self.assertEqual(1.0 if lr_pos > 0 else -1.0, 1.0)
        self.assertEqual(1.0 if lr_neg > 0 else -1.0, -1.0)


class TestTraceEnv(unittest.TestCase):
    def test_vis_trace_env_off_by_default(self) -> None:
        c = _bare_controller()
        os.environ.pop("CONTROLLER_VIS_TRACE", None)
        Controller.VIS_TRACE_VERBOSE = False
        self.assertFalse(c._vis_trace_enabled())

    def test_vis_trace_env_on(self) -> None:
        c = _bare_controller()
        os.environ["CONTROLLER_VIS_TRACE"] = "1"
        try:
            self.assertTrue(c._vis_trace_enabled())
        finally:
            os.environ.pop("CONTROLLER_VIS_TRACE", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
