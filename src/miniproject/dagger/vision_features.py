"""Bio-inspired feature extractor for the DAgger vision policy.

The extractor reads *real* observations from a `MiniprojectSimulation`
instance and returns a compact, fixed-size float32 vector. The feature
layout mirrors (and refactors) the hand-crafted logic already present
in `miniproject/submission/controller.py::_vision_avoid_bias_and_danger`,
so the learned policy has access to the same signals the scripted
module used to decide on a turn bias + speed scale.

Feature order (see `FEATURE_NAMES`):
    0  left_area        - mean obstacle strength in left-eye ROI
    1  right_area       - mean obstacle strength in right-eye ROI
    2  center_area      - mean obstacle strength in center columns (both eyes)
    3  total_area       - EMA of 0.5 * (left_area + right_area)
    4  d_total_area     - EMA of temporal derivative of total_area (looming)
    5  left_x           - azimuth centroid of left-eye obstacle mass in [-1, 1]
    6  right_x          - azimuth centroid of right-eye obstacle mass in [-1, 1]
    7  odor_grad        - log-olfaction left minus right (weighted palp+antenna)
    8  mean_log_odor    - mean of log-olfaction (weak odor <-> big negative)
    9  target_bearing   - signed angle to banana in body frame, rad in [-pi, pi]
   10  target_dist_log  - log(1 + distance to banana)
   11  contact_fmax     - max horizontal external contact force (EMA smoothed)
   12  d_contact_fmax   - temporal derivative of contact_fmax (EMA smoothed)
   13  prev_turn        - last turn output of the policy (memory)
   14  prev_speed       - last speed_scale output of the policy (memory)

Notes on stability:
    * All moving averages live *inside* the extractor; resetting it at
      episode boundaries is important. Call `reset()` after `sim.reset()`.
    * Vision reads are cached by flygym at 500 Hz; extracting features
      every decision step (~20 Hz) is cheap.
"""

from __future__ import annotations

import numpy as np

# ROI / strength defaults mirror the controller constants so the learned
# policy "sees" the same thing as the hand-crafted module.
_VIS_ROI_R0 = 0.08
_VIS_ROI_R1 = 0.72
_VIS_ROI_C0 = 0.08
_VIS_ROI_C1 = 0.92
_VIS_CENTER_C0 = 0.40
_VIS_CENTER_C1 = 0.60
_VIS_DARK = 0.12
_VIS_EDGE = 0.10
_VIS_EMA = 0.70
_VIS_D_EMA = 0.80

# Olfaction weights (same as controller: PALP=9, ANTENNA=1).
_PALP_W = 9.0
_ANTENNA_W = 1.0

FEATURE_NAMES = [
    "left_area",
    "right_area",
    "center_area",
    "total_area",
    "d_total_area",
    "left_x",
    "right_x",
    "odor_grad",
    "mean_log_odor",
    "target_bearing",
    "target_dist_log",
    "contact_fmax",
    "d_contact_fmax",
    "prev_turn",
    "prev_speed",
]
N_FEATURES = len(FEATURE_NAMES)


def _to_float01(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.max() > 1.0:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def _obstacle_strength(gray: np.ndarray) -> np.ndarray:
    """Continuous obstacle proxy in [0, 1] from strong edges + dark pixels."""
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edge = 0.5 * (dx + dy)
    edge_thr = max(1e-6, float(_VIS_EDGE))
    edge_s = np.clip((edge - edge_thr) / edge_thr, 0.0, 1.0)
    dark_s = np.clip((_VIS_DARK - gray) / max(1e-6, _VIS_DARK), 0.0, 1.0)
    return np.maximum(edge_s, dark_s).astype(np.float32)


class VisionFeatureExtractor:
    """Build a compact fixed-size feature vector from simulator state."""

    def __init__(self, sim):
        # Discover the same indices the controller uses, from the sim object.
        fly_segs = sim.fly.get_bodysegs_order()
        self._thorax_idx = next(
            (i for i, s in enumerate(fly_segs) if s.name == "c_thorax"), 0
        )
        body_ids = sim._internal_bodyids_by_fly[sim.fly.name]
        self._thorax_body_id = int(body_ids[self._thorax_idx])
        self._contact_body_ids = sim._internal_contact_body_segment_ids_by_fly[
            sim.fly.name
        ]

        # Banana position might not be available yet (e.g. pre-spawn).
        try:
            self._banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
        except Exception:
            self._banana_xy = None

        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all EMA state. Call after `sim.reset()` or between episodes."""
        self._left_area_ema = 0.0
        self._right_area_ema = 0.0
        self._center_area_ema = 0.0
        self._total_area_ema = 0.0
        self._total_area_prev = 0.0
        self._d_total_area_ema = 0.0
        self._contact_ema = 0.0
        self._contact_prev = 0.0
        self._d_contact_ema = 0.0

    # ------------------------------------------------------------------
    def _roi(self, img01: np.ndarray) -> np.ndarray:
        h, w = img01.shape[0], img01.shape[1]
        r0, r1 = int(h * _VIS_ROI_R0), int(h * _VIS_ROI_R1)
        c0, c1 = int(w * _VIS_ROI_C0), int(w * _VIS_ROI_C1)
        return img01[r0:r1, c0:c1, :]

    @staticmethod
    def _area_x(strength: np.ndarray) -> tuple[float, float]:
        area = float(strength.mean()) if strength.size > 0 else 0.0
        if area <= 1e-9 or strength.size == 0:
            return 0.0, 0.0
        cols = strength.mean(axis=0)
        xs = np.linspace(-1.0, 1.0, cols.shape[0], dtype=np.float32)
        x_mean = float((cols * xs).sum() / max(1e-9, float(cols.sum())))
        return area, x_mean

    def _center_area(self, strength: np.ndarray) -> float:
        if strength.size == 0:
            return 0.0
        w = strength.shape[1]
        c0 = int(w * _VIS_CENTER_C0)
        c1 = int(w * _VIS_CENTER_C1)
        if c1 <= c0:
            return float(strength.mean())
        return float(strength[:, c0:c1].mean())

    # ------------------------------------------------------------------
    def _vision_block(self, sim) -> tuple[float, float, float, float, float, float, float]:
        """Return (left_area, right_area, center_area, total_area,
        d_total_area, left_x, right_x), updating EMAs in place."""
        frames = None
        try:
            frames = sim.get_raw_vision(sim.fly.name)
        except Exception:
            frames = None

        if frames is None or len(frames) == 0:
            # Keep EMAs but decay toward zero so stale values don't linger.
            self._total_area_ema *= _VIS_EMA
            self._left_area_ema *= _VIS_EMA
            self._right_area_ema *= _VIS_EMA
            self._center_area_ema *= _VIS_EMA
            return (
                float(self._left_area_ema),
                float(self._right_area_ema),
                float(self._center_area_ema),
                float(self._total_area_ema),
                float(self._d_total_area_ema),
                0.0,
                0.0,
            )

        left_img01 = _to_float01(frames[0])
        right_img01 = _to_float01(frames[1] if len(frames) > 1 else frames[0])

        lroi = self._roi(left_img01)
        rroi = self._roi(right_img01)
        lgray = (
            0.299 * lroi[..., 0] + 0.587 * lroi[..., 1] + 0.114 * lroi[..., 2]
        ).astype(np.float32)
        rgray = (
            0.299 * rroi[..., 0] + 0.587 * rroi[..., 1] + 0.114 * rroi[..., 2]
        ).astype(np.float32)
        lst = _obstacle_strength(lgray)
        rst = _obstacle_strength(rgray)

        left_area, left_x = self._area_x(lst)
        right_area, right_x = self._area_x(rst)
        center = 0.5 * (self._center_area(lst) + self._center_area(rst))
        total = 0.5 * (left_area + right_area)

        # EMAs (match controller semantics)
        self._left_area_ema = _VIS_EMA * self._left_area_ema + (1.0 - _VIS_EMA) * left_area
        self._right_area_ema = _VIS_EMA * self._right_area_ema + (1.0 - _VIS_EMA) * right_area
        self._center_area_ema = _VIS_EMA * self._center_area_ema + (1.0 - _VIS_EMA) * center

        prev_total = float(self._total_area_ema)
        self._total_area_ema = _VIS_EMA * prev_total + (1.0 - _VIS_EMA) * total
        d_raw = float(self._total_area_ema - self._total_area_prev)
        self._total_area_prev = float(self._total_area_ema)
        self._d_total_area_ema = (
            _VIS_D_EMA * self._d_total_area_ema + (1.0 - _VIS_D_EMA) * d_raw
        )

        return (
            float(self._left_area_ema),
            float(self._right_area_ema),
            float(self._center_area_ema),
            float(self._total_area_ema),
            float(self._d_total_area_ema),
            float(left_x),
            float(right_x),
        )

    # ------------------------------------------------------------------
    def _odor_block(self, sim) -> tuple[float, float]:
        try:
            odor_log = sim.get_olfaction(sim.fly.name, log=True)
        except Exception:
            return 0.0, 0.0
        lp, rp, la, ra = odor_log[:, 0]
        odor_l = _PALP_W * float(lp) + _ANTENNA_W * float(la)
        odor_r = _PALP_W * float(rp) + _ANTENNA_W * float(ra)
        return float(odor_l - odor_r), float(0.5 * (odor_l + odor_r))

    # ------------------------------------------------------------------
    def _target_block(self, sim) -> tuple[float, float]:
        # Lazy refresh of banana_xy in case it was not ready at construction.
        if self._banana_xy is None:
            try:
                self._banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
            except Exception:
                return 0.0, 0.0
        if self._banana_xy is None:
            return 0.0, 0.0

        try:
            thorax_xy = sim.get_body_positions(sim.fly.name)[self._thorax_idx, :2]
        except Exception:
            thorax_xy = sim.mj_data.xpos[self._thorax_body_id, :2]
        thorax_xy = np.asarray(thorax_xy, dtype=float)

        to_target = self._banana_xy - thorax_xy
        dist = float(np.linalg.norm(to_target))
        if dist < 1e-9:
            return 0.0, 0.0

        xmat = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
        heading_xy = xmat[:2, 0].copy()
        lateral_xy = xmat[:2, 1].copy()
        hn = float(np.linalg.norm(heading_xy))
        ln = float(np.linalg.norm(lateral_xy))
        if hn > 1e-12:
            heading_xy /= hn
        if ln > 1e-12:
            lateral_xy /= ln

        to_target_u = to_target / dist
        forward_err = float(np.dot(heading_xy, to_target_u))
        lateral_err = float(np.dot(lateral_xy, to_target_u))
        bearing = float(np.arctan2(lateral_err, forward_err))
        return bearing, float(np.log1p(dist))

    # ------------------------------------------------------------------
    def _contact_block(self, sim) -> tuple[float, float]:
        contact_max = 0.0
        try:
            cf = sim.mj_data.cfrc_ext[self._contact_body_ids, :3]
            mag = np.linalg.norm(cf[:, :2], axis=1)
            contact_max = float(np.max(mag)) if mag.size > 0 else 0.0
        except Exception:
            contact_max = 0.0

        prev = float(self._contact_ema)
        self._contact_ema = 0.90 * prev + 0.10 * contact_max
        d = float(self._contact_ema - self._contact_prev)
        self._contact_prev = float(self._contact_ema)
        self._d_contact_ema = 0.80 * self._d_contact_ema + 0.20 * d
        return float(self._contact_ema), float(self._d_contact_ema)

    # ------------------------------------------------------------------
    def extract(
        self, sim, prev_turn: float = 0.0, prev_speed: float = 1.0
    ) -> np.ndarray:
        """Build the feature vector for the current sim state.

        Parameters
        ----------
        sim : MiniprojectSimulation
        prev_turn, prev_speed : last policy outputs, used as 1-step memory.

        Returns
        -------
        np.ndarray, shape (N_FEATURES,), dtype float32.
        """
        (
            left_area,
            right_area,
            center_area,
            total_area,
            d_total_area,
            left_x,
            right_x,
        ) = self._vision_block(sim)
        odor_grad, mean_log_odor = self._odor_block(sim)
        bearing, dist_log = self._target_block(sim)
        contact_fmax, d_contact = self._contact_block(sim)

        vec = np.array(
            [
                left_area,
                right_area,
                center_area,
                total_area,
                d_total_area,
                left_x,
                right_x,
                odor_grad,
                mean_log_odor,
                bearing,
                dist_log,
                contact_fmax,
                d_contact,
                float(prev_turn),
                float(prev_speed),
            ],
            dtype=np.float32,
        )
        # Guard against NaNs from any numerical weirdness.
        if not np.isfinite(vec).all():
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec
