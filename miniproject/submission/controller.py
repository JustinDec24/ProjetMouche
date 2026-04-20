import numpy as np
from miniproject.simulation import MiniprojectSimulation


class Controller:
    """Simple controller for levels 0–1.

    Level 0 (flat): go to banana using olfaction only.
    Level 1 (terrain): same goal, but slow down on slopes, steer uphill,
    and increase grip to avoid rollovers.
    """

    # --- scheduling ---
    DECISION_INTERVAL_S = 0.05

    # --- olfaction (Level 0) ---
    PALP_WEIGHT = 9
    ANTENNA_WEIGHT = 1
    # Log-olfaction steering is distance-independent (more stable far away)
    ATTRACTIVE_GAIN = -2.5
    BIAS_SMOOTHING = 0.40
    EPS_ODOR = 1e-12
    STOP_ODOR_THRESHOLD = 2e-5

    # --- drives ---
    BASE_DRIVE_FAST = 1.65
    MAX_DRIVE = 1.90
    MIN_DRIVE = 0.45
    MIN_DRIVE_TERRAIN = 0.35
    MIN_SIDE_DRIVE = 0.20
    MIN_SIDE_DRIVE_TERRAIN = 0.12
    TURN_MOD = 0.8

    # --- terrain (Level 1) ---
    DOWNHILL_BRAKE = 3.5
    STEEP_BRAKE = 2.5
    TURN_STEEP_GAIN = 2.0

    # --- grip (adhesion) ---
    CONTACT_THRESHOLD = 0.15
    GRIP_SLOPE = 0.12
    GRIP_TILT = 0.25

    # --- vision avoidance (for level 2+) ---
    VISION_ENABLE = True
    VISION_DECISION_EVERY = 1  # computed at same cadence as olfaction by default
    # "Danger" tuning: make the fly react earlier/stronger to rough/dark terrain.
    VISION_EDGE_THRESH = 0.04
    VISION_DARK_THRESH = 0.20
    VISION_VERY_DARK_THRESH = 0.10
    VISION_AVOID_GAIN = 10.0
    VISION_AVOID_MAX = 7.0
    VISION_SMOOTHING = 0.55

    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        self._decision_every = int(self.DECISION_INTERVAL_S / sim.timestep)
        self._step_count = 0
        self._drives = np.array([1.0, 1.0])
        self._stopped = False

        fly_segs = sim.fly.get_bodysegs_order()
        self._thorax_idx = next(
            i for i, s in enumerate(fly_segs) if s.name == "c_thorax"
        )
        body_ids = sim._internal_bodyids_by_fly[sim.fly.name]
        self._thorax_body_id = body_ids[self._thorax_idx]
        self._contact_body_ids = sim._internal_contact_body_segment_ids_by_fly[sim.fly.name]
        self._smooth_bias = 0.0
        self._smooth_avoid = 0.0
        self._vision_rows = None
        self._vision_rect_idx = None
        self._vision_rect_mask = None
        self._vision_step_count = 0

        # Precompute row mapping for hex->rect conversion (week4 exercise pattern)
        try:
            retina = sim.fly.retina
            # Build a padded rectangular index table from the ommatidia id map.
            # ommatidia_id_map contains IDs in [0..N], where 0 means "no ommatidium".
            id_rows = [np.unique(row) for row in retina.ommatidia_id_map]
            idx_rows = []
            max_len = 0
            for ids in id_rows:
                ids = ids[ids > 0]
                idx = (ids - 1).astype(int)  # to [0..N-1]
                idx_rows.append(idx)
                max_len = max(max_len, len(idx))
            rect_idx = -np.ones((len(idx_rows), max_len), dtype=int)
            for r, idx in enumerate(idx_rows):
                rect_idx[r, : len(idx)] = idx
            self._vision_rect_idx = rect_idx
            self._vision_rect_mask = rect_idx >= 0
        except Exception:
            self._vision_rect_idx = None
            self._vision_rect_mask = None

    # ------------------------------------------------------------------
    def step(self, sim: MiniprojectSimulation):
        if self._step_count % self._decision_every == 0:
            self._drives = self._compute_drives(sim)
        self._step_count += 1

        joint_angles, adhesion = self.turning_controller.step(self._drives)

        # --- Level 1 grip: increase stance adhesion on slopes / when tilted ---
        _, _, uprightness = self._get_orientation(sim)
        tilt = max(0.0, 1.0 - uprightness)
        slope_mag = 0.0
        if callable(getattr(getattr(sim, "world", None), "get_normal", None)):
            _, _, slope_mag = self._get_slope_signals(sim)

        # Grip: plus de colle quand ça penche ou que la pente est forte.
        if tilt > self.GRIP_TILT or slope_mag > self.GRIP_SLOPE:
            adhesion = np.where(adhesion > 0.0, 1.0, adhesion)

        # Targeted boost for stance legs not touching the ground
        try:
            contact_forces = sim.get_external_force(
                sim.fly.name, subtract_adhesion_force=True
            )
        except Exception:
            contact_forces = sim.mj_data.cfrc_ext[self._contact_body_ids, 3:]

        contact_forces = np.asarray(contact_forces, dtype=float)
        if contact_forces.ndim != 2 or contact_forces.shape[1] != 3:
            contact_mag = np.zeros(6, dtype=float)
        else:
            contact_mag = np.linalg.norm(contact_forces, axis=1)
        n = min(6, len(contact_mag))
        for i in range(n):
            if adhesion[i] > 0 and contact_mag[i] < self.CONTACT_THRESHOLD:
                adhesion[i] = 1.0

        adhesion = np.clip(adhesion, 0.0, 1.0)

        return joint_angles, adhesion

    # ------------------------------------------------------------------
    def _get_orientation(self, sim):
        """(pitch, roll_ind, uprightness) from thorax rotation matrix."""
        xmat = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
        pitch = np.arcsin(np.clip(xmat[2, 0], -1.0, 1.0))
        return pitch, xmat[2, 1], xmat[2, 2]

    def _get_body_frame_xy(self, sim) -> tuple[np.ndarray, np.ndarray]:
        """Return (heading_xy, lateral_xy) unit vectors from thorax rotation."""
        xmat = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
        heading_xy = xmat[:2, 0].copy()
        lateral_xy = xmat[:2, 1].copy()
        hn = np.linalg.norm(heading_xy)
        ln = np.linalg.norm(lateral_xy)
        if hn > 1e-12:
            heading_xy /= hn
        if ln > 1e-12:
            lateral_xy /= ln
        return heading_xy, lateral_xy

    def _get_slope_signals(self, sim) -> tuple[float, float, float]:
        """Return (slope_forward, slope_lateral, slope_mag) from terrain normal.

        These are slopes of z(x, y) projected onto the body forward/lateral axes.
        Positive slope_forward means uphill in the forward direction.
        Positive slope_lateral means uphill towards the body's lateral +Y direction.
        """
        world = getattr(sim, "world", None)
        get_normal = getattr(world, "get_normal", None)
        if not callable(get_normal):
            return 0.0, 0.0, 0.0

        try:
            thorax_xy = sim.get_body_positions(sim.fly.name)[self._thorax_idx, :2]
        except Exception:
            # Fallback to MuJoCo direct access if needed
            thorax_xy = sim.mj_data.xpos[self._thorax_body_id, :2]

        n = np.asarray(get_normal(float(thorax_xy[0]), float(thorax_xy[1])), dtype=float)
        if n.shape != (3,) or not np.isfinite(n).all() or abs(n[2]) < 1e-6:
            return 0.0, 0.0, 0.0

        # From normal to gradient: n ~ (-dz/dx, -dz/dy, 1)
        grad = np.array([-n[0] / n[2], -n[1] / n[2]], dtype=float)
        slope_mag = float(np.linalg.norm(grad))

        heading_xy, lateral_xy = self._get_body_frame_xy(sim)
        slope_forward = float(np.dot(heading_xy, grad))
        slope_lateral = float(np.dot(lateral_xy, grad))
        return slope_forward, slope_lateral, slope_mag

    # ------------------------------------------------------------------
    def _compute_drives(self, sim) -> np.ndarray:
        if self._stopped:
            return np.array([0.0, 0.0])

        # ---- Level 0: olfaction steering ----
        odor_lin = sim.get_olfaction(sim.fly.name)  # shape (4, 1)
        # Sensor order from olfaction.yaml:
        # [l_palp, r_palp, l_antenna, r_antenna]
        lp, rp, la, ra = odor_lin[:, 0]
        odor_l = self.PALP_WEIGHT * float(lp) + self.ANTENNA_WEIGHT * float(la)
        odor_r = self.PALP_WEIGHT * float(rp) + self.ANTENNA_WEIGHT * float(ra)
        mean_odor = 0.5 * (odor_l + odor_r)

        if mean_odor > self.STOP_ODOR_THRESHOLD:
            self._stopped = True
            return np.array([0.0, 0.0])

        odor_log = sim.get_olfaction(sim.fly.name, log=True)
        lp, rp, la, ra = odor_log[:, 0]
        odor_l = self.PALP_WEIGHT * float(lp) + self.ANTENNA_WEIGHT * float(la)
        odor_r = self.PALP_WEIGHT * float(rp) + self.ANTENNA_WEIGHT * float(ra)
        bias = float(self.ATTRACTIVE_GAIN * (odor_l - odor_r))

        # ---- vision-based avoidance bias (simple, week4-style preprocessing) ----
        if self.VISION_ENABLE and self._vision_rect_idx is not None:
            if self._vision_step_count % self.VISION_DECISION_EVERY == 0:
                avoid_raw = float(self._vision_avoid_bias(sim))
                # Smooth to avoid oscillations / zig-zag.
                self._smooth_avoid += (1 - self.VISION_SMOOTHING) * (
                    avoid_raw - self._smooth_avoid
                )

                # Near the banana, odor is strong: prioritize goal over avoidance.
                odor_close = float(np.clip(mean_odor / max(self.EPS_ODOR, self.STOP_ODOR_THRESHOLD), 0.0, 1.0))
                avoid_cap = self.VISION_AVOID_MAX * (1.0 - 0.55 * odor_close)
                bias += float(np.clip(self._smooth_avoid, -avoid_cap, avoid_cap))
            self._vision_step_count += 1

        # ---- Level 1: terrain adjustments (only if normal available) ----
        base_drive = self.BASE_DRIVE_FAST
        turn_mod = self.TURN_MOD

        if callable(getattr(getattr(sim, "world", None), "get_normal", None)):
            slope_forward, _, slope_mag = self._get_slope_signals(sim)
            downhill = max(0.0, -slope_forward)

            # Freinage simple: downhill + pente forte
            speed_scale = 1.0 / (
                1.0
                + self.DOWNHILL_BRAKE * downhill
                + self.STEEP_BRAKE * max(0.0, slope_mag)
            )
            base_drive = base_drive * speed_scale

            # Baisser un peu le virage sur terrain raide
            turn_mod = turn_mod / (1.0 + self.TURN_STEEP_GAIN * max(0.0, slope_mag))

        # ---- EMA smoothing + saturation ----
        self._smooth_bias += (1 - self.BIAS_SMOOTHING) * (bias - self._smooth_bias)
        bias_norm = float(np.tanh(self._smooth_bias))

        min_drive = self.MIN_DRIVE
        min_side = self.MIN_SIDE_DRIVE
        if callable(getattr(getattr(sim, "world", None), "get_normal", None)):
            min_drive = self.MIN_DRIVE_TERRAIN
            min_side = self.MIN_SIDE_DRIVE_TERRAIN

        base_drive = float(np.clip(base_drive, min_drive, self.MAX_DRIVE))

        drives = np.full(2, base_drive, dtype=float)
        side = int(bias_norm > 0)
        drives[side] -= abs(bias_norm) * turn_mod * base_drive
        drives[side] = max(min_side, drives[side])
        drives = np.clip(drives, 0.0, self.MAX_DRIVE)
        return drives

    def _vision_avoid_bias(self, sim: MiniprojectSimulation) -> float:
        """Return a left/right avoidance bias from vision.

        This follows the week4 vision exercise idea: convert ommatidia to a
        rect image and compute simple features. We steer away from the side
        that contains stronger edges/darker patches in the forward view.
        """
        try:
            om = sim.get_ommatidia_readouts(sim.fly.name)  # (2, 721, 2)
        except Exception:
            return 0.0

        # grayscale per eye in [0,1]
        gray = om.max(axis=-1)  # (2, 721)

        # crop hex to rect with padding: (2, nrows, ncols)
        rect_idx = self._vision_rect_idx
        mask = self._vision_rect_mask
        rect = np.zeros((2, rect_idx.shape[0], rect_idx.shape[1]), dtype=float)
        flat_idx = rect_idx[mask]
        # broadcast assignment for both eyes
        rect[:, mask] = gray[:, flat_idx]

        # Focus on "forward-ish" region of the rect patch.
        # The rectification is not a pinhole camera; empirically a central band
        # reacts better than the extreme bottom rows.
        nrows, ncols = rect.shape[1], rect.shape[2]
        r0 = int(nrows * 0.20)
        r1 = int(nrows * 0.75)
        c0 = int(ncols * 0.15)
        c1 = int(ncols * 0.85)
        roi = rect[:, r0:r1, c0:c1]

        # Side-wise danger (holes/pics/rough slopes tend to create edges + dark patches).
        dx = np.abs(np.diff(roi, axis=2))
        dy = np.abs(np.diff(roi, axis=1))
        edge_energy = float(0.5 * (dx.mean() + dy.mean()))

        dark = (roi < self.VISION_DARK_THRESH).astype(float)
        very_dark = (roi < self.VISION_VERY_DARK_THRESH).astype(float)
        dark_frac = float(dark.mean())
        very_dark_frac = float(very_dark.mean())

        danger_global = edge_energy + 0.9 * dark_frac + 1.2 * very_dark_frac
        if danger_global < self.VISION_EDGE_THRESH:
            return 0.0

        mid = roi.shape[2] // 2
        left_roi = roi[:, :, :mid]
        right_roi = roi[:, :, mid:]
        left_edge = float(
            0.5 * np.abs(np.diff(left_roi, axis=2)).mean()
            + 0.5 * np.abs(np.diff(left_roi, axis=1)).mean()
        )
        right_edge = float(
            0.5 * np.abs(np.diff(right_roi, axis=2)).mean()
            + 0.5 * np.abs(np.diff(right_roi, axis=1)).mean()
        )
        left_dark = float((left_roi < self.VISION_DARK_THRESH).mean())
        right_dark = float((right_roi < self.VISION_DARK_THRESH).mean())
        left_vdark = float((left_roi < self.VISION_VERY_DARK_THRESH).mean())
        right_vdark = float((right_roi < self.VISION_VERY_DARK_THRESH).mean())

        left_score = left_edge + 0.9 * left_dark + 1.2 * left_vdark
        right_score = right_edge + 0.9 * right_dark + 1.2 * right_vdark

        # If right is more dangerous, steer left (negative bias), and vice versa.
        delta = float(right_score - left_score)
        steer = -float(np.tanh(4.0 * delta))

        # Hard danger mode: very dark patches usually mean pits/steep drops.
        hard = (very_dark_frac > 0.06) or (max(left_vdark, right_vdark) > 0.08)
        if hard:
            mag = self.VISION_AVOID_MAX
        else:
            mag = min(
                self.VISION_AVOID_MAX,
                self.VISION_AVOID_GAIN * max(0.0, danger_global - self.VISION_EDGE_THRESH),
            )
        return float(steer * mag)
