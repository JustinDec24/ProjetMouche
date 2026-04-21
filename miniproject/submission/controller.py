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
    ATTRACTIVE_GAIN_GRASS = -6.0
    BIAS_SMOOTHING = 0.40
    EPS_ODOR = 1e-12
    # NOTE: In practice, the odor magnitude at ~8–10 units distance can already
    # exceed 2e-5 with the default arena parameters; keep this threshold higher
    # so the fly does not stop prematurely.
    STOP_ODOR_THRESHOLD = 5e-4

    # --- drives ---
    BASE_DRIVE_FAST = 1.65
    MAX_DRIVE = 1.90
    MAX_DRIVE_TERRAIN = 1.20
    MIN_DRIVE = 0.45
    MIN_DRIVE_TERRAIN = 0.35
    MIN_SIDE_DRIVE = 0.20
    MIN_SIDE_DRIVE_TERRAIN = 0.12
    TURN_MOD = 0.8

    # --- terrain (Level 1) ---
    DOWNHILL_BRAKE = 1.6
    STEEP_BRAKE = 1.2
    TURN_STEEP_GAIN = 2.0
    SLOPE_STEER_GAIN = 6.0
    SLOPE_STEER_MAX = 3.0

    # --- grip (adhesion) ---
    CONTACT_THRESHOLD = 0.15
    GRIP_SLOPE = 0.25
    GRIP_TILT = 0.25
    GRIP_UPRIGHTNESS = 0.60

    # --- vision avoidance (for level 2+) ---
    VISION_ENABLE = True
    VISION_DECISION_EVERY = 1  # computed at same cadence as olfaction by default
    # Prefer using raw RGB frames. This avoids the diluted / sparse ommatidia signal.
    VISION_USE_RAW = True
    # "Danger" tuning: make the fly react earlier/stronger to rough/dark terrain.
    VISION_EDGE_THRESH = 0.03
    VISION_DARK_THRESH = 0.20
    VISION_VERY_DARK_THRESH = 0.10
    VISION_AVOID_GAIN = 12.0
    VISION_AVOID_MAX = 8.0
    VISION_SMOOTHING = 0.55
    VISION_FAR_ODOR_SCALE = 0.25
    # Sky-blue heuristic (robust in NeLy visual pipeline): steer towards open sky.
    SKY_MIN_B = 0.55
    SKY_B_OVER_R = 0.10
    SKY_B_OVER_G = 0.05
    SKY_GAIN = 2.5
    OBSTACLE_GAIN = 3.0
    # If avoidance is pushing against the target-bearing turn, damp it. This
    # prevents long "runaways" where avoidance dominates and the fly drifts away.
    AVOID_OPPOSE_TARGET_ENABLE = True
    AVOID_OPPOSE_TARGET_MIN_TARGET_BIAS = 0.15
    AVOID_OPPOSE_TARGET_DAMP = 0.25
    AVOID_OPPOSE_TARGET_DIST = 45.0

    # --- obstacle avoidance mode (Level 2+) ---
    AVOID_MODE_ENABLE = True
    AVOID_MODE_DANGER_ON = 0.090
    AVOID_MODE_DANGER_OFF = 0.060
    AVOID_MODE_MIN_DECISIONS = 10
    AVOID_MODE_MAX_DECISIONS = 40
    AVOID_MODE_GAIN = 2.0
    AVOID_MODE_ODOR_DAMP = 0.65

    # --- anti-runaway (Level 2+) ---
    RUNAWAY_ENABLE = True
    RUNAWAY_CLOSE_DIST = 15.0
    # If we once got close but then drift far away, reset quickly. This improves
    # robustness in Level 2 where avoidance can temporarily pull the fly off the
    # odor trail near obstacles.
    RUNAWAY_FAR_DIST = 30.0
    RUNAWAY_NO_IMPROVE_DECISIONS = 20
    RUNAWAY_FAR_DECISIONS = 6

    # --- go-to-goal latch (Level 2+) ---
    GOAL_MODE_ENABLE = True
    GOAL_MODE_ODOR_ON = 2.5e-6
    GOAL_MODE_ODOR_OFF = 1.2e-6
    GOAL_MODE_MIN_DECISIONS = 12
    GOAL_MODE_MAX_DECISIONS = 60
    GOAL_MODE_AVOID_SCALE = 0.05
    GOAL_MODE_SEARCH_DISABLE = True
    GOAL_MODE_TURN_SCALE = 0.65
    GOAL_MODE_MIN_DRIVE = 1.00

    # --- final approach (Level 2+) ---
    FINAL_APPROACH_ENABLE = True
    FINAL_APPROACH_DIST = 18.0
    # Keep a small amount of avoidance so we can go around obstacles near goal.
    FINAL_APPROACH_AVOID_SCALE = 0.20
    FINAL_APPROACH_BLEND_TARGET = 0.95
    FINAL_APPROACH_MIN_DRIVE = 1.05
    FINAL_CAST_ENABLE = True
    FINAL_CAST_DIST = 9.0
    FINAL_CAST_NO_IMPROVE_DECISIONS = 10
    FINAL_CAST_DURATION_DECISIONS = 18
    FINAL_CAST_GAIN = 0.9
    # Extra-stable close range behavior (prevents overshoot/runaway).
    CLOSE_STABLE_ENABLE = True
    CLOSE_STABLE_DIST = 16.0
    CLOSE_STABLE_MAX_BIAS = 0.7
    CLOSE_STABLE_TURN_SCALE = 0.45
    CLOSE_STABLE_MAX_DRIVE = 0.95
    CLOSE_STABLE_AVOID_SCALE = 0.10

    # If we are near-ish but keep bouncing away, request a reset.
    NEAR_RESET_ENABLE = True
    NEAR_RESET_DIST = 16.0
    NEAR_RESET_MIN_DELTA = 10.0
    NEAR_RESET_DECISIONS = 1

    # --- mid-range recovery (Level 2+) ---
    MID_APPROACH_ENABLE = True
    MID_APPROACH_DIST = 26.0
    MID_CAST_NO_IMPROVE_DECISIONS = 12
    MID_CAST_DURATION_DECISIONS = 26
    MID_CAST_GAIN = 1.45

    # --- bearing-to-target steering (uses banana position) ---
    TARGET_STEER_ENABLE = True
    TARGET_STEER_GAIN = 3.5
    TARGET_STEER_GAIN_CLOSE = 16.0
    TARGET_STEER_CLOSE_DIST = 25.0
    # In Level 2 the environment can break the odor trail; rely more on target bearing.
    TARGET_STEER_BLEND_WHEN_GRASS = 0.65
    TARGET_STEER_BLEND_GOALMODE = 0.90

    # --- homing recovery (Level 2+) ---
    HOMING_ENABLE = True
    HOMING_DRIFT_DIST = 3.0
    HOMING_TRIGGER_DECISIONS = 6
    HOMING_DURATION_DECISIONS = 35
    HOMING_AVOID_SCALE = 0.10

    # --- reorientation if no early progress (Level 2+) ---
    REORIENT_ENABLE = True
    REORIENT_NO_PROGRESS_DECISIONS = 18
    REORIENT_MIN_IMPROVE = 1.0
    REORIENT_DURATION_DECISIONS = 22
    REORIENT_TURN_GAIN = 2.2
    REORIENT_PIVOT_BASE = 0.45

    # --- exploration when odor is weak (Level 2+) ---
    EXPLORE_ENABLE = True
    EXPLORE_ODOR_WEAK = 8e-7
    EXPLORE_CAST_GAIN = 2.4
    EXPLORE_DRIVE = 1.05
    EXPLORE_TOGGLE_DECISIONS = 16
    EXPLORE_SLOPE_MAX = 0.65
    EXPLORE_UPRIGHT_MIN = 0.60

    # --- search / recovery (Level 2+) ---
    SEARCH_ENABLE = True
    SEARCH_TRIGGER_DIST = 12.0
    SEARCH_NO_IMPROVE_DECISIONS = 30
    SEARCH_IMPROVE_EPS = 0.10
    SEARCH_DURATION_DECISIONS = 25
    SEARCH_CAST_GAIN = 1.6
    SEARCH_DRIVE = 0.95
    SEARCH_SLOPE_MAX = 0.45
    SEARCH_UPRIGHT_MIN = 0.65

    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        self._decision_every = int(self.DECISION_INTERVAL_S / sim.timestep)
        self._step_count = 0
        self._drives = np.array([1.0, 1.0])
        self._stopped = False
        self._enable_terrain = bool(getattr(sim, "enable_terrain", False))
        self._enable_grass = bool(getattr(sim, "enable_grass", False))

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
        self._last_xy = None
        self._stuck_decisions = 0
        self._escape_decisions_left = 0
        self._escape_dir = 1
        self._flip_decisions = 0
        self._avoid_mode_left = 0
        self._smooth_danger = 0.0
        self._banana_xy = None
        self._best_dist = None
        self._no_improve = 0
        self._search_left = 0
        self._search_dir = 1
        self._start_dist = None
        self._request_reset = False
        self._runaway_far = 0
        self._near_worsen = 0
        self._goal_mode_left = 0
        self._explore_phase = 1
        self._explore_left = 0
        self._homing_left = 0
        self._drift_decisions = 0
        self._final_cast_left = 0
        self._final_cast_dir = 1
        self._final_best = None
        self._final_no_improve = 0
        self._start_dist = self._start_dist
        self._reorient_left = 0
        self._reorient_no_progress = 0
        self._mid_cast_left = 0
        self._mid_cast_dir = 1
        self._mid_best = None
        self._mid_no_improve = 0

        try:
            self._banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
        except Exception:
            self._banana_xy = None
        if self._banana_xy is not None:
            try:
                thorax_xy0 = sim.get_body_positions(sim.fly.name)[self._thorax_idx, :2]
            except Exception:
                thorax_xy0 = sim.mj_data.xpos[self._thorax_body_id, :2]
            self._start_dist = float(np.linalg.norm(np.asarray(thorax_xy0, dtype=float) - self._banana_xy))

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
        is_decision_step = self._step_count % self._decision_every == 0
        if is_decision_step:
            self._drives = self._compute_drives(sim)
        self._step_count += 1

        joint_angles, adhesion = self.turning_controller.step(self._drives)

        # --- Grip control (terrain only): extra adhesion on slopes / when tilted ---
        # Keep grip logic conservative: excessive adhesion can freeze locomotion.
        if self._enable_terrain and is_decision_step:
            if self._request_reset:
                try:
                    sim.reset()
                except Exception:
                    pass
                self._request_reset = False
                self._flip_decisions = 0
                self._stopped = False
                self._smooth_bias = 0.0
                self._smooth_avoid = 0.0
                self._smooth_danger = 0.0
                self._avoid_mode_left = 0
                self._escape_decisions_left = 0
                self._stuck_decisions = 0
                self._last_xy = None
                self._best_dist = None
                self._no_improve = 0
                self._search_left = 0
                return joint_angles, adhesion

            # During escape, reduce adhesion to allow the fly to pivot and unstick.
            if self._escape_decisions_left > 0:
                adhesion = np.zeros_like(adhesion)
                return joint_angles, adhesion

            _, _, uprightness = self._get_orientation(sim)
            # Flip safety: reset aggressively if the body gets too inverted.
            if uprightness < -0.4:
                self._flip_decisions = 999
            elif uprightness < 0.2:
                self._flip_decisions += 1
            else:
                self._flip_decisions = 0

            # Hard recovery: if the fly stays flipped for too long, reset.
            if self._flip_decisions >= 18:
                try:
                    sim.reset()
                except Exception:
                    pass
                self._flip_decisions = 0
                self._stopped = False
                self._smooth_bias = 0.0
                self._smooth_avoid = 0.0
                self._escape_decisions_left = 0
                self._stuck_decisions = 0
                self._last_xy = None
                return joint_angles, adhesion

            tilt = max(0.0, 1.0 - uprightness)
            _, _, slope_mag = self._get_slope_signals(sim)

            if (
                uprightness < self.GRIP_UPRIGHTNESS
                or tilt > self.GRIP_TILT
                or slope_mag > self.GRIP_SLOPE
            ):
                # Only stick legs that are actually in contact. This prevents
                # "gluing" swing legs in the air and freezing locomotion.
                try:
                    contact_forces = sim.mj_data.cfrc_ext[self._contact_body_ids, 3:]
                    contact_mag = np.linalg.norm(contact_forces, axis=1)
                    stance = contact_mag > self.CONTACT_THRESHOLD
                    adhesion = np.zeros_like(adhesion)
                    n = min(len(adhesion), len(stance))
                    adhesion[:n] = stance[:n].astype(float)
                except Exception:
                    adhesion = np.where(adhesion > 0.0, 1.0, adhesion)

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

        # Distance bookkeeping for Level 2 robustness (runaway detection).
        dist_to_banana = None
        if self._banana_xy is not None:
            try:
                thorax_xy = sim.get_body_positions(sim.fly.name)[self._thorax_idx, :2]
            except Exception:
                thorax_xy = sim.mj_data.xpos[self._thorax_body_id, :2]
            thorax_xy = np.asarray(thorax_xy, dtype=float)
            dist_to_banana = float(np.linalg.norm(thorax_xy - self._banana_xy))
            if self._best_dist is None:
                self._best_dist = dist_to_banana
            else:
                self._best_dist = min(self._best_dist, dist_to_banana)

            if (
                self.NEAR_RESET_ENABLE
                and self._enable_grass
                and self._best_dist <= self.NEAR_RESET_DIST
                and dist_to_banana >= (self._best_dist + self.NEAR_RESET_MIN_DELTA)
            ):
                self._near_worsen += 1
                if self._near_worsen >= int(self.NEAR_RESET_DECISIONS):
                    self._request_reset = True
                    self._near_worsen = 0
            else:
                self._near_worsen = 0

            # Emergency reset: if we got fairly close once but are now far, reset immediately.
            if (
                self.NEAR_RESET_ENABLE
                and self._enable_grass
                and self._best_dist <= self.NEAR_RESET_DIST
                and dist_to_banana >= (self.RUNAWAY_FAR_DIST + 5.0)
            ):
                self._request_reset = True

            # If we haven't improved at all for a while, do a short reorientation
            # towards the banana (helps escaping long wrong-way trajectories).
            if self.REORIENT_ENABLE and self._enable_grass and self._start_dist is not None:
                if self._best_dist <= (self._start_dist - self.REORIENT_MIN_IMPROVE):
                    self._reorient_no_progress = 0
                else:
                    self._reorient_no_progress += 1
                if (
                    self._reorient_left <= 0
                    and self._reorient_no_progress >= int(self.REORIENT_NO_PROGRESS_DECISIONS)
                ):
                    self._reorient_left = int(self.REORIENT_DURATION_DECISIONS)
                    self._reorient_no_progress = 0
                    self._avoid_mode_left = 0

            # If we are persistently drifting away after having been closer,
            # force a short "homing" phase towards the banana.
            if (
                self.HOMING_ENABLE
                and self._enable_grass
                and self._best_dist is not None
                and dist_to_banana > (self._best_dist + self.HOMING_DRIFT_DIST)
            ):
                self._drift_decisions += 1
            else:
                self._drift_decisions = 0

            if (
                self.HOMING_ENABLE
                and self._enable_grass
                and self._homing_left <= 0
                and self._drift_decisions >= int(self.HOMING_TRIGGER_DECISIONS)
            ):
                self._homing_left = int(self.HOMING_DURATION_DECISIONS)
                self._avoid_mode_left = 0

        # ---- Level 2 search mode: if no progress towards banana, start casting ----
        if self.SEARCH_ENABLE and self._enable_grass and self._banana_xy is not None:
            try:
                thorax_xy = sim.get_body_positions(sim.fly.name)[self._thorax_idx, :2]
            except Exception:
                thorax_xy = sim.mj_data.xpos[self._thorax_body_id, :2]
            thorax_xy = np.asarray(thorax_xy, dtype=float)
            dist = float(np.linalg.norm(thorax_xy - self._banana_xy))

            if self._best_dist is None:
                self._best_dist = dist

            improved = dist < (self._best_dist - self.SEARCH_IMPROVE_EPS)
            if improved:
                self._best_dist = dist
                self._no_improve = 0
            else:
                self._no_improve += 1

            # Disable/soften search on steep slopes or when unstable.
            slope_mag = 0.0
            uprightness = 1.0
            try:
                _, _, uprightness = self._get_orientation(sim)
            except Exception:
                uprightness = 1.0
            if self._enable_terrain:
                try:
                    _, _, slope_mag = self._get_slope_signals(sim)
                except Exception:
                    slope_mag = 0.0

            safe_for_search = (slope_mag <= self.SEARCH_SLOPE_MAX) and (uprightness >= self.SEARCH_UPRIGHT_MIN)

            if self._search_left > 0 and safe_for_search:
                self._search_left -= 1
                cast = float(self._search_dir) * self.SEARCH_CAST_GAIN
                # Keep moving forward while sweeping left/right.
                return np.array(
                    [
                        np.clip(self.SEARCH_DRIVE + max(0.0, cast), 0.0, self.MAX_DRIVE_TERRAIN),
                        np.clip(self.SEARCH_DRIVE + max(0.0, -cast), 0.0, self.MAX_DRIVE_TERRAIN),
                    ],
                    dtype=float,
                )
            if not safe_for_search:
                self._search_left = 0

            if (
                dist > self.SEARCH_TRIGGER_DIST
                and self._no_improve >= self.SEARCH_NO_IMPROVE_DECISIONS
                and safe_for_search
            ):
                self._no_improve = 0
                self._search_left = self.SEARCH_DURATION_DECISIONS
                self._search_dir *= -1
                cast = float(self._search_dir) * self.SEARCH_CAST_GAIN
                return np.array(
                    [
                        np.clip(self.SEARCH_DRIVE + max(0.0, cast), 0.0, self.MAX_DRIVE_TERRAIN),
                        np.clip(self.SEARCH_DRIVE + max(0.0, -cast), 0.0, self.MAX_DRIVE_TERRAIN),
                    ],
                    dtype=float,
                )

        # Anti-runaway: if we once got close but are now far, reset quickly.
        if (
            self.RUNAWAY_ENABLE
            and self._enable_grass
            and dist_to_banana is not None
            and self._best_dist is not None
            and self._best_dist <= self.RUNAWAY_CLOSE_DIST
            and dist_to_banana >= self.RUNAWAY_FAR_DIST
        ):
            self._runaway_far += 1
            if (
                self._runaway_far >= self.RUNAWAY_FAR_DECISIONS
                or self._no_improve >= self.RUNAWAY_NO_IMPROVE_DECISIONS
            ):
                self._request_reset = True
                self._runaway_far = 0
        else:
            self._runaway_far = 0

        # ---- stuck detection + escape maneuver (terrain can trap the fly) ----
        try:
            thorax_xy = sim.get_body_positions(sim.fly.name)[self._thorax_idx, :2]
        except Exception:
            thorax_xy = sim.mj_data.xpos[self._thorax_body_id, :2]
        thorax_xy = np.asarray(thorax_xy, dtype=float)

        if self._last_xy is not None:
            moved = float(np.linalg.norm(thorax_xy - self._last_xy))
            # MuJoCo contact jitter can create tiny displacements even when
            # effectively stuck; use a slightly looser threshold.
            if moved < 5e-3:
                self._stuck_decisions += 1
            else:
                self._stuck_decisions = 0
        self._last_xy = thorax_xy

        if self._escape_decisions_left > 0:
            self._escape_decisions_left -= 1
            # Strong pivot in place (alternate direction between escapes).
            if self._escape_dir > 0:
                return np.array([self.MAX_DRIVE, self.MIN_SIDE_DRIVE_TERRAIN])
            return np.array([self.MIN_SIDE_DRIVE_TERRAIN, self.MAX_DRIVE])

        if self._enable_terrain and self._stuck_decisions >= 25:
            self._stuck_decisions = 0
            self._escape_decisions_left = 10
            self._escape_dir *= -1
            if self._escape_dir > 0:
                return np.array([self.MAX_DRIVE, self.MIN_SIDE_DRIVE_TERRAIN])
            return np.array([self.MIN_SIDE_DRIVE_TERRAIN, self.MAX_DRIVE])

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

        # ---- exploration: when odor is very weak, do wide smooth casting ----
        if (
            self.EXPLORE_ENABLE
            and self._enable_grass
            and self._banana_xy is None
            and mean_odor < self.EXPLORE_ODOR_WEAK
            and self._goal_mode_left <= 0
        ):
            slope_mag = 0.0
            uprightness = 1.0
            if self._enable_terrain:
                try:
                    _, _, uprightness = self._get_orientation(sim)
                    _, _, slope_mag = self._get_slope_signals(sim)
                except Exception:
                    slope_mag = 0.0
                    uprightness = 1.0
            safe_explore = (slope_mag <= self.EXPLORE_SLOPE_MAX) and (uprightness >= self.EXPLORE_UPRIGHT_MIN)

            if self._explore_left <= 0:
                self._explore_left = int(self.EXPLORE_TOGGLE_DECISIONS)
                self._explore_phase *= -1
            else:
                self._explore_left -= 1
            cast = float(self._explore_phase) * (self.EXPLORE_CAST_GAIN if safe_explore else 0.0)
            max_drive = self.MAX_DRIVE_TERRAIN if self._enable_terrain else self.MAX_DRIVE
            left = float(np.clip(self.EXPLORE_DRIVE + max(0.0, cast), 0.0, max_drive))
            right = float(np.clip(self.EXPLORE_DRIVE + max(0.0, -cast), 0.0, max_drive))
            return np.array([left, right], dtype=float)

        # ---- target-bearing steering (helps re-acquire after avoidance) ----
        target_bias = 0.0
        if self.TARGET_STEER_ENABLE and self._banana_xy is not None:
            to_target = np.asarray(self._banana_xy, dtype=float) - thorax_xy
            dist_tt = float(np.linalg.norm(to_target))
            if dist_tt > 1e-9:
                to_target /= dist_tt
                heading_xy, lateral_xy = self._get_body_frame_xy(sim)
                lateral_err = float(np.dot(lateral_xy, to_target))
                forward_err = float(np.dot(heading_xy, to_target))
                g = (
                    self.TARGET_STEER_GAIN_CLOSE
                    if dist_tt < self.TARGET_STEER_CLOSE_DIST
                    else self.TARGET_STEER_GAIN
                )
                # Use signed bearing angle so we also turn when the target is behind.
                bearing = float(np.arctan2(lateral_err, forward_err))
                target_bias = -g * bearing

        # Mid-range casting: when we can smell it a bit / know target direction but
        # keep failing to get closer, do a short sweep to escape local traps.
        if (
            self.MID_APPROACH_ENABLE
            and self._enable_grass
            and dist_to_banana is not None
            and dist_to_banana < self.MID_APPROACH_DIST
            and (dist_to_banana >= self.FINAL_CAST_DIST)
            and self._goal_mode_left <= 0
        ):
            if self._mid_best is None:
                self._mid_best = dist_to_banana
                self._mid_no_improve = 0
            else:
                if dist_to_banana < (self._mid_best - 0.25):
                    self._mid_best = dist_to_banana
                    self._mid_no_improve = 0
                else:
                    self._mid_no_improve += 1

            if self._mid_cast_left > 0:
                self._mid_cast_left -= 1
                cast = float(self._mid_cast_dir) * self.MID_CAST_GAIN
                max_drive = self.MAX_DRIVE_TERRAIN if self._enable_terrain else self.MAX_DRIVE
                base = float(np.clip(max(1.05, self.SEARCH_DRIVE), 0.0, max_drive))
                return np.array(
                    [
                        np.clip(base + max(0.0, cast), 0.0, max_drive),
                        np.clip(base + max(0.0, -cast), 0.0, max_drive),
                    ],
                    dtype=float,
                )

            if self._mid_no_improve >= int(self.MID_CAST_NO_IMPROVE_DECISIONS):
                self._mid_no_improve = 0
                self._mid_cast_left = int(self.MID_CAST_DURATION_DECISIONS)
                self._mid_cast_dir *= -1
        else:
            self._mid_best = None
            self._mid_no_improve = 0
            self._mid_cast_left = 0

        # Reorientation: pivot towards the banana before moving on.
        if self._reorient_left > 0 and self._banana_xy is not None:
            self._reorient_left -= 1
            max_drive = self.MAX_DRIVE_TERRAIN if self._enable_terrain else self.MAX_DRIVE
            # Use target_bias sign to decide which way to pivot.
            turn = float(np.clip(self.REORIENT_TURN_GAIN * target_bias, -3.5, 3.5))
            base = float(np.clip(self.REORIENT_PIVOT_BASE, 0.0, max_drive))
            left = float(np.clip(base + max(0.0, turn), 0.0, max_drive))
            right = float(np.clip(base + max(0.0, -turn), 0.0, max_drive))
            return np.array([left, right], dtype=float)

        # Final casting: if we're close but not improving, do short sweeps to
        # slip around local obstacles while still moving forward.
        if (
            self.FINAL_APPROACH_ENABLE
            and self.FINAL_CAST_ENABLE
            and self._enable_grass
            and dist_to_banana is not None
            and dist_to_banana < self.FINAL_CAST_DIST
        ):
            if self._final_best is None:
                self._final_best = dist_to_banana
                self._final_no_improve = 0
            else:
                if dist_to_banana < (self._final_best - 0.15):
                    self._final_best = dist_to_banana
                    self._final_no_improve = 0
                else:
                    self._final_no_improve += 1

            if self._final_cast_left > 0:
                self._final_cast_left -= 1
                cast = float(self._final_cast_dir) * self.FINAL_CAST_GAIN
                max_drive = self.MAX_DRIVE_TERRAIN if self._enable_terrain else self.MAX_DRIVE
                base = float(np.clip(self.FINAL_APPROACH_MIN_DRIVE, 0.0, max_drive))
                return np.array(
                    [
                        np.clip(base + max(0.0, cast), 0.0, max_drive),
                        np.clip(base + max(0.0, -cast), 0.0, max_drive),
                    ],
                    dtype=float,
                )

            if self._final_no_improve >= int(self.FINAL_CAST_NO_IMPROVE_DECISIONS):
                self._final_no_improve = 0
                self._final_cast_left = int(self.FINAL_CAST_DURATION_DECISIONS)
                self._final_cast_dir *= -1
        else:
            self._final_best = None
            self._final_no_improve = 0
            self._final_cast_left = 0

        # Final approach: once we're close, prioritize going to the banana and
        # avoid letting avoidance modes pull us away for long.
        if (
            self.FINAL_APPROACH_ENABLE
            and self._enable_grass
            and dist_to_banana is not None
            and dist_to_banana < self.FINAL_APPROACH_DIST
        ):
            self._avoid_mode_left = 0
            self._search_left = 0
            self._homing_left = 0

        # Homing mode: override steering towards the target for a few decisions.
        # This is a safety net against long avoidance-induced runaways.
        if self._homing_left > 0 and self._banana_xy is not None:
            self._homing_left -= 1
            bias = float(target_bias)
            mean_odor = float(mean_odor)  # keep downstream computations stable
        else:
            bias = None

        # ---- go-to-goal latch (when odor is strong, avoid "dropping the trail") ----
        if self.GOAL_MODE_ENABLE and self._enable_grass:
            if self._goal_mode_left > 0:
                self._goal_mode_left -= 1
            elif mean_odor >= self.GOAL_MODE_ODOR_ON:
                self._goal_mode_left = int(self.GOAL_MODE_MIN_DECISIONS)
            if self._goal_mode_left > 0 and mean_odor >= self.GOAL_MODE_ODOR_OFF:
                self._goal_mode_left = min(
                    int(self.GOAL_MODE_MAX_DECISIONS),
                    self._goal_mode_left + 1,
                )
            if self._goal_mode_left > 0 and self.GOAL_MODE_SEARCH_DISABLE:
                self._search_left = 0

        odor_log = sim.get_olfaction(sim.fly.name, log=True)
        lp, rp, la, ra = odor_log[:, 0]
        odor_l = self.PALP_WEIGHT * float(lp) + self.ANTENNA_WEIGHT * float(la)
        odor_r = self.PALP_WEIGHT * float(rp) + self.ANTENNA_WEIGHT * float(ra)
        gain = self.ATTRACTIVE_GAIN_GRASS if self._enable_grass else self.ATTRACTIVE_GAIN
        odor_bias = float(gain * (odor_l - odor_r))
        if self._enable_grass:
            blend = self.TARGET_STEER_BLEND_WHEN_GRASS
            if self._goal_mode_left > 0:
                blend = max(blend, self.TARGET_STEER_BLEND_GOALMODE)
            if (
                self.FINAL_APPROACH_ENABLE
                and dist_to_banana is not None
                and dist_to_banana < self.FINAL_APPROACH_DIST
            ):
                blend = max(blend, self.FINAL_APPROACH_BLEND_TARGET)
            if (
                self.CLOSE_STABLE_ENABLE
                and dist_to_banana is not None
                and dist_to_banana < self.CLOSE_STABLE_DIST
            ):
                blend = 1.0
            if bias is None:
                bias = float((1.0 - blend) * odor_bias + blend * target_bias)
        else:
            if bias is None:
                bias = odor_bias

        # ---- vision-based avoidance bias + avoidance mode (Level 2+) ----
        if self.VISION_ENABLE and self._enable_grass:
            if self._vision_step_count % self.VISION_DECISION_EVERY == 0:
                avoid_raw, danger = self._vision_avoid_bias_and_danger(sim)

                # Smooth to avoid oscillations / zig-zag.
                self._smooth_avoid += (1 - self.VISION_SMOOTHING) * (
                    avoid_raw - self._smooth_avoid
                )
                self._smooth_danger += (1 - self.VISION_SMOOTHING) * (
                    danger - self._smooth_danger
                )

                # Near the banana, odor is strong: prioritize goal over avoidance.
                odor_close = float(
                    np.clip(
                        mean_odor / max(self.EPS_ODOR, self.STOP_ODOR_THRESHOLD),
                        0.0,
                        1.0,
                    )
                )
                avoid_cap = self.VISION_AVOID_MAX * (1.0 - 0.55 * odor_close)
                avoid_scale = self.VISION_FAR_ODOR_SCALE + (1.0 - self.VISION_FAR_ODOR_SCALE) * odor_close
                # If odor is already strong, avoid should not yank us away.
                if odor_close > 0.35:
                    avoid_scale *= 0.35
                avoid_term = float(avoid_scale * np.clip(self._smooth_avoid, -avoid_cap, avoid_cap))

                # Latch an "avoid mode" for a few decisions when forward danger is high.
                if self.AVOID_MODE_ENABLE:
                    if self._avoid_mode_left > 0:
                        self._avoid_mode_left -= 1
                    elif self._smooth_danger >= self.AVOID_MODE_DANGER_ON:
                        self._avoid_mode_left = int(self.AVOID_MODE_MIN_DECISIONS)

                    if (
                        self._avoid_mode_left > 0
                        and self._smooth_danger >= self.AVOID_MODE_DANGER_OFF
                    ):
                        self._avoid_mode_left = min(
                            int(self.AVOID_MODE_MAX_DECISIONS),
                            self._avoid_mode_left + 1,
                        )

                if self._avoid_mode_left > 0:
                    avoid_term *= self.AVOID_MODE_GAIN
                    bias *= self.AVOID_MODE_ODOR_DAMP

                if self._goal_mode_left > 0:
                    avoid_term *= self.GOAL_MODE_AVOID_SCALE

                # Close to the banana we want to commit: avoidance can otherwise
                # kick us away and we may never re-acquire the trail.
                if dist_to_banana is not None and dist_to_banana < 20.0:
                    avoid_term *= 0.25
                    self._avoid_mode_left = 0

                if self._homing_left > 0:
                    avoid_term *= self.HOMING_AVOID_SCALE
                    self._avoid_mode_left = 0

                if (
                    self.FINAL_APPROACH_ENABLE
                    and dist_to_banana is not None
                    and dist_to_banana < self.FINAL_APPROACH_DIST
                ):
                    avoid_term *= self.FINAL_APPROACH_AVOID_SCALE
                    self._avoid_mode_left = 0

                if (
                    self.CLOSE_STABLE_ENABLE
                    and dist_to_banana is not None
                    and dist_to_banana < self.CLOSE_STABLE_DIST
                ):
                    avoid_term *= self.CLOSE_STABLE_AVOID_SCALE
                    self._avoid_mode_left = 0

                # If avoidance tries to turn opposite to the target-bearing steer,
                # damp it strongly. This keeps the overall behavior goal-directed.
                if (
                    self.AVOID_OPPOSE_TARGET_ENABLE
                    and dist_to_banana is not None
                    and dist_to_banana < self.AVOID_OPPOSE_TARGET_DIST
                    and abs(target_bias) >= self.AVOID_OPPOSE_TARGET_MIN_TARGET_BIAS
                ):
                    # Compare signs: bias is the current (odor/target blended) steer.
                    # Avoidance should not flip the sign when we have a clear target cue.
                    if np.sign(bias) != 0 and np.sign(avoid_term) != 0:
                        if np.sign(bias + avoid_term) != np.sign(bias):
                            avoid_term *= self.AVOID_OPPOSE_TARGET_DAMP
                            self._avoid_mode_left = 0

                bias += avoid_term

            self._vision_step_count += 1

        # ---- Level 1: terrain adjustments (only if normal available) ----
        base_drive = self.BASE_DRIVE_FAST
        turn_mod = self.TURN_MOD

        if self._enable_terrain:
            slope_forward, slope_lateral, slope_mag = self._get_slope_signals(sim)
            downhill = max(0.0, -slope_forward)

            # Freinage simple: downhill + pente forte
            steep_weight = 0.25 + 0.75 * downhill  # keep power when climbing
            speed_scale = 1.0 / (
                1.0
                + self.DOWNHILL_BRAKE * downhill
                + self.STEEP_BRAKE * steep_weight * max(0.0, slope_mag)
            )
            base_drive = base_drive * speed_scale

            # Steering correction: when going downhill, turn towards uphill side
            # to avoid sliding off ridges and to keep climbing back to the target.
            slope_bias = -self.SLOPE_STEER_GAIN * float(slope_lateral) * float(downhill)
            bias += float(np.clip(slope_bias, -self.SLOPE_STEER_MAX, self.SLOPE_STEER_MAX))

            # Baisser un peu le virage sur terrain raide
            turn_mod = turn_mod / (1.0 + self.TURN_STEEP_GAIN * max(0.0, slope_mag))

        # ---- EMA smoothing + saturation ----
        self._smooth_bias += (1 - self.BIAS_SMOOTHING) * (bias - self._smooth_bias)
        bias_norm = float(np.tanh(self._smooth_bias))
        if (
            self.CLOSE_STABLE_ENABLE
            and dist_to_banana is not None
            and dist_to_banana < self.CLOSE_STABLE_DIST
        ):
            bias_norm = float(np.clip(bias_norm, -self.CLOSE_STABLE_MAX_BIAS, self.CLOSE_STABLE_MAX_BIAS))

        min_drive = self.MIN_DRIVE
        min_side = self.MIN_SIDE_DRIVE
        if self._enable_terrain:
            min_drive = self.MIN_DRIVE_TERRAIN
            min_side = self.MIN_SIDE_DRIVE_TERRAIN

        max_drive = self.MAX_DRIVE_TERRAIN if self._enable_terrain else self.MAX_DRIVE
        base_drive = float(np.clip(base_drive, min_drive, max_drive))

        if self._goal_mode_left > 0:
            base_drive = float(np.clip(max(base_drive, self.GOAL_MODE_MIN_DRIVE), min_drive, max_drive))
            turn_mod = float(turn_mod * self.GOAL_MODE_TURN_SCALE)

        if (
            self.FINAL_APPROACH_ENABLE
            and dist_to_banana is not None
            and dist_to_banana < self.FINAL_APPROACH_DIST
        ):
            base_drive = float(np.clip(max(base_drive, self.FINAL_APPROACH_MIN_DRIVE), min_drive, max_drive))

        if (
            self.CLOSE_STABLE_ENABLE
            and dist_to_banana is not None
            and dist_to_banana < self.CLOSE_STABLE_DIST
        ):
            base_drive = float(np.clip(base_drive, min_drive, min(self.CLOSE_STABLE_MAX_DRIVE, max_drive)))
            turn_mod = float(turn_mod * self.CLOSE_STABLE_TURN_SCALE)

        drives = np.full(2, base_drive, dtype=float)
        side = int(bias_norm > 0)
        drives[side] -= abs(bias_norm) * turn_mod * base_drive
        drives[side] = max(min_side, drives[side])
        drives = np.clip(drives, 0.0, max_drive)
        return drives

    def _vision_avoid_bias_and_danger(
        self, sim: MiniprojectSimulation
    ) -> tuple[float, float]:
        """Return a left/right avoidance bias from vision.

        Prefer using the raw RGB fish-eye frames (NeLy pipeline) and a robust
        heuristic: steer towards open sky-blue regions and away from dark/edgy
        patches that typically correlate with grass/obstacles.
        """
        frames = None
        if self.VISION_USE_RAW:
            try:
                frames = sim.get_raw_vision(sim.fly.name)
            except Exception:
                frames = None

        if frames is None:
            # Fallback: ommatidia-based features (less reliable / sparser).
            try:
                om = sim.get_ommatidia_readouts(sim.fly.name)  # (2, 721, 2)
            except Exception:
                return 0.0, 0.0
            gray = om.max(axis=-1)  # (2, 721)
            rect_idx = self._vision_rect_idx
            mask = self._vision_rect_mask
            if rect_idx is None or mask is None:
                return 0.0, 0.0
            rect = np.zeros((2, rect_idx.shape[0], rect_idx.shape[1]), dtype=float)
            flat_idx = rect_idx[mask]
            rect[:, mask] = gray[:, flat_idx]
            imgs = rect
        else:
            # frames: list of 2 arrays (H,W,3), dtype uint8 or float; normalize to [0,1]
            imgs = []
            for f in frames[:2]:
                a = np.asarray(f)
                if a.dtype != np.float32 and a.dtype != np.float64:
                    a = a.astype(np.float32) / 255.0
                else:
                    a = np.clip(a.astype(np.float32), 0.0, 1.0)
                imgs.append(a)

        def _roi_features(img: np.ndarray) -> tuple[float, float, float]:
            # img: (H,W,3) in [0,1] or (H,W) grayscale in [0,1]
            if img.ndim == 2:
                gray2 = img
                sky_frac = 0.0
            else:
                # focus on forward-ish center band
                h, w = img.shape[0], img.shape[1]
                r0, r1 = int(h * 0.20), int(h * 0.80)
                c0, c1 = int(w * 0.15), int(w * 0.85)
                roi = img[r0:r1, c0:c1, :]
                r = roi[..., 0]
                g = roi[..., 1]
                b = roi[..., 2]
                # sky-blue mask
                sky = (b >= self.SKY_MIN_B) & ((b - r) >= self.SKY_B_OVER_R) & (
                    (b - g) >= self.SKY_B_OVER_G
                )
                sky_frac = float(np.mean(sky))
                gray2 = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)

            # obstacle proxy: darkness + edge density
            dark = float(np.mean(gray2 < self.VISION_DARK_THRESH))
            vdark = float(np.mean(gray2 < self.VISION_VERY_DARK_THRESH))
            dx = np.abs(np.diff(gray2, axis=1))
            dy = np.abs(np.diff(gray2, axis=0))
            edge = float(0.5 * (dx.mean() + dy.mean()))
            obstacle = edge + 0.9 * dark + 1.2 * vdark
            return sky_frac, obstacle, edge

        # Interpret eye images as left/right hemispheres.
        if isinstance(imgs, list):
            left_img = imgs[0]
            right_img = imgs[1] if len(imgs) > 1 else imgs[0]
            sky_l, obs_l, edge_l = _roi_features(left_img)
            sky_r, obs_r, edge_r = _roi_features(right_img)
        else:
            # rectified grayscale: split columns
            rect = np.asarray(imgs, dtype=float)
            # rect shape (2,H,W) if from ommatidia; take both eyes same way
            left_img = rect[0]
            right_img = rect[1]
            sky_l, obs_l, edge_l = _roi_features(left_img)
            sky_r, obs_r, edge_r = _roi_features(right_img)

        # Steer towards sky (open) and away from obstacle proxy.
        sky_delta = float(sky_r - sky_l)
        obs_delta = float(obs_r - obs_l)
        steer = float(np.tanh(self.SKY_GAIN * sky_delta) - np.tanh(self.OBSTACLE_GAIN * obs_delta))

        danger_global = float(0.5 * (obs_l + obs_r))
        if danger_global < self.VISION_EDGE_THRESH:
            return 0.0, danger_global

        mag = min(self.VISION_AVOID_MAX, self.VISION_AVOID_GAIN * max(0.0, danger_global - self.VISION_EDGE_THRESH))
        return float(steer * mag), danger_global

    def _vision_avoid_bias(self, sim: MiniprojectSimulation) -> float:
        """Backward-compatible wrapper (returns bias only)."""
        b, _ = self._vision_avoid_bias_and_danger(sim)
        return float(b)
