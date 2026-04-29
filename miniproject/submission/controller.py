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
    STOP_DIST = 2.0

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

    # --- anti-roll grip (any level): activates EARLY on lateral tilt ---
    # The terrain-gated GRIP_* block above only fires on slopes / heavy tilt.
    # On flat levels (e.g. level 2 with obstacles only) the fly can still roll
    # over after a sharp turn or a bump, because nothing glues the stance legs
    # in time. This block runs on every level and reacts at smaller tilt
    # angles, sticking only the legs that are currently in contact (so it
    # never freezes swing legs in mid-air).
    TILT_GRIP_ENABLE = True
    # Lateral roll trigger: |xmat[2, 1]| above this -> stick stance legs.
    # 0.18 ~ ~10 deg of side-lean; below this we leave adhesion alone.
    TILT_GRIP_ROLL_ON = 0.18
    # Uprightness trigger (covers pitch + roll combined). Earlier than the
    # 0.60 used by the terrain block so we react before the fly is committed
    # to falling.
    TILT_GRIP_UPRIGHT_ON = 0.85

    # --- active roll compensation: lean body INTO the fall to counter it ---
    # Passive grip alone cannot reverse a roll once started. We additionally
    # extend (push down) the legs on the SAME side the body is leaning toward
    # ("down-side"), which raises that side of the thorax and produces a
    # counter-roll torque. This is applied to every physics step (not only
    # decisions) so the correction is continuous. Joint layout:
    #   joint_angles shape (42,) = 6 legs * 7 joints, order
    #     [lf, lm, lh, rf, rm, rh]
    #   joint index 3 within each leg = CTr_pitch ("knee") -- extending it
    #   lengthens the leg, pushing the body up on that side.
    TILT_LEAN_ENABLE = True
    # Below this |roll|, no compensation (avoids fighting normal gait sway).
    TILT_LEAN_ROLL_ON = 0.10
    # |roll| at which the compensation reaches full strength.
    TILT_LEAN_ROLL_FULL = 0.40
    # Maximum additive offset (radians) on the CTr_pitch joint. Positive
    # values flex more under flygym's standing convention; we want to
    # EXTEND, hence the negative sign on application.
    TILT_LEAN_GAIN = 0.30
    # Flip to +1.0 if the correction makes things worse (sign convention
    # depends on the joint frame). -1.0 = extend the down-side legs.
    TILT_LEAN_SIGN = +1.0

    # --- vision (Level 2+) ---
    # Use raw fish-eye corrected frames: sim.get_raw_vision(sim.fly.name)
    VISION_ENABLE = True
    VISION_USE_RAW = True
    VISION_DECISION_EVERY = 1  # computed at same cadence as olfaction by default

    # Vision ROI: "forward ground" region. Fractions of (H, W).
    # See further forward (obstacles often appear higher than the ground band).
    # Too low R0 delays detection until the fly is almost touching the obstacle.
    VIS_ROI_R0 = 0.08
    # Avoid the bottom band (ground edges are very strong and cause false "obstacle").
    VIS_ROI_R1 = 0.72
    VIS_ROI_C0 = 0.08
    VIS_ROI_C1 = 0.92
    VIS_CENTER_C0 = 0.40
    VIS_CENTER_C1 = 0.60

    # Obstacle proxy (no colors): strong edges + very dark pixels.
    # Thresholds / scales for obstacle "strength".
    VIS_DARK = 0.12
    VIS_EDGE = 0.10

    # EMA smoothing for areas and derivatives.
    VIS_EMA = 0.70
    VIS_D_EMA = 0.80

    # Normal avoidance (continuous).
    VIS_TURN_GAIN = 3.0
    VIS_TURN_MAX = 2.0
    VIS_CENTER_TURN_GAIN = 2.6
    VIS_SPEED_MIN = 0.60
    VIS_SPEED_CENTER_GAIN = 1.0
    # "Stop & steer around" behavior when obstacle is frontal/large.
    VIS_STOP_ENABLE = True
    # In the miniproject camera, obstacles often occupy a small fraction of ROI
    # until very close. Use lower thresholds to trigger "stop & go around" earlier.
    VIS_STOP_CENTER_ON = 0.030
    VIS_STOP_CENTER_OFF = 0.018
    VIS_STOP_TOTAL_ON = 0.030
    VIS_STOP_TOTAL_OFF = 0.018
    VIS_STOP_SPEED = 0.32
    VIS_STOP_LATCH_DECISIONS = 6
    VIS_STOP_TURN_BOOST = 1.8

    # Looming reflex (priority).
    LOOM_ENABLE = True
    LOOM_CENTER_ON = 0.14
    LOOM_CENTER_OFF = 0.10
    LOOM_DAREA_ON = 0.030
    LOOM_DAREA_OFF = 0.018
    LOOM_LATCH_DECISIONS = 10
    LOOM_TURN_GAIN = 3.0
    LOOM_TURN_MAX = 2.2
    LOOM_SPEED_MIN = 0.30

    # Contact-based bump reflex (highest priority): if we are physically pushing
    # into something, stop and pivot to go around even if vision is ambiguous.
    BUMP_ENABLE = True
    # Contact proxy uses external force XY norm (not torque).
    BUMP_CONTACT_ON = 4.0
    BUMP_CONTACT_OFF = 2.0
    BUMP_DCONTACT_ON = 6.0
    BUMP_DCONTACT_OFF = 2.5
    BUMP_LATCH_DECISIONS = 12
    BUMP_SPEED = 0.10
    BUMP_TURN_GAIN = 2.3
    BUMP_TURN_MAX = 2.2

    # Jam reflex: if commanded to move but not translating (blocked), stop and pivot.
    JAM_ENABLE = True
    JAM_MOVE_EPS = 2.0e-2
    JAM_MIN_DRIVE = 0.70
    JAM_TRIGGER_DECISIONS = 4
    JAM_LATCH_DECISIONS = 14

    # Fusion / stability.
    VIS_DIRECTION_MEMORY = True
    VIS_LAST_DIR_DECAY = 0.90
    VIS_TARGET_PROTECT = 0.65  # if target steer strong, visual turn can't flip it

    # --- debugging ---
    DEBUG = True
    # With default miniproject dt (~1e-4), decision period is ~500 steps.
    # 4 decisions ~= 2000 steps.
    DEBUG_EVERY_DECISIONS = 4
    DEBUG_MAX_DECISIONS = 260
    # If avoidance is pushing against the target-bearing turn, damp it. This
    # prevents long "runaways" where avoidance dominates and the fly drifts away.
    AVOID_OPPOSE_TARGET_ENABLE = True
    AVOID_OPPOSE_TARGET_MIN_TARGET_BIAS = 0.15
    AVOID_OPPOSE_TARGET_DAMP = 0.25
    AVOID_OPPOSE_TARGET_DIST = 45.0

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
    # Restored from the level 0/1 baseline (commit 7c6bdff). The previous
    # values (6e-7 / 2.5e-7) latched goal-mode almost immediately because
    # background odor is already in that range far from the banana, which
    # disabled all the search/recovery behaviors on level 2.
    GOAL_MODE_ODOR_ON = 2.5e-6
    GOAL_MODE_ODOR_OFF = 1.2e-6
    GOAL_MODE_MIN_DECISIONS = 12
    GOAL_MODE_MAX_DECISIONS = 60
    GOAL_MODE_SEARCH_DISABLE = True
    GOAL_MODE_TURN_SCALE = 0.65
    GOAL_MODE_MIN_DRIVE = 1.00

    # --- final approach (Level 2+) ---
    FINAL_APPROACH_ENABLE = True
    # Restored from baseline (was 12.0 in the last regression). At 12 the
    # final-approach mode kicks in too late, after avoidance has already
    # carried the fly far off course.
    FINAL_APPROACH_DIST = 18.0
    FINAL_APPROACH_BLEND_TARGET = 0.95
    FINAL_APPROACH_MIN_DRIVE = 1.05
    FINAL_CAST_ENABLE = True
    FINAL_CAST_DIST = 9.0
    FINAL_CAST_NO_IMPROVE_DECISIONS = 10
    FINAL_CAST_DURATION_DECISIONS = 18
    FINAL_CAST_GAIN = 0.9
    # Extra-stable close range behavior (prevents overshoot/runaway).
    CLOSE_STABLE_ENABLE = True
    CLOSE_STABLE_DIST = 8.0
    # Allow stronger convergence near goal (avoid orbiting at 3-6 units).
    CLOSE_STABLE_MAX_BIAS = 1.2
    CLOSE_STABLE_TURN_SCALE = 0.70
    CLOSE_STABLE_MAX_DRIVE = 0.60

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
    TARGET_STEER_GAIN = 4.0
    TARGET_STEER_GAIN_CLOSE = 10.0
    TARGET_STEER_CLOSE_DIST = 24.0
    TARGET_STEER_BIAS_SCALE = 0.25
    # In Level 2 the environment can break the odor trail; rely more on target bearing.
    TARGET_STEER_BLEND_WHEN_GRASS = 1.00
    TARGET_STEER_BLEND_GOALMODE = 1.00

    # --- homing recovery (Level 2+) ---
    HOMING_ENABLE = True
    HOMING_DRIFT_DIST = 3.0
    HOMING_TRIGGER_DECISIONS = 3
    HOMING_DURATION_DECISIONS = 45

    # --- reorientation if no early progress (Level 2+) ---
    REORIENT_ENABLE = True
    REORIENT_NO_PROGRESS_DECISIONS = 10
    REORIENT_MIN_IMPROVE = 0.7
    REORIENT_DURATION_DECISIONS = 28
    REORIENT_TURN_GAIN = 2.2
    REORIENT_PIVOT_BASE = 0.45

    # --- near-goal alignment (prevents orbiting) ---
    # Use a continuous near-goal controller instead of a latched pivot to avoid
    # getting stuck in long arcs around the banana.
    NEAR_GOAL_ENABLE = True
    NEAR_GOAL_DIST = 8.0
    NEAR_GOAL_V_MIN = 0.18
    NEAR_GOAL_V_MAX = 0.60
    NEAR_GOAL_W_GAIN = 0.55
    NEAR_GOAL_W_MAX = 0.45

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
    SEARCH_NO_IMPROVE_DECISIONS = 18
    SEARCH_IMPROVE_EPS = 0.10
    SEARCH_DURATION_DECISIONS = 25
    SEARCH_CAST_GAIN = 1.6
    SEARCH_DRIVE = 0.95
    SEARCH_SLOPE_MAX = 0.45
    SEARCH_UPRIGHT_MIN = 0.65

    # --- DAgger learned vision policy hook (optional) ---
    # When a path is provided (either on the class or via `__init__`), the
    # scripted `_vision_avoid_bias_and_danger` is replaced by the learned
    # `VisionPolicy`. Scripted safety reflexes (bump / looming) still win
    # if their thresholds are crossed (see `_dagger_vision_bias_and_speed`).
    DAGGER_POLICY_PATH: str | None = None
    DAGGER_BLEND = 1.0            # 1.0 = policy replaces vision module output
    DAGGER_TURN_EMA = 0.70        # EMA smoothing on policy turn (prevents zig-zags)
    DAGGER_SPEED_MIN = 0.35       # lower bound for speed_scale returned by policy
    DAGGER_BUMP_CONTACT_ON = 4.0  # fallback to scripted reflex above this force
    DAGGER_LOOM_DAREA_ON = 0.030  # fallback to scripted reflex above this looming

    def __init__(
        self,
        sim: MiniprojectSimulation,
        dagger_policy_path: str | None = None,
    ):
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
        self._vision_rect_idx = None
        self._vision_rect_mask = None
        self._vision_step_count = 0
        self._last_xy = None
        self._stuck_decisions = 0
        self._escape_decisions_left = 0
        self._escape_dir = 1
        self._flip_decisions = 0
        self._tilt_decisions = 0
        self._banana_xy = None
        self._best_dist = None
        self._no_improve = 0
        self._search_left = 0
        self._search_dir = 1
        self._start_dist = None
        self._request_reset = False
        self._runaway_far = 0
        self._near_worsen = 0
        self._last_target_bearing = 0.0
        self._last_dist_to_banana = None
        self._debug_decisions = 0
        # Vision state (features + looming latch)
        self._vis_left_area = 0.0
        self._vis_right_area = 0.0
        self._vis_center_area = 0.0
        self._vis_total_area = 0.0
        self._vis_total_area_prev = 0.0
        self._vis_d_total_area = 0.0
        self._vis_speed_scale = 1.0
        self._vis_last_dir = 1.0
        self._loom_left = 0
        self._stop_left = 0
        self._bump_left = 0
        self._bump_contact_ema = 0.0
        self._bump_contact_prev = 0.0
        self._jam_left = 0
        self._jam_dir = 1
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

        # --- DAgger learned vision policy (optional drop-in replacement) ---
        # `dagger_policy_path` arg wins over the class attribute, so external
        # code can do `Controller(sim, dagger_policy_path="path/to/policy.pt")`.
        self._dagger_policy = None
        self._dagger_feat = None
        self._dagger_turn_ema = 0.0
        self._dagger_prev_turn = 0.0
        self._dagger_prev_speed = 1.0
        path = dagger_policy_path if dagger_policy_path is not None else self.DAGGER_POLICY_PATH
        if path is not None:
            try:
                from miniproject.dagger import VisionFeatureExtractor, VisionPolicy

                self._dagger_policy = VisionPolicy.load(path, map_location="cpu")
                self._dagger_feat = VisionFeatureExtractor(sim)
                print(f"[Controller] Loaded DAgger vision policy from {path}", flush=True)
            except Exception as e:
                # Don't crash the sim if the checkpoint fails to load --
                # fall back silently to the scripted vision module.
                print(f"[Controller] WARN: failed to load DAgger policy ({e}); using scripted vision.", flush=True)
                self._dagger_policy = None
                self._dagger_feat = None

    # ------------------------------------------------------------------
    def step(self, sim: MiniprojectSimulation):
        is_decision_step = self._step_count % self._decision_every == 0
        if is_decision_step:
            self._drives = self._compute_drives(sim)
            self._debug_decisions = self._step_count // self._decision_every
        self._step_count += 1

        joint_angles, adhesion = self.turning_controller.step(self._drives)

        # --- Active roll compensation (every level) ---
        # When the body leans sideways, extend the down-side legs to push that
        # half of the thorax up, producing a counter-roll torque. Needed on
        # terrain (level 1+) to keep the fly upright on slopes; on level 2 the
        # reflex-pivot override below also handles obstacle escape, so the two
        # mechanisms compose without fighting each other.
        if self.TILT_LEAN_ENABLE:
            try:
                xmat_lean = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
                roll_lean = float(xmat_lean[2, 1])
            except Exception:
                roll_lean = 0.0
            roll_excess = abs(roll_lean) - float(self.TILT_LEAN_ROLL_ON)
            if roll_excess > 0.0 and self._escape_decisions_left <= 0:
                ramp = min(
                    1.0,
                    roll_excess
                    / max(
                        1e-3,
                        float(self.TILT_LEAN_ROLL_FULL)
                        - float(self.TILT_LEAN_ROLL_ON),
                    ),
                )
                offset = (
                    float(self.TILT_LEAN_SIGN)
                    * float(self.TILT_LEAN_GAIN)
                    * ramp
                )
                # roll_lean = xmat[2, 1] = z-component of body's +Y (left)
                # axis in world frame. >0 means LEFT side points UP, i.e.
                # the body is rolled to the RIGHT (right side is down).
                # We push UP the down-side, so:
                #   roll_lean > 0  -> right side down -> extend RIGHT legs
                #   roll_lean < 0  -> left  side down -> extend LEFT  legs
                if roll_lean > 0:
                    leg_indices = (3, 4, 5)  # rf, rm, rh
                else:
                    leg_indices = (0, 1, 2)  # lf, lm, lh
                for li in leg_indices:
                    base = li * 7
                    # Apply to CTr_pitch (knee, joint 3) AND FTi_pitch (joint
                    # 5) -- both control vertical leg extension. Halve FTi
                    # to keep the gait shape coherent.
                    joint_angles[base + 3] += offset
                    joint_angles[base + 5] += 0.5 * offset

        # Reset request: apply at decision cadence, independent of grip logic.
        if is_decision_step and self._request_reset:
            try:
                sim.reset()
            except Exception:
                pass
            self._request_reset = False
            self._flip_decisions = 0
            self._tilt_decisions = 0
            self._stopped = False
            self._smooth_bias = 0.0
            self._escape_decisions_left = 0
            self._stuck_decisions = 0
            self._last_xy = None
            self._best_dist = None
            self._no_improve = 0
            self._search_left = 0
            self._bump_left = 0
            self._jam_left = 0
            return joint_angles, adhesion

        # --- Orientation safety (terrain levels): if persistently tilted, reset ---
        # On flat worlds the "upright" signal can fluctuate due to contacts; avoid
        # resetting there so we can finish simple obstacle scenarios.
        if self._enable_terrain and is_decision_step:
            try:
                _, _, uprightness = self._get_orientation(sim)
            except Exception:
                uprightness = 1.0

            if uprightness < 0.55:
                self._tilt_decisions += 1
            else:
                self._tilt_decisions = 0

            if self._tilt_decisions >= 12:
                self._request_reset = True
                self._tilt_decisions = 0

        # --- Grip control (terrain only): extra adhesion on slopes / when tilted ---
        # Keep grip logic conservative: excessive adhesion can freeze locomotion.
        if self._enable_terrain and is_decision_step:
            # During escape, reduce adhesion to allow the fly to pivot and unstick.
            if self._escape_decisions_left > 0:
                adhesion = np.zeros_like(adhesion)
                return joint_angles, adhesion

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

        # --- Anti-roll grip (every level): early reaction to side-tilt ---
        # Runs independently of the terrain block above. We want this on
        # level 2 (flat + obstacles) where the existing block is gated off.
        # When the fly starts rolling sideways or tipping forward/back, we
        # glue all stance legs (legs currently in contact with the ground)
        # to arrest the rotation before it becomes a fall. Swing legs are
        # left alone so the alternating-tripod gait keeps walking.
        if self.TILT_GRIP_ENABLE and is_decision_step and not self._enable_terrain:
            try:
                xmat = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
                roll_ind = float(xmat[2, 1])
                uprightness = float(xmat[2, 2])
            except Exception:
                roll_ind = 0.0
                uprightness = 1.0

            tilted = (
                abs(roll_ind) >= float(self.TILT_GRIP_ROLL_ON)
                or uprightness <= float(self.TILT_GRIP_UPRIGHT_ON)
            )
            if tilted and self._escape_decisions_left <= 0:
                try:
                    contact_forces = sim.mj_data.cfrc_ext[self._contact_body_ids, 3:]
                    contact_mag = np.linalg.norm(contact_forces, axis=1)
                    stance = contact_mag > self.CONTACT_THRESHOLD
                    n = min(len(adhesion), len(stance))
                    # Force adhesion to 1.0 on every leg that is currently
                    # touching the ground; do NOT touch swing legs.
                    adhesion[:n] = np.where(stance[:n], 1.0, adhesion[:n])
                    adhesion = np.clip(adhesion, 0.0, 1.0)
                except Exception:
                    pass

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

        # Banana position might not be available at __init__ time depending on how the
        # simulation/world is constructed. Re-try lazily so distance-based stop works.
        if self._banana_xy is None:
            try:
                self._banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
            except Exception:
                self._banana_xy = None

        # Distance bookkeeping for Level 2 robustness (runaway detection).
        dist_to_banana = None
        if self._banana_xy is not None:
            try:
                thorax_xy = sim.get_body_positions(sim.fly.name)[self._thorax_idx, :2]
            except Exception:
                thorax_xy = sim.mj_data.xpos[self._thorax_body_id, :2]
            thorax_xy = np.asarray(thorax_xy, dtype=float)
            dist_to_banana = float(np.linalg.norm(thorax_xy - self._banana_xy))
            self._last_dist_to_banana = dist_to_banana
            if dist_to_banana <= float(self.STOP_DIST):
                self._stopped = True
                return np.array([0.0, 0.0])
            if self._best_dist is None:
                self._best_dist = dist_to_banana
            else:
                self._best_dist = min(self._best_dist, dist_to_banana)

            if (
                self.NEAR_RESET_ENABLE
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
                and self._best_dist <= self.NEAR_RESET_DIST
                and dist_to_banana >= (self.RUNAWAY_FAR_DIST + 5.0)
            ):
                self._request_reset = True

            # If we haven't improved at all for a while, do a short reorientation
            # towards the banana (helps escaping long wrong-way trajectories).
            if self.REORIENT_ENABLE and self._start_dist is not None:
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

            # If we are persistently drifting away after having been closer,
            # force a short "homing" phase towards the banana.
            if (
                self.HOMING_ENABLE
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

        moved = float("inf")
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

        # Jam reflex (all levels): if we're commanded to move but not translating, pivot.
        if self.JAM_ENABLE:
            max_drive_cmd = float(np.max(self._drives)) if getattr(self, "_drives", None) is not None else 0.0
            if moved < float(self.JAM_MOVE_EPS) and max_drive_cmd >= float(self.JAM_MIN_DRIVE):
                self._stuck_decisions += 1
            else:
                # Do not keep counting "stuck" forever across normal motion.
                self._stuck_decisions = 0

            if self._jam_left > 0:
                self._jam_left -= 1

            if self._jam_left <= 0 and self._stuck_decisions >= int(self.JAM_TRIGGER_DECISIONS):
                self._jam_left = int(self.JAM_LATCH_DECISIONS)
                self._stuck_decisions = 0
                self._jam_dir *= -1

            if self._jam_left > 0:
                maxd = self.MAX_DRIVE_TERRAIN if self._enable_terrain else self.MAX_DRIVE
                mind = self.MIN_SIDE_DRIVE_TERRAIN if self._enable_terrain else self.MIN_SIDE_DRIVE
                if self._jam_dir > 0:
                    return np.array([maxd, mind], dtype=float)
                return np.array([mind, maxd], dtype=float)

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
                target_bias = -float(self.TARGET_STEER_BIAS_SCALE) * g * bearing
                self._last_target_bearing = bearing

                # Near-goal continuous controller: left=v-w, right=v+w (no backward).
                if (
                    self.NEAR_GOAL_ENABLE
                    and dist_to_banana is not None
                    and dist_to_banana < float(self.NEAR_GOAL_DIST)
                ):
                    max_drive = self.MAX_DRIVE_TERRAIN if self._enable_terrain else self.MAX_DRIVE
                    # Slow down near target to prevent orbiting.
                    v = float(np.clip(0.12 * float(dist_to_banana), self.NEAR_GOAL_V_MIN, self.NEAR_GOAL_V_MAX))
                    w = float(np.clip(self.NEAR_GOAL_W_GAIN * bearing, -self.NEAR_GOAL_W_MAX, self.NEAR_GOAL_W_MAX))
                    left = float(np.clip(v - w, 0.0, max_drive))
                    right = float(np.clip(v + w, 0.0, max_drive))
                    return np.array([left, right], dtype=float)
        else:
            self._last_dist_to_banana = None
            self._last_target_bearing = 0.0

        if (
            self.DEBUG
            and self._debug_decisions <= self.DEBUG_MAX_DECISIONS
            and (self._debug_decisions % self.DEBUG_EVERY_DECISIONS == 0)
        ):
            print(
                f"[dbg d={self._debug_decisions:4d}] dist={dist_to_banana} "
                f"best={self._best_dist} "
                f"target_bias={target_bias:.3f} bearing={getattr(self, '_last_target_bearing', 0.0):.3f} "
                f"mean_odor={mean_odor:.3e}",
                flush=True,
            )

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

        # ---- vision (Level 2+): avoidance + looming, hierarchical ----
        self._vis_speed_scale = 1.0
        if self.VISION_ENABLE and self._enable_grass:
            if self._vision_step_count % self.VISION_DECISION_EVERY == 0:
                # Route through the DAgger policy if one was loaded, otherwise
                # fall back to the hand-crafted vision module.
                if self._dagger_policy is not None and self._dagger_feat is not None:
                    vision_turn_bias, danger = self._dagger_vision_bias_and_speed(sim)
                else:
                    vision_turn_bias, danger = self._vision_avoid_bias_and_danger(sim)

                # Apply visual speed modulation (set inside vision functions).
                # Combine with the nominal controller bias; do not allow vision to flip
                # a strong goal-directed steering signal.
                if abs(target_bias) >= self.VIS_TARGET_PROTECT and np.sign(bias) != 0:
                    if np.sign(bias + vision_turn_bias) != np.sign(bias):
                        vision_turn_bias = 0.0

                bias += float(vision_turn_bias)

                if (
                    self.DEBUG
                    and self._debug_decisions <= self.DEBUG_MAX_DECISIONS
                    and (self._debug_decisions % self.DEBUG_EVERY_DECISIONS == 0)
                ):
                    print(
                        f"[dbg vis d={self._debug_decisions:4d}] "
                        f"turn={vision_turn_bias:.3f} danger={danger:.3f} speed_scale={self._vis_speed_scale:.2f} "
                        f"areas=({self._vis_left_area:.3f},{self._vis_right_area:.3f}) center={self._vis_center_area:.3f} "
                        f"total={self._vis_total_area:.3f} d_total={self._vis_d_total_area:.3f} "
                        f"bump={bool(self._bump_left>0)} jam={bool(self._jam_left>0)} "
                        f"loom={bool(self._loom_left>0)} stop={bool(self._stop_left>0)}",
                        flush=True,
                    )

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
        # Vision can slow down forward motion in front of obstacles.
        base_drive = float(np.clip(base_drive * float(self._vis_speed_scale), min_drive, max_drive))

        # Near the goal, reduce forward speed when not facing the target.
        # This prevents stable "orbiting" at a few units distance.
        if dist_to_banana is not None and dist_to_banana < 8.0:
            bearing = float(getattr(self, "_last_target_bearing", 0.0))
            align = float(np.clip(1.0 - 0.85 * (abs(bearing) / np.pi), 0.20, 1.0))
            base_drive = float(np.clip(base_drive * align, min_drive, max_drive))

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

        # ---- Reflex pivot override (Level 2+): when a vision/contact reflex is
        # latched, the standard "asymmetric base_drive" formula only produces
        # ~30% asymmetry, which is not enough to escape an obstacle the fly is
        # already pressed against. In that case we ignore base_drive entirely
        # and command a strong pivot in place: max drive on one side, min side
        # drive on the other. The direction is taken from the smoothed bias
        # (which already encodes target_bias + vision turn). This is what
        # actually unsticks the fly from "drive into wall" loops on level 2.
        reflex_active = (
            self._enable_grass
            and (
                self._bump_left > 0
                or self._loom_left > 0
                or self._stop_left > 0
            )
        )
        if reflex_active:
            pivot_dir = 1.0 if self._smooth_bias > 0.0 else -1.0
            if pivot_dir > 0:
                drives = np.array([max_drive, min_side], dtype=float)
            else:
                drives = np.array([min_side, max_drive], dtype=float)
        else:
            drives = np.full(2, base_drive, dtype=float)
            side = int(bias_norm > 0)
            drives[side] -= abs(bias_norm) * turn_mod * base_drive
            drives[side] = max(min_side, drives[side])
            drives = np.clip(drives, 0.0, max_drive)

        if (
            self.DEBUG
            and self._debug_decisions <= self.DEBUG_MAX_DECISIONS
            and (self._debug_decisions % self.DEBUG_EVERY_DECISIONS == 0)
        ):
            print(
                f"[dbg out d={self._debug_decisions:4d}] bias={bias:.3f} bias_norm={bias_norm:.3f} "
                f"turn_mod={turn_mod:.3f} base={base_drive:.3f} drives=({drives[0]:.3f},{drives[1]:.3f}) "
                f"reflex={reflex_active}",
                flush=True,
            )
        return drives

    def _vision_avoid_bias_and_danger(
        self, sim: MiniprojectSimulation
    ) -> tuple[float, float]:
        """Return a visual turning bias and a danger scalar.

        This is a simple two-layer architecture:
        - normal obstacle avoidance: continuous turn + speed scaling
        - looming reflex (priority): triggers if obstacle grows fast or occupies center

        We use raw vision, but keep the computation local and deterministic.
        """

        def _to_float01(img: np.ndarray) -> np.ndarray:
            a = np.asarray(img, dtype=np.float32)
            if a.ndim == 2:
                a = np.stack([a, a, a], axis=-1)
            if a.max() > 1.0:
                a = a / 255.0
            return np.clip(a, 0.0, 1.0)

        def _roi(img01: np.ndarray) -> np.ndarray:
            h, w = img01.shape[0], img01.shape[1]
            r0, r1 = int(h * self.VIS_ROI_R0), int(h * self.VIS_ROI_R1)
            c0, c1 = int(w * self.VIS_ROI_C0), int(w * self.VIS_ROI_C1)
            return img01[r0:r1, c0:c1, :]

        def _obstacle_strength(gray: np.ndarray) -> np.ndarray:
            """Continuous obstacle strength in [0,1] from edges + darkness."""
            dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
            dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
            edge = 0.5 * (dx + dy)
            # Only count sufficiently strong edges (avoid constant low-level texture).
            edge_thr = max(1e-6, float(self.VIS_EDGE))
            edge_s = np.clip((edge - edge_thr) / edge_thr, 0.0, 1.0)
            dark_s = np.clip((float(self.VIS_DARK) - gray) / max(1e-6, float(self.VIS_DARK)), 0.0, 1.0)
            return np.maximum(edge_s, dark_s).astype(np.float32)

        def extract_visual_features(left_img01: np.ndarray, right_img01: np.ndarray) -> dict:
            # grayscale in ROI
            lroi = _roi(left_img01)
            rroi = _roi(right_img01)
            lgray = (0.299 * lroi[..., 0] + 0.587 * lroi[..., 1] + 0.114 * lroi[..., 2]).astype(np.float32)
            rgray = (0.299 * rroi[..., 0] + 0.587 * rroi[..., 1] + 0.114 * rroi[..., 2]).astype(np.float32)

            lst = _obstacle_strength(lgray)
            rst = _obstacle_strength(rgray)

            def _area_x(strength: np.ndarray) -> tuple[float, float]:
                area = float(strength.mean())
                if area <= 1e-9:
                    return 0.0, 0.0
                cols = strength.mean(axis=0)  # (W,)
                xs = np.linspace(-1.0, 1.0, cols.shape[0], dtype=np.float32)
                x_mean = float((cols * xs).sum() / max(1e-9, float(cols.sum())))
                return area, x_mean

            left_area, left_x = _area_x(lst)
            right_area, right_x = _area_x(rst)

            # center area: both eyes, center columns only
            def _center_area(strength: np.ndarray) -> float:
                w = strength.shape[1]
                c0 = int(w * self.VIS_CENTER_C0)
                c1 = int(w * self.VIS_CENTER_C1)
                if c1 <= c0:
                    return float(strength.mean())
                return float(strength[:, c0:c1].mean())

            center_area = 0.5 * (_center_area(lst) + _center_area(rst))
            total_area = 0.5 * (left_area + right_area)

            # temporal derivative (EMA)
            prev = float(self._vis_total_area)
            self._vis_total_area = float(self.VIS_EMA * prev + (1.0 - self.VIS_EMA) * total_area)
            d_raw = float(self._vis_total_area - self._vis_total_area_prev)
            self._vis_total_area_prev = float(self._vis_total_area)
            self._vis_d_total_area = float(self.VIS_D_EMA * self._vis_d_total_area + (1.0 - self.VIS_D_EMA) * d_raw)

            # store smoothed areas for debugging/fusion
            self._vis_left_area = float(self.VIS_EMA * self._vis_left_area + (1.0 - self.VIS_EMA) * left_area)
            self._vis_right_area = float(self.VIS_EMA * self._vis_right_area + (1.0 - self.VIS_EMA) * right_area)
            self._vis_center_area = float(self.VIS_EMA * self._vis_center_area + (1.0 - self.VIS_EMA) * center_area)

            return {
                "left_area": float(self._vis_left_area),
                "right_area": float(self._vis_right_area),
                "left_x": float(left_x),
                "right_x": float(right_x),
                "center_area": float(self._vis_center_area),
                "total_area": float(self._vis_total_area),
                "d_total_area": float(self._vis_d_total_area),
            }

        def compute_obstacle_avoidance(feat: dict) -> tuple[float, float, float]:
            # turn: obstacle on left => turn right (positive)
            lr = float(feat["left_area"] - feat["right_area"])
            center = float(feat["center_area"])
            turn = float(np.clip(self.VIS_TURN_GAIN * lr, -self.VIS_TURN_MAX, self.VIS_TURN_MAX))
            # if center occupied, amplify turn and slow down
            turn *= float(1.0 + self.VIS_CENTER_TURN_GAIN * np.clip(center / 0.12, 0.0, 1.0))
            speed_scale = float(1.0 - self.VIS_SPEED_CENTER_GAIN * np.clip(center / 0.18, 0.0, 1.0))
            speed_scale = float(np.clip(speed_scale, self.VIS_SPEED_MIN, 1.0))
            danger = float(np.clip(feat["total_area"] + 0.8 * center, 0.0, 1.0))
            return turn, speed_scale, danger

        def compute_stop_and_go_around(feat: dict) -> tuple[bool, float, float]:
            """When obstacle is frontal/large: slow almost to stop and commit to an avoid direction."""
            if not self.VIS_STOP_ENABLE:
                return False, 0.0, 1.0
            center = float(feat["center_area"])
            total = float(feat["total_area"])

            on = (center >= self.VIS_STOP_CENTER_ON) or (total >= self.VIS_STOP_TOTAL_ON)
            off = (center <= self.VIS_STOP_CENTER_OFF) and (total <= self.VIS_STOP_TOTAL_OFF)

            if self._stop_left > 0:
                self._stop_left -= 1
            if on:
                self._stop_left = max(self._stop_left, int(self.VIS_STOP_LATCH_DECISIONS))
            if off and self._stop_left <= 0:
                self._stop_left = 0

            active = self._stop_left > 0
            if not active:
                return False, 0.0, 1.0

            lr = float(feat["left_area"] - feat["right_area"])
            # choose a stable direction if symmetric
            if abs(lr) < 1e-4 and self.VIS_DIRECTION_MEMORY:
                dir_sign = float(self._vis_last_dir)
            else:
                dir_sign = 1.0 if lr > 0 else -1.0  # obstacle left => turn right (+)

            turn = float(np.clip(self.VIS_STOP_TURN_BOOST * dir_sign, -self.VIS_TURN_MAX, self.VIS_TURN_MAX))
            speed = float(self.VIS_STOP_SPEED)
            return True, turn, speed

        def compute_looming_reflex(feat: dict) -> tuple[bool, float, float]:
            if not self.LOOM_ENABLE:
                return False, 0.0, 1.0
            center = float(feat["center_area"])
            d_area = float(feat["d_total_area"])

            on = (center >= self.LOOM_CENTER_ON) or (d_area >= self.LOOM_DAREA_ON)
            off = (center <= self.LOOM_CENTER_OFF) and (d_area <= self.LOOM_DAREA_OFF)

            if self._loom_left > 0:
                self._loom_left -= 1
            if on:
                self._loom_left = max(self._loom_left, int(self.LOOM_LATCH_DECISIONS))
            if off and self._loom_left <= 0:
                self._loom_left = 0

            active = self._loom_left > 0
            if not active:
                return False, 0.0, 1.0

            lr = float(feat["left_area"] - feat["right_area"])
            if abs(lr) < 1e-4 and self.VIS_DIRECTION_MEMORY:
                dir_sign = float(self._vis_last_dir)
            else:
                dir_sign = 1.0 if lr > 0 else -1.0  # obstacle left => turn right (+)

            turn = float(np.clip(self.LOOM_TURN_GAIN * dir_sign, -self.LOOM_TURN_MAX, self.LOOM_TURN_MAX))
            speed = float(self.LOOM_SPEED_MIN)
            return True, turn, speed

        def compute_bump_reflex(feat: dict) -> tuple[bool, float, float, float]:
            """Highest-priority reflex based on physical contacts."""
            if not self.BUMP_ENABLE:
                return False, 0.0, 1.0, 0.0

            contact_max = 0.0
            try:
                # cfrc_ext = [force(3), torque(3)] in global frame. We want a proxy
                # for "pushing into something", so use horizontal force norm.
                cf = sim.mj_data.cfrc_ext[self._contact_body_ids, :3]
                mag = np.linalg.norm(cf[:, :2], axis=1)
                contact_max = float(np.max(mag)) if mag.size > 0 else 0.0
            except Exception:
                contact_max = 0.0

            # Smooth and differentiate to avoid latching on constant stance forces.
            prev = float(self._bump_contact_ema)
            self._bump_contact_ema = float(0.90 * prev + 0.10 * contact_max)
            d_contact = float(self._bump_contact_ema - self._bump_contact_prev)
            self._bump_contact_prev = float(self._bump_contact_ema)

            # To avoid latching at spawn (normal stance forces), require both:
            # - large horizontal contact force, and
            # - some visual evidence of obstacle in ROI.
            vis_hint = (float(feat["center_area"]) >= 0.012) or (float(feat["total_area"]) >= 0.018)

            on = (d_contact >= float(self.BUMP_DCONTACT_ON)) and vis_hint
            off = (d_contact <= float(self.BUMP_DCONTACT_OFF)) and (not vis_hint)

            if self._bump_left > 0:
                self._bump_left -= 1
            if on:
                self._bump_left = max(self._bump_left, int(self.BUMP_LATCH_DECISIONS))
            if off and self._bump_left <= 0:
                self._bump_left = 0

            active = self._bump_left > 0
            if not active:
                return False, 0.0, 1.0, contact_max

            lr = float(feat["left_area"] - feat["right_area"])
            if abs(lr) < 1e-4 and self.VIS_DIRECTION_MEMORY:
                dir_sign = float(self._vis_last_dir)
            else:
                dir_sign = 1.0 if lr > 0 else -1.0  # obstacle left => turn right (+)

            turn = float(np.clip(self.BUMP_TURN_GAIN * dir_sign, -self.BUMP_TURN_MAX, self.BUMP_TURN_MAX))
            speed = float(self.BUMP_SPEED)
            return True, turn, speed, contact_max

        # Read raw frames
        frames = None
        if self.VISION_USE_RAW:
            try:
                frames = sim.get_raw_vision(sim.fly.name)
            except Exception:
                frames = None

        if frames is None or len(frames) == 0:
            self._vis_speed_scale = 1.0
            return 0.0, 0.0

        left_img01 = _to_float01(frames[0])
        right_img01 = _to_float01(frames[1] if len(frames) > 1 else frames[0])

        feat = extract_visual_features(left_img01, right_img01)
        bump_active, bump_turn, bump_speed, bump_cmax = compute_bump_reflex(feat)
        looming_active, loom_turn, loom_speed = compute_looming_reflex(feat)
        stop_active, stop_turn, stop_speed = compute_stop_and_go_around(feat)
        avoid_turn, avoid_speed, danger = compute_obstacle_avoidance(feat)

        if bump_active:
            turn = float(bump_turn)
            self._vis_speed_scale = float(bump_speed)
        elif looming_active:
            turn = float(loom_turn)
            self._vis_speed_scale = float(loom_speed)
        elif stop_active:
            turn = float(stop_turn)
            self._vis_speed_scale = float(stop_speed)
        else:
            turn = float(avoid_turn)
            self._vis_speed_scale = float(avoid_speed)

        # Update memory of last chosen avoid direction (smooth, prevents flip-flop).
        if self.VIS_DIRECTION_MEMORY:
            if abs(turn) > 1e-6:
                self._vis_last_dir = float(self.VIS_LAST_DIR_DECAY * self._vis_last_dir + (1.0 - self.VIS_LAST_DIR_DECAY) * np.sign(turn))
                if abs(self._vis_last_dir) < 1e-6:
                    self._vis_last_dir = 1.0

        return float(turn), float(danger)

    def _vision_avoid_bias(self, sim: MiniprojectSimulation) -> float:
        """Backward-compatible wrapper (returns bias only)."""
        b, _ = self._vision_avoid_bias_and_danger(sim)
        return float(b)

    # ------------------------------------------------------------------
    def _dagger_vision_bias_and_speed(
        self, sim: MiniprojectSimulation
    ) -> tuple[float, float]:
        """Learned drop-in replacement for `_vision_avoid_bias_and_danger`.

        Behavior:
            * Extract the compact feature vector.
            * If a safety condition is met (big contact force OR fast looming),
              delegate to the scripted reflex pipeline. This guarantees that
              an untrained / still-learning policy cannot override emergencies.
            * Otherwise query the MLP and return (turn, danger) while setting
              `self._vis_speed_scale` from the policy's speed output.
        """
        feat = self._dagger_feat.extract(
            sim,
            prev_turn=self._dagger_prev_turn,
            prev_speed=self._dagger_prev_speed,
        )

        contact_fmax = float(feat[11])
        d_total_area = float(feat[4])
        if (
            contact_fmax >= float(self.DAGGER_BUMP_CONTACT_ON)
            or d_total_area >= float(self.DAGGER_LOOM_DAREA_ON)
        ):
            # Safety: hand control back to scripted reflex pipeline.
            return self._vision_avoid_bias_and_danger(sim)

        turn_raw, speed_raw = self._dagger_policy.act(feat)
        self._dagger_prev_turn = float(turn_raw)
        self._dagger_prev_speed = float(speed_raw)

        # EMA smoothing on turn to damp high-frequency zig-zags.
        self._dagger_turn_ema = (
            self.DAGGER_TURN_EMA * self._dagger_turn_ema
            + (1.0 - self.DAGGER_TURN_EMA) * float(turn_raw)
        )
        turn = float(np.clip(
            self.DAGGER_BLEND * self._dagger_turn_ema,
            -self.VIS_TURN_MAX,
            self.VIS_TURN_MAX,
        ))

        speed = float(np.clip(speed_raw, float(self.DAGGER_SPEED_MIN), 1.0))
        self._vis_speed_scale = speed

        # danger proxy used by downstream debug prints / FSM; conservative.
        danger = float(np.clip(
            float(feat[3]) + 0.8 * float(feat[2]),
            0.0,
            1.0,
        ))
        return turn, danger
