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
    # Centre band widened (30-70% instead of 40-60%): the fly is BIG compared to a
    # grass blade; an obstacle just off the geometric centre still requires
    # avoidance. A wider centre means the turn boost / stop logic kicks in
    # whenever the blade is anywhere in the forward hemisphere.
    VIS_CENTER_C0 = 0.30
    VIS_CENTER_C1 = 0.70

    # Obstacle proxy: strong edges + very dark pixels (legacy fallback).
    # Thresholds / scales for obstacle "strength".
    VIS_DARK = 0.12
    VIS_EDGE = 0.10

    # --- color-based perception (Level 2+) ---
    # In the miniproject scene every "interesting" object has a fixed RGBA:
    #   * grass blades and the terrain itself : (0, 1, 0, 1)  -> "pure green"
    #   * banana                              : (1, 1, 0, 1)  -> yellow
    #   * dragonfly head/thorax               : (0.15, 0.55, 0.2, 1)  -> dark green
    #   * dragonfly eyes                      : (0.6, 0.1, 0.1, 1)  -> red
    #   * dragonfly wings                     : (0.7, 0.9, 1.0, 0.35) -> light blue
    # See src/miniproject/arena/{grass,terrain,banana,dragonfly}.py.
    #
    # Probe in `tmp_color_probe.py` shows that the *ground itself* is rendered
    # close to (0, 1, 0) too, so we can NOT just mask "every pure-green pixel
    # is an obstacle". What works:
    #   * pure-green pixels in the *upper half* of the ROI come from grass
    #     blades that stick up above the ground (or from the dragonfly body
    #     against the sky), so they are reliable obstacle hints.
    #   * pure-green pixels in the lower half are dominated by the ground, so
    #     we still rely on the legacy edge / dark detector there.
    #   * pure-red pixels are a unique colour in the scene -> dragonfly eyes,
    #     used as a separate "danger" channel.
    #
    # All thresholds are conservative; the detector composes (`max`) with the
    # legacy edge/dark strength so we are never *worse* than before.
    VIS_COLOR_ENABLE = True
    # Pure-green chroma test (terrain *and* blade pixels both pass it):
    #   (g - r) > GREEN_DELTA AND (g - b) > GREEN_DELTA AND g > GREEN_MIN
    # Same thresholds work for the grass mask everywhere in the file.
    VIS_GRASS_GREEN_DELTA = 0.20
    VIS_GRASS_GREEN_MIN = 0.45
    # Multiplicative weight kept for legacy callers that fuse the grass
    # channel with other strengths via a weighted max.
    VIS_GRASS_WEIGHT = 1.0

    # Dragonfly (red eyes) detector:
    #   r - g > RED_DELTA AND r - b > RED_DELTA AND r > RED_MIN
    VIS_DF_RED_DELTA = 0.20
    VIS_DF_RED_MIN = 0.40
    # Treat the dragonfly as an emergency above this fraction of red pixels
    # in the ROI. The eyes are tiny so the signal stays small even up close.
    VIS_DF_AREA_ON = 0.0040
    VIS_DF_AREA_OFF = 0.0015
    # Latch the avoidance for a few decision steps to keep a steady turn.
    VIS_DF_LATCH_DECISIONS = 10
    VIS_DF_TURN_GAIN = 3.5
    VIS_DF_TURN_MAX = 2.4
    VIS_DF_SPEED_MIN = 0.20
    # EMA on the smoothed dragonfly area (separate from the main `total_area`
    # EMA so the colour channel reacts faster than the edge channel).
    VIS_DF_EMA = 0.55

    # --- vision debug overlay (cheap visual sanity check) ---
    # When enabled, the controller can render a per-pixel mask overlay on top
    # of the raw fly vision, useful to tune the colour thresholds in the
    # `run_with_controller.py --debug-vision` mode. The flag here only
    # *allows* the overlay; the actual call site is the interactive script.
    VIS_DEBUG_OVERLAY = True

    # EMA smoothing for areas and derivatives.
    # Lower α → faster response (less lag), at the cost of slightly more
    # noise. 0.45 settles to 90 % of a step change in ~4 decisions instead
    # of ~7 with the previous 0.70. Raw (unsmoothed) lr is now used for
    # the proportional turn so this only affects reflex thresholds.
    VIS_EMA = 0.45
    VIS_D_EMA = 0.70

    # Normal avoidance (continuous).
    # Raw lr (instantaneous, no EMA lag) is used for the turn direction;
    # gains and clipping are applied after. TURN_MAX raised to 5.0 so the
    # fly can commit to a near-90° pivot when very close to a blade.
    VIS_TURN_GAIN = 10.0
    VIS_TURN_MAX = 5.0
    VIS_CENTER_TURN_GAIN = 2.6
    VIS_SPEED_MIN = 0.20
    VIS_SPEED_CENTER_GAIN = 1.4
    # Proximity boost: the bigger total_area, the closer the obstacle, so
    # we ramp the turn further. VIS_PROX_REF lowered from 0.020 to 0.008
    # so the boost saturates at the same total_area level as before but
    # triggers much earlier as the fly approaches.
    VIS_PROX_TURN_GAIN = 4
    VIS_PROX_REF = 0.008

    # "Stop & steer around" behavior when obstacle is frontal/large.
    # Lower thresholds → triggered earlier (at medium range).
    # Low STOP_SPEED + high STOP_TURN_BOOST → near-rotation-in-place.
    VIS_STOP_ENABLE = True
    VIS_STOP_CENTER_ON = 0.008
    VIS_STOP_CENTER_OFF = 0.003
    VIS_STOP_TOTAL_ON = 0.008
    VIS_STOP_TOTAL_OFF = 0.003
    VIS_STOP_SPEED = 0.08        # nearly stops; body can rotate without moving forward
    VIS_STOP_LATCH_DECISIONS = 30
    VIS_STOP_TURN_BOOST = 6.0   # clipped to VIS_TURN_MAX → always full-authority turn

    # --- Grass-spike detection pipeline ----------------------------------
    # Stages inside the ROI:
    #   1. GREEN MASK — chroma `(g−r), (g−b), g_min` → terrain + herbe.
    #   2. HORIZON — `first_green[col]` lissé par un percentile élevé → sol.
    #   3. COULEUR DE RÉFÉRENCE — médiane RGB sur les pixels verts **au-dessus**
    #      de la ligne d’horizon uniquement (ce qui dépasse le sol).
    #   4. MATCH ADAPTATIF — pixels verts proches de cette référence (L∞ ≤ tol).
    #   5. TRIANGLE — pour chaque colonne où le pic dépasse nettement le sol,
    #      bande verticale `first_green … horizon` ∩ match couleur → pic entier.
    VIS_TIP_MIN_HEIGHT = 4
    # Half-width (cols) of the horizon-smoothing window.  Window length is
    # 2*HALF + 1.  Should be wider than any single blade silhouette base.
    VIS_TIP_HORIZON_HALF = 40
    # Robust statistic of `first_green` inside the window.  Spikes pull
    # `first_green` upward (smaller row index); a HIGH percentile (closer to
    # max) therefore tracks the GROUND and ignores the spikes.  75 keeps a
    # safety margin against ground-row anti-alias noise.
    VIS_TIP_HORIZON_PERCENTILE = 75.0
    # Apex must beat the local horizon by at least this many pixels to count
    # as a candidate (filters horizon noise / anti-alias stair-stepping).
    VIS_TIP_APEX_MARGIN = 3
    # Max abs. RGB deviation from the per-frame median reference colour
    # sampled above the horizon (L∞ norm).  Larger → fuller triangles in
    # shadow / anti-alias; smaller → stricter match to the protruding tip.
    VIS_SPIKE_COLOR_TOL = 0.11

    # --- Blade proximity reflex (consumes the spike pixel count) ---------
    # The reflex stays unchanged in spirit: two speeds, FAR for gentle
    # anticipation, NEAR for a hard pivot.  Only the upstream signal — the
    # spike pixel count — has been rewritten above.
    VIS_BLADE_ENABLE = True
    VIS_BLADE_FAR_THRESH = 50
    VIS_BLADE_NEAR_THRESH = 250
    VIS_BLADE_LATCH_DECISIONS = 25
    VIS_BLADE_SPEED = 0.08

    # Wide-arc clearance: kicks in earlier than STOP and KEEPS the fly
    # turning while it is moving, so a fly-sized body actually clears the
    # blade laterally instead of brushing past it.
    VIS_WIDE_ENABLE = True
    VIS_WIDE_TOTAL_ON = 0.001   # trigger on ANY detectable blade
    VIS_WIDE_TOTAL_OFF = 0.0005
    VIS_WIDE_LATCH_DECISIONS = 60  # ~3 s of committed turning
    VIS_WIDE_TURN_GAIN = 4.0
    VIS_WIDE_SPEED_CENTER_GAIN = 1.2
    VIS_WIDE_SPEED_MIN = 0.25
    VIS_WIDE_AVOID_BLEND = 0.8

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
    # Higher = longer memory. Bumped from 0.90 so that, once we commit to
    # "go right around this blade", we don't flip-flop when the blade
    # leaves the centre band briefly.
    VIS_LAST_DIR_DECAY = 0.95
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
        # Wide-arc clearance latch (keeps a steady turn around close blades).
        self._wide_left = 0
        # Blade proximity reflex (pixel-count threshold).
        self._blade_left = 0
        self._blade_left_px = 0   # cached pixel counts for debug
        self._blade_right_px = 0
        # Color-based perception (Level 2+).
        self._vis_grass_left = 0.0
        self._vis_grass_right = 0.0
        self._vis_grass_center = 0.0
        self._vis_dragonfly_area = 0.0
        self._vis_dragonfly_x = 0.0
        self._dragonfly_left = 0
        # Latest debug-overlay frame (filled by `compute_vision_debug_overlay`).
        self._vis_debug_overlay = None
        self._vis_spike_roi_left = None   # last ROI spike mask, left eye (bool)
        self._vis_spike_roi_right = None  # idem right eye
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
            self._wide_left = 0
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
                vision_turn_bias, danger = self._vision_avoid_bias_and_danger(sim)

                # Apply visual speed modulation (set inside vision functions).
                # Do NOT allow vision to flip a strong goal-directed steering
                # signal UNLESS a reflex is already latched -- in that case the
                # obstacle takes absolute priority over the olfactory target.
                reflex_latched = (
                    self._bump_left > 0
                    or self._blade_left > 0
                    or self._loom_left > 0
                    or self._stop_left > 0
                    or self._wide_left > 0
                )
                if (
                    not reflex_latched
                    and abs(target_bias) >= self.VIS_TARGET_PROTECT
                    and np.sign(bias) != 0
                ):
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
                        f"blade=({self._blade_left_px},{self._blade_right_px}) "
                        f"bump={bool(self._bump_left>0)} jam={bool(self._jam_left>0)} "
                        f"loom={bool(self._loom_left>0)} stop={bool(self._stop_left>0)} "
                        f"wide={bool(self._wide_left>0)} blade_latch={bool(self._blade_left>0)}",
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
        # (which already encodes target_bias + vision turn, with the
        # VIS_TARGET_PROTECT check bypassed so vision always dominates).
        # WIDE is included: the fly pivots sharply at any blade detection,
        # not just when STOP/BUMP/LOOM fire.
        reflex_active = (
            self._enable_grass
            and (
                self._bump_left > 0
                or self._blade_left > 0
                or self._loom_left > 0
                or self._stop_left > 0
                or self._wide_left > 0
            )
        )
        if reflex_active:
            # Prefer smooth_bias direction (vision turn already dominates
            # because VIS_TARGET_PROTECT is bypassed when reflex is latched).
            # Fall back to _vis_last_dir when smooth_bias is near zero (rare
            # case where olfaction and vision cancel exactly).
            if abs(self._smooth_bias) > 1e-3:
                pivot_dir = 1.0 if self._smooth_bias > 0.0 else -1.0
            else:
                pivot_dir = float(np.sign(self._vis_last_dir)) if abs(self._vis_last_dir) > 1e-6 else 1.0
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

    # ------------------------------------------------------------------
    # Colour masks. Factored out of `_vision_avoid_bias_and_danger` so the
    # debug overlay can reuse the *exact* same logic and never drift from
    # what the controller actually sees.
    def _compute_green_mask(self, roi_rgb: np.ndarray) -> np.ndarray:
        """Pure-green chroma mask over the ROI (terrain + spikes)."""
        if not self.VIS_COLOR_ENABLE:
            return np.zeros(roi_rgb.shape[:2], dtype=bool)
        r = roi_rgb[..., 0]
        g = roi_rgb[..., 1]
        b = roi_rgb[..., 2]
        d = float(self.VIS_GRASS_GREEN_DELTA)
        gmin = float(self.VIS_GRASS_GREEN_MIN)
        return ((g - r) > d) & ((g - b) > d) & (g > gmin)

    # Backwards-compatible alias for the few legacy callers that still ask
    # for a "grass mask".  All grass pixels are simply pixels of the green
    # mask: terrain *and* blades.  The downstream code that needed a
    # spike-only signal now consumes the tip profile instead.
    def _compute_grass_mask(self, roi_rgb: np.ndarray) -> np.ndarray:
        return self._compute_green_mask(roi_rgb)

    def _compute_dragonfly_mask(self, roi_rgb: np.ndarray) -> np.ndarray:
        """Bool mask of saturated-red pixels (dragonfly eyes / head)."""
        if not self.VIS_COLOR_ENABLE:
            return np.zeros(roi_rgb.shape[:2], dtype=bool)
        r = roi_rgb[..., 0]
        g = roi_rgb[..., 1]
        b = roi_rgb[..., 2]
        d = float(self.VIS_DF_RED_DELTA)
        rmin = float(self.VIS_DF_RED_MIN)
        return ((r - g) > d) & ((r - b) > d) & (r > rmin)

    # ------------------------------------------------------------------
    # Grass-spike image processing — built from scratch.
    #
    # Stages:
    # ------------------------------------------------------------------
    # Grass-spike image processing.
    #
    # Output (per eye):
    #   tips[col]       — nombre de pixels du pic dans la colonne
    #   apex_rows[col]  — ligne du sommet (premier pixel du masque), ou h_roi
    #   horizon[col]    — ligne d’horizon (sol) lissée
    #   spike_full      — masque booléen (h_roi, w_roi) du triangle / pic entier
    # ------------------------------------------------------------------
    def _compute_tip_profile(
        self, eye_img01: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h_full, w_full = eye_img01.shape[:2]
        r0 = int(float(self.VIS_ROI_R0) * h_full)
        r1 = int(float(self.VIS_ROI_R1) * h_full)
        c0 = int(float(self.VIS_ROI_C0) * w_full)
        c1 = int(float(self.VIS_ROI_C1) * w_full)
        roi = eye_img01[r0:r1, c0:c1, :]

        h, w = roi.shape[:2]
        empty_mask = np.zeros((h, w), dtype=bool)
        empty = (
            np.zeros(w, np.float32),
            np.full(w, float(h), np.float32),
            np.full(w, float(h), np.float32),
            empty_mask,
        )
        if h < 2 or w < 1 or not self.VIS_COLOR_ENABLE:
            return empty

        # --- Stage 1: green mask (terrain + spikes) ---
        green = self._compute_green_mask(roi)
        if not green.any():
            return empty

        # --- Stage 2a: per-column topmost-green row ---
        has_green = green.any(axis=0)
        first_green = np.where(
            has_green, np.argmax(green, axis=0).astype(np.float32), float(h)
        )

        # --- Stage 2b: horizon = rolling HIGH percentile of first_green ---
        half = int(max(1, self.VIS_TIP_HORIZON_HALF))
        win = 2 * half + 1
        pct = float(self.VIS_TIP_HORIZON_PERCENTILE)
        if w >= win:
            padded = np.pad(first_green, (half, half), mode="edge")
            from numpy.lib.stride_tricks import sliding_window_view
            sw = sliding_window_view(padded, win)
            horizon = np.percentile(sw, pct, axis=1).astype(np.float32)
        else:
            horizon = np.full(
                w, float(np.percentile(first_green, pct)), dtype=np.float32
            )

        rr = np.arange(h, dtype=np.float32)[:, None]
        above_h = rr < horizon[None, :]

        # --- Stage 3: référence couleur = médiane RGB au-dessus du sol ---
        seed = green & above_h
        if seed.any():
            ref_r = float(np.median(roi[..., 0][seed]))
            ref_g = float(np.median(roi[..., 1][seed]))
            ref_b = float(np.median(roi[..., 2][seed]))
        else:
            ref_r = float(np.median(roi[..., 0][green]))
            ref_g = float(np.median(roi[..., 1][green]))
            ref_b = float(np.median(roi[..., 2][green]))

        tol = float(self.VIS_SPIKE_COLOR_TOL)
        dr = np.abs(roi[..., 0] - ref_r)
        dg = np.abs(roi[..., 1] - ref_g)
        db = np.abs(roi[..., 2] - ref_b)
        color_match = green & (np.maximum(np.maximum(dr, dg), db) <= tol)

        # --- Stage 4: slab verticale par colonne + pic détecté (hauteur > seuil) ---
        apex_margin = float(self.VIS_TIP_APEX_MARGIN)
        min_height = float(self.VIS_TIP_MIN_HEIGHT)
        height_above = horizon - first_green
        geom_ok = has_green & (height_above >= max(min_height, apex_margin))

        slab = geom_ok[None, :] & (rr >= first_green[None, :]) & (
            rr <= horizon[None, :]
        )
        spike_full = color_match & slab

        tips = spike_full.sum(axis=0).astype(np.float32)
        apex_rows = np.where(
            spike_full.any(axis=0),
            np.argmax(spike_full, axis=0).astype(np.float32),
            float(h),
        )

        return tips, apex_rows, horizon, spike_full

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

        def extract_visual_features(left_img01: np.ndarray, right_img01: np.ndarray) -> dict:
            # ----------------------------------------------------------------
            # Spike detection: horizon → median RGB above horizon → adaptive
            # colour match → full wedge mask per column between apex and horizon.
            left_tips, _l_apex, _l_horizon, l_spike = self._compute_tip_profile(
                left_img01
            )
            right_tips, _r_apex, _r_horizon, r_spike = self._compute_tip_profile(
                right_img01
            )
            self._vis_spike_roi_left = l_spike
            self._vis_spike_roi_right = r_spike

            # ROI geometry for normalisation (tips are in absolute pixels;
            # we normalise to fractions 0-1 for threshold compatibility).
            h_full = left_img01.shape[0]
            h_roi  = int(self.VIS_ROI_R1 * h_full) - int(self.VIS_ROI_R0 * h_full)
            w_roi  = max(1, int(len(left_tips)))
            roi_area = max(1, h_roi * w_roi)

            def _tip_area_x(tips: np.ndarray) -> tuple[float, float]:
                """Normalised area fraction + weighted centroid X in [-1,1]."""
                total = float(tips.sum())
                area  = total / roi_area
                if total < 1e-9:
                    return area, 0.0
                xs     = np.linspace(-1.0, 1.0, len(tips), dtype=np.float32)
                x_mean = float((tips * xs).sum() / total)
                return area, x_mean

            left_area,  left_x  = _tip_area_x(left_tips)
            right_area, right_x = _tip_area_x(right_tips)

            # Centre columns (straight-ahead portion of each eye)
            cL = int(self.VIS_CENTER_C0 * w_roi)
            cR = max(cL + 1, int(self.VIS_CENTER_C1 * w_roi))
            center_area_raw = 0.5 * (
                float(left_tips[cL:cR].sum())  / max(1, h_roi * (cR - cL))
                + float(right_tips[cL:cR].sum()) / max(1, h_roi * (cR - cL))
            )

            total_area_raw = 0.5 * (left_area + right_area)

            # EMA smoothing (same structure as before for threshold compat.)
            prev = float(self._vis_total_area)
            self._vis_total_area = float(
                self.VIS_EMA * prev + (1.0 - self.VIS_EMA) * total_area_raw
            )
            d_raw = float(self._vis_total_area - self._vis_total_area_prev)
            self._vis_total_area_prev = float(self._vis_total_area)
            self._vis_d_total_area = float(
                self.VIS_D_EMA * self._vis_d_total_area
                + (1.0 - self.VIS_D_EMA) * d_raw
            )
            self._vis_left_area = float(
                self.VIS_EMA * self._vis_left_area + (1.0 - self.VIS_EMA) * left_area
            )
            self._vis_right_area = float(
                self.VIS_EMA * self._vis_right_area + (1.0 - self.VIS_EMA) * right_area
            )
            self._vis_center_area = float(
                self.VIS_EMA * self._vis_center_area
                + (1.0 - self.VIS_EMA) * center_area_raw
            )

            # ---- Dragonfly: colour-based (kept, it is the only red object) ----
            lroi = _roi(left_img01)
            rroi = _roi(right_img01)
            l_drag = self._compute_dragonfly_mask(lroi).astype(np.float32)
            r_drag = self._compute_dragonfly_mask(rroi).astype(np.float32)

            def _df_area_x(mask2d: np.ndarray) -> tuple[float, float]:
                area = float(mask2d.mean())
                if area <= 1e-9:
                    return 0.0, 0.0
                cols   = mask2d.mean(axis=0)
                xs     = np.linspace(-1.0, 1.0, cols.shape[0], dtype=np.float32)
                x_mean = float((cols * xs).sum() / max(1e-9, float(cols.sum())))
                return area, x_mean

            df_left_area,  df_left_x  = _df_area_x(l_drag)
            df_right_area, df_right_x = _df_area_x(r_drag)
            df_total = 0.5 * (df_left_area + df_right_area)
            df_w     = df_left_area + df_right_area
            df_x_raw = (
                df_left_x * df_left_area + df_right_x * df_right_area
            ) / max(1e-9, df_w)
            self._vis_dragonfly_area = float(
                self.VIS_DF_EMA * self._vis_dragonfly_area
                + (1.0 - self.VIS_DF_EMA) * df_total
            )
            self._vis_dragonfly_x = float(
                self.VIS_DF_EMA * self._vis_dragonfly_x
                + (1.0 - self.VIS_DF_EMA) * df_x_raw
            )

            # ---- Blade proximity counts (raw tip sums, no EMA) ----
            # Using the full tip profile (all tip pixels, both below and above
            # the sky zone) gives a richer proximity signal than the old
            # colour-based sky-zone-only count.
            left_blade_px  = int(left_tips.sum())
            right_blade_px = int(right_tips.sum())
            self._blade_left_px  = left_blade_px
            self._blade_right_px = right_blade_px

            return {
                "left_area":      float(self._vis_left_area),
                "right_area":     float(self._vis_right_area),
                "raw_left_area":  left_area,
                "raw_right_area": right_area,
                "left_x":         left_x,
                "right_x":        right_x,
                "center_area":    float(self._vis_center_area),
                "total_area":     float(self._vis_total_area),
                "d_total_area":   float(self._vis_d_total_area),
                "dragonfly_area": float(self._vis_dragonfly_area),
                "dragonfly_x":    float(self._vis_dragonfly_x),
                "left_blade_px":  left_blade_px,
                "right_blade_px": right_blade_px,
            }

        def compute_obstacle_avoidance(feat: dict) -> tuple[float, float, float]:
            # Use RAW (unsmoothed) lr for the turn direction so the fly
            # reacts within the same decision step it first sees the blade.
            # The EMA-smoothed areas are noisy only when the scene changes
            # abruptly, but the grass mask is already structurally filtered
            # so a single raw frame is reliable enough for direction.
            raw_lr = float(feat["raw_left_area"] - feat["raw_right_area"])
            ema_lr = float(feat["left_area"] - feat["right_area"])
            # Take the max-magnitude signal: raw for the freshest direction
            # bias, but keep the EMA in case raw momentarily drops to zero.
            lr = raw_lr if abs(raw_lr) >= abs(ema_lr) else ema_lr
            center = float(feat["center_area"])
            total = float(feat["total_area"])
            turn = float(np.clip(self.VIS_TURN_GAIN * lr, -self.VIS_TURN_MAX, self.VIS_TURN_MAX))
            # 1) center boost: obstacle dead-ahead -> turn harder
            turn *= float(1.0 + self.VIS_CENTER_TURN_GAIN * np.clip(center / 0.12, 0.0, 1.0))
            # 2) proximity boost: more total area -> closer obstacle -> turn even
            #    harder, regardless of whether it is centred. This is what makes
            #    the fly trace a wider arc instead of grazing the blade.
            turn *= float(
                1.0
                + self.VIS_PROX_TURN_GAIN
                * np.clip(total / max(1e-6, float(self.VIS_PROX_REF)), 0.0, 1.0)
            )
            turn = float(np.clip(turn, -self.VIS_TURN_MAX, self.VIS_TURN_MAX))
            speed_scale = float(1.0 - self.VIS_SPEED_CENTER_GAIN * np.clip(center / 0.18, 0.0, 1.0))
            speed_scale = float(np.clip(speed_scale, self.VIS_SPEED_MIN, 1.0))
            danger = float(np.clip(total + 0.8 * center, 0.0, 1.0))
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

            # Use raw lr for direction (fastest possible signal).
            raw_lr = float(feat["raw_left_area"] - feat["raw_right_area"])
            ema_lr = float(feat["left_area"] - feat["right_area"])
            lr = raw_lr if abs(raw_lr) >= abs(ema_lr) else ema_lr
            # choose a stable direction if symmetric
            if abs(lr) < 1e-4 and self.VIS_DIRECTION_MEMORY:
                dir_sign = float(self._vis_last_dir)
            else:
                dir_sign = 1.0 if lr > 0 else -1.0  # obstacle left => turn right (+)

            # STOP_TURN_BOOST >> VIS_TURN_MAX so this is always a full-authority
            # pivot regardless of numeric value: the clip does the capping.
            turn = float(np.clip(self.VIS_STOP_TURN_BOOST * dir_sign, -self.VIS_TURN_MAX, self.VIS_TURN_MAX))
            speed = float(self.VIS_STOP_SPEED)
            return True, turn, speed

        def compute_wide_clearance(feat: dict) -> tuple[bool, float, float]:
            """Maintain a steady turn while ANY blade-sized obstacle remains in the ROI.

            The fly is large compared to a single grass blade. After the
            continuous-avoidance turn pushes the blade off-centre, the
            classic logic relaxes the turn -- and the body still grazes
            the blade. WIDE mode stays committed to the chosen direction
            until the blade has cleared the entire ROI, producing a
            generous arc instead of a knife-edge pass.
            """
            if not self.VIS_WIDE_ENABLE:
                return False, 0.0, 1.0
            total = float(feat["total_area"])
            center = float(feat["center_area"])

            on = total >= float(self.VIS_WIDE_TOTAL_ON)
            off = total <= float(self.VIS_WIDE_TOTAL_OFF)

            if self._wide_left > 0:
                self._wide_left -= 1
            if on:
                self._wide_left = max(
                    self._wide_left, int(self.VIS_WIDE_LATCH_DECISIONS)
                )
            if off and self._wide_left <= 0:
                self._wide_left = 0

            if self._wide_left <= 0:
                return False, 0.0, 1.0

            raw_lr = float(feat["raw_left_area"] - feat["raw_right_area"])
            ema_lr = float(feat["left_area"] - feat["right_area"])
            lr = raw_lr if abs(raw_lr) >= abs(ema_lr) else ema_lr
            if abs(lr) < 1e-4 and self.VIS_DIRECTION_MEMORY:
                dir_sign = float(self._vis_last_dir)
            else:
                dir_sign = 1.0 if lr > 0 else -1.0
            turn = float(np.clip(
                self.VIS_WIDE_TURN_GAIN * dir_sign,
                -self.VIS_TURN_MAX,
                self.VIS_TURN_MAX,
            ))
            speed = float(np.clip(
                1.0
                - self.VIS_WIDE_SPEED_CENTER_GAIN
                * np.clip(center / 0.05, 0.0, 1.0),
                self.VIS_WIDE_SPEED_MIN,
                1.0,
            ))
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

        def compute_blade_proximity_reflex(feat: dict) -> tuple[bool, float, float]:
            """Two-speed blade avoidance based on sky-zone pixel count.

            The sky-zone count (above horizon, thin-filtered) encodes proximity:
              count > FAR_THRESH  → blade visible but distant → gentle pre-turn
                                    (anticipation: start steering early)
              count > NEAR_THRESH → blade close / large → full-authority pivot
                                    (near-stop + max turn to clear)

            Direction: whichever eye sees more blade pixels → danger on that
            side → turn away from it (+1 = turn right, -1 = turn left).
            """
            if not self.VIS_BLADE_ENABLE:
                return False, 0.0, 1.0

            left_px  = int(feat.get("left_blade_px", 0))
            right_px = int(feat.get("right_blade_px", 0))
            max_px   = max(left_px, right_px)

            near_thresh = int(self.VIS_BLADE_NEAR_THRESH)
            far_thresh  = int(self.VIS_BLADE_FAR_THRESH)

            # Trigger on FAR_THRESH for early detection; clear below half that.
            on  = max_px >= far_thresh
            off = max_px < far_thresh // 2

            if self._blade_left > 0:
                self._blade_left -= 1
            if on:
                # Stay latched; NEAR gets a longer commitment than FAR.
                latch = int(self.VIS_BLADE_LATCH_DECISIONS) if max_px >= near_thresh else max(8, int(self.VIS_BLADE_LATCH_DECISIONS) // 3)
                self._blade_left = max(self._blade_left, latch)
            if off and self._blade_left <= 0:
                self._blade_left = 0

            if self._blade_left <= 0:
                return False, 0.0, 1.0

            # Turn away from the heavier eye; use memory if symmetric.
            diff = left_px - right_px
            if abs(diff) < 10 and self.VIS_DIRECTION_MEMORY:
                dir_sign = float(np.sign(self._vis_last_dir)) if abs(self._vis_last_dir) > 1e-6 else 1.0
            else:
                dir_sign = 1.0 if diff > 0 else -1.0  # more pixels on left → turn right (+)

            if max_px >= near_thresh:
                # CLOSE: hard pivot, nearly stop.
                turn  = float(np.clip(dir_sign * self.VIS_TURN_MAX, -self.VIS_TURN_MAX, self.VIS_TURN_MAX))
                speed = float(self.VIS_BLADE_SPEED)
            else:
                # FAR: gentle anticipatory turn, keep moving.
                gentle_max = 2.0
                turn  = float(np.clip(dir_sign * gentle_max, -gentle_max, gentle_max))
                speed = 0.50
            return True, turn, speed

        def compute_dragonfly_reflex(feat: dict) -> tuple[bool, float, float]:
            """Hard turn away whenever a saturated red blob is in sight.

            Distinct from `compute_looming_reflex`: this fires on COLOUR
            (the dragonfly eyes are the only red object in the scene), not
            on edge growth, so it triggers earlier and is robust against
            slow-approaching drones that would not loom much in a single
            decision step.
            """
            if not self.VIS_COLOR_ENABLE:
                return False, 0.0, 1.0
            df_area = float(feat.get("dragonfly_area", 0.0))
            df_x = float(feat.get("dragonfly_x", 0.0))

            on = df_area >= float(self.VIS_DF_AREA_ON)
            off = df_area <= float(self.VIS_DF_AREA_OFF)

            if self._dragonfly_left > 0:
                self._dragonfly_left -= 1
            if on:
                self._dragonfly_left = max(
                    self._dragonfly_left, int(self.VIS_DF_LATCH_DECISIONS)
                )
            if off and self._dragonfly_left <= 0:
                self._dragonfly_left = 0

            if self._dragonfly_left <= 0:
                return False, 0.0, 1.0

            # df_x in [-1, 1]: -1 = full left, +1 = full right.
            # Steer AWAY from the dragonfly: if it is on the left (df_x<0),
            # we turn right (+turn).
            if abs(df_x) < 1e-3 and self.VIS_DIRECTION_MEMORY:
                dir_sign = float(self._vis_last_dir)
            else:
                dir_sign = -1.0 if df_x < 0 else 1.0
            turn = float(np.clip(
                self.VIS_DF_TURN_GAIN * dir_sign,
                -self.VIS_DF_TURN_MAX,
                self.VIS_DF_TURN_MAX,
            ))
            speed = float(self.VIS_DF_SPEED_MIN)
            return True, turn, speed

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
        blade_active, blade_turn, blade_speed = compute_blade_proximity_reflex(feat)
        df_active, df_turn, df_speed = compute_dragonfly_reflex(feat)
        looming_active, loom_turn, loom_speed = compute_looming_reflex(feat)
        stop_active, stop_turn, stop_speed = compute_stop_and_go_around(feat)
        wide_active, wide_turn, wide_speed = compute_wide_clearance(feat)
        avoid_turn, avoid_speed, danger = compute_obstacle_avoidance(feat)

        # Priority order:
        #   1. BUMP: physical contact -> emergency stop
        #   2. BLADE: pixel-count threshold -> full pivot away from blade
        #   3. DRAGONFLY (color): saturated red blob -> peel away early
        #   4. LOOM: rapid edge growth -> dodge
        #   5. STOP: large frontal blob -> commit to a direction
        #   6. WIDE: any blade in ROI -> keep arcing around it
        #   7. AVOID: continuous obstacle steering
        if bump_active:
            turn = float(bump_turn)
            self._vis_speed_scale = float(bump_speed)
        elif blade_active:
            turn = float(blade_turn)
            self._vis_speed_scale = float(blade_speed)
        elif df_active:
            turn = float(df_turn)
            self._vis_speed_scale = float(df_speed)
            # boost danger so downstream code (target steer blend, debug) reacts
            danger = float(np.clip(max(danger, 0.6 + float(feat["dragonfly_area"])), 0.0, 1.0))
        elif looming_active:
            turn = float(loom_turn)
            self._vis_speed_scale = float(loom_speed)
        elif stop_active:
            turn = float(stop_turn)
            self._vis_speed_scale = float(stop_speed)
        elif wide_active:
            # Compose with the continuous avoidance signal so we keep the
            # proportional response (helps when two blades are visible at
            # once and `lr` swings).
            turn = float(np.clip(
                wide_turn + self.VIS_WIDE_AVOID_BLEND * avoid_turn,
                -self.VIS_TURN_MAX,
                self.VIS_TURN_MAX,
            ))
            self._vis_speed_scale = float(min(wide_speed, avoid_speed))
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
    def compute_vision_debug_overlay(
        self, sim: MiniprojectSimulation
    ) -> "np.ndarray | None":
        """Build an RGB debug image showing what the colour detectors see.

        For each eye, returns the raw fly view dimmed to ~50 % luminance,
        with three overlay layers in vivid colours:
          * green   -> pixels classified as 'grass / generic obstacle'
                        (i.e. either pure green in the upper ROI or a
                        strong edge / dark pixel)
          * red     -> pixels classified as 'dragonfly' (saturated red)
          * cyan    -> ROI rectangle outline, so we can see what the
                        controller actually looks at
        Output shape: `(H, 2*W, 3)` uint8, suitable to vstack with the
        scene-camera frame in the interactive viewer.

        This method is *side-effect free* w.r.t. the FSM state: it does
        NOT update EMAs, danger, or any latch counter -- only `self._vis_debug_overlay`
        is set so external callers can read it back.
        """
        try:
            frames = sim.get_raw_vision(sim.fly.name)
        except Exception:
            return None
        if frames is None or len(frames) == 0:
            return None

        eye_imgs = []
        for img in frames[:2]:
            a = np.asarray(img)
            if a.ndim == 2:
                a = np.stack([a, a, a], axis=-1)
            if a.shape[-1] > 3:
                a = a[..., :3]
            if a.dtype != np.uint8:
                a = np.clip(a, 0, 255).astype(np.uint8)
            eye_imgs.append(a)
        # Pad to two eyes if only one was returned.
        if len(eye_imgs) == 1:
            eye_imgs.append(eye_imgs[0])

        out_eyes: list[np.ndarray] = []
        for raw in eye_imgs:
            h, w = raw.shape[:2]
            r0, r1 = int(h * self.VIS_ROI_R0), int(h * self.VIS_ROI_R1)
            c0, c1 = int(w * self.VIS_ROI_C0), int(w * self.VIS_ROI_C1)

            img01 = raw.astype(np.float32) / 255.0

            # Compute spike mask (same pipeline as the controller).
            _, _, horizon, spike_roi = self._compute_tip_profile(img01)

            # Dragonfly colour mask (still colour-based).
            roi = img01[r0:r1, c0:c1, :]
            df_mask = self._compute_dragonfly_mask(roi)

            # Compose overlay: darken the original, then paint on top.
            base = (raw.astype(np.float32) * 0.45).astype(np.uint8)
            overlay = base.copy()
            roi_view = overlay[r0:r1, c0:c1, :]  # mutable window into overlay

            h_roi = r1 - r0
            w_roi = c1 - c0

            # ---- 1. Full spike shape (adaptive colour match inside apex→horizon) ----
            if spike_roi.shape == (h_roi, w_roi):
                roi_view[spike_roi] = np.array([20, 255, 40], dtype=np.uint8)

            # ---- 2. Apex markers (yellow): top row of spike mask per column ----
            for col_i in range(w_roi):
                if not spike_roi[:, col_i].any():
                    continue
                r_int = int(np.argmax(spike_roi[:, col_i]))
                roi_view[r_int, col_i, :] = np.array([255, 220, 0], dtype=np.uint8)

            # ---- 3. Horizon line (white, per column) ----
            for col_i in range(w_roi):
                hr = int(np.clip(horizon[col_i], 0, h_roi - 1))
                roi_view[hr, col_i, :] = np.array([255, 255, 255], dtype=np.uint8)

            # ---- 4. Dragonfly red eyes: drawn last so they win ----
            if df_mask.shape == roi_view.shape[:2]:
                roi_view[df_mask] = np.array([255, 30, 30], dtype=np.uint8)

            # ---- 5. ROI rectangle outline (cyan) ----
            cyan = np.array([0, 220, 220], dtype=np.uint8)
            overlay[r0 : r0 + 1, c0:c1, :] = cyan
            overlay[r1 - 1 : r1, c0:c1, :] = cyan
            overlay[r0:r1, c0 : c0 + 1, :] = cyan
            overlay[r0:r1, c1 - 1 : c1, :] = cyan

            out_eyes.append(overlay)

        result = np.concatenate(out_eyes, axis=1).astype(np.uint8)
        self._vis_debug_overlay = result
        return result
