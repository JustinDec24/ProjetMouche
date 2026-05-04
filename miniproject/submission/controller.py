import os
from dataclasses import dataclass

import numpy as np

from miniproject.simulation import MiniprojectSimulation

try:
    from scipy import ndimage as _SCIPY_NDIMAGE
except ImportError:
    _SCIPY_NDIMAGE = None

_SPIKE_CONN = np.ones((3, 3), dtype=np.int32)


@dataclass
class _VisFlags:
    """Flat container of visual reflex flags emitted by `_vision_step`."""

    bump: bool = False
    loom: bool = False
    dragonfly: bool = False


class Controller:
    """Cap banane + interrupt vision.

    Steering = `target_bias` (cap banane) ; vision agit en interruption :
    quand un pic obstrue la trajectoire, on bascule en AVOID (PIVOT
    frontal ou ARC latéral) jusqu'à ce que la voie soit dégagée.
    """

    # --- scheduling ---
    DECISION_INTERVAL_S = 0.025  # 25 ms = 40 decisions/s

    # --- olfaction (stop only) ---
    PALP_WEIGHT = 9
    ANTENNA_WEIGHT = 1
    EPS_ODOR = 1e-12
    # NOTE: In practice, the odor magnitude at ~8–10 units distance can already
    # exceed 2e-5 with the default arena parameters; keep this threshold higher
    # so the fly does not stop prematurely.
    STOP_ODOR_THRESHOLD = 5e-4
    STOP_DIST = 2.0

    # --- drives ---
    BASE_DRIVE_FAST = 2.40
    MAX_DRIVE = 2.80
    MAX_DRIVE_TERRAIN = 2.00
    MIN_DRIVE = 0.80
    MIN_DRIVE_TERRAIN = 0.60
    MIN_SIDE_DRIVE = 0.40
    MIN_SIDE_DRIVE_TERRAIN = 0.25
    TURN_MOD = 0.8
    # Couple différentiel plein en esquive (évite l'amortissement 0.8 + pente).
    AVOID_TURN_MOD = 1.0

    # --- target steering (cap banane) ---
    TARGET_STEER_GAIN = 3.0
    TARGET_STEER_GAIN_CLOSE = 6.0
    TARGET_STEER_CLOSE_DIST = 24.0
    TARGET_STEER_BIAS_SCALE = 0.25

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

    # --- anti-roll grip (any level) ---
    TILT_GRIP_ENABLE = True
    TILT_GRIP_ROLL_ON = 0.18
    TILT_GRIP_UPRIGHT_ON = 0.85

    # --- active roll compensation ---
    TILT_LEAN_ENABLE = True
    TILT_LEAN_ROLL_ON = 0.10
    TILT_LEAN_ROLL_FULL = 0.40
    TILT_LEAN_GAIN = 0.30
    TILT_LEAN_SIGN = +1.0

    # --- grip boost ---
    TERRAIN_GRIP_FORCE = 2.85  # plus d'adhérence sur collines (évite basculements → reset)
    WIND_GRIP_FORCE = 10.0    # adhesion > 1 pour pattes au sol lors de vent

    # --- orientation safety (Level 1+): resets renvoient la mouche au spawn — trop agressif = échecs L2 ---
    TERRAIN_UPRIGHT_TILT_WARN = 0.48  # sous ce seuil on accumule des ticks « très penché »
    TERRAIN_TILT_RESET_HOLD = 26  # nb de décisions consécutives avant request_reset (était 12 @ 0.55)
    TERRAIN_FLIP_WEAK_UPRIGHT = 0.12  # en dessous: fly vraiment couchée (était 0.2, trop sensible)
    TERRAIN_FLIP_RESET_HOLD = 34  # décisions consécutives avant sim.reset() flip (était 18)

    # --- vision (Level 2+) ---
    VISION_ENABLE = True
    VISION_USE_RAW = True
    VISION_DECISION_EVERY = 1

    # ROI vertical (commun aux deux yeux) — bande un peu plus haute, toujours centrée.
    VIS_ROI_R0 = 0.05
    VIS_ROI_R1 = 0.78
    # ROI horizontal : même couloir convergent nasal, mais plus large (temporal).
    VIS_ROI_C0_LEFT = 0.40
    VIS_ROI_C1_LEFT = 0.99
    VIS_ROI_C0_RIGHT = 0.01
    VIS_ROI_C1_RIGHT = 0.60
    # Bande « centre avant » dans chaque ROI (fractions [0,1]).
    VIS_CENTER_C0 = 0.30
    VIS_CENTER_C1 = 0.70

    # Colour perception.
    VIS_COLOR_ENABLE = True
    VIS_GRASS_GREEN_DELTA = 0.20
    VIS_GRASS_GREEN_MIN = 0.45

    # Dragonfly (red eyes) detector.
    VIS_DF_RED_DELTA = 0.20
    VIS_DF_RED_MIN = 0.40
    VIS_DF_AREA_ON = 0.0040
    VIS_DF_AREA_OFF = 0.0015
    VIS_DF_LATCH_DECISIONS = 10
    VIS_DF_EMA = 0.55

    VIS_DEBUG_OVERLAY = True

    VIS_EMA = 0.55
    VIS_D_EMA = 0.70

    # Saturation universelle des biais (le seul gain "turn" non AVOID).
    VIS_TURN_MAX = 3.5

    # Spike = grass pixel inside a horizontal sky/cloud–blade–sky/cloud segment.
    VIS_SKY_LUM_MAX = 0.42
    VIS_SKY_LUM_LOOSE = 0.58
    VIS_SKY_BLUE_MARGIN = 0.02
    VIS_CLOUD_LUM_MIN = 0.28
    VIS_CLOUD_LUM_MAX = 1.0
    VIS_CLOUD_RGB_SPREAD_MAX = 0.14

    VIS_SPIKE_MIN_AREA_PX = 1600  # ignore CC blobs smaller than this (≤0 = off)
    # Filtre de forme par blob : un vrai pic est haut+étroit, alors que les
    # silhouettes d'arbres lointains au-dessus de l'horizon sont courtes+étalées.
    # Un blob doit satisfaire les trois critères (aire, hauteur bbox, aspect H/W).
    VIS_SPIKE_MIN_HEIGHT_PX = 40  # hauteur minimale (en pixels) du bbox du blob
    VIS_SPIKE_MIN_ASPECT = 1.2   # ratio min hauteur/largeur du bbox (≤0 = off)

    # Looming reflex (priority).
    LOOM_ENABLE = True
    LOOM_CENTER_ON = 0.14
    LOOM_CENTER_OFF = 0.10
    LOOM_DAREA_ON = 0.030
    LOOM_DAREA_OFF = 0.018
    LOOM_LATCH_DECISIONS = 10

    # Contact-based bump reflex (highest priority).
    BUMP_ENABLE = True
    BUMP_CONTACT_ON = 4.0
    BUMP_CONTACT_OFF = 2.0
    BUMP_DCONTACT_ON = 6.0
    BUMP_DCONTACT_OFF = 2.5
    BUMP_LATCH_DECISIONS = 12

    # Jam reflex: if commanded to move but not translating (blocked), stop and pivot.
    JAM_ENABLE = True
    JAM_MOVE_EPS = 2.0e-2
    JAM_MIN_DRIVE = 0.70
    JAM_TRIGGER_DECISIONS = 4
    JAM_LATCH_DECISIONS = 14

    # Direction memory (anti flip-flop sur obstacles symétriques).
    VIS_DIRECTION_MEMORY = True
    VIS_LAST_DIR_DECAY = 0.95
    VIS_LR_EPS_PIXELS = 10
    VIS_LR_MIN_SPIKE_PIXELS = 4

    # --- AVOID FSM (interruption vision) ---
    # Trigger ON : pic au centre OU spike total > seuil OU bump/loom/dragonfly.
    AVOID_CENTER_ON = 0.008
    AVOID_TOTAL_ON = 0.008
    # Trigger OFF : center ET total redescendus, pendant N décisions.
    AVOID_CENTER_OFF = 0.003
    AVOID_TOTAL_OFF = 0.003
    AVOID_CLEAR_DECISIONS = 1
    AVOID_MIN_DURATION = 4
    # Hystérésis pour rafraîchir dodge_dir pendant l'AVOID (vision live, pas latch figé).
    # |lr_avoid| courant ≥ ce seuil pour ré-écrire dodge_dir si le signe diffère du latch.
    AVOID_LR_REFRESH = 0.003

    # Sous-mode PIVOT vs ARC : seuil très haut = PIVOT réservé aux obstacles frontaux proches.
    AVOID_PIVOT_CENTER = 0.050

    AVOID_PIVOT_TURN = 2.8
    # > MIN_DRIVE_TERRAIN/BASE pour garder une base réelle > plancher (couple pivot).
    AVOID_PIVOT_SPEED = 0.80
    AVOID_ARC_TURN = 0.5
    AVOID_ARC_SPEED = 0.97
    # Blend partiel vers la banane pendant ARC pour éviter les spirales.
    AVOID_ARC_TARGET_BLEND = 0.25

    # --- Initial alignment (pivot in place towards banana once) ---
    # Au tout début (et après chaque reset), la mouche pivote sur place vers la
    # banane avec une direction latchée. Sortie quand `|bearing| <= ALIGN_BEARING_OK`,
    # quand un réflexe vision impose d'avorter, ou par timeout.
    ALIGN_INITIAL_ENABLE = True
    ALIGN_BEARING_OK = 0.20      # ~11.5 deg
    ALIGN_MAX_DECISIONS = 60     # ~3 s safety cap

    # Après une épisode AVOID suffisamment longue (beaucoup d'étapes), on relance
    # ALIGN avec la banane pour réorienter la locomotion avant GO.
    ALIGN_AFTER_AVOID_ENABLE = False
    ALIGN_AFTER_AVOID_MIN_DECISIONS = 16   # réaligner seulement après un évitement assez long.

    # --- debugging ---
    DEBUG = False
    DEBUG_EVERY_DECISIONS = 4
    DEBUG_MAX_DECISIONS = 260
    # Prints détaillés vision / drives (True ou env CONTROLLER_VIS_TRACE=1).
    VIS_TRACE_VERBOSE = False
    # Seuil relatif symétrique vs pivot dans [VIS_TRACE command] (fraction du référencement amplitude).
    VIS_TRACE_DRIVE_EPS_FRAC = 0.015

    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        self._decision_every = int(self.DECISION_INTERVAL_S / sim.timestep)
        self._step_count = 0
        self._drives = np.array([1.0, 1.0])
        self._last_vision_turn_bias = 0.0
        self._last_vis_trace = None
        self._stopped = False
        self._enable_terrain = bool(getattr(sim, "enable_terrain", False))
        self._enable_grass = bool(getattr(sim, "enable_grass", False))
        self._enable_wind = bool(getattr(sim, "enable_wind", False))

        fly_segs = sim.fly.get_bodysegs_order()
        self._thorax_idx = next(
            i for i, s in enumerate(fly_segs) if s.name == "c_thorax"
        )
        body_ids = sim._internal_bodyids_by_fly[sim.fly.name]
        self._thorax_body_id = body_ids[self._thorax_idx]
        self._contact_body_ids = sim._internal_contact_body_segment_ids_by_fly[sim.fly.name]
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
        self._request_reset = False
        self._last_target_bearing = 0.0
        self._last_dist_to_banana = None
        self._debug_decisions = 0

        # Vision EMAs (kept for stability + debug trace).
        self._vis_left_area = 0.0
        self._vis_right_area = 0.0
        self._vis_center_area = 0.0
        self._vis_total_area = 0.0
        self._vis_total_area_prev = 0.0
        self._vis_d_total_area = 0.0
        self._vis_last_dir = 1.0

        # Reflex latches.
        self._loom_left = 0
        self._bump_left = 0
        self._bump_contact_ema = 0.0
        self._bump_contact_prev = 0.0
        self._dragonfly_left = 0
        self._vis_dragonfly_area = 0.0
        self._vis_dragonfly_x = 0.0

        # Spike LR pixel counts (used by reflexes & traces).
        self._blade_left_px = 0
        self._blade_right_px = 0

        # Latest debug-overlay frame.
        self._vis_debug_overlay = None
        self._vis_spike_roi_left = None
        self._vis_spike_roi_right = None

        # Jam latch.
        self._jam_left = 0
        self._jam_dir = 1

        # AVOID FSM state.
        self._avoid_left = 0
        self._avoid_min_left = 0
        self._avoid_clear = 0
        self._dodge_dir = 0.0
        self._avoid_session_ticks = 0
        self._avoid_sub = "GO"

        # Initial alignment state.
        self._align_done = False
        self._align_dir = 0.0     # +1.0 = pivot LEFT, -1.0 = pivot RIGHT
        self._align_left = 0      # safety counter (decisions remaining)
        self._align_saw_obstacle = False  # obstacle détecté devant pendant ALIGN

        try:
            self._banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
        except Exception:
            self._banana_xy = None

        # Lift fly at spawn to free any leg clipped into terrain mesh.
        try:
            import mujoco as _mj
            for _jnt in range(sim.mj_model.njnt):
                if sim.mj_model.jnt_type[_jnt] == _mj.mjtJoint.mjJNT_FREE:
                    _addr = sim.mj_model.jnt_qposadr[_jnt]
                    sim.mj_data.qpos[_addr + 2] += 2.0  # +2 mm above terrain
                    _mj.mj_forward(sim.mj_model, sim.mj_data)
                    break
        except Exception:
            pass

        # Precompute row mapping for hex->rect conversion (week4 exercise pattern).
        try:
            retina = sim.fly.retina
            id_rows = [np.unique(row) for row in retina.ommatidia_id_map]
            idx_rows = []
            max_len = 0
            for ids in id_rows:
                ids = ids[ids > 0]
                idx = (ids - 1).astype(int)
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
                # roll_lean > 0 -> right side down -> extend RIGHT legs ; <0 -> LEFT.
                if roll_lean > 0:
                    leg_indices = (3, 4, 5)
                else:
                    leg_indices = (0, 1, 2)
                for li in leg_indices:
                    base = li * 7
                    joint_angles[base + 3] += offset
                    joint_angles[base + 5] += 0.5 * offset

        # Reset request: now treated as level failure (stop outputs).
        if is_decision_step and self._request_reset:
            self._request_reset = False
            self._stopped = True
            return np.zeros_like(joint_angles), np.zeros_like(adhesion)

        # --- Orientation safety (terrain levels) ---
        if self._enable_terrain and is_decision_step:
            try:
                _, _, uprightness = self._get_orientation(sim)
            except Exception:
                uprightness = 1.0

            if uprightness < float(self.TERRAIN_UPRIGHT_TILT_WARN):
                self._tilt_decisions += 1
            else:
                self._tilt_decisions = 0

            if self._tilt_decisions >= int(self.TERRAIN_TILT_RESET_HOLD):
                self._stopped = True
                self._tilt_decisions = 0
                return np.zeros_like(joint_angles), np.zeros_like(adhesion)

        # --- Grip control (terrain only) ---
        if self._enable_terrain and is_decision_step:
            if self._escape_decisions_left > 0:
                adhesion = np.zeros_like(adhesion)
                return joint_angles, adhesion

            if uprightness < -0.4:
                self._flip_decisions = 999
            elif uprightness < float(self.TERRAIN_FLIP_WEAK_UPRIGHT):
                self._flip_decisions += 1
            else:
                self._flip_decisions = 0

            if self._flip_decisions >= int(self.TERRAIN_FLIP_RESET_HOLD):
                self._stopped = True
                return np.zeros_like(joint_angles), np.zeros_like(adhesion)

            grip_val = float(self.WIND_GRIP_FORCE) if self._enable_wind else float(self.TERRAIN_GRIP_FORCE)
            try:
                contact_forces = sim.mj_data.cfrc_ext[self._contact_body_ids, 3:]
                contact_mag = np.linalg.norm(contact_forces, axis=1)
                stance = contact_mag > self.CONTACT_THRESHOLD
                adhesion = np.zeros_like(adhesion)
                n = min(len(adhesion), len(stance))
                if self._enable_wind:
                    # Toutes les pattes reçoivent le grip max pendant le vent.
                    adhesion[:n] = grip_val
                else:
                    adhesion[:n] = stance[:n].astype(float) * grip_val
            except Exception:
                adhesion = np.full_like(adhesion, grip_val) if self._enable_wind else np.where(adhesion > 0.0, grip_val, adhesion)

            adhesion = np.clip(adhesion, 0.0, grip_val)

        # --- Anti-roll grip (every level) ---
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
        """Return (slope_forward, slope_lateral, slope_mag) from terrain normal."""
        world = getattr(sim, "world", None)
        get_normal = getattr(world, "get_normal", None)
        if not callable(get_normal):
            return 0.0, 0.0, 0.0

        try:
            thorax_xy = sim.get_body_positions(sim.fly.name)[self._thorax_idx, :2]
        except Exception:
            thorax_xy = sim.mj_data.xpos[self._thorax_body_id, :2]

        n = np.asarray(get_normal(float(thorax_xy[0]), float(thorax_xy[1])), dtype=float)
        if n.shape != (3,) or not np.isfinite(n).all() or abs(n[2]) < 1e-6:
            return 0.0, 0.0, 0.0

        grad = np.array([-n[0] / n[2], -n[1] / n[2]], dtype=float)
        slope_mag = float(np.linalg.norm(grad))

        heading_xy, lateral_xy = self._get_body_frame_xy(sim)
        slope_forward = float(np.dot(heading_xy, grad))
        slope_lateral = float(np.dot(lateral_xy, grad))
        return slope_forward, slope_lateral, slope_mag

    # ------------------------------------------------------------------
    def _compute_target_bias(
        self, sim, thorax_xy: np.ndarray
    ) -> tuple[float, float]:
        """Steering bias vers la banane à partir du bearing (cap banane).

        Retourne (bias, bearing). Bias positif = il faut tourner à gauche
        (bearing < 0 dans la convention du contrôleur).
        """
        if self._banana_xy is None:
            return 0.0, 0.0
        to_target = np.asarray(self._banana_xy, dtype=float) - np.asarray(
            thorax_xy, dtype=float
        )
        dist_tt = float(np.linalg.norm(to_target))
        if dist_tt < 1e-9:
            return 0.0, 0.0
        to_target /= dist_tt
        heading_xy, lateral_xy = self._get_body_frame_xy(sim)
        lateral_err = float(np.dot(lateral_xy, to_target))
        forward_err = float(np.dot(heading_xy, to_target))
        # Banane derrière et sur le côté : demi-tour court ; actif au-delà de 13 m (évite
        # la phase finale) — pas de borne haute pour ne pas couper les trajectoires longues.
        if (
            self._enable_grass
            and self._enable_terrain
            and dist_tt > 13.0
            and forward_err < -0.03
            and abs(lateral_err) > 0.28
        ):
            s = 1.0 if lateral_err >= 0.0 else -1.0
            bearing = float(s * (np.pi - 0.42))
        else:
            # L2 : petit forward_err>0 → atan2 trop sensible ; plancher sur fe.
            fe = float(forward_err)
            if self._enable_grass and self._enable_terrain and dist_tt < 30.0 and fe > 0.04:
                fe = max(fe, 0.11)
            bearing = float(np.arctan2(lateral_err, fe))
        g = (
            self.TARGET_STEER_GAIN_CLOSE
            if dist_tt < self.TARGET_STEER_CLOSE_DIST
            else self.TARGET_STEER_GAIN
        )
        bias = -float(self.TARGET_STEER_BIAS_SCALE) * g * bearing
        return bias, bearing

    # ------------------------------------------------------------------
    def _compute_drives(self, sim) -> np.ndarray:
        self._vis_trace_sep_emitted_this_drive = False
        if self._stopped:
            return np.array([0.0, 0.0])

        # Lazy banana XY (peut être indispo au __init__).
        if self._banana_xy is None:
            try:
                self._banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
            except Exception:
                self._banana_xy = None

        # Position thorax (utilisée partout).
        try:
            thorax_xy = sim.get_body_positions(sim.fly.name)[self._thorax_idx, :2]
        except Exception:
            thorax_xy = sim.mj_data.xpos[self._thorax_body_id, :2]
        thorax_xy = np.asarray(thorax_xy, dtype=float)

        dist_to_banana = None
        if self._banana_xy is not None:
            dist_to_banana = float(np.linalg.norm(thorax_xy - self._banana_xy))
            self._last_dist_to_banana = dist_to_banana
            if dist_to_banana <= float(self.STOP_DIST):
                self._stopped = True
                return np.array([0.0, 0.0])

        # ---- stuck detection + escape maneuver (terrain peut piéger la mouche) ----
        moved = float("inf")
        if self._last_xy is not None:
            moved = float(np.linalg.norm(thorax_xy - self._last_xy))
            if moved < 5e-3:
                self._stuck_decisions += 1
            else:
                self._stuck_decisions = 0
        self._last_xy = thorax_xy

        if self._escape_decisions_left > 0:
            self._escape_decisions_left -= 1
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

        # ---- Jam reflex (any level) ----
        if self.JAM_ENABLE:
            max_drive_cmd = float(np.max(self._drives)) if getattr(self, "_drives", None) is not None else 0.0
            if moved < float(self.JAM_MOVE_EPS) and max_drive_cmd >= float(self.JAM_MIN_DRIVE):
                self._stuck_decisions += 1
            else:
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

        # ---- Stop sur banane (olfaction linéaire) ----
        odor_lin = sim.get_olfaction(sim.fly.name)  # shape (4, 1)
        lp, rp, la, ra = odor_lin[:, 0]
        odor_l = self.PALP_WEIGHT * float(lp) + self.ANTENNA_WEIGHT * float(la)
        odor_r = self.PALP_WEIGHT * float(rp) + self.ANTENNA_WEIGHT * float(ra)
        mean_odor = 0.5 * (odor_l + odor_r)
        if mean_odor > self.STOP_ODOR_THRESHOLD:
            self._stopped = True
            return np.array([0.0, 0.0])

        # ---- Cap banane ----
        target_bias, bearing = self._compute_target_bias(sim, thorax_xy)
        self._last_target_bearing = bearing

        # ---- Vision : un seul appel (feat + flags). ----
        feat: dict | None = None
        vis = _VisFlags()
        if self.VISION_ENABLE and self._enable_grass:
            if self._vision_step_count % self.VISION_DECISION_EVERY == 0:
                feat, vis = self._vision_step(sim)
            self._vision_step_count += 1

        # ---- Initial alignment: pivot in place to face the banana ----
        # Début de niveau / reset, ou réalignement après évitement prolongé (voir
        # `_schedule_realign_after_long_avoid`). Avorter si bump/loom/dragonfly.
        if (
            self.ALIGN_INITIAL_ENABLE
            and not self._align_done
            and self._banana_xy is not None
        ):
            if self._align_dir == 0.0:
                # Latch direction au premier tick : bearing > 0 = banane à gauche.
                self._align_dir = 1.0 if bearing > 0.0 else -1.0
                self._align_left = int(self.ALIGN_MAX_DECISIONS)

            aligned = abs(bearing) <= float(self.ALIGN_BEARING_OK)
            abort = bool(vis.bump or vis.loom or vis.dragonfly)
            if aligned or abort or self._align_left <= 0:
                self._align_done = True
                self._align_left = 0
            else:
                self._align_left -= 1
                max_drive = self.MAX_DRIVE_TERRAIN if self._enable_terrain else self.MAX_DRIVE
                min_side = self.MIN_SIDE_DRIVE_TERRAIN if self._enable_terrain else self.MIN_SIDE_DRIVE
                if self._align_dir > 0:
                    # bearing>0 ⇒ banane à GAUCHE → drives=[min,max] (drives[0]<drives[1]) → tourne à GAUCHE
                    drives = np.array([min_side, max_drive], dtype=float)
                else:
                    # bearing≤0 ⇒ banane à DROITE → drives=[max,min] (drives[0]>drives[1]) → tourne à DROITE
                    drives = np.array([max_drive, min_side], dtype=float)
                if self._vis_trace_enabled():
                    bd_ref = 0.5 * (float(drives[0]) + float(drives[1]))
                    turn_cmd = self._vis_trace_command_line(
                        float(drives[0]),
                        float(drives[1]),
                        bd_ref,
                        "ALIGN",
                        "ALIGN",
                    )
                    self._trace_sep_before_bundle()
                    print(
                        "[VIS_TRACE drives] mode=ALIGN dir={:+d} bearing={:+.3f} "
                        "left={} aligned={} abort={} drives=({:.3f},{:.3f})".format(
                            int(self._align_dir),
                            float(bearing),
                            int(self._align_left),
                            bool(aligned),
                            bool(abort),
                            float(drives[0]),
                            float(drives[1]),
                        ),
                        flush=True,
                    )
                    print("[VIS_TRACE command] {}".format(turn_cmd), flush=True)
                    if not isinstance(self._last_vis_trace, dict):
                        self._last_vis_trace = {}
                    self._last_vis_trace["turn_command"] = turn_cmd
                    self._last_vis_trace["drives_cmd"] = (
                        float(drives[0]),
                        float(drives[1]),
                    )
                if (
                    self.DEBUG
                    and self._debug_decisions <= self.DEBUG_MAX_DECISIONS
                    and (self._debug_decisions % self.DEBUG_EVERY_DECISIONS == 0)
                ):
                    print(
                        f"[dbg d={self._debug_decisions:4d}] mode=ALIGN dir={int(self._align_dir):+d} "
                        f"bearing={bearing:+.3f} align_left={self._align_left} "
                        f"drives=({drives[0]:.3f},{drives[1]:.3f})",
                        flush=True,
                    )
                return drives

        # ---- FSM AVOID/GO ----
        self._update_avoid_state(feat, vis)

        if self._avoid_left > 0:
            bias, base_drive, sub_mode = self._avoid_command(target_bias, feat, vis)
            self._last_vision_turn_bias = float(bias - target_bias)
        else:
            bias = float(target_bias)
            base_drive = float(self.BASE_DRIVE_FAST)
            sub_mode = "GO"
            self._last_vision_turn_bias = 0.0
        self._avoid_sub = sub_mode

        # ---- Terrain (Level 1) ----
        turn_mod = self.TURN_MOD
        if self._enable_terrain:
            slope_forward, slope_lateral, slope_mag = self._get_slope_signals(sim)
            downhill = max(0.0, -slope_forward)
            steep_weight = 0.25 + 0.75 * downhill  # keep power when climbing
            speed_scale = 1.0 / (
                1.0
                + self.DOWNHILL_BRAKE * downhill
                + self.STEEP_BRAKE * steep_weight * max(0.0, slope_mag)
            )
            base_drive = base_drive * speed_scale
            slope_bias = -self.SLOPE_STEER_GAIN * float(slope_lateral) * float(downhill)
            bias += float(np.clip(slope_bias, -self.SLOPE_STEER_MAX, self.SLOPE_STEER_MAX))
            turn_mod = turn_mod / (1.0 + self.TURN_STEEP_GAIN * max(0.0, slope_mag))

        if self._avoid_left > 0:
            turn_mod = float(self.AVOID_TURN_MOD)

        # ---- tanh + asymétrie roues (logique walking actuelle) ----
        bias_norm = float(np.tanh(bias))

        min_drive = self.MIN_DRIVE_TERRAIN if self._enable_terrain else self.MIN_DRIVE
        min_side = self.MIN_SIDE_DRIVE_TERRAIN if self._enable_terrain else self.MIN_SIDE_DRIVE
        max_drive = self.MAX_DRIVE_TERRAIN if self._enable_terrain else self.MAX_DRIVE
        base_drive = float(np.clip(base_drive, min_drive, max_drive))

        drives = np.full(2, base_drive, dtype=float)
        side = int(bias_norm > 0)
        drives[side] -= abs(bias_norm) * turn_mod * base_drive
        drives[side] = max(min_side, drives[side])
        drives = np.clip(drives, 0.0, max_drive)

        # ---- Memory of last avoid direction (anti flip-flop) ----
        if self.VIS_DIRECTION_MEMORY and abs(self._dodge_dir) > 1e-6:
            self._vis_last_dir = float(
                self.VIS_LAST_DIR_DECAY * self._vis_last_dir
                + (1.0 - self.VIS_LAST_DIR_DECAY) * self._dodge_dir
            )
            if abs(self._vis_last_dir) < 1e-6:
                self._vis_last_dir = 1.0

        if self._vis_trace_enabled() and self.VISION_ENABLE and self._enable_grass:
            mode = "AVOID" if self._avoid_left > 0 else "GO"
            ix_reduce = 1 if bias_norm > 0 else 0
            self._trace_sep_before_bundle()
            print(
                "[VIS_TRACE drives] mode={} sub={} dodge_dir={:+.2f} avoid_left={} "
                "bias_total={:+.5f} bias_norm={:+.5f} | diminue drives[{}]".format(
                    mode,
                    sub_mode,
                    float(self._dodge_dir),
                    int(self._avoid_left),
                    float(bias),
                    float(bias_norm),
                    ix_reduce,
                ),
                flush=True,
            )
            print(
                "[VIS_TRACE drives] drives=({:.4f},{:.4f}) base={:.3f} target_bias={:+.4f}".format(
                    float(drives[0]),
                    float(drives[1]),
                    float(base_drive),
                    float(target_bias),
                ),
                flush=True,
            )
            lr_for_cmd = (
                float(self._avoid_lr_from_feat(feat)) if feat is not None else 0.0
            )
            turn_cmd = self._vis_trace_command_line(
                float(drives[0]),
                float(drives[1]),
                float(base_drive),
                mode,
                sub_mode,
                lr_avoid=lr_for_cmd,
            )
            print("[VIS_TRACE command] {}".format(turn_cmd), flush=True)
            if not isinstance(self._last_vis_trace, dict):
                self._last_vis_trace = {}
            self._last_vis_trace["turn_command"] = turn_cmd
            self._last_vis_trace["drives_cmd"] = (
                float(drives[0]),
                float(drives[1]),
            )

        if (
            self.DEBUG
            and self._debug_decisions <= self.DEBUG_MAX_DECISIONS
            and (self._debug_decisions % self.DEBUG_EVERY_DECISIONS == 0)
        ):
            print(
                f"[dbg d={self._debug_decisions:4d}] mode={'AVOID' if self._avoid_left>0 else 'GO':5s} "
                f"sub={sub_mode:5s} dodge={self._dodge_dir:+.1f} align_done={self._align_done} "
                f"dist={dist_to_banana} bearing={bearing:+.3f} target_bias={target_bias:+.3f} "
                f"bias={bias:+.3f} bias_norm={bias_norm:+.3f} drives=({drives[0]:.3f},{drives[1]:.3f})",
                flush=True,
            )
        return drives

    # ------------------------------------------------------------------
    # FSM AVOID
    # ------------------------------------------------------------------
    def _schedule_realign_after_long_avoid(self) -> None:
        """Si l'épisode AVOID était assez long, relancer ALIGN vers la banane."""
        ticks = int(self._avoid_session_ticks)
        self._avoid_session_ticks = 0
        if (
            not self.ALIGN_INITIAL_ENABLE
            or not self.ALIGN_AFTER_AVOID_ENABLE
        ):
            return
        if ticks >= int(self.ALIGN_AFTER_AVOID_MIN_DECISIONS):
            self._align_done = False
            self._align_dir = 0.0
            self._align_left = 0
            if self._vis_trace_enabled():
                self._trace_sep_before_bundle()
                print(
                    "[VIS_TRACE align] post_avoid_realign ticks={}>={} "
                    "(réalignement banane)".format(
                        ticks,
                        int(self.ALIGN_AFTER_AVOID_MIN_DECISIONS),
                    ),
                    flush=True,
                )

    def _update_avoid_state(self, feat: dict | None, vis: _VisFlags) -> None:
        """Trigger / exit / latch dodge_dir for the AVOID state."""
        if feat is None:
            # Pas de vision dispo : on ne change pas l'état mais on continue à
            # décrémenter le compte minimal d'engagement et on sort si on est
            # déjà en AVOID sans signal.
            if self._avoid_left > 0:
                self._avoid_session_ticks += 1
                self._avoid_min_left = max(0, self._avoid_min_left - 1)
                self._avoid_clear += 1
                if (
                    self._avoid_min_left <= 0
                    and self._avoid_clear >= int(self.AVOID_CLEAR_DECISIONS)
                ):
                    self._schedule_realign_after_long_avoid()
                    self._avoid_left = 0
                    self._avoid_clear = 0
                    self._dodge_dir = 0.0
            return

        center_area = float(feat.get("center_area", 0.0))
        total_area = float(feat.get("total_area", 0.0))

        on = (
            center_area >= float(self.AVOID_CENTER_ON)
            or total_area >= float(self.AVOID_TOTAL_ON)
            or bool(vis.bump)
            or bool(vis.loom)
            or bool(vis.dragonfly)
        )
        off = (
            center_area <= float(self.AVOID_CENTER_OFF)
            and total_area <= float(self.AVOID_TOTAL_OFF)
            and not bool(vis.bump)
            and not bool(vis.loom)
            and not bool(vis.dragonfly)
        )

        if self._avoid_left > 0:
            # Déjà en AVOID : décompte minimum + compte clear.
            self._avoid_session_ticks += 1
            self._avoid_min_left = max(0, self._avoid_min_left - 1)
            if off:
                self._avoid_clear += 1
            else:
                self._avoid_clear = 0
            # Vision live : on recalcule le côté de l'obstacle à CHAQUE tick AVOID
            # (utilisateur veut une décision recalculée à chaque étape). Convention :
            # dodge = +sign(lr). Si lr est sous le seuil de bruit, on garde le latch.
            lr_now = float(self._avoid_lr_from_feat(feat))
            if abs(lr_now) >= float(self.AVOID_LR_REFRESH):
                self._dodge_dir = float(np.sign(lr_now))
            if (
                self._avoid_min_left <= 0
                and self._avoid_clear >= int(self.AVOID_CLEAR_DECISIONS)
            ):
                self._schedule_realign_after_long_avoid()
                self._avoid_left = 0
                self._avoid_clear = 0
                self._dodge_dir = 0.0
                return
            self._avoid_left += 1
        else:
            if not on:
                return
            # Entrée en AVOID : latch dodge_dir.
            # Convention motrice (validée empiriquement par observation utilisateur) :
            #   drives[0]>drives[1] ([max,min]) ⇒ tourne à DROITE ;
            #   drives[0]<drives[1] ([min,max]) ⇒ tourne à GAUCHE.
            # bias_norm>0 réduit drives[1] → [max,min] → DROITE.
            # bias_norm<0 réduit drives[0] → [min,max] → GAUCHE.
            # lr>0 obstacle à GAUCHE → esquiver à DROITE → bias>0 → dodge=+1 = +sign(lr).
            # lr<0 obstacle à DROITE → esquiver à GAUCHE → bias<0 → dodge=-1 = +sign(lr).
            lr = float(self._avoid_lr_from_feat(feat))
            if abs(lr) > 1e-9:
                dodge = float(np.sign(lr))
            elif self.VIS_DIRECTION_MEMORY and abs(self._vis_last_dir) > 1e-6:
                dodge = float(np.sign(self._vis_last_dir))
            else:
                dodge = 1.0
            self._dodge_dir = dodge
            self._avoid_session_ticks = 0
            self._avoid_left = 1
            self._avoid_min_left = int(self.AVOID_MIN_DURATION)
            self._avoid_clear = 0

    def _avoid_command(
        self, target_bias: float, feat: dict | None, vis: _VisFlags
    ) -> tuple[float, float, str]:
        """Renvoie (bias, base_drive, sub_mode) — PIVOT (frontal) ou ARC (latéral)."""
        center_area = float(feat.get("center_area", 0.0)) if feat is not None else 0.0
        dodge = self._dodge_dir if abs(self._dodge_dir) > 1e-6 else 1.0
        pivot = center_area >= float(self.AVOID_PIVOT_CENTER) or bool(vis.bump)
        max_turn = float(self.VIS_TURN_MAX)
        if pivot:
            bias = float(np.clip(self.AVOID_PIVOT_TURN * dodge, -max_turn, max_turn))
            base = float(self.AVOID_PIVOT_SPEED * self.BASE_DRIVE_FAST)
            return bias, base, "PIVOT"
        bias = self.AVOID_ARC_TURN * dodge + self.AVOID_ARC_TARGET_BLEND * float(target_bias)
        bias = float(np.clip(bias, -max_turn, max_turn))
        base = float(self.AVOID_ARC_SPEED * self.BASE_DRIVE_FAST)
        return bias, base, "ARC"

    # ------------------------------------------------------------------
    # Colour masks (kept verbatim — used by both detection and overlay).
    # ------------------------------------------------------------------
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

    def _compute_sky_mask(self, roi_rgb: np.ndarray) -> np.ndarray:
        """Pixels classified as sky / clouds / upper background (not grass).

        Grass uses the saturated chroma gate.  Sky-like pixels are: dark or
        blue-tinted (classic sky), OR bright neutral patches (clouds: similar
        R,G,B so spike silhouettes still read as sky–blade–sky).
        """
        if not self.VIS_COLOR_ENABLE:
            return np.zeros(roi_rgb.shape[:2], dtype=bool)
        r = roi_rgb[..., 0]
        g = roi_rgb[..., 1]
        b = roi_rgb[..., 2]
        lum = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
        grass = self._compute_green_mask(roi_rgb)
        bm = float(self.VIS_SKY_BLUE_MARGIN)
        blue_ok = (b + bm >= r) & (b + bm >= g)
        dark_ok = lum <= float(self.VIS_SKY_LUM_MAX)
        loose_ok = lum <= float(self.VIS_SKY_LUM_LOOSE)
        sky_blue = dark_ok | (blue_ok & loose_ok)
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        rgb_spread = (mx - mn).astype(np.float32)
        cmin = float(self.VIS_CLOUD_LUM_MIN)
        cmax = float(self.VIS_CLOUD_LUM_MAX)
        spr_max = float(self.VIS_CLOUD_RGB_SPREAD_MAX)
        cloud_ok = (
            (lum >= cmin)
            & (lum <= cmax)
            & (rgb_spread <= spr_max)
        )
        return (~grass) & (sky_blue | cloud_ok)

    def _compute_grass_mask(self, roi_rgb: np.ndarray) -> np.ndarray:
        """Backwards-compatible alias for legacy callers."""
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

    # Sky/blade sandwich spikes + CC filter (aire / hauteur / aspect).
    def _discard_small_spike_blobs(self, spike_bool: np.ndarray) -> np.ndarray:
        """Garde seulement les composantes connexes qui ressemblent à un pic.

        Trois critères par blob (tous doivent passer) :
          1. aire ≥ VIS_SPIKE_MIN_AREA_PX (élimine le bruit fin),
          2. hauteur du bbox ≥ VIS_SPIKE_MIN_HEIGHT_PX (élimine les bandes
             horizontales fines comme les silhouettes d'arbres lointains),
          3. ratio hauteur/largeur du bbox ≥ VIS_SPIKE_MIN_ASPECT (un vrai pic
             est plus haut que large).
        """
        min_px = int(self.VIS_SPIKE_MIN_AREA_PX)
        min_h = int(self.VIS_SPIKE_MIN_HEIGHT_PX)
        min_aspect = float(self.VIS_SPIKE_MIN_ASPECT)
        any_filter = (min_px > 0) or (min_h > 0) or (min_aspect > 0.0)
        if (
            not any_filter
            or _SCIPY_NDIMAGE is None
            or spike_bool.size == 0
            or not np.asarray(spike_bool, dtype=bool).any()
        ):
            return spike_bool
        m = np.asarray(spike_bool, dtype=bool)
        lbl, n_labels = _SCIPY_NDIMAGE.label(m, structure=_SPIKE_CONN)
        if n_labels == 0:
            return spike_bool

        counts = np.bincount(lbl.ravel())
        # bool array indexé par label, True = à supprimer.
        bad = np.zeros(counts.shape[0], dtype=bool)
        bad[0] = False  # background
        if min_px > 0:
            bad |= counts < min_px

        if min_h > 0 or min_aspect > 0.0:
            slices = _SCIPY_NDIMAGE.find_objects(lbl)
            for i, sl in enumerate(slices, start=1):
                if sl is None or bad[i]:
                    continue
                row_sl, col_sl = sl
                bb_h = int(row_sl.stop - row_sl.start)
                bb_w = int(col_sl.stop - col_sl.start)
                if min_h > 0 and bb_h < min_h:
                    bad[i] = True
                    continue
                if min_aspect > 0.0 and bb_w > 0:
                    aspect = bb_h / float(bb_w)
                    if aspect < min_aspect:
                        bad[i] = True

        return m & ~bad[lbl]

    def _vis_roi_col_frac(self, eye: str) -> tuple[float, float]:
        """Colonnes du ROI vision : eye est 'left' ou 'right'."""
        if eye == "left":
            return float(self.VIS_ROI_C0_LEFT), float(self.VIS_ROI_C1_LEFT)
        return float(self.VIS_ROI_C0_RIGHT), float(self.VIS_ROI_C1_RIGHT)

    def _compute_tip_profile(
        self, eye_img01: np.ndarray, eye: str
    ) -> tuple[np.ndarray, np.ndarray]:
        h_full, w_full = eye_img01.shape[:2]
        r0 = int(self.VIS_ROI_R0 * h_full)
        r1 = int(self.VIS_ROI_R1 * h_full)
        c0f, c1f = self._vis_roi_col_frac(eye)
        c0 = int(c0f * w_full)
        c1 = int(c1f * w_full)
        roi = eye_img01[r0:r1, c0:c1, :]

        h, w = roi.shape[:2]
        empty_mask = np.zeros((h, w), dtype=bool)
        empty = (np.zeros(w, np.float32), empty_mask)
        if h < 2 or w < 2 or not self.VIS_COLOR_ENABLE:
            return empty

        green = self._compute_green_mask(roi)
        sky = self._compute_sky_mask(roi)
        if not green.any():
            return empty

        spike_full = np.zeros((h, w), dtype=bool)
        for r in range(h):
            rs = sky[r]
            rg = green[r]
            c = 0
            while c < w:
                while c < w and rs[c]:
                    c += 1
                if c >= w:
                    break
                s = c
                while c < w and not rs[c]:
                    c += 1
                e = c
                if s > 0 and rs[s - 1] and e < w and rs[e]:
                    spike_full[r, s:e] = rg[s:e]

        spike_full = self._discard_small_spike_blobs(spike_full)
        tips = spike_full.sum(axis=0).astype(np.float32)
        return tips, spike_full

    def _drives_pivot_label(self, d0: float, d1: float, base_drive: float) -> str:
        """Pivot moteur effectif (convention empirique validée par observation utilisateur) :
        drives[0]>drives[1] ⇒ tourne à DROITE ; drives[0]<drives[1] ⇒ tourne à GAUCHE.
        """
        ref = max(float(base_drive), abs(float(d0)), abs(float(d1)), 1e-9)
        eps = max(1e-9, float(self.VIS_TRACE_DRIVE_EPS_FRAC) * ref)
        delta = float(d0) - float(d1)
        if abs(delta) <= eps:
            return "moteur quasi symétrique"
        if delta > eps:
            return "tourne à DROITE"
        return "tourne à GAUCHE"

    def _vis_trace_command_line(
        self,
        d0: float,
        d1: float,
        base_drive: float,
        mode: str,
        sub_mode: str,
        lr_avoid: float | None = None,
    ) -> str:
        """Libellé de décision pour la trace.

        - ALIGN : « banane à GAUCHE/DROITE du cap » selon `_align_dir`.
        - AVOID : côté de l'obstacle = signe de `lr_avoid` actuel (vision live) ;
                  pivot moteur = comparaison réelle des drives.
        - GO    : pivot moteur basé sur les drives.
        """
        if mode == "ALIGN":
            if float(self._align_dir) > 0:
                return "commande: ALIGN pivot vers la banane (elle est à GAUCHE du cap, bearing>0)"
            if float(self._align_dir) < 0:
                return "commande: ALIGN pivot vers la banane (elle est à DROITE du cap, bearing≤0)"
            return "commande: ALIGN (direction non latchée)"

        pivot_txt = self._drives_pivot_label(d0, d1, base_drive)

        if mode == "AVOID":
            sm = sub_mode or "?"
            lr = float(lr_avoid) if lr_avoid is not None else 0.0
            if lr > 1e-9:
                obs_txt = "obstacle à GAUCHE (lr>0)"
            elif lr < -1e-9:
                obs_txt = "obstacle à DROITE (lr<0)"
            else:
                # Vision floue : retombe sur le latch dodge_dir (= +sign(lr)).
                dd = float(self._dodge_dir)
                if dd > 0:
                    obs_txt = "obstacle (latch) à GAUCHE"
                elif dd < 0:
                    obs_txt = "obstacle (latch) à DROITE"
                else:
                    obs_txt = "obstacle (latch indéterminé)"
            return "commande: AVOID [{}] {} → {}".format(sm, obs_txt, pivot_txt)

        # GO
        return "commande: GO {}".format(pivot_txt)

    def _vis_trace_enabled(self) -> bool:
        if bool(getattr(self.__class__, "VIS_TRACE_VERBOSE", False)):
            return True
        return os.environ.get("CONTROLLER_VIS_TRACE", "").lower() in (
            "1",
            "true",
            "yes",
        )

    def _trace_sep_before_bundle(self) -> None:
        """Ligne vide avant le premier bloc de trace de cet appel à _compute_drives."""
        if not self._vis_trace_enabled():
            return
        if getattr(self, "_vis_trace_sep_emitted_this_drive", False):
            return
        print("", flush=True)
        self._vis_trace_sep_emitted_this_drive = True

    def _avoid_lr_from_feat(self, feat: dict) -> float:
        """LEFT vs RIGHT spike : signal de côté en repère corps.

        Signal *primaire* : `raw_left_area − raw_right_area` (= « quel œil voit plus
        de pic »). Si l'œil gauche voit plus, l'obstacle est sur la GAUCHE du corps,
        peu importe sa position dans la colonne d'image.
        Le delta moitiés-d'image (`spike_half_left_px − spike_half_right_px`) est
        AMBIGU dans la zone binoculaire (un obstacle vu par un seul œil mais dans
        la moitié centrale de cet œil bascule artificiellement le signe). On ne
        l'utilise plus qu'en *départage* lorsque les deux yeux sont équilibrés.

        Convention motrice (validée empiriquement par observation utilisateur) :
          drives[0]>drives[1] ([max,min]) ⇒ tourne à DROITE ;
          drives[0]<drives[1] ([min,max]) ⇒ tourne à GAUCHE.
        Donc `_dodge_dir = +sign(lr)` : lr>0 (obstacle gauche) → dodge=+1
        → bias>0 → réduit drives[1] → [max,min] → yaw à DROITE.
        """
        raw_l = float(feat.get("raw_left_area", 0.0))
        raw_r = float(feat.get("raw_right_area", 0.0))
        ema_l = float(feat.get("left_area", 0.0))
        ema_r = float(feat.get("right_area", 0.0))

        raw_diff = raw_l - raw_r
        ema_diff = ema_l - ema_r
        # Choisit la signature la plus forte (raw réagit plus vite, EMA filtre le bruit).
        eye_diff = raw_diff if abs(raw_diff) >= abs(ema_diff) else ema_diff
        eye_total = max(raw_l + raw_r, ema_l + ema_r, 1e-6)
        eye_imbalance = abs(eye_diff) / eye_total  # ∈ [0, 1]

        # 1) Un œil domine clairement → côté corps non ambigu, on tranche.
        if eye_imbalance >= 0.20:
            return float(eye_diff)

        # 2) Les deux yeux voient à peu près pareil (obstacle frontal binoculaire).
        #    On tente le départage par moitiés d'image (signal moins fiable mais utile en frontal pur).
        ls = float(feat.get("spike_half_left_px", 0))
        rs = float(feat.get("spike_half_right_px", 0))
        roi_bp = float(feat.get("roi_binocular_px", 1.0))
        tot_sp = ls + rs
        eps_px = float(self.VIS_LR_EPS_PIXELS)
        min_sp = float(self.VIS_LR_MIN_SPIKE_PIXELS)
        if tot_sp >= min_sp:
            diff_px = ls - rs
            if abs(diff_px) >= eps_px:
                return float(diff_px) / max(roi_bp, 1.0)

        # 3) Mémoire directionnelle (anti flip-flop sur signaux quasi nuls).
        if self.VIS_DIRECTION_MEMORY and abs(self._vis_last_dir) > 1e-6:
            ta = float(feat.get("total_area", 0.0))
            mag = float(max(abs(eye_diff), 0.5 * ta, 1e-6))
            return float(np.sign(self._vis_last_dir)) * mag

        return float(eye_diff)

    # ------------------------------------------------------------------
    # Vision step: extract features + reflex flags (bump / loom / dragonfly).
    # Détection identique à l'ancienne cascade, mais centralisée.
    # ------------------------------------------------------------------
    def _vision_step(
        self, sim: MiniprojectSimulation
    ) -> tuple[dict | None, _VisFlags]:
        flags = _VisFlags()

        def _to_float01(img: np.ndarray) -> np.ndarray:
            a = np.asarray(img, dtype=np.float32)
            if a.ndim == 2:
                a = np.stack([a, a, a], axis=-1)
            if a.max() > 1.0:
                a = a / 255.0
            return np.clip(a, 0.0, 1.0)

        def _roi(img01: np.ndarray, eye_tag: str) -> np.ndarray:
            h, w = img01.shape[0], img01.shape[1]
            r0, r1 = int(h * self.VIS_ROI_R0), int(h * self.VIS_ROI_R1)
            c0f, c1f = self._vis_roi_col_frac(eye_tag)
            c0, c1 = int(w * c0f), int(w * c1f)
            return img01[r0:r1, c0:c1, :]

        frames = None
        if self.VISION_USE_RAW:
            try:
                frames = sim.get_raw_vision(sim.fly.name)
            except Exception:
                frames = None

        if frames is None or len(frames) == 0:
            return None, flags

        left_img01 = _to_float01(frames[0])
        right_img01 = _to_float01(frames[1] if len(frames) > 1 else frames[0])

        # ---- Spikes / tips per eye ----
        left_tips, l_spike = self._compute_tip_profile(left_img01, "left")
        right_tips, r_spike = self._compute_tip_profile(right_img01, "right")
        self._vis_spike_roi_left = l_spike
        self._vis_spike_roi_right = r_spike

        h_full = left_img01.shape[0]
        h_roi = int(self.VIS_ROI_R1 * h_full) - int(self.VIS_ROI_R0 * h_full)
        wl = max(1, int(left_tips.shape[0]))
        wr = max(1, int(right_tips.shape[0]))
        roi_area_left = max(1, h_roi * wl)
        roi_area_right = max(1, h_roi * wr)
        roi_binocular_px = float(roi_area_left + roi_area_right)

        def _half_roi_spike_px(mask: np.ndarray) -> tuple[int, int]:
            _, ww = mask.shape
            mid = ww // 2
            left = int(mask[:, :mid].sum())
            right = int(mask[:, mid:].sum())
            return left, right

        ll, lr_h = _half_roi_spike_px(l_spike)
        rl, rr = _half_roi_spike_px(r_spike)
        spike_half_left = ll + rl
        spike_half_right = lr_h + rr

        left_area = float(left_tips.sum()) / float(roi_area_left)
        right_area = float(right_tips.sum()) / float(roi_area_right)

        # Centre columns (straight-ahead portion of each eye ROI).
        cL_L = int(self.VIS_CENTER_C0 * wl)
        cR_L = max(cL_L + 1, int(self.VIS_CENTER_C1 * wl))
        cL_R = int(self.VIS_CENTER_C0 * wr)
        cR_R = max(cL_R + 1, int(self.VIS_CENTER_C1 * wr))
        center_area_raw = 0.5 * (
            float(left_tips[cL_L:cR_L].sum())
            / max(1, h_roi * (cR_L - cL_L))
            + float(right_tips[cL_R:cR_R].sum())
            / max(1, h_roi * (cR_R - cL_R))
        )

        total_area_raw = 0.5 * (left_area + right_area)

        # ---- EMA smoothing (compat. seuils) ----
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

        # ---- Dragonfly (red eyes) ----
        lroi = _roi(left_img01, "left")
        rroi = _roi(right_img01, "right")
        l_drag = self._compute_dragonfly_mask(lroi).astype(np.float32)
        r_drag = self._compute_dragonfly_mask(rroi).astype(np.float32)

        def _df_area_x(mask2d: np.ndarray) -> tuple[float, float]:
            area = float(mask2d.mean())
            if area <= 1e-9:
                return 0.0, 0.0
            cols = mask2d.mean(axis=0)
            xs = np.linspace(-1.0, 1.0, cols.shape[0], dtype=np.float32)
            x_mean = float((cols * xs).sum() / max(1e-9, float(cols.sum())))
            return area, x_mean

        df_left_area, df_left_x = _df_area_x(l_drag)
        df_right_area, df_right_x = _df_area_x(r_drag)
        df_total = 0.5 * (df_left_area + df_right_area)
        df_w = df_left_area + df_right_area
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

        # ---- Blade pixel counts (LEFT vs RIGHT, binocular) ----
        self._blade_left_px = spike_half_left
        self._blade_right_px = spike_half_right

        feat: dict = {
            "left_area":           float(self._vis_left_area),
            "right_area":          float(self._vis_right_area),
            "raw_left_area":       left_area,
            "raw_right_area":      right_area,
            "center_area":         float(self._vis_center_area),
            "total_area":          float(self._vis_total_area),
            "d_total_area":        float(self._vis_d_total_area),
            "dragonfly_area":      float(self._vis_dragonfly_area),
            "dragonfly_x":         float(self._vis_dragonfly_x),
            "left_blade_px":       spike_half_left,
            "right_blade_px":      spike_half_right,
            "spike_half_left_px":  spike_half_left,
            "spike_half_right_px": spike_half_right,
            "roi_binocular_px":    roi_binocular_px,
        }

        # ---- Reflex latches: bump / loom / dragonfly ----
        # Bump: contact horizontal force + visual hint.
        if self.BUMP_ENABLE:
            contact_max = 0.0
            try:
                cf = sim.mj_data.cfrc_ext[self._contact_body_ids, :3]
                mag = np.linalg.norm(cf[:, :2], axis=1)
                contact_max = float(np.max(mag)) if mag.size > 0 else 0.0
            except Exception:
                contact_max = 0.0
            prev_c = float(self._bump_contact_ema)
            self._bump_contact_ema = float(0.90 * prev_c + 0.10 * contact_max)
            d_contact = float(self._bump_contact_ema - self._bump_contact_prev)
            self._bump_contact_prev = float(self._bump_contact_ema)

            vis_hint = (
                feat["center_area"] >= 0.012
                or feat["total_area"] >= 0.018
            )
            on_b = (d_contact >= float(self.BUMP_DCONTACT_ON)) and vis_hint
            off_b = (d_contact <= float(self.BUMP_DCONTACT_OFF)) and (not vis_hint)

            if self._bump_left > 0:
                self._bump_left -= 1
            if on_b:
                self._bump_left = max(self._bump_left, int(self.BUMP_LATCH_DECISIONS))
            if off_b and self._bump_left <= 0:
                self._bump_left = 0
            flags.bump = self._bump_left > 0

        # Looming: large/grown pic in front of the fly.
        if self.LOOM_ENABLE:
            on_l = (
                feat["center_area"] >= float(self.LOOM_CENTER_ON)
                or feat["d_total_area"] >= float(self.LOOM_DAREA_ON)
            )
            off_l = (
                feat["center_area"] <= float(self.LOOM_CENTER_OFF)
                and feat["d_total_area"] <= float(self.LOOM_DAREA_OFF)
            )
            if self._loom_left > 0:
                self._loom_left -= 1
            if on_l:
                self._loom_left = max(self._loom_left, int(self.LOOM_LATCH_DECISIONS))
            if off_l and self._loom_left <= 0:
                self._loom_left = 0
            flags.loom = self._loom_left > 0

        # Dragonfly: saturated red blob.
        if self.VIS_COLOR_ENABLE:
            df_area = feat["dragonfly_area"]
            on_d = df_area >= float(self.VIS_DF_AREA_ON)
            off_d = df_area <= float(self.VIS_DF_AREA_OFF)
            if self._dragonfly_left > 0:
                self._dragonfly_left -= 1
            if on_d:
                self._dragonfly_left = max(
                    self._dragonfly_left, int(self.VIS_DF_LATCH_DECISIONS)
                )
            if off_d and self._dragonfly_left <= 0:
                self._dragonfly_left = 0
            flags.dragonfly = self._dragonfly_left > 0

        if self._vis_trace_enabled():
            ls_px = int(feat["spike_half_left_px"])
            rs_px = int(feat["spike_half_right_px"])
            lr_avoid = float(self._avoid_lr_from_feat(feat))
            self._trace_sep_before_bundle()
            print(
                "[VIS_TRACE feats] spike_half_px L={} R={} | lr_avoid={:+.6f} "
                "(entrées vision uniquement ; décision latérale → ligne VIS_TRACE command)".format(
                    ls_px,
                    rs_px,
                    lr_avoid,
                ),
                flush=True,
            )
            print(
                "[VIS_TRACE feats] raw_area L={:.5f} R={:.5f} total={:.5f} center={:.5f} "
                "d_total={:+.5f} df_area={:.5f}".format(
                    float(feat["raw_left_area"]),
                    float(feat["raw_right_area"]),
                    float(feat["total_area"]),
                    float(feat["center_area"]),
                    float(feat["d_total_area"]),
                    float(feat["dragonfly_area"]),
                ),
                flush=True,
            )
            print(
                "[VIS_TRACE flags] bump={} loom={} dragonfly={} (latches: bump={} loom={} df={})".format(
                    bool(flags.bump),
                    bool(flags.loom),
                    bool(flags.dragonfly),
                    int(self._bump_left),
                    int(self._loom_left),
                    int(self._dragonfly_left),
                ),
                flush=True,
            )

        self._last_vis_trace = {
            "spike_half_left_px": feat["spike_half_left_px"],
            "spike_half_right_px": feat["spike_half_right_px"],
            "lr_avoid": float(self._avoid_lr_from_feat(feat)),
            "bump": bool(flags.bump),
            "loom": bool(flags.loom),
            "dragonfly": bool(flags.dragonfly),
            "turn_command": None,
            "drives_cmd": None,
        }

        return feat, flags

    # ------------------------------------------------------------------
    def compute_vision_debug_overlay(
        self, sim: MiniprojectSimulation
    ) -> "np.ndarray | None":
        """Debug RGB overlay (sky tint, spikes, apex dots, dragonfly, ROI frame)."""
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
        if len(eye_imgs) == 1:
            eye_imgs.append(eye_imgs[0])

        out_eyes: list[np.ndarray] = []
        for ei, raw in enumerate(eye_imgs):
            eye = "left" if ei == 0 else "right"
            h, w = raw.shape[:2]
            r0, r1 = int(h * self.VIS_ROI_R0), int(h * self.VIS_ROI_R1)
            c0f, c1f = self._vis_roi_col_frac(eye)
            c0, c1 = int(w * c0f), int(w * c1f)

            img01 = raw.astype(np.float32) / 255.0

            _, spike_roi = self._compute_tip_profile(img01, eye)

            roi = img01[r0:r1, c0:c1, :]
            df_mask = self._compute_dragonfly_mask(roi)

            base = (raw.astype(np.float32) * 0.45).astype(np.uint8)
            overlay = base.copy()
            roi_view = overlay[r0:r1, c0:c1, :]

            sky_roi = self._compute_sky_mask(roi)
            tint = np.array([45.0, 88.0, 188.0], dtype=np.float32)
            roi_view[sky_roi] = (
                roi_view[sky_roi].astype(np.float32) * 0.52 + tint * 0.48
            ).astype(np.uint8)

            if spike_roi.shape == roi_view.shape[:2]:
                roi_view[spike_roi] = np.array([20, 255, 40], dtype=np.uint8)

            cols = np.flatnonzero(np.any(spike_roi, axis=0))
            if cols.size > 0:
                apex_r = np.argmax(spike_roi[:, cols], axis=0)
                roi_view[apex_r, cols, :] = np.array([255, 220, 0], dtype=np.uint8)

            if df_mask.shape == roi_view.shape[:2]:
                roi_view[df_mask] = np.array([255, 30, 30], dtype=np.uint8)

            cyan = np.array([0, 220, 220], dtype=np.uint8)
            overlay[r0 : r0 + 1, c0:c1, :] = cyan
            overlay[r1 - 1 : r1, c0:c1, :] = cyan
            overlay[r0:r1, c0 : c0 + 1, :] = cyan
            overlay[r0:r1, c1 - 1 : c1, :] = cyan

            out_eyes.append(overlay)

        result = np.concatenate(out_eyes, axis=1).astype(np.uint8)
        self._vis_debug_overlay = result
        return result
