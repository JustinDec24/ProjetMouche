"""Controller — MERGE de spike-ciel-sky-border et REFONTE_COMPLETE.

Combinaison :
  - FSM ALIGN / GO / AVOID arc-dodge (spike-ciel-sky-border)
  - Steering tropotaxis olfactif canonique week4 (gain=-500, bias²) +
    scan/cast quand l'odeur est perdue (REFONTE_COMPLETE)
  - Vision silhouette ommatidia (REFONTE_COMPLETE) adaptée en
    (obs_size, obs_x) consommé par la FSM AVOID
  - Réflexe head-collision 3 phases : BACKUP → TURN → BYPASS (REFONTE)
  - Anti-jam : SIDESTEP_NOPROG (REFONTE) — déclenche backup + side-step si la
    mouche ne progresse plus vers la banane ou si son roll devient instable
  - Compensation pente, grip terrain, tilt-lean, recovery flip (commun)
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage as _SCIPY_NDIMAGE
except ImportError:
    _SCIPY_NDIMAGE = None

from miniproject.simulation import MiniprojectSimulation


class Controller:
    """Controller mergé : olfaction tropotaxis + vision silhouette + AVOID FSM."""

    # --- scheduling ---
    DECISION_INTERVAL_S = 0.025  # 25 ms = 40 décisions/s

    # --- olfaction (steering tropotaxis + stop) ---
    PALP_WEIGHT = 9
    ANTENNA_WEIGHT = 1
    EPS_ODOR = 1e-12
    STOP_ODOR_THRESHOLD = 5e-4
    STOP_DIST = 2.0
    ODOR_STEER_GAIN = -500.0       # canonique week4
    ODOR_MIN_SUM = 1e-10

    # --- scan/cast quand odeur perdue ---
    SCAN_TRIGGER_DECISIONS = 8
    SCAN_DURATION_DECISIONS = 20
    ODOR_TREND_WINDOW = 16
    ODOR_TREND_LOSS_RATIO = 0.6

    # --- drives ---
    BASE_DRIVE_FAST = 2.40
    MAX_DRIVE = 2.80
    MAX_DRIVE_TERRAIN = 2.00
    MIN_DRIVE = 0.80
    MIN_DRIVE_TERRAIN = 0.60
    MIN_SIDE_DRIVE = 0.40
    MIN_SIDE_DRIVE_TERRAIN = 0.25
    TURN_MOD = 0.8
    AVOID_TURN_MOD = 1.0  # plein différentiel pendant AVOID

    # --- terrain (Level 1+) ---
    DOWNHILL_BRAKE = 1.6
    STEEP_BRAKE = 1.2
    TURN_STEEP_GAIN = 2.0
    SLOPE_STEER_GAIN = 6.0
    SLOPE_STEER_MAX = 3.0

    # --- grip ---
    CONTACT_THRESHOLD = 0.15
    TERRAIN_GRIP_FORCE = 6.0
    WIND_GRIP_FORCE = 10.0

    # --- anti-roll grip (non-terrain) ---
    TILT_GRIP_ENABLE = True
    TILT_GRIP_ROLL_ON = 0.18
    TILT_GRIP_UPRIGHT_ON = 0.85

    # --- active roll compensation ---
    TILT_LEAN_ENABLE = True
    TILT_LEAN_ROLL_ON = 0.10
    TILT_LEAN_ROLL_FULL = 0.40
    TILT_LEAN_GAIN = 0.30
    TILT_LEAN_SIGN = +1.0

    # --- orientation safety ---
    TERRAIN_UPRIGHT_TILT_WARN = 0.48
    TERRAIN_TILT_RESET_HOLD = 26
    TERRAIN_FLIP_WEAK_UPRIGHT = 0.12
    TERRAIN_FLIP_RESET_HOLD = 34

    # --- stuck detection / escape (terrain trap) ---
    STUCK_MOVE_EPS = 5e-3
    STUCK_TRIGGER_DECISIONS = 25
    ESCAPE_DURATION_DECISIONS = 10

    # --- SIDESTEP no-progress (L2 anti-grass-blade jamming) ---
    NOPROG_WINDOW = 12
    NOPROG_MIN_DELTA = 0.4
    SIDESTEP_BACKUP_DECISIONS = 5
    SIDESTEP_TURN_DECISIONS = 12
    SIDESTEP_BACKUP_DRIVE = -0.6
    SIDESTEP_DRIVE_FAST = 1.80
    SIDESTEP_DRIVE_SLOW = 0.30
    ROLL_TRIGGER_THRESH = 0.30
    ROLL_TRIGGER_HOLD = 3

    # --- initial alignment ---
    ALIGN_ENABLE = True
    ALIGN_TOLERANCE_RAD = 0.18          # ~10°
    ALIGN_MAX_DECISIONS = 80
    ALIGN_SPIN_FAST = 1.30
    ALIGN_SPIN_SLOW = -0.30
    ALIGN_TRANSITION_DECISIONS = 5
    ALIGN_TRANSITION_DRIVE = 1.50
    # Réalignement après un AVOID prolongé
    ALIGN_AFTER_AVOID_ENABLE = True
    ALIGN_AFTER_AVOID_MIN_DECISIONS = 8

    # --- head-collision recovery (3 phases) ---
    HEAD_COLLISION_ENABLE = True
    HEAD_COLLISION_FORCE_THRESH = 5.0
    HEAD_BACKUP_DECISIONS = 14
    HEAD_BACKUP_DRIVE = -0.8
    HEAD_TURN_DECISIONS = 14
    HEAD_TURN_DRIVE_FAST = 1.20
    HEAD_TURN_DRIVE_SLOW = -0.55
    HEAD_BYPASS_DECISIONS = 18
    HEAD_BYPASS_DRIVE_FORWARD = 1.40
    HEAD_BYPASS_DRIVE_SIDE = 1.00
    HEAD_COLLISION_COOLDOWN = 4

    # --- vision silhouette (Level 2) ---
    # Test "ce qui dépasse de l'horizon" : un pixel sombre n'est compté comme
    # silhouette de brin que si, dans la même colonne rétinienne au-dessus de
    # lui, on trouve un ommatidium clair (= du ciel). Élimine les pixels
    # sombres qui ne sont pas vraiment des obstacles (texture sol, fond, etc).
    VISION_ENABLE = True
    # Détection inspirée de spike-ciel-sky-border : on travaille directement
    # sur l'image RGB brute (pas d'aggrégation par ommatidium). Pour chaque
    # LIGNE de pixels dans la ROI frontale-haute, on cherche les segments
    # VERTS bordés par du CIEL des deux côtés horizontalement (= silhouette
    # de brin dépassant sur fond bleu/gris).
    #
    # Tout est vectorisé via maximum.accumulate (cumul sky vers la gauche
    # et vers la droite par ligne) → un pixel est silhouette si :
    #   - il est vert
    #   - il existe au moins un pixel ciel à sa gauche dans la même ligne
    #   - et au moins un pixel ciel à sa droite dans la même ligne
    VIS_GREEN_DELTA = 0.07          # G - R > delta ET G - B > delta
    VIS_GREEN_MIN = 0.22            # G doit être > 0.22 (catches darker piques)
    # Sky strict : bleu-dominant OU achromatique brillant. Évite que les
    # zones tan/brun (transition horizon) soient classées comme sky.
    VIS_SKY_BLUE_STRICT = 0.05      # B - max(R, G) > 0.05 → vrai bleu
    VIS_SKY_WHITE_SPREAD = 0.10     # max-min < 0.10 → vraiment achromatique
    VIS_SKY_WHITE_MIN_SUM = 1.50    # somme RGB > 1.50 pour le nuage blanc
    VIS_SKY_BLUE_MARGIN = 0.03      # B + margin >= R et B + margin >= G
    VIS_SKY_GREY_SPREAD_MAX = 0.18  # max(R,G,B) - min(R,G,B) < spread → achromatique
    VIS_SKY_MIN_SUM = 0.90          # somme RGB > 0.90
    # Filtre de forme par composante connexe (style spike-ciel) :
    #   - VIS_SPIKE_MIN_HEIGHT : rejette les petites composantes (bruit, piques
    #     très distants à peine visibles)
    #   - VIS_SPIKE_MIN_ASPECT : ratio h/w minimum → garde les structures
    #     VERTICALES (piques) et rejette les masses larges/horizontales (hills)
    VIS_SPIKE_MIN_HEIGHT = 6        # blob doit s'étendre verticalement
    VIS_SPIKE_MAX_WIDTH = 35        # largeur max ciel-borders (allow close blades)
    # Bande CENTRALE pour déclencher AVOID : on détecte tous les piques pour la
    # direction (obs_x) mais on ne déclenche AVOID que si la zone DIRECTEMENT
    # DEVANT la mouche est bloquée. Pour l'œil gauche, le frontal pur = côté
    # nasal = bord DROIT de la ROI. Symétrique pour œil droit.
    VIS_CENTRAL_FRAC = 0.22         # ~22% nasal-most de la ROI par œil
    BLADE_COUNT_URGENT = 600        # count central élevé → trigger seulement sur gros blocker
    VIS_EMA = 0.55
    VIS_STARTUP_DELAY_DECISIONS = 8
    # ROI : on étend la zone verticale jusqu'à 0.75 pour capturer le CORPS des
    # piques (pas juste la pointe). L'horizon-band est plate et large → rejetée
    # par le filtre aspect-ratio. Frontal nasal à 0.70 reste assez large.
    VIS_FRONTAL_FRAC = 0.70
    VIS_UPPER_FRAC = 0.75

    # --- AVOID FSM (déclenchement / sortie) ---
    # AVOID FSM désactivé (seuil très haut) — remplacé par champ de répulsion
    # continu en addition au target_bias dans _compute_drives. Le champ donne
    # un steering progressif qui trouve les gaps dans une forêt dense, là où
    # le AVOID arc-dodge oscillait.
    AVOID_SIZE_ON = 2.0
    AVOID_SIZE_OFF = 1.5
    AVOID_SIZE_MED = 0.85

    # Champ de répulsion : bias += REPULSION_GAIN × (-obs_x) × obs_size
    # obs_x négatif = piques à gauche → bias positif → steer droite.
    # obs_size = magnitude (forte si gros piques en face).
    REPULSION_GAIN = 2.5
    AVOID_MIN_DURATION = 4
    AVOID_CLEAR_DECISIONS = 1
    AVOID_DISABLE_CLOSE_DIST = 8.0  # < 8 m banane : on fonce
    AVOID_TURN_MAX = 1.0
    AVOID_CENTER_EPS = 0.05         # |obs_x| < eps → tiebreak côté banane
    AVOID_BANANA_BLEND = 0.5        # mélange du target_bias pendant AVOID
    AVOID_REFRESH_DELTA = 0.30      # latch refresh si signal très différent
    AVOID_SPEED_FRAC = 0.60         # vitesse pendant AVOID (× BASE_DRIVE_FAST)
    VIS_TURN_MAX = 3.0              # saturation universelle du bias

    # --- debug ---
    DEBUG = True
    DEBUG_EVERY_DECISIONS = 4
    DEBUG_MAX_DECISIONS = 5000

    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        self._decision_every = int(self.DECISION_INTERVAL_S / sim.timestep)
        self._step_count = 0
        self._drives = np.array([1.0, 1.0])
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
        try:
            head_idx = next(i for i, s in enumerate(fly_segs) if s.name == "c_head")
            self._head_body_id = body_ids[head_idx]
        except StopIteration:
            self._head_body_id = None
        self._contact_body_ids = sim._internal_contact_body_segment_ids_by_fly[sim.fly.name]

        # General state
        self._last_xy = None
        self._stuck_decisions = 0
        self._escape_decisions_left = 0
        self._escape_dir = 1
        self._flip_decisions = 0
        self._tilt_decisions = 0
        self._banana_xy = None
        self._last_dist_to_banana = None
        self._request_reset = False
        self._debug_decisions = 0

        # Olfaction state
        self._odor_history: list[float] = []
        self._no_signal_decisions = 0
        self._scan_decisions_left = 0
        self._scan_dir = +1
        self._last_target_bias = 0.0

        # Vision state — détection sky-vert-sky sur l'image RGB brute
        self._vis_obs_size = 0.0
        self._vis_obs_x = 0.0
        self._vis_debug_overlay = None

        # AVOID FSM state
        self._avoid_left = 0
        self._avoid_min_left = 0
        self._avoid_clear = 0
        self._latched_obs_x = 0.0
        self._avoid_session_ticks = 0

        # SIDESTEP no-progress state
        self._dist_history: list[float] = []
        self._sidestep_decisions_left = 0
        self._sidestep_dir = +1
        self._sidestep_cooldown = 0
        self._roll_high_count = 0

        # Initial alignment state
        self._align_done = not self.ALIGN_ENABLE
        self._align_decisions = 0
        self._align_transition_left = 0

        # Head-collision recovery state (3 phases)
        self._head_backup_left = 0
        self._head_turn_left = 0
        self._head_bypass_left = 0
        self._head_maneuver_dir = +1
        self._head_collision_cd = 0
        self._head_force_max = 0.0

        # Plus aucun précompute pour la vision : on travaille directement sur
        # l'image RGB par eye à chaque frame.

        try:
            self._banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
        except Exception:
            self._banana_xy = None

        # Soulève la mouche au spawn (libère les pattes du mesh terrain)
        try:
            import mujoco as _mj
            for _jnt in range(sim.mj_model.njnt):
                if sim.mj_model.jnt_type[_jnt] == _mj.mjtJoint.mjJNT_FREE:
                    _addr = sim.mj_model.jnt_qposadr[_jnt]
                    sim.mj_data.qpos[_addr + 2] += 2.0
                    _mj.mj_forward(sim.mj_model, sim.mj_data)
                    break
        except Exception:
            pass

    # ------------------------------------------------------------------
    def step(self, sim: MiniprojectSimulation):
        is_decision_step = self._step_count % self._decision_every == 0

        # Accumuler max(head force) entre décisions pour capter contacts brefs
        if (
            self.HEAD_COLLISION_ENABLE
            and self._enable_grass
            and self._head_body_id is not None
        ):
            try:
                hf = float(np.linalg.norm(
                    sim.mj_data.cfrc_ext[self._head_body_id, 3:]
                ))
            except Exception:
                hf = 0.0
            if hf > self._head_force_max:
                self._head_force_max = hf

        if is_decision_step:
            self._drives = self._compute_drives(sim)
            self._debug_decisions = self._step_count // self._decision_every
            self._head_force_max = 0.0
        self._step_count += 1

        joint_angles, adhesion = self.turning_controller.step(self._drives)

        # --- Active roll compensation ---
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
                        float(self.TILT_LEAN_ROLL_FULL) - float(self.TILT_LEAN_ROLL_ON),
                    ),
                )
                offset = float(self.TILT_LEAN_SIGN) * float(self.TILT_LEAN_GAIN) * ramp
                if roll_lean > 0:
                    leg_indices = (3, 4, 5)
                else:
                    leg_indices = (0, 1, 2)
                for li in leg_indices:
                    base = li * 7
                    joint_angles[base + 3] += offset
                    joint_angles[base + 5] += 0.5 * offset

        if is_decision_step and self._request_reset:
            self._request_reset = False
            self._stopped = True
            return np.zeros_like(joint_angles), np.zeros_like(adhesion)

        # --- Orientation safety (terrain) ---
        uprightness = 1.0
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

        # --- Grip control (terrain) ---
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

            in_avoid = self._avoid_left > 0
            grip_val = (
                float(self.WIND_GRIP_FORCE)
                if (self._enable_wind or in_avoid)
                else float(self.TERRAIN_GRIP_FORCE)
            )
            try:
                contact_forces = sim.mj_data.cfrc_ext[self._contact_body_ids, 3:]
                contact_mag = np.linalg.norm(contact_forces, axis=1)
                stance = contact_mag > self.CONTACT_THRESHOLD
                adhesion = np.zeros_like(adhesion)
                n = min(len(adhesion), len(stance))
                if self._enable_wind:
                    adhesion[:n] = grip_val
                else:
                    adhesion[:n] = stance[:n].astype(float) * grip_val
            except Exception:
                adhesion = (
                    np.full_like(adhesion, grip_val)
                    if self._enable_wind
                    else np.where(adhesion > 0.0, grip_val, adhesion)
                )
            adhesion = np.clip(adhesion, 0.0, grip_val)

        # --- Anti-roll grip (non-terrain) ---
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
        xmat = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
        pitch = np.arcsin(np.clip(xmat[2, 0], -1.0, 1.0))
        return pitch, xmat[2, 1], xmat[2, 2]

    def _get_body_frame_xy(self, sim) -> tuple[np.ndarray, np.ndarray]:
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

    def _compute_bearing_to_banana(self, sim, thorax_xy: np.ndarray) -> float:
        """Bearing géométrique vers la banane (rad). Utilisé uniquement par ALIGN."""
        if self._banana_xy is None:
            return 0.0
        to_banana = np.asarray(self._banana_xy, dtype=float) - np.asarray(thorax_xy, dtype=float)
        tbn = float(np.linalg.norm(to_banana))
        if tbn < 1e-9:
            return 0.0
        heading_xy, _ = self._get_body_frame_xy(sim)
        to_banana_n = to_banana / tbn
        cos_a = float(np.dot(heading_xy, to_banana_n))
        cross = float(heading_xy[0] * to_banana_n[1] - heading_xy[1] * to_banana_n[0])
        return float(np.arctan2(cross, cos_a))

    # ------------------------------------------------------------------
    def _compute_drives(self, sim) -> np.ndarray:
        if self._stopped:
            return np.array([0.0, 0.0])

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

        # ---- Head-collision recovery (3 phases, priorité maximale) ----
        head_drives = self._head_collision_step(sim, thorax_xy)
        if head_drives is not None:
            return head_drives

        # ---- Initial alignment ----
        if not self._align_done and self._banana_xy is not None:
            bearing = self._compute_bearing_to_banana(sim, thorax_xy)
            if (
                abs(bearing) < float(self.ALIGN_TOLERANCE_RAD)
                or self._align_decisions >= int(self.ALIGN_MAX_DECISIONS)
            ):
                if self.DEBUG:
                    print(
                        f"[ALIGN-DONE d={self._debug_decisions}] "
                        f"bearing={bearing:+.3f} after {self._align_decisions} dec",
                        flush=True,
                    )
                self._align_done = True
                self._align_transition_left = int(self.ALIGN_TRANSITION_DECISIONS)
            else:
                self._align_decisions += 1
                if bearing > 0:
                    return np.array([self.ALIGN_SPIN_SLOW, self.ALIGN_SPIN_FAST])
                return np.array([self.ALIGN_SPIN_FAST, self.ALIGN_SPIN_SLOW])

        # ---- Transition après ALIGN (resync CPG) ----
        if self._align_transition_left > 0:
            self._align_transition_left -= 1
            d = float(self.ALIGN_TRANSITION_DRIVE)
            return np.array([d, d])

        # ---- Stuck detection / escape (terrain) ----
        moved = float("inf")
        if self._last_xy is not None:
            moved = float(np.linalg.norm(thorax_xy - self._last_xy))
            if moved < float(self.STUCK_MOVE_EPS):
                self._stuck_decisions += 1
            else:
                self._stuck_decisions = 0
        self._last_xy = thorax_xy

        if self._escape_decisions_left > 0:
            self._escape_decisions_left -= 1
            if self._escape_dir > 0:
                return np.array([self.MAX_DRIVE, self.MIN_SIDE_DRIVE_TERRAIN])
            return np.array([self.MIN_SIDE_DRIVE_TERRAIN, self.MAX_DRIVE])

        if (
            self._enable_terrain
            and self._stuck_decisions >= int(self.STUCK_TRIGGER_DECISIONS)
        ):
            self._stuck_decisions = 0
            self._escape_decisions_left = int(self.ESCAPE_DURATION_DECISIONS)
            self._escape_dir *= -1
            if self._escape_dir > 0:
                return np.array([self.MAX_DRIVE, self.MIN_SIDE_DRIVE_TERRAIN])
            return np.array([self.MIN_SIDE_DRIVE_TERRAIN, self.MAX_DRIVE])

        # ---- SIDESTEP no-progress (anti-grass-blade jamming) ----
        sidestep_drives = self._sidestep_step(sim)
        if sidestep_drives is not None:
            return sidestep_drives

        # ---- Olfaction : tropotaxis + stop + scan/cast ----
        odor_lin = sim.get_olfaction(sim.fly.name)
        lp, rp, la, ra = odor_lin[:, 0]
        odor_l = self.PALP_WEIGHT * float(lp) + self.ANTENNA_WEIGHT * float(la)
        odor_r = self.PALP_WEIGHT * float(rp) + self.ANTENNA_WEIGHT * float(ra)
        odor_sum = odor_l + odor_r
        odor_diff = odor_l - odor_r
        mean_odor = 0.5 * odor_sum

        if mean_odor > self.STOP_ODOR_THRESHOLD:
            self._stopped = True
            return np.array([0.0, 0.0])

        self._odor_history.append(mean_odor)
        if len(self._odor_history) > int(self.ODOR_TREND_WINDOW):
            self._odor_history.pop(0)

        odor_decreasing = False
        if len(self._odor_history) >= int(self.ODOR_TREND_WINDOW):
            past = self._odor_history[0]
            if past > 1e-15:
                odor_decreasing = mean_odor < float(self.ODOR_TREND_LOSS_RATIO) * past

        signal_present = mean_odor > float(self.ODOR_MIN_SUM)
        if not signal_present:
            self._no_signal_decisions += 1
        else:
            self._no_signal_decisions = 0

        if self._scan_decisions_left <= 0 and (
            self._no_signal_decisions >= int(self.SCAN_TRIGGER_DECISIONS)
            or odor_decreasing
        ):
            self._scan_decisions_left = int(self.SCAN_DURATION_DECISIONS)
            self._scan_dir *= -1
            self._odor_history.clear()
            self._no_signal_decisions = 0

        if self._scan_decisions_left > 0:
            self._scan_decisions_left -= 1
            min_side = (
                self.MIN_SIDE_DRIVE_TERRAIN
                if self._enable_terrain
                else self.MIN_SIDE_DRIVE
            )
            max_drive = (
                self.MAX_DRIVE_TERRAIN if self._enable_terrain else self.MAX_DRIVE
            )
            if self._scan_dir > 0:
                return np.array([max_drive, min_side])
            return np.array([min_side, max_drive])

        # Tropotaxis canonique (week4 : gain=-500, bias²)
        if mean_odor > 0.0:
            bias_raw = float(self.ODOR_STEER_GAIN) * (odor_diff / mean_odor)
        else:
            bias_raw = 0.0
        target_bias = float(np.tanh(bias_raw * bias_raw)) * float(np.sign(bias_raw))
        self._last_target_bias = target_bias

        # ---- Vision silhouette (Level 2+) ----
        obs_size, obs_x = 0.0, 0.0
        if self.VISION_ENABLE and self._enable_grass:
            obs_size, obs_x = self._vision_step(sim)

        # ---- Champ de répulsion (= remplace AVOID FSM) ----
        # bias = target_bias + REPULSION_GAIN * (-obs_x) * obs_size
        # obs_x négatif (piques à gauche) → bias positif → steer droite.
        # obs_size = magnitude scalaire (proximité). AVOID FSM seuil tellement
        # haut qu'il ne se déclenche jamais ; on garde le code pour le debug
        # mais en pratique on est toujours en GO + repulsion.
        repulsion = float(self.REPULSION_GAIN) * (-obs_x) * obs_size
        self._update_avoid_state(obs_size, obs_x)

        if self._avoid_left > 0:
            try:
                _, _, _upright = self._get_orientation(sim)
            except Exception:
                _upright = 1.0
            bias, base_drive, sub_mode = self._avoid_command(
                target_bias, obs_x, obs_size, _upright
            )
        else:
            bias = float(target_bias) + repulsion
            base_drive = float(self.BASE_DRIVE_FAST)
            sub_mode = "GO"

        # ---- Terrain (slope) ----
        turn_mod = self.TURN_MOD
        slope_mag = 0.0
        if self._enable_terrain:
            slope_forward, slope_lateral, slope_mag = self._get_slope_signals(sim)
            downhill = max(0.0, -slope_forward)
            steep_weight = 0.25 + 0.75 * downhill
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

        # Slope-adaptive AVOID : sur pente raide, blend pour éviter flip
        if self._avoid_left > 0 and self._enable_terrain and slope_mag > 0.20:
            blend = 0.50 * float(np.clip((slope_mag - 0.20) / 0.20, 0.0, 1.0))
            mean_d = float(np.mean(drives))
            drives = drives * (1.0 - blend) + mean_d * blend
            drives = np.clip(drives, 0.0, max_drive)

        if (
            self.DEBUG
            and self._debug_decisions <= self.DEBUG_MAX_DECISIONS
            and (self._debug_decisions % self.DEBUG_EVERY_DECISIONS == 0)
        ):
            dist_str = f"{dist_to_banana:.2f}" if dist_to_banana is not None else "?"
            print(
                f"[dbg d={self._debug_decisions:4d}] mode={'AVOID' if self._avoid_left>0 else 'GO':5s} "
                f"sub={sub_mode:5s} obs_x={obs_x:+.3f} obs_sz={obs_size:.3f} "
                f"odor_mean={mean_odor:.3e} target={target_bias:+.3f} dist={dist_str} "
                f"bias={bias:+.3f} drives=({drives[0]:.2f},{drives[1]:.2f})",
                flush=True,
            )
        return drives

    # ------------------------------------------------------------------
    # Head collision FSM (3 phases : BACKUP → TURN → BYPASS)
    # ------------------------------------------------------------------
    def _head_collision_step(self, sim, thorax_xy: np.ndarray) -> "np.ndarray | None":
        """Réflexe collision tête. Retourne drives ou None."""
        # Phase 1 : BACKUP
        if self._head_backup_left > 0:
            self._head_backup_left -= 1
            if self._head_backup_left == 0:
                self._head_turn_left = int(self.HEAD_TURN_DECISIONS)
                if self.DEBUG:
                    print(
                        f"[HEAD-TURN d={self._debug_decisions}] "
                        f"dir={self._head_maneuver_dir:+d}",
                        flush=True,
                    )
            backup = float(self.HEAD_BACKUP_DRIVE)
            return np.array([backup, backup])

        # Phase 2 : TURN vers la banane
        if self._head_turn_left > 0:
            self._head_turn_left -= 1
            if self._head_turn_left == 0:
                self._head_bypass_left = int(self.HEAD_BYPASS_DECISIONS)
                if self.DEBUG:
                    print(
                        f"[HEAD-BYPASS d={self._debug_decisions}] "
                        f"dir={self._head_maneuver_dir:+d}",
                        flush=True,
                    )
            if self._head_maneuver_dir > 0:
                return np.array([self.HEAD_TURN_DRIVE_FAST, self.HEAD_TURN_DRIVE_SLOW])
            return np.array([self.HEAD_TURN_DRIVE_SLOW, self.HEAD_TURN_DRIVE_FAST])

        # Phase 3 : BYPASS
        if self._head_bypass_left > 0:
            self._head_bypass_left -= 1
            if self._head_bypass_left == 0:
                self._head_collision_cd = int(self.HEAD_COLLISION_COOLDOWN)
            if self._head_maneuver_dir > 0:
                return np.array([self.HEAD_BYPASS_DRIVE_FORWARD, self.HEAD_BYPASS_DRIVE_SIDE])
            return np.array([self.HEAD_BYPASS_DRIVE_SIDE, self.HEAD_BYPASS_DRIVE_FORWARD])

        # Cooldown ou détection
        if self._head_collision_cd > 0:
            self._head_collision_cd -= 1
            return None

        if not (
            self.HEAD_COLLISION_ENABLE
            and self._enable_grass
            and self._head_body_id is not None
        ):
            return None

        head_force = float(self._head_force_max)
        if head_force <= float(self.HEAD_COLLISION_FORCE_THRESH):
            return None

        maneuver_dir = +1
        if self._banana_xy is not None:
            try:
                xmat_hc = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
                heading_xy = xmat_hc[:2, 0]
                hn = float(np.linalg.norm(heading_xy))
                to_banana = self._banana_xy - thorax_xy
                tbn = float(np.linalg.norm(to_banana))
                if hn > 1e-9 and tbn > 1e-9:
                    heading_xy = heading_xy / hn
                    to_banana_n = to_banana / tbn
                    cross = float(
                        heading_xy[0] * to_banana_n[1]
                        - heading_xy[1] * to_banana_n[0]
                    )
                    # cross > 0 = banane à gauche → maneuver_dir = -1 (pivote gauche)
                    maneuver_dir = -1 if cross > 0 else +1
            except Exception:
                pass
        self._head_maneuver_dir = maneuver_dir

        if self.DEBUG:
            side = "gauche" if maneuver_dir < 0 else "droite"
            print(
                f"[HEAD-COLLISION d={self._debug_decisions}] "
                f"F={head_force:.1f}N -> BACKUP (banane {side})",
                flush=True,
            )

        # Reset des autres FSM pour ne pas interférer
        self._head_backup_left = int(self.HEAD_BACKUP_DECISIONS)
        self._avoid_left = 0
        self._avoid_clear = 0
        self._latched_obs_x = 0.0
        self._sidestep_decisions_left = 0
        self._stuck_decisions = 0

        backup = float(self.HEAD_BACKUP_DRIVE)
        return np.array([backup, backup])

    # ------------------------------------------------------------------
    # SIDESTEP : recul + side-step quand pas de progression
    # ------------------------------------------------------------------
    def _sidestep_step(self, sim) -> "np.ndarray | None":
        # Continuation d'un sidestep en cours
        if self._sidestep_decisions_left > 0:
            self._sidestep_decisions_left -= 1
            n_left = self._sidestep_decisions_left
            if n_left >= int(self.SIDESTEP_TURN_DECISIONS):
                d = float(self.SIDESTEP_BACKUP_DRIVE)
                return np.array([d, d])
            if n_left == 0:
                self._sidestep_cooldown = int(self.NOPROG_WINDOW)
                self._dist_history.clear()
            if self._sidestep_dir > 0:
                return np.array([self.SIDESTEP_DRIVE_FAST, self.SIDESTEP_DRIVE_SLOW])
            return np.array([self.SIDESTEP_DRIVE_SLOW, self.SIDESTEP_DRIVE_FAST])

        if not (self._enable_grass and self._banana_xy is not None):
            return None

        self._dist_history.append(float(self._last_dist_to_banana))
        if len(self._dist_history) > int(self.NOPROG_WINDOW):
            self._dist_history.pop(0)

        try:
            xmat_pre = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
            roll_pre = float(xmat_pre[2, 1])
            upright_pre = float(xmat_pre[2, 2])
        except Exception:
            roll_pre, upright_pre = 0.0, 1.0
        roll_alarm = (
            abs(roll_pre) > float(self.ROLL_TRIGGER_THRESH)
            and upright_pre > float(self.TERRAIN_FLIP_WEAK_UPRIGHT)
        )
        if roll_alarm:
            self._roll_high_count += 1
        else:
            self._roll_high_count = 0

        trigger_now = False
        trigger_prev_diff = 0.0

        if self._roll_high_count >= int(self.ROLL_TRIGGER_HOLD):
            trigger_now = True
            trigger_prev_diff = float(self._drives[0]) - float(self._drives[1])

        if self._sidestep_cooldown > 0:
            self._sidestep_cooldown -= 1

        if (
            not trigger_now
            and self._sidestep_cooldown == 0
            and len(self._dist_history) >= int(self.NOPROG_WINDOW)
        ):
            progress = self._dist_history[0] - self._dist_history[-1]
            if progress < float(self.NOPROG_MIN_DELTA):
                trigger_now = True
                trigger_prev_diff = float(self._drives[0]) - float(self._drives[1])

        if not trigger_now:
            return None

        if trigger_prev_diff > 0.05:
            self._sidestep_dir = +1
        elif trigger_prev_diff < -0.05:
            self._sidestep_dir = -1
        else:
            self._sidestep_dir = -self._sidestep_dir or +1

        self._sidestep_decisions_left = (
            int(self.SIDESTEP_BACKUP_DECISIONS)
            + int(self.SIDESTEP_TURN_DECISIONS)
        )
        self._roll_high_count = 0
        if self.DEBUG:
            print(
                f"[SIDESTEP d={self._debug_decisions}] dir={self._sidestep_dir:+d} "
                f"(roll={roll_pre:+.2f})",
                flush=True,
            )
        d = float(self.SIDESTEP_BACKUP_DRIVE)
        return np.array([d, d])

    # ------------------------------------------------------------------
    # Vision : silhouette ommatidia → (obs_size, obs_x) EMA
    # ------------------------------------------------------------------
    def _color_masks(self, eye_img01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calcule (is_green, is_sky) sur une image RGB normalisée [0, 1]."""
        r = eye_img01[..., 0]
        g = eye_img01[..., 1]
        b = eye_img01[..., 2]
        delta = float(self.VIS_GREEN_DELTA)
        gmin = float(self.VIS_GREEN_MIN)
        is_green = ((g - r) > delta) & ((g - b) > delta) & (g > gmin)
        # Sky strict : VRAI bleu (B - max(R, G) > margin) OU vraiment blanc/gris
        # achromatique brillant. Élimine les tans/bruns du test sky.
        blue_strict = (b - np.maximum(r, g)) > float(self.VIS_SKY_BLUE_STRICT)
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        sum_int = r + g + b
        white_strict = ((mx - mn) < float(self.VIS_SKY_WHITE_SPREAD)) & (
            sum_int > float(self.VIS_SKY_WHITE_MIN_SUM)
        )
        is_sky = (blue_strict | white_strict) & (~is_green)
        return is_green, is_sky

    def _eye_spike_mask(
        self, eye_img: np.ndarray, is_left_eye: bool
    ) -> "tuple[np.ndarray, tuple[int, int, int, int]] | None":
        """Retourne (spike_mask, (r0, r1, c0, c1)) ou None.

        spike_mask : bool (h_roi, w_roi). True = pixel appartenant à un blob
        green valide (TALL et VERTICAL — aspect h/w >= MIN_ASPECT) avec du
        ciel directement au-dessus du sommet du blob.
        """
        a = np.asarray(eye_img, dtype=np.float32)
        if a.ndim != 3 or a.shape[-1] < 3:
            return None
        if a.max() > 1.5:
            a = a / 255.0
        h, w = a.shape[:2]
        r1 = int(h * float(self.VIS_UPPER_FRAC))
        frontal = float(self.VIS_FRONTAL_FRAC)
        if is_left_eye:
            c0, c1 = int(w * (1.0 - frontal)), w
        else:
            c0, c1 = 0, int(w * frontal)
        roi = a[0:r1, c0:c1, :3]
        if roi.shape[0] < 2 or roi.shape[1] < 2:
            return None
        is_green, is_sky = self._color_masks(roi)
        nrows, ncols = is_green.shape
        if not is_green.any():
            return np.zeros_like(is_green), (0, r1, c0, c1)

        # Étape 1 : per-row sky-borders. Un pixel green compte si il existe
        # un pixel sky à <= K pixels à GAUCHE OU à DROITE dans sa ligne.
        # OR (pas AND) pour capter les BORDS de piques larges/proches.
        # → rejette le sol vert (aucun sky dans la même ligne pour le sol).
        K = int(self.VIS_SPIKE_MAX_WIDTH)
        cum = np.zeros((nrows, ncols + 1), dtype=np.int32)
        cum[:, 1:] = np.cumsum(is_sky.astype(np.int32), axis=1)
        c_idx = np.arange(ncols)
        a_left = np.maximum(0, c_idx - K)
        has_sky_left = (cum[:, c_idx] - cum[:, a_left]) > 0
        b_right = np.minimum(ncols, c_idx + K + 1)
        a_right = np.minimum(ncols, c_idx + 1)
        has_sky_right = (cum[:, b_right] - cum[:, a_right]) > 0
        candidate = is_green & (has_sky_left | has_sky_right)
        if not candidate.any():
            return np.zeros_like(is_green), (0, r1, c0, c1)

        # Étape 2 : CC sur candidate + filtre par hauteur. Rejette les slivers
        # horizontaux d'1 ligne (transitions horizon) qui auraient pu passer.
        min_h = int(self.VIS_SPIKE_MIN_HEIGHT)
        spike_mask = np.zeros_like(is_green)
        if _SCIPY_NDIMAGE is None:
            # Fallback sans scipy : on garde candidate tel quel
            return candidate, (0, r1, c0, c1)
        labels, n_blobs = _SCIPY_NDIMAGE.label(candidate)
        if n_blobs == 0:
            return spike_mask, (0, r1, c0, c1)
        slices = _SCIPY_NDIMAGE.find_objects(labels)
        for i, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            r_sl, c_sl = sl
            hh = r_sl.stop - r_sl.start
            if hh < min_h:
                continue
            spike_mask[r_sl, c_sl] |= (labels[r_sl, c_sl] == i)
        return spike_mask, (0, r1, c0, c1)

    def _eye_spike_counts(
        self, eye_img: np.ndarray, is_left_eye: bool
    ) -> tuple[int, int]:
        """Retourne (total_count, central_count).

        - total : tous les spike pixels dans la ROI (utilisé pour obs_x direction)
        - central : seuls les spike pixels dans la bande centrale FRONTALE de la
          ROI (= directement devant la mouche). Utilisé pour déclencher AVOID.
        """
        result = self._eye_spike_mask(eye_img, is_left_eye)
        if result is None:
            return 0, 0
        mask = result[0]
        if mask.size == 0:
            return 0, 0
        total = int(mask.sum())
        w_roi = mask.shape[1]
        cf = float(self.VIS_CENTRAL_FRAC)
        central_w = max(1, int(w_roi * cf))
        if is_left_eye:
            # ROI de l'œil gauche couvre cols [(1-frontal_frac)*W .. W].
            # Le côté nasal (forward-looking) = cols hautes = bord DROIT de la ROI.
            central_mask = mask[:, -central_w:]
        else:
            # Œil droit : nasal = bord GAUCHE de la ROI.
            central_mask = mask[:, :central_w]
        central = int(central_mask.sum())
        return total, central

    def _vision_step(self, sim) -> tuple[float, float]:
        """Sky-vert-sky sur image RGB brute → (obs_size, obs_x).

        Pour chaque ligne de pixels de la ROI frontale-haute :
          pixel = silhouette si vert ET (sky à gauche dans la ligne)
          ET (sky à droite dans la ligne).

        Inspiré de spike-ciel-sky-border, vectorisé via maximum.accumulate.
        """
        if self._debug_decisions < int(self.VIS_STARTUP_DELAY_DECISIONS):
            return float(self._vis_obs_size), float(self._vis_obs_x)
        try:
            frames = sim.get_raw_vision(sim.fly.name)
        except Exception:
            return float(self._vis_obs_size), float(self._vis_obs_x)
        if frames is None or len(frames) == 0:
            return float(self._vis_obs_size), float(self._vis_obs_x)

        tot_L, _ = self._eye_spike_counts(frames[0], is_left_eye=True)
        tot_R, _ = self._eye_spike_counts(
            frames[1] if len(frames) > 1 else frames[0], is_left_eye=False
        )
        # Pour le champ de répulsion :
        # - obs_size = magnitude totale (proximité × nombre de piques)
        # - obs_x    = position angulaire moyenne des piques en frame body
        total_full = tot_L + tot_R
        if total_full < 1:
            size_raw = 0.0
            x_raw = 0.0
        else:
            size_raw = min(1.0, total_full / float(self.BLADE_COUNT_URGENT))
            x_raw = float(tot_R - tot_L) / float(total_full)

        ema = float(self.VIS_EMA)
        self._vis_obs_size = ema * self._vis_obs_size + (1.0 - ema) * size_raw
        self._vis_obs_x = ema * self._vis_obs_x + (1.0 - ema) * x_raw
        return float(self._vis_obs_size), float(self._vis_obs_x)

    # ------------------------------------------------------------------
    # AVOID FSM (arc-dodge)
    # ------------------------------------------------------------------
    def _schedule_realign_after_long_avoid(self) -> None:
        """Re-déclenche ALIGN si l'épisode AVOID était long."""
        ticks = int(self._avoid_session_ticks)
        self._avoid_session_ticks = 0
        if not (self.ALIGN_ENABLE and self.ALIGN_AFTER_AVOID_ENABLE):
            return
        if ticks >= int(self.ALIGN_AFTER_AVOID_MIN_DECISIONS):
            self._align_done = False
            self._align_decisions = 0

    def _update_avoid_state(self, obs_size: float, obs_x: float) -> None:
        """Trigger / exit AVOID, latch obs_x à l'entrée."""
        close_to_banana = (
            self._last_dist_to_banana is not None
            and self._last_dist_to_banana < float(self.AVOID_DISABLE_CLOSE_DIST)
        )
        if close_to_banana:
            if self._avoid_left > 0:
                self._avoid_left = 0
                self._avoid_clear = 0
                self._latched_obs_x = 0.0
            return

        on = obs_size >= float(self.AVOID_SIZE_ON)
        off = obs_size <= float(self.AVOID_SIZE_OFF)

        if self._avoid_left > 0:
            self._avoid_session_ticks += 1
            self._avoid_min_left = max(0, self._avoid_min_left - 1)
            if off:
                self._avoid_clear += 1
            else:
                self._avoid_clear = 0

            if (
                obs_size >= float(self.AVOID_SIZE_ON)
                and abs(obs_x - self._latched_obs_x) > float(self.AVOID_REFRESH_DELTA)
            ):
                self._latched_obs_x = float(obs_x)

            if (
                self._avoid_min_left <= 0
                and self._avoid_clear >= int(self.AVOID_CLEAR_DECISIONS)
            ):
                dist_str = (
                    f"{self._last_dist_to_banana:.1f}"
                    if self._last_dist_to_banana is not None
                    else "?"
                )
                print(
                    f"[OBS-] cleared after {self._avoid_session_ticks} ticks dist={dist_str}",
                    flush=True,
                )
                self._schedule_realign_after_long_avoid()
                self._avoid_left = 0
                self._avoid_clear = 0
                self._latched_obs_x = 0.0
                return
            self._avoid_left += 1
        else:
            if not on:
                return
            self._latched_obs_x = float(obs_x)
            self._avoid_session_ticks = 0
            self._avoid_left = 1
            self._avoid_min_left = int(self.AVOID_MIN_DURATION)
            self._avoid_clear = 0
            dist_str = (
                f"{self._last_dist_to_banana:.1f}"
                if self._last_dist_to_banana is not None
                else "?"
            )
            print(
                f"[OBS+] dist={dist_str} obs_x={obs_x:+.3f} obs_sz={obs_size:.3f}",
                flush=True,
            )

    def _avoid_command(
        self,
        target_bias: float,
        live_obs_x: float,
        live_obs_size: float,
        uprightness: float = 1.0,
    ) -> tuple[float, float, str]:
        """Esquive arc linéaire : centralité × taille × tilt-recovery."""
        live_x = float(np.clip(live_obs_x, -1.0, 1.0))
        latched = float(np.clip(self._latched_obs_x, -1.0, 1.0))

        centrality = 1.0 - live_x * live_x
        size_factor = float(np.clip(live_obs_size / float(self.AVOID_SIZE_MED), 0.8, 1.5))
        tilt_factor = float(np.clip(uprightness / 0.95, 0.3, 1.0))
        dodge_mag = float(self.AVOID_TURN_MAX) * centrality * size_factor * tilt_factor

        eps = float(self.AVOID_CENTER_EPS)
        if abs(latched) > eps:
            dodge_dir = -1.0 if latched > 0.0 else 1.0
        elif abs(live_x) > eps and live_obs_size >= float(self.AVOID_SIZE_ON):
            dodge_dir = -1.0 if live_x > 0.0 else 1.0
        else:
            if abs(target_bias) > 1e-9:
                dodge_dir = 1.0 if target_bias > 0.0 else -1.0
            else:
                dodge_dir = 1.0

        obstacle_bias = dodge_dir * dodge_mag
        bias = obstacle_bias + float(self.AVOID_BANANA_BLEND) * float(target_bias)
        bias = float(np.clip(bias, -float(self.VIS_TURN_MAX), float(self.VIS_TURN_MAX)))
        base = float(self.BASE_DRIVE_FAST) * float(self.AVOID_SPEED_FRAC)
        return bias, base, "ARC"

    # ------------------------------------------------------------------
    def compute_vision_debug_overlay(self, sim: MiniprojectSimulation):
        """Overlay debug : silhouette pixels en rouge sur l'image brute."""
        if not self._enable_grass:
            return None
        try:
            frames = sim.get_raw_vision(sim.fly.name)
        except Exception:
            return None
        if frames is None or len(frames) == 0:
            return None
        outputs = []
        for eye_idx in range(min(2, len(frames))):
            a = np.asarray(frames[eye_idx], dtype=np.float32)
            if a.ndim != 3 or a.shape[-1] < 3:
                continue
            if a.max() > 1.5:
                a = a / 255.0
            h, w = a.shape[:2]
            is_left_eye = (eye_idx == 0)
            result = self._eye_spike_mask(frames[eye_idx], is_left_eye)
            if result is None:
                continue
            spike, (r0, r1, c0, c1) = result
            # Masque central (= déclenche AVOID) vs périphérique (info seule)
            w_roi = spike.shape[1]
            cf = float(self.VIS_CENTRAL_FRAC)
            central_w = max(1, int(w_roi * cf))
            central_band = np.zeros_like(spike)
            if is_left_eye:
                central_band[:, -central_w:] = True
            else:
                central_band[:, :central_w] = True
            spike_central = spike & central_band
            spike_periph = spike & ~central_band

            base = (a * 0.55 * 255.0).astype(np.uint8)
            roi_view = base[r0:r1, c0:c1].copy()
            # Rouge plein = central (déclenche AVOID)
            roi_view[spike_central] = [255, 0, 0]
            # Orange = périphérique (info seule, donne obs_x)
            roi_view[spike_periph] = [255, 165, 0]
            base[r0:r1, c0:c1] = roi_view
            cyan = np.array([0, 220, 220], dtype=np.uint8)
            magenta = np.array([220, 0, 220], dtype=np.uint8)
            base[r0:r0 + 1, c0:c1] = cyan
            if r1 - 1 < h:
                base[r1 - 1:r1, c0:c1] = cyan
            base[r0:r1, c0:c0 + 1] = cyan
            if c1 - 1 < w:
                base[r0:r1, c1 - 1:c1] = cyan
            # Cadre magenta autour de la bande centrale (= zone "in path")
            if is_left_eye:
                cc0 = c1 - central_w
                cc1 = c1
            else:
                cc0 = c0
                cc1 = c0 + central_w
            base[r0:r1, cc0:cc0 + 1] = magenta
            if cc1 - 1 < w:
                base[r0:r1, cc1 - 1:cc1] = magenta
            outputs.append(base)
        if not outputs:
            return None
        try:
            self._vis_debug_overlay = np.concatenate(outputs, axis=1)
            return self._vis_debug_overlay
        except Exception:
            return None
