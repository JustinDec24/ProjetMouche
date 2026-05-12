import numpy as np

from miniproject.simulation import MiniprojectSimulation

try:
    from scipy import ndimage as _SCIPY_NDIMAGE
except ImportError:
    _SCIPY_NDIMAGE = None

_SPIKE_CONN = np.ones((3, 3), dtype=np.int32)


class Controller:
    """UNION : cap banane + vision panorama + FSM AVOID + anti-jam multi-niveau.

    Hybride combinant les forces de spike-ciel-sky-border et MERGE pour
    maximiser le taux de succès sur L2.

    Priorité dans _compute_drives :
      1. Stop (dist banane <= STOP_DIST OU odor seuil)
      2. Terrain stuck/escape (mouche immobilisée par mesh terrain)
      3. JAM_REFLEX : anti-stuck rapide non-visuel (mouvement = 0)
      4. SIDESTEP_NOPROG : anti-stuck par PROGRESSION distance (mouvement
         latéral mais pas de progrès vers banane) — port depuis MERGE
      5. ALIGN initial (pivot sur place vers banane via bearing géométrique)
      6. AVOID FSM : arc-dodge si vision détecte pique
      7. GO normal : cap banane + GO_SWEEP si signal vision ambigu

    Vision : panorama bi-oculaire (cols nasales de chaque œil concaténées).
    Détection sky-grass-sky horizontal par ligne + CC filter (height, aspect).
    Outputs (obs_size, obs_x) ∈ [0, 1] × [-1, +1].

    AVOID arc-dodge :
      dodge_mag  = AVOID_TURN_MAX × centralité × size_factor × tilt_factor
      dodge_dir  = -sign(latched_obs_x) avec tiebreak côté banane
      bias       = dodge × dodge_dir + AVOID_BANANA_BLEND × target_bias

    PAS de head-collision recovery (cause des flips pathologiques sur terrain
    pentu — voir seed 513 sur MERGE qui finissait à dist=1264 m).
    """

    # --- scheduling ---
    DECISION_INTERVAL_S = 0.025  # 25 ms = 40 décisions/s

    # --- olfaction (stop only) ---
    PALP_WEIGHT = 9
    ANTENNA_WEIGHT = 1
    EPS_ODOR = 1e-12
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
    AVOID_TURN_MOD = 1.0  # plein différentiel pendant AVOID

    # --- target steering (cap banane) ---
    TARGET_STEER_GAIN = 3.0
    TARGET_STEER_GAIN_CLOSE = 6.0
    TARGET_STEER_CLOSE_DIST = 24.0
    TARGET_STEER_BIAS_SCALE = 0.25

    # --- terrain (Level 1+) ---
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

    # --- anti-roll grip ---
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
    TERRAIN_GRIP_FORCE = 6.0    # bumped 4.0→6.0 : évite que la mouche tombe sur le côté
    WIND_GRIP_FORCE = 10.0

    # --- orientation safety ---
    TERRAIN_UPRIGHT_TILT_WARN = 0.48
    TERRAIN_TILT_RESET_HOLD = 26
    TERRAIN_FLIP_WEAK_UPRIGHT = 0.12
    TERRAIN_FLIP_RESET_HOLD = 34

    # --- jam reflex (anti-stuck, non-visuel) ---
    JAM_ENABLE = True
    JAM_MOVE_EPS = 2.0e-2
    JAM_MIN_DRIVE = 0.70
    JAM_TRIGGER_DECISIONS = 4
    JAM_LATCH_DECISIONS = 14

    # --- SIDESTEP no-progress (anti-jam complémentaire — détecte le stuck
    # quand la mouche bouge latéralement sans progresser vers la banane).
    # JAM_REFLEX rate ces cas car il y a du mouvement physique ; SIDESTEP
    # se base sur la distance à la banane sur une fenêtre temporelle.
    SIDESTEP_ENABLE = True
    NOPROG_WINDOW = 30              # ~750 ms de fenêtre d'analyse
    NOPROG_MIN_DELTA = 0.4          # progression < 0.4 m sur la fenêtre → stuck
    SIDESTEP_BACKUP_DECISIONS = 5
    SIDESTEP_TURN_DECISIONS = 14
    SIDESTEP_BACKUP_DRIVE = -0.6
    SIDESTEP_DRIVE_FAST = 1.80
    SIDESTEP_DRIVE_SLOW = 0.30
    SIDESTEP_DISABLE_CLOSE_DIST = 6.0   # < 6 m banane : on ne sidestep plus
    ROLL_TRIGGER_THRESH = 0.30
    ROLL_TRIGGER_HOLD = 3

    # --- VISION (Level 2+) ---------------------------------------------------
    # Activée seulement si _enable_grass=True. Détection panoramique
    # bi-oculaire fusionnée en un seul vecteur de colonnes couvrant le frontal-large.
    VISION_ENABLE = True

    # ROI vertical : bande horizon (haut = sky/cloud, bas = sol/grass)
    VIS_ROI_R0 = 0.05
    VIS_ROI_R1 = 0.78

    # ROI horizontal par œil : zone fronto-nasale élargie à 70%.
    # Œil gauche : moitié droite de l'image (cols hautes = côté nasal = vers l'avant).
    # Œil droit  : moitié gauche de l'image (cols basses = côté nasal).
    # Quand on concatène [left_ROI | right_ROI], le centre du panorama tombe
    # exactement sur le frontal binoculaire = "pile en face".
    VIS_ROI_C0_LEFT = 0.30
    VIS_ROI_C1_LEFT = 1.00
    VIS_ROI_C0_RIGHT = 0.00
    VIS_ROI_C1_RIGHT = 0.70

    # Couleurs (grass + sky/cloud unifiés)
    VIS_GRASS_GREEN_DELTA = 0.20
    VIS_GRASS_GREEN_MIN = 0.45
    VIS_SKY_BLUE_MARGIN = 0.02
    VIS_CLOUD_LUM_MIN = 0.20      # plus permissif (catches dark blue sky)
    VIS_CLOUD_LUM_MAX = 1.0
    VIS_CLOUD_RGB_SPREAD_MAX = 0.22  # plus permissif (catches tinted clouds)

    # Filtre blob CC : un vrai pic est haut+étroit
    VIS_SPIKE_MIN_AREA_PX = 1600
    VIS_SPIKE_MIN_HEIGHT_PX = 40
    VIS_SPIKE_MIN_ASPECT = 1.2

    # EMA pour stabilité du signal
    VIS_EMA = 0.55

    # --- AVOID FSM ---
    # obs_size = densité au pic (col la plus bloquée / hauteur ROI) ∈ [0, 1].
    # Un brin d'herbe typique en face donne peak_smoothed ≈ 20-30 sur h_p ≈ 100,
    # soit obs_size ≈ 0.20-0.30. Bruit < 0.05.
    AVOID_SIZE_ON = 0.10         # taille obstacle pour entrer AVOID
    AVOID_SIZE_OFF = 0.04        # taille obstacle pour sortir AVOID
    AVOID_SIZE_MED = 0.20        # taille "obstacle moyen" → dodge normal (1.0×)
    AVOID_MIN_DURATION = 4
    AVOID_CLEAR_DECISIONS = 1
    AVOID_DISABLE_CLOSE_DIST = 8.0  # < 8 m de la banane : on fonce

    # Esquive linéaire en centralité
    AVOID_TURN_MAX = 1.0
    AVOID_CENTER_EPS = 0.05      # |obs_x| < eps → tiebreak banane
    AVOID_BANANA_BLEND = 0.5     # mélange cap banane pendant AVOID
    AVOID_REFRESH_DELTA = 0.30   # ré-écriture latch si nouveau obs_x très différent

    # Saturation universelle bias (anti-violent turn)
    VIS_TURN_MAX = 3.0

    # Vitesse pendant AVOID (fraction de BASE_DRIVE_FAST)
    AVOID_SPEED_FRAC = 0.60   # ralenti 0.65→0.60

    # --- GO sweep (balayage gauche-droite pour détecter le frontal) ---
    # Pendant GO la mouche oscille latéralement : son champ visuel balaye
    # le frontal et un obstacle pile-en-face entre alternativement dans la
    # zone détectable (au lieu de rester dans l'angle mort frontal).
    GO_SWEEP_ENABLE = True
    GO_SWEEP_AMPLITUDE = 0.20        # ampli du bias ajouté en GO
    GO_SWEEP_PERIOD_S = 1.5          # période complète sin
    GO_SWEEP_DISABLE_CLOSE_DIST = 10.0   # désactivé < 10 m (sprint final propre)
    # Sweep conditionnel : activé seulement quand obs_size dans [LOW, AVOID_SIZE_ON].
    # → zone "ambiguë" : la mouche voit qqch mais pas assez pour déclencher AVOID,
    #   le sweep aide à confirmer/lever l'ambiguïté en balayant le frontal.
    # → si obs_size ≈ 0 (vision parfaitement propre, ex. seeds 10/17), pas de sweep
    #   → trajet droit préservé.
    GO_SWEEP_OBS_SIZE_MIN = 0.02

    # --- Initial alignment ---
    ALIGN_INITIAL_ENABLE = True
    ALIGN_BEARING_OK = 0.20
    ALIGN_MAX_DECISIONS = 60
    ALIGN_MAX_DRIVE_TERRAIN = 2.00
    ALIGN_MIN_SIDE_TERRAIN = 0.25

    # Réalignement après AVOID prolongé
    ALIGN_AFTER_AVOID_ENABLE = True
    ALIGN_AFTER_AVOID_MIN_DECISIONS = 8

    # --- debugging ---
    DEBUG = True
    DEBUG_EVERY_DECISIONS = 2
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
        self._contact_body_ids = sim._internal_contact_body_segment_ids_by_fly[sim.fly.name]

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

        # Vision state (EMA)
        self._vis_obs_size = 0.0       # densité de pixels-pic dans le panorama
        self._vis_obs_x = 0.0          # position centroïde ∈ [-1, +1]
        self._vis_debug_overlay = None

        # AVOID FSM state
        self._avoid_left = 0
        self._avoid_min_left = 0
        self._avoid_clear = 0
        self._latched_obs_x = 0.0
        self._avoid_session_ticks = 0

        # Jam reflex
        self._jam_left = 0
        self._jam_dir = 1

        # SIDESTEP no-progress state
        self._dist_history: list[float] = []
        self._sidestep_decisions_left = 0
        self._sidestep_dir = +1
        self._sidestep_cooldown = 0
        self._roll_high_count = 0

        # Initial alignment state
        self._align_done = False
        self._align_dir = 0.0
        self._align_left = 0

        try:
            self._banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
        except Exception:
            self._banana_xy = None

        # Lift fly at spawn to free legs from terrain mesh.
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
        if is_decision_step:
            self._drives = self._compute_drives(sim)
            self._debug_decisions = self._step_count // self._decision_every
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
                        float(self.TILT_LEAN_ROLL_FULL)
                        - float(self.TILT_LEAN_ROLL_ON),
                    ),
                )
                offset = (
                    float(self.TILT_LEAN_SIGN)
                    * float(self.TILT_LEAN_GAIN)
                    * ramp
                )
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
        """(pitch, roll_ind, uprightness) à partir de la matrice thorax."""
        xmat = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
        pitch = np.arcsin(np.clip(xmat[2, 0], -1.0, 1.0))
        return pitch, xmat[2, 1], xmat[2, 2]

    def _get_body_frame_xy(self, sim) -> tuple[np.ndarray, np.ndarray]:
        """(heading_xy, lateral_xy) unitaires depuis la matrice thorax."""
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
        """(slope_forward, slope_lateral, slope_mag) depuis la normale terrain."""
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
        """Steering bias vers la banane (cap banane).

        Convention :
          bearing > 0 → banane à GAUCHE  → target_bias < 0 → drives=[min,max] → tourne à GAUCHE
          bearing < 0 → banane à DROITE → target_bias > 0 → drives=[max,min] → tourne à DROITE
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
        # Banane derrière + sur le côté → demi-tour court
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
        if self._stopped:
            return np.array([0.0, 0.0])

        if self._banana_xy is None:
            try:
                self._banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
            except Exception:
                self._banana_xy = None

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

        # ---- stuck detection / escape (terrain trap) ----
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

        # ---- Jam reflex ----
        if self.JAM_ENABLE:
            max_drive_cmd = (
                float(np.max(self._drives))
                if getattr(self, "_drives", None) is not None
                else 0.0
            )
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

        # ---- SIDESTEP no-progress (anti-jam complémentaire) ----
        # Détecte le stuck quand la mouche bouge latéralement (donc JAM ne
        # fire pas) mais ne progresse pas vers la banane. Force backup+pivot.
        sidestep_drives = self._sidestep_step(sim)
        if sidestep_drives is not None:
            return sidestep_drives

        # ---- Stop sur banane (olfaction) ----
        odor_lin = sim.get_olfaction(sim.fly.name)
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

        # ---- Vision panorama (Level 2+) ----
        obs_size = 0.0
        obs_x = 0.0
        if self.VISION_ENABLE and self._enable_grass:
            obs_size, obs_x = self._vision_step(sim)

        # ---- Initial alignment ----
        if (
            self.ALIGN_INITIAL_ENABLE
            and not self._align_done
            and self._banana_xy is not None
        ):
            if self._align_dir == 0.0:
                self._align_dir = 1.0 if bearing > 0.0 else -1.0
                self._align_left = int(self.ALIGN_MAX_DECISIONS)

            aligned = abs(bearing) <= float(self.ALIGN_BEARING_OK)
            if aligned or self._align_left <= 0:
                self._align_done = True
                self._align_left = 0
            else:
                self._align_left -= 1
                if self._enable_terrain:
                    max_drive = float(self.ALIGN_MAX_DRIVE_TERRAIN)
                    min_side = float(self.ALIGN_MIN_SIDE_TERRAIN)
                else:
                    max_drive = float(self.MAX_DRIVE)
                    min_side = float(self.MIN_SIDE_DRIVE)
                if self._align_dir > 0:
                    drives = np.array([min_side, max_drive], dtype=float)
                else:
                    drives = np.array([max_drive, min_side], dtype=float)
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
        self._update_avoid_state(obs_size, obs_x)

        if self._avoid_left > 0:
            # Récupère uprightness pour adapter l'agressivité du dodge :
            # mouche penchée → dodge réduit pour permettre la récupération.
            try:
                _, _, _upright = self._get_orientation(sim)
            except Exception:
                _upright = 1.0
            bias, base_drive, sub_mode = self._avoid_command(target_bias, obs_x, obs_size, _upright)
        else:
            bias = float(target_bias)
            base_drive = float(self.BASE_DRIVE_FAST)
            sub_mode = "GO"
            # Sweep gauche-droite conditionnel : balaye le frontal QUAND la
            # mouche voit déjà qqch d'ambigu (obs_size dans [LOW, AVOID_SIZE_ON]).
            # Si la vision est totalement propre (obs_size ≈ 0), pas de sweep =
            # trajet direct préservé.
            if (
                self.GO_SWEEP_ENABLE
                and dist_to_banana is not None
                and dist_to_banana > float(self.GO_SWEEP_DISABLE_CLOSE_DIST)
                and obs_size > float(self.GO_SWEEP_OBS_SIZE_MIN)
                and obs_size < float(self.AVOID_SIZE_ON)
            ):
                period_dec = max(
                    1,
                    int(float(self.GO_SWEEP_PERIOD_S) / float(self.DECISION_INTERVAL_S)),
                )
                phase = 2.0 * np.pi * float(self._debug_decisions % period_dec) / float(period_dec)
                bias += float(self.GO_SWEEP_AMPLITUDE) * float(np.sin(phase))

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

        # ---- tanh + asymétrie roues ----
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

        # Slope-adaptive AVOID : sur pente raide, blend des drives pour éviter flip
        if self._avoid_left > 0 and self._enable_terrain and slope_mag > 0.20:
            _slope_blend = 0.50 * float(np.clip((slope_mag - 0.20) / 0.20, 0.0, 1.0))
            _mean_d = float(np.mean(drives))
            drives = drives * (1.0 - _slope_blend) + _mean_d * _slope_blend
            drives = np.clip(drives, 0.0, max_drive)

        if (
            self.DEBUG
            and self._debug_decisions <= self.DEBUG_MAX_DECISIONS
            and (self._debug_decisions % self.DEBUG_EVERY_DECISIONS == 0)
        ):
            dist_str = f"{dist_to_banana:.2f}" if dist_to_banana is not None else "?"
            print(
                f"[dbg d={self._debug_decisions:4d}] mode={'AVOID' if self._avoid_left>0 else 'GO':5s} "
                f"sub={sub_mode:5s} obs_x={obs_x:+.3f} obs_sz={obs_size:.4f} "
                f"latch={self._latched_obs_x:+.3f} dist={dist_str} bearing={bearing:+.3f} "
                f"target_bias={target_bias:+.3f} bias={bias:+.3f} "
                f"drives=({drives[0]:.3f},{drives[1]:.3f})",
                flush=True,
            )
        return drives

    # ------------------------------------------------------------------
    # SIDESTEP no-progress (anti-jam complémentaire à JAM_REFLEX)
    # ------------------------------------------------------------------
    def _sidestep_step(self, sim) -> "np.ndarray | None":
        """Backup + sidestep si la mouche ne progresse plus vers la banane.

        Détecte deux situations :
          - distance à banane stagne sur NOPROG_WINDOW décisions
          - roll dépasse ROLL_TRIGGER_THRESH pendant ROLL_TRIGGER_HOLD ticks
        Retourne drives ou None si inactif.
        """
        if not self.SIDESTEP_ENABLE:
            return None
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

        # Pas de banana → pas de notion de progression
        if self._last_dist_to_banana is None:
            return None

        # Désactivé proche de la banane (final sprint)
        if self._last_dist_to_banana < float(self.SIDESTEP_DISABLE_CLOSE_DIST):
            self._dist_history.clear()
            return None

        # Update dist history
        self._dist_history.append(float(self._last_dist_to_banana))
        if len(self._dist_history) > int(self.NOPROG_WINDOW):
            self._dist_history.pop(0)

        # Alarme roll (mouche penche fortement sans tomber)
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

        # Choisit la direction du sidestep selon le dernier diff de drives
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
        # Reset autres états pour éviter conflits
        self._avoid_left = 0
        self._avoid_clear = 0
        self._latched_obs_x = 0.0
        if self.DEBUG:
            print(
                f"[SIDESTEP d={self._debug_decisions}] dir={self._sidestep_dir:+d} "
                f"(roll={roll_pre:+.2f} prog={self._dist_history[0]-self._dist_history[-1]:.2f})",
                flush=True,
            )
        d = float(self.SIDESTEP_BACKUP_DRIVE)
        return np.array([d, d])

    # ------------------------------------------------------------------
    # AVOID FSM
    # ------------------------------------------------------------------
    def _schedule_realign_after_long_avoid(self) -> None:
        """Si l'épisode AVOID était assez long, relancer ALIGN vers la banane."""
        ticks = int(self._avoid_session_ticks)
        self._avoid_session_ticks = 0
        if not (self.ALIGN_INITIAL_ENABLE and self.ALIGN_AFTER_AVOID_ENABLE):
            return
        if ticks >= int(self.ALIGN_AFTER_AVOID_MIN_DECISIONS):
            self._align_done = False
            self._align_dir = 0.0
            self._align_left = 0

    def _update_avoid_state(self, obs_size: float, obs_x: float) -> None:
        """Trigger / exit AVOID, latch obs_x à l'entrée."""
        # Près de la banane : on ignore les obstacles
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

            # Refresh latch si nouveau signal franchement de l'autre côté
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
                f"[OBS+] dist={dist_str} bearing={self._last_target_bearing:+.3f} "
                f"obs_x={obs_x:+.3f} obs_sz={obs_size:.4f}",
                flush=True,
            )

    def _avoid_command(
        self,
        target_bias: float,
        live_obs_x: float,
        live_obs_size: float,
        uprightness: float = 1.0,
    ) -> tuple[float, float, str]:
        """Esquive ARC linéaire en centralité × taille × tilt-recovery.

        Magnitude = AVOID_TURN_MAX × centrality × size_factor × tilt_factor
          - centrality   : 1 - obs_x² → plus l'objet est central, plus on tourne
          - size_factor  : ∈ [0.8, 1.5] selon obs_size (gros = plus de dodge)
          - tilt_factor  : ∈ [0.3, 1.0] selon uprightness — si la mouche penche,
                           on réduit l'agressivité pour qu'elle se rééquilibre
                           (sinon flip post-AVOID).
        Direction = signe latché (stable). Si le live signal devient franc de
        l'autre côté (≥ AVOID_SIZE_ON et opposé au latch), on suit le live.
        """
        live_x = float(np.clip(live_obs_x, -1.0, 1.0))
        latched = float(np.clip(self._latched_obs_x, -1.0, 1.0))

        # Centralité (live) : adaptation arc à mesure que la mouche tourne
        centrality = 1.0 - live_x * live_x
        # Size factor : objet loin (petit) → dodge un peu réduit ; objet proche (gros) → dodge boosté.
        # 0.10 → 0.5 → clipé à 0.8 (= petit dodge léger mais pas trop)
        # 0.20 → 1.0 (medium)
        # 0.30+ → 1.5 (gros/proche)
        size_factor = float(np.clip(live_obs_size / float(self.AVOID_SIZE_MED), 0.8, 1.5))
        # Tilt factor : réduit dodge si mouche penchée pour permettre récupération.
        # up>0.95 → 1.0 (pas d'effet pour léger tilt) ; 0.5 → 0.53 ; 0.0 → 0.3
        tilt_factor = float(np.clip(uprightness / 0.95, 0.3, 1.0))
        dodge_mag = float(self.AVOID_TURN_MAX) * centrality * size_factor * tilt_factor

        # Direction stable via latch
        eps = float(self.AVOID_CENTER_EPS)
        if abs(latched) > eps:
            dodge_dir = -1.0 if latched > 0.0 else 1.0
        elif abs(live_x) > eps and live_obs_size >= float(self.AVOID_SIZE_ON):
            dodge_dir = -1.0 if live_x > 0.0 else 1.0
        else:
            # Obstacle dead-center : tiebreak côté banane
            if abs(target_bias) > 1e-9:
                dodge_dir = 1.0 if target_bias > 0.0 else -1.0
            else:
                dodge_dir = 1.0

        obstacle_bias = dodge_dir * dodge_mag
        bias = obstacle_bias + float(self.AVOID_BANANA_BLEND) * float(target_bias)
        bias = float(
            np.clip(bias, -float(self.VIS_TURN_MAX), float(self.VIS_TURN_MAX))
        )
        base = float(self.BASE_DRIVE_FAST) * float(self.AVOID_SPEED_FRAC)
        return bias, base, "ARC"

    # ------------------------------------------------------------------
    # Vision : panorama bi-oculaire frontal
    # ------------------------------------------------------------------
    def _compute_green_mask(self, roi_rgb: np.ndarray) -> np.ndarray:
        """Masque grass (chroma vert saturé)."""
        r = roi_rgb[..., 0]
        g = roi_rgb[..., 1]
        b = roi_rgb[..., 2]
        d = float(self.VIS_GRASS_GREEN_DELTA)
        gmin = float(self.VIS_GRASS_GREEN_MIN)
        return ((g - r) > d) & ((g - b) > d) & (g > gmin)

    def _compute_sky_mask(self, roi_rgb: np.ndarray) -> np.ndarray:
        """Masque unifié sky+cloud (= "tout l'arrière-plan non-herbe").

        Un seul masque pour ciel bleu, nuages blancs et zones de transition.
        Évite que des pixels nuage soient ratés et brisent la chaîne sky-grass-sky
        autour d'un obstacle (problème : on ne détectait que la pointe au-dessus
        du nuage, pas la partie traversant le nuage).

        Critère : pixel non-herbe ET pas trop sombre ET (bleu-froid OU peu saturé).
        - bleu-froid couvre le ciel bleu sombre (b ≥ r, b ≥ g).
        - peu saturé couvre les nuages neutres et transitions ciel/nuage.
        - exclut les couleurs chaudes saturées (brun = sol/troncs).
        """
        r = roi_rgb[..., 0]
        g = roi_rgb[..., 1]
        b = roi_rgb[..., 2]
        lum = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
        grass = self._compute_green_mask(roi_rgb)
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        rgb_spread = (mx - mn).astype(np.float32)
        bm = float(self.VIS_SKY_BLUE_MARGIN)
        blue_or_cool = (b + bm >= r) & (b + bm >= g)
        low_saturation = rgb_spread <= float(self.VIS_CLOUD_RGB_SPREAD_MAX)
        not_too_dark = lum >= float(self.VIS_CLOUD_LUM_MIN)
        not_too_bright = lum <= float(self.VIS_CLOUD_LUM_MAX)
        return (~grass) & not_too_dark & not_too_bright & (blue_or_cool | low_saturation)

    def _discard_small_blobs(self, mask: np.ndarray) -> np.ndarray:
        """Garde seulement les composantes connexes hautes+étroites (= vrai pic)."""
        if (
            _SCIPY_NDIMAGE is None
            or mask.size == 0
            or not np.asarray(mask, dtype=bool).any()
        ):
            return mask
        min_px = int(self.VIS_SPIKE_MIN_AREA_PX)
        min_h = int(self.VIS_SPIKE_MIN_HEIGHT_PX)
        min_aspect = float(self.VIS_SPIKE_MIN_ASPECT)
        if min_px <= 0 and min_h <= 0 and min_aspect <= 0.0:
            return mask
        m = np.asarray(mask, dtype=bool)
        lbl, n_labels = _SCIPY_NDIMAGE.label(m, structure=_SPIKE_CONN)
        if n_labels == 0:
            return mask
        counts = np.bincount(lbl.ravel())
        bad = np.zeros(counts.shape[0], dtype=bool)
        bad[0] = False
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

    def _eye_spike_mask(self, eye_img01: np.ndarray, eye: str) -> np.ndarray:
        """Détecte les pics (sky-blade-sky horizontal) dans la ROI d'un œil."""
        h_full, w_full = eye_img01.shape[:2]
        r0 = int(self.VIS_ROI_R0 * h_full)
        r1 = int(self.VIS_ROI_R1 * h_full)
        if eye == "left":
            c0 = int(self.VIS_ROI_C0_LEFT * w_full)
            c1 = int(self.VIS_ROI_C1_LEFT * w_full)
        else:
            c0 = int(self.VIS_ROI_C0_RIGHT * w_full)
            c1 = int(self.VIS_ROI_C1_RIGHT * w_full)
        roi = eye_img01[r0:r1, c0:c1, :]
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return np.zeros((max(1, h), max(1, w)), dtype=bool)

        green = self._compute_green_mask(roi)
        sky = self._compute_sky_mask(roi)
        if not green.any():
            return np.zeros((h, w), dtype=bool)

        # Pour chaque ligne, on cherche les segments grass entourés horizontalement
        # par sky/cloud → ce sont des pics qui dépassent du sol.
        spike = np.zeros((h, w), dtype=bool)
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
                    spike[r, s:e] = rg[s:e]

        spike = self._discard_small_blobs(spike)
        return spike

    def _vision_step(self, sim: MiniprojectSimulation) -> tuple[float, float]:
        """Calcule (obs_size, obs_x) à partir du panorama bi-oculaire frontal.

        Panorama = [œil_gauche_ROI | œil_droit_ROI] concaténés horizontalement.
        Le centre du panorama tombe sur le frontal binoculaire ("droit devant").
        Retour :
          obs_size ∈ [0, 1] : densité de pixels-pic dans le panorama (EMA).
          obs_x    ∈ [-1, +1] : position centroïde (EMA), -1=gauche, +1=droite.
        """
        try:
            frames = sim.get_raw_vision(sim.fly.name)
        except Exception:
            frames = None
        if frames is None or len(frames) == 0:
            return float(self._vis_obs_size), float(self._vis_obs_x)

        def _to_float01(img: np.ndarray) -> np.ndarray:
            a = np.asarray(img, dtype=np.float32)
            if a.ndim == 2:
                a = np.stack([a, a, a], axis=-1)
            if a.max() > 1.0:
                a = a / 255.0
            return np.clip(a, 0.0, 1.0)

        left_img = _to_float01(frames[0])
        right_img = _to_float01(frames[1] if len(frames) > 1 else frames[0])

        left_spike = self._eye_spike_mask(left_img, "left")
        right_spike = self._eye_spike_mask(right_img, "right")

        # Aligne les hauteurs si différentes (rare)
        if left_spike.shape[0] != right_spike.shape[0]:
            h = min(left_spike.shape[0], right_spike.shape[0])
            left_spike = left_spike[:h]
            right_spike = right_spike[:h]

        panorama = np.concatenate([left_spike, right_spike], axis=1)
        h_p, w_p = panorama.shape
        if w_p < 2:
            return float(self._vis_obs_size), float(self._vis_obs_x)

        # === Détection par composants connexes ===
        # Chaque obstacle = un blob. On calcule pour chacun :
        #   - pixel_count : taille = "à quel point l'objet est gros/proche"
        #     (un objet proche occupe plus de pixels)
        #   - centroid_col : position en colonnes du panorama
        # Score d'impact = pixel_count × (1 - position²) → on prend l'objet qui
        # influence le plus le chemin (gros + central = forte priorité).
        size_raw = 0.0
        x_raw = 0.0
        if _SCIPY_NDIMAGE is not None and panorama.any():
            labeled, n_comp = _SCIPY_NDIMAGE.label(panorama, structure=_SPIKE_CONN)
            if n_comp >= 1:
                # Pixel count par composant (label 0 = bg, exclus)
                pix_count = np.bincount(labeled.ravel())[1:].astype(np.float32)

                # Centroïde colonne par composant
                cols_grid = np.broadcast_to(
                    np.arange(w_p, dtype=np.float32), panorama.shape
                ).ravel()
                lbl_flat = labeled.ravel()
                col_sum = np.bincount(
                    lbl_flat, weights=cols_grid, minlength=n_comp + 1
                )[1:]
                centroid_col = col_sum / np.maximum(pix_count, 1.0)

                # Position normalisée ∈ [-1, +1] par composant
                pos = (centroid_col / float(max(1, w_p - 1))) * 2.0 - 1.0
                pos = np.clip(pos, -1.0, 1.0)

                # Score d'impact : pixel_count × centralité (objet gros+central = priorité)
                centrality = 1.0 - pos * pos
                impact = pix_count * centrality

                # Composant le plus impactant pour le chemin
                top = int(np.argmax(impact))
                x_raw = float(pos[top])
                # obs_size : pixel_count normalisé
                # Calibration : un brin frontal typique ~250 pixels dans h_p×~30 cols
                # → obs_size ≈ 0.25 (h_p=100, divisé par 1000 pour échelle [0,1])
                size_raw = float(pix_count[top]) / float(h_p * 10)
                size_raw = min(size_raw, 1.0)

        # EMA
        ema = float(self.VIS_EMA)
        self._vis_obs_size = ema * self._vis_obs_size + (1.0 - ema) * size_raw
        self._vis_obs_x = ema * self._vis_obs_x + (1.0 - ema) * x_raw

        return float(self._vis_obs_size), float(self._vis_obs_x)

    # ------------------------------------------------------------------
    def compute_vision_debug_overlay(
        self, sim: MiniprojectSimulation
    ) -> "np.ndarray | None":
        """Debug RGB overlay : ROI + sky tint + spikes (vert)."""
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

        out_eyes = []
        for ei, raw in enumerate(eye_imgs):
            eye = "left" if ei == 0 else "right"
            h, w = raw.shape[:2]
            r0, r1 = int(h * self.VIS_ROI_R0), int(h * self.VIS_ROI_R1)
            if eye == "left":
                c0 = int(w * self.VIS_ROI_C0_LEFT)
                c1 = int(w * self.VIS_ROI_C1_LEFT)
            else:
                c0 = int(w * self.VIS_ROI_C0_RIGHT)
                c1 = int(w * self.VIS_ROI_C1_RIGHT)

            img01 = raw.astype(np.float32) / 255.0
            spike_roi = self._eye_spike_mask(img01, eye)

            base = (raw.astype(np.float32) * 0.45).astype(np.uint8)
            overlay = base.copy()
            roi_view = overlay[r0:r1, c0:c1, :]

            sky_roi = self._compute_sky_mask(img01[r0:r1, c0:c1])
            tint = np.array([45.0, 88.0, 188.0], dtype=np.float32)
            if sky_roi.shape == roi_view.shape[:2]:
                roi_view[sky_roi] = (
                    roi_view[sky_roi].astype(np.float32) * 0.52 + tint * 0.48
                ).astype(np.uint8)

            if spike_roi.shape == roi_view.shape[:2]:
                roi_view[spike_roi] = np.array([20, 255, 40], dtype=np.uint8)

            cyan = np.array([0, 220, 220], dtype=np.uint8)
            overlay[r0 : r0 + 1, c0:c1, :] = cyan
            overlay[r1 - 1 : r1, c0:c1, :] = cyan
            overlay[r0:r1, c0 : c0 + 1, :] = cyan
            overlay[r0:r1, c1 - 1 : c1, :] = cyan

            out_eyes.append(overlay)

        result = np.concatenate(out_eyes, axis=1).astype(np.uint8)
        self._vis_debug_overlay = result
        return result
