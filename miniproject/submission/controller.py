import sys

import numpy as np

from miniproject.simulation import MiniprojectSimulation

# Robustesse éval : les prints de debug contiennent des caractères Unicode
# (→, é, …). Sur une console Windows cp1252 un print non encodable lève
# UnicodeEncodeError et tue le contrôleur en pleine simulation. On reconfigure
# stdout/stderr en UTF-8 avec fallback 'replace' pour qu'un print ne puisse
# JAMAIS crasher le contrôleur.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from scipy import ndimage as _SCIPY_NDIMAGE
except ImportError:
    _SCIPY_NDIMAGE = None


class Controller:
    """Vision edge-based + repulsion field steering pour Level 2.

    Pipeline :
      1. Stop sur banane (distance ou odor)
      2. Terrain stuck/escape (anti-trap mesh)
      3. ALIGN initial (pivot sur place vers banane)
      4. Vision : détection arêtes verticales + colonnes solid-green sur
         panorama bi-oculaire → liste de spikes (pos, strength)
      5. Steering : champ de répulsion sur tous les spikes + cap banane
      6. Terrain : slope braking, slope-bias steering, tilt-adaptive grip
    """

    # --- scheduling ---
    DECISION_INTERVAL_S = 0.025  # 25 ms = 40 décisions/s

    # --- olfaction (stop only) ---
    PALP_WEIGHT = 9
    ANTENNA_WEIGHT = 1
    EPS_ODOR = 1e-12
    STOP_ODOR_THRESHOLD = 5e-3      # 10x plus permissif : laisse la mouche
                                     # s'approcher jusqu'à STOP_DIST physique
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

    # --- grip boost (merge 19/05 : grip doux kreslo + startup REPULSION) ---
    # kreslo : grip FAIBLE appliqué uniquement aux pattes en appui — ne fige
    # jamais la démarche (le grip fort toutes-pattes de REPULSION bloquait la
    # translation). On garde néanmoins deux exceptions à grip FORT LOCAL :
    #   - startup (stabilisation au spawn sur le terrain),
    #   - phase de recul head-collision (besoin de coller au sol pour reculer).
    TERRAIN_GRIP_FORCE = 1.45   # kreslo : grip doux, pattes en appui
    WIND_GRIP_FORCE = 1.80     # kreslo : vent, pattes en appui (ne fige pas la marche)
    COLLISION_BACKUP_GRIP_FORCE = 6.0  # grip fort LOCAL pendant le recul head-collision
    STARTUP_GRIP_FORCE = 6.0    # grip fort LOCAL pour stabiliser au spawn
    # Grip fort sur toutes les pattes pendant les N premiers SIM STEPS (pas
    # décisions), pour stabiliser la mouche dès le spawn sur le terrain.
    STARTUP_MAX_GRIP_STEPS = 3000   # ~0.15 s @ timestep 1e-4

    # --- orientation safety ---
    TERRAIN_UPRIGHT_TILT_WARN = 0.35    # plus permissif (~ 70° au lieu de ~60°)
    TERRAIN_TILT_RESET_HOLD = 80        # ~2s : laisse + de temps pour récup
    TERRAIN_FLIP_WEAK_UPRIGHT = 0.12
    TERRAIN_FLIP_RESET_HOLD = 60        # ~1.5s pour flip total

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
    VIS_ROI_C0_LEFT = 0.55
    VIS_ROI_C1_LEFT = 0.98
    VIS_ROI_C0_RIGHT = 0.02
    VIS_ROI_C1_RIGHT = 0.45

    # EMA pour stabilité du signal (bas = lissage plus fort, anti-wind sway)
    VIS_EMA = 0.30

    # --- Edge-based detection (vertical edge pairing) ---
    # Un pique = 2 arêtes verticales parallèles séparées de quelques pixels.
    # Robuste : pas de seuil couleur, juste géométrie. Les collines ont des
    # arêtes horizontales → exclues. Les pics ont des arêtes verticales.
    VIS_EDGE_MAGNITUDE_THRESHOLD = 0.08    # seuil de force d'arête
    VIS_EDGE_VERTICALITY_THRESHOLD = 0.20  # |gx|/(|gx|+|gy|) > seuil = vertical (tolère pics penchés par vent)
    VIS_EDGE_MIN_DENSITY = 0.8            # densité par colonne (haut = filtre les piques lointains)
    VIS_EDGE_MIN_SPIKE_WIDTH = 0           # largeur min entre 2 arêtes paires
    VIS_EDGE_MAX_SPIKE_WIDTH = 30          # largeur max entre 2 arêtes paires

    # --- Solid-green-column detector (pique proche/large) ---
    # Complète le pairing d'arêtes : un pique très proche ou qui sort du cadre
    # n'a pas de paire d'arêtes (centre uniforme vert OU 1 seul bord visible).
    # Une colonne saturée de vert = pique massif/proche.
    VIS_GREEN_SOLID_THRESHOLD = 0.60       # fraction de colonne en vert (compense oscillations vent)
    VIS_GREEN_DELTA = 0.025                # g > r+delta ET g > b+delta = vert (tolère ombrage vent)

    # --- Steering : seuil de proximité banane (sprint final) ---
    AVOID_DISABLE_CLOSE_DIST = 8.0  # < 8 m de la banane : on fonce sans répulsion

    # --- Repulsion field steering ---
    # Mode alternatif au FSM AVOID : chaque spike détecté pousse la mouche dans
    # la direction opposée. La somme des répulsions donne le bias de steering.
    # Avantage : la mouche passe naturellement DANS les gaps entre piques au
    # lieu de devoir choisir un côté global.
    REPULSION_FIELD_ENABLE = True
    REPULSION_GAIN = 12.0                 # MAX agressif sur la répulsion
    REPULSION_FALLOFF_ALPHA = 1.2         # exp(-alpha × p²) : étendue plus large
    REPULSION_MIN_SIZE = 0.04             # très réactif (déclenche sur petits piques)
    REPULSION_CENTRAL_EPS = 0.12          # zone "central" un peu plus large
    REPULSION_CENTRAL_BOOST = 200         # gros boost central
    REPULSION_BANANA_BLEND = 0.15         # banane quasi ignorée pendant l'esquive
    # PIVOT en mode REPUL : si le plus gros pic dépasse ce seuil, on pivote
    # comme un fou (drives asymétriques max), peu importe le bias calculé.
    REPULSION_PIVOT_SIZE = 0.12           # strength min pour engager le pivot REPUL

    # Saturation universelle bias (anti-violent turn)
    VIS_TURN_MAX = 10.0

    # --- Initial alignment ---
    ALIGN_INITIAL_ENABLE = True
    ALIGN_BEARING_OK = 0.20
    ALIGN_MAX_DECISIONS = 60
    ALIGN_MAX_DRIVE_TERRAIN = 2.00
    ALIGN_MIN_SIDE_TERRAIN = 0.25

    # --- Head collision recovery (séquence fixe) ---
    # Quand la mouche se cogne la tête (force externe > seuil), elle exécute
    # une séquence en 3 phases : BACKUP → PIVOT vers banane → ARC autour.
    HEAD_COLLISION_ENABLE = True
    HEAD_COLLISION_FORCE_THRESH = 3.0           # N : très bas pour détecter même les frôlements
    HEAD_COLLISION_COOLDOWN_DECISIONS = 0       # période sourde après une séquence
    HEAD_COLLISION_BACKUP_DECISIONS = 45         # ~1 s en arrière
    HEAD_COLLISION_PIVOT_DECISIONS = 3          # ~0.5 s de pivot vers banane
    HEAD_COLLISION_ARC_DECISIONS = 0            # ~0.75 s d'arc autour de l'obstacle
    HEAD_COLLISION_BACKUP_DRIVE = -0.6           # marche arrière (drives identiques)
    HEAD_COLLISION_PIVOT_MAX = 2.0
    HEAD_COLLISION_PIVOT_MIN = 0.25
    HEAD_COLLISION_ARC_OUTER = 2.0               # roue extérieure de l'arc
    HEAD_COLLISION_ARC_INNER = 0.6               # roue intérieure de l'arc
    # Pose naturelle préservée : on n'écrase PAS toutes les articulations.
    # On applique uniquement un OFFSET (additif au CPG) sur le coxa-pitch
    # des pattes avant pour les lever juste assez à décrocher du pic.
    # Femur/tibia restent contrôlés par le CPG → allure naturelle.
    HEAD_COLLISION_FRONT_TUCK_DECISIONS = 15      # durée du lift (~250 ms)
    HEAD_COLLISION_FRONT_LIFT_COXA_OFFSET = -0.5   # offset coxa-pitch (+1) : lève la patte

    # --- GO mode pivot (réalignement continu) ---
    # En GO (pas d'obstacle), si la mouche est mal alignée avec la banane,
    # elle pivote FORT (drives asymétriques max) jusqu'à être presque dans l'axe.
    GO_PIVOT_ENABLE = True
    GO_PIVOT_BEARING_ON = 0.30      # |bearing| > seuil = pivote fort
    GO_PIVOT_BEARING_OFF = 0.12     # |bearing| < seuil = arrête de pivoter (hystérésis)

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
        try:
            head_idx = next(i for i, s in enumerate(fly_segs) if s.name == "c_head")
            self._head_body_id = body_ids[head_idx]
        except StopIteration:
            self._head_body_id = self._thorax_body_id

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
        self._vis_obs_size = 0.0       # taille pic prioritaire (pour overlay debug)
        self._vis_obs_x = 0.0          # position pic prioritaire (pour overlay debug)
        self._vis_debug_overlay = None
        # Liste de tous les spikes détectés au dernier frame (pour repulsion field)
        # Chaque entrée : (pos_normalized ∈ [-1,+1], strength ∈ [0,1])
        self._vis_all_spikes: list[tuple[float, float]] = []

        # Initial alignment state
        self._align_done = False
        self._align_dir = 0.0
        self._align_left = 0
        # GO-mode pivot state (hystérésis)
        self._go_pivot_active = False
        # Head collision recovery state (séquence en phases)
        # 0 = idle, 1 = backup, 2 = pivot toward banana, 3 = arc around
        self._collision_phase = 0
        self._collision_left = 0
        self._collision_arc_dir = 1.0
        self._collision_cooldown = 0
        # Peak head force tracking entre 2 décisions (échantillonnage @ sim step)
        self._head_force_peak = 0.0

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
                    sim.mj_data.qpos[_addr + 2] += 0.4
                    _mj.mj_forward(sim.mj_model, sim.mj_data)
                    break
        except Exception:
            pass


    # ------------------------------------------------------------------
    def step(self, sim: MiniprojectSimulation):
        # Track peak head force chaque sim step (catch impacts brefs entre décisions)
        if self.HEAD_COLLISION_ENABLE:
            try:
                hf = float(np.linalg.norm(
                    sim.mj_data.cfrc_ext[self._head_body_id, 3:]
                ))
                if hf > self._head_force_peak:
                    self._head_force_peak = hf
            except Exception:
                pass

        is_decision_step = self._step_count % self._decision_every == 0
        if is_decision_step:
            self._drives = self._compute_drives(sim)
            self._debug_decisions = self._step_count // self._decision_every
        self._step_count += 1

        joint_angles, adhesion = self.turning_controller.step(self._drives)

        # --- Front legs lift au tout début du BACKUP head-collision ---
        # Offset additif sur la coxa-pitch des pattes avant pour les lever
        # juste assez à décrocher du pic. Femur/tibia restent CPG → allure
        # naturelle. Actif seulement les N premières décisions du BACKUP.
        tuck_threshold = int(self.HEAD_COLLISION_BACKUP_DECISIONS) - int(
            self.HEAD_COLLISION_FRONT_TUCK_DECISIONS
        )
        if self._collision_phase == 1 and self._collision_left > tuck_threshold:
            for li in (0, 3):
                base = li * 7
                joint_angles[base + 1] += float(self.HEAD_COLLISION_FRONT_LIFT_COXA_OFFSET)

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

            # Log uprightness toutes les 5 décisions pour diagnostic
            if self._debug_decisions % 5 == 0:
                print(f"[UP d={self._debug_decisions}] upright={uprightness:.2f}", flush=True)

            if uprightness < float(self.TERRAIN_UPRIGHT_TILT_WARN):
                self._tilt_decisions += 1
            else:
                self._tilt_decisions = 0

            # JAMAIS de stop permanent sur tilt non plus. La mouche peut
            # rester tiltée longtemps (sur pente) sans être vraiment KO.
            if self._tilt_decisions >= int(self.TERRAIN_TILT_RESET_HOLD):
                self._tilt_decisions = 0  # juste reset le compteur

        # --- Grip control (terrain) ---
        if self._enable_terrain and is_decision_step:
            if self._escape_decisions_left > 0:
                adhesion = np.zeros_like(adhesion)
                return joint_angles, adhesion

            # JAMAIS de stop permanent sur flip. Si renversée, on freeze
            # les drives jusqu'à récupération naturelle (slide sur pente).
            # La mouche peut souvent ressortir d'un flip après quelques secondes.
            if uprightness < 0.0:
                return joint_angles, np.zeros_like(adhesion)

            # Merge 19/05 : grip doux kreslo (pattes en appui seulement, ne
            # fige pas la marche) avec deux exceptions à grip FORT LOCAL.
            #   - startup : grip fort TOUTES pattes pour stabiliser au spawn,
            #   - collision backup : grip fort sur pattes en appui (recul collé
            #     au sol), pattes AVANT libérées pour ne pas bloquer la marche
            #     arrière.
            in_collision_backup = self._collision_phase == 1
            in_startup_grip = self._step_count < int(self.STARTUP_MAX_GRIP_STEPS)
            if in_startup_grip:
                grip_val = float(self.STARTUP_GRIP_FORCE)
            elif in_collision_backup:
                grip_val = float(self.COLLISION_BACKUP_GRIP_FORCE)
            elif self._enable_wind:
                grip_val = float(self.WIND_GRIP_FORCE)
            else:
                grip_val = float(self.TERRAIN_GRIP_FORCE)
            try:
                contact_forces = sim.mj_data.cfrc_ext[self._contact_body_ids, 3:]
                contact_mag = np.linalg.norm(contact_forces, axis=1)
                stance = contact_mag > self.CONTACT_THRESHOLD
                adhesion = np.zeros_like(adhesion)
                n = min(len(adhesion), len(stance))
                if in_startup_grip:
                    # Spawn : grip fort sur TOUTES les pattes (REPULSION).
                    adhesion[:n] = grip_val
                else:
                    # kreslo : grip uniquement sur pattes en appui — ne fige
                    # jamais la démarche (vrai pour vent, collision et normal).
                    adhesion[:n] = stance[:n].astype(float) * grip_val
                # En BACKUP : on désactive les pattes AVANT (index 0=lf, 3=rf)
                # ordre = [lf, lm, lh, rf, rm, rh]
                if in_collision_backup:
                    if n > 0:
                        adhesion[0] = 0.0  # left front
                    if n > 3:
                        adhesion[3] = 0.0  # right front
            except Exception:
                adhesion = (
                    np.full_like(adhesion, grip_val)
                    if in_startup_grip
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

        # ---- Head collision recovery (séquence fixe en 3 phases) ----
        if self.HEAD_COLLISION_ENABLE:
            if self._collision_cooldown > 0:
                self._collision_cooldown -= 1
            # Force = peak depuis la dernière décision (catch impacts brefs)
            head_force = float(self._head_force_peak)
            self._head_force_peak = 0.0
            # Déclenchement : pas en séquence + pas en cooldown + force seuil
            if (
                self._collision_phase == 0
                and self._collision_cooldown == 0
                and head_force > float(self.HEAD_COLLISION_FORCE_THRESH)
            ):
                # Calcule la direction banane (gauche/droite) pour l'arc
                _tb, _br = self._compute_target_bias(sim, thorax_xy)
                self._collision_arc_dir = 1.0 if _br < 0.0 else -1.0
                # Phase 1 : BACKUP
                self._collision_phase = 1
                self._collision_left = int(self.HEAD_COLLISION_BACKUP_DECISIONS)
                print(
                    f"[HEAD-COLL d={self._debug_decisions}] head_force={head_force:.1f}N "
                    f"banana_side={'L' if self._collision_arc_dir > 0 else 'R'} → BACKUP",
                    flush=True,
                )

            if self._collision_phase > 0:
                self._collision_left -= 1
                phase = self._collision_phase
                if phase == 1:
                    # BACKUP : reculer en ligne droite
                    d = float(self.HEAD_COLLISION_BACKUP_DRIVE)
                    drives = np.array([d, d])
                elif phase == 2:
                    # PIVOT vers banane : pivote fort côté banana
                    mx = float(self.HEAD_COLLISION_PIVOT_MAX)
                    mn = float(self.HEAD_COLLISION_PIVOT_MIN)
                    if self._collision_arc_dir > 0:
                        # banane à gauche → tourner à gauche : drive_right > drive_left
                        drives = np.array([mn, mx])
                    else:
                        drives = np.array([mx, mn])
                else:  # phase == 3
                    # ARC autour de l'obstacle : courbe vers la banane
                    out = float(self.HEAD_COLLISION_ARC_OUTER)
                    inn = float(self.HEAD_COLLISION_ARC_INNER)
                    if self._collision_arc_dir > 0:
                        # tourne à gauche → roue gauche = inner (lente)
                        drives = np.array([inn, out])
                    else:
                        drives = np.array([out, inn])

                # Transition de phase
                if self._collision_left <= 0:
                    if phase == 1:
                        self._collision_phase = 2
                        self._collision_left = int(self.HEAD_COLLISION_PIVOT_DECISIONS)
                        print(f"[HEAD-COLL d={self._debug_decisions}] → PIVOT", flush=True)
                    elif phase == 2:
                        self._collision_phase = 3
                        self._collision_left = int(self.HEAD_COLLISION_ARC_DECISIONS)
                        print(f"[HEAD-COLL d={self._debug_decisions}] → ARC", flush=True)
                    else:
                        self._collision_phase = 0
                        self._collision_cooldown = int(self.HEAD_COLLISION_COOLDOWN_DECISIONS)
                        print(f"[HEAD-COLL d={self._debug_decisions}] sequence done", flush=True)

                if (
                    self.DEBUG
                    and self._debug_decisions <= self.DEBUG_MAX_DECISIONS
                    and (self._debug_decisions % self.DEBUG_EVERY_DECISIONS == 0)
                ):
                    phase_name = {1: "BACKUP", 2: "PIVOT", 3: "ARC"}.get(phase, "?")
                    print(
                        f"[dbg d={self._debug_decisions:4d}] mode=COLL-{phase_name} "
                        f"left={self._collision_left} "
                        f"drives=({drives[0]:.3f},{drives[1]:.3f})",
                        flush=True,
                    )
                return drives

        # ---- Stop sur banane (olfaction) ----
        odor_lin = sim.get_olfaction(sim.fly.name)
        lp, rp, la, ra = odor_lin[:, 0]
        odor_l = self.PALP_WEIGHT * float(lp) + self.ANTENNA_WEIGHT * float(la)
        odor_r = self.PALP_WEIGHT * float(rp) + self.ANTENNA_WEIGHT * float(ra)
        mean_odor = 0.5 * (odor_l + odor_r)
        if mean_odor > self.STOP_ODOR_THRESHOLD:
            print(f"[STOP REASON] ODOR d={self._debug_decisions} mean_odor={mean_odor:.3e} dist={dist_to_banana:.2f}", flush=True)
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

        # ---- REPULSION FIELD / GO ----
        # Tous les piques détectés exercent une force de répulsion. Pas de FSM
        # AVOID. Si pas de piques (ou trop proche banane), GO direct vers banane.
        close_to_banana = (
            dist_to_banana is not None
            and dist_to_banana < float(self.AVOID_DISABLE_CLOSE_DIST)
        )

        if close_to_banana:
            bias = float(target_bias)
            base_drive = float(self.BASE_DRIVE_FAST)
            sub_mode = "GO"
        else:
            rep_bias, rep_base, rep_mode, _rep_active = self._compute_repulsion_bias(
                target_bias, list(self._vis_all_spikes)
            )
            bias = rep_bias
            base_drive = rep_base
            sub_mode = rep_mode

        # ---- REPUL-PIVOT : pivot hyper-aggressif sur gros pic détecté ----
        # Si on est en REPUL et que le pic max dépasse le seuil, on pivote à fond.
        if sub_mode == "REPUL" and self._vis_all_spikes:
            p_max, s_max = max(self._vis_all_spikes, key=lambda ps: ps[1])
            if s_max >= float(self.REPULSION_PIVOT_SIZE):
                # Direction = signe(p_max) inversé (spike à droite → pivote gauche)
                if abs(p_max) < float(self.REPULSION_CENTRAL_EPS):
                    # Pile en face : utilise côté banane (sinon droite par défaut)
                    if abs(target_bias) > 1e-9:
                        dir_pivot = 1.0 if target_bias > 0.0 else -1.0
                    else:
                        dir_pivot = 1.0
                else:
                    dir_pivot = -1.0 if p_max > 0.0 else 1.0
                if self._enable_terrain:
                    pivot_max = float(self.ALIGN_MAX_DRIVE_TERRAIN)
                    pivot_min = float(self.ALIGN_MIN_SIDE_TERRAIN)
                else:
                    pivot_max = float(self.MAX_DRIVE)
                    pivot_min = float(self.MIN_SIDE_DRIVE)
                if dir_pivot > 0:
                    pivot_drives = np.array([pivot_max, pivot_min], dtype=float)
                else:
                    pivot_drives = np.array([pivot_min, pivot_max], dtype=float)
                if (
                    self.DEBUG
                    and self._debug_decisions <= self.DEBUG_MAX_DECISIONS
                    and (self._debug_decisions % self.DEBUG_EVERY_DECISIONS == 0)
                ):
                    dist_str = (
                        f"{dist_to_banana:.2f}" if dist_to_banana is not None else "?"
                    )
                    print(
                        f"[dbg d={self._debug_decisions:4d}] mode=REPUL-PIV "
                        f"p_max={p_max:+.3f} s_max={s_max:.3f} dir={int(dir_pivot):+d} "
                        f"dist={dist_str} drives=({pivot_drives[0]:.3f},{pivot_drives[1]:.3f})",
                        flush=True,
                    )
                return pivot_drives

        # ---- GO-mode pivot (réalignement continu vers banane) ----
        # Si en GO et bearing > seuil ON → pivote fort. Hystérésis : continue
        # tant que bearing > seuil OFF, puis désengage.
        if self.GO_PIVOT_ENABLE and sub_mode == "GO":
            ab = abs(float(bearing))
            if self._go_pivot_active:
                if ab < float(self.GO_PIVOT_BEARING_OFF):
                    self._go_pivot_active = False
            else:
                if ab > float(self.GO_PIVOT_BEARING_ON):
                    self._go_pivot_active = True

            if self._go_pivot_active:
                if self._enable_terrain:
                    pivot_max = float(self.ALIGN_MAX_DRIVE_TERRAIN)
                    pivot_min = float(self.ALIGN_MIN_SIDE_TERRAIN)
                else:
                    pivot_max = float(self.MAX_DRIVE)
                    pivot_min = float(self.MIN_SIDE_DRIVE)
                if bearing > 0:
                    pivot_drives = np.array([pivot_min, pivot_max], dtype=float)
                else:
                    pivot_drives = np.array([pivot_max, pivot_min], dtype=float)
                if (
                    self.DEBUG
                    and self._debug_decisions <= self.DEBUG_MAX_DECISIONS
                    and (self._debug_decisions % self.DEBUG_EVERY_DECISIONS == 0)
                ):
                    dist_str = (
                        f"{dist_to_banana:.2f}" if dist_to_banana is not None else "?"
                    )
                    print(
                        f"[dbg d={self._debug_decisions:4d}] mode=GO-PIV "
                        f"bearing={bearing:+.3f} dist={dist_str} "
                        f"drives=({pivot_drives[0]:.3f},{pivot_drives[1]:.3f})",
                        flush=True,
                    )
                return pivot_drives
        else:
            self._go_pivot_active = False

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
            # Atténuation forte du slope_bias quand on évite des piques
            # (sinon la pente annule la commande de répulsion).
            if sub_mode == "REPUL":
                slope_bias *= 0.15
            bias += float(np.clip(slope_bias, -self.SLOPE_STEER_MAX, self.SLOPE_STEER_MAX))
            turn_mod = turn_mod / (1.0 + self.TURN_STEEP_GAIN * max(0.0, slope_mag))

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

        # TILT-adaptive : si la mouche tilte (uprightness < 0.7), blend
        # vers drives symétriques pour éviter le flip.
        try:
            xmat_t = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
            tilt_up = float(xmat_t[2, 2])
        except Exception:
            tilt_up = 1.0
        if tilt_up < 0.70:
            tilt_blend = min(0.90, (0.70 - tilt_up) / 0.50)
            _mean_d = float(np.mean(drives))
            drives = drives * (1.0 - tilt_blend) + _mean_d * tilt_blend
            drives = np.clip(drives, 0.0, max_drive)

        if (
            self.DEBUG
            and self._debug_decisions <= self.DEBUG_MAX_DECISIONS
            and (self._debug_decisions % self.DEBUG_EVERY_DECISIONS == 0)
        ):
            dist_str = f"{dist_to_banana:.2f}" if dist_to_banana is not None else "?"
            print(
                f"[dbg d={self._debug_decisions:4d}] mode={sub_mode:5s} "
                f"obs_x={obs_x:+.3f} obs_sz={obs_size:.4f} "
                f"n_spikes={len(self._vis_all_spikes)} dist={dist_str} "
                f"bearing={bearing:+.3f} target_bias={target_bias:+.3f} "
                f"bias={bias:+.3f} drives=({drives[0]:.3f},{drives[1]:.3f})",
                flush=True,
            )
        return drives

    # ------------------------------------------------------------------
    # Repulsion field steering
    # ------------------------------------------------------------------
    def _compute_repulsion_bias(
        self, target_bias: float, spikes: "list[tuple[float, float]]"
    ) -> "tuple[float, float, str, bool]":
        """Répulsion calculée UNIQUEMENT sur le plus gros pic à l'écran.

        On sélectionne le spike avec la plus grande strength s ∈ [0, 1]
        (= taille × intensité, donc le pic le plus dangereux/proche).
        Tous les autres sont ignorés. Évite que des piques mineurs noient le
        signal du vrai obstacle imminent.

        kernel(p) = -sign(p) × s × factor
          - p > 0 (à droite)  → bias < 0 → tourne à GAUCHE
          - p < 0 (à gauche)  → bias > 0 → tourne à DROITE
          - |p| < eps (central) → boost de répulsion, sign = côté banane

        Returns:
          bias, base_drive, mode_str, active
        """
        if not spikes:
            return float(target_bias), float(self.BASE_DRIVE_FAST), "GO", False

        # Pique le plus fort
        p_max, s_max = max(spikes, key=lambda ps: ps[1])
        if s_max < float(self.REPULSION_MIN_SIZE):
            # Pic trop faible → ignoré
            return float(target_bias), float(self.BASE_DRIVE_FAST), "GO", False

        alpha = float(self.REPULSION_FALLOFF_ALPHA)
        central_eps = float(self.REPULSION_CENTRAL_EPS)
        central_boost = float(self.REPULSION_CENTRAL_BOOST)

        if abs(p_max) < central_eps:
            # Pile en face : direction = signe du target_bias (côté banane),
            # sinon par défaut droite.
            if abs(target_bias) > 1e-9:
                sgn = 1.0 if target_bias > 0.0 else -1.0
            else:
                sgn = 1.0
            factor = 1.0 + central_boost
        else:
            sgn = 1.0 if p_max > 0.0 else -1.0
            factor = float(np.exp(-alpha * p_max * p_max))

        bias = -sgn * float(s_max) * factor * float(self.REPULSION_GAIN)
        # Mélange avec cap banane (attraction)
        bias += float(self.REPULSION_BANANA_BLEND) * float(target_bias)
        bias = float(
            np.clip(bias, -float(self.VIS_TURN_MAX), float(self.VIS_TURN_MAX))
        )
        base = float(self.BASE_DRIVE_FAST)
        return bias, base, "REPUL", True

    # ------------------------------------------------------------------
    # Vision : edge-based vertical spike detection
    # ------------------------------------------------------------------
    def _detect_vertical_edges(
        self, roi_rgb: np.ndarray
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Masque (h, w) des pixels arête verticale (= bord de pic).

        Une arête verticale a un gradient horizontal fort (|gx| grand) et un
        gradient vertical faible (|gy| petit). Collines = arêtes horizontales,
        donc exclues. Pics = arêtes verticales, donc gardées.
        """
        gray = (
            0.299 * roi_rgb[..., 0]
            + 0.587 * roi_rgb[..., 1]
            + 0.114 * roi_rgb[..., 2]
        ).astype(np.float32)

        # Sobel-like gradients via convolution discrète (sans dépendance scipy)
        # gx : différence horizontale → détecte arêtes verticales
        # gy : différence verticale → détecte arêtes horizontales
        if _SCIPY_NDIMAGE is not None:
            from scipy.ndimage import sobel as _sobel

            gx = _sobel(gray, axis=1)
            gy = _sobel(gray, axis=0)
        else:
            gx = np.zeros_like(gray)
            gy = np.zeros_like(gray)
            gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
            gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

        gx_abs = np.abs(gx)
        gy_abs = np.abs(gy)
        magnitude = np.sqrt(gx_abs * gx_abs + gy_abs * gy_abs)
        verticality = gx_abs / (gx_abs + gy_abs + 1e-6)

        edge_mask = (
            magnitude > float(self.VIS_EDGE_MAGNITUDE_THRESHOLD)
        ) & (verticality > float(self.VIS_EDGE_VERTICALITY_THRESHOLD))
        return edge_mask, magnitude, verticality

    def _vertical_edge_density(self, edge_mask: np.ndarray) -> np.ndarray:
        """Densité par colonne d'arête verticale ∈ [0, 1]."""
        h, _ = edge_mask.shape
        if h == 0:
            return np.zeros(edge_mask.shape[1], dtype=np.float32)
        return edge_mask.sum(axis=0).astype(np.float32) / float(h)

    def _find_spike_columns(self, density: np.ndarray) -> list:
        """Trouve les pics par pairing d'arêtes verticales adjacentes.

        Returns: list de (center_col, width, intensity).
        """
        w = int(len(density))
        if w < 2:
            return []
        edge_cols = np.where(density > float(self.VIS_EDGE_MIN_DENSITY))[0]
        if len(edge_cols) < 2:
            return []

        # Groupe les colonnes adjacentes (arête épaisse = 2-3 colonnes)
        groups: list[list[int]] = []
        current: list[int] = [int(edge_cols[0])]
        for c in edge_cols[1:]:
            c = int(c)
            if c - current[-1] <= 2:
                current.append(c)
            else:
                groups.append(current)
                current = [c]
        groups.append(current)

        centers = [int(np.mean(g)) for g in groups]
        strengths = [float(density[g].max()) for g in groups]

        min_w = int(self.VIS_EDGE_MIN_SPIKE_WIDTH)
        max_w = int(self.VIS_EDGE_MAX_SPIKE_WIDTH)
        spikes: list[tuple[int, int, float]] = []
        for i in range(len(centers) - 1):
            left = centers[i]
            right = centers[i + 1]
            width = right - left
            if min_w <= width <= max_w:
                center = (left + right) // 2
                intensity = (strengths[i] + strengths[i + 1]) * 0.5
                spikes.append((center, width, intensity))
        return spikes

    def _find_solid_green_spikes(
        self, panorama_rgb: np.ndarray
    ) -> "tuple[list, np.ndarray]":
        """Détecte les piques massifs/proches via colonnes saturées de vert.

        Complète le pairing d'arêtes : un pique trop large pour le pairing,
        ou qui sort du cadre (1 seul bord visible), ne déclenche pas le
        pairing mais sature plusieurs colonnes en vert. On les regroupe.

        Returns:
          spikes : list de (center, width, intensity)
          green_col_mask : bool array (w,) marquant les colonnes solid-green
        """
        h, w = panorama_rgb.shape[:2]
        if h < 2 or w < 2:
            return [], np.zeros(w, dtype=bool)

        r = panorama_rgb[..., 0]
        g = panorama_rgb[..., 1]
        b = panorama_rgb[..., 2]
        d_g = float(self.VIS_GREEN_DELTA)
        green_pix = (g > r + d_g) & (g > b + d_g)
        green_frac_per_col = green_pix.sum(axis=0).astype(np.float32) / float(h)
        solid_cols = green_frac_per_col > float(self.VIS_GREEN_SOLID_THRESHOLD)

        spikes: list[tuple[int, int, float]] = []
        if not solid_cols.any():
            return spikes, solid_cols

        # Regroupe les colonnes solid-green adjacentes en piques distincts
        cols_idx = np.where(solid_cols)[0]
        current: list[int] = [int(cols_idx[0])]
        for c in cols_idx[1:]:
            c = int(c)
            if c - current[-1] <= 1:
                current.append(c)
            else:
                center = int(np.mean(current))
                width = max(2, current[-1] - current[0] + 1)
                intensity = float(green_frac_per_col[current].max())
                spikes.append((center, width, intensity))
                current = [c]
        center = int(np.mean(current))
        width = max(2, current[-1] - current[0] + 1)
        intensity = float(green_frac_per_col[current].max())
        spikes.append((center, width, intensity))
        return spikes, solid_cols

    def _eye_spike_mask(self, eye_img01: np.ndarray, eye: str) -> np.ndarray:
        """Renvoie un masque par-œil pour le DEBUG OVERLAY uniquement.

        Marque les colonnes qui appartiennent à une paire d'arêtes verticales
        détectée comme pique. Le _vision_step utilise une détection sur le
        panorama complet, plus précise (les pics au centre-binoculaire sont
        bien vus). Ici on duplique la détection par œil pour visualisation.
        """
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

        edge_mask, _, _ = self._detect_vertical_edges(roi)
        density = self._vertical_edge_density(edge_mask)
        edge_spikes = self._find_spike_columns(density)
        _green_spikes, solid_cols = self._find_solid_green_spikes(roi)

        mask = np.zeros((h, w), dtype=bool)
        for center, width, _intensity in edge_spikes:
            half = max(1, width // 2 + 1)
            c_lo = max(0, center - half)
            c_hi = min(w, center + half + 1)
            mask[:, c_lo:c_hi] = True
        # Colonnes solid-green : on marque les cols saturées directement
        if solid_cols.shape[0] == w:
            mask[:, solid_cols] = True
        return mask

    def _vision_step(self, sim: MiniprojectSimulation) -> tuple[float, float]:
        """Calcule (obs_size, obs_x) par détection d'arêtes verticales.

        Panorama = [œil_gauche_ROI | œil_droit_ROI] concaténés. Le centre
        tombe sur le frontal binoculaire. La détection se fait sur le panorama
        complet RGB (Sobel + verticality filter + pairing d'arêtes).

        Retour :
          obs_size ∈ [0, 1] : intensité × largeur du pic le plus prioritaire.
          obs_x    ∈ [-1, +1] : position centroïde du pic prioritaire.
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

        # Crop ROI par œil
        def _extract_roi(img: np.ndarray, eye: str) -> np.ndarray:
            h_full, w_full = img.shape[:2]
            r0 = int(self.VIS_ROI_R0 * h_full)
            r1 = int(self.VIS_ROI_R1 * h_full)
            if eye == "left":
                c0 = int(self.VIS_ROI_C0_LEFT * w_full)
                c1 = int(self.VIS_ROI_C1_LEFT * w_full)
            else:
                c0 = int(self.VIS_ROI_C0_RIGHT * w_full)
                c1 = int(self.VIS_ROI_C1_RIGHT * w_full)
            return img[r0:r1, c0:c1, :]

        left_roi = _extract_roi(left_img, "left")
        right_roi = _extract_roi(right_img, "right")

        # Aligne les hauteurs
        h = min(left_roi.shape[0], right_roi.shape[0])
        if h < 2:
            return float(self._vis_obs_size), float(self._vis_obs_x)
        left_roi = left_roi[:h]
        right_roi = right_roi[:h]

        panorama_rgb = np.concatenate([left_roi, right_roi], axis=1)
        h_p, w_p = panorama_rgb.shape[:2]
        if w_p < 2:
            return float(self._vis_obs_size), float(self._vis_obs_x)

        # Détection arêtes verticales sur le panorama complet
        edge_mask, _mag, _vert = self._detect_vertical_edges(panorama_rgb)
        density = self._vertical_edge_density(edge_mask)
        edge_spikes = self._find_spike_columns(density)

        # Détection complémentaire : colonnes solid-green (piques massifs/proches
        # qui sortent du cadre ou trop larges pour le pairing d'arêtes).
        green_spikes, _green_cols = self._find_solid_green_spikes(panorama_rgb)

        # Fusion : on prend l'union des deux listes de candidats
        all_spikes = list(edge_spikes) + list(green_spikes)

        # Stocke tous les spikes (pos, strength) pour le repulsion field.
        self._vis_all_spikes = []
        for center, width, intensity in all_spikes:
            p = (center / float(max(1, w_p - 1))) * 2.0 - 1.0
            p = float(np.clip(p, -1.0, 1.0))
            s = min(1.0, (float(width) * float(intensity)) / 10.0)
            self._vis_all_spikes.append((p, s))

        size_raw = 0.0
        x_raw = 0.0
        if all_spikes:
            best_score = -1.0
            best_center = w_p // 2
            best_size = 0.0
            for center, width, intensity in all_spikes:
                pos = (center / float(max(1, w_p - 1))) * 2.0 - 1.0
                centrality = 1.0 - pos * pos
                # Score d'impact = largeur × intensité × centralité
                score = float(width) * float(intensity) * float(centrality)
                if score > best_score:
                    best_score = score
                    best_center = center
                    best_size = min(1.0, (float(width) * float(intensity)) / 10.0)
            x_raw = (best_center / float(max(1, w_p - 1))) * 2.0 - 1.0
            x_raw = float(np.clip(x_raw, -1.0, 1.0))
            size_raw = float(best_size)

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

            # Visualiser également le masque d'arêtes verticales (jaune pâle)
            # pour distinguer la détection brute (arêtes) du résultat (pics).
            roi_rgb = img01[r0:r1, c0:c1]
            edge_mask, _, _ = self._detect_vertical_edges(roi_rgb)
            if edge_mask.shape == roi_view.shape[:2]:
                yellow = np.array([220, 220, 60], dtype=np.float32)
                roi_view[edge_mask] = (
                    roi_view[edge_mask].astype(np.float32) * 0.5 + yellow * 0.5
                ).astype(np.uint8)

            if spike_roi.shape == roi_view.shape[:2]:
                # Rouge vif : colonnes de pics détectés
                roi_view[spike_roi] = np.array([255, 30, 30], dtype=np.uint8)

            cyan = np.array([0, 220, 220], dtype=np.uint8)
            overlay[r0 : r0 + 1, c0:c1, :] = cyan
            overlay[r1 - 1 : r1, c0:c1, :] = cyan
            overlay[r0:r1, c0 : c0 + 1, :] = cyan
            overlay[r0:r1, c1 - 1 : c1, :] = cyan

            out_eyes.append(overlay)

        result = np.concatenate(out_eyes, axis=1).astype(np.uint8)
        self._vis_debug_overlay = result
        return result
