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

    # --- olfaction (navigation + stop) ---
    # Implémentation 1:1 de notebooks/week4/solutions_olfaction.ipynb :
    #   attractive_intensities = avg(odor[:,0].reshape(2,2), axis=0, weights=[9,1])
    #   attractive_bias = ATTRACTIVE_GAIN * (L - R) / mean
    #   effective_norm  = tanh(bias²) * sign(bias)   ∈ [-1, +1]
    # Le gain -500 est crucial : il transforme la moindre asymétrie L vs R
    # en effective_norm saturé à ±1, donc le pivot est franc dès qu'il y a
    # un soupçon de gradient. Sans ça, la mouche reste droit-droit.
    PALP_WEIGHT = 9
    ANTENNA_WEIGHT = 1
    EPS_ODOR = 1e-12
    STOP_ODOR_THRESHOLD = 5e-3          # stop quand mean_odor > seuil = au but
    # EMA temporel sur les capteurs olfactifs bruts. Sous vent random
    # (changement d'angle toutes les 100 ms, uniform 0-360°), le plume
    # balaie toutes les directions ; moyenner ~1.5 s récupère la concentration
    # time-averaged qui est isotrope (radiale, centrée sur la banane).
    # α = 0.017 → τ ≈ 1.5 s à 40 Hz (DECISION_INTERVAL_S = 0.025).
    # Activé UNIQUEMENT sur niveaux avec vent (L3/L4). Sans vent, le plume
    # est déjà stable et lisser ralentirait inutilement la réactivité.
    ODOR_EMA_ALPHA_WIND = 0.017
    # Stop physique (ground-truth banana_xy) : sécurité finale. La navigation
    # reste 100% olfactive, banana_xy n'est utilisé QUE pour ce critère d'arrêt.
    STOP_DIST = 2.0                     # mm
    # Gain "attractif" du notebook (signe négatif = source attractive).
    OLF_ATTRACTIVE_GAIN = -500.0
    # effective_norm → target_bias : tanh(target_bias) sera ensuite calculé
    # côté downstream, donc 3.0 ⇒ tanh saturé à ±0.995.
    OLF_TARGET_BIAS_SCALE = 3.0
    # Amplification raw_asym → bearing. NOTE : à distance (~30 mm), raw_asym
    # est tout petit (~0.03) ET son signe est bruité. Amplifier (ex K=10) fait
    # qu'ALIGN engage un pivot dur dans une direction potentiellement fausse
    # pendant 1.5 s → mouche regarde dans la mauvaise direction puis fonce
    # tout droit. K=1 (pas d'amplification) ⇒ ALIGN exit immédiat, la mouche
    # avance et la chimiotaxie corrige la trajectoire en mode GO.
    OLF_BEARING_GAIN = 1.0
    # Seuils de proximité (en mean_odor) — remplacent dist_to_banana :
    ODOR_CLOSE_THRESHOLD = 1e-5         # mean_odor > X → gain serré (TARGET_STEER_GAIN_CLOSE)
    ODOR_SPRINT_THRESHOLD = 1e-4        # mean_odor > X → sprint final (pas de répulsion)

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
    TARGET_STEER_GAIN = 4.0
    TARGET_STEER_GAIN_CLOSE = 8.0
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

    # --- Tilt freeze (Option B v2) : détection précoce de basculement ---
    # v2 améliorations vs v1 :
    #   - Détection PLUS PRÉCOCE (0.60 → 0.75) : catch le tilt avant la chute
    #   - Drive total à 0 (pas 0.10) : legs immobiles = ancrage MAXIMAL
    #   - Min duration : reste en freeze ≥ 15 décisions même si upright remonte
    #   - Exit plus strict (0.85 → 0.92) : ne reprend que totalement stabilisée
    #   - Boost TILT_LEAN_GAIN ×2 pendant freeze : compensation active du roll
    TILT_FREEZE_ENABLE = True
    TILT_FREEZE_ENTER_UPRIGHT = 0.75       # déclenche freeze + tôt
    TILT_FREEZE_EXIT_UPRIGHT = 0.92        # exit plus strict
    TILT_FREEZE_DRIVE = 0.0                # full stop : legs immobiles
    TILT_FREEZE_MIN_DECISIONS = 15         # reste freeze min N décisions (~0.4s)
    TILT_FREEZE_LEAN_BOOST = 2.0           # multiplie TILT_LEAN_GAIN pendant freeze

    # --- grip boost (merge 19/05 : grip doux kreslo + startup REPULSION) ---
    # kreslo : grip FAIBLE appliqué uniquement aux pattes en appui — ne fige
    # jamais la démarche (le grip fort toutes-pattes de REPULSION bloquait la
    # translation). On garde néanmoins deux exceptions à grip FORT LOCAL :
    #   - startup (stabilisation au spawn sur le terrain),
    #   - phase de recul head-collision (besoin de coller au sol pour reculer).
    TERRAIN_GRIP_FORCE = 12.0   # grip fort partout
    WIND_GRIP_FORCE = 12.0      # grip fort partout (vent)
    COLLISION_BACKUP_GRIP_FORCE = 12.0  # grip fort LOCAL pendant le recul head-collision (x2)
    STARTUP_GRIP_FORCE = 28.0   # grip fort LOCAL pour stabiliser au spawn (x2)
    # Grip fort sur toutes les pattes pendant les N premiers SIM STEPS (pas
    # décisions), pour stabiliser la mouche dès le spawn sur le terrain.
    STARTUP_MAX_GRIP_STEPS = 7000   # 0.7 s @ timestep 1e-4

    # --- orientation safety ---
    TERRAIN_UPRIGHT_TILT_WARN = 0.35    # plus permissif (~ 70° au lieu de ~60°)
    TERRAIN_TILT_RESET_HOLD = 80        # ~2s : laisse + de temps pour récup
    TERRAIN_FLIP_WEAK_UPRIGHT = 0.12
    TERRAIN_FLIP_RESET_HOLD = 60        # ~1.5s pour flip total

    # --- VISION (Level 2+) ---------------------------------------------------
    # Activée seulement si _enable_grass=True. Détection panoramique
    # bi-oculaire fusionnée en un seul vecteur de colonnes couvrant le frontal-large.
    VISION_ENABLE = True

    # ROI = vision raw COMPLÈTE (pas de crop). Chaque œil utilise toute
    # sa rétine pour la détection. Le panorama bi-oculaire concaténé couvre
    # alors le champ visuel total des deux yeux.
    VIS_ROI_R0 = 0.0
    VIS_ROI_R1 = 1.0
    VIS_ROI_C0_LEFT = 0.0
    VIS_ROI_C1_LEFT = 1.0
    VIS_ROI_C0_RIGHT = 0.0
    VIS_ROI_C1_RIGHT = 1.0

    # EMA pour stabilité du signal (bas = lissage plus fort, anti-wind sway)
    VIS_EMA = 0.30

    # --- Détection par excédent de silhouette ---
    # Pipeline (cf. _vision_step) :
    #   1. green_mask = (g > r+δ) ET (g > b+δ)              → pixels foreground
    #   2. silhouette_height[c] = h - row(topmost vert)     → courbe 1D
    #   3. baseline[c] = rolling-min sur fenêtre large       → niveau du terrain local
    #   4. excess[c] = silhouette_height[c] - baseline[c]    → "ça dépasse de combien"
    #   5. score[c] = excess²[c] × centralité²(c)            → privilégie centre & gros
    #   6. on garde LA colonne avec score max, bbox tight    → un seul rectangle
    # S'adapte naturellement aux pentes (la baseline suit le terrain) et capte
    # les pics même attachés au sol (pas de masquage horizon).
    VIS_GREEN_DELTA = 0.04125              # seuil chroma vert (g > r+δ et g > b+δ)
    VIS_BASELINE_WINDOW_FRAC = 0.35        # fenêtre rolling-min = X×largeur panorama
    VIS_EXCESS_MIN_PIXELS = 6              # excess en pixels mini pour considérer
    VIS_BBOX_REL_HEIGHT = 0.5              # bbox = cols où excess > X × max_excess
    VIS_CENTRAL_BOOST = 2.0                # exposant centralité : score = excess² × (1-p²)^B
    # Filtre aspect-ratio : un PIC est plus haut que large (h/w ≥ ce ratio).
    # Une COLLINE est large et basse → rejetée. Calculé sur la bbox de l'excès.
    VIS_PEAK_MIN_ASPECT_RATIO = 1.5

    # --- Steering : seuil de proximité banane (sprint final) ---
    AVOID_DISABLE_CLOSE_DIST = 8.0  # < 8 m de la banane : on fonce sans répulsion

    # --- Repulsion field steering ---
    REPULSION_FIELD_ENABLE = True
    REPULSION_GAIN = 12.0                  # MAX agressif sur la répulsion
    REPULSION_FALLOFF_ALPHA = 1.2          # exp(-alpha × p²) : étendue plus large
    REPULSION_CENTRAL_EPS = 0.12           # zone "central" un peu plus large
    REPULSION_CENTRAL_BOOST = 200          # gros boost central
    REPULSION_BANANA_BLEND = 0.50          # pull banane très forte pendant l'esquive

    # === Seuils en pixels verts (taille apparente du pic) ===
    # Augmentés pour ne réagir qu'aux pics suffisamment proches/menaçants.
    REPULSION_MIN_PIXELS = 2000            # esquive douce si pic ≥ X pixels
    REPULSION_PIVOT_PIXELS = 3000          # pivot serré si pic ≥ X pixels
    REPULSION_NORM_PIXELS = 5000           # normalisation interne s = min(1, px/NORM)

    # Saturation universelle bias (anti-violent turn). Réduit de 10→6 pour
    # adoucir l'esquive sur le chemin de répulsion normal (le pivot dur en cas
    # de pic central reste inchangé, c'est juste le « turn doux » qui diminue).
    VIS_TURN_MAX = 6.0

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
        self._request_reset = False
        self._last_target_bearing = 0.0
        self._debug_decisions = 0

        # Vision state (EMA)
        self._vis_obs_size = 0.0       # taille pic prioritaire (pour overlay debug)
        self._vis_obs_x = 0.0          # position pic prioritaire (pour overlay debug)
        self._vis_debug_overlay = None
        # Liste de tous les spikes détectés au dernier frame (pour repulsion field)
        # Chaque entrée : (p ∈ [-1,+1], s ∈ [0,1], px_count, col_lo, col_hi).
        self._vis_all_spikes: list = []
        # Per-eye spike info pour l'overlay debug-vision : (col_lo, col_hi, px).
        self._vis_spikes_left: list = []
        self._vis_spikes_right: list = []

        # Initial alignment state
        self._align_done = False
        self._align_dir = 0.0
        self._align_left = 0
        self._align_initial_sign = 0.0  # signe de effective_norm au lancement d'ALIGN
        # GO-mode pivot state (hystérésis)
        self._go_pivot_active = False
        # Tilt freeze state (Option B v2)
        self._tilt_freeze_active = False
        self._tilt_freeze_left = 0
        # Head collision recovery state (séquence en phases)
        # 0 = idle, 1 = backup, 2 = pivot toward banana, 3 = arc around
        self._collision_phase = 0
        self._collision_left = 0
        self._collision_arc_dir = 1.0
        self._collision_cooldown = 0
        # Peak head force tracking entre 2 décisions (échantillonnage @ sim step)
        self._head_force_peak = 0.0

        # Position banane (UNIQUEMENT pour le stop physique à STOP_DIST).
        # La navigation reste 100 % olfactive.
        try:
            self._banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
        except Exception:
            self._banana_xy = None

        # EMA des capteurs olfactifs (None = pas encore initialisé).
        # Utilisée seulement quand self._enable_wind (cf. _read_olfaction).
        self._odor_ema = None

        # NOTE : sim.world.banana_xy interdit. Navigation = olfaction uniquement.

        # Lift fly at spawn + aligne le corps SUR LA PENTE (z_body = normale
        # terrain au point de spawn). Préserve le yaw initial choisi par le sim.
        # Le free joint stocke qpos = [x, y, z, qw, qx, qy, qz].
        try:
            import mujoco as _mj
            for _jnt in range(sim.mj_model.njnt):
                if sim.mj_model.jnt_type[_jnt] == _mj.mjtJoint.mjJNT_FREE:
                    _addr = sim.mj_model.jnt_qposadr[_jnt]
                    spawn_x = float(sim.mj_data.qpos[_addr + 0])
                    spawn_y = float(sim.mj_data.qpos[_addr + 1])
                    # Lift
                    sim.mj_data.qpos[_addr + 2] += 0.1

                    # Yaw initial (rotation autour de Z monde)
                    qw = float(sim.mj_data.qpos[_addr + 3])
                    qx = float(sim.mj_data.qpos[_addr + 4])
                    qy = float(sim.mj_data.qpos[_addr + 5])
                    qz = float(sim.mj_data.qpos[_addr + 6])
                    yaw = float(np.arctan2(
                        2.0 * (qw * qz + qx * qy),
                        1.0 - 2.0 * (qy * qy + qz * qz),
                    ))

                    # Normale du terrain au spawn (Z_body cible)
                    n = np.array([0.0, 0.0, 1.0], dtype=float)
                    world = getattr(sim, "world", None)
                    get_normal = getattr(world, "get_normal", None)
                    if callable(get_normal):
                        try:
                            nn = np.asarray(get_normal(spawn_x, spawn_y), dtype=float)
                            if nn.shape == (3,) and np.isfinite(nn).all():
                                nnorm = float(np.linalg.norm(nn))
                                if nnorm > 1e-9:
                                    n = nn / nnorm
                        except Exception:
                            pass

                    # X_body = projection du forward yaw sur le plan perpendiculaire à n
                    forward_world = np.array([
                        float(np.cos(yaw)),
                        float(np.sin(yaw)),
                        0.0,
                    ])
                    x_body = forward_world - float(np.dot(forward_world, n)) * n
                    xn = float(np.linalg.norm(x_body))
                    if xn < 1e-6:
                        # Cas dégénéré : forward parallèle à n → fallback identité
                        x_body = np.array([1.0, 0.0, 0.0])
                    else:
                        x_body = x_body / xn
                    # Y_body = n × x_body (left-handed body frame standard)
                    y_body = np.cross(n, x_body)
                    yn = float(np.linalg.norm(y_body))
                    if yn > 1e-9:
                        y_body = y_body / yn
                    # Matrice 3x3 : colonnes = (x_body, y_body, n)
                    R = np.column_stack([x_body, y_body, n])
                    # Conversion matrice → quaternion (algo standard)
                    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
                    if tr > 0.0:
                        S = 2.0 * float(np.sqrt(tr + 1.0))
                        new_qw = 0.25 * S
                        new_qx = (R[2, 1] - R[1, 2]) / S
                        new_qy = (R[0, 2] - R[2, 0]) / S
                        new_qz = (R[1, 0] - R[0, 1]) / S
                    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                        S = 2.0 * float(np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
                        new_qw = (R[2, 1] - R[1, 2]) / S
                        new_qx = 0.25 * S
                        new_qy = (R[0, 1] + R[1, 0]) / S
                        new_qz = (R[0, 2] + R[2, 0]) / S
                    elif R[1, 1] > R[2, 2]:
                        S = 2.0 * float(np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
                        new_qw = (R[0, 2] - R[2, 0]) / S
                        new_qx = (R[0, 1] + R[1, 0]) / S
                        new_qy = 0.25 * S
                        new_qz = (R[1, 2] + R[2, 1]) / S
                    else:
                        S = 2.0 * float(np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
                        new_qw = (R[1, 0] - R[0, 1]) / S
                        new_qx = (R[0, 2] + R[2, 0]) / S
                        new_qy = (R[1, 2] + R[2, 1]) / S
                        new_qz = 0.25 * S

                    sim.mj_data.qpos[_addr + 3] = float(new_qw)
                    sim.mj_data.qpos[_addr + 4] = float(new_qx)
                    sim.mj_data.qpos[_addr + 5] = float(new_qy)
                    sim.mj_data.qpos[_addr + 6] = float(new_qz)
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

            # --- Tilt freeze (Option B v2) : détection précoce + freeze ---
            # Lit uprightness, update l'état avec hystérésis + min duration,
            # override les drives en freeze.
            if self.TILT_FREEZE_ENABLE:
                try:
                    upr = float(sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)[2, 2])
                except Exception:
                    upr = 1.0
                if self._tilt_freeze_active:
                    # Décrémenter le compteur min ; ne sortir que si min écoulé
                    # ET uprightness > EXIT.
                    if self._tilt_freeze_left > 0:
                        self._tilt_freeze_left -= 1
                    if (
                        self._tilt_freeze_left <= 0
                        and upr > float(self.TILT_FREEZE_EXIT_UPRIGHT)
                    ):
                        self._tilt_freeze_active = False
                else:
                    if upr < float(self.TILT_FREEZE_ENTER_UPRIGHT):
                        self._tilt_freeze_active = True
                        self._tilt_freeze_left = int(self.TILT_FREEZE_MIN_DECISIONS)
                if self._tilt_freeze_active:
                    fd = float(self.TILT_FREEZE_DRIVE)
                    self._drives = np.array([fd, fd], dtype=float)
                    if (
                        self.DEBUG
                        and self._debug_decisions <= self.DEBUG_MAX_DECISIONS
                        and (self._debug_decisions % self.DEBUG_EVERY_DECISIONS == 0)
                    ):
                        print(
                            f"[dbg d={self._debug_decisions:4d}] mode=TILT-FREEZE "
                            f"upright={upr:.3f} left={self._tilt_freeze_left} "
                            f"drives=({fd:.3f},{fd:.3f})",
                            flush=True,
                        )
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
                gain_eff = float(self.TILT_LEAN_GAIN)
                if self._tilt_freeze_active:
                    gain_eff *= float(self.TILT_FREEZE_LEAN_BOOST)
                offset = (
                    float(self.TILT_LEAN_SIGN)
                    * gain_eff
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
            in_tilt_freeze = self._tilt_freeze_active
            if in_tilt_freeze:
                # Tilt freeze (Option B) : grip MAX sur toutes les pattes,
                # priorité absolue pour ancrer la mouche en train de tomber.
                grip_val = max(float(self.WIND_GRIP_FORCE), float(self.TERRAIN_GRIP_FORCE))
            elif in_startup_grip:
                grip_val = float(self.STARTUP_GRIP_FORCE)
                # Fix L3/L4 : pendant le startup avec vent, on prend le max
                # pour ne pas être en-dessous du grip vent normal (sinon la
                # mouche se fait souffler dès le spawn).
                if self._enable_wind:
                    grip_val = max(grip_val, float(self.WIND_GRIP_FORCE))
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
                if in_tilt_freeze:
                    # Toutes les pattes ancrées pour résister à la chute.
                    adhesion[:n] = grip_val
                elif in_startup_grip:
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
    def _read_olfaction(self, sim) -> tuple[float, float, float]:
        """Calque 1:1 sur notebooks/week4/solutions_olfaction.ipynb.

        Returns (mean_odor, effective_norm, raw_asym) avec :
          - attractive_intensities = avg(odor[:,0].reshape(2,2), axis=0,
                                         weights=[PALP_WEIGHT, ANTENNA_WEIGHT])
          - raw_asym         = (L − R) / mean   (~ proportionnel à l'angle off-axis)
          - attractive_bias  = ATTRACTIVE_GAIN × raw_asym   (notebook)
          - effective_norm   = tanh(bias²) × sign(bias)   ∈ [-1, +1]
            → sature à ±1 dès |raw_asym| ≳ 0.01 grâce au gain -500.

        Convention de signe :
          raw_asym, effective_norm < 0 ⇔ banane à GAUCHE (index 0)
          raw_asym, effective_norm > 0 ⇔ banane à DROITE (index 1)
        """
        try:
            odor_intensities = np.asarray(sim.get_olfaction(sim.fly.name), dtype=float)
        except Exception:
            return 0.0, 0.0, 0.0
        if odor_intensities.size < 4:
            return 0.0, 0.0, 0.0
        # EMA temporel en présence de vent : le plume tourne très vite (sim
        # randomise l'angle toutes les 100 ms) ; moyenner ~1.5 s récupère le
        # plume "time-averaged" qui est isotrope et exploitable comme sans vent.
        if self._enable_wind:
            if self._odor_ema is None:
                self._odor_ema = odor_intensities.copy()
            else:
                a = float(self.ODOR_EMA_ALPHA_WIND)
                self._odor_ema = (1.0 - a) * self._odor_ema + a * odor_intensities
            odor_for_formula = self._odor_ema
        else:
            odor_for_formula = odor_intensities
        # Canal 0 = attractif (banane). reshape (2,2) → [[palp0, palp1], [ant0, ant1]]
        # weighted avg axis=0 → [9·palp0 + ant0, 9·palp1 + ant1] / 10
        attractive_intensities = np.average(
            odor_for_formula[:, 0].reshape(2, 2),
            axis=0,
            weights=[float(self.PALP_WEIGHT), float(self.ANTENNA_WEIGHT)],
        )
        mean_odor = float(attractive_intensities.mean())
        if mean_odor <= 0.0:
            return 0.0, 0.0, 0.0
        diff = float(attractive_intensities[0] - attractive_intensities[1])
        raw_asym = diff / mean_odor                                # ∈ [-2, +2] en pratique
        attractive_bias = float(self.OLF_ATTRACTIVE_GAIN) * raw_asym
        effective_norm = float(np.tanh(attractive_bias ** 2) * np.sign(attractive_bias))
        return mean_odor, effective_norm, float(raw_asym)

    def _compute_target_bias(
        self, sim, _thorax_xy=None
    ) -> tuple[float, float]:
        """Cap banane via OLFACTION (formule notebook).

        target_bias = effective_norm × SCALE   (saturé, pousse fort vers la source)
        bearing     = raw_asym × π/2           (smooth, reflète l'erreur d'angle)

        Cette séparation évite le bug 360° : effective_norm sature à ±1 dès
        qu'il y a 1 % d'asymétrie, donc un bearing dérivé directement de
        effective_norm garderait ALIGN/GO_PIVOT activés en permanence.
        En utilisant raw_asym (linéaire en l'erreur) pour le bearing, ALIGN
        et GO_PIVOT se désengagent dès que la mouche est ~face à la source.

        Convention controller (inchangée) :
          target_bias < 0 → tourne à GAUCHE
          bearing     > 0 → banane à GAUCHE (signe opposé à target_bias)
        """
        mean_odor, effective_norm, raw_asym = self._read_olfaction(sim)
        target_bias = float(effective_norm) * float(self.OLF_TARGET_BIAS_SCALE)
        # Amplifie raw_asym (typiquement ~0.03 au spawn) avant scaling angulaire,
        # puis clip à ±1 pour borner |bearing| ≤ π/2. Garde la proportionnalité
        # (bearing→0 quand asym→0) mais avec une dynamique utilisable par ALIGN.
        scaled = float(np.clip(raw_asym * float(self.OLF_BEARING_GAIN), -1.0, 1.0))
        bearing = -scaled * (np.pi / 2.0)
        return target_bias, bearing

    # ------------------------------------------------------------------
    def _compute_drives(self, sim) -> np.ndarray:
        if self._stopped:
            return np.array([0.0, 0.0])

        # Navigation = OLFACTION uniquement. banana_xy n'est utilisé que pour
        # le critère d'arrêt physique (STOP_DIST) ci-dessous.
        try:
            thorax_xy = sim.get_body_positions(sim.fly.name)[self._thorax_idx, :2]
        except Exception:
            thorax_xy = sim.mj_data.xpos[self._thorax_body_id, :2]
        thorax_xy = np.asarray(thorax_xy, dtype=float)

        # ---- Stop physique : ||thorax - banana_xy|| ≤ STOP_DIST ----
        if self._banana_xy is not None:
            dist_to_banana = float(np.linalg.norm(thorax_xy - self._banana_xy))
            if dist_to_banana <= float(self.STOP_DIST):
                print(
                    f"[STOP REASON] DIST d={self._debug_decisions} dist={dist_to_banana:.2f}",
                    flush=True,
                )
                self._stopped = True
                return np.array([0.0, 0.0])

        # Lecture olfaction unique pour ce tick.
        mean_odor, effective_norm, _raw_asym = self._read_olfaction(sim)
        # Stop par odeur : signal saturé = on est sur la banane (fallback).
        if mean_odor > self.STOP_ODOR_THRESHOLD:
            print(f"[STOP REASON] ODOR d={self._debug_decisions} mean_odor={mean_odor:.3e}", flush=True)
            self._stopped = True
            return np.array([0.0, 0.0])
        # Proxies utilisés à la place de dist_to_banana.
        close_to_banana = mean_odor > float(self.ODOR_SPRINT_THRESHOLD)

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

        # ---- Cap banane (via olfaction asymétrie L/R) ----
        target_bias, bearing = self._compute_target_bias(sim, thorax_xy)
        self._last_target_bearing = bearing

        # ---- Vision panorama (Level 2+) ----
        obs_size = 0.0
        obs_x = 0.0
        if self.VISION_ENABLE and self._enable_grass:
            obs_size, obs_x = self._vision_step(sim)

        # ---- Initial alignment ----
        # ALIGN FORCÉ basé sur signe(effective_norm) au lancement.
        # À distance, |bearing| est trop petit pour engager ALIGN via le seuil
        # ALIGN_BEARING_OK. Mais effective_norm est saturé à ±1 dès qu'il y a
        # une asymétrie L/R (gain -500 du notebook) → signe FIABLE pour décider
        # le côté. La phase pivote en place jusqu'à ce que le signe s'inverse
        # (la mouche a dépassé l'axe banane) ou timeout.
        if (
            self.ALIGN_INITIAL_ENABLE
            and not self._align_done
        ):
            if self._align_dir == 0.0:
                # Premier appel : on attend un signal clair pour fixer la direction.
                if abs(effective_norm) >= 0.5:
                    # effective_norm < 0 ⇔ banane à GAUCHE → align_dir = +1 (drives=[min,max] = pivot gauche)
                    # effective_norm > 0 ⇔ banane à DROITE → align_dir = -1 (drives=[max,min] = pivot droite)
                    self._align_dir = -float(np.sign(effective_norm))
                    self._align_initial_sign = float(np.sign(effective_norm))
                    self._align_left = int(self.ALIGN_MAX_DECISIONS)
                else:
                    # Pas de signal exploitable au spawn → on saute ALIGN
                    self._align_done = True

            # Sortie : signe d'effective_norm s'inverse (dépassé l'axe banane)
            # OU effective_norm devient nul (parfaitement aligné, rare en pratique)
            # OU timeout.
            cur_sign = float(np.sign(effective_norm)) if abs(effective_norm) >= 0.5 else 0.0
            sign_flipped = (
                cur_sign != 0.0
                and self._align_initial_sign != 0.0
                and cur_sign != self._align_initial_sign
            )
            if not self._align_done and (sign_flipped or self._align_left <= 0):
                self._align_done = True
                self._align_left = 0

            if not self._align_done:
                self._align_left -= 1
                _slope_forward = 0.0
                _slope_lateral = 0.0
                _slope_mag = 0.0
                if self._enable_terrain:
                    max_drive = float(self.ALIGN_MAX_DRIVE_TERRAIN)
                    min_side = float(self.ALIGN_MIN_SIDE_TERRAIN)
                    # Slope-adapt : sur pente forte, on freine le max_drive
                    # pour éviter que le pivot HARD ne fasse basculer la mouche.
                    try:
                        _slope_forward, _slope_lateral, _slope_mag = self._get_slope_signals(sim)
                    except Exception:
                        pass
                    if _slope_mag > 0.0:
                        brake = 1.0 / (
                            1.0 + float(self.STEEP_BRAKE) * float(_slope_mag)
                        )
                        max_drive = max(min_side, max_drive * brake)
                else:
                    max_drive = float(self.MAX_DRIVE)
                    min_side = float(self.MIN_SIDE_DRIVE)
                if self._align_dir > 0:
                    drives = np.array([min_side, max_drive], dtype=float)
                else:
                    drives = np.array([max_drive, min_side], dtype=float)
                # Slope-bias additif : compense la dérive latérale due à la pente
                # en boostant la roue uphill et réduisant la roue downhill.
                if self._enable_terrain and _slope_mag > 0.0:
                    downhill = max(0.0, -_slope_forward)
                    slope_bias = -float(self.SLOPE_STEER_GAIN) * float(_slope_lateral) * float(downhill)
                    slope_bias = float(np.clip(
                        slope_bias, -float(self.SLOPE_STEER_MAX), float(self.SLOPE_STEER_MAX),
                    ))
                    # Scale conservatif (0.15) pour ne pas dominer le pivot ALIGN
                    slope_diff = 0.15 * slope_bias
                    drives[0] += slope_diff
                    drives[1] -= slope_diff
                    drives = np.clip(drives, 0.0, max_drive)
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
        # AVOID. Si on est proche de la banane (odeur forte), GO direct.
        # `close_to_banana` est déjà calculé plus haut depuis mean_odor.

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
        # Si on est en REPUL et que le pic max (en pixels) dépasse le seuil PIVOT,
        # on pivote à fond.
        if sub_mode == "REPUL" and self._vis_all_spikes:
            best = max(self._vis_all_spikes, key=lambda t: t[2])
            p_max = float(best[0])
            s_max = float(best[1])
            px_max = int(best[2])
            if px_max >= int(self.REPULSION_PIVOT_PIXELS):
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
                    print(
                        f"[dbg d={self._debug_decisions:4d}] mode=REPUL-PIV "
                        f"p_max={p_max:+.3f} px_max={px_max} (s={s_max:.3f}) dir={int(dir_pivot):+d} "
                        f"odor={mean_odor:.2e} drives=({pivot_drives[0]:.3f},{pivot_drives[1]:.3f})",
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
                    print(
                        f"[dbg d={self._debug_decisions:4d}] mode=GO-PIV "
                        f"bearing={bearing:+.3f} odor={mean_odor:.2e} "
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
            print(
                f"[dbg d={self._debug_decisions:4d}] mode={sub_mode:5s} "
                f"obs_x={obs_x:+.3f} obs_sz={obs_size:.4f} "
                f"n_spikes={len(self._vis_all_spikes)} odor={mean_odor:.2e} "
                f"bearing={bearing:+.3f} target_bias={target_bias:+.3f} "
                f"bias={bias:+.3f} drives=({drives[0]:.3f},{drives[1]:.3f})",
                flush=True,
            )
        return drives

    # ------------------------------------------------------------------
    # Repulsion field steering
    # ------------------------------------------------------------------
    def _compute_repulsion_bias(
        self, target_bias: float, spikes: list
    ) -> "tuple[float, float, str, bool]":
        """Répulsion calculée UNIQUEMENT sur le pic avec le plus de pixels verts.

        spikes : list de (p, s, px_count, col_lo, col_hi)
        Seuil de déclenchement : REPULSION_MIN_PIXELS (compte de pixels verts
        dans la bbox du pic). Beaucoup plus interprétable que l'ancien s∈[0,1].

        kernel(p) = -sign(p) × s × factor
          - p > 0 (à droite)  → bias < 0 → tourne à GAUCHE
          - p < 0 (à gauche)  → bias > 0 → tourne à DROITE
          - |p| < eps (central) → boost de répulsion, sign = côté banane

        Returns:
          bias, base_drive, mode_str, active
        """
        if not spikes:
            return float(target_bias), float(self.BASE_DRIVE_FAST), "GO", False

        # Pique avec le plus de pixels (= le plus gros/proche)
        best = max(spikes, key=lambda t: t[2])
        p_max = float(best[0])
        s_max = float(best[1])
        px_max = int(best[2])
        if px_max < int(self.REPULSION_MIN_PIXELS):
            # Pic trop petit (peu de pixels verts) → ignoré
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
    # Vision : silhouette-curve peak detection
    # ------------------------------------------------------------------
    def _compute_silhouette_top(self, panorama_rgb: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        """Pour chaque colonne, calcule la row du topmost pixel foreground (vert).

        Retourne :
          silhouette_top    : (w,) row du pixel vert le plus haut par colonne
                              (= h si aucun vert dans la colonne).
          silhouette_height : (w,) hauteur de silhouette = h - silhouette_top.
        """
        h, w = panorama_rgb.shape[:2]
        r = panorama_rgb[..., 0]
        g = panorama_rgb[..., 1]
        b = panorama_rgb[..., 2]
        d_g = float(self.VIS_GREEN_DELTA)
        green_mask = (g > r + d_g) & (g > b + d_g)
        has_fg = green_mask.any(axis=0)
        silhouette_top = np.where(has_fg, green_mask.argmax(axis=0), h).astype(np.int32)
        silhouette_height = (h - silhouette_top).astype(np.int32)
        return silhouette_top, silhouette_height


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

        # === DÉTECTION PAR EXCÉDENT DE SILHOUETTE ============================
        # 1. green_mask sur tout le panorama
        # 2. silhouette_height[c] = h - row du topmost vert dans col c (0 = sky pur)
        # 3. baseline[c] = rolling-min sur fenêtre large = niveau du terrain local
        # 4. excess[c] = silhouette_height[c] - baseline[c]
        # 5. score[c] = excess² × centralité²
        # 6. on garde la col du score max ; bbox = colonnes adjacentes où
        #    excess > VIS_BBOX_REL_HEIGHT × max_excess.
        r_chan = panorama_rgb[..., 0]
        g_chan = panorama_rgb[..., 1]
        b_chan = panorama_rgb[..., 2]
        d_g = float(self.VIS_GREEN_DELTA)
        green_mask = (g_chan > r_chan + d_g) & (g_chan > b_chan + d_g)
        norm_px = max(1.0, float(self.REPULSION_NORM_PIXELS))

        detected_peaks: list = []  # 0 ou 1 entrée : (p, n_pixels, col_lo, col_hi, row_lo, row_hi)
        if green_mask.any():
            # 2) silhouette top par colonne
            has_fg = green_mask.any(axis=0)
            silhouette_top = np.where(has_fg, green_mask.argmax(axis=0), h_p).astype(np.int32)
            silhouette_height = (h_p - silhouette_top).astype(np.int32)
            # silhouette_height ∈ [0, h_p] ; 0 si pas de FG dans la col.

            # 3) baseline locale = rolling-min sur fenêtre large
            # On exclut les cols sans FG (sky pur) pour ne pas tirer la baseline à 0
            sentinel = h_p + 1
            sil_for_min = np.where(has_fg, silhouette_height, sentinel)
            win = max(3, int(float(self.VIS_BASELINE_WINDOW_FRAC) * w_p))
            if _SCIPY_NDIMAGE is not None:
                baseline = _SCIPY_NDIMAGE.minimum_filter1d(sil_for_min, size=win, mode="nearest")
            else:
                # Fallback : pad puis min via convolution naïve
                pad = win // 2
                padded = np.concatenate([sil_for_min[:pad][::-1], sil_for_min, sil_for_min[-pad:][::-1]])
                baseline = np.array([padded[i:i+win].min() for i in range(w_p)], dtype=np.int32)
            # Si la fenêtre n'a rencontré que des sentinelles, la baseline = sentinel.
            # Clamp pour qu'elle ne dépasse pas la silhouette réelle.
            baseline = np.minimum(baseline, silhouette_height)
            baseline = np.maximum(baseline, 0)

            # 4) excess = ce qui dépasse la baseline locale
            excess = (silhouette_height - baseline).astype(np.int32)

            # 5) score = excess² × centralité²
            cols = np.arange(w_p, dtype=np.float32)
            p_per_col = (cols / float(max(1, w_p - 1))) * 2.0 - 1.0
            centrality = np.maximum(0.0, 1.0 - p_per_col * p_per_col)
            boost = float(self.VIS_CENTRAL_BOOST)
            score = (excess.astype(np.float32) ** 2) * (centrality ** boost)

            # 6) max + bbox
            best_col = int(score.argmax())
            max_excess = int(excess[best_col])
            if max_excess >= int(self.VIS_EXCESS_MIN_PIXELS):
                thresh = float(self.VIS_BBOX_REL_HEIGHT) * float(max_excess)
                # Scan gauche
                left = best_col
                while left > 0 and excess[left - 1] > thresh:
                    left -= 1
                # Scan droite
                right = best_col
                while right < w_p - 1 and excess[right + 1] > thresh:
                    right += 1
                col_lo = int(left)
                col_hi = int(right + 1)
                # Bbox rows : du sommet du pic au niveau de la baseline locale
                row_lo = int(silhouette_top[col_lo:col_hi].min())
                row_hi = int(h_p - int(baseline[col_lo:col_hi].min()))
                row_hi = max(row_lo + 1, min(row_hi, h_p))
                # FILTRE ASPECT-RATIO : un pic doit être plus haut que large.
                # Une colline est large + basse → ratio h/w faible → rejeté.
                bbox_w = max(1, col_hi - col_lo)
                bbox_h = max(1, row_hi - row_lo)
                aspect = bbox_h / float(bbox_w)
                if aspect >= float(self.VIS_PEAK_MIN_ASPECT_RATIO):
                    n_px = int(green_mask[:, col_lo:col_hi].sum())
                    p = (float(best_col) / float(max(1, w_p - 1))) * 2.0 - 1.0
                    p = float(np.clip(p, -1.0, 1.0))
                    detected_peaks.append((p, n_px, col_lo, col_hi, row_lo, row_hi))

        # Stockage uniforme pour la suite. _vis_all_spikes = (p, s, px, cl, ch)
        # consommé par _compute_repulsion_bias et REPUL-PIVOT.
        # _vis_spikes_left/right = (col_lo, col_hi, row_lo, row_hi, px) pour
        # l'overlay debug (coords IMAGE œil, pas panorama).
        self._vis_all_spikes = []
        self._vis_spikes_left = []
        self._vis_spikes_right = []
        spike_debug_lines = []
        w_left = left_roi.shape[1]
        h_left_eye_full = left_img.shape[0]
        w_left_eye_full = left_img.shape[1]
        w_right_eye_full = right_img.shape[1]
        c0_left_px = int(self.VIS_ROI_C0_LEFT * w_left_eye_full)
        c0_right_px = int(self.VIS_ROI_C0_RIGHT * w_right_eye_full)
        r0_eye_px = int(self.VIS_ROI_R0 * h_left_eye_full)  # même r0 pour les 2 yeux
        for p, px_count, col_lo, col_hi, row_lo, row_hi in detected_peaks:
            s = min(1.0, float(px_count) / norm_px)
            self._vis_all_spikes.append((p, s, px_count, col_lo, col_hi))
            eye_row_lo = r0_eye_px + int(row_lo)
            eye_row_hi = r0_eye_px + int(row_hi)
            if col_lo < w_left:
                e_lo = c0_left_px + col_lo
                e_hi = c0_left_px + min(col_hi, w_left)
                self._vis_spikes_left.append((e_lo, e_hi, eye_row_lo, eye_row_hi, px_count))
            if col_hi > w_left:
                start = max(col_lo, w_left) - w_left
                end = col_hi - w_left
                e_lo = c0_right_px + start
                e_hi = c0_right_px + end
                self._vis_spikes_right.append((e_lo, e_hi, eye_row_lo, eye_row_hi, px_count))
            spike_debug_lines.append((p, px_count))

        # DEBUG print : marque M = passe le seuil esquive, P = passe pivot
        if self.DEBUG and self._debug_decisions % self.DEBUG_EVERY_DECISIONS == 0:
            min_pix = int(self.REPULSION_MIN_PIXELS)
            piv_pix = int(self.REPULSION_PIVOT_PIXELS)
            if spike_debug_lines:
                spike_debug_lines.sort(key=lambda x: -x[1])
                parts = []
                for p, px in spike_debug_lines[:6]:
                    mark = "P" if px >= piv_pix else ("M" if px >= min_pix else "")
                    parts.append(f"({p:+.2f}:{px}{mark})")
                print(
                    f"[VIS-PEAKS d={self._debug_decisions:4d}] n={len(spike_debug_lines):2d} "
                    f"thresh(M/P)={min_pix}/{piv_pix} " + " ".join(parts),
                    flush=True,
                )
            else:
                print(
                    f"[VIS-PEAKS d={self._debug_decisions:4d}] n=0 (aucun pic de prominence détecté)",
                    flush=True,
                )

        # obs_size / obs_x legacy : stats agrégées du pic le plus impactant.
        size_raw = 0.0
        x_raw = 0.0
        if detected_peaks:
            best_score = -1.0
            for p_blob, n_px, _cl, _ch, _rl, _rh in detected_peaks:
                centrality = 1.0 - p_blob * p_blob
                score = float(n_px) * float(centrality)
                if score > best_score:
                    best_score = score
                    x_raw = float(np.clip(p_blob, -1.0, 1.0))
                    size_raw = min(1.0, float(n_px) / norm_px)

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

            base = (raw.astype(np.float32) * 0.45).astype(np.uint8)
            overlay = base.copy()
            roi_view = overlay[r0:r1, c0:c1, :]

            # Ligne jaune sur le top de chaque silhouette (= "ce que la mouche voit
            # comme contour foreground/background"). Calculé exactement comme dans
            # _vision_step pour rester cohérent.
            roi_rgb = raw[r0:r1, c0:c1, :].astype(np.float32) / 255.0
            _silh_top, silh_h = self._compute_silhouette_top(roi_rgb)
            roi_h = roi_view.shape[0]
            roi_w = roi_view.shape[1]
            yellow = np.array([220, 220, 60], dtype=np.uint8)
            for col in range(roi_w):
                row = int(_silh_top[col])
                if 0 <= row < roi_h:
                    roi_view[row, col, :] = yellow

            # Cadre cyan du ROI
            cyan = np.array([0, 220, 220], dtype=np.uint8)
            overlay[r0 : r0 + 1, c0:c1, :] = cyan
            overlay[r1 - 1 : r1, c0:c1, :] = cyan
            overlay[r0:r1, c0 : c0 + 1, :] = cyan
            overlay[r0:r1, c1 - 1 : c1, :] = cyan

            # ROUGE : pics ACTUELLEMENT détectés par la détection silhouette-peaks
            # (= ceux qui peuvent déclencher esquive). Coords IMAGE œil (déjà
            # remappées dans _vision_step). Ligne rouge verticale sur le bbox
            # du pic, du row_lo au row_hi, sur toute la largeur du bbox.
            spike_list = self._vis_spikes_left if eye == "left" else self._vis_spikes_right
            min_pix = int(self.REPULSION_MIN_PIXELS)
            piv_pix = int(self.REPULSION_PIVOT_PIXELS)
            for e_lo, e_hi, e_r_lo, e_r_hi, px in spike_list:
                col_lo = max(0, min(int(e_lo), w - 1))
                col_hi = max(col_lo + 1, min(int(e_hi), w))
                row_lo = max(0, min(int(e_r_lo), h - 1))
                row_hi = max(row_lo + 1, min(int(e_r_hi), h))
                # Couleur selon seuil franchi :
                #   rouge vif      = passe seuil PIVOT (esquive serrée)
                #   orange         = passe seuil MIN  (esquive douce)
                #   rose pâle      = détecté mais sous le seuil (info seulement)
                if px >= piv_pix:
                    col = np.array([255, 30, 30], dtype=np.uint8)
                elif px >= min_pix:
                    col = np.array([255, 140, 0], dtype=np.uint8)
                else:
                    col = np.array([220, 120, 160], dtype=np.uint8)
                # Cadre rectangulaire 2px d'épaisseur
                overlay[row_lo:row_lo+2, col_lo:col_hi, :] = col
                overlay[row_hi-2:row_hi, col_lo:col_hi, :] = col
                overlay[row_lo:row_hi, col_lo:col_lo+2, :] = col
                overlay[row_lo:row_hi, col_hi-2:col_hi, :] = col

            out_eyes.append(overlay)

        result = np.concatenate(out_eyes, axis=1).astype(np.uint8)
        self._vis_debug_overlay = result
        return result
