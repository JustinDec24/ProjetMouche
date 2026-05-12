"""Controller — Levels 0, 1 et 2.

Stratégie L0/L1 : steering par gradient olfactif (canonique week4 :
gain=-500, bias²), terrain-aware (slope/grip/tilt), stuck-recovery, scan/cast
quand le signal est perdu.

Level 2 ajoute des brins d'herbe verticaux de MÊME COULEUR que le sol.
Détection : silhouette sombre dans la moitié haute du champ visuel (les
brins poussent au-dessus de l'horizon où il y a du ciel brillant).
Recovery via cfrc_ext[c_head] pour les collisions frontales en angle mort.
"""
from __future__ import annotations

import numpy as np

from miniproject.simulation import MiniprojectSimulation


class Controller:
    """Olfaction-based controller L0/L1/L2 with silhouette-vision avoidance."""

    # --- scheduling ---
    DECISION_INTERVAL_S = 0.025  # 25 ms = 40 décisions/s

    # --- olfaction ---
    PALP_WEIGHT = 9
    ANTENNA_WEIGHT = 1
    EPS_ODOR = 1e-12
    STOP_ODOR_THRESHOLD = 5e-4
    STOP_DIST = 2.0
    ODOR_STEER_GAIN = -500.0
    ODOR_MIN_SUM = 1e-10

    # --- scan/cast when odor lost ---
    SCAN_TRIGGER_DECISIONS = 8
    SCAN_DURATION_DECISIONS = 20
    ODOR_TREND_WINDOW = 16
    ODOR_TREND_LOSS_RATIO = 0.6

    # --- drives ---
    BASE_DRIVE = 2.40
    MAX_DRIVE = 2.80
    MAX_DRIVE_TERRAIN = 2.00
    MAX_DRIVE_GRASS = 2.00  # idem L1 (no reduction)
    MIN_SIDE_DRIVE = 0.40
    MIN_SIDE_DRIVE_TERRAIN = 0.25
    MIN_DRIVE = 0.80
    MIN_DRIVE_TERRAIN = 0.60
    TURN_MOD = 0.8

    # --- terrain (Level 1) ---
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

    # --- active roll compensation (leg posture) ---
    TILT_LEAN_ENABLE = True
    TILT_LEAN_ROLL_ON = 0.10
    TILT_LEAN_ROLL_FULL = 0.40
    TILT_LEAN_GAIN = 0.30
    TILT_LEAN_SIGN = +1.0

    # --- orientation safety (terrain) ---
    TERRAIN_UPRIGHT_TILT_WARN = 0.48
    TERRAIN_TILT_RESET_HOLD = 26
    TERRAIN_FLIP_WEAK_UPRIGHT = 0.12
    TERRAIN_FLIP_RESET_HOLD = 34

    # --- stuck detection / escape (terrain trap) ---
    STUCK_MOVE_EPS = 5e-3
    STUCK_TRIGGER_DECISIONS = 25
    ESCAPE_DURATION_DECISIONS = 10

    # --- no-progress sidestep (L2 anti-grass-blade jamming) ---
    NOPROG_WINDOW = 12
    NOPROG_MIN_DELTA = 0.4
    SIDESTEP_BACKUP_DECISIONS = 5
    SIDESTEP_TURN_DECISIONS = 12
    SIDESTEP_BACKUP_DRIVE = -0.6
    SIDESTEP_DRIVE_FAST = 1.80
    SIDESTEP_DRIVE_SLOW = 0.30
    ROLL_TRIGGER_THRESH = 0.30
    ROLL_TRIGGER_HOLD = 3

    # --- initial heading alignment (TOUJOURS actif au départ) ---
    ALIGN_ENABLE = True
    ALIGN_NEED_THRESHOLD_RAD = 0.0      # 0 = align toujours
    ALIGN_TOLERANCE_RAD = 0.18          # ~10° = aligné précis
    ALIGN_MAX_DECISIONS = 80
    ALIGN_SPIN_FAST = 1.30
    ALIGN_SPIN_SLOW = -0.30
    # Transition après alignement : drives symétriques pour resynchroniser
    # le CPG de marche (sinon "lag" car les pattes étaient en pattern de pivot)
    ALIGN_TRANSITION_DECISIONS = 5
    ALIGN_TRANSITION_DRIVE = 1.50

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

    # --- vision : silhouette de brin contre le ciel ---
    ENABLE_VISION_AVOIDANCE = True
    BLADE_DARK_THRESH = 0.20
    BLADE_COUNT_THRESH = 25
    BLADE_COUNT_URGENT = 45
    BLADE_ASYM_MIN = 8
    BLADE_HOLD_FRAMES = 2
    BLADE_LATERAL_RATIO = 1.5
    BLADE_TURN_DECISIONS = 11
    BLADE_TURN_DRIVE_FAST = 1.40
    BLADE_TURN_DRIVE_SLOW = 0.10
    BLADE_COOLDOWN_DECISIONS = 4
    BLADE_STARTUP_DELAY = 8
    BLADE_TRAPPED_BACKUP_DECISIONS = 5
    BLADE_TRAPPED_BACKUP_DRIVE = -0.5

    # --- debug ---
    DEBUG = True
    DEBUG_EVERY_DECISIONS = 4
    DEBUG_MAX_DECISIONS = 5000
    DEBUG_VERBOSE = False

    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        self._decision_every = int(self.DECISION_INTERVAL_S / sim.timestep)
        self._step_count = 0
        self._drives = np.array([1.0, 1.0])
        self._stopped = False
        self._enable_terrain = bool(getattr(sim, "enable_terrain", False))
        self._enable_wind = bool(getattr(sim, "enable_wind", False))
        self._enable_grass = bool(getattr(sim, "enable_grass", False))

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

        # Vision / blade-silhouette state (L2)
        self._blade_decisions_left = 0
        self._blade_dir = +1
        self._blade_cooldown = 0
        self._blade_hold_count = 0
        self._blade_trapped_left = 0
        self._upper_half_mask: np.ndarray | None = None
        self._last_blade_signals: dict | None = None

        # No-progress / sidestep state (L2 anti-grass-blade jamming)
        self._dist_history: list[float] = []
        self._sidestep_decisions_left = 0
        self._sidestep_dir = +1
        self._sidestep_cooldown = 0
        self._roll_high_count = 0

        # Initial alignment state
        self._align_done = not self.ALIGN_ENABLE
        self._align_decisions = 0
        self._align_dir = +1
        self._align_transition_left = 0

        # Head-collision recovery state (3 phases)
        self._head_backup_left = 0
        self._head_turn_left = 0
        self._head_bypass_left = 0
        self._head_maneuver_dir = +1
        self._head_collision_cd = 0
        self._head_force_max = 0.0

        if self._enable_grass:
            try:
                retina = sim.world.fly_lookup[sim.fly.name].retina
                self._upper_half_mask = self._compute_upper_half_mask(retina)
            except Exception:
                self._upper_half_mask = None

        try:
            self._banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
        except Exception:
            self._banana_xy = None

        # Soulève la mouche pour éviter les pattes coincées au spawn
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
        # Accumuler max head force entre décisions (contacts brefs)
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
                    / max(1e-3, float(self.TILT_LEAN_ROLL_FULL) - float(self.TILT_LEAN_ROLL_ON)),
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

            grip_val = (
                float(self.WIND_GRIP_FORCE)
                if self._enable_wind
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

    # ------------------------------------------------------------------
    def _compute_drives(self, sim) -> np.ndarray:
        if self._stopped:
            return np.array([0.0, 0.0])

        try:
            thorax_xy = sim.get_body_positions(sim.fly.name)[self._thorax_idx, :2]
        except Exception:
            thorax_xy = sim.mj_data.xpos[self._thorax_body_id, :2]
        thorax_xy = np.asarray(thorax_xy, dtype=float)

        # Safety stop distance
        if self._banana_xy is not None:
            dist_to_banana = float(np.linalg.norm(thorax_xy - self._banana_xy))
            self._last_dist_to_banana = dist_to_banana
            if dist_to_banana <= float(self.STOP_DIST):
                self._stopped = True
                return np.array([0.0, 0.0])

        # ---- Head-collision recovery 3 phases (priorité haute) ----
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
        # Cooldown
        if self._head_collision_cd > 0:
            self._head_collision_cd -= 1
        elif (
            self.HEAD_COLLISION_ENABLE
            and self._enable_grass
            and self._head_body_id is not None
        ):
            head_force = float(self._head_force_max)
            if head_force > float(self.HEAD_COLLISION_FORCE_THRESH):
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
                            maneuver_dir = -1 if cross > 0 else +1
                    except Exception:
                        pass
                self._head_maneuver_dir = maneuver_dir
                if self.DEBUG:
                    print(
                        f"[HEAD-COLLISION d={self._debug_decisions}] "
                        f"F={head_force:.1f}N -> BACKUP "
                        f"(dir={maneuver_dir:+d} = banana {'gauche' if maneuver_dir<0 else 'droite'})",
                        flush=True,
                    )
                self._head_backup_left = int(self.HEAD_BACKUP_DECISIONS)
                self._blade_decisions_left = 0
                self._blade_trapped_left = 0
                self._blade_cooldown = int(self.BLADE_COOLDOWN_DECISIONS)
                self._sidestep_decisions_left = 0
                backup = float(self.HEAD_BACKUP_DRIVE)
                return np.array([backup, backup])

        # ---- Initial heading alignment ----
        if (
            not self._align_done
            and self._banana_xy is not None
        ):
            try:
                xmat_align = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
                heading_xy = xmat_align[:2, 0]
                hn = float(np.linalg.norm(heading_xy))
                to_banana = self._banana_xy - thorax_xy
                tbn = float(np.linalg.norm(to_banana))
                if tbn > 1e-9 and hn > 1e-9:
                    heading_xy = heading_xy / hn
                    to_banana_n = to_banana / tbn
                    cos_a = float(np.dot(heading_xy, to_banana_n))
                    cross = float(
                        heading_xy[0] * to_banana_n[1]
                        - heading_xy[1] * to_banana_n[0]
                    )
                    angle = float(np.arctan2(cross, cos_a))

                    if (
                        abs(angle) < float(self.ALIGN_TOLERANCE_RAD)
                        or self._align_decisions >= int(self.ALIGN_MAX_DECISIONS)
                    ):
                        if self.DEBUG:
                            print(
                                f"[ALIGN-DONE d={self._debug_decisions}] "
                                f"angle={angle:+.2f}rad after {self._align_decisions} dec "
                                f"-> transition {self.ALIGN_TRANSITION_DECISIONS} dec",
                                flush=True,
                            )
                        self._align_done = True
                        self._align_transition_left = int(self.ALIGN_TRANSITION_DECISIONS)
                    else:
                        self._align_decisions += 1
                        if angle > 0:
                            return np.array([self.ALIGN_SPIN_SLOW, self.ALIGN_SPIN_FAST])
                        else:
                            return np.array([self.ALIGN_SPIN_FAST, self.ALIGN_SPIN_SLOW])
            except Exception:
                self._align_done = True

        # ---- Transition après alignement : resync CPG ----
        if self._align_transition_left > 0:
            self._align_transition_left -= 1
            d = float(self.ALIGN_TRANSITION_DRIVE)
            return np.array([d, d])

        # ---- Stuck detection / escape ----
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

        # ---- No-progress recovery (L2 anti-grass-blade jamming) ----
        if self._sidestep_decisions_left > 0:
            self._sidestep_decisions_left -= 1
            n_left = self._sidestep_decisions_left
            if n_left >= int(self.SIDESTEP_TURN_DECISIONS):
                return np.array(
                    [self.SIDESTEP_BACKUP_DRIVE, self.SIDESTEP_BACKUP_DRIVE]
                )
            if n_left == 0:
                self._sidestep_cooldown = int(self.NOPROG_WINDOW)
                self._dist_history.clear()
            if self._sidestep_dir > 0:
                return np.array([self.SIDESTEP_DRIVE_FAST, self.SIDESTEP_DRIVE_SLOW])
            return np.array([self.SIDESTEP_DRIVE_SLOW, self.SIDESTEP_DRIVE_FAST])

        # Surveillance dist + roll trigger
        if self._enable_grass and self._banana_xy is not None:
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
                dist_past = self._dist_history[0]
                dist_now = self._dist_history[-1]
                progress = dist_past - dist_now
                if progress < float(self.NOPROG_MIN_DELTA):
                    trigger_now = True
                    trigger_prev_diff = float(self._drives[0]) - float(self._drives[1])

            if trigger_now:
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
                return np.array(
                    [self.SIDESTEP_BACKUP_DRIVE, self.SIDESTEP_BACKUP_DRIVE]
                )

        # ---- Vision : silhouette de brin (L2) ----
        if (
            self.ENABLE_VISION_AVOIDANCE
            and self._enable_grass
            and self._upper_half_mask is not None
        ):
            drive_or_none = self._blade_avoidance(sim)
            if drive_or_none is not None:
                return drive_or_none

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

        # ---- Olfaction-based steering ----
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

        # --- Scan/cast quand odeur perdue ---
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

        # Tropotaxis canonique (week4)
        if mean_odor > 0.0:
            bias_raw = float(self.ODOR_STEER_GAIN) * (odor_diff / mean_odor)
        else:
            bias_raw = 0.0
        bias = float(np.tanh(bias_raw * bias_raw)) * float(np.sign(bias_raw))

        base_drive = float(self.BASE_DRIVE)
        turn_mod = float(self.TURN_MOD)
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

        bias_norm = float(np.clip(bias, -1.0, 1.0))

        min_drive = self.MIN_DRIVE_TERRAIN if self._enable_terrain else self.MIN_DRIVE
        min_side = self.MIN_SIDE_DRIVE_TERRAIN if self._enable_terrain else self.MIN_SIDE_DRIVE
        max_drive = self.MAX_DRIVE_TERRAIN if self._enable_terrain else self.MAX_DRIVE
        if self._enable_grass:
            max_drive = min(max_drive, float(self.MAX_DRIVE_GRASS))
        base_drive = float(np.clip(base_drive, min_drive, max_drive))

        drives = np.full(2, base_drive, dtype=float)
        side = int(bias_norm > 0)
        drives[side] -= abs(bias_norm) * turn_mod * base_drive
        drives[side] = max(min_side, drives[side])
        drives = np.clip(drives, 0.0, max_drive)

        return drives

    # ------------------------------------------------------------------
    # Vision-based avoidance (Level 2) — silhouette de brin
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_upper_half_mask(retina) -> np.ndarray:
        ommatidia_id_map = np.asarray(retina.ommatidia_id_map)
        nrows = ommatidia_id_map.shape[0]
        n_omm = int(ommatidia_id_map.max())
        row_per_omm = np.full(n_omm, nrows, dtype=float)
        for i in range(1, n_omm + 1):
            ys, _ = np.where(ommatidia_id_map == i)
            if len(ys) > 0:
                row_per_omm[i - 1] = float(ys.mean())
        return row_per_omm < (nrows * 0.5)

    def _blade_avoidance(self, sim: MiniprojectSimulation) -> np.ndarray | None:
        # 0) Continuation TRAPPED
        if self._blade_trapped_left > 0:
            self._blade_trapped_left -= 1
            backup = float(self.BLADE_TRAPPED_BACKUP_DRIVE)
            return np.array([backup, backup])

        # 1) Continuation BLADE-AVOID avec REPLAN
        if self._blade_decisions_left > 0:
            try:
                ommatidia_c = sim.get_ommatidia_readouts(sim.fly.name)
                sum_int = ommatidia_c.sum(axis=-1)
                dark_c = sum_int < float(self.BLADE_DARK_THRESH)
                dL = int((dark_c[0] & self._upper_half_mask).sum())
                dR = int((dark_c[1] & self._upper_half_mask).sum())
            except Exception:
                dL, dR = 0, 0
            target_thresh = max(15, int(self.BLADE_COUNT_THRESH) // 2)
            blocked_on_target = False
            if self._blade_dir > 0 and dR > target_thresh:
                blocked_on_target = True
            elif self._blade_dir < 0 and dL > target_thresh:
                blocked_on_target = True
            if blocked_on_target:
                opposite_clear = (
                    (self._blade_dir > 0 and dL < dR * 0.7)
                    or (self._blade_dir < 0 and dR < dL * 0.7)
                )
                really_trapped = (
                    not opposite_clear
                    and dL > 50
                    and dR > 50
                )
                if opposite_clear:
                    self._blade_dir = -self._blade_dir
                    self._blade_decisions_left = int(self.BLADE_TURN_DECISIONS)
                elif really_trapped:
                    self._blade_decisions_left = 0
                    self._blade_trapped_left = int(self.BLADE_TRAPPED_BACKUP_DECISIONS)
                    self._blade_cooldown = int(self.BLADE_COOLDOWN_DECISIONS)
                    backup = float(self.BLADE_TRAPPED_BACKUP_DRIVE)
                    return np.array([backup, backup])
                else:
                    self._blade_decisions_left = 0
                    self._blade_cooldown = int(self.BLADE_COOLDOWN_DECISIONS)
                    return None
            self._blade_decisions_left -= 1
            if self._blade_dir > 0:
                return np.array([self.BLADE_TURN_DRIVE_FAST, self.BLADE_TURN_DRIVE_SLOW])
            return np.array([self.BLADE_TURN_DRIVE_SLOW, self.BLADE_TURN_DRIVE_FAST])

        # Cooldown
        if self._blade_cooldown > 0:
            self._blade_cooldown -= 1
            return None

        # 2) Récupération signaux visuels
        try:
            ommatidia = sim.get_ommatidia_readouts(sim.fly.name)
        except Exception:
            return None
        sum_intensity = ommatidia.sum(axis=-1)

        # 3) Compter pixels sombres
        dark_mask = sum_intensity < float(self.BLADE_DARK_THRESH)
        dark_L = int((dark_mask[0] & self._upper_half_mask).sum())
        dark_R = int((dark_mask[1] & self._upper_half_mask).sum())

        # Glitch filter
        upper_total = int(self._upper_half_mask.sum())
        if dark_L > upper_total * 0.7 and dark_R > upper_total * 0.7:
            dark_L = 0
            dark_R = 0

        self._last_blade_signals = {
            "dark_L": dark_L,
            "dark_R": dark_R,
        }

        # 4) Trigger
        thresh = int(self.BLADE_COUNT_THRESH)
        urgent = int(self.BLADE_COUNT_URGENT)
        asym = abs(dark_L - dark_R)
        max_dark = max(dark_L, dark_R)
        signal_present = (max_dark >= thresh) and (asym >= int(self.BLADE_ASYM_MIN))
        is_urgent = (max_dark >= urgent) and (asym >= int(self.BLADE_ASYM_MIN))
        if signal_present:
            self._blade_hold_count += 1
        else:
            self._blade_hold_count = 0
        required_hold = 1 if is_urgent else int(self.BLADE_HOLD_FRAMES)
        if (
            self._blade_hold_count < required_hold
            or self._debug_decisions < int(self.BLADE_STARTUP_DELAY)
        ):
            return None

        # 5) Choisir côté
        if dark_L > dark_R:
            self._blade_dir = +1
        else:
            self._blade_dir = -1
        self._blade_hold_count = 0
        self._blade_decisions_left = int(self.BLADE_TURN_DECISIONS)
        if self.DEBUG:
            print(
                f"[BLADE-AVOID d={self._debug_decisions}] "
                f"dark=L{dark_L}/R{dark_R} dir={self._blade_dir:+d}",
                flush=True,
            )
        if self._blade_dir > 0:
            return np.array([self.BLADE_TURN_DRIVE_FAST, self.BLADE_TURN_DRIVE_SLOW])
        return np.array([self.BLADE_TURN_DRIVE_SLOW, self.BLADE_TURN_DRIVE_FAST])

    def compute_vision_debug_overlay(self, sim: MiniprojectSimulation):
        """Overlay simple pour run_with_controller.py --debug-vision."""
        if not self._enable_grass or self._upper_half_mask is None:
            return None
        try:
            ommatidia = sim.get_ommatidia_readouts(sim.fly.name)
            retina = sim.world.fly_lookup[sim.fly.name].retina
        except Exception:
            return None
        sum_intensity = ommatidia.sum(axis=-1)
        try:
            frames = []
            for eye_idx in range(2):
                base = retina.hex_pxls_to_human_readable(sum_intensity[eye_idx])
                base_norm = (np.clip(base / max(base.max(), 1e-6), 0, 1) * 255).astype(np.uint8)
                rgb = np.stack([base_norm, base_norm, base_norm], axis=-1)
                dark = (sum_intensity[eye_idx] < self.BLADE_DARK_THRESH) & self._upper_half_mask
                if dark.any():
                    mark_img = retina.hex_pxls_to_human_readable(dark.astype(np.float32))
                    mask = mark_img > 0.1
                    rgb[mask, 0] = 255
                    rgb[mask, 1] = 0
                    rgb[mask, 2] = 0
                frames.append(rgb)
            return np.concatenate(frames, axis=1)
        except Exception:
            return None
