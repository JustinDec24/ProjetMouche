import numpy as np
from miniproject.simulation import MiniprojectSimulation


class Controller:
    """Goal-first controller with map-based grass bypassing.

    The important difference from the previous vision-only controllers is that
    the banana always remains the dominant objective. Grass is treated only as
    a local obstacle around the current banana path. The controller extracts the
    grass base positions from the MuJoCo model and uses them to choose short
    bypass waypoints, so the fly goes through gaps instead of avoiding the whole
    grass field.
    """

    DECISION_INTERVAL_S = 0.025

    # Banana stopping
    PALP_WEIGHT = 9.0
    ANTENNA_WEIGHT = 1.0
    STOP_ODOR_THRESHOLD = 5e-3
    STOP_DIST = 2.0

    # Locomotion
    BASE_DRIVE = 2.35
    BYPASS_DRIVE = 2.05
    CLOSE_DRIVE = 2.35
    MAX_DRIVE = 2.80
    MAX_DRIVE_TERRAIN = 2.05
    MIN_DRIVE = 0.75
    MIN_DRIVE_TERRAIN = 0.62
    MIN_SIDE = 0.40
    MIN_SIDE_TERRAIN = 0.32
    TURN_MOD = 0.85

    # Goal steering. Convention: bearing > 0 means target left, so bias < 0.
    TARGET_GAIN = 3.6
    TARGET_GAIN_CLOSE = 8.5
    TARGET_CLOSE_DIST = 18.0
    TARGET_SCALE = 0.25

    # Terrain / grip
    CONTACT_THRESHOLD = 0.15
    TERRAIN_GRIP_FORCE = 6.0
    WIND_GRIP_FORCE = 30.0
    STARTUP_MAX_GRIP_STEPS = 3500

    DOWNHILL_BRAKE = 1.45
    STEEP_BRAKE = 1.05
    TURN_STEEP_GAIN = 1.7
    SLOPE_STEER_GAIN = 4.5
    SLOPE_STEER_MAX = 2.2

    TILT_LEAN_ENABLE = True
    TILT_LEAN_ROLL_ON = 0.10
    TILT_LEAN_ROLL_FULL = 0.40
    TILT_LEAN_GAIN = 0.28
    TILT_LEAN_SIGN = 1.0

    # Geometric grass planning
    GRASS_POS_MIN_RADIUS = 4.0
    GRASS_LOOKAHEAD = 12.0
    GRASS_CORRIDOR = 2.8
    GRASS_SAFETY = 2.25
    GRASS_REPULSE_RANGE = 6.5
    BYPASS_OFFSET = 4.4
    BYPASS_LATCH_DECISIONS = 18
    BYPASS_RELEASE_PROJ = -1.2
    BYPASS_SIDE_SWITCH_MARGIN = 1.0

    # Repulsion is deliberately mild. It should only refine the bypass, not
    # replace banana attraction.
    REPULSION_GAIN = 1.65
    REPULSION_GAIN_GOAL_LOCK = 0.75
    REPULSION_FORWARD_RANGE = 7.0
    REPULSION_LATERAL_RANGE = 4.6
    REPULSION_TARGET_BLEND = 1.35
    VIS_TURN_MAX = 5.0
    GOAL_LOCK_DIST = 15.0
    AVOID_DISABLE_CLOSE_DIST = 5.5

    # Emergency only when the fly is truly near grass/contact. This avoids the
    # previous behavior where it backed up from far-away grass.
    FRONT_DANGER_FORWARD = 2.5
    FRONT_DANGER_LATERAL = 1.35
    SIDE_DANGER_DIST = 1.55
    FRONT_ESCAPE_BACKUP_DECISIONS = 8
    FRONT_ESCAPE_PIVOT_DECISIONS = 12
    FRONT_ESCAPE_COOLDOWN_DECISIONS = 16
    FRONT_ESCAPE_BACKUP_DRIVE = -0.55
    FRONT_ESCAPE_PIVOT_MAX = 1.90
    FRONT_ESCAPE_PIVOT_MIN = 0.35
    FRONT_ESCAPE_FRONT_TUCK_DECISIONS = 6
    FRONT_ESCAPE_FRONT_LIFT_COXA_OFFSET = -0.45
    POST_ESCAPE_GOAL_LOCK_DECISIONS = 22

    SIDE_CONTACT_FORCE_THRESH = 24.0
    SIDE_CONTACT_COOLDOWN_DECISIONS = 8

    HEAD_COLLISION_FORCE_THRESH = 85.0
    HEAD_COLLISION_COOLDOWN_DECISIONS = 10
    HEAD_COLLISION_BACKUP_DECISIONS = 10
    HEAD_COLLISION_PIVOT_DECISIONS = 5
    HEAD_COLLISION_ARC_DECISIONS = 8
    HEAD_COLLISION_BACKUP_DRIVE = -0.45
    HEAD_COLLISION_PIVOT_MAX = 1.85
    HEAD_COLLISION_PIVOT_MIN = 0.40
    HEAD_COLLISION_ARC_OUTER = 1.85
    HEAD_COLLISION_ARC_INNER = 0.60
    HEAD_COLLISION_FRONT_TUCK_DECISIONS = 6
    HEAD_COLLISION_FRONT_LIFT_COXA_OFFSET = -0.45

    # Initial and recovery alignment
    ALIGN_BEARING_OK = 0.14
    ALIGN_MAX_DECISIONS = 75
    ALIGN_MAX_DRIVE = 1.75
    ALIGN_MIN_SIDE = 0.42

    GO_PIVOT_ON = 0.65
    GO_PIVOT_OFF = 0.20
    GOAL_RECOVERY_BEARING_ON = 1.05
    GOAL_RECOVERY_BEARING_OFF = 0.28
    GOAL_RECOVERY_DECISIONS = 20
    GOAL_RECOVERY_NO_PROGRESS_DECISIONS = 24
    GOAL_RECOVERY_DIST_LOSS = 1.1

    DEBUG = False
    DEBUG_EVERY_DECISIONS = 4
    DEBUG_MAX_DECISIONS = 5000

    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        self._decision_every = max(1, int(self.DECISION_INTERVAL_S / sim.timestep))
        self._step_count = 0
        self._debug_decisions = 0
        self._drives = np.array([1.0, 1.0], dtype=float)
        self._stopped = False

        self._enable_terrain = bool(getattr(sim, "enable_terrain", False))
        self._enable_grass = bool(getattr(sim, "enable_grass", False))
        self._enable_wind = bool(getattr(sim, "enable_wind", False))

        fly_segs = sim.fly.get_bodysegs_order()
        self._thorax_idx = next(i for i, s in enumerate(fly_segs) if s.name == "c_thorax")
        body_ids = sim._internal_bodyids_by_fly[sim.fly.name]
        self._thorax_body_id = body_ids[self._thorax_idx]
        self._contact_body_ids = sim._internal_contact_body_segment_ids_by_fly[sim.fly.name]
        try:
            head_idx = next(i for i, s in enumerate(fly_segs) if s.name == "c_head")
            self._head_body_id = body_ids[head_idx]
        except StopIteration:
            self._head_body_id = self._thorax_body_id

        try:
            self._banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
        except Exception:
            self._banana_xy = None

        self._grass_xy = self._extract_grass_xy(sim)

        self._last_xy = None
        self._last_target_bearing = 0.0
        self._last_dist_to_banana = None
        self._best_dist_to_banana = np.inf
        self._no_progress_decisions = 0
        self._goal_recovery_left = 0
        self._post_escape_goal_lock_left = 0

        self._align_done = False
        self._align_left = 0
        self._go_pivot_active = False
        self._stuck_decisions = 0
        self._escape_decisions_left = 0
        self._escape_dir = 1.0

        self._bypass_left = 0
        self._bypass_side = 0.0
        self._bypass_grass = None

        self._front_escape_phase = 0
        self._front_escape_left = 0
        self._front_escape_dir = 0.0
        self._front_escape_cooldown = 0

        self._side_force_left_peak = 0.0
        self._side_force_right_peak = 0.0
        self._side_contact_cooldown = 0

        self._collision_phase = 0
        self._collision_left = 0
        self._collision_arc_dir = 1.0
        self._collision_cooldown = 0
        self._head_force_peak = 0.0

        try:
            import mujoco as _mj
            for j in range(sim.mj_model.njnt):
                if sim.mj_model.jnt_type[j] == _mj.mjtJoint.mjJNT_FREE:
                    addr = sim.mj_model.jnt_qposadr[j]
                    sim.mj_data.qpos[addr + 2] += 0.4
                    _mj.mj_forward(sim.mj_model, sim.mj_data)
                    break
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main simulation step
    # ------------------------------------------------------------------
    def step(self, sim: MiniprojectSimulation):
        self._track_contact_peaks(sim)

        is_decision_step = self._step_count % self._decision_every == 0
        if is_decision_step:
            self._drives = self._compute_drives(sim)
            self._debug_decisions = self._step_count // self._decision_every
        self._step_count += 1

        joint_angles, adhesion = self.turning_controller.step(self._drives)

        front_tuck = (
            self._front_escape_phase == 1
            and self._front_escape_left
            > self.FRONT_ESCAPE_BACKUP_DECISIONS - self.FRONT_ESCAPE_FRONT_TUCK_DECISIONS
        )
        coll_tuck = (
            self._collision_phase == 1
            and self._collision_left
            > self.HEAD_COLLISION_BACKUP_DECISIONS - self.HEAD_COLLISION_FRONT_TUCK_DECISIONS
        )
        if front_tuck or coll_tuck:
            offset = self.FRONT_ESCAPE_FRONT_LIFT_COXA_OFFSET if front_tuck else self.HEAD_COLLISION_FRONT_LIFT_COXA_OFFSET
            for li in (0, 3):
                joint_angles[li * 7 + 1] += float(offset)

        if self.TILT_LEAN_ENABLE:
            try:
                xmat = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
                roll = float(xmat[2, 1])
            except Exception:
                roll = 0.0
            excess = abs(roll) - self.TILT_LEAN_ROLL_ON
            if excess > 0.0 and self._escape_decisions_left <= 0:
                ramp = min(1.0, excess / max(1e-3, self.TILT_LEAN_ROLL_FULL - self.TILT_LEAN_ROLL_ON))
                offset = self.TILT_LEAN_SIGN * self.TILT_LEAN_GAIN * ramp
                legs = (3, 4, 5) if roll > 0 else (0, 1, 2)
                for li in legs:
                    joint_angles[li * 7 + 3] += offset
                    joint_angles[li * 7 + 5] += 0.5 * offset

        if self._enable_terrain and is_decision_step:
            try:
                upright = float(sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)[2, 2])
            except Exception:
                upright = 1.0

            in_backup = self._front_escape_phase == 1 or self._collision_phase == 1
            in_startup = self._step_count < self.STARTUP_MAX_GRIP_STEPS
            grip = self.WIND_GRIP_FORCE if (self._enable_wind or in_backup or in_startup) else self.TERRAIN_GRIP_FORCE

            # On L2, if already fully upside down, avoid making the physics worse.
            if upright < -0.15 and not self._enable_wind:
                return joint_angles, np.zeros_like(adhesion)

            try:
                forces = sim.mj_data.cfrc_ext[self._contact_body_ids, 3:]
                mag = np.linalg.norm(forces, axis=1)
                stance = mag > self.CONTACT_THRESHOLD
                adhesion = np.zeros_like(adhesion)
                n = min(len(adhesion), len(stance))

                if self._enable_wind or in_backup or in_startup:
                    adhesion[:n] = grip
                else:
                    adhesion[:n] = stance[:n].astype(float) * grip

                if in_backup:
                    if n > 0:
                        adhesion[0] = 0.0
                    if n > 3:
                        adhesion[3] = 0.0
            except Exception:
                adhesion = np.full_like(adhesion, grip)

            adhesion = np.clip(adhesion, 0.0, grip)

        return joint_angles, adhesion

    # ------------------------------------------------------------------
    # Sensors and geometry
    # ------------------------------------------------------------------
    def _extract_grass_xy(self, sim) -> np.ndarray:
        if not self._enable_grass:
            return np.empty((0, 2), dtype=float)
        out = []
        try:
            rgba = np.asarray(sim.mj_model.geom_rgba)
            body_ids = np.asarray(sim.mj_model.geom_bodyid)
            for gid in range(sim.mj_model.ngeom):
                r, g, b, a = rgba[gid]
                if g > 0.75 and r < 0.25 and b < 0.25 and a > 0.5:
                    bid = int(body_ids[gid])
                    xy = np.asarray(sim.mj_data.xpos[bid, :2], dtype=float)
                    if np.linalg.norm(xy) > self.GRASS_POS_MIN_RADIUS:
                        out.append(xy.copy())
        except Exception:
            return np.empty((0, 2), dtype=float)

        if not out:
            return np.empty((0, 2), dtype=float)

        unique = []
        for p in out:
            if all(np.linalg.norm(p - q) > 0.5 for q in unique):
                unique.append(p)
        return np.asarray(unique, dtype=float)

    def _track_contact_peaks(self, sim) -> None:
        try:
            hf = float(np.linalg.norm(sim.mj_data.cfrc_ext[self._head_body_id, 3:]))
            self._head_force_peak = max(self._head_force_peak, hf)
        except Exception:
            pass

        if self._enable_grass:
            try:
                forces = sim.mj_data.cfrc_ext[self._contact_body_ids, 3:]
                mag = np.linalg.norm(forces, axis=1)
                if len(mag) >= 6:
                    left = float(max(mag[0], mag[1], mag[2]))
                    right = float(max(mag[3], mag[4], mag[5]))
                else:
                    half = max(1, len(mag) // 2)
                    left = float(np.max(mag[:half]))
                    right = float(np.max(mag[half:]))
                self._side_force_left_peak = max(self._side_force_left_peak, left)
                self._side_force_right_peak = max(self._side_force_right_peak, right)
            except Exception:
                pass

    def _body_frame_xy(self, sim):
        xmat = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
        heading = xmat[:2, 0].copy()
        lateral = xmat[:2, 1].copy()
        hn = np.linalg.norm(heading)
        ln = np.linalg.norm(lateral)
        if hn > 1e-12:
            heading /= hn
        if ln > 1e-12:
            lateral /= ln
        return heading, lateral

    def _slope_signals(self, sim):
        world = getattr(sim, "world", None)
        get_normal = getattr(world, "get_normal", None)
        if not callable(get_normal):
            return 0.0, 0.0, 0.0
        try:
            xy = sim.get_body_positions(sim.fly.name)[self._thorax_idx, :2]
        except Exception:
            xy = sim.mj_data.xpos[self._thorax_body_id, :2]
        n = np.asarray(get_normal(float(xy[0]), float(xy[1])), dtype=float)
        if n.shape != (3,) or not np.isfinite(n).all() or abs(n[2]) < 1e-6:
            return 0.0, 0.0, 0.0
        grad = np.array([-n[0] / n[2], -n[1] / n[2]], dtype=float)
        heading, lateral = self._body_frame_xy(sim)
        return float(np.dot(heading, grad)), float(np.dot(lateral, grad)), float(np.linalg.norm(grad))

    def _bearing_to_point(self, sim, xy, target_xy):
        v = np.asarray(target_xy, dtype=float) - np.asarray(xy, dtype=float)
        d = float(np.linalg.norm(v))
        if d < 1e-9:
            return 0.0, 0.0
        v /= d
        heading, lateral = self._body_frame_xy(sim)
        lat = float(np.dot(lateral, v))
        fwd = float(np.dot(heading, v))
        return float(np.arctan2(lat, fwd)), d

    def _target_bias_bearing(self, sim, xy, target_xy=None):
        if target_xy is None:
            target_xy = self._banana_xy
        if target_xy is None:
            return 0.0, 0.0
        bearing, dist = self._bearing_to_point(sim, xy, target_xy)
        if target_xy is self._banana_xy or np.allclose(target_xy, self._banana_xy):
            if abs(bearing) > 1e-6:
                self._last_target_bearing = bearing
        gain = self.TARGET_GAIN_CLOSE if dist < self.TARGET_CLOSE_DIST else self.TARGET_GAIN
        bias = -self.TARGET_SCALE * gain * bearing
        return float(bias), float(bearing)

    # ------------------------------------------------------------------
    # Drive helpers
    # ------------------------------------------------------------------
    def _turn_drives(self, direction, mx, mn):
        # direction +1 -> turn right, direction -1 -> turn left
        return np.array([mx, mn], dtype=float) if direction >= 0 else np.array([mn, mx], dtype=float)

    def _target_pivot(self, bearing):
        direction = -1.0 if bearing > 0.0 else 1.0
        return self._turn_drives(direction, self.ALIGN_MAX_DRIVE, self.ALIGN_MIN_SIDE)

    def _direction_toward_banana(self, bearing):
        if abs(float(bearing)) > 0.12:
            return -1.0 if bearing > 0.0 else 1.0
        if abs(self._last_target_bearing) > 0.12:
            return -1.0 if self._last_target_bearing > 0.0 else 1.0
        return 1.0

    def _apply_progress_supervisor(self, dist, bearing):
        if dist is None:
            return
        if dist < self._best_dist_to_banana - 0.10:
            self._best_dist_to_banana = dist
            self._no_progress_decisions = 0
        else:
            self._no_progress_decisions += 1
        lost = dist > self._best_dist_to_banana + self.GOAL_RECOVERY_DIST_LOSS
        stalled = self._no_progress_decisions >= self.GOAL_RECOVERY_NO_PROGRESS_DECISIONS
        behind = abs(float(bearing)) > self.GOAL_RECOVERY_BEARING_ON and dist > 9.0
        if (lost or stalled or behind) and self._goal_recovery_left <= 0:
            self._goal_recovery_left = self.GOAL_RECOVERY_DECISIONS

    # ------------------------------------------------------------------
    # Grass map planner
    # ------------------------------------------------------------------
    def _choose_bypass_side(self, xy, blocker, u, n):
        # Candidate side +1 means waypoint left of the direct goal line;
        # candidate side -1 means waypoint right of it.
        best_side = 1.0
        best_score = float("inf")
        for side in (1.0, -1.0):
            wp = blocker + side * self.BYPASS_OFFSET * n
            if self._banana_xy is not None:
                score = 0.25 * float(np.linalg.norm(wp - self._banana_xy))
            else:
                score = 0.0
            if self._bypass_left > 0 and side == self._bypass_side:
                score -= 0.8
            if self._grass_xy.size:
                d = np.linalg.norm(self._grass_xy - wp[None, :], axis=1)
                close = d[d < 5.5]
                if close.size:
                    score += float(np.sum((5.5 - close) ** 2))
            if score < best_score:
                best_score = score
                best_side = side
        return best_side

    def _plan_waypoint(self, xy):
        if self._banana_xy is None or not self._enable_grass or self._grass_xy.size == 0:
            return self._banana_xy, "GO"

        to_goal = self._banana_xy - xy
        dist_goal = float(np.linalg.norm(to_goal))
        if dist_goal < 1e-9:
            return self._banana_xy, "GO"

        u = to_goal / dist_goal
        n = np.array([-u[1], u[0]], dtype=float)

        # Keep the previous bypass until we have passed its obstacle.
        if self._bypass_left > 0 and self._bypass_grass is not None:
            rel_old = self._bypass_grass - xy
            proj_old = float(np.dot(rel_old, u))
            if proj_old > self.BYPASS_RELEASE_PROJ:
                self._bypass_left -= 1
                return self._bypass_grass + self._bypass_side * self.BYPASS_OFFSET * n, "BYPASS"
            self._bypass_left = 0
            self._bypass_grass = None

        rel = self._grass_xy - xy[None, :]
        proj = rel @ u
        perp = rel @ n
        lookahead = min(self.GRASS_LOOKAHEAD, max(4.0, dist_goal - self.STOP_DIST))

        mask = (
            (proj > 0.5)
            & (proj < lookahead)
            & (np.abs(perp) < self.GRASS_CORRIDOR)
        )
        if not np.any(mask):
            return self._banana_xy, "GO"

        idxs = np.where(mask)[0]
        # Closest blocker along the current banana line, slightly preferring the
        # most central one.
        scores = proj[idxs] + 0.35 * np.abs(perp[idxs])
        idx = int(idxs[int(np.argmin(scores))])
        blocker = self._grass_xy[idx]
        side = self._choose_bypass_side(xy, blocker, u, n)

        self._bypass_left = self.BYPASS_LATCH_DECISIONS
        self._bypass_side = side
        self._bypass_grass = blocker.copy()
        waypoint = blocker + side * self.BYPASS_OFFSET * n
        return waypoint, "BYPASS"

    def _grass_repulsion_bias(self, sim, xy, target_bias, goal_locked):
        if not self._enable_grass or self._grass_xy.size == 0:
            return 0.0

        heading, lateral = self._body_frame_xy(sim)
        rel = self._grass_xy - xy[None, :]
        fwd = rel @ heading
        lat = rel @ lateral
        dist = np.linalg.norm(rel, axis=1)

        gain = self.REPULSION_GAIN_GOAL_LOCK if goal_locked else self.REPULSION_GAIN
        rep = 0.0

        for fd, lt, d in zip(fwd, lat, dist):
            if fd < -0.8 or fd > self.REPULSION_FORWARD_RANGE:
                continue
            if abs(lt) > self.REPULSION_LATERAL_RANGE and d > self.GRASS_REPULSE_RANGE:
                continue

            forward_w = max(0.0, 1.0 - max(0.0, fd) / self.REPULSION_FORWARD_RANGE)
            lateral_w = max(0.0, 1.0 - abs(lt) / self.REPULSION_LATERAL_RANGE)
            dist_w = np.exp(-max(0.0, d - self.GRASS_SAFETY) / 2.0)
            strength = forward_w * lateral_w * dist_w
            if strength <= 1e-4:
                continue

            if abs(lt) < 0.25:
                side_bias = np.sign(target_bias) if abs(target_bias) > 1e-6 else np.sign(self._bypass_side)
                if side_bias == 0.0:
                    side_bias = 1.0
            else:
                side_bias = 1.0 if lt > 0.0 else -1.0
            rep += side_bias * strength

        return float(gain * rep)

    def _front_or_side_danger(self, sim, xy, bearing):
        if not self._enable_grass or self._grass_xy.size == 0:
            return False, 0.0
        heading, lateral = self._body_frame_xy(sim)
        rel = self._grass_xy - xy[None, :]
        fwd = rel @ heading
        lat = rel @ lateral
        dist = np.linalg.norm(rel, axis=1)

        front_mask = (
            (fwd > 0.0)
            & (fwd < self.FRONT_DANGER_FORWARD)
            & (np.abs(lat) < self.FRONT_DANGER_LATERAL)
        )
        side_mask = dist < self.SIDE_DANGER_DIST

        if not np.any(front_mask | side_mask):
            return False, 0.0

        threat_idxs = np.where(front_mask | side_mask)[0]
        # Turn away from the average side of the nearest threats. If central,
        # choose the side that moves toward the banana.
        lt = float(np.mean(lat[threat_idxs]))
        if abs(lt) > 0.25:
            direction = 1.0 if lt > 0.0 else -1.0
        else:
            direction = self._direction_toward_banana(bearing)
        return True, direction

    # ------------------------------------------------------------------
    # Escape sequences
    # ------------------------------------------------------------------
    def _start_front_escape(self, direction):
        self._front_escape_phase = 1
        self._front_escape_left = self.FRONT_ESCAPE_BACKUP_DECISIONS
        self._front_escape_dir = float(direction if abs(direction) > 0.1 else 1.0)
        self._collision_phase = 0
        self._bypass_left = 0

    def _front_escape_drives(self):
        if self._front_escape_phase == 1:
            self._front_escape_left -= 1
            if self._front_escape_left <= 0:
                self._front_escape_phase = 2
                self._front_escape_left = self.FRONT_ESCAPE_PIVOT_DECISIONS
            return np.array([self.FRONT_ESCAPE_BACKUP_DRIVE, self.FRONT_ESCAPE_BACKUP_DRIVE], dtype=float)

        if self._front_escape_phase == 2:
            self._front_escape_left -= 1
            drives = self._turn_drives(self._front_escape_dir, self.FRONT_ESCAPE_PIVOT_MAX, self.FRONT_ESCAPE_PIVOT_MIN)
            if self._front_escape_left <= 0:
                self._front_escape_phase = 0
                self._front_escape_cooldown = self.FRONT_ESCAPE_COOLDOWN_DECISIONS
                self._post_escape_goal_lock_left = self.POST_ESCAPE_GOAL_LOCK_DECISIONS
            return drives

        return np.array([1.0, 1.0], dtype=float)

    # ------------------------------------------------------------------
    # Decision policy
    # ------------------------------------------------------------------
    def _compute_drives(self, sim):
        if self._stopped:
            return np.array([0.0, 0.0], dtype=float)

        if self._banana_xy is None:
            try:
                self._banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
            except Exception:
                pass

        if self._enable_grass and self._grass_xy.size == 0:
            self._grass_xy = self._extract_grass_xy(sim)

        try:
            xy = sim.get_body_positions(sim.fly.name)[self._thorax_idx, :2]
        except Exception:
            xy = sim.mj_data.xpos[self._thorax_body_id, :2]
        xy = np.asarray(xy, dtype=float)

        dist = None
        if self._banana_xy is not None:
            dist = float(np.linalg.norm(xy - self._banana_xy))
            self._last_dist_to_banana = dist
            if dist <= self.STOP_DIST:
                self._stopped = True
                return np.array([0.0, 0.0], dtype=float)

        if self._last_xy is not None:
            moved = float(np.linalg.norm(xy - self._last_xy))
            self._stuck_decisions = self._stuck_decisions + 1 if moved < 5e-3 else 0
        self._last_xy = xy

        if self._escape_decisions_left > 0:
            self._escape_decisions_left -= 1
            return self._turn_drives(self._escape_dir, self.SIDE_ESCAPE_OUTER, self.SIDE_ESCAPE_INNER)

        if self._enable_terrain and self._stuck_decisions >= 28:
            self._stuck_decisions = 0
            self._escape_decisions_left = 9
            self._escape_dir *= -1.0
            return self._turn_drives(self._escape_dir, self.SIDE_ESCAPE_OUTER, self.SIDE_ESCAPE_INNER)

        odor = sim.get_olfaction(sim.fly.name)
        lp, rp, la, ra = odor[:, 0]
        mean_odor = 0.5 * (
            self.PALP_WEIGHT * float(lp)
            + self.ANTENNA_WEIGHT * float(la)
            + self.PALP_WEIGHT * float(rp)
            + self.ANTENNA_WEIGHT * float(ra)
        )
        if mean_odor > self.STOP_ODOR_THRESHOLD:
            self._stopped = True
            return np.array([0.0, 0.0], dtype=float)

        banana_bias, banana_bearing = self._target_bias_bearing(sim, xy, self._banana_xy)
        self._last_target_bearing = banana_bearing
        self._apply_progress_supervisor(dist, banana_bearing)

        close_to_banana = dist is not None and dist < self.AVOID_DISABLE_CLOSE_DIST
        goal_lock = dist is not None and dist < self.GOAL_LOCK_DIST

        if self._front_escape_cooldown > 0:
            self._front_escape_cooldown -= 1
        if self._front_escape_phase > 0:
            return self._front_escape_drives()

        danger, danger_dir = self._front_or_side_danger(sim, xy, banana_bearing)
        if self._enable_grass and not close_to_banana and self._front_escape_cooldown <= 0 and danger:
            self._start_front_escape(danger_dir)
            return self._front_escape_drives()

        if self._side_contact_cooldown > 0:
            self._side_contact_cooldown -= 1
        lf = float(self._side_force_left_peak)
        rf = float(self._side_force_right_peak)
        self._side_force_left_peak = 0.0
        self._side_force_right_peak = 0.0
        if self._enable_grass and self._side_contact_cooldown <= 0 and max(lf, rf) > self.SIDE_CONTACT_FORCE_THRESH:
            direction = 1.0 if lf >= rf else -1.0
            self._side_contact_cooldown = self.SIDE_CONTACT_COOLDOWN_DECISIONS
            self._start_front_escape(direction)
            return self._front_escape_drives()

        if self._collision_cooldown > 0:
            self._collision_cooldown -= 1
        hf = float(self._head_force_peak)
        self._head_force_peak = 0.0
        if self._collision_phase == 0 and self._collision_cooldown == 0 and hf > self.HEAD_COLLISION_FORCE_THRESH:
            self._collision_arc_dir = -1.0 if banana_bearing > 0.0 else 1.0
            self._collision_phase = 1
            self._collision_left = self.HEAD_COLLISION_BACKUP_DECISIONS
        if self._collision_phase > 0:
            self._collision_left -= 1
            phase = self._collision_phase
            if phase == 1:
                drives = np.array([self.HEAD_COLLISION_BACKUP_DRIVE, self.HEAD_COLLISION_BACKUP_DRIVE], dtype=float)
            elif phase == 2:
                drives = self._turn_drives(self._collision_arc_dir, self.HEAD_COLLISION_PIVOT_MAX, self.HEAD_COLLISION_PIVOT_MIN)
            else:
                drives = self._turn_drives(self._collision_arc_dir, self.HEAD_COLLISION_ARC_OUTER, self.HEAD_COLLISION_ARC_INNER)
            if self._collision_left <= 0:
                if phase == 1:
                    self._collision_phase = 2
                    self._collision_left = self.HEAD_COLLISION_PIVOT_DECISIONS
                elif phase == 2:
                    self._collision_phase = 3
                    self._collision_left = self.HEAD_COLLISION_ARC_DECISIONS
                else:
                    self._collision_phase = 0
                    self._collision_cooldown = self.HEAD_COLLISION_COOLDOWN_DECISIONS
            return drives

        if not self._align_done:
            if self._align_left == 0:
                self._align_left = self.ALIGN_MAX_DECISIONS
            if abs(banana_bearing) <= self.ALIGN_BEARING_OK or self._align_left <= 0:
                self._align_done = True
                self._align_left = 0
            else:
                self._align_left -= 1
                return self._target_pivot(banana_bearing)

        if self._post_escape_goal_lock_left > 0:
            self._post_escape_goal_lock_left -= 1
            if abs(banana_bearing) > 0.20:
                return self._target_pivot(banana_bearing)

        if self._goal_recovery_left > 0:
            self._goal_recovery_left -= 1
            if abs(banana_bearing) < self.GOAL_RECOVERY_BEARING_OFF:
                self._goal_recovery_left = 0
            return self._target_pivot(banana_bearing)

        waypoint, mode = self._plan_waypoint(xy)
        if waypoint is None:
            waypoint = self._banana_xy
        wp_bias, wp_bearing = self._target_bias_bearing(sim, xy, waypoint)

        if close_to_banana:
            bias = banana_bias
            base = self.CLOSE_DRIVE
            mode = "GO"
        else:
            # Banana attraction remains dominant. The waypoint changes the
            # direction only while an obstacle blocks the straight banana line.
            rep = self._grass_repulsion_bias(sim, xy, wp_bias, goal_lock)
            if mode == "BYPASS":
                bias = self.REPULSION_TARGET_BLEND * wp_bias + rep + 0.35 * banana_bias
                base = self.BYPASS_DRIVE
            else:
                bias = self.REPULSION_TARGET_BLEND * banana_bias + rep
                base = self.BASE_DRIVE
            bias = float(np.clip(bias, -self.VIS_TURN_MAX, self.VIS_TURN_MAX))

        if mode == "GO":
            ab = abs(float(banana_bearing))
            if self._go_pivot_active:
                if ab < self.GO_PIVOT_OFF:
                    self._go_pivot_active = False
            else:
                if ab > self.GO_PIVOT_ON:
                    self._go_pivot_active = True
            if self._go_pivot_active:
                return self._target_pivot(banana_bearing)
        else:
            self._go_pivot_active = False

        turn_mod = self.TURN_MOD
        if self._enable_terrain:
            sf, sl, sm = self._slope_signals(sim)
            downhill = max(0.0, -sf)
            steep_weight = 0.25 + 0.75 * downhill
            base *= 1.0 / (1.0 + self.DOWNHILL_BRAKE * downhill + self.STEEP_BRAKE * steep_weight * max(0.0, sm))
            slope_bias = -self.SLOPE_STEER_GAIN * sl * downhill
            if mode == "BYPASS":
                slope_bias *= 0.25
            bias += float(np.clip(slope_bias, -self.SLOPE_STEER_MAX, self.SLOPE_STEER_MAX))
            turn_mod /= 1.0 + self.TURN_STEEP_GAIN * max(0.0, sm)

        max_drive = self.MAX_DRIVE_TERRAIN if self._enable_terrain else self.MAX_DRIVE
        min_drive = self.MIN_DRIVE_TERRAIN if self._enable_terrain else self.MIN_DRIVE
        min_side = self.MIN_SIDE_TERRAIN if self._enable_terrain else self.MIN_SIDE
        base = float(np.clip(base, min_drive, max_drive))

        bias_norm = float(np.tanh(bias))
        drives = np.full(2, base, dtype=float)
        side = int(bias_norm > 0.0)
        drives[side] -= abs(bias_norm) * turn_mod * base
        drives[side] = max(min_side, drives[side])
        drives = np.clip(drives, 0.0, max_drive)

        try:
            upright = float(sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)[2, 2])
        except Exception:
            upright = 1.0
        if upright < 0.70:
            blend = min(0.85, (0.70 - upright) / 0.55)
            mean_drive = float(np.mean(drives))
            drives = drives * (1.0 - blend) + mean_drive * blend
            drives = np.clip(drives, 0.0, max_drive)

        return drives

    # ------------------------------------------------------------------
    # Optional vision overlay hook used by run_with_controller.py
    # ------------------------------------------------------------------
    def compute_vision_debug_overlay(self, sim: MiniprojectSimulation):
        try:
            frames = sim.get_raw_vision(sim.fly.name)
        except Exception:
            return None
        if frames is None or len(frames) == 0:
            return None
        imgs = []
        for img in frames[:2]:
            a = np.asarray(img)
            if a.ndim == 2:
                a = np.stack([a, a, a], axis=-1)
            if a.shape[-1] > 3:
                a = a[..., :3]
            if a.dtype != np.uint8:
                af = a.astype(np.float32)
                if af.size and af.max() <= 1.0:
                    af *= 255.0
                a = np.clip(af, 0, 255).astype(np.uint8)
            imgs.append(a)
        if len(imgs) == 1:
            imgs.append(imgs[0])
        return np.concatenate(imgs, axis=1).astype(np.uint8)