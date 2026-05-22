"""
BIOENG-456 mini-project controller v5: odor-only banana navigation.

Main idea:
    1. Never read sim.world.banana_xy inside the controller.
    2. Estimate the direction of the banana from the spatial gradient of odor
       across the fly's olfactory sensors.
    3. Estimate a rough source range from the known odor concentration model.
    4. Use this odor-derived moving goal for path tracking, local grass
       repulsion, recovery, and final approach.

The evaluation scripts may still use the real banana position to score success,
but the controller itself does not use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import heapq
import math
import numpy as np

from miniproject.simulation import MiniprojectSimulation


@dataclass
class CFG:
    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    DEBUG: bool = True
    DEBUG_EVERY: int = 1000

    # ------------------------------------------------------------------
    # Locomotion speed. Increase CPG_FREQ / DRIVE_* if stable but slow.
    # Reduce CPG_FREQ to 13.5 if the fly flips on hills.
    # ------------------------------------------------------------------
    CPG_FREQ: float = 15.5
    CPG_CONVERGENCE: float = 30.0

    DRIVE_SPRINT: float = 1.22
    DRIVE_FAST: float = 1.12
    DRIVE_PATH: float = 1.08
    DRIVE_AVOID: float = 0.98
    DRIVE_HILL: float = 0.78
    DRIVE_FINAL: float = 0.62
    DRIVE_BACK: float = -0.56
    DRIVE_MAX: float = 1.30

    # During normal steering we keep both sides stepping forward. This is what
    # makes the fly travel quickly instead of pivoting in place.
    MIN_FORWARD_FAST: float = 0.34
    MIN_FORWARD_PATH: float = 0.26
    MIN_FORWARD_AVOID: float = 0.22

    TURN_GAIN_FAST: float = 1.15
    TURN_GAIN_PATH: float = 1.55
    TURN_GAIN_AVOID: float = 1.85
    TURN_GAIN_FINAL: float = 1.45
    TURN_SHARPNESS: float = 1.55

    # Pivot only when the path direction is very far from the fly heading.
    PIVOT_ERR: float = 1.25
    PIVOT_RELEASE_ERR: float = 0.35
    PIVOT_OUTER: float = 1.00
    PIVOT_INNER_BACK: float = -0.30

    # ------------------------------------------------------------------
    # Global grass-aware path planner.
    # Grass centers are treated as forbidden disks. This radius is the main
    # tuning knob for how widely the fly bypasses grass.
    # ------------------------------------------------------------------
    USE_GLOBAL_PATH_PLANNER: bool = True
    PATH_RESOLUTION: float = 0.75
    PATH_PAD: float = 8.0
    PATH_BLOCK_RADIUS: float = 2.8      # increase to 3.0 for wider clearance
    PATH_SOFT_RADIUS: float = 5.8       # penalize near-grass cells
    PATH_SOFT_WEIGHT: float = 3.5
    PATH_SMOOTH_MARGIN: float = 0.20
    PATH_LOOKAHEAD: float = 4.2
    PATH_REACHED_RADIUS: float = 2.3
    PATH_REPLAN_EVERY_STEPS: int = 9000
    PATH_REPLAN_IF_GRASS_CLOSER_THAN: float = 2.0
    PATH_EMERGENCY_REPLAN_COOLDOWN_STEPS: int = 500

    # Local repulsion fine-tunes the planned path without dominating it.
    REPULSE_RADIUS: float = 5.6
    REPULSE_GAIN_PATH: float = 0.85
    REPULSE_GAIN_HOME: float = 0.55
    HARD_DANGER_RADIUS: float = 1.45

    # Do not add the synthetic banana/2 grass point here: doing so would leak
    # information about the banana position. We only use grass geoms that are
    # actually present in the scene.
    ADD_SYNTHETIC_MID_GRASS: bool = False

    # ------------------------------------------------------------------
    # Odor-only banana navigation.
    # The banana position is inferred from olfactory sensor readings, never
    # from sim.world.banana_xy. The range estimate uses the published odor
    # model constants from miniproject/arena/banana.py.
    # ------------------------------------------------------------------
    ODOR_EPS: float = 1e-30
    ODOR_DIR_ALPHA: float = 0.38
    ODOR_GOAL_MIN_RANGE: float = 3.0
    ODOR_GOAL_MAX_RANGE: float = 35.0
    ODOR_STOP_THRESHOLD: float = 2.0e-5
    ODOR_FINAL_THRESHOLD: float = 5.0e-6
    ODOR_NEAR_THRESHOLD: float = 8.0e-7
    ODOR_MIN_GRAD_NORM: float = 1e-5
    ODOR_CAST_PERIOD_STEPS: int = 2400
    ODOR_CAST_BEARING_RAD: float = 0.65
    ODOR_PALP_WEIGHT: float = 9.0
    ODOR_ANTENNA_WEIGHT: float = 1.0
    ODOR_MODEL_DECAY: float = 500.0
    ODOR_MODEL_DIFFUSIVITY: float = 20000.0
    ODOR_MODEL_EMISSION_RATE: float = 1.0

    # ------------------------------------------------------------------
    # Stuck detection / recovery. Recovery is deliberately rare.
    # ------------------------------------------------------------------
    STUCK_WINDOW_STEPS: int = 3500
    STUCK_MIN_DISPLACEMENT: float = 0.50
    STUCK_MIN_PROGRESS: float = 0.20
    RECOVERY_BACK_STEPS: int = 360
    RECOVERY_PIVOT_STEPS: int = 540
    RECOVERY_COOLDOWN_STEPS: int = 2600

    # ------------------------------------------------------------------
    # Terrain and final approach
    # ------------------------------------------------------------------
    NEAR_TARGET_RADIUS: float = 7.0
    FINAL_TARGET_RADIUS: float = 4.2
    TILT_SLOW_RAD: float = 0.90
    TILT_RECOVER_RAD: float = 1.25

    # ------------------------------------------------------------------
    # Dragonfly level
    # ------------------------------------------------------------------
    DRAGON_LOOK_RADIUS: float = 25.0
    DRAGON_DANGER_RADIUS: float = 13.0
    DRAGON_BACKUP_STEPS: int = 500


# =============================================================================
# Helpers
# =============================================================================

def wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def segment_clearance(a: np.ndarray, b: np.ndarray, obstacles: np.ndarray, step: float = 0.35) -> float:
    if len(obstacles) == 0:
        return np.inf
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    length = norm(b - a)
    n = max(2, int(length / step) + 1)
    samples = a[None, :] + (b - a)[None, :] * np.linspace(0.0, 1.0, n)[:, None]
    dmin = np.inf
    for g in obstacles:
        dmin = min(dmin, float(np.min(np.linalg.norm(samples - g[None, :], axis=1))))
    return dmin


# =============================================================================
# Controller
# =============================================================================

class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.cfg = CFG()
        self.fly_name = sim.fly.name

        body_segments = sim.fly.get_bodysegs_order()
        self.thorax_idx = next(
            (i for i, seg in enumerate(body_segments) if getattr(seg, "name", "") == "c_thorax"),
            0,
        )
        self.thorax_body_id = sim._internal_bodyids_by_fly[self.fly_name][self.thorax_idx]

        # IMPORTANT: do not read sim.world.banana_xy here.
        # The moving goal below is estimated only from odor readings.
        self.odor_dir_world = np.array([1.0, 0.0], dtype=float)
        self.odor_goal_xy = np.array([self.cfg.ODOR_GOAL_MAX_RANGE, 0.0], dtype=float)
        self.odor_range_est = self.cfg.ODOR_GOAL_MAX_RANGE
        self.mean_odor = 0.0
        self.odor_grad_norm = 0.0
        self._odor_dir_initialized = False
        self._odor_lut_r, self._odor_lut_logc = self._make_odor_range_lut()

        self.grass_xy = self._extract_grass_xy(sim)

        self.turning_controller = TurningController(
            sim.timestep,
            intrinsic_freqs=np.ones(6) * self.cfg.CPG_FREQ,
            convergence_coefs=np.ones(6) * self.cfg.CPG_CONVERGENCE,
        )

        self.last_drives = np.array([self.cfg.DRIVE_FAST, self.cfg.DRIVE_FAST], dtype=float)
        # Some local evaluation/debug scripts read controller._drives after each step.
        self._drives = self.last_drives.copy()
        self._mode = "init"
        self._dist = float("inf")
        self._grass_dist = float("inf")
        self.pose_hist: deque[tuple[int, np.ndarray, float]] = deque()
        self.recovery_mode: str | None = None
        self.recovery_left = 0
        self.recovery_side = 1.0
        self.recovery_cooldown = 0
        self.dragon_backup_left = 0
        self.is_pivoting = False

        self.path: np.ndarray | None = None
        self.path_i = 0
        self.last_plan_step = -10**9
        self.plan_failed = False

        # Plan from the actual settled spawn position if possible, using the
        # initial odor-derived goal.
        try:
            start_pos, start_heading, _, _ = self._pose(sim)
            start_xy = start_pos[:2]
            self._update_odor_navigation(sim, start_xy, start_heading, step_i=0)
        except Exception:
            start_xy = np.zeros(2, dtype=float)
        self._plan_path(start_xy, step_i=0, force=True)

        if self.cfg.DEBUG:
            print("[controller] ODOR v5: banana_xy is not used by the controller")
            print(
                f"[controller] initial odor_goal={np.round(self.odor_goal_xy, 3)} "
                f"range_est={self.odor_range_est:.2f} mean_odor={self.mean_odor:.3e}"
            )
            print(f"[controller] CPG_FREQ={self.cfg.CPG_FREQ}, extracted grass={len(self.grass_xy)}")
            if len(self.grass_xy):
                print(f"[controller] first grass={np.round(self.grass_xy[:8], 2)}")
            if self.path is not None:
                print(f"[controller] planned path has {len(self.path)} waypoints:")
                print(np.round(self.path, 2))
            elif self.cfg.USE_GLOBAL_PATH_PLANNER:
                print("[controller] planner failed, using direct homing + local repulsion")

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------

    def _pose(self, sim: MiniprojectSimulation) -> tuple[np.ndarray, float, float, float]:
        pos = np.asarray(sim.get_body_positions(self.fly_name)[self.thorax_idx], dtype=float)
        xmat = np.asarray(sim.mj_data.xmat[self.thorax_body_id], dtype=float).reshape(3, 3)
        heading = math.atan2(xmat[1, 0], xmat[0, 0])
        roll = math.atan2(xmat[2, 1], xmat[2, 2])
        pitch = math.atan2(-xmat[2, 0], math.sqrt(xmat[2, 1] ** 2 + xmat[2, 2] ** 2))
        return pos, heading, roll, pitch

    def _body_axes(self, heading: float) -> tuple[np.ndarray, np.ndarray]:
        """Return forward and left unit vectors in world XY."""
        fwd = np.array([math.cos(heading), math.sin(heading)], dtype=float)
        left = np.array([-math.sin(heading), math.cos(heading)], dtype=float)
        return fwd, left

    def _odor_sensor_positions(self, sim: MiniprojectSimulation) -> np.ndarray:
        """World positions of the fly's olfactory sensors, in the same order as get_olfaction."""
        internal_ids = sim._intern_odor_sensorids_by_fly[self.fly_name]
        idx = sim.mj_model.sensor_adr[internal_ids][:, None] + np.arange(3)
        return np.asarray(sim.mj_data.sensordata[idx], dtype=float)

    def _make_odor_range_lut(self) -> tuple[np.ndarray, np.ndarray]:
        """Precompute concentration-vs-distance for the odor model.

        This uses only the public odor plume parameters, not the banana position.
        """
        try:
            from scipy.special import k0e
        except Exception:
            return np.array([self.cfg.ODOR_GOAL_MIN_RANGE, self.cfg.ODOR_GOAL_MAX_RANGE]), np.array([0.0, -40.0])

        r = np.linspace(0.35, 60.0, 2000)
        diffusivity = self.cfg.ODOR_MODEL_DIFFUSIVITY
        decay = self.cfg.ODOR_MODEL_DECAY
        emission_rate = self.cfg.ODOR_MODEL_EMISSION_RATE
        kappa_r = np.sqrt(decay / diffusivity) * r
        log_c = (
            np.log(emission_rate / (2.0 * np.pi * diffusivity))
            + np.log(k0e(kappa_r))
            - kappa_r
        )
        return r.astype(float), log_c.astype(float)

    def _range_from_mean_odor(self, mean_odor: float) -> float:
        """Convert odor concentration to a rough source distance estimate."""
        if not np.isfinite(mean_odor) or mean_odor <= 0.0:
            return self.cfg.ODOR_GOAL_MAX_RANGE
        logc = float(np.log(max(mean_odor, self.cfg.ODOR_EPS)))
        # log concentration monotonically decreases with distance, so reverse for np.interp.
        r = float(np.interp(logc, self._odor_lut_logc[::-1], self._odor_lut_r[::-1]))
        return float(np.clip(r, self.cfg.ODOR_GOAL_MIN_RANGE, self.cfg.ODOR_GOAL_MAX_RANGE))

    def _weighted_mean_odor(self, odor: np.ndarray) -> float:
        """Weighted mean odor matching the old palp/antenna weighting."""
        vals = np.asarray(odor, dtype=float).reshape(-1)
        if vals.size < 4:
            return float(np.mean(vals)) if vals.size else 0.0
        lp, rp, la, ra = vals[:4]
        weighted_sum = (
            self.cfg.ODOR_PALP_WEIGHT * float(lp)
            + self.cfg.ODOR_PALP_WEIGHT * float(rp)
            + self.cfg.ODOR_ANTENNA_WEIGHT * float(la)
            + self.cfg.ODOR_ANTENNA_WEIGHT * float(ra)
        )
        return weighted_sum / (2.0 * (self.cfg.ODOR_PALP_WEIGHT + self.cfg.ODOR_ANTENNA_WEIGHT))

    def _update_odor_navigation(self, sim: MiniprojectSimulation, xy: np.ndarray, heading: float, step_i: int) -> None:
        """Update self.odor_dir_world and self.odor_goal_xy from olfaction only.

        We estimate the local gradient of log odor concentration by fitting a plane
        through the four olfactory sensor readings. The gradient points toward
        increasing concentration, i.e. toward the banana odor source.
        """
        try:
            odor = np.asarray(sim.get_olfaction(self.fly_name), dtype=float)[:, 0]
            sensor_xy = self._odor_sensor_positions(sim)[:, :2]
        except Exception:
            odor = np.zeros(4, dtype=float)
            sensor_xy = np.zeros((4, 2), dtype=float)

        self.mean_odor = self._weighted_mean_odor(odor)
        self.odor_range_est = self._range_from_mean_odor(self.mean_odor)

        direction = None
        grad_norm = 0.0
        if odor.size >= 3 and sensor_xy.shape[0] == odor.size and np.all(np.isfinite(sensor_xy)):
            y = np.log(np.maximum(odor, self.cfg.ODOR_EPS))
            A = sensor_xy - np.mean(sensor_xy, axis=0, keepdims=True)
            b = y - float(np.mean(y))
            try:
                grad, *_ = np.linalg.lstsq(A, b, rcond=None)
                grad = np.asarray(grad, dtype=float)
                grad_norm = norm(grad)
                if np.all(np.isfinite(grad)) and grad_norm > self.cfg.ODOR_MIN_GRAD_NORM:
                    direction = unit(grad)
            except Exception:
                direction = None

        self.odor_grad_norm = float(grad_norm)
        if direction is None or norm(direction) < 1e-9:
            # If the odor gradient is temporarily ambiguous, keep moving roughly
            # in the last known odor direction. Add a slow casting oscillation so
            # the fly can reacquire the gradient if it points almost straight ahead
            # or if the body is temporarily tilted on hills.
            fwd, left = self._body_axes(heading)
            phase = 2.0 * np.pi * ((step_i % self.cfg.ODOR_CAST_PERIOD_STEPS) / self.cfg.ODOR_CAST_PERIOD_STEPS)
            cast = math.sin(phase) * self.cfg.ODOR_CAST_BEARING_RAD
            direction = unit(math.cos(cast) * self.odor_dir_world + math.sin(cast) * left + 0.15 * fwd)
        else:
            # On the first valid odor gradient, accept it immediately. After that,
            # smooth the odor direction to avoid zig-zagging from tiny sensor noise.
            if not self._odor_dir_initialized:
                direction = unit(direction)
                self._odor_dir_initialized = True
            else:
                direction = unit((1.0 - self.cfg.ODOR_DIR_ALPHA) * self.odor_dir_world + self.cfg.ODOR_DIR_ALPHA * direction)

        if norm(direction) < 1e-9:
            direction = self._body_axes(heading)[0]
        self.odor_dir_world = direction
        goal_range = float(np.clip(self.odor_range_est, self.cfg.ODOR_GOAL_MIN_RANGE, self.cfg.ODOR_GOAL_MAX_RANGE))
        self.odor_goal_xy = np.asarray(xy, dtype=float) + goal_range * self.odor_dir_world

    def _extract_grass_xy(self, sim: MiniprojectSimulation) -> np.ndarray:
        points: list[np.ndarray] = []
        if getattr(sim, "enable_grass", False):
            rgba = np.asarray(sim.mj_model.geom_rgba)
            for gid in range(sim.mj_model.ngeom):
                name = (sim.mj_model.geom(gid).name or "").lower()
                if "ground" in name or "terrain" in name or "hfield" in name or "floor" in name:
                    continue
                color = rgba[gid]
                # Grass blades are green mesh geoms. We avoid relying on names because
                # grass geoms are UUIDs in the generator.
                is_green = color[1] > 0.70 and color[0] < 0.45 and color[2] < 0.45 and color[3] > 0.5
                if not is_green:
                    continue
                try:
                    bid = int(sim.mj_model.geom_bodyid[gid])
                    p = np.asarray(sim.mj_data.xpos[bid][:2], dtype=float)
                except Exception:
                    p = np.asarray(sim.mj_data.geom_xpos[gid][:2], dtype=float)
                if np.all(np.isfinite(p)):
                    points.append(p)

        # Do not infer/add grass at banana/2 because that would require knowing
        # the banana position.

        if not points:
            return np.zeros((0, 2), dtype=float)

        arr = np.asarray(points, dtype=float)
        rounded = np.round(arr / 0.2) * 0.2
        _, idx = np.unique(rounded, axis=0, return_index=True)
        return arr[np.sort(idx)]

    def _dragon_xy(self, sim: MiniprojectSimulation) -> np.ndarray | None:
        if not getattr(sim, "enable_dragonfly", False):
            return None
        try:
            mocap_id = sim.world._get_dragonfly_mocap_id(sim)
            p = np.asarray(sim.mj_data.mocap_pos[mocap_id], dtype=float)
            if p[2] < -20:
                return None
            return p[:2]
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Global grass-aware path planning
    # ------------------------------------------------------------------

    def _nearest_grass_dist(self, xy: np.ndarray) -> float:
        if len(self.grass_xy) == 0:
            return np.inf
        return float(np.min(np.linalg.norm(self.grass_xy - xy[None, :], axis=1)))

    def _plan_bounds(self, start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pad = self.cfg.PATH_PAD
        if len(self.grass_xy):
            mn = np.minimum.reduce([start, goal, np.min(self.grass_xy, axis=0)]) - pad
            mx = np.maximum.reduce([start, goal, np.max(self.grass_xy, axis=0)]) + pad
        else:
            mn = np.minimum(start, goal) - pad
            mx = np.maximum(start, goal) + pad
        return mn.astype(float), mx.astype(float)

    def _world_to_grid(self, p: np.ndarray, mn: np.ndarray, res: float, nx: int, ny: int) -> tuple[int, int]:
        ij = np.round((np.asarray(p, dtype=float) - mn) / res).astype(int)
        ij[0] = int(np.clip(ij[0], 0, nx - 1))
        ij[1] = int(np.clip(ij[1], 0, ny - 1))
        return int(ij[0]), int(ij[1])

    def _grid_to_world(self, ij: tuple[int, int], mn: np.ndarray, res: float) -> np.ndarray:
        return mn + res * np.asarray(ij, dtype=float)

    def _line_is_clear(self, a: np.ndarray, b: np.ndarray, radius: float | None = None) -> bool:
        if len(self.grass_xy) == 0:
            return True
        if radius is None:
            radius = self.cfg.PATH_BLOCK_RADIUS + self.cfg.PATH_SMOOTH_MARGIN
        return segment_clearance(a, b, self.grass_xy) > radius

    def _smooth_path(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) <= 2:
            return pts
        smoothed = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            best = i + 1
            # Jump as far as possible without cutting through a forbidden disk.
            for j in range(len(pts) - 1, i, -1):
                if self._line_is_clear(pts[i], pts[j]):
                    best = j
                    break
            smoothed.append(pts[best])
            i = best
        return np.asarray(smoothed, dtype=float)

    def _plan_path(self, start_xy: np.ndarray, step_i: int, force: bool = False) -> None:
        if not self.cfg.USE_GLOBAL_PATH_PLANNER or len(self.grass_xy) == 0:
            self.path = None
            self.plan_failed = True
            return
        if not force and step_i - self.last_plan_step < self.cfg.PATH_REPLAN_EVERY_STEPS:
            return

        start = np.asarray(start_xy, dtype=float)
        goal = np.asarray(self.odor_goal_xy, dtype=float)
        if not np.all(np.isfinite(goal)):
            self.path = None
            self.plan_failed = True
            return
        res = self.cfg.PATH_RESOLUTION
        mn, mx = self._plan_bounds(start, goal)
        nx = int(np.ceil((mx[0] - mn[0]) / res)) + 1
        ny = int(np.ceil((mx[1] - mn[1]) / res)) + 1
        if nx <= 2 or ny <= 2 or nx * ny > 60_000:
            self.path = None
            self.plan_failed = True
            return

        xs = mn[0] + np.arange(nx) * res
        ys = mn[1] + np.arange(ny) * res
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        occ = np.zeros((nx, ny), dtype=bool)
        soft = np.zeros((nx, ny), dtype=float)

        block_r = self.cfg.PATH_BLOCK_RADIUS
        soft_r = max(self.cfg.PATH_SOFT_RADIUS, block_r + 0.1)
        for g in self.grass_xy:
            D = np.hypot(X - g[0], Y - g[1])
            occ |= D < block_r
            mask = (D >= block_r) & (D < soft_r)
            soft[mask] += self.cfg.PATH_SOFT_WEIGHT * ((soft_r - D[mask]) / (soft_r - block_r)) ** 2

        s_idx = self._world_to_grid(start, mn, res, nx, ny)
        g_idx = self._world_to_grid(goal, mn, res, nx, ny)
        occ[s_idx] = False
        occ[g_idx] = False

        moves = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]
        open_heap: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, s_idx))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {s_idx: 0.0}
        closed: set[tuple[int, int]] = set()
        found = False

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == g_idx:
                found = True
                break
            closed.add(cur)

            for dx, dy in moves:
                nb = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nb[0] < nx and 0 <= nb[1] < ny):
                    continue
                if occ[nb]:
                    continue
                step_cost = math.hypot(dx, dy) * res
                # Soft penalty keeps the path centered between grass branches.
                tentative = g_score[cur] + step_cost * (1.0 + soft[nb])
                if tentative < g_score.get(nb, np.inf):
                    came_from[nb] = cur
                    g_score[nb] = tentative
                    h = norm((np.asarray(nb) - np.asarray(g_idx)) * res)
                    heapq.heappush(open_heap, (tentative + h, nb))

        self.last_plan_step = step_i
        if not found:
            self.path = None
            self.path_i = 0
            self.plan_failed = True
            return

        cells = []
        cur = g_idx
        while cur != s_idx:
            cells.append(cur)
            cur = came_from[cur]
        cells.append(s_idx)
        cells.reverse()

        pts = np.asarray([self._grid_to_world(c, mn, res) for c in cells], dtype=float)
        pts[0] = start
        pts[-1] = goal
        pts = self._smooth_path(pts)
        pts[0] = start
        pts[-1] = goal

        self.path = pts
        self.path_i = 0
        self.plan_failed = False

        if self.cfg.DEBUG:
            min_clear = np.inf
            if len(pts) >= 2 and len(self.grass_xy):
                min_clear = min(segment_clearance(a, b, self.grass_xy) for a, b in zip(pts[:-1], pts[1:]))
            print(f"[controller] replanned path at step={step_i}, waypoints={len(pts)}, min_clear={min_clear:.2f}")
            print(np.round(pts, 2))

    def _path_goal(self, xy: np.ndarray) -> tuple[np.ndarray, str]:
        if self.path is None or len(self.path) == 0:
            return self.odor_goal_xy, "odor_home"

        # Move the path index forward; never move it backward.
        if self.path_i < len(self.path) - 1:
            window = self.path[self.path_i:]
            nearest_rel = int(np.argmin(np.linalg.norm(window - xy[None, :], axis=1)))
            self.path_i = max(self.path_i, self.path_i + nearest_rel)

        while self.path_i < len(self.path) - 1 and norm(self.path[self.path_i] - xy) < self.cfg.PATH_REACHED_RADIUS:
            self.path_i += 1

        # Pure pursuit: choose a point a few mm ahead along the path.
        goal_idx = self.path_i
        accum = 0.0
        last = xy
        for j in range(self.path_i, len(self.path)):
            accum += norm(self.path[j] - last)
            last = self.path[j]
            goal_idx = j
            if accum >= self.cfg.PATH_LOOKAHEAD:
                break

        if goal_idx >= len(self.path) - 1:
            return self.odor_goal_xy, "odor_home"
        return self.path[goal_idx], "path"

    # ------------------------------------------------------------------
    # Local obstacle terms and recovery
    # ------------------------------------------------------------------

    def _local_repulsion(self, xy: np.ndarray, mode: str) -> np.ndarray:
        rep = np.zeros(2, dtype=float)
        if len(self.grass_xy) == 0:
            return rep
        gain = self.cfg.REPULSE_GAIN_PATH if mode == "path" else self.cfg.REPULSE_GAIN_HOME
        for g in self.grass_xy:
            rel = xy - g
            d = norm(rel)
            if 1e-6 < d < self.cfg.REPULSE_RADIUS:
                strength = gain * ((self.cfg.REPULSE_RADIUS - d) / self.cfg.REPULSE_RADIUS) ** 2
                rep += strength * unit(rel)
        return rep

    def _update_stuck_window(self, step: int, xy: np.ndarray, dist: float) -> bool:
        self.pose_hist.append((step, xy.copy(), float(dist)))
        while len(self.pose_hist) > 2 and step - self.pose_hist[0][0] > self.cfg.STUCK_WINDOW_STEPS:
            self.pose_hist.popleft()
        if len(self.pose_hist) < 2:
            return False
        old_step, old_xy, old_dist = self.pose_hist[0]
        if step - old_step < self.cfg.STUCK_WINDOW_STEPS:
            return False
        displacement = norm(xy - old_xy)
        progress = old_dist - dist
        return displacement < self.cfg.STUCK_MIN_DISPLACEMENT and progress < self.cfg.STUCK_MIN_PROGRESS

    def _start_recovery(self, err: float, reason: str) -> None:
        if self.recovery_mode is not None or self.recovery_cooldown > 0:
            return
        self.recovery_mode = "back"
        self.recovery_left = self.cfg.RECOVERY_BACK_STEPS
        self.recovery_side = +1.0 if err >= 0 else -1.0
        if self.cfg.DEBUG:
            print(f"[controller] START recovery reason={reason} side={self.recovery_side:+.0f}")

    def _recovery_drives(self) -> np.ndarray | None:
        if self.recovery_cooldown > 0:
            self.recovery_cooldown -= 1
        if self.recovery_mode is None:
            return None

        self.recovery_left -= 1
        s = self.recovery_side
        if self.recovery_mode == "back":
            drives = np.array([
                self.cfg.DRIVE_BACK * (1.0 + 0.16 * s),
                self.cfg.DRIVE_BACK * (1.0 - 0.16 * s),
            ])
            if self.recovery_left <= 0:
                self.recovery_mode = "pivot"
                self.recovery_left = self.cfg.RECOVERY_PIVOT_STEPS
            return drives

        if self.recovery_mode == "pivot":
            drives = np.array([-s * 0.75, s * 0.75], dtype=float)
            if self.recovery_left <= 0:
                self.recovery_mode = None
                self.recovery_cooldown = self.cfg.RECOVERY_COOLDOWN_STEPS
                self.pose_hist.clear()
                # After a real recovery, recompute route from current location soon.
                self.last_plan_step = -10**9
            return drives

        self.recovery_mode = None
        return None

    # ------------------------------------------------------------------
    # Drive synthesis
    # ------------------------------------------------------------------

    def _pivot_drives(self, err: float) -> np.ndarray:
        if err >= 0:  # target/path point is to the left
            return np.array([self.cfg.PIVOT_INNER_BACK, self.cfg.PIVOT_OUTER], dtype=float)
        return np.array([self.cfg.PIVOT_OUTER, self.cfg.PIVOT_INNER_BACK], dtype=float)

    def _walk_drives(self, err: float, base: float, gain: float, min_forward: float) -> np.ndarray:
        turn = gain * math.tanh(self.cfg.TURN_SHARPNESS * err)
        cap = max(0.0, base - min_forward)
        turn = float(np.clip(turn, -cap, cap))
        # Positive err = desired direction left. Slow left legs, speed right legs.
        left = base - turn
        right = base + turn
        return np.clip(np.array([left, right], dtype=float), -self.cfg.DRIVE_MAX, self.cfg.DRIVE_MAX)

    def _smooth(self, drives: np.ndarray, hard: bool) -> np.ndarray:
        alpha = 0.12 if not hard else 0.02
        self.last_drives = alpha * self.last_drives + (1.0 - alpha) * drives
        return self.last_drives.copy()

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, sim: MiniprojectSimulation):
        pos, heading, roll, pitch = self._pose(sim)
        xy = pos[:2]
        step_i = int(getattr(sim, "_curr_step", 0))
        self._update_odor_navigation(sim, xy, heading, step_i)

        # Odor-derived distance proxy. This replaces distance to sim.world.banana_xy.
        dist = float(self.odor_range_est)
        grass_dist = self._nearest_grass_dist(xy)

        # Replan occasionally, and immediately if the planned route is bringing us
        # too close to grass. The planning goal is self.odor_goal_xy, not banana_xy.
        if self.cfg.USE_GLOBAL_PATH_PLANNER and len(self.grass_xy):
            emergency_replan_ok = (
                grass_dist < self.cfg.PATH_REPLAN_IF_GRASS_CLOSER_THAN
                and step_i - self.last_plan_step >= self.cfg.PATH_EMERGENCY_REPLAN_COOLDOWN_STEPS
            )
            odor_goal_drift = self.path is not None and norm(self.path[-1] - self.odor_goal_xy) > 4.5
            if emergency_replan_ok or odor_goal_drift:
                self._plan_path(xy, step_i=step_i, force=True)
            else:
                self._plan_path(xy, step_i=step_i, force=False)

        current_goal, mode = self._path_goal(xy)
        desired_vec = unit(current_goal - xy)
        if norm(desired_vec) < 1e-9:
            desired_vec = self.odor_dir_world.copy()
            mode = "odor_home"

        # Local repulsion is only a correction around the odor/path direction.
        rep = self._local_repulsion(xy, mode)
        if mode == "path":
            desired_vec = unit(2.2 * desired_vec + rep)
        else:
            desired_vec = unit(2.0 * desired_vec + rep)

        # Dragonfly avoidance for level 4.
        dragon = self._dragon_xy(sim)
        if dragon is not None:
            d_dragon = norm(dragon - xy)
            if d_dragon < self.cfg.DRAGON_DANGER_RADIUS:
                self.dragon_backup_left = self.cfg.DRAGON_BACKUP_STEPS
            elif d_dragon < self.cfg.DRAGON_LOOK_RADIUS:
                desired_vec = unit(desired_vec + 1.5 * unit(xy - dragon))
                mode = "dragon"

        desired_heading = math.atan2(desired_vec[1], desired_vec[0])
        err = wrap_pi(desired_heading - heading)

        stuck = self._update_stuck_window(step_i, xy, dist)
        tilted = max(abs(roll), abs(pitch)) > self.cfg.TILT_RECOVER_RAD
        if stuck:
            self._start_recovery(err, "stuck_window")
        elif tilted:
            self._start_recovery(err, "tilt")
        elif grass_dist < self.cfg.HARD_DANGER_RADIUS:
            self._start_recovery(err, "grass_emergency")

        hard = False
        recovery = self._recovery_drives()
        if recovery is not None:
            drives = recovery
            hard = True
            mode = self.recovery_mode or "recover"
        elif self.dragon_backup_left > 0:
            self.dragon_backup_left -= 1
            drives = np.array([self.cfg.DRIVE_BACK, self.cfg.DRIVE_BACK], dtype=float)
            hard = True
            mode = "dragon_back"
        else:
            if self.mean_odor > self.cfg.ODOR_STOP_THRESHOLD:
                # We are essentially on the source: stop instead of overshooting.
                base = 0.0
                gain = 0.0
                min_forward = 0.0
                mode = "odor_stop"
            elif self.mean_odor > self.cfg.ODOR_FINAL_THRESHOLD or dist < self.cfg.FINAL_TARGET_RADIUS:
                base = self.cfg.DRIVE_FINAL
                gain = self.cfg.TURN_GAIN_FINAL
                min_forward = 0.20
            elif self.mean_odor > self.cfg.ODOR_NEAR_THRESHOLD or dist < self.cfg.NEAR_TARGET_RADIUS:
                base = self.cfg.DRIVE_HILL
                gain = self.cfg.TURN_GAIN_FINAL
                min_forward = 0.22
            elif max(abs(roll), abs(pitch)) > self.cfg.TILT_SLOW_RAD:
                base = self.cfg.DRIVE_HILL
                gain = self.cfg.TURN_GAIN_PATH
                min_forward = 0.24
                mode = mode + "+tilt"
            elif grass_dist < self.cfg.PATH_BLOCK_RADIUS + 0.7:
                base = self.cfg.DRIVE_AVOID
                gain = self.cfg.TURN_GAIN_AVOID
                min_forward = self.cfg.MIN_FORWARD_AVOID
                mode = mode + "+closegrass"
            elif mode == "path":
                base = self.cfg.DRIVE_PATH
                gain = self.cfg.TURN_GAIN_PATH
                min_forward = self.cfg.MIN_FORWARD_PATH
            else:
                base = self.cfg.DRIVE_SPRINT if dist > 14.0 else self.cfg.DRIVE_FAST
                gain = self.cfg.TURN_GAIN_FAST
                min_forward = self.cfg.MIN_FORWARD_FAST

            if mode == "odor_stop":
                self.is_pivoting = False
                drives = np.array([0.0, 0.0], dtype=float)
                hard = True
            else:
                if self.is_pivoting:
                    if abs(err) < self.cfg.PIVOT_RELEASE_ERR:
                        self.is_pivoting = False
                elif abs(err) > self.cfg.PIVOT_ERR:
                    self.is_pivoting = True

                if self.is_pivoting:
                    drives = self._pivot_drives(err)
                    hard = True
                    mode = "pivot"
                else:
                    drives = self._walk_drives(err, base, gain, min_forward)

        drives = self._smooth(drives, hard=hard)

        # Public debug attributes used by scripts/eval_miniproject_controller.py.
        self._drives = drives.copy()
        self._mode = str(mode)
        self._dist = float(dist)
        self._grass_dist = float(grass_dist)

        if self.cfg.DEBUG and step_i % self.cfg.DEBUG_EVERY == 0:
            goal_txt = f"({current_goal[0]:+.1f},{current_goal[1]:+.1f})"
            ptxt = "none" if self.path is None else f"{self.path_i}/{len(self.path)-1}"
            print(
                f"[controller] step={step_i:6d} xy=({xy[0]:+.1f},{xy[1]:+.1f}) "
                f"odor_range={dist:5.2f} odor={self.mean_odor:.2e} grad={self.odor_grad_norm:.2e} "
                f"hdg={heading:+.2f} err={err:+.2f} "
                f"mode={mode:>15s} grass={grass_dist:4.1f} path={ptxt:>7s} "
                f"goal={goal_txt:>15s} drives=({drives[0]:+.2f},{drives[1]:+.2f})"
            )

        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
