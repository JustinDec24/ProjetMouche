"""Hierarchical controller for the COBAR 2026 miniproject.

Architecture (intentionally simple, three thin layers):

    Perception          ->   Navigation           ->   Motor
    (odor / vision /         (odor + avoid-grass        (drives, adhesion,
     terrain)                  + seek-sky, blended)      TurningController)

Design notes:
    * One decision point (``_decide``) called at 20 Hz; CPG runs every step.
    * Vision uses the **raw RGB camera images** (already cached by flygym).
      Grass geoms are rgba=(0, 1, 0, 1) so a pure-green mask is a very
      clean "obstacle" detector, and a blue mask gives a "free sky" map.
      This is far more robust than the per-ommatidium hex channels, which
      have half their entries zero and dilute the signal.
    * Navigation combines three signals with explicit priorities:
        odor_steer    (toward the banana, log-odor L/R)
        avoid_steer   (away from the side with more grass)
        seek_steer    (toward the side with more visible sky)
      The weight between odor and vision grows smoothly with the front
      obstacle density (soft blending, no on/off hack).
    * Terrain modulates speed and damps steering; no behavior change on
      flat worlds (Level 0 identical to before).
    * All tunables are grouped by layer at the top of the class.

The interface is identical to the previous controller:
    Controller(sim).step(sim) -> (joint_angles, adhesion)
"""

from __future__ import annotations

import numpy as np

from miniproject.simulation import MiniprojectSimulation


class Controller:
    # =====================================================================
    # PARAMETERS
    # =====================================================================

    # --- scheduling ---
    DECISION_INTERVAL_S = 0.05  # 20 Hz high-level decision

    # --- DEBUG ---
    DEBUG = False               # set True externally to enable per-decision prints
    DEBUG_EVERY = 1             # print every N decisions

    # --- ODOR (Level 0) ---
    PALP_WEIGHT = 9.0
    ANTENNA_WEIGHT = 1.0
    ODOR_GAIN = -1.5            # negative = attractive (avoid saturating tanh too fast)
    ODOR_SMOOTHING = 0.45       # EMA factor (higher = smoother)
    STOP_ODOR_THRESHOLD = 2e-5  # arrived at banana

    # --- VISION (Level 2) ---
    # IMPORTANT: the terrain ground plane is rendered with the SAME rgba as
    # grass (0, 1, 0). We therefore CANNOT use "pure green" to detect grass.
    # Instead we use SKY DENSITY as the inverse of "how much stuff is blocking
    # forward view". Sky is blue; grass blades poking up block the blue. This
    # is independent of ground color and robust to rolling hills.
    VALID_PIXEL_THRESHOLD = 0.04  # drop black fisheye-border pixels (RGB sum)
    SKY_B_MIN = 0.35
    SKY_MARGIN = 0.04             # require B > G + SKY_MARGIN to be "sky"
    OPENNESS_REFERENCE = 0.22     # typical sky fraction in open terrain (calibrated)
    # A "front blocked" signal in [0, 1] is built from (1 - sky_density / ref).
    VISION_GAIN = 2.5             # tanh gain on left/right asymmetry
    VISION_SMOOTHING = 0.45       # EMA on lr (commit-to-turn)
    VISION_FRONT_SMOOTHING = 0.35
    OBSTACLE_LOW = 0.30           # below: odor only
    OBSTACLE_HIGH = 0.80           # above: vision fully engaged (but still bounded)
    VISION_BIAS_MAX = 0.45        # vision's steering contribution is CAPPED at this
                                   # (odor always remains the primary signal)
    CLUTTER_BRAKE = 0.3           # very mild slowdown in cluttered scenes
    OVERRIDE_SPEED_FACTOR = 0.90  # slight slowdown when vision dominates

    # --- TERRAIN (Level 1) ---
    DOWNHILL_BRAKE = 1.2
    STEEP_BRAKE = 1.8           # strong brake on steep slopes (avoid cresting hills fast)
    VERY_STEEP = 0.70           # slope_mag above which we aggressively walk along contour
    TURN_STEEP_GAIN = 1.0       # moderate damping from slope
    TURN_TILT_GAIN = 4.0        # strong damping from actual body tilt
    TILT_SAFE = 0.30            # only give up on steering when REALLY tilted (up<0.7)
    AIRBORNE_N = 3              # consecutive decisions with 0 leg contact -> freeze

    # --- DRIVES / MOTOR ---
    BASE_DRIVE = 1.65
    MAX_DRIVE = 1.90
    MIN_DRIVE = 0.55
    MIN_DRIVE_TERRAIN = 0.50
    MIN_SIDE_DRIVE = 0.30
    MIN_SIDE_DRIVE_TERRAIN = 0.30
    TURN_MOD = 0.50             # asymmetry strength (keeps drive ratio > 0.5)
    STEER_CAP = 0.40            # hard cap on |steer| (prevents rollover)

    # --- GRIP ---
    CONTACT_THRESHOLD = 0.15
    GRIP_TILT = 0.10            # activate very early: up < 0.90 forces full adhesion
    GRIP_SLOPE = 0.06

    # =====================================================================
    # INIT
    # =====================================================================

    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        self._decision_every = max(1, int(self.DECISION_INTERVAL_S / sim.timestep))
        self._step_count = 0
        self._drives = np.array([1.0, 1.0])
        self._stopped = False

        fly_segs = sim.fly.get_bodysegs_order()
        self._thorax_idx = next(
            i for i, s in enumerate(fly_segs) if s.name == "c_thorax"
        )
        body_ids = sim._internal_bodyids_by_fly[sim.fly.name]
        self._thorax_body_id = body_ids[self._thorax_idx]
        self._contact_body_ids = sim._internal_contact_body_segment_ids_by_fly[
            sim.fly.name
        ]

        # Smoothing / navigation state
        self._smooth_odor = 0.0
        self._smooth_vision_lr = 0.0
        self._smooth_vision_front = 0.0
        self._airborne_counter = 0

        # Pre-probe world capabilities
        self._has_terrain = callable(getattr(getattr(sim, "world", None), "get_normal", None))

        # Debug counters
        self._dbg_decision_count = 0

    # =====================================================================
    # MAIN STEP
    # =====================================================================

    def step(self, sim: MiniprojectSimulation):
        if self._step_count % self._decision_every == 0:
            self._drives = self._decide(sim)
        self._step_count += 1

        joint_angles, adhesion = self.turning_controller.step(self._drives)
        adhesion = self._adhesion_postprocess(sim, adhesion)
        return joint_angles, adhesion

    # =====================================================================
    # DECISION
    # =====================================================================

    def _decide(self, sim) -> np.ndarray:
        if self._stopped:
            self._debug_print(sim, None, None, None, None, "STOPPED")
            return np.array([0.0, 0.0])

        # ---- PERCEPTION ----
        odor = self._perceive_odor(sim)               # (lin_mean, log_bias)
        lin_mean, log_bias = odor

        if lin_mean > self.STOP_ODOR_THRESHOLD:
            self._stopped = True
            self._debug_print(sim, odor, None, None, None, "REACHED_SOURCE")
            return np.array([0.0, 0.0])

        vision = self._perceive_vision(sim)           # (front, lr, sL, sR, obscL, obscR)
        terrain = self._perceive_terrain(sim)         # (downhill, slope_mag, uphill)

        # Airborne safeguard: if the fly has no leg contact for several
        # consecutive decisions, stop commanding motion so it can settle.
        try:
            cf = sim.get_external_force(sim.fly.name, subtract_adhesion_force=True)
            cmag = np.linalg.norm(np.asarray(cf, dtype=float), axis=1) \
                if np.asarray(cf).ndim == 2 else np.zeros(6)
            nc = int((cmag[:6] > self.CONTACT_THRESHOLD).sum())
        except Exception:
            nc = 6
        if nc == 0:
            self._airborne_counter += 1
        else:
            self._airborne_counter = 0
        if self._airborne_counter >= self.AIRBORNE_N:
            # Walk straight slowly; wait to regain contact.
            return np.array([self.MIN_DRIVE_TERRAIN, self.MIN_DRIVE_TERRAIN])

        # ---- NAVIGATION ----
        # Odor steering (log-space bias, smoothed).
        self._smooth_odor += (1.0 - self.ODOR_SMOOTHING) * (
            self.ODOR_GAIN * log_bias - self._smooth_odor
        )
        odor_steer = float(np.tanh(self._smooth_odor))

        # Vision steering (commit-to-turn EMA).
        front, lr_raw, _, _, _, _ = vision
        self._smooth_vision_lr += (1.0 - self.VISION_SMOOTHING) * (
            lr_raw - self._smooth_vision_lr
        )
        self._smooth_vision_front += (1.0 - self.VISION_FRONT_SMOOTHING) * (
            front - self._smooth_vision_front
        )
        vision_steer = float(np.tanh(self.VISION_GAIN * self._smooth_vision_lr))
        front_s = float(self._smooth_vision_front)

        # Vision adds a *bounded nudge* on top of the odor steer. The odor
        # direction is always the primary signal; vision only biases the
        # trajectory to avoid blocked directions. This is critical because
        # both hills (Level 1, to cross) and grass (Level 2, to avoid) look
        # similar to the sky-density detector, and we must not let vision
        # override odor when it is wrong.
        w = self._smoothstep(self.OBSTACLE_LOW, self.OBSTACLE_HIGH, front_s)
        vision_bias = float(np.clip(
            w * vision_steer,
            -self.VISION_BIAS_MAX,
            +self.VISION_BIAS_MAX,
        ))
        steer = odor_steer + vision_bias

        # Anti-rollover: damp steering on steep ground AND when already tilted.
        downhill, slope_mag, uphill = terrain
        xmat = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
        tilt = max(0.0, 1.0 - float(xmat[2, 2]))  # 0 upright -> 1 sideways
        steer_damp = 1.0 / (
            1.0
            + self.TURN_STEEP_GAIN * slope_mag
            + self.TURN_TILT_GAIN * tilt
        )
        steer = steer * steer_damp
        # Hard safety: if already very tilted, go straight forward.
        if tilt > self.TILT_SAFE:
            steer = 0.0

        # Speed: brake for downhill, uphill, slope and clutter.
        # Cresting a hill too fast makes the fly lose contact and tumble;
        # that's why we brake BOTH uphill (to avoid launching off a ridge)
        # and downhill (not to overshoot into a valley).
        speed_scale = 1.0 / (
            1.0
            + self.DOWNHILL_BRAKE * downhill
            + self.DOWNHILL_BRAKE * uphill
            + self.STEEP_BRAKE * slope_mag
            + self.CLUTTER_BRAKE * front_s
        )
        if w > 0.7:
            speed_scale *= self.OVERRIDE_SPEED_FACTOR

        drives = self._to_drives(steer, speed_scale)

        self._debug_print(
            sim, odor, vision, terrain,
            {"odor_steer": odor_steer, "vision_steer": vision_bias, "w": w,
             "steer": steer, "speed_scale": speed_scale, "drives": drives},
            "",
        )
        return drives

    # ---------------------------------------------------------------------
    # Perception
    # ---------------------------------------------------------------------

    def _perceive_odor(self, sim) -> tuple[float, float]:
        """Return ``(linear_mean, log_bias = left - right)``.

        ``linear_mean`` is used for stop detection.
        ``log_bias`` is a distance-independent left/right steering signal.
        """
        odor_lin = sim.get_olfaction(sim.fly.name)                    # (4, 1)
        odor_log = sim.get_olfaction(sim.fly.name, log=True)
        lp_l, rp_l, la_l, ra_l = odor_lin[:, 0]
        lp_g, rp_g, la_g, ra_g = odor_log[:, 0]

        left_lin = self.PALP_WEIGHT * float(lp_l) + self.ANTENNA_WEIGHT * float(la_l)
        right_lin = self.PALP_WEIGHT * float(rp_l) + self.ANTENNA_WEIGHT * float(ra_l)
        lin_mean = 0.5 * (left_lin + right_lin)

        left_log = self.PALP_WEIGHT * float(lp_g) + self.ANTENNA_WEIGHT * float(la_g)
        right_log = self.PALP_WEIGHT * float(rp_g) + self.ANTENNA_WEIGHT * float(ra_g)
        log_bias = left_log - right_log

        return lin_mean, log_bias

    def _perceive_vision(self, sim) -> tuple[float, float, float, float, float, float]:
        """Return ``(front_blocked, lr_score, sL, sR, obscL, obscR)``.

        The ground plane geom in the terrain arena uses the **same rgba as
        grass** (pure green), so a green-color detector cannot distinguish
        obstacles from the floor. Instead we use SKY DENSITY: sky is the
        blue background that fills the open parts of the view. When a
        grass blade is in front, it occludes the blue.

        We normalize fractions by the count of VALID (non-black) pixels
        because ``correct_fisheye`` produces large black borders outside
        the fisheye disc.

        - ``front_blocked`` in [0, 1]: overall obscuration averaged over
          both eyes, relative to OPENNESS_REFERENCE.
        - ``lr_score`` in [-1, 1]: + => the RIGHT eye sees MORE sky, so
          turning right goes toward open space (convention: steer>0 =
          turn right).
        - ``sL, sR``: sky density per eye (valid pixels).
        - ``obscL, obscR``: 1 - sL/ref, 1 - sR/ref (for debug).
        """
        try:
            raws = sim.get_raw_vision(sim.fly.name)  # [(H,W,3) left, right]
        except Exception:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        def sky_density(img_u8):
            img = img_u8.astype(np.float32) / 255.0
            Rc, Gc, Bc = img[..., 0], img[..., 1], img[..., 2]
            valid = (Rc + Gc + Bc) > self.VALID_PIXEL_THRESHOLD
            n_valid = int(valid.sum())
            if n_valid < 100:
                return 0.0
            sky = (Bc > self.SKY_B_MIN) & (Bc > Gc + self.SKY_MARGIN)
            return float(sky.sum()) / float(n_valid)

        sL = sky_density(raws[0])
        sR = sky_density(raws[1])

        # Reference = expected sky density in fully open terrain.
        ref = max(self.OPENNESS_REFERENCE, 1e-3)
        obscL = float(np.clip(1.0 - sL / ref, 0.0, 1.0))
        obscR = float(np.clip(1.0 - sR / ref, 0.0, 1.0))
        front_blocked = 0.5 * (obscL + obscR)

        # Sign convention: steer > 0 => turn RIGHT.
        # We want to turn TOWARD the more open eye (more sky).
        # If right eye has more sky: steer > 0 (turn right). => (sR - sL) > 0.
        denom = max(sL + sR, 1e-3)
        lr_score = float(np.clip((sR - sL) / denom, -1.0, 1.0))

        return float(front_blocked), lr_score, sL, sR, obscL, obscR

    def _perceive_terrain(self, sim) -> tuple[float, float, float]:
        """Return ``(downhill, slope_mag, uphill)``.

        - ``slope_mag``: norm of the terrain gradient at the fly's feet.
        - ``downhill``: slope projected along forward heading, clipped >= 0
          when the fly faces DOWN the slope.
        - ``uphill``: same, clipped >= 0 when facing UP. Mirror of downhill.

        All zero on flat terrain (Level 0).
        """
        if not self._has_terrain:
            return 0.0, 0.0, 0.0

        try:
            thorax_xy = sim.get_body_positions(sim.fly.name)[self._thorax_idx, :2]
        except Exception:
            thorax_xy = sim.mj_data.xpos[self._thorax_body_id, :2]

        try:
            n = np.asarray(
                sim.world.get_normal(float(thorax_xy[0]), float(thorax_xy[1])),
                dtype=float,
            )
        except Exception:
            return 0.0, 0.0, 0.0

        if n.shape != (3,) or not np.isfinite(n).all() or abs(n[2]) < 1e-6:
            return 0.0, 0.0, 0.0

        grad = np.array([-n[0] / n[2], -n[1] / n[2]], dtype=float)
        slope_mag = float(np.linalg.norm(grad))

        xmat = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
        heading_xy = xmat[:2, 0]
        hn = np.linalg.norm(heading_xy)
        if hn > 1e-12:
            heading_xy = heading_xy / hn
        slope_forward = float(np.dot(heading_xy, grad))
        downhill = max(0.0, -slope_forward)
        uphill = max(0.0, slope_forward)
        return downhill, slope_mag, uphill

    # ---------------------------------------------------------------------
    # Motor mapping
    # ---------------------------------------------------------------------

    def _to_drives(self, steer: float, speed_scale: float) -> np.ndarray:
        """Map (steer in [-1,1], speed_scale > 0) -> [dL, dR].

        Convention: ``steer > 0`` => turn RIGHT (slow right side, keep left side).
        """
        if self._has_terrain:
            min_drive = self.MIN_DRIVE_TERRAIN
            min_side = self.MIN_SIDE_DRIVE_TERRAIN
        else:
            min_drive = self.MIN_DRIVE
            min_side = self.MIN_SIDE_DRIVE

        base = float(np.clip(self.BASE_DRIVE * speed_scale, min_drive, self.MAX_DRIVE))
        # Cap the steering magnitude: without this, saturated tanh + asymmetric
        # drive gives ratios > 2:1 which spin the fly in place instead of
        # producing a usable curved trajectory.
        steer = float(np.clip(steer, -self.STEER_CAP, self.STEER_CAP))
        drives = np.array([base, base], dtype=float)
        side = 1 if steer > 0.0 else 0
        drives[side] = max(min_side, base - abs(steer) * self.TURN_MOD * base)
        return np.clip(drives, 0.0, self.MAX_DRIVE)

    # ---------------------------------------------------------------------
    # Adhesion post-processing (Level 1 grip)
    # ---------------------------------------------------------------------

    def _adhesion_postprocess(self, sim, adhesion: np.ndarray) -> np.ndarray:
        adhesion = np.asarray(adhesion, dtype=float).copy()

        xmat = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
        uprightness = xmat[2, 2]
        tilt = max(0.0, 1.0 - uprightness)

        slope_mag = 0.0
        if self._has_terrain:
            _, slope_mag, _ = self._perceive_terrain(sim)

        if tilt > self.GRIP_TILT or slope_mag > self.GRIP_SLOPE:
            adhesion = np.where(adhesion > 0.0, 1.0, adhesion)

        try:
            contact_forces = sim.get_external_force(
                sim.fly.name, subtract_adhesion_force=True
            )
        except Exception:
            contact_forces = sim.mj_data.cfrc_ext[self._contact_body_ids, 3:]

        contact_forces = np.asarray(contact_forces, dtype=float)
        if contact_forces.ndim == 2 and contact_forces.shape[1] == 3:
            contact_mag = np.linalg.norm(contact_forces, axis=1)
            n = min(6, len(contact_mag), len(adhesion))
            for i in range(n):
                if adhesion[i] > 0.0 and contact_mag[i] < self.CONTACT_THRESHOLD:
                    adhesion[i] = 1.0

        return np.clip(adhesion, 0.0, 1.0)

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _smoothstep(lo: float, hi: float, x: float) -> float:
        if hi <= lo:
            return 0.0 if x < lo else 1.0
        t = float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    # ---------------------------------------------------------------------
    # Debug
    # ---------------------------------------------------------------------

    def _debug_print(self, sim, odor, vision, terrain, nav, tag: str):
        if not self.DEBUG:
            return
        self._dbg_decision_count += 1
        if self._dbg_decision_count % self.DEBUG_EVERY != 0:
            return

        try:
            pos = sim.get_body_positions(sim.fly.name)[self._thorax_idx]
            xmat = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
            heading = float(np.degrees(np.arctan2(xmat[1, 0], xmat[0, 0])))
            uprightness = float(xmat[2, 2])
            banana_xy = getattr(sim.world, "banana_xy", (0.0, 0.0))
            dx, dy = banana_xy[0] - pos[0], banana_xy[1] - pos[1]
            dist = float(np.hypot(dx, dy))
            target_bearing = float(np.degrees(np.arctan2(dy, dx)))
            rel_bearing = ((target_bearing - heading + 180.0) % 360.0) - 180.0
            # number of legs with non-trivial contact force (for stuck detection)
            try:
                cf = sim.get_external_force(sim.fly.name, subtract_adhesion_force=True)
                cmag = np.linalg.norm(np.asarray(cf, dtype=float), axis=1) if cf.ndim == 2 else np.zeros(6)
                n_contact = int((cmag[:6] > self.CONTACT_THRESHOLD).sum())
            except Exception:
                n_contact = -1
        except Exception:
            pos = (0.0, 0.0, 0.0); heading = 0.0; uprightness = 1.0
            dist = 0.0; rel_bearing = 0.0; n_contact = -1

        t = float(sim.time) if hasattr(sim, "time") else 0.0

        parts = [f"t={t:6.2f}s"]
        parts.append(f"pos=({pos[0]:+6.2f},{pos[1]:+6.2f},z={pos[2]:+5.2f})")
        parts.append(f"hdg={heading:+7.1f} up={uprightness:+.2f} nc={n_contact}")
        parts.append(f"toBanana[d={dist:5.2f} bear={rel_bearing:+6.1f}]")

        if odor is not None:
            lin_mean, log_bias = odor
            parts.append(f"odor[mean={lin_mean:.2e} logB={log_bias:+.3f} smooth={self._smooth_odor:+.3f}]")
        if vision is not None:
            fr, lr, sL, sR, oL, oR = vision
            parts.append(
                f"vis[blk={fr:.3f} sL={sL:.3f} sR={sR:.3f} obsc=({oL:.2f},{oR:.2f}) lr={lr:+.3f}]"
            )
        if terrain is not None:
            dh, sm, uh = terrain
            parts.append(f"terr[dh={dh:.3f} uh={uh:.3f} slope={sm:.3f}]")
        if nav is not None:
            parts.append(
                f"nav[os={nav['odor_steer']:+.2f} vs={nav['vision_steer']:+.2f} "
                f"w={nav['w']:.2f} steer={nav['steer']:+.2f} spd={nav['speed_scale']:.2f} "
                f"drv=({nav['drives'][0]:.2f},{nav['drives'][1]:.2f})]"
            )
        if tag:
            parts.append(f"<{tag}>")
        print(" | ".join(parts))
