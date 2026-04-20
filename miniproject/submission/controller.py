import numpy as np
from miniproject.simulation import MiniprojectSimulation


class Controller:
    """Hierarchical controller for levels 0–1 (plan architecture).

    Layer 0 — olfaction: bilateral log-concentration comparison for steering.
    Layer 1 — terrain: tilt-based speed, downhill lateral correction, adaptive adhesion (tilt + floating-leg boost via contact forces).
    Low level — CPG TurningController produces joint angles and stance/swing.
    """

    # --- olfaction (Level 0) ---
    ATTRACTIVE_GAIN = -500.0
    PALP_WEIGHT = 9.0
    ANTENNA_WEIGHT = 1.0
    STOP_THRESHOLD = 2e-5
    DECISION_INTERVAL_S = 0.05

    # --- terrain (Level 1) ---
    BIAS_SMOOTHING = 0.3
    TILT_BRAKE = 1.2
    MIN_DRIVE = 0.4
    MIN_SIDE_DRIVE = 0.2
    MAX_TURN_MODULATION = 0.8
    SLOPE_STEER_GAIN = 5.0

    # --- adhesion post-processing (Level 1) ---
    ADHESION_BOOST = 40.0
    CONTACT_THRESHOLD = 0.1
    FLOATING_LEG_BOOST = 3.0

    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        self._decision_every = max(1, int(self.DECISION_INTERVAL_S / sim.timestep))
        self._step_count = 0
        self._drives = np.array([1.0, 1.0], dtype=float)
        self._stopped = False
        self._smooth_bias = 0.0

        fly_segs = sim.fly.get_bodysegs_order()
        body_ids = sim._internal_bodyids_by_fly[sim.fly.name]
        self._thorax_body_id = body_ids[
            next(i for i, s in enumerate(fly_segs) if s.name == "c_thorax")
        ]

    def step(self, sim: MiniprojectSimulation):
        if self._step_count % self._decision_every == 0:
            self._drives = self._compute_drives(sim)
        self._step_count += 1

        joint_angles, adhesion = self.turning_controller.step(self._drives)

        _, _, uprightness = self._get_orientation(sim)
        tilt = max(0.0, 1.0 - uprightness)

        if tilt > 0.01:
            adhesion = adhesion * (1.0 + tilt * self.ADHESION_BOOST)

        forces = sim.get_external_force(sim.fly.name, subtract_adhesion_force=False)
        contact_mag = np.linalg.norm(forces, axis=1)
        for i in range(6):
            if adhesion[i] > 0 and contact_mag[i] < self.CONTACT_THRESHOLD:
                adhesion[i] *= self.FLOATING_LEG_BOOST

        return joint_angles, adhesion

    def _get_orientation(self, sim):
        """(pitch, roll_ind, uprightness) from thorax rotation matrix."""
        xmat = sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3)
        pitch = np.arcsin(np.clip(xmat[2, 0], -1.0, 1.0))
        return pitch, xmat[2, 1], xmat[2, 2]

    def _compute_drives(self, sim) -> np.ndarray:
        if self._stopped:
            return np.array([0.0, 0.0])

        log_olf = sim.get_olfaction(sim.fly.name, log=True)
        arr = log_olf[:, 0].reshape(2, 2)
        log_lr = np.average(
            arr,
            axis=0,
            weights=[self.PALP_WEIGHT, self.ANTENNA_WEIGHT],
        )

        lin_mean = np.exp(log_lr.mean())
        if lin_mean > self.STOP_THRESHOLD:
            self._stopped = True
            return np.array([0.0, 0.0])

        steering_bias = self.ATTRACTIVE_GAIN * (log_lr[0] - log_lr[1])

        pitch, roll_ind, uprightness = self._get_orientation(sim)
        tilt = max(0.0, 1.0 - uprightness)
        downhill = max(0.0, -pitch)
        slope_steer = -self.SLOPE_STEER_GAIN * roll_ind * downhill

        total_bias = steering_bias + slope_steer

        self._smooth_bias += (1.0 - self.BIAS_SMOOTHING) * (
            total_bias - self._smooth_bias
        )
        bias_norm = np.tanh(self._smooth_bias)

        base_drive = max(self.MIN_DRIVE, 1.0 - tilt * self.TILT_BRAKE)

        drives = np.full(2, base_drive, dtype=float)
        side = int(bias_norm > 0)
        drives[side] -= abs(bias_norm) * self.MAX_TURN_MODULATION * base_drive
        drives[side] = max(self.MIN_SIDE_DRIVE, drives[side])

        return drives
