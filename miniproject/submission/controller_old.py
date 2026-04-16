import numpy as np
from miniproject.simulation import MiniprojectSimulation


class Controller:
    """Hierarchical controller for levels 0–1: food-seeking with terrain awareness.

    High level: bilateral odor comparison steers toward the banana.
    Low level: CPG-based TurningController converts asymmetric descending
    drives into coordinated leg movements.
    Terrain layer (Level 1): body orientation is used to brake on downhill
    slopes and add corrective steering via the body roll signal,
    preventing the fly from racing downhill.
    """

    # --- tunable parameters (all in one place) ---
    ATTRACTIVE_GAIN = -30
    PALP_WEIGHT = 9
    ANTENNA_WEIGHT = 1
    MAX_TURN_MODULATION = 0.8
    DECISION_INTERVAL_S = 0.05  # seconds between control updates
    STOP_ODOR_THRESHOLD = 2e-5  # ~3-4 mm from source with default arena params

    # --- terrain-awareness parameters ---
    BIAS_SMOOTHING = 0.3      # EMA factor for steering (higher = smoother)
    DOWNHILL_BRAKE = 1.5      # drive reduction per radian of downhill pitch
    MIN_DRIVE = 0.4           # floor on base forward drive
    SLOPE_STEER_GAIN = 8.0    # corrective steering toward uphill side (downhill only)

    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        # you may also implement your own turning controller
        # you may also implement your own turning controller
        # you may also implement your own turning controller
        # you may also implement your own turning controller
        # you may also implement your own turning controller
        # you may also implement your own turning controller
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        self._decision_every = int(self.DECISION_INTERVAL_S / sim.timestep)
        self._step_count = 0
        self._drives = np.array([1.0, 1.0])
        self._stopped = False

        # Thorax body index for orientation queries
        fly_segs = sim.fly.get_bodysegs_order()
        self._thorax_idx = next(
            (i for i, s in enumerate(fly_segs) if s.name == "c_thorax"), 0
        )

        self._smooth_bias = 0.0  # EMA state for steering signal

    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)
        # get other observations as needed
        drives = np.array([1.0, 1.0])  # replace with your control logic
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
