import dm_control.mjcf as mjcf
import numpy as np
from tqdm import trange

from flygym.compose import ActuatorType
from flygym.compose.world import FlatGroundWorld
from flygym.examples.locomotion import TurningController
from flygym.simulation import Simulation
from utils import create_fly, create_simulation


def crop_hex_to_rect(visual_input, ommatidia_id_map):
    """Extract a rectangular crop from the hexagonal ommatidium layout."""
    rows = [np.unique(row) for row in ommatidia_id_map]
    max_width = max(len(row) for row in rows)
    rows = np.array([row for row in rows if len(row) == max_width])[:, 1:] - 1
    cols = [np.unique(col) for col in rows.T]
    min_height = min(len(col) for col in cols)
    cols = [col[:min_height] for col in cols]
    rows = np.array(cols).T
    return visual_input[..., rows]


def get_heading_angle(body) -> float:
    """Extract the fly heading angle from a MuJoCo body quaternion."""
    w, x, y, z = body.xquat
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class TargetMixin:
    """Mixin that adds a trackable spherical target to any world."""

    mjcf_root: mjcf.RootElement

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_names = []

    def add_target(
        self,
        pos=(0, 0, 1),
        size=(1, 1),
        rgba=(0, 0, 0, 1),
        name=None,
        **kwargs,
    ):
        """Add a spherical target to the arena.

        Parameters
        ----------
        pos : tuple
            Initial (x, y, z) position.
        size : tuple
            Size parameters for the geom.
        rgba : tuple
            Color of the object (RGBA, each in [0, 1]).
        name : str, optional
            Unique name for the object.
        """
        if name is None:
            name = f"object_{len(self.object_names)}"
        assert name not in self.object_names, f"Object with name {name} already exists"
        ball_body = self.mjcf_root.worldbody.add(
            "body",
            name=f"{name}_mocap",
            mocap=True,
            pos=pos,
            gravcomp=1,
        )
        ball_body.add(
            "geom",
            name=f"{name}_geom",
            type="sphere",
            size=size,
            rgba=rgba,
            **kwargs,
        )
        self.object_names.append(name)

    def set_target_positions(self, sim: Simulation, positions):
        """Update the positions of all targets."""
        mocap_ids = np.array(
            [
                sim.mj_model.body(f"{name}_mocap").mocapid.item()
                for name in self.object_names
            ]
        )
        sim.mj_data.mocap_pos[mocap_ids, :2] = np.atleast_2d(positions)


class TargetWorld(TargetMixin, FlatGroundWorld):
    pass


N_SAMPLES = 10_000
WARMUP_STEPS = 500
CONTROL_SIGNAL = np.array([1.0, 1.0])

world = TargetWorld()
world.add_target()
fly = create_fly(enable_vision=True)
sim = create_simulation(fly, world)
controller = TurningController(sim.timestep)

rng = np.random.default_rng(seed=0)
r_list = rng.uniform(2, 10, N_SAMPLES)
theta_list = rng.uniform(-np.pi, np.pi, N_SAMPLES)

for _ in range(WARMUP_STEPS):
    sim.step()

images = None
for i in trange(N_SAMPLES):
    joint_angles, adhesion = controller.step(CONTROL_SIGNAL)
    sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, joint_angles)
    sim.set_actuator_inputs(fly.name, ActuatorType.ADHESION, adhesion)
    body = sim.mj_data.body(f"{fly.name}/")
    theta = theta_list[i] + get_heading_angle(body)
    pos_fly = body.xpos[:2]
    target_pos = pos_fly + r_list[i] * np.array([np.cos(theta), np.sin(theta)])
    world.set_target_positions(sim, target_pos)
    sim.step()
    sim.get_raw_vision.cache_clear()
    sim.get_ommatidia_readouts.cache_clear()
    ommatidia_readouts = sim.get_ommatidia_readouts(fly.name).max(-1)
    image = crop_hex_to_rect(ommatidia_readouts, fly.retina.ommatidia_id_map)
    if images is None:
        images = np.zeros((N_SAMPLES, *image.shape), dtype=np.float32)
    images[i] = image

np.savez_compressed(
    "assets/data.npz",
    images=images,
    r_list=r_list,
    theta_list=theta_list,
)
