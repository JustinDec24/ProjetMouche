from typing import Iterable

from flygym import Simulation, assets_dir
from flygym.anatomy import (
    ActuatedDOFPreset,
    AxisOrder,
    BodySegment,
    JointPreset,
    Skeleton,
)
from flygym.compose import ActuatorType, BaseWorld, Fly
from flygym.compose.pose import KinematicPose
from flygym.utils.math import Rotation3D, Vec3

LEG_NAMES: list[str] = [f"{side}{pos}" for side in "lr" for pos in "fmh"]


def create_fly(
    enable_vision: bool = False,
    enable_olfaction: bool = False,
    adhesion_gain: float = 50,
    position_gain: float = 50,
    adhesion_segments: Iterable[str] | None = tuple(
        f"{leg}_tarsus5" for leg in LEG_NAMES
    ),
    axis_order: AxisOrder = AxisOrder.YAW_PITCH_ROLL,
    joint_preset: JointPreset = JointPreset.LEGS_ONLY,
    dof_preset: ActuatedDOFPreset = ActuatedDOFPreset.LEGS_ACTIVE_ONLY,
    actuator_type: ActuatorType = ActuatorType.POSITION,
    neutral_pose_path=assets_dir / "model/pose/neutral.yaml",
    **kwargs,
):
    """Create a fly with the default week 5 tutorial configuration."""
    fly = Fly(**kwargs)

    skeleton = Skeleton(axis_order=axis_order, joint_preset=joint_preset)
    neutral_pose = KinematicPose(path=neutral_pose_path)
    fly.add_joints(skeleton, neutral_pose=neutral_pose)

    actuated_dofs = fly.skeleton.get_actuated_dofs_from_preset(dof_preset)
    fly.add_actuators(
        actuated_dofs,
        actuator_type=actuator_type,
        kp=position_gain,
        neutral_input=neutral_pose,
    )

    if enable_olfaction:
        fly.add_odor_sensors()

    if enable_vision:
        fly.add_vision()

    if adhesion_segments is not None:
        adhesion_segments = [
            seg if isinstance(seg, BodySegment) else BodySegment(seg)
            for seg in adhesion_segments
        ]
        fly.add_adhesion_actuators(segments=adhesion_segments, gain=adhesion_gain)

    fly.colorize()
    return fly


def create_simulation(
    fly: Fly,
    world: BaseWorld,
    spawn_position: Vec3 = (0, 0, 0.7),
    spawn_rotation: Rotation3D = Rotation3D("quat", (1, 0, 0, 0)),
    camera_kwargs: dict | None = None,
    playback_speed: float = 0.2,
    output_fps: int = 25,
    simulation_class=Simulation,
):
    """Create a simulation, camera, and renderer for a world and fly."""
    camera = world.add_camera(**(camera_kwargs or {}))
    world.add_fly(fly, spawn_position=spawn_position, spawn_rotation=spawn_rotation)
    world.add_light()
    sim = simulation_class(world)
    sim.set_renderer(
        camera=camera,
        playback_speed=playback_speed,
        output_fps=output_fps,
    )
    return sim


def show_video(sim: Simulation, title: str | None = None) -> None:
    """Display the first renderer's video inline in the notebook."""
    sim.renderer.show_in_notebook(title=title)
