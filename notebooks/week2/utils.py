"""Simulation utilities for the Week 2 CPG-controller notebooks.

This module provides:
- Shared constants (leg names, DoF templates)
- Helpers for loading and interpolating preprogrammed stepping data
- A convenience ``run_simulation`` function that builds a fly model,
  (optionally) replays joint-angle trajectories, and returns the
  :class:`Simulation` object.
"""

from __future__ import annotations

from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import trange

from flygym import Simulation, assets_dir
from flygym.anatomy import (
    ActuatedDOFPreset,
    AxisOrder,
    BodySegment,
    JointPreset,
    Skeleton,
)
from flygym.compose import ActuatorType, BaseWorld, FlatGroundWorld, Fly
from flygym.compose.pose import KinematicPose
from flygym.utils.math import Rotation3D


# ── Constants ────────────────────────────────────────────────────────────────

LEG_NAMES: list[str] = [f"{side}{pos}" for side in "lr" for pos in "fmh"]
"""Canonical leg ordering: lf, lm, lh, rf, rm, rh."""

DOFS_PER_LEG: list[str] = [
    "c_thorax-{leg}_coxa-yaw",
    "c_thorax-{leg}_coxa-pitch",
    "c_thorax-{leg}_coxa-roll",
    "{leg}_coxa-{leg}_trochanterfemur-pitch",
    "{leg}_coxa-{leg}_trochanterfemur-roll",
    "{leg}_trochanterfemur-{leg}_tibia-pitch",
    "{leg}_tibia-{leg}_tarsus1-pitch",
]
"""DoF name templates for each leg.  Call ``dof.format(leg=leg_name)``."""


# ── Preprogrammed stepping helpers ───────────────────────────────────────────


def load_preprogrammed_steps(path=None) -> dict:
    """Load single-step kinematics from an ``.npz`` file.

    Returns a dict with keys: ``dof_angles``, ``dof_names``, ``timestep``,
    ``swing_time``, ``stance_time``.
    """
    if path is None:
        path = assets_dir / "demo" / "single_steps_untethered.npz"
    data = np.load(path, allow_pickle=True)
    return {
        "dof_angles": data["dof_angles"],
        "dof_names": [str(s) for s in data["dof_order"]],
        "timestep": data["timestep"].item(),
        "swing_time": data["swing_time"].item(),
        "stance_time": data["stance_time"].item(),
    }


def build_step_interpolators(
    dof_angles: np.ndarray,
    dof_names: list[str],
) -> dict[str, CubicSpline]:
    """Build per-leg cubic-spline interpolators mapping phase → joint angles.

    Parameters
    ----------
    dof_angles : (n_frames, n_dofs) array
        Joint angles for one complete stepping cycle.
    dof_names : list of str
        Column names in *dof_angles*.

    Returns
    -------
    dict mapping leg name → ``CubicSpline`` that accepts phase ∈ [0, 2π].
    """
    n_frames = dof_angles.shape[0]
    phase_grid = np.linspace(0, 2 * np.pi, n_frames)
    interpolators = {}
    for leg in LEG_NAMES:
        col_idx = [dof_names.index(dof.format(leg=leg)) for dof in DOFS_PER_LEG]
        interpolators[leg] = CubicSpline(
            x=phase_grid,
            y=dof_angles[:, col_idx],
            axis=0,
            bc_type="periodic",
        )
    return interpolators


def compute_swing_stance_phases(
    swing_time: dict,
    stance_time: dict,
    n_frames: int,
    data_timestep: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert swing/stance start times (seconds) to phases (radians).

    Returns
    -------
    swing_start_phase, swing_end_phase, rest_phases : each (6,) array.
        *rest_phases* is the midpoint between end-of-swing and the next
        swing onset — a safe neutral phase for amplitude modulation.
    """
    step_duration = n_frames * data_timestep
    swing_start = np.array([swing_time[leg] for leg in LEG_NAMES])
    swing_end = np.array([stance_time[leg] for leg in LEG_NAMES])
    swing_start_phase = swing_start / step_duration * (2 * np.pi)
    swing_end_phase = swing_end / step_duration * (2 * np.pi)
    rest_phases = (swing_end_phase + 2 * np.pi) / 2
    return swing_start_phase, swing_end_phase, rest_phases


def get_adhesion_onoff(
    theta: np.ndarray,
    swing_start_phase: np.ndarray,
    swing_end_phase: np.ndarray,
) -> np.ndarray:
    """Return a boolean adhesion signal: True during stance, False during swing."""
    theta = theta % (2 * np.pi)
    return ~((theta > swing_start_phase) & (theta < swing_end_phase)).squeeze()


# ── Visualisation helpers ────────────────────────────────────────────────────


def show_video(sim: Simulation, title: str | None = None) -> None:
    """Display the first renderer's video inline in the notebook."""
    sim.renderer.show_in_notebook(title=title)


def plot_gait_diagram(
    timestep: float,
    leg_ap_positions: np.ndarray,
    title: str,
) -> None:
    """Detect swing/stance phases from leg A-P positions and plot a gait diagram.

    Parameters
    ----------
    timestep : float
        Physics timestep (seconds).
    leg_ap_positions : (n_steps, 6) array
        Anterior–posterior position of each leg tip relative to the body.
    title : str
        Title for the figure (gait name).
    """
    leg_labels = [name.upper() for name in LEG_NAMES]
    fig, axs = plt.subplots(
        6, 1, figsize=(7, 2), sharex=True, gridspec_kw={"hspace": 0}
    )
    patch = None
    for i, (ax, y) in enumerate(zip(axs, leg_ap_positions.T)):
        vel = gaussian_filter1d(y, sigma=50, order=1)
        _, prop = find_peaks(vel, distance=600, width=100)
        intervals = np.column_stack([prop["left_ips"], prop["right_ips"]])
        for interval in intervals * timestep:
            patch = ax.axvspan(*interval, color="k", lw=0, label="Swing")
        t = np.arange(len(y)) * timestep
        lines = ax.plot(t, y, color="r", label="Leg position (higher = more anterior)")
        ax.set_ylabel(leg_labels[i], rotation=0, ha="right", va="center")
        ax.set_xlim(0, 1)
        ax.set_yticks([])

    if patch is not None:
        handles = [Patch(fc="w", ec="k", label="Stance"), patch, lines[0]]
        axs[0].legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1),
            ncols=3,
            frameon=False,
        )
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{title.capitalize()} gait", y=1.15)


# ── Simulation ───────────────────────────────────────────────────────────────


def run_simulation(
    dof_angles: np.ndarray | None = None,
    dof_names: Iterable[str] | None = None,
    adhesion_segments: Iterable[str] | None = None,
    adhesion_signals: np.ndarray | None = None,
    adhesion_gain: float = 50,
    position_gain: float = 50,
    warmup_steps: int = 0,
    spawn_position: tuple = (0, 0, 0.7),
    spawn_rotation: Rotation3D = Rotation3D("quat", (1, 0, 0, 0)),
    axis_order: AxisOrder = AxisOrder.YAW_PITCH_ROLL,
    joint_preset: JointPreset = JointPreset.LEGS_ONLY,
    dof_preset: ActuatedDOFPreset = ActuatedDOFPreset.LEGS_ACTIVE_ONLY,
    actuator_type: ActuatorType = ActuatorType.POSITION,
    neutral_pose_path=assets_dir / "model/pose/neutral.yaml",
    playback_speed: float = 0.2,
    output_fps: int = 25,
    world: BaseWorld | None = None,
    camera_kwargs: dict | None = None,
    step_callback: Callable[[Simulation], object] | None = None,
) -> Simulation | tuple[Simulation, list]:
    """Build a fly simulation, optionally replay DoF angles, and return it.

    If *dof_angles* is ``None`` the simulation is created but **no** physics
    steps are executed (useful for launching the interactive viewer).

    Parameters
    ----------
    dof_angles : (n_steps, n_dofs) array, optional
        Joint-angle trajectory in radians.
    dof_names : list of str, optional
        DoF names matching columns of *dof_angles* (used for reordering).
    adhesion_segments : list of str, optional
        Body segments to attach adhesion actuators to.
    adhesion_signals : (n_steps, n_legs) array, optional
        Per-step binary adhesion on/off signals.
    adhesion_gain : float
        Gain of the adhesion actuators.
    position_gain : float
        Proportional gain (kp) of the position actuators.
    warmup_steps : int
        Physics steps to let the fly settle before replay begins.
    world : BaseWorld, optional
        Custom world (defaults to :class:`FlatGroundWorld`).
    camera_kwargs : dict, optional
        Extra keyword arguments forwarded to ``fly.add_tracking_camera()``.
    step_callback : callable, optional
        ``callback(sim)`` called after every physics step.  Return values
        are collected into a list that is returned alongside the simulation.

    Returns
    -------
    sim : Simulation
        If *dof_angles* is ``None``.
    (sim, callback_data) : tuple
        If *dof_angles* is provided.
    """
    # ── Build the fly ────────────────────────────────────────────────────
    fly = Fly()

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

    if adhesion_segments is not None:
        adhesion_segments = [
            seg if isinstance(seg, BodySegment) else BodySegment(seg)
            for seg in adhesion_segments
        ]
        fly.add_adhesion_actuators(segments=adhesion_segments, gain=adhesion_gain)

    fly.colorize()

    tracking_cam = fly.add_tracking_camera(**(camera_kwargs or {}))

    # ── Build the world & simulation ─────────────────────────────────────
    if world is None:
        world = FlatGroundWorld()
    world.add_fly(fly, spawn_position, spawn_rotation)

    sim = Simulation(world)
    sim.set_renderer(
        camera=tracking_cam,
        playback_speed=playback_speed,
        output_fps=output_fps,
    )

    if dof_angles is None:
        return sim

    # ── Reorder columns to match simulation actuator order ───────────────
    if dof_names is None:
        dof_angles_sorted = dof_angles
    else:
        sim_dof_order = [
            dof.name for dof in fly.get_actuated_jointdofs_order(actuator_type)
        ]
        dof_names = [str(dof) for dof in dof_names]
        reorder_idx = [dof_names.index(dof) for dof in sim_dof_order]
        dof_angles_sorted = dof_angles[:, reorder_idx]

    n_sim_steps = len(dof_angles)
    callback_data: list = []

    # ── Physics loop ─────────────────────────────────────────────────────
    for step in trange(warmup_steps + n_sim_steps):
        if step >= warmup_steps:
            t = step - warmup_steps
            sim.set_actuator_inputs(fly.name, actuator_type, dof_angles_sorted[t])
            if adhesion_signals is not None:
                sim.set_actuator_inputs(
                    fly.name, ActuatorType.ADHESION, adhesion_signals[t]
                )
        sim.step()
        if step_callback is not None:
            callback_data.append(step_callback(sim))
        sim.render_as_needed()

    return sim, callback_data


def get_control_signals(
    step_data: dict,
    phase_biases: np.ndarray,
    intrinsic_freqs: np.ndarray,
    intrinsic_amps: np.ndarray,
    convergence_coefs: np.ndarray,
    coupling_weights: np.ndarray,
    timestep: float = 1e-4,
    run_time: float = 1.0,
    warmup_time: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate CPG-driven joint-angle and adhesion trajectories.

    Parameters
    ----------
    step_data : dict
        Output of :func:`load_preprogrammed_steps`.
    phase_biases, intrinsic_freqs, intrinsic_amps, convergence_coefs,
    coupling_weights
        CPG network parameters (see :class:`cpg_network.CPGNetwork`).
    timestep : float
        Integration timestep (seconds).
    run_time : float
        Duration of the output trajectory (seconds).
    warmup_time : float
        CPG warm-up time before recording (seconds).

    Returns
    -------
    joint_angles : (n_steps, n_total_dofs) array
    adhesion_signals : (n_steps, 6) array
    dof_order : list of str
    """
    from cpg_network import CPGNetwork

    psi_funcs = build_step_interpolators(
        step_data["dof_angles"], step_data["dof_names"]
    )
    swing_start_phase, swing_end_phase, rest_phases = compute_swing_stance_phases(
        step_data["swing_time"],
        step_data["stance_time"],
        step_data["dof_angles"].shape[0],
        step_data["timestep"],
    )

    cpg_network = CPGNetwork(
        timestep=timestep,
        intrinsic_freqs=intrinsic_freqs,
        intrinsic_amps=intrinsic_amps,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=convergence_coefs,
    )

    # Warm up the CPG so oscillators synchronise before recording
    for _ in range(int(warmup_time / timestep)):
        cpg_network.step()

    n_steps = int(run_time / timestep)
    n_legs = len(LEG_NAMES)
    n_dofs = len(DOFS_PER_LEG)
    joint_angles = np.zeros((n_steps, n_legs, n_dofs))
    adhesion_signals = np.zeros((n_steps, n_legs))

    for i in range(n_steps):
        cpg_network.step()
        for j, leg in enumerate(LEG_NAMES):
            psi = psi_funcs[leg](cpg_network.curr_phases[j])
            psi_rest = psi_funcs[leg](rest_phases[j])
            joint_angles[i, j] = (
                psi_rest + (psi - psi_rest) * cpg_network.curr_magnitudes[j]
            )
        adhesion_signals[i] = get_adhesion_onoff(
            cpg_network.curr_phases, swing_start_phase, swing_end_phase
        )

    dof_order = [dof.format(leg=leg) for leg in LEG_NAMES for dof in DOFS_PER_LEG]
    return joint_angles.reshape((n_steps, -1)), adhesion_signals, dof_order
