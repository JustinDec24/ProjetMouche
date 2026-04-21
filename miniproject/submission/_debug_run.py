"""Quick debug run for the controller.

Usage: uv run python -m miniproject.submission._debug_run --level 2 --seed 5 --steps 15000 --print-every 20
"""
from __future__ import annotations

import argparse
import os
import sys

# Make 'submission.*' importable no matter where this script is launched from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MINIPROJECT_DIR = os.path.dirname(_THIS_DIR)
if _MINIPROJECT_DIR not in sys.path:
    sys.path.insert(0, _MINIPROJECT_DIR)

import numpy as np
from flygym.compose import ActuatorType

from miniproject.simulation import MiniprojectSimulation
from submission.controller import Controller


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--print-every", type=int, default=20,
                    help="Print every N decisions (1 decision = 500 steps by default).")
    args = ap.parse_args()

    print(f"[cfg] level={args.level} seed={args.seed} steps={args.steps} print_every={args.print_every}")

    sim = MiniprojectSimulation(
        level=args.level, seed=args.seed, back_cam=False, top_cam=False
    )

    banana_xy = getattr(sim.world, "banana_xy", None)
    if banana_xy is not None:
        print(f"[cfg] banana_xy = ({banana_xy[0]:+.2f}, {banana_xy[1]:+.2f})")

    # Enable controller debug prints
    Controller.DEBUG = True
    Controller.DEBUG_EVERY = args.print_every
    controller = Controller(sim)

    # Track travelled distance
    fly_segs = sim.fly.get_bodysegs_order()
    thorax_idx = next(i for i, s in enumerate(fly_segs) if s.name == "c_thorax")
    last_pos = sim.get_body_positions(sim.fly.name)[thorax_idx, :2].copy()
    total_dist = 0.0
    max_stuck_steps = 0
    stuck_counter = 0

    start_pos = last_pos.copy()

    for i in range(args.steps):
        joint_angles, adhesion = controller.step(sim)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
        sim.step()

        if i % 200 == 0:
            pos = sim.get_body_positions(sim.fly.name)[thorax_idx, :2]
            d = float(np.linalg.norm(pos - last_pos))
            total_dist += d
            # "stuck" = progress below 0.01 mm in 200 steps (0.02 s of sim time)
            if d < 0.01:
                stuck_counter += 1
                max_stuck_steps = max(max_stuck_steps, stuck_counter)
            else:
                stuck_counter = 0
            last_pos = pos.copy()

    end_pos = sim.get_body_positions(sim.fly.name)[thorax_idx, :2]
    straight = float(np.linalg.norm(end_pos - start_pos))
    remaining = float(np.linalg.norm(np.asarray(banana_xy) - end_pos)) if banana_xy is not None else -1
    print(
        f"[stats] travelled={total_dist:.2f}mm  start_to_end={straight:.2f}mm  "
        f"remaining_to_banana={remaining:.2f}mm  max_stuck_windows={max_stuck_steps}"
    )


if __name__ == "__main__":
    main()
