"""Batch runner for Level 2 seeds 10-19."""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "miniproject"))
from submission.controller import Controller
from miniproject import MiniprojectSimulation
from flygym.compose import ActuatorType
import numpy as np

_rng = np.random.RandomState(42)
SEEDS = sorted(int(s) for s in _rng.choice(range(0, 1000), size=30, replace=False))
LEVEL = 2
MAX_STEPS = 200_000
PROGRESS_EVERY = 5_000
SUCCESS_DIST = 2.0

results = []

for seed in SEEDS:
    print(f"Running level {LEVEL} seed={seed} ...", flush=True)
    sim = MiniprojectSimulation(level=LEVEL, seed=seed)
    ctrl = Controller(sim)
    t0 = time.time()

    # get start dist
    try:
        thorax_idx = ctrl._thorax_idx
        thorax_xy = sim.get_body_positions(sim.fly.name)[thorax_idx, :2]
        banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
        start_dist = float(np.linalg.norm(thorax_xy - banana_xy))
    except Exception:
        start_dist = float("nan")

    min_dist = start_dist
    final_dist = start_dist

    for step in range(MAX_STEPS):
        joint_angles, adhesion = ctrl.step(sim)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
        sim.step()

        try:
            thorax_xy = sim.get_body_positions(sim.fly.name)[thorax_idx, :2]
            dist = float(np.linalg.norm(thorax_xy - banana_xy))
        except Exception:
            dist = float("nan")

        if np.isfinite(dist):
            min_dist = min(min_dist, dist)
            final_dist = dist

        if PROGRESS_EVERY > 0 and (step + 1) % PROGRESS_EVERY == 0:
            wall = time.time() - t0
            speed_sim = (step + 1) * sim.timestep
            upright = "?"
            try:
                import mujoco as mj
                xmat = sim.mj_data.xmat[ctrl._thorax_body_id].reshape(3, 3)
                upright = f"{float(xmat[2,2]):.2f}"
            except Exception:
                pass
            print(
                f"  [seed={seed:3d} step={step+1:7d}] dist={dist:.2f} min={min_dist:.2f}"
                f" speed={speed_sim/wall:.2f} mean_odor={0:.3e}"
                f" drives=({ctrl._drives[0]:.2f},{ctrl._drives[1]:.2f})"
                f" stopped={ctrl._stopped} upright={upright}"
                f" wall_s={wall:.1f}",
                flush=True,
            )

        # Only break early on success (dist <= SUCCESS_DIST).
        # Do NOT break on ctrl._stopped — the real 20s test runs the full duration
        # (a flipped fly may slide toward the banana and still pass).
        if np.isfinite(dist) and dist <= SUCCESS_DIST:
            break

    wall_s = time.time() - t0
    sim_s = (step + 1) * sim.timestep
    ok = np.isfinite(min_dist) and min_dist <= SUCCESS_DIST
    status = "OK" if ok else "FAIL"
    line = (
        f"[L2 seed={seed:3d}] {status}"
        f"  start_dist={start_dist:.2f}"
        f"  min_dist={min_dist:.2f}"
        f"  final_dist={final_dist:.2f}"
        f"  sim_s={sim_s:.1f}"
        f"  wall_s={wall_s:.1f}"
    )
    print(line, flush=True)
    results.append((seed, ok, min_dist, final_dist))

n_ok = sum(r[1] for r in results)
print(f"Summary: {n_ok}/{len(results)} succeeded", flush=True)
print("Worst failures (up to 10):", flush=True)
failures = [(s, md, fd) for s, ok, md, fd in results if not ok]
for s, md, fd in sorted(failures, key=lambda x: x[1], reverse=True)[:10]:
    print(f"  seed={s:3d}  min_dist={md:.2f}  final_dist={fd:.2f}", flush=True)
