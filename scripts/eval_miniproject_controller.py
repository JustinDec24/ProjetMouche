import argparse
import importlib
import math
import sys
from pathlib import Path
import time

import numpy as np

from flygym.compose import ActuatorType
from miniproject.simulation import MiniprojectSimulation


def _thorax_index(sim: MiniprojectSimulation) -> int:
    segs = sim.fly.get_bodysegs_order()
    for i, s in enumerate(segs):
        if s.name == "c_thorax":
            return i
    return 0


def _get_thorax_xy(sim: MiniprojectSimulation, thorax_idx: int) -> np.ndarray:
    pos = sim.get_body_positions(sim.fly.name)[thorax_idx]
    return np.asarray(pos[:2], dtype=float)


def run_episode(
    *,
    level: int,
    seed: int,
    max_steps: int,
    success_radius: float,
    render_every: int,
    progress_every: int,
) -> dict:
    sim = MiniprojectSimulation(level=level, seed=seed, back_cam=False, top_cam=False)

    # Import here so it always loads the user's submission version.
    repo_root = Path(__file__).resolve().parents[1]
    submission_root = repo_root / "miniproject"
    if str(submission_root) not in sys.path:
        sys.path.insert(0, str(submission_root))
    Controller = importlib.import_module("submission.controller").Controller

    controller = Controller(sim)
    thorax_idx = _thorax_index(sim)
    banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
    start_dist = float(np.linalg.norm(_get_thorax_xy(sim, thorax_idx) - banana_xy))

    n_steps = int(max_steps)
    min_dist = float("inf")
    stopped_steps = 0

    t0 = time.time()
    last_xy = _get_thorax_xy(sim, thorax_idx)
    for step in range(n_steps):
        joint_angles, adhesion = controller.step(sim)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
        sim.step()

        if render_every > 0 and step % render_every == 0:
            sim.render_as_needed()

        xy = _get_thorax_xy(sim, thorax_idx)
        d = float(np.linalg.norm(xy - banana_xy))
        min_dist = min(min_dist, d)

        if progress_every > 0 and (step % progress_every == 0) and step > 0:
            odor = sim.get_olfaction(sim.fly.name)[:, 0]
            mean_odor = float(
                0.5
                * (
                    9.0 * float(odor[0])
                    + 1.0 * float(odor[2])
                    + 9.0 * float(odor[1])
                    + 1.0 * float(odor[3])
                )
            )
            upright = None
            slope_mag = None
            if level >= 1:
                # thorax body id
                segs = sim.fly.get_bodysegs_order()
                thorax_i = 0
                for i, s in enumerate(segs):
                    if s.name == "c_thorax":
                        thorax_i = i
                        break
                body_id = sim._internal_bodyids_by_fly[sim.fly.name][thorax_i]
                xmat = sim.mj_data.xmat[body_id].reshape(3, 3)
                upright = float(xmat[2, 2])
                try:
                    n = np.asarray(sim.world.get_normal(float(xy[0]), float(xy[1])), dtype=float)
                    if n.shape == (3,) and np.isfinite(n).all() and abs(n[2]) > 1e-6:
                        grad = np.array([-n[0] / n[2], -n[1] / n[2]], dtype=float)
                        slope_mag = float(np.linalg.norm(grad))
                except Exception:
                    slope_mag = None

            dt = time.time() - t0
            speed = float(np.linalg.norm(xy - last_xy) / (progress_every * sim.timestep))
            last_xy = xy
            extra = ""
            if upright is not None:
                extra += f" upright={upright:.2f}"
            if slope_mag is not None:
                extra += f" slope={slope_mag:.2f}"
            print(
                f"  [seed={seed:3d} step={step:7d}] dist={d:.2f} min={min_dist:.2f} "
                f"speed={speed:.2f} mean_odor={mean_odor:.3e} "
                f"drives=({controller._drives[0]:.2f},{controller._drives[1]:.2f}) "
                f"stopped={bool(getattr(controller, '_stopped', False))}{extra} wall_s={dt:.1f}",
                flush=True,
            )

        if np.allclose(joint_angles, 0.0, atol=1e-12) and np.allclose(
            adhesion, 0.0, atol=1e-12
        ):
            stopped_steps += 1

        if d <= success_radius:
            return {
                "level": level,
                "seed": seed,
                "success": True,
                "steps": step + 1,
                "seconds": (step + 1) * sim.timestep,
                "start_dist": start_dist,
                "min_dist": min_dist,
                "final_dist": d,
                "wall_s": time.time() - t0,
                "stopped_steps": stopped_steps,
            }

    final_dist = float(np.linalg.norm(_get_thorax_xy(sim, thorax_idx) - banana_xy))
    return {
        "level": level,
        "seed": seed,
        "success": False,
        "steps": n_steps,
        "seconds": n_steps * sim.timestep,
        "start_dist": start_dist,
        "min_dist": min_dist,
        "final_dist": final_dist,
        "wall_s": time.time() - t0,
        "stopped_steps": stopped_steps,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seed-count", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--success-radius", type=float, default=2.0)
    p.add_argument("--progress-every", type=int, default=50_000)
    p.add_argument(
        "--render-every",
        type=int,
        default=0,
        help="Render every N steps (0 disables rendering).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    level = args.level
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))

    results = []
    ok = 0
    for seed in seeds:
        print(f"Running level {level} seed={seed} ...", flush=True)
        r = run_episode(
            level=level,
            seed=seed,
            max_steps=args.max_steps,
            success_radius=args.success_radius,
            render_every=args.render_every,
            progress_every=args.progress_every,
        )
        results.append(r)
        ok += int(r["success"])
        status = "OK" if r["success"] else "FAIL"
        print(
            f"[L{level} seed={seed:3d}] {status}  "
            f"start_dist={r['start_dist']:.2f}  min_dist={r['min_dist']:.2f}  final_dist={r['final_dist']:.2f}  "
            f"sim_s={r['seconds']:.1f}  wall_s={r['wall_s']:.1f}"
        )

    fails = [r for r in results if not r["success"]]
    print()
    print(f"Summary: {ok}/{len(results)} succeeded")
    if fails:
        worst = sorted(fails, key=lambda x: (-x["min_dist"], -x["final_dist"]))[:10]
        print("Worst failures (up to 10):")
        for r in worst:
            print(
                f"  seed={r['seed']:3d}  min_dist={r['min_dist']:.2f}  "
                f"final_dist={r['final_dist']:.2f}"
            )
    print()
    # Return non-zero on failure for easier automation.
    raise SystemExit(0 if ok == len(results) else 2)


if __name__ == "__main__":
    main()
