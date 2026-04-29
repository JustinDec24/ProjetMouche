"""Evaluate the DAgger vision policy on a batch of seeds.

Compares:
  - baseline  : stock controller (scripted vision module)          [optional]
  - policy    : controller with `dagger_policy_path=<checkpoint>`

Metrics per seed:
  - success       (did the fly reach within --success-radius of banana)
  - steps         (#physics steps used, capped at --max-steps)
  - min_dist      (minimum distance to banana over the trajectory)
  - final_dist    (distance at end of episode)
  - collisions    (proxy: #decision steps with horizontal contact force >
                   --collision-force-threshold)
  - mean_speed    (mm/s proxy from thorax displacement)

Usage
-----
python scripts/eval_dagger.py --level 2 \\
    --policy miniproject/dagger/models/policy_1.pt \\
    --seeds 0-19 --baseline
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

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


def _import_controller_cls():
    """Dynamically import the submission Controller class."""
    repo_root = Path(__file__).resolve().parents[1]
    submission_root = repo_root / "miniproject"
    if str(submission_root) not in sys.path:
        sys.path.insert(0, str(submission_root))
    return importlib.import_module("submission.controller").Controller


def run_episode(
    *,
    level: int,
    seed: int,
    policy_path: str | None,
    max_steps: int,
    success_radius: float,
    collision_thresh: float,
    decision_every: int,
) -> dict:
    sim = MiniprojectSimulation(level=level, seed=seed, back_cam=False, top_cam=False)
    Controller = _import_controller_cls()
    if policy_path is not None:
        controller = Controller(sim, dagger_policy_path=policy_path)
    else:
        controller = Controller(sim)

    thorax_idx = _thorax_index(sim)
    banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
    start_xy = _get_thorax_xy(sim, thorax_idx)
    start_dist = float(np.linalg.norm(start_xy - banana_xy))

    contact_ids = sim._internal_contact_body_segment_ids_by_fly[sim.fly.name]

    min_dist = float("inf")
    collisions = 0
    last_xy = start_xy.copy()
    total_displacement = 0.0
    t0 = time.time()

    for step in range(int(max_steps)):
        joint_angles, adhesion = controller.step(sim)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
        sim.step()

        # Decision-cadence bookkeeping.
        if (step % decision_every) == 0:
            xy = _get_thorax_xy(sim, thorax_idx)
            d = float(np.linalg.norm(xy - banana_xy))
            min_dist = min(min_dist, d)

            total_displacement += float(np.linalg.norm(xy - last_xy))
            last_xy = xy

            try:
                cf = sim.mj_data.cfrc_ext[contact_ids, :3]
                fmax = float(np.max(np.linalg.norm(cf[:, :2], axis=1)))
            except Exception:
                fmax = 0.0
            if fmax >= collision_thresh:
                collisions += 1

            if d <= success_radius:
                return {
                    "level": level,
                    "seed": seed,
                    "policy": policy_path,
                    "success": True,
                    "steps": step + 1,
                    "start_dist": start_dist,
                    "min_dist": min_dist,
                    "final_dist": d,
                    "collisions": collisions,
                    "mean_speed": total_displacement
                    / max(1, step + 1)
                    / sim.timestep,
                    "wall_s": time.time() - t0,
                }

    final_xy = _get_thorax_xy(sim, thorax_idx)
    final_dist = float(np.linalg.norm(final_xy - banana_xy))
    return {
        "level": level,
        "seed": seed,
        "policy": policy_path,
        "success": False,
        "steps": int(max_steps),
        "start_dist": start_dist,
        "min_dist": min_dist,
        "final_dist": final_dist,
        "collisions": collisions,
        "mean_speed": total_displacement / max(1, int(max_steps)) / sim.timestep,
        "wall_s": time.time() - t0,
    }


def parse_seeds(spec: str) -> list[int]:
    """Accept '0-9' or '0,2,5' or '0 2 5'."""
    out: list[int] = []
    for part in spec.replace(",", " ").split():
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def summarize(results: list[dict], label: str) -> None:
    if not results:
        print(f"[{label}] (no results)")
        return
    n = len(results)
    ok = sum(int(r["success"]) for r in results)
    mean_min = float(np.mean([r["min_dist"] for r in results]))
    mean_final = float(np.mean([r["final_dist"] for r in results]))
    mean_coll = float(np.mean([r["collisions"] for r in results]))
    mean_sp = float(np.mean([r["mean_speed"] for r in results]))
    print(
        f"[{label}] success={ok}/{n}  "
        f"mean_min_dist={mean_min:.2f}  mean_final_dist={mean_final:.2f}  "
        f"mean_collisions={mean_coll:.1f}  mean_speed={mean_sp:.3f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, default=2)
    p.add_argument("--policy", type=str, required=True)
    p.add_argument("--seeds", type=str, default="0-9")
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--success-radius", type=float, default=2.0)
    p.add_argument("--collision-force-threshold", type=float, default=4.0)
    p.add_argument("--decision-every", type=int, default=500)
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Also evaluate the scripted vision module (no policy) on the same seeds.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    print(f"Evaluating on seeds={seeds}  level={args.level}")

    policy_results: list[dict] = []
    baseline_results: list[dict] = []

    for seed in seeds:
        print(f"  - seed={seed}  [policy]", flush=True)
        r = run_episode(
            level=args.level,
            seed=seed,
            policy_path=args.policy,
            max_steps=args.max_steps,
            success_radius=args.success_radius,
            collision_thresh=args.collision_force_threshold,
            decision_every=args.decision_every,
        )
        policy_results.append(r)
        print(
            f"    success={r['success']}  min={r['min_dist']:.2f}  "
            f"final={r['final_dist']:.2f}  collisions={r['collisions']}"
        )

        if args.baseline:
            print(f"  - seed={seed}  [baseline]", flush=True)
            r0 = run_episode(
                level=args.level,
                seed=seed,
                policy_path=None,
                max_steps=args.max_steps,
                success_radius=args.success_radius,
                collision_thresh=args.collision_force_threshold,
                decision_every=args.decision_every,
            )
            baseline_results.append(r0)
            print(
                f"    success={r0['success']}  min={r0['min_dist']:.2f}  "
                f"final={r0['final_dist']:.2f}  collisions={r0['collisions']}"
            )

    print()
    summarize(policy_results, f"POLICY ({Path(args.policy).name})")
    if args.baseline:
        summarize(baseline_results, "BASELINE (scripted)")

    # Non-zero exit if policy failed on any seed.
    n_ok = sum(int(r["success"]) for r in policy_results)
    raise SystemExit(0 if n_ok == len(policy_results) else 2)


if __name__ == "__main__":
    main()
