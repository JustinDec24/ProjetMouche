"""Collect DAgger *correction* labels by rolling out a learned policy.

This is the proper DAgger iteration step:

  1. The fly is driven by the **learned vision policy**.
  2. At every decision step we ALSO query the **scripted expert**
     (`_vision_avoid_bias_and_danger`) on the *same state the policy reached*
     and log its output as a target label, **without letting the expert take
     control**.
  3. We aggregate these (state-from-policy, expert-action) pairs with the
     existing dataset and retrain. This corrects the distribution-shift
     problem that pure behaviour cloning suffers from: the policy is
     re-trained on states **it actually visits**, with labels coming from
     an oracle that knows what to do there.

Usage
-----
    python scripts/collect_dagger_correction.py \\
        --policy miniproject/dagger/models/policy_scripted_clean.pt \\
        --level 2 --seeds 5,7,8,9 --max-steps 150000 \\
        --out miniproject/dagger/data/demos_correction_0.npz

The output `.npz` is the same format consumed by
`miniproject.dagger.train_dagger`. Use it together with the previous
clean datasets:

    python -m miniproject.dagger.train_dagger \\
        --dataset miniproject/dagger/data/demos_scripted_0.npz \\
        --dataset miniproject/dagger/data/demos_scripted_1_clean.npz \\
        --dataset miniproject/dagger/data/demos_correction_0.npz \\
        --out    miniproject/dagger/models/policy_dagger_v2.pt \\
        --epochs 50 --hidden 64 --batch 128 --lr 1e-3 --val-frac 0.1

Notes
-----
* `--keep-only-success` default behaviour matches the scripted collector:
  episodes that fail to reach the banana are dropped to avoid poisoning the
  dataset with stuck-state labels. Pass `--keep-all` to bypass this.
* For DAgger to do its job, you should run on seeds where the *current
  policy* fails. A failed-by-policy seed is exactly the one where new
  labels add information.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np

from flygym.compose import ActuatorType

from miniproject import MiniprojectSimulation
from miniproject.dagger import DaggerDataset, VisionFeatureExtractor


def _import_controller_cls():
    repo_root = Path(__file__).resolve().parents[1]
    submission_root = repo_root / "miniproject"
    if str(submission_root) not in sys.path:
        sys.path.insert(0, str(submission_root))
    return importlib.import_module("submission.controller").Controller


def _thorax_index(sim: MiniprojectSimulation) -> int:
    segs = sim.fly.get_bodysegs_order()
    for i, s in enumerate(segs):
        if s.name == "c_thorax":
            return i
    return 0


def _get_thorax_xy(sim: MiniprojectSimulation, thorax_idx: int) -> np.ndarray:
    pos = sim.get_body_positions(sim.fly.name)[thorax_idx]
    return np.asarray(pos[:2], dtype=float)


def make_correction_controller_cls(BaseController):
    """Subclass that records scripted labels at policy-visited states.

    The parent class already wires `_dagger_vision_bias_and_speed` so the
    learned policy drives the fly when `dagger_policy_path` is provided.
    We override that method to:

        1. Extract features at the current state.
        2. Query the scripted expert via `_vision_avoid_bias_and_danger`.
           This computes (expert_turn, expert_danger) and as a side effect
           sets `self._vis_speed_scale` to the scripted speed -- we capture
           it then restore the previous value so the expert does NOT leak
           into the actual control output.
        3. Log (feat, expert_turn, expert_speed).
        4. Delegate to `super()._dagger_vision_bias_and_speed(sim)` so the
           policy is the one that actually drives.
    """

    class DaggerCorrectController(BaseController):
        def __init__(self, sim, policy_path: str) -> None:
            super().__init__(sim, dagger_policy_path=policy_path)
            if self._dagger_policy is None:
                raise RuntimeError(
                    f"Failed to load DAgger policy from {policy_path!r}. "
                    "Cannot do correction collection without a working policy."
                )
            if self._dagger_feat is None:
                self._dagger_feat = VisionFeatureExtractor(sim)
            self._records: list[tuple[np.ndarray, float, float]] = []

        def _dagger_vision_bias_and_speed(self, sim):
            feat = self._dagger_feat.extract(
                sim,
                prev_turn=self._dagger_prev_turn,
                prev_speed=self._dagger_prev_speed,
            )

            saved_vis_speed = self._vis_speed_scale
            expert_turn, _expert_danger = self._vision_avoid_bias_and_danger(sim)
            expert_speed = float(self._vis_speed_scale)
            self._vis_speed_scale = saved_vis_speed

            self._records.append(
                (
                    np.asarray(feat, dtype=np.float32).copy(),
                    float(expert_turn),
                    expert_speed,
                )
            )

            return super()._dagger_vision_bias_and_speed(sim)

        def pop_records(self) -> list[tuple[np.ndarray, float, float]]:
            r = self._records
            self._records = []
            return r

    return DaggerCorrectController


def parse_seeds(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def collect_one_episode(
    *,
    ControllerCls,
    policy_path: str,
    level: int,
    seed: int,
    max_steps: int,
    success_radius: float,
) -> tuple[list[tuple[np.ndarray, float, float]], dict]:
    sim = MiniprojectSimulation(level=level, seed=seed, back_cam=False, top_cam=False)
    controller = ControllerCls(sim, policy_path)

    thorax_idx = _thorax_index(sim)
    banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
    start_dist = float(np.linalg.norm(_get_thorax_xy(sim, thorax_idx) - banana_xy))

    min_dist = float("inf")
    success = False
    t0 = time.time()
    for step in range(int(max_steps)):
        joint_angles, adhesion = controller.step(sim)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
        sim.step()

        if step % 500 == 0:
            d = float(np.linalg.norm(_get_thorax_xy(sim, thorax_idx) - banana_xy))
            if d < min_dist:
                min_dist = d
            if d <= success_radius:
                success = True
                break

    final_dist = float(np.linalg.norm(_get_thorax_xy(sim, thorax_idx) - banana_xy))
    info = {
        "seed": seed,
        "steps": step + 1,
        "start_dist": start_dist,
        "min_dist": float(min(min_dist, final_dist)),
        "final_dist": final_dist,
        "success": success,
        "elapsed_s": time.time() - t0,
        "n_records": len(controller._records),
    }
    return controller.pop_records(), info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to the .pt VisionPolicy that drives the rollout.",
    )
    p.add_argument("--level", type=int, default=2)
    p.add_argument(
        "--seeds",
        type=str,
        default="5,7,8,9",
        help="Range '5-9' or comma-separated '5,7,8,9'. One episode per seed.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=150_000,
        help="Hard cap per episode (physics steps).",
    )
    p.add_argument(
        "--success-radius",
        type=float,
        default=2.0,
        help="Stop the episode early when within this distance of the banana.",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output .npz path (will be created).",
    )
    p.add_argument(
        "--keep-only-success",
        dest="keep_only_success",
        action="store_true",
        default=False,
        help="Discard records from episodes that did not succeed. By default "
        "we keep all episodes because failed-by-policy states are exactly "
        "what DAgger needs to correct.",
    )
    p.add_argument(
        "--keep-all",
        dest="keep_only_success",
        action="store_false",
        help="(default) Keep records from every episode, including failed ones.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    out_path = Path(args.out)
    policy_path = str(Path(args.policy).resolve())

    BaseController = _import_controller_cls()
    ControllerCls = make_correction_controller_cls(BaseController)

    dataset = DaggerDataset()
    print(
        f"DAgger correction rollout: policy={policy_path} "
        f"level={args.level} seeds={seeds} max_steps={args.max_steps} "
        f"keep_only_success={args.keep_only_success}",
        flush=True,
    )

    total_seen = 0
    total_kept = 0
    n_success = 0
    t0 = time.time()
    for episode_id, seed in enumerate(seeds):
        records, info = collect_one_episode(
            ControllerCls=ControllerCls,
            policy_path=policy_path,
            level=args.level,
            seed=seed,
            max_steps=args.max_steps,
            success_radius=args.success_radius,
        )
        total_seen += len(records)
        kept = 0
        if info["success"] or not args.keep_only_success:
            for feat, turn, speed in records:
                dataset.append(feat, (turn, speed), source=1, episode=episode_id)
            kept = len(records)
            total_kept += kept
        if info["success"]:
            n_success += 1
        status = "KEPT " if kept > 0 else "DROP "
        print(
            f"  {status}seed={info['seed']:3d}  steps={info['steps']:6d}  "
            f"start={info['start_dist']:5.1f}  min={info['min_dist']:5.1f}  "
            f"final={info['final_dist']:5.1f}  success={info['success']}  "
            f"records={info['n_records']:5d}  kept={kept:5d}  "
            f"elapsed={info['elapsed_s']:5.1f}s",
            flush=True,
        )

    if len(dataset) == 0:
        print(
            "\n[WARN] No records to save (no episode succeeded). "
            "Try --keep-all to also save labels from failed episodes.",
            flush=True,
        )
        return

    dataset.save(out_path)
    elapsed = time.time() - t0
    turns = dataset.labels[:, 0]
    speeds = dataset.labels[:, 1]
    print(
        f"\nSaved {len(dataset)} samples to {out_path} "
        f"(success={n_success}/{len(seeds)}, kept {total_kept}/{total_seen} records, "
        f"elapsed={elapsed:.1f}s)\n"
        f"  turn:  mean={turns.mean():+.3f}  std={turns.std():.3f}  "
        f"min={turns.min():+.3f}  max={turns.max():+.3f}\n"
        f"  speed: mean={speeds.mean():+.3f}  std={speeds.std():.3f}  "
        f"min={speeds.min():.3f}  max={speeds.max():.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
