"""Collect DAgger labels by cloning the scripted expert.

The submission `Controller` already contains a hand-crafted vision module
(`_vision_avoid_bias_and_danger`). Instead of asking a human to drive the
fly with the keyboard, we let the scripted controller pilot the simulation
autonomously and we record, at every decision step that goes through the
vision pipeline, the tuple

        (features, scripted_turn, scripted_speed_scale)

These tuples are exactly the (input, target) pairs the `VisionPolicy` MLP
must learn to reproduce: train on them and the policy becomes a learned
drop-in replacement for `_vision_avoid_bias_and_danger`.

Usage
-----
    python scripts/collect_scripted_expert.py \\
        --level 2 --seeds 0-9 --max-steps 12000 \\
        --out miniproject/dagger/data/demos_scripted_0.npz

The output `.npz` is the same format consumed by
`miniproject.dagger.train_dagger`, so afterwards:

    python -m miniproject.dagger.train_dagger \\
        --dataset miniproject/dagger/data/demos_scripted_0.npz \\
        --out    miniproject/dagger/models/policy_scripted_0.pt \\
        --epochs 50 --hidden 64 --batch 128 --lr 1e-3 --val-frac 0.1
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
    """Dynamically import the submission `Controller` class.

    `miniproject/submission/controller.py` lives next to the project's
    `miniproject/` directory, but `submission` is not part of the
    installed package (`src/miniproject/`). Replicate the trick used by
    `scripts/eval_dagger.py`.
    """
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


def make_expert_controller_cls(BaseController):
    """Return a Controller subclass that records scripted vision labels.

    The trick: set `_dagger_policy = self` (truthy sentinel) so the parent's
    `_compute_drives` routes the vision query through
    `_dagger_vision_bias_and_speed`, which we override to:

      1. Extract the same features the learned policy would observe at
         eval time.
      2. Call the scripted `_vision_avoid_bias_and_danger(sim)` to get
         the (turn, danger) the expert produces. As a side effect this
         sets `self._vis_speed_scale` to the scripted speed.
      3. Buffer (features, scripted_turn, vis_speed_scale) for later
         export.
      4. Update `_dagger_prev_*` so the next feature extraction sees a
         consistent autoregressive history.

    The fly is genuinely driven by the scripted controller (no policy
    influence), so the trajectory we record is the expert trajectory.
    """

    class ScriptedExpertController(BaseController):
        def __init__(self, sim) -> None:
            super().__init__(sim)
            if self._dagger_feat is None:
                self._dagger_feat = VisionFeatureExtractor(sim)
            self._dagger_policy = self
            self._records: list[tuple[np.ndarray, float, float]] = []

        def _dagger_vision_bias_and_speed(self, sim):
            feat = self._dagger_feat.extract(
                sim,
                prev_turn=self._dagger_prev_turn,
                prev_speed=self._dagger_prev_speed,
            )
            turn, danger = self._vision_avoid_bias_and_danger(sim)
            speed = float(self._vis_speed_scale)

            self._records.append((np.asarray(feat, dtype=np.float32).copy(), float(turn), speed))

            self._dagger_prev_turn = float(turn)
            self._dagger_prev_speed = speed
            self._dagger_turn_ema = (
                self.DAGGER_TURN_EMA * self._dagger_turn_ema
                + (1.0 - self.DAGGER_TURN_EMA) * float(turn)
            )
            return float(turn), float(danger)

        def pop_records(self) -> list[tuple[np.ndarray, float, float]]:
            r = self._records
            self._records = []
            return r

    return ScriptedExpertController


def parse_seeds(spec: str) -> list[int]:
    """Parse '0-9' or '0,1,2' or '7' into a list[int]."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def collect_one_episode(
    *,
    ExpertCls,
    level: int,
    seed: int,
    max_steps: int,
    success_radius: float,
) -> tuple[list[tuple[np.ndarray, float, float]], dict]:
    sim = MiniprojectSimulation(level=level, seed=seed, back_cam=False, top_cam=False)
    controller = ExpertCls(sim)

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
    p.add_argument("--level", type=int, default=2)
    p.add_argument(
        "--seeds",
        type=str,
        default="0-9",
        help="Range '0-9' or comma-separated '0,1,2'. One episode per seed.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=12_000,
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
        default=True,
        help="(default) Discard records from episodes that did NOT reach the "
        "banana. Avoids polluting the dataset with stuck-state labels.",
    )
    p.add_argument(
        "--keep-all",
        dest="keep_only_success",
        action="store_false",
        help="Keep records from every episode, even failed ones.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    out_path = Path(args.out)

    BaseController = _import_controller_cls()
    ExpertCls = make_expert_controller_cls(BaseController)

    dataset = DaggerDataset()
    print(
        f"Collecting scripted-expert demos: level={args.level} "
        f"seeds={seeds} max_steps={args.max_steps} "
        f"keep_only_success={args.keep_only_success}",
        flush=True,
    )

    total_seen = 0
    total_kept = 0
    n_success = 0
    t0 = time.time()
    for episode_id, seed in enumerate(seeds):
        records, info = collect_one_episode(
            ExpertCls=ExpertCls,
            level=args.level,
            seed=seed,
            max_steps=args.max_steps,
            success_radius=args.success_radius,
        )
        total_seen += len(records)
        kept = 0
        if info["success"] or not args.keep_only_success:
            for feat, turn, speed in records:
                dataset.append(feat, (turn, speed), source=0, episode=episode_id)
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
            "Try a different seed range, raise --max-steps, or rerun with --keep-all.",
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
