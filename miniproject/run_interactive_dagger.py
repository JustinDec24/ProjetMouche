"""Interactive viewer with DAgger data collection / policy rollout.

This is a derivative of `miniproject/run_interactive.py` that keeps the
same ergonomics (pygame window, WASD/Q keyboard, hold/sticky modes) and
adds three operating modes:

    --mode expert   Classic human driving. Every decision step, the expert
                    command is logged as a supervised label. Use this for
                    phase-1 initial demonstrations.

    --mode policy   The loaded policy drives the fly. Keyboard is ignored
                    for motion (you can still quit / reset). No labels are
                    logged. Use this to inspect the current policy.

    --mode dagger   HG-DAgger: the policy acts at every decision step;
                    whenever the human presses a movement key, the human
                    takes over AND that human action is logged as the
                    supervised label for the observed state. Use this for
                    phase-3 corrective data collection.

Additional keys:
    SPACE   request reset (sim.reset() + extractor.reset())
    T       toggle recording on/off
    O       toggle policy override on/off (dagger mode only)
    ESC     quit

Typical commands (run as a script -- the file lives in `miniproject/`,
which is NOT a Python package, so `python -m ...` won't find it):

    python miniproject/run_interactive_dagger.py --level 2 --mode expert \\
        --out miniproject/dagger/data/demos_0.npz

    python miniproject/run_interactive_dagger.py --level 2 --mode dagger \\
        --policy miniproject/dagger/models/policy_0.pt \\
        --out    miniproject/dagger/data/demos_1.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pygame

from flygym.compose import ActuatorType

from miniproject.interactive import KeyboardControl, GameState
from miniproject import MiniprojectSimulation
from miniproject.dagger import (
    DaggerDataset,
    VisionFeatureExtractor,
)

# `submission/` lives next to this script (miniproject/submission/controller.py)
# but is NOT part of the installed `miniproject` package (which is `src/miniproject/`).
# `run_controller.ipynb` does `from submission.controller import Controller` because
# it runs from the miniproject/ directory; we replicate that here by adding the
# script's parent directory to sys.path before importing.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from submission.controller import Controller  # noqa: E402


WINDOW_NAME = "COBAR 2026 Miniproject - DAgger"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--mode",
        choices=["expert", "policy", "dagger"],
        default="expert",
        help="Operating mode. See module docstring for details.",
    )
    p.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to a .pt checkpoint (required for 'policy' and 'dagger').",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Path to save the recorded dataset (.npz). If omitted, nothing "
            "is saved. Ignored in 'policy' mode."
        ),
    )
    p.add_argument(
        "-m",
        "--keyboard-mode",
        choices=["hold", "sticky"],
        default="hold",
        help="Keyboard control mode (same as run_interactive.py).",
    )
    p.add_argument("-l", "--level", type=int, default=2)
    p.add_argument("-s", "--seed", type=int, default=0)
    p.add_argument(
        "--decision-every",
        type=int,
        default=500,
        help=(
            "Physics steps between decisions (extraction + policy / label). "
            "Default 500 matches the controller's 0.05 s cadence at "
            "timestep 1e-4."
        ),
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Hard cap on physics steps (0 = no cap).",
    )
    return p.parse_args()


class InteractiveController(Controller):
    """Submission `Controller` driven by keyboard for demo collection.

    The human's (turn, speed) is injected at the same point the eval-time
    DAgger policy is — i.e. inside `_dagger_vision_bias_and_speed`, treated
    by the rest of the controller as the "vision module" output. This way:

      * The fly inherits the full eval-time stability stack: drive smoothing,
        slope brake, anti-flip auto-reset, target-bearing steering, and the
        scripted safety reflexes (BUMP / LOOM / JAM).
      * Demo collection happens in **exactly** the same physics regime the
        learned policy will be deployed in, so the labelled data matches.
      * The keyboard semantics stay simple: `(gL, gR)` -> `(turn, speed)`
        through `label_from_gains`, and the controller layers target seeking
        and stability on top.

    The `--policy` checkpoint, if provided, is forwarded to the parent so
    `--mode policy` and `--mode dagger` keep using it when the human releases
    the keys.
    """

    def __init__(self, sim, dagger_policy_path: str | None = None) -> None:
        # We always need a feature extractor (for label logging and so the
        # parent's `_compute_drives` routes vision through our override). The
        # parent only creates one when a checkpoint path is provided, so we
        # forward the path AND make sure the extractor exists either way.
        super().__init__(sim, dagger_policy_path=dagger_policy_path)
        if self._dagger_feat is None:
            self._dagger_feat = VisionFeatureExtractor(sim)
        # Sentinel: keep `_dagger_policy` truthy so the parent's
        # `_compute_drives` calls our `_dagger_vision_bias_and_speed`. We
        # remember the actually-loaded policy (if any) separately so we can
        # use it when the human is inactive.
        self._loaded_policy = self._dagger_policy
        self._dagger_policy = self  # truthy sentinel; we override `.act`-using code below

        # Toggleable from the keyboard ('O' key in the UI).
        self._loaded_policy_enabled = self._loaded_policy is not None

        # Latched human command, refreshed at every decision step.
        self._human_active = False
        self._human_turn = 0.0
        self._human_speed = 1.0

    # --- public API used by the interactive loop ---------------------------------

    def set_human_command(self, turn: float, speed: float, active: bool) -> None:
        self._human_turn = float(turn)
        self._human_speed = float(speed)
        self._human_active = bool(active)

    def set_loaded_policy_enabled(self, enabled: bool) -> None:
        self._loaded_policy_enabled = bool(enabled) and (self._loaded_policy is not None)

    def has_loaded_policy(self) -> bool:
        return self._loaded_policy is not None

    def extract_features(self, sim) -> np.ndarray:
        """Compute the feature vector for label logging.

        Uses the same `prev_turn` / `prev_speed` the next vision-bias call
        will see, so the recorded feature vector matches what the policy
        observes at eval time.
        """
        return self._dagger_feat.extract(
            sim,
            prev_turn=self._dagger_prev_turn,
            prev_speed=self._dagger_prev_speed,
        )

    def reset_dagger_state(self) -> None:
        """Clear feature EMAs and turn-EMA after a `sim.reset()`."""
        self._dagger_feat.reset()
        self._dagger_turn_ema = 0.0
        self._dagger_prev_turn = 0.0
        self._dagger_prev_speed = 1.0
        self._human_active = False

    # --- override: when human is driving, bypass _compute_drives entirely ----

    def _compute_drives(self, sim):
        """When the human is driving, return their (gL, gR) DIRECTLY.

        The parent's `_compute_drives` does target steering, olfaction
        seeking, FSM, search casts, etc. -- great for autonomous driving but
        it would heavily filter / blend the keyboard input, leaving the user
        with almost no real control. For interactive demo collection we want
        full direct control, so when `_human_active=True` we short-circuit
        and just clip to the same terrain caps the parent would apply.

        Stability features still come from the rest of the pipeline:
          * the parent's `step()` runs the auto-reset logic on persistent
            tilt / flip,
          * the grip-control block at decision steps still applies,
          * adhesion override on slopes still applies,
          * the natural alternating-tripod gait (CPG) still drives the legs.

        When `_human_active=False`, we fall through to the parent so the
        loaded policy / scripted vision drive the fly normally (this is what
        `--mode policy` and the autopilot phase of `--mode dagger` use).
        """
        if not self._human_active:
            return super()._compute_drives(sim)

        # Map (turn, speed) -> (gL, gR).  Same convention as label_from_gains.
        gL = float(self._human_speed - self._human_turn)
        gR = float(self._human_speed + self._human_turn)

        # Terrain-aware drive cap (same constants the parent uses).
        max_drive = (
            self.MAX_DRIVE_TERRAIN if self._enable_terrain else self.MAX_DRIVE
        )
        if self._enable_terrain:
            try:
                _, _, slope_mag = self._get_slope_signals(sim)
            except Exception:
                slope_mag = 0.0
            if slope_mag > 0.15:
                # Progressive divider above the threshold; capped at /1.6.
                t = min(1.0, (slope_mag - 0.15) / 0.85)
                max_drive = max_drive / (1.0 + 0.6 * t)

        mag = max(abs(gL), abs(gR))
        if mag > max_drive and mag > 1e-6:
            ratio = max_drive / mag
            gL *= ratio
            gR *= ratio

        # Keep the EMA / prev fields in sync so when the user releases keys
        # the loaded policy resumes from a sensible turn-EMA state.
        self._dagger_prev_turn = self._human_turn
        self._dagger_prev_speed = self._human_speed
        self._dagger_turn_ema = self._human_turn

        return np.array([gL, gR], dtype=float)

    # --- override: vision module driven by loaded policy / scripted vision ----

    def _dagger_vision_bias_and_speed(self, sim):
        """Routes the vision module output for the autopilot path.

        Called by the parent's `_compute_drives` when the human is INACTIVE
        (otherwise we never reach this since `_compute_drives` short-circuits
        above).  Picks the loaded policy if enabled, else falls back to the
        scripted vision module.
        """
        if self._loaded_policy is not None and self._loaded_policy_enabled:
            feat = self._dagger_feat.extract(
                sim,
                prev_turn=self._dagger_prev_turn,
                prev_speed=self._dagger_prev_speed,
            )
            # Same safety-reflex bypass as the parent: hand back to scripted
            # logic on big contacts or fast looming.
            contact_fmax = float(feat[11])
            d_total_area = float(feat[4])
            if (
                contact_fmax >= float(self.DAGGER_BUMP_CONTACT_ON)
                or d_total_area >= float(self.DAGGER_LOOM_DAREA_ON)
            ):
                return self._vision_avoid_bias_and_danger(sim)

            turn_raw, speed_raw = self._loaded_policy.act(feat)
            self._dagger_prev_turn = float(turn_raw)
            self._dagger_prev_speed = float(speed_raw)
            self._dagger_turn_ema = (
                self.DAGGER_TURN_EMA * self._dagger_turn_ema
                + (1.0 - self.DAGGER_TURN_EMA) * float(turn_raw)
            )
            self._vis_speed_scale = float(np.clip(
                speed_raw, float(self.DAGGER_SPEED_MIN), 1.0
            ))
            return (
                float(np.clip(
                    self._dagger_turn_ema,
                    -self.VIS_TURN_MAX,
                    self.VIS_TURN_MAX,
                )),
                float(np.clip(float(feat[3]) + 0.8 * float(feat[2]), 0.0, 1.0)),
            )

        # Neither human nor enabled policy: fall back to scripted vision.
        return self._vision_avoid_bias_and_danger(sim)


def label_from_gains(gain_left: float, gain_right: float) -> tuple[float, float]:
    """Convert human (gain_left, gain_right) to (turn_label, speed_label).

    See dagger_dataset docstring for the conventions.
    """
    turn = 0.5 * (gain_right - gain_left)
    speed = float(np.clip(0.5 * (gain_left + gain_right), 0.0, 1.0))
    return float(turn), speed


def main() -> None:
    args = parse_args()
    if args.mode in ("policy", "dagger") and args.policy is None:
        raise SystemExit("--policy is required for 'policy' and 'dagger' modes.")

    # --- Simulation + InteractiveController (= submission Controller with
    #     keyboard input piped into the DAgger-policy injection point) ---
    sim = MiniprojectSimulation(level=args.level, seed=args.seed)
    controller = InteractiveController(sim, dagger_policy_path=args.policy)

    # --- Dataset for recording ---
    record_path = Path(args.out) if args.out else None
    dataset = DaggerDataset()
    episode_id = 0
    recording_enabled = record_path is not None and args.mode != "policy"
    # `policy_enabled` controls whether the loaded checkpoint is used when the
    # human is inactive. Toggleable via 'O'.
    policy_enabled = args.mode in ("policy", "dagger") and controller.has_loaded_policy()
    controller.set_loaded_policy_enabled(policy_enabled)

    # --- Pygame setup ---
    # Window shows backcam (left) + birdeyecam (right) side by side, so the
    # banana is visible from above while collecting demos.
    pygame.init()
    display_size = (1536, 768)
    screen = pygame.display.set_mode(display_size)
    pygame.display.set_caption(WINDOW_NAME + "  [left=backcam | right=top view]")
    game_state = GameState()
    controls = KeyboardControl(game_state, control_mode=args.keyboard_mode)

    step = 0
    decisions = 0
    human_labels = 0
    t_start = time.time()
    print(
        f"Mode={args.mode}  record={recording_enabled}  policy={policy_enabled}  "
        f"level={args.level}  seed={args.seed}",
        flush=True,
    )

    TOGGLE_KEYS = {pygame.K_t, pygame.K_o}

    while not game_state.get_quit():
        events = pygame.event.get()
        controls.process_events(events)
        # Handle extra keys (T / O) via direct event processing.
        for event in events:
            if event.type == pygame.KEYDOWN and event.key in TOGGLE_KEYS:
                if event.key == pygame.K_t and record_path is not None:
                    recording_enabled = not recording_enabled
                    print(f"[toggle] recording={recording_enabled}", flush=True)
                elif event.key == pygame.K_o and controller.has_loaded_policy():
                    policy_enabled = not policy_enabled
                    controller.set_loaded_policy_enabled(policy_enabled)
                    print(f"[toggle] policy_enabled={policy_enabled}", flush=True)

        # Handle reset request triggered by SPACE.
        if game_state.get_reset():
            try:
                sim.reset()
            except Exception as e:
                print(f"[warn] sim.reset() failed: {e}", flush=True)
            controller.reset_dagger_state()
            episode_id += 1
            game_state.set_reset(False)

        if game_state.get_quit():
            break

        keys_pressed = pygame.key.get_pressed()
        human_active = controls.any_key_pressed()
        gain_left_h, gain_right_h = controls.get_actions(keys_pressed)
        turn_h, speed_h = label_from_gains(gain_left_h, gain_right_h)

        # Decision step: choose who drives, and log the label if requested.
        is_decision = (step % max(1, args.decision_every)) == 0
        if is_decision:
            # Tell the controller who is steering this decision window.
            if args.mode == "expert":
                # Always human; no checkpoint usage.
                controller.set_human_command(turn_h, speed_h, active=True)
            elif args.mode == "policy":
                # Policy only (or scripted vision if disabled / no checkpoint).
                controller.set_human_command(0.0, 0.0, active=False)
            else:  # dagger
                # Human overrides policy when keys are pressed.
                if human_active:
                    controller.set_human_command(turn_h, speed_h, active=True)
                else:
                    controller.set_human_command(0.0, 0.0, active=False)

            # --- Label logging ---
            if recording_enabled:
                if args.mode == "expert":
                    feat = controller.extract_features(sim)
                    dataset.append(feat, (turn_h, speed_h), 0, episode_id)
                    human_labels += 1
                elif args.mode == "dagger" and human_active:
                    feat = controller.extract_features(sim)
                    dataset.append(feat, (turn_h, speed_h), 1, episode_id)
                    human_labels += 1

            decisions += 1
            if decisions % 20 == 0:
                print(
                    f"[t={time.time() - t_start:6.1f}s step={step:7d} "
                    f"dec={decisions:5d}] labels={human_labels:5d} "
                    f"dataset={len(dataset):6d} "
                    f"human=({turn_h:+.2f},{speed_h:.2f}) "
                    f"recording={recording_enabled} policy={policy_enabled}",
                    flush=True,
                )

        # --- Drive the fly through the full submission Controller pipeline.
        # This call internally handles target steering, slope brake, anti-flip
        # auto-reset, BUMP/LOOM/JAM reflexes, and hands the chosen vision-bias
        # source (human / loaded policy / scripted) to its `_compute_drives`.
        joint_angles, adhesion = controller.step(sim)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
        sim.step()

        # --- Render ---
        if sim.render_as_needed():
            try:
                frame = np.concatenate(
                    [frames[-1] for frames in sim.renderer.frames.values()], axis=-2
                )
                frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                if frame_surface.get_size() != display_size:
                    frame_surface = pygame.transform.smoothscale(
                        frame_surface, display_size
                    )
                screen.blit(frame_surface, (0, 0))
                pygame.display.flip()
            except Exception as e:
                print(f"[warn] render failed: {e}", flush=True)

        step += 1
        if args.max_steps > 0 and step >= args.max_steps:
            break

    controls.quit()
    pygame.quit()

    # --- Persist dataset ---
    if record_path is not None and len(dataset) > 0:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save(record_path)
        print(
            f"Saved {len(dataset)} labelled samples "
            f"(expert={int((dataset.source == 0).sum())}, "
            f"dagger={int((dataset.source == 1).sum())}) -> {record_path}",
            flush=True,
        )
    elif record_path is not None:
        print("No labelled samples collected; dataset not saved.", flush=True)


if __name__ == "__main__":
    main()
