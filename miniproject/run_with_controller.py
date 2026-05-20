"""Run the submission `Controller` autonomously and watch it live.

This is the script-equivalent of `run_controller.ipynb`, but with a live
pygame window (back-camera + birds-eye + optional fly vision overlay)
and CLI flags for the most useful runtime options.

Typical commands (run from the repo root):

    # Level 4 (terrain + grass + wind + dragonfly), with raw fly vision
    # rendered on top of the scene cameras.
    .\\.venv\\Scripts\\python.exe miniproject/run_with_controller.py \\
        --level 4 --seed 42 --render-fly-vision

    # Same but show vision debug overlay (bright green = spike pixels inside
    # SKY–blade–SKY segments, yellow = apex row, white = horizon estimate,
    # red = dragonfly, cyan = ROI).
    .\\.venv\\Scripts\\python.exe miniproject/run_with_controller.py \\
        --level 4 --seed 42 --debug-vision

    # No live window, just step through 100k physics steps as fast as
    # possible and print progress every 10k steps.
    .\\.venv\\Scripts\\python.exe miniproject/run_with_controller.py \\
        --level 2 --seed 5 --no-display --max-steps 100000

Keys (when display is enabled):
    SPACE  call sim.reset()
    ESC    quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pygame

from flygym.compose import ActuatorType

from miniproject import MiniprojectSimulation

# `submission/` lives next to this script (miniproject/submission/controller.py)
# but is NOT part of the installed `miniproject` package (which is `src/miniproject/`).
# `run_controller.ipynb` does `from submission.controller import Controller` because
# it runs from the miniproject/ directory; we replicate that here by adding the
# script's parent directory to sys.path before importing.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from submission.controller import Controller  # noqa: E402


WINDOW_NAME = "COBAR 2026 Miniproject - controller viewer"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-l", "--level", type=int, default=4,
        help="Simulation level (0=flat, 1=+terrain, 2=+grass, 3=+wind, 4=+dragonfly).",
    )
    p.add_argument("-s", "--seed", type=int, default=42)
    p.add_argument(
        "--max-steps", type=int, default=100_000,
        help="Hard cap on physics steps (default 100k = ~10 s of sim).",
    )
    p.add_argument(
        "--no-display", action="store_true",
        help="Headless run: do not open a pygame window. Useful for benchmarking.",
    )
    p.add_argument(
        "--render-fly-vision", action=argparse.BooleanOptionalAction,
        help=(
            "Stack raw RGB fly vision (left + right eye) above the scene "
            "cameras (same idea as miniproject/run_interactive.py)."
        ),
    )
    p.add_argument(
        "--debug-vision", action=argparse.BooleanOptionalAction,
        help=(
            "Stack a colour-mask debug view of the controller's perception "
            "above the scene cameras (green=grass, red=dragonfly, blue=edges, "
            "cyan=ROI). Takes precedence over --render-fly-vision."
        ),
    )
    p.add_argument(
        "--progress-every", type=int, default=10_000,
        help="Print a progress line every N physics steps (0 disables).",
    )
    return p.parse_args()


def _make_top_strip(
    sim: MiniprojectSimulation,
    controller: Controller,
    args: argparse.Namespace,
) -> np.ndarray | None:
    """Compose the upper banner shown above the scene cameras."""
    if args.debug_vision:
        return controller.compute_vision_debug_overlay(sim)
    if args.render_fly_vision:
        try:
            return np.concatenate(sim.get_raw_vision(sim.fly.name), axis=-2)
        except Exception:
            return None
    return None


def main() -> None:
    args = parse_args()

    sim = MiniprojectSimulation(level=args.level, seed=args.seed)
    controller = Controller(sim)

    show_window = not args.no_display
    show_top_strip = bool(args.render_fly_vision or args.debug_vision)
    screen = None
    display_size: tuple[int, int] | None = None

    if show_window:
        pygame.init()
        display_size = (2200, 2200 if show_top_strip else 1100)
        screen = pygame.display.set_mode(display_size)
        cap = WINDOW_NAME + f"  level={args.level} seed={args.seed}"
        if args.debug_vision:
            cap += "  | top = debug overlay (green=grass, red=dragonfly)"
        elif args.render_fly_vision:
            cap += "  | top = raw fly vision"
        pygame.display.set_caption(cap)

    print(
        f"level={args.level} seed={args.seed} max_steps={args.max_steps} "
        f"display={show_window} fly_vision={bool(args.render_fly_vision)} "
        f"debug_vision={bool(args.debug_vision)}",
        flush=True,
    )

    t_start = time.time()
    quit_requested = False
    for step in range(args.max_steps):
        if show_window:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        quit_requested = True
                    elif event.key == pygame.K_SPACE:
                        try:
                            sim.reset()
                            print("[info] sim.reset()", flush=True)
                        except Exception as e:
                            print(f"[warn] sim.reset() failed: {e}", flush=True)
            if quit_requested:
                break

        joint_angles, adhesion = controller.step(sim)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
        sim.step()

        if show_window and sim.render_as_needed():
            try:
                frame = np.concatenate(
                    [frames[-1] for frames in sim.renderer.frames.values()],
                    axis=-2,
                )
                top_strip = _make_top_strip(sim, controller, args)
                if top_strip is not None:
                    pad_total = frame.shape[1] - top_strip.shape[1]
                    pad_left = max(0, pad_total // 2)
                    pad_right = max(0, pad_total - pad_left)
                    top_strip = np.pad(
                        top_strip,
                        ([0, 0], [pad_left, pad_right], [0, 0]),
                    )
                    frame = np.vstack((top_strip, frame))
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                assert display_size is not None
                if surf.get_size() != display_size:
                    surf = pygame.transform.smoothscale(surf, display_size)
                assert screen is not None
                screen.blit(surf, (0, 0))
                pygame.display.flip()
            except Exception as e:
                print(f"[warn] render failed: {e}", flush=True)

        if args.progress_every > 0 and (step + 1) % args.progress_every == 0:
            elapsed = time.time() - t_start
            sps = (step + 1) / max(elapsed, 1e-6)
            print(
                f"step {step + 1:7d}/{args.max_steps}  "
                f"elapsed={elapsed:6.1f}s  speed={sps:6.0f} steps/s  "
                f"drives=({controller._drives[0]:.2f},{controller._drives[1]:.2f})",
                flush=True,
            )

    if show_window:
        pygame.quit()
    print(
        f"done. total_steps={step + 1} wall_s={time.time() - t_start:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
