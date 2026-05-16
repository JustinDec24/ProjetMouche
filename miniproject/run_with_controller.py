"""Run the submission Controller autonomously.

Examples:

    uv run python miniproject/run_with_controller.py \
        --level 2 --seed 6 \
        --display-backend cv2 \
        --debug-vision \
        --save-video videos/L2_seed6_debug.mp4 \
        --progress-every 5000

    uv run python miniproject/run_with_controller.py \
        --level 2 --seed 6 \
        --display-backend none \
        --progress-every 5000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from flygym.compose import ActuatorType
from miniproject import MiniprojectSimulation

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from submission.controller import Controller  # noqa: E402


WINDOW_NAME = "COBAR 2026 Miniproject - controller viewer"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the miniproject submission controller.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-l", "--level", type=int, default=4)
    p.add_argument("-s", "--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=100_000)

    # IMPORTANT:
    # The official success criterion is radius = 3 around the banana / odor source.
    # Before, this was 2.0, so runs with min_dist around 2.5 were incorrectly
    # marked as failures even though they should count as successful.
    p.add_argument("--success-radius", type=float, default=3.0)

    p.add_argument(
        "--display-backend",
        choices=("pygame", "cv2", "none"),
        default="pygame",
        help="Display backend. Use cv2 if pygame gives a black screen.",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Shortcut for --display-backend none.",
    )
    p.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Optional path to save an mp4 video.",
    )

    p.add_argument(
        "--render-fly-vision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stack raw fly vision above the scene cameras.",
    )
    p.add_argument(
        "--debug-vision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stack the controller vision debug overlay above the scene cameras.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Print progress every N physics steps. Use 0 to disable.",
    )
    return p.parse_args()


def _get_fly_xy(sim: MiniprojectSimulation, controller: Controller) -> np.ndarray:
    try:
        return np.asarray(
            sim.get_body_positions(sim.fly.name)[controller._thorax_idx, :2],
            dtype=float,
        )
    except Exception:
        return np.asarray(
            sim.mj_data.xpos[controller._thorax_body_id, :2],
            dtype=float,
        )


def _get_upright(sim: MiniprojectSimulation, controller: Controller) -> float:
    try:
        return float(sim.mj_data.xmat[controller._thorax_body_id].reshape(3, 3)[2, 2])
    except Exception:
        return 1.0


def _get_banana_xy(sim: MiniprojectSimulation, controller: Controller) -> np.ndarray | None:
    try:
        return np.asarray(sim.world.banana_xy, dtype=float)
    except Exception:
        try:
            return np.asarray(controller._banana_xy, dtype=float)
        except Exception:
            return None


def _make_top_strip(
    sim: MiniprojectSimulation,
    controller: Controller,
    args: argparse.Namespace,
) -> np.ndarray | None:
    if args.debug_vision:
        try:
            return controller.compute_vision_debug_overlay(sim)
        except Exception as e:
            print(f"[warn] debug vision overlay failed: {e}", flush=True)
            return None

    if args.render_fly_vision:
        try:
            return np.concatenate(sim.get_raw_vision(sim.fly.name), axis=-2)
        except Exception as e:
            print(f"[warn] raw fly vision failed: {e}", flush=True)
            return None

    return None


def _compose_frame(
    sim: MiniprojectSimulation,
    controller: Controller,
    args: argparse.Namespace,
) -> np.ndarray | None:
    try:
        frame = np.concatenate(
            [frames[-1] for frames in sim.renderer.frames.values()],
            axis=-2,
        )

        top_strip = _make_top_strip(sim, controller, args)
        if top_strip is not None:
            if top_strip.dtype != frame.dtype:
                top_strip = top_strip.astype(frame.dtype)

            pad_total = frame.shape[1] - top_strip.shape[1]
            if pad_total >= 0:
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                top_strip = np.pad(
                    top_strip,
                    ([0, 0], [pad_left, pad_right], [0, 0]),
                    mode="constant",
                )
            else:
                top_strip = top_strip[:, : frame.shape[1], :]

            frame = np.vstack((top_strip, frame))

        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        return frame
    except Exception as e:
        print(f"[warn] compose frame failed: {e}", flush=True)
        return None


def main() -> None:
    args = parse_args()

    if args.no_display:
        args.display_backend = "none"

    sim = MiniprojectSimulation(level=args.level, seed=args.seed)
    controller = Controller(sim)

    banana_xy = _get_banana_xy(sim, controller)
    min_dist = float("inf")
    final_dist = float("inf")

    success = False
    success_step = None

    show_pygame = args.display_backend == "pygame"
    show_cv2 = args.display_backend == "cv2"
    need_render = show_pygame or show_cv2 or args.save_video is not None

    pygame = None
    screen = None
    display_size = None

    cv2 = None
    video_writer = None

    if show_pygame:
        import pygame as _pygame

        pygame = _pygame
        pygame.init()
        display_size = (
            1536,
            1536 if (args.render_fly_vision or args.debug_vision) else 768,
        )
        screen = pygame.display.set_mode(display_size)
        pygame.display.set_caption(f"{WINDOW_NAME} level={args.level} seed={args.seed}")

    if show_cv2 or args.save_video is not None:
        import cv2 as _cv2

        cv2 = _cv2

    if args.save_video is not None:
        Path(args.save_video).parent.mkdir(parents=True, exist_ok=True)

    print(
        f"level={args.level} seed={args.seed} max_steps={args.max_steps} "
        f"success_radius={args.success_radius:.2f} "
        f"backend={args.display_backend} save_video={args.save_video}",
        flush=True,
    )

    t_start = time.time()
    quit_requested = False
    steps_done = 0

    for step in range(args.max_steps):
        steps_done = step + 1
        success_reached_this_step = False

        if show_pygame:
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

        xy = _get_fly_xy(sim, controller)
        if banana_xy is not None:
            final_dist = float(np.linalg.norm(xy - banana_xy))
            min_dist = min(min_dist, final_dist)

            if final_dist <= args.success_radius:
                success = True
                success_step = steps_done
                success_reached_this_step = True

        if need_render and sim.render_as_needed():
            frame = _compose_frame(sim, controller, args)
            if frame is not None:
                if args.save_video is not None:
                    h, w = frame.shape[:2]
                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            args.save_video,
                            fourcc,
                            30.0,
                            (w, h),
                        )
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                if show_cv2:
                    cv2.imshow(WINDOW_NAME, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):
                        quit_requested = True

                if show_pygame:
                    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    if surf.get_size() != display_size:
                        surf = pygame.transform.smoothscale(surf, display_size)
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()

        if success_reached_this_step:
            print(
                f"success reached at step={success_step} "
                f"dist={final_dist:.2f} min_dist={min_dist:.2f}",
                flush=True,
            )
            break

        if quit_requested:
            break

        if args.progress_every > 0 and steps_done % args.progress_every == 0:
            elapsed = time.time() - t_start
            sps = steps_done / max(elapsed, 1e-6)
            upright = _get_upright(sim, controller)
            dist_str = f"{final_dist:.2f}" if np.isfinite(final_dist) else "?"
            min_str = f"{min_dist:.2f}" if np.isfinite(min_dist) else "?"
            print(
                f"step {steps_done:7d}/{args.max_steps} "
                f"dist={dist_str} min={min_str} upright={upright:.2f} "
                f"drives=({controller._drives[0]:.2f},{controller._drives[1]:.2f}) "
                f"speed={sps:.0f} steps/s",
                flush=True,
            )

    if video_writer is not None:
        video_writer.release()

    if show_cv2 and cv2 is not None:
        cv2.destroyAllWindows()

    if show_pygame and pygame is not None:
        pygame.quit()

    # Keep this as a backup: even if the loop was interrupted right after entering
    # the goal radius, the run should still be reported as success.
    success = bool(success or (np.isfinite(min_dist) and min_dist <= args.success_radius))

    print(
        f"done. success={success} min_dist={min_dist:.2f} "
        f"final_dist={final_dist:.2f} steps={steps_done} "
        f"wall_s={time.time() - t_start:.1f}",
        flush=True,
    )

    if args.save_video is not None:
        print(f"saved video: {args.save_video}", flush=True)


if __name__ == "__main__":
    main()
