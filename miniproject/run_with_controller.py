"""Run the submission Controller autonomously, optionally display it, and save videos.

Use --display-backend cv2 on Linux if pygame opens a black window.
Use --display-backend none together with --save-video for a fully headless recording.
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
        description="Run the miniproject controller, optionally save a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-l", "--level", type=int, default=4)
    p.add_argument("-s", "--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--success-radius", type=float, default=3.0)
    p.add_argument(
        "--display-backend",
        choices=("cv2", "pygame", "none"),
        default="cv2",
        help="Live display backend. Use cv2 on Linux Mint if pygame is black.",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Shortcut for --display-backend none.",
    )
    p.add_argument(
        "--dont-use-pygame-rendering",
        action="store_true",
        help="Backward-compatible shortcut for --display-backend cv2.",
    )
    p.add_argument(
        "--render-fly-vision",
        action=argparse.BooleanOptionalAction,
        help="Stack raw left/right fly vision above the scene cameras.",
    )
    p.add_argument(
        "--debug-vision",
        action=argparse.BooleanOptionalAction,
        help="Stack controller perception overlay above the scene cameras.",
    )
    p.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Path to an output video, e.g. videos/L2_seed2.mp4.",
    )
    p.add_argument("--video-fps", type=float, default=30.0)
    p.add_argument(
        "--video-codec",
        type=str,
        default="mp4v",
        help="FourCC codec for cv2.VideoWriter. mp4v is the safest default.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Print progress every N physics steps. 0 disables progress lines.",
    )
    return p.parse_args()


def _thorax_index(sim: MiniprojectSimulation) -> int:
    for i, seg in enumerate(sim.fly.get_bodysegs_order()):
        if seg.name == "c_thorax":
            return i
    return 0


def _thorax_xy(sim: MiniprojectSimulation, thorax_idx: int) -> np.ndarray:
    return np.asarray(sim.get_body_positions(sim.fly.name)[thorax_idx][:2], dtype=float)


def _distance_to_goal(sim: MiniprojectSimulation, thorax_idx: int) -> float:
    banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
    return float(np.linalg.norm(_thorax_xy(sim, thorax_idx) - banana_xy))


def _uprightness(sim: MiniprojectSimulation, thorax_idx: int) -> float | None:
    try:
        body_id = sim._internal_bodyids_by_fly[sim.fly.name][thorax_idx]
        xmat = sim.mj_data.xmat[body_id].reshape(3, 3)
        return float(xmat[2, 2])
    except Exception:
        return None


def _as_uint8_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _fit_width(strip: np.ndarray, width: int) -> np.ndarray:
    strip = _as_uint8_rgb(strip)
    if strip.shape[1] == width:
        return strip
    if strip.shape[1] < width:
        pad_total = width - strip.shape[1]
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(strip, ([0, 0], [pad_left, pad_right], [0, 0]))
    crop_total = strip.shape[1] - width
    crop_left = crop_total // 2
    return strip[:, crop_left : crop_left + width, :]


def _make_top_strip(
    sim: MiniprojectSimulation,
    controller: Controller,
    args: argparse.Namespace,
) -> np.ndarray | None:
    if args.debug_vision:
        try:
            return controller.compute_vision_debug_overlay(sim)
        except Exception:
            return None
    if args.render_fly_vision:
        try:
            return np.concatenate(sim.get_raw_vision(sim.fly.name), axis=-2)
        except Exception:
            return None
    return None


def _compose_frame(
    sim: MiniprojectSimulation,
    controller: Controller,
    args: argparse.Namespace,
) -> np.ndarray | None:
    scene_frames = [frames[-1] for frames in sim.renderer.frames.values() if len(frames)]
    if not scene_frames:
        return None
    frame = np.concatenate(scene_frames, axis=-2)
    frame = _as_uint8_rgb(frame)

    top_strip = _make_top_strip(sim, controller, args)
    if top_strip is not None:
        top_strip = _fit_width(top_strip, frame.shape[1])
        frame = np.vstack((top_strip, frame))
    return np.ascontiguousarray(frame)


class VideoSink:
    def __init__(self, path: Path | None, fps: float, codec: str):
        self.path = path
        self.fps = float(fps)
        self.codec = str(codec)
        self.writer = None

    def write(self, frame_rgb: np.ndarray) -> None:
        if self.path is None:
            return
        import cv2

        frame_rgb = _as_uint8_rgb(frame_rgb)
        h, w = frame_rgb.shape[:2]
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self.codec[:4])
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {self.path}")
        self.writer.write(frame_rgb[..., ::-1])

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def main() -> None:
    args = parse_args()
    if args.no_display:
        args.display_backend = "none"
    if args.dont_use_pygame_rendering:
        args.display_backend = "cv2"

    sim = MiniprojectSimulation(level=args.level, seed=args.seed)
    controller = Controller(sim)
    thorax_idx = _thorax_index(sim)

    video = VideoSink(args.save_video, args.video_fps, args.video_codec)
    display_size: tuple[int, int] | None = None
    screen = None

    if args.display_backend == "pygame":
        import pygame

        pygame.init()
        display_size = (1536, 1536 if (args.render_fly_vision or args.debug_vision) else 768)
        screen = pygame.display.set_mode(display_size)
        pygame.display.set_caption(f"{WINDOW_NAME} | level={args.level} seed={args.seed}")
    elif args.display_backend == "cv2":
        import cv2

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1200, 1200 if (args.render_fly_vision or args.debug_vision) else 700)

    print(
        f"level={args.level} seed={args.seed} max_steps={args.max_steps} "
        f"backend={args.display_backend} save_video={args.save_video}",
        flush=True,
    )

    t_start = time.time()
    success = False
    final_dist = _distance_to_goal(sim, thorax_idx)
    min_dist = final_dist

    try:
        for step in range(args.max_steps):
            final_dist = _distance_to_goal(sim, thorax_idx)
            min_dist = min(min_dist, final_dist)
            if final_dist <= args.success_radius:
                success = True
                print(f"Got to goal in {step} timesteps. dist={final_dist:.2f}", flush=True)
                break

            if args.display_backend == "pygame":
                import pygame

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
            elif args.display_backend == "cv2":
                import cv2

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    raise KeyboardInterrupt

            joint_angles, adhesion = controller.step(sim)
            sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
            sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
            sim.step()

            need_frame = args.display_backend != "none" or args.save_video is not None
            if need_frame and sim.render_as_needed():
                frame = _compose_frame(sim, controller, args)
                if frame is not None:
                    video.write(frame)
                    if args.display_backend == "pygame":
                        import pygame

                        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                        assert display_size is not None
                        if surf.get_size() != display_size:
                            surf = pygame.transform.smoothscale(surf, display_size)
                        assert screen is not None
                        screen.blit(surf, (0, 0))
                        pygame.display.flip()
                    elif args.display_backend == "cv2":
                        import cv2

                        cv2.imshow(WINDOW_NAME, frame[..., ::-1])

            if args.progress_every > 0 and (step + 1) % args.progress_every == 0:
                elapsed = time.time() - t_start
                sps = (step + 1) / max(elapsed, 1e-9)
                upright = _uprightness(sim, thorax_idx)
                upright_txt = "" if upright is None else f" upright={upright:.2f}"
                drives = getattr(controller, "_drives", np.array([np.nan, np.nan]))
                print(
                    f"step {step + 1:7d}/{args.max_steps} "
                    f"dist={final_dist:.2f} min={min_dist:.2f}{upright_txt} "
                    f"drives=({drives[0]:.2f},{drives[1]:.2f}) "
                    f"speed={sps:.0f} steps/s",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("Interrupted.", flush=True)
    finally:
        video.close()
        if args.display_backend == "pygame":
            import pygame

            pygame.quit()
        elif args.display_backend == "cv2":
            import cv2

            cv2.destroyAllWindows()

    print(
        f"done. success={success} min_dist={min_dist:.2f} final_dist={final_dist:.2f} "
        f"wall_s={time.time() - t_start:.1f}",
        flush=True,
    )
    if args.save_video is not None:
        print(f"saved video: {args.save_video}", flush=True)


if __name__ == "__main__":
    main()
