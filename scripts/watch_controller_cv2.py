import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tqdm

from flygym.compose import ActuatorType
from miniproject import MiniprojectSimulation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "miniproject"))
from submission.controller import Controller


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-l", "--level", type=int, required=True)
    p.add_argument("-s", "--seed", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--success-radius", type=float, default=3.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--show", action="store_true", help="Show OpenCV window while saving.")
    p.add_argument("--render-fly-vision", action="store_true")
    return p.parse_args()


def to_uint8_rgb(frame):
    if frame.dtype == np.uint8:
        return frame
    frame = np.asarray(frame)
    if frame.max() <= 1.0:
        frame = frame * 255
    return np.clip(frame, 0, 255).astype(np.uint8)


def reached_goal(sim, radius):
    banana_xy = np.asarray(sim.world.banana_xy)
    fly_xy = np.asarray(sim.get_body_positions(sim.fly.name)[0][:2])
    dist = float(np.linalg.norm(fly_xy - banana_xy))
    return dist <= radius, dist


def build_frame(sim, render_fly_vision=False):
    frame = np.concatenate(
        [frames[-1] for frames in sim.renderer.frames.values()],
        axis=-2,
    )

    if render_fly_vision:
        fly_vision = np.concatenate(sim.get_raw_vision(sim.fly.name), axis=-2)
        fly_vision = to_uint8_rgb(fly_vision)

        frame = to_uint8_rgb(frame)
        pad_total = max(0, frame.shape[1] - fly_vision.shape[1])
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        fly_vision = cv2.copyMakeBorder(
            fly_vision,
            0,
            0,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        frame = np.vstack((fly_vision, frame))

    return to_uint8_rgb(frame)


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sim = MiniprojectSimulation(level=args.level, seed=args.seed)
    controller = Controller(sim)

    writer = None
    window_name = f"L{args.level}_seed{args.seed}"

    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_dist = None

    for step in tqdm.tqdm(range(args.max_steps)):
        ok, dist = reached_goal(sim, args.success_radius)
        last_dist = dist

        if ok:
            print(f"SUCCESS: reached goal at step {step}, dist={dist:.3f}")
            break

        joint_angles, adhesion_signals = controller.step(sim)

        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion_signals)
        sim.step()

        if sim.render_as_needed():
            frame_rgb = build_frame(sim, args.render_fly_vision)
            frame_bgr = frame_rgb[..., ::-1]

            if writer is None:
                h, w = frame_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer: {out_path}")

            writer.write(frame_bgr)

            if args.show:
                cv2.imshow(window_name, frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Stopped by user.")
                    break

    else:
        print(f"FAILED: max steps reached, final_dist={last_dist:.3f}")

    if writer is not None:
        writer.release()

    if args.show:
        cv2.destroyAllWindows()

    print(f"Saved video to: {out_path}")


if __name__ == "__main__":
    main()
