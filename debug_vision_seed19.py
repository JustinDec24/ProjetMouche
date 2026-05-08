"""Debug vision détaillé seed 19 — focus sur le flip instantané post-AVOID."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "miniproject"))
from submission.controller import Controller
from miniproject import MiniprojectSimulation
from flygym.compose import ActuatorType
import numpy as np
from PIL import Image, ImageDraw

SEED = 19
LEVEL = 2
MAX_STEPS = 50_000

OUT_DIR = Path(__file__).parent / "vision_debug_seed19"
OUT_DIR.mkdir(exist_ok=True)
for f in OUT_DIR.glob("*.png"):
    f.unlink()

sim = MiniprojectSimulation(level=LEVEL, seed=SEED)
ctrl = Controller(sim)
thorax_idx = ctrl._thorax_idx
banana_xy = np.asarray(sim.world.banana_xy, dtype=float)

frame_idx = 0
last_dec = -1
log = []
prev_avoid = False
flip_start_step = -1

# Wider sampling rate around critical moments
def should_save(step, in_avoid, prev_avoid_local, up):
    decision = step // ctrl._decision_every
    if in_avoid != prev_avoid_local:
        return True  # transition
    if up < 0.7:
        return True  # tilting
    # Pendant AVOID : tous les 10 décisions
    if in_avoid and decision % 10 == 0:
        return True
    # Hors AVOID : tous les 30 décisions
    if not in_avoid and decision % 30 == 0:
        return True
    return False


for step in range(MAX_STEPS):
    joint_angles, adhesion = ctrl.step(sim)
    sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
    sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
    sim.step()

    thorax_xy = sim.get_body_positions(sim.fly.name)[thorax_idx, :2]
    dist = float(np.linalg.norm(thorax_xy - banana_xy))
    if dist <= 2.0:
        print(f"SUCCESS at step {step}", flush=True)
        break

    decision = step // ctrl._decision_every
    cur_avoid = ctrl._avoid_left > 0
    xmat = sim.mj_data.xmat[ctrl._thorax_body_id].reshape(3, 3)
    up = float(xmat[2, 2])
    roll = float(xmat[2, 1])
    pitch = float(xmat[2, 0])

    # Detect flip start
    if up < 0.7 and flip_start_step < 0:
        flip_start_step = step
        print(f"FLIP DETECTED at step={step} dist={dist:.2f} up={up:.2f}", flush=True)

    # Save frames
    if decision != last_dec and should_save(step, cur_avoid, prev_avoid, up):
        last_dec = decision
        overlay = ctrl.compute_vision_debug_overlay(sim)
        if overlay is not None:
            img = Image.fromarray(overlay)
            draw = ImageDraw.Draw(img)
            txt = (
                f"step={step:5d} dec={decision:4d} dist={dist:.2f}\n"
                f"obs_size={ctrl._vis_obs_size:.4f} obs_x={ctrl._vis_obs_x:+.3f}\n"
                f"avoid={cur_avoid} latch={ctrl._latched_obs_x:+.3f}\n"
                f"upright={up:.2f} roll={roll:+.2f} pitch={pitch:+.2f}"
            )
            draw.text((5, 5), txt, fill=(255, 255, 0))
            img.save(OUT_DIR / f"frame_{frame_idx:04d}_step{step:06d}.png")
            frame_idx += 1
            log.append((step, dist, ctrl._vis_obs_size, ctrl._vis_obs_x, cur_avoid, up, roll))

    prev_avoid = cur_avoid

print(f"=== Saved {frame_idx} frames ===", flush=True)
print("=== Log around critical moments ===", flush=True)
for s, d, sz, x, av, up, roll in log:
    marker = ""
    if up < 0.5: marker = " <FLIP>"
    elif up < 0.85: marker = " <tilt>"
    print(f"  step={s:6d} dist={d:5.2f} sz={sz:.3f} x={x:+.2f} avoid={av} up={up:+.2f} roll={roll:+.2f}{marker}", flush=True)
