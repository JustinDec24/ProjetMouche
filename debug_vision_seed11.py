"""Debug vision overlay for seed 11 — sauvegarde frames + log obs_size/obs_x."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "miniproject"))
from submission.controller import Controller
from miniproject import MiniprojectSimulation
from flygym.compose import ActuatorType
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEED = 11
LEVEL = 2
MAX_STEPS = 60_000
SAVE_EVERY_DECISIONS = 5  # frames saved every N decisions (1 decision = ~6 sim steps)

OUT_DIR = Path(__file__).parent / "vision_debug_seed11"
OUT_DIR.mkdir(exist_ok=True)
# Clean old frames
for f in OUT_DIR.glob("*.png"):
    f.unlink()

sim = MiniprojectSimulation(level=LEVEL, seed=SEED)
ctrl = Controller(sim)
thorax_idx = ctrl._thorax_idx
banana_xy = np.asarray(sim.world.banana_xy, dtype=float)

frame_idx = 0
last_decision = -1
log = []

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
    if decision != last_decision and decision % SAVE_EVERY_DECISIONS == 0:
        last_decision = decision
        overlay = ctrl.compute_vision_debug_overlay(sim)
        if overlay is not None:
            img = Image.fromarray(overlay)
            # Annotation : info en surimpression
            draw = ImageDraw.Draw(img)
            xmat = sim.mj_data.xmat[ctrl._thorax_body_id].reshape(3, 3)
            up = float(xmat[2, 2])
            obs_size = float(ctrl._vis_obs_size)
            obs_x = float(ctrl._vis_obs_x)
            avoid = ctrl._avoid_left > 0
            txt = (
                f"step={step:5d} dec={decision:4d} dist={dist:.2f}\n"
                f"obs_size={obs_size:.4f} obs_x={obs_x:+.3f}\n"
                f"avoid={avoid} latch={ctrl._latched_obs_x:+.3f}\n"
                f"upright={up:.2f} stopped={ctrl._stopped}"
            )
            draw.text((5, 5), txt, fill=(255, 255, 0))
            img.save(OUT_DIR / f"frame_{frame_idx:04d}_step{step:06d}.png")
            frame_idx += 1
            log.append((step, dist, obs_size, obs_x, avoid, up))

print(f"=== Saved {frame_idx} frames in {OUT_DIR} ===", flush=True)
print("=== Decision log (key transitions) ===", flush=True)
prev_avoid = False
for s, d, sz, x, av, up in log:
    if av != prev_avoid:
        marker = "  >>> AVOID ENTER" if av else "  <<< AVOID EXIT"
    else:
        marker = ""
    print(f"  step={s:5d} dist={d:.2f} sz={sz:.4f} x={x:+.3f} avoid={av} up={up:.2f}{marker}", flush=True)
    prev_avoid = av
