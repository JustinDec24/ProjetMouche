"""Debug vision for all seeds 10-19 — sauvegarde frames clés + log par seed.

Pour chaque seed : sauvegarde une frame à chaque changement d'état important :
  - entrée AVOID
  - milieu AVOID (toutes les 30 décisions pendant AVOID)
  - sortie AVOID
  - début flip (upright < 0.5)
  - upright très bas (<-0.5)

À la fin : log structuré pour chaque seed.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "miniproject"))
from submission.controller import Controller
from miniproject import MiniprojectSimulation
from flygym.compose import ActuatorType
import numpy as np
from PIL import Image, ImageDraw

LEVEL = 2
MAX_STEPS = 100_000
SEEDS = list(range(10, 20))

OUT_DIR = Path(__file__).parent / "vision_debug_all"
OUT_DIR.mkdir(exist_ok=True)
# Clean old frames
for f in OUT_DIR.glob("*.png"):
    f.unlink()


def annotate(img_arr, **kwargs):
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    txt = "\n".join(f"{k}={v}" for k, v in kwargs.items())
    draw.text((5, 5), txt, fill=(255, 255, 0))
    return img


def save_frame(ctrl, sim, seed, tag, dist):
    overlay = ctrl.compute_vision_debug_overlay(sim)
    if overlay is None:
        return
    xmat = sim.mj_data.xmat[ctrl._thorax_body_id].reshape(3, 3)
    up = float(xmat[2, 2])
    img = annotate(
        overlay,
        seed=seed, tag=tag, dist=f"{dist:.2f}",
        sz=f"{ctrl._vis_obs_size:.4f}", x=f"{ctrl._vis_obs_x:+.3f}",
        latch=f"{ctrl._latched_obs_x:+.3f}",
        avoid=ctrl._avoid_left > 0, up=f"{up:.2f}", stop=ctrl._stopped,
    )
    img.save(OUT_DIR / f"seed{seed:02d}_{tag}.png")


for seed in SEEDS:
    print(f"\n=== SEED {seed} ===", flush=True)
    sim = MiniprojectSimulation(level=LEVEL, seed=seed)
    ctrl = Controller(sim)
    thorax_idx = ctrl._thorax_idx
    banana_xy = np.asarray(sim.world.banana_xy, dtype=float)
    thorax_xy = sim.get_body_positions(sim.fly.name)[thorax_idx, :2]
    start_dist = float(np.linalg.norm(thorax_xy - banana_xy))

    min_dist = start_dist
    prev_avoid = False
    avoid_count = 0
    flip_logged = False
    deep_flip_logged = False
    transitions = []

    for step in range(MAX_STEPS):
        joint_angles, adhesion = ctrl.step(sim)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
        sim.step()
        thorax_xy = sim.get_body_positions(sim.fly.name)[thorax_idx, :2]
        dist = float(np.linalg.norm(thorax_xy - banana_xy))
        min_dist = min(min_dist, dist)
        if dist <= 2.0:
            print(f"  SUCCESS at step {step}", flush=True)
            break

        # Detect transitions
        cur_avoid = ctrl._avoid_left > 0
        xmat = sim.mj_data.xmat[ctrl._thorax_body_id].reshape(3, 3)
        up = float(xmat[2, 2])

        if cur_avoid and not prev_avoid:
            # AVOID enter
            avoid_count += 1
            tag = f"avoid{avoid_count}_enter_step{step:06d}"
            save_frame(ctrl, sim, seed, tag, dist)
            transitions.append(
                f"  step={step:6d} dist={dist:5.2f} ENTER_AVOID#{avoid_count}"
                f" sz={ctrl._vis_obs_size:.3f} x={ctrl._vis_obs_x:+.2f} up={up:.2f}"
            )
        elif (not cur_avoid) and prev_avoid:
            # AVOID exit
            tag = f"avoid{avoid_count}_exit_step{step:06d}"
            save_frame(ctrl, sim, seed, tag, dist)
            transitions.append(
                f"  step={step:6d} dist={dist:5.2f} EXIT_AVOID#{avoid_count}"
                f" up={up:.2f}"
            )
        elif cur_avoid and (step % 5000 == 0):
            # Mid-AVOID snapshot (every 5000 steps = ~0.5s)
            tag = f"avoid{avoid_count}_mid_step{step:06d}"
            save_frame(ctrl, sim, seed, tag, dist)

        # Flip detection
        if up < 0.5 and not flip_logged:
            tag = f"flip_start_step{step:06d}"
            save_frame(ctrl, sim, seed, tag, dist)
            transitions.append(
                f"  step={step:6d} dist={dist:5.2f} FLIP_START up={up:.2f}"
                f" avoid={cur_avoid}"
            )
            flip_logged = True
        if up < -0.3 and not deep_flip_logged:
            tag = f"flip_deep_step{step:06d}"
            save_frame(ctrl, sim, seed, tag, dist)
            transitions.append(
                f"  step={step:6d} dist={dist:5.2f} FLIP_DEEP up={up:.2f}"
            )
            deep_flip_logged = True

        prev_avoid = cur_avoid

    status = "OK" if min_dist <= 2.0 else "FAIL"
    print(f"  RESULT: {status} min={min_dist:.2f} avoid_episodes={avoid_count}", flush=True)
    for t in transitions:
        print(t, flush=True)

print(f"\n=== Frames saved in {OUT_DIR} ===", flush=True)
