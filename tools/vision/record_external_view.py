"""Enregistre une video MP4 depuis la camera EXTERIEURE fixe
(shadow_scene.EXTERNAL_CAMERA_NAME, cf shadow_scene.py) plutot que la camera
fantome pelvis (premiere personne) -- demande utilisateur 2026-08-20 :
"vue exterieure, on voit le robot bien".

Autonome, PAS un node ROS -- rend directement la scene MuJoCo fantome
(comme pelvis_camera_sim.py) et ecrit les frames dans un fichier video, sans
passer par un topic image intermediaire (plus simple pour un enregistrement
ponctuel, pas un flux temps reel a consommer ailleurs).

Usage :
    source /opt/ros/humble/setup.bash  # pour lcm (deja installe globalement)
    python3 record_external_view.py [duree_secondes] [chemin_sortie.mp4]
"""
import os
import sys
import time

import cv2
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shadow_scene  # noqa: E402
from pelvis_camera_sim import SimStateListener, _composite_marker  # noqa: E402

WIDTH, HEIGHT = 640, 480
RENDER_HZ = 4.0


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/sim_capture_external.mp4"

    tmpdir = "/tmp/record_external_view_scene"
    os.makedirs(tmpdir, exist_ok=True)
    scene_path = shadow_scene.build_shadow_scene(tmpdir)
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, shadow_scene.EXTERNAL_CAMERA_NAME)
    if cam_id < 0:
        raise RuntimeError(f"Camera '{shadow_scene.EXTERNAL_CAMERA_NAME}' introuvable dans la scene.")
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    marker_bgr = shadow_scene.load_marker_pattern_bgr()

    sim_state = SimStateListener()
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), RENDER_HZ, (WIDTH, HEIGHT))

    print(f"Enregistrement {duration:.0f}s -> {out_path} (attente de sim_state...)", flush=True)
    t0 = time.time()
    n_frames = 0
    warned = False
    while time.time() - t0 < duration:
        state = sim_state.latest()
        if state is None:
            if not warned:
                print("Aucun sim_state recu -- run.sh/run_mujoco.sh actifs ?", flush=True)
                warned = True
            time.sleep(1.0 / RENDER_HZ)
            continue
        data.qpos[0:3] = state.base_link_position
        data.qpos[3:7] = state.base_link_quaternion
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=shadow_scene.EXTERNAL_CAMERA_NAME)
        img_rgb = renderer.render()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr = _composite_marker(
            img_bgr,
            data.cam_xpos[cam_id].copy(),
            data.cam_xmat[cam_id].copy(),
            shadow_scene.EXTERNAL_CAMERA_FOVY,
            WIDTH, HEIGHT,
            marker_bgr,
        )
        writer.write(img_bgr)
        n_frames += 1
        time.sleep(1.0 / RENDER_HZ)

    writer.release()
    print(f"frames={n_frames} saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
