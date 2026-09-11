"""Enregistre une video MP4 depuis la camera EXTERIEURE fixe (meme camera que
record_external_view.py), mais anime AUSSI les 24 articulations (bras/jambes/
buste), pas seulement la position du corps -- record_external_view.py ne lit
que sim_state.base_link_position/quaternion (qpos[0:7]), le reste du fantome
MuJoCo reste fige a sa pose de depart, invisible pour verifier un geste de
bras. Demande explicite de l'utilisateur (visualiser precisement le geste des
bras plutot que deviner sur description texte).

Sources :
- sim_state (canal LCM, deja utilise par les autres outils vision/) pour
  qpos[0:3]/[3:7] (position/orientation du bassin).
- /hardware/joint_state (topic ROS2, QoS BEST_EFFORT/VOLATILE comme
  /motion/joint_override_command) pour qpos[7:31] -- msg.position est un
  tableau de 24 valeurs, MEME INDEXATION que NUM_JOINTS/LEFT_JOINT_INDICES
  dans lever.py (positionnel, pas de champ "name" dans ce message) :
  0-5 jambe gauche, 6-11 jambe droite, 12 buste (waist yaw), 13-17 bras
  gauche, 18-22 bras droit, 23 tete.

Autonome, PAS un node de production -- rend directement la scene MuJoCo
fantome et ecrit les frames dans un fichier video, comme
record_external_view.py.

Usage :
    export ROS_LOCALHOST_ONLY=1
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    source /opt/ros/humble/setup.bash
    source .../build/ros2_env/install/local_setup.bash
    python3 record_arms_view.py [duree_secondes] [chemin_sortie.mp4]
"""
import os
import sys
import time

import cv2
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

import rclpy  # noqa: E402
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy  # noqa: E402
from interface_protocol.msg import JointState  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shadow_scene  # noqa: E402
from pelvis_camera_sim import SimStateListener  # noqa: E402

WIDTH, HEIGHT = 640, 480
RENDER_HZ = 8.0
NUM_JOINTS = 24


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/sim_capture_arms.mp4"

    tmpdir = "/tmp/record_arms_view_scene"
    os.makedirs(tmpdir, exist_ok=True)
    scene_path = shadow_scene.build_shadow_scene(tmpdir)
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, shadow_scene.EXTERNAL_CAMERA_NAME)
    if cam_id < 0:
        raise RuntimeError(f"Camera '{shadow_scene.EXTERNAL_CAMERA_NAME}' introuvable dans la scene.")
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    sim_state = SimStateListener()

    rclpy.init()
    node = rclpy.create_node("record_arms_view")
    joint_positions = {"data": None}

    def _on_joint_state(msg):
        joint_positions["data"] = list(msg.position)

    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )
    node.create_subscription(JointState, "/hardware/joint_state", _on_joint_state, qos)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), RENDER_HZ, (WIDTH, HEIGHT))

    print(f"Enregistrement {duration:.0f}s -> {out_path} (attente de sim_state + joint_state...)", flush=True)
    t0 = time.time()
    n_frames = 0
    warned = False
    while time.time() - t0 < duration:
        rclpy.spin_once(node, timeout_sec=0.0)
        state = sim_state.latest()
        joints = joint_positions["data"]
        if state is None or joints is None or len(joints) != NUM_JOINTS:
            if not warned:
                print("En attente de sim_state ET /hardware/joint_state -- "
                      "run.sh/run_mujoco.sh actifs ?", flush=True)
                warned = True
            time.sleep(1.0 / RENDER_HZ)
            continue
        data.qpos[0:3] = state.base_link_position
        data.qpos[3:7] = state.base_link_quaternion
        data.qpos[7:7 + NUM_JOINTS] = joints
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=shadow_scene.EXTERNAL_CAMERA_NAME)
        img_rgb = renderer.render()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)
        n_frames += 1
        time.sleep(1.0 / RENDER_HZ)

    writer.release()
    node.destroy_node()
    rclpy.try_shutdown()
    print(f"frames={n_frames} saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
