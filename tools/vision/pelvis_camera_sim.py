"""Node ROS 'pelvis_camera_sim' : "camera fantome" -- publie une image
sensor_msgs/Image + CameraInfo qui imite ce que verrait une camera montee sur
le bassin (LINK_BASE) du PM01, alors qu'AUCUNE camera n'existe reellement
dans la simulation pm01_edu_carton (verifie 2026-08-18 : aucun <camera> dans
le MJCF, aucun node/topic image dans ros2_bridge/default.yaml -- voir memoire
projet). Ne modifie ni ne rebuild le binaire C++ de run_mujoco.sh.

Principe : un modele MuJoCo separe (shadow_scene.py, memes fichiers robot/
carton/podium que la vraie scene + une camera ajoutee) est mis a jour en
CINEMATIQUE SEULE (mj_forward, jamais mj_step) a partir de la pose reelle du
robot lue sur le canal LCM `sim_state` (meme canal que walk_to_xy.py), puis
rendu hors-ligne (MUJOCO_GL=egl) depuis cette camera. Le marqueur ArUco est
ensuite COMPOSITE en 2D sur l'image rendue par homographie (voir
_composite_marker) -- le texture-mapping natif de MuJoCo pour un motif ArUco
plat s'est revele peu fiable (voir shadow_scene.py), le compositing OpenCV
est pixel-parfait et independant du moteur de rendu.

Limite assumee (comme shadow_scene.py) : le carton est suppose IMMOBILE a sa
position XML initiale -- `sim_state` n'expose pas sa pose reelle (aucun canal
LCM ne la publie). Valide pour detecter + se positionner AVANT la prise,
invalide une fois le carton saisi/deplace.

Sortie conçue pour etre consommee TELLE QUELLE par
tools/vision/aruco_carton_test.py --image-topic /camera/pelvis_sim/image_raw
--marker-size 0.1016 (taille THEORIQUE du PNG genere, rien n'est imprime ici).
"""
import math
import os
import sys
import tempfile
import threading
import time

import cv2
import lcm
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shadow_scene  # noqa: E402

sys.path.insert(0, "/home/equansrobotic/engineai_robotics_native_sdk/tools/virtual_gamepad")
from lcm_msgs.data import SimState  # noqa: E402

LCM_URL = "udpm://239.255.76.67:7667?ttl=1"
SIM_STATE_CHANNEL = "sim_state"

IMAGE_TOPIC = "/camera/pelvis_sim/image_raw"
CAMERA_INFO_TOPIC = "/camera/pelvis_sim/camera_info"
# RENDER_HZ abaisse (etait 10Hz) pour reduire la charge du rendu offscreen
# EGL sans GPU, qui sature un coeur au point de retarder la boucle physique
# de run_mujoco.sh et de faire tomber le robot pendant une marche/un stand
# actif (contention CPU confirmee le 19/08 -- voir memoire projet). RESOLUTION
# gardee a 640x480 : essaye 320x240 d'abord, le marqueur (101.6mm) devient
# trop petit en pixels a ~2m de distance pour etre detecte par ArUco -- a ne
# pas re-baisser sans re-verifier la detection a la distance de spawn.
# Combiner avec `renice -n 15` sur ce process (voir
# docs/COMMENT_TESTER_CAMERA_FANTOME.txt) pour proteger la boucle physique.
WIDTH, HEIGHT = 640, 480
RENDER_HZ = 4.0


class SimStateListener:
    """Copie du meme pattern que test_passe/walk_to_xy.py::SimStateListener --
    garde le dernier message sim_state recu, thread LCM dedie."""

    def __init__(self, lcm_url=LCM_URL):
        self._lc = lcm.LCM(lcm_url)
        self._lc.subscribe(SIM_STATE_CHANNEL, self._on_message)
        self._latest = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _on_message(self, channel, data):
        state = SimState.decode(data)
        with self._lock:
            self._latest = state

    def _spin(self):
        while True:
            self._lc.handle()

    def latest(self):
        with self._lock:
            return self._latest


def _composite_marker(img_bgr, cam_pos, cam_mat, fovy_deg, width, height, marker_bgr):
    """Projette les 4 coins 3D (connus, fixes) du marqueur dans l'image via les
    intrinseques/extrinseques de la camera MuJoCo, puis warp+colle le PNG du
    marqueur dessus (cv2.warpPerspective) -- remplace le texture-mapping
    MuJoCo natif, peu fiable pour ce motif (voir shadow_scene.py)."""
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    fx = fy
    cxp, cyp = width / 2.0, height / 2.0

    half = shadow_scene.MARKER_SIZE / 2.0
    cx, cy, cz = shadow_scene.CARTON_POS
    mx = cx - shadow_scene.CARTON_HALF_X - 0.0006
    # Ordre TL, TR, BR, BL tel que vu par le robot en approchant (+X monde) --
    # "droite" = -Y, "haut" = +Z pour un observateur regardant vers +X avec
    # +Z vers le haut (meme convention que aruco_carton_test.py::object_points).
    corners_world = np.array([
        [mx, cy + half, cz + half],
        [mx, cy - half, cz + half],
        [mx, cy - half, cz - half],
        [mx, cy + half, cz - half],
    ])

    R = cam_mat.reshape(3, 3)
    pix = np.empty((4, 2), dtype=np.float32)
    for i, p in enumerate(corners_world):
        pc = R.T @ (p - cam_pos)
        xc, yc, zc = pc
        depth = -zc
        if depth <= 0.01:
            return img_bgr  # marqueur derriere la camera -- rien a coller
        pix[i] = [cxp + fx * xc / depth, cyp - fy * yc / depth]

    h_src, w_src = marker_bgr.shape[:2]
    src = np.array([[0, 0], [w_src - 1, 0], [w_src - 1, h_src - 1], [0, h_src - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(src, pix)
    warped = cv2.warpPerspective(marker_bgr, transform, (width, height))
    mask = cv2.warpPerspective(
        np.full((h_src, w_src), 255, dtype=np.uint8), transform, (width, height)
    )
    out = img_bgr.copy()
    out[mask > 0] = warped[mask > 0]
    return out


class PelvisCameraSim(Node):
    def __init__(self):
        super().__init__("pelvis_camera_sim")
        self._tmpdir = tempfile.mkdtemp(prefix="pelvis_camera_sim_")
        scene_path = shadow_scene.build_shadow_scene(self._tmpdir)
        self._model = mujoco.MjModel.from_xml_path(scene_path)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, self._data)
        self._cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, shadow_scene.CAMERA_NAME)
        self._renderer = mujoco.Renderer(self._model, height=HEIGHT, width=WIDTH)

        # Rogne a la zone motif seule (sans la marge blanche du PNG imprimable) --
        # sinon le motif compose apparait plus petit que MARKER_SIZE, faussant
        # la distance solvePnP d'un facteur ~1.4 (voir shadow_scene.py).
        self._marker_bgr = shadow_scene.load_marker_pattern_bgr()

        self._sim_state = SimStateListener()
        self._image_pub = self.create_publisher(Image, IMAGE_TOPIC, 10)
        self._info_pub = self.create_publisher(CameraInfo, CAMERA_INFO_TOPIC, 10)
        self._camera_info_msg = self._build_camera_info()

        self._warned_no_state = False
        self.create_timer(1.0 / RENDER_HZ, self._on_timer)
        self.get_logger().info(
            f"Camera fantome pelvis prete -- publie sur {IMAGE_TOPIC} / {CAMERA_INFO_TOPIC} "
            f"a {RENDER_HZ:.0f}Hz, en attente de 'sim_state' sur LCM..."
        )

    def _build_camera_info(self) -> CameraInfo:
        fy = (HEIGHT / 2.0) / math.tan(math.radians(shadow_scene.CAMERA_FOVY) / 2.0)
        fx = fy
        cxp, cyp = WIDTH / 2.0, HEIGHT / 2.0
        msg = CameraInfo()
        msg.width = WIDTH
        msg.height = HEIGHT
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [fx, 0.0, cxp, 0.0, fy, cyp, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cxp, 0.0, 0.0, fy, cyp, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    def _on_timer(self):
        state = self._sim_state.latest()
        if state is None:
            if not self._warned_no_state:
                self.get_logger().warn("Aucun message 'sim_state' recu -- run.sh/run_mujoco.sh actifs ?")
                self._warned_no_state = True
            return
        self._warned_no_state = False

        self._data.qpos[0:3] = state.base_link_position
        # SimState stocke (w,x,y,z), qpos MuJoCo pour un freejoint attend aussi (w,x,y,z) -- meme ordre.
        self._data.qpos[3:7] = state.base_link_quaternion
        mujoco.mj_forward(self._model, self._data)

        self._renderer.update_scene(self._data, camera=shadow_scene.CAMERA_NAME)
        img_rgb = self._renderer.render()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr = _composite_marker(
            img_bgr,
            self._data.cam_xpos[self._cam_id].copy(),
            self._data.cam_xmat[self._cam_id].copy(),
            shadow_scene.CAMERA_FOVY,
            WIDTH, HEIGHT,
            self._marker_bgr,
        )

        stamp = self.get_clock().now().to_msg()
        img_msg = Image()
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = "pelvis_cam_sim"
        img_msg.height, img_msg.width = img_bgr.shape[:2]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.step = img_msg.width * 3
        img_msg.data = img_bgr.tobytes()
        self._image_pub.publish(img_msg)

        self._camera_info_msg.header.stamp = stamp
        self._camera_info_msg.header.frame_id = "pelvis_cam_sim"
        self._info_pub.publish(self._camera_info_msg)


def main():
    rclpy.init()
    node = PelvisCameraSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
