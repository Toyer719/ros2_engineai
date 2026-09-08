"""Script de test standalone (etapes 5 et 6 de PLAN_TEST_ARUCO_CARTON.txt) : detecte le
marqueur ArUco colle sur le carton via le topic camera reel, calcule sa pose 3D, et
republie une image annotee pour verification visuelle dans Foxglove.

NE TOUCHE PAS a choreo_native.py -- but unique : valider que detection + pose marchent
avant toute integration (etape 9, plus tard).

Conversion image ROS2 <-> OpenCV faite A LA MAIN (pas cv_bridge) : evite le conflit
numpy 1.x/2.x rencontre avec cv_bridge dans cet environnement (voir etape 0 du plan) --
pour un encoding brut bgr8/rgb8 (sensor_msgs/Image, PAS compresse), c'est un simple
reshape, cv_bridge n'apporte rien ici.

Usage (topic camera confirme le 2026-08-12 : /camera/pelvis/image_raw) :
    python3 aruco_carton_test.py --marker-size 0.1016
    (remplacer 0.1016 par la taille REELLEMENT MESUREE au double-decimetre apres
    impression, etape 1 du plan -- PAS forcement la valeur theorique)

Si le topic camera_info n'est pas /camera/pelvis/camera_info (deduit automatiquement du
topic image en remplacant image_raw -> camera_info), le donner explicitement :
    python3 aruco_carton_test.py --marker-size 0.1016 --camera-info-topic /autre/topic
"""

import argparse
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

DEFAULT_IMAGE_TOPIC = "/camera/pelvis/image_raw"
DEFAULT_MARKER_ID = 0
DEFAULT_DICT = cv2.aruco.DICT_4X4_50
PRINT_PERIOD_S = 1.0  # limite l'affichage console a 1x/seconde (le flux image tourne bien plus vite)


def _image_msg_to_bgr(msg: Image) -> np.ndarray:
    """Convertit un sensor_msgs/Image brut (bgr8/rgb8/mono8) en tableau BGR OpenCV,
    sans passer par cv_bridge (voir docstring du module)."""
    if msg.encoding not in ("bgr8", "rgb8", "mono8"):
        raise ValueError(f"Encoding '{msg.encoding}' non gere par ce script (attendu bgr8/rgb8/mono8) -- "
                          "si la camera publie en compresse (sensor_msgs/CompressedImage), utiliser "
                          "cv2.imdecode a la place, pas ce chemin.")
    channels = 1 if msg.encoding == "mono8" else 3
    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
    if msg.encoding == "mono8":
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if msg.encoding == "rgb8":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame.copy()  # deja bgr8


def _bgr_to_image_msg(frame: np.ndarray) -> Image:
    msg = Image()
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = frame.tobytes()
    return msg


class ArucoCartonLocator(Node):
    """S'abonne a l'image camera + camera_info, detecte le marqueur du carton, calcule
    sa pose (repere camera) via solvePnP, et republie une image annotee pour Foxglove."""

    def __init__(self, args):
        super().__init__("aruco_carton_test")
        self.marker_id = args.marker_id
        self.marker_size = args.marker_size
        self.half = self.marker_size / 2.0
        # Ordre des coins retourne par detectMarkers : haut-gauche, haut-droit,
        # bas-droit, bas-gauche (sens horaire) -- doit matcher cet ordre.
        self.object_points = np.array([
            [-self.half,  self.half, 0.0],
            [ self.half,  self.half, 0.0],
            [ self.half, -self.half, 0.0],
            [-self.half, -self.half, 0.0],
        ], dtype=np.float64)

        aruco_dict = cv2.aruco.getPredefinedDictionary(args.dict)
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

        self.camera_matrix = None
        self.dist_coeffs = None
        self._last_print = 0.0

        self.create_subscription(Image, args.image_topic, self._on_image, 10)
        self.create_subscription(CameraInfo, args.camera_info_topic, self._on_camera_info, 10)

        self.debug_pub = None
        if args.publish_debug:
            debug_topic = args.image_topic.rsplit("/", 1)[0] + "/aruco_debug"
            self.debug_pub = self.create_publisher(Image, debug_topic, 10)
            self.get_logger().info(f"Image annotee republiee sur {debug_topic} (a ouvrir dans Foxglove)")

        self.get_logger().info(
            f"En attente d'images sur {args.image_topic} et intrinseques sur "
            f"{args.camera_info_topic} -- marker_id={self.marker_id} marker_size={self.marker_size}m"
        )

    def _on_camera_info(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)
            self.get_logger().info(f"Intrinseques recues : K=\n{self.camera_matrix}\nD={self.dist_coeffs}")

    def _on_image(self, msg: Image):
        try:
            frame = _image_msg_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e))
            return

        corners, ids, _ = self.detector.detectMarkers(frame)
        found = ids is not None and self.marker_id in ids.flatten()

        if found:
            idx = list(ids.flatten()).index(self.marker_id)
            marker_corners = corners[idx]
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if self.camera_matrix is not None:
                ok, rvec, tvec = cv2.solvePnP(
                    self.object_points, marker_corners[0], self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if ok:
                    distance = float(np.linalg.norm(tvec))
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec,
                                       length=self.half)
                    now = time.time()
                    if now - self._last_print > PRINT_PERIOD_S:
                        self._last_print = now
                        self.get_logger().info(
                            f"marqueur {self.marker_id} detecte -- distance={distance:.3f}m "
                            f"tvec={tvec.flatten()} rvec={rvec.flatten()}"
                        )
            else:
                now = time.time()
                if now - self._last_print > PRINT_PERIOD_S:
                    self._last_print = now
                    self.get_logger().warn(
                        f"marqueur {self.marker_id} detecte mais pas encore d'intrinseques "
                        "(camera_info) -- pose 3D impossible pour l'instant"
                    )

        if self.debug_pub is not None:
            self.debug_pub.publish(_bgr_to_image_msg(frame))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--camera-info-topic", default=None,
                         help="Deduit par defaut de --image-topic (remplace 'image_raw' par 'camera_info').")
    parser.add_argument("--marker-id", type=int, default=DEFAULT_MARKER_ID)
    parser.add_argument("--marker-size", type=float, required=True,
                         help="Taille reelle du cote du marqueur en metres -- MESUREE apres impression "
                              "(etape 1 du plan), pas la valeur theorique.")
    parser.add_argument("--dict", type=int, default=DEFAULT_DICT,
                         help="Constante cv2.aruco.DICT_* (defaut DICT_4X4_50).")
    parser.add_argument("--publish-debug", action="store_true", default=True)
    parser.add_argument("--no-publish-debug", dest="publish_debug", action="store_false")
    args, ros_args = parser.parse_known_args()

    if args.camera_info_topic is None:
        args.camera_info_topic = args.image_topic.replace("image_raw", "camera_info")

    rclpy.init(args=ros_args)
    node = ArucoCartonLocator(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
