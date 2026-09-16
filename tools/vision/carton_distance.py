"""Node ROS 'carton_distance' : distance metrique au carton via la depth RealSense,
utilisable MEME quand le marqueur ArUco n'est pas entierement visible (carton trop
pres -- cv2.aruco exige de voir le quadrilatere complet du marqueur, voir echange du
2026-09-15 : approche rapprochee avant la sequence de levee).

Principe : lit le flux depth ALIGNE sur la couleur (aligned_depth_to_color -- memes
pixels que l'image couleur, donc directement comparable a une detection ArUco faite
sur cette meme image) et prend la distance MEDIANE (robuste aux trous de depth :
reflets, bords, hors-portee) dans une petite zone (ROI) autour d'un point de reference :

  - si le marqueur EST detecte (reutilise aruco_carton_test.py, PAS duplique) : la ROI
    se recentre sur son centre pixel -- distance precise au marqueur ;
  - sinon (carton trop pres, marqueur coupe/hors champ) : la ROI reste au centre de
    l'image -- couvre exactement le cas qui a motive ce script, le carton remplissant
    alors le centre du champ pendant l'approche finale.

Publie un Float32 (metres) sur --distance-topic, et republie (optionnel) une image
depth colorisee avec le cadre de la ROI dessine, a ouvrir dans Foxglove pour verifier
visuellement que la zone mesuree est bien sur le carton et pas a cote.

Usage (topics reels confirmes le 2026-09-15) :
    python3 carton_distance.py --marker-size 0.098
"""
import argparse
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

import aruco_carton_test as act

ROI_HALF_SIZE_PX = 40  # cote ~80px de la zone dont on prend la depth mediane
PRINT_PERIOD_S = 1.0


def _depth_msg_to_array(msg: Image) -> np.ndarray:
    """Convertit un sensor_msgs/Image depth (16UC1, millimetres) en tableau numpy --
    meme logique de conversion manuelle que act._image_msg_to_bgr (pas de cv_bridge)."""
    if msg.encoding != "16UC1":
        raise ValueError(f"Encoding depth '{msg.encoding}' non gere (attendu 16UC1).")
    return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)


class CartonDistance(Node):

    def __init__(self, args):
        super().__init__("carton_distance")
        self.marker_id = args.marker_id
        aruco_dict = cv2.aruco.getPredefinedDictionary(args.dict)
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

        self._roi_center = None  # (u, v) pixels -- recentre des que le marqueur est vu
        self._last_print = 0.0

        self.create_subscription(Image, args.color_topic, self._on_color, 10)
        self.create_subscription(Image, args.depth_topic, self._on_depth, 10)
        self.dist_pub = self.create_publisher(Float32, args.distance_topic, 10)

        self.debug_pub = None
        if args.publish_debug:
            debug_topic = args.depth_topic.rsplit("/", 1)[0] + "/distance_debug"
            self.debug_pub = self.create_publisher(Image, debug_topic, 10)
            self.get_logger().info(f"Image depth annotee republiee sur {debug_topic}")

        self.get_logger().info(
            f"carton_distance pret -- {args.color_topic} (recentrage ROI) + "
            f"{args.depth_topic} (mesure) -> {args.distance_topic} ; ROI = centre de "
            f"l'image tant qu'aucun marqueur n'est detecte."
        )

    def _on_color(self, msg: Image):
        try:
            frame = act._image_msg_to_bgr(msg)
        except ValueError:
            return
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is not None and self.marker_id in ids.flatten():
            idx = list(ids.flatten()).index(self.marker_id)
            c = corners[idx][0]
            self._roi_center = (float(c[:, 0].mean()), float(c[:, 1].mean()))
        else:
            self._roi_center = None

    def _on_depth(self, msg: Image):
        try:
            depth_mm = _depth_msg_to_array(msg)
        except ValueError as e:
            self.get_logger().error(str(e))
            return

        h, w = depth_mm.shape
        if self._roi_center is not None:
            cu, cv_ = self._roi_center
            source = "marqueur"
        else:
            cu, cv_ = w / 2.0, h / 2.0  # marqueur non vu (trop pres) -> centre image
            source = "centre image (marqueur non vu)"

        u0, u1 = max(0, int(cu) - ROI_HALF_SIZE_PX), min(w, int(cu) + ROI_HALF_SIZE_PX)
        v0, v1 = max(0, int(cv_) - ROI_HALF_SIZE_PX), min(h, int(cv_) + ROI_HALF_SIZE_PX)
        roi = depth_mm[v0:v1, u0:u1]
        valid = roi[roi > 0]  # 0 = pixel depth invalide (trou/hors portee)

        if valid.size == 0:
            self.get_logger().warn("carton_distance : aucune depth valide dans la ROI.")
            return

        distance_m = float(np.median(valid)) / 1000.0
        self.dist_pub.publish(Float32(data=distance_m))

        now = time.time()
        if now - self._last_print > PRINT_PERIOD_S:
            self._last_print = now
            self.get_logger().info(f"distance carton = {distance_m:.3f}m (ROI sur {source})")

        if self.debug_pub is not None:
            vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            cv2.rectangle(vis, (u0, v0), (u1, v1), (255, 255, 255), 2)
            debug_msg = Image()
            debug_msg.height, debug_msg.width = vis.shape[:2]
            debug_msg.encoding = "bgr8"
            debug_msg.is_bigendian = 0
            debug_msg.step = debug_msg.width * 3
            debug_msg.data = vis.tobytes()
            self.debug_pub.publish(debug_msg)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--color-topic", default="/camera/pelvis/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/pelvis/aligned_depth_to_color/image_raw")
    parser.add_argument("--distance-topic", default="/vision/carton_distance")
    parser.add_argument("--marker-id", type=int, default=act.DEFAULT_MARKER_ID)
    parser.add_argument("--dict", type=int, default=act.DEFAULT_DICT)
    parser.add_argument("--publish-debug", action="store_true", default=True)
    parser.add_argument("--no-publish-debug", dest="publish_debug", action="store_false")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = CartonDistance(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
