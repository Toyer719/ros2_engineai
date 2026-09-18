"""Detection du marqueur ArUco colle SUR LE DESSUS du carton (pas la face avant, contrairement
a aruco_carton_test.py/carton_pose_publisher.py) -- meme detection/solvePnP, deja valide et
inchangee (la pose d'un marqueur plan ne depend pas de la face sur laquelle il est colle), mais
deduction differente du point utile pour la prise : le centre de la face avant est maintenant
un decalage VERS LE BAS + VERS L'AVANT depuis le marqueur (au lieu d'un decalage le long de la
normale, qui pointait vers l'avant pour un marqueur en face avant).

CONVENTION DE COLLAGE (a respecter, sinon les axes ci-dessous ne correspondent plus a la
realite) -- carton pose sur le podium, marqueur A PLAT, CENTRE sur le dessus, printed-side
vers le HAUT :
  - le bord "haut de l'image" du marqueur imprime (entre les coins TL et TR retournes par
    detectMarkers) doit pointer vers la face AVANT du carton (celle que le robot approche
    pour la prise) -- PAS vers l'arriere, PAS de cote.
Avec cette orientation :
  - axe local X du marqueur (TL->TR)  = axe LARGEUR du carton (lateral, gauche/droite)
  - axe local Y du marqueur (image "haut") = axe PROFONDEUR du carton, pointant vers l'AVANT
  - axe local Z du marqueur (normale, hors de la page) = axe VERTICAL, pointant vers le HAUT

Dimensions carton (docs/pm01_box_geometry.txt + PLAN_TEST_ARUCO_CARTON.txt, etape 7) :
  largeur (X) = 191mm, hauteur (Z) = 292mm, profondeur (Y) = 275mm
  => demi-largeur=95.5mm, demi-hauteur=146mm, demi-profondeur=137.5mm

Le centre de la face avant (point de reference pour pinch_x/pinch_y/pinch_z, meme convention
que package_sequence_bras_reel/levee.py, repere LINK_BASE/bassin) vaut, dans le repere local
du marqueur : (0, +demi-profondeur, -demi-hauteur).

PREREQUIS NON ENCORE FAIT (a ne pas oublier avant de faire confiance aux chiffres en sortie) :
la transformation camera->robot (repere LINK_BASE) du VRAI robot n'a jamais ete calibree
(voir tools/vision/calibrate_camera_extrinsics.py -- ecrit le 2026-09-14 sans acces au robot
reel, jamais lance). Sans elle, ce script affiche la position en repere CAMERA uniquement
(distance, x/y/z relatifs a la camera) -- utile pour valider la detection/pose (meme demarche
que aruco_carton_test.py, etapes 5-6 du plan), mais PAS encore directement utilisable comme
--pinch-x/--pinch-yaw-offset pour levee_pivot.py tant que la calibration n'est pas faite.

Usage (meme marker_size que le marqueur deja imprime, mesure au double-decimetre) :
    python3 aruco_carton_top.py --marker-size 0.1016
"""
import argparse
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

import aruco_carton_test as act

HALF_WIDTH = 0.0955
HALF_HEIGHT = 0.146
HALF_DEPTH = 0.1375
# Decalage du centre de la face avant depuis le marqueur, repere LOCAL du marqueur
# (X=largeur, Y=profondeur vers l'avant, Z=normale/vertical) -- voir docstring module.
FRONT_FACE_OFFSET_LOCAL = np.array([0.0, HALF_DEPTH, -HALF_HEIGHT])


class ArucoCartonTopLocator(Node):
    """Meme detection que aruco_carton_test.py, mais affiche EN PLUS la position deduite du
    centre de la face avant (marqueur mounted sur le dessus, voir docstring module)."""

    def __init__(self, args):
        super().__init__("aruco_carton_top")
        self.marker_id = args.marker_id
        self.marker_size = args.marker_size
        half = self.marker_size / 2.0
        self.object_points = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
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
            debug_topic = args.image_topic.rsplit("/", 1)[0] + "/aruco_top_debug"
            self.debug_pub = self.create_publisher(Image, debug_topic, 10)
            self.get_logger().info(f"Image annotee republiee sur {debug_topic} (a ouvrir dans Foxglove)")

        self.get_logger().info(
            f"En attente d'images sur {args.image_topic} et intrinseques sur "
            f"{args.camera_info_topic} -- marker_id={self.marker_id} marker_size={self.marker_size}m "
            f"(marqueur SUR LE DESSUS -- voir docstring module pour la convention de collage)"
        )

    def _on_camera_info(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)
            self.get_logger().info(f"Intrinseques recues : K=\n{self.camera_matrix}\nD={self.dist_coeffs}")

    def _on_image(self, msg: Image):
        try:
            frame = act._image_msg_to_bgr(msg)
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
                if not ok or not np.all(np.isfinite(tvec)):
                    # meme singularite (vue quasi frontale) que carton_pose_publisher.py --
                    # repli sur ITERATIVE, voir son commentaire pour le detail.
                    ok, rvec, tvec = cv2.solvePnP(
                        self.object_points, marker_corners[0], self.camera_matrix, self.dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                if ok and np.all(np.isfinite(tvec)):
                    r_cam_from_marker, _ = cv2.Rodrigues(rvec)
                    marker_pos_cam = tvec.flatten()
                    front_face_cam = marker_pos_cam + r_cam_from_marker @ FRONT_FACE_OFFSET_LOCAL
                    distance = float(np.linalg.norm(marker_pos_cam))
                    front_distance = float(np.linalg.norm(front_face_cam))
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec,
                                       length=HALF_WIDTH)
                    now = time.time()
                    if now - self._last_print > act.PRINT_PERIOD_S:
                        self._last_print = now
                        self.get_logger().info(
                            f"marqueur {self.marker_id} -- distance={distance:.3f}m "
                            f"pos_cam={marker_pos_cam} | face avant deduite -- "
                            f"distance={front_distance:.3f}m pos_cam={front_face_cam} "
                            "(repere CAMERA -- pas encore repere robot, extrinseques reel "
                            "non calibrees, voir docstring module)"
                        )
            else:
                now = time.time()
                if now - self._last_print > act.PRINT_PERIOD_S:
                    self._last_print = now
                    self.get_logger().warn(
                        f"marqueur {self.marker_id} detecte mais pas encore d'intrinseques "
                        "(camera_info) -- pose 3D impossible pour l'instant"
                    )

        if self.debug_pub is not None:
            self.debug_pub.publish(act._bgr_to_image_msg(frame))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-topic", default=act.DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--camera-info-topic", default=None,
                         help="Deduit par defaut de --image-topic (remplace 'image_raw' par 'camera_info').")
    parser.add_argument("--marker-id", type=int, default=act.DEFAULT_MARKER_ID)
    parser.add_argument("--marker-size", type=float, required=True,
                         help="Taille reelle du cote du marqueur en metres -- MESUREE apres "
                              "impression, pas la valeur theorique (voir PLAN_TEST_ARUCO_CARTON.txt).")
    parser.add_argument("--dict", type=int, default=act.DEFAULT_DICT)
    parser.add_argument("--publish-debug", action="store_true", default=True)
    parser.add_argument("--no-publish-debug", dest="publish_debug", action="store_false")
    args, ros_args = parser.parse_known_args()

    if args.camera_info_topic is None:
        args.camera_info_topic = args.image_topic.replace("image_raw", "camera_info")

    rclpy.init(args=ros_args)
    node = ArucoCartonTopLocator(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
