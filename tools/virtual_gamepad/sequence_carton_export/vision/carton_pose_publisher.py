"""Node ROS 'carton_pose_publisher' : reprend la detection+solvePnP de
aruco_carton_test.py (NON modifie -- reste l'outil de validation/debug) mais
convertit la pose detectee (repere CAMERA, convention OpenCV) vers le repere
ROBOT (LINK_BASE) en appliquant la transformation camera->bassin CONNUE, et
publie le resultat comme geometry_msgs/PoseStamped exploitable par une
Action ROS (approach_carton dans chef_node.py).

Par defaut, la transformation utilisee est celle de la "camera fantome"
(shadow_scene.CAMERA_POS/CAMERA_XYAXES, simulation uniquement) -- passer
--camera-pos/--camera-xyaxes pour une autre camera (ex. robot reel, une fois
sa transformation connue, cf memoire projet [[project_pm01_carton_vision]]).

Conversion camera(OpenCV: X droite, Y bas, Z avant) -> camera(MuJoCo local:
X droite, Y haut, Z arriere) : X_mj=X_cv, Y_mj=-Y_cv, Z_mj=-Z_cv (verifiee
empiriquement 2026-08-19 contre la verite terrain MuJoCo -- erreur < 5mm).
Puis P_base = camera_pos_base + R_base_from_cam @ P_mj, ou R_base_from_cam a
pour colonnes les axes locaux de la camera (X,Y,Z=X×Y) exprimes dans le
repere du corps parent (deduits de xyaxes, meme convention MJCF).
"""
import argparse
import math

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

import aruco_carton_test as act


def _parse_xyaxes(s: str) -> np.ndarray:
    v = [float(x) for x in s.split()]
    x_axis, y_axis = np.array(v[0:3]), np.array(v[3:6])
    z_axis = np.cross(x_axis, y_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


class CartonPosePublisher(Node):
    def __init__(self, args):
        super().__init__("carton_pose_publisher")
        self.marker_id = args.marker_id
        self.marker_size = args.marker_size
        half = self.marker_size / 2.0
        self.object_points = np.array([
            [-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0],
        ], dtype=np.float64)

        aruco_dict = cv2.aruco.getPredefinedDictionary(args.dict)
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        self.camera_matrix = None
        self.dist_coeffs = None

        self.cam_pos_base = np.array([float(x) for x in args.camera_pos.split()])
        self.r_base_from_cam = _parse_xyaxes(args.camera_xyaxes)

        self.create_subscription(Image, args.image_topic, self._on_image, 10)
        self.create_subscription(CameraInfo, args.camera_info_topic, self._on_camera_info, 10)
        self.pose_pub = self.create_publisher(PoseStamped, args.pose_topic, 10)
        self.get_logger().info(
            f"carton_pose_publisher pret -- {args.image_topic} -> {args.pose_topic} "
            f"(repere {args.frame_id})"
        )

    def _on_camera_info(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def _on_image(self, msg: Image):
        if self.camera_matrix is None:
            return
        try:
            frame = act._image_msg_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e))
            return

        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None or self.marker_id not in ids.flatten():
            return
        idx = list(ids.flatten()).index(self.marker_id)

        ok, rvec, tvec = cv2.solvePnP(
            self.object_points, corners[idx][0], self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok or not np.all(np.isfinite(tvec)):
            # SOLVEPNP_IPPE_SQUARE degenere (NaN) quand le marqueur est vu
            # quasi pile de face (robot centre devant le carton, cas typique
            # juste apres un stand() bien aligne) -- constate empiriquement
            # le 19/08, reproductible aussi sur aruco_carton_test.py, pas
            # specifique a ce script. SOLVEPNP_ITERATIVE n'a pas cette
            # singularite (raffinement iteratif, pas de solution analytique
            # fermee sur le cas plan/frontal) -- utilise en repli.
            ok, rvec, tvec = cv2.solvePnP(
                self.object_points, corners[idx][0], self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok or not np.all(np.isfinite(tvec)):
                return

        tvec_cv = tvec.flatten()
        p_mj = np.array([tvec_cv[0], -tvec_cv[1], -tvec_cv[2]])
        p_base = self.cam_pos_base + self.r_base_from_cam @ p_mj

        r_cam_from_marker_cv, _ = cv2.Rodrigues(rvec)
        # Meme conversion d'axes que pour la position, appliquee a la rotation.
        flip = np.diag([1.0, -1.0, -1.0])
        r_cam_mj_from_marker = flip @ r_cam_from_marker_cv
        r_base_from_marker = self.r_base_from_cam @ r_cam_mj_from_marker
        quat = _rotmat_to_quat(r_base_from_marker)

        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "LINK_BASE"
        out.pose.position.x, out.pose.position.y, out.pose.position.z = p_base
        out.pose.orientation.w, out.pose.orientation.x, out.pose.orientation.y, out.pose.orientation.z = quat
        self.pose_pub.publish(out)


def _rotmat_to_quat(r: np.ndarray):
    """Shepperd's method, evite les divisions instables pres des singularites."""
    trace = np.trace(r)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r[2, 1] - r[1, 2]) * s
        y = (r[0, 2] - r[2, 0]) * s
        z = (r[1, 0] - r[0, 1]) * s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return w, x, y, z


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-topic", default="/camera/pelvis_sim/image_raw")
    parser.add_argument("--camera-info-topic", default=None)
    parser.add_argument("--marker-id", type=int, default=act.DEFAULT_MARKER_ID)
    parser.add_argument("--marker-size", type=float, required=True)
    parser.add_argument("--dict", type=int, default=act.DEFAULT_DICT)
    parser.add_argument("--pose-topic", default="/vision/carton_pose_base")
    parser.add_argument("--frame-id", default="LINK_BASE")
    parser.add_argument("--camera-pos", default="0.08 0 0.05",
                         help="Position (m) de la camera dans le repere robot -- defaut = camera "
                              "fantome (shadow_scene.CAMERA_POS).")
    parser.add_argument("--camera-xyaxes", default="0 -1 0 0 0 1",
                         help="xyaxes (convention MJCF) de la camera dans le repere robot -- defaut "
                              "= camera fantome (shadow_scene.CAMERA_XYAXES).")
    args, ros_args = parser.parse_known_args()
    if args.camera_info_topic is None:
        args.camera_info_topic = args.image_topic.replace("image_raw", "camera_info")

    rclpy.init(args=ros_args)
    node = CartonPosePublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
