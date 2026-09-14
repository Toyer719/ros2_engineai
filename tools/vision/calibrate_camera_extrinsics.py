"""Calibre la transformation camera->robot (position + orientation de la camera bassin
dans le repere LINK_BASE) necessaire a carton_pose_publisher.py pour fonctionner sur le
ROBOT REEL -- actuellement carton_pose_publisher.py utilise par defaut la position de la
camera FANTOME de la simulation (shadow_scene.CAMERA_POS/CAMERA_XYAXES), jamais la vraie.

ECRIT SANS ACCES AU ROBOT REEL (2026-09-14) -- la partie detection (`record`) n'a PAS pu
etre testee sur une vraie image camera, seule la partie maths (`solve`, Kabsch/Procrustes)
a ete validee numeriquement (voir _self_test() en bas de fichier, lancer avec --self-test).
A FAIRE avant confiance totale : lancer `record` une premiere fois et verifier que la
distance/pose affichees sont plausibles (comparer visuellement/au double-decimetre), avant
d'enchainer sur les N points de calibration.

PRINCIPE (calibration extrinseque par points de controle, pas de damier/chessboard) :
  1. Placer LE MEME marqueur ArUco (deja imprime, cf PLAN_TEST_ARUCO_CARTON.txt) a
     plusieurs positions DIFFERENTES et bien visibles par la camera bassin, en MESURANT
     a chaque fois sa position (centre du marqueur) dans le repere du robot -- meme
     origine/convention que celle deja utilisee partout ailleurs dans ce projet pour
     "bassin"/LINK_BASE (PINCH_X/PINCH_Y/pinch_z dans levee.py etc. sont TOUS relatifs a
     cette meme origine -- utiliser LA MEME, pas en choisir une nouvelle). Au moins 3
     points, IDEALEMENT 6-8+ repartis sur toute la profondeur/largeur utile (0.5m a 2m,
     large eventail lateral) pour un ajustement robuste -- 3 points exacts suffisent en
     theorie mais n'importe quelle petite erreur de mesure n'est alors pas amortie.
  2. Pour chaque position : `record --robot-x/-y/-z <mesure>` -- capture UNE detection,
     l'ajoute au fichier d'accumulation (JSON, --points-file).
  3. Une fois tous les points enregistres : `solve` -- calcule la transformation rigide
     (rotation + translation) qui fait le MEILLEUR ajustement (moindres carres, SVD) entre
     les positions marqueur vues par la camera et les positions mesurees a la main. Affiche
     --camera-pos/--camera-xyaxes prets a copier-coller dans carton_pose_publisher.py, PLUS
     l'erreur residuelle (RMS, en metres) -- si elle est grande (>quelques cm), verifier les
     mesures ou ajouter des points avant de faire confiance au resultat.

Reutilise le meme detecteur/solvePnP que aruco_carton_test.py (memes conventions : ordre des
coins, dictionnaire, conversion image sans cv_bridge) et la meme conversion d'axes
OpenCV->MJCF que carton_pose_publisher.py (X_mj=X_cv, Y_mj=-Y_cv, Z_mj=-Z_cv, verifiee a
<5mm contre la simu le 2026-08-19 -- pas revalidee sur le reel, mais c'est une conversion
d'axes fixe/geometrique, pas une mesure, donc pas de raison qu'elle differe)."""
import argparse
import json
import os
import time

import cv2
import numpy as np

import aruco_carton_test as act

DEFAULT_POINTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "camera_calibration_points.json")


def _detect_once(image_topic, camera_info_topic, marker_id, marker_size, dict_id, timeout_s=5.0):
    """S'abonne brievement (rclpy, en dehors de tout node/contexte partage -- ce script
    tourne standalone, pas dans un process ROS2 deja actif) et retourne le premier tvec_cv
    valide (repere CAMERA OpenCV) pour `marker_id`, ou None si rien recu avant `timeout_s`."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image

    half = marker_size / 2.0
    object_points = np.array([
        [-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0],
    ], dtype=np.float64)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    result = {}

    class _Once(Node):
        def __init__(self):
            super().__init__("calibrate_camera_extrinsics_once")
            self.camera_matrix = None
            self.dist_coeffs = None
            self.create_subscription(Image, image_topic, self._on_image, 10)
            self.create_subscription(CameraInfo, camera_info_topic, self._on_info, 10)

        def _on_info(self, msg):
            if self.camera_matrix is None:
                self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
                self.dist_coeffs = np.array(msg.d, dtype=np.float64)

        def _on_image(self, msg):
            if "tvec" in result or self.camera_matrix is None:
                return
            try:
                frame = act._image_msg_to_bgr(msg)
            except ValueError as e:
                self.get_logger().error(str(e))
                return
            corners, ids, _ = detector.detectMarkers(frame)
            if ids is None or marker_id not in ids.flatten():
                return
            idx = list(ids.flatten()).index(marker_id)
            ok, rvec, tvec = cv2.solvePnP(
                object_points, corners[idx][0], self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok or not np.all(np.isfinite(tvec)):
                # meme repli que carton_pose_publisher.py -- IPPE_SQUARE degenere
                # quand le marqueur est vu quasi pile de face.
                ok, rvec, tvec = cv2.solvePnP(
                    object_points, corners[idx][0], self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not ok or not np.all(np.isfinite(tvec)):
                    return
            result["tvec"] = tvec.flatten()
            result["rvec"] = rvec.flatten()

    rclpy.init()
    node = _Once()
    t0 = time.time()
    try:
        while "tvec" not in result and time.time() - t0 < timeout_s:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return result.get("tvec"), result.get("rvec")


def _cv_to_mj(tvec_cv):
    """Meme conversion que carton_pose_publisher.py (X_mj=X_cv, Y_mj=-Y_cv, Z_mj=-Z_cv)."""
    return np.array([tvec_cv[0], -tvec_cv[1], -tvec_cv[2]])


def cmd_record(args):
    tvec_cv, rvec_cv = _detect_once(
        args.image_topic, args.camera_info_topic or args.image_topic.replace("image_raw", "camera_info"),
        args.marker_id, args.marker_size, args.dict, timeout_s=args.timeout,
    )
    if tvec_cv is None:
        print(f"[ERREUR] marqueur {args.marker_id} non detecte apres {args.timeout:.1f}s -- "
              "verifier que la camera publie (ros2 topic hz), que le marqueur est bien visible, "
              "et --marker-size/--dict.")
        return

    p_mj = _cv_to_mj(tvec_cv)
    p_robot = np.array([args.robot_x, args.robot_y, args.robot_z])
    distance = float(np.linalg.norm(tvec_cv))
    print(f"[OK] marqueur detecte -- distance camera={distance:.3f}m, "
          f"p_camera_locale(MJCF)={np.round(p_mj, 4)}, p_robot(mesure)={np.round(p_robot, 4)}")

    points = _load_points(args.points_file)
    points.append({
        "p_camera_local_mj": p_mj.tolist(),
        "p_robot": p_robot.tolist(),
        "distance_m": distance,
        "timestamp": time.time(),
    })
    _save_points(args.points_file, points)
    print(f"[INFO] point {len(points)} enregistre dans {args.points_file}")


def _load_points(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _save_points(path, points):
    with open(path, "w") as f:
        json.dump(points, f, indent=2)


def cmd_clear(args):
    if os.path.exists(args.points_file):
        os.remove(args.points_file)
        print(f"[INFO] {args.points_file} supprime.")
    else:
        print(f"[INFO] {args.points_file} n'existait pas -- rien a faire.")


def kabsch_fit(p_source, p_target):
    """Transformation rigide (R, t) qui minimise sum(||R @ p_source[i] + t - p_target[i]||^2)
    -- algorithme de Kabsch/Procrustes (SVD), solution optimale au sens des moindres carres
    (PAS de fitting iteratif/gradient -- exact et deterministe pour ce type de probleme).
    p_source, p_target : (N,3) arrays, N>=3. Retourne (R (3,3), t (3,), rms_residual (float))."""
    p_source = np.asarray(p_source, dtype=np.float64)
    p_target = np.asarray(p_target, dtype=np.float64)
    assert p_source.shape == p_target.shape and p_source.shape[0] >= 3, \
        "Il faut au moins 3 points, source et target de meme forme (N,3)."

    centroid_source = p_source.mean(axis=0)
    centroid_target = p_target.mean(axis=0)
    source_c = p_source - centroid_source
    target_c = p_target - centroid_target

    H = source_c.T @ target_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1.0, 1.0, d])  # evite une reflexion (det=-1) sur des points quasi-plans
    R = Vt.T @ correction @ U.T
    t = centroid_target - R @ centroid_source

    predicted = (R @ p_source.T).T + t
    residuals = np.linalg.norm(predicted - p_target, axis=1)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return R, t, rms


def _rotmat_to_xyaxes(R):
    """Inverse de _parse_xyaxes (carton_pose_publisher.py) -- les colonnes de R sont deja
    les axes X/Y/Z locaux de la camera exprimes dans le repere robot, xyaxes = X puis Y
    (Z est deduit par produit vectoriel cote lecture, pas besoin de le donner)."""
    x_axis, y_axis = R[:, 0], R[:, 1]
    return f"{x_axis[0]:.6f} {x_axis[1]:.6f} {x_axis[2]:.6f} {y_axis[0]:.6f} {y_axis[1]:.6f} {y_axis[2]:.6f}"


def cmd_solve(args):
    points = _load_points(args.points_file)
    if len(points) < 3:
        print(f"[ERREUR] {len(points)} point(s) dans {args.points_file} -- il en faut au moins 3 "
              "(6-8+ recommande pour un ajustement robuste). Utiliser `record` pour en ajouter.")
        return

    p_source = np.array([p["p_camera_local_mj"] for p in points])  # repere camera locale (MJCF)
    p_target = np.array([p["p_robot"] for p in points])            # repere robot (mesure a la main)
    R, t, rms = kabsch_fit(p_source, p_target)

    print(f"\n[RESULTAT] ajustement sur {len(points)} points, erreur residuelle RMS = {rms * 1000:.1f}mm")
    if rms > 0.03:
        print("[ATTENTION] erreur residuelle > 3cm -- verifier les mesures (points mal releves ?) "
              "ou ajouter/refaire des points avant de faire confiance a ce resultat.")
    print(f"\ncamera_pos_base (m) = {t[0]:.6f} {t[1]:.6f} {t[2]:.6f}")
    print(f"camera_xyaxes       = {_rotmat_to_xyaxes(R)}")
    print("\nA copier-coller dans carton_pose_publisher.py (ou passer en CLI) :")
    print(f"    --camera-pos \"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\" \\")
    print(f"    --camera-xyaxes \"{_rotmat_to_xyaxes(R)}\"")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "camera_pos_base": t.tolist(),
                "camera_xyaxes": _rotmat_to_xyaxes(R),
                "rotation_matrix": R.tolist(),
                "rms_residual_m": rms,
                "n_points": len(points),
            }, f, indent=2)
        print(f"\n[INFO] resultat aussi ecrit dans {args.output}")


def _self_test():
    """Valide kabsch_fit() avec des donnees synthetiques (transformation connue + bruit) --
    AUCUN besoin de robot/camera. A lancer avant de faire confiance au fichier."""
    rng = np.random.default_rng(42)

    # Transformation "verite terrain" arbitraire (camera penchee vers le bas, decalee du bassin).
    true_t = np.array([0.08, 0.0, 0.25])
    true_axis = np.array([1.0, 0.3, 0.0]); true_axis /= np.linalg.norm(true_axis)
    true_angle = np.radians(25.0)
    K = np.array([[0, -true_axis[2], true_axis[1]],
                  [true_axis[2], 0, -true_axis[0]],
                  [-true_axis[1], true_axis[0], 0]])
    true_R = np.eye(3) + np.sin(true_angle) * K + (1 - np.cos(true_angle)) * (K @ K)

    n_points = 8
    p_source = rng.uniform(-1.0, 1.0, size=(n_points, 3))
    p_target_clean = (true_R @ p_source.T).T + true_t
    noise = rng.normal(0.0, 0.002, size=p_target_clean.shape)  # 2mm de bruit de mesure simule
    p_target = p_target_clean + noise

    R_fit, t_fit, rms = kabsch_fit(p_source, p_target)

    r_err = np.degrees(np.arccos(np.clip((np.trace(true_R.T @ R_fit) - 1) / 2, -1, 1)))
    t_err = np.linalg.norm(true_t - t_fit)
    print(f"[SELF-TEST] erreur rotation={r_err:.3f}deg erreur translation={t_err*1000:.2f}mm "
          f"rms_residuel={rms*1000:.2f}mm (bruit simule=2mm)")
    ok = r_err < 1.0 and t_err < 0.01
    print("[SELF-TEST] " + ("OK -- kabsch_fit() recupere bien la transformation." if ok
                             else "ECHEC -- verifier kabsch_fit()."))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true",
                         help="Valide la partie maths (Kabsch) avec des donnees synthetiques, "
                              "sans robot/camera, puis quitte.")
    sub = parser.add_subparsers(dest="command")

    p_record = sub.add_parser("record", help="Capture une detection et l'ajoute au fichier de points.")
    p_record.add_argument("--robot-x", type=float, required=True,
                           help="Position mesuree du CENTRE du marqueur, repere robot (m) -- "
                                "meme origine/convention que PINCH_X/Y/Z ailleurs dans ce projet.")
    p_record.add_argument("--robot-y", type=float, required=True)
    p_record.add_argument("--robot-z", type=float, required=True)
    p_record.add_argument("--image-topic", default=act.DEFAULT_IMAGE_TOPIC)
    p_record.add_argument("--camera-info-topic", default=None)
    p_record.add_argument("--marker-id", type=int, default=act.DEFAULT_MARKER_ID)
    p_record.add_argument("--marker-size", type=float, required=True,
                           help="Taille REELLE mesuree du marqueur (m), pas la valeur theorique.")
    p_record.add_argument("--dict", type=int, default=act.DEFAULT_DICT)
    p_record.add_argument("--timeout", type=float, default=5.0)
    p_record.add_argument("--points-file", default=DEFAULT_POINTS_FILE)
    p_record.set_defaults(func=cmd_record)

    p_solve = sub.add_parser("solve", help="Calcule camera-pos/camera-xyaxes a partir des points enregistres.")
    p_solve.add_argument("--points-file", default=DEFAULT_POINTS_FILE)
    p_solve.add_argument("--output", default=None, help="Fichier JSON optionnel pour sauvegarder le resultat.")
    p_solve.set_defaults(func=cmd_solve)

    p_clear = sub.add_parser("clear", help="Supprime le fichier de points accumules (recommencer a zero).")
    p_clear.add_argument("--points-file", default=DEFAULT_POINTS_FILE)
    p_clear.set_defaults(func=cmd_clear)

    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
