"""Capture rapide : sauvegarde une frame annotee (detection ArUco si presente) sur
disque, pour visualiser le retour camera sans passer par Foxglove/X11 -- juste
lire le PNG produit. Reutilise ArucoCartonLocator telle quelle (meme detection que
aruco_carton_test.py), ne change aucun comportement existant."""
import argparse
import sys
import time

import cv2
import rclpy

from aruco_carton_test import ArucoCartonLocator, DEFAULT_DICT, DEFAULT_IMAGE_TOPIC, DEFAULT_MARKER_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--camera-info-topic", default=None)
    parser.add_argument("--marker-id", type=int, default=DEFAULT_MARKER_ID)
    parser.add_argument("--marker-size", type=float, default=0.20,
                         help="Cote reel du motif noir/blanc (metres) -- 0.20 mesure au "
                              "double-decimetre le 2026-09-01 (le motif seul, PAS la marge "
                              "blanche), remplace la valeur theorique 0.1016 jamais bonne.")
    parser.add_argument("--dict", type=int, default=DEFAULT_DICT)
    parser.add_argument("--out", default="/tmp/camera_snapshot.png")
    parser.add_argument("--window-s", type=float, default=4.0)
    args = parser.parse_args()
    if args.camera_info_topic is None:
        args.camera_info_topic = args.image_topic.replace("image_raw", "camera_info")
    args.publish_debug = True

    rclpy.init()
    node = ArucoCartonLocator(args)
    t0 = time.time()
    while time.time() - t0 < args.window_s:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()

    if node.last_frame is None:
        print("[ERREUR] aucune frame recue -- le node camera tourne-t-il ?")
        sys.exit(1)

    cv2.imwrite(args.out, node.last_frame)
    if node.last_distance is not None:
        print(f"[OK] frame sauvegardee : {args.out} -- marqueur detecte, distance={node.last_distance:.3f}m")
    else:
        print(f"[OK] frame sauvegardee : {args.out} -- aucun marqueur detecte sur cette frame")


if __name__ == "__main__":
    main()
