"""Test PRUDENT : bouge UN SEUL genou de quelques degres seulement, pour
verifier si l'override d'articulation a un effet sur les JAMBES en mode
lower_body_balance (le guide officiel suggere que ce mode gere lui-meme
l'equilibre des jambes -- teste ici si l'override est simplement ignore,
ou s'il a un effet reel, AVANT de lancer une sequence complete comme
levee.py qui bouge 6 joints de jambe a ~35deg).

Prerequis : robot deja en mode lower_body_balance (voir
switch_to_target_motion_example.py --target-motion lower_body_balance).

Usage :
    python3 test_leg_override.py
"""
import sys
import time

import numpy as np
import rclpy

sys.path.insert(0, "/home/equansrobotic/stagiaire_1/tools/joint_angle_commander")
from lever import Lever  # noqa: E402

LEFT_KNEE_PITCH = 3
TARGET_ANGLE_RAD = np.radians(5.0)  # TRES petit mouvement de test
RAMP_SECONDS = 2.0
HOLD_SECONDS = 1.5
RATE_HZ = 30


def _ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def main():
    rclpy.init()
    node = rclpy.create_node("test_leg_override")

    print("Connexion a /motion/joint_override_command...", flush=True)
    lever = Lever(node, subscriber_timeout=10.0)
    print("Connecte. Test genou gauche (+5deg seulement)...", flush=True)

    n = int(RAMP_SECONDS * RATE_HZ)
    for i in range(n + 1):
        a = _ease(i / n)
        lever[LEFT_KNEE_PITCH] = float(a * TARGET_ANGLE_RAD)
        time.sleep(1.0 / RATE_HZ)

    print(f"Maintien {HOLD_SECONDS:.1f}s -- le genou a-t-il bouge ?", flush=True)
    time.sleep(HOLD_SECONDS)

    print("Relachement.", flush=True)
    lever.release()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
