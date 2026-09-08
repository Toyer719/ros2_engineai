import sys
import time

import numpy as np
import rclpy

sys.path.insert(0, "/home/equansrobotic/stagiaire_1/tools/joint_angle_commander")
from lever import Lever  # noqa: E402

# Numeros de joint (voir docstring de lever.py pour la liste complete) :
LEFT_SHOULDER_PITCH = 13
RIGHT_SHOULDER_PITCH = 18

TARGET_ANGLE_RAD = np.radians(-30.0)  # leve le bras vers l'avant (~30deg)
RAMP_SECONDS = 3.0   # monte LENTEMENT (mouvement progressif, pas un saut)
HOLD_SECONDS = 2.0   # maintien en position
RATE_HZ = 30


def _ease(t):
    """Interpolation douce (acceleration/deceleration en douceur), 0->1."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def main():
    rclpy.init()
    node = rclpy.create_node("move_arms_demo")

    print("Connexion a /motion/joint_override_command...", flush=True)
    lever = Lever(node, subscriber_timeout=10.0)
    print("Connecte. Montee progressive des bras...", flush=True)

    n = int(RAMP_SECONDS * RATE_HZ)
    for i in range(n + 1):
        a = _ease(i / n)
        angle = a * TARGET_ANGLE_RAD
        lever[LEFT_SHOULDER_PITCH] = float(angle)
        lever[RIGHT_SHOULDER_PITCH] = float(angle)
        time.sleep(1.0 / RATE_HZ)

    print(f"Maintien {HOLD_SECONDS:.1f}s...", flush=True)
    time.sleep(HOLD_SECONDS)

    print("Relachement (rend la main a la politique active).", flush=True)
    lever.release()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
