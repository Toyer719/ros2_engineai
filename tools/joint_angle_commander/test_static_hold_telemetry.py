"""Diagnostic vibration : le bras oscille-t-il meme en MAINTIEN STATIQUE (consigne
fixe, vitesse visee nulle) ? Isole le phenomene de la rampe/du 30Hz en logguant
position/vitesse/couple (/hardware/joint_state) pendant un hold prolonge, separement
de la montee -- meme mouvement que move_arms_demo.py (2 epaules, -30deg), juste
instrumente.

Usage (sur Nezha, run.sh deja lance) :
    python3 test_static_hold_telemetry.py
"""
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

sys.path.insert(0, "/home/user/projects/arm_test")
from interface_protocol.msg import JointState  # noqa: E402
from lever import Lever  # noqa: E402
from motion_state import ensure_motion_state  # noqa: E402

LEFT_SHOULDER_PITCH = 13
RIGHT_SHOULDER_PITCH = 18

TARGET_ANGLE_RAD = np.radians(-30.0)
RAMP_SECONDS = 3.0
HOLD_SECONDS = 10.0
RELEASE_RAMP_SECONDS = 2.0
RATE_HZ = 30


def _ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def main():
    rclpy.init()
    node = rclpy.create_node("test_static_hold_telemetry")

    latest = {"position": None, "velocity": None, "torque": None}

    def _on_joint_state(msg):
        latest["position"] = list(msg.position)
        latest["velocity"] = list(msg.velocity)
        latest["torque"] = list(msg.torque)

    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )
    node.create_subscription(JointState, "/hardware/joint_state", _on_joint_state, qos)

    print("Bascule en lower_body_balance...", flush=True)
    ensure_motion_state(node, "lower_body_balance")

    print("Connexion a /motion/joint_override_command...", flush=True)
    lever = Lever(node, subscriber_timeout=10.0)

    print(f"Montee progressive ({RAMP_SECONDS:.0f}s)...", flush=True)
    n = int(RAMP_SECONDS * RATE_HZ)
    for i in range(n + 1):
        a = _ease(i / n)
        angle = a * TARGET_ANGLE_RAD
        lever[LEFT_SHOULDER_PITCH] = float(angle)
        lever[RIGHT_SHOULDER_PITCH] = float(angle)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(1.0 / RATE_HZ)

    print(f"Maintien statique {HOLD_SECONDS:.0f}s -- consigne FIXE, "
          "echantillonnage telemetrie en cours (/hardware/joint_state ~500Hz)...",
          flush=True)
    samples = []
    t0 = time.time()
    while time.time() - t0 < HOLD_SECONDS:
        rclpy.spin_once(node, timeout_sec=0.002)
        if latest["position"] is not None:
            samples.append((
                time.time() - t0,
                latest["position"][LEFT_SHOULDER_PITCH],
                latest["position"][RIGHT_SHOULDER_PITCH],
                latest["velocity"][LEFT_SHOULDER_PITCH],
                latest["velocity"][RIGHT_SHOULDER_PITCH],
                latest["torque"][LEFT_SHOULDER_PITCH],
                latest["torque"][RIGHT_SHOULDER_PITCH],
            ))

    print("Relachement progressif...", flush=True)
    n = int(RELEASE_RAMP_SECONDS * RATE_HZ)
    for i in range(n + 1):
        lever.set_weight(1.0 - i / n)
        time.sleep(1.0 / RATE_HZ)
    lever.release()

    node.destroy_node()
    rclpy.shutdown()

    if len(samples) < 5:
        print("Pas assez d'echantillons recus -- verifier /hardware/joint_state.", flush=True)
        return

    arr = np.array(samples)
    pos_deg = np.degrees(arr[:, 1:3])
    vel = arr[:, 3:5]
    tor = arr[:, 5:7]

    print(f"\n{len(samples)} echantillons sur {arr[-1, 0]:.1f}s\n", flush=True)
    for name, idx, i in (("gauche", LEFT_SHOULDER_PITCH, 0), ("droit", RIGHT_SHOULDER_PITCH, 1)):
        print(f"Epaule {name} (idx {idx}) pendant le maintien statique :")
        print(f"  position : {pos_deg[:, i].mean():+.2f}deg moy, "
              f"ecart-type {pos_deg[:, i].std():.3f}deg, "
              f"pic-a-pic {pos_deg[:, i].ptp():.3f}deg")
        print(f"  vitesse  : ecart-type {vel[:, i].std():.4f} rad/s, "
              f"pic-a-pic {vel[:, i].ptp():.4f} rad/s")
        print(f"  couple   : {tor[:, i].mean():+.2f}Nm moy, "
              f"ecart-type {tor[:, i].std():.3f}Nm, "
              f"pic-a-pic {tor[:, i].ptp():.3f}Nm")
        print(flush=True)


if __name__ == "__main__":
    main()
