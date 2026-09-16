"""Enregistre en continu (x, y, z, roll, pitch, yaw) du bassin (LCM sim_state)
dans un CSV -- pour diagnostiquer une perte d'equilibre pendant une sequence
sans deviner : lancer ce script AVANT le ros2 launch, le laisser tourner
pendant toute la sequence, puis inspecter le CSV (z qui chute, roll/pitch qui
s'ecartent de 0 = signe de bascule)."""
import argparse
import csv
import math
import sys
import time

sys.path.insert(0, "/home/equansrobotic/stagiaire_1/tools/robot_arm_ik")
from lift_carton import SimStateListener


def _rpy_from_quaternion(w, x, y, z):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    return roll, pitch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/sim_pose_telemetry.csv")
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    args = parser.parse_args()

    listener = SimStateListener()
    print(f"[log_sim_pose] attente du premier message sim_state...", flush=True)
    if not listener.wait_for_first_message(10.0):
        print("[log_sim_pose] ERREUR : aucun message sim_state apres 10s -- run_mujoco.sh actif ?")
        return
    print(f"[log_sim_pose] enregistrement -> {args.out} pendant {args.duration:.0f}s", flush=True)

    t0 = time.time()
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg"])
        while time.time() - t0 < args.duration:
            state = listener._latest
            if state is not None:
                x, y, z = state.base_link_position
                w, qx, qy, qz = state.base_link_quaternion
                roll, pitch = _rpy_from_quaternion(w, qx, qy, qz)
                _, _, _, yaw = listener.pose()
                writer.writerow([
                    round(time.time() - t0, 3), round(x, 4), round(y, 4), round(z, 4),
                    round(math.degrees(roll), 2), round(math.degrees(pitch), 2),
                    round(math.degrees(yaw), 2),
                ])
            time.sleep(1.0 / args.rate_hz)
    print("[log_sim_pose] termine.", flush=True)


if __name__ == "__main__":
    main()
