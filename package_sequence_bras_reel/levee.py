"""Pince et souleve un carton reel avec les deux mains du PM01."""
import argparse
import os
import sys
import time

import numpy as np
import rclpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lever import Lever
from motion_state import ensure_motion_state

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot_arm_ik"))
from lift_carton import (
    LEFT_CHAIN, RIGHT_CHAIN, HAND_OFFSET_LEFT, HAND_OFFSET_RIGHT, solve_arm_ik, ease,
    forward_kinematics, mirror_left_to_right,
)

LEFT_JOINT_INDICES = [13, 14, 15, 16, 17]
RIGHT_JOINT_INDICES = [18, 19, 20, 21, 22]
WRIST_CHAIN_INDEX = 4

Q_LEFT_HOME = np.array([0.000879, 0.075284, -0.000233, -0.126397, -0.000033])
Q_RIGHT_HOME = np.array([0.000885, -0.075161, 0.000241, -0.126390, 0.000033])

WAYPOINT_Q_LEFT = np.radians([30.0, 5.0, 0.0, -110.0, 0.0])
WAYPOINT_Q_RIGHT = np.radians([30.0, -5.0, 0.0, -110.0, 0.0])

LEFT_HIP_PITCH_INDEX = 0
RIGHT_HIP_PITCH_INDEX = 6
LEFT_KNEE_PITCH_INDEX = 3
RIGHT_KNEE_PITCH_INDEX = 9
LEFT_ANKLE_PITCH_INDEX = 4
RIGHT_ANKLE_PITCH_INDEX = 10
WALK_STANCE_HIP_PITCH_L = np.radians(-6.9)
WALK_STANCE_HIP_PITCH_R = np.radians(-5.2)
WALK_STANCE_KNEE_L = np.radians(11.7)
WALK_STANCE_KNEE_R = np.radians(10.5)
WALK_STANCE_ANKLE_PITCH_L = np.radians(-4.8)
WALK_STANCE_ANKLE_PITCH_R = np.radians(-5.2)

RATE_HZ = 65

PINCH_X = 0.216
PINCH_Y = 0.22
SQUEEZE_Y = 0.13
LIFT_Z = 0.15
APPROACH_DURATION = 1.8
WAYPOINT_DURATION = 2.5
APPROACH_LIFT_DURATION = 0.7
SQUEEZE_DURATION = 2.5
LIFT_DURATION = 3.0
HOLD_SECONDS = 0.0
RELEASE_RAMP_SECONDS = 4.0
MOTION_STATE_TIMEOUT = 3.0

ELBOW_YAW_ROTATION_DEG = 0.0

WALK_STANCE_SCALE = 0.0
WALK_STANCE_STIFFNESS_SCALE = 1.8
WALK_STANCE_DURATION = 2.2

LEVEE_STIFFNESS = 130.0
LEVEE_DAMPING = 3.0


def _checkpoint(message, confirm):
    print(f"[ETAPE] {message}", flush=True)
    if confirm:
        input("        Verifie le robot, puis Entree pour continuer (Ctrl+C pour arreter)... ")


def _rotate_xy(point, yaw_offset):
    x, y, z = point
    c, s = np.cos(yaw_offset), np.sin(yaw_offset)
    return np.array([x * c - y * s, x * s + y * c, z])


def _quintic_ease(t):
    t = max(0.0, min(1.0, t))
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


def _bend_knees(lever, scale, stiffness_scale, duration, rate_hz=RATE_HZ, dry_run=False):
    if dry_run:
        print(f"    [dry-run] flexion genoux -- scale={scale} duration={duration}s")
        return
    for idx, kp, kd in [
        (LEFT_HIP_PITCH_INDEX, 200.0, 5.0), (RIGHT_HIP_PITCH_INDEX, 200.0, 5.0),
        (LEFT_KNEE_PITCH_INDEX, 450.0, 5.0), (RIGHT_KNEE_PITCH_INDEX, 450.0, 5.0),
        (LEFT_ANKLE_PITCH_INDEX, 400.0, 2.0), (RIGHT_ANKLE_PITCH_INDEX, 400.0, 2.0),
    ]:
        lever.set_gains(idx, kp * stiffness_scale, kd * stiffness_scale)
    n = max(1, int(duration * rate_hz))
    for i in range(n + 1):
        a = _quintic_ease(i / n)
        lever[LEFT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_L)
        lever[RIGHT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_R)
        lever[LEFT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_L)
        lever[RIGHT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_R)
        lever[LEFT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_L)
        lever[RIGHT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_R)
        time.sleep(1.0 / rate_hz)


def _straighten_knees(lever, scale, stiffness_scale, duration, rate_hz=RATE_HZ, dry_run=False):
    if dry_run:
        print(f"    [dry-run] redressement genoux -- scale={scale} duration={duration}s")
        return
    n = max(1, int(duration * rate_hz))
    for i in range(n + 1):
        a = 1.0 - _quintic_ease(i / n)
        lever[LEFT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_L)
        lever[RIGHT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_R)
        lever[LEFT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_L)
        lever[RIGHT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_R)
        lever[LEFT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_L)
        lever[RIGHT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_R)
        time.sleep(1.0 / rate_hz)


def _publish(lever, qL, qR):
    lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))


def move_arms(lever, qL0, qL1, qR0, qR1, duration, dry_run=False):
    n = max(1, int(duration * RATE_HZ))
    for i in range(n + 1):
        a = ease(i / n)
        qL = qL0 + a * (qL1 - qL0)
        qR = qR0 + a * (qR1 - qR0)
        if dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i/RATE_HZ:.2f}s  qL={np.round(qL, 4)}  qR={np.round(qR, 4)}")
            continue
        _publish(lever, qL, qR)
        time.sleep(1.0 / RATE_HZ)
    return qL, qR


def cartesian_ramp(lever, q_init, wrist_rotation, start, end, anchor_start, anchor_end,
                    duration, dry_run):
    qL = q_init.copy()
    n = max(1, int(duration * RATE_HZ))
    for i in range(n + 1):
        a = ease(i / n)
        target = start + a * (end - start)
        anchor = (1.0 - a) * anchor_start + a * anchor_end
        qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, target, qL,
                           lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation, iters=10,
                           null_space_pref=anchor)
        qR = mirror_left_to_right(qL)
        if dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i/RATE_HZ:.2f}s  qL={np.round(qL, 4)}  qR={np.round(qR, 4)}")
            continue
        _publish(lever, qL, qR)
        time.sleep(1.0 / RATE_HZ)
    return qL, mirror_left_to_right(qL)


def _build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pinch-x", type=float, default=PINCH_X)
    parser.add_argument("--pinch-z", type=float, default=0.05)
    parser.add_argument("--pinch-yaw-offset", type=float, default=0.0)
    parser.add_argument("--wrist-rotation-deg", type=float, default=ELBOW_YAW_ROTATION_DEG)
    parser.add_argument("--walk-stance-scale", type=float, default=WALK_STANCE_SCALE)
    parser.add_argument("--skip-motion-state", action="store_true")
    parser.add_argument("--only-phase", choices=["approche", "serrage", "levee"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-confirm", action="store_true")
    return parser


def run_lift_sequence(node, lever, args):
    confirm = not args.no_confirm
    wrist_rotation = np.radians(getattr(args, "wrist_rotation_deg", ELBOW_YAW_ROTATION_DEG))
    walk_stance_scale = getattr(args, "walk_stance_scale", WALK_STANCE_SCALE)

    straight_pref_L = WAYPOINT_Q_LEFT.copy()
    straight_pref_L[3] = 0.0
    q_pinch_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                              _rotate_xy([args.pinch_x, PINCH_Y, args.pinch_z], args.pinch_yaw_offset),
                              Q_LEFT_HOME, lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                              null_space_pref=straight_pref_L)
    q_pinch_R = mirror_left_to_right(q_pinch_L)

    squeeze_L = _rotate_xy([args.pinch_x, SQUEEZE_Y, args.pinch_z], args.pinch_yaw_offset)
    q_squeeze_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_pinch_L,
                                lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                                null_space_pref=q_pinch_L)
    q_squeeze_R = mirror_left_to_right(q_squeeze_L)

    run_approche = args.only_phase in (None, "approche")
    run_serrage = args.only_phase in (None, "serrage")
    run_levee = args.only_phase in (None, "levee")

    qL, qR = Q_LEFT_HOME, Q_RIGHT_HOME

    if run_approche and walk_stance_scale > 0:
        _checkpoint(f"flexion genoux -- scale={walk_stance_scale}, {WALK_STANCE_DURATION}s", confirm)
        _bend_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE, WALK_STANCE_DURATION,
                    dry_run=args.dry_run)

    if run_approche:
        _checkpoint(f"point de passage -- coudes vers l'arriere, {WAYPOINT_DURATION}s", confirm)
        qL, qR = move_arms(lever, Q_LEFT_HOME, WAYPOINT_Q_LEFT, Q_RIGHT_HOME, WAYPOINT_Q_RIGHT,
                            WAYPOINT_DURATION, dry_run=args.dry_run)

        pinch_target_L = _rotate_xy([args.pinch_x, PINCH_Y, args.pinch_z], args.pinch_yaw_offset)
        waypoint_hand_L = forward_kinematics(LEFT_CHAIN, HAND_OFFSET_LEFT, WAYPOINT_Q_LEFT)
        raised_point_L = np.array([waypoint_hand_L[0], waypoint_hand_L[1], pinch_target_L[2]])

        _checkpoint(f"approche -- ajustement vertical, {APPROACH_LIFT_DURATION}s", confirm)
        qL, qR = cartesian_ramp(lever, WAYPOINT_Q_LEFT, wrist_rotation, waypoint_hand_L,
                                 raised_point_L, WAYPOINT_Q_LEFT, WAYPOINT_Q_LEFT,
                                 APPROACH_LIFT_DURATION, args.dry_run)

        _checkpoint(
            f"approche -- ligne horizontale vers pinch (x={args.pinch_x} y=+-{PINCH_Y} "
            f"z={args.pinch_z}, poignet pivote de {np.degrees(wrist_rotation):.0f}deg), "
            f"{APPROACH_DURATION}s", confirm)
        qL, qR = cartesian_ramp(lever, qL, wrist_rotation, raised_point_L, pinch_target_L,
                                 WAYPOINT_Q_LEFT, straight_pref_L, APPROACH_DURATION, args.dry_run)

    if run_serrage:
        _checkpoint(
            f"serrage -- Y +-{PINCH_Y} -> +-{SQUEEZE_Y}, {SQUEEZE_DURATION}s "
            "(LE CONTACT AVEC LE CARTON COMMENCE ICI)", confirm)
        qL, qR = move_arms(lever, q_pinch_L, q_squeeze_L, q_pinch_R, q_squeeze_R,
                            SQUEEZE_DURATION, dry_run=args.dry_run)

    if run_levee:
        _checkpoint(
            f"levee -- Z {args.pinch_z} -> {LIFT_Z}, {LIFT_DURATION}s puis maintien "
            f"{HOLD_SECONDS}s", confirm)
        if not args.dry_run:
            for idx in LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES:
                lever.set_gains(idx, stiffness=LEVEE_STIFFNESS, damping=LEVEE_DAMPING)

        lift_start = _rotate_xy([args.pinch_x, SQUEEZE_Y, args.pinch_z], args.pinch_yaw_offset)
        lift_end = _rotate_xy([args.pinch_x, SQUEEZE_Y, LIFT_Z], args.pinch_yaw_offset)
        qL, qR = cartesian_ramp(lever, q_squeeze_L, wrist_rotation, lift_start, lift_end,
                                 q_squeeze_L, q_squeeze_L, LIFT_DURATION, args.dry_run)

        if not args.dry_run:
            for _ in range(max(1, int(HOLD_SECONDS * RATE_HZ))):
                _publish(lever, qL, qR)
                time.sleep(1.0 / RATE_HZ)

    if not args.dry_run and args.only_phase in (None, "levee"):
        if run_levee and walk_stance_scale > 0:
            _checkpoint("redressement genoux -- avant relachement", confirm)
            _straighten_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE,
                               WALK_STANCE_DURATION, dry_run=args.dry_run)
        _checkpoint("release() -- rend la main au runner pd_stand actif", confirm)
        if run_levee:
            n = max(1, int(RELEASE_RAMP_SECONDS * RATE_HZ))
            for i in range(n + 1):
                lever.set_weight(1.0 - i / n)
                _publish(lever, qL, qR)
                time.sleep(1.0 / RATE_HZ)
        lever.release()

    print("[INFO] Sequence terminee.")


def main():
    args = _build_arg_parser().parse_args()

    node = lever = None
    if not args.dry_run:
        rclpy.init()
        node = rclpy.create_node("levee")

        if not args.skip_motion_state:
            ok = ensure_motion_state(node, "lower_body_balance", timeout=MOTION_STATE_TIMEOUT)
            if not ok:
                print("[ERREUR] impossible de passer en lower_body_balance.", flush=True)
                node.destroy_node()
                rclpy.shutdown()
                return

        lever = Lever(node)

    try:
        run_lift_sequence(node, lever, args)
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
