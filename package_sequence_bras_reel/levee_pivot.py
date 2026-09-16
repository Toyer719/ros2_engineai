"""Prise du carton reel, pivot du buste, puis depose -- un seul process/Lever partage."""
import argparse
import sys
import os
import time

import numpy as np
import rclpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lever import Lever
from motion_state import ensure_motion_state
from levee import (
    APPROACH_LIFT_DURATION,
    LEFT_CHAIN, RIGHT_CHAIN, HAND_OFFSET_LEFT, HAND_OFFSET_RIGHT,
    LEFT_JOINT_INDICES, RIGHT_JOINT_INDICES, Q_LEFT_HOME, Q_RIGHT_HOME,
    WAYPOINT_Q_LEFT, WAYPOINT_Q_RIGHT, WAYPOINT_DURATION, WRIST_CHAIN_INDEX,
    PINCH_X, PINCH_Y, SQUEEZE_Y, LIFT_Z, APPROACH_DURATION, SQUEEZE_DURATION,
    LIFT_DURATION, HOLD_SECONDS, RATE_HZ, WALK_STANCE_SCALE,
    WALK_STANCE_STIFFNESS_SCALE, WALK_STANCE_DURATION, LEVEE_STIFFNESS, LEVEE_DAMPING,
    MOTION_STATE_TIMEOUT, ELBOW_YAW_ROTATION_DEG,
    _checkpoint, _rotate_xy, solve_arm_ik, _bend_knees, _straighten_knees,
    _publish, move_arms, ease, forward_kinematics, cartesian_ramp,
)
from pivot_real import WAIST_JOINT_INDEX, WAIST_KP, WAIST_KD, _ease as _pivot_ease

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot_arm_ik"))
from lift_carton import mirror_left_to_right


def _publish_with_waist(lever, qL, qR, waist):
    lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES + [WAIST_JOINT_INDEX],
                     list(qL) + list(qR) + [waist])


def run_lift_and_pivot(node, lever, args):
    confirm = not args.no_confirm
    wrist_rotation = np.radians(args.wrist_rotation_deg)
    walk_stance_scale = args.walk_stance_scale

    straight_pref_L = WAYPOINT_Q_LEFT.copy()
    straight_pref_L[3] = 0.0
    q_pinch_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                              _rotate_xy([args.pinch_x, PINCH_Y, args.pinch_z], args.pinch_yaw_offset),
                              WAYPOINT_Q_LEFT, lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
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
    run_pivot = args.only_phase in (None, "pivot")
    run_depose = args.only_phase is None

    qL, qR = Q_LEFT_HOME, Q_RIGHT_HOME

    if run_approche and walk_stance_scale > 0:
        _checkpoint(f"flexion genoux -- scale={walk_stance_scale}, {WALK_STANCE_DURATION:.1f}s", confirm)
        _bend_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE, WALK_STANCE_DURATION,
                    dry_run=args.dry_run)
        if not args.dry_run:
            time.sleep(2.0)

    if run_approche:
        _checkpoint(f"point de passage -- coudes vers l'arriere, {WAYPOINT_DURATION:.1f}s", confirm)
        qL, qR = move_arms(lever, Q_LEFT_HOME, WAYPOINT_Q_LEFT, Q_RIGHT_HOME, WAYPOINT_Q_RIGHT,
                            WAYPOINT_DURATION, dry_run=args.dry_run)

        pinch_target_L = _rotate_xy([args.pinch_x, PINCH_Y, args.pinch_z], args.pinch_yaw_offset)
        waypoint_hand_L = forward_kinematics(LEFT_CHAIN, HAND_OFFSET_LEFT, WAYPOINT_Q_LEFT)
        raised_point_L = np.array([waypoint_hand_L[0], waypoint_hand_L[1], pinch_target_L[2]])

        _checkpoint(f"approche -- ajustement vertical, {APPROACH_LIFT_DURATION:.1f}s", confirm)
        qL, qR = cartesian_ramp(lever, WAYPOINT_Q_LEFT, wrist_rotation, waypoint_hand_L,
                                 raised_point_L, WAYPOINT_Q_LEFT, WAYPOINT_Q_LEFT,
                                 APPROACH_LIFT_DURATION, args.dry_run)

        _checkpoint(f"approche -- ligne horizontale vers pinch, {APPROACH_DURATION:.1f}s", confirm)
        qL, qR = cartesian_ramp(lever, qL, wrist_rotation, raised_point_L, pinch_target_L,
                                 WAYPOINT_Q_LEFT, straight_pref_L, APPROACH_DURATION, args.dry_run)

    if run_serrage:
        _checkpoint(f"serrage -- Y +-{PINCH_Y} -> +-{SQUEEZE_Y}, {SQUEEZE_DURATION:.1f}s "
                    "(LE CONTACT AVEC LE CARTON COMMENCE ICI)", confirm)
        qL, qR = move_arms(lever, q_pinch_L, q_squeeze_L, q_pinch_R, q_squeeze_R,
                            SQUEEZE_DURATION, dry_run=args.dry_run)

    if run_levee:
        _checkpoint(f"levee -- Z {args.pinch_z} -> {LIFT_Z}, {LIFT_DURATION:.1f}s puis maintien "
                    f"{HOLD_SECONDS:.1f}s", confirm)
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

    if not run_pivot:
        print("[INFO] only_phase termine -- carton tenu, pas de pivot dans cet appel.")
        return

    retract_start = _rotate_xy([args.pinch_x, SQUEEZE_Y, LIFT_Z], args.pinch_yaw_offset)
    retract_end = _rotate_xy([args.retract_x, SQUEEZE_Y, LIFT_Z], args.pinch_yaw_offset)
    _checkpoint(f"rapproche le carton avant pivot -- X {args.pinch_x} -> {args.retract_x} "
                f"({args.retract_duration:.1f}s)", confirm)
    qL, qR = cartesian_ramp(lever, qL, wrist_rotation, retract_start, retract_end,
                             qL, qL, args.retract_duration, args.dry_run)

    angle_target = np.radians(args.angle_deg)
    n = max(1, int(args.pivot_duration * RATE_HZ))

    if not args.dry_run:
        lever.set_gains(WAIST_JOINT_INDEX, WAIST_KP, WAIST_KD)

    _checkpoint(f"pivot buste -- 0 -> {args.angle_deg:.0f}deg ({args.pivot_duration:.1f}s)", confirm)
    for i in range(n + 1):
        a = _pivot_ease(i / n)
        waist = a * angle_target
        if args.dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i / RATE_HZ:.2f}s  waist={np.degrees(waist):.1f}deg")
            continue
        _publish_with_waist(lever, qL, qR, waist)
        time.sleep(1.0 / RATE_HZ)

    _checkpoint(f"maintien pivote -- {args.hold_seconds:.1f}s", confirm)
    if not args.dry_run:
        for _ in range(max(1, int(args.hold_seconds * RATE_HZ))):
            _publish_with_waist(lever, qL, qR, angle_target)
            time.sleep(1.0 / RATE_HZ)

    depose_target = _rotate_xy([args.depose_x, SQUEEZE_Y, LIFT_Z], args.pinch_yaw_offset)
    _checkpoint(f"tend les bras pour deposer -- X {args.retract_x} -> {args.depose_x} "
                f"({args.extend_duration:.1f}s), buste encore tourne", confirm)
    qL, qR = cartesian_ramp(lever, qL, wrist_rotation, retract_end, depose_target,
                             qL, qL, args.extend_duration, args.dry_run)

    if not run_depose:
        print("[INFO] only_phase termine -- carton tenu, buste toujours tourne, pas de depose.")
        return

    drop_z = LIFT_Z - args.pre_release_drop
    if args.pre_release_drop > 0:
        _checkpoint(f"baisse avant relachement -- Z {LIFT_Z} -> {drop_z:.3f} "
                    f"({args.pre_release_drop_duration:.1f}s), buste encore tourne", confirm)
        drop_start = _rotate_xy([args.depose_x, SQUEEZE_Y, LIFT_Z], args.pinch_yaw_offset)
        drop_end = _rotate_xy([args.depose_x, SQUEEZE_Y, drop_z], args.pinch_yaw_offset)
        qL, qR = cartesian_ramp(lever, qL, wrist_rotation, drop_start, drop_end, qL, qL,
                                 args.pre_release_drop_duration, args.dry_run)

    _checkpoint(f"desserrage -- Y +-{SQUEEZE_Y} -> +-{PINCH_Y} ({args.open_duration:.1f}s) "
                "-- LE CARTON EST RELACHE ICI, buste encore tourne", confirm)
    q_open_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                             _rotate_xy([args.depose_x, PINCH_Y, drop_z], args.pinch_yaw_offset),
                             qL, lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation)
    q_open_R = mirror_left_to_right(q_open_L)
    qL, qR = move_arms(lever, qL, q_open_L, qR, q_open_R, args.open_duration, dry_run=args.dry_run)

    _checkpoint(f"ecartement -- Y +-{PINCH_Y} -> +-{PINCH_Y + args.ecartement_gap_y:.3f} "
                f"({args.ecartement_duration:.1f}s)", confirm)
    ecart_L = _rotate_xy([args.depose_x, PINCH_Y + args.ecartement_gap_y, drop_z], args.pinch_yaw_offset)
    q_ecart_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, ecart_L, qL,
                              lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                              null_space_pref=qL)
    q_ecart_R = mirror_left_to_right(q_ecart_L)
    qL, qR = move_arms(lever, qL, q_ecart_L, qR, q_ecart_R, args.ecartement_duration,
                        dry_run=args.dry_run)

    _checkpoint(f"translation arriere -- X {args.depose_x} -> {args.retreat_back_x} "
                f"({args.retreat_back_duration:.1f}s), Y reste large, main ramenee pres du corps",
                confirm)
    retreat_end = np.array([args.retreat_back_x, ecart_L[1], ecart_L[2]])
    qL, qR = cartesian_ramp(lever, qL, wrist_rotation, ecart_L, retreat_end, qL, qL,
                             args.retreat_back_duration, args.dry_run)

    _checkpoint(f"degagement -- coudes vers l'arriere ({args.degagement_waypoint_duration:.1f}s)",
                confirm)
    qL, qR = move_arms(lever, qL, WAYPOINT_Q_LEFT, qR, WAYPOINT_Q_RIGHT,
                        args.degagement_waypoint_duration, dry_run=args.dry_run)

    _checkpoint(f"retour bras le long du corps ({args.retreat_duration:.1f}s), "
                "avant le depivot", confirm)
    qL, qR = move_arms(lever, qL, Q_LEFT_HOME, qR, Q_RIGHT_HOME, args.retreat_duration,
                        dry_run=args.dry_run)

    _checkpoint(f"depivot -- {args.angle_deg:.0f}deg -> 0deg ({args.pivot_duration:.1f}s), "
                "CARTON DEJA LACHE, bras replies -- buste seul", confirm)
    for i in range(n + 1):
        a = _pivot_ease(i / n)
        waist = (1.0 - a) * angle_target
        if args.dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i / RATE_HZ:.2f}s  waist={np.degrees(waist):.1f}deg")
            continue
        lever[WAIST_JOINT_INDEX] = float(waist)
        time.sleep(1.0 / RATE_HZ)

    if not args.dry_run and walk_stance_scale > 0:
        _checkpoint("redressement genoux -- avant relachement final", confirm)
        _straighten_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE,
                           WALK_STANCE_DURATION, dry_run=args.dry_run)

    if not args.dry_run:
        _checkpoint(f"relachement final -- rampe {args.release_ramp_seconds:.1f}s", confirm)
        n2 = max(1, int(args.release_ramp_seconds * RATE_HZ))
        for i in range(n2 + 1):
            lever.set_weight(1.0 - i / n2)
            _publish_with_waist(lever, qL, qR, 0.0)
            time.sleep(1.0 / RATE_HZ)
        lever.release()

    print("[INFO] Sequence terminee.")


def _build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pinch-x", type=float, default=PINCH_X)
    parser.add_argument("--pinch-z", type=float, default=0.05)
    parser.add_argument("--pinch-yaw-offset", type=float, default=0.0)
    parser.add_argument("--wrist-rotation-deg", type=float, default=ELBOW_YAW_ROTATION_DEG)
    parser.add_argument("--walk-stance-scale", type=float, default=WALK_STANCE_SCALE)
    parser.add_argument("--angle-deg", type=float, default=20.0)
    parser.add_argument("--pivot-duration", type=float, default=4.0)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--retract-x", type=float, default=0.19)
    parser.add_argument("--depose-x", type=float, default=0.40)
    parser.add_argument("--retract-duration", type=float, default=1.0)
    parser.add_argument("--extend-duration", type=float, default=1.0)
    parser.add_argument("--release-ramp-seconds", type=float, default=1.5)
    parser.add_argument("--pre-release-drop", type=float, default=0.03)
    parser.add_argument("--pre-release-drop-duration", type=float, default=2.0)
    parser.add_argument("--depose-duration", type=float, default=4.0)
    parser.add_argument("--open-duration", type=float, default=2.0)
    parser.add_argument("--ecartement-gap-y", type=float, default=0.025)
    parser.add_argument("--ecartement-duration", type=float, default=2.0)
    parser.add_argument("--retreat-back-x", type=float, default=0.19)
    parser.add_argument("--retreat-back-duration", type=float, default=2.0)
    parser.add_argument("--degagement-waypoint-duration", type=float, default=2.0)
    parser.add_argument("--retreat-duration", type=float, default=2.0)
    parser.add_argument("--skip-motion-state", action="store_true")
    parser.add_argument("--only-phase", choices=["approche", "serrage", "levee", "pivot"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-confirm", action="store_true")
    return parser


def main():
    args = _build_arg_parser().parse_args()

    node = lever = None
    if not args.dry_run:
        rclpy.init()
        node = rclpy.create_node("levee_pivot")

        if not args.skip_motion_state:
            ok = ensure_motion_state(node, "lower_body_balance", timeout=MOTION_STATE_TIMEOUT)
            if not ok:
                print("[ERREUR] impossible de passer en lower_body_balance -- arret.", flush=True)
                node.destroy_node()
                rclpy.shutdown()
                return

        lever = Lever(node)

    try:
        run_lift_and_pivot(node, lever, args)
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
