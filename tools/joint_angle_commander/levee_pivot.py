"""Prise du carton REEL puis pivot du BUSTE, carton en main -- UN SEUL processus/
Lever partage entre la levee et le pivot (pas de saut entre deux scripts separes,
la posture bras deja en memoire est reutilisee directement pour le pivot).

Reutilise integralement la sequence approche/serrage/levee de levee.py (memes
constantes PINCH_X/Y, SQUEEZE_Y, LIFT_Z, LEVEE_STIFFNESS -- deja validee sur le
robot reel le 2026-09-01), puis enchaine sur le mecanisme de pivot_real.py
(deja valide buste seul jusqu'a 120deg, jambes flechies).

PREMIERE COMBINAISON JAMAIS TESTEE : pivoter en tenant reellement le carton
n'a encore jamais ete essaye, ni en simu (retire le 08/09 apres chute au
redressement des jambes, dans un scenario DIFFERENT -- marche en tenant,
pas pivot statique) ni sur le robot reel. Le poids du carton deplace le
centre de masse plus loin du buste que les bras seuls -- attends-toi a un
angle maximal stable PLUS PETIT qu'a vide (120deg), pas le meme.

AUCUNE SIMULATION -- envoie de vraies commandes au robot.

Prerequis : identiques a levee.py (lower_body_balance, memes exports/sources).

Securite :
  - --dry-run d'abord, toujours.
  - --angle-deg PETIT au premier essai (20deg suggere -- BEAUCOUP moins que
    les 120deg a vide) et remonter par petits paliers en observant a
    chaque fois, jamais un grand saut.
  - --only-phase approche seul, puis serrage, puis pivot -- valider chaque
    etape separement avant d'enchainer, comme pour levee.py.
  - Le carton peut tomber si l'equilibre casse pendant le pivot -- ne pas
    se tenir dans sa trajectoire de chute.
"""
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
    LEFT_CHAIN, RIGHT_CHAIN, HAND_OFFSET_LEFT, HAND_OFFSET_RIGHT,
    LEFT_JOINT_INDICES, RIGHT_JOINT_INDICES, Q_LEFT_HOME, Q_RIGHT_HOME,
    PINCH_X, PINCH_Y, SQUEEZE_Y, LIFT_Z, APPROACH_DURATION, SQUEEZE_DURATION,
    LIFT_DURATION, HOLD_SECONDS, RATE_HZ, WALK_STANCE_SCALE,
    WALK_STANCE_STIFFNESS_SCALE, WALK_STANCE_DURATION, LEVEE_STIFFNESS,
    MOTION_STATE_TIMEOUT, ELBOW_YAW_ROTATION_DEG,
    _checkpoint, _rotate_xy, _solve_ik_locked_wrist, _bend_knees, _straighten_knees,
    _publish, move_arms, ease,
)
from pivot_real import WAIST_JOINT_INDEX, WAIST_KP, WAIST_KD, _ease as _pivot_ease


def run_lift_and_pivot(node, lever, args):
    confirm = not args.no_confirm
    wrist_rotation = np.radians(args.wrist_rotation_deg)
    walk_stance_scale = args.walk_stance_scale

    q_pinch_L = _solve_ik_locked_wrist(LEFT_CHAIN, HAND_OFFSET_LEFT,
                                        _rotate_xy([PINCH_X, PINCH_Y, args.pinch_z], args.pinch_yaw_offset),
                                        Q_LEFT_HOME, wrist_rotation)
    q_pinch_R = _solve_ik_locked_wrist(RIGHT_CHAIN, HAND_OFFSET_RIGHT,
                                        _rotate_xy([PINCH_X, -PINCH_Y, args.pinch_z], args.pinch_yaw_offset),
                                        Q_RIGHT_HOME, -wrist_rotation)
    squeeze_L = _rotate_xy([PINCH_X, SQUEEZE_Y, args.pinch_z], args.pinch_yaw_offset)
    squeeze_R = _rotate_xy([PINCH_X, -SQUEEZE_Y, args.pinch_z], args.pinch_yaw_offset)
    q_squeeze_L = _solve_ik_locked_wrist(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_pinch_L, wrist_rotation)
    q_squeeze_R = _solve_ik_locked_wrist(RIGHT_CHAIN, HAND_OFFSET_RIGHT, squeeze_R, q_pinch_R, -wrist_rotation)

    run_approche = args.only_phase in (None, "approche")
    run_serrage = args.only_phase in (None, "serrage")
    run_levee = args.only_phase in (None, "levee")
    run_pivot = args.only_phase in (None, "pivot")

    qL, qR = Q_LEFT_HOME, Q_RIGHT_HOME

    if run_approche and walk_stance_scale > 0:
        _checkpoint(f"flexion genoux -- scale={walk_stance_scale}, {WALK_STANCE_DURATION:.1f}s", confirm)
        _bend_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE, WALK_STANCE_DURATION,
                    dry_run=args.dry_run)
        if not args.dry_run:
            time.sleep(2.0)

    if run_approche:
        _checkpoint(f"approche -- mains vers pinch, {APPROACH_DURATION:.1f}s", confirm)
        qL, qR = move_arms(lever, Q_LEFT_HOME, q_pinch_L, Q_RIGHT_HOME, q_pinch_R,
                            APPROACH_DURATION, dry_run=args.dry_run)

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
                lever.set_gains(idx, stiffness=LEVEE_STIFFNESS)
        qL, qR = q_squeeze_L.copy(), q_squeeze_R.copy()
        n = max(1, int(LIFT_DURATION * RATE_HZ))
        for i in range(n + 1):
            a = ease(i / n)
            z = args.pinch_z + a * (LIFT_Z - args.pinch_z)
            qL = _solve_ik_locked_wrist(LEFT_CHAIN, HAND_OFFSET_LEFT,
                                         _rotate_xy([PINCH_X, SQUEEZE_Y, z], args.pinch_yaw_offset),
                                         qL, wrist_rotation, iters=30)
            qR = _solve_ik_locked_wrist(RIGHT_CHAIN, HAND_OFFSET_RIGHT,
                                         _rotate_xy([PINCH_X, -SQUEEZE_Y, z], args.pinch_yaw_offset),
                                         qR, -wrist_rotation, iters=30)
            if args.dry_run:
                if i in (0, n):
                    print(f"    [dry-run] t={i / RATE_HZ:.2f}s  qL={np.round(qL, 4)}  qR={np.round(qR, 4)}")
                continue
            _publish(lever, qL, qR)
            time.sleep(1.0 / RATE_HZ)
        if not args.dry_run:
            for _ in range(max(1, int(HOLD_SECONDS * RATE_HZ))):
                _publish(lever, qL, qR)
                time.sleep(1.0 / RATE_HZ)

    if not run_pivot:
        print("[INFO] only_phase termine -- carton tenu, pas de pivot dans cet appel.")
        return

    angle_target = np.radians(args.angle_deg)
    n = max(1, int(args.pivot_duration * RATE_HZ))

    if not args.dry_run:
        lever.set_gains(WAIST_JOINT_INDEX, WAIST_KP, WAIST_KD)

    _checkpoint(f"pivot buste -- 0 -> {args.angle_deg:.0f}deg ({args.pivot_duration:.1f}s), "
                "CARTON EN MAIN -- republication bras+jambes+buste en continu", confirm)
    for i in range(n + 1):
        a = _pivot_ease(i / n)
        waist = a * angle_target
        if args.dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i / RATE_HZ:.2f}s  waist={np.degrees(waist):.1f}deg")
            continue
        _publish(lever, qL, qR)
        lever[WAIST_JOINT_INDEX] = float(waist)
        time.sleep(1.0 / RATE_HZ)

    _checkpoint(f"maintien pivote -- {args.hold_seconds:.1f}s", confirm)
    if not args.dry_run:
        for _ in range(max(1, int(args.hold_seconds * RATE_HZ))):
            _publish(lever, qL, qR)
            lever[WAIST_JOINT_INDEX] = float(angle_target)
            time.sleep(1.0 / RATE_HZ)

    _checkpoint(f"depivot -- {args.angle_deg:.0f}deg -> 0deg ({args.pivot_duration:.1f}s)", confirm)
    for i in range(n + 1):
        a = _pivot_ease(i / n)
        waist = (1.0 - a) * angle_target
        if args.dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i / RATE_HZ:.2f}s  waist={np.degrees(waist):.1f}deg")
            continue
        _publish(lever, qL, qR)
        lever[WAIST_JOINT_INDEX] = float(waist)
        time.sleep(1.0 / RATE_HZ)

    if not args.dry_run:
        if walk_stance_scale > 0:
            _checkpoint("redressement genoux -- avant relachement", confirm)
            _straighten_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE,
                               WALK_STANCE_DURATION, dry_run=args.dry_run)
        _checkpoint(f"relachement -- rampe {args.release_ramp_seconds:.1f}s", confirm)
        n2 = max(1, int(args.release_ramp_seconds * RATE_HZ))
        for i in range(n2 + 1):
            lever.set_weight(1.0 - i / n2)
            _publish(lever, qL, qR)
            lever[WAIST_JOINT_INDEX] = 0.0
            time.sleep(1.0 / RATE_HZ)
        lever.release()

    print("[INFO] Sequence terminee.")


def _build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pinch-z", type=float, default=-0.139)
    parser.add_argument("--pinch-yaw-offset", type=float, default=0.0)
    parser.add_argument("--wrist-rotation-deg", type=float, default=ELBOW_YAW_ROTATION_DEG)
    parser.add_argument("--walk-stance-scale", type=float, default=WALK_STANCE_SCALE)
    parser.add_argument("--angle-deg", type=float, default=20.0,
                         help="Angle de pivot (deg), CARTON EN MAIN. PETIT par defaut (20) -- "
                              "premiere combinaison jamais testee, attends-toi a un maximum "
                              "stable plus bas qu'a vide (120deg sans carton). Remonter par "
                              "petits paliers, jamais un grand saut.")
    parser.add_argument("--pivot-duration", type=float, default=6.0)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--release-ramp-seconds", type=float, default=1.5)
    parser.add_argument("--skip-motion-state", action="store_true")
    parser.add_argument("--only-phase", choices=["approche", "serrage", "levee", "pivot"], default=None,
                         help="N'execute qu'une phase puis s'arrete -- valider approche, puis "
                              "serrage, puis levee, puis pivot separement au premier essai.")
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
