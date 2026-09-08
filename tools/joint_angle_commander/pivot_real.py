"""Fait pivoter le BUSTE (J12_WAIST_YAW) du PM01 REEL, SEUL -- pas de prise de
carton, bras et jambes restent sous controle natif (Lever ne publie que les
articulations explicitement touchees, ici seulement l'index 12).

Premiere validation reelle de ce mouvement : le pivot buste n'a ete teste
qu'en simulation jusqu'ici (pivot.py, virtual_gamepad_ros, valide a 180deg
le 2026-09-08). Ce script est un port MINIMAL et ISOLE -- pas de flexion de
genoux, pas de bras leves -- pour valider la rotation elle-meme avant de la
combiner avec quoi que ce soit d'autre (charge du carton, jambes flechies).

AUCUNE SIMULATION -- envoie de vraies commandes au robot.

Prerequis :
  - Robot en mode "lower_body_balance" (bascule automatiquement, comme
    levee.py -- detour par pd_stand si besoin).
  - source /opt/ros/humble/setup.bash
    source ~/source/engineai_workspace/install/setup.bash
    export ROS_DOMAIN_ID=69 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=0

Securite : valider d'abord avec --dry-run. Angle de depart VOLONTAIREMENT
PETIT (30deg par defaut, pas 180deg comme en simu) -- augmenter
progressivement seulement si stable, jamais un grand saut. Bouton "passive"
de la telecommande = arret d'urgence (rend le robot mou, dernier recours).
"""
import argparse
import os
import sys
import time

import numpy as np
import rclpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lever import Lever
from motion_state import ensure_motion_state

WAIST_JOINT_INDEX = 12
WAIST_KP = 150.0
WAIST_KD = 3.0
RATE_HZ = 30
MOTION_STATE_TIMEOUT = 3.0


def _checkpoint(message, confirm):
    print(f"[ETAPE] {message}", flush=True)
    if confirm:
        input("        Verifie le robot, puis Entree pour continuer (Ctrl+C pour arreter)... ")


def _ease(t):
    """Meme courbe que lift_carton.py::ease, reprise ici pour ne pas dependre
    d'un import supplementaire pour une seule fonction."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def run_pivot(lever, angle_deg, pivot_duration, hold_seconds, release_ramp_seconds,
              dry_run, confirm):
    angle_target = np.radians(angle_deg)
    n = max(1, int(pivot_duration * RATE_HZ))

    if not dry_run:
        lever.set_gains(WAIST_JOINT_INDEX, WAIST_KP, WAIST_KD)

    _checkpoint(f"pivot buste -- 0 -> {angle_deg:.0f}deg ({pivot_duration:.1f}s)", confirm)
    for i in range(n + 1):
        a = _ease(i / n)
        target = a * angle_target
        if dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i / RATE_HZ:.2f}s  waist={np.degrees(target):.1f}deg")
            continue
        lever[WAIST_JOINT_INDEX] = float(target)
        time.sleep(1.0 / RATE_HZ)

    _checkpoint(f"maintien pivote -- {hold_seconds:.1f}s", confirm)
    if not dry_run:
        for _ in range(max(1, int(hold_seconds * RATE_HZ))):
            lever[WAIST_JOINT_INDEX] = float(angle_target)
            time.sleep(1.0 / RATE_HZ)

    _checkpoint(f"depivot -- {angle_deg:.0f}deg -> 0deg ({pivot_duration:.1f}s)", confirm)
    for i in range(n + 1):
        a = _ease(i / n)
        target = (1.0 - a) * angle_target
        if dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i / RATE_HZ:.2f}s  waist={np.degrees(target):.1f}deg")
            continue
        lever[WAIST_JOINT_INDEX] = float(target)
        time.sleep(1.0 / RATE_HZ)

    if not dry_run:
        _checkpoint(f"relachement -- rampe {release_ramp_seconds:.1f}s", confirm)
        n2 = max(1, int(release_ramp_seconds * RATE_HZ))
        for i in range(n2 + 1):
            lever.set_weight(1.0 - i / n2)
            lever[WAIST_JOINT_INDEX] = 0.0
            time.sleep(1.0 / RATE_HZ)
        lever.release()

    print("[INFO] Pivot termine.")


def _build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--angle-deg", type=float, default=30.0,
                         help="Angle de pivot (deg). PETIT par defaut (30) pour un 1er essai "
                              "reel -- PAS 180 comme en simu. Positif = vers la gauche du robot "
                              "(convention pivot.py/lift_carton_real.py). Augmenter "
                              "progressivement seulement si stable.")
    parser.add_argument("--pivot-duration", type=float, default=3.0)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--release-ramp-seconds", type=float, default=1.5)
    parser.add_argument("--skip-motion-state", action="store_true",
                         help="Suppose que le robot est deja en lower_body_balance.")
    parser.add_argument("--dry-run", action="store_true",
                         help="N'envoie rien au robot -- affiche juste les angles cibles.")
    parser.add_argument("--no-confirm", action="store_true",
                         help="Pas de pause interactive entre phases -- jamais pour un premier essai.")
    return parser


def main():
    args = _build_arg_parser().parse_args()
    confirm = not args.no_confirm

    node = lever = None
    if not args.dry_run:
        rclpy.init()
        node = rclpy.create_node("pivot_real")

        if not args.skip_motion_state:
            ok = ensure_motion_state(node, "lower_body_balance", timeout=MOTION_STATE_TIMEOUT)
            if not ok:
                print("[ERREUR] impossible de passer en lower_body_balance -- "
                      "arret (l'override du buste n'aurait aucun effet).", flush=True)
                node.destroy_node()
                rclpy.shutdown()
                return

        lever = Lever(node)

    try:
        run_pivot(lever, args.angle_deg, args.pivot_duration, args.hold_seconds,
                  args.release_ramp_seconds, args.dry_run, confirm)
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
