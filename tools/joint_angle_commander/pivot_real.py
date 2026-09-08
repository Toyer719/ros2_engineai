"""Fait pivoter le BUSTE (J12_WAIST_YAW) du PM01 REEL -- pas de prise de
carton, seuls buste et (optionnellement) jambes sont touches (Lever ne
publie que les articulations explicitement modifiees).

Historique reel :
  - 2026-09-08 : premier essai, buste seul, jambes DROITES, --angle-deg 180
    -- le robot a perdu l'equilibre et est tombe. Cause probable : a 180deg,
    le centre de gravite (buste+bras) se deplace beaucoup, et sans flexion
    de genoux (contrairement au pivot simu valide a 180deg le meme jour,
    qui tournait avec walk_stance_scale=4.5, genoux flechis, CoG abaisse),
    la base de sustentation est trop etroite pour compenser.
  - Meme jour, --angle-deg 30 (jambes droites) : stable, robot releve sans
    dommage apparent apres la chute a 180.
  - Ajout de la flexion de genoux ci-dessous (portee de levee.py, meme
    mecanisme, meme echelle prudente WALK_STANCE_SCALE=1.0) -- jamais
    combinee avec le pivot sur le robot reel avant ce jour. Angle par
    defaut RESTE 30 (deja valide) le temps de confirmer que la flexion de
    genoux elle-meme n'introduit pas d'instabilite nouvelle, avant de
    remonter en angle.

AUCUNE SIMULATION -- envoie de vraies commandes au robot.

Prerequis :
  - Robot en mode "lower_body_balance" (bascule automatiquement, comme
    levee.py -- detour par pd_stand si besoin).
  - source /opt/ros/humble/setup.bash
    source ~/source/engineai_workspace/install/setup.bash
    export ROS_DOMAIN_ID=69 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=0

Securite : valider d'abord avec --dry-run. Ne JAMAIS sauter directement a
un grand angle -- remonter par petits pas (30 -> 45 -> 60...) en observant
a chaque fois, jamais un saut comme 30 -> 180. Bouton "passive" de la
telecommande = arret d'urgence (rend le robot mou, dernier recours).
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

# Flexion des genoux -- portee telle quelle de levee.py (memes indices,
# memes angles mesures, meme echelle prudente par defaut). Abaisse le CoG
# pendant le pivot, comme le fait pivot.py cote simu (walk_stance_scale).
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
WALK_STANCE_STIFFNESS_SCALE = 1.8
WALK_STANCE_DURATION = 3.0


def _checkpoint(message, confirm):
    print(f"[ETAPE] {message}", flush=True)
    if confirm:
        input("        Verifie le robot, puis Entree pour continuer (Ctrl+C pour arreter)... ")


def _ease(t):
    """Meme courbe que lift_carton.py::ease, reprise ici pour ne pas dependre
    d'un import supplementaire pour une seule fonction."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def _quintic_ease(t):
    """Identique a levee.py::_quintic_ease -- vitesse ET acceleration
    nulles aux deux bords, imite la transition native de pd_stand."""
    t = max(0.0, min(1.0, t))
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


def _bend_knees(lever, scale, stiffness_scale, duration, dry_run=False):
    """Identique a levee.py::_bend_knees -- jambes droites -> flechies
    (posture walk au repos x`scale`), hanche+genou+cheville COORDONNES."""
    if dry_run:
        print(f"    [dry-run] flexion genoux -- scale={scale} duration={duration}s")
        return
    for idx, kp, kd in [
        (LEFT_HIP_PITCH_INDEX, 200.0, 5.0), (RIGHT_HIP_PITCH_INDEX, 200.0, 5.0),
        (LEFT_KNEE_PITCH_INDEX, 450.0, 5.0), (RIGHT_KNEE_PITCH_INDEX, 450.0, 5.0),
        (LEFT_ANKLE_PITCH_INDEX, 400.0, 2.0), (RIGHT_ANKLE_PITCH_INDEX, 400.0, 2.0),
    ]:
        lever.set_gains(idx, kp * stiffness_scale, kd * stiffness_scale)
    n = max(1, int(duration * RATE_HZ))
    for i in range(n + 1):
        a = _quintic_ease(i / n)
        lever[LEFT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_L)
        lever[RIGHT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_R)
        lever[LEFT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_L)
        lever[RIGHT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_R)
        lever[LEFT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_L)
        lever[RIGHT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_R)
        time.sleep(1.0 / RATE_HZ)


def _straighten_knees(lever, scale, stiffness_scale, duration, dry_run=False):
    """Identique a levee.py::_straighten_knees -- inverse de _bend_knees(),
    a faire AVANT release() pour eviter un saut de posture brutal."""
    if dry_run:
        print(f"    [dry-run] redressement genoux -- scale={scale} duration={duration}s")
        return
    n = max(1, int(duration * RATE_HZ))
    for i in range(n + 1):
        a = 1.0 - _quintic_ease(i / n)
        lever[LEFT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_L)
        lever[RIGHT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_R)
        lever[LEFT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_L)
        lever[RIGHT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_R)
        lever[LEFT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_L)
        lever[RIGHT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_R)
        time.sleep(1.0 / RATE_HZ)


def run_pivot(lever, angle_deg, pivot_duration, hold_seconds, release_ramp_seconds,
              walk_stance_scale, dry_run, confirm):
    angle_target = np.radians(angle_deg)
    n = max(1, int(pivot_duration * RATE_HZ))

    if walk_stance_scale > 0:
        _checkpoint(f"flexion genoux -- scale={walk_stance_scale}, {WALK_STANCE_DURATION:.1f}s", confirm)
        _bend_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE, WALK_STANCE_DURATION,
                    dry_run=dry_run)
        if not dry_run:
            time.sleep(2.0)

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

    if walk_stance_scale > 0:
        _checkpoint("redressement genoux -- avant relachement", confirm)
        _straighten_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE,
                           WALK_STANCE_DURATION, dry_run=dry_run)

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
                         help="Angle de pivot (deg). Reste a 30 (deja valide stable, jambes "
                              "droites) tant que la flexion de genoux n'a pas ete confirmee "
                              "stable a son tour -- ne JAMAIS sauter directement a un grand "
                              "angle (180 a cause une chute le 08/09).")
    parser.add_argument("--pivot-duration", type=float, default=6.0,
                         help="Duree du pivot ALLER (et du depivot), secondes -- montee a 6.0 "
                              "(etait 3.0) apres la chute du 08/09 pour un mouvement plus lent.")
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--release-ramp-seconds", type=float, default=1.5)
    parser.add_argument("--walk-stance-scale", type=float, default=1.0,
                         help="Echelle de flexion des genoux avant/pendant le pivot -- 0 pour "
                              "desactiver (jambes droites, comportement du 1er essai). 1.0 = "
                              "posture mesuree telle quelle (meme defaut prudent que levee.py) "
                              "-- PAS ENCORE COMBINEE avec le pivot sur le robot reel avant ce "
                              "jour, valider a 30deg avant de remonter en angle.")
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
                  args.release_ramp_seconds, args.walk_stance_scale, args.dry_run, confirm)
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
