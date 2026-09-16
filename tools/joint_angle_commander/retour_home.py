"""Ramene les bras vers Q_HOME puis relache -- pour abandonner proprement un essai apres
--only-phase approche de levee.py/levee_pivot.py (PAS de serrage, carton pas touche).

2026-09-15 : reecrit une 2e fois -- la version precedente recalculait la posture de
depart par IK a partir de --pinch-x/--pinch-z/etc., qui devaient etre retapes a
l'identique de l'approche sous peine de saut au demarrage (bug rapporte le meme jour
apres un essai a --pinch-x 0.30 sans le repasser ici). Lit maintenant la position
REELLE des bras sur /hardware/joint_state (meme topic/pattern que
tools/vision/record_arms_view.py) au lieu de la recalculer -- s'adapte automatiquement
a n'importe quelle valeur de --pinch-x/pinch-z/wrist-rotation utilisee pour l'approche,
plus besoin de rien repasser en argument.

Retrait en 3 temps, symetrique de l'approche de levee.py/levee_pivot.py (meme raison :
pinch et point de passage n'ont ni la meme hauteur ni le meme X/Y, une ligne "droite"
directe entre les deux n'est ni horizontale ni a vitesse raisonnable une fois la
portee eloignee du defaut) :
  1. rampe CARTESIENNE horizontale (Z constant = hauteur actuelle de la main) : position
     reelle -> point intermediaire (meme X/Y que le point de passage), ancre glissante
     tendue au depart -> repliee a l'arrivee. Duree = APPROACH_DURATION (meme vitesse
     que l'approche).
  2. rampe CARTESIENNE verticale : point intermediaire -> point de passage reel (Z seul
     change). Duree = APPROACH_LIFT_DURATION.
  3. mouvement articulaire point de passage -> Q_HOME (deja etabli ailleurs dans le
     projet pour ce trajet, loin du carton, sans risque particulier).
Puis relachement (lever.release()).

AUCUNE SIMULATION -- envoie de vraies commandes au robot.
"""
import argparse
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lever import Lever
from levee import (
    LEFT_CHAIN, RIGHT_CHAIN, HAND_OFFSET_LEFT, HAND_OFFSET_RIGHT,
    LEFT_JOINT_INDICES, RIGHT_JOINT_INDICES,
    Q_LEFT_HOME, Q_RIGHT_HOME, WAYPOINT_Q_LEFT, WRIST_CHAIN_INDEX, RATE_HZ,
    APPROACH_DURATION, APPROACH_LIFT_DURATION,
    solve_arm_ik, forward_kinematics, mirror_left_to_right, move_arms, ease, _publish,
)

from interface_protocol.msg import JointState  # noqa: E402

RETREAT_HOME_DURATION = 3.0        # point de passage -> Q_HOME, articulaire
JOINT_STATE_TIMEOUT = 5.0          # attente max d'un message /hardware/joint_state


def _read_current_arms(node):
    """Bloque jusqu'a recevoir un message /hardware/joint_state, retourne (qL, qR) reels
    -- meme topic/QoS que tools/vision/record_arms_view.py."""
    received = {"msg": None}

    def _on_joint_state(msg):
        received["msg"] = msg

    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )
    sub = node.create_subscription(JointState, "/hardware/joint_state", _on_joint_state, qos)

    t0 = time.time()
    while received["msg"] is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > JOINT_STATE_TIMEOUT:
            node.destroy_subscription(sub)
            raise RuntimeError(
                f"aucun message recu sur /hardware/joint_state apres {JOINT_STATE_TIMEOUT:.0f}s "
                "-- le robot est-il allume et le bon environnement source (ROS_DOMAIN_ID=69) ?")
    node.destroy_subscription(sub)

    position = list(received["msg"].position)
    qL = np.array([position[i] for i in LEFT_JOINT_INDICES])
    qR = np.array([position[i] for i in RIGHT_JOINT_INDICES])
    return qL, qR


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                         help="N'envoie rien, affiche juste les angles de debut/fin de chaque etape.")
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("retour_home")
    lever = Lever(node)

    print("[ETAPE] lecture de la position reelle des bras (/hardware/joint_state)...", flush=True)
    qL, qR = _read_current_arms(node)
    # Poignet fige a l'angle REEL actuel (dernier joint de la chaine, WRIST_CHAIN_INDEX)
    # pendant tout le retrait -- pas besoin de --wrist-rotation-deg, deja dans qL lu.
    wrist_rotation = float(qL[WRIST_CHAIN_INDEX])
    pinch_now_L = forward_kinematics(LEFT_CHAIN, HAND_OFFSET_LEFT, qL)
    waypoint_hand_L = forward_kinematics(LEFT_CHAIN, HAND_OFFSET_LEFT, WAYPOINT_Q_LEFT)
    raised_point_L = np.array([waypoint_hand_L[0], waypoint_hand_L[1], pinch_now_L[2]])
    print(f"    position actuelle main gauche (repere robot) : {np.round(pinch_now_L, 3)}", flush=True)

    print(f"[ETAPE] retrait -- ligne horizontale position actuelle -> point intermediaire "
          f"({APPROACH_DURATION:.1f}s)...", flush=True)
    q_start_L = qL.copy()
    n = max(1, int(APPROACH_DURATION * RATE_HZ))
    for i in range(n + 1):
        a = ease(i / n)
        target = pinch_now_L + a * (raised_point_L - pinch_now_L)
        # Ancre glissante en sens inverse de l'approche : tendue (posture reelle) au
        # depart, repliee (WAYPOINT_Q_LEFT) a l'arrivee.
        anchor_L = (1.0 - a) * q_start_L + a * WAYPOINT_Q_LEFT
        qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, target, qL,
                           lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation, iters=30,
                           null_space_pref=anchor_L)
        qR = mirror_left_to_right(qL)
        if args.dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i / RATE_HZ:.2f}s  qL={np.round(qL, 4)}  qR={np.round(qR, 4)}")
            continue
        _publish(lever, qL, qR)
        time.sleep(1.0 / RATE_HZ)

    print(f"[ETAPE] retrait -- ajustement vertical point intermediaire -> point de passage "
          f"({APPROACH_LIFT_DURATION:.1f}s)...", flush=True)
    n_lift = max(1, int(APPROACH_LIFT_DURATION * RATE_HZ))
    for i in range(n_lift + 1):
        a = ease(i / n_lift)
        target = raised_point_L + a * (waypoint_hand_L - raised_point_L)
        qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, target, qL,
                           lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation, iters=30,
                           null_space_pref=WAYPOINT_Q_LEFT)
        qR = mirror_left_to_right(qL)
        if args.dry_run:
            if i in (0, n_lift):
                print(f"    [dry-run] t={i / RATE_HZ:.2f}s  qL={np.round(qL, 4)}  qR={np.round(qR, 4)}")
            continue
        _publish(lever, qL, qR)
        time.sleep(1.0 / RATE_HZ)

    print(f"[ETAPE] retour bras home depuis le point de passage ({RETREAT_HOME_DURATION:.1f}s)...",
          flush=True)
    qL, qR = move_arms(lever, qL, Q_LEFT_HOME, qR, Q_RIGHT_HOME, RETREAT_HOME_DURATION,
                        dry_run=args.dry_run)

    if not args.dry_run:
        print("[ETAPE] relachement.", flush=True)
        lever.release()

    node.destroy_node()
    rclpy.shutdown()
    print("[INFO] Termine.", flush=True)


if __name__ == "__main__":
    main()
