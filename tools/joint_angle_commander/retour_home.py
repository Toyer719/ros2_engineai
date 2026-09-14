"""Ramene les bras (deja a la position 'pinch' laissee par levee.py/levee_pivot.py
--only-phase approche, PAS de serrage) vers Q_HOME puis relache -- pour abandonner
proprement un essai apres --only-phase approche sans enchainer serrage/levee.

NE PAS utiliser si le carton est deja serre (--only-phase serrage ou plus loin) --
ceci suppose les bras a la position pinch (ouverte), pas au contact du carton.

AUCUNE SIMULATION -- envoie de vraies commandes au robot.
"""
import os
import sys
import time

import numpy as np
import rclpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lever import Lever
from levee import (
    LEFT_CHAIN, RIGHT_CHAIN, HAND_OFFSET_LEFT, HAND_OFFSET_RIGHT,
    Q_LEFT_HOME, Q_RIGHT_HOME, PINCH_X, PINCH_Y, WRIST_CHAIN_INDEX,
    _rotate_xy, solve_arm_ik, move_arms,
)

RETREAT_DURATION = 3.0


def main():
    pinch_z = float(sys.argv[1]) if len(sys.argv) > 1 else 0.106
    wrist_rotation = 0.0

    rclpy.init()
    node = rclpy.create_node("retour_home")
    lever = Lever(node)

    q_pinch_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                              _rotate_xy([PINCH_X, PINCH_Y, pinch_z], 0.0),
                              Q_LEFT_HOME, lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation)
    q_pinch_R = solve_arm_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT,
                              _rotate_xy([PINCH_X, -PINCH_Y, pinch_z], 0.0),
                              Q_RIGHT_HOME, lock_index=WRIST_CHAIN_INDEX, lock_angle=-wrist_rotation)

    print("[ETAPE] retour bras home depuis la position pinch...", flush=True)
    move_arms(lever, q_pinch_L, Q_LEFT_HOME, q_pinch_R, Q_RIGHT_HOME, RETREAT_DURATION)

    print("[ETAPE] relachement.", flush=True)
    lever.release()

    node.destroy_node()
    rclpy.shutdown()
    print("[INFO] Termine.", flush=True)


if __name__ == "__main__":
    main()
