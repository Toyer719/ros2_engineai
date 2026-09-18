"""Action Server "depose" pour le robot reel -- process SEPARE de
pivot_node.py, reconstruit la posture tenue depuis pinch_x/depivot_from_deg
plutot que de recevoir qL en memoire.

CRITIQUE (verifie numeriquement) : l'ancre du null-space doit etre
q_squeeze_L, la posture de serrage D'ORIGINE (calculee a g.pinch_x,
l'approche initiale) -- PAS une ancre fraiche recalculee a DEPOSE_X.
pivot_node.py garde cette MEME ancre (q_squeeze_L, figee AVANT tout
mouvement) tout au long de sa boucle fusionnee retract/pivot/extend --
une ancre differente ici donnerait un residu mesure de ~2.9deg/6mm.
Avec la bonne ancre, l'ecart tombe a 0.001deg (bruit numerique)."""
import os
import signal
import sys
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

for _candidate in (
    "/home/equansrobotic/stagiaire_1/package_sequence_bras_reel",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
):
    if os.path.isfile(os.path.join(_candidate, "levee.py")):
        sys.path.insert(0, _candidate)
        break
from lever import Lever
from levee import (
    LEFT_CHAIN, HAND_OFFSET_LEFT, WRIST_CHAIN_INDEX,
    Q_LEFT_HOME, Q_RIGHT_HOME, WAYPOINT_Q_LEFT, WAYPOINT_Q_RIGHT,
    LEFT_JOINT_INDICES, RIGHT_JOINT_INDICES,
    PINCH_Y, SQUEEZE_Y, MOTION_STATE_TIMEOUT,
    _rotate_xy, solve_arm_ik, ease, move_arms, cartesian_ramp, mirror_left_to_right,
)
from motion_state_safe import MotionStateWaiter

from virtual_gamepad_interfaces.action import Depose

WAIST_JOINT_INDEX = 12
DEPOSE_X = 0.40  # doit correspondre a pivot_node.py::DEPOSE_X
ECARTEMENT_GAP_Y = 0.025
ECARTEMENT_DURATION = 1.3
RETREAT_BACK_X = 0.19
RETREAT_BACK_DURATION = 1.3
RETREAT_DURATION = 1.0


def _publish_with_waist(lever, qL, qR, waist):
    lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES + [WAIST_JOINT_INDEX],
                     list(qL) + list(qR) + [waist])


class DeposeActionServer(Node):
    def __init__(self):
        super().__init__("depose")
        self._lever = None
        self._motion_waiter = MotionStateWaiter(self)
        self._server = ActionServer(
            self, Depose, "depose", self._execute, cancel_callback=self._on_cancel,
        )

    def _on_cancel(self, goal_handle):
        return CancelResponse.ACCEPT

    def _ensure_lever(self) -> Lever:
        if self._lever is None:
            ok = self._motion_waiter.ensure("lower_body_balance", timeout=MOTION_STATE_TIMEOUT)
            if not ok:
                raise RuntimeError("impossible de passer en lower_body_balance")
            self.get_logger().info(
                "Attente d'un abonne sur /motion/joint_override_command (run.sh actif ?)..."
            )
            self._lever = Lever(self, subscriber_timeout=10.0)
        return self._lever

    def _execute(self, goal_handle):
        result = Depose.Result()
        g = goal_handle.request

        try:
            lever = self._ensure_lever()
        except RuntimeError as exc:
            self.get_logger().error(f"depose indisponible : {exc}")
            goal_handle.abort()
            result.success = False
            return result

        # Reconstruction de la posture tenue -- CRITIQUE : l'ancre du
        # null-space doit etre q_squeeze_L, la posture de serrage
        # D'ORIGINE (calculee a g.pinch_x, l'approche initiale -- PAS
        # depose_x). C'est cette meme ancre que pivot_node.py garde FIXE
        # tout au long de sa boucle fusionnee retract/pivot/extend -- une
        # ancre recalculee a depose_x (une "aim_L" fraiche) donne un
        # ecart mesure de ~2.9deg/6mm (verifie numeriquement). Avec la
        # bonne ancre (q_squeeze_L), l'ecart tombe a 0.001deg (bruit
        # numerique) -- verification faite AVANT d'ecrire cette version.
        straight_pref_L = WAYPOINT_Q_LEFT.copy()
        straight_pref_L[3] = 0.0
        q_pinch_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                                  _rotate_xy([g.pinch_x, PINCH_Y, g.pinch_z], g.pinch_yaw_offset),
                                  Q_LEFT_HOME, lock_index=WRIST_CHAIN_INDEX, lock_angle=0.0,
                                  null_space_pref=straight_pref_L)
        squeeze_L = _rotate_xy([g.pinch_x, SQUEEZE_Y, g.pinch_z], g.pinch_yaw_offset)
        q_squeeze_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_pinch_L,
                                    lock_index=WRIST_CHAIN_INDEX, lock_angle=0.0,
                                    null_space_pref=q_pinch_L)

        hold_L = _rotate_xy([DEPOSE_X, SQUEEZE_Y, g.hold_z], g.pinch_yaw_offset)
        qL_hold = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, hold_L, q_squeeze_L,
                                lock_index=WRIST_CHAIN_INDEX, lock_angle=0.0,
                                null_space_pref=q_squeeze_L)
        qR_hold = mirror_left_to_right(qL_hold)
        waist_hold = np.radians(g.depivot_from_deg)
        qL, qR = qL_hold, qR_hold
        _publish_with_waist(lever, qL, qR, waist_hold)

        self.get_logger().info(f"depose -- Z {g.hold_z} -> {g.drop_z} ({g.depose_duration:.1f}s)")
        n = max(1, int(g.depose_duration * 30))
        q_drop_start_L = qL.copy()
        for i in range(n + 1):
            a = ease(i / n)
            z = g.hold_z + a * (g.drop_z - g.hold_z)
            qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                               np.array([hold_L[0], hold_L[1], z]), qL,
                               lock_index=WRIST_CHAIN_INDEX, lock_angle=0.0, iters=30,
                               null_space_pref=q_drop_start_L)
            qR = mirror_left_to_right(qL)
            _publish_with_waist(lever, qL, qR, waist_hold)
            time.sleep(1.0 / 30)

        self.get_logger().info(f"desserrage -- Y +-{SQUEEZE_Y} -> +-{PINCH_Y}")
        q_open_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                                 _rotate_xy([DEPOSE_X, PINCH_Y, g.drop_z], g.pinch_yaw_offset),
                                 qL, lock_index=WRIST_CHAIN_INDEX, lock_angle=0.0)
        q_open_R = mirror_left_to_right(q_open_L)
        qL, qR = move_arms(lever, qL, q_open_L, qR, q_open_R, g.tendre_duration)

        self.get_logger().info(f"ecartement -- ({ECARTEMENT_DURATION:.1f}s)")
        ecart_L = _rotate_xy([DEPOSE_X, PINCH_Y + ECARTEMENT_GAP_Y, g.drop_z], g.pinch_yaw_offset)
        q_ecart_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, ecart_L, qL,
                                  lock_index=WRIST_CHAIN_INDEX, lock_angle=0.0, null_space_pref=qL)
        q_ecart_R = mirror_left_to_right(q_ecart_L)
        qL, qR = move_arms(lever, qL, q_ecart_L, qR, q_ecart_R, ECARTEMENT_DURATION)

        self.get_logger().info(f"translation arriere -- ({RETREAT_BACK_DURATION:.1f}s)")
        n_retreat = max(1, int(RETREAT_BACK_DURATION * 30))
        q_retreat_start_L = qL.copy()
        for i in range(n_retreat + 1):
            a = ease(i / n_retreat)
            x = ecart_L[0] + a * (RETREAT_BACK_X - ecart_L[0])
            qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                               np.array([x, ecart_L[1], ecart_L[2]]), qL,
                               lock_index=WRIST_CHAIN_INDEX, lock_angle=0.0, iters=30,
                               null_space_pref=q_retreat_start_L)
            qR = mirror_left_to_right(qL)
            _publish_with_waist(lever, qL, qR, waist_hold)
            time.sleep(1.0 / 30)

        self.get_logger().info(
            f"degagement -- coudes vers l'arriere ({g.degagement_waypoint_duration:.1f}s) "
            "-- LE CARTON EST RELACHE ICI"
        )
        qL, qR = move_arms(lever, qL, WAYPOINT_Q_LEFT, qR, WAYPOINT_Q_RIGHT,
                            g.degagement_waypoint_duration)

        if g.depivot_from_deg != 0.0:
            self.get_logger().info(
                f"depivot -- {g.depivot_from_deg:.0f}deg -> 0deg ({g.depivot_duration:.1f}s), "
                "avant le retour des bras"
            )
            n_depivot = max(1, int(g.depivot_duration * 30))
            for i in range(n_depivot + 1):
                a = ease(i / n_depivot)
                waist = (1.0 - a) * waist_hold
                _publish_with_waist(lever, qL, qR, waist)
                time.sleep(1.0 / 30)

        self.get_logger().info(f"retour bras le long du corps ({RETREAT_DURATION:.1f}s)")
        qL, qR = move_arms(lever, qL, Q_LEFT_HOME, qR, Q_RIGHT_HOME, RETREAT_DURATION)

        self.get_logger().info(f"relachement final -- rampe {g.release_ramp_seconds:.1f}s")
        n2 = max(1, int(g.release_ramp_seconds * 30))
        for i in range(n2 + 1):
            lever.set_weight(1.0 - i / n2)
            lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))
            time.sleep(1.0 / 30)
        lever.release()

        goal_handle.succeed()
        result.success = True
        return result


def main():
    rclpy.init()
    node = DeposeActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
