"""Action Server "pivot" pour le robot reel -- process SEPARE de lift_node.py
(son propre Lever), doit donc RECALCULER la posture tenue depuis
pinch_x/pinch_z plutot que de recevoir le qL interne de lift_node.py. Meme
piege deja rencontre et corrige en simulation (tools/virtual_gamepad/.../
pivot.py) : sans reconstruire EXACTEMENT la meme chaine de calcul
(seed WAYPOINT_Q_LEFT -> aim_L -> squeeze_L ancre sur aim_L), ce process
independant peut converger sur une branche epaule/avant-bras differente de
celle que lift_node.py tenait reellement (jusqu'a ~20deg d'ecart en
shoulder_yaw pour la meme position de main). Applique ici DES LE DEPART
(pas en correctif apres coup)."""
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
    _rotate_xy, solve_arm_ik, ease, mirror_left_to_right,
)
from pivot_real import WAIST_JOINT_INDEX, WAIST_KP, WAIST_KD, _ease as _pivot_ease
from motion_state_safe import MotionStateWaiter

from virtual_gamepad_interfaces.action import Pivot

RETRACT_X = 0.19
DEPOSE_X = 0.40
RETRACT_DURATION = 0.6
EXTEND_DURATION = 0.6
WAIST_KP_RELEASE, WAIST_KD_RELEASE = 80.0, 2.0


def _publish_with_waist(lever, qL, qR, waist):
    lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES + [WAIST_JOINT_INDEX],
                     list(qL) + list(qR) + [waist])


class PivotActionServer(Node):
    def __init__(self):
        super().__init__("pivot")
        self._lever = None
        self._motion_waiter = MotionStateWaiter(self)
        self._server = ActionServer(
            self, Pivot, "pivot", self._execute, cancel_callback=self._on_cancel,
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
        result = Pivot.Result()
        g = goal_handle.request

        try:
            lever = self._ensure_lever()
        except RuntimeError as exc:
            self.get_logger().error(f"pivot indisponible : {exc}")
            goal_handle.abort()
            result.success = False
            return result

        # Reconstruction BIT-A-BIT IDENTIQUE de la posture tenue par
        # lift_node.py -- meme chaine que dans levee.py::run_lift_sequence.
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
        q_squeeze_R = mirror_left_to_right(q_squeeze_L)

        retract_start = _rotate_xy([g.pinch_x, SQUEEZE_Y, g.lift_z], g.pinch_yaw_offset)
        retract_end = _rotate_xy([RETRACT_X, SQUEEZE_Y, g.lift_z], g.pinch_yaw_offset)
        depose_target = _rotate_xy([DEPOSE_X, SQUEEZE_Y, g.lift_z], g.pinch_yaw_offset)

        angle_target = np.radians(g.angle_deg)
        n = max(1, int(g.pivot_duration * 30))
        if RETRACT_DURATION + EXTEND_DURATION > g.pivot_duration:
            self.get_logger().error(
                f"retract+extend ({RETRACT_DURATION + EXTEND_DURATION}s) depasse "
                f"pivot_duration ({g.pivot_duration}s) -- arret.")
            goal_handle.abort()
            result.success = False
            return result

        lever.set_gains(WAIST_JOINT_INDEX, WAIST_KP, WAIST_KD)

        # Fusion rapproche/pivot/tendu -- meme technique que
        # package_sequence_bras_reel/levee_pivot.py : le buste tourne en
        # continu, la main ne bouge que pendant les 1eres/dernieres
        # secondes de la fenetre. Ancre null-space FIXE, capturee avant la
        # boucle, jamais reassignee dedans.
        anchor = q_squeeze_L.copy()
        qL, qR = q_squeeze_L, q_squeeze_R
        self.get_logger().info(
            f"pivot buste + rapproche/tend les bras -- 0 -> {g.angle_deg:.0f}deg "
            f"({g.pivot_duration:.1f}s)"
        )
        cancelled = False
        for i in range(n + 1):
            if goal_handle.is_cancel_requested:
                cancelled = True
                break
            t = i / 30.0
            waist = _pivot_ease(i / n) * angle_target
            if t < RETRACT_DURATION:
                a_arm = ease(t / RETRACT_DURATION)
                target = retract_start + a_arm * (retract_end - retract_start)
            elif t > g.pivot_duration - EXTEND_DURATION:
                a_arm = ease((t - (g.pivot_duration - EXTEND_DURATION)) / EXTEND_DURATION)
                target = retract_end + a_arm * (depose_target - retract_end)
            else:
                target = retract_end
            qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, target, qL,
                               lock_index=WRIST_CHAIN_INDEX, lock_angle=0.0, iters=10,
                               null_space_pref=anchor)
            qR = mirror_left_to_right(qL)
            _publish_with_waist(lever, qL, qR, waist)
            time.sleep(1.0 / 30)

        if not cancelled:
            self.get_logger().info(f"maintien pivote -- {g.hold_seconds:.1f}s")
            elapsed = 0.0
            step = 0.1
            while elapsed < g.hold_seconds:
                if goal_handle.is_cancel_requested:
                    cancelled = True
                    break
                _publish_with_waist(lever, qL, qR, angle_target)
                time.sleep(step)
                elapsed += step

        do_depivot = g.depivot_before_release or cancelled
        if do_depivot:
            self.get_logger().info(f"depivot -- {g.angle_deg:.0f}deg -> 0deg ({g.pivot_duration:.1f}s)")
            for i in range(n + 1):
                a = ease(i / n)
                kp = WAIST_KP + a * (WAIST_KP_RELEASE - WAIST_KP)
                kd = WAIST_KD + a * (WAIST_KD_RELEASE - WAIST_KD)
                lever.set_gains(WAIST_JOINT_INDEX, kp, kd)
                _publish_with_waist(lever, qL, qR, (1.0 - a) * angle_target)
                time.sleep(1.0 / 30)
            final_waist = 0.0
        else:
            lever.set_gains(WAIST_JOINT_INDEX, WAIST_KP_RELEASE, WAIST_KD_RELEASE)
            final_waist = angle_target

        if cancelled or g.release_after:
            self.get_logger().info(f"relachement -- rampe {g.release_ramp_seconds:.1f}s")
            n2 = max(1, int(g.release_ramp_seconds * 30))
            for i in range(n2 + 1):
                lever.set_weight(1.0 - i / n2)
                _publish_with_waist(lever, qL, qR, final_waist)
                time.sleep(1.0 / 30)
            lever.release()
        else:
            self.get_logger().info(
                "release_after=false -- bras/buste restent tenus (depose est cense suivre)."
            )

        if cancelled:
            goal_handle.canceled()
            result.success = False
            return result

        goal_handle.succeed()
        result.success = True
        return result


def main():
    rclpy.init()
    node = PivotActionServer()
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
