import signal
import sys
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

sys.path.insert(0, "/home/equansrobotic/stagiaire_1/tools/joint_angle_commander")
from lever import Lever

sys.path.insert(0, "/home/equansrobotic/stagiaire_1/tools/robot_arm_ik")
from lift_carton import (
    LEFT_CHAIN, RIGHT_CHAIN, HAND_OFFSET_LEFT, HAND_OFFSET_RIGHT, solve_ik, ease,
)

from virtual_gamepad_interfaces.action import Pivot

LEFT_JOINT_INDICES = [13, 14, 15, 16, 17]
RIGHT_JOINT_INDICES = [18, 19, 20, 21, 22]
WAIST_JOINT_INDEX = 12
LEG_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

Q_LEFT_HOME = np.array([0.000879, 0.075284, -0.000233, -0.126397, -0.000033])
Q_RIGHT_HOME = np.array([0.000885, -0.075161, 0.000241, -0.126390, 0.000033])

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

LEG_JOINTS = [
    (LEFT_HIP_PITCH_INDEX, WALK_STANCE_HIP_PITCH_L, 200.0, 5.0),
    (RIGHT_HIP_PITCH_INDEX, WALK_STANCE_HIP_PITCH_R, 200.0, 5.0),
    (LEFT_KNEE_PITCH_INDEX, WALK_STANCE_KNEE_L, 450.0, 5.0),
    (RIGHT_KNEE_PITCH_INDEX, WALK_STANCE_KNEE_R, 450.0, 5.0),
    (LEFT_ANKLE_PITCH_INDEX, WALK_STANCE_ANKLE_PITCH_L, 400.0, 2.0),
    (RIGHT_ANKLE_PITCH_INDEX, WALK_STANCE_ANKLE_PITCH_R, 400.0, 2.0),
]


def _rotate_xy(point, yaw_offset):
    """Identique a lift.py::_rotate_xy -- tourne (X,Y) autour de Z."""
    x, y, z = point
    c, s = np.cos(yaw_offset), np.sin(yaw_offset)
    return np.array([x * c - y * s, x * s + y * c, z])


class PivotActionServer(Node):
    def __init__(self):
        super().__init__("pivot")
        self._lever = None
        self._server = ActionServer(
            self, Pivot, "pivot", self._execute, cancel_callback=self._on_cancel,
        )

    def _on_cancel(self, goal_handle):
        return CancelResponse.ACCEPT

    def _ensure_lever(self) -> Lever:
        if self._lever is None:
            self.get_logger().info(
                "Attente d'un abonne sur /motion/joint_override_command (run.sh actif ?)..."
            )
            self._lever = Lever(self, subscriber_timeout=10.0)
        return self._lever

    def _publish_pose(self, lever, qL, qR, waist_angle, leg_targets, stiffness_scale):
        """Republie EN UN SEUL message (via Lever, qui agrege tout ce qui a
        deja ete touche) bras+jambes+buste -- voir note en tete de fichier."""
        for idx, angle in zip(LEFT_JOINT_INDICES, qL):
            lever[idx] = float(angle)
        for idx, angle in zip(RIGHT_JOINT_INDICES, qR):
            lever[idx] = float(angle)
        for idx, target, kp, kd in leg_targets:
            lever.set_gains(idx, kp * stiffness_scale, kd * stiffness_scale)
            lever[idx] = float(target)
        lever[WAIST_JOINT_INDEX] = float(waist_angle)

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

        pinch_L = _rotate_xy([g.pinch_x, g.pinch_y, g.pinch_z], g.pinch_yaw_offset)
        pinch_R = _rotate_xy([g.pinch_x, -g.pinch_y, g.pinch_z], g.pinch_yaw_offset)
        q_pinch_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, pinch_L, Q_LEFT_HOME)
        q_pinch_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT, pinch_R, Q_RIGHT_HOME)
        squeeze_L = _rotate_xy([g.pinch_x, g.squeeze_y, g.pinch_z], g.pinch_yaw_offset)
        squeeze_R = _rotate_xy([g.pinch_x, -g.squeeze_y, g.pinch_z], g.pinch_yaw_offset)
        q_squeeze_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_pinch_L)
        q_squeeze_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT, squeeze_R, q_pinch_R)
        n_replay = max(1, int(3.0 * 30))
        for i in range(n_replay + 1):
            a = ease(i / n_replay)
            z = g.pinch_z + a * (g.lift_z - g.pinch_z)
            q_squeeze_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                                    _rotate_xy([g.pinch_x, g.squeeze_y, z], g.pinch_yaw_offset),
                                    q_squeeze_L, iters=30)
            q_squeeze_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT,
                                    _rotate_xy([g.pinch_x, -g.squeeze_y, z], g.pinch_yaw_offset),
                                    q_squeeze_R, iters=30)

        leg_targets = [
            (idx, target * g.walk_stance_scale, kp, kd) for idx, target, kp, kd in LEG_JOINTS
        ]
        lever.set_gains(WAIST_JOINT_INDEX, 150.0, 3.0)

        angle_target = np.radians(g.angle_deg)

        self.get_logger().info(
            f"pivot buste -- 0 -> {g.angle_deg:.0f}deg ({g.pivot_duration:.1f}s, "
            "bras/jambes republies en continu pour ne pas lacher le carton)"
        )
        n = max(1, int(g.pivot_duration * 30))
        for i in range(n + 1):
            a = ease(i / n)
            self._publish_pose(lever, q_squeeze_L, q_squeeze_R, a * angle_target,
                                leg_targets, g.walk_stance_stiffness_scale)
            time.sleep(1.0 / 30)

        self.get_logger().info(f"maintien pivote -- {g.hold_seconds:.1f}s")
        elapsed = 0.0
        step = 0.1
        cancelled = False
        while elapsed < g.hold_seconds:
            if goal_handle.is_cancel_requested:
                cancelled = True
                break
            self._publish_pose(lever, q_squeeze_L, q_squeeze_R, angle_target,
                                leg_targets, g.walk_stance_stiffness_scale)
            time.sleep(step)
            elapsed += step

        do_depivot = g.depivot_before_release or cancelled
        if do_depivot:
            self.get_logger().info(f"depivot -- {g.angle_deg:.0f}deg -> 0deg ({g.pivot_duration:.1f}s)")
            n = max(1, int(g.pivot_duration * 30))
            for i in range(n + 1):
                a = ease(i / n)
                self._publish_pose(lever, q_squeeze_L, q_squeeze_R, (1.0 - a) * angle_target,
                                    leg_targets, g.walk_stance_stiffness_scale)
                time.sleep(1.0 / 30)
            final_waist = 0.0
        else:
            final_waist = angle_target

        if cancelled or g.release_after:
            self.get_logger().info(f"relachement -- rampe {g.release_ramp_seconds:.1f}s")
            if g.release_ramp_seconds > 0:
                n = max(1, int(g.release_ramp_seconds * 30))
                for i in range(n + 1):
                    lever.set_weight(1.0 - i / n)
                    self._publish_pose(lever, q_squeeze_L, q_squeeze_R, final_waist,
                                        leg_targets, g.walk_stance_stiffness_scale)
                    time.sleep(1.0 / 30)
            lever.release()
        elif g.free_legs_for_walk:
            self.get_logger().info(
                "liberation des jambes (marche) -- bras/buste restent tenus a poids plein"
            )
            self._publish_pose(lever, q_squeeze_L, q_squeeze_R, final_waist,
                                leg_targets, g.walk_stance_stiffness_scale)
            lever.untouch(LEG_INDICES)
        else:
            self.get_logger().info(
                "release_after=false -- pas de relachement, bras/jambes restent tenus "
                "(un autre node est cense suivre)."
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
