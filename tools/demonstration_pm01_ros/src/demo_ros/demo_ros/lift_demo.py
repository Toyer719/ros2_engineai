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

from demo_interfaces.action import Lift

LEFT_JOINT_INDICES = [13, 14, 15, 16, 17]
RIGHT_JOINT_INDICES = [18, 19, 20, 21, 22]

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

LEVEE_ARM_STIFFNESS = 90.0


def _quintic_ease(t):
    t = max(0.0, min(1.0, t))
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


def _rotate_xy(point, yaw_offset):
    x, y, z = point
    c, s = np.cos(yaw_offset), np.sin(yaw_offset)
    return np.array([x * c - y * s, x * s + y * c, z])


class LiftDemoActionServer(Node):
    def __init__(self):
        super().__init__("lift_demo")
        self._lever = None
        self._server = ActionServer(
            self, Lift, "lift_demo", self._execute, cancel_callback=self._on_cancel,
        )

    def _on_cancel(self, goal_handle):
        return CancelResponse.ACCEPT

    def _ensure_lever(self) -> Lever:
        if self._lever is None:
            self._lever = Lever(self, subscriber_timeout=10.0)
        return self._lever

    def _bend_knees(self, lever, scale, stiffness_scale, duration, rate_hz=30):
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

    def _straighten_knees(self, lever, scale, stiffness_scale, duration, rate_hz=30):
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

    def _move_arms(self, lever, qL0, qL1, qR0, qR1, duration, rate_hz=30):
        n = max(1, int(duration * rate_hz))
        for i in range(n + 1):
            a = ease(i / n)
            qL = qL0 + a * (qL1 - qL0)
            qR = qR0 + a * (qR1 - qR0)
            for idx, angle in zip(LEFT_JOINT_INDICES, qL):
                lever[idx] = float(angle)
            for idx, angle in zip(RIGHT_JOINT_INDICES, qR):
                lever[idx] = float(angle)
            time.sleep(1.0 / rate_hz)
        return qL, qR

    def _hold(self, lever, qL, qR, goal_handle, duration, rate_hz=10):
        elapsed = 0.0
        step = 1.0 / rate_hz
        while elapsed < duration:
            if goal_handle.is_cancel_requested:
                return False
            for idx, angle in zip(LEFT_JOINT_INDICES, qL):
                lever[idx] = float(angle)
            for idx, angle in zip(RIGHT_JOINT_INDICES, qR):
                lever[idx] = float(angle)
            time.sleep(step)
            elapsed += step
        return True

    def _move_z(self, lever, pinch_x, squeeze_y, pinch_yaw_offset, qL, qR, z0, z1, duration):
        n = max(1, int(duration * 30))
        for i in range(n + 1):
            a = ease(i / n)
            z = z0 + a * (z1 - z0)
            qL = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                           _rotate_xy([pinch_x, squeeze_y, z], pinch_yaw_offset), qL, iters=30)
            qR = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT,
                           _rotate_xy([pinch_x, -squeeze_y, z], pinch_yaw_offset), qR, iters=30)
            for idx, angle in zip(LEFT_JOINT_INDICES, qL):
                lever[idx] = float(angle)
            for idx, angle in zip(RIGHT_JOINT_INDICES, qR):
                lever[idx] = float(angle)
            time.sleep(1.0 / 30)
        return qL, qR

    def _release(self, lever, qL, qR, ramp_seconds, rate_hz=30):
        if ramp_seconds > 0:
            n = max(1, int(ramp_seconds * rate_hz))
            for i in range(n + 1):
                lever.set_weight(1.0 - i / n)
                for idx, angle in zip(LEFT_JOINT_INDICES, qL):
                    lever[idx] = float(angle)
                for idx, angle in zip(RIGHT_JOINT_INDICES, qR):
                    lever[idx] = float(angle)
                time.sleep(1.0 / rate_hz)
        lever.release()

    def _execute(self, goal_handle):
        result = Lift.Result()
        g = goal_handle.request
        lever = self._ensure_lever()
        lever.forget()

        if g.walk_stance:
            self._bend_knees(lever, g.walk_stance_scale, g.walk_stance_stiffness_scale,
                              g.walk_stance_duration)
            time.sleep(2.0)

        pinch_L = _rotate_xy([g.pinch_x, g.pinch_y, g.pinch_z], g.pinch_yaw_offset)
        pinch_R = _rotate_xy([g.pinch_x, -g.pinch_y, g.pinch_z], g.pinch_yaw_offset)
        q_pinch_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, pinch_L, Q_LEFT_HOME)
        q_pinch_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT, pinch_R, Q_RIGHT_HOME)

        squeeze_L = _rotate_xy([g.pinch_x, g.squeeze_y, g.pinch_z], g.pinch_yaw_offset)
        squeeze_R = _rotate_xy([g.pinch_x, -g.squeeze_y, g.pinch_z], g.pinch_yaw_offset)
        q_squeeze_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_pinch_L)
        q_squeeze_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT, squeeze_R, q_pinch_R)

        self._move_arms(lever, Q_LEFT_HOME, q_pinch_L, Q_RIGHT_HOME, q_pinch_R, g.approach_duration)
        self._hold(lever, q_pinch_L, q_pinch_R, goal_handle, 2.0)

        self._move_arms(lever, q_pinch_L, q_squeeze_L, q_pinch_R, q_squeeze_R, g.squeeze_duration)
        self._hold(lever, q_squeeze_L, q_squeeze_R, goal_handle, 2.0)

        for idx in LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES:
            lever.set_gains(idx, stiffness=LEVEE_ARM_STIFFNESS)
        qL, qR = self._move_z(lever, g.pinch_x, g.squeeze_y, g.pinch_yaw_offset,
                               q_squeeze_L.copy(), q_squeeze_R.copy(), g.pinch_z, g.lift_z, g.lift_duration)

        if not self._hold(lever, qL, qR, goal_handle, g.hold_seconds):
            self._release(lever, qL, qR, g.release_ramp_seconds)
            goal_handle.canceled()
            result.success = False
            return result

        if g.release_after:
            qL, qR = self._move_z(lever, g.pinch_x, g.squeeze_y, g.pinch_yaw_offset,
                                   qL, qR, g.lift_z, g.pinch_z, g.lift_duration)
            for idx in LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES:
                lever.set_gains(idx, stiffness=250.0)

            qL, qR = self._move_arms(lever, qL, Q_LEFT_HOME, qR, Q_RIGHT_HOME, g.approach_duration)
            self._straighten_knees(lever, g.walk_stance_scale, g.walk_stance_stiffness_scale,
                                    g.walk_stance_duration)
            self._release(lever, qL, qR, g.release_ramp_seconds)

        goal_handle.succeed()
        result.success = True
        return result


def main():
    rclpy.init()
    node = LiftDemoActionServer()
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
