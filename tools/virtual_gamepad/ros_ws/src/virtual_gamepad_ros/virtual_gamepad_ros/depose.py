import signal
import sys
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

sys.path.insert(0, "/home/equansrobotic/stagiaire_1/tools/joint_angle_commander")
from lever import Lever  # noqa: E402

sys.path.insert(0, "/home/equansrobotic/stagiaire_1/tools/robot_arm_ik")
from lift_carton import (  # noqa: E402
    LEFT_CHAIN, RIGHT_CHAIN, HAND_OFFSET_LEFT, HAND_OFFSET_RIGHT, solve_ik, ease,
)

from virtual_gamepad_interfaces.action import Depose  # noqa: E402

# Memes indices/constantes que lift.py/pivot.py -- voir leur en-tete pour le
# detail (JointOverrideCommand = remplacement complet, republier bras+jambes
# ensemble a chaque tick).
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


def _quintic_ease(t):
    """Identique a lift.py::_quintic_ease."""
    t = max(0.0, min(1.0, t))
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


def _rotate_xy(point, yaw_offset):
    """Identique a lift.py::_rotate_xy."""
    x, y, z = point
    c, s = np.cos(yaw_offset), np.sin(yaw_offset)
    return np.array([x * c - y * s, x * s + y * c, z])


class DeposeActionServer(Node):
    def __init__(self):
        super().__init__("depose")
        self._lever = None
        self._server = ActionServer(
            self, Depose, "depose", self._execute, cancel_callback=self._on_cancel,
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

    def _bend_knees(self, lever, scale, stiffness_scale, duration, rate_hz=30):
        """Identique a lift.py::_bend_knees -- refait ICI car walk_to (entre
        pivot et ce node) a rendu les jambes a pd_stand (jambes DROITES) le
        temps de la marche vers le 2e poste -- il faut refaire la flexion
        avant de bouger les bras, sinon meme risque de bascule que sans
        flexion du tout (cf memoire projet)."""
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

    def _release(self, lever, qL, qR, ramp_seconds, rate_hz=30):
        """Identique a lift.py::_release."""
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
        result = Depose.Result()
        g = goal_handle.request

        try:
            lever = self._ensure_lever()
        except RuntimeError as exc:
            self.get_logger().error(f"depose indisponible : {exc}")
            goal_handle.abort()
            result.success = False
            return result

        if g.walk_stance:
            self.get_logger().info(
                f"posture jambes -- droites (pd_stand, rendues pendant la marche) -> "
                f"flechies x{g.walk_stance_scale:.1f}, {g.walk_stance_duration:.1f}s"
            )
            self._bend_knees(lever, g.walk_stance_scale, g.walk_stance_stiffness_scale,
                              g.walk_stance_duration)

        # REJOUE le meme chemin que lift.py/pivot.py (Q_HOME -> pinch a
        # g.pinch_z -> serrage a g.pinch_z -> montee jusqu'a g.hold_z) au
        # lieu d'un solve_ik direct depuis Q_HOME vers la position tenue --
        # meme fix que pivot.py (24/08, "il le lache pour pivoter") : un
        # solve direct peut converger vers une solution articulaire
        # differente de celle reellement tenue, meme pour une cible
        # cartesienne identique (solveur iteratif/local sur bras redondant).
        pinch_L = _rotate_xy([g.pinch_x, g.pinch_y, g.pinch_z], g.pinch_yaw_offset)
        pinch_R = _rotate_xy([g.pinch_x, -g.pinch_y, g.pinch_z], g.pinch_yaw_offset)
        q_squeeze_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, pinch_L, Q_LEFT_HOME)
        q_squeeze_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT, pinch_R, Q_RIGHT_HOME)
        squeeze_L = _rotate_xy([g.pinch_x, g.squeeze_y, g.pinch_z], g.pinch_yaw_offset)
        squeeze_R = _rotate_xy([g.pinch_x, -g.squeeze_y, g.pinch_z], g.pinch_yaw_offset)
        q_squeeze_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_squeeze_L)
        q_squeeze_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT, squeeze_R, q_squeeze_R)
        n_replay = max(1, int(3.0 * 30))
        for i in range(n_replay + 1):
            a = ease(i / n_replay)
            z = g.pinch_z + a * (g.hold_z - g.pinch_z)
            q_squeeze_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                                    _rotate_xy([g.pinch_x, g.squeeze_y, z], g.pinch_yaw_offset),
                                    q_squeeze_L, iters=30)
            q_squeeze_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT,
                                    _rotate_xy([g.pinch_x, -g.squeeze_y, z], g.pinch_yaw_offset),
                                    q_squeeze_R, iters=30)
        # Republie immediatement cette pose (avant tout mouvement) pour ne
        # pas laisser un trou entre la derniere pose tenue par pivot.py et
        # la premiere de ce node.
        for idx, angle in zip(LEFT_JOINT_INDICES, q_squeeze_L):
            lever[idx] = float(angle)
        for idx, angle in zip(RIGHT_JOINT_INDICES, q_squeeze_R):
            lever[idx] = float(angle)

        self.get_logger().info(
            f"depose -- Z {g.hold_z:.3f} -> {g.drop_z:.3f} ({g.depose_duration:.1f}s) "
            "-- LE CARTON REDESCEND, TOUJOURS SERRE"
        )
        qL, qR = q_squeeze_L.copy(), q_squeeze_R.copy()
        n = max(1, int(g.depose_duration * 30))
        for i in range(n + 1):
            a = ease(i / n)
            z = g.hold_z + a * (g.drop_z - g.hold_z)
            qL = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                           _rotate_xy([g.pinch_x, g.squeeze_y, z], g.pinch_yaw_offset), qL, iters=30)
            qR = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT,
                           _rotate_xy([g.pinch_x, -g.squeeze_y, z], g.pinch_yaw_offset), qR, iters=30)
            for idx, angle in zip(LEFT_JOINT_INDICES, qL):
                lever[idx] = float(angle)
            for idx, angle in zip(RIGHT_JOINT_INDICES, qR):
                lever[idx] = float(angle)
            time.sleep(1.0 / 30)

        self.get_logger().info(
            f"desserrage -- Y +-{g.squeeze_y:.3f} -> +-{g.open_y:.3f} "
            f"({g.desserrage_duration:.1f}s) -- LE CARTON EST RELACHE ICI"
        )
        q_open_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                             _rotate_xy([g.pinch_x, g.open_y, g.drop_z], g.pinch_yaw_offset), qL)
        q_open_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT,
                             _rotate_xy([g.pinch_x, -g.open_y, g.drop_z], g.pinch_yaw_offset), qR)
        qL, qR = self._move_arms(lever, qL, q_open_L, qR, q_open_R, g.desserrage_duration)

        self.get_logger().info(f"degagement -- retour bras home ({g.degagement_duration:.1f}s)")
        qL, qR = self._move_arms(lever, qL, Q_LEFT_HOME, qR, Q_RIGHT_HOME, g.degagement_duration)

        self.get_logger().info(f"relachement final -- rampe {g.release_ramp_seconds:.1f}s")
        self._release(lever, qL, qR, g.release_ramp_seconds)

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
