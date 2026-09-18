"""Action Server "lift" pour le robot reel -- meme architecture que la
simulation (tools/virtual_gamepad/.../lift.py) : process separe, son propre
Lever, appele par un orchestrateur (chef_reel.py) via ros2 Action au lieu
d'un appel de fonction direct. Reutilise les memes constantes/fonctions
que package_sequence_bras_reel/levee.py (move_arms, cartesian_ramp,
geometrie) -- seule la coordination (Action Server, gestion de
release_after pour laisser pivot_node.py prendre le relais) est propre a
ce fichier."""
import os
import signal
import sys
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

# PC dev : package_sequence_bras_reel est un dossier frere de celui-ci.
# Nezha (/home/user/projects/arm_test) : levee.py/lever.py sont a plat, au
# meme niveau que ce fichier une fois deploye dans un sous-dossier -- donc
# le parent direct. On detecte lequel des deux existe plutot que de forcer
# un chemin en dur incompatible avec l'un des deux environnements.
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
    PINCH_Y, SQUEEZE_Y, LEVEE_STIFFNESS, LEVEE_DAMPING, MOTION_STATE_TIMEOUT,
    _rotate_xy, solve_arm_ik, move_arms, cartesian_ramp, _publish, mirror_left_to_right,
)

from motion_state_safe import MotionStateWaiter

from virtual_gamepad_interfaces.action import Lift


class LiftActionServer(Node):
    def __init__(self):
        super().__init__("lift")
        self._lever = None
        # Cree UNE FOIS ici (thread principal, avant que l'executor ne
        # spinne) -- voir motion_state_safe.py pour pourquoi ce n'est PAS
        # motion_state.py::ensure_motion_state (spin_once() dangereux
        # depuis le thread d'execution d'un but, sous MultiThreadedExecutor).
        self._motion_waiter = MotionStateWaiter(self)
        self._server = ActionServer(
            self, Lift, "lift", self._execute, cancel_callback=self._on_cancel,
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
        result = Lift.Result()
        g = goal_handle.request

        try:
            lever = self._ensure_lever()
        except RuntimeError as exc:
            self.get_logger().error(f"lift indisponible : {exc}")
            goal_handle.abort()
            result.success = False
            return result

        run_approche = g.only_phase in ("", "approche")
        run_serrage = g.only_phase in ("", "serrage")
        run_levee = g.only_phase in ("", "levee")

        # 2026-09-17 : posture tenue recalculee ICI depuis pinch_x/pinch_z
        # (pas de vision/sim_state sur le robot reel -- ces valeurs sont
        # deja dans le repere du bassin, transmises telles quelles par
        # chef_reel.py). Meme chaine straight_pref_L -> q_pinch_L ->
        # q_squeeze_L que levee.py::run_lift_sequence -- reproduite ici
        # bit-a-bit identique pour que pivot_node.py/depose_node.py (des
        # process SEPARES) reconvergent sur la meme posture (voir leur
        # docstring pour la verification numerique).
        straight_pref_L = WAYPOINT_Q_LEFT.copy()
        straight_pref_L[3] = 0.0
        q_pinch_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                                  _rotate_xy([g.pinch_x, PINCH_Y, g.pinch_z], g.pinch_yaw_offset),
                                  Q_LEFT_HOME, lock_index=WRIST_CHAIN_INDEX, lock_angle=0.0,
                                  null_space_pref=straight_pref_L)
        q_pinch_R = mirror_left_to_right(q_pinch_L)
        squeeze_L = _rotate_xy([g.pinch_x, SQUEEZE_Y, g.pinch_z], g.pinch_yaw_offset)
        q_squeeze_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_pinch_L,
                                    lock_index=WRIST_CHAIN_INDEX, lock_angle=0.0,
                                    null_space_pref=q_pinch_L)
        q_squeeze_R = mirror_left_to_right(q_squeeze_L)

        qL, qR = Q_LEFT_HOME, Q_RIGHT_HOME

        if run_approche:
            self.get_logger().info("point de passage -- coudes vers l'arriere")
            qL, qR = move_arms(lever, Q_LEFT_HOME, WAYPOINT_Q_LEFT, Q_RIGHT_HOME, WAYPOINT_Q_RIGHT,
                                g.waypoint_duration)
            self.get_logger().info(f"approche -- x={g.pinch_x} y=+-{PINCH_Y} z={g.pinch_z}")
            qL, qR = move_arms(lever, WAYPOINT_Q_LEFT, q_pinch_L, WAYPOINT_Q_RIGHT, q_pinch_R,
                                g.approach_duration)

        if run_serrage:
            self.get_logger().info(f"serrage -- Y +-{PINCH_Y} -> +-{SQUEEZE_Y}")
            qL, qR = move_arms(lever, q_pinch_L, q_squeeze_L, q_pinch_R, q_squeeze_R,
                                g.squeeze_duration)

        if not run_levee:
            self.get_logger().info(
                f"only_phase={g.only_phase!r} termine -- bras tenus, pas de levee."
            )
            goal_handle.succeed()
            result.success = True
            return result

        if g.only_phase == "levee":
            _publish(lever, q_squeeze_L, q_squeeze_R)
            qL, qR = q_squeeze_L, q_squeeze_R

        self.get_logger().info(f"levee -- Z {g.pinch_z} -> {g.lift_z} ({g.lift_duration:.1f}s)")
        for idx in LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES:
            lever.set_gains(idx, stiffness=LEVEE_STIFFNESS, damping=LEVEE_DAMPING)
        lift_start = _rotate_xy([g.pinch_x, SQUEEZE_Y, g.pinch_z], g.pinch_yaw_offset)
        lift_end = _rotate_xy([g.pinch_x, SQUEEZE_Y, g.lift_z], g.pinch_yaw_offset)
        qL, qR = cartesian_ramp(lever, q_squeeze_L, 0.0, lift_start, lift_end,
                                 q_squeeze_L, q_squeeze_L, g.lift_duration, False)

        self.get_logger().info(f"maintien -- {g.hold_seconds:.1f}s")
        elapsed = 0.0
        step = 0.1
        cancelled = False
        while elapsed < g.hold_seconds:
            if goal_handle.is_cancel_requested:
                cancelled = True
                break
            _publish(lever, qL, qR)
            time.sleep(step)
            elapsed += step

        if not g.release_after and not cancelled:
            self.get_logger().info(
                "release_after=false -- bras/jambes restent tenus (pivot est cense suivre)."
            )
            goal_handle.succeed()
            result.success = True
            return result

        self.get_logger().info(f"relachement -- rampe {g.release_ramp_seconds:.1f}s")
        n = max(1, int(g.release_ramp_seconds * 30))
        for i in range(n + 1):
            lever.set_weight(1.0 - i / n)
            _publish(lever, qL, qR)
            time.sleep(1.0 / 30)
        lever.release()

        if cancelled:
            goal_handle.canceled()
            result.success = False
            return result

        goal_handle.succeed()
        result.success = True
        return result


def main():
    rclpy.init()
    node = LiftActionServer()
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
