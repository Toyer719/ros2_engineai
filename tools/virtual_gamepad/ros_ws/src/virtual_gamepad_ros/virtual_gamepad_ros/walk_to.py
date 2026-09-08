"""ActionServer "walk_to". Deux modes, choisis au demarrage (`--real` sur la ligne de
commande, cf. main()) :

- SIM (defaut, inchange) : emulation manette LCM (LB+B puis sticks sur les topics
  /virtual_gamepad/cmd/*, agreges par chef_node.py) -- `forward`/`turn` a l'echelle
  stick -1..1.
- REEL (`--real`) : /motion/body_vel_cmd (BodyVelCmd), meme mecanisme QUE
  tools/joint_angle_commander/marche.py (bascule rl_terrain -> publie vitesse -> vitesse
  zero -> retour lower_body_balance) -- confirme sur le robot reel le 2026-08-27/28,
  REMPLACE l'emulation LCM ci-dessus qui est refusee par l'arbitre de commande cote reel
  (barriere de securite volontaire, voir memoire projet). `forward`/`turn` deviennent des
  m/s et rad/s REELS, PAS l'echelle stick du mode sim -- chef_node.py calibre
  actuellement pour le mode sim (WALK_STICK, STICK_TO_MPS...), ces valeurs NE
  TRANSPOSENT PAS telles quelles au reel (cf. marche.py::DEFAULT_FORWARD_MPS=0.45,
  calibration separee et deja empirique). JAMAIS TESTE sur le robot reel via ce fichier
  -- valider comme marche.py (distance/duree tres faible, --no-confirm jamais en premier
  essai, quelqu'un pret a couper) avant tout usage.

Le mode reel a besoin, EN PLUS du setup.bash de ce ros_ws, de l'overlay SDK
(`build/ros2_env/install/local_setup.bash`, meme prerequis que lift.py -- import
interface_protocol.msg.BodyVelCmd/MotionState*) et de motion_state switching -- NE PEUT
PAS reutiliser tools/joint_angle_commander/motion_state.py::ensure_motion_state() tel
quel : cette fonction appelle rclpy.spin_once(node, ...) en interne, correct pour un
script standalone (marche.py) mais dangereux ici -- ce node tourne deja sous un
MultiThreadedExecutor (voir main()) qui spin le MEME node dans un thread separe pendant
l'execution du goal ; spin_once() en plus, depuis le thread d'execution du goal,
créerait une ressource concurrente sur le node (meme categorie de race C que celle deja
documentee dans chef_node.py::main()). La bascule d'etat est donc reimplementee ci-dessous
(_switch_motion_state) : sub/pub crees UNE FOIS dans _init_real(), le callback (delivre
par l'executor, thread separe) met juste a jour self._motion_state, et l'attente se fait
par time.sleep() pur, jamais spin_once().
"""
import signal
import sys
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Float32

from virtual_gamepad_interfaces.action import WalkTo
from virtual_gamepad_ros.field_topics import field_topic

REAL_WALK_MOTION_STATE = "rl_terrain"
REAL_RATE_HZ = 100.0
REAL_ZERO_PUBLISH_CYCLES = 10
REAL_MOTION_STATE_TIMEOUT = 3.0
REAL_MOTION_STATE_DETOUR = "pd_stand"


class WalkToActionServer(Node):
    def __init__(self, real: bool = False):
        super().__init__("walk_to")
        self._real = real

        self._lb_pub = self.create_publisher(Bool, field_topic("LB"), 10)
        self._b_pub = self.create_publisher(Bool, field_topic("B"), 10)
        self._left_x_pub = self.create_publisher(Float32, field_topic("LEFT_STICK_X"), 10)
        self._right_y_pub = self.create_publisher(Float32, field_topic("RIGHT_STICK_Y"), 10)

        if self._real:
            self._init_real()

        self._server = ActionServer(
            self, WalkTo, "walk_to", self._execute, cancel_callback=self._on_cancel,
        )

    def _on_cancel(self, goal_handle):
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        forward = goal_handle.request.forward
        turn = goal_handle.request.turn
        duration = goal_handle.request.duration
        if self._real:
            return self._execute_real(goal_handle, forward, turn, duration)
        return self._execute_sim(goal_handle, forward, turn, duration)


    def _push_sticks(self, forward: float, turn: float) -> None:
        self._left_x_pub.publish(Float32(data=float(forward)))
        self._right_y_pub.publish(Float32(data=float(-turn)))

    def _enter_walk(self, combo_hold_seconds=0.5, walk_settle_seconds=1.5) -> None:
        self.get_logger().info("Passage en walk...")
        self._lb_pub.publish(Bool(data=True))
        self._b_pub.publish(Bool(data=True))
        time.sleep(combo_hold_seconds)
        self._lb_pub.publish(Bool(data=False))
        self._b_pub.publish(Bool(data=False))
        time.sleep(walk_settle_seconds)
        self.get_logger().info("walk actif.")

    def _execute_sim(self, goal_handle, forward, turn, duration):
        result = WalkTo.Result()
        self._enter_walk()

        self.get_logger().info(f"walk_to (sim) : forward={forward} turn={turn} duration={duration}s")
        self._push_sticks(forward, turn)

        elapsed = 0.0
        step = 0.1
        while elapsed < duration:
            if goal_handle.is_cancel_requested:
                self._push_sticks(0.0, 0.0)
                goal_handle.canceled()
                result.success = False
                return result
            time.sleep(step)
            elapsed += step

        self._push_sticks(0.0, 0.0)
        goal_handle.succeed()
        result.success = True
        return result


    def _init_real(self):
        from interface_protocol.msg import BodyVelCmd, MotionState, MotionStateRequest
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        self._BodyVelCmd = BodyVelCmd
        self._MotionStateRequest = MotionStateRequest

        state_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE)
        request_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.VOLATILE)
        vel_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                              durability=DurabilityPolicy.VOLATILE)

        self._motion_state = {"current": "", "available": []}
        self.create_subscription(MotionState, "/motion/motion_state", self._on_motion_state, state_qos)
        self._motion_state_pub = self.create_publisher(MotionStateRequest, "/motion/set_motion_state", request_qos)
        self._vel_pub = self.create_publisher(BodyVelCmd, "/motion/body_vel_cmd", vel_qos)

        self.get_logger().warn(
            "walk_to demarre en mode REEL (/motion/body_vel_cmd) -- forward/turn sont des "
            "m/s et rad/s reels, PAS l'echelle stick -1..1 du mode sim. JAMAIS teste sur le "
            "robot reel via ce fichier -- recalibrer forward comme marche.py "
            "(0.12m/s confirme trop faible, 0.45m/s confirme fonctionnel) avant tout essai, "
            "en commencant par une duree tres courte."
        )

    def _on_motion_state(self, msg) -> None:
        self._motion_state["current"] = msg.current_motion_task
        self._motion_state["available"] = list(msg.available_transition_motions)

    def _wait_for_motion_state(self, timeout=1.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._motion_state["current"]:
                return
            time.sleep(0.05)

    def _switch_motion_state(self, name: str, timeout: float) -> bool:
        self._wait_for_motion_state()
        if self._motion_state["current"] == name:
            return True
        if name not in self._motion_state["available"]:
            return False
        msg = self._MotionStateRequest()
        msg.target_motion_name = name
        self._motion_state_pub.publish(msg)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._motion_state["current"] == name:
                return True
            time.sleep(0.05)
        return False

    def _ensure_motion_state(self, target: str, timeout: float = REAL_MOTION_STATE_TIMEOUT) -> bool:
        """Meme logique que motion_state.py::ensure_motion_state() (detour automatique
        par REAL_MOTION_STATE_DETOUR si target n'est pas atteignable directement) --
        voir docstring module pour pourquoi c'est reimplemente ici plutot que reutilise
        tel quel."""
        self._wait_for_motion_state()
        if self._motion_state["current"] == target:
            return True
        if (target not in self._motion_state["available"]
                and REAL_MOTION_STATE_DETOUR in self._motion_state["available"]):
            if not self._switch_motion_state(REAL_MOTION_STATE_DETOUR, timeout):
                return False
        return self._switch_motion_state(target, timeout)

    def _execute_real(self, goal_handle, forward, turn, duration):
        result = WalkTo.Result()

        if not self._ensure_motion_state(REAL_WALK_MOTION_STATE):
            self.get_logger().error(f"walk_to (reel) : impossible de passer en {REAL_WALK_MOTION_STATE}.")
            goal_handle.abort()
            result.success = False
            return result

        self.get_logger().info(
            f"walk_to (reel) : forward={forward}m/s turn={turn}rad/s duration={duration}s"
        )
        period = 1.0 / REAL_RATE_HZ
        elapsed = 0.0
        cancelled = False
        while elapsed < duration:
            if goal_handle.is_cancel_requested:
                cancelled = True
                break
            msg = self._BodyVelCmd()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "body"
            msg.linear_velocity = [float(forward), 0.0]
            msg.yaw_velocity = float(turn)
            self._vel_pub.publish(msg)
            time.sleep(period)
            elapsed += period

        for _ in range(REAL_ZERO_PUBLISH_CYCLES):
            msg = self._BodyVelCmd()
            msg.linear_velocity = [0.0, 0.0]
            msg.yaw_velocity = 0.0
            self._vel_pub.publish(msg)
            time.sleep(period)

        if not self._ensure_motion_state("lower_body_balance"):
            self.get_logger().error("walk_to (reel) : impossible de repasser en lower_body_balance.")
            goal_handle.abort()
            result.success = False
            return result

        if cancelled:
            goal_handle.canceled()
            result.success = False
            return result

        goal_handle.succeed()
        result.success = True
        return result


def main():
    real = "--real" in sys.argv

    rclpy.init()
    node = WalkToActionServer(real=real)
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
