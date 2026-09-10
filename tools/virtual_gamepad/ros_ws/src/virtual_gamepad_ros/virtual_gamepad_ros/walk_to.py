"""ActionServer "walk_to" -- UNE SEULE logique, `/motion/body_vel_cmd` (BodyVelCmd),
meme mecanisme QUE tools/joint_angle_commander/marche.py (bascule rl_terrain ->
publie vitesse -> vitesse zero -> retour lower_body_balance) -- confirme sur le
robot reel le 2026-08-27/28. `forward`/`turn` sont des m/s et rad/s REELS.

2026-09-10 : ancien mode sim (emulation manette LCM directe, sans passer par
/motion/body_vel_cmd) SUPPRIME -- le sim/reel etait la seule branche de ce
fichier qui differait, et elle est remplacee par un node externe,
`body_vel_bridge.py`, lance UNIQUEMENT dans les launch files sim (jamais sur
le robot reel) : il traduit /motion/body_vel_cmd + /motion/set_motion_state en
l'emulation manette que MuJoCo comprend. Ce fichier n'a donc plus besoin de
savoir s'il tourne en sim ou sur le robot reel -- meme code, meme protocole,
seul ce qui ecoute de l'autre cote du topic change. Calibration vitesse->stick
du bridge PAS physiquement lineaire sur toute la plage (voir sa docstring) :
seul le point forward=0.45m/s (calibre reel, marche.py::DEFAULT_FORWARD_MPS)
est fidele en sim, chef_node.py n'utilise que celui-la.

A besoin, EN PLUS du setup.bash de ce ros_ws, de l'overlay SDK
(`build/ros2_env/install/local_setup.bash`, meme prerequis que lift.py --
import interface_protocol.msg.BodyVelCmd/MotionState*) et de motion_state
switching -- NE PEUT PAS reutiliser tools/joint_angle_commander/motion_state.py
::ensure_motion_state() tel quel : cette fonction appelle rclpy.spin_once(node,
...) en interne, correct pour un script standalone (marche.py) mais dangereux
ici -- ce node tourne deja sous un MultiThreadedExecutor (voir main()) qui
spin le MEME node dans un thread separe pendant l'execution du goal ;
spin_once() en plus, depuis le thread d'execution du goal, créerait une
ressource concurrente sur le node (meme categorie de race C que celle deja
documentee dans chef_node.py::main()). La bascule d'etat est donc
reimplementee ci-dessous (_switch_motion_state) : sub/pub crees UNE FOIS dans
__init__, le callback (delivre par l'executor, thread separe) met juste a
jour self._motion_state, et l'attente se fait par time.sleep() pur, jamais
spin_once().
"""
import signal
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from virtual_gamepad_interfaces.action import WalkTo

WALK_MOTION_STATE = "rl_terrain"
RATE_HZ = 100.0
ZERO_PUBLISH_CYCLES = 10
MOTION_STATE_TIMEOUT = 3.0
MOTION_STATE_DETOUR = "pd_stand"


class WalkToActionServer(Node):
    def __init__(self):
        super().__init__("walk_to")
        from interface_protocol.msg import BodyVelCmd, MotionState, MotionStateRequest

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

        self._server = ActionServer(
            self, WalkTo, "walk_to", self._execute, cancel_callback=self._on_cancel,
        )

    def _on_cancel(self, goal_handle):
        return CancelResponse.ACCEPT

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

    def _ensure_motion_state(self, target: str, timeout: float = MOTION_STATE_TIMEOUT) -> bool:
        """Detour automatique par MOTION_STATE_DETOUR si target n'est pas atteignable
        directement -- meme logique que tools/joint_angle_commander/motion_state.py
        ::ensure_motion_state(), voir docstring module pour pourquoi c'est
        reimplemente ici plutot que reutilise tel quel."""
        self._wait_for_motion_state()
        if self._motion_state["current"] == target:
            return True
        if (target not in self._motion_state["available"]
                and MOTION_STATE_DETOUR in self._motion_state["available"]):
            if not self._switch_motion_state(MOTION_STATE_DETOUR, timeout):
                return False
        return self._switch_motion_state(target, timeout)

    def _execute(self, goal_handle):
        forward = goal_handle.request.forward
        turn = goal_handle.request.turn
        duration = goal_handle.request.duration
        result = WalkTo.Result()

        if not self._ensure_motion_state(WALK_MOTION_STATE):
            self.get_logger().error(f"walk_to : impossible de passer en {WALK_MOTION_STATE}.")
            goal_handle.abort()
            result.success = False
            return result

        self.get_logger().info(f"walk_to : forward={forward}m/s turn={turn}rad/s duration={duration}s")
        period = 1.0 / RATE_HZ
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

        for _ in range(ZERO_PUBLISH_CYCLES):
            msg = self._BodyVelCmd()
            msg.linear_velocity = [0.0, 0.0]
            msg.yaw_velocity = 0.0
            self._vel_pub.publish(msg)
            time.sleep(period)

        if not self._ensure_motion_state("lower_body_balance"):
            self.get_logger().error("walk_to : impossible de repasser en lower_body_balance.")
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
    rclpy.init()
    node = WalkToActionServer()
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
