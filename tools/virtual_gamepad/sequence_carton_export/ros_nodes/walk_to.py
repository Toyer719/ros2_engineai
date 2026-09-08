import signal
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Float32

from virtual_gamepad_interfaces.action import WalkTo  # noqa: E402
from virtual_gamepad_ros.field_topics import field_topic  # noqa: E402


class WalkToActionServer(Node):
    def __init__(self):
        super().__init__("walk_to")

        self._lb_pub = self.create_publisher(Bool, field_topic("LB"), 10)
        self._b_pub = self.create_publisher(Bool, field_topic("B"), 10)
        self._left_x_pub = self.create_publisher(Float32, field_topic("LEFT_STICK_X"), 10)
        self._right_y_pub = self.create_publisher(Float32, field_topic("RIGHT_STICK_Y"), 10)
        self._server = ActionServer(
            self, WalkTo, "walk_to", self._execute, cancel_callback=self._on_cancel,
        )

    def _on_cancel(self, goal_handle):
        return CancelResponse.ACCEPT

    def _push_sticks(self, forward: float, turn: float) -> None:
        self._left_x_pub.publish(Float32(data=float(forward)))
        self._right_y_pub.publish(Float32(data=float(-turn)))

    def _enter_walk(self, combo_hold_seconds=0.5, walk_settle_seconds=1.5) -> None:
        """Envoie LB+B (walk) puis attend que la politique RL prenne le relais
        -- memes durees que l'ancien chef_node.start_in_walk()."""
        self.get_logger().info("Passage en walk...")
        self._lb_pub.publish(Bool(data=True))
        self._b_pub.publish(Bool(data=True))
        time.sleep(combo_hold_seconds)
        self._lb_pub.publish(Bool(data=False))
        self._b_pub.publish(Bool(data=False))
        time.sleep(walk_settle_seconds)
        self.get_logger().info("walk actif.")

    def _execute(self, goal_handle):
        result = WalkTo.Result()
        forward = goal_handle.request.forward
        turn = goal_handle.request.turn
        duration = goal_handle.request.duration

        self._enter_walk()

        self.get_logger().info(f"walk_to : forward={forward} turn={turn} duration={duration}s")
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


def main():
    rclpy.init()
    node = WalkToActionServer()
    # MultiThreadedExecutor : l'execution d'un goal (boucle bloquante ci-dessus)
    # tourne dans son propre thread pendant qu'un autre thread reste libre pour
    # traiter les requetes de cancel entrantes.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Un 2e Ctrl-C pendant le nettoyage (destroy_node) relance un
        # KeyboardInterrupt en plein milieu et laisse une trace moche -- on est
        # deja engages dans l'arret, donc on ignore les SIGINT supplementaires.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
