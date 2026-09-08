import signal
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool

from virtual_gamepad_interfaces.action import Stand  # noqa: E402
from virtual_gamepad_ros.field_topics import field_topic  # noqa: E402


class StandActionServer(Node):
    def __init__(self):
        super().__init__("stand")
        self._lb_pub = self.create_publisher(Bool, field_topic("LB"), 10)
        self._a_pub = self.create_publisher(Bool, field_topic("A"), 10)
        self._server = ActionServer(
            self, Stand, "stand", self._execute, cancel_callback=self._on_cancel,
        )

    def _on_cancel(self, goal_handle):
        return CancelResponse.ACCEPT

    def _wait_cancelable(self, goal_handle, duration, step=0.1):
        """Attend `duration` secondes par pas de `step`, en verifiant les
        demandes d'annulation. Renvoie False si annule en cours de route."""
        elapsed = 0.0
        while elapsed < duration:
            if goal_handle.is_cancel_requested:
                return False
            time.sleep(step)
            elapsed += step
        return True

    def _execute(self, goal_handle, combo_hold_seconds=0.5):
        result = Stand.Result()
        settle_seconds = goal_handle.request.settle_seconds

        # Pulse LB+A puis relache -- meme principe que send_raw()
        # (gamepad_api.py) et l'ancien Conductor._send_button_combo() :
        # un combo est un appui BREF, pas un maintien continu. Le tenir
        # enfonce pendant toute la duree de stabilisation (bug corrige ici)
        # divergeait du comportement prouve par main.py/core/chef.py.
        self.get_logger().info("Passage en pd_stand...")
        self._lb_pub.publish(Bool(data=True))
        self._a_pub.publish(Bool(data=True))
        if not self._wait_cancelable(goal_handle, combo_hold_seconds):
            self._lb_pub.publish(Bool(data=False))
            self._a_pub.publish(Bool(data=False))
            goal_handle.canceled()
            result.success = False
            return result
        self._lb_pub.publish(Bool(data=False))
        self._a_pub.publish(Bool(data=False))

        # Boutons deja relaches -- attente separee de la stabilisation.
        if not self._wait_cancelable(goal_handle, settle_seconds):
            goal_handle.canceled()
            result.success = False
            return result

        self.get_logger().info("pd_stand stabilise.")
        goal_handle.succeed()
        result.success = True
        return result


def main():
    rclpy.init()
    node = StandActionServer()
    # MultiThreadedExecutor : meme raison que walk_to.py -- l'execution d'un
    # goal (boucle bloquante ci-dessus, via create_rate) tourne dans son
    # propre thread pendant qu'un autre thread reste libre pour traiter les
    # requetes de cancel entrantes.
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
