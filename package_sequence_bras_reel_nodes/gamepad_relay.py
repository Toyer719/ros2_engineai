"""Relais manette virtuelle -> LCM, SIM UNIQUEMENT -- extrait de
chef_node.py (role 1 de son __init__/_publish_to_lcm), sans le role 2
(orchestrateur de sequence, qui est desormais chef_reel.py). Sans ce
relais, les topics /virtual_gamepad/cmd/* publies par le node "stand"
(LB+A) ne sont RELAYES nulle part -- MuJoCo/src_executor n'ecoutent que
le canal LCM 'virtual_gamepad/gamepad_keys', pas les topics ROS2
directement. Deja rencontre : stand() rapportait success=True sans que
le robot ne se leve reellement, faute de ce relais."""
import sys
import time

import lcm
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32

sys.path.insert(0, "/home/equansrobotic/engineai_robotics_native_sdk/tools/virtual_gamepad")
from lcm_msgs.data import GamepadKeys

from virtual_gamepad_ros.field_topics import ANALOG_INDEX, BUTTON_INDEX, field_topic

LCM_URL = "udpm://239.255.76.67:7667?ttl=1"
CHANNEL = "virtual_gamepad/gamepad_keys"


class GamepadRelay(Node):
    def __init__(self, rate_hz: float = 20.0):
        super().__init__("gamepad_relay")
        self._lcm = lcm.LCM(LCM_URL)
        self._state = GamepadKeys()

        for name, idx in BUTTON_INDEX.items():
            self.create_subscription(Bool, field_topic(name), self._button_cb(idx), 10)
        for name, idx in ANALOG_INDEX.items():
            self.create_subscription(Float32, field_topic(name), self._analog_cb(idx), 10)

        self.create_timer(1.0 / rate_hz, self._publish_to_lcm)
        self.get_logger().info("gamepad_relay pret -- relais /virtual_gamepad/cmd/* -> LCM.")

    def _button_cb(self, idx: int):
        def cb(msg: Bool) -> None:
            self._state.digital_states[idx] = int(msg.data)
        return cb

    def _analog_cb(self, idx: int):
        def cb(msg: Float32) -> None:
            self._state.analog_states[idx] = float(msg.data)
        return cb

    def _publish_to_lcm(self) -> None:
        self._state.timestamp = int(time.time() * 1_000_000)
        self._lcm.publish(CHANNEL, self._state.encode())


def main():
    rclpy.init()
    node = GamepadRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
