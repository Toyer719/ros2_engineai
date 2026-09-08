import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from virtual_gamepad_interfaces.action import Lift


class ChefTest1(Node):
    def __init__(self):
        super().__init__("chef_test1")
        self._client = ActionClient(self, Lift, "lift")

    def lift(self):
        print("connexion a l'Action /lift...")
        self._client.wait_for_server()
        print("connecte -- envoi de l'objectif (flexion genoux, approche, serrage, levee)...")

        goal = Lift.Goal(
            pinch_x=0.35, pinch_y=0.35, pinch_z=-0.139, squeeze_y=0.095,
            approach_duration=4.0, squeeze_duration=5.0, lift_duration=5.0,
            hold_seconds=3.0, release_ramp_seconds=4.0,
            walk_stance=True, walk_stance_scale=4.5,
        )

        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        print("objectif accepte -- execution en cours (~60-90s, regarder MuJoCo)...")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result().result.success


def main():
    rclpy.init()
    node = ChefTest1()

    ok = node.lift()
    print("reussi" if ok else "echoue")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
