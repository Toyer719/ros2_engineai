"""Test autonome de la phase "levee" SEULE (stand + lift approche/serrage/levee),
sans marche/pivot/depose -- demande explicite de l'utilisateur ("je peux pas faire
genre chef_node.py avec juste la phase levee ?"). Evite d'avoir a lancer lift.py
puis taper les goals ros2 action send_goal a la main avec tous les champs.

Reprend EXACTEMENT les memes parametres que chef_node.py::run_sequence() (etapes
20-50) -- PROVEN_PINCH_X/PINCH_Y/SQUEEZE_Y/PINCH_Z, WALK_STANCE_SCALE=0.0 (pas de
flexion genoux avant lift, comme chef_node.py) -- pour rester comparable a la
sequence complete. Difference : release_after=True sur le dernier appel (relache
proprement a la fin, pas de carton laisse en l'air indefiniment comme dans
chef_node.py ou un pivot est cense suivre).

Lancement : memes 3 source que lift.py (humble + overlay SDK + overlay
virtual_gamepad_ros), PUIS lancer lift.py et stand.py chacun dans son propre
terminal AVANT ce script (ce script est juste un CLIENT, pas un serveur d'action --
il ne fait que passer les 3 goals, comme chef_node.py le ferait). Suppose le robot
deja en pd_stand, immobile, devant le carton (pas de marche ici).
"""
import signal
import sys
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from virtual_gamepad_interfaces.action import Lift, Stand

SERVER_TIMEOUT_S = 10.0
GOAL_TIMEOUT_S = 300.0

# Memes valeurs que chef_node.py -- garder synchronise si elles changent la-bas.
PROVEN_PINCH_X = 0.345
PINCH_Y = 0.22
SQUEEZE_Y = 0.095
PINCH_Z = 0.106
LIFT_Z = 0.20
WALK_STANCE_SCALE = 0.0


class TestLiftOnlyNode(Node):
    def __init__(self):
        super().__init__("test_lift_only")
        self._stand_client = ActionClient(self, Stand, "stand")
        self._lift_client = ActionClient(self, Lift, "lift")

    def _send_goal(self, client: ActionClient, name: str, goal) -> bool:
        if not client.wait_for_server(timeout_sec=SERVER_TIMEOUT_S):
            raise RuntimeError(f"Action server '{name}' indisponible -- lance ?")

        goal_done = threading.Event()
        outcome = {}

        def on_result(result_future):
            outcome["result"] = result_future.result().result
            goal_done.set()

        def on_goal_response(goal_future):
            goal_handle = goal_future.result()
            if not goal_handle.accepted:
                outcome["error"] = RuntimeError(f"Goal {name} refuse.")
                goal_done.set()
                return
            goal_handle.get_result_async().add_done_callback(on_result)

        client.send_goal_async(goal).add_done_callback(on_goal_response)
        if not goal_done.wait(timeout=GOAL_TIMEOUT_S):
            raise RuntimeError(f"Goal {name} sans resultat apres {GOAL_TIMEOUT_S}s.")
        if "error" in outcome:
            raise outcome["error"]

        success = outcome["result"].success
        self.get_logger().info(f"{name} termine : success={success}")
        return success

    def stand(self, settle_seconds: float = 10.0) -> bool:
        return self._send_goal(self._stand_client, "stand", Stand.Goal(settle_seconds=float(settle_seconds)))

    def lift(self, **kwargs) -> bool:
        fields = {k: (v if isinstance(v, (bool, str)) else float(v)) for k, v in kwargs.items()}
        return self._send_goal(self._lift_client, "lift", Lift.Goal(**fields))

    def run_lift_only(self) -> None:
        self.get_logger().info("stand...")
        self.stand()

        self.get_logger().info("lift -- approche...")
        if not self.lift(pinch_x=PROVEN_PINCH_X, pinch_y=PINCH_Y, pinch_z=PINCH_Z, squeeze_y=SQUEEZE_Y,
                          approach_duration=4.0, walk_stance_scale=WALK_STANCE_SCALE,
                          only_phase="approche", release_after=False):
            self.get_logger().error("lift(approche) a echoue -- arret.")
            return

        self.get_logger().info("lift -- serrage...")
        if not self.lift(pinch_x=PROVEN_PINCH_X, pinch_y=PINCH_Y, pinch_z=PINCH_Z, squeeze_y=SQUEEZE_Y,
                          squeeze_duration=5.0, walk_stance=False, only_phase="serrage",
                          release_after=False):
            self.get_logger().error("lift(serrage) a echoue -- arret.")
            return

        self.get_logger().info("lift -- levee (release_after=True, relache proprement a la fin)...")
        if not self.lift(pinch_x=PROVEN_PINCH_X, pinch_y=PINCH_Y, pinch_z=PINCH_Z, squeeze_y=SQUEEZE_Y,
                          lift_z=LIFT_Z, lift_duration=5.0, hold_seconds=3.0, walk_stance=False,
                          walk_stance_scale=WALK_STANCE_SCALE, only_phase="levee", release_after=True):
            self.get_logger().error("lift(levee) a echoue -- arret.")
            return

        self.get_logger().info("phase levee terminee.")


def main():
    rclpy.init()
    node = TestLiftOnlyNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run_lift_only()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
