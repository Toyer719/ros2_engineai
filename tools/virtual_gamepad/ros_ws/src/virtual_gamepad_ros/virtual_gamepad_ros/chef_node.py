"""Chef ROS : agrege l'etat gamepad (boutons/analogiques) publie par les noeuds
de comportement et le republie sur LCM a frequence fixe -- meme canal/URL que
gamepad_api.py, a garder synchronise. Pilote aussi la choregraphie GRAFCET
marche+prise+transport+depose via des Actions ROS (stand/walk_to/lift/pivot/
depose). La sequence tourne dans le thread principal, un MultiThreadedExecutor
traite les callbacks ROS dans un thread separe.
"""

import signal
import sys
import threading
import time

import lcm
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32

sys.path.insert(0, "/home/equansrobotic/engineai_robotics_native_sdk/tools/virtual_gamepad")
from lcm_msgs.data import GamepadKeys  # noqa: E402

from virtual_gamepad_interfaces.action import Depose, Lift, Pivot, Stand, WalkTo  # noqa: E402
from virtual_gamepad_ros.field_topics import ANALOG_INDEX, BUTTON_INDEX, field_topic  # noqa: E402

LCM_URL = "udpm://239.255.76.67:7667?ttl=1"
CHANNEL = "virtual_gamepad/gamepad_keys"
SERVER_TIMEOUT_S = 10.0
GOAL_TIMEOUT_S = 300.0

PROVEN_PINCH_X = 0.35
PINCH_Y = 0.35
SQUEEZE_Y = 0.095
PINCH_Z = -0.139

WALK_STICK = 0.85


class ChefNode(Node):
    def __init__(self, rate_hz: float = 20.0):
        super().__init__("chef")
        self._lcm = lcm.LCM(LCM_URL)
        self._state = GamepadKeys()
        self._period = 1.0 / rate_hz

        for name, idx in BUTTON_INDEX.items():
            self.create_subscription(Bool, field_topic(name), self._button_cb(idx), 10)
        for name, idx in ANALOG_INDEX.items():
            self.create_subscription(Float32, field_topic(name), self._analog_cb(idx), 10)

        self._stand_client = ActionClient(self, Stand, "stand")
        self._walk_to_client = ActionClient(self, WalkTo, "walk_to")
        self._lift_client = ActionClient(self, Lift, "lift")
        self._pivot_client = ActionClient(self, Pivot, "pivot")
        self._depose_client = ActionClient(self, Depose, "depose")
        self._step_pub = self.create_publisher(Int32, "/chef/current_step", 10)

        self.create_timer(self._period, self._publish_to_lcm)

    # --- Etat gamepad -> LCM ---------------------------------------------------

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


    def _send_goal(self, client: ActionClient, name: str, goal) -> bool:
        if not client.wait_for_server(timeout_sec=SERVER_TIMEOUT_S):
            raise RuntimeError(f"Action server '{name}' indisponible.")

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

    @staticmethod
    def _goal_fields(kwargs: dict) -> dict:
        return {k: (v if isinstance(v, (bool, str)) else float(v)) for k, v in kwargs.items()}

    def stand(self, settle_seconds: float = 10.0) -> bool:
        return self._send_goal(self._stand_client, "stand", Stand.Goal(settle_seconds=float(settle_seconds)))

    def walk_to(self, forward: float, turn: float = 0.0, duration: float = 1.0) -> bool:
        goal = WalkTo.Goal(forward=float(forward), turn=float(turn), duration=float(duration))
        return self._send_goal(self._walk_to_client, "walk_to", goal)

    def lift(self, **kwargs) -> bool:
        return self._send_goal(self._lift_client, "lift", Lift.Goal(**self._goal_fields(kwargs)))

    def pivot(self, **kwargs) -> bool:
        return self._send_goal(self._pivot_client, "pivot", Pivot.Goal(**self._goal_fields(kwargs)))

    def depose(self, **kwargs) -> bool:
        return self._send_goal(self._depose_client, "depose", Depose.Goal(**self._goal_fields(kwargs)))

    def _publish_step(self, step: int) -> None:
        """Numero d'etape GRAFCET courant, observable via
        `ros2 topic echo /chef/current_step`."""
        self._step_pub.publish(Int32(data=int(step)))
        self.get_logger().info(f"--- etape {step} ---")

    def run_sequence(self) -> None:
        """GRAFCET marche+prise+pivot (voir capture utilisateur du 08/09) --
        etapes numerotees comme _publish_step() pour matcher le schema.
        Etape 20 "Demande Localisation" existe pour la tracabilite mais
        n'appelle pas encore la vision : le marqueur ArUco est perdu a
        courte distance (cf memoire projet), pinch_x reste donc fixe
        (PROVEN_PINCH_X) pour l'instant.

        Transport+depose (etapes 60-120 du schema, marche EN TENANT le
        carton apres le pivot) retire le 08/09 apres test : la trace
        sim_state montre un pic anormal de hauteur (z 0.75 -> 0.94) au
        moment ou free_legs_for_walk rend les jambes a la marche RL --
        chute quasi immediate ensuite (z<0.15, tilt>100deg, jamais
        recupere), confirme visuellement par l'utilisateur ("il s'est
        redresse et c'est parti en vrille"). Le pivot lui-meme (etape 60)
        est stable -- seul le relachement des jambes vers la marche pose
        probleme. En attendant d'investiguer ce point precis, pivot()
        termine sa propre sequence (depivote + relache, comportement par
        defaut) et la choregraphie s'arrete la, suivie d'un stand()."""
        WALK_DURATION = 3.0
        TURN_CORRECTION = 0.0
        WALK_STANCE_SCALE = 4.5

        self._publish_step(0)
        self.stand()

        self._publish_step(10)
        if not self.walk_to(forward=WALK_STICK, turn=TURN_CORRECTION, duration=WALK_DURATION):
            self.get_logger().error("run_sequence : walk_to(Posage 1) a echoue -- arret.")
            return
        time.sleep(2.0)
        self.stand(settle_seconds=3.0)

        self._publish_step(20)
        # Localisation non branchee (voir docstring) -- pinch_x fixe et deja valide.
        pinch_x = PROVEN_PINCH_X

        self._publish_step(30)
        if not self.lift(pinch_x=pinch_x, pinch_y=PINCH_Y, pinch_z=PINCH_Z, squeeze_y=SQUEEZE_Y,
                          approach_duration=4.0, walk_stance_scale=WALK_STANCE_SCALE,
                          only_phase="approche", release_after=False):
            self.get_logger().error("run_sequence : lift() a echoue (approche) -- arret.")
            return

        self._publish_step(40)
        if not self.lift(pinch_x=pinch_x, pinch_y=PINCH_Y, pinch_z=PINCH_Z, squeeze_y=SQUEEZE_Y,
                          squeeze_duration=5.0, walk_stance=False, only_phase="serrage",
                          release_after=False):
            self.get_logger().error("run_sequence : lift() a echoue (serrage) -- arret.")
            return

        self._publish_step(50)
        if not self.lift(pinch_x=pinch_x, pinch_y=PINCH_Y, pinch_z=PINCH_Z, squeeze_y=SQUEEZE_Y,
                          lift_duration=5.0, hold_seconds=3.0, walk_stance=False,
                          walk_stance_scale=WALK_STANCE_SCALE, only_phase="levee", release_after=False):
            self.get_logger().error("run_sequence : lift() a echoue (levee) -- arret.")
            return

        # Pivot BUSTE seul, carton toujours tenu (bras+jambes). release_after
        # et depivot_before_release restent a leurs defauts (true) -- pivot()
        # depivote et relache lui-meme en fin d'appel, pas de free_legs_for_walk
        # (handoff identifie instable le 08/09, cf docstring).
        self._publish_step(60)
        if not self.pivot(pinch_x=pinch_x, pinch_y=PINCH_Y, pinch_z=PINCH_Z, squeeze_y=SQUEEZE_Y,
                           angle_deg=180.0, walk_stance_scale=WALK_STANCE_SCALE):
            self.get_logger().error("run_sequence : pivot(180) a echoue -- arret.")
            return

        self.stand(settle_seconds=3.0)
        self.get_logger().info("run_sequence : sequence complete terminee.")
        self._publish_step(0)


def main():
    rclpy.init()
    node = ChefNode(rate_hz=20.0)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run_sequence()
        node.get_logger().info("sequence terminee -- heartbeat LCM maintenu (Ctrl+C pour arreter).")
        spin_thread.join()
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