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
from lcm_msgs.data import GamepadKeys

from virtual_gamepad_interfaces.action import Depose, Lift, Pivot, Stand, WalkTo
from virtual_gamepad_ros.field_topics import ANALOG_INDEX, BUTTON_INDEX, field_topic

LCM_URL = "udpm://239.255.76.67:7667?ttl=1"
CHANNEL = "virtual_gamepad/gamepad_keys"
SERVER_TIMEOUT_S = 10.0
GOAL_TIMEOUT_S = 300.0

# 2026-09-09 : PROVEN_PINCH_X=0.26 (marge shoulder-roll seulement +3.95deg a la
# levee) restait trop court pour atteindre le carton a la distance de marche
# WALK_DURATION=2.3 (seule distance jugee sure a l'oeil par l'utilisateur --
# plus proche = le robot touche le podium et bascule). Augmenter juste
# PINCH_X depasse vite la limite J14/J19_SHOULDER_ROLL ([-35,135]deg,
# Guide_PM01_FR.pdf) : 0.28 la viole deja (-2.41deg).
# Fix : elargir PINCH_Y (largeur d'approche AVANT le serrage, n'affecte pas le
# point de prise final -- SQUEEZE_Y/PINCH_Z restent la cible physique reelle)
# fait converger l'IK (redondant, solutions multiples) vers une autre branche
# articulaire qui laisse bien plus de marge au meme point de prise. Verifie
# numeriquement (solve_ik, chaine complete approche->serrage->levee, les 5
# articulations du bras, pas juste le roll) : PINCH_X=0.34/PINCH_Y=0.45 donne
# marge=+14.6deg (levee) contre +3.95deg avant, tout en portant 8cm plus loin.
# PAS ENCORE TESTE en simu (calcul IK seul, physique MuJoCo non verifiee).
PROVEN_PINCH_X = 0.34
PINCH_Y = 0.45
SQUEEZE_Y = 0.095
PINCH_Z = 0.106
LIFT_Z = 0.20

# 2026-09-09 : premier essai walk_to en mode --real (BodyVelCmd) abandonne --
# le noeud ROS2 qui gererait /motion/body_vel_cmd et /motion/motion_state
# cote reel (locomotion_interface_node) n'existe pas en simulation (absent
# de `ros2 node list`, absent du SDK). Repris le 2026-09-10 : walk_to.py
# n'a plus qu'une seule logique (toujours BodyVelCmd), un node-pont sim
# uniquement (body_vel_bridge.py) traduit vers l'emulation manette LCM que
# MuJoCo comprend deja. WALK_FORWARD_MPS=0.45 est la valeur REELLE deja
# prouvee (marche.py::DEFAULT_FORWARD_MPS), c'est le SEUL point ou la
# calibration vitesse->stick du bridge est fidele (voir sa docstring --
# pas physiquement lineaire sur toute la plage) : ne pas changer cette
# valeur sans reverifier en sim par telemetrie.
WALK_FORWARD_MPS = 0.45


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
        WALK_DURATION = 2.2
        TURN_CORRECTION = 0.0
        WALK_STANCE_SCALE = 0.0

        self._publish_step(0)
        self.stand()

        self._publish_step(10)
        if not self.walk_to(forward=WALK_FORWARD_MPS, turn=TURN_CORRECTION, duration=WALK_DURATION):
            self.get_logger().error("run_sequence : walk_to(Posage 1) a echoue -- arret.")
            return
        time.sleep(2.0)
        self.stand(settle_seconds=3.0)

        self._publish_step(20)
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
                          lift_z=LIFT_Z, lift_duration=5.0, hold_seconds=3.0, walk_stance=False,
                          walk_stance_scale=WALK_STANCE_SCALE, only_phase="levee", release_after=False):
            self.get_logger().error("run_sequence : lift() a echoue (levee) -- arret.")
            return

        self._publish_step(60)
        # 2026-09-09 : cause racine trouvee -- WAIST_KP/KD=150/3.0 (pivot.py,
        # valeurs qui marchent sur le VRAI robot) ne produisait QUASIMENT
        # AUCUNE rotation reelle en sim (confirme via /hardware/joint_state :
        # <5deg de bruit au lieu de 45-180deg vises) -- donc TOUTES les
        # chutes precedentes (180/90/45deg, angle sans effet observable)
        # venaient en realite du depivot+relachement, pas d'une vraie
        # rotation. Gains montes a 500/10.0 dans pivot.py.
        # 45deg : confirme a 45.1deg reel (waist_log), STABLE de bout en
        # bout, confirme visuellement par l'utilisateur (2/2).
        # 60deg ET 90deg : reproductiblement bloques (<3deg de bruit, meme
        # apres nettoyage complet DDS -- pas un probleme de ressources).
        # Limite matterielle reelle de J12_WAIST_YAW (Guide_PM01_FR.pdf p.10,
        # ligne 35) : -4.014 a 1.57 rad = -230 a +90deg -- 90deg est pile
        # sur la limite haute (rejet attendu), mais 60deg est theoriquement
        # dans la plage et bloque quand meme -- deuxieme restriction non
        # identifiee (probablement l'arbitre de securite), pas creusee plus
        # loin. 45deg retenu comme valeur fiable pour la tache.
        if not self.pivot(pinch_x=pinch_x, pinch_y=PINCH_Y, pinch_z=PINCH_Z, squeeze_y=SQUEEZE_Y,
                           lift_z=LIFT_Z, angle_deg=45.0, walk_stance_scale=WALK_STANCE_SCALE):
            self.get_logger().error("run_sequence : pivot(45) a echoue -- arret.")
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
