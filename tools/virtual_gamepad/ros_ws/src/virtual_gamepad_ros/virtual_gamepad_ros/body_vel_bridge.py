"""Pont SIM UNIQUEMENT entre le protocole "reel" (/motion/body_vel_cmd,
/motion/set_motion_state, /motion/motion_state -- interface_protocol) et
l'emulation manette LCM que la simulation MuJoCo comprend deja (les memes
topics /virtual_gamepad/cmd/* que walk_to.py pilotait avant en mode sim).

But : permettre a walk_to.py de tourner avec UNE SEULE logique (toujours
/motion/body_vel_cmd, jamais de branche sim/reel separee) -- ce node
traduit en coulisses, absent du lancement reel puisque le vrai robot
possede deja son propre noeud qui consomme ce protocole nativement.

Portee volontairement minimale (pas une vraie machine a etats) : seuls les
etats dont walk_to.py a besoin sont geres --
  - "rl_terrain" (marche) : declenche la combo manette LB+B (meme sequence
    que l'ancien _enter_walk() de walk_to.py) puis traduit chaque
    BodyVelCmd recu en sticks gauche/droit.
  - "pd_stand" / "lower_body_balance" (arret) : accepte immediatement,
    aucune action physique -- en sim, la stabilisation reelle est deja
    geree par le node `stand` separe que chef_node.py appelle juste apres
    walk_to() dans sa propre sequence.
Les trois noms sont donc toujours annonces comme disponibles
(available_transition_motions) : pas de detour necessaire.

Calibration vitesse -> stick : PAS PHYSIQUEMENT LINEAIRE sur toute la
plage (mesure telemetrie sim_state le 2026-09-10 : stick 0.85->0.50m/s
mesures, stick 0.50->0.16m/s -- x1.7 de stick donne x3.1 de deplacement,
la marche en sim repond par a-coups sur des commandes courtes, pas comme
un controleur de vitesse lineaire). Calibre donc sur UN SEUL point
d'ancrage, celui reellement utilise par chef_node.py : forward=0.45m/s
(valeur reelle prouvee, marche.py::DEFAULT_FORWARD_MPS) <-> stick=0.85
(valeur sim prouvee stable, ancien WALK_STICK). Fidele au point utilise,
approximatif ailleurs -- non calibre pour turn (jamais utilise a une
valeur non nulle dans la sequence actuelle), meme facteur applique par
defaut faute de meilleure donnee.
"""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32

from virtual_gamepad_ros.field_topics import field_topic

MPS_ANCHOR = 0.45
STICK_ANCHOR = 0.85
MPS_TO_STICK = STICK_ANCHOR / MPS_ANCHOR

WALK_STATE = "rl_terrain"
STOP_STATES = ("pd_stand", "lower_body_balance")
AVAILABLE_STATES = [WALK_STATE, *STOP_STATES]

ENTER_WALK_COMBO_HOLD_S = 0.5
ENTER_WALK_SETTLE_S = 1.5


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class BodyVelBridge(Node):
    def __init__(self):
        super().__init__("body_vel_bridge")
        from interface_protocol.msg import BodyVelCmd, MotionState, MotionStateRequest

        state_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE)
        request_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.VOLATILE)
        vel_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                              durability=DurabilityPolicy.VOLATILE)

        self._MotionState = MotionState
        self._current = "lower_body_balance"
        self._last_stick = (0.0, 0.0)

        self._lb_pub = self.create_publisher(Bool, field_topic("LB"), 10)
        self._b_pub = self.create_publisher(Bool, field_topic("B"), 10)
        self._left_x_pub = self.create_publisher(Float32, field_topic("LEFT_STICK_X"), 10)
        self._right_y_pub = self.create_publisher(Float32, field_topic("RIGHT_STICK_Y"), 10)

        self._state_pub = self.create_publisher(MotionState, "/motion/motion_state", state_qos)
        self.create_subscription(MotionStateRequest, "/motion/set_motion_state",
                                  self._on_state_request, request_qos)
        self.create_subscription(BodyVelCmd, "/motion/body_vel_cmd", self._on_vel_cmd, vel_qos)

        self.create_timer(0.1, self._publish_state)
        self.get_logger().info(
            f"body_vel_bridge pret -- calibration {MPS_ANCHOR}m/s <-> stick {STICK_ANCHOR} "
            f"(facteur {MPS_TO_STICK:.3f}, non lineaire hors de ce point, voir docstring)."
        )

    def _publish_state(self) -> None:
        msg = self._MotionState()
        msg.current_motion_task = self._current
        msg.available_transition_motions = list(AVAILABLE_STATES)
        self._state_pub.publish(msg)

    def _enter_walk(self) -> None:
        self.get_logger().info("passage en walk (combo manette LB+B)...")
        self._lb_pub.publish(Bool(data=True))
        self._b_pub.publish(Bool(data=True))
        time.sleep(ENTER_WALK_COMBO_HOLD_S)
        self._lb_pub.publish(Bool(data=False))
        self._b_pub.publish(Bool(data=False))
        time.sleep(ENTER_WALK_SETTLE_S)
        self.get_logger().info("walk actif.")

    def _on_state_request(self, msg) -> None:
        target = msg.target_motion_name
        if target == self._current:
            return
        if target == WALK_STATE:
            self._enter_walk()
            self._last_stick = (0.0, 0.0)
        # pd_stand/lower_body_balance : aucune action physique, voir docstring module.
        self._current = target
        self._publish_state()

    def _on_vel_cmd(self, msg) -> None:
        if self._current != WALK_STATE:
            return
        forward_stick = _clamp(msg.linear_velocity[0] * MPS_TO_STICK)
        turn_stick = _clamp(msg.yaw_velocity * MPS_TO_STICK)
        # 2026-09-10 : republier a chaque body_vel_cmd recu (jusqu'a 100Hz,
        # meme valeur inchangee) a fait marcher le robot ~22cm plus loin
        # qu'avant (telemetrie sim_state) pour le MEME stick nominal et une
        # duree quasi identique -- cause exacte non confirmee (reponse par
        # a-coups deja documentee, un signal different peut faire basculer
        # sur un pas de plus), mais la chute a disparu en repliquant le
        # comportement one-shot de l'ancien _push_sticks() : un seul
        # publish par CHANGEMENT de valeur, pas un flux continu.
        if (forward_stick, turn_stick) == self._last_stick:
            return
        self._last_stick = (forward_stick, turn_stick)
        self._left_x_pub.publish(Float32(data=float(forward_stick)))
        self._right_y_pub.publish(Float32(data=float(-turn_stick)))


def main():
    rclpy.init()
    node = BodyVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
