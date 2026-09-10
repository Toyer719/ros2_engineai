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
# Teste en simu (2026-09-10, sequence complete stable, cf commits) -- carton
# saisi mais pas assez profondement selon retour utilisateur (prise pres du
# bord, pas assez engagee). Essai PINCH_X=0.38 : marge shoulder-roll excellente
# (+49.3deg) mais bascule l'IK sur une branche coude/epaule TORDUE (epaule
# -111.9deg, coude +68.3deg) -- confirme visuellement par l'utilisateur
# ("flexion des coudes bizarre"). Viser plus bas sur le carton (pinch_z
# negatif) retrouve une posture naturelle a ce PINCH_X, mais l'utilisateur
# a rejete cette option ("trop bas") -- il veut rester au CENTRE du carton
# (PINCH_Z=0.106 inchange) et gagner en profondeur uniquement via
# PINCH_X/PINCH_Y. Balayage complet a PINCH_Z=0.106 fixe (meme methode,
# chaine complete approche->serrage->levee) : la posture naturelle ET une
# marge positive ne survivent que jusqu'a ~0.345-0.35 -- au-dela (0.355+),
# soit la marge shoulder-roll devient negative (viole la limite), soit l'IK
# bascule sur la meme branche tordue que 0.38. PINCH_X=0.345/PINCH_Y=0.50
# retenu : posture naturelle (epaule -38.4deg, coude -72.5deg, tres proche
# de l'ancien 0.34/0.45), marge=+22.5deg (MEILLEURE que l'ancien 0.34/0.45,
# +14.7deg) -- gain de profondeur modeste (+0.5cm) mais c'est le maximum
# atteignable a cette hauteur sans re-tomber sur la branche tordue ou violer
# la limite. PAS ENCORE TESTE en simu physique.
PROVEN_PINCH_X = 0.345
PINCH_Y = 0.50
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
        # 2026-09-10 : free_legs_for_walk=True essaye puis ABANDONNE apres
        # nouvelle chute confirmee par telemetrie -- meme signature EXACTE
        # que celle deja documentee le 08/09 (z 0.82->0.97 (pic) ->0.12 en
        # moins d'1s, ~0.2s apres que pivot() ait rendu les jambes a la
        # marche RL). Ni le fix de gain de pivot ni le pont body_vel_cmd
        # d'aujourd'hui n'ont d'effet ici -- ce n'est pas le meme mecanisme
        # (celui-ci se produit AU MOMENT du handoff jambes lui-meme, avant
        # meme que walk_to() ne commence a marcher). Reste un probleme
        # ouvert, non resolu aujourd'hui. **Approche alternative retenue :**
        # ne plus jamais combiner "marche" et "objet tenu" -- depose() le
        # carton SUR PLACE juste apres le pivot (release_after=False,
        # free_legs_for_walk=False -- carton toujours tenu, jambes PAS
        # rendues a la marche), PUIS un stand() complet (plus rien tenu),
        # PUIS une marche a vide (deja prouvee sure) vers un 2e point.
        # 2026-09-10 (suite) : ordre repris du robot reel, valide ce meme jour
        # apres plusieurs iterations en direct -- depivot_before_release=False
        # : le buste RESTE tourne (45deg) quand pivot() rend la main, carton
        # toujours tenu. Le relachement (dans depose(), ci-dessous) se fait
        # PENDANT que le buste est encore tourne, PUIS le buste revient au
        # centre -- pas l'inverse (ancien ordre, faisait retraverser aux bras
        # tendus l'espace ou le carton venait d'etre pose, choc constate sur
        # le reel).
        PIVOT_ANGLE_DEG = 45.0
        if not self.pivot(pinch_x=pinch_x, pinch_y=PINCH_Y, pinch_z=PINCH_Z, squeeze_y=SQUEEZE_Y,
                           lift_z=LIFT_Z, angle_deg=PIVOT_ANGLE_DEG, walk_stance_scale=WALK_STANCE_SCALE,
                           release_after=False, free_legs_for_walk=False,
                           depivot_before_release=False):
            self.get_logger().error(f"run_sequence : pivot({PIVOT_ANGLE_DEG:.0f}) a echoue -- arret.")
            return

        # depose() SUR PLACE (pas de marche entre pivot et depose -- evite le
        # handoff jambes instable). pinch_z/hold_z = LIFT_Z (PAS les defauts
        # perimes de Depose.action) : c'est la hauteur ou pivot() tenait
        # reellement le carton, un ecart ferait sauter les bras au premier
        # message de depose() (meme piege que celui deja corrige dans
        # depose.py). drop_z = LIFT_Z - 0.03 (PETITE baisse avant relachement,
        # PAS un retour complet a PINCH_Z -- meme reglage que le robot reel ce
        # jour). depivot_from_deg=PIVOT_ANGLE_DEG : le buste est ENCORE tourne
        # a l'entree de ce node (voir commentaire ci-dessus), depose.py le
        # tient a cet angle pendant baisse/desserrage/degagement puis le
        # ramene a 0 juste avant le relachement final. walk_stance=true
        # (defaut) volontairement garde : re-flechit les genoux (deja
        # droits, WALK_STANCE_SCALE=0.0 pendant lift/pivot) avant de
        # manipuler la charge -- meme pattern deja prouve stable partout
        # ailleurs (lift.py), plutot que la combinaison jambes droites
        # statiques + charge qui descend, jamais testee.
        self._publish_step(80)
        if not self.depose(pinch_x=pinch_x, pinch_y=PINCH_Y, pinch_z=LIFT_Z, squeeze_y=SQUEEZE_Y,
                            hold_z=LIFT_Z, drop_z=LIFT_Z - 0.03,
                            depivot_from_deg=PIVOT_ANGLE_DEG, depivot_duration=2.0):
            self.get_logger().error("run_sequence : depose() a echoue -- arret.")
            return

        self.stand(settle_seconds=3.0)

        # Marche a vide (rien tenu) vers un 2e point -- deja prouvee sure de
        # nombreuses fois aujourd'hui, sert juste a montrer que le robot
        # peut encore se deplacer apres la tache, sans reproduire le
        # handoff instable.
        self._publish_step(90)
        if not self.walk_to(forward=WALK_FORWARD_MPS, turn=0.0, duration=1.5):
            self.get_logger().error("run_sequence : walk_to(depart) a echoue -- arret.")
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
