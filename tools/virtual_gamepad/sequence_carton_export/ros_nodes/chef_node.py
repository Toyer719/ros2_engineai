"""Chef d'orchestre ROS : agrege les topics /virtual_gamepad/cmd/* publies par les
noeuds de comportement (walk_to, ...) dans un etat GamepadKeys persistant, publie
sur LCM a frequence fixe. Remplace core/chef.py (Conductor pur Python) -- meme
canal/URL que test_passe/gamepad_api.py, a garder synchronise si l'un des deux
change.

Pilote aussi la choregraphie (sequence d'appels d'Actions ROS comme walk_to) --
seul fichier qui a le droit de savoir quelle sequence executer, meme role que
l'ancien main.py. La sequence tourne dans le thread principal pendant qu'un
MultiThreadedExecutor traite les callbacks ROS (subscriptions + action client)
dans un thread separe.
"""

import signal
import sys
import threading
import time

import lcm
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Float32

sys.path.insert(0, "/home/equansrobotic/engineai_robotics_native_sdk/tools/virtual_gamepad")
from lcm_msgs.data import GamepadKeys  # noqa: E402

from virtual_gamepad_interfaces.action import Lift, Pivot, Stand, WalkTo  # noqa: E402
from virtual_gamepad_ros.field_topics import ANALOG_INDEX, BUTTON_INDEX, field_topic  # noqa: E402

LCM_URL = "udpm://239.255.76.67:7667?ttl=1"
CHANNEL = "virtual_gamepad/gamepad_keys"


class ChefNode(Node):
    def __init__(self, rate_hz: float = 20.0):
        super().__init__("chef")
        self._lcm = lcm.LCM(LCM_URL)
        # Etat persistant : cree une seule fois, jamais recree, seulement modifie
        # en place par les callbacks de subscription -- un champ non touche garde
        # sa derniere valeur d'un tick a l'autre (meme contrat que l'ancien
        # core/node.py).
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

        # Pose du carton en repere robot (LINK_BASE), publiee par
        # tools/vision/carton_pose_publisher.py -- consommee par
        # approach_carton() ci-dessous. Lock car ecrite depuis le thread de
        # l'executor (callback ROS) et lue depuis le thread principal
        # (run_sequence()), meme precaution que _state ci-dessus (implicite
        # la, explicite ici car approach_carton() lit un objet, pas un champ
        # scalaire modifie en place).
        self._carton_pose_lock = threading.Lock()
        self._carton_pose = None
        self._carton_pose_received_at = None
        self.create_subscription(PoseStamped, "/vision/carton_pose_base", self._on_carton_pose, 10)

        self.create_timer(self._period, self._publish_to_lcm)

    def _on_carton_pose(self, msg: PoseStamped) -> None:
        with self._carton_pose_lock:
            self._carton_pose = msg
            self._carton_pose_received_at = time.monotonic()

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

    def stand(self, settle_seconds: float = 10.0) -> bool:
        """Bloque jusqu'a ce que le node stand ait stabilise la station debout
        (pd_stand, LB+A) ou echoue. Suppose que le robot demarre en `passive`,
        comme au lancement de run.sh. Meme pattern threading.Event que
        walk_to() ci-dessous."""
        self.get_logger().info(f"stand(settle_seconds={settle_seconds})...")
        if not self._stand_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Action server 'stand' indisponible.")

        goal = Stand.Goal(settle_seconds=float(settle_seconds))
        goal_done = threading.Event()
        outcome = {}

        def on_result(result_future):
            outcome["result"] = result_future.result().result
            goal_done.set()

        def on_goal_response(goal_future):
            goal_handle = goal_future.result()
            if not goal_handle.accepted:
                outcome["error"] = RuntimeError("Goal stand refuse.")
                goal_done.set()
                return
            goal_handle.get_result_async().add_done_callback(on_result)

        send_future = self._stand_client.send_goal_async(goal)
        send_future.add_done_callback(on_goal_response)

        goal_done.wait()
        if "error" in outcome:
            raise outcome["error"]

        result = outcome["result"]
        self.get_logger().info(f"stand termine : success={result.success}")
        return result.success

    def walk_to(self, forward: float, turn: float = 0.0, duration: float = 1.0) -> bool:
        """Bloque jusqu'a ce que le node walk_to ait fini de pousser la commande
        de vitesse (boucle ouverte, pas de retour de position) ou echoue.
        A appeler depuis run_sequence() (thread principal) -- utilise un
        threading.Event plutot que spin_until_future_complete pour ne pas entrer
        en conflit avec le MultiThreadedExecutor qui tourne deja dans son propre
        thread (voir main())."""
        self.get_logger().info(f"walk_to(forward={forward}, turn={turn}, duration={duration})...")
        if not self._walk_to_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Action server 'walk_to' indisponible.")

        goal = WalkTo.Goal(forward=float(forward), turn=float(turn), duration=float(duration))
        goal_done = threading.Event()
        outcome = {}

        def on_result(result_future):
            outcome["result"] = result_future.result().result
            goal_done.set()

        def on_goal_response(goal_future):
            goal_handle = goal_future.result()
            if not goal_handle.accepted:
                outcome["error"] = RuntimeError("Goal walk_to refuse.")
                goal_done.set()
                return
            goal_handle.get_result_async().add_done_callback(on_result)

        send_future = self._walk_to_client.send_goal_async(goal)
        send_future.add_done_callback(on_goal_response)

        goal_done.wait()
        if "error" in outcome:
            raise outcome["error"]

        result = outcome["result"]
        self.get_logger().info(f"walk_to termine : success={result.success}")
        return result.success

    def lift(self, **kwargs) -> bool:
        """Bloque jusqu'a ce que le node lift ait fini la sequence
        approche/serrage/levee/maintien/relachement, ou echoue. `**kwargs`
        transmis tels quels a Lift.Goal (pinch_x, pinch_y, pinch_yaw_offset,
        hold_seconds, ...) -- un champ omis garde la valeur par defaut
        definie dans Lift.action. Suppose le robot deja en pd_stand,
        immobile, positionne devant le carton (ex: apres self.walk_to(...)).
        Meme pattern threading.Event que walk_to()/stand() ci-dessus."""
        self.get_logger().info(f"lift(kwargs={kwargs})...")
        if not self._lift_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Action server 'lift' indisponible.")

        # bool NON converti en float ici (release_after) -- Lift.Goal(**kwargs)
        # plante sinon (float(False)=0.0 rejete par le setter bool du message).
        goal = Lift.Goal(**{k: (v if isinstance(v, bool) else float(v)) for k, v in kwargs.items()})
        goal_done = threading.Event()
        outcome = {}

        def on_result(result_future):
            outcome["result"] = result_future.result().result
            goal_done.set()

        def on_goal_response(goal_future):
            goal_handle = goal_future.result()
            if not goal_handle.accepted:
                outcome["error"] = RuntimeError("Goal lift refuse.")
                goal_done.set()
                return
            goal_handle.get_result_async().add_done_callback(on_result)

        send_future = self._lift_client.send_goal_async(goal)
        send_future.add_done_callback(on_goal_response)

        goal_done.wait()
        if "error" in outcome:
            raise outcome["error"]

        result = outcome["result"]
        self.get_logger().info(f"lift termine : success={result.success}")
        return result.success

    def pivot(self, **kwargs) -> bool:
        """Bloque jusqu'a ce que le node pivot ait fini pivot/maintien/depivot/
        relachement. Suppose que self.lift(..., release_after=False) vient
        d'etre appele (carton en l'air, non relache) -- pivot.py reprend la
        tenue complete bras+jambes (recalculee depuis pinch_x/y/z, MEME
        convention que Lift.Goal) en plus du buste, voir Pivot.action.
        Meme pattern threading.Event que lift()/walk_to()/stand()."""
        self.get_logger().info(f"pivot(kwargs={kwargs})...")
        if not self._pivot_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Action server 'pivot' indisponible.")

        goal = Pivot.Goal(**{k: (v if isinstance(v, bool) else float(v)) for k, v in kwargs.items()})
        goal_done = threading.Event()
        outcome = {}

        def on_result(result_future):
            outcome["result"] = result_future.result().result
            goal_done.set()

        def on_goal_response(goal_future):
            goal_handle = goal_future.result()
            if not goal_handle.accepted:
                outcome["error"] = RuntimeError("Goal pivot refuse.")
                goal_done.set()
                return
            goal_handle.get_result_async().add_done_callback(on_result)

        send_future = self._pivot_client.send_goal_async(goal)
        send_future.add_done_callback(on_goal_response)

        goal_done.wait()
        if "error" in outcome:
            raise outcome["error"]

        result = outcome["result"]
        self.get_logger().info(f"pivot termine : success={result.success}")
        return result.success

    def _get_fresh_carton_pose(self, max_pose_age=1.5):
        """Renvoie (x, y) en repere robot depuis /vision/carton_pose_base si
        une detection est arrivee il y a moins de `max_pose_age` secondes,
        sinon None -- une detection perimee (marqueur sorti du champ) ne doit
        jamais etre reutilisee telle quelle (constate le 19/08 : sans ce
        garde-fou, approach_carton tournait sur une valeur figee jusqu'a
        epuiser max_pulses, 9 iterations identiques observees)."""
        with self._carton_pose_lock:
            pose = self._carton_pose
            received_at = self._carton_pose_received_at
        age = None if received_at is None else time.monotonic() - received_at
        if pose is None or age is None or age > max_pose_age:
            self.get_logger().warn(
                f"_get_fresh_carton_pose : aucune detection fraiche (age="
                f"{age if age is not None else 'n/a'})."
            )
            return None
        return pose.pose.position.x, pose.pose.position.y

    def walk_to_xy(self, target_x: float, target_y: float, standoff: float = 0.42,
                    forward_speed: float = 0.5, turn_gain: float = 1.5, max_turn: float = 0.6,
                    speed_calibration: float = 0.255) -> bool:
        """Marche en UN SEUL mouvement continu vers une position (repere
        robot) detectee UNE FOIS, plutot que la boucle detecter->pulse
        court->re-detecter d'approach_carton(). Corrige la vraie cause du
        mouvement saccade constate le 19/08 : `walk_to.py::_enter_walk()`
        (pulse LB+B + 2s d'attente AVANT de pousser les manettes) est refait
        a CHAQUE appel a l'Action walk_to, meme en plein milieu d'une marche
        deja active -- avec 15-25 pulses de 1s chacun, le robot passait ~2/3
        du temps a l'arret entre deux pas de marche reelle. Un seul appel =
        un seul `_enter_walk()`, plus de coupures.

        `turn` reutilise le meme gain/signe que approach_carton (verifie
        empiriquement le 19/08 : turn = +turn_gain * y). Ne modifie PAS
        WalkTo.action ni walk_to.py -- reste un appel Action ouvert classique
        (decision du 17/08, cf memoire projet), juste calcule ICI a partir
        d'une cible x,y au lieu de forward/turn/duration donnes directement.

        Compromis assume (a verifier empiriquement, PAS aussi robuste que
        approach_carton) : `turn` est calcule UNE FOIS et garde CONSTANT
        pendant toute la duree du trajet -- ca trace un arc, pas une ligne
        droite, si turn != 0, et rien ne corrige une erreur de calibration
        de vitesse/virage en cours de route (contrairement a approach_carton
        qui re-detecte a chaque pulse, au prix du saccade) -- VOULU : la
        vision ne recale qu'AVANT (cible x,y) et APRES (passe fine
        approach_carton dans run_sequence), jamais PENDANT ce mouvement.
        `speed_calibration=0.255m/s` (etait 0.19, sous-estimee) -- recalibre
        le 20/08 a partir de 2 essais reels avec cette methode : le robot
        parcourait ~0.253-0.258m/s reel a forward=0.5, pas 0.19 -- l'ancienne
        valeur venait d'une mesure differente (marche a duree fixe, pas ce
        chemin de code) et faisait systematiquement DEPASSER la cible
        (observe : viser standoff=0.8m amenait le robot a ~0.32-0.36m,
        constate en enquetant sur "il marche trop au debut, vision pas
        censee compenser" -- reponse : conforme au design (boucle ouverte
        assumee), mais la calibration elle-meme etait fausse, corrigee ici."""
        distance = max(0.0, target_x - standoff)
        if distance <= 0.0:
            self.get_logger().info("walk_to_xy : deja a la distance cible, rien a faire.")
            return True
        turn = max(-max_turn, min(max_turn, turn_gain * target_y))
        duration = distance / speed_calibration
        self.get_logger().info(
            f"walk_to_xy : cible=({target_x:.3f},{target_y:.3f}) -> "
            f"forward={forward_speed} turn={turn:.3f} duration={duration:.2f}s"
        )
        return self.walk_to(forward=forward_speed, turn=turn, duration=duration)

    def approach_carton(self, standoff=0.42, y_tolerance=0.03, forward_speed=0.5,
                         pulse_duration=1.0, turn_gain=1.5, max_turn=0.6,
                         max_pulses=10, settle_seconds=0.5, max_pose_age=1.5) -> bool:
        """Approche iterative pilotee par la vision (carton_pose_publisher.py) :
        boucle detecter -> pulse walk_to() COURT -> re-detecter, jusqu'a etre a
        `standoff` metres du marqueur (repere robot, axe X = distance devant)
        et centre lateralement (|y| < y_tolerance). Ne modifie PAS WalkTo.action
        ni walk_to.py -- chaque appel reste un pulse en boucle OUVERTE (decision
        du 17/08, cf memoire projet), le servoing visuel se fait ICI, au niveau
        de la choregraphie, pas dans le node walk_to lui-meme.

        Signe de `turn` CORRIGE (verifie empiriquement le 19/08, cf memoire
        projet) : `turn = -turn_gain * y` (deduit par analogie d'un pivot en
        pd_stand) divergeait reellement en marche active -- y partait de
        ~0 et derivait jusqu'a -0.84 en 5 pulses au lieu de converger vers 0.
        Le signe correct est `turn = +turn_gain * y`.

        Plus lent/saccade que walk_to_xy() (chaque pulse refait
        _enter_walk(), voir docstring walk_to_xy) mais s'auto-corrige en
        route -- utile en fine-tuning apres un walk_to_xy() approximatif, ou
        si la calibration de vitesse/virage n'est pas fiable."""
        for i in range(max_pulses):
            got = self._get_fresh_carton_pose(max_pose_age)
            if got is None:
                self.get_logger().warn("approach_carton : abandon (aucune detection fraiche).")
                return False
            x, y = got
            self.get_logger().info(f"approach_carton[{i}] : x={x:.3f} y={y:.3f} (cible x<={standoff+0.05:.3f} |y|<={y_tolerance})")
            if x - standoff <= 0.05 and abs(y) <= y_tolerance:
                self.get_logger().info("approach_carton : cible atteinte.")
                return True

            turn = max(-max_turn, min(max_turn, turn_gain * y))
            fwd = forward_speed if x > standoff else 0.0
            self.walk_to(forward=fwd, turn=turn, duration=pulse_duration)
            time.sleep(settle_seconds)  # laisse une detection fraiche arriver avant le prochain tour

        self.get_logger().warn(f"approach_carton : {max_pulses} pulses sans converger, abandon.")
        return False

    def run_sequence(self) -> None:
        """Choregraphie du chef -- seul endroit qui sait quelle sequence executer.
        Detecte le carton UNE FOIS apres stand() puis marche vers lui en UN
        SEUL mouvement continu (walk_to_xy, cf. plus haut) -- remplace la
        boucle approach_carton() (detecter->pulse court->re-detecter)
        utilisee dans une version precedente de cette sequence le 19/08 :
        saccadee car walk_to.py::_enter_walk() (2s d'attente) est refait a
        chaque pulse. walk_to_xy() ne fait ca qu'une fois. Necessite
        pelvis_camera_sim.py ET carton_pose_publisher.py lances en plus de
        walk_to/stand (voir docs/COMMENT_TESTER_CAMERA_FANTOME.txt).

        Compromis assume (voir docstring walk_to_xy) : approche en boucle
        OUVERTE apres la detection initiale, pas de correction en cours de
        route -- moins precis qu'approach_carton() mais fluide.

        Recentrage fin AJOUTE apres walk_to_xy() (constate le 20/08 : sans
        ca, le robot tremble en levant le carton et "fait n'importe quoi"
        apres -- lift.py calcule ses cibles de pince de facon SYMETRIQUE
        (+-pinch_y autour de Y=0, voir lift.py::_execute), ce qui suppose le
        robot bien centre face au carton. walk_to_xy() seul laisse un residu
        lateral de 0.14-0.24m dans les essais faits -- suffisant pour "etre
        a cote" mais pas pour une prise a deux bras symetrique : une pince
        rate son point de contact pendant que l'autre serre normalement,
        d'ou le desequilibre/tremblement pendant la levee.

        `walk_to_xy()` + `approach_carton()` visent un standoff de 0.25m
        (essaye 0.6m d'abord le 20/08 pour la stabilite -- confirme stable
        2/2, MAIS l'utilisateur a signale que les bras n'ouvraient plus
        assez pour attraper le carton). Diagnostic direct (solve_ik +
        forward_kinematics, hors ROS) : le bras (LEFT_CHAIN, pinch_y=0.28,
        pinch_z=-0.054) atteint sa cible avec une erreur negligeable jusqu'a
        pinch_x~0.38-0.39, PUIS l'erreur explose au-dela de 0.40 -- ET ca
        degrade l'ecartement lateral EN MEME TEMPS (le solveur IK sacrifie
        les deux axes a la fois pour rester dans son enveloppe de portee).
        A pinch_x=0.49 (calcule pour standoff=0.6), l'ecartement Y reellement
        atteint tombe a ~0.265 au lieu des 0.28 demandes, ET la profondeur X
        plafonne a ~0.40 -- d'ou "les bras n'ouvrent pas assez". La cause
        n'etait donc PAS un pinch_y trop petit (l'augmenter aurait aggrave le
        probleme, moins de marge de portee disponible) mais standoff trop
        grand pour la portee reelle du bras. pinch_x adaptatif =
        marqueur_x+0.1375 (cf plus bas) -- pour retomber pres de 0.38 (bon
        fit IK), il faut marqueur_x~0.24, d'ou standoff=0.25. La flexion des
        genoux (cf plus haut) reste le fix de stabilite reel, pas la
        distance -- garde meme a cette distance plus proche.
        standoff PASSE AUX DEUX appels (walk_to_xy ET approach_carton) pour
        rester coherent : sans ca, `approach_carton` continuerait a comparer
        x a son propre defaut (0.42).

        STANDOFF=0.40 (0.25 puis 0.30 juges encore trop pres du podium par
        l'utilisateur le 20/08, malgre le fix forward_speed=0.0 de la passe
        fine -- le souci etait bien la distance d'arret elle-meme, pas
        juste l'avance parasite). Le podium (plateau 0.19x0.17m,
        `pm01_edu_carton.xml`) depasse de 0.0525m DEVANT le carton lui-meme
        (bord podium en X monde=2.01 vs face avant carton=2.0625) --
        clearance robot->bord de plateau = standoff-0.0525 :
        0.25->0.197m, 0.30->0.247m, 0.40->0.348m. Verifie via
        solve_ik+forward_kinematics hors ROS que la degradation de portee
        du bras reste GRADUELLE, pas un mur : Y atteint 97% a standoff=0.30,
        94% a 0.40, chute vraiment a 83% seulement a 0.45 -- 0.40 est donc
        un bon compromis (marge podium quasi 35cm, perte de prise encore
        marginale)."""
        STANDOFF = 0.53
        # Marge/plafond ajoutes le 20/08 (retour utilisateur : un walk_to_xy()
        # qui sous-shoote gravement -- x reste a 1.77m au lieu de ~0.30 --
        # n'etait PAS re-tente, chef enchainait quand meme sur lift() avec un
        # pinch_x=1.9 absurde (bras ne peut pas atteindre ca), resultat
        # "n'importe quoi". "Avec la vision il aurait du recommencer sa
        # marche" -- exactement ce que fait la boucle ci-dessous desormais :
        # re-detecter + re-marcher tant qu'on n'est pas raisonnablement pres,
        # et refuser purement et simplement lift() si on ne l'est jamais
        # (plutot que d'envoyer un pinch_x hors de portee).
        CLOSE_ENOUGH_X = STANDOFF + 0.15  # tolerance avant de considerer l'approche reussie
        MAX_REACH_PINCH_X = 0.55  # au-dela, refus pur (prise dans le vide, pas de recuperation possible)
        # RELIABLE_REACH_PINCH_X=0.40 (20/08, retour utilisateur : "il s'appuie
        # sur la table, l'IK vise loin") : verifie via solve_ik+forward_kinematics
        # que le bras suit sa cible avec une erreur negligeable jusqu'a ~0.38-0.40,
        # puis l'erreur EXPLOSE au-dela. Demander plus loin que ca (ex: 0.45-0.51,
        # frequent avec STANDOFF>=0.40) ne fait PAS juste rater la cible en douceur --
        # le robot semble compenser en penchant le buste vers l'avant jusqu'a
        # toucher/s'appuyer sur le podium. Plafonner ICI (pas juste refuser) pour
        # ne JAMAIS demander a l'IK plus que ce qu'il sait faire proprement.
        RELIABLE_REACH_PINCH_X = 0.40
        CARTON_HALF_X = 0.1375
        MAX_ATTEMPTS = 3

        self.stand()
        x_final = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # got == None juste apres stand() peut arriver meme quand la
            # camera tourne bien (cas degenere IPPE_SQUARE robot pile de
            # face, cf memoire projet, normalement rattrape par le repli
            # ITERATIVE -- mais peut prendre plus d'un cycle camera 4Hz).
            # Reessayer une poignee de fois AVEC un court delai avant
            # d'abandonner cette tentative -- sans ca, les 3 tentatives de
            # la boucle externe se consomment instantanement (aucune chance
            # qu'une nouvelle image arrive entre deux), constate le 20/08.
            got = None
            for _ in range(6):
                got = self._get_fresh_carton_pose()
                if got is not None:
                    break
                time.sleep(0.5)
            if got is None:
                self.get_logger().error(
                    f"run_sequence[tentative {attempt}] : aucune detection apres 3s, walk_to_xy() non tente."
                )
                continue
            x, y = got
            ok = self.walk_to_xy(x, y, standoff=STANDOFF)
            if not ok:
                self.get_logger().error(f"run_sequence[tentative {attempt}] : walk_to_xy() a echoue.")
                continue
            time.sleep(1.0)  # laisse la camera (4Hz) produire une detection fraiche de la pose post-marche
            # forward_speed=0.0 ICI (20/08, retour utilisateur : "trop pres du
            # podium, il s'arrete pas assez tot") : approach_carton() pousse
            # fwd=forward_speed (marche) des que x>standoff, MEME de tres peu
            # (x=0.320 pour standoff=0.30 a deja declenche un pulse complet de
            # 1s a 0.5m/s, soit ~0.25m d'avance supplementaire non voulue) --
            # walk_to_xy() a deja couvert toute la distance necessaire, cette
            # passe fine ne doit servir qu'a corriger l'angle (pivot pur), pas
            # a re-avancer.
            self.approach_carton(max_pulses=2, standoff=STANDOFF, forward_speed=0.0)

            got_check = self._get_fresh_carton_pose(max_pose_age=10.0)
            if got_check is None:
                self.get_logger().warn(f"run_sequence[tentative {attempt}] : position finale inconnue.")
                continue
            x_final = got_check[0]
            if x_final <= CLOSE_ENOUGH_X:
                self.get_logger().info(
                    f"run_sequence[tentative {attempt}] : assez pres (x={x_final:.3f} <= "
                    f"{CLOSE_ENOUGH_X:.3f}), approche terminee."
                )
                break
            self.get_logger().warn(
                f"run_sequence[tentative {attempt}] : encore trop loin (x={x_final:.3f} > "
                f"{CLOSE_ENOUGH_X:.3f}) -- nouvelle tentative de marche."
            )
        else:
            self.get_logger().error(
                f"run_sequence : jamais assez pres apres {MAX_ATTEMPTS} tentatives -- lift() abandonne."
            )
            return

        # pinch_x ADAPTATIF (20/08) : lift.py utilise des cibles IK fixes
        # (pinch_x=0.38 par defaut) qui supposent le robot a une distance
        # PRECISE du carton, jamais verifiee contre la marche vision. Formule
        # tiree de tools/robot_arm_ik/lift_carton.py::distance_to_carton()
        # (`CARTON_XY[0] - pinch_x - home_x == 0`) : pinch_x = distance
        # robot->CENTRE du carton, PAS distance au marqueur (qui est sur la
        # face avant). marqueur_x + CARTON_HALF_X (0.1375m, demi-profondeur
        # du carton) = distance au centre.
        pinch_x_adaptive = x_final + CARTON_HALF_X
        if pinch_x_adaptive > MAX_REACH_PINCH_X:
            self.get_logger().error(
                f"run_sequence : pinch_x calcule ({pinch_x_adaptive:.3f}) hors de la portee "
                f"fiable du bras (>{MAX_REACH_PINCH_X}) -- lift() abandonne plutot que de "
                "tenter une prise dans le vide."
            )
            return
        if pinch_x_adaptive > RELIABLE_REACH_PINCH_X:
            self.get_logger().warn(
                f"lift() : pinch_x calcule ({pinch_x_adaptive:.3f}) au-dela de la portee "
                f"fiable ({RELIABLE_REACH_PINCH_X}) -- plafonne pour eviter que le robot "
                "ne compense en penchant le buste vers le podium."
            )
            pinch_x_adaptive = RELIABLE_REACH_PINCH_X
        self.get_logger().info(
            f"lift() : pinch_x adaptatif = {pinch_x_adaptive:.3f} "
            f"(marqueur_x={x_final:.3f} + demi-profondeur {CARTON_HALF_X})"
        )
        # pinch_y augmente 0.28 -> 0.35 (20/08, demande utilisateur "augmente
        # l'ecart des bras") : verifie via solve_ik+forward_kinematics a
        # pinch_x~0.45-0.47 (plage reelle observee) que l'ecartement PHYSIQUE
        # reellement atteint augmente (0.271 -> ~0.31-0.33m, +15-20%) malgre
        # une perte de precision en % (97%->91-93%) -- reste dans une plage
        # ou X ne se degrade que legerement (0.40->0.385-0.39m).
        PINCH_Y = 0.35
        # Pivot DESACTIVE (20/08, demande utilisateur "annule la rotation") --
        # release_after repasse a True (defaut) pour que lift() relache
        # normalement lui-meme, self.pivot() plus appelee. Code du pivot
        # garde intact (Pivot.action/pivot.py) pour reactivation future,
        # juste plus chaine ici.
        self.lift(pinch_x=pinch_x_adaptive, pinch_y=PINCH_Y)


def main():
    rclpy.init()
    node = ChefNode(rate_hz=20.0)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run_sequence()
        # NE PAS quitter ici (20/08) : le process sortait immediatement apres
        # run_sequence(), coupant le heartbeat LCM de chef (_publish_to_lcm,
        # 20Hz) -- l'arbitre de commande cote robot detecte l'absence de
        # heartbeat en ~200ms et "Releasing adapter control" (comportement
        # documente et attendu quand le robot est simplement DEBOUT, voir
        # memoire projet 17/08 -- mais catastrophique ici, robot bras tendus
        # en train de tenir le carton juste apres lift(), perd sa source de
        # commande d'un coup). Garder le process (et donc le heartbeat) vivant
        # apres la choregraphie -- Ctrl+C pour arreter proprement.
        node.get_logger().info(
            "run_sequence() termine -- heartbeat LCM maintenu (Ctrl+C pour arreter)."
        )
        spin_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        # Un 2e Ctrl-C pendant le nettoyage relance un KeyboardInterrupt en
        # plein milieu -- on est deja engages dans l'arret, donc on ignore les
        # SIGINT supplementaires (voir aussi walk_to.py).
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # executor.spin() tourne dans spin_thread en parallele du thread
        # principal -- appeler rclpy.shutdown() pendant qu'il spin encore fait
        # planter rclpy/rcl (race C -- "terminate called without an active
        # exception"). Il faut d'abord arreter l'executeur et joindre le
        # thread, PUIS fermer le node et le contexte.
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
