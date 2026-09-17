# Le code de la simulation, bloc par bloc

Ce document explique **tout** le code de la simulation
(`tools/virtual_gamepad/ros_ws/src/virtual_gamepad_ros/virtual_gamepad_ros/`)
-- `chef_node.py` et les 5 Action Servers qu'il orchestre, plus les
fichiers de support, **du début à la fin de chaque méthode**. Pour
l'équivalent robot réel, voir [`CODE_ROBOT_REEL.md`](CODE_ROBOT_REEL.md) --
les deux partagent la même cinématique inverse (`solve_arm_ik`,
`forward_kinematics`...), déjà expliquée là-bas ligne par ligne ; ce
document ne la reproduit pas mais ne saute AUCUN bloc qui lui est propre.

Ordre de lecture : les fondations d'abord (topics, géométrie, IK
partagée), puis chaque Action Server dans l'ordre où `chef_node.py` les
appelle (`stand` → `walk_to` → `lift` → `pivot` → `depose`),
l'orchestrateur en dernier.

---

## 1. `field_topics.py` (21 lignes) -- la table de correspondance manette virtuelle

```python
BUTTON_INDEX = {
    "LB": 0, "RB": 1, "A": 2, "B": 3, "X": 4, "Y": 5,
    "BACK": 6, "START": 7, "UP": 8, "DOWN": 9, "LEFT": 10, "RIGHT": 11,
}
ANALOG_INDEX = {
    "LEFT_STICK_X": 2, "LEFT_STICK_Y": 3, "RIGHT_STICK_X": 4, "RIGHT_STICK_Y": 5,
}
# LT/RT (analog_states[0]/[1]) non exposes ici -- ajoute-les si un node en a besoin.

def field_topic(name: str) -> str:
    return f"/virtual_gamepad/cmd/{name.lower()}"
```

Chaque bouton/stick de la manette virtuelle correspond à un topic ROS2
séparé (`/virtual_gamepad/cmd/lb`, `/virtual_gamepad/cmd/left_stick_x`,
etc.) -- un `Bool` par bouton, un `Float32` par stick.
`ANALOG_INDEX`/`BUTTON_INDEX` donnent l'index dans le message LCM
`GamepadKeys` (`digital_states`/`analog_states`) correspondant à chaque
nom -- `chef_node.py` les utilise pour peupler ce message, `stand.py` et
`body_vel_bridge.py` pour savoir sur quel topic publier "appuyer sur LB".

---

## 2. `lift_carton.py` -- ce qui est spécifique à la simulation

Le fichier complet (`tools/robot_arm_ik/lift_carton.py`) contient la même
cinématique inverse que la version robot réel (chaînes,
`forward_kinematics`, `solve_ik`, `solve_arm_ik`, `mirror_left_to_right`,
`ease` -- identiques ligne pour ligne, voir la section 1 de
[`CODE_ROBOT_REEL.md`](CODE_ROBOT_REEL.md)),
plus 3 éléments **uniquement utiles en simulation** :

### Constantes de scène (lignes 55-75)

```python
REPO_DIR = "/home/equansrobotic/engineai_robotics_native_sdk"
ROBOT_XML = os.path.join(REPO_DIR, "assets/resource/robot/pm01_edu/xml/serial_pm01_edu.xml")
ROBOT_DIR = os.path.dirname(ROBOT_XML)
ENV_DIR = os.path.join(REPO_DIR, "assets/resource/environment")

CARTON_XY = (1.8, 0.0)
CARTON_Z = 0.796
# CARTON_XY/CARTON_Z ci-dessus sont des constantes FIGEES, perimees --
# gardees pour ne pas casser build_dynamic_scene() (outil de calibrage IK
# hors ligne). NE PAS s'en servir pour un calcul touchant la scene
# VIVANTE, utiliser carton_face_centers() a la place.

LIVE_SCENE_XML = "/home/equansrobotic/engineai_robotics_native_sdk/assets/resource/pm01_edu_carton.xml"
SQUEEZE_OFFSET_Y = 0.010
```

`CARTON_XY`/`CARTON_Z` sont des vestiges d'un outil de calibrage
hors-ligne (`build_dynamic_scene`, génère sa PROPRE scène de test, sans
rapport avec `run_mujoco.sh`) -- explicitement marqués comme périmés dans
le code, jamais utilisés par `lift.py`/`pivot.py`/`depose.py`.
`SQUEEZE_OFFSET_Y=0.010` : le même décalage de serrage utilisé sur le
robot réel, partagé ici via ce fichier commun.

### `carton_face_centers()` (lignes 78-121) -- lire la position du carton dans la scène MuJoCo

```python
def carton_face_centers(scene_xml=LIVE_SCENE_XML):
    src = open(scene_xml).read()
    m = re.search(r'<body name="carton" pos="([^"]+)"', src)
    if not m:
        raise RuntimeError(f"Body 'carton' introuvable dans {scene_xml}.")
    cx, cy, cz = (float(v) for v in m.group(1).split())
    m = re.search(r'<geom name="carton_box"[^>]*\bsize="([^"]+)"', src)
    if not m:
        raise RuntimeError(f"Geom 'carton_box' introuvable dans {scene_xml}.")
    sx, sy, sz = (float(v) for v in m.group(1).split())
    face_gauche = np.array([cx, cy + sy, cz])
    face_droite = np.array([cx, cy - sy, cz])
    return face_gauche, face_droite
```

Lit **directement le fichier XML** de la scène MuJoCo en cours
d'exécution avec une regex, plutôt que de coder en dur la position du
carton -- pour ne jamais avoir une valeur périmée. `size` dans MuJoCo est
une **demi-dimension** (d'où `cy + sy`/`cy - sy` pour les deux faces).
Conventions du repère (documentées en détail dans la docstring du
fichier) : +X = vers l'avant, +Z = vers le haut, +Y = vers la GAUCHE du
robot (même convention que `pinch_y` partout ailleurs). Le carton n'a
aucune rotation dans la scène -- ses faces sont donc exactement à
`Y = cy ± sy`, mêmes X/Z que le centre.

### `SimStateListener` (lignes 137-181) -- lire la position réelle du robot dans MuJoCo

```python
SIM_LCM_URL = "udpm://239.255.76.67:7667?ttl=1"
SIM_STATE_CHANNEL = "sim_state"

def _yaw_from_quaternion(w, x, y, z):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

class SimStateListener:
    def __init__(self, lcm_url=SIM_LCM_URL):
        sys.path.insert(0, "/home/equansrobotic/engineai_robotics_native_sdk/tools/virtual_gamepad")
        from lcm_msgs.data import SimState
        self._SimState = SimState
        self._lc = lcm.LCM(lcm_url)
        self._lc.subscribe(SIM_STATE_CHANNEL, self._on_message)
        self._latest = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _on_message(self, channel, data):
        state = self._SimState.decode(data)
        with self._lock:
            self._latest = state

    def _spin(self):
        while True:
            self._lc.handle()

    def wait_for_first_message(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._latest is not None:
                    return True
            time.sleep(0.05)
        return False

    def pose(self):
        with self._lock:
            state = self._latest
        if state is None:
            return None
        x, y, z = state.base_link_position
        w, qx, qy, qz = state.base_link_quaternion
        return x, y, z, _yaw_from_quaternion(w, qx, qy, qz)
```

`_yaw_from_quaternion` : extrait le cap (rotation autour de Z) d'un
quaternion `(w,x,y,z)` -- formule standard `atan2` du yaw, ignore
roll/pitch (le robot est supposé rester vertical). `SimStateListener`
s'abonne au canal LCM `sim_state` -- la **vérité terrain** publiée par
MuJoCo. `__init__` démarre un THREAD séparé (`_spin`, `daemon=True`) qui
boucle sur `self._lc.handle()` (bloquant, traite les messages LCM entrants)
-- indispensable car LCM n'a pas son propre mécanisme d'exécuteur comme
ROS2, il faut "pomper" les messages manuellement. `_on_message` protège
l'écriture de `_latest` par un verrou (`_lock`) car appelé depuis le
thread `_spin`, lu depuis le thread principal ROS2. `wait_for_first_message`
: poll jusqu'à recevoir au moins un message (utile juste après la
connexion, où `_latest` vaut encore `None`). `pose()` : renvoie
`(x, y, z, yaw)` du bassin, `None` si aucun message reçu.

### `world_to_robot_local()` (lignes 184-212) -- convertir une position monde en cible IK

```python
def world_to_robot_local(world_point, base_pose):
    base_x, base_y, base_z, base_yaw = base_pose
    dx = world_point[0] - base_x
    dy = world_point[1] - base_y
    dz = world_point[2] - base_z
    c, s = math.cos(base_yaw), math.sin(base_yaw)
    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c
    return np.array([local_x, local_y, dz])
```

Combine les deux fonctions précédentes : soustrait la position du bassin
(`dx`/`dy`/`dz`), puis applique la rotation INVERSE du cap du robot
(`c`/`s` = cos/sin de `base_yaw`, la formule `local_x = dx·c + dy·s` /
`local_y = -dx·s + dy·c` est la rotation d'angle `-base_yaw`) -- seule la
rotation autour de Z compte, le robot est supposé rester vertical.
Résultat directement utilisable comme cible pour `solve_arm_ik`. Appelée
à **chaque** exécution de `lift`/`pivot`/`depose` (jamais mise en cache).

---

## 3. `chef_node.py` (371 lignes) -- l'orchestrateur

### Imports et constantes géométriques (lignes 1-92)

```python
import signal, sys, threading, time
import lcm
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32

sys.path.insert(0, "/home/equansrobotic/engineai_robotics_native_sdk/tools/virtual_gamepad")
from lcm_msgs.data import GamepadKeys

sys.path.insert(0, "/home/equansrobotic/stagiaire_1/tools/robot_arm_ik")
from lift_carton import carton_face_centers

from virtual_gamepad_interfaces.action import Depose, Lift, Pivot, Stand, WalkTo
from virtual_gamepad_ros.field_topics import ANALOG_INDEX, BUTTON_INDEX, field_topic

LCM_URL = "udpm://239.255.76.67:7667?ttl=1"
CHANNEL = "virtual_gamepad/gamepad_keys"
SERVER_TIMEOUT_S = 10.0
GOAL_TIMEOUT_S = 300.0

PROVEN_PINCH_X = 0.345
PINCH_Y = 0.22
SQUEEZE_Y = 0.095
PINCH_Z = 0.106
LIFT_Z = 0.20

WALK_FORWARD_MPS = 0.45
```

`GamepadKeys` (LCM) est le message que ce node construit et republie en
continu -- l'émulation complète d'une manette physique. `PROVEN_PINCH_X`
etc. ont un très long historique en commentaire dans le fichier (plusieurs
itérations pour trouver une distance/largeur d'approche qui évite à la
fois une posture de bras tordue et une violation des limites articulaires
réelles -- voir le fichier source pour le raisonnement complet). Ces
valeurs ne servent presque plus depuis le passage à la visée par
`carton_face_centers()` : `lift.py` vise directement le centre réel des
faces, pas une position calculée depuis ces constantes. `WALK_FORWARD_MPS
= 0.45` : LA seule valeur de vitesse de marche utilisée, choisie car
c'est le seul point où la calibration du pont manette (`body_vel_bridge.py`)
est fidèle.

### `__init__` -- deux rôles en un seul node (lignes 95-114)

```python
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
```

S'abonne à TOUS les topics `/virtual_gamepad/cmd/*` (une boucle sur
`BUTTON_INDEX`, une sur `ANALOG_INDEX`) et crée un `ActionClient` par
Action Server (`stand`/`walk_to`/`lift`/`pivot`/`depose`). `create_timer`
déclenche `_publish_to_lcm` 20 fois par seconde -- le relais continu vers
MuJoCo, indépendant de la séquence.

### Callbacks et relais LCM (lignes 117-129)

```python
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
```

`_button_cb`/`_analog_cb` sont des **fabriques de callbacks** (closures) :
chaque appel crée une fonction qui capture `idx` -- nécessaire car
`create_subscription` a besoin d'UNE fonction par topic, mais toutes
doivent écrire dans le MÊME tableau `self._state.digital_states` à un
index différent. `_publish_to_lcm` : encode et publie l'état complet
accumulé, avec un timestamp en microsecondes.

### `_send_goal` (lignes 132-159) -- rendre une Action asynchrone bloquante

```python
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
```

Le patron classique pour transformer une Action ROS2 asynchrone (callbacks)
en appel synchrone/bloquant : `wait_for_server` échoue vite si le node
cible n'existe pas. `goal_done` (un `threading.Event`) se déclenche quand
le résultat arrive -- `goal_done.wait(...)` bloque `run_sequence()`
jusque-là. `on_goal_response` : si le but est refusé
(`goal_handle.accepted == False`), stocke l'erreur et débloque tout de
suite ; sinon enchaîne sur `get_result_async` (attend la FIN de
l'exécution, pas juste l'acceptation). `on_result` : récupère le résultat
final et débloque. À la fin, propage toute erreur stockée
(`raise outcome["error"]`), sinon retourne `success`.

**Point d'attention** : `success=True` ne garantit PAS que le mouvement a
réellement eu l'effet physique attendu -- juste que l'Action Server a
terminé sans erreur logicielle. Vérifier via la télémétrie `sim_state`
(voir `tools/vision/log_sim_pose.py`), pas seulement ce booléen.

### `_goal_fields` et les méthodes de haut niveau (lignes 161-179)

```python
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
```

`_goal_fields` : convertit automatiquement tous les kwargs numériques en
`float` (sauf `bool`/`str`, laissés tels quels) -- évite d'écrire
`float(...)` sur chacun des ~15 paramètres de `lift`/`pivot`/`depose` à
chaque appel. `lift`/`pivot`/`depose` acceptent `**kwargs` génériques
(construits dynamiquement en `Goal(**...)`), alors que `stand`/`walk_to`
ont des paramètres nommés explicites -- différence purement historique
(ces deux actions ont moins de paramètres, jamais refactorées vers le
même style `**kwargs`).

### `_publish_step` (lignes 181-185)

```python
def _publish_step(self, step: int) -> None:
    self._step_pub.publish(Int32(data=int(step)))
    self.get_logger().info(f"--- etape {step} ---")
```

Numéro d'étape GRAFCET courant, observable via `ros2 topic echo
/chef/current_step` -- purement informatif, ne pilote rien.

### `run_sequence()` (lignes 187-344) -- la chorégraphie complète

```python
def run_sequence(self) -> None:
    WALK_DURATION = 2.3
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
```

Constantes locales à la séquence (pas des constantes de module, car
propres à CETTE chorégraphie). Étape 0 : `stand()` (position debout de
départ). Étape 10 : marche avant (`WALK_FORWARD_MPS=0.45`, `WALK_DURATION
=2.3` -- distance jugée sûre par l'utilisateur, plus proche fait toucher
le podium), `time.sleep(2.0)` laisse le temps à la marche de se stabiliser
avant un nouveau `stand()`. Chaque appel `if not self.xxx(...): ... return`
arrête TOUTE la séquence dès le premier échec -- pas de tentative de
reprise.

```python
    self._publish_step(20)
    pinch_x = PROVEN_PINCH_X
    face_gauche_monde, face_droite_monde = carton_face_centers()
    face_kwargs = dict(
        face_gauche_x=float(face_gauche_monde[0]), face_gauche_y=float(face_gauche_monde[1]),
        face_gauche_z=float(face_gauche_monde[2]),
        face_droite_x=float(face_droite_monde[0]), face_droite_y=float(face_droite_monde[1]),
        face_droite_z=float(face_droite_monde[2]),
    )
```

**`carton_face_centers()` appelé UNE SEULE FOIS** ici (étape 20), le
résultat empaqueté dans `face_kwargs` -- transmis tel quel à CHACUN des
appels `lift`/`pivot`/`depose` suivants (via `**face_kwargs`) plutôt que
laisser chaque node relire lui-même le XML. Prépare le terrain pour la
vision réelle (où la mesure ne serait prise qu'une fois, pas répétée).

```python
    self._publish_step(30)
    if not self.lift(pinch_x=pinch_x, pinch_y=PINCH_Y, pinch_z=PINCH_Z, squeeze_y=SQUEEZE_Y,
                      approach_duration=4.0, walk_stance_scale=WALK_STANCE_SCALE,
                      only_phase="approche", release_after=False, **face_kwargs):
        self.get_logger().error("run_sequence : lift() a echoue (approche) -- arret.")
        return

    self._publish_step(40)
    if not self.lift(pinch_x=pinch_x, pinch_y=PINCH_Y, pinch_z=PINCH_Z, squeeze_y=SQUEEZE_Y,
                      squeeze_duration=5.0, walk_stance=False, only_phase="serrage",
                      release_after=False, **face_kwargs):
        self.get_logger().error("run_sequence : lift() a echoue (serrage) -- arret.")
        return

    self._publish_step(50)
    if not self.lift(pinch_x=pinch_x, pinch_y=PINCH_Y, pinch_z=PINCH_Z, squeeze_y=SQUEEZE_Y,
                      lift_z=LIFT_Z, lift_duration=3.5, hold_seconds=1.0, walk_stance=False,
                      walk_stance_scale=WALK_STANCE_SCALE, only_phase="levee", release_after=False,
                      **face_kwargs):
        self.get_logger().error("run_sequence : lift() a echoue (levee) -- arret.")
        return
```

**`lift()` appelé 3 fois**, chacune une Action ROS2 SÉPARÉE (voir
`_publish_step` entre chaque -- permet d'observer étape par étape via
`ros2 topic echo /chef/current_step`). `pinch_x`/`pinch_y`/`pinch_z`/
`squeeze_y` transmis à chaque appel mais **ne servent presque plus**
depuis le passage à `carton_face_centers()` (voir section 6, `lift.py`
recalcule sa cible depuis `face_kwargs`, pas depuis ces 4 valeurs).
`release_after=False` partout : le carton reste tenu d'un appel à l'autre.

```python
    self._publish_step(60)
    PIVOT_ANGLE_DEG = 90.0
    if not self.pivot(pinch_x=pinch_x, pinch_y=PINCH_Y, pinch_z=PINCH_Z, squeeze_y=SQUEEZE_Y,
                       lift_z=LIFT_Z, angle_deg=PIVOT_ANGLE_DEG, walk_stance_scale=WALK_STANCE_SCALE,
                       pivot_duration=1.5, hold_seconds=1.0,
                       release_after=False, free_legs_for_walk=False,
                       depivot_before_release=False, **face_kwargs):
        self.get_logger().error(f"run_sequence : pivot({PIVOT_ANGLE_DEG:.0f}) a echoue -- arret.")
        return
```

`PIVOT_ANGLE_DEG=90.0` : la valeur retenue après plusieurs itérations
(voir le long historique en commentaire du fichier -- 60°/90° étaient
initialement bloqués pour une raison de gains PD insuffisants, corrigée
depuis). `release_after=False, free_legs_for_walk=False` : ni
relâchement ni marche entre pivot et dépose -- le carton reste tenu, les
jambes restent dans leur posture de pivot. `depivot_before_release=False`
: le buste reste tourné quand `pivot()` rend la main -- c'est `depose()`
qui gère le dépivotage (voir section 8).

```python
    self._publish_step(80)
    if not self.depose(pinch_x=pinch_x, pinch_y=PINCH_Y, pinch_z=LIFT_Z, squeeze_y=SQUEEZE_Y,
                        hold_z=LIFT_Z, drop_z=PINCH_Z,
                        depivot_from_deg=PIVOT_ANGLE_DEG, depivot_duration=1.5,
                        tendre_duration=2.0, depose_duration=2.0,
                        degagement_waypoint_duration=2.0, degagement_duration=2.0,
                        **face_kwargs):
        self.get_logger().error("run_sequence : depose() a echoue -- arret.")
        return

    self.stand(settle_seconds=3.0)
    self.get_logger().info("run_sequence : sequence complete terminee.")
    self._publish_step(0)
```

`hold_z=LIFT_Z` (PAS un défaut périmé de `Depose.action`) : la hauteur où
`pivot()` tenait RÉELLEMENT le carton -- un écart ferait sauter les bras
au premier message de `depose()` (même piège déjà corrigé ailleurs).
`drop_z=PINCH_Z` : le podium de dépose fait la même hauteur (0.8m) que le
podium de prise, donc viser `PINCH_Z` retombe juste dessus.
`depivot_from_deg=PIVOT_ANGLE_DEG` : `depose()` doit savoir depuis quel
angle redresser le buste. Note l'étape "90" (une marche à vide vers un
2ᵉ point) qui existait avant a été **retirée** -- `depose()` se fait
maintenant SUR PLACE, la séquence se termine directement sur ce `stand()`.

### `main()` (lignes 347-371)

```python
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
```

Particularité : le node tourne dans un **thread séparé**
(`spin_thread.start()`) PENDANT que `run_sequence()` s'exécute dans le
thread PRINCIPAL -- nécessaire pour que `_publish_to_lcm` (déclenché par
un timer ROS2, donc seulement traité si le node "spin") continue de
tourner en arrière-plan pendant que `run_sequence()` attend chaque
`_send_goal` de façon bloquante. Une fois la séquence terminée, le node
continue de tourner (`spin_thread.join()` SANS timeout, bloque
indéfiniment) -- sinon le relais manette s'arrêterait net. `finally` :
`signal.signal(signal.SIGINT, signal.SIG_IGN)` ignore un second Ctrl+C
pendant le nettoyage (évite une interruption en plein `executor.shutdown()`),
`spin_thread.join(timeout=5.0)` cette fois AVEC un timeout (ne bloque pas
indéfiniment si le thread ne se termine pas proprement).

---

## 4. `stand.py` (82 lignes) -- passage en position debout stabilisée

### `__init__` (lignes 14-21)

```python
class StandActionServer(Node):
    def __init__(self):
        super().__init__("stand")
        self._lb_pub = self.create_publisher(Bool, field_topic("LB"), 10)
        self._a_pub = self.create_publisher(Bool, field_topic("A"), 10)
        self._server = ActionServer(
            self, Stand, "stand", self._execute, cancel_callback=self._on_cancel,
        )
```

Crée seulement 2 publishers (boutons LB et A) -- ce node ne fait
QU'appuyer sur cette combo, rien d'autre.

### `_on_cancel`/`_wait_cancelable` (lignes 23-35)

```python
def _on_cancel(self, goal_handle):
    return CancelResponse.ACCEPT

def _wait_cancelable(self, goal_handle, duration, step=0.1):
    elapsed = 0.0
    while elapsed < duration:
        if goal_handle.is_cancel_requested:
            return False
        time.sleep(step)
        elapsed += step
    return True
```

`_on_cancel` accepte TOUJOURS une demande d'annulation (pas de logique de
refus). `_wait_cancelable` : attente découpée en petits pas (0.1s) qui
vérifie `is_cancel_requested` à chaque pas -- pour qu'une annulation soit
prise en compte rapidement, plutôt qu'un `time.sleep(duration)` bloquant
qui l'ignorerait pendant toute sa durée.

### `_execute` (lignes 37-61)

```python
def _execute(self, goal_handle, combo_hold_seconds=0.5):
    result = Stand.Result()
    settle_seconds = goal_handle.request.settle_seconds

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

    if not self._wait_cancelable(goal_handle, settle_seconds):
        goal_handle.canceled()
        result.success = False
        return result

    self.get_logger().info("pd_stand stabilise.")
    goal_handle.succeed()
    result.success = True
    return result
```

Simule un appui manette : LB+A maintenu `combo_hold_seconds` (0.5s) puis
relâché -- déclenche la transition `pd_stand` côté `src_executor`. Si
annulé PENDANT le maintien du combo, relâche explicitement les deux
boutons avant de retourner (évite de les laisser "collés" à `True`).
Attend ensuite `settle_seconds` (paramètre du but, 10s par défaut) que la
stabilisation se termine, sans rien publier de plus pendant ce temps.

### `main()` (lignes 64-81)

Structure identique à tous les Action Servers de ce projet :
`MultiThreadedExecutor`, `executor.spin()` dans un `try`, nettoyage dans
`finally` (`signal.signal(signal.SIGINT, signal.SIG_IGN)`,
`executor.shutdown()`, `node.destroy_node()`, `rclpy.try_shutdown()`).
Cette structure est RÉPÉTÉE À L'IDENTIQUE dans `walk_to.py`, `lift.py`,
`pivot.py`, `depose.py` -- ne sera pas reproduite à nouveau ci-dessous
pour chacun (uniquement leurs différences, s'il y en a).

---

## 5. La marche : `walk_to.py` + `body_vel_bridge.py`

### `walk_to.py` -- même protocole que le robot réel

#### Constantes (lignes 45-49)

```python
WALK_MOTION_STATE = "rl_terrain"
RATE_HZ = 100.0
ZERO_PUBLISH_CYCLES = 10
MOTION_STATE_TIMEOUT = 3.0
MOTION_STATE_DETOUR = "pd_stand"
```

`RATE_HZ=100.0` : bien plus rapide que les rampes de bras (30-65Hz) --
une commande de vitesse a besoin d'un rafraîchissement plus fréquent
qu'une rampe de position.

#### `__init__` (lignes 52-74)

```python
def __init__(self):
    super().__init__("walk_to")
    from interface_protocol.msg import BodyVelCmd, MotionState, MotionStateRequest
    self._BodyVelCmd = BodyVelCmd
    self._MotionStateRequest = MotionStateRequest

    state_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                            durability=DurabilityPolicy.VOLATILE)
    request_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
    vel_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                          durability=DurabilityPolicy.VOLATILE)

    self._motion_state = {"current": "", "available": []}
    self.create_subscription(MotionState, "/motion/motion_state", self._on_motion_state, state_qos)
    self._motion_state_pub = self.create_publisher(MotionStateRequest, "/motion/set_motion_state", request_qos)
    self._vel_pub = self.create_publisher(BodyVelCmd, "/motion/body_vel_cmd", vel_qos)

    self._server = ActionServer(
        self, WalkTo, "walk_to", self._execute, cancel_callback=self._on_cancel,
    )
```

3 QoS distinctes : état (`BEST_EFFORT`, dépôt 1), demande de changement
d'état (`RELIABLE`, dépôt 1), commande de vitesse (`BEST_EFFORT`, dépôt
**10** -- plusieurs messages peuvent s'accumuler avant traitement, une
commande de vitesse en retard reste utile contrairement à une commande
périmée).

#### `_on_motion_state`/`_wait_for_motion_state`/`_switch_motion_state`/`_ensure_motion_state` (lignes 79-118)

```python
def _on_motion_state(self, msg) -> None:
    self._motion_state["current"] = msg.current_motion_task
    self._motion_state["available"] = list(msg.available_transition_motions)

def _wait_for_motion_state(self, timeout=1.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if self._motion_state["current"]:
            return
        time.sleep(0.05)

def _switch_motion_state(self, name: str, timeout: float) -> bool:
    self._wait_for_motion_state()
    if self._motion_state["current"] == name:
        return True
    if name not in self._motion_state["available"]:
        return False
    msg = self._MotionStateRequest()
    msg.target_motion_name = name
    self._motion_state_pub.publish(msg)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if self._motion_state["current"] == name:
            return True
        time.sleep(0.05)
    return False

def _ensure_motion_state(self, target: str, timeout: float = MOTION_STATE_TIMEOUT) -> bool:
    self._wait_for_motion_state()
    if self._motion_state["current"] == target:
        return True
    if (target not in self._motion_state["available"]
            and MOTION_STATE_DETOUR in self._motion_state["available"]):
        if not self._switch_motion_state(MOTION_STATE_DETOUR, timeout):
            return False
    return self._switch_motion_state(target, timeout)
```

**Réimplémentation LOCALE** de la même logique que
`motion_state.py::ensure_motion_state()` du robot réel (même mécanisme :
détour automatique si la cible n'est pas atteignable directement) --
**pas une copie paresseuse** : la version réelle appelle
`rclpy.spin_once(node, ...)` en interne, ce qui créerait une ressource
concurrente avec le `MultiThreadedExecutor` qui fait déjà tourner CE MÊME
node dans un thread séparé (`main()` plus bas). Cette version-ci utilise
`time.sleep()` pur (jamais `spin_once`) -- le callback `_on_motion_state`
est déjà appelé par l'executor dans son propre thread, pas besoin de le
déclencher manuellement ici.

#### `_execute` (lignes 120-169)

```python
def _execute(self, goal_handle):
    forward = goal_handle.request.forward
    turn = goal_handle.request.turn
    duration = goal_handle.request.duration
    result = WalkTo.Result()

    if not self._ensure_motion_state(WALK_MOTION_STATE):
        self.get_logger().error(f"walk_to : impossible de passer en {WALK_MOTION_STATE}.")
        goal_handle.abort()
        result.success = False
        return result

    self.get_logger().info(f"walk_to : forward={forward}m/s turn={turn}rad/s duration={duration}s")
    period = 1.0 / RATE_HZ
    elapsed = 0.0
    cancelled = False
    while elapsed < duration:
        if goal_handle.is_cancel_requested:
            cancelled = True
            break
        msg = self._BodyVelCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "body"
        msg.linear_velocity = [float(forward), 0.0]
        msg.yaw_velocity = float(turn)
        self._vel_pub.publish(msg)
        time.sleep(period)
        elapsed += period

    for _ in range(ZERO_PUBLISH_CYCLES):
        msg = self._BodyVelCmd()
        msg.linear_velocity = [0.0, 0.0]
        msg.yaw_velocity = 0.0
        self._vel_pub.publish(msg)
        time.sleep(period)

    if not self._ensure_motion_state("lower_body_balance"):
        self.get_logger().error("walk_to : impossible de repasser en lower_body_balance.")
        goal_handle.abort()
        result.success = False
        return result

    if cancelled:
        goal_handle.canceled()
        result.success = False
        return result

    goal_handle.succeed()
    result.success = True
    return result
```

Bascule d'abord en `rl_terrain` (état de marche). Boucle principale :
publie `BodyVelCmd` à 100Hz (`linear_velocity=[forward, 0.0]` -- pas de
déplacement latéral possible, `yaw_velocity=turn`) jusqu'à `duration`
écoulée OU annulation. `ZERO_PUBLISH_CYCLES=10` : publie 10 fois une
vitesse nulle APRÈS la marche -- s'assurer que la dernière commande reçue
par le contrôleur est bien "arrêt" (une seule publication pourrait se
perdre en QoS `BEST_EFFORT`). Repasse ensuite en `lower_body_balance`
inconditionnellement (même si annulé) -- pour ne jamais laisser le robot
en `rl_terrain` avec plus personne pour lui donner de commandes.

### `body_vel_bridge.py` -- SIM UNIQUEMENT, absent du lancement réel

#### Constantes (lignes 44-53)

```python
MPS_ANCHOR = 0.45
STICK_ANCHOR = 0.85
MPS_TO_STICK = STICK_ANCHOR / MPS_ANCHOR

WALK_STATE = "rl_terrain"
STOP_STATES = ("pd_stand", "lower_body_balance")
AVAILABLE_STATES = [WALK_STATE, *STOP_STATES]

ENTER_WALK_COMBO_HOLD_S = 0.5
ENTER_WALK_SETTLE_S = 1.5
```

`MPS_TO_STICK` : le facteur de conversion vitesse->stick, calibré sur UN
SEUL point (0.45 m/s <-> stick 0.85, la valeur réellement utilisée par
`chef_node.py`) -- documenté comme NON linéaire sur le reste de la plage.

#### `_clamp` (lignes 56-57)

```python
def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
```

Utilitaire simple : borne une valeur de stick entre -1 et 1 (les sticks
physiques ne dépassent jamais cette plage).

#### `__init__` (lignes 61-90)

```python
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
```

Ce node joue les DEUX rôles à la fois : il **publie** `/motion/motion_state`
(comme le ferait `src_executor` réel) ET **s'abonne** à
`/motion/set_motion_state`/`/motion/body_vel_cmd` (comme le ferait
`walk_to.py` sur le vrai robot) -- une simulation complète du protocole
réel, entièrement traduite en topics manette virtuelle de l'autre côté.
`create_timer(0.1, ...)` : republie l'état 10 fois par seconde en continu.

#### `_publish_state`/`_enter_walk` (lignes 92-106)

```python
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
```

`_publish_state` : annonce TOUJOURS les 3 états (`rl_terrain`,
`pd_stand`, `lower_body_balance`) comme disponibles -- pas de vraie
machine à états, juste ce dont `walk_to.py` a besoin. `_enter_walk` :
simule l'appui combo LB+B (0.5s maintenu, puis 1.5s de stabilisation) --
identique en esprit à l'ancien `_enter_walk()` que `walk_to.py` faisait
lui-même avant ce refactor.

#### `_on_state_request`/`_on_vel_cmd` (lignes 108-136)

```python
def _on_state_request(self, msg) -> None:
    target = msg.target_motion_name
    if target == self._current:
        return
    if target == WALK_STATE:
        self._enter_walk()
        self._last_stick = (0.0, 0.0)
    self._current = target
    self._publish_state()

def _on_vel_cmd(self, msg) -> None:
    if self._current != WALK_STATE:
        return
    forward_stick = _clamp(msg.linear_velocity[0] * MPS_TO_STICK)
    turn_stick = _clamp(msg.yaw_velocity * MPS_TO_STICK)
    if (forward_stick, turn_stick) == self._last_stick:
        return
    self._last_stick = (forward_stick, turn_stick)
    self._left_x_pub.publish(Float32(data=float(forward_stick)))
    self._right_y_pub.publish(Float32(data=float(-turn_stick)))
```

`_on_state_request` : si la cible demandée est `rl_terrain`, déclenche
`_enter_walk()` (combo physique) et réinitialise `_last_stick` ; pour
`pd_stand`/`lower_body_balance`, aucune action physique (la stabilisation
réelle est gérée par le node `stand` séparé). `_on_vel_cmd` : ignore tout
message si pas en marche (`self._current != WALK_STATE`). Convertit
vitesse -> stick, **republie SEULEMENT si la valeur a changé**
(`if (forward_stick, turn_stick) == self._last_stick: return`) -- fix
d'un bug déjà documenté : republier en continu (même valeur inchangée)
faisait marcher le robot ~22cm plus loin pour la même commande nominale
(réponse par à-coups de la marche en sim).

#### `main()` (lignes 139-152)

```python
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
```

Plus simple que les Action Servers : `rclpy.spin(node)` direct (pas de
`MultiThreadedExecutor`) -- ce node n'a pas d'Action à exécuter en
parallèle d'un timer, un seul thread suffit.

---

## 6. `lift.py` (447 lignes) -- prise du carton

### Imports et constantes (lignes 1-67)

```python
sys.path.insert(0, "/home/equansrobotic/stagiaire_1/tools/joint_angle_commander")
from lever import Lever

sys.path.insert(0, "/home/equansrobotic/stagiaire_1/tools/robot_arm_ik")
from lift_carton import (
    LEFT_CHAIN, RIGHT_CHAIN, HAND_OFFSET_LEFT, HAND_OFFSET_RIGHT, solve_ik,
    solve_arm_ik, ease, SimStateListener, world_to_robot_local,
    SQUEEZE_OFFSET_Y, mirror_left_to_right,
)

ELBOW_PITCH_CHAIN_INDEX = 3
LEFT_JOINT_INDICES = [13, 14, 15, 16, 17]
RIGHT_JOINT_INDICES = [18, 19, 20, 21, 22]

Q_LEFT_HOME = np.array([0.000879, 0.075284, -0.000233, -0.126397, -0.000033])
Q_RIGHT_HOME = np.array([0.000885, -0.075161, 0.000241, -0.126390, 0.000033])
WAYPOINT_Q_LEFT = np.radians([30.0, 5.0, 0.0, -110.0, 0.0])
WAYPOINT_Q_RIGHT = np.radians([30.0, -5.0, 0.0, -110.0, 0.0])

LEFT_HIP_PITCH_INDEX = 0
RIGHT_HIP_PITCH_INDEX = 6
LEFT_KNEE_PITCH_INDEX = 3
RIGHT_KNEE_PITCH_INDEX = 9
LEFT_ANKLE_PITCH_INDEX = 4
RIGHT_ANKLE_PITCH_INDEX = 10
WALK_STANCE_HIP_PITCH_L = np.radians(-6.9)
WALK_STANCE_HIP_PITCH_R = np.radians(-5.2)
WALK_STANCE_KNEE_L = np.radians(11.7)
WALK_STANCE_KNEE_R = np.radians(10.5)
WALK_STANCE_ANKLE_PITCH_L = np.radians(-4.8)
WALK_STANCE_ANKLE_PITCH_R = np.radians(-5.2)

LEVEE_ARM_STIFFNESS = 150.0
```

**IMPORTANT** : `Lever` est importé depuis `tools/joint_angle_commander/`
-- le MÊME fichier `lever.py` que celui utilisé par le robot réel
(chemin absolu, `sys.path.insert`), PAS une copie locale à la
simulation. Idem pour `LEFT_CHAIN`/`solve_arm_ik`/etc, importés depuis
`tools/robot_arm_ik/lift_carton.py` -- le fichier complet (828 lignes),
PAS la version trimée `package_sequence_bras_reel/lift_carton.py`
(103 lignes) que le robot réel utilise. `ELBOW_PITCH_CHAIN_INDEX=3` :
l'index verrouillé ici (coude), contrairement au robot réel qui verrouille
le poignet (index 4).

### `_quintic_ease`/`_rotate_xy` (lignes 70-84)

```python
def _quintic_ease(t):
    t = max(0.0, min(1.0, t))
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)

def _rotate_xy(point, yaw_offset):
    x, y, z = point
    c, s = np.cos(yaw_offset), np.sin(yaw_offset)
    return np.array([x * c - y * s, x * s + y * c, z])
```

Identiques aux versions du robot réel (`levee.py`) -- dupliquées ici
plutôt que partagées, les deux codebases n'important pas d'un fichier
commun pour ces utilitaires.

### `__init__`/`_on_cancel` (lignes 87-97)

```python
class LiftActionServer(Node):
    def __init__(self):
        super().__init__("lift")
        self._lever = None
        self._sim_state = None
        self._server = ActionServer(
            self, Lift, "lift", self._execute, cancel_callback=self._on_cancel,
        )

    def _on_cancel(self, goal_handle):
        return CancelResponse.ACCEPT
```

`_lever`/`_sim_state` initialisés à `None` -- créés paresseusement
(lazy) au premier besoin par `_ensure_lever`/`_ensure_sim_state`
ci-dessous, réutilisés ensuite pour tous les buts suivants de ce process.

### `_ensure_sim_state`/`_ensure_lever` (lignes 99-119)

```python
def _ensure_sim_state(self) -> SimStateListener:
    if self._sim_state is None:
        self._sim_state = SimStateListener()
        if not self._sim_state.wait_for_first_message(5.0):
            raise RuntimeError(
                "Aucun message sur le canal LCM 'sim_state' apres 5s -- "
                "run_mujoco.sh actif ?"
            )
    return self._sim_state

def _ensure_lever(self) -> Lever:
    if self._lever is None:
        self.get_logger().info(
            "Attente d'un abonne sur /motion/joint_override_command (run.sh actif ?)..."
        )
        self._lever = Lever(self, subscriber_timeout=10.0)
    return self._lever
```

Motif "lazy singleton" identique pour les deux ressources -- créées une
seule fois, réutilisées à chaque but suivant reçu par ce node (qui reste
en vie entre les 3 appels `lift()` de `chef_node.py`).

### `_bend_knees` (lignes 121-144)

```python
def _bend_knees(self, lever, scale, stiffness_scale, duration, rate_hz=30):
    for idx, kp, kd in [
        (LEFT_HIP_PITCH_INDEX, 200.0, 5.0), (RIGHT_HIP_PITCH_INDEX, 200.0, 5.0),
        (LEFT_KNEE_PITCH_INDEX, 450.0, 5.0), (RIGHT_KNEE_PITCH_INDEX, 450.0, 5.0),
        (LEFT_ANKLE_PITCH_INDEX, 400.0, 2.0), (RIGHT_ANKLE_PITCH_INDEX, 400.0, 2.0),
    ]:
        lever.set_gains(idx, kp * stiffness_scale, kd * stiffness_scale)
    n = max(1, int(duration * rate_hz))
    leg_indices = [LEFT_HIP_PITCH_INDEX, RIGHT_HIP_PITCH_INDEX, LEFT_KNEE_PITCH_INDEX,
                   RIGHT_KNEE_PITCH_INDEX, LEFT_ANKLE_PITCH_INDEX, RIGHT_ANKLE_PITCH_INDEX]
    for i in range(n + 1):
        a = _quintic_ease(i / n)
        lever.set_batch(leg_indices, [
            a * scale * WALK_STANCE_HIP_PITCH_L, a * scale * WALK_STANCE_HIP_PITCH_R,
            a * scale * WALK_STANCE_KNEE_L, a * scale * WALK_STANCE_KNEE_R,
            a * scale * WALK_STANCE_ANKLE_PITCH_L, a * scale * WALK_STANCE_ANKLE_PITCH_R,
        ])
        time.sleep(1.0 / rate_hz)
```

Même logique que la version robot réel (`levee.py::_bend_knees`), mais
**avec `set_batch`** (publication atomique des 6 joints en un message) au
lieu de 6 `lever[idx]=` séparés -- cette version côté simulation a déjà
le fix anti-vibration, contrairement à celle du robot réel qui a été
laissée telle quelle (moins critique, la simulation n'a pas de problème
de vibration matérielle réelle, mais le fix a été appliqué ici quand
même par cohérence de style).

### `_move_arms`/`_hold` (lignes 146-166)

```python
def _move_arms(self, lever, qL0, qL1, qR0, qR1, duration, rate_hz=30):
    n = max(1, int(duration * rate_hz))
    for i in range(n + 1):
        a = ease(i / n)
        qL = qL0 + a * (qL1 - qL0)
        qR = qR0 + a * (qR1 - qR0)
        lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))
        time.sleep(1.0 / rate_hz)
    return qL, qR

def _hold(self, lever, qL, qR, goal_handle, duration, rate_hz=10):
    elapsed = 0.0
    step = 1.0 / rate_hz
    while elapsed < duration:
        if goal_handle.is_cancel_requested:
            return False
        lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))
        time.sleep(step)
        elapsed += step
    return True
```

`_move_arms` : interpolation articulaire, identique en principe à
`move_arms` du robot réel. `_hold` **n'existe PAS côté robot réel** (qui
republie un maintien inline dans `run_lift_and_pivot` sans fonction
dédiée) -- ici, isolée en méthode car appelée à plusieurs endroits, ET
parce qu'elle doit vérifier `goal_handle.is_cancel_requested` (une
Action ROS2 peut être annulée en cours de maintien, un script CLI n'a
pas cette notion).

### `_straighten_knees`/`_release` (lignes 168-197)

```python
def _straighten_knees(self, lever, scale, stiffness_scale, duration, rate_hz=30):
    n = max(1, int(duration * rate_hz))
    leg_indices = [LEFT_HIP_PITCH_INDEX, RIGHT_HIP_PITCH_INDEX, LEFT_KNEE_PITCH_INDEX,
                   RIGHT_KNEE_PITCH_INDEX, LEFT_ANKLE_PITCH_INDEX, RIGHT_ANKLE_PITCH_INDEX]
    for i in range(n + 1):
        a = 1.0 - _quintic_ease(i / n)
        lever.set_batch(leg_indices, [
            a * scale * WALK_STANCE_HIP_PITCH_L, a * scale * WALK_STANCE_HIP_PITCH_R,
            a * scale * WALK_STANCE_KNEE_L, a * scale * WALK_STANCE_KNEE_R,
            a * scale * WALK_STANCE_ANKLE_PITCH_L, a * scale * WALK_STANCE_ANKLE_PITCH_R,
        ])
        time.sleep(1.0 / rate_hz)

def _release(self, lever, qL, qR, ramp_seconds, rate_hz=30):
    if ramp_seconds > 0:
        n = max(1, int(ramp_seconds * rate_hz))
        for i in range(n + 1):
            lever.set_weight(1.0 - i / n)
            lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))
            time.sleep(1.0 / rate_hz)
    lever.release()
```

Inverse exact de `_bend_knees` (courbe à l'envers). `_release` : rampe le
poids 1.0->0.0 en republiant la posture tenue à chaque pas, puis
`lever.release()` -- identique dans l'esprit au robot réel.

### `_execute` -- setup (lignes 199-270)

```python
def _execute(self, goal_handle):
    result = Lift.Result()
    g = goal_handle.request

    try:
        lever = self._ensure_lever()
    except RuntimeError as exc:
        self.get_logger().error(f"lift indisponible : {exc}")
        goal_handle.abort()
        result.success = False
        return result

    run_approche = g.only_phase in ("", "approche")
    run_serrage = g.only_phase in ("", "serrage")
    if run_approche:
        lever.forget()
    run_levee = g.only_phase in ("", "levee")

    if g.walk_stance:
        self.get_logger().info(...)
        self._bend_knees(lever, g.walk_stance_scale, g.walk_stance_stiffness_scale,
                          g.walk_stance_duration)
        self.get_logger().info("pause 2.0s (flexion genoux terminee)")
        time.sleep(2.0)

    q_waypoint_L = WAYPOINT_Q_LEFT.copy()
    q_waypoint_R = WAYPOINT_Q_RIGHT.copy()
```

`g.only_phase` : pour cette Action ROS2, `only_phase == ""` (chaîne vide)
correspond au défaut (pas `None` comme côté CLI robot réel -- les champs
`.action` ROS2 n'ont pas de vraie notion de "non fourni", donc le défaut
d'un `string` est `""`). `lever.forget()` appelé UNIQUEMENT si
`run_approche` -- oublie l'état local (`_touched`/gains) SANS publier,
pour repartir propre au début d'un nouvel appel complet, mais SANS effacer
ce qu'un appel précédent (`only_phase="approche"` puis `"serrage"`
séparément) aurait déjà tenu.

```python
    sim_state = self._ensure_sim_state()
    pose = sim_state.pose()
    if pose is None:
        self.get_logger().error("lift : aucune pose sim_state disponible -- abandon.")
        goal_handle.abort()
        result.success = False
        return result
    face_gauche_monde = np.array([g.face_gauche_x, g.face_gauche_y, g.face_gauche_z])
    face_droite_monde = np.array([g.face_droite_x, g.face_droite_y, g.face_droite_z])
    pinch_L = world_to_robot_local(face_gauche_monde, pose)
    pinch_R = world_to_robot_local(face_droite_monde, pose)
    AIM_LOWER_Z_OFFSET = -0.05
    pinch_L[2] += AIM_LOWER_Z_OFFSET
    pinch_R[2] += AIM_LOWER_Z_OFFSET
```

Reconvertit les coordonnées monde transmises par `chef_node.py`
(`g.face_gauche_x/y/z`) en repère local via `world_to_robot_local`, avec
la pose ACTUELLE du robot (jamais mise en cache). `AIM_LOWER_Z_OFFSET
=-0.05` : décale la visée 5cm plus bas que le centre exact des faces --
réglage empirique sur demande utilisateur ("les bras visent plus bas").

```python
    RETREAT_GAP_Y = 0.125
    aim_L = np.array([pinch_L[0], pinch_L[1] + RETREAT_GAP_Y, pinch_L[2]])
    aim_R = np.array([pinch_R[0], pinch_R[1] - RETREAT_GAP_Y, pinch_R[2]])
    q_aim_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, aim_L, q_waypoint_L,
                            lock_index=ELBOW_PITCH_CHAIN_INDEX,
                            lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX])
    q_aim_R = mirror_left_to_right(q_aim_L)

    squeeze_L = np.array([pinch_L[0], pinch_L[1] - SQUEEZE_OFFSET_Y, pinch_L[2]])
    squeeze_R = np.array([pinch_R[0], pinch_R[1] + SQUEEZE_OFFSET_Y, pinch_R[2]])
    q_squeeze_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_aim_L,
                                lock_index=ELBOW_PITCH_CHAIN_INDEX,
                                lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX],
                                null_space_pref=q_aim_L)
    q_squeeze_R = mirror_left_to_right(q_squeeze_L)
```

Calcule les 2 postures cibles (`q_aim_L`=visée en retrait de
`RETREAT_GAP_Y`=12.5cm, `q_squeeze_L`=serrage légèrement au-delà de la
surface) -- même logique que le robot réel, mais seedée/ancrée sur
`q_waypoint_L` (le point de passage) plutôt que `Q_LEFT_HOME`. `q_squeeze_L`
ancré sur `q_aim_L` (`null_space_pref=q_aim_L`) -- évite un changement de
branche épaule/coude entre visée et serrage.

### `_execute` -- phases approche/serrage (lignes 314-339)

```python
    if run_approche:
        self.get_logger().info(...)
        self._move_arms(lever, Q_LEFT_HOME, q_waypoint_L, Q_RIGHT_HOME, q_waypoint_R, g.waypoint_duration)
        self.get_logger().info(...)
        self._move_arms(lever, q_waypoint_L, q_aim_L, q_waypoint_R, q_aim_R, g.approach_duration)
        self.get_logger().info("pause 2.0s (approche terminee)")
        self._hold(lever, q_aim_L, q_aim_R, goal_handle, 2.0)

    if run_serrage:
        self.get_logger().info(...)
        self._move_arms(lever, q_aim_L, q_squeeze_L, q_aim_R, q_squeeze_R, g.squeeze_duration)
```

Approche : repos -> point de passage -> visée, puis maintien 2s (`_hold`,
annulable). Serrage : visée -> serrage, `move_arms` articulaire pur (pas
de `cartesian_ramp` ici, contrairement au robot réel qui utilise
`cartesian_ramp` pour l'approche -- **différence notable** : la
simulation utilise `move_arms` (articulaire) pour TOUTE l'approche/serrage,
le robot réel utilise `cartesian_ramp` pour l'approche (garantir une
ligne droite) et `move_arms` seulement pour le serrage. Cette différence
n'a pas encore été alignée entre les deux implémentations à ce jour.

### `_execute` -- fin de l'approche/serrage isolés (lignes 341-357)

```python
    if not run_levee:
        self.get_logger().info(
            f"only_phase={g.only_phase!r} termine -- bras tenus a leur derniere position, "
            "pas de levee/relachement dans cet appel."
        )
        goal_handle.succeed()
        result.success = True
        return result

    if g.only_phase == "levee":
        self.get_logger().info(
            f"only_phase=levee : publication directe de la position serree "
            f"(gauche={np.round(squeeze_L, 3)}, droite={np.round(squeeze_R, 3)}), "
            "suppose deja atteinte par un appel precedent."
        )
        lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(q_squeeze_L) + list(q_squeeze_R))
        self._hold(lever, q_squeeze_L, q_squeeze_R, goal_handle, 1.0)
```

Si `only_phase` n'est ni vide ni `"levee"`, s'arrête ici (bras tenus,
retourne succès). Si `only_phase == "levee"` EXACTEMENT (appel isolé
juste pour lever), republie directement la position serrée calculée
plus haut -- SANS avoir exécuté le `move_arms` du serrage dans CET appel
(il est supposé avoir été fait par un appel précédent) -- garantit que le
`Lever` de CE process a bien les bras "touchés" à la bonne valeur avant
de continuer.

### `_execute` -- levée (lignes 359-427)

```python
    self.get_logger().info(...)
    for idx in LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES:
        lever.set_gains(idx, stiffness=LEVEE_ARM_STIFFNESS)
    qL, qR = q_squeeze_L.copy(), q_squeeze_R.copy()
    n = max(1, int(g.lift_duration * 30))
    for i in range(n + 1):
        a = ease(i / n)
        z = squeeze_L[2] + a * (g.lift_z - squeeze_L[2])
        qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                           np.array([squeeze_L[0], squeeze_L[1], z]), qL,
                           lock_index=ELBOW_PITCH_CHAIN_INDEX,
                           lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX], iters=30,
                           null_space_pref=q_squeeze_L)
        qR = mirror_left_to_right(qL)
        lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))
        time.sleep(1.0 / 30)

    self.get_logger().info(f"maintien -- {g.hold_seconds:.1f}s")
    if not self._hold(lever, qL, qR, goal_handle, g.hold_seconds):
        self.get_logger().info("annule pendant le maintien -- relachement immediat.")
        self._release(lever, qL, qR, g.release_ramp_seconds)
        goal_handle.canceled()
        result.success = False
        return result
```

`LEVEE_ARM_STIFFNESS=150` (au lieu de 250 par défaut) : rigidité réduite
pendant le port de charge. Boucle de levée : résout l'IK À CHAQUE PAS
(pas un `cartesian_ramp` séparé -- codée inline ici), `iters=30` (plus que
les 10 du robot réel -- moins critique en simulation où la latence de
calcul n'a pas le même impact temps-réel). `qR` toujours dérivé par
miroir. Maintien via `_hold` -- si annulé PENDANT le maintien, relâche
IMMÉDIATEMENT (`_release` avec la posture courante) avant de retourner
l'échec.

```python
    if g.release_after:
        self.get_logger().info(...)
        n = max(1, int(g.lift_duration * 30))
        q_start_L, q_start_R = qL.copy(), qR.copy()
        for i in range(n + 1):
            a = ease(i / n)
            z = g.lift_z + a * (squeeze_L[2] - g.lift_z)
            qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                               np.array([squeeze_L[0], squeeze_L[1], z]), qL,
                               lock_index=ELBOW_PITCH_CHAIN_INDEX,
                               lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX], iters=30,
                               null_space_pref=q_start_L)
            qR = mirror_left_to_right(qL)
            lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))
            time.sleep(1.0 / 30)
        for idx in LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES:
            lever.set_gains(idx, stiffness=250.0)

        self.get_logger().info(...)
        qL, qR = self._move_arms(lever, qL, Q_LEFT_HOME, qR, Q_RIGHT_HOME, g.approach_duration)
        self.get_logger().info(...)
        self._straighten_knees(lever, g.walk_stance_scale, g.walk_stance_stiffness_scale,
                                g.walk_stance_duration)
        self.get_logger().info(f"relachement -- rampe {g.release_ramp_seconds:.1f}s")
        self._release(lever, qL, qR, g.release_ramp_seconds)
    else:
        self.get_logger().info(
            "release_after=false -- pas de relachement, bras/jambes restent tenus "
            "(un pivot est cense suivre)."
        )

    goal_handle.succeed()
    result.success = True
    return result
```

**Si `release_after=True`** (jamais le cas dans `chef_node.py::run_sequence`
actuel, qui passe toujours `False` pour `lift`) : redescend le carton
(Z inverse), remet les gains à 250 (rigidité normale), retire les bras
vers le repos, redresse les genoux, puis relâche complètement -- une
séquence "lift" AUTONOME qui pose le carton là où elle l'a pris. Sinon,
s'arrête bras/carton tenus, en attente de `pivot()`.

### `main()` (lignes 430-447) -- identique au motif de `stand.py`

---

## 7. `pivot.py` (341 lignes) -- rotation du buste, carton tenu

### Constantes propres (lignes 21-77)

```python
ELBOW_PITCH_CHAIN_INDEX = 3
LEFT_JOINT_INDICES = [13, 14, 15, 16, 17]
RIGHT_JOINT_INDICES = [18, 19, 20, 21, 22]
WAIST_JOINT_INDEX = 12
RETRACT_X = 0.35
RETRACT_DURATION = 1.0
EXTEND_DURATION = 1.0
RETREAT_GAP_Y = 0.125
WAIST_KP_HOLD, WAIST_KD_HOLD = 500.0, 10.0
WAIST_KP_RELEASE, WAIST_KD_RELEASE = 80.0, 2.0
LEG_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

Q_LEFT_HOME = np.array([0.000879, 0.075284, -0.000233, -0.126397, -0.000033])
Q_RIGHT_HOME = np.array([0.000885, -0.075161, 0.000241, -0.126390, 0.000033])
WAYPOINT_Q_LEFT = np.radians([30.0, 5.0, 0.0, -110.0, 0.0])
WAYPOINT_Q_RIGHT = np.radians([30.0, -5.0, 0.0, -110.0, 0.0])

LEFT_HIP_PITCH_INDEX = 0
RIGHT_HIP_PITCH_INDEX = 6
LEFT_KNEE_PITCH_INDEX = 3
RIGHT_KNEE_PITCH_INDEX = 9
LEFT_ANKLE_PITCH_INDEX = 4
RIGHT_ANKLE_PITCH_INDEX = 10
WALK_STANCE_HIP_PITCH_L = np.radians(-6.9)
WALK_STANCE_HIP_PITCH_R = np.radians(-5.2)
WALK_STANCE_KNEE_L = np.radians(11.7)
WALK_STANCE_KNEE_R = np.radians(10.5)
WALK_STANCE_ANKLE_PITCH_L = np.radians(-4.8)
WALK_STANCE_ANKLE_PITCH_R = np.radians(-5.2)

LEG_JOINTS = [
    (LEFT_HIP_PITCH_INDEX, WALK_STANCE_HIP_PITCH_L, 200.0, 5.0),
    (RIGHT_HIP_PITCH_INDEX, WALK_STANCE_HIP_PITCH_R, 200.0, 5.0),
    (LEFT_KNEE_PITCH_INDEX, WALK_STANCE_KNEE_L, 450.0, 5.0),
    (RIGHT_KNEE_PITCH_INDEX, WALK_STANCE_KNEE_R, 450.0, 5.0),
    (LEFT_ANKLE_PITCH_INDEX, WALK_STANCE_ANKLE_PITCH_L, 400.0, 2.0),
    (RIGHT_ANKLE_PITCH_INDEX, WALK_STANCE_ANKLE_PITCH_R, 400.0, 2.0),
]
```

`WAIST_KP_HOLD/KD_HOLD=500/10` : gains FORTS nécessaires pour une
rotation réelle du buste sous charge en simulation (un gain plus faible,
150/3 comme sur le robot réel, produisait une rotation quasi
imperceptible -- voir l'historique dans `chef_node.py`).
`WAIST_KP_RELEASE/KD_RELEASE=80/2` : gains assouplis pour le relâchement.
`LEG_JOINTS` : liste de tuples (index, angle de référence, kp de base, kd
de base) pour les 6 joints de jambe -- utilisée pour construire
`leg_targets` (mis à l'échelle par `walk_stance_scale`) plus bas.

### `__init__`/`_ensure_lever`/`_ensure_sim_state` (lignes 87-122)

Identiques en structure à `lift.py` (voir section 6) -- mêmes lazy
singletons `_lever`/`_sim_state`.

### `_publish_pose` (lignes 124-131) -- bras + jambes + buste en un message

```python
def _publish_pose(self, lever, qL, qR, waist_angle, leg_targets, stiffness_scale):
    for idx, _, kp, kd in leg_targets:
        lever.set_gains(idx, kp * stiffness_scale, kd * stiffness_scale)
    indices = LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES + [idx for idx, *_ in leg_targets] + [WAIST_JOINT_INDEX]
    angles = list(qL) + list(qR) + [target for _, target, *_ in leg_targets] + [waist_angle]
    lever.set_batch(indices, angles)
```

Règle d'abord les gains de CHAQUE joint de jambe (`leg_targets` contient
`(idx, target, kp, kd)` par joint, mis à l'échelle de `stiffness_scale` à
chaque appel -- redondant si la valeur ne change pas d'un appel à
l'autre, mais garde le code simple). Construit ensuite UNE liste
d'indices et UNE liste d'angles couvrant bras+jambes+buste, publiées en
un seul `set_batch`.

### `_execute` -- setup et recalcul de la posture tenue (lignes 133-202)

```python
def _execute(self, goal_handle):
    result = Pivot.Result()
    g = goal_handle.request

    try:
        lever = self._ensure_lever()
    except RuntimeError as exc:
        ...

    try:
        sim_state = self._ensure_sim_state()
    except RuntimeError as exc:
        ...
    pose = sim_state.pose()
    if pose is None:
        ...

    face_gauche_monde = np.array([g.face_gauche_x, g.face_gauche_y, g.face_gauche_z])
    face_droite_monde = np.array([g.face_droite_x, g.face_droite_y, g.face_droite_z])
    pinch_L = world_to_robot_local(face_gauche_monde, pose)
    pinch_R = world_to_robot_local(face_droite_monde, pose)
    pinch_L = np.array([pinch_L[0], pinch_L[1], g.lift_z])
    pinch_R = np.array([pinch_R[0], pinch_R[1], g.lift_z])
    squeeze_L = np.array([pinch_L[0], pinch_L[1] - SQUEEZE_OFFSET_Y, pinch_L[2]])
    squeeze_R = np.array([pinch_R[0], pinch_R[1] + SQUEEZE_OFFSET_Y, pinch_R[2]])

    aim_L = np.array([pinch_L[0], pinch_L[1] + RETREAT_GAP_Y, pinch_L[2]])
    q_aim_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, aim_L, WAYPOINT_Q_LEFT,
                            lock_index=ELBOW_PITCH_CHAIN_INDEX,
                            lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX])
    q_squeeze_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_aim_L,
                                lock_index=ELBOW_PITCH_CHAIN_INDEX,
                                lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX],
                                null_space_pref=q_aim_L)
    q_squeeze_R = mirror_left_to_right(q_squeeze_L)

    leg_targets = [
        (idx, target * g.walk_stance_scale, kp, kd) for idx, target, kp, kd in LEG_JOINTS
    ]
    lever.set_gains(WAIST_JOINT_INDEX, WAIST_KP_HOLD, WAIST_KD_HOLD)
```

**`pivot.py` est un process séparé** de `lift.py` (son propre `Lever`,
sa propre mémoire) -- il ne connaît PAS le `qL` interne que `lift.py`
vient de calculer, donc reconstruit lui-même la posture tenue à partir
des coordonnées monde transmises. Reconstruit `aim_L`/`q_aim_L` EXACTEMENT
comme `lift.py` (même cible, même seed `WAYPOINT_Q_LEFT`) pour reconverger
sur une posture **bit-à-bit identique** (fix vérifié numériquement le
2026-09-16, voir mémoire projet -- avant ce fix, jusqu'à 20° d'écart en
SHOULDER_YAW pour la même position de main, cause d'une rotation visible
de l'avant-bras au relais). `leg_targets` : construit à partir de
`LEG_JOINTS`, chaque cible mise à l'échelle de `g.walk_stance_scale`.

### `_execute` -- rapproché fusionné avec le début du pivot (lignes 209-224)

```python
    pinch_far_L = squeeze_L.copy()
    pinch_near_L = np.array([RETRACT_X, squeeze_L[1], squeeze_L[2]])
    self.get_logger().info(f"rapproche le carton avant pivot ({RETRACT_DURATION:.1f}s)")
    n_retract = max(1, int(RETRACT_DURATION * 30))
    retract_anchor_L = q_squeeze_L.copy()
    for i in range(n_retract + 1):
        a = ease(i / n_retract)
        target = pinch_far_L + a * (pinch_near_L - pinch_far_L)
        q_squeeze_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, target, q_squeeze_L,
                                    lock_index=ELBOW_PITCH_CHAIN_INDEX,
                                    lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX],
                                    null_space_pref=retract_anchor_L)
        q_squeeze_R = mirror_left_to_right(q_squeeze_L)
        self._publish_pose(lever, q_squeeze_L, q_squeeze_R, 0.0,
                            leg_targets, g.walk_stance_stiffness_scale)
        time.sleep(1.0 / 30)
```

Contrairement au robot réel (où retract/pivot/extend sont fusionnés en
UNE seule boucle continue), la simulation garde ici 3 étapes
**séquentielles** distinctes : le rapproché est fait AVANT que le buste
commence à tourner (`waist=0.0` fixe pendant toute cette boucle).
`retract_anchor_L` : capturée UNE FOIS avant la boucle, jamais réassignée
dedans (règle anti-dérive, déjà appliquée ici -- fix du 2026-09-16).

### `_execute` -- pivot pur (lignes 226-237)

```python
    angle_target = np.radians(g.angle_deg)
    self.get_logger().info(...)
    n = max(1, int(g.pivot_duration * 30))
    for i in range(n + 1):
        a = ease(i / n)
        self._publish_pose(lever, q_squeeze_L, q_squeeze_R, a * angle_target,
                            leg_targets, g.walk_stance_stiffness_scale)
        time.sleep(1.0 / 30)
```

Rotation pure du buste (0 -> `angle_target`), bras FIGÉS à `q_squeeze_L`/
`q_squeeze_R` (la valeur atteinte à la fin du rapproché) tout du long.

### `_execute` -- maintien annulable (lignes 239-250)

```python
    self.get_logger().info(f"maintien pivote -- {g.hold_seconds:.1f}s")
    elapsed = 0.0
    step = 0.1
    cancelled = False
    while elapsed < g.hold_seconds:
        if goal_handle.is_cancel_requested:
            cancelled = True
            break
        self._publish_pose(lever, q_squeeze_L, q_squeeze_R, angle_target,
                            leg_targets, g.walk_stance_stiffness_scale)
        time.sleep(step)
        elapsed += step
```

Contrairement à `lift.py::_hold` (méthode dédiée), cette boucle est codée
INLINE ici -- même logique (vérifie l'annulation à chaque pas de 0.1s).

### `_execute` -- extension fusionnée avec la fin (lignes 252-266)

```python
    if not cancelled:
        self.get_logger().info(f"tend les bras pour deposer ({EXTEND_DURATION:.1f}s)")
        n_extend = max(1, int(EXTEND_DURATION * 30))
        extend_anchor_L = q_squeeze_L.copy()
        for i in range(n_extend + 1):
            a = ease(i / n_extend)
            target = pinch_near_L + a * (pinch_far_L - pinch_near_L)
            q_squeeze_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, target, q_squeeze_L,
                                        lock_index=ELBOW_PITCH_CHAIN_INDEX,
                                        lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX],
                                        null_space_pref=extend_anchor_L)
            q_squeeze_R = mirror_left_to_right(q_squeeze_L)
            self._publish_pose(lever, q_squeeze_L, q_squeeze_R, angle_target,
                                leg_targets, g.walk_stance_stiffness_scale)
            time.sleep(1.0 / 30)
```

Seulement si PAS annulé pendant le maintien. Symétrique au rapproché :
`pinch_near_L -> pinch_far_L` (retour à la position de prise d'origine),
buste FIGÉ à `angle_target` pendant toute cette boucle, ancre
`extend_anchor_L` fixe capturée avant.

### `_execute` -- dépivotage conditionnel (lignes 268-289)

```python
    do_depivot = g.depivot_before_release or cancelled
    if do_depivot:
        self.get_logger().info(f"depivot -- {g.angle_deg:.0f}deg -> 0deg ({g.pivot_duration:.1f}s)")
        n = max(1, int(g.pivot_duration * 30))
        for i in range(n + 1):
            a = ease(i / n)
            kp = WAIST_KP_HOLD + a * (WAIST_KP_RELEASE - WAIST_KP_HOLD)
            kd = WAIST_KD_HOLD + a * (WAIST_KD_RELEASE - WAIST_KD_HOLD)
            lever.set_gains(WAIST_JOINT_INDEX, kp, kd)
            self._publish_pose(lever, q_squeeze_L, q_squeeze_R, (1.0 - a) * angle_target,
                                leg_targets, g.walk_stance_stiffness_scale)
            time.sleep(1.0 / 30)
        final_waist = 0.0
    else:
        lever.set_gains(WAIST_JOINT_INDEX, WAIST_KP_RELEASE, WAIST_KD_RELEASE)
        final_waist = angle_target
```

`do_depivot` : vrai si demandé explicitement OU si le but a été annulé
(dans les deux cas, il faut redresser avant de rendre la main). Si vrai :
rampe SIMULTANÉE de l'angle (rigide->0) ET des gains (`WAIST_KP_HOLD`
vers `WAIST_KP_RELEASE`) -- le buste devient déjà souple AVANT que les
bras ne lâchent le carton, évitant un choc double (perte de charge +
buste encore rigide). Sinon (cas normal de `chef_node.py`,
`depivot_before_release=False`) : assouplit quand même les gains
(prudence avant tout relâchement futur) mais garde l'angle tel quel --
c'est `depose()` qui redressera plus tard.

### `_execute` -- relâchement / libération partielle / maintien (lignes 291-321)

```python
    if cancelled or g.release_after:
        self.get_logger().info(f"relachement -- rampe {g.release_ramp_seconds:.1f}s")
        if g.release_ramp_seconds > 0:
            n = max(1, int(g.release_ramp_seconds * 30))
            for i in range(n + 1):
                lever.set_weight(1.0 - i / n)
                self._publish_pose(lever, q_squeeze_L, q_squeeze_R, final_waist,
                                    leg_targets, g.walk_stance_stiffness_scale)
                time.sleep(1.0 / 30)
        lever.release()
    elif g.free_legs_for_walk:
        self.get_logger().info(
            "liberation des jambes (marche) -- bras/buste restent tenus a poids plein"
        )
        self._publish_pose(lever, q_squeeze_L, q_squeeze_R, final_waist,
                            leg_targets, g.walk_stance_stiffness_scale)
        lever.untouch(LEG_INDICES)
    else:
        self.get_logger().info(
            "release_after=false -- pas de relachement, bras/jambes restent tenus "
            "(un autre node est cense suivre)."
        )

    if cancelled:
        goal_handle.canceled()
        result.success = False
        return result

    goal_handle.succeed()
    result.success = True
    return result
```

3 branches mutuellement exclusives : (1) `cancelled` OU `release_after`
demandé -> rampe complète puis `release()` total ; (2) sinon si
`free_legs_for_walk` -- publie une dernière fois, PUIS `lever.untouch(LEG_INDICES)`
(relâchement PARTIEL : rend les jambes au contrôleur natif pour marcher,
bras/buste restent tenus à poids plein) ; (3) sinon (cas de
`chef_node.py`, ni l'un ni l'autre) -- rien de plus, tout reste tenu pour
`depose()`.

### `main()` (lignes 324-341) -- identique au motif de `stand.py`

---

## 8. `depose.py` (442 lignes) -- repose le carton

### Constantes propres (lignes 23-53)

```python
LEFT_JOINT_INDICES = [13, 14, 15, 16, 17]
RIGHT_JOINT_INDICES = [18, 19, 20, 21, 22]
WAIST_JOINT_INDEX = 12
WAIST_HOLD_KP, WAIST_HOLD_KD = 80.0, 2.0
RETREAT_GAP_Y = 0.125

Q_LEFT_HOME = np.array([0.000879, 0.075284, -0.000233, -0.126397, -0.000033])
Q_RIGHT_HOME = np.array([0.000885, -0.075161, 0.000241, -0.126390, 0.000033])
WAYPOINT_Q_LEFT = np.radians([30.0, 5.0, 0.0, -110.0, 0.0])
WAYPOINT_Q_RIGHT = np.radians([30.0, -5.0, 0.0, -110.0, 0.0])

LEFT_HIP_PITCH_INDEX = 0
RIGHT_HIP_PITCH_INDEX = 6
LEFT_KNEE_PITCH_INDEX = 3
RIGHT_KNEE_PITCH_INDEX = 9
LEFT_ANKLE_PITCH_INDEX = 4
RIGHT_ANKLE_PITCH_INDEX = 10
WALK_STANCE_HIP_PITCH_L = np.radians(-6.9)
WALK_STANCE_HIP_PITCH_R = np.radians(-5.2)
WALK_STANCE_KNEE_L = np.radians(11.7)
WALK_STANCE_KNEE_R = np.radians(10.5)
WALK_STANCE_ANKLE_PITCH_L = np.radians(-4.8)
WALK_STANCE_ANKLE_PITCH_R = np.radians(-5.2)

ELBOW_PITCH_CHAIN_INDEX = 3
```

`WAIST_HOLD_KP/KD=80/2` -- déjà les valeurs "souples" (contrairement à
`pivot.py` qui commence FORT à 500/10 puis s'assouplit) : `depose.py`
prend le relais APRÈS que `pivot.py` a déjà commencé à s'assouplir, donc
n'a jamais besoin des gains forts.

### `_quintic_ease`/`_rotate_xy` (lignes 56-66) -- dupliquées, identiques aux autres fichiers

### `__init__`/`_ensure_sim_state`/`_ensure_lever` (lignes 72-103) -- structure identique à `pivot.py`

### `_bend_knees` (lignes 105-144) -- avec republication forcée bras+buste

```python
def _bend_knees(self, lever, scale, stiffness_scale, duration, qL_hold, qR_hold,
                 waist_hold, rate_hz=30):
    for idx, kp, kd in [
        (LEFT_HIP_PITCH_INDEX, 200.0, 5.0), (RIGHT_HIP_PITCH_INDEX, 200.0, 5.0),
        (LEFT_KNEE_PITCH_INDEX, 450.0, 5.0), (RIGHT_KNEE_PITCH_INDEX, 450.0, 5.0),
        (LEFT_ANKLE_PITCH_INDEX, 400.0, 2.0), (RIGHT_ANKLE_PITCH_INDEX, 400.0, 2.0),
    ]:
        lever.set_gains(idx, kp * stiffness_scale, kd * stiffness_scale)
    lever.set_gains(WAIST_JOINT_INDEX, WAIST_HOLD_KP, WAIST_HOLD_KD)
    n = max(1, int(duration * rate_hz))
    leg_indices = [LEFT_HIP_PITCH_INDEX, RIGHT_HIP_PITCH_INDEX, LEFT_KNEE_PITCH_INDEX,
                   RIGHT_KNEE_PITCH_INDEX, LEFT_ANKLE_PITCH_INDEX, RIGHT_ANKLE_PITCH_INDEX]
    for i in range(n + 1):
        a = _quintic_ease(i / n)
        leg_angles = [
            a * scale * WALK_STANCE_HIP_PITCH_L, a * scale * WALK_STANCE_HIP_PITCH_R,
            a * scale * WALK_STANCE_KNEE_L, a * scale * WALK_STANCE_KNEE_R,
            a * scale * WALK_STANCE_ANKLE_PITCH_L, a * scale * WALK_STANCE_ANKLE_PITCH_R,
        ]
        indices = leg_indices + LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES + [WAIST_JOINT_INDEX]
        angles = leg_angles + list(qL_hold) + list(qR_hold) + [float(waist_hold)]
        lever.set_batch(indices, angles)
        time.sleep(1.0 / rate_hz)
```

**Différence clé avec `lift.py::_bend_knees`** : celle-ci prend en plus
`qL_hold`/`qR_hold`/`waist_hold` et les republie À CHAQUE tick, pas
seulement les jambes. Raison : `/motion/joint_override_command` est un
**remplacement complet** à chaque message -- si le tout premier message
de ce node (nouveau `Lever`, tout juste créé) ne contenait QUE les
jambes, le carton serait instantanément lâché avant même que ce fichier
recalcule la position des bras plus bas.

### `_move_arms`/`_straighten_knees`/`_release` (lignes 146-184) -- identiques à `lift.py`

### `_execute` -- setup et recalcul (lignes 186-255)

```python
def _execute(self, goal_handle):
    result = Depose.Result()
    g = goal_handle.request

    try:
        lever = self._ensure_lever()
    except RuntimeError as exc:
        ...

    try:
        sim_state = self._ensure_sim_state()
    except RuntimeError as exc:
        ...
    pose = sim_state.pose()
    if pose is None:
        ...

    face_gauche_monde = np.array([g.face_gauche_x, g.face_gauche_y, g.face_gauche_z])
    face_droite_monde = np.array([g.face_droite_x, g.face_droite_y, g.face_droite_z])
    pinch_L = world_to_robot_local(face_gauche_monde, pose)
    pinch_R = world_to_robot_local(face_droite_monde, pose)
    pinch_L = np.array([pinch_L[0], pinch_L[1], g.hold_z])
    pinch_R = np.array([pinch_R[0], pinch_R[1], g.hold_z])
    hold_L = np.array([pinch_L[0], pinch_L[1] - SQUEEZE_OFFSET_Y, pinch_L[2]])
    hold_R = np.array([pinch_R[0], pinch_R[1] + SQUEEZE_OFFSET_Y, pinch_R[2]])

    aim_L = np.array([pinch_L[0], pinch_L[1] + RETREAT_GAP_Y, pinch_L[2]])
    q_aim_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, aim_L, WAYPOINT_Q_LEFT,
                            lock_index=ELBOW_PITCH_CHAIN_INDEX,
                            lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX])
    qL_hold = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, hold_L, q_aim_L,
                            lock_index=ELBOW_PITCH_CHAIN_INDEX,
                            lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX],
                            null_space_pref=q_aim_L)
    qR_hold = mirror_left_to_right(qL_hold)
    waist_hold = np.radians(g.depivot_from_deg)
```

Même reconstruction bit-à-bit identique que `pivot.py` (chaîne
`aim_L -> q_aim_L -> qL_hold`, fix du 2026-09-16). `g.hold_z` : la
hauteur RÉELLEMENT tenue par `pivot.py` (transmise par `chef_node.py`),
utilisée à la place de tout défaut périmé de `Depose.action`.
`waist_hold` : l'angle où le buste se trouve à l'entrée de ce node (le
buste n'a PAS été redressé par `pivot.py`, voir `depivot_before_release
=False`).

```python
    if g.walk_stance:
        self.get_logger().info(...)
        self._bend_knees(lever, g.walk_stance_scale, g.walk_stance_stiffness_scale,
                          g.walk_stance_duration, qL_hold, qR_hold, waist_hold)

    q_squeeze_L, q_squeeze_R = qL_hold, qR_hold
    lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES + [WAIST_JOINT_INDEX],
                     list(q_squeeze_L) + list(q_squeeze_R) + [float(waist_hold)])
```

Flexion des genoux (republiant bras+buste en continu, voir plus haut) --
nécessaire ici car `walk_to()` (une éventuelle marche ENTRE pivot et
dépose, pas utilisée dans la séquence actuelle mais le code le prévoit)
aurait rendu les jambes à `pd_stand` (droites) entre-temps. Republie une
dernière fois après (couvre le cas `walk_stance=False`, jamais publié
sinon).

### `_execute` -- tendre les bras, sans lâcher (lignes 274-283)

```python
    self.get_logger().info(...)
    q_tendu_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, hold_L, q_squeeze_L,
                              lock_index=ELBOW_PITCH_CHAIN_INDEX,
                              lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX])
    q_tendu_R = mirror_left_to_right(q_tendu_L)
    qL, qR = self._move_arms(lever, q_squeeze_L, q_tendu_L, q_squeeze_R, q_tendu_R,
                              g.tendre_duration)
```

Redresse les coudes SUR PLACE (même position de main `hold_L`, juste
changement de la préférence de posture puisque le seed change de
`q_squeeze_L` -- **note** : pas de `null_space_pref` explicite ici, donc
défaut = seed).

### `_execute` -- descente (lignes 285-301)

```python
    self.get_logger().info(
        f"depose -- Z {hold_L[2]:.3f} -> {g.drop_z:.3f} ({g.depose_duration:.1f}s) "
        "-- LE CARTON REDESCEND, TOUJOURS SERRE"
    )
    n = max(1, int(g.depose_duration * 30))
    q_drop_start_L = qL.copy()
    for i in range(n + 1):
        a = ease(i / n)
        z = hold_L[2] + a * (g.drop_z - hold_L[2])
        qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                           np.array([hold_L[0], hold_L[1], z]), qL,
                           lock_index=ELBOW_PITCH_CHAIN_INDEX,
                           lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX], iters=30,
                           null_space_pref=q_drop_start_L)
        qR = mirror_left_to_right(qL)
        lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))
        time.sleep(1.0 / 30)
```

Rampe verticale pure (X/Y fixes), le carton TOUJOURS serré pendant toute
la descente -- ancre fixe `q_drop_start_L` capturée avant la boucle.

### `_execute` -- ouverture de la prise (lignes 303-325)

```python
    self.get_logger().info(...)
    release_L = np.array([hold_L[0], hold_L[1] + SQUEEZE_OFFSET_Y, g.drop_z])
    q_release_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, release_L, qL,
                                lock_index=ELBOW_PITCH_CHAIN_INDEX,
                                lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX])
    q_release_R = mirror_left_to_right(q_release_L)
    qL, qR = self._move_arms(lever, qL, q_release_L, qR, q_release_R, g.tendre_duration)
```

`release_L` utilise `g.drop_z` (la hauteur RÉELLEMENT atteinte à la fin
de la descente précédente), PAS `hold_L[2]` (qui vaut toujours la
hauteur de DÉPART, jamais réassigné par la boucle de descente -- piège
déjà évité en commentaire dans le code source). Ouvre exactement de
`SQUEEZE_OFFSET_Y` (annule le décalage de serrage) -- le carton n'est
PAS encore relâché à cette étape (voir "dégagement" plus bas).

### `_execute` -- écartement (lignes 327-344)

```python
    ECARTEMENT_GAP_Y = 0.08
    ECARTEMENT_DURATION = 1.0
    self.get_logger().info(...)
    ecart_L = np.array([hold_L[0], hold_L[1] + ECARTEMENT_GAP_Y, g.drop_z])
    q_ecart_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, ecart_L, qL,
                              lock_index=ELBOW_PITCH_CHAIN_INDEX,
                              lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX],
                              null_space_pref=qL)
    q_ecart_R = mirror_left_to_right(q_ecart_L)
    qL, qR = self._move_arms(lever, qL, q_ecart_L, qR, q_ecart_R, ECARTEMENT_DURATION)
```

`ECARTEMENT_GAP_Y`/`ECARTEMENT_DURATION` définies LOCALEMENT (pas des
constantes de module, ni des champs `.action`) -- écarte encore de 8cm
pour dégager de la surface du carton avant de retirer la main.

### `_execute` -- translation arrière (lignes 346-371)

```python
    RETREAT_BACK_X = 0.05
    RETREAT_BACK_DURATION = 1.0
    self.get_logger().info(...)
    n_retreat = max(1, int(RETREAT_BACK_DURATION * 30))
    q_retreat_start_L = qL.copy()
    for i in range(n_retreat + 1):
        a = ease(i / n_retreat)
        x = ecart_L[0] + a * (RETREAT_BACK_X - ecart_L[0])
        qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                           np.array([x, ecart_L[1], ecart_L[2]]), qL,
                           lock_index=ELBOW_PITCH_CHAIN_INDEX,
                           lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX], iters=30,
                           null_space_pref=q_retreat_start_L)
        qR = mirror_left_to_right(qL)
        lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))
        time.sleep(1.0 / 30)
```

Rampe cartésienne EN X SEUL (Y/Z inchangés, valeurs de `ecart_L`
réutilisées) -- ramène la main près du corps avant le saut articulaire
vers le point de passage.

### `_execute` -- dégagement, relâchement du carton (lignes 373-379)

```python
    self.get_logger().info(
        f"degagement -- coudes vers l'arriere ({g.degagement_waypoint_duration:.1f}s) "
        "-- LE CARTON EST RELACHE ICI (l'ecartement des mains vers ce point de passage "
        "suffit, plus besoin d'un desserrage separe)"
    )
    qL, qR = self._move_arms(lever, qL, WAYPOINT_Q_LEFT, qR, WAYPOINT_Q_RIGHT,
                              g.degagement_waypoint_duration)
```

**LE CARTON EST RELÂCHÉ ICI** (au sens conceptuel -- l'écartement
accumulé depuis "ouverture de la prise" + "écartement" suffit, pas de
désserrage séparé comme sur le robot réel). `move_arms` articulaire
(pas cartésien) vers `WAYPOINT_Q_LEFT`/`WAYPOINT_Q_RIGHT`.

### `_execute` -- dépivot AVANT le retour des bras (lignes 381-408)

```python
    if g.depivot_from_deg != 0.0:
        self.get_logger().info(
            f"depivot -- {g.depivot_from_deg:.0f}deg -> 0deg ({g.depivot_duration:.1f}s), "
            "carton deja lache, buste seul (avant retour bras home)"
        )
        n_depivot = max(1, int(g.depivot_duration * 30))
        for i in range(n_depivot + 1):
            a = ease(i / n_depivot)
            waist = (1.0 - a) * waist_hold
            lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES + [WAIST_JOINT_INDEX],
                             list(qL) + list(qR) + [float(waist)])
            time.sleep(1.0 / 30)

    self.get_logger().info(f"degagement -- retour bras home ({g.degagement_duration:.1f}s)")
    qL, qR = self._move_arms(lever, qL, Q_LEFT_HOME, qR, Q_RIGHT_HOME, g.degagement_duration)
```

**Ordre fixé le 2026-09-16/17** : le dépivot (buste vers 0°) se fait
AVANT le retour des bras -- l'inverse (retour des bras d'abord) créait un
déséquilibre statique mesuré en télémétrie (~14° de torsion du bassin en
moins d'une seconde), car les bras arrivaient à leur posture de repos
(masse contre le corps) pendant que le buste était encore tourné, décalant
la masse par rapport aux pieds. Le carton est déjà lâché à ce stade
(voir "dégagement" ci-dessus), donc pas de risque de "chute avec charge"
comme celui déjà géré dans `pivot.py`.

### `_execute` -- fin (lignes 410-422)

```python
    if g.walk_stance:
        self.get_logger().info(
            f"retrait jambes -- flechies -> droites ({g.walk_stance_duration:.1f}s)"
        )
        self._straighten_knees(lever, g.walk_stance_scale, g.walk_stance_stiffness_scale,
                                g.walk_stance_duration)

    self.get_logger().info(f"relachement final -- rampe {g.release_ramp_seconds:.1f}s")
    self._release(lever, qL, qR, g.release_ramp_seconds)

    goal_handle.succeed()
    result.success = True
    return result
```

Redressement des genoux (si `walk_stance`), puis relâchement final
identique au motif déjà vu (rampe de poids + `release()`).

### `main()` (lignes 425-442) -- identique au motif de `stand.py`

---

## Récapitulatif : différences avec le robot réel

| | Simulation | Robot réel |
|---|---|---|
| Architecture | 5 Action Servers séparés, 1 orchestrateur (`chef_node.py`) | 1 seul process/`Lever` |
| Position du carton | Lue dans le XML de la scène (`carton_face_centers`) | Mesurée par vision (ArUco), fournie en argument CLI |
| Position du robot | Vérité terrain LCM (`SimStateListener`) | Proprioception réelle |
| Verrou IK | Coude (`ELBOW_PITCH_CHAIN_INDEX=3`) | Poignet (`WRIST_CHAIN_INDEX=4`) |
| Approche | `move_arms` articulaire de bout en bout | `cartesian_ramp` (ligne droite garantie) |
| Retract/pivot/extend | 3 étapes séquentielles (`pivot.py`) | Fusionnées en une boucle continue (`levee_pivot.py`) |
| Marche | Protocole réel + pont de traduction (`body_vel_bridge.py`) | Protocole réel direct (pas de pont) |
| Confirmation manuelle | Aucune (séquence automatique) | Entre chaque étape par défaut (`_checkpoint`) |
| `lift_carton.py` | Fichier complet (828 lignes), avec vision/LCM | Copie trimée (104 lignes), IK seulement |
