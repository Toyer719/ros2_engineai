# Le code de la simulation, bloc par bloc

Ce document explique le code ROS2 de la simulation (`tools/virtual_gamepad/
ros_ws/src/virtual_gamepad_ros/virtual_gamepad_ros/`) : `chef_node.py` et
les 5 Action Servers qu'il orchestre, plus les fichiers de support. Pour
l'équivalent robot réel, voir [`CODE_ROBOT_REEL.md`](CODE_ROBOT_REEL.md) --
les deux partagent la même cinématique inverse (`solve_arm_ik`,
`forward_kinematics`...), déjà expliquée là-bas ; ce document ne la
réexplique pas, il couvre ce qui est **spécifique à la simulation**.

Ordre de lecture : les fondations d'abord (topics, géométrie, IK partagée),
puis chaque Action Server dans l'ordre où `chef_node.py` les appelle
(`stand` → `walk_to` → `lift` → `pivot` → `depose`), l'orchestrateur en
dernier -- comme pour le document robot réel, pas l'ordre alphabétique.

---

## 1. `field_topics.py` -- la table de correspondance manette virtuelle

```python
BUTTON_INDEX = {"LB": 0, "RB": 1, "A": 2, "B": 3, ...}
ANALOG_INDEX = {"LEFT_STICK_X": 2, "LEFT_STICK_Y": 3, ...}

def field_topic(name: str) -> str:
    return f"/virtual_gamepad/cmd/{name.lower()}"
```

Chaque bouton/stick de la manette virtuelle correspond à un topic ROS2
séparé (`/virtual_gamepad/cmd/lb`, `/virtual_gamepad/cmd/left_stick_x`,
etc.) -- un `Bool` par bouton, un `Float32` par stick. `chef_node.py`
s'abonne à TOUS ces topics pour reconstituer l'état complet de la manette
et le relayer à MuJoCo par LCM ; `stand.py` et `body_vel_bridge.py`
publient sur certains de ces mêmes topics pour "appuyer sur les boutons"
programmatiquement (LB+A pour `pd_stand`, LB+B pour la marche).

---

## 2. `lift_carton.py` -- ce qui est spécifique à la simulation

Le fichier complet (`tools/robot_arm_ik/lift_carton.py`, ~800 lignes)
contient la même cinématique inverse que la version robot réel (chaînes,
`forward_kinematics`, `solve_arm_ik`, `mirror_left_to_right`, `ease` --
voir [`CODE_ROBOT_REEL.md`](CODE_ROBOT_REEL.md#1-lift_cartonpy----la-géométrie-et-la-cinématique-inverse)),
plus 3 éléments **uniquement utiles en simulation** :

### `carton_face_centers()` -- lire la position du carton dans la scène MuJoCo

```python
def carton_face_centers(scene_xml=LIVE_SCENE_XML):
    src = open(scene_xml).read()
    m = re.search(r'<body name="carton" pos="([^"]+)"', src)
    cx, cy, cz = (float(v) for v in m.group(1).split())
    m = re.search(r'<geom name="carton_box"[^>]*\bsize="([^"]+)"', src)
    sx, sy, sz = (float(v) for v in m.group(1).split())
    face_gauche = np.array([cx, cy + sy, cz])
    face_droite = np.array([cx, cy - sy, cz])
    return face_gauche, face_droite
```

Lit **directement le fichier XML** de la scène MuJoCo en cours
d'exécution (`LIVE_SCENE_XML`) avec une regex, plutôt que de coder en dur
la position du carton -- pour ne jamais avoir une valeur périmée si la
scène change. `size` dans MuJoCo est une **demi-dimension** (d'où
`cy + sy`/`cy - sy` pour les deux faces). Sur le robot réel, l'équivalent
serait la position mesurée par vision (ArUco) -- ce fichier n'a pas
d'accès à un XML de scène puisqu'il n'y a pas de scène.

### `SimStateListener` -- lire la position réelle du robot dans MuJoCo

```python
class SimStateListener:
    def __init__(self, lcm_url=SIM_LCM_URL):
        ...
        self._lc.subscribe(SIM_STATE_CHANNEL, self._on_message)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def pose(self):
        x, y, z = state.base_link_position
        w, qx, qy, qz = state.base_link_quaternion
        return x, y, z, _yaw_from_quaternion(w, qx, qy, qz)
```

S'abonne au canal LCM `sim_state` -- la **vérité terrain** publiée par
MuJoCo (position/orientation réelles du robot, pas une estimation). Tourne
dans un thread séparé (`_spin`) pour ne jamais bloquer le node ROS2
principal en attendant un message LCM. `pose()` renvoie `(x, y, z, yaw)`
du bassin en repère MONDE. Le robot réel n'a pas d'équivalent direct --
il utiliserait sa propre odométrie/proprioception au lieu d'une vérité
terrain de simulateur.

### `world_to_robot_local()` -- convertir une position monde en cible IK

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

Combine les deux fonctions précédentes : `carton_face_centers()` donne où
est le carton dans le MONDE, `SimStateListener.pose()` donne où est le
robot dans le MONDE -- cette fonction fait la **soustraction + rotation**
(seulement autour de Z, le robot est supposé rester vertical) pour obtenir
la position du carton **dans le repère du bassin du robot**, directement
utilisable comme cible pour `solve_arm_ik`. C'est l'opération inverse de
`_rotate_xy` (utilisée dans `lift.py`/`pivot.py`/`depose.py` pour l'effet
symétrique). Appelée à **chaque** exécution de `lift`/`pivot`/`depose`
(jamais une position mise en cache) -- le robot peut légèrement dériver
entre deux étapes, une pose figée donnerait une cible obsolète.

---

## 3. `chef_node.py` -- l'orchestrateur

Le seul fichier qui décide de l'**ordre** des actions et de leurs
paramètres. Contrairement au robot réel, il ne bouge jamais rien
lui-même -- il envoie des buts (`send_goal`) à 5 Action Servers
indépendants et attend leur résultat.

### Deux rôles en un seul node (lignes 95-129)

```python
class ChefNode(Node):
    def __init__(self, rate_hz: float = 20.0):
        super().__init__("chef")
        self._lcm = lcm.LCM(LCM_URL)
        self._state = GamepadKeys()
        ...
        for name, idx in BUTTON_INDEX.items():
            self.create_subscription(Bool, field_topic(name), self._button_cb(idx), 10)
        for name, idx in ANALOG_INDEX.items():
            self.create_subscription(Float32, field_topic(name), self._analog_cb(idx), 10)
        ...
        self.create_timer(self._period, self._publish_to_lcm)
```

**Rôle 1 -- relais manette virtuelle vers LCM** : s'abonne à tous les
topics `/virtual_gamepad/cmd/*` (boutons + sticks), les accumule dans
`self._state` (un message `GamepadKeys`), et republie cet état complet
vers MuJoCo par LCM 20 fois par seconde (`create_timer`). C'est ce
mécanisme que `stand.py` et `body_vel_bridge.py` utilisent pour "appuyer"
virtuellement sur les boutons de la manette.

**Rôle 2 -- orchestrateur de séquence** : les `ActionClient` (lignes
107-111) et la méthode `run_sequence()` (plus bas) qui enchaîne
stand/walk_to/lift/pivot/depose.

### `_send_goal` (lignes 132-159) -- envoyer un but et attendre le résultat

```python
def _send_goal(self, client, name, goal):
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
    ...
    return outcome["result"].success
```

Le patron classique pour transformer une **Action ROS2 asynchrone** (qui
fonctionne par callbacks) en un appel **synchrone/bloquant** : un
`threading.Event` qui se déclenche quand le résultat arrive, et
`goal_done.wait(...)` qui bloque `run_sequence()` jusque-là. Sans ça,
`run_sequence()` devrait elle-même être écrite en callbacks imbriqués,
beaucoup moins lisible pour une séquence linéaire.

**Point d'attention (déjà rencontré dans ce projet) :** `success=True` ne
garantit PAS que le mouvement a réellement eu l'effet physique attendu --
juste que l'Action Server a terminé sans erreur logicielle. Pour vérifier
qu'un mouvement a vraiment eu lieu, il faut regarder la télémétrie
`sim_state` (voir `tools/vision/log_sim_pose.py`), pas seulement ce
booléen.

### `PROVEN_PINCH_X`/`PINCH_Y`/`SQUEEZE_Y`/`PINCH_Z` (lignes 27-79)

Constantes géométriques, avec un très long historique en commentaire
(plusieurs itérations pour trouver une distance/largeur d'approche qui
évite à la fois une posture de bras "tordue" et une violation des limites
articulaires réelles du robot -- voir le commentaire du fichier pour le
détail complet du raisonnement). Contrairement au robot réel où
`PINCH_X`/`PINCH_Y` restent des paramètres directement utilisés, en
simulation ces valeurs ne servent presque plus depuis le passage à la
visée par `carton_face_centers()` (2026-09-11) -- `lift.py` vise
directement le centre réel des faces, pas une position calculée depuis
ces constantes.

### `run_sequence()` (lignes 187-344) -- la chorégraphie complète

```python
self._publish_step(0)
self.stand()

self._publish_step(10)
self.walk_to(forward=WALK_FORWARD_MPS, turn=TURN_CORRECTION, duration=WALK_DURATION)
time.sleep(2.0)
self.stand(settle_seconds=3.0)

self._publish_step(20)
face_gauche_monde, face_droite_monde = carton_face_centers()
face_kwargs = dict(face_gauche_x=..., face_gauche_y=..., ...)

self._publish_step(30)
self.lift(..., only_phase="approche", release_after=False, **face_kwargs)

self._publish_step(40)
self.lift(..., only_phase="serrage", release_after=False, **face_kwargs)

self._publish_step(50)
self.lift(..., only_phase="levee", release_after=False, **face_kwargs)

self._publish_step(60)
self.pivot(..., release_after=False, free_legs_for_walk=False,
           depivot_before_release=False, **face_kwargs)

self._publish_step(80)
self.depose(..., depivot_from_deg=PIVOT_ANGLE_DEG, **face_kwargs)

self.stand(settle_seconds=3.0)
```

Points clés :

- **`carton_face_centers()` appelé UNE SEULE FOIS** (étape 20), puis le
  résultat (`face_kwargs`) est transmis à `lift`/`pivot`/`depose` via les
  champs des messages `.action` -- chacun des 3 nodes recalcule ensuite
  sa propre cible IK à partir de CES MÊMES coordonnées monde (pas une
  nouvelle lecture du XML), pour que les 3 visent physiquement le même
  point même si le carton "bougeait" entre deux appels (il ne bouge pas
  en pratique, mais ça prépare le terrain pour la vision réelle où la
  mesure ne serait prise qu'une fois).
- **`lift()` appelé 3 fois** avec `only_phase="approche"` puis
  `"serrage"` puis `"levee"` -- pas une seule fois avec la séquence
  complète. Chaque appel est une Action ROS2 séparée (voir
  `_publish_step` entre chaque), ce qui permet d'observer/déboguer étape
  par étape via `ros2 topic echo /chef/current_step`.
- **`release_after=False`** sur `lift` et `pivot` : le carton reste tenu
  d'un appel à l'autre -- c'est `depose()` qui fait le relâchement final.
- **`free_legs_for_walk=False`, pas de marche entre pivot et depose** :
  historique de plusieurs chutes documentées quand la séquence combinait
  "objet tenu" + "marche" (voir les longs commentaires du fichier) --
  `depose()` se fait maintenant SUR PLACE, juste après le pivot.
- **`depivot_before_release=False`** : le buste reste tourné quand
  `pivot()` rend la main à `depose()`, qui gère lui-même le dépivotage
  (voir section 8) -- ordre repris du robot réel après plusieurs
  itérations en direct.

### `main()` (lignes 347-371)

```python
executor = MultiThreadedExecutor()
executor.add_node(node)
spin_thread = threading.Thread(target=executor.spin, daemon=True)
spin_thread.start()

node.run_sequence()
node.get_logger().info("... heartbeat LCM maintenu ...")
spin_thread.join()
```

Particularité par rapport aux Action Servers (`lift.py`, etc.) : le node
tourne dans un **thread séparé** (`spin_thread`) PENDANT que
`run_sequence()` s'exécute dans le thread principal -- nécessaire pour
que `_publish_to_lcm` (le relais manette, appelé par un timer ROS2)
continue de fonctionner en arrière-plan pendant que `run_sequence()`
attend chaque `_send_goal` de façon bloquante. Une fois la séquence
terminée, le node continue de tourner ("heartbeat LCM maintenu") au lieu
de s'arrêter -- sinon le relais manette (et donc tout contrôle manuel
restant) s'arrêterait net.

---

## 4. `stand.py` -- passage en position debout stabilisée

Le plus court des 5 Action Servers. Simule un appui manette : LB+A
(maintenu 0.5s) déclenche la transition `pd_stand` côté `src_executor`,
puis attend `settle_seconds` que ça se stabilise.

```python
def _execute(self, goal_handle, combo_hold_seconds=0.5):
    self._lb_pub.publish(Bool(data=True))
    self._a_pub.publish(Bool(data=True))
    if not self._wait_cancelable(goal_handle, combo_hold_seconds):
        ...
    self._lb_pub.publish(Bool(data=False))
    self._a_pub.publish(Bool(data=False))
    if not self._wait_cancelable(goal_handle, settle_seconds):
        ...
```

`_wait_cancelable` (lignes 26-35) : une attente découpée en petits pas
(0.1s) qui vérifie `goal_handle.is_cancel_requested` à chaque pas -- pour
qu'une annulation de l'Action (`ros2 action cancel`) soit prise en compte
rapidement, plutôt qu'un seul `time.sleep(settle_seconds)` bloquant qui
ignorerait toute demande d'annulation pendant toute sa durée.

---

## 5. La marche : `walk_to.py` + `body_vel_bridge.py`

La marche est le SEUL mécanisme qui utilise un protocole "comme le robot
réel" (`/motion/body_vel_cmd`) même en simulation -- mais MuJoCo ne
comprend nativement que l'émulation manette LCM, d'où un node-pont séparé.

### `walk_to.py` -- même code que le robot réel

```python
msg = self._BodyVelCmd()
msg.linear_velocity = [float(forward), 0.0]
msg.yaw_velocity = float(turn)
self._vel_pub.publish(msg)
```

Ce fichier publie sur `/motion/body_vel_cmd` -- **exactement le même
protocole que sur le robot réel** (`tools/joint_angle_commander/marche.py`).
Il n'a besoin de savoir ni où il tourne, ni s'il communique avec
`src_executor` réel ou simulé. Deux détails d'implémentation :

- **`_switch_motion_state` réimplémenté localement** (lignes 90-118),
  au lieu de réutiliser `motion_state.py::ensure_motion_state()` du robot
  réel : cette fonction-là appelle `rclpy.spin_once()` en interne, ce qui
  créerait une ressource concurrente avec le `MultiThreadedExecutor` qui
  fait déjà tourner ce même node dans un thread séparé (voir la docstring
  du fichier pour le détail de la race condition évitée).
- **`ZERO_PUBLISH_CYCLES`** (ligne 149-154) : publie 10 fois une vitesse
  nulle après la marche, avant de repasser en `lower_body_balance` --
  s'assurer que la dernière commande reçue par le contrôleur est bien
  "arrêt", pas la dernière vitesse non-nulle qui pourrait ne jamais avoir
  été traitée (perte de message possible en QoS `BEST_EFFORT`).

### `body_vel_bridge.py` -- SIM UNIQUEMENT, absent du lancement réel

```python
MPS_ANCHOR = 0.45
STICK_ANCHOR = 0.85
MPS_TO_STICK = STICK_ANCHOR / MPS_ANCHOR

def _on_vel_cmd(self, msg) -> None:
    if self._current != WALK_STATE:
        return
    forward_stick = _clamp(msg.linear_velocity[0] * MPS_TO_STICK)
    ...
    if (forward_stick, turn_stick) == self._last_stick:
        return
    self._last_stick = (forward_stick, turn_stick)
    self._left_x_pub.publish(Float32(data=float(forward_stick)))
```

Ce node **traduit** `/motion/body_vel_cmd` (protocole "réel") vers les
topics `/virtual_gamepad/cmd/*` que MuJoCo comprend déjà -- il n'existe
QUE dans le lancement simulation (`virtual_gamepad.launch.py`), jamais
sur le robot réel qui a son propre node natif pour ce protocole.

Deux points notables :

- **Calibration NON linéaire** (voir docstring complète) : la conversion
  vitesse -> position de stick n'est fidèle qu'en **un seul point**
  (0.45 m/s ↔ stick 0.85, la valeur réellement utilisée par
  `chef_node.py`) -- mesuré par télémétrie que la réponse de MuJoCo à la
  marche n'est PAS un contrôleur de vitesse linéaire sur toute la plage.
- **Publication "one-shot"** (lignes 132-134) : republie le stick
  SEULEMENT quand sa valeur change, pas en continu à chaque message
  reçu -- un bug déjà rencontré ("le robot marche 22cm plus loin qu'avant
  pour la même commande") venait d'une republication continue qui faisait
  parcourir une distance différente au robot pour un stick nominal
  identique.

---

## 6. `lift.py` -- prise du carton (Action Server)

Le plus long des 3 nodes bras (447 lignes). Trois phases
(`approche`/`serrage`/`levee`), chacune déclenchable séparément via
`only_phase` (voir l'appel triple dans `chef_node.py::run_sequence`).

### Setup : où vise chaque bras (lignes 251-312)

```python
sim_state = self._ensure_sim_state()
pose = sim_state.pose()
face_gauche_monde = np.array([g.face_gauche_x, g.face_gauche_y, g.face_gauche_z])
pinch_L = world_to_robot_local(face_gauche_monde, pose)
...
RETREAT_GAP_Y = 0.125
aim_L = np.array([pinch_L[0], pinch_L[1] + RETREAT_GAP_Y, pinch_L[2]])
q_aim_L = solve_arm_ik(..., aim_L, q_waypoint_L, lock_index=ELBOW_PITCH_CHAIN_INDEX, ...)
...
squeeze_L = np.array([pinch_L[0], pinch_L[1] - SQUEEZE_OFFSET_Y, pinch_L[2]])
q_squeeze_L = solve_arm_ik(..., squeeze_L, q_aim_L, ..., null_space_pref=q_aim_L)
```

Reconvertit les coordonnées monde transmises par `chef_node.py` en repère
local (`world_to_robot_local`), avec la pose **actuelle** du robot (lue
en direct, jamais mise en cache). Calcule ensuite 2 postures cibles AVANT
de bouger : `q_aim_L` (visée, main en retrait de `RETREAT_GAP_Y` du
carton) et `q_squeeze_L` (serrage, main légèrement au-delà de la surface)
-- `q_squeeze_L` est **ancré** (`null_space_pref=q_aim_L`) sur la posture
de visée pour éviter que le solveur change de branche épaule/coude entre
les deux (le bug de rotation d'avant-bras déjà documenté).

### Phase "approche" (lignes 314-327)

```python
self._move_arms(lever, Q_LEFT_HOME, q_waypoint_L, ..., g.waypoint_duration)
self._move_arms(lever, q_waypoint_L, q_aim_L, ..., g.approach_duration)
self._hold(lever, q_aim_L, q_aim_R, goal_handle, 2.0)
```

Repos -> point de passage "coudes vers l'arrière" -> visée. `_move_arms`
interpole en **espace articulaire** (comme `move_arms` côté robot réel).
Se termine par un maintien de 2s à la position de visée.

### Phase "serrage" (lignes 329-339)

```python
self._move_arms(lever, q_aim_L, q_squeeze_L, q_aim_R, q_squeeze_R, g.squeeze_duration)
```

Ferme tout l'écart Y restant depuis `q_aim_L` jusqu'à `q_squeeze_L` --
**le contact physique avec le carton commence ici**.

### Phase "levée" (lignes 359-405)

```python
for idx in LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES:
    lever.set_gains(idx, stiffness=LEVEE_ARM_STIFFNESS)
...
for i in range(n + 1):
    z = squeeze_L[2] + a * (g.lift_z - squeeze_L[2])
    qL = solve_arm_ik(..., np.array([squeeze_L[0], squeeze_L[1], z]), qL,
                       ..., null_space_pref=q_squeeze_L)
    qR = mirror_left_to_right(qL)
    lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))
```

`LEVEE_ARM_STIFFNESS=150` (au lieu de 250 par défaut) : rigidité réduite
pendant le port de charge, même logique que `LEVEE_STIFFNESS`/
`LEVEE_DAMPING` côté robot réel. La rampe résout l'IK à **chaque pas**
(Z monte progressivement), ancrée sur `q_squeeze_L` fixe pour ne pas
lâcher la prise en dérivant vers une autre posture. Si `release_after`,
la fonction redescend, retire les bras et relâche -- sinon (`False`, cas
utilisé par `chef_node.py`) elle s'arrête là, bras/carton tenus, en
attente de `pivot()`.

---

## 7. `pivot.py` -- rotation du buste, carton tenu (Action Server)

Le plus dense des 3, à cause de la fusion retract/pivot/extend et de la
publication combinée bras+jambes+buste.

### Recalcul indépendant de la posture tenue (lignes 145-202)

```python
pinch_L = world_to_robot_local(face_gauche_monde, pose)
...
aim_L = np.array([pinch_L[0], pinch_L[1] + RETREAT_GAP_Y, pinch_L[2]])
q_aim_L = solve_arm_ik(..., aim_L, WAYPOINT_Q_LEFT, ...)
q_squeeze_L = solve_arm_ik(..., squeeze_L, q_aim_L, ..., null_space_pref=q_aim_L)
```

`pivot.py` est un **process séparé** de `lift.py` (son propre `Lever`,
sa propre mémoire) -- il ne connaît PAS le `qL` interne que `lift.py`
vient de calculer. Il doit donc reconstruire lui-même la posture tenue,
à partir des mêmes coordonnées monde transmises par `chef_node.py`.
**Point critique** (corrigé le 2026-09-16, voir commentaire du fichier) :
cette reconstruction utilise EXACTEMENT la même chaîne de calcul que
`lift.py` (même cible `aim_L`, même seed `WAYPOINT_Q_LEFT`) pour
reconverger sur une posture **bit-à-bit identique** -- vérifié
numériquement. Avant ce fix, les deux calculs indépendants pouvaient
diverger de ~20° en `SHOULDER_YAW` pour la même position de main, visible
comme une rotation de l'avant-bras au relais.

### `_publish_pose` (lignes 124-131) -- bras + jambes + buste en un message

```python
def _publish_pose(self, lever, qL, qR, waist_angle, leg_targets, stiffness_scale):
    for idx, _, kp, kd in leg_targets:
        lever.set_gains(idx, kp * stiffness_scale, kd * stiffness_scale)
    indices = LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES + [idx for idx, *_ in leg_targets] + [WAIST_JOINT_INDEX]
    angles = list(qL) + list(qR) + [target for _, target, *_ in leg_targets] + [waist_angle]
    lever.set_batch(indices, angles)
```

Contrairement à `lift.py` (qui ne touche jamais aux jambes), `pivot.py`
doit **republier les jambes en continu** pendant tout le pivot (elles
sont dans une posture de "marche au repos" hérité de la marche
précédente, `leg_targets`) -- sinon elles reviendraient à la posture
native du contrôleur, incohérent avec le reste du corps.

### Fusion rapproché/pivot/tendu (lignes 209-266)

```python
pinch_far_L = squeeze_L.copy()
pinch_near_L = np.array([RETRACT_X, squeeze_L[1], squeeze_L[2]])
retract_anchor_L = q_squeeze_L.copy()
for i in range(n_retract + 1):
    target = pinch_far_L + a * (pinch_near_L - pinch_far_L)
    q_squeeze_L = solve_arm_ik(..., target, q_squeeze_L, ..., null_space_pref=retract_anchor_L)
    self._publish_pose(lever, q_squeeze_L, q_squeeze_R, 0.0, leg_targets, ...)
...
for i in range(n + 1):  # pivot pur, bras figés a pinch_near_L
    self._publish_pose(lever, q_squeeze_L, q_squeeze_R, a * angle_target, leg_targets, ...)
...
# maintien pivote, puis :
extend_anchor_L = q_squeeze_L.copy()
for i in range(n_extend + 1):
    target = pinch_near_L + a * (pinch_far_L - pinch_near_L)
    ...
```

Contrairement au robot réel où retract/extend sont maintenant **fusionnés
avec la rotation du buste** (une seule boucle continue, voir
`CODE_ROBOT_REEL.md`), la simulation garde ici **3 étapes séquentielles**
distinctes : rapproché (bras s'arrêtent), pivot pur (bras figés), tendu
(bras s'arrêtent à nouveau). Chaque étape a sa propre ancre null-space
fixe (`retract_anchor_L`, `extend_anchor_L`) capturée avant sa boucle --
mais PAS de fusion temporelle entre les trois comme côté réel. C'est une
divergence actuelle entre les deux implémentations qui pourrait être
portée ici plus tard si besoin.

### Dépivotage avec gains qui s'assouplissent (lignes 268-289)

```python
if do_depivot:
    for i in range(n + 1):
        kp = WAIST_KP_HOLD + a * (WAIST_KP_RELEASE - WAIST_KP_HOLD)
        kd = WAIST_KD_HOLD + a * (WAIST_KD_RELEASE - WAIST_KD_HOLD)
        lever.set_gains(WAIST_JOINT_INDEX, kp, kd)
        self._publish_pose(lever, q_squeeze_L, q_squeeze_R, (1.0 - a) * angle_target, ...)
```

Rampe SIMULTANÉE de l'angle du buste (rigide -> 0°) ET des gains PD du
buste (`WAIST_KP_HOLD=500`/`WAIST_KD_HOLD=10`, nécessaires pour une vraie
rotation, -> `WAIST_KP_RELEASE=80`/`WAIST_KD_RELEASE=2`, valeurs stables
au relâchement) -- pour que le buste soit déjà souple AVANT que les bras
ne lâchent le carton, évitant le choc "perte brutale de charge + buste
encore rigide" qui causait une chute (confirmée par télémétrie). Ce
`do_depivot` n'a lieu que si `depivot_before_release=True` (pas le cas
dans `chef_node.py::run_sequence` actuel -- voir section 3) ou en cas
d'annulation.

---

## 8. `depose.py` -- repose le carton (Action Server)

Structure similaire à `pivot.py` (même recalcul indépendant de la
posture, même vérification bit-à-bit identique) mais gère toute la
séquence de dépose : baisse, ouverture, écartement, retrait, dépivot.

### `_bend_knees` avec republication forcée (lignes 105-144)

```python
def _bend_knees(self, lever, scale, stiffness_scale, duration, qL_hold, qR_hold, waist_hold, rate_hz=30):
    ...
    for i in range(n + 1):
        indices = leg_indices + LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES + [WAIST_JOINT_INDEX]
        angles = leg_angles + list(qL_hold) + list(qR_hold) + [float(waist_hold)]
        lever.set_batch(indices, angles)
```

Différence notable avec `lift.py::_bend_knees` : celui-ci prend en plus
`qL_hold`/`qR_hold`/`waist_hold` et les republie À CHAQUE tick, pas
seulement les jambes. Raison (voir docstring complète) : `/motion/
joint_override_command` est un **remplacement complet** à chaque
message (pas une fusion additive) -- si le tout premier message de ce
node (nouveau `Lever`, tout juste créé) ne contenait QUE les jambes, le
carton serait instantanément lâché avant même que ce fichier ait eu le
temps de recalculer la position des bras plus bas dans la fonction.

### Séquence complète (lignes 274-408)

```
tendre les bras (coudes redressés)
  -> depose (Z descend, TOUJOURS serré)
  -> tendre les bras (ouvre le serrage, PAS encore relâché)
  -> écartement (Y élargi)
  -> translation arrière (X ramené près du corps)
  -> dégagement (coudes vers l'arrière) -- LE CARTON EST RELÂCHÉ ICI
  -> depivot (buste revient à 0°)
  -> retour bras le long du corps
  -> retrait jambes (si walk_stance)
  -> relâchement final
```

Chaque transition a sa propre justification en commentaire dans le
fichier (toutes ajoutées sur demande explicite de l'utilisateur au fil
des tests -- écartement pour ne pas racler le carton, translation arrière
pour ne pas retraverser la zone de serrage en espace articulaire, etc.).
Le point le plus récent et le plus important architecturalement :

```python
if g.depivot_from_deg != 0.0:
    # depivot AVANT le retour bras home (etait apres) -- evite un
    # desequilibre STATIQUE (masse des bras decalee par rapport aux
    # pieds pendant que le buste est encore tourne).
    for i in range(n_depivot + 1):
        waist = (1.0 - a) * waist_hold
        lever.set_batch(..., list(qL) + list(qR) + [float(waist)])

self.get_logger().info(f"degagement -- retour bras home ({g.degagement_duration:.1f}s)")
qL, qR = self._move_arms(lever, qL, Q_LEFT_HOME, qR, Q_RIGHT_HOME, g.degagement_duration)
```

L'ordre **dépivot avant retour des bras** (et non l'inverse) est le fix
le plus récent de ce fichier : ramener les bras au corps AVANT de
redresser le buste créait un déséquilibre mesuré en télémétrie (~14° de
torsion du bassin en moins d'une seconde) au moment précis où les bras
arrivaient à leur posture finale pendant que le buste était encore
tourné. Validé par l'utilisateur après correction ("l'inversion a l'air
nickel").

---

## Récapitulatif : différences avec le robot réel

| | Simulation | Robot réel |
|---|---|---|
| Architecture | 5 Action Servers séparés, 1 orchestrateur (`chef_node.py`) | 1 seul process/`Lever` |
| Position du carton | Lue dans le XML de la scène (`carton_face_centers`) | Mesurée par vision (ArUco) |
| Position du robot | Vérité terrain LCM (`SimStateListener`) | Proprioception réelle |
| Retract/pivot/extend | 3 étapes séquentielles (`pivot.py`) | Fusionnées en une boucle continue (`levee_pivot.py`) |
| Marche | Protocole réel + pont de traduction (`body_vel_bridge.py`) | Protocole réel direct (pas de pont) |
| Confirmation manuelle | Aucune (séquence automatique) | Entre chaque étape par défaut (`_checkpoint`) |
