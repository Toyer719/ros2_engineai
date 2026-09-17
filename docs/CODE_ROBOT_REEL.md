# Le code du robot réel, bloc par bloc

Ce document explique **tout** le code de `package_sequence_bras_reel/` --
les 6 fichiers qui font marcher la séquence "prise du carton -> pivot ->
dépose" sur le PM01 physique, **du premier à la dernier ligne de chaque
fonction** -- pas seulement les blocs les plus intéressants. Ordre de
lecture pensé pour être pédagogique (les fondations d'abord, l'orchestrateur
en dernier), pas l'ordre alphabétique.

Pour l'architecture générale (pourquoi un seul process, comment ça
communique avec le robot), voir le [README principal](../README.md). Pour
un tableau croisé "quelle capacité, quelle fonction", voir l'artefact
personnel dédié (catalogue des fonctions de mouvement).

---

## 1. `lift_carton.py` (104 lignes) -- la géométrie et la cinématique inverse

### Les chaînes cinématiques (lignes 3-19)

```python
LEFT_CHAIN = [
    ("J13_SHOULDER_PITCH_L", np.array([0, 1, 0]), np.array([-0.027105, 0.12916, 0.21549])),
    ("J14_SHOULDER_ROLL_L",  np.array([1, 0, 0]), np.array([-0.0371, 0.066941, -0.020838])),
    ("J15_SHOULDER_YAW_L",   np.array([0, 0, 1]), np.array([0.0371, 0.017645, -0.070132])),
    ("J16_ELBOW_PITCH_L",    np.array([0, 1, 0]), np.array([0, 0.0065994, -0.10487])),
    ("J17_ELBOW_YAW_L",      np.array([0, 0, 1]), np.array([0.013817, 0.0097723, -0.1547])),
]
HAND_OFFSET_LEFT = np.array([0.03, -0.02, -0.14])

RIGHT_CHAIN = [ ... ]  # memes valeurs, Y inverse (symetrie miroir)
HAND_OFFSET_RIGHT = np.array([0.03, 0.02, -0.14])
```

Chaque ligne décrit **une articulation** : son nom, l'**axe** autour duquel
elle tourne (`[0,1,0]` = autour de Y, etc.), et le **vecteur de décalage**
depuis l'articulation précédente (en mètres, dans le repère de
l'articulation parente). `HAND_OFFSET_LEFT`/`HAND_OFFSET_RIGHT` sont le
dernier décalage, de la 5ᵉ articulation (poignet) jusqu'au centre de la
main. `RIGHT_CHAIN` est la même chose côté droit, valeurs Y inversées.

### `rotation_matrix` (lignes 22-25)

```python
def rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
```

Formule de **Rodrigues** : convertit un axe + un angle en matrice de
rotation 3x3 (`k` est la matrice antisymétrique du produit vectoriel par
`axis`). Utilisée pour composer les rotations d'une articulation à la
suivante.

### `forward_kinematics` (lignes 28-34) -- "si je connais les 5 angles, où est la main ?"

```python
def forward_kinematics(chain, hand_offset, q):
    pos = np.zeros(3)
    rot = np.eye(3)
    for (_, axis, offset), angle in zip(chain, q):
        pos = pos + rot @ offset
        rot = rot @ rotation_matrix(axis, angle)
    return pos + rot @ hand_offset
```

Parcourt les 5 articulations dans l'ordre : à chaque étape, avance la
position (`pos`) du décalage de cette articulation (tourné par
l'orientation accumulée jusque-là, `rot`), puis met à jour l'orientation
avec la rotation de CETTE articulation (`rot = rot @ rotation_matrix(...)`,
composition de rotations). À la fin, ajoute le décalage vers la main
(lui aussi tourné par l'orientation finale accumulée). C'est le calcul
**direct** (angles connus -> position de la main).

### `numerical_jacobian` (lignes 37-44) -- "si je bouge un peu chaque angle, comment bouge la main ?"

```python
def numerical_jacobian(chain, hand_offset, q, eps=1e-6):
    p0 = forward_kinematics(chain, hand_offset, q)
    J = np.zeros((3, len(q)))
    for i in range(len(q)):
        dq = q.copy()
        dq[i] += eps
        J[:, i] = (forward_kinematics(chain, hand_offset, dq) - p0) / eps
    return J
```

Pour chaque articulation `i`, bouge son angle d'un tout petit peu
(`eps=1e-6` radian), recalcule la position de la main (`dq` copie `q` puis
modifie SEULEMENT l'angle `i`), et la différence (divisée par `eps`)
donne la **sensibilité** de la position de la main à cette articulation
(dérivée numérique). Empilées dans `J` (3 lignes X/Y/Z, 5 colonnes = 5
articulations), ces sensibilités forment la matrice **Jacobienne**.

### `solve_ik` (lignes 47-56) -- cinématique inverse SANS verrou, 5 degrés de liberté libres

```python
def solve_ik(chain, hand_offset, target, q_init, iters=150, damping=0.05):
    q = q_init.copy()
    for _ in range(iters):
        error = target - forward_kinematics(chain, hand_offset, q)
        if np.linalg.norm(error) < 1e-6:
            break
        J = numerical_jacobian(chain, hand_offset, q)
        JJt = J @ J.T + damping ** 2 * np.eye(3)
        q = q + J.T @ np.linalg.solve(JJt, error)
    return q
```

Algorithme de **Newton amorti** ("damped least squares") : part de
`q_init`, mesure l'écart (`error`) entre la main actuelle et la cible,
s'arrête si l'écart est déjà minuscule (`< 1e-6`), sinon calcule la
Jacobienne et résout `JJt . x = error` (`JJt = J@J.T + damping²·I`, le
terme d'amortissement pour éviter les mouvements erratiques près d'une
singularité) pour trouver le pas `x`, puis avance `q` de `J.T @ x`. Répété
jusqu'à `iters` fois. 5 inconnues (les 5 angles) pour seulement 3
équations (X,Y,Z) -- infinité de solutions possibles, celle trouvée
dépend de `q_init` (le point de départ).

### `solve_arm_ik` (lignes 59-90) -- la vraie fonction utilisée partout, avec verrou + préférence

```python
def solve_arm_ik(chain, hand_offset, target, q_init, lock_index=None, lock_angle=None,
                  iters=200, damping=0.05, null_space_gain=0.2, null_space_pref=None):
    if lock_index is None:
        return solve_ik(chain, hand_offset, target, q_init, iters=iters, damping=damping)

    free_idx = [i for i in range(len(q_init)) if i != lock_index]
    q_pref = q_init.copy() if null_space_pref is None else null_space_pref.copy()
    q = q_init.copy()
    q[lock_index] = lock_angle
    n_free = len(free_idx)
    for _ in range(iters):
        p0 = forward_kinematics(chain, hand_offset, q)
        error = target - p0
        if np.linalg.norm(error) < 1e-7:
            break
        J = np.zeros((3, n_free))
        for k, i in enumerate(free_idx):
            dq = q.copy()
            dq[i] += 1e-6
            J[:, k] = (forward_kinematics(chain, hand_offset, dq) - p0) / 1e-6
        JJt = J @ J.T + damping ** 2 * np.eye(3)
        J_pinv = J.T @ np.linalg.inv(JJt)
        step = J_pinv @ error
        if null_space_gain:
            q_free = np.array([q[i] for i in free_idx])
            q_pref_free = np.array([q_pref[i] for i in free_idx])
            null_proj = np.eye(n_free) - J_pinv @ J
            step = step + null_space_gain * (null_proj @ (q_pref_free - q_free))
        for k, i in enumerate(free_idx):
            q[i] += step[k]
        q[lock_index] = lock_angle
    return q
```

Ligne par ligne :

- **`if lock_index is None`** : sans verrou demandé, retombe simplement
  sur `solve_ik` (IK standard, 5 DOF libres).
- **`free_idx`** : la liste des indices d'articulation SAUF celui
  verrouillé (ex : `[0,1,2,4]` si `lock_index=3`).
- **`q_pref`** : la préférence null-space -- si `null_space_pref` n'est
  pas fourni, elle vaut `q_init` par défaut (**piège déjà rencontré** :
  si on rappelle cette fonction en boucle en réutilisant `q` comme SEED
  et préférence à chaque fois, la préférence glisse avec le mouvement).
- **`q[lock_index] = lock_angle`** : fige tout de suite l'articulation
  verrouillée à sa valeur cible.
- **Boucle** : à chaque itération, calcule l'erreur de position
  (`error`), construit la Jacobienne UNIQUEMENT sur les colonnes libres
  (`n_free` colonnes au lieu de 5), résout le pas normal (`step`) par
  pseudo-inverse amortie (`J_pinv`).
- **`null_proj = np.eye(n_free) - J_pinv @ J`** : la projection sur le
  **noyau** du Jacobien -- l'ensemble des mouvements qui NE changent PAS
  la position de la main. `step + null_space_gain * (null_proj @ (...))`
  ajoute donc un petit rappel vers `q_pref`, garanti sans effet sur la
  position déjà atteinte.
- **`q[lock_index] = lock_angle`** (répété en fin de boucle) : au cas où
  la mise à jour des indices libres aurait indirectement affecté
  l'articulation verrouillée (elle ne le fait pas dans ce code, mais
  cette ligne le garantit explicitement à chaque tour).

### `mirror_left_to_right` (lignes 93-98)

```python
def mirror_left_to_right(q_left):
    q_right = q_left.copy()
    q_right[1] *= -1
    q_right[2] *= -1
    q_right[4] *= -1
    return q_right
```

Inverse les index 1 (SHOULDER_ROLL), 2 (SHOULDER_YAW) et 4 (ELBOW_YAW) ;
0 (SHOULDER_PITCH) et 3 (ELBOW_PITCH) restent identiques -- convention
géométrique de ce squelette (les rotations "pitch" sont symétriques par
nature, les rotations autour d'un axe vertical/horizontal-latéral
s'inversent en miroir). Le bras droit n'a **jamais** sa propre résolution
IK dans ce projet -- toujours dérivé du gauche.

### `ease` (lignes 101-103)

```python
def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)
```

Un **smoothstep** classique (3t²-2t³) : vitesse nulle en `t=0` et `t=1`,
maximale au milieu. Utilisée pour interpoler en douceur (pas de
démarrage/arrêt brutal) toutes les rampes de mouvement du projet.

---

## 2. `lever.py` (116 lignes) -- la classe qui parle au robot

### Constantes (lignes 8-12)

```python
NUM_JOINTS = 24
TOPIC = "/motion/joint_override_command"
DEFAULT_STIFFNESS = [float(v) for v in [200, 200, 380, 450, 400, 200] * 2 + [200] + [250] * 10 + [100]]
DEFAULT_DAMPING = [float(v) for v in [5, 5, 5, 5, 2, 2] * 2 + [1] + [1] * 10 + [1]]
```

24 joints au total. `DEFAULT_STIFFNESS`/`DEFAULT_DAMPING` construisent la
liste des 24 gains par défaut : `[200,200,380,450,400,200]` répété 2 fois
(12 valeurs = les 2 jambes, hanche/hanche/hanche/genou/cheville/cheville
x2), puis `[200]` (1 valeur = le buste, index 12), puis `[250]*10` (les 2
bras, 5 joints chacun), puis `[100]` (la tête, dernier index).

### `__init__` (lignes 17-31)

```python
def __init__(self, node, subscriber_timeout=5.0):
    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )
    self._node = node
    self._pub = node.create_publisher(JointOverrideCommand, TOPIC, qos)
    self._position = [0.0] * NUM_JOINTS
    self._touched = [False] * NUM_JOINTS
    self._stiffness = list(DEFAULT_STIFFNESS)
    self._damping = list(DEFAULT_DAMPING)
    self._weight = 1.0
    self._wait_for_subscriber(subscriber_timeout)
```

Crée le publisher ROS2 avec une QoS `BEST_EFFORT`/`VOLATILE`/`KEEP_LAST(1)`
(ne retransmet jamais un message manqué, ne garde que le dernier). Initialise
`_position`/`_touched` (24 zéros/`False`), copie les gains par défaut, et
`_weight=1.0` (contrôle total dès le départ). Termine par
`_wait_for_subscriber` -- ne retourne qu'une fois un abonné réel détecté.

### `set_weight` (lignes 33-36)

```python
def set_weight(self, weight):
    self._weight = float(weight)
    if any(self._touched):
        self._publish()
```

Change le poids global de l'override. Si au moins un joint est déjà
"touché" (actif), republie immédiatement pour appliquer le nouveau poids
tout de suite plutôt que d'attendre la prochaine écriture de position.

### `set_gains` (lignes 38-44)

```python
def set_gains(self, joint_index, stiffness=None, damping=None):
    if stiffness is not None:
        self._stiffness[joint_index] = float(stiffness)
    if damping is not None:
        self._damping[joint_index] = float(damping)
    if self._touched[joint_index]:
        self._publish()
```

Change kp/kd d'UN joint (les deux sont optionnels : passer seulement
`stiffness=...` laisse `damping` inchangé). Republie immédiatement si ce
joint précis est déjà actif.

### `_wait_for_subscriber` (lignes 46-51)

```python
def _wait_for_subscriber(self, timeout):
    t0 = time.time()
    while self._pub.get_subscription_count() == 0:
        if time.time() - t0 > timeout:
            raise RuntimeError(f"Aucun abonne sur {TOPIC} apres {timeout}s.")
        time.sleep(0.05)
```

Boucle d'attente active (poll toutes les 50ms) jusqu'à ce qu'un abonné
apparaisse sur le topic, ou lève une exception après `timeout` secondes.
Nécessaire car la QoS `BEST_EFFORT`/`VOLATILE` ne retransmet rien -- publier
avant que la découverte DDS soit terminée côté `src_executor` enverrait les
premiers messages dans le vide, sans erreur visible.

### `__setitem__`/`set_batch` (lignes 53-62) -- le cœur de la classe

```python
def __setitem__(self, joint_index, angle_rad):
    self._position[joint_index] = angle_rad
    self._touched[joint_index] = True
    self._publish()

def set_batch(self, indices, angles):
    for idx, angle in zip(indices, angles):
        self._position[idx] = float(angle)
        self._touched[idx] = True
    self._publish()
```

`lever[13] = 0.5` modifie **un seul** joint et publie **immédiatement**.
Faire ça pour 10 joints d'affilée envoie 10 messages ROS2 séparés --
cause des vibrations de bras déjà diagnostiquée dans ce projet.
`set_batch` corrige ça : met à jour TOUS les indices donnés en mémoire
**avant** d'appeler `_publish()` une seule fois.

### `__getitem__` (lignes 64-65)

```python
def __getitem__(self, joint_index):
    return self._position[joint_index]
```

Lit la dernière position ÉCRITE localement pour un joint (pas une lecture
en direct du robot -- juste l'état interne mémorisé).

### `release`/`forget`/`untouch` (lignes 67-81)

```python
def release(self):
    msg = JointOverrideCommand()
    msg.weight = 0.0
    self._pub.publish(msg)
    self._touched = [False] * NUM_JOINTS

def forget(self):
    self._touched = [False] * NUM_JOINTS
    self._stiffness = list(DEFAULT_STIFFNESS)
    self._damping = list(DEFAULT_DAMPING)

def untouch(self, indices):
    for i in indices:
        self._touched[i] = False
    self._publish()
```

- **`release()`** : publie un message `weight=0.0` (indices vides, donc
  RIEN n'est indiqué comme override actif) et remet tous les `_touched`
  à `False` -- rend TOUT au contrôleur natif d'un coup.
- **`forget()`** : remet `_touched`/gains à zéro/défaut SANS publier --
  pour repartir propre en tête d'une nouvelle séquence dans le même
  process, sans envoyer de message.
- **`untouch(indices)`** : relâchement PARTIEL, republie ensuite (donc
  les joints retirés de `_touched` disparaissent du prochain message,
  les autres restent tenus).

### `_publish` (lignes 83-95) -- ce qui part réellement sur le réseau

```python
def _publish(self):
    indices = [i for i, touched in enumerate(self._touched) if touched]
    msg = JointOverrideCommand()
    msg.header.stamp = self._node.get_clock().now().to_msg()
    msg.weight = self._weight
    msg.joint_indices = indices
    msg.position = [self._position[i] for i in indices]
    msg.velocity = [0.0] * len(indices)
    msg.feed_forward_torque = [0.0] * len(indices)
    msg.torque = [0.0] * len(indices)
    msg.stiffness = [self._stiffness[i] for i in indices]
    msg.damping = [self._damping[i] for i in indices]
    self._pub.publish(msg)
```

Construit le message avec uniquement les indices `_touched` (les autres
ne sont pas mentionnés, laissés au contrôleur natif). `velocity`/
`feed_forward_torque`/`torque` restent toujours à zéro dans ce projet --
seul le contrôle en POSITION (avec gains PD) est utilisé, jamais le
contrôle en effort/vitesse directe.

### `main` (lignes 98-115) -- démo minimale, exécutable seule

```python
def main():
    rclpy.init()
    node = rclpy.create_node("lever_control")
    lever = Lever(node)
    lever[13] = 0.5
    lever[18] = 0.5
    lever[23] = 0.3
    time.sleep(3)
    lever.release()
    node.destroy_node()
    rclpy.shutdown()
```

Un script de test autonome (`python3 lever.py`) : lève un peu l'épaule
gauche (joint 13) et droite (joint 18), tourne un peu la tête (joint 23),
maintient 3 secondes, relâche. Sert à vérifier que `Lever`/le topic
fonctionnent, indépendamment du reste de la séquence.

---

## 3. `motion_state.py` (69 lignes) -- le verrou de sécurité

Une seule fonction : `ensure_motion_state(node, target, timeout, detour)`.

### Setup des topics (lignes 4-20)

```python
def ensure_motion_state(node, target, timeout=3.0, detour="pd_stand"):
    from interface_protocol.msg import MotionState, MotionStateRequest
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    import rclpy

    state = {"current": "", "available": []}

    def _cb(msg):
        state["current"] = msg.current_motion_task
        state["available"] = list(msg.available_transition_motions)

    state_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                            durability=DurabilityPolicy.VOLATILE)
    request_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
    sub = node.create_subscription(MotionState, "/motion/motion_state", _cb, state_qos)
    pub = node.create_publisher(MotionStateRequest, "/motion/set_motion_state", request_qos)
```

Les imports sont faits À L'INTÉRIEUR de la fonction (pas en tête de
fichier) -- évite de dépendre de `interface_protocol` au simple `import
motion_state`, seulement quand la fonction est réellement appelée. `state`
(dict local) est mis à jour par le callback `_cb` à chaque message reçu
sur `/motion/motion_state`. Note la QoS différente : lecture d'état en
`BEST_EFFORT` (peu grave de rater un message, le prochain arrive vite),
demande de changement en `RELIABLE` (celui-là ne doit pas se perdre).

### `_wait_for_state`/`_switch_to` (lignes 22-43)

```python
def _wait_for_state(t=1.0):
    t0 = time.time()
    while time.time() - t0 < t:
        rclpy.spin_once(node, timeout_sec=0.05)
        if state["current"]:
            return

def _switch_to(name):
    _wait_for_state()
    if state["current"] == name:
        return True
    if name not in state["available"]:
        return False
    msg = MotionStateRequest()
    msg.target_motion_name = name
    pub.publish(msg)
    t0 = time.time()
    while time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.05)
        if state["current"] == name:
            return True
    return False
```

`_wait_for_state` : appelle `rclpy.spin_once` en boucle pour laisser
passer au moins un message (le callback `_cb` ne se déclenche que si le
node "spin"). `_switch_to(name)` : si déjà dans l'état voulu, ne fait
rien (`True` immédiat) ; si l'état cible n'est même pas listé comme
atteignable (`available`), échoue tout de suite (`False`) ; sinon publie
la demande et attend (en spinnant) jusqu'à `timeout` que `state["current"]`
devienne `name`.

### Le corps de la fonction : détour automatique (lignes 45-68)

```python
_wait_for_state()
print(f"[ETAPE] etat de mouvement actuel : {state['current']}", flush=True)
if state["current"] == target:
    node.destroy_subscription(sub)
    node.destroy_publisher(pub)
    return True

if target not in state["available"] and detour in state["available"]:
    print(f"[ETAPE] {target} non atteignable directement -- detour par {detour}...", flush=True)
    if not _switch_to(detour):
        print(f"[ERREUR] echec du passage par {detour}.", flush=True)
        node.destroy_subscription(sub)
        node.destroy_publisher(pub)
        return False

print(f"[ETAPE] bascule vers {target}...", flush=True)
ok = _switch_to(target)
node.destroy_subscription(sub)
node.destroy_publisher(pub)
if not ok:
    print(f"[ERREUR] echec du passage en {target} (etat actuel : {state['current']}).", flush=True)
else:
    print(f"[ETAPE] mode {target} actif.", flush=True)
return ok
```

Logique complète : (1) si déjà dans l'état cible, retourne tout de suite
`True` ; (2) si la cible n'est pas directement atteignable MAIS que le
détour l'est, bascule d'abord vers le détour ; (3) tente ensuite la cible
finale. `destroy_subscription`/`destroy_publisher` sont appelés à CHAQUE
sortie (succès ou échec) pour ne pas laisser de ressources ROS2 orphelines
sur le node appelant (qui, lui, continue de vivre après cet appel).

---

## 4. `levee.py` (298 lignes) -- bibliothèque de mouvement + script "lever seul"

### Constantes géométriques et de timing (lignes 20-65)

```python
LEFT_JOINT_INDICES = [13, 14, 15, 16, 17]
RIGHT_JOINT_INDICES = [18, 19, 20, 21, 22]
WRIST_CHAIN_INDEX = 4

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

RATE_HZ = 65

PINCH_X = 0.216
PINCH_Y = 0.22
SQUEEZE_Y = 0.13
LIFT_Z = 0.15
APPROACH_DURATION = 1.5
WAYPOINT_DURATION = 2.1
APPROACH_LIFT_DURATION = 0.6
SQUEEZE_DURATION = 2.1
LIFT_DURATION = 2.0
HOLD_SECONDS = 0.0
RELEASE_RAMP_SECONDS = 4.0
MOTION_STATE_TIMEOUT = 3.0

ELBOW_YAW_ROTATION_DEG = 0.0

WALK_STANCE_SCALE = 0.0
WALK_STANCE_STIFFNESS_SCALE = 1.8
WALK_STANCE_DURATION = 1.9

LEVEE_STIFFNESS = 130.0
LEVEE_DAMPING = 3.0
```

- `LEFT_JOINT_INDICES`/`RIGHT_JOINT_INDICES` : les 5 index de chaque bras
  dans le tableau des 24 joints.
- `WRIST_CHAIN_INDEX = 4` : l'index verrouillé par `solve_arm_ik` sur le
  robot réel -- c'est l'ELBOW_YAW dans la convention de la chaîne
  (dernier maillon avant la main), utilisé ici comme "poignet".
- `Q_LEFT_HOME`/`Q_RIGHT_HOME` : posture "bras le long du corps", mesurée
  sur le robot réel (pas une valeur ronde).
- `WAYPOINT_Q_LEFT`/`WAYPOINT_Q_RIGHT` : posture "coudes vers l'arrière,
  avant-bras horizontal", validée par cinématique directe (voir la
  docstring de `WAYPOINT_Q_LEFT` dans `lift.py`/sim pour l'historique
  complet de sa mise au point).
- `RATE_HZ = 65` : fréquence de publication de toutes les rampes de ce
  fichier (choisie pour dépasser la fréquence du contrôleur bas niveau
  réel, ~500Hz, sans le saturer -- voir mémoire projet pour l'historique
  de ce réglage anti-vibration).
- `LEVEE_STIFFNESS`/`LEVEE_DAMPING` : gains PD réduits/renforcés
  spécifiquement pour porter une charge (250 par défaut sinon).
- Toutes les `*_DURATION` : réglages de vitesse de la séquence,
  actuellement les valeurs accélérées de ~40% par rapport à l'origine
  (voir historique git).

### `_checkpoint` (lignes 68-71)

```python
def _checkpoint(message, confirm):
    print(f"[ETAPE] {message}", flush=True)
    if confirm:
        input("        Verifie le robot, puis Entree pour continuer (Ctrl+C pour arreter)... ")
```

Affiche l'étape en cours et, si `confirm=True` (comportement par défaut),
**bloque** en attendant une touche Entrée.

### `_rotate_xy` (lignes 74-77)

```python
def _rotate_xy(point, yaw_offset):
    x, y, z = point
    c, s = np.cos(yaw_offset), np.sin(yaw_offset)
    return np.array([x * c - y * s, x * s + y * c, z])
```

Rotation 2D standard autour de Z, Z inchangé -- corrige la cible de pince
si le robot ne s'arrête pas parfaitement de face au carton
(`pinch_yaw_offset`, quasiment toujours 0.0 en pratique).

### `_quintic_ease` (lignes 80-82)

```python
def _quintic_ease(t):
    t = max(0.0, min(1.0, t))
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)
```

Interpolation quintique (vitesse ET accélération nulles aux deux bords)
-- plus douce qu'un smoothstep simple, imite le comportement natif du
contrôleur `pd_stand` (utilisée uniquement pour la flexion des genoux).

### `_bend_knees` (lignes 85-104)

```python
def _bend_knees(lever, scale, stiffness_scale, duration, rate_hz=RATE_HZ, dry_run=False):
    if dry_run:
        print(f"    [dry-run] flexion genoux -- scale={scale} duration={duration}s")
        return
    for idx, kp, kd in [
        (LEFT_HIP_PITCH_INDEX, 200.0, 5.0), (RIGHT_HIP_PITCH_INDEX, 200.0, 5.0),
        (LEFT_KNEE_PITCH_INDEX, 450.0, 5.0), (RIGHT_KNEE_PITCH_INDEX, 450.0, 5.0),
        (LEFT_ANKLE_PITCH_INDEX, 400.0, 2.0), (RIGHT_ANKLE_PITCH_INDEX, 400.0, 2.0),
    ]:
        lever.set_gains(idx, kp * stiffness_scale, kd * stiffness_scale)
    n = max(1, int(duration * rate_hz))
    for i in range(n + 1):
        a = _quintic_ease(i / n)
        lever[LEFT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_L)
        lever[RIGHT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_R)
        lever[LEFT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_L)
        lever[RIGHT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_R)
        lever[LEFT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_L)
        lever[RIGHT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_R)
        time.sleep(1.0 / rate_hz)
```

En `dry_run`, se contente d'un print et retourne -- ne touche jamais au
`lever` (permet d'appeler cette fonction même sans robot connecté). Sinon,
règle d'abord les 6 gains (hanche/genou/cheville x2, `kp`/`kd` de base
multipliés par `stiffness_scale`), puis interpole les 6 angles de 0 vers
`scale * WALK_STANCE_*` avec la courbe quintique. **Remarque** : ici
chaque `lever[idx] = ...` est un appel SÉPARÉ (pas `set_batch`) -- 6
messages ROS2 par pas pour cette fonction précise (contrairement au reste
du fichier qui utilise systématiquement `set_batch`/`_publish`).

### `_straighten_knees` (lignes 107-120)

```python
def _straighten_knees(lever, scale, stiffness_scale, duration, rate_hz=RATE_HZ, dry_run=False):
    if dry_run:
        print(f"    [dry-run] redressement genoux -- scale={scale} duration={duration}s")
        return
    n = max(1, int(duration * rate_hz))
    for i in range(n + 1):
        a = 1.0 - _quintic_ease(i / n)
        lever[LEFT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_L)
        # ... (memes 6 joints, meme structure que _bend_knees)
        time.sleep(1.0 / rate_hz)
```

Inverse de `_bend_knees` : `a = 1.0 - _quintic_ease(...)` parcourt la
MÊME courbe mais à l'envers (1→0 au lieu de 0→1) -- redescend vers 0 avec
la même douceur aux deux bords. Ne réinitialise PAS les gains (ceux fixés
par `_bend_knees` restent, supposés déjà corrects).

### `_publish` (lignes 123-124)

```python
def _publish(lever, qL, qR):
    lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))
```

Le point de passage unique par lequel TOUTES les fonctions de ce fichier
(sauf `_bend_knees`/`_straighten_knees`) publient les bras -- garantit
l'usage systématique de `set_batch` (publication atomique) plutôt qu'un
oubli ponctuel.

### `move_arms` (lignes 127-139)

```python
def move_arms(lever, qL0, qL1, qR0, qR1, duration, dry_run=False):
    n = max(1, int(duration * RATE_HZ))
    for i in range(n + 1):
        a = ease(i / n)
        qL = qL0 + a * (qL1 - qL0)
        qR = qR0 + a * (qR1 - qR0)
        if dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i/RATE_HZ:.2f}s  qL={np.round(qL, 4)}  qR={np.round(qR, 4)}")
            continue
        _publish(lever, qL, qR)
        time.sleep(1.0 / RATE_HZ)
    return qL, qR
```

Interpolation ARTICULAIRE directe (`ease` = smoothstep de `lift_carton.py`)
entre 2 postures complètes. En `dry_run`, affiche seulement le premier et
dernier pas (`i in (0, n)`) pour ne pas noyer la console, mais calcule
quand même TOUTE la boucle (vérifie qu'aucune valeur n'explose en cours
de route). Retourne la dernière posture atteinte, pour que l'appelant
enchaîne dessus.

### `cartesian_ramp` (lignes 142-160)

```python
def cartesian_ramp(lever, q_init, wrist_rotation, start, end, anchor_start, anchor_end,
                    duration, dry_run):
    qL = q_init.copy()
    n = max(1, int(duration * RATE_HZ))
    for i in range(n + 1):
        a = ease(i / n)
        target = start + a * (end - start)
        anchor = (1.0 - a) * anchor_start + a * anchor_end
        qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, target, qL,
                           lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation, iters=10,
                           null_space_pref=anchor)
        qR = mirror_left_to_right(qL)
        if dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i/RATE_HZ:.2f}s  qL={np.round(qL, 4)}  qR={np.round(qR, 4)}")
            continue
        _publish(lever, qL, qR)
        time.sleep(1.0 / RATE_HZ)
    return qL, mirror_left_to_right(qL)
```

Interpole la POSITION 3D cible (`target`) en ligne droite, ET l'ancre du
null-space (`anchor`) séparément -- les deux avec la même courbe `a`, mais
ce sont deux interpolations distinctes (position de la main d'un côté,
préférence de posture de l'autre). Résout `solve_arm_ik` à CHAQUE pas
avec seulement `iters=10` (pas 200) -- suffisant car `qL` (le seed) est
déjà très proche de la solution du pas précédent, donc l'IK converge en
peu d'itérations. `qR` toujours dérivé par miroir, jamais recalculé.

### `_build_arg_parser` (lignes 163-174) -- CLI de `levee.py` seul

```python
def _build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pinch-x", type=float, default=PINCH_X)
    parser.add_argument("--pinch-z", type=float, default=0.05)
    parser.add_argument("--pinch-yaw-offset", type=float, default=0.0)
    parser.add_argument("--wrist-rotation-deg", type=float, default=ELBOW_YAW_ROTATION_DEG)
    parser.add_argument("--walk-stance-scale", type=float, default=WALK_STANCE_SCALE)
    parser.add_argument("--skip-motion-state", action="store_true")
    parser.add_argument("--only-phase", choices=["approche", "serrage", "levee"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-confirm", action="store_true")
    return parser
```

Le CLI de `levee.py` exécuté SEUL (`python3 levee.py`) -- un sous-ensemble
volontairement plus petit que `levee_pivot.py` : pas de `--angle-deg`, pas
des durées de pivot/dépose, puisque ce script s'arrête après la levée (pas
de pivot, pas de dépose). `--only-phase` n'accepte que 3 valeurs ici
(`approche`/`serrage`/`levee`), pas `pivot` (qui n'existe pas dans ce
fichier).

### `run_lift_sequence` (lignes 177-267) -- approche/serrage/levée en standalone

```python
def run_lift_sequence(node, lever, args):
    confirm = not args.no_confirm
    wrist_rotation = np.radians(getattr(args, "wrist_rotation_deg", ELBOW_YAW_ROTATION_DEG))
    walk_stance_scale = getattr(args, "walk_stance_scale", WALK_STANCE_SCALE)

    straight_pref_L = WAYPOINT_Q_LEFT.copy()
    straight_pref_L[3] = 0.0
    q_pinch_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                              _rotate_xy([args.pinch_x, PINCH_Y, args.pinch_z], args.pinch_yaw_offset),
                              Q_LEFT_HOME, lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                              null_space_pref=straight_pref_L)
    q_pinch_R = mirror_left_to_right(q_pinch_L)

    squeeze_L = _rotate_xy([args.pinch_x, SQUEEZE_Y, args.pinch_z], args.pinch_yaw_offset)
    q_squeeze_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_pinch_L,
                                lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                                null_space_pref=q_pinch_L)
    q_squeeze_R = mirror_left_to_right(q_squeeze_L)
```

`getattr(args, ..., DEFAUT)` (au lieu d'un accès direct `args.x`) --
défensif, au cas où cette fonction serait appelée avec un objet `args`
incomplet (pas construit par `_build_arg_parser` de ce fichier). Calcule
ensuite les 2 postures cibles AVANT de bouger : `q_pinch_L` (position de
visée, seed=`Q_LEFT_HOME` -- ICI la posture de REPOS, pas
`WAYPOINT_Q_LEFT` comme dans `lift.py` côté simulation, une petite
différence entre les deux implémentations), `q_squeeze_L` (serrage,
ancré sur `q_pinch_L`, `straight_pref_L` = posture du point de passage
avec le coude à 0° -- une préférence "bras tendu" pour l'IK sans verrou
de `q_pinch_L`).

```python
    run_approche = args.only_phase in (None, "approche")
    run_serrage = args.only_phase in (None, "serrage")
    run_levee = args.only_phase in (None, "levee")

    qL, qR = Q_LEFT_HOME, Q_RIGHT_HOME

    if run_approche and walk_stance_scale > 0:
        _checkpoint(f"flexion genoux -- scale={walk_stance_scale}, {WALK_STANCE_DURATION}s", confirm)
        _bend_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE, WALK_STANCE_DURATION,
                    dry_run=args.dry_run)
```

Flags dérivés de `--only-phase`. Position de départ = repos. Flexion des
genoux SEULEMENT si `walk_stance_scale > 0` (= 0.0 par défaut, donc
sautée en pratique).

```python
    if run_approche:
        _checkpoint(f"point de passage -- coudes vers l'arriere, {WAYPOINT_DURATION}s", confirm)
        qL, qR = move_arms(lever, Q_LEFT_HOME, WAYPOINT_Q_LEFT, Q_RIGHT_HOME, WAYPOINT_Q_RIGHT,
                            WAYPOINT_DURATION, dry_run=args.dry_run)

        pinch_target_L = _rotate_xy([args.pinch_x, PINCH_Y, args.pinch_z], args.pinch_yaw_offset)
        waypoint_hand_L = forward_kinematics(LEFT_CHAIN, HAND_OFFSET_LEFT, WAYPOINT_Q_LEFT)
        raised_point_L = np.array([waypoint_hand_L[0], waypoint_hand_L[1], pinch_target_L[2]])

        _checkpoint(f"approche -- ajustement vertical, {APPROACH_LIFT_DURATION}s", confirm)
        qL, qR = cartesian_ramp(lever, WAYPOINT_Q_LEFT, wrist_rotation, waypoint_hand_L,
                                 raised_point_L, WAYPOINT_Q_LEFT, WAYPOINT_Q_LEFT,
                                 APPROACH_LIFT_DURATION, args.dry_run)

        _checkpoint(
            f"approche -- ligne horizontale vers pinch (x={args.pinch_x} y=+-{PINCH_Y} "
            f"z={args.pinch_z}, poignet pivote de {np.degrees(wrist_rotation):.0f}deg), "
            f"{APPROACH_DURATION}s", confirm)
        qL, qR = cartesian_ramp(lever, qL, wrist_rotation, raised_point_L, pinch_target_L,
                                 WAYPOINT_Q_LEFT, straight_pref_L, APPROACH_DURATION, args.dry_run)
```

L'approche en 3 temps : (1) `move_arms` articulaire de repos vers le
point de passage "coudes-arrière" ; (2) `cartesian_ramp` VERTICALE
(`waypoint_hand_L` -> `raised_point_L`, même X/Y, Z du carton) -- ancre
null-space FIXE (`WAYPOINT_Q_LEFT` des deux côtés) ; (3) `cartesian_ramp`
HORIZONTALE (`raised_point_L` -> `pinch_target_L`) -- ancre qui GLISSE de
`WAYPOINT_Q_LEFT` vers `straight_pref_L`. Découper en vertical PUIS
horizontal (au lieu d'une diagonale directe) évite que la main décrive un
arc imprévisible.

```python
    if run_serrage:
        _checkpoint(
            f"serrage -- Y +-{PINCH_Y} -> +-{SQUEEZE_Y}, {SQUEEZE_DURATION}s "
            "(LE CONTACT AVEC LE CARTON COMMENCE ICI)", confirm)
        qL, qR = move_arms(lever, q_pinch_L, q_squeeze_L, q_pinch_R, q_squeeze_R,
                            SQUEEZE_DURATION, dry_run=args.dry_run)
```

Serrage : `move_arms` articulaire (pas cartésien -- déplacement Y pur,
suffisamment court pour ne pas nécessiter la garantie de trajectoire de
`cartesian_ramp`) entre les 2 postures déjà précalculées.

```python
    if run_levee:
        _checkpoint(
            f"levee -- Z {args.pinch_z} -> {LIFT_Z}, {LIFT_DURATION}s puis maintien "
            f"{HOLD_SECONDS}s", confirm)
        if not args.dry_run:
            for idx in LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES:
                lever.set_gains(idx, stiffness=LEVEE_STIFFNESS, damping=LEVEE_DAMPING)

        lift_start = _rotate_xy([args.pinch_x, SQUEEZE_Y, args.pinch_z], args.pinch_yaw_offset)
        lift_end = _rotate_xy([args.pinch_x, SQUEEZE_Y, LIFT_Z], args.pinch_yaw_offset)
        qL, qR = cartesian_ramp(lever, q_squeeze_L, wrist_rotation, lift_start, lift_end,
                                 q_squeeze_L, q_squeeze_L, LIFT_DURATION, args.dry_run)

        if not args.dry_run:
            for _ in range(max(1, int(HOLD_SECONDS * RATE_HZ))):
                _publish(lever, qL, qR)
                time.sleep(1.0 / RATE_HZ)
```

Levée : réduit d'abord les gains des 10 joints de bras
(`LEVEE_STIFFNESS`/`LEVEE_DAMPING`), puis `cartesian_ramp` verticale pure
(X/Y fixes, seul Z change), ancre FIXE sur `q_squeeze_L`. Maintien
ensuite -- avec `HOLD_SECONDS=0.0`, cette boucle ne fait qu'un seul tour
(`max(1, int(0.0 * 65)) = 1`).

```python
    if not args.dry_run and args.only_phase in (None, "levee"):
        if run_levee and walk_stance_scale > 0:
            _checkpoint("redressement genoux -- avant relachement", confirm)
            _straighten_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE,
                               WALK_STANCE_DURATION, dry_run=args.dry_run)
        _checkpoint("release() -- rend la main au runner pd_stand actif", confirm)
        if run_levee:
            n = max(1, int(RELEASE_RAMP_SECONDS * RATE_HZ))
            for i in range(n + 1):
                lever.set_weight(1.0 - i / n)
                _publish(lever, qL, qR)
                time.sleep(1.0 / RATE_HZ)
        lever.release()

    print("[INFO] Sequence terminee.")
```

Relâchement final -- seulement si on est allé jusqu'à `only_phase in
(None, "levee")` (donc jamais après un `--only-phase approche` ou
`serrage` isolé, qui laissent les bras tenus pour un appel suivant).
Redresse d'abord les genoux si besoin, puis rampe le poids à 0 avant
`release()` (identique au motif déjà vu dans `lift.py` côté simulation).
Utilise `RELEASE_RAMP_SECONDS=4.0` -- PAS `--release-ramp-seconds` (cet
argument n'existe même pas dans le CLI de ce fichier, contrairement à
`levee_pivot.py`).

### `main` (lignes 270-297)

```python
def main():
    args = _build_arg_parser().parse_args()
    node = lever = None
    if not args.dry_run:
        rclpy.init()
        node = rclpy.create_node("levee")
        if not args.skip_motion_state:
            ok = ensure_motion_state(node, "lower_body_balance", timeout=MOTION_STATE_TIMEOUT)
            if not ok:
                print("[ERREUR] impossible de passer en lower_body_balance.", flush=True)
                node.destroy_node()
                rclpy.shutdown()
                return
        lever = Lever(node)
    try:
        run_lift_sequence(node, lever, args)
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()
```

En `--dry-run`, `node`/`lever` restent `None` -- ROS2 n'est jamais
initialisé du tout, `run_lift_sequence` doit donc gérer ce cas (tous les
appels `lever.xxx` sont protégés par `if not args.dry_run` dans le corps
de la fonction). `finally` garantit le nettoyage même en cas d'exception
ou de Ctrl+C.

---

## 5. `pivot_real.py` (193 lignes) -- le pivot seul, en standalone

Sert deux usages : bibliothèque de constantes pour `levee_pivot.py`
(`WAIST_JOINT_INDEX`, `WAIST_KP`, `WAIST_KD`, `_ease` renommée
`_pivot_ease`), ET un script autonome complet.

### Constantes (lignes 14-33)

```python
WAIST_JOINT_INDEX = 12
WAIST_KP = 150.0
WAIST_KD = 3.0
RATE_HZ = 30
MOTION_STATE_TIMEOUT = 3.0

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
WALK_STANCE_STIFFNESS_SCALE = 1.8
WALK_STANCE_DURATION = 3.0
```

**Différence notable avec `levee.py`/`levee_pivot.py`** : `RATE_HZ = 30`
ici, alors que `levee.py` utilise `RATE_HZ = 65`. Ce fichier standalone
n'a jamais été aligné sur le fix de fréquence de contrôle appliqué à la
séquence bras (voir mémoire projet), puisqu'il ne bouge que le buste
(charge différente, pas de bras sous charge).

### `_checkpoint`/`_ease`/`_quintic_ease` (lignes 36-49)

```python
def _checkpoint(message, confirm):
    print(f"[ETAPE] {message}", flush=True)
    if confirm:
        input("        Verifie le robot, puis Entree pour continuer (Ctrl+C pour arreter)... ")

def _ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)

def _quintic_ease(t):
    t = max(0.0, min(1.0, t))
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)
```

`_checkpoint` identique à `levee.py`. `_ease` (smoothstep) : une copie
locale de la fonction `ease` de `lift_carton.py`, sous un autre nom --
c'est CETTE fonction (importée sous l'alias `_pivot_ease`) que
`levee_pivot.py` utilise pour la courbe de vitesse du pivot. `_quintic_ease` :
identique à celle de `levee.py`, dupliquée ici pour la flexion des genoux
de ce script standalone.

### `_bend_knees`/`_straighten_knees` (lignes 52-87)

```python
def _bend_knees(lever, scale, stiffness_scale, duration, dry_run=False):
    if dry_run:
        print(f"    [dry-run] flexion genoux -- scale={scale} duration={duration}s")
        return
    for idx, kp, kd in [
        (LEFT_HIP_PITCH_INDEX, 200.0, 5.0), (RIGHT_HIP_PITCH_INDEX, 200.0, 5.0),
        (LEFT_KNEE_PITCH_INDEX, 450.0, 5.0), (RIGHT_KNEE_PITCH_INDEX, 450.0, 5.0),
        (LEFT_ANKLE_PITCH_INDEX, 400.0, 2.0), (RIGHT_ANKLE_PITCH_INDEX, 400.0, 2.0),
    ]:
        lever.set_gains(idx, kp * stiffness_scale, kd * stiffness_scale)
    n = max(1, int(duration * RATE_HZ))
    for i in range(n + 1):
        a = _quintic_ease(i / n)
        lever[LEFT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_L)
        lever[RIGHT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_R)
        lever[LEFT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_L)
        lever[RIGHT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_R)
        lever[LEFT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_L)
        lever[RIGHT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_R)
        time.sleep(1.0 / RATE_HZ)
```

Copie quasi identique de `levee.py::_bend_knees` (même structure, mêmes
gains, même interpolation quintique) -- seule différence : `RATE_HZ` fixe
(30, pas de paramètre `rate_hz`), et pas de `set_batch` ici non plus
(6 `lever[idx]=` séparés par pas). `_straighten_knees` (lignes 74-87) :
inverse exact, même relation que dans `levee.py`.

### `run_pivot` (lignes 90-147) -- la séquence complète autonome

```python
def run_pivot(lever, angle_deg, pivot_duration, hold_seconds, release_ramp_seconds,
              walk_stance_scale, dry_run, confirm):
    angle_target = np.radians(angle_deg)
    n = max(1, int(pivot_duration * RATE_HZ))

    if walk_stance_scale > 0:
        _checkpoint(f"flexion genoux -- scale={walk_stance_scale}, {WALK_STANCE_DURATION:.1f}s", confirm)
        _bend_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE, WALK_STANCE_DURATION,
                    dry_run=dry_run)
        if not dry_run:
            time.sleep(2.0)

    if not dry_run:
        lever.set_gains(WAIST_JOINT_INDEX, WAIST_KP, WAIST_KD)
```

Flexion des genoux optionnelle (avec une pause fixe de 2s après, cette
fois RÉELLEMENT atteignable si `--walk-stance-scale` > 0 est passé à ce
script -- contrairement à la même pause dans `levee_pivot.py` qui, elle,
n'est jamais utilisée car ce fichier-ci gère son propre pivot
indépendamment). Puis règle les gains du buste AVANT de commencer à
bouger.

```python
    _checkpoint(f"pivot buste -- 0 -> {angle_deg:.0f}deg ({pivot_duration:.1f}s)", confirm)
    for i in range(n + 1):
        a = _ease(i / n)
        target = a * angle_target
        if dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i / RATE_HZ:.2f}s  waist={np.degrees(target):.1f}deg")
            continue
        lever[WAIST_JOINT_INDEX] = float(target)
        time.sleep(1.0 / RATE_HZ)

    _checkpoint(f"maintien pivote -- {hold_seconds:.1f}s", confirm)
    if not dry_run:
        for _ in range(max(1, int(hold_seconds * RATE_HZ))):
            lever[WAIST_JOINT_INDEX] = float(angle_target)
            time.sleep(1.0 / RATE_HZ)

    _checkpoint(f"depivot -- {angle_deg:.0f}deg -> 0deg ({pivot_duration:.1f}s)", confirm)
    for i in range(n + 1):
        a = _ease(i / n)
        target = (1.0 - a) * angle_target
        if dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i / RATE_HZ:.2f}s  waist={np.degrees(target):.1f}deg")
            continue
        lever[WAIST_JOINT_INDEX] = float(target)
        time.sleep(1.0 / RATE_HZ)
```

Pivot pur (0 -> `angle_target`), maintien, dépivot (`angle_target` -> 0) --
3 boucles distinctes, chacune un simple `lever[WAIST_JOINT_INDEX] = ...`
(un seul joint, pas de `set_batch` nécessaire puisqu'aucun bras n'est
tenu par ce script). Contrairement à `levee_pivot.py`, il n'y a **aucun
carton tenu** ici -- ce script teste uniquement la mécanique de rotation
du buste, seul.

```python
    if walk_stance_scale > 0:
        _checkpoint("redressement genoux -- avant relachement", confirm)
        _straighten_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE,
                           WALK_STANCE_DURATION, dry_run=dry_run)

    if not dry_run:
        _checkpoint(f"relachement -- rampe {release_ramp_seconds:.1f}s", confirm)
        n2 = max(1, int(release_ramp_seconds * RATE_HZ))
        for i in range(n2 + 1):
            lever.set_weight(1.0 - i / n2)
            lever[WAIST_JOINT_INDEX] = 0.0
            time.sleep(1.0 / RATE_HZ)
        lever.release()

    print("[INFO] Pivot termine.")
```

Redressement des genoux si besoin, puis rampe de poids classique avant
`release()` -- `lever[WAIST_JOINT_INDEX] = 0.0` republié à chaque pas de
la rampe (le buste doit rester à 0° pendant tout le relâchement, pas
suivre une nouvelle valeur).

### `_build_arg_parser` (lignes 150-160)

```python
def _build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--angle-deg", type=float, default=30.0)
    parser.add_argument("--pivot-duration", type=float, default=6.0)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--release-ramp-seconds", type=float, default=1.5)
    parser.add_argument("--walk-stance-scale", type=float, default=1.0)
    parser.add_argument("--skip-motion-state", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-confirm", action="store_true")
    return parser
```

**Défauts très différents de `levee_pivot.py`** : `pivot_duration=6.0`
(bien plus lent, pas encore accéléré comme la séquence bras),
`walk_stance_scale=1.0` par défaut (flexion des genoux ACTIVE ici, alors
qu'elle est à 0.0 dans `levee.py`/`levee_pivot.py`) -- cohérent avec le
fait que ce script teste le pivot SEUL, sans bras tendus vers l'avant qui
changeraient l'équilibre, donc les concepteurs ont gardé la prudence des
jambes fléchies par défaut ici.

### `main` (lignes 163-192)

```python
def main():
    args = _build_arg_parser().parse_args()
    confirm = not args.no_confirm
    node = lever = None
    if not args.dry_run:
        rclpy.init()
        node = rclpy.create_node("pivot_real")
        if not args.skip_motion_state:
            ok = ensure_motion_state(node, "lower_body_balance", timeout=MOTION_STATE_TIMEOUT)
            if not ok:
                print("[ERREUR] impossible de passer en lower_body_balance.", flush=True)
                node.destroy_node()
                rclpy.shutdown()
                return
        lever = Lever(node)
    try:
        run_pivot(lever, args.angle_deg, args.pivot_duration, args.hold_seconds,
                  args.release_ramp_seconds, args.walk_stance_scale, args.dry_run, confirm)
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()
```

Structure identique à `levee.py::main()` (même schéma init/vérif
état/créer Lever/exécuter/nettoyer) -- node ROS2 nommé `"pivot_real"`.

---

## 6. `levee_pivot.py` (287 lignes) -- l'orchestrateur final

Importe TOUT le reste (`lever.py`, `motion_state.py`, `levee.py`,
`pivot_real.py`, `lift_carton.py`) et enchaîne les étapes dans
`run_lift_and_pivot`, une seule fonction longue plutôt que plusieurs
petites (contrairement à `levee.py`/`pivot_real.py` qui restent
utilisables séparément).

### Imports et helper (lignes 1-33)

```python
"""Prise du carton reel, pivot du buste, puis depose -- un seul process/Lever partage."""
import argparse
import sys
import os
import time

import numpy as np
import rclpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lever import Lever
from motion_state import ensure_motion_state
from levee import (
    APPROACH_LIFT_DURATION,
    LEFT_CHAIN, RIGHT_CHAIN, HAND_OFFSET_LEFT, HAND_OFFSET_RIGHT,
    LEFT_JOINT_INDICES, RIGHT_JOINT_INDICES, Q_LEFT_HOME, Q_RIGHT_HOME,
    WAYPOINT_Q_LEFT, WAYPOINT_Q_RIGHT, WAYPOINT_DURATION, WRIST_CHAIN_INDEX,
    PINCH_X, PINCH_Y, SQUEEZE_Y, LIFT_Z, APPROACH_DURATION, SQUEEZE_DURATION,
    LIFT_DURATION, HOLD_SECONDS, RATE_HZ, WALK_STANCE_SCALE,
    WALK_STANCE_STIFFNESS_SCALE, WALK_STANCE_DURATION, LEVEE_STIFFNESS, LEVEE_DAMPING,
    MOTION_STATE_TIMEOUT, ELBOW_YAW_ROTATION_DEG,
    _checkpoint, _rotate_xy, solve_arm_ik, _bend_knees, _straighten_knees,
    _publish, move_arms, ease, forward_kinematics, cartesian_ramp,
)
from pivot_real import WAIST_JOINT_INDEX, WAIST_KP, WAIST_KD, _ease as _pivot_ease

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot_arm_ik"))
from lift_carton import mirror_left_to_right


def _publish_with_waist(lever, qL, qR, waist):
    lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES + [WAIST_JOINT_INDEX],
                     list(qL) + list(qR) + [waist])
```

Réimporte QUASIMENT TOUT ce que `levee.py` exporte (la liste d'import est
longue -- ce fichier réutilise la bibliothèque complète plutôt que d'en
dupliquer le code). `_publish_with_waist` : la seule fonction propre à ce
fichier avant `run_lift_and_pivot` -- étend `set_batch` à bras+buste en un
seul message, pour les étapes où le buste bouge en même temps que les bras.

### Setup (lignes 36-52)

```python
def run_lift_and_pivot(node, lever, args):
    confirm = not args.no_confirm
    wrist_rotation = np.radians(args.wrist_rotation_deg)
    walk_stance_scale = args.walk_stance_scale

    straight_pref_L = WAYPOINT_Q_LEFT.copy()
    straight_pref_L[3] = 0.0
    q_pinch_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                              _rotate_xy([args.pinch_x, PINCH_Y, args.pinch_z], args.pinch_yaw_offset),
                              WAYPOINT_Q_LEFT, lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                              null_space_pref=straight_pref_L)
    q_pinch_R = mirror_left_to_right(q_pinch_L)
    squeeze_L = _rotate_xy([args.pinch_x, SQUEEZE_Y, args.pinch_z], args.pinch_yaw_offset)
    q_squeeze_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_pinch_L,
                                lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                                null_space_pref=q_pinch_L)
    q_squeeze_R = mirror_left_to_right(q_squeeze_L)
```

**Différence avec `levee.py::run_lift_sequence`** : ici `q_pinch_L` est
seedé/ancré sur `WAYPOINT_Q_LEFT` (le point de passage), PAS `Q_LEFT_HOME`
comme dans `levee.py` -- cohérent avec le fait que dans ce fichier, le
robot passe TOUJOURS par le point de passage avant de viser (voir plus
bas), donc ancrer directement sur cette posture évite un petit écart
inutile.

### Flags et flexion des genoux (lignes 54-67)

```python
    run_approche = args.only_phase in (None, "approche")
    run_serrage = args.only_phase in (None, "serrage")
    run_levee = args.only_phase in (None, "levee")
    run_pivot = args.only_phase in (None, "pivot")
    run_depose = args.only_phase is None

    qL, qR = Q_LEFT_HOME, Q_RIGHT_HOME

    if run_approche and walk_stance_scale > 0:
        _checkpoint(f"flexion genoux -- scale={walk_stance_scale}, {WALK_STANCE_DURATION:.1f}s", confirm)
        _bend_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE, WALK_STANCE_DURATION,
                    dry_run=args.dry_run)
        if not args.dry_run:
            time.sleep(2.0)
```

5 flags (un de plus que `levee.py` : `run_pivot`, et `run_depose` qui
n'existe QUE si `only_phase` est totalement vide -- pas de "only depose"
possible séparément, la dépose fait toujours partie de la séquence
complète). La pause fixe de 2.0s après flexion des genoux existe ici
aussi mais reste inatteignable par défaut (`walk_stance_scale=0.0`).

### Approche (lignes 69-85)

```python
    if run_approche:
        _checkpoint(f"point de passage -- coudes vers l'arriere, {WAYPOINT_DURATION:.1f}s", confirm)
        qL, qR = move_arms(lever, Q_LEFT_HOME, WAYPOINT_Q_LEFT, Q_RIGHT_HOME, WAYPOINT_Q_RIGHT,
                            WAYPOINT_DURATION, dry_run=args.dry_run)

        pinch_target_L = _rotate_xy([args.pinch_x, PINCH_Y, args.pinch_z], args.pinch_yaw_offset)
        waypoint_hand_L = forward_kinematics(LEFT_CHAIN, HAND_OFFSET_LEFT, WAYPOINT_Q_LEFT)
        raised_point_L = np.array([waypoint_hand_L[0], waypoint_hand_L[1], pinch_target_L[2]])

        _checkpoint(f"approche -- ajustement vertical, {APPROACH_LIFT_DURATION:.1f}s", confirm)
        qL, qR = cartesian_ramp(lever, WAYPOINT_Q_LEFT, wrist_rotation, waypoint_hand_L,
                                 raised_point_L, WAYPOINT_Q_LEFT, WAYPOINT_Q_LEFT,
                                 APPROACH_LIFT_DURATION, args.dry_run)

        _checkpoint(f"approche -- ligne horizontale vers pinch, {APPROACH_DURATION:.1f}s", confirm)
        qL, qR = cartesian_ramp(lever, qL, wrist_rotation, raised_point_L, pinch_target_L,
                                 WAYPOINT_Q_LEFT, straight_pref_L, APPROACH_DURATION, args.dry_run)
```

**Rigoureusement identique** à `levee.py::run_lift_sequence`, phase
approche (même 3 étapes : point de passage, ajustement vertical, ligne
horizontale) -- seul le message du 2e `_checkpoint` est raccourci ici
(pas le détail x/y/z/poignet affiché par `levee.py`).

### Serrage et levée (lignes 87-106)

```python
    if run_serrage:
        _checkpoint(f"serrage -- Y +-{PINCH_Y} -> +-{SQUEEZE_Y}, {SQUEEZE_DURATION:.1f}s "
                    "(LE CONTACT AVEC LE CARTON COMMENCE ICI)", confirm)
        qL, qR = move_arms(lever, q_pinch_L, q_squeeze_L, q_pinch_R, q_squeeze_R,
                            SQUEEZE_DURATION, dry_run=args.dry_run)

    if run_levee:
        _checkpoint(f"levee -- Z {args.pinch_z} -> {LIFT_Z}, {LIFT_DURATION:.1f}s puis maintien "
                    f"{HOLD_SECONDS:.1f}s", confirm)
        if not args.dry_run:
            for idx in LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES:
                lever.set_gains(idx, stiffness=LEVEE_STIFFNESS, damping=LEVEE_DAMPING)
        lift_start = _rotate_xy([args.pinch_x, SQUEEZE_Y, args.pinch_z], args.pinch_yaw_offset)
        lift_end = _rotate_xy([args.pinch_x, SQUEEZE_Y, LIFT_Z], args.pinch_yaw_offset)
        qL, qR = cartesian_ramp(lever, q_squeeze_L, wrist_rotation, lift_start, lift_end,
                                 q_squeeze_L, q_squeeze_L, LIFT_DURATION, args.dry_run)
        if not args.dry_run:
            for _ in range(max(1, int(HOLD_SECONDS * RATE_HZ))):
                _publish(lever, qL, qR)
                time.sleep(1.0 / RATE_HZ)
```

Également identique à `levee.py` pour ces 2 phases -- **mais SANS le
redressement des genoux et le relâchement final** qui suivaient la levée
dans `levee.py::run_lift_sequence` (logique : ici la séquence continue
avec le pivot, il ne faut PAS relâcher le carton après la levée).

### Guard : pas de pivot demandé (lignes 108-110)

```python
    if not run_pivot:
        print("[INFO] only_phase termine -- carton tenu, pas de pivot dans cet appel.")
        return
```

### Rapproché + Pivot + Extension fusionnés (lignes 112-159)

```python
    retract_start = _rotate_xy([args.pinch_x, SQUEEZE_Y, LIFT_Z], args.pinch_yaw_offset)
    retract_end = _rotate_xy([args.retract_x, SQUEEZE_Y, LIFT_Z], args.pinch_yaw_offset)
    depose_target = _rotate_xy([args.depose_x, SQUEEZE_Y, LIFT_Z], args.pinch_yaw_offset)

    angle_target = np.radians(args.angle_deg)
    n = max(1, int(args.pivot_duration * RATE_HZ))
    if args.retract_duration + args.extend_duration > args.pivot_duration:
        print(f"[ERREUR] --retract-duration ({args.retract_duration}) + --extend-duration "
              f"({args.extend_duration}) depasse --pivot-duration ({args.pivot_duration}) -- "
              "les deux se chevaucheraient, arret.", flush=True)
        return

    if not args.dry_run:
        lever.set_gains(WAIST_JOINT_INDEX, WAIST_KP, WAIST_KD)

    anchor = qL.copy()
    _checkpoint(
        f"pivot buste + rapproche/tend les bras -- 0 -> {args.angle_deg:.0f}deg "
        f"({args.pivot_duration:.1f}s, mouvement simultane)", confirm)
    for i in range(n + 1):
        t = i / RATE_HZ
        waist = _pivot_ease(i / n) * angle_target
        if t < args.retract_duration and args.retract_duration > 0:
            a_arm = ease(t / args.retract_duration)
            target = retract_start + a_arm * (retract_end - retract_start)
        elif t > args.pivot_duration - args.extend_duration and args.extend_duration > 0:
            a_arm = ease((t - (args.pivot_duration - args.extend_duration)) / args.extend_duration)
            target = retract_end + a_arm * (depose_target - retract_end)
        else:
            target = retract_end
        qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, target, qL,
                           lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation, iters=10,
                           null_space_pref=anchor)
        qR = mirror_left_to_right(qL)
        if args.dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={t:.2f}s  waist={np.degrees(waist):.1f}deg  x={target[0]:.3f}")
            continue
        _publish_with_waist(lever, qL, qR, waist)
        time.sleep(1.0 / RATE_HZ)
```

3 points X précalculés (`retract_start`=position de prise, `retract_end`
=rapprochée du corps, `depose_target`=tendue pour déposer), Y/Z toujours
`SQUEEZE_Y`/`LIFT_Z`. Garde de sécurité si les 2 fenêtres (retract+extend)
dépasseraient la durée totale du pivot (chevauchement impossible à gérer
correctement). `anchor = qL.copy()` -- capturé UNE FOIS avant la boucle,
jamais réassigné dedans (règle anti-dérive). Dans la boucle : `waist`
suit toujours `_pivot_ease` sur toute la durée ; `target` (position de la
main) ne bouge que pendant les `retract_duration` premières secondes
(`t < args.retract_duration`) OU les `extend_duration` dernières
(`t > pivot_duration - extend_duration`) -- entre les deux, `target`
reste figé à `retract_end` (le buste tourne seul). `solve_arm_ik` avec
seulement 10 itérations (le seed `qL` est déjà proche de la solution du
pas précédent).

```python
    _checkpoint(f"maintien pivote -- {args.hold_seconds:.1f}s", confirm)
    if not args.dry_run:
        for _ in range(max(1, int(args.hold_seconds * RATE_HZ))):
            _publish_with_waist(lever, qL, qR, angle_target)
            time.sleep(1.0 / RATE_HZ)
```

Maintien : republie en continu bras+buste (`angle_target` figé) pendant
`hold_seconds`.

### Guard : pas de dépose demandée (lignes 167-169)

```python
    if not run_depose:
        print("[INFO] only_phase termine -- carton tenu, buste toujours tourne, pas de depose.")
        return
```

### Baisse avant relâchement (lignes 171-178)

```python
    drop_z = LIFT_Z - args.pre_release_drop
    if args.pre_release_drop > 0:
        _checkpoint(f"baisse avant relachement -- Z {LIFT_Z} -> {drop_z:.3f} "
                    f"({args.pre_release_drop_duration:.1f}s), buste encore tourne", confirm)
        drop_start = _rotate_xy([args.depose_x, SQUEEZE_Y, LIFT_Z], args.pinch_yaw_offset)
        drop_end = _rotate_xy([args.depose_x, SQUEEZE_Y, drop_z], args.pinch_yaw_offset)
        qL, qR = cartesian_ramp(lever, qL, wrist_rotation, drop_start, drop_end, qL, qL,
                                 args.pre_release_drop_duration, args.dry_run)
```

`drop_z` = `LIFT_Z` moins une petite baisse (3cm par défaut). Si cette
baisse est demandée (`> 0`), rampe cartésienne verticale pure, ancre
FIXE (`qL`, `qL` -- les deux mêmes, donc PAS de glissement ici, à la
différence de `cartesian_ramp` utilisée en section approche où l'ancre
glisse).

### Désserrage (lignes 180-186)

```python
    _checkpoint(f"desserrage -- Y +-{SQUEEZE_Y} -> +-{PINCH_Y} ({args.open_duration:.1f}s) "
                "-- LE CARTON EST RELACHE ICI, buste encore tourne", confirm)
    q_open_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                             _rotate_xy([args.depose_x, PINCH_Y, drop_z], args.pinch_yaw_offset),
                             qL, lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation)
    q_open_R = mirror_left_to_right(q_open_L)
    qL, qR = move_arms(lever, qL, q_open_L, qR, q_open_R, args.open_duration, dry_run=args.dry_run)
```

**LE CARTON EST RELÂCHÉ ICI** (au sens "la prise s'ouvre" -- Y repasse de
`SQUEEZE_Y` à `PINCH_Y`) : un seul appel `solve_arm_ik` (pas de boucle,
pas de `null_space_pref` explicite -- défaut = `qL`) pour calculer la
cible ouverte, puis `move_arms` articulaire pour y aller.

### Écartement (lignes 188-196)

```python
    _checkpoint(f"ecartement -- Y +-{PINCH_Y} -> +-{PINCH_Y + args.ecartement_gap_y:.3f} "
                f"({args.ecartement_duration:.1f}s)", confirm)
    ecart_L = _rotate_xy([args.depose_x, PINCH_Y + args.ecartement_gap_y, drop_z], args.pinch_yaw_offset)
    q_ecart_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, ecart_L, qL,
                              lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                              null_space_pref=qL)
    q_ecart_R = mirror_left_to_right(q_ecart_L)
    qL, qR = move_arms(lever, qL, q_ecart_L, qR, q_ecart_R, args.ecartement_duration,
                        dry_run=args.dry_run)
```

Écarte encore un peu plus (`ecartement_gap_y`, 2.5cm par défaut) pour
dégager de la surface du carton avant de retirer la main -- ici
`null_space_pref=qL` EST précisé explicitement (identique au défaut, mais
écrit pour la lisibilité/cohérence du code).

### Translation arrière (lignes 198-203)

```python
    _checkpoint(f"translation arriere -- X {args.depose_x} -> {args.retreat_back_x} "
                f"({args.retreat_back_duration:.1f}s), Y reste large, main ramenee pres du corps",
                confirm)
    retreat_end = np.array([args.retreat_back_x, ecart_L[1], ecart_L[2]])
    qL, qR = cartesian_ramp(lever, qL, wrist_rotation, ecart_L, retreat_end, qL, qL,
                             args.retreat_back_duration, args.dry_run)
```

Rampe cartésienne EN X SEUL (Y/Z inchangés, `ecart_L[1]`/`ecart_L[2]`
réutilisés tels quels) -- ramène la main près du corps AVANT le saut vers
le point de passage suivant, pour que ce saut articulaire soit court/sûr.

### Dégagement, dépivot, retour des bras (lignes 205-227)

```python
    _checkpoint(f"degagement -- coudes vers l'arriere ({args.degagement_waypoint_duration:.1f}s)",
                confirm)
    current_hand_L = forward_kinematics(LEFT_CHAIN, HAND_OFFSET_LEFT, qL)
    waypoint_hand_L = forward_kinematics(LEFT_CHAIN, HAND_OFFSET_LEFT, WAYPOINT_Q_LEFT)
    qL, qR = cartesian_ramp(lever, qL, wrist_rotation, current_hand_L, waypoint_hand_L,
                             qL, qL, args.degagement_waypoint_duration, args.dry_run)

    _checkpoint(f"depivot -- {args.angle_deg:.0f}deg -> 0deg ({args.pivot_duration:.1f}s), "
                "CARTON DEJA LACHE, avant le retour des bras", confirm)
    for i in range(n + 1):
        a = _pivot_ease(i / n)
        waist = (1.0 - a) * angle_target
        if args.dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i / RATE_HZ:.2f}s  waist={np.degrees(waist):.1f}deg")
            continue
        lever[WAIST_JOINT_INDEX] = float(waist)
        time.sleep(1.0 / RATE_HZ)

    _checkpoint(f"retour bras le long du corps ({args.retreat_duration:.1f}s), "
                "buste deja droit", confirm)
    qL, qR = move_arms(lever, qL, Q_LEFT_HOME, qR, Q_RIGHT_HOME, args.retreat_duration,
                        dry_run=args.dry_run)
```

**Dégagement** : `cartesian_ramp` de la main courante vers la position
qu'AURAIT la main dans la posture `WAYPOINT_Q_LEFT` (calculée par
`forward_kinematics`, pas en interpolant les angles directement) -- garantit
un chemin de main en ligne droite jusqu'à ce point, ancre fixe. **Dépivot** :
boucle manuelle sur `WAIST_JOINT_INDEX` seul (`lever[WAIST_JOINT_INDEX] =
...`, pas `set_batch` -- les bras ne bougent pas pendant cette boucle,
donc pas besoin de les republier ici). Réutilise le même `n` (nombre de
pas) que la boucle de pivot plus haut, mais parcourt `angle_target -> 0`
au lieu de `0 -> angle_target`. **Retour des bras** : `move_arms`
articulaire vers `Q_LEFT_HOME`/`Q_RIGHT_HOME` -- placé APRÈS le dépivot
(fix du 2026-09-16/17, voir mémoire projet : l'ordre inverse créait un
déséquilibre statique, masse des bras décalée pendant que le buste était
encore tourné).

### Redressement des genoux et relâchement final (lignes 229-241)

```python
    if not args.dry_run and walk_stance_scale > 0:
        _checkpoint("redressement genoux -- avant relachement final", confirm)
        _straighten_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE,
                           WALK_STANCE_DURATION, dry_run=args.dry_run)

    if not args.dry_run:
        _checkpoint(f"relachement final -- rampe {args.release_ramp_seconds:.1f}s", confirm)
        n2 = max(1, int(args.release_ramp_seconds * RATE_HZ))
        for i in range(n2 + 1):
            lever.set_weight(1.0 - i / n2)
            _publish_with_waist(lever, qL, qR, 0.0)
            time.sleep(1.0 / RATE_HZ)
        lever.release()

    print("[INFO] Sequence terminee.")
```

Redressement optionnel (comme partout ailleurs, sauté par défaut). Rampe
de poids finale : republie bras+buste (`0.0` fixe pour le buste, déjà
dépivoté) à CHAQUE pas via `_publish_with_waist`, pendant que `weight`
descend de 1.0 à 0.0 -- puis `lever.release()`.

### `_build_arg_parser` (lignes 246-274) -- toutes les options CLI

Voir le tableau complet dans le [README principal](../README.md#toutes-les-options)
(23 options, avec valeur par défaut et effet de chacune) -- non reproduit
ici pour éviter la duplication, mais RIEN n'est omis côté README.

### `main` (lignes 278-305)

```python
def main():
    args = _build_arg_parser().parse_args()
    node = lever = None
    if not args.dry_run:
        rclpy.init()
        node = rclpy.create_node("levee_pivot")
        if not args.skip_motion_state:
            ok = ensure_motion_state(node, "lower_body_balance", timeout=MOTION_STATE_TIMEOUT)
            if not ok:
                print("[ERREUR] impossible de passer en lower_body_balance -- arret.", flush=True)
                node.destroy_node()
                rclpy.shutdown()
                return
        lever = Lever(node)
    try:
        run_lift_and_pivot(node, lever, args)
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()
```

Structure identique à `levee.py`/`pivot_real.py::main()` -- node ROS2
nommé `"levee_pivot"`.
