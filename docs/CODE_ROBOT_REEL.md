# Le code du robot réel, bloc par bloc

Ce document explique **tout** le code de `package_sequence_bras_reel/` --
les 6 fichiers qui font marcher la séquence "prise du carton -> pivot ->
dépose" sur le PM01 physique. Ordre de lecture pensé pour être pédagogique
(les fondations d'abord, l'orchestrateur en dernier), pas l'ordre
alphabétique.

Pour l'architecture générale (pourquoi un seul process, comment ça
communique avec le robot), voir le [README principal](../README.md).

---

## 1. `lift_carton.py` -- la géométrie et la cinématique inverse

C'est la base mathématique : "si je veux que la main soit à TEL endroit,
quels angles donner aux 5 articulations du bras ?" Ce fichier ne connaît
rien à ROS2, aux topics, ni au reste de la séquence -- juste des maths.

### Les chaînes cinématiques (lignes 3-19)

```python
LEFT_CHAIN = [
    ("J13_SHOULDER_PITCH_L", np.array([0, 1, 0]), np.array([-0.027105, 0.12916, 0.21549])),
    ...
]
HAND_OFFSET_LEFT = np.array([0.03, -0.02, -0.14])
```

Chaque ligne décrit **une articulation** : son nom, l'**axe** autour duquel
elle tourne (`[0,1,0]` = autour de Y, etc.), et le **vecteur de décalage**
depuis l'articulation précédente (en mètres, dans le repère de
l'articulation parente). `HAND_OFFSET_LEFT` est le dernier décalage, de la
5ᵉ articulation (poignet) jusqu'au centre de la main. `RIGHT_CHAIN` est la
même chose côté droit -- les valeurs Y sont inversées (le bras droit est le
symétrique miroir du gauche par rapport au plan central du robot).

### `rotation_matrix` (lignes 22-25)

Formule de **Rodrigues** : convertit un axe + un angle en matrice de
rotation 3x3. Utilisée pour composer les rotations d'une articulation à la
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
avec la rotation de CETTE articulation. À la fin, ajoute le décalage vers
la main. C'est le calcul **direct** (angles connus -> position de la
main) -- le sens facile.

### `numerical_jacobian` (lignes 37-44) -- "si je bouge un peu chaque angle, comment bouge la main ?"

Pour chaque articulation `i`, bouge son angle d'un tout petit peu
(`eps=1e-6` radian), recalcule la position de la main, et la différence
(divisée par `eps`) donne la **sensibilité** de la position de la main à
cette articulation. Empilées, ces 5 sensibilités (3 lignes X/Y/Z, 5
colonnes = 5 articulations) forment la matrice **Jacobienne**.

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
`q_init`, mesure l'écart entre la main actuelle et la cible, utilise la
Jacobienne pour savoir dans quelle direction bouger les angles pour
réduire cet écart, répète jusqu'à convergence (ou `iters` fois maximum).
Le terme `damping` évite les mouvements erratiques quand la Jacobienne est
mal conditionnée (proche d'une singularité, bras presque tendu). Cette
fonction a **5 inconnues (les 5 angles) pour seulement 3 équations
(X,Y,Z)** -- il existe donc en général une infinité de solutions, et
celle trouvée dépend de `q_init` (le point de départ).

### `solve_arm_ik` (lignes 59-90) -- la vraie fonction utilisée partout, avec verrou + préférence

C'est **cette** fonction qui est appelée dans tout le reste du code (pas
`solve_ik` directement), avec deux ajouts :

- **`lock_index`/`lock_angle`** : fige UNE articulation à un angle fixe
  (par exemple le poignet, `WRIST_CHAIN_INDEX=4`) et ne laisse l'IK
  résoudre que sur les 4 autres. Utile pour forcer une forme de bras
  précise (poignet dans une orientation donnée) plutôt que de laisser le
  solveur choisir au hasard.
- **`null_space_pref`** : avec 1 articulation verrouillée, il reste encore
  **1 degré de liberté redondant** parmi les 4 articulations libres (3
  équations, 4 inconnues). Sans régularisation, le solveur peut converger
  vers une posture bizarre qui atteint quand même la cible (bras qui
  semble tordu). Ce paramètre ajoute un rappel doux vers `null_space_pref`
  **dans le noyau du Jacobien** (ligne 85 : `null_proj`) -- un mouvement
  qui ne change PAS la position de la main, mais réoriente le coude/épaule
  vers la préférence donnée.

```python
free_idx = [i for i in range(len(q_init)) if i != lock_index]
...
q[lock_index] = lock_angle
...
    step = J_pinv @ error
    if null_space_gain:
        ...
        null_proj = np.eye(n_free) - J_pinv @ J
        step = step + null_space_gain * (null_proj @ (q_pref_free - q_free))
    for k, i in enumerate(free_idx):
        q[i] += step[k]
    q[lock_index] = lock_angle
```

À chaque itération : calcule le pas normal vers la cible (`step`), puis
lui AJOUTE un petit poussée (`null_space_gain=0.2` par défaut) vers
`null_space_pref`, projetée pour ne pas perturber la position déjà
atteinte. L'articulation verrouillée est réécrasée à `lock_angle` à
chaque tour, pour rester garantie fixe.

**Point critique (déjà découvert et corrigé plusieurs fois dans ce
projet) :** si `null_space_pref` n'est PAS fourni, il vaut `q_init` par
défaut (ligne 65). Si on appelle cette fonction plusieurs fois de suite en
réutilisant `q` comme SEED **et** comme préférence à chaque fois (au lieu
d'une valeur fixe), la préférence "glisse" avec le mouvement au lieu de
rester ancrée -- ça a causé des rotations d'avant-bras visibles à
plusieurs endroits du projet, toujours corrigées en figeant l'ancre AVANT
la boucle.

### `mirror_left_to_right` (lignes 93-98)

```python
def mirror_left_to_right(q_left):
    q_right = q_left.copy()
    q_right[1] *= -1
    q_right[2] *= -1
    q_right[4] *= -1
    return q_right
```

Le bras droit n'a **jamais** sa propre résolution IK dans ce projet --
il est toujours dérivé du gauche par symétrie miroir. Les indices 1
(SHOULDER_ROLL), 2 (SHOULDER_YAW) et 4 (ELBOW_YAW) sont inversés ; 0
(SHOULDER_PITCH) et 3 (ELBOW_PITCH) restent identiques -- c'est la
convention géométrique de ce squelette de bras (les rotations "pitch"
sont symétriques par nature, les rotations autour d'un axe vertical/
horizontal-latéral s'inversent en miroir).

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

## 2. `lever.py` -- la classe qui parle au robot

Encapsule TOUTE la communication ROS2 avec `src_executor`. C'est la seule
chose dans tout le code qui publie réellement sur le topic du robot.

### Constantes (lignes 8-12)

```python
NUM_JOINTS = 24
TOPIC = "/motion/joint_override_command"
DEFAULT_STIFFNESS = [...]  # kp par joint
DEFAULT_DAMPING = [...]    # kd par joint
```

24 joints au total (jambes + buste + bras + tête). Chaque joint a un gain
de rigidité (`stiffness`/kp) et d'amortissement (`damping`/kd) par défaut,
qui peuvent être changés à la volée par joint via `set_gains`.

### `__init__` (lignes 17-31)

```python
qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST, depth=1,
)
self._pub = node.create_publisher(JointOverrideCommand, TOPIC, qos)
self._position = [0.0] * NUM_JOINTS
self._touched = [False] * NUM_JOINTS
...
self._wait_for_subscriber(subscriber_timeout)
```

`_position`/`_touched` : l'état interne complet des 24 joints. `_touched`
distingue "ce joint a une consigne active" de "ce joint est à 0.0 par
défaut, ignoré". La QoS `BEST_EFFORT`/`VOLATILE` ne retransmet jamais un
message manqué -- d'où l'attente explicite d'un abonné réel avant de
publier quoi que ce soit (sinon les premiers messages partent dans le
vide, sans erreur visible).

### `set_weight`/`set_gains` (lignes 33-44)

```python
def set_weight(self, weight):
    self._weight = float(weight)
    if any(self._touched):
        self._publish()

def set_gains(self, joint_index, stiffness=None, damping=None):
    if stiffness is not None:
        self._stiffness[joint_index] = float(stiffness)
    ...
    if self._touched[joint_index]:
        self._publish()
```

`weight` = poids GLOBAL de l'override (0.0 = aucune influence, 1.0 =
contrôle total, écrase la politique active). Les deux méthodes republient
IMMÉDIATEMENT si le changement concerne un joint déjà actif -- pour
appliquer l'effet tout de suite plutôt que d'attendre la prochaine
écriture de position.

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

`lever[13] = 0.5` (syntaxe `__setitem__`) modifie **un seul** joint et
publie **immédiatement** -- si on fait ça pour 10 joints d'affilée
(bras gauche + droit), ça envoie **10 messages ROS2 séparés** pour ce qui
devrait être une seule consigne de posture. C'est exactement le bug
diagnostiqué plus tôt dans ce projet comme cause des vibrations de bras.
`set_batch` corrige ça : met à jour TOUS les indices donnés en mémoire
**avant** d'appeler `_publish()` une seule fois -- un seul message ROS2
pour toute une posture.

### `release`/`forget`/`untouch` (lignes 67-81)

- **`release()`** : rend TOUT au contrôleur natif d'un coup (weight=0,
  tous les `_touched` remis à `False`).
- **`forget()`** : oublie l'état local SANS rien publier -- pour repartir
  propre en tête d'une nouvelle séquence sans envoyer de message.
- **`untouch(indices)`** : relâchement PARTIEL, seulement certains joints
  (utile pour rendre les jambes à la marche tout en gardant les bras).

### `_publish` (lignes 83-95) -- ce qui part réellement sur le réseau

```python
def _publish(self):
    indices = [i for i, touched in enumerate(self._touched) if touched]
    msg = JointOverrideCommand()
    msg.header.stamp = self._node.get_clock().now().to_msg()
    msg.weight = self._weight
    msg.joint_indices = indices
    msg.position = [self._position[i] for i in indices]
    ...
    self._pub.publish(msg)
```

Construit le message ROS2 avec **uniquement** les joints marqués
`_touched` (les autres ne sont pas mentionnés -- laissés au contrôleur
natif), et l'envoie. Toute la classe existe pour que cette fonction ne
soit appelée qu'**une fois par posture complète**, jamais une fois par
joint.

---

## 3. `motion_state.py` -- le verrou de sécurité

Une seule fonction : `ensure_motion_state(node, target, timeout, detour)`.
Sans elle, envoyer des positions sur `/motion/joint_override_command`
**n'a aucun effet visible** si le robot n'est pas dans le bon état
(`lower_body_balance`) -- pas d'erreur, juste rien qui bouge.

### Le mécanisme (lignes 9-43)

```python
sub = node.create_subscription(MotionState, "/motion/motion_state", _cb, state_qos)
pub = node.create_publisher(MotionStateRequest, "/motion/set_motion_state", request_qos)
```

S'abonne à l'état courant (`/motion/motion_state`, publié en continu par
`src_executor`) et prépare un publisher pour DEMANDER un changement d'état
(`/motion/set_motion_state`). `_switch_to(name)` : si déjà dans l'état
demandé, ne fait rien ; sinon publie la demande et attend (jusqu'à
`timeout`) que l'état courant devienne bien `name`.

### La logique de détour (lignes 45-68)

```python
if target not in state["available"] and detour in state["available"]:
    print(f"... non atteignable directement -- detour par {detour}...")
    if not _switch_to(detour):
        ...
```

La machine à états n'autorise pas toutes les transitions directement --
`state["available"]` liste les états atteignables DEPUIS l'état courant.
Si `lower_body_balance` n'est pas directement accessible mais que
`pd_stand` (le détour par défaut) l'est, la fonction passe d'abord par
`pd_stand` avant de retenter la cible finale.

---

## 4. `levee.py` -- la bibliothèque de mouvement + script "lever seul"

Ce fichier a un double rôle : c'est la **bibliothèque** que `levee_pivot.py`
importe (constantes + fonctions de rampe), ET un script autonome
utilisable seul (`python3 levee.py`) pour tester juste la prise, sans
pivot ni dépose.

### Constantes géométriques et de timing (lignes 20-65)

- `LEFT_JOINT_INDICES`/`RIGHT_JOINT_INDICES` : les index (13-17, 18-22)
  des 5 joints de chaque bras dans le tableau des 24 joints du robot.
- `Q_LEFT_HOME`/`Q_RIGHT_HOME` : la posture "bras le long du corps", quasi
  droite (mesurée, pas ronde -- ce sont les vraies valeurs du robot au
  repos).
- `WAYPOINT_Q_LEFT`/`WAYPOINT_Q_RIGHT` : posture "coudes vers l'arrière,
  avant-bras horizontal" -- point de passage intermédiaire, validé par
  cinématique directe pour avoir une forme en "L" précise.
- `PINCH_X`/`PINCH_Y`/`SQUEEZE_Y`/`LIFT_Z` : la géométrie du carton --
  distance de visée, écart Y avant contact, écart Y de serrage, hauteur
  de levée.
- Toutes les `*_DURATION` : les temps de chaque étape (ajustés plusieurs
  fois cette semaine pour accélérer/fluidifier la séquence).
- `LEVEE_STIFFNESS`/`LEVEE_DAMPING` (130/3.0) : gains PD réduits/renforcés
  spécifiquement pour porter une charge -- c'est le fix anti-vibration
  (rigidité `kp` standard des bras = 250, trop raide sous charge sans
  amortissement suffisant).

### `_checkpoint` (lignes 68-71)

```python
def _checkpoint(message, confirm):
    print(f"[ETAPE] {message}", flush=True)
    if confirm:
        input("        Verifie le robot, puis Entree pour continuer...")
```

Affiche l'étape en cours et, si `confirm=True` (comportement par défaut,
désactivable avec `--no-confirm`), **bloque** en attendant une touche
Entrée -- le garde-fou principal pour vérifier visuellement chaque
mouvement avant de continuer sur le vrai robot.

### `_bend_knees`/`_straighten_knees` (lignes 85-120)

Flexion/redressement coordonné hanche+genou+cheville, interpolation
**quintique** (`_quintic_ease`, vitesse ET accélération nulles aux deux
bords -- plus doux qu'un smoothstep simple, imite le comportement natif
du contrôleur `pd_stand`). N'est utilisé que si `walk_stance_scale > 0`
(= 0.0 par défaut sur le vrai robot -- désactivé, contrairement à la
sim qui fléchit toujours après une marche).

### `_publish`/`move_arms`/`cartesian_ramp` (lignes 123-160) -- les 2 façons de bouger un bras

```python
def move_arms(lever, qL0, qL1, qR0, qR1, duration, dry_run=False):
    n = max(1, int(duration * RATE_HZ))
    for i in range(n + 1):
        a = ease(i / n)
        qL = qL0 + a * (qL1 - qL0)
        qR = qR0 + a * (qR1 - qR0)
        ...
        _publish(lever, qL, qR)
```

**`move_arms`** : interpole en **espace articulaire** -- directement entre
deux jeux d'angles. Simple et rapide, mais la trajectoire de la MAIN dans
l'espace peut être imprévisible (courbe) car la relation angles->position
n'est pas linéaire.

```python
def cartesian_ramp(lever, q_init, wrist_rotation, start, end, anchor_start, anchor_end,
                    duration, dry_run):
    qL = q_init.copy()
    for i in range(n + 1):
        a = ease(i / n)
        target = start + a * (end - start)
        anchor = (1.0 - a) * anchor_start + a * anchor_end
        qL = solve_arm_ik(..., target, qL, ..., null_space_pref=anchor)
        qR = mirror_left_to_right(qL)
        _publish(lever, qL, qR)
    return qL, mirror_left_to_right(qL)
```

**`cartesian_ramp`** : interpole en **ligne droite dans l'espace 3D** de
la main (`target`), puis résout l'IK à CHAQUE pas pour trouver les angles
correspondants. Plus coûteux en calcul mais garantit une trajectoire de
main prévisible -- utilisé partout où la forme du chemin compte (éviter un
arc, éviter de retraverser une zone). Note `anchor` : l'ancre du
null-space peut elle-même glisser de `anchor_start` à `anchor_end` au fil
de la rampe (contrairement à la boucle fusionnée de `levee_pivot.py` qui
utilise une ancre fixe) -- les deux techniques existent dans ce projet
selon le besoin.

### `run_lift_sequence` (lignes 177-267) -- approche/serrage/levée en standalone

C'est un sous-ensemble de ce que fait `levee_pivot.py` (voir section 6) :
même logique d'approche/serrage/levée, mais s'arrête après avoir levé le
carton (pas de pivot, pas de dépose) -- utile pour tester/calibrer
seulement la prise.

### `main` (lignes 270-297)

Même structure que `levee_pivot.py::main()` : init ROS2, vérifie/force
`lower_body_balance`, crée le `Lever`, exécute, nettoie. Voir section 6
pour le détail (identique).

---

## 5. `pivot_real.py` -- le pivot seul, en standalone

Comme `levee.py`, ce fichier sert **deux usages** : bibliothèque de
constantes pour `levee_pivot.py` (`WAIST_JOINT_INDEX`, `WAIST_KP`,
`WAIST_KD`, la fonction `_ease` renommée `_pivot_ease` à l'import), ET un
script autonome (`python3 pivot_real.py`) pour tester la rotation du buste
**seule**, sans carton (flexion genoux -> pivot -> dépivot -> redressement
-> relâchement, tout ça sans les bras).

### Différence de constantes avec `levee_pivot.py`

Remarque : `WAIST_KP=150.0`/`WAIST_KD=3.0` ici, alors que
`levee_pivot.py` utilise les MÊMES noms importés (`WAIST_KP`, `WAIST_KD`)
-- ce sont bien les mêmes valeurs partagées par import, pas une
coïncidence. Par contre `RATE_HZ=30` ici est **différent** de
`RATE_HZ=65` dans `levee.py` -- ce fichier standalone n'est pas concerné
par le fix de fréquence de contrôle appliqué à la séquence bras (voir
mémoire projet : la boucle bas niveau tourne à ~500Hz, publier à 30Hz au
lieu de 65Hz produit une référence "en escalier" plus grossière). Comme
`pivot_real.py` ne bouge que le buste (charge différente, pas de bras
sous charge), ce n'a pas été jugé nécessaire de l'aligner.

### `run_pivot` (lignes 90-147)

Séquence complète autonome : flexion genoux (si demandé) -> gains buste ->
pivot 0->angle -> maintien -> **dépivot** angle->0 -> redressement genoux
-> relâchement. Notez que `levee_pivot.py` **n'utilise PAS** cette
fonction -- il réimplémente sa propre boucle de pivot (avec le carton en
main, fusionnée avec le rapproché/tendu des bras). `pivot_real.py` reste
utile pour un test isolé du buste, indépendamment de toute prise.

---

## 6. `levee_pivot.py` -- l'orchestrateur final

C'est le script qu'on lance pour la séquence complète. Il importe TOUT le
reste (`lever.py`, `motion_state.py`, `levee.py`, `pivot_real.py`,
`lift_carton.py`) et enchaîne les étapes dans une seule fonction,
`run_lift_and_pivot`.

### Import et helper (lignes 1-33)

```python
from pivot_real import WAIST_JOINT_INDEX, WAIST_KP, WAIST_KD, _ease as _pivot_ease
...
def _publish_with_waist(lever, qL, qR, waist):
    lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES + [WAIST_JOINT_INDEX],
                     list(qL) + list(qR) + [waist])
```

`_publish_with_waist` étend le principe de `set_batch` : publie bras
gauche + bras droit + buste en **un seul** message, pour les étapes où
les trois bougent en même temps (le pivot).

### Setup (lignes 36-60)

Précalcule `q_pinch_L`/`q_squeeze_L` (les 2 postures de visée et de
serrage) **avant même de commencer à bouger** -- ces valeurs servent de
cibles pour les étapes suivantes, calculées une fois pour toutes.
`run_approche`/`run_serrage`/`run_levee`/`run_pivot`/`run_depose` : des
booléens dérivés de `--only-phase`, pour pouvoir rejouer une seule étape.

### Approche -> serrage -> levée (lignes 62-106)

Identique à `run_lift_sequence` de `levee.py` (même logique, dupliquée
car ce fichier a besoin d'enchaîner directement avec le pivot après, sans
s'arrêter). Voir section 4 pour le détail des rampes cartésiennes.

### Pivot fusionné avec rapproché/tendu des bras (lignes 108-159)

Le bloc le plus récent et le plus dense. Voir l'explication complète
donnée en conversation -- résumé : le buste tourne en continu de 0° à
l'angle cible sur `pivot_duration`, PENDANT que les bras rapprochent le
carton (les `retract_duration` premières secondes) puis le tendent pour
la dépose (les `extend_duration` dernières secondes), au lieu de 3 étapes
séparées à l'arrêt. L'ancre du null-space (`anchor`, ligne 135) est figée
une seule fois avant la boucle -- jamais réassignée dedans, pour éviter
toute dérive.

### Dépose (lignes 171-227)

Dans l'ordre : baisse légère -> désserrage (**LE CARTON EST RELÂCHÉ ICI**)
-> écartement -> translation arrière -> dégagement (coudes vers l'arrière)
-> **dépivot** (buste revient à 0°) -> **retour bras le long du corps**.
L'ordre dépivot-puis-retour-bras (et non l'inverse) est un fix récent :
ramener les bras au corps AVANT de redresser le buste créait un
déséquilibre (masse des bras décalée par rapport aux pieds pendant que le
buste est encore tourné), mesuré en simulation (~14° de torsion du bassin
en moins d'une seconde) puis porté ici.

### Relâchement final (lignes 234-241)

```python
n2 = max(1, int(args.release_ramp_seconds * RATE_HZ))
for i in range(n2 + 1):
    lever.set_weight(1.0 - i / n2)
    _publish_with_waist(lever, qL, qR, 0.0)
    time.sleep(1.0 / RATE_HZ)
lever.release()
```

Le poids de l'override (`weight`) descend progressivement de 1.0 à 0.0
AVANT `release()` -- un `release()` instantané rendrait le contrôle d'un
coup à la politique native, provoquant une chute visible et brutale des
bras. La rampe laisse le temps à la transition d'être douce.

### `_build_arg_parser`/`main` (lignes 246-305)

Chaque paramètre de durée/distance est exposé en argument CLI (permet de
tester une valeur différente sans toucher au code). `main()` : init ROS2,
force `lower_body_balance` via `ensure_motion_state`, crée le `Lever`,
exécute `run_lift_and_pivot`, nettoie dans un `finally` (garanti même en
cas d'erreur ou de Ctrl+C).
