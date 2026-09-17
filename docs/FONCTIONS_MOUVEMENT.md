# Catalogue des fonctions de mouvement

Liste de **toutes les capacités** utilisées pour faire bouger le PM01 dans
ce projet -- pas fichier par fichier (voir [`CODE_ROBOT_REEL.md`](CODE_ROBOT_REEL.md)
et [`CODE_SIMULATION.md`](CODE_SIMULATION.md) pour ça), mais **capacité par
capacité** : qu'est-ce qui existe, quelle fonction l'implémente, comment on
l'appelle, et quelle technique elle utilise.

Sauf mention contraire, les signatures données sont celles du **robot réel**
(`package_sequence_bras_reel/`). Les différences avec la simulation sont
notées à chaque fois qu'elles existent.

---

## A. Cinématique inverse -- les briques de base

### 1. Viser un point 3D, IK "libre" (5 degrés de liberté)

```python
solve_ik(chain, hand_offset, target, q_init, iters=150, damping=0.05)
```

- **Entrée** : `target` = `np.array([x, y, z])` en mètres, repère du
  bassin. `q_init` = les 5 angles de départ (radians).
- **Technique** : Newton amorti ("damped least squares"). 5 inconnues (les
  5 angles) pour 3 équations (X,Y,Z) -- infinité de solutions possibles,
  celle trouvée est la plus proche de `q_init`.
- **Utilisation réelle dans ce projet** : quasiment jamais appelée
  directement -- presque tous les appels passent par `solve_arm_ik`
  ci-dessous, qui l'appelle en interne si aucun verrou n'est demandé.

### 2. Viser un point 3D avec UNE articulation verrouillée (la vraie fonction utilisée partout)

```python
solve_arm_ik(chain, hand_offset, target, q_init,
             lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
             iters=200, damping=0.05, null_space_gain=0.2, null_space_pref=None)
```

C'est la version de ton exemple ("on utilise de l'IK et on lui donne des
coordonnées x,y,z") -- **exacte**, avec deux précisions importantes :

- **`lock_index`/`lock_angle`** : fige UNE des 5 articulations à un angle
  fixe, ne résout que les 4 autres. Sur le **robot réel**, c'est le
  **poignet** (`WRIST_CHAIN_INDEX=4`) qui est verrouillé -- permet de
  tourner la main d'un angle donné (`--wrist-rotation-deg`) sans que l'IK
  ne le change. En **simulation**, c'est le **coude**
  (`ELBOW_PITCH_CHAIN_INDEX=3`) qui est verrouillé à la place -- pour
  garder un "coude tendu" esthétique pendant la visée.
- **`null_space_pref`** : avec 1 articulation verrouillée, il reste
  1 degré de liberté redondant parmi les 4 restants (3 équations,
  4 inconnues) -- sans ce paramètre, l'IK peut choisir une posture
  bizarre (épaule tordue) qui atteint quand même la cible. Ce paramètre
  dit "en cas de choix, préfère te rapprocher de CETTE posture".
  **Règle d'or de ce projet** (retrouvée plusieurs fois en corrigeant des
  bugs de rotation d'avant-bras) : cette préférence doit être une valeur
  **FIXE**, capturée une fois avant toute boucle qui appelle `solve_arm_ik`
  plusieurs fois de suite -- jamais une variable qui change à chaque
  itération, sinon la posture "dérive" visiblement au fil du mouvement.

### 3. Dériver le bras droit depuis le bras gauche

```python
mirror_left_to_right(q_left)  # inverse les index 1, 2, 4 (roll/yaw d'epaule, yaw de coude)
```

Le bras droit n'a **jamais** sa propre résolution IK dans ce projet --
toujours dérivé du gauche par symétrie, pour garantir une prise parfaitement
symétrique (au prix d'une précision légèrement moindre côté droit si le
robot n'est pas parfaitement centré).

---

## B. Bouger le bras -- les 2 façons de passer d'une posture à une autre

### 4. Interpolation articulaire ("aller vers cette posture", sans se soucier du chemin de la main)

```python
move_arms(lever, qL0, qL1, qR0, qR1, duration, dry_run=False)
```

- **Entrée** : 2 postures complètes (5 angles chacune) pour chaque bras,
  une durée.
- **Technique** : interpole directement les ANGLES entre le départ et
  l'arrivée (`qL = qL0 + a * (qL1 - qL0)`, avec `a` = courbe smoothstep).
  Pas d'IK à chaque pas -- rapide, mais la trajectoire de la MAIN dans
  l'espace peut être imprévisible (courbe/arc) puisque la relation
  angles→position n'est pas linéaire.
- **Utilisée pour** : les mouvements où seule la posture FINALE compte
  (ex : "va au point de passage coudes-vers-l'arrière", "reviens à la
  posture de repos") -- pas les mouvements où la main doit suivre un
  chemin précis (éviter un obstacle, ne pas retraverser une zone).

### 5. Interpolation cartésienne ("déplace la main en ligne droite jusqu'à ce point")

```python
cartesian_ramp(lever, q_init, wrist_rotation, start, end, anchor_start, anchor_end, duration, dry_run)
```

- **Entrée** : `start`/`end` = positions 3D de la main (pas des angles).
- **Technique** : interpole la POSITION cible en ligne droite dans
  l'espace 3D à chaque pas, puis résout `solve_arm_ik` (poignet
  verrouillé) pour trouver les angles correspondants CE pas-ci, en
  partant des angles du pas précédent. `anchor_start`/`anchor_end` :
  l'ancre du null-space peut elle-même glisser d'une valeur à l'autre au
  fil de la rampe (contrairement à la règle "ancre fixe" ci-dessus --
  cette fonction gère le glissement PROPREMENT, en l'interpolant à
  chaque pas comme la position, ce qui est différent d'une ancre qui
  dérive accidentellement).
- **Utilisée pour** : tout ce qui doit suivre un chemin garanti -- la
  main qui monte verticalement puis avance horizontalement (évite un arc
  qui ferait monter la main trop haut), le retrait/l'extension pendant
  le pivot, la translation arrière avant de replier les bras.
- **Version fusionnée avec le pivot** (robot réel uniquement,
  `levee_pivot.py`) : une variante où la boucle mélange interpolation de
  la position de la main (les X premières/dernières secondes) ET
  rotation du buste en même temps -- voir capacité 6 ci-dessous.

---

## C. Bouger le buste

### 6. Pivoter (pas d'IK -- un seul joint, angle direct)

```python
lever[WAIST_JOINT_INDEX] = float(waist_angle)  # ou lever.set_batch(..., [WAIST_JOINT_INDEX], ...)
```

Le buste n'a qu'**une seule articulation** de rotation (yaw) -- pas
besoin d'IK, on calcule directement l'angle voulu à chaque pas d'une
rampe temporelle (`_pivot_ease(i/n) * angle_cible`) et on le publie. La
seule complexité est la **coordination** avec les bras (les publier dans
le même message via `set_batch`/`_publish_with_waist`) et avec les gains
PD (`set_gains(WAIST_JOINT_INDEX, kp, kd)` -- montés pendant la tenue
active du pivot, redescendus avant le relâchement pour éviter un choc).

Sur le robot réel (`levee_pivot.py`), cette rotation est **fusionnée**
avec le rapproché/l'extension des bras (capacité 5) dans une seule
boucle : le buste tourne en continu du début à la fin, pendant que la
cible de la main change seulement au début (rapproché) et à la fin
(extension) de cette même fenêtre de temps -- un seul mouvement au lieu
de 3 étapes séparées. En simulation (`pivot.py`), ces 3 étapes restent
séquentielles (pas encore fusionnées à ce jour).

---

## D. Bouger les jambes

### 7. Fléchir / redresser les genoux (posture de "marche au repos")

```python
_bend_knees(lever, scale, stiffness_scale, duration)       # droites -> flechies
_straighten_knees(lever, scale, stiffness_scale, duration)  # flechies -> droites
```

- **Technique** : contrôle direct de 6 joints (hanche+genou+cheville x2
  côtés), interpolation **quintique** (`_quintic_ease` -- vitesse ET
  accélération nulles aux deux bords, plus doux qu'un smoothstep simple,
  imite le comportement natif du contrôleur `pd_stand`), gains PD
  renforcés (`stiffness_scale`) pour tenir la posture sous charge.
- **Sur le robot réel** : désactivé par défaut (`WALK_STANCE_SCALE=0.0` --
  jambes droites, séquence bras-seul). En **simulation**, toujours activé
  après une marche (le robot vient de marcher, ses jambes sont déjà dans
  cette posture, et `pivot.py`/`depose.py` doivent la maintenir en la
  republiant en continu).

---

## E. Comment ça part réellement vers le robot (la couche `Lever`)

### 8. Changer un seul joint (publie immédiatement)

```python
lever[13] = 0.5   # __setitem__ -- publie tout de suite, UN message ROS2
```

### 9. Changer plusieurs joints d'un coup (publication atomique -- LA bonne pratique)

```python
lever.set_batch([13, 14, 15, 16, 17, 18, 19, 20, 21, 22], list(qL) + list(qR))
```

Met à jour tous les indices donnés EN MÉMOIRE avant de publier **une
seule fois**. C'est le fix anti-vibration central de ce projet : appeler
`lever[idx] = ...` en boucle pour 10 joints envoie 10 messages ROS2
séparés pour ce qui devrait être une seule consigne de posture --
`set_batch` corrige ça partout où c'est utilisé (bras, jambes, buste).

### 10. Changer les gains PD d'un joint à la volée

```python
lever.set_gains(joint_index, stiffness=130.0, damping=3.0)
```

Utilisé pour réduire la rigidité et augmenter l'amortissement pendant le
port de charge (`LEVEE_STIFFNESS`/`LEVEE_DAMPING`), ou pour assouplir le
buste juste avant un relâchement (`WAIST_KP_RELEASE`/`WAIST_KD_RELEASE`).

### 11. Relâcher le contrôle progressivement

```python
for i in range(n + 1):
    lever.set_weight(1.0 - i / n)   # rampe le poids de l'override 1.0 -> 0.0
    lever.set_batch(..., qL, qR)     # continue de republier la posture tenue
lever.release()                      # rend TOUT au controleur natif
```

Un `release()` instantané (sans rampe de poids avant) rend le contrôle
d'un coup à la politique active -- chute visible et brutale des bras.
Toujours précédé d'une rampe de `weight`.

---

## F. Sécurité -- vérifier/forcer l'état du robot

### 12. S'assurer que le robot est dans le bon état avant de bouger les bras

```python
ensure_motion_state(node, "lower_body_balance", timeout=3.0, detour="pd_stand")
```

Sans ça, publier sur `/motion/joint_override_command` **n'a aucun effet
visible** si le robot n'est pas déjà dans l'état `lower_body_balance` --
pas d'erreur, juste rien qui bouge. Gère aussi le cas où l'état cible
n'est pas atteignable directement (passe par un détour, `pd_stand` par
défaut).

---

## G. Marcher (mécanisme complètement séparé des bras)

### 13. Marche avant/rotation

```python
# Publie sur /motion/body_vel_cmd, PAS du tout de l'IK ni du Lever
msg.linear_velocity = [forward_mps, 0.0]
msg.yaw_velocity = turn_rad_s
```

Vitesse linéaire (m/s) + vitesse angulaire (rad/s), publiées en continu
pendant `duration` secondes -- le contrôleur de marche natif (une
politique RL, `rl_terrain`) gère lui-même la génération de la démarche
(placement des pieds, etc.), ce projet ne fait que lui donner une
consigne de vitesse. En simulation, un node-pont (`body_vel_bridge.py`)
traduit ce protocole vers l'émulation manette que MuJoCo comprend --
absent du lancement réel (le vrai `src_executor` consomme ce protocole
nativement).

---

## H. Trouver où est le carton

### 14. Position du carton -- robot réel

Le script `levee_pivot.py` **ne calcule rien lui-même** : `--pinch-x`,
`--pinch-z` etc. sont des arguments CLI, supposés déjà mesurés/calibrés
en amont (vision ArUco, voir `tools/vision/`) avant de lancer la
séquence.

### 15. Position du carton -- simulation

```python
face_gauche, face_droite = carton_face_centers()          # lit le XML de la scene MuJoCo
pose = SimStateListener().pose()                            # (x, y, z, yaw) du robot, verite terrain
cible_locale = world_to_robot_local(face_gauche, pose)       # repere bassin, utilisable par solve_arm_ik
```

Trois fonctions combinées : lire où est le carton dans le MONDE (XML),
lire où est le robot dans le MONDE (LCM `sim_state`), puis convertir la
première par rapport à la seconde (soustraction + rotation autour de Z
uniquement) pour obtenir une cible IK directement utilisable. Recalculé
à chaque appel (jamais mis en cache) -- le robot peut légèrement dériver
entre deux étapes.

---

## Vocabulaire de haut niveau : comment ces briques s'assemblent

| Étape de la séquence | Briques utilisées |
|---|---|
| Point de passage (coudes arrière) | `move_arms` (interpolation articulaire) |
| Approche (viser le carton) | `cartesian_ramp` (ligne droite, poignet/coude verrouillé) |
| Serrage | `move_arms` (petit déplacement articulaire) |
| Levée | `cartesian_ramp` (Z pur), gains réduits (`set_gains`) |
| Rapproché + Pivot + Extension | boucle fusionnée : angle buste direct + `solve_arm_ik` à chaque pas |
| Baisse avant relâchement | `cartesian_ramp` (Z pur) |
| Désserrage / écartement | `solve_arm_ik` (cible ponctuelle) + `move_arms` |
| Translation arrière | `cartesian_ramp` (X pur) |
| Dégagement | `cartesian_ramp` puis `move_arms` vers la posture de repos |
| Dépivot | angle buste direct, gains qui s'assouplissent |
| Relâchement final | rampe de `weight` + `release()` |
