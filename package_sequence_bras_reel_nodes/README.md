# Séquence bras réel -- architecture multi-nodes

Variante EXPÉRIMENTALE de [`package_sequence_bras_reel/`](../package_sequence_bras_reel/)
qui reprend l'**architecture de la simulation** (plusieurs Action Servers
ROS2 indépendants, orchestrés par un "chef") au lieu d'un seul process
linéaire. **Pas encore testée sur le robot physique** -- uniquement
vérifiée numériquement (voir plus bas) et par compilation.

## Pourquoi cette variante existe

`package_sequence_bras_reel/` (un seul process/`Lever`) reste la version
**de référence, validée sur le robot réel**. Celle-ci sert à explorer
l'architecture multi-nodes sur le robot réel, sans risquer la version qui
marche déjà.

## Fichiers

- `lift_node.py` -- Action Server "lift" (approche/serrage/levée)
- `pivot_node.py` -- Action Server "pivot" (rapproché+pivot+extension fusionnés,
  même technique que `levee_pivot.py`)
- `depose_node.py` -- Action Server "depose"
- `chef_reel.py` -- orchestrateur, envoie les buts dans l'ordre (pas de
  marche/stand ici -- la séquence réelle reste bras-seul, jambes droites)

Réutilisent directement `Lever`, `ensure_motion_state` et les fonctions
IK/géométrie de `package_sequence_bras_reel/` (aucune duplication de la
cinématique inverse) -- seule la coordination change.

## Le piège déjà identifié, corrigé dès le départ

Comme en simulation (`pivot.py`/`depose.py`), chaque node est un **process
séparé** qui doit **recalculer** la posture tenue plutôt que de recevoir
`qL` en mémoire. Sans précaution, deux calculs IK indépendants peuvent
converger sur des postures différentes pour la même position de main.

**Ancre du null-space** : `pivot_node.py` garde une ancre **fixe**
(`q_squeeze_L`, la posture de serrage initiale) tout au long de sa boucle
retract/pivot/extend. `depose_node.py` doit reconstruire la **même**
ancre (calculée depuis `pinch_x`, PAS `depose_x`) -- une ancre différente
donnait un écart mesuré de ~2.9° / 6mm. Avec la bonne ancre, l'écart tombe
à 0.001° (bruit numérique) -- **vérifié par script avant d'écrire le
code**, pas après coup.

## État actuel

- ✅ Compile (`py_compile` sur les 4 fichiers)
- ✅ Vérifié numériquement (chaîne lift→pivot→depose, écart < 0.002°)
- ❌ **Jamais lancé sur ROS2** (ni sim, ni robot réel) -- nécessite de
  builder/sourcer `virtual_gamepad_interfaces` (déjà buildé dans
  `tools/virtual_gamepad/ros_ws/install/`)
- ❌ **Jamais testé sur le robot physique** -- à faire uniquement après
  confirmation explicite, comme pour tout code touchant le vrai robot

## Lancer (une fois prêt à tester)

```bash
source /opt/ros/humble/setup.bash
source <sdk>/build/ros2_env/install/local_setup.bash
source /home/equansrobotic/stagiaire_1/tools/virtual_gamepad/ros_ws/install/setup.bash

# 3 terminaux (ou 3 process en arriere-plan)
python3 lift_node.py
python3 pivot_node.py
python3 depose_node.py

# puis, dans un 4e terminal
python3 chef_reel.py --angle-deg 90
```
