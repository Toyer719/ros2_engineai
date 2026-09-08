# ros2_engineai

Pipeline ROS2 pour piloter un robot humanoïde PM01 (EngineAI) : marche jusqu'à
un carton, prise à deux bras, pivot du buste -- en simulation MuJoCo et sur
le robot physique.

## Le projet

Codé sous `tools/virtual_gamepad/`, ce package ROS2 (`virtual_gamepad_ros`)
orchestre une séquence GRAFCET via 5 Actions ROS2 indépendantes :

| node | rôle |
|---|---|
| `chef_node.py` | orchestrateur -- envoie les buts, ne bouge jamais rien lui-même |
| `walk_to.py` | marche (émulation manette LCM en sim, topic ROS2 natif sur le robot réel) |
| `stand.py` | passage en position debout stabilisée (`pd_stand`) |
| `lift.py` | prise du carton -- approche, serrage, levée (cinématique inverse) |
| `pivot.py` | rotation du buste, carton tenu |
| `depose.py` | repose le carton à un second poste |

La marche (jambes) et la prise (bras) utilisent deux mécanismes de commande
complètement séparés : la marche passe par une manette virtuelle émulée sur
LCM, la prise publie directement des positions articulaires cibles sur le
topic ROS2 natif `/motion/joint_override_command`.

Pour le robot réel, l'équivalent vit dans `tools/joint_angle_commander/`
(`orchestrateur.py`, `marche.py`, `levee.py`) -- même logique de prise
(`lever.py`, `robot_arm_ik/lift_carton.py`), partagée avec la simulation.

## Séquence validée (2026-09-08)

```
stand() → walk_to(Posage 1) → lift(approche) → lift(serrage)
        → lift(levée) → pivot(180°) → stand()
```

Testée en simulation (MuJoCo), stable de bout en bout. Le transport en
tenant le carton vers un second poste (marche après le pivot, puis
`depose()`) a été essayé puis retiré de la séquence par défaut après une
chute reproduite en télémétrie -- `depose.py` reste fonctionnel et
appelable isolément.

## Lancer en simulation

Dans le conteneur avec `run.sh` + `run_mujoco.sh` déjà démarrés :

```bash
source /opt/ros/humble/setup.bash
source <sdk>/build/ros2_env/install/local_setup.bash
source tools/virtual_gamepad/ros_ws/install/setup.bash
ros2 launch virtual_gamepad_ros virtual_gamepad_blind.launch.py
```

## Structure du dépôt

```
tools/virtual_gamepad/ros_ws/    package ROS2 principal (nodes + interfaces .action)
tools/joint_angle_commander/     scripts robot réel (marche, levée, IK)
tools/robot_arm_ik/              cinématique inverse des bras, partagée sim/réel
tools/vision/                    détection ArUco du carton (expérimental)
assets/                          scène MuJoCo, configs du robot simulé
docs/                            guides de lancement (Docker, ROS2, caméra)
```
