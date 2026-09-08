"""Lance le chef et les noeuds de comportement -- seul fichier qui sait quels
noeuds existent au demarrage, equivalent ROS de l'ancien main.py.

Note : le node `lift` importe `interface_protocol.msg.JointOverrideCommand`
(via lever.py) au chargement -- en plus du setup.bash de ce ros_ws, il faut
avoir source `build/ros2_env/install/local_setup.bash` du SDK AVANT de
lancer ce launch file, sinon `lift` plante au demarrage (les autres nodes
n'en ont pas besoin).

`pelvis_camera_sim`/`carton_pose_publisher` (tools/vision/, hors package ROS,
scripts standalone) sont lances ici en `nice -n 15` -- sans ca, le rendu
MuJoCo offscreen sature un coeur au point de faire tomber le robot pendant
stand()/walk (contention CPU confirmee le 19/08, voir memoire projet
[[project_pm01_carton_vision]]). Chaque Action (stand/walk_to/lift) attend
deja son serveur jusqu'a 10s (`wait_for_server`) cote chef_node.py, donc
l'ordre de demarrage simultane ci-dessous est sans risque.

DELAI de 13s AJOUTE sur le demarrage de la vision (24/08) : `nice -n 15`
seul s'est avere INSUFFISANT lors d'une session chargee (nombreux colcon
build/redemarrages en parallele) -- confirme par test isole : stand()
reussit systematiquement SANS la vision active (z=0.82 immediat), et
echoue systematiquement AVEC (z=0.08-0.13, jamais recupere) dans les MEMES
conditions par ailleurs. Le rendu MuJoCo offscreen de pelvis_camera_sim.py
consomme ~35-40% CPU en continu des son demarrage, meme avec nice -- assez
pour perturber la transition passive->pd_stand (fenetre sensible ~10s,
meme duree que stand.py::settle_seconds). Demarrer la vision APRES cette
fenetre (13s de marge) elimine la contention pendant le moment critique.

DECALAGE de 1s ENTRE les deux process vision (24/08, suite) : les demarrer
au MEME instant (meme TimerAction) a fait planter pelvis_camera_sim.py au
demarrage ("Failed to find a free participant index for domain 69" --
meme erreur transitoire de creation de participant CycloneDDS observee
plusieurs fois cette session lors de demarrages ROS2 simultanes/rapproches)
alors que carton_pose_publisher.py survivait -- decaler leur creation de
participant dans le temps reduit la contention au moment critique.

RESTE A LANCER A PART (2 binaires C++, pas des nodes ROS Python) AVANT ce
launch file, voir docs/COMMENT_LANCER_VIRTUAL_GAMEPAD_ROS.txt :
  ./scripts/run_mujoco.sh pm01_edu_carton   (physique)
  ./run.sh pm01_edu_carton                  (machine a etats / arbitre)"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction

VISION_DIR = "/home/equansrobotic/stagiaire_1/tools/vision"
VISION_START_DELAY_S = 13.0
VISION_STAGGER_S = 1.0


def generate_launch_description():
    return LaunchDescription([
        Node(package='virtual_gamepad_ros', executable='walk_to', name='walk_to', output='screen'),
        Node(package='virtual_gamepad_ros', executable='stand', name='stand', output='screen'),
        Node(package='virtual_gamepad_ros', executable='lift', name='lift', output='screen'),
        Node(package='virtual_gamepad_ros', executable='pivot', name='pivot', output='screen'),
        Node(package='virtual_gamepad_ros', executable='depose', name='depose', output='screen'),
        TimerAction(period=VISION_START_DELAY_S, actions=[
            ExecuteProcess(
                cmd=['nice', '-n', '15', 'python3', 'pelvis_camera_sim.py'],
                cwd=VISION_DIR, output='screen', name='pelvis_camera_sim',
            ),
        ]),
        TimerAction(period=VISION_START_DELAY_S + VISION_STAGGER_S, actions=[
            ExecuteProcess(
                cmd=['nice', '-n', '15', 'python3', 'carton_pose_publisher.py',
                     '--marker-size', '0.17',
                     '--image-topic', '/camera/pelvis_sim/image_raw'],
                cwd=VISION_DIR, output='screen', name='carton_pose_publisher',
            ),
        ]),
        Node(package='virtual_gamepad_ros', executable='chef', name='chef', output='screen'),
    ])
