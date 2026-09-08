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

RESTE A LANCER A PART (2 binaires C++, pas des nodes ROS Python) AVANT ce
launch file, voir docs/COMMENT_LANCER_VIRTUAL_GAMEPAD_ROS.txt :
  ./scripts/run_mujoco.sh pm01_edu_carton   (physique)
  ./run.sh pm01_edu_carton                  (machine a etats / arbitre)"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

VISION_DIR = "/home/equansrobotic/stagiaire_1/tools/vision"


def generate_launch_description():
    return LaunchDescription([
        Node(package='virtual_gamepad_ros', executable='walk_to', name='walk_to', output='screen'),
        Node(package='virtual_gamepad_ros', executable='stand', name='stand', output='screen'),
        Node(package='virtual_gamepad_ros', executable='lift', name='lift', output='screen'),
        Node(package='virtual_gamepad_ros', executable='pivot', name='pivot', output='screen'),
        ExecuteProcess(
            cmd=['nice', '-n', '15', 'python3', 'pelvis_camera_sim.py'],
            cwd=VISION_DIR, output='screen', name='pelvis_camera_sim',
        ),
        ExecuteProcess(
            cmd=['nice', '-n', '15', 'python3', 'carton_pose_publisher.py',
                 '--marker-size', '0.1016',
                 '--image-topic', '/camera/pelvis_sim/image_raw'],
            cwd=VISION_DIR, output='screen', name='carton_pose_publisher',
        ),
        Node(package='virtual_gamepad_ros', executable='chef', name='chef', output='screen'),
    ])
