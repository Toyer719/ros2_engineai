"""Variante allegee de virtual_gamepad.launch.py : lance le chef et les 5
ActionServers, SANS la vision (pelvis_camera_sim/carton_pose_publisher) --
run_sequence() ne l'appelle plus depuis le 08/09 (chemin vision retire), donc
la lancer ne fait que consommer du CPU pour rien et ajoute des participants
DDS inutiles (risque de "Failed to find a free participant index" en cas de
lancements rapproches, cf memoire projet).

Mêmes prerequis que virtual_gamepad.launch.py (voir son docstring) : sourcer
build/ros2_env/install/local_setup.bash du SDK AVANT ce launch (sinon `lift`
plante au demarrage), et avoir `run.sh`/`run_mujoco.sh` deja lances a part.

2026-09-10 : ajoute `body_vel_bridge` -- walk_to.py n'a plus qu'une seule
logique (toujours /motion/body_vel_cmd), ce node traduit vers l'emulation
manette LCM que MuJoCo comprend en sim (SIM UNIQUEMENT, absent du lancement
reel -- voir docstring de body_vel_bridge.py).
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='virtual_gamepad_ros', executable='walk_to', name='walk_to', output='screen'),
        Node(package='virtual_gamepad_ros', executable='stand', name='stand', output='screen'),
        Node(package='virtual_gamepad_ros', executable='lift', name='lift', output='screen'),
        Node(package='virtual_gamepad_ros', executable='pivot', name='pivot', output='screen'),
        Node(package='virtual_gamepad_ros', executable='depose', name='depose', output='screen'),
        Node(package='virtual_gamepad_ros', executable='body_vel_bridge', name='body_vel_bridge', output='screen'),
        Node(package='virtual_gamepad_ros', executable='chef', name='chef', output='screen'),
    ])
