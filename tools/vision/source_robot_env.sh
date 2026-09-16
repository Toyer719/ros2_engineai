# A sourcer (pas executer) avant tout script ROS2 de ce dossier qui parle au
# vrai robot : `source source_robot_env.sh`
#
# Necessaire car ce PC a 2 cartes reseau (ens33 + ens37) -- sans ca, le
# robot reste invisible (aucune erreur, juste aucun topic/node distant).

source /opt/ros/humble/setup.bash

# Le robot et ce PC doivent parler sur le meme "domaine" ROS2 pour se voir
# (comme un numero de canal radio commun). 69 = celui utilise par Nezha/Jetson.
export ROS_DOMAIN_ID=69

# Middleware DDS a utiliser pour la communication ROS2. Le robot tourne sur
# CycloneDDS -- ce PC doit utiliser le meme, sinon deux reseaux invisibles
# l'un a l'autre coexistent silencieusement (meme domaine, DDS different).
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# A 0 (pas la valeur par defaut 1) pour autoriser la decouverte de nodes sur
# d'autres machines du reseau -- a 1, ROS2 ignorerait tout ce qui n'est pas
# sur cette machine, donc jamais le robot.
export ROS_LOCALHOST_ONLY=0

# Dit a CycloneDDS quelle carte reseau utiliser pour la decouverte (ens37,
# 192.168.0.200, celle sur le meme sous-reseau que le robot 192.168.0.x) --
# sans ca, CycloneDDS peut choisir ens33 (192.168.138.x) a la place, et la
# decouverte multicast ne passe jamais jusqu'au robot (ping/ssh marchent
# quand meme, car ils utilisent le routage IP normal, pas concerne par ca).
export CYCLONEDDS_URI=file:///home/equansrobotic/stagiaire_1/tools/vision/cyclonedds_ens37.xml
