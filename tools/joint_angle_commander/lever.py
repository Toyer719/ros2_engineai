                      
"""Pilotage du PM01 par angle articulaire, style tableau : lever[i] = angle.

Prérequis (le robot ou la sim doit déjà tourner : ./run.sh pm01_edu) :
    source /opt/ros/humble/setup.bash
    source build/ros2_env/install/local_setup.bash
    python3 tools/joint_angle_commander/lever.py

Numéros de joint (assets/config/pm01_edu/model/default.yaml) :
     0 J00_HIP_PITCH_L      6 J06_HIP_PITCH_R     12 J12_WAIST_YAW
     1 J01_HIP_ROLL_L       7 J07_HIP_ROLL_R      13 J13_SHOULDER_PITCH_L
     2 J02_HIP_YAW_L        8 J08_HIP_YAW_R       14 J14_SHOULDER_ROLL_L
     3 J03_KNEE_PITCH_L     9 J09_KNEE_PITCH_R    15 J15_SHOULDER_YAW_L
     4 J04_ANKLE_PITCH_L   10 J10_ANKLE_PITCH_R   16 J16_ELBOW_PITCH_L
     5 J05_ANKLE_ROLL_L    11 J11_ANKLE_ROLL_R    17 J17_ELBOW_YAW_L
                                                   18 J18_SHOULDER_PITCH_R
                                                   19 J19_SHOULDER_ROLL_R
                                                   20 J20_SHOULDER_YAW_R
                                                   21 J21_ELBOW_PITCH_R
                                                   22 J22_ELBOW_YAW_R
                                                   23 J23_HEAD_YAW
"""
import time

import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from interface_protocol.msg import JointOverrideCommand

NUM_JOINTS = 24
TOPIC = "/motion/joint_override_command"

                                                                               
DEFAULT_STIFFNESS = [float(v) for v in [200, 200, 380, 450, 400, 200] * 2 + [200] + [250] * 10 + [100]]
# Bras (indices 13-22) = 250 -- valeur eprouvee pour bouger le bras contre son propre poids/
# frottement en espace libre (approche/serrage, aucune charge externe). Teste le 2026-08-25 :
# 40 (essai precedent, cf. Guide_PM01_FR.pdf/exemple officiel kp=20) etait TROP FAIBLE, le
# bras n'a pas bouge du tout -- 250 reste necessaire pour l'authorite de mouvement en espace
# libre. Le risque de saturation (moteurs de bras = "petit moteur" Q25H, couple max 50 N.m)
# n'existe QUE sous charge reelle (tenue du carton) -- voir LEVEE_STIFFNESS dans
# levee.py, qui reduit le kp juste pour cette phase-la via set_gains(), plutot que
# d'affaiblir tout le monde ici.
DEFAULT_DAMPING = [float(v) for v in [5, 5, 5, 5, 2, 2] * 2 + [1] + [1] * 10 + [1]]


class Lever:
    """lever[i] = angle (radians) envoie immédiatement la commande au robot."""

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

    def set_weight(self, weight):
        """Change le poids GLOBAL de l'override (0.0=aucune influence, 1.0=controle total,
        ecrase toute sortie de la politique active sur les joints touches) pour les
        publications suivantes -- republie tout de suite si des joints sont deja touches."""
        self._weight = float(weight)
        if any(self._touched):
            self._publish()

    def set_gains(self, joint_index, stiffness=None, damping=None):
        """Change les gains PD d'UN joint pour les publications suivantes (republie tout de
        suite si ce joint est deja touche, pour appliquer le changement immediatement plutot
        que d'attendre le prochain __setitem__). stiffness/damping=None laisse la valeur
        actuelle inchangee (pratique pour ne fournir que --hold-stiffness OU --hold-damping)."""
        if stiffness is not None:
            self._stiffness[joint_index] = float(stiffness)
        if damping is not None:
            self._damping[joint_index] = float(damping)
        if self._touched[joint_index]:
            self._publish()

    def _wait_for_subscriber(self, timeout):
        """QoS BEST_EFFORT/VOLATILE ne retransmet rien : publier avant que la decouverte
        DDS ait fini (cote src_executor) fait partir les premiers messages dans le vide,
        sans erreur -- le reste du script continue et affiche un succes alors que rien
        n'a bouge sur le robot (constate en pratique). On attend donc explicitement un
        abonne reel, et on echoue bruyamment plutot que de deviner un delai fixe."""
        t0 = time.time()
        while self._pub.get_subscription_count() == 0:
            if time.time() - t0 > timeout:
                raise RuntimeError(
                    f"Aucun abonne sur {TOPIC} apres {timeout}s -- src_executor tourne-t-il "
                    "(et avec le bon RMW_IMPLEMENTATION/ROS_LOCALHOST_ONLY) ?"
                )
            time.sleep(0.05)

    def __setitem__(self, joint_index, angle_rad):
        self._position[joint_index] = angle_rad
        self._touched[joint_index] = True
        self._publish()

    def __getitem__(self, joint_index):
        return self._position[joint_index]

    def release(self):
        """Rend la main au Runner actif (pd_stand, walk, ...) -- TOUT ou rien,
        aucun des joints touches ne reste tenu apres ca (voir untouch() pour
        un relachement partiel)."""
        msg = JointOverrideCommand()
        msg.weight = 0.0
        self._pub.publish(msg)
        self._touched = [False] * NUM_JOINTS

    def forget(self):
        """Oublie l'etat local (`_touched`/`_stiffness`/`_damping` remis a
        neuf) SANS rien publier -- contrairement a release(), ne rend PAS
        la main au Runner actif (aucun message envoye). Sert a repartir sur
        une base propre en DEBUT d'une nouvelle sequence, dans un process
        longue-duree qui appelle ce Lever plusieurs fois de suite (ex: un
        ActionServer ROS2) -- sans ca, un joint touche par un appel
        anterieur JAMAIS relache explicitement (ex: only_phase="approche"
        seul, qui laisse expres les bras tenus pour un appel suivant) reste
        `_touched=True` indefiniment et pollue chaque publish() suivant,
        meme d'une sequence totalement independante (constate le 2026-09-02
        : bras/jambes d'un test precedent reapparaissant dans les messages
        d'un test sans rapport)."""
        self._touched = [False] * NUM_JOINTS
        self._stiffness = list(DEFAULT_STIFFNESS)
        self._damping = list(DEFAULT_DAMPING)

    def untouch(self, indices):
        """Relachement PARTIEL : rend la main au Runner actif sur les joints
        de `indices` uniquement (typiquement les jambes, pour rendre la
        marche a la politique RL) SANS toucher au poids global ni aux autres
        joints deja tenus (typiquement les bras, qui restent au dernier
        angle publie, meme poids) -- indispensable pour tenir un objet en
        MARCHANT (le override en place-tout-ou-rien de release() forcerait
        les jambes a rester figees, en conflit direct avec la politique de
        marche qui a besoin de les controler entierement)."""
        for i in indices:
            self._touched[i] = False
        self._publish()

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


if __name__ == "__main__":
    main()
