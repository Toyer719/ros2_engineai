"""Table des champs GamepadKeys <-> topics ROS, partagee entre chef_node (qui les
agrege vers LCM) et les noeuds de comportement (walk_to, ...) qui les publient.

Copie de BUTTON_INDEX/ANALOG_INDEX (ancien core/chef.py) -- garder synchronise si
l'un des deux change. Chaque champ correspond a un topic std_msgs distinct :
Bool pour les boutons (digital_states), Float32 pour les sticks (analog_states).
"""

BUTTON_INDEX = {
    "LB": 0, "RB": 1, "A": 2, "B": 3, "X": 4, "Y": 5,
    "BACK": 6, "START": 7, "UP": 8, "DOWN": 9, "LEFT": 10, "RIGHT": 11,
}
ANALOG_INDEX = {
    "LEFT_STICK_X": 2, "LEFT_STICK_Y": 3, "RIGHT_STICK_X": 4, "RIGHT_STICK_Y": 5,
}
# LT/RT (analog_states[0]/[1]) non exposes ici -- ajoute-les si un node en a besoin.


def field_topic(name: str) -> str:
    return f"/virtual_gamepad/cmd/{name.lower()}"
