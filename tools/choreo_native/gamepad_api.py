                      
"""
API Python pour piloter le PM01 comme le fait le Virtual Gamepad (meme canal LCM,
meme message GamepadKeys, meme frequence 20Hz) -- pensee pour etre appelee depuis un
script de sequencement (Grafcet) plutot que cliquee a la souris.

A lancer avec le python du venv de virtual_gamepad (paquet `lcm` requis) :
    ./tools/virtual_gamepad/.venv/bin/python3 tools/virtual_gamepad/gamepad_api.py --help

------------------------------------------------------------------------------------
ETATS ET TRANSITIONS REELS -- lus directement dans
assets/config/pm01_edu_carton/task_motion/default.yaml (identique pour pm01_edu) :

    idle -----------[LB,START]-----------> passive
    passive --------[LB,RB]---------------> idle
    passive --------[LB,A]----------------> pd_stand
    pd_stand -------[LB,RB]---------------> passive
    pd_stand -------[LB,B]-----------------> walk
    pd_stand -------[LB,X]-----------------> rl_lab
    pd_stand -------[RB,B]-----------------> dance
    walk -----------[LB,RB]---------------> passive
    walk -----------[LB,A]-----------------> pd_stand
    walk -----------[LB,X]-----------------> rl_lab
    walk -----------[RB,B]-----------------> dance
    rl_lab ---------[LB,RB]---------------> passive
    rl_lab ---------[LB,A]-----------------> pd_stand
    rl_lab ---------[LB,B]-----------------> walk
    rl_lab ---------[RB,B]-----------------> dance
    dance ----------[LB,RB]---------------> passive
    dance ----------[LB,A]-----------------> pd_stand
    dance ----------[LB,B]-----------------> walk

[LB,RB] (passive) est accepte depuis N'IMPORTE QUEL etat -- c'est l'arret d'urgence
"doux" documente dans le README ("Global Safety Mechanism").

IMPORTANT : le robot REJETTE silencieusement toute transition qui ne figure pas dans
le graphe ci-dessus pour son etat COURANT (voir logs robotics.service :
"Transition from [X] to [Y] rejected by runner"). Ce module envoie juste les boutons ;
il ne lit PAS l'etat courant du robot (pas d'abonnement ROS2 /motion/motion_state ici).
C'est a ton Grafcet de respecter ce graphe et de laisser assez de temps a chaque
transition avant d'envoyer la suivante.
------------------------------------------------------------------------------------

"Soulever le carton" (soulever_carton) n'est PAS un etat de cette machine -- c'est une
simulation MuJoCo totalement separee (voir tools/robot_arm_ik/lift_carton.py), lancee
en sous-processus. Elle ne parle pas LCM et n'agit pas sur le run.sh/run_mujoco.sh en
cours.
"""

import argparse
import os
import subprocess
import sys
import time

import lcm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Deplace vers stagiaire_1 le 2026-08-12 : lcm_msgs (paquet ORIGINAL du SDK,
# GamepadKeys.py + SimState.py symlinke dedans) reste dans native_sdk, pas ici.
sys.path.insert(0, "/home/equansrobotic/engineai_robotics_native_sdk/tools/virtual_gamepad")
from lcm_msgs.data import GamepadKeys

LCM_URL = "udpm://239.255.76.67:7667?ttl=1"
CHANNEL = "virtual_gamepad/gamepad_keys"
RATE_HZ = 20                                                             
HOLD_SECONDS = 0.5                                                                                  

BUTTON_INDEX = {
    "LB": 0, "RB": 1, "A": 2, "B": 3, "X": 4, "Y": 5,
    "BACK": 6, "START": 7, "UP": 8, "DOWN": 9, "LEFT": 10, "RIGHT": 11,
}
ANALOG_INDEX = {
    "LEFT_STICK_X": 2, "LEFT_STICK_Y": 3, "RIGHT_STICK_X": 4, "RIGHT_STICK_Y": 5,
}

                                                                                                 
STATE_KEYS = {
    "idle": ["LB", "START"],
    "passive": ["LB", "RB"],
    "pd_stand": ["LB", "A"],
    "walk": ["LB", "B"],
    "rl_lab": ["LB", "X"],
    "dance": ["RB", "B"],
}

                                                                             
ALLOWED_TRANSITIONS = {
    "idle": {"passive"},
    "passive": {"idle", "pd_stand"},
    "pd_stand": {"passive", "walk", "rl_lab", "dance"},
    "walk": {"passive", "pd_stand", "rl_lab", "dance"},
    "rl_lab": {"passive", "pd_stand", "walk", "dance"},
    "dance": {"passive", "pd_stand", "walk"},
}

ROBOT_ARM_IK_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot_arm_ik")
)
LIFT_CARTON_VENV_PYTHON = os.path.join(ROBOT_ARM_IK_DIR, ".venv", "bin", "python3")
LIFT_CARTON_SCRIPT = os.path.join(ROBOT_ARM_IK_DIR, "lift_carton.py")


class PM01Gamepad:
    """Un objet = une connexion LCM. Utilisation typique :

        with PM01Gamepad() as gp:
            gp.passive()
            time.sleep(0.5)
            gp.stand()
            time.sleep(2.0)
            gp.walk()
            gp.set_walk_velocity(forward=0.3, duration=3.0)
            gp.passive()  # arret
    """

    def __init__(self, lcm_url=LCM_URL):
        self._lcm = lcm.LCM(lcm_url)
        self._analog = [0.0] * 6

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass                                                                   

    def send_raw(self, digital_names=(), hold_seconds=HOLD_SECONDS, rate_hz=RATE_HZ):
        """Primitive bas niveau : publie une combinaison de boutons pendant `hold_seconds`,
        a `rate_hz`, avec les valeurs de stick actuelles (voir set_stick)."""
        digital = [0] * 12
        for name in digital_names:
            if name not in BUTTON_INDEX:
                raise ValueError(f"Bouton inconnu : {name} (valides : {list(BUTTON_INDEX)})")
            digital[BUTTON_INDEX[name]] = 1
        n = max(1, round(hold_seconds * rate_hz))
        period = 1.0 / rate_hz
        for _ in range(n):
            msg = GamepadKeys()
            msg.timestamp = int(time.time() * 1_000_000)
            msg.digital_states = digital
            msg.analog_states = list(self._analog)
            self._lcm.publish(CHANNEL, msg.encode())
            time.sleep(period)

    def send_state(self, state_name, hold_seconds=HOLD_SECONDS):
        """Envoie la combinaison de touches associee a `state_name` (voir STATE_KEYS)."""
        if state_name not in STATE_KEYS:
            raise ValueError(f"Etat inconnu : {state_name} (valides : {list(STATE_KEYS)})")
        self.send_raw(STATE_KEYS[state_name], hold_seconds=hold_seconds)

                                                     
    def idle(self):
        self.send_state("idle")

    def passive(self):
        self.send_state("passive")

    def stand(self):
        """pd_stand -- station debout stabilisee par PD."""
        self.send_state("pd_stand")

    def walk(self):
        self.send_state("walk")

    def rl_lab(self):
        self.send_state("rl_lab")

    def dance(self):
        self.send_state("dance")

    def emergency_stop(self):
        """Retour force a passive depuis n'importe quel etat -- meme mecanisme que le
        combo physique LB+RB (voir README 'Global Safety Mechanism')."""
        self.passive()

    def set_stick(self, left_x=0.0, left_y=0.0, right_x=0.0, right_y=0.0):
        """Fixe les valeurs de stick (persistantes) utilisees par les prochains send_raw/
        send_state, sans rien publier tout de suite. Valeurs attendues dans [-1.0, 1.0]."""
        self._analog[ANALOG_INDEX["LEFT_STICK_X"]] = max(-1.0, min(1.0, left_x))
        self._analog[ANALOG_INDEX["LEFT_STICK_Y"]] = max(-1.0, min(1.0, left_y))
        self._analog[ANALOG_INDEX["RIGHT_STICK_X"]] = max(-1.0, min(1.0, right_x))
        self._analog[ANALOG_INDEX["RIGHT_STICK_Y"]] = max(-1.0, min(1.0, right_y))

    def set_walk_velocity(self, forward=0.0, lateral=0.0, turn=0.0, duration=1.0):
        """Pousse une commande de vitesse pendant `duration` secondes (etat walk deja actif
        requis -- appelle walk() avant), puis remet les sticks a zero. Unites/echelle non
        documentees officiellement -- testees empiriquement plutot que garanties par EngineAI.

        Mapping verifie contre src/runner/rl_walking_example/src/rl_walking_example_runner.cc
        (celui qui consomme reellement ces valeurs) -- PAS Left Stick Y=forward comme le
        laisserait penser une correspondance manette "intuitive" :
            command_.x() (avant/arriere) = gamepad->LeftStick_X
            command_.y() (lateral)       = gamepad->LeftStick_Y
            command_.z() (rotation)      = gamepad->RightStick_Y   (pas RightStick_X)
        Et virtual_gamepad_input_adapter.cc NEGATIVE LeftStick_Y et RightStick_Y par rapport
        au stick brut (analog_states[3]/[5]) mais PAS LeftStick_X -- d'ou les signes
        asymetriques ci-dessous pour que forward/lateral/turn positifs restent intuitifs
        cote appelant."""
        self.set_stick(left_x=forward, left_y=-lateral, right_y=-turn)
        self.send_raw([], hold_seconds=duration)
        self.set_stick()                                                

    def soulever_carton(self, hold_seconds=3.0, **lift_kwargs):
        """Lance tools/robot_arm_ik/lift_carton.py en sous-processus BLOQUANT et attend
        la fin. C'EST UNE SIMULATION MUJOCO SEPAREE (pas de LCM, pas de lien avec l'etat
        idle/passive/walk/... ci-dessus) -- voir le docstring du module pour le detail.
        `hold_seconds` : combien de temps la fenetre reste ouverte apres la levee avant
        de se fermer toute seule (necessaire pour un usage scripte -- sinon elle reste
        ouverte indefiniment comme en usage interactif). `lift_kwargs` : voir
        lift_carton.py --help (pinch_x, pinch_y, pinch_z, squeeze_y, lift_z,
        approach_duration, squeeze_duration, lift_duration).
        """
        args = [LIFT_CARTON_SCRIPT, "--hold-seconds", str(hold_seconds)]
        for key, value in lift_kwargs.items():
            args += [f"--{key.replace('_', '-')}", str(value)]
        result = subprocess.run(
            [LIFT_CARTON_VENV_PYTHON] + args,
            capture_output=True, text=True,
        )
        print(result.stdout, end="")
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr, end="")
        return result.returncode == 0


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "action",
        choices=["idle", "passive", "stand", "walk", "rl_lab", "dance", "emergency_stop", "soulever_carton"],
        help="Action a executer une fois, puis quitter.",
    )
    parser.add_argument("--hold-seconds", type=float, default=HOLD_SECONDS,
                         help="Duree de maintien de la combinaison de touches (defaut 0.5s, "
                              "ou duree de maintien post-levee pour soulever_carton, defaut 3.0s).")
    args = parser.parse_args()

    with PM01Gamepad() as gp:
        if args.action == "soulever_carton":
            gp.soulever_carton(hold_seconds=args.hold_seconds if args.hold_seconds != HOLD_SECONDS else 3.0)
        else:
            getattr(gp, args.action)()
    print(f"[INFO] Action '{args.action}' envoyee.")


if __name__ == "__main__":
    _cli()
