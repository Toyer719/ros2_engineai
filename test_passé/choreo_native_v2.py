
"""Variante EXPERIMENTALE de choreo_native.py : le robot s'arrete plus loin du
podium (pour degager de l'espace pour un pivot du corps entier) et pivote avec
les hanches en plus du buste (--hip-pivot-fraction 1.0, meme sens que le
buste -- un bug de signe faisait tourner les hanches a l'oppose du buste,
corrige le 2026-08-11 dans lift_carton_real.py).

Geometrie reelle du podium (assets/resource/pm01_edu_carton.xml) : plateau
0.38x0.34m (demi-tailles 0.19/0.17) a hauteur 0.63m, pied cylindrique fin
(rayon 0.06m). Carton 0.35x0.30x0.30m pose dessus. A la distance d'origine
(~0.30m du centre du podium une fois la marche terminee), le buste/carton
tenu chevauche quasiment le plateau -- d'ou le blocage observe par
l'utilisateur en pivotant. STANDOFF vise maintenant a atterrir pres de
0.40-0.42m (le vrai maximum de portee du bras, mesure par IK directe sur
LEFT_CHAIN/solve_ik : erreur quasi nulle jusqu'a ~0.40m puis croissante --
24mm a 0.42m, 59mm a 0.46m) pour dégager le plateau tout en gardant une
prise a peu pres correcte -- pas encore reverifie en simulation apres ce
changement, a valider par telemetrie complete (walk arrival + pivot +
release) avant de faire confiance au resultat.

Stabilite du pivot hanches : marginal/probabiliste, pas un simple oui/non.
Tests 2026-08-11 sur cette meme copie (fraction=1.0, avant la correction du
signe) : 3 essais complets, 3 reussites (pivot+maintien+depivot+relachement
stables, z~0.82m tout du long). Tests isoles anterieurs (lift_carton_real.py
seul, sans marche prealable) : 2 echecs sur 2 a des fractions differentes.
Ne pas conclure "stable" ou "instable" sur un seul essai -- toujours verifier
par telemetrie sim_state (hauteur/roll/pitch) bien au-dela du message
"Termine" avant de faire confiance a un resultat.
"""

import math
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamepad_api import PM01Gamepad
from walk_to_xy import SimStateListener, walk_to_xy

STAGIAIRE1_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
NATIVE_SDK_DIR = "/home/equansrobotic/engineai_robotics_native_sdk"
LIFT_SCRIPT = os.path.join(STAGIAIRE1_DIR, "tools", "joint_angle_commander", "lift_carton_real.py")

CARTON_X, CARTON_Y = 2.2, 0.0
ARM_REACH_MIN = 0.25
# La marche depasse systematiquement sa cible de ~0.13-0.18m pendant la
# stabilisation post walk_to_xy (inertie de la politique RL, meme a vitesse
# commandee nulle -- voir "Apres stabilisation en walk" dans les logs).
# IMPORTANT (2026-08-11) : solve_ik bascule sur une branche de solution
# anormale (coude/ELBOW_YAW saute de ~-7deg a ~+75deg) entre x=0.39m et
# x=0.40m -- verifie directement (LEFT_CHAIN/solve_ik) et confirme en sim :
# a pinch_x=0.419m le robot tombe pendant le SERRAGE, avant meme la levee,
# et desserrer (squeeze_y 0.14->0.15) NE CHANGE RIEN -- la cause est la
# bascule IK, pas la force de prise. PINCH_X_MAX doit donc rester < 0.39m,
# PAS ~0.40-0.42m comme suppose precedemment (l'erreur de reach residuelle
# seule sous-estimait le risque -- ce n'est pas une degradation progressive,
# c'est un seuil dur).
STANDOFF = 0.55
PINCH_X_MAX = 0.38
PINCH_Y = 0.28  # 2026-08-11 : grande amplitude d'approche pour degager la plateforme du
                # podium (demi-largeur 0.17m, plus large que le carton 0.0955m) -- le
                # serrage (squeeze_y, defaut de lift_carton_real.py) referme ensuite jusqu'a
                # la vraie largeur du carton. ATTENTION avant ce fix, cette constante etait a
                # 0.211, calculee avec les mauvaises dimensions (191mm pris pour une demi-
                # largeur alors que c'etait la largeur complete) et JAMAIS resynchronisee avec
                # lift_carton_real.py apres la correction -- verifier la coherence des deux
                # fichiers si cette valeur est retouchee.

DOCKER_CONTAINER = "engineai_robotics_env"
LIFT_APPROACH_S = 2.0
LIFT_SQUEEZE_S = 1.5
LIFT_LIFT_S = 3.0
STEP_BACK_HOLD_S = 12.0
PIVOT_DEGREES = 60.0
PIVOT_DURATION_S = 3.0
HIP_PIVOT_FRACTION = 1.0


def _launch_lift_carton(pinch_x, pinch_yaw_offset=0.0, hold_seconds=STEP_BACK_HOLD_S):
    pinch_x = max(ARM_REACH_MIN, min(PINCH_X_MAX, pinch_x))
    print(f"[choreo] Levee des bras -- pinch_x={pinch_x:.3f}m (calcule depuis la position finale) "
          f"pinch_yaw_offset={math.degrees(pinch_yaw_offset):.1f}deg")
    inner_cmd = (
        f"cd {NATIVE_SDK_DIR} && "
        "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=1 && "
        "source /opt/ros/humble/setup.bash && "
        "source build/ros2_env/install/local_setup.bash && "
        f"python3 {LIFT_SCRIPT} --pinch-x {pinch_x:.3f} --pinch-y {PINCH_Y} "
        f"--pinch-yaw-offset {pinch_yaw_offset:.4f} "
        f"--approach-duration {LIFT_APPROACH_S} --squeeze-duration {LIFT_SQUEEZE_S} "
        f"--lift-duration {LIFT_LIFT_S} --hold-seconds {hold_seconds:.1f} "
        f"--pivot-degrees {PIVOT_DEGREES} --pivot-duration {PIVOT_DURATION_S} "
        f"--hip-pivot-fraction {HIP_PIVOT_FRACTION} "
        "--hold-weight 1.0 --no-confirm"
    )
    if shutil.which("docker"):
        return subprocess.Popen(["docker", "exec", DOCKER_CONTAINER, "bash", "-c", inner_cmd])
    print("[choreo] 'docker' introuvable -- deja dans le conteneur, execution directe.")
    return subprocess.Popen(["bash", "-c", inner_cmd])

def main():
    listener = SimStateListener()
    print("[choreo] Abonnement au canal LCM 'sim_state'...", flush=True)
    if not listener.wait_for_first_message():
        print("[choreo][ERREUR] Aucun message sur 'sim_state' -- run_mujoco.sh tourne-t-il ?",
              file=sys.stderr)
        sys.exit(1)

    with PM01Gamepad() as gp:
        gp.stand()
        time.sleep(10.0)

        gp.walk()
        time.sleep(1.5)

        standoff = STANDOFF
        heading = math.atan2(CARTON_Y - 0.0, CARTON_X - 0.0)
        target_x = CARTON_X - standoff
        target_y = CARTON_Y
        print(f"[choreo] Marche (vraie, RL) vers ({target_x:.2f}, {target_y:.2f})...", flush=True)
        reached = walk_to_xy(gp, listener, target_x, target_y, tolerance=0.08, forward_speed=0.5,
                              turn_gain=1.2, max_turn=0.4, command_period=0.3, timeout=30.0)

        print("[choreo] Stabilisation a vitesse nulle (encore en walk) avant pd_stand...", flush=True)
        gp.set_walk_velocity(forward=0.0, lateral=0.0, turn=0.0, duration=2.0)
        x, y, yaw = listener.pose()
        print(f"[choreo] Apres stabilisation en walk : x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f}deg")

        gp.stand()
        time.sleep(3.0)

        if not reached:
            print("[choreo][ATTENTION] Cible non atteinte -- levee quand meme tentee, "
                  "mais la distance au carton est probablement fausse.")

        x, y, yaw = listener.pose()
        pinch_x = math.hypot(CARTON_X - x, CARTON_Y - y)
        # Cap ideal pour regarder le carton pile en face, moins le cap reel du robot --
        # la marche ne s'arrete jamais parfaitement alignee (quelques deg d'ecart typiques),
        # negligeable sur l'ancien gros carton mais suffisant pour rater le centre du
        # nouveau carton, plus etroit (2026-08-11, voir --pinch-yaw-offset).
        ideal_heading = math.atan2(CARTON_Y - y, CARTON_X - x)
        pinch_yaw_offset = ideal_heading - yaw
        print(f"[choreo] Position finale : x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f}deg "
              f"-- distance au carton {pinch_x:.3f}m -- correction de cap "
              f"{math.degrees(pinch_yaw_offset):.1f}deg", flush=True)

        lift_proc = _launch_lift_carton(pinch_x, pinch_yaw_offset)
        print("[choreo] Attente fin approche/serrage/levee...", flush=True)
        time.sleep(LIFT_APPROACH_S + LIFT_SQUEEZE_S + LIFT_LIFT_S + 0.5)

        x0, y0, yaw0 = listener.pose()
        print(f"[choreo] Pivot corps entier (bras toujours en prise, pieds fixes en pd_stand) -- "
              f"avant : x={x0:.3f} y={y0:.3f} yaw={math.degrees(yaw0):.1f}deg", flush=True)
        time.sleep(PIVOT_DURATION_S + 2.0)
        x1, y1, yaw1 = listener.pose()
        print(f"[choreo] Apres pivot : x={x1:.3f} y={y1:.3f} yaw={math.degrees(yaw1):.1f}deg "
              f"(deplacement={math.hypot(x1 - x0, y1 - y0):.3f}m, "
              f"delta_yaw={math.degrees(yaw1 - yaw0):.1f}deg)", flush=True)

    print("[choreo] Attente fin de la sequence de levee (maintien + relachement)...", flush=True)
    ok = lift_proc.wait() == 0
    print(f"[choreo] {'Termine (avec pivot).' if ok else 'La levee/pivot a echoue -- voir sortie ci-dessus.'}")


if __name__ == "__main__":
    main()
