
"""Variante EXPERIMENTALE de choreo_native.py : PAS de pivot par override
d'articulations -- au lieu de tourner sur place via JointOverrideCommand, le
robot pivote en MARCHANT (gp.walk(), vraie politique RL, commande `turn`),
une fois le carton en main, pour degager du podium.

ATTENTION -- ceci reteste un scenario documente comme FATAL dans une session
anterieure (2026-08-10, voir memoire projet) : re-entrer en `walk` pendant
que les bras sont tenus par JointOverrideCommand a fait tomber le robot a
chaque essai a l'epoque. Ce qui a change depuis (2026-08-11, cette session) :
--walk-stance (lift_carton_real.py) corrige "tombe des la prise du carton"
en pd_stand, MAIS des que gp.walk() est appele la politique de marche RL
reprend ENTIEREMENT le controle des jambes -- le fix jambes ne s'applique
plus a ce moment-la.

Teste isolement le 2026-08-11 (SANS les bras tenus, juste gp.walk() +
turn) :
  - turn=0.4 : ~4deg en 5s -- stable mais tres lent.
  - turn=1.0 : ~-117deg en 5s (~23deg/s), stable en walk. MAIS transition
    walk->pd_stand instable si l'arret est trop brusque (chute a 1s d'arret
    a vitesse nulle) -- stable avec 3s d'arret a vitesse nulle avant
    pd_stand (le robot derive encore un peu pendant l'arret -- ~40deg de
    plus observes -- pas juste du moment angulaire residuel a la coupure).
Valeurs ci-dessous calibrees sur ce test SANS les bras -- a revalider AVEC
le carton en main, comme pour le recul (qui degrade fortement des que les
bras sont tenus, cf RETREAT_SPEED ci-dessous garde en reference).

Gains bras adoucis (--hold-stiffness 30 --hold-damping 0.3, ~8x plus souple
que pd_stand) actifs pendant le pivot pour la meme raison que le recul
initial : matcher la souplesse attendue par la politique de marche.
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
STANDOFF = 0.55
PINCH_X_MAX = 0.38
PINCH_Y = 0.28  # 2026-08-11 : grande amplitude d'approche pour degager la plateforme du
                # podium (demi-largeur 0.17m, plus large que le carton 0.0955m) -- le
                # serrage (squeeze_y, defaut de lift_carton_real.py) referme ensuite jusqu'a
                # la vraie largeur du carton.

DOCKER_CONTAINER = "engineai_robotics_env"
LIFT_APPROACH_S = 2.0
LIFT_SQUEEZE_S = 1.5
LIFT_LIFT_S = 3.0

# Calibre sans les bras (2026-08-11) : turn=1.0 -> ~23deg/s en walk.
TURN_MAGNITUDE = 1.0
TURN_TARGET_DEG = 90.0
TURN_RATE_DEG_S = 23.0
TURN_DURATION_S = TURN_TARGET_DEG / TURN_RATE_DEG_S      # ~3.9s
TURN_SETTLE_S = 3.0         # vitesse nulle tenue avant pd_stand -- 1s a fait tomber le
                             # robot en test sans bras (moment angulaire residuel), 3s stable.
SETTLE_BEFORE_WALK_S = 2.0  # laisse le carton se stabiliser en main avant de remarcher
SETTLE_AFTER_WALK_S = 3.0   # laisse pd_stand se stabiliser avant le relachement

LIFT_DISABLED = True  # 2026-08-11 : tombe au moment de pivoter avec le carton en main --
                       # isole marche avant + pivot en marchant seuls, sans la levee.

# Couvre : levee (via lift_carton_real.py) + attente avant remarche + pivot + stabilisation.
HOLD_SECONDS = (SETTLE_BEFORE_WALK_S + 1.5 + TURN_DURATION_S + TURN_SETTLE_S
                + SETTLE_AFTER_WALK_S + 2.0)


def _launch_lift_carton(pinch_x, pinch_yaw_offset=0.0):
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
        f"--lift-duration {LIFT_LIFT_S} --hold-seconds {HOLD_SECONDS:.1f} "
        "--pivot-degrees 0 "
        "--hold-stiffness 30 --hold-damping 0.3 --hold-weight 1.0 --no-confirm"
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
            print("[choreo][ATTENTION] Cible non atteinte.")

        x, y, yaw = listener.pose()
        pinch_x = math.hypot(CARTON_X - x, CARTON_Y - y)
        ideal_heading = math.atan2(CARTON_Y - y, CARTON_X - x)
        pinch_yaw_offset = ideal_heading - yaw
        print(f"[choreo] Position finale : x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f}deg "
              f"-- distance au carton {pinch_x:.3f}m -- correction de cap "
              f"{math.degrees(pinch_yaw_offset):.1f}deg", flush=True)

        lift_proc = None
        if not LIFT_DISABLED:
            lift_proc = _launch_lift_carton(pinch_x, pinch_yaw_offset)
            print("[choreo] Attente fin approche/serrage/levee...", flush=True)
            time.sleep(LIFT_APPROACH_S + LIFT_SQUEEZE_S + LIFT_LIFT_S + SETTLE_BEFORE_WALK_S)
        else:
            print("[choreo] Levee DESACTIVEE (LIFT_DISABLED).", flush=True)
            time.sleep(1.0)

        x0, y0, yaw0 = listener.pose()
        print(f"[choreo] Pivot en marchant (turn={TURN_MAGNITUDE}, {TURN_DURATION_S:.1f}s vise "
              f"{TURN_TARGET_DEG:.0f}deg) -- avant : x={x0:.3f} y={y0:.3f} "
              f"yaw={math.degrees(yaw0):.1f}deg", flush=True)
        gp.walk()
        time.sleep(1.5)
        gp.set_walk_velocity(forward=0.0, lateral=0.0, turn=TURN_MAGNITUDE,
                              duration=TURN_DURATION_S)
        x1, y1, yaw1 = listener.pose()
        print(f"[choreo] Apres pivot actif (encore en walk) : x={x1:.3f} y={y1:.3f} "
              f"yaw={math.degrees(yaw1):.1f}deg "
              f"(delta_yaw={math.degrees(yaw1 - yaw0):.1f}deg)", flush=True)

        gp.set_walk_velocity(forward=0.0, lateral=0.0, turn=0.0, duration=TURN_SETTLE_S)
        x2, y2, yaw2 = listener.pose()
        print(f"[choreo] Apres stabilisation vitesse nulle : x={x2:.3f} y={y2:.3f} "
              f"yaw={math.degrees(yaw2):.1f}deg "
              f"(delta_yaw_total={math.degrees(yaw2 - yaw0):.1f}deg)", flush=True)

        gp.stand()
        time.sleep(SETTLE_AFTER_WALK_S)
        x3, y3, yaw3 = listener.pose()
        print(f"[choreo] Apres retour pd_stand : x={x3:.3f} y={y3:.3f} "
              f"yaw={math.degrees(yaw3):.1f}deg", flush=True)

    if lift_proc is not None:
        print("[choreo] Attente fin de la sequence de levee (maintien + relachement)...", flush=True)
        ok = lift_proc.wait() == 0
        print(f"[choreo] {'Termine (avec pivot en marchant).' if ok else 'La levee/pivot a echoue.'}")
    else:
        print("[choreo] Termine (pivot en marchant seul, sans levee).")


if __name__ == "__main__":
    main()
