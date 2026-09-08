                      
import argparse
import math
import os
import sys
import threading
import time

import lcm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/equansrobotic/engineai_robotics_native_sdk/tools/virtual_gamepad")
from gamepad_api import PM01Gamepad
from lcm_msgs.data import SimState

LCM_URL = "udpm://239.255.76.67:7667?ttl=1"
SIM_STATE_CHANNEL = "sim_state"


def _yaw_from_quaternion(w, x, y, z):
    """Cap (rotation autour de Z) a partir du quaternion (w,x,y,z) de base_link_quaternion
    -- copie directe des donnees MuJoCo (meme convention w,x,y,z que qpos)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(a):
    """Ramene un angle dans [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class SimStateListener:
    """S'abonne au canal LCM `sim_state` (verite terrain MuJoCo, publiee par
    simulation/mujoco/src/lcm_interface a 500Hz) et garde le dernier message recu.
    N'existe et ne fonctionne qu'avec la SIMULATION en cours d'execution -- ce canal
    n'est pas publie sur le robot reel."""

    def __init__(self, lcm_url=LCM_URL):
        self._lc = lcm.LCM(lcm_url)
        self._lc.subscribe(SIM_STATE_CHANNEL, self._on_message)
        self._latest = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _on_message(self, channel, data):
        state = SimState.decode(data)
        with self._lock:
            self._latest = state

    def _spin(self):
        while True:
            self._lc.handle()

    def wait_for_first_message(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._latest is not None:
                    return True
            time.sleep(0.05)
        return False

    def pose(self):
        """(x, y, yaw) courants, ou None si aucun message recu."""
        with self._lock:
            state = self._latest
        if state is None:
            return None
        x, y = state.base_link_position[0], state.base_link_position[1]
        w, qx, qy, qz = state.base_link_quaternion
        return x, y, _yaw_from_quaternion(w, qx, qy, qz)


def walk_to_xy(gp, listener, target_x, target_y, tolerance=0.08, forward_speed=0.2,
               turn_gain=1.2, max_turn=0.4, slow_turn_threshold=0.5, command_period=0.4,
               slow_radius=0.5, min_forward_speed=0.12, timeout=60.0):
    """Boucle fermee : recalcule distance/cap vers (target_x, target_y) a chaque cycle,
    pousse une commande de vitesse courte (command_period) vers set_walk_velocity(), et
    s'arrete quand la distance est sous `tolerance` ou apres `timeout` secondes.

    Ralentit lineairement en dessous de `slow_radius` (jusqu'a `min_forward_speed`) --
    s'arreter net depuis forward_speed plein en sortant de `walk` fait trebucher le robot
    (observe empiriquement : un arret a forward=0.5 a produit un deplacement incontrole
    de plus d'1m apres coup). Ariver lentement rend la transition walk->pd_stand propre."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        pose = listener.pose()
        if pose is None:
            print("[ATTENTE] pas encore de message sim_state -- la simulation tourne-t-elle ?")
            time.sleep(0.5)
            continue
        x, y, yaw = pose
        dx, dy = target_x - x, target_y - y
        dist = math.hypot(dx, dy)
        if dist < tolerance:
            print(f"[OK] Cible atteinte : ({x:.3f}, {y:.3f}), distance restante {dist:.3f}m")
            return True

        heading_error = _wrap_angle(math.atan2(dy, dx) - yaw)
        turn = max(-max_turn, min(max_turn, turn_gain * heading_error))
                                                                                         
                                                                                           
        if dist < slow_radius:
            ramp = dist / slow_radius
            target_speed = min_forward_speed + ramp * (forward_speed - min_forward_speed)
        else:
            target_speed = forward_speed
                                                                                                
        forward = target_speed if abs(heading_error) < slow_turn_threshold else target_speed * 0.3

        print(f"[MARCHE] pos=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):.1f} deg  "
              f"dist={dist:.3f}m  cap_erreur={math.degrees(heading_error):.1f} deg  "
              f"-> forward={forward:.2f} turn={turn:.2f}")
        gp.set_walk_velocity(forward=forward, turn=turn, duration=command_period)

    print(f"[TIMEOUT] Cible non atteinte apres {timeout}s.")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--x", type=float, required=True, help="Coordonnee X cible, repere monde (m)")
    parser.add_argument("--y", type=float, default=0.0, help="Coordonnee Y cible, repere monde (m)")
    parser.add_argument("--tolerance", type=float, default=0.08, help="Distance d'arret (m)")
    parser.add_argument("--forward-speed", type=float, default=0.2)
    parser.add_argument("--turn-gain", type=float, default=1.2)
    parser.add_argument("--max-turn", type=float, default=0.4)
    parser.add_argument("--command-period", type=float, default=0.4,
                         help="Duree de chaque rafale de commande vitesse (s)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--from-state", choices=["idle", "passive", "pd_stand", "walk"], default="idle",
        help="Etat de DEPART suppose du robot/sim (defaut idle) -- saute les etapes deja "
             "franchies si deja plus loin dans la sequence.",
    )
    parser.add_argument("--no-confirm", action="store_true",
                         help="Pas de pause avant de commencer a marcher.")
    args = parser.parse_args()

    listener = SimStateListener()
    print("[INFO] Abonnement au canal LCM 'sim_state'...", flush=True)
    if not listener.wait_for_first_message():
        print("[ERREUR] Aucun message sur 'sim_state' recu en 5s -- la simulation "
              "(./scripts/run_mujoco.sh pm01_edu_carton) tourne-t-elle ?", file=sys.stderr)
        sys.exit(1)
    x0, y0, yaw0 = listener.pose()
    print(f"[INFO] Position initiale : x={x0:.3f} y={y0:.3f} yaw={math.degrees(yaw0):.1f} deg")

    order = ["idle", "passive", "pd_stand", "walk"]
    start_idx = order.index(args.from_state)

    with PM01Gamepad() as gp:
        if start_idx <= order.index("idle"):
            gp.idle()                                                         
            time.sleep(1.0)
        if start_idx <= order.index("passive"):
            gp.stand()                                    
            time.sleep(2.0)
        if start_idx <= order.index("pd_stand"):
            gp.walk()                                 
            time.sleep(1.0)

        if not args.no_confirm:
            input(f"[PRET] Marche vers ({args.x}, {args.y}) -- Entree pour lancer (Ctrl+C pour arreter)... ")

        walk_to_xy(
            gp, listener, args.x, args.y,
            tolerance=args.tolerance, forward_speed=args.forward_speed, turn_gain=args.turn_gain,
            max_turn=args.max_turn, command_period=args.command_period, timeout=args.timeout,
        )

        gp.stand()                                   
        time.sleep(1.0)

    print("[INFO] Termine -- robot en pd_stand.")


if __name__ == "__main__":
    main()
