"""Fait marcher le PM01 REEL en publiant /motion/body_vel_cmd (BodyVelCmd), l'API ROS2
officielle de controle de vitesse corporelle -- REMPLACE walk_real.py (canal LCM
virtual_gamepad/gamepad_keys), confirme MORT sur le robot reel le 2026-08-27/28 : voir le
bandeau d'avertissement en tete de walk_real.py pour le detail complet de la confirmation
(aucun abonne sur le multicast, input_command_arbiter_runner n'enregistre jamais
virtual_gamepad comme source cote reel, contrairement a la simu).

Topic confirme (`ros2 topic info -v /motion/body_vel_cmd`, 2026-08-27) :
interface_protocol/msg/BodyVelCmd, UN SEUL abonne live -- locomotion_interface_node (le
meme node qui gere deja /motion/joint_override_command, utilise avec succes par
levee.py). `linear_velocity=[x,y]` en VRAIS m/s (PAS l'echelle -1..1 de l'ancien
stick manette -- ne pas reutiliser les anciennes valeurs "prudentes" de walk_real.py,
elles ne veulent plus rien dire ici), `yaw_velocity` en rad/s.

Sequence (main() / marcher()) :
  1. ensure_motion_state(node, WALK_MOTION_STATE) -- bascule lower_body_balance ->
     rl_terrain (ou detour), reutilise le meme mecanisme que levee.py (voir
     motion_state.py, factorise le 2026-08-28 pour etre partage par les deux scripts --
     rclpy.init() ne doit avoir lieu qu'UNE SEULE fois par process).
  2. Publie BodyVelCmd a RATE_HZ (100Hz, meme frequence que l'exemple officiel EngineAI
     body_velocity_control_example.py) pendant `duration` secondes, a (forward, lateral,
     turn) constants.
  3. Publie explicitement une vitesse ZERO pendant quelques cycles avant de rebasculer --
     NECESSAIRE MEME SI /motion/body_vel_cmd s'avere avoir un watchdog qui coupe tout seul
     (comportement NON VERIFIE au 2026-08-28) : si au contraire il n'y a AUCUN watchdog et
     que la derniere vitesse publiee reste active indefiniment, ne pas le faire laisserait
     le robot marcher tout seul apres la fin du script.
  4. ensure_motion_state(node, "lower_body_balance") -- necessaire avant d'enchainer la
     levee (levee.py suppose ce mode).

Inconnue NON VERIFIEE (a confirmer au premier test reel, ne bloque pas ce code) : la
transition directe rl_terrain -> lower_body_balance existe-t-elle, ou faut-il un detour
(ex: par pd_stand, comme le detour par defaut de ensure_motion_state) ? Le premier essai
reel (motion-state-switch-only, forward=0.0) sert justement a repondre a cette question.

AUCUNE SIMULATION -- ceci envoie de vraies commandes de marche au robot.

Securite : commencer par une distance/duree TRES faible (DEFAULT_FORWARD_MPS ci-dessous),
harnais tendu, espace degage, quelqu'un pret a couper (bouton "passive" de la telecommande
= arret d'urgence). NE JAMAIS lancer ce script sur le robot reel sans validation explicite
prealable -- voir COMMENT_LANCER_ROBOT_REEL.txt pour la sequence de test par etapes.
"""
import argparse
import os
import sys
import time

import rclpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_state import ensure_motion_state

WALK_MOTION_STATE = "rl_terrain"   # confirme disponible depuis lower_body_balance
                                    # (available_transition_motions, 2026-08-27) --
                                    # alternatives possibles : rl_basic, rl_amp, passive
RATE_HZ = 100.0                     # meme frequence que l'exemple officiel EngineAI
                                     # body_velocity_control_example.py
ZERO_PUBLISH_CYCLES = 10            # ~0.1s de vitesse zero avant de rebasculer, cf.
                                     # point 3 du docstring module (watchdog non verifie)
MOTION_STATE_TIMEOUT = 3.0
DEFAULT_FORWARD_MPS = 0.45          # 2026-08-28, premier essai reel : 0.12 CONFIRME TROP
                                     # FAIBLE -- le robot bascule bien en rl_terrain (leger
                                     # ajustement de posture visible) mais ne declenche aucun
                                     # pas, avec duration=1.0s ET duration=3.0s (donc pas un
                                     # probleme de duree trop courte -- ecarte). 0.45 CONFIRME
                                     # MARCHE (premier pas reel observe). Le seuil exact entre
                                     # 0.12 et 0.45 n'a pas ete cherche -- 0.45 est la valeur
                                     # la plus basse testee qui fonctionne, pas forcement le
                                     # minimum. RIEN A VOIR avec l'ancien defaut --forward 0.3
                                     # de walk_real.py/orchestrateur.py (echelle -1..1 du
                                     # stick manette, pas des m/s reels).


def _checkpoint(message, confirm):
    print(f"[ETAPE] {message}", flush=True)
    if confirm:
        input("        Verifie le robot, puis Entree pour continuer (Ctrl+C pour arreter)... ")


def publier_vitesse(pub, node, forward: float, lateral: float, yaw: float,
                     duration: float, rate_hz: float = RATE_HZ, dry_run: bool = False):
    """Publie BodyVelCmd(linear_velocity=[forward, lateral], yaw_velocity=yaw) a `rate_hz`
    pendant `duration` secondes. dry_run=True n'envoie rien, affiche juste UNE FOIS la
    valeur qui aurait ete publiee (pas d'interpolation ici contrairement a move_arms --
    une commande de vitesse constante n'a rien a montrer a chaque cycle)."""
    from interface_protocol.msg import BodyVelCmd
    from std_msgs.msg import Header

    if dry_run:
        print(f"    [dry-run] BodyVelCmd linear_velocity=[{forward}, {lateral}] "
              f"yaw_velocity={yaw} pendant {duration}s a {rate_hz}Hz")
        return

    period = 1.0 / rate_hz
    t0 = time.time()
    while time.time() - t0 < duration:
        msg = BodyVelCmd()
        msg.header = Header()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "body"
        msg.linear_velocity = [float(forward), float(lateral)]
        msg.yaw_velocity = float(yaw)
        pub.publish(msg)
        time.sleep(period)


def marcher(node, forward: float, lateral: float, turn: float, duration: float,
            rate_hz: float = RATE_HZ, motion_state_timeout: float = MOTION_STATE_TIMEOUT,
            walk_motion_state: str = WALK_MOTION_STATE, dry_run: bool = False,
            skip_motion_state: bool = False) -> bool:
    """Sequence complete : ensure_motion_state -> walk_motion_state, publier_vitesse
    pendant `duration`, vitesse zero explicite, ensure_motion_state -> lower_body_balance.
    Retourne True si les deux bascules d'etat ont reussi (False sinon -- l'appelant decide
    s'il enchaine quand meme sur la levee ou s'arrete). Cree/detruit son propre publisher
    BodyVelCmd sur `node` a chaque appel (meme convention que ensure_motion_state pour sa
    propre subscription/publisher -- reutilisable sans etat residuel)."""
    if dry_run:
        print(f"[ETAPE] (dry-run) bascule vers {walk_motion_state}...")
        publier_vitesse(None, node, forward, lateral, turn, duration, rate_hz, dry_run=True)
        print(f"[ETAPE] (dry-run) vitesse zero puis retour en lower_body_balance.")
        return True

    if not skip_motion_state:
        ok = ensure_motion_state(node, walk_motion_state, timeout=motion_state_timeout)
        if not ok:
            print(f"[ERREUR] impossible de passer en {walk_motion_state} -- "
                  "marche annulee.", flush=True)
            return False

    from interface_protocol.msg import BodyVelCmd
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    vel_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                          durability=DurabilityPolicy.VOLATILE)
    pub = node.create_publisher(BodyVelCmd, "/motion/body_vel_cmd", vel_qos)

    print(f"[ETAPE] marche -- forward={forward} lateral={lateral} turn={turn} "
          f"duree={duration}s", flush=True)
    publier_vitesse(pub, node, forward, lateral, turn, duration, rate_hz)

    print("[ETAPE] vitesse zero...", flush=True)
    period = 1.0 / rate_hz
    for _ in range(ZERO_PUBLISH_CYCLES):
        msg = BodyVelCmd()
        msg.linear_velocity = [0.0, 0.0]
        msg.yaw_velocity = 0.0
        pub.publish(msg)
        time.sleep(period)

    node.destroy_publisher(pub)

    if not skip_motion_state:
        ok = ensure_motion_state(node, "lower_body_balance", timeout=motion_state_timeout)
        if not ok:
            print("[ERREUR] impossible de repasser en lower_body_balance apres la marche.",
                  flush=True)
            return False

    print("[INFO] Marche terminee.", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forward", type=float, default=DEFAULT_FORWARD_MPS,
                         help="Vitesse avant en VRAIS m/s (defaut prudent, jamais valide "
                              "sur ce robot).")
    parser.add_argument("--lateral", type=float, default=0.0, help="Vitesse laterale en m/s.")
    parser.add_argument("--turn", type=float, default=0.0, help="Vitesse de lacet en rad/s.")
    parser.add_argument("--duration", type=float, default=1.0, help="Duree de la marche en secondes.")
    parser.add_argument("--rate-hz", type=float, default=RATE_HZ)
    parser.add_argument("--walk-motion-state", type=str, default=WALK_MOTION_STATE)
    parser.add_argument("--skip-motion-state", action="store_true",
                         help="Suppose que le robot est deja dans un etat de marche.")
    parser.add_argument("--dry-run", action="store_true",
                         help="N'envoie rien au robot -- affiche juste ce qui serait publie.")
    parser.add_argument("--no-confirm", action="store_true",
                         help="Pas de pause interactive avant la marche -- jamais pour un premier essai.")
    args = parser.parse_args()

    node = None
    if not args.dry_run:
        rclpy.init()
        node = rclpy.create_node("marche")

    try:
        _checkpoint(
            f"marche -- forward={args.forward}m/s lateral={args.lateral}m/s "
            f"turn={args.turn}rad/s duree={args.duration}s (etat cible : "
            f"{args.walk_motion_state})",
            not args.no_confirm,
        )
        marcher(node, args.forward, args.lateral, args.turn, args.duration,
                rate_hz=args.rate_hz, walk_motion_state=args.walk_motion_state,
                dry_run=args.dry_run, skip_motion_state=args.skip_motion_state)
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
