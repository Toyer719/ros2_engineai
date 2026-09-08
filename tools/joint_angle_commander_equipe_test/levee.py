"""Fait pincer et soulever un carton REEL par les deux mains du PM01, en pilotant les
articulations bras par surcharge ponderee (Lever / JointOverrideCommand, voir lever.py).
AUCUNE SIMULATION -- ceci envoie de vraies commandes au robot. Reutilise la meme geometrie
de bras (LEFT_CHAIN/RIGHT_CHAIN, solve_ik) que la simulation MuJoCo
(tools/robot_arm_ik/lift_carton.py).

Prerequis :
  - Robot en mode "lower_body_balance" -- seul mode ou /motion/joint_override_command a un
    effet sur les bras (en pd_stand la commande est acceptee mais rien ne bouge). Ce script
    y bascule lui-meme au demarrage (_ensure_motion_state), avec detour automatique par
    pd_stand si besoin.
  - source /opt/ros/humble/setup.bash
    source ~/source/engineai_workspace/install/setup.bash
    export ROS_DOMAIN_ID=69 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=0

Securite : valider d'abord avec --dry-run (rien n'est envoye, angles juste affiches), puis
--only-phase approche seul en observant le robot, avant d'enchainer serrage/levee. Bouton
"passive" de la telecommande = arret d'urgence (rend le robot mou, dernier recours).

Pas de pivot ici (retire le 2026-08-24, jamais teste sur robot reel + instable en simu) --
pour tourner le buste avec le carton en main, voir pivot.py (virtual_gamepad_ros).

2026-09-03 : 2 ajouts sur le robot reel :
  - ELBOW_YAW_ROTATION_DEG : fait pivoter l'avant-bras de 90deg pour attraper le carton avec
    le COTE de la main plutot que la paume. 3 essais avant la version stable :
      1) rotation appliquee APRES le calcul IK de la position -- decalait le point de contact
         reel de plusieurs cm (HAND_OFFSET_LEFT/RIGHT non symetrique) sans le compenser, la
         main visait bien plus profond dans le carton que prevu -> ERR_MOTOR_FAULT au serrage.
      2) hand_offset tourne AVANT l'IK (solve_ik(chain, hand_offset_tourne, ...)) -- semblait
         correct en test isole (PINCH_X=0.35, erreur position=0) mais solve_ik ne vise QUE la
         position (3 DDL, 5 articulations = 2 DDL redondants) : rien n'oblige le solveur a
         mettre la rotation dans le poignet plutot qu'ailleurs. Sur le robot reel a la vraie
         distance (PINCH_X=0.216), le solveur a laisse le poignet quasi droit (qL[-1]=-3.7deg
         au lieu de 90deg, log [DEBUG] poignet) -- pas de rotation visible, ET position reelle
         incoherente avec celle calculee pour un poignet a 90deg -> "trop serre" au serrage.
      3) FIX RETENU : _solve_ik_locked_wrist -- fige le dernier angle (ELBOW_YAW) exactement a
         +-wrist_rotation et ne laisse l'IK resoudre la position qu'avec les 4 autres
         articulations. Poignet tourne alors GARANTI de l'angle demande (plus de hasard du
         solveur), hand_offset redevient HAND_OFFSET_LEFT/RIGHT tel quel (plus besoin de le
         tourner a la main, forward_kinematics applique deja la rotation du dernier joint).
         Verifie numeriquement (hors robot) : erreur de position ~3e-8 aux 4 cibles pince/
         serrage gauche/droite a PINCH_X=0.216, tous les joints dans la limite +-150deg.
         Signe : +90deg gauche/-90deg droite (chaine en miroir, confirme visuellement essai 1).
  - Flexion des genoux (WALK_STANCE_*, _bend_knees/_straighten_knees) : jusqu'ici, levee.py NE
    fléchissait PAS du tout les genoux (contrairement a lift.py cote simu, qui le fait depuis
    le 20/08) -- port direct de ce mecanisme, jamais teste sur le robot reel. Echelle de depart
    volontairement PRUDENTE (WALK_STANCE_SCALE=1.0, la posture "marche au repos" MESUREE telle
    quelle, PAS l'echelle x4.5 utilisee en simu) -- augmenter progressivement via
    --walk-stance-scale si stable, jamais un grand saut.
"""
import argparse
import os
import sys
import time

import numpy as np
import rclpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lever import Lever
from motion_state import ensure_motion_state

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot_arm_ik"))
from lift_carton import (
    LEFT_CHAIN, RIGHT_CHAIN, HAND_OFFSET_LEFT, HAND_OFFSET_RIGHT, solve_ik, ease,
    forward_kinematics,
)

LEFT_JOINT_INDICES = [13, 14, 15, 16, 17]
RIGHT_JOINT_INDICES = [18, 19, 20, 21, 22]

Q_LEFT_HOME = np.array([0.000879, 0.075284, -0.000233, -0.126397, -0.000033])
Q_RIGHT_HOME = np.array([0.000885, -0.075161, 0.000241, -0.126390, 0.000033])

# Posture jambes flechies (2026-09-03, port depuis lift.py/simu -- jamais teste sur le robot
# reel avant ce jour). Memes indices/angles mesures que cote simu.
LEFT_HIP_PITCH_INDEX = 0
RIGHT_HIP_PITCH_INDEX = 6
LEFT_KNEE_PITCH_INDEX = 3
RIGHT_KNEE_PITCH_INDEX = 9
LEFT_ANKLE_PITCH_INDEX = 4
RIGHT_ANKLE_PITCH_INDEX = 10
WALK_STANCE_HIP_PITCH_L = np.radians(-6.9)
WALK_STANCE_HIP_PITCH_R = np.radians(-5.2)
WALK_STANCE_KNEE_L = np.radians(11.7)
WALK_STANCE_KNEE_R = np.radians(10.5)
WALK_STANCE_ANKLE_PITCH_L = np.radians(-4.8)
WALK_STANCE_ANKLE_PITCH_R = np.radians(-5.2)

RATE_HZ = 30

# Geometrie/calibration -- ne se changent pas au lancement, seule --pinch-z (hauteur de
# table) varie en pratique. Regroupees ici plutot qu'en CLI pour eviter d'ecraser par
# erreur une valeur de securite (ex: SQUEEZE_Y trop petit a deja cause un ERR_MOTOR_FAULT).
PINCH_X = 0.216           # 2026-08-27 : recalibre pour un placement du robot a 14.6cm du bord du
                           # podium (bassin -- mesure au bord proche, robot+carton flush avec le
                           # bord du podium), valeur validee dans la simu ROS/docker (walk_to reel
                           # + lift.py, meme geometrie pm01_edu_carton) apres une marche vers le
                           # carton -- PAS ENCORE TESTEE sur le robot reel avec cette distance --
                           # valider par --only-phase approche avant d'enchainer serrage/levee.
PINCH_Y = 0.22            # ecart Y avant serrage (demi-largeur carton = 0.0955m) -- aligne sur la
                           # valeur validee en simu le 2026-08-27 (etait 0.28)
SQUEEZE_Y = 0.090         # 2026-09-03 : retour a la valeur initiale (0.090, prise paume, avant
# tout essai de rotation de poignet) -- essais 0.095 puis 0.105 avec le poignet fige a 90deg
# toujours juges "trop serre" malgre le desserrage progressif de la position -> la cause n'est
# probablement PAS SQUEEZE_Y mais la nouvelle geometrie de coude qu'impose le poignet a 90deg
# (coude a -54.7deg au lieu de -7deg, cf. ELBOW_YAW_ROTATION_DEG ci-dessous) -- possible contact
# avant-bras/poignet contre le carton independant du point vise par la main. Rotation de
# poignet DESACTIVEE par defaut en attendant de trancher (voir ELBOW_YAW_ROTATION_DEG) ; cette
# valeur 0.090 est donc a nouveau utilisee SANS rotation, comme avant le 2026-09-03 (historique
# complet de calibration de SQUEEZE_Y dans l'ancienne version de ce commentaire, cf. git/backup
# si besoin -- 0.1405/0.086/0.05 = valeurs simu non transposables telles quelles au reel,
# 0.095 juge "pas assez serre" le 31/08 SANS rotation de poignet).
LIFT_Z = 0.146
APPROACH_DURATION = 4.0
SQUEEZE_DURATION = 4.0
LIFT_DURATION = 5.0
HOLD_SECONDS = 3.0
RELEASE_RAMP_SECONDS = 4.0   # rampe de poids avant release() -- une coupure instantanee fait
                             # tomber les bras d'un coup vers la pose de la politique de marche
MOTION_STATE_TIMEOUT = 3.0

# 2026-09-03 : angle de pivot de l'avant-bras (dernier joint de chaque chaine, ELBOW_YAW) pour
# attraper le carton avec le cote de la main plutot que la paume -- voir docstring module.
# DESACTIVE PAR DEFAUT (0.0) le 2026-09-03 : le fix _solve_ik_locked_wrist fait bien tourner le
# poignet a 90deg (verifie), mais ca force le coude a -54.7deg (vs -7deg avant) pour compenser --
# geometrie de bras tres differente, correlee avec des essais "trop serre" resistants a
# SQUEEZE_Y (0.095 puis 0.105, tous deux juges trop serres). Hypothese non encore confirmee :
# l'avant-bras/poignet cogne le carton independamment de SQUEEZE_Y a cause de cette nouvelle
# trajectoire de coude. Code garde intact (fonctionnel, poignet exactement fige a l'angle demande
# -- voir _solve_ik_locked_wrist) pour reactivation via --wrist-rotation-deg 90 une fois la
# question du contact avant-bras tranchee (comparer avec/sans rotation, meme SQUEEZE_Y).
ELBOW_YAW_ROTATION_DEG = 0.0

# 2026-09-03 : flexion des genoux pour stabiliser la levee -- voir docstring module.
# Echelle DELIBEREMENT prudente pour un 1er essai reel (1.0 = posture mesuree telle quelle,
# contre 4.5 utilise en simu -- augmenter progressivement via --walk-stance-scale).
WALK_STANCE_SCALE = 1.0
WALK_STANCE_STIFFNESS_SCALE = 1.8
WALK_STANCE_DURATION = 3.0

# kp reduit (vs 250 par defaut dans lever.py) applique aux bras SEULEMENT pendant la levee/
# maintien (seule phase avec une charge reelle soutenue). 250 sature le couple max des
# moteurs de bras (Q25H, 50 N.m) des ~11deg d'erreur sous charge -> ERR_MOTOR_FAULT constate
# le 2026-08-24. 40 partout (essai du 2026-08-25) etait trop faible pour bouger le bras meme
# a vide. 90 = compromis, laisse jusqu'a ~29deg d'erreur avant saturation -- PAS ENCORE
# VALIDE sur le robot reel, a confirmer au prochain test.
# 2026-08-31 : le code valait 25.0 ici, en contradiction avec le commentaire ci-dessus
# (encore plus faible que le 40 deja juge insuffisant) -- corrige a 90 pour matcher la
# valeur reellement decidee. Explique probablement "il leve mal le carton" (retour
# utilisateur du jour) : une rigidite trop faible ne tient pas fermement la prise sous
# le poids du carton pendant la montee. Toujours PAS VALIDE sur le robot reel.
LEVEE_STIFFNESS = 90.0


def _checkpoint(message, confirm):
    print(f"[ETAPE] {message}", flush=True)
    if confirm:
        input("        Verifie le robot, puis Entree pour continuer (Ctrl+C pour arreter)... ")


def _rotate_xy(point, yaw_offset):
    """Tourne (X,Y) autour de Z de yaw_offset (radians), Z inchange -- corrige la cible de
    pince pour le cap reel du robot au lieu de supposer le carton parfaitement de face."""
    x, y, z = point
    c, s = np.cos(yaw_offset), np.sin(yaw_offset)
    return np.array([x * c - y * s, x * s + y * c, z])


def _solve_ik_locked_wrist(chain, hand_offset, target, q_init, last_angle, iters=200, damping=0.05):
    """Comme solve_ik (lift_carton.py) mais fige le DERNIER angle du chain (ELBOW_YAW,
    poignet) a `last_angle` et ne resout la position qu'avec les 4 AUTRES articulations.
    solve_ik normal laisse les 2 DDL redondants du chain a 5 joints/3 DDL de position au
    hasard du solveur -- constate le 2026-09-03 que ca ne met PAS la rotation demandee dans
    le poignet a toutes les distances (cf. commentaire dans run_lift_sequence). Verifie
    numeriquement (hors robot) : erreur de position ~3e-8 sur les 4 cibles pince/serrage
    gauche/droite a PINCH_X=0.216, poignet exactement a l'angle demande, tous les joints
    dans la limite mecanique +-150deg."""
    q = q_init.copy()
    q[-1] = last_angle
    n_free = len(q) - 1
    for _ in range(iters):
        p0 = forward_kinematics(chain, hand_offset, q)
        error = target - p0
        if np.linalg.norm(error) < 1e-7:
            break
        J = np.zeros((3, n_free))
        for i in range(n_free):
            dq = q.copy()
            dq[i] += 1e-6
            J[:, i] = (forward_kinematics(chain, hand_offset, dq) - p0) / 1e-6
        JJt = J @ J.T + damping ** 2 * np.eye(3)
        step = J.T @ np.linalg.solve(JJt, error)
        q[:n_free] = q[:n_free] + step
        q[-1] = last_angle
    return q


def _quintic_ease(t):
    """Meme forme que math::QuinticInterpolate utilise par pd_stand_runner.cc pour sa
    propre transition (vitesse ET acceleration nulles aux deux bords) -- port depuis
    lift.py (simu)."""
    t = max(0.0, min(1.0, t))
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


def _bend_knees(lever, scale, stiffness_scale, duration, rate_hz=RATE_HZ, dry_run=False):
    """Transition jambes droites (pd_stand) -> flechies (posture walk au repos x`scale`),
    hanche+genou+cheville COORDONNES (pas le genou seul -- mauvais signe, deja fait tomber
    le robot immediatement lors d'un essai anterieur en simu). 2026-09-03, port depuis
    lift.py -- JAMAIS teste sur le robot reel avant ce jour."""
    if dry_run:
        print(f"    [dry-run] flexion genoux -- scale={scale} duration={duration}s")
        return
    for idx, kp, kd in [
        (LEFT_HIP_PITCH_INDEX, 200.0, 5.0), (RIGHT_HIP_PITCH_INDEX, 200.0, 5.0),
        (LEFT_KNEE_PITCH_INDEX, 450.0, 5.0), (RIGHT_KNEE_PITCH_INDEX, 450.0, 5.0),
        (LEFT_ANKLE_PITCH_INDEX, 400.0, 2.0), (RIGHT_ANKLE_PITCH_INDEX, 400.0, 2.0),
    ]:
        lever.set_gains(idx, kp * stiffness_scale, kd * stiffness_scale)
    n = max(1, int(duration * rate_hz))
    for i in range(n + 1):
        a = _quintic_ease(i / n)
        lever[LEFT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_L)
        lever[RIGHT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_R)
        lever[LEFT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_L)
        lever[RIGHT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_R)
        lever[LEFT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_L)
        lever[RIGHT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_R)
        time.sleep(1.0 / rate_hz)


def _straighten_knees(lever, scale, stiffness_scale, duration, rate_hz=RATE_HZ, dry_run=False):
    """Inverse de _bend_knees() -- ramene les jambes droites AVANT release(), pour eviter
    un saut de posture brutal au moment ou le controle est rendu (meme categorie de bug que
    celui trouve et corrige en simu le 2026-09-02)."""
    if dry_run:
        print(f"    [dry-run] redressement genoux -- scale={scale} duration={duration}s")
        return
    n = max(1, int(duration * rate_hz))
    for i in range(n + 1):
        a = 1.0 - _quintic_ease(i / n)
        lever[LEFT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_L)
        lever[RIGHT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_R)
        lever[LEFT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_L)
        lever[RIGHT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_R)
        lever[LEFT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_L)
        lever[RIGHT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_R)
        time.sleep(1.0 / rate_hz)


def _publish(lever, qL, qR):
    for idx, angle in zip(LEFT_JOINT_INDICES, qL):
        lever[idx] = float(angle)
    for idx, angle in zip(RIGHT_JOINT_INDICES, qR):
        lever[idx] = float(angle)


def move_arms(lever, qL0, qL1, qR0, qR1, duration, dry_run=False):
    """Interpole les 2x5 angles bras de (qL0,qR0) a (qL1,qR1) sur `duration` secondes."""
    n = max(1, int(duration * RATE_HZ))
    for i in range(n + 1):
        a = ease(i / n)
        qL = qL0 + a * (qL1 - qL0)
        qR = qR0 + a * (qR1 - qR0)
        if dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i/RATE_HZ:.2f}s  qL={np.round(qL, 4)}  qR={np.round(qR, 4)}")
            continue
        _publish(lever, qL, qR)
        time.sleep(1.0 / RATE_HZ)
    return qL, qR


def _build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pinch-z", type=float, default=-0.139,
                         help="2026-08-26 : recalibre pour le nouveau podium reel (53.5cm de "
                              "haut, contre la table 72cm d'avant) -- carton a 0.535+0.146="
                              "0.681m (monde), bassin pd_stand ~0.82m -> pinch_z=0.681-0.82="
                              "-0.139. PAS ENCORE TESTE sur le robot reel avec ce podium.")
    parser.add_argument("--pinch-yaw-offset", type=float, default=0.0,
                         help="Radians -- corrige la cible si le robot ne s'arrete pas "
                              "exactement de face au carton.")
    parser.add_argument("--wrist-rotation-deg", type=float, default=ELBOW_YAW_ROTATION_DEG,
                         help="Pivot de l'avant-bras (deg) pour attraper avec le cote de la "
                              "main -- 0 pour desactiver (comportement d'avant). Sens PAS "
                              "verifie sur le robot reel, inverser le signe si besoin.")
    parser.add_argument("--walk-stance-scale", type=float, default=WALK_STANCE_SCALE,
                         help="Echelle de flexion des genoux -- 0 pour desactiver (comportement "
                              "d'avant, jambes droites). Augmenter progressivement, jamais un "
                              "grand saut sur un 1er essai reel.")
    parser.add_argument("--skip-motion-state", action="store_true",
                         help="Suppose que le robot est deja en lower_body_balance.")
    parser.add_argument("--only-phase", choices=["approche", "serrage", "levee"], default=None,
                         help="N'execute qu'une phase puis s'arrete (pour valider pas a pas).")
    parser.add_argument("--dry-run", action="store_true",
                         help="N'envoie rien au robot -- affiche juste les angles cibles.")
    parser.add_argument("--no-confirm", action="store_true",
                         help="Pas de pause interactive entre phases -- jamais pour un premier essai.")
    return parser


def run_lift_sequence(node, lever, args):
    """Execute la sequence pince/serrage/levee sur un `node` (et un `lever` construit sur ce
    `node`) DEJA CREES par l'appelant -- ne fait NI rclpy.init()/shutdown() NI
    node.destroy_node(), c'est la responsabilite de l'appelant (reutilisee telle quelle par
    main() ci-dessous pour l'usage standalone, ET par orchestrateur.py pour l'usage
    combine sur un node/contexte rclpy partage avec la marche). En --dry-run, `node`/`lever`
    valent None (aucun effet reel, angles juste affiches) -- meme convention que
    move_arms(..., dry_run=...). `args` est un argparse.Namespace avec les memes champs que
    _build_arg_parser() (pinch_z, pinch_yaw_offset, wrist_rotation_deg, walk_stance_scale,
    skip_motion_state, only_phase, dry_run, no_confirm)."""
    confirm = not args.no_confirm
    wrist_rotation = np.radians(getattr(args, "wrist_rotation_deg", ELBOW_YAW_ROTATION_DEG))
    walk_stance_scale = getattr(args, "walk_stance_scale", WALK_STANCE_SCALE)

    # 2026-09-03, fix #2 : tourner hand_offset avant l'IK (fix #1, meme date) ne marche PAS de
    # facon fiable -- solve_ik ne vise QUE la position (3 DDL) avec 5 articulations (2 DDL
    # redondants), donc rien n'oblige le solveur a mettre la rotation dans le poignet
    # (ELBOW_YAW, dernier joint) plutot que dans les autres articulations. Constate sur le
    # robot reel le 2026-09-03 : a PINCH_X=0.216 (distance reelle validee), le solveur a laisse
    # qL[-1] a -3.7deg au lieu des ~90deg attendus (log [DEBUG] poignet) -- poignet quasi pas
    # tourne, ET position reelle du point de contact incoherente avec celle calculee pour un
    # poignet a 90deg -> "trop serre" au serrage. Fix definitif : FIGER le dernier angle
    # (ELBOW_YAW) a +-wrist_rotation et ne laisser l'IK resoudre la position qu'avec les 4
    # AUTRES articulations (_solve_ik_locked_wrist ci-dessous) -- le poignet tourne alors
    # GARANTI de l'angle demande, plus de hasard du solveur. hand_offset redevient
    # HAND_OFFSET_LEFT/RIGHT tel quel (pas tourne a la main) : forward_kinematics applique deja
    # la rotation du dernier joint au offset, inutile de la precalculer.
    q_pinch_L = _solve_ik_locked_wrist(LEFT_CHAIN, HAND_OFFSET_LEFT,
                                        _rotate_xy([PINCH_X, PINCH_Y, args.pinch_z], args.pinch_yaw_offset),
                                        Q_LEFT_HOME, wrist_rotation)
    q_pinch_R = _solve_ik_locked_wrist(RIGHT_CHAIN, HAND_OFFSET_RIGHT,
                                        _rotate_xy([PINCH_X, -PINCH_Y, args.pinch_z], args.pinch_yaw_offset),
                                        Q_RIGHT_HOME, -wrist_rotation)
    print(f"[DEBUG] poignet -- qL[-1]={np.degrees(q_pinch_L[-1]):.1f}deg "
          f"qR[-1]={np.degrees(q_pinch_R[-1]):.1f}deg (limite mecanique : +-150deg)")

    squeeze_L = _rotate_xy([PINCH_X, SQUEEZE_Y, args.pinch_z], args.pinch_yaw_offset)
    squeeze_R = _rotate_xy([PINCH_X, -SQUEEZE_Y, args.pinch_z], args.pinch_yaw_offset)
    q_squeeze_L = _solve_ik_locked_wrist(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_pinch_L, wrist_rotation)
    q_squeeze_R = _solve_ik_locked_wrist(RIGHT_CHAIN, HAND_OFFSET_RIGHT, squeeze_R, q_pinch_R, -wrist_rotation)

    run_approche = args.only_phase in (None, "approche")
    run_serrage = args.only_phase in (None, "serrage")
    run_levee = args.only_phase in (None, "levee")

    qL, qR = Q_LEFT_HOME, Q_RIGHT_HOME

    if run_approche and walk_stance_scale > 0:
        _checkpoint(f"flexion genoux -- scale={walk_stance_scale}, {WALK_STANCE_DURATION}s", confirm)
        _bend_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE, WALK_STANCE_DURATION,
                    dry_run=args.dry_run)
        if not args.dry_run:
            time.sleep(2.0)

    if run_approche:
        _checkpoint(
            f"approche -- mains vers pinch (x={PINCH_X} y=+-{PINCH_Y} z={args.pinch_z}, "
            f"poignet pivote de {np.degrees(wrist_rotation):.0f}deg), {APPROACH_DURATION}s",
            confirm,
        )
        qL, qR = move_arms(lever, Q_LEFT_HOME, q_pinch_L, Q_RIGHT_HOME, q_pinch_R,
                            APPROACH_DURATION, dry_run=args.dry_run)

    if run_serrage:
        _checkpoint(
            f"serrage -- Y +-{PINCH_Y} -> +-{SQUEEZE_Y}, {SQUEEZE_DURATION}s "
            "(LE CONTACT AVEC LE CARTON COMMENCE ICI)",
            confirm,
        )
        qL, qR = move_arms(lever, q_pinch_L, q_squeeze_L, q_pinch_R, q_squeeze_R,
                            SQUEEZE_DURATION, dry_run=args.dry_run)

    if run_levee:
        _checkpoint(
            f"levee -- Z {args.pinch_z} -> {LIFT_Z}, {LIFT_DURATION}s puis maintien "
            f"{HOLD_SECONDS}s",
            confirm,
        )
        if not args.dry_run:
            for idx in LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES:
                lever.set_gains(idx, stiffness=LEVEE_STIFFNESS)

        qL_prev, qR_prev = q_squeeze_L.copy(), q_squeeze_R.copy()
        n = max(1, int(LIFT_DURATION * RATE_HZ))
        for i in range(n + 1):
            a = ease(i / n)
            z = args.pinch_z + a * (LIFT_Z - args.pinch_z)
            qL_prev = _solve_ik_locked_wrist(LEFT_CHAIN, HAND_OFFSET_LEFT,
                                              _rotate_xy([PINCH_X, SQUEEZE_Y, z], args.pinch_yaw_offset),
                                              qL_prev, wrist_rotation, iters=30)
            qR_prev = _solve_ik_locked_wrist(RIGHT_CHAIN, HAND_OFFSET_RIGHT,
                                              _rotate_xy([PINCH_X, -SQUEEZE_Y, z], args.pinch_yaw_offset),
                                              qR_prev, -wrist_rotation, iters=30)
            if args.dry_run:
                if i in (0, n):
                    print(f"    [dry-run] t={i/RATE_HZ:.2f}s  qL={np.round(qL_prev, 4)}  qR={np.round(qR_prev, 4)}")
                continue
            _publish(lever, qL_prev, qR_prev)
            time.sleep(1.0 / RATE_HZ)

        if not args.dry_run:
            for _ in range(max(1, int(HOLD_SECONDS * RATE_HZ))):
                _publish(lever, qL_prev, qR_prev)
                time.sleep(1.0 / RATE_HZ)

    # 2026-08-31, bug trouve sur le robot reel : lever.release() tournait
    # INCONDITIONNELLEMENT en fin de fonction, meme apres --only-phase serrage seul --
    # annulant la prise tout juste etablie juste avant l'appel SEPARE --only-phase
    # levee suivant (procedure de test par etapes recommandee par la doc). Symptome
    # observe : "il desserre avant la levee". Fix : ne relacher qu'a la fin d'une
    # sequence qui atteint reellement levee (only_phase=None ou "levee") -- une
    # execution --only-phase approche/serrage seule doit laisser la prise active
    # pour l'appel suivant.
    if not args.dry_run and args.only_phase in (None, "levee"):
        if run_levee and walk_stance_scale > 0:
            _checkpoint("redressement genoux -- avant relachement", confirm)
            _straighten_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE,
                               WALK_STANCE_DURATION, dry_run=args.dry_run)
        _checkpoint("release() -- rend la main au runner pd_stand actif", confirm)
        if run_levee:
            n = max(1, int(RELEASE_RAMP_SECONDS * RATE_HZ))
            for i in range(n + 1):
                lever.set_weight(1.0 - i / n)
                _publish(lever, qL_prev, qR_prev)
                time.sleep(1.0 / RATE_HZ)
        lever.release()

    print("[INFO] Sequence terminee.")


def main():
    args = _build_arg_parser().parse_args()

    node = lever = None
    if not args.dry_run:
        rclpy.init()
        node = rclpy.create_node("levee")

        if not args.skip_motion_state:
            ok = ensure_motion_state(node, "lower_body_balance", timeout=MOTION_STATE_TIMEOUT)
            if not ok:
                print("[ERREUR] impossible de passer en lower_body_balance -- "
                      "arret (l'override des bras n'aurait aucun effet).", flush=True)
                node.destroy_node()
                rclpy.shutdown()
                return

        lever = Lever(node)

    try:
        run_lift_sequence(node, lever, args)
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
