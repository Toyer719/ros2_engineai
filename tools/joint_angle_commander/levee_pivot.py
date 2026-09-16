"""Prise du carton REEL puis pivot du BUSTE, carton en main, puis DEPOSE -- UN SEUL
processus/Lever partage de bout en bout (pas de saut entre scripts separes, la
posture bras deja en memoire est reutilisee directement d'une phase a l'autre).

2026-09-10 : phase "depose" ajoutee (port depuis depose.py cote simu, virtual_gamepad_ros)
-- descend le carton (LIFT_Z -> args.pinch_z, meme hauteur qu'a la prise), ouvre les
mains (SQUEEZE_Y -> PINCH_Y), retire les bras (retour Q_HOME), PUIS relache -- remplace
le relachement direct qui suivait le depivot. Volontairement dans LE MEME processus/
Lever que la levee et le pivot (pas un script separe) : /motion/joint_override_command
est un REMPLACEMENT COMPLET a chaque message (pas une fusion) -- un depose_real.py
separe, avec son PROPRE Lever, aurait du recalculer par IK une pose "deja tenue" depuis
zero (graine Q_HOME) au moment de prendre le relais, sans garantie de retomber sur la
MEME configuration articulaire que celle deja tenue par ce process -- meme piege que
celui trouve et corrige le meme jour cote simu (depose.py, bug reel : bascule sur une
branche articulaire differente au relais, chute intermittente confirmee par
telemetrie). Rester dans le meme processus evite le probleme a la racine (aucun
recalcul, aucune graine, la pose q L/qR est deja exacte).

Reutilise integralement la sequence approche/serrage/levee de levee.py (memes
constantes PINCH_X/Y, SQUEEZE_Y, LIFT_Z, LEVEE_STIFFNESS -- deja validee sur le
robot reel le 2026-09-01), puis enchaine sur le mecanisme de pivot_real.py
(deja valide buste seul jusqu'a 120deg, jambes flechies).

PREMIERE COMBINAISON JAMAIS TESTEE : pivoter en tenant reellement le carton
n'a encore jamais ete essaye, ni en simu (retire le 08/09 apres chute au
redressement des jambes, dans un scenario DIFFERENT -- marche en tenant,
pas pivot statique) ni sur le robot reel. Le poids du carton deplace le
centre de masse plus loin du buste que les bras seuls -- attends-toi a un
angle maximal stable PLUS PETIT qu'a vide (120deg), pas le meme.

AUCUNE SIMULATION -- envoie de vraies commandes au robot.

Prerequis : identiques a levee.py (lower_body_balance, memes exports/sources).

Securite :
  - --dry-run d'abord, toujours.
  - --angle-deg PETIT au premier essai (20deg suggere -- BEAUCOUP moins que
    les 120deg a vide) et remonter par petits paliers en observant a
    chaque fois, jamais un grand saut.
  - --only-phase approche seul, puis serrage, puis pivot -- valider chaque
    etape separement avant d'enchainer, comme pour levee.py.
  - Le carton peut tomber si l'equilibre casse pendant le pivot -- ne pas
    se tenir dans sa trajectoire de chute.
"""
import argparse
import sys
import os
import time

import numpy as np
import rclpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lever import Lever
from motion_state import ensure_motion_state
from levee import (
    APPROACH_LIFT_DURATION,
    LEFT_CHAIN, RIGHT_CHAIN, HAND_OFFSET_LEFT, HAND_OFFSET_RIGHT,
    LEFT_JOINT_INDICES, RIGHT_JOINT_INDICES, Q_LEFT_HOME, Q_RIGHT_HOME,
    WAYPOINT_Q_LEFT, WAYPOINT_Q_RIGHT, WAYPOINT_DURATION, WRIST_CHAIN_INDEX,
    PINCH_X, PINCH_Y, SQUEEZE_Y, LIFT_Z, APPROACH_DURATION, SQUEEZE_DURATION,
    LIFT_DURATION, HOLD_SECONDS, RATE_HZ, WALK_STANCE_SCALE,
    WALK_STANCE_STIFFNESS_SCALE, WALK_STANCE_DURATION, LEVEE_STIFFNESS,
    MOTION_STATE_TIMEOUT, ELBOW_YAW_ROTATION_DEG,
    _checkpoint, _rotate_xy, solve_arm_ik, _bend_knees, _straighten_knees,
    _publish, move_arms, ease, forward_kinematics,
)
from pivot_real import WAIST_JOINT_INDEX, WAIST_KP, WAIST_KD, _ease as _pivot_ease

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot_arm_ik"))
from lift_carton import mirror_left_to_right


def _publish_with_waist(lever, qL, qR, waist):
    lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES + [WAIST_JOINT_INDEX],
                     list(qL) + list(qR) + [waist])


# 2026-09-14 : run_depose_phase() (code MORT depuis toujours -- confirme par grep, zero
# appelant dans ce depot) supprimee ici. Elle divergeait deja de la depose REELLEMENT
# executee (bloc en ligne dans run_lift_and_pivot ci-dessous : gere le buste/
# pre_release_drop, celle-ci non) -- la corriger aurait ete ecrire une 3e implementation.
# La depose reste pour l'instant une sequence en ligne, pas une phase independante/
# --only-phase selectionnable -- voir le commentaire sur `run_depose` plus bas pour
# pourquoi (meme piege de rebascule sur une autre branche IK que celui deja documente
# ici et corrige cote simu).


def run_lift_and_pivot(node, lever, args):
    confirm = not args.no_confirm
    wrist_rotation = np.radians(args.wrist_rotation_deg)
    walk_stance_scale = args.walk_stance_scale

    # 2026-09-15 : reference "coude tendu" pour le null-space -- sans null_space_pref
    # explicite, solve_arm_ik retombe sur q_init (WAYPOINT_Q_LEFT, coude a -110deg) comme
    # preference par defaut (voir solve_arm_ik dans lift_carton.py) : q_pinch_L n'etait
    # donc JAMAIS garanti d'etre un bras tendu, juste "la solution la plus proche de la
    # posture repliee" -- constate sur le robot reel (bras qui leve bien mais ne se tend
    # pas). ELBOW_PITCH (index 3) mis a 0 = coude aussi droit que la geometrie le permet.
    straight_pref_L = WAYPOINT_Q_LEFT.copy()
    straight_pref_L[3] = 0.0
    q_pinch_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                              _rotate_xy([args.pinch_x, PINCH_Y, args.pinch_z], args.pinch_yaw_offset),
                              WAYPOINT_Q_LEFT, lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                              null_space_pref=straight_pref_L)
    q_pinch_R = mirror_left_to_right(q_pinch_L)
    # 2026-09-14 : null_space_pref=q_pinch_L -- meme fix que levee.py (cf son commentaire),
    # consolide ici depuis l'ancienne _solve_ik_locked_wrist (pas de terme null-space du tout).
    squeeze_L = _rotate_xy([args.pinch_x, SQUEEZE_Y, args.pinch_z], args.pinch_yaw_offset)
    q_squeeze_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_pinch_L,
                                lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                                null_space_pref=q_pinch_L)
    q_squeeze_R = mirror_left_to_right(q_squeeze_L)

    run_approche = args.only_phase in (None, "approche")
    run_serrage = args.only_phase in (None, "serrage")
    run_levee = args.only_phase in (None, "levee")
    run_pivot = args.only_phase in (None, "pivot")
    # depose n'est JAMAIS un --only-phase independant (contrairement aux autres) --
    # ne peut tourner que dans la FOULEE d'un pivot deja execute dans ce meme
    # processus/Lever (qL/qR deja exacts, pas recalcules par IK depuis une graine).
    # Une invocation separee refait par IK depuis Q_HOME comme le fait deja
    # --only-phase levee/pivot pour les autres phases -- MEME PIEGE que celui
    # trouve et corrige cote simu le meme jour (bascule possible sur une autre
    # branche articulaire au relais), voir docstring module. Valider pivot seul
    # via --only-phase pivot (le carton reste tenu, rien ne se relache) puis
    # lancer la sequence complete (sans --only-phase) pour atteindre depose.
    run_depose = args.only_phase is None

    qL, qR = Q_LEFT_HOME, Q_RIGHT_HOME

    if run_approche and walk_stance_scale > 0:
        _checkpoint(f"flexion genoux -- scale={walk_stance_scale}, {WALK_STANCE_DURATION:.1f}s", confirm)
        _bend_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE, WALK_STANCE_DURATION,
                    dry_run=args.dry_run)
        if not args.dry_run:
            time.sleep(2.0)

    if run_approche:
        _checkpoint(f"point de passage -- coudes vers l'arriere, {WAYPOINT_DURATION:.1f}s", confirm)
        qL, qR = move_arms(lever, Q_LEFT_HOME, WAYPOINT_Q_LEFT, Q_RIGHT_HOME, WAYPOINT_Q_RIGHT,
                            WAYPOINT_DURATION, dry_run=args.dry_run)

        # 2026-09-11 : ramp en ESPACE CARTESIEN (pas move_arms en espace articulaire) -- la
        # main monte trop haut en route sinon (arc, pas une ligne droite), meme fix que
        # levee.py --only-phase approche, voir son commentaire pour le detail.
        # 2026-09-15 : mesure numerique -- waypoint_hand_L et pinch_target_L n'ont PAS la
        # meme hauteur (waypoint Z=0.039m, pinch Z=0.106m, ecart 6.7cm), donc la ligne
        # "droite" precedente (waypoint -> pinch directement) montait en diagonale sur toute
        # l'approche -- ce n'etait pas un arc/bug de solveur, une geometrie de bout en bout
        # differente. Demande explicite utilisateur : la partie "approche" doit etre une
        # ligne HORIZONTALE (Z constant), quitte a corriger la hauteur separement avant.
        # Scinde donc en 2 segments :
        #   1) ajustement vertical (meme X/Y que le point de passage, Z -> hauteur pince)
        #   2) approche horizontale (Z constant = hauteur pince, X/Y -> cible pince)
        pinch_target_L = _rotate_xy([args.pinch_x, PINCH_Y, args.pinch_z], args.pinch_yaw_offset)
        waypoint_hand_L = forward_kinematics(LEFT_CHAIN, HAND_OFFSET_LEFT, WAYPOINT_Q_LEFT)
        raised_point_L = np.array([waypoint_hand_L[0], waypoint_hand_L[1], pinch_target_L[2]])
        qL, qR = WAYPOINT_Q_LEFT.copy(), WAYPOINT_Q_RIGHT.copy()

        _checkpoint(f"approche -- ajustement vertical, {APPROACH_LIFT_DURATION:.1f}s", confirm)
        n_lift = max(1, int(APPROACH_LIFT_DURATION * RATE_HZ))
        for i in range(n_lift + 1):
            a = ease(i / n_lift)
            target = waypoint_hand_L + a * (raised_point_L - waypoint_hand_L)
            # Reste ancre sur la posture repliee pendant l'ajustement vertical -- seule
            # l'approche horizontale ci-dessous doit se tendre (demande utilisateur).
            qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, target, qL,
                               lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation, iters=30,
                               null_space_pref=WAYPOINT_Q_LEFT)
            qR = mirror_left_to_right(qL)
            if args.dry_run:
                if i in (0, n_lift):
                    print(f"    [dry-run] t={i/RATE_HZ:.2f}s  qL={np.round(qL, 4)}  qR={np.round(qR, 4)}")
                continue
            _publish(lever, qL, qR)
            time.sleep(1.0 / RATE_HZ)

        _checkpoint(f"approche -- ligne horizontale vers pinch, {APPROACH_DURATION:.1f}s", confirm)
        # Ancre GLISSANTE entre WAYPOINT_Q_LEFT et straight_pref_L (PAS q_pinch_L -- q_pinch_L
        # n'etait lui-meme jamais garanti tendu, voir son commentaire plus haut) au meme
        # rythme `a` que la position de la main -- repliee en debut de segment, coude aussi
        # droit que possible en fin de segment (vise le centre de la face du carton, bras
        # vraiment tendu). PAS ENCORE VALIDE sur le robot reel.
        n = max(1, int(APPROACH_DURATION * RATE_HZ))
        for i in range(n + 1):
            a = ease(i / n)
            target = raised_point_L + a * (pinch_target_L - raised_point_L)
            anchor_L = (1.0 - a) * WAYPOINT_Q_LEFT + a * straight_pref_L
            qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, target, qL,
                               lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation, iters=30,
                               null_space_pref=anchor_L)
            qR = mirror_left_to_right(qL)
            if args.dry_run:
                if i in (0, n):
                    print(f"    [dry-run] t={i/RATE_HZ:.2f}s  qL={np.round(qL, 4)}  qR={np.round(qR, 4)}")
                continue
            _publish(lever, qL, qR)
            time.sleep(1.0 / RATE_HZ)

    if run_serrage:
        _checkpoint(f"serrage -- Y +-{PINCH_Y} -> +-{SQUEEZE_Y}, {SQUEEZE_DURATION:.1f}s "
                    "(LE CONTACT AVEC LE CARTON COMMENCE ICI)", confirm)
        qL, qR = move_arms(lever, q_pinch_L, q_squeeze_L, q_pinch_R, q_squeeze_R,
                            SQUEEZE_DURATION, dry_run=args.dry_run)

    if run_levee:
        _checkpoint(f"levee -- Z {args.pinch_z} -> {LIFT_Z}, {LIFT_DURATION:.1f}s puis maintien "
                    f"{HOLD_SECONDS:.1f}s", confirm)
        if not args.dry_run:
            for idx in LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES:
                lever.set_gains(idx, stiffness=LEVEE_STIFFNESS)
        # 2026-09-14 : null_space_pref=q_squeeze_L (pose fixe) -- meme fix que levee.py.
        qL, qR = q_squeeze_L.copy(), q_squeeze_R.copy()
        n = max(1, int(LIFT_DURATION * RATE_HZ))
        for i in range(n + 1):
            a = ease(i / n)
            z = args.pinch_z + a * (LIFT_Z - args.pinch_z)
            qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                               _rotate_xy([args.pinch_x, SQUEEZE_Y, z], args.pinch_yaw_offset),
                               qL, lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                               iters=30, null_space_pref=q_squeeze_L)
            qR = mirror_left_to_right(qL)
            if args.dry_run:
                if i in (0, n):
                    print(f"    [dry-run] t={i / RATE_HZ:.2f}s  qL={np.round(qL, 4)}  qR={np.round(qR, 4)}")
                continue
            _publish(lever, qL, qR)
            time.sleep(1.0 / RATE_HZ)
        if not args.dry_run:
            for _ in range(max(1, int(HOLD_SECONDS * RATE_HZ))):
                _publish(lever, qL, qR)
                time.sleep(1.0 / RATE_HZ)

    if not run_pivot:
        print("[INFO] only_phase termine -- carton tenu, pas de pivot dans cet appel.")
        return

    angle_target = np.radians(args.angle_deg)
    n = max(1, int(args.pivot_duration * RATE_HZ))

    if not args.dry_run:
        lever.set_gains(WAIST_JOINT_INDEX, WAIST_KP, WAIST_KD)

    _checkpoint(f"pivot buste -- 0 -> {args.angle_deg:.0f}deg ({args.pivot_duration:.1f}s), "
                "CARTON EN MAIN -- republication bras+jambes+buste en continu", confirm)
    for i in range(n + 1):
        a = _pivot_ease(i / n)
        waist = a * angle_target
        if args.dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i / RATE_HZ:.2f}s  waist={np.degrees(waist):.1f}deg")
            continue
        _publish_with_waist(lever, qL, qR, waist)
        time.sleep(1.0 / RATE_HZ)

    _checkpoint(f"maintien pivote -- {args.hold_seconds:.1f}s", confirm)
    if not args.dry_run:
        for _ in range(max(1, int(args.hold_seconds * RATE_HZ))):
            _publish_with_waist(lever, qL, qR, angle_target)
            time.sleep(1.0 / RATE_HZ)

    if not run_depose:
        print("[INFO] only_phase termine -- carton tenu, buste toujours tourne, "
              "pas de depose dans cet appel.")
        return

    # 2026-09-10, demande explicite utilisateur : le carton se lache PENDANT que le
    # buste est encore tourne (angle_target), le buste ne revient au centre
    # qu'APRES -- inverse de l'ordre precedent (depivot puis depose). depose_phase
    # avec final_release=False laisse bras+jambes tenus a poids plein une fois
    # retires en Q_HOME ; le buste garde AUTOMATIQUEMENT angle_target pendant toute
    # la dépose (meme Lever, _position partagee, pas de republication necessaire).
    # 2026-09-10, simplifie sur demande utilisateur : plus de descente complete ni de
    # retour bras home ici -- juste ecarter les mains (SQUEEZE_Y -> PINCH_Y) SUR PLACE,
    # puis depivot juste apres. Ajout ensuite (meme jour) : petite baisse partielle
    # (--pre-release-drop, PAS jusqu'a pinch_z) juste avant l'ecartement, sur demande
    # utilisateur.
    drop_z = LIFT_Z - args.pre_release_drop
    if args.pre_release_drop > 0:
        _checkpoint(f"baisse avant relachement -- Z {LIFT_Z} -> {drop_z:.3f} "
                    f"({args.pre_release_drop_duration:.1f}s), buste encore tourne", confirm)
        # 2026-09-14 : null_space_pref=pre_drop_anchor_L (pose fixe capturee avant la boucle) --
        # meme fix que les rampes precedentes.
        pre_drop_anchor_L = qL.copy()
        n3 = max(1, int(args.pre_release_drop_duration * RATE_HZ))
        for i in range(n3 + 1):
            a = ease(i / n3)
            z = LIFT_Z + a * (drop_z - LIFT_Z)
            qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                               _rotate_xy([args.pinch_x, SQUEEZE_Y, z], args.pinch_yaw_offset),
                               qL, lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                               iters=30, null_space_pref=pre_drop_anchor_L)
            qR = mirror_left_to_right(qL)
            if args.dry_run:
                if i in (0, n3):
                    print(f"    [dry-run] t={i / RATE_HZ:.2f}s  qL={np.round(qL, 4)}  qR={np.round(qR, 4)}")
                continue
            _publish(lever, qL, qR)
            time.sleep(1.0 / RATE_HZ)

    _checkpoint(f"desserrage -- Y +-{SQUEEZE_Y} -> +-{PINCH_Y} ({args.open_duration:.1f}s) "
                "-- LE CARTON EST RELACHE ICI, buste encore tourne", confirm)
    q_open_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                             _rotate_xy([args.pinch_x, PINCH_Y, drop_z], args.pinch_yaw_offset),
                             qL, lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation)
    q_open_R = mirror_left_to_right(q_open_L)
    qL, qR = move_arms(lever, qL, q_open_L, qR, q_open_R, args.open_duration, dry_run=args.dry_run)

    # 2026-09-14 : ECARTEMENT + translation arriere ajoutees ici, portage depuis depose.py
    # cote simu (ECARTEMENT_GAP_Y/RETREAT_BACK_X, ajoutees le meme jour) -- sur le robot,
    # l'ancien enchainement sautait DIRECTEMENT de la pose juste ouverte (desserrage) vers
    # Q_HOME en espace articulaire (move_arms), sans aucune etape intermediaire -- meme
    # bug/risque que celui trouve cote simu (la main peut retraverser pres du point de
    # serrage/carton en route), potentiellement pire ici puisque cote simu il y avait au
    # moins un arret WAYPOINT_Q entre les deux. Valeurs de depart VOLONTAIREMENT distinctes
    # de celles de la simu (echelles non comparables : cible dynamique cote simu vs
    # constantes statiques ici) -- PAS ENCORE VALIDEES sur le robot reel, a ajuster via
    # --ecartement-gap-y/--retreat-back-x en observant le robot, jamais un grand saut.
    _checkpoint(f"ecartement -- Y +-{PINCH_Y} -> +-{PINCH_Y + args.ecartement_gap_y:.3f} "
                f"({args.ecartement_duration:.1f}s)", confirm)
    ecart_L = _rotate_xy([args.pinch_x, PINCH_Y + args.ecartement_gap_y, drop_z], args.pinch_yaw_offset)
    q_ecart_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, ecart_L, qL,
                              lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                              null_space_pref=qL)
    q_ecart_R = mirror_left_to_right(q_ecart_L)
    qL, qR = move_arms(lever, qL, q_ecart_L, qR, q_ecart_R, args.ecartement_duration,
                        dry_run=args.dry_run)

    # 2026-09-14 : un premier essai qui ne retirait QUE X avait ete juge trop extreme
    # (coude au-dela de WAYPOINT_Q) -- Y avait alors ete retire EN MEME TEMPS que X, ce
    # qui creait un large ecartement suivi d'un resserrement brusque ("comme un serrage a
    # l'envers", retour utilisateur du 2026-09-15). 2026-09-15 : revient a la sequence
    # EXACTE de depose.py cote simu -- X SEUL bouge, Y reste a la valeur ELARGIE de
    # l'ecartement jusqu'au point de passage. Reverifie numeriquement a retreat_back_x=0.19
    # (valeur actuelle, pas 0.10 comme le vieux test qui donnait -138deg) : coude a
    # -101deg, proche de WAYPOINT_Q (-110deg), pas de posture extreme.
    _checkpoint(f"translation arriere -- X {args.pinch_x} -> {args.retreat_back_x} "
                f"({args.retreat_back_duration:.1f}s), Y reste large, main ramenee pres du corps",
                confirm)
    n4 = max(1, int(args.retreat_back_duration * RATE_HZ))
    q_retreat_start_L = qL.copy()
    for i in range(n4 + 1):
        a = ease(i / n4)
        x = ecart_L[0] + a * (args.retreat_back_x - ecart_L[0])
        qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                           np.array([x, ecart_L[1], ecart_L[2]]), qL,
                           lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation, iters=30,
                           null_space_pref=q_retreat_start_L)
        qR = mirror_left_to_right(qL)
        if args.dry_run:
            if i in (0, n4):
                print(f"    [dry-run] t={i / RATE_HZ:.2f}s  qL={np.round(qL, 4)}  qR={np.round(qR, 4)}")
            continue
        _publish(lever, qL, qR)
        time.sleep(1.0 / RATE_HZ)

    # 2026-09-15 : etape manquante par rapport a depose.py cote simu -- reajoutee (point de
    # passage coudes-vers-l'arriere entre la translation arriere et Q_HOME, "meme sequence").
    _checkpoint(f"degagement -- coudes vers l'arriere ({args.degagement_waypoint_duration:.1f}s)",
                confirm)
    qL, qR = move_arms(lever, qL, WAYPOINT_Q_LEFT, qR, WAYPOINT_Q_RIGHT,
                        args.degagement_waypoint_duration, dry_run=args.dry_run)

    # 2026-09-10, sur demande utilisateur (les bras tendus retraversaient l'espace ou
    # le carton vient d'etre pose pendant le depivot, choc constate) : retour des bras
    # a Q_HOME (le long du corps) AVANT le depivot, pas apres -- le buste tourne
    # ensuite avec les bras deja replies, plus rien a heurter.
    _checkpoint(f"retour bras le long du corps ({args.retreat_duration:.1f}s), "
                "avant le depivot", confirm)
    qL, qR = move_arms(lever, qL, Q_LEFT_HOME, qR, Q_RIGHT_HOME, args.retreat_duration,
                        dry_run=args.dry_run)

    _checkpoint(f"depivot -- {args.angle_deg:.0f}deg -> 0deg ({args.pivot_duration:.1f}s), "
                "CARTON DEJA LACHE, bras replies -- buste seul", confirm)
    for i in range(n + 1):
        a = _pivot_ease(i / n)
        waist = (1.0 - a) * angle_target
        if args.dry_run:
            if i in (0, n):
                print(f"    [dry-run] t={i / RATE_HZ:.2f}s  waist={np.degrees(waist):.1f}deg")
            continue
        lever[WAIST_JOINT_INDEX] = float(waist)
        time.sleep(1.0 / RATE_HZ)

    if not args.dry_run and walk_stance_scale > 0:
        _checkpoint("redressement genoux -- avant relachement final", confirm)
        _straighten_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE,
                           WALK_STANCE_DURATION, dry_run=args.dry_run)

    if not args.dry_run:
        _checkpoint(f"relachement final -- rampe {args.release_ramp_seconds:.1f}s", confirm)
        n2 = max(1, int(args.release_ramp_seconds * RATE_HZ))
        for i in range(n2 + 1):
            lever.set_weight(1.0 - i / n2)
            _publish_with_waist(lever, qL, qR, 0.0)
            time.sleep(1.0 / RATE_HZ)
        lever.release()

    print("[INFO] Sequence terminee.")


def _build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pinch-x", type=float, default=PINCH_X,
                         help="2026-09-15 : expose PINCH_X (distance de portee, defaut "
                              f"{PINCH_X}m calibre pour 14.6cm du bord du podium) en CLI pour "
                              "tester une portee differente sans modifier la constante calibree.")
    parser.add_argument("--pinch-z", type=float, default=0.05,
                         help="2026-09-10 : defaut corrige de -0.139 (perime, geometrie "
                              "d'avant le podium remesure a 0.8m) vers 0.106, cense matcher "
                              "levee.py -- MAIS n'a jamais suivi le changement ulterieur de "
                              "levee.py vers 0.05 (desync trouvee + confirmee sur le robot "
                              "reel le 2026-09-15 : 0.106 levait trop haut, 0.05 correct).")
    parser.add_argument("--pinch-yaw-offset", type=float, default=0.0)
    parser.add_argument("--wrist-rotation-deg", type=float, default=ELBOW_YAW_ROTATION_DEG)
    parser.add_argument("--walk-stance-scale", type=float, default=WALK_STANCE_SCALE)
    parser.add_argument("--angle-deg", type=float, default=20.0,
                         help="Angle de pivot (deg), CARTON EN MAIN. PETIT par defaut (20) -- "
                              "premiere combinaison jamais testee, attends-toi a un maximum "
                              "stable plus bas qu'a vide (120deg sans carton). Remonter par "
                              "petits paliers, jamais un grand saut.")
    parser.add_argument("--pivot-duration", type=float, default=4.0,
                         help="2026-09-10 : etait 6.0, reduit ~35% apres la 1ere sequence "
                              "complete validee sans incident (jugee 'tres lente').")
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--release-ramp-seconds", type=float, default=1.5)
    parser.add_argument("--pre-release-drop", type=float, default=0.03,
                         help="2026-09-10, demande utilisateur : baisse partielle du carton "
                              "(metres, PAS jusqu'a --pinch-z) juste avant l'ecartement des "
                              "mains, buste encore tourne -- 0 pour desactiver.")
    parser.add_argument("--pre-release-drop-duration", type=float, default=2.0)
    parser.add_argument("--depose-duration", type=float, default=4.0,
                         help="Descente du carton (LIFT_Z -> --pinch-z), secondes. "
                              "2026-09-10 : etait 6.0, meme reduction moderee.")
    parser.add_argument("--open-duration", type=float, default=2.0,
                         help="Desserrage (SQUEEZE_Y -> PINCH_Y) -- LE CARTON EST RELACHE ICI. "
                              "2026-09-10 : etait 3.0.")
    parser.add_argument("--ecartement-gap-y", type=float, default=0.025,
                         help="2026-09-14, port depuis depose.py (simu) : ecart Y supplementaire "
                              "des mains apres le desserrage, avant le retour bras home -- evite "
                              "de sauter directement (espace articulaire) depuis une pose encore "
                              "proche du carton. 2026-09-15 : reduit de 0.05 a 0.025 (retour "
                              "utilisateur -- 0.05 ecartait trop les mains apres relachement).")
    parser.add_argument("--ecartement-duration", type=float, default=2.0,
                         help="2026-09-14 -- defaut volontairement LENT (pas 1.0s comme la simu, "
                              "mouvement jamais teste sur le materiel).")
    parser.add_argument("--retreat-back-x", type=float, default=0.19,
                         help="2026-09-14, port depuis depose.py (simu) : translation X vers "
                              "l'arriere (main ramenee plus pres du corps) juste avant le retour "
                              "en Q_HOME. 2026-09-15 : verifie numeriquement que Y RESTE LARGE "
                              "(voir plus bas) a cette valeur -- coude a -101deg, proche de "
                              "WAYPOINT_Q (-110deg), pas de posture extreme.")
    parser.add_argument("--retreat-back-duration", type=float, default=2.0,
                         help="2026-09-14 -- defaut volontairement LENT (pas 1.0s comme la simu).")
    parser.add_argument("--degagement-waypoint-duration", type=float, default=2.0,
                         help="2026-09-15 : etape manquante par rapport a depose.py (simu) -- "
                              "reajoutee ici. Point de passage entre la translation arriere et "
                              "Q_HOME (meme valeur que g.degagement_waypoint_duration en simu).")
    parser.add_argument("--retreat-duration", type=float, default=2.0,
                         help="Retour des bras a Q_HOME apres le desserrage. "
                              "2026-09-10 : etait 3.0.")
    parser.add_argument("--skip-motion-state", action="store_true")
    parser.add_argument("--only-phase", choices=["approche", "serrage", "levee", "pivot"], default=None,
                         help="N'execute qu'une phase puis s'arrete -- valider approche, puis "
                              "serrage, puis levee, puis pivot separement au premier essai. "
                              "'pivot' laisse le carton TENU (pas de depose) -- depose ne "
                              "tourne que dans la sequence complete (sans --only-phase), voir "
                              "docstring module.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-confirm", action="store_true")
    return parser


def main():
    args = _build_arg_parser().parse_args()

    node = lever = None
    if not args.dry_run:
        rclpy.init()
        node = rclpy.create_node("levee_pivot")

        if not args.skip_motion_state:
            ok = ensure_motion_state(node, "lower_body_balance", timeout=MOTION_STATE_TIMEOUT)
            if not ok:
                print("[ERREUR] impossible de passer en lower_body_balance -- arret.", flush=True)
                node.destroy_node()
                rclpy.shutdown()
                return

        lever = Lever(node)

    try:
        run_lift_and_pivot(node, lever, args)
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
