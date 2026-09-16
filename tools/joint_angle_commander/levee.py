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
      3) FIX RETENU (a l'epoque : _solve_ik_locked_wrist local a ce fichier, remplace le
         2026-09-14 par solve_arm_ik(..., lock_index=WRIST_CHAIN_INDEX) -- consolidation cote
         simu du 2026-09-11, jamais repercutee ici avant ce jour) -- fige le dernier angle
         (ELBOW_YAW) exactement a +-wrist_rotation et ne laisse l'IK resoudre la position
         qu'avec les 4 autres articulations. Poignet tourne alors GARANTI de l'angle demande
         (plus de hasard du solveur), hand_offset redevient HAND_OFFSET_LEFT/RIGHT tel quel
         (plus besoin de le tourner a la main, forward_kinematics applique deja la rotation du
         dernier joint). Verifie numeriquement (hors robot) : erreur de position ~3e-8 aux 4
         cibles pince/serrage gauche/droite a PINCH_X=0.216, tous les joints dans la limite
         +-150deg. Signe : +90deg gauche/-90deg droite (chaine en miroir, confirme visuellement
         essai 1). 2026-09-14 : le remplacement par solve_arm_ik ajoute AUSSI un terme
         null-space (absent de l'ancienne _solve_ik_locked_wrist) -- meme classe de bug que
         celle trouvee et corrigee cote simu ce jour-la (l'unique DDL redondant restant apres
         le verrouillage peut deriver vers une posture de bras aberrante entre 2 appels
         proches, visible comme un saut brusque de l'avant-bras). PAS ENCORE VALIDE sur le
         robot reel.
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
    LEFT_CHAIN, RIGHT_CHAIN, HAND_OFFSET_LEFT, HAND_OFFSET_RIGHT, solve_arm_ik, ease,
    forward_kinematics, mirror_left_to_right,
)

LEFT_JOINT_INDICES = [13, 14, 15, 16, 17]
RIGHT_JOINT_INDICES = [18, 19, 20, 21, 22]
# 2026-09-14 : index de l'articulation figee (ELBOW_YAW, poignet) dans LEFT_CHAIN/
# RIGHT_CHAIN -- meme convention de nommage que ELBOW_PITCH_CHAIN_INDEX cote simu
# (lift.py/depose.py/pivot.py), utilise avec solve_arm_ik(..., lock_index=WRIST_CHAIN_INDEX).
# Le robot reel fige le POIGNET (pas le coude comme la simu) -- decision volontaire, voir
# _solve_ik_locked_wrist ci-dessous (remplace par solve_arm_ik le 2026-09-14) : le
# verrouillage du poignet sert ELBOW_YAW_ROTATION_DEG (prise par le cote de la main), sans
# equivalent cote simu -- ne pas changer sans revalider toute la geometrie calibree
# (PINCH_X/SQUEEZE_Y/LIFT_Z) qui suppose ce choix de verrouillage.
WRIST_CHAIN_INDEX = 4

Q_LEFT_HOME = np.array([0.000879, 0.075284, -0.000233, -0.126397, -0.000033])
Q_RIGHT_HOME = np.array([0.000885, -0.075161, 0.000241, -0.126390, 0.000033])

WAYPOINT_Q_LEFT = np.radians([30.0, 5.0, 0.0, -110.0, 0.0])
WAYPOINT_Q_RIGHT = np.radians([30.0, -5.0, 0.0, -110.0, 0.0])

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
SQUEEZE_Y = 0.13          # 2026-09-11 : observe en direct sur le robot reel ("va serrer trop
                          # fort" pendant l'approche) -- le carton actuellement sur la table
                          # semble avoir une demi-largeur REELLE plus petite que celle supposee
                          # ici (0.0955m) -- elargi par prudence en attendant une mesure directe.
                          # A resserrer progressivement via --only-phase serrage si trop large
                          # (pas de contact), jamais un grand saut vers le bas.
                          # 2026-09-10, demande explicite utilisateur ("large pour y aller a
                          # tatillon") apres que 0.095 (0.5mm de compression nominale
                          # seulement) ait ete juge encore trop ferme. AU-DELA de la
                          # demi-largeur reelle du carton (0.0955m) -- verifie en git log/
                          # memoire, AUCUNE valeur >0.105 jamais utilisee sur ce robot avant
                          # ce jour (0.105 avec rotation de poignet, desactivee, deja juge
                          # trop serre). En theorie les mains ne compriment plus le carton du
                          # tout a cette largeur (l'effleurent au mieux) -- accepte pour
                          # debloquer le test et iterer, au prix d'un risque de glissement/
                          # prise plus faible pendant la levee/le pivot. Si le carton glisse
                          # ou tombe pendant le maintien, redescendre progressivement vers
                          # 0.0955 plutot que revenir directement a 0.093/0.095 (deja juges
                          # trop fermes) -- le vrai probleme sous-jacent reste probablement la
                          # geometrie du coude (-102/-111deg, contact avant-bras/torse
                          # suspecte), pas cette valeur seule.
                          # 2026-09-09 : 0.090 juge "ferme trop" sur le robot reel avec la
                          # nouvelle geometrie bras-vers-le-haut (meme valeur nominale, mais
                          # angle de coude different -> compression differente) -- desserre
                          # legerement (+3mm). A reajuster encore si besoin.
                          # 2026-09-03 : retour a la valeur initiale (0.090, prise paume, avant
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
LIFT_Z = 0.15                 # 2026-09-09, CORRECTION D'URGENCE : 0.391 (qui gardait la
                              # meme amplitude de montee 0.285m que l'ancienne geometrie
                              # bras-vers-le-bas) fait tomber J14_SHOULDER_ROLL_L/
                              # J19_SHOULDER_ROLL_R hors de leur limite mecanique
                              # (Guide_PM01_FR.pdf : [-0.6108, 2.3562] rad = [-35, 135]deg --
                              # 0.391 calcule -49deg, ~14deg au-dela) -- cause confirmee du
                              # "passe en rouge et s'eteint" en fin de levee (1er essai reel
                              # avec le podium 0.8m). Avec le bras qui vise desormais AU-DESSUS
                              # du bassin, il n'a plus la meme course vers le haut : verifie
                              # (solve_ik + limites du guide, les 5 articulations, approche/
                              # serrage/levee) que 0.15 reste dans toutes les limites avec
                              # ~4.6deg de marge sur l'epaule (la plus juste). Levee reelle
                              # desormais modeste (~4-5cm au-dessus du point de prise), pas
                              # 28.5cm comme avant -- la geometrie ne permet plus plus.
APPROACH_DURATION = 4.0      # 2026-09-10 : etait 5.0 puis 3.5 -- validait alors TOUTE
                              # l'approche (vertical+horizontal en un seul mouvement diagonal).
                              # 2026-09-15 : ne couvre plus que le segment HORIZONTAL depuis le
                              # scindage vertical/horizontal (voir run_lift_sequence). Passe a
                              # 2.0s (demande "plus vite") COMBINE a --pinch-x 0.30 (demande
                              # "plus loin") -> vitesse de la main 1.3cm/s (validee) -> 6.4cm/s
                              # (x5), constate "tremble a mort" sur le robot reel -- remonte a
                              # 4.0s pour ramener la vitesse a ~3.2cm/s (encore ~2.5x plus
                              # rapide qu'avant, mais loin du x5 qui fait trembler). PAS ENCORE
                              # VALIDE sur le robot reel a cette vitesse -- reajuster encore
                              # (plus haut ou plus bas) selon observation.
WAYPOINT_DURATION = 3.5      # point de passage coudes-vers-l'arriere avant l'approche, meme
                              # rythme que APPROACH_DURATION.
APPROACH_LIFT_DURATION = 1.0  # 2026-09-15 : ajustement vertical avant la ligne horizontale
                              # de l'approche -- voir commentaire dans run_lift_sequence.
SQUEEZE_DURATION = 3.5       # 2026-09-10 : etait 5.0, meme reduction moderee.
LIFT_DURATION = 4.0          # 2026-09-10 : etait 6.0, meme reduction moderee.
# 2026-09-09 (historique) : ces 3 valeurs etaient a 4.0/4.0/5.0, ralenties suite a un
# retour utilisateur "va trop vite" sur la geometrie d'alors -- garder ces nouvelles
# valeurs AU-DESSUS de 4.0/4.0/5.0 si une future demande d'acceleration arrive.
HOLD_SECONDS = 0.0          # 2026-09-09 : etait 3.0 (maintien immobile en haut avant
                             # redressement genoux + relachement) -- retire a la demande
                             # de l'utilisateur pour accelerer (perceptible comme "il
                             # pause avant de descendre"). Enchaine directement sur le
                             # redressement/relachement.
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

# 2026-09-09 : DESACTIVEE (etait 1.0) -- en simu, la combinaison genoux flechis +
# nouveau mouvement de bras (podium remesure a 0.8m, bras vise vers le haut) fait
# tomber le robot (chute confirmee par telemetrie PENDANT l'approche, pas la levee --
# z stable tout du long de la flexion+pause, s'effondre des que le bras bouge). Jamais
# reteste sur le robot reel depuis ce changement de geometrie -- desactivee par
# precaution en attendant. Remettre a une valeur >0 seulement apres validation.
WALK_STANCE_SCALE = 0.0
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
LEVEE_STIFFNESS = 130.0
# 2026-09-07 : retour utilisateur -- 90 encore juge "un peu faible" (prise/force,
# PAS la vitesse) pendant la levee. 90 -> 130 : a tau=kp*erreur, la saturation moteur
# (50 N.m, Q25H) tombe a ~22deg d'erreur avec kp=130, contre ~29deg a 90 -- reste une
# bonne marge sous les ~11-12deg ou kp=250 saturait et causait l'ERR_MOTOR_FAULT du
# 2026-08-25. Pas encore teste sur le robot reel -- valider avec --only-phase levee
# seul avant d'enchainer la sequence complete ; si toujours insuffisant, remonter par
# petits pas (ex. 130 -> 160) plutot qu'un grand saut. SQUEEZE_Y (phase serrage,
# distincte de la levee) laisse a 0.090 -- a revoir separement si le probleme est la
# prise elle-meme plutot que la tenue en l'air.


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
    lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))


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
    parser.add_argument("--pinch-x", type=float, default=PINCH_X,
                         help="2026-09-15 : expose PINCH_X (distance de portee, defaut "
                              f"{PINCH_X}m calibre pour 14.6cm du bord du podium) en CLI pour "
                              "tester une portee differente sans modifier la constante calibree.")
    parser.add_argument("--pinch-z", type=float, default=0.05,
                         help="2026-09-09 : podium reel remesure a 0.8m de haut (etait 0.535m) "
                              "-- calcul d'origine : carton a 0.8+0.146=0.946m (monde), bassin "
                              "pd_stand ~0.82m -> pinch_z=+0.126. Baisse a +0.106 (essai reel : "
                              "\"leve les bras un peu trop\" a +0.126) -- LIFT_Z suit ce "
                              "changement (voir plus haut). Point de prise reste AU-DESSUS du "
                              "bassin (etait -0.139, en dessous, avant le podium remesure) -- "
                              "geometrie bras-vers-le-haut, a reajuster encore si besoin.")
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
    # AUTRES articulations (solve_arm_ik, lock_index=WRIST_CHAIN_INDEX -- 2026-09-14,
    # consolide ici depuis l'ancienne _solve_ik_locked_wrist locale a ce fichier) -- le
    # poignet tourne alors GARANTI de l'angle demande, plus de hasard du solveur.
    # hand_offset redevient HAND_OFFSET_LEFT/RIGHT tel quel (pas tourne a la main) :
    # forward_kinematics applique deja la rotation du dernier joint au offset, inutile de la
    # precalculer.
    # 2026-09-15 : reference "coude tendu" pour le null-space -- sans null_space_pref
    # explicite, solve_arm_ik retombe sur q_init (ici Q_LEFT_HOME) comme preference par
    # defaut (voir solve_arm_ik dans lift_carton.py) : q_pinch_L n'etait donc jamais
    # garanti d'etre un bras tendu -- constate sur le robot reel (bras qui leve bien mais
    # ne se tend pas). ELBOW_PITCH (index 3) mis a 0 = coude aussi droit que la geometrie
    # le permet, base sur WAYPOINT_Q_LEFT (pas Q_LEFT_HOME) pour rester coherent avec la
    # meme reference que la rampe d'approche ci-dessous.
    straight_pref_L = WAYPOINT_Q_LEFT.copy()
    straight_pref_L[3] = 0.0
    q_pinch_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                              _rotate_xy([args.pinch_x, PINCH_Y, args.pinch_z], args.pinch_yaw_offset),
                              Q_LEFT_HOME, lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                              null_space_pref=straight_pref_L)
    q_pinch_R = mirror_left_to_right(q_pinch_L)
    print(f"[DEBUG] poignet -- qL[-1]={np.degrees(q_pinch_L[-1]):.1f}deg "
          f"qR[-1]={np.degrees(q_pinch_R[-1]):.1f}deg (limite mecanique : +-150deg)")

    # 2026-09-14 : null_space_pref=q_pinch_L ajoute -- sans lui, le solveur (1 DDL redondant
    # apres verrouillage du poignet) peut converger sur une branche d'epaule/coude differente
    # de q_pinch_L pour ce petit deplacement Y (serrage), meme classe de bug/fix que celui
    # trouve cote simu le meme jour (lift.py, squeeze-phase). PAS ENCORE VALIDE sur le robot
    # reel.
    squeeze_L = _rotate_xy([args.pinch_x, SQUEEZE_Y, args.pinch_z], args.pinch_yaw_offset)
    q_squeeze_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_pinch_L,
                                lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                                null_space_pref=q_pinch_L)
    q_squeeze_R = mirror_left_to_right(q_squeeze_L)

    run_approche = args.only_phase in (None, "approche")
    run_serrage = args.only_phase in (None, "serrage")
    run_levee = args.only_phase in (None, "levee")

    qL, qR = Q_LEFT_HOME, Q_RIGHT_HOME

    if run_approche and walk_stance_scale > 0:
        _checkpoint(f"flexion genoux -- scale={walk_stance_scale}, {WALK_STANCE_DURATION}s", confirm)
        _bend_knees(lever, walk_stance_scale, WALK_STANCE_STIFFNESS_SCALE, WALK_STANCE_DURATION,
                    dry_run=args.dry_run)

    if run_approche:
        _checkpoint(
            f"point de passage -- coudes vers l'arriere, {WAYPOINT_DURATION}s",
            confirm,
        )
        qL, qR = move_arms(lever, Q_LEFT_HOME, WAYPOINT_Q_LEFT, Q_RIGHT_HOME, WAYPOINT_Q_RIGHT,
                            WAYPOINT_DURATION, dry_run=args.dry_run)

        # 2026-09-11 : move_arms() interpole en ESPACE ARTICULAIRE (angles lineaires) entre
        # le point de passage et pinch -- constate sur le robot reel que la main monte trop
        # haut en cours de route (arc, pas une ligne droite) meme si les 2 postures aux
        # extremites sont correctes, a cause de la non-linearite de la geometrie du bras.
        # Fix : ramp en ESPACE CARTESIEN comme la levee ci-dessous.
        # 2026-09-15 : mesure numerique -- waypoint_hand_L et pinch_target_L n'ont PAS la
        # meme hauteur (waypoint Z=0.039m, pinch Z=0.106m, ecart 6.7cm), donc la ligne
        # "droite" precedente (waypoint -> pinch directement) montait en diagonale sur toute
        # l'approche -- pas un bug de solveur, une geometrie de bout en bout differente.
        # Demande explicite utilisateur : l'approche doit etre une ligne HORIZONTALE (Z
        # constant), quitte a corriger la hauteur separement avant. Scinde en 2 segments :
        #   1) ajustement vertical (meme X/Y que le point de passage, Z -> hauteur pince)
        #   2) approche horizontale (Z constant = hauteur pince, X/Y -> cible pince)
        pinch_target_L = _rotate_xy([args.pinch_x, PINCH_Y, args.pinch_z], args.pinch_yaw_offset)
        waypoint_hand_L = forward_kinematics(LEFT_CHAIN, HAND_OFFSET_LEFT, WAYPOINT_Q_LEFT)
        raised_point_L = np.array([waypoint_hand_L[0], waypoint_hand_L[1], pinch_target_L[2]])
        qL, qR = WAYPOINT_Q_LEFT.copy(), WAYPOINT_Q_RIGHT.copy()

        _checkpoint(f"approche -- ajustement vertical, {APPROACH_LIFT_DURATION}s", confirm)
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

        _checkpoint(
            f"approche -- ligne horizontale vers pinch (x={args.pinch_x} y=+-{PINCH_Y} "
            f"z={args.pinch_z}, poignet pivote de {np.degrees(wrist_rotation):.0f}deg), "
            f"{APPROACH_DURATION}s",
            confirm,
        )
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

        # 2026-09-14 : null_space_pref=q_squeeze_L (pose FIXE, capturee avant la boucle) au
        # lieu d'un warm-start glissant sur qL_prev -- meme fix que la rampe d'approche
        # ci-dessus et que les rampes cote simu (depose.py/pivot.py).
        qL_prev, qR_prev = q_squeeze_L.copy(), q_squeeze_R.copy()
        n = max(1, int(LIFT_DURATION * RATE_HZ))
        for i in range(n + 1):
            a = ease(i / n)
            z = args.pinch_z + a * (LIFT_Z - args.pinch_z)
            qL_prev = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                                    _rotate_xy([args.pinch_x, SQUEEZE_Y, z], args.pinch_yaw_offset),
                                    qL_prev, lock_index=WRIST_CHAIN_INDEX, lock_angle=wrist_rotation,
                                    iters=30, null_space_pref=q_squeeze_L)
            qR_prev = mirror_left_to_right(qL_prev)
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
