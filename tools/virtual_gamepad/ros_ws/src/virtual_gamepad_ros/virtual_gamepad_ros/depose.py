import signal
import sys
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

sys.path.insert(0, "/home/equansrobotic/stagiaire_1/tools/joint_angle_commander")
from lever import Lever

sys.path.insert(0, "/home/equansrobotic/stagiaire_1/tools/robot_arm_ik")
from lift_carton import (
    LEFT_CHAIN, RIGHT_CHAIN, HAND_OFFSET_LEFT, HAND_OFFSET_RIGHT, solve_ik,
    solve_arm_ik, ease, SimStateListener, world_to_robot_local,
    SQUEEZE_OFFSET_Y, mirror_left_to_right,
)

from virtual_gamepad_interfaces.action import Depose

LEFT_JOINT_INDICES = [13, 14, 15, 16, 17]
RIGHT_JOINT_INDICES = [18, 19, 20, 21, 22]
WAIST_JOINT_INDEX = 12
WAIST_HOLD_KP, WAIST_HOLD_KD = 80.0, 2.0
RETREAT_GAP_Y = 0.125  # identique a lift.py -- meme chaine aim_L, voir _execute

Q_LEFT_HOME = np.array([0.000879, 0.075284, -0.000233, -0.126397, -0.000033])
Q_RIGHT_HOME = np.array([0.000885, -0.075161, 0.000241, -0.126390, 0.000033])

# 2026-09-11 : point de passage "coudes vers l'arriere" -- meme posture, meme
# logique, que WAYPOINT_Q_LEFT/RIGHT dans lift.py (voir sa docstring pour le
# detail de la verification par cinematique directe -- forme en "L" validee
# sur capture d'ecran par l'utilisateur : bras vers l'arriere-bas, coude
# ~90deg, avant-bras HORIZONTAL), utilise ICI dans l'AUTRE sens : en repliant
# les bras le long du corps (degagement, apres le pivot), demande explicite
# de l'utilisateur ("meme logique" au retour).
WAYPOINT_Q_LEFT = np.radians([30.0, 5.0, 0.0, -110.0, 0.0])
WAYPOINT_Q_RIGHT = np.radians([30.0, -5.0, 0.0, -110.0, 0.0])

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


def _quintic_ease(t):
    """Identique a lift.py::_quintic_ease."""
    t = max(0.0, min(1.0, t))
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


def _rotate_xy(point, yaw_offset):
    """Identique a lift.py::_rotate_xy."""
    x, y, z = point
    c, s = np.cos(yaw_offset), np.sin(yaw_offset)
    return np.array([x * c - y * s, x * s + y * c, z])


ELBOW_PITCH_CHAIN_INDEX = 3  # index de ELBOW_PITCH dans LEFT_CHAIN/RIGHT_CHAIN --
                              # voir lift_carton.py::solve_arm_ik (lock_index)

class DeposeActionServer(Node):
    def __init__(self):
        super().__init__("depose")
        self._lever = None
        self._sim_state = None
        self._server = ActionServer(
            self, Depose, "depose", self._execute, cancel_callback=self._on_cancel,
        )

    def _on_cancel(self, goal_handle):
        return CancelResponse.ACCEPT

    def _ensure_sim_state(self) -> SimStateListener:
        """2026-09-11 : meme pattern que lift.py/pivot.py -- process separe
        (propre Lever), doit RECALCULER la position tenue plutot que de
        dependre de g.pinch_x/g.squeeze_y/g.hold_z STATIQUES."""
        if self._sim_state is None:
            self._sim_state = SimStateListener()
            if not self._sim_state.wait_for_first_message(5.0):
                raise RuntimeError(
                    "Aucun message sur le canal LCM 'sim_state' apres 5s -- "
                    "run_mujoco.sh actif ?"
                )
        return self._sim_state

    def _ensure_lever(self) -> Lever:
        if self._lever is None:
            self.get_logger().info(
                "Attente d'un abonne sur /motion/joint_override_command (run.sh actif ?)..."
            )
            self._lever = Lever(self, subscriber_timeout=10.0)
        return self._lever

    def _bend_knees(self, lever, scale, stiffness_scale, duration, qL_hold, qR_hold,
                     waist_hold, rate_hz=30):
        """Identique a lift.py::_bend_knees -- refait ICI car walk_to (entre
        pivot et ce node) a rendu les jambes a pd_stand (jambes DROITES) le
        temps de la marche vers le 2e poste -- il faut refaire la flexion
        avant de bouger les bras, sinon meme risque de bascule que sans
        flexion du tout (cf memoire projet).

        2026-09-10 : republie aussi bras+buste (qL_hold/qR_hold, deja tenus
        par pivot.py juste avant) A CHAQUE tick, pas seulement les jambes --
        /motion/joint_override_command est un REMPLACEMENT COMPLET a chaque
        message (pas une fusion, cf pivot.py), et ce node cree son PROPRE
        Lever (processus separe de celui de pivot.py) : sans ca, le tout
        premier message de ce node (jambes seules) aurait fait lacher le
        carton instantanement, avant meme que ce fichier ne recalcule la
        position des bras plus bas -- meme piege que celui deja corrige
        dans pivot.py, jamais teste en conditions reelles jusqu'ici (le seul
        test de depose.py documente etait isole, avec sa propre prise
        fraiche, pas un relais depuis pivot.py)."""
        for idx, kp, kd in [
            (LEFT_HIP_PITCH_INDEX, 200.0, 5.0), (RIGHT_HIP_PITCH_INDEX, 200.0, 5.0),
            (LEFT_KNEE_PITCH_INDEX, 450.0, 5.0), (RIGHT_KNEE_PITCH_INDEX, 450.0, 5.0),
            (LEFT_ANKLE_PITCH_INDEX, 400.0, 2.0), (RIGHT_ANKLE_PITCH_INDEX, 400.0, 2.0),
        ]:
            lever.set_gains(idx, kp * stiffness_scale, kd * stiffness_scale)
        lever.set_gains(WAIST_JOINT_INDEX, WAIST_HOLD_KP, WAIST_HOLD_KD)
        n = max(1, int(duration * rate_hz))
        leg_indices = [LEFT_HIP_PITCH_INDEX, RIGHT_HIP_PITCH_INDEX, LEFT_KNEE_PITCH_INDEX,
                       RIGHT_KNEE_PITCH_INDEX, LEFT_ANKLE_PITCH_INDEX, RIGHT_ANKLE_PITCH_INDEX]
        for i in range(n + 1):
            a = _quintic_ease(i / n)
            leg_angles = [
                a * scale * WALK_STANCE_HIP_PITCH_L, a * scale * WALK_STANCE_HIP_PITCH_R,
                a * scale * WALK_STANCE_KNEE_L, a * scale * WALK_STANCE_KNEE_R,
                a * scale * WALK_STANCE_ANKLE_PITCH_L, a * scale * WALK_STANCE_ANKLE_PITCH_R,
            ]
            indices = leg_indices + LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES + [WAIST_JOINT_INDEX]
            angles = leg_angles + list(qL_hold) + list(qR_hold) + [float(waist_hold)]
            lever.set_batch(indices, angles)
            time.sleep(1.0 / rate_hz)

    def _move_arms(self, lever, qL0, qL1, qR0, qR1, duration, rate_hz=30):
        n = max(1, int(duration * rate_hz))
        for i in range(n + 1):
            a = ease(i / n)
            qL = qL0 + a * (qL1 - qL0)
            qR = qR0 + a * (qR1 - qR0)
            lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))
            time.sleep(1.0 / rate_hz)
        return qL, qR

    def _straighten_knees(self, lever, scale, stiffness_scale, duration, rate_hz=30):
        """2026-09-11 : identique a lift.py::_straighten_knees -- MANQUAIT dans ce
        fichier (bug reel trouve par telemetrie : le controle natif reprenait la
        main sur des jambes encore artificiellement flechies par _bend_knees, jamais
        redressees avant `_release`, chute au `stand()` qui suit immediatement dans
        chef_node.py). Inverse de _bend_knees() : ramene hanche+genou+cheville de la
        posture flechie (x`scale`) vers droites (angle 0), interpolation quintique
        PARCOURUE A L'ENVERS (a=1->0)."""
        n = max(1, int(duration * rate_hz))
        leg_indices = [LEFT_HIP_PITCH_INDEX, RIGHT_HIP_PITCH_INDEX, LEFT_KNEE_PITCH_INDEX,
                       RIGHT_KNEE_PITCH_INDEX, LEFT_ANKLE_PITCH_INDEX, RIGHT_ANKLE_PITCH_INDEX]
        for i in range(n + 1):
            a = 1.0 - _quintic_ease(i / n)
            lever.set_batch(leg_indices, [
                a * scale * WALK_STANCE_HIP_PITCH_L, a * scale * WALK_STANCE_HIP_PITCH_R,
                a * scale * WALK_STANCE_KNEE_L, a * scale * WALK_STANCE_KNEE_R,
                a * scale * WALK_STANCE_ANKLE_PITCH_L, a * scale * WALK_STANCE_ANKLE_PITCH_R,
            ])
            time.sleep(1.0 / rate_hz)

    def _release(self, lever, qL, qR, ramp_seconds, rate_hz=30):
        """Identique a lift.py::_release."""
        if ramp_seconds > 0:
            n = max(1, int(ramp_seconds * rate_hz))
            for i in range(n + 1):
                lever.set_weight(1.0 - i / n)
                lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))
                time.sleep(1.0 / rate_hz)
        lever.release()

    def _execute(self, goal_handle):
        result = Depose.Result()
        g = goal_handle.request

        try:
            lever = self._ensure_lever()
        except RuntimeError as exc:
            self.get_logger().error(f"depose indisponible : {exc}")
            goal_handle.abort()
            result.success = False
            return result

        # 2026-09-11 : meme bug que pivot.py -- depose.py est un process
        # separe (propre Lever) qui recalculait hold_L/hold_R depuis les
        # valeurs STATIQUES g.pinch_x/g.squeeze_y/g.hold_z passees par
        # chef_node.py, au lieu du centre des faces REELLEMENT vise/tenu.
        # On recalcule ici depuis sim_state + le centre des faces, comme
        # lift.py et pivot.py, pour eviter tout saut a la prise de relais.
        # 2026-09-14 : le centre des faces vient maintenant des champs
        # g.face_gauche_x/y/z (Depose.action) -- CAPTURES UNE SEULE FOIS par
        # chef_node.py et transmis, au lieu que ce node rappelle lui-meme
        # carton_face_centers() -- voir le commentaire detaille dans lift.py.
        try:
            sim_state = self._ensure_sim_state()
        except RuntimeError as exc:
            self.get_logger().error(f"depose : {exc}")
            goal_handle.abort()
            result.success = False
            return result
        pose = sim_state.pose()
        if pose is None:
            self.get_logger().error("depose : aucune pose sim_state disponible -- abandon.")
            goal_handle.abort()
            result.success = False
            return result
        # 2026-09-14 : RETRACT_PINCH_X (forcait X=0.20 des la toute premiere pose,
        # publiee INSTANTANEMENT, pas en rampe) supprime -- pivot.py ne retire plus
        # les bras avant de rendre la main (meme demande utilisateur, voir son
        # commentaire), donc ce X force ne correspondait plus a ce que pivot.py
        # tenait reellement -> saut brusque des avant-bras constate juste apres la
        # fin de la rotation. hold_L/R gardent maintenant hold_L[0]/hold_R[0] tels
        # que calcules (position REELLEMENT tenue, identique a la derniere pose de
        # pivot.py -- MEMES g.face_gauche_x/y/z transmis par chef_node.py, meme
        # pose sim_state lue en direct ici).
        face_gauche_monde = np.array([g.face_gauche_x, g.face_gauche_y, g.face_gauche_z])
        face_droite_monde = np.array([g.face_droite_x, g.face_droite_y, g.face_droite_z])
        pinch_L = world_to_robot_local(face_gauche_monde, pose)
        pinch_R = world_to_robot_local(face_droite_monde, pose)
        pinch_L = np.array([pinch_L[0], pinch_L[1], g.hold_z])
        pinch_R = np.array([pinch_R[0], pinch_R[1], g.hold_z])
        hold_L = np.array([pinch_L[0], pinch_L[1] - SQUEEZE_OFFSET_Y, pinch_L[2]])
        hold_R = np.array([pinch_R[0], pinch_R[1] + SQUEEZE_OFFSET_Y, pinch_R[2]])
        # 2026-09-16 : qL_hold n'est plus resolu directement depuis WAYPOINT_Q_LEFT --
        # verifie numeriquement (comme pour le meme fix dans pivot.py) que ca fait
        # converger le solveur sur une branche epaule/avant-bras DIFFERENTE de celle
        # que lift.py/pivot.py tiennent reellement (jusqu'a ~20deg d'ecart en
        # SHOULDER_YAW pour une position de main quasi identique). lift.py ancre son
        # propre solve sur q_aim_L (issu de RETREAT_GAP_Y), pas sur WAYPOINT_Q_LEFT --
        # on reconstruit ICI exactement la meme chaine pour reconverger sur une
        # posture BIT-A-BIT IDENTIQUE (verifie : diff exactement 0.0).
        aim_L = np.array([pinch_L[0], pinch_L[1] + RETREAT_GAP_Y, pinch_L[2]])
        q_aim_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, aim_L, WAYPOINT_Q_LEFT,
                                lock_index=ELBOW_PITCH_CHAIN_INDEX,
                                lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX])
        qL_hold = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, hold_L, q_aim_L,
                                lock_index=ELBOW_PITCH_CHAIN_INDEX,
                                lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX],
                                null_space_pref=q_aim_L)
        qR_hold = mirror_left_to_right(qL_hold)
        waist_hold = np.radians(g.depivot_from_deg)

        if g.walk_stance:
            self.get_logger().info(
                f"posture jambes -- droites (pd_stand, rendues pendant la marche) -> "
                f"flechies x{g.walk_stance_scale:.1f}, {g.walk_stance_duration:.1f}s "
                "(bras/buste republies en continu pour ne pas lacher le carton)"
            )
            self._bend_knees(lever, g.walk_stance_scale, g.walk_stance_stiffness_scale,
                              g.walk_stance_duration, qL_hold, qR_hold, waist_hold)

        # q_squeeze_L/R = qL_hold/qR_hold, deja calcules a la hauteur reellement
        # tenue (g.hold_z) et deja republies en continu pendant _bend_knees
        # ci-dessus (si walk_stance) -- republie ici une derniere fois pour
        # couvrir le cas walk_stance=false (jamais publie sinon).
        q_squeeze_L, q_squeeze_R = qL_hold, qR_hold
        lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES + [WAIST_JOINT_INDEX],
                         list(q_squeeze_L) + list(q_squeeze_R) + [float(waist_hold)])

        self.get_logger().info(
            f"tendre les bras -- coudes redresses sur place, meme position de main "
            f"({g.tendre_duration:.1f}s)"
        )
        q_tendu_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, hold_L, q_squeeze_L,
                                  lock_index=ELBOW_PITCH_CHAIN_INDEX,
                                  lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX])
        q_tendu_R = mirror_left_to_right(q_tendu_L)
        qL, qR = self._move_arms(lever, q_squeeze_L, q_tendu_L, q_squeeze_R, q_tendu_R,
                                  g.tendre_duration)

        self.get_logger().info(
            f"depose -- Z {hold_L[2]:.3f} -> {g.drop_z:.3f} ({g.depose_duration:.1f}s) "
            "-- LE CARTON REDESCEND, TOUJOURS SERRE"
        )
        n = max(1, int(g.depose_duration * 30))
        q_drop_start_L = qL.copy()
        for i in range(n + 1):
            a = ease(i / n)
            z = hold_L[2] + a * (g.drop_z - hold_L[2])
            qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                               np.array([hold_L[0], hold_L[1], z]), qL,
                               lock_index=ELBOW_PITCH_CHAIN_INDEX,
                               lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX], iters=30,
                               null_space_pref=q_drop_start_L)
            qR = mirror_left_to_right(qL)
            lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))
            time.sleep(1.0 / 30)

        # 2026-09-11 : "tendre les bras" AVANT relachement -- sur demande
        # explicite de l'utilisateur ("ajoute tendre les bras avant de lacher
        # le carton"). Depuis la reactivation du serrage (SQUEEZE_OFFSET_Y),
        # les mains sont encore ENFONCEES de 15mm dans le carton a ce stade
        # (le meme decalage tenu depuis lift.py) -- sauter directement vers le
        # point de passage "coudes vers l'arriere" (angles fixes, pas d'IK)
        # les ferait potentiellement racler/traverser le carton en chemin.
        # Cette etape ouvre d'abord la prise EN LIGNE DROITE (IK, coude
        # toujours fige) jusqu'au centre de face (desserre exactement les
        # SQUEEZE_OFFSET_Y de serrage), puis SEULEMENT ensuite le degagement
        # (coudes vers l'arriere) peut suivre sans accrocher le carton.
        self.get_logger().info(
            f"tendre les bras -- ouverture du serrage avant relachement ({g.tendre_duration:.1f}s)"
        )
        # Z = g.drop_z (hauteur REELLEMENT atteinte a la fin de la descente
        # ci-dessus), PAS hold_L[2]/hold_R[2] qui valent toujours la hauteur
        # de DEPART (jamais reassignes par la boucle de descente).
        release_L = np.array([hold_L[0], hold_L[1] + SQUEEZE_OFFSET_Y, g.drop_z])
        q_release_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, release_L, qL,
                                    lock_index=ELBOW_PITCH_CHAIN_INDEX,
                                    lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX])
        q_release_R = mirror_left_to_right(q_release_L)
        qL, qR = self._move_arms(lever, qL, q_release_L, qR, q_release_R, g.tendre_duration)

        # 2026-09-14 : etape d'ECARTEMENT ajoutee sur demande explicite de l'utilisateur
        # ("il faut ecarter les bras avant de les retirer") -- avant, le degagement
        # sautait DIRECTEMENT de release_L (mains encore proches du carton, juste
        # ouvertes de SQUEEZE_OFFSET_Y ~9.5cm) vers WAYPOINT_Q_LEFT/RIGHT (posture
        # articulaire fixe, coudes vers l'arriere) -- les mains restaient donc pres du
        # volume du carton/podium pendant tout le debut du retrait. Ecarte ici
        # explicitement en cartesien (meme X/Z que release_L, Y ELARGI) avant de
        # rejoindre le point de passage. PAS ENCORE VALIDE en sim.
        ECARTEMENT_GAP_Y = 0.08
        ECARTEMENT_DURATION = 1.0
        self.get_logger().info(f"ecartement -- mains ecartees avant retrait ({ECARTEMENT_DURATION:.1f}s)")
        ecart_L = np.array([hold_L[0], hold_L[1] + ECARTEMENT_GAP_Y, g.drop_z])
        q_ecart_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, ecart_L, qL,
                                  lock_index=ELBOW_PITCH_CHAIN_INDEX,
                                  lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX],
                                  null_space_pref=qL)
        q_ecart_R = mirror_left_to_right(q_ecart_L)
        qL, qR = self._move_arms(lever, qL, q_ecart_L, qR, q_ecart_R, ECARTEMENT_DURATION)

        # 2026-09-14 : translation ARRIERE (X) ajoutee sur demande explicite de
        # l'utilisateur ("il repasse par le serrage... translate vers l'arriere") --
        # sans ca, le saut EN ESPACE ARTICULAIRE (_move_arms) de q_ecart_L vers
        # WAYPOINT_Q_LEFT/RIGHT (posture tres differente, coude 0deg -> -110deg) ne
        # garantit RIEN sur la trajectoire de la main en cartesien -- elle peut
        # repasser pres du point de serrage avant de rejoindre le point de passage.
        # Ramp cartesien EN X SEUL (meme technique que le retrait de pivot.py),
        # main tiree pres du corps AVANT le saut vers WAYPOINT_Q -- ce saut devient
        # alors beaucoup plus court/sur puisque la main est deja proche du corps.
        # PAS ENCORE VALIDE en sim.
        RETREAT_BACK_X = 0.05
        RETREAT_BACK_DURATION = 1.0
        self.get_logger().info(f"translation arriere -- main ramenee pres du corps ({RETREAT_BACK_DURATION:.1f}s)")
        n_retreat = max(1, int(RETREAT_BACK_DURATION * 30))
        q_retreat_start_L = qL.copy()
        for i in range(n_retreat + 1):
            a = ease(i / n_retreat)
            x = ecart_L[0] + a * (RETREAT_BACK_X - ecart_L[0])
            qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                               np.array([x, ecart_L[1], ecart_L[2]]), qL,
                               lock_index=ELBOW_PITCH_CHAIN_INDEX,
                               lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX], iters=30,
                               null_space_pref=q_retreat_start_L)
            qR = mirror_left_to_right(qL)
            lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES, list(qL) + list(qR))
            time.sleep(1.0 / 30)

        self.get_logger().info(
            f"degagement -- coudes vers l'arriere ({g.degagement_waypoint_duration:.1f}s) "
            "-- LE CARTON EST RELACHE ICI (l'ecartement des mains vers ce point de passage "
            "suffit, plus besoin d'un desserrage separe)"
        )
        qL, qR = self._move_arms(lever, qL, WAYPOINT_Q_LEFT, qR, WAYPOINT_Q_RIGHT,
                                  g.degagement_waypoint_duration)

        if g.depivot_from_deg != 0.0:
            # 2026-09-16 : depivot AVANT le retour bras home (etait apres) --
            # rapporte par l'utilisateur : "grosse perte d'equilibre" visible
            # pile quand les bras reviennent le long du corps, meme a vitesse
            # normale (donc pas un souci de vitesse/elan). Hypothese : les bras
            # arrivaient a la posture Q_HOME (repliee contre le corps) pendant
            # que le buste etait ENCORE tourne (ex: 90deg) -- masse des bras
            # decalee par rapport aux pieds (qui eux n'ont pas tourne), donc
            # desequilibre STATIQUE meme sans mouvement rapide. Le carton est
            # deja lache a ce stade (voir "degagement -- coudes vers
            # l'arriere" ci-dessus), donc redresser le buste avant plutot
            # qu'apres ne pose pas le risque de chute-avec-charge deja gere
            # dans pivot.py. VALIDE en sim (utilisateur : "l'inversion a l'air
            # nickel").
            self.get_logger().info(
                f"depivot -- {g.depivot_from_deg:.0f}deg -> 0deg ({g.depivot_duration:.1f}s), "
                "carton deja lache, buste seul (avant retour bras home)"
            )
            n_depivot = max(1, int(g.depivot_duration * 30))
            for i in range(n_depivot + 1):
                a = ease(i / n_depivot)
                waist = (1.0 - a) * waist_hold
                lever.set_batch(LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES + [WAIST_JOINT_INDEX],
                                 list(qL) + list(qR) + [float(waist)])
                time.sleep(1.0 / 30)

        self.get_logger().info(f"degagement -- retour bras home ({g.degagement_duration:.1f}s)")
        qL, qR = self._move_arms(lever, qL, Q_LEFT_HOME, qR, Q_RIGHT_HOME, g.degagement_duration)

        if g.walk_stance:
            self.get_logger().info(
                f"retrait jambes -- flechies -> droites ({g.walk_stance_duration:.1f}s)"
            )
            self._straighten_knees(lever, g.walk_stance_scale, g.walk_stance_stiffness_scale,
                                    g.walk_stance_duration)

        self.get_logger().info(f"relachement final -- rampe {g.release_ramp_seconds:.1f}s")
        self._release(lever, qL, qR, g.release_ramp_seconds)

        goal_handle.succeed()
        result.success = True
        return result


def main():
    rclpy.init()
    node = DeposeActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
