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
    solve_arm_ik, ease, carton_face_centers, SimStateListener, world_to_robot_local,
    SQUEEZE_OFFSET_Y, mirror_left_to_right,
)

from virtual_gamepad_interfaces.action import Lift

ELBOW_PITCH_CHAIN_INDEX = 3  # index de ELBOW_PITCH dans LEFT_CHAIN/RIGHT_CHAIN --
                              # voir lift_carton.py::solve_arm_ik (lock_index)

LEFT_JOINT_INDICES = [13, 14, 15, 16, 17]
RIGHT_JOINT_INDICES = [18, 19, 20, 21, 22]

Q_LEFT_HOME = np.array([0.000879, 0.075284, -0.000233, -0.126397, -0.000033])
Q_RIGHT_HOME = np.array([0.000885, -0.075161, 0.000241, -0.126390, 0.000033])

# 2026-09-11 : point de passage "coudes vers l'arriere, avant-bras a hauteur de
# prise" avant d'etendre les bras vers le carton, sur demande explicite de
# l'utilisateur. 2 essais precedents rejetes par l'utilisateur (voir capture
# d'ecran partagee -- forme en "L" : bras qui descend vers l'ARRIERE depuis
# l'epaule, coude plie ~90deg, avant-bras HORIZONTAL, PAS remonte) :
#   1) cible cartesienne proche du corps a la MEME hauteur via IK standard --
#      l'IK a converge vers les bras qui se LEVENT (coudes vers le HAUT), pas
#      vers l'arriere ("XDDD").
#   2) posture articulaire fixe SHOULDER_PITCH=+40/ELBOW_PITCH=-129 -- coude
#      bien derriere MAIS avant-bras replie vers le haut (pas horizontal),
#      "pas ce que je veux".
# Corrige (3e essai, valide sur capture d'ecran) : SHOULDER_PITCH=+30deg
# (bascule l'epaule/coude vers l'ARRIERE-BAS, coude en x=-0.124 contre epaule
# en x=-0.027, ~10cm derriere), ELBOW_PITCH=-110deg -- verifie par cinematique
# directe (forward_kinematics hors robot) : vecteur coude->main = [0.297,
# -0.023, -0.009] -- quasi parfaitement HORIZONTAL (deviation verticale <1cm),
# main a [0.173, 0.213, 0.039]. SHOULDER_ROLL=+-5deg (marge [-35,135]deg large).
# Indices du chain : [SHOULDER_PITCH, SHOULDER_ROLL, SHOULDER_YAW, ELBOW_PITCH,
# ELBOW_YAW] -- meme convention miroir que Q_LEFT/RIGHT_HOME (SHOULDER_PITCH et
# ELBOW_PITCH identiques des deux cotes, SHOULDER_ROLL/YAW et ELBOW_YAW inverses).
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

LEVEE_ARM_STIFFNESS = 150.0


def _quintic_ease(t):
    """Meme forme que math::QuinticInterpolate utilise par pd_stand_runner.cc
    pour sa propre transition initiale (vitesse ET acceleration nulles aux
    deux bords) -- imite fidelement le PD natif de pd_stand quand on vise
    une cible differente (genoux flechis), cf lift_carton_real.py."""
    t = max(0.0, min(1.0, t))
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


def _rotate_xy(point, yaw_offset):
    """Tourne (X,Y) autour de Z de yaw_offset (radians), Z inchange -- corrige la cible
    de pince quand le robot ne s'arrete pas parfaitement de face au carton."""
    x, y, z = point
    c, s = np.cos(yaw_offset), np.sin(yaw_offset)
    return np.array([x * c - y * s, x * s + y * c, z])


class LiftActionServer(Node):
    def __init__(self):
        super().__init__("lift")
        self._lever = None
        self._sim_state = None
        self._server = ActionServer(
            self, Lift, "lift", self._execute, cancel_callback=self._on_cancel,
        )

    def _on_cancel(self, goal_handle):
        return CancelResponse.ACCEPT

    def _ensure_sim_state(self) -> SimStateListener:
        """2026-09-11 : demande explicite de l'utilisateur ("je veux que chaque
        bras vise les coordonnees du centre des faces [du carton]") -- lazy comme
        _ensure_lever ci-dessous, un seul abonnement LCM reutilise entre tous les
        goals de ce process."""
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

    def _bend_knees(self, lever, scale, stiffness_scale, duration, rate_hz=30):
        """Transition jambes droites (pd_stand) -> flechies (posture walk au
        repos x`scale`), hanche+genou+cheville COORDONNES, gains Kp/Kd
        renforces (x`stiffness_scale` des valeurs natives pd_stand : 200/5
        hanche, 450/5 genou, 400/2 cheville), interpolation quintique comme
        pd_stand_runner.cc. Porte de lift_carton_real.py (script robot reel),
        jamais fait dans ce node avant le 20/08."""
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

    def _move_arms(self, lever, qL0, qL1, qR0, qR1, duration, rate_hz=30):
        n = max(1, int(duration * rate_hz))
        for i in range(n + 1):
            a = ease(i / n)
            qL = qL0 + a * (qL1 - qL0)
            qR = qR0 + a * (qR1 - qR0)
            for idx, angle in zip(LEFT_JOINT_INDICES, qL):
                lever[idx] = float(angle)
            for idx, angle in zip(RIGHT_JOINT_INDICES, qR):
                lever[idx] = float(angle)
            time.sleep(1.0 / rate_hz)
        return qL, qR

    def _hold(self, lever, qL, qR, goal_handle, duration, rate_hz=10):
        """Republie (qL, qR) en continu pendant `duration`s -- renvoie False si annule."""
        elapsed = 0.0
        step = 1.0 / rate_hz
        while elapsed < duration:
            if goal_handle.is_cancel_requested:
                return False
            for idx, angle in zip(LEFT_JOINT_INDICES, qL):
                lever[idx] = float(angle)
            for idx, angle in zip(RIGHT_JOINT_INDICES, qR):
                lever[idx] = float(angle)
            time.sleep(step)
            elapsed += step
        return True

    def _straighten_knees(self, lever, scale, stiffness_scale, duration, rate_hz=30):
        """Inverse de _bend_knees() : ramene les jambes de la posture flechie
        (x`scale`) vers droites (pd_stand natif, angle 0), meme interpolation
        quintique mais PARCOURUE A L'ENVERS (a=1->0 au lieu de 0->1) -- cf.
        docstring _release() ci-dessous, meme raison : liberer les joints en
        override alors qu'ils sont loin de la cible native du controleur
        actif produit un saut brutal a la reprise de controle."""
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

    def _release(self, lever, qL, qR, ramp_seconds, rate_hz=30):
        """Rampe le poids de l'override 1.0 -> 0.0 avant release() -- un
        release() instantane rend le controle des bras d'un coup a la
        politique active, chute visible et brutale des bras sinon."""
        if ramp_seconds > 0:
            n = max(1, int(ramp_seconds * rate_hz))
            for i in range(n + 1):
                lever.set_weight(1.0 - i / n)
                for idx, angle in zip(LEFT_JOINT_INDICES, qL):
                    lever[idx] = float(angle)
                for idx, angle in zip(RIGHT_JOINT_INDICES, qR):
                    lever[idx] = float(angle)
                time.sleep(1.0 / rate_hz)
        lever.release()

    def _execute(self, goal_handle):
        result = Lift.Result()
        g = goal_handle.request

        try:
            lever = self._ensure_lever()
        except RuntimeError as exc:
            self.get_logger().error(f"lift indisponible : {exc}")
            goal_handle.abort()
            result.success = False
            return result

        run_approche = g.only_phase in ("", "approche")
        run_serrage = g.only_phase in ("", "serrage")
        if run_approche:
            lever.forget()
        run_levee = g.only_phase in ("", "levee")

        if g.walk_stance:
            self.get_logger().info(
                f"posture jambes -- droites (pd_stand) -> flechies x{g.walk_stance_scale:.1f} "
                f"(hanche+genou+cheville, gains x{g.walk_stance_stiffness_scale:.1f}), "
                f"{g.walk_stance_duration:.1f}s"
            )
            self._bend_knees(lever, g.walk_stance_scale, g.walk_stance_stiffness_scale,
                              g.walk_stance_duration)
            self.get_logger().info("pause 2.0s (flexion genoux terminee)")
            time.sleep(2.0)

        # Posture articulaire FIXE (pas de cible cartesienne, pas de pinch_yaw_offset --
        # toujours 0.0 en pratique cote chef_node.py) -- voir WAYPOINT_Q_LEFT/RIGHT.
        q_waypoint_L = WAYPOINT_Q_LEFT.copy()
        q_waypoint_R = WAYPOINT_Q_RIGHT.copy()

        # 2026-09-11 : "tendre les bras" (coudes redresses via solve_arm_ik, coude
        # fige) vise desormais DIRECTEMENT les coordonnees REELLES du centre des
        # faces du carton -- demande explicite de l'utilisateur ("je veux que
        # chaque bras vise les coordonnees du centre des faces") -- plus de
        # pinch_x/pinch_y fixes a deviner/recalibrer a chaque fois que le carton
        # ou la marche changent. carton_face_centers() lit la scene live (jamais
        # perime) en repere MONDE, world_to_robot_local() les convertit en
        # repere bassin avec la pose ACTUELLE du robot (lue en direct via LCM
        # sim_state, PAS une pose supposee/mesuree avant une marche precedente --
        # lecon deja tiree ailleurs dans ce projet sur la derive du robot).
        sim_state = self._ensure_sim_state()
        pose = sim_state.pose()
        if pose is None:
            self.get_logger().error("lift : aucune pose sim_state disponible -- abandon.")
            goal_handle.abort()
            result.success = False
            return result
        face_gauche_monde, face_droite_monde = carton_face_centers()
        pinch_L = world_to_robot_local(face_gauche_monde, pose)
        pinch_R = world_to_robot_local(face_droite_monde, pose)

        # 2026-09-11 : REDESIGN sur demande explicite de l'utilisateur --
        # "lors de la visee je veux qu'il ne touche pas le carton (juste vise
        # les coordonnees en x et z) et y sera utile pour le serrage". X/Z
        # visent donc DEJA exactement pinch_L/pinch_R (precis des la visee),
        # mais Y reste en RETRAIT (RETREAT_GAP_Y au-dela de la face, meme
        # ecart que l'ancien PINCH_Y-SQUEEZE_Y=0.22-0.095=0.125m avant la
        # fusion visee+serrage) -- la main ne touche pas encore le carton.
        # Le serrage (plus bas) ferme ensuite TOUTE la distance Y restante,
        # de aim_L/R jusqu'a squeeze_L/R (legerement au-dela de la surface).
        RETREAT_GAP_Y = 0.125
        aim_L = np.array([pinch_L[0], pinch_L[1] + RETREAT_GAP_Y, pinch_L[2]])
        aim_R = np.array([pinch_R[0], pinch_R[1] - RETREAT_GAP_Y, pinch_R[2]])
        q_aim_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, aim_L, q_waypoint_L,
                                lock_index=ELBOW_PITCH_CHAIN_INDEX,
                                lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX])
        q_aim_R = mirror_left_to_right(q_aim_L)

        # cible de SERRAGE -- au-dela de la surface (centre de face) vers
        # l'interieur du carton de SQUEEZE_OFFSET_Y, pour generer une vraie
        # force normale/friction. +Y = gauche du robot (cf
        # carton_face_centers()) donc "vers l'interieur" = Y qui DIMINUE pour
        # la main gauche (positive), Y qui AUGMENTE pour la main droite
        # (negative) -- seed = q_aim (position REELLE de la main a ce stade,
        # pas encore au contact), coude toujours fige. squeeze_L/R servent
        # aussi de cible pendant la levee, pour ne pas relacher la prise en
        # montant.
        squeeze_L = np.array([pinch_L[0], pinch_L[1] - SQUEEZE_OFFSET_Y, pinch_L[2]])
        squeeze_R = np.array([pinch_R[0], pinch_R[1] + SQUEEZE_OFFSET_Y, pinch_R[2]])
        q_squeeze_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_aim_L,
                                    lock_index=ELBOW_PITCH_CHAIN_INDEX,
                                    lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX])
        q_squeeze_R = mirror_left_to_right(q_squeeze_L)

        if run_approche:
            self.get_logger().info(
                f"point de passage -- coudes vers l'arriere, avant-bras a hauteur de prise "
                f"({g.waypoint_duration:.1f}s)"
            )
            self._move_arms(lever, Q_LEFT_HOME, q_waypoint_L, Q_RIGHT_HOME, q_waypoint_R, g.waypoint_duration)
            self.get_logger().info(
                f"tendre les bras -- x/z exacts du carton, y en retrait "
                f"(gauche={np.round(aim_L, 3)}, droite={np.round(aim_R, 3)}, "
                f"repere bassin, carton a Y=+-{pinch_L[1]:.3f}/{pinch_R[1]:.3f}) ({g.approach_duration:.1f}s)"
            )
            self._move_arms(lever, q_waypoint_L, q_aim_L, q_waypoint_R, q_aim_R, g.approach_duration)
            self.get_logger().info("pause 2.0s (approche terminee)")
            self._hold(lever, q_aim_L, q_aim_R, goal_handle, 2.0)

        if run_serrage:
            # 2026-09-11 : serrage ferme desormais TOUTE la distance Y depuis
            # le retrait (q_aim, atteint par l'appel only_phase="approche"
            # precedent, main pas encore au contact) jusqu'a q_squeeze
            # (legerement au-dela de la surface), pour une vraie prise
            # (friction/force normale).
            self.get_logger().info(
                f"serrage -- fermeture Y (retrait -> surface + {SQUEEZE_OFFSET_Y*1000:.0f}mm) "
                f"({g.squeeze_duration:.1f}s)"
            )
            self._move_arms(lever, q_aim_L, q_squeeze_L, q_aim_R, q_squeeze_R, g.squeeze_duration)

        if not run_levee:
            self.get_logger().info(
                f"only_phase={g.only_phase!r} termine -- bras tenus a leur derniere position, "
                "pas de levee/relachement dans cet appel."
            )
            goal_handle.succeed()
            result.success = True
            return result

        if g.only_phase == "levee":
            self.get_logger().info(
                f"only_phase=levee : publication directe de la position serree "
                f"(gauche={np.round(squeeze_L, 3)}, droite={np.round(squeeze_R, 3)}), "
                "suppose deja atteinte par un appel precedent."
            )
            for idx, angle in zip(LEFT_JOINT_INDICES, q_squeeze_L):
                lever[idx] = float(angle)
            for idx, angle in zip(RIGHT_JOINT_INDICES, q_squeeze_R):
                lever[idx] = float(angle)
            self._hold(lever, q_squeeze_L, q_squeeze_R, goal_handle, 1.0)

        self.get_logger().info(
            f"levee -- Z {squeeze_L[2]:.3f} -> {g.lift_z:.3f} ({g.lift_duration:.1f}s), "
            f"rigidite bras reduite a {LEVEE_ARM_STIFFNESS:.0f} (etait {250.0:.0f})"
        )
        for idx in LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES:
            lever.set_gains(idx, stiffness=LEVEE_ARM_STIFFNESS)
        qL, qR = q_squeeze_L.copy(), q_squeeze_R.copy()
        n = max(1, int(g.lift_duration * 30))
        for i in range(n + 1):
            a = ease(i / n)
            z = squeeze_L[2] + a * (g.lift_z - squeeze_L[2])
            qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                               np.array([squeeze_L[0], squeeze_L[1], z]), qL,
                               lock_index=ELBOW_PITCH_CHAIN_INDEX,
                               lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX], iters=30,
                               null_space_pref=q_squeeze_L)
            qR = mirror_left_to_right(qL)
            for idx, angle in zip(LEFT_JOINT_INDICES, qL):
                lever[idx] = float(angle)
            for idx, angle in zip(RIGHT_JOINT_INDICES, qR):
                lever[idx] = float(angle)
            time.sleep(1.0 / 30)

        self.get_logger().info(f"maintien -- {g.hold_seconds:.1f}s")
        if not self._hold(lever, qL, qR, goal_handle, g.hold_seconds):
            self.get_logger().info("annule pendant le maintien -- relachement immediat.")
            self._release(lever, qL, qR, g.release_ramp_seconds)
            goal_handle.canceled()
            result.success = False
            return result

        if g.release_after:
            self.get_logger().info(
                f"pose -- Z {g.lift_z:.3f} -> {squeeze_L[2]:.3f} ({g.lift_duration:.1f}s)"
            )
            n = max(1, int(g.lift_duration * 30))
            q_start_L, q_start_R = qL.copy(), qR.copy()
            for i in range(n + 1):
                a = ease(i / n)
                z = g.lift_z + a * (squeeze_L[2] - g.lift_z)
                qL = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                                   np.array([squeeze_L[0], squeeze_L[1], z]), qL,
                                   lock_index=ELBOW_PITCH_CHAIN_INDEX,
                                   lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX], iters=30,
                                   null_space_pref=q_start_L)
                qR = mirror_left_to_right(qL)
                for idx, angle in zip(LEFT_JOINT_INDICES, qL):
                    lever[idx] = float(angle)
                for idx, angle in zip(RIGHT_JOINT_INDICES, qR):
                    lever[idx] = float(angle)
                time.sleep(1.0 / 30)
            for idx in LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES:
                lever.set_gains(idx, stiffness=250.0)

            self.get_logger().info(
                f"retrait bras -- retour vers la posture de repos ({g.approach_duration:.1f}s), "
                "avant relachement (evite le saut de position a la reprise de controle native)"
            )
            qL, qR = self._move_arms(lever, qL, Q_LEFT_HOME, qR, Q_RIGHT_HOME, g.approach_duration)
            self.get_logger().info(
                f"retrait jambes -- flechies -> droites ({g.walk_stance_duration:.1f}s)"
            )
            self._straighten_knees(lever, g.walk_stance_scale, g.walk_stance_stiffness_scale,
                                    g.walk_stance_duration)
            self.get_logger().info(f"relachement -- rampe {g.release_ramp_seconds:.1f}s")
            self._release(lever, qL, qR, g.release_ramp_seconds)
        else:
            self.get_logger().info(
                "release_after=false -- pas de relachement, bras/jambes restent tenus "
                "(un pivot est cense suivre)."
            )

        goal_handle.succeed()
        result.success = True
        return result


def main():
    rclpy.init()
    node = LiftActionServer()
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
