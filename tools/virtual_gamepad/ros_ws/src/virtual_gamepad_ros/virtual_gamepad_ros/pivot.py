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

ELBOW_PITCH_CHAIN_INDEX = 3  # voir lift_carton.py::solve_arm_ik (lock_index)

from virtual_gamepad_interfaces.action import Pivot

LEFT_JOINT_INDICES = [13, 14, 15, 16, 17]
RIGHT_JOINT_INDICES = [18, 19, 20, 21, 22]
WAIST_JOINT_INDEX = 12
WAIST_KP_HOLD, WAIST_KD_HOLD = 500.0, 10.0
# 2026-09-10 : 150/3.0 (valeur reelle) suffisait a eviter la chute quand la
# marche precedente passait par l'ancienne emulation manette directe, mais
# pas quand walk_to passe par /motion/body_vel_cmd + body_vel_bridge (meme
# distance finale mesuree, x~1.29m dans les deux cas, mais chute reproduite
# au meme instant -- relachement du pivot -- avec ce chemin de marche,
# posture/elan residuel apparemment different). Durci encore pour absorber
# cette marge plus fine.
WAIST_KP_RELEASE, WAIST_KD_RELEASE = 80.0, 2.0
LEG_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

Q_LEFT_HOME = np.array([0.000879, 0.075284, -0.000233, -0.126397, -0.000033])
Q_RIGHT_HOME = np.array([0.000885, -0.075161, 0.000241, -0.126390, 0.000033])

# 2026-09-11 : meme valeurs que lift.py/depose.py (posture "coudes vers
# l'arriere" validee -- voir leur docstring). Utilisee ici comme SEED pour
# l'IK du carton tenu, PAS pour bouger physiquement les bras vers ce point
# de passage (pivot.py ne le fait jamais lui-meme) -- juste pour que le
# solveur converge vers une posture PROCHE de celle que lift.py tenait
# reellement, au lieu de Q_LEFT_HOME (posture de REPOS, tres differente
# d'un bras tendu vers le carton) qui faisait deriver l'epaule vers une
# orientation visiblement differente au relais (constate par l'utilisateur :
# "les bras pivotent... il y a 2 fois un serrage").
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

LEG_JOINTS = [
    (LEFT_HIP_PITCH_INDEX, WALK_STANCE_HIP_PITCH_L, 200.0, 5.0),
    (RIGHT_HIP_PITCH_INDEX, WALK_STANCE_HIP_PITCH_R, 200.0, 5.0),
    (LEFT_KNEE_PITCH_INDEX, WALK_STANCE_KNEE_L, 450.0, 5.0),
    (RIGHT_KNEE_PITCH_INDEX, WALK_STANCE_KNEE_R, 450.0, 5.0),
    (LEFT_ANKLE_PITCH_INDEX, WALK_STANCE_ANKLE_PITCH_L, 400.0, 2.0),
    (RIGHT_ANKLE_PITCH_INDEX, WALK_STANCE_ANKLE_PITCH_R, 400.0, 2.0),
]


def _rotate_xy(point, yaw_offset):
    """Identique a lift.py::_rotate_xy -- tourne (X,Y) autour de Z."""
    x, y, z = point
    c, s = np.cos(yaw_offset), np.sin(yaw_offset)
    return np.array([x * c - y * s, x * s + y * c, z])


class PivotActionServer(Node):
    def __init__(self):
        super().__init__("pivot")
        self._lever = None
        self._sim_state = None
        self._server = ActionServer(
            self, Pivot, "pivot", self._execute, cancel_callback=self._on_cancel,
        )

    def _on_cancel(self, goal_handle):
        return CancelResponse.ACCEPT

    def _ensure_lever(self) -> Lever:
        if self._lever is None:
            self.get_logger().info(
                "Attente d'un abonne sur /motion/joint_override_command (run.sh actif ?)..."
            )
            self._lever = Lever(self, subscriber_timeout=10.0)
        return self._lever

    def _ensure_sim_state(self) -> SimStateListener:
        """2026-09-11 : meme pattern que lift.py -- pivot.py est un process
        SEPARE (son propre Lever), il doit RECALCULER la position tenue plutot
        que de dependre d'une valeur pinch_x/pinch_y STATIQUE passee par
        chef_node.py (bug reel trouve : l'ancienne version utilisait
        g.pinch_x=PROVEN_PINCH_X fige, different de ce que lift.py avait
        REELLEMENT atteint dynamiquement -- saut visible au relais, remarque
        par l'utilisateur ("il vise vers le centre... pas ce que je veux"))."""
        if self._sim_state is None:
            self._sim_state = SimStateListener()
            if not self._sim_state.wait_for_first_message(5.0):
                raise RuntimeError(
                    "Aucun message sur le canal LCM 'sim_state' apres 5s -- "
                    "run_mujoco.sh actif ?"
                )
        return self._sim_state

    def _publish_pose(self, lever, qL, qR, waist_angle, leg_targets, stiffness_scale):
        """Republie EN UN SEUL message (via Lever, qui agrege tout ce qui a
        deja ete touche) bras+jambes+buste -- voir note en tete de fichier."""
        for idx, angle in zip(LEFT_JOINT_INDICES, qL):
            lever[idx] = float(angle)
        for idx, angle in zip(RIGHT_JOINT_INDICES, qR):
            lever[idx] = float(angle)
        for idx, target, kp, kd in leg_targets:
            lever.set_gains(idx, kp * stiffness_scale, kd * stiffness_scale)
            lever[idx] = float(target)
        lever[WAIST_JOINT_INDEX] = float(waist_angle)

    def _execute(self, goal_handle):
        result = Pivot.Result()
        g = goal_handle.request

        try:
            lever = self._ensure_lever()
        except RuntimeError as exc:
            self.get_logger().error(f"pivot indisponible : {exc}")
            goal_handle.abort()
            result.success = False
            return result

        # 2026-09-11 : position tenue RECALCULEE dynamiquement (centre reel des
        # faces du carton, meme technique que lift.py -- carton_face_centers()
        # + world_to_robot_local() avec la pose ACTUELLE du robot) au lieu des
        # anciennes cibles g.pinch_x/g.pinch_y/g.squeeze_y STATIQUES -- bug reel
        # trouve : pivot.py (process separe, propre Lever) recalculait depuis
        # Q_HOME avec des valeurs figees, differentes de ce que lift.py avait
        # REELLEMENT atteint -- saut visible au relais entre les deux nodes.
        try:
            sim_state = self._ensure_sim_state()
        except RuntimeError as exc:
            self.get_logger().error(f"pivot : {exc}")
            goal_handle.abort()
            result.success = False
            return result
        pose = sim_state.pose()
        if pose is None:
            self.get_logger().error("pivot : aucune pose sim_state disponible -- abandon.")
            goal_handle.abort()
            result.success = False
            return result
        face_gauche_monde, face_droite_monde = carton_face_centers()
        pinch_L = world_to_robot_local(face_gauche_monde, pose)
        pinch_R = world_to_robot_local(face_droite_monde, pose)
        # 2026-09-11 : lift.py serre desormais de SQUEEZE_OFFSET_Y au-dela de
        # la surface (voir lift.py) -- pivot.py doit tenir la MEME position
        # serree (pas juste le centre de face) pour ne pas relacher la prise
        # au relais lift -> pivot.
        pinch_L = np.array([pinch_L[0], pinch_L[1] - SQUEEZE_OFFSET_Y, g.lift_z])
        pinch_R = np.array([pinch_R[0], pinch_R[1] + SQUEEZE_OFFSET_Y, g.lift_z])
        # 2026-09-11 : solve_arm_ik (coude fige + regularisation null-space)
        # au lieu de solve_ik standard -- sinon ce process (pivot.py, propre
        # Lever) recalcule une posture INDEPENDANTE de celle que lift.py
        # tenait reellement (meme position XYZ, mais coude/epaule
        # potentiellement tres differents, cf solve_ik = "solution la plus
        # proche du seed" et le seed ici etait Q_LEFT_HOME, PAS la posture
        # coude-tendu de lift.py) -- saut brutal au relais, prise
        # asymetrique/desequilibree et geste sec au relachement (constate
        # par l'utilisateur : "le carton n'est pas equilibre... il lance le
        # carton"). Meme convention que lift.py partout desormais.
        q_squeeze_L = solve_arm_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, pinch_L, WAYPOINT_Q_LEFT,
                                    lock_index=ELBOW_PITCH_CHAIN_INDEX,
                                    lock_angle=Q_LEFT_HOME[ELBOW_PITCH_CHAIN_INDEX])
        q_squeeze_R = mirror_left_to_right(q_squeeze_L)

        leg_targets = [
            (idx, target * g.walk_stance_scale, kp, kd) for idx, target, kp, kd in LEG_JOINTS
        ]
        # 2026-09-09 : 150/3.0 (valeurs qui marchent sur le VRAI robot,
        # pivot_real.py) ne produit quasiment AUCUNE rotation mesurable en
        # sim -- confirme via /hardware/joint_state (position[12] reste
        # <5deg de bruit alors que la commande vise 45-180deg). Teste avec
        # des gains bien plus forts pour voir si le controleur actif de la
        # sim resiste juste plus que le vrai robot (pas encore confirme).
        # 2026-09-10 : chute confirmee par telemetrie sim_state (LCM, pas
        # les logs ROS -- ceux-la rapportaient success=True partout) pile
        # au relachement du pivot(45deg) avec 500/10.0 + la rampe
        # RETRACT_PINCH_X ci-dessous (x=1.30->0.35, z=0.82->0.19 en 1s,
        # robot au sol ~13s avant reset auto du simulateur). Cause isolee
        # par contre-essai telemetrie : repasser a 150/3.0 tout du long
        # supprime la chute (z parfaitement stable, 0.820-0.821m sans
        # variation) mais reproduit alors l'AUTRE probleme deja documente
        # (quasiment pas de rotation mesurable, x ne bouge que de 6mm) --
        # donc le gain fort n'est pas fautif pendant le MAINTIEN (ca
        # tenait), le probleme est de le lacher d'un coup pile au moment
        # ou les bras relachent le carton (perte brutale de la charge
        # portee alors que le buste reste rigide) -- cf ramp-down plus bas
        # avant le relachement, gains forts gardes seulement pendant la
        # rotation/le maintien.
        lever.set_gains(WAIST_JOINT_INDEX, WAIST_KP_HOLD, WAIST_KD_HOLD)

        # 2026-09-10 (historique) : cette section ramenait le carton pres du corps
        # (RETRACT_PINCH_X) juste avant le pivot. Le commentaire d'origine disait deja
        # ce retrait "inutile ici puisque le bras est deja retracte avant meme la
        # levee -- supprime plus bas", mais le CODE n'avait en fait jamais ete
        # supprime -- retire pour de bon le 2026-09-14, sur demande explicite de
        # l'utilisateur ("je veux que pour tourner on garde la meme pose des bras") :
        # ce retrait bougeait activement les bras juste avant/pendant le debut du
        # pivot, visible comme "les avant-bras tournent" (en plus de l'entrainement
        # rigide normal du bras par la rotation du buste, attendu et inevitable).
        # q_squeeze_L/R restent maintenant EXACTEMENT la pose calculee plus haut
        # (position de serrage reellement tenue, a hauteur g.lift_z), inchangee tout
        # du long du pivot.
        angle_target = np.radians(g.angle_deg)

        self.get_logger().info(
            f"pivot buste -- 0 -> {g.angle_deg:.0f}deg ({g.pivot_duration:.1f}s, "
            "bras/jambes republies en continu pour ne pas lacher le carton)"
        )
        n = max(1, int(g.pivot_duration * 30))
        for i in range(n + 1):
            a = ease(i / n)
            self._publish_pose(lever, q_squeeze_L, q_squeeze_R, a * angle_target,
                                leg_targets, g.walk_stance_stiffness_scale)
            time.sleep(1.0 / 30)

        self.get_logger().info(f"maintien pivote -- {g.hold_seconds:.1f}s")
        elapsed = 0.0
        step = 0.1
        cancelled = False
        while elapsed < g.hold_seconds:
            if goal_handle.is_cancel_requested:
                cancelled = True
                break
            self._publish_pose(lever, q_squeeze_L, q_squeeze_R, angle_target,
                                leg_targets, g.walk_stance_stiffness_scale)
            time.sleep(step)
            elapsed += step

        do_depivot = g.depivot_before_release or cancelled
        if do_depivot:
            # 2026-09-10 : gains buste ramenes de WAIST_KP_HOLD/KD_HOLD (necessaires
            # pour une rotation reelle en sim) vers WAIST_KP_RELEASE/KD_RELEASE
            # (valeurs prouvees stables au relachement) PENDANT le depivot, pour
            # etre deja souple AVANT que les bras ne lachent le carton -- cf
            # chute confirmee par telemetrie quand le gain fort restait jusqu'au
            # relachement (choc : perte brutale de charge + buste encore rigide).
            self.get_logger().info(f"depivot -- {g.angle_deg:.0f}deg -> 0deg ({g.pivot_duration:.1f}s)")
            n = max(1, int(g.pivot_duration * 30))
            for i in range(n + 1):
                a = ease(i / n)
                kp = WAIST_KP_HOLD + a * (WAIST_KP_RELEASE - WAIST_KP_HOLD)
                kd = WAIST_KD_HOLD + a * (WAIST_KD_RELEASE - WAIST_KD_HOLD)
                lever.set_gains(WAIST_JOINT_INDEX, kp, kd)
                self._publish_pose(lever, q_squeeze_L, q_squeeze_R, (1.0 - a) * angle_target,
                                    leg_targets, g.walk_stance_stiffness_scale)
                time.sleep(1.0 / 30)
            final_waist = 0.0
        else:
            lever.set_gains(WAIST_JOINT_INDEX, WAIST_KP_RELEASE, WAIST_KD_RELEASE)
            final_waist = angle_target

        if cancelled or g.release_after:
            self.get_logger().info(f"relachement -- rampe {g.release_ramp_seconds:.1f}s")
            if g.release_ramp_seconds > 0:
                n = max(1, int(g.release_ramp_seconds * 30))
                for i in range(n + 1):
                    lever.set_weight(1.0 - i / n)
                    self._publish_pose(lever, q_squeeze_L, q_squeeze_R, final_waist,
                                        leg_targets, g.walk_stance_stiffness_scale)
                    time.sleep(1.0 / 30)
            lever.release()
        elif g.free_legs_for_walk:
            self.get_logger().info(
                "liberation des jambes (marche) -- bras/buste restent tenus a poids plein"
            )
            self._publish_pose(lever, q_squeeze_L, q_squeeze_R, final_waist,
                                leg_targets, g.walk_stance_stiffness_scale)
            lever.untouch(LEG_INDICES)
        else:
            self.get_logger().info(
                "release_after=false -- pas de relachement, bras/jambes restent tenus "
                "(un autre node est cense suivre)."
            )

        if cancelled:
            goal_handle.canceled()
            result.success = False
            return result

        goal_handle.succeed()
        result.success = True
        return result


def main():
    rclpy.init()
    node = PivotActionServer()
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
