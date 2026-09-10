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
    LEFT_CHAIN, RIGHT_CHAIN, HAND_OFFSET_LEFT, HAND_OFFSET_RIGHT, solve_ik, ease,
)

from virtual_gamepad_interfaces.action import Pivot

LEFT_JOINT_INDICES = [13, 14, 15, 16, 17]
RIGHT_JOINT_INDICES = [18, 19, 20, 21, 22]
WAIST_JOINT_INDEX = 12
WAIST_KP_HOLD, WAIST_KD_HOLD = 500.0, 10.0
WAIST_KP_RELEASE, WAIST_KD_RELEASE = 150.0, 3.0
LEG_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

Q_LEFT_HOME = np.array([0.000879, 0.075284, -0.000233, -0.126397, -0.000033])
Q_RIGHT_HOME = np.array([0.000885, -0.075161, 0.000241, -0.126390, 0.000033])

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

        pinch_L = _rotate_xy([g.pinch_x, g.pinch_y, g.pinch_z], g.pinch_yaw_offset)
        pinch_R = _rotate_xy([g.pinch_x, -g.pinch_y, g.pinch_z], g.pinch_yaw_offset)
        q_pinch_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, pinch_L, Q_LEFT_HOME)
        q_pinch_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT, pinch_R, Q_RIGHT_HOME)
        squeeze_L = _rotate_xy([g.pinch_x, g.squeeze_y, g.pinch_z], g.pinch_yaw_offset)
        squeeze_R = _rotate_xy([g.pinch_x, -g.squeeze_y, g.pinch_z], g.pinch_yaw_offset)
        q_squeeze_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_pinch_L)
        q_squeeze_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT, squeeze_R, q_pinch_R)

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

        # 2026-09-10 : cause racine trouvee pour la chute au moment de la
        # levee -- la rampe IK pinch_z->lift_z qui etait ici (a g.pinch_x
        # fixe) ne publiait JAMAIS ses poses intermediaires (ni
        # self._publish_pose ni sleep) : elle ne faisait que preparer une
        # graine IK. La levee reelle sur le robot etait donc un SAUT INSTANTANE
        # a lift_z au tout premier appel publie qui suivait (l'ancienne boucle
        # de retrait, executee apres la levee, avant le pivot) -- pas de rampe
        # du tout, choc sur l'equilibre pile a la levee. Corrige en suivant la
        # suggestion de l'utilisateur : saisir, RAPPROCHER le carton du corps
        # (pinch_x -> RETRACT_PINCH_X, a hauteur pinch_z inchangee, comme une
        # personne qui ramene une caisse contre elle), PUIS lever -- et les
        # deux phases sont maintenant reellement publiees en rampe (avant,
        # seule l'ancienne boucle de retrait, 1s, etait publiee ; la levee de
        # 3s ne l'etait pas). Le retrait avant le pivot (qui existait deja
        # depuis hier) devient inutile ici puisque le bras est deja retracte
        # avant meme la levee -- supprime plus bas.
        # Marge shoulder-roll a RETRACT_PINCH_X=0.20 verifiee (solve_ik) a
        # hauteur lift_z (~13deg) ; PAS reverifiee a hauteur pinch_z (plus
        # basse, marge attendue egale ou meilleure mais pas calculee).
        RETRACT_PINCH_X = 0.20
        n_retract = max(1, int(1.0 * 30))
        for i in range(n_retract + 1):
            a = ease(i / n_retract)
            x = g.pinch_x + a * (RETRACT_PINCH_X - g.pinch_x)
            q_squeeze_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                                    _rotate_xy([x, g.squeeze_y, g.pinch_z], g.pinch_yaw_offset),
                                    q_squeeze_L, iters=30)
            q_squeeze_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT,
                                    _rotate_xy([x, -g.squeeze_y, g.pinch_z], g.pinch_yaw_offset),
                                    q_squeeze_R, iters=30)
            self._publish_pose(lever, q_squeeze_L, q_squeeze_R, 0.0,
                                leg_targets, g.walk_stance_stiffness_scale)
            time.sleep(1.0 / 30)

        self.get_logger().info(
            f"levee -- {g.pinch_z:.3f}m -> {g.lift_z:.3f}m (carton deja rapproche, "
            f"pinch_x={RETRACT_PINCH_X:.2f}m)"
        )
        n_lift = max(1, int(3.0 * 30))
        for i in range(n_lift + 1):
            a = ease(i / n_lift)
            z = g.pinch_z + a * (g.lift_z - g.pinch_z)
            q_squeeze_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                                    _rotate_xy([RETRACT_PINCH_X, g.squeeze_y, z], g.pinch_yaw_offset),
                                    q_squeeze_L, iters=30)
            q_squeeze_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT,
                                    _rotate_xy([RETRACT_PINCH_X, -g.squeeze_y, z], g.pinch_yaw_offset),
                                    q_squeeze_R, iters=30)
            self._publish_pose(lever, q_squeeze_L, q_squeeze_R, 0.0,
                                leg_targets, g.walk_stance_stiffness_scale)
            time.sleep(1.0 / 30)

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
