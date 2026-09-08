import signal
import sys
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

sys.path.insert(0, "/home/equansrobotic/stagiaire_1/tools/joint_angle_commander")
from lever import Lever  # noqa: E402

sys.path.insert(0, "/home/equansrobotic/stagiaire_1/tools/robot_arm_ik")
from lift_carton import (  # noqa: E402
    LEFT_CHAIN, RIGHT_CHAIN, HAND_OFFSET_LEFT, HAND_OFFSET_RIGHT, solve_ik, ease,
)

from virtual_gamepad_interfaces.action import Lift  # noqa: E402

LEFT_JOINT_INDICES = [13, 14, 15, 16, 17]
RIGHT_JOINT_INDICES = [18, 19, 20, 21, 22]

Q_LEFT_HOME = np.array([0.000879, 0.075284, -0.000233, -0.126397, -0.000033])
Q_RIGHT_HOME = np.array([0.000885, -0.075161, 0.000241, -0.126390, 0.000033])

# Posture jambes flechies (portee de lift_carton_real.py, non repris ici avant
# le 20/08) : indices + angles "walk au repos" mesures via sim_state le
# 2026-08-11 (voir lift_carton_real.py pour le detail complet). Coordonner
# les 3 (hanche+genou+cheville) est essentiel -- le genou SEUL, mauvais
# signe, a fait tomber le robot immediatement lors d'un essai anterieur.
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

# 2026-09-02 : bug trouve par analogie avec LEVEE_STIFFNESS de levee.py (script
# robot reel) -- ce node n'a JAMAIS reduit la rigidite des bras pendant la
# levee, contrairement aux jambes (_bend_knees ci-dessous). Les bras restent
# donc a lever.py::DEFAULT_STIFFNESS=250 (indices 13-22) meme pendant la
# SEULE phase a charge reelle soutenue (levee/maintien) -- meme categorie de
# probleme que celui trouve et corrige sur le robot reel le 31/08 (25.0 au
# lieu du 90.0 documente). Meme protocole JointOverrideCommand natif des 2
# cotes (pas juste une analogie de nom), donc meme valeur reutilisee comme
# point de depart. A ajuster si le robot reste instable/trop mou pendant la
# levee malgre ce fix -- jamais teste avant cette session.
LEVEE_ARM_STIFFNESS = 90.0


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
        self._server = ActionServer(
            self, Lift, "lift", self._execute, cancel_callback=self._on_cancel,
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

        # only_phase (2026-09-02, meme principe que --only-phase sur levee.py) :
        # meme protection contre "il desserre avant la levee" (cf. Lift.action) --
        # relachement seulement si only_phase in ("", "levee").
        run_approche = g.only_phase in ("", "approche")
        run_serrage = g.only_phase in ("", "serrage")
        # forget() SEULEMENT si cet appel inclut l'approche (2026-09-02) : c'est
        # toujours le tout DEBUT d'une nouvelle sequence de saisie -- nettoie
        # l'etat _touched/_stiffness residuel d'un appel precedent SANS RAPPORT
        # (root cause de "il ne leve pas les bras"/"ne flechit pas les genoux",
        # cf. memoire projet -- ce node vit longtemps, plusieurs sequences
        # independantes s'enchainent dessus). Ne PAS le faire pour
        # serrage/levee seuls : ces appels DEPENDENT expres de l'etat laisse
        # par l'appel approche precedent DANS LA MEME sequence.
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

        pinch_L = _rotate_xy([g.pinch_x, g.pinch_y, g.pinch_z], g.pinch_yaw_offset)
        pinch_R = _rotate_xy([g.pinch_x, -g.pinch_y, g.pinch_z], g.pinch_yaw_offset)
        q_pinch_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, pinch_L, Q_LEFT_HOME)
        q_pinch_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT, pinch_R, Q_RIGHT_HOME)

        squeeze_L = _rotate_xy([g.pinch_x, g.squeeze_y, g.pinch_z], g.pinch_yaw_offset)
        squeeze_R = _rotate_xy([g.pinch_x, -g.squeeze_y, g.pinch_z], g.pinch_yaw_offset)
        q_squeeze_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_pinch_L)
        q_squeeze_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT, squeeze_R, q_pinch_R)

        if run_approche:
            self.get_logger().info(
                f"approche -- pinch_x={g.pinch_x:.3f} pinch_y=+-{g.pinch_y:.3f} "
                f"pinch_z={g.pinch_z:.3f} ({g.approach_duration:.1f}s)"
            )
            self._move_arms(lever, Q_LEFT_HOME, q_pinch_L, Q_RIGHT_HOME, q_pinch_R, g.approach_duration)
            self.get_logger().info("pause 2.0s (approche terminee)")
            self._hold(lever, q_pinch_L, q_pinch_R, goal_handle, 2.0)

        if run_serrage:
            self.get_logger().info(
                f"serrage -- Y +-{g.pinch_y:.3f} -> +-{g.squeeze_y:.3f} "
                f"({g.squeeze_duration:.1f}s) -- LE CONTACT AVEC LE CARTON COMMENCE ICI"
            )
            self._move_arms(lever, q_pinch_L, q_squeeze_L, q_pinch_R, q_squeeze_R, g.squeeze_duration)
            self.get_logger().info("pause 2.0s (serrage termine)")
            self._hold(lever, q_squeeze_L, q_squeeze_R, goal_handle, 2.0)

        if not run_levee:
            self.get_logger().info(
                f"only_phase={g.only_phase!r} termine -- bras tenus a leur derniere position, "
                "pas de levee/relachement dans cet appel."
            )
            goal_handle.succeed()
            result.success = True
            return result

        if g.only_phase == "levee":
            # Suppose approche+serrage DEJA effectues par un appel precedent (meme
            # convention que levee.py --only-phase levee, robot reel) -- publie
            # directement la position de serrage (recalculee via solve_ik ci-dessus,
            # pas un etat persistant) au lieu de la rejouer.
            self.get_logger().info(
                f"only_phase=levee : publication directe de la position de serrage "
                f"(pinch_x={g.pinch_x:.3f} pinch_y=+-{g.squeeze_y:.3f} pinch_z={g.pinch_z:.3f}), "
                "suppose deja atteinte par un appel precedent."
            )
            for idx, angle in zip(LEFT_JOINT_INDICES, q_squeeze_L):
                lever[idx] = float(angle)
            for idx, angle in zip(RIGHT_JOINT_INDICES, q_squeeze_R):
                lever[idx] = float(angle)
            self._hold(lever, q_squeeze_L, q_squeeze_R, goal_handle, 1.0)

        self.get_logger().info(
            f"levee -- Z {g.pinch_z:.3f} -> {g.lift_z:.3f} ({g.lift_duration:.1f}s), "
            f"rigidite bras reduite a {LEVEE_ARM_STIFFNESS:.0f} (etait {250.0:.0f})"
        )
        for idx in LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES:
            lever.set_gains(idx, stiffness=LEVEE_ARM_STIFFNESS)
        qL, qR = q_squeeze_L.copy(), q_squeeze_R.copy()
        n = max(1, int(g.lift_duration * 30))
        for i in range(n + 1):
            a = ease(i / n)
            z = g.pinch_z + a * (g.lift_z - g.pinch_z)
            qL = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                           _rotate_xy([g.pinch_x, g.squeeze_y, z], g.pinch_yaw_offset), qL, iters=30)
            qR = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT,
                           _rotate_xy([g.pinch_x, -g.squeeze_y, z], g.pinch_yaw_offset), qR, iters=30)
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
            # DESCENTE AJOUTEE ICI (2026-09-02, demande utilisateur "et le
            # reposer") : symetrique inverse de la montee ci-dessus (Z lift_z
            # -> pinch_z), pour reposer le carton a sa hauteur de prise
            # d'origine (podium) plutot que de le relacher/tirer en l'air.
            self.get_logger().info(
                f"pose -- Z {g.lift_z:.3f} -> {g.pinch_z:.3f} ({g.lift_duration:.1f}s)"
            )
            n = max(1, int(g.lift_duration * 30))
            for i in range(n + 1):
                a = ease(i / n)
                z = g.lift_z + a * (g.pinch_z - g.lift_z)
                qL = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT,
                               _rotate_xy([g.pinch_x, g.squeeze_y, z], g.pinch_yaw_offset), qL, iters=30)
                qR = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT,
                               _rotate_xy([g.pinch_x, -g.squeeze_y, z], g.pinch_yaw_offset), qR, iters=30)
                for idx, angle in zip(LEFT_JOINT_INDICES, qL):
                    lever[idx] = float(angle)
                for idx, angle in zip(RIGHT_JOINT_INDICES, qR):
                    lever[idx] = float(angle)
                time.sleep(1.0 / 30)
            # rigidite bras restauree a 250 (defaut) une fois repose -- plus de
            # charge soutenue a partir d'ici (LEVEE_ARM_STIFFNESS=90 n'etait
            # necessaire QUE pendant la levee/maintien, cf. plus haut).
            for idx in LEFT_JOINT_INDICES + RIGHT_JOINT_INDICES:
                lever.set_gains(idx, stiffness=250.0)

            # RETRAIT (2026-09-02) : root cause de "il pete un cable et ejecte le
            # carton" (retour utilisateur direct, confirme par trace instrumentee
            # -- vitesse/couple bras QUASI NULS pendant toute l'approche/serrage/
            # levee/maintien, puis pic brutal (12.5 rad/s, 20.9 N.m, choc visible
            # sur base_link z 0.82->0.79) EXACTEMENT au moment ou _release()
            # rendait la main -- confirme aussi en isolant SEULEMENT la levee via
            # only_phase=levee, meme chute). _release() rampe le POIDS de
            # l'override (1.0->0.0) mais republie la MEME position tenue (bras
            # loin de leur pose native, jambes flechies) jusqu'a la toute fin --
            # au moment ou le poids atteint reellement 0, le controleur natif
            # (pd_stand) reprend d'un coup avec sa PROPRE cible (bras/jambes
            # ~Q_HOME/droites), tres loin de la position tenue -- saut de
            # position brutal, pas un blend progressif malgre la rampe de poids.
            # Fix : ramener bras (interpolation articulaire directe vers Q_HOME,
            # ouvre le serrage au passage) ET jambes (_straighten_knees, inverse
            # de _bend_knees) PRES de leur cible native AVANT de rendre le
            # controle -- une fois la-bas, meme un release() a poids nul ne
            # produit plus de saut.
            # NOTE 2026-09-02 (2e passage) : le redressement des jambes est
            # desormais INCONDITIONNEL des que release_after=True (plus gate par
            # g.walk_stance) -- un appel only_phase="levee" enchaine peut avoir
            # walk_stance=False (jambes DEJA flechies par un appel precedent,
            # pas besoin de re-flechir en DEBUT de CET appel) tout en ayant
            # quand meme besoin de redresser en FIN d'appel avant de relacher --
            # les 2 evenements (bend au debut, straighten a la fin) ne doivent
            # plus partager le meme flag.
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
    # MultiThreadedExecutor : meme raison que walk_to.py/stand.py -- l'execution
    # d'un goal (boucle bloquante ci-dessus) tourne dans son propre thread
    # pendant qu'un autre thread reste libre pour traiter les cancel entrants.
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
