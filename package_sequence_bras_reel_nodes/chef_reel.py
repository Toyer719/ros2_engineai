"""Orchestrateur pour le robot reel -- architecture multi-nodes (comme la
simulation) : envoie des buts ROS2 Action a lift_node.py/pivot_node.py/
depose_node.py, ne bouge jamais rien lui-meme. Reprend exactement la
sequence deja validee de package_sequence_bras_reel/levee_pivot.py
(approche -> serrage -> levee -> pivot -> depose), mais via 3 process
separes au lieu d'un seul."""
import argparse
import os
import sys
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

for _candidate in (
    "/home/equansrobotic/stagiaire_1/package_sequence_bras_reel",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
):
    if os.path.isfile(os.path.join(_candidate, "levee.py")):
        sys.path.insert(0, _candidate)
        break
from levee import PINCH_X

from virtual_gamepad_interfaces.action import Lift, Pivot, Depose, Stand, WalkTo

SERVER_TIMEOUT_S = 10.0
GOAL_TIMEOUT_S = 120.0

# 2026-09-17 : la scene MuJoCo (pm01_edu_carton) place le carton a la
# distance atteinte APRES la marche de chef_node.py (WALK_FORWARD_MPS/
# WALK_DURATION ci-dessous, memes valeurs que chef_node.py -- seul point
# ou le pont vitesse->stick de body_vel_bridge.py est calibre fidelement),
# pas a la distance reelle-robot (PINCH_X de levee.py, carton pose
# directement devant un robot immobile). Sans cette marche, --pinch-x par
# defaut (0.216, reel) vise un carton bien plus loin que prevu en sim --
# constate : chute en fin de sequence malgre success=True partout.
WALK_FORWARD_MPS = 0.45
WALK_DURATION = 2.3


class ChefReelNode(Node):
    def __init__(self):
        super().__init__("chef_reel")
        self._stand_client = ActionClient(self, Stand, "stand")
        self._walk_client = ActionClient(self, WalkTo, "walk_to")
        self._lift_client = ActionClient(self, Lift, "lift")
        self._pivot_client = ActionClient(self, Pivot, "pivot")
        self._depose_client = ActionClient(self, Depose, "depose")

    def _send_goal(self, client, name, goal):
        if not client.wait_for_server(timeout_sec=SERVER_TIMEOUT_S):
            raise RuntimeError(f"Action server '{name}' indisponible.")

        goal_done = threading.Event()
        outcome = {}

        def on_result(result_future):
            outcome["result"] = result_future.result().result
            goal_done.set()

        def on_goal_response(goal_future):
            goal_handle = goal_future.result()
            if not goal_handle.accepted:
                outcome["error"] = RuntimeError(f"Goal {name} refuse.")
                goal_done.set()
                return
            goal_handle.get_result_async().add_done_callback(on_result)

        client.send_goal_async(goal).add_done_callback(on_goal_response)
        if not goal_done.wait(timeout=GOAL_TIMEOUT_S):
            raise RuntimeError(f"Goal {name} sans resultat apres {GOAL_TIMEOUT_S}s.")
        if "error" in outcome:
            raise outcome["error"]

        success = outcome["result"].success
        self.get_logger().info(f"{name} termine : success={success}")
        return success

    def stand(self, settle_seconds: float = 10.0):
        return self._send_goal(self._stand_client, "stand",
                                Stand.Goal(settle_seconds=float(settle_seconds)))

    def walk_to(self, forward: float, turn: float = 0.0, duration: float = 1.0):
        goal = WalkTo.Goal(forward=float(forward), turn=float(turn), duration=float(duration))
        return self._send_goal(self._walk_client, "walk_to", goal)

    def lift(self, **kwargs):
        return self._send_goal(self._lift_client, "lift", Lift.Goal(**kwargs))

    def pivot(self, **kwargs):
        return self._send_goal(self._pivot_client, "pivot", Pivot.Goal(**kwargs))

    def depose(self, **kwargs):
        return self._send_goal(self._depose_client, "depose", Depose.Goal(**kwargs))

    def run_sequence(self, args):
        pinch_x = args.pinch_x
        pinch_z = args.pinch_z

        if args.stand:
            self.get_logger().info("--- stand ---")
            if not self.stand(settle_seconds=5.0):
                self.get_logger().error("sequence : stand a echoue -- arret.")
                return False

        if args.walk:
            self.get_logger().info("--- marche ---")
            if not self.walk_to(forward=WALK_FORWARD_MPS, duration=WALK_DURATION):
                self.get_logger().error("sequence : walk_to a echoue -- arret.")
                return False
            time.sleep(2.0)
            if not self.stand(settle_seconds=3.0):
                self.get_logger().error("sequence : stand (post-marche) a echoue -- arret.")
                return False

        self.get_logger().info("--- approche ---")
        if not self.lift(pinch_x=pinch_x, pinch_z=pinch_z, pinch_yaw_offset=args.pinch_yaw_offset,
                          waypoint_duration=2.1, approach_duration=1.5,
                          only_phase="approche", release_after=False):
            self.get_logger().error("sequence : lift(approche) a echoue -- arret.")
            return False

        self.get_logger().info("--- serrage ---")
        if not self.lift(pinch_x=pinch_x, pinch_z=pinch_z, pinch_yaw_offset=args.pinch_yaw_offset,
                          squeeze_duration=2.1, only_phase="serrage", release_after=False):
            self.get_logger().error("sequence : lift(serrage) a echoue -- arret.")
            return False

        self.get_logger().info("--- levee ---")
        if not self.lift(pinch_x=pinch_x, pinch_z=pinch_z, pinch_yaw_offset=args.pinch_yaw_offset,
                          lift_z=args.lift_z, lift_duration=2.0, hold_seconds=0.5,
                          only_phase="levee", release_after=False):
            self.get_logger().error("sequence : lift(levee) a echoue -- arret.")
            return False

        self.get_logger().info("--- pivot ---")
        if not self.pivot(pinch_x=pinch_x, pinch_z=pinch_z, pinch_yaw_offset=args.pinch_yaw_offset,
                           lift_z=args.lift_z, angle_deg=args.angle_deg,
                           pivot_duration=args.pivot_duration, hold_seconds=0.9,
                           depivot_before_release=False, release_after=False,
                           free_legs_for_walk=False):
            self.get_logger().error("sequence : pivot a echoue -- arret.")
            return False

        self.get_logger().info("--- depose ---")
        # pinch_x = celui D'ORIGINE (approche), PAS depose_x -- depose_node.py
        # en a besoin pour reconstruire la MEME ancre null-space (q_squeeze_L)
        # que pivot_node.py a gardee fixe tout du long (verifie numeriquement,
        # voir la docstring de depose_node.py). Le X reellement tenu (DEPOSE_X)
        # est une constante interne a depose_node.py, pas transmise ici.
        if not self.depose(pinch_x=pinch_x, pinch_z=pinch_z, pinch_yaw_offset=args.pinch_yaw_offset,
                            hold_z=args.lift_z, drop_z=args.lift_z - 0.03,
                            depivot_from_deg=args.angle_deg, depivot_duration=args.pivot_duration,
                            tendre_duration=1.3, depose_duration=0.9,
                            degagement_waypoint_duration=1.3):
            self.get_logger().error("sequence : depose a echoue -- arret.")
            return False

        self.get_logger().info("sequence terminee.")
        return True


def _build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinch-x", type=float, default=PINCH_X)
    parser.add_argument("--pinch-z", type=float, default=0.05)
    parser.add_argument("--pinch-yaw-offset", type=float, default=0.0)
    parser.add_argument("--lift-z", type=float, default=0.15)
    parser.add_argument("--angle-deg", type=float, default=90.0)
    parser.add_argument("--pivot-duration", type=float, default=2.5)
    parser.add_argument("--walk", action="store_true",
                         help="marche avant l'approche (necessaire en simulation -- la "
                              "scene place le carton a distance de marche, PAS a la "
                              "distance reelle-robot ; jamais utilise sur le vrai robot)")
    parser.add_argument("--stand", action="store_true",
                         help="appelle l'action 'stand' avant la sequence (necessaire en "
                              "simulation, ou le robot demarre couche dans MuJoCo ; sur le "
                              "vrai robot il est deja debout a l'allumage et cette action "
                              "n'existe meme pas -- NE JAMAIS l'utiliser sur le vrai robot)")
    return parser


def main():
    args = _build_arg_parser().parse_args()
    rclpy.init()
    node = ChefReelNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run_sequence(args)
    except KeyboardInterrupt:
        pass
    finally:
        # 2026-09-17 : ordre important -- rclpy.shutdown() D'ABORD (fait
        # sortir rclpy.spin() dans spin_thread), PUIS join (attend que le
        # thread ait vraiment fini), PUIS destroy_node() -- detruire le
        # node pendant qu'un autre thread le spinne encore a fait planter
        # le process a l'arret (std::terminate) lors du premier test.
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()


if __name__ == "__main__":
    main()
