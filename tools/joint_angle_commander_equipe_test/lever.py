import time

import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from interface_protocol.msg import JointOverrideCommand

NUM_JOINTS = 24
TOPIC = "/motion/joint_override_command"

DEFAULT_STIFFNESS = [float(v) for v in [200, 200, 380, 450, 400, 200] * 2 + [200] + [250] * 10 + [100]]
DEFAULT_DAMPING = [float(v) for v in [5, 5, 5, 5, 2, 2] * 2 + [1] + [1] * 10 + [1]]


class Lever:

    def __init__(self, node, subscriber_timeout=5.0):
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._node = node
        self._pub = node.create_publisher(JointOverrideCommand, TOPIC, qos)
        self._position = [0.0] * NUM_JOINTS
        self._touched = [False] * NUM_JOINTS
        self._stiffness = list(DEFAULT_STIFFNESS)
        self._damping = list(DEFAULT_DAMPING)
        self._weight = 1.0
        self._wait_for_subscriber(subscriber_timeout)

    def set_weight(self, weight):
        self._weight = float(weight)
        if any(self._touched):
            self._publish()

    def set_gains(self, joint_index, stiffness=None, damping=None):
        if stiffness is not None:
            self._stiffness[joint_index] = float(stiffness)
        if damping is not None:
            self._damping[joint_index] = float(damping)
        if self._touched[joint_index]:
            self._publish()

    def _wait_for_subscriber(self, timeout):
        t0 = time.time()
        while self._pub.get_subscription_count() == 0:
            if time.time() - t0 > timeout:
                raise RuntimeError(f"Aucun abonne sur {TOPIC} apres {timeout}s.")
            time.sleep(0.05)

    def __setitem__(self, joint_index, angle_rad):
        self._position[joint_index] = angle_rad
        self._touched[joint_index] = True
        self._publish()

    def __getitem__(self, joint_index):
        return self._position[joint_index]

    def is_touched(self, joint_index):
        """True si cette articulation est actuellement sous override. Permet a un
        appelant enchainant plusieurs goals sur le MEME Lever (cf. chef_node.py, qui
        appelle lift() trois fois avec only_phase=approche/serrage/levee) de savoir si
        une pose a deja ete etablie par un appel precedent."""
        return self._touched[joint_index]

    def release(self):
        msg = JointOverrideCommand()
        msg.weight = 0.0
        self._pub.publish(msg)
        self._touched = [False] * NUM_JOINTS

    def forget(self):
        self._touched = [False] * NUM_JOINTS
        self._stiffness = list(DEFAULT_STIFFNESS)
        self._damping = list(DEFAULT_DAMPING)

    def untouch(self, indices):
        for i in indices:
            self._touched[i] = False
        self._publish()

    def _publish(self):
        indices = [i for i, touched in enumerate(self._touched) if touched]
        msg = JointOverrideCommand()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.weight = self._weight
        msg.joint_indices = indices
        msg.position = [self._position[i] for i in indices]
        msg.velocity = [0.0] * len(indices)
        msg.feed_forward_torque = [0.0] * len(indices)
        msg.torque = [0.0] * len(indices)
        msg.stiffness = [self._stiffness[i] for i in indices]
        msg.damping = [self._damping[i] for i in indices]
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = rclpy.create_node("lever_control")
    lever = Lever(node)

    lever[13] = 0.5
    lever[18] = 0.5

    time.sleep(3)
    lever.release()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
