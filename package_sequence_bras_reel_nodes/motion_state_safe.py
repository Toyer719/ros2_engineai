"""Version thread-safe de package_sequence_bras_reel/motion_state.py, pour
les Action Servers de ce dossier qui tournent sous un MultiThreadedExecutor.

La version originale (motion_state.py) appelle rclpy.spin_once(node, ...) en
interne -- correct pour un script standalone (levee_pivot.py, un seul
thread), mais DANGEREUX ici : chaque _execute() de nos Action Servers
tourne deja dans un thread gere par l'executor (voir walk_to.py en
simulation, qui documente exactement ce meme risque et l'evite de la meme
facon) -- spin_once() en plus, depuis ce thread, creerait une ressource
concurrente sur le node. Cette version utilise une subscription persistante
(creee UNE FOIS dans __init__ du node appelant) dont le callback est
appele par l'executor lui-meme -- l'attente ici se fait par time.sleep()
pur, jamais spin_once()."""
import time


class MotionStateWaiter:
    def __init__(self, node):
        from interface_protocol.msg import MotionState, MotionStateRequest
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        self._state = {"current": "", "available": []}
        state_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE)
        request_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.VOLATILE)
        self._sub = node.create_subscription(MotionState, "/motion/motion_state",
                                              self._on_state, state_qos)
        self._pub = node.create_publisher(MotionStateRequest, "/motion/set_motion_state",
                                           request_qos)
        self._MotionStateRequest = MotionStateRequest

    def _on_state(self, msg):
        self._state["current"] = msg.current_motion_task
        self._state["available"] = list(msg.available_transition_motions)

    def _wait_for_state(self, timeout=1.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._state["current"]:
                return
            time.sleep(0.05)

    def _switch_to(self, name, timeout):
        self._wait_for_state()
        if self._state["current"] == name:
            return True
        if name not in self._state["available"]:
            return False
        msg = self._MotionStateRequest()
        msg.target_motion_name = name
        self._pub.publish(msg)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._state["current"] == name:
                return True
            time.sleep(0.05)
        return False

    def ensure(self, target, timeout=3.0, detour="pd_stand"):
        self._wait_for_state()
        if self._state["current"] == target:
            return True
        if target not in self._state["available"] and detour in self._state["available"]:
            if not self._switch_to(detour, timeout):
                return False
        return self._switch_to(target, timeout)
