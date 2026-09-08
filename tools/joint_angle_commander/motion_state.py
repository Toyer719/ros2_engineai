"""Helper partage pour basculer l'etat de mouvement du PM01 (reel) via
/motion/set_motion_state, bloquant jusqu'a confirmation via /motion/motion_state.

Factorise le 2026-08-28 depuis levee.py (ou cette logique s'appelait
_ensure_motion_state) -- desormais utilise AUSSI par marche.py (bascule
lower_body_balance -> rl_terrain avant de publier une vitesse, puis retour en
lower_body_balance avant d'enchainer la levee). rclpy.init() ne doit avoir lieu qu'UNE
SEULE fois par process -- ce module ne fait ni init ni shutdown, il suppose un `node` deja
cree par l'appelant (meme convention que lever.py::Lever)."""
import time


def ensure_motion_state(node, target, timeout=3.0, detour="pd_stand"):
    """Bascule vers `target` (bloquant), avec detour automatique par `detour` si l'etat
    courant n'a pas de transition directe vers `target` (ex: joint_bridge -> lower_body_balance
    passe forcement par pd_stand). Cree sa propre subscription MotionState et son propre
    publisher MotionStateRequest, les detruit avant de retourner -- reutilisable a
    l'identique par plusieurs appelants sur le meme node, sans etat residuel entre deux
    appels."""
    from interface_protocol.msg import MotionState, MotionStateRequest
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    import rclpy

    state = {"current": "", "available": []}

    def _cb(msg):
        state["current"] = msg.current_motion_task
        state["available"] = list(msg.available_transition_motions)

    state_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                            durability=DurabilityPolicy.VOLATILE)
    request_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
    sub = node.create_subscription(MotionState, "/motion/motion_state", _cb, state_qos)
    pub = node.create_publisher(MotionStateRequest, "/motion/set_motion_state", request_qos)

    def _wait_for_state(t=1.0):
        t0 = time.time()
        while time.time() - t0 < t:
            rclpy.spin_once(node, timeout_sec=0.05)
            if state["current"]:
                return

    def _switch_to(name):
        _wait_for_state()
        if state["current"] == name:
            return True
        if name not in state["available"]:
            return False
        msg = MotionStateRequest()
        msg.target_motion_name = name
        pub.publish(msg)
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(node, timeout_sec=0.05)
            if state["current"] == name:
                return True
        return False

    _wait_for_state()
    print(f"[ETAPE] etat de mouvement actuel : {state['current']}", flush=True)
    if state["current"] == target:
        node.destroy_subscription(sub)
        node.destroy_publisher(pub)
        return True

    if target not in state["available"] and detour in state["available"]:
        print(f"[ETAPE] {target} non atteignable directement -- detour par {detour}...", flush=True)
        if not _switch_to(detour):
            print(f"[ERREUR] echec du passage par {detour}.", flush=True)
            node.destroy_subscription(sub)
            node.destroy_publisher(pub)
            return False

    print(f"[ETAPE] bascule vers {target}...", flush=True)
    ok = _switch_to(target)
    node.destroy_subscription(sub)
    node.destroy_publisher(pub)
    if not ok:
        print(f"[ERREUR] echec du passage en {target} (etat actuel : {state['current']}).", flush=True)
    else:
        print(f"[ETAPE] mode {target} actif.", flush=True)
    return ok
