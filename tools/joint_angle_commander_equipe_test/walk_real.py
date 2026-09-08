import argparse
import struct
import time

import lcm

LCM_URL = "udpm://239.255.76.67:7667?ttl=1"
CHANNEL = "virtual_gamepad/gamepad_keys"

BUTTON_INDEX = {
    "LB": 0, "RB": 1, "A": 2, "B": 3, "X": 4, "Y": 5,
    "BACK": 6, "START": 7, "UP": 8, "DOWN": 9, "LEFT": 10, "RIGHT": 11,
}
ANALOG_INDEX = {
    "LEFT_STICK_X": 2, "LEFT_STICK_Y": 3, "RIGHT_STICK_X": 4, "RIGHT_STICK_Y": 5,
}

def _packed_fingerprint():
    # Reproduit exactement _get_hash_recursive([]) de GamepadKeys.py : rotate-left
    # de 1 bit sur 64 bits d'une constante fixe -- valeur unique, jamais changee
    # tant que la definition .lcm ne change pas.
    h = 0xd6cae60f8643a772 & 0xffffffffffffffff
    h = ((h << 1) & 0xffffffffffffffff) + (h >> 63)
    h &= 0xffffffffffffffff
    return struct.pack(">Q", h)


def _encode_gamepad_keys(digital_states, analog_states):
    buf = _packed_fingerprint()
    buf += struct.pack(">q", int(time.time() * 1_000_000))
    buf += struct.pack(">12i", *digital_states)
    buf += struct.pack(">6d", *analog_states)
    return buf


class GamepadState:
    def __init__(self):
        self.digital = [0] * 12
        self.analog = [0.0] * 6

    def set_button(self, name, value: bool):
        self.digital[BUTTON_INDEX[name]] = int(value)

    def set_analog(self, name, value: float):
        self.analog[ANALOG_INDEX[name]] = float(value)

    def encode(self):
        return _encode_gamepad_keys(self.digital, self.analog)


def marcher(lc, state, forward: float, turn: float, duration: float, rate_hz: float = 20.0):
    """Meme sequence que virtual_gamepad_ros/walk_to.py::_execute (LB+B puis
    sticks), rejouee ici en publiant directement sur LCM au lieu de passer par
    des topics ROS relayes par chef_node.py."""
    period = 1.0 / rate_hz

    print("[ETAPE] Passage en walk (LB+B)...", flush=True)
    state.set_button("LB", True)
    state.set_button("B", True)
    t0 = time.time()
    while time.time() - t0 < 0.5:
        lc.publish(CHANNEL, state.encode())
        time.sleep(period)
    state.set_button("LB", False)
    state.set_button("B", False)
    t0 = time.time()
    while time.time() - t0 < 1.5:
        lc.publish(CHANNEL, state.encode())
        time.sleep(period)
    print("[ETAPE] walk actif.", flush=True)

    print(f"[ETAPE] marche -- forward={forward} turn={turn} duree={duration}s", flush=True)
    state.set_analog("LEFT_STICK_X", forward)
    state.set_analog("RIGHT_STICK_Y", -turn)
    t0 = time.time()
    while time.time() - t0 < duration:
        lc.publish(CHANNEL, state.encode())
        time.sleep(period)

    state.set_analog("LEFT_STICK_X", 0.0)
    state.set_analog("RIGHT_STICK_Y", 0.0)
    for _ in range(5):
        lc.publish(CHANNEL, state.encode())
        time.sleep(period)
    print("[INFO] Marche terminee.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forward", type=float, required=True, help="Vitesse avant (meme echelle que le stick gauche, ex: 0.3 a 0.6)")
    parser.add_argument("--turn", type=float, default=0.0, help="Virage (meme echelle que le stick droit)")
    parser.add_argument("--duration", type=float, required=True, help="Duree de la marche en secondes")
    parser.add_argument("--rate-hz", type=float, default=20.0)
    args = parser.parse_args()

    lc = lcm.LCM(LCM_URL)
    state = GamepadState()
    marcher(lc, state, args.forward, args.turn, args.duration, args.rate_hz)


if __name__ == "__main__":
    main()
