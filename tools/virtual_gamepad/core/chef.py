import sys
import time

import lcm

sys.path.insert(0, "/home/equansrobotic/engineai_robotics_native_sdk/tools/virtual_gamepad")
from lcm_msgs.data import GamepadKeys

sys.path.insert(0, "/home/equansrobotic/stagiaire_1/test_passé")
from walk_to_xy import SimStateListener  # noqa: E402

LCM_URL = "udpm://239.255.76.67:7667?ttl=1"
CHANNEL = "virtual_gamepad/gamepad_keys"
STANDING_HEIGHT_THRESHOLD = 0.75

BUTTON_INDEX = {
    "LB": 0, "RB": 1, "A": 2, "B": 3, "X": 4, "Y": 5,
    "BACK": 6, "START": 7, "UP": 8, "DOWN": 9, "LEFT": 10, "RIGHT": 11,
}
ANALOG_INDEX = {
    "LEFT_STICK_X": 2, "LEFT_STICK_Y": 3, "RIGHT_STICK_X": 4, "RIGHT_STICK_Y": 5,
}

class Conductor:
    def __init__(self, rate_hz: float = 20.0, lcm_url: str = LCM_URL):
        self._lcm = lcm.LCM(lcm_url)
        self._period = 1.0 / rate_hz
        self._state = GamepadKeys()
        self._pose_listener = SimStateListener()

    def _send_button_combo(self, button_names, hold_seconds=0.5) -> None:
        for name in button_names:
            self._state.digital_states[BUTTON_INDEX[name]] = 1

        n_ticks = max(1, round(hold_seconds / self._period))
        for _ in range(n_ticks):
            self._state.timestamp = int(time.time() * 1_000_000)
            self._lcm.publish(CHANNEL, self._state.encode())
            time.sleep(self._period)

        for name in button_names:
            self._state.digital_states[BUTTON_INDEX[name]] = 0

    def _start_in_stand(self, settle_seconds=10.0) -> None:
        print("[chef] Passage en pd_stand...", flush=True)
        self._send_button_combo(["LB", "A"])

        t0 = time.time()
        height = None
        while time.time() - t0 < settle_seconds:
            height = self._pose_listener.height()
            if height is not None and height > STANDING_HEIGHT_THRESHOLD:
                print(f"[chef] pd_stand stabilise (z={height:.3f}m).", flush=True)
                return
            time.sleep(0.1)

        raise RuntimeError(
            f"pd_stand n'a pas atteint z>{STANDING_HEIGHT_THRESHOLD}m apres "
            f"{settle_seconds}s (derniere hauteur mesuree : {height}) -- le "
            "robot est probablement reste effondre. Ne pas enchainer sur "
            "walk_to() sur un robot qui n'est pas debout."
        )

    def walk_to(self, x, y, duration=2.0, forward=0.5) -> None:
        print(f"[chef] walk_to({x}, {y}) -- pas encore utilise, poussee fixe forward={forward} pendant {duration}s...", flush=True)
        self._send_button_combo(["LB", "B"])

        self._state.analog_states[ANALOG_INDEX["LEFT_STICK_X"]] = forward
        n_ticks = max(1, round(duration / self._period))
        for _ in range(n_ticks):
            self._state.timestamp = int(time.time() * 1_000_000)
            self._lcm.publish(CHANNEL, self._state.encode())
            time.sleep(self._period)
        self._state.analog_states[ANALOG_INDEX["LEFT_STICK_X"]] = 0.0

        print("[chef] walk_to termine.", flush=True)

    def run_sequence(self, target_x, target_y, duration=2.0) -> None:
        self._start_in_stand()
        self.walk_to(target_x, target_y, duration=duration)