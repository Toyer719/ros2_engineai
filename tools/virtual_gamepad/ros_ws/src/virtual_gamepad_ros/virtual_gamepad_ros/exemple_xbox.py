#!/usr/bin/env python3
"""
Real Xbox controller (evdev) -> /hardware/gamepad_keys bridge.
Same message shape as the manual `ros2 topic pub` gamepad_keys test, but
sourced from a real physical Xbox controller instead of hand-crafted values.
"""
import argparse
import select
import time

import evdev
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from interface_protocol.msg import GamepadKeys

PUBLISH_HZ = 50.0

# GamepadKeys.msg digital_states indices
LB, RB, A, B, X, Y, BACK, START = 0, 1, 2, 3, 4, 5, 6, 7
CROSS_X_UP, CROSS_X_DOWN, CROSS_Y_LEFT, CROSS_Y_RIGHT = 8, 9, 10, 11

# GamepadKeys.msg analog_states indices
LT, RT, LEFT_STICK_X, LEFT_STICK_Y, RIGHT_STICK_X, RIGHT_STICK_Y = 0, 1, 2, 3, 4, 5

BUTTON_MAP = {
    evdev.ecodes.BTN_TL: LB,
    evdev.ecodes.BTN_TR: RB,
    evdev.ecodes.BTN_SOUTH: A,
    evdev.ecodes.BTN_EAST: B,
    evdev.ecodes.BTN_WEST: X,
    evdev.ecodes.BTN_NORTH: Y,
    evdev.ecodes.BTN_SELECT: BACK,
    evdev.ecodes.BTN_START: START,
}

AXIS_MAP = {
    evdev.ecodes.ABS_X: LEFT_STICK_X,
    evdev.ecodes.ABS_Y: LEFT_STICK_Y,
    evdev.ecodes.ABS_RX: RIGHT_STICK_X,
    evdev.ecodes.ABS_RY: RIGHT_STICK_Y,
    evdev.ecodes.ABS_Z: LT,
    evdev.ecodes.ABS_RZ: RT,
}


def find_device(name_substring: str):
    for path in evdev.list_devices():
        dev = evdev.InputDevice(path)
        if name_substring.lower() in dev.name.lower():
            return dev
    raise RuntimeError(
        f"No evdev device matching '{name_substring}'. "
        f"Available: {[evdev.InputDevice(p).name for p in evdev.list_devices()]}"
    )


STICK_DEADZONE = 0.05  # residual noise on top of the calibrated center


class AxisNorm:
    """Normalizes to [-1, 1] around a *measured* rest position, not the
    theoretical (min+max)/2 — worn sticks rest off-center. A one-shot
    calibration at startup isn't enough though: worn sticks also have
    mechanical hysteresis, so they don't spring back to the exact same raw
    value after being pushed and released. Fix: re-anchor the center every
    time the axis transitions from "outside the deadzone" (being actively
    pushed) back to "inside it" (released) — i.e. recalibrate on release,
    not just once at process start."""

    def __init__(self, absinfo, invert=False, deadzone=0.0, center=None, stick=False):
        self.min = absinfo.min
        self.max = absinfo.max
        self.center = center if center is not None else (absinfo.min + absinfo.max) / 2.0
        self.invert = invert
        self.deadzone = deadzone
        self.stick = stick
        self._active = False

    def __call__(self, raw):
        if raw >= self.center:
            span = max(1, self.max - self.center)
            v = (raw - self.center) / span
        else:
            span = max(1, self.center - self.min)
            v = (raw - self.center) / span

        if self.stick:
            if abs(v) < self.deadzone:
                if self._active:
                    self.center = raw  # just released -> re-anchor to wherever it actually settled
                    self._active = False
                v = 0.0
            else:
                self._active = True

        return -v if self.invert else v


STICK_AXES = (evdev.ecodes.ABS_X, evdev.ecodes.ABS_Y, evdev.ecodes.ABS_RX, evdev.ecodes.ABS_RY)


class GamepadBridge(Node):
    def __init__(self, name_substring: str):
        super().__init__("xbox_gamepad_bridge")
        self.dev = find_device(name_substring)
        self.get_logger().info(f"Using device: {self.dev.path} ({self.dev.name})")
        self.get_logger().info("Calibrating stick centers — leave the sticks untouched...")
        time.sleep(0.5)  # let any in-flight motion from handling the controller settle

        caps = dict(self.dev.capabilities(absinfo=True).get(evdev.ecodes.EV_ABS, []))
        self.axis_norm = {}
        for code, info in caps.items():
            invert = code in (evdev.ecodes.ABS_Y, evdev.ecodes.ABS_RY)
            if code in STICK_AXES:
                # re-read live (not the cached absinfo) to get the actual current rest value
                center = self.dev.absinfo(code).value
                self.axis_norm[code] = AxisNorm(
                    info, invert=invert, deadzone=STICK_DEADZONE, center=center, stick=True
                )
            else:
                self.axis_norm[code] = AxisNorm(info, invert=invert, deadzone=0.0, stick=False)

        for code in STICK_AXES:
            if code in self.axis_norm:
                name = evdev.ecodes.ABS[code] if code in evdev.ecodes.ABS else code
                self.get_logger().info(f"  {name}: calibrated center = {self.axis_norm[code].center}")

        self.digital_states = [0] * 12
        self.analog_states = [0.0] * 6
        self.hat_x = 0
        self.hat_y = 0

        self.pub = self.create_publisher(GamepadKeys, "/hardware/gamepad_keys", 1)
        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self.publish_state)

    def read_events_nonblocking(self):
        r, _, _ = select.select([self.dev.fd], [], [], 0.0)
        if not r:
            return
        try:
            for event in self.dev.read():
                self.handle_event(event)
        except (OSError, BlockingIOError):
            pass

    def handle_event(self, event):
        if event.type == evdev.ecodes.EV_KEY and event.code in BUTTON_MAP:
            self.digital_states[BUTTON_MAP[event.code]] = 1 if event.value else 0
        elif event.type == evdev.ecodes.EV_ABS:
            if event.code == evdev.ecodes.ABS_HAT0X:
                self.hat_x = event.value
            elif event.code == evdev.ecodes.ABS_HAT0Y:
                self.hat_y = event.value
            elif event.code in AXIS_MAP and event.code in self.axis_norm:
                self.analog_states[AXIS_MAP[event.code]] = self.axis_norm[event.code](event.value)

        self.digital_states[CROSS_X_UP] = 1 if self.hat_y < 0 else 0
        self.digital_states[CROSS_X_DOWN] = 1 if self.hat_y > 0 else 0
        self.digital_states[CROSS_Y_LEFT] = 1 if self.hat_x < 0 else 0
        self.digital_states[CROSS_Y_RIGHT] = 1 if self.hat_x > 0 else 0

    def publish_state(self):
        self.read_events_nonblocking()
        msg = GamepadKeys()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.hardware_connected = True
        msg.digital_states = self.digital_states
        msg.analog_states = self.analog_states
        self.pub.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="xbox", help="Case-insensitive device name substring.")
    args = parser.parse_args()

    rclpy.init()
    node = GamepadBridge(args.name)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
user@user-UP-ADLN01:~/source/engineai_workspace/src/interface_example/scripts$ 

