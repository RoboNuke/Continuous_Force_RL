#!/usr/bin/env python3
"""
Standalone EE velocity teleop for Franka via pylibfranka.

Keyboard-driven end-effector velocity control. Twist commands are specified
in the EE frame and transformed to the world frame before sending.

Usage:
    python ee_velocity_teleop.py --ip 192.168.1.11
"""

import argparse
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import pylibfranka
from pylibfranka import (
    CartesianVelocities,
    ControllerMode,
    RealtimeConfig,
    Robot,
)

# ─── Tunable Parameters ─────────────────────────────────────────────────────

# Velocity magnitudes when keys are held (EE frame)
LINEAR_SPEED = 0.02        # m/s per axis
ANGULAR_SPEED = 0.2        # rad/s per axis

# Safety clamps (world frame, applied after transform)
MAX_LINEAR_VEL = 0.05      # m/s per axis
MAX_ANGULAR_VEL = 0.3      # rad/s per axis

# Acceleration limits (per 1kHz tick)
MAX_LINEAR_ACCEL_PER_TICK = 0.0005   # 0.5 m/s^2 at 1kHz
MAX_ANGULAR_ACCEL_PER_TICK = 0.003   # 3.0 rad/s^2 at 1kHz

# Key auto-release timeout (seconds). If no repeat keypress within this
# window, assume the key was released and zero the desired twist.
KEY_RELEASE_TIMEOUT = 0.15

# Impedance and collision settings
JOINT_IMPEDANCE = [3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0]
COLLISION_TORQUE = [100.0] * 7
COLLISION_FORCE = [100.0] * 6

# ─── Key Mapping ─────────────────────────────────────────────────────────────

# Maps a character → index into [vx, vy, vz, wx, wy, wz] and sign
KEY_MAP = {
    # Linear (EE frame)
    "w": (0, +1),   # +X  forward
    "s": (0, -1),   # -X  backward
    "a": (1, +1),   # +Y  left
    "d": (1, -1),   # -Y  right
    "q": (2, +1),   # +Z  up
    "e": (2, -1),   # -Z  down
    # Angular (EE frame)
    "i": (3, +1),   # +Rx pitch up
    "k": (3, -1),   # -Rx pitch down
    "j": (4, +1),   # +Ry yaw left
    "l": (4, -1),   # -Ry yaw right
    "u": (5, +1),   # +Rz roll CCW
    "o": (5, -1),   # -Rz roll CW
}

HELP_TEXT = """
╔══════════════════════════════════════════════════════╗
║           EE Velocity Teleop (pylibfranka)           ║
╠══════════════════════════════════════════════════════╣
║  Linear (EE frame):       Angular (EE frame):        ║
║    W/S  → ±X (fwd/back)    I/K  → ±Rx (pitch)       ║
║    A/D  → ±Y (left/right)  J/L  → ±Ry (yaw)         ║
║    Q/E  → ±Z (up/down)     U/O  → ±Rz (roll)        ║
║                                                      ║
║  SPACE → immediate zero    ESC  → quit               ║
╚══════════════════════════════════════════════════════╝
"""


def ee_twist_to_world(twist_ee: np.ndarray, O_T_EE: list) -> np.ndarray:
    """Transform a twist from EE frame to world frame.

    Args:
        twist_ee: [vx, vy, vz, wx, wy, wz] in EE frame.
        O_T_EE: 16-element column-major 4x4 homogeneous transform.

    Returns:
        [vx, vy, vz, wx, wy, wz] in world frame.
    """
    R = np.array([
        [O_T_EE[0], O_T_EE[4], O_T_EE[8]],
        [O_T_EE[1], O_T_EE[5], O_T_EE[9]],
        [O_T_EE[2], O_T_EE[6], O_T_EE[10]],
    ])
    twist_world = np.zeros(6)
    twist_world[:3] = R @ twist_ee[:3]
    twist_world[3:] = R @ twist_ee[3:]
    return twist_world


def clamp_twist(twist: np.ndarray) -> np.ndarray:
    """Clamp twist to safety limits (per-axis)."""
    twist[:3] = np.clip(twist[:3], -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
    twist[3:] = np.clip(twist[3:], -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
    return twist


def ramp_toward(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Ramp current twist toward target, respecting acceleration limits."""
    delta = target - current
    # Linear axes [0:3]
    delta[:3] = np.clip(delta[:3],
                        -MAX_LINEAR_ACCEL_PER_TICK,
                        MAX_LINEAR_ACCEL_PER_TICK)
    # Angular axes [3:6]
    delta[3:] = np.clip(delta[3:],
                        -MAX_ANGULAR_ACCEL_PER_TICK,
                        MAX_ANGULAR_ACCEL_PER_TICK)
    return current + delta


class KeyboardReader:
    """Non-blocking keyboard reader using raw terminal mode.

    Runs in a dedicated thread. Tracks held keys via auto-release timeout.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._desired_twist_ee = np.zeros(6)
        self._stop_event = threading.Event()
        self._quit_event = threading.Event()
        self._old_settings = None

    @property
    def should_quit(self) -> bool:
        return self._quit_event.is_set()

    def get_desired_twist(self) -> np.ndarray:
        with self._lock:
            return self._desired_twist_ee.copy()

    def start(self):
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        thread = threading.Thread(target=self._read_loop, daemon=True)
        thread.start()

    def stop(self):
        self._stop_event.set()
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    def _read_loop(self):
        # Track per-axis last-active time for auto-release
        last_active = np.zeros(6)

        while not self._stop_event.is_set():
            # Poll stdin with 50ms timeout
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)

            now = time.monotonic()

            if ready:
                ch = sys.stdin.read(1).lower()

                if ch == "\x1b":  # ESC
                    self._quit_event.set()
                    return

                if ch == " ":
                    with self._lock:
                        self._desired_twist_ee[:] = 0.0
                    last_active[:] = 0.0
                    continue

                if ch in KEY_MAP:
                    axis, sign = KEY_MAP[ch]
                    speed = LINEAR_SPEED if axis < 3 else ANGULAR_SPEED
                    with self._lock:
                        self._desired_twist_ee[axis] = sign * speed
                    last_active[axis] = now

            # Auto-release: zero any axis that hasn't been refreshed recently
            for axis in range(6):
                if last_active[axis] > 0 and (now - last_active[axis]) > KEY_RELEASE_TIMEOUT:
                    with self._lock:
                        self._desired_twist_ee[axis] = 0.0
                    last_active[axis] = 0.0


def control_loop(robot: Robot, keyboard: KeyboardReader):
    """1kHz Cartesian velocity control loop."""

    robot.set_joint_impedance(JOINT_IMPEDANCE)
    robot.set_collision_behavior(
        COLLISION_TORQUE, COLLISION_TORQUE,
        COLLISION_FORCE, COLLISION_FORCE,
    )

    active_control = robot.start_cartesian_velocity_control(
        ControllerMode.CartesianImpedance
    )

    current_twist_world = np.zeros(6)
    print("\nControl active. Use keys to move. ESC to quit.\n")

    try:
        while not keyboard.should_quit:
            state, duration = active_control.readOnce()

            # Get desired twist in EE frame from keyboard
            desired_ee = keyboard.get_desired_twist()

            # Transform EE frame → world frame
            desired_world = ee_twist_to_world(desired_ee, state.O_T_EE)

            # Ramp toward desired (acceleration limiting)
            current_twist_world = ramp_toward(current_twist_world, desired_world)

            # Safety clamp
            cmd_twist = clamp_twist(current_twist_world.copy())

            # Send command
            cmd = CartesianVelocities(cmd_twist.tolist())
            active_control.writeOnce(cmd)

    except pylibfranka.CommandException as e:
        print(f"\nCommand exception: {e}")
    except pylibfranka.ControlException as e:
        print(f"\nControl exception: {e}")

    # Graceful stop: ramp to zero
    print("\nRamping to zero velocity...")
    try:
        for _ in range(500):  # 500ms ramp-down
            state, duration = active_control.readOnce()
            current_twist_world = ramp_toward(current_twist_world, np.zeros(6))
            cmd = CartesianVelocities(current_twist_world.tolist())

            if np.allclose(current_twist_world, 0.0, atol=1e-6):
                cmd.motion_finished = True
                active_control.writeOnce(cmd)
                break

            active_control.writeOnce(cmd)
    except Exception:
        pass  # Best-effort ramp-down; robot will stop on its own if we disconnect

    robot.stop()
    print("Robot stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="EE velocity teleop for Franka via pylibfranka."
    )
    parser.add_argument(
        "--ip", type=str, required=True, help="Robot IP address"
    )
    parser.add_argument(
        "--no-rt", action="store_true",
        help="Disable real-time scheduling enforcement (RealtimeConfig.kIgnore)"
    )
    args = parser.parse_args()

    rt_config = RealtimeConfig.kIgnore if args.no_rt else RealtimeConfig.kEnforce

    print(f"Connecting to robot at {args.ip} ...")
    robot = Robot(args.ip, rt_config)
    robot.automatic_error_recovery()
    print("Connected.\n")

    print(HELP_TEXT)

    keyboard = KeyboardReader()
    keyboard.start()

    try:
        control_loop(robot, keyboard)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        robot.stop()
    finally:
        keyboard.stop()
        print("Terminal restored. Done.")


if __name__ == "__main__":
    main()
