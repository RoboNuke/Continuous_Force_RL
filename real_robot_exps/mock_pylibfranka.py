"""
Mock pylibfranka interface for offline integration testing.

Mimics the pylibfranka API with correct data types and shapes so that
MATCH deployment code can be developed and tested without robot access.

API verified against pylibfranka 0.20.4 docs + live introspection on FR3.

Key findings from validation:
  - pylibfranka.Frame does NOT exist (not exposed to Python)
  - model.zero_jacobian(state) defaults to EE frame; overload with frame uses
    internal franka::Frame enum not accessible from Python
  - ctrl.readOnce() returns (RobotState, Duration) tuple, not just RobotState

Usage:
    try:
        import pylibfranka
        robot = pylibfranka.Robot("192.168.1.11")
    except Exception:
        from mock_pylibfranka import Robot
        robot = Robot("192.168.1.11")
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


class Torques:
    def __init__(self, tau_J: list):
        assert len(tau_J) == 7, f"Expected 7 torques, got {len(tau_J)}"
        self.tau_J = list(tau_J)
        self.motion_finished = False


class ControllerMode:
    """Mimics pylibfranka.ControllerMode enum."""
    JointImpedance = 0
    CartesianImpedance = 1


class CartesianPose:
    """Mimics pylibfranka.CartesianPose command."""
    def __init__(self, O_T_EE: list):
        assert len(O_T_EE) == 16, f"Expected 16 elements, got {len(O_T_EE)}"
        self.O_T_EE = list(O_T_EE)
        self.motion_finished = False


class JointPositions:
    """Mimics pylibfranka.JointPositions command."""
    def __init__(self, q: list):
        assert len(q) == 7, f"Expected 7 joint positions, got {len(q)}"
        self.q = list(q)
        self.motion_finished = False


class Duration:
    """Mimics pylibfranka.Duration (time since last readOnce)."""
    def __init__(self, milliseconds: int = 1):
        self._ms = milliseconds

    def toSec(self) -> float:
        return self._ms / 1000.0

    def toMSec(self) -> int:
        return self._ms

    def __repr__(self):
        return f"Duration({self._ms}ms)"


# Default FR3 home configuration
_HOME_Q = [-0.0062, -0.7871, 0.0146, -2.352, 0.0013, 1.5679, 0.7698]

# Realistic O_T_EE for home config (column-major, 16 elements)
_HOME_O_T_EE = [
    0.9987, -0.0028, -0.0512, 0.0,
    -0.0045, -0.9996, -0.0281, 0.0,
    -0.0511, 0.0283, -0.9983, 0.0,
    0.3070, 0.0020, 0.3657, 1.0,
]

# Realistic mass matrix at home config (symmetric 7x7)
_HOME_MASS = np.array([
    [0.7976, -0.0138, -0.0256, 0.2465, -0.0055, 0.0327, 0.0000],
    [-0.0138, 1.4537, 0.0490, -0.4363, -0.0198, 0.0954, -0.0013],
    [-0.0256, 0.0490, 0.5717, -0.0025, -0.0400, 0.0035, -0.0012],
    [0.2465, -0.4363, -0.0025, 0.8557, 0.0197, -0.1195, 0.0013],
    [-0.0055, -0.0198, -0.0400, 0.0197, 0.0316, -0.0045, 0.0030],
    [0.0327, 0.0954, 0.0035, -0.1195, -0.0045, 0.0360, -0.0003],
    [0.0000, -0.0013, -0.0012, 0.0013, 0.0030, -0.0003, 0.0012],
])

# Realistic gravity vector at home config
_HOME_GRAVITY = np.array([0.0, -5.1843, -0.1257, 6.0498, -0.0659, 0.8780, 0.0])

# Realistic coriolis vector (near-zero at rest)
_HOME_CORIOLIS = np.zeros(7)

# Realistic Jacobian at home config (6x7)
_HOME_JACOBIAN = np.array([
    [-0.0020, -0.3657, 0.0055, 0.2698, -0.0023, -0.0593, 0.0],
    [0.3070,  0.0003, -0.3073, -0.0020,  0.0621, -0.0003, 0.0],
    [0.0,    -0.3070, -0.0005, 0.2355,  0.0003, -0.0449, 0.0],
    [0.0,     0.0512,  0.0028, -0.0286, -0.0511,  0.0337, -0.9983],
    [0.0,     0.9983, -0.0511, -0.9979,  0.0283, -0.9375,  0.0283],
    [1.0,    -0.0045,  0.9987, -0.0045,  0.9983, -0.0515,  0.0512],
])


@dataclass
class RobotState:
    """Mimics franka::RobotState with all fields MATCH needs."""
    # Joint state
    q: List[float] = field(default_factory=lambda: list(_HOME_Q))
    q_d: List[float] = field(default_factory=lambda: list(_HOME_Q))
    dq: List[float] = field(default_factory=lambda: [0.0] * 7)
    dq_d: List[float] = field(default_factory=lambda: [0.0] * 7)
    ddq_d: List[float] = field(default_factory=lambda: [0.0] * 7)

    # Torques
    tau_J: List[float] = field(default_factory=lambda: [0.0] * 7)
    tau_J_d: List[float] = field(default_factory=lambda: [0.0] * 7)
    dtau_J: List[float] = field(default_factory=lambda: [0.0] * 7)
    tau_ext_hat_filtered: List[float] = field(
        default_factory=lambda: [0.2694, 1.0158, 0.057, -2.1104, 0.2811, -0.3341, 0.0566]
    )

    # End-effector pose (column-major 4x4 -> 16 elements)
    O_T_EE: List[float] = field(default_factory=lambda: list(_HOME_O_T_EE))
    O_T_EE_d: List[float] = field(default_factory=lambda: list(_HOME_O_T_EE))
    F_T_EE: List[float] = field(
        default_factory=lambda: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    )
    EE_T_K: List[float] = field(
        default_factory=lambda: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    )

    # External forces
    O_F_ext_hat_K: List[float] = field(
        default_factory=lambda: [-1.0419, 0.6595, -4.8659, -0.1699, 0.6677, 0.1935]
    )
    K_F_ext_hat_K: List[float] = field(
        default_factory=lambda: [0.0] * 6
    )

    # Misc
    time: float = 0.0
    control_command_success_rate: float = 1.0


class Model:
    """Mimics pylibfranka.Model with correct return shapes.

    zero_jacobian has two overloads in pylibfranka:
      1. zero_jacobian(state)           -> EE frame (default)
      2. zero_jacobian(frame, state)    -> specified frame (internal C++ enum)

    Since franka::Frame is NOT exposed to Python, deployment code should
    always use overload 1: model.zero_jacobian(state)
    """

    def mass(self, state: RobotState) -> list:
        """Returns 49-element list (7x7 mass matrix, column-major)."""
        return _HOME_MASS.flatten().tolist()

    def coriolis(self, state: RobotState) -> list:
        """Returns 7-element list (Coriolis vector)."""
        dq = np.array(state.dq)
        return (_HOME_CORIOLIS + 0.01 * dq).tolist()

    def gravity(self, state: RobotState) -> list:
        """Returns 7-element list (gravity compensation torques)."""
        return _HOME_GRAVITY.tolist()

    def zero_jacobian(self, *args) -> list:
        """Returns 42-element list (6x7 Jacobian, column-major).

        Overloads:
            zero_jacobian(state)         -> EE Jacobian (primary usage)
            zero_jacobian(frame, state)  -> Jacobian for given frame
        """
        return _HOME_JACOBIAN.flatten().tolist()


class ActiveControl:
    """Mimics the ActiveControlBase returned by start_torque_control()."""

    def __init__(self, state: RobotState):
        self._state = state
        self._step = 0
        self._last_time = time.time()
        self._last_tau = np.zeros(7)

    def readOnce(self) -> Tuple[RobotState, Duration]:
        """Returns (RobotState, Duration) tuple. Blocks to approximate 1kHz."""
        # Simulate 1kHz timing
        now = time.time()
        elapsed = now - self._last_time
        sleep_time = 0.001 - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._last_time = time.time()

        # Update time
        self._state.time = self._step * 0.001
        self._step += 1

        # Very simple forward integration for more realistic testing
        dt = 0.001
        dq = np.array(self._state.dq)
        q = np.array(self._state.q)

        M_diag = np.diag(_HOME_MASS)
        ddq = self._last_tau / M_diag * 0.001  # heavily damped
        dq_new = dq + ddq * dt
        dq_new *= 0.99  # damping
        q_new = q + dq_new * dt

        self._state.q = q_new.tolist()
        self._state.dq = dq_new.tolist()
        self._state.tau_J = self._last_tau.tolist()

        duration = Duration(milliseconds=1)
        return self._state, duration

    def writeOnce(self, torques: Torques):
        """Accept torque command."""
        assert isinstance(torques, Torques), f"Expected Torques, got {type(torques)}"
        self._last_tau = np.array(torques.tau_J)
        if torques.motion_finished:
            self._last_tau = np.zeros(7)


class CartesianActiveControl:
    """Mimics ActiveControlBase for Cartesian pose control.

    Instantly tracks commanded pose (no dynamics simulation).
    """

    def __init__(self, state: RobotState):
        self._state = state
        self._step = 0
        self._last_time = time.time()

    def readOnce(self) -> Tuple[RobotState, Duration]:
        """Returns (RobotState, Duration). Blocks to approximate 1kHz."""
        now = time.time()
        elapsed = now - self._last_time
        sleep_time = 0.001 - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._last_time = time.time()
        self._state.time = self._step * 0.001
        self._step += 1
        return self._state, Duration(milliseconds=1)

    def writeOnce(self, cmd: CartesianPose):
        """Accept Cartesian pose command. Instantly sets O_T_EE."""
        assert isinstance(cmd, CartesianPose), f"Expected CartesianPose, got {type(cmd)}"
        self._state.O_T_EE = list(cmd.O_T_EE)


class JointActiveControl:
    """Mimics ActiveControlBase for joint position control.

    Instantly tracks commanded joint positions (no dynamics simulation).
    """

    def __init__(self, state: RobotState):
        self._state = state
        self._step = 0
        self._last_time = time.time()

    def readOnce(self) -> Tuple[RobotState, Duration]:
        """Returns (RobotState, Duration). Blocks to approximate 1kHz."""
        now = time.time()
        elapsed = now - self._last_time
        sleep_time = 0.001 - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._last_time = time.time()
        self._state.time = self._step * 0.001
        self._step += 1
        return self._state, Duration(milliseconds=1)

    def writeOnce(self, cmd: JointPositions):
        """Accept joint position command. Instantly sets q."""
        assert isinstance(cmd, JointPositions), f"Expected JointPositions, got {type(cmd)}"
        self._state.q = list(cmd.q)
        self._state.q_d = list(cmd.q)


class Robot:
    """Mimics pylibfranka.Robot."""

    def __init__(self, ip: str):
        self._ip = ip
        self._state = RobotState()
        self._model = Model()
        self._active_control = None
        print(f"[MockRobot] Connected to mock robot at {ip}")
        print(f"[MockRobot] Using fixed home state. No physics simulation.")

    def read_once(self) -> RobotState:
        """Read robot state once (outside of control loop)."""
        return self._state

    def load_model(self) -> Model:
        """Load robot dynamics model."""
        return self._model

    def start_torque_control(self) -> ActiveControl:
        """Start torque control mode, returns ActiveControl handle."""
        self._active_control = ActiveControl(RobotState())  # fresh state
        print("[MockRobot] Torque control started.")
        return self._active_control

    def start_cartesian_pose_control(self, controller_mode) -> CartesianActiveControl:
        """Start Cartesian pose control mode."""
        self._active_control = CartesianActiveControl(RobotState())
        print("[MockRobot] Cartesian pose control started.")
        return self._active_control

    def start_joint_position_control(self, controller_mode) -> JointActiveControl:
        """Start joint position control mode."""
        self._active_control = JointActiveControl(RobotState())
        print("[MockRobot] Joint position control started.")
        return self._active_control

    def stop(self):
        """Stop the robot."""
        self._active_control = None
        print("[MockRobot] Stopped.")

    def automatic_error_recovery(self):
        print("[MockRobot] Error recovery (no-op).")

    def set_collision_behavior(self, *args, **kwargs):
        print("[MockRobot] Collision behavior set (no-op).")

    def set_joint_impedance(self, *args, **kwargs):
        print("[MockRobot] Joint impedance set (no-op).")

    def set_cartesian_impedance(self, *args, **kwargs):
        print("[MockRobot] Cartesian impedance set (no-op).")

    def set_load(self, *args, **kwargs):
        print("[MockRobot] Load set (no-op).")

    def set_EE(self, *args, **kwargs):
        print("[MockRobot] EE frame set (no-op).")

    def set_K(self, *args, **kwargs):
        print("[MockRobot] Stiffness frame set (no-op).")


@dataclass
class GripperState:
    """Mimics pylibfranka.GripperState."""
    width: float = 0.008
    max_width: float = 0.08
    is_grasped: bool = True
    temperature: float = 25.0


class Gripper:
    """Mimics pylibfranka.Gripper."""

    def __init__(self, ip: str):
        self._ip = ip
        self._state = GripperState()
        print(f"[MockGripper] Connected to mock gripper at {ip}")

    def homing(self) -> bool:
        print("[MockGripper] Homing (no-op).")
        self._state.width = self._state.max_width
        return True

    def grasp(self, width: float, speed: float, force: float,
              epsilon_inner: float = 0.005, epsilon_outer: float = 0.005) -> bool:
        print(f"[MockGripper] Grasp: width={width}m, speed={speed}m/s, force={force}N")
        self._state.is_grasped = True
        return True

    def move(self, width: float, speed: float) -> bool:
        print(f"[MockGripper] Move: width={width}m, speed={speed}m/s")
        self._state.width = width
        self._state.is_grasped = False
        return True

    def stop(self) -> bool:
        print("[MockGripper] Stop (no-op).")
        return True

    def read_once(self) -> GripperState:
        return self._state
