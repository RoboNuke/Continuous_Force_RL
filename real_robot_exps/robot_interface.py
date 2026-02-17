"""
Real Robot Interface (pylibfranka)

Direct communication with FR3 via pylibfranka at ~1kHz.
A background thread runs the 1kHz readOnce/writeOnce loop continuously,
keeping the robot fed with torque commands while the main thread computes
policy actions at 15Hz. EMA-filters F/T at 1kHz within the loop.

Mock mode (use_mock: true) uses mock_pylibfranka for offline integration testing.
"""

import math
import time
import threading
from collections import namedtuple

import numpy as np
import torch


StateSnapshot = namedtuple('StateSnapshot', [
    'ee_pos',           # [3] torch.float32
    'ee_quat',          # [4] torch.float32 (w,x,y,z)
    'ee_linvel',        # [3] torch.float32
    'ee_angvel',        # [3] torch.float32
    'force_torque',     # [6] torch.float32
    'joint_pos',        # [7] torch.float32
    'joint_vel',        # [7] torch.float32
    'joint_torques',    # [7] torch.float32
    'jacobian',         # [6,7] torch.float32
    'mass_matrix',      # [7,7] torch.float32
])


class SafetyViolation(Exception):
    """Raised when a safety check fails on the real robot."""
    pass


def _rotation_matrix_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to (w, x, y, z) quaternion via Shepperd's method.

    Numerically stable for all rotation matrices.

    Args:
        R: [3, 3] rotation matrix (torch.Tensor).

    Returns:
        [4] quaternion (w, x, y, z).
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0].item() - R[1, 1].item() - R[2, 2].item())
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1].item() - R[0, 0].item() - R[2, 2].item())
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2].item() - R[0, 0].item() - R[1, 1].item())
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = torch.tensor([w, x, y, z], dtype=R.dtype, device=R.device)
    return q / torch.norm(q)  # ensure unit quaternion


def _rotation_matrix_to_quat_wxyz_np(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to (w, x, y, z) quaternion via Shepperd's method (numpy)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions (w, x, y, z)."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1


def _quat_wxyz_to_rotation_matrix_np(q: np.ndarray) -> np.ndarray:
    """Convert (w, x, y, z) quaternion to 3x3 rotation matrix (numpy)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def make_ee_target_pose(position: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """Construct 4x4 homogeneous transform from position [3] and RPY [3].

    Args:
        position: [3] XYZ position in meters.
        rpy: [3] Roll, Pitch, Yaw in radians (XYZ intrinsic Euler angles).

    Returns:
        [4, 4] homogeneous transform (row-major numpy array).
    """
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

    # Rotation matrices for each axis
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll) — extrinsic XYZ = intrinsic ZYX
    # But IsaacLab uses intrinsic XYZ, so R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R = Rx @ Ry @ Rz

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


# FR3 joint limits (from Franka Emika specifications)
_FR3_JOINT_POS_LIMITS = np.array([
    [-2.8973, 2.8973],
    [-1.7628, 1.7628],
    [-2.8973, 2.8973],
    [-3.0718, -0.0698],
    [-2.8973, 2.8973],
    [-0.0175, 3.7525],
    [-2.8973, 2.8973],
])

_FR3_JOINT_VEL_LIMITS = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])

# Max torque change per 1kHz step: 1000 Nm/s * 0.001s = 1.0 Nm theoretical max
# (from libfranka rate_limiting.h kMaxTorqueRate)
# Use 0.75 for safety margin
_MAX_TORQUE_DELTA = 0.75

_FR3_JOINT_TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

# Safety margins (percentage of limit before triggering violation)
_SAFETY_MARGIN_POS = 0.95  # warn at 95% of position limit
_SAFETY_MARGIN_VEL = 0.95
_SAFETY_MARGIN_FORCE = 50.0  # N, max L2 force norm


class FrankaInterface:
    """pylibfranka interface for Franka FR3.

    A background thread runs the 1kHz readOnce/writeOnce loop continuously.
    The main thread sets target torques via send_joint_torques() and reads
    the latest state via get_state_snapshot(). The background thread ramps
    commanded torques toward the held target and builds immutable StateSnapshot
    objects that the main thread reads lock-free via GIL reference swap.

    Args:
        config: Dictionary loaded from real_robot_exps/config.yaml.
        device: Torch device for tensor outputs (default: "cpu").
    """

    def __init__(self, config: dict, device: str = "cpu"):
        self._device = device
        self._config = config
        robot_cfg = config['robot']

        # Import pylibfranka or mock
        use_mock = robot_cfg.get('use_mock', False)
        if use_mock:
            import real_robot_exps.mock_pylibfranka as _pylibfranka
            print("[FrankaInterface] Using mock pylibfranka")
        else:
            import pylibfranka as _pylibfranka
            print("[FrankaInterface] Using real pylibfranka")
        self._pylibfranka = _pylibfranka
        self._Torques = _pylibfranka.Torques

        # Connect to robot
        ip = robot_cfg['ip']
        self._robot = _pylibfranka.Robot(ip)

        # Set EE and stiffness frames
        NE_T_EE = robot_cfg.get('NE_T_EE', [
            0.7071, -0.7071, 0.0, 0.0,
            0.7071, 0.7071, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.1034, 1.0,
        ])
        EE_T_K = robot_cfg.get('EE_T_K', [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
        self._robot.set_EE(NE_T_EE)
        self._robot.set_K(EE_T_K)

        # Raise collision thresholds to prevent premature reflex errors during contact
        self._robot.set_collision_behavior(
            [100.0] * 7, [100.0] * 7,  # joint torque: lower/upper thresholds
            [100.0] * 6, [100.0] * 6,  # Cartesian force: lower/upper thresholds
        )

        # Load dynamics model
        self._model = self._robot.load_model()

        # Control parameters
        self._control_rate_hz = robot_cfg.get('control_rate_hz', 15.0)
        self._reset_duration_sec = robot_cfg.get('reset_duration_sec', 3.0)

        # F/T EMA filter state
        self._ft_ema_alpha = robot_cfg.get('ft_ema_alpha', 0.2)
        self._ft_ema = np.zeros(6)

        # Held torques (target set by main thread, read by background thread)
        self._held_torques = [0.0] * 7
        # Commanded torques (actual values sent, ramped toward _held_torques)
        # ONLY touched by background thread — no lock needed
        self._cmd_torques = [0.0] * 7

        # Torque control handle (set by start_torque_mode)
        self._ctrl = None

        # State validity flag (for check_safety guard)
        self._state_valid = False

        # Background thread synchronization
        self._torque_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._comm_thread = None

        # Latest snapshot (immutable NamedTuple, swapped atomically via GIL)
        self._latest_snapshot = None

        # 15Hz policy step timing
        self._last_send_time = None

        print(f"[FrankaInterface] control_rate={self._control_rate_hz}Hz, "
              f"ft_ema_alpha={self._ft_ema_alpha}")

    # -------------------------------------------------------------------------
    # Background 1kHz communication loop
    # -------------------------------------------------------------------------

    def _comm_loop(self):
        """1kHz readOnce/writeOnce loop running in background thread."""
        while not self._stop_event.is_set():
            state, _ = self._ctrl.readOnce()

            # EMA filter F/T at 1kHz
            raw_ft = np.array(state.O_F_ext_hat_K)
            self._ft_ema = self._ft_ema_alpha * raw_ft + (1.0 - self._ft_ema_alpha) * self._ft_ema

            # Ramp toward held torques (rate-limited)
            with self._torque_lock:
                held = list(self._held_torques)
            for j in range(7):
                delta = held[j] - self._cmd_torques[j]
                if delta > _MAX_TORQUE_DELTA:
                    self._cmd_torques[j] += _MAX_TORQUE_DELTA
                elif delta < -_MAX_TORQUE_DELTA:
                    self._cmd_torques[j] -= _MAX_TORQUE_DELTA
                else:
                    self._cmd_torques[j] = held[j]

            self._ctrl.writeOnce(self._Torques(self._cmd_torques))

            # Build snapshot every iteration (immutable namedtuple, GIL-atomic swap)
            self._latest_snapshot = self._build_snapshot(state)

    # -------------------------------------------------------------------------
    # Core methods
    # -------------------------------------------------------------------------

    def send_joint_torques(self, torques: torch.Tensor):
        """Set target [7] torques for the background 1kHz loop.

        The background thread ramps commanded torques toward this target
        at the rate-limited _MAX_TORQUE_DELTA per 1kHz step.

        Also starts the 15Hz policy step timer.

        Args:
            torques: [7] joint torques tensor.

        Raises:
            ValueError: If shape is not (7,).
        """
        if torques.shape != (7,):
            raise ValueError(f"Expected [7] torques, got {torques.shape}")
        with self._torque_lock:
            self._held_torques = torques.detach().cpu().tolist()
        # Start 15Hz timer when policy output is sent
        self._last_send_time = time.time()

    def get_state_snapshot(self) -> StateSnapshot:
        """Get latest robot state (lock-free read of immutable snapshot).

        Returns:
            StateSnapshot namedtuple with all robot state fields.

        Raises:
            RuntimeError: If no snapshot available (call start_torque_mode first).
        """
        snap = self._latest_snapshot
        if snap is None:
            raise RuntimeError("No state snapshot available. Call start_torque_mode() first.")
        return snap

    def wait_for_policy_step(self):
        """Block until 1/control_rate_hz has elapsed since last send_joint_torques().

        Timer starts when send_joint_torques() is called (when policy produces output),
        NOT when we start computing the next observation/action.
        """
        if self._last_send_time is None:
            return  # First step, no waiting needed
        target_dt = 1.0 / self._control_rate_hz
        elapsed = time.time() - self._last_send_time
        remaining = target_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def retract_up(self, height_m: float):
        """Retract EE vertically upward by the specified height.

        Uses Cartesian pose control with cosine ramp (same as reset motion).
        Keeps orientation fixed, only moves in world +Z.

        Args:
            height_m: Distance to retract upward in meters.

        Raises:
            ValueError: If height_m <= 0.
        """
        if height_m <= 0:
            raise ValueError(f"retract height must be > 0, got {height_m}")

        ctrl = self._robot.start_cartesian_pose_control(
            self._pylibfranka.ControllerMode.JointImpedance
        )

        state, _ = ctrl.readOnce()
        start_flat = np.array(state.O_T_EE)  # column-major flat [16]
        start_t = start_flat[12:15].copy()

        target_t = start_t.copy()
        target_t[2] += height_m  # move up in world Z

        # Use 1 second for retract (1000 steps at 1kHz)
        n_steps = 1000
        for i in range(n_steps):
            alpha = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / n_steps))

            interp_t = (1.0 - alpha) * start_t + alpha * target_t

            # Keep original rotation, only update translation
            interp_flat = start_flat.copy()
            interp_flat[12] = interp_t[0]
            interp_flat[13] = interp_t[1]
            interp_flat[14] = interp_t[2]

            pose_cmd = self._pylibfranka.CartesianPose(interp_flat.tolist())

            if i == n_steps - 1:
                pose_cmd.motion_finished = True
                ctrl.writeOnce(pose_cmd)
                break
            ctrl.writeOnce(pose_cmd)
            state, _ = ctrl.readOnce()

        self.end_control()
        print(f"[FrankaInterface] Retracted {height_m*100:.1f}cm upward")

    def reset_to_start_pose(self, target_pose_4x4: np.ndarray):
        """Move to target pose via Cartesian control, then stop.

        Leaves the robot idle after motion completes. Call start_torque_mode()
        separately to begin torque control for the rollout.

        Stores a StateSnapshot from the final Cartesian state so that
        get_state_snapshot() works between reset and torque mode start.

        Args:
            target_pose_4x4: [4, 4] numpy homogeneous transform (row-major)
                for target EE pose.
        """
        ctrl = self._robot.start_cartesian_pose_control(
            self._pylibfranka.ControllerMode.JointImpedance
        )

        state, _ = ctrl.readOnce()
        start_flat = np.array(state.O_T_EE)  # column-major flat [16]

        # Extract start rotation and translation from column-major flat
        start_R = np.array([
            [start_flat[0], start_flat[4], start_flat[8]],
            [start_flat[1], start_flat[5], start_flat[9]],
            [start_flat[2], start_flat[6], start_flat[10]],
        ])
        start_t = start_flat[12:15]

        # Extract target rotation and translation from row-major 4x4
        target_R = target_pose_4x4[:3, :3]
        target_t = target_pose_4x4[:3, 3]

        # Convert rotations to quaternions for SLERP
        start_q = _rotation_matrix_to_quat_wxyz_np(start_R)
        target_q = _rotation_matrix_to_quat_wxyz_np(target_R)

        n_steps = int(self._reset_duration_sec * 1000)  # e.g. 3s = 3000 steps at 1kHz
        for i in range(n_steps):
            # Cosine ramp: zero velocity/acceleration at start and end
            alpha = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / n_steps))

            # Lerp translation, slerp rotation
            interp_t = (1.0 - alpha) * start_t + alpha * target_t
            interp_q = _quat_slerp(start_q, target_q, alpha)
            interp_R = _quat_wxyz_to_rotation_matrix_np(interp_q)

            # Reconstruct column-major flat [16]
            interp_flat = np.array([
                interp_R[0, 0], interp_R[1, 0], interp_R[2, 0], 0.0,
                interp_R[0, 1], interp_R[1, 1], interp_R[2, 1], 0.0,
                interp_R[0, 2], interp_R[1, 2], interp_R[2, 2], 0.0,
                interp_t[0], interp_t[1], interp_t[2], 1.0,
            ])

            pose_cmd = self._pylibfranka.CartesianPose(interp_flat.tolist())

            if i == n_steps - 1:
                pose_cmd.motion_finished = True
                ctrl.writeOnce(pose_cmd)
                break
            ctrl.writeOnce(pose_cmd)
            state, _ = ctrl.readOnce()

        # Build snapshot from final Cartesian state so get_state_snapshot() works
        # between reset and torque mode start (e.g. for controller.reset())
        self._latest_snapshot = self._build_snapshot(state)
        self._state_valid = True

        self.end_control()
        print("[FrankaInterface] Reset to start pose complete")

    def start_torque_mode(self):
        """Start torque control mode with background 1kHz communication thread.

        Begins torque control, runs a 100ms warm-up at 1kHz with zero
        torques to establish a stable control stream, then starts the
        background thread that continues the 1kHz loop.
        """
        self._held_torques = [0.0] * 7
        self._cmd_torques = [0.0] * 7
        self._ft_ema = np.zeros(6)
        self._ctrl = self._robot.start_torque_control()

        # Synchronous warmup (100 steps at 1kHz = 100ms)
        warmup_steps = 100
        for wi in range(warmup_steps):
            state, _ = self._ctrl.readOnce()
            self._ctrl.writeOnce(self._Torques([0.0] * 7))

        # Build initial snapshot from warmup
        self._latest_snapshot = self._build_snapshot(state)
        self._state_valid = True

        # Start background thread
        self._stop_event.clear()
        self._comm_thread = threading.Thread(target=self._comm_loop, daemon=True)
        self._comm_thread.start()
        self._last_send_time = time.time()

        print(f"[FrankaInterface] Torque control started (warmup {warmup_steps} steps OK), "
              f"background 1kHz thread running")

    def check_safety(self, snapshot: StateSnapshot):
        """Check joint pos/vel limits and force magnitude.

        Args:
            snapshot: StateSnapshot to check.

        Raises:
            SafetyViolation: If any limit exceeded.
        """
        # Joint position limits
        q = snapshot.joint_pos.cpu().numpy()
        for i in range(7):
            low = _FR3_JOINT_POS_LIMITS[i, 0] * _SAFETY_MARGIN_POS
            high = _FR3_JOINT_POS_LIMITS[i, 1] * _SAFETY_MARGIN_POS
            if q[i] < low or q[i] > high:
                raise SafetyViolation(
                    f"Joint {i} position {q[i]:.4f} rad outside safe range "
                    f"[{low:.4f}, {high:.4f}]"
                )

        # Joint velocity limits
        dq = snapshot.joint_vel.cpu().numpy()
        for i in range(7):
            limit = _FR3_JOINT_VEL_LIMITS[i] * _SAFETY_MARGIN_VEL
            if abs(dq[i]) > limit:
                raise SafetyViolation(
                    f"Joint {i} velocity {dq[i]:.4f} rad/s exceeds limit {limit:.4f}"
                )

        # Force magnitude
        force_mag = torch.norm(snapshot.force_torque[:3]).item()
        if force_mag > _SAFETY_MARGIN_FORCE:
            raise SafetyViolation(
                f"Force magnitude {force_mag:.2f}N exceeds limit {_SAFETY_MARGIN_FORCE:.1f}N"
            )

    def end_control(self):
        """End the active control session. Keeps the robot connection alive.

        Stops the background 1kHz thread, drops the ActiveControl handle,
        and calls robot.stop() to cleanly reset the robot to idle.

        No-op if no control session is active.
        """
        if self._ctrl is not None:
            # Stop background thread first
            self._stop_event.set()
            if self._comm_thread is not None:
                self._comm_thread.join(timeout=2.0)
                self._comm_thread = None
            self._ctrl = None
            self._robot.stop()
            self._state_valid = False
            self._latest_snapshot = None
            self._last_send_time = None

    def shutdown(self):
        """End control session and close robot connection."""
        self.end_control()
        self._robot.stop()
        print("[FrankaInterface] Shutdown complete.")

    # -------------------------------------------------------------------------
    # Internal state building
    # -------------------------------------------------------------------------

    def _build_snapshot(self, state) -> StateSnapshot:
        """Build immutable StateSnapshot from pylibfranka RobotState.

        Args:
            state: pylibfranka.RobotState (or mock equivalent).

        Returns:
            StateSnapshot namedtuple with all robot state as torch tensors.
        """
        device = self._device

        # O_T_EE is column-major flat [16]: columns are [R|t; 0 0 0 1]
        T = np.array(state.O_T_EE)

        # Position: translation is at indices [12, 13, 14] in column-major
        ee_pos = torch.tensor(
            [T[12], T[13], T[14]], device=device, dtype=torch.float32
        )

        # Rotation matrix from O_T_EE columns (column-major layout)
        R = torch.tensor([
            [T[0], T[4], T[8]],
            [T[1], T[5], T[9]],
            [T[2], T[6], T[10]],
        ], device=device, dtype=torch.float32)
        ee_quat = _rotation_matrix_to_quat_wxyz(R)

        # Joint state
        joint_pos = torch.tensor(state.q, device=device, dtype=torch.float32)
        joint_vel = torch.tensor(state.dq, device=device, dtype=torch.float32)
        joint_torques = torch.tensor(state.tau_J, device=device, dtype=torch.float32)

        # Jacobian: model.zero_jacobian(state) returns 42-element list (column-major 6x7)
        jac_flat = np.array(self._model.zero_jacobian(state))
        jacobian = torch.tensor(
            jac_flat.reshape(6, 7), device=device, dtype=torch.float32
        )

        # Mass matrix: model.mass(state) returns 49-element list (column-major 7x7)
        mass_flat = np.array(self._model.mass(state))
        mass_matrix = torch.tensor(
            mass_flat.reshape(7, 7), device=device, dtype=torch.float32
        )

        # EE velocity: J @ dq
        dq = joint_vel.unsqueeze(1)  # [7, 1]
        ee_vel = (jacobian @ dq).squeeze(1)  # [6]
        ee_linvel = ee_vel[:3]
        ee_angvel = ee_vel[3:]

        # Force/torque: negate EMA-filtered O_F_ext_hat_K (to match training convention)
        force_torque = torch.tensor(
            -self._ft_ema, device=device, dtype=torch.float32
        )

        return StateSnapshot(
            ee_pos, ee_quat, ee_linvel, ee_angvel, force_torque,
            joint_pos, joint_vel, joint_torques, jacobian, mass_matrix,
        )

