"""
Real Robot Interface (pylibfranka)

Direct communication with FR3 via pylibfranka at ~1kHz.
Policy runs at 15Hz; read_state() bridges by running a tight 1kHz
readOnce/writeOnce loop for one policy timestep, EMA-filtering F/T at 1kHz.

Mock mode (use_mock: true) uses mock_pylibfranka for offline integration testing.
"""

import math
import time

import numpy as np
import torch


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

_FR3_JOINT_TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

# Safety margins (percentage of limit before triggering violation)
_SAFETY_MARGIN_POS = 0.95  # warn at 95% of position limit
_SAFETY_MARGIN_VEL = 0.95
_SAFETY_MARGIN_FORCE = 50.0  # N, max L2 force norm


class FrankaInterface:
    """pylibfranka interface for Franka FR3.

    Communicates at ~1kHz via pylibfranka's ActiveControl.
    read_state() runs the 1kHz loop for one policy timestep, EMA-filtering
    F/T readings at 1kHz. send_joint_torques() just stores torques for the
    next read_state() cycle.

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

        # Load dynamics model
        self._model = self._robot.load_model()

        # Control parameters
        self._control_rate_hz = robot_cfg.get('control_rate_hz', 15.0)
        self._reset_duration_sec = robot_cfg.get('reset_duration_sec', 3.0)

        # F/T EMA filter state
        self._ft_ema_alpha = robot_cfg.get('ft_ema_alpha', 0.2)
        self._ft_ema = np.zeros(6)

        # Held torques (applied in read_state's 1kHz loop)
        self._held_torques = [0.0] * 7

        # Torque control handle (set by reset_to_start_pose)
        self._ctrl = None

        # State validity flag (fail-fast for get_*() before read_state)
        self._state_valid = False

        # Cached state tensors (populated by _parse_state)
        self._ee_pos = None
        self._ee_quat = None
        self._ee_linvel = None
        self._ee_angvel = None
        self._force_torque = None
        self._joint_pos = None
        self._joint_vel = None
        self._joint_torques = None
        self._jacobian = None
        self._mass_matrix = None

        print(f"[FrankaInterface] control_rate={self._control_rate_hz}Hz, "
              f"ft_ema_alpha={self._ft_ema_alpha}")

    # -------------------------------------------------------------------------
    # Core methods
    # -------------------------------------------------------------------------

    def read_state(self):
        """Run 1kHz readOnce/writeOnce loop for one policy step.

        Applies EMA low-pass filter to F/T at 1kHz within the loop.
        Caches final state as torch tensors.

        Raises:
            RuntimeError: If torque control not started (call reset_to_start_pose first).
        """
        if self._ctrl is None:
            raise RuntimeError("Torque control not started. Call reset_to_start_pose() first.")

        target_dt = 1.0 / self._control_rate_hz
        start = time.time()

        while True:
            state, duration = self._ctrl.readOnce()

            # EMA filter on F/T at 1kHz
            raw_ft = np.array(state.O_F_ext_hat_K)
            self._ft_ema = (
                self._ft_ema_alpha * raw_ft
                + (1.0 - self._ft_ema_alpha) * self._ft_ema
            )

            # Send held torques
            cmd = self._Torques(self._held_torques)
            self._ctrl.writeOnce(cmd)

            if time.time() - start >= target_dt:
                break

        self._parse_state(state)

    def send_joint_torques(self, torques: torch.Tensor):
        """Store [7] torques for next read_state() cycle.

        Args:
            torques: [7] joint torques tensor.

        Raises:
            ValueError: If shape is not (7,).
        """
        if torques.shape != (7,):
            raise ValueError(f"Expected [7] torques, got {torques.shape}")
        self._held_torques = torques.detach().cpu().tolist()

    def reset_to_start_pose(self, target_pose_4x4: np.ndarray):
        """Move to target pose via Cartesian control, then start torque control.

        Single code path for both real robot and mock.

        Args:
            target_pose_4x4: [4, 4] numpy homogeneous transform (row-major)
                for target EE pose.
        """
        # Phase 1: Cartesian pose control to target
        ctrl = self._robot.start_cartesian_pose_control(
            self._pylibfranka.ControllerMode.JointImpedance
        )

        # Convert row-major 4x4 to column-major flat (libfranka convention)
        target_flat = target_pose_4x4.flatten(order='F')  # column-major

        state, _ = ctrl.readOnce()
        start_pose = np.array(state.O_T_EE)  # already column-major flat

        n_steps = int(self._reset_duration_sec * 1000)  # e.g. 3s = 3000 steps at 1kHz
        for i in range(n_steps):
            alpha = min(1.0, (i + 1) / n_steps)
            interp = (1.0 - alpha) * start_pose + alpha * target_flat
            pose_cmd = self._pylibfranka.CartesianPose(interp.tolist())
            if i == n_steps - 1:
                pose_cmd.motion_finished = True
            ctrl.writeOnce(pose_cmd)
            state, _ = ctrl.readOnce()

        # Phase 2: Switch to torque control for rollout
        self._ctrl = self._robot.start_torque_control()
        state, _ = self._ctrl.readOnce()
        self._ctrl.writeOnce(self._Torques([0.0] * 7))
        self._parse_state(state)

        # Reset internal state
        self._held_torques = [0.0] * 7
        self._ft_ema = np.zeros(6)
        self._state_valid = True

    def check_safety(self):
        """Check joint pos/vel limits and force magnitude on cached state.

        Raises:
            SafetyViolation: If any limit exceeded.
        """
        if not self._state_valid:
            raise RuntimeError("No valid state. Call read_state() first.")

        # Joint position limits
        q = self._joint_pos.cpu().numpy()
        for i in range(7):
            low = _FR3_JOINT_POS_LIMITS[i, 0] * _SAFETY_MARGIN_POS
            high = _FR3_JOINT_POS_LIMITS[i, 1] * _SAFETY_MARGIN_POS
            if q[i] < low or q[i] > high:
                raise SafetyViolation(
                    f"Joint {i} position {q[i]:.4f} rad outside safe range "
                    f"[{low:.4f}, {high:.4f}]"
                )

        # Joint velocity limits
        dq = self._joint_vel.cpu().numpy()
        for i in range(7):
            limit = _FR3_JOINT_VEL_LIMITS[i] * _SAFETY_MARGIN_VEL
            if abs(dq[i]) > limit:
                raise SafetyViolation(
                    f"Joint {i} velocity {dq[i]:.4f} rad/s exceeds limit {limit:.4f}"
                )

        # Force magnitude
        force_mag = torch.norm(self._force_torque[:3]).item()
        if force_mag > _SAFETY_MARGIN_FORCE:
            raise SafetyViolation(
                f"Force magnitude {force_mag:.2f}N exceeds limit {_SAFETY_MARGIN_FORCE:.1f}N"
            )

    def shutdown(self):
        """Send motion_finished=True via Torques, call robot.stop()."""
        if self._ctrl is not None:
            cmd = self._Torques([0.0] * 7)
            cmd.motion_finished = True
            self._ctrl.writeOnce(cmd)
        self._robot.stop()
        print("[FrankaInterface] Shutdown complete.")

    # -------------------------------------------------------------------------
    # State getters (return cached tensors, fail-fast if no valid state)
    # -------------------------------------------------------------------------

    def _require_valid_state(self):
        if not self._state_valid:
            raise RuntimeError(
                "No valid robot state. Call reset_to_start_pose() and read_state() first."
            )

    def get_ee_position(self) -> torch.Tensor:
        """Get end-effector (fingertip midpoint) position in world frame.

        Returns:
            [3] tensor in meters.
        """
        self._require_valid_state()
        return self._ee_pos

    def get_ee_orientation(self) -> torch.Tensor:
        """Get end-effector orientation as quaternion.

        Returns:
            [4] tensor (w, x, y, z).
        """
        self._require_valid_state()
        return self._ee_quat

    def get_ee_linear_velocity(self) -> torch.Tensor:
        """Get end-effector linear velocity (J @ dq).

        Returns:
            [3] tensor in m/s.
        """
        self._require_valid_state()
        return self._ee_linvel

    def get_ee_angular_velocity(self) -> torch.Tensor:
        """Get end-effector angular velocity (J @ dq).

        Returns:
            [3] tensor in rad/s.
        """
        self._require_valid_state()
        return self._ee_angvel

    def get_force_torque(self) -> torch.Tensor:
        """Get force/torque at end-effector (negated EMA-filtered O_F_ext_hat_K).

        Returns:
            [6] tensor (Fx, Fy, Fz, Tx, Ty, Tz) in N and Nm.
        """
        self._require_valid_state()
        return self._force_torque

    def get_joint_positions(self) -> torch.Tensor:
        """Get 7-DOF arm joint positions.

        Returns:
            [7] tensor in radians.
        """
        self._require_valid_state()
        return self._joint_pos

    def get_joint_velocities(self) -> torch.Tensor:
        """Get 7-DOF arm joint velocities.

        Returns:
            [7] tensor in rad/s.
        """
        self._require_valid_state()
        return self._joint_vel

    def get_joint_torques(self) -> torch.Tensor:
        """Get 7-DOF arm joint torques (measured).

        Returns:
            [7] tensor in Nm.
        """
        self._require_valid_state()
        return self._joint_torques

    def get_jacobian(self) -> torch.Tensor:
        """Get geometric Jacobian at end-effector frame.

        Returns:
            [6, 7] tensor mapping joint velocities to task-space velocities.
        """
        self._require_valid_state()
        return self._jacobian

    def get_mass_matrix(self) -> torch.Tensor:
        """Get 7x7 arm mass (inertia) matrix.

        Returns:
            [7, 7] tensor in kg*m^2.
        """
        self._require_valid_state()
        return self._mass_matrix

    # -------------------------------------------------------------------------
    # Internal state parsing
    # -------------------------------------------------------------------------

    def _parse_state(self, state):
        """Parse pylibfranka RobotState into cached torch tensors.

        Args:
            state: pylibfranka.RobotState (or mock equivalent).
        """
        device = self._device

        # O_T_EE is column-major flat [16]: columns are [R|t; 0 0 0 1]
        T = np.array(state.O_T_EE)

        # Position: translation is at indices [12, 13, 14] in column-major
        self._ee_pos = torch.tensor(
            [T[12], T[13], T[14]], device=device, dtype=torch.float32
        )

        # Rotation matrix from O_T_EE columns (column-major layout)
        # col0 = T[0:4], col1 = T[4:8], col2 = T[8:12]
        R = torch.tensor([
            [T[0], T[4], T[8]],
            [T[1], T[5], T[9]],
            [T[2], T[6], T[10]],
        ], device=device, dtype=torch.float32)
        self._ee_quat = _rotation_matrix_to_quat_wxyz(R)

        # Joint state
        self._joint_pos = torch.tensor(state.q, device=device, dtype=torch.float32)
        self._joint_vel = torch.tensor(state.dq, device=device, dtype=torch.float32)
        self._joint_torques = torch.tensor(state.tau_J, device=device, dtype=torch.float32)

        # Jacobian: model.zero_jacobian(state) returns 42-element list (column-major 6x7)
        # Reshape to row-major [6, 7] — validated on real robot
        jac_flat = np.array(self._model.zero_jacobian(state))
        self._jacobian = torch.tensor(
            jac_flat.reshape(6, 7), device=device, dtype=torch.float32
        )

        # Mass matrix: model.mass(state) returns 49-element list (column-major 7x7)
        # Reshape to row-major [7, 7]
        mass_flat = np.array(self._model.mass(state))
        self._mass_matrix = torch.tensor(
            mass_flat.reshape(7, 7), device=device, dtype=torch.float32
        )

        # EE velocity: J @ dq
        dq = self._joint_vel.unsqueeze(1)  # [7, 1]
        ee_vel = (self._jacobian @ dq).squeeze(1)  # [6]
        self._ee_linvel = ee_vel[:3]
        self._ee_angvel = ee_vel[3:]

        # Force/torque: negate EMA-filtered O_F_ext_hat_K (to match training convention)
        self._force_torque = torch.tensor(
            -self._ft_ema, device=device, dtype=torch.float32
        )
