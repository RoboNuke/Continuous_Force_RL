"""
Hybrid Force/Position Controller for Real Robot

Pure PyTorch reimplementation of the sim control pipeline. No Isaac Sim dependency.

Replicates logic from:
- wrappers/control/hybrid_force_position_wrapper.py (hybrid EMA, target computation, blending)
- wrappers/control/factory_control_utils.py (pose wrench, force wrench, J^T torque mapping)
- IsaacLab FactoryEnv._apply_action() (base pose-only control)

Supports both control modes:
- POSE-ONLY (sigma_idx == 0): 6D actions [pos_x, pos_y, pos_z, rot_roll, rot_pitch, rot_yaw]
- HYBRID (sigma_idx > 0): 12/14/18D actions [selection, position, rotation, force]

Also outputs intermediate Cartesian targets for alternative robot control modes
(Cartesian impedance, joint position) - selected by ros2.control_mode in config.
"""

import math
from typing import Dict, Tuple

import torch


# ============================================================================
# Quaternion utilities (pure PyTorch, no Isaac Sim)
# ============================================================================

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of quaternion (w, x, y, z). Negates the vector part."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def axis_angle_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to axis-angle representation.

    Returns [3] axis-angle vector (direction = axis, magnitude = angle).
    """
    # Ensure w >= 0 for consistent angle extraction
    q = torch.where(q[..., 0:1] < 0, -q, q)
    sin_half = torch.norm(q[..., 1:4], dim=-1, keepdim=True).clamp(min=1e-12)
    angle = 2.0 * torch.atan2(sin_half, q[..., 0:1].abs())
    axis = q[..., 1:4] / sin_half
    return axis * angle


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Convert angle (scalar) and axis [3] to quaternion (w, x, y, z)."""
    half_angle = angle * 0.5
    sin_half = torch.sin(half_angle)
    cos_half = torch.cos(half_angle)
    # axis should be unit vector; if angle ~= 0 the axis doesn't matter
    return torch.cat([cos_half.unsqueeze(-1), axis * sin_half.unsqueeze(-1)], dim=-1)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles (XYZ intrinsic) to quaternion (w, x, y, z).

    Each input is a scalar tensor.
    """
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return torch.stack([w, x, y, z], dim=-1)


def get_euler_xyz(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract Euler angles (XYZ intrinsic) from quaternion (w, x, y, z).

    Returns (roll, pitch, yaw) as scalar tensors.
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# ============================================================================
# Wrench computation (matches factory_control_utils.py)
# ============================================================================

def compute_pose_error(
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute position error and rotation axis-angle error.

    Matches factory_control_utils.get_pose_error() with jacobian_type="geometric"
    and rot_error_type="axis_angle".

    All inputs are [N, D] batched or [D] unbatched.
    """
    pos_error = target_pos - ee_pos

    # Quaternion error: q_err = q_target * q_current^{-1}
    # Ensure shortest path
    quat_dot = (target_quat * ee_quat).sum(dim=-1, keepdim=True)
    target_quat_adj = torch.where(quat_dot >= 0, target_quat, -target_quat)

    # q_inv = q_conj / |q|^2, for unit quaternions |q|=1 so q_inv = q_conj
    ee_quat_conj = quat_conjugate(ee_quat)
    quat_norm_sq = (ee_quat * ee_quat).sum(dim=-1, keepdim=True)
    ee_quat_inv = ee_quat_conj / quat_norm_sq

    quat_error = quat_mul(target_quat_adj, ee_quat_inv)
    aa_error = axis_angle_from_quat(quat_error)

    return pos_error, aa_error


def compute_pose_task_wrench(
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    ee_linvel: torch.Tensor,
    ee_angvel: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    task_prop_gains: torch.Tensor,
    task_deriv_gains: torch.Tensor,
) -> torch.Tensor:
    """Compute task-space wrench for pose control.

    Matches factory_control_utils.compute_pose_task_wrench().

    Returns:
        [6] wrench (Fx, Fy, Fz, Tx, Ty, Tz).
    """
    pos_error, aa_error = compute_pose_error(ee_pos, ee_quat, target_pos, target_quat)
    delta_pose = torch.cat([pos_error, aa_error], dim=-1)

    # PD control: wrench = Kp * error - Kd * velocity
    wrench = torch.zeros_like(delta_pose)
    wrench[..., :3] = task_prop_gains[..., :3] * pos_error + task_deriv_gains[..., :3] * (0.0 - ee_linvel)
    wrench[..., 3:6] = task_prop_gains[..., 3:6] * aa_error + task_deriv_gains[..., 3:6] * (0.0 - ee_angvel)
    return wrench


def compute_force_task_wrench(
    force_torque: torch.Tensor,
    target_force: torch.Tensor,
    task_gains: torch.Tensor,
    task_deriv_gains: torch.Tensor = None,
    prev_force_error: torch.Tensor = None,
    task_integ_gains: torch.Tensor = None,
    force_integral_error: torch.Tensor = None,
    enable_derivative: bool = False,
    enable_integral: bool = False,
) -> torch.Tensor:
    """Compute task-space wrench for force control with optional PID.

    Matches factory_control_utils.compute_force_task_wrench().

    Returns:
        [6] wrench.
    """
    force_error = target_force - force_torque

    # P term
    wrench = task_gains * force_error

    # D term
    if enable_derivative and task_deriv_gains is not None and prev_force_error is not None:
        error_delta = force_error - prev_force_error
        wrench = wrench + task_deriv_gains * error_delta

    # I term
    if enable_integral and task_integ_gains is not None and force_integral_error is not None:
        wrench = wrench + task_integ_gains * force_integral_error

    return wrench


def compute_joint_torques_from_wrench(
    task_wrench: torch.Tensor,
    jacobian: torch.Tensor,
    mass_matrix: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    default_dof_pos: torch.Tensor,
    kp_null: float = 10.0,
    kd_null: float = 6.3246,
) -> torch.Tensor:
    """Compute joint torques via J^T mapping with null-space compensation.

    Matches factory_control_utils.compute_dof_torque_from_wrench() for single env.

    Args:
        task_wrench: [6] task-space wrench.
        jacobian: [6, 7] geometric Jacobian.
        mass_matrix: [7, 7] arm mass matrix.
        joint_pos: [7] current joint positions.
        joint_vel: [7] current joint velocities.
        default_dof_pos: [7] default/home joint positions for null-space.
        kp_null: Null-space position gain.
        kd_null: Null-space damping gain.

    Returns:
        [7] joint torques in Nm.
    """
    # J^T * wrench
    jacobian_T = jacobian.T  # [7, 6]
    torque = jacobian_T @ task_wrench  # [7]

    # Null-space compensation
    M_inv = torch.inverse(mass_matrix)  # [7, 7]
    M_task = torch.inverse(jacobian @ M_inv @ jacobian_T)  # [6, 6]
    J_inv = M_task @ jacobian @ M_inv  # [6, 7]

    # Distance to default pose (wrapped to [-pi, pi])
    dist = default_dof_pos - joint_pos
    dist = (dist + math.pi) % (2 * math.pi) - math.pi

    u_null = kd_null * (-joint_vel) + kp_null * dist  # [7]
    u_null = mass_matrix @ u_null  # [7]

    # Null-space projector: (I - J^T @ J_inv^T)
    I7 = torch.eye(7, device=joint_pos.device, dtype=joint_pos.dtype)
    null_proj = I7 - jacobian_T @ J_inv
    torque_null = null_proj @ u_null  # [7]

    torque = torque + torque_null

    # Clamp to safe limits (matching sim)
    torque = torch.clamp(torque, min=-100.0, max=100.0)
    return torque


# ============================================================================
# Main controller class
# ============================================================================

class RealRobotController:
    """Operational space controller supporting both pose-only and hybrid control.

    Detects mode from training configs and replicates the exact control pipeline
    used during training (EMA smoothing, target computation, wrench blending,
    J^T torque mapping with null-space compensation).

    Args:
        configs: Training configuration dict from WandB (with 'wrappers', 'primary',
                 'environment' sections). Loaded by reconstruct_config_from_wandb().
        real_config: Real robot config dict loaded from config.yaml.
        device: Torch device.
    """

    def __init__(self, configs: dict, real_config: dict, device: str = "cpu"):
        self.device = device
        self.configs = configs

        # Detect control mode from training config
        hybrid_cfg = configs['wrappers'].hybrid_control
        self.hybrid_enabled = hybrid_cfg.enabled
        self.ctrl_mode = getattr(configs['primary'], 'ctrl_mode', 'force_only')

        # Force size from ctrl_mode
        from configs.cfg_exts.ctrl_mode import get_force_size
        self.force_size = get_force_size(self.ctrl_mode) if self.hybrid_enabled else 0

        # Action dimensions
        if self.hybrid_enabled:
            self.action_dim = 2 * self.force_size + 6
        else:
            self.action_dim = 6

        # Control parameters from training config
        ctrl_cfg = configs['environment'].ctrl if hasattr(configs['environment'], 'ctrl') else configs['wrappers'].ctrl
        self.ema_factor = getattr(ctrl_cfg, 'ema_factor', 0.2)

        # Position/rotation thresholds and bounds
        self.pos_threshold = torch.tensor(
            ctrl_cfg.pos_action_threshold, device=device, dtype=torch.float32
        )
        self.rot_threshold = torch.tensor(
            ctrl_cfg.rot_action_threshold, device=device, dtype=torch.float32
        )
        self.pos_bounds = torch.tensor(
            ctrl_cfg.pos_action_bounds, device=device, dtype=torch.float32
        )

        # PD gains for pose control
        self.task_prop_gains = torch.tensor(
            ctrl_cfg.default_task_prop_gains, device=device, dtype=torch.float32
        )
        # Derive default_task_deriv_gains from prop gains (matching IsaacLab factory default)
        # kd = 2 * sqrt(kp) for critical damping
        if hasattr(ctrl_cfg, 'default_task_deriv_gains') and ctrl_cfg.default_task_deriv_gains is not None:
            self.task_deriv_gains = torch.tensor(
                ctrl_cfg.default_task_deriv_gains, device=device, dtype=torch.float32
            )
        else:
            self.task_deriv_gains = 2.0 * torch.sqrt(self.task_prop_gains)

        # Null-space parameters
        self.default_dof_pos = torch.tensor(
            ctrl_cfg.default_dof_pos_tensor, device=device, dtype=torch.float32
        )
        self.kp_null = getattr(ctrl_cfg, 'kp_null', 10.0)
        self.kd_null = getattr(ctrl_cfg, 'kd_null', 6.3246)

        # Hybrid-specific parameters (only if hybrid enabled)
        if self.hybrid_enabled:
            self.no_sel_ema = getattr(ctrl_cfg, 'no_sel_ema', True)
            self.apply_ema_force = getattr(ctrl_cfg, 'apply_ema_force', True)
            self.use_delta_force = getattr(ctrl_cfg, 'use_delta_force', False)
            self.async_z_bounds = getattr(ctrl_cfg, 'async_z_force_bounds', True)
            self.ema_mode = getattr(ctrl_cfg, 'ema_mode', 'action')

            # Force gains
            self.force_kp = torch.tensor(
                ctrl_cfg.default_task_force_gains, device=device, dtype=torch.float32
            )
            # Zero out torque gains based on ctrl_mode
            if self.ctrl_mode == "force_only":
                self.force_kp[3:] = 0.0
            elif self.ctrl_mode == "force_tz":
                self.force_kp[3:5] = 0.0

            # Force bounds and thresholds
            self.force_bounds = torch.tensor(
                ctrl_cfg.force_action_bounds, device=device, dtype=torch.float32
            )
            self.force_threshold = torch.tensor(
                ctrl_cfg.force_action_threshold, device=device, dtype=torch.float32
            )

            # Torque bounds/thresholds (if applicable)
            if self.ctrl_mode in ["force_tz", "force_torque"]:
                self.torque_bounds = torch.tensor(
                    ctrl_cfg.torque_action_bounds, device=device, dtype=torch.float32
                )
                self.torque_threshold = torch.tensor(
                    ctrl_cfg.torque_action_threshold, device=device, dtype=torch.float32
                )

            # PID parameters
            self.enable_force_derivative = getattr(ctrl_cfg, 'enable_force_derivative', False)
            self.enable_force_integral = getattr(ctrl_cfg, 'enable_force_integral', False)
            self.force_integral_clamp = getattr(ctrl_cfg, 'force_integral_clamp', 50.0)

            if self.enable_force_derivative:
                self.force_deriv_scale = getattr(ctrl_cfg, 'force_deriv_scale', 1.0)
                self.force_kd = self.force_deriv_scale * 2.0 * torch.sqrt(self.force_kp)
                if self.ctrl_mode == "force_only":
                    self.force_kd[3:] = 0.0
                elif self.ctrl_mode == "force_tz":
                    self.force_kd[3:5] = 0.0
            else:
                self.force_kd = None

            if self.enable_force_integral:
                self.force_ki = torch.tensor(
                    ctrl_cfg.default_task_force_integ_gains, device=device, dtype=torch.float32
                )
                if self.ctrl_mode == "force_only":
                    self.force_ki[3:] = 0.0
                elif self.ctrl_mode == "force_tz":
                    self.force_ki[3:5] = 0.0
            else:
                self.force_ki = None

        # Real robot control mode (effort, cartesian_impedance, position)
        self.robot_control_mode = real_config['ros2']['control_mode']

        # State (initialized by reset())
        self.ema_actions = None
        self.ema_task_wrench = None
        self.force_integral_error = None
        self.prev_force_error = None
        self.derivative_needs_init = True
        self.prev_sel_matrix = None

        print(f"[RealRobotController] mode={'hybrid' if self.hybrid_enabled else 'pose-only'}, "
              f"action_dim={self.action_dim}, ctrl_mode={self.ctrl_mode}")
        print(f"[RealRobotController] ema_factor={self.ema_factor}, "
              f"pos_threshold={self.pos_threshold.tolist()}")
        if self.hybrid_enabled:
            print(f"[RealRobotController] force_bounds={self.force_bounds.tolist()}, "
                  f"use_delta_force={self.use_delta_force}")

    def reset(self, ee_pos: torch.Tensor, goal_position: torch.Tensor):
        """Reset controller state for new episode.

        Back-calculates initial position actions to match sim reset behavior
        (see HybridForcePositionWrapper._reset_ema_actions).

        Args:
            ee_pos: [3] current EE position.
            goal_position: [3] fixed asset position (action frame origin).
        """
        self.ema_actions = torch.zeros(self.action_dim, device=self.device)
        self.ema_task_wrench = torch.zeros(6, device=self.device)

        if self.hybrid_enabled:
            self.force_integral_error = torch.zeros(6, device=self.device)
            self.prev_force_error = torch.zeros(6, device=self.device)
            self.derivative_needs_init = True
            self.prev_sel_matrix = torch.zeros(6, device=self.device)

        # Back-calculate initial position actions (matching sim reset behavior)
        # sim does: actions = (fingertip_pos - fixed_pos_action_frame) / pos_action_bounds
        init_pos_actions = (ee_pos - goal_position) / self.pos_bounds
        if self.hybrid_enabled:
            self.ema_actions[self.force_size:self.force_size+3] = init_pos_actions
        else:
            self.ema_actions[:3] = init_pos_actions

        print(f"[RealRobotController] Reset: init_pos_actions={init_pos_actions.tolist()}")

    def compute_action(
        self,
        raw_action: torch.Tensor,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        ee_linvel: torch.Tensor,
        ee_angvel: torch.Tensor,
        force_torque: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        jacobian: torch.Tensor,
        mass_matrix: torch.Tensor,
        goal_position: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute joint torques and intermediate targets from policy action.

        Args:
            raw_action: [action_dim] raw policy output (after sigmoid/tanh).
            ee_pos: [3] current EE position.
            ee_quat: [4] current EE orientation (w,x,y,z).
            ee_linvel: [3] current EE linear velocity.
            ee_angvel: [3] current EE angular velocity.
            force_torque: [6] current force/torque sensor readings.
            joint_pos: [7] current arm joint positions.
            joint_vel: [7] current arm joint velocities.
            jacobian: [6, 7] geometric Jacobian.
            mass_matrix: [7, 7] arm mass matrix.
            goal_position: [3] fixed asset position (action frame origin).

        Returns:
            Dict with:
                'joint_torques': [7] arm joint torques
                'ema_actions': [action_dim] EMA-smoothed actions (for prev_actions obs)
                'target_pos': [3] Cartesian position target
                'target_quat': [4] Cartesian orientation target
                'target_force': [6] Force target (hybrid only)
                'sel_matrix': [6] Selection matrix (hybrid only)
                'task_wrench': [6] Final task-space wrench
        """
        if self.hybrid_enabled:
            return self._compute_hybrid(
                raw_action, ee_pos, ee_quat, ee_linvel, ee_angvel,
                force_torque, joint_pos, joint_vel, jacobian, mass_matrix,
                goal_position,
            )
        else:
            return self._compute_pose_only(
                raw_action, ee_pos, ee_quat, ee_linvel, ee_angvel,
                joint_pos, joint_vel, jacobian, mass_matrix, goal_position,
            )

    # ========================================================================
    # POSE-ONLY control (sigma_idx == 0)
    # ========================================================================

    def _compute_pose_only(
        self,
        raw_action: torch.Tensor,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        ee_linvel: torch.Tensor,
        ee_angvel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        jacobian: torch.Tensor,
        mass_matrix: torch.Tensor,
        goal_position: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Pose-only control path (matches FactoryEnv._apply_action)."""
        # Step 1: EMA on all 6 actions uniformly
        self.ema_actions = (
            self.ema_factor * raw_action
            + (1 - self.ema_factor) * self.ema_actions
        )

        # Step 2: Position target
        pos_actions = self.ema_actions[:3] * self.pos_threshold
        pos_target = ee_pos + pos_actions
        delta_pos = pos_target - goal_position
        pos_error_clipped = torch.clamp(delta_pos, -self.pos_bounds, self.pos_bounds)
        target_pos = goal_position + pos_error_clipped

        # Step 3: Rotation target
        rot_actions = self.ema_actions[3:6] * self.rot_threshold
        angle = torch.norm(rot_actions)
        axis = rot_actions / (angle + 1e-6)

        if angle > 1e-6:
            rot_quat = quat_from_angle_axis(angle, axis)
        else:
            rot_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        target_quat = quat_mul(rot_quat, ee_quat)

        # Restrict to upright orientation (roll=pi, pitch=0)
        roll, pitch, yaw = get_euler_xyz(target_quat)
        roll = torch.tensor(math.pi, device=self.device, dtype=torch.float32)
        pitch = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        target_quat = quat_from_euler_xyz(roll, pitch, yaw)

        # Step 4: Compute pose wrench (PD control)
        task_wrench = compute_pose_task_wrench(
            ee_pos, ee_quat, ee_linvel, ee_angvel,
            target_pos, target_quat,
            self.task_prop_gains, self.task_deriv_gains,
        )

        # Step 5: J^T mapping + null-space compensation
        joint_torques = compute_joint_torques_from_wrench(
            task_wrench, jacobian, mass_matrix,
            joint_pos, joint_vel, self.default_dof_pos,
            self.kp_null, self.kd_null,
        )

        return {
            'joint_torques': joint_torques,
            'ema_actions': self.ema_actions.clone(),
            'target_pos': target_pos,
            'target_quat': target_quat,
            'target_force': torch.zeros(6, device=self.device),
            'sel_matrix': torch.zeros(6, device=self.device),
            'task_wrench': task_wrench,
        }

    # ========================================================================
    # HYBRID control (sigma_idx > 0)
    # ========================================================================

    def _compute_hybrid(
        self,
        raw_action: torch.Tensor,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        ee_linvel: torch.Tensor,
        ee_angvel: torch.Tensor,
        force_torque: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        jacobian: torch.Tensor,
        mass_matrix: torch.Tensor,
        goal_position: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Hybrid force-position control path.

        Matches HybridForcePositionWrapper._apply_ema_to_actions() +
        _compute_control_targets() + _wrapped_apply_action().
        """
        fs = self.force_size

        # ----------------------------------------------------------------
        # Step 1: Apply per-stream EMA (matching lines 556-637)
        # ----------------------------------------------------------------
        # Selection EMA
        sel_actions = raw_action[:fs]
        if self.no_sel_ema:
            self.ema_actions[:fs] = sel_actions
        else:
            self.ema_actions[:fs] = (
                self.ema_factor * sel_actions
                + (1 - self.ema_factor) * self.ema_actions[:fs]
            )

        # Pose + force EMA
        apply_ema_force_effective = self.apply_ema_force or (self.ema_mode == 'wrench')
        if apply_ema_force_effective:
            # EMA on both pose and force
            pf_start, pf_end = fs, 2*fs + 6
        else:
            # EMA on pose only, raw force
            pf_start, pf_end = fs, fs + 6
            self.ema_actions[fs+6:] = raw_action[fs+6:]

        self.ema_actions[pf_start:pf_end] = (
            self.ema_factor * raw_action[pf_start:pf_end]
            + (1 - self.ema_factor) * self.ema_actions[pf_start:pf_end]
        )

        # Select control actions (raw for wrench mode, ema for action mode)
        if self.ema_mode == 'wrench':
            control_actions = raw_action
        else:
            control_actions = self.ema_actions

        # ----------------------------------------------------------------
        # Step 2: Selection matrix (matching _compute_control_targets)
        # ----------------------------------------------------------------
        sel_matrix = torch.zeros(6, device=self.device)
        if self.ctrl_mode == "force_tz":
            sel_matrix[:3] = (self.ema_actions[:3] > 0.5).float()
            sel_matrix[3:5] = 0.0  # Rx, Ry always position
            sel_matrix[5] = (self.ema_actions[3] > 0.5).float()
        elif self.ctrl_mode == "force_torque":
            sel_matrix[:6] = (self.ema_actions[:6] > 0.5).float()
        else:  # force_only
            sel_matrix[:3] = (self.ema_actions[:3] > 0.5).float()
            sel_matrix[3:] = 0.0  # All rotation is position control

        # ----------------------------------------------------------------
        # Step 3: Position target
        # ----------------------------------------------------------------
        pos_actions = control_actions[fs:fs+3] * self.pos_threshold
        pos_target = ee_pos + pos_actions
        delta_pos = pos_target - goal_position
        pos_error_clipped = torch.clamp(delta_pos, -self.pos_bounds, self.pos_bounds)
        target_pos = goal_position + pos_error_clipped

        # ----------------------------------------------------------------
        # Step 4: Rotation target
        # ----------------------------------------------------------------
        rot_actions = control_actions[fs+3:fs+6] * self.rot_threshold
        angle = torch.norm(rot_actions)
        axis = rot_actions / (angle + 1e-6)

        if angle > 1e-6:
            rot_quat = quat_from_angle_axis(angle, axis)
        else:
            rot_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        target_quat = quat_mul(rot_quat, ee_quat)

        # Restrict to upright
        roll, pitch, yaw = get_euler_xyz(target_quat)
        roll = torch.tensor(math.pi, device=self.device, dtype=torch.float32)
        pitch = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        target_quat = quat_from_euler_xyz(roll, pitch, yaw)

        # ----------------------------------------------------------------
        # Step 5: Force target
        # ----------------------------------------------------------------
        force_actions = control_actions[fs+6:2*fs+6]
        target_force = torch.zeros(6, device=self.device)

        if self.use_delta_force:
            force_delta = force_actions[:3] * self.force_threshold
            target_force[:3] = torch.clamp(
                force_delta + force_torque[:3],
                -self.force_bounds, self.force_bounds,
            )
        else:
            target_force[:3] = force_actions[:3] * self.force_bounds

        # Async Z bounds: map Z force to [-bounds, 0] (downward only)
        if self.async_z_bounds:
            target_force[2] = (target_force[2] - self.force_bounds[2]) / 2.0

        # Handle torque based on ctrl_mode
        if self.ctrl_mode == "force_tz":
            if self.use_delta_force:
                tz_delta = force_actions[3] * self.torque_threshold[2] / self.force_threshold[0]
                target_force[5] = torch.clamp(
                    tz_delta + force_torque[5],
                    -self.torque_bounds[2], self.torque_bounds[2],
                )
            else:
                target_force[5] = force_actions[3] * self.torque_bounds[2]
        elif self.ctrl_mode == "force_torque":
            if self.use_delta_force:
                torque_delta = force_actions[3:] * self.torque_threshold / self.force_threshold[:3]
                target_force[3:] = torch.clamp(
                    torque_delta + force_torque[3:],
                    -self.torque_bounds, self.torque_bounds,
                )
            else:
                target_force[3:] = force_actions[3:] * self.torque_bounds

        # ----------------------------------------------------------------
        # Step 6: Integral state update (if enabled)
        # ----------------------------------------------------------------
        if self.enable_force_integral:
            force_error = target_force - force_torque
            force_ctrl_mask = sel_matrix > 0.5

            # Reset integral for axes that just switched to force control
            just_switched = (self.prev_sel_matrix <= 0.5) & (sel_matrix > 0.5)
            self.force_integral_error = torch.where(
                just_switched, torch.zeros_like(self.force_integral_error),
                self.force_integral_error,
            )

            # Accumulate only for force-controlled axes
            self.force_integral_error = torch.where(
                force_ctrl_mask,
                self.force_integral_error + force_error,
                self.force_integral_error,
            )

            # Anti-windup
            self.force_integral_error = torch.clamp(
                self.force_integral_error,
                -self.force_integral_clamp, self.force_integral_clamp,
            )

        # Update previous sel matrix
        self.prev_sel_matrix = sel_matrix.clone()

        # ----------------------------------------------------------------
        # Step 7: Derivative initialization (avoid spike on first step)
        # ----------------------------------------------------------------
        if self.enable_force_derivative and self.derivative_needs_init:
            self.prev_force_error = (target_force - force_torque).clone()
            self.derivative_needs_init = False

        # ----------------------------------------------------------------
        # Step 8: Compute wrenches
        # ----------------------------------------------------------------
        pose_wrench = compute_pose_task_wrench(
            ee_pos, ee_quat, ee_linvel, ee_angvel,
            target_pos, target_quat,
            self.task_prop_gains, self.task_deriv_gains,
        )

        force_wrench = compute_force_task_wrench(
            force_torque, target_force, self.force_kp,
            task_deriv_gains=self.force_kd,
            prev_force_error=self.prev_force_error if self.enable_force_derivative else None,
            task_integ_gains=self.force_ki,
            force_integral_error=self.force_integral_error if self.enable_force_integral else None,
            enable_derivative=self.enable_force_derivative,
            enable_integral=self.enable_force_integral,
        )

        # Update prev force error (only for force-controlled axes)
        if self.enable_force_derivative:
            current_force_error = target_force - force_torque
            force_ctrl_mask = sel_matrix > 0.5
            self.prev_force_error = torch.where(
                force_ctrl_mask, current_force_error, self.prev_force_error,
            )

        # ----------------------------------------------------------------
        # Step 9: Blend wrenches
        # ----------------------------------------------------------------
        task_wrench = (1.0 - sel_matrix) * pose_wrench + sel_matrix * force_wrench

        # Wrench EMA (if enabled)
        if self.ema_mode == 'wrench':
            task_wrench = (
                self.ema_factor * task_wrench
                + (1.0 - self.ema_factor) * self.ema_task_wrench
            )
            self.ema_task_wrench = task_wrench.clone()

        # ----------------------------------------------------------------
        # Step 10: Bounds constraint (zero wrench pushing further OOB)
        # ----------------------------------------------------------------
        delta_from_goal = ee_pos - goal_position
        for i in range(3):
            if delta_from_goal[i] <= -self.pos_bounds[i] and task_wrench[i] < 0:
                task_wrench[i] = 0.0
            if delta_from_goal[i] >= self.pos_bounds[i] and task_wrench[i] > 0:
                task_wrench[i] = 0.0

        # ----------------------------------------------------------------
        # Step 11: Rotation override based on ctrl_mode
        # ----------------------------------------------------------------
        if self.ctrl_mode == "force_only":
            task_wrench[3:] = pose_wrench[3:]
        elif self.ctrl_mode == "force_tz":
            task_wrench[3:5] = pose_wrench[3:5]  # Rx, Ry always position

        # ----------------------------------------------------------------
        # Step 12: J^T + null-space
        # ----------------------------------------------------------------
        joint_torques = compute_joint_torques_from_wrench(
            task_wrench, jacobian, mass_matrix,
            joint_pos, joint_vel, self.default_dof_pos,
            self.kp_null, self.kd_null,
        )

        return {
            'joint_torques': joint_torques,
            'ema_actions': self.ema_actions.clone(),
            'target_pos': target_pos,
            'target_quat': target_quat,
            'target_force': target_force,
            'sel_matrix': sel_matrix,
            'task_wrench': task_wrench,
        }
