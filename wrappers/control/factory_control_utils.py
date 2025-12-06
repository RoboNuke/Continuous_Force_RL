"""
Factory Control Utilities

Control functions extracted from envs/factory/factory_control.py for use in control wrappers.
These functions implement operational space control and hybrid force-position control.

Extracted functions:
- compute_pose_task_wrench (factory_control.py:70-106)
- compute_force_task_wrench (factory_control.py:108-116)
- compute_dof_torque_from_wrench (factory_control.py:120-159)
- get_pose_error (factory_control.py:161-202)
- _apply_task_space_gains (factory_control.py:245-267)
"""

import math
import torch

try:
    import isaacsim.core.utils.torch as torch_utils
except ImportError:
    try:
        import omni.isaac.core.utils.torch as torch_utils
    except ImportError:
        torch_utils = None

try:
    from isaaclab.utils.math import axis_angle_from_quat
except ImportError:
    try:
        from omni.isaac.lab.utils.math import axis_angle_from_quat
    except ImportError:
        try:
            from omni.isaac.lab.utils.math import axis_angle_from_quat
        except ImportError:
            axis_angle_from_quat = None


def compute_pose_task_wrench(
    cfg,
    dof_pos,
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    task_prop_gains,
    task_deriv_gains,
    device
):
    """Compute task-space wrench for pose control."""
    pos_error, axis_angle_error = get_pose_error(
        fingertip_midpoint_pos=fingertip_midpoint_pos,
        fingertip_midpoint_quat=fingertip_midpoint_quat,
        ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
        jacobian_type="geometric",
        rot_error_type="axis_angle",
    )
    delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

    # Set tau = k_p * task_pos_error - k_d * task_vel_error
    task_wrench_motion = _apply_task_space_gains(
        delta_fingertip_pose=delta_fingertip_pose,
        fingertip_midpoint_linvel=fingertip_midpoint_linvel,
        fingertip_midpoint_angvel=fingertip_midpoint_angvel,
        task_prop_gains=task_prop_gains,
        task_deriv_gains=task_deriv_gains,
    )
    return task_wrench_motion


def compute_force_task_wrench(
    cfg,
    dof_pos,
    eef_force,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    ctrl_target_force,
    task_gains,
    task_deriv_gains,
    device,
    # PID control parameters (optional)
    task_integ_gains=None,
    force_integral_error=None,
    enable_derivative=False,
    enable_integral=False,
):
    """Compute task-space wrench for force control with optional PID.

    Args:
        cfg: Environment configuration
        dof_pos: Joint positions
        eef_force: End-effector force/torque measurements
        fingertip_midpoint_linvel: Fingertip linear velocity
        fingertip_midpoint_angvel: Fingertip angular velocity
        ctrl_target_force: Target force/torque
        task_gains: Proportional gains (Kp)
        task_deriv_gains: Derivative gains (Kd) - derived from Kp as 2*sqrt(Kp)
        device: Torch device
        task_integ_gains: Integral gains (Ki) - optional, required if enable_integral=True
        force_integral_error: Accumulated integral error - optional, required if enable_integral=True
        enable_derivative: Enable D term (velocity damping)
        enable_integral: Enable I term
    """
    # Proportional term (always active)
    force_wrench = task_gains * (ctrl_target_force - eef_force)

    # Derivative term (velocity damping) - optional
    if enable_derivative:
        force_wrench[:, 0:3] += task_deriv_gains[:, 0:3] * (0.0 - fingertip_midpoint_linvel)
        force_wrench[:, 3:6] += task_deriv_gains[:, 3:6] * (0.0 - fingertip_midpoint_angvel)

    # Integral term - optional
    if enable_integral and task_integ_gains is not None and force_integral_error is not None:
        force_wrench += task_integ_gains * force_integral_error

    return force_wrench


def compute_dof_torque_from_wrench(
    cfg,
    dof_pos,
    dof_vel,
    task_wrench,
    jacobian,
    arm_mass_matrix,
    device,
):
    """Compute joint torques for given task wrench with null space compensation."""
    num_envs = cfg.scene.num_envs
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)

    # Set tau = J^T * tau, i.e., map tau into joint space as desired
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    dof_torque[:, 0:7] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # Null space computation for natural arm posture
    arm_mass_matrix_inv = torch.inverse(arm_mass_matrix)
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    arm_mass_matrix_task = torch.inverse(
        jacobian @ torch.inverse(arm_mass_matrix) @ jacobian_T
    )
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv

    default_dof_pos_tensor = torch.tensor(cfg.ctrl.default_dof_pos_tensor, device=device).repeat((num_envs, 1))

    # Nullspace computation
    distance_to_default_dof_pos = default_dof_pos_tensor - dof_pos[:, :7]
    distance_to_default_dof_pos = (distance_to_default_dof_pos + math.pi) % (
        2 * math.pi
    ) - math.pi  # normalize to [-pi, pi]

    u_null = cfg.ctrl.kd_null * -dof_vel[:, :7] + cfg.ctrl.kp_null * distance_to_default_dof_pos
    u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
    torque_null = (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(jacobian, 1, 2) @ j_eef_inv) @ u_null
    #print(f"Null-space torques: {torque_null[0].squeeze()}")
    dof_torque[:, 0:7] += torque_null.squeeze(-1)

    # Clamp torques to safe limits
    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)
    return dof_torque, task_wrench


def get_pose_error(
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    jacobian_type,
    rot_error_type,
):
    """Compute task-space error between target and current fingertip pose."""
    if torch_utils is None:
        raise ImportError("torch_utils not available. Please ensure Isaac Sim is properly installed.")

    if axis_angle_from_quat is None:
        raise ImportError("axis_angle_from_quat not available. Please ensure Isaac Lab is properly installed.")

    # Compute pos error
    pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos

    # Compute rot error
    if jacobian_type == "geometric":
        # Check for shortest path using quaternion dot product
        quat_dot = (ctrl_target_fingertip_midpoint_quat * fingertip_midpoint_quat).sum(dim=1, keepdim=True)
        ctrl_target_fingertip_midpoint_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0, ctrl_target_fingertip_midpoint_quat, -ctrl_target_fingertip_midpoint_quat
        )

        fingertip_midpoint_quat_norm = torch_utils.quat_mul(
            fingertip_midpoint_quat, torch_utils.quat_conjugate(fingertip_midpoint_quat)
        )[:, 0]  # scalar component

        fingertip_midpoint_quat_inv = torch_utils.quat_conjugate(
            fingertip_midpoint_quat
        ) / fingertip_midpoint_quat_norm.unsqueeze(-1)

        quat_error = torch_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_midpoint_quat_inv)

        # Convert to axis-angle error
        axis_angle_error = axis_angle_from_quat(quat_error)

    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        return pos_error, axis_angle_error


def _apply_task_space_gains(
    delta_fingertip_pose,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    task_prop_gains,
    task_deriv_gains
):
    """Apply PD gains to task-space error."""
    task_wrench = torch.zeros_like(delta_fingertip_pose)

    # Apply gains to linear error components
    lin_error = delta_fingertip_pose[:, 0:3]
    task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * lin_error + task_deriv_gains[:, 0:3] * (
        0.0 - fingertip_midpoint_linvel
    )

    # Apply gains to rotational error components
    rot_error = delta_fingertip_pose[:, 3:6]
    task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * rot_error + task_deriv_gains[:, 3:6] * (
        0.0 - fingertip_midpoint_angvel
    )

    return task_wrench