"""
Hybrid Force-Position Control Wrapper

This wrapper implements hybrid force-position control where the robot can switch between
position control and force control on different axes based on a selection matrix.

Extracted from: hybrid_control_action_wrapper.py
Control law: tau = J^T(S*k_f*(f_d - f) + (1-S) * k_p * (x_d - x))

Required functions from factory_control.py:
- compute_pose_task_wrench
- compute_force_task_wrench
- compute_dof_torque_from_wrench
"""

import torch
import gymnasium as gym
import numpy as np
from .factory_control_utils import compute_pose_task_wrench, compute_force_task_wrench, compute_dof_torque_from_wrench

try:
    import isaacsim.core.utils.torch as torch_utils
except ImportError:
    try:
        import omni.isaac.core.utils.torch as torch_utils
    except ImportError:
        torch_utils = None


class HybridForcePositionWrapper(gym.Wrapper):
    """
    Wrapper implementing hybrid force-position control for factory environments.

    Features:
    - Selection matrix for force/position control per axis
    - Force goal calculation with bounds checking
    - Multiple reward computation strategies
    - Out-of-bounds position handling
    - Action space management for hybrid control
    """

    def __init__(self, env, ctrl_torque=False, reward_type="simp"):
        """
        Initialize hybrid force-position control wrapper.

        Args:
            env: Base environment to wrap
            ctrl_torque: Whether to control torques (6DOF) or just forces (3DOF)
            reward_type: Reward computation strategy
                - "simp": Simple force activity reward
                - "dirs": Direction-specific force rewards
                - "delta": Penalize selection matrix changes
                - "base": No hybrid control rewards
                - "pos_simp": Position-focused force rewards
                - "wrench_norm": Low wrench magnitude reward
        """
        super().__init__(env)

        if torch_utils is None:
            raise ImportError("torch_utils not available. Please ensure Isaac Sim is properly installed.")

        self.ctrl_torque = ctrl_torque
        self.force_size = 6 if ctrl_torque else 3
        self.reward_type = reward_type

        # Update action space to include selection matrix + position + force
        original_action_space = 6  # Original position + rotation actions
        self.action_space_size = self.force_size + original_action_space + self.force_size

        # Update environment action space
        if hasattr(self.unwrapped, 'cfg'):
            self.unwrapped.cfg.action_space = self.action_space_size
            if hasattr(self.unwrapped, '_configure_gym_env_spaces'):
                self.unwrapped._configure_gym_env_spaces()

        # Initialize state variables
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # Selection matrix and force actions
        self.sel_matrix = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_action = torch.zeros((self.num_envs, 6), device=self.device)

        # Force gains (from config or default)
        force_task_gains = getattr(env.unwrapped.cfg.ctrl, 'default_task_force_gains',
                                 [0.1, 0.1, 0.1, 0.001, 0.001, 0.001])
        self.kp = torch.tensor(force_task_gains, device=self.device).repeat((self.num_envs, 1))

        # Zero out torque gains if not controlling torques
        if not ctrl_torque:
            self.kp[:, 3:] = 0.0

        # Store original methods
        self._original_pre_physics_step = None
        self._original_apply_action = None
        self._original_update_rew_buf = None

        # Reward tracking for delta strategy
        if reward_type == "delta":
            self._old_sel_matrix = torch.zeros_like(self.sel_matrix)

        # Initialize wrapper
        self._wrapper_initialized = False
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def _initialize_wrapper(self):
        """Initialize wrapper by overriding environment methods."""
        if self._wrapper_initialized:
            return

        # Check for required force-torque data
        if not hasattr(self.unwrapped, 'robot_force_torque'):
            raise ValueError("Hybrid force-position control requires force-torque sensor data. "
                           "Please apply ForceTorqueWrapper first.")

        # Store and override methods
        if hasattr(self.unwrapped, '_pre_physics_step'):
            self._original_pre_physics_step = self.unwrapped._pre_physics_step
            self.unwrapped._pre_physics_step = self._wrapped_pre_physics_step

        if hasattr(self.unwrapped, '_apply_action'):
            self._original_apply_action = self.unwrapped._apply_action
            self.unwrapped._apply_action = self._wrapped_apply_action

        if hasattr(self.unwrapped, '_update_rew_buf') and self.reward_type != "base":
            self._original_update_rew_buf = self.unwrapped._update_rew_buf
            self.unwrapped._update_rew_buf = self._wrapped_update_rew_buf

        self._wrapper_initialized = True

    def _wrapped_pre_physics_step(self, action):
        """Process actions and update selection matrix and force goals."""
        # Extract selection matrix from first part of action
        self.sel_matrix[:, :self.force_size] = torch.where(action[:, :self.force_size] > 0.5, 1.0, 0.0)

        # Log selection matrix for debugging
        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['Hybrid Controller / Force Control X'] = action[:, 0]
            self.unwrapped.extras['Hybrid Controller / Force Control Y'] = action[:, 1]
            self.unwrapped.extras['Hybrid Controller / Force Control Z'] = action[:, 2]

            if self.force_size > 3:
                self.unwrapped.extras['Hybrid Controller / Force Control RX'] = action[:, 3]
                self.unwrapped.extras['Hybrid Controller / Force Control RY'] = action[:, 4]
                self.unwrapped.extras['Hybrid Controller / Force Control RZ'] = action[:, 5]

        # Handle reset environments
        env_ids = self.unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0 and hasattr(self.unwrapped, '_reset_buffers'):
            self.unwrapped._reset_buffers(env_ids)

        # Store previous actions
        if hasattr(self.unwrapped, 'actions'):
            self.unwrapped.prev_action = self.unwrapped.actions.clone()

        # Apply EMA smoothing (configurable)
        ema_factor = getattr(self.unwrapped.cfg.ctrl, 'ema_factor', 0.2)
        no_sel_ema = getattr(self.unwrapped.cfg.ctrl, 'no_sel_ema', True)

        if no_sel_ema:
            # Apply EMA to everything except selection matrix
            self.unwrapped.actions[:, self.force_size:] = (
                ema_factor * action[:, self.force_size:].clone().to(self.device) +
                (1 - ema_factor) * self.unwrapped.actions[:, self.force_size:]
            )
            self.unwrapped.actions[:, :self.force_size] = action[:, :self.force_size]
        else:
            # Apply EMA to all actions
            self.unwrapped.actions = (
                ema_factor * action.clone().to(self.device) +
                (1 - ema_factor) * self.unwrapped.actions
            )

        # Calculate control targets
        self._calc_ctrl_pos(min_idx=self.force_size, max_idx=self.force_size + 3)
        self._calc_ctrl_quat(min_idx=self.force_size + 3, max_idx=self.force_size + 6)
        self._calc_ctrl_force()

        # Update intermediate values
        if hasattr(self.unwrapped, '_compute_intermediate_values'):
            self.unwrapped._compute_intermediate_values(dt=self.unwrapped.physics_dt)

        # Update smoothness metrics
        if hasattr(self.unwrapped, 'ep_ssv'):
            self.unwrapped.ep_ssv += torch.linalg.norm(self.unwrapped.ee_linvel_fd, axis=1)

        if hasattr(self.unwrapped, 'ep_sum_force'):
            self.unwrapped.ep_sum_force += torch.linalg.norm(self.unwrapped.robot_force_torque[:, :3], axis=1)
            self.unwrapped.ep_sum_torque += torch.linalg.norm(self.unwrapped.robot_force_torque[:, 3:], axis=1)
            self.unwrapped.ep_max_force = torch.max(
                self.unwrapped.ep_max_force,
                torch.linalg.norm(self.unwrapped.robot_force_torque[:, :3], axis=1)
            )
            self.unwrapped.ep_max_torque = torch.max(
                self.unwrapped.ep_max_torque,
                torch.linalg.norm(self.unwrapped.robot_force_torque[:, 3:])
            )

    def _calc_ctrl_pos(self, min_idx=0, max_idx=3):
        """Calculate position control targets."""
        pos_actions = self.unwrapped.actions[:, min_idx:max_idx] * self.unwrapped.pos_threshold
        self.unwrapped.ctrl_target_fingertip_midpoint_pos = self.unwrapped.fingertip_midpoint_pos + pos_actions

        # Enforce position bounds
        delta_pos = self.unwrapped.ctrl_target_fingertip_midpoint_pos - self.unwrapped.fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos,
            -self.unwrapped.cfg.ctrl.pos_action_bounds[0],
            self.unwrapped.cfg.ctrl.pos_action_bounds[1]
        )
        self.unwrapped.ctrl_target_fingertip_midpoint_pos = self.unwrapped.fixed_pos_action_frame + pos_error_clipped

    def _calc_ctrl_quat(self, min_idx=3, max_idx=6):
        """Calculate quaternion control targets."""
        rot_actions = self.unwrapped.actions[:, min_idx:max_idx]

        # Handle unidirectional rotation if configured
        if getattr(self.unwrapped.cfg_task, 'unidirectional_rot', False):
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5

        rot_actions = rot_actions * self.unwrapped.rot_threshold

        # Convert to quaternion
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.unwrapped.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(
            rot_actions_quat, self.unwrapped.fingertip_midpoint_quat
        )

        # Restrict to upright orientation
        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.unwrapped.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Roll
        target_euler_xyz[:, 1] = 0.0      # Pitch

        self.unwrapped.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

    def _calc_ctrl_force(self):
        """Calculate force control targets."""
        min_idx = self.force_size + 6
        max_idx = min_idx + self.force_size

        # Extract force actions and scale by threshold
        force_threshold = getattr(self.unwrapped.cfg.ctrl, 'force_action_threshold', [10, 10, 10])
        self.force_action[:, :self.force_size] = (
            self.unwrapped.actions[:, min_idx:max_idx] * force_threshold[0]
        )

        # Add current force to get absolute target
        force_bounds = getattr(self.unwrapped.cfg.ctrl, 'force_action_bounds', [50, 50, 50])
        self.force_action[:, :3] = torch.clip(
            self.force_action[:, :3] + self.unwrapped.robot_force_torque[:, :3],
            -force_bounds[0], force_bounds[0]
        )

        # Handle torque if enabled
        if self.force_size > 3:
            torque_threshold = getattr(self.unwrapped.cfg.ctrl, 'torque_action_threshold', [0.1, 0.1, 0.1])
            torque_bounds = getattr(self.unwrapped.cfg.ctrl, 'torque_action_bounds', [0.5, 0.5, 0.5])

            self.force_action[:, 3:] *= torque_threshold[0] / force_threshold[0]
            self.force_action[:, 3:] = torch.clip(
                self.force_action[:, 3:] + self.unwrapped.robot_force_torque[:, 3:],
                -torque_bounds[0], torque_bounds[0]
            )

    def _get_target_out_of_bounds(self):
        """Check if fingertip target is out of position bounds."""
        delta = self.unwrapped.fingertip_midpoint_pos - self.unwrapped.fixed_pos_action_frame
        pos_bounds = self.unwrapped.cfg.ctrl.pos_action_bounds

        out_of_bounds = torch.logical_or(
            delta <= -pos_bounds[0],
            delta >= pos_bounds[1]
        )
        return out_of_bounds

    def _wrapped_apply_action(self):
        """Apply hybrid force-position control."""
        # Get current yaw for success checking
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.unwrapped.fingertip_midpoint_quat)
        self.unwrapped.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Update intermediate values if needed
        if self.unwrapped.last_update_timestamp < self.unwrapped._robot._data._sim_timestamp:
            self.unwrapped._compute_intermediate_values(dt=self.unwrapped.physics_dt)

        self.unwrapped.ctrl_target_gripper_dof_pos = 0.0

        # Check for out-of-bounds positions
        out_of_bounds = self._get_target_out_of_bounds()

        # Compute pose wrench
        pose_wrench = compute_pose_task_wrench(
            cfg=self.unwrapped.cfg,
            dof_pos=self.unwrapped.joint_pos,
            fingertip_midpoint_pos=self.unwrapped.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.unwrapped.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.unwrapped.ee_linvel_fd,
            fingertip_midpoint_angvel=self.unwrapped.ee_angvel_fd,
            ctrl_target_fingertip_midpoint_pos=self.unwrapped.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.unwrapped.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.unwrapped.task_prop_gains,
            task_deriv_gains=self.unwrapped.task_deriv_gains,
            device=self.unwrapped.device
        )

        # Compute force wrench
        force_wrench = compute_force_task_wrench(
            cfg=self.unwrapped.cfg,
            dof_pos=self.unwrapped.joint_pos,
            eef_force=self.unwrapped.robot_force_torque,
            ctrl_target_force=self.force_action,
            task_gains=self.kp,
            device=self.unwrapped.device
        )

        # Zero force wrench if position is out of bounds
        force_wrench[:, :3][out_of_bounds] = 0.0

        # Combine wrenches using selection matrix
        task_wrench = (1 - self.sel_matrix) * pose_wrench + self.sel_matrix * force_wrench

        # For torque, always use position control
        if self.force_size > 3:
            task_wrench[:, 3:] = pose_wrench[:, 3:]

        # Compute joint torques
        self.unwrapped.joint_torque, task_wrench = compute_dof_torque_from_wrench(
            cfg=self.unwrapped.cfg,
            dof_pos=self.unwrapped.joint_pos,
            dof_vel=self.unwrapped.joint_vel_fd,
            task_wrench=task_wrench,
            jacobian=self.unwrapped.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.unwrapped.arm_mass_matrix,
            device=self.unwrapped.device
        )

        # Set gripper targets
        self.unwrapped.ctrl_target_joint_pos[:, 7:9] = self.unwrapped.ctrl_target_gripper_dof_pos
        self.unwrapped.joint_torque[:, 7:9] = 0.0

        # Apply torques
        self.unwrapped._robot.set_joint_position_target(self.unwrapped.ctrl_target_joint_pos)
        self.unwrapped._robot.set_joint_effort_target(self.unwrapped.joint_torque)

    def _wrapped_update_rew_buf(self, curr_successes):
        """Update reward buffer with hybrid control rewards."""
        # Get base rewards
        rew_buf = self._original_update_rew_buf(curr_successes) if self._original_update_rew_buf else torch.zeros_like(curr_successes).float()

        # Add hybrid control specific rewards
        if self.reward_type == "simp":
            rew_buf += self._simple_force_reward()
        elif self.reward_type == "dirs":
            rew_buf += self._directional_force_reward()
        elif self.reward_type == "delta":
            rew_buf += self._delta_selection_reward()
        elif self.reward_type == "pos_simp":
            rew_buf += self._position_simple_reward()
        elif self.reward_type == "wrench_norm":
            rew_buf += self._low_wrench_reward()

        return rew_buf

    def _simple_force_reward(self):
        """Simple force activity reward."""
        force_threshold = getattr(self.unwrapped.cfg_task, 'force_active_threshold', 1.0)
        good_force_reward = getattr(self.unwrapped.cfg_task, 'good_force_cmd_rew', 0.1)
        bad_force_reward = getattr(self.unwrapped.cfg_task, 'bad_force_cmd_rew', -0.1)

        active_force = torch.abs(self.unwrapped.robot_force_torque[:, :self.force_size]) > force_threshold
        force_ctrl = self.sel_matrix[:, :self.force_size].bool()

        good_force_cmd = torch.logical_and(torch.any(active_force, dim=1), torch.any(force_ctrl, dim=1))
        bad_force_cmd = torch.logical_and(torch.all(~active_force, dim=1), torch.any(force_ctrl, dim=1))

        sel_rew = good_force_reward * good_force_cmd + bad_force_reward * bad_force_cmd

        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['Reward / Selection Matrix'] = sel_rew

        return sel_rew

    def _directional_force_reward(self):
        """Direction-specific force reward."""
        force_threshold = getattr(self.unwrapped.cfg_task, 'force_active_threshold', 1.0)
        good_force_reward = getattr(self.unwrapped.cfg_task, 'good_force_cmd_rew', 0.1)
        bad_force_reward = getattr(self.unwrapped.cfg_task, 'bad_force_cmd_rew', -0.1)

        active_force = torch.abs(self.unwrapped.robot_force_torque[:, :self.force_size]) > force_threshold
        if self.force_size > 3:
            torque_threshold = getattr(self.unwrapped.cfg_task, 'torque_active_threshold', 0.1)
            active_force[:, 3:] = torch.abs(self.unwrapped.robot_force_torque[:, 3:]) > torque_threshold

        force_ctrl = self.sel_matrix[:, :self.force_size].bool()

        good_dims = torch.logical_and(force_ctrl, active_force) | torch.logical_and(~force_ctrl, ~active_force)
        bad_dims = torch.logical_and(force_ctrl, ~active_force) | torch.logical_and(~force_ctrl, active_force)

        good_rew = good_force_reward * torch.sum(good_dims, dim=1)
        bad_rew = bad_force_reward * torch.sum(bad_dims, dim=1)

        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['Reward / Selection Matrix'] = good_rew + bad_rew

        return good_rew + bad_rew

    def _delta_selection_reward(self):
        """Penalize changes in selection matrix."""
        bad_force_reward = getattr(self.unwrapped.cfg_task, 'bad_force_cmd_rew', -0.1)
        sel_rew = torch.sum(torch.abs(self.sel_matrix - self._old_sel_matrix), dim=1) * bad_force_reward
        self._old_sel_matrix = self.sel_matrix.clone()

        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['Reward / Selection Matrix'] = sel_rew

        return sel_rew

    def _position_simple_reward(self):
        """Position-focused simple force reward."""
        force_threshold = getattr(self.unwrapped.cfg_task, 'force_active_threshold', 1.0)
        good_force_reward = getattr(self.unwrapped.cfg_task, 'good_force_cmd_rew', 0.1)

        active_force = torch.abs(self.unwrapped.robot_force_torque[:, :self.force_size]) > force_threshold
        force_ctrl = self.sel_matrix[:, :self.force_size].bool()

        good_force_cmd = torch.logical_and(torch.any(active_force, dim=1), torch.any(force_ctrl, dim=1))
        sel_rew = good_force_reward * good_force_cmd

        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['Reward / Selection Matrix'] = sel_rew

        return sel_rew

    def _low_wrench_reward(self):
        """Reward for low wrench magnitude."""
        wrench_scale = getattr(self.unwrapped.cfg_task, 'wrench_norm_scale', 0.01)
        wrench_norm = self.unwrapped.actions[:, self.force_size:].norm(dim=-1)
        return -wrench_norm * wrench_scale

    def step(self, action):
        """Step environment and ensure wrapper is initialized."""
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()
        return super().step(action)

    def reset(self, **kwargs):
        """Reset environment and ensure wrapper is initialized."""
        obs, info = super().reset(**kwargs)
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()
        return obs, info