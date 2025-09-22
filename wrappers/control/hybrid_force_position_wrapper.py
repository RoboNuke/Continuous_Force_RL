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
from .hybrid_control_cfg import HybridCtrlCfg, HybridTaskCfg

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

    def __init__(self, env, ctrl_torque=False, reward_type="simp", ctrl_cfg=None, task_cfg=None):
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
            ctrl_cfg: HybridCtrlCfg instance. Must be provided - no defaults.
            task_cfg: HybridTaskCfg instance. Must be provided - no defaults.
        """
        # Store original action space size before modification
        self._original_action_size = getattr(env.unwrapped.cfg, 'action_space', 6)

        super().__init__(env)

        # Store control configuration for action space calculation
        self.ctrl_torque = ctrl_torque

        # Require explicit configuration - no fallback defaults
        if ctrl_cfg is None:
            raise ValueError(
                "ctrl_cfg cannot be None. Please provide a HybridCtrlCfg instance with "
                "explicit configuration. Example: ctrl_cfg = HybridCtrlCfg(ema_factor=0.2, ...)"
            )
        if task_cfg is None:
            raise ValueError(
                "task_cfg cannot be None. Please provide a HybridTaskCfg instance with "
                "explicit configuration. Example: task_cfg = HybridTaskCfg(force_active_threshold=0.1, ...)"
            )

        if torch_utils is None:
            raise ImportError(
                "torch_utils not available. This wrapper requires Isaac Sim/Lab torch utilities. "
                "Please ensure Isaac Sim is properly installed and accessible."
            )
        self.torch_utils = torch_utils

        self.ctrl_torque = ctrl_torque
        self.force_size = 6 if ctrl_torque else 3
        self.reward_type = reward_type

        # Store explicit configuration (no fallbacks)
        self.ctrl_cfg = ctrl_cfg
        self.task_cfg = task_cfg

        # Cache config values for faster access
        self.ema_factor = self.ctrl_cfg.ema_factor
        self.no_sel_ema = self.ctrl_cfg.no_sel_ema
        self.target_init_mode = self.ctrl_cfg.target_init_mode

        # Update action space to include selection matrix + position + force
        original_action_space = 6  # Original position + rotation actions
        self.action_space_size = 2*self.force_size + original_action_space

        # Update environment action space
        if hasattr(self.unwrapped, 'cfg'):
            self.unwrapped.cfg.action_space = self.action_space_size
            if hasattr(self.unwrapped, '_configure_gym_env_spaces'):
                self.unwrapped._configure_gym_env_spaces()

        # Initialize state variables
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # Current goals from actions (computed each step)
        self.sel_goal = torch.zeros((self.num_envs, 6), device=self.device)
        self.pose_goal = torch.zeros((self.num_envs, 7), device=self.device)  # pos(3) + quat(4)
        self.force_goal = torch.zeros((self.num_envs, 6), device=self.device)

        # EMA-filtered targets (used for control)
        self.target_selection = torch.zeros((self.num_envs, 6), device=self.device)
        self.target_pose = torch.zeros((self.num_envs, 7), device=self.device)  # pos(3) + quat(4)
        self.target_force = torch.zeros((self.num_envs, 6), device=self.device)

        # Selection matrix for control (derived from target_selection)
        self.sel_matrix = torch.zeros((self.num_envs, 6), device=self.device)

        # Force gains (must be explicitly configured)
        if self.ctrl_cfg.default_task_force_gains is None:
            raise ValueError(
                "ctrl_cfg.default_task_force_gains cannot be None. Please provide explicit force gains "
                "in your HybridCtrlCfg. Example: default_task_force_gains=[0.1, 0.1, 0.1, 0.001, 0.001, 0.001]"
            )
        self.kp = torch.tensor(self.ctrl_cfg.default_task_force_gains, device=self.device).repeat((self.num_envs, 1))

        # Zero out torque gains if not controlling torques
        if not ctrl_torque:
            self.kp[:, 3:] = 0.0

        # Flag to track if targets have been initialized
        self._targets_initialized = False

        # Initialize targets to zeros initially (will be properly set after first action)
        if self.target_init_mode == "zero":
            self._targets_initialized = True

        # Store original methods
        self._original_pre_physics_step = None
        self._original_apply_action = None
        self._original_update_rew_buf = None

        # Reward tracking for delta strategy
        if reward_type == "delta":
            self._old_sel_matrix = torch.zeros_like(self.sel_matrix)

        # Calculate new action space size based on control configuration
        # Action space: 6 (pose) + 6 (force) + 0/6 (torque selection if ctrl_torque)
        self._new_action_size = 12 if not ctrl_torque else 18

        # Update observation/state dimensions for action space change
        self._update_observation_dimensions()

        # Initialize wrapper
        self._wrapper_initialized = False
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def _update_observation_dimensions(self):
        """Update observation and state space dimensions for action space change."""
        action_diff = self._new_action_size - self._original_action_size

        if hasattr(self.unwrapped.cfg, 'observation_space'):
            old_obs_space = self.unwrapped.cfg.observation_space
            self.unwrapped.cfg.observation_space += action_diff
            print(f"[HYBRID]: Updated observation_space by {action_diff}: {old_obs_space} -> {self.unwrapped.cfg.observation_space}")

        if hasattr(self.unwrapped.cfg, 'state_space'):
            old_state_space = self.unwrapped.cfg.state_space
            self.unwrapped.cfg.state_space += action_diff
            print(f"[HYBRID]: Updated state_space by {action_diff}: {old_state_space} -> {self.unwrapped.cfg.state_space}")

        if hasattr(self.unwrapped.cfg, 'action_space'):
            old_action_space = self.unwrapped.cfg.action_space
            self.unwrapped.cfg.action_space = self._new_action_size
            print(f"[HYBRID]: Updated action_space: {old_action_space} -> {self._new_action_size} (ctrl_torque={self.ctrl_torque})")

        self._update_gym_spaces()

    def _update_gym_spaces(self):
        """
        Update gymnasium environment spaces if needed.

        Attempts to reconfigure the environment's observation and action spaces
        to account for the additional force-torque sensor dimensions. Silently
        continues if reconfiguration fails to maintain compatibility.
        """
        if hasattr(self.unwrapped, '_configure_gym_env_spaces'):
            try:
                self.unwrapped._configure_gym_env_spaces()
            except Exception:
                # Silently continue if reconfiguration fails
                RuntimeError("Unable to configure gym env spaces in hybrid controller")
    def _initialize_wrapper(self):
        """Initialize wrapper by overriding environment methods."""
        if self._wrapper_initialized:
            return

        # Check for required force-torque data
        # Search through the wrapper chain to find robot_force_torque
        force_torque_found = False
        current_env = self.env

        while current_env is not None:
            if hasattr(current_env, 'has_force_torque_sensor'):
                force_torque_found = current_env.has_force_torque_sensor
                break
            # Move up the wrapper chain
            if hasattr(current_env, 'env'):
                current_env = current_env.env
            elif hasattr(current_env, 'unwrapped'):
                current_env = current_env.unwrapped
                break
            else:
                break

        if not force_torque_found:
            raise ValueError(
                "ERROR: Hybrid force-position control requires force-torque sensor data.\n"
                "\n"
                "SOLUTION: Enable the force-torque sensor in your configuration file:\n"
                "  wrappers:\n"
                "    force_torque_sensor:\n"
                "      enabled: true\n"
                "\n"
                "The ForceTorqueWrapper must be enabled and applied before HybridForcePositionWrapper.\n"
                "This is a required dependency that must be explicitly configured."
            )

        # Store reference to the environment that has force-torque data
        self._force_torque_env = current_env

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

    @property
    def robot_force_torque(self):
        """Access force-torque data from the correct environment in the wrapper chain."""
        return self._force_torque_env.robot_force_torque


    def _wrapped_pre_physics_step(self, action):
        """Process actions through goal->target->control flow."""
        # Handle reset environments first
        env_ids = self.unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            if hasattr(self.unwrapped, '_reset_buffers'):
                self.unwrapped._reset_buffers(env_ids)
            self._reset_targets(env_ids)

        # Store previous actions for backward compatibility
        if hasattr(self.unwrapped, 'actions'):
            self.unwrapped.prev_action = self.unwrapped.actions.clone()

        # Step 1: Extract goals from current action
        self._extract_goals_from_action(action)

        # Step 2: Apply EMA filtering to get targets
        self._update_targets_with_ema()

        # Step 3: Set control targets from filtered targets
        self._set_control_targets_from_targets()

        # Log selection matrix for debugging
        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['Hybrid Controller / Force Control X'] = self.target_selection[:, 0]
            self.unwrapped.extras['Hybrid Controller / Force Control Y'] = self.target_selection[:, 1]
            self.unwrapped.extras['Hybrid Controller / Force Control Z'] = self.target_selection[:, 2]

            if self.force_size > 3:
                self.unwrapped.extras['Hybrid Controller / Force Control RX'] = self.target_selection[:, 3]
                self.unwrapped.extras['Hybrid Controller / Force Control RY'] = self.target_selection[:, 4]
                self.unwrapped.extras['Hybrid Controller / Force Control RZ'] = self.target_selection[:, 5]

        # Update intermediate values
        if hasattr(self.unwrapped, '_compute_intermediate_values'):
            self.unwrapped._compute_intermediate_values(dt=self.unwrapped.physics_dt)

    def _reset_targets(self, env_ids):
        """Reset targets for given environment IDs."""
        if self.target_init_mode == "zero":
            # Initialize to zeros
            self.target_selection[env_ids] = 0.0
            self.target_pose[env_ids] = 0.0
            self.target_force[env_ids] = 0.0
        # For "first_goal" mode, targets will be set from first goal after reset

        self._targets_initialized = True

    def _extract_goals_from_action(self, action):
        """Extract pose, force, and selection goals from action."""
        # Extract selection goal (raw values before threshold)
        self.sel_goal[:, :self.force_size] = action[:, :self.force_size]

        # Store actions for goal calculation
        self.unwrapped.actions = action.clone().to(self.device)

        # Extract pose goal using existing calculation methods
        self._calc_pose_goal()

        # Extract force goal using existing calculation methods
        self._calc_force_goal()

    def _calc_pose_goal(self):
        """Calculate pose goal from action."""
        # Position goal
        pos_actions = self.unwrapped.actions[:, self.force_size:self.force_size+3] * self.unwrapped.pos_threshold
        pos_goal = self.unwrapped.fingertip_midpoint_pos + pos_actions

        # Enforce position bounds for goal - require explicit configuration
        delta_pos = pos_goal - self.unwrapped.fixed_pos_action_frame
        if self.ctrl_cfg.pos_action_bounds is None:
            if not hasattr(self.unwrapped.ctrl, 'pos_action_bounds'):
                raise ValueError(
                    "Position action bounds not configured. Please set ctrl_cfg.pos_action_bounds "
                    "or ensure environment has ctrl.pos_action_bounds. Example: pos_action_bounds=[0.05, 0.05, 0.05]"
                )
            pos_bounds = self.unwrapped.ctrl.pos_action_bounds
        else:
            pos_bounds = self.ctrl_cfg.pos_action_bounds
        pos_error_clipped = torch.clip(delta_pos, -pos_bounds[0], pos_bounds[1])
        self.pose_goal[:, :3] = self.unwrapped.fixed_pos_action_frame + pos_error_clipped

        # Quaternion goal
        rot_actions = self.unwrapped.actions[:, self.force_size+3:self.force_size+6]

        # Handle unidirectional rotation if configured
        if getattr(self.unwrapped.task, 'unidirectional_rot', False):
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5

        rot_actions = rot_actions * self.unwrapped.rot_threshold

        # Convert to quaternion
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / (angle.unsqueeze(-1) + 1e-6)  # Add epsilon to prevent division by zero

        rot_actions_quat = self.torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        quat_goal = self.torch_utils.quat_mul(rot_actions_quat, self.unwrapped.fingertip_midpoint_quat)

        # Restrict to upright orientation
        target_euler_xyz = torch.stack(self.torch_utils.get_euler_xyz(quat_goal), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Roll
        target_euler_xyz[:, 1] = 0.0      # Pitch

        self.pose_goal[:, 3:] = self.torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

    def _calc_force_goal(self):
        """Calculate force goal from action."""
        min_idx = self.force_size + 6
        max_idx = min_idx + self.force_size

        # Extract force actions and scale by threshold - require explicit configuration
        if self.ctrl_cfg.force_action_threshold is None:
            if not hasattr(self.unwrapped.ctrl, 'force_action_threshold'):
                raise ValueError(
                    "Force action threshold not configured. Please set ctrl_cfg.force_action_threshold "
                    "or ensure environment has ctrl.force_action_threshold. Example: force_action_threshold=[10, 10, 10]"
                )
            force_threshold = self.unwrapped.ctrl.force_action_threshold
        else:
            force_threshold = self.ctrl_cfg.force_action_threshold
        force_delta = self.unwrapped.actions[:, min_idx:max_idx] * force_threshold[0]

        # Add current force to get absolute goal - require explicit bounds
        if self.ctrl_cfg.force_action_bounds is None:
            if not hasattr(self.unwrapped.ctrl, 'force_action_bounds'):
                raise ValueError(
                    "Force action bounds not configured. Please set ctrl_cfg.force_action_bounds "
                    "or ensure environment has ctrl.force_action_bounds. Example: force_action_bounds=[50, 50, 50]"
                )
            force_bounds = self.unwrapped.ctrl.force_action_bounds
        else:
            force_bounds = self.ctrl_cfg.force_action_bounds
        self.force_goal[:, :3] = torch.clip(
            force_delta[:, :3] + self.robot_force_torque[:, :3],
            -force_bounds[0], force_bounds[0]
        )

        # Handle torque if enabled - require explicit configuration
        if self.force_size > 3:
            if self.ctrl_cfg.torque_action_threshold is None:
                if not hasattr(self.unwrapped.ctrl, 'torque_action_threshold'):
                    raise ValueError(
                        "Torque action threshold not configured for 6DOF control. Please set ctrl_cfg.torque_action_threshold "
                        "or ensure environment has ctrl.torque_action_threshold. Example: torque_action_threshold=[0.1, 0.1, 0.1]"
                    )
                torque_threshold = self.unwrapped.ctrl.torque_action_threshold
            else:
                torque_threshold = self.ctrl_cfg.torque_action_threshold

            if self.ctrl_cfg.torque_action_bounds is None:
                if not hasattr(self.unwrapped.ctrl, 'torque_action_bounds'):
                    raise ValueError(
                        "Torque action bounds not configured for 6DOF control. Please set ctrl_cfg.torque_action_bounds "
                        "or ensure environment has ctrl.torque_action_bounds. Example: torque_action_bounds=[0.5, 0.5, 0.5]"
                    )
                torque_bounds = self.unwrapped.ctrl.torque_action_bounds
            else:
                torque_bounds = self.ctrl_cfg.torque_action_bounds

            torque_delta = force_delta[:, 3:] * torque_threshold[0] / force_threshold[0]
            self.force_goal[:, 3:] = torch.clip(
                torque_delta + self.robot_force_torque[:, 3:],
                -torque_bounds[0], torque_bounds[0]
            )

    def _update_targets_with_ema(self):
        """Update targets using EMA filtering of goals."""
        if not self._targets_initialized or self.target_init_mode == "first_goal":
            # Initialize targets to first goals
            self.target_selection.copy_(self.sel_goal)
            self.target_pose.copy_(self.pose_goal)
            self.target_force.copy_(self.force_goal)
            self._targets_initialized = True
        else:
            # Apply EMA filtering
            # Selection matrix EMA (apply to raw values before threshold)
            if self.no_sel_ema:
                self.target_selection[:, :self.force_size] = self.sel_goal[:, :self.force_size]
            else:
                self.target_selection[:, :self.force_size] = (
                    self.ema_factor * self.sel_goal[:, :self.force_size] +
                    (1 - self.ema_factor) * self.target_selection[:, :self.force_size]
                )

            # Pose EMA
            self.target_pose = (
                self.ema_factor * self.pose_goal +
                (1 - self.ema_factor) * self.target_pose
            )

            # Force EMA
            self.target_force = (
                self.ema_factor * self.force_goal +
                (1 - self.ema_factor) * self.target_force
            )

    def _set_control_targets_from_targets(self):
        """Set environment control targets from filtered targets."""
        # Update selection matrix from target (apply threshold here)
        self.sel_matrix[:, :self.force_size] = torch.where(
            self.target_selection[:, :self.force_size] > 0.5, 1.0, 0.0
        )

        # Set pose targets
        self.unwrapped.ctrl_target_fingertip_midpoint_pos = self.target_pose[:, :3]
        self.unwrapped.ctrl_target_fingertip_midpoint_quat = self.target_pose[:, 3:]

        # Store force target for use in apply_action
        self.target_force_for_control = self.target_force.clone()


    def _get_target_out_of_bounds(self):
        """Check if fingertip target is out of position bounds."""
        delta = self.unwrapped.fingertip_midpoint_pos - self.unwrapped.fixed_pos_action_frame
        pos_bounds = self.unwrapped.ctrl.pos_action_bounds

        out_of_bounds = torch.logical_or(
            delta <= -pos_bounds[0],
            delta >= pos_bounds[1]
        )
        return out_of_bounds

    def _wrapped_apply_action(self):
        """Apply hybrid force-position control using filtered targets."""
        # Get current yaw for success checking
        _, _, curr_yaw = self.torch_utils.get_euler_xyz(self.unwrapped.fingertip_midpoint_quat)
        self.unwrapped.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Update intermediate values if needed
        if self.unwrapped.last_update_timestamp < self.unwrapped._robot._data._sim_timestamp:
            self.unwrapped._compute_intermediate_values(dt=self.unwrapped.physics_dt)

        self.unwrapped.ctrl_target_gripper_dof_pos = 0.0

        # Check for out-of-bounds positions
        out_of_bounds = self._get_target_out_of_bounds()

        # Compute pose wrench using filtered targets
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

        # Compute force wrench using filtered targets
        force_wrench = compute_force_task_wrench(
            cfg=self.unwrapped.cfg,
            dof_pos=self.unwrapped.joint_pos,
            eef_force=self.robot_force_torque,
            ctrl_target_force=self.target_force_for_control,
            task_gains=self.kp,
            device=self.unwrapped.device
        )

        # Zero force wrench if position is out of bounds
        force_wrench[:, :3][out_of_bounds] = 0.0

        # Combine wrenches using filtered selection matrix
        task_wrench = (1 - self.sel_matrix) * pose_wrench + self.sel_matrix * force_wrench

        # For torque, always use position control (if not controlling torques)
        if not self.ctrl_torque:
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
        active_force = torch.abs(self.robot_force_torque[:, :self.force_size]) > self.task_cfg.force_active_threshold
        if self.force_size > 3:
            active_force[:, 3:] = torch.abs(self.robot_force_torque[:, 3:]) > self.task_cfg.torque_active_threshold

        force_ctrl = self.sel_matrix[:, :self.force_size].bool()

        good_force_cmd = torch.logical_and(torch.any(active_force, dim=1), torch.any(force_ctrl, dim=1))
        bad_force_cmd = torch.logical_and(torch.all(~active_force, dim=1), torch.any(force_ctrl, dim=1))

        sel_rew = self.task_cfg.good_force_cmd_rew * good_force_cmd + self.task_cfg.bad_force_cmd_rew * bad_force_cmd

        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['Reward / Selection Matrix'] = sel_rew

        return sel_rew

    def _directional_force_reward(self):
        """Direction-specific force reward."""
        active_force = torch.abs(self.robot_force_torque[:, :self.force_size]) > self.task_cfg.force_active_threshold
        if self.force_size > 3:
            active_force[:, 3:] = torch.abs(self.robot_force_torque[:, 3:]) > self.task_cfg.torque_active_threshold

        force_ctrl = self.sel_matrix[:, :self.force_size].bool()

        good_dims = torch.logical_and(force_ctrl, active_force) | torch.logical_and(~force_ctrl, ~active_force)
        bad_dims = torch.logical_and(force_ctrl, ~active_force) | torch.logical_and(~force_ctrl, active_force)

        good_rew = self.task_cfg.good_force_cmd_rew * torch.sum(good_dims, dim=1)
        bad_rew = self.task_cfg.bad_force_cmd_rew * torch.sum(bad_dims, dim=1)

        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['Reward / Selection Matrix'] = good_rew + bad_rew

        return good_rew + bad_rew

    def _delta_selection_reward(self):
        """Penalize changes in selection matrix."""
        sel_rew = torch.sum(torch.abs(self.sel_matrix - self._old_sel_matrix), dim=1) * self.task_cfg.bad_force_cmd_rew
        self._old_sel_matrix = self.sel_matrix.clone()

        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['Reward / Selection Matrix'] = sel_rew

        return sel_rew

    def _position_simple_reward(self):
        """Position-focused simple force reward."""
        active_force = torch.abs(self.robot_force_torque[:, :self.force_size]) > self.task_cfg.force_active_threshold
        if self.force_size > 3:
            active_force[:, 3:] = torch.abs(self.robot_force_torque[:, 3:]) > self.task_cfg.torque_active_threshold

        force_ctrl = self.sel_matrix[:, :self.force_size].bool()

        good_force_cmd = torch.logical_and(torch.any(active_force, dim=1), torch.any(force_ctrl, dim=1))
        sel_rew = self.task_cfg.good_force_cmd_rew * good_force_cmd

        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['Reward / Selection Matrix'] = sel_rew

        return sel_rew

    def _low_wrench_reward(self):
        """Reward for low wrench magnitude."""
        wrench_norm = self.unwrapped.actions[:, self.force_size:].norm(dim=-1)
        return -wrench_norm * self.task_cfg.wrench_norm_scale

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