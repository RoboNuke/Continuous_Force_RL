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

    def __init__(
            self,
            env,
            ctrl_torque=False,
            reward_type="simp",
            ctrl_cfg=None,
            task_cfg=None,
            num_agents=1
        ):
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
            num_agents: Number of agents for static environment assignment.
        """
        # Store original action space size before modification
        self._original_action_size = getattr(env.unwrapped.cfg, 'action_space', 6)

        super().__init__(env)

        # Store and validate num_agents
        self.num_agents = num_agents
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        if self.num_envs % self.num_agents != 0:
            raise ValueError(f"Number of environments ({self.num_envs}) must be divisible by number of agents ({self.num_agents})")

        self.envs_per_agent = self.num_envs // self.num_agents

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

        print(f"[HYBRID DEBUG] Action space calculation:")
        print(f"  force_size: {self.force_size}")
        print(f"  original_action_space: {original_action_space}")
        print(f"  calculated action_space_size: {self.action_space_size}")

        # Update environment action space - ensure both integer and gym.Space are consistent
        if hasattr(self.unwrapped, 'cfg'):
            old_cfg_action_space = getattr(self.unwrapped.cfg, 'action_space', None)
            self.unwrapped.cfg.action_space = self.action_space_size
            print(f"[HYBRID DEBUG] Updated cfg.action_space: {old_cfg_action_space} -> {self.action_space_size}")

            if hasattr(self.unwrapped, '_configure_gym_env_spaces'):
                self.unwrapped._configure_gym_env_spaces()

        # Update the gym.Space action_space to match the integer config
        old_gym_action_space = getattr(self.unwrapped, 'action_space', None)
        old_gym_shape = old_gym_action_space.shape if hasattr(old_gym_action_space, 'shape') else None

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_space_size,), dtype=np.float32
        )
        self.unwrapped.action_space = self.action_space

        print(f"[HYBRID DEBUG] Updated gym action_space: {old_gym_shape} -> {self.action_space.shape}")

        # Validation: ensure both sources match
        cfg_size = getattr(self.unwrapped.cfg, 'action_space', None)
        gym_size = self.unwrapped.action_space.shape[0] if hasattr(self.unwrapped.action_space, 'shape') else None
        print(f"[HYBRID DEBUG] Action space validation: cfg={cfg_size}, gym={gym_size}")

        if cfg_size != gym_size:
            raise ValueError(f"Action space mismatch: cfg.action_space={cfg_size} != gym.action_space.shape[0]={gym_size}")

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

        # Note: Contact detection (active_force) has been moved to ForceTorqueWrapper
        # Access via self.unwrapped.in_contact instead

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

        # Flag to control wrench logging once per step
        self._should_log_wrenches = False

        # Store original methods
        self._original_pre_physics_step = None
        self._original_apply_action = None
        self._original_update_rew_buf = None

        # Reward tracking for delta strategy
        if reward_type == "delta":
            self._old_sel_matrix = torch.zeros_like(self.sel_matrix)

        # Per-step tracking for action effect metrics
        self._prev_step_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_step_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_step_force = torch.zeros((self.num_envs, 3), device=self.device)
        self._first_step_set = False

        # Calculate new action space size based on control configuration
        # Action space: 6 (pose) + 6 (force) + 0/6 (torque selection if ctrl_torque)
        self._new_action_size = 12 if not ctrl_torque else 18

        # Update observation/state dimensions for action space change
        self._update_observation_dimensions()

        # Initialize wrapper
        self._wrapper_initialized = False
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()
        if hasattr(self.unwrapped, 'extras'):
            if 'to_log' not in self.unwrapped.extras.keys():
                self.unwrapped.extras['to_log'] = {}

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
                "      contact_force_threshold: 0.1  # Configure contact detection\n"
                "      contact_torque_threshold: 0.01\n"
                "      log_contact_state: true\n"
                "\n"
                "The ForceTorqueWrapper must be enabled and applied before HybridForcePositionWrapper.\n"
                "This is a required dependency that must be explicitly configured."
            )

        # Store reference to the environment that has force-torque data
        self._force_torque_env = current_env

        # Note: in_contact tensor validation is deferred to _validate_contact_detection()
        # because it's created during env._init_tensors() which happens after wrapper initialization

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

        # Call original pre_physics_step to maintain wrapper chain
        # (allows ForceTorqueWrapper and other wrappers to execute)
        # Base env will apply EMA, but we override it below for hybrid control
        if self._original_pre_physics_step:
            self._original_pre_physics_step(action)

        # Override base env's EMA - hybrid control does its own EMA on goals
        self.unwrapped.actions = action.clone().to(self.device)

        # Step 1: Extract goals from current action
        self._extract_goals_from_action(action)

        # Step 2: Apply EMA filtering to get targets
        self._update_targets_with_ema()

        # Step 3: Set control targets from filtered targets
        self._set_control_targets_from_targets()

        # Log action effect metrics every step
        self._log_action_effect_metrics()

        # Note: Contact detection now happens in ForceTorqueWrapper
        # Access contact state via self.unwrapped.in_contact

        # Log selection matrix (contact state now logged in ForceTorqueWrapper)
        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['to_log']['Control Mode / Force Control X'] = self.target_selection[:, 0]
            self.unwrapped.extras['to_log']['Control Mode / Force Control Y'] = self.target_selection[:, 1]
            self.unwrapped.extras['to_log']['Control Mode / Force Control Z'] = self.target_selection[:, 2]

            if self.force_size > 3:
                self.unwrapped.extras['to_log']['Control Mode / Force Control RX'] = self.target_selection[:, 3]
                self.unwrapped.extras['to_log']['Control Mode / Force Control RY'] = self.target_selection[:, 4]
                self.unwrapped.extras['to_log']['Control Mode / Force Control RZ'] = self.target_selection[:, 5]

        # Update intermediate values
        if hasattr(self.unwrapped, '_compute_intermediate_values'):
            self.unwrapped._compute_intermediate_values(dt=self.unwrapped.physics_dt)

        # Enable wrench logging for the first apply_action call this step
        self._should_log_wrenches = True

    def _reset_targets(self, env_ids):
        """Reset targets for given environment IDs."""
        if self.target_init_mode == "zero":
            # Initialize to zeros
            self.target_selection[env_ids] = 0.0
            self.target_pose[env_ids] = 0.0
            self.target_force[env_ids] = 0.0
        # For "first_goal" mode, targets will be set from first goal after reset

        self._targets_initialized = True

        # Reset per-step tracking states
        if self._first_step_set:
            # Reset to current state values for reset environments
            self._prev_step_pos[env_ids] = self.unwrapped.fingertip_midpoint_pos[env_ids]
            self._prev_step_vel[env_ids] = self.unwrapped.fingertip_midpoint_linvel[env_ids] #self.unwrapped.ee_linvel_fd[env_ids]
            self._prev_step_force[env_ids] = self.robot_force_torque[env_ids, :3]

    def _extract_goals_from_action(self, action):
        """Extract pose, force, and selection goals from action."""
        # Extract selection goal (raw values before threshold)
        self.sel_goal[:, :self.force_size] = action[:, :self.force_size]

        # Note: actions are already stored in _wrapped_pre_physics_step

        # Log raw network actions
        if hasattr(self.unwrapped, 'extras'):
            # Raw position actions (before scaling)
            raw_sel_actions = action[:, 0:self.force_size]
            self.unwrapped.extras['to_log']['Network Output / Raw Sel Action X'] = raw_sel_actions[:, 0]#.abs()
            self.unwrapped.extras['to_log']['Network Output / Raw Sel Action Y'] = raw_sel_actions[:, 1]#.abs()
            self.unwrapped.extras['to_log']['Network Output / Raw Sel Action Z'] = raw_sel_actions[:, 2]#.abs()

            # Raw position actions (before scaling)
            raw_pos_actions = action[:, self.force_size:self.force_size+3]
            self.unwrapped.extras['to_log']['Network Output / Raw Pos Action X'] = raw_pos_actions[:, 0].abs()
            self.unwrapped.extras['to_log']['Network Output / Raw Pos Action Y'] = raw_pos_actions[:, 1].abs()
            self.unwrapped.extras['to_log']['Network Output / Raw Pos Action Z'] = raw_pos_actions[:, 2].abs()

            # Raw force actions (before scaling)
            raw_force_actions = action[:, self.force_size+6:2*self.force_size+6]
            self.unwrapped.extras['to_log']['Network Output / Raw Force Action X'] = raw_force_actions[:, 0].abs()
            self.unwrapped.extras['to_log']['Network Output / Raw Force Action Y'] = raw_force_actions[:, 1].abs()
            self.unwrapped.extras['to_log']['Network Output / Raw Force Action Z'] = raw_force_actions[:, 2].abs()

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
            if not hasattr(self.unwrapped.cfg.ctrl, 'pos_action_bounds'):
                raise ValueError(
                    "Position action bounds not configured. Please set ctrl_cfg.pos_action_bounds "
                    "or ensure environment has ctrl.pos_action_bounds. Example: pos_action_bounds=[0.05, 0.05, 0.05]"
                )
            pos_bounds = self.unwrapped.cfg.ctrl.pos_action_bounds
        else:
            pos_bounds = self.ctrl_cfg.pos_action_bounds
        pos_error_clipped = torch.clip(delta_pos, -pos_bounds[0], pos_bounds[1])
        self.pose_goal[:, :3] = self.unwrapped.fixed_pos_action_frame + pos_error_clipped

        # Quaternion goal
        rot_actions = self.unwrapped.actions[:, self.force_size+3:self.force_size+6]

        # Handle unidirectional rotation if configured
        if getattr(self.unwrapped.cfg_task, 'unidirectional_rot', False):
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

        # Log L2 norm of position goal
        if hasattr(self.unwrapped, 'extras'):
            pos_goal_norm = torch.norm(self.pose_goal[:, :3], p=2, dim=-1)
            self.unwrapped.extras['to_log']['Control Target / Pos Goal Norm'] = pos_goal_norm

    def _calc_force_goal(self):
        """Calculate force goal from action."""
        min_idx = self.force_size + 6
        max_idx = min_idx + self.force_size

        # Extract force actions and scale by threshold - require explicit configuration
        if self.ctrl_cfg.force_action_threshold is None:
            if not hasattr(self.unwrapped.cfg.ctrl, 'force_action_threshold'):
                raise ValueError(
                    "Force action threshold not configured. Please set ctrl_cfg.force_action_threshold "
                    "or ensure environment has ctrl.force_action_threshold. Example: force_action_threshold=[10, 10, 10]"
                )
            force_threshold = self.unwrapped.cfg.ctrl.force_action_threshold
        else:
            force_threshold = self.ctrl_cfg.force_action_threshold
        force_delta = self.unwrapped.actions[:, min_idx:max_idx] * force_threshold[0]

        # Add current force to get absolute goal - require explicit bounds
        if self.ctrl_cfg.force_action_bounds is None:
            if not hasattr(self.unwrapped.cfg.ctrl, 'force_action_bounds'):
                raise ValueError(
                    "Force action bounds not configured. Please set ctrl_cfg.force_action_bounds "
                    "or ensure environment has ctrl.force_action_bounds. Example: force_action_bounds=[50, 50, 50]"
                )
            force_bounds = self.unwrapped.cfg.ctrl.force_action_bounds
        else:
            force_bounds = self.ctrl_cfg.force_action_bounds
        self.force_goal[:, :3] = torch.clip(
            force_delta[:, :3] + self.robot_force_torque[:, :3],
            -force_bounds[0], force_bounds[0]
        )

        # Handle torque if enabled - require explicit configuration
        if self.force_size > 3:
            if self.ctrl_cfg.torque_action_threshold is None:
                if not hasattr(self.unwrapped.cfg.ctrl, 'torque_action_threshold'):
                    raise ValueError(
                        "Torque action threshold not configured for 6DOF control. Please set ctrl_cfg.torque_action_threshold "
                        "or ensure environment has ctrl.torque_action_threshold. Example: torque_action_threshold=[0.1, 0.1, 0.1]"
                    )
                torque_threshold = self.unwrapped.cfg.ctrl.torque_action_threshold
            else:
                torque_threshold = self.ctrl_cfg.torque_action_threshold

            if self.ctrl_cfg.torque_action_bounds is None:
                if not hasattr(self.unwrapped.cfg.ctrl, 'torque_action_bounds'):
                    raise ValueError(
                        "Torque action bounds not configured for 6DOF control. Please set ctrl_cfg.torque_action_bounds "
                        "or ensure environment has ctrl.torque_action_bounds. Example: torque_action_bounds=[0.5, 0.5, 0.5]"
                    )
                torque_bounds = self.unwrapped.cfg.ctrl.torque_action_bounds
            else:
                torque_bounds = self.ctrl_cfg.torque_action_bounds

            torque_delta = force_delta[:, 3:] * torque_threshold[0] / force_threshold[0]
            self.force_goal[:, 3:] = torch.clip(
                torque_delta + self.robot_force_torque[:, 3:],
                -torque_bounds[0], torque_bounds[0]
            )

        # Log L2 norm of force goal
        if hasattr(self.unwrapped, 'extras'):
            force_goal_norm = torch.norm(self.force_goal[:, :3], p=2, dim=-1)
            self.unwrapped.extras['to_log']['Control Target / Force Goal Norm'] = force_goal_norm

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

    def _log_action_effect_metrics(self):
        """Log per-step changes in position, velocity, and force attributed to current control mode."""
        if not hasattr(self.unwrapped, 'extras'):
            return

        # Get current state
        current_pos = self.unwrapped.fingertip_midpoint_pos
        current_vel = self.unwrapped.fingertip_midpoint_linvel #self.unwrapped.ee_linvel_fd
        current_force = self.robot_force_torque[:, :3]

        # First time initialization
        if not self._first_step_set:
            self._prev_step_pos.copy_(current_pos)
            self._prev_step_vel.copy_(current_vel)
            self._prev_step_force.copy_(current_force)
            self._first_step_set = True
            return

        # Calculate per-step deltas
        pos_delta = current_pos - self._prev_step_pos
        vel_delta = current_vel - self._prev_step_vel
        force_delta = current_force - self._prev_step_force

        # Log metrics aggregated by agent and by current control mode (per axis)
        axis_names = ['X', 'Y', 'Z']
        for i in range(3):
            # Get current control mode for this axis
            is_force_control = self.sel_matrix[:, i] == 1.0
            is_pos_control = self.sel_matrix[:, i] == 0.0

            # Create per-agent tensors for force control mode
            force_pos_change = torch.full((self.num_agents,), float('nan'), device=self.device)
            force_vel_change = torch.full((self.num_agents,), float('nan'), device=self.device)
            force_force_change = torch.full((self.num_agents,), float('nan'), device=self.device)

            # Create per-agent tensors for position control mode
            pos_pos_change = torch.full((self.num_agents,), float('nan'), device=self.device)
            pos_vel_change = torch.full((self.num_agents,), float('nan'), device=self.device)
            pos_force_change = torch.full((self.num_agents,), float('nan'), device=self.device)

            # Aggregate by agent
            for agent_id in range(self.num_agents):
                start_idx = agent_id * self.envs_per_agent
                end_idx = (agent_id + 1) * self.envs_per_agent

                # Get control modes for this agent
                agent_force_control = is_force_control[start_idx:end_idx]
                agent_pos_control = is_pos_control[start_idx:end_idx]

                # Force control mode metrics
                if torch.any(agent_force_control):
                    force_pos_change[agent_id] = pos_delta[start_idx:end_idx, i][agent_force_control].abs().mean()
                    force_vel_change[agent_id] = vel_delta[start_idx:end_idx, i][agent_force_control].abs().mean()
                    force_force_change[agent_id] = force_delta[start_idx:end_idx, i][agent_force_control].abs().mean()

                # Position control mode metrics
                if torch.any(agent_pos_control):
                    pos_pos_change[agent_id] = pos_delta[start_idx:end_idx, i][agent_pos_control].abs().mean()
                    pos_vel_change[agent_id] = vel_delta[start_idx:end_idx, i][agent_pos_control].abs().mean()
                    pos_force_change[agent_id] = force_delta[start_idx:end_idx, i][agent_pos_control].abs().mean()

            # Log force control mode changes
            if not torch.all(torch.isnan(force_pos_change)):
                self.unwrapped.extras['to_log'][f'Action Effect / Force Control Pos Change {axis_names[i]}'] = force_pos_change
            if not torch.all(torch.isnan(force_vel_change)):
                self.unwrapped.extras['to_log'][f'Action Effect / Force Control Vel Change {axis_names[i]}'] = force_vel_change
            if not torch.all(torch.isnan(force_force_change)):
                self.unwrapped.extras['to_log'][f'Action Effect / Force Control Force Change {axis_names[i]}'] = force_force_change

            # Log position control mode changes
            if not torch.all(torch.isnan(pos_pos_change)):
                self.unwrapped.extras['to_log'][f'Action Effect / Position Control Pos Change {axis_names[i]}'] = pos_pos_change
            if not torch.all(torch.isnan(pos_vel_change)):
                self.unwrapped.extras['to_log'][f'Action Effect / Position Control Vel Change {axis_names[i]}'] = pos_vel_change
            if not torch.all(torch.isnan(pos_force_change)):
                self.unwrapped.extras['to_log'][f'Action Effect / Position Control Force Change {axis_names[i]}'] = pos_force_change

        # Update previous step state for next iteration
        self._prev_step_pos.copy_(current_pos)
        self._prev_step_vel.copy_(current_vel)
        self._prev_step_force.copy_(current_force)

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
        """Apply hybrid force-position control using filtered targets."""
        # Get current yaw for success checking
        _, _, curr_yaw = self.torch_utils.get_euler_xyz(self.unwrapped.fingertip_midpoint_quat)
        self.unwrapped.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Update intermediate values if needed
        if self.unwrapped.last_update_timestamp < self.unwrapped._robot._data._sim_timestamp:
            self.unwrapped._compute_intermediate_values(dt=self.unwrapped.physics_dt)

        self.unwrapped.ctrl_target_gripper_dof_pos = 0.0

        # Compute pose wrench using filtered targets
        pose_wrench = compute_pose_task_wrench(
            cfg=self.unwrapped.cfg,
            dof_pos=self.unwrapped.joint_pos,
            fingertip_midpoint_pos=self.unwrapped.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.unwrapped.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.unwrapped.fingertip_midpoint_linvel, #self.unwrapped.ee_linvel_fd, # TODO FD HERE TOO!
            fingertip_midpoint_angvel=self.unwrapped.fingertip_midpoint_angvel, #self.unwrapped.ee_angvel_fd,
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
            fingertip_midpoint_linvel=self.unwrapped.fingertip_midpoint_linvel, #self.unwrapped.ee_linvel_fd,
            fingertip_midpoint_angvel=self.unwrapped.fingertip_midpoint_angvel, #self.unwrapped.ee_angvel_fd,
            ctrl_target_force=self.target_force_for_control,
            task_gains=self.kp,
            task_deriv_gains=self.unwrapped.task_deriv_gains,
            device=self.unwrapped.device
        )

        # Log active wrench magnitudes based on control mode, aggregated per agent
        # Only log once per step (first apply_action call)
        if self._should_log_wrenches and hasattr(self.unwrapped, 'extras'):
            # Log force and position errors
            force_error = self.target_force_for_control[:, :3] - self.robot_force_torque[:, :3]
            self.unwrapped.extras['to_log']['Controller Output / Force Error X'] = force_error[:, 0]
            self.unwrapped.extras['to_log']['Controller Output / Force Error Y'] = force_error[:, 1]
            self.unwrapped.extras['to_log']['Controller Output / Force Error Z'] = force_error[:, 2]

            position_error = self.unwrapped.ctrl_target_fingertip_midpoint_pos - self.unwrapped.fingertip_midpoint_pos
            self.unwrapped.extras['to_log']['Controller Output / Position Error X'] = position_error[:, 0]
            self.unwrapped.extras['to_log']['Controller Output / Position Error Y'] = position_error[:, 1]
            self.unwrapped.extras['to_log']['Controller Output / Position Error Z'] = position_error[:, 2]

            axis_names = ['X', 'Y', 'Z']
            for i in range(3):  # Position axes
                # Create per-agent tensors initialized with NaN
                force_wrench_per_agent = torch.full((self.num_agents,), float('nan'), device=self.device)
                pose_wrench_per_agent = torch.full((self.num_agents,), float('nan'), device=self.device)

                # Aggregate by agent
                for agent_id in range(self.num_agents):
                    # Get this agent's environment slice
                    start_idx = agent_id * self.envs_per_agent
                    end_idx = (agent_id + 1) * self.envs_per_agent

                    # Force control aggregation
                    force_control_mask = self.sel_matrix[start_idx:end_idx, i] == 1.0
                    if torch.any(force_control_mask):
                        force_wrench_per_agent[agent_id] = force_wrench[start_idx:end_idx, i][force_control_mask].abs().mean()

                    # Position control aggregation
                    pos_control_mask = self.sel_matrix[start_idx:end_idx, i] == 0.0
                    if torch.any(pos_control_mask):
                        pose_wrench_per_agent[agent_id] = pose_wrench[start_idx:end_idx, i][pos_control_mask].abs().mean()

                # Only log if at least one agent has a non-NaN value
                if not torch.all(torch.isnan(force_wrench_per_agent)):
                    self.unwrapped.extras['to_log'][f'Controller Output / Active Force Wrench {axis_names[i]}'] = force_wrench_per_agent
                if not torch.all(torch.isnan(pose_wrench_per_agent)):
                    self.unwrapped.extras['to_log'][f'Controller Output / Active Pos Wrench {axis_names[i]}'] = pose_wrench_per_agent

            # Future: Add logging for torque/rotation wrenches when ctrl_torque is enabled
            if self.force_size > 3:
                # TODO: Add logging for rotational axes (RX, RY, RZ)
                # - Active Torque Wrench RX/RY/RZ (when sel_matrix[:, 3:] == 1.0)
                # - Active Rotation Wrench RX/RY/RZ (when sel_matrix[:, 3:] == 0.0)
                pass

            # Disable wrench logging for subsequent apply_action calls this step
            self._should_log_wrenches = False

        # Combine wrenches using filtered selection matrix
        task_wrench = (1 - self.sel_matrix) * pose_wrench + self.sel_matrix * force_wrench

        # Apply bounds constraint to final wrench - prevent motion outside boundaries
        delta_pos = self.unwrapped.fingertip_midpoint_pos - self.unwrapped.fixed_pos_action_frame
        pos_bounds = self.unwrapped.cfg.ctrl.pos_action_bounds

        # Zero wrench components that would drive further out of bounds
        for i in range(3):  # x, y, z
            # If at negative bound and wrench would push more negative, zero it
            at_neg_bound = delta_pos[:, i] <= -pos_bounds[0]
            neg_wrench = task_wrench[:, i] < 0
            task_wrench[:, i] = torch.where(at_neg_bound & neg_wrench, 0.0, task_wrench[:, i])

            # If at positive bound and wrench would push more positive, zero it
            at_pos_bound = delta_pos[:, i] >= pos_bounds[1]
            pos_wrench = task_wrench[:, i] > 0
            task_wrench[:, i] = torch.where(at_pos_bound & pos_wrench, 0.0, task_wrench[:, i])

        # For torque, always use position control (if not controlling torques)
        if not self.ctrl_torque:
            task_wrench[:, 3:] = pose_wrench[:, 3:]

        # Compute joint torques
        self.unwrapped.joint_torque, task_wrench = compute_dof_torque_from_wrench(
            cfg=self.unwrapped.cfg,
            dof_pos=self.unwrapped.joint_pos,
            dof_vel=self.unwrapped.joint_vel, #self.unwrapped.joint_vel_fd, #TODO DIFF JOINT VELS!
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
        force_ctrl = self.sel_matrix[:, :self.force_size].bool()

        good_force_cmd = torch.logical_and(torch.any(self.unwrapped.in_contact[:, :self.force_size], dim=1), torch.any(force_ctrl, dim=1))
        bad_force_cmd = torch.logical_and(torch.all(~self.unwrapped.in_contact[:, :self.force_size], dim=1), torch.any(force_ctrl, dim=1))

        sel_rew = self.task_cfg.good_force_cmd_rew * good_force_cmd + self.task_cfg.bad_force_cmd_rew * bad_force_cmd

        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['logs_rew_Selection Matrix'] = sel_rew

        return sel_rew

    def _directional_force_reward(self):
        """Direction-specific force reward."""
        force_ctrl = self.sel_matrix[:, :self.force_size].bool()

        good_dims = torch.logical_and(force_ctrl, self.unwrapped.in_contact[:, :self.force_size]) | torch.logical_and(~force_ctrl, ~self.unwrapped.in_contact[:, :self.force_size])
        bad_dims = torch.logical_and(force_ctrl, ~self.unwrapped.in_contact[:, :self.force_size]) | torch.logical_and(~force_ctrl, self.unwrapped.in_contact[:, :self.force_size])

        good_rew = self.task_cfg.good_force_cmd_rew * torch.sum(good_dims, dim=1)
        bad_rew = self.task_cfg.bad_force_cmd_rew * torch.sum(bad_dims, dim=1)

        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['logs_rew_Selection Matrix'] = good_rew + bad_rew

        return good_rew + bad_rew

    def _delta_selection_reward(self):
        """Penalize changes in selection matrix."""
        sel_rew = torch.sum(torch.abs(self.sel_matrix - self._old_sel_matrix), dim=1) * self.task_cfg.bad_force_cmd_rew
        self._old_sel_matrix = self.sel_matrix.clone()

        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['logs_rew_Selection Matrix'] = sel_rew

        return sel_rew

    def _position_simple_reward(self):
        """Position-focused simple force reward."""
        force_ctrl = self.sel_matrix[:, :self.force_size].bool()

        good_force_cmd = torch.logical_and(torch.any(self.unwrapped.in_contact[:, :self.force_size], dim=1), torch.any(force_ctrl, dim=1))
        sel_rew = self.task_cfg.good_force_cmd_rew * good_force_cmd

        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['logs_rew_Selection Matrix'] = sel_rew

        return sel_rew

    def _low_wrench_reward(self):
        """Reward for low wrench magnitude."""
        wrench_norm = self.unwrapped.actions[:, self.force_size:].norm(dim=-1)
        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['log_rew_Wrench_Norm'] = wrench_norm
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