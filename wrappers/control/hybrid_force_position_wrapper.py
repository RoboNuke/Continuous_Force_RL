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
            num_agents=1,
            use_ground_truth_selection=False
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
            use_ground_truth_selection: If True, override selection actions with ground truth contact state.
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
        self.use_ground_truth_selection = use_ground_truth_selection
        self.use_delta_force = getattr(self.ctrl_cfg, 'use_delta_force', False)
        self.async_z_bounds = self.ctrl_cfg.async_z_force_bounds
        self.apply_ema_force = self.ctrl_cfg.apply_ema_force

        # PID force control flags
        self.enable_force_derivative = getattr(self.ctrl_cfg, 'enable_force_derivative', False)
        self.enable_force_integral = getattr(self.ctrl_cfg, 'enable_force_integral', False)
        self.force_integral_clamp = getattr(self.ctrl_cfg, 'force_integral_clamp', 50.0)

        # Update action space to include selection matrix + position + force
        original_action_space = 6  # Original position + rotation actions
        self.action_space_size = 2*self.force_size + original_action_space

        # Update environment action space - ensure both integer and gym.Space are consistent
        if hasattr(self.unwrapped, 'cfg'):
            self.unwrapped.cfg.action_space = self.action_space_size

            if hasattr(self.unwrapped, '_configure_gym_env_spaces'):
                self.unwrapped._configure_gym_env_spaces()

        # Update the gym.Space action_space to match the integer config
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_space_size,), dtype=np.float32
        )
        self.unwrapped.action_space = self.action_space

        # Validation: ensure both sources match
        cfg_size = getattr(self.unwrapped.cfg, 'action_space', None)
        gym_size = self.unwrapped.action_space.shape[0] if hasattr(self.unwrapped.action_space, 'shape') else None

        if cfg_size != gym_size:
            raise ValueError(f"Action space mismatch: cfg.action_space={cfg_size} != gym.action_space.shape[0]={gym_size}")

        # EMA-filtered actions (Isaac Lab style - EMA on deltas, not absolute targets)
        self.ema_actions = torch.zeros((self.num_envs, self.action_space_size), device=self.device)

        # Selection matrix for control (derived from ema_actions)
        self.sel_matrix = torch.zeros((self.num_envs, 6), device=self.device)

        # Task wrench (commanded wrench to robot, stored for external access/plotting)
        self.task_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        # Integral state for force control (accumulates force error)
        self.force_integral_error = torch.zeros((self.num_envs, 6), device=self.device)

        # Track previous contact state for integral reset on contact change
        self._prev_contact_state = None

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

        # Initialize integral gains if enabled
        if self.enable_force_integral:
            if self.ctrl_cfg.default_task_force_integ_gains is None:
                raise ValueError(
                    "enable_force_integral=True but default_task_force_integ_gains not set in ctrl_cfg. "
                    "Please provide integral gains, e.g., [0.01, 0.01, 0.01, 0.001, 0.001, 0.001]"
                )
            self.ki = torch.tensor(
                self.ctrl_cfg.default_task_force_integ_gains,
                device=self.device
            ).unsqueeze(0).repeat(self.num_envs, 1)
            # Zero out torque integral gains if not controlling torques
            if not ctrl_torque:
                self.ki[:, 3:] = 0.0
        else:
            self.ki = None

        # Initialize threshold and bounds as class variables (per-environment tensors)
        # These are independent of base env to keep wrapper self-contained
        # Dynamics randomization wrapper can override these tensors

        # Force threshold (hybrid-specific)
        if self.ctrl_cfg.force_action_threshold is not None:
            force_thresh_val = self.ctrl_cfg.force_action_threshold
        elif hasattr(self.unwrapped.cfg.ctrl, 'force_action_threshold'):
            force_thresh_val = self.unwrapped.cfg.ctrl.force_action_threshold
        else:
            raise ValueError(
                "force_action_threshold must be provided in ctrl_cfg or env cfg.ctrl. "
                "This is required for hybrid force-position control."
            )
        self.force_threshold = torch.tensor(force_thresh_val, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # Torque threshold (if controlling torques)
        if self.ctrl_torque:
            if self.ctrl_cfg.torque_action_threshold is not None:
                torque_thresh_val = self.ctrl_cfg.torque_action_threshold
            elif hasattr(self.unwrapped.cfg.ctrl, 'torque_action_threshold'):
                torque_thresh_val = self.unwrapped.cfg.ctrl.torque_action_threshold
            else:
                raise ValueError(
                    "torque_action_threshold must be provided in ctrl_cfg or env cfg.ctrl when ctrl_torque=True"
                )
            self.torque_threshold = torch.tensor(torque_thresh_val, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # Bounds: shape (num_envs, 3) - symmetric bounds matching base environment format
        # Position bounds
        if self.ctrl_cfg.pos_action_bounds is not None:
            pos_bounds_val = self.ctrl_cfg.pos_action_bounds
        elif hasattr(self.unwrapped.cfg.ctrl, 'pos_action_bounds'):
            pos_bounds_val = self.unwrapped.cfg.ctrl.pos_action_bounds
        else:
            raise ValueError("pos_action_bounds must be provided in ctrl_cfg or env cfg.ctrl")
        self.pos_bounds = torch.tensor(pos_bounds_val, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # Force bounds
        if self.ctrl_cfg.force_action_bounds is not None:
            force_bounds_val = self.ctrl_cfg.force_action_bounds
        elif hasattr(self.unwrapped.cfg.ctrl, 'force_action_bounds'):
            force_bounds_val = self.unwrapped.cfg.ctrl.force_action_bounds
        else:
            raise ValueError("force_action_bounds must be provided in ctrl_cfg or env cfg.ctrl")
        self.force_bounds = torch.tensor(force_bounds_val, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # Torque bounds (if controlling torques)
        if self.ctrl_torque:
            if self.ctrl_cfg.torque_action_bounds is not None:
                torque_bounds_val = self.ctrl_cfg.torque_action_bounds
            elif hasattr(self.unwrapped.cfg.ctrl, 'torque_action_bounds'):
                torque_bounds_val = self.unwrapped.cfg.ctrl.torque_action_bounds
            else:
                raise ValueError("torque_action_bounds must be provided in ctrl_cfg or env cfg.ctrl when ctrl_torque=True")
            self.torque_bounds = torch.tensor(torque_bounds_val, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

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

        # Per-agent cumulative termination tracking by control mode
        self.termination_counts = {
            'force_x': [0] * self.num_agents,
            'force_y': [0] * self.num_agents,
            'force_z': [0] * self.num_agents,
            'pos_x': [0] * self.num_agents,
            'pos_y': [0] * self.num_agents,
            'pos_z': [0] * self.num_agents,
        }

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
            self.unwrapped.cfg.observation_space += action_diff

        if hasattr(self.unwrapped.cfg, 'state_space'):
            self.unwrapped.cfg.state_space += action_diff

        if hasattr(self.unwrapped.cfg, 'action_space'):
            self.unwrapped.cfg.action_space = self._new_action_size

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

        if hasattr(self.unwrapped, '_get_rewards') and self.reward_type != "base":
            self._original_get_rewards = self.unwrapped._get_rewards
            self.unwrapped._get_rewards = self._wrapped_get_rewards

        if hasattr(self.unwrapped, '_reset_idx'):
            self._original_reset_idx = self.unwrapped._reset_idx
            self.unwrapped._reset_idx = self._wrapped_reset_idx

        self._wrapper_initialized = True

    @property
    def robot_force_torque(self):
        """Access force-torque data from the correct environment in the wrapper chain."""
        return self._force_torque_env.robot_force_torque


    def _wrapped_pre_physics_step(self, action):
        """Process actions through EMA->compute targets->control flow (Isaac Lab style)."""
        # Check for contact change from previous step and reset integral if needed
        # This compares prev contact (stored before last physics step) with current contact (after last physics step)
        if self.enable_force_integral and self._prev_contact_state is not None and hasattr(self.unwrapped, 'in_contact'):
            curr_contact = self.unwrapped.in_contact[:, :self.force_size]
            self._reset_integral_on_contact_change(self._prev_contact_state, curr_contact)

        # Store current contact state for next step's comparison
        if self.enable_force_integral and hasattr(self.unwrapped, 'in_contact'):
            self._prev_contact_state = self.unwrapped.in_contact[:, :self.force_size].clone()

        # Handle reset environments first
        env_ids = self.unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            if hasattr(self.unwrapped, '_reset_buffers'):
                self.unwrapped._reset_buffers(env_ids)
            self._reset_ema_actions(env_ids)

        # Store previous actions for backward compatibility
        if hasattr(self.unwrapped, 'actions'):
            self.unwrapped.prev_action = self.unwrapped.actions.clone()

        # Call original pre_physics_step to maintain wrapper chain
        # (allows ForceTorqueWrapper and other wrappers to execute)
        # Base env will apply EMA, but we override it below for hybrid control
        if self._original_pre_physics_step:
            self._original_pre_physics_step(action.clone())

        # TESTING MODE: Override selection actions with ground truth contact state
        # Create a modified action tensor that replaces selection values
        # Clone preserves gradients for position/force, detach prevents gradients for ground truth
        modified_action = action.clone()
        if self.use_ground_truth_selection:
            modified_action[:, :self.force_size] = self.unwrapped.in_contact[:, :self.force_size].float().detach()

        # Override base env's EMA - hybrid control does its own EMA on actions
        # Use modified action for storage
        self.unwrapped.actions = modified_action.clone().to(self.device)

        # Step 1: Apply EMA to modified actions
        self._apply_ema_to_actions(modified_action)

        # Step 2: Compute control targets from current state + EMA actions
        self._compute_control_targets()

        # Log action effect metrics every step
        self._log_action_effect_metrics()

        # Note: Contact detection now happens in ForceTorqueWrapper
        # Access contact state via self.unwrapped.in_contact

        # Update intermediate values
        if hasattr(self.unwrapped, '_compute_intermediate_values'):
            self.unwrapped._compute_intermediate_values(dt=self.unwrapped.physics_dt)

        # Enable wrench logging for the first apply_action call this step
        self._should_log_wrenches = True

    def _reset_ema_actions(self, env_ids):
        """Reset EMA actions for given environment IDs (Isaac Lab style)."""
        # Reset EMA actions to zeros (Isaac Lab approach)
        self.ema_actions[env_ids] = 0.0

        ##########################################################################
        # TEMPORARY: Match base factory position action back-calculation bug
        #
        # Base factory back-calculates position actions during reset:
        #   actions = (fingertip_pos - fixed_pos_action_frame) / pos_action_bounds
        # But _apply_action uses pos_threshold for scaling:
        #   pos_actions = actions * pos_threshold
        #
        # Since pos_action_bounds=[0.05,0.05,0.05] and pos_threshold=[0.02,0.02,0.02],
        # the first step moves target AWAY from hole by 0.4x the initial displacement.
        #
        # Copy base env's back-calculated position actions to match this behavior.
        # Remove this block to revert to zero-initialization (true no-op on first step).
        ##########################################################################
        if hasattr(self.unwrapped, 'actions') and len(env_ids) > 0:
            self.ema_actions[env_ids, self.force_size:self.force_size+3] = \
                self.unwrapped.actions[env_ids, 0:3].clone()
        ##########################################################################
        # END TEMPORARY: Base factory position back-calculation match
        ##########################################################################

        # Reset integral state on environment reset
        self.force_integral_error[env_ids] = 0.0

        # Reset per-step tracking states
        if self._first_step_set:
            # Reset to current state values for reset environments
            self._prev_step_pos[env_ids] = self.unwrapped.fingertip_midpoint_pos[env_ids]
            self._prev_step_vel[env_ids] = self.unwrapped.fingertip_midpoint_linvel[env_ids]
            self._prev_step_force[env_ids] = self.robot_force_torque[env_ids, :3]

    def _reset_integral_on_contact_change(self, prev_contact, curr_contact):
        """Reset integral when contact state changes (anti-windup protection)."""
        if not self.enable_force_integral:
            return

        # Check each axis for contact state change
        contact_changed = prev_contact != curr_contact
        # Reset any env where any axis changed contact state
        env_changed = contact_changed.any(dim=1)
        if env_changed.any():
            self.force_integral_error[env_changed] = 0.0

    def _apply_ema_to_actions(self, action):
        """Apply EMA to actions, with special handling for selection terms."""
        # Selection actions (handle no_sel_ema flag)
        sel_actions = action[:, :self.force_size]
        if self.no_sel_ema:
            self.ema_actions[:, :self.force_size] = sel_actions
        else:
            self.ema_actions[:, :self.force_size] = (
                self.ema_factor * sel_actions +
                (1 - self.ema_factor) * self.ema_actions[:, :self.force_size]
            )

        # Position, rotation, and force actions (combined into one operation)
        # All use the same EMA factor, so we can apply it in one go
        if self.apply_ema_force:
            # apply ema to both force and pose commands
            pose_force_start = self.force_size
            pose_force_end = 2 * self.force_size + 6
        else:
            # apply ema to only pose commands
            pose_force_start = self.force_size
            pose_force_end = self.force_size + 6
            self.ema_actions[:, pose_force_end:] = action[:, pose_force_end:]

        self.ema_actions[:, pose_force_start:pose_force_end] = (
            self.ema_factor * action[:, pose_force_start:pose_force_end] +
            (1 - self.ema_factor) * self.ema_actions[:, pose_force_start:pose_force_end]
        )


        # Log raw network outputs
        if hasattr(self.unwrapped, 'extras'):
            # Raw selection actions (clone to prevent reference issues)
            self.unwrapped.extras['to_log']['Network Output / Raw Sel Action X'] = sel_actions[:, 0].clone()
            self.unwrapped.extras['to_log']['Network Output / Raw Sel Action Y'] = sel_actions[:, 1].clone()
            self.unwrapped.extras['to_log']['Network Output / Raw Sel Action Z'] = sel_actions[:, 2].clone()

            # Raw position actions (clone to prevent reference issues)
            raw_pos_actions = action[:, self.force_size:self.force_size+3]
            self.unwrapped.extras['to_log']['Network Output / Raw Pos Action X'] = raw_pos_actions[:, 0].abs().clone()
            self.unwrapped.extras['to_log']['Network Output / Raw Pos Action Y'] = raw_pos_actions[:, 1].abs().clone()
            self.unwrapped.extras['to_log']['Network Output / Raw Pos Action Z'] = raw_pos_actions[:, 2].abs().clone()

            # Raw force actions (clone to prevent reference issues)
            raw_force_actions = action[:, self.force_size+6:2*self.force_size+6]
            self.unwrapped.extras['to_log']['Network Output / Raw Force Action X'] = raw_force_actions[:, 0].abs().clone()
            self.unwrapped.extras['to_log']['Network Output / Raw Force Action Y'] = raw_force_actions[:, 1].abs().clone()
            self.unwrapped.extras['to_log']['Network Output / Raw Force Action Z'] = raw_force_actions[:, 2].abs().clone()

            # EMA'd position actions (clone to prevent reference issues)
            ema_pos = self.ema_actions[:, self.force_size:self.force_size+3]
            self.unwrapped.extras['to_log']['Network Output / EMA Pos Action X'] = ema_pos[:, 0].abs().clone()
            self.unwrapped.extras['to_log']['Network Output / EMA Pos Action Y'] = ema_pos[:, 1].abs().clone()
            self.unwrapped.extras['to_log']['Network Output / EMA Pos Action Z'] = ema_pos[:, 2].abs().clone()

            # EMA'd force actions (clone to prevent reference issues)
            ema_force = self.ema_actions[:, self.force_size+6:2*self.force_size+6]
            self.unwrapped.extras['to_log']['Network Output / EMA Force Action X'] = ema_force[:, 0].abs().clone()
            self.unwrapped.extras['to_log']['Network Output / EMA Force Action Y'] = ema_force[:, 1].abs().clone()
            self.unwrapped.extras['to_log']['Network Output / EMA Force Action Z'] = ema_force[:, 2].abs().clone()

            #if not self.no_sel_ema:
            # EMA'd selection actions (clone to prevent reference issues)
            ema_sel = self.ema_actions[:, :self.force_size]
            self.unwrapped.extras['to_log']['Network Output / EMA Sel Action X'] = ema_sel[:, 0].clone()
            self.unwrapped.extras['to_log']['Network Output / EMA Sel Action Y'] = ema_sel[:, 1].clone()
            self.unwrapped.extras['to_log']['Network Output / EMA Sel Action Z'] = ema_sel[:, 2].clone()

    def _compute_control_targets(self):
        """Compute control targets from current state + EMA actions (Isaac Lab style)."""
        # 1. Selection matrix from EMA'd selection actions
        self.sel_matrix[:, :self.force_size] = torch.where(
            self.ema_actions[:, :self.force_size] > 0.5, 1.0, 0.0
        )

        # Log selection matrix
        if hasattr(self.unwrapped, 'extras'):
            sel_mat = self.sel_matrix[:,:self.force_size].clone()
            self.unwrapped.extras['to_log']['Control Mode / Force Control X'] = sel_mat[:, 0]
            self.unwrapped.extras['to_log']['Control Mode / Force Control Y'] = sel_mat[:, 1]
            self.unwrapped.extras['to_log']['Control Mode / Force Control Z'] = sel_mat[:, 2]
            if self.force_size > 3:
                self.unwrapped.extras['to_log']['Control Mode / Force Control RX'] = sel_mat[:, 3]
                self.unwrapped.extras['to_log']['Control Mode / Force Control RY'] = sel_mat[:, 4]
                self.unwrapped.extras['to_log']['Control Mode / Force Control RZ'] = sel_mat[:, 5]

            # Log 4-case control mode statistics for precision/recall analysis
            in_contact = self.unwrapped.in_contact[:, :self.force_size]
            force_control = self.sel_matrix[:, :self.force_size] > 0.5
            pos_control = self.sel_matrix[:, :self.force_size] < 0.5

            # Case 1: In contact + Force control
            case_contact_force = torch.logical_and(in_contact, force_control).float()
            # Case 2: In contact + Position control
            case_contact_pos = torch.logical_and(in_contact, pos_control).float()
            # Case 3: No contact + Force control
            case_no_contact_force = torch.logical_and(~in_contact, force_control).float()
            # Case 4: No contact + Position control
            case_no_contact_pos = torch.logical_and(~in_contact, pos_control).float()

            # Log translational axes
            self.unwrapped.extras['to_log']['Control Mode / Contact+Force X summed_total'] = case_contact_force[:, 0]
            self.unwrapped.extras['to_log']['Control Mode / Contact+Pos X summed_total'] = case_contact_pos[:, 0]
            self.unwrapped.extras['to_log']['Control Mode / NoContact+Force X summed_total'] = case_no_contact_force[:, 0]
            self.unwrapped.extras['to_log']['Control Mode / NoContact+Pos X summed_total'] = case_no_contact_pos[:, 0]

            self.unwrapped.extras['to_log']['Control Mode / Contact+Force Y summed_total'] = case_contact_force[:, 1]
            self.unwrapped.extras['to_log']['Control Mode / Contact+Pos Y summed_total'] = case_contact_pos[:, 1]
            self.unwrapped.extras['to_log']['Control Mode / NoContact+Force Y summed_total'] = case_no_contact_force[:, 1]
            self.unwrapped.extras['to_log']['Control Mode / NoContact+Pos Y summed_total'] = case_no_contact_pos[:, 1]

            self.unwrapped.extras['to_log']['Control Mode / Contact+Force Z summed_total'] = case_contact_force[:, 2]
            self.unwrapped.extras['to_log']['Control Mode / Contact+Pos Z summed_total'] = case_contact_pos[:, 2]
            self.unwrapped.extras['to_log']['Control Mode / NoContact+Force Z summed_total'] = case_no_contact_force[:, 2]
            self.unwrapped.extras['to_log']['Control Mode / NoContact+Pos Z summed_total'] = case_no_contact_pos[:, 2]

            # Log rotational axes if controlling torques
            if self.force_size > 3:
                self.unwrapped.extras['to_log']['Control Mode / Contact+Force RX summed_total'] = case_contact_force[:, 3]
                self.unwrapped.extras['to_log']['Control Mode / Contact+Pos RX summed_total'] = case_contact_pos[:, 3]
                self.unwrapped.extras['to_log']['Control Mode / NoContact+Force RX summed_total'] = case_no_contact_force[:, 3]
                self.unwrapped.extras['to_log']['Control Mode / NoContact+Pos RX summed_total'] = case_no_contact_pos[:, 3]

                self.unwrapped.extras['to_log']['Control Mode / Contact+Force RY summed_total'] = case_contact_force[:, 4]
                self.unwrapped.extras['to_log']['Control Mode / Contact+Pos RY summed_total'] = case_contact_pos[:, 4]
                self.unwrapped.extras['to_log']['Control Mode / NoContact+Force RY summed_total'] = case_no_contact_force[:, 4]
                self.unwrapped.extras['to_log']['Control Mode / NoContact+Pos RY summed_total'] = case_no_contact_pos[:, 4]

                self.unwrapped.extras['to_log']['Control Mode / Contact+Force RZ summed_total'] = case_contact_force[:, 5]
                self.unwrapped.extras['to_log']['Control Mode / Contact+Pos RZ summed_total'] = case_contact_pos[:, 5]
                self.unwrapped.extras['to_log']['Control Mode / NoContact+Force RZ summed_total'] = case_no_contact_force[:, 5]
                self.unwrapped.extras['to_log']['Control Mode / NoContact+Pos RZ summed_total'] = case_no_contact_pos[:, 5]

        # 2. Position target (Isaac Lab style: current_pos + scaled_action)
        pos_actions = self.ema_actions[:, self.force_size:self.force_size+3] * self.unwrapped.pos_threshold
        pos_target = self.unwrapped.fingertip_midpoint_pos + pos_actions

        # Apply bounds constraint
        delta_pos = pos_target - self.unwrapped.fixed_pos_action_frame
        # Use class variable (per-env tensor shape: num_envs, 3 for symmetric bounds)
        pos_error_clipped = torch.clip(delta_pos, -self.pos_bounds, self.pos_bounds)
        self.unwrapped.ctrl_target_fingertip_midpoint_pos = self.unwrapped.fixed_pos_action_frame + pos_error_clipped

        # Log position goal norm
        if hasattr(self.unwrapped, 'extras'):
            pos_goal_norm = torch.norm(self.unwrapped.ctrl_target_fingertip_midpoint_pos, p=2, dim=-1)
            self.unwrapped.extras['to_log']['Control Target / Pos Goal Norm'] = pos_goal_norm
            pos_goal = self.unwrapped.ctrl_target_fingertip_midpoint_pos
            self.unwrapped.extras['to_log']['Control Target / Pos X Goal'] = torch.abs(pos_goal[:, 0])
            self.unwrapped.extras['to_log']['Control Target / Pos Y Goal'] = torch.abs(pos_goal[:, 1])
            self.unwrapped.extras['to_log']['Control Target / Pos Z Goal'] = torch.abs(pos_goal[:, 2])

        # 3. Rotation target (Isaac Lab style)
        rot_actions = self.ema_actions[:, self.force_size+3:self.force_size+6]

        # Handle unidirectional rotation if configured
        if getattr(self.unwrapped.cfg_task, 'unidirectional_rot', False):
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5

        rot_actions = rot_actions * self.unwrapped.rot_threshold

        # Convert to quaternion
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / (angle.unsqueeze(-1) + 1e-6)

        rot_actions_quat = self.torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        quat_target = self.torch_utils.quat_mul(rot_actions_quat, self.unwrapped.fingertip_midpoint_quat)

        # Restrict to upright orientation
        target_euler_xyz = torch.stack(self.torch_utils.get_euler_xyz(quat_target), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Roll
        target_euler_xyz[:, 1] = 0.0      # Pitch

        self.unwrapped.ctrl_target_fingertip_midpoint_quat = self.torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        # 4. Force target computation
        # Two modes controlled by use_delta_force config:
        # - Delta mode (use_delta_force=True): target = action * threshold + current_force
        # - Absolute mode (use_delta_force=False): target = action * bounds
        force_actions = self.ema_actions[:, self.force_size+6:2*self.force_size+6]

        self.target_force_for_control = torch.zeros((self.num_envs, 6), device=self.device)

        if self.use_delta_force:
            # Delta mode: target relative to current measured force
            force_delta = force_actions[:, :3] * self.force_threshold
            self.target_force_for_control[:, :3] = torch.clip(
                force_delta + self.robot_force_torque[:, :3],
                -self.force_bounds, self.force_bounds
            )
        else:
            # Absolute mode: action directly specifies target force
            # action=0 → target=0N, action=±1 → target=±force_bounds
            self.target_force_for_control[:, :3] = force_actions[:, :3] * self.force_bounds
        
        # makes it so that z control commands cannot be positive
        if self.async_z_bounds:
            self.target_force_for_control[:,2] = (self.target_force_for_control[:,2] - self.force_bounds[:, 2])/2.0


        # Handle torque if enabled
        if self.force_size > 3:
            if self.use_delta_force:
                # Delta mode for torque
                torque_delta = force_actions[:, 3:] * self.torque_threshold / self.force_threshold
                self.target_force_for_control[:, 3:] = torch.clip(
                    torque_delta + self.robot_force_torque[:, 3:],
                    -self.torque_bounds, self.torque_bounds
                )
            else:
                # Absolute mode for torque
                self.target_force_for_control[:, 3:] = force_actions[:, 3:] * self.torque_bounds

        # 5. Update integral state if enabled (for PID force control)
        if self.enable_force_integral:
            force_error = self.target_force_for_control - self.robot_force_torque
            # Accumulate error (will be scaled by Ki in wrench computation)
            self.force_integral_error += force_error
            # Anti-windup clamping
            self.force_integral_error = torch.clamp(
                self.force_integral_error,
                -self.force_integral_clamp,
                self.force_integral_clamp
            )

        # Log force goal norm
        if hasattr(self.unwrapped, 'extras'):
            force_goal_norm = torch.norm(self.target_force_for_control[:, :3], p=2, dim=-1)
            self.unwrapped.extras['to_log']['Control Target / Force Goal Norm'] = force_goal_norm
            force_goal = self.target_force_for_control[:, :3]
            self.unwrapped.extras['to_log']['Control Target / Force X Goal'] = torch.abs(force_goal[:, 0])
            self.unwrapped.extras['to_log']['Control Target / Force Y Goal'] = torch.abs(force_goal[:, 1])
            self.unwrapped.extras['to_log']['Control Target / Force Z Goal'] = torch.abs(force_goal[:, 2])

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
            is_force_control = self.sel_matrix[:, i] > 0.5
            is_pos_control = self.sel_matrix[:, i] <= 0.5

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

    def _wrapped_reset_idx(self, env_ids):
        """Track early terminations by control mode before resetting."""
        # Convert to tensor if needed
        if not isinstance(env_ids, torch.Tensor):    
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # FIRST: Track early terminations (always, whether partial or full reset)
        if hasattr(self.unwrapped, 'episode_length_buf'):
            episode_lengths = self.unwrapped.episode_length_buf[env_ids]
            max_length = self.unwrapped.max_episode_length - 1

            # Find environments that terminated early (not timeout)
            early_term_mask = episode_lengths < max_length
            early_term_env_ids = env_ids[early_term_mask]

            # Track control mode at termination for each early-terminated environment
            for env_id in early_term_env_ids:
                env_idx = env_id.item()
                agent_id = env_idx // self.envs_per_agent

                # Check control mode for each axis (sel_matrix > 0.5 = force control)
                for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
                    if self.sel_matrix[env_idx, axis_idx] > 0.5:
                        # Force control was active on this axis
                        self.termination_counts[f'force_{axis_name}'][agent_id] += 1
                    else:
                        # Position control was active on this axis
                        self.termination_counts[f'pos_{axis_name}'][agent_id] += 1

        # SECOND: If full reset, log and reset counts
        if len(env_ids) == self.num_envs:
            # Full reset - truncation period ended
            # Log cumulative termination counts per agent (including current reset)
            if hasattr(self.unwrapped, 'extras'):
                for axis in ['x', 'y', 'z']:
                    axis_upper = axis.upper()
                    # Convert lists to tensors for logging
                    force_counts = torch.tensor(self.termination_counts[f'force_{axis}'],
                                                 dtype=torch.float32, device=self.device)
                    pos_counts = torch.tensor(self.termination_counts[f'pos_{axis}'],
                                               dtype=torch.float32, device=self.device)

                    self.unwrapped.extras['to_log'][f'Termination / Force Control {axis_upper} Count'] = force_counts
                    self.unwrapped.extras['to_log'][f'Termination / Position Control {axis_upper} Count'] = pos_counts

            # Reset termination counts for the next truncation period
            self._reset_termination_counts()

        # Call original _reset_idx to maintain wrapper chain
        if self._original_reset_idx is not None:
            self._original_reset_idx(env_ids)

    def _reset_termination_counts(self):
        """Reset termination counts for new truncation period."""
        for key in self.termination_counts:
            self.termination_counts[key] = [0] * self.num_agents

    def _get_target_out_of_bounds(self):
        """Check if fingertip target is out of position bounds."""
        delta = self.unwrapped.fingertip_midpoint_pos - self.unwrapped.fixed_pos_action_frame

        # Use class variable (per-env tensor for symmetric bounds)
        out_of_bounds = torch.logical_or(
            delta <= -self.pos_bounds,
            delta >= self.pos_bounds
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
        """
        if self.sel_matrix[0,2]:
            self.zero_z = True
        fake_target = self.unwrapped.fixed_pos + self.unwrapped.scene.env_origins
        try:
            if self.zero_z:
                fake_target[0,2] = 0
        except:
            self.zero_z = False
        fake_target[:,0] = self.unwrapped.fingertip_midpoint_pos[:,0]
        """
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

        # Compute force wrench using filtered targets (with optional PID control)
        force_wrench = compute_force_task_wrench(
            cfg=self.unwrapped.cfg,
            dof_pos=self.unwrapped.joint_pos,
            eef_force=self.robot_force_torque,
            fingertip_midpoint_linvel=self.unwrapped.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.unwrapped.fingertip_midpoint_angvel,
            ctrl_target_force=self.target_force_for_control,
            task_gains=self.kp,
            task_deriv_gains=self.unwrapped.task_deriv_gains,
            device=self.unwrapped.device,
            # PID control parameters
            task_integ_gains=self.ki if self.enable_force_integral else None,
            force_integral_error=self.force_integral_error if self.enable_force_integral else None,
            enable_derivative=self.enable_force_derivative,
            enable_integral=self.enable_force_integral,
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

        # Zero wrench components that would drive further out of bounds
        # Use class variable (per-env tensor for symmetric bounds)
        for i in range(3):  # x, y, z
            # If at negative bound and wrench would push more negative, zero it
            at_neg_bound = delta_pos[:, i] <= -self.pos_bounds[:, i]
            neg_wrench = task_wrench[:, i] < 0
            task_wrench[:, i] = torch.where(at_neg_bound & neg_wrench, 0.0, task_wrench[:, i])

            # If at positive bound and wrench would push more positive, zero it
            at_pos_bound = delta_pos[:, i] >= self.pos_bounds[:, i]
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

        # Store task wrench for external access (e.g., plotting/debugging)
        self.task_wrench = task_wrench.clone()

        # Set gripper targets
        self.unwrapped.ctrl_target_joint_pos[:, 7:9] = self.unwrapped.ctrl_target_gripper_dof_pos
        self.unwrapped.joint_torque[:, 7:9] = 0.0

        # Apply torques
        self.unwrapped._robot.set_joint_position_target(self.unwrapped.ctrl_target_joint_pos)
        self.unwrapped._robot.set_joint_effort_target(self.unwrapped.joint_torque)

    def _wrapped_get_rewards(self):
        """Get rewards with hybrid control rewards added."""
        # Get base rewards from original method
        base_rewards = self._original_get_rewards() if self._original_get_rewards else torch.zeros(self.num_envs, device=self.device)

        # Add hybrid control specific rewards
        if self.reward_type == "simp":
            base_rewards += self._simple_force_reward()
        elif self.reward_type == "dirs":
            base_rewards += self._directional_force_reward()
        elif self.reward_type == "delta":
            base_rewards += self._delta_selection_reward()
        elif self.reward_type == "pos_simp":
            base_rewards += self._position_simple_reward()
        elif self.reward_type == "wrench_norm":
            base_rewards += self._low_wrench_reward()
        elif self.reward_type == "force_z":
            base_rewards += self._force_z_reward()

        return base_rewards

    def _simple_force_reward(self):
        """Simple force activity reward."""
        force_ctrl = (self.sel_matrix[:, :self.force_size] > 0.5) #.bool()

        good_force_cmd = torch.logical_and(
            torch.any(self.unwrapped.in_contact[:, :self.force_size], dim=1), 
            torch.any(force_ctrl, dim=1)
        )

        bad_force_cmd = torch.logical_and(
            torch.all(~self.unwrapped.in_contact[:, :self.force_size], dim=1), 
            torch.any(force_ctrl, dim=1)
        )

        sel_rew = self.task_cfg.good_force_cmd_rew * good_force_cmd + self.task_cfg.bad_force_cmd_rew * bad_force_cmd

        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['logs_rew_Selection Matrix'] = sel_rew

        return sel_rew

    def _directional_force_reward(self):
        """Direction-specific force reward."""
        force_ctrl = (self.sel_matrix[:, :self.force_size] > 0.5) #.bool()

        good_dims = torch.logical_and(
            force_ctrl, 
            self.unwrapped.in_contact[:, :self.force_size]) | torch.logical_and(~force_ctrl, ~self.unwrapped.in_contact[:, :self.force_size])
        bad_dims = torch.logical_and(
            force_ctrl, 
            ~self.unwrapped.in_contact[:, :self.force_size]) | torch.logical_and(~force_ctrl, self.unwrapped.in_contact[:, :self.force_size])

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
        force_ctrl = self.sel_matrix[:, :self.force_size] > 0.5 #.bool()

        good_force_cmd = torch.logical_and(
            torch.any(self.unwrapped.in_contact[:, :self.force_size], dim=1), 
            torch.any(force_ctrl, dim=1)
        )
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

    def _force_z_reward(self):
        """A reward for debugging which punishes not using force in z, and not using pos in xy"""
        x_error = 1.0-self.ema_actions[:, 2]
        y_error = 1.0-self.ema_actions[:, 2]
        z_error = self.ema_actions[:, 2]
        sel_rew = self.task_cfg.good_force_cmd_rew * (x_error + y_error + z_error)/3.0

        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['logs_rew_Selection Matrix'] = sel_rew

        return sel_rew
    
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