"""
Pose Contact Logging Wrapper

This wrapper logs action effects and controller outputs for pose-only control mode.
Provides consistent metrics matching the hybrid control wrapper for comparison.

Logs:
- Network output (raw actions)
- Control targets (goals after EMA/bounds)
- Controller output (errors and wrenches)
- Action effects (per-step deltas of pos/vel/force)
"""

import torch
import gymnasium as gym
from wrappers.control.factory_control_utils import compute_pose_task_wrench

try:
    from isaacsim.core.api.robots import RobotView
except ImportError:
    try:
        from omni.isaac.core.articulations import ArticulationView as RobotView
    except ImportError:
        RobotView = None


class PoseContactLoggingWrapper(gym.Wrapper):
    """
    Wrapper for logging pose control metrics and action effects.

    Features:
    - Logs raw network outputs before scaling
    - Logs control targets (position goals)
    - Logs controller outputs (position errors, pose wrenches)
    - Logs per-step action effects (position, velocity, force changes)

    All metrics aggregated per-agent for multi-agent setups.
    """

    def __init__(self, env, num_agents=1):
        """
        Initialize pose contact logging wrapper.

        Args:
            env: Base environment to wrap
            num_agents: Number of agents for per-agent metric aggregation
        """
        super().__init__(env)

        self.num_agents = num_agents
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        if self.num_envs % self.num_agents != 0:
            raise ValueError(
                f"Number of environments ({self.num_envs}) must be divisible "
                f"by number of agents ({self.num_agents})"
            )

        self.envs_per_agent = self.num_envs // self.num_agents

        # Per-step tracking for action effect metrics
        self._prev_step_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_step_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_step_force = torch.zeros((self.num_envs, 3), device=self.device)
        self._first_step_set = False

        # Flag to track if wrapper is initialized
        self._wrapper_initialized = False

        # Store original methods
        self._original_pre_physics_step = None
        self._original_apply_action = None

        # Check if force-torque sensor is available
        self._has_force_sensor = self._check_force_sensor()

        # Initialize wrapper
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        if hasattr(self.unwrapped, 'extras'):
            if 'to_log' not in self.unwrapped.extras.keys():
                self.unwrapped.extras['to_log'] = {}

    def _check_force_sensor(self):
        """Check if force-torque sensor is available in wrapper chain."""
        current_env = self.env

        while current_env is not None:
            if hasattr(current_env, 'has_force_torque_sensor'):
                return current_env.has_force_torque_sensor
            # Move up the wrapper chain
            if hasattr(current_env, 'env'):
                current_env = current_env.env
            elif hasattr(current_env, 'unwrapped'):
                current_env = current_env.unwrapped
                break
            else:
                break

        return False

    @property
    def robot_force_torque(self):
        """Access force-torque data if available."""
        if not self._has_force_sensor:
            return None

        # Access force-torque data from unwrapped environment
        if hasattr(self.unwrapped, 'robot_force_torque'):
            return self.unwrapped.robot_force_torque
        return None

    def _initialize_wrapper(self):
        """Initialize wrapper by overriding environment methods."""
        if self._wrapper_initialized:
            return

        # Store and override methods
        if hasattr(self.unwrapped, '_pre_physics_step'):
            self._original_pre_physics_step = self.unwrapped._pre_physics_step
            self.unwrapped._pre_physics_step = self._wrapped_pre_physics_step

        if hasattr(self.unwrapped, '_apply_action'):
            self._original_apply_action = self.unwrapped._apply_action
            self.unwrapped._apply_action = self._wrapped_apply_action

        self._wrapper_initialized = True

    def _wrapped_pre_physics_step(self, action):
        """Process actions and log network outputs and control targets."""
        # Handle reset environments first
        env_ids = self.unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_tracking(env_ids)

        # Call original pre_physics_step if it exists
        if self._original_pre_physics_step:
            self._original_pre_physics_step(action)

        # Log network outputs (raw actions before scaling)
        if hasattr(self.unwrapped, 'extras'):
            raw_pos_actions = action[:, :3]
            self.unwrapped.extras['to_log']['Network Output / Raw Pos Action X'] = raw_pos_actions[:, 0]
            self.unwrapped.extras['to_log']['Network Output / Raw Pos Action Y'] = raw_pos_actions[:, 1]
            self.unwrapped.extras['to_log']['Network Output / Raw Pos Action Z'] = raw_pos_actions[:, 2]

        # Log control targets (after EMA and bounds, if applicable)
        self._log_control_targets()

        # Log action effect metrics
        self._log_action_effect_metrics()

    def _wrapped_apply_action(self):
        """Apply action and log controller outputs."""
        # Call original apply_action
        if self._original_apply_action:
            self._original_apply_action()

        # Log controller outputs
        self._log_controller_outputs()

    def _reset_tracking(self, env_ids):
        """Reset tracking states for reset environments."""
        if self._first_step_set:
            self._prev_step_pos[env_ids] = self.unwrapped.fingertip_midpoint_pos[env_ids]
            self._prev_step_vel[env_ids] = self.unwrapped.ee_linvel_fd[env_ids]
            if self._has_force_sensor and self.robot_force_torque is not None:
                self._prev_step_force[env_ids] = self.robot_force_torque[env_ids, :3]

    def _log_control_targets(self):
        """Log control target metrics."""
        if not hasattr(self.unwrapped, 'extras'):
            return

        # Log position goal norm (L2 norm of target position)
        if hasattr(self.unwrapped, 'ctrl_target_fingertip_midpoint_pos'):
            pos_goal_norm = torch.norm(
                self.unwrapped.ctrl_target_fingertip_midpoint_pos,
                p=2, dim=-1
            )
            self.unwrapped.extras['to_log']['Control Target / Pos Goal Norm'] = pos_goal_norm

    def _log_controller_outputs(self):
        """Log controller output metrics (errors and wrenches)."""
        if not hasattr(self.unwrapped, 'extras'):
            return

        # Log position errors
        if hasattr(self.unwrapped, 'ctrl_target_fingertip_midpoint_pos'):
            position_error = (
                self.unwrapped.ctrl_target_fingertip_midpoint_pos -
                self.unwrapped.fingertip_midpoint_pos
            )
            self.unwrapped.extras['to_log']['Controller Output / Position Error X'] = position_error[:, 0]
            self.unwrapped.extras['to_log']['Controller Output / Position Error Y'] = position_error[:, 1]
            self.unwrapped.extras['to_log']['Controller Output / Position Error Z'] = position_error[:, 2]

        # Log pose wrenches (aggregated per agent)
        # Note: We would need access to the wrench calculation from factory_env
        # For now, we'll approximate using joint torques if available
        if hasattr(self.unwrapped, 'joint_torque'):
            self._log_pose_wrenches()

    def _log_pose_wrenches(self):
        """Log pose control wrenches aggregated by agent."""
        # Check if we have all required attributes
        required_attrs = [
            'ctrl_target_fingertip_midpoint_pos',
            'ctrl_target_fingertip_midpoint_quat',
            'fingertip_midpoint_pos',
            'fingertip_midpoint_quat',
            'ee_linvel_fd',
            'ee_angvel_fd',
            'joint_pos',
            'task_prop_gains',
            'task_deriv_gains'
        ]

        for attr in required_attrs:
            if not hasattr(self.unwrapped, attr):
                return

        # Compute pose wrench using factory control utils (same as hybrid wrapper)
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

        # Aggregate by agent (same pattern as hybrid wrapper)
        axis_names = ['X', 'Y', 'Z']
        for i in range(3):
            pose_wrench_per_agent = torch.full((self.num_agents,), float('nan'), device=self.device)

            for agent_id in range(self.num_agents):
                start_idx = agent_id * self.envs_per_agent
                end_idx = (agent_id + 1) * self.envs_per_agent

                # Average absolute wrench for this agent's environments
                pose_wrench_per_agent[agent_id] = pose_wrench[start_idx:end_idx, i].abs().mean()

            # Only log if at least one agent has a non-NaN value
            if not torch.all(torch.isnan(pose_wrench_per_agent)):
                self.unwrapped.extras['to_log'][f'Controller Output / Pose Wrench {axis_names[i]}'] = pose_wrench_per_agent

    def _log_action_effect_metrics(self):
        """Log per-step changes in position, velocity, and force."""
        if not hasattr(self.unwrapped, 'extras'):
            return

        # Get current state
        current_pos = self.unwrapped.fingertip_midpoint_pos
        current_vel = self.unwrapped.ee_linvel_fd
        current_force = None
        if self._has_force_sensor and self.robot_force_torque is not None:
            current_force = self.robot_force_torque[:, :3]

        # First time initialization
        if not self._first_step_set:
            self._prev_step_pos.copy_(current_pos)
            self._prev_step_vel.copy_(current_vel)
            if current_force is not None:
                self._prev_step_force.copy_(current_force)
            self._first_step_set = True
            return

        # Calculate per-step deltas
        pos_delta = current_pos - self._prev_step_pos
        vel_delta = current_vel - self._prev_step_vel
        force_delta = None
        if current_force is not None:
            force_delta = current_force - self._prev_step_force

        # Log metrics aggregated by agent (all attributed to pose control)
        axis_names = ['X', 'Y', 'Z']
        for i in range(3):
            # Create per-agent tensors
            pose_pos_change = torch.full((self.num_agents,), float('nan'), device=self.device)
            pose_vel_change = torch.full((self.num_agents,), float('nan'), device=self.device)
            pose_force_change = torch.full((self.num_agents,), float('nan'), device=self.device)

            # Aggregate by agent
            for agent_id in range(self.num_agents):
                start_idx = agent_id * self.envs_per_agent
                end_idx = (agent_id + 1) * self.envs_per_agent

                # Average absolute deltas for this agent
                pose_pos_change[agent_id] = pos_delta[start_idx:end_idx, i].abs().mean()
                pose_vel_change[agent_id] = vel_delta[start_idx:end_idx, i].abs().mean()
                if force_delta is not None:
                    pose_force_change[agent_id] = force_delta[start_idx:end_idx, i].abs().mean()

            # Log pose control metrics
            if not torch.all(torch.isnan(pose_pos_change)):
                self.unwrapped.extras['to_log'][f'Action Effect / Pose Control Pos Change {axis_names[i]}'] = pose_pos_change
            if not torch.all(torch.isnan(pose_vel_change)):
                self.unwrapped.extras['to_log'][f'Action Effect / Pose Control Vel Change {axis_names[i]}'] = pose_vel_change
            if force_delta is not None and not torch.all(torch.isnan(pose_force_change)):
                self.unwrapped.extras['to_log'][f'Action Effect / Pose Control Force Change {axis_names[i]}'] = pose_force_change

        # Update previous step state for next iteration
        self._prev_step_pos.copy_(current_pos)
        self._prev_step_vel.copy_(current_vel)
        if current_force is not None:
            self._prev_step_force.copy_(current_force)

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
