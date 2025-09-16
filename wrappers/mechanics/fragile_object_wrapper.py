"""
Fragile Object Wrapper for Factory Environment

This wrapper adds fragile object functionality where objects break under excessive force,
with support for different break forces per agent group in multi-agent scenarios.

Extracted from: factory_env.py lines 57-70, 560-564
"""

import torch
import gymnasium as gym


class FragileObjectWrapper(gym.Wrapper):
    """
    Wrapper that adds fragile object functionality to environments.

    Features:
    - Objects break when force exceeds threshold
    - Support for different break forces per environment group
    - Static agent assignment for multi-agent scenarios
    - Force violation detection and episode termination
    """

    def __init__(self, env, break_force, num_agents=1):
        """
        Initialize the fragile object wrapper.

        Args:
            env: Base environment to wrap
            break_force: Force threshold(s) for breaking objects
                        - Single float: Same threshold for all environments
                        - List of floats: Different threshold per agent group
                        - Use -1 in list for unbreakable (very high threshold)
            num_agents: Number of agents for static environment assignment
        """
        super().__init__(env)

        self.num_agents = num_agents
        self.num_envs = env.unwrapped.num_envs

        # Validate num_agents
        if self.num_envs % self.num_agents != 0:
            raise ValueError(f"Number of environments ({self.num_envs}) must be divisible by number of agents ({self.num_agents})")

        self.envs_per_agent = self.num_envs // self.num_agents

        # Initialize break force tensor
        device = env.unwrapped.device
        self.break_force = torch.ones((self.num_envs,), dtype=torch.float32, device=device)

        # Configure break forces
        if isinstance(break_force, list):
            if len(break_force) != self.num_agents:
                raise ValueError(f"Break force list length ({len(break_force)}) must match number of agents ({self.num_agents})")

            # Assign break forces per agent group
            for i, force in enumerate(break_force):
                start_idx = i * self.envs_per_agent
                end_idx = (i + 1) * self.envs_per_agent

                if force == -1:
                    # Unbreakable - set very high threshold
                    self.break_force[start_idx:end_idx] *= 2**23
                else:
                    self.break_force[start_idx:end_idx] *= force
        else:
            # Single value for all environments
            self.break_force *= break_force

        # Determine if any objects are fragile
        self.fragile = torch.any(self.break_force < 2**20)  # Threshold to distinguish "unbreakable"

        # Store original methods
        self._original_get_dones = None

        # Flag to track if wrapper is initialized
        self._wrapper_initialized = False

        # Initialize wrapper if environment is ready
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def _initialize_wrapper(self):
        """Initialize the wrapper after the base environment is set up."""
        if self._wrapper_initialized:
            return

        # Store and override _get_dones method
        if hasattr(self.unwrapped, '_get_dones'):
            self._original_get_dones = self.unwrapped._get_dones
            self.unwrapped._get_dones = self._wrapped_get_dones

        self._wrapper_initialized = True

    def _wrapped_get_dones(self):
        """Check for episode termination including force violations."""
        # Get original termination conditions
        if self._original_get_dones:
            terminated, time_out = self._original_get_dones()
        else:
            # Fallback if original method doesn't exist
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.unwrapped.device)
            time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.unwrapped.device)

        # Check for force violations if objects are fragile
        if self.fragile and self._has_force_torque_data():
            force_magnitude = torch.linalg.norm(self.unwrapped.robot_force_torque[:, :3], axis=1)
            force_violations = force_magnitude >= self.break_force
            terminated = torch.logical_or(terminated, force_violations)

        return terminated, time_out

    def _has_force_torque_data(self):
        """Check if force-torque data is available from a force-torque wrapper."""
        return hasattr(self.unwrapped, 'robot_force_torque')

    def step(self, action):
        """Step the environment and ensure wrapper is initialized."""
        # Initialize wrapper if not done yet
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        return super().step(action)

    def reset(self, **kwargs):
        """Reset the environment and ensure wrapper is initialized."""
        obs, info = super().reset(**kwargs)

        # Initialize wrapper if not done yet
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        return obs, info

    def get_agent_assignment(self):
        """Get the environment indices assigned to each agent."""
        assignments = {}
        for i in range(self.num_agents):
            start_idx = i * self.envs_per_agent
            end_idx = (i + 1) * self.envs_per_agent
            assignments[i] = list(range(start_idx, end_idx))
        return assignments

    def get_break_forces(self):
        """Get the break force thresholds for all environments."""
        return self.break_force.clone()

    def get_agent_break_force(self, agent_id):
        """Get the break force threshold for a specific agent's environments."""
        if agent_id >= self.num_agents:
            raise ValueError(f"Agent ID ({agent_id}) must be less than number of agents ({self.num_agents})")

        start_idx = agent_id * self.envs_per_agent
        end_idx = (agent_id + 1) * self.envs_per_agent
        return self.break_force[start_idx:end_idx].clone()

    def is_fragile(self):
        """Check if any objects in the environment are fragile."""
        return self.fragile

    def get_force_violations(self):
        """Get current force violations if force-torque data is available."""
        if self.fragile and self._has_force_torque_data():
            force_magnitude = torch.linalg.norm(self.unwrapped.robot_force_torque[:, :3], axis=1)
            return force_magnitude >= self.break_force
        else:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.unwrapped.device)