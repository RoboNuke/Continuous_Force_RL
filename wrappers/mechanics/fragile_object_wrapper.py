"""
Fragile Object Wrapper for Factory Environment

This wrapper adds fragile object functionality where objects break under excessive force,
with support for different break forces per agent group in multi-agent scenarios.

The wrapper monitors force sensor data and automatically terminates episodes when
force thresholds are exceeded, simulating object fragility in manipulation tasks.

Extracted from: factory_env.py lines 57-70, 560-564
"""

import torch
import gymnasium as gym


class FragileObjectWrapper(gym.Wrapper):
    """
    Wrapper that adds fragile object functionality to manipulation environments.

    This wrapper monitors force sensor data (typically from a force-torque wrapper)
    and automatically terminates episodes when force thresholds are exceeded,
    simulating fragile object breaking in manipulation tasks.

    The wrapper supports multi-agent scenarios with static environment assignment,
    allowing different break force thresholds for different agent groups.

    Key Features:
    - Automatic episode termination when force exceeds break threshold
    - Per-agent break force configuration in multi-agent setups
    - Static environment assignment (environments evenly divided among agents)
    - Lazy initialization to work with wrapper chains
    - Support for unbreakable objects (break_force = -1)

    Args:
        env (gym.Env): Base environment to wrap. Should have force-torque data available
                      via `robot_force_torque` attribute (typically from ForceTorqueWrapper).
        break_force (float or list): Force threshold(s) for breaking objects:
            - Single float: Same threshold for all environments
            - List of floats: Different threshold per agent group (length must equal num_agents)
            - Use -1 for unbreakable objects (sets very high threshold)
        num_agents (int): Number of agents for static environment assignment.
                         Total environments must be divisible by this number.

    Example:
        >>> # Single break force for all environments
        >>> env = FragileObjectWrapper(env, break_force=10.0)

        >>> # Different break forces per agent (2 agents)
        >>> env = FragileObjectWrapper(env, break_force=[5.0, 15.0], num_agents=2)

        >>> # Mixed breakable and unbreakable objects (3 agents)
        >>> env = FragileObjectWrapper(env, break_force=[8.0, -1, 12.0], num_agents=3)

    Note:
        - Requires force-torque data from environment (typically via ForceTorqueWrapper)
        - Uses L2 norm of force vector (first 3 components of robot_force_torque)
        - Environments are statically assigned: Agent 0 gets envs [0:N//k], Agent 1 gets [N//k:2*N//k], etc.
        - Lazy initialization allows wrapper to work even if base environment isn't fully initialized
    """

    def __init__(self, env, break_force, num_agents=1, config=None):
        """
        Initialize the fragile object wrapper.

        Args:
            env (gym.Env): Base environment to wrap. Should have force-torque data available.
            break_force (float or list): Force threshold(s) for breaking objects:
                - Single float: Same threshold for all environments
                - List of floats: Different threshold per agent group (length must equal num_agents)
                - Use -1 for unbreakable objects (sets very high threshold: 2^23)
            num_agents (int): Number of agents for static environment assignment.
                            Total environments must be divisible by this number.
            config (dict): Configuration dictionary containing wrapper parameters including peg_break_rew.

        Raises:
            ValueError: If num_envs is not divisible by num_agents, or if break_force list
                       length doesn't match num_agents.
        """
        super().__init__(env)

        self.num_agents = num_agents
        self.num_envs = env.unwrapped.num_envs
        self.config = config or {}

        # Extract peg break reward configuration
        self.enabled = self.config.enabled
        self.peg_break_rew = self.config.peg_break_rew

        # Validate num_agents
        if self.num_envs % self.num_agents != 0:
            raise ValueError(f"Number of environments ({self.num_envs}) must be divisible by number of agents ({self.num_agents})")

        self.envs_per_agent = self.num_envs // self.num_agents

        # Initialize break force tensor
        device = env.unwrapped.device
        break_force = break_force if type(break_force) == list else [break_force]
        self.break_force = torch.ones((self.num_envs,), dtype=torch.float32, device=device)


        envs_per_break_force = self.num_envs // len(break_force)

        for i, force in enumerate(break_force):
            start_idx = i * envs_per_break_force
            end_idx = (i+1) * envs_per_break_force
            if force == -1:
                self.break_force[start_idx:end_idx] *= 2**23
            else:
                self.break_force[start_idx:end_idx] *= force

        # Determine if any objects are fragile
        self.fragile = torch.any(self.break_force < 2**20)  # Threshold to distinguish "unbreakable"

        # Store original methods
        self._original_get_dones = None
        self._original_get_rewards = None

        # Flag to track if wrapper is initialized
        self._wrapper_initialized = False

        # Initialize wrapper if environment is ready
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def _initialize_wrapper(self):
        """Initialize the wrapper after the base environment is set up.

        This method implements lazy initialization to handle cases where the
        base environment isn't fully initialized when the wrapper is created.
        It overrides the environment's _get_dones method to include force-based
        termination conditions.
        """
        if self._wrapper_initialized:
            return

        # Store and override _get_dones method
        if hasattr(self.unwrapped, '_get_dones'):
            self._original_get_dones = self.unwrapped._get_dones
            self.unwrapped._get_dones = self._wrapped_get_dones

        # Store and override _get_rewards method for peg break rewards
        if hasattr(self.unwrapped, '_get_rewards'):
            self._original_get_rewards = self.unwrapped._get_rewards
            self.unwrapped._get_rewards = self._wrapped_get_rewards

        self._wrapper_initialized = True

    def _wrapped_get_dones(self):
        """Check for episode termination including force violations.

        This method extends the original environment's termination logic to include
        force-based termination when objects are broken due to excessive force.

        Returns:
            tuple: (terminated, time_out) where:
                - terminated (torch.Tensor): Boolean tensor indicating which environments
                  have terminated due to task completion or force violations
                - time_out (torch.Tensor): Boolean tensor indicating which environments
                  have timed out (unchanged from original environment)
        """
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
            terminated = force_violations #torch.logical_or(terminated, force_violations)

        return terminated, time_out

    def _has_force_torque_data(self):
        """Check if force-torque data is available from a force-torque wrapper.

        Returns:
            bool: True if the environment has robot_force_torque attribute
                 (typically provided by ForceTorqueWrapper), False otherwise.
        """
        return hasattr(self.unwrapped, 'robot_force_torque')

    def step(self, action):
        """Execute one environment step with fragile object monitoring.

        Ensures wrapper is properly initialized (lazy initialization) and
        then executes the base environment step. Force violation checking
        happens automatically in the _wrapped_get_dones method.

        Args:
            action (torch.Tensor): Actions to execute in the environment.

        Returns:
            tuple: Standard gymnasium step return: (obs, rewards, terminated, truncated, info)
        """
        # Initialize wrapper if not done yet
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        return super().step(action)

    def reset(self, **kwargs):
        """Reset the environment and ensure wrapper is initialized.

        Performs standard environment reset and ensures wrapper is properly
        initialized via lazy initialization pattern.

        Args:
            **kwargs: Additional keyword arguments passed to base environment reset.

        Returns:
            tuple: (observations, info) from the base environment reset.
        """
        obs, info = super().reset(**kwargs)

        # Initialize wrapper if not done yet
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        return obs, info

    def get_agent_assignment(self):
        """Get the environment indices assigned to each agent.

        Returns a dictionary mapping agent IDs to their assigned environment indices.
        Environments are statically assigned in contiguous blocks.

        Returns:
            dict: Mapping from agent_id (int) to list of environment indices.
                 For example, with 8 envs and 2 agents: {0: [0,1,2,3], 1: [4,5,6,7]}
        """
        assignments = {}
        for i in range(self.num_agents):
            start_idx = i * self.envs_per_agent
            end_idx = (i + 1) * self.envs_per_agent
            assignments[i] = list(range(start_idx, end_idx))
        return assignments

    def get_break_forces(self):
        """Get the break force thresholds for all environments.

        Returns:
            torch.Tensor: Break force thresholds for each environment.
                         Shape: (num_envs,). Values of 2^23 indicate unbreakable objects.
        """
        return self.break_force.clone()

    def get_agent_break_force(self, agent_id):
        """Get the break force threshold for a specific agent's environments.

        Args:
            agent_id (int): ID of the agent (0 to num_agents-1).

        Returns:
            torch.Tensor: Break force thresholds for the specified agent's environments.
                         Shape: (envs_per_agent,).

        Raises:
            ValueError: If agent_id is greater than or equal to num_agents.
        """
        if agent_id >= self.num_agents:
            raise ValueError(f"Agent ID ({agent_id}) must be less than number of agents ({self.num_agents})")

        start_idx = agent_id * self.envs_per_agent
        end_idx = (agent_id + 1) * self.envs_per_agent
        return self.break_force[start_idx:end_idx].clone()

    def is_fragile(self):
        """Check if any objects in the environment are fragile.

        Returns:
            bool: True if any objects can break (have break_force < 2^20),
                 False if all objects are unbreakable.
        """
        return self.fragile

    def get_force_violations(self):
        """Get current force violations if force-torque data is available.

        Checks which environments currently have force magnitudes exceeding
        their break force thresholds.

        Returns:
            torch.Tensor: Boolean tensor indicating which environments have
                         force violations. Shape: (num_envs,).
                         Returns all False if no force data or no fragile objects.
        """
        if self.fragile and self._has_force_torque_data():
            force_magnitude = torch.linalg.norm(self.unwrapped.robot_force_torque[:, :3], axis=1)
            return force_magnitude >= self.break_force
        else:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.unwrapped.device)

    def _wrapped_get_rewards(self):
        """
        Calculate rewards including peg break penalties.

        Extends the original environment's reward calculation to include
        negative rewards when force violations (peg breaks) occur.

        Returns:
            torch.Tensor: Combined rewards including base rewards and peg break penalties.
                         Shape: (num_envs,).
        """
        # Get original rewards
        base_rewards = self._original_get_rewards()

        # If wrapper is disabled, return base rewards only
        if not self.enabled:
            return base_rewards

        # Calculate peg break penalties
        peg_break_rewards = torch.zeros_like(base_rewards)

        # Check for force violations and apply penalty
        if self.fragile and self._has_force_torque_data():
            force_violations = self.get_force_violations()
            peg_break_rewards = torch.where(
                force_violations,
                torch.full_like(base_rewards, self.peg_break_rew),
                torch.zeros_like(base_rewards)
            )

        # Log peg break reward component
        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['logs_rew_peg_break'] = peg_break_rewards

        return base_rewards + peg_break_rewards