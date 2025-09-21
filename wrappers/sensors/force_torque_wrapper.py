"""
Force-Torque Sensor Wrapper for Factory Environment

This wrapper adds force-torque sensor functionality to any environment,
enabling measurement and observation of contact forces and torques at the robot end-effector.

Extracted from: factory_env.py lines 56, 78-79, 108-116, 199-201, 324-328, 365-371, 388-392, 415-419
"""

import torch
import gymnasium as gym

try:
    from isaacsim.core.api.robots import RobotView
except ImportError:
    try:
        from omni.isaac.core.articulations import ArticulationView as RobotView
    except ImportError:
        RobotView = None


class ForceTorqueWrapper(gym.Wrapper):
    """
    Wrapper that adds force-torque sensor functionality to factory environments.

    Features:
    - Initializes force-torque sensor on robot end-effector
    - Provides force-torque measurements in observations
    - Tracks force/torque statistics for episode metrics
    - Supports tanh scaling for force readings
    """

    def __init__(self, env, use_tanh_scaling=False, tanh_scale=0.03):
        """
        Initialize the force-torque sensor wrapper.

        Args:
            env: Base environment to wrap
            use_tanh_scaling: Whether to apply tanh scaling to force-torque readings
            tanh_scale: Scale factor for tanh transformation
        """
        super().__init__(env)

        # Store configuration
        self.use_tanh_scaling = use_tanh_scaling
        self.tanh_scale = tanh_scale

        # Update observation and state dimensions using Isaac Lab's native approach
        if hasattr(self.unwrapped, 'cfg'):
            self._update_observation_config()

        # Flag to track if sensor is initialized
        self._sensor_initialized = False

        # Store original methods
        self._original_init_tensors = None
        self._original_compute_intermediate_values = None
        self._original_reset_buffers = None
        self._original_pre_physics_step = None

        # Initialize after the base environment is ready
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def _update_observation_config(self):
        """
        Update observation configuration to include force_torque dimensions.

        This method uses Isaac Lab's native configuration approach instead of
        relying on local factory_env_cfg imports.
        """
        env_cfg = self.unwrapped.cfg

        # Method 1: Try to update observation/state order and recalculate dimensions
        if hasattr(env_cfg, 'obs_order') and hasattr(env_cfg, 'state_order'):
            # Add force_torque to observation orders if not present
            if 'force_torque' not in env_cfg.obs_order:
                env_cfg.obs_order.append('force_torque')
            if 'force_torque' not in env_cfg.state_order:
                env_cfg.state_order.append('force_torque')

            # Try to find and update dimension configurations
            self._update_dimension_configs(env_cfg)

        # Method 2: Directly update observation/state space if available
        elif hasattr(env_cfg, 'observation_space') and hasattr(env_cfg, 'state_space'):
            # Add 6 dimensions for force_torque (3 force + 3 torque)
            if hasattr(env_cfg, 'observation_space') and isinstance(env_cfg.observation_space, int):
                env_cfg.observation_space += 6
            if hasattr(env_cfg, 'state_space') and isinstance(env_cfg.state_space, int):
                env_cfg.state_space += 6

        # Method 3: Update gym environment spaces
        self._update_gym_spaces()

    def _update_dimension_configs(self, env_cfg):
        """
        Update dimension configurations for force-torque sensor integration.

        Uses Isaac Lab's native OBS_DIM_CFG and STATE_DIM_CFG as the single source
        of truth for observation dimensions. Only updates component_attr_map for
        attribute mapping when it exists.

        Args:
            env_cfg: Environment configuration object

        Note:
            Force-torque dimensions should already be added to Isaac Lab's dimension
            dictionaries by factory_runnerv2.py when the sensor is enabled.
        """
        # Only update component_attr_map if it exists (for attribute mapping)
        if hasattr(env_cfg, 'component_attr_map'):
            # Add force_torque attribute mapping if not already present
            if 'force_torque' not in env_cfg.component_attr_map:
                env_cfg.component_attr_map['force_torque'] = 'robot_force_torque'

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
                pass

    def _initialize_wrapper(self):
        """
        Initialize the wrapper after the base environment is set up.

        This method performs lazy initialization by overriding environment methods
        and setting up the force-torque sensor interface. It's called automatically
        when the environment's robot attribute is detected or during first step/reset.

        The initialization includes:
        - Storing and overriding environment methods (_init_tensors, _compute_intermediate_values, etc.)
        - Initializing the Isaac Sim RobotView for force-torque sensor access
        - Setting up sensor data buffers
        """
        if self._sensor_initialized:
            return

        # Store original methods
        if hasattr(self.unwrapped, '_init_tensors'):
            self._original_init_tensors = self.unwrapped._init_tensors
            self.unwrapped._init_tensors = self._wrapped_init_tensors

        if hasattr(self.unwrapped, '_compute_intermediate_values'):
            self._original_compute_intermediate_values = self.unwrapped._compute_intermediate_values
            self.unwrapped._compute_intermediate_values = self._wrapped_compute_intermediate_values

        if hasattr(self.unwrapped, '_reset_buffers'):
            self._original_reset_buffers = self.unwrapped._reset_buffers
            self.unwrapped._reset_buffers = self._wrapped_reset_buffers

        if hasattr(self.unwrapped, '_pre_physics_step'):
            self._original_pre_physics_step = self.unwrapped._pre_physics_step
            self.unwrapped._pre_physics_step = self._wrapped_pre_physics_step

        # Initialize force-torque sensor
        self._init_force_torque_sensor()
        self._sensor_initialized = True

    def _init_force_torque_sensor(self):
        """
        Initialize the force-torque sensor interface using Isaac Sim's RobotView.

        Creates a RobotView instance to access joint force measurements from the
        Isaac Sim physics simulation. The sensor targets joint 8 (end-effector)
        to provide 6-DOF force-torque measurements.

        Raises:
            ImportError: If RobotView cannot be imported from Isaac Sim

        Note:
            Gracefully handles initialization failures by setting _robot_av to None
            and printing a warning message.
        """
        if RobotView is None:
            raise ImportError("Could not import RobotView. Please ensure Isaac Sim is properly installed.")

        try:
            self._robot_av = RobotView(prim_paths_expr="/World/envs/env_.*/Robot")
            self._robot_av.initialize()
        except Exception as e:
            print(f"Warning: Failed to initialize force-torque sensor: {e}")
            self._robot_av = None

    def _wrapped_init_tensors(self):
        """
        Initialize tensors including force-torque sensor data.

        This method wraps the environment's original _init_tensors method and
        additionally initializes the robot_force_torque tensor for storing
        6-DOF force-torque measurements (3 force + 3 torque components).

        The force-torque tensor is initialized as zeros with shape (num_envs, 6)
        on the same device as the environment.
        """
        # Call original initialization
        if self._original_init_tensors:
            self._original_init_tensors()

        # Initialize force-torque tensors
        num_envs = self.unwrapped.num_envs
        device = self.unwrapped.device

        # Main force-torque sensor data
        self.unwrapped.robot_force_torque = torch.zeros(
            (num_envs, 6), dtype=torch.float32, device=device
        )


    def _wrapped_compute_intermediate_values(self, dt):
        """
        Compute intermediate values including force-torque measurements.

        This method wraps the environment's original _compute_intermediate_values
        method and additionally updates force-torque sensor readings from the
        Isaac Sim physics simulation.

        Args:
            dt (float): Physics simulation timestep

        The method:
        1. Calls the original compute intermediate values method
        2. Retrieves joint forces from the RobotView sensor interface
        3. Extracts force-torque data from joint 8 (end-effector)
        4. Updates the environment's robot_force_torque tensor
        5. Gracefully handles sensor failures by setting data to zeros
        """
        # Call original computation
        if self._original_compute_intermediate_values:
            self._original_compute_intermediate_values(dt)

        # Get force-torque sensor data
        if self._robot_av is not None:
            try:
                # Force-torque sensor is at joint 8 (end-effector)
                joint_forces = self._robot_av.get_measured_joint_forces()
                if joint_forces.shape[1] > 8:  # Ensure joint 8 exists
                    self.unwrapped.robot_force_torque = joint_forces[:, 8, :]
                else:
                    # Fallback to zeros if joint 8 doesn't exist
                    self.unwrapped.robot_force_torque.fill_(0.0)
            except Exception as e:
                # Fallback to zeros if measurement fails
                self.unwrapped.robot_force_torque.fill_(0.0)

    def _wrapped_reset_buffers(self, env_ids):
        """
        Reset force-torque buffers for specified environments.

        This method wraps the environment's original _reset_buffers method
        and additionally resets force-torque sensor readings to zero for
        the specified environment indices.

        Args:
            env_ids (torch.Tensor): Indices of environments to reset
        """
        # Call original reset
        if self._original_reset_buffers:
            self._original_reset_buffers(env_ids)

        # Reset force-torque sensor readings
        if hasattr(self.unwrapped, 'robot_force_torque'):
            self.unwrapped.robot_force_torque[env_ids] = 0.0

    def _wrapped_pre_physics_step(self, action):
        """Update during physics step."""
        # Call original pre-physics step
        if self._original_pre_physics_step:
            self._original_pre_physics_step(action)

    def step(self, action):
        """Step the environment and ensure wrapper is initialized."""
        # Initialize wrapper if not done yet
        if not self._sensor_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        return super().step(action)

    def reset(self, **kwargs):
        """Reset the environment and ensure wrapper is initialized."""
        obs, info = super().reset(**kwargs)

        # Initialize wrapper if not done yet
        if not self._sensor_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        return obs, info

    def get_current_force_torque(self):
        """
        Get current force-torque readings split into force and torque components.

        Returns:
            dict: Dictionary containing:
                - 'current_force': Force components (Nx3 tensor)
                - 'current_torque': Torque components (Nx3 tensor)
                Empty dict if no force-torque data is available
        """
        if hasattr(self.unwrapped, 'robot_force_torque'):
            return {
                'current_force': self.unwrapped.robot_force_torque[:, :3],
                'current_torque': self.unwrapped.robot_force_torque[:, 3:],
            }
        return {}

    def has_force_torque_data(self):
        """
        Check if force-torque data is available in the environment.

        Returns:
            bool: True if robot_force_torque attribute exists, False otherwise
        """
        return hasattr(self.unwrapped, 'robot_force_torque')

    def get_force_torque_observation(self):
        """
        Get force-torque data formatted for observations.

        Retrieves the current force-torque sensor readings and optionally
        applies tanh scaling for bounded output values. This method is
        suitable for including force-torque data in observation vectors.

        Returns:
            torch.Tensor: Force-torque observation data with shape (num_envs, 6)
                         Optionally scaled with tanh if use_tanh_scaling is True.
                         Returns zeros if no sensor data is available.

        Note:
            When tanh scaling is enabled, the output values are bounded to [-1, 1]
            using the formula: tanh(tanh_scale * force_torque_data)
        """
        if self.has_force_torque_data():
            force_torque_obs = self.unwrapped.robot_force_torque.clone()

            if self.use_tanh_scaling:
                force_torque_obs = torch.tanh(self.tanh_scale * force_torque_obs)

            return force_torque_obs
        else:
            # Return zeros if no data available
            return torch.zeros((self.unwrapped.num_envs, 6), device=self.unwrapped.device)