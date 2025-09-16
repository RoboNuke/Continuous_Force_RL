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
        """Update dimension configurations using multiple fallback approaches."""

        # Approach 1: Try to import and update the config that the environment actually uses
        try:
            # Get the module that the environment was defined in
            env_module = self.unwrapped.__class__.__module__
            if env_module:
                import importlib
                module = importlib.import_module(env_module)

                # Look for dimension config dictionaries in the environment's module
                for attr_name in ['OBS_DIM_CFG', 'STATE_DIM_CFG']:
                    if hasattr(module, attr_name):
                        dim_cfg = getattr(module, attr_name)
                        if isinstance(dim_cfg, dict):
                            dim_cfg['force_torque'] = 6

        except (ImportError, AttributeError):
            pass

        # Approach 2: Try to find Isaac Lab's built-in factory environment config
        try:
            # Look for Isaac Lab's factory environment configuration
            from isaaclab.envs.manipulation.factory import factory_env_cfg
            if hasattr(factory_env_cfg, 'OBS_DIM_CFG'):
                factory_env_cfg.OBS_DIM_CFG['force_torque'] = 6
            if hasattr(factory_env_cfg, 'STATE_DIM_CFG'):
                factory_env_cfg.STATE_DIM_CFG['force_torque'] = 6
        except (ImportError, AttributeError, ModuleNotFoundError):
            pass

        # Approach 3: Try alternative Isaac Lab paths
        try:
            from omni.isaac.lab.envs.manipulation.factory import factory_env_cfg
            if hasattr(factory_env_cfg, 'OBS_DIM_CFG'):
                factory_env_cfg.OBS_DIM_CFG['force_torque'] = 6
            if hasattr(factory_env_cfg, 'STATE_DIM_CFG'):
                factory_env_cfg.STATE_DIM_CFG['force_torque'] = 6
        except (ImportError, AttributeError, ModuleNotFoundError):
            pass

        # Approach 4: Create local dimension mapping if environment supports it
        if hasattr(env_cfg, 'obs_dims') or hasattr(self.unwrapped, 'obs_dims'):
            obs_dims = getattr(env_cfg, 'obs_dims', getattr(self.unwrapped, 'obs_dims', {}))
            obs_dims['force_torque'] = 6

        if hasattr(env_cfg, 'state_dims') or hasattr(self.unwrapped, 'state_dims'):
            state_dims = getattr(env_cfg, 'state_dims', getattr(self.unwrapped, 'state_dims', {}))
            state_dims['force_torque'] = 6

    def _update_gym_spaces(self):
        """Update gym environment spaces if needed."""
        if hasattr(self.unwrapped, '_configure_gym_env_spaces'):
            try:
                self.unwrapped._configure_gym_env_spaces()
            except Exception:
                # Silently continue if reconfiguration fails
                pass

    def _initialize_wrapper(self):
        """Initialize the wrapper after the base environment is set up."""
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
        """Initialize the force-torque sensor interface."""
        if RobotView is None:
            raise ImportError("Could not import RobotView. Please ensure Isaac Sim is properly installed.")

        try:
            self._robot_av = RobotView(prim_paths_expr="/World/envs/env_.*/Robot")
            self._robot_av.initialize()
        except Exception as e:
            print(f"Warning: Failed to initialize force-torque sensor: {e}")
            self._robot_av = None

    def _wrapped_init_tensors(self):
        """Initialize tensors including force-torque sensor data."""
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
        """Compute intermediate values including force-torque measurements."""
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
        """Reset force-torque buffers."""
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
        """Get current force-torque readings."""
        if hasattr(self.unwrapped, 'robot_force_torque'):
            return {
                'current_force': self.unwrapped.robot_force_torque[:, :3],
                'current_torque': self.unwrapped.robot_force_torque[:, 3:],
            }
        return {}

    def has_force_torque_data(self):
        """Check if force-torque data is available."""
        return hasattr(self.unwrapped, 'robot_force_torque')

    def get_force_torque_observation(self):
        """Get force-torque data formatted for observations."""
        if self.has_force_torque_data():
            force_torque_obs = self.unwrapped.robot_force_torque.clone()

            if self.use_tanh_scaling:
                force_torque_obs = torch.tanh(self.tanh_scale * force_torque_obs)

            return force_torque_obs
        else:
            # Return zeros if no data available
            return torch.zeros((self.unwrapped.num_envs, 6), device=self.unwrapped.device)