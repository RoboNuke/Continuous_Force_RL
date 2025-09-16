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

        # Update observation and state dimensions
        # Add force_torque to the dimension configs if not present
        if hasattr(self.unwrapped, 'cfg'):
            # Update OBS_DIM_CFG and STATE_DIM_CFG on the environment's config module
            try:
                from envs.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
                OBS_DIM_CFG["force_torque"] = 6
                STATE_DIM_CFG["force_torque"] = 6
            except ImportError:
                # Fallback - define locally if config module not available
                pass

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

        # Episode statistics for force-torque
        self.unwrapped.ep_max_force = torch.zeros((num_envs,), device=device)
        self.unwrapped.ep_max_torque = torch.zeros((num_envs,), device=device)
        self.unwrapped.ep_sum_force = torch.zeros((num_envs,), device=device)
        self.unwrapped.ep_sum_torque = torch.zeros((num_envs,), device=device)

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

        # Reset force-torque statistics
        if hasattr(self.unwrapped, 'ep_max_force'):
            self.unwrapped.ep_max_force[env_ids] = 0
            self.unwrapped.ep_max_torque[env_ids] = 0
            self.unwrapped.ep_sum_force[env_ids] = 0
            self.unwrapped.ep_sum_torque[env_ids] = 0

    def _wrapped_pre_physics_step(self, action):
        """Update force-torque statistics during physics step."""
        # Call original pre-physics step
        if self._original_pre_physics_step:
            self._original_pre_physics_step(action)

        # Update force-torque statistics
        if hasattr(self.unwrapped, 'robot_force_torque'):
            force_magnitude = torch.linalg.norm(self.unwrapped.robot_force_torque[:, :3], axis=1)
            torque_magnitude = torch.linalg.norm(self.unwrapped.robot_force_torque[:, 3:], axis=1)

            self.unwrapped.ep_sum_force += force_magnitude
            self.unwrapped.ep_sum_torque += torque_magnitude
            self.unwrapped.ep_max_force = torch.max(self.unwrapped.ep_max_force, force_magnitude)
            self.unwrapped.ep_max_torque = torch.max(self.unwrapped.ep_max_torque, torque_magnitude)

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

    def get_force_torque_stats(self):
        """Get current force-torque statistics."""
        if hasattr(self.unwrapped, 'robot_force_torque') and hasattr(self.unwrapped, 'ep_max_force'):
            return {
                'current_force': self.unwrapped.robot_force_torque[:, :3],
                'current_torque': self.unwrapped.robot_force_torque[:, 3:],
                'max_force': self.unwrapped.ep_max_force,
                'max_torque': self.unwrapped.ep_max_torque,
                'avg_force': self.unwrapped.ep_sum_force / max(1, getattr(self.unwrapped, 'common_step_counter', 1)),
                'avg_torque': self.unwrapped.ep_sum_torque / max(1, getattr(self.unwrapped, 'common_step_counter', 1)),
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