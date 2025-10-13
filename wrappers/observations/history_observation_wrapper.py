"""
Simplified History Observation Wrapper

This wrapper provides selective historical observation tracking for specified
observation components, using the environment's configuration to determine
component dimensions.

Key principles:
- Uses environment configuration for component dimensions (no hardcoded values)
- Only tracks history for explicitly specified components
- No acceleration calculation (unrelated to history tracking)
- Simple, single-purpose methods
"""

import torch
import gymnasium as gym


class HistoryObservationWrapper(gym.Wrapper):
    """
    Wrapper that adds selective historical observation tracking.

    Features:
    - Only tracks history for specified observation components
    - Uses environment configuration for component dimensions
    - Configurable history length and sampling
    - Simple history buffer management
    """

    def __init__(self, env, history_components=None, history_length=None, history_samples=None):
        """
        Initialize history observation wrapper.

        Args:
            env: Base environment to wrap
            history_components: List of observation component names to track history for
                               Must be explicitly specified (cannot be None)
            history_length: Length of history buffer (if None, uses decimation from config)
            history_samples: Number of samples to keep from history (if None, uses config)
        """
        super().__init__(env)

        # Require explicit history components - no silent defaults
        if history_components is None:
            raise ValueError(
                "history_components cannot be None. Please explicitly specify which observation "
                "components should track history, e.g., ['force_torque', 'ee_linvel', 'ee_angvel']. "
                "Use an empty list [] if no history tracking is needed."
            )
        self.history_components = history_components

        # Validate that all requested components exist in environment configuration
        self._validate_components()

        # Set history parameters
        self.history_length = history_length or getattr(env.unwrapped.cfg, 'decimation', 8)

        # Configure history sampling
        self.num_samples = history_samples
        if self.num_samples is None:
            self.num_samples = getattr(env.unwrapped.cfg, 'history_samples', -1)
        if self.num_samples == -1:
            self.num_samples = self.history_length

        # Calculate sample indices
        self.keep_idxs = torch.linspace(0.0, self.history_length - 1.0, self.num_samples).type(torch.int32)
        if self.num_samples == 1:  # Special case: always want last value
            self.keep_idxs[0] = self.history_length - 1

        # Device and environment info
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # Update observation space dimensions for history components
        self._update_observation_dimensions()

        # History buffers (will be initialized when wrapper is activated)
        self.history_buffers = {}

        # Store original methods
        self._original_get_observations = None
        self._original_reset_idx = None
        self._original_pre_physics_step = None

        # Initialize wrapper
        self._wrapper_initialized = False
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def _validate_components(self):
        """Validate that all requested history components exist in environment configuration."""
        if not self.history_components:
            return  # Empty list is valid

        # Get component dimensions from environment configuration
        component_dims = self._get_component_dimensions_from_config()

        # Check that all requested components are available
        missing_components = []
        for component in self.history_components:
            if component not in component_dims:
                missing_components.append(component)

        if missing_components:
            available_components = list(component_dims.keys())
            raise ValueError(
                f"History components {missing_components} not found in environment configuration. "
                f"Available components: {available_components}"
            )

    def _get_component_dimensions_from_config(self):
        """Get component dimensions from Isaac Lab's native dimension dictionaries."""
        try:
            # Import Isaac Lab's native dimension configurations (single source of truth)
            try:
                from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
            except ImportError:
                from omni.isaac.lab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG

            # Combine both observation and state dimensions for comprehensive component tracking
            # History wrapper needs access to all available components
            component_dims = {**OBS_DIM_CFG, **STATE_DIM_CFG}

            return component_dims

        except ImportError:
            # Check if we're in test environment by looking for test marker
            if hasattr(self.unwrapped.cfg, '_is_mock_test_config'):
                # In tests, import the mock configs from the test environment
                try:
                    from tests.mocks.mock_isaac_lab import OBS_DIM_CFG_WITH_FORCE as OBS_DIM_CFG, STATE_DIM_CFG_WITH_FORCE as STATE_DIM_CFG
                    component_dims = {**OBS_DIM_CFG, **STATE_DIM_CFG}
                    return component_dims
                except ImportError:
                    # Fallback if mock not available
                    pass

            raise ValueError(
                "Could not import Isaac Lab's native dimension configurations (OBS_DIM_CFG, STATE_DIM_CFG). "
                "Please ensure Isaac Lab is properly installed and factory tasks are available. "
                "The history wrapper requires these configurations as the single source of truth "
                "for observation component dimensions."
            )

    def _update_observation_dimensions(self):
        """Update observation space dimensions for history components."""
        if not self.history_components:
            return

        component_dims = self._get_component_dimensions_from_config()

        # Update observation space dimensions
        try:
            if hasattr(self.unwrapped.cfg, 'observation_space'):
                additional_dims = 0
                for component in self.history_components:
                    base_dim = component_dims[component]
                    # Add (num_samples - 1) * base_dim for history
                    additional_dims += (self.num_samples - 1) * base_dim

                self.unwrapped.cfg.observation_space += additional_dims

            if hasattr(self.unwrapped.cfg, 'state_space'):
                additional_dims = 0
                for component in self.history_components:
                    if component in getattr(self.unwrapped.cfg, 'state_order', []):
                        base_dim = component_dims[component]
                        # Add (num_samples - 1) * base_dim for history
                        additional_dims += (self.num_samples - 1) * base_dim

                self.unwrapped.cfg.state_space += additional_dims

        except Exception as e:
            print(f"Warning: Could not update observation dimensions: {e}")

    def _initialize_wrapper(self):
        """Initialize wrapper by overriding environment methods."""
        if self._wrapper_initialized:
            return

        # Store and override methods
        if hasattr(self.unwrapped, '_get_observations'):
            self._original_get_observations = self.unwrapped._get_observations
            self.unwrapped._get_observations = self._wrapped_get_observations

        if hasattr(self.unwrapped, '_reset_idx'):
            self._original_reset_idx = self.unwrapped._reset_idx
            self.unwrapped._reset_idx = self._wrapped_reset_idx

        if hasattr(self.unwrapped, '_pre_physics_step'):
            self._original_pre_physics_step = self.unwrapped._pre_physics_step
            self.unwrapped._pre_physics_step = self._wrapped_pre_physics_step

        # Initialize history buffers
        self._init_history_buffers()

        self._wrapper_initialized = True

    def _init_history_buffers(self):
        """Initialize history buffers for specified components."""
        if not self.history_components:
            return

        component_dims = self._get_component_dimensions_from_config()

        for component in self.history_components:
            dim = component_dims[component]
            self.history_buffers[component] = torch.zeros(
                (self.num_envs, self.history_length, dim), device=self.device
            )

    def _wrapped_get_observations(self):
        """Get observations with history components."""
        # Get base observations
        obs_dict = self._original_get_observations() if self._original_get_observations else {}

        # Add history for specified components
        if self.history_components and self.history_buffers:
            obs_dict.update(self._get_history_observations())

        return obs_dict

    def _get_history_observations(self):
        """Get current observations for history tracking components."""
        obs = {}

        # Get observations based on environment's configured observation sources
        for component in self.history_components:
            if self._has_observation_source(component):
                obs[component] = self._get_observation_value(component)

        return obs

    def _get_component_attr_mapping(self):
        """Get component to environment attribute mapping from configuration."""
        # Require explicit component attribute mapping
        if not hasattr(self.unwrapped.cfg, 'component_attr_map'):
            raise ValueError(
                "Environment configuration must have 'component_attr_map' dictionary defining "
                "mapping from component names to environment attributes. Example: "
                "cfg.component_attr_map = {'force_torque': 'robot_force_torque', 'ee_linvel': 'ee_linvel_fd'}"
            )

        component_attr_map = self.unwrapped.cfg.component_attr_map

        if not isinstance(component_attr_map, dict):
            raise ValueError(
                "Environment configuration 'component_attr_map' must be a dictionary mapping "
                "component names to environment attribute names."
            )

        return component_attr_map

    def _has_observation_source(self, component):
        """Check if environment has the observation source for a component."""
        component_attr_map = self._get_component_attr_mapping()
        attr_name = component_attr_map.get(component, component)
        return hasattr(self.unwrapped, attr_name)

    def _get_observation_value(self, component):
        """Get the observation value for a component."""
        component_attr_map = self._get_component_attr_mapping()
        attr_name = component_attr_map.get(component, component)

        # Handle special computed components first
        if component == "fingertip_pos_rel_fixed":
            return self._get_relative_position("fingertip_pos", "fingertip_midpoint_pos")
        elif component == "held_pos_rel_fixed":
            return self._get_relative_position("held_pos", "held_pos")

        # Get basic attribute value
        if hasattr(self.unwrapped, attr_name):
            return getattr(self.unwrapped, attr_name)
        else:
            raise ValueError(f"Environment attribute '{attr_name}' for component '{component}' not found")

    def _get_relative_position(self, component_name, attr_name):
        """Get relative position component."""
        if not hasattr(self.unwrapped, 'fixed_pos_obs_frame'):
            raise ValueError(f"Cannot compute {component_name}_rel_fixed: fixed_pos_obs_frame not available")

        base_pos = getattr(self.unwrapped, attr_name)
        fixed_pos = self.unwrapped.fixed_pos_obs_frame

        # Add noise if available
        if hasattr(self.unwrapped, 'init_fixed_pos_obs_noise'):
            fixed_pos = fixed_pos + self.unwrapped.init_fixed_pos_obs_noise

        return base_pos - fixed_pos

    def _wrapped_reset_idx(self, env_ids):
        """Reset history buffers for specified environments."""
        # Call original reset
        if self._original_reset_idx:
            self._original_reset_idx(env_ids)

        # Reset history buffers for specified environments
        if self.history_buffers and len(env_ids) > 0:
            for buffer in self.history_buffers.values():
                buffer[env_ids] = 0

    def _wrapped_pre_physics_step(self, actions):
        """Update history buffers before physics step."""
        # Call original pre-physics step
        if self._original_pre_physics_step:
            self._original_pre_physics_step(actions)

        # Update history buffers
        self._update_history()

    def _update_history(self):
        """Update history buffers with current observations."""
        if not self.history_buffers:
            return

        current_obs = self._get_history_observations()

        for component, obs_value in current_obs.items():
            if component in self.history_buffers:
                # Roll buffer (move history back) and add new observation
                self.history_buffers[component] = torch.roll(
                    self.history_buffers[component], -1, 1
                )
                self.history_buffers[component][:, -1, :] = obs_value

    def get_history_stats(self):
        """
        Get statistics about current history buffers.

        Returns:
            dict: Statistics including:
                - history_components: List of components being tracked
                - history_length: Length of history buffer
                - num_samples: Number of samples kept from history
                - keep_indices: Indices used for sampling
                - wrapper_initialized: Whether wrapper is initialized
                - buffer_count: Number of active history buffers
                - {component}_buffer_shape: Shape of each component's buffer
        """
        stats = {
            'history_components': self.history_components,
            'history_length': self.history_length,
            'num_samples': self.num_samples,
            'keep_indices': self.keep_idxs.tolist(),
            'wrapper_initialized': self._wrapper_initialized,
            'buffer_count': len(self.history_buffers),
        }

        if self.history_buffers:
            for component, buffer in self.history_buffers.items():
                stats[f'{component}_buffer_shape'] = list(buffer.shape)

        return stats

    def get_component_history(self, component):
        """
        Get history buffer for a specific component.

        Args:
            component (str): Name of the component to get history for.
                           Must be in history_components list.

        Returns:
            torch.Tensor: Clone of the history buffer for the component.
                         Shape: (num_envs, history_length, component_dim)

        Raises:
            ValueError: If component is not found in history buffers.
        """
        if component not in self.history_buffers:
            raise ValueError(f"Component '{component}' not found in history buffers")

        return self.history_buffers[component].clone()

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