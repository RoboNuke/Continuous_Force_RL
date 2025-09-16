"""
Selective History Observation Wrapper

This wrapper provides selective historical observation tracking, where only specified
observation components maintain history buffers while others remain current-only.

Extracted from: obs_factory_env.py
Key features:
- Configurable history components
- Acceleration calculation for specified components
- History buffer management
- Observation space resizing for history components
- Support for force jerk/snap calculations
"""

import torch
import gymnasium as gym


class HistoryObservationWrapper(gym.Wrapper):
    """
    Wrapper that adds selective historical observation tracking.

    Features:
    - Only tracks history for specified observation components
    - Configurable history length and sampling
    - Acceleration calculation for specified components
    - Maintains original observation structure with expanded tensors
    - Support for force derivative calculations (jerk, snap)
    """

    def __init__(self, env, history_components=None, history_length=None, history_samples=None, calc_acceleration=False):
        """
        Initialize history observation wrapper.

        Args:
            env: Base environment to wrap
            history_components: List of observation component names to track history for
                               If None, defaults to common components
            history_length: Length of history buffer (if None, uses decimation from config)
            history_samples: Number of samples to keep from history (if None, uses config)
            calc_acceleration: Whether to calculate acceleration for specified components
        """
        super().__init__(env)

        # Set default history components if not provided
        if history_components is None:
            history_components = ["force_torque", "ee_linvel", "ee_angvel"]
        self.history_components = history_components

        # Set history parameters
        self.history_length = history_length or getattr(env.unwrapped.cfg, 'decimation', 8)
        self.calc_acceleration = calc_acceleration

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

        # Update observation space dimensions
        self._update_observation_dimensions()

        # History buffers (will be initialized later)
        self.history_buffers = {}
        self.acceleration_buffers = {}

        # Store original methods
        self._original_get_observations = None
        self._original_reset_idx = None
        self._original_pre_physics_step = None

        # Initialize wrapper
        self._wrapper_initialized = False
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def _update_observation_dimensions(self):
        """Update observation space dimensions for history components."""
        try:
            from envs.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG

            # Update dimensions for history components
            for component in self.history_components:
                if component in OBS_DIM_CFG:
                    original_dim = OBS_DIM_CFG[component]
                    OBS_DIM_CFG[component] = original_dim * self.num_samples

                if component in STATE_DIM_CFG:
                    original_dim = STATE_DIM_CFG[component]
                    STATE_DIM_CFG[component] = original_dim * self.num_samples

            # Add acceleration components if enabled
            if self.calc_acceleration:
                for component in self.history_components:
                    if component in ["ee_linvel", "ee_angvel"]:
                        acc_component = component.replace("vel", "acc")
                        OBS_DIM_CFG[acc_component] = OBS_DIM_CFG[component]
                        STATE_DIM_CFG[acc_component] = STATE_DIM_CFG[component]
                    elif component == "force_torque":
                        OBS_DIM_CFG["force_jerk"] = OBS_DIM_CFG[component]
                        OBS_DIM_CFG["force_snap"] = OBS_DIM_CFG[component]

            # Update environment observation space
            if hasattr(self.unwrapped, 'cfg'):
                obs_total = sum([OBS_DIM_CFG.get(obs, 0) for obs in self.unwrapped.cfg.obs_order])
                state_total = sum([STATE_DIM_CFG.get(state, 0) for state in self.unwrapped.cfg.state_order])

                self.unwrapped.cfg.observation_space = obs_total
                self.unwrapped.cfg.state_space = state_total

                # Reconfigure gym env spaces
                if hasattr(self.unwrapped, '_configure_gym_env_spaces'):
                    self.unwrapped._configure_gym_env_spaces()

        except ImportError:
            print("Warning: Could not import OBS_DIM_CFG and STATE_DIM_CFG. Dimension updates skipped.")

    def _initialize_wrapper(self):
        """Initialize wrapper by setting up buffers and overriding methods."""
        if self._wrapper_initialized:
            return

        # Initialize history buffers
        self._init_history_buffers()

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

        self._wrapper_initialized = True

    def _init_history_buffers(self):
        """Initialize history buffers for specified components."""
        # Map component names to their dimensions
        component_dims = {
            "fingertip_pos": 3,
            "fingertip_pos_rel_fixed": 3,
            "fingertip_quat": 4,
            "ee_linvel": 3,
            "ee_angvel": 3,
            "force_torque": 6,
            "held_pos": 3,
            "held_pos_rel_fixed": 3,
            "held_quat": 4,
        }

        # Initialize buffers for each history component
        for component in self.history_components:
            if component in component_dims:
                dim = component_dims[component]
                self.history_buffers[component] = torch.zeros(
                    (self.num_envs, self.history_length, dim), device=self.device
                )

                # Initialize acceleration buffers if needed
                if self.calc_acceleration:
                    if component in ["ee_linvel", "ee_angvel"]:
                        acc_component = component.replace("vel", "acc")
                        self.acceleration_buffers[acc_component] = torch.zeros(
                            (self.num_envs, self.history_length, dim), device=self.device
                        )
                    elif component == "force_torque":
                        self.acceleration_buffers["force_jerk"] = torch.zeros(
                            (self.num_envs, self.history_length, 6), device=self.device
                        )
                        self.acceleration_buffers["force_snap"] = torch.zeros(
                            (self.num_envs, self.history_length, 6), device=self.device
                        )

    def _wrapped_get_observations(self):
        """Get observations with history for specified components."""
        # Get original observations
        original_obs = self._original_get_observations() if self._original_get_observations else {"policy": torch.zeros(0), "critic": torch.zeros(0)}

        # If original returns dict format, work with that
        if isinstance(original_obs, dict):
            return self._process_dict_observations(original_obs)
        else:
            # Fallback for other formats
            return original_obs

    def _process_dict_observations(self, original_obs):
        """Process observations in dictionary format."""
        # Build observation dictionary with history
        obs_dict, state_dict = self._build_observation_dicts()

        # Concatenate based on configured order
        if hasattr(self.unwrapped.cfg, 'obs_order'):
            obs_tensors = []
            for obs_name in self.unwrapped.cfg.obs_order:
                if obs_name in obs_dict:
                    obs_tensors.append(obs_dict[obs_name])

            if obs_tensors:
                obs_concat = torch.cat(obs_tensors, dim=-1)
            else:
                obs_concat = original_obs.get("policy", torch.zeros(self.num_envs, 1, device=self.device))
        else:
            obs_concat = original_obs.get("policy", torch.zeros(self.num_envs, 1, device=self.device))

        if hasattr(self.unwrapped.cfg, 'state_order'):
            state_tensors = []
            for state_name in self.unwrapped.cfg.state_order:
                if state_name in state_dict:
                    state_tensors.append(state_dict[state_name])

            if state_tensors:
                state_concat = torch.cat(state_tensors, dim=-1)
            else:
                state_concat = original_obs.get("critic", torch.zeros(self.num_envs, 1, device=self.device))
        else:
            state_concat = original_obs.get("critic", torch.zeros(self.num_envs, 1, device=self.device))

        return {"policy": obs_concat, "critic": state_concat}

    def _build_observation_dicts(self):
        """Build observation dictionaries with history data."""
        obs_dict = {}
        state_dict = {}

        # Get current observations from environment
        current_obs = self._get_current_observations()

        # Process each observation component
        for component, data in current_obs.items():
            if component in self.history_components and component in self.history_buffers:
                # Use history data
                history_data = self.history_buffers[component][:, self.keep_idxs, :].view(self.num_envs, -1)
                obs_dict[component] = history_data
                state_dict[component] = history_data
            else:
                # Use current data
                obs_dict[component] = data
                state_dict[component] = data

        # Add acceleration components if enabled
        if self.calc_acceleration:
            for acc_component, acc_buffer in self.acceleration_buffers.items():
                acc_data = acc_buffer[:, self.keep_idxs, :].view(self.num_envs, -1)
                obs_dict[acc_component] = acc_data
                state_dict[acc_component] = acc_data

        return obs_dict, state_dict

    def _get_current_observations(self):
        """Get current observation data from the environment."""
        obs = {}

        # Get basic observations if available
        if hasattr(self.unwrapped, 'fingertip_midpoint_pos'):
            obs["fingertip_pos"] = self.unwrapped.fingertip_midpoint_pos

        if hasattr(self.unwrapped, 'fingertip_midpoint_quat'):
            obs["fingertip_quat"] = self.unwrapped.fingertip_midpoint_quat

        if hasattr(self.unwrapped, 'ee_linvel_fd'):
            obs["ee_linvel"] = self.unwrapped.ee_linvel_fd

        if hasattr(self.unwrapped, 'ee_angvel_fd'):
            obs["ee_angvel"] = self.unwrapped.ee_angvel_fd

        if hasattr(self.unwrapped, 'robot_force_torque'):
            obs["force_torque"] = self.unwrapped.robot_force_torque

        if hasattr(self.unwrapped, 'held_pos'):
            obs["held_pos"] = self.unwrapped.held_pos

        if hasattr(self.unwrapped, 'held_quat'):
            obs["held_quat"] = self.unwrapped.held_quat

        # Add relative positions
        if hasattr(self.unwrapped, 'fixed_pos_obs_frame'):
            if "fingertip_pos" in obs:
                noisy_fixed_pos = self.unwrapped.fixed_pos_obs_frame
                if hasattr(self.unwrapped, 'init_fixed_pos_obs_noise'):
                    noisy_fixed_pos = noisy_fixed_pos + self.unwrapped.init_fixed_pos_obs_noise
                obs["fingertip_pos_rel_fixed"] = obs["fingertip_pos"] - noisy_fixed_pos

            if "held_pos" in obs:
                obs["held_pos_rel_fixed"] = obs["held_pos"] - self.unwrapped.fixed_pos_obs_frame

        return obs

    def _update_history(self, reset=False):
        """Update history buffers."""
        current_obs = self._get_current_observations()

        for component in self.history_components:
            if component in self.history_buffers and component in current_obs:
                if reset:
                    # Fill entire history with current value
                    self.history_buffers[component][:, :, :] = current_obs[component][:, None, :]
                else:
                    # Roll buffer and add new value
                    self.history_buffers[component] = torch.roll(self.history_buffers[component], -1, 1)
                    self.history_buffers[component][:, -1, :] = current_obs[component]

        # Update acceleration buffers if enabled
        if self.calc_acceleration:
            for component in self.history_components:
                if component in ["ee_linvel", "ee_angvel"]:
                    acc_component = component.replace("vel", "acc")
                    if acc_component in self.acceleration_buffers:
                        if reset:
                            self.acceleration_buffers[acc_component][:, :, :] = 0
                        else:
                            self.acceleration_buffers[acc_component] = torch.roll(
                                self.acceleration_buffers[acc_component], -1, 1
                            )
                            self.acceleration_buffers[acc_component][:, -1, :] = self._finite_difference(
                                self.history_buffers[component]
                            )

                elif component == "force_torque" and component in self.history_buffers:
                    if "force_jerk" in self.acceleration_buffers:
                        if reset:
                            self.acceleration_buffers["force_jerk"][:, :, :] = 0
                            self.acceleration_buffers["force_snap"][:, :, :] = 0
                        else:
                            self.acceleration_buffers["force_jerk"] = torch.roll(
                                self.acceleration_buffers["force_jerk"], -1, 1
                            )
                            self.acceleration_buffers["force_jerk"][:, -1, :] = self._finite_difference(
                                self.history_buffers[component]
                            )

                            self.acceleration_buffers["force_snap"] = torch.roll(
                                self.acceleration_buffers["force_snap"], -1, 1
                            )
                            self.acceleration_buffers["force_snap"][:, -1, :] = self._finite_difference(
                                self.acceleration_buffers["force_jerk"]
                            )

    def _finite_difference(self, history_buffer):
        """Calculate finite difference for acceleration."""
        if history_buffer.shape[1] < 2:
            return torch.zeros_like(history_buffer[:, -1, :])

        dt = getattr(self.unwrapped.cfg.sim, 'dt', 1/120)
        return (history_buffer[:, -1, :] - history_buffer[:, -2, :]) / dt

    def _wrapped_reset_idx(self, env_ids):
        """Reset with history initialization."""
        # Call original reset
        if self._original_reset_idx:
            self._original_reset_idx(env_ids)

        # Reset history buffers
        self._update_history(reset=True)

    def _wrapped_pre_physics_step(self, action):
        """Update history during pre-physics step."""
        # Call original pre-physics step
        if self._original_pre_physics_step:
            self._original_pre_physics_step(action)

        # Update history
        self._update_history(reset=False)

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

    def get_history_stats(self):
        """Get statistics about history buffers."""
        stats = {
            'history_components': self.history_components,
            'history_length': self.history_length,
            'num_samples': self.num_samples,
            'calc_acceleration': self.calc_acceleration,
            'buffer_count': len(self.history_buffers),
            'acceleration_buffer_count': len(self.acceleration_buffers),
        }
        return stats

    def get_component_history(self, component):
        """Get history for a specific component."""
        if component in self.history_buffers:
            return self.history_buffers[component].clone()
        elif component in self.acceleration_buffers:
            return self.acceleration_buffers[component].clone()
        else:
            return None