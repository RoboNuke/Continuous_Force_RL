"""
Observation Manager Wrapper

This wrapper enforces the standard {"policy": tensor, "critic": tensor} observation format
and handles observation component composition, noise injection, and observation space validation.

Features:
- Standard observation format enforcement
- Observation component composition using obs_order and state_order
- Noise injection management
- Dynamic observation space validation
"""

import torch
import gymnasium as gym
import numpy as np


class ObservationManagerWrapper(gym.Wrapper):
    """
    Wrapper that manages observation composition and format standardization.

    Features:
    - Enforces standard {"policy": tensor, "critic": tensor} format
    - Handles observation component composition
    - Manages noise injection for observations
    - Validates observation space consistency
    """

    def __init__(self, env, use_obs_noise=False):
        """
        Initialize observation manager wrapper.

        Args:
            env: Base environment to wrap
            use_obs_noise: Whether to apply observation noise
        """
        super().__init__(env)

        self.use_obs_noise = use_obs_noise
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # Store noise configuration if available
        self.obs_noise_mean = {}
        self.obs_noise_std = {}
        if hasattr(env.unwrapped.cfg, 'obs_noise_mean'):
            self.obs_noise_mean = env.unwrapped.cfg.obs_noise_mean
        if hasattr(env.unwrapped.cfg, 'obs_noise_std'):
            self.obs_noise_std = env.unwrapped.cfg.obs_noise_std

        # Store original methods
        self._original_get_observations = None

        # Initialize wrapper
        self._wrapper_initialized = False
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def _initialize_wrapper(self):
        """Initialize wrapper by overriding observation methods."""
        if self._wrapper_initialized:
            return

        # Store and override methods
        if hasattr(self.unwrapped, '_get_observations'):
            self._original_get_observations = self.unwrapped._get_observations
            self.unwrapped._get_observations = self._wrapped_get_observations

        self._wrapper_initialized = True

    def _wrapped_get_observations(self):
        """Get observations in standard format with noise injection."""
        # Get base observations
        obs = self._original_get_observations() if self._original_get_observations else {}

        # Ensure standard format
        if not isinstance(obs, dict) or "policy" not in obs or "critic" not in obs:
            obs = self._convert_to_standard_format(obs)

        # Apply noise if enabled
        if self.use_obs_noise:
            obs = self._apply_observation_noise(obs)

        # Validate observation format
        self._validate_observations(obs)

        return obs

    def _convert_to_standard_format(self, obs):
        """Convert observations to standard {"policy": tensor, "critic": tensor} format."""
        if isinstance(obs, dict) and "policy" in obs and "critic" in obs:
            # Already in correct format
            return obs
        elif isinstance(obs, torch.Tensor):
            # Single tensor - use for both policy and critic
            return {"policy": obs, "critic": obs}
        elif isinstance(obs, dict):
            # Dict format - try to compose tensors
            return self._compose_observations_from_dict(obs)
        else:
            # Unknown format - create zeros
            return {
                "policy": torch.zeros((self.num_envs, 1), device=self.device),
                "critic": torch.zeros((self.num_envs, 1), device=self.device)
            }

    def _compose_observations_from_dict(self, obs_dict):
        """Compose observations from dictionary format using configured order."""
        policy_tensors = []
        critic_tensors = []

        # Use configured observation order if available
        if hasattr(self.unwrapped.cfg, 'obs_order'):
            for obs_name in self.unwrapped.cfg.obs_order:
                if obs_name in obs_dict:
                    tensor = obs_dict[obs_name]
                    if isinstance(tensor, torch.Tensor):
                        policy_tensors.append(tensor)

        # Use configured state order if available
        if hasattr(self.unwrapped.cfg, 'state_order'):
            for state_name in self.unwrapped.cfg.state_order:
                if state_name in obs_dict:
                    tensor = obs_dict[state_name]
                    if isinstance(tensor, torch.Tensor):
                        critic_tensors.append(tensor)

        # Fallback: use all available tensors if no order specified
        if not policy_tensors:
            policy_tensors = [v for v in obs_dict.values() if isinstance(v, torch.Tensor)]
        if not critic_tensors:
            critic_tensors = [v for v in obs_dict.values() if isinstance(v, torch.Tensor)]

        # Concatenate tensors
        policy_obs = torch.cat(policy_tensors, dim=-1) if policy_tensors else torch.zeros((self.num_envs, 1), device=self.device)
        critic_obs = torch.cat(critic_tensors, dim=-1) if critic_tensors else torch.zeros((self.num_envs, 1), device=self.device)

        return {"policy": policy_obs, "critic": critic_obs}

    def _apply_observation_noise(self, obs):
        """Apply noise to observations."""
        if not self.use_obs_noise:
            return obs

        noisy_obs = {}
        for key, tensor in obs.items():
            if isinstance(tensor, torch.Tensor):
                # Apply noise if configuration is available
                noise_mean = self._get_noise_config(key, 'mean', tensor.shape[-1])
                noise_std = self._get_noise_config(key, 'std', tensor.shape[-1])

                if noise_std is not None and torch.any(torch.tensor(noise_std) > 0):
                    noise = torch.randn_like(tensor) * torch.tensor(noise_std, device=self.device)
                    if noise_mean is not None:
                        noise += torch.tensor(noise_mean, device=self.device)
                    noisy_obs[key] = tensor + noise
                else:
                    noisy_obs[key] = tensor
            else:
                noisy_obs[key] = tensor

        return noisy_obs

    def _get_noise_config(self, obs_key, noise_type, tensor_dim):
        """Get noise configuration for a specific observation component."""
        config_dict = self.obs_noise_mean if noise_type == 'mean' else self.obs_noise_std

        # Try direct key lookup
        if obs_key in config_dict:
            return config_dict[obs_key]

        # Try to find matching component
        for config_key, config_value in config_dict.items():
            if config_key in obs_key or obs_key in config_key:
                return config_value

        # Default to zeros
        return [0.0] * tensor_dim

    def _validate_observations(self, obs):
        """Validate observation format and dimensions."""
        if not isinstance(obs, dict):
            raise ValueError("Observations must be in dictionary format")

        if "policy" not in obs or "critic" not in obs:
            raise ValueError("Observations must contain 'policy' and 'critic' keys")

        for key, tensor in obs.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Observation '{key}' must be a torch.Tensor")

            if tensor.shape[0] != self.num_envs:
                raise ValueError(f"Observation '{key}' first dimension must match num_envs ({self.num_envs})")

            if len(tensor.shape) != 2:
                raise ValueError(f"Observation '{key}' must be 2D (num_envs, features)")

        # Check for NaN or inf values
        for key, tensor in obs.items():
            if torch.isnan(tensor).any():
                print(f"Warning: NaN values detected in observation '{key}'")
            if torch.isinf(tensor).any():
                print(f"Warning: Inf values detected in observation '{key}'")

    def get_observation_info(self):
        """Get information about current observation format."""
        try:
            obs = self.unwrapped._get_observations()
            if isinstance(obs, dict):
                info = {}
                for key, tensor in obs.items():
                    if isinstance(tensor, torch.Tensor):
                        info[key] = {
                            'shape': list(tensor.shape),
                            'dtype': str(tensor.dtype),
                            'device': str(tensor.device),
                            'min': tensor.min().item(),
                            'max': tensor.max().item(),
                            'mean': tensor.mean().item(),
                            'std': tensor.std().item()
                        }
                return info
            else:
                return {"error": "Observations not in expected dictionary format"}
        except Exception as e:
            return {"error": f"Failed to get observation info: {e}"}

    def get_observation_space_info(self):
        """Get information about configured observation spaces."""
        info = {}

        if hasattr(self.unwrapped.cfg, 'obs_order'):
            info['obs_order'] = self.unwrapped.cfg.obs_order
        if hasattr(self.unwrapped.cfg, 'state_order'):
            info['state_order'] = self.unwrapped.cfg.state_order
        if hasattr(self.unwrapped.cfg, 'observation_space'):
            info['observation_space'] = self.unwrapped.cfg.observation_space
        if hasattr(self.unwrapped.cfg, 'state_space'):
            info['state_space'] = self.unwrapped.cfg.state_space

        return info

    def validate_wrapper_stack(self):
        """Validate that the wrapper stack is properly configured."""
        issues = []

        # Check for required observation components
        try:
            obs = self.unwrapped._get_observations()
            if not isinstance(obs, dict):
                issues.append("Observations not in dictionary format")
            elif "policy" not in obs or "critic" not in obs:
                issues.append("Missing 'policy' or 'critic' in observations")
        except Exception as e:
            issues.append(f"Failed to get observations: {e}")

        # Check observation space configuration
        if not hasattr(self.unwrapped.cfg, 'obs_order'):
            issues.append("Missing 'obs_order' in configuration")
        if not hasattr(self.unwrapped.cfg, 'state_order'):
            issues.append("Missing 'state_order' in configuration")

        # Check for dimension consistency
        try:
            if hasattr(self.unwrapped.cfg, 'obs_order'):
                from envs.factory.factory_env_cfg import OBS_DIM_CFG
                expected_obs_dim = sum([OBS_DIM_CFG.get(obs, 0) for obs in self.unwrapped.cfg.obs_order])
                if hasattr(self.unwrapped.cfg, 'observation_space'):
                    if expected_obs_dim != self.unwrapped.cfg.observation_space:
                        issues.append(f"Observation space mismatch: expected {expected_obs_dim}, got {self.unwrapped.cfg.observation_space}")
        except ImportError:
            issues.append("Could not import OBS_DIM_CFG for validation")

        return issues

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