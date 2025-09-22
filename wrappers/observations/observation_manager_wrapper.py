"""
Observation Manager Wrapper

This wrapper handles Isaac Lab factory environment observations and converts them
from {"policy": tensor, "critic": tensor} format to single tensor format for SKRL compatibility.

Features:
- Converts factory environment dict observations to single tensor
- Handles observation component composition using obs_order and state_order
- Configurable observation merging strategies
- Clean format conversion without noise injection (use ObservationNoiseWrapper for noise)
"""

import torch
import gymnasium as gym
import numpy as np


class ObservationManagerWrapper(gym.Wrapper):
    """
    Wrapper that converts Isaac Lab factory environment observations to SKRL-compatible format.

    The factory environment outputs {"policy": tensor, "critic": tensor} but SKRL and
    original BlockSimBa models expect single tensor observations. This wrapper handles
    the conversion with configurable merging strategies.

    Key Features:
    - Converts Isaac Lab dict observations to single tensor format
    - Supports multiple merge strategies (concatenate, policy_only, critic_only, average)
    - Handles observation component composition using env configuration
    - Validates observation format and provides diagnostic information
    - Lazy initialization for compatibility with wrapper chains

    Args:
        env (gym.Env): Base environment to wrap. Should output Isaac Lab format observations.
        merge_strategy (str): Strategy for merging policy/critic observations:
            - "concatenate": Concatenate policy and critic observations (default)
            - "policy_only": Use only policy observations
            - "critic_only": Use only critic observations
            - "average": Average policy and critic observations (requires same dimensions)

    Example:
        >>> env = FactoryEnv()  # Outputs {"policy": tensor, "critic": tensor}
        >>> wrapped_env = ObservationManagerWrapper(env, merge_strategy="concatenate")
        >>> obs, info = wrapped_env.reset()
        >>> # obs is now a single tensor instead of dict

    Note:
        - Requires environment to have _get_observations method that returns Isaac Lab format
        - Uses lazy initialization to work with environments that aren't fully initialized
        - Provides validation methods to check wrapper stack compatibility
    """

    def __init__(self, env, merge_strategy="concatenate"):
        """
        Initialize observation manager wrapper.

        Args:
            env (gym.Env): Base environment to wrap. Should have _get_observations method.
            merge_strategy (str): How to merge policy/critic observations:
                - "concatenate": Concatenate policy and critic observations (default)
                - "policy_only": Use only policy observations
                - "critic_only": Use only critic observations
                - "average": Average policy and critic observations (same dimensions)

        Raises:
            ValueError: If environment doesn't have required attributes after initialization.
        """
        super().__init__(env)

        self.merge_strategy = merge_strategy
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

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
        """Get observations and preserve Isaac Lab factory format for SKRL compatibility."""
        # Get base observations from factory environment
        obs = self._original_get_observations() if self._original_get_observations else {}

        # Check if this is Isaac Lab factory format that SKRL expects
        if isinstance(obs, dict) and "policy" in obs and "critic" in obs:
            # This is the format SKRL expects - preserve it!
            return obs
        else:
            # Convert other formats to single tensor
            single_tensor_obs = self._convert_to_single_tensor(obs)
            return single_tensor_obs

    def _convert_to_single_tensor(self, obs):
        """Convert factory environment observations to single tensor format."""
        if isinstance(obs, dict) and "policy" in obs and "critic" in obs:
            # Factory environment format - merge according to strategy
            policy_obs = obs["policy"]
            critic_obs = obs["critic"]

            if self.merge_strategy == "policy_only":
                return policy_obs
            elif self.merge_strategy == "critic_only":
                return critic_obs
            elif self.merge_strategy == "concatenate":
                return torch.cat([policy_obs, critic_obs], dim=-1)
            elif self.merge_strategy == "average":
                # Only works if policy and critic have same dimensions
                if policy_obs.shape == critic_obs.shape:
                    return (policy_obs + critic_obs) / 2.0
                else:
                    print(f"Warning: Cannot average different shapes {policy_obs.shape} vs {critic_obs.shape}, using policy only")
                    return policy_obs
            else:
                print(f"Warning: Unknown merge strategy '{self.merge_strategy}', using policy only")
                return policy_obs

        elif isinstance(obs, torch.Tensor):
            # Already single tensor
            return obs
        elif isinstance(obs, dict):
            # Try to compose from other dict format
            return self._compose_single_tensor_from_dict(obs)
        else:
            # Unknown format - create zeros
            print(f"Warning: Unknown observation format {type(obs)}, returning zeros")
            return torch.zeros((self.num_envs, 1), device=self.device)

    def _compose_single_tensor_from_dict(self, obs_dict):
        """Compose single tensor from dictionary format using configured order."""
        tensors = []

        # Try to use observation order first
        if hasattr(self.unwrapped.cfg, 'obs_order'):
            for obs_name in self.unwrapped.cfg.obs_order:
                if obs_name in obs_dict:
                    tensor = obs_dict[obs_name]
                    if isinstance(tensor, torch.Tensor):
                        tensors.append(tensor)

        # Fallback: use all available tensors if no order specified
        if not tensors:
            tensors = [v for v in obs_dict.values() if isinstance(v, torch.Tensor)]

        # Concatenate all tensors
        if tensors:
            return torch.cat(tensors, dim=-1)
        else:
            return torch.zeros((self.num_envs, 1), device=self.device)


    def _validate_observations(self, obs):
        """Validate single tensor observation format and dimensions."""
        if not isinstance(obs, torch.Tensor):
            raise ValueError("Observations must be a torch.Tensor")

        if obs.shape[0] != self.num_envs:
            raise ValueError(f"Observation first dimension must match num_envs ({self.num_envs})")

        if len(obs.shape) != 2:
            raise ValueError("Observation must be 2D (num_envs, features)")

        # Check for NaN or inf values
        if torch.isnan(obs).any():
            print("Warning: NaN values detected in observations")
        if torch.isinf(obs).any():
            print("Warning: Inf values detected in observations")

    def get_observation_info(self):
        """
        Get information about current observation format.

        Returns:
            dict: Information about observation format including:
                - observation: Dict with shape, dtype, device, min, max, mean, std
                - merge_strategy: Current merge strategy being used
                - error: Error message if observation retrieval fails
        """
        try:
            obs = self._wrapped_get_observations()
            if isinstance(obs, torch.Tensor):
                return {
                    'observation': {
                        'shape': list(obs.shape),
                        'dtype': str(obs.dtype),
                        'device': str(obs.device),
                        'min': obs.min().item(),
                        'max': obs.max().item(),
                        'mean': obs.mean().item(),
                        'std': obs.std().item()
                    },
                    'merge_strategy': self.merge_strategy
                }
            else:
                return {"error": f"Observations not in expected tensor format, got {type(obs)}"}
        except Exception as e:
            return {"error": f"Failed to get observation info: {e}"}

    def get_observation_space_info(self):
        """
        Get information about configured observation spaces.

        Returns:
            dict: Configuration information including:
                - obs_order: List of observation component names for policy
                - state_order: List of observation component names for critic
                - observation_space: Policy observation space size
                - state_space: Critic observation space size
        """
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
        """
        Validate that the wrapper stack is properly configured.

        Checks that observations can be retrieved and converted properly,
        and that the base environment produces expected Isaac Lab format.

        Returns:
            list: List of validation issues found. Empty list means no issues.
        """
        issues = []

        # Check that we can get observations
        try:
            obs = self._wrapped_get_observations()
            if isinstance(obs, torch.Tensor):
                self._validate_observations(obs)
            elif isinstance(obs, dict) and "policy" in obs and "critic" in obs:
                # Isaac Lab factory format - validate both tensors
                self._validate_observations(obs["policy"])
                self._validate_observations(obs["critic"])
            else:
                issues.append(f"Expected tensor or Isaac Lab factory format observations, got {type(obs)}")
        except Exception as e:
            issues.append(f"Failed to get or validate observations: {e}")

        # Check that base environment produces factory format
        try:
            base_obs = self._original_get_observations() if self._original_get_observations else None
            if base_obs is not None:
                if isinstance(base_obs, dict) and "policy" in base_obs and "critic" in base_obs:
                    policy_shape = base_obs["policy"].shape
                    critic_shape = base_obs["critic"].shape
                    print(f"✓ Factory format detected: policy={policy_shape}, critic={critic_shape}")
                else:
                    issues.append("Base environment doesn't produce factory {'policy': tensor, 'critic': tensor} format")
        except Exception as e:
            issues.append(f"Failed to check base environment format: {e}")

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