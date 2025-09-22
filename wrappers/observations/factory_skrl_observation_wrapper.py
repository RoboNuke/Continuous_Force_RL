"""
Factory SKRL Observation Wrapper

This wrapper flattens factory environment observations for SKRL compatibility.
It converts the {"policy": tensor, "critic": tensor} format returned by factory
observation wrappers into a single tensor that SKRL can process.
"""

import torch
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class FactorySKRLObservationWrapper(gym.Wrapper):
    """
    Flatten factory observations for SKRL compatibility.

    Converts the {"policy": obs_tensor, "critic": state_tensor} format returned
    by factory observation wrappers into a single observation tensor that SKRL
    can process. The single tensor is used for both actor and critic networks.
    """

    def __init__(self, env, use_critic_for_obs=False):
        """
        Initialize the factory SKRL observation wrapper.

        Args:
            env: Environment to wrap
            use_critic_for_obs: If True, use critic observations as main observations.
                              If False, use policy observations (default).
        """
        super().__init__(env)
        self.use_critic_for_obs = use_critic_for_obs
        print(f"[INFO]: FactorySKRLObservationWrapper initialized - using {'critic' if use_critic_for_obs else 'policy'} observations")

        # Set up observation and state spaces for SKRL compatibility
        self._setup_spaces()

    def _setup_spaces(self):
        """Set up observation and state spaces for SKRL compatibility."""
        # Get dimensions from the underlying environment cfg
        obs_dim = getattr(self.unwrapped.cfg, 'observation_space', 0)
        state_dim = getattr(self.unwrapped.cfg, 'state_space', 0)

        print(f"[INFO]: FactorySKRLObservationWrapper spaces - obs_dim: {obs_dim}, state_dim: {state_dim}")

        # Create gymnasium Box spaces
        if obs_dim > 0:
            self._policy_space = Box(low=-float('inf'), high=float('inf'), shape=(obs_dim,), dtype=np.float32)
        else:
            self._policy_space = None

        if state_dim > 0:
            self._critic_space = Box(low=-float('inf'), high=float('inf'), shape=(state_dim,), dtype=np.float32)
        else:
            self._critic_space = None

    @property
    def single_observation_space(self):
        """Provide single_observation_space for SKRL compatibility."""
        spaces = {}
        if self._policy_space is not None:
            spaces["policy"] = self._policy_space
        if self._critic_space is not None:
            spaces["critic"] = self._critic_space
        return spaces

    @property
    def observation_space(self):
        """Provide observation_space for SKRL compatibility."""
        if hasattr(self, '_policy_space') and self._policy_space is not None:
            return {"policy": self._policy_space}
        return super().observation_space

    @property
    def state_space(self):
        """Provide state_space for SKRL compatibility."""
        if hasattr(self, '_critic_space') and self._critic_space is not None:
            return self._critic_space
        return getattr(super(), 'state_space', None)

    def _get_observations(self):
        """
        Get observations and convert to SKRL-compatible format.

        Returns:
            dict: Observations in {"policy": tensor} format for SKRL compatibility
        """
        obs = self.env._get_observations()
        return self._convert_observations(obs)

    def step(self, action):
        """Step environment and ensure observations are flattened."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Convert observations to SKRL-compatible format
        obs = self._convert_observations(obs)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and ensure observations are flattened."""
        obs, info = self.env.reset(**kwargs)

        # Convert observations to SKRL-compatible format
        obs = self._convert_observations(obs)

        return obs, info

    def _convert_observations(self, obs):
        """
        Convert observations to SKRL-compatible format.

        SKRL Isaac Lab wrapper expects observations in {"policy": tensor} format,
        so we need to maintain that structure but with our chosen tensor.
        """
        if isinstance(obs, dict) and 'policy' in obs and 'critic' in obs:
            # Choose which observation to use
            chosen_obs = obs['critic'] if self.use_critic_for_obs else obs['policy']

            # Return in the format SKRL expects
            return {"policy": chosen_obs}

        # If not factory dict format, assume it's already a single tensor
        # Wrap it in the expected format for SKRL
        return {"policy": obs}