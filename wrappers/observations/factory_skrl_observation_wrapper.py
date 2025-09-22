"""
Factory SKRL Observation Wrapper

This wrapper concatenates factory environment observations for SKRL compatibility.
It converts the {"policy": tensor, "critic": tensor} format returned by factory
observation wrappers into a single concatenated tensor under the "policy" key.
"""

import torch
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class FactorySKRLObservationWrapper(gym.Wrapper):
    """
    Concatenate factory observations for SKRL compatibility.

    Always concatenates policy + critic observations into a single tensor
    under the "policy" key that SKRL can process.
    """

    def __init__(self, env):
        """
        Initialize the factory SKRL observation wrapper.

        Args:
            env: Environment to wrap
        """
        super().__init__(env)

        # Get dimensions from environment config (DO NOT MODIFY env.cfg - models need original values)
        obs_dim = getattr(self.unwrapped.cfg, 'observation_space', 0)
        state_dim = getattr(self.unwrapped.cfg, 'state_space', 0)
        combined_dim = obs_dim + state_dim

        if combined_dim <= 0:
            raise ValueError(f"Invalid combined dimension: obs={obs_dim} + state={state_dim} = {combined_dim}")

        self._combined_dim = combined_dim

        # Create the correct Box space for SKRL without touching original cfg
        self._policy_space = Box(low=-float('inf'), high=float('inf'), shape=(combined_dim,), dtype=np.float32)

        print(f"[INFO]: FactorySKRLObservationWrapper - obs_dim: {obs_dim}, state_dim: {state_dim}, combined: {combined_dim}")
        print(f"[INFO]: Preserving original env.cfg values for model initialization")

    @property
    def single_observation_space(self):
        """Expose combined observation space size to SKRL."""
        return {"policy": self._policy_space}

    @property
    def observation_space(self):
        """Override for SKRL - return combined space while preserving original env.cfg."""
        return {"policy": self._policy_space}

    @property
    def state_space(self):
        """No separate state space since everything is concatenated into policy."""
        return None

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
        Concatenate policy and critic observations into single tensor.

        Args:
            obs: Observation dict with 'policy' and 'critic' keys

        Returns:
            dict: {"policy": concatenated_tensor}
        """
        if isinstance(obs, dict) and 'policy' in obs and 'critic' in obs:
            # Always concatenate policy + critic
            concatenated = torch.cat([obs['policy'], obs['critic']], dim=-1)
            return {"policy": concatenated}
        else:
            raise ValueError(f"Expected dict with 'policy' and 'critic' keys, got {type(obs)} with keys {list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")