"""
Factory SKRL Observation Wrapper

This wrapper flattens factory environment observations for SKRL compatibility.
It converts the {"policy": tensor, "critic": tensor} format returned by factory
observation wrappers into a single tensor that SKRL can process.
"""

import torch
import gymnasium as gym


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

    def _get_observations(self):
        """
        Get observations and flatten dict format to single tensor.

        Returns:
            torch.Tensor: Single observation tensor for SKRL compatibility
        """
        obs = self.env._get_observations()

        # If observations are in factory dict format, flatten them
        if isinstance(obs, dict) and 'policy' in obs and 'critic' in obs:
            if self.use_critic_for_obs:
                return obs['critic']
            else:
                return obs['policy']

        # If already a tensor, return as-is
        return obs

    def step(self, action):
        """Step environment and ensure observations are flattened."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Flatten observations if needed
        if isinstance(obs, dict) and 'policy' in obs and 'critic' in obs:
            if self.use_critic_for_obs:
                obs = obs['critic']
            else:
                obs = obs['policy']

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and ensure observations are flattened."""
        obs, info = self.env.reset(**kwargs)

        # Flatten observations if needed
        if isinstance(obs, dict) and 'policy' in obs and 'critic' in obs:
            if self.use_critic_for_obs:
                obs = obs['critic']
            else:
                obs = obs['policy']

        return obs, info