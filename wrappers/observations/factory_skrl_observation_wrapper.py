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