"""
AsyncCriticIsaacLabWrapper

Custom SKRL wrapper that handles both policy and critic observations by processing
the complete factory observation dictionary {"policy": tensor, "critic": tensor}
instead of only the policy observations.
"""

from typing import Any, Tuple
import torch
import gymnasium
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space
import numpy as np

from gymnasium.spaces import Box, Dict
class AsyncCriticIsaacLabWrapper(Wrapper):
    """
    SKRL wrapper for Isaac Lab that processes both policy and critic observations.

    This wrapper replaces the standard IsaacLabWrapper to handle factory environments
    that return observations in {"policy": tensor, "critic": tensor} format.
    It automatically concatenates both observation types into a single tensor.
    """

    def __init__(self, env: Any) -> None:
        """
        Initialize the AsyncCriticIsaacLabWrapper.

        Args:
            env: Isaac Lab environment to wrap
        """
        super().__init__(env)
        self._reset_once = True
        self._observations = None
        self._info = {}

        # Create state_space if missing (Isaac Lab doesn't set gymnasium state_space by default)
        if not hasattr(env, 'state_space') or getattr(env, 'state_space', None) is None:
            state_dim = getattr(env.unwrapped.cfg, 'state_space', 0)
            if state_dim > 0:
                env.state_space = Box(low=-float('inf'), high=float('inf'), shape=(state_dim,), dtype=np.float32)

        # Store correct action space dimensions for property override
        self._correct_action_dim = getattr(env.unwrapped.cfg, 'action_space', 6)
        self._correct_action_space = Box(low=-float('inf'), high=float('inf'), shape=(self._correct_action_dim,), dtype=np.float32)


    @property
    def action_space(self):
        """Override action_space property to return correct dimensions."""
        return self._correct_action_space

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        Step the environment and process both policy and critic observations.

        Args:
            actions: Action tensor to execute

        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        # SKRL already provides correct action shape [num_envs, action_dim], no unflatten needed
        # Skip unflatten operation since Isaac Lab action space incorrectly includes batch dimension

        # Validate we have correct shape for environment
        expected_action_dim = getattr(self._env.unwrapped.cfg, 'action_space', 6)
        expected_shape = (actions.shape[0], expected_action_dim)  # Use actual batch size

        if actions.shape != expected_shape:
            raise ValueError(f"Unexpected action shape from SKRL: expected {expected_shape}, got {actions.shape}")

        # Step environment
        observations, reward, terminated, truncated, self._info = self._env.step(actions)

        # Process both policy and critic observations together
        # Simply concatenate in [policy, critic] order (NOT alphabetically sorted like SKRL does)
        # The observations are already correctly shaped [num_envs, features] from wrappers
        if isinstance(observations, dict) and "policy" in observations and "critic" in observations:
            # Concatenate in [policy, critic] order to match model expectations
            # Actor uses first num_observations, critic uses last num_observations
            self._observations = torch.cat([observations["policy"], observations["critic"]], dim=-1)
        elif isinstance(observations, dict) and "policy" in observations:
            # Only policy observations available
            self._observations = observations["policy"]
        else:
            # Fallback to old behavior for compatibility
            obs_dim = getattr(self._env.unwrapped.cfg, 'observation_space', 0)
            state_dim = getattr(self._env.unwrapped.cfg, 'state_space', 0)

            env_obs_space = Box(low=-float('inf'), high=float('inf'), shape=(obs_dim,), dtype=np.float32)
            env_state_space = Box(low=-float('inf'), high=float('inf'), shape=(state_dim,), dtype=np.float32) if state_dim > 0 else None

            if env_state_space is not None:
                spaces = Dict({'policy': env_obs_space, 'critic': env_state_space})
                self._observations = flatten_tensorized_space(tensorize_space(spaces, observations))
            else:
                self._observations = flatten_tensorized_space(
                    tensorize_space(env_obs_space, observations["policy"])
                )

        return (
            self._observations,
            reward.view(-1, 1),
            terminated.view(-1, 1),
            truncated.view(-1, 1),
            self._info
        )

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """
        Reset the environment and process both policy and critic observations.

        Returns:
            Tuple of (observations, info)
        """
        # ALWAYS call the underlying environment's reset - don't block subsequent resets!
        observations, self._info = self._env.reset()

        # Process both policy and critic observations together
        # Simply concatenate in [policy, critic] order (NOT alphabetically sorted like SKRL does)
        # The observations are already correctly shaped [num_envs, features] from wrappers
        if isinstance(observations, dict) and "policy" in observations and "critic" in observations:
            # Concatenate in [policy, critic] order to match model expectations
            # Actor uses first num_observations, critic uses last num_observations
            self._observations = torch.cat([observations["policy"], observations["critic"]], dim=-1)
        elif isinstance(observations, dict) and "policy" in observations:
            # Only policy observations available
            self._observations = observations["policy"]
        else:
            # Fallback to old behavior for compatibility
            obs_dim = getattr(self._env.unwrapped.cfg, 'observation_space', 0)
            state_dim = getattr(self._env.unwrapped.cfg, 'state_space', 0)

            env_obs_space = Box(low=-float('inf'), high=float('inf'), shape=(obs_dim,), dtype=np.float32)
            env_state_space = Box(low=-float('inf'), high=float('inf'), shape=(state_dim,), dtype=np.float32) if state_dim > 0 else None

            if env_state_space is not None:
                spaces = Dict({'policy': env_obs_space, 'critic': env_state_space})
                self._observations = flatten_tensorized_space(tensorize_space(spaces, observations))
            else:
                self._observations = flatten_tensorized_space(
                    tensorize_space(env_obs_space, observations["policy"])
                )

        return self._observations, self._info