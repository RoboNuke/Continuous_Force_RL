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

        print(f"[INFO]: AsyncCriticIsaacLabWrapper initialized")
        print(f"[INFO]: Environment observation space: {getattr(env, 'observation_space', 'Not available')}")
        print(f"[INFO]: Environment state space: {getattr(env, 'state_space', 'Not available')}")

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        Step the environment and process both policy and critic observations.

        Args:
            actions: Action tensor to execute

        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        # Unflatten actions for environment
        actions = unflatten_tensorized_space(self.action_space, actions)

        # Step environment
        observations, reward, terminated, truncated, self._info = self._env.step(actions)

        # Process both policy and critic observations together
        env_obs_space = self._env.observation_space
        env_state_space = getattr(self._env, 'state_space', None)

        if env_state_space is not None:
            # Create dict of spaces to match observation structure for SKRL
            spaces = {'policy': env_obs_space, 'critic': env_state_space}
            self._observations = flatten_tensorized_space(tensorize_space(spaces, observations))
        else:
            # Fallback to just observation space if no state space
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
        if self._reset_once:
            observations, self._info = self._env.reset()

            # Process both policy and critic observations together
            env_obs_space = self._env.observation_space
            env_state_space = getattr(self._env, 'state_space', None)

            if env_state_space is not None:
                # Create dict of spaces to match observation structure for SKRL
                spaces = {'policy': env_obs_space, 'critic': env_state_space}
                self._observations = flatten_tensorized_space(tensorize_space(spaces, observations))
            else:
                # Fallback to just observation space if no state space
                self._observations = flatten_tensorized_space(
                    tensorize_space(env_obs_space, observations["policy"])
                )

            self._reset_once = False

        return self._observations, self._info