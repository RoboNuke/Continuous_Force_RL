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
                print(f"[INFO]: Created env.state_space with {state_dim} dimensions")
            else:
                print(f"[INFO]: No state space needed (state_dim = {state_dim})")

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
        # DEBUG: Track action shape transformations
        print(f"[DEBUG] Input actions shape: {actions.shape}")
        print(f"[DEBUG] Input actions dtype: {actions.dtype}")
        print(f"[DEBUG] Using action_space: {self.action_space}")

        # Unflatten actions for environment
        actions = unflatten_tensorized_space(self.action_space, actions)

        print(f"[DEBUG] After unflatten actions shape: {actions.shape}")
        print(f"[DEBUG] After unflatten actions dtype: {actions.dtype}")

        # Validate action shape before passing to environment
        expected_action_dim = getattr(self._env.unwrapped.cfg, 'action_space', 6)
        expected_shape = (512, expected_action_dim)
        print(f"[DEBUG] Expected action shape: {expected_shape}")

        if actions.shape != expected_shape:
            print(f"[ERROR] Action shape mismatch: expected {expected_shape}, got {actions.shape}")
            print(f"[ERROR] Environment action_space config: {expected_action_dim}")
            print(f"[ERROR] Wrapper action_space: {self.action_space}")
            raise ValueError(f"Action shape mismatch: expected {expected_shape}, got {actions.shape}")

        # Step environment
        observations, reward, terminated, truncated, self._info = self._env.step(actions)

        # Process both policy and critic observations together
        # Create proper per-environment spaces (Isaac Lab incorrectly includes batch dimension)
        obs_dim = getattr(self._env.unwrapped.cfg, 'observation_space', 0)
        state_dim = getattr(self._env.unwrapped.cfg, 'state_space', 0)

        env_obs_space = Box(low=-float('inf'), high=float('inf'), shape=(obs_dim,), dtype=np.float32)
        env_state_space = Box(low=-float('inf'), high=float('inf'), shape=(state_dim,), dtype=np.float32) if state_dim > 0 else None

        if env_state_space is not None:
            # Create dict of spaces to match observation structure for SKRL
            spaces = Dict({'policy': env_obs_space, 'critic': env_state_space})
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
            # Create proper per-environment spaces (Isaac Lab incorrectly includes batch dimension)
            obs_dim = getattr(self._env.unwrapped.cfg, 'observation_space', 0)
            state_dim = getattr(self._env.unwrapped.cfg, 'state_space', 0)

            env_obs_space = Box(low=-float('inf'), high=float('inf'), shape=(obs_dim,), dtype=np.float32)
            env_state_space = Box(low=-float('inf'), high=float('inf'), shape=(state_dim,), dtype=np.float32) if state_dim > 0 else None

            if env_state_space is not None:
                # Create dict of spaces to match observation structure for SKRL
                spaces = Dict({'policy': env_obs_space, 'critic': env_state_space})
                self._observations = flatten_tensorized_space(tensorize_space(spaces, observations))
            else:
                # Fallback to just observation space if no state space
                self._observations = flatten_tensorized_space(
                    tensorize_space(env_obs_space, observations["policy"])
                )

            self._reset_once = False

        return self._observations, self._info