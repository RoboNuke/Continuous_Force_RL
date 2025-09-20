"""
Action wrapper for forcing gripper to closed position.

This module provides GripperCloseEnv, a gymnasium action wrapper that ensures
the gripper component of robot actions is always set to the closed position (-1.0).
Useful for manipulation tasks where objects should be securely grasped.
"""

import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.spaces import Box
import torch


class GripperCloseEnv(gym.ActionWrapper):
    """
    Action wrapper that forces gripper to closed position.

    This wrapper modifies the last component of the action tensor (assumed to be the gripper
    control signal) to always be -1.0, effectively keeping the gripper closed throughout
    the episode. This is useful for manipulation tasks where objects need to be securely
    grasped without the policy having to learn gripper control.

    The wrapper preserves all other action components and tensor properties (device, dtype).

    Args:
        env (gym.Env): The base environment to wrap. Must have action space with at least
                      one dimension for gripper control (typically the last dimension).

    Example:
        >>> env = MyManipulationEnv()
        >>> env = GripperCloseEnv(env)  # Gripper will always be closed
        >>> action = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5]])  # Last value is gripper
        >>> modified_action = env.action(action)
        >>> print(modified_action)  # [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0]]

    Note:
        - Assumes action tensor shape is (num_envs, action_dim) where the last dimension
          controls the gripper
        - Modifies actions in-place for efficiency
        - Includes debug print statements showing action transformation
    """

    def __init__(self, env):
        """
        Initialize the gripper close wrapper.

        Args:
            env (gym.Env): The base environment to wrap.
        """
        super().__init__(env)

    def action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Transform actions to force gripper to closed position.

        Modifies the last component of the action tensor to -1.0 (closed gripper)
        while preserving all other action components and tensor properties.

        Args:
            action (torch.Tensor): Input actions with shape (num_envs, action_dim).
                                 The last dimension is assumed to control the gripper.

        Returns:
            torch.Tensor: Modified actions with gripper forced to closed position (-1.0).
                         Same tensor object as input (modified in-place).

        Example:
            >>> action = torch.tensor([[1.0, 2.0, 0.8], [0.5, -0.3, 0.2]])
            >>> modified = wrapper.action(action)
            >>> print(modified)  # [[1.0, 2.0, -1.0], [0.5, -0.3, -1.0]]
        """
        print(action)
        action[:, -1] = -1.0
        print("Post:", action)
        return action

    def step(self, action):
        """
        Execute one environment step with gripper forced to closed position.

        Applies the action transformation to ensure gripper is closed, then
        executes the environment step with the modified actions.

        Args:
            action (torch.Tensor): Actions from the policy with shape (num_envs, action_dim).

        Returns:
            tuple: (observations, rewards, terminated, truncated, info) following
                  gymnasium environment step interface.
        """
        obs, rew, done, truncated, info = self.env.step(self.action(action))
        return obs, rew, done, truncated, info
