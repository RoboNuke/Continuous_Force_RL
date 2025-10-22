"""
Misalignment Penalty Wrapper for Factory Environment

This wrapper zeros out keypoint rewards when the held object is below the
engagement height threshold but not properly aligned in XY. This represents
situations where the peg/gear/nut is hitting the wall of the hole/base/bolt
instead of being properly aligned for insertion.

The wrapper modifies both the total reward and the component rewards logged
in extras to ensure proper wandb logging.
"""

import torch
import gymnasium as gym
from typing import Dict, Any
import sys
sys.path.append('/home/hunter/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory')
import factory_utils


class MisalignmentPenaltyWrapper(gym.Wrapper):
    """
    Wrapper that zeros out keypoint rewards when object is below engagement but misaligned.

    This wrapper detects "wall stuck" conditions where:
    - The held object Z-height is below the engagement threshold (z_disp < height_threshold)
    - BUT the XY alignment is not good enough (xy_dist >= xy_threshold)

    When this condition is detected, all keypoint rewards (kp_baseline, kp_coarse, kp_fine)
    are zeroed out in both the total reward and the component logs.

    Features:
    - Configurable XY and height thresholds
    - Works for all factory tasks (peg_insert, gear_mesh, nut_thread)
    - Updates both total reward and component reward logs
    - Logs misalignment statistics to wandb
    """

    def __init__(self, env, config: Dict[str, Any]):
        """
        Initialize the misalignment penalty wrapper.

        Args:
            env: Base environment to wrap
            config: Dictionary containing configuration parameters:
                - enabled: bool - Enable/disable wrapper
                - xy_threshold: float - XY distance threshold for alignment (default: 0.0025)
                - height_threshold_fraction: float - Fraction of fixed asset height (default: 0.9)
        """
        super().__init__(env)

        # Store configuration
        self.config = config
        self.enabled = config.get('enabled', False)

        # Early exit if wrapper is disabled
        if not self.enabled:
            return

        # Initialize environment attributes
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # Configurable thresholds
        self.xy_threshold = config.get('xy_threshold', 0.0025)
        self.height_threshold_fraction = config.get('height_threshold_fraction', 0.9)

        # Store original methods for wrapping
        self._original_get_rewards = None

        # Lazy initialization flag
        self._wrapper_initialized = False

        # Initialize if environment is ready
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def _initialize_wrapper(self):
        """Initialize the wrapper after the base environment is set up."""
        if self._wrapper_initialized or not self.enabled:
            return

        # Store and override _get_rewards method
        if hasattr(self.unwrapped, '_get_rewards'):
            self._original_get_rewards = self.unwrapped._get_rewards
            self.unwrapped._get_rewards = self._wrapped_get_rewards
        else:
            raise ValueError("Environment missing required _get_rewards method")

        self._wrapper_initialized = True

    def _wrapped_get_rewards(self):
        """Calculate rewards with misalignment penalty applied."""
        # Get original rewards
        base_rewards = self._original_get_rewards()

        if not self.enabled:
            return base_rewards

        # Detect misalignment condition
        is_misaligned = self._detect_misalignment()

        # Calculate total keypoint reward to subtract from total
        keypoint_reward_total = torch.zeros(self.num_envs, device=self.device)

        # Zero out keypoint component rewards in extras (for logging)
        for key in ['logs_rew_kp_baseline', 'logs_rew_kp_coarse', 'logs_rew_kp_fine']:
            if key in self.unwrapped.extras:
                original_value = self.unwrapped.extras[key].clone()
                keypoint_reward_total += original_value

                # Zero out in extras for proper logging
                self.unwrapped.extras[key] = torch.where(
                    is_misaligned,
                    torch.zeros_like(original_value),
                    original_value
                )

        # Subtract keypoint rewards from total for misaligned environments
        adjusted_rewards = torch.where(
            is_misaligned,
            base_rewards - keypoint_reward_total,
            base_rewards
        )

        # Log misalignment statistics
        self.unwrapped.extras['logs_misaligned_count'] = is_misaligned.float().mean()

        return adjusted_rewards

    def _detect_misalignment(self) -> torch.Tensor:
        """
        Detect environments where object is below engagement threshold but not aligned.

        Returns:
            torch.Tensor: Boolean mask of shape (num_envs,) indicating misaligned environments
        """
        # Get held and fixed positions using factory utils
        held_base_pos, _ = factory_utils.get_held_base_pose(
            self.unwrapped.held_pos,
            self.unwrapped.held_quat,
            self.unwrapped.cfg_task.name,
            self.unwrapped.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device
        )
        target_held_base_pos, _ = factory_utils.get_target_held_base_pose(
            self.unwrapped.fixed_pos,
            self.unwrapped.fixed_quat,
            self.unwrapped.cfg_task.name,
            self.unwrapped.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device
        )

        # Calculate XY distance and Z displacement
        xy_dist = torch.linalg.vector_norm(
            target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1
        )
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        # Get height threshold based on task type
        height_threshold = self._get_height_threshold()

        # Detect misalignment: below height threshold AND not centered
        is_below_threshold = z_disp < height_threshold
        is_not_centered = xy_dist >= self.xy_threshold
        is_misaligned = torch.logical_and(is_below_threshold, is_not_centered)

        return is_misaligned

    def _get_height_threshold(self):
        """
        Calculate height threshold based on task configuration.

        Different tasks use different height references:
        - peg_insert/gear_mesh: fixed_cfg.height * fraction
        - nut_thread: fixed_cfg.thread_pitch * fraction

        Returns:
            float: Height threshold for engagement detection
        """
        cfg_task = self.unwrapped.cfg_task
        fixed_cfg = cfg_task.fixed_asset_cfg

        if cfg_task.name == "peg_insert" or cfg_task.name == "gear_mesh":
            return fixed_cfg.height * self.height_threshold_fraction
        elif cfg_task.name == "nut_thread":
            return fixed_cfg.thread_pitch * self.height_threshold_fraction
        else:
            raise NotImplementedError(f"Task {cfg_task.name} not supported by MisalignmentPenaltyWrapper")

    def step(self, action):
        """Execute one environment step with misalignment penalty processing."""
        # Initialize wrapper if not done yet
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        return super().step(action)

    def reset(self, **kwargs):
        """Reset the environment and ensure wrapper is initialized."""
        obs, info = super().reset(**kwargs)

        # Initialize wrapper if not done yet
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        return obs, info
