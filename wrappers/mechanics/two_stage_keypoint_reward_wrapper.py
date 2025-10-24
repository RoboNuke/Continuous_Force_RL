"""
Two-Stage Keypoint Reward Wrapper for Factory Environment

Modifies keypoint distance calculation to use a two-stage approach:
- Stage 1 (not centered): Distance to top of hole + height offset
- Stage 2 (centered): Distance to bottom of hole (normal)
"""

import torch
import gymnasium as gym
from typing import Dict, Any


class TwoStageKeypointRewardWrapper(gym.Wrapper):
    """Two-stage keypoint reward wrapper."""

    def __init__(self, env, config: Dict[str, Any]):
        super().__init__(env)

        self.config = config
        self.enabled = config.get('enabled', False)

        if not self.enabled:
            return

        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.xy_threshold = config.get('xy_threshold', 0.0025)

        self._original_compute_intermediate_values = None
        self._wrapper_initialized = False

        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def _initialize_wrapper(self):
        if self._wrapper_initialized or not self.enabled:
            return

        if hasattr(self.unwrapped, '_compute_intermediate_values'):
            self._original_compute_intermediate_values = self.unwrapped._compute_intermediate_values
            self.unwrapped._compute_intermediate_values = self._wrapped_compute_intermediate_values
        else:
            raise ValueError("Environment missing required _compute_intermediate_values method")

        self._wrapper_initialized = True

    def _wrapped_compute_intermediate_values(self, dt):
        """Compute intermediate values and modify keypoint_dist for two-stage reward."""
        # Call original computation
        self._original_compute_intermediate_values(dt)

        if not self.enabled:
            return

        # Check if centered (Stage 2 condition)
        xy_dist = torch.linalg.vector_norm(self.unwrapped.target_held_base_pos[:, 0:2] - self.unwrapped.held_base_pos[:, 0:2], dim=1)
        is_centered = xy_dist < self.xy_threshold

        # Check if below hole top
        below_hole = self.unwrapped.held_base_pos[:, 2] < self.unwrapped.fixed_pos_obs_frame[:, 2]

        # Modify keypoint_dist for Stage 1
        fixed_asset_height = self.unwrapped.cfg_task.fixed_asset_cfg.height
        top_keypoints_fixed = self.unwrapped.keypoints_fixed.clone()
        top_keypoints_fixed[:, :, 2] += fixed_asset_height
        stage1_keypoint_dist = torch.norm(self.unwrapped.keypoints_held - top_keypoints_fixed, p=2, dim=-1).mean(-1)
        keypoint_dist = torch.where(
            is_centered,
            self.unwrapped.keypoint_dist,  # Stage 2: use as-is
            stage1_keypoint_dist
        )

        # Zero reward if below hole in Stage 1
        keypoint_dist = torch.where(
            torch.logical_and(~is_centered, below_hole),
            torch.full_like(keypoint_dist, 1e6),
            keypoint_dist
        )

        # Override keypoint_dist
        self.unwrapped.keypoint_dist = keypoint_dist

    def step(self, action):
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()
        return super().step(action)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()
        return obs, info
