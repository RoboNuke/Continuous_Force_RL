"""
Goal Offset Wrapper

Applies XY offset to goal positions for multi-hole plate scenarios.
The plate origin is used for spawning, but goal positions are offset
to target a specific hole on the plate.

The goal_offset is specified in the plate's local coordinate frame,
so it rotates with the plate when the plate orientation is randomized.
"""

import torch
import gymnasium as gym
from typing import Dict, Any

# Import Isaac Lab utilities for quaternion rotation
try:
    import omni.isaac.lab.utils.torch as torch_utils
except ImportError:
    try:
        import isaacsim.core.utils.torch as torch_utils
    except ImportError:
        raise ImportError("Could not import Isaac Lab torch utilities")


class GoalOffsetWrapper(gym.Wrapper):
    """
    Wrapper that applies XY goal offset to target positions.

    For multi-hole plates where different pegs target different holes,
    this wrapper applies an XY offset from the plate origin to the
    specific target hole.

    Modifies:
    - fixed_success_pos_local (once at init) - success target in local frame
    - fixed_pos_obs_frame (after each reset) - observation reference point
    - fixed_pos_action_frame (after each reset) - action reference point

    The offset is in the plate's local frame and is transformed to world
    frame using the plate's quaternion after each reset.
    """

    def __init__(self, env, config: Dict[str, Any]):
        """
        Initialize the goal offset wrapper.

        Args:
            env: Base environment to wrap
            config: Configuration dictionary (currently unused, offset comes from task config)
        """
        super().__init__(env)

        self.config = config
        self.enabled = config.get('enabled', True)

        if not self.enabled:
            return

        # Defer initialization until environment is ready
        self._wrapper_initialized = False
        self._original_randomize_initial_state = None

        # Get goal offset from task config
        task_cfg = env.unwrapped.cfg_task
        goal_offset = getattr(task_cfg, 'goal_offset', (0.0, 0.0))

        # Skip if offset is zero
        if goal_offset[0] == 0.0 and goal_offset[1] == 0.0:
            self.enabled = False
            print("[GoalOffsetWrapper] No goal offset specified, wrapper disabled")
            return

        # Store as local-frame tensor (will be transformed per-reset)
        self.goal_offset_local = torch.tensor(
            [goal_offset[0], goal_offset[1], 0.0],
            dtype=torch.float32,
            device=env.unwrapped.device
        )

        print(f"[GoalOffsetWrapper] Initialized with offset ({goal_offset[0]*1000:.1f}mm, {goal_offset[1]*1000:.1f}mm)")

        # Initialize if environment is ready
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def _initialize_wrapper(self):
        """Initialize the wrapper after the base environment is set up."""
        if self._wrapper_initialized or not self.enabled:
            return

        # 1. Modify fixed_success_pos_local (in plate's local frame, no transform needed)
        # This affects success detection and reward computation
        if hasattr(self.unwrapped, 'fixed_success_pos_local'):
            self.unwrapped.fixed_success_pos_local[:, 0:2] += self.goal_offset_local[0:2]
            print(f"[GoalOffsetWrapper] Applied offset to fixed_success_pos_local")

        # 2. Override randomize_initial_state to apply offset after reset
        if hasattr(self.unwrapped, 'randomize_initial_state'):
            self._original_randomize_initial_state = self.unwrapped.randomize_initial_state
            self.unwrapped.randomize_initial_state = self._wrapped_randomize_initial_state
        else:
            raise ValueError("Environment missing required randomize_initial_state method")

        self._wrapper_initialized = True
        print("[GoalOffsetWrapper] Wrapper initialized")

    def _wrapped_randomize_initial_state(self, env_ids):
        """
        Randomize initial state and apply goal offset.

        Calls the original randomize_initial_state (which may be from another
        wrapper like SpawnHeightCurriculumWrapper), then applies the goal
        offset to the observation and action frames.
        """
        # Call original (sets fixed_pos_obs_frame, fixed_pos_action_frame)
        self._original_randomize_initial_state(env_ids)

        # Transform local offset to world frame using plate quaternion
        # fixed_quat is the plate's orientation after randomization
        goal_offset_world = self._transform_local_to_world(
            self.goal_offset_local.unsqueeze(0).expand(self.unwrapped.num_envs, -1),
            self.unwrapped.fixed_quat
        )

        # Apply world-frame offset to observation frame
        self.unwrapped.fixed_pos_obs_frame[:, 0:2] += goal_offset_world[:, 0:2]

        # Apply world-frame offset to action frame
        self.unwrapped.fixed_pos_action_frame[:, 0:2] += goal_offset_world[:, 0:2]

    def _transform_local_to_world(self, local_offset, quat):
        """
        Transform local offset by quaternion to get world offset.

        Args:
            local_offset: (N, 3) tensor of local frame offsets
            quat: (N, 4) tensor of quaternions (w, x, y, z format)

        Returns:
            (N, 3) tensor of world frame offsets
        """
        return torch_utils.quat_rotate(quat, local_offset)

    def step(self, action):
        """Step the environment and ensure wrapper is initialized."""
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        return super().step(action)

    def reset(self, **kwargs):
        """Reset the environment and ensure wrapper is initialized."""
        obs, info = super().reset(**kwargs)

        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        return obs, info
