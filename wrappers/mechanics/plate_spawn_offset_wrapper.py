"""
Plate Spawn Offset Wrapper

Offsets the hole plate spawn position for multi-hole plates so that
the TARGET HOLE ends up at the standard position (where a single-hole
plate would be).

This keeps the robot workspace consistent across all peg/hole variants.

Key behavior:
- Base env generates position/orientation with noise
- This wrapper shifts the plate so target hole is at that position
- Observation frames are NOT modified (they already point to target hole)
"""

import torch
import gymnasium as gym

# Import Isaac Lab utilities for quaternion rotation
try:
    import omni.isaac.lab.utils.math as torch_utils
except ImportError:
    try:
        import isaaclab.utils.math as torch_utils
    except ImportError:
        raise ImportError("Could not import Isaac Lab utilities")


class PlateSpawnOffsetWrapper(gym.Wrapper):
    """
    Wrapper that offsets hole plate spawn position for multi-hole plates.

    For multi-hole plates where different pegs target different holes,
    this wrapper shifts the plate spawn position so that the TARGET HOLE
    ends up at the standard position (where a single-hole plate would be).

    This keeps the robot workspace consistent across all peg/hole variants.

    The offset is in the plate's local frame and is transformed to world
    frame using the plate's quaternion after randomization.
    """

    def __init__(self, env):
        """
        Initialize the plate spawn offset wrapper.

        Args:
            env: Base environment to wrap
        """
        super().__init__(env)

        self._wrapper_initialized = False
        self._original_randomize_initial_state = None

        # Get goal offset from task config
        task_cfg = env.unwrapped.cfg_task
        goal_offset = getattr(task_cfg, 'goal_offset', (0.0, 0.0))

        # Validate - this wrapper should only be instantiated when offset is non-zero
        if goal_offset[0] == 0.0 and goal_offset[1] == 0.0:
            raise ValueError(
                "[PlateSpawnOffsetWrapper] Wrapper instantiated with zero goal_offset. "
                "This wrapper should only be applied when goal_offset is non-zero. "
                "Check the conditional logic in launch_utils_v3.py."
            )

        # Store as local-frame tensor (XY only, Z=0)
        self.goal_offset_local = torch.tensor(
            [goal_offset[0], goal_offset[1], 0.0],
            dtype=torch.float32,
            device=env.unwrapped.device
        )

        print(f"[PlateSpawnOffsetWrapper] Initialized with offset "
              f"({goal_offset[0]*1000:.1f}mm, {goal_offset[1]*1000:.1f}mm)")

        # Storage for pending plate position updates (applied after reset completes)
        self._pending_plate_pos = None
        self._pending_plate_quat = None
        self._pending_env_ids = None

        # Initialize if environment is ready
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def _initialize_wrapper(self):
        """Initialize the wrapper after the base environment is set up."""
        if self._wrapper_initialized:
            return

        # NOTE: We do NOT modify fixed_success_pos_local here.
        # The plate is moved so the target hole ends up at the standard position.
        # Success detection uses fixed_success_pos_local which is in plate's local frame,
        # and the base env sets this correctly for the target hole.

        # Override randomize_initial_state
        if hasattr(self.unwrapped, 'randomize_initial_state'):
            self._original_randomize_initial_state = self.unwrapped.randomize_initial_state
            self.unwrapped.randomize_initial_state = self._wrapped_randomize_initial_state
        else:
            raise ValueError(
                "[PlateSpawnOffsetWrapper] Environment missing required "
                "randomize_initial_state method"
            )

        self._wrapper_initialized = True

    def _wrapped_randomize_initial_state(self, env_ids):
        """
        Randomize initial state and shift plate so target hole is at standard position.

        1. Let base env do ALL the work (position, orientation, obs frames, robot, etc.)
        2. Read generated fixed_pos and fixed_quat
        3. Compute world offset and shift plate position
        4. Store new plate position (applied after reset completes)
        5. Do NOT modify observation frames (they already point to target hole)
        """
        # (1) Let base env do ALL the work
        self._original_randomize_initial_state(env_ids)

        # (2) Read the generated plate position and orientation
        fixed_pos = self.unwrapped.fixed_pos[env_ids].clone()
        fixed_quat = self.unwrapped.fixed_quat[env_ids].clone()

        # (3) Transform goal_offset from local to world frame
        local_offset = self.goal_offset_local.unsqueeze(0).expand(len(env_ids), -1)
        world_offset = torch_utils.quat_rotate(fixed_quat, local_offset)

        # (4) Compute new plate position: shift so target hole is at standard position
        # goal_offset points FROM plate_center TO hex_hole, so:
        # hex_hole = plate_center + world_offset
        # To put hex_hole at S (fixed_pos): plate_center = S - world_offset
        new_plate_pos = fixed_pos.clone()
        new_plate_pos[:, 0:2] -= world_offset[:, 0:2]  # XY offset only

        # (5) Store the new position to be applied after reset completes
        # (The base env has already written positions to sim during randomize_initial_state,
        # so we need to write again after reset() returns)
        self._pending_plate_pos = new_plate_pos.clone()
        self._pending_plate_quat = fixed_quat.clone()
        self._pending_env_ids = env_ids.clone()

        # NOTE: Do NOT modify fixed_pos_obs_frame or fixed_pos_action_frame!
        # They already point to the "standard" position, which is now where
        # the target hole is located. This is the desired behavior.

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

        # Apply pending plate position update AFTER reset completes
        if self._pending_plate_pos is not None:
            env_ids = self._pending_env_ids
            new_plate_pos = self._pending_plate_pos
            fixed_quat = self._pending_plate_quat

            # Convert to world coordinates
            env_origins = self.unwrapped.scene.env_origins[env_ids]
            new_plate_pos_world = new_plate_pos + env_origins

            # Update internal fixed_pos buffer to keep state consistent
            self.unwrapped.fixed_pos[env_ids] = new_plate_pos

            # Write new plate position to simulation
            self.unwrapped._fixed_asset.write_root_pose_to_sim(
                torch.cat([new_plate_pos_world, fixed_quat], dim=-1),
                env_ids=env_ids
            )

            # Also write root state (includes velocities) to ensure update takes effect
            root_state = self.unwrapped._fixed_asset.data.root_state_w[env_ids].clone()
            root_state[:, 0:3] = new_plate_pos_world
            root_state[:, 3:7] = fixed_quat
            root_state[:, 7:13] = 0  # Zero velocities
            self.unwrapped._fixed_asset.write_root_state_to_sim(root_state, env_ids=env_ids)

            # Clear pending update
            self._pending_plate_pos = None
            self._pending_plate_quat = None
            self._pending_env_ids = None

        return obs, info
