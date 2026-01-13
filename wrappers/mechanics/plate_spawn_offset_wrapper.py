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
- fixed_pos is NOT modified (it stays at target hole location for reward/success)
"""

import torch
import gymnasium as gym

# Import Isaac Lab utilities for quaternion rotation
try:
    import omni.isaac.lab.utils.math as torch_utils
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
except ImportError:
    try:
        import isaaclab.utils.math as torch_utils
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
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

    This wrapper intercepts _reset_idx to ensure plate offset is applied
    for both full resets and partial resets during rollouts.
    """

    def __init__(self, env):
        """
        Initialize the plate spawn offset wrapper.

        Args:
            env: Base environment to wrap
        """
        super().__init__(env)

        self._wrapper_initialized = False
        self._original_reset_idx = None
        self._original_compute_intermediate_values = None

        # Storage for target hole position (to restore fixed_pos after base env overwrites it)
        # This gets populated during reset and used every frame in _compute_intermediate_values
        self._target_hole_pos = None

        # Flag to skip fixed_pos restoration during reset
        # (because _compute_intermediate_values is called DURING reset before we can store the value)
        self._in_reset = False

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

        # Debug print (disabled - uncomment to enable)
        # print(f"[PlateSpawnOffsetWrapper] Initialized with offset "
        #       f"({goal_offset[0]*1000:.1f}mm, {goal_offset[1]*1000:.1f}mm)")

        # Debug marker storage (local coordinates)
        self._debug_goal_before = None
        self._debug_plate_center_before = None
        self._debug_plate_center_final = None
        self._debug_goal_final = None
        self._debug_env_ids = None
        self._debug_fixed_quat = None

        # Create debug visualization markers (disabled - uncomment to enable)
        self._debug_markers = None
        # self._create_debug_markers()

        # Initialize if environment is ready
        if hasattr(self.unwrapped, '_reset_idx'):
            self._initialize_wrapper()

    def _create_debug_markers(self):
        """Create debug visualization markers for plate offset debugging."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/PlateOffsetDebug",
            markers={
                "goal_before": sim_utils.SphereCfg(
                    radius=0.010,  # 10mm sphere
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0)  # Red
                    ),
                ),
                "plate_center_before": sim_utils.SphereCfg(
                    radius=0.010,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.0, 1.0)  # Blue
                    ),
                ),
                "plate_center_final": sim_utils.SphereCfg(
                    radius=0.010,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0)  # Green
                    ),
                ),
                "goal_final": sim_utils.SphereCfg(
                    radius=0.010,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 1.0, 0.0)  # Yellow
                    ),
                ),
            },
        )
        self._debug_markers = VisualizationMarkers(marker_cfg)
        print("[PlateSpawnOffsetWrapper] Debug markers created:")
        print("  - Red: Goal before changes")
        print("  - Blue: Plate center before changes")
        print("  - Green: Plate center final")
        print("  - Yellow: Goal final")

    def _initialize_wrapper(self):
        """Initialize the wrapper after the base environment is set up."""
        if self._wrapper_initialized:
            return

        # Override _reset_idx to intercept all resets (both full and partial)
        if hasattr(self.unwrapped, '_reset_idx'):
            self._original_reset_idx = self.unwrapped._reset_idx
            self.unwrapped._reset_idx = self._wrapped_reset_idx
        else:
            raise ValueError(
                "[PlateSpawnOffsetWrapper] Environment missing required "
                "_reset_idx method"
            )

        # Override _compute_intermediate_values to restore fixed_pos every frame
        # The base env overwrites fixed_pos from simulation data, which would point
        # to the actual plate center instead of the target hole location.
        if hasattr(self.unwrapped, '_compute_intermediate_values'):
            self._original_compute_intermediate_values = self.unwrapped._compute_intermediate_values
            self.unwrapped._compute_intermediate_values = self._wrapped_compute_intermediate_values
        else:
            raise ValueError(
                "[PlateSpawnOffsetWrapper] Environment missing required "
                "_compute_intermediate_values method"
            )

        # Initialize target hole position tensor
        self._target_hole_pos = torch.zeros(
            (self.unwrapped.num_envs, 3),
            dtype=torch.float32,
            device=self.unwrapped.device
        )

        self._wrapper_initialized = True
        # Debug print (disabled - uncomment to enable)
        # print("[PlateSpawnOffsetWrapper] Wrapper initialized (using _reset_idx interface)")

    def _wrapped_reset_idx(self, env_ids):
        """
        Wrapped reset that shifts plate so target hole is at standard position.

        This method intercepts _reset_idx to ensure plate offset is applied
        for both full resets and partial resets during rollouts.

        1. Let base env do ALL the work (position, orientation, obs frames, robot, etc.)
        2. Store target hole position (fixed_pos at this point, before we move plate)
        3. Read generated plate position and orientation
        4. Compute world offset and shift plate position
        5. Write new plate position to simulation
        6. Do NOT modify fixed_pos (it stays at target hole location for rewards)
        """
        # Set flag to skip fixed_pos restoration during reset
        # (_compute_intermediate_values is called DURING reset before we can store the value)
        self._in_reset = True

        # (1) Let base env do ALL the work
        self._original_reset_idx(env_ids)

        # (2) Store target hole position AFTER base env has set it
        # At this point, fixed_pos = standard position (where target hole will be)
        # This is used by _wrapped_compute_intermediate_values to restore fixed_pos
        # after the base env overwrites it from simulation data every frame.
        self._target_hole_pos[env_ids] = self.unwrapped.fixed_pos[env_ids].clone()

        # Clear the reset flag now that we have the correct value stored
        self._in_reset = False

        # (3) Read the generated plate position and orientation from simulation
        env_origins = self.unwrapped.scene.env_origins[env_ids]
        fixed_pos_world = self.unwrapped._fixed_asset.data.root_pos_w[env_ids].clone()
        fixed_pos = fixed_pos_world - env_origins  # Convert to local coords
        fixed_quat = self.unwrapped._fixed_asset.data.root_quat_w[env_ids].clone()

        # Store debug positions BEFORE any changes
        self._debug_plate_center_before = fixed_pos.clone()
        self._debug_goal_before = fixed_pos.clone()
        self._debug_env_ids = env_ids.clone()
        self._debug_fixed_quat = fixed_quat.clone()

        # (3) Transform goal_offset from local to world frame
        local_offset = self.goal_offset_local.unsqueeze(0).expand(len(env_ids), -1)
        world_offset = torch_utils.quat_rotate(fixed_quat, local_offset)

        # (4) Compute new plate position: shift so target hole is at standard position
        # goal_offset points FROM plate_center TO hex_hole, so:
        # hex_hole = plate_center + world_offset
        # To put hex_hole at S (fixed_pos): plate_center = S - world_offset
        new_plate_pos = fixed_pos.clone()
        new_plate_pos[:, 0:2] -= world_offset[:, 0:2]  # XY offset only

        # Store debug positions AFTER changes
        self._debug_plate_center_final = new_plate_pos.clone()
        self._debug_goal_final = fixed_pos.clone()

        # (5) Convert to world coordinates and write to simulation
        new_plate_pos_world = new_plate_pos + env_origins

        # NOTE: Do NOT update fixed_pos here!
        # fixed_pos should remain at the original position (where target hole ends up)
        # because reward/success calculations use fixed_pos to compute target_held_base_pos.
        # The plate is physically moved, but fixed_pos stays at the target hole location.

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

        # Visualize debug markers
        self._visualize_debug_markers(env_ids)

    def _wrapped_compute_intermediate_values(self, dt):
        """
        Wrap _compute_intermediate_values to restore fixed_pos after base env updates it.

        The base env's _compute_intermediate_values() updates fixed_pos from simulation:
            self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins

        For multi-hole plates, this would set fixed_pos to the actual plate center
        (which we moved), not the target hole location. We need to restore fixed_pos
        to the target hole position that we stored during reset.

        This ensures reward/success calculations and critic observations use the
        correct target hole position, not the plate center.
        """
        # Let base env compute all intermediate values (including overwriting fixed_pos)
        self._original_compute_intermediate_values(dt)

        # Skip restoration during reset - we haven't stored the correct value yet
        # (_compute_intermediate_values is called DURING _reset_idx before we can store it)
        if self._in_reset:
            return

        # Restore fixed_pos to the target hole position
        # This overwrites what the base env set from simulation data
        if self._target_hole_pos is not None:
            self.unwrapped.fixed_pos[:] = self._target_hole_pos

            # Also recompute target_held_base_pos since it was computed with wrong fixed_pos
            # This is critical for engagement/success detection in _get_curr_successes
            # Note: combine_frame_transforms takes (t01, q01, t12, q12) and returns (pos, quat)
            # whereas tf_combine takes (q01, t01, q12, t12) and returns (quat, pos)
            self.unwrapped.target_held_base_pos[:], self.unwrapped.target_held_base_quat[:] = torch_utils.combine_frame_transforms(
                self.unwrapped.fixed_pos,
                self.unwrapped.fixed_quat,
                self.unwrapped.fixed_success_pos_local,
                self.unwrapped.identity_quat
            )

            # Also recompute keypoints_fixed since they depend on target_held_base_pos
            for idx, keypoint_offset in enumerate(self.unwrapped.keypoint_offsets):
                self.unwrapped.keypoints_fixed[:, idx] = torch_utils.combine_frame_transforms(
                    self.unwrapped.target_held_base_pos,
                    self.unwrapped.target_held_base_quat,
                    keypoint_offset.repeat(self.unwrapped.num_envs, 1),
                    self.unwrapped.identity_quat
                )[0]  # [0] is position (combine_frame_transforms returns pos, quat)

            # Recompute keypoint_dist
            self.unwrapped.keypoint_dist = torch.norm(
                self.unwrapped.keypoints_held - self.unwrapped.keypoints_fixed, p=2, dim=-1
            ).mean(-1)

    def step(self, action):
        """Step the environment and ensure wrapper is initialized."""
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_reset_idx'):
            self._initialize_wrapper()

        return super().step(action)

    def reset(self, **kwargs):
        """Reset the environment and ensure wrapper is initialized."""
        # Initialize BEFORE super().reset() so _reset_idx override is in place
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_reset_idx'):
            self._initialize_wrapper()

        return super().reset(**kwargs)

    def _visualize_debug_markers(self, env_ids):
        """Visualize the 4 debug markers showing goal and plate positions."""
        if self._debug_markers is None:
            return

        if (self._debug_goal_before is None or
            self._debug_plate_center_before is None or
            self._debug_plate_center_final is None or
            self._debug_goal_final is None):
            return

        num_envs = len(env_ids)
        env_origins = self.unwrapped.scene.env_origins[env_ids]

        # Z offset to place markers above the hole for visibility
        z_offset = 0.02  # 20mm above

        # Convert local positions to world coordinates
        plate_center_before_world = self._debug_plate_center_before + env_origins
        plate_center_final_world = self._debug_plate_center_final + env_origins

        # Goal before = base env's intended goal position (fixed_pos, the standard position)
        # Use stored value directly - do NOT add goal_offset
        goal_before_world = self._debug_goal_before + env_origins

        # Goal final = target hole after plate shift (plate_center_final + goal_offset)
        # This should end up at the same location as goal_before, confirming correct placement
        fixed_quat = self._debug_fixed_quat
        local_offset = self.goal_offset_local.unsqueeze(0).expand(num_envs, -1)
        world_offset_goal = torch_utils.quat_rotate(fixed_quat, local_offset)

        goal_final_world = plate_center_final_world.clone()
        goal_final_world[:, 0:2] += world_offset_goal[:, 0:2]

        # Apply Z offset to all markers for visibility
        goal_before_world[:, 2] += z_offset
        plate_center_before_world[:, 2] += z_offset
        plate_center_final_world[:, 2] += z_offset
        goal_final_world[:, 2] += z_offset

        # Stack all marker positions: [num_envs * 4, 3]
        all_positions = torch.cat([
            goal_before_world,
            plate_center_before_world,
            plate_center_final_world,
            goal_final_world,
        ], dim=0)

        # Create marker indices
        marker_indices = torch.cat([
            torch.zeros(num_envs, dtype=torch.int32, device=all_positions.device),
            torch.ones(num_envs, dtype=torch.int32, device=all_positions.device),
            torch.full((num_envs,), 2, dtype=torch.int32, device=all_positions.device),
            torch.full((num_envs,), 3, dtype=torch.int32, device=all_positions.device),
        ], dim=0)

        # Visualize all markers
        self._debug_markers.visualize(
            translations=all_positions,
            marker_indices=marker_indices,
        )
