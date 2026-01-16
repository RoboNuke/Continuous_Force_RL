"""
Keypoint Offset Wrapper for Factory Environment

Replaces the default z-axis-only keypoint distribution with customizable patterns:
- Axis mode: 3D cross pattern with keypoints along all 3 axes
- Polygon mode: Radial lines arranged in a polygon pattern around z-axis

Note: The peg points downward and hole points upward, creating a ~180 degree
orientation difference. To compensate, we negate X/Y offsets for the fixed
keypoints so they align correctly in world space.
"""

import torch
import gymnasium as gym
from typing import Dict, Any, List
from isaaclab.utils.math import combine_frame_transforms


class KeypointOffsetWrapper(gym.Wrapper):
    """
    Wrapper that replaces default keypoint offsets with custom patterns.

    This wrapper MUST be applied early in the wrapper chain, before any
    wrappers that iterate over keypoint_offsets or modify keypoint calculations.

    Supported modes:
    - 'axis': 3D cross pattern with keypoints along X, Y, and Z axes
    - 'polygon': Radial lines arranged in polygon pattern around Z-axis
    """

    def __init__(self, env, config: Dict[str, Any]):
        """
        Initialize the keypoint offset wrapper.

        Args:
            env: Base environment to wrap
            config: Dictionary containing configuration parameters:
                - enabled: bool - Enable/disable wrapper
                - mode: str - 'axis' or 'polygon'
                Axis mode:
                - num_keypoints: int - Keypoints per axis
                - x_scale, y_scale, z_scale: float - Scale for each axis
                Polygon mode:
                - num_lines: int - Number of radial lines (polygon sides)
                - num_keypoints_z: int - Keypoints along z per radial position
                - num_keypoints_radial: int - Keypoints from center to edge
                - xy_scale: float - Max radial distance (edge)
                - z_scale: float - Scale for z-axis
        """
        super().__init__(env)

        self.config = config
        self.enabled = config.get('enabled', False)
        self._debug_step_count = 0

        if not self.enabled:
            return

        # Validate mode
        self.mode = config.get('mode', 'axis')
        if self.mode not in ['axis', 'polygon']:
            raise ValueError(f"Invalid keypoint mode: {self.mode}. Must be 'axis' or 'polygon'.")

        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        self._wrapper_initialized = False
        self._original_compute_intermediate_values = None

        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def _initialize_wrapper(self):
        """Initialize the wrapper after the base environment is set up."""
        if self._wrapper_initialized or not self.enabled:
            return

        # # DEBUG: Store original keypoint_offsets for debugging
        # original_offsets = self.unwrapped.keypoint_offsets
        # original_count = len(original_offsets)
        # print(f"\n{'='*60}")
        # print(f"[KeypointOffsetWrapper] DEBUG: BEFORE OVERRIDE")
        # print(f"{'='*60}")
        # print(f"  Original keypoint count: {original_count}")
        # print(f"  Original keypoint offsets (local coordinates):")
        # for i, offset in enumerate(original_offsets):
        #     print(f"    [{i}]: ({offset[0].item():.6f}, {offset[1].item():.6f}, {offset[2].item():.6f})")

        # Generate new keypoint offsets based on mode
        if self.mode == 'axis':
            new_offsets = self._generate_axis_keypoints()
        else:  # polygon
            new_offsets = self._generate_polygon_keypoints()

        # Create fixed offsets with negated X/Y to compensate for 180-degree rotation
        # The peg points down, hole points up - this negation aligns them in world space
        self.keypoint_offsets_held = new_offsets
        self.keypoint_offsets_fixed = []
        for offset in new_offsets:
            # Negate X and Y, keep Z the same
            fixed_offset = torch.tensor([-offset[0].item(), -offset[1].item(), offset[2].item()], device=self.device)
            self.keypoint_offsets_fixed.append(fixed_offset)

        # Replace keypoint_offsets in base environment (used for held)
        self.unwrapped.keypoint_offsets = new_offsets

        # Resize keypoint tensors to match new count
        num_keypoints = len(new_offsets)
        self.unwrapped.keypoints_held = torch.zeros(
            (self.num_envs, num_keypoints, 3),
            device=self.device
        )
        self.unwrapped.keypoints_fixed = torch.zeros(
            (self.num_envs, num_keypoints, 3),
            device=self.device
        )

        # Store and override _compute_intermediate_values to use our fixed offsets
        self._original_compute_intermediate_values = self.unwrapped._compute_intermediate_values
        self.unwrapped._compute_intermediate_values = self._wrapped_compute_intermediate_values

        # Store identity quaternion for transforms
        self.identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # # DEBUG: Print keypoint offsets after override
        # print(f"\n{'='*60}")
        # print(f"[KeypointOffsetWrapper] DEBUG: AFTER OVERRIDE")
        # print(f"{'='*60}")
        # print(f"  Mode: '{self.mode}'")
        # print(f"  New keypoint count: {num_keypoints}")
        # print(f"  Keypoint offsets HELD (local coordinates):")
        # for i, offset in enumerate(new_offsets):
        #     print(f"    [{i}]: ({offset[0].item():.6f}, {offset[1].item():.6f}, {offset[2].item():.6f})")
        # print(f"  Keypoint offsets FIXED (X/Y negated for 180-deg compensation):")
        # for i, offset in enumerate(self.keypoint_offsets_fixed):
        #     print(f"    [{i}]: ({offset[0].item():.6f}, {offset[1].item():.6f}, {offset[2].item():.6f})")
        # print(f"{'='*60}\n")

        print(f"[KeypointOffsetWrapper] Initialized: mode='{self.mode}', {num_keypoints} keypoints")

        self._wrapper_initialized = True
        self._debug_step_count = 0

    def _wrapped_compute_intermediate_values(self, dt):
        """
        Wrapped _compute_intermediate_values that recomputes keypoints_fixed
        using negated X/Y offsets to compensate for the 180-degree orientation
        difference between peg (pointing down) and hole (pointing up).
        """
        # Call original to compute everything including keypoints_held
        self._original_compute_intermediate_values(dt)

        # Recompute keypoints_fixed using our negated X/Y offsets
        env = self.unwrapped
        for idx, keypoint_offset in enumerate(self.keypoint_offsets_fixed):
            env.keypoints_fixed[:, idx] = combine_frame_transforms(
                env.target_held_base_pos,
                env.target_held_base_quat,
                keypoint_offset.repeat(self.num_envs, 1),
                self.identity_quat
            )[0]  # [0] is position from combine_frame_transforms

        # Recompute keypoint_dist with corrected fixed keypoints
        env.keypoint_dist = torch.norm(env.keypoints_held - env.keypoints_fixed, p=2, dim=-1).mean(-1)

    def _generate_axis_keypoints(self) -> List[torch.Tensor]:
        """
        Generate keypoints along all 3 axes forming a 3D cross pattern.

        With num_keypoints=4, creates 12 total keypoints (4 per axis).
        Each axis has its own scale factor.

        Returns:
            List of torch.Tensor offsets, each of shape (3,)
        """
        num_keypoints = self.config.get('num_keypoints', 4)
        x_scale = self.config.get('x_scale', 0.01)
        y_scale = self.config.get('y_scale', 0.01)
        z_scale = self.config.get('z_scale', 0.05)

        offsets = []

        # Linspace from -0.5 to 0.5, centered at origin
        t = torch.linspace(-0.5, 0.5, num_keypoints, device=self.device)

        # X-axis keypoints: vary x, y=0, z=0
        for i in range(num_keypoints):
            offsets.append(torch.tensor([t[i].item() * x_scale, 0.0, 0.0], device=self.device))

        # Y-axis keypoints: x=0, vary y, z=0
        for i in range(num_keypoints):
            offsets.append(torch.tensor([0.0, t[i].item() * y_scale, 0.0], device=self.device))

        # Z-axis keypoints: x=0, y=0, vary z
        for i in range(num_keypoints):
            offsets.append(torch.tensor([0.0, 0.0, t[i].item() * z_scale], device=self.device))

        return offsets

    def _generate_polygon_keypoints(self) -> List[torch.Tensor]:
        """
        Generate keypoints as radial lines in XY plane plus a separate Z line.

        Structure:
        - XY radial pattern: num_lines radial lines at z=0, each with num_keypoints_radial points
        - Z line: num_keypoints_z points along z-axis through center

        Radial points are evenly spaced from 0 to xy_scale, excluding zero to avoid
        overlapping points at center (uses linspace then skips first element).

        With num_lines=6, num_keypoints_radial=4, num_keypoints_z=4:
        Total keypoints = 6 * 4 + 4 = 28

        Returns:
            List of torch.Tensor offsets, each of shape (3,)
        """
        num_lines = self.config.get('num_lines', 6)
        num_keypoints_z = self.config.get('num_keypoints_z', 4)
        num_keypoints_radial = self.config.get('num_keypoints_radial', 4)
        xy_scale = self.config.get('xy_scale', 0.005)
        z_scale = self.config.get('z_scale', 0.05)

        offsets = []

        # Radial positions: evenly spaced from 0 to xy_scale, excluding zero
        # linspace(0, xy_scale, n+1)[1:] gives n points that skip zero
        r_vals = torch.linspace(0, xy_scale, num_keypoints_radial + 1, device=self.device)[1:]

        # Angles for polygon vertices (exclude 2*pi which equals 0)
        angles = torch.linspace(0, 2 * torch.pi, num_lines + 1, device=self.device)[:-1]

        # XY radial pattern at z=0
        for angle in angles:
            for r in r_vals:
                # Convert polar to cartesian
                x = r * torch.cos(angle)
                y = r * torch.sin(angle)
                offsets.append(torch.tensor([x.item(), y.item(), 0.0], device=self.device))

        # Separate Z line through center
        z_vals = torch.linspace(-0.5, 0.5, num_keypoints_z, device=self.device) * z_scale
        for z in z_vals:
            offsets.append(torch.tensor([0.0, 0.0, z.item()], device=self.device))

        return offsets

    def _print_debug_keypoint_info(self):
        """Print debug info about keypoint positions and distances."""
        env = self.unwrapped

        print(f"\n{'='*60}")
        print(f"[KeypointOffsetWrapper] DEBUG: STEP {self._debug_step_count} KEYPOINT INFO")
        print(f"{'='*60}")

        # Print the poses used for transforms
        print(f"  HELD object pose (env 0):")
        held_pos = env.held_base_pos[0]
        held_quat = env.held_base_quat[0]
        print(f"    pos: ({held_pos[0].item():.6f}, {held_pos[1].item():.6f}, {held_pos[2].item():.6f})")
        print(f"    quat (wxyz): ({held_quat[0].item():.4f}, {held_quat[1].item():.4f}, {held_quat[2].item():.4f}, {held_quat[3].item():.4f})")

        print(f"  TARGET/FIXED pose (env 0):")
        target_pos = env.target_held_base_pos[0]
        target_quat = env.target_held_base_quat[0]
        print(f"    pos: ({target_pos[0].item():.6f}, {target_pos[1].item():.6f}, {target_pos[2].item():.6f})")
        print(f"    quat (wxyz): ({target_quat[0].item():.4f}, {target_quat[1].item():.4f}, {target_quat[2].item():.4f}, {target_quat[3].item():.4f})")

        # Compute position and orientation errors
        pos_error = torch.norm(held_pos - target_pos).item()
        # Quaternion dot product gives cos(theta/2) where theta is rotation angle
        quat_dot = torch.abs(torch.dot(held_quat, target_quat)).item()
        quat_dot = min(quat_dot, 1.0)  # Clamp for numerical stability
        angle_error_rad = 2 * torch.acos(torch.tensor(quat_dot)).item()
        angle_error_deg = angle_error_rad * 180 / 3.14159

        print(f"  ERRORS (env 0):")
        print(f"    Position error: {pos_error:.6f} m")
        print(f"    Orientation error: {angle_error_deg:.2f} deg")

        # Print keypoint_dist (used in reward calculation)
        keypoint_dist = env.keypoint_dist
        print(f"\n  keypoint_dist (env 0): {keypoint_dist[0].item():.6f}")
        print(f"  keypoint_dist mean: {keypoint_dist.mean().item():.6f}")

        # Print world positions of keypoints for env 0
        print(f"\n  Keypoints HELD (world coords, env 0):")
        for i in range(min(len(env.keypoint_offsets), 12)):  # Limit to first 12 for readability
            pos = env.keypoints_held[0, i]
            print(f"    [{i}]: ({pos[0].item():.6f}, {pos[1].item():.6f}, {pos[2].item():.6f})")
        if len(env.keypoint_offsets) > 12:
            print(f"    ... and {len(env.keypoint_offsets) - 12} more keypoints")

        print(f"\n  Keypoints FIXED/TARGET (world coords, env 0):")
        for i in range(min(len(env.keypoint_offsets), 12)):
            pos = env.keypoints_fixed[0, i]
            print(f"    [{i}]: ({pos[0].item():.6f}, {pos[1].item():.6f}, {pos[2].item():.6f})")
        if len(env.keypoint_offsets) > 12:
            print(f"    ... and {len(env.keypoint_offsets) - 12} more keypoints")

        # Print per-keypoint distances
        print(f"\n  Per-keypoint L2 distances (env 0):")
        per_kp_dist = torch.norm(env.keypoints_held[0] - env.keypoints_fixed[0], p=2, dim=-1)
        for i in range(min(len(env.keypoint_offsets), 12)):
            print(f"    [{i}]: {per_kp_dist[i].item():.6f}")
        if len(env.keypoint_offsets) > 12:
            print(f"    ... and {len(env.keypoint_offsets) - 12} more")
        print(f"  Mean of per-keypoint distances: {per_kp_dist.mean().item():.6f}")

        print(f"{'='*60}\n")

    def step(self, action):
        """Execute one environment step."""
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        obs, reward, terminated, truncated, info = super().step(action)

        # # DEBUG: Print keypoint_dist and errors every step
        # if self.enabled:
        #     env = self.unwrapped
        #
        #     # Compute errors
        #     held_pos = env.held_base_pos[0]
        #     held_quat = env.held_base_quat[0]
        #     target_pos = env.target_held_base_pos[0]
        #     target_quat = env.target_held_base_quat[0]
        #
        #     pos_error = torch.norm(held_pos - target_pos).item()
        #     quat_dot = torch.abs(torch.dot(held_quat, target_quat)).item()
        #     quat_dot = min(quat_dot, 1.0)
        #     angle_error_rad = 2 * torch.acos(torch.tensor(quat_dot)).item()
        #     angle_error_deg = angle_error_rad * 180 / 3.14159
        #
        #     # Print keypoint_dist and errors every step
        #     keypoint_dist = env.keypoint_dist
        #     print(f"[KeypointOffsetWrapper] Step {self._debug_step_count}: "
        #           f"keypoint_dist={keypoint_dist[0].item():.6f}, "
        #           f"pos_err={pos_error:.6f}m, "
        #           f"rot_err={angle_error_deg:.2f}deg")
        #
        #     # Full debug output for first 3 steps only
        #     if self._debug_step_count < 3:
        #         self._print_debug_keypoint_info()
        #
        #     self._debug_step_count += 1

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and ensure wrapper is initialized."""
        # Reset debug step counter on each reset
        if self.enabled:
            self._debug_step_count = 0

        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()
        return super().reset(**kwargs)
