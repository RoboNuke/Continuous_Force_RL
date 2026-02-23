"""
Observation Builder for Real Robot Evaluation

Constructs observation tensors matching the exact format used during training.
Reads robot state from FrankaROS2Interface and concatenates components in the
same obs_order used by the IsaacLab factory environment + wrappers.

Two classes:
  - ObservationBuilder: Constructs raw obs tensor from robot state
  - ObservationNormalizer: Applies frozen RunningStandardScaler normalization

No Isaac Sim dependency - pure PyTorch.
"""

import torch
import math


# Dimension of each observation component (must match IsaacLab factory_env_cfg.py OBS_DIM_CFG)
OBS_DIM_MAP = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "fingertip_yaw_rel_fixed": 1,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "force_torque": 6,
    "in_contact": 3,
}


class ObservationBuilder:
    """Builds observation tensors matching training format from real robot state.

    On the real robot, natural sensor noise replaces simulated Gaussian noise.
    No noise is injected - the real sensor readings are used directly.

    Args:
        obs_order: List of observation component names in training order.
                   Loaded from WandB training config (cfg.obs_order).
        action_dim: Dimension of the action space (for prev_actions).
                    6 for pose-only, 2*force_size+6 for hybrid.
        use_tanh_ft_scaling: Whether tanh scaling was applied to force/torque
                             during training (from training config).
        tanh_ft_scale: Scale factor for tanh transform (default 0.03).
        contact_force_threshold: Force threshold for in_contact detection (N).
        fixed_asset_yaw: Known yaw angle of the fixed asset on the table (radians).
                         Used to compute fingertip_yaw_rel_fixed matching sim.
        ee_pose_noise_enabled: Whether EEPoseNoiseWrapper was enabled during training.
                               When True and use_tanh_ft_scaling is also True, tanh
                               scaling is auto-disabled because EEPoseNoiseWrapper
                               bypasses ForceTorqueWrapper's tanh in sim.
        device: Torch device.
    """

    def __init__(
        self,
        obs_order: list,
        action_dim: int,
        use_tanh_ft_scaling: bool = False,
        tanh_ft_scale: float = 0.03,
        contact_force_threshold: float = 1.5,
        fixed_asset_yaw: float = 0.0,
        ee_pose_noise_enabled: bool = False,
        device: str = "cpu",
    ):
        self.obs_order = list(obs_order)
        self.action_dim = action_dim
        self.use_tanh_ft_scaling = use_tanh_ft_scaling
        self.tanh_ft_scale = tanh_ft_scale
        self.contact_force_threshold = contact_force_threshold
        self.fixed_asset_yaw = fixed_asset_yaw
        self.device = device

        # Detect F/T tanh scaling mismatch: EEPoseNoiseWrapper bypasses
        # ForceTorqueWrapper's get_force_torque_observation() in sim, so tanh
        # scaling was never actually applied during training even if configured.
        if use_tanh_ft_scaling and ee_pose_noise_enabled:
            print("\033[91m[WARNING] use_tanh_ft_scaling=True but ee_pose_noise was enabled during training.")
            print("  EEPoseNoiseWrapper bypasses ForceTorqueWrapper's tanh scaling in sim.")
            print("  The policy trained on RAW F/T values. Disabling tanh to match.\033[0m")
            self.use_tanh_ft_scaling = False

        # Validate all obs components are known
        for obs_name in self.obs_order:
            if obs_name not in OBS_DIM_MAP:
                raise ValueError(
                    f"Unknown observation component '{obs_name}' in obs_order. "
                    f"Known components: {list(OBS_DIM_MAP.keys())}"
                )

        # Calculate expected obs dimension (obs_order components + prev_actions)
        self.obs_dim = sum(OBS_DIM_MAP[name] for name in self.obs_order) + action_dim
        print(f"[ObservationBuilder] obs_order={self.obs_order}")
        idx = 0
        for name in self.obs_order:
            dim = OBS_DIM_MAP[name]
            print(f"  [{idx}:{idx+dim}] {name} (dim={dim})")
            idx += dim
        print(f"  [{idx}:{idx+action_dim}] prev_actions (dim={action_dim})")
        idx += action_dim
        print(f"[ObservationBuilder] obs_dim={self.obs_dim} "
              f"(components={self.obs_dim - action_dim} + prev_actions={action_dim})")

    def validate_against_checkpoint(self, checkpoint_obs_dim: int):
        """Validate that our obs_dim matches the checkpoint's expected input size.

        Args:
            checkpoint_obs_dim: obs_dim from checkpoint state_preprocessor running_mean.

        Raises:
            ValueError: If dimensions don't match.
        """
        if self.obs_dim != checkpoint_obs_dim:
            raise ValueError(
                f"Observation dimension mismatch! "
                f"ObservationBuilder produces {self.obs_dim} but checkpoint expects {checkpoint_obs_dim}. "
                f"obs_order={self.obs_order}, action_dim={self.action_dim}. "
                f"Check that obs_order and action_dim match the training configuration."
            )

    def build_observation(
        self,
        snapshot,
        goal_position: torch.Tensor,
        prev_actions: torch.Tensor,
        fixed_yaw_offset: float = 0.0,
    ) -> torch.Tensor:
        """Construct observation vector from robot state snapshot.

        Args:
            snapshot: StateSnapshot namedtuple from FrankaInterface.
            goal_position: [3] observation frame position (hole tip + per-episode noise).
                          Matches sim's (fixed_pos_obs_frame + init_fixed_pos_obs_noise).
            prev_actions: [action_dim] previous EMA-smoothed actions (matching sim's
                          self.actions.clone(), which is EMA-smoothed in _pre_physics_step).
            fixed_yaw_offset: Per-episode yaw noise for fingertip_yaw_rel_fixed (radians).
                             Added to fixed_asset_yaw before computing relative yaw.

        Returns:
            [obs_dim] observation tensor (single environment, unbatched).
        """
        # Read state from snapshot
        ee_pos = snapshot.ee_pos            # [3]
        ee_quat = snapshot.ee_quat          # [4] (w,x,y,z)
        ee_linvel = snapshot.ee_linvel      # [3]
        ee_angvel = snapshot.ee_angvel      # [3]
        force_torque = snapshot.force_torque # [6]
        joint_pos = snapshot.joint_pos      # [7]

        # Build component dictionary
        components = {}

        # Absolute EE position
        components["fingertip_pos"] = ee_pos

        # Relative EE position (ee_pos - fixed_asset_position)
        # This matches: fingertip_midpoint_pos - (fixed_pos_obs_frame + init_fixed_pos_obs_noise)
        # On real robot there's no obs noise - goal_position includes any calibration offset
        components["fingertip_pos_rel_fixed"] = ee_pos - goal_position

        # EE orientation quaternion — fixed to [0,1,0,0] (policy-only override)
        components["fingertip_quat"] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device, dtype=torch.float32)

        # Relative yaw (if in obs_order)
        # Matches sim's EEPoseNoiseWrapper._compute_fingertip_yaw_rel_fixed:
        #   1. Un-rotate EE quat by 180° around X (gripper points down)
        #   2. Extract yaw from un-rotated quat
        #   3. Subtract fixed asset yaw (with per-episode noise offset)
        #   4. Wrap to [-pi, pi]
        if "fingertip_yaw_rel_fixed" in self.obs_order:
            fingertip_yaw = _fingertip_yaw_unrotated(ee_quat)
            noisy_fixed_yaw = self.fixed_asset_yaw + fixed_yaw_offset
            rel_yaw = fingertip_yaw - noisy_fixed_yaw
            # Wrap to [-pi, pi]
            if rel_yaw > math.pi:
                rel_yaw -= 2 * math.pi
            elif rel_yaw < -math.pi:
                rel_yaw += 2 * math.pi
            components["fingertip_yaw_rel_fixed"] = torch.tensor(
                [rel_yaw], device=self.device, dtype=torch.float32
            )

        # Velocities
        components["ee_linvel"] = ee_linvel
        components["ee_angvel"] = ee_angvel

        # Joint positions
        components["joint_pos"] = joint_pos

        # Force/torque (with optional tanh scaling matching training)
        if self.use_tanh_ft_scaling:
            ft_obs = torch.tanh(self.tanh_ft_scale * force_torque)
        else:
            ft_obs = force_torque.clone()
        # HACK: Zero out Tz (index 5) — real robot has ~-0.17 Nm bias that
        # reads as -43σ after normalization. Training saw ~0. Remove to test.
        #ft_obs[0] = 0.0
        #ft_obs[1] = 0.0
        #ft_obs[2] = 0.0
        ft_obs[3] = 0.0
        ft_obs[4] = 0.0
        ft_obs[5] = 0.0
        components["force_torque"] = ft_obs

        # Contact detection from force thresholds (matches ForceTorqueWrapper logic)
        # in_contact[:3] = force magnitude per axis > threshold
        force_magnitudes = force_torque[:3].abs()
        in_contact = (force_magnitudes >= self.contact_force_threshold).float()
        components["in_contact"] = in_contact

        # Concatenate in obs_order, then append prev_actions
        obs_parts = []
        for obs_name in self.obs_order:
            component = components[obs_name]
            if component.dim() == 0:
                component = component.unsqueeze(0)
            expected_dim = OBS_DIM_MAP[obs_name]
            if component.shape[0] != expected_dim:
                raise RuntimeError(
                    f"Observation component '{obs_name}' has dimension {component.shape[0]} "
                    f"but expected {expected_dim}"
                )
            obs_parts.append(component)

        obs_parts.append(prev_actions)
        obs = torch.cat(obs_parts, dim=0)

        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(
                f"Built observation has dimension {obs.shape[0]} but expected {self.obs_dim}. "
                f"This is a bug in ObservationBuilder."
            )

        return obs


class ObservationNormalizer:
    """Applies frozen RunningStandardScaler normalization from training checkpoint.

    Loads mean and variance from the checkpoint's state_preprocessor and applies:
        normalized = (obs - mean) / sqrt(var + eps)

    The normalizer is frozen (no updates during evaluation).

    Args:
        checkpoint_preprocessor: Dict with 'running_mean', 'running_variance',
                                 and 'current_count' from training checkpoint.
        device: Torch device.
        eps: Epsilon for numerical stability (default 1e-8, matching SKRL).
    """

    def __init__(self, checkpoint_preprocessor: dict, device: str = "cpu",
                 eps: float = 1e-8, obs_dim: int = None):
        self.device = device
        self.eps = eps

        if "running_mean" not in checkpoint_preprocessor:
            raise ValueError("Checkpoint state_preprocessor missing 'running_mean'")
        if "running_variance" not in checkpoint_preprocessor:
            raise ValueError("Checkpoint state_preprocessor missing 'running_variance'")

        full_mean = checkpoint_preprocessor["running_mean"].to(device).float()
        full_var = checkpoint_preprocessor["running_variance"].to(device).float()
        full_dim = full_mean.shape[0]

        # The preprocessor may contain stats for both policy and critic observations.
        # If obs_dim is provided, slice to only the policy's portion.
        if obs_dim is not None and obs_dim < full_dim:
            self.running_mean = full_mean[:obs_dim]
            self.running_variance = full_var[:obs_dim]
            self.obs_dim = obs_dim
        elif obs_dim is not None and obs_dim != full_dim:
            raise ValueError(
                f"obs_dim={obs_dim} is larger than preprocessor dim={full_dim}. "
                f"This should not happen — check obs_order reconstruction."
            )
        else:
            self.running_mean = full_mean
            self.running_variance = full_var
            self.obs_dim = full_dim


    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observation using frozen training statistics.

        Args:
            obs: [batch_size, obs_dim] or [obs_dim] observation tensor.

        Returns:
            Normalized observation tensor with same shape.
        """
        return (obs - self.running_mean) / torch.sqrt(self.running_variance + self.eps)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two (w, x, y, z) quaternions.

    Args:
        q1: [4] quaternion tensor (w, x, y, z).
        q2: [4] quaternion tensor (w, x, y, z).

    Returns:
        [4] product quaternion (w, x, y, z).
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# 180° rotation around X: quat_from_euler(roll=-pi, pitch=0, yaw=0) = (0, -1, 0, 0)
_UNROT_180X_QUAT = torch.tensor([0.0, -1.0, 0.0, 0.0], dtype=torch.float32)


def _fingertip_yaw_unrotated(ee_quat: torch.Tensor) -> float:
    """Extract yaw after un-rotating EE quaternion by 180° around X.

    The Franka gripper points down (180° X rotation from upright). The sim
    un-rotates this before extracting yaw so the yaw is in the "upright"
    frame. This function replicates that exact computation.

    Matches EEPoseNoiseWrapper._compute_fingertip_yaw_rel_fixed in sim.

    Args:
        ee_quat: [4] EE quaternion (w, x, y, z).

    Returns:
        Yaw angle in radians (in un-rotated frame).
    """
    unrot = _UNROT_180X_QUAT.to(device=ee_quat.device, dtype=ee_quat.dtype)
    q_unrot = _quat_mul(unrot, ee_quat)

    # Standard ZYX Euler yaw extraction
    w, x, y, z = q_unrot[0].item(), q_unrot[1].item(), q_unrot[2].item(), q_unrot[3].item()
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)
