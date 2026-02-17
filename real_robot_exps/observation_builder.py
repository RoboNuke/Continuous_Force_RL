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
        device: Torch device.
    """

    def __init__(
        self,
        obs_order: list,
        action_dim: int,
        use_tanh_ft_scaling: bool = False,
        tanh_ft_scale: float = 0.03,
        contact_force_threshold: float = 1.5,
        device: str = "cpu",
    ):
        self.obs_order = list(obs_order)
        self.action_dim = action_dim
        self.use_tanh_ft_scaling = use_tanh_ft_scaling
        self.tanh_ft_scale = tanh_ft_scale
        self.contact_force_threshold = contact_force_threshold
        self.device = device

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
        print(f"[ObservationBuilder] Dimension validated: {self.obs_dim} matches checkpoint")

    def build_observation(
        self,
        robot_interface,
        goal_position: torch.Tensor,
        prev_actions: torch.Tensor,
        fixed_yaw_offset: float = 0.0,
    ) -> torch.Tensor:
        """Construct observation vector from robot state.

        Args:
            robot_interface: FrankaROS2Interface instance.
            goal_position: [3] fixed asset position (for relative observations).
                          This is fixed_pos in sim, NOT target_peg_base_pos.
            prev_actions: [action_dim] previous EMA-smoothed actions.
            fixed_yaw_offset: Yaw offset for fingertip_yaw_rel_fixed (radians).

        Returns:
            [obs_dim] observation tensor (single environment, unbatched).
        """
        # Read robot state
        ee_pos = robot_interface.get_ee_position()       # [3]
        ee_quat = robot_interface.get_ee_orientation()    # [4] (w,x,y,z)
        ee_linvel = robot_interface.get_ee_linear_velocity()   # [3]
        ee_angvel = robot_interface.get_ee_angular_velocity()  # [3]
        force_torque = robot_interface.get_force_torque()      # [6]
        joint_pos = robot_interface.get_joint_positions()       # [7]

        # Build component dictionary
        components = {}

        # Absolute EE position
        components["fingertip_pos"] = ee_pos

        # Relative EE position (ee_pos - fixed_asset_position)
        # This matches: fingertip_midpoint_pos - (fixed_pos_obs_frame + init_fixed_pos_obs_noise)
        # On real robot there's no obs noise - goal_position includes any calibration offset
        components["fingertip_pos_rel_fixed"] = ee_pos - goal_position

        # EE orientation quaternion
        components["fingertip_quat"] = ee_quat

        # Relative yaw (if in obs_order)
        if "fingertip_yaw_rel_fixed" in self.obs_order:
            ee_yaw = _quat_to_yaw(ee_quat)
            # Normalize yaw difference similar to sim
            yaw_rel = ee_yaw + fixed_yaw_offset
            components["fingertip_yaw_rel_fixed"] = torch.tensor(
                [yaw_rel], device=self.device, dtype=torch.float32
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
            ft_obs = force_torque
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
            print(f"[ObservationNormalizer] Sliced preprocessor stats: "
                  f"policy obs_dim={obs_dim} from full dim={full_dim}")
        elif obs_dim is not None and obs_dim != full_dim:
            raise ValueError(
                f"obs_dim={obs_dim} is larger than preprocessor dim={full_dim}. "
                f"This should not happen — check obs_order reconstruction."
            )
        else:
            self.running_mean = full_mean
            self.running_variance = full_var
            self.obs_dim = full_dim

        count = checkpoint_preprocessor.get("current_count", torch.tensor(1.0))
        print(f"[ObservationNormalizer] Loaded frozen stats: obs_dim={self.obs_dim}, "
              f"sample_count={count.item():.0f}")

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observation using frozen training statistics.

        Args:
            obs: [batch_size, obs_dim] or [obs_dim] observation tensor.

        Returns:
            Normalized observation tensor with same shape.
        """
        return (obs - self.running_mean) / torch.sqrt(self.running_variance + self.eps)


def _quat_to_yaw(quat: torch.Tensor) -> float:
    """Extract yaw angle from quaternion (w, x, y, z).

    Uses the standard aerospace ZYX Euler angle extraction for yaw.

    Args:
        quat: [4] quaternion tensor (w, x, y, z).

    Returns:
        Yaw angle in radians.
    """
    w, x, y, z = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
    # ZYX Euler: yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw
