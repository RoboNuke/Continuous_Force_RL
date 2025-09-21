#!/usr/bin/env python3
"""
Observation Noise Wrapper

This wrapper applies domain randomization noise to observations based on semantic
observation groups (fingertip_pos, joint_pos, etc.) rather than individual indices.

Key Features:
- Group-based noise application using factory cfg observation groups
- Configurable noise per observation group type
- Controlled noise timing (per step, per episode, or per policy update)
- Compatible with Isaac Lab factory environment configurations
- Works with history wrapper when placed before it in the stack
"""

import torch
import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field


@dataclass
class NoiseGroupConfig:
    """Configuration for noise applied to a specific observation group."""

    group_name: str
    """Name of the observation group (e.g., 'fingertip_pos', 'joint_pos')"""

    noise_type: str = "gaussian"
    """Type of noise: 'gaussian', 'uniform', 'none'"""

    std: Union[float, List[float]] = 0.01
    """Standard deviation for gaussian noise (scalar or per-dimension)"""

    mean: Union[float, List[float]] = 0.0
    """Mean for gaussian noise (scalar or per-dimension)"""

    scale: Union[float, List[float]] = 0.01
    """Scale for uniform noise (scalar or per-dimension)"""

    enabled: bool = True
    """Whether to apply noise to this group"""

    timing: str = "step"
    """When to apply noise: 'step', 'episode', 'policy_update'"""

    clip_range: Optional[tuple] = None
    """Optional clipping range for noisy observations"""


@dataclass
class ObservationNoiseConfig:
    """Complete configuration for observation noise wrapper."""

    # Noise group configurations
    noise_groups: Dict[str, NoiseGroupConfig] = field(default_factory=dict)

    # Global settings
    global_noise_scale: float = 1.0
    """Global multiplier for all noise"""

    policy_update_interval: int = 32
    """Steps between policy updates (for timing control)"""

    enabled: bool = True
    """Global enable/disable for all noise"""

    apply_to_critic: bool = True
    """Whether to apply noise to critic observations (set False for noise-free critic)"""

    seed: Optional[int] = None
    """Random seed for reproducible noise"""

    def add_group_noise(self, group_config: NoiseGroupConfig):
        """Add noise configuration for a specific group."""
        self.noise_groups[group_config.group_name] = group_config

    def disable_group(self, group_name: str):
        """Disable noise for a specific group."""
        if group_name in self.noise_groups:
            self.noise_groups[group_name].enabled = False


class ObservationNoiseWrapper(gym.Wrapper):
    """
    Wrapper that applies semantic group-based noise to observations.

    This wrapper understands Isaac Lab factory environment observation groups
    and applies appropriate noise to each group (position, orientation, velocity, etc.)
    rather than treating all observation dimensions uniformly.
    """

    def __init__(self, env, noise_config: ObservationNoiseConfig):
        """
        Initialize observation noise wrapper.

        Args:
            env: Base environment to wrap
            noise_config: Configuration for noise application
        """
        super().__init__(env)

        self.config = noise_config
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # Set random seed if specified
        if noise_config.seed is not None:
            try:
                torch.manual_seed(noise_config.seed)
            except RuntimeError as e:
                if "TORCH_LIBRARY" in str(e):
                    # Workaround for PyTorch library registration conflicts during testing
                    # This can happen when running many tests that import PyTorch modules
                    pass
                else:
                    raise
            np.random.seed(noise_config.seed)

        # Load observation group dimensions
        self.obs_dim_cfg, self.state_dim_cfg = self._load_observation_configs()

        # Build observation group mappings
        self.policy_group_mapping = self._build_group_mapping('obs_order', self.obs_dim_cfg)
        self.critic_group_mapping = self._build_group_mapping('state_order', self.state_dim_cfg)

        # Noise state for timing control
        self.step_count = 0
        self.episode_count = 0
        self.last_noise_step = -1
        self.current_noise = {}

        print(f"✓ Observation Noise Wrapper initialized")
        print(f"  - Policy groups: {list(self.policy_group_mapping.keys())}")
        print(f"  - Critic groups: {list(self.critic_group_mapping.keys())}")
        print(f"  - Configured noise groups: {list(self.config.noise_groups.keys())}")

    def _load_observation_configs(self) -> tuple:
        """Load observation dimension configurations from Isaac Lab's native dictionaries."""
        try:
            # Import Isaac Lab's native dimension configurations
            try:
                from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
            except ImportError:
                from omni.isaac.lab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG

            return OBS_DIM_CFG, STATE_DIM_CFG

        except ImportError:
            raise ValueError(
                "Could not import Isaac Lab's native dimension configurations (OBS_DIM_CFG, STATE_DIM_CFG). "
                "Please ensure Isaac Lab is properly installed and factory tasks are available. "
                "The observation noise wrapper requires these configurations as the single source of truth "
                "for observation dimensions."
            )

    def _build_group_mapping(self, order_attr: str, dim_cfg: Dict[str, int]) -> Dict[str, tuple]:
        """Build mapping from observation groups to tensor indices."""
        mapping = {}

        if not hasattr(self.unwrapped.cfg, order_attr):
            print(f"Warning: Environment has no {order_attr}, cannot apply group-based noise")
            return mapping

        order = getattr(self.unwrapped.cfg, order_attr)
        current_idx = 0

        for group_name in order:
            if group_name in dim_cfg:
                group_dim = dim_cfg[group_name]
                mapping[group_name] = (current_idx, current_idx + group_dim)
                current_idx += group_dim
            else:
                print(f"Warning: Unknown observation group '{group_name}' in {order_attr}")

        return mapping

    def _should_update_noise(self) -> bool:
        """Determine if noise should be updated based on timing configuration."""
        if not self.config.enabled:
            return False

        # Check if we need to update noise based on different timing strategies
        for group_config in self.config.noise_groups.values():
            if not group_config.enabled:
                continue

            if group_config.timing == "step":
                return True  # Update every step
            elif group_config.timing == "episode":
                return self.step_count == 0  # Update at episode start
            elif group_config.timing == "policy_update":
                # Update every N steps (policy update interval)
                return (self.step_count % self.config.policy_update_interval) == 0

        return False

    def _generate_group_noise(self, group_name: str, group_shape: tuple) -> torch.Tensor:
        """Generate noise for a specific observation group."""
        if group_name not in self.config.noise_groups:
            return torch.zeros(group_shape, device=self.device)

        group_config = self.config.noise_groups[group_name]
        if not group_config.enabled:
            return torch.zeros(group_shape, device=self.device)

        # Generate noise based on type
        if group_config.noise_type == "gaussian":
            noise = torch.randn(group_shape, device=self.device)

            # Apply std and mean
            if isinstance(group_config.std, (list, tuple)):
                std_tensor = torch.tensor(group_config.std, device=self.device)
            else:
                std_tensor = torch.full((group_shape[-1],), group_config.std, device=self.device)

            if isinstance(group_config.mean, (list, tuple)):
                mean_tensor = torch.tensor(group_config.mean, device=self.device)
            else:
                mean_tensor = torch.full((group_shape[-1],), group_config.mean, device=self.device)

            noise = noise * std_tensor + mean_tensor

        elif group_config.noise_type == "uniform":
            noise = torch.rand(group_shape, device=self.device) * 2 - 1  # [-1, 1]

            if isinstance(group_config.scale, (list, tuple)):
                scale_tensor = torch.tensor(group_config.scale, device=self.device)
            else:
                scale_tensor = torch.full((group_shape[-1],), group_config.scale, device=self.device)

            noise = noise * scale_tensor

        else:  # noise_type == "none"
            noise = torch.zeros(group_shape, device=self.device)

        # Apply global scale
        noise *= self.config.global_noise_scale

        return noise

    def _apply_noise_to_observations(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply group-based noise to observations."""
        if not self.config.enabled:
            return obs

        # Update noise if needed
        if self._should_update_noise():
            self._update_noise_cache()

        if isinstance(obs, dict):
            # Handle dictionary observations (pre-conversion)
            noisy_obs = {}
            for key, tensor in obs.items():
                if key == "policy":
                    noisy_obs[key] = self._apply_noise_to_tensor(tensor, self.policy_group_mapping)
                elif key == "critic":
                    # Apply noise to critic only if enabled
                    if self.config.apply_to_critic:
                        noisy_obs[key] = self._apply_noise_to_tensor(tensor, self.critic_group_mapping)
                    else:
                        noisy_obs[key] = tensor  # No noise for critic
                else:
                    noisy_obs[key] = tensor
            return noisy_obs
        else:
            # Handle single tensor (post-conversion) - assume policy observations
            return self._apply_noise_to_tensor(obs, self.policy_group_mapping)

    def _apply_noise_to_tensor(self, tensor: torch.Tensor, group_mapping: Dict[str, tuple]) -> torch.Tensor:
        """Apply cached noise to a tensor using group mapping."""
        if not group_mapping:
            return tensor

        noisy_tensor = tensor.clone()

        for group_name, (start_idx, end_idx) in group_mapping.items():
            if start_idx < tensor.shape[-1] and group_name in self.current_noise:
                group_noise = self.current_noise[group_name]

                # Ensure noise matches tensor batch size
                if group_noise.shape[0] != tensor.shape[0]:
                    repeat_factor = tensor.shape[0] // group_noise.shape[0]
                    if tensor.shape[0] % group_noise.shape[0] != 0:
                        repeat_factor += 1
                    group_noise = group_noise.repeat(repeat_factor, 1)[:tensor.shape[0]]

                # Apply noise to the group's dimensions
                tensor_slice = noisy_tensor[..., start_idx:end_idx]
                noisy_slice = tensor_slice + group_noise[..., :tensor_slice.shape[-1]]

                # Apply clipping if configured
                group_config = self.config.noise_groups.get(group_name)
                if group_config and group_config.clip_range:
                    clip_min, clip_max = group_config.clip_range
                    noisy_slice = torch.clamp(noisy_slice, clip_min, clip_max)

                noisy_tensor[..., start_idx:end_idx] = noisy_slice

        return noisy_tensor

    def _update_noise_cache(self):
        """Update cached noise for all groups, ensuring shared groups get identical noise."""
        self.current_noise.clear()

        # Get all unique group names from both policy and critic
        all_groups = set(self.policy_group_mapping.keys()) | set(self.critic_group_mapping.keys())

        # Generate noise for each unique group based on its semantic dimensions
        for group_name in all_groups:
            # Determine group dimensions from configuration
            if group_name in self.obs_dim_cfg:
                group_dim = self.obs_dim_cfg[group_name]
            elif group_name in self.state_dim_cfg:
                group_dim = self.state_dim_cfg[group_name]
            else:
                # Fallback: use policy mapping if available, otherwise critic mapping
                if group_name in self.policy_group_mapping:
                    start_idx, end_idx = self.policy_group_mapping[group_name]
                    group_dim = end_idx - start_idx
                else:
                    start_idx, end_idx = self.critic_group_mapping[group_name]
                    group_dim = end_idx - start_idx

            # Generate single noise tensor for this group (shared between policy and critic)
            group_shape = (self.num_envs, group_dim)
            self.current_noise[group_name] = self._generate_group_noise(group_name, group_shape)

    def step(self, action):
        """Step environment and apply noise."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Apply noise to observations
        noisy_obs = self._apply_noise_to_observations(obs)

        self.step_count += 1

        return noisy_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and noise state."""
        obs, info = super().reset(**kwargs)

        # Reset noise state
        self.step_count = 0
        self.episode_count += 1
        self.current_noise.clear()

        # Apply noise to initial observations
        noisy_obs = self._apply_noise_to_observations(obs)

        return noisy_obs, info

    def get_noise_info(self) -> Dict[str, Any]:
        """
        Get information about current noise configuration.

        Returns:
            dict: Information about noise state including:
                - enabled: Whether noise is globally enabled
                - global_scale: Global noise scaling factor
                - step_count: Current step count
                - episode_count: Current episode count
                - configured_groups: Dict of configured noise groups with their settings
                - active_groups: List of currently active group names
        """
        info = {
            "enabled": self.config.enabled,
            "global_scale": self.config.global_noise_scale,
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "configured_groups": {},
            "active_groups": list(self.current_noise.keys())
        }

        for group_name, group_config in self.config.noise_groups.items():
            info["configured_groups"][group_name] = {
                "enabled": group_config.enabled,
                "noise_type": group_config.noise_type,
                "timing": group_config.timing,
                "std": group_config.std,
                "mean": group_config.mean
            }

        return info


# Configuration presets for common use cases

def create_position_noise_config(
    position_std: float = 0.001,  # 1mm noise
    orientation_std: float = 0.017,  # ~1 degree noise
    velocity_std: float = 0.01,
    timing: str = "step",
    apply_to_critic: bool = True
) -> ObservationNoiseConfig:
    """Create noise config focused on position/orientation domain randomization."""
    config = ObservationNoiseConfig(apply_to_critic=apply_to_critic)

    # Position groups
    for group_name in ["fingertip_pos", "held_pos", "fixed_pos"]:
        config.add_group_noise(NoiseGroupConfig(
            group_name=group_name,
            noise_type="gaussian",
            std=position_std,
            timing=timing
        ))

    # Orientation groups
    for group_name in ["fingertip_quat", "held_quat", "fixed_quat"]:
        config.add_group_noise(NoiseGroupConfig(
            group_name=group_name,
            noise_type="gaussian",
            std=orientation_std,
            timing=timing
        ))

    # Velocity groups
    for group_name in ["ee_linvel", "ee_angvel"]:
        config.add_group_noise(NoiseGroupConfig(
            group_name=group_name,
            noise_type="gaussian",
            std=velocity_std,
            timing=timing
        ))

    return config


def create_joint_noise_config(
    joint_std: float = 0.01,  # Joint position noise
    timing: str = "policy_update",
    apply_to_critic: bool = True
) -> ObservationNoiseConfig:
    """Create noise config for joint-space domain randomization."""
    config = ObservationNoiseConfig(apply_to_critic=apply_to_critic)

    config.add_group_noise(NoiseGroupConfig(
        group_name="joint_pos",
        noise_type="gaussian",
        std=joint_std,
        timing=timing
    ))

    return config


def create_minimal_noise_config() -> ObservationNoiseConfig:
    """Create minimal noise configuration for testing."""
    config = ObservationNoiseConfig()

    # Very small noise on fingertip position only
    config.add_group_noise(NoiseGroupConfig(
        group_name="fingertip_pos",
        noise_type="gaussian",
        std=0.0001,  # 0.1mm
        timing="step"
    ))

    return config


# Example usage
if __name__ == "__main__":
    """Example usage of observation noise wrapper."""

    print("Observation Noise Wrapper - Example")
    print("=" * 50)

    # Example configurations
    print("✓ Available noise configurations:")
    print("  - Position/Orientation noise (domain randomization)")
    print("  - Joint space noise")
    print("  - Minimal noise (testing)")
    print("  - Custom group-based configurations")

    print("\n✓ Key features:")
    print("  - Group-based noise (fingertip_pos, joint_pos, etc.)")
    print("  - Configurable timing (step, episode, policy_update)")
    print("  - Multiple noise types (gaussian, uniform)")
    print("  - Compatible with history wrapper")
    print("  - Respects observation group semantics")