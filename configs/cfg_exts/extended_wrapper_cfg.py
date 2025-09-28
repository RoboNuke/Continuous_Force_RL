"""
Extended Wrapper Configuration

This module defines ExtendedWrapperConfig which contains configuration
for all environment wrappers. This is our own organizational structure
for wrapper configuration.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .version_compat import get_isaac_lab_ctrl_imports
from .wrapper_sub_configs import (
    ForceTorqueSensorConfig, HybridControlConfig, ObservationNoiseConfig,
    WandbLoggingConfig, ActionLoggingConfig, ForceRewardConfig, FragileObjectConfig
)

# Get configclass decorator with version compatibility
configclass, _ = get_isaac_lab_ctrl_imports()


@configclass
@dataclass
class ExtendedWrapperConfig:
    """
    Extended wrapper configuration for environment wrappers.

    Contains enable/disable flags and parameters for all wrapper types
    used in the factory environment.
    """

    # Nested wrapper configurations
    force_torque_sensor: ForceTorqueSensorConfig = field(default_factory=ForceTorqueSensorConfig)
    """Force-torque sensor wrapper configuration"""

    hybrid_control: HybridControlConfig = field(default_factory=HybridControlConfig)
    """Hybrid control wrapper configuration"""

    observation_noise: ObservationNoiseConfig = field(default_factory=ObservationNoiseConfig)
    """Observation noise wrapper configuration"""

    wandb_logging: WandbLoggingConfig = field(default_factory=WandbLoggingConfig)
    """Wandb logging wrapper configuration"""

    action_logging: ActionLoggingConfig = field(default_factory=ActionLoggingConfig)
    """Action logging wrapper configuration"""

    force_reward: ForceRewardConfig = field(default_factory=ForceRewardConfig)
    """Force reward wrapper configuration"""

    # Simple wrapper configurations (flat)
    fragile_objects: FragileObjectConfig = field(default_factory=FragileObjectConfig)
    """Enable fragile objects wrapper"""

    efficient_reset_enabled: bool = True
    """Enable efficient environment resetting"""

    observation_manager_enabled: bool = True
    """Enable observation format conversion wrapper"""

    observation_manager_merge_strategy: str = "concatenate"
    """Strategy for merging observations ('concatenate', etc.)"""

    factory_metrics_enabled: bool = True
    """Enable factory-specific metrics tracking"""

    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_wrapper_params()
        self._setup_defaults()

    def _validate_wrapper_params(self):
        """Validate wrapper configuration parameters."""
        # Validate string parameters
        valid_merge_strategies = ["concatenate"]
        if self.observation_manager_merge_strategy not in valid_merge_strategies:
            raise ValueError(f"observation_manager_merge_strategy must be one of {valid_merge_strategies}, got {self.observation_manager_merge_strategy}")

    def _setup_defaults(self):
        """Set up default values for optional parameters."""
        # No additional setup needed - nested configs handle their own defaults
        pass

    def apply_primary_cfg(self, primary_cfg) -> None:
        """
        Apply primary configuration values to wrapper config.

        Args:
            primary_cfg: PrimaryConfig instance containing shared parameters
        """
        # Store reference to primary config
        self._primary_cfg = primary_cfg

        # Apply primary config values that affect wrappers
        # (Most wrapper configs are independent, but some may use primary values)

    def get_fragile_objects_config(self) -> Dict[str, Any]:
        """Get fragile objects wrapper configuration."""
        if not hasattr(self, '_primary_cfg'):
            raise ValueError("Primary config not applied. Call apply_primary_cfg() first.")

        # The fragile object wrapper expects:
        # - break_force list length to match the TOTAL number of agents
        # - num_agents to be the TOTAL number of agents (for environment division)
        # Expand break forces from conditions to per-agent assignment
        expanded_break_forces = []
        for break_force in self._primary_cfg.break_forces:
            # Add this break force for each agent in this break force group
            for _ in range(self._primary_cfg.agents_per_break_force):
                expanded_break_forces.append(break_force)

        return {
            'enabled': self.fragile_objects_enabled,
            'break_force': expanded_break_forces,  # [10, 10, 20, 20] - one per agent
            'num_agents': self._primary_cfg.total_agents  # 4 - total number of agents
        }

    def get_force_torque_sensor_config(self) -> Dict[str, Any]:
        """Get force-torque sensor wrapper configuration."""
        return {
            'enabled': self.force_torque_sensor.enabled,
            'use_tanh_scaling': self.force_torque_sensor.use_tanh_scaling,
            'tanh_scale': self.force_torque_sensor.tanh_scale,
            'add_force_obs': self.force_torque_sensor.add_force_obs
        }

    def get_observation_noise_config(self) -> Dict[str, Any]:
        """Get observation noise wrapper configuration."""
        return {
            'enabled': self.observation_noise.enabled,
            'global_scale': self.observation_noise.global_scale,
            'apply_to_critic': self.observation_noise.apply_to_critic,
            'seed': self.observation_noise.seed
        }

    def get_hybrid_control_config(self) -> Dict[str, Any]:
        """Get hybrid control wrapper configuration."""
        if not hasattr(self, '_primary_cfg'):
            raise ValueError("Primary config not applied. Call apply_primary_cfg() first.")

        return {
            'enabled': self.hybrid_control.enabled,
            'ctrl_torque': self._primary_cfg.ctrl_torque,
            'reward_type': self.hybrid_control.reward_type
        }

    def get_wandb_config(self) -> Dict[str, Any]:
        """Get Wandb logging wrapper configuration."""
        return {
            'enabled': self.wandb_logging.enabled,
            'wandb_project': self.wandb_logging.wandb_project,
            'wandb_entity': self.wandb_logging.wandb_entity,
            'wandb_name': self.wandb_logging.wandb_name,
            'wandb_group': self.wandb_logging.wandb_group,
            'wandb_tags': self.wandb_logging.wandb_tags
        }

    def get_action_logging_config(self) -> Dict[str, Any]:
        """Get action logging wrapper configuration."""
        return {
            'enabled': self.action_logging.enabled,
            'track_selection': self.action_logging.track_selection,
            'track_pos': self.action_logging.track_pos,
            'track_rot': self.action_logging.track_rot,
            'track_force': self.action_logging.track_force,
            'track_torque': self.action_logging.track_torque,
            'force_size': self.action_logging.force_size,
            'logging_frequency': self.action_logging.logging_frequency
        }

    def get_factory_metrics_config(self) -> Dict[str, Any]:
        """Get factory metrics wrapper configuration."""
        if not hasattr(self, '_primary_cfg'):
            raise ValueError("Primary config not applied. Call apply_primary_cfg() first.")

        return {
            'enabled': self.factory_metrics_enabled,
            'num_agents': self._primary_cfg.total_agents
        }

    def get_force_reward_config(self) -> Dict[str, Any]:
        """Get force reward wrapper configuration."""
        return {
            'enabled': self.force_reward.enabled,
            'contact_force_threshold': self.force_reward.contact_force_threshold,
            'contact_window_size': self.force_reward.contact_window_size,
            'enable_force_magnitude_reward': self.force_reward.enable_force_magnitude_reward,
            'force_magnitude_reward_weight': self.force_reward.force_magnitude_reward_weight,
            'force_magnitude_base_force': self.force_reward.force_magnitude_base_force,
            'force_magnitude_keep_sign': self.force_reward.force_magnitude_keep_sign,
            'enable_alignment_award': self.force_reward.enable_alignment_award,
            'alignment_award_reward_weight': self.force_reward.alignment_award_reward_weight,
            'alignment_goal_orientation': self.force_reward.alignment_goal_orientation,
            'enable_force_action_error': self.force_reward.enable_force_action_error,
            'force_action_error_reward_weight': self.force_reward.force_action_error_reward_weight,
            'enable_contact_consistency': self.force_reward.enable_contact_consistency,
            'contact_consistency_reward_weight': self.force_reward.contact_consistency_reward_weight,
            'contact_consistency_beta': self.force_reward.contact_consistency_beta,
            'contact_consistency_use_ema': self.force_reward.contact_consistency_use_ema,
            'contact_consistency_ema_alpha': self.force_reward.contact_consistency_ema_alpha,
            'enable_oscillation_penalty': self.force_reward.enable_oscillation_penalty,
            'oscillation_penalty_reward_weight': self.force_reward.oscillation_penalty_reward_weight,
            'oscillation_penalty_window_size': self.force_reward.oscillation_penalty_window_size,
            'enable_contact_transition_reward': self.force_reward.enable_contact_transition_reward,
            'contact_transition_reward_weight': self.force_reward.contact_transition_reward_weight,
            'enable_efficiency': self.force_reward.enable_efficiency,
            'efficiency_reward_weight': self.force_reward.efficiency_reward_weight,
            'enable_force_ratio': self.force_reward.enable_force_ratio,
            'force_ratio_reward_weight': self.force_reward.force_ratio_reward_weight
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary structure for launch_utils."""
        # Only call methods that require primary_cfg if it has been applied
        if hasattr(self, '_primary_cfg'):
            return {
                'fragile_objects': self.get_fragile_objects_config(),
                'efficient_reset': {
                    'enabled': self.efficient_reset_enabled
                },
                'force_torque_sensor': self.get_force_torque_sensor_config(),
                'observation_manager': {
                    'enabled': self.observation_manager_enabled,
                    'merge_strategy': self.observation_manager_merge_strategy
                },
                'observation_noise': self.get_observation_noise_config(),
                'hybrid_control': self.get_hybrid_control_config(),
                'factory_metrics': self.get_factory_metrics_config(),
                'wandb_logging': self.get_wandb_config(),
                'action_logging': self.get_action_logging_config(),
                'force_reward': self.get_force_reward_config()
            }
        else:
            # Fallback for when primary config not applied yet
            return {
                'fragile_objects': {'enabled': self.fragile_objects_enabled},
                'efficient_reset': {'enabled': self.efficient_reset_enabled},
                'force_torque_sensor': self.get_force_torque_sensor_config(),
                'observation_manager': {
                    'enabled': self.observation_manager_enabled,
                    'merge_strategy': self.observation_manager_merge_strategy
                },
                'observation_noise': self.get_observation_noise_config(),
                'hybrid_control': {'enabled': self.hybrid_control.enabled},
                'factory_metrics': {'enabled': self.factory_metrics_enabled},
                'wandb_logging': self.get_wandb_config(),
                'action_logging': self.get_action_logging_config(),
                'force_reward': self.get_force_reward_config()
            }


    def __repr__(self) -> str:
        """String representation for debugging."""
        enabled_wrappers = []
        if self.fragile_objects_enabled:
            enabled_wrappers.append("fragile_objects")
        if self.force_torque_sensor.enabled:
            enabled_wrappers.append("force_torque")
        if self.hybrid_control.enabled:
            enabled_wrappers.append("hybrid_control")
        if self.observation_noise.enabled:
            enabled_wrappers.append("obs_noise")
        if self.action_logging.enabled:
            enabled_wrappers.append("action_logging")
        if self.force_reward.enabled:
            enabled_wrappers.append("force_reward")

        return f"ExtendedWrapperConfig(enabled: {', '.join(enabled_wrappers) if enabled_wrappers else 'none'})"