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
    WandbLoggingConfig, ActionLoggingConfig
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

    # Simple wrapper configurations (flat)
    fragile_objects_enabled: bool = True
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

        return {
            'enabled': self.fragile_objects_enabled,
            'break_force': self._primary_cfg.break_forces,
            'num_agents': self._primary_cfg.total_agents
        }

    def get_force_torque_sensor_config(self) -> Dict[str, Any]:
        """Get force-torque sensor wrapper configuration."""
        return {
            'enabled': self.force_torque_sensor.enabled,
            'use_tanh_scaling': self.force_torque_sensor.use_tanh_scaling,
            'tanh_scale': self.force_torque_sensor.tanh_scale
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary structure for launch_utils."""
        return {
            'fragile_objects': {
                'enabled': self.fragile_objects_enabled,
                # num_agents and break_force computed from primary config
            },
            'efficient_reset': {
                'enabled': self.efficient_reset_enabled
            },
            'force_torque_sensor': {
                'enabled': self.force_torque_sensor.enabled,
                'use_tanh_scaling': self.force_torque_sensor.use_tanh_scaling,
                'tanh_scale': self.force_torque_sensor.tanh_scale
            },
            'observation_manager': {
                'enabled': self.observation_manager_enabled,
                'merge_strategy': self.observation_manager_merge_strategy
            },
            'observation_noise': {
                'enabled': self.observation_noise.enabled,
                'global_scale': self.observation_noise.global_scale,
                'apply_to_critic': self.observation_noise.apply_to_critic,
                'seed': self.observation_noise.seed
            },
            'hybrid_control': {
                'enabled': self.hybrid_control.enabled,
                'reward_type': self.hybrid_control.reward_type
            },
            'factory_metrics': {
                'enabled': self.factory_metrics_enabled
            },
            'wandb_logging': {
                'enabled': self.wandb_logging.enabled,
                'wandb_project': self.wandb_logging.wandb_project,
                'wandb_entity': self.wandb_logging.wandb_entity,
                'wandb_name': self.wandb_logging.wandb_name,
                'wandb_group': self.wandb_logging.wandb_group,
                'wandb_tags': self.wandb_logging.wandb_tags
            },
            'action_logging': {
                'enabled': self.action_logging.enabled,
                'track_selection': self.action_logging.track_selection,
                'track_pos': self.action_logging.track_pos,
                'track_rot': self.action_logging.track_rot,
                'track_force': self.action_logging.track_force,
                'track_torque': self.action_logging.track_torque,
                'force_size': self.action_logging.force_size,
                'logging_frequency': self.action_logging.logging_frequency
            }
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

        return f"ExtendedWrapperConfig(enabled: {', '.join(enabled_wrappers) if enabled_wrappers else 'none'})"