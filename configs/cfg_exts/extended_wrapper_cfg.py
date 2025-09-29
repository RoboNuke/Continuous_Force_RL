"""
Extended Wrapper Configuration

This module defines ExtendedWrapperConfig which contains configuration
for all environment wrappers. This is our own organizational structure
for wrapper configuration.
"""

from dataclasses import dataclass, field

from .version_compat import get_isaac_lab_ctrl_imports
from .wrapper_sub_configs import (
    ForceTorqueSensorConfig, HybridControlConfig, ObservationNoiseConfig,
    WandbLoggingConfig, ActionLoggingConfig, ForceRewardConfig, FragileObjectConfig, ObsManagerConfig
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

    observation_manager: ObsManagerConfig = field(default_factory=ObsManagerConfig)

    efficient_reset_enabled: bool = True
    """Enable efficient environment resetting"""

    factory_metrics_enabled: bool = True
    """Enable factory-specific metrics tracking"""

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