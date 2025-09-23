"""
Extended Nut Thread Task Configuration

This module defines ExtendedFactoryTaskNutThreadCfg which extends Isaac Lab's
FactoryTaskNutThreadCfg with our custom parameters and computed properties.
"""

from typing import Optional
from .version_compat import get_isaac_lab_task_imports
from .extended_factory_env_cfg import ExtendedFactoryEnvCfg

# Get Isaac Lab imports with version compatibility
configclass, _, _, FactoryTaskNutThreadCfg = get_isaac_lab_task_imports()


@configclass
class ExtendedFactoryTaskNutThreadCfg(FactoryTaskNutThreadCfg):
    """
    Extended nut thread task configuration.

    Inherits from Isaac Lab's FactoryTaskNutThreadCfg and adds our custom
    parameters and computed properties. Uses our ExtendedFactoryEnvCfg features.
    """

    def __post_init__(self):
        """Post-initialization to set up extended configurations."""
        # Apply our extended factory env post_init logic
        ExtendedFactoryEnvCfg.__post_init__(self)

    def apply_primary_cfg(self, primary_cfg) -> None:
        """
        Apply primary configuration values to this task config.

        Args:
            primary_cfg: PrimaryConfig instance containing shared parameters
        """
        # Use the extended factory env apply_primary_cfg method
        ExtendedFactoryEnvCfg.apply_primary_cfg(self, primary_cfg)

    def validate_configuration(self) -> None:
        """
        Validate that the nut thread task configuration is complete and consistent.

        Raises:
            ValueError: If configuration is invalid or incomplete
        """
        # Use the extended factory env validation
        ExtendedFactoryEnvCfg.validate_configuration(self)

        # Nut thread specific validation
        if not hasattr(self, 'task_name') or self.task_name != "nut_thread":
            raise ValueError(f"task_name must be 'nut_thread', got {getattr(self, 'task_name', 'None')}")

        # Validate episode length is reasonable for nut thread
        if self.episode_length_s < 15.0 or self.episode_length_s > 180.0:
            raise ValueError(f"episode_length_s for nut thread should be 15-180 seconds, got {self.episode_length_s}")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ExtendedFactoryTaskNutThreadCfg(task_name={self.task_name}, episode_length_s={self.episode_length_s}, decimation={self.decimation})"