"""
Extended Gear Mesh Task Configuration

This module defines ExtendedFactoryTaskGearMeshCfg which extends Isaac Lab's
FactoryTaskGearMeshCfg with our custom parameters and computed properties.
"""

from typing import Optional
from .version_compat import get_isaac_lab_task_imports
from .extended_factory_env_cfg import ExtendedFactoryEnvCfg

# Get Isaac Lab imports with version compatibility
configclass, _, FactoryTaskGearMeshCfg, _ = get_isaac_lab_task_imports()


@configclass
class ExtendedFactoryTaskGearMeshCfg(FactoryTaskGearMeshCfg):
    """
    Extended gear mesh task configuration.

    Inherits from Isaac Lab's FactoryTaskGearMeshCfg and adds our custom
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
        Validate that the gear mesh task configuration is complete and consistent.

        Raises:
            ValueError: If configuration is invalid or incomplete
        """
        # Use the extended factory env validation
        ExtendedFactoryEnvCfg.validate_configuration(self)

        # Gear mesh specific validation
        if not hasattr(self, 'task_name') or self.task_name != "gear_mesh":
            raise ValueError(f"task_name must be 'gear_mesh', got {getattr(self, 'task_name', 'None')}")

        # Validate episode length is reasonable for gear mesh
        if self.episode_length_s < 10.0 or self.episode_length_s > 120.0:
            raise ValueError(f"episode_length_s for gear mesh should be 10-120 seconds, got {self.episode_length_s}")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ExtendedFactoryTaskGearMeshCfg(task_name={self.task_name}, episode_length_s={self.episode_length_s}, decimation={self.decimation})"