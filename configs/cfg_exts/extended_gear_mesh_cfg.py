"""
Extended Gear Mesh Task Configuration

This module defines ExtendedFactoryTaskGearMeshCfg which extends Isaac Lab's
FactoryTaskGearMeshCfg with our custom parameters.
"""

from .version_compat import get_isaac_lab_task_imports

# Get Isaac Lab imports with version compatibility
configclass, _, FactoryTaskGearMeshCfg, _ = get_isaac_lab_task_imports()


@configclass
class ExtendedFactoryTaskGearMeshCfg(FactoryTaskGearMeshCfg):
    """
    Extended gear mesh task configuration.

    Inherits from Isaac Lab's FactoryTaskGearMeshCfg and adds our custom parameters.
    """
    use_ft_sensor: bool = False
    obs_type: str = ''
    ctrl_type: str = ''
    agent_type: str = ''

    def apply_primary_cfg(self, primary_cfg) -> None:
        """Apply primary configuration values to this task config."""
        self.decimation = primary_cfg.decimation
        if hasattr(self, 'scene') and self.scene is not None:
            self.scene.num_envs = primary_cfg.total_num_envs
        self._primary_cfg = primary_cfg