"""
Extended Nut Thread Task Configuration

This module defines ExtendedFactoryTaskNutThreadCfg which extends Isaac Lab's
FactoryTaskNutThreadCfg with our custom parameters.
"""

from .version_compat import get_isaac_lab_task_imports

# Get Isaac Lab imports with version compatibility
configclass, _, _, FactoryTaskNutThreadCfg = get_isaac_lab_task_imports()


@configclass
class ExtendedFactoryTaskNutThreadCfg(FactoryTaskNutThreadCfg):
    """
    Extended nut thread task configuration.

    Inherits from Isaac Lab's FactoryTaskNutThreadCfg and adds our custom parameters.
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