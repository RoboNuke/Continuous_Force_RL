"""
Extended Peg Insert Task Configuration

This module defines ExtendedFactoryTaskPegInsertCfg which extends Isaac Lab's
FactoryTaskPegInsertCfg with our custom parameters.
"""

from .version_compat import get_isaac_lab_task_imports

# Get Isaac Lab imports with version compatibility
configclass, FactoryTaskPegInsertCfg, _, _ = get_isaac_lab_task_imports()
from configs.cfg_exts.ctrl_cfg import ExtendedCtrlCfg

@configclass
class ExtendedFactoryTaskPegInsertCfg(FactoryTaskPegInsertCfg):
    """
    Extended peg insert task configuration.

    Inherits from Isaac Lab's FactoryTaskPegInsertCfg and adds our custom parameters.
    """
    use_ft_sensor: bool = False
    obs_type: str = ''
    ctrl_type: str = ''
    agent_type: str = ''
    ctrl: ExtendedCtrlCfg = None

    def __post_init__(self):
        self.ctrl = ExtendedCtrlCfg()

    def apply_primary_cfg(self, primary_cfg) -> None:
        """Apply primary configuration values to this task config."""
        self.decimation = primary_cfg.decimation
        if hasattr(self, 'scene') and self.scene is not None:
            self.scene.num_envs = primary_cfg.total_num_envs
        self._primary_cfg = primary_cfg