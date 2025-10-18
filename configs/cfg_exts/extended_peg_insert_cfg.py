"""
Extended Peg Insert Task Configuration

This module defines ExtendedFactoryTaskPegInsertCfg which extends Isaac Lab's
FactoryTaskPegInsertCfg with our custom parameters.
"""

from .version_compat import get_isaac_lab_task_imports

# Get Isaac Lab imports with version compatibility
configclass, PegInsert, _, _ = get_isaac_lab_task_imports()
from configs.cfg_exts.ctrl_cfg import ExtendedCtrlCfg

try:
    from isaaclab.sensors import ContactSensorCfg
except:
    from omni.isaac.lab.sensors import ContactSensorCfg


@configclass
class ExtendedFactoryTaskPegInsertCfg(PegInsert):
    """
    Extended peg insert task configuration.

    Inherits from Isaac Lab's FactoryTaskPegInsertCfg and adds our custom parameters.
    """
    use_ft_sensor: bool = False
    obs_type: str = ''
    ctrl_type: str = ''
    agent_type: str = ''
    ctrl: ExtendedCtrlCfg = None

    held_fixed_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/HeldAsset",
        update_period=0.0,
        history_length=0,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/FixedAsset"],
    )

    def __post_init__(self):
        self.ctrl = ExtendedCtrlCfg()

    def apply_primary_cfg(self, primary_cfg) -> None:
        """Apply primary configuration values to this task config."""
        self.decimation = primary_cfg.decimation
        if hasattr(self, 'scene') and self.scene is not None:
            if isinstance(self.scene, dict):
                self.scene['num_envs'] = primary_cfg.total_num_envs
            else:
                self.scene.num_envs = primary_cfg.total_num_envs
        self._primary_cfg = primary_cfg