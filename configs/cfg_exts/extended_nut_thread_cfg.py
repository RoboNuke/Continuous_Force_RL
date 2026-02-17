"""
Extended Nut Thread Task Configuration

This module defines ExtendedFactoryTaskNutThreadCfg which extends Isaac Lab's
FactoryTaskNutThreadCfg with our custom parameters.
"""

from .version_compat import get_isaac_lab_task_imports, get_contact_sensor_cfg

# Get Isaac Lab imports with version compatibility
configclass, _, _, NutThread = get_isaac_lab_task_imports()
from configs.cfg_exts.ctrl_cfg import ExtendedCtrlCfg

ContactSensorCfg = get_contact_sensor_cfg()


@configclass
class ExtendedFactoryTaskNutThreadCfg(NutThread):
    """
    Extended nut thread task configuration.

    Inherits from Isaac Lab's FactoryTaskNutThreadCfg and adds our custom parameters.
    """
    use_ft_sensor: bool = False
    obs_type: str = ''
    ctrl_type: str = ''
    agent_type: str = ''
    ctrl: ExtendedCtrlCfg = None

    held_fixed_contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/factory_nut_loose",
        update_period=0.0,
        history_length=0,  # Track last 10 timesteps of contact history
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*/FixedAsset/factory_bolt_loose"],
        track_air_time=True,  # Enable tracking of time in contact vs air time
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
