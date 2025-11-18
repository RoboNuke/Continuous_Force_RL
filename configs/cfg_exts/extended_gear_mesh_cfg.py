"""
Extended Gear Mesh Task Configuration

This module defines ExtendedFactoryTaskGearMeshCfg which extends Isaac Lab's
FactoryTaskGearMeshCfg with our custom parameters.
"""

from .version_compat import get_isaac_lab_task_imports

# Get Isaac Lab imports with version compatibility
configclass, _, GearMesh, _ = get_isaac_lab_task_imports()
from configs.cfg_exts.ctrl_cfg import ExtendedCtrlCfg

try:
    from isaaclab.sensors import ContactSensorCfg
except:
    from omni.isaac.lab.sensors import ContactSensorCfg


@configclass
class ExtendedFactoryTaskGearMeshCfg(GearMesh):
    """
    Extended gear mesh task configuration.

    Inherits from Isaac Lab's FactoryTaskGearMeshCfg and adds our custom parameters.
    """
    use_ft_sensor: bool = False
    obs_type: str = ''
    ctrl_type: str = ''
    agent_type: str = ''
    ctrl: ExtendedCtrlCfg = None

    held_fixed_contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/factory_gear_medium",
        update_period=0.0,
        history_length=0,  # Track last 10 timesteps of contact history
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*/FixedAsset/factory_gear_large",
                                "/World/envs/env_.*/FixedAsset/factory_gear_small",
                                "/World/envs/env_.*/FixedAsset/factory_gear_base_loose"],
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
