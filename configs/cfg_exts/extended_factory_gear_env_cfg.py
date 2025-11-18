"""
Extended Factory Gear Mesh Environment Configuration

This module defines ExtendedFactoryGearEnvCfg which extends Isaac Lab's FactoryEnvCfg
for the gear mesh task with our custom parameters.
"""

from .version_compat import get_isaac_lab_factory_imports

# Get Isaac Lab imports with version compatibility
configclass, _, GearFactoryEnv, _ = get_isaac_lab_factory_imports()
from configs.cfg_exts.extended_gear_mesh_cfg import ExtendedFactoryTaskGearMeshCfg
from configs.cfg_exts.extended_factory_peg_env_cfg import ExtendedObsRandCfg
from configs.cfg_exts.ctrl_cfg import ExtendedCtrlCfg

@configclass
class ExtendedFactoryGearEnvCfg(GearFactoryEnv):
    """
    Extended factory gear mesh environment configuration.

    Inherits from Isaac Lab's FactoryEnvCfg and adds our custom parameters.
    """
    task: ExtendedFactoryTaskGearMeshCfg = None

    ctrl: ExtendedCtrlCfg = None

    obs_rand: ExtendedObsRandCfg = ExtendedObsRandCfg()
    """Observation randomization configuration with force-torque noise support"""

    # Additional data fields can be added here if needed
    filter_collisions: bool = True
    """Enable collision filtering"""

    component_attr_map: dict = None
    """Component attribute mapping"""


    def apply_primary_cfg(self, primary_cfg) -> None:
        """
        Apply primary configuration values to this environment config.

        Args:
            primary_cfg: PrimaryConfig instance containing shared parameters
        """
        # Apply primary config values that belong to environment
        self.decimation = primary_cfg.decimation

        # Update scene configuration for multi-agent setup
        if hasattr(self, 'scene') and self.scene is not None:
            self.scene.num_envs = primary_cfg.total_num_envs
            # Isaac Lab scene configs typically have replicate_physics
            if hasattr(self.scene, 'replicate_physics'):
                self.scene.replicate_physics = True

        # Store reference to primary config for computed properties
        self._primary_cfg = primary_cfg
