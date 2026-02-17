"""
Extended Factory Nut Thread Environment Configuration

This module defines ExtendedFactoryNutEnvCfg which extends Isaac Lab's FactoryEnvCfg
for the nut thread task with our custom parameters.
"""

from .version_compat import get_isaac_lab_factory_imports

# Get Isaac Lab imports with version compatibility
configclass, _, _, NutFactoryEnv = get_isaac_lab_factory_imports()
from configs.cfg_exts.extended_nut_thread_cfg import ExtendedFactoryTaskNutThreadCfg
from configs.cfg_exts.extended_factory_peg_env_cfg import ExtendedObsRandCfg
from configs.cfg_exts.ctrl_cfg import ExtendedCtrlCfg

@configclass
class ExtendedFactoryNutEnvCfg(NutFactoryEnv):
    """
    Extended factory nut thread environment configuration.

    Inherits from Isaac Lab's FactoryEnvCfg and adds our custom parameters.
    """
    task: ExtendedFactoryTaskNutThreadCfg = None

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
            if isinstance(self.scene, dict):
                self.scene['num_envs'] = primary_cfg.total_num_envs
            else:
                self.scene.num_envs = primary_cfg.total_num_envs
                if hasattr(self.scene, 'replicate_physics'):
                    self.scene.replicate_physics = True

        # Store reference to primary config for computed properties
        self._primary_cfg = primary_cfg
