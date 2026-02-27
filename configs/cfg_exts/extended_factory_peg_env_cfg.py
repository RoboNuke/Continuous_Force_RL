"""
Extended Factory Environment Configuration

This module defines ExtendedFactoryEnvCfg which extends Isaac Lab's FactoryEnvCfg
with our custom parameters.
"""

from .version_compat import get_isaac_lab_factory_imports

# Get Isaac Lab imports with version compatibility
configclass, PegFactoryEnv, _, _, = get_isaac_lab_factory_imports()
from configs.cfg_exts.extended_peg_insert_cfg import ExtendedFactoryTaskPegInsertCfg
from configs.cfg_exts.ctrl_cfg import ExtendedCtrlCfg

@configclass
class ExtendedObsRandCfg:
    """Extended observation randomization config with force-torque noise."""

    fixed_asset_pos: list = [0.001, 0.001, 0.001]
    """Position noise for fixed asset (meters) - default 1mm per axis"""

    fixed_asset_yaw: float = 0.0523599
    """Yaw noise for fixed asset (radians) - default 3 degrees"""

    use_fixed_asset_yaw_noise: bool = False
    """Enable fixed asset yaw observation noise - automatically adds fingertip_yaw_rel_fixed to observations"""

    ee_pos: list = [0.00025, 0.00025, 0.00025]
    """Position noise for end-effector (meters) - default 0.25mm per axis"""

    ee_rpy: list = [0.0, 0.0, 0.0]
    """RPY noise for end-effector orientation (radians) - [roll, pitch, yaw] - default disabled"""

    force_torque: list = [0.25, 0.25, 0.25, 0.01, 0.01, 0.01]
    """Force-torque sensor noise [Fx, Fy, Fz, Tx, Ty, Tz] in N and Nm - applied per step"""

@configclass
class ExtendedFactoryPegEnvCfg(PegFactoryEnv):
    """
    Extended factory environment configuration.

    Inherits from Isaac Lab's FactoryEnvCfg and adds our custom parameters.
    """
    task: ExtendedFactoryTaskPegInsertCfg = None

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