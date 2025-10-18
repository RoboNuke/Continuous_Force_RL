"""
Isaac Lab Version Compatibility Handler

Handles imports across different Isaac Lab versions and provides fallbacks for testing.
"""

# Cache for consistent class identities across imports
_cached_imports = None


def get_isaac_lab_ctrl_imports():
    """Get Isaac Lab control config imports with version compatibility."""
    global _cached_imports

    # Return cached imports if available
    if _cached_imports is not None:
        return _cached_imports

    try:
        # Isaac Lab v2+ (isaaclab_tasks)
        from isaaclab.utils.configclass import configclass
        from isaaclab_tasks.direct.factory.factory_env_cfg import CtrlCfg
        print("[CONFIG]: Using Isaac Lab v2+ imports")
        _cached_imports = (configclass, CtrlCfg)
        return _cached_imports
    except ImportError:
        try:
            # Isaac Lab v1.4.1 (omni.isaac.lab_tasks)
            from omni.isaac.lab.utils.configclass import configclass
            from omni.isaac.lab_tasks.direct.factory.factory_env_cfg import CtrlCfg
            print("[CONFIG]: Using Isaac Lab v1.4.1 imports")
            _cached_imports = (configclass, CtrlCfg)
            return _cached_imports
        except ImportError:
            # Fallback for testing/development
            from dataclasses import dataclass

            # Mock CtrlCfg for testing with proper dataclass structure
            @dataclass
            class MockCtrlCfg:
                ema_factor: float = 0.2
                pos_action_bounds: list = None
                rot_action_bounds: list = None
                pos_action_threshold: list = None
                rot_action_threshold: list = None
                reset_joints: list = None
                reset_task_prop_gains: list = None
                reset_rot_deriv_scale: float = 10.0
                default_task_prop_gains: list = None
                default_dof_pos_tensor: list = None
                kp_null: float = 10.0
                kd_null: float = 6.3246

                def __post_init__(self):
                    if self.pos_action_bounds is None:
                        self.pos_action_bounds = [0.05, 0.05, 0.05]
                    if self.rot_action_bounds is None:
                        self.rot_action_bounds = [1.0, 1.0, 1.0]
                    if self.pos_action_threshold is None:
                        self.pos_action_threshold = [0.02, 0.02, 0.02]
                    if self.rot_action_threshold is None:
                        self.rot_action_threshold = [0.097, 0.097, 0.097]
                    if self.reset_joints is None:
                        self.reset_joints = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
                    if self.reset_task_prop_gains is None:
                        self.reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
                    if self.default_task_prop_gains is None:
                        self.default_task_prop_gains = [100, 100, 100, 30, 30, 30]
                    if self.default_dof_pos_tensor is None:
                        self.default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]

            print("[CONFIG]: Using fallback mock imports for testing")
            _cached_imports = (dataclass, MockCtrlCfg)
            return _cached_imports


def get_isaac_lab_factory_imports():
    """Get Isaac Lab factory environments config imports with version compatibility."""
    try:
        # Isaac Lab v2+ (isaaclab_tasks)
        from isaaclab.utils.configclass import configclass
        from isaaclab_tasks.direct.factory.factory_env_cfg import (
            FactoryTaskPegInsertCfg,
            FactoryTaskGearMeshCfg,
            FactoryTaskNutThreadCfg
        )
        print("[CONFIG]: Using Isaac Lab v2+ task imports")
        return (configclass, FactoryTaskPegInsertCfg, FactoryTaskGearMeshCfg, FactoryTaskNutThreadCfg)
    except ImportError:
        # Isaac Lab v1.4.1 (omni.isaac.lab_tasks)
        from omni.isaac.lab.utils.configclass import configclass
        from omni.isaac.lab_tasks.direct.factory.factory_env_cfg import (
            FactoryTaskPegInsertCfg,
            FactoryTaskGearMeshCfg,
            FactoryTaskNutThreadCfg
        )
        print("[CONFIG]: Using Isaac Lab v1.4.1 task imports")
        return (configclass, FactoryTaskPegInsertCfg, FactoryTaskGearMeshCfg, FactoryTaskNutThreadCfg)


def get_isaac_lab_task_imports():
    try:
        # Isaac Lab v2+ (isaaclab_tasks)
        from isaaclab.utils.configclass import configclass
        from isaaclab_tasks.direct.factory.factory_tasks_cfg import (
            PegInsert,
            GearMesh,
            NutThread
        )
        print("[CONFIG]: Using Isaac Lab v2+ task imports")
        return (
            configclass, 
            PegInsert,
            GearMesh,
            NutThread
        )
    except ImportError:
        # Isaac Lab v1.4.1 (omni.isaac.lab_tasks)
        from omni.isaac.lab.utils.configclass import configclass
        from omni.isaac.lab_tasks.direct.factory.factory_env_cfg import (
            PegInsert,
            GearMesh,
            NutThread
        )
        print("[CONFIG]: Using Isaac Lab v1.4.1 task imports")
        return (
            configclass, 
            PegInsert,
            GearMesh,
            NutThread
        )
        