"""
Isaac Lab Version Compatibility Handler

Handles imports across different Isaac Lab versions and provides fallbacks for testing.

When running outside Isaac Sim (e.g., real robot evaluation), call set_no_sim_mode(True)
BEFORE importing any modules that depend on Isaac Lab. This provides mock base classes
so the config infrastructure can be used without the simulator.
"""

# Cache for consistent class identities across imports
_cached_imports = None

# When True, skip Isaac Sim imports and return mock classes instead
_no_sim_mode = False


def set_no_sim_mode(enabled=True):
    """Enable standalone mode to skip all Isaac Sim imports and use mock classes.

    Must be called BEFORE any modules that use get_isaac_lab_factory_imports()
    or get_isaac_lab_task_imports() are imported.
    """
    global _no_sim_mode
    _no_sim_mode = enabled
    if enabled:
        print("[CONFIG]: No-sim mode enabled — Isaac Sim imports will be skipped")


def _mock_configclass(cls):
    """Mock replacement for Isaac Lab's @configclass that handles mutable defaults.

    Isaac Lab's configclass auto-wraps mutable defaults (list, dict, set) with
    dataclasses.field(default_factory=...). Standard @dataclass does not, so this
    wrapper replicates that behavior.
    """
    import dataclasses
    from copy import deepcopy

    annotations = getattr(cls, '__annotations__', {})
    for name, _ in annotations.items():
        if hasattr(cls, name):
            val = getattr(cls, name)
            if isinstance(val, (list, dict, set)):
                # Replace mutable default with a field using default_factory
                default_val = val
                setattr(cls, name, dataclasses.field(default_factory=lambda v=default_val: deepcopy(v)))

    return dataclasses.dataclass(cls)


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
    if _no_sim_mode:
        @_mock_configclass
        class MockFactoryTaskPegInsertCfg:
            pass

        @_mock_configclass
        class MockFactoryTaskGearMeshCfg:
            pass

        @_mock_configclass
        class MockFactoryTaskNutThreadCfg:
            pass

        print("[CONFIG]: Using no-sim mock factory env imports")
        return (_mock_configclass, MockFactoryTaskPegInsertCfg, MockFactoryTaskGearMeshCfg, MockFactoryTaskNutThreadCfg)

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
    """Get Isaac Lab task config imports with version compatibility."""
    if _no_sim_mode:
        @_mock_configclass
        class MockPegInsert:
            pass

        @_mock_configclass
        class MockGearMesh:
            pass

        @_mock_configclass
        class MockNutThread:
            pass

        print("[CONFIG]: Using no-sim mock task imports")
        return (_mock_configclass, MockPegInsert, MockGearMesh, MockNutThread)

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


def get_contact_sensor_cfg():
    """Get ContactSensorCfg class with version compatibility."""
    if _no_sim_mode:
        @_mock_configclass
        class MockContactSensorCfg:
            prim_path: str = ""
            update_period: float = 0.0
            history_length: int = 0
            debug_vis: bool = False
            filter_prim_paths_expr: list = None
            track_air_time: bool = False

        return MockContactSensorCfg

    try:
        from isaaclab.sensors import ContactSensorCfg
        return ContactSensorCfg
    except ImportError:
        from omni.isaac.lab.sensors import ContactSensorCfg
        return ContactSensorCfg