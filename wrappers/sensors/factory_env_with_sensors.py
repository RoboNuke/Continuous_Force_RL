"""
Factory Environment with Contact Sensor Support

This module provides functionality to dynamically create factory environment classes
with contact sensor initialization properly integrated into the scene setup process.

The ContactSensor must be initialized during _setup_scene() before scene cloning occurs.
This module enables sensor injection without modifying the base IsaacLab factory environment code.
"""

try:
    from isaaclab.sensors import ContactSensor
except ImportError:
    try:
        from omni.isaac.lab.sensors import ContactSensor
    except ImportError:
        raise ImportError("Could not import ContactSensor from Isaac Lab. Please ensure Isaac Lab is installed.")


# Debug flag - set to True to enable detailed prim path verification
DEBUG_PRIM_PATHS = False


def _verify_prim_paths(env, sensor_cfg):
    """
    Verify that contact sensor prim paths exist in the scene.

    This function traverses the USD stage after scene setup to verify that:
    1. The expected HeldAsset and FixedAsset prims exist
    2. The child prims referenced by the contact sensor config exist
    3. The prims have the expected physics properties

    Args:
        env: The factory environment instance
        sensor_cfg: The ContactSensorCfg with prim_path and filter_prim_paths_expr
    """
    if not DEBUG_PRIM_PATHS:
        return

    try:
        from pxr import Usd, UsdPhysics
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[DEBUG PRIM] Warning: Could not get USD stage")
            return

        print("\n" + "="*80)
        print("[DEBUG PRIM] Contact Sensor Prim Path Verification")
        print("="*80)

        # Print sensor config
        prim_path_pattern = sensor_cfg.prim_path
        filter_patterns = sensor_cfg.filter_prim_paths_expr

        print(f"[DEBUG PRIM] Sensor prim_path pattern: {prim_path_pattern}")
        print(f"[DEBUG PRIM] Sensor filter patterns: {filter_patterns}")

        # Try to find matching prims for env_0
        test_prim_path = prim_path_pattern.replace("env_.*", "env_0")
        test_filter_path = filter_patterns[0].replace("env_.*", "env_0") if filter_patterns else None

        print(f"\n[DEBUG PRIM] Testing concrete paths for env_0:")
        print(f"  HeldAsset prim path: {test_prim_path}")

        held_prim = stage.GetPrimAtPath(test_prim_path)
        if held_prim.IsValid():
            print(f"  ✓ HeldAsset prim EXISTS")
            print(f"    Type: {held_prim.GetTypeName()}")
            has_rigid = held_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            has_collision = held_prim.HasAPI(UsdPhysics.CollisionAPI)
            print(f"    RigidBodyAPI: {has_rigid}, CollisionAPI: {has_collision}")

            # List children
            children = held_prim.GetChildren()
            if children:
                print(f"    Children ({len(children)}):")
                for child in children[:5]:  # Limit to first 5
                    print(f"      - {child.GetPath()} ({child.GetTypeName()})")
        else:
            print(f"  ✗ HeldAsset prim NOT FOUND at: {test_prim_path}")
            # Try to find what IS at HeldAsset
            parent_path = "/World/envs/env_0/HeldAsset"
            parent_prim = stage.GetPrimAtPath(parent_path)
            if parent_prim.IsValid():
                print(f"    HeldAsset parent exists. Children:")
                for child in parent_prim.GetChildren():
                    print(f"      - {child.GetPath()} ({child.GetTypeName()})")

        if test_filter_path:
            print(f"\n  FixedAsset prim path: {test_filter_path}")
            fixed_prim = stage.GetPrimAtPath(test_filter_path)
            if fixed_prim.IsValid():
                print(f"  ✓ FixedAsset prim EXISTS")
                print(f"    Type: {fixed_prim.GetTypeName()}")
                has_rigid = fixed_prim.HasAPI(UsdPhysics.RigidBodyAPI)
                has_collision = fixed_prim.HasAPI(UsdPhysics.CollisionAPI)
                print(f"    RigidBodyAPI: {has_rigid}, CollisionAPI: {has_collision}")
            else:
                print(f"  ✗ FixedAsset prim NOT FOUND at: {test_filter_path}")
                # Try to find what IS at FixedAsset
                parent_path = "/World/envs/env_0/FixedAsset"
                parent_prim = stage.GetPrimAtPath(parent_path)
                if parent_prim.IsValid():
                    print(f"    FixedAsset parent exists. Children:")
                    for child in parent_prim.GetChildren():
                        child_children = child.GetChildren()
                        print(f"      - {child.GetPath()} ({child.GetTypeName()})")
                        for cc in child_children[:3]:
                            print(f"        - {cc.GetPath()} ({cc.GetTypeName()})")

        print("="*80 + "\n")

    except Exception as e:
        print(f"[DEBUG PRIM] Error during prim verification: {e}")
        import traceback
        traceback.print_exc()


def create_sensor_enabled_factory_env(base_env_class):
    """
    Creates a factory environment subclass with contact sensor support.

    This function dynamically generates a subclass of the provided factory environment
    that properly initializes a ContactSensor during scene setup. The sensor is created
    before scene cloning to ensure proper initialization.

    Args:
        base_env_class: The base factory environment class to extend

    Returns:
        A new class that extends base_env_class with sensor initialization

    Usage:
        SensorEnabledEnv = create_sensor_enabled_factory_env(FactoryEnv)
        env = SensorEnabledEnv(cfg=env_cfg, render_mode=None)

    Note:
        The environment config (cfg.cfg_task) must have a 'held_fixed_contact_sensor'
        attribute containing a ContactSensorCfg for this to work properly.
    """

    class SensorEnabledFactoryEnv(base_env_class):
        """
        Factory environment with integrated contact sensor initialization.

        This class overrides _setup_scene to inject ContactSensor creation
        before delegating to the base class for standard scene setup.
        """

        def _setup_scene(self):
            """
            Initialize simulation scene with contact sensor support.

            Creates and registers the ContactSensor before calling the base
            class _setup_scene method. This ensures the sensor is properly
            initialized before scene cloning occurs.
            """
            # Validate that contact sensor config exists
            if not hasattr(self.cfg_task, 'held_fixed_contact_sensor'):
                raise ValueError(
                    "Environment config is missing 'held_fixed_contact_sensor'. "
                    "This is required when using sensor-enabled factory environment. "
                    "Add ContactSensorCfg to your task config."
                )

            # Create and register contact sensor BEFORE base class setup
            # The sensor registration must happen before clone_environments() is called
            self._held_fixed_contact_sensor = ContactSensor(
                self.cfg_task.held_fixed_contact_sensor
            )
            self.scene.sensors["held_fixed_contact_sensor"] = self._held_fixed_contact_sensor


            # Delegate to base class for standard factory environment setup
            # This will create articulations, clone environments, and complete scene setup
            super()._setup_scene()

            # Verify prim paths after scene setup
            _verify_prim_paths(self, self.cfg_task.held_fixed_contact_sensor)


    # Set a descriptive name for the dynamic class
    SensorEnabledFactoryEnv.__name__ = f"SensorEnabled{base_env_class.__name__}"
    SensorEnabledFactoryEnv.__qualname__ = f"SensorEnabled{base_env_class.__qualname__}"

    return SensorEnabledFactoryEnv
