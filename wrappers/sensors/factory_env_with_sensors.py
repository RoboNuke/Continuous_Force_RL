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
            """
            # DEBUG: Print what's actually in the scene
            print("\n" + "="*100)
            print("Scene articulations:", list(self.scene.articulations.keys()))
            print("Looking for contact sensor at:", self.cfg_task.held_fixed_contact_sensor.prim_path)

            if hasattr(self, '_held_asset'):
                print("\nHELD ASSET:")
                if hasattr(self._held_asset, 'prim_paths'):
                    print("  prim_paths:", self._held_asset.prim_paths)
                if hasattr(self._held_asset, 'cfg') and hasattr(self._held_asset.cfg, 'prim_path'):
                    print("  cfg.prim_path:", self._held_asset.cfg.prim_path)
                print("  body_names:", self._held_asset.body_names if hasattr(self._held_asset, 'body_names') else "N/A")
                print("  num_bodies:", self._held_asset.num_bodies if hasattr(self._held_asset, 'num_bodies') else "N/A")
                if hasattr(self._held_asset, 'cfg') and hasattr(self._held_asset.cfg, 'spawn'):
                    spawn_cfg = self._held_asset.cfg.spawn
                    print("  spawn config type:", type(spawn_cfg).__name__)
                    if hasattr(spawn_cfg, 'activate_contact_sensors'):
                        print("  activate_contact_sensors:", spawn_cfg.activate_contact_sensors)

            if hasattr(self, '_fixed_asset'):
                print("\nFIXED ASSET:")
                if hasattr(self._fixed_asset, 'prim_paths'):
                    print("  prim_paths:", self._fixed_asset.prim_paths)
                if hasattr(self._fixed_asset, 'cfg') and hasattr(self._fixed_asset.cfg, 'prim_path'):
                    print("  cfg.prim_path:", self._fixed_asset.cfg.prim_path)
                print("  body_names:", self._fixed_asset.body_names if hasattr(self._fixed_asset, 'body_names') else "N/A")
                print("  num_bodies:", self._fixed_asset.num_bodies if hasattr(self._fixed_asset, 'num_bodies') else "N/A")
                if hasattr(self._fixed_asset, 'cfg') and hasattr(self._fixed_asset.cfg, 'spawn'):
                    spawn_cfg = self._fixed_asset.cfg.spawn
                    print("  spawn config type:", type(spawn_cfg).__name__)
                    if hasattr(spawn_cfg, 'activate_contact_sensors'):
                        print("  activate_contact_sensors:", spawn_cfg.activate_contact_sensors)

            print("="*100 + "\n")

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


    # Set a descriptive name for the dynamic class
    SensorEnabledFactoryEnv.__name__ = f"SensorEnabled{base_env_class.__name__}"
    SensorEnabledFactoryEnv.__qualname__ = f"SensorEnabled{base_env_class.__qualname__}"

    return SensorEnabledFactoryEnv
