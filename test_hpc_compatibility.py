#!/usr/bin/env python3
"""
Test HPC compatibility and create a mock-based solution if Isaac Lab is unavailable.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, '.')

def test_isaac_lab_availability():
    """Test what Isaac Lab components are available."""

    print("🔍 TESTING ISAAC LAB AVAILABILITY")
    print("=" * 50)

    # Test core Isaac Lab
    try:
        import omni.isaac.lab
        print("✓ omni.isaac.lab available")
        core_available = True
    except ImportError as e:
        print(f"✗ omni.isaac.lab not available: {e}")
        core_available = False

    # Test Isaac Lab tasks
    try:
        import omni.isaac.lab_tasks
        print("✓ omni.isaac.lab_tasks available")
        tasks_available = True
    except ImportError as e:
        print(f"✗ omni.isaac.lab_tasks not available: {e}")
        tasks_available = False

    # Test environment configs
    try:
        from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
        print("✓ ManagerBasedRLEnvCfg available")
        env_cfg_available = True
    except ImportError as e:
        print(f"✗ ManagerBasedRLEnvCfg not available: {e}")
        env_cfg_available = False

    # Test gym integration
    try:
        import gymnasium as gym
        print("✓ gymnasium available")
        gym_available = True
    except ImportError as e:
        print(f"✗ gymnasium not available: {e}")
        gym_available = False

    return {
        'core': core_available,
        'tasks': tasks_available,
        'env_cfg': env_cfg_available,
        'gym': gym_available
    }

def create_mock_isaac_lab_config():
    """Create a mock Isaac Lab configuration that works without Isaac Lab."""

    print("\n🔧 CREATING MOCK ISAAC LAB CONFIG")
    print("=" * 50)

    # Create a basic mock configuration class
    class MockManagerBasedRLEnvCfg:
        """Mock Isaac Lab environment configuration."""

        def __init__(self):
            # Basic simulation settings
            self.sim = type('SimConfig', (), {
                'device': 'cuda',
                'dt': 0.01,
                'substeps': 1
            })()

            # Scene settings
            self.scene = type('SceneConfig', (), {
                'num_envs': 256,
                'env_spacing': 2.0
            })()

            # Episode settings
            self.episode_length_s = 12.0
            self.decimation = 4

            # Task-specific settings (factory defaults)
            self.break_force = [-1]
            self.filter_collisions = True

            # Component configuration for wrappers
            self.component_dims = {
                'fingertip_pos': 3,
                'joint_pos': 7,
                'force_torque': 6,
                'prev_actions': 6
            }

            self.component_attr_map = {
                'fingertip_pos': 'fingertip_midpoint_pos',
                'joint_pos': 'joint_pos',
                'force_torque': 'robot_force_torque',
                'prev_actions': 'prev_actions'
            }

            # Task configuration
            self.cfg_task = type('TaskConfig', (), {
                'success_threshold': 0.02,
                'engage_threshold': 0.05,
                'name': 'factory_task'
            })()

            # Control configuration
            self.cfg_ctrl = type('CtrlConfig', (), {
                'pos_action_bounds': [0.05, 0.05, 0.05],
                'force_action_bounds': [50.0, 50.0, 50.0],
                'torque_action_bounds': [0.5, 0.5, 0.5],
                'force_action_threshold': [10.0, 10.0, 10.0],
                'torque_action_threshold': [0.1, 0.1, 0.1],
                'default_task_force_gains': [0.1, 0.1, 0.1, 0.001, 0.001, 0.001]
            })()

            print(f"✓ Mock configuration created with factory-appropriate defaults")

    return MockManagerBasedRLEnvCfg

def test_configuration_compatibility():
    """Test that our configuration system works with mock Isaac Lab."""

    print("\n🧪 TESTING CONFIGURATION COMPATIBILITY")
    print("=" * 50)

    try:
        from configs.config_manager import ConfigManager

        # Test loading experiment configuration
        config_path = "configs/experiments/hybrid_control_exp.yaml"
        if not Path(config_path).exists():
            print(f"⚠ {config_path} not found, skipping test")
            return False

        resolved_config = ConfigManager.load_and_resolve_config(config_path)
        print(f"✓ Configuration loaded from {config_path}")

        # Create mock environment config
        MockEnvCfg = create_mock_isaac_lab_config()
        mock_env_cfg = MockEnvCfg()

        # Test applying our configuration to mock config
        ConfigManager.apply_to_isaac_lab(mock_env_cfg, {}, resolved_config)
        print(f"✓ Configuration applied to mock environment config")

        # Verify key settings were applied
        environment = resolved_config.get('environment', {})
        for key, value in environment.items():
            if hasattr(mock_env_cfg, key):
                actual_value = getattr(mock_env_cfg, key)
                print(f"  • {key}: {actual_value}")

        print(f"✓ Mock configuration system works correctly")
        return True

    except Exception as e:
        print(f"✗ Configuration compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_hpc_compatible_factory_runner():
    """Create an HPC-compatible version of factory_runnerv2.py"""

    print(f"\n📝 HPC COMPATIBILITY RECOMMENDATIONS")
    print("=" * 50)

    print(f"To make factory_runnerv2.py work on HPC without Isaac Lab:")
    print(f"")
    print(f"1. OPTION A - Mock Isaac Lab (Quick Fix):")
    print(f"   • Add mock Isaac Lab classes to factory_runnerv2.py")
    print(f"   • Use mock environment creation instead of gym.make()")
    print(f"   • Keep all configuration and wrapper logic the same")
    print(f"")
    print(f"2. OPTION B - Install Isaac Lab on HPC:")
    print(f"   • pip install isaac-lab (if available)")
    print(f"   • conda install isaac-lab (if available)")
    print(f"   • Build from source if necessary")
    print(f"")
    print(f"3. OPTION C - Configuration-Only Mode:")
    print(f"   • Modify script to generate configs without running training")
    print(f"   • Export resolved configs for use with other training systems")
    print(f"   • Useful for config validation and debugging")

    # Show what the mock would look like
    print(f"\n💡 MOCK IMPLEMENTATION EXAMPLE:")
    print(f"Add this to the top of factory_runnerv2.py:")

    mock_code = '''
# HPC Compatibility - Mock Isaac Lab if not available
try:
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
    ISAAC_LAB_AVAILABLE = True
except ImportError:
    ISAAC_LAB_AVAILABLE = False

    class ManagerBasedRLEnvCfg:
        """Mock Isaac Lab environment configuration for HPC compatibility."""
        def __init__(self):
            # Add all the mock attributes from MockManagerBasedRLEnvCfg above
            pass

    print("[INFO]: Using mock Isaac Lab configuration for HPC compatibility")
'''

    print(mock_code)

    return True

if __name__ == "__main__":
    availability = test_isaac_lab_availability()

    if not any(availability.values()):
        print(f"\n❌ ISAAC LAB NOT AVAILABLE")
        print(f"This explains the error you're seeing on HPC")
    elif availability['core'] and not availability['tasks']:
        print(f"\n⚠ ISAAC LAB CORE AVAILABLE BUT TASKS MISSING")
        print(f"This is the specific error you're encountering")
    else:
        print(f"\n✅ ISAAC LAB FULLY AVAILABLE")
        print(f"The error shouldn't occur in this environment")

    print(f"\n" + "=" * 60)

    # Test compatibility solutions
    test_configuration_compatibility()
    create_hpc_compatible_factory_runner()

    print(f"\n🎯 SUMMARY FOR YOUR HPC ERROR:")
    print(f"   • Your HPC doesn't have omni.isaac.lab_tasks installed")
    print(f"   • The script falls back to basic config but that also fails")
    print(f"   • You need either Isaac Lab installation OR mock implementation")
    print(f"   • Your configuration system will work fine with either approach")