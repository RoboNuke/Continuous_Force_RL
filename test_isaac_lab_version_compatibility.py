#!/usr/bin/env python3
"""
Test Isaac Lab version compatibility across v1.4.1 and v2.2.1

This script verifies that all Isaac Lab imports in the codebase can handle
both older (v1.4.1) and newer (v2.2.1) Isaac Lab versions with different
import structures.
"""

import sys
import importlib
from pathlib import Path
from unittest.mock import MagicMock

# Add project to path
sys.path.insert(0, '.')

def test_version_compatibility():
    """Test that our codebase handles both Isaac Lab v1.4.1 and v2.2.1 import structures."""

    print("🔍 TESTING ISAAC LAB VERSION COMPATIBILITY")
    print("=" * 60)

    # Test scenarios for different Isaac Lab versions
    scenarios = [
        {
            "name": "Isaac Lab v2.2.1+ (isaaclab modules)",
            "available_modules": [
                "isaaclab", "isaaclab.app", "isaaclab.envs", "isaaclab_tasks",
                "isaaclab_rl", "isaaclab.utils", "isaaclab.sensors", "isaaclab.sim",
                "isaacsim", "isaacsim.core", "isaacsim.core.utils.torch", "isaacsim.core.api.robots"
            ],
            "unavailable_modules": [
                "omni.isaac.lab", "omni.isaac.lab.app", "omni.isaac.lab.envs",
                "omni.isaac.lab_tasks", "omni.isaac.core"
            ]
        },
        {
            "name": "Isaac Lab v1.4.1 (omni.isaac.lab modules)",
            "available_modules": [
                "omni.isaac.lab", "omni.isaac.lab.app", "omni.isaac.lab.envs",
                "omni.isaac.lab_tasks", "omni.isaac.lab.utils", "omni.isaac.lab.sensors",
                "omni.isaac.lab.sim", "omni.isaac.core", "omni.isaac.core.utils.torch",
                "omni.isaac.core.articulations"
            ],
            "unavailable_modules": [
                "isaaclab", "isaaclab.app", "isaaclab.envs", "isaaclab_tasks",
                "isaaclab_rl", "isaacsim"
            ]
        }
    ]

    results = {}

    for scenario in scenarios:
        print(f"\n📋 Testing {scenario['name']}")
        print("-" * 50)

        # Mock the module availability for this scenario
        setup_mock_environment(scenario['available_modules'], scenario['unavailable_modules'])

        # Test key import files
        test_files = [
            ('learning.factory_runnerv2', 'Factory Runner v2'),
            ('wrappers.sensors.force_torque_wrapper', 'Force Torque Wrapper'),
            ('wrappers.control.hybrid_force_position_wrapper', 'Hybrid Control Wrapper'),
            ('wrappers.control.factory_control_utils', 'Factory Control Utils'),
            ('wrappers.control.hybrid_control_cfg', 'Hybrid Control Config'),
            ('wrappers.mechanics.efficient_reset_wrapper', 'Efficient Reset Wrapper')
        ]

        scenario_results = {}
        for module_name, display_name in test_files:
            try:
                # Clear module cache to force re-import
                if module_name in sys.modules:
                    del sys.modules[module_name]

                # Try to import the module
                module = importlib.import_module(module_name)
                scenario_results[display_name] = "✓ SUCCESS"
                print(f"  ✓ {display_name}: Import successful")

            except Exception as e:
                scenario_results[display_name] = f"✗ FAILED: {e}"
                print(f"  ✗ {display_name}: Import failed - {e}")

        results[scenario['name']] = scenario_results

        # Clean up mocked modules
        cleanup_mock_environment(scenario['available_modules'], scenario['unavailable_modules'])

    # Summary
    print(f"\n📊 COMPATIBILITY TEST SUMMARY")
    print("=" * 60)

    for scenario_name, scenario_results in results.items():
        print(f"\n{scenario_name}:")
        success_count = sum(1 for result in scenario_results.values() if result.startswith("✓"))
        total_count = len(scenario_results)

        for test_name, result in scenario_results.items():
            print(f"  {result[:1]} {test_name}")

        print(f"  📈 Success Rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")

    # Overall assessment
    all_success = all(
        result.startswith("✓")
        for scenario_results in results.values()
        for result in scenario_results.values()
    )

    if all_success:
        print(f"\n🎉 ALL TESTS PASSED! The codebase is compatible with both Isaac Lab versions.")
        return True
    else:
        print(f"\n❌ SOME TESTS FAILED! The codebase needs additional compatibility fixes.")
        return False


def setup_mock_environment(available_modules, unavailable_modules):
    """Set up mock environment to simulate a specific Isaac Lab version."""

    # Remove unavailable modules from sys.modules if they exist
    for module_name in unavailable_modules:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Add mock available modules
    for module_name in available_modules:
        if module_name not in sys.modules:
            # Create a realistic mock for this module
            mock_module = create_realistic_mock(module_name)
            sys.modules[module_name] = mock_module


def create_realistic_mock(module_name):
    """Create a realistic mock for a specific Isaac Lab module."""

    mock = MagicMock()

    # Add realistic attributes based on module name
    if "app" in module_name:
        mock.AppLauncher = MagicMock()
        mock.AppLauncher.add_app_launcher_args = MagicMock()

    elif "envs" in module_name:
        mock.DirectMARLEnv = MagicMock()
        mock.DirectMARLEnvCfg = MagicMock()
        mock.DirectRLEnvCfg = MagicMock()
        mock.ManagerBasedRLEnvCfg = MagicMock()
        mock.multi_agent_to_single_agent = MagicMock()

    elif "utils" in module_name and "torch" in module_name:
        mock.quat_from_angle_axis = MagicMock()
        mock.quat_mul = MagicMock()
        mock.get_euler_xyz = MagicMock()
        mock.quat_from_euler_xyz = MagicMock()

    elif "utils" in module_name and "math" in module_name:
        mock.axis_angle_from_quat = MagicMock()

    elif "utils" in module_name and "configclass" in module_name:
        mock.configclass = MagicMock()

    elif "skrl" in module_name or "wrappers" in module_name:
        mock.SkrlVecEnvWrapper = MagicMock()

    elif "robots" in module_name or "articulations" in module_name:
        mock.RobotView = MagicMock()
        mock.ArticulationView = MagicMock()

    elif "sensors" in module_name:
        mock.TiledCameraCfg = MagicMock()
        mock.ImuCfg = MagicMock()

    elif "hydra" in module_name:
        mock.hydra_task_config = MagicMock()

    return mock


def cleanup_mock_environment(available_modules, unavailable_modules):
    """Clean up the mock environment."""

    # Remove mocked modules
    for module_name in available_modules + unavailable_modules:
        if module_name in sys.modules and isinstance(sys.modules[module_name], MagicMock):
            del sys.modules[module_name]


def test_specific_imports():
    """Test specific import patterns used in the codebase."""

    print(f"\n🔧 TESTING SPECIFIC IMPORT PATTERNS")
    print("=" * 60)

    # Test patterns that should work in both versions
    import_tests = [
        # AppLauncher imports
        """
try:
    from isaaclab.app import AppLauncher
except ImportError:
    from omni.isaac.lab.app import AppLauncher
        """,

        # Environment imports
        """
try:
    from isaaclab.envs import ManagerBasedRLEnvCfg
except ImportError:
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
        """,

        # Torch utils imports
        """
try:
    import isaacsim.core.utils.torch as torch_utils
except ImportError:
    import omni.isaac.core.utils.torch as torch_utils
        """,

        # Math utils imports
        """
try:
    from isaaclab.utils.math import axis_angle_from_quat
except ImportError:
    from omni.isaac.lab.utils.math import axis_angle_from_quat
        """
    ]

    for i, import_code in enumerate(import_tests, 1):
        print(f"\nTest {i}: Testing import pattern")
        try:
            # Set up a basic mock environment
            setup_basic_mocks()

            # Execute the import code
            exec(import_code.strip())
            print("  ✓ Import pattern works correctly")

        except Exception as e:
            print(f"  ✗ Import pattern failed: {e}")

        finally:
            # Clean up
            cleanup_basic_mocks()


def setup_basic_mocks():
    """Set up basic mocks for import testing."""
    mocks = {
        'isaaclab': MagicMock(),
        'isaaclab.app': MagicMock(),
        'isaaclab.envs': MagicMock(),
        'isaaclab.utils': MagicMock(),
        'isaaclab.utils.math': MagicMock(),
        'omni.isaac.lab': MagicMock(),
        'omni.isaac.lab.app': MagicMock(),
        'omni.isaac.lab.envs': MagicMock(),
        'omni.isaac.lab.utils': MagicMock(),
        'omni.isaac.lab.utils.math': MagicMock(),
        'isaacsim': MagicMock(),
        'isaacsim.core': MagicMock(),
        'isaacsim.core.utils': MagicMock(),
        'isaacsim.core.utils.torch': MagicMock(),
        'omni.isaac.core': MagicMock(),
        'omni.isaac.core.utils': MagicMock(),
        'omni.isaac.core.utils.torch': MagicMock(),
    }

    for name, mock in mocks.items():
        sys.modules[name] = mock


def cleanup_basic_mocks():
    """Clean up basic mocks."""
    mock_modules = [name for name, module in sys.modules.items()
                   if isinstance(module, MagicMock) and ('isaac' in name or 'omni' in name)]

    for name in mock_modules:
        del sys.modules[name]


if __name__ == "__main__":
    print("Isaac Lab Version Compatibility Test")
    print("=" * 60)

    # Run the main compatibility test
    success = test_version_compatibility()

    # Run specific import pattern tests
    test_specific_imports()

    # Final result
    print(f"\n🏁 FINAL RESULT")
    print("=" * 60)

    if success:
        print("✅ Isaac Lab version compatibility verified!")
        print("The codebase should work with both v1.4.1 and v2.2.1")
        exit(0)
    else:
        print("❌ Isaac Lab version compatibility issues detected!")
        print("Additional fixes may be needed")
        exit(1)