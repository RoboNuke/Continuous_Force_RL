#!/usr/bin/env python3
"""
Test Isaac Lab import compatibility without executing full modules.

This script tests the import patterns used in our codebase to ensure
they work with both Isaac Lab v1.4.1 and v2.2.1.
"""

import sys
import importlib
from unittest.mock import MagicMock

# Add project to path
sys.path.insert(0, '.')

def test_import_patterns():
    """Test specific import patterns used in our codebase."""

    print("🔧 TESTING ISAAC LAB IMPORT PATTERNS")
    print("=" * 60)

    # Test each import pattern individually
    patterns = [
        {
            "name": "AppLauncher (v2.2.1 -> v1.4.1)",
            "code": """
try:
    from isaaclab.app import AppLauncher
    version = "v2.2.1+"
except ImportError:
    from omni.isaac.lab.app import AppLauncher
    version = "v1.4.1"
print(f"AppLauncher imported from {version}")
            """
        },
        {
            "name": "Environment Classes (v2.2.1 -> v1.4.1)",
            "code": """
try:
    from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
    version = "v2.2.1+"
except ImportError:
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
    version = "v1.4.1"
print(f"Environment classes imported from {version}")
            """
        },
        {
            "name": "Tasks Module (v2.2.1 -> v1.4.1)",
            "code": """
try:
    import isaaclab_tasks
    version = "v2.2.1+"
except ImportError:
    import omni.isaac.lab_tasks as isaaclab_tasks
    version = "v1.4.1"
print(f"Tasks module imported from {version}")
            """
        },
        {
            "name": "SKRL Wrapper (v2.2.1 -> v1.4.1)",
            "code": """
try:
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    version = "v2.2.1+"
except ImportError:
    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
    version = "v1.4.1"
print(f"SKRL wrapper imported from {version}")
            """
        },
        {
            "name": "Torch Utils (v2.2.1 -> v1.4.1)",
            "code": """
try:
    import isaacsim.core.utils.torch as torch_utils
    version = "v2.2.1+"
except ImportError:
    import omni.isaac.core.utils.torch as torch_utils
    version = "v1.4.1"
print(f"Torch utils imported from {version}")
            """
        },
        {
            "name": "Math Utils (v2.2.1 -> v1.4.1)",
            "code": """
try:
    from isaaclab.utils.math import axis_angle_from_quat
    version = "v2.2.1+"
except ImportError:
    from omni.isaac.lab.utils.math import axis_angle_from_quat
    version = "v1.4.1"
print(f"Math utils imported from {version}")
            """
        },
        {
            "name": "Robot View (v2.2.1 -> v1.4.1)",
            "code": """
try:
    from isaacsim.core.api.robots import RobotView
    version = "v2.2.1+"
except ImportError:
    from omni.isaac.core.articulations import ArticulationView as RobotView
    version = "v1.4.1"
print(f"RobotView imported from {version}")
            """
        },
        {
            "name": "Config Class (v2.2.1 -> v1.4.1 -> fallback)",
            "code": """
try:
    from isaaclab.utils.configclass import configclass
    version = "v2.2.1+"
except ImportError:
    try:
        from omni.isaac.lab.utils.configclass import configclass
        version = "v1.4.1"
    except ImportError:
        from isaacsim.core.utils.configclass import configclass
        version = "Isaac Sim"
print(f"Config class imported from {version}")
            """
        }
    ]

    results = []

    for pattern in patterns:
        print(f"\n📋 Testing: {pattern['name']}")
        print("-" * 50)

        # Test with v2.2.1 mocks (newer version available)
        print("  Scenario 1: Isaac Lab v2.2.1+ available")
        try:
            setup_v2_mocks()
            exec(pattern['code'].strip())
            results.append(f"✓ {pattern['name']} - v2.2.1 scenario")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results.append(f"✗ {pattern['name']} - v2.2.1 scenario: {e}")
        finally:
            cleanup_mocks()

        # Test with v1.4.1 mocks (older version available)
        print("  Scenario 2: Isaac Lab v1.4.1 available")
        try:
            setup_v1_mocks()
            exec(pattern['code'].strip())
            results.append(f"✓ {pattern['name']} - v1.4.1 scenario")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results.append(f"✗ {pattern['name']} - v1.4.1 scenario: {e}")
        finally:
            cleanup_mocks()

    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 60)
    success_count = sum(1 for result in results if result.startswith("✓"))
    total_count = len(results)

    for result in results:
        print(f"  {result}")

    print(f"\n📈 Overall Success Rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")

    if success_count == total_count:
        print("🎉 ALL IMPORT PATTERNS WORK CORRECTLY!")
        return True
    else:
        print("❌ SOME IMPORT PATTERNS FAILED!")
        return False


def setup_v2_mocks():
    """Set up mocks for Isaac Lab v2.2.1+"""
    mocks = {
        'isaaclab': MagicMock(),
        'isaaclab.app': MagicMock(),
        'isaaclab.envs': MagicMock(),
        'isaaclab.utils': MagicMock(),
        'isaaclab.utils.math': MagicMock(),
        'isaaclab.utils.configclass': MagicMock(),
        'isaaclab_tasks': MagicMock(),
        'isaaclab_rl': MagicMock(),
        'isaaclab_rl.skrl': MagicMock(),
        'isaacsim': MagicMock(),
        'isaacsim.core': MagicMock(),
        'isaacsim.core.utils': MagicMock(),
        'isaacsim.core.utils.torch': MagicMock(),
        'isaacsim.core.api': MagicMock(),
        'isaacsim.core.api.robots': MagicMock(),
    }

    # Add realistic attributes
    mocks['isaaclab.app'].AppLauncher = MagicMock()
    mocks['isaaclab.envs'].ManagerBasedRLEnvCfg = MagicMock()
    mocks['isaaclab.envs'].DirectRLEnvCfg = MagicMock()
    mocks['isaaclab.utils.math'].axis_angle_from_quat = MagicMock()
    mocks['isaaclab.utils.configclass'].configclass = MagicMock()
    mocks['isaaclab_rl.skrl'].SkrlVecEnvWrapper = MagicMock()
    mocks['isaacsim.core.api.robots'].RobotView = MagicMock()

    for name, mock in mocks.items():
        sys.modules[name] = mock


def setup_v1_mocks():
    """Set up mocks for Isaac Lab v1.4.1"""
    mocks = {
        'omni': MagicMock(),
        'omni.isaac': MagicMock(),
        'omni.isaac.lab': MagicMock(),
        'omni.isaac.lab.app': MagicMock(),
        'omni.isaac.lab.envs': MagicMock(),
        'omni.isaac.lab.utils': MagicMock(),
        'omni.isaac.lab.utils.math': MagicMock(),
        'omni.isaac.lab.utils.configclass': MagicMock(),
        'omni.isaac.lab_tasks': MagicMock(),
        'omni.isaac.lab_tasks.utils': MagicMock(),
        'omni.isaac.lab_tasks.utils.wrappers': MagicMock(),
        'omni.isaac.lab_tasks.utils.wrappers.skrl': MagicMock(),
        'omni.isaac.core': MagicMock(),
        'omni.isaac.core.utils': MagicMock(),
        'omni.isaac.core.utils.torch': MagicMock(),
        'omni.isaac.core.articulations': MagicMock(),
    }

    # Add realistic attributes
    mocks['omni.isaac.lab.app'].AppLauncher = MagicMock()
    mocks['omni.isaac.lab.envs'].ManagerBasedRLEnvCfg = MagicMock()
    mocks['omni.isaac.lab.envs'].DirectRLEnvCfg = MagicMock()
    mocks['omni.isaac.lab.utils.math'].axis_angle_from_quat = MagicMock()
    mocks['omni.isaac.lab.utils.configclass'].configclass = MagicMock()
    mocks['omni.isaac.lab_tasks.utils.wrappers.skrl'].SkrlVecEnvWrapper = MagicMock()
    mocks['omni.isaac.core.articulations'].ArticulationView = MagicMock()

    for name, mock in mocks.items():
        sys.modules[name] = mock


def cleanup_mocks():
    """Clean up all mocked modules"""
    mock_modules = [name for name, module in sys.modules.items()
                   if isinstance(module, MagicMock) and
                   ('isaac' in name or 'omni' in name or 'isaaclab' in name)]

    for name in mock_modules:
        del sys.modules[name]


if __name__ == "__main__":
    success = test_import_patterns()

    if success:
        print("\n✅ All Isaac Lab import patterns are compatible!")
        exit(0)
    else:
        print("\n❌ Some Isaac Lab import patterns need fixes!")
        exit(1)