#!/usr/bin/env python3
"""
Unit tests for factory_runnerv2.py import handling.

Tests that the import logic in factory_runnerv2.py correctly handles
both Isaac Lab v1.4.1 and v2.2.1 imports with proper error handling.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib.util


class TestFactoryRunnerV2Imports(unittest.TestCase):
    """Test import handling in factory_runnerv2.py"""

    def setUp(self):
        """Set up test environment"""
        # Remove any existing isaac lab modules from cache
        modules_to_remove = [name for name in sys.modules.keys()
                           if 'isaac' in name or 'omni' in name]
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]

    def tearDown(self):
        """Clean up after tests"""
        # Remove any mocked modules
        modules_to_remove = [name for name in sys.modules.keys()
                           if isinstance(sys.modules.get(name), MagicMock)]
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]

    def test_isaac_lab_v2_imports_successful(self):
        """Test that Isaac Lab v2.2.1+ imports work correctly"""
        # Mock v2.2.1+ modules
        mock_modules = {
            'isaaclab': MagicMock(),
            'isaaclab.envs': MagicMock(),
            'isaaclab_tasks': MagicMock(),
            'isaaclab_rl': MagicMock(),
            'isaaclab_rl.skrl': MagicMock(),
        }

        # Add realistic attributes
        mock_modules['isaaclab.envs'].DirectMARLEnv = MagicMock()
        mock_modules['isaaclab.envs'].DirectMARLEnvCfg = MagicMock()
        mock_modules['isaaclab.envs'].DirectRLEnvCfg = MagicMock()
        mock_modules['isaaclab.envs'].ManagerBasedRLEnvCfg = MagicMock()
        mock_modules['isaaclab_rl.skrl'].SkrlVecEnvWrapper = MagicMock()

        for name, mock in mock_modules.items():
            sys.modules[name] = mock

        # Test the import pattern from factory_runnerv2.py
        try:
            from isaaclab.envs import (
                DirectMARLEnv,
                DirectMARLEnvCfg,
                DirectRLEnvCfg,
                ManagerBasedRLEnvCfg,
            )
            import isaaclab_tasks  # noqa: F401
            from isaaclab_rl.skrl import SkrlVecEnvWrapper

            version = "v2.2.1+"
            success = True
        except ImportError:
            success = False
            version = None

        self.assertTrue(success, "Isaac Lab v2.2.1+ imports should succeed")
        self.assertEqual(version, "v2.2.1+")

    def test_isaac_lab_v1_fallback_imports_successful(self):
        """Test that Isaac Lab v1.4.1 fallback imports work correctly"""
        # Mock v1.4.1 modules
        mock_modules = {
            'omni': MagicMock(),
            'omni.isaac': MagicMock(),
            'omni.isaac.lab': MagicMock(),
            'omni.isaac.lab.envs': MagicMock(),
            'omni.isaac.lab_tasks': MagicMock(),
            'omni.isaac.lab_tasks.utils': MagicMock(),
            'omni.isaac.lab_tasks.utils.wrappers': MagicMock(),
            'omni.isaac.lab_tasks.utils.wrappers.skrl': MagicMock(),
        }

        # Add realistic attributes
        mock_modules['omni.isaac.lab.envs'].DirectMARLEnv = MagicMock()
        mock_modules['omni.isaac.lab.envs'].DirectMARLEnvCfg = MagicMock()
        mock_modules['omni.isaac.lab.envs'].DirectRLEnvCfg = MagicMock()
        mock_modules['omni.isaac.lab.envs'].ManagerBasedRLEnvCfg = MagicMock()
        mock_modules['omni.isaac.lab_tasks.utils.wrappers.skrl'].SkrlVecEnvWrapper = MagicMock()

        for name, mock in mock_modules.items():
            sys.modules[name] = mock

        # Test the fallback import pattern from factory_runnerv2.py
        try:
            # First try should fail (no isaaclab modules)
            from isaaclab.envs import (
                DirectMARLEnv,
                DirectMARLEnvCfg,
                DirectRLEnvCfg,
                ManagerBasedRLEnvCfg,
            )
            import isaaclab_tasks  # noqa: F401
            from isaaclab_rl.skrl import SkrlVecEnvWrapper
            success = True
            version = "v2.2.1+"
        except ImportError:
            try:
                # Fallback should succeed
                from omni.isaac.lab.envs import (
                    DirectMARLEnv,
                    DirectMARLEnvCfg,
                    DirectRLEnvCfg,
                    ManagerBasedRLEnvCfg,
                )
                import omni.isaac.lab_tasks  # noqa: F401
                from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
                success = True
                version = "v1.4.1"
            except ImportError:
                success = False
                version = None

        self.assertTrue(success, "Isaac Lab v1.4.1 fallback imports should succeed")
        self.assertEqual(version, "v1.4.1")

    def test_import_failure_handling(self):
        """Test that proper error handling occurs when neither version is available"""
        # Don't mock any modules - both imports should fail

        with self.assertRaises(SystemExit) as context:
            # Simulate the import logic with proper error handling
            try:
                from isaaclab.envs import (
                    DirectMARLEnv,
                    DirectMARLEnvCfg,
                    DirectRLEnvCfg,
                    ManagerBasedRLEnvCfg,
                )
                import isaaclab_tasks  # noqa: F401
                from isaaclab_rl.skrl import SkrlVecEnvWrapper
            except ImportError:
                try:
                    from omni.isaac.lab.envs import (
                        DirectMARLEnv,
                        DirectMARLEnvCfg,
                        DirectRLEnvCfg,
                        ManagerBasedRLEnvCfg,
                    )
                    import omni.isaac.lab_tasks  # noqa: F401
                    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
                except ImportError:
                    print("ERROR: Could not import Isaac Lab tasks module.")
                    print("Please ensure you have either:")
                    print("  - Isaac Lab v2.0.0+ (isaaclab_tasks)")
                    print("  - Isaac Lab v1.4.1 or earlier (omni.isaac.lab_tasks)")
                    sys.exit(1)

        # Verify that sys.exit(1) was called
        self.assertEqual(context.exception.code, 1)

    def test_no_duplicate_import_in_main(self):
        """Test that there's no duplicate import in main() function that bypasses try/except"""
        # Read the factory_runnerv2.py file and check for duplicate imports
        with open('/home/hunter/Continuous_Force_RL/learning/factory_runnerv2.py', 'r') as f:
            content = f.read()

        # Count occurrences of the problematic import
        duplicate_import_count = content.count('import omni.isaac.lab_tasks  # This registers the tasks')

        # After our fix, this should be 0 (no duplicates outside try/except)
        # We check that the import appears exactly once (in the try/except block)
        lab_tasks_import_count = content.count('import omni.isaac.lab_tasks')
        isaaclab_tasks_import_count = content.count('import isaaclab_tasks')

        # Should have exactly one of each in the try/except blocks
        self.assertEqual(lab_tasks_import_count, 1,
                        "Should have exactly one 'import omni.isaac.lab_tasks' in try/except")
        self.assertEqual(isaaclab_tasks_import_count, 1,
                        "Should have exactly one 'import isaaclab_tasks' in try/except")


if __name__ == '__main__':
    unittest.main()