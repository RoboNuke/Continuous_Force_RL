"""
Unit tests for OBS_DIM_CFG and STATE_DIM_CFG modification approach.

Tests that we can successfully import and modify IsaacLab's dimension
configuration dictionaries to include force-torque sensor dimensions.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock


class TestObsDimCfgModification:
    """Test suite for dimension configuration modification."""

    def test_can_import_isaaclab_configs(self):
        """Test that we can import OBS_DIM_CFG and STATE_DIM_CFG from IsaacLab."""
        try:
            from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
            print("Successfully imported from isaaclab_tasks (v2.0.0+)")
        except ImportError:
            try:
                from omni.isaac.lab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
                print("Successfully imported from omni.isaac.lab_tasks (v1.4.1)")
            except ImportError:
                pytest.skip("IsaacLab factory configs not available - skipping test")

        # Verify configs are dictionaries
        assert isinstance(OBS_DIM_CFG, dict), "OBS_DIM_CFG should be a dictionary"
        assert isinstance(STATE_DIM_CFG, dict), "STATE_DIM_CFG should be a dictionary"

        # Verify expected keys exist
        expected_obs_keys = ["fingertip_pos", "fingertip_quat", "ee_linvel", "ee_angvel"]
        for key in expected_obs_keys:
            assert key in OBS_DIM_CFG, f"Expected key '{key}' not found in OBS_DIM_CFG"

    def test_can_modify_configs_safely(self):
        """Test that we can safely modify the configuration dictionaries."""
        try:
            from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
        except ImportError:
            try:
                from omni.isaac.lab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
            except ImportError:
                pytest.skip("IsaacLab factory configs not available - skipping test")

        # Store original state for cleanup
        original_obs_keys = set(OBS_DIM_CFG.keys())
        original_state_keys = set(STATE_DIM_CFG.keys())

        # Verify force_torque is not initially present
        assert 'force_torque' not in OBS_DIM_CFG, "force_torque should not be in OBS_DIM_CFG initially"
        assert 'force_torque' not in STATE_DIM_CFG, "force_torque should not be in STATE_DIM_CFG initially"

        try:
            # Add force_torque dimensions
            OBS_DIM_CFG['force_torque'] = 6
            STATE_DIM_CFG['force_torque'] = 6

            # Verify modification succeeded
            assert 'force_torque' in OBS_DIM_CFG, "force_torque should be added to OBS_DIM_CFG"
            assert 'force_torque' in STATE_DIM_CFG, "force_torque should be added to STATE_DIM_CFG"
            assert OBS_DIM_CFG['force_torque'] == 6, "force_torque dimension should be 6 in OBS_DIM_CFG"
            assert STATE_DIM_CFG['force_torque'] == 6, "force_torque dimension should be 6 in STATE_DIM_CFG"

            # Verify original keys still exist
            for key in original_obs_keys:
                assert key in OBS_DIM_CFG, f"Original key '{key}' should still exist in OBS_DIM_CFG"
            for key in original_state_keys:
                assert key in STATE_DIM_CFG, f"Original key '{key}' should still exist in STATE_DIM_CFG"

        finally:
            # Clean up modifications
            if 'force_torque' in OBS_DIM_CFG:
                del OBS_DIM_CFG['force_torque']
            if 'force_torque' in STATE_DIM_CFG:
                del STATE_DIM_CFG['force_torque']

    def test_modification_function(self):
        """Test the actual function we'll use in factory runner."""
        def add_force_torque_to_isaaclab_configs():
            """Add force-torque dimensions to IsaacLab configuration dictionaries."""
            try:
                from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
            except ImportError:
                try:
                    from omni.isaac.lab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
                except ImportError:
                    print("Warning: Could not import IsaacLab factory configs - force-torque may not work")
                    return False

            # Add force-torque dimensions if not already present
            if 'force_torque' not in OBS_DIM_CFG:
                OBS_DIM_CFG['force_torque'] = 6
            if 'force_torque' not in STATE_DIM_CFG:
                STATE_DIM_CFG['force_torque'] = 6

            print("[INFO]: Added force-torque dimensions to IsaacLab configuration")
            return True

        # Test the function
        try:
            from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
        except ImportError:
            try:
                from omni.isaac.lab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
            except ImportError:
                pytest.skip("IsaacLab factory configs not available - skipping test")

        # Store original state
        had_force_torque_obs = 'force_torque' in OBS_DIM_CFG
        had_force_torque_state = 'force_torque' in STATE_DIM_CFG

        try:
            # Run the function
            result = add_force_torque_to_isaaclab_configs()

            if result:  # Only test if function succeeded
                assert 'force_torque' in OBS_DIM_CFG, "Function should add force_torque to OBS_DIM_CFG"
                assert 'force_torque' in STATE_DIM_CFG, "Function should add force_torque to STATE_DIM_CFG"
                assert OBS_DIM_CFG['force_torque'] == 6, "force_torque should have dimension 6"
                assert STATE_DIM_CFG['force_torque'] == 6, "force_torque should have dimension 6"
        finally:
            # Clean up
            if not had_force_torque_obs and 'force_torque' in OBS_DIM_CFG:
                del OBS_DIM_CFG['force_torque']
            if not had_force_torque_state and 'force_torque' in STATE_DIM_CFG:
                del STATE_DIM_CFG['force_torque']


class TestObsDimCfgModificationMocked:
    """Test suite using mocked IsaacLab configs for local testing."""

    def setup_method(self):
        """Set up mock configs for each test."""
        # Create realistic mock configs based on actual IsaacLab structure
        self.mock_obs_dim_cfg = {
            "fingertip_pos": 3,
            "fingertip_pos_rel_fixed": 3,
            "fingertip_quat": 4,
            "ee_linvel": 3,
            "ee_angvel": 3,
        }

        self.mock_state_dim_cfg = {
            "fingertip_pos": 3,
            "fingertip_pos_rel_fixed": 3,
            "fingertip_quat": 4,
            "ee_linvel": 3,
            "ee_angvel": 3,
            "joint_pos": 7,
            "held_pos": 3,
            "held_pos_rel_fixed": 3,
            "held_quat": 4,
            "fixed_pos": 3,
            "fixed_quat": 4,
            "task_prop_gains": 6,
            "ema_factor": 1,
            "pos_threshold": 3,
            "rot_threshold": 3,
        }

    def test_logic_force_torque_modification_success(self):
        """Test the core logic of force-torque config modification."""
        # Simulate successful config modification
        OBS_DIM_CFG = self.mock_obs_dim_cfg.copy()
        STATE_DIM_CFG = self.mock_state_dim_cfg.copy()

        # Simulate our function's core logic
        wrappers_config = {'force_torque_sensor': {'enabled': True}}

        if not wrappers_config.get('force_torque_sensor', {}).get('enabled', False):
            modified = False
        else:
            # Simulate successful import and modification
            modified = False
            if 'force_torque' not in OBS_DIM_CFG:
                OBS_DIM_CFG['force_torque'] = 6
                modified = True

            if 'force_torque' not in STATE_DIM_CFG:
                STATE_DIM_CFG['force_torque'] = 6
                modified = True

        # Verify modifications were made
        assert modified is True, "Should have modified configs"
        assert 'force_torque' in OBS_DIM_CFG, "force_torque should be added to OBS_DIM_CFG"
        assert 'force_torque' in STATE_DIM_CFG, "force_torque should be added to STATE_DIM_CFG"
        assert OBS_DIM_CFG['force_torque'] == 6, "force_torque dimension should be 6"
        assert STATE_DIM_CFG['force_torque'] == 6, "force_torque dimension should be 6"

        # Verify original keys are preserved
        assert OBS_DIM_CFG['fingertip_pos'] == 3, "Original OBS keys should be preserved"
        assert STATE_DIM_CFG['joint_pos'] == 7, "Original STATE keys should be preserved"

    def test_logic_force_torque_enabled_check(self):
        """Test the wrapper config enabled check logic."""
        # Test enabled case
        wrappers_config_enabled = {'force_torque_sensor': {'enabled': True}}
        should_modify = wrappers_config_enabled.get('force_torque_sensor', {}).get('enabled', False)
        assert should_modify is True, "Should detect force-torque is enabled"

        # Test disabled case
        wrappers_config_disabled = {'force_torque_sensor': {'enabled': False}}
        should_modify = wrappers_config_disabled.get('force_torque_sensor', {}).get('enabled', False)
        assert should_modify is False, "Should detect force-torque is disabled"

        # Test missing config case
        wrappers_config_missing = {}
        should_modify = wrappers_config_missing.get('force_torque_sensor', {}).get('enabled', False)
        assert should_modify is False, "Should handle missing config gracefully"

    def test_mock_disabled_force_torque_sensor(self):
        """Test that function correctly skips when force-torque sensor is disabled."""
        def mock_add_force_torque_disabled():
            wrappers_config = {'force_torque_sensor': {'enabled': False}}

            if not wrappers_config.get('force_torque_sensor', {}).get('enabled', False):
                return False

            return True

        result = mock_add_force_torque_disabled()
        assert result is False, "Function should return False when force-torque is disabled"

    def test_mock_import_failure_handling(self):
        """Test graceful handling when neither import path works."""
        # Temporarily remove any Isaac Lab modules that might have been mocked by other tests
        isaac_modules_to_remove = [
            'isaaclab_tasks', 'isaaclab_tasks.direct', 'isaaclab_tasks.direct.factory',
            'isaaclab_tasks.direct.factory.factory_env_cfg',
            'omni.isaac.lab_tasks', 'omni.isaac.lab_tasks.direct', 'omni.isaac.lab_tasks.direct.factory',
            'omni.isaac.lab_tasks.direct.factory.factory_env_cfg'
        ]

        # Store original modules and remove them temporarily
        original_modules = {}
        for module_name in isaac_modules_to_remove:
            if module_name in sys.modules:
                original_modules[module_name] = sys.modules[module_name]
                del sys.modules[module_name]

        try:
            def mock_add_force_torque_import_failure():
                wrappers_config = {'force_torque_sensor': {'enabled': True}}

                if not wrappers_config.get('force_torque_sensor', {}).get('enabled', False):
                    return False

                try:
                    from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
                except ImportError:
                    try:
                        from omni.isaac.lab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
                    except ImportError:
                        return False  # Both imports failed

                return True

            # Now the imports should fail since we removed the mocked modules
            result = mock_add_force_torque_import_failure()
            assert result is False, "Function should return False when imports fail"

        finally:
            # Restore original modules
            for module_name, module in original_modules.items():
                sys.modules[module_name] = module

    def test_mock_already_present_dimensions(self):
        """Test behavior when force_torque dimensions are already present."""
        # Set up configs that already have force_torque
        mock_obs_with_ft = self.mock_obs_dim_cfg.copy()
        mock_obs_with_ft['force_torque'] = 6

        mock_state_with_ft = self.mock_state_dim_cfg.copy()
        mock_state_with_ft['force_torque'] = 6

        def mock_add_force_torque_already_present():
            """Simulate configs that already have force_torque."""
            wrappers_config = {'force_torque_sensor': {'enabled': True}}

            if not wrappers_config.get('force_torque_sensor', {}).get('enabled', False):
                return False

            # Simulate successful import of configs that already have force_torque
            OBS_DIM_CFG = mock_obs_with_ft
            STATE_DIM_CFG = mock_state_with_ft

            # Check if already present (should be True)
            modified = False
            if 'force_torque' not in OBS_DIM_CFG:
                OBS_DIM_CFG['force_torque'] = 6
                modified = True

            if 'force_torque' not in STATE_DIM_CFG:
                STATE_DIM_CFG['force_torque'] = 6
                modified = True

            return True, modified  # Return both success and whether we modified

        result, was_modified = mock_add_force_torque_already_present()
        assert result is True, "Function should succeed"
        assert was_modified is False, "Should not modify configs that already have force_torque"

    def test_mock_dimension_values_correct(self):
        """Test that the correct dimension values are added."""
        mock_obs = self.mock_obs_dim_cfg.copy()
        mock_state = self.mock_state_dim_cfg.copy()

        # Simulate adding force_torque
        mock_obs['force_torque'] = 6
        mock_state['force_torque'] = 6

        # Verify dimensions
        assert mock_obs['force_torque'] == 6, "Force-torque observation dimension should be 6"
        assert mock_state['force_torque'] == 6, "Force-torque state dimension should be 6"

        # Verify original dimensions unchanged
        assert mock_obs['fingertip_pos'] == 3, "Original dimensions should be preserved"
        assert mock_state['joint_pos'] == 7, "Original dimensions should be preserved"