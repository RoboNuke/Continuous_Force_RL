"""
Unit tests for OBS_DIM_CFG and STATE_DIM_CFG modification approach.

Tests that we can successfully import and modify IsaacLab's dimension
configuration dictionaries to include force-torque sensor dimensions.
"""

import pytest
import sys


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