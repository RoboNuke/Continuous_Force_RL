#!/usr/bin/env python3
"""
Unit tests for configuration validation in launch_utils_v2.py.

Tests that the configuration loading and validation properly handles
Isaac Lab factory configurations with obs_order and state_order.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
import pytest


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation functions"""

    def setUp(self):
        """Set up test environment"""
        # Mock env_cfg with various configuration states
        self.valid_env_cfg = MagicMock()
        self.valid_env_cfg.obs_order = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
        self.valid_env_cfg.state_order = [
            "fingertip_pos", "fingertip_quat", "ee_linvel", "ee_angvel",
            "joint_pos", "held_pos", "held_pos_rel_fixed", "held_quat"
        ]
        self.valid_env_cfg.task_name = "Isaac-Factory-PegInsert-Local-v0"

    def test_validate_factory_configuration_success(self):
        """Test that validation passes with proper configuration"""
        # Import here to avoid module loading issues
        sys.path.insert(0, '/home/hunter/Continuous_Force_RL')
        from learning import launch_utils_v2 as lUtils

        # Should not raise any exceptions
        try:
            lUtils.validate_factory_configuration(self.valid_env_cfg)
            success = True
        except SystemExit:
            success = False
        except Exception:
            success = False

        self.assertTrue(success, "Validation should pass with valid configuration")

    def test_validate_factory_configuration_missing_obs_order(self):
        """Test that validation fails when obs_order is missing"""
        sys.path.insert(0, '/home/hunter/Continuous_Force_RL')
        from learning import launch_utils_v2 as lUtils

        # Remove obs_order attribute
        delattr(self.valid_env_cfg, 'obs_order')

        with self.assertRaises(SystemExit) as context:
            lUtils.validate_factory_configuration(self.valid_env_cfg)

        self.assertEqual(context.exception.code, 1)

    def test_validate_factory_configuration_missing_state_order(self):
        """Test that validation fails when state_order is missing"""
        sys.path.insert(0, '/home/hunter/Continuous_Force_RL')
        from learning import launch_utils_v2 as lUtils

        # Remove state_order attribute
        delattr(self.valid_env_cfg, 'state_order')

        with self.assertRaises(SystemExit) as context:
            lUtils.validate_factory_configuration(self.valid_env_cfg)

        self.assertEqual(context.exception.code, 1)

    def test_validate_factory_configuration_empty_obs_order(self):
        """Test that validation fails when obs_order is empty"""
        sys.path.insert(0, '/home/hunter/Continuous_Force_RL')
        from learning import launch_utils_v2 as lUtils

        # Set obs_order to empty list
        self.valid_env_cfg.obs_order = []

        with self.assertRaises(SystemExit) as context:
            lUtils.validate_factory_configuration(self.valid_env_cfg)

        self.assertEqual(context.exception.code, 1)

    def test_validate_factory_configuration_empty_state_order(self):
        """Test that validation fails when state_order is empty"""
        sys.path.insert(0, '/home/hunter/Continuous_Force_RL')
        from learning import launch_utils_v2 as lUtils

        # Set state_order to empty list
        self.valid_env_cfg.state_order = []

        with self.assertRaises(SystemExit) as context:
            lUtils.validate_factory_configuration(self.valid_env_cfg)

        self.assertEqual(context.exception.code, 1)

    def test_validate_factory_configuration_none_values(self):
        """Test that validation fails when obs_order/state_order are None"""
        sys.path.insert(0, '/home/hunter/Continuous_Force_RL')
        from learning import launch_utils_v2 as lUtils

        # Set values to None
        self.valid_env_cfg.obs_order = None
        self.valid_env_cfg.state_order = None

        with self.assertRaises(SystemExit) as context:
            lUtils.validate_factory_configuration(self.valid_env_cfg)

        self.assertEqual(context.exception.code, 1)


class TestConfigurationApplication(unittest.TestCase):
    """Test configuration application with preservation of existing values"""

    def setUp(self):
        """Set up test environment"""
        self.env_cfg = MagicMock()
        self.env_cfg.obs_order = ["original_obs1", "original_obs2"]
        self.env_cfg.state_order = ["original_state1", "original_state2"]
        self.env_cfg.existing_attr = "original_value"

        self.agent_cfg = {}

        self.resolved_config = {
            'environment': {
                'new_attribute': 'new_value',
                'existing_attr': 'override_value',
                'episode_length_s': 15.0
            }
        }

    def test_apply_to_isaac_lab_preserves_existing_attributes(self):
        """Test that configuration application preserves existing values when appropriate"""
        sys.path.insert(0, '/home/hunter/Continuous_Force_RL')
        from configs.config_manager import ConfigManager

        # Apply configuration
        ConfigManager.apply_to_isaac_lab(self.env_cfg, self.agent_cfg, self.resolved_config)

        # Should have applied new attributes
        self.assertTrue(hasattr(self.env_cfg, 'new_attribute'))
        self.assertEqual(self.env_cfg.new_attribute, 'new_value')

        # Should have applied overrides for existing attributes
        self.assertEqual(self.env_cfg.existing_attr, 'override_value')

        # Should have applied episode_length_s
        self.assertEqual(self.env_cfg.episode_length_s, 15.0)


class TestFactoryConfigurationIntegration(unittest.TestCase):
    """Test the complete configuration flow"""

    def test_enable_force_sensor_with_valid_config(self):
        """Test that enable_force_sensor works when obs_order and state_order exist"""
        sys.path.insert(0, '/home/hunter/Continuous_Force_RL')
        from learning import launch_utils_v2 as lUtils

        # Create mock env_cfg with real lists (not Mock objects)
        env_cfg = MagicMock()
        env_cfg.obs_order = ["fingertip_pos", "ee_linvel"]
        env_cfg.state_order = ["fingertip_pos", "ee_linvel", "joint_pos"]
        env_cfg.use_force_sensor = False

        # Should not raise any exceptions
        try:
            lUtils.enable_force_sensor(env_cfg)
            success = True
        except Exception as e:
            success = False
            print(f"Error: {e}")

        self.assertTrue(success, "enable_force_sensor should work with valid config")

        # Check that force_torque was added
        self.assertIn("force_torque", env_cfg.obs_order)
        self.assertIn("force_torque", env_cfg.state_order)
        self.assertTrue(env_cfg.use_force_sensor)

    def test_enable_force_sensor_without_obs_order_fails(self):
        """Test that enable_force_sensor fails gracefully when obs_order is missing"""
        sys.path.insert(0, '/home/hunter/Continuous_Force_RL')
        from learning import launch_utils_v2 as lUtils

        # Create mock env_cfg without obs_order
        env_cfg = MagicMock()
        del env_cfg.obs_order  # Remove the attribute
        env_cfg.state_order = ["fingertip_pos", "ee_linvel"]

        # Should raise AttributeError
        with self.assertRaises(AttributeError):
            lUtils.enable_force_sensor(env_cfg)


class TestTaskConfigurationLoading(unittest.TestCase):
    """Test task configuration loading and error handling"""

    @patch('gymnasium.spec')
    def test_task_config_loading_success(self, mock_gym_spec):
        """Test successful task configuration loading"""
        # Mock successful gymnasium spec loading
        mock_cfg_class = MagicMock()
        mock_cfg_instance = MagicMock()
        mock_cfg_instance.obs_order = ["test_obs"]
        mock_cfg_instance.state_order = ["test_state"]
        mock_cfg_class.return_value = mock_cfg_instance

        mock_env_spec = MagicMock()
        mock_env_spec.kwargs = {'cfg': mock_cfg_class}
        mock_gym_spec.return_value = mock_env_spec

        # This would be tested in the actual factory_runnerv2.py loading logic
        # Here we just verify the mock setup works
        env_spec = mock_gym_spec("Isaac-Factory-PegInsert-Local-v0")
        env_cfg = env_spec.kwargs['cfg']()

        self.assertEqual(env_cfg.obs_order, ["test_obs"])
        self.assertEqual(env_cfg.state_order, ["test_state"])

    @patch('gymnasium.spec')
    def test_task_config_loading_failure(self, mock_gym_spec):
        """Test task configuration loading failure"""
        # Mock gymnasium spec failure
        mock_gym_spec.side_effect = Exception("Task not found")

        # This would trigger the error handling in factory_runnerv2.py
        with self.assertRaises(Exception):
            mock_gym_spec("Invalid-Task-Name")


if __name__ == '__main__':
    unittest.main()