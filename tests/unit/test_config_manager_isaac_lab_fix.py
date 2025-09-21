"""
Unit tests for ConfigManager Isaac Lab configuration handling fix.

Tests that the ConfigManager properly handles Isaac Lab task and control configuration
objects, ensuring that 'task' config maps to 'cfg_task' and 'ctrl' maps to 'cfg_ctrl'.
"""

import pytest
from unittest.mock import Mock, MagicMock
from configs.config_manager import ConfigManager


class TestConfigManagerIsaacLabFix:
    """Test the Isaac Lab configuration mapping fix."""

    def test_task_config_maps_to_cfg_task(self):
        """Test that 'task' configuration correctly maps to 'cfg_task' attribute."""
        # Create mock Isaac Lab environment config
        env_cfg = Mock()

        # Create mock task configuration object (not a dictionary)
        mock_task_config = Mock()
        mock_task_config.duration_s = 10.0
        mock_task_config.fixed_asset = Mock()
        mock_task_config.held_asset = Mock()

        # Set the task attribute (Isaac Lab naming)
        env_cfg.task = mock_task_config

        # Create configuration with task overrides
        resolved_config = {
            'environment': {
                'task': {
                    'success_threshold': 0.02,
                    'engage_threshold': 0.05,
                    'name': 'factory_task'
                }
            }
        }

        agent_cfg = {}

        # Apply configuration - should not raise an error
        ConfigManager.apply_to_isaac_lab(env_cfg, agent_cfg, resolved_config)

        # Verify that the task configuration was merged into task
        assert hasattr(mock_task_config, 'success_threshold')
        assert mock_task_config.success_threshold == 0.02
        assert hasattr(mock_task_config, 'engage_threshold')
        assert mock_task_config.engage_threshold == 0.05
        assert hasattr(mock_task_config, 'name')
        assert mock_task_config.name == 'factory_task'

    def test_ctrl_config_maps_to_ctrl(self):
        """Test that 'ctrl' configuration correctly maps to 'ctrl' attribute."""
        # Create mock Isaac Lab environment config
        env_cfg = Mock()

        # Create mock control configuration object (not a dictionary)
        mock_ctrl_config = Mock()
        mock_ctrl_config.default_task_prop_gains = [1.0, 1.0, 1.0]
        mock_ctrl_config.default_task_force_gains = [0.1, 0.1, 0.1]

        # Set the ctrl attribute (Isaac Lab naming)
        env_cfg.ctrl = mock_ctrl_config

        # Create configuration with ctrl overrides
        resolved_config = {
            'environment': {
                'ctrl': {
                    'pos_action_bounds': [0.05, 0.05, 0.05],
                    'force_action_bounds': [50.0, 50.0, 50.0]
                }
            }
        }

        agent_cfg = {}

        # Apply configuration - should not raise an error
        ConfigManager.apply_to_isaac_lab(env_cfg, agent_cfg, resolved_config)

        # Verify that the ctrl configuration was merged into ctrl
        assert hasattr(mock_ctrl_config, 'pos_action_bounds')
        assert mock_ctrl_config.pos_action_bounds == [0.05, 0.05, 0.05]
        assert hasattr(mock_ctrl_config, 'force_action_bounds')
        assert mock_ctrl_config.force_action_bounds == [50.0, 50.0, 50.0]

    def test_validation_fails_when_task_missing(self):
        """Test that validation fails when task attribute is missing."""
        # Create a real object without task attribute
        class FakeEnvConfig:
            pass

        env_cfg = FakeEnvConfig()

        # Create configuration with task overrides
        resolved_config = {
            'environment': {
                'task': {
                    'success_threshold': 0.02
                }
            }
        }

        agent_cfg = {}

        # Should raise ValueError due to missing task
        with pytest.raises(ValueError, match="missing required 'task' attribute"):
            ConfigManager.apply_to_isaac_lab(env_cfg, agent_cfg, resolved_config)

    def test_validation_fails_when_task_is_dict(self):
        """Test that validation fails when task is a dictionary instead of object."""
        # Create mock Isaac Lab environment config
        env_cfg = Mock()

        # Set task as a dictionary (should be an object)
        env_cfg.task = {'duration_s': 10.0}

        # Create configuration with task overrides
        resolved_config = {
            'environment': {
                'task': {
                    'success_threshold': 0.02
                }
            }
        }

        agent_cfg = {}

        # Should raise ValueError due to task being a dict
        with pytest.raises(ValueError, match="configuration is a dictionary instead of a proper"):
            ConfigManager.apply_to_isaac_lab(env_cfg, agent_cfg, resolved_config)

    def test_validation_fails_when_task_missing_isaac_lab_attributes(self):
        """Test that validation fails when task doesn't have Isaac Lab attributes."""
        # Create a simple object without Isaac Lab attributes
        class FakeTaskConfig:
            pass

        env_cfg = Mock()
        env_cfg.task = FakeTaskConfig()

        # Create configuration with task overrides
        resolved_config = {
            'environment': {
                'task': {
                    'success_threshold': 0.02
                }
            }
        }

        agent_cfg = {}

        # Should raise ValueError due to missing Isaac Lab attributes
        with pytest.raises(ValueError, match="does not have expected Isaac Lab attributes"):
            ConfigManager.apply_to_isaac_lab(env_cfg, agent_cfg, resolved_config)

    def test_no_fallback_object_creation_for_critical_configs(self):
        """Test that no fallback objects are created for missing critical configs."""
        # Create a real object without task attribute
        class FakeEnvConfig:
            pass

        env_cfg = FakeEnvConfig()

        # Create configuration with task overrides
        resolved_config = {
            'environment': {
                'task': {
                    'success_threshold': 0.02
                }
            }
        }

        agent_cfg = {}

        # Should raise ValueError and not create a fallback task object
        with pytest.raises(ValueError):
            ConfigManager.apply_to_isaac_lab(env_cfg, agent_cfg, resolved_config)

        # Ensure no 'task' attribute was created as fallback
        assert not hasattr(env_cfg, 'task')