"""
Unit tests to verify that configclass objects are preserved during configuration application.

This test ensures that when we start with proper configclass objects for task and ctrl,
they remain as configclass objects after ConfigManager.apply_to_isaac_lab() is called.
"""

import pytest
from dataclasses import dataclass
from unittest.mock import Mock
from configs.config_manager import ConfigManager


@dataclass
class MockTaskConfig:
    """Mock Isaac Lab task configuration class."""
    duration_s: float = 10.0
    fixed_asset: object = None
    held_asset: object = None

    def __post_init__(self):
        if self.fixed_asset is None:
            self.fixed_asset = Mock()
        if self.held_asset is None:
            self.held_asset = Mock()


@dataclass
class MockCtrlConfig:
    """Mock Isaac Lab control configuration class."""
    default_task_prop_gains: list = None
    default_task_force_gains: list = None

    def __post_init__(self):
        if self.default_task_prop_gains is None:
            self.default_task_prop_gains = [1.0, 1.0, 1.0]
        if self.default_task_force_gains is None:
            self.default_task_force_gains = [0.1, 0.1, 0.1]


class MockEnvConfig:
    """Mock Isaac Lab environment configuration."""
    def __init__(self):
        self.task = MockTaskConfig()
        self.ctrl = MockCtrlConfig()


class TestConfigClassPreservation:
    """Test that configclass objects are preserved during configuration."""

    def test_configclass_types_preserved_with_overrides(self):
        """Test that task and ctrl remain as configclass objects after applying overrides."""
        # Create mock Isaac Lab environment config with proper configclass objects
        env_cfg = MockEnvConfig()

        # Verify initial types are configclass objects (not dictionaries)
        initial_task_type = type(env_cfg.task)
        initial_ctrl_type = type(env_cfg.ctrl)

        assert not isinstance(env_cfg.task, dict), "Initial task should not be a dictionary"
        assert not isinstance(env_cfg.ctrl, dict), "Initial ctrl should not be a dictionary"
        assert hasattr(env_cfg.task, 'fixed_asset'), "Initial task should have Isaac Lab attributes"
        assert hasattr(env_cfg.ctrl, 'default_task_prop_gains'), "Initial ctrl should have Isaac Lab attributes"

        # Create configuration with task and ctrl overrides
        resolved_config = {
            'environment': {
                'task': {
                    'success_threshold': 0.02,
                    'engage_threshold': 0.05,
                    'name': 'factory_task'
                },
                'ctrl': {
                    'pos_action_bounds': [0.05, 0.05, 0.05],
                    'force_action_bounds': [50.0, 50.0, 50.0],
                    'torque_action_bounds': [0.5, 0.5, 0.5]
                }
            }
        }

        agent_cfg = {}

        # Apply configuration - this should preserve object types
        ConfigManager.apply_to_isaac_lab(env_cfg, agent_cfg, resolved_config)

        # Verify that types are preserved after configuration
        final_task_type = type(env_cfg.task)
        final_ctrl_type = type(env_cfg.ctrl)

        # Critical assertion: types must be preserved
        assert final_task_type == initial_task_type, (
            f"Task type changed from {initial_task_type} to {final_task_type}. "
            f"ConfigManager should preserve configclass objects!"
        )

        assert final_ctrl_type == initial_ctrl_type, (
            f"Ctrl type changed from {initial_ctrl_type} to {final_ctrl_type}. "
            f"ConfigManager should preserve configclass objects!"
        )

        # Verify they're still not dictionaries
        assert not isinstance(env_cfg.task, dict), "Final task should not be a dictionary"
        assert not isinstance(env_cfg.ctrl, dict), "Final ctrl should not be a dictionary"

        # Verify original Isaac Lab attributes are preserved
        assert hasattr(env_cfg.task, 'fixed_asset'), "Task should still have Isaac Lab fixed_asset"
        assert hasattr(env_cfg.task, 'duration_s'), "Task should still have Isaac Lab duration_s"
        assert hasattr(env_cfg.ctrl, 'default_task_prop_gains'), "Ctrl should still have Isaac Lab gains"

        # Verify new attributes were added
        assert hasattr(env_cfg.task, 'success_threshold'), "Task should have new success_threshold"
        assert env_cfg.task.success_threshold == 0.02, "New task attribute should have correct value"
        assert hasattr(env_cfg.ctrl, 'pos_action_bounds'), "Ctrl should have new pos_action_bounds"
        assert env_cfg.ctrl.pos_action_bounds == [0.05, 0.05, 0.05], "New ctrl attribute should have correct value"

        # Test that we can still access Isaac Lab attributes as objects (not dict access)
        try:
            _ = env_cfg.task.fixed_asset  # This should work for configclass
            _ = env_cfg.ctrl.default_task_prop_gains  # This should work for configclass
        except AttributeError:
            pytest.fail("Should be able to access configclass attributes with dot notation")

    def test_configclass_attributes_can_be_accessed_like_isaac_lab_expects(self):
        """Test that after configuration, we can access attributes exactly like Isaac Lab does."""
        env_cfg = MockEnvConfig()

        # Apply some configuration
        resolved_config = {
            'environment': {
                'task': {
                    'custom_param': 'test_value'
                }
            }
        }

        ConfigManager.apply_to_isaac_lab(env_cfg, {}, resolved_config)

        # Test Isaac Lab style access patterns that should work
        # This simulates what Isaac Lab does in factory_env.py line 169
        try:
            # This is what Isaac Lab does: self.cfg_task = cfg.task
            cfg_task = env_cfg.task

            # Then later: self.cfg_task.fixed_asset
            fixed_asset = cfg_task.fixed_asset

            # Should be able to access both original and new attributes
            duration = cfg_task.duration_s
            custom = cfg_task.custom_param

            assert fixed_asset is not None, "Should be able to access fixed_asset"
            assert duration == 10.0, "Should be able to access original duration_s"
            assert custom == 'test_value', "Should be able to access new custom_param"

        except AttributeError as e:
            pytest.fail(f"Isaac Lab style attribute access failed: {e}")

    def test_configclass_preservation_with_complex_nested_config(self):
        """Test preservation with more complex nested configuration."""
        env_cfg = MockEnvConfig()

        initial_task_type = type(env_cfg.task)
        initial_ctrl_type = type(env_cfg.ctrl)

        # More complex configuration
        resolved_config = {
            'environment': {
                'task': {
                    'success_threshold': 0.02,
                    'engage_threshold': 0.05,
                    'name': 'factory_task',
                    'timing': {
                        'max_episode_steps': 1000,
                        'dt': 0.01
                    }
                },
                'ctrl': {
                    'pos_action_bounds': [0.05, 0.05, 0.05],
                    'force_action_bounds': [50.0, 50.0, 50.0],
                    'gains': {
                        'kp': [100.0, 100.0, 100.0],
                        'kd': [10.0, 10.0, 10.0]
                    }
                },
                'some_other_param': 'should_be_handled_separately'
            }
        }

        ConfigManager.apply_to_isaac_lab(env_cfg, {}, resolved_config)

        # Types must be preserved even with complex config
        assert type(env_cfg.task) == initial_task_type, "Task type should be preserved with complex config"
        assert type(env_cfg.ctrl) == initial_ctrl_type, "Ctrl type should be preserved with complex config"

        # Check that nested values were set correctly
        assert env_cfg.task.success_threshold == 0.02
        assert env_cfg.task.timing == {'max_episode_steps': 1000, 'dt': 0.01}
        assert env_cfg.ctrl.pos_action_bounds == [0.05, 0.05, 0.05]
        assert env_cfg.ctrl.gains == {'kp': [100.0, 100.0, 100.0], 'kd': [10.0, 10.0, 10.0]}

    def test_no_configuration_changes_preserve_types(self):
        """Test that even with no task/ctrl config, types are preserved."""
        env_cfg = MockEnvConfig()

        initial_task_type = type(env_cfg.task)
        initial_ctrl_type = type(env_cfg.ctrl)

        # Configuration with no task or ctrl changes
        resolved_config = {
            'environment': {
                'some_other_setting': 'value'
            }
        }

        ConfigManager.apply_to_isaac_lab(env_cfg, {}, resolved_config)

        # Types should be completely unchanged
        assert type(env_cfg.task) == initial_task_type, "Task type should be unchanged when no task config provided"
        assert type(env_cfg.ctrl) == initial_ctrl_type, "Ctrl type should be unchanged when no ctrl config provided"