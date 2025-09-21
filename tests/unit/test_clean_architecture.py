"""
Unit tests for the clean configuration architecture.

Tests that apply_complete_configuration handles all configuration in one place
and that configclass objects are preserved throughout the process.
"""

import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass
from configs.config_manager import ConfigManager


@dataclass
class MockTaskConfig:
    """Mock Isaac Lab task configuration class."""
    duration_s: float = 10.0
    fixed_asset: object = None
    held_asset: object = None
    success_threshold: float = 0.04
    engage_threshold: float = 0.9
    name: str = "peg_insert"

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
    pos_action_bounds: list = None

    def __post_init__(self):
        if self.default_task_prop_gains is None:
            self.default_task_prop_gains = [1.0, 1.0, 1.0]
        if self.default_task_force_gains is None:
            self.default_task_force_gains = [0.1, 0.1, 0.1]
        if self.pos_action_bounds is None:
            self.pos_action_bounds = [0.05, 0.05, 0.05]


class MockEnvConfig:
    """Mock Isaac Lab environment configuration."""
    def __init__(self):
        self.task = MockTaskConfig()
        self.ctrl = MockCtrlConfig()
        self.episode_length_s = 15.0
        self.obs_order = ['fingertip_pos', 'ee_linvel']
        self.state_order = ['fingertip_pos', 'held_pos']
        self.scene = Mock()
        self.scene.num_envs = 128
        self.scene.replicate_physics = True
        self.sim = Mock()
        self.sim.device = None


class TestCleanArchitecture:
    """Test the clean configuration architecture."""

    def test_complete_configuration_preserves_isaac_lab_objects(self):
        """Test that apply_complete_configuration preserves Isaac Lab configclass objects."""
        # Create mock Isaac Lab environment config
        env_cfg = MockEnvConfig()
        agent_cfg = {'agent': {}, 'models': {'policy': {}, 'value': {}}}

        # Store initial types
        initial_task_type = type(env_cfg.task)
        initial_ctrl_type = type(env_cfg.ctrl)

        # Create comprehensive resolved configuration
        resolved_config = {
            'environment': {
                'task': {
                    'success_threshold': 0.02,
                    'engage_threshold': 0.05,
                    'name': 'factory_task'
                },
                'ctrl': {
                    'force_action_bounds': [50.0, 50.0, 50.0],
                    'torque_action_bounds': [0.5, 0.5, 0.5]
                },
                'episode_length_s': 15.0
            },
            'primary': {
                'break_forces': -1,
                'debug_mode': False
            },
            'derived': {
                'total_agents': 2,
                'total_num_envs': 512,
                'rollout_steps': 225
            },
            'model': {
                'use_hybrid_agent': True,
                'actor': {'n': 1, 'latent_size': 256},
                'critic': {'n': 3, 'latent_size': 1024}
            },
            'wrappers': {
                'force_torque_sensor': {'enabled': True}
            },
            'experiment': {
                'name': 'test_experiment',
                'wandb_project': 'Test_Project'
            }
        }

        # Apply complete configuration
        with patch('torch.cuda.is_available', return_value=True):
            ConfigManager.apply_complete_configuration(env_cfg, agent_cfg, resolved_config)

        # Verify that Isaac Lab objects are preserved
        assert type(env_cfg.task) == initial_task_type, "Task type should be preserved"
        assert type(env_cfg.ctrl) == initial_ctrl_type, "Ctrl type should be preserved"

        # Verify that objects are not dictionaries
        assert not isinstance(env_cfg.task, dict), "Task should not be a dictionary"
        assert not isinstance(env_cfg.ctrl, dict), "Ctrl should not be a dictionary"

        # Verify that Isaac Lab attributes are still accessible
        assert hasattr(env_cfg.task, 'fixed_asset'), "Task should still have fixed_asset"
        assert hasattr(env_cfg.task, 'duration_s'), "Task should still have duration_s"
        assert hasattr(env_cfg.ctrl, 'default_task_prop_gains'), "Ctrl should still have gains"

        # Verify that new attributes were added correctly
        assert env_cfg.task.success_threshold == 0.02, "New task attributes should be set"
        assert env_cfg.task.engage_threshold == 0.05, "New task attributes should be set"
        assert env_cfg.task.name == 'factory_task', "New task attributes should be set"

        assert hasattr(env_cfg.ctrl, 'force_action_bounds'), "New ctrl attributes should be added"
        assert env_cfg.ctrl.force_action_bounds == [50.0, 50.0, 50.0], "New ctrl attributes should be set"

    def test_complete_configuration_handles_all_config_sections(self):
        """Test that apply_complete_configuration handles all configuration sections."""
        env_cfg = MockEnvConfig()
        agent_cfg = {'agent': {}, 'models': {'policy': {}, 'value': {}}}

        resolved_config = {
            'environment': {'episode_length_s': 20.0},
            'primary': {'break_forces': [10.0, 20.0], 'debug_mode': True},
            'derived': {'total_agents': 4, 'total_num_envs': 1024},
            'model': {'use_hybrid_agent': False, 'critic_output_init_mean': 100},
            'wrappers': {'force_torque_sensor': {'enabled': True}},
            'experiment': {'name': 'comprehensive_test'}
        }

        with patch('torch.cuda.is_available', return_value=True):
            ConfigManager.apply_complete_configuration(env_cfg, agent_cfg, resolved_config)

        # Verify all sections were processed
        assert env_cfg.episode_length_s == 20.0, "Environment config should be applied"
        assert env_cfg.break_force == [10.0, 20.0], "Primary config should be applied"
        assert env_cfg.scene.num_envs == 1024, "Derived config should be applied"
        assert agent_cfg['models']['critic_output_init_mean'] == 100, "Model config should be applied"
        assert 'force_torque' in env_cfg.obs_order, "Wrapper config should be applied"
        assert agent_cfg['agent']['experiment']['name'] == 'comprehensive_test', "Experiment config should be applied"

    def test_complete_configuration_easy_mode(self):
        """Test that easy mode configuration is applied correctly."""
        env_cfg = MockEnvConfig()
        agent_cfg = {'agent': {}}

        resolved_config = {
            'environment': {},
            'primary': {'debug_mode': True, 'break_forces': -1},
            'derived': {'total_agents': 1, 'total_num_envs': 128},
            'model': {},
            'wrappers': {},
            'experiment': {}
        }

        with patch('torch.cuda.is_available', return_value=True):
            ConfigManager.apply_complete_configuration(env_cfg, agent_cfg, resolved_config)

        # Verify easy mode was applied
        assert agent_cfg['agent']['easy_mode'] == True, "Easy mode should be enabled in agent config"
        assert env_cfg.task.hand_init_pos == [0.0, 0.0, 0.035], "Easy mode should set hand position"
        assert env_cfg.task.fixed_asset_init_pos_noise == [0.0, 0.0, 0.0], "Easy mode should reduce noise"

    def test_complete_configuration_device_setup(self):
        """Test that device configuration is handled correctly."""
        env_cfg = MockEnvConfig()
        agent_cfg = {'agent': {}}

        resolved_config = {
            'environment': {},
            'primary': {'break_forces': -1},
            'derived': {'total_agents': 1, 'total_num_envs': 128},
            'model': {},
            'wrappers': {},
            'experiment': {}
        }

        # Test with CUDA available
        with patch('torch.cuda.is_available', return_value=True):
            ConfigManager.apply_complete_configuration(env_cfg, agent_cfg, resolved_config)
            assert env_cfg.sim.device == 'cuda:0', "Should set CUDA device when available"

        # Reset and test with CUDA unavailable
        env_cfg.sim.device = None
        with patch('torch.cuda.is_available', return_value=False):
            ConfigManager.apply_complete_configuration(env_cfg, agent_cfg, resolved_config)
            assert env_cfg.sim.device == 'cpu', "Should set CPU device when CUDA unavailable"

    def test_deprecated_apply_to_isaac_lab_warning(self):
        """Test that the deprecated method shows a warning."""
        env_cfg = MockEnvConfig()
        agent_cfg = {'agent': {}}
        resolved_config = {'environment': {}}

        with patch('builtins.print') as mock_print:
            ConfigManager.apply_to_isaac_lab(env_cfg, agent_cfg, resolved_config)

            # Check that deprecation warning was printed
            warning_printed = any('WARNING - Using deprecated apply_to_isaac_lab' in str(call)
                                for call in mock_print.call_args_list)
            assert warning_printed, "Deprecated method should show warning"

    def test_isaac_lab_attribute_access_after_configuration(self):
        """Test that Isaac Lab can access attributes correctly after configuration."""
        env_cfg = MockEnvConfig()
        agent_cfg = {'agent': {}}

        resolved_config = {
            'environment': {
                'task': {'custom_param': 'test_value'}
            },
            'primary': {'break_forces': -1},
            'derived': {'total_agents': 1, 'total_num_envs': 128},
            'model': {},
            'wrappers': {},
            'experiment': {}
        }

        with patch('torch.cuda.is_available', return_value=True):
            ConfigManager.apply_complete_configuration(env_cfg, agent_cfg, resolved_config)

        # Simulate Isaac Lab's access pattern: self.cfg_task = cfg.task
        cfg_task = env_cfg.task

        # Then Isaac Lab accesses: self.cfg_task.fixed_asset
        try:
            fixed_asset = cfg_task.fixed_asset
            duration = cfg_task.duration_s
            custom = cfg_task.custom_param

            assert fixed_asset is not None, "Should access Isaac Lab fixed_asset"
            assert duration == 10.0, "Should access Isaac Lab duration_s"
            assert custom == 'test_value', "Should access custom parameter"

        except AttributeError as e:
            pytest.fail(f"Isaac Lab style attribute access failed: {e}")