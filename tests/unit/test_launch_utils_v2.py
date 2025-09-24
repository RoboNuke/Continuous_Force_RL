"""
Unit tests for launch_utils_v2.py functions.

Tests all utility functions for configuration-based factory training
including environment configuration, agent setup, wrapper application,
and model creation functions.
"""

import pytest
import torch
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from copy import deepcopy

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set up minimal mocks for Isaac Lab, Isaac Sim, and Wandb only
from tests.mocks.mock_launch_utils_deps import setup_minimal_mocks, MockEnvConfig, create_mock_agent_config, MockIsaacLabEnv, MockWandb
setup_minimal_mocks()

# Mock wandb for this specific test file
from unittest.mock import patch
sys.modules['wandb'] = MockWandb

# Import the module under test
import learning.launch_utils_v2 as launch_utils


class TestEnvironmentConfigurationFunctions:
    """Test environment configuration functions."""

    def test_apply_easy_mode(self):
        """Test apply_easy_mode function."""
        env_cfg = MockEnvConfig()
        agent_cfg = create_mock_agent_config()

        launch_utils.apply_easy_mode(env_cfg, agent_cfg)

        # Check agent config changes
        assert agent_cfg['agent']['easy_mode'] == True

        # Check environment config changes
        assert env_cfg.task.duration_s == env_cfg.episode_length_s
        assert env_cfg.task.hand_init_pos == [0.0, 0.0, 0.035]
        assert env_cfg.task.hand_init_pos_noise == [0.0025, 0.0025, 0.00]
        assert env_cfg.task.hand_init_orn_noise == [0.0, 0.0, 0.0]
        assert env_cfg.task.fixed_asset_init_pos_noise == [0.0, 0.0, 0.0]
        assert env_cfg.task.fixed_asset_init_orn_deg == 0.0
        assert env_cfg.task.fixed_asset_init_orn_range_deg == 0.0
        assert env_cfg.task.held_asset_pos_noise == [0.0, 0.0, 0.0]
        assert env_cfg.task.held_asset_rot_init == 0.0

    def test_configure_environment_scene(self):
        """Test configure_environment_scene function."""
        env_cfg = MockEnvConfig()
        primary = {'num_envs': 32}
        derived = {'total_num_envs': 64, 'total_agents': 4}

        launch_utils.configure_environment_scene(env_cfg, primary, derived)

        assert env_cfg.scene.num_envs == 64
        assert env_cfg.scene.replicate_physics == True
        assert env_cfg.num_agents == 4

    def test_enable_force_sensor(self):
        """Test enable_force_sensor function."""
        env_cfg = MockEnvConfig()

        # Test initial state
        assert env_cfg.use_force_sensor == False
        assert 'force_torque' not in env_cfg.obs_order
        assert 'force_torque' not in env_cfg.state_order

        launch_utils.enable_force_sensor(env_cfg)

        assert env_cfg.use_force_sensor == True
        assert 'force_torque' in env_cfg.obs_order
        assert 'force_torque' in env_cfg.state_order

    def test_enable_force_sensor_already_present(self):
        """Test enable_force_sensor when force_torque already in lists."""
        env_cfg = MockEnvConfig()
        env_cfg.obs_order.append('force_torque')
        env_cfg.state_order.append('force_torque')

        original_obs_len = len(env_cfg.obs_order)
        original_state_len = len(env_cfg.state_order)

        launch_utils.enable_force_sensor(env_cfg)

        # Should not add duplicates
        assert len(env_cfg.obs_order) == original_obs_len
        assert len(env_cfg.state_order) == original_state_len
        assert env_cfg.obs_order.count('force_torque') == 1
        assert env_cfg.state_order.count('force_torque') == 1

    def test_apply_environment_overrides_direct_attributes(self):
        """Test apply_environment_overrides with direct attributes."""
        env_cfg = MockEnvConfig()
        environment_config = {
            'episode_length_s': 15.0,
            'num_agents': 8
        }

        launch_utils.apply_environment_overrides(env_cfg, environment_config)

        assert env_cfg.episode_length_s == 15.0
        assert env_cfg.num_agents == 8

    def test_apply_environment_overrides_nested_attributes(self):
        """Test apply_environment_overrides with nested attributes."""
        env_cfg = MockEnvConfig()
        environment_config = {
            'task.duration_s': 20.0,
            'sim.device': 'cuda'
        }

        launch_utils.apply_environment_overrides(env_cfg, environment_config)

        assert env_cfg.task.duration_s == 20.0
        assert env_cfg.sim.device == 'cuda'

    def test_apply_environment_overrides_nonexistent_attribute(self):
        """Test apply_environment_overrides with non-existent attribute."""
        env_cfg = MockEnvConfig()
        environment_config = {
            'nonexistent_attr': 'value'
        }

        # Should not raise error, just skip
        launch_utils.apply_environment_overrides(env_cfg, environment_config)

        # Attribute should not be added
        assert not hasattr(env_cfg, 'nonexistent_attr')


class TestAgentConfigurationFunctions:
    """Test agent configuration functions."""

    def test_apply_learning_config(self):
        """Test apply_learning_config function."""
        agent_cfg = create_mock_agent_config()
        learning_config = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'entropy_coeff': 0.01
        }
        max_rollout_steps = 128

        launch_utils.apply_learning_config(agent_cfg, learning_config, max_rollout_steps)

        # Check rollout configuration
        assert agent_cfg['agent']['rollouts'] == 128
        assert agent_cfg['agent']['experiment']['write_interval'] == 128
        assert agent_cfg['agent']['experiment']['checkpoint_interval'] == 1280

        # Check learning parameters
        assert agent_cfg['agent']['learning_rate'] == 0.001
        assert agent_cfg['agent']['batch_size'] == 64
        assert agent_cfg['agent']['entropy_coeff'] == 0.01

    def test_apply_learning_config_nonexistent_params(self):
        """Test apply_learning_config with parameters not in agent config."""
        agent_cfg = create_mock_agent_config()
        learning_config = {
            'nonexistent_param': 'value',
            'learning_rate': 0.001
        }
        max_rollout_steps = 64

        launch_utils.apply_learning_config(agent_cfg, learning_config, max_rollout_steps)

        # Should skip non-existent params
        assert 'nonexistent_param' not in agent_cfg['agent']
        # Should apply existing params
        assert agent_cfg['agent']['learning_rate'] == 0.001

    def test_apply_model_config_direct_params(self):
        """Test apply_model_config with direct model parameters."""
        agent_cfg = create_mock_agent_config()
        model_config = {
            'policy': {'hidden_size': 256},
            'value': {'hidden_size': 128}
        }

        launch_utils.apply_model_config(agent_cfg, model_config)

        assert agent_cfg['models']['policy'] == {'hidden_size': 256}
        assert agent_cfg['models']['value'] == {'hidden_size': 128}

    def test_apply_model_config_nested_params(self):
        """Test apply_model_config with nested model parameters."""
        agent_cfg = create_mock_agent_config()
        agent_cfg['models']['network'] = {'layers': []}
        model_config = {
            'network.layers': [64, 32, 16]
        }

        launch_utils.apply_model_config(agent_cfg, model_config)

        assert agent_cfg['models']['network']['layers'] == [64, 32, 16]

    def test_setup_experiment_logging(self):
        """Test setup_experiment_logging function."""
        env_cfg = MockEnvConfig()
        agent_cfg = create_mock_agent_config()
        resolved_config = {
            'primary': {'break_forces': [50.0, 100.0], 'agents_per_break_force': 1},
            'derived': {'total_num_envs': 64, 'total_agents': 2},
            'experiment': {
                'wandb_project': 'test_project',
                'name': 'test_experiment',
                'tags': ['test'],
                'group': 'test_group'
            }
        }

        launch_utils.setup_experiment_logging(env_cfg, agent_cfg, resolved_config)

        # Check experiment configuration
        assert agent_cfg['agent']['experiment']['project'] == 'test_project'
        assert agent_cfg['agent']['experiment']['tags'] == ['test', 'MockTask']  # MockTask is automatically added
        assert agent_cfg['agent']['experiment']['group'] == 'test_group'

        # Check agent-specific data
        assert agent_cfg['agent']['break_force'] == [50.0, 100.0]
        assert agent_cfg['agent']['num_envs'] == 64

    def test_setup_experiment_logging_with_defaults(self):
        """Test setup_experiment_logging with default values."""
        env_cfg = MockEnvConfig()
        agent_cfg = create_mock_agent_config()
        resolved_config = {
            'primary': {'break_forces': [50.0], 'agents_per_break_force': 1},
            'derived': {'total_num_envs': 32, 'total_agents': 1}
        }

        launch_utils.setup_experiment_logging(env_cfg, agent_cfg, resolved_config)

        # Check defaults
        assert agent_cfg['agent']['experiment']['project'] == 'Continuous_Force_RL'
        assert agent_cfg['agent']['experiment']['tags'] == ['MockTask']  # MockTask is automatically added
        assert agent_cfg['agent']['experiment']['group'] == ''


class TestWrapperApplicationFunctions:
    """Test wrapper application functions."""

    def test_apply_fragile_object_wrapper(self):
        """Test apply_fragile_object_wrapper function."""
        env = MockIsaacLabEnv()
        wrapper_config = {'break_force': [75.0, 80.0]}  # Must match total_agents
        primary = {'break_forces': [50.0]}
        derived = {'total_agents': 2}

        wrapped_env = launch_utils.apply_fragile_object_wrapper(env, wrapper_config, primary, derived)

        # Should use wrapper_config break_force over primary
        assert torch.allclose(wrapped_env.break_force[:8], torch.tensor([75.0] * 8))  # First 8 envs for agent 0
        assert torch.allclose(wrapped_env.break_force[8:], torch.tensor([80.0] * 8))  # Last 8 envs for agent 1
        assert wrapped_env.num_agents == 2
        assert wrapped_env.env == env

    def test_apply_fragile_object_wrapper_use_primary_break_force(self):
        """Test apply_fragile_object_wrapper using primary break_force."""
        env = MockIsaacLabEnv()
        wrapper_config = {}  # No break_force specified
        primary = {'break_forces': [100.0]}
        derived = {'total_agents': 1}

        wrapped_env = launch_utils.apply_fragile_object_wrapper(env, wrapper_config, primary, derived)

        # Should use primary break_forces
        assert torch.allclose(wrapped_env.break_force, torch.tensor([100.0] * 16))  # All 16 envs get same break force

    def test_apply_force_torque_wrapper(self):
        """Test apply_force_torque_wrapper function."""
        env = MockIsaacLabEnv()
        wrapper_config = {
            'use_tanh_scaling': True,
            'tanh_scale': 0.05
        }

        wrapped_env = launch_utils.apply_force_torque_wrapper(env, wrapper_config)

        assert wrapped_env.use_tanh_scaling == True
        assert wrapped_env.tanh_scale == 0.05
        assert wrapped_env.env == env

    def test_apply_force_torque_wrapper_defaults(self):
        """Test apply_force_torque_wrapper with default values."""
        env = MockIsaacLabEnv()
        wrapper_config = {}

        wrapped_env = launch_utils.apply_force_torque_wrapper(env, wrapper_config)

        assert wrapped_env.use_tanh_scaling == False
        assert wrapped_env.tanh_scale == 0.03

    def test_apply_observation_manager_wrapper(self):
        """Test apply_observation_manager_wrapper function."""
        env = MockIsaacLabEnv()
        wrapper_config = {'merge_strategy': 'policy_only'}

        wrapped_env = launch_utils.apply_observation_manager_wrapper(env, wrapper_config)

        assert wrapped_env.merge_strategy == 'policy_only'
        assert wrapped_env.env == env

    def test_apply_observation_manager_wrapper_default(self):
        """Test apply_observation_manager_wrapper with default strategy."""
        env = MockIsaacLabEnv()
        wrapper_config = {}

        wrapped_env = launch_utils.apply_observation_manager_wrapper(env, wrapper_config)

        assert wrapped_env.merge_strategy == 'concatenate'

    def test_apply_observation_noise_wrapper(self):
        """Test apply_observation_noise_wrapper function."""
        env = MockIsaacLabEnv()
        wrapper_config = {
            'global_noise_scale': 0.5,
            'enabled': True,
            'apply_to_critic': False,
            'noise_groups': {
                'fingertip_pos': {
                    'noise_type': 'gaussian',
                    'std': 0.02,
                    'mean': 0.0,
                    'enabled': True,
                    'timing': 'step'
                }
            }
        }

        wrapped_env = launch_utils.apply_observation_noise_wrapper(env, wrapper_config)

        # Check noise config
        assert wrapped_env.config.global_noise_scale == 0.5
        assert wrapped_env.config.enabled == True
        assert wrapped_env.config.apply_to_critic == False

        # Check noise group
        assert 'fingertip_pos' in wrapped_env.config.noise_groups
        noise_group = wrapped_env.config.noise_groups['fingertip_pos']
        assert noise_group.noise_type == 'gaussian'
        assert noise_group.std == 0.02

    def test_apply_hybrid_control_wrapper(self):
        """Test apply_hybrid_control_wrapper function."""
        env = MockIsaacLabEnv()
        wrapper_config = {
            'ctrl_torque': True,
            'reward_type': 'delta',
            'ema_factor': 0.3,
            'force_active_threshold': 0.2
        }

        wrapped_env = launch_utils.apply_hybrid_control_wrapper(env, wrapper_config)

        assert wrapped_env.ctrl_torque == True
        assert wrapped_env.reward_type == 'delta'
        assert wrapped_env.env == env

        # Check action space update for ctrl_torque=True
        assert wrapped_env.action_space.shape == (18,)  # 6 + 12

        # Check component_dims update
        assert env.cfg.component_dims['prev_actions'] == 18

    def test_apply_hybrid_control_wrapper_force_only(self):
        """Test apply_hybrid_control_wrapper with force only."""
        env = MockIsaacLabEnv()
        wrapper_config = {
            'ctrl_torque': False,
            'reward_type': 'simp'
        }

        wrapped_env = launch_utils.apply_hybrid_control_wrapper(env, wrapper_config)

        assert wrapped_env.ctrl_torque == False
        assert wrapped_env.action_space.shape == (12,)  # 6 + 6
        assert env.cfg.component_dims['prev_actions'] == 12

    def test_apply_factory_metrics_wrapper(self):
        """Test apply_factory_metrics_wrapper function."""
        env = MockIsaacLabEnv()
        derived = {'total_agents': 4}  # 16 environments / 4 agents = 4 envs per agent

        wrapped_env = launch_utils.apply_factory_metrics_wrapper(env, derived)

        assert wrapped_env.num_agents == 4
        assert wrapped_env.env == env

    def test_apply_wandb_logging_wrapper(self):
        """Test apply_wandb_logging_wrapper function."""
        env = MockIsaacLabEnv()
        wrapper_config = {}
        derived = {'total_agents': 2}
        agent_cfg = create_mock_agent_config()
        env_cfg = MockEnvConfig()
        resolved_config = {
            'agent_specific_configs': {
                'agent_0': {'agent_index': 0, 'break_force': 50.0},
                'agent_1': {'agent_index': 1, 'break_force': 100.0}
            }
        }

        wrapped_env = launch_utils.apply_wandb_logging_wrapper(
            env, wrapper_config, derived, agent_cfg, env_cfg, resolved_config
        )

        assert wrapped_env.num_agents == 2
        assert wrapped_env.env == env
        assert hasattr(wrapped_env, 'clean_env_cfg')  # Wrapper stores clean env_cfg after removing agent_configs

    def test_apply_wandb_logging_wrapper_missing_configs(self):
        """Test apply_wandb_logging_wrapper with missing agent_specific_configs."""
        env = MockIsaacLabEnv()
        wrapper_config = {}
        derived = {'total_agents': 1}
        agent_cfg = create_mock_agent_config()
        env_cfg = MockEnvConfig()
        resolved_config = {}  # Missing agent_specific_configs

        with pytest.raises(ValueError, match="agent_specific_configs not found"):
            launch_utils.apply_wandb_logging_wrapper(
                env, wrapper_config, derived, agent_cfg, env_cfg, resolved_config
            )

    def test_apply_enhanced_action_logging_wrapper(self):
        """Test apply_enhanced_action_logging_wrapper function."""
        env = MockIsaacLabEnv()
        wrapper_config = {
            'track_selection': True,
            'track_pos': True,
            'track_rot': False,
            'track_force': True,
            'track_torque': False,
            'force_size': 3,
            'logging_frequency': 50
        }

        wrapped_env = launch_utils.apply_enhanced_action_logging_wrapper(env, wrapper_config)

        assert wrapped_env.track_selection == True
        assert wrapped_env.track_pos == True
        assert wrapped_env.track_rot == False
        assert wrapped_env.force_size == 3
        assert wrapped_env.logging_frequency == 50


class TestPreprocessorSetup:
    """Test preprocessor setup functions."""

    def test_setup_preprocessors(self):
        """Test setup_preprocessors function."""
        env_cfg = MockEnvConfig()
        agent_cfg = create_mock_agent_config()
        env = MockIsaacLabEnv()
        learning_config = {
            'state_preprocessor': True,
            'value_preprocessor': True
        }

        launch_utils.setup_preprocessors(env_cfg, agent_cfg, env, learning_config)

        # Check state preprocessor
        assert agent_cfg['agent']['state_preprocessor'] is not None
        assert 'state_preprocessor_kwargs' in agent_cfg['agent']

        # Dynamic calculation: should equal observation_space + state_space
        expected_size = env.cfg.observation_space + env.cfg.state_space
        assert agent_cfg['agent']['state_preprocessor_kwargs']['size'] == expected_size, \
            f"State preprocessor size should be {expected_size} (obs_space + state_space), got {agent_cfg['agent']['state_preprocessor_kwargs']['size']}"
        assert agent_cfg['agent']['state_preprocessor_kwargs']['device'] == 'cpu'

        # Check value preprocessor
        assert agent_cfg['agent']['value_preprocessor'] is not None
        assert 'value_preprocessor_kwargs' in agent_cfg['agent']
        assert agent_cfg['agent']['value_preprocessor_kwargs']['size'] == 1

    def test_setup_preprocessors_disabled(self):
        """Test setup_preprocessors with disabled preprocessors."""
        env_cfg = MockEnvConfig()
        agent_cfg = create_mock_agent_config()
        env = MockIsaacLabEnv()
        learning_config = {
            'state_preprocessor': False,
            'value_preprocessor': False
        }

        launch_utils.setup_preprocessors(env_cfg, agent_cfg, env, learning_config)

        # Should not set preprocessors
        assert 'state_preprocessor' not in agent_cfg['agent']
        assert 'value_preprocessor' not in agent_cfg['agent']

    def test_setup_per_agent_preprocessors(self):
        """Test setup_per_agent_preprocessors function."""
        env_cfg = MockEnvConfig()
        env = MockIsaacLabEnv()
        learning_config = {
            'state_preprocessor': True,
            'value_preprocessor': True
        }
        num_agents = 4

        preprocessor_configs = launch_utils.setup_per_agent_preprocessors(
            env_cfg, env, learning_config, num_agents
        )

        # Check returned config
        assert 'state_preprocessor' in preprocessor_configs
        assert 'state_preprocessor_kwargs' in preprocessor_configs

        # Dynamic calculation: should equal observation_space + state_space
        expected_size = env.cfg.observation_space + env.cfg.state_space
        assert preprocessor_configs['state_preprocessor_kwargs']['size'] == expected_size, \
            f"State preprocessor size should be {expected_size} (obs_space + state_space), got {preprocessor_configs['state_preprocessor_kwargs']['size']}"

        assert 'value_preprocessor' in preprocessor_configs
        assert 'value_preprocessor_kwargs' in preprocessor_configs
        assert preprocessor_configs['value_preprocessor_kwargs']['size'] == 1

    def test_setup_per_agent_preprocessors_disabled(self):
        """Test setup_per_agent_preprocessors with disabled preprocessors."""
        env_cfg = MockEnvConfig()
        env = MockIsaacLabEnv()
        learning_config = {
            'state_preprocessor': False,
            'value_preprocessor': False
        }
        num_agents = 2

        preprocessor_configs = launch_utils.setup_per_agent_preprocessors(
            env_cfg, env, learning_config, num_agents
        )

        # Should return empty config
        assert len(preprocessor_configs) == 0


class TestModelCreation:
    """Test model creation functions."""

    def test_create_policy_and_value_models_standard(self):
        """Test create_policy_and_value_models with standard agent."""
        env_cfg = MockEnvConfig()
        agent_cfg = create_mock_agent_config()
        env = MockIsaacLabEnv()
        model_config = {
            'use_hybrid_agent': False,
            'act_init_std': 0.1,
            'actor': {'n': 3, 'latent_size': 256},
            'critic': {'n': 2, 'latent_size': 128},
            'critic_output_init_mean': 0.0,
            'last_layer_scale': 0.01
        }
        wrappers_config = {'hybrid_control': {'enabled': False}}
        derived = {'total_agents': 2}

        models = launch_utils.create_policy_and_value_models(
            env_cfg, agent_cfg, env, model_config, wrappers_config, derived
        )

        assert 'policy' in models
        assert 'value' in models
        assert models['policy'].num_agents == 2
        assert models['value'].num_agents == 2

    def test_create_policy_and_value_models_hybrid(self):
        """Test create_policy_and_value_models with hybrid agent."""
        env_cfg = MockEnvConfig()
        agent_cfg = create_mock_agent_config()
        env = MockIsaacLabEnv()
        model_config = {
            'use_hybrid_agent': True,
            'actor': {'n': 3, 'latent_size': 256},
            'critic': {'n': 2, 'latent_size': 128},
            'critic_output_init_mean': 0.0
        }
        wrappers_config = {'hybrid_control': {'enabled': True}}
        derived = {'total_agents': 1}

        models = launch_utils.create_policy_and_value_models(
            env_cfg, agent_cfg, env, model_config, wrappers_config, derived
        )

        assert 'policy' in models
        assert 'value' in models
        assert models['policy'].num_agents == 1

    def test_create_standard_policy_model_with_hybrid_control(self):
        """Test _create_standard_policy_model with hybrid control."""
        env = MockIsaacLabEnv()
        model_config = {
            'act_init_std': 0.1,
            'actor': {'n': 3, 'latent_size': 256},
            'last_layer_scale': 0.01
        }
        wrappers_config = {
            'hybrid_control': {
                'enabled': True,
                'ctrl_torque': True
            }
        }
        derived = {'total_agents': 1}

        policy = launch_utils._create_standard_policy_model(
            env, model_config, wrappers_config, derived
        )

        # Should have sigma_idx=6 for ctrl_torque=True
        assert hasattr(policy, 'num_agents')

    def test_create_value_model(self):
        """Test _create_value_model function."""
        env = MockIsaacLabEnv()
        model_config = {
            'critic_output_init_mean': 1.0,
            'critic': {'n': 2, 'latent_size': 128}
        }
        agent_cfg = create_mock_agent_config()
        derived = {'total_agents': 3}

        value_model = launch_utils._create_value_model(env, model_config, agent_cfg, derived)

        assert hasattr(value_model, 'critic')  # Check that the model has the expected critic component
        assert value_model.num_agents == 3


class TestAgentCreation:
    """Test agent creation functions."""

    def test_create_block_wandb_agents(self):
        """Test create_block_wandb_agents function."""
        env_cfg = MockEnvConfig()
        agent_cfg = create_mock_agent_config()
        # Add agent_0 config for copying
        agent_cfg['agent']['agent_0'] = {
            'experiment': {
                'project': 'test',
                'tags': ['test_tag'],
                'group': 'test_group'
            }
        }
        env = MockIsaacLabEnv()
        models = {'policy': Mock(), 'value': Mock()}
        memory = Mock()
        derived = {'total_agents': 2}
        learning_config = {
            'policy_learning_rate': 0.001,
            'critic_learning_rate': 0.001,
            'optimizer': {
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'weight_decay': 0.0
            }
        }

        # Should raise ImportError since BlockWandbLoggerPPO was removed during refactoring
        with pytest.raises(ImportError, match="BlockWandbLoggerPPO is not available"):
            launch_utils.create_block_wandb_agents(
                env_cfg, agent_cfg, env, models, memory, derived, learning_config
            )

    @patch('learning.launch_utils_v2.BlockPPO')
    @patch('learning.launch_utils_v2.make_agent_optimizer')
    def test_create_block_ppo_agents(self, mock_optimizer, mock_block_ppo):
        """Test create_block_ppo_agents function."""
        env_cfg = MockEnvConfig()
        agent_cfg = create_mock_agent_config()
        # Add agent configs
        agent_cfg['agent']['agent_0'] = {'experiment': {'name': 'agent_0'}}
        agent_cfg['agent']['agent_1'] = {'experiment': {'name': 'agent_1'}}
        env = MockIsaacLabEnv()
        models = {'policy': Mock(), 'value': Mock()}
        memory = Mock()
        derived = {'total_agents': 2}
        learning_config = {
            'state_preprocessor': True,
            'value_preprocessor': True,
            'policy_learning_rate': 0.001,
            'critic_learning_rate': 0.001,
            'optimizer': {
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'weight_decay': 0.0
            }
        }

        # Mock the agent instance
        mock_agent = Mock()
        mock_block_ppo.return_value = mock_agent
        mock_optimizer.return_value = Mock()

        agent = launch_utils.create_block_ppo_agents(
            env_cfg, agent_cfg, env, models, memory, derived
        )

        # Check that BlockPPO was called
        mock_block_ppo.assert_called_once()

        # Check that optimizer was created
        mock_optimizer.assert_called_once()

        # Check that agent exp configs were stored
        assert hasattr(agent, 'agent_exp_cfgs')


class TestHelperFunctions:
    """Test helper functions."""

    def test_set_nested_attr(self):
        """Test _set_nested_attr function."""
        class TestObj:
            def __init__(self):
                self.level1 = TestObj2()

        class TestObj2:
            def __init__(self):
                self.level2 = TestObj3()

        class TestObj3:
            def __init__(self):
                self.value = 'original'

        obj = TestObj()

        launch_utils._set_nested_attr(obj, 'level1.level2.value', 'modified')

        assert obj.level1.level2.value == 'modified'

    def test_set_nested_attr_single_level(self):
        """Test _set_nested_attr with single level."""
        class TestObj:
            def __init__(self):
                self.attr = 'original'

        obj = TestObj()

        launch_utils._set_nested_attr(obj, 'attr', 'modified')

        assert obj.attr == 'modified'

    def test_setup_individual_agent_logging(self):
        """Test _setup_individual_agent_logging function."""
        agent_cfg = create_mock_agent_config()
        resolved_config = {
            'primary': {
                'break_forces': [50.0, 100.0],
                'agents_per_break_force': 2
            },
            'derived': {'total_agents': 4},
            'experiment': {
                'name': 'test_experiment',
                'wandb_entity': 'test_user'
            }
        }

        launch_utils._setup_individual_agent_logging(agent_cfg, resolved_config)

        # Check that agent configs were created
        assert 'agent_0' in agent_cfg['agent']
        assert 'agent_1' in agent_cfg['agent']
        assert 'agent_2' in agent_cfg['agent']
        assert 'agent_3' in agent_cfg['agent']

        # Check break forces assignment
        assert agent_cfg['agent']['agent_0']['break_force'] == 50.0
        assert agent_cfg['agent']['agent_1']['break_force'] == 50.0
        assert agent_cfg['agent']['agent_2']['break_force'] == 100.0
        assert agent_cfg['agent']['agent_3']['break_force'] == 100.0

        # Check experiment configs
        for i in range(4):
            assert 'experiment' in agent_cfg['agent'][f'agent_{i}']
            assert 'directory' in agent_cfg['agent'][f'agent_{i}']['experiment']
            assert 'wandb_kwargs' in agent_cfg['agent'][f'agent_{i}']['experiment']

        # Check agent_specific_configs was created
        assert 'agent_specific_configs' in resolved_config
        assert len(resolved_config['agent_specific_configs']) == 4

    def test_setup_individual_agent_logging_single_break_force(self):
        """Test _setup_individual_agent_logging with single break force."""
        agent_cfg = create_mock_agent_config()
        resolved_config = {
            'primary': {
                'break_forces': 75.0,  # Single value, not list
                'agents_per_break_force': 1
            },
            'derived': {'total_agents': 1},
            'experiment': {
                'name': 'single_test'
            }
        }

        launch_utils._setup_individual_agent_logging(agent_cfg, resolved_config)

        # Should handle single value by converting to list
        assert 'agent_0' in agent_cfg['agent']
        assert agent_cfg['agent']['agent_0']['break_force'] == 75.0

    def test_configure_hybrid_agent_parameters(self):
        """Test _configure_hybrid_agent_parameters function."""
        env_cfg = MockEnvConfig()
        agent_cfg = create_mock_agent_config()
        agent_cfg['agent']['hybrid_agent']['unit_std_init'] = True
        model_config = {}
        wrappers_config = {'hybrid_control': {'ctrl_torque': False}}

        launch_utils._configure_hybrid_agent_parameters(env_cfg, agent_cfg, model_config, wrappers_config)

        # Check that parameters were calculated and set
        assert 'pos_init_std' in agent_cfg['agent']['hybrid_agent']
        assert 'rot_init_std' in agent_cfg['agent']['hybrid_agent']
        assert 'force_init_std' in agent_cfg['agent']['hybrid_agent']
        assert 'pos_scale' in agent_cfg['agent']['hybrid_agent']
        assert 'rot_scale' in agent_cfg['agent']['hybrid_agent']
        assert 'force_scale' in agent_cfg['agent']['hybrid_agent']
        assert 'torque_scale' in agent_cfg['agent']['hybrid_agent']

        # Values should be calculated based on environment control gains
        assert agent_cfg['agent']['hybrid_agent']['pos_scale'] > 0
        assert agent_cfg['agent']['hybrid_agent']['force_scale'] > 0

    def test_configure_hybrid_agent_parameters_no_unit_std(self):
        """Test _configure_hybrid_agent_parameters without unit_std_init."""
        env_cfg = MockEnvConfig()
        agent_cfg = create_mock_agent_config()
        agent_cfg['agent']['hybrid_agent']['unit_std_init'] = False
        model_config = {}
        wrappers_config = {'hybrid_control': {'ctrl_torque': True}}

        original_pos_init_std = agent_cfg['agent']['hybrid_agent']['pos_init_std']

        launch_utils._configure_hybrid_agent_parameters(env_cfg, agent_cfg, model_config, wrappers_config)

        # Should not modify init_std values when unit_std_init is False
        assert agent_cfg['agent']['hybrid_agent']['pos_init_std'] == original_pos_init_std

        # But should still set scale values
        assert 'pos_scale' in agent_cfg['agent']['hybrid_agent']
        assert 'force_scale' in agent_cfg['agent']['hybrid_agent']


class TestExperimentTagMerging:
    """Test experiment tag merging in setup_experiment_logging."""

    def test_setup_experiment_logging_tag_merging_no_cli(self):
        """Test tag merging when no CLI tags provided."""
        agent_cfg = create_mock_agent_config()
        env_cfg = MockEnvConfig()
        resolved_config = {
            'primary': {'break_forces': [50.0, 75.0], 'agents_per_break_force': 2},
            'derived': {'total_agents': 4, 'total_num_envs': 512},
            'experiment': {
                'name': 'test_exp',
                'tags': ['baseline', 'factory'],  # Base + experiment merged tags
                'group': 'test_group',
                'wandb_project': 'test_project',
                'wandb_entity': 'test_entity'
            }
        }

        # No config_bundle (no CLI tags)
        launch_utils.setup_experiment_logging(env_cfg, agent_cfg, resolved_config, config_bundle=None)

        # Should use experiment tags + task name
        assert agent_cfg['agent']['experiment']['tags'] == ['baseline', 'factory', 'MockTask']

    def test_setup_experiment_logging_tag_merging_with_cli(self):
        """Test tag merging with CLI tags."""
        from configs.config_manager_v2 import ConfigBundle
        from configs.cfg_exts.primary_cfg import PrimaryConfig

        agent_cfg = create_mock_agent_config()
        env_cfg = MockEnvConfig()
        resolved_config = {
            'primary': {'break_forces': [50.0], 'agents_per_break_force': 1},
            'derived': {'total_agents': 1, 'total_num_envs': 256},
            'experiment': {
                'name': 'test_exp',
                'tags': ['baseline', 'factory'],  # Base + experiment merged tags
                'group': 'test_group',
                'wandb_project': 'test_project',
                'wandb_entity': 'test_entity'
            }
        }

        # Mock config bundle with CLI tags
        config_bundle = ConfigBundle(
            env_cfg=None, agent_cfg=None, primary_cfg=PrimaryConfig(),
            model_cfg=None, wrapper_cfg=None, task_name='test'
        )
        config_bundle._cli_experiment_tags = ['debug', 'v2']

        launch_utils.setup_experiment_logging(env_cfg, agent_cfg, resolved_config, config_bundle)

        # Should merge experiment + CLI tags + task name (no duplicates)
        expected_tags = ['baseline', 'factory', 'debug', 'v2', 'MockTask']
        assert agent_cfg['agent']['experiment']['tags'] == expected_tags

    def test_setup_experiment_logging_tag_merging_with_cli_duplicates(self):
        """Test tag merging with CLI tags that duplicate existing ones."""
        from configs.config_manager_v2 import ConfigBundle
        from configs.cfg_exts.primary_cfg import PrimaryConfig

        agent_cfg = create_mock_agent_config()
        env_cfg = MockEnvConfig()
        resolved_config = {
            'primary': {'break_forces': [50.0], 'agents_per_break_force': 1},
            'derived': {'total_agents': 1, 'total_num_envs': 256},
            'experiment': {
                'name': 'test_exp',
                'tags': ['baseline', 'factory', 'common'],
                'group': 'test_group'
            }
        }

        # Mock config bundle with overlapping CLI tags
        config_bundle = ConfigBundle(
            env_cfg=None, agent_cfg=None, primary_cfg=PrimaryConfig(),
            model_cfg=None, wrapper_cfg=None, task_name='test'
        )
        config_bundle._cli_experiment_tags = ['debug', 'factory', 'common']

        launch_utils.setup_experiment_logging(env_cfg, agent_cfg, resolved_config, config_bundle)

        # Should preserve order, no duplicates + task name
        expected_tags = ['baseline', 'factory', 'common', 'debug', 'MockTask']
        assert agent_cfg['agent']['experiment']['tags'] == expected_tags


if __name__ == "__main__":
    pytest.main([__file__, "-v"])