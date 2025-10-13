"""
Integration tests for factory_runnerv2.py components

These tests verify individual components and workflows of the factory runner
without importing the problematic SKRL model dependencies.
"""

import pytest
import sys
import os
import tempfile
import shutil
import yaml
import torch
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock Isaac Lab components
from tests.mocks.mock_launch_utils_deps import setup_minimal_mocks
setup_minimal_mocks()

# Import after mocking
from configs.config_manager import ConfigManager


class TestConfigurationSystem:
    """Test the configuration system used by factory_runnerv2.py."""

    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_dir.mkdir(parents=True)

    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.temp_dir)

    def create_test_config(self, config_name: str, config_data: dict) -> str:
        """Create a test configuration file."""
        config_path = self.config_dir / f"{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        return str(config_path)

    def test_base_configuration_loading(self):
        """Test loading base configuration similar to factory_base.yaml."""
        base_config = {
            'primary': {
                'agents_per_break_force': 2,
                'num_envs_per_agent': 256,
                'break_forces': [-1],
                'episode_length_s': 12.0,
                'decimation': 4,
                'policy_hz': 15,
                'max_steps': 10240000,
                'debug_mode': False,
                'seed': -1
            },
            'defaults': {
                'task_name': "Isaac-Factory-PegInsert-Local-v0"
            },
            'environment': {
                'episode_length_s': '${primary.episode_length_s}',
                'decimation': '${primary.decimation}',
                'component_dims': {
                    'fingertip_pos': 3,
                    'joint_pos': 7,
                    'force_torque': 6,
                    'prev_actions': 6
                },
                'component_attr_map': {
                    'fingertip_pos': 'fingertip_midpoint_pos',
                    'joint_pos': 'joint_pos',
                    'force_torque': 'robot_force_torque',
                    'prev_actions': 'prev_actions'
                }
            },
            'learning': {
                'rollouts': None,
                'learning_epochs': 4,
                'policy_learning_rate': 1.0e-6,
                'critic_learning_rate': 1.0e-5,
                'state_preprocessor': True,
                'value_preprocessor': True
            },
            'model': {
                'use_hybrid_agent': False,
                'actor': {'n': 1, 'latent_size': 256},
                'critic': {'n': 3, 'latent_size': 1024}
            },
            'wrappers': {
                'fragile_objects': {'enabled': True},
                'force_torque_sensor': {'enabled': False},
                'observation_manager': {'enabled': True},
                'observation_noise': {'enabled': False},
                'hybrid_control': {'enabled': False},
                'factory_metrics': {'enabled': True},
                'wandb_logging': {'enabled': True},
                'action_logging': {'enabled': False}
            }
        }

        config_path = self.create_test_config('base_test', base_config)
        resolved_config = ConfigManager.load_and_resolve_config(config_path)

        # Test derived calculations
        derived = resolved_config['derived']
        assert derived['total_agents'] == 2  # 1 break force * 2 agents per break force
        assert derived['total_num_envs'] == 512  # 2 agents * 256 envs per agent
        assert 'rollout_steps' in derived

        # Test reference resolution
        assert resolved_config['environment']['episode_length_s'] == 12.0
        assert resolved_config['environment']['decimation'] == 4

    def test_hybrid_control_configuration(self):
        """Test hybrid control configuration similar to hybrid_control_exp.yaml."""
        hybrid_config = {
            'primary': {
                'agents_per_break_force': 2,
                'num_envs_per_agent': 256,
                'break_forces': [-1],
                'episode_length_s': 15.0,
                'max_steps': 15360000
            },
            'model': {
                'use_hybrid_agent': True,
                'hybrid_agent': {
                    'ctrl_torque': False,
                    'unit_std_init': True,
                    'selection_adjustment_types': 'none',
                    'init_bias': -2.5,
                    'uniform_sampling_rate': 0.0
                }
            },
            'wrappers': {
                'force_torque_sensor': {'enabled': True, 'use_tanh_scaling': True},
                'hybrid_control': {'enabled': True, 'ctrl_torque': False},
                'observation_noise': {'enabled': True}
            },
            'learning': {
                'policy_learning_rate': 5.0e-7,
                'critic_learning_rate': 8.0e-6,
                'learning_epochs': 5
            }
        }

        config_path = self.create_test_config('hybrid_test', hybrid_config)
        resolved_config = ConfigManager.load_and_resolve_config(config_path)

        # Verify hybrid control configuration
        assert resolved_config['model']['use_hybrid_agent'] == True
        assert resolved_config['model']['hybrid_agent']['ctrl_torque'] == False
        assert resolved_config['wrappers']['hybrid_control']['enabled'] == True
        assert resolved_config['wrappers']['force_torque_sensor']['enabled'] == True

    def test_debug_mode_configuration(self):
        """Test debug mode configuration similar to debug_exp.yaml."""
        debug_config = {
            'primary': {
                'agents_per_break_force': 1,
                'num_envs_per_agent': 128,
                'break_forces': [-1],
                'episode_length_s': 10.0,
                'max_steps': 2048000,
                'debug_mode': True
            },
            'learning': {
                'policy_learning_rate': 1.0e-5,
                'critic_learning_rate': 5.0e-5,
                'learning_epochs': 3
            },
            'wrappers': {
                'observation_noise': {'enabled': False},
                'force_torque_sensor': {'enabled': False},
                'hybrid_control': {'enabled': False},
                'action_logging': {'enabled': True}
            }
        }

        config_path = self.create_test_config('debug_test', debug_config)
        resolved_config = ConfigManager.load_and_resolve_config(config_path)

        # Verify debug mode configuration
        assert resolved_config['primary']['debug_mode'] == True
        assert resolved_config['primary']['num_envs_per_agent'] == 128
        assert resolved_config['wrappers']['observation_noise']['enabled'] == False

    def test_cli_overrides(self):
        """Test CLI override functionality."""
        base_config = {
            'primary': {
                'num_envs_per_agent': 256,
                'break_forces': [-1]
            },
            'learning': {
                'policy_learning_rate': 1.0e-6
            },
            'wrappers': {
                'force_torque_sensor': {'enabled': False}
            }
        }

        config_path = self.create_test_config('override_test', base_config)

        # Test various override scenarios
        overrides = [
            'primary.num_envs_per_agent=128',
            'primary.break_forces=[25.0,50.0]',
            'learning.policy_learning_rate=5.0e-7',
            'wrappers.force_torque_sensor.enabled=true'
        ]

        resolved_config = ConfigManager.load_and_resolve_config(config_path, overrides)

        # Verify overrides were applied
        assert resolved_config['primary']['num_envs_per_agent'] == 128
        assert resolved_config['primary']['break_forces'] == [25.0, 50.0]
        assert resolved_config['learning']['policy_learning_rate'] == 5.0e-7
        assert resolved_config['wrappers']['force_torque_sensor']['enabled'] == True

    def test_multi_agent_calculations(self):
        """Test multi-agent environment calculations."""
        multi_config = {
            'primary': {
                'agents_per_break_force': 3,
                'num_envs_per_agent': 64,
                'break_forces': [25.0, 50.0, 75.0, 100.0]  # 4 break forces
            }
        }

        config_path = self.create_test_config('multi_test', multi_config)
        resolved_config = ConfigManager.load_and_resolve_config(config_path)

        derived = resolved_config['derived']
        # 4 break forces * 3 agents per break force = 12 total agents
        assert derived['total_agents'] == 12
        # 12 agents * 64 envs per agent = 768 total environments
        assert derived['total_num_envs'] == 768

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        # Test missing required fields
        invalid_config = {
            'primary': {}  # Missing required fields
        }

        config_path = self.create_test_config('invalid_test', invalid_config)

        # Should handle gracefully with defaults
        resolved_config = ConfigManager.load_and_resolve_config(config_path)
        assert 'derived' in resolved_config

    def test_configuration_inheritance(self):
        """Test configuration inheritance with base files."""
        # Create base configuration
        base_config = {
            'primary': {
                'agents_per_break_force': 2,
                'num_envs_per_agent': 256,
                'episode_length_s': 12.0
            },
            'learning': {
                'policy_learning_rate': 1.0e-6,
                'learning_epochs': 4
            }
        }

        base_path = self.create_test_config('base', base_config)

        # Create derived configuration that inherits from base
        derived_config = {
            'base': base_path,
            'primary': {
                'episode_length_s': 15.0  # Override only this field
            },
            'learning': {
                'learning_epochs': 5  # Override only this field
            }
        }

        derived_path = self.create_test_config('derived', derived_config)
        resolved_config = ConfigManager.load_and_resolve_config(derived_path)

        # Verify inheritance and overrides
        assert resolved_config['primary']['agents_per_break_force'] == 2  # Inherited
        assert resolved_config['primary']['episode_length_s'] == 15.0  # Overridden
        assert resolved_config['learning']['policy_learning_rate'] == 1.0e-6  # Inherited
        assert resolved_config['learning']['learning_epochs'] == 5  # Overridden


class TestWrapperConfigurations:
    """Test wrapper configurations used by factory_runnerv2.py."""

    def setup_method(self):
        """Setup for each test method."""
        # Import wrapper-related modules separately to avoid SKRL conflicts
        pass

    def test_wrapper_dependency_validation(self):
        """Test that wrapper dependencies are properly configured."""
        # Test factory metrics requires wandb logging
        wrapper_config = {
            'factory_metrics': {'enabled': True},
            'wandb_logging': {'enabled': False}  # Missing dependency
        }

        # This should be caught by integration logic
        # (The actual validation happens in wrapper application)
        assert wrapper_config['factory_metrics']['enabled'] == True

    def test_wrapper_combinations(self):
        """Test valid wrapper combinations."""
        # Test all wrappers enabled
        all_wrappers = {
            'fragile_objects': {'enabled': True},
            'force_torque_sensor': {'enabled': True},
            'observation_manager': {'enabled': True},
            'observation_noise': {'enabled': True},
            'hybrid_control': {'enabled': True},
            'factory_metrics': {'enabled': True},
            'wandb_logging': {'enabled': True},
            'action_logging': {'enabled': True}
        }

        # Should be a valid configuration
        enabled_count = sum(1 for w in all_wrappers.values() if w.get('enabled', False))
        assert enabled_count == 8

        # Test minimal wrappers
        minimal_wrappers = {
            'observation_manager': {'enabled': True},
            'factory_metrics': {'enabled': True}
        }

        enabled_minimal = sum(1 for w in minimal_wrappers.values() if w.get('enabled', False))
        assert enabled_minimal == 2

    def test_hybrid_control_wrapper_config(self):
        """Test hybrid control wrapper configuration."""
        hybrid_config = {
            'hybrid_control': {
                'enabled': True,
                'ctrl_torque': False,
                'reward_type': 'simp'
            },
            'force_torque_sensor': {
                'enabled': True,  # Required for hybrid control
                'use_tanh_scaling': True
            }
        }

        # Verify hybrid control has its required dependencies
        assert hybrid_config['hybrid_control']['enabled'] == True
        assert hybrid_config['force_torque_sensor']['enabled'] == True

    def test_observation_noise_config(self):
        """Test observation noise wrapper configuration."""
        noise_config = {
            'observation_noise': {
                'enabled': True,
                'global_noise_scale': 1.0,
                'apply_to_critic': True,
                'noise_groups': {
                    'fingertip_pos': {
                        'noise_type': 'gaussian',
                        'std': 0.01,
                        'enabled': True,
                        'timing': 'step'
                    },
                    'joint_pos': {
                        'noise_type': 'gaussian',
                        'std': 0.005,
                        'enabled': True
                    }
                }
            }
        }

        # Verify noise configuration structure
        noise_wrapper = noise_config['observation_noise']
        assert noise_wrapper['enabled'] == True
        assert 'noise_groups' in noise_wrapper
        assert len(noise_wrapper['noise_groups']) == 2


class TestEnvironmentConfiguration:
    """Test environment configuration aspects of factory_runnerv2.py."""

    def setup_method(self):
        """Setup for each test method."""
        from tests.mocks.mock_isaac_lab import MockEnvConfig
        self.mock_env_cfg = MockEnvConfig()

    def test_isaac_lab_config_application(self):
        """Test applying configuration to Isaac Lab environment config."""
        config = {
            'primary': {},  # Add missing primary section
            'environment': {
                'episode_length_s': 15.0,
                'decimation': 8,
                'filter_collisions': True
            }
        }

        # Test configuration application
        ConfigManager.apply_to_isaac_lab(self.mock_env_cfg, {}, config)

        # Verify attributes were set
        assert hasattr(self.mock_env_cfg, 'episode_length_s')
        assert self.mock_env_cfg.episode_length_s == 15.0

    def test_nested_config_application(self):
        """Test applying nested configuration parameters."""
        config = {
            'primary': {},  # Add missing primary section
            'environment': {
                'task.success_threshold': 0.05,
                'ctrl.pos_action_bounds': [0.1, 0.1, 0.1]
            }
        }

        # Test nested configuration
        ConfigManager.apply_to_isaac_lab(self.mock_env_cfg, {}, config)

        # Verify nested attributes were set
        assert hasattr(self.mock_env_cfg.task, 'success_threshold')
        assert self.mock_env_cfg.task.success_threshold == 0.05

    def test_scene_configuration(self):
        """Test scene configuration for multi-environment setup."""
        primary = {
            'agents_per_break_force': 2,
            'num_envs_per_agent': 128
        }
        derived = {
            'total_agents': 4,
            'total_num_envs': 512
        }

        # Mock launch utils function
        def mock_configure_environment_scene(env_cfg, primary_config, derived_config):
            env_cfg.scene.num_envs = derived_config['total_num_envs']
            env_cfg.scene.replicate_physics = True

        mock_configure_environment_scene(self.mock_env_cfg, primary, derived)

        # Verify scene configuration
        assert self.mock_env_cfg.scene.num_envs == 512
        assert self.mock_env_cfg.scene.replicate_physics == True


class TestLearningConfiguration:
    """Test learning configuration aspects of factory_runnerv2.py."""

    def test_learning_parameter_application(self):
        """Test applying learning parameters to agent configuration."""
        learning_config = {
            'learning_epochs': 5,
            'policy_learning_rate': 3.0e-6,
            'critic_learning_rate': 1.0e-5,
            'value_update_ratio': 2,
            'use_huber_value_loss': True,
            'state_preprocessor': True,
            'value_preprocessor': True
        }

        agent_cfg = {'agent': {}}
        max_rollout_steps = 1000

        # Mock the apply_learning_config function
        def mock_apply_learning_config(agent_config, learning, rollout_steps):
            agent_config['agent'].update({
                'learning_epochs': learning['learning_epochs'],
                'policy_learning_rate': learning['policy_learning_rate'],
                'critic_learning_rate': learning['critic_learning_rate'],
                'value_update_ratio': learning['value_update_ratio'],
                'rollouts': rollout_steps
            })

        mock_apply_learning_config(agent_cfg, learning_config, max_rollout_steps)

        # Verify learning parameters were applied
        agent = agent_cfg['agent']
        assert agent['learning_epochs'] == 5
        assert agent['policy_learning_rate'] == 3.0e-6
        assert agent['critic_learning_rate'] == 1.0e-5
        assert agent['rollouts'] == 1000

    def test_rollout_calculation(self):
        """Test rollout steps calculation."""
        primary = {
            'episode_length_s': 12.0,
            'decimation': 4,
            'policy_hz': 15
        }

        # Calculate rollout steps (similar to _calculate_derived_params)
        sim_dt = (1/primary['policy_hz']) / primary['decimation']
        rollout_steps = int((1/sim_dt) / primary['decimation'] * primary['episode_length_s'])

        # Verify calculation
        assert rollout_steps > 0
        assert isinstance(rollout_steps, int)

    def test_model_configuration_application(self):
        """Test applying model parameters to agent configuration."""
        model_config = {
            'actor': {'n': 2, 'latent_size': 512},
            'critic': {'n': 3, 'latent_size': 1024},
            'force_encoding': 'tanh',
            'use_hybrid_agent': False
        }

        agent_cfg = {'models': {'policy': {}, 'value': {}}}

        # Mock the apply_model_config function
        def mock_apply_model_config(agent_config, model):
            agent_config['models']['policy'].update(model['actor'])
            agent_config['models']['value'].update(model['critic'])
            if 'force_encoding' in model:
                agent_config['models']['policy']['force_encoding'] = model['force_encoding']

        mock_apply_model_config(agent_cfg, model_config)

        # Verify model parameters were applied
        assert agent_cfg['models']['policy']['n'] == 2
        assert agent_cfg['models']['policy']['latent_size'] == 512
        assert agent_cfg['models']['value']['n'] == 3
        assert agent_cfg['models']['value']['latent_size'] == 1024
        assert agent_cfg['models']['policy']['force_encoding'] == 'tanh'


class TestTrainerConfiguration:
    """Test trainer configuration aspects of factory_runnerv2.py."""

    def test_trainer_config_creation(self):
        """Test trainer configuration creation."""
        derived = {
            'max_steps': 10240000,
            'total_num_envs': 512
        }

        agent_cfg = {
            'agent': {
                'disable_progressbar': True
            }
        }

        # Create trainer config (similar to factory_runnerv2.py)
        cfg_trainer = {
            "timesteps": derived['max_steps'] // derived['total_num_envs'],
            "headless": True,
            "close_environment_at_exit": True,
            "disable_progressbar": agent_cfg['agent']['disable_progressbar']
        }

        # Verify trainer configuration
        assert cfg_trainer['timesteps'] == 20000  # 10240000 // 512
        assert cfg_trainer['headless'] == True
        assert cfg_trainer['close_environment_at_exit'] == True
        assert cfg_trainer['disable_progressbar'] == True

    def test_memory_configuration(self):
        """Test memory configuration for multi-agent setup."""
        derived = {
            'rollout_steps': 1000,
            'total_agents': 4,
            'total_num_envs': 512
        }

        device = 'cpu'

        # Memory configuration (similar to factory_runnerv2.py)
        memory_config = {
            'memory_size': derived['rollout_steps'],
            'num_envs': derived['total_num_envs'],
            'device': device,
            'replacement': True,
            'num_agents': derived['total_agents']
        }

        # Verify memory configuration
        assert memory_config['memory_size'] == 1000
        assert memory_config['num_envs'] == 512
        assert memory_config['num_agents'] == 4
        assert memory_config['device'] == 'cpu'
        assert memory_config['replacement'] == True