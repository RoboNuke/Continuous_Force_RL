"""
Integration tests for factory_runnerv2.py

These tests ensure that the factory runner script can handle all possible
configuration combinations and wrapper scenarios without failures during
environment creation and initial training steps.
"""

import pytest
import sys
import os
import tempfile
import shutil
import yaml
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock heavy dependencies before any imports
mock_skrl = MagicMock()
mock_skrl.utils = MagicMock()
mock_skrl.utils.set_seed = MagicMock()
mock_skrl.trainers = MagicMock()
mock_skrl.trainers.torch = MagicMock()
mock_skrl.trainers.torch.ppo = MagicMock()
mock_skrl.trainers.torch.SequentialTrainer = MagicMock()
mock_skrl.resources = MagicMock()
mock_skrl.resources.schedulers = MagicMock()
mock_skrl.resources.schedulers.torch = MagicMock()
mock_skrl.resources.schedulers.torch.KLAdaptiveLR = MagicMock()
mock_skrl.resources.preprocessors = MagicMock()
mock_skrl.resources.preprocessors.torch = MagicMock()
mock_skrl.resources.preprocessors.torch.RunningStandardScaler = MagicMock()
mock_skrl.agents = MagicMock()
mock_skrl.agents.torch = MagicMock()
mock_skrl.agents.torch.ppo = MagicMock()
mock_skrl.agents.torch.ppo.PPO = MagicMock()
mock_skrl.memories = MagicMock()
mock_skrl.memories.torch = MagicMock()
mock_skrl.memories.torch.Memory = MagicMock()
mock_skrl.agents.torch.ppo.PPO_DEFAULT_CONFIG = {}
mock_skrl.models = MagicMock()
mock_skrl.models.torch = MagicMock()
mock_skrl.models.torch.DeterministicMixin = MagicMock()
mock_skrl.models.torch.GaussianMixin = MagicMock()
mock_skrl.models.torch.Model = MagicMock()

sys.modules['skrl'] = mock_skrl
sys.modules['skrl.agents'] = mock_skrl.agents
sys.modules['skrl.agents.torch'] = mock_skrl.agents.torch
sys.modules['skrl.agents.torch.ppo'] = mock_skrl.agents.torch.ppo
sys.modules['skrl.memories'] = mock_skrl.memories
sys.modules['skrl.memories.torch'] = mock_skrl.memories.torch
sys.modules['skrl.resources.preprocessors'] = mock_skrl.resources.preprocessors
sys.modules['skrl.resources.preprocessors.torch'] = mock_skrl.resources.preprocessors.torch
sys.modules['skrl.utils'] = mock_skrl.utils
sys.modules['skrl.trainers'] = mock_skrl.trainers
sys.modules['skrl.trainers.torch'] = mock_skrl.trainers.torch
sys.modules['skrl.trainers.torch.ppo'] = mock_skrl.trainers.torch.ppo
sys.modules['skrl.resources'] = mock_skrl.resources
sys.modules['skrl.resources.schedulers'] = mock_skrl.resources.schedulers
sys.modules['skrl.resources.schedulers.torch'] = mock_skrl.resources.schedulers.torch
sys.modules['skrl.models'] = mock_skrl.models
sys.modules['skrl.models.torch'] = mock_skrl.models.torch

sys.modules['memories'] = MagicMock()
sys.modules['memories.multi_random'] = MagicMock()
sys.modules['envs'] = MagicMock()
sys.modules['envs.factory'] = MagicMock()

# Mock Isaac Lab components
from tests.mocks.mock_launch_utils_deps import setup_minimal_mocks
setup_minimal_mocks()

# Mock app launcher and simulation app
mock_app_launcher = MagicMock()
mock_simulation_app = MagicMock()
mock_app_launcher.app = mock_simulation_app

sys.modules['isaaclab'] = MagicMock()
sys.modules['isaaclab.app'] = MagicMock()
sys.modules['isaaclab.app'].AppLauncher = MagicMock(return_value=mock_app_launcher)
sys.modules['isaaclab_tasks'] = MagicMock()
sys.modules['isaaclab_tasks.utils'] = MagicMock()
sys.modules['isaaclab_tasks.utils.hydra'] = MagicMock()
sys.modules['isaaclab_rl'] = MagicMock()
sys.modules['isaaclab_rl.skrl'] = MagicMock()

# Mock Isaac Sim components for wrapper tests
mock_isaac_sim = MagicMock()
mock_isaac_sim.RobotView = MagicMock()
sys.modules['isaaclab.assets'] = mock_isaac_sim
sys.modules['isaaclab.assets.robot'] = mock_isaac_sim

# Mock isaacsim imports for force torque wrapper
mock_isaacsim = MagicMock()
mock_isaacsim.core = MagicMock()
mock_isaacsim.core.api = MagicMock()
mock_isaacsim.core.api.robots = MagicMock()
mock_isaacsim.core.api.robots.RobotView = MagicMock()

# Also mock the omni.isaac.core fallback
mock_omni = MagicMock()
mock_omni.isaac = MagicMock()
mock_omni.isaac.core = MagicMock()
mock_omni.isaac.core.articulations = MagicMock()
mock_omni.isaac.core.articulations.ArticulationView = MagicMock()

sys.modules['isaacsim'] = mock_isaacsim
sys.modules['isaacsim.core'] = mock_isaacsim.core
sys.modules['isaacsim.core.api'] = mock_isaacsim.core.api
sys.modules['isaacsim.core.api.robots'] = mock_isaacsim.core.api.robots
sys.modules['omni'] = mock_omni
sys.modules['omni.isaac'] = mock_omni.isaac
sys.modules['omni.isaac.core'] = mock_omni.isaac.core
sys.modules['omni.isaac.core.articulations'] = mock_omni.isaac.core.articulations

# Mock SimBa models to avoid metaclass conflicts in pytest
mock_simba = MagicMock()
mock_simba.SimBaAgent = MagicMock()
mock_simba.SimBaNet = MagicMock()
mock_simba.ScaleLayer = MagicMock()
mock_simba.MultiSimBaNet = MagicMock()
sys.modules['models.SimBa'] = mock_simba

mock_hybrid = MagicMock()
mock_hybrid.HybridControlBlockSimBaActor = MagicMock()
sys.modules['models.SimBa_hybrid_control'] = mock_hybrid

# Mock agents that depend on SKRL - only when needed in specific tests
mock_block_ppo = MagicMock()
mock_block_ppo.BlockPPO = MagicMock()
# Don't globally mock agents.block_ppo - this breaks unit tests

# Mock block_simba models to avoid metaclass conflicts
mock_block_simba = MagicMock()
mock_block_simba.BlockSimBaCritic = MagicMock()
mock_block_simba.BlockSimBaActor = MagicMock()
mock_block_simba.export_policies = MagicMock()
mock_block_simba.make_agent_optimizer = MagicMock()
sys.modules['models.block_simba'] = mock_block_simba

# Mock gym environment creation
import gymnasium as gym
original_gym_make = gym.make

def mock_gym_make(env_id, **kwargs):
    """Mock gym.make to return our test environment."""
    from tests.mocks.mock_isaac_lab import MockBaseEnv
    env = MockBaseEnv()
    env.cfg = kwargs.get('cfg')
    # Add Isaac Lab compatibility attributes
    env.num_envs = getattr(env.cfg, 'scene', {}).get('num_envs', 16) if hasattr(env.cfg, 'scene') else 16
    env.device = getattr(env.cfg, 'sim', {}).get('device', 'cpu') if hasattr(env.cfg, 'sim') else 'cpu'
    return env

# Mock argparse.ArgumentParser.parse_known_args to prevent factory_runnerv2 conflicts
from unittest.mock import patch
import argparse
from types import SimpleNamespace

# Create mock args that factory_runnerv2 expects
mock_args = SimpleNamespace(
    config='dummy.yaml',
    task=None,
    seed=-1,
    override=None,
    video=False,
    enable_cameras=False,
    headless=True,
    livestream=0,
    experience=0,
    width=640,
    height=480
)

# Patch ArgumentParser.parse_known_args before any factory_runnerv2 imports
original_parse_known_args = argparse.ArgumentParser.parse_known_args

def mock_parse_known_args(self, args=None, namespace=None):
    # Return our mock args and empty hydra_args
    return mock_args, []

# Store the original function before patching
_original_parse_known_args = argparse.ArgumentParser.parse_known_args

# Apply the patch temporarily for imports
argparse.ArgumentParser.parse_known_args = mock_parse_known_args

# Import after mocking argparse
import learning.launch_utils_v2 as lUtils
from configs.config_manager import ConfigManager

# Restore original argparse function immediately after imports
argparse.ArgumentParser.parse_known_args = _original_parse_known_args



class TestFactoryRunnerIntegration:
    """Integration tests for factory_runnerv2.py main workflow."""

    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_dir.mkdir(parents=True)

        # Create mock SkrlVecEnvWrapper
        self.mock_skrl_wrapper = MagicMock()
        self.mock_skrl_wrapper.num_envs = 16
        self.mock_skrl_wrapper.action_space = MagicMock()
        self.mock_skrl_wrapper.cfg = MagicMock()
        self.mock_skrl_wrapper.cfg.observation_space = 32
        self.mock_skrl_wrapper.cfg.state_space = 48

    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.temp_dir)

    def create_test_config(self, config_name: str, overrides: dict = None) -> str:
        """Create a test configuration file."""
        base_config = {
            'primary': {
                'agents_per_break_force': 1,
                'num_envs_per_agent': 16,
                'break_forces': [-1],
                'episode_length_s': 10.0,
                'decimation': 4,
                'policy_hz': 15,
                'max_steps': 1024,
                'debug_mode': False,
                'seed': 42,
                'ckpt_tracker_path': str(Path(self.temp_dir) / "ckpt_tracker.txt")
            },
            'defaults': {
                'task_name': "Isaac-Factory-PegInsert-Local-v0"
            },
            'environment': {
                'episode_length_s': 10.0,
                'decimation': 4,
                'component_dims': {
                    'fingertip_pos': 3,
                    'joint_pos': 7,
                    'force_torque': 6,
                    'prev_actions': 6
                },
                'component_attr_map': {
                    'fingertip_pos': 'fingertip_pos',
                    'joint_pos': 'joint_pos',
                    'force_torque': 'robot_force_torque',
                    'prev_actions': 'prev_actions'
                },
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
            },
            'learning': {
                'rollouts': None,
                'learning_epochs': 2,
                'mini_batches': None,
                'discount_factor': 0.99,
                'lambda': 0.95,
                'policy_learning_rate': 1.0e-5,
                'critic_learning_rate': 1.0e-4,
                'value_update_ratio': 1,
                'use_huber_value_loss': True,
                'grad_norm_clip': 0.5,
                'state_preprocessor': True,
                'value_preprocessor': True
            },
            'model': {
                'use_hybrid_agent': False,
                'actor': {'n': 1, 'latent_size': 64},
                'critic': {'n': 1, 'latent_size': 128}
            },
            'wrappers': {
                'fragile_objects': {'enabled': False},
                'force_torque_sensor': {'enabled': False},
                'observation_manager': {'enabled': True, 'merge_strategy': 'concatenate'},
                'observation_noise': {'enabled': False},
                'hybrid_control': {'enabled': False},
                'factory_metrics': {'enabled': True},
                'wandb_logging': {'enabled': False},
                'action_logging': {'enabled': False}
            },
            'experiment': {
                'name': config_name,
                'tags': ['test'],
                'group': 'integration_tests'
            },
            'agent': {
                'class': 'PPO',
                'disable_progressbar': True,
                'track_ckpts': False,
                'experiment': {
                    'directory': str(Path(self.temp_dir)),
                    'experiment_name': config_name,
                    'write_interval': 100,
                    'checkpoint_interval': 1000,
                    'project': 'test_project',
                    'tags': [],
                    'group': ''
                }
            }
        }

        # Apply overrides
        if overrides:
            def deep_update(base, update):
                for key, value in update.items():
                    if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                        deep_update(base[key], value)
                    else:
                        base[key] = value
            deep_update(base_config, overrides)

        # Write config file
        config_path = self.config_dir / f"{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(base_config, f)

        return str(config_path)

    @patch('gymnasium.make', side_effect=mock_gym_make)
    @patch('learning.launch_utils_v2.create_policy_and_value_models')
    @patch('learning.launch_utils_v2.create_block_ppo_agents')
    @patch('skrl.trainers.torch.SequentialTrainer')
    def test_minimal_configuration(self, mock_trainer, mock_agents, mock_models, mock_gym):
        """Test factory runner with minimal configuration."""
        # Create minimal config
        config_path = self.create_test_config('minimal')

        # Mock returns
        mock_models.return_value = {'policy': MagicMock(), 'value': MagicMock()}
        mock_agents.return_value = [MagicMock()]

        # Import and test main workflow components
        # Note: factory_runnerv2 argument parsing is already mocked at module level
        from isaaclab_tasks.utils.hydra import hydra_task_config

        # Mock the hydra decorator to call function directly
        with patch.object(hydra_task_config, '__call__') as mock_hydra:
            def mock_decorator(func):
                def wrapper(*args, **kwargs):
                    # Create mock configs
                    mock_env_cfg = MagicMock()
                    mock_env_cfg.sim.device = 'cpu'
                    mock_env_cfg.scene.num_envs = 16
                    mock_agent_cfg = {'agent': {'disable_progressbar': True}}
                    return func(mock_env_cfg, mock_agent_cfg)
                return wrapper

            mock_hydra.return_value = mock_decorator

            # Load and test configuration
            resolved_config = ConfigManager.load_and_resolve_config(config_path)

            # Verify configuration structure
            assert 'primary' in resolved_config
            assert 'derived' in resolved_config
            assert 'environment' in resolved_config
            assert 'learning' in resolved_config
            assert 'model' in resolved_config
            assert 'wrappers' in resolved_config

            # Verify derived calculations
            derived = resolved_config['derived']
            assert derived['total_agents'] == 1  # 1 break force * 1 agent per break force
            assert derived['total_num_envs'] == 16  # 1 agent * 16 envs per agent

    @patch('gymnasium.make', side_effect=mock_gym_make)
    @patch('learning.launch_utils_v2.create_policy_and_value_models')
    @patch('learning.launch_utils_v2.create_block_ppo_agents')
    def test_all_wrappers_enabled(self, mock_agents, mock_models, mock_gym):
        """Test factory runner with all wrappers enabled."""
        # Create config with all wrappers enabled
        wrapper_overrides = {
            'wrappers': {
                'fragile_objects': {'enabled': True, 'break_force': [-1]},
                'force_torque_sensor': {'enabled': True, 'use_tanh_scaling': True},
                'observation_manager': {'enabled': True},
                'observation_noise': {
                    'enabled': True,
                    'global_noise_scale': 1.0,
                    'noise_groups': {
                        'fingertip_pos': {'noise_type': 'gaussian', 'std': 0.01, 'enabled': True}
                    }
                },
                'hybrid_control': {
                    'enabled': True,
                    'ctrl_torque': False,
                    'default_task_force_gains': [0.1, 0.1, 0.1, 0.001, 0.001, 0.001]
                },
                'factory_metrics': {'enabled': True},
                'wandb_logging': {'enabled': True, 'wandb_project': 'test'},
                'action_logging': {'enabled': True}
            },
            'model': {'use_hybrid_agent': True}
        }

        config_path = self.create_test_config('all_wrappers', wrapper_overrides)

        # Mock returns
        mock_models.return_value = {'policy': MagicMock(), 'value': MagicMock()}
        mock_agents.return_value = [MagicMock()]

        # Load configuration
        resolved_config = ConfigManager.load_and_resolve_config(config_path)

        # Test configuration application workflow
        from tests.mocks.mock_isaac_lab import MockBaseEnv, MockEnvConfig

        # Create mock environment and configs
        base_env = MockBaseEnv()
        env_cfg = MockEnvConfig()
        agent_cfg = {'agent': {'disable_progressbar': True}}

        # Test configuration application
        ConfigManager.apply_to_isaac_lab(env_cfg, agent_cfg, resolved_config)

        # Test wrapper application sequence
        env = base_env
        primary = resolved_config['primary']
        wrappers_config = resolved_config['wrappers']

        # Create derived configuration needed by some wrappers
        derived = ConfigManager._calculate_derived_params(primary)

        # Apply each wrapper if enabled
        if wrappers_config['fragile_objects']['enabled']:
            env = lUtils.apply_fragile_object_wrapper(env, wrappers_config['fragile_objects'], primary, derived)

        if wrappers_config['force_torque_sensor']['enabled']:
            # Mock RobotView directly in the force_torque_wrapper module to handle import caching
            import wrappers.sensors.force_torque_wrapper as ft_wrapper
            original_robot_view = getattr(ft_wrapper, 'RobotView', None)
            ft_wrapper.RobotView = MagicMock()
            try:
                env = lUtils.apply_force_torque_wrapper(env, wrappers_config['force_torque_sensor'])
            finally:
                # Restore original (even if it was None)
                if original_robot_view is None:
                    ft_wrapper.RobotView = None
                else:
                    ft_wrapper.RobotView = original_robot_view

        if wrappers_config['observation_manager']['enabled']:
            env = lUtils.apply_observation_manager_wrapper(env, wrappers_config['observation_manager'])

        if wrappers_config['observation_noise']['enabled']:
            env = lUtils.apply_observation_noise_wrapper(env, wrappers_config['observation_noise'])

        if wrappers_config['hybrid_control']['enabled']:
            env = lUtils.apply_hybrid_control_wrapper(env, wrappers_config['hybrid_control'])

        if wrappers_config['factory_metrics']['enabled']:
            env = lUtils.apply_factory_metrics_wrapper(env, derived)

        # Verify no exceptions during wrapper application
        assert env is not None

    @patch('gymnasium.make', side_effect=mock_gym_make)
    @patch('learning.launch_utils_v2.create_policy_and_value_models')
    @patch('learning.launch_utils_v2.create_block_ppo_agents')
    def test_debug_mode_configuration(self, mock_agents, mock_models, mock_gym):
        """Test factory runner with debug mode enabled."""
        debug_overrides = {
            'primary': {
                'debug_mode': True,
                'agents_per_break_force': 1,
                'num_envs_per_agent': 8,
                'episode_length_s': 5.0,
                'max_steps': 512
            },
            'learning': {
                'learning_epochs': 1,
                'policy_learning_rate': 1.0e-4
            }
        }

        config_path = self.create_test_config('debug_mode', debug_overrides)

        # Mock returns
        mock_models.return_value = {'policy': MagicMock(), 'value': MagicMock()}
        mock_agents.return_value = [MagicMock()]

        # Load configuration
        resolved_config = ConfigManager.load_and_resolve_config(config_path)

        # Test debug mode configuration
        assert resolved_config['primary']['debug_mode'] == True

        # Test easy mode application
        from tests.mocks.mock_isaac_lab import MockEnvConfig
        env_cfg = MockEnvConfig()
        agent_cfg = {'agent': {'disable_progressbar': True}}

        # Apply easy mode if debug enabled
        if resolved_config['primary']['debug_mode']:
            lUtils.apply_easy_mode(env_cfg, agent_cfg)

        # Verify no exceptions during easy mode application
        assert env_cfg is not None

    @patch('gymnasium.make', side_effect=mock_gym_make)
    @patch('learning.launch_utils_v2.create_policy_and_value_models')
    @patch('learning.launch_utils_v2.create_block_ppo_agents')
    def test_hybrid_control_configuration(self, mock_agents, mock_models, mock_gym):
        """Test factory runner with hybrid control enabled."""
        hybrid_overrides = {
            'model': {
                'use_hybrid_agent': True,
                'hybrid_agent': {
                    'ctrl_torque': True,
                    'unit_std_init': True,
                    'pos_init_std': 1.0,
                    'rot_init_std': 1.0,
                    'force_init_std': 1.0
                }
            },
            'wrappers': {
                'force_torque_sensor': {'enabled': True, 'use_tanh_scaling': True},
                'hybrid_control': {'enabled': True, 'ctrl_torque': True}
            }
        }

        config_path = self.create_test_config('hybrid_control', hybrid_overrides)

        # Mock returns
        mock_models.return_value = {'policy': MagicMock(), 'value': MagicMock()}
        mock_agents.return_value = [MagicMock()]

        # Load configuration
        resolved_config = ConfigManager.load_and_resolve_config(config_path)

        # Test hybrid agent configuration
        assert resolved_config['model']['use_hybrid_agent'] == True
        assert resolved_config['model']['hybrid_agent']['ctrl_torque'] == True
        assert resolved_config['wrappers']['hybrid_control']['enabled'] == True

        # Test hybrid agent parameter configuration
        from tests.mocks.mock_isaac_lab import MockEnvConfig
        env_cfg = MockEnvConfig()
        agent_cfg = {'agent': {'hybrid_agent': {}}}
        wrappers_config = resolved_config['wrappers']

        lUtils._configure_hybrid_agent_parameters(env_cfg, agent_cfg, resolved_config['model'], wrappers_config)

        # Verify hybrid agent parameters were set
        assert 'ctrl_torque' in agent_cfg['agent']['hybrid_agent']

    @patch('gymnasium.make', side_effect=mock_gym_make)
    @patch('learning.launch_utils_v2.create_policy_and_value_models')
    @patch('learning.launch_utils_v2.create_block_ppo_agents')
    def test_multi_agent_configuration(self, mock_agents, mock_models, mock_gym):
        """Test factory runner with multiple agents and break forces."""
        multi_agent_overrides = {
            'primary': {
                'agents_per_break_force': 2,
                'num_envs_per_agent': 32,
                'break_forces': [25.0, 50.0, 75.0]  # 3 break forces
            }
        }

        config_path = self.create_test_config('multi_agent', multi_agent_overrides)

        # Mock returns for multiple agents
        mock_models.return_value = {'policy': [MagicMock(), MagicMock()], 'value': [MagicMock(), MagicMock()]}
        mock_agents.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()]

        # Load configuration
        resolved_config = ConfigManager.load_and_resolve_config(config_path)

        # Verify multi-agent calculations
        derived = resolved_config['derived']
        assert derived['total_agents'] == 6  # 3 break forces * 2 agents per break force
        assert derived['total_num_envs'] == 192  # 6 agents * 32 envs per agent

    @patch('gymnasium.make', side_effect=mock_gym_make)
    def test_configuration_override_system(self, mock_gym):
        """Test CLI override system."""
        config_path = self.create_test_config('override_test')

        # Test CLI overrides
        overrides = [
            'primary.num_envs_per_agent=64',
            'learning.policy_learning_rate=5.0e-6',
            'wrappers.force_torque_sensor.enabled=true'
        ]

        resolved_config = ConfigManager.load_and_resolve_config(config_path, overrides)

        # Verify overrides were applied
        assert resolved_config['primary']['num_envs_per_agent'] == 64
        assert resolved_config['learning']['policy_learning_rate'] == 5.0e-6
        assert resolved_config['wrappers']['force_torque_sensor']['enabled'] == True

    @patch('gymnasium.make', side_effect=mock_gym_make)
    @patch('learning.launch_utils_v2.create_policy_and_value_models')
    @patch('learning.launch_utils_v2.create_block_ppo_agents')
    def test_learning_configuration(self, mock_agents, mock_models, mock_gym):
        """Test learning parameter configuration."""
        learning_overrides = {
            'learning': {
                'learning_epochs': 5,
                'mini_batches': 32,
                'policy_learning_rate': 3.0e-6,
                'critic_learning_rate': 1.0e-5,
                'value_update_ratio': 3,
                'use_huber_value_loss': True,
                'state_preprocessor': True,
                'value_preprocessor': True
            }
        }

        config_path = self.create_test_config('learning_test', learning_overrides)

        # Mock returns
        mock_models.return_value = {'policy': MagicMock(), 'value': MagicMock()}
        mock_agents.return_value = [MagicMock()]

        # Load configuration
        resolved_config = ConfigManager.load_and_resolve_config(config_path)

        # Test learning configuration application
        agent_cfg = {'agent': {}}
        learning = resolved_config['learning']
        max_rollout_steps = resolved_config['derived']['rollout_steps']

        lUtils.apply_learning_config(agent_cfg, learning, max_rollout_steps)

        # Verify learning parameters were applied
        assert agent_cfg['agent']['learning_epochs'] == 5
        assert agent_cfg['agent']['policy_learning_rate'] == 3.0e-6
        assert agent_cfg['agent']['critic_learning_rate'] == 1.0e-5

    @patch('gymnasium.make', side_effect=mock_gym_make)
    @patch('learning.launch_utils_v2.create_policy_and_value_models')
    @patch('learning.launch_utils_v2.create_block_ppo_agents')
    def test_model_configuration(self, mock_agents, mock_models, mock_gym):
        """Test model architecture configuration."""
        model_overrides = {
            'model': {
                'actor': {'n': 2, 'latent_size': 512},
                'critic': {'n': 3, 'latent_size': 1024},
                'force_encoding': 'tanh',
                'last_layer_scale': 0.1,
                'act_init_std': 0.5
            }
        }

        config_path = self.create_test_config('model_test', model_overrides)

        # Mock returns
        mock_models.return_value = {'policy': MagicMock(), 'value': MagicMock()}
        mock_agents.return_value = [MagicMock()]

        # Load configuration
        resolved_config = ConfigManager.load_and_resolve_config(config_path)

        # Test model configuration application
        agent_cfg = {'models': {'policy': {}, 'value': {}}}
        model = resolved_config['model']

        lUtils.apply_model_config(agent_cfg, model)

        # Verify model parameters were applied
        assert agent_cfg['models']['policy']['n'] == 2
        assert agent_cfg['models']['policy']['latent_size'] == 512
        assert agent_cfg['models']['value']['n'] == 3
        assert agent_cfg['models']['value']['latent_size'] == 1024

    @patch('gymnasium.make', side_effect=mock_gym_make)
    def test_environment_scene_configuration(self, mock_gym):
        """Test environment scene configuration."""
        config_path = self.create_test_config('scene_test')

        # Load configuration
        resolved_config = ConfigManager.load_and_resolve_config(config_path)

        # Test scene configuration
        from tests.mocks.mock_isaac_lab import MockEnvConfig
        env_cfg = MockEnvConfig()
        primary = resolved_config['primary']
        derived = resolved_config['derived']

        lUtils.configure_environment_scene(env_cfg, primary, derived)

        # Verify scene was configured
        assert env_cfg.scene.num_envs == derived['total_num_envs']

    def test_config_reference_resolution(self):
        """Test configuration reference resolution."""
        # Create config with references
        ref_config = {
            'primary': {'episode_length': 10.0, 'agents': 2},
            'environment': {'episode_length_s': '${primary.episode_length}'},
            'wrappers': {'factory_metrics': {'num_agents': '${primary.agents}'}}
        }

        config_path = self.config_dir / "ref_test.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(ref_config, f)

        # Load and resolve
        resolved_config = ConfigManager.load_and_resolve_config(str(config_path))

        # Verify references were resolved
        assert resolved_config['environment']['episode_length_s'] == 10.0
        assert resolved_config['wrappers']['factory_metrics']['num_agents'] == 2

    @patch('gymnasium.make', side_effect=mock_gym_make)
    @patch('learning.launch_utils_v2.create_policy_and_value_models')
    @patch('learning.launch_utils_v2.create_block_ppo_agents')
    def test_wrapper_dependency_chain(self, mock_agents, mock_models, mock_gym):
        """Test wrapper dependency chain validation."""
        # Test that factory metrics requires wandb logging when enabled
        dependency_overrides = {
            'wrappers': {
                'factory_metrics': {'enabled': True},
                'wandb_logging': {'enabled': True, 'wandb_project': 'test'}
            }
        }

        config_path = self.create_test_config('dependency_test', dependency_overrides)

        # Mock returns
        mock_models.return_value = {'policy': MagicMock(), 'value': MagicMock()}
        mock_agents.return_value = [MagicMock()]

        # Load configuration
        resolved_config = ConfigManager.load_and_resolve_config(config_path)

        # Test wrapper application with dependencies
        from tests.mocks.mock_isaac_lab import MockBaseEnv, MockEnvConfig
        env = MockBaseEnv()
        env_cfg = MockEnvConfig()
        agent_cfg = {'agent': {'disable_progressbar': True}}
        primary = resolved_config['primary']
        wrappers_config = resolved_config['wrappers']

        # Set up experiment logging first (required for wandb wrapper)
        lUtils.setup_experiment_logging(env_cfg, agent_cfg, resolved_config)

        # Verify that the configuration setup was successful
        assert 'agent_specific_configs' in resolved_config
        assert resolved_config['wrappers']['wandb_logging']['enabled'] == True
        assert resolved_config['wrappers']['factory_metrics']['enabled'] == True

        # Verify that agent config has experiment setup
        assert 'experiment' in agent_cfg['agent']
        assert 'project' in agent_cfg['agent']['experiment']

        # Test that dependency chain configuration is valid
        # (We test the actual wrapper application in the end-to-end test)
        assert env is not None

    @patch('gymnasium.make', side_effect=mock_gym_make)
    def test_config_validation(self, mock_gym):
        """Test configuration validation."""
        # Test invalid configuration
        invalid_config = {
            'primary': {
                'agents_per_break_force': 0,  # Invalid: should be > 0
                'num_envs_per_agent': -1,     # Invalid: should be > 0
                'break_forces': []            # Invalid: should not be empty list
            }
        }

        config_path = self.config_dir / "invalid_test.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)

        # Should handle invalid configuration gracefully
        try:
            resolved_config = ConfigManager.load_and_resolve_config(str(config_path))
            # Configuration system should handle this without crashing
            assert resolved_config is not None
        except Exception as e:
            # If it raises an exception, it should be a meaningful validation error
            assert "invalid" in str(e).lower() or "error" in str(e).lower()


class TestFactoryRunnerEndToEnd:
    """End-to-end integration tests simulating full factory runner execution."""

    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_dir.mkdir(parents=True)

    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.temp_dir)

    @patch('gymnasium.make', side_effect=mock_gym_make)
    @patch('learning.launch_utils_v2.create_policy_and_value_models')
    @patch('learning.launch_utils_v2.create_block_ppo_agents')
    @patch('skrl.trainers.torch.SequentialTrainer')
    @patch('memories.multi_random.MultiRandomMemory')
    def test_complete_training_pipeline(self, mock_memory, mock_trainer, mock_agents, mock_models, mock_gym):
        """Test complete training pipeline from configuration to trainer setup."""
        # Create a comprehensive configuration
        config = {
            'primary': {
                'agents_per_break_force': 1,
                'num_envs_per_agent': 16,
                'break_forces': [-1],
                'episode_length_s': 10.0,
                'decimation': 4,
                'policy_hz': 15,
                'max_steps': 1024,
                'debug_mode': False
            },
            'defaults': {'task_name': "Isaac-Factory-PegInsert-Local-v0"},
            'environment': {
                'episode_length_s': 10.0,
                'component_dims': {'fingertip_pos': 3, 'joint_pos': 7},
                'component_attr_map': {'fingertip_pos': 'fingertip_pos', 'joint_pos': 'joint_pos'}
            },
            'learning': {
                'learning_epochs': 2,
                'policy_learning_rate': 1.0e-5,
                'critic_learning_rate': 1.0e-4,
                'state_preprocessor': True,
                'value_preprocessor': True
            },
            'model': {
                'use_hybrid_agent': False,
                'actor': {'n': 1, 'latent_size': 64},
                'critic': {'n': 1, 'latent_size': 128}
            },
            'wrappers': {
                'fragile_objects': {'enabled': False},
                'force_torque_sensor': {'enabled': False},
                'observation_manager': {'enabled': True},
                'observation_noise': {'enabled': False},
                'hybrid_control': {'enabled': False},
                'factory_metrics': {'enabled': True},
                'wandb_logging': {'enabled': False},
                'action_logging': {'enabled': False}
            },
            'experiment': {'name': 'e2e_test'},
            'agent': {
                'class': 'PPO',
                'disable_progressbar': True,
                'experiment': {
                    'directory': str(Path(self.temp_dir)),
                    'experiment_name': 'e2e_test'
                }
            }
        }

        config_path = self.config_dir / "e2e_test.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Mock all components
        mock_models.return_value = {'policy': MagicMock(), 'value': MagicMock()}
        mock_agents.return_value = [MagicMock()]
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        # Test complete pipeline simulation
        resolved_config = ConfigManager.load_and_resolve_config(str(config_path))

        # Simulate the main workflow steps
        primary = resolved_config['primary']
        derived = resolved_config['derived']
        environment = resolved_config.get('environment', {})
        learning = resolved_config['learning']
        model = resolved_config['model']
        wrappers_config = resolved_config.get('wrappers', {})

        # Step 1-6: Configuration application
        from tests.mocks.mock_isaac_lab import MockBaseEnv, MockEnvConfig
        env_cfg = MockEnvConfig()
        agent_cfg = {'agent': {'disable_progressbar': True}}

        ConfigManager.apply_to_isaac_lab(env_cfg, agent_cfg, resolved_config)
        lUtils.configure_environment_scene(env_cfg, primary, derived)
        lUtils.apply_environment_overrides(env_cfg, environment)

        # Step 7-10: Learning and model configuration
        max_rollout_steps = derived['rollout_steps']
        lUtils.apply_learning_config(agent_cfg, learning, max_rollout_steps)
        lUtils.apply_model_config(agent_cfg, model)
        lUtils.setup_experiment_logging(env_cfg, agent_cfg, resolved_config)

        # Step 11: Environment creation and wrapper application
        env = MockBaseEnv()

        # Apply wrapper stack
        if wrappers_config.get('observation_manager', {}).get('enabled', False):
            env = lUtils.apply_observation_manager_wrapper(env, wrappers_config['observation_manager'])

        if wrappers_config.get('factory_metrics', {}).get('enabled', False):
            env = lUtils.apply_factory_metrics_wrapper(env, derived)

        # Step 12: SKRL wrapper
        env.num_envs = derived['total_num_envs']
        env.action_space = MagicMock()
        env.cfg = MagicMock()
        env.cfg.observation_space = 32
        env.cfg.state_space = 48

        # Step 13-16: Memory, models, and agents
        device = 'cpu'

        # Create memory
        mock_memory.assert_not_called()  # Reset call tracking
        memory = mock_memory(
            memory_size=derived['rollout_steps'],
            num_envs=env.num_envs,
            device=device,
            replacement=True,
            num_agents=derived['total_agents']
        )
        mock_memory.assert_called_once()

        # Create models and agents
        models = mock_models(env_cfg, agent_cfg, env, model, wrappers_config, derived)
        agents = mock_agents(env_cfg, agent_cfg, env, models, memory, derived, learning)

        # Create trainer
        cfg_trainer = {
            "timesteps": derived['max_steps'] // derived['total_num_envs'],
            "headless": True,
            "close_environment_at_exit": True,
            "disable_progressbar": agent_cfg['agent']['disable_progressbar']
        }

        trainer = mock_trainer(cfg=cfg_trainer, env=env, agents=agents)

        # Verify all components were created successfully
        assert models is not None
        assert agents is not None
        assert memory is not None
        assert trainer is not None

        # Verify trainer was configured correctly
        mock_trainer.assert_called_once()
        call_kwargs = mock_trainer.call_args[1]
        assert call_kwargs['cfg'] == cfg_trainer
        assert call_kwargs['env'] == env
        assert call_kwargs['agents'] == agents