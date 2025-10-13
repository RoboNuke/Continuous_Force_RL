"""
Pytest configuration and fixtures for testing.
Sets up mocks for isaac lab, isaac sim, and wandb.
"""

import sys
import os
import pytest
import torch
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our mocks
from tests.mocks.mock_wandb import MockWandb
from tests.mocks.mock_isaac_lab import MockBaseEnv, MockIsaacLabEnv
from tests.mocks.mock_isaac_sim import MockSimulationApp, MockWorld, MockRobot


@pytest.fixture(autouse=True)
def mock_external_modules():
    """Automatically mock external modules for all tests."""
    # Mock wandb
    wandb_mock = MockWandb()
    with patch.dict('sys.modules', {
        'wandb': wandb_mock,
    }):
        # Mock isaac lab (there might be different import paths)
        isaac_lab_mock = MagicMock()
        isaac_lab_mock.envs.BaseEnv = MockBaseEnv
        isaac_lab_mock.make_env = lambda *args, **kwargs: MockIsaacLabEnv()

        with patch.dict('sys.modules', {
            'isaac_lab': isaac_lab_mock,
            'isaac_lab.envs': isaac_lab_mock.envs,
            'omni.isaac.lab': isaac_lab_mock,
            'omni.isaac.lab.envs': isaac_lab_mock.envs,
        }):
            # Mock isaac sim
            isaac_sim_mock = MagicMock()
            isaac_sim_mock.core.SimulationApp = MockSimulationApp
            isaac_sim_mock.core.World = MockWorld
            isaac_sim_mock.robots.Robot = MockRobot

            with patch.dict('sys.modules', {
                'isaac_sim': isaac_sim_mock,
                'isaac_sim.core': isaac_sim_mock.core,
                'isaac_sim.robots': isaac_sim_mock.robots,
                'omni.isaac.core': isaac_sim_mock.core,
                'omni.isaac.manipulators': isaac_sim_mock.robots,
            }):
                yield


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration for BlockPPO."""
    return {
        # Required configuration keys based on factory_base.yaml
        'ckpt_tracker_path': os.path.join(temp_dir, 'test_ckpt_tracker.txt'),
        'track_ckpts': True,
        'value_update_ratio': 2,
        'use_huber_value_loss': True,
        'random_value_timesteps': 150,

        # Agent experiment configurations (required for write_checkpoint)
        'agent_0': {
            'experiment': {
                'directory': temp_dir,
                'experiment_name': 'test_experiment'
            }
        },
        'agent_1': {
            'experiment': {
                'directory': temp_dir,
                'experiment_name': 'test_experiment'
            }
        },

        # Preprocessor configuration
        'state_preprocessor': None,
        'state_preprocessor_kwargs': {},
        'value_preprocessor': None,
        'value_preprocessor_kwargs': {},

        # PPO configuration (from skrl PPO_DEFAULT_CONFIG)
        'rollouts': 16,
        'learning_epochs': 4,
        'mini_batches': 4,
        'discount_factor': 0.99,
        'lambda': 0.95,
        'learning_rate': 3e-4,
        'random_timesteps': 0,
        'learning_starts': 0,
        'grad_norm_clip': 0.5,
        'ratio_clip': 0.2,
        'value_clip': 0.2,
        'clip_predicted_values': False,
        'entropy_loss_scale': 0.01,
        'value_loss_scale': 1.0,
        'kl_threshold': 0.05,
        'rewards_shaper': None,
        'time_limit_bootstrap': True,
        'experiment': {
            'directory': temp_dir,
            'experiment_name': 'test_experiment',
            'checkpoint_interval': 100,
            'write_interval': 10
        }
    }


@pytest.fixture
def mock_env():
    """Create a mock environment with wrapper support."""
    env = MockBaseEnv()
    # Ensure it has the required wrapper interface
    env.unwrapped = env
    env.add_metrics = MagicMock()
    return env


@pytest.fixture
def mock_models():
    """Create mock models for BlockPPO."""
    # Create mock policy model
    policy_mock = MagicMock()
    policy_mock.actor_mean = MagicMock()
    policy_mock.actor_logstd = torch.zeros(6)  # Assuming 6 actions
    policy_mock.act = MagicMock(return_value=(
        torch.randn(256, 6),  # actions
        torch.randn(256),     # log_prob
        {}                    # additional outputs
    ))
    policy_mock.get_entropy = MagicMock(return_value=torch.randn(256, 1))
    policy_mock.parameters = MagicMock(return_value=[torch.randn(10, 10, requires_grad=True)])
    policy_mock.reduce_parameters = MagicMock()

    # Create mock value model
    value_mock = MagicMock()
    value_mock.critic = MagicMock()
    value_mock.act = MagicMock(return_value=(
        torch.randn(256, 1),  # values
        torch.randn(256),     # log_prob (not used for value)
        {}                    # additional outputs
    ))
    value_mock.parameters = MagicMock(return_value=[torch.randn(10, 10, requires_grad=True)])
    value_mock.reduce_parameters = MagicMock()
    value_mock.train = MagicMock()

    return {
        'policy': policy_mock,
        'value': value_mock
    }


@pytest.fixture
def mock_memory():
    """Create mock memory for BlockPPO."""
    memory_mock = MagicMock()
    memory_mock.create_tensor = MagicMock()
    memory_mock.add_samples = MagicMock()
    memory_mock.get_tensor_by_name = MagicMock(return_value=torch.randn(16, 256, 1))
    memory_mock.set_tensor_by_name = MagicMock()
    memory_mock.sample_all = MagicMock(return_value=[
        (
            torch.randn(64, 64),    # states
            torch.randn(64, 6),     # actions
            torch.randn(64, 1),     # log_prob
            torch.randn(64, 1),     # values
            torch.randn(64, 1),     # returns
            torch.randn(64, 1),     # advantages
        )
    ])
    return memory_mock


@pytest.fixture
def mock_optimizer():
    """Create mock optimizer for BlockPPO."""
    optimizer_mock = MagicMock()
    optimizer_mock.zero_grad = MagicMock()
    optimizer_mock.step = MagicMock()
    optimizer_mock.state = {}
    optimizer_mock.defaults = {
        'lr': 1e-4,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }
    return optimizer_mock


@pytest.fixture
def mock_scaler():
    """Create mock gradient scaler for BlockPPO."""
    scaler_mock = MagicMock()
    scaler_mock.scale = MagicMock(return_value=MagicMock())
    scaler_mock.unscale_ = MagicMock()
    scaler_mock.step = MagicMock()
    scaler_mock.update = MagicMock()
    return scaler_mock