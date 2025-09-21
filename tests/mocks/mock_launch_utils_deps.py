"""
Mock dependencies for launch_utils_v2.py testing.

This file provides minimal mock implementations for Isaac Lab, Isaac Sim, and Wandb
to enable unit testing, while relying on the actual installed packages for everything else.
"""

import torch
import gymnasium as gym
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union


# ===== MOCK ISAAC LAB =====

class MockIsaacLabEnv(gym.Env):
    """Mock Isaac Lab environment."""

    def __init__(self):
        super().__init__()
        self.cfg = MockEnvConfig()
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(32,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))
        self.device = torch.device('cpu')
        self.num_envs = 16

    def step(self, action):
        """Mock step method."""
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Mock reset method."""
        obs = self.observation_space.sample()
        info = {}
        return obs, info

    def add_metrics(self, metrics):
        """Mock add_metrics method for wandb logging."""
        pass

    def _configure_gym_env_spaces(self):
        """Mock space configuration method."""
        # Update action space based on cfg.action_space if it exists
        if hasattr(self.cfg, 'action_space'):
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(self.cfg.action_space,)
            )


class MockEnvConfig:
    """Mock environment configuration object."""

    def __init__(self):
        # Scene configuration
        self.scene = Mock()
        self.scene.num_envs = 16
        self.scene.replicate_physics = True

        # Task configuration
        self.task = Mock()
        self.task.duration_s = 10.0
        self.task.hand_init_pos = [0.0, 0.0, 0.035]
        self.task.hand_init_pos_noise = [0.0025, 0.0025, 0.00]
        self.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
        self.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
        self.task.fixed_asset_init_orn_deg = 0.0
        self.task.fixed_asset_init_orn_range_deg = 0.0
        self.task.held_asset_pos_noise = [0.0, 0.0, 0.0]
        self.task.held_asset_rot_init = 0.0
        self.task_name = 'MockTask'

        # Simulation configuration
        self.sim = Mock()
        self.sim.device = 'cpu'

        # Control configuration - use dict-like object for new naming
        class CtrlConfig:
            def __init__(self):
                self.default_task_prop_gains = [1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0]
                self.pos_action_threshold = [0.1, 0.1, 0.1]
                self.rot_action_threshold = [0.1, 0.1, 0.1]
                self.default_task_force_gains = [100.0, 100.0, 100.0, 10.0, 10.0, 10.0]
                self.force_action_threshold = [10.0, 10.0, 10.0]
                self.torque_action_bounds = [1.0, 1.0, 1.0]
                self.force_action_bounds = [50.0, 50.0, 50.0]
                self.torque_action_threshold = [1.0, 1.0, 1.0]
                self.pos_action_bounds = [0.2, 0.2, 0.2]
                self.rot_action_bounds = [0.2, 0.2, 0.2]

            def get(self, key, default=None):
                """Support dict-like access."""
                return getattr(self, key, default)

        self.ctrl = CtrlConfig()

        # Configuration attributes
        self.cfg_ctrl = {
            'default_task_force_gains': [100.0, 100.0, 100.0, 10.0, 10.0, 10.0],
            'force_action_bounds': [50.0, 50.0, 50.0],
            'torque_action_bounds': [5.0, 5.0, 5.0],
            'force_action_threshold': [10.0, 10.0, 10.0],
            'torque_action_threshold': [1.0, 1.0, 1.0],
            'pos_action_bounds': [0.2, 0.2, 0.2],
            'rot_action_bounds': [0.2, 0.2, 0.2]
        }

        # Environment attributes
        self.num_agents = 1
        self.episode_length_s = 10.0
        self.use_force_sensor = False
        self.action_space = 6  # Default action space size

        # Observation configuration
        self.obs_order = ['fingertip_pos', 'joint_pos']
        self.state_order = ['fingertip_pos', 'joint_pos']
        self.observation_space = 32
        self.state_space = 48

        # Component configuration
        self.component_dims = {
            'fingertip_pos': 3,
            'joint_pos': 7,
            'force_torque': 6,
            'prev_actions': 6
        }

        # Component attribute mapping
        self.component_attr_map = {
            'fingertip_pos': 'fingertip_pos',
            'joint_pos': 'joint_pos',
            'force_torque': 'robot_force_torque',
            'prev_actions': 'prev_actions'
        }


# ===== MOCK ISAAC SIM =====

class MockIsaacSim:
    """Mock Isaac Sim components."""

    @staticmethod
    def initialize():
        """Mock Isaac Sim initialization."""
        pass


# ===== MOCK WANDB =====

class MockWandb:
    """Mock Wandb module."""

    @staticmethod
    def init(**kwargs):
        """Mock wandb.init()."""
        return Mock()

    @staticmethod
    def log(data):
        """Mock wandb.log()."""
        pass

    @staticmethod
    def finish():
        """Mock wandb.finish()."""
        pass


def setup_minimal_mocks():
    """Set up minimal mocks for Isaac Lab, Isaac Sim, and Wandb only."""
    import sys
    from unittest.mock import MagicMock

    # Mock Isaac Lab components
    try:
        import isaac_lab
    except ImportError:
        # Only mock if not already installed
        mock_isaac_lab = MagicMock()
        sys.modules['isaac_lab'] = mock_isaac_lab

    # Mock Isaac Sim components
    try:
        import isaacsim
    except ImportError:
        # Only mock if not already installed
        mock_isaacsim = MagicMock()
        sys.modules['isaacsim'] = mock_isaacsim
        # Mock torch utilities specifically with proper function behavior
        mock_torch_utils = MagicMock()
        mock_torch_utils.quat_from_angle_axis = MagicMock(side_effect=lambda x: torch.zeros(x.shape[0], 4) if hasattr(x, 'shape') else torch.zeros(1, 4))
        mock_torch_utils.quat_mul = MagicMock(side_effect=lambda x, y: torch.zeros(x.shape[0], 4) if hasattr(x, 'shape') else torch.zeros(1, 4))
        mock_torch_utils.get_euler_xyz = MagicMock(side_effect=lambda x: (torch.zeros(x.shape[0]), torch.zeros(x.shape[0]), torch.zeros(x.shape[0])) if hasattr(x, 'shape') else (torch.zeros(1), torch.zeros(1), torch.zeros(1)))
        mock_torch_utils.quat_from_euler_xyz = MagicMock(side_effect=lambda x, y, z: torch.zeros(x.shape[0], 4) if hasattr(x, 'shape') else torch.zeros(1, 4))
        sys.modules['isaacsim.core.utils.torch'] = mock_torch_utils

    try:
        import omni
    except ImportError:
        # Only mock if not already installed
        mock_omni = MagicMock()
        sys.modules['omni'] = mock_omni
        # Mock Isaac Core torch utilities with proper function behavior
        mock_omni_torch_utils = MagicMock()
        mock_omni_torch_utils.quat_from_angle_axis = MagicMock(side_effect=lambda x: torch.zeros(x.shape[0], 4) if hasattr(x, 'shape') else torch.zeros(1, 4))
        mock_omni_torch_utils.quat_mul = MagicMock(side_effect=lambda x, y: torch.zeros(x.shape[0], 4) if hasattr(x, 'shape') else torch.zeros(1, 4))
        mock_omni_torch_utils.get_euler_xyz = MagicMock(side_effect=lambda x: (torch.zeros(x.shape[0]), torch.zeros(x.shape[0]), torch.zeros(x.shape[0])) if hasattr(x, 'shape') else (torch.zeros(1), torch.zeros(1), torch.zeros(1)))
        mock_omni_torch_utils.quat_from_euler_xyz = MagicMock(side_effect=lambda x, y, z: torch.zeros(x.shape[0], 4) if hasattr(x, 'shape') else torch.zeros(1, 4))
        mock_omni.isaac = MagicMock()
        mock_omni.isaac.core = MagicMock()
        mock_omni.isaac.core.utils = MagicMock()
        mock_omni.isaac.core.utils.torch = mock_omni_torch_utils
        sys.modules['omni.isaac'] = mock_omni.isaac
        sys.modules['omni.isaac.core'] = mock_omni.isaac.core
        sys.modules['omni.isaac.core.utils'] = mock_omni.isaac.core.utils
        sys.modules['omni.isaac.core.utils.torch'] = mock_omni_torch_utils

    # Mock Wandb
    try:
        import wandb
    except ImportError:
        # Only mock if not already installed
        sys.modules['wandb'] = MockWandb


# ===== MOCK AGENT CONFIG FOR TESTING =====

def create_mock_agent_config():
    """Create mock agent configuration dictionary."""
    return {
        'agent': {
            'rollouts': 32,
            'experiment': {
                'write_interval': 32,
                'checkpoint_interval': 320,
                'project': 'test_project',
                'tags': [],
                'group': 'test_group'
            },
            'easy_mode': False,
            'break_force': [50.0],
            'num_envs': 16,
            'rewards_shaper_scale': 1.0,
            'learning_rate': 0.0003,
            'batch_size': 32,
            'entropy_coeff': 0.0,
            'hybrid_agent': {
                'unit_std_init': False,
                'pos_init_std': 0.1,
                'rot_init_std': 0.1,
                'force_init_std': 0.1,
                'pos_scale': 1.0,
                'rot_scale': 1.0,
                'force_scale': 1.0,
                'torque_scale': 1.0
            }
        },
        'models': {
            'policy': {},
            'value': {}
        }
    }