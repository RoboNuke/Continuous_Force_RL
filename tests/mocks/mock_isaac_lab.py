"""
Mock Isaac Lab modules for testing.
Simple stub implementations to replace Isaac Lab functionality.
"""

import torch
import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional


def axis_angle_from_quat(quat):
    """Convert quaternion to axis-angle representation."""
    # Normalize quaternion
    quat_norm = torch.norm(quat, dim=1, keepdim=True)
    quat = quat / (quat_norm + 1e-8)

    # Extract angle and axis
    w = quat[:, 0]
    xyz = quat[:, 1:4]

    # Compute angle
    angle = 2 * torch.acos(torch.clamp(torch.abs(w), 0, 1))

    # Compute axis
    sin_half_angle = torch.sin(angle / 2)
    axis = xyz / (sin_half_angle.unsqueeze(-1) + 1e-8)

    # Handle small angles
    small_angle_mask = angle < 1e-6
    axis[small_angle_mask] = torch.tensor([1.0, 0.0, 0.0], device=quat.device)

    return axis * angle.unsqueeze(-1)


class MockIsaacLabEnv(gym.Env):
    """Mock Isaac Lab environment."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or {}
        self.num_envs = getattr(cfg, 'num_envs', 256)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(64,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))
        self.device = torch.device("cpu")
        self._episode_length = 500
        self._current_step = 0

        # Add wrapper integration support
        self._unwrapped = self

        # Add robot for wrapper initialization detection
        self._robot = True

        # Add force-torque data for force torque wrapper
        self.robot_force_torque = torch.randn(self.num_envs, 6, device=self.device)

        # Add observation data for wrappers
        self._setup_observation_data()

    @property
    def unwrapped(self):
        return self._unwrapped

    @unwrapped.setter
    def unwrapped(self, value):
        self._unwrapped = value

    def _setup_observation_data(self):
        """Setup observation data for wrapper testing."""
        self.fingertip_pos = torch.randn(self.num_envs, 3, device=self.device)
        self.ee_linvel = torch.randn(self.num_envs, 3, device=self.device)
        self.ee_angvel = torch.randn(self.num_envs, 3, device=self.device)
        self.joint_pos = torch.randn(self.num_envs, 7, device=self.device)
        self.fingertip_quat = torch.randn(self.num_envs, 4, device=self.device)

    def _get_factory_obs_state_dict(self):
        """Mock factory observation state dict method for force torque wrapper."""
        # Return Isaac Lab factory format: obs_dict, state_dict
        obs_dict = {
            'fingertip_pos': self.fingertip_pos,
            'joint_pos': self.joint_pos,
            'ee_linvel': self.ee_linvel,
        }
        state_dict = {
            'fingertip_pos': self.fingertip_pos,
            'joint_pos': self.joint_pos,
            'fingertip_quat': self.fingertip_quat,
        }
        return obs_dict, state_dict

    def _get_observations(self):
        """Mock get observations method for observation manager wrapper."""
        # Return Isaac Lab factory format: {"policy": tensor, "critic": tensor}
        policy_obs = torch.randn(self.num_envs, 32, device=self.device)
        critic_obs = torch.randn(self.num_envs, 48, device=self.device)
        return {"policy": policy_obs, "critic": critic_obs}

    def _init_tensors(self):
        """Mock init tensors method for force torque wrapper."""
        pass

    def _compute_intermediate_values(self, dt=None):
        """Mock compute intermediate values for force torque wrapper."""
        # Update force-torque data
        self.robot_force_torque = torch.randn(self.num_envs, 6, device=self.device)

    def _pre_physics_step(self, actions):
        """Mock pre-physics step for history wrapper."""
        # Update observation data
        self.fingertip_pos += torch.randn_like(self.fingertip_pos) * 0.01

    def reset(self, seed=None):
        self._current_step = 0
        obs = torch.randn(self.num_envs, 64)
        info = {"timeout": torch.zeros(self.num_envs, dtype=torch.bool)}
        return obs, info

    def step(self, actions):
        self._current_step += 1
        obs = torch.randn(self.num_envs, 64)
        rewards = torch.randn(self.num_envs)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.full((self.num_envs,), self._current_step >= self._episode_length, dtype=torch.bool)
        info = {
            "timeout": truncated,
            "current_successes": torch.zeros(self.num_envs),
            "current_engagements": torch.zeros(self.num_envs),
        }
        return obs, rewards, terminated, truncated, info

    def close(self):
        pass


class MockArticulation:
    """Mock articulation for scene management."""

    def __init__(self, device="cpu", num_envs=4):
        self.device = device
        self.num_envs = num_envs

    def write_root_pose_to_sim(self, pose, env_ids=None):
        """Mock writing root pose to simulation."""
        pass

    def write_root_velocity_to_sim(self, velocity, env_ids=None):
        """Mock writing root velocity to simulation."""
        pass

    def write_joint_state_to_sim(self, position, velocity, env_ids=None):
        """Mock writing joint state to simulation."""
        pass

    def reset(self, env_ids=None):
        """Mock articulation reset."""
        pass


class MockRobotView:
    """Mock RobotView for Isaac Sim force-torque sensor integration."""

    def __init__(self, prim_paths_expr="/World/envs/env_.*/Robot", name="robot_view"):
        self.name = name
        self.prim_paths_expr = prim_paths_expr
        self._is_initialized = False
        self.count = 4  # Default number of robots
        self.device = torch.device("cpu")

    def initialize(self, physics_sim_view=None):
        """Mock initialization."""
        self._is_initialized = True

    def is_initialized(self):
        """Check if view is initialized."""
        return self._is_initialized

    def get_joint_efforts(self, indices=None, joint_indices=None):
        """Mock getting joint efforts (forces/torques)."""
        if not self._is_initialized:
            raise RuntimeError("RobotView must be initialized before use")

        if indices is None:
            num_envs = self.count
        else:
            num_envs = len(indices)

        if joint_indices is None:
            num_joints = 7  # Default joint count
        else:
            num_joints = len(joint_indices)

        # Return random force-torque data
        return torch.randn(num_envs, num_joints, device=self.device)

    def get_measured_joint_forces(self, indices=None, joint_indices=None):
        """Mock getting measured joint forces."""
        if not self._is_initialized:
            raise RuntimeError("RobotView must be initialized before use")

        if indices is None:
            num_envs = self.count
        else:
            num_envs = len(indices)

        # Return force-torque data with 9 joints (including joint 8 for end-effector)
        num_joints = 9
        forces = torch.randn(num_envs, num_joints, 6, device=self.device)

        # Make sure joint 8 has realistic force-torque values
        forces[:, 8, :] = torch.randn(num_envs, 6, device=self.device) * 0.1

        return forces

    def get_dof_names(self):
        """Mock getting DOF names."""
        return [f"joint_{i}" for i in range(7)]

    def get_articulation_root_poses(self, indices=None):
        """Mock getting root poses."""
        if indices is None:
            num_envs = self.count
        else:
            num_envs = len(indices)
        return torch.randn(num_envs, 7, device=self.device)

    def set_joint_efforts(self, efforts, indices=None, joint_indices=None):
        """Mock setting joint efforts."""
        pass


class MockScene:
    """Mock scene for environment state management."""

    def __init__(self, device="cpu", num_envs=4):
        self.device = device
        self.num_envs = num_envs
        self.env_origins = torch.zeros((num_envs, 3), device=device)
        self.articulations = {
            "robot": MockArticulation(device, num_envs),
            "object": MockArticulation(device, num_envs)
        }

    def get_state(self):
        """Mock getting scene state."""
        return {
            'articulation': {
                'robot': {
                    'root_pose': torch.randn(self.num_envs, 7, device=self.device),
                    'root_velocity': torch.randn(self.num_envs, 6, device=self.device),
                    'joint_position': torch.randn(self.num_envs, 6, device=self.device),
                    'joint_velocity': torch.randn(self.num_envs, 6, device=self.device)
                },
                'object': {
                    'root_pose': torch.randn(self.num_envs, 7, device=self.device),
                    'root_velocity': torch.randn(self.num_envs, 6, device=self.device),
                }
            }
        }


class MockBaseEnv(gym.Env):
    """Mock base environment class that inherits from gym.Env."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or MockEnvConfig()
        self.num_envs = getattr(self.cfg, 'num_envs', 4)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(64,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))
        self.device = torch.device("cpu")
        self.max_episode_length = 500

        # Add wrapper integration support
        self._unwrapped = self

        # Add scene for efficient reset wrapper
        self.scene = MockScene(self.device, self.num_envs)

        # Add robot for wrapper initialization detection
        self._robot = True

        # Add force-torque data for force torque wrapper and fragile object wrapper
        self.robot_force_torque = torch.randn(self.num_envs, 6, device=self.device)

        # Add joint efforts for force-torque sensor wrapper
        self._joint_efforts = torch.randn(self.num_envs, 7, device=self.device)

        # For force-torque wrapper tanh scaling
        self.force_torque_tanh_scaling = 1.0

        # Add observation data for observation wrappers
        self._setup_observation_data()

        # Add actions attribute for dynamic observation calculation
        self.actions = torch.randn(self.num_envs, 6, device=self.device)

        # Add extras dict for wandb wrapper integration
        self.extras = {}

        # Add episode tracking for wandb wrapper
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    @property
    def unwrapped(self):
        return self._unwrapped

    @unwrapped.setter
    def unwrapped(self, value):
        self._unwrapped = value

    def add_metrics(self, metrics_dict):
        """Mock add_metrics method for wrapper integration."""
        pass

    def _reset_idx(self, env_ids):
        """Mock reset for specific environment indices."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # Simulate some state changes
        self.robot_force_torque = torch.randn(self.num_envs, 6, device=self.device)

    def _get_dones(self):
        """Mock getting done states."""
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, time_out

    def reset(self, seed=None, **kwargs):
        obs = torch.randn(self.num_envs, 64)
        info = {"timeout": torch.zeros(self.num_envs, dtype=torch.bool)}
        return obs, info

    def step(self, actions):
        # Update force-torque data
        self.robot_force_torque = torch.randn(self.num_envs, 6, device=self.device)

        obs = torch.randn(self.num_envs, 64)
        rewards = torch.randn(self.num_envs)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        info = {"timeout": truncated}
        return obs, rewards, terminated, truncated, info

    def _setup_observation_data(self):
        """Setup observation data for observation wrapper testing."""
        # Observation components for history wrapper
        self.fingertip_pos = torch.randn(self.num_envs, 3, device=self.device)
        self.ee_linvel = torch.randn(self.num_envs, 3, device=self.device)
        self.ee_angvel = torch.randn(self.num_envs, 3, device=self.device)
        self.joint_pos = torch.randn(self.num_envs, 7, device=self.device)
        self.fingertip_quat = torch.randn(self.num_envs, 4, device=self.device)

        # For relative position calculations
        self.fingertip_midpoint_pos = torch.randn(self.num_envs, 3, device=self.device)
        self.held_pos = torch.randn(self.num_envs, 3, device=self.device)
        self.fixed_pos_obs_frame = torch.randn(self.num_envs, 3, device=self.device)
        self.init_fixed_pos_obs_noise = torch.randn(self.num_envs, 3, device=self.device) * 0.01

    def _get_observations(self):
        """Mock get observations method for observation manager wrapper."""
        # Return Isaac Lab factory format: {"policy": tensor, "critic": tensor}
        policy_obs = torch.randn(self.num_envs, 32, device=self.device)
        critic_obs = torch.randn(self.num_envs, 48, device=self.device)
        return {"policy": policy_obs, "critic": critic_obs}

    def _get_factory_obs_state_dict(self):
        """Mock factory observation state dict method for force torque wrapper."""
        # Return Isaac Lab factory format: obs_dict, state_dict
        obs_dict = {
            'fingertip_pos': self.fingertip_pos,
            'joint_pos': self.joint_pos,
            'ee_linvel': self.ee_linvel,
        }
        state_dict = {
            'fingertip_pos': self.fingertip_pos,
            'joint_pos': self.joint_pos,
            'fingertip_quat': self.fingertip_quat,
        }
        return obs_dict, state_dict

    def _pre_physics_step(self, actions):
        """Mock pre-physics step for history wrapper."""
        # Update observation data to simulate environment changes
        self.fingertip_pos += torch.randn_like(self.fingertip_pos) * 0.01
        self.ee_linvel += torch.randn_like(self.ee_linvel) * 0.01
        self.ee_angvel += torch.randn_like(self.ee_angvel) * 0.01

    def _init_tensors(self):
        """Mock init tensors method for force torque wrapper."""
        # Initialize force-torque sensor data
        self._joint_efforts = torch.randn(self.num_envs, 7, device=self.device)

    def _compute_intermediate_values(self, dt=None):
        """Mock compute intermediate values for force torque wrapper."""
        # Update joint efforts to simulate sensor readings
        self._joint_efforts = torch.randn(self.num_envs, 7, device=self.device)

        # Convert joint efforts to force-torque sensor data
        # Typically end-effector force-torque is last 6 DOF of joint efforts
        self.robot_force_torque = self._joint_efforts[:, -6:]

    def close(self):
        pass


from dataclasses import dataclass

configclass = dataclass

# Mock Isaac Lab's native dimension configurations (single source of truth)
OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,  # Joint positions can be in observations
    "held_pos": 3,   # Object positions can be in observations
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
}

# Separate configs for tests that need force_torque
OBS_DIM_CFG_WITH_FORCE = {**OBS_DIM_CFG, "force_torque": 6}
STATE_DIM_CFG_WITH_FORCE = {**STATE_DIM_CFG, "force_torque": 6}


# Environment creation function
def make_env(task_name, **kwargs):
    """Mock environment creation function."""
    return MockIsaacLabEnv()


# Mock configuration classes
class MockEnvConfig:
    def __init__(self):
        self.num_envs = 4
        self.episode_length_s = 12.0
        self.decimation = 8
        # Add scene and other required attributes
        self.scene = MockScene("cpu", self.num_envs)
        self.task = MockTask()
        self.ctrl = MockCtrl()

        # Observation configuration for observation wrappers
        self.obs_order = ["fingertip_pos", "ee_linvel", "joint_pos", "force_torque"]
        self.state_order = ["fingertip_pos", "ee_linvel", "joint_pos", "fingertip_quat", "force_torque"]
        self.action_space = 6  # Action space dimensions (following Isaac Lab's integer pattern)

        # These will be calculated dynamically by the wrapper, but set for compatibility
        # fingertip_pos(3) + ee_linvel(3) + joint_pos(7) + force_torque(6) + actions(6) = 25
        self.observation_space = 25
        # fingertip_pos(3) + ee_linvel(3) + joint_pos(7) + fingertip_quat(4) + force_torque(6) + actions(6) = 29
        self.state_space = 29

        # Mark this as a test config to enable test-specific behavior
        self._is_mock_test_config = True

        # Component dimensions now come from Isaac Lab's native OBS_DIM_CFG/STATE_DIM_CFG

        # Component attribute mapping for history wrapper
        self.component_attr_map = {
            "fingertip_pos": "fingertip_pos",
            "ee_linvel": "ee_linvel",
            "ee_angvel": "ee_angvel",
            "joint_pos": "joint_pos",
            "fingertip_quat": "fingertip_quat",
            "force_torque": "robot_force_torque"
        }

        # History configuration
        self.history_samples = 4


# Additional mock classes for integration tests
class MockTask:
    """Mock task configuration."""
    def __init__(self):
        # Isaac Lab required attributes
        self.duration_s = 10.0
        self.fixed_asset = "mock_fixed_asset"
        self.held_asset = "mock_held_asset"

        # Custom attributes (for testing overrides)
        self.success_threshold = 0.02
        self.engage_threshold = 0.05
        self.name = "factory_task"


class MockCtrl:
    """Mock control configuration."""
    def __init__(self):
        self.pos_action_bounds = [0.05, 0.05, 0.05]
        self.force_action_bounds = [50.0, 50.0, 50.0]
        self.torque_action_bounds = [0.5, 0.5, 0.5]

        # Additional attributes needed for hybrid control
        self.default_task_prop_gains = [1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0]
        self.pos_action_threshold = [0.05, 0.05, 0.05]
        self.rot_action_threshold = [0.1, 0.1, 0.1]
        self.default_task_force_gains = [0.1, 0.1, 0.1, 0.001, 0.001, 0.001]
        self.force_action_threshold = [10.0, 10.0, 10.0]
        self.torque_action_threshold = [1.0, 1.0, 1.0]

    def get(self, key, default=None):
        """Support dict-like access."""
        return getattr(self, key, default)


# Mock Isaac Lab modules structure
class isaac_lab:
    class envs:
        BaseEnv = MockBaseEnv
        EnvCfg = MockEnvConfig

    class utils:
        class math:
            @staticmethod
            def axis_angle_from_quat(quat):
                # Return proper tensor instead of calling the function recursively
                if hasattr(quat, 'shape'):
                    return torch.zeros(quat.shape[0], 3, device=quat.device if hasattr(quat, 'device') else 'cpu')
                else:
                    return torch.zeros(1, 3)

        configclass = configclass

    @staticmethod
    def make_env(task_name, **kwargs):
        return MockIsaacLabEnv()


# Mock Isaac Sim components for force-torque wrapper
class omni:
    class isaac:
        class core:
            class utils:
                class prims:
                    class articulation_view:
                        ArticulationView = MockRobotView

        class sensor:
            class utils:
                RobotView = MockRobotView

# For compatibility with different import patterns
try:
    from omni.isaac.sensor.utils import RobotView
except ImportError:
    RobotView = MockRobotView


class DirectRLEnv(MockBaseEnv):
    """Mock DirectRLEnv class for efficient reset wrapper testing."""

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def _reset_idx(self, env_ids):
        """Lightweight DirectRLEnv reset implementation."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # Lightweight reset - just update force-torque data
        self.robot_force_torque = torch.randn(self.num_envs, 6, device=self.device)


# Mock the direct_rl_env module
class direct_rl_env:
    DirectRLEnv = DirectRLEnv


# Make it available as module attribute
envs = isaac_lab.envs
utils = isaac_lab.utils
make_env = isaac_lab.make_env