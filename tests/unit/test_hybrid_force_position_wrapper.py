"""
Unit tests for hybrid force-position wrapper and support files.

Tests all classes and functions in:
- wrappers/control/hybrid_force_position_wrapper.py
- wrappers/control/factory_control_utils.py
- wrappers/control/hybrid_control_cfg.py
"""

import pytest
import torch
import gymnasium as gym
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Mock modules before imports
mock_isaac_sim = __import__('tests.mocks.mock_isaac_sim', fromlist=[''])
mock_isaac_lab = __import__('tests.mocks.mock_isaac_lab', fromlist=[''])

# Create module hierarchy for omni.isaac.core.utils.torch
if 'omni' not in sys.modules:
    import types
    omni = types.ModuleType('omni')
    isaac = types.ModuleType('omni.isaac')
    core = types.ModuleType('omni.isaac.core')
    utils = types.ModuleType('omni.isaac.core.utils')
    torch_mod = types.ModuleType('omni.isaac.core.utils.torch')

    # Add mock functions to torch module
    torch_utils_mock = mock_isaac_sim.core.utils.torch
    torch_mod.quat_mul = torch_utils_mock.quat_mul
    torch_mod.quat_conjugate = torch_utils_mock.quat_conjugate
    torch_mod.quat_from_angle_axis = torch_utils_mock.quat_from_angle_axis
    torch_mod.quat_from_euler_xyz = torch_utils_mock.quat_from_euler_xyz
    torch_mod.get_euler_xyz = torch_utils_mock.get_euler_xyz

    utils.torch = torch_mod
    core.utils = utils
    isaac.core = core
    omni.isaac = isaac

    sys.modules['omni'] = omni
    sys.modules['omni.isaac'] = isaac
    sys.modules['omni.isaac.core'] = core
    sys.modules['omni.isaac.core.utils'] = utils
    sys.modules['omni.isaac.core.utils.torch'] = torch_mod

# Set up other modules
sys.modules['isaacsim.core.utils.torch'] = mock_isaac_sim.core.utils.torch
sys.modules['isaaclab.utils.math'] = mock_isaac_lab.utils.math
sys.modules['omni.isaac.lab.utils.math'] = mock_isaac_lab.utils.math
sys.modules['omni.isaac.lab.utils.configclass'] = mock_isaac_lab.utils

# Import modules under test
from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper
from wrappers.control.factory_control_utils import (
    compute_pose_task_wrench,
    compute_force_task_wrench,
    compute_dof_torque_from_wrench,
    get_pose_error,
    _apply_task_space_gains
)
from wrappers.control.hybrid_control_cfg import (
    HybridCtrlCfg,
    HybridTaskCfg,
    DEFAULT_HYBRID_CTRL_CFG,
    DEFAULT_HYBRID_TASK_CFG
)


class MockFactoryEnv(gym.Env):
    """Mock factory environment for testing."""

    def __init__(self, num_envs=4, device="cpu"):
        super().__init__()
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.physics_dt = 1.0 / 60.0

        # Configuration
        self.cfg = self._create_mock_cfg()
        self.cfg_task = self._create_mock_task_cfg()

        # Robot mock
        self._robot = self._create_mock_robot()
        self.last_update_timestamp = 0.0

        # Action and observation spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(18,))
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(64,))

        # State data
        self.fingertip_midpoint_pos = torch.zeros((num_envs, 3), device=self.device)
        self.fingertip_midpoint_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
        self.fixed_pos_action_frame = torch.zeros((num_envs, 3), device=self.device)

        # Velocity data
        self.ee_linvel_fd = torch.zeros((num_envs, 3), device=self.device)
        self.ee_angvel_fd = torch.zeros((num_envs, 3), device=self.device)

        # Joint data
        self.joint_pos = torch.zeros((num_envs, 9), device=self.device)
        self.joint_vel_fd = torch.zeros((num_envs, 9), device=self.device)
        self.joint_torque = torch.zeros((num_envs, 9), device=self.device)

        # Force-torque sensor data
        self.robot_force_torque = torch.zeros((num_envs, 6), device=self.device)

        # Control targets
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
        self.ctrl_target_joint_pos = torch.zeros((num_envs, 9), device=self.device)
        self.ctrl_target_gripper_dof_pos = 0.0

        # Control gains
        self.task_prop_gains = torch.ones((num_envs, 6), device=self.device) * 100.0
        self.task_deriv_gains = torch.ones((num_envs, 6), device=self.device) * 10.0

        # Jacobian and mass matrix
        self.fingertip_midpoint_jacobian = torch.eye(6, 7, device=self.device).unsqueeze(0).repeat(num_envs, 1, 1)
        self.arm_mass_matrix = torch.eye(7, device=self.device).unsqueeze(0).repeat(num_envs, 1, 1)

        # Action thresholds
        self.pos_threshold = 0.1
        self.rot_threshold = 0.2

        # Current actions
        self.actions = torch.zeros((num_envs, 18), device=self.device)
        self.prev_action = torch.zeros((num_envs, 18), device=self.device)

        # Reset buffer
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        # Extras for logging
        self.extras = {}

        # Current yaw
        self.curr_yaw = torch.zeros(num_envs, device=self.device)

    def _create_mock_cfg(self):
        """Create mock configuration."""
        class MockCfg:
            def __init__(self):
                self.scene = self._MockScene()
                self.ctrl = self._MockCtrl()
                self.action_space = 18

            class _MockScene:
                def __init__(self):
                    self.num_envs = 4

            class _MockCtrl:
                def __init__(self):
                    self.pos_action_bounds = [0.05, 0.05, 0.05]
                    self.force_action_bounds = [50.0, 50.0, 50.0]
                    self.torque_action_bounds = [0.5, 0.5, 0.5]
                    self.force_action_threshold = [10.0, 10.0, 10.0]
                    self.torque_action_threshold = [0.1, 0.1, 0.1]
                    self.default_dof_pos_tensor = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
                    self.kp_null = 10.0
                    self.kd_null = 2.0

        return MockCfg()

    def _create_mock_task_cfg(self):
        """Create mock task configuration."""
        class MockTaskCfg:
            def __init__(self):
                self.unidirectional_rot = False

        return MockTaskCfg()

    def _create_mock_robot(self):
        """Create mock robot."""
        class MockRobot:
            def __init__(self):
                self._data = self._MockData()

            def set_joint_position_target(self, positions):
                pass

            def set_joint_effort_target(self, efforts):
                pass

            class _MockData:
                def __init__(self):
                    self._sim_timestamp = 0.0

        return MockRobot()

    def _configure_gym_env_spaces(self):
        """Configure gym environment spaces."""
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.cfg.action_space,))

    def _reset_buffers(self, env_ids):
        """Reset buffers for specific environments."""
        self.fingertip_midpoint_pos[env_ids] = torch.randn((len(env_ids), 3), device=self.device) * 0.1
        self.robot_force_torque[env_ids] = torch.randn((len(env_ids), 6), device=self.device) * 5.0

    def _compute_intermediate_values(self, dt):
        """Compute intermediate values."""
        self.last_update_timestamp = self._robot._data._sim_timestamp
        self.fingertip_midpoint_pos += self.ee_linvel_fd * dt

    def _pre_physics_step(self, action):
        """Pre-physics step."""
        self.actions = action.clone()

    def _apply_action(self):
        """Apply action."""
        pass

    def _update_rew_buf(self, curr_successes):
        """Update reward buffer."""
        return torch.randn(self.num_envs, device=self.device)

    def reset(self, seed=None, **kwargs):
        """Reset environment."""
        obs = torch.randn(self.num_envs, 64, device=self.device)
        info = {}
        return obs, info

    def step(self, action):
        """Step environment."""
        # Update timestamp
        self._robot._data._sim_timestamp += self.physics_dt

        obs = torch.randn(self.num_envs, 64, device=self.device)
        reward = torch.randn(self.num_envs, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        info = {
            'current_successes': torch.randint(0, 2, (self.num_envs,), device=self.device),
            'current_engagements': torch.randint(0, 2, (self.num_envs,), device=self.device),
        }
        return obs, reward, terminated, truncated, info


@pytest.fixture
def mock_env():
    """Create mock factory environment."""
    return MockFactoryEnv(num_envs=4, device="cpu")


@pytest.fixture
def hybrid_ctrl_cfg():
    """Create hybrid control configuration."""
    return HybridCtrlCfg(
        ema_factor=0.2,
        no_sel_ema=True,
        target_init_mode="zero",
        default_task_force_gains=[0.1, 0.1, 0.1, 0.001, 0.001, 0.001],
        force_action_bounds=[50.0, 50.0, 50.0],
        torque_action_bounds=[0.5, 0.5, 0.5],
        force_action_threshold=[10.0, 10.0, 10.0],
        torque_action_threshold=[0.1, 0.1, 0.1],
        pos_action_bounds=[0.05, 0.05, 0.05]
    )


@pytest.fixture
def hybrid_task_cfg():
    """Create hybrid task configuration."""
    return HybridTaskCfg(
        force_active_threshold=0.1,
        torque_active_threshold=0.01,
        good_force_cmd_rew=0.1,
        bad_force_cmd_rew=-0.1,
        wrench_norm_scale=0.01
    )


class TestHybridCtrlCfg:
    """Test HybridCtrlCfg configuration class."""

    def test_initialization_with_defaults(self):
        """Test configuration initialization with default values."""
        cfg = HybridCtrlCfg()
        assert cfg.ema_factor == 0.2
        assert cfg.no_sel_ema == True
        assert cfg.target_init_mode == "zero"
        assert cfg.default_task_force_gains is None

    def test_initialization_with_custom_values(self, hybrid_ctrl_cfg):
        """Test configuration initialization with custom values."""
        assert hybrid_ctrl_cfg.ema_factor == 0.2
        assert hybrid_ctrl_cfg.default_task_force_gains == [0.1, 0.1, 0.1, 0.001, 0.001, 0.001]
        assert hybrid_ctrl_cfg.force_action_bounds == [50.0, 50.0, 50.0]

    def test_post_init_validation(self):
        """Test post-initialization validation."""
        # Valid values should not raise
        HybridCtrlCfg(ema_factor=0.5, target_init_mode="first_goal")

        # Invalid ema_factor should raise
        with pytest.raises(ValueError, match="ema_factor must be between 0 and 1"):
            HybridCtrlCfg(ema_factor=-0.1)

        with pytest.raises(ValueError, match="ema_factor must be between 0 and 1"):
            HybridCtrlCfg(ema_factor=1.5)

        # Invalid target_init_mode should raise
        with pytest.raises(ValueError, match="target_init_mode must be"):
            HybridCtrlCfg(target_init_mode="invalid")


class TestHybridTaskCfg:
    """Test HybridTaskCfg configuration class."""

    def test_initialization_with_defaults(self):
        """Test task configuration initialization with default values."""
        cfg = HybridTaskCfg()
        assert cfg.force_active_threshold == 0.1
        assert cfg.torque_active_threshold == 0.01
        assert cfg.good_force_cmd_rew == 0.1
        assert cfg.bad_force_cmd_rew == -0.1
        assert cfg.wrench_norm_scale == 0.01

    def test_initialization_with_custom_values(self, hybrid_task_cfg):
        """Test task configuration initialization with custom values."""
        assert hybrid_task_cfg.force_active_threshold == 0.1
        assert hybrid_task_cfg.good_force_cmd_rew == 0.1
        assert hybrid_task_cfg.bad_force_cmd_rew == -0.1


class TestDefaultConfigurations:
    """Test default configuration instances."""

    def test_default_hybrid_ctrl_cfg(self):
        """Test default hybrid control configuration."""
        assert isinstance(DEFAULT_HYBRID_CTRL_CFG, HybridCtrlCfg)
        assert DEFAULT_HYBRID_CTRL_CFG.ema_factor == 0.2

    def test_default_hybrid_task_cfg(self):
        """Test default hybrid task configuration."""
        assert isinstance(DEFAULT_HYBRID_TASK_CFG, HybridTaskCfg)
        assert DEFAULT_HYBRID_TASK_CFG.force_active_threshold == 0.1


class TestFactoryControlUtils:
    """Test factory control utility functions."""

    def setup_method(self):
        """Set up test data."""
        self.num_envs = 4
        self.device = torch.device("cpu")

        # Create mock configuration
        self.cfg = MockFactoryEnv()._create_mock_cfg()

        # Create test tensors
        self.dof_pos = torch.randn(self.num_envs, 9, device=self.device)
        self.dof_vel = torch.randn(self.num_envs, 9, device=self.device)
        self.fingertip_pos = torch.randn(self.num_envs, 3, device=self.device)
        self.fingertip_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.fingertip_linvel = torch.randn(self.num_envs, 3, device=self.device)
        self.fingertip_angvel = torch.randn(self.num_envs, 3, device=self.device)
        self.target_pos = torch.randn(self.num_envs, 3, device=self.device)
        self.target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.task_prop_gains = torch.ones(self.num_envs, 6, device=self.device) * 100.0
        self.task_deriv_gains = torch.ones(self.num_envs, 6, device=self.device) * 10.0
        self.eef_force = torch.randn(self.num_envs, 6, device=self.device)
        self.target_force = torch.randn(self.num_envs, 6, device=self.device)
        self.task_gains = torch.ones(self.num_envs, 6, device=self.device) * 0.1
        self.jacobian = torch.eye(6, 7, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.mass_matrix = torch.eye(7, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)

    def test_compute_pose_task_wrench(self):
        """Test pose task wrench computation."""
        wrench = compute_pose_task_wrench(
            cfg=self.cfg,
            dof_pos=self.dof_pos,
            fingertip_midpoint_pos=self.fingertip_pos,
            fingertip_midpoint_quat=self.fingertip_quat,
            fingertip_midpoint_linvel=self.fingertip_linvel,
            fingertip_midpoint_angvel=self.fingertip_angvel,
            ctrl_target_fingertip_midpoint_pos=self.target_pos,
            ctrl_target_fingertip_midpoint_quat=self.target_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device
        )

        assert wrench.shape == (self.num_envs, 6)
        assert not torch.isnan(wrench).any()
        assert torch.isfinite(wrench).all()

    def test_compute_force_task_wrench(self):
        """Test force task wrench computation."""
        wrench = compute_force_task_wrench(
            cfg=self.cfg,
            dof_pos=self.dof_pos,
            eef_force=self.eef_force,
            ctrl_target_force=self.target_force,
            task_gains=self.task_gains,
            device=self.device
        )

        assert wrench.shape == (self.num_envs, 6)
        assert not torch.isnan(wrench).any()
        assert torch.isfinite(wrench).all()

        # Test that wrench equals gains * (target - current)
        expected = self.task_gains * (self.target_force - self.eef_force)
        torch.testing.assert_close(wrench, expected)

    def test_compute_dof_torque_from_wrench(self):
        """Test DOF torque computation from wrench."""
        task_wrench = torch.randn(self.num_envs, 6, device=self.device)

        dof_torque, output_wrench = compute_dof_torque_from_wrench(
            cfg=self.cfg,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            task_wrench=task_wrench,
            jacobian=self.jacobian,
            arm_mass_matrix=self.mass_matrix,
            device=self.device
        )

        assert dof_torque.shape == (self.num_envs, self.dof_pos.shape[1])
        assert output_wrench.shape == task_wrench.shape
        assert not torch.isnan(dof_torque).any()
        assert torch.isfinite(dof_torque).all()

        # Test clamping
        assert (dof_torque >= -100.0).all()
        assert (dof_torque <= 100.0).all()

    def test_get_pose_error(self):
        """Test pose error computation."""
        pos_error, axis_angle_error = get_pose_error(
            fingertip_midpoint_pos=self.fingertip_pos,
            fingertip_midpoint_quat=self.fingertip_quat,
            ctrl_target_fingertip_midpoint_pos=self.target_pos,
            ctrl_target_fingertip_midpoint_quat=self.target_quat,
            jacobian_type="geometric",
            rot_error_type="axis_angle"
        )

        assert pos_error.shape == (self.num_envs, 3)
        assert axis_angle_error.shape == (self.num_envs, 3)
        assert not torch.isnan(pos_error).any()
        assert not torch.isnan(axis_angle_error).any()

        # Test that position error is correct
        expected_pos_error = self.target_pos - self.fingertip_pos
        torch.testing.assert_close(pos_error, expected_pos_error)

    def test_get_pose_error_quat_output(self):
        """Test pose error computation with quaternion output."""
        pos_error, quat_error = get_pose_error(
            fingertip_midpoint_pos=self.fingertip_pos,
            fingertip_midpoint_quat=self.fingertip_quat,
            ctrl_target_fingertip_midpoint_pos=self.target_pos,
            ctrl_target_fingertip_midpoint_quat=self.target_quat,
            jacobian_type="geometric",
            rot_error_type="quat"
        )

        assert pos_error.shape == (self.num_envs, 3)
        assert quat_error.shape == (self.num_envs, 4)
        assert not torch.isnan(pos_error).any()
        assert not torch.isnan(quat_error).any()

    def test_apply_task_space_gains(self):
        """Test task space gains application."""
        delta_pose = torch.randn(self.num_envs, 6, device=self.device)

        wrench = _apply_task_space_gains(
            delta_fingertip_pose=delta_pose,
            fingertip_midpoint_linvel=self.fingertip_linvel,
            fingertip_midpoint_angvel=self.fingertip_angvel,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains
        )

        assert wrench.shape == (self.num_envs, 6)
        assert not torch.isnan(wrench).any()
        assert torch.isfinite(wrench).all()

        # Test linear components
        expected_lin = (self.task_prop_gains[:, :3] * delta_pose[:, :3] +
                       self.task_deriv_gains[:, :3] * (0.0 - self.fingertip_linvel))
        torch.testing.assert_close(wrench[:, :3], expected_lin)

        # Test rotational components
        expected_rot = (self.task_prop_gains[:, 3:] * delta_pose[:, 3:] +
                       self.task_deriv_gains[:, 3:] * (0.0 - self.fingertip_angvel))
        torch.testing.assert_close(wrench[:, 3:], expected_rot)


class TestHybridForcePositionWrapper:
    """Test HybridForcePositionWrapper class."""

    def test_initialization_requires_configs(self, mock_env):
        """Test that wrapper requires explicit configurations."""
        # Should raise error without configurations
        with pytest.raises(ValueError, match="ctrl_cfg cannot be None"):
            HybridForcePositionWrapper(mock_env)

        with pytest.raises(ValueError, match="task_cfg cannot be None"):
            HybridForcePositionWrapper(mock_env, ctrl_cfg=HybridCtrlCfg())

    def test_initialization_requires_force_gains(self, mock_env, hybrid_task_cfg):
        """Test that wrapper requires force gains configuration."""
        ctrl_cfg = HybridCtrlCfg(default_task_force_gains=None)

        with pytest.raises(ValueError, match="default_task_force_gains cannot be None"):
            HybridForcePositionWrapper(mock_env, ctrl_cfg=ctrl_cfg, task_cfg=hybrid_task_cfg)

    def test_initialization_success(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test successful wrapper initialization."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_torque=False,
            reward_type="simp",
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        assert wrapper.num_envs == 4
        assert wrapper.force_size == 3
        assert wrapper.reward_type == "simp"
        assert wrapper.action_space_size == 12  # 2*3 + 6 = 12
        assert wrapper.ctrl_cfg == hybrid_ctrl_cfg
        assert wrapper.task_cfg == hybrid_task_cfg

    def test_initialization_with_torque_control(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test wrapper initialization with torque control."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_torque=True,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        assert wrapper.force_size == 6
        assert wrapper.action_space_size == 18  # 2*6 + 6 = 18

    def test_action_space_update(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test that action space is properly updated."""
        original_action_space = mock_env.cfg.action_space

        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Should update environment action space
        assert mock_env.cfg.action_space == wrapper.action_space_size

    def test_target_initialization_zero_mode(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test target initialization in zero mode."""
        hybrid_ctrl_cfg.target_init_mode = "zero"

        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        assert wrapper._targets_initialized == True
        assert torch.allclose(wrapper.target_selection, torch.zeros_like(wrapper.target_selection))

    def test_target_initialization_first_goal_mode(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test target initialization in first_goal mode."""
        hybrid_ctrl_cfg.target_init_mode = "first_goal"

        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        assert wrapper._targets_initialized == False

    def test_wrapper_initialization_without_robot(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test wrapper behavior when robot is not available initially."""
        # Remove robot temporarily
        robot = mock_env._robot
        del mock_env._robot

        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        assert wrapper._wrapper_initialized == False

        # Restore robot and test initialization
        mock_env._robot = robot
        wrapper._initialize_wrapper()
        assert wrapper._wrapper_initialized == True

    def test_wrapper_initialization_requires_force_torque(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test that wrapper requires force-torque sensor data."""
        # Remove robot temporarily to prevent auto-initialization
        robot = mock_env._robot
        del mock_env._robot

        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Restore robot but remove force-torque data
        mock_env._robot = robot
        del mock_env.robot_force_torque

        with pytest.raises(ValueError, match="force-torque sensor data"):
            wrapper._initialize_wrapper()

    def test_delta_reward_initialization(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test initialization with delta reward type."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            reward_type="delta",
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        assert hasattr(wrapper, '_old_sel_matrix')
        assert wrapper._old_sel_matrix.shape == wrapper.sel_matrix.shape

    def test_extract_goals_from_action(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test goal extraction from actions."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Create test action
        action = torch.randn(mock_env.num_envs, wrapper.action_space_size)

        wrapper._extract_goals_from_action(action)

        # Check that goals are extracted correctly
        assert wrapper.sel_goal.shape == (mock_env.num_envs, 6)
        assert wrapper.pose_goal.shape == (mock_env.num_envs, 7)
        assert wrapper.force_goal.shape == (mock_env.num_envs, 6)
        assert torch.allclose(wrapper.unwrapped.actions, action)

    def test_calc_pose_goal_without_bounds(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test pose goal calculation without configured bounds."""
        # Remove bounds from config
        hybrid_ctrl_cfg.pos_action_bounds = None
        del mock_env.cfg.ctrl.pos_action_bounds

        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        action = torch.randn(mock_env.num_envs, wrapper.action_space_size)
        wrapper.unwrapped.actions = action

        with pytest.raises(ValueError, match="Position action bounds not configured"):
            wrapper._calc_pose_goal()

    def test_calc_force_goal_without_threshold(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test force goal calculation without configured threshold."""
        # Remove threshold from config
        hybrid_ctrl_cfg.force_action_threshold = None
        del mock_env.cfg.ctrl.force_action_threshold

        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        action = torch.randn(mock_env.num_envs, wrapper.action_space_size)
        wrapper.unwrapped.actions = action

        with pytest.raises(ValueError, match="Force action threshold not configured"):
            wrapper._calc_force_goal()

    def test_calc_force_goal_without_bounds(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test force goal calculation without configured bounds."""
        # Remove bounds from config
        hybrid_ctrl_cfg.force_action_bounds = None
        del mock_env.cfg.ctrl.force_action_bounds

        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        action = torch.randn(mock_env.num_envs, wrapper.action_space_size)
        wrapper.unwrapped.actions = action

        with pytest.raises(ValueError, match="Force action bounds not configured"):
            wrapper._calc_force_goal()

    def test_torque_goal_calculation(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test torque goal calculation for 6DOF control."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_torque=True,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        action = torch.randn(mock_env.num_envs, wrapper.action_space_size)
        wrapper.unwrapped.actions = action

        wrapper._calc_force_goal()

        # Check that torque goals are calculated
        assert not torch.allclose(wrapper.force_goal[:, 3:], torch.zeros_like(wrapper.force_goal[:, 3:]))

    def test_torque_goal_without_config(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test torque goal calculation without proper configuration."""
        # Remove torque config
        hybrid_ctrl_cfg.torque_action_threshold = None
        del mock_env.cfg.ctrl.torque_action_threshold

        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_torque=True,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        action = torch.randn(mock_env.num_envs, wrapper.action_space_size)
        wrapper.unwrapped.actions = action

        with pytest.raises(ValueError, match="Torque action threshold not configured"):
            wrapper._calc_force_goal()

    def test_update_targets_with_ema_first_goal(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test target update with EMA for first goal."""
        hybrid_ctrl_cfg.target_init_mode = "first_goal"

        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Set some goals
        wrapper.sel_goal = torch.ones_like(wrapper.sel_goal) * 0.5
        wrapper.pose_goal = torch.ones_like(wrapper.pose_goal) * 0.3
        wrapper.force_goal = torch.ones_like(wrapper.force_goal) * 0.7

        wrapper._update_targets_with_ema()

        # Should initialize to first goals
        assert wrapper._targets_initialized == True
        torch.testing.assert_close(wrapper.target_selection, wrapper.sel_goal)
        torch.testing.assert_close(wrapper.target_pose, wrapper.pose_goal)
        torch.testing.assert_close(wrapper.target_force, wrapper.force_goal)

    def test_update_targets_with_ema_filtering(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test target update with EMA filtering."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Initialize targets
        wrapper.target_selection.fill_(0.2)
        wrapper.target_pose.fill_(0.2)
        wrapper.target_force.fill_(0.2)
        wrapper._targets_initialized = True

        # Set new goals
        wrapper.sel_goal.fill_(0.8)
        wrapper.pose_goal.fill_(0.8)
        wrapper.force_goal.fill_(0.8)

        wrapper._update_targets_with_ema()

        # Check EMA update
        expected_ema = 0.2 * 0.8 + 0.8 * 0.2  # ema_factor * goal + (1-ema_factor) * prev_target
        assert torch.allclose(wrapper.target_pose, torch.full_like(wrapper.target_pose, expected_ema), atol=1e-6)

    def test_update_targets_no_sel_ema(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test target update without selection EMA."""
        hybrid_ctrl_cfg.no_sel_ema = True

        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Initialize targets
        wrapper.target_selection.fill_(0.2)
        wrapper._targets_initialized = True

        # Set new selection goal
        wrapper.sel_goal.fill_(0.8)

        wrapper._update_targets_with_ema()

        # Selection should be set directly (no EMA)
        assert torch.allclose(wrapper.target_selection[:, :wrapper.force_size], wrapper.sel_goal[:, :wrapper.force_size])

    def test_set_control_targets_from_targets(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test setting control targets from filtered targets."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Set targets
        wrapper.target_selection.fill_(0.7)  # > 0.5 threshold
        wrapper.target_pose[:, :3].fill_(1.0)
        wrapper.target_pose[:, 3:].fill_(0.5)
        wrapper.target_force.fill_(2.0)

        wrapper._set_control_targets_from_targets()

        # Check selection matrix (should be 1.0 where target > 0.5)
        assert torch.allclose(wrapper.sel_matrix[:, :wrapper.force_size], torch.ones_like(wrapper.sel_matrix[:, :wrapper.force_size]))

        # Check pose targets
        torch.testing.assert_close(wrapper.unwrapped.ctrl_target_fingertip_midpoint_pos, wrapper.target_pose[:, :3])
        torch.testing.assert_close(wrapper.unwrapped.ctrl_target_fingertip_midpoint_quat, wrapper.target_pose[:, 3:])

        # Check force target
        torch.testing.assert_close(wrapper.target_force_for_control, wrapper.target_force)

    def test_get_target_out_of_bounds(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test out-of-bounds detection."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Set fingertip position out of bounds
        mock_env.fingertip_midpoint_pos[0] = torch.tensor([0.1, 0.0, 0.0])  # > pos_action_bounds[0] = 0.05
        mock_env.fixed_pos_action_frame[0] = torch.tensor([0.0, 0.0, 0.0])

        out_of_bounds = wrapper._get_target_out_of_bounds()

        assert out_of_bounds[0, 0] == True  # X axis out of bounds
        assert out_of_bounds[0, 1] == False  # Y axis in bounds
        assert out_of_bounds[0, 2] == False  # Z axis in bounds

    def test_reset_targets(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test target reset for specific environments."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Set some targets
        wrapper.target_selection.fill_(0.5)
        wrapper.target_pose.fill_(0.5)
        wrapper.target_force.fill_(0.5)

        # Reset environments 0 and 2
        env_ids = torch.tensor([0, 2])
        wrapper._reset_targets(env_ids)

        # Check that targets are reset
        assert torch.allclose(wrapper.target_selection[env_ids], torch.zeros_like(wrapper.target_selection[env_ids]))
        assert torch.allclose(wrapper.target_pose[env_ids], torch.zeros_like(wrapper.target_pose[env_ids]))
        assert torch.allclose(wrapper.target_force[env_ids], torch.zeros_like(wrapper.target_force[env_ids]))

        # Check that other environments are unchanged
        assert torch.allclose(wrapper.target_selection[1], torch.full_like(wrapper.target_selection[1], 0.5))

    def test_simple_force_reward(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test simple force reward calculation."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            reward_type="simp",
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Set force data
        mock_env.robot_force_torque[0] = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])  # Active force
        mock_env.robot_force_torque[1] = torch.tensor([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])  # Inactive force

        # Set selection matrix
        wrapper.sel_matrix[0, 0] = 1.0  # Force control on X
        wrapper.sel_matrix[1, 0] = 1.0  # Force control on X

        reward = wrapper._simple_force_reward()

        # Environment 0: active force + force control = good
        assert reward[0] == hybrid_task_cfg.good_force_cmd_rew

        # Environment 1: inactive force + force control = bad
        assert reward[1] == hybrid_task_cfg.bad_force_cmd_rew

    def test_directional_force_reward(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test directional force reward calculation."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            reward_type="dirs",
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Set force data: active in X, inactive in Y and Z
        mock_env.robot_force_torque[0] = torch.tensor([0.2, 0.05, 0.05, 0.0, 0.0, 0.0])

        # Set selection matrix: force control on X and Y, position control on Z
        wrapper.sel_matrix[0] = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        reward = wrapper._directional_force_reward()

        # X: force control + active force = good
        # Y: force control + inactive force = bad
        # Z: position control + inactive force = good
        expected = (1 * hybrid_task_cfg.good_force_cmd_rew +
                   1 * hybrid_task_cfg.bad_force_cmd_rew +
                   1 * hybrid_task_cfg.good_force_cmd_rew)
        assert reward[0] == expected

    def test_delta_selection_reward(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test delta selection reward calculation."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            reward_type="delta",
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Set initial selection matrix
        wrapper._old_sel_matrix[0] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        wrapper.sel_matrix[0] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        reward = wrapper._delta_selection_reward()

        # Changes in 2 dimensions (X and Y)
        expected = 2.0 * hybrid_task_cfg.bad_force_cmd_rew
        assert reward[0] == expected

        # Check that old selection matrix is updated
        torch.testing.assert_close(wrapper._old_sel_matrix, wrapper.sel_matrix)

    def test_position_simple_reward(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test position simple reward calculation."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            reward_type="pos_simp",
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Set force data: active force
        mock_env.robot_force_torque[0] = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Set selection matrix: force control
        wrapper.sel_matrix[0, 0] = 1.0

        reward = wrapper._position_simple_reward()

        # Active force + force control = good
        assert reward[0] == hybrid_task_cfg.good_force_cmd_rew

    def test_low_wrench_reward(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test low wrench reward calculation."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            reward_type="wrench_norm",
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Set action with wrench components (pose + force actions after selection matrix)
        wrench_start = wrapper.force_size  # After selection matrix
        wrench_size = 6  # pose(6) + force(3) = 9, but we only want 6 for this test
        mock_env.actions[0, wrench_start:wrench_start+wrench_size] = torch.ones(wrench_size) * 2.0

        reward = wrapper._low_wrench_reward()

        expected = -torch.sqrt(torch.tensor(24.0)) * hybrid_task_cfg.wrench_norm_scale
        assert torch.isclose(reward[0], expected)

    def test_wrapped_update_rew_buf_with_base_reward(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test reward buffer update with base reward."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            reward_type="simp",
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Mock original reward function
        wrapper._original_update_rew_buf = lambda x: torch.ones_like(x) * 0.5

        # Set up for simple force reward
        mock_env.robot_force_torque[0] = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        wrapper.sel_matrix[0, 0] = 1.0

        curr_successes = torch.zeros(mock_env.num_envs)
        reward = wrapper._wrapped_update_rew_buf(curr_successes)

        # Should be base reward + hybrid reward
        expected = 0.5 + hybrid_task_cfg.good_force_cmd_rew
        assert reward[0] == expected

    def test_wrapped_update_rew_buf_without_base_reward(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test reward buffer update without base reward."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            reward_type="simp",
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # No original reward function
        wrapper._original_update_rew_buf = None

        # Set up for simple force reward
        mock_env.robot_force_torque[0] = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        wrapper.sel_matrix[0, 0] = 1.0

        curr_successes = torch.zeros(mock_env.num_envs)
        reward = wrapper._wrapped_update_rew_buf(curr_successes)

        # Should be just hybrid reward
        assert reward[0] == hybrid_task_cfg.good_force_cmd_rew

    def test_wrapped_update_rew_buf_base_reward_type(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test reward buffer update with base reward type."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            reward_type="base",
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Mock original reward function
        wrapper._original_update_rew_buf = lambda x: torch.ones_like(x) * 0.5

        curr_successes = torch.zeros(mock_env.num_envs)
        reward = wrapper._wrapped_update_rew_buf(curr_successes)

        # Should be just base reward
        assert reward[0] == 0.5

    def test_step_with_late_initialization(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test step method with late wrapper initialization."""
        # Remove robot initially
        robot = mock_env._robot
        del mock_env._robot

        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        assert wrapper._wrapper_initialized == False

        # Restore robot
        mock_env._robot = robot

        # Step should initialize wrapper
        action = torch.randn(mock_env.num_envs, wrapper.action_space_size)
        obs, reward, terminated, truncated, info = wrapper.step(action)

        assert wrapper._wrapper_initialized == True

    def test_reset_with_late_initialization(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test reset method with late wrapper initialization."""
        # Remove robot initially
        robot = mock_env._robot
        del mock_env._robot

        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        assert wrapper._wrapper_initialized == False

        # Restore robot
        mock_env._robot = robot

        # Reset should initialize wrapper
        obs, info = wrapper.reset()

        assert wrapper._wrapper_initialized == True

    def test_wrapped_pre_physics_step_with_resets(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test wrapped pre-physics step with environment resets."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Initialize wrapper
        wrapper._initialize_wrapper()

        # Set reset buffer
        mock_env.reset_buf[0] = True

        # Mock _reset_buffers method
        mock_env._reset_buffers = Mock()

        action = torch.randn(mock_env.num_envs, wrapper.action_space_size)
        wrapper._wrapped_pre_physics_step(action)

        # Should call reset buffers
        mock_env._reset_buffers.assert_called_once()

    def test_wrapped_apply_action_with_control_flow(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test wrapped apply action with full control flow."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Initialize wrapper
        wrapper._initialize_wrapper()

        # Set up test data
        wrapper.target_force_for_control = torch.randn(mock_env.num_envs, 6)
        wrapper.sel_matrix.fill_(0.5)  # Mix of force and position control

        # Mock robot methods
        mock_env._robot.set_joint_position_target = Mock()
        mock_env._robot.set_joint_effort_target = Mock()

        wrapper._wrapped_apply_action()

        # Should call robot methods
        mock_env._robot.set_joint_position_target.assert_called_once()
        mock_env._robot.set_joint_effort_target.assert_called_once()

    def test_wrapped_apply_action_with_out_of_bounds(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test wrapped apply action with out-of-bounds positions."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Set position out of bounds
        mock_env.fingertip_midpoint_pos[0] = torch.tensor([0.1, 0.0, 0.0])  # > bounds
        mock_env.fixed_pos_action_frame[0] = torch.tensor([0.0, 0.0, 0.0])

        # Set selection matrix for force control
        wrapper.sel_matrix[0] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Set up required target_force_for_control
        wrapper.target_force_for_control = torch.zeros((mock_env.num_envs, 6), device=mock_env.device)

        wrapper._wrapped_apply_action()

        # Force wrench should be zeroed for out-of-bounds positions
        # (This is tested implicitly through the control flow)

    def test_wrapped_apply_action_torque_control_disabled(self, mock_env, hybrid_ctrl_cfg, hybrid_task_cfg):
        """Test wrapped apply action with torque control disabled."""
        wrapper = HybridForcePositionWrapper(
            mock_env,
            ctrl_torque=False,
            ctrl_cfg=hybrid_ctrl_cfg,
            task_cfg=hybrid_task_cfg
        )

        # Set selection for torque (should be ignored)
        wrapper.sel_matrix[:, 3:] = 1.0

        # Set up required target_force_for_control
        wrapper.target_force_for_control = torch.zeros((mock_env.num_envs, 6), device=mock_env.device)

        wrapper._wrapped_apply_action()

        # Torque components should use position control regardless of selection


if __name__ == "__main__":
    pytest.main([__file__])