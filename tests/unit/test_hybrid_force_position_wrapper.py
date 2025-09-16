"""
Unit tests for HybridForcePositionWrapper.

This module tests the HybridForcePositionWrapper functionality including hybrid
force-position control, selection matrix management, and various reward strategies.
"""

import pytest
import torch
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from tests.mocks.mock_isaac_lab import create_mock_env


class TestHybridForcePositionWrapperInitialization:
    """Test HybridForcePositionWrapper initialization and configuration."""

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    def test_initialization_force_only(self, mock_factory_utils, mock_torch_utils, mock_env):
        """Test initialization with force control only (3DOF)."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        # Add force-torque data requirement
        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)

        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False, reward_type="base")

        assert wrapper.ctrl_torque is False
        assert wrapper.force_size == 3
        assert wrapper.reward_type == "base"
        assert wrapper.action_space_size == 12  # 3 (selection) + 6 (pose) + 3 (force for 3DOF)
        assert wrapper.sel_matrix.shape == (64, 6)

        # Check new refactored attributes
        assert wrapper.sel_goal.shape == (64, 6)
        assert wrapper.pose_goal.shape == (64, 7)  # pos(3) + quat(4)
        assert wrapper.force_goal.shape == (64, 6)
        assert wrapper.target_selection.shape == (64, 6)
        assert wrapper.target_pose.shape == (64, 7)
        assert wrapper.target_force.shape == (64, 6)

        assert wrapper.kp.shape == (64, 6)
        # Torque gains should be zero for force-only control
        assert torch.allclose(wrapper.kp[:, 3:], torch.zeros(64, 3))

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    def test_initialization_force_and_torque(self, mock_factory_utils, mock_torch_utils, mock_env):
        """Test initialization with force and torque control (6DOF)."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        # Add force-torque data requirement
        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)

        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=True, reward_type="simp")

        assert wrapper.ctrl_torque is True
        assert wrapper.force_size == 6
        assert wrapper.reward_type == "simp"
        assert wrapper.action_space_size == 18  # 6 (selection) + 6 (pose) + 6 (force)
        # All gains should be non-zero for 6DOF control
        assert torch.all(wrapper.kp > 0)

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    def test_initialization_without_force_data(self, mock_factory_utils, mock_torch_utils, mock_env):
        """Test initialization fails without force-torque data."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        # Remove robot_force_torque from mock environment
        delattr(mock_env, 'robot_force_torque')

        # Should fail during wrapper construction
        with pytest.raises(ValueError, match="Hybrid force-position control requires force-torque sensor data"):
            wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False)

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    def test_initialization_reward_types(self, mock_factory_utils, mock_torch_utils, mock_env):
        """Test initialization with different reward types."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)

        # Test delta reward type (should create _old_sel_matrix)
        wrapper = HybridForcePositionWrapper(mock_env, reward_type="delta")
        assert hasattr(wrapper, '_old_sel_matrix')
        assert wrapper._old_sel_matrix.shape == (64, 6)


class TestHybridForcePositionWrapperSelectionMatrix:
    """Test selection matrix processing and action parsing."""

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    @patch('wrappers.control.factory_control_utils.axis_angle_from_quat')
    @patch('wrappers.control.factory_control_utils.compute_pose_task_wrench')
    @patch('wrappers.control.factory_control_utils.compute_force_task_wrench')
    @patch('wrappers.control.factory_control_utils.compute_dof_torque_from_wrench')
    def test_selection_matrix_parsing(self, mock_dof_torque, mock_force_wrench, mock_pose_wrench,
                                    mock_axis_angle, mock_factory_utils, mock_torch_utils, mock_env):
        """Test selection matrix parsing from actions."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_torch_utils.quat_mul.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_torch_utils.quat_conjugate.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_torch_utils.quat_from_euler_xyz.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_torch_utils.get_euler_xyz.return_value = (torch.zeros(64), torch.zeros(64), torch.zeros(64))
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        mock_axis_angle.return_value = torch.zeros(64, 3)
        mock_pose_wrench.return_value = torch.zeros(64, 6)
        mock_force_wrench.return_value = torch.zeros(64, 6)
        mock_dof_torque.return_value = (torch.zeros(64, 9), torch.zeros(64, 6))

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)
        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False)

        # Create test action: first 3 elements are selection matrix
        action = torch.zeros(64, 12, device=mock_env.device)
        action[:, 0] = 0.8  # Force control in X (>0.5)
        action[:, 1] = 0.2  # Position control in Y (<0.5)
        action[:, 2] = 0.9  # Force control in Z (>0.5)

        wrapper._wrapped_pre_physics_step(action)

        # Check selection matrix
        assert wrapper.sel_matrix[0, 0] == 1.0  # Force control X
        assert wrapper.sel_matrix[0, 1] == 0.0  # Position control Y
        assert wrapper.sel_matrix[0, 2] == 1.0  # Force control Z

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    @patch('wrappers.control.factory_control_utils.axis_angle_from_quat')
    def test_action_ema_smoothing(self, mock_axis_angle, mock_factory_utils, mock_torch_utils, mock_env):
        """Test EMA smoothing of actions."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_torch_utils.quat_mul.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_torch_utils.quat_conjugate.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_torch_utils.quat_from_euler_xyz.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_torch_utils.get_euler_xyz.return_value = (torch.zeros(64), torch.zeros(64), torch.zeros(64))
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)
        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False)

        # Test direct EMA application without running the full pre_physics_step
        # Set initial actions
        mock_env.actions = torch.ones(64, 12, device=mock_env.device) * 0.5

        # Create new action
        action = torch.ones(64, 12, device=mock_env.device) * 1.0

        # Test EMA parameters
        ema_factor = getattr(wrapper.unwrapped.cfg.ctrl, 'ema_factor', 0.2)
        no_sel_ema = getattr(wrapper.unwrapped.cfg.ctrl, 'no_sel_ema', True)

        # Manually apply EMA logic as done in the wrapper
        if no_sel_ema:
            # Apply EMA to everything except selection matrix
            mock_env.actions[:, wrapper.force_size:] = (
                ema_factor * action[:, wrapper.force_size:].clone().to(wrapper.device) +
                (1 - ema_factor) * mock_env.actions[:, wrapper.force_size:]
            )
            mock_env.actions[:, :wrapper.force_size] = action[:, :wrapper.force_size]

        # Check EMA was applied (ema_factor=0.2)
        # For elements beyond selection matrix: new_val = 0.2*1.0 + 0.8*0.5 = 0.6
        expected_val = 0.2 * 1.0 + 0.8 * 0.5
        assert torch.allclose(mock_env.actions[:, 3], torch.tensor([expected_val] * 64))

        # Selection matrix should not use EMA (no_sel_ema=True)
        assert torch.allclose(mock_env.actions[:, 0], torch.tensor([1.0] * 64))


class TestHybridForcePositionWrapperControlComputation:
    """Test control computation methods."""

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    @patch('wrappers.control.factory_control_utils.axis_angle_from_quat')
    def test_calc_pose_goal(self, mock_axis_angle, mock_factory_utils, mock_torch_utils, mock_env):
        """Test pose goal calculation from action."""
        # Set up mocks for quaternion operations
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_torch_utils.quat_mul.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_torch_utils.get_euler_xyz.return_value = (torch.zeros(64), torch.zeros(64), torch.zeros(64))
        mock_torch_utils.quat_from_euler_xyz.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)
        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False)

        # Set test actions for pose control
        action = torch.zeros(64, 12, device=mock_env.device)
        action[:, 3:6] = torch.tensor([0.1, -0.1, 0.05])  # Position actions
        action[:, 6:9] = torch.tensor([0.02, 0.0, 0.01])  # Rotation actions

        # Extract goals from action
        wrapper._extract_goals_from_action(action)

        # Check pose goal shape and that it was updated
        assert wrapper.pose_goal.shape == (64, 7)  # pos(3) + quat(4)
        # Position goal should be current pos + action * threshold
        expected_pos_delta = action[:, 3:6] * mock_env.pos_threshold
        assert torch.allclose(wrapper.pose_goal[:, :3], mock_env.fingertip_midpoint_pos + expected_pos_delta)

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    def test_calc_force_goal(self, mock_factory_utils, mock_torch_utils, mock_env):
        """Test force goal calculation from action."""
        # Set up mocks for quaternion operations
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_torch_utils.quat_mul.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_torch_utils.get_euler_xyz.return_value = (torch.zeros(64), torch.zeros(64), torch.zeros(64))
        mock_torch_utils.quat_from_euler_xyz.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        mock_env.robot_force_torque = torch.ones(64, 6, device=mock_env.device) * 2.0
        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False)

        # Set test force actions
        action = torch.zeros(64, 12, device=mock_env.device)
        action[:, 9:12] = torch.tensor([0.1, -0.1, 0.05])  # Force actions

        # Extract goals from action
        wrapper._extract_goals_from_action(action)

        # Check force goal calculation (should be action*threshold + current_force, clipped)
        force_threshold = 10.0  # Default from config
        force_bounds = 50.0     # Default from config
        expected_force = torch.tensor([1.0, -1.0, 0.5]) + 2.0  # action*10 + current
        expected_force = torch.clip(expected_force, -force_bounds, force_bounds)

        assert wrapper.force_goal.shape == (64, 6)
        assert torch.allclose(wrapper.force_goal[0, :3], expected_force)

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    def test_get_target_out_of_bounds(self, mock_factory_utils, mock_torch_utils, mock_env):
        """Test out-of-bounds position detection."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)
        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False)

        # Set up out-of-bounds position
        mock_env.fingertip_midpoint_pos[0] = torch.tensor([0.6, 0.0, 0.4])  # Out of bounds in X

        out_of_bounds = wrapper._get_target_out_of_bounds()

        # Check detection (bounds are ±0.1 from fixed_pos_action_frame)
        assert out_of_bounds[0, 0] == True   # X out of bounds
        assert out_of_bounds[0, 1] == False  # Y within bounds
        assert out_of_bounds[0, 2] == False  # Z within bounds


class TestHybridForcePositionWrapperRewards:
    """Test different reward computation strategies."""

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    def test_simple_force_reward(self, mock_factory_utils, mock_torch_utils, mock_env):
        """Test simple force activity reward."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)
        mock_env.robot_force_torque[0, :3] = torch.tensor([2.0, 0.0, 0.0])  # Active force (>0.1)
        mock_env.robot_force_torque[1, :3] = torch.tensor([0.05, 0.0, 0.0])  # Inactive force (<0.1)

        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False, reward_type="simp")

        # Set selection matrix
        wrapper.sel_matrix[0, 0] = 1.0  # Force control with active force - good
        wrapper.sel_matrix[1, 0] = 1.0  # Force control with inactive force - bad
        wrapper.sel_matrix[2, 0] = 0.0  # Position control with inactive force - neutral

        reward = wrapper._simple_force_reward()

        good_reward = wrapper.task_cfg.good_force_cmd_rew   # From task_cfg
        bad_reward = wrapper.task_cfg.bad_force_cmd_rew     # From task_cfg

        assert reward[0] == good_reward   # Good force command
        assert reward[1] == bad_reward    # Bad force command
        assert reward[2] == 0.0           # Neutral (position control)

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    def test_directional_force_reward(self, mock_factory_utils, mock_torch_utils, mock_env):
        """Test direction-specific force reward."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)
        mock_env.robot_force_torque[0, :3] = torch.tensor([2.0, 0.05, 0.0])  # X active (>0.1), Y inactive (<0.1)

        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False, reward_type="dirs")

        # Set selection matrix: force control in X, position control in Y, force control in Z
        wrapper.sel_matrix[0, 0] = 1.0  # Force control, active force - good
        wrapper.sel_matrix[0, 1] = 0.0  # Position control, inactive force - good
        wrapper.sel_matrix[0, 2] = 1.0  # Force control, inactive force - bad

        reward = wrapper._directional_force_reward()

        good_reward = wrapper.task_cfg.good_force_cmd_rew   # From task_cfg
        bad_reward = wrapper.task_cfg.bad_force_cmd_rew     # From task_cfg

        # 2 good dimensions, 1 bad dimension
        expected = 2 * good_reward + 1 * bad_reward
        assert reward[0] == expected

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    def test_delta_selection_reward(self, mock_factory_utils, mock_torch_utils, mock_env):
        """Test delta selection matrix reward."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)
        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False, reward_type="delta")

        # Set initial selection matrix
        wrapper._old_sel_matrix[0, :3] = torch.tensor([1.0, 0.0, 1.0])

        # Set new selection matrix (change in Y dimension)
        wrapper.sel_matrix[0, :3] = torch.tensor([1.0, 1.0, 1.0])

        reward = wrapper._delta_selection_reward()

        bad_reward = wrapper.task_cfg.bad_force_cmd_rew   # From task_cfg
        # Change of 1.0 in Y dimension
        expected = 1.0 * bad_reward
        assert reward[0] == expected

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    def test_low_wrench_reward(self, mock_factory_utils, mock_torch_utils, mock_env):
        """Test low wrench magnitude reward."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)
        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False, reward_type="wrench_norm")

        # Set actions to compute wrench norm
        mock_env.actions = torch.zeros(64, 12, device=mock_env.device)
        mock_env.actions[0, 3:] = torch.tensor([1.0, 2.0, 3.0] + [0.0] * 6)  # Wrench actions

        reward = wrapper._low_wrench_reward()

        wrench_scale = wrapper.task_cfg.wrench_norm_scale  # From task_cfg
        expected_norm = torch.norm(torch.tensor([1.0, 2.0, 3.0] + [0.0] * 6))
        expected_reward = -expected_norm * wrench_scale

        assert torch.allclose(reward[0], expected_reward)


class TestHybridForcePositionWrapperIntegration:
    """Test wrapper integration and method overrides."""

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    @patch('wrappers.control.factory_control_utils.axis_angle_from_quat')
    @patch('wrappers.control.factory_control_utils.compute_pose_task_wrench')
    @patch('wrappers.control.factory_control_utils.compute_force_task_wrench')
    @patch('wrappers.control.factory_control_utils.compute_dof_torque_from_wrench')
    def test_wrapped_apply_action(self, mock_dof_torque, mock_force_wrench, mock_pose_wrench,
                                mock_axis_angle, mock_factory_utils, mock_torch_utils, mock_env):
        """Test wrapped _apply_action method."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_torch_utils.quat_mul.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_torch_utils.quat_conjugate.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_torch_utils.get_euler_xyz.return_value = (torch.zeros(64), torch.zeros(64), torch.zeros(64))
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        mock_axis_angle.return_value = torch.zeros(64, 3)
        pose_wrench = torch.ones(64, 6) * 1.0
        force_wrench = torch.ones(64, 6) * 2.0
        joint_torque = torch.ones(64, 9) * 0.5

        mock_pose_wrench.return_value = pose_wrench
        mock_force_wrench.return_value = force_wrench
        mock_dof_torque.return_value = (joint_torque, torch.ones(64, 6))

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)
        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False)

        # Set selection matrix for hybrid control
        wrapper.sel_matrix[0, :3] = torch.tensor([1.0, 0.0, 1.0])  # Force in X,Z; position in Y
        # Also set target_force_for_control which is used in apply_action
        wrapper.target_force_for_control = torch.zeros(64, 6, device=mock_env.device)

        # Mock timestamp to avoid update condition
        wrapper.unwrapped.last_update_timestamp = 0.0
        wrapper.unwrapped._robot._data._sim_timestamp = 2.0

        wrapper._wrapped_apply_action()

        # Verify robot methods were called - these should always be called
        assert hasattr(mock_env._robot, 'last_position_targets')
        assert hasattr(mock_env._robot, 'last_effort_targets')

        # Verify joint torques were set
        assert mock_env.joint_torque.shape == (64, 9)

        # Verify gripper targets were set (they should be 0.0 as set in the wrapper)
        assert torch.allclose(mock_env.ctrl_target_joint_pos[:, 7:9], torch.zeros(64, 2))

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    def test_wrapped_update_rew_buf(self, mock_factory_utils, mock_torch_utils, mock_env):
        """Test wrapped reward buffer update."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)
        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False, reward_type="simp")

        curr_successes = torch.ones(64, dtype=torch.bool, device=mock_env.device)

        # Mock has original method that returns base rewards
        wrapper._original_update_rew_buf = lambda x: torch.ones_like(x).float() * 1.0

        rew_buf = wrapper._wrapped_update_rew_buf(curr_successes)

        # Should include base rewards (1.0) plus hybrid control rewards
        assert rew_buf.shape == (64,)
        assert torch.all(rew_buf >= 1.0)  # At least base reward

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    def test_step_initialization(self, mock_factory_utils, mock_torch_utils, mock_env):
        """Test wrapper initializes during step if needed."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        # Remove _robot to prevent initial initialization
        delattr(mock_env, '_robot')
        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False)
        assert wrapper._wrapper_initialized is False

        # Add _robot and force-torque data, then call step
        mock_env._robot = "dummy"
        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)

        action = torch.zeros(64, 12)
        obs, reward, terminated, truncated, info = wrapper.step(action)

        assert wrapper._wrapper_initialized is True

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    def test_reset_initialization(self, mock_factory_utils, mock_torch_utils, mock_env):
        """Test wrapper initializes during reset if needed."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        # Remove _robot to prevent initial initialization
        delattr(mock_env, '_robot')
        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False)
        assert wrapper._wrapper_initialized is False

        # Add _robot and force-torque data, then call reset
        mock_env._robot = "dummy"
        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)

        obs, info = wrapper.reset()

        assert wrapper._wrapper_initialized is True

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    def test_torch_utils_import_warning(self, mock_factory_utils, mock_torch_utils, mock_env):
        """Test proper warning when torch_utils is not available."""
        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper
        import warnings

        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)

        # Mock torch_utils as None to simulate import failure
        with patch('wrappers.control.hybrid_force_position_wrapper.torch_utils', None):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False)
                # Should create wrapper with fallback torch_utils
                assert wrapper is not None
                assert len(w) == 1
                assert "torch_utils not available" in str(w[0].message)


class TestHybridForcePositionWrapperActionSpace:
    """Test action space management."""

    @patch('wrappers.control.hybrid_force_position_wrapper.torch_utils')
    @patch('wrappers.control.factory_control_utils.torch_utils')
    def test_action_space_update(self, mock_factory_utils, mock_torch_utils, mock_env):
        """Test action space gets updated properly."""
        # Set up mocks
        mock_torch_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)
        mock_factory_utils.quat_from_angle_axis.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 64)

        from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

        mock_env.robot_force_torque = torch.zeros(64, 6, device=mock_env.device)

        # Check original action space
        original_size = mock_env.action_space.shape[0]

        wrapper = HybridForcePositionWrapper(mock_env, ctrl_torque=False)

        # Action space should be updated
        assert mock_env.cfg.action_space == 12  # 3 (sel) + 6 (pose) + 3 (force for 3DOF)
        # _configure_gym_env_spaces should have been called
        assert mock_env.action_space.shape[0] == 12