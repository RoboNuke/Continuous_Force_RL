"""
Unit tests for ForceTorqueWrapper.

This module tests the ForceTorqueWrapper functionality in isolation using mocked
Isaac Lab components. Tests focus on individual methods and their behavior.
"""

import pytest
import torch
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from tests.mocks.mock_isaac_lab import create_mock_env, MockRobotView


class TestForceTorqueWrapperGetStats:
    """Test the get_force_torque_stats() method."""

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_get_force_torque_stats_with_valid_data(self, mock_env):
        """
        Test get_force_torque_stats() returns correct dictionary structure with valid data.

        Expected behavior:
        - Returns dict with keys: 'current_force', 'current_torque', 'max_force', 'max_torque', 'avg_force', 'avg_torque'
        - current_force shape: (64, 3)
        - current_torque shape: (64, 3)
        - max/avg values are scalars (tensors)
        - Values come from mock data: [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        """
        # Import here to ensure patch is applied
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        # Create wrapper with mock environment
        wrapper = ForceTorqueWrapper(mock_env)

        # Mock the force-torque data manually to ensure it exists
        wrapper.unwrapped.robot_force_torque = torch.tensor([
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        ] * 64, device=mock_env.device)

        # Add episode statistics manually
        wrapper.unwrapped.ep_max_force = torch.tensor([5.0] * 64, device=mock_env.device)
        wrapper.unwrapped.ep_max_torque = torch.tensor([1.5] * 64, device=mock_env.device)
        wrapper.unwrapped.ep_sum_force = torch.tensor([100.0] * 64, device=mock_env.device)
        wrapper.unwrapped.ep_sum_torque = torch.tensor([30.0] * 64, device=mock_env.device)
        wrapper.unwrapped.common_step_counter = 10

        # Call the method
        stats = wrapper.get_force_torque_stats()

        # Verify return type and structure
        assert isinstance(stats, dict)
        expected_keys = {'current_force', 'current_torque', 'max_force', 'max_torque', 'avg_force', 'avg_torque'}
        assert set(stats.keys()) == expected_keys

        # Verify current force values and shape
        assert stats['current_force'].shape == (64, 3)
        assert torch.allclose(stats['current_force'], torch.tensor([1.0, 2.0, 3.0]))

        # Verify current torque values and shape
        assert stats['current_torque'].shape == (64, 3)
        assert torch.allclose(stats['current_torque'], torch.tensor([0.1, 0.2, 0.3]))

        # Verify episode statistics
        assert stats['max_force'].shape == (64,)
        assert torch.allclose(stats['max_force'], torch.tensor([5.0] * 64))

        assert stats['max_torque'].shape == (64,)
        assert torch.allclose(stats['max_torque'], torch.tensor([1.5] * 64))

        # Verify average calculations (sum / step_counter)
        assert stats['avg_force'].shape == (64,)
        assert torch.allclose(stats['avg_force'], torch.tensor([10.0] * 64))  # 100.0 / 10

        assert stats['avg_torque'].shape == (64,)
        assert torch.allclose(stats['avg_torque'], torch.tensor([3.0] * 64))   # 30.0 / 10

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_get_force_torque_stats_without_data(self, mock_env):
        """
        Test get_force_torque_stats() returns stats with zero values when wrapper initializes data.

        Expected behavior:
        - Returns stats dict with zero values when no actual sensor data is available
        - Wrapper automatically creates force-torque tensors during initialization
        """
        # Import here to ensure patch is applied
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        # Create wrapper - this will initialize force-torque tensors with zeros
        wrapper = ForceTorqueWrapper(mock_env)

        # Call the method
        stats = wrapper.get_force_torque_stats()

        # Verify stats dict returned with appropriate keys and zero values
        assert isinstance(stats, dict)
        assert 'current_force' in stats
        assert 'current_torque' in stats
        assert 'max_force' in stats
        assert 'max_torque' in stats
        assert 'avg_force' in stats
        assert 'avg_torque' in stats

        # Check that values are tensors with appropriate shapes
        assert stats['current_force'].shape == (64, 3)
        assert stats['current_torque'].shape == (64, 3)
        assert stats['max_force'].shape == (64,)
        assert stats['max_torque'].shape == (64,)

        # Verify all values are zeros (since no real data was provided)
        assert torch.allclose(stats['current_force'], torch.zeros(64, 3))
        assert torch.allclose(stats['current_torque'], torch.zeros(64, 3))
        assert torch.allclose(stats['max_force'], torch.zeros(64))
        assert torch.allclose(stats['max_torque'], torch.zeros(64))

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_get_force_torque_stats_missing_episode_stats(self, mock_env):
        """
        Test get_force_torque_stats() behavior when only some attributes are present.

        Expected behavior:
        - Wrapper initializes all required attributes during initialization
        - Even if manually removing attributes, wrapper should handle gracefully
        """
        # Import here to ensure patch is applied
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        # Create wrapper - this initializes all force-torque tensors
        wrapper = ForceTorqueWrapper(mock_env)

        # Override with custom force-torque data
        wrapper.unwrapped.robot_force_torque = torch.tensor([
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        ] * 64, device=mock_env.device)

        # Manually remove episode stats to test missing attribute handling
        delattr(wrapper.unwrapped, 'ep_max_force')

        # Call the method
        stats = wrapper.get_force_torque_stats()

        # Should return empty dict when required episode stats are missing
        assert stats == {}

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_get_force_torque_stats_zero_step_counter(self, mock_env):
        """
        Test get_force_torque_stats() handles zero step counter correctly.

        Expected behavior:
        - Uses max(1, step_counter) to avoid division by zero
        - When step_counter is 0 or missing, uses 1 as denominator
        """
        # Import here to ensure patch is applied
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        # Create wrapper with all data
        wrapper = ForceTorqueWrapper(mock_env)
        wrapper.unwrapped.robot_force_torque = torch.tensor([
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        ] * 64, device=mock_env.device)
        wrapper.unwrapped.ep_max_force = torch.tensor([5.0] * 64, device=mock_env.device)
        wrapper.unwrapped.ep_max_torque = torch.tensor([1.5] * 64, device=mock_env.device)
        wrapper.unwrapped.ep_sum_force = torch.tensor([100.0] * 64, device=mock_env.device)
        wrapper.unwrapped.ep_sum_torque = torch.tensor([30.0] * 64, device=mock_env.device)

        # Don't set common_step_counter (should default to 1)

        # Call the method
        stats = wrapper.get_force_torque_stats()

        # Verify averages use denominator of 1
        assert torch.allclose(stats['avg_force'], torch.tensor([100.0] * 64))  # 100.0 / 1
        assert torch.allclose(stats['avg_torque'], torch.tensor([30.0] * 64))  # 30.0 / 1

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_get_force_torque_stats_tensor_devices(self, mock_env):
        """
        Test get_force_torque_stats() preserves tensor devices.

        Expected behavior:
        - Returned tensors should be on the same device as input tensors
        """
        # Import here to ensure patch is applied
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        # Create wrapper with data
        wrapper = ForceTorqueWrapper(mock_env)
        device = mock_env.device

        wrapper.unwrapped.robot_force_torque = torch.tensor([
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        ] * 64, device=device)
        wrapper.unwrapped.ep_max_force = torch.tensor([5.0] * 64, device=device)
        wrapper.unwrapped.ep_max_torque = torch.tensor([1.5] * 64, device=device)
        wrapper.unwrapped.ep_sum_force = torch.tensor([100.0] * 64, device=device)
        wrapper.unwrapped.ep_sum_torque = torch.tensor([30.0] * 64, device=device)
        wrapper.unwrapped.common_step_counter = 1

        # Call the method
        stats = wrapper.get_force_torque_stats()

        # Verify all tensors are on correct device
        for key, tensor in stats.items():
            assert tensor.device == device, f"Tensor {key} on wrong device: {tensor.device} != {device}"


class TestForceTorqueWrapperHasData:
    """Test the has_force_torque_data() method."""

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_has_force_torque_data_with_data(self, mock_env):
        """Test has_force_torque_data() returns True when data exists."""
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        wrapper = ForceTorqueWrapper(mock_env)
        wrapper.unwrapped.robot_force_torque = torch.zeros(64, 6)

        assert wrapper.has_force_torque_data() is True

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_has_force_torque_data_without_data(self, mock_env):
        """Test has_force_torque_data() behavior after wrapper initialization."""
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        wrapper = ForceTorqueWrapper(mock_env)

        # After initialization, the wrapper creates robot_force_torque attribute
        # So the method should return True even when no real sensor data is available
        assert wrapper.has_force_torque_data() is True

        # Manually remove the attribute to test the False case
        delattr(wrapper.unwrapped, 'robot_force_torque')
        assert wrapper.has_force_torque_data() is False


class TestForceTorqueWrapperGetObservation:
    """Test the get_force_torque_observation() method."""

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_get_force_torque_observation_with_data(self, mock_env):
        """Test get_force_torque_observation() returns correct tensor."""
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        wrapper = ForceTorqueWrapper(mock_env)
        test_data = torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]] * 64, device=mock_env.device)
        wrapper.unwrapped.robot_force_torque = test_data

        obs = wrapper.get_force_torque_observation()

        assert obs.shape == (64, 6)
        assert torch.allclose(obs, test_data)

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_get_force_torque_observation_without_data(self, mock_env):
        """Test get_force_torque_observation() returns zeros when no data."""
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        wrapper = ForceTorqueWrapper(mock_env)
        # Don't add robot_force_torque attribute

        obs = wrapper.get_force_torque_observation()

        assert obs.shape == (64, 6)
        assert torch.allclose(obs, torch.zeros(64, 6))

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_get_force_torque_observation_with_tanh_scaling(self, mock_env):
        """Test get_force_torque_observation() applies tanh scaling."""
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        wrapper = ForceTorqueWrapper(mock_env, use_tanh_scaling=True, tanh_scale=0.5)
        test_data = torch.tensor([[10.0, 20.0, 30.0, 1.0, 2.0, 3.0]] * 64, device=mock_env.device)
        wrapper.unwrapped.robot_force_torque = test_data

        obs = wrapper.get_force_torque_observation()

        expected = torch.tanh(0.5 * test_data)
        assert obs.shape == (64, 6)
        assert torch.allclose(obs, expected)