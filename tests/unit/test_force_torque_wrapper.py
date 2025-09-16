"""
Unit tests for ForceTorqueWrapper.

This module tests the ForceTorqueWrapper functionality in isolation using mocked
Isaac Lab components. Tests focus on individual methods and their behavior.

Note: ForceTorqueWrapper is now focused only on providing force-torque sensor data.
Episode statistics tracking has been moved to FactoryMetricsWrapper for proper
separation of concerns.
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


class TestForceTorqueWrapperGetCurrentData:
    """Test the get_current_force_torque() method."""

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_get_current_force_torque_with_valid_data(self, mock_env):
        """
        Test get_current_force_torque() returns correct dictionary structure with valid data.

        Expected behavior:
        - Returns dict with keys: 'current_force', 'current_torque'
        - current_force shape: (64, 3)
        - current_torque shape: (64, 3)
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

        # Call the method
        data = wrapper.get_current_force_torque()

        # Verify the data structure
        expected_keys = {'current_force', 'current_torque'}
        assert set(data.keys()) == expected_keys

        # Verify shapes and values
        assert data['current_force'].shape == (64, 3)
        assert data['current_torque'].shape == (64, 3)
        assert torch.allclose(data['current_force'], torch.tensor([[1.0, 2.0, 3.0]] * 64, device=mock_env.device))
        assert torch.allclose(data['current_torque'], torch.tensor([[0.1, 0.2, 0.3]] * 64, device=mock_env.device))

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_get_current_force_torque_without_data(self, mock_env):
        """
        Test get_current_force_torque() returns empty dict when no force-torque data is available.

        Expected behavior:
        - Returns empty dict when wrapper doesn't have robot_force_torque attribute
        """
        # Import here to ensure patch is applied
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        # Create wrapper but remove force-torque data
        wrapper = ForceTorqueWrapper(mock_env)
        delattr(wrapper.unwrapped, 'robot_force_torque')

        # Call the method
        data = wrapper.get_current_force_torque()

        # Should return empty dict
        assert data == {}

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_get_current_force_torque_tensor_devices(self, mock_env):
        """
        Test get_current_force_torque() preserves tensor devices.

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

        # Call the method
        data = wrapper.get_current_force_torque()

        # Verify device preservation
        assert data['current_force'].device == device
        assert data['current_torque'].device == device


class TestForceTorqueWrapperHasData:
    """Test the has_force_torque_data() method."""

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_has_force_torque_data_with_data(self, mock_env):
        """
        Test has_force_torque_data() returns True when data is available.

        Expected behavior:
        - Returns True when robot_force_torque attribute exists
        """
        # Import here to ensure patch is applied
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        # Create wrapper - should initialize robot_force_torque
        wrapper = ForceTorqueWrapper(mock_env)

        # Should have force-torque data
        assert wrapper.has_force_torque_data() is True

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_has_force_torque_data_without_data(self, mock_env):
        """
        Test has_force_torque_data() returns False when data is not available.

        Expected behavior:
        - Returns False when robot_force_torque attribute doesn't exist
        """
        # Import here to ensure patch is applied
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        # Create wrapper and remove force-torque data
        wrapper = ForceTorqueWrapper(mock_env)
        delattr(wrapper.unwrapped, 'robot_force_torque')

        # Should not have force-torque data
        assert wrapper.has_force_torque_data() is False


class TestForceTorqueWrapperGetObservation:
    """Test the get_force_torque_observation() method."""

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_get_force_torque_observation_with_data(self, mock_env):
        """
        Test get_force_torque_observation() returns force-torque readings as observation.

        Expected behavior:
        - Returns robot_force_torque tensor when data is available
        - Shape should be (num_envs, 6)
        """
        # Import here to ensure patch is applied
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        # Create wrapper
        wrapper = ForceTorqueWrapper(mock_env)

        # Mock force-torque data
        expected_data = torch.tensor([
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        ] * 64, device=mock_env.device)
        wrapper.unwrapped.robot_force_torque = expected_data

        # Call the method
        obs = wrapper.get_force_torque_observation()

        # Verify observation
        assert torch.allclose(obs, expected_data)
        assert obs.shape == (64, 6)

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_get_force_torque_observation_without_data(self, mock_env):
        """
        Test get_force_torque_observation() returns zero tensor when no data is available.

        Expected behavior:
        - Returns zero tensor when robot_force_torque attribute doesn't exist
        """
        # Import here to ensure patch is applied
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        # Create wrapper and remove force-torque data
        wrapper = ForceTorqueWrapper(mock_env)
        delattr(wrapper.unwrapped, 'robot_force_torque')

        # Call the method
        obs = wrapper.get_force_torque_observation()

        # Should return zero tensor of correct shape
        expected_zeros = torch.zeros((64, 6), device=mock_env.device)
        assert torch.allclose(obs, expected_zeros)
        assert obs.shape == (64, 6)

    @patch('wrappers.sensors.force_torque_wrapper.RobotView', MockRobotView)
    def test_get_force_torque_observation_with_tanh_scaling(self, mock_env):
        """
        Test get_force_torque_observation() applies tanh scaling when enabled.

        Expected behavior:
        - When use_tanh_scaling=True, applies tanh(data / tanh_scale)
        - Scales input by tanh_scale before applying tanh
        """
        # Import here to ensure patch is applied
        from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper

        # Create wrapper with tanh scaling
        wrapper = ForceTorqueWrapper(mock_env, use_tanh_scaling=True, tanh_scale=0.1)

        # Mock force-torque data
        raw_data = torch.tensor([
            [0.1, 0.2, 0.3, 0.01, 0.02, 0.03]
        ] * 64, device=mock_env.device)
        wrapper.unwrapped.robot_force_torque = raw_data

        # Call the method
        obs = wrapper.get_force_torque_observation()

        # Calculate expected scaled data
        expected_obs = torch.tanh(raw_data * 0.1)

        # Verify scaled observation
        assert torch.allclose(obs, expected_obs)


@pytest.fixture(scope="function")
def mock_env():
    """Create a mock environment for testing."""
    return create_mock_env()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])