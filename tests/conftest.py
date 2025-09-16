"""
Pytest configuration and fixtures for factory wrapper unit tests.
"""

import pytest
import torch
import sys
import os

# Add the project root to the Python path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from tests.mocks.mock_isaac_lab import create_mock_env, MockRobotView


@pytest.fixture
def mock_env():
    """Create a mock environment for testing."""
    return create_mock_env(num_envs=64, device="cpu")


@pytest.fixture
def mock_robot_view():
    """Create a mock robot view for testing."""
    robot_view = MockRobotView(device="cpu")
    robot_view.initialize()
    return robot_view


@pytest.fixture
def device():
    """Default device for testing."""
    return torch.device("cpu")