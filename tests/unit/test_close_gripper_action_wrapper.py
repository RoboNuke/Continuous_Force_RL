"""
Unit tests for close_gripper_action_wrapper.py functionality.
Tests GripperCloseEnv action wrapper.
"""

import pytest
import torch
import gymnasium as gym
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock modules before imports
sys.modules['omni.isaac.lab'] = __import__('tests.mocks.mock_isaac_lab', fromlist=[''])
sys.modules['omni.isaac.lab.envs'] = __import__('tests.mocks.mock_isaac_lab', fromlist=['envs'])
sys.modules['omni.isaac.lab.utils'] = __import__('tests.mocks.mock_isaac_lab', fromlist=['utils'])

from wrappers.mechanics.close_gripper_action_wrapper import GripperCloseEnv
from tests.mocks.mock_isaac_lab import MockBaseEnv


class TestGripperCloseEnv:
    """Test GripperCloseEnv action wrapper functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.base_env = MockBaseEnv()
        self.base_env.num_envs = 4
        self.base_env.device = torch.device("cpu")
        self.base_env.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,))  # 7-dim for gripper

    def test_initialization(self):
        """Test wrapper initialization."""
        wrapper = GripperCloseEnv(self.base_env)

        assert wrapper.env == self.base_env
        assert hasattr(wrapper, 'action_space')
        assert hasattr(wrapper, 'observation_space')

    def test_action_transformation(self):
        """Test that actions are modified to close gripper."""
        wrapper = GripperCloseEnv(self.base_env)

        # Create test actions with gripper open (positive value)
        actions = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0],  # Last value is gripper
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 0.5],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]
        ])

        # Transform actions
        transformed_actions = wrapper.action(actions)

        # Check that gripper values (last column) are all set to -1.0 (closed)
        assert torch.all(transformed_actions[:, -1] == -1.0)

        # Check that other action components are unchanged
        assert torch.allclose(transformed_actions[:, :-1], actions[:, :-1])

    def test_action_inplace_modification(self):
        """Test that action modification is done in-place."""
        wrapper = GripperCloseEnv(self.base_env)

        # Create test actions
        original_actions = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0],
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 0.5]
        ])
        actions = original_actions.clone()

        # Transform actions
        transformed_actions = wrapper.action(actions)

        # Check that the returned tensor is the same object (in-place modification)
        assert transformed_actions is actions

        # Check that original tensor was modified
        assert torch.all(actions[:, -1] == -1.0)

    def test_step_functionality(self):
        """Test step method applies action transformation."""
        wrapper = GripperCloseEnv(self.base_env)

        actions = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0],
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 0.5],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]
        ])

        # Mock the base environment step method to capture the actions passed to it
        received_actions = None

        def mock_step(actions):
            nonlocal received_actions
            received_actions = actions.clone()
            obs = torch.randn(4, 64)
            rewards = torch.randn(4)
            terminated = torch.zeros(4, dtype=torch.bool)
            truncated = torch.zeros(4, dtype=torch.bool)
            info = {"timeout": truncated}
            return obs, rewards, terminated, truncated, info

        wrapper.env.step = mock_step

        # Call step
        result = wrapper.step(actions)

        # Check that step returns correct format
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

        # Check that the base environment received actions with gripper closed
        assert received_actions is not None
        assert torch.all(received_actions[:, -1] == -1.0)

        # Check that other action components were preserved
        assert torch.allclose(received_actions[:, :-1], actions[:, :-1])

    def test_action_with_different_shapes(self):
        """Test action transformation with different tensor shapes."""
        wrapper = GripperCloseEnv(self.base_env)

        # Test with different numbers of environments
        test_cases = [
            torch.randn(1, 7),  # Single environment
            torch.randn(2, 7),  # Two environments
            torch.randn(8, 7),  # Eight environments
        ]

        for actions in test_cases:
            transformed = wrapper.action(actions)
            assert torch.all(transformed[:, -1] == -1.0)
            assert torch.allclose(transformed[:, :-1], actions[:, :-1])

    def test_action_with_edge_cases(self):
        """Test action transformation with edge case values."""
        wrapper = GripperCloseEnv(self.base_env)

        # Test with extreme values
        actions = torch.tensor([
            [float('inf'), -float('inf'), 1e10, -1e10, 0.0, 1e-10, float('inf')],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -float('inf')],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, float('nan')],
        ])

        transformed = wrapper.action(actions)

        # Gripper should always be -1.0 regardless of input
        assert torch.all(transformed[:, -1] == -1.0)

        # Other components should be preserved (including extreme values)
        assert torch.equal(transformed[:, :-1], actions[:, :-1])  # NaN == NaN returns False, but we want to preserve it

    def test_action_wrapper_inheritance(self):
        """Test that wrapper properly inherits from gym.ActionWrapper."""
        wrapper = GripperCloseEnv(self.base_env)

        assert isinstance(wrapper, gym.ActionWrapper)
        assert isinstance(wrapper, gym.Wrapper)

    def test_wrapper_properties(self):
        """Test that wrapper properly delegates properties."""
        self.base_env.action_space = gym.spaces.Box(low=-2, high=2, shape=(7,))
        self.base_env.observation_space = gym.spaces.Box(low=-3, high=3, shape=(64,))

        wrapper = GripperCloseEnv(self.base_env)

        # Should delegate to base environment
        assert wrapper.action_space.shape == self.base_env.action_space.shape
        assert wrapper.observation_space.shape == self.base_env.observation_space.shape

    def test_unwrapped_property(self):
        """Test unwrapped property access."""
        wrapper = GripperCloseEnv(self.base_env)

        assert wrapper.unwrapped == self.base_env

    def test_close_functionality(self):
        """Test wrapper close method."""
        wrapper = GripperCloseEnv(self.base_env)

        # Should not raise any errors
        wrapper.close()

    def test_reset_functionality(self):
        """Test wrapper reset method."""
        wrapper = GripperCloseEnv(self.base_env)

        # Call reset
        result = wrapper.reset()

        # Check return format
        assert len(result) == 2
        obs, info = result
        assert obs.shape[0] == 4

    def test_action_tensor_preservation(self):
        """Test that tensor properties are preserved."""
        wrapper = GripperCloseEnv(self.base_env)

        # Test with different devices and dtypes
        actions_cpu = torch.randn(4, 7, dtype=torch.float32, device=torch.device("cpu"))
        actions_double = torch.randn(4, 7, dtype=torch.float64, device=torch.device("cpu"))

        transformed_cpu = wrapper.action(actions_cpu)
        transformed_double = wrapper.action(actions_double)

        # Check that device and dtype are preserved
        assert transformed_cpu.device == actions_cpu.device
        assert transformed_cpu.dtype == actions_cpu.dtype
        assert transformed_double.device == actions_double.device
        assert transformed_double.dtype == actions_double.dtype

    def test_gripper_close_consistency(self):
        """Test that gripper is consistently closed across multiple calls."""
        wrapper = GripperCloseEnv(self.base_env)

        actions = torch.randn(4, 7)

        # Call action transformation multiple times
        for _ in range(10):
            # Reset actions to random values
            actions[:, -1] = torch.randn(4)
            transformed = wrapper.action(actions)
            assert torch.all(transformed[:, -1] == -1.0)

    def test_print_debugging_output(self):
        """Test that wrapper includes print statements for debugging."""
        wrapper = GripperCloseEnv(self.base_env)

        actions = torch.randn(4, 7)

        # Capture print output (wrapper has print statements for debugging)
        with patch('builtins.print') as mock_print:
            wrapper.action(actions)
            # Should have called print at least twice (before and after modification)
            assert mock_print.call_count >= 2

    def test_multiple_step_calls(self):
        """Test multiple consecutive step calls."""
        wrapper = GripperCloseEnv(self.base_env)

        actions = torch.randn(4, 7)

        # Call step multiple times
        for i in range(5):
            result = wrapper.step(actions)
            assert len(result) == 5

            # Reset actions to ensure gripper value changes
            actions[:, -1] = torch.randn(4)

    def test_wrapper_chain_compatibility(self):
        """Test that wrapper works in a chain with other wrappers."""
        # Create a simple wrapper chain
        intermediate_wrapper = gym.Wrapper(self.base_env)
        wrapper = GripperCloseEnv(intermediate_wrapper)

        assert wrapper.unwrapped == self.base_env

        actions = torch.randn(4, 7)
        result = wrapper.step(actions)
        assert len(result) == 5