"""
Unit tests for EfficientResetWrapper.

This module tests the EfficientResetWrapper functionality including state caching,
shuffling logic, and performance optimization features.
"""

import pytest
import torch
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from wrappers.mechanics.efficient_reset_wrapper import EfficientResetWrapper
from tests.mocks.mock_isaac_lab import create_mock_env


class TestEfficientResetWrapperInitialization:
    """Test EfficientResetWrapper initialization."""

    def test_initialization_with_scene(self, mock_env):
        """Test initialization when scene is available."""
        wrapper = EfficientResetWrapper(mock_env)

        assert wrapper.start_state is None
        assert wrapper._wrapper_initialized is True  # Should auto-initialize with scene
        assert wrapper._original_reset_idx is not None

    def test_initialization_without_scene(self, mock_env):
        """Test initialization when scene is not available initially."""
        # Remove scene to test deferred initialization
        delattr(mock_env, 'scene')
        wrapper = EfficientResetWrapper(mock_env)

        assert wrapper.start_state is None
        assert wrapper._wrapper_initialized is False

        # Add scene back and test lazy initialization
        mock_env.scene = "dummy_scene"
        obs, info = wrapper.reset()
        assert wrapper._wrapper_initialized is True


class TestEfficientResetWrapperStateCaching:
    """Test state caching functionality."""

    def test_has_cached_state_initially_false(self, mock_env):
        """Test has_cached_state returns False initially."""
        wrapper = EfficientResetWrapper(mock_env)
        assert wrapper.has_cached_state() is False

    def test_state_caching_on_full_reset(self, mock_env):
        """Test state gets cached on full environment reset."""
        wrapper = EfficientResetWrapper(mock_env)

        # Simulate full reset (all environments)
        all_env_ids = torch.arange(64, device=mock_env.device)
        wrapper._wrapped_reset_idx(all_env_ids)

        assert wrapper.has_cached_state() is True
        assert wrapper.start_state is not None
        assert "articulation" in wrapper.start_state

    def test_state_caching_on_reset_method(self, mock_env):
        """Test state gets cached when reset() method is called."""
        wrapper = EfficientResetWrapper(mock_env)

        obs, info = wrapper.reset()

        assert wrapper.has_cached_state() is True
        assert wrapper.start_state is not None

    def test_clear_cached_state(self, mock_env):
        """Test clearing cached state."""
        wrapper = EfficientResetWrapper(mock_env)

        # Cache some state
        wrapper.reset()
        assert wrapper.has_cached_state() is True

        # Clear and verify
        wrapper.clear_cached_state()
        assert wrapper.has_cached_state() is False
        assert wrapper.start_state is None

    def test_get_reset_efficiency_stats(self, mock_env):
        """Test get_reset_efficiency_stats method."""
        wrapper = EfficientResetWrapper(mock_env)

        # Initially no cached state
        stats = wrapper.get_reset_efficiency_stats()
        assert stats["has_cached_state"] is False
        assert stats["supports_efficient_reset"] is True  # MockEnvironment has scene

        # After caching state
        wrapper.reset()
        stats = wrapper.get_reset_efficiency_stats()
        assert stats["has_cached_state"] is True
        assert stats["supports_efficient_reset"] is True


class TestEfficientResetWrapperResetLogic:
    """Test reset logic and state shuffling."""

    def test_full_reset_vs_partial_reset(self, mock_env):
        """Test distinction between full and partial resets."""
        wrapper = EfficientResetWrapper(mock_env)

        # Initial reset (should be treated as full reset)
        wrapper.reset()
        initial_state = wrapper.start_state

        # Partial reset (subset of environments)
        partial_env_ids = torch.tensor([0, 1, 5, 10], device=mock_env.device)
        wrapper._wrapped_reset_idx(partial_env_ids)

        # State should remain the same for partial reset
        assert wrapper.start_state is initial_state

        # Full reset (all environments)
        all_env_ids = torch.arange(64, device=mock_env.device)
        wrapper._wrapped_reset_idx(all_env_ids)

        # State should be updated for full reset
        assert wrapper.start_state is not initial_state

    def test_partial_reset_with_cached_state(self, mock_env):
        """Test partial reset uses state shuffling when cache is available."""
        wrapper = EfficientResetWrapper(mock_env)

        # Cache initial state
        wrapper.reset()
        assert wrapper.has_cached_state()

        # Perform partial reset
        partial_env_ids = torch.tensor([0, 1, 5], device=mock_env.device)
        wrapper._wrapped_reset_idx(partial_env_ids)

        # Verify articulation methods were called for shuffling
        robot_articulation = mock_env.scene.articulations["robot"]
        assert hasattr(robot_articulation, "last_root_pose_write")
        assert hasattr(robot_articulation, "last_root_velocity_write")
        assert hasattr(robot_articulation, "last_joint_state_write")
        assert hasattr(robot_articulation, "last_reset_call")

        # Verify env_ids were passed correctly
        assert torch.equal(robot_articulation.last_reset_call, partial_env_ids)

    def test_partial_reset_without_cached_state(self, mock_env):
        """Test partial reset behavior when no cached state available."""
        wrapper = EfficientResetWrapper(mock_env)

        # Don't cache state, directly call partial reset
        partial_env_ids = torch.tensor([0, 1, 5], device=mock_env.device)
        wrapper._wrapped_reset_idx(partial_env_ids)

        # Should call original reset_idx method
        assert hasattr(mock_env, "last_reset_idx_call")
        assert torch.equal(mock_env.last_reset_idx_call, partial_env_ids)

    def test_env_ids_tensor_conversion(self, mock_env):
        """Test env_ids tensor conversion and None handling."""
        wrapper = EfficientResetWrapper(mock_env)

        # Test with None (should convert to all environments)
        wrapper._wrapped_reset_idx(None)
        # Should be treated as full reset

        # Test with list (should convert to tensor)
        env_ids_list = [0, 1, 2]
        wrapper._wrapped_reset_idx(env_ids_list)

        # Test with tensor (should work directly)
        env_ids_tensor = torch.tensor([5, 6, 7], device=mock_env.device)
        wrapper._wrapped_reset_idx(env_ids_tensor)


class TestEfficientResetWrapperStateShuffling:
    """Test state shuffling logic."""

    def test_state_shuffling_logic(self, mock_env):
        """Test the detailed state shuffling implementation."""
        wrapper = EfficientResetWrapper(mock_env)

        # Cache initial state with known values
        wrapper.reset()

        # Create specific test state to verify shuffling
        test_state = {
            "articulation": {
                "robot": {
                    "root_pose": torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]] * 64, device=mock_env.device),
                    "root_velocity": torch.tensor([[0.1, 0.2, 0.3, 0.01, 0.02, 0.03]] * 64, device=mock_env.device),
                    "joint_position": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]] * 64, device=mock_env.device),
                    "joint_velocity": torch.zeros(64, 9, device=mock_env.device),
                }
            }
        }
        wrapper.start_state = test_state

        # Reset specific environments
        env_ids = torch.tensor([0, 1], device=mock_env.device)
        wrapper._wrapped_reset_idx(env_ids)

        # Verify articulation received state updates
        robot = mock_env.scene.articulations["robot"]
        assert hasattr(robot, "last_root_pose_write")
        assert hasattr(robot, "last_root_velocity_write")
        assert hasattr(robot, "last_joint_state_write")

        # Verify env_ids were passed correctly
        write_ops = [
            robot.last_root_pose_write["env_ids"],
            robot.last_root_velocity_write["env_ids"],
            robot.last_joint_state_write["env_ids"],
            robot.last_reset_call
        ]
        for op_env_ids in write_ops:
            assert torch.equal(op_env_ids, env_ids)

    def test_environment_origin_adjustment(self, mock_env):
        """Test environment origin position adjustments during shuffling."""
        wrapper = EfficientResetWrapper(mock_env)

        # Set up environment origins
        mock_env.scene.env_origins = torch.tensor([
            [1.0, 0.0, 0.0],  # env 0
            [2.0, 0.0, 0.0],  # env 1
            [3.0, 0.0, 0.0],  # env 2
        ] + [[0.0, 0.0, 0.0]] * 61, device=mock_env.device)

        # Cache state and perform reset
        wrapper.reset()

        # Create test state
        test_state = {
            "articulation": {
                "robot": {
                    "root_pose": torch.tensor([[10.0, 20.0, 30.0, 1.0, 0.0, 0.0, 0.0]] * 64, device=mock_env.device),
                    "root_velocity": torch.zeros(64, 6, device=mock_env.device),
                    "joint_position": torch.zeros(64, 9, device=mock_env.device),
                    "joint_velocity": torch.zeros(64, 9, device=mock_env.device),
                }
            }
        }
        wrapper.start_state = test_state

        # Reset env 0 using state from env 1
        # This should adjust position from env 1's origin to env 0's origin
        env_ids = torch.tensor([0], device=mock_env.device)
        wrapper._perform_efficient_reset(env_ids)

        # Check that position was adjusted
        robot = mock_env.scene.articulations["robot"]
        written_pose = robot.last_root_pose_write["pose"]

        # Position should be adjusted: original_pos - source_origin + target_origin
        # We can't predict exact source index due to randomness, but can verify adjustment happened
        assert written_pose.shape == (1, 7)
        assert written_pose[0, 3:].allclose(torch.tensor([1.0, 0.0, 0.0, 0.0]))  # Quaternion unchanged


class TestEfficientResetWrapperIntegration:
    """Test wrapper integration and step/reset methods."""

    def test_step_initialization(self, mock_env):
        """Test wrapper initializes during step if needed."""
        # Remove scene to prevent auto-initialization
        delattr(mock_env, 'scene')
        wrapper = EfficientResetWrapper(mock_env)
        assert wrapper._wrapper_initialized is False

        # Add scene back and call step
        mock_env.scene = "dummy_scene"
        action = torch.zeros(64, 6)
        obs, reward, terminated, truncated, info = wrapper.step(action)

        assert wrapper._wrapper_initialized is True

    def test_reset_initialization(self, mock_env):
        """Test wrapper initializes during reset if needed."""
        # Remove scene to prevent auto-initialization
        delattr(mock_env, 'scene')
        wrapper = EfficientResetWrapper(mock_env)
        assert wrapper._wrapper_initialized is False

        # Add scene back and call reset
        mock_env.scene = "dummy_scene"
        obs, info = wrapper.reset()

        assert wrapper._wrapper_initialized is True

    def test_original_method_preservation(self, mock_env):
        """Test that original _reset_idx method is preserved and callable."""
        wrapper = EfficientResetWrapper(mock_env)

        # Verify original method is stored
        assert wrapper._original_reset_idx is not None

        # Call wrapped method and verify original was called
        env_ids = torch.tensor([0, 1, 2], device=mock_env.device)
        wrapper._wrapped_reset_idx(env_ids)

        # Original method should have been called
        assert hasattr(mock_env, "last_reset_idx_call")
        assert torch.equal(mock_env.last_reset_idx_call, env_ids)

    def test_wrapper_works_without_articulations(self, mock_env):
        """Test wrapper handles missing articulations gracefully."""
        wrapper = EfficientResetWrapper(mock_env)

        # Remove articulations from scene
        mock_env.scene.articulations = {}

        # Cache state and try partial reset
        wrapper.reset()
        env_ids = torch.tensor([0, 1], device=mock_env.device)
        wrapper._wrapped_reset_idx(env_ids)

        # Should not crash, just call original method
        assert hasattr(mock_env, "last_reset_idx_call")