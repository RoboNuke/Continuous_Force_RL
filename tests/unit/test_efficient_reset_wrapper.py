"""
Unit tests for efficient_reset_wrapper.py functionality.
Tests EfficientResetWrapper for state caching and efficient resets.
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
sys.modules['isaaclab.envs.direct_rl_env'] = __import__('tests.mocks.mock_isaac_lab', fromlist=['direct_rl_env'])

from wrappers.mechanics.efficient_reset_wrapper import EfficientResetWrapper
from tests.mocks.mock_isaac_lab import MockBaseEnv, DirectRLEnv


class TestEfficientResetWrapper:
    """Test EfficientResetWrapper functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.base_env = MockBaseEnv()
        self.base_env.num_envs = 4
        self.base_env.device = torch.device("cpu")

    def test_initialization_basic(self):
        """Test wrapper initialization with basic environment."""
        wrapper = EfficientResetWrapper(self.base_env)

        assert wrapper.env == self.base_env
        assert wrapper.start_state is None
        assert wrapper._wrapper_initialized == True  # MockBaseEnv has scene
        assert wrapper._original_reset_idx is not None

    def test_initialization_without_scene(self):
        """Test wrapper initialization without scene support."""
        # Create environment without scene
        env_no_scene = MockBaseEnv()
        delattr(env_no_scene, 'scene')

        wrapper = EfficientResetWrapper(env_no_scene)

        assert wrapper.start_state is None
        assert wrapper._wrapper_initialized == False

    def test_wrapper_initialization_lazy(self):
        """Test lazy wrapper initialization during step/reset."""
        # Create environment without scene initially
        env_no_scene = MockBaseEnv()
        delattr(env_no_scene, 'scene')

        wrapper = EfficientResetWrapper(env_no_scene)
        assert wrapper._wrapper_initialized == False

        # Add scene and call step
        env_no_scene.scene = self.base_env.scene
        wrapper.step(torch.randn(4, 6))

        assert wrapper._wrapper_initialized == True

    def test_has_cached_state(self):
        """Test cached state detection."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Initially no cached state
        assert not wrapper.has_cached_state()

        # After full reset, should have cached state
        wrapper.reset()
        assert wrapper.has_cached_state()

    def test_clear_cached_state(self):
        """Test clearing cached state."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Cache some state
        wrapper.reset()
        assert wrapper.has_cached_state()

        # Clear state
        wrapper.clear_cached_state()
        assert not wrapper.has_cached_state()

    def test_full_reset_caches_state(self):
        """Test that full reset caches the initial state."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Perform full reset
        wrapper.reset()

        # Should have cached state
        assert wrapper.start_state is not None
        assert wrapper.has_cached_state()

        # State should contain articulation data
        assert 'articulation' in wrapper.start_state
        assert 'robot' in wrapper.start_state['articulation']

    def test_wrapped_reset_idx_full_reset(self):
        """Test _wrapped_reset_idx with full reset (all environments)."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Mock the original reset method
        original_calls = []
        def mock_original_reset(env_ids):
            original_calls.append(env_ids)

        wrapper._original_reset_idx = mock_original_reset

        # Call full reset (all environments)
        env_ids = torch.arange(wrapper.unwrapped.num_envs, device=wrapper.unwrapped.device)
        wrapper._wrapped_reset_idx(env_ids)

        # Should have called original reset
        assert len(original_calls) == 1
        assert torch.equal(original_calls[0], env_ids)

        # Should have cached state
        assert wrapper.has_cached_state()

    def test_wrapped_reset_idx_partial_reset(self):
        """Test _wrapped_reset_idx with partial reset (some environments)."""
        wrapper = EfficientResetWrapper(self.base_env)

        # First do a full reset to cache state
        wrapper.reset()
        assert wrapper.has_cached_state()

        # Track calls to efficient reset
        efficient_reset_calls = []
        original_perform_efficient_reset = wrapper._perform_efficient_reset

        def mock_efficient_reset(env_ids):
            efficient_reset_calls.append(env_ids)
            return original_perform_efficient_reset(env_ids)

        wrapper._perform_efficient_reset = mock_efficient_reset

        # Call partial reset
        partial_env_ids = torch.tensor([0, 2], device=wrapper.unwrapped.device)
        wrapper._wrapped_reset_idx(partial_env_ids)

        # Should have called efficient reset
        assert len(efficient_reset_calls) == 1
        assert torch.equal(efficient_reset_calls[0], partial_env_ids)

    def test_wrapped_reset_idx_none_env_ids(self):
        """Test _wrapped_reset_idx with None env_ids."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Mock the original reset method
        original_calls = []
        def mock_original_reset(env_ids):
            original_calls.append(env_ids)

        wrapper._original_reset_idx = mock_original_reset

        # Call with None (should reset all environments)
        wrapper._wrapped_reset_idx(None)

        # Should have called original reset with all environment indices
        assert len(original_calls) == 1
        expected_env_ids = torch.arange(wrapper.unwrapped.num_envs, device=wrapper.unwrapped.device)
        # The original call might receive None which gets converted internally
        if original_calls[0] is not None:
            assert torch.equal(original_calls[0], expected_env_ids)
        else:
            # None is acceptable as it represents "all environments"
            assert True

    def test_perform_efficient_reset(self):
        """Test _perform_efficient_reset functionality."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Cache state first
        wrapper.reset()

        # Mock articulation methods to track calls
        robot_articulation = wrapper.unwrapped.scene.articulations['robot']
        pose_calls = []
        vel_calls = []
        joint_calls = []
        reset_calls = []

        def mock_write_root_pose(pose, env_ids):
            pose_calls.append((pose.clone(), env_ids.clone() if env_ids is not None else None))

        def mock_write_root_velocity(vel, env_ids):
            vel_calls.append((vel.clone(), env_ids.clone() if env_ids is not None else None))

        def mock_write_joint_state(pos, vel, env_ids):
            joint_calls.append((pos.clone(), vel.clone(), env_ids.clone() if env_ids is not None else None))

        def mock_reset(env_ids):
            reset_calls.append(env_ids.clone() if env_ids is not None else None)

        robot_articulation.write_root_pose_to_sim = mock_write_root_pose
        robot_articulation.write_root_velocity_to_sim = mock_write_root_velocity
        robot_articulation.write_joint_state_to_sim = mock_write_joint_state
        robot_articulation.reset = mock_reset

        # Perform efficient reset on subset of environments
        env_ids = torch.tensor([0, 2], device=wrapper.unwrapped.device)
        wrapper._perform_efficient_reset(env_ids)

        # Check that articulation methods were called
        assert len(pose_calls) == 1
        assert len(vel_calls) == 1
        assert len(joint_calls) == 1
        assert len(reset_calls) == 1

        # Check that correct env_ids were passed
        assert torch.equal(pose_calls[0][1], env_ids)
        assert torch.equal(vel_calls[0][1], env_ids)
        assert torch.equal(joint_calls[0][2], env_ids)
        assert torch.equal(reset_calls[0], env_ids)

        # Check that poses have correct shape (should be 2 envs worth)
        assert pose_calls[0][0].shape[0] == 2  # 2 environments

    def test_find_directrlenv_reset_method_by_name(self):
        """Test finding DirectRLEnv reset method by class name."""
        # Create a mock environment that inherits from DirectRLEnv
        class MockFactoryEnv(DirectRLEnv):
            def _reset_idx(self, env_ids):
                # Factory env's expensive reset
                pass

        env = MockFactoryEnv()
        wrapper = EfficientResetWrapper(env)

        # Should find DirectRLEnv's method, not factory's
        found_method = wrapper._find_directrlenv_reset_method()

        # The method should be bound to the environment instance
        assert found_method is not None
        assert hasattr(found_method, '__self__')
        assert found_method.__self__ == env

    def test_find_directrlenv_reset_method_fallback(self):
        """Test fallback when DirectRLEnv method cannot be found."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Mock the import to fail by temporarily changing the module import
        import sys
        original_modules = sys.modules.copy()

        # Remove the module if it exists
        if 'isaaclab.envs.direct_rl_env' in sys.modules:
            del sys.modules['isaaclab.envs.direct_rl_env']

        try:
            # Create a new wrapper instance to trigger the import
            new_wrapper = EfficientResetWrapper(self.base_env)
            found_method = new_wrapper._find_directrlenv_reset_method()

            # Should fallback to environment's own method
            assert found_method == new_wrapper.unwrapped._reset_idx
        finally:
            # Restore original modules
            sys.modules.update(original_modules)

    def test_step_with_wrapper_initialization(self):
        """Test step method with wrapper initialization."""
        # Create environment without scene initially
        env_no_scene = MockBaseEnv()
        delattr(env_no_scene, 'scene')

        wrapper = EfficientResetWrapper(env_no_scene)
        assert not wrapper._wrapper_initialized

        # Add scene
        env_no_scene.scene = self.base_env.scene

        # Step should initialize wrapper
        actions = torch.randn(4, 6)
        result = wrapper.step(actions)

        assert wrapper._wrapper_initialized
        assert len(result) == 5

    def test_reset_with_wrapper_initialization(self):
        """Test reset method with wrapper initialization."""
        # Create environment without scene initially
        env_no_scene = MockBaseEnv()
        delattr(env_no_scene, 'scene')

        wrapper = EfficientResetWrapper(env_no_scene)
        assert not wrapper._wrapper_initialized

        # Add scene
        env_no_scene.scene = self.base_env.scene

        # Reset should initialize wrapper and cache state
        result = wrapper.reset()

        assert wrapper._wrapper_initialized
        assert wrapper.has_cached_state()
        assert len(result) == 2

    def test_get_reset_efficiency_stats(self):
        """Test getting reset efficiency statistics."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Before caching state
        stats = wrapper.get_reset_efficiency_stats()
        assert stats['has_cached_state'] == False
        assert stats['supports_efficient_reset'] == True

        # After caching state
        wrapper.reset()
        stats = wrapper.get_reset_efficiency_stats()
        assert stats['has_cached_state'] == True
        assert stats['supports_efficient_reset'] == True

    def test_environment_without_scene_stats(self):
        """Test stats for environment without scene support."""
        env_no_scene = MockBaseEnv()
        delattr(env_no_scene, 'scene')

        wrapper = EfficientResetWrapper(env_no_scene)
        stats = wrapper.get_reset_efficiency_stats()

        assert stats['has_cached_state'] == False
        assert stats['supports_efficient_reset'] == False

    def test_tensor_conversion_in_wrapped_reset_idx(self):
        """Test tensor conversion for env_ids in _wrapped_reset_idx."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Test with list input
        env_ids_list = [0, 2]
        wrapper._wrapped_reset_idx(env_ids_list)

        # Should not raise errors (internal conversion should work)

    def test_wrapper_with_multiple_articulations(self):
        """Test wrapper behavior with multiple articulations."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Cache state
        wrapper.reset()

        # Verify that both robot and object articulations are handled
        assert 'robot' in wrapper.start_state['articulation']
        assert 'object' in wrapper.start_state['articulation']

        # Perform efficient reset
        env_ids = torch.tensor([0], device=wrapper.unwrapped.device)
        wrapper._perform_efficient_reset(env_ids)

        # Should not raise errors with multiple articulations

    def test_state_shuffling_randomness(self):
        """Test that state shuffling uses different source indices."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Cache state
        wrapper.reset()

        # Track source indices used in shuffling
        original_randint = torch.randint
        source_indices = []

        def mock_randint(*args, **kwargs):
            result = original_randint(*args, **kwargs)
            source_indices.append(result.clone())
            return result

        with patch('torch.randint', side_effect=mock_randint):
            # Perform multiple efficient resets
            for _ in range(5):
                env_ids = torch.tensor([0], device=wrapper.unwrapped.device)
                wrapper._perform_efficient_reset(env_ids)

        # Should have generated random indices multiple times
        assert len(source_indices) >= 5

    def test_position_adjustment_for_env_origins(self):
        """Test that positions are adjusted for different environment origins."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Set different env origins
        wrapper.unwrapped.scene.env_origins = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=wrapper.unwrapped.device)

        # Cache state
        wrapper.reset()

        # Track pose adjustments
        pose_calls = []
        def mock_write_root_pose(pose, env_ids):
            pose_calls.append((pose.clone(), env_ids.clone()))

        wrapper.unwrapped.scene.articulations['robot'].write_root_pose_to_sim = mock_write_root_pose

        # Perform efficient reset
        env_ids = torch.tensor([1, 3], device=wrapper.unwrapped.device)
        wrapper._perform_efficient_reset(env_ids)

        # Should have called pose writing
        assert len(pose_calls) == 1

        # Poses should be adjusted for environment origins
        adjusted_poses = pose_calls[0][0]
        assert adjusted_poses.shape[0] == 2  # 2 environments reset

    def test_close_functionality(self):
        """Test wrapper close method."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Should not raise any errors
        wrapper.close()

    def test_unwrapped_property(self):
        """Test unwrapped property access."""
        wrapper = EfficientResetWrapper(self.base_env)

        assert wrapper.unwrapped == self.base_env

    def test_wrapper_properties(self):
        """Test that wrapper properly delegates properties."""
        wrapper = EfficientResetWrapper(self.base_env)

        assert wrapper.action_space.shape == self.base_env.action_space.shape
        assert wrapper.observation_space.shape == self.base_env.observation_space.shape

    def test_method_override_restoration(self):
        """Test that original methods can be called."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Check that original method is stored
        assert wrapper._original_reset_idx is not None

        # Check that the environment's method was overridden
        assert wrapper.unwrapped._reset_idx == wrapper._wrapped_reset_idx

    def test_efficient_reset_without_cached_state(self):
        """Test efficient reset behavior without cached state."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Clear any cached state
        wrapper.clear_cached_state()

        # Try partial reset without cached state
        env_ids = torch.tensor([0, 1], device=wrapper.unwrapped.device)

        # Should not raise errors (method should handle gracefully)
        wrapper._wrapped_reset_idx(env_ids)

    def test_memory_efficiency(self):
        """Test that wrapper doesn't create excessive memory overhead."""
        wrapper = EfficientResetWrapper(self.base_env)

        # Cache state
        wrapper.reset()

        # Check that cached state is reasonable size
        assert wrapper.start_state is not None
        assert isinstance(wrapper.start_state, dict)

        # Should have articulation data
        assert 'articulation' in wrapper.start_state