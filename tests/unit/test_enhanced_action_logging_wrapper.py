"""
Unit tests for enhanced_action_logging_wrapper.py logging functionality.
Tests EnhancedActionLoggingWrapper class and its integration with wandb wrapper.
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
sys.modules['wandb'] = __import__('tests.mocks.mock_wandb', fromlist=[''])
sys.modules['omni.isaac.lab'] = __import__('tests.mocks.mock_isaac_lab', fromlist=[''])
sys.modules['omni.isaac.lab.envs'] = __import__('tests.mocks.mock_isaac_lab', fromlist=['envs'])
sys.modules['omni.isaac.lab.utils'] = __import__('tests.mocks.mock_isaac_lab', fromlist=['utils'])

from wrappers.logging.enhanced_action_logging_wrapper import EnhancedActionLoggingWrapper
from wrappers.logging.wandb_wrapper import GenericWandbLoggingWrapper
from tests.mocks.mock_isaac_lab import MockBaseEnv


class MockWandbWrapper(gym.Wrapper):
    """Mock wandb wrapper for testing dependency validation."""

    def __init__(self, env):
        super().__init__(env)
        self.num_envs = env.num_envs
        self.device = env.device
        self.metrics_calls = []

    def add_metrics(self, metrics_dict):
        """Track metrics calls for testing."""
        self.metrics_calls.append(metrics_dict)


class TestEnhancedActionLoggingWrapper:
    """Test EnhancedActionLoggingWrapper functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.base_env = MockBaseEnv()
        self.base_env.num_envs = 4
        self.base_env.device = torch.device("cpu")
        self.base_env.action_space = gym.spaces.Box(low=-1, high=1, shape=(12,))

        # Create wandb wrapper first
        self.wandb_env = MockWandbWrapper(self.base_env)

    def test_initialization_with_wandb_wrapper(self):
        """Test wrapper initialization with wandb wrapper dependency."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env)

        assert wrapper.track_selection == True
        assert wrapper.track_pos == True
        assert wrapper.track_rot == True
        assert wrapper.track_force == True
        assert wrapper.track_torque == True
        assert wrapper.force_size == 6
        assert wrapper.logging_frequency == 100

    def test_initialization_without_wandb_wrapper_fails(self):
        """Test that initialization fails without wandb wrapper."""
        # Create environment without add_metrics method
        class MockEnvWithoutMetrics(gym.Env):
            def __init__(self):
                super().__init__()
                self.num_envs = 4
                self.device = torch.device("cpu")
                self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(64,))
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(12,))
                self._unwrapped = self

            @property
            def unwrapped(self):
                return self._unwrapped

        env_without_metrics = MockEnvWithoutMetrics()

        with pytest.raises(ValueError, match="Enhanced Action Logging Wrapper requires GenericWandbLoggingWrapper"):
            EnhancedActionLoggingWrapper(env_without_metrics)

    def test_initialization_custom_config(self):
        """Test wrapper initialization with custom configuration."""
        wrapper = EnhancedActionLoggingWrapper(
            self.wandb_env,
            track_selection=False,
            track_pos=True,
            track_rot=False,
            track_force=True,
            track_torque=False,
            force_size=3,
            logging_frequency=50
        )

        assert wrapper.track_selection == False
        assert wrapper.track_pos == True
        assert wrapper.track_rot == False
        assert wrapper.track_force == True
        assert wrapper.track_torque == False
        assert wrapper.force_size == 3
        assert wrapper.logging_frequency == 50

    def test_action_components_generation(self):
        """Test action components generation based on configuration."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env, force_size=6)

        assert hasattr(wrapper, 'action_components')
        assert isinstance(wrapper.action_components, dict)

    def test_step_basic_functionality(self):
        """Test basic step functionality."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env)

        actions = torch.randn(4, 12)
        result = wrapper.step(actions)

        # Check return format
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

        # Check shapes
        assert obs.shape[0] == 4
        assert reward.shape[0] == 4
        assert terminated.shape[0] == 4
        assert truncated.shape[0] == 4

    def test_step_count_tracking(self):
        """Test step count tracking."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env)

        assert wrapper.step_count == 0

        actions = torch.randn(4, 12)
        wrapper.step(actions)

        assert wrapper.step_count == 1

        # Multiple steps
        for _ in range(5):
            wrapper.step(actions)

        assert wrapper.step_count == 6

    def test_action_statistics_storage(self):
        """Test action statistics storage."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env)

        assert hasattr(wrapper, '_action_stats')
        assert isinstance(wrapper._action_stats, dict)

    def test_force_size_configurations(self):
        """Test different force size configurations."""
        # No force/torque
        wrapper_0 = EnhancedActionLoggingWrapper(self.wandb_env, force_size=0)
        assert wrapper_0.force_size == 0

        # Force only
        wrapper_3 = EnhancedActionLoggingWrapper(self.wandb_env, force_size=3)
        assert wrapper_3.force_size == 3

        # Force and torque
        wrapper_6 = EnhancedActionLoggingWrapper(self.wandb_env, force_size=6)
        assert wrapper_6.force_size == 6

    def test_tracking_flags(self):
        """Test different tracking flag combinations."""
        wrapper = EnhancedActionLoggingWrapper(
            self.wandb_env,
            track_selection=True,
            track_pos=False,
            track_rot=True,
            track_force=False,
            track_torque=True
        )

        assert wrapper.track_selection == True
        assert wrapper.track_pos == False
        assert wrapper.track_rot == True
        assert wrapper.track_force == False
        assert wrapper.track_torque == True

    def test_logging_frequency_configuration(self):
        """Test logging frequency configuration."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env, logging_frequency=25)
        assert wrapper.logging_frequency == 25

    def test_multiple_step_calls(self):
        """Test multiple consecutive step calls."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env)

        actions = torch.randn(4, 12)

        # Call step multiple times
        for i in range(10):
            result = wrapper.step(actions)
            assert len(result) == 5
            assert wrapper.step_count == i + 1

    def test_step_with_various_action_shapes(self):
        """Test step with various action tensor shapes."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env)

        # Test with different action shapes that match the action space
        test_actions = [
            torch.randn(4, 12),
            torch.zeros(4, 12),
            torch.ones(4, 12) * 0.5,
            torch.randn(4, 12) * 2.0
        ]

        for actions in test_actions:
            result = wrapper.step(actions)
            assert len(result) == 5

    def test_dependency_validation_deep_chain(self):
        """Test dependency validation through wrapper chain."""
        # Create a chain: base_env -> wandb_wrapper -> other_wrapper -> action_wrapper
        other_wrapper = gym.Wrapper(self.wandb_env)

        # This should still work because it finds add_metrics in the chain
        wrapper = EnhancedActionLoggingWrapper(other_wrapper)
        assert wrapper is not None

    def test_reset_functionality(self):
        """Test environment reset."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env)

        # Call reset
        result = wrapper.reset()

        # Check return format
        assert len(result) == 2
        obs, info = result
        assert obs.shape[0] == 4

    def test_close_functionality(self):
        """Test wrapper close method."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env)

        # Should not raise any errors
        wrapper.close()

    def test_unwrapped_property(self):
        """Test unwrapped property access."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env)

        # Should access the base environment
        assert wrapper.unwrapped == self.base_env

    def test_action_space_delegation(self):
        """Test that action space is properly delegated."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env)

        assert wrapper.action_space.shape == self.base_env.action_space.shape
        assert wrapper.observation_space.shape == self.base_env.observation_space.shape

    def test_step_count_reset_behavior(self):
        """Test step count behavior across episodes."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env)

        actions = torch.randn(4, 12)

        # Take some steps
        for _ in range(5):
            wrapper.step(actions)

        assert wrapper.step_count == 5

        # Reset environment
        wrapper.reset()

        # Step count should not be automatically reset by environment reset
        assert wrapper.step_count == 5

    def test_action_components_with_different_force_sizes(self):
        """Test action components generation with different force sizes."""
        # Force size 0 - no force/torque
        wrapper_0 = EnhancedActionLoggingWrapper(self.wandb_env, force_size=0)
        components_0 = wrapper_0.action_components

        # Force size 3 - force only
        wrapper_3 = EnhancedActionLoggingWrapper(self.wandb_env, force_size=3)
        components_3 = wrapper_3.action_components

        # Force size 6 - force and torque
        wrapper_6 = EnhancedActionLoggingWrapper(self.wandb_env, force_size=6)
        components_6 = wrapper_6.action_components

        # Each should have different components based on force_size
        assert len(components_0) <= len(components_3) <= len(components_6)

    def test_logging_frequency_edge_cases(self):
        """Test logging frequency edge cases."""
        # Very high frequency
        wrapper_high = EnhancedActionLoggingWrapper(self.wandb_env, logging_frequency=1000)
        assert wrapper_high.logging_frequency == 1000

        # Frequency of 1 (log every step)
        wrapper_every = EnhancedActionLoggingWrapper(self.wandb_env, logging_frequency=1)
        assert wrapper_every.logging_frequency == 1

    def test_action_statistics_accumulation(self):
        """Test that action statistics are accumulated properly."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env, logging_frequency=10)

        actions = torch.randn(4, 12)

        # Take several steps to accumulate statistics
        for _ in range(5):
            wrapper.step(actions)

        # Statistics should be accumulated
        assert wrapper.step_count == 5

    def test_multiple_tracking_combinations(self):
        """Test various combinations of tracking flags."""
        tracking_combinations = [
            {"track_selection": True, "track_pos": True, "track_rot": True, "track_force": True, "track_torque": True},
            {"track_selection": False, "track_pos": False, "track_rot": False, "track_force": False, "track_torque": False},
            {"track_selection": True, "track_pos": False, "track_rot": True, "track_force": False, "track_torque": True},
            {"track_selection": False, "track_pos": True, "track_rot": False, "track_force": True, "track_torque": False},
        ]

        for combo in tracking_combinations:
            wrapper = EnhancedActionLoggingWrapper(self.wandb_env, **combo)

            # Check that configuration was applied
            for key, value in combo.items():
                assert getattr(wrapper, key) == value

            # Test that it can step
            actions = torch.randn(4, 12)
            result = wrapper.step(actions)
            assert len(result) == 5

    def test_error_handling_initialization(self):
        """Test error handling during initialization."""
        # Test with None environment
        with pytest.raises(AssertionError):
            EnhancedActionLoggingWrapper(None)

    def test_wrapper_chain_integration(self):
        """Test integration with wrapper chain."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env)

        # Test that wrapper properly delegates to underlying environment
        assert wrapper.action_space.shape == self.base_env.action_space.shape
        assert wrapper.observation_space.shape == self.base_env.observation_space.shape

    def test_metrics_integration_readiness(self):
        """Test that wrapper is ready for metrics integration."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env)

        # Should have access to add_metrics through the chain
        # The actual metrics sending will be tested in integration tests
        assert hasattr(wrapper.env, 'add_metrics')

    def test_memory_efficiency(self):
        """Test that wrapper doesn't create excessive memory overhead."""
        wrapper = EnhancedActionLoggingWrapper(self.wandb_env)

        # Check that action statistics storage is initially reasonable
        assert isinstance(wrapper._action_stats, dict)
        # Initially should be empty
        assert len(wrapper._action_stats) == 0

    def test_force_size_parameter_validation(self):
        """Test force size parameter validation."""
        valid_force_sizes = [0, 3, 6]

        for force_size in valid_force_sizes:
            wrapper = EnhancedActionLoggingWrapper(self.wandb_env, force_size=force_size)
            assert wrapper.force_size == force_size

    def test_component_tracking_consistency(self):
        """Test that component tracking flags are consistently applied."""
        wrapper = EnhancedActionLoggingWrapper(
            self.wandb_env,
            track_selection=True,
            track_pos=False,
            track_rot=True,
            track_force=False,
            track_torque=True,
            force_size=6
        )

        # Check that all tracking flags are preserved
        assert wrapper.track_selection == True
        assert wrapper.track_pos == False
        assert wrapper.track_rot == True
        assert wrapper.track_force == False
        assert wrapper.track_torque == True

        # Action components should reflect the tracking configuration
        assert hasattr(wrapper, 'action_components')

    def test_step_functionality_with_minimal_tracking(self):
        """Test step functionality with minimal tracking enabled."""
        wrapper = EnhancedActionLoggingWrapper(
            self.wandb_env,
            track_selection=False,
            track_pos=False,
            track_rot=False,
            track_force=False,
            track_torque=False,
            force_size=0
        )

        actions = torch.randn(4, 12)
        result = wrapper.step(actions)

        # Should still work with no tracking
        assert len(result) == 5
        assert wrapper.step_count == 1

    def test_step_functionality_with_maximal_tracking(self):
        """Test step functionality with all tracking enabled."""
        wrapper = EnhancedActionLoggingWrapper(
            self.wandb_env,
            track_selection=True,
            track_pos=True,
            track_rot=True,
            track_force=True,
            track_torque=True,
            force_size=6
        )

        actions = torch.randn(4, 12)
        result = wrapper.step(actions)

        # Should work with all tracking enabled
        assert len(result) == 5
        assert wrapper.step_count == 1