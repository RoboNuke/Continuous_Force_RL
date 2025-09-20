"""
Unit tests for factory_metrics_wrapper.py logging functionality.
Tests FactoryMetricsWrapper class and its integration with wandb wrapper.
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

from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper
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


class TestFactoryMetricsWrapper:
    """Test FactoryMetricsWrapper functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.base_env = MockBaseEnv()
        self.base_env.num_envs = 4
        self.base_env.device = torch.device("cpu")

        # Create wandb wrapper first
        self.wandb_env = MockWandbWrapper(self.base_env)

    def test_initialization_with_wandb_wrapper(self):
        """Test wrapper initialization with wandb wrapper dependency."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        assert wrapper.num_envs == 4
        assert wrapper.device == torch.device("cpu")
        assert wrapper.num_agents == 1
        assert wrapper.envs_per_agent == 4

    def test_initialization_without_wandb_wrapper_fails(self):
        """Test that initialization fails without wandb wrapper."""
        # Create environment without add_metrics method
        class MockEnvWithoutMetrics(gym.Env):
            def __init__(self):
                super().__init__()
                self.num_envs = 4
                self.device = torch.device("cpu")
                self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(64,))
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))
                self._unwrapped = self

            @property
            def unwrapped(self):
                return self._unwrapped

        env_without_metrics = MockEnvWithoutMetrics()

        with pytest.raises(ValueError, match="Factory Metrics Wrapper requires GenericWandbLoggingWrapper"):
            FactoryMetricsWrapper(env_without_metrics)

    def test_initialization_multiple_agents(self):
        """Test wrapper initialization with multiple agents."""
        wrapper = FactoryMetricsWrapper(self.wandb_env, num_agents=2)

        assert wrapper.num_agents == 2
        assert wrapper.envs_per_agent == 2

    def test_environment_division_validation(self):
        """Test validation of environment division by agents."""
        # 4 environments can't be evenly divided by 3 agents
        with pytest.raises(ValueError, match="Number of environments .* must be divisible"):
            FactoryMetricsWrapper(self.wandb_env, num_agents=3)

    def test_state_initialization(self):
        """Test internal state initialization."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        # Check tracking variables exist
        assert hasattr(wrapper, 'ep_succeeded')
        assert hasattr(wrapper, 'ep_success_times')
        assert hasattr(wrapper, 'ep_engaged')
        assert hasattr(wrapper, 'ep_engaged_times')
        assert hasattr(wrapper, 'ep_engaged_length')
        assert hasattr(wrapper, 'ep_ssv')
        assert hasattr(wrapper, 'ep_ssjv')

        # Check shapes
        assert wrapper.ep_succeeded.shape == (4,)
        assert wrapper.ep_success_times.shape == (4,)
        assert wrapper.ep_engaged.shape == (4,)
        assert wrapper.ep_engaged_times.shape == (4,)
        assert wrapper.ep_engaged_length.shape == (4,)
        assert wrapper.ep_ssv.shape == (4,)
        assert wrapper.ep_ssjv.shape == (4,)

    def test_step_basic_functionality(self):
        """Test basic step functionality."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        actions = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])

        result = wrapper.step(actions)

        # Check return format
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

        # Check shapes
        assert obs.shape[0] == 4
        assert reward.shape[0] == 4
        assert terminated.shape[0] == 4
        assert truncated.shape[0] == 4

    def test_force_torque_detection(self):
        """Test force/torque data detection."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        # Should detect if robot_force_torque is available
        assert hasattr(wrapper, 'has_force_data')
        # In our mock environment, this should be True because MockBaseEnv has robot_force_torque
        assert wrapper.has_force_data == True

    def test_wrapper_initialization_flag(self):
        """Test wrapper initialization flag."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        assert hasattr(wrapper, '_wrapper_initialized')
        # Should be True for mock environment with _robot
        assert wrapper._wrapper_initialized == True

    def test_reset_functionality(self):
        """Test environment reset."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        # Call reset
        result = wrapper.reset()

        # Check return format
        assert len(result) == 2
        obs, info = result
        assert obs.shape[0] == 4

    def test_close_functionality(self):
        """Test wrapper close method."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        # Should not raise any errors
        wrapper.close()

    def test_unwrapped_property(self):
        """Test unwrapped property access."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        # Should access the base environment
        assert wrapper.unwrapped == self.base_env

    def test_tracking_variables_device_placement(self):
        """Test that tracking variables are on correct device."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        # All tracking variables should be on the specified device
        assert wrapper.ep_succeeded.device == torch.device("cpu")
        assert wrapper.ep_success_times.device == torch.device("cpu")
        assert wrapper.ep_engaged.device == torch.device("cpu")
        assert wrapper.ep_engaged_times.device == torch.device("cpu")
        assert wrapper.ep_engaged_length.device == torch.device("cpu")
        assert wrapper.ep_ssv.device == torch.device("cpu")
        assert wrapper.ep_ssjv.device == torch.device("cpu")

    def test_tracking_variables_data_types(self):
        """Test that tracking variables have correct data types."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        # Boolean variables
        assert wrapper.ep_succeeded.dtype == torch.bool
        assert wrapper.ep_engaged.dtype == torch.bool

        # Long (integer) variables
        assert wrapper.ep_success_times.dtype == torch.long
        assert wrapper.ep_engaged_times.dtype == torch.long
        assert wrapper.ep_engaged_length.dtype == torch.long

        # Float variables
        assert wrapper.ep_ssv.dtype == torch.float32
        assert wrapper.ep_ssjv.dtype == torch.float32

    def test_tracking_variables_initial_values(self):
        """Test that tracking variables start with correct initial values."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        # Boolean variables should start False
        assert torch.all(wrapper.ep_succeeded == False)
        assert torch.all(wrapper.ep_engaged == False)

        # Numeric variables should start at zero
        assert torch.all(wrapper.ep_success_times == 0)
        assert torch.all(wrapper.ep_engaged_times == 0)
        assert torch.all(wrapper.ep_engaged_length == 0)
        assert torch.all(wrapper.ep_ssv == 0.0)
        assert torch.all(wrapper.ep_ssjv == 0.0)

    def test_force_data_initialization_when_available(self):
        """Test force/torque data initialization when robot_force_torque is available."""
        # Create a mock environment with robot_force_torque
        self.base_env.robot_force_torque = torch.randn(4, 6)
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        # Should detect force data and initialize tracking variables
        assert wrapper.has_force_data == True
        assert hasattr(wrapper, 'ep_max_force')
        assert hasattr(wrapper, 'ep_max_torque')
        assert hasattr(wrapper, 'ep_sum_force')
        assert hasattr(wrapper, 'ep_sum_torque')

        # Check shapes and device placement
        assert wrapper.ep_max_force.shape == (4,)
        assert wrapper.ep_max_torque.shape == (4,)
        assert wrapper.ep_sum_force.shape == (4,)
        assert wrapper.ep_sum_torque.shape == (4,)

        assert wrapper.ep_max_force.device == torch.device("cpu")
        assert wrapper.ep_max_torque.device == torch.device("cpu")
        assert wrapper.ep_sum_force.device == torch.device("cpu")
        assert wrapper.ep_sum_torque.device == torch.device("cpu")

        # Clean up
        delattr(self.base_env, 'robot_force_torque')

    def test_wrapper_with_robot_initialization(self):
        """Test wrapper initialization when robot is available."""
        # Create a mock environment with _robot
        self.base_env._robot = MagicMock()
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        # Should trigger wrapper initialization
        assert wrapper._wrapper_initialized == True

        # Clean up
        delattr(self.base_env, '_robot')

    def test_multiple_step_calls(self):
        """Test multiple consecutive step calls."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        actions = torch.randn(4, 6)

        # Call step multiple times
        for i in range(5):
            result = wrapper.step(actions)
            assert len(result) == 5

    def test_step_with_various_action_shapes(self):
        """Test step with various action tensor shapes."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        # Test with different action dimensions
        test_actions = [
            torch.randn(4, 6),
            torch.randn(4, 12),
            torch.zeros(4, 6),
            torch.ones(4, 6) * 0.5
        ]

        for actions in test_actions:
            result = wrapper.step(actions)
            assert len(result) == 5

    def test_dependency_validation_deep_chain(self):
        """Test dependency validation through wrapper chain."""
        # Create a chain: base_env -> wandb_wrapper -> other_wrapper -> factory_wrapper
        other_wrapper = gym.Wrapper(self.wandb_env)

        # This should still work because it finds add_metrics in the chain
        wrapper = FactoryMetricsWrapper(other_wrapper)
        assert wrapper is not None

    def test_error_handling_initialization(self):
        """Test error handling during initialization."""
        # Test with None environment
        with pytest.raises(AssertionError):
            FactoryMetricsWrapper(None)

        # Test with invalid num_agents
        with pytest.raises(ZeroDivisionError):
            FactoryMetricsWrapper(self.wandb_env, num_agents=0)

    def test_wrapper_chain_integration(self):
        """Test integration with wrapper chain."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        # Test that wrapper properly delegates to underlying environment
        assert wrapper.action_space.shape == self.base_env.action_space.shape
        assert wrapper.observation_space.shape == self.base_env.observation_space.shape

    def test_metrics_integration_readiness(self):
        """Test that wrapper is ready for metrics integration."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        # Should have access to add_metrics through the chain
        # The actual metrics sending will be tested in integration tests
        assert hasattr(wrapper.env, 'add_metrics')

    def test_memory_efficiency(self):
        """Test that wrapper doesn't create excessive memory overhead."""
        wrapper = FactoryMetricsWrapper(self.wandb_env)

        # Check that tracking variables are reasonably sized
        total_elements = (
            wrapper.ep_succeeded.numel() +
            wrapper.ep_success_times.numel() +
            wrapper.ep_engaged.numel() +
            wrapper.ep_engaged_times.numel() +
            wrapper.ep_engaged_length.numel() +
            wrapper.ep_ssv.numel() +
            wrapper.ep_ssjv.numel()
        )

        # Should be proportional to num_envs
        expected_elements = wrapper.num_envs * 7  # 7 tracking variables
        assert total_elements == expected_elements