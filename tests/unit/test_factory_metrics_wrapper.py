"""
Unit tests for FactoryMetricsWrapper.

Tests the factory-specific metrics tracking including success rates,
engagement tracking, smoothness metrics, and force/torque statistics.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tests.mocks.mock_isaac_lab import MockEnvironment, MockCfg, MockCfgTask
from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper


class TestFactoryMetricsWrapper:
    """Test suite for FactoryMetricsWrapper."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""
        env = MockEnvironment(num_envs=8, device='cpu')

        # Add factory-specific attributes
        env.cfg_task = MockCfgTask()
        env.cfg_task.success_threshold = 0.02
        env.cfg_task.engage_threshold = 0.05
        env.cfg_task.name = "test_task"

        # Add robot and force/torque data
        env._robot = Mock()
        env.robot_force_torque = torch.zeros((8, 6), device='cpu')
        env.ee_linvel_fd = torch.zeros((8, 3), device='cpu')

        # Add episode tracking
        env.episode_length_buf = torch.zeros(8, dtype=torch.long, device='cpu')
        env.max_episode_length = 100

        # Add configuration
        env.cfg = MockCfg()
        env.cfg.decimation = 4

        return env

    @pytest.fixture
    def wrapper_single_agent(self, mock_env):
        """Create wrapper with single agent."""
        return FactoryMetricsWrapper(mock_env, num_agents=1)

    @pytest.fixture
    def wrapper_multi_agent(self, mock_env):
        """Create wrapper with multiple agents."""
        return FactoryMetricsWrapper(mock_env, num_agents=2)

    def test_initialization_single_agent(self, wrapper_single_agent):
        """Test wrapper initialization with single agent."""
        wrapper = wrapper_single_agent

        assert wrapper.num_agents == 1
        assert wrapper.num_envs == 8
        assert wrapper.envs_per_agent == 8
        assert str(wrapper.device) == 'cpu'
        assert wrapper.has_force_data == True

        # Check tracking variables are initialized
        assert wrapper.ep_succeeded.shape == (8,)
        assert wrapper.ep_success_times.shape == (8,)
        assert wrapper.ep_engaged.shape == (8,)
        assert wrapper.ep_engaged_times.shape == (8,)
        assert wrapper.ep_engaged_length.shape == (8,)
        assert wrapper.ep_ssv.shape == (8,)
        assert wrapper.ep_max_force.shape == (8,)
        assert wrapper.ep_max_torque.shape == (8,)
        assert wrapper.ep_sum_force.shape == (8,)
        assert wrapper.ep_sum_torque.shape == (8,)

    def test_initialization_multi_agent(self, wrapper_multi_agent):
        """Test wrapper initialization with multiple agents."""
        wrapper = wrapper_multi_agent

        assert wrapper.num_agents == 2
        assert wrapper.num_envs == 8
        assert wrapper.envs_per_agent == 4
        assert str(wrapper.device) == 'cpu'

    def test_initialization_invalid_agent_count(self, mock_env):
        """Test initialization with invalid agent count."""
        with pytest.raises(ValueError, match="Number of environments \\(8\\) must be divisible by number of agents \\(3\\)"):
            FactoryMetricsWrapper(mock_env, num_agents=3)

    def test_initialization_without_robot(self, mock_env):
        """Test initialization when robot is not available initially."""
        delattr(mock_env, '_robot')
        wrapper = FactoryMetricsWrapper(mock_env, num_agents=1)
        assert not wrapper._wrapper_initialized

    def test_initialization_without_force_data(self, mock_env):
        """Test initialization without force/torque data."""
        delattr(mock_env, 'robot_force_torque')
        wrapper = FactoryMetricsWrapper(mock_env, num_agents=1)
        assert wrapper.has_force_data == False
        assert not hasattr(wrapper, 'ep_max_force')

    def test_wrapper_initialization(self, wrapper_single_agent):
        """Test wrapper method override initialization."""
        wrapper = wrapper_single_agent

        # Should be initialized since robot exists
        assert wrapper._wrapper_initialized == True

        # Check that methods are stored and overridden
        assert wrapper._original_reset_buffers is not None
        assert wrapper._original_pre_physics_step is not None
        # _original_get_rewards should be set since MockEnvironment has _get_rewards
        assert wrapper._original_get_rewards is not None

    def test_wrapped_reset_buffers(self, wrapper_single_agent):
        """Test reset buffers functionality."""
        wrapper = wrapper_single_agent

        # Set some values
        wrapper.ep_succeeded[0:3] = True
        wrapper.ep_success_times[0:3] = 50
        wrapper.ep_engaged[0:3] = True
        wrapper.ep_engaged_times[0:3] = 25
        wrapper.ep_engaged_length[0:3] = 30
        wrapper.ep_ssv[0:3] = 1.5
        wrapper.ep_max_force[0:3] = 10.0
        wrapper.ep_max_torque[0:3] = 5.0
        wrapper.ep_sum_force[0:3] = 100.0
        wrapper.ep_sum_torque[0:3] = 50.0

        # Reset environments 0-2
        env_ids = torch.tensor([0, 1, 2])
        wrapper._wrapped_reset_buffers(env_ids)

        # Check that values are reset
        assert not wrapper.ep_succeeded[0:3].any()
        assert (wrapper.ep_success_times[0:3] == 0).all()
        assert not wrapper.ep_engaged[0:3].any()
        assert (wrapper.ep_engaged_times[0:3] == 0).all()
        assert (wrapper.ep_engaged_length[0:3] == 0).all()
        assert (wrapper.ep_ssv[0:3] == 0).all()
        assert (wrapper.ep_max_force[0:3] == 0).all()
        assert (wrapper.ep_max_torque[0:3] == 0).all()
        assert (wrapper.ep_sum_force[0:3] == 0).all()
        assert (wrapper.ep_sum_torque[0:3] == 0).all()

        # Check that other environments are unchanged
        assert wrapper.ep_succeeded[3:].sum() == 0  # Should still be False
        assert wrapper.ep_success_times[3:].sum() == 0  # Should still be 0

    def test_wrapped_pre_physics_step(self, wrapper_single_agent):
        """Test pre-physics step metrics updates."""
        wrapper = wrapper_single_agent

        # Set up velocity data
        wrapper.unwrapped.ee_linvel_fd = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [2.0, 2.0, 2.0],
            [0.1, 0.1, 0.1],
            [3.0, 0.0, 4.0]
        ], device='cpu')

        # Set up force/torque data
        wrapper.unwrapped.robot_force_torque = torch.tensor([
            [1.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0, 0.3],
            [1.0, 1.0, 1.0, 0.1, 0.1, 0.1],
            [0.5, 0.5, 0.5, 0.05, 0.05, 0.05],
            [2.0, 2.0, 2.0, 0.2, 0.2, 0.2],
            [0.1, 0.1, 0.1, 0.01, 0.01, 0.01],
            [3.0, 0.0, 4.0, 0.3, 0.0, 0.4]
        ], device='cpu')

        action = torch.zeros((8, 10), device='cpu')

        # Call wrapped pre-physics step
        wrapper._wrapped_pre_physics_step(action)

        # Check SSV updates (should be L2 norm of velocity)
        expected_ssv = torch.tensor([1.0, 2.0, 3.0, 1.732, 0.866, 3.464, 0.173, 5.0], device='cpu')
        torch.testing.assert_close(wrapper.ep_ssv, expected_ssv, atol=1e-3, rtol=1e-3)

        # Check force/torque updates
        expected_force_mag = torch.tensor([1.0, 2.0, 3.0, 1.732, 0.866, 3.464, 0.173, 5.0], device='cpu')
        expected_torque_mag = torch.tensor([0.1, 0.2, 0.3, 0.173, 0.087, 0.346, 0.017, 0.5], device='cpu')

        torch.testing.assert_close(wrapper.ep_sum_force, expected_force_mag, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(wrapper.ep_sum_torque, expected_torque_mag, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(wrapper.ep_max_force, expected_force_mag, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(wrapper.ep_max_torque, expected_torque_mag, atol=1e-3, rtol=1e-3)

    def test_wrapped_pre_physics_step_without_velocity(self, wrapper_single_agent):
        """Test pre-physics step without velocity data."""
        wrapper = wrapper_single_agent

        # Remove velocity attribute
        delattr(wrapper.unwrapped, 'ee_linvel_fd')

        action = torch.zeros((8, 10), device='cpu')
        wrapper._wrapped_pre_physics_step(action)

        # SSV should remain zero
        assert (wrapper.ep_ssv == 0).all()

    def test_wrapped_get_rewards(self, wrapper_single_agent):
        """Test get rewards with metrics tracking."""
        wrapper = wrapper_single_agent

        # Mock the original get_rewards method
        wrapper._original_get_rewards = Mock(return_value=torch.ones(8, device='cpu'))

        # Mock success and engagement detection
        with patch.object(wrapper, '_update_success_engagement_tracking') as mock_update_tracking, \
             patch.object(wrapper, '_update_extras') as mock_update_extras:

            result = wrapper._wrapped_get_rewards()

            # Check that methods were called
            mock_update_tracking.assert_called_once()
            mock_update_extras.assert_called_once()

            # Check reward buffer
            expected_rewards = torch.ones(8, device='cpu')
            torch.testing.assert_close(result, expected_rewards)

    def test_wrapped_get_rewards_without_original(self, wrapper_single_agent):
        """Test get rewards without original method."""
        wrapper = wrapper_single_agent
        wrapper._original_get_rewards = None

        with patch.object(wrapper, '_update_success_engagement_tracking'), \
             patch.object(wrapper, '_update_extras'):

            result = wrapper._wrapped_get_rewards()

            # Should return zeros
            expected_rewards = torch.zeros(8, device='cpu')
            torch.testing.assert_close(result, expected_rewards)

    def test_update_success_engagement_tracking(self, wrapper_single_agent):
        """Test success and engagement tracking updates."""
        wrapper = wrapper_single_agent

        # Mock success detection method
        def mock_get_curr_successes(threshold, check_rot):
            if threshold == 0.02:  # Success threshold
                return torch.tensor([True, False, True, False, False, True, False, False], device='cpu')
            elif threshold == 0.05:  # Engagement threshold
                return torch.tensor([True, True, True, True, False, True, True, False], device='cpu')
            return torch.zeros(8, dtype=torch.bool, device='cpu')

        wrapper.unwrapped._get_curr_successes = mock_get_curr_successes
        wrapper.unwrapped.episode_length_buf = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], device='cpu')
        wrapper.unwrapped.extras = {}

        # Call tracking update
        wrapper._update_success_engagement_tracking()

        # Check success tracking
        expected_successes = torch.tensor([True, False, True, False, False, True, False, False], device='cpu')
        torch.testing.assert_close(wrapper.ep_succeeded, expected_successes)

        # Check success times (only for first successes)
        expected_success_times = torch.tensor([10, 0, 30, 0, 0, 60, 0, 0], device='cpu')
        torch.testing.assert_close(wrapper.ep_success_times, expected_success_times)

        # Check engagement tracking
        expected_engaged = torch.tensor([True, True, True, True, False, True, True, False], device='cpu')
        torch.testing.assert_close(wrapper.ep_engaged, expected_engaged)

        # Check engagement times (only for first engagements)
        expected_engage_times = torch.tensor([10, 20, 30, 40, 0, 60, 70, 0], device='cpu')
        torch.testing.assert_close(wrapper.ep_engaged_times, expected_engage_times)

        # Check engagement lengths
        expected_engage_lengths = torch.tensor([1, 1, 1, 1, 0, 1, 1, 0], device='cpu')
        torch.testing.assert_close(wrapper.ep_engaged_length, expected_engage_lengths)

        # Check extras
        assert 'current_engagements' in wrapper.unwrapped.extras
        assert 'current_successes' in wrapper.unwrapped.extras

    def test_update_success_engagement_tracking_without_methods(self, wrapper_single_agent):
        """Test tracking update without success detection methods."""
        wrapper = wrapper_single_agent

        # Remove success detection method
        if hasattr(wrapper.unwrapped, '_get_curr_successes'):
            delattr(wrapper.unwrapped, '_get_curr_successes')

        # Should not crash
        wrapper._update_success_engagement_tracking()

        # All tracking should remain false/zero
        assert not wrapper.ep_succeeded.any()
        assert not wrapper.ep_engaged.any()

    def test_update_success_engagement_tracking_exceptions(self, wrapper_single_agent):
        """Test tracking update when methods raise exceptions."""
        wrapper = wrapper_single_agent

        # Mock method that raises exception
        wrapper.unwrapped._get_curr_successes = Mock(side_effect=Exception("Test error"))

        # Should not crash
        wrapper._update_success_engagement_tracking()

        # All tracking should remain false/zero
        assert not wrapper.ep_succeeded.any()
        assert not wrapper.ep_engaged.any()

    def test_update_extras_with_logging(self, wrapper_single_agent):
        """Test extras update when logging should occur."""
        wrapper = wrapper_single_agent
        wrapper.unwrapped.extras = {}

        # Set up some metric values
        wrapper.ep_succeeded[0] = True
        wrapper.ep_success_times[0] = 25
        wrapper.ep_engaged[0] = True
        wrapper.ep_engaged_times[0] = 10
        wrapper.ep_engaged_length[0] = 15
        wrapper.ep_ssv[0] = 2.5
        wrapper.ep_max_force[0] = 8.0
        wrapper.ep_max_torque[0] = 4.0
        wrapper.ep_sum_force[0] = 80.0
        wrapper.ep_sum_torque[0] = 40.0

        # Mock get_dones to return timeout
        wrapper.unwrapped._get_dones = Mock(return_value=(torch.zeros(8, dtype=torch.bool), torch.tensor([True] + [False]*7)))

        wrapper._update_extras()

        # Check that episode metrics are logged
        assert 'Episode / successes' in wrapper.unwrapped.extras
        assert 'Episode / success_times' in wrapper.unwrapped.extras
        assert 'Episode / engaged' in wrapper.unwrapped.extras
        assert 'Episode / engage_times' in wrapper.unwrapped.extras
        assert 'Episode / engage_lengths' in wrapper.unwrapped.extras

        # Check smoothness metrics
        assert 'smoothness' in wrapper.unwrapped.extras
        assert 'Smoothness / Sum Square Velocity' in wrapper.unwrapped.extras['smoothness']
        assert 'Smoothness / Avg Force' in wrapper.unwrapped.extras['smoothness']
        assert 'Smoothness / Max Force' in wrapper.unwrapped.extras['smoothness']
        assert 'Smoothness / Avg Torque' in wrapper.unwrapped.extras['smoothness']
        assert 'Smoothness / Max Torque' in wrapper.unwrapped.extras['smoothness']

    def test_update_extras_without_logging(self, wrapper_single_agent):
        """Test extras update when logging should not occur."""
        wrapper = wrapper_single_agent
        wrapper.unwrapped.extras = {}

        # Mock get_dones to return no timeout
        wrapper.unwrapped._get_dones = Mock(return_value=(torch.zeros(8, dtype=torch.bool), torch.zeros(8, dtype=torch.bool)))

        wrapper._update_extras()

        # Should not add metrics
        assert 'Episode / successes' not in wrapper.unwrapped.extras

    def test_update_extras_without_get_dones(self, wrapper_single_agent):
        """Test extras update without get_dones method."""
        wrapper = wrapper_single_agent
        wrapper.unwrapped.extras = {}

        # Mock the _get_dones method to raise AttributeError
        original_method = wrapper.unwrapped._get_dones

        def raise_attribute_error():
            raise AttributeError("'MockEnvironment' object has no attribute '_get_dones'")

        wrapper.unwrapped._get_dones = raise_attribute_error

        try:
            wrapper._update_extras()

            # Should still log (fallback behavior)
            assert 'Episode / successes' in wrapper.unwrapped.extras
        finally:
            # Restore original method
            wrapper.unwrapped._get_dones = original_method

    def test_update_extras_without_extras_attribute(self, wrapper_single_agent):
        """Test extras update without extras attribute."""
        wrapper = wrapper_single_agent

        # Remove extras attribute
        if hasattr(wrapper.unwrapped, 'extras'):
            delattr(wrapper.unwrapped, 'extras')

        # Should not crash
        wrapper._update_extras()

    def test_get_agent_assignment_single_agent(self, wrapper_single_agent):
        """Test agent assignment for single agent."""
        assignments = wrapper_single_agent.get_agent_assignment()

        expected = {0: [0, 1, 2, 3, 4, 5, 6, 7]}
        assert assignments == expected

    def test_get_agent_assignment_multi_agent(self, wrapper_multi_agent):
        """Test agent assignment for multiple agents."""
        assignments = wrapper_multi_agent.get_agent_assignment()

        expected = {
            0: [0, 1, 2, 3],
            1: [4, 5, 6, 7]
        }
        assert assignments == expected

    def test_get_success_stats(self, wrapper_single_agent):
        """Test success statistics calculation."""
        wrapper = wrapper_single_agent

        # Set up test data
        wrapper.ep_succeeded = torch.tensor([True, False, True, True, False, False, True, False], device='cpu')
        wrapper.ep_success_times = torch.tensor([10, 0, 20, 30, 0, 0, 40, 0], device='cpu')
        wrapper.ep_engaged = torch.tensor([True, True, True, True, False, True, True, False], device='cpu')
        wrapper.ep_engaged_times = torch.tensor([5, 8, 15, 25, 0, 35, 38, 0], device='cpu')
        wrapper.ep_engaged_length = torch.tensor([10, 15, 20, 25, 0, 30, 35, 0], device='cpu')

        stats = wrapper.get_success_stats()

        # Check calculations
        assert stats['success_rate'] == 0.5  # 4/8
        assert stats['avg_success_time'] == 25.0  # (10+20+30+40)/4
        assert stats['engagement_rate'] == 0.75  # 6/8
        assert stats['avg_engagement_time'] == pytest.approx(21.0, abs=1e-3)  # (5+8+15+25+35+38)/6
        assert stats['avg_engagement_length'] == pytest.approx(22.5, abs=1e-3)  # (10+15+20+25+30+35)/6

    def test_get_success_stats_no_successes(self, wrapper_single_agent):
        """Test success statistics with no successes."""
        wrapper = wrapper_single_agent

        # No successes or engagements
        wrapper.ep_succeeded = torch.zeros(8, dtype=torch.bool, device='cpu')
        wrapper.ep_engaged = torch.zeros(8, dtype=torch.bool, device='cpu')

        stats = wrapper.get_success_stats()

        assert stats['success_rate'] == 0.0
        assert stats['avg_success_time'] == 0.0
        assert stats['engagement_rate'] == 0.0
        assert stats['avg_engagement_time'] == 0.0
        assert stats['avg_engagement_length'] == 0.0

    def test_get_smoothness_stats(self, wrapper_single_agent):
        """Test smoothness statistics calculation."""
        wrapper = wrapper_single_agent

        # Set up test data
        wrapper.ep_ssv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device='cpu')
        wrapper.ep_max_force = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], device='cpu')
        wrapper.ep_max_torque = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device='cpu')
        wrapper.ep_sum_force = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0], device='cpu')
        wrapper.ep_sum_torque = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], device='cpu')

        stats = wrapper.get_smoothness_stats()

        # Check calculations
        assert stats['avg_ssv'] == 4.5  # mean of 1-8
        assert stats['std_ssv'] == pytest.approx(2.449, abs=1e-3)  # std of 1-8
        assert stats['avg_max_force'] == 45.0  # mean of 10-80 by 10s
        assert stats['avg_max_torque'] == 4.5  # mean of 1-8
        assert stats['avg_sum_force'] == 450.0  # mean of 100-800 by 100s
        assert stats['avg_sum_torque'] == 45.0  # mean of 10-80 by 10s

    def test_get_smoothness_stats_without_force_data(self, mock_env):
        """Test smoothness statistics without force data."""
        delattr(mock_env, 'robot_force_torque')
        wrapper = FactoryMetricsWrapper(mock_env, num_agents=1)

        wrapper.ep_ssv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device='cpu')

        stats = wrapper.get_smoothness_stats()

        # Should only have SSV stats
        assert stats['avg_ssv'] == 4.5
        assert stats['std_ssv'] == pytest.approx(2.449, abs=1e-3)
        assert 'avg_max_force' not in stats
        assert 'avg_max_torque' not in stats
        assert 'avg_sum_force' not in stats
        assert 'avg_sum_torque' not in stats

    def test_get_agent_metrics(self, wrapper_multi_agent):
        """Test agent-specific metrics calculation."""
        wrapper = wrapper_multi_agent

        # Set up test data for 8 environments, 2 agents (4 envs each)
        wrapper.ep_succeeded = torch.tensor([True, False, True, True, False, False, True, False], device='cpu')
        wrapper.ep_engaged = torch.tensor([True, True, False, True, True, False, True, True], device='cpu')
        wrapper.ep_ssv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device='cpu')
        wrapper.ep_max_force = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], device='cpu')
        wrapper.ep_max_torque = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device='cpu')

        # Agent 0 metrics (envs 0-3)
        metrics_0 = wrapper.get_agent_metrics(0)
        assert metrics_0['success_rate'] == 0.75  # 3/4 successes
        assert metrics_0['engagement_rate'] == 0.75  # 3/4 engagements
        assert metrics_0['avg_ssv'] == 2.5  # mean of [1,2,3,4]
        assert metrics_0['avg_max_force'] == 25.0  # mean of [10,20,30,40]
        assert metrics_0['avg_max_torque'] == 2.5  # mean of [1,2,3,4]

        # Agent 1 metrics (envs 4-7)
        metrics_1 = wrapper.get_agent_metrics(1)
        assert metrics_1['success_rate'] == 0.25  # 1/4 successes
        assert metrics_1['engagement_rate'] == 0.75  # 3/4 engagements
        assert metrics_1['avg_ssv'] == 6.5  # mean of [5,6,7,8]
        assert metrics_1['avg_max_force'] == 65.0  # mean of [50,60,70,80]
        assert metrics_1['avg_max_torque'] == 6.5  # mean of [5,6,7,8]

    def test_get_agent_metrics_invalid_agent(self, wrapper_multi_agent):
        """Test agent metrics with invalid agent ID."""
        with pytest.raises(ValueError, match="Agent ID \\(2\\) must be less than number of agents \\(2\\)"):
            wrapper_multi_agent.get_agent_metrics(2)

    def test_step_initialization(self, mock_env):
        """Test step method initializes wrapper when robot becomes available."""
        # Start without robot
        if hasattr(mock_env, '_robot'):
            delattr(mock_env, '_robot')

        wrapper = FactoryMetricsWrapper(mock_env, num_agents=1)
        assert not wrapper._wrapper_initialized

        # Add robot and call step
        mock_env._robot = Mock()
        action = torch.zeros((8, 10), device='cpu')

        with patch.object(wrapper.env, 'step', return_value=(torch.zeros(8, 10), torch.zeros(8), torch.zeros(8, dtype=torch.bool), torch.zeros(8, dtype=torch.bool), {})):
            wrapper.step(action)

        assert wrapper._wrapper_initialized

    def test_reset_initialization(self, mock_env):
        """Test reset method initializes wrapper when robot becomes available."""
        # Start without robot
        if hasattr(mock_env, '_robot'):
            delattr(mock_env, '_robot')

        wrapper = FactoryMetricsWrapper(mock_env, num_agents=1)
        assert not wrapper._wrapper_initialized

        # Add robot and call reset
        mock_env._robot = Mock()

        with patch.object(wrapper.env, 'reset', return_value=(torch.zeros(8, 10), {})):
            wrapper.reset()

        assert wrapper._wrapper_initialized

    def test_integration_full_episode(self, wrapper_single_agent):
        """Test full episode integration."""
        wrapper = wrapper_single_agent

        # Set up environment state
        wrapper.unwrapped.episode_length_buf = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], device='cpu')
        wrapper.unwrapped.ee_linvel_fd = torch.ones((8, 3), device='cpu')
        wrapper.unwrapped.robot_force_torque = torch.ones((8, 6), device='cpu') * 2.0
        wrapper.unwrapped.extras = {}

        # Mock success detection
        wrapper.unwrapped._get_curr_successes = Mock(side_effect=[
            torch.tensor([False, False, False, False, False, False, False, False], device='cpu'),  # Success check
            torch.tensor([True, True, False, True, False, False, True, False], device='cpu')      # Engagement check
        ])

        # Mock get_dones for logging
        wrapper.unwrapped._get_dones = Mock(return_value=(torch.zeros(8, dtype=torch.bool), torch.ones(8, dtype=torch.bool)))

        # Simulate pre-physics step
        action = torch.zeros((8, 10), device='cpu')
        wrapper._wrapped_pre_physics_step(action)

        # Simulate get rewards
        wrapper._original_get_rewards = Mock(return_value=torch.ones(8, device='cpu'))
        rewards = wrapper._wrapped_get_rewards()

        # Check that metrics are tracked and logged
        assert wrapper.ep_engaged.sum() > 0
        assert wrapper.ep_ssv.sum() > 0
        assert wrapper.ep_sum_force.sum() > 0
        assert 'Episode / engaged' in wrapper.unwrapped.extras
        assert 'smoothness' in wrapper.unwrapped.extras

        # Verify rewards are returned correctly
        expected_rewards = torch.ones(8, device='cpu')
        torch.testing.assert_close(rewards, expected_rewards)


if __name__ == '__main__':
    pytest.main([__file__])