"""
Integration tests for the logging wrapper system.

These tests verify that the different logging wrappers work correctly together
and integrate properly with the factory metrics wrapper and other components.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from wrappers.logging.generic_wandb_wrapper import GenericWandbLoggingWrapper
from wrappers.logging.logging_config import LoggingConfig, LoggingConfigPresets, MetricConfig, load_config_from_file
from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper
from tests.mocks.mock_isaac_lab import MockEnvironment


class TestFactoryMetricsAndLoggingIntegration:
    """Test integration between factory metrics wrapper and logging wrappers."""

    @pytest.fixture
    def mock_env(self):
        """Create mock factory environment."""
        env = MockEnvironment()
        env.unwrapped.num_envs = 4
        env.unwrapped.device = torch.device('cpu')

        # Add robot and scene for initialization
        env.unwrapped._robot = Mock()
        env.unwrapped.scene = Mock()

        # Add factory-specific attributes
        env.unwrapped.robot_force_torque = torch.zeros((4, 6))
        env.unwrapped.ee_linvel_fd = torch.zeros((4, 3))
        env.unwrapped.episode_length_buf = torch.ones(4)
        env.unwrapped.max_episode_length = 100

        # Add cfg for decimation
        env.unwrapped.cfg = Mock()
        env.unwrapped.cfg.decimation = 4

        # Add cfg_task
        env.unwrapped.cfg_task = Mock()
        env.unwrapped.cfg_task.engage_threshold = 0.05
        env.unwrapped.cfg_task.success_threshold = 0.02
        env.unwrapped.cfg_task.name = "factory_task"

        # Add extras and required methods
        env.unwrapped.extras = {}

        # Mock required methods
        env.unwrapped._reset_buffers = Mock()
        env.unwrapped._pre_physics_step = Mock()
        env.unwrapped._get_rewards = Mock(return_value=torch.zeros(4))
        env.unwrapped._get_dones = Mock(return_value=(torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)))

        return env

    @pytest.fixture
    def mock_wandb(self):
        """Mock wandb for testing."""
        with patch('wrappers.logging.generic_wandb_wrapper.wandb') as mock_wandb:
            mock_run = Mock()
            mock_run.id = "test_run_id"
            mock_wandb.init.return_value = mock_run
            yield mock_wandb

    def test_factory_metrics_to_wandb_flow(self, mock_env, mock_wandb):
        """Test complete flow from factory metrics to wandb logging."""
        # Wrap environment with factory metrics wrapper
        factory_wrapper = FactoryMetricsWrapper(mock_env)

        # Wrap with generic wandb wrapper using factory config
        config = LoggingConfigPresets.factory_config()
        config.wandb_entity = 'test_entity'
        config.wandb_project = 'test_project'
        config.wandb_name = 'test_name'
        wandb_wrapper = GenericWandbLoggingWrapper(factory_wrapper, config)

        # Simulate environment step
        action = torch.tensor([1.0, 2.0, 3.0, 4.0])

        # Mock step return values
        obs = torch.zeros((4, 10))
        reward = torch.tensor([1.0, 2.0, 3.0, 4.0])
        terminated = torch.tensor([False, False, True, False])
        truncated = torch.tensor([False, False, False, False])
        info = {}

        with patch.object(factory_wrapper, 'step', return_value=(obs, reward, terminated, truncated, info)):
            # Set up some factory metrics in extras
            factory_wrapper.unwrapped.extras['current_engagements'] = torch.tensor([1.0, 0.0, 1.0, 0.0])
            factory_wrapper.unwrapped.extras['current_successes'] = torch.tensor([0.0, 1.0, 0.0, 1.0])

            result = wandb_wrapper.step(action)

        # Check that step completed successfully
        assert len(result) == 5
        assert torch.allclose(result[1], reward)

        # Verify that wandb tracking was set up
        assert len(wandb_wrapper.trackers) == 1
        mock_wandb.init.assert_called_once()

    def test_multi_agent_factory_integration(self, mock_env, mock_wandb):
        """Test multi-agent integration with factory metrics."""
        # Adjust environment for multi-agent (8 envs for 2 agents)
        mock_env.unwrapped.num_envs = 8
        mock_env.unwrapped.robot_force_torque = torch.zeros((8, 6))
        mock_env.unwrapped.ee_linvel_fd = torch.zeros((8, 3))
        mock_env.unwrapped.episode_length_buf = torch.ones(8)

        # Wrap with factory metrics
        factory_wrapper = FactoryMetricsWrapper(mock_env, num_agents=2)

        # Wrap with generic wandb wrapper using factory config
        config = LoggingConfigPresets.factory_config()
        wandb_wrapper = GenericWandbLoggingWrapper(factory_wrapper, config, num_agents=2)

        # Check agent assignment
        assignments = wandb_wrapper.get_agent_assignment()
        assert len(assignments) == 2
        assert assignments[0] == [0, 1, 2, 3]
        assert assignments[1] == [4, 5, 6, 7]

        # Check wandb trackers
        assert len(wandb_wrapper.trackers) == 2
        assert mock_wandb.init.call_count == 2

    def test_configurable_metrics_integration(self, mock_env, mock_wandb):
        """Test integration with custom metric configuration."""
        # Create custom logging config
        config = LoggingConfig()
        config.wandb_project = "test_project"

        # Add custom metrics
        config.add_metric(MetricConfig(
            name="custom_engagement",
            metric_type="boolean",
            wandb_name="Custom Engagement Rate"
        ))

        config.add_metric(MetricConfig(
            name="custom_force",
            metric_type="scalar",
            wandb_name="Custom Force Metric"
        ))

        # Wrap environment
        factory_wrapper = FactoryMetricsWrapper(mock_env)
        wandb_wrapper = GenericWandbLoggingWrapper(factory_wrapper, config)

        # Check configuration was applied
        assert 'custom_engagement' in wandb_wrapper.config.tracked_metrics
        assert 'custom_force' in wandb_wrapper.config.tracked_metrics

    def test_force_metrics_integration(self, mock_env, mock_wandb):
        """Test force/torque metrics flow from wrapper to logging."""
        # Set up force data
        mock_env.unwrapped.robot_force_torque = torch.tensor([
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
            [2.0, 3.0, 4.0, 0.2, 0.3, 0.4],
            [3.0, 4.0, 5.0, 0.3, 0.4, 0.5],
            [4.0, 5.0, 6.0, 0.4, 0.5, 0.6]
        ])

        # Wrap environment
        factory_wrapper = FactoryMetricsWrapper(mock_env)
        config = LoggingConfigPresets.factory_config()
        wandb_wrapper = GenericWandbLoggingWrapper(factory_wrapper, config)

        # Initialize wrappers
        factory_wrapper._initialize_wrapper()

        # Simulate some steps to accumulate force data
        for _ in range(5):
            # Update force statistics (simulate pre_physics_step)
            if hasattr(factory_wrapper, '_wrapped_pre_physics_step'):
                factory_wrapper._wrapped_pre_physics_step(torch.zeros(4))

        # Check that force metrics are being tracked
        assert hasattr(factory_wrapper, 'ep_max_force')
        assert hasattr(factory_wrapper, 'ep_sum_force')

        # Force the _update_extras to think we should log by mocking _get_dones to return timeout
        factory_wrapper.unwrapped._get_dones = Mock(return_value=(torch.zeros(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool)))

        # Check that force metrics would be available in extras/info
        factory_wrapper._update_extras()
        assert 'smoothness' in factory_wrapper.unwrapped.extras

    def test_learning_metrics_integration(self, mock_env, mock_wandb):
        """Test learning metrics logging integration."""
        factory_wrapper = FactoryMetricsWrapper(mock_env)
        config = LoggingConfigPresets.factory_config()
        wandb_wrapper = GenericWandbLoggingWrapper(factory_wrapper, config)

        # Test learning metrics logging
        wandb_wrapper.log_learning_metrics(
            returns=torch.tensor([1.0, 2.0, 3.0]),
            values=torch.tensor([0.9, 1.9, 2.9]),
            policy_losses=torch.tensor([0.1, 0.2, 0.3]),
            value_losses=torch.tensor([0.05, 0.06, 0.07])
        )

        # Test action metrics logging
        actions = torch.tensor([0.5, -0.3, 0.8, -0.9])
        wandb_wrapper.log_action_metrics(actions, global_step=100)

        # Should not raise errors and metrics should be logged
        for tracker in wandb_wrapper.trackers:
            assert len(tracker.learning_metrics) == 1

    def test_publish_metrics_integration(self, mock_env, mock_wandb):
        """Test end-to-end metrics publishing."""
        factory_wrapper = FactoryMetricsWrapper(mock_env)
        config = LoggingConfigPresets.factory_config()
        wandb_wrapper = GenericWandbLoggingWrapper(factory_wrapper, config)

        # Simulate an episode with some data
        action = torch.tensor([1.0, 2.0, 3.0, 4.0])
        obs = torch.zeros((4, 10))
        reward = torch.tensor([1.0, 2.0, 3.0, 4.0])
        terminated = torch.tensor([False, False, True, False])
        truncated = torch.tensor([False, False, False, False])
        info = {
            'current_engagements': torch.tensor([1.0, 0.0, 1.0, 0.0]),
            'Reward / reach_reward': torch.tensor([0.1, 0.2, 0.3, 0.4])
        }

        with patch.object(factory_wrapper, 'step', return_value=(obs, reward, terminated, truncated, info)):
            # Step multiple times
            for _ in range(3):
                wandb_wrapper.step(action)

        # Publish metrics
        wandb_wrapper.publish_metrics()

        # Check that wandb.log was called
        for tracker in wandb_wrapper.trackers:
            tracker.run.log.assert_called()


class TestGenericLoggingIntegration:
    """Test integration of generic logging wrapper with different environments."""

    @pytest.fixture
    def mock_basic_env(self):
        """Create mock basic environment."""
        env = MockEnvironment()
        env.unwrapped.num_envs = 4
        env.unwrapped.device = torch.device('cpu')
        return env

    @pytest.fixture
    def mock_wandb(self):
        """Mock wandb for testing."""
        with patch('wrappers.logging.generic_wandb_wrapper.wandb') as mock_wandb:
            mock_run = Mock()
            mock_run.id = "test_run_id"
            mock_wandb.init.return_value = mock_run
            yield mock_wandb

    def test_basic_environment_integration(self, mock_basic_env, mock_wandb):
        """Test generic wrapper with basic environment."""
        config = LoggingConfigPresets.basic_config()
        config.wandb_project = "test_project"

        wrapper = GenericWandbLoggingWrapper(mock_basic_env, config)

        # Simulate step
        action = torch.tensor([1.0, 2.0, 3.0, 4.0])
        obs = torch.zeros((4, 10))
        reward = torch.tensor([1.0, 2.0, 3.0, 4.0])
        terminated = torch.tensor([False, False, True, False])
        truncated = torch.tensor([False, False, False, False])
        info = {'reward': reward}

        with patch.object(mock_basic_env, 'step', return_value=(obs, reward, terminated, truncated, info)):
            result = wrapper.step(action)

        assert len(result) == 5
        assert torch.allclose(result[1], reward)

    def test_locomotion_environment_integration(self, mock_basic_env, mock_wandb):
        """Test generic wrapper with locomotion configuration."""
        config = LoggingConfigPresets.locomotion_config()
        config.wandb_project = "test_project"

        wrapper = GenericWandbLoggingWrapper(mock_basic_env, config)

        # Simulate step with locomotion metrics
        action = torch.tensor([1.0, 2.0, 3.0, 4.0])
        obs = torch.zeros((4, 10))
        reward = torch.tensor([1.0, 2.0, 3.0, 4.0])
        terminated = torch.tensor([False, False, True, False])
        truncated = torch.tensor([False, False, False, False])
        info = {
            'reward': reward,
            'distance_traveled': torch.tensor([5.0, 6.0, 7.0, 8.0]),
            'velocity': torch.tensor([1.0, 1.2, 1.4, 1.6])
        }

        with patch.object(mock_basic_env, 'step', return_value=(obs, reward, terminated, truncated, info)):
            result = wrapper.step(action)

        assert len(result) == 5
        assert torch.allclose(result[1], reward)

        # Check that locomotion metrics are tracked
        assert 'distance_traveled' in wrapper.config.tracked_metrics
        assert 'velocity' in wrapper.config.tracked_metrics

    def test_config_file_integration(self, mock_basic_env, mock_wandb):
        """Test integration with configuration files."""
        import tempfile
        import yaml

        config_dict = {
            'wandb_project': 'integration_test',
            'num_agents': 1,
            'tracked_metrics': {
                'custom_metric': {
                    'default_value': 0.0,
                    'metric_type': 'scalar',
                    'wandb_name': 'Custom Test Metric'
                }
            }
        }

        # Skip if PyYAML not available
        pytest.importorskip("yaml")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            config = load_config_from_file(temp_path)

            wrapper = GenericWandbLoggingWrapper(mock_basic_env, config)

            assert 'custom_metric' in wrapper.config.tracked_metrics
            assert wrapper.config.wandb_project == 'integration_test'

        finally:
            import os
            os.unlink(temp_path)


class TestBackwardCompatibility:
    """Test backward compatibility with legacy wandb wrapper."""

    @pytest.fixture
    def mock_env(self):
        """Create mock environment."""
        env = MockEnvironment()
        env.unwrapped.num_envs = 4
        env.unwrapped.device = torch.device('cpu')
        return env

    @pytest.fixture
    def mock_wandb(self):
        """Mock wandb for testing."""
        with patch('wrappers.logging.wandb_logging_wrapper.wandb') as mock_wandb:
            mock_run = Mock()
            mock_run.id = "test_run_id"
            mock_wandb.init.return_value = mock_run
            yield mock_wandb

    def test_legacy_wrapper_still_works(self, mock_env, mock_wandb):
        """Test that legacy wrapper still functions for backward compatibility."""
        from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper

        wandb_config = {
            'entity': 'test_entity',
            'project': 'test_project',
            'name': 'test_name'
        }

        # Should still work but be deprecated
        wrapper = WandbLoggingWrapper(mock_env, wandb_config)

        # Check basic functionality
        assert wrapper.num_envs == 4
        assert wrapper.num_agents == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])