"""
Unit tests for generic Wandb logging wrapper.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from wrappers.logging.generic_wandb_wrapper import GenericWandbLoggingWrapper, GenericEpisodeTracker
from wrappers.logging.logging_config import LoggingConfig, MetricConfig
from tests.mocks.mock_isaac_lab import MockEnvironment


class TestGenericEpisodeTracker:
    """Test GenericEpisodeTracker class."""

    @pytest.fixture
    def logging_config(self):
        """Create test logging configuration."""
        config = LoggingConfig()
        config.track_episodes = True
        config.track_terminations = True
        config.track_learning_metrics = True
        config.track_action_metrics = True

        # Add test metrics
        config.add_metric(MetricConfig(
            name="test_scalar",
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Test Scalar"
        ))

        config.add_metric(MetricConfig(
            name="test_boolean",
            metric_type="boolean",
            aggregation="mean",
            wandb_name="Test Boolean"
        ))

        return config

    @pytest.fixture
    def mock_wandb(self):
        """Mock wandb for testing."""
        with patch('wrappers.logging.generic_wandb_wrapper.wandb') as mock_wandb:
            mock_run = Mock()
            mock_run.id = "test_run_id"
            mock_wandb.init.return_value = mock_run
            yield mock_wandb

    @pytest.fixture
    def tracker(self, logging_config, mock_wandb):
        """Create test episode tracker."""
        return GenericEpisodeTracker(logging_config, num_envs=4, device=torch.device('cpu'))

    def test_initialization(self, tracker, mock_wandb):
        """Test tracker initialization."""
        assert tracker.num_envs == 4
        assert tracker.device == torch.device('cpu')
        assert tracker.config.track_episodes is True

        # Check wandb initialization
        mock_wandb.init.assert_called_once()

        # Check tracking variables
        assert hasattr(tracker, 'env_returns')
        assert hasattr(tracker, 'env_steps')
        assert hasattr(tracker, 'env_terminations')
        assert hasattr(tracker, 'metric_accumulators')

    def test_reset_all(self, tracker):
        """Test reset_all method."""
        # Modify some values
        tracker.env_returns.fill_(5.0)
        tracker.env_steps.fill_(10)
        tracker.metric_accumulators['test_scalar'].fill_(3.0)

        # Reset
        tracker.reset_all()

        # Check values are reset
        assert torch.all(tracker.env_returns == 0.0)
        assert torch.all(tracker.env_steps == 0)
        assert torch.all(tracker.metric_accumulators['test_scalar'] == 0.0)

    def test_step_basic(self, tracker):
        """Test basic step functionality."""
        reward = torch.tensor([1.0, 2.0, 3.0, 4.0])
        terminated = torch.tensor([False, False, False, False])  # No termination for basic test
        truncated = torch.tensor([False, False, False, False])
        infos = {
            'test_scalar': torch.tensor([0.5, 1.0, 1.5, 2.0]),
            'test_boolean': torch.tensor([True, False, True, False])
        }

        tracker.step(reward, terminated, truncated, infos)

        # Check core tracking
        assert torch.allclose(tracker.env_returns, reward)
        assert torch.all(tracker.env_steps == 1)
        assert torch.allclose(tracker.env_terminations, terminated.long())

        # Check metric tracking
        assert torch.allclose(tracker.metric_accumulators['test_scalar'], infos['test_scalar'])
        assert torch.equal(tracker.metric_accumulators['test_boolean'], infos['test_boolean'])

    def test_step_with_done(self, tracker):
        """Test step with episode completion."""
        reward = torch.tensor([1.0, 2.0, 3.0, 4.0])
        terminated = torch.tensor([False, False, True, False])
        truncated = torch.tensor([False, True, False, False])
        infos = {'test_scalar': torch.tensor([0.5, 1.0, 1.5, 2.0])}

        tracker.step(reward, terminated, truncated, infos)

        # Check that finished metrics were recorded
        assert len(tracker.finished_metrics) == 1

        # Check that done environments were reset
        done_mask = torch.logical_or(terminated, truncated)
        done_indices = torch.nonzero(done_mask, as_tuple=False).flatten()

        assert tracker.env_returns[done_indices].sum() == 0.0
        assert tracker.env_steps[done_indices].sum() == 0

    def test_extract_metric_value(self, tracker, logging_config):
        """Test metric value extraction."""
        infos = {
            'direct_metric': torch.tensor([1.0, 2.0, 3.0, 4.0]),
            'Reward': {
                'component_name': torch.tensor([0.5, 1.0, 1.5, 2.0])
            }
        }

        # Test direct metric extraction
        metric_config = MetricConfig(name="direct_metric")
        value = tracker._extract_metric_value(infos, metric_config)
        assert torch.allclose(value, infos['direct_metric'])

        # Test nested metric extraction
        metric_config = MetricConfig(name="Reward / component_name")
        value = tracker._extract_metric_value(infos, metric_config)
        assert torch.allclose(value, infos['Reward']['component_name'])

        # Test default value
        metric_config = MetricConfig(name="nonexistent_metric", default_value=5.0)
        value = tracker._extract_metric_value(infos, metric_config)
        assert torch.allclose(value, torch.full((4,), 5.0))

    def test_convert_to_tensor(self, tracker, logging_config):
        """Test value conversion to tensor."""
        metric_config = MetricConfig(name="test_metric")

        # Test tensor input
        tensor_input = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = tracker._convert_to_tensor(tensor_input, metric_config)
        assert torch.allclose(result, tensor_input)

        # Test list input
        list_input = [1.0, 2.0, 3.0, 4.0]
        result = tracker._convert_to_tensor(list_input, metric_config)
        assert torch.allclose(result, torch.tensor(list_input))

        # Test scalar input
        scalar_input = 5.0
        result = tracker._convert_to_tensor(scalar_input, metric_config)
        assert torch.allclose(result, torch.full((4,), 5.0))

    def test_gather_metrics(self, tracker):
        """Test metrics gathering for completed episodes."""
        # Set up some episode data
        tracker.env_returns = torch.tensor([10.0, 20.0, 30.0, 40.0])
        tracker.env_steps = torch.tensor([100, 200, 300, 400])
        tracker.env_terminations = torch.tensor([1, 0, 1, 0])
        tracker.metric_accumulators['test_scalar'] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        tracker.metric_accumulators['test_boolean'] = torch.tensor([True, False, True, False])

        # Gather metrics for environments 1 and 3
        mask = torch.tensor([False, True, False, True])
        infos = {}

        metrics = tracker._gather_metrics(mask, infos)

        # Check that metrics were gathered
        assert "Episode / Return (Avg)" in metrics
        assert "Episode / Episode Length" in metrics
        assert "Test Scalar" in metrics
        assert "Test Boolean" in metrics

        # Check values
        expected_returns = torch.tensor([20.0, 40.0])
        assert torch.allclose(metrics["Episode / Return (Avg)"], expected_returns)

    def test_pop_finished(self, tracker):
        """Test popping finished metrics."""
        # Add some finished metrics
        metrics1 = {"test_metric": torch.tensor([1.0, 2.0])}
        metrics2 = {"test_metric": torch.tensor([3.0, 4.0])}
        tracker.finished_metrics = [metrics1, metrics2]

        popped = tracker.pop_finished()

        # Check merged metrics
        assert "test_metric" in popped
        assert torch.allclose(popped["test_metric"], torch.tensor([1.0, 2.0, 3.0, 4.0]))

        # Check finished metrics cleared
        assert len(tracker.finished_metrics) == 0

    def test_pop_finished_empty(self, tracker):
        """Test popping finished metrics when empty."""
        popped = tracker.pop_finished()
        assert popped == {}

    def test_learning_metrics(self, tracker):
        """Test learning metrics functionality."""
        # Test log_minibatch_update
        tracker.log_minibatch_update(
            returns=torch.tensor([1.0, 2.0, 3.0]),
            values=torch.tensor([0.9, 1.9, 2.9]),
            advantages=torch.tensor([0.1, 0.1, 0.1]),
            old_log_probs=torch.tensor([-1.0, -1.1, -1.2]),
            new_log_probs=torch.tensor([-1.0, -1.0, -1.1]),
            entropies=torch.tensor([0.5, 0.6, 0.7]),
            policy_losses=torch.tensor([0.1, 0.2, 0.3]),
            value_losses=torch.tensor([0.05, 0.06, 0.07])
        )

        assert len(tracker.learning_metrics) == 1

        # Test aggregate_learning_metrics
        aggregated = tracker.aggregate_learning_metrics()
        assert "Policy / KL-Divergence (Avg)" in aggregated
        assert "Policy / Entropy (Avg)" in aggregated
        assert "Critic / Loss (Avg)" in aggregated

        # Check metrics cleared
        assert len(tracker.learning_metrics) == 0

    def test_action_metrics(self, tracker):
        """Test action metrics functionality."""
        actions = torch.tensor([0.5, -0.3, 0.8, -0.9])

        tracker.one_time_learning_metrics(actions, global_step=100)

        assert "Action / Mean" in tracker.metrics
        assert "Action / Std" in tracker.metrics
        assert "Action / Max" in tracker.metrics
        assert "Action / Min" in tracker.metrics
        assert "Action / Saturation" in tracker.metrics
        assert "Total Steps" in tracker.metrics
        assert tracker.metrics["Total Steps"] == 100


class TestGenericWandbLoggingWrapper:
    """Test GenericWandbLoggingWrapper class."""

    @pytest.fixture
    def mock_env(self):
        """Create mock environment."""
        env = MockEnvironment()
        env.unwrapped.num_envs = 4
        env.unwrapped.device = torch.device('cpu')
        return env

    @pytest.fixture
    def logging_config(self):
        """Create test logging configuration."""
        config = LoggingConfig()
        config.wandb_entity = "test_entity"
        config.wandb_project = "test_project"
        config.wandb_name = "test_name"

        config.add_metric(MetricConfig(
            name="test_metric",
            metric_type="scalar",
            wandb_name="Test Metric"
        ))

        return config

    @pytest.fixture
    def mock_wandb(self):
        """Mock wandb for testing."""
        with patch('wrappers.logging.generic_wandb_wrapper.wandb') as mock_wandb:
            mock_run = Mock()
            mock_run.id = "test_run_id"
            mock_wandb.init.return_value = mock_run
            yield mock_wandb

    @pytest.fixture
    def wrapper(self, mock_env, logging_config, mock_wandb):
        """Create test wrapper."""
        return GenericWandbLoggingWrapper(mock_env, logging_config, num_agents=1)

    def test_initialization(self, wrapper, mock_wandb):
        """Test wrapper initialization."""
        assert wrapper.num_agents == 1
        assert wrapper.num_envs == 4
        assert wrapper.envs_per_agent == 4
        assert len(wrapper.trackers) == 1

        # Check wandb was initialized
        mock_wandb.init.assert_called()

    def test_multi_agent_initialization(self, mock_env, logging_config, mock_wandb):
        """Test multi-agent wrapper initialization."""
        wrapper = GenericWandbLoggingWrapper(mock_env, logging_config, num_agents=2)

        assert wrapper.num_agents == 2
        assert wrapper.envs_per_agent == 2
        assert len(wrapper.trackers) == 2

        # Check wandb was initialized for each agent
        assert mock_wandb.init.call_count == 2

    def test_invalid_agent_assignment(self, mock_env, logging_config):
        """Test invalid agent assignment raises error."""
        with pytest.raises(ValueError, match="must be divisible"):
            GenericWandbLoggingWrapper(mock_env, logging_config, num_agents=3)

    def test_get_agent_assignment(self, wrapper):
        """Test agent assignment method."""
        assignments = wrapper.get_agent_assignment()

        assert len(assignments) == 1
        assert assignments[0] == [0, 1, 2, 3]

    def test_step(self, wrapper):
        """Test step method."""
        action = torch.tensor([1.0, 2.0, 3.0, 4.0])

        # Mock the environment step
        obs = torch.zeros((4, 10))
        reward = torch.tensor([1.0, 2.0, 3.0, 4.0])
        terminated = torch.tensor([False, False, True, False])
        truncated = torch.tensor([False, False, False, False])
        info = {'test_metric': torch.tensor([0.5, 1.0, 1.5, 2.0])}

        with patch.object(wrapper.env, 'step', return_value=(obs, reward, terminated, truncated, info)):
            result = wrapper.step(action)

        # Check return value
        assert len(result) == 5
        assert torch.allclose(result[1], reward)

    def test_extract_agent_info(self, wrapper):
        """Test agent info extraction."""
        info = {
            'scalar_metric': 5.0,
            'tensor_metric': torch.tensor([1.0, 2.0, 3.0, 4.0]),
            'nested_info': {
                'sub_metric': torch.tensor([0.1, 0.2, 0.3, 0.4])
            }
        }

        agent_info = wrapper._extract_agent_info(info, start_idx=0, end_idx=4)

        # Check scalar is preserved
        assert agent_info['scalar_metric'] == 5.0

        # Check tensor is sliced correctly
        assert torch.allclose(agent_info['tensor_metric'], torch.tensor([1.0, 2.0, 3.0, 4.0]))

        # Check nested info is handled
        assert 'nested_info' in agent_info
        assert torch.allclose(agent_info['nested_info']['sub_metric'], torch.tensor([0.1, 0.2, 0.3, 0.4]))

    def test_add_metric(self, wrapper):
        """Test adding metrics."""
        # Test with tensor value
        tensor_value = torch.tensor([1.0, 2.0, 3.0, 4.0])
        wrapper.add_metric("test_tensor", tensor_value)

        # Test with scalar value
        wrapper.add_metric("test_scalar", 5.0)

        # Should not raise errors (actual testing would require mocking tracker.add_metric)

    def test_close(self, wrapper):
        """Test closing wrapper."""
        # Mock the run objects
        for tracker in wrapper.trackers:
            tracker.run = Mock()

        wrapper.close()

        # Check that all runs were finished
        for tracker in wrapper.trackers:
            tracker.run.finish.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])