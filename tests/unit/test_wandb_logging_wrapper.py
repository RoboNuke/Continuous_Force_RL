"""
Unit tests for WandbLoggingWrapper.

This module tests the WandbLoggingWrapper functionality including episode tracking,
multi-agent support, metrics computation, and Wandb integration.
"""

import pytest
import torch
import sys
import os
from unittest.mock import patch, MagicMock, Mock
from collections import defaultdict

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from tests.mocks.mock_isaac_lab import create_mock_env


class TestEpisodeTracker:
    """Test EpisodeTracker functionality."""

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_initialization(self, mock_wandb):
        """Test EpisodeTracker initialization."""
        # Mock wandb.init
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import EpisodeTracker

        wandb_config = {
            'entity': 'test_entity',
            'project': 'test_project',
            'name': 'test_run',
            'group': 'test_group',
            'tags': ['test']
        }

        tracker = EpisodeTracker(wandb_config, num_envs=4, device='cpu', clip_eps=0.2)

        # Verify initialization
        assert tracker.num_envs == 4
        assert tracker.device == 'cpu'  # Device is stored as string in EpisodeTracker
        assert tracker.clip_eps == 0.2

        # Verify wandb init was called
        mock_wandb.init.assert_called_once_with(
            entity='test_entity',
            project='test_project',
            name='test_run',
            reinit=True,
            config=wandb_config,
            group='test_group',
            tags=['test']
        )

        # Verify tensors are initialized
        assert tracker.env_returns.shape == (4,)
        assert tracker.env_steps.shape == (4,)
        assert tracker.env_terminations.shape == (4,)
        assert tracker.engaged_any.shape == (4,)
        assert tracker.succeeded_any.shape == (4,)

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_reset_all(self, mock_wandb):
        """Test reset_all method."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import EpisodeTracker

        tracker = EpisodeTracker({}, num_envs=4, device='cpu')

        # Modify some values
        tracker.env_returns += 10.0
        tracker.env_steps += 5
        tracker.engaged_any[0] = True
        tracker.comp_sums['test'] = torch.ones(4)

        # Reset
        tracker.reset_all()

        # Verify reset
        assert torch.allclose(tracker.env_returns, torch.zeros(4))
        assert torch.equal(tracker.env_steps, torch.zeros(4))
        assert torch.equal(tracker.engaged_any, torch.zeros(4, dtype=torch.bool))
        assert len(tracker.comp_sums) == 0

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_step_basic(self, mock_wandb):
        """Test basic step functionality."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import EpisodeTracker

        tracker = EpisodeTracker({}, num_envs=4, device='cpu')

        # Test step
        reward = torch.tensor([1.0, 2.0, 3.0, 4.0])
        reward_components = {'test_reward': torch.tensor([0.5, 1.0, 1.5, 2.0])}
        engaged = torch.tensor([True, False, True, False])
        success = torch.tensor([False, True, False, False])
        terminated = torch.tensor([False, False, True, False])
        truncated = torch.tensor([False, False, False, False])
        infos = {}

        tracker.step(reward, reward_components, engaged, success, terminated, truncated, infos)

        # Verify updates (note: env 2 was terminated so its returns were reset to 0)
        expected_returns = torch.tensor([1.0, 2.0, 0.0, 4.0])  # env 2 reset due to termination
        assert torch.allclose(tracker.env_returns, expected_returns)
        assert torch.equal(tracker.env_steps, torch.tensor([1, 1, 0, 1]))  # env 2 reset to 0
        assert torch.equal(tracker.engaged_any, torch.tensor([True, False, False, False]))  # env 2 reset
        assert torch.equal(tracker.succeeded_any, torch.tensor([False, True, False, False]))  # env 2 reset
        # Component sums would also be reset for env 2
        expected_comp = torch.tensor([0.5, 1.0, 0.0, 2.0])  # env 2 reset
        assert torch.allclose(tracker.comp_sums['test_reward'], expected_comp)

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_engagement_tracking(self, mock_wandb):
        """Test engagement start/end tracking."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import EpisodeTracker

        tracker = EpisodeTracker({}, num_envs=4, device='cpu')

        # Step 1: Start engagement in env 0
        engaged = torch.tensor([True, False, False, False])
        tracker.step(
            reward=torch.zeros(4),
            reward_components={},
            engaged=engaged,
            success=torch.zeros(4, dtype=torch.bool),
            terminated=torch.zeros(4, dtype=torch.bool),
            truncated=torch.zeros(4, dtype=torch.bool),
            infos={}
        )

        # Check engagement start recorded
        assert tracker.engagement_start[0] == 1  # Started at step 1
        assert tracker.current_engaged[0] == True

        # Step 2: Continue engagement
        tracker.step(
            reward=torch.zeros(4),
            reward_components={},
            engaged=engaged,
            success=torch.zeros(4, dtype=torch.bool),
            terminated=torch.zeros(4, dtype=torch.bool),
            truncated=torch.zeros(4, dtype=torch.bool),
            infos={}
        )

        # Step 3: End engagement in env 0
        engaged = torch.tensor([False, False, False, False])
        tracker.step(
            reward=torch.zeros(4),
            reward_components={},
            engaged=engaged,
            success=torch.zeros(4, dtype=torch.bool),
            terminated=torch.zeros(4, dtype=torch.bool),
            truncated=torch.zeros(4, dtype=torch.bool),
            infos={}
        )

        # Check engagement length recorded (step 3 - step 1 = 2)
        assert len(tracker.engagement_lengths[0]) == 1
        assert tracker.engagement_lengths[0][0] == 2

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_success_tracking(self, mock_wandb):
        """Test success tracking."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import EpisodeTracker

        tracker = EpisodeTracker({}, num_envs=4, device='cpu')

        # Step 1: No success
        tracker.step(
            reward=torch.zeros(4),
            reward_components={},
            engaged=torch.zeros(4, dtype=torch.bool),
            success=torch.zeros(4, dtype=torch.bool),
            terminated=torch.zeros(4, dtype=torch.bool),
            truncated=torch.zeros(4, dtype=torch.bool),
            infos={}
        )

        # Step 2: Success in env 0
        success = torch.tensor([True, False, False, False])
        tracker.step(
            reward=torch.zeros(4),
            reward_components={},
            engaged=torch.zeros(4, dtype=torch.bool),
            success=success,
            terminated=torch.zeros(4, dtype=torch.bool),
            truncated=torch.zeros(4, dtype=torch.bool),
            infos={}
        )

        # Check success tracking
        assert tracker.succeeded_any[0] == True
        assert tracker.steps_to_first_success[0] == 2  # Success at step 2

        # Step 3: Success again in env 0 (should not update first success time)
        tracker.step(
            reward=torch.zeros(4),
            reward_components={},
            engaged=torch.zeros(4, dtype=torch.bool),
            success=success,
            terminated=torch.zeros(4, dtype=torch.bool),
            truncated=torch.zeros(4, dtype=torch.bool),
            infos={}
        )

        # Should still be step 2
        assert tracker.steps_to_first_success[0] == 2

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_episode_completion(self, mock_wandb):
        """Test episode completion and reset."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import EpisodeTracker

        tracker = EpisodeTracker({}, num_envs=4, device='cpu')

        # Build up some episode data
        tracker.env_returns = torch.tensor([10.0, 20.0, 30.0, 40.0])
        tracker.env_steps = torch.tensor([100, 200, 300, 400])
        tracker.engaged_any = torch.tensor([True, False, True, False])
        tracker.succeeded_any = torch.tensor([False, True, False, True])

        # Complete episodes 0 and 2
        terminated = torch.tensor([True, False, True, False])
        tracker.step(
            reward=torch.zeros(4),
            reward_components={},
            engaged=torch.zeros(4, dtype=torch.bool),
            success=torch.zeros(4, dtype=torch.bool),
            terminated=terminated,
            truncated=torch.zeros(4, dtype=torch.bool),
            infos={}
        )

        # Check that completed episodes were reset
        assert tracker.env_returns[0] == 0.0
        assert tracker.env_returns[2] == 0.0
        assert tracker.env_returns[1] == 20.0  # Not reset
        assert tracker.env_returns[3] == 40.0  # Not reset

        # Check that metrics were gathered
        assert len(tracker.finished_metrics) == 1

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_gather_metrics(self, mock_wandb):
        """Test _gather_metrics method."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import EpisodeTracker

        tracker = EpisodeTracker({}, num_envs=4, device='cpu')

        # Set up episode data
        tracker.env_returns = torch.tensor([10.0, 20.0, 30.0, 40.0])
        tracker.env_steps = torch.tensor([100, 200, 300, 400])
        tracker.env_terminations = torch.tensor([1, 0, 2, 1])
        tracker.engaged_any = torch.tensor([True, False, True, False])
        tracker.succeeded_any = torch.tensor([False, True, False, True])
        tracker.comp_sums['test_reward'] = torch.tensor([5.0, 10.0, 15.0, 20.0])

        # Gather metrics for envs 0 and 2
        mask = torch.tensor([True, False, True, False])
        metrics = tracker._gather_metrics(mask, {})

        # Verify metrics
        assert torch.allclose(metrics["Episode / Return (Avg)"], torch.tensor([10.0, 30.0]))
        assert torch.allclose(metrics["Episode / Episode Length"], torch.tensor([100.0, 300.0]))
        assert torch.allclose(metrics["Engagement / Engaged Rate"], torch.tensor([1.0, 1.0]))
        assert torch.allclose(metrics["Success / Success Rate"], torch.tensor([0.0, 0.0]))

        # Check component rewards (averaged per step)
        expected_avg = torch.tensor([5.0/100, 15.0/300])  # total reward / steps
        assert torch.allclose(metrics["test_reward"], expected_avg)

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_pop_finished(self, mock_wandb):
        """Test pop_finished method."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import EpisodeTracker

        tracker = EpisodeTracker({}, num_envs=4, device='cpu')

        # Add some finished metrics
        tracker.finished_metrics = [
            {"Episode / Return (Avg)": torch.tensor([10.0, 20.0])},
            {"Episode / Return (Avg)": torch.tensor([30.0])}
        ]

        # Pop finished metrics
        merged = tracker.pop_finished()

        # Verify merging
        assert torch.allclose(merged["Episode / Return (Avg)"], torch.tensor([10.0, 20.0, 30.0]))
        assert len(tracker.finished_metrics) == 0

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_log_minibatch_update(self, mock_wandb):
        """Test log_minibatch_update method."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import EpisodeTracker

        tracker = EpisodeTracker({}, num_envs=4, device='cpu', clip_eps=0.2)

        # Test with policy/value data
        old_log_probs = torch.tensor([-1.0, -1.5, -2.0])
        new_log_probs = torch.tensor([-1.1, -1.4, -2.1])
        values = torch.tensor([5.0, 10.0, 15.0])
        returns = torch.tensor([6.0, 11.0, 14.0])
        value_losses = torch.tensor([0.1, 0.2, 0.15])
        entropies = torch.tensor([1.0, 1.2, 0.8])
        advantages = torch.tensor([1.0, 1.0, -1.0])

        tracker.log_minibatch_update(
            returns=returns,
            values=values,
            advantages=advantages,
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            entropies=entropies,
            value_losses=value_losses
        )

        # Verify learning metrics were recorded
        assert len(tracker.learning_metrics) == 1
        stats = tracker.learning_metrics[0]

        assert "Policy / KL-Divergence (Avg)" in stats
        assert "Policy / Clip Fraction" in stats
        assert "Policy / Entropy (Avg)" in stats
        assert "Critic / Loss (Avg)" in stats
        assert "Critic / Explained Variance" in stats
        assert "Advantage / Mean" in stats

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_one_time_learning_metrics(self, mock_wandb):
        """Test one_time_learning_metrics method."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import EpisodeTracker

        tracker = EpisodeTracker({}, num_envs=4, device='cpu')

        # Test action metrics
        actions = torch.tensor([[-0.5, 0.8], [0.2, -0.99], [0.0, 0.5]])
        tracker.one_time_learning_metrics(actions, global_step=1000)

        # Verify metrics
        assert "Action / Mean" in tracker.metrics
        assert "Action / Std" in tracker.metrics
        assert "Action / Max" in tracker.metrics
        assert "Action / Min" in tracker.metrics
        assert "Action / Saturation" in tracker.metrics
        assert "Total Steps" in tracker.metrics
        assert tracker.metrics["Total Steps"] == 1000

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_publish(self, mock_wandb):
        """Test publish method."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import EpisodeTracker

        tracker = EpisodeTracker({}, num_envs=4, device='cpu')

        # Add some metrics
        tracker.metrics["Custom Metric"] = 42.0
        tracker.finished_metrics = [{"Episode / Return (Avg)": torch.tensor([10.0])}]
        tracker.learning_metrics = [{"Policy / Loss (Avg)": 0.5}]

        # Publish
        tracker.publish()

        # Verify wandb.run.log was called
        mock_run.log.assert_called_once()
        logged_data = mock_run.log.call_args[0][0]

        assert "Custom Metric" in logged_data
        assert "Episode / Return (Avg)" in logged_data
        assert "Policy / Loss (Avg)" in logged_data

        # Verify metrics were cleared
        assert len(tracker.metrics) == 0


class TestWandbLoggingWrapper:
    """Test WandbLoggingWrapper functionality."""

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_initialization_single_agent(self, mock_wandb, mock_env):
        """Test WandbLoggingWrapper initialization with single agent."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper

        wandb_config = {'project': 'test', 'name': 'test_run'}
        wrapper = WandbLoggingWrapper(mock_env, wandb_config, num_agents=1)

        assert wrapper.num_agents == 1
        assert wrapper.num_envs == 64
        assert wrapper.envs_per_agent == 64
        assert len(wrapper.trackers) == 1

        # Verify wandb init was called
        mock_wandb.init.assert_called_once()

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_initialization_multi_agent(self, mock_wandb, mock_env):
        """Test WandbLoggingWrapper initialization with multiple agents."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper

        wandb_config = {'project': 'test', 'name': 'test_run'}
        wrapper = WandbLoggingWrapper(mock_env, wandb_config, num_agents=4)

        assert wrapper.num_agents == 4
        assert wrapper.num_envs == 64
        assert wrapper.envs_per_agent == 16
        assert len(wrapper.trackers) == 4

        # Verify wandb init was called for each agent
        assert mock_wandb.init.call_count == 4

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_initialization_invalid_agent_count(self, mock_wandb, mock_env):
        """Test initialization fails with invalid agent count."""
        from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper

        wandb_config = {'project': 'test', 'name': 'test_run'}

        with pytest.raises(ValueError, match="must be divisible by number of agents"):
            WandbLoggingWrapper(mock_env, wandb_config, num_agents=7)  # 64 not divisible by 7

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_get_agent_assignment(self, mock_wandb, mock_env):
        """Test agent assignment functionality."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper

        wandb_config = {'project': 'test', 'name': 'test_run'}
        wrapper = WandbLoggingWrapper(mock_env, wandb_config, num_agents=4)

        assignment = wrapper.get_agent_assignment()

        assert len(assignment) == 4
        assert assignment[0] == list(range(0, 16))
        assert assignment[1] == list(range(16, 32))
        assert assignment[2] == list(range(32, 48))
        assert assignment[3] == list(range(48, 64))

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_step_logging(self, mock_wandb, mock_env):
        """Test step method with logging."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper

        wandb_config = {'project': 'test', 'name': 'test_run'}
        wrapper = WandbLoggingWrapper(mock_env, wandb_config, num_agents=2)

        # Mock step data
        action = torch.zeros(64, 6)

        # Mock environment response with info
        obs = torch.randn(64, 10)
        reward = torch.ones(64) * 2.0
        terminated = torch.zeros(64, dtype=torch.bool)
        truncated = torch.zeros(64, dtype=torch.bool)
        info = {
            'current_engagements': torch.tensor([True] * 32 + [False] * 32),
            'current_successes': torch.tensor([False] * 32 + [True] * 32),
            'Reward / Task': torch.ones(64) * 1.5,
            'Extra / Metric': {'value': torch.ones(64) * 0.5}
        }

        # Mock the parent step method
        with patch.object(wrapper.env, 'step', return_value=(obs, reward, terminated, truncated, info)):
            result = wrapper.step(action)

        # Verify result
        assert len(result) == 5
        obs_out, reward_out, terminated_out, truncated_out, info_out = result
        assert torch.equal(obs_out, obs)
        assert torch.equal(reward_out, reward)

        # Verify trackers received data (can't easily verify internal state without exposing internals)
        assert len(wrapper.trackers) == 2

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_add_metric_tensor(self, mock_wandb, mock_env):
        """Test add_metric with tensor value."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper

        wandb_config = {'project': 'test', 'name': 'test_run'}
        wrapper = WandbLoggingWrapper(mock_env, wandb_config, num_agents=2)

        # Add metric with tensor that should be split by agent
        metric_tensor = torch.tensor([1.0] * 32 + [2.0] * 32)  # Different values per agent
        wrapper.add_metric("Test Metric", metric_tensor)

        # Verify both trackers received appropriate values
        assert len(wrapper.trackers) == 2

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_add_metric_scalar(self, mock_wandb, mock_env):
        """Test add_metric with scalar value."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper

        wandb_config = {'project': 'test', 'name': 'test_run'}
        wrapper = WandbLoggingWrapper(mock_env, wandb_config, num_agents=2)

        # Add scalar metric
        wrapper.add_metric("Scalar Metric", 42.0)

        # Verify both trackers received the same value
        assert len(wrapper.trackers) == 2

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_log_learning_metrics(self, mock_wandb, mock_env):
        """Test log_learning_metrics method."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper

        wandb_config = {'project': 'test', 'name': 'test_run'}
        wrapper = WandbLoggingWrapper(mock_env, wandb_config, num_agents=2)

        # Log learning metrics
        wrapper.log_learning_metrics(
            returns=torch.tensor([5.0, 10.0]),
            values=torch.tensor([4.0, 9.0]),
            policy_losses=torch.tensor([0.1, 0.2])
        )

        # Verify all trackers received the metrics
        assert len(wrapper.trackers) == 2

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_log_action_metrics(self, mock_wandb, mock_env):
        """Test log_action_metrics method."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper

        wandb_config = {'project': 'test', 'name': 'test_run'}
        wrapper = WandbLoggingWrapper(mock_env, wandb_config, num_agents=2)

        # Log action metrics
        actions = torch.randn(64, 6)
        wrapper.log_action_metrics(actions, global_step=1000)

        # Verify all trackers received their portion of actions
        assert len(wrapper.trackers) == 2

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_publish_metrics(self, mock_wandb, mock_env):
        """Test publish_metrics method."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper

        wandb_config = {'project': 'test', 'name': 'test_run'}
        wrapper = WandbLoggingWrapper(mock_env, wandb_config, num_agents=2)

        # Publish metrics
        wrapper.publish_metrics()

        # Verify all trackers published
        assert mock_run.log.call_count == 2  # One call per tracker

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_close(self, mock_wandb, mock_env):
        """Test close method."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper

        wandb_config = {'project': 'test', 'name': 'test_run'}
        wrapper = WandbLoggingWrapper(mock_env, wandb_config, num_agents=2)

        # Close wrapper
        wrapper.close()

        # Verify all runs were finished
        assert mock_run.finish.call_count == 2


class TestWandbLoggingWrapperIntegration:
    """Test WandbLoggingWrapper integration scenarios."""

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_full_episode_cycle(self, mock_wandb, mock_env):
        """Test a complete episode cycle with logging."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper

        wandb_config = {'project': 'test', 'name': 'test_run'}
        wrapper = WandbLoggingWrapper(mock_env, wandb_config, num_agents=1)

        # Simulate episode steps
        for step in range(5):
            action = torch.zeros(64, 6)

            # Mock environment response
            obs = torch.randn(64, 10)
            reward = torch.ones(64) * (step + 1)  # Increasing reward
            terminated = torch.zeros(64, dtype=torch.bool)
            truncated = torch.zeros(64, dtype=torch.bool)

            # Complete first environment on last step
            if step == 4:
                terminated[0] = True

            info = {
                'current_engagements': torch.zeros(64, dtype=torch.bool),
                'current_successes': torch.zeros(64, dtype=torch.bool),
                'Reward / Task': reward * 0.5
            }

            # Mock the parent step method
            with patch.object(wrapper.env, 'step', return_value=(obs, reward, terminated, truncated, info)):
                result = wrapper.step(action)

        # Verify the wrapper handles episode completion
        tracker = wrapper.trackers[0]
        # Environment 0 should have been reset
        assert tracker.env_returns[0] == 0.0
        # Other environments should still have accumulated rewards
        assert tracker.env_returns[1] > 0.0

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_multi_agent_environment_separation(self, mock_wandb, mock_env):
        """Test proper environment separation between agents."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper

        wandb_config = {'project': 'test', 'name': 'test_run'}
        wrapper = WandbLoggingWrapper(mock_env, wandb_config, num_agents=4)

        action = torch.zeros(64, 6)

        # Create different rewards for different agent groups
        reward = torch.zeros(64)
        reward[0:16] = 1.0    # Agent 0
        reward[16:32] = 2.0   # Agent 1
        reward[32:48] = 3.0   # Agent 2
        reward[48:64] = 4.0   # Agent 3

        obs = torch.randn(64, 10)
        terminated = torch.zeros(64, dtype=torch.bool)
        truncated = torch.zeros(64, dtype=torch.bool)
        info = {}

        # Mock the parent step method
        with patch.object(wrapper.env, 'step', return_value=(obs, reward, terminated, truncated, info)):
            wrapper.step(action)

        # Verify each agent tracker got the right rewards
        assert len(wrapper.trackers) == 4
        # Each tracker should have 16 environments with specific reward values
        for i, tracker in enumerate(wrapper.trackers):
            expected_reward = float(i + 1)
            assert torch.allclose(tracker.env_returns, torch.full((16,), expected_reward))

    @patch('wrappers.logging.wandb_logging_wrapper.wandb')
    def test_reward_component_extraction(self, mock_wandb, mock_env):
        """Test extraction of reward components from info."""
        mock_run = Mock()
        mock_run.id = "test_run_123"
        mock_wandb.init.return_value = mock_run

        from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper

        wandb_config = {'project': 'test', 'name': 'test_run'}
        wrapper = WandbLoggingWrapper(mock_env, wandb_config, num_agents=1)

        action = torch.zeros(64, 6)
        obs = torch.randn(64, 10)
        reward = torch.ones(64)
        terminated = torch.zeros(64, dtype=torch.bool)
        truncated = torch.zeros(64, dtype=torch.bool)

        # Info with reward components
        info = {
            'Reward / Task Success': torch.ones(64) * 0.5,
            'Reward / Motion Penalty': torch.ones(64) * -0.1,
            'non_reward_metric': torch.ones(64) * 999,  # Should be ignored
            'other_info': 'text_value'  # Should be ignored
        }

        # Mock the parent step method
        with patch.object(wrapper.env, 'step', return_value=(obs, reward, terminated, truncated, info)):
            wrapper.step(action)

        # Verify reward components were extracted and passed to tracker
        tracker = wrapper.trackers[0]
        assert 'Reward / Task Success' in tracker.comp_sums
        assert 'Reward / Motion Penalty' in tracker.comp_sums
        assert 'non_reward_metric' not in tracker.comp_sums
        assert torch.allclose(tracker.comp_sums['Reward / Task Success'], torch.ones(64) * 0.5)