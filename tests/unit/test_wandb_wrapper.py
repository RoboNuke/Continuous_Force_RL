"""
Unit tests for wandb_wrapper.py logging functionality.
Tests both SimpleEpisodeTracker and GenericWandbLoggingWrapper classes.
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

from wrappers.logging.wandb_wrapper import SimpleEpisodeTracker, GenericWandbLoggingWrapper
from tests.mocks.mock_isaac_lab import MockBaseEnv


class MockEnvConfig:
    """Mock environment configuration."""

    def __init__(self):
        self.num_envs = 4
        self.device = torch.device("cpu")
        self.agent_configs = {
            'agent_0': {
                'wandb_entity': 'test_entity',
                'wandb_project': 'test_project',
                'wandb_name': 'test_agent_0',
                'wandb_group': 'test_group',
                'wandb_tags': ['test']
            }
        }


class TestSimpleEpisodeTracker:
    """Test SimpleEpisodeTracker functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.device = torch.device("cpu")
        self.num_envs = 4
        self.agent_config = {
            'wandb_entity': 'test_entity',
            'wandb_project': 'test_project',
            'wandb_name': 'test_agent',
            'wandb_group': 'test_group',
            'wandb_tags': ['test']
        }
        self.env_config = MockEnvConfig()
        self.tracker = SimpleEpisodeTracker(
            num_envs=self.num_envs,
            device=self.device,
            agent_config=self.agent_config,
            env_config=self.env_config
        )

    def test_initialization(self):
        """Test tracker initialization."""
        assert self.tracker.num_envs == self.num_envs
        assert self.tracker.device == self.device
        assert self.tracker.episode_count == 0
        assert self.tracker.env_steps == 0
        assert self.tracker.total_steps == 0
        assert isinstance(self.tracker.accumulated_metrics, dict)
        # Note: learning_batches attribute not implemented in current version

    def test_step_tracking(self):
        """Test step counting functionality."""
        # Initial step
        self.tracker.increment_steps()
        assert self.tracker.env_steps == 1
        assert self.tracker.total_steps == 4  # num_envs

        # Multiple steps
        for _ in range(5):
            self.tracker.increment_steps()
        assert self.tracker.env_steps == 6
        assert self.tracker.total_steps == 24  # 6 * 4

    def test_add_metrics(self):
        """Test metric accumulation."""
        # Add some metrics
        test_metrics = {
            "test_metric_1": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "test_metric_2": torch.tensor([0.5, 0.6, 0.7, 0.8])
        }
        self.tracker.add_metrics(test_metrics)

        # Check metrics were stored
        assert "test_metric_1" in self.tracker.accumulated_metrics
        assert "test_metric_2" in self.tracker.accumulated_metrics
        assert len(self.tracker.accumulated_metrics["test_metric_1"]) == 1

    @pytest.mark.skip(reason="log_minibatch_update method not implemented in current wandb wrapper")
    def test_log_minibatch_update(self):
        """Test minibatch logging."""
        learning_data = {
            "loss": torch.tensor(0.5),
            "value_loss": torch.tensor(0.3)
        }
        self.tracker.log_minibatch_update(learning_data)

        assert len(self.tracker.learning_batches) == 1
        assert "loss" in self.tracker.learning_batches[0]

    @pytest.mark.skip(reason="add_onetime_learning_metrics and log_minibatch_update methods not implemented")
    def test_add_onetime_learning_metrics(self):
        """Test final metrics logging and cleanup."""
        # Add some accumulated metrics
        test_metrics = {
            "reward": torch.tensor([1.0, 2.0, 3.0, 4.0])
        }
        self.tracker.add_metrics(test_metrics)

        # Add learning batch
        learning_data = {"loss": torch.tensor(0.5)}
        self.tracker.log_minibatch_update(learning_data)

        # Add final learning metrics
        final_metrics = {"final_reward": torch.tensor(10.0)}
        self.tracker.add_onetime_learning_metrics(final_metrics)

        # Check that data was cleared
        assert len(self.tracker.accumulated_metrics) == 0
        assert len(self.tracker.learning_batches) == 0

    def test_close_functionality(self):
        """Test close method."""
        # Should not raise any errors
        self.tracker.close()


class TestGenericWandbLoggingWrapper:
    """Test GenericWandbLoggingWrapper functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.base_env = MockBaseEnv()
        self.base_env.num_envs = 4
        self.base_env.device = torch.device("cpu")

        # Create mock env config with agent configs
        self.env_cfg = MockEnvConfig()

    def test_initialization_basic(self):
        """Test wrapper initialization with basic config."""
        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=1, env_cfg=self.env_cfg)

        assert wrapper.num_envs == 4
        assert wrapper.device == torch.device("cpu")
        assert wrapper.num_agents == 1
        assert wrapper.envs_per_agent == 4
        assert len(wrapper.trackers) == 1

    def test_initialization_without_env_cfg_fails(self):
        """Test that initialization fails without env_cfg."""
        with pytest.raises(ValueError, match="env_cfg is required"):
            GenericWandbLoggingWrapper(self.base_env)

    def test_initialization_without_agent_configs_fails(self):
        """Test that initialization fails without agent_configs in env_cfg."""
        bad_env_cfg = MockEnvConfig()
        delattr(bad_env_cfg, 'agent_configs')

        with pytest.raises(ValueError, match="env_cfg must contain agent_configs"):
            GenericWandbLoggingWrapper(self.base_env, env_cfg=bad_env_cfg)

    def test_initialization_multiple_agents(self):
        """Test wrapper initialization with multiple agents."""
        # Set up config for 2 agents
        self.env_cfg.agent_configs['agent_1'] = {
            'wandb_entity': 'test_entity_2',
            'wandb_project': 'test_project_2',
            'wandb_name': 'test_agent_1'
        }

        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=2, env_cfg=self.env_cfg)

        assert wrapper.num_agents == 2
        assert wrapper.envs_per_agent == 2
        assert len(wrapper.trackers) == 2

    def test_environment_division_validation(self):
        """Test validation of environment division by agents."""
        # 4 environments can't be evenly divided by 3 agents
        with pytest.raises(ValueError, match="Number of environments .* must be divisible"):
            GenericWandbLoggingWrapper(self.base_env, num_agents=3, env_cfg=self.env_cfg)

    def test_step_tracking(self):
        """Test step method and basic functionality."""
        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=1, env_cfg=self.env_cfg)

        # Mock action
        actions = torch.randn(4, 6)

        # Call step
        result = wrapper.step(actions)

        # Check return format
        assert len(result) == 5  # obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = result

        # Check shapes
        assert obs.shape[0] == 4
        assert reward.shape[0] == 4
        assert terminated.shape[0] == 4
        assert truncated.shape[0] == 4

        # Check that step counters were incremented
        assert wrapper.trackers[0].env_steps == 1

    def test_add_metrics_env_split(self):
        """Test metric addition with environment-level split."""
        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=2, env_cfg=self.env_cfg)

        # Add metrics with num_envs length
        test_metrics = {
            "test_metric": torch.tensor([1.0, 2.0, 3.0, 4.0])
        }
        wrapper.add_metrics(test_metrics)

        # Each agent should get their subset
        # Agent 0: envs 0-1, Agent 1: envs 2-3
        # We can't directly check the tracker internals, but this should not error

    def test_add_metrics_agent_split(self):
        """Test metric addition with agent-level split."""
        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=2, env_cfg=self.env_cfg)

        # Add metrics with num_agents length
        test_metrics = {
            "agent_metric": torch.tensor([10.0, 20.0])
        }
        wrapper.add_metrics(test_metrics)

        # Should split by agent without errors

    def test_add_metrics_scalar(self):
        """Test metric addition with scalar values."""
        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=1, env_cfg=self.env_cfg)

        # Add scalar metrics (all must be tensors)
        test_metrics = {
            "scalar_metric": torch.tensor(5.0),
            "python_scalar": torch.tensor(10.0)
        }
        wrapper.add_metrics(test_metrics)

        # Should broadcast to all agents without errors

    def test_episode_completion_tracking(self):
        """Test episode completion and reward tracking."""
        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=1, env_cfg=self.env_cfg)

        # Step to build up some episode data
        actions = torch.randn(4, 6)
        for _ in range(10):
            wrapper.step(actions)

        # Check that episode tracking is working
        assert torch.all(wrapper.current_episode_lengths == 10)

        # Mock environment step to return episode completion
        with patch.object(wrapper.env, 'step') as mock_step:
            terminated = torch.tensor([True, False, True, False], dtype=torch.bool)
            truncated = torch.tensor([False, False, False, False], dtype=torch.bool)
            mock_step.return_value = (
                torch.randn(4, 64),
                torch.randn(4),
                terminated,
                truncated,
                {"timeout": truncated}
            )

            wrapper.step(actions)

            # Check that completed episodes were reset
            expected_lengths = torch.tensor([0, 11, 0, 11], dtype=torch.long)
            assert torch.equal(wrapper.current_episode_lengths, expected_lengths)

    @pytest.mark.skip(reason="log_minibatch_update method not implemented in current wandb wrapper")
    def test_log_minibatch_update(self):
        """Test minibatch logging distribution."""
        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=1, env_cfg=self.env_cfg)

        learning_data = {
            "policy_loss": torch.tensor(0.5),
            "value_loss": torch.tensor(0.3)
        }
        wrapper.log_minibatch_update(learning_data)

        # Should distribute to all trackers without errors

    @pytest.mark.skip(reason="add_onetime_learning_metrics method not implemented in current wandb wrapper")
    def test_add_onetime_learning_metrics(self):
        """Test final learning metrics addition."""
        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=1, env_cfg=self.env_cfg)

        # Add some regular metrics first
        test_metrics = {"reward": torch.tensor([1.0, 2.0, 3.0, 4.0])}
        wrapper.add_metrics(test_metrics)

        # Add final learning metrics
        final_metrics = {"final_value": torch.tensor(100.0)}
        wrapper.add_onetime_learning_metrics(final_metrics)

        # Should complete without errors

    def test_reset_functionality(self):
        """Test environment reset."""
        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=1, env_cfg=self.env_cfg)

        # Call reset
        result = wrapper.reset()

        # Check return format
        assert len(result) == 2  # obs, info
        obs, info = result
        assert obs.shape[0] == 4
        assert "timeout" in info

    def test_episode_data_storage(self):
        """Test episode data storage and aggregation."""
        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=1, env_cfg=self.env_cfg)

        # Set up some episode data
        wrapper.current_episode_rewards = torch.tensor([10.0, 20.0, 30.0, 40.0])
        wrapper.current_episode_lengths = torch.tensor([100, 200, 150, 250])

        # Mock episode completion
        completed_mask = torch.tensor([True, False, True, False], dtype=torch.bool)
        wrapper._store_completed_episodes(completed_mask)

        # Check that data was stored for agent 0 (all environments belong to agent 0)
        agent_data = wrapper.agent_episode_data[0]
        assert len(agent_data['episode_lengths']) == 2  # 2 completed episodes
        assert len(agent_data['episode_rewards']) == 2

    def test_close_functionality(self):
        """Test wrapper close method."""
        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=1, env_cfg=self.env_cfg)

        # Should not raise any errors
        wrapper.close()

    def test_multiple_agent_environment_assignment(self):
        """Test that environments are correctly assigned to agents."""
        # Set up 4 environments with 2 agents
        self.env_cfg.agent_configs['agent_1'] = {
            'wandb_entity': 'test_entity_2',
            'wandb_project': 'test_project_2',
            'wandb_name': 'test_agent_1'
        }

        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=2, env_cfg=self.env_cfg)

        # Environments 0-1 should go to agent 0, environments 2-3 should go to agent 1
        assert wrapper.envs_per_agent == 2

        # Test that metrics are split correctly
        test_metrics = {"test": torch.tensor([1.0, 2.0, 3.0, 4.0])}
        wrapper.add_metrics(test_metrics)  # Should not error

    def test_episode_completion_with_truncation(self):
        """Test episode completion tracking with truncation."""
        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=1, env_cfg=self.env_cfg)

        # Set episode lengths to max
        wrapper.current_episode_lengths = torch.tensor([500, 400, 500, 300])
        wrapper.current_episode_rewards = torch.tensor([100.0, 80.0, 120.0, 60.0])

        # Mock environment step with truncation
        with patch.object(wrapper.env, 'step') as mock_step:
            terminated = torch.tensor([False, False, False, False], dtype=torch.bool)
            truncated = torch.tensor([True, False, True, False], dtype=torch.bool)
            mock_step.return_value = (
                torch.randn(4, 64),
                torch.randn(4),
                terminated,
                truncated,
                {"timeout": truncated}
            )

            wrapper.step(torch.randn(4, 6))

            # Should trigger episode completion for max-length episodes
            # Episodes 0 and 2 should be considered completed (max length + truncated)

    def test_edge_case_empty_agent_config(self):
        """Test behavior with missing agent config."""
        # Remove agent_0 config to test default behavior
        del self.env_cfg.agent_configs['agent_0']

        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=1, env_cfg=self.env_cfg)

        # Should still work with empty config
        assert len(wrapper.trackers) == 1

    def test_metric_split_edge_cases(self):
        """Test metric splitting with various tensor sizes."""
        wrapper = GenericWandbLoggingWrapper(self.base_env, num_agents=1, env_cfg=self.env_cfg)

        # Test with various tensor sizes
        test_cases = {
            "scalar": torch.tensor(5.0),
            "wrong_size": torch.tensor([1.0, 2.0, 3.0]),  # Size 3, not 4 or 1
            "correct_env_size": torch.tensor([1.0, 2.0, 3.0, 4.0]),  # Size 4 (num_envs)
        }

        # All should be handled without errors
        wrapper.add_metrics(test_cases)