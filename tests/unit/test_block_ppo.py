"""
Comprehensive unit tests for agents/block_ppo.py
Tests all methods with various parameter inputs and edge cases.
"""

import pytest
import torch
import torch.nn as nn
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch, mock_open
import gymnasium as gym

# Import the class we're testing
from agents.block_ppo import BlockPPO


class TestBlockPPOInit:
    """Test BlockPPO.__init__ method with various configurations."""

    def test_init_with_valid_env(self, mock_models, mock_memory, test_config, mock_env):
        """Test successful initialization with valid environment."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            state_size=64,
            track_ckpt_paths=True,
            task="Test-Task-v0",
            num_agents=2,
            num_envs=256,
            env=mock_env
        )

        assert agent.env == mock_env
        assert agent.num_agents == 2
        assert agent.num_envs == 256
        assert agent.envs_per_agent == 128
        assert agent.task_name == "Test-Task-v0"
        assert agent.state_size == 64
        assert agent.track_ckpt_paths == True

    def test_init_without_env_raises_error(self, mock_models, mock_memory, test_config):
        """Test that initialization without env parameter raises ValueError."""
        with pytest.raises(ValueError, match="env parameter is required"):
            BlockPPO(
                models=mock_models,
                memory=mock_memory,
                observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
                action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
                device=torch.device("cpu"),
                cfg=test_config,
                env=None
            )

    def test_init_with_invalid_env_wrapper(self, mock_models, mock_memory, test_config):
        """Test initialization with env that doesn't have required wrapper methods."""
        invalid_env = Mock()
        # Don't set up the required unwrapped.add_metrics
        with pytest.raises(ValueError, match="Environment must have GenericWandbLoggingWrapper"):
            BlockPPO(
                models=mock_models,
                memory=mock_memory,
                observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
                action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
                device=torch.device("cpu"),
                cfg=test_config,
                env=invalid_env
            )

    def test_init_with_auto_state_size(self, mock_models, mock_memory, test_config, mock_env):
        """Test initialization with auto-calculated state size."""
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(128,))
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=observation_space,
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            state_size=-1,  # Auto-calculate
            env=mock_env
        )

        assert agent.state_size == observation_space

    def test_init_creates_tracker_file(self, mock_models, mock_memory, test_config, mock_env, temp_dir):
        """Test that tracker file is created when track_ckpt_paths is True."""
        tracker_path = os.path.join(temp_dir, "test_tracker.txt")
        test_config['ckpt_tracker_path'] = tracker_path

        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            track_ckpt_paths=True,
            env=mock_env
        )

        assert os.path.exists(tracker_path)

    def test_init_with_different_num_agents(self, mock_models, mock_memory, test_config, mock_env):
        """Test initialization with different number of agents."""
        for num_agents in [1, 3, 5, 10]:
            agent = BlockPPO(
                models=mock_models,
                memory=mock_memory,
                observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
                action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
                device=torch.device("cpu"),
                cfg=test_config,
                num_agents=num_agents,
                num_envs=256,
                env=mock_env
            )

            assert agent.num_agents == num_agents
            assert agent.envs_per_agent == 256 // num_agents


class TestBlockPPOInit:
    """Test BlockPPO.init method."""

    @patch('agents.block_ppo.os.makedirs')
    def test_init_creates_checkpoint_directories(self, mock_makedirs, mock_models, mock_memory, test_config, mock_env):
        """Test that init creates checkpoint directories for all agents."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            num_agents=2,
            env=mock_env
        )

        # Mock required attributes for init
        agent.write_interval = "auto"
        agent.checkpoint_interval = "auto"
        agent.memory = mock_memory
        agent.state_size = 64
        agent.action_space = 6

        trainer_cfg = {"timesteps": 1000}
        agent.init(trainer_cfg)

        # Should create directories for both agents
        # Filter out PyTorch internal calls and count only our checkpoint directories
        our_calls = [call for call in mock_makedirs.call_args_list
                     if 'checkpoints' in str(call)]
        assert len(our_calls) == 2

    def test_init_with_memory_creates_tensors(self, mock_models, mock_memory, test_config, mock_env):
        """Test that init creates memory tensors when memory is provided."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            env=mock_env
        )

        # Mock required attributes
        agent.write_interval = 10
        agent.checkpoint_interval = 100
        agent.state_size = 64
        agent.action_space = 6

        agent.init()

        # Verify tensors were created
        expected_tensors = ["states", "actions", "rewards", "terminated", "truncated", "log_prob", "values", "returns", "advantages"]
        for tensor_name in expected_tensors:
            mock_memory.create_tensor.assert_any_call(name=tensor_name, size=agent.state_size if tensor_name == "states" else (agent.action_space if tensor_name == "actions" else 1), dtype=torch.float32 if tensor_name not in ["terminated", "truncated"] else torch.bool)

    def test_init_without_memory(self, mock_models, test_config, mock_env):
        """Test init method when memory is None."""
        agent = BlockPPO(
            models=mock_models,
            memory=None,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            env=mock_env
        )

        agent.write_interval = 10
        agent.checkpoint_interval = 100

        # Should not raise error
        agent.init()


class TestBlockPPOWriteCheckpoint:
    """Test BlockPPO.write_checkpoint method."""

    @patch('agents.block_ppo.export_policies')
    @patch('builtins.open', new_callable=mock_open)
    def test_write_checkpoint_exports_policies(self, mock_file, mock_export, mock_models, mock_memory, test_config, mock_env, temp_dir):
        """Test that write_checkpoint exports policies for all agents."""
        # Set up agent experiment configs
        for i in range(2):
            test_config[f'agent_{i}'] = {
                'experiment': {
                    'directory': temp_dir,
                    'experiment_name': 'test_exp'
                }
            }

        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            num_agents=2,
            env=mock_env
        )

        agent.agent_exp_cfgs = [test_config['agent_0'], test_config['agent_1']]
        agent.loggers = [Mock(wandb_cfg={"project": "test", "run_id": "123"}), Mock(wandb_cfg={"project": "test", "run_id": "456"})]

        agent.write_checkpoint(timestep=100, timesteps=1000)

        # Should call export_policies twice (policy and critic)
        assert mock_export.call_count == 2

    def test_write_checkpoint_tracks_paths(self, mock_models, mock_memory, test_config, mock_env, temp_dir):
        """Test that write_checkpoint tracks checkpoint paths when enabled."""
        tracker_path = os.path.join(temp_dir, "tracker.txt")
        test_config['ckpt_tracker_path'] = tracker_path

        # Set up agent experiment configs
        for i in range(2):
            test_config[f'agent_{i}'] = {
                'experiment': {
                    'directory': temp_dir,
                    'experiment_name': 'test_exp'
                }
            }

        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            num_agents=2,
            track_ckpt_paths=True,
            env=mock_env
        )

        agent.agent_exp_cfgs = [test_config['agent_0'], test_config['agent_1']]
        agent.loggers = [Mock(wandb_cfg={"project": "test", "run_id": "123"}), Mock(wandb_cfg={"project": "test", "run_id": "456"})]

        with patch('agents.block_ppo.export_policies'):
            agent.write_checkpoint(timestep=100, timesteps=1000)

        # Verify tracker file was written to
        assert os.path.exists(tracker_path)


class TestBlockPPOLoad:
    """Test BlockPPO.load method."""

    def test_load_policy_checkpoint(self, mock_models, mock_memory, test_config, mock_env, temp_dir):
        """Test loading policy checkpoint."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            env=mock_env
        )

        # Create mock checkpoint file
        checkpoint_path = os.path.join(temp_dir, "agent_0_1000.pt")
        mock_checkpoint = {
            'net_state_dict': {'weight': torch.randn(10, 10)},
            'log_std': torch.ones(6)
        }

        with patch('torch.load', return_value=mock_checkpoint):
            with patch('os.path.exists', return_value=False):  # No critic file
                agent.load(checkpoint_path)

        # Verify policy model load_state_dict was called
        mock_models['policy'].actor_mean.load_state_dict.assert_called_once()

    def test_load_with_critic_checkpoint(self, mock_models, mock_memory, test_config, mock_env, temp_dir):
        """Test loading with both policy and critic checkpoints."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            env=mock_env
        )

        checkpoint_path = os.path.join(temp_dir, "agent_0_1000.pt")
        critic_path = os.path.join(temp_dir, "critic_0_1000.pt")

        mock_policy_checkpoint = {'net_state_dict': {'weight': torch.randn(10, 10)}}
        mock_critic_checkpoint = {'net_state_dict': {'weight': torch.randn(5, 5)}}

        def mock_load(path, **kwargs):
            if 'critic' in path:
                return mock_critic_checkpoint
            return mock_policy_checkpoint

        with patch('torch.load', side_effect=mock_load):
            with patch('os.path.exists', return_value=True):
                agent.load(checkpoint_path)

        # Verify both models were loaded
        mock_models['policy'].actor_mean.load_state_dict.assert_called_once()
        mock_models['value'].critic.load_state_dict.assert_called_once()

    def test_load_fallback_format(self, mock_models, mock_memory, test_config, mock_env, temp_dir):
        """Test loading checkpoint in fallback format (no 'net_state_dict' key)."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            env=mock_env
        )

        checkpoint_path = os.path.join(temp_dir, "agent_0_1000.pt")
        # Checkpoint without 'net_state_dict' key (fallback format)
        mock_checkpoint = {'weight': torch.randn(10, 10), 'bias': torch.randn(10)}

        with patch('torch.load', return_value=mock_checkpoint):
            with patch('os.path.exists', return_value=False):
                agent.load(checkpoint_path)

        # Should still load the model using fallback
        mock_models['policy'].actor_mean.load_state_dict.assert_called_once_with(mock_checkpoint)


class TestBlockPPORecordTransition:
    """Test BlockPPO.record_transition method."""

    def test_record_transition_basic(self, mock_models, mock_memory, test_config, mock_env):
        """Test basic record_transition functionality."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            num_agents=2,
            num_envs=256,
            env=mock_env
        )

        # Set up required attributes
        agent._current_log_prob = torch.randn(256, 1)
        agent._rewards_shaper = None
        agent._time_limit_bootstrap = False
        agent._per_agent_state_preprocessors = [None, None]
        agent._per_agent_value_preprocessors = [None, None]
        agent.secondary_memories = []

        # Mock value model
        agent.value = Mock()
        agent.value.act = Mock(return_value=(torch.randn(256, 1), None, None))

        # Create test data
        states = torch.randn(256, 64)
        actions = torch.randn(256, 6)
        rewards = torch.randn(256)
        next_states = torch.randn(256, 64)
        terminated = torch.zeros(256, dtype=torch.bool)
        truncated = torch.zeros(256, dtype=torch.bool)
        infos = {}

        agent.record_transition(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=100,
            timesteps=1000
        )

        # Verify memory.add_samples was called
        mock_memory.add_samples.assert_called_once()

    def test_record_transition_with_reward_shaping(self, mock_models, mock_memory, test_config, mock_env):
        """Test record_transition with reward shaping."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            num_agents=2,
            num_envs=256,
            env=mock_env
        )

        # Set up required attributes
        agent._current_log_prob = torch.randn(256, 1)
        agent._rewards_shaper = Mock(return_value=torch.randn(256))
        agent._time_limit_bootstrap = False
        agent._per_agent_state_preprocessors = [None, None]
        agent._per_agent_value_preprocessors = [None, None]
        agent.secondary_memories = []

        agent.value = Mock()
        agent.value.act = Mock(return_value=(torch.randn(256, 1), None, None))

        # Create test data
        states = torch.randn(256, 64)
        actions = torch.randn(256, 6)
        rewards = torch.randn(256)
        next_states = torch.randn(256, 64)
        terminated = torch.zeros(256, dtype=torch.bool)
        truncated = torch.zeros(256, dtype=torch.bool)
        infos = {}

        agent.record_transition(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=100,
            timesteps=1000
        )

        # Verify reward shaper was called
        agent._rewards_shaper.assert_called_once()

    def test_record_transition_with_time_limit_bootstrap(self, mock_models, mock_memory, test_config, mock_env):
        """Test record_transition with time limit bootstrapping."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            num_agents=2,
            num_envs=256,
            env=mock_env
        )

        # Set up required attributes
        agent._current_log_prob = torch.randn(256, 1)
        agent._rewards_shaper = None
        agent._time_limit_bootstrap = True
        agent._discount_factor = 0.99
        agent._per_agent_state_preprocessors = [None, None]
        agent._per_agent_value_preprocessors = [None, None]
        agent.secondary_memories = []

        agent.value = Mock()
        agent.value.act = Mock(return_value=(torch.randn(256, 1), None, None))

        # Create test data with truncation
        states = torch.randn(256, 64)
        actions = torch.randn(256, 6)
        rewards = torch.randn(256)
        next_states = torch.randn(256, 64)
        terminated = torch.zeros(256, dtype=torch.bool)
        truncated = torch.ones(256, dtype=torch.bool)  # All truncated
        infos = {}

        agent.record_transition(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=100,
            timesteps=1000
        )

        # Should still call add_samples
        mock_memory.add_samples.assert_called_once()


class TestBlockPPOPostInteraction:
    """Test BlockPPO.post_interaction method."""

    def test_post_interaction_updates_on_rollout(self, mock_models, mock_memory, test_config, mock_env):
        """Test that post_interaction calls update when rollout condition is met."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            env=mock_env
        )

        # Set up required attributes
        agent._rollout = 15  # Will be 16 after increment
        agent._rollouts = 16
        agent._learning_starts = 0
        agent.checkpoint_interval = 0  # Disable checkpointing
        agent.write_interval = 0  # Disable writing
        agent.global_step = 0
        agent.num_envs = 256

        # Mock the _update method
        agent._update = Mock()
        agent.set_mode = Mock()

        agent.post_interaction(timestep=100, timesteps=1000)

        # Verify _update was called
        agent._update.assert_called_once_with(100, 1000)

    def test_post_interaction_writes_checkpoint(self, mock_models, mock_memory, test_config, mock_env):
        """Test that post_interaction writes checkpoint at intervals."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            env=mock_env
        )

        # Set up required attributes
        agent._rollout = 0
        agent._rollouts = 16
        agent._learning_starts = 10
        agent.checkpoint_interval = 10  # Every 10 timesteps
        agent.write_interval = 0
        agent.global_step = 0
        agent.num_envs = 256

        # Mock methods
        agent.write_checkpoint = Mock()
        agent.write_tracking_data = Mock()

        # Call at timestep that should trigger checkpoint
        agent.post_interaction(timestep=10, timesteps=1000)

        # Verify checkpoint was written
        agent.write_checkpoint.assert_called_once_with(10, 1000)

    def test_post_interaction_writes_tracking_data(self, mock_models, mock_memory, test_config, mock_env):
        """Test that post_interaction writes tracking data at intervals."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            env=mock_env
        )

        # Set up required attributes
        agent._rollout = 0
        agent._rollouts = 16
        agent._learning_starts = 10
        agent.checkpoint_interval = 0
        agent.write_interval = 5  # Every 5 timesteps
        agent.global_step = 0
        agent.num_envs = 256

        # Mock methods
        agent.write_checkpoint = Mock()
        agent.write_tracking_data = Mock()

        # Call at timestep that should trigger tracking
        agent.post_interaction(timestep=5, timesteps=1000)

        # Verify tracking was written
        agent.write_tracking_data.assert_called_once_with(5, 1000)


class TestBlockPPOUpdateNets:
    """Test BlockPPO.update_nets method."""

    def test_update_nets_basic(self, mock_models, mock_memory, test_config, mock_env, mock_optimizer, mock_scaler):
        """Test basic update_nets functionality."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            env=mock_env
        )

        # Set up required attributes
        agent.optimizer = mock_optimizer
        agent.scaler = mock_scaler
        agent._grad_norm_clip = 0.5
        agent.policy = mock_models['policy']
        agent.value = mock_models['value']

        # Mock loss
        loss = torch.tensor(0.5, requires_grad=True)

        with patch('agents.block_ppo.config') as mock_config:
            mock_config.torch.is_distributed = False
            agent.update_nets(loss)

        # Verify optimizer operations
        mock_optimizer.zero_grad.assert_called()
        mock_scaler.scale.assert_called_with(loss)
        mock_scaler.step.assert_called_with(mock_optimizer)
        mock_scaler.update.assert_called()

    def test_update_nets_with_grad_clipping(self, mock_models, mock_memory, test_config, mock_env, mock_optimizer, mock_scaler):
        """Test update_nets with gradient clipping."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            env=mock_env
        )

        # Set up required attributes
        agent.optimizer = mock_optimizer
        agent.scaler = mock_scaler
        agent._grad_norm_clip = 1.0  # Enable grad clipping
        agent.policy = mock_models['policy']
        agent.value = mock_models['value']

        loss = torch.tensor(0.5, requires_grad=True)

        with patch('agents.block_ppo.config') as mock_config:
            with patch('agents.block_ppo.nn.utils.clip_grad_norm_') as mock_clip:
                mock_config.torch.is_distributed = False
                agent.update_nets(loss)

                # Verify gradient clipping was called
                mock_scaler.unscale_.assert_called_with(mock_optimizer)
                mock_clip.assert_called()


class TestBlockPPOCalcValueLoss:
    """Test BlockPPO.calc_value_loss method."""

    def test_calc_value_loss_mse(self, mock_models, mock_memory, test_config, mock_env):
        """Test calc_value_loss with MSE loss."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            num_agents=2,
            num_envs=256,
            env=mock_env
        )

        # Set up required attributes
        agent.huber_value_loss = False
        agent._value_loss_scale = 1.0
        agent._clip_predicted_values = False
        agent._device_type = "cpu"
        agent._mixed_precision = False

        # Create test data
        sample_size = 64
        sampled_states = torch.randn(sample_size, 2, 128, 64)
        sampled_values = torch.randn(sample_size, 2, 128)
        sampled_returns = torch.randn(sample_size, 2, 128)
        keep_mask = torch.tensor([True, True])

        # Mock value model to return correct size
        batch_total_envs = sample_size * agent.num_agents * agent.envs_per_agent
        mock_models['value'].act = Mock(return_value=(torch.randn(batch_total_envs, 1), None, None))
        agent.value = mock_models['value']

        value_loss, value_losses, predicted_values = agent.calc_value_loss(
            sampled_states, sampled_values, sampled_returns, keep_mask, sample_size
        )

        # Verify types and shapes
        assert isinstance(value_loss, torch.Tensor)
        assert isinstance(value_losses, torch.Tensor)
        assert isinstance(predicted_values, torch.Tensor)

    def test_calc_value_loss_huber(self, mock_models, mock_memory, test_config, mock_env):
        """Test calc_value_loss with Huber loss."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            num_agents=2,
            num_envs=256,
            env=mock_env
        )

        # Set up required attributes
        agent.huber_value_loss = True
        agent._value_loss_scale = 1.0
        agent._clip_predicted_values = False
        agent._device_type = "cpu"
        agent._mixed_precision = False

        # Create test data
        sample_size = 64
        sampled_states = torch.randn(sample_size, 2, 128, 64)
        sampled_values = torch.randn(sample_size, 2, 128)
        sampled_returns = torch.randn(sample_size, 2, 128)
        keep_mask = torch.tensor([True, True])

        # Mock value model to return correct size
        batch_total_envs = sample_size * agent.num_agents * agent.envs_per_agent
        mock_models['value'].act = Mock(return_value=(torch.randn(batch_total_envs, 1), None, None))
        agent.value = mock_models['value']

        value_loss, value_losses, predicted_values = agent.calc_value_loss(
            sampled_states, sampled_values, sampled_returns, keep_mask, sample_size
        )

        # Verify types and shapes
        assert isinstance(value_loss, torch.Tensor)
        assert isinstance(value_losses, torch.Tensor)
        assert isinstance(predicted_values, torch.Tensor)


class TestBlockPPOPreprocessing:
    """Test BlockPPO preprocessing methods."""

    def test_setup_per_agent_preprocessors_none(self, mock_models, mock_memory, test_config, mock_env):
        """Test _setup_per_agent_preprocessors with no preprocessors."""
        test_config['state_preprocessor'] = None
        test_config['value_preprocessor'] = None

        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            num_agents=2,
            env=mock_env
        )

        agent._setup_per_agent_preprocessors()

        # Verify preprocessor lists are created with None values
        assert len(agent._per_agent_state_preprocessors) == 2
        assert len(agent._per_agent_value_preprocessors) == 2
        assert all(p is None for p in agent._per_agent_state_preprocessors)
        assert all(p is None for p in agent._per_agent_value_preprocessors)

    def test_apply_per_agent_preprocessing_none(self, mock_models, mock_memory, test_config, mock_env):
        """Test _apply_per_agent_preprocessing with no preprocessors."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            num_agents=2,
            num_envs=256,
            env=mock_env
        )

        agent._per_agent_state_preprocessors = [None, None]
        agent.envs_per_agent = 128

        # Test with 2D input
        input_tensor = torch.randn(256, 64)
        result = agent._apply_per_agent_preprocessing(input_tensor, agent._per_agent_state_preprocessors)

        # Should return unchanged tensor
        assert torch.equal(result, input_tensor)

    def test_apply_per_agent_preprocessing_3d_input(self, mock_models, mock_memory, test_config, mock_env):
        """Test _apply_per_agent_preprocessing with 3D input."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            num_agents=2,
            num_envs=256,
            env=mock_env
        )

        agent._per_agent_state_preprocessors = [None, None]
        agent.envs_per_agent = 128

        # Test with 3D input
        input_tensor = torch.randn(4, 256, 64)  # batch_size, total_envs, features
        result = agent._apply_per_agent_preprocessing(input_tensor, agent._per_agent_state_preprocessors)

        # Should return tensor with same shape
        assert result.shape == input_tensor.shape

    def test_validate_wrapper_integration_success(self, mock_models, mock_memory, test_config, mock_env):
        """Test _validate_wrapper_integration with valid wrapper."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            env=mock_env
        )

        result = agent._validate_wrapper_integration()
        assert result == True

    def test_get_logging_wrapper_success(self, mock_models, mock_memory, test_config, mock_env):
        """Test _get_logging_wrapper returns correct wrapper."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            env=mock_env
        )

        wrapper = agent._get_logging_wrapper()
        assert wrapper == mock_env.unwrapped

    def test_get_logging_wrapper_none(self, mock_models, mock_memory, test_config):
        """Test _get_logging_wrapper returns None for invalid env."""
        invalid_env = Mock()
        # Set up unwrapped but without add_metrics method
        invalid_env.unwrapped = Mock()
        # The Mock automatically has add_metrics as a mock attribute, so we need to delete it
        del invalid_env.unwrapped.add_metrics

        # This will fail during init due to validation, so we need to catch that
        with pytest.raises(ValueError):
            BlockPPO(
                models=mock_models,
                memory=mock_memory,
                observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
                action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
                device=torch.device("cpu"),
                cfg=test_config,
                env=invalid_env
            )


class TestBlockPPOLogging:
    """Test BlockPPO logging methods."""

    def test_log_minibatch_update_no_wrapper(self, mock_models, mock_memory, test_config, mock_env):
        """Test _log_minibatch_update with no logging wrapper."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            num_agents=2,
            env=mock_env
        )

        # Mock _get_logging_wrapper to return None
        agent._get_logging_wrapper = Mock(return_value=None)

        # Should not raise error
        agent._log_minibatch_update()

    def test_log_minibatch_update_with_data(self, mock_models, mock_memory, test_config, mock_env):
        """Test _log_minibatch_update with actual data."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            num_agents=2,
            env=mock_env
        )

        agent.envs_per_agent = 128
        agent._ratio_clip = 0.2

        # Create test data
        sample_size = 64
        returns = torch.randn(sample_size, 2, 128)
        values = torch.randn(sample_size, 2, 128)
        advantages = torch.randn(sample_size, 2, 128)
        old_log_probs = torch.randn(sample_size, 2, 128)
        new_log_probs = torch.randn(sample_size, 2, 128)

        # Should not raise error even with complex data
        agent._log_minibatch_update(
            returns=returns,
            values=values,
            advantages=advantages,
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs
        )

    def test_get_network_state(self, mock_models, mock_memory, test_config, mock_env, mock_optimizer):
        """Test _get_network_state method."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            num_agents=2,
            env=mock_env
        )

        agent.value = mock_models['value']
        agent.policy = mock_models['policy']
        agent.optimizer = mock_optimizer

        # Create mock parameters with gradients
        param = torch.randn(2, 10, 10, requires_grad=True)
        param.grad = torch.randn(2, 10, 10)

        # Mock named_parameters to return our test parameter
        mock_models['policy'].named_parameters = Mock(return_value=[('test_param', param)])
        mock_models['value'].named_parameters = Mock(return_value=[('test_param', param)])

        state = agent._get_network_state(agent_idx=0)

        # Verify structure
        assert 'policy' in state
        assert 'critic' in state
        assert 'gradients' in state['policy']
        assert 'weight_norms' in state['policy']


class TestBlockPPOEdgeCases:
    """Test edge cases and error conditions."""

    def test_adaptive_huber_delta(self, mock_models, mock_memory, test_config, mock_env):
        """Test adaptive_huber_delta method."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            env=mock_env
        )

        predicted = torch.randn(100)
        sampled = torch.randn(100)

        delta = agent.adaptive_huber_delta(predicted, sampled)

        assert isinstance(delta, float)
        assert delta > 0

    def test_add_sample_to_memory_with_secondary(self, mock_models, mock_memory, test_config, mock_env):
        """Test add_sample_to_memory with secondary memories."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            env=mock_env
        )

        # Add secondary memories
        secondary_mock1 = Mock()
        secondary_mock2 = Mock()
        agent.secondary_memories = [secondary_mock1, secondary_mock2]

        test_data = {
            'states': torch.randn(256, 64),
            'actions': torch.randn(256, 6)
        }

        agent.add_sample_to_memory(**test_data)

        # Verify primary memory was called
        mock_memory.add_samples.assert_called_once_with(**test_data)

        # Verify secondary memories were called
        secondary_mock1.add_samples.assert_called_once_with(**test_data)
        secondary_mock2.add_samples.assert_called_once_with(**test_data)

    def test_zero_agents_edge_case(self, mock_models, mock_memory, test_config, mock_env):
        """Test behavior with zero agents (edge case)."""
        with pytest.raises((ValueError, AssertionError, ZeroDivisionError)):
            BlockPPO(
                models=mock_models,
                memory=mock_memory,
                observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
                action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
                device=torch.device("cpu"),
                cfg=test_config,
                num_agents=0,  # Invalid
                env=mock_env
            )

    def test_mismatched_envs_agents(self, mock_models, mock_memory, test_config, mock_env):
        """Test behavior when num_envs doesn't divide evenly by num_agents."""
        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            num_agents=3,
            num_envs=256,  # 256 // 3 = 85.33... -> 85
            env=mock_env
        )

        # Should handle integer division
        assert agent.envs_per_agent == 85


# Test Configuration Variations
class TestBlockPPOConfigVariations:
    """Test BlockPPO with different configuration variations."""

    def test_with_track_ckpts_false(self, mock_models, mock_memory, test_config, mock_env):
        """Test with track_ckpts disabled."""
        test_config['track_ckpts'] = False

        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            track_ckpt_paths=False,
            env=mock_env
        )

        assert agent.track_ckpt_paths == False

    def test_with_huber_loss_disabled(self, mock_models, mock_memory, test_config, mock_env):
        """Test with huber value loss disabled."""
        test_config['use_huber_value_loss'] = False

        agent = BlockPPO(
            models=mock_models,
            memory=mock_memory,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
            action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
            device=torch.device("cpu"),
            cfg=test_config,
            env=mock_env
        )

        assert agent.huber_value_loss == False

    def test_with_different_value_update_ratios(self, mock_models, mock_memory, test_config, mock_env):
        """Test with different value update ratios."""
        for ratio in [1, 2, 3, 5]:
            test_config['value_update_ratio'] = ratio

            agent = BlockPPO(
                models=mock_models,
                memory=mock_memory,
                observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
                action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
                device=torch.device("cpu"),
                cfg=test_config,
                env=mock_env
            )

            assert agent.value_update_ratio == ratio

    def test_with_different_random_value_timesteps(self, mock_models, mock_memory, test_config, mock_env):
        """Test with different random value timesteps."""
        for timesteps in [0, 50, 100, 200]:
            test_config['random_value_timesteps'] = timesteps

            agent = BlockPPO(
                models=mock_models,
                memory=mock_memory,
                observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
                action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
                device=torch.device("cpu"),
                cfg=test_config,
                env=mock_env
            )

            assert agent._random_value_timesteps == timesteps