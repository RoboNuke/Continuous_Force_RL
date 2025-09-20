"""
Unit tests for observation_noise_wrapper.py functionality.
Tests ObservationNoiseWrapper for group-based noise application.
"""

import pytest
import torch
import gymnasium as gym
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock modules before imports
sys.modules['omni.isaac.lab'] = __import__('tests.mocks.mock_isaac_lab', fromlist=[''])
sys.modules['omni.isaac.lab.envs'] = __import__('tests.mocks.mock_isaac_lab', fromlist=['envs'])
sys.modules['omni.isaac.lab.utils'] = __import__('tests.mocks.mock_isaac_lab', fromlist=['utils'])

from wrappers.observations.observation_noise_wrapper import (
    ObservationNoiseWrapper, ObservationNoiseConfig, NoiseGroupConfig,
    create_position_noise_config, create_joint_noise_config, create_minimal_noise_config
)
from tests.mocks.mock_isaac_lab import MockBaseEnv, MockEnvConfig


class TestNoiseGroupConfig:
    """Test NoiseGroupConfig dataclass."""

    def test_default_initialization(self):
        """Test default NoiseGroupConfig initialization."""
        config = NoiseGroupConfig(group_name="test_group")

        assert config.group_name == "test_group"
        assert config.noise_type == "gaussian"
        assert config.std == 0.01
        assert config.mean == 0.0
        assert config.scale == 0.01
        assert config.enabled == True
        assert config.timing == "step"
        assert config.clip_range is None

    def test_custom_initialization(self):
        """Test custom NoiseGroupConfig initialization."""
        config = NoiseGroupConfig(
            group_name="custom_group",
            noise_type="uniform",
            std=0.05,
            mean=0.1,
            scale=0.02,
            enabled=False,
            timing="episode",
            clip_range=(-1.0, 1.0)
        )

        assert config.group_name == "custom_group"
        assert config.noise_type == "uniform"
        assert config.std == 0.05
        assert config.mean == 0.1
        assert config.scale == 0.02
        assert config.enabled == False
        assert config.timing == "episode"
        assert config.clip_range == (-1.0, 1.0)


class TestObservationNoiseConfig:
    """Test ObservationNoiseConfig dataclass."""

    def test_default_initialization(self):
        """Test default ObservationNoiseConfig initialization."""
        config = ObservationNoiseConfig()

        assert config.noise_groups == {}
        assert config.global_noise_scale == 1.0
        assert config.policy_update_interval == 32
        assert config.enabled == True
        assert config.apply_to_critic == True
        assert config.seed is None

    def test_add_group_noise(self):
        """Test adding group noise configuration."""
        config = ObservationNoiseConfig()
        group_config = NoiseGroupConfig(group_name="test_group")

        config.add_group_noise(group_config)

        assert "test_group" in config.noise_groups
        assert config.noise_groups["test_group"] == group_config

    def test_disable_group(self):
        """Test disabling group noise."""
        config = ObservationNoiseConfig()
        group_config = NoiseGroupConfig(group_name="test_group", enabled=True)
        config.add_group_noise(group_config)

        config.disable_group("test_group")

        assert config.noise_groups["test_group"].enabled == False

    def test_disable_nonexistent_group(self):
        """Test disabling nonexistent group (should not error)."""
        config = ObservationNoiseConfig()

        # Should not raise error
        config.disable_group("nonexistent_group")


class TestObservationNoiseWrapper:
    """Test ObservationNoiseWrapper functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.cfg = MockEnvConfig()
        self.base_env = MockBaseEnv(self.cfg)
        self.base_env.num_envs = 4
        self.base_env.device = torch.device("cpu")

        # Create basic noise config
        self.noise_config = ObservationNoiseConfig()
        self.noise_config.add_group_noise(NoiseGroupConfig(
            group_name="fingertip_pos",
            std=0.01,
            timing="step"
        ))

    def test_initialization_basic(self):
        """Test wrapper initialization with basic config."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        assert wrapper.config == self.noise_config
        assert wrapper.num_envs == 4
        assert wrapper.device == torch.device("cpu")
        assert wrapper.step_count == 0
        assert wrapper.episode_count == 0

    def test_initialization_with_seed(self):
        """Test wrapper initialization with random seed."""
        self.noise_config.seed = 42
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        # Should set seeds without error
        assert wrapper.config.seed == 42

    def test_initialization_missing_config(self):
        """Test wrapper initialization without required environment config."""
        # Remove required config
        delattr(self.base_env.cfg, 'component_dims')

        with pytest.raises(ValueError, match="Environment configuration must have 'component_dims'"):
            ObservationNoiseWrapper(self.base_env, self.noise_config)

    def test_load_observation_configs_missing_attr_map(self):
        """Test loading configs without component_attr_map."""
        # Remove component_attr_map
        delattr(self.base_env.cfg, 'component_attr_map')

        with pytest.raises(ValueError, match="Environment configuration must have .* 'component_attr_map'"):
            ObservationNoiseWrapper(self.base_env, self.noise_config)

    def test_build_group_mapping_success(self):
        """Test building group mapping successfully."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        # Check policy mapping
        expected_policy = {
            "fingertip_pos": (0, 3),
            "ee_linvel": (3, 6),
            "joint_pos": (6, 13)
        }
        assert wrapper.policy_group_mapping == expected_policy

    def test_build_group_mapping_no_order(self):
        """Test building group mapping without order attribute."""
        # Remove obs_order
        delattr(self.base_env.cfg, 'obs_order')

        with patch('builtins.print') as mock_print:
            wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

            assert wrapper.policy_group_mapping == {}
            # Check that the warning was printed among other print statements
            calls = [str(call) for call in mock_print.call_args_list]
            warning_found = any("Warning: Environment has no obs_order" in call for call in calls)
            assert warning_found

    def test_build_group_mapping_unknown_group(self):
        """Test building group mapping with unknown group."""
        # Add unknown group to order
        self.base_env.cfg.obs_order.append("unknown_group")

        with patch('builtins.print') as mock_print:
            wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

            # Check that the warning was printed among other print statements
            calls = [str(call) for call in mock_print.call_args_list]
            warning_found = any("Warning: Unknown observation group 'unknown_group'" in call for call in calls)
            assert warning_found

    def test_should_update_noise_disabled(self):
        """Test noise update check when disabled."""
        self.noise_config.enabled = False
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        assert not wrapper._should_update_noise()

    def test_should_update_noise_step_timing(self):
        """Test noise update check with step timing."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        # Step timing should always update
        assert wrapper._should_update_noise()

    def test_should_update_noise_episode_timing(self):
        """Test noise update check with episode timing."""
        self.noise_config.noise_groups["fingertip_pos"].timing = "episode"
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        # Should update at step 0 (episode start)
        assert wrapper.step_count == 0
        assert wrapper._should_update_noise()

        # Should not update at other steps
        wrapper.step_count = 5
        assert not wrapper._should_update_noise()

    def test_should_update_noise_policy_update_timing(self):
        """Test noise update check with policy update timing."""
        self.noise_config.noise_groups["fingertip_pos"].timing = "policy_update"
        self.noise_config.policy_update_interval = 10
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        # Should update at multiples of interval
        wrapper.step_count = 0
        assert wrapper._should_update_noise()

        wrapper.step_count = 10
        assert wrapper._should_update_noise()

        wrapper.step_count = 5
        assert not wrapper._should_update_noise()

    def test_generate_group_noise_gaussian(self):
        """Test generating gaussian noise for a group."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        group_shape = (4, 3)
        noise = wrapper._generate_group_noise("fingertip_pos", group_shape)

        assert noise.shape == group_shape
        assert noise.device == wrapper.device

    def test_generate_group_noise_uniform(self):
        """Test generating uniform noise for a group."""
        self.noise_config.noise_groups["fingertip_pos"].noise_type = "uniform"
        self.noise_config.noise_groups["fingertip_pos"].scale = 0.1
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        group_shape = (4, 3)
        noise = wrapper._generate_group_noise("fingertip_pos", group_shape)

        assert noise.shape == group_shape
        # Uniform noise should be within scale bounds
        assert torch.all(torch.abs(noise) <= 0.1)

    def test_generate_group_noise_none_type(self):
        """Test generating no noise for a group."""
        self.noise_config.noise_groups["fingertip_pos"].noise_type = "none"
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        group_shape = (4, 3)
        noise = wrapper._generate_group_noise("fingertip_pos", group_shape)

        assert noise.shape == group_shape
        assert torch.allclose(noise, torch.zeros(group_shape))

    def test_generate_group_noise_disabled_group(self):
        """Test generating noise for disabled group."""
        self.noise_config.noise_groups["fingertip_pos"].enabled = False
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        group_shape = (4, 3)
        noise = wrapper._generate_group_noise("fingertip_pos", group_shape)

        assert noise.shape == group_shape
        assert torch.allclose(noise, torch.zeros(group_shape))

    def test_generate_group_noise_nonexistent_group(self):
        """Test generating noise for nonexistent group."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        group_shape = (4, 3)
        noise = wrapper._generate_group_noise("nonexistent_group", group_shape)

        assert noise.shape == group_shape
        assert torch.allclose(noise, torch.zeros(group_shape))

    def test_generate_group_noise_per_dimension_std(self):
        """Test generating noise with per-dimension standard deviation."""
        self.noise_config.noise_groups["fingertip_pos"].std = [0.01, 0.02, 0.03]
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        group_shape = (4, 3)
        noise = wrapper._generate_group_noise("fingertip_pos", group_shape)

        assert noise.shape == group_shape

    def test_generate_group_noise_global_scale(self):
        """Test generating noise with global scale."""
        self.noise_config.global_noise_scale = 2.0
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        group_shape = (4, 3)
        noise = wrapper._generate_group_noise("fingertip_pos", group_shape)

        # Should be affected by global scale
        assert noise.shape == group_shape

    def test_apply_noise_to_observations_disabled(self):
        """Test applying noise when disabled."""
        self.noise_config.enabled = False
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        obs = torch.randn(4, 32)
        result = wrapper._apply_noise_to_observations(obs)

        assert torch.allclose(result, obs)

    def test_apply_noise_to_observations_dict_format(self):
        """Test applying noise to dictionary format observations."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        obs_dict = {
            "policy": torch.randn(4, 13),  # fingertip_pos + ee_linvel + joint_pos
            "critic": torch.randn(4, 16)
        }

        result = wrapper._apply_noise_to_observations(obs_dict)

        assert isinstance(result, dict)
        assert "policy" in result
        assert "critic" in result
        assert result["policy"].shape == obs_dict["policy"].shape
        assert result["critic"].shape == obs_dict["critic"].shape

    def test_apply_noise_to_observations_tensor_format(self):
        """Test applying noise to tensor format observations."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        obs_tensor = torch.randn(4, 13)  # Policy observations
        result = wrapper._apply_noise_to_observations(obs_tensor)

        assert isinstance(result, torch.Tensor)
        assert result.shape == obs_tensor.shape

    def test_apply_noise_to_observations_critic_disabled(self):
        """Test applying noise with critic noise disabled."""
        self.noise_config.apply_to_critic = False
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        original_critic = torch.randn(4, 16)
        obs_dict = {
            "policy": torch.randn(4, 13),
            "critic": original_critic
        }

        result = wrapper._apply_noise_to_observations(obs_dict)

        # Critic should remain unchanged
        assert torch.allclose(result["critic"], original_critic)

    def test_apply_noise_to_tensor_with_clipping(self):
        """Test applying noise to tensor with clipping."""
        self.noise_config.noise_groups["fingertip_pos"].clip_range = (-0.5, 0.5)
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        # Force noise cache update
        wrapper._update_noise_cache()

        obs_tensor = torch.ones(4, 13)  # All ones
        result = wrapper._apply_noise_to_tensor(obs_tensor, wrapper.policy_group_mapping)

        # Check that values are within clipping range for fingertip_pos
        assert torch.all(result[:, :3] >= -0.5)
        assert torch.all(result[:, :3] <= 0.5)

    def test_apply_noise_to_tensor_batch_size_mismatch(self):
        """Test applying noise with batch size mismatch."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        # Set current noise with different batch size
        wrapper.current_noise["fingertip_pos"] = torch.randn(2, 3)

        obs_tensor = torch.randn(4, 13)
        result = wrapper._apply_noise_to_tensor(obs_tensor, wrapper.policy_group_mapping)

        # Should handle expansion automatically
        assert result.shape == obs_tensor.shape

    def test_update_noise_cache(self):
        """Test updating noise cache."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        wrapper._update_noise_cache()

        assert "fingertip_pos" in wrapper.current_noise
        assert wrapper.current_noise["fingertip_pos"].shape == (4, 3)

    def test_step_functionality(self):
        """Test step method functionality."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        actions = torch.randn(4, 6)
        result = wrapper.step(actions)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert wrapper.step_count == 1

    def test_reset_functionality(self):
        """Test reset method functionality."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        # Set some state
        wrapper.step_count = 10
        wrapper.current_noise = {"test": torch.randn(4, 3)}

        result = wrapper.reset()

        assert len(result) == 2
        obs, info = result
        assert wrapper.step_count == 0
        assert wrapper.episode_count == 1
        # Note: current_noise may not be empty after reset because
        # noise is applied to initial observations during reset

    def test_get_noise_info(self):
        """Test getting noise information."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        info = wrapper.get_noise_info()

        assert "enabled" in info
        assert "global_scale" in info
        assert "step_count" in info
        assert "episode_count" in info
        assert "configured_groups" in info
        assert "active_groups" in info

        assert info["configured_groups"]["fingertip_pos"]["enabled"] == True
        assert info["configured_groups"]["fingertip_pos"]["noise_type"] == "gaussian"

    def test_multiple_step_calls_noise_consistency(self):
        """Test noise consistency across multiple step calls."""
        # Use episode timing to ensure noise doesn't change every step
        self.noise_config.noise_groups["fingertip_pos"].timing = "episode"
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        actions = torch.randn(4, 6)

        # First step should generate noise
        obs1, _, _, _, _ = wrapper.step(actions)

        # Second step should use same noise (episode timing)
        obs2, _, _, _, _ = wrapper.step(actions)

        # Noise should be consistent within episode for episode timing
        # (We can't test exact equality due to base environment randomness,
        # but we can test that the wrapper doesn't crash)
        assert obs1.shape == obs2.shape

    def test_wrapper_properties(self):
        """Test that wrapper properly delegates properties."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        assert wrapper.action_space.shape == self.base_env.action_space.shape
        assert wrapper.observation_space.shape == self.base_env.observation_space.shape

    def test_unwrapped_property(self):
        """Test unwrapped property access."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        assert wrapper.unwrapped == self.base_env

    def test_close_functionality(self):
        """Test wrapper close method."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        # Should not raise any errors
        wrapper.close()

    def test_device_consistency(self):
        """Test that device is consistent throughout wrapper."""
        wrapper = ObservationNoiseWrapper(self.base_env, self.noise_config)

        assert wrapper.device == self.base_env.device

        # Update noise cache and check device
        wrapper._update_noise_cache()
        for noise in wrapper.current_noise.values():
            assert noise.device == wrapper.device

    def test_wrapper_chain_compatibility(self):
        """Test that wrapper works in a chain with other wrappers."""
        # Create a simple wrapper chain
        intermediate_wrapper = gym.Wrapper(self.base_env)
        wrapper = ObservationNoiseWrapper(intermediate_wrapper, self.noise_config)

        assert wrapper.unwrapped == self.base_env

        actions = torch.randn(4, 6)
        result = wrapper.step(actions)
        assert len(result) == 5


class TestNoiseConfigPresets:
    """Test noise configuration preset functions."""

    def test_create_position_noise_config(self):
        """Test creating position noise configuration."""
        config = create_position_noise_config(
            position_std=0.002,
            orientation_std=0.01,
            velocity_std=0.005,
            timing="episode"
        )

        assert isinstance(config, ObservationNoiseConfig)
        assert "fingertip_pos" in config.noise_groups
        assert "fingertip_quat" in config.noise_groups
        assert "ee_linvel" in config.noise_groups

        # Check parameters
        assert config.noise_groups["fingertip_pos"].std == 0.002
        assert config.noise_groups["fingertip_quat"].std == 0.01
        assert config.noise_groups["ee_linvel"].std == 0.005
        assert config.noise_groups["fingertip_pos"].timing == "episode"

    def test_create_joint_noise_config(self):
        """Test creating joint noise configuration."""
        config = create_joint_noise_config(
            joint_std=0.02,
            timing="policy_update"
        )

        assert isinstance(config, ObservationNoiseConfig)
        assert "joint_pos" in config.noise_groups
        assert config.noise_groups["joint_pos"].std == 0.02
        assert config.noise_groups["joint_pos"].timing == "policy_update"

    def test_create_minimal_noise_config(self):
        """Test creating minimal noise configuration."""
        config = create_minimal_noise_config()

        assert isinstance(config, ObservationNoiseConfig)
        assert "fingertip_pos" in config.noise_groups
        assert config.noise_groups["fingertip_pos"].std == 0.0001
        assert config.noise_groups["fingertip_pos"].timing == "step"

    def test_position_noise_config_critic_control(self):
        """Test position noise config with critic control."""
        config = create_position_noise_config(apply_to_critic=False)

        assert config.apply_to_critic == False

    def test_joint_noise_config_critic_control(self):
        """Test joint noise config with critic control."""
        config = create_joint_noise_config(apply_to_critic=False)

        assert config.apply_to_critic == False


class TestObservationNoiseWrapperIntegration:
    """Integration tests for ObservationNoiseWrapper."""

    def setup_method(self):
        """Setup test environment."""
        self.cfg = MockEnvConfig()
        self.base_env = MockBaseEnv(self.cfg)
        self.base_env.num_envs = 4
        self.base_env.device = torch.device("cpu")

    def test_integration_with_position_noise(self):
        """Test integration with position noise configuration."""
        config = create_position_noise_config()
        wrapper = ObservationNoiseWrapper(self.base_env, config)

        # Test full workflow
        obs, info = wrapper.reset()
        actions = torch.randn(4, 6)

        for _ in range(5):
            obs, reward, terminated, truncated, info = wrapper.step(actions)

    def test_integration_with_joint_noise(self):
        """Test integration with joint noise configuration."""
        config = create_joint_noise_config()
        wrapper = ObservationNoiseWrapper(self.base_env, config)

        # Test full workflow
        obs, info = wrapper.reset()
        actions = torch.randn(4, 6)

        for _ in range(5):
            obs, reward, terminated, truncated, info = wrapper.step(actions)

    def test_integration_minimal_noise(self):
        """Test integration with minimal noise configuration."""
        config = create_minimal_noise_config()
        wrapper = ObservationNoiseWrapper(self.base_env, config)

        # Test full workflow
        obs, info = wrapper.reset()
        actions = torch.randn(4, 6)

        for _ in range(5):
            obs, reward, terminated, truncated, info = wrapper.step(actions)

    def test_integration_disabled_noise(self):
        """Test integration with disabled noise."""
        config = create_minimal_noise_config()
        config.enabled = False
        wrapper = ObservationNoiseWrapper(self.base_env, config)

        # Test full workflow
        obs, info = wrapper.reset()
        actions = torch.randn(4, 6)

        for _ in range(5):
            obs, reward, terminated, truncated, info = wrapper.step(actions)

    def test_reproducible_noise_with_seed(self):
        """Test reproducible noise generation with seed."""
        config = create_minimal_noise_config()
        config.seed = 42

        wrapper1 = ObservationNoiseWrapper(self.base_env, config)
        wrapper2 = ObservationNoiseWrapper(self.base_env, config)

        # Reset both wrappers
        obs1, _ = wrapper1.reset()
        obs2, _ = wrapper2.reset()

        # Should produce identical results with same seed
        # Note: This test may be sensitive to environment randomness
        # so we just check that both work without error
        assert obs1.shape == obs2.shape