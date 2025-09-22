"""
Unit tests for observation_manager_wrapper.py functionality.
Tests ObservationManagerWrapper for Isaac Lab format conversion and SKRL compatibility.
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

from wrappers.observations.observation_manager_wrapper import ObservationManagerWrapper
from tests.mocks.mock_isaac_lab import MockBaseEnv, MockEnvConfig


class TestObservationManagerWrapper:
    """Test ObservationManagerWrapper functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.cfg = MockEnvConfig()
        self.base_env = MockBaseEnv(self.cfg)
        self.base_env.num_envs = 4
        self.base_env.device = torch.device("cpu")

    def test_initialization_basic(self):
        """Test wrapper initialization with default settings."""
        wrapper = ObservationManagerWrapper(self.base_env)

        assert wrapper.env == self.base_env
        assert wrapper.merge_strategy == "concatenate"
        assert wrapper.num_envs == 4
        assert wrapper.device == torch.device("cpu")
        assert wrapper._wrapper_initialized == True

    def test_initialization_with_merge_strategy(self):
        """Test wrapper initialization with different merge strategies."""
        strategies = ["concatenate", "policy_only", "critic_only", "average"]

        for strategy in strategies:
            wrapper = ObservationManagerWrapper(self.base_env, merge_strategy=strategy)
            assert wrapper.merge_strategy == strategy

    def test_initialization_without_robot(self):
        """Test wrapper initialization without robot attribute."""
        env_no_robot = MockBaseEnv(self.cfg)
        delattr(env_no_robot, '_robot')

        wrapper = ObservationManagerWrapper(env_no_robot)
        assert wrapper._wrapper_initialized == False

    def test_lazy_wrapper_initialization(self):
        """Test lazy wrapper initialization during step/reset."""
        env_no_robot = MockBaseEnv(self.cfg)
        delattr(env_no_robot, '_robot')

        wrapper = ObservationManagerWrapper(env_no_robot)
        assert not wrapper._wrapper_initialized

        # Add robot and call step
        env_no_robot._robot = True
        wrapper.step(torch.randn(4, 6))

        assert wrapper._wrapper_initialized

    def test_convert_to_single_tensor_concatenate(self):
        """Test conversion with concatenate strategy."""
        wrapper = ObservationManagerWrapper(self.base_env, merge_strategy="concatenate")

        # Test factory format conversion
        obs_dict = {
            "policy": torch.randn(4, 32),
            "critic": torch.randn(4, 48)
        }

        result = wrapper._convert_to_single_tensor(obs_dict)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 80)  # 32 + 48
        assert torch.allclose(result[:, :32], obs_dict["policy"])
        assert torch.allclose(result[:, 32:], obs_dict["critic"])

    def test_convert_to_single_tensor_policy_only(self):
        """Test conversion with policy_only strategy."""
        wrapper = ObservationManagerWrapper(self.base_env, merge_strategy="policy_only")

        obs_dict = {
            "policy": torch.randn(4, 32),
            "critic": torch.randn(4, 48)
        }

        result = wrapper._convert_to_single_tensor(obs_dict)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 32)
        assert torch.allclose(result, obs_dict["policy"])

    def test_convert_to_single_tensor_critic_only(self):
        """Test conversion with critic_only strategy."""
        wrapper = ObservationManagerWrapper(self.base_env, merge_strategy="critic_only")

        obs_dict = {
            "policy": torch.randn(4, 32),
            "critic": torch.randn(4, 48)
        }

        result = wrapper._convert_to_single_tensor(obs_dict)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 48)
        assert torch.allclose(result, obs_dict["critic"])

    def test_convert_to_single_tensor_average_same_shape(self):
        """Test conversion with average strategy and same shapes."""
        wrapper = ObservationManagerWrapper(self.base_env, merge_strategy="average")

        obs_dict = {
            "policy": torch.ones(4, 32),
            "critic": torch.ones(4, 32) * 2
        }

        result = wrapper._convert_to_single_tensor(obs_dict)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 32)
        # Should be average: (1 + 2) / 2 = 1.5
        assert torch.allclose(result, torch.ones(4, 32) * 1.5)

    def test_convert_to_single_tensor_average_different_shapes(self):
        """Test conversion with average strategy and different shapes."""
        wrapper = ObservationManagerWrapper(self.base_env, merge_strategy="average")

        obs_dict = {
            "policy": torch.randn(4, 32),
            "critic": torch.randn(4, 48)
        }

        with patch('builtins.print') as mock_print:
            result = wrapper._convert_to_single_tensor(obs_dict)

            # Should fallback to policy only and print warning
            assert isinstance(result, torch.Tensor)
            assert result.shape == (4, 32)
            assert torch.allclose(result, obs_dict["policy"])
            mock_print.assert_called_once()

    def test_convert_to_single_tensor_already_tensor(self):
        """Test conversion when input is already a tensor."""
        wrapper = ObservationManagerWrapper(self.base_env)

        obs_tensor = torch.randn(4, 64)
        result = wrapper._convert_to_single_tensor(obs_tensor)

        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, obs_tensor)

    def test_convert_to_single_tensor_unknown_strategy(self):
        """Test conversion with unknown merge strategy."""
        wrapper = ObservationManagerWrapper(self.base_env, merge_strategy="unknown")

        obs_dict = {
            "policy": torch.randn(4, 32),
            "critic": torch.randn(4, 48)
        }

        with patch('builtins.print') as mock_print:
            result = wrapper._convert_to_single_tensor(obs_dict)

            # Should fallback to policy only and print warning
            assert isinstance(result, torch.Tensor)
            assert result.shape == (4, 32)
            assert torch.allclose(result, obs_dict["policy"])
            mock_print.assert_called_once()

    def test_convert_to_single_tensor_unknown_format(self):
        """Test conversion with unknown observation format."""
        wrapper = ObservationManagerWrapper(self.base_env)

        with patch('builtins.print') as mock_print:
            result = wrapper._convert_to_single_tensor("unknown_format")

            # Should return zeros and print warning
            assert isinstance(result, torch.Tensor)
            assert result.shape == (4, 1)
            assert torch.allclose(result, torch.zeros(4, 1))
            mock_print.assert_called_once()

    def test_compose_single_tensor_from_dict_with_order(self):
        """Test composition from dict using obs_order."""
        wrapper = ObservationManagerWrapper(self.base_env)

        obs_dict = {
            "fingertip_pos": torch.randn(4, 3),
            "ee_linvel": torch.randn(4, 3),
            "joint_pos": torch.randn(4, 7)
        }

        result = wrapper._compose_single_tensor_from_dict(obs_dict)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 13)  # 3 + 3 + 7

    def test_compose_single_tensor_from_dict_no_order(self):
        """Test composition from dict without obs_order."""
        wrapper = ObservationManagerWrapper(self.base_env)
        # Remove obs_order from config
        delattr(wrapper.unwrapped.cfg, 'obs_order')

        obs_dict = {
            "component1": torch.randn(4, 5),
            "component2": torch.randn(4, 3)
        }

        result = wrapper._compose_single_tensor_from_dict(obs_dict)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 8)  # 5 + 3

    def test_compose_single_tensor_from_dict_empty(self):
        """Test composition from empty dict."""
        wrapper = ObservationManagerWrapper(self.base_env)

        result = wrapper._compose_single_tensor_from_dict({})
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 1)
        assert torch.allclose(result, torch.zeros(4, 1))

    def test_validate_observations_valid(self):
        """Test observation validation with valid tensor."""
        wrapper = ObservationManagerWrapper(self.base_env)

        obs = torch.randn(4, 64)
        # Should not raise any exceptions
        wrapper._validate_observations(obs)

    def test_validate_observations_invalid_type(self):
        """Test observation validation with invalid type."""
        wrapper = ObservationManagerWrapper(self.base_env)

        with pytest.raises(ValueError, match="Observations must be a torch.Tensor"):
            wrapper._validate_observations([1, 2, 3])

    def test_validate_observations_wrong_num_envs(self):
        """Test observation validation with wrong number of environments."""
        wrapper = ObservationManagerWrapper(self.base_env)

        obs = torch.randn(8, 64)  # Wrong first dimension
        with pytest.raises(ValueError, match="Observation first dimension must match num_envs"):
            wrapper._validate_observations(obs)

    def test_validate_observations_wrong_shape(self):
        """Test observation validation with wrong shape."""
        wrapper = ObservationManagerWrapper(self.base_env)

        obs = torch.randn(4, 64, 32)  # 3D instead of 2D
        with pytest.raises(ValueError, match="Observation must be 2D"):
            wrapper._validate_observations(obs)

    def test_validate_observations_nan_values(self):
        """Test observation validation with NaN values."""
        wrapper = ObservationManagerWrapper(self.base_env)

        obs = torch.randn(4, 64)
        obs[0, 0] = float('nan')

        with patch('builtins.print') as mock_print:
            wrapper._validate_observations(obs)
            mock_print.assert_called_with("Warning: NaN values detected in observations")

    def test_validate_observations_inf_values(self):
        """Test observation validation with infinite values."""
        wrapper = ObservationManagerWrapper(self.base_env)

        obs = torch.randn(4, 64)
        obs[0, 0] = float('inf')

        with patch('builtins.print') as mock_print:
            wrapper._validate_observations(obs)
            mock_print.assert_called_with("Warning: Inf values detected in observations")

    def test_wrapped_get_observations(self):
        """Test wrapped get observations method."""
        wrapper = ObservationManagerWrapper(self.base_env, merge_strategy="concatenate")

        # Mock original method to return factory format
        def mock_get_observations():
            return {
                "policy": torch.ones(4, 32),
                "critic": torch.ones(4, 48) * 2
            }

        wrapper._original_get_observations = mock_get_observations

        result = wrapper._wrapped_get_observations()
        # Should preserve Isaac Lab factory format for SKRL compatibility
        assert isinstance(result, dict)
        assert "policy" in result
        assert "critic" in result
        assert isinstance(result["policy"], torch.Tensor)
        assert isinstance(result["critic"], torch.Tensor)
        assert result["policy"].shape == (4, 32)
        assert result["critic"].shape == (4, 48)
        assert torch.allclose(result["policy"], torch.ones(4, 32))
        assert torch.allclose(result["critic"], torch.ones(4, 48) * 2)

    def test_wrapped_get_observations_no_original(self):
        """Test wrapped get observations without original method."""
        wrapper = ObservationManagerWrapper(self.base_env)
        wrapper._original_get_observations = None

        result = wrapper._wrapped_get_observations()
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 1)
        assert torch.allclose(result, torch.zeros(4, 1))

    def test_get_observation_info_success(self):
        """Test getting observation info successfully."""
        wrapper = ObservationManagerWrapper(self.base_env)

        # Mock wrapped method to return valid tensor
        def mock_wrapped_get_observations():
            return torch.randn(4, 64)

        wrapper._wrapped_get_observations = mock_wrapped_get_observations

        info = wrapper.get_observation_info()
        assert 'observation' in info
        assert 'shape' in info['observation']
        assert 'dtype' in info['observation']
        assert 'device' in info['observation']
        assert 'merge_strategy' in info
        assert info['observation']['shape'] == [4, 64]

    def test_get_observation_info_error(self):
        """Test getting observation info with error."""
        wrapper = ObservationManagerWrapper(self.base_env)

        # Mock wrapped method to raise exception
        def mock_wrapped_get_observations():
            raise RuntimeError("Test error")

        wrapper._wrapped_get_observations = mock_wrapped_get_observations

        info = wrapper.get_observation_info()
        assert 'error' in info
        assert "Failed to get observation info" in info['error']

    def test_get_observation_space_info(self):
        """Test getting observation space info."""
        wrapper = ObservationManagerWrapper(self.base_env)

        info = wrapper.get_observation_space_info()
        assert 'obs_order' in info
        assert 'state_order' in info
        assert 'observation_space' in info
        assert 'state_space' in info

    def test_validate_wrapper_stack_success(self):
        """Test wrapper stack validation success."""
        wrapper = ObservationManagerWrapper(self.base_env)

        issues = wrapper.validate_wrapper_stack()
        assert isinstance(issues, list)
        # Should be empty if no issues
        assert len(issues) == 0

    def test_validate_wrapper_stack_failure(self):
        """Test wrapper stack validation with failures."""
        wrapper = ObservationManagerWrapper(self.base_env)

        # Mock wrapped method to return invalid observations
        def mock_wrapped_get_observations():
            return "invalid_format"

        wrapper._wrapped_get_observations = mock_wrapped_get_observations

        issues = wrapper.validate_wrapper_stack()
        assert isinstance(issues, list)
        assert len(issues) > 0

    def test_step_functionality(self):
        """Test step method functionality."""
        wrapper = ObservationManagerWrapper(self.base_env)

        actions = torch.randn(4, 6)
        result = wrapper.step(actions)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape[0] == 4

    def test_step_with_lazy_initialization(self):
        """Test step method with lazy initialization."""
        env_no_robot = MockBaseEnv(self.cfg)
        delattr(env_no_robot, '_robot')

        wrapper = ObservationManagerWrapper(env_no_robot)
        assert not wrapper._wrapper_initialized

        # Add robot and call step
        env_no_robot._robot = True
        actions = torch.randn(4, 6)
        wrapper.step(actions)

        assert wrapper._wrapper_initialized

    def test_reset_functionality(self):
        """Test reset method functionality."""
        wrapper = ObservationManagerWrapper(self.base_env)

        result = wrapper.reset()
        assert len(result) == 2
        obs, info = result
        assert obs.shape[0] == 4

    def test_reset_with_lazy_initialization(self):
        """Test reset method with lazy initialization."""
        env_no_robot = MockBaseEnv(self.cfg)
        delattr(env_no_robot, '_robot')

        wrapper = ObservationManagerWrapper(env_no_robot)
        assert not wrapper._wrapper_initialized

        # Add robot and call reset
        env_no_robot._robot = True
        wrapper.reset()

        assert wrapper._wrapper_initialized

    def test_wrapper_properties(self):
        """Test that wrapper properly delegates properties."""
        wrapper = ObservationManagerWrapper(self.base_env)

        assert wrapper.action_space.shape == self.base_env.action_space.shape
        assert wrapper.observation_space.shape == self.base_env.observation_space.shape

    def test_unwrapped_property(self):
        """Test unwrapped property access."""
        wrapper = ObservationManagerWrapper(self.base_env)

        assert wrapper.unwrapped == self.base_env

    def test_close_functionality(self):
        """Test wrapper close method."""
        wrapper = ObservationManagerWrapper(self.base_env)

        # Should not raise any errors
        wrapper.close()

    def test_device_consistency(self):
        """Test that device is consistent throughout wrapper."""
        wrapper = ObservationManagerWrapper(self.base_env)

        assert wrapper.device == self.base_env.device

        # Test with GPU device if available
        if torch.cuda.is_available():
            self.base_env.device = torch.device("cuda:0")
            wrapper_gpu = ObservationManagerWrapper(self.base_env)
            assert wrapper_gpu.device == torch.device("cuda:0")

    def test_multiple_step_calls(self):
        """Test multiple consecutive step calls."""
        wrapper = ObservationManagerWrapper(self.base_env)

        actions = torch.randn(4, 6)

        # Call step multiple times
        for i in range(5):
            result = wrapper.step(actions)
            assert len(result) == 5

    def test_wrapper_chain_compatibility(self):
        """Test that wrapper works in a chain with other wrappers."""
        # Create a simple wrapper chain
        intermediate_wrapper = gym.Wrapper(self.base_env)
        wrapper = ObservationManagerWrapper(intermediate_wrapper)

        assert wrapper.unwrapped == self.base_env

        actions = torch.randn(4, 6)
        result = wrapper.step(actions)
        assert len(result) == 5

    def test_different_merge_strategies_integration(self):
        """Test different merge strategies in actual step/reset."""
        strategies = ["concatenate", "policy_only", "critic_only", "average"]

        for strategy in strategies:
            wrapper = ObservationManagerWrapper(self.base_env, merge_strategy=strategy)

            # Test reset
            obs, info = wrapper.reset()
            assert isinstance(obs, torch.Tensor)
            assert obs.shape[0] == 4

            # Test step
            actions = torch.randn(4, 6)
            obs, reward, terminated, truncated, info = wrapper.step(actions)
            assert isinstance(obs, torch.Tensor)
            assert obs.shape[0] == 4