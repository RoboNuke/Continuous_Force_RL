"""
Unit tests for ObservationManagerWrapper.

Tests the observation management including format standardization,
noise injection, observation composition, and validation.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tests.mocks.mock_isaac_lab import MockEnvironment, MockConfig
from wrappers.observations.observation_manager_wrapper import ObservationManagerWrapper


class TestObservationManagerWrapper:
    """Test suite for ObservationManagerWrapper."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""
        env = MockEnvironment(num_envs=4, device='cpu')

        # Add configuration for observation manager
        env.cfg.obs_order = ["fingertip_pos", "force_torque", "ee_linvel"]
        env.cfg.state_order = ["fingertip_pos", "force_torque", "ee_linvel", "fingertip_quat"]
        env.cfg.observation_space = 12  # 3 + 6 + 3
        env.cfg.state_space = 16  # 3 + 6 + 3 + 4

        # Add noise configuration
        env.cfg.obs_noise_mean = {
            "force_torque": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "policy": [0.0] * 12
        }
        env.cfg.obs_noise_std = {
            "force_torque": [0.1, 0.1, 0.1, 0.01, 0.01, 0.01],
            "policy": [0.05] * 12
        }

        # Add robot for initialization
        env._robot = Mock()

        return env

    @pytest.fixture
    def wrapper_no_noise(self, mock_env):
        """Create wrapper without noise."""
        return ObservationManagerWrapper(mock_env, use_obs_noise=False)

    @pytest.fixture
    def wrapper_with_noise(self, mock_env):
        """Create wrapper with noise enabled."""
        return ObservationManagerWrapper(mock_env, use_obs_noise=True)

    def test_initialization_no_noise(self, wrapper_no_noise):
        """Test wrapper initialization without noise."""
        wrapper = wrapper_no_noise

        assert wrapper.use_obs_noise == False
        assert wrapper.num_envs == 4
        assert str(wrapper.device) == 'cpu'
        assert wrapper._wrapper_initialized == True
        assert len(wrapper.obs_noise_mean) > 0
        assert len(wrapper.obs_noise_std) > 0

    def test_initialization_with_noise(self, wrapper_with_noise):
        """Test wrapper initialization with noise enabled."""
        wrapper = wrapper_with_noise

        assert wrapper.use_obs_noise == True
        assert wrapper.num_envs == 4
        assert str(wrapper.device) == 'cpu'
        assert wrapper._wrapper_initialized == True

    def test_initialization_without_robot(self, mock_env):
        """Test initialization when robot is not available initially."""
        delattr(mock_env, '_robot')
        wrapper = ObservationManagerWrapper(mock_env)
        assert not wrapper._wrapper_initialized

    def test_initialization_without_noise_config(self, mock_env):
        """Test initialization without noise configuration."""
        delattr(mock_env.cfg, 'obs_noise_mean')
        delattr(mock_env.cfg, 'obs_noise_std')

        wrapper = ObservationManagerWrapper(mock_env, use_obs_noise=True)

        assert wrapper.obs_noise_mean == {}
        assert wrapper.obs_noise_std == {}

    def test_wrapper_initialization(self, wrapper_no_noise):
        """Test wrapper method override initialization."""
        wrapper = wrapper_no_noise

        # Should be initialized since robot exists
        assert wrapper._wrapper_initialized == True

        # Check that method is stored and overridden (if it exists)
        if hasattr(wrapper.unwrapped, '_get_observations'):
            assert wrapper._original_get_observations is not None

    def test_wrapped_get_observations_standard_format(self, wrapper_no_noise):
        """Test wrapped get observations with standard format input."""
        wrapper = wrapper_no_noise

        # Mock original get_observations to return standard format
        standard_obs = {
            "policy": torch.randn(4, 12, device='cpu'),
            "critic": torch.randn(4, 16, device='cpu')
        }
        wrapper._original_get_observations = Mock(return_value=standard_obs)

        result = wrapper._wrapped_get_observations()

        # Should return the same observations
        assert 'policy' in result
        assert 'critic' in result
        torch.testing.assert_close(result['policy'], standard_obs['policy'])
        torch.testing.assert_close(result['critic'], standard_obs['critic'])

    def test_wrapped_get_observations_tensor_input(self, wrapper_no_noise):
        """Test wrapped get observations with tensor input."""
        wrapper = wrapper_no_noise

        # Mock original get_observations to return tensor
        tensor_obs = torch.randn(4, 10, device='cpu')
        wrapper._original_get_observations = Mock(return_value=tensor_obs)

        result = wrapper._wrapped_get_observations()

        # Should convert to standard format
        assert 'policy' in result
        assert 'critic' in result
        torch.testing.assert_close(result['policy'], tensor_obs)
        torch.testing.assert_close(result['critic'], tensor_obs)

    def test_wrapped_get_observations_dict_input(self, wrapper_no_noise):
        """Test wrapped get observations with dictionary input."""
        wrapper = wrapper_no_noise

        # Mock original get_observations to return dict
        dict_obs = {
            "fingertip_pos": torch.randn(4, 3, device='cpu'),
            "force_torque": torch.randn(4, 6, device='cpu'),
            "ee_linvel": torch.randn(4, 3, device='cpu'),
            "fingertip_quat": torch.randn(4, 4, device='cpu')
        }
        wrapper._original_get_observations = Mock(return_value=dict_obs)

        result = wrapper._wrapped_get_observations()

        # Should compose observations according to order
        assert 'policy' in result
        assert 'critic' in result
        # Policy: fingertip_pos + force_torque + ee_linvel = 3 + 6 + 3 = 12
        assert result['policy'].shape == (4, 12)
        # Critic: fingertip_pos + force_torque + ee_linvel + fingertip_quat = 3 + 6 + 3 + 4 = 16
        assert result['critic'].shape == (4, 16)

    def test_wrapped_get_observations_no_original(self, wrapper_no_noise):
        """Test wrapped get observations without original method."""
        wrapper = wrapper_no_noise
        wrapper._original_get_observations = None

        result = wrapper._wrapped_get_observations()

        # Should return default empty observations
        assert 'policy' in result
        assert 'critic' in result
        assert result['policy'].shape == (4, 1)
        assert result['critic'].shape == (4, 1)

    def test_wrapped_get_observations_with_noise(self, wrapper_with_noise):
        """Test wrapped get observations with noise applied."""
        wrapper = wrapper_with_noise

        # Mock original get_observations
        standard_obs = {
            "policy": torch.zeros(4, 12, device='cpu'),
            "critic": torch.zeros(4, 16, device='cpu')
        }
        wrapper._original_get_observations = Mock(return_value=standard_obs)

        result = wrapper._wrapped_get_observations()

        # Should have noise applied (non-zero values)
        assert 'policy' in result
        assert 'critic' in result
        # Due to noise, should not be exactly equal to original zeros
        assert not torch.allclose(result['policy'], standard_obs['policy'], atol=1e-6)

    def test_convert_to_standard_format_dict(self, wrapper_no_noise):
        """Test conversion from dictionary to standard format."""
        wrapper = wrapper_no_noise

        obs_dict = {
            "fingertip_pos": torch.randn(4, 3, device='cpu'),
            "force_torque": torch.randn(4, 6, device='cpu'),
            "ee_linvel": torch.randn(4, 3, device='cpu'),
            "fingertip_quat": torch.randn(4, 4, device='cpu')
        }

        result = wrapper._convert_to_standard_format(obs_dict)

        assert 'policy' in result
        assert 'critic' in result
        assert result['policy'].shape == (4, 12)  # 3 + 6 + 3
        assert result['critic'].shape == (4, 16)  # 3 + 6 + 3 + 4

    def test_convert_to_standard_format_tensor(self, wrapper_no_noise):
        """Test conversion from tensor to standard format."""
        wrapper = wrapper_no_noise

        tensor_obs = torch.randn(4, 10, device='cpu')
        result = wrapper._convert_to_standard_format(tensor_obs)

        assert 'policy' in result
        assert 'critic' in result
        torch.testing.assert_close(result['policy'], tensor_obs)
        torch.testing.assert_close(result['critic'], tensor_obs)

    def test_convert_to_standard_format_already_standard(self, wrapper_no_noise):
        """Test conversion when already in standard format."""
        wrapper = wrapper_no_noise

        standard_obs = {
            "policy": torch.randn(4, 12, device='cpu'),
            "critic": torch.randn(4, 16, device='cpu')
        }

        result = wrapper._convert_to_standard_format(standard_obs)

        # Should return unchanged
        assert result is standard_obs

    def test_convert_to_standard_format_unknown(self, wrapper_no_noise):
        """Test conversion from unknown format."""
        wrapper = wrapper_no_noise

        unknown_obs = "invalid"
        result = wrapper._convert_to_standard_format(unknown_obs)

        assert 'policy' in result
        assert 'critic' in result
        assert result['policy'].shape == (4, 1)
        assert result['critic'].shape == (4, 1)
        assert torch.all(result['policy'] == 0)
        assert torch.all(result['critic'] == 0)

    def test_compose_observations_from_dict(self, wrapper_no_noise):
        """Test observation composition from dictionary."""
        wrapper = wrapper_no_noise

        obs_dict = {
            "fingertip_pos": torch.tensor([[1.0, 2.0, 3.0]] * 4, device='cpu'),
            "force_torque": torch.tensor([[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]] * 4, device='cpu'),
            "ee_linvel": torch.tensor([[10.0, 11.0, 12.0]] * 4, device='cpu'),
            "fingertip_quat": torch.tensor([[13.0, 14.0, 15.0, 16.0]] * 4, device='cpu'),
            "extra": torch.tensor([[17.0, 18.0]] * 4, device='cpu')  # Not in order
        }

        result = wrapper._compose_observations_from_dict(obs_dict)

        # Policy should follow obs_order: fingertip_pos + force_torque + ee_linvel
        expected_policy = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]] * 4, device='cpu')
        torch.testing.assert_close(result['policy'], expected_policy)

        # Critic should follow state_order: fingertip_pos + force_torque + ee_linvel + fingertip_quat
        expected_critic = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]] * 4, device='cpu')
        torch.testing.assert_close(result['critic'], expected_critic)

    def test_compose_observations_no_order_config(self, mock_env):
        """Test observation composition without order configuration."""
        # Remove order configuration
        delattr(mock_env.cfg, 'obs_order')
        delattr(mock_env.cfg, 'state_order')

        wrapper = ObservationManagerWrapper(mock_env)

        obs_dict = {
            "tensor1": torch.tensor([[1.0, 2.0]] * 4, device='cpu'),
            "tensor2": torch.tensor([[3.0, 4.0, 5.0]] * 4, device='cpu'),
            "non_tensor": "invalid"
        }

        result = wrapper._compose_observations_from_dict(obs_dict)

        # Should use all tensors
        assert result['policy'].shape[1] == 5  # 2 + 3
        assert result['critic'].shape[1] == 5  # 2 + 3

    def test_compose_observations_empty_dict(self, wrapper_no_noise):
        """Test observation composition with empty dictionary."""
        wrapper = wrapper_no_noise

        result = wrapper._compose_observations_from_dict({})

        assert result['policy'].shape == (4, 1)
        assert result['critic'].shape == (4, 1)
        assert torch.all(result['policy'] == 0)
        assert torch.all(result['critic'] == 0)

    def test_apply_observation_noise_disabled(self, wrapper_no_noise):
        """Test noise application when disabled."""
        wrapper = wrapper_no_noise

        obs = {
            "policy": torch.ones(4, 12, device='cpu'),
            "critic": torch.ones(4, 16, device='cpu')
        }

        result = wrapper._apply_observation_noise(obs)

        # Should return unchanged
        torch.testing.assert_close(result['policy'], obs['policy'])
        torch.testing.assert_close(result['critic'], obs['critic'])

    def test_apply_observation_noise_enabled(self, wrapper_with_noise):
        """Test noise application when enabled."""
        wrapper = wrapper_with_noise

        # Override noise config to match tensor dimensions
        wrapper.obs_noise_mean = {"policy": [0.0] * 12, "critic": [0.0] * 16}
        wrapper.obs_noise_std = {"policy": [0.1] * 12, "critic": [0.05] * 16}

        obs = {
            "policy": torch.zeros(4, 12, device='cpu'),
            "critic": torch.zeros(4, 16, device='cpu')
        }

        result = wrapper._apply_observation_noise(obs)

        # Should have noise applied (non-zero values)
        assert not torch.allclose(result['policy'], obs['policy'], atol=1e-6)
        assert not torch.allclose(result['critic'], obs['critic'], atol=1e-6)

    def test_apply_observation_noise_zero_std(self, wrapper_with_noise):
        """Test noise application with zero standard deviation."""
        wrapper = wrapper_with_noise

        # Override noise config with zero std
        wrapper.obs_noise_std = {"policy": [0.0] * 12, "critic": [0.0] * 16}

        obs = {
            "policy": torch.ones(4, 12, device='cpu'),
            "critic": torch.ones(4, 16, device='cpu')
        }

        result = wrapper._apply_observation_noise(obs)

        # Should return unchanged due to zero std
        torch.testing.assert_close(result['policy'], obs['policy'])
        torch.testing.assert_close(result['critic'], obs['critic'])

    def test_get_noise_config_direct_match(self, wrapper_with_noise):
        """Test noise configuration retrieval with direct key match."""
        wrapper = wrapper_with_noise

        # Test direct match
        mean_config = wrapper._get_noise_config('force_torque', 'mean', 6)
        std_config = wrapper._get_noise_config('force_torque', 'std', 6)

        assert mean_config == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert std_config == [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]

    def test_get_noise_config_partial_match(self, wrapper_with_noise):
        """Test noise configuration retrieval with partial key match."""
        wrapper = wrapper_with_noise

        # Test partial match (policy should match for any key containing 'policy')
        mean_config = wrapper._get_noise_config('policy_obs', 'mean', 12)
        std_config = wrapper._get_noise_config('obs_policy', 'std', 12)

        assert len(mean_config) == 12
        assert len(std_config) == 12

    def test_get_noise_config_no_match(self, wrapper_with_noise):
        """Test noise configuration retrieval with no match."""
        wrapper = wrapper_with_noise

        # Test no match - should return zeros
        mean_config = wrapper._get_noise_config('unknown', 'mean', 8)
        std_config = wrapper._get_noise_config('unknown', 'std', 8)

        assert mean_config == [0.0] * 8
        assert std_config == [0.0] * 8

    def test_validate_observations_valid(self, wrapper_no_noise):
        """Test observation validation with valid observations."""
        wrapper = wrapper_no_noise

        valid_obs = {
            "policy": torch.randn(4, 12, device='cpu'),
            "critic": torch.randn(4, 16, device='cpu')
        }

        # Should not raise any exception
        wrapper._validate_observations(valid_obs)

    def test_validate_observations_not_dict(self, wrapper_no_noise):
        """Test observation validation with non-dictionary input."""
        wrapper = wrapper_no_noise

        with pytest.raises(ValueError, match="Observations must be in dictionary format"):
            wrapper._validate_observations(torch.randn(4, 10))

    def test_validate_observations_missing_keys(self, wrapper_no_noise):
        """Test observation validation with missing keys."""
        wrapper = wrapper_no_noise

        invalid_obs = {"policy": torch.randn(4, 12, device='cpu')}

        with pytest.raises(ValueError, match="Observations must contain 'policy' and 'critic' keys"):
            wrapper._validate_observations(invalid_obs)

    def test_validate_observations_non_tensor(self, wrapper_no_noise):
        """Test observation validation with non-tensor values."""
        wrapper = wrapper_no_noise

        invalid_obs = {
            "policy": torch.randn(4, 12, device='cpu'),
            "critic": "not_a_tensor"
        }

        with pytest.raises(ValueError, match="Observation 'critic' must be a torch.Tensor"):
            wrapper._validate_observations(invalid_obs)

    def test_validate_observations_wrong_shape(self, wrapper_no_noise):
        """Test observation validation with wrong tensor shapes."""
        wrapper = wrapper_no_noise

        # Wrong first dimension
        invalid_obs = {
            "policy": torch.randn(3, 12, device='cpu'),  # Should be 4
            "critic": torch.randn(4, 16, device='cpu')
        }

        with pytest.raises(ValueError, match="Observation 'policy' first dimension must match num_envs"):
            wrapper._validate_observations(invalid_obs)

        # Wrong number of dimensions
        invalid_obs = {
            "policy": torch.randn(4, 12, 3, device='cpu'),  # Should be 2D
            "critic": torch.randn(4, 16, device='cpu')
        }

        with pytest.raises(ValueError, match="Observation 'policy' must be 2D"):
            wrapper._validate_observations(invalid_obs)

    def test_validate_observations_nan_inf_warnings(self, wrapper_no_noise, capsys):
        """Test observation validation with NaN and Inf values."""
        wrapper = wrapper_no_noise

        obs_with_nan = {
            "policy": torch.tensor([[float('nan'), 1.0]] * 4, device='cpu'),
            "critic": torch.tensor([[float('inf'), 1.0]] * 4, device='cpu')
        }

        wrapper._validate_observations(obs_with_nan)

        # Check that warnings were printed
        captured = capsys.readouterr()
        assert "Warning: NaN values detected in observation 'policy'" in captured.out
        assert "Warning: Inf values detected in observation 'critic'" in captured.out

    def test_get_observation_info(self, wrapper_no_noise):
        """Test getting observation information."""
        wrapper = wrapper_no_noise

        # Mock get_observations
        mock_obs = {
            "policy": torch.tensor([[1.0, 2.0, 3.0]] * 4, device='cpu'),
            "critic": torch.tensor([[4.0, 5.0]] * 4, device='cpu')
        }
        wrapper.unwrapped._get_observations = Mock(return_value=mock_obs)

        info = wrapper.get_observation_info()

        assert 'policy' in info
        assert 'critic' in info
        assert info['policy']['shape'] == [4, 3]
        assert info['critic']['shape'] == [4, 2]
        assert 'min' in info['policy']
        assert 'max' in info['policy']
        assert 'mean' in info['policy']
        assert 'std' in info['policy']

    def test_get_observation_info_non_dict(self, wrapper_no_noise):
        """Test getting observation information with non-dictionary observations."""
        wrapper = wrapper_no_noise

        # Mock get_observations to return tensor
        wrapper.unwrapped._get_observations = Mock(return_value=torch.randn(4, 10))

        info = wrapper.get_observation_info()

        assert 'error' in info
        assert "not in expected dictionary format" in info['error']

    def test_get_observation_info_exception(self, wrapper_no_noise):
        """Test getting observation information when exception occurs."""
        wrapper = wrapper_no_noise

        # Mock get_observations to raise exception
        wrapper.unwrapped._get_observations = Mock(side_effect=Exception("Test error"))

        info = wrapper.get_observation_info()

        assert 'error' in info
        assert "Failed to get observation info" in info['error']

    def test_get_observation_space_info(self, wrapper_no_noise):
        """Test getting observation space configuration info."""
        wrapper = wrapper_no_noise

        info = wrapper.get_observation_space_info()

        assert 'obs_order' in info
        assert 'state_order' in info
        assert 'observation_space' in info
        assert 'state_space' in info
        assert info['obs_order'] == ["fingertip_pos", "force_torque", "ee_linvel"]
        assert info['observation_space'] == 12
        assert info['state_space'] == 16

    def test_get_observation_space_info_missing_config(self, mock_env):
        """Test getting observation space info with missing configuration."""
        # Remove some config attributes
        delattr(mock_env.cfg, 'obs_order')
        delattr(mock_env.cfg, 'observation_space')

        wrapper = ObservationManagerWrapper(mock_env)
        info = wrapper.get_observation_space_info()

        assert 'obs_order' not in info
        assert 'observation_space' not in info
        assert 'state_order' in info  # Should still have this
        assert 'state_space' in info   # Should still have this

    def test_validate_wrapper_stack_valid(self, wrapper_no_noise):
        """Test wrapper stack validation with valid configuration."""
        wrapper = wrapper_no_noise

        # Mock get_observations to return valid format
        wrapper.unwrapped._get_observations = Mock(return_value={
            "policy": torch.randn(4, 12),
            "critic": torch.randn(4, 16)
        })

        with patch('builtins.__import__') as mock_import:
            mock_module = Mock()
            mock_module.OBS_DIM_CFG = {"fingertip_pos": 3, "force_torque": 6, "ee_linvel": 3}

            def side_effect(name, *args, **kwargs):
                if name == 'envs.factory.factory_env_cfg':
                    return mock_module
                return __import__.__wrapped__(name, *args, **kwargs)

            mock_import.side_effect = side_effect

            issues = wrapper.validate_wrapper_stack()

        assert len(issues) == 0

    def test_validate_wrapper_stack_missing_keys(self, wrapper_no_noise):
        """Test wrapper stack validation with missing observation keys."""
        wrapper = wrapper_no_noise

        # Mock get_observations to return missing keys
        wrapper.unwrapped._get_observations = Mock(return_value={
            "policy": torch.randn(4, 12)
            # Missing "critic"
        })

        issues = wrapper.validate_wrapper_stack()

        assert len(issues) > 0
        assert any("Missing 'policy' or 'critic'" in issue for issue in issues)

    def test_validate_wrapper_stack_non_dict_obs(self, wrapper_no_noise):
        """Test wrapper stack validation with non-dictionary observations."""
        wrapper = wrapper_no_noise

        # Mock get_observations to return tensor
        wrapper.unwrapped._get_observations = Mock(return_value=torch.randn(4, 10))

        issues = wrapper.validate_wrapper_stack()

        assert len(issues) > 0
        assert any("not in dictionary format" in issue for issue in issues)

    def test_validate_wrapper_stack_missing_config(self, mock_env):
        """Test wrapper stack validation with missing configuration."""
        # Remove config attributes
        delattr(mock_env.cfg, 'obs_order')
        delattr(mock_env.cfg, 'state_order')

        wrapper = ObservationManagerWrapper(mock_env)
        issues = wrapper.validate_wrapper_stack()

        assert len(issues) > 0
        assert any("Missing 'obs_order'" in issue for issue in issues)
        assert any("Missing 'state_order'" in issue for issue in issues)

    def test_validate_wrapper_stack_dimension_mismatch(self, wrapper_no_noise):
        """Test wrapper stack validation with dimension mismatch."""
        wrapper = wrapper_no_noise

        # Mock valid observations
        wrapper.unwrapped._get_observations = Mock(return_value={
            "policy": torch.randn(4, 12),
            "critic": torch.randn(4, 16)
        })

        # Mock import with wrong dimensions
        with patch('builtins.__import__') as mock_import:
            mock_module = Mock()
            mock_module.OBS_DIM_CFG = {"fingertip_pos": 3, "force_torque": 6, "ee_linvel": 5}  # Wrong dimension

            def side_effect(name, *args, **kwargs):
                if name == 'envs.factory.factory_env_cfg':
                    return mock_module
                return __import__.__wrapped__(name, *args, **kwargs)

            mock_import.side_effect = side_effect

            issues = wrapper.validate_wrapper_stack()

        assert len(issues) > 0
        assert any("Observation space mismatch" in issue for issue in issues)

    def test_validate_wrapper_stack_import_error(self, wrapper_no_noise):
        """Test wrapper stack validation with import error."""
        wrapper = wrapper_no_noise

        # Mock valid observations
        wrapper.unwrapped._get_observations = Mock(return_value={
            "policy": torch.randn(4, 12),
            "critic": torch.randn(4, 16)
        })

        # Mock import to raise ImportError
        with patch('builtins.__import__', side_effect=ImportError("Test import error")):
            issues = wrapper.validate_wrapper_stack()

        assert len(issues) > 0
        assert any("Could not import OBS_DIM_CFG" in issue for issue in issues)

    def test_validate_wrapper_stack_observation_exception(self, wrapper_no_noise):
        """Test wrapper stack validation when getting observations raises exception."""
        wrapper = wrapper_no_noise

        # Mock get_observations to raise exception
        wrapper.unwrapped._get_observations = Mock(side_effect=Exception("Test observation error"))

        issues = wrapper.validate_wrapper_stack()

        assert len(issues) > 0
        assert any("Failed to get observations" in issue for issue in issues)

    def test_step_initialization(self, mock_env):
        """Test step method initializes wrapper when robot becomes available."""
        # Start without robot
        delattr(mock_env, '_robot')

        wrapper = ObservationManagerWrapper(mock_env)
        assert not wrapper._wrapper_initialized

        # Add robot and call step
        mock_env._robot = Mock()
        action = torch.zeros((4, 6), device='cpu')

        with patch.object(wrapper.env, 'step', return_value=({"policy": torch.zeros(4, 10), "critic": torch.zeros(4, 10)}, torch.zeros(4), torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool), {})):
            wrapper.step(action)

        assert wrapper._wrapper_initialized

    def test_reset_initialization(self, mock_env):
        """Test reset method initializes wrapper when robot becomes available."""
        # Start without robot
        delattr(mock_env, '_robot')

        wrapper = ObservationManagerWrapper(mock_env)
        assert not wrapper._wrapper_initialized

        # Add robot and call reset
        mock_env._robot = Mock()

        with patch.object(wrapper.env, 'reset', return_value=({"policy": torch.zeros(4, 10), "critic": torch.zeros(4, 10)}, {})):
            wrapper.reset()

        assert wrapper._wrapper_initialized

    def test_integration_full_cycle(self, wrapper_with_noise):
        """Test full integration cycle with different input formats."""
        wrapper = wrapper_with_noise

        # Test with dictionary input
        dict_obs = {
            "fingertip_pos": torch.randn(4, 3, device='cpu'),
            "force_torque": torch.randn(4, 6, device='cpu'),
            "ee_linvel": torch.randn(4, 3, device='cpu'),
            "fingertip_quat": torch.randn(4, 4, device='cpu')
        }
        wrapper._original_get_observations = Mock(return_value=dict_obs)

        result = wrapper._wrapped_get_observations()

        # Should compose, apply noise, and validate
        assert 'policy' in result
        assert 'critic' in result
        assert result['policy'].shape == (4, 12)
        assert result['critic'].shape == (4, 16)

        # Test with tensor input - temporarily disable noise to avoid dimension issues
        wrapper.use_obs_noise = False
        tensor_obs = torch.ones(4, 8, device='cpu')
        wrapper._original_get_observations = Mock(return_value=tensor_obs)

        result = wrapper._wrapped_get_observations()

        # Should convert to standard format
        assert 'policy' in result
        assert 'critic' in result
        assert result['policy'].shape == tensor_obs.shape
        assert result['critic'].shape == tensor_obs.shape
        wrapper.use_obs_noise = True  # Restore for next test

        # Test with already standard format
        # Override noise config to match tensor dimensions
        wrapper.obs_noise_mean = {"policy": [0.0] * 12, "critic": [0.0] * 16}
        wrapper.obs_noise_std = {"policy": [0.1] * 12, "critic": [0.05] * 16}

        standard_obs = {
            "policy": torch.zeros(4, 12, device='cpu'),
            "critic": torch.zeros(4, 16, device='cpu')
        }
        wrapper._original_get_observations = Mock(return_value=standard_obs)

        result = wrapper._wrapped_get_observations()

        # Should apply noise only
        assert 'policy' in result
        assert 'critic' in result
        assert not torch.allclose(result['policy'], standard_obs['policy'], atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__])