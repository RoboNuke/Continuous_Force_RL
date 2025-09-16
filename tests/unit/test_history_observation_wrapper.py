"""
Unit tests for HistoryObservationWrapper.

Tests the selective historical observation tracking including history buffers,
acceleration calculations, and observation space management.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tests.mocks.mock_isaac_lab import MockEnvironment, MockConfig
from wrappers.observations.history_observation_wrapper import HistoryObservationWrapper


class TestHistoryObservationWrapper:
    """Test suite for HistoryObservationWrapper."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""
        env = MockEnvironment(num_envs=4, device='cpu')

        # Add configuration for history wrapper
        env.cfg.decimation = 8
        env.cfg.history_samples = 4
        env.cfg.sim = Mock()
        env.cfg.sim.dt = 1/120

        # Add observation and state orders
        env.cfg.obs_order = ["fingertip_pos", "force_torque", "ee_linvel"]
        env.cfg.state_order = ["fingertip_pos", "force_torque", "ee_linvel", "fingertip_quat"]
        env.cfg.observation_space = 12
        env.cfg.state_space = 16

        # Add robot and observation data
        env._robot = Mock()
        env.fingertip_midpoint_pos = torch.tensor([[0.5, 0.0, 0.5]] * 4, device='cpu')
        env.fingertip_midpoint_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4, device='cpu')
        env.ee_linvel_fd = torch.tensor([[0.1, 0.0, 0.0]] * 4, device='cpu')
        env.ee_angvel_fd = torch.tensor([[0.0, 0.1, 0.0]] * 4, device='cpu')
        env.robot_force_torque = torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]] * 4, device='cpu')
        env.held_pos = torch.tensor([[0.6, 0.1, 0.6]] * 4, device='cpu')
        env.held_quat = torch.tensor([[0.9, 0.1, 0.0, 0.0]] * 4, device='cpu')
        env.fixed_pos_obs_frame = torch.tensor([[0.5, 0.0, 0.5]] * 4, device='cpu')
        env.init_fixed_pos_obs_noise = torch.tensor([[0.01, 0.01, 0.01]] * 4, device='cpu')

        return env

    @pytest.fixture
    def wrapper_default(self, mock_env):
        """Create wrapper with default settings."""
        return HistoryObservationWrapper(mock_env)

    @pytest.fixture
    def wrapper_custom(self, mock_env):
        """Create wrapper with custom settings."""
        return HistoryObservationWrapper(
            mock_env,
            history_components=["force_torque", "ee_linvel"],
            history_length=6,
            history_samples=3,
            calc_acceleration=True
        )

    def test_initialization_default(self, wrapper_default):
        """Test wrapper initialization with default settings."""
        wrapper = wrapper_default

        assert wrapper.history_components == ["force_torque", "ee_linvel", "ee_angvel"]
        assert wrapper.history_length == 8  # From mock env decimation
        assert wrapper.num_samples == 4  # From mock env history_samples
        assert wrapper.calc_acceleration == False
        assert wrapper.num_envs == 4
        assert str(wrapper.device) == 'cpu'
        assert wrapper._wrapper_initialized == True

    def test_initialization_custom(self, wrapper_custom):
        """Test wrapper initialization with custom settings."""
        wrapper = wrapper_custom

        assert wrapper.history_components == ["force_torque", "ee_linvel"]
        assert wrapper.history_length == 6
        assert wrapper.num_samples == 3
        assert wrapper.calc_acceleration == True
        assert wrapper.num_envs == 4
        assert str(wrapper.device) == 'cpu'
        assert wrapper._wrapper_initialized == True

    def test_initialization_without_robot(self, mock_env):
        """Test initialization when robot is not available initially."""
        delattr(mock_env, '_robot')
        wrapper = HistoryObservationWrapper(mock_env)
        assert not wrapper._wrapper_initialized

    def test_keep_indices_calculation(self, wrapper_custom):
        """Test calculation of keep indices for history sampling."""
        wrapper = wrapper_custom

        # With history_length=6 and num_samples=3, should sample indices 0, 2.5, 5 -> 0, 2, 5
        expected_indices = torch.tensor([0, 2, 5], dtype=torch.int32)
        torch.testing.assert_close(wrapper.keep_idxs, expected_indices)

    def test_keep_indices_single_sample(self, mock_env):
        """Test keep indices when only one sample is requested."""
        wrapper = HistoryObservationWrapper(mock_env, history_samples=1)

        # Should always take the last value (history_length - 1)
        assert wrapper.keep_idxs[0] == wrapper.history_length - 1

    def test_keep_indices_full_history(self, mock_env):
        """Test keep indices when samples equal history length."""
        wrapper = HistoryObservationWrapper(mock_env, history_length=4, history_samples=4)

        expected_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        torch.testing.assert_close(wrapper.keep_idxs, expected_indices)

    def test_update_observation_dimensions(self, mock_env):
        """Test observation dimension updates."""
        # Create wrapper to test dimension updates
        wrapper = HistoryObservationWrapper(
            mock_env,
            history_components=["force_torque", "ee_linvel"],
            history_samples=2
        )

        # Test that the wrapper was created successfully
        assert wrapper.history_components == ["force_torque", "ee_linvel"]
        assert wrapper.num_samples == 2

        # Test the dimension calculation logic directly
        component_dims = {
            "force_torque": 6,
            "ee_linvel": 3,
            "ee_angvel": 3,
            "fingertip_pos": 3,
        }

        # Create a test dimension config
        test_dim_cfg = {'force_torque': 6, 'ee_linvel': 3, 'ee_angvel': 3, 'fingertip_pos': 3}

        # Test the scaling function directly
        wrapper._apply_history_scaling(test_dim_cfg, component_dims)

        # force_torque: 6 * 2 = 12, ee_linvel: 3 * 2 = 6
        assert test_dim_cfg['force_torque'] == 12
        assert test_dim_cfg['ee_linvel'] == 6
        # Non-history components should remain unchanged
        assert test_dim_cfg['ee_angvel'] == 3

    def test_update_observation_dimensions_with_acceleration(self, mock_env):
        """Test observation dimension updates with acceleration."""
        # Create wrapper with acceleration enabled
        wrapper = HistoryObservationWrapper(
            mock_env,
            history_components=["force_torque", "ee_linvel"],
            history_samples=2,
            calc_acceleration=True
        )

        # Test that the wrapper was created successfully
        assert wrapper.history_components == ["force_torque", "ee_linvel"]
        assert wrapper.num_samples == 2
        assert wrapper.calc_acceleration == True

        # Test the dimension calculation logic directly
        component_dims = {
            "force_torque": 6,
            "ee_linvel": 3,
            "ee_angvel": 3,
        }

        # Create a test dimension config
        test_dim_cfg = {'force_torque': 6, 'ee_linvel': 3, 'ee_angvel': 3}

        # Test the scaling function directly
        wrapper._apply_history_scaling(test_dim_cfg, component_dims)

        # Original dimensions scaled
        assert test_dim_cfg['force_torque'] == 12  # 6 * 2
        assert test_dim_cfg['ee_linvel'] == 6  # 3 * 2

        # Acceleration components added
        assert test_dim_cfg['ee_linacc'] == 6  # Same as ee_linvel
        assert test_dim_cfg['force_jerk'] == 12  # Same as force_torque
        assert test_dim_cfg['force_snap'] == 12  # Same as force_torque

    def test_init_history_buffers(self, wrapper_custom):
        """Test initialization of history buffers."""
        wrapper = wrapper_custom

        # Should have buffers for force_torque and ee_linvel
        assert 'force_torque' in wrapper.history_buffers
        assert 'ee_linvel' in wrapper.history_buffers
        assert 'ee_angvel' not in wrapper.history_buffers

        # Check buffer shapes
        assert wrapper.history_buffers['force_torque'].shape == (4, 6, 6)  # (num_envs, history_length, dim)
        assert wrapper.history_buffers['ee_linvel'].shape == (4, 6, 3)

        # Should have acceleration buffers since calc_acceleration=True
        assert 'ee_linacc' in wrapper.acceleration_buffers
        assert 'force_jerk' in wrapper.acceleration_buffers
        assert 'force_snap' in wrapper.acceleration_buffers

    def test_init_history_buffers_no_acceleration(self, wrapper_default):
        """Test initialization of history buffers without acceleration."""
        wrapper = wrapper_default

        # Should have history buffers
        assert len(wrapper.history_buffers) == 3
        assert 'force_torque' in wrapper.history_buffers
        assert 'ee_linvel' in wrapper.history_buffers
        assert 'ee_angvel' in wrapper.history_buffers

        # Should not have acceleration buffers
        assert len(wrapper.acceleration_buffers) == 0

    def test_wrapper_initialization(self, wrapper_default):
        """Test wrapper method override initialization."""
        wrapper = wrapper_default

        # Should be initialized since robot exists
        assert wrapper._wrapper_initialized == True

        # Check that methods are stored and overridden (if they exist)
        # Note: MockEnvironment doesn't have all methods, so some may be None
        if hasattr(wrapper.unwrapped, '_get_observations'):
            assert wrapper._original_get_observations is not None
        if hasattr(wrapper.unwrapped, '_reset_idx'):
            assert wrapper._original_reset_idx is not None
        if hasattr(wrapper.unwrapped, '_pre_physics_step'):
            assert wrapper._original_pre_physics_step is not None

    def test_get_current_observations(self, wrapper_default):
        """Test getting current observations from environment."""
        wrapper = wrapper_default

        obs = wrapper._get_current_observations()

        # Check that observations are extracted correctly
        assert 'fingertip_pos' in obs
        assert 'fingertip_quat' in obs
        assert 'ee_linvel' in obs
        assert 'ee_angvel' in obs
        assert 'force_torque' in obs
        assert 'held_pos' in obs
        assert 'held_quat' in obs

        # Check relative positions are calculated
        assert 'fingertip_pos_rel_fixed' in obs
        assert 'held_pos_rel_fixed' in obs

        # Check values
        torch.testing.assert_close(obs['fingertip_pos'], wrapper.unwrapped.fingertip_midpoint_pos)
        torch.testing.assert_close(obs['force_torque'], wrapper.unwrapped.robot_force_torque)

    def test_get_current_observations_missing_attributes(self, mock_env):
        """Test getting observations when some environment attributes are missing."""
        # Remove some attributes
        delattr(mock_env, 'held_pos')
        delattr(mock_env, 'held_quat')
        delattr(mock_env, 'fixed_pos_obs_frame')

        wrapper = HistoryObservationWrapper(mock_env)
        obs = wrapper._get_current_observations()

        # Should still work without missing attributes
        assert 'fingertip_pos' in obs
        assert 'force_torque' in obs
        assert 'held_pos' not in obs
        assert 'held_quat' not in obs
        assert 'fingertip_pos_rel_fixed' not in obs

    def test_update_history_initial(self, wrapper_custom):
        """Test initial history update (reset=True)."""
        wrapper = wrapper_custom

        # Initial update should fill entire history with current values
        wrapper._update_history(reset=True)

        # Check that all history steps have the same value
        for component in wrapper.history_components:
            if component in wrapper.history_buffers:
                buffer = wrapper.history_buffers[component]
                # All time steps should have the same value (current observation)
                first_step = buffer[:, 0, :]
                last_step = buffer[:, -1, :]
                torch.testing.assert_close(first_step, last_step)

    def test_update_history_step(self, wrapper_custom):
        """Test history update during stepping."""
        wrapper = wrapper_custom

        # Initialize history
        wrapper._update_history(reset=True)

        # Get initial state
        initial_force = wrapper.history_buffers['force_torque'][:, -1, :].clone()

        # Change environment state
        wrapper.unwrapped.robot_force_torque = torch.tensor([[2.0, 3.0, 4.0, 0.2, 0.3, 0.4]] * 4, device='cpu')

        # Update history
        wrapper._update_history(reset=False)

        # Check that history was rolled and new value added
        new_force = wrapper.history_buffers['force_torque'][:, -1, :]
        old_force = wrapper.history_buffers['force_torque'][:, -2, :]

        torch.testing.assert_close(old_force, initial_force)
        torch.testing.assert_close(new_force, wrapper.unwrapped.robot_force_torque)

    def test_update_history_acceleration(self, wrapper_custom):
        """Test acceleration calculation during history update."""
        wrapper = wrapper_custom

        # Initialize history
        wrapper._update_history(reset=True)

        # Acceleration buffers should be zero after reset
        assert torch.all(wrapper.acceleration_buffers['ee_linacc'] == 0)
        assert torch.all(wrapper.acceleration_buffers['force_jerk'] == 0)
        assert torch.all(wrapper.acceleration_buffers['force_snap'] == 0)

        # Change environment state
        wrapper.unwrapped.ee_linvel_fd = torch.tensor([[0.2, 0.1, 0.0]] * 4, device='cpu')
        wrapper.unwrapped.robot_force_torque = torch.tensor([[2.0, 3.0, 4.0, 0.2, 0.3, 0.4]] * 4, device='cpu')

        # Update history
        wrapper._update_history(reset=False)

        # Check that acceleration was calculated
        assert not torch.all(wrapper.acceleration_buffers['ee_linacc'][:, -1, :] == 0)
        assert not torch.all(wrapper.acceleration_buffers['force_jerk'][:, -1, :] == 0)

    def test_finite_difference(self, wrapper_custom):
        """Test finite difference calculation."""
        wrapper = wrapper_custom

        # Create test history buffer
        history_buffer = torch.zeros((4, 3, 6), device='cpu')
        history_buffer[:, 0, :] = 1.0
        history_buffer[:, 1, :] = 2.0
        history_buffer[:, 2, :] = 4.0

        # Calculate finite difference
        diff = wrapper._finite_difference(history_buffer)

        # Expected: (4.0 - 2.0) / dt = 2.0 / (1/120) = 240.0
        dt = 1/120
        expected_diff = torch.full((4, 6), 240.0, device='cpu')
        torch.testing.assert_close(diff, expected_diff)

    def test_finite_difference_insufficient_history(self, wrapper_custom):
        """Test finite difference with insufficient history."""
        wrapper = wrapper_custom

        # Create history buffer with only one timestep
        history_buffer = torch.zeros((4, 1, 6), device='cpu')

        # Should return zeros
        diff = wrapper._finite_difference(history_buffer)
        expected_diff = torch.zeros((4, 6), device='cpu')
        torch.testing.assert_close(diff, expected_diff)

    def test_build_observation_dicts(self, wrapper_custom):
        """Test building observation dictionaries."""
        wrapper = wrapper_custom

        # Initialize history
        wrapper._update_history(reset=True)

        # Build observation dictionaries
        obs_dict, state_dict = wrapper._build_observation_dicts()

        # Check that history components use history data
        assert 'force_torque' in obs_dict
        assert 'ee_linvel' in obs_dict

        # Check that non-history components use current data
        assert 'fingertip_pos' in obs_dict
        torch.testing.assert_close(obs_dict['fingertip_pos'], wrapper.unwrapped.fingertip_midpoint_pos)

        # Check acceleration components
        assert 'ee_linacc' in obs_dict
        assert 'force_jerk' in obs_dict
        assert 'force_snap' in obs_dict

        # Check shapes - history components should be flattened
        # force_torque: (4, 3, 6) -> (4, 18) where 3 is num_samples, 6 is force dimension
        assert obs_dict['force_torque'].shape == (4, 18)
        # ee_linvel: (4, 3, 3) -> (4, 9)
        assert obs_dict['ee_linvel'].shape == (4, 9)

    def test_process_dict_observations(self, wrapper_default):
        """Test processing observations in dictionary format."""
        wrapper = wrapper_default

        # Mock original observations
        original_obs = {
            "policy": torch.randn(4, 10, device='cpu'),
            "critic": torch.randn(4, 12, device='cpu')
        }

        # Mock methods
        wrapper._build_observation_dicts = Mock(return_value=({
            'fingertip_pos': torch.randn(4, 3, device='cpu'),
            'force_torque': torch.randn(4, 24, device='cpu'),  # 6 * 4 samples
            'ee_linvel': torch.randn(4, 12, device='cpu'),     # 3 * 4 samples
        }, {
            'fingertip_pos': torch.randn(4, 3, device='cpu'),
            'force_torque': torch.randn(4, 24, device='cpu'),
            'ee_linvel': torch.randn(4, 12, device='cpu'),
            'fingertip_quat': torch.randn(4, 4, device='cpu'),
        }))

        result = wrapper._process_dict_observations(original_obs)

        # Should return dictionary with policy and critic keys
        assert 'policy' in result
        assert 'critic' in result

        # Check that observations are concatenated according to order
        # obs_order = ["fingertip_pos", "force_torque", "ee_linvel"] = 3 + 24 + 12 = 39
        assert result['policy'].shape == (4, 39)
        # state_order includes fingertip_quat too: 3 + 24 + 12 + 4 = 43
        assert result['critic'].shape == (4, 43)

    def test_wrapped_get_observations(self, wrapper_default):
        """Test wrapped get observations method."""
        wrapper = wrapper_default

        # Mock original get_observations
        wrapper._original_get_observations = Mock(return_value={
            "policy": torch.randn(4, 10, device='cpu'),
            "critic": torch.randn(4, 12, device='cpu')
        })

        # Mock process method
        expected_result = {
            "policy": torch.randn(4, 20, device='cpu'),
            "critic": torch.randn(4, 25, device='cpu')
        }
        wrapper._process_dict_observations = Mock(return_value=expected_result)

        result = wrapper._wrapped_get_observations()

        # Should call process method and return its result
        wrapper._process_dict_observations.assert_called_once()
        assert result == expected_result

    def test_wrapped_get_observations_no_original(self, wrapper_default):
        """Test wrapped get observations without original method."""
        wrapper = wrapper_default
        wrapper._original_get_observations = None

        # Mock process method
        expected_result = {
            "policy": torch.randn(4, 20, device='cpu'),
            "critic": torch.randn(4, 25, device='cpu')
        }
        wrapper._process_dict_observations = Mock(return_value=expected_result)

        result = wrapper._wrapped_get_observations()

        # Should call process method - check it was called (tensor comparison is tricky with empty tensors)
        wrapper._process_dict_observations.assert_called_once()
        assert result == expected_result

    def test_wrapped_get_observations_non_dict(self, wrapper_default):
        """Test wrapped get observations with non-dictionary return."""
        wrapper = wrapper_default

        # Mock original get_observations to return non-dict
        tensor_obs = torch.randn(4, 10, device='cpu')
        wrapper._original_get_observations = Mock(return_value=tensor_obs)

        result = wrapper._wrapped_get_observations()

        # Should return original tensor unchanged
        torch.testing.assert_close(result, tensor_obs)

    def test_wrapped_reset_idx(self, wrapper_default):
        """Test wrapped reset idx method."""
        wrapper = wrapper_default

        # Mock original reset_idx
        wrapper._original_reset_idx = Mock()

        # Mock update_history
        wrapper._update_history = Mock()

        env_ids = torch.tensor([0, 1])
        wrapper._wrapped_reset_idx(env_ids)

        # Should call original method and update history
        wrapper._original_reset_idx.assert_called_once_with(env_ids)
        wrapper._update_history.assert_called_once_with(reset=True)

    def test_wrapped_reset_idx_no_original(self, wrapper_default):
        """Test wrapped reset idx without original method."""
        wrapper = wrapper_default
        wrapper._original_reset_idx = None

        # Mock update_history
        wrapper._update_history = Mock()

        env_ids = torch.tensor([0, 1])
        wrapper._wrapped_reset_idx(env_ids)

        # Should still update history
        wrapper._update_history.assert_called_once_with(reset=True)

    def test_wrapped_pre_physics_step(self, wrapper_default):
        """Test wrapped pre-physics step method."""
        wrapper = wrapper_default

        # Mock original pre_physics_step
        wrapper._original_pre_physics_step = Mock()

        # Mock update_history
        wrapper._update_history = Mock()

        action = torch.randn(4, 6, device='cpu')
        wrapper._wrapped_pre_physics_step(action)

        # Should call original method and update history
        wrapper._original_pre_physics_step.assert_called_once_with(action)
        wrapper._update_history.assert_called_once_with(reset=False)

    def test_wrapped_pre_physics_step_no_original(self, wrapper_default):
        """Test wrapped pre-physics step without original method."""
        wrapper = wrapper_default
        wrapper._original_pre_physics_step = None

        # Mock update_history
        wrapper._update_history = Mock()

        action = torch.randn(4, 6, device='cpu')
        wrapper._wrapped_pre_physics_step(action)

        # Should still update history
        wrapper._update_history.assert_called_once_with(reset=False)

    def test_step_initialization(self, mock_env):
        """Test step method initializes wrapper when robot becomes available."""
        # Start without robot
        delattr(mock_env, '_robot')

        wrapper = HistoryObservationWrapper(mock_env)
        assert not wrapper._wrapper_initialized

        # Add robot and call step
        mock_env._robot = Mock()
        action = torch.zeros((4, 6), device='cpu')

        with patch.object(wrapper.env, 'step', return_value=(torch.zeros(4, 10), torch.zeros(4), torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool), {})):
            wrapper.step(action)

        assert wrapper._wrapper_initialized

    def test_reset_initialization(self, mock_env):
        """Test reset method initializes wrapper when robot becomes available."""
        # Start without robot
        delattr(mock_env, '_robot')

        wrapper = HistoryObservationWrapper(mock_env)
        assert not wrapper._wrapper_initialized

        # Add robot and call reset
        mock_env._robot = Mock()

        with patch.object(wrapper.env, 'reset', return_value=(torch.zeros(4, 10), {})):
            wrapper.reset()

        assert wrapper._wrapper_initialized

    def test_get_history_stats(self, wrapper_custom):
        """Test getting history statistics."""
        wrapper = wrapper_custom

        stats = wrapper.get_history_stats()

        expected_stats = {
            'history_components': ["force_torque", "ee_linvel"],
            'history_length': 6,
            'num_samples': 3,
            'calc_acceleration': True,
            'buffer_count': 2,  # force_torque, ee_linvel
            'acceleration_buffer_count': 3,  # ee_linacc, force_jerk, force_snap
        }

        assert stats == expected_stats

    def test_get_component_history(self, wrapper_custom):
        """Test getting history for specific components."""
        wrapper = wrapper_custom

        # Initialize history
        wrapper._update_history(reset=True)

        # Test getting history buffer
        force_history = wrapper.get_component_history('force_torque')
        assert force_history is not None
        assert force_history.shape == (4, 6, 6)
        # Check that returned history is a copy (different data pointers)
        assert force_history.data_ptr() != wrapper.history_buffers['force_torque'].data_ptr()

        # Test getting acceleration buffer
        acc_history = wrapper.get_component_history('ee_linacc')
        assert acc_history is not None
        assert acc_history.shape == (4, 6, 3)

        # Test getting non-existent component
        no_history = wrapper.get_component_history('nonexistent')
        assert no_history is None

    def test_integration_full_cycle(self, wrapper_custom):
        """Test full integration cycle with step and reset."""
        wrapper = wrapper_custom

        # Initialize
        wrapper._update_history(reset=True)

        # Simulate several steps
        for i in range(5):
            # Change environment state
            wrapper.unwrapped.robot_force_torque = torch.tensor([[float(i), float(i+1), float(i+2), 0.1*i, 0.1*(i+1), 0.1*(i+2)]] * 4, device='cpu')
            wrapper.unwrapped.ee_linvel_fd = torch.tensor([[0.1*i, 0.1*(i+1), 0.1*(i+2)]] * 4, device='cpu')

            # Update history
            wrapper._wrapped_pre_physics_step(torch.zeros(4, 6, device='cpu'))

        # Check that history contains different values
        force_buffer = wrapper.history_buffers['force_torque']
        first_step = force_buffer[:, 0, :]
        last_step = force_buffer[:, -1, :]

        # Should be different (not all the same due to rolling)
        assert not torch.allclose(first_step, last_step)

        # Check acceleration calculation
        acc_buffer = wrapper.acceleration_buffers['ee_linacc']
        assert not torch.all(acc_buffer == 0)

        # Test reset
        wrapper._wrapped_reset_idx(torch.tensor([0, 1]))

        # After reset, all history steps should have current value
        force_buffer_after = wrapper.history_buffers['force_torque']
        first_step_after = force_buffer_after[:, 0, :]
        last_step_after = force_buffer_after[:, -1, :]
        torch.testing.assert_close(first_step_after, last_step_after)

    def test_integration_with_observations(self, wrapper_default):
        """Test integration with observation generation."""
        wrapper = wrapper_default

        # Mock original get_observations
        wrapper._original_get_observations = Mock(return_value={
            "policy": torch.randn(4, 15, device='cpu'),
            "critic": torch.randn(4, 19, device='cpu')
        })

        # Initialize and get observations
        wrapper._update_history(reset=True)
        obs = wrapper._wrapped_get_observations()

        # Should return observations
        assert 'policy' in obs
        assert 'critic' in obs
        assert obs['policy'].shape[0] == 4
        assert obs['critic'].shape[0] == 4


if __name__ == '__main__':
    pytest.main([__file__])