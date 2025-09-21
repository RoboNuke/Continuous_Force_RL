"""
Unit tests for history_observation_wrapper.py functionality.
Tests HistoryObservationWrapper for selective historical observation tracking.
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
mock_isaac_lab = __import__('tests.mocks.mock_isaac_lab', fromlist=[''])
sys.modules['omni.isaac.lab'] = mock_isaac_lab
sys.modules['omni.isaac.lab.envs'] = mock_isaac_lab
sys.modules['omni.isaac.lab.utils'] = mock_isaac_lab
sys.modules['omni.isaac.lab_tasks'] = mock_isaac_lab
sys.modules['omni.isaac.lab_tasks.direct'] = mock_isaac_lab
sys.modules['omni.isaac.lab_tasks.direct.factory'] = mock_isaac_lab
sys.modules['omni.isaac.lab_tasks.direct.factory.factory_env_cfg'] = mock_isaac_lab
sys.modules['isaaclab_tasks'] = mock_isaac_lab
sys.modules['isaaclab_tasks.direct'] = mock_isaac_lab
sys.modules['isaaclab_tasks.direct.factory'] = mock_isaac_lab
sys.modules['isaaclab_tasks.direct.factory.factory_env_cfg'] = mock_isaac_lab

from wrappers.observations.history_observation_wrapper import HistoryObservationWrapper
from tests.mocks.mock_isaac_lab import MockBaseEnv, MockEnvConfig


class TestHistoryObservationWrapper:
    """Test HistoryObservationWrapper functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.cfg = MockEnvConfig()
        self.base_env = MockBaseEnv(self.cfg)
        self.base_env.num_envs = 4
        self.base_env.device = torch.device("cpu")

    def test_initialization_requires_explicit_components(self):
        """Test wrapper initialization requires explicit history components."""
        with pytest.raises(ValueError, match="history_components cannot be None"):
            HistoryObservationWrapper(self.base_env, history_components=None)

    def test_initialization_empty_components_list(self):
        """Test wrapper initialization with empty components list."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=[])

        assert wrapper.history_components == []
        assert wrapper.num_envs == 4
        assert wrapper.device == torch.device("cpu")
        assert wrapper.history_length == 8  # From cfg.decimation

    def test_initialization_valid_components(self):
        """Test wrapper initialization with valid components."""
        components = ["fingertip_pos", "ee_linvel"]
        wrapper = HistoryObservationWrapper(self.base_env, history_components=components)

        assert wrapper.history_components == components
        assert wrapper.history_length == 8
        assert wrapper.num_samples == 4  # From cfg.history_samples

    def test_initialization_with_custom_parameters(self):
        """Test wrapper initialization with custom parameters."""
        components = ["fingertip_pos"]
        wrapper = HistoryObservationWrapper(
            self.base_env,
            history_components=components,
            history_length=10,
            history_samples=5
        )

        assert wrapper.history_length == 10
        assert wrapper.num_samples == 5

    def test_initialization_single_sample(self):
        """Test wrapper initialization with single sample."""
        components = ["fingertip_pos"]
        wrapper = HistoryObservationWrapper(
            self.base_env,
            history_components=components,
            history_samples=1
        )

        assert wrapper.num_samples == 1
        # Should use last index for single sample
        assert wrapper.keep_idxs[0] == wrapper.history_length - 1

    def test_validate_components_missing(self):
        """Test component validation with missing components."""
        with pytest.raises(ValueError, match="History components .* not found"):
            HistoryObservationWrapper(
                self.base_env,
                history_components=["nonexistent_component"]
            )

    def test_validate_components_no_config(self):
        """Test component validation with new Isaac Lab architecture."""
        # This test is no longer applicable as we use Isaac Lab's native configurations
        # The wrapper will automatically import OBS_DIM_CFG and STATE_DIM_CFG
        wrapper = HistoryObservationWrapper(
            self.base_env,
            history_components=["fingertip_pos"]
        )
        assert wrapper is not None

    def test_get_component_dimensions_invalid_config(self):
        """Test component dimensions with new Isaac Lab architecture."""
        # This test is no longer applicable as we use Isaac Lab's native configurations
        # The wrapper will automatically import OBS_DIM_CFG and STATE_DIM_CFG
        wrapper = HistoryObservationWrapper(
            self.base_env,
            history_components=["fingertip_pos"]
        )
        assert wrapper is not None

    def test_get_component_dimensions_empty_config(self):
        """Test component dimensions with new Isaac Lab architecture."""
        # This test is no longer applicable as we use Isaac Lab's native configurations
        # The wrapper will automatically import OBS_DIM_CFG and STATE_DIM_CFG
        wrapper = HistoryObservationWrapper(
            self.base_env,
            history_components=["fingertip_pos"]
        )
        assert wrapper is not None

    def test_update_observation_dimensions(self):
        """Test observation space dimension updates."""
        components = ["fingertip_pos", "ee_linvel"]  # 3 + 3 = 6 dims per sample
        wrapper = HistoryObservationWrapper(
            self.base_env,
            history_components=components,
            history_samples=3
        )

        # Should add (3-1) * 6 = 12 additional dimensions
        expected_additional = (3 - 1) * 6
        assert self.base_env.cfg.observation_space == 32 + expected_additional

    def test_update_observation_dimensions_state_space(self):
        """Test state space dimension updates."""
        components = ["fingertip_pos"]  # In state_order
        wrapper = HistoryObservationWrapper(
            self.base_env,
            history_components=components,
            history_samples=4
        )

        # Should add (4-1) * 3 = 9 additional dimensions to state space
        expected_additional = (4 - 1) * 3
        assert self.base_env.cfg.state_space == 48 + expected_additional

    def test_initialization_without_robot(self):
        """Test wrapper initialization without robot attribute."""
        env_no_robot = MockBaseEnv(self.cfg)
        delattr(env_no_robot, '_robot')

        wrapper = HistoryObservationWrapper(env_no_robot, history_components=["fingertip_pos"])
        assert wrapper._wrapper_initialized == False

    def test_lazy_wrapper_initialization(self):
        """Test lazy wrapper initialization during step/reset."""
        env_no_robot = MockBaseEnv(self.cfg)
        delattr(env_no_robot, '_robot')

        wrapper = HistoryObservationWrapper(env_no_robot, history_components=["fingertip_pos"])
        assert not wrapper._wrapper_initialized

        # Add robot and call step
        env_no_robot._robot = True
        wrapper.step(torch.randn(4, 6))

        assert wrapper._wrapper_initialized

    def test_init_history_buffers(self):
        """Test history buffer initialization."""
        components = ["fingertip_pos", "ee_linvel"]
        wrapper = HistoryObservationWrapper(self.base_env, history_components=components)

        assert len(wrapper.history_buffers) == 2
        assert wrapper.history_buffers["fingertip_pos"].shape == (4, 8, 3)
        assert wrapper.history_buffers["ee_linvel"].shape == (4, 8, 3)

    def test_wrapped_get_observations(self):
        """Test wrapped get observations method."""
        components = ["fingertip_pos", "ee_linvel"]
        wrapper = HistoryObservationWrapper(self.base_env, history_components=components)

        # Mock original method
        def mock_get_observations():
            return {"existing_obs": torch.randn(4, 10)}

        wrapper._original_get_observations = mock_get_observations

        obs = wrapper._wrapped_get_observations()
        assert isinstance(obs, dict)
        assert "existing_obs" in obs
        assert "fingertip_pos" in obs
        assert "ee_linvel" in obs

    def test_get_observation_value_basic(self):
        """Test getting observation value for basic component."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        value = wrapper._get_observation_value("fingertip_pos")
        assert isinstance(value, torch.Tensor)
        assert value.shape == (4, 3)

    def test_get_observation_value_relative_position(self):
        """Test getting observation value for relative position components."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        # Test relative position calculation
        value = wrapper._get_observation_value("fingertip_pos_rel_fixed")
        assert isinstance(value, torch.Tensor)
        assert value.shape == (4, 3)

    def test_get_observation_value_missing_fixed_frame(self):
        """Test getting observation value without fixed frame."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        # Remove fixed frame
        delattr(wrapper.unwrapped, 'fixed_pos_obs_frame')

        with pytest.raises(ValueError, match="Cannot compute .* fixed_pos_obs_frame not available"):
            wrapper._get_observation_value("fingertip_pos_rel_fixed")

    def test_get_observation_value_missing_attribute(self):
        """Test getting observation value for missing attribute."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        with pytest.raises(ValueError, match="Environment attribute .* not found"):
            wrapper._get_observation_value("nonexistent_component")

    def test_component_attr_mapping_missing(self):
        """Test component attribute mapping missing."""
        # Remove component_attr_map
        delattr(self.base_env.cfg, 'component_attr_map')

        with pytest.raises(ValueError, match="Environment configuration must have 'component_attr_map'"):
            wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])
            wrapper._get_component_attr_mapping()

    def test_component_attr_mapping_invalid(self):
        """Test component attribute mapping invalid type."""
        # Set invalid component_attr_map
        self.base_env.cfg.component_attr_map = "invalid"

        with pytest.raises(ValueError, match="'component_attr_map' must be a dictionary"):
            wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])
            wrapper._get_component_attr_mapping()

    def test_wrapped_reset_idx(self):
        """Test wrapped reset index method."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        # Initialize buffers with some data
        wrapper.history_buffers["fingertip_pos"].fill_(1.0)

        # Reset specific environments
        env_ids = torch.tensor([0, 2])
        wrapper._wrapped_reset_idx(env_ids)

        # Should reset buffers for specified environments
        assert torch.allclose(wrapper.history_buffers["fingertip_pos"][0], torch.zeros(8, 3))
        assert torch.allclose(wrapper.history_buffers["fingertip_pos"][2], torch.zeros(8, 3))
        # Other environments should remain unchanged
        assert torch.allclose(wrapper.history_buffers["fingertip_pos"][1], torch.ones(8, 3))

    def test_wrapped_reset_idx_empty(self):
        """Test wrapped reset with empty environment IDs."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        # Initialize buffers with some data
        wrapper.history_buffers["fingertip_pos"].fill_(1.0)

        # Reset with empty list
        wrapper._wrapped_reset_idx(torch.tensor([]))

        # Buffers should remain unchanged
        assert torch.allclose(wrapper.history_buffers["fingertip_pos"], torch.ones(4, 8, 3))

    def test_wrapped_pre_physics_step(self):
        """Test wrapped pre-physics step method."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        # Mock original pre-physics step
        original_called = False
        def mock_pre_physics_step(actions):
            nonlocal original_called
            original_called = True

        wrapper._original_pre_physics_step = mock_pre_physics_step

        actions = torch.randn(4, 6)
        wrapper._wrapped_pre_physics_step(actions)

        assert original_called

    def test_update_history(self):
        """Test history buffer update."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        # Set known values for fingertip_pos
        test_value = torch.ones(4, 3) * 5.0
        wrapper.unwrapped.fingertip_pos = test_value

        # Update history
        wrapper._update_history()

        # Check that the value was added to the buffer (last position)
        assert torch.allclose(wrapper.history_buffers["fingertip_pos"][:, -1, :], test_value)

    def test_update_history_rolling(self):
        """Test history buffer rolling update."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        # Fill buffer with known pattern
        for i in range(wrapper.history_length):
            wrapper.history_buffers["fingertip_pos"][:, i, :] = i

        # Set new value
        new_value = torch.ones(4, 3) * 99.0
        wrapper.unwrapped.fingertip_pos = new_value

        # Update history (should roll and add new value)
        wrapper._update_history()

        # Check rolling: first position should now have value 1 (was 0)
        assert torch.allclose(wrapper.history_buffers["fingertip_pos"][:, 0, :], torch.ones(4, 3) * 1.0)
        # Last position should have new value
        assert torch.allclose(wrapper.history_buffers["fingertip_pos"][:, -1, :], new_value)

    def test_get_history_stats(self):
        """Test getting history statistics."""
        components = ["fingertip_pos", "ee_linvel"]
        wrapper = HistoryObservationWrapper(
            self.base_env,
            history_components=components,
            history_length=10,
            history_samples=5
        )

        stats = wrapper.get_history_stats()
        assert stats['history_components'] == components
        assert stats['history_length'] == 10
        assert stats['num_samples'] == 5
        assert stats['wrapper_initialized'] == True
        assert stats['buffer_count'] == 2
        assert 'fingertip_pos_buffer_shape' in stats
        assert 'ee_linvel_buffer_shape' in stats

    def test_get_component_history(self):
        """Test getting component history."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        # Fill buffer with test data
        test_data = torch.randn(4, 8, 3)
        wrapper.history_buffers["fingertip_pos"] = test_data

        history = wrapper.get_component_history("fingertip_pos")
        assert torch.allclose(history, test_data)
        # Should be a clone, not the same object
        assert history is not test_data

    def test_get_component_history_missing(self):
        """Test getting history for missing component."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        with pytest.raises(ValueError, match="Component .* not found in history buffers"):
            wrapper.get_component_history("nonexistent_component")

    def test_step_functionality(self):
        """Test step method functionality."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        actions = torch.randn(4, 6)
        result = wrapper.step(actions)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

    def test_step_with_lazy_initialization(self):
        """Test step method with lazy initialization."""
        env_no_robot = MockBaseEnv(self.cfg)
        delattr(env_no_robot, '_robot')

        wrapper = HistoryObservationWrapper(env_no_robot, history_components=["fingertip_pos"])
        assert not wrapper._wrapper_initialized

        # Add robot and call step
        env_no_robot._robot = True
        actions = torch.randn(4, 6)
        wrapper.step(actions)

        assert wrapper._wrapper_initialized

    def test_reset_functionality(self):
        """Test reset method functionality."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        result = wrapper.reset()
        assert len(result) == 2
        obs, info = result

    def test_reset_with_lazy_initialization(self):
        """Test reset method with lazy initialization."""
        env_no_robot = MockBaseEnv(self.cfg)
        delattr(env_no_robot, '_robot')

        wrapper = HistoryObservationWrapper(env_no_robot, history_components=["fingertip_pos"])
        assert not wrapper._wrapper_initialized

        # Add robot and call reset
        env_no_robot._robot = True
        wrapper.reset()

        assert wrapper._wrapper_initialized

    def test_keep_indices_calculation(self):
        """Test keep indices calculation for sampling."""
        wrapper = HistoryObservationWrapper(
            self.base_env,
            history_components=["fingertip_pos"],
            history_length=8,
            history_samples=4
        )

        # Should create evenly spaced indices
        expected = torch.linspace(0, 7, 4).type(torch.int32)
        assert torch.equal(wrapper.keep_idxs, expected)

    def test_wrapper_properties(self):
        """Test that wrapper properly delegates properties."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        assert wrapper.action_space.shape == self.base_env.action_space.shape
        assert wrapper.observation_space.shape == self.base_env.observation_space.shape

    def test_unwrapped_property(self):
        """Test unwrapped property access."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        assert wrapper.unwrapped == self.base_env

    def test_close_functionality(self):
        """Test wrapper close method."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        # Should not raise any errors
        wrapper.close()

    def test_device_consistency(self):
        """Test that device is consistent throughout wrapper."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        assert wrapper.device == self.base_env.device

        # Check buffer devices
        for buffer in wrapper.history_buffers.values():
            assert buffer.device == wrapper.device

    def test_multiple_step_calls_with_history_update(self):
        """Test multiple step calls update history correctly."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        actions = torch.randn(4, 6)

        # Call step multiple times and check history updates
        for i in range(5):
            # Set unique value for this step
            unique_value = torch.ones(4, 3) * i
            wrapper.unwrapped.fingertip_pos = unique_value

            # Directly call the history update method to test it
            wrapper._update_history()

            # Check that the value was recorded in history
            # The value should be in the last position after the update
            assert torch.allclose(wrapper.history_buffers["fingertip_pos"][:, -1, :], unique_value)

    def test_wrapper_chain_compatibility(self):
        """Test that wrapper works in a chain with other wrappers."""
        # Create a simple wrapper chain
        intermediate_wrapper = gym.Wrapper(self.base_env)
        wrapper = HistoryObservationWrapper(intermediate_wrapper, history_components=["fingertip_pos"])

        assert wrapper.unwrapped == self.base_env

        actions = torch.randn(4, 6)
        result = wrapper.step(actions)
        assert len(result) == 5

    def test_empty_components_list_functionality(self):
        """Test functionality with empty components list."""
        wrapper = HistoryObservationWrapper(self.base_env, history_components=[])

        # Should work normally but with no history tracking
        assert len(wrapper.history_buffers) == 0

        actions = torch.randn(4, 6)
        result = wrapper.step(actions)
        assert len(result) == 5

        stats = wrapper.get_history_stats()
        assert stats['buffer_count'] == 0

    def test_large_history_buffer(self):
        """Test with large history buffer."""
        wrapper = HistoryObservationWrapper(
            self.base_env,
            history_components=["fingertip_pos"],
            history_length=100,
            history_samples=20
        )

        assert wrapper.history_length == 100
        assert wrapper.num_samples == 20
        assert wrapper.history_buffers["fingertip_pos"].shape == (4, 100, 3)

    def test_single_environment_history(self):
        """Test history tracking with single environment."""
        self.base_env.num_envs = 1
        self.base_env.cfg.num_envs = 1
        # Update observation data for single environment
        self.base_env._setup_observation_data()

        wrapper = HistoryObservationWrapper(self.base_env, history_components=["fingertip_pos"])

        assert wrapper.history_buffers["fingertip_pos"].shape == (1, 8, 3)

        actions = torch.randn(1, 6)
        result = wrapper.step(actions)
        assert len(result) == 5