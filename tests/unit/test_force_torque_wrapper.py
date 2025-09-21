"""
Unit tests for Force Torque Wrapper.

Tests the ForceTorqueWrapper class that adds force-torque sensor functionality
to environments using Isaac Sim's RobotView for sensor integration.
"""

import pytest
import torch
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Mock Isaac Sim imports before importing wrapper
with patch.dict('sys.modules', {
    'omni': MagicMock(),
    'omni.isaac': MagicMock(),
    'omni.isaac.sensor': MagicMock(),
    'omni.isaac.sensor.utils': MagicMock(),
}):
    # Import the mock and set up RobotView
    from tests.mocks.mock_isaac_lab import MockRobotView, MockBaseEnv, MockEnvConfig

    # Mock the RobotView import in the wrapper module
    import wrappers.sensors.force_torque_wrapper as ft_module
    ft_module.RobotView = MockRobotView

    from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper


class TestForceTorqueWrapperInitialization:
    """Test force torque wrapper initialization."""

    def test_wrapper_initialization_basic(self):
        """Test basic wrapper initialization."""
        env = MockBaseEnv()

        wrapper = ForceTorqueWrapper(env)

        assert wrapper.use_tanh_scaling == False
        assert wrapper.tanh_scale == 0.03
        # Since MockBaseEnv has _robot, initialization happens immediately
        assert wrapper._sensor_initialized

    def test_wrapper_initialization_with_scaling(self):
        """Test wrapper initialization with custom scaling."""
        env = MockBaseEnv()
        tanh_scale = 0.05

        wrapper = ForceTorqueWrapper(env, use_tanh_scaling=True, tanh_scale=tanh_scale)

        assert wrapper.use_tanh_scaling == True
        assert wrapper.tanh_scale == tanh_scale

    def test_wrapper_initialization_triggers_when_robot_exists(self):
        """Test that wrapper initializes when robot attribute exists."""
        env = MockBaseEnv()
        env._robot = True  # Ensure robot exists

        wrapper = ForceTorqueWrapper(env)

        # Should initialize automatically
        assert wrapper._sensor_initialized

    def test_wrapper_initialization_deferred_when_no_robot(self):
        """Test that wrapper defers initialization when no robot attribute."""
        env = MockBaseEnv()
        del env._robot  # Remove robot attribute

        wrapper = ForceTorqueWrapper(env)

        # Should not initialize
        assert not wrapper._sensor_initialized


class TestForceTorqueWrapperSensorSetup:
    """Test force-torque sensor setup."""

    def test_force_torque_sensor_initialization_success(self):
        """Test successful force-torque sensor initialization."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)

        # Initialize wrapper
        wrapper._initialize_wrapper()

        assert wrapper._robot_av is not None
        assert wrapper._sensor_initialized

    def test_force_torque_sensor_initialization_import_error(self):
        """Test behavior when RobotView import fails."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)

        # Temporarily set RobotView to None
        original_robot_view = ft_module.RobotView
        ft_module.RobotView = None

        try:
            with pytest.raises(ImportError, match="Could not import RobotView"):
                wrapper._init_force_torque_sensor()
        finally:
            ft_module.RobotView = original_robot_view

    def test_force_torque_sensor_initialization_exception_handling(self):
        """Test graceful handling of sensor initialization exceptions."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)

        # Mock RobotView to raise exception on initialize
        with patch.object(MockRobotView, 'initialize', side_effect=Exception("Mock sensor error")):
            wrapper._init_force_torque_sensor()

            # Should handle exception gracefully
            assert wrapper._robot_av is None

    def test_force_torque_sensor_config_update(self):
        """Test that force-torque sensor updates configuration."""
        env = MockBaseEnv()
        original_obs_space = env.cfg.observation_space
        original_state_space = env.cfg.state_space

        wrapper = ForceTorqueWrapper(env)
        # Initialization happens automatically for MockBaseEnv

        # Should only add force_torque to component attribute map (not component_dims)
        # Note: component_dims is deprecated - Isaac Lab's OBS_DIM_CFG/STATE_DIM_CFG are the single source of truth
        assert 'force_torque' in env.cfg.component_attr_map
        assert env.cfg.component_attr_map['force_torque'] == 'robot_force_torque'


class TestForceTorqueWrapperMethodOverrides:
    """Test method override functionality."""

    def test_init_tensors_override(self):
        """Test _init_tensors method override."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        # Store original method
        original_method = wrapper._original_init_tensors

        # Call wrapped method
        wrapper.unwrapped._init_tensors()

        # Should call original and have force-torque sensor initialized
        assert wrapper._robot_av is not None

    def test_compute_intermediate_values_override(self):
        """Test _compute_intermediate_values method override."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        # Call wrapped method with dt parameter
        wrapper.unwrapped._compute_intermediate_values(0.02)

        # Should update force-torque data
        assert hasattr(env, 'robot_force_torque')
        assert env.robot_force_torque.shape == (env.num_envs, 6)

    def test_method_override_calls_original(self):
        """Test that wrapped methods call original methods."""
        env = MockBaseEnv()

        # Add mock original methods
        env._init_tensors = Mock()
        env._compute_intermediate_values = Mock()

        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        # Call wrapped methods
        wrapper.unwrapped._init_tensors()
        wrapper.unwrapped._compute_intermediate_values(0.02)

        # Should call original methods
        wrapper._original_init_tensors.assert_called_once()
        wrapper._original_compute_intermediate_values.assert_called_once_with(0.02)


class TestForceTorqueWrapperDataCollection:
    """Test force-torque data collection."""

    def test_force_torque_data_collection(self):
        """Test force-torque data collection from sensor."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        # Simulate compute intermediate values
        wrapper.unwrapped._compute_intermediate_values(0.02)

        # Should have force-torque data
        assert hasattr(env, 'robot_force_torque')
        assert isinstance(env.robot_force_torque, torch.Tensor)
        assert env.robot_force_torque.shape == (env.num_envs, 6)

    def test_force_torque_data_scaling(self):
        """Test force-torque data scaling with tanh."""
        env = MockBaseEnv()
        tanh_scale = 0.05
        wrapper = ForceTorqueWrapper(env, use_tanh_scaling=True, tanh_scale=tanh_scale)
        wrapper._initialize_wrapper()

        # Simulate compute intermediate values
        wrapper.unwrapped._compute_intermediate_values(0.02)

        # Check that scaling is configured
        assert wrapper.use_tanh_scaling == True
        assert wrapper.tanh_scale == tanh_scale

        # Data should be available and properly shaped
        assert env.robot_force_torque.shape == (env.num_envs, 6)

    def test_force_torque_data_without_sensor(self):
        """Test behavior when sensor is not available."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)

        # Don't initialize sensor
        wrapper._robot_av = None

        # Should not crash when calling compute intermediate values
        wrapper._wrapped_compute_intermediate_values(0.02)

    def test_force_torque_data_device_consistency(self):
        """Test that force-torque data is on correct device."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        wrapper.unwrapped._compute_intermediate_values(0.02)

        assert env.robot_force_torque.device == env.device


class TestForceTorqueWrapperIntegration:
    """Test wrapper integration with environment."""

    def test_step_integration(self):
        """Test wrapper integration during environment step."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)

        # Step should trigger initialization if not already done
        action = torch.zeros(env.num_envs, env.action_space.shape[0])
        obs, reward, terminated, truncated, info = wrapper.step(action)

        assert wrapper._sensor_initialized
        assert obs is not None

    def test_reset_integration(self):
        """Test wrapper integration during environment reset."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)

        # Reset should trigger initialization if not already done
        obs, info = wrapper.reset()

        assert wrapper._sensor_initialized
        assert obs is not None

    def test_lazy_initialization_step(self):
        """Test lazy initialization during step."""
        env = MockBaseEnv()
        del env._robot  # Remove robot to prevent immediate initialization
        wrapper = ForceTorqueWrapper(env)

        assert not wrapper._sensor_initialized

        # Add robot back and step
        env._robot = True
        action = torch.zeros(env.num_envs, env.action_space.shape[0])
        wrapper.step(action)

        assert wrapper._sensor_initialized

    def test_lazy_initialization_reset(self):
        """Test lazy initialization during reset."""
        env = MockBaseEnv()
        del env._robot  # Remove robot to prevent immediate initialization
        wrapper = ForceTorqueWrapper(env)

        assert not wrapper._sensor_initialized

        # Add robot back and reset
        env._robot = True
        wrapper.reset()

        assert wrapper._sensor_initialized


class TestForceTorqueWrapperErrorHandling:
    """Test error handling and edge cases."""

    def test_double_initialization_safe(self):
        """Test that double initialization is safe."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)

        # Initialize twice
        wrapper._initialize_wrapper()
        wrapper._initialize_wrapper()

        # Should remain initialized
        assert wrapper._sensor_initialized

    def test_missing_environment_methods(self):
        """Test behavior when environment is missing expected methods."""
        # Create a minimal environment class without the wrapper methods
        import gymnasium as gym_module
        class MinimalEnv(gym_module.Env):
            def __init__(self):
                super().__init__()
                self.num_envs = 4
                self.device = torch.device("cpu")
                self.cfg = MockEnvConfig()
                self.observation_space = gym_module.spaces.Box(low=-1, high=1, shape=(10,))
                self.action_space = gym_module.spaces.Box(low=-1, high=1, shape=(3,))

            def step(self, action):
                return self.observation_space.sample(), 0.0, False, False, {}

            def reset(self, **kwargs):
                return self.observation_space.sample(), {}

        simple_env = MinimalEnv()

        wrapper = ForceTorqueWrapper(simple_env)
        wrapper._initialize_wrapper()

        # Should handle missing methods gracefully
        assert wrapper._original_init_tensors is None
        assert wrapper._original_compute_intermediate_values is None

    def test_sensor_failure_handling(self):
        """Test handling of sensor failures."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        # Mock sensor to fail during operation
        mock_robot_av = Mock()
        mock_robot_av.get_measured_joint_forces.side_effect = Exception("Sensor error")
        wrapper._robot_av = mock_robot_av

        # Should handle sensor errors gracefully and not crash
        wrapper._wrapped_compute_intermediate_values(0.02)

        # Force-torque data should be zero after sensor failure
        assert torch.all(env.robot_force_torque == 0.0)

    def test_configuration_edge_cases(self):
        """Test configuration edge cases."""
        env = MockBaseEnv()

        # Test with missing configuration attributes
        if hasattr(env.cfg, 'observation_space'):
            del env.cfg.observation_space
        if hasattr(env.cfg, 'state_space'):
            del env.cfg.state_space

        wrapper = ForceTorqueWrapper(env)

        # Should handle missing configuration gracefully
        wrapper._initialize_wrapper()


class TestForceTorqueWrapperUtilityMethods:
    """Test force-torque wrapper utility methods."""

    def test_has_force_torque_data(self):
        """Test checking if force-torque data is available."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        # Should have data after initialization
        assert wrapper.has_force_torque_data()

    def test_get_current_force_torque(self):
        """Test getting current force-torque readings."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        # Generate some force-torque data
        wrapper.unwrapped._compute_intermediate_values(0.02)

        current_data = wrapper.get_current_force_torque()

        assert 'current_force' in current_data
        assert 'current_torque' in current_data
        assert current_data['current_force'].shape == (env.num_envs, 3)
        assert current_data['current_torque'].shape == (env.num_envs, 3)

    def test_get_force_torque_observation(self):
        """Test getting force-torque observation data."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        obs = wrapper.get_force_torque_observation()

        assert obs.shape == (env.num_envs, 6)
        assert obs.device == env.device

    def test_get_force_torque_observation_with_scaling(self):
        """Test force-torque observation with tanh scaling."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env, use_tanh_scaling=True, tanh_scale=0.05)
        wrapper._initialize_wrapper()

        obs = wrapper.get_force_torque_observation()

        assert obs.shape == (env.num_envs, 6)
        # With tanh scaling, values should be bounded
        assert torch.all(obs >= -1.0)
        assert torch.all(obs <= 1.0)

    def test_get_force_torque_observation_without_data(self):
        """Test observation when no force-torque data is available."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)

        # Don't initialize, so no data
        if hasattr(env, 'robot_force_torque'):
            del env.robot_force_torque

        obs = wrapper.get_force_torque_observation()

        # Should return zeros
        assert obs.shape == (env.num_envs, 6)
        assert torch.all(obs == 0.0)


class TestForceTorqueWrapperFactoryObservation:
    """Test force torque wrapper factory observation injection."""

    def test_wrapped_get_factory_obs_state_dict_with_original(self):
        """Test wrapped factory obs state dict method with original method."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        # Create mock original method
        def mock_original():
            return {
                'fingertip_pos': torch.randn(env.num_envs, 3),
                'ee_linvel': torch.randn(env.num_envs, 3),
            }, {
                'fingertip_pos': torch.randn(env.num_envs, 3),
                'joint_pos': torch.randn(env.num_envs, 7),
            }

        wrapper._original_get_factory_obs_state_dict = mock_original

        obs_dict, state_dict = wrapper._wrapped_get_factory_obs_state_dict()

        # Check that original observations are preserved
        assert 'fingertip_pos' in obs_dict
        assert 'ee_linvel' in obs_dict
        assert 'fingertip_pos' in state_dict
        assert 'joint_pos' in state_dict

        # Check that force_torque is injected
        assert 'force_torque' in obs_dict
        assert 'force_torque' in state_dict
        assert obs_dict['force_torque'].shape == (env.num_envs, 6)
        assert state_dict['force_torque'].shape == (env.num_envs, 6)

    def test_wrapped_get_factory_obs_state_dict_without_original(self):
        """Test wrapped factory obs state dict method without original method."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        # No original method
        wrapper._original_get_factory_obs_state_dict = None

        obs_dict, state_dict = wrapper._wrapped_get_factory_obs_state_dict()

        # Should only contain force_torque
        assert 'force_torque' in obs_dict
        assert 'force_torque' in state_dict
        assert obs_dict['force_torque'].shape == (env.num_envs, 6)
        assert state_dict['force_torque'].shape == (env.num_envs, 6)

    def test_wrapped_get_factory_obs_state_dict_without_sensor_data(self):
        """Test wrapped method when force-torque sensor data is unavailable."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)

        # Remove robot_force_torque attribute to simulate unavailable sensor data
        if hasattr(env, 'robot_force_torque'):
            delattr(env, 'robot_force_torque')

        def mock_original():
            return {'original_obs': torch.randn(env.num_envs, 3)}, {'original_state': torch.randn(env.num_envs, 3)}

        wrapper._original_get_factory_obs_state_dict = mock_original

        obs_dict, state_dict = wrapper._wrapped_get_factory_obs_state_dict()

        # Should have original data but no force_torque
        assert 'original_obs' in obs_dict
        assert 'original_state' in state_dict
        assert 'force_torque' not in obs_dict
        assert 'force_torque' not in state_dict

    def test_method_override_registration(self):
        """Test that the _get_factory_obs_state_dict method override is properly registered."""
        env = MockBaseEnv()

        # Add _get_factory_obs_state_dict method to mock env
        def mock_factory_obs_state_dict():
            return {}, {}

        env._get_factory_obs_state_dict = mock_factory_obs_state_dict

        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        # Check that original method is stored
        assert wrapper._original_get_factory_obs_state_dict is not None

        # Check that method is overridden
        assert hasattr(env, '_get_factory_obs_state_dict')
        assert env._get_factory_obs_state_dict == wrapper._wrapped_get_factory_obs_state_dict

    def test_fallback_get_observations_override(self):
        """Test fallback _get_observations override when _get_factory_obs_state_dict is not available."""
        env = MockBaseEnv()

        # Remove _get_factory_obs_state_dict method to trigger fallback
        if hasattr(env, '_get_factory_obs_state_dict'):
            delattr(env, '_get_factory_obs_state_dict')

        # Add _get_observations method
        def mock_get_observations():
            return {"policy": torch.randn(env.num_envs, 32), "critic": torch.randn(env.num_envs, 48)}

        env._get_observations = mock_get_observations

        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        # Check that fallback override was used
        assert wrapper._original_get_observations is not None
        assert hasattr(env, '_get_observations')
        assert env._get_observations == wrapper._wrapped_get_observations

    def test_wrapped_get_observations_fallback_keyerror_handling(self):
        """Test that fallback method handles KeyError for force_torque."""
        env = MockBaseEnv()

        # Remove _get_factory_obs_state_dict to trigger fallback
        if hasattr(env, '_get_factory_obs_state_dict'):
            delattr(env, '_get_factory_obs_state_dict')

        # Create a mock _get_observations that raises KeyError for force_torque
        def mock_get_observations_with_keyerror():
            raise KeyError("'force_torque'")

        env._get_observations = mock_get_observations_with_keyerror

        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        # The wrapper should catch the KeyError and create observations manually
        result = wrapper._wrapped_get_observations()

        assert isinstance(result, dict)
        assert "policy" in result
        assert "critic" in result
        assert isinstance(result["policy"], torch.Tensor)
        assert isinstance(result["critic"], torch.Tensor)

    def test_create_minimal_observations(self):
        """Test minimal observation creation fallback."""
        env = MockBaseEnv()
        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        result = wrapper._create_minimal_observations()

        assert isinstance(result, dict)
        assert "policy" in result
        assert "critic" in result
        assert isinstance(result["policy"], torch.Tensor)
        assert isinstance(result["critic"], torch.Tensor)

        # Check tensor dimensions
        assert result["policy"].shape[0] == env.num_envs
        assert result["critic"].shape[0] == env.num_envs

        # Should have at least force_torque (6) + prev_actions (6) + fingertip_pos (3) = 15 dimensions
        assert result["policy"].shape[1] >= 15

    def test_create_minimal_observations_without_attributes(self):
        """Test minimal observation creation when environment attributes are missing."""
        env = MockBaseEnv()

        # Remove attributes to test fallback behavior
        delattr(env, 'fingertip_midpoint_pos')
        if hasattr(env, 'actions'):
            delattr(env, 'actions')

        wrapper = ForceTorqueWrapper(env)
        wrapper._initialize_wrapper()

        result = wrapper._create_minimal_observations()

        assert isinstance(result, dict)
        assert "policy" in result
        assert "critic" in result
        assert isinstance(result["policy"], torch.Tensor)
        assert isinstance(result["critic"], torch.Tensor)

        # Check tensor dimensions - should still have force_torque (6) + prev_actions (6) + fingertip_pos (3) = 15
        assert result["policy"].shape[0] == env.num_envs
        assert result["policy"].shape[1] == 15  # Force torque + actions + fingertip pos


if __name__ == "__main__":
    pytest.main([__file__])