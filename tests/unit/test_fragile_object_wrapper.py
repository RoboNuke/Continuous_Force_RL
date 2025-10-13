"""
Unit tests for fragile_object_wrapper.py functionality.
Tests FragileObjectWrapper for force threshold monitoring and multi-agent support.
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

from wrappers.mechanics.fragile_object_wrapper import FragileObjectWrapper
from tests.mocks.mock_isaac_lab import MockBaseEnv


class TestFragileObjectWrapper:
    """Test FragileObjectWrapper functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.base_env = MockBaseEnv()
        self.base_env.num_envs = 4
        self.base_env.device = torch.device("cpu")

    def test_initialization_single_break_force(self):
        """Test wrapper initialization with single break force."""
        break_force = 10.0
        wrapper = FragileObjectWrapper(self.base_env, break_force)

        assert wrapper.num_agents == 1
        assert wrapper.num_envs == 4
        assert wrapper.envs_per_agent == 4
        assert torch.all(wrapper.break_force == break_force)
        assert wrapper.fragile == True

    def test_initialization_multiple_agents(self):
        """Test wrapper initialization with multiple agents."""
        break_forces = [5.0, 10.0]
        wrapper = FragileObjectWrapper(self.base_env, break_forces, num_agents=2)

        assert wrapper.num_agents == 2
        assert wrapper.envs_per_agent == 2

        # Check break forces are assigned correctly per agent
        assert torch.all(wrapper.break_force[0:2] == 5.0)  # Agent 0
        assert torch.all(wrapper.break_force[2:4] == 10.0)  # Agent 1

    def test_initialization_unbreakable_objects(self):
        """Test initialization with unbreakable objects (-1 force)."""
        break_force = -1
        wrapper = FragileObjectWrapper(self.base_env, break_force)

        # Should set very high threshold
        assert torch.all(wrapper.break_force == 2**23)
        assert wrapper.fragile == False  # Should be False for unbreakable

    def test_initialization_mixed_breakable_unbreakable(self):
        """Test initialization with mixed breakable and unbreakable objects."""
        break_forces = [5.0, -1]  # Agent 0: breakable, Agent 1: unbreakable
        wrapper = FragileObjectWrapper(self.base_env, break_forces, num_agents=2)

        assert torch.all(wrapper.break_force[0:2] == 5.0)  # Agent 0: breakable
        assert torch.all(wrapper.break_force[2:4] == 2**23)  # Agent 1: unbreakable
        assert wrapper.fragile == True  # Should be True because some are breakable

    def test_initialization_validation_errors(self):
        """Test initialization validation errors."""
        # Test non-divisible environments
        with pytest.raises(ValueError, match="Number of environments .* must be divisible"):
            FragileObjectWrapper(self.base_env, 10.0, num_agents=3)

        # Test mismatched break force list length
        with pytest.raises(ValueError, match="Break force list length .* must match"):
            FragileObjectWrapper(self.base_env, [5.0, 10.0, 15.0], num_agents=2)

    def test_wrapper_initialization_with_robot(self):
        """Test wrapper initialization when robot is available."""
        wrapper = FragileObjectWrapper(self.base_env, 10.0)

        # MockBaseEnv has _robot, so wrapper should be initialized
        assert wrapper._wrapper_initialized == True
        assert wrapper._original_get_dones is not None

    def test_wrapper_initialization_without_robot(self):
        """Test wrapper initialization without robot."""
        # Create environment without _robot
        env_no_robot = MockBaseEnv()
        delattr(env_no_robot, '_robot')

        wrapper = FragileObjectWrapper(env_no_robot, 10.0)

        assert wrapper._wrapper_initialized == False

    def test_lazy_wrapper_initialization(self):
        """Test lazy wrapper initialization during step/reset."""
        # Create environment without robot initially
        env_no_robot = MockBaseEnv()
        delattr(env_no_robot, '_robot')

        wrapper = FragileObjectWrapper(env_no_robot, 10.0)
        assert wrapper._wrapper_initialized == False

        # Add robot and call step
        env_no_robot._robot = True
        wrapper.step(torch.randn(4, 6))

        assert wrapper._wrapper_initialized == True

    def test_wrapped_get_dones_no_violations(self):
        """Test _wrapped_get_dones with no force violations."""
        wrapper = FragileObjectWrapper(self.base_env, 100.0)  # High threshold

        # Set low force values
        wrapper.unwrapped.robot_force_torque = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # Low force
            [2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
            [1.5, 1.5, 1.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        ], device=self.base_env.device)

        terminated, time_out = wrapper._wrapped_get_dones()

        # No environments should be terminated due to force violations
        assert torch.all(~terminated)

    def test_wrapped_get_dones_with_violations(self):
        """Test _wrapped_get_dones with force violations."""
        wrapper = FragileObjectWrapper(self.base_env, 5.0)  # Low threshold

        # Set high force values that exceed threshold
        wrapper.unwrapped.robot_force_torque = torch.tensor([
            [3.0, 4.0, 0.0, 0.0, 0.0, 0.0],  # Force magnitude = 5.0 (at threshold)
            [6.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Force magnitude = 6.0 (exceeds)
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # Force magnitude ~1.73 (below)
            [4.0, 3.0, 0.0, 0.0, 0.0, 0.0]   # Force magnitude = 5.0 (at threshold)
        ], device=self.base_env.device)

        terminated, time_out = wrapper._wrapped_get_dones()

        # Check force magnitudes
        force_magnitude = torch.linalg.norm(wrapper.unwrapped.robot_force_torque[:, :3], axis=1)
        expected_violations = force_magnitude >= wrapper.break_force

        # Environments with force violations should be terminated
        assert torch.equal(terminated, expected_violations)

    def test_wrapped_get_dones_unbreakable_objects(self):
        """Test _wrapped_get_dones with unbreakable objects."""
        wrapper = FragileObjectWrapper(self.base_env, -1)  # Unbreakable

        # Set extremely high force values
        wrapper.unwrapped.robot_force_torque = torch.tensor([
            [1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0],
            [2000.0, 2000.0, 2000.0, 0.0, 0.0, 0.0],
            [5000.0, 5000.0, 5000.0, 0.0, 0.0, 0.0],
            [10000.0, 10000.0, 10000.0, 0.0, 0.0, 0.0]
        ], device=self.base_env.device)

        terminated, time_out = wrapper._wrapped_get_dones()

        # No environments should be terminated (unbreakable)
        assert torch.all(~terminated)

    def test_wrapped_get_dones_with_original_termination(self):
        """Test _wrapped_get_dones preserves original termination conditions."""
        wrapper = FragileObjectWrapper(self.base_env, 10.0)

        # Mock original get_dones to return some terminations
        def mock_original_get_dones():
            terminated = torch.tensor([True, False, False, True], dtype=torch.bool, device=self.base_env.device)
            time_out = torch.tensor([False, False, True, False], dtype=torch.bool, device=self.base_env.device)
            return terminated, time_out

        wrapper._original_get_dones = mock_original_get_dones

        # Set low forces (no violations)
        wrapper.unwrapped.robot_force_torque = torch.zeros(4, 6, device=self.base_env.device)

        terminated, time_out = wrapper._wrapped_get_dones()

        # Should preserve original termination conditions
        expected_terminated = torch.tensor([True, False, False, True], dtype=torch.bool, device=self.base_env.device)
        expected_time_out = torch.tensor([False, False, True, False], dtype=torch.bool, device=self.base_env.device)

        assert torch.equal(terminated, expected_terminated)
        assert torch.equal(time_out, expected_time_out)

    def test_wrapped_get_dones_combined_termination(self):
        """Test _wrapped_get_dones combines original and force terminations."""
        wrapper = FragileObjectWrapper(self.base_env, 5.0)

        # Mock original get_dones
        def mock_original_get_dones():
            terminated = torch.tensor([True, False, False, False], dtype=torch.bool, device=self.base_env.device)
            time_out = torch.tensor([False, False, False, False], dtype=torch.bool, device=self.base_env.device)
            return terminated, time_out

        wrapper._original_get_dones = mock_original_get_dones

        # Set forces that cause violations in env 1 and 3
        wrapper.unwrapped.robot_force_torque = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # Low force
            [6.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Force violation
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # Low force
            [5.0, 1.0, 0.0, 0.0, 0.0, 0.0]   # Force violation
        ], device=self.base_env.device)

        terminated, time_out = wrapper._wrapped_get_dones()

        # Should combine: env 0 (original), env 1 (force), env 3 (force)
        expected_terminated = torch.tensor([True, True, False, True], dtype=torch.bool, device=self.base_env.device)
        assert torch.equal(terminated, expected_terminated)

    def test_has_force_torque_data(self):
        """Test _has_force_torque_data detection."""
        wrapper = FragileObjectWrapper(self.base_env, 10.0)

        # MockBaseEnv has robot_force_torque
        assert wrapper._has_force_torque_data() == True

        # Remove force-torque data
        delattr(wrapper.unwrapped, 'robot_force_torque')
        assert wrapper._has_force_torque_data() == False

    def test_wrapped_get_dones_without_force_data(self):
        """Test _wrapped_get_dones without force-torque data."""
        wrapper = FragileObjectWrapper(self.base_env, 10.0)

        # Remove force-torque data
        delattr(wrapper.unwrapped, 'robot_force_torque')

        # Should not raise errors and should only use original termination
        terminated, time_out = wrapper._wrapped_get_dones()

        assert terminated.shape == (4,)
        assert time_out.shape == (4,)

    def test_get_agent_assignment(self):
        """Test agent assignment functionality."""
        wrapper = FragileObjectWrapper(self.base_env, [5.0, 10.0], num_agents=2)

        assignments = wrapper.get_agent_assignment()

        assert len(assignments) == 2
        assert assignments[0] == [0, 1]  # Agent 0: environments 0-1
        assert assignments[1] == [2, 3]  # Agent 1: environments 2-3

    def test_get_break_forces(self):
        """Test getting break force thresholds."""
        break_forces = [5.0, 10.0]
        wrapper = FragileObjectWrapper(self.base_env, break_forces, num_agents=2)

        forces = wrapper.get_break_forces()

        # Should return a copy
        assert forces is not wrapper.break_force  # Different objects
        assert torch.allclose(forces, wrapper.break_force)   # Same values

    def test_get_agent_break_force(self):
        """Test getting break force for specific agent."""
        break_forces = [5.0, 10.0]
        wrapper = FragileObjectWrapper(self.base_env, break_forces, num_agents=2)

        agent_0_forces = wrapper.get_agent_break_force(0)
        agent_1_forces = wrapper.get_agent_break_force(1)

        assert torch.all(agent_0_forces == 5.0)
        assert torch.all(agent_1_forces == 10.0)
        assert agent_0_forces.shape == (2,)  # 2 envs per agent
        assert agent_1_forces.shape == (2,)

    def test_get_agent_break_force_validation(self):
        """Test agent break force validation."""
        wrapper = FragileObjectWrapper(self.base_env, 10.0, num_agents=1)

        # Valid agent ID
        forces = wrapper.get_agent_break_force(0)
        assert forces.shape == (4,)

        # Invalid agent ID
        with pytest.raises(ValueError, match="Agent ID .* must be less than"):
            wrapper.get_agent_break_force(1)

    def test_is_fragile(self):
        """Test fragile object detection."""
        # Breakable objects
        wrapper_breakable = FragileObjectWrapper(self.base_env, 10.0)
        assert wrapper_breakable.is_fragile() == True

        # Unbreakable objects
        wrapper_unbreakable = FragileObjectWrapper(self.base_env, -1)
        assert wrapper_unbreakable.is_fragile() == False

        # Mixed breakable/unbreakable
        wrapper_mixed = FragileObjectWrapper(self.base_env, [10.0, -1], num_agents=2)
        assert wrapper_mixed.is_fragile() == True

    def test_get_force_violations(self):
        """Test getting current force violations."""
        wrapper = FragileObjectWrapper(self.base_env, 5.0)

        # Set force values
        wrapper.unwrapped.robot_force_torque = torch.tensor([
            [3.0, 4.0, 0.0, 0.0, 0.0, 0.0],  # Force magnitude = 5.0 (at threshold)
            [6.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Force magnitude = 6.0 (exceeds)
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # Force magnitude ~1.73 (below)
            [0.0, 0.0, 4.9, 0.0, 0.0, 0.0]   # Force magnitude = 4.9 (below)
        ], device=self.base_env.device)

        violations = wrapper.get_force_violations()

        # Calculate expected violations
        force_magnitude = torch.linalg.norm(wrapper.unwrapped.robot_force_torque[:, :3], axis=1)
        expected = force_magnitude >= wrapper.break_force

        assert torch.equal(violations, expected)

    def test_get_force_violations_without_force_data(self):
        """Test get_force_violations without force-torque data."""
        wrapper = FragileObjectWrapper(self.base_env, 5.0)

        # Remove force-torque data
        delattr(wrapper.unwrapped, 'robot_force_torque')

        violations = wrapper.get_force_violations()

        # Should return all False
        assert torch.all(~violations)
        assert violations.shape == (4,)

    def test_get_force_violations_unbreakable(self):
        """Test get_force_violations with unbreakable objects."""
        wrapper = FragileObjectWrapper(self.base_env, -1)

        # Set high forces
        wrapper.unwrapped.robot_force_torque = torch.ones(4, 6, device=self.base_env.device) * 1000

        violations = wrapper.get_force_violations()

        # Should return all False (unbreakable)
        assert torch.all(~violations)

    def test_step_functionality(self):
        """Test step method and wrapper initialization."""
        # Create environment without robot initially
        env_no_robot = MockBaseEnv()
        delattr(env_no_robot, '_robot')

        wrapper = FragileObjectWrapper(env_no_robot, 10.0)
        assert not wrapper._wrapper_initialized

        # Add robot and call step
        env_no_robot._robot = True
        actions = torch.randn(4, 6)
        result = wrapper.step(actions)

        assert wrapper._wrapper_initialized
        assert len(result) == 5

    def test_reset_functionality(self):
        """Test reset method and wrapper initialization."""
        # Create environment without robot initially
        env_no_robot = MockBaseEnv()
        delattr(env_no_robot, '_robot')

        wrapper = FragileObjectWrapper(env_no_robot, 10.0)
        assert not wrapper._wrapper_initialized

        # Add robot and call reset
        env_no_robot._robot = True
        result = wrapper.reset()

        assert wrapper._wrapper_initialized
        assert len(result) == 2

    def test_wrapped_get_dones_fallback(self):
        """Test _wrapped_get_dones fallback when original method doesn't exist."""
        # Create environment with proper size
        base_env = MockBaseEnv()
        base_env.num_envs = 4  # Make sure sizes match
        base_env.device = torch.device("cpu")
        # Fix the robot_force_torque size to match num_envs
        base_env.robot_force_torque = torch.randn(4, 6, device=base_env.device)

        wrapper = FragileObjectWrapper(base_env, 10.0)

        # Remove original method
        wrapper._original_get_dones = None

        # Should not raise errors and should use fallback
        terminated, time_out = wrapper._wrapped_get_dones()

        # Should return default values (all False)
        assert torch.all(~terminated)
        assert torch.all(~time_out)

    def test_device_consistency(self):
        """Test that all tensors are on correct device."""
        wrapper = FragileObjectWrapper(self.base_env, 10.0)

        assert wrapper.break_force.device == self.base_env.device

        # Test with different break forces per agent
        wrapper_multi = FragileObjectWrapper(self.base_env, [5.0, 10.0], num_agents=2)
        assert wrapper_multi.break_force.device == self.base_env.device

    def test_close_functionality(self):
        """Test wrapper close method."""
        wrapper = FragileObjectWrapper(self.base_env, 10.0)

        # Should not raise any errors
        wrapper.close()

    def test_unwrapped_property(self):
        """Test unwrapped property access."""
        wrapper = FragileObjectWrapper(self.base_env, 10.0)

        assert wrapper.unwrapped == self.base_env

    def test_wrapper_properties(self):
        """Test that wrapper properly delegates properties."""
        wrapper = FragileObjectWrapper(self.base_env, 10.0)

        assert wrapper.action_space.shape == self.base_env.action_space.shape
        assert wrapper.observation_space.shape == self.base_env.observation_space.shape

    def test_break_force_tensor_properties(self):
        """Test break force tensor properties."""
        wrapper = FragileObjectWrapper(self.base_env, 15.5)

        assert wrapper.break_force.dtype == torch.float32
        assert wrapper.break_force.device == self.base_env.device
        assert wrapper.break_force.shape == (4,)

    def test_fragile_threshold_detection(self):
        """Test fragile threshold detection logic."""
        # Test boundary case
        high_but_breakable = 2**19  # Below threshold
        wrapper_borderline = FragileObjectWrapper(self.base_env, high_but_breakable)
        assert wrapper_borderline.is_fragile() == True

        # Test just above threshold
        unbreakable_threshold = 2**21  # Above threshold
        wrapper_high = FragileObjectWrapper(self.base_env, unbreakable_threshold)
        assert wrapper_high.is_fragile() == False

    def test_multiple_step_calls_with_force_tracking(self):
        """Test multiple step calls with force violation tracking."""
        wrapper = FragileObjectWrapper(self.base_env, 5.0)

        actions = torch.randn(4, 6)

        # Multiple steps should update force-torque data
        for i in range(5):
            result = wrapper.step(actions)
            assert len(result) == 5

            # Force-torque data should be updated each step
            violations = wrapper.get_force_violations()
            assert violations.shape == (4,)

    def test_wrapper_chain_compatibility(self):
        """Test that wrapper works in a chain with other wrappers."""
        # Create a simple wrapper chain
        intermediate_wrapper = gym.Wrapper(self.base_env)
        wrapper = FragileObjectWrapper(intermediate_wrapper, 10.0)

        assert wrapper.unwrapped == self.base_env

        actions = torch.randn(4, 6)
        result = wrapper.step(actions)
        assert len(result) == 5