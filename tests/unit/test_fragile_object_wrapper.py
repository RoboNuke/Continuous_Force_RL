"""
Unit tests for FragileObjectWrapper.

This module tests the FragileObjectWrapper functionality including break force
configuration, multi-agent support, and force violation detection.
"""

import pytest
import torch
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from wrappers.mechanics.fragile_object_wrapper import FragileObjectWrapper
from tests.mocks.mock_isaac_lab import create_mock_env


class TestFragileObjectWrapperInitialization:
    """Test FragileObjectWrapper initialization and configuration."""

    def test_initialization_single_break_force(self, mock_env):
        """Test initialization with single break force value."""
        wrapper = FragileObjectWrapper(mock_env, break_force=50.0, num_agents=1)

        assert wrapper.num_agents == 1
        assert wrapper.num_envs == 64
        assert wrapper.envs_per_agent == 64
        assert wrapper.break_force.shape == (64,)
        assert torch.allclose(wrapper.break_force, torch.tensor([50.0] * 64))
        assert wrapper.is_fragile().item() == True

    def test_initialization_multi_agent_break_forces(self, mock_env):
        """Test initialization with per-agent break forces."""
        break_forces = [30.0, 50.0, 70.0, -1]  # 4 agents, last one unbreakable
        wrapper = FragileObjectWrapper(mock_env, break_force=break_forces, num_agents=4)

        assert wrapper.num_agents == 4
        assert wrapper.envs_per_agent == 16  # 64/4
        assert wrapper.break_force.shape == (64,)

        # Check break forces per agent
        assert torch.allclose(wrapper.break_force[0:16], torch.tensor([30.0] * 16))
        assert torch.allclose(wrapper.break_force[16:32], torch.tensor([50.0] * 16))
        assert torch.allclose(wrapper.break_force[32:48], torch.tensor([70.0] * 16))
        assert torch.all(wrapper.break_force[48:64] > 1e6)  # Very high (unbreakable)

    def test_initialization_unbreakable_objects(self, mock_env):
        """Test initialization with all unbreakable objects."""
        wrapper = FragileObjectWrapper(mock_env, break_force=[-1, -1], num_agents=2)

        assert wrapper.is_fragile().item() == False
        assert torch.all(wrapper.break_force > 1e6)

    def test_initialization_invalid_agent_count(self, mock_env):
        """Test initialization fails with invalid agent count."""
        with pytest.raises(ValueError, match="must be divisible by number of agents"):
            FragileObjectWrapper(mock_env, break_force=50.0, num_agents=7)  # 64 not divisible by 7

    def test_initialization_mismatched_break_force_list(self, mock_env):
        """Test initialization fails with wrong break force list length."""
        with pytest.raises(ValueError, match="Break force list length .* must match number of agents"):
            FragileObjectWrapper(mock_env, break_force=[30.0, 50.0], num_agents=4)  # 64 divisible by 4


class TestFragileObjectWrapperAgentAssignment:
    """Test agent assignment functionality."""

    def test_get_agent_assignment_single_agent(self, mock_env):
        """Test agent assignment with single agent."""
        wrapper = FragileObjectWrapper(mock_env, break_force=50.0, num_agents=1)
        assignment = wrapper.get_agent_assignment()

        assert len(assignment) == 1
        assert assignment[0] == list(range(64))

    def test_get_agent_assignment_multi_agent(self, mock_env):
        """Test agent assignment with multiple agents."""
        wrapper = FragileObjectWrapper(mock_env, break_force=[30.0, 50.0, 70.0, 90.0], num_agents=4)
        assignment = wrapper.get_agent_assignment()

        assert len(assignment) == 4
        assert assignment[0] == list(range(0, 16))
        assert assignment[1] == list(range(16, 32))
        assert assignment[2] == list(range(32, 48))
        assert assignment[3] == list(range(48, 64))

    def test_get_break_forces(self, mock_env):
        """Test get_break_forces returns tensor clone."""
        wrapper = FragileObjectWrapper(mock_env, break_force=50.0, num_agents=1)
        forces = wrapper.get_break_forces()

        assert forces.shape == (64,)
        assert torch.allclose(forces, torch.tensor([50.0] * 64))
        # Verify it's a clone, not the same tensor
        forces[0] = 999.0
        assert wrapper.break_force[0] == 50.0

    def test_get_agent_break_force(self, mock_env):
        """Test get_agent_break_force for specific agents."""
        wrapper = FragileObjectWrapper(mock_env, break_force=[30.0, 50.0], num_agents=2)

        agent0_forces = wrapper.get_agent_break_force(0)
        agent1_forces = wrapper.get_agent_break_force(1)

        assert agent0_forces.shape == (32,)
        assert agent1_forces.shape == (32,)
        assert torch.allclose(agent0_forces, torch.tensor([30.0] * 32))
        assert torch.allclose(agent1_forces, torch.tensor([50.0] * 32))

    def test_get_agent_break_force_invalid_id(self, mock_env):
        """Test get_agent_break_force with invalid agent ID."""
        wrapper = FragileObjectWrapper(mock_env, break_force=50.0, num_agents=2)

        with pytest.raises(ValueError, match="must be less than number of agents"):
            wrapper.get_agent_break_force(2)


class TestFragileObjectWrapperForceViolations:
    """Test force violation detection."""

    def test_get_force_violations_with_violations(self, mock_env):
        """Test force violation detection with violations present."""
        wrapper = FragileObjectWrapper(mock_env, break_force=10.0, num_agents=1)

        # Mock force-torque data with some violations
        force_data = torch.zeros(64, 6, device=mock_env.device)
        force_data[0, :3] = torch.tensor([15.0, 0.0, 0.0])  # Violation: |15| > 10
        force_data[1, :3] = torch.tensor([5.0, 5.0, 5.0])   # No violation: |8.66| < 10
        force_data[2, :3] = torch.tensor([8.0, 6.0, 0.0])   # Violation: |10| >= 10
        wrapper.unwrapped.robot_force_torque = force_data

        violations = wrapper.get_force_violations()

        assert violations.shape == (64,)
        assert violations[0] == True   # 15.0 > 10.0
        assert violations[1] == False  # sqrt(5²+5²+5²) = 8.66 < 10.0
        assert violations[2] == True   # sqrt(8²+6²) = 10.0 >= 10.0

    def test_get_force_violations_no_violations(self, mock_env):
        """Test force violation detection with no violations."""
        wrapper = FragileObjectWrapper(mock_env, break_force=50.0, num_agents=1)

        # Mock force-torque data below threshold
        force_data = torch.ones(64, 6, device=mock_env.device) * 5.0  # All forces = 5.0
        wrapper.unwrapped.robot_force_torque = force_data

        violations = wrapper.get_force_violations()

        assert violations.shape == (64,)
        assert not violations.any()  # No violations

    def test_get_force_violations_without_force_data(self, mock_env):
        """Test force violation detection without force-torque data."""
        wrapper = FragileObjectWrapper(mock_env, break_force=10.0, num_agents=1)
        # Don't add robot_force_torque attribute

        violations = wrapper.get_force_violations()

        assert violations.shape == (64,)
        assert not violations.any()  # No violations when no data

    def test_get_force_violations_unbreakable_objects(self, mock_env):
        """Test force violation detection with unbreakable objects."""
        wrapper = FragileObjectWrapper(mock_env, break_force=[-1], num_agents=1)  # Use list format for proper -1 handling

        # Mock very high forces
        force_data = torch.ones(64, 6, device=mock_env.device) * 1000.0
        wrapper.unwrapped.robot_force_torque = force_data

        violations = wrapper.get_force_violations()

        assert violations.shape == (64,)
        # Since wrapper.fragile should be False for unbreakable objects,
        # no violations should be detected regardless of force
        assert not violations.any()  # No violations for unbreakable objects


class TestFragileObjectWrapperEpisodeTermination:
    """Test episode termination due to force violations."""

    def test_wrapped_get_dones_with_violations(self, mock_env):
        """Test _wrapped_get_dones includes force violations."""
        # Mock original _get_dones BEFORE creating wrapper
        original_terminated = torch.zeros(64, dtype=torch.bool, device=mock_env.device)
        original_terminated[5] = True  # Environment 5 already terminated
        original_time_out = torch.zeros(64, dtype=torch.bool, device=mock_env.device)

        def mock_get_dones():
            return original_terminated, original_time_out

        mock_env._get_dones = mock_get_dones

        # Create wrapper (this should store the original method)
        wrapper = FragileObjectWrapper(mock_env, break_force=10.0, num_agents=1)

        # Mock force violations
        force_data = torch.zeros(64, 6, device=mock_env.device)
        force_data[0, :3] = torch.tensor([15.0, 0.0, 0.0])  # Force violation
        force_data[5, :3] = torch.tensor([20.0, 0.0, 0.0])  # Force violation + already terminated
        wrapper.unwrapped.robot_force_torque = force_data

        # Initialize wrapper to set up method override (should have been done automatically)
        if not wrapper._wrapper_initialized:
            wrapper._initialize_wrapper()

        terminated, time_out = wrapper.unwrapped._get_dones()

        assert terminated[0] == True   # New force violation
        assert terminated[5] == True   # Already terminated + force violation
        assert terminated[1] == False  # No violation, not already terminated
        assert torch.equal(time_out, original_time_out)  # Time out unchanged

    def test_wrapped_get_dones_without_force_data(self, mock_env):
        """Test _wrapped_get_dones without force-torque data."""
        wrapper = FragileObjectWrapper(mock_env, break_force=10.0, num_agents=1)

        # Mock original _get_dones
        original_terminated = torch.zeros(64, dtype=torch.bool, device=mock_env.device)
        original_time_out = torch.zeros(64, dtype=torch.bool, device=mock_env.device)

        def mock_get_dones():
            return original_terminated, original_time_out

        wrapper.unwrapped._get_dones = mock_get_dones

        # Don't add force-torque data
        wrapper._initialize_wrapper()

        terminated, time_out = wrapper.unwrapped._get_dones()

        # Should return original values since no force checking possible
        assert torch.equal(terminated, original_terminated)
        assert torch.equal(time_out, original_time_out)

    def test_wrapped_get_dones_fallback_no_original(self, mock_env):
        """Test _wrapped_get_dones fallback when no original method exists."""
        wrapper = FragileObjectWrapper(mock_env, break_force=10.0, num_agents=1)

        # Don't set original _get_dones method
        wrapper._initialize_wrapper()

        terminated, time_out = wrapper.unwrapped._get_dones()

        # Should return all False when no original method
        assert terminated.shape == (64,)
        assert time_out.shape == (64,)
        assert not terminated.any()
        assert not time_out.any()


class TestFragileObjectWrapperIntegration:
    """Test wrapper integration and initialization."""

    def test_step_initialization(self, mock_env):
        """Test wrapper initializes during step if needed."""
        wrapper = FragileObjectWrapper(mock_env, break_force=50.0, num_agents=1)

        # Remove _robot to prevent initial initialization
        delattr(wrapper.unwrapped, '_robot')
        wrapper._wrapper_initialized = False

        # Add _robot back and call step
        wrapper.unwrapped._robot = "dummy"

        action = torch.zeros(64, 6)
        obs, reward, terminated, truncated, info = wrapper.step(action)

        assert wrapper._wrapper_initialized is True

    def test_reset_initialization(self, mock_env):
        """Test wrapper initializes during reset if needed."""
        wrapper = FragileObjectWrapper(mock_env, break_force=50.0, num_agents=1)

        # Remove _robot to prevent initial initialization
        delattr(wrapper.unwrapped, '_robot')
        wrapper._wrapper_initialized = False

        # Add _robot back and call reset
        wrapper.unwrapped._robot = "dummy"

        obs, info = wrapper.reset()

        assert wrapper._wrapper_initialized is True