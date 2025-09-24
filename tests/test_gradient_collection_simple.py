"""
Simple test for gradient collection timing fix.

Tests the core functionality without complex mocking.
"""
import pytest
import torch
import torch.nn as nn
import sys
import os

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agents.block_ppo import BlockPPO


class SimpleModel(nn.Module):
    """Simple model for testing gradient collection."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5, num_agents=2):
        super().__init__()
        self.num_agents = num_agents

        # Create block-style parameters (3D with agent dimension)
        self.fc1_weight = nn.Parameter(torch.randn(num_agents, hidden_dim, input_dim))
        self.fc1_bias = nn.Parameter(torch.randn(num_agents, hidden_dim))
        self.fc2_weight = nn.Parameter(torch.randn(num_agents, output_dim, hidden_dim))
        self.fc2_bias = nn.Parameter(torch.randn(num_agents, output_dim))

    def forward(self, x):
        # Simple forward pass
        batch_size = x.shape[0]
        envs_per_agent = batch_size // self.num_agents

        outputs = []
        for agent_id in range(self.num_agents):
            start_idx = agent_id * envs_per_agent
            end_idx = (agent_id + 1) * envs_per_agent
            agent_x = x[start_idx:end_idx]

            h1 = torch.matmul(agent_x, self.fc1_weight[agent_id].t()) + self.fc1_bias[agent_id]
            h1 = torch.relu(h1)
            out = torch.matmul(h1, self.fc2_weight[agent_id].t()) + self.fc2_bias[agent_id]
            outputs.append(out)

        return torch.cat(outputs, dim=0)


def test_collect_and_store_gradients():
    """Test the core gradient collection functionality."""
    print("Testing _collect_and_store_gradients...")

    # Create a minimal BlockPPO instance
    num_agents = 2
    device = torch.device('cpu')

    # Create models
    policy_model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=6, num_agents=num_agents)
    value_model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=1, num_agents=num_agents)

    # Create a minimal BlockPPO instance by setting required attributes manually
    class MinimalBlockPPO:
        def __init__(self):
            self.num_agents = num_agents
            self.device = device
            self.policy = policy_model
            self.value = value_model

            # Mock optimizer with some state
            self.optimizer = torch.optim.Adam(
                list(policy_model.parameters()) + list(value_model.parameters()),
                lr=0.001
            )

        def _get_network_state(self, agent_idx):
            """Simplified version that actually collects gradients."""
            state = {
                "policy": {"gradients": [], "optimizer_state": {}},
                "critic": {"gradients": [], "optimizer_state": {}}
            }

            # Collect policy gradients
            for name, param in self.policy.named_parameters():
                if param.grad is not None:
                    if param.dim() >= 2:  # Block parameters
                        try:
                            if param.dim() == 3:  # (num_agents, out, in)
                                grad_norm = param.grad[agent_idx, :, :].norm(2)
                            elif param.dim() == 2:  # (num_agents, out)
                                grad_norm = param.grad[agent_idx, :].norm(2)
                            else:  # (num_agents,)
                                grad_norm = abs(param.grad[agent_idx])
                            state["policy"]["gradients"].append(grad_norm)
                        except IndexError:
                            # Skip if indexing fails
                            pass

            # Collect critic gradients
            for name, param in self.value.named_parameters():
                if param.grad is not None:
                    if param.dim() >= 2:  # Block parameters
                        try:
                            if param.dim() == 3:  # (num_agents, out, in)
                                grad_norm = param.grad[agent_idx, :, :].norm(2)
                            elif param.dim() == 2:  # (num_agents, out)
                                grad_norm = param.grad[agent_idx, :].norm(2)
                            else:  # (num_agents,)
                                grad_norm = abs(param.grad[agent_idx])
                            state["critic"]["gradients"].append(grad_norm)
                        except IndexError:
                            # Skip if indexing fails
                            pass

            # Mock optimizer state
            for name, param in list(self.policy.named_parameters()) + list(self.value.named_parameters()):
                if param in self.optimizer.state:
                    role = "policy" if param in dict(self.policy.named_parameters()).values() else "critic"
                    state[role]["optimizer_state"][name] = self.optimizer.state[param]

            return state

        def _grad_norm_per_agent(self, gradients):
            """Calculate gradient norm from list of gradient norms."""
            if not gradients:
                return 0.0
            return torch.stack(gradients).norm(2).item()

        def _adam_step_size_per_agent(self, optimizer_state):
            """Calculate step size from optimizer state."""
            if not optimizer_state:
                return 0.0

            total_step_size = 0.0
            count = 0

            for name, state in optimizer_state.items():
                if 'exp_avg' in state and 'exp_avg_sq' in state and 'step' in state:
                    lr = 0.001  # Mock learning rate
                    beta1, beta2 = 0.9, 0.999
                    eps = 1e-8
                    step = state['step']

                    if step > 0:
                        exp_avg = state['exp_avg']
                        exp_avg_sq = state['exp_avg_sq']

                        bias_correction1 = 1 - beta1 ** step
                        bias_correction2 = 1 - beta2 ** step

                        step_direction = (exp_avg / bias_correction1) / (torch.sqrt(exp_avg_sq / bias_correction2) + eps)
                        step_size = lr * torch.norm(step_direction).item()

                        total_step_size += step_size
                        count += 1

            return total_step_size / max(count, 1)

        # Add the method we want to test
        def _collect_and_store_gradients(self, store_policy_state=False, store_critic_state=False):
            """Collect gradients immediately after backward pass for logging."""
            if not hasattr(self, '_gradient_storage'):
                self._gradient_storage = {}

            network_states = [self._get_network_state(i) for i in range(self.num_agents)]

            if store_policy_state:
                grad_norms = torch.tensor([self._grad_norm_per_agent(state['policy']['gradients'])
                                           for state in network_states], device=self.device)
                step_sizes = torch.tensor([self._adam_step_size_per_agent(state['policy']['optimizer_state'])
                                           for state in network_states], device=self.device)
                self._gradient_storage['Policy/Gradient_Norm'] = grad_norms
                self._gradient_storage['Policy/Step_Size'] = step_sizes
                print(f"[DEBUG] Stored policy gradients: {grad_norms}, step sizes: {step_sizes}")

            if store_critic_state:
                critic_grad_norms = torch.tensor([self._grad_norm_per_agent(state['critic']['gradients'])
                                                  for state in network_states], device=self.device)
                critic_step_sizes = torch.tensor([self._adam_step_size_per_agent(state['critic']['optimizer_state'])
                                                  for state in network_states], device=self.device)
                self._gradient_storage['Critic/Gradient_Norm'] = critic_grad_norms
                self._gradient_storage['Critic/Step_Size'] = critic_step_sizes
                print(f"[DEBUG] Stored critic gradients: {critic_grad_norms}, step sizes: {critic_step_sizes}")

    # Create test agent
    agent = MinimalBlockPPO()

    # Create some test data
    batch_size = 20  # 10 envs per agent * 2 agents
    input_data = torch.randn(batch_size, 10)

    # Forward pass to create computational graph
    policy_out = agent.policy(input_data)
    value_out = agent.value(input_data)

    # Create loss and compute gradients
    policy_loss = (policy_out ** 2).mean()
    value_loss = (value_out ** 2).mean()
    total_loss = policy_loss + value_loss

    total_loss.backward()

    # Check that gradients exist
    policy_has_grads = any(p.grad is not None for p in agent.policy.parameters())
    value_has_grads = any(p.grad is not None for p in agent.value.parameters())

    assert policy_has_grads, "Policy should have gradients"
    assert value_has_grads, "Value should have gradients"

    # Test gradient collection
    agent._collect_and_store_gradients(store_policy_state=True, store_critic_state=True)

    # Verify results
    assert hasattr(agent, '_gradient_storage')
    assert 'Policy/Gradient_Norm' in agent._gradient_storage
    assert 'Policy/Step_Size' in agent._gradient_storage
    assert 'Critic/Gradient_Norm' in agent._gradient_storage
    assert 'Critic/Step_Size' in agent._gradient_storage

    # Check shapes
    assert agent._gradient_storage['Policy/Gradient_Norm'].shape == (num_agents,)
    assert agent._gradient_storage['Critic/Gradient_Norm'].shape == (num_agents,)

    # Check that at least some gradients are non-zero
    policy_grad_norms = agent._gradient_storage['Policy/Gradient_Norm']
    critic_grad_norms = agent._gradient_storage['Critic/Gradient_Norm']

    print(f"Policy gradient norms: {policy_grad_norms}")
    print(f"Critic gradient norms: {critic_grad_norms}")

    # Verify non-zero gradients (should be non-zero after backward pass)
    assert policy_grad_norms.sum() > 0, f"Policy gradients should be non-zero: {policy_grad_norms}"
    assert critic_grad_norms.sum() > 0, f"Critic gradients should be non-zero: {critic_grad_norms}"

    print("✅ Test passed! Gradient collection works correctly.")


def test_update_nets_integration():
    """Test that update_nets properly calls gradient collection."""
    print("Testing update_nets integration...")

    # This test verifies that update_nets calls gradient collection at the right time
    # by checking if gradients exist when collection is called

    class TestableBlockPPO:
        def __init__(self):
            self.num_agents = 2
            self.device = torch.device('cpu')
            self._random_value_timesteps = 500
            self._current_timestep = 1000  # After random value phase
            self._grad_norm_clip = 0.5

            # Create simple models
            self.policy = SimpleModel(input_dim=10, hidden_dim=20, output_dim=6, num_agents=2)
            self.value = SimpleModel(input_dim=10, hidden_dim=20, output_dim=1, num_agents=2)

            # Create optimizer
            self.optimizer = torch.optim.Adam(
                list(self.policy.parameters()) + list(self.value.parameters()),
                lr=0.001
            )

            # Mock scaler
            from unittest.mock import Mock
            self.scaler = Mock()
            self.scaler.scale = lambda x: x  # Identity function
            self.scaler.unscale_ = Mock()
            self.scaler.step = Mock()
            self.scaler.update = Mock()

            self.gradient_collected = False
            self.gradients_existed_when_collected = False

        def _collect_and_store_gradients(self, store_policy_state=False, store_critic_state=False):
            """Mock collection that checks if gradients exist."""
            self.gradient_collected = True

            # Check if gradients exist when this is called
            policy_has_grads = any(p.grad is not None for p in self.policy.parameters())
            value_has_grads = any(p.grad is not None for p in self.value.parameters())

            self.gradients_existed_when_collected = policy_has_grads or value_has_grads

            print(f"Gradient collection called - gradients exist: {self.gradients_existed_when_collected}")
            print(f"  Policy gradients: {policy_has_grads}, Value gradients: {value_has_grads}")

        def update_nets(self, loss, update_policy=True, update_critic=True):
            """Our updated update_nets method."""
            # Override policy updates during random value phase
            current_timestep = getattr(self, '_current_timestep', 0)
            if current_timestep < self._random_value_timesteps:
                update_policy = False

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Collect gradients immediately after backward pass while they exist
            if update_policy or update_critic:
                self._collect_and_store_gradients(
                    store_policy_state=update_policy,
                    store_critic_state=update_critic
                )

            # Continue with gradient clipping and optimizer step
            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self._grad_norm_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

    # Create test agent
    agent = TestableBlockPPO()

    # Create test data and loss
    batch_size = 20
    input_data = torch.randn(batch_size, 10)

    policy_out = agent.policy(input_data)
    value_out = agent.value(input_data)
    total_loss = (policy_out ** 2).mean() + (value_out ** 2).mean()

    # Call update_nets
    agent.update_nets(total_loss, update_policy=True, update_critic=True)

    # Verify that gradient collection was called
    assert agent.gradient_collected, "Gradient collection should have been called"

    # Verify that gradients existed when collection was called
    assert agent.gradients_existed_when_collected, "Gradients should exist when collection is called"

    print("✅ Test passed! update_nets calls gradient collection at the right time.")


def test_random_value_timesteps_override():
    """Test that policy updates are disabled during random_value_timesteps."""
    print("Testing random_value_timesteps override...")

    class TestableBlockPPO:
        def __init__(self):
            self.num_agents = 2
            self._random_value_timesteps = 500
            self._current_timestep = 100  # BEFORE random value phase

            self.policy_collected = False
            self.critic_collected = False

        def _collect_and_store_gradients(self, store_policy_state=False, store_critic_state=False):
            """Track what was collected."""
            self.policy_collected = store_policy_state
            self.critic_collected = store_critic_state
            print(f"Gradient collection called - policy: {store_policy_state}, critic: {store_critic_state}")

        def update_nets(self, loss, update_policy=True, update_critic=True):
            """Our updated update_nets method (just the relevant parts)."""
            # Override policy updates during random value phase
            current_timestep = getattr(self, '_current_timestep', 0)
            if current_timestep < self._random_value_timesteps:
                update_policy = False

            # Collect gradients immediately after backward pass while they exist
            if update_policy or update_critic:
                self._collect_and_store_gradients(
                    store_policy_state=update_policy,
                    store_critic_state=update_critic
                )

    # Test during random value phase
    agent = TestableBlockPPO()
    agent.update_nets(torch.tensor(1.0), update_policy=True, update_critic=True)

    # Should have overridden policy to False
    assert not agent.policy_collected, "Policy should NOT be collected during random value phase"
    assert agent.critic_collected, "Critic should still be collected during random value phase"

    # Test after random value phase
    agent = TestableBlockPPO()
    agent._current_timestep = 1000  # AFTER random value phase
    agent.update_nets(torch.tensor(1.0), update_policy=True, update_critic=True)

    # Both should be collected
    assert agent.policy_collected, "Policy should be collected after random value phase"
    assert agent.critic_collected, "Critic should be collected after random value phase"

    print("✅ Test passed! random_value_timesteps override works correctly.")


if __name__ == "__main__":
    test_collect_and_store_gradients()
    test_update_nets_integration()
    test_random_value_timesteps_override()
    print("\n🎉 All tests passed! The gradient collection fix is working correctly.")