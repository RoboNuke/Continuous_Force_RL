"""
Test the simplified gradient collection that sends directly to wrapper.
"""
import torch
import torch.nn as nn
from unittest.mock import Mock
import sys
import os

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


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


def test_simplified_gradient_collection():
    """Test that gradient collection directly calls wrapper.add_metrics()."""
    print("Testing simplified gradient collection...")

    # Create a minimal BlockPPO-like class
    class SimplifiedBlockPPO:
        def __init__(self):
            self.num_agents = 2
            self.device = torch.device('cpu')

            # Create models
            self.policy = SimpleModel(input_dim=10, hidden_dim=20, output_dim=6, num_agents=2)
            self.value = SimpleModel(input_dim=10, hidden_dim=20, output_dim=1, num_agents=2)

            # Create optimizer
            self.optimizer = torch.optim.Adam(
                list(self.policy.parameters()) + list(self.value.parameters()),
                lr=0.001
            )

            # Mock logging wrapper that captures what was sent
            self.wrapper_calls = []
            self.mock_wrapper = Mock()
            self.mock_wrapper.add_metrics = Mock(side_effect=self._capture_wrapper_call)

        def _capture_wrapper_call(self, metrics):
            """Capture what was sent to wrapper."""
            self.wrapper_calls.append(metrics.copy())
            print(f"[CAPTURED] Wrapper received: {list(metrics.keys())}")

        def _get_logging_wrapper(self):
            """Return mock wrapper."""
            return self.mock_wrapper

        def _get_network_state(self, agent_idx):
            """Simplified network state collection."""
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
                            pass

            return state

        def _grad_norm_per_agent(self, gradients):
            """Calculate gradient norm from list of gradient norms."""
            if not gradients:
                return 0.0
            return torch.stack(gradients).norm(2).item()

        def _adam_step_size_per_agent(self, optimizer_state):
            """Calculate step size from optimizer state (simplified)."""
            return 0.001  # Mock step size

        # The simplified collection method (the one we're testing)
        def _collect_and_store_gradients(self, store_policy_state=False, store_critic_state=False):
            """Collect gradients immediately after backward pass and send to logging wrapper."""
            wrapper = self._get_logging_wrapper()
            if not wrapper:
                return

            network_states = [self._get_network_state(i) for i in range(self.num_agents)]
            gradient_metrics = {}

            if store_policy_state:
                grad_norms = torch.tensor([self._grad_norm_per_agent(state['policy']['gradients'])
                                           for state in network_states], device=self.device)
                step_sizes = torch.tensor([self._adam_step_size_per_agent(state['policy']['optimizer_state'])
                                           for state in network_states], device=self.device)
                gradient_metrics['Policy/Gradient_Norm'] = grad_norms
                gradient_metrics['Policy/Step_Size'] = step_sizes
                print(f"[DEBUG] Collected policy gradients: {grad_norms}, step sizes: {step_sizes}")

            if store_critic_state:
                critic_grad_norms = torch.tensor([self._grad_norm_per_agent(state['critic']['gradients'])
                                                  for state in network_states], device=self.device)
                critic_step_sizes = torch.tensor([self._adam_step_size_per_agent(state['critic']['optimizer_state'])
                                                  for state in network_states], device=self.device)
                gradient_metrics['Critic/Gradient_Norm'] = critic_grad_norms
                gradient_metrics['Critic/Step_Size'] = critic_step_sizes
                print(f"[DEBUG] Collected critic gradients: {critic_grad_norms}, step sizes: {critic_step_sizes}")

            if gradient_metrics:
                print(f"[DEBUG] Sending gradient metrics directly to wrapper: {list(gradient_metrics.keys())}")
                wrapper.add_metrics(gradient_metrics)

    # Create test agent
    agent = SimplifiedBlockPPO()

    # Create test data and compute gradients
    batch_size = 20
    input_data = torch.randn(batch_size, 10)

    policy_out = agent.policy(input_data)
    value_out = agent.value(input_data)
    total_loss = (policy_out ** 2).mean() + (value_out ** 2).mean()

    total_loss.backward()

    # Test the simplified gradient collection
    agent._collect_and_store_gradients(store_policy_state=True, store_critic_state=True)

    # Verify results
    assert len(agent.wrapper_calls) == 1, f"Expected 1 wrapper call, got {len(agent.wrapper_calls)}"

    captured_metrics = agent.wrapper_calls[0]

    # Check that the right metrics were sent
    expected_keys = {'Policy/Gradient_Norm', 'Policy/Step_Size', 'Critic/Gradient_Norm', 'Critic/Step_Size'}
    actual_keys = set(captured_metrics.keys())

    assert expected_keys == actual_keys, f"Expected {expected_keys}, got {actual_keys}"

    # Check that gradients are non-zero (since we did backward pass)
    policy_grad_norms = captured_metrics['Policy/Gradient_Norm']
    critic_grad_norms = captured_metrics['Critic/Gradient_Norm']

    assert torch.sum(policy_grad_norms) > 0, f"Policy gradients should be non-zero: {policy_grad_norms}"
    assert torch.sum(critic_grad_norms) > 0, f"Critic gradients should be non-zero: {critic_grad_norms}"

    print("✅ Simplified gradient collection works correctly!")
    print(f"   📊 Captured metrics: {list(captured_metrics.keys())}")
    print(f"   🔥 Policy gradient norms: {policy_grad_norms}")
    print(f"   🔥 Critic gradient norms: {critic_grad_norms}")


def test_no_intermediate_storage():
    """Test that no intermediate storage variables are used."""
    print("Testing that no intermediate storage is used...")

    class TestableBlockPPO:
        def __init__(self):
            self.num_agents = 2
            self.device = torch.device('cpu')
            self.mock_wrapper = Mock()

        def _get_logging_wrapper(self):
            return self.mock_wrapper

        def _get_network_state(self, agent_idx):
            return {
                "policy": {"gradients": [torch.tensor(1.0)], "optimizer_state": {}},
                "critic": {"gradients": [torch.tensor(2.0)], "optimizer_state": {}}
            }

        def _grad_norm_per_agent(self, gradients):
            return 1.0 if gradients else 0.0

        def _adam_step_size_per_agent(self, optimizer_state):
            return 0.001

        # The method we're testing
        def _collect_and_store_gradients(self, store_policy_state=False, store_critic_state=False):
            """Collect gradients immediately after backward pass and send to logging wrapper."""
            wrapper = self._get_logging_wrapper()
            if not wrapper:
                return

            network_states = [self._get_network_state(i) for i in range(self.num_agents)]
            gradient_metrics = {}

            if store_policy_state:
                grad_norms = torch.tensor([self._grad_norm_per_agent(state['policy']['gradients'])
                                           for state in network_states], device=self.device)
                step_sizes = torch.tensor([self._adam_step_size_per_agent(state['policy']['optimizer_state'])
                                           for state in network_states], device=self.device)
                gradient_metrics['Policy/Gradient_Norm'] = grad_norms
                gradient_metrics['Policy/Step_Size'] = step_sizes

            if gradient_metrics:
                wrapper.add_metrics(gradient_metrics)

    agent = TestableBlockPPO()

    # Call the method
    agent._collect_and_store_gradients(store_policy_state=True)

    # Verify no storage attributes were created
    assert not hasattr(agent, '_gradient_storage'), "Should not create _gradient_storage attribute"

    # Verify wrapper was called directly
    assert agent.mock_wrapper.add_metrics.called, "Wrapper should be called directly"

    print("✅ No intermediate storage used - metrics sent directly to wrapper!")


if __name__ == "__main__":
    test_simplified_gradient_collection()
    test_no_intermediate_storage()
    print("\n🎉 All simplified gradient collection tests passed!")