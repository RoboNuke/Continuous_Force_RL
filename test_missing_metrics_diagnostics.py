#!/usr/bin/env python3
"""
Diagnostic test to identify missing metrics issues locally using mocks.
This test simulates the exact conditions causing missing metrics in production.
"""

import pytest
import torch
import gymnasium as gym
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Mock modules before imports
sys.modules['wandb'] = __import__('tests.mocks.mock_wandb', fromlist=[''])
sys.modules['omni.isaac.lab'] = __import__('tests.mocks.mock_isaac_lab', fromlist=[''])
sys.modules['omni.isaac.lab.envs'] = __import__('tests.mocks.mock_isaac_lab', fromlist=['envs'])
sys.modules['omni.isaac.lab.utils'] = __import__('tests.mocks.mock_isaac_lab', fromlist=['utils'])

from agents.block_ppo import BlockPPO
from wrappers.logging.wandb_wrapper import GenericWandbLoggingWrapper
from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper
from tests.mocks.mock_isaac_lab import MockBaseEnv


def create_test_config():
    """Create a test configuration for BlockPPO matching conftest.py."""
    return {
        'ckpt_tracker_path': '/tmp/test_ckpt_tracker.txt',
        'track_ckpts': True,
        'value_update_ratio': 2,
        'use_huber_value_loss': True,
        'random_value_timesteps': 150,
        'agent_0': {
            'experiment': {
                'directory': '/tmp',
                'experiment_name': 'test_experiment'
            }
        },
        'agent_1': {
            'experiment': {
                'directory': '/tmp',
                'experiment_name': 'test_experiment'
            }
        }
    }


class MockExtrasEnv(MockBaseEnv):
    """Mock environment with extras and _get_dones for factory metrics testing."""

    def __init__(self, num_envs=4):
        super().__init__()  # MockBaseEnv doesn't take num_envs parameter
        self.num_envs = num_envs
        self.extras = {
            'current_engagements': torch.zeros(num_envs),
            'current_successes': torch.zeros(num_envs),
            'Episode / successes': torch.ones(num_envs),
            'smoothness': {
                'Smoothness / Sum Square Velocity': torch.randn(num_envs),
                'Smoothness / Max Force': torch.randn(num_envs),
            }
        }

    def _get_dones(self):
        """Mock dones method for factory metrics."""
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        time_out = torch.zeros(self.num_envs, dtype=torch.bool)
        # Simulate one environment timing out occasionally
        time_out[0] = True
        return terminated, time_out


def test_network_states_collection():
    """Test Issue #1: Network states collection for step size/gradient norms."""
    print("\n=== Testing Network States Collection ===")

    # Create test configuration
    config = create_test_config()

    # Create mock models
    mock_models = {
        'policy': MagicMock(),
        'value': MagicMock()
    }
    mock_memory = MagicMock()
    mock_env = MockBaseEnv()
    mock_env.num_envs = 4

    # Create agent
    agent = BlockPPO(
        models=mock_models,
        memory=mock_memory,
        observation_space=gym.spaces.Box(low=-1, high=1, shape=(64,)),
        action_space=gym.spaces.Box(low=-1, high=1, shape=(6,)),
        device=torch.device("cpu"),
        cfg=config,
        num_agents=2,
        env=mock_env
    )

    # Test _get_network_state method
    print("Testing _get_network_state method...")
    network_state = agent._get_network_state(0)
    print(f"Network state returned: {network_state is not None}")
    if network_state:
        print(f"Network state keys: {list(network_state.keys())}")
        print(f"Policy gradients present: {'policy' in network_state and 'gradients' in network_state.get('policy', {})}")
        print(f"Critic gradients present: {'critic' in network_state and 'gradients' in network_state.get('critic', {})}")
    else:
        print("❌ Network state is None - this explains missing step size metrics!")

    return network_state is not None


def test_factory_metrics_episode_completion():
    """Test Issue #2: Factory metrics episode completion detection."""
    print("\n=== Testing Factory Metrics Episode Completion ===")

    # Create environment with extras and _get_dones
    base_env = MockExtrasEnv(num_envs=4)

    # Create factory metrics wrapper
    factory_wrapper = FactoryMetricsWrapper(base_env, num_agents=2)

    print("Testing _get_dones method availability...")
    has_get_dones = hasattr(factory_wrapper.unwrapped, '_get_dones')
    print(f"Environment has _get_dones: {has_get_dones}")

    if has_get_dones:
        try:
            _, time_out = factory_wrapper.unwrapped._get_dones()
            timeout_any = torch.any(time_out)
            print(f"Timeout detection works: {timeout_any}")
            print(f"Timeout count: {time_out.sum().item()}")
        except Exception as e:
            print(f"❌ _get_dones failed: {e}")
            return False
    else:
        print("❌ No _get_dones method - factory metrics won't collect!")
        return False

    # Test metrics collection
    metrics = factory_wrapper._collect_factory_metrics()
    print(f"Factory metrics collected: {len(metrics)} items")
    print(f"Factory metrics keys: {list(metrics.keys())}")

    return len(metrics) > 0


def test_wrapper_chain_access():
    """Test Issue #4: Wrapper chain access between factory and wandb wrappers."""
    print("\n=== Testing Wrapper Chain Access ===")

    # Create environment with proper wrapper chain order
    base_env = MockExtrasEnv(num_envs=4)

    # Create env config for wandb wrapper
    env_cfg = MagicMock()
    env_cfg.agent_configs = {
        'agent_0': {
            'wandb_entity': 'test_entity',
            'wandb_project': 'test_project',
            'wandb_name': 'test_agent_0',
            'wandb_group': 'test_group',
            'wandb_tags': ['test']
        },
        'agent_1': {
            'wandb_entity': 'test_entity',
            'wandb_project': 'test_project',
            'wandb_name': 'test_agent_1',
            'wandb_group': 'test_group',
            'wandb_tags': ['test']
        }
    }

    # Apply wrappers in correct order: Wandb first, then Factory
    print("Applying wandb wrapper first...")
    wandb_wrapper = GenericWandbLoggingWrapper(base_env, num_agents=2, env_cfg=env_cfg)

    print("Applying factory wrapper second...")
    factory_wrapper = FactoryMetricsWrapper(wandb_wrapper, num_agents=2)

    # Test wrapper chain access
    print("Testing wrapper chain access...")
    has_unwrapped_add_metrics = hasattr(factory_wrapper.env.unwrapped, 'add_metrics')
    has_env_add_metrics = hasattr(factory_wrapper.env, 'add_metrics')

    print(f"factory_wrapper.env.unwrapped type: {type(factory_wrapper.env.unwrapped)}")
    print(f"factory_wrapper.env type: {type(factory_wrapper.env)}")
    print(f"env.unwrapped has add_metrics: {has_unwrapped_add_metrics}")
    print(f"env has add_metrics: {has_env_add_metrics}")

    if not has_unwrapped_add_metrics and not has_env_add_metrics:
        print("❌ Factory wrapper cannot access wandb wrapper add_metrics!")
        return False
    else:
        print("✅ Factory wrapper can access wandb wrapper")
        return True


def test_tensor_type_validation():
    """Test Issue #3: Tensor type validation in metrics."""
    print("\n=== Testing Tensor Type Validation ===")

    # Create test metrics with different types
    test_metrics = {
        'correct_tensor': torch.tensor([1.0, 2.0]),
        'list_metric': [1.0, 2.0],  # This should fail
        'numpy_metric': torch.tensor([3.0, 4.0]).numpy(),  # This should fail
        'scalar_metric': 5.0  # This should fail
    }

    from wrappers.logging.wandb_wrapper import SimpleEpisodeTracker

    # Create tracker
    agent_config = {
        'wandb_entity': 'test_entity',
        'wandb_project': 'test_project',
        'wandb_name': 'test_tracker',
        'wandb_group': 'test_group',
        'wandb_tags': ['test']
    }
    env_config = MagicMock()

    tracker = SimpleEpisodeTracker(
        num_envs=2,
        device=torch.device("cpu"),
        agent_config=agent_config,
        env_config=env_config
    )

    # Test each metric type
    for metric_name, metric_value in test_metrics.items():
        try:
            tracker.add_metrics({metric_name: metric_value})
            print(f"✅ {metric_name} ({type(metric_value)}): accepted")
        except TypeError as e:
            print(f"❌ {metric_name} ({type(metric_value)}): rejected - {e}")

    return True


def test_environment_info_extraction():
    """Test Issue #5: Environment info and extras for component rewards."""
    print("\n=== Testing Environment Info/Extras Extraction ===")

    # Create environment with comprehensive extras
    base_env = MockExtrasEnv(num_envs=4)

    # Add component reward info to extras
    base_env.extras.update({
        'reward_components': {
            'distance_reward': torch.randn(4),
            'orientation_reward': torch.randn(4),
            'force_penalty': torch.randn(4),
            'success_bonus': torch.randn(4)
        },
        'current_actions': torch.randn(4, 6),
        'log_probs': torch.randn(4, 6)
    })

    print("Environment extras available:")
    print(f"  Extras keys: {list(base_env.extras.keys())}")

    if 'reward_components' in base_env.extras:
        print(f"  Component rewards: {list(base_env.extras['reward_components'].keys())}")

    if 'smoothness' in base_env.extras:
        print(f"  Smoothness metrics: {list(base_env.extras['smoothness'].keys())}")

    # Simulate environment step to check info
    obs, reward, terminated, truncated, info = base_env.step(torch.randn(4, 6))
    print(f"Step info keys: {list(info.keys()) if info else 'None'}")

    return True


def run_all_diagnostics():
    """Run all diagnostic tests and summarize results."""
    print("🔍 Running Missing Metrics Diagnostics")
    print("=" * 50)

    results = {}

    try:
        results['network_states'] = test_network_states_collection()
    except Exception as e:
        print(f"❌ Network states test failed: {e}")
        results['network_states'] = False

    try:
        results['factory_metrics'] = test_factory_metrics_episode_completion()
    except Exception as e:
        print(f"❌ Factory metrics test failed: {e}")
        results['factory_metrics'] = False

    try:
        results['wrapper_chain'] = test_wrapper_chain_access()
    except Exception as e:
        print(f"❌ Wrapper chain test failed: {e}")
        results['wrapper_chain'] = False

    try:
        results['tensor_validation'] = test_tensor_type_validation()
    except Exception as e:
        print(f"❌ Tensor validation test failed: {e}")
        results['tensor_validation'] = False

    try:
        results['environment_info'] = test_environment_info_extraction()
    except Exception as e:
        print(f"❌ Environment info test failed: {e}")
        results['environment_info'] = False

    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")

    failed_tests = [name for name, passed in results.items() if not passed]
    if failed_tests:
        print(f"\n🚨 LIKELY CAUSES OF MISSING METRICS:")
        for test_name in failed_tests:
            if test_name == 'network_states':
                print("  - Policy/Critic step size & gradient norms missing")
            elif test_name == 'factory_metrics':
                print("  - Factory success/smoothness metrics missing")
            elif test_name == 'wrapper_chain':
                print("  - Factory metrics not reaching wandb wrapper")
            elif test_name == 'environment_info':
                print("  - Component rewards not extracted from environment")
    else:
        print("\n✅ All tests passed - issue may be in production configuration")


if __name__ == "__main__":
    run_all_diagnostics()