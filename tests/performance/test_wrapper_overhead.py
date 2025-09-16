"""
Performance tests for wrapper overhead benchmarking.

Tests the performance impact of individual wrappers and wrapper combinations
compared to baseline environment performance.
"""

import pytest
import torch
import time
import psutil
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock Isaac Lab imports before importing wrappers
with patch.dict('sys.modules', {
    'omni.isaac.lab.managers': MagicMock(),
    'omni.isaac.lab.utils.assets': MagicMock(),
    'omni.isaac.lab.utils.configclass': MagicMock(),
    'isaacsim.core.api.robots': MagicMock(),
    'omni.isaac.core.articulations': MagicMock(),
    'wrappers.control.factory_control_utils': MagicMock(),
    'isaacsim.core.utils.torch': MagicMock(),
    'omni.isaac.core.utils.torch': MagicMock(),
}):
    from tests.mocks.mock_isaac_lab import MockEnvironment, MockConfig
    from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper
    from wrappers.mechanics.fragile_object_wrapper import FragileObjectWrapper
    from wrappers.mechanics.efficient_reset_wrapper import EfficientResetWrapper
    from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper
    from wrappers.observations.history_observation_wrapper import HistoryObservationWrapper
    from wrappers.observations.observation_manager_wrapper import ObservationManagerWrapper


class TestWrapperOverhead:
    """Test performance overhead of different wrapper configurations."""

    @pytest.fixture(params=[256, 512, 1024])
    def num_envs(self, request):
        """Test with different environment counts."""
        return request.param

    @pytest.fixture
    def base_env(self, num_envs):
        """Create baseline environment."""
        return MockEnvironment(num_envs=num_envs, device='cpu')

    @pytest.fixture
    def performance_config(self):
        """Configuration for performance tests."""
        return {
            'num_steps': 100,  # Number of steps to benchmark
            'num_resets': 10,  # Number of resets to benchmark
            'warmup_steps': 5,  # Warmup steps before measurement
            'overhead_threshold': 2.0,  # Maximum allowed overhead ratio
        }

    def measure_step_performance(self, env, config):
        """Measure step performance of an environment."""
        # Warmup
        for _ in range(config['warmup_steps']):
            env.reset()
            action = torch.zeros((env.unwrapped.num_envs, 6), dtype=torch.float32)
            env.step(action)

        # Measure step time
        start_time = time.time()
        for _ in range(config['num_steps']):
            action = torch.zeros((env.unwrapped.num_envs, 6), dtype=torch.float32)
            env.step(action)
        step_time = time.time() - start_time

        return step_time / config['num_steps']  # Average time per step

    def measure_reset_performance(self, env, config):
        """Measure reset performance of an environment."""
        # Warmup
        for _ in range(config['warmup_steps']):
            env.reset()

        # Measure reset time
        start_time = time.time()
        for _ in range(config['num_resets']):
            env.reset()
        reset_time = time.time() - start_time

        return reset_time / config['num_resets']  # Average time per reset

    def measure_memory_usage(self):
        """Measure current memory usage."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    def test_baseline_performance(self, base_env, performance_config):
        """Measure baseline environment performance."""
        step_time = self.measure_step_performance(base_env, performance_config)
        reset_time = self.measure_reset_performance(base_env, performance_config)
        memory_usage = self.measure_memory_usage()

        # Store baseline metrics for comparison
        baseline_metrics = {
            'step_time': step_time,
            'reset_time': reset_time,
            'memory_usage': memory_usage,
            'num_envs': base_env.num_envs
        }

        print(f"Baseline ({base_env.num_envs} envs): "
              f"Step: {step_time*1000:.2f}ms, "
              f"Reset: {reset_time*1000:.2f}ms, "
              f"Memory: {memory_usage:.1f}MB")

        # Verify baseline is reasonable
        assert step_time < 0.1, f"Baseline step time too high: {step_time}s"
        assert reset_time < 0.1, f"Baseline reset time too high: {reset_time}s"

        return baseline_metrics

    def test_force_torque_wrapper_overhead(self, base_env, performance_config):
        """Test ForceTorqueWrapper performance overhead."""
        # Create wrapped environment
        wrapped_env = ForceTorqueWrapper(base_env)

        # Measure performance
        step_time = self.measure_step_performance(wrapped_env, performance_config)
        reset_time = self.measure_reset_performance(wrapped_env, performance_config)
        memory_usage = self.measure_memory_usage()

        print(f"ForceTorque ({base_env.num_envs} envs): "
              f"Step: {step_time*1000:.2f}ms, "
              f"Reset: {reset_time*1000:.2f}ms, "
              f"Memory: {memory_usage:.1f}MB")

        # Compare with baseline (assuming baseline was measured)
        # Note: In real tests, you would store baseline metrics and compare
        assert step_time < 0.2, f"ForceTorque step time too high: {step_time}s"
        assert reset_time < 0.2, f"ForceTorque reset time too high: {reset_time}s"

    def test_fragile_object_wrapper_overhead(self, base_env, performance_config):
        """Test FragileObjectWrapper performance overhead."""
        # Create wrapped environment (2 agents for multi-agent testing)
        num_agents = 2
        wrapped_env = FragileObjectWrapper(base_env, break_force=[100.0, 150.0], num_agents=num_agents)

        # Measure performance
        step_time = self.measure_step_performance(wrapped_env, performance_config)
        reset_time = self.measure_reset_performance(wrapped_env, performance_config)
        memory_usage = self.measure_memory_usage()

        print(f"FragileObject ({base_env.num_envs} envs, {num_agents} agents): "
              f"Step: {step_time*1000:.2f}ms, "
              f"Reset: {reset_time*1000:.2f}ms, "
              f"Memory: {memory_usage:.1f}MB")

        assert step_time < 0.2, f"FragileObject step time too high: {step_time}s"
        assert reset_time < 0.2, f"FragileObject reset time too high: {reset_time}s"

    def test_efficient_reset_wrapper_overhead(self, base_env, performance_config):
        """Test EfficientResetWrapper performance overhead."""
        wrapped_env = EfficientResetWrapper(base_env)

        # Measure performance
        step_time = self.measure_step_performance(wrapped_env, performance_config)
        reset_time = self.measure_reset_performance(wrapped_env, performance_config)
        memory_usage = self.measure_memory_usage()

        print(f"EfficientReset ({base_env.num_envs} envs): "
              f"Step: {step_time*1000:.2f}ms, "
              f"Reset: {reset_time*1000:.2f}ms, "
              f"Memory: {memory_usage:.1f}MB")

        assert step_time < 0.2, f"EfficientReset step time too high: {step_time}s"
        assert reset_time < 0.2, f"EfficientReset reset time too high: {reset_time}s"

    def test_factory_metrics_wrapper_overhead(self, base_env, performance_config):
        """Test FactoryMetricsWrapper performance overhead."""
        num_agents = 2
        wrapped_env = FactoryMetricsWrapper(base_env, num_agents=num_agents)

        # Measure performance
        step_time = self.measure_step_performance(wrapped_env, performance_config)
        reset_time = self.measure_reset_performance(wrapped_env, performance_config)
        memory_usage = self.measure_memory_usage()

        print(f"FactoryMetrics ({base_env.num_envs} envs, {num_agents} agents): "
              f"Step: {step_time*1000:.2f}ms, "
              f"Reset: {reset_time*1000:.2f}ms, "
              f"Memory: {memory_usage:.1f}MB")

        assert step_time < 0.2, f"FactoryMetrics step time too high: {step_time}s"
        assert reset_time < 0.2, f"FactoryMetrics reset time too high: {reset_time}s"

    def test_history_observation_wrapper_overhead(self, base_env, performance_config):
        """Test HistoryObservationWrapper performance overhead."""
        wrapped_env = HistoryObservationWrapper(base_env, history_length=5)

        # Measure performance
        step_time = self.measure_step_performance(wrapped_env, performance_config)
        reset_time = self.measure_reset_performance(wrapped_env, performance_config)
        memory_usage = self.measure_memory_usage()

        print(f"HistoryObservation ({base_env.num_envs} envs, history=5): "
              f"Step: {step_time*1000:.2f}ms, "
              f"Reset: {reset_time*1000:.2f}ms, "
              f"Memory: {memory_usage:.1f}MB")

        assert step_time < 0.2, f"HistoryObservation step time too high: {step_time}s"
        assert reset_time < 0.2, f"HistoryObservation reset time too high: {reset_time}s"

    def test_observation_manager_wrapper_overhead(self, base_env, performance_config):
        """Test ObservationManagerWrapper performance overhead."""
        wrapped_env = ObservationManagerWrapper(base_env)

        # Measure performance
        step_time = self.measure_step_performance(wrapped_env, performance_config)
        reset_time = self.measure_reset_performance(wrapped_env, performance_config)
        memory_usage = self.measure_memory_usage()

        print(f"ObservationManager ({base_env.num_envs} envs): "
              f"Step: {step_time*1000:.2f}ms, "
              f"Reset: {reset_time*1000:.2f}ms, "
              f"Memory: {memory_usage:.1f}MB")

        assert step_time < 0.2, f"ObservationManager step time too high: {step_time}s"
        assert reset_time < 0.2, f"ObservationManager reset time too high: {reset_time}s"

    def test_lightweight_wrapper_stack_overhead(self, base_env, performance_config):
        """Test lightweight wrapper stack performance."""
        # Create a lightweight stack
        env = ForceTorqueWrapper(base_env)
        env = EfficientResetWrapper(env)

        # Measure performance
        step_time = self.measure_step_performance(env, performance_config)
        reset_time = self.measure_reset_performance(env, performance_config)
        memory_usage = self.measure_memory_usage()

        print(f"Lightweight Stack ({base_env.num_envs} envs): "
              f"Step: {step_time*1000:.2f}ms, "
              f"Reset: {reset_time*1000:.2f}ms, "
              f"Memory: {memory_usage:.1f}MB")

        assert step_time < 0.3, f"Lightweight stack step time too high: {step_time}s"
        assert reset_time < 0.3, f"Lightweight stack reset time too high: {reset_time}s"

    def test_full_wrapper_stack_overhead(self, base_env, performance_config):
        """Test full wrapper stack performance."""
        # Create full stack (excluding HybridForcePositionWrapper due to dependencies)
        env = ForceTorqueWrapper(base_env)
        env = FragileObjectWrapper(env, break_force=[100.0, 150.0], num_agents=2)
        env = EfficientResetWrapper(env)
        env = HistoryObservationWrapper(env, history_length=3)
        env = ObservationManagerWrapper(env)
        env = FactoryMetricsWrapper(env, num_agents=2)

        # Measure performance
        step_time = self.measure_step_performance(env, performance_config)
        reset_time = self.measure_reset_performance(env, performance_config)
        memory_usage = self.measure_memory_usage()

        print(f"Full Stack ({base_env.num_envs} envs, 2 agents): "
              f"Step: {step_time*1000:.2f}ms, "
              f"Reset: {reset_time*1000:.2f}ms, "
              f"Memory: {memory_usage:.1f}MB")

        # Full stack should still be reasonable
        assert step_time < 0.5, f"Full stack step time too high: {step_time}s"
        assert reset_time < 0.5, f"Full stack reset time too high: {reset_time}s"

    def test_multi_agent_scaling_performance(self, performance_config):
        """Test performance scaling with different agent counts (as specified in test plan)."""
        agent_configs = [
            (2, 512),   # 2 agents, 256 envs per agent = 512 total
            (3, 768),   # 3 agents, 256 envs per agent = 768 total
            (4, 1024),  # 4 agents, 256 envs per agent = 1024 total
            (5, 1280),  # 5 agents, 256 envs per agent = 1280 total
        ]

        results = []

        for num_agents, total_envs in agent_configs:
            # Create environment with specified agent configuration
            base_env = MockEnvironment(num_envs=total_envs, device='cpu')

            # Create multi-agent wrapper stack
            env = FragileObjectWrapper(base_env, break_force=[100.0] * num_agents, num_agents=num_agents)
            env = FactoryMetricsWrapper(env, num_agents=num_agents)

            # Measure performance
            step_time = self.measure_step_performance(env, performance_config)
            reset_time = self.measure_reset_performance(env, performance_config)
            memory_usage = self.measure_memory_usage()

            result = {
                'num_agents': num_agents,
                'total_envs': total_envs,
                'envs_per_agent': total_envs // num_agents,
                'step_time': step_time,
                'reset_time': reset_time,
                'memory_usage': memory_usage
            }
            results.append(result)

            print(f"Multi-agent ({num_agents} agents, {total_envs} envs): "
                  f"Step: {step_time*1000:.2f}ms, "
                  f"Reset: {reset_time*1000:.2f}ms, "
                  f"Memory: {memory_usage:.1f}MB")

            # Performance should scale reasonably
            assert step_time < 1.0, f"Multi-agent step time too high: {step_time}s"
            assert reset_time < 1.0, f"Multi-agent reset time too high: {reset_time}s"

        # Verify scaling characteristics
        step_times = [r['step_time'] for r in results]

        # Performance shouldn't degrade drastically with more agents
        max_step_time = max(step_times)
        min_step_time = min(step_times)
        scaling_ratio = max_step_time / min_step_time if min_step_time > 0 else 1

        print(f"Multi-agent scaling ratio: {scaling_ratio:.2f}x")
        assert scaling_ratio < 3.0, f"Multi-agent scaling too poor: {scaling_ratio}x"

        return results

    @pytest.mark.benchmark
    def test_wrapper_overhead_summary(self, performance_config):
        """Comprehensive wrapper overhead summary test."""
        print("\n" + "="*80)
        print("WRAPPER PERFORMANCE OVERHEAD SUMMARY")
        print("="*80)

        # Test different environment sizes
        env_sizes = [256, 512, 1024]

        for num_envs in env_sizes:
            print(f"\nEnvironment Size: {num_envs} environments")
            print("-" * 50)

            base_env = MockEnvironment(num_envs=num_envs, device='cpu')

            # Test individual wrappers
            wrappers_to_test = [
                ("Baseline", lambda env: env),
                ("ForceTorque", lambda env: ForceTorqueWrapper(env)),
                ("FragileObject", lambda env: FragileObjectWrapper(env, break_force=100.0, num_agents=1)),
                ("EfficientReset", lambda env: EfficientResetWrapper(env)),
                ("FactoryMetrics", lambda env: FactoryMetricsWrapper(env, num_agents=1)),
                ("HistoryObservation", lambda env: HistoryObservationWrapper(env, history_length=3)),
                ("ObservationManager", lambda env: ObservationManagerWrapper(env)),
            ]

            baseline_step_time = None

            for wrapper_name, wrapper_func in wrappers_to_test:
                env = wrapper_func(base_env)
                step_time = self.measure_step_performance(env, performance_config)
                reset_time = self.measure_reset_performance(env, performance_config)

                if wrapper_name == "Baseline":
                    baseline_step_time = step_time
                    overhead = 1.0
                else:
                    overhead = step_time / baseline_step_time if baseline_step_time > 0 else 1.0

                print(f"{wrapper_name:15s}: Step {step_time*1000:6.2f}ms "
                      f"Reset {reset_time*1000:6.2f}ms "
                      f"Overhead {overhead:4.2f}x")

                # Individual wrapper overhead should be reasonable
                assert overhead < 3.0, f"{wrapper_name} overhead too high: {overhead}x"

        print("\n" + "="*80)