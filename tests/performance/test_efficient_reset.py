"""
Performance tests for efficient reset functionality.

Tests the performance improvements provided by the EfficientResetWrapper
compared to full environment resets.
"""

import pytest
import torch
import time
import statistics
from unittest.mock import Mock, patch, MagicMock
import sys
import os

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
    from wrappers.mechanics.efficient_reset_wrapper import EfficientResetWrapper


class TestEfficientResetPerformance:
    """Test performance improvements from efficient reset functionality."""

    @pytest.fixture(params=[256, 512, 1024])
    def num_envs(self, request):
        """Test with different environment counts."""
        return request.param

    @pytest.fixture
    def base_env(self, num_envs):
        """Create baseline environment."""
        return MockEnvironment(num_envs=num_envs, device='cpu')

    @pytest.fixture
    def efficient_reset_env(self, base_env):
        """Create environment with efficient reset wrapper."""
        return EfficientResetWrapper(base_env)

    @pytest.fixture
    def performance_config(self):
        """Configuration for performance tests."""
        return {
            'warmup_resets': 5,      # Warmup resets before measurement
            'measurement_resets': 50, # Number of resets to measure
            'cache_setup_resets': 3,  # Initial resets to set up cache
            'improvement_threshold': 1.2,  # Minimum expected improvement ratio
        }

    def measure_reset_times(self, env, num_resets, warmup_resets=0):
        """Measure reset times for an environment."""
        reset_times = []

        # Warmup
        for _ in range(warmup_resets):
            env.reset()

        # Measure individual reset times
        for _ in range(num_resets):
            start_time = time.time()
            env.reset()
            reset_time = time.time() - start_time
            reset_times.append(reset_time)

        return reset_times

    def calculate_reset_statistics(self, reset_times):
        """Calculate statistics for reset times."""
        return {
            'mean': statistics.mean(reset_times),
            'median': statistics.median(reset_times),
            'std': statistics.stdev(reset_times) if len(reset_times) > 1 else 0,
            'min': min(reset_times),
            'max': max(reset_times),
            'count': len(reset_times)
        }

    def test_baseline_reset_performance(self, base_env, performance_config):
        """Measure baseline reset performance without efficient reset."""
        reset_times = self.measure_reset_times(
            base_env,
            performance_config['measurement_resets'],
            performance_config['warmup_resets']
        )

        stats = self.calculate_reset_statistics(reset_times)

        print(f"Baseline Reset ({base_env.num_envs} envs): "
              f"Mean: {stats['mean']*1000:.2f}ms, "
              f"Median: {stats['median']*1000:.2f}ms, "
              f"Std: {stats['std']*1000:.2f}ms")

        # Verify baseline performance is reasonable
        assert stats['mean'] < 0.1, f"Baseline reset time too high: {stats['mean']}s"
        assert stats['std'] < 0.05, f"Baseline reset variance too high: {stats['std']}s"

        return stats

    def test_efficient_reset_cold_performance(self, efficient_reset_env, performance_config):
        """Test efficient reset performance before cache is established."""
        # Test cold resets (no cache available)
        reset_times = self.measure_reset_times(
            efficient_reset_env,
            performance_config['cache_setup_resets'],  # Just a few resets to set up cache
            0  # No warmup for cold test
        )

        stats = self.calculate_reset_statistics(reset_times)

        print(f"Efficient Reset Cold ({efficient_reset_env.unwrapped.num_envs} envs): "
              f"Mean: {stats['mean']*1000:.2f}ms, "
              f"Median: {stats['median']*1000:.2f}ms")

        # Cold resets should still be reasonable
        assert stats['mean'] < 0.2, f"Cold efficient reset time too high: {stats['mean']}s"

        # Verify cache is now available
        assert efficient_reset_env.has_cached_state(), "Cache should be available after initial resets"

        return stats

    def test_efficient_reset_warm_performance(self, efficient_reset_env, performance_config):
        """Test efficient reset performance after cache is established."""
        # Set up cache first
        for _ in range(performance_config['cache_setup_resets']):
            efficient_reset_env.reset()

        # Verify cache is available
        assert efficient_reset_env.has_cached_state(), "Cache should be available before warm test"

        # Measure warm reset performance
        reset_times = self.measure_reset_times(
            efficient_reset_env,
            performance_config['measurement_resets'],
            performance_config['warmup_resets']
        )

        stats = self.calculate_reset_statistics(reset_times)

        print(f"Efficient Reset Warm ({efficient_reset_env.unwrapped.num_envs} envs): "
              f"Mean: {stats['mean']*1000:.2f}ms, "
              f"Median: {stats['median']*1000:.2f}ms, "
              f"Std: {stats['std']*1000:.2f}ms")

        # Warm resets should be fast
        assert stats['mean'] < 0.1, f"Warm efficient reset time too high: {stats['mean']}s"
        assert stats['std'] < 0.02, f"Warm reset variance too high: {stats['std']}s"

        return stats

    def test_reset_performance_comparison(self, base_env, efficient_reset_env, performance_config):
        """Compare baseline vs efficient reset performance."""
        # Measure baseline performance
        baseline_times = self.measure_reset_times(
            base_env,
            performance_config['measurement_resets'],
            performance_config['warmup_resets']
        )
        baseline_stats = self.calculate_reset_statistics(baseline_times)

        # Set up efficient reset cache
        for _ in range(performance_config['cache_setup_resets']):
            efficient_reset_env.reset()

        # Measure efficient reset performance
        efficient_times = self.measure_reset_times(
            efficient_reset_env,
            performance_config['measurement_resets'],
            performance_config['warmup_resets']
        )
        efficient_stats = self.calculate_reset_statistics(efficient_times)

        # Calculate improvement
        improvement_ratio = baseline_stats['mean'] / efficient_stats['mean'] if efficient_stats['mean'] > 0 else 1.0

        print(f"\nReset Performance Comparison ({base_env.num_envs} envs):")
        print(f"  Baseline:        {baseline_stats['mean']*1000:6.2f}ms ± {baseline_stats['std']*1000:5.2f}ms")
        print(f"  Efficient Reset: {efficient_stats['mean']*1000:6.2f}ms ± {efficient_stats['std']*1000:5.2f}ms")
        print(f"  Improvement:     {improvement_ratio:6.2f}x")

        # Verify improvement meets threshold
        expected_improvement = performance_config['improvement_threshold']
        assert improvement_ratio >= expected_improvement, \
            f"Efficient reset improvement {improvement_ratio:.2f}x below threshold {expected_improvement}x"

        return {
            'baseline': baseline_stats,
            'efficient': efficient_stats,
            'improvement_ratio': improvement_ratio
        }

    def test_cache_state_management(self, efficient_reset_env, performance_config):
        """Test cache state management functionality."""
        # Initially no cache
        assert not efficient_reset_env.has_cached_state(), "Should not have cache initially"

        # Reset to create cache
        efficient_reset_env.reset()
        assert efficient_reset_env.has_cached_state(), "Should have cache after first reset"

        # Clear cache
        efficient_reset_env.clear_cached_state()
        assert not efficient_reset_env.has_cached_state(), "Should not have cache after clearing"

        # Rebuild cache and verify performance impact
        start_time = time.time()
        efficient_reset_env.reset()  # Should be slower (rebuilding cache)
        rebuild_time = time.time() - start_time

        start_time = time.time()
        efficient_reset_env.reset()  # Should be faster (using cache)
        cached_time = time.time() - start_time

        print(f"Cache rebuild time: {rebuild_time*1000:.2f}ms")
        print(f"Cached reset time:  {cached_time*1000:.2f}ms")

        # Cached reset should be faster than rebuild
        improvement = rebuild_time / cached_time if cached_time > 0 else 1.0
        assert improvement > 1.0, f"Cached reset should be faster than rebuild: {improvement:.2f}x"

    def test_partial_vs_full_reset_performance(self, efficient_reset_env, performance_config):
        """Test performance difference between partial and full resets."""
        # Set up cache
        for _ in range(performance_config['cache_setup_resets']):
            efficient_reset_env.reset()

        # Measure partial reset times (using cache)
        partial_reset_times = []
        for _ in range(performance_config['measurement_resets']):
            start_time = time.time()
            efficient_reset_env.reset()
            reset_time = time.time() - start_time
            partial_reset_times.append(reset_time)

        # Clear cache and measure full reset times
        efficient_reset_env.clear_cached_state()
        full_reset_times = []
        for _ in range(performance_config['measurement_resets']):
            start_time = time.time()
            efficient_reset_env.reset()
            reset_time = time.time() - start_time
            full_reset_times.append(reset_time)

        partial_stats = self.calculate_reset_statistics(partial_reset_times)
        full_stats = self.calculate_reset_statistics(full_reset_times)

        improvement = full_stats['mean'] / partial_stats['mean'] if partial_stats['mean'] > 0 else 1.0

        print(f"\nPartial vs Full Reset ({efficient_reset_env.unwrapped.num_envs} envs):")
        print(f"  Partial (cached): {partial_stats['mean']*1000:6.2f}ms ± {partial_stats['std']*1000:5.2f}ms")
        print(f"  Full (no cache):  {full_stats['mean']*1000:6.2f}ms ± {full_stats['std']*1000:5.2f}ms")
        print(f"  Improvement:      {improvement:6.2f}x")

        assert improvement > 1.0, f"Partial reset should be faster than full reset: {improvement:.2f}x"

        return {
            'partial': partial_stats,
            'full': full_stats,
            'improvement': improvement
        }

    def test_reset_scaling_with_environment_count(self, performance_config):
        """Test how reset performance scales with environment count."""
        env_sizes = [256, 512, 1024, 2048]
        results = []

        for num_envs in env_sizes:
            base_env = MockEnvironment(num_envs=num_envs, device='cpu')
            efficient_env = EfficientResetWrapper(base_env)

            # Set up cache
            for _ in range(performance_config['cache_setup_resets']):
                efficient_env.reset()

            # Measure performance
            reset_times = self.measure_reset_times(
                efficient_env,
                performance_config['measurement_resets'] // 2,  # Fewer resets for large envs
                performance_config['warmup_resets']
            )

            stats = self.calculate_reset_statistics(reset_times)

            result = {
                'num_envs': num_envs,
                'mean_time': stats['mean'],
                'time_per_env': stats['mean'] / num_envs * 1000000,  # microseconds per env
            }
            results.append(result)

            print(f"Scaling ({num_envs:4d} envs): "
                  f"{stats['mean']*1000:6.2f}ms total, "
                  f"{result['time_per_env']:6.2f}μs/env")

        # Verify scaling is reasonable (time per environment shouldn't increase drastically)
        times_per_env = [r['time_per_env'] for r in results]
        max_time_per_env = max(times_per_env)
        min_time_per_env = min(times_per_env)
        scaling_ratio = max_time_per_env / min_time_per_env if min_time_per_env > 0 else 1.0

        print(f"Time per environment scaling ratio: {scaling_ratio:.2f}x")
        assert scaling_ratio < 3.0, f"Reset scaling too poor: {scaling_ratio}x"

        return results

    def test_efficient_reset_cache_effectiveness(self, efficient_reset_env, performance_config):
        """Test cache effectiveness with different reset patterns."""
        # Test cache hit rate and effectiveness

        # Pattern 1: Sequential resets (best case for cache)
        for _ in range(performance_config['cache_setup_resets']):
            efficient_reset_env.reset()

        sequential_times = self.measure_reset_times(
            efficient_reset_env,
            20,  # Smaller sample for pattern test
            2
        )

        # Pattern 2: Reset with cache clearing (worst case)
        cache_clearing_times = []
        for _ in range(20):
            efficient_reset_env.clear_cached_state()
            start_time = time.time()
            efficient_reset_env.reset()
            reset_time = time.time() - start_time
            cache_clearing_times.append(reset_time)

        sequential_stats = self.calculate_reset_statistics(sequential_times)
        clearing_stats = self.calculate_reset_statistics(cache_clearing_times)

        cache_effectiveness = clearing_stats['mean'] / sequential_stats['mean'] if sequential_stats['mean'] > 0 else 1.0

        print(f"\nCache Effectiveness ({efficient_reset_env.unwrapped.num_envs} envs):")
        print(f"  Sequential resets:   {sequential_stats['mean']*1000:6.2f}ms")
        print(f"  Cache clearing:      {clearing_stats['mean']*1000:6.2f}ms")
        print(f"  Cache effectiveness: {cache_effectiveness:6.2f}x")

        assert cache_effectiveness > 1.5, f"Cache effectiveness too low: {cache_effectiveness:.2f}x"

        return {
            'sequential': sequential_stats,
            'clearing': clearing_stats,
            'effectiveness': cache_effectiveness
        }

    @pytest.mark.benchmark
    def test_efficient_reset_comprehensive_benchmark(self, performance_config):
        """Comprehensive benchmark of efficient reset functionality."""
        print("\n" + "="*80)
        print("EFFICIENT RESET PERFORMANCE BENCHMARK")
        print("="*80)

        env_sizes = [256, 512, 1024]
        comprehensive_results = {}

        for num_envs in env_sizes:
            print(f"\nTesting with {num_envs} environments:")
            print("-" * 50)

            base_env = MockEnvironment(num_envs=num_envs, device='cpu')
            efficient_env = EfficientResetWrapper(base_env)

            # Baseline performance
            baseline_times = self.measure_reset_times(base_env, 30, 5)
            baseline_stats = self.calculate_reset_statistics(baseline_times)

            # Cold efficient reset performance
            cold_times = self.measure_reset_times(efficient_env, 3, 0)
            cold_stats = self.calculate_reset_statistics(cold_times)

            # Warm efficient reset performance
            warm_times = self.measure_reset_times(efficient_env, 30, 5)
            warm_stats = self.calculate_reset_statistics(warm_times)

            # Calculate improvements
            cold_improvement = baseline_stats['mean'] / cold_stats['mean'] if cold_stats['mean'] > 0 else 1.0
            warm_improvement = baseline_stats['mean'] / warm_stats['mean'] if warm_stats['mean'] > 0 else 1.0

            results = {
                'num_envs': num_envs,
                'baseline': baseline_stats,
                'cold': cold_stats,
                'warm': warm_stats,
                'cold_improvement': cold_improvement,
                'warm_improvement': warm_improvement
            }

            comprehensive_results[num_envs] = results

            print(f"  Baseline:        {baseline_stats['mean']*1000:6.2f}ms")
            print(f"  Cold efficient:  {cold_stats['mean']*1000:6.2f}ms ({cold_improvement:4.2f}x)")
            print(f"  Warm efficient:  {warm_stats['mean']*1000:6.2f}ms ({warm_improvement:4.2f}x)")

            # Verify improvements meet expectations
            assert warm_improvement >= 1.2, f"Warm improvement {warm_improvement:.2f}x below threshold"

        print("\n" + "="*80)
        return comprehensive_results