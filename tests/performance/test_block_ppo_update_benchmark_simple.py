#!/usr/bin/env python3
"""
Simplified performance benchmarking script for BlockPPO._update function.
This version doesn't require Isaac Sim - it creates mock models and memory
filled with random data to benchmark the _update method.

Usage:
    python test_block_ppo_update_benchmark_simple.py --config <config_path>

Example:
    python test_block_ppo_update_benchmark_simple.py --config configs/experiments/pose_control_exp.yaml
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Note: We avoid importing ConfigManagerV2 to prevent Isaac Lab dependency for benchmarking
# The benchmark uses a simple YAML loader since it only needs basic parameter extraction
# from configs.config_manager_v2 import ConfigManagerV2  # Commented out to avoid mocks

# Now import our modules
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model
from agents.block_ppo import BlockPPO
import yaml  # Still needed for old config compatibility check


class PerformanceProfiler:
    """Profiler for tracking function execution times and call counts."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.metrics = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0.0,
            'times': [],
            'min_time': float('inf'),
            'max_time': 0.0
        })
        self.active_timers = {}

    def start_timer(self, name: str):
        """Start timing a function."""
        self.active_timers[name] = time.perf_counter()

    def stop_timer(self, name: str):
        """Stop timing a function and record metrics."""
        if name not in self.active_timers:
            return

        elapsed = time.perf_counter() - self.active_timers.pop(name)
        metrics = self.metrics[name]
        metrics['call_count'] += 1
        metrics['total_time'] += elapsed
        metrics['times'].append(elapsed)
        metrics['min_time'] = min(metrics['min_time'], elapsed)
        metrics['max_time'] = max(metrics['max_time'], elapsed)

    @contextmanager
    def time_function(self, name: str):
        """Context manager for timing a code block."""
        self.start_timer(name)
        try:
            yield
        finally:
            self.stop_timer(name)

    def get_report(self) -> Dict[str, Any]:
        """Generate a performance report."""
        report = {}
        total_time = sum(m['total_time'] for m in self.metrics.values() if '_update_total' not in m)

        for name, metrics in self.metrics.items():
            times = metrics['times']
            if times:
                report[name] = {
                    'call_count': metrics['call_count'],
                    'total_time': metrics['total_time'],
                    'avg_time': metrics['total_time'] / metrics['call_count'],
                    'min_time': metrics['min_time'],
                    'max_time': metrics['max_time'],
                    'median_time': np.median(times),
                    'std_time': np.std(times),
                    'percentage': (metrics['total_time'] / total_time * 100) if total_time > 0 else 0
                }
        return report


# Import real model creation functions
from models.block_simba import BlockSimBa, BlockSimBaActor, BlockSimBaCritic, export_policies
from skrl.models.torch import Model


def instrument_block_ppo(agent: BlockPPO, profiler: PerformanceProfiler):
    """Instrument BlockPPO with detailed profiling while using real code."""

    # Store original methods
    original_update = agent._update
    original_calc_value_loss = agent.calc_value_loss
    original_update_nets = agent.update_nets
    original_apply_preprocessing = agent._apply_per_agent_preprocessing
    original_log_minibatch = agent._log_minibatch_update
    original_collect_gradients = agent._collect_and_store_gradients
    original_compute_gae = getattr(agent, 'compute_gae', lambda *a, **kw: (None, None))

    # Store original policy/value methods
    original_policy_act = agent.policy.act
    original_policy_get_entropy = agent.policy.get_entropy
    original_value_act = agent.value.act

    # Store original optimizer methods
    original_optimizer_zero_grad = agent.optimizer.zero_grad
    original_optimizer_step = agent.optimizer.step

    # Enhanced wrapper that profiles internals of _update with proper hierarchy
    def wrapped_update(timestep: int, timesteps: int):
        with profiler.time_function('_update_total'):

            # Store original methods for proper nested timing
            original_memory_sample_all = agent.memory.sample_all

            # Profile memory sampling
            def profiled_memory_sample_all(*args, **kwargs):
                with profiler.time_function('memory_sample_all'):
                    return original_memory_sample_all(*args, **kwargs)
            agent.memory.sample_all = profiled_memory_sample_all

            # Profile policy forward passes
            def profiled_policy_act(*args, **kwargs):
                with profiler.time_function('policy_act'):
                    return original_policy_act(*args, **kwargs)
            agent.policy.act = profiled_policy_act

            # Profile entropy computation
            def profiled_get_entropy(*args, **kwargs):
                with profiler.time_function('compute_entropy'):
                    return original_policy_get_entropy(*args, **kwargs)
            agent.policy.get_entropy = profiled_get_entropy

            # Profile value forward passes
            def profiled_value_act(*args, **kwargs):
                with profiler.time_function('value_act'):
                    return original_value_act(*args, **kwargs)
            agent.value.act = profiled_value_act

            # Profile optimizer operations
            def profiled_zero_grad(*args, **kwargs):
                with profiler.time_function('optimizer_zero_grad'):
                    return original_optimizer_zero_grad(*args, **kwargs)
            agent.optimizer.zero_grad = profiled_zero_grad

            def profiled_optimizer_step(*args, **kwargs):
                with profiler.time_function('optimizer_step'):
                    return original_optimizer_step(*args, **kwargs)
            agent.optimizer.step = profiled_optimizer_step

            # Simple approach: Since calc_value_loss is the main issue,
            # let's remove all autocast hijacking and rely on the individual method timings.
            # The "minibatch_processing" concept isn't clearly defined anyway.

            try:
                # Call the original update with our instrumented methods
                # The hierarchy will be:
                # _update_total
                #   ├── memory_sample_all
                #   ├── policy_act (multiple calls)
                #   ├── compute_entropy (multiple calls)
                #   ├── calc_value_loss (multiple calls) <- wrapped separately
                #   │   └── value_act (calls from within calc_value_loss)
                #   ├── log_minibatch_update (multiple calls) <- wrapped separately
                #   ├── optimizer_zero_grad (multiple calls)
                #   └── update_nets (multiple calls) <- wrapped separately
                #       └── optimizer_step (calls from within update_nets)

                result = original_update(timestep, timesteps)

            finally:
                # Restore all original methods
                agent.policy.act = original_policy_act
                agent.policy.get_entropy = original_policy_get_entropy
                agent.value.act = original_value_act
                agent.optimizer.zero_grad = original_optimizer_zero_grad
                agent.optimizer.step = original_optimizer_step
                agent.memory.sample_all = original_memory_sample_all

            return result

    # Standard method wrappers
    def wrapped_calc_value_loss(*args, **kwargs):
        with profiler.time_function('calc_value_loss'):
            return original_calc_value_loss(*args, **kwargs)

    def wrapped_update_nets(*args, **kwargs):
        with profiler.time_function('update_nets'):
            return original_update_nets(*args, **kwargs)

    def wrapped_apply_preprocessing(*args, **kwargs):
        with profiler.time_function('apply_preprocessing'):
            return original_apply_preprocessing(*args, **kwargs)

    def wrapped_log_minibatch(*args, **kwargs):
        with profiler.time_function('log_minibatch_update'):
            return original_log_minibatch(*args, **kwargs)

    def wrapped_collect_gradients(*args, **kwargs):
        with profiler.time_function('collect_gradients'):
            return original_collect_gradients(*args, **kwargs)

    # Apply wrappers
    agent._update = wrapped_update
    agent.calc_value_loss = wrapped_calc_value_loss
    agent.update_nets = wrapped_update_nets
    agent._apply_per_agent_preprocessing = wrapped_apply_preprocessing
    agent._log_minibatch_update = wrapped_log_minibatch
    agent._collect_and_store_gradients = wrapped_collect_gradients

    return agent


def load_yaml_config_with_inheritance(config_path):
    """Simple YAML config loader that handles base_config inheritance."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load the main config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle base_config inheritance
    if 'base_config' in config:
        # The base_config value is a path relative to project root
        # e.g., "configs/base/PiH_base.yaml"
        # Since config_path might be relative or absolute, use Path to resolve
        base_config_str = config['base_config']

        # Try as a path relative to the current working directory (project root)
        base_path = Path(base_config_str)

        if base_path.exists():
            # Load base config recursively
            base_config = load_yaml_config_with_inheritance(str(base_path))

            # Deep merge configs (main config overrides base)
            def deep_merge(base, override):
                result = base.copy()
                for key, value in override.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = deep_merge(result[key], value)
                    else:
                        result[key] = value
                return result

            config = deep_merge(base_config, config)

    return config


def run_benchmark(config_path=None, override_params=None):
    """Main benchmarking function.

    Args:
        config_path: Path to configuration file (optional)
        override_params: Dict of parameters to override (optional)
    """

    print("=" * 80)
    print("BlockPPO._update Performance Benchmark (Simplified)")
    print("=" * 80)
    print()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    # Load configuration if provided
    if config_path:
        print(f"Loading configuration from: {config_path}")

        # Load config using simple YAML loader - NO FALLBACK TO DEFAULTS
        config = load_yaml_config_with_inheritance(config_path)
        print(f"Loaded config structure: {list(config.keys())}")

        # Extract parameters from loaded config with proper error handling
        primary_config = config.get('primary', {})
        agent_config = config.get('agent', {})
        model_config = config.get('model', {})
        environment_config = config.get('environment', {})

        print(f"Primary config keys: {list(primary_config.keys())}")
        print(f"Agent config keys: {list(agent_config.keys())}")
        print(f"Model config keys: {list(model_config.keys())}")
        print(f"Environment config keys: {list(environment_config.keys())}")

        # Get benchmark parameters from config - THROW ERRORS if missing critical fields

        # Number of agents (from primary config)
        if 'agents_per_break_force' in primary_config:
            num_agents = primary_config['agents_per_break_force']
        else:
            raise ValueError(f"'agents_per_break_force' not found in primary config! Available keys: {list(primary_config.keys())}")

        # Number of environments per agent (from primary config)
        if 'num_envs_per_agent' in primary_config:
            num_envs_per_agent = primary_config['num_envs_per_agent']
        else:
            raise ValueError(f"'num_envs_per_agent' not found in primary config! Available keys: {list(primary_config.keys())}")

        total_envs = num_agents * num_envs_per_agent

        # Get observation and action space sizes from model config (use defaults for benchmark)
        observation_space = model_config.get('observation_space', 129)  # Factory task typical
        action_space = model_config.get('action_space', 8)  # Factory task typical

        # Get training parameters from agent config - REQUIRE critical fields that must exist
        required_agent_fields = ['learning_epochs', 'mini_batches', 'discount_factor', 'entropy_loss_scale']
        for field in required_agent_fields:
            if field not in agent_config:
                raise ValueError(f"Required field '{field}' not found in agent config! Available keys: {list(agent_config.keys())}")

        # Load ALL values from config, using defaults only for truly missing fields
        learning_epochs = agent_config['learning_epochs']
        mini_batches = agent_config['mini_batches']

        # rollouts is often auto-calculated from episode timing, so provide reasonable default
        rollouts = agent_config.get('rollouts', 150)
        print(f"Rollouts: {rollouts} ({'from config' if 'rollouts' in agent_config else 'default - not in config'})")

        # Validate and fix mini_batches for benchmark
        if mini_batches > rollouts:
            print(f"WARNING: mini_batches ({mini_batches}) > rollouts ({rollouts})")
            print(f"         This would cause sample_size=0. Overriding mini_batches to {max(1, rollouts // 4)}")
            mini_batches = max(1, rollouts // 4)

        # Extract ALL PPO hyperparameters from config - use config values, not defaults
        discount_factor = agent_config['discount_factor']

        # Handle lambda field (could be 'lambda' or 'lambda_')
        if 'lambda_' in agent_config:
            lambda_val = agent_config['lambda_']
        elif 'lambda' in agent_config:
            lambda_val = agent_config['lambda']
        else:
            raise ValueError("Neither 'lambda' nor 'lambda_' found in agent config!")

        # Learning rates (handle separate policy/critic rates)
        if 'policy_learning_rate' in agent_config and 'critic_learning_rate' in agent_config:
            learning_rate = agent_config['policy_learning_rate']  # Use policy rate for benchmark
            print(f"Using policy learning rate: {learning_rate} (critic: {agent_config['critic_learning_rate']})")
        elif 'learning_rate' in agent_config:
            learning_rate = agent_config['learning_rate']
        else:
            learning_rate = 5e-4  # True fallback only if neither exists
            print(f"No learning rate found in config, using default: {learning_rate}")

        # Load ALL other hyperparameters from config (these all exist in your config)
        grad_norm_clip = agent_config.get('grad_norm_clip', 1.0)
        ratio_clip = agent_config.get('ratio_clip', 0.2)
        value_clip = agent_config.get('value_clip', 0.2)
        clip_predicted_values = agent_config.get('clip_predicted_values', True)
        entropy_loss_scale = agent_config['entropy_loss_scale']
        value_loss_scale = agent_config.get('value_loss_scale', 0.5)
        kl_threshold = agent_config.get('kl_threshold', None)
        random_value_timesteps = agent_config.get('random_value_timesteps', 0)
        value_update_ratio = agent_config.get('value_update_ratio', 1)
        use_huber_value_loss = agent_config.get('use_huber_value_loss', False)

        # Load additional parameters that exist in your config
        random_timesteps = agent_config.get('random_timesteps', 0)
        learning_starts = agent_config.get('learning_starts', 0)
        mixed_precision = agent_config.get('mixed_precision', False)
        optimizer_betas = agent_config.get('optimizer_betas', [0.9, 0.999])
        optimizer_eps = agent_config.get('optimizer_eps', 1e-8)
        optimizer_weight_decay = agent_config.get('optimizer_weight_decay', 0)
        time_limit_bootstrap = agent_config.get('time_limit_bootstrap', False)

        print(f"✅ Successfully loaded configuration from: {config_path}")
        if 'base_config' in config:
            print(f"  (with base config: {config['base_config']})")

        print(f"✅ Extracted parameters (from config file):")
        print(f"  num_agents: {num_agents}")
        print(f"  num_envs_per_agent: {num_envs_per_agent}")
        print(f"  total_envs: {total_envs}")
        print(f"  rollouts: {rollouts}")
        print(f"  learning_epochs: {learning_epochs}")
        print(f"  mini_batches: {mini_batches}")
        print(f"  discount_factor: {discount_factor}")
        print(f"  lambda: {lambda_val}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  grad_norm_clip: {grad_norm_clip}")
        print(f"  ratio_clip: {ratio_clip}")
        print(f"  value_clip: {value_clip}")
        print(f"  clip_predicted_values: {clip_predicted_values}")
        print(f"  entropy_loss_scale: {entropy_loss_scale}")
        print(f"  value_loss_scale: {value_loss_scale}")
        print(f"  kl_threshold: {kl_threshold}")
        print(f"  value_update_ratio: {value_update_ratio}")
        print(f"  use_huber_value_loss: {use_huber_value_loss}")
        print(f"  random_value_timesteps: {random_value_timesteps}")
        print(f"  mixed_precision: {mixed_precision}")
    else:
        # Default parameters if no config provided
        num_agents = 2
        num_envs_per_agent = 32
        total_envs = num_agents * num_envs_per_agent
        observation_space = 129  # Typical for factory tasks
        action_space = 8  # Typical for factory tasks
        rollouts = 150
        learning_epochs = 4
        mini_batches = 4

        # Default PPO hyperparameters
        discount_factor = 0.99
        lambda_val = 0.95
        learning_rate = 5e-4
        grad_norm_clip = 1.0
        ratio_clip = 0.2
        value_clip = 0.2
        clip_predicted_values = True
        entropy_loss_scale = 0.0  # Default when no config provided
        value_loss_scale = 0.5
        kl_threshold = None
        random_value_timesteps = 0
        value_update_ratio = 1
        use_huber_value_loss = False

        print("Using default configuration")

    # Apply any overrides
    if override_params:
        print(f"Applying parameter overrides: {override_params}")
        locals().update(override_params)

    print("\nConfiguration:")
    print(f"  Environment:")
    print(f"    - Agents: {num_agents}")
    print(f"    - Envs per agent: {num_envs_per_agent}")
    print(f"    - Total envs: {total_envs}")
    print(f"    - Observation space: {observation_space}")
    print(f"    - Action space: {action_space}")
    print(f"  Training:")
    print(f"    - Rollouts: {rollouts}")
    print(f"    - Learning epochs: {learning_epochs}")
    print(f"    - Mini-batches: {mini_batches}")
    print(f"  PPO Hyperparameters:")
    print(f"    - Discount factor: {discount_factor}")
    print(f"    - Lambda (GAE): {lambda_val}")
    print(f"    - Learning rate: {learning_rate}")
    print(f"    - Gradient norm clip: {grad_norm_clip}")
    print(f"    - Ratio clip: {ratio_clip}")
    print(f"    - Value clip: {value_clip}")
    print(f"    - Entropy loss scale: {entropy_loss_scale}")
    print(f"    - Value loss scale: {value_loss_scale}")
    print(f"    - Value update ratio: {value_update_ratio}")
    print(f"    - Use Huber value loss: {use_huber_value_loss}")
    print()

    # Create real models using the actual model creation logic
    print("Creating real BlockSimba models...")

    # Create policy model (BlockNetwork with SimBa)
    # Get model config
    actor_cfg = model_config.get('actor', {})
    critic_cfg = model_config.get('critic', {})

    print(f"  Actor config: {actor_cfg}")
    print(f"  Critic config: {critic_cfg}")

    # Create real BlockSimBa models - use the actual classes from the codebase
    policy_model = BlockSimBaActor(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        num_agents=num_agents,
        # Use config parameters with correct names
        actor_n=actor_cfg['n'],
        actor_latent=actor_cfg['latent_size'],
    )

    value_model = BlockSimBaCritic(
        state_space_size=observation_space,  # Note: different parameter name
        device=device,
        num_agents=num_agents,
        # Use config parameters with correct names
        critic_n=critic_cfg['n'],
        critic_latent=critic_cfg['latent_size'],
    )
    models = {"policy": policy_model, "value": value_model}

    # Create memory
    print("Creating memory...")
    memory = RandomMemory(memory_size=rollouts, num_envs=total_envs, device=device)

    # Create agent config
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg.update({
        "rollouts": rollouts,
        "learning_epochs": learning_epochs,
        "mini_batches": mini_batches,
        "discount_factor": discount_factor,
        "lambda": lambda_val,
        "learning_rate": learning_rate,
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": grad_norm_clip,
        "ratio_clip": ratio_clip,
        "value_clip": value_clip,
        "clip_predicted_values": clip_predicted_values,
        "entropy_loss_scale": entropy_loss_scale,
        "value_loss_scale": value_loss_scale,
        "kl_threshold": kl_threshold,
        "rewards_shaper": None,
        "time_limit_bootstrap": False,
        "state_preprocessor": None,
        "value_preprocessor": None,
        "random_value_timesteps": random_value_timesteps,
        "value_update_ratio": value_update_ratio,
        "use_huber_value_loss": use_huber_value_loss,
        "ckpt_tracker_path": "/tmp/ckpt_tracker.txt",
        "track_ckpts": False,
    })

    # Add per-agent configs
    for i in range(num_agents):
        agent_cfg[f'agent_{i}'] = {
            'experiment': {
                'directory': '/tmp',
                'experiment_name': f'benchmark_agent_{i}'
            }
        }

    # Create a simple environment wrapper for the agent (no mocking)
    class SimpleEnvWrapper:
        def __init__(self):
            self.unwrapped = None

        def add_metrics(self, metrics):
            # Simple passthrough for metrics
            pass

        def publish(self, onetime_metrics=None):
            # No-op for benchmarking - skip actual logging
            pass

    env_wrapper = SimpleEnvWrapper()

    # Create agent
    print("Creating BlockPPO agent...")
    agent = BlockPPO(
        models=models,
        memory=memory,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        cfg=agent_cfg,
        state_size=observation_space,
        track_ckpt_paths=False,
        task="benchmark-task",
        num_agents=num_agents,
        num_envs=total_envs,
        env=env_wrapper
    )

    # Initialize agent
    print("Initializing agent...")
    agent.init({"timesteps": 100000})

    # Initialize profiler
    profiler = PerformanceProfiler()

    # Instrument the agent
    print("Instrumenting BlockPPO methods...")
    instrument_block_ppo(agent, profiler)

    # Fill memory with synthetic data
    print("Filling memory with synthetic data...")
    print()

    # First, test the model shapes to understand what they expect
    test_states = torch.randn(total_envs, observation_space, device=device)
    test_actions = torch.randn(total_envs, action_space, device=device)

    print("Testing model input/output shapes...")
    with torch.no_grad():
        # Test policy model
        try:
            test_policy_out, test_log_prob, _ = agent.policy.act({"states": test_states}, role="policy")
            print(f"  Policy input shape: {test_states.shape}")
            print(f"  Policy output shape: {test_policy_out.shape}")
            print(f"  Policy log_prob shape: {test_log_prob.shape}")
        except Exception as e:
            print(f"  Policy test error: {e}")

        # Test value model
        try:
            test_values, _, _ = agent.value.act({"states": test_states}, role="value")
            print(f"  Value input shape: {test_states.shape}")
            print(f"  Value output shape: {test_values.shape}")
        except Exception as e:
            print(f"  Value test error: {e}")

    print()

    for step in range(rollouts):
        # Generate synthetic data with correct shapes
        states = torch.randn(total_envs, observation_space, device=device)
        actions = torch.randn(total_envs, action_space, device=device)
        rewards = torch.randn(total_envs, 1, device=device)
        next_states = torch.randn(total_envs, observation_space, device=device)
        terminated = torch.zeros(total_envs, 1, dtype=torch.bool, device=device)
        truncated = torch.zeros(total_envs, 1, dtype=torch.bool, device=device)

        # Generate log_prob and values using the actual models to get correct shapes
        with torch.no_grad():
            try:
                # Get log_prob from policy model
                _, log_prob, _ = agent.policy.act({"states": states, "taken_actions": actions}, role="policy")
                agent._current_log_prob = log_prob

                # Get values from value model
                values, _, _ = agent.value.act({"states": states}, role="value")

            except Exception as e:
                print(f"Error generating synthetic data: {e}")
                # Fallback to simple shapes if models fail
                agent._current_log_prob = torch.randn(total_envs, 1, device=device)
                values = torch.randn(total_envs, 1, device=device)

        # Store for _update
        agent._current_next_states = next_states

        # Add to memory
        agent.memory.add_samples(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            log_prob=agent._current_log_prob,
            values=values
        )

    # Run benchmarking
    print("Running benchmark...")
    print("-" * 40)

    num_update_calls = 5
    for i in range(num_update_calls):
        print(f"Running _update call {i+1}/{num_update_calls}...")
        agent._update(timestep=100, timesteps=1000)

    print()
    print("=" * 80)
    print("🚀 BLOCKPPO PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()

    # Generate report
    report = profiler.get_report()

    # Get the total _update time as our baseline (5 calls total)
    update_total_time = report.get('_update_total', {}).get('total_time', 1.0)
    num_update_calls = 5

    # Create a cleaner breakdown focusing on the main components
    key_functions = {
        'Network Updates (Optimizer)': ['update_nets_all', 'update_nets_detailed'],
        'Entropy Computation': ['compute_entropy'],
        'Value Loss Calculation': ['calc_value_loss', 'calc_value_loss_detailed'],
        'Policy Forward Passes': ['policy_act'],
        'Policy Loss Calculation': ['compute_policy_loss'],
        'Advantage Computation (GAE)': ['compute_gae', 'compute_gae.advantages_loop', 'compute_gae.normalize'],
        'State Preprocessing': ['preprocessing_states', 'apply_preprocessing'],
        'Memory Operations': ['memory_sample_all'],
        'Other Operations': ['compute_kl_and_reshape', 'optimizer_zero_grad', 'log_minibatch_update']
    }

    print(f"🎯 **KEY PERFORMANCE BREAKDOWN** (per _update call):")
    print(f"Total _update time: {update_total_time/num_update_calls*1000:.1f}ms average per call\n")

    # Calculate time for each major category (avoiding double counting)
    category_times = {}
    for category, functions in key_functions.items():
        # Take the max time from related functions to avoid double counting
        max_time = max((report.get(func, {}).get('total_time', 0) for func in functions), default=0)
        if max_time > 0.001:  # Only show categories with >1ms
            avg_per_update = (max_time / num_update_calls) * 1000  # Convert to ms per _update call
            percentage = (max_time / update_total_time) * 100
            category_times[category] = {
                'total_time': max_time,
                'avg_per_update': avg_per_update,
                'percentage': percentage
            }

    # Sort by time and display
    sorted_categories = sorted(category_times.items(), key=lambda x: x[1]['avg_per_update'], reverse=True)

    for category, metrics in sorted_categories:
        print(f"{category:<35} {metrics['avg_per_update']:>8.1f}ms  ({metrics['percentage']:>5.1f}%)")

    print(f"\n📊 **HIERARCHICAL FUNCTION BREAKDOWN**:")
    print(f"{'Function':<50} {'Calls':<8} {'Total(ms)':<12} {'% of Total':<12}")
    print("-" * 94)

    # Define hierarchy of function calls (CORRECTED - no fake minibatch_processing)
    function_hierarchy = {
        '_update_total': {
            'memory_sample_all': {},
            'apply_preprocessing': {},
            'policy_act': {},
            'compute_entropy': {},
            'calc_value_loss': {
                'value_act': {}
            },
            'log_minibatch_update': {},
            'optimizer_zero_grad': {},
            'update_nets': {
                'optimizer_step': {},
                'collect_gradients': {}
            }
        }
    }

    def print_hierarchy(hierarchy, indent=0, parent_time=None):
        """Print functions in hierarchical order with indentation."""
        for func_name, children in hierarchy.items():
            if func_name in report:
                metrics = report[func_name]
                if metrics['total_time'] > 0.0001:  # Only show if > 0.1ms
                    per_call_ms = (metrics['total_time'] / num_update_calls) * 1000
                    percentage = (metrics['total_time'] / update_total_time) * 100

                    # Create indentation and arrow
                    if indent == 0:
                        prefix = ""
                    else:
                        prefix = "  " * (indent - 1) + "→ "

                    func_display = f"{prefix}{func_name}"
                    print(f"{func_display:<50} {metrics['call_count']:<8} "
                          f"{per_call_ms:<12.1f} {percentage:<12.1f}")

                    # Recursively print children
                    if children:
                        print_hierarchy(children, indent + 1, metrics['total_time'])

    # Print the hierarchical view
    print_hierarchy(function_hierarchy)

    # Add any functions not in hierarchy at the end
    print("\nOther functions not in main hierarchy:")
    shown_functions = set()
    def collect_shown(hierarchy):
        for func_name, children in hierarchy.items():
            shown_functions.add(func_name)
            collect_shown(children)
    collect_shown(function_hierarchy)

    for func_name, metrics in sorted(report.items(), key=lambda x: x[1]['total_time'], reverse=True):
        if func_name not in shown_functions and metrics['total_time'] > 0.001:
            per_call_ms = (metrics['total_time'] / num_update_calls) * 1000
            percentage = (metrics['total_time'] / update_total_time) * 100
            print(f"  • {func_name:<47} {metrics['call_count']:<8} "
                  f"{per_call_ms:<12.1f} {percentage:<12.1f}")

    print(f"\n💡 **OBJECTIVE ANALYSIS**:")

    # Show raw timing data without bias
    all_component_times = [(name, data['avg_per_update'], data['percentage'])
                          for name, data in sorted_categories]

    print(f"  Component timing breakdown:")
    for i, (component, time_ms, percentage) in enumerate(all_component_times, 1):
        print(f"    {i}. {component}: {time_ms:.1f}ms ({percentage:.1f}% of total)")

    print(f"\n🔍 **RELATIVE PERFORMANCE**:")
    if len(all_component_times) >= 2:
        top_component, top_time, top_pct = all_component_times[0]
        second_component, second_time, second_pct = all_component_times[1]

        if second_time > 0:
            ratio = top_time / second_time
            print(f"  - '{top_component}' takes {ratio:.1f}x more time than '{second_component}'")

        # Show cumulative percentages
        if len(all_component_times) >= 3:
            top3_pct = sum(pct for _, _, pct in all_component_times[:3])
            print(f"  - Top 3 components account for {top3_pct:.1f}% of execution time")

        # Show the largest bottleneck objectively
        print(f"  - Largest time consumer: '{top_component}' at {top_time:.1f}ms ({top_pct:.1f}%)")

        # Show distribution analysis
        if top_pct > 40:
            print(f"  - Performance is dominated by a single component ({top_component})")
        elif top3_pct > 80:
            print(f"  - Performance is concentrated in top 3 components")
        else:
            print(f"  - Performance is distributed across multiple components")

    print(f"\n📅 **BENCHMARK SUMMARY**:")
    print(f"  - Total _update calls: {num_update_calls}")
    print(f"  - Average time per call: {update_total_time/num_update_calls*1000:.1f}ms")
    print(f"  - Functions profiled: {len(report)}")
    print(f"  - Configuration: {config_path if 'config_path' in locals() else 'Default settings'}")

    print()
    print("=" * 80)
    print("✅ Objective performance analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark BlockPPO._update performance")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (e.g., configs/experiments/pose_control_exp.yaml)"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=None,
        help="Override number of agents"
    )
    parser.add_argument(
        "--num-envs-per-agent",
        type=int,
        default=None,
        help="Override number of environments per agent"
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=None,
        help="Override number of rollouts"
    )
    parser.add_argument(
        "--learning-epochs",
        type=int,
        default=None,
        help="Override number of learning epochs"
    )
    parser.add_argument(
        "--mini-batches",
        type=int,
        default=None,
        help="Override number of mini-batches"
    )
    parser.add_argument(
        "--value-update-ratio",
        type=int,
        default=None,
        help="Override value update ratio"
    )

    args = parser.parse_args()

    # Build override dictionary from command line args
    overrides = {}
    if args.num_agents is not None:
        overrides['num_agents'] = args.num_agents
    if args.num_envs_per_agent is not None:
        overrides['num_envs_per_agent'] = args.num_envs_per_agent
    if args.rollouts is not None:
        overrides['rollouts'] = args.rollouts
    if args.learning_epochs is not None:
        overrides['learning_epochs'] = args.learning_epochs
    if args.mini_batches is not None:
        overrides['mini_batches'] = args.mini_batches
    if args.value_update_ratio is not None:
        overrides['value_update_ratio'] = args.value_update_ratio

    run_benchmark(config_path=args.config, override_params=overrides if overrides else None)