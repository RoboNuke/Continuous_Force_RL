"""
Environment Factory Utility

This utility provides a convenient interface for creating Isaac Lab factory environments
with modular wrapper composition. It supports different configuration presets and
flexible wrapper application.

Features:
- Direct Isaac Lab factory environment integration
- Modular wrapper composition
- Configuration presets for common use cases
- Multi-agent environment setup
- Flexible parameter overrides
"""

import gymnasium as gym
from typing import Dict, List, Optional, Any, Union
import importlib.util
import sys
import os

# Import wrappers
from .sensors.force_torque_wrapper import ForceTorqueWrapper
from .mechanics.fragile_object_wrapper import FragileObjectWrapper
from .mechanics.efficient_reset_wrapper import EfficientResetWrapper
from .control.hybrid_force_position_wrapper import HybridForcePositionWrapper
from .logging.wandb_logging_wrapper import WandbLoggingWrapper
from .logging.factory_metrics_wrapper import FactoryMetricsWrapper
from .observations.history_observation_wrapper import HistoryObservationWrapper
from .observations.observation_manager_wrapper import ObservationManagerWrapper


class FactoryEnvironmentBuilder:
    """
    Builder class for creating factory environments with modular wrapper composition.

    This class provides a flexible interface for creating Isaac Lab factory environments
    with various wrappers applied in the correct order.
    """

    def __init__(self):
        """Initialize the environment builder."""
        self.wrappers = []
        self.config_overrides = {}

    def with_force_torque_sensor(self, **kwargs) -> 'FactoryEnvironmentBuilder':
        """Add force-torque sensor wrapper."""
        self.wrappers.append(('force_torque', ForceTorqueWrapper, kwargs))
        return self

    def with_fragile_objects(self, num_agents: int = 1, **kwargs) -> 'FactoryEnvironmentBuilder':
        """Add fragile object wrapper with multi-agent support."""
        kwargs['num_agents'] = num_agents
        self.wrappers.append(('fragile_objects', FragileObjectWrapper, kwargs))
        return self

    def with_efficient_reset(self, cache_size_ratio: float = 0.1, **kwargs) -> 'FactoryEnvironmentBuilder':
        """Add efficient reset wrapper."""
        kwargs['cache_size_ratio'] = cache_size_ratio
        self.wrappers.append(('efficient_reset', EfficientResetWrapper, kwargs))
        return self

    def with_hybrid_control(self, reward_strategy: str = "simp", **kwargs) -> 'FactoryEnvironmentBuilder':
        """Add hybrid force-position control wrapper."""
        kwargs['reward_strategy'] = reward_strategy
        self.wrappers.append(('hybrid_control', HybridForcePositionWrapper, kwargs))
        return self

    def with_wandb_logging(self, project_name: str, num_agents: int = 1, **kwargs) -> 'FactoryEnvironmentBuilder':
        """Add Wandb logging wrapper."""
        kwargs.update({'project_name': project_name, 'num_agents': num_agents})
        self.wrappers.append(('wandb_logging', WandbLoggingWrapper, kwargs))
        return self

    def with_factory_metrics(self, num_agents: int = 1, **kwargs) -> 'FactoryEnvironmentBuilder':
        """Add factory-specific metrics wrapper."""
        kwargs['num_agents'] = num_agents
        self.wrappers.append(('factory_metrics', FactoryMetricsWrapper, kwargs))
        return self

    def with_history_observations(self, history_components: Optional[List[str]] = None,
                                 history_length: Optional[int] = None, **kwargs) -> 'FactoryEnvironmentBuilder':
        """Add selective history observation wrapper."""
        if history_components is not None:
            kwargs['history_components'] = history_components
        if history_length is not None:
            kwargs['history_length'] = history_length
        self.wrappers.append(('history_observations', HistoryObservationWrapper, kwargs))
        return self

    def with_observation_manager(self, use_obs_noise: bool = False, **kwargs) -> 'FactoryEnvironmentBuilder':
        """Add observation manager wrapper."""
        kwargs['use_obs_noise'] = use_obs_noise
        self.wrappers.append(('observation_manager', ObservationManagerWrapper, kwargs))
        return self

    def with_config_override(self, **config_overrides) -> 'FactoryEnvironmentBuilder':
        """Add configuration overrides for the base environment."""
        self.config_overrides.update(config_overrides)
        return self

    def build(self, env_cfg, task_name: str = "Factory-Task-Direct-v0") -> gym.Env:
        """
        Build the environment with all configured wrappers.

        Args:
            env_cfg: Environment configuration object
            task_name: Isaac Lab task name for gym.make

        Returns:
            Wrapped environment ready for training
        """
        # Apply configuration overrides
        for key, value in self.config_overrides.items():
            if hasattr(env_cfg, key):
                setattr(env_cfg, key, value)

        # Create base Isaac Lab environment
        env = gym.make(task_name, cfg=env_cfg)

        # Apply wrappers in order
        for wrapper_name, wrapper_class, kwargs in self.wrappers:
            try:
                env = wrapper_class(env, **kwargs)
            except Exception as e:
                print(f"Warning: Failed to apply {wrapper_name} wrapper: {e}")
                continue

        return env


def create_factory_environment(env_cfg,
                             task_name: str = "Factory-Task-Direct-v0",
                             preset: Optional[str] = None,
                             num_agents: int = 1,
                             **kwargs) -> gym.Env:
    """
    Create a factory environment with optional preset configurations.

    Args:
        env_cfg: Environment configuration object
        task_name: Isaac Lab task name
        preset: Preset configuration name ('basic', 'training', 'research', 'multi_agent')
        num_agents: Number of agents for multi-agent setups
        **kwargs: Additional configuration overrides

    Returns:
        Configured environment with wrappers applied
    """
    builder = FactoryEnvironmentBuilder()

    # Apply preset configurations
    if preset == "basic":
        # Basic setup with essential wrappers
        builder = (builder
                  .with_force_torque_sensor()
                  .with_observation_manager()
                  .with_factory_metrics(num_agents=num_agents))

    elif preset == "training":
        # Training setup with performance optimizations
        builder = (builder
                  .with_force_torque_sensor()
                  .with_efficient_reset(cache_size_ratio=0.2)
                  .with_observation_manager(use_obs_noise=True)
                  .with_factory_metrics(num_agents=num_agents))

    elif preset == "research":
        # Research setup with comprehensive logging and analysis
        builder = (builder
                  .with_force_torque_sensor()
                  .with_fragile_objects(num_agents=num_agents)
                  .with_efficient_reset()
                  .with_history_observations(
                      history_components=["force_torque", "ee_linvel", "ee_angvel"],
                      calc_acceleration=True
                  )
                  .with_observation_manager()
                  .with_factory_metrics(num_agents=num_agents))

    elif preset == "multi_agent":
        # Multi-agent setup with logging
        if 'project_name' in kwargs:
            project_name = kwargs.pop('project_name')
            builder = (builder
                      .with_force_torque_sensor()
                      .with_fragile_objects(num_agents=num_agents)
                      .with_observation_manager()
                      .with_factory_metrics(num_agents=num_agents)
                      .with_wandb_logging(project_name=project_name, num_agents=num_agents))
        else:
            builder = (builder
                      .with_force_torque_sensor()
                      .with_fragile_objects(num_agents=num_agents)
                      .with_observation_manager()
                      .with_factory_metrics(num_agents=num_agents))

    elif preset == "control_research":
        # Control research with hybrid force-position control
        reward_strategy = kwargs.pop('reward_strategy', 'simp')
        builder = (builder
                  .with_force_torque_sensor()
                  .with_hybrid_control(reward_strategy=reward_strategy)
                  .with_history_observations(
                      history_components=["force_torque", "ee_linvel"],
                      calc_acceleration=True
                  )
                  .with_observation_manager()
                  .with_factory_metrics(num_agents=num_agents))

    # Apply any additional configuration overrides
    if kwargs:
        builder = builder.with_config_override(**kwargs)

    return builder.build(env_cfg, task_name)


def get_available_presets() -> Dict[str, str]:
    """Get available preset configurations with descriptions."""
    return {
        "basic": "Essential wrappers for basic functionality",
        "training": "Optimized setup for training with noise and efficient resets",
        "research": "Comprehensive setup with history and detailed metrics",
        "multi_agent": "Multi-agent setup with static environment assignment",
        "control_research": "Setup for control research with hybrid force-position control",
    }


def create_multi_agent_environment(env_cfg,
                                 num_agents: int,
                                 task_name: str = "Factory-Task-Direct-v0",
                                 project_name: Optional[str] = None,
                                 **kwargs) -> gym.Env:
    """
    Convenience function for creating multi-agent environments.

    Args:
        env_cfg: Environment configuration
        num_agents: Number of agents
        task_name: Isaac Lab task name
        project_name: Wandb project name (optional)
        **kwargs: Additional configuration

    Returns:
        Multi-agent environment with static assignment
    """
    # Ensure environment count is divisible by agent count
    if hasattr(env_cfg.scene, 'num_envs'):
        if env_cfg.scene.num_envs % num_agents != 0:
            # Adjust to nearest valid count
            adjusted_count = (env_cfg.scene.num_envs // num_agents) * num_agents
            print(f"Adjusting num_envs from {env_cfg.scene.num_envs} to {adjusted_count} for {num_agents} agents")
            env_cfg.scene.num_envs = adjusted_count

    preset_kwargs = kwargs.copy()
    if project_name:
        preset_kwargs['project_name'] = project_name

    return create_factory_environment(
        env_cfg=env_cfg,
        task_name=task_name,
        preset="multi_agent",
        num_agents=num_agents,
        **preset_kwargs
    )


def validate_environment_config(env_cfg, num_agents: int = 1) -> List[str]:
    """
    Validate environment configuration for wrapper compatibility.

    Args:
        env_cfg: Environment configuration to validate
        num_agents: Number of agents for validation

    Returns:
        List of validation issues (empty if valid)
    """
    issues = []

    # Check if num_envs is divisible by num_agents
    if hasattr(env_cfg.scene, 'num_envs'):
        if env_cfg.scene.num_envs % num_agents != 0:
            issues.append(f"num_envs ({env_cfg.scene.num_envs}) must be divisible by num_agents ({num_agents})")
    else:
        issues.append("Missing scene.num_envs configuration")

    # Check for required Isaac Lab components
    required_attrs = ['scene', 'sim']
    for attr in required_attrs:
        if not hasattr(env_cfg, attr):
            issues.append(f"Missing required configuration: {attr}")

    # Check observation configuration if present
    if hasattr(env_cfg, 'obs_order') and hasattr(env_cfg, 'observation_space'):
        try:
            from envs.factory.factory_env_cfg import OBS_DIM_CFG
            expected_dim = sum([OBS_DIM_CFG.get(obs, 0) for obs in env_cfg.obs_order])
            if expected_dim != env_cfg.observation_space:
                issues.append(f"Observation space mismatch: expected {expected_dim}, got {env_cfg.observation_space}")
        except ImportError:
            issues.append("Could not validate observation dimensions - OBS_DIM_CFG not available")

    return issues


# Example usage patterns for documentation
EXAMPLE_USAGE = """
# Basic usage with preset
env = create_factory_environment(
    env_cfg=my_config,
    preset="training",
    num_agents=2
)

# Custom wrapper composition
env = (FactoryEnvironmentBuilder()
       .with_force_torque_sensor()
       .with_hybrid_control(reward_strategy="delta")
       .with_history_observations(history_components=["force_torque"])
       .with_observation_manager(use_obs_noise=True)
       .build(my_config))

# Multi-agent with logging
env = create_multi_agent_environment(
    env_cfg=my_config,
    num_agents=4,
    project_name="factory_multi_agent_exp"
)

# Validation before creation
issues = validate_environment_config(my_config, num_agents=2)
if not issues:
    env = create_factory_environment(my_config, preset="research")
"""