"""
Usage Examples for the Refactored Logging System

This file demonstrates how to use the new configurable logging wrappers
for different types of environments and use cases.
"""

import torch
from unittest.mock import Mock

# Example 1: Basic usage with factory environments
def example_factory_environment_logging():
    """Example: Using factory metrics wrapper with generic wandb logging."""
    from wrappers.logging.wandb_wrapper import GenericWandbLoggingWrapper
    from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper
    from configs.config_manager import LoggingConfigPresets

    # Create your factory environment
    env = create_factory_environment()  # Your factory env creation

    # Wrap with factory metrics to compute engagement/success/smoothness
    env = FactoryMetricsWrapper(env)

    # Create factory logging configuration
    config = LoggingConfigPresets.factory_config()
    config.wandb_entity = 'your_wandb_entity'
    config.wandb_project = 'factory_manipulation'
    config.wandb_name = 'experiment_name'
    config.wandb_tags = ['factory', 'manipulation']

    # Wrap with generic wandb logging (reads metrics from config)
    env = GenericWandbLoggingWrapper(env, config)

    # Use environment normally - metrics are automatically tracked and logged
    obs, info = env.reset()
    for step in range(1000):
        action = policy.get_action(obs)  # Your policy
        obs, reward, terminated, truncated, info = env.step(action)

        # Log learning metrics during training
        env.log_learning_metrics(
            returns=returns, values=values, policy_losses=policy_losses
        )

        # Publish metrics periodically
        if step % 100 == 0:
            env.publish_metrics()


# Example 2: Using configuration files
def example_config_file_usage():
    """Example: Using YAML configuration for flexible metric tracking."""
    from wrappers.logging.wandb_wrapper import GenericWandbLoggingWrapper
    from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper
    from configs.config_manager import load_config_from_file

    env = create_factory_environment()

    # Wrap with factory metrics first
    env = FactoryMetricsWrapper(env)

    # Load configuration from file
    config = load_config_from_file("configs/logging/factory_logging.yaml")
    env = GenericWandbLoggingWrapper(env, config)

    # Environment is ready with configured metrics


# Example 3: Generic wrapper with custom configuration
def example_generic_wrapper_custom_config():
    """Example: Using generic wrapper with custom metric configuration."""
    from wrappers.logging.wandb_wrapper import GenericWandbLoggingWrapper
    from configs.config_manager import LoggingConfig, MetricConfig

    env = create_any_environment()  # Any gymnasium environment

    # Create custom logging configuration
    config = LoggingConfig()
    config.wandb_project = "custom_rl_experiment"
    config.wandb_entity = "your_entity"

    # Add custom metrics
    config.add_metric(MetricConfig(
        name="custom_reward_component",
        metric_type="scalar",
        wandb_name="Custom Reward / Component",
        aggregation="mean"
    ))

    config.add_metric(MetricConfig(
        name="exploration_bonus",
        metric_type="scalar",
        wandb_name="Exploration / Bonus",
        normalize_by_episode_length=True
    ))

    # Create wrapper
    env = GenericWandbLoggingWrapper(env, config)


# Example 4: Multi-agent factory environment
def example_multi_agent_factory():
    """Example: Multi-agent factory environment with per-agent logging."""
    from wrappers.logging.wandb_wrapper import GenericWandbLoggingWrapper
    from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper
    from configs.config_manager import LoggingConfigPresets

    # Create environment with multiple agents (e.g., 1024 envs, 4 agents)
    env = create_factory_environment(num_envs=1024)

    # Wrap with factory metrics
    env = FactoryMetricsWrapper(env, num_agents=4)

    # Create factory config for multi-agent
    config = LoggingConfigPresets.factory_config()
    config.wandb_entity = 'multi_agent_team'
    config.wandb_project = 'factory_multi_agent'
    config.wandb_group = 'experiment_group'
    config.wandb_name = 'multi_agent_run'  # Will become multi_agent_run_0, multi_agent_run_1, etc.

    # Wrap with wandb logging - creates separate wandb runs for each agent
    env = GenericWandbLoggingWrapper(env, config, num_agents=4)

    # Get agent assignments
    assignments = env.get_agent_assignment()
    print(f"Agent 0 controls environments: {assignments[0]}")
    print(f"Agent 1 controls environments: {assignments[1]}")

    # Training loop works the same - metrics are automatically split by agent
    obs, info = env.reset()
    for step in range(1000):
        # actions should have shape [num_envs, action_dim]
        actions = multi_agent_policy.get_actions(obs)  # Your multi-agent policy
        obs, reward, terminated, truncated, info = env.step(actions)


# Example 5: Locomotion environment with preset configuration
def example_locomotion_environment():
    """Example: Using locomotion preset for walking/running tasks."""
    from wrappers.logging.wandb_wrapper import GenericWandbLoggingWrapper
    from configs.config_manager import LoggingConfigPresets

    env = create_locomotion_environment()  # Your locomotion env

    # Use locomotion preset configuration
    config = LoggingConfigPresets.locomotion_config()
    config.wandb_project = "locomotion_experiments"
    config.wandb_entity = "robotics_team"

    # Customize if needed
    config.enable_metric("falls", True)  # Enable fall tracking
    config.add_metric(MetricConfig(
        name="gait_quality",
        metric_type="scalar",
        wandb_name="Locomotion / Gait Quality"
    ))

    env = GenericWandbLoggingWrapper(env, config)




# Example 7: Advanced configuration with flags
def example_advanced_factory_config():
    """Example: Advanced factory configuration with custom metrics."""
    from wrappers.logging.wandb_wrapper import GenericWandbLoggingWrapper
    from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper
    from configs.config_manager import LoggingConfigPresets, MetricConfig

    env = create_factory_environment()

    # Wrap with factory metrics first
    env = FactoryMetricsWrapper(env)

    # Create custom factory config
    config = LoggingConfigPresets.factory_config()
    config.wandb_project = "advanced_factory"

    # Disable some metrics
    config.enable_metric("smoothness / Smoothness / Max Force", False)
    config.enable_metric("smoothness / Smoothness / Avg Torque", False)

    # Add custom factory metrics
    config.add_metric(MetricConfig(
        name="task_progress",
        metric_type="scalar",
        wandb_name="Task / Progress",
        aggregation="mean"
    ))

    # Create wrapper
    env = GenericWandbLoggingWrapper(env, config)


# Example 8: Custom metric extraction from environment
def example_custom_metric_extraction():
    """Example: Environment that provides custom metrics in info dict."""
    from wrappers.logging.wandb_wrapper import GenericWandbLoggingWrapper
    from configs.config_manager import LoggingConfig, MetricConfig

    class CustomEnvironment:
        """Example environment that provides custom metrics."""

        def step(self, action):
            # ... environment logic ...

            # Provide custom metrics in info
            info = {
                'custom_metric': torch.tensor([1.0, 2.0, 3.0, 4.0]),
                'nested_metrics': {
                    'sub_metric_1': torch.tensor([0.1, 0.2, 0.3, 0.4]),
                    'sub_metric_2': torch.tensor([0.5, 0.6, 0.7, 0.8])
                },
                'Reward / exploration': torch.tensor([0.01, 0.02, 0.03, 0.04])
            }

            return obs, reward, terminated, truncated, info

    env = CustomEnvironment()

    # Configure to track the custom metrics
    config = LoggingConfig()
    config.wandb_project = "custom_metrics"

    config.add_metric(MetricConfig(name="custom_metric"))
    config.add_metric(MetricConfig(name="nested_metrics / sub_metric_1"))
    config.add_metric(MetricConfig(name="Reward / exploration"))

    env = GenericWandbLoggingWrapper(env, config)


# Helper functions (would be replaced with actual environment creation)
def create_factory_environment(num_envs=256):
    """Placeholder for factory environment creation."""
    pass

def create_any_environment():
    """Placeholder for any gymnasium environment creation."""
    pass

def create_locomotion_environment():
    """Placeholder for locomotion environment creation."""
    pass


if __name__ == "__main__":
    print("Logging Usage Examples")
    print("======================")
    print("This file contains usage examples for the refactored logging system.")
    print("See the function definitions for detailed examples of:")
    print("- Factory environment logging")
    print("- Configuration file usage")
    print("- Custom metric tracking")
    print("- Multi-agent logging")
    print("- Locomotion environments")
    print("- Backward compatibility")
    print("- Advanced configurations")