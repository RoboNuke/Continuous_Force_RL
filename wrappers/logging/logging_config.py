"""
Logging Configuration System

Defines configuration classes and presets for different types of logging wrappers.
This allows for flexible, configurable metric tracking without hardcoded assumptions.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
import torch


@dataclass
class MetricConfig:
    """Configuration for a single tracked metric."""

    name: str
    """Name of the metric in the info dictionary."""

    default_value: Union[float, int, torch.Tensor] = 0.0
    """Default value if metric is not present in info."""

    metric_type: str = "scalar"
    """Type of metric: 'scalar', 'boolean', 'tensor', 'histogram'."""

    aggregation: str = "mean"
    """How to aggregate across environments: 'mean', 'sum', 'max', 'min', 'median'."""

    wandb_name: Optional[str] = None
    """Custom name for Wandb logging. If None, uses the metric name."""

    enabled: bool = True
    """Whether to track this metric."""

    normalize_by_episode_length: bool = False
    """Whether to normalize by episode length for episode metrics."""


@dataclass
class LoggingConfig:
    """Main configuration for logging wrappers."""

    # Wandb configuration
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[List[str]] = None

    # Tracking configuration
    tracked_metrics: Dict[str, MetricConfig] = field(default_factory=dict)
    track_learning_metrics: bool = True
    track_action_metrics: bool = True

    # Episode tracking
    track_episodes: bool = True
    track_rewards: bool = True
    track_episode_length: bool = True
    track_terminations: bool = True

    # PPO-specific settings
    clip_eps: float = 0.2

    # Multi-agent settings
    num_agents: int = 1

    def add_metric(self, metric: MetricConfig):
        """Add a metric to the tracking configuration."""
        self.tracked_metrics[metric.name] = metric

    def remove_metric(self, metric_name: str):
        """Remove a metric from tracking."""
        if metric_name in self.tracked_metrics:
            del self.tracked_metrics[metric_name]

    def enable_metric(self, metric_name: str, enabled: bool = True):
        """Enable or disable a specific metric."""
        if metric_name in self.tracked_metrics:
            self.tracked_metrics[metric_name].enabled = enabled

    def to_wandb_config(self) -> Dict[str, Any]:
        """Convert to Wandb configuration dictionary."""
        return {
            'entity': self.wandb_entity,
            'project': self.wandb_project,
            'name': self.wandb_name,
            'group': self.wandb_group,
            'tags': self.wandb_tags or [],
            'num_agents': self.num_agents,
            'tracked_metrics': [metric.name for metric in self.tracked_metrics.values() if metric.enabled]
        }


class LoggingConfigPresets:
    """Predefined logging configurations for different environment types."""

    @staticmethod
    def basic_config() -> LoggingConfig:
        """Basic configuration with minimal tracking."""
        config = LoggingConfig()

        # Only track basic episode metrics
        config.add_metric(MetricConfig(
            name="reward",
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Episode / Reward"
        ))

        return config

    @staticmethod
    def factory_config() -> LoggingConfig:
        """Configuration optimized for factory manipulation tasks."""
        config = LoggingConfig()

        # Core episode metrics
        config.add_metric(MetricConfig(
            name="reward",
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Episode / Reward"
        ))

        # Factory-specific metrics
        config.add_metric(MetricConfig(
            name="current_engagements",
            default_value=0.0,
            metric_type="boolean",
            aggregation="mean",
            wandb_name="Engagement / Engaged Rate"
        ))

        config.add_metric(MetricConfig(
            name="current_successes",
            default_value=0.0,
            metric_type="boolean",
            aggregation="mean",
            wandb_name="Success / Success Rate"
        ))

        # Smoothness metrics
        config.add_metric(MetricConfig(
            name="sum_squared_velocity",
            default_value=0.0,
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Smoothness / Sum Squared Velocity",
            normalize_by_episode_length=True
        ))

        # Force/torque metrics (if available)
        config.add_metric(MetricConfig(
            name="max_force",
            default_value=0.0,
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Force / Max Force"
        ))

        config.add_metric(MetricConfig(
            name="max_torque",
            default_value=0.0,
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Force / Max Torque"
        ))

        config.add_metric(MetricConfig(
            name="avg_force",
            default_value=0.0,
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Force / Avg Force"
        ))

        config.add_metric(MetricConfig(
            name="avg_torque",
            default_value=0.0,
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Force / Avg Torque"
        ))

        # Reward components (will be detected dynamically)
        for component in ["reach_reward", "grasp_reward", "lift_reward", "align_reward"]:
            config.add_metric(MetricConfig(
                name=f"Reward / {component}",
                default_value=0.0,
                metric_type="scalar",
                aggregation="mean",
                wandb_name=f"Reward / {component.replace('_', ' ').title()}"
            ))

        return config

    @staticmethod
    def locomotion_config() -> LoggingConfig:
        """Configuration for locomotion tasks."""
        config = LoggingConfig()

        # Core episode metrics
        config.add_metric(MetricConfig(
            name="reward",
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Episode / Reward"
        ))

        # Locomotion-specific metrics
        config.add_metric(MetricConfig(
            name="distance_traveled",
            default_value=0.0,
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Locomotion / Distance Traveled"
        ))

        config.add_metric(MetricConfig(
            name="velocity",
            default_value=0.0,
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Locomotion / Average Velocity"
        ))

        config.add_metric(MetricConfig(
            name="energy_consumption",
            default_value=0.0,
            metric_type="scalar",
            aggregation="sum",
            wandb_name="Locomotion / Energy Consumption",
            normalize_by_episode_length=True
        ))

        return config

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> LoggingConfig:
        """Create logging config from dictionary (e.g., loaded from YAML/JSON)."""
        config = LoggingConfig()

        # Set basic config
        config.wandb_entity = config_dict.get('wandb_entity')
        config.wandb_project = config_dict.get('wandb_project')
        config.wandb_name = config_dict.get('wandb_name')
        config.wandb_group = config_dict.get('wandb_group')
        config.wandb_tags = config_dict.get('wandb_tags', [])
        config.num_agents = config_dict.get('num_agents', 1)
        config.clip_eps = config_dict.get('clip_eps', 0.2)

        # Set tracking flags
        config.track_learning_metrics = config_dict.get('track_learning_metrics', True)
        config.track_action_metrics = config_dict.get('track_action_metrics', True)
        config.track_episodes = config_dict.get('track_episodes', True)
        config.track_rewards = config_dict.get('track_rewards', True)
        config.track_episode_length = config_dict.get('track_episode_length', True)
        config.track_terminations = config_dict.get('track_terminations', True)

        # Add tracked metrics
        for metric_name, metric_dict in config_dict.get('tracked_metrics', {}).items():
            metric = MetricConfig(
                name=metric_name,
                default_value=metric_dict.get('default_value', 0.0),
                metric_type=metric_dict.get('metric_type', 'scalar'),
                aggregation=metric_dict.get('aggregation', 'mean'),
                wandb_name=metric_dict.get('wandb_name'),
                enabled=metric_dict.get('enabled', True),
                normalize_by_episode_length=metric_dict.get('normalize_by_episode_length', False)
            )
            config.add_metric(metric)

        return config


def load_config_from_file(config_path: str) -> LoggingConfig:
    """Load logging configuration from YAML or JSON file."""
    import os
    from pathlib import Path

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
        try:
            import yaml
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML config files. Install with: pip install PyYAML")
    elif config_file.suffix.lower() == '.json':
        import json
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_file.suffix}")

    return LoggingConfigPresets.from_dict(config_dict)