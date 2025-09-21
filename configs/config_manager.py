"""
Configuration Manager for hierarchical configuration system.

Handles loading, resolving, and applying configurations for the factory training system.
Includes integrated logging configuration management.
"""

import yaml
import os
from typing import Dict, Any, List, Optional, Union
import copy
from dataclasses import dataclass, field
import torch


class ConfigManager:
    """Manages configuration loading and resolution."""

    # Class-level source tracking
    _config_sources = {}

    @staticmethod
    def load_and_resolve_config(config_path: str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load and resolve configuration with base inheritance and CLI overrides.

        Args:
            config_path: Path to the configuration file
            overrides: List of override strings in format "key=value"

        Returns:
            Fully resolved configuration dictionary
        """
        # Clear previous source tracking
        ConfigManager._config_sources = {}

        # Load main config
        config = ConfigManager._load_yaml_file(config_path)

        # Load base configuration if specified
        if 'base' in config:
            base_path = ConfigManager._resolve_config_path(config['base'], config_path)
            base_config = ConfigManager._load_yaml_file(base_path)

            # Track base config sources
            ConfigManager._track_config_sources(base_config, 'base')

            # Track local override sources
            ConfigManager._track_config_sources(config, 'local_override')

            config = ConfigManager._merge_configs(base_config, config)
        else:
            # No base config, all current values are from local file
            ConfigManager._track_config_sources(config, 'local_override')

        # Apply CLI overrides
        if overrides:
            for override in overrides:
                key, _ = override.split('=', 1)
                ConfigManager._config_sources[key] = 'cli_override'
                ConfigManager._apply_override(config, override)

        # Resolve configuration
        resolved_config = ConfigManager._resolve_config(config)

        return resolved_config

    @staticmethod
    def apply_to_isaac_lab(env_cfg, agent_cfg, resolved_config):
        """Apply configuration to Isaac Lab environment and agent configs."""
        environment = resolved_config.get('environment', {})
        primary = resolved_config.get('primary', {})

        # Apply environment overrides
        for key, value in environment.items():
            # Log what we're applying
            existing_value = getattr(env_cfg, key, "NOT_SET") if hasattr(env_cfg, key) else "NOT_SET"
            print(f"[CONFIG]: Applying {key}: {existing_value} -> {value}")

            if '.' in key:
                ConfigManager._set_nested_attr(env_cfg, key, value)
            else:
                # Handle dictionary configurations that need to be merged with existing Isaac Lab configs
                if isinstance(value, dict):
                    existing_config = getattr(env_cfg, key, None)
                    if existing_config is not None:
                        # Check if existing config is an object (with attributes) or a dictionary
                        if isinstance(existing_config, dict):
                            # Existing config is a dictionary, merge directly
                            print(f"[CONFIG]: Merging {key} properties into existing dictionary configuration")
                            for prop_key, prop_value in value.items():
                                print(f"[CONFIG]:   Setting {key}[{prop_key}] = {prop_value}")
                                existing_config[prop_key] = prop_value
                        else:
                            # Existing config is an object, merge using setattr
                            print(f"[CONFIG]: Merging {key} properties into existing Isaac Lab configuration object")
                            for prop_key, prop_value in value.items():
                                print(f"[CONFIG]:   Setting {key}.{prop_key} = {prop_value}")
                                setattr(existing_config, prop_key, prop_value)
                    else:
                        # No existing config, create a new object from our dictionary
                        print(f"[CONFIG]: Creating new {key} configuration object")
                        config_obj = type(f'{key.title()}Config', (), value)()
                        setattr(env_cfg, key, config_obj)
                else:
                    # Set attribute directly (whether it exists or not)
                    setattr(env_cfg, key, value)

    @staticmethod
    def _load_yaml_file(file_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _resolve_config_path(config_path: str, current_file: str) -> str:
        """Resolve configuration path relative to configs directory."""
        if os.path.isabs(config_path):
            return config_path

        # Get configs directory (parent of current file's directory)
        configs_dir = os.path.dirname(os.path.dirname(current_file))
        if not config_path.endswith('.yaml'):
            config_path += '.yaml'

        return os.path.join(configs_dir, config_path)

    @staticmethod
    def _merge_configs(base_config: Dict, override_config: Dict) -> Dict:
        """Merge two configuration dictionaries."""
        merged = copy.deepcopy(base_config)

        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    @staticmethod
    def _apply_override(config: Dict, override_str: str):
        """Apply a CLI override to configuration."""
        if '=' not in override_str:
            raise ValueError(f"Invalid override format: {override_str}. Expected key=value")

        key, value = override_str.split('=', 1)

        # Parse value
        try:
            # Try to parse as Python literal (for lists, numbers, booleans)
            import ast
            value = ast.literal_eval(value)
        except:
            # Handle common string representations
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'none':
                value = None
            # Otherwise keep as string if literal_eval fails

        # Set nested key
        ConfigManager._set_nested_key(config, key, value)

    @staticmethod
    def _set_nested_key(config: Dict, key: str, value: Any):
        """Set nested key in configuration dictionary."""
        keys = key.split('.')
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    @staticmethod
    def _set_nested_attr(obj, key: str, value: Any):
        """Set nested attribute using dot notation."""
        keys = key.split('.')
        current = obj

        for k in keys[:-1]:
            current = getattr(current, k)

        setattr(current, keys[-1], value)

    @staticmethod
    def _resolve_config(config: Dict) -> Dict:
        """Resolve configuration and calculate derived parameters."""
        resolved = copy.deepcopy(config)

        # Calculate derived parameters
        derived = ConfigManager._calculate_derived_params(resolved.get('primary', {}))
        resolved['derived'] = derived

        # Resolve parameter references
        resolved = ConfigManager._resolve_references(resolved)

        return resolved

    @staticmethod
    def _calculate_derived_params(primary: Dict) -> Dict:
        """Calculate derived parameters from primary configuration."""
        derived = {}

        # Basic calculations using new structure
        agents_per_break_force = primary.get('agents_per_break_force', 2)
        num_envs_per_agent = primary.get('num_envs_per_agent', 256)
        break_forces = primary.get('break_forces', -1)

        # Calculate total agents and environments
        # total_agents = number_of_break_forces * agents_per_break_force
        # total_envs = total_agents * num_envs_per_agent
        num_break_forces = len(break_forces) if isinstance(break_forces, list) else 1
        derived['total_agents'] = num_break_forces * agents_per_break_force
        derived['total_num_envs'] = derived['total_agents'] * num_envs_per_agent

        # Calculate rollout steps from timing parameters
        episode_length_s = primary.get('episode_length_s', 12.0)
        decimation = primary.get('decimation', 4)
        policy_hz = primary.get('policy_hz', 15)
        sim_dt = (1/policy_hz) / decimation
        derived['rollout_steps'] = int((1/sim_dt) / decimation * episode_length_s)

        # Training parameters
        max_steps = primary.get('max_steps', 10240000)
        derived['max_steps'] = max_steps

        # Other derived values
        derived['sim_dt'] = sim_dt
        derived['ckpt_tracker_path'] = primary.get('ckpt_tracker_path', '/nfs/stak/users/brownhun/ckpt_tracker2.txt')
        derived['seed'] = primary.get('seed', -1)

        return derived

    @staticmethod
    def _resolve_references(config: Dict, context: Optional[Dict] = None) -> Dict:
        """Resolve ${section.key} references in configuration."""
        if context is None:
            context = config

        if isinstance(config, dict):
            resolved = {}
            for key, value in config.items():
                resolved[key] = ConfigManager._resolve_references(value, context)
            return resolved
        elif isinstance(config, list):
            return [ConfigManager._resolve_references(item, context) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            # Resolve reference
            ref_path = config[2:-1]  # Remove ${ and }
            return ConfigManager._get_nested_value(context, ref_path)
        else:
            return config

    @staticmethod
    def _track_config_sources(config: Dict, source_type: str, path_prefix: str = ""):
        """
        Recursively track configuration sources for all keys.

        Args:
            config: Configuration dictionary to track
            source_type: Type of source ('base', 'local_override', 'cli_override')
            path_prefix: Current path prefix for nested keys
        """
        for key, value in config.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key

            # Track this key's source
            ConfigManager._config_sources[current_path] = source_type

            # Recursively track nested dictionaries
            if isinstance(value, dict):
                ConfigManager._track_config_sources(value, source_type, current_path)

    @staticmethod
    def _get_nested_value(config: Dict, key_path: str) -> Any:
        """Get nested value from configuration using dot notation."""
        keys = key_path.split('.')
        current = config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and key.isdigit():
                # Handle list indexing
                index = int(key)
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    raise ValueError(f"Reference not found: {key_path} (list index {index} out of range)")
            else:
                raise ValueError(f"Reference not found: {key_path}")

        return current

    @staticmethod
    def print_env_config(env_cfg: Any) -> None:
        """Print environment configuration with color-coded source tracking."""
        from learning.config_printer import print_config
        ConfigManager._print_color_legend()
        print_config("Environment Configuration", env_cfg, color_map=ConfigManager._config_sources)

    @staticmethod
    def print_agent_config(agent_cfg: Any) -> None:
        """Print agent configuration with color-coded source tracking."""
        from learning.config_printer import print_config
        print_config("Agent Configuration", agent_cfg, color_map=ConfigManager._config_sources)

    @staticmethod
    def print_model_config(model_cfg: Any) -> None:
        """Print model configuration with color-coded source tracking."""
        from learning.config_printer import print_config
        print_config("Model Configuration", model_cfg, color_map=ConfigManager._config_sources)

    @staticmethod
    def print_task_config(task_cfg: Any) -> None:
        """Print task configuration with color-coded source tracking."""
        from learning.config_printer import print_config
        print_config("Task Configuration", task_cfg, color_map=ConfigManager._config_sources)

    @staticmethod
    def print_control_config(ctrl_cfg: Any) -> None:
        """Print control configuration with color-coded source tracking."""
        from learning.config_printer import print_config
        print_config("Control Configuration", ctrl_cfg, color_map=ConfigManager._config_sources)

    @staticmethod
    def print_config_sources() -> None:
        """Print all tracked configuration sources for debugging."""
        print("\n🔍 Configuration Source Tracking:")
        print("=" * 50)
        if not ConfigManager._config_sources:
            print("No source tracking data available.")
            return

        from learning.config_printer import Colors
        for path, source in sorted(ConfigManager._config_sources.items()):
            if source == 'local_override':
                print(f"{Colors.BLUE}{path}{Colors.RESET}: {source}")
            elif source == 'cli_override':
                print(f"{Colors.GREEN}{path}{Colors.RESET}: {source}")
            else:
                print(f"{path}: {source}")
        print("=" * 50)

    @staticmethod
    def _print_color_legend():
        """Print color legend for configuration source tracking."""
        from learning.config_printer import Colors
        print(f"\n🎨 Configuration Colors: {Colors.BLUE}Local overrides{Colors.RESET} | {Colors.GREEN}CLI overrides{Colors.RESET} | Default (base values)")


# ===== INTEGRATED LOGGING CONFIGURATION SYSTEM =====

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
    """
    DEPRECATED: Predefined logging configurations.

    This class is obsolete as the logging system has been simplified to use
    direct function calls instead of configuration files. The new system
    uses simple boolean parameters for wrapper configuration.

    See GenericWandbLoggingWrapper and EnhancedActionLoggingWrapper for the
    new simplified approach.
    """

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
    def from_dict(config_dict: Dict[str, Any]) -> LoggingConfig:
        """Create logging config from dictionary (integrated with ConfigManager YAML loading)."""
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

    @staticmethod
    def create_from_config(config_dict: Dict[str, Any], preset: str = "factory") -> LoggingConfig:
        """Create logging config from configuration with preset base and overrides."""
        # Start with preset
        if preset == "factory":
            config = LoggingConfigPresets.factory_config()
        elif preset == "basic":
            config = LoggingConfigPresets.basic_config()
        else:
            config = LoggingConfig()

        # Apply overrides from config_dict
        if config_dict:
            override_config = LoggingConfigPresets.from_dict(config_dict)
            # Merge configurations (prioritize override values)
            for key, value in override_config.__dict__.items():
                if key == 'tracked_metrics':
                    # Only merge if override has metrics to add
                    if value:
                        config.tracked_metrics.update(value)
                elif value is not None:
                    # Override any existing value with the new one
                    setattr(config, key, value)

        return config


def load_config_from_file(file_path: str) -> LoggingConfig:
    """Load logging configuration from YAML or JSON file."""
    from pathlib import Path
    import json

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    # Load file based on extension
    if file_path.suffix.lower() in ['.yaml', '.yml']:
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to load YAML configuration files")

        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif file_path.suffix.lower() == '.json':
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {file_path.suffix}. Use .yaml, .yml, or .json")

    return LoggingConfigPresets.from_dict(config_dict)