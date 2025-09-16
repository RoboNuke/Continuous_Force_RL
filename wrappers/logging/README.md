# Configurable Logging System

This directory contains a completely refactored logging system that provides truly configurable, environment-agnostic Wandb logging with proper separation of concerns.

## Overview

The new system replaces the previous hardcoded, factory-specific logging with a flexible, configuration-driven approach:

- **Factory Metrics Wrapper**: Computes factory-specific metrics and puts them in the `info` dictionary
- **Generic Wandb Wrapper**: Truly environment-agnostic logging that tracks only configured metrics from `info`
- **Configuration System**: YAML/JSON-based configuration for flexible metric tracking
- **Backward Compatibility**: Legacy wrapper still works but is deprecated

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Environment Stack                             │
├─────────────────────────────────────────────────────────────────┤
│  GenericWandbLoggingWrapper (configurable metric tracking)     │
│  └── FactoryMetricsWrapper (computes factory metrics)          │
│      └── Base Environment                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### ✅ **Truly Generic Logging**
- No hardcoded assumptions about environment type
- Configurable metric tracking via YAML/JSON files
- Works with any gymnasium environment

### ✅ **Factory-Specific Convenience**
- `FactoryMetricsWrapper` pre-configured for factory manipulation tasks
- Automatic engagement/success/force metric computation and extraction
- Backward-compatible with existing factory code

### ✅ **Multi-Agent Support**
- Static environment assignment across agents
- Separate Wandb runs per agent
- Automatic metric splitting

### ✅ **Configuration-Driven**
- YAML/JSON configuration files
- Runtime metric enable/disable
- Preset configurations for common environment types

### ✅ **Separation of Concerns**
- Factory metrics computation (`FactoryMetricsWrapper`) separated from logging
- Generic logging (`GenericWandbLoggingWrapper`) separated from environment-specific logic
- Clean two-layer architecture

## Quick Start

### Factory Environments

```python
from wrappers.logging.generic_wandb_wrapper import GenericWandbLoggingWrapper
from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper
from wrappers.logging.logging_config import LoggingConfigPresets

# Wrap environment with factory metrics computation
env = FactoryMetricsWrapper(env)

# Create factory configuration
config = LoggingConfigPresets.factory_config()
config.wandb_entity = 'your_entity'
config.wandb_project = 'factory_project'

# Wrap with generic logging (reads metrics from config)
env = GenericWandbLoggingWrapper(env, config)
```

### Generic Environments

```python
from wrappers.logging.generic_wandb_wrapper import GenericWandbLoggingWrapper
from wrappers.logging.logging_config import LoggingConfigPresets

# Use preset configuration
config = LoggingConfigPresets.basic_config()
config.wandb_project = "my_project"
env = GenericWandbLoggingWrapper(env, config)
```

### Configuration Files

```python
from wrappers.logging.generic_wandb_wrapper import GenericWandbLoggingWrapper
from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper
from wrappers.logging.logging_config import load_config_from_file

# Wrap with factory metrics first
env = FactoryMetricsWrapper(env)

# Load from config file
config = load_config_from_file("configs/logging/factory_logging.yaml")
env = GenericWandbLoggingWrapper(env, config)
```

## Configuration Format

### YAML Configuration Example

```yaml
# Wandb Settings
wandb_entity: "your_entity"
wandb_project: "factory_manipulation"
wandb_name: "experiment_name"
wandb_tags: ["factory", "manipulation"]

# General Settings
num_agents: 1
track_learning_metrics: true
track_action_metrics: true

# Tracked Metrics
tracked_metrics:
  current_engagements:
    default_value: 0.0
    metric_type: "boolean"
    aggregation: "mean"
    wandb_name: "Engagement / Engaged Rate"
    enabled: true

  max_force:
    default_value: 0.0
    metric_type: "scalar"
    aggregation: "mean"
    wandb_name: "Force / Max Force"
    normalize_by_episode_length: false
    enabled: true
```

### Python Configuration

```python
from wrappers.logging.logging_config import LoggingConfig, MetricConfig

config = LoggingConfig()
config.wandb_project = "my_project"

config.add_metric(MetricConfig(
    name="custom_metric",
    metric_type="scalar",
    wandb_name="Custom / Metric",
    aggregation="mean",
    enabled=True
))
```

## Metric Types

### Supported Metric Types

- **`scalar`**: Numeric values (rewards, forces, distances)
- **`boolean`**: Binary states (engagement, success, failures)
- **`tensor`**: Multi-dimensional data (positions, velocities)
- **`histogram`**: Distribution tracking (planned)

### Aggregation Methods

- **`mean`**: Average across environments
- **`sum`**: Total across environments
- **`max`**: Maximum value
- **`min`**: Minimum value
- **`median`**: Median value

## Preset Configurations

### Factory Configuration
Pre-configured for factory manipulation tasks:
- Engagement/success tracking
- Force/torque metrics
- Smoothness metrics
- Reward component tracking

```python
config = LoggingConfigPresets.factory_config()
```

### Basic Configuration
Minimal configuration for simple environments:
- Episode rewards
- Episode length
- Basic learning metrics

```python
config = LoggingConfigPresets.basic_config()
```

### Locomotion Configuration
Optimized for locomotion tasks:
- Distance traveled
- Velocity tracking
- Energy consumption
- Stability metrics

```python
config = LoggingConfigPresets.locomotion_config()
```

## Multi-Agent Support

The system supports multi-agent scenarios with static environment assignment:

```python
# 1024 environments, 4 agents = 256 environments per agent
env = FactoryMetricsWrapper(env, num_agents=4)
env = GenericWandbLoggingWrapper(env, config, num_agents=4)

# Get agent assignments
assignments = env.get_agent_assignment()
# {0: [0-255], 1: [256-511], 2: [512-767], 3: [768-1023]}

# Each agent gets its own Wandb run
# Metrics are automatically split and tracked per agent
```

## Migration Guide

### From Legacy Wrapper

**Old (Deprecated):**
```python
from wrappers.logging.wandb_logging_wrapper import WandbLoggingWrapper
env = WandbLoggingWrapper(env, wandb_config)
```

**New (Recommended):**
```python
from wrappers.logging.generic_wandb_wrapper import GenericWandbLoggingWrapper
from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper
from wrappers.logging.logging_config import LoggingConfigPresets

env = FactoryMetricsWrapper(env)
config = LoggingConfigPresets.factory_config()
config.wandb_entity = wandb_config['entity']
config.wandb_project = wandb_config['project']
config.wandb_name = wandb_config['name']
env = GenericWandbLoggingWrapper(env, config)
```

### Benefits of Migration

1. **No Hardcoded Assumptions**: Generic wrapper works with any environment
2. **Configurable Metrics**: Track only what you need
3. **Better Performance**: Reduced overhead from unused features
4. **Cleaner Code**: Proper separation of concerns
5. **Future-Proof**: Extensible configuration system

## API Reference

### GenericWandbLoggingWrapper

```python
class GenericWandbLoggingWrapper(gym.Wrapper):
    def __init__(self, env, logging_config, num_agents=1)
    def add_metric(self, tag, value)
    def log_learning_metrics(self, **kwargs)
    def log_action_metrics(self, actions, global_step=-1)
    def publish_metrics(self)
    def get_agent_assignment(self)
    def close(self)
```

### FactoryMetricsWrapper

```python
class FactoryMetricsWrapper(gym.Wrapper):
    def __init__(self, env, num_agents=1)
    def get_success_stats(self)
    def get_smoothness_stats(self)
    def get_agent_metrics(self, agent_id)
    def get_agent_assignment(self)
```

### LoggingConfig

```python
class LoggingConfig:
    def add_metric(self, metric: MetricConfig)
    def remove_metric(self, metric_name: str)
    def enable_metric(self, metric_name: str, enabled: bool = True)
    def to_wandb_config(self) -> Dict[str, Any]
```

## File Structure

```
wrappers/logging/
├── README.md                     # This file
├── logging_config.py             # Configuration system
├── generic_wandb_wrapper.py      # Generic wrapper
├── factory_metrics_wrapper.py    # Factory metrics computation
└── wandb_logging_wrapper.py      # Legacy wrapper (deprecated)

configs/logging/
├── factory_logging.yaml          # Factory preset
├── basic_logging.yaml            # Basic preset
└── locomotion_logging.yaml       # Locomotion preset

tests/unit/
├── test_logging_config.py        # Configuration tests
└── test_generic_wandb_wrapper.py # Generic wrapper tests

tests/integration/
└── test_logging_integration.py   # Integration tests

examples/
└── logging_usage_examples.py     # Usage examples
```

## Advanced Usage

### Custom Metric Extraction

Environments can provide custom metrics in the `info` dictionary:

```python
def step(self, action):
    # ... environment logic ...

    info = {
        'custom_metric': torch.tensor([1.0, 2.0, 3.0, 4.0]),
        'nested_metrics': {
            'sub_metric': torch.tensor([0.1, 0.2, 0.3, 0.4])
        },
        'Reward / exploration': torch.tensor([0.01, 0.02, 0.03, 0.04])
    }

    return obs, reward, terminated, truncated, info
```

Configure to track these metrics:

```python
config.add_metric(MetricConfig(name="custom_metric"))
config.add_metric(MetricConfig(name="nested_metrics / sub_metric"))
config.add_metric(MetricConfig(name="Reward / exploration"))
```

### Runtime Configuration

```python
# Enable/disable metrics at runtime
wrapper.config.enable_metric("force_metrics", False)

# Add new metrics dynamically
wrapper.config.add_metric(MetricConfig(
    name="runtime_metric",
    metric_type="scalar"
))
```

### Performance Monitoring

```python
# Get factory-specific statistics
stats = wrapper.get_factory_stats()
print(f"Overall engagement rate: {stats['overall_engagement_rate']}")

# Get per-agent metrics
for agent_id in range(num_agents):
    agent_stats = wrapper.get_agent_metrics(agent_id)
    print(f"Agent {agent_id} success rate: {agent_stats['success_rate']}")
```

## Testing

Run the test suite:

```bash
# Unit tests
python -m pytest tests/unit/test_logging_config.py -v
python -m pytest tests/unit/test_generic_wandb_wrapper.py -v
python -m pytest tests/unit/test_factory_wandb_wrapper.py -v

# Integration tests
python -m pytest tests/integration/test_logging_integration.py -v

# All logging tests
python -m pytest tests/unit/test_*logging* tests/integration/test_*logging* -v
```

## Contributing

When adding new metric types or features:

1. Update the `MetricConfig` class with new options
2. Add handling in `GenericEpisodeTracker`
3. Add preset configurations if widely applicable
4. Add comprehensive tests
5. Update this README

## Troubleshooting

### Common Issues

**Issue**: Metrics not appearing in Wandb
**Solution**: Check that metrics are enabled in configuration and present in `info` dict

**Issue**: Multi-agent assignment errors
**Solution**: Ensure `num_envs` is divisible by `num_agents`

**Issue**: Import errors for legacy wrapper
**Solution**: Update to new wrapper system following migration guide

**Issue**: Configuration file not loading
**Solution**: Check file format (YAML/JSON) and install required dependencies (`PyYAML`)

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Wrapper will now provide detailed logging
```

---

For more examples and advanced usage patterns, see `examples/logging_usage_examples.py`.