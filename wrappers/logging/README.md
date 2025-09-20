# Configurable Logging System

This directory contains a completely refactored logging system that provides truly configurable, environment-agnostic Wandb logging with proper separation of concerns.

## Overview

The new system replaces the previous hardcoded, factory-specific logging with a flexible, configuration-driven approach:

- **Factory Metrics Wrapper**: Computes factory-specific metrics and puts them in the `info` dictionary
- **Generic Wandb Wrapper**: Truly environment-agnostic logging that tracks only configured metrics from `info`
- **Configuration System**: YAML/JSON-based configuration for flexible metric tracking

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
from wrappers.logging.wandb_wrapper import GenericWandbLoggingWrapper
from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper
from configs.config_manager import LoggingConfigPresets

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
from wrappers.logging.wandb_wrapper import GenericWandbLoggingWrapper
from configs.config_manager import LoggingConfigPresets

# Use preset configuration
config = LoggingConfigPresets.basic_config()
config.wandb_project = "my_project"
env = GenericWandbLoggingWrapper(env, config)
```

### Configuration Files

```python
from wrappers.logging.wandb_wrapper import GenericWandbLoggingWrapper
from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper
from configs.config_manager import load_config_from_file

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
from configs.config_manager import LoggingConfig, MetricConfig

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


## API Reference

### GenericWandbLoggingWrapper

**Core Wandb Integration Wrapper**

Provides multi-agent Wandb logging with static environment assignment and episode tracking.

```python
class GenericWandbLoggingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, num_agents: int = 1, env_cfg: Any = None)
    def add_metrics(self, metrics: Dict[str, torch.Tensor])
    def log_minibatch_update(self, learning_data: Dict[str, torch.Tensor])
    def add_onetime_learning_metrics(self, learning_metrics: Dict[str, torch.Tensor])
    def step(self, action)
    def close(self)
```

**Key Features:**
- Multi-agent support with static environment assignment
- Automatic episode tracking and aggregation
- Separate Wandb runs per agent
- Metric splitting by environment/agent assignment
- Learning metrics integration

**Dependencies:**
- Requires `env_cfg` with `agent_configs` containing Wandb configuration per agent
- Environment division must be evenly divisible by number of agents

### SimpleEpisodeTracker

**Individual Agent Episode Tracking**

Tracks metrics and publishes to individual agent Wandb runs.

```python
class SimpleEpisodeTracker:
    def __init__(self, num_envs: int, device: torch.device, agent_config: Dict[str, Any], env_config: Any)
    def increment_steps(self)
    def add_metrics(self, metrics: Dict[str, torch.Tensor])
    def log_minibatch_update(self, learning_data: Dict[str, torch.Tensor])
    def add_onetime_learning_metrics(self, learning_metrics: Dict[str, torch.Tensor])
    def close(self)
```

**Key Features:**
- Per-agent metric accumulation
- Wandb run management
- Step counting for x-axis
- Learning metrics aggregation
- Automatic cleanup on publish

### FactoryMetricsWrapper

**Factory-Specific Metrics Computation**

Computes and tracks factory manipulation task metrics including success, engagement, smoothness, and force/torque statistics.

```python
class FactoryMetricsWrapper(gym.Wrapper):
    def __init__(self, env, num_agents=1)
    def step(self, action)
    def reset(self, **kwargs)
    def close(self)
```

**Tracked Metrics:**
- `ep_succeeded`: Boolean success tracking per environment
- `ep_success_times`: Time to success per environment
- `ep_engaged`: Boolean engagement tracking per environment
- `ep_engaged_times`: Time to engagement per environment
- `ep_engaged_length`: Total engagement duration per environment
- `ep_ssv`: Sum squared velocity (smoothness metric)
- `ep_ssjv`: Sum squared joint velocity (smoothness metric)
- `ep_max_force`, `ep_max_torque`: Force/torque maximums (if available)
- `ep_sum_force`, `ep_sum_torque`: Force/torque sums (if available)

**Dependencies:**
- Requires an environment with `add_metrics` method (e.g., GenericWandbLoggingWrapper)
- Optionally detects `robot_force_torque` attribute for force/torque tracking
- Optionally detects `_robot` attribute for advanced wrapper initialization

### EnhancedActionLoggingWrapper

**Detailed Action Component Analysis**

Provides comprehensive action statistics tracking with configurable component-wise analysis.

```python
class EnhancedActionLoggingWrapper(gym.Wrapper):
    def __init__(self, env, track_selection=True, track_pos=True, track_rot=True,
                 track_force=True, track_torque=True, force_size=6, logging_frequency=100)
    def step(self, action)
    def reset(self, **kwargs)
    def close(self)
```

**Configuration Options:**
- `track_selection`: Track selection component statistics
- `track_pos`: Track position component statistics
- `track_rot`: Track rotation component statistics
- `track_force`: Track force component statistics
- `track_torque`: Track torque component statistics
- `force_size`: Action space force configuration (0, 3, or 6)
- `logging_frequency`: How often to collect and log statistics

**Key Features:**
- Configurable component tracking
- Automatic action space layout detection
- Component-wise statistics (mean, std, min, max, abs_mean, magnitude)
- Flexible force/torque configuration
- Step-based logging frequency control

**Dependencies:**
- Requires an environment with `add_metrics` method in the wrapper chain
- Validates wrapper chain for Wandb integration

### Configuration Classes

**MockEnvConfig (for testing)**

```python
class MockEnvConfig:
    def __init__(self):
        self.num_envs = 4
        self.device = torch.device("cpu")
        self.agent_configs = {
            'agent_0': {
                'wandb_entity': 'test_entity',
                'wandb_project': 'test_project',
                'wandb_name': 'test_agent_0',
                'wandb_group': 'test_group',
                'wandb_tags': ['test']
            }
        }
```

## File Structure

```
wrappers/logging/
├── README.md                     # This file
├── wandb_wrapper.py               # Generic wandb wrapper
└── factory_metrics_wrapper.py    # Factory metrics computation

configs/
├── config_manager.py             # Unified configuration system (includes logging)
└── logging/
    ├── factory_logging.yaml      # Factory preset
    └── basic_logging.yaml        # Basic preset

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
python -m pytest tests/unit/test_factory_metrics_wrapper.py -v

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