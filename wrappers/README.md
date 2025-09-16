# Factory Environment Wrappers API Reference

This document provides the complete API reference for all wrapper classes extracted from the factory environment refactoring. Each wrapper is documented with its actual methods, parameters, and verified functionality.

## Table of Contents

1. [Sensor Wrappers](#sensor-wrappers)
2. [Mechanics Wrappers](#mechanics-wrappers)
3. [Control Wrappers](#control-wrappers)
4. [Logging Wrappers](#logging-wrappers)
5. [Observation Wrappers](#observation-wrappers)
6. [Environment Factory](#environment-factory)
7. [Testing Status](#testing-status)

---

## Sensor Wrappers

### ForceTorqueWrapper
**File**: `wrappers/sensors/force_torque_wrapper.py`
**Purpose**: Adds force-torque sensor functionality to any environment

#### Constructor
```python
ForceTorqueWrapper(env, use_tanh_scaling=False, tanh_scale=0.03)
```
- `env`: Base environment to wrap
- `use_tanh_scaling`: Whether to apply tanh scaling to force-torque readings
- `tanh_scale`: Scale factor for tanh transformation

#### Public Methods
- `get_force_torque_stats()` → dict
  - Returns: `{'current_force': tensor(N,3), 'current_torque': tensor(N,3), 'max_force': tensor(N), 'max_torque': tensor(N), 'avg_force': tensor(N), 'avg_torque': tensor(N)}`
  - Location: lines 199-210

- `has_force_torque_data()` → bool
  - Returns: True if force-torque data is available
  - Location: lines 213-214

- `get_force_torque_observation()` → tensor
  - Returns: Force-torque data formatted for observations (N, 6)
  - Applies tanh scaling if enabled
  - Location: lines 217-227

#### Environment Additions
- Adds `robot_force_torque` attribute: tensor(N, 6) - current force/torque readings
- Adds episode statistics: `ep_max_force`, `ep_max_torque`, `ep_sum_force`, `ep_sum_torque`

#### Verified During Testing
- [ ] Force sensor initialization works
- [ ] Statistics collection accurate
- [ ] Tanh scaling functionality
- [ ] Integration with RobotView/ArticulationView

---

## Mechanics Wrappers

### FragileObjectWrapper
**File**: `wrappers/mechanics/fragile_object_wrapper.py`
**Purpose**: Implements fragile object system where objects break under excessive force

#### Constructor
```python
FragileObjectWrapper(env, break_force, num_agents=1)
```
- `env`: Base environment to wrap
- `break_force`: Force threshold(s). Single float or list of floats per agent. Use -1 for unbreakable
- `num_agents`: Number of agents for static environment assignment

#### Public Methods
- `get_agent_assignment()` → dict
  - Returns: `{agent_id: [env_indices]}` mapping
  - Location: lines 137-143

- `get_break_forces()` → tensor
  - Returns: Break force thresholds for all environments (N,)
  - Location: lines 145-147

- `get_agent_break_force(agent_id)` → tensor
  - Returns: Break force threshold for specific agent's environments
  - Location: lines 149-156

- `is_fragile()` → bool
  - Returns: True if any objects are fragile
  - Location: lines 158-160

- `get_force_violations()` → tensor
  - Returns: Boolean mask of current force violations (N,)
  - Location: lines 162-168

#### Multi-Agent Support
- Static environment assignment: `envs_per_agent = num_envs // num_agents`
- Per-agent break force configuration
- Validation: `num_envs % num_agents == 0`

#### Verified During Testing
- [ ] Break force thresholds work correctly
- [ ] Multi-agent assignment functions
- [ ] Force violation detection
- [ ] Integration with ForceTorqueWrapper

---

### EfficientResetWrapper
**File**: `wrappers/mechanics/efficient_reset_wrapper.py`
**Purpose**: Efficient environment resetting using state caching and shuffling

#### Constructor
```python
EfficientResetWrapper(env)
```
- `env`: Base environment to wrap

#### Public Methods
- `has_cached_state()` → bool
  - Returns: True if initial state is cached
  - Location: lines 155-156

- `clear_cached_state()` → None
  - Clears cached state, forces full reset next time
  - Location: lines 158-160

- `get_reset_efficiency_stats()` → dict
  - Returns: `{'has_cached_state': bool, 'supports_efficient_reset': bool}`
  - Location: lines 162-167

#### Implementation Details
- Full reset: All environments reset → cache state
- Partial reset: Use state shuffling for individual environments
- State shuffling includes: root poses, velocities, joint positions
- Handles environment origin adjustments

#### Verified During Testing
- [ ] State caching works correctly
- [ ] Partial reset via shuffling
- [ ] Performance improvement measurable
- [ ] Scene state consistency maintained

---

## Control Wrappers

### HybridForcePositionWrapper
**File**: `wrappers/control/hybrid_force_position_wrapper.py`
**Purpose**: Hybrid force-position control with selection matrix switching

#### Constructor
```python
HybridForcePositionWrapper(env, ctrl_torque=False, reward_type="simp")
```
- `env`: Base environment to wrap
- `ctrl_torque`: Whether to control torques (6DOF) or just forces (3DOF)
- `reward_type`: Reward strategy - "simp", "dirs", "delta", "base", "pos_simp", "wrench_norm"

#### Action Space
- Original: 6 (position + rotation)
- Extended: `force_size + 6 + force_size`
  - Selection matrix: (3 or 6)
  - Position/rotation: (6)
  - Force/torque: (3 or 6)

#### Key Attributes
- `sel_matrix`: tensor(N, 6) - force/position selection per axis
- `force_action`: tensor(N, 6) - desired forces/torques
- `kp`: tensor(N, 6) - force control gains

#### Reward Strategies
- `"simp"`: Simple force activity reward
- `"dirs"`: Direction-specific force rewards
- `"delta"`: Penalize selection matrix changes
- `"base"`: No hybrid control rewards
- `"pos_simp"`: Position-focused force rewards
- `"wrench_norm"`: Low wrench magnitude reward

#### Verified During Testing
- [ ] Action space expansion correct
- [ ] Selection matrix functionality
- [ ] Reward strategies work
- [ ] Integration with factory_control_utils
- [ ] Force/position switching

---

### factory_control_utils.py
**File**: `wrappers/control/factory_control_utils.py`
**Purpose**: Control functions extracted from factory_control.py

#### Functions
- `compute_pose_task_wrench()` - Operational space pose control
- `compute_force_task_wrench()` - Force control calculations
- `compute_dof_torque_from_wrench()` - Convert wrench to joint torques
- `get_pose_error()` - Position and orientation error calculation
- `_apply_task_space_gains()` - Apply gains to task space errors

#### Verified During Testing
- [ ] All functions work independently
- [ ] Integration with hybrid wrapper
- [ ] Operational space control accuracy

---

## Logging Wrappers

### WandbLoggingWrapper
**File**: `wrappers/logging/wandb_logging_wrapper.py`
**Purpose**: Environment-agnostic Wandb logging with multi-agent support

#### Constructor
```python
WandbLoggingWrapper(env, wandb_config, num_agents=1, clip_eps=0.2)
```
- `env`: Base environment to wrap
- `wandb_config`: Wandb configuration dictionary
- `num_agents`: Number of agents for static assignment
- `clip_eps`: Clipping epsilon for PPO metrics

#### Public Methods
- `get_agent_assignment()` → dict
  - Returns: Agent to environment mapping
  - Location: lines 416-424

- `add_metric(tag, value)` → None
  - Add custom metric for logging
  - Location: lines 477-490

- `log_learning_metrics(**kwargs)` → None
  - Log RL training metrics
  - Location: lines 491-495

- `log_action_metrics(actions, global_step)` → None
  - Log action distribution metrics
  - Location: lines 496-503

- `publish_metrics()` → None
  - Publish all accumulated metrics to wandb
  - Location: lines 504-508

#### EpisodeTracker Class
Internal class managing episode metrics and statistics across environments.

#### Verified During Testing
- [ ] Multi-agent environment assignment
- [ ] Episode tracking and aggregation
- [ ] Wandb integration works
- [ ] Custom metrics logging
- [ ] Performance impact acceptable

---

### FactoryMetricsWrapper
**File**: `wrappers/logging/factory_metrics_wrapper.py`
**Purpose**: Factory-specific metrics tracking (success, engagement, smoothness)

#### Constructor
```python
FactoryMetricsWrapper(env, num_agents=1)
```
- `env`: Base environment to wrap
- `num_agents`: Number of agents for static assignment

#### Public Methods
- `get_agent_assignment()` → dict
  - Returns: Agent to environment mapping
  - Location: lines 244-251

- `get_success_stats()` → dict
  - Returns: `{'success_rate': float, 'avg_success_time': float, 'engagement_rate': float, 'avg_engagement_time': float, 'avg_engagement_length': float}`
  - Location: lines 253-261

- `get_smoothness_stats()` → dict
  - Returns: Smoothness metrics including SSV, force/torque statistics
  - Location: lines 263-278

- `get_agent_metrics(agent_id)` → dict
  - Returns: Metrics for specific agent
  - Location: lines 280-300

#### Tracked Metrics
- Episode success/engagement tracking
- Success and engagement timing
- Sum squared velocity (smoothness)
- Force/torque statistics (max, sum, avg)

#### Verified During Testing
- [ ] Success detection works
- [ ] Engagement tracking accurate
- [ ] Smoothness calculation correct
- [ ] Force/torque metrics integration
- [ ] Multi-agent metrics separation

---

## Observation Wrappers

### HistoryObservationWrapper
**File**: `wrappers/observations/history_observation_wrapper.py`
**Purpose**: Selective historical observation tracking for specified components

#### Constructor
```python
HistoryObservationWrapper(env, history_components=None, history_length=None, history_samples=None, calc_acceleration=False)
```
- `env`: Base environment to wrap
- `history_components`: List of observation component names to track (default: ["force_torque", "ee_linvel", "ee_angvel"])
- `history_length`: Length of history buffer (default: from config decimation)
- `history_samples`: Number of samples to keep from history (default: history_length)
- `calc_acceleration`: Whether to calculate acceleration for specified components

#### Public Methods
- `get_history_stats()` → dict
  - Returns: `{'history_components': list, 'history_length': int, 'num_samples': int, 'calc_acceleration': bool, 'buffer_count': int, 'acceleration_buffer_count': int}`
  - Location: lines 393-403

- `get_component_history(component)` → tensor
  - Returns: History for specific component or None
  - Location: lines 405-412

#### Features
- Configurable history components and sampling
- Acceleration calculation for velocity components
- Force derivative calculations (jerk, snap)
- Dimension updates for observation space
- Sample indices calculation with `torch.linspace()`

#### Verified During Testing
- [ ] History buffer management
- [ ] Selective component tracking
- [ ] Acceleration calculations
- [ ] Observation space updates
- [ ] Sample index generation

---

### ObservationManagerWrapper
**File**: `wrappers/observations/observation_manager_wrapper.py`
**Purpose**: Enforces standard {"policy": tensor, "critic": tensor} observation format

#### Constructor
```python
ObservationManagerWrapper(env, use_obs_noise=False)
```
- `env`: Base environment to wrap
- `use_obs_noise`: Whether to apply observation noise

#### Public Methods
- `get_observation_info()` → dict
  - Returns: Information about current observation format
  - Location: lines 206-227

- `get_observation_space_info()` → dict
  - Returns: Configured observation spaces information
  - Location: lines 229-242

- `validate_wrapper_stack()` → list
  - Returns: List of validation issues (empty if valid)
  - Location: lines 244-275

#### Features
- Standard observation format enforcement
- Observation component composition using obs_order/state_order
- Noise injection management
- Dynamic observation space validation
- Format conversion from various input types

#### Verified During Testing
- [ ] Standard format enforcement
- [ ] Observation composition works
- [ ] Noise injection functionality
- [ ] Validation system works
- [ ] Compatibility with other wrappers

---

## Environment Factory

### FactoryEnvironmentBuilder
**File**: `wrappers/factory.py`
**Purpose**: Builder pattern for creating environments with modular wrapper composition

#### Constructor & Builder Methods
```python
builder = FactoryEnvironmentBuilder()
builder.with_force_torque_sensor(**kwargs)
builder.with_fragile_objects(num_agents=1, **kwargs)
builder.with_efficient_reset(cache_size_ratio=0.1, **kwargs)
builder.with_hybrid_control(reward_strategy="simp", **kwargs)
builder.with_wandb_logging(project_name, num_agents=1, **kwargs)
builder.with_factory_metrics(num_agents=1, **kwargs)
builder.with_history_observations(history_components=None, **kwargs)
builder.with_observation_manager(use_obs_noise=False, **kwargs)
builder.with_config_override(**config_overrides)
env = builder.build(env_cfg, task_name="Factory-Task-Direct-v0")
```

#### Convenience Functions
- `create_factory_environment(env_cfg, preset=None, num_agents=1, **kwargs)`
- `create_multi_agent_environment(env_cfg, num_agents, project_name=None, **kwargs)`
- `validate_environment_config(env_cfg, num_agents=1)`
- `get_available_presets()`

#### Available Presets
- `"basic"`: Essential wrappers for basic functionality
- `"training"`: Optimized for training with noise and efficient resets
- `"research"`: Comprehensive setup with history and detailed metrics
- `"multi_agent"`: Multi-agent setup with static environment assignment
- `"control_research"`: Setup for control research with hybrid force-position control

#### Verified During Testing
- [ ] Builder pattern works correctly
- [ ] Preset configurations functional
- [ ] Multi-agent environment creation
- [ ] Configuration validation
- [ ] Isaac Lab integration

---

## Testing Status

### Test Coverage Progress

#### Unit Tests Completed
- [ ] ForceTorqueWrapper
- [ ] FragileObjectWrapper
- [ ] EfficientResetWrapper
- [ ] HybridForcePositionWrapper
- [ ] WandbLoggingWrapper
- [ ] FactoryMetricsWrapper
- [ ] HistoryObservationWrapper
- [ ] ObservationManagerWrapper
- [ ] FactoryEnvironmentBuilder

#### Integration Tests Completed
- [ ] Wrapper combination compatibility
- [ ] Multi-agent functionality
- [ ] Performance benchmarks
- [ ] Isaac Lab integration

#### Known Issues
- None discovered yet

#### Performance Benchmarks
- Original vs wrapped environment speed: TBD
- Memory usage comparison: TBD
- Reset efficiency improvement: TBD

---

## Usage Examples

### Basic Factory Environment
```python
from wrappers.factory import create_factory_environment

env = create_factory_environment(
    env_cfg=my_config,
    preset="basic",
    num_agents=1
)
```

### Multi-Agent Research Setup
```python
env = create_factory_environment(
    env_cfg=my_config,
    preset="research",
    num_agents=4,
    project_name="factory_research"
)
```

### Custom Wrapper Composition
```python
from wrappers.factory import FactoryEnvironmentBuilder

env = (FactoryEnvironmentBuilder()
       .with_force_torque_sensor()
       .with_fragile_objects(num_agents=2, break_force=[50.0, -1])
       .with_history_observations(history_components=["force_torque"])
       .with_observation_manager(use_obs_noise=True)
       .build(my_config))
```

### Hybrid Control Research
```python
env = create_factory_environment(
    env_cfg=my_config,
    preset="control_research",
    reward_strategy="delta",
    num_agents=1
)
```

---

**Last Updated**: Initial documentation created
**Testing Status**: Ready for unit testing phase
**Next Steps**: Implement comprehensive unit tests for each wrapper