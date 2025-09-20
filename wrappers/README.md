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
- `get_current_force_torque()` → dict
  - Returns: `{'current_force': tensor(N,3), 'current_torque': tensor(N,3)}` or empty dict if no data
  - Location: lines 233-240

- `has_force_torque_data()` → bool
  - Returns: True if force-torque data is available
  - Location: lines 242-244

- `get_force_torque_observation()` → tensor
  - Returns: Force-torque data formatted for observations (N, 6)
  - Applies tanh scaling if enabled, returns zeros if no data available
  - Location: lines 246-257

#### Environment Additions
- Adds `robot_force_torque` attribute: tensor(N, 6) - current force/torque readings
- Adds force-torque sensor interface via Isaac Sim RobotView
- Updates observation and state space configurations automatically
- Supports integration with component-based observation systems

#### Verified During Testing
- [x] Force sensor initialization works correctly
- [x] Isaac Sim RobotView integration functional
- [x] Tanh scaling functionality validated
- [x] Configuration updates work for observation spaces
- [x] Lazy initialization handles wrapper chains
- [x] Error handling and graceful fallbacks

---

## Mechanics Wrappers

### GripperCloseEnv
**File**: `wrappers/mechanics/close_gripper_action_wrapper.py`
**Purpose**: Action wrapper that forces gripper to closed position (-1.0) for secure grasping

#### Constructor
```python
GripperCloseEnv(env)
```
- `env`: Base environment to wrap. Must have action space with at least one dimension for gripper control

#### Public Methods
- `action(action)` → tensor
  - Transforms actions to force gripper to closed position
  - Modifies last component of action tensor to -1.0 in-place
  - Preserves all other action components and tensor properties
  - Location: lines 53-76

- `step(action)` → tuple
  - Executes environment step with gripper forced closed
  - Returns: (obs, rewards, terminated, truncated, info)
  - Location: lines 78-93

#### Features
- In-place action modification for efficiency
- Preserves tensor device, dtype, and shape
- Debug print statements show action transformation
- Works with any action tensor shape (num_envs, action_dim)

#### Verified During Testing
- [x] Action transformation works correctly
- [x] In-place modification preserves tensor properties
- [x] Works with different tensor shapes and edge cases
- [x] Integration with gymnasium wrapper system

---

### FragileObjectWrapper
**File**: `wrappers/mechanics/fragile_object_wrapper.py`
**Purpose**: Adds fragile object functionality where objects break under excessive force

#### Constructor
```python
FragileObjectWrapper(env, break_force, num_agents=1)
```
- `env`: Base environment to wrap (should have force-torque data available)
- `break_force`: Force threshold(s). Single float or list of floats per agent. Use -1 for unbreakable
- `num_agents`: Number of agents for static environment assignment

#### Public Methods
- `get_agent_assignment()` → dict
  - Returns: `{agent_id: [env_indices]}` mapping
  - Location: lines 179-188

- `get_break_forces()` → tensor
  - Returns: Break force thresholds for all environments (N,)
  - Location: lines 190-196

- `get_agent_break_force(agent_id)` → tensor
  - Returns: Break force threshold for specific agent's environments
  - Location: lines 198-212

- `is_fragile()` → bool
  - Returns: True if any objects can break (break_force < 2^20)
  - Location: lines 214-220

- `get_force_violations()` → tensor
  - Returns: Boolean mask of current force violations (N,)
  - Location: lines 222-236

#### Multi-Agent Support
- Static environment assignment: `envs_per_agent = num_envs // num_agents`
- Per-agent break force configuration with different thresholds
- Validation: `num_envs % num_agents == 0`
- Unbreakable objects: Use -1 to set very high threshold (2^23)

#### Implementation Details
- Monitors L2 norm of force vector (first 3 components of robot_force_torque)
- Automatic episode termination when force exceeds break threshold
- Lazy initialization to work with wrapper chains
- Integrates with original environment termination conditions

#### Verified During Testing
- [x] Break force thresholds work correctly
- [x] Multi-agent assignment functions properly
- [x] Force violation detection accurate
- [x] Integration with ForceTorqueWrapper
- [x] Lazy initialization handles wrapper chains

---

### EfficientResetWrapper
**File**: `wrappers/mechanics/efficient_reset_wrapper.py`
**Purpose**: Efficient environment resetting using state caching and shuffling to avoid expensive simulation

#### Constructor
```python
EfficientResetWrapper(env)
```
- `env`: Base environment to wrap

#### Public Methods
- `has_cached_state()` → bool
  - Returns: True if initial state is cached and available for efficient resets
  - Location: lines 194-196

- `clear_cached_state()` → None
  - Clears cached state, forces full reset on next reset call
  - Location: lines 198-200

- `get_reset_efficiency_stats()` → dict
  - Returns: `{'has_cached_state': bool, 'supports_efficient_reset': bool}`
  - Location: lines 202-207

#### Implementation Details
- **Full reset**: All environments reset simultaneously → cache initial state
- **Partial reset**: Individual environments reset using state shuffling from cache
- **State shuffling**: Root poses, velocities, joint positions copied from random source environments
- **Environment origins**: Automatic position adjustment for different environment origins
- **DirectRLEnv integration**: Finds lightweight DirectRLEnv._reset_idx instead of expensive factory reset

#### Advanced Features
- Method override pattern: Replaces environment's _reset_idx with wrapped version
- Lazy initialization: Works even if environment isn't fully initialized at wrapper creation
- Fallback mechanism: Uses original reset method if DirectRLEnv can't be found
- State validation: Checks for scene and articulation availability

#### Verified During Testing
- [x] State caching works correctly after full reset
- [x] Partial reset via state shuffling functions
- [x] DirectRLEnv method detection and fallback
- [x] Environment origin position adjustments
- [x] Lazy initialization with wrapper chains

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

### GenericWandbLoggingWrapper
**File**: `wrappers/logging/generic_wandb_wrapper.py`
**Purpose**: Environment-agnostic Wandb logging with configurable metrics

#### Constructor
```python
GenericWandbLoggingWrapper(env, logging_config, num_agents=1)
```
- `env`: Base environment to wrap
- `logging_config`: LoggingConfig object with metric configuration
- `num_agents`: Number of agents for static assignment

#### Public Methods
- `get_agent_assignment()` → dict
  - Returns: Agent to environment mapping

- `add_metric(tag, value)` → None
  - Add custom metric for logging

- `log_learning_metrics(**kwargs)` → None
  - Log RL training metrics

- `log_action_metrics(actions, global_step)` → None
  - Log action distribution metrics

- `publish_metrics()` → None
  - Publish all accumulated metrics to wandb

#### Features
- Configurable metric tracking via LoggingConfig
- Multi-agent support with separate Wandb runs
- Environment-agnostic design
- Preset configurations for common use cases

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

### ObservationManagerWrapper
**File**: `wrappers/observations/observation_manager_wrapper.py`
**Purpose**: Converts Isaac Lab {"policy": tensor, "critic": tensor} format to single tensor for SKRL compatibility

#### Constructor
```python
ObservationManagerWrapper(env, merge_strategy="concatenate")
```
- `env`: Base environment to wrap (should output Isaac Lab format observations)
- `merge_strategy`: How to merge policy/critic observations ("concatenate", "policy_only", "critic_only", "average")

#### Public Methods
- `get_observation_info()` → dict
  - Returns: Information about observation format (shape, dtype, device, stats, merge_strategy)
  - Location: lines 151-171

- `get_observation_space_info()` → dict
  - Returns: Environment configuration information (obs_order, state_order, observation_space, state_space)
  - Location: lines 173-186

- `validate_wrapper_stack()` → list
  - Returns: List of validation issues found (empty if valid)
  - Location: lines 188-215

#### Features
- Converts Isaac Lab dict observations to single tensor format
- Multiple merge strategies with automatic fallback handling
- Observation component composition using environment configuration
- Comprehensive validation and diagnostic information
- Lazy initialization for wrapper chain compatibility

#### Verified During Testing
- [x] Isaac Lab format conversion works correctly (39 tests, 100% pass)
- [x] All merge strategies function properly
- [x] Validation system detects configuration issues
- [x] Lazy initialization handles wrapper chains
- [x] Error handling and fallback mechanisms work

---

### HistoryObservationWrapper
**File**: `wrappers/observations/history_observation_wrapper.py`
**Purpose**: Selective historical observation tracking with explicit component configuration

#### Constructor
```python
HistoryObservationWrapper(env, history_components, history_length=None, history_samples=None)
```
- `env`: Base environment to wrap (must have component_dims and component_attr_map configuration)
- `history_components`: List of observation component names to track (required, cannot be None)
- `history_length`: Length of history buffer (default: from config decimation)
- `history_samples`: Number of samples to keep from history (default: history_length)

#### Public Methods
- `get_history_stats()` → dict
  - Returns: Statistics about history buffers (components, length, samples, buffer_count, etc.)
  - Location: lines 318-333

- `get_component_history(component)` → tensor
  - Returns: Clone of history buffer for specific component
  - Location: lines 335-340

#### Key Features
- Requires explicit component specification (no silent defaults)
- Uses environment configuration for component dimensions and attribute mapping
- Configurable history length and sampling with torch.linspace()
- Automatic observation space dimension updates
- Rolling buffer management with efficient tensor operations
- Support for relative position calculations

#### Environment Configuration Requirements
- `component_dims`: Dict mapping component names to their dimensions
- `component_attr_map`: Dict mapping component names to environment attributes
- `obs_order`/`state_order`: Lists defining observation component order

#### Verified During Testing
- [x] Explicit component requirement prevents silent failures (43 tests, 100% pass)
- [x] History buffer management with rolling updates
- [x] Environment configuration validation
- [x] Relative position calculations work
- [x] Observation space dimension updates

---

### ObservationNoiseWrapper
**File**: `wrappers/observations/observation_noise_wrapper.py`
**Purpose**: Group-based observation noise injection with semantic understanding and timing control

#### Constructor
```python
ObservationNoiseWrapper(env, noise_config: ObservationNoiseConfig)
```
- `env`: Base environment to wrap (must have component_dims and component_attr_map configuration)
- `noise_config`: ObservationNoiseConfig object defining noise groups and settings

#### Public Methods
- `get_noise_info()` → dict
  - Returns: Information about noise configuration and current state
  - Location: lines 343-363

#### Key Features
- **Group-based noise**: Applies different noise to semantic groups (fingertip_pos, joint_pos, etc.)
- **Timing control**: Configurable noise update timing (per step, per episode, per policy update)
- **Multi-noise types**: Gaussian, uniform, or no noise per group
- **Critic control**: Option to disable noise for critic observations
- **Clipping support**: Optional value clipping per group
- **Batch size handling**: Automatic noise tensor resizing for different batch sizes

#### Noise Configuration Classes
- `NoiseGroupConfig`: Configuration for individual observation groups
- `ObservationNoiseConfig`: Complete wrapper configuration with global settings
- Preset functions: `create_position_noise_config()`, `create_joint_noise_config()`, `create_minimal_noise_config()`

#### Environment Configuration Requirements
- `component_dims`: Dict mapping component names to dimensions
- `component_attr_map`: Dict mapping component names to environment attributes
- `obs_order`/`state_order`: Lists defining observation order for group mapping

#### Verified During Testing
- [x] Group-based noise application works correctly (50 tests, 100% pass)
- [x] Multiple timing strategies function properly
- [x] Noise types (gaussian, uniform, none) work as expected
- [x] Batch size mismatch handling resolved
- [x] Preset configurations create valid noise setups

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
builder.with_generic_logging(logging_config, num_agents=1, **kwargs)
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
- [x] ForceTorqueWrapper (28 tests, 100% pass)
- [x] GripperCloseEnv (16 tests, 100% pass)
- [x] FragileObjectWrapper (34 tests, 100% pass)
- [x] EfficientResetWrapper (26 tests, 100% pass)
- [ ] HybridForcePositionWrapper
- [ ] GenericWandbLoggingWrapper
- [ ] FactoryMetricsWrapper
- [x] ObservationManagerWrapper (39 tests, 100% pass)
- [x] HistoryObservationWrapper (43 tests, 100% pass)
- [x] ObservationNoiseWrapper (50 tests, 100% pass)
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